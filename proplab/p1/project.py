"""Forward projections for the next slate with the locked M1 configuration.

Point-in-time honesty for a week that hasn't been played:
- Projected actives = each team's actives in its most recent completed game.
  This week's injuries and inactives are NOT known yet and are not used.
- Game lines = the current consensus spread/total from the nflverse schedule,
  stamped with the time they were fetched (not closing lines).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from proplab import db
from proplab.p1 import backtest, data, model as M

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase1" / "projections.json"
REC_SHOW = [3.5, 4.5, 5.5]
YDS_SHOW = [39.5, 49.5, 59.5, 69.5]


def next_week(u: pl.DataFrame, season: int) -> int:
    pending = u.filter((pl.col("season") == season) & ~pl.col("played"))
    return int(pending["week"].min())


def main(season: int = 2026, n_sims: int = 4000) -> None:
    conn = db.connect()
    u, tg, cy = data.load_universe(conn), data.load_team_games(conn), data.load_catch_yards(conn)
    cfg = backtest.load_configs()["M1"]
    week = next_week(u, season)

    # current lines for the week, fetched now
    fetched = dt.datetime.now(dt.timezone.utc)
    sched = nfl.load_schedules([season]).filter(pl.col("week") == week).select(
        "game_id", "home_team", "away_team", "spread_line", "total_line", "gameday", "gametime")
    lines = pl.concat([
        sched.select("game_id", pl.col("home_team").alias("team"), pl.col("spread_line").alias("ts"), "total_line"),
        sched.select("game_id", pl.col("away_team").alias("team"), (-pl.col("spread_line")).alias("ts"), "total_line"),
    ])
    tg = (tg.join(lines, on=["game_id", "team"], how="left")
            .with_columns(pl.coalesce("team_spread", "ts").alias("team_spread"),
                          pl.coalesce(pl.col("total_line"), pl.col("total_line_right")).alias("total_line"))
            .drop("ts", "total_line_right"))

    # projected actives: each team's actives in its most recent completed game
    played = u.filter(pl.col("played"))
    last_game = played.group_by("team").agg(pl.col("kickoff_utc").max().alias("last_kick"))
    last_actives = (played.join(last_game, on="team")
                    .filter(pl.col("kickoff_utc") == pl.col("last_kick"))
                    .select("gsis_id", "name", "pos", "team"))
    games = u.filter((pl.col("season") == season) & (pl.col("week") == week)) \
             .select("game_id", "kickoff_utc", "team", "opp", "is_home").unique()
    future = (last_actives.join(games, on="team")
              .with_columns(pl.lit(season).alias("season"), pl.lit(week).alias("week"),
                            pl.lit(0).alias("targets"), pl.lit(0).alias("receptions"),
                            pl.lit(0.0).alias("rec_yds"), pl.lit(False).alias("played"))
              .select(u.columns)
              .cast(dict(u.schema)))
    u2 = pl.concat([u.filter(~((pl.col("season") == season) & (pl.col("week") == week))), future]) \
           .sort("gsis_id", "kickoff_utc", "team")

    feat = M.assemble(u2, tg, cfg.half_life)
    tf = M.team_features(tg)
    params, k = M.prepare_season(feat, tf, cy, season, cfg)
    rows = (M.eligible(params.filter((pl.col("season") == season) & (pl.col("week") == week)))
            .sort("kickoff_utc", "team", "gsis_id"))
    rec, yds = M.simulate(rows, k, cfg.model, n_sims, seed=606)

    out = []
    for i, r in enumerate(rows.iter_rows(named=True)):
        rd, yd = rec[i], yds[i]
        out.append({
            "player": r["name"], "gsis_id": r["gsis_id"], "pos": r["pos"], "team": r["team"], "opp": r["opp"],
            "home": r["is_home"], "kickoff": r["kickoff_utc"].isoformat(), "game_id": r["game_id"],
            "rec": {"mean": float(rd.mean()), "median": float(np.median(rd)),
                    "p10": float(np.quantile(rd, 0.1)), "p90": float(np.quantile(rd, 0.9)),
                    "over": {str(L): float((rd > L).mean()) for L in REC_SHOW}},
            "yds": {"mean": float(yd.mean()), "median": float(np.median(yd)),
                    "p10": float(np.quantile(yd, 0.1)), "p90": float(np.quantile(yd, 0.9)),
                    "over": {str(L): float((yd > L).mean()) for L in YDS_SHOW}},
            "drivers": {"exp_targets": float(r["mu_t"]), "team_targets": float(r["team_hat"]),
                        "share": float(r["mu_t"] / r["team_hat"]), "catch_rate": float(r["p_catch"]),
                        "yds_per_catch": float(r["ypr"]), "team_spread": r["team_spread"],
                        "total": r["total_line"], "games_history": int(r["n_prev"])},
        })
    meta = {"season": season, "week": week, "model": cfg.label(), "players": len(out),
            "lines_fetched_utc": fetched.isoformat(timespec="minutes"),
            "rec_lines": REC_SHOW, "yds_lines": YDS_SHOW,
            "assumption": "Actives = each team's actives in its last completed game; "
                          "this week's injury reports and inactives are not yet known."}
    OUT.write_text(json.dumps({"meta": meta, "players": out}, indent=2))
    print(json.dumps(meta, indent=2))
    top = sorted(out, key=lambda p: -p["yds"]["mean"])[:8]
    for p in top:
        print(f"  {p['player']:24s} {p['pos']} {p['team']} v {p['opp']}: rec {p['rec']['mean']:.1f} "
              f"[{p['rec']['p10']:.0f}-{p['rec']['p90']:.0f}]  yds {p['yds']['mean']:.0f} "
              f"[{p['yds']['p10']:.0f}-{p['yds']['p90']:.0f}]  share {p['drivers']['share']:.0%}")


if __name__ == "__main__":
    main()
