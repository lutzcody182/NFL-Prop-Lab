"""Simulate next week's games with the Phase 4 simulator and export draws for the dashboard.

Same point-in-time rules as the other projection scripts: projected actives = each team's
actives in its last completed game; game lines = current consensus, stamped when fetched.

    python -m proplab.p4.export     # -> phase4/week_sims.json
"""
from __future__ import annotations

import base64
import datetime as dt
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from proplab.p4 import sim as S

ROOT = Path(__file__).resolve().parents[2]
N = 1000
MAX_RECEIVERS = 6


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.rint(a), -32768, 32767).astype("<i2").tobytes()).decode()


def main(season: int = 2026) -> None:
    D, feat, tf = S.setup()
    u, tg = D["u"], D["tg"]
    week = int(u.filter((pl.col("season") == season) & ~pl.col("played"))["week"].min())
    fetched = dt.datetime.now(dt.timezone.utc)
    sched = nfl.load_schedules([season]).filter(pl.col("week") == week)
    lines = pl.concat([
        sched.select("game_id", pl.col("home_team").alias("team"), pl.col("spread_line").alias("ts"), pl.col("total_line").alias("tl")),
        sched.select("game_id", pl.col("away_team").alias("team"), (-pl.col("spread_line")).alias("ts"), pl.col("total_line").alias("tl")),
    ])
    tg = (tg.join(lines, on=["game_id", "team"], how="left")
            .with_columns(pl.coalesce("team_spread", "ts").alias("team_spread"), pl.coalesce("total_line", "tl").alias("total_line"))
            .drop("ts", "tl"))
    played = u.filter(pl.col("played"))
    last = played.group_by("team").agg(pl.col("kickoff_utc").max().alias("last_kick"))
    acts = played.join(last, on="team").filter(pl.col("kickoff_utc") == pl.col("last_kick")).select("gsis_id", "name", "pos", "team")
    target = (pl.col("season") == season) & (pl.col("week") == week)
    games = u.filter(target).select("game_id", "kickoff_utc", "team", "opp", "is_home", "season", "week").unique()
    zeros = [pl.lit(0).alias(c) for c in ("targets", "receptions", "rec_tds", "carries", "rush_tds", "attempts",
                                          "completions", "pass_tds", "ints")]
    fz = [pl.lit(0.0).alias(c) for c in ("rec_yds", "rush_yds", "pass_yds")]
    future = acts.join(games, on="team").with_columns(*zeros, *fz, pl.lit(False).alias("played")).select(u.columns).cast(dict(u.schema))
    u2 = pl.concat([u.filter(~target), future]).sort("gsis_id", "kickoff_utc", "team")
    D2 = {"u": u2, "tg": tg, "pools": D["pools"]}
    feat2 = {h: S.E.features(u2, tg, h) for h in feat}
    tf2 = S.E.team_features(tg)

    inps = S.inputs_for(D, feat, tf, [season - 3, season - 2, season - 1])
    fit = S.fit_params([inps[season - 3], inps[season - 2], inps[season - 1]])
    cur = S.season_inputs(D2, feat2, tf2, season)
    sims = S.simulate_season(cur, fit, S.sigma(), n=N, weeks=(week, week), only_played=False)

    kick = dict(games.select("game_id", "kickoff_utc").unique().iter_rows())
    out_games = {}
    for (gid, team), v in sims.items():
        g = out_games.setdefault(gid, {"game_id": gid, "kickoff": kick[gid].isoformat(), "teams": {}})
        recv, sim = v["recv"], v["sim"]
        order = np.argsort(-recv["tgt_mu"].to_numpy())[:MAX_RECEIVERS]
        players = []
        if v["qb"] is not None:
            q = sim["qb"]
            players.append({"name": v["qb"]["name"], "gsis_id": v["qb"]["gsis_id"], "pos": "QB",
                            "stats": {"Pass yds": b64(q["yds"]), "Completions": b64(q["comp"]), "Pass TDs": b64(q["td"]),
                                      "Pass attempts": b64(q["att"])}})
        for i in order:
            players.append({"name": recv["name"][int(i)], "gsis_id": recv["gsis_id"][int(i)], "pos": recv["pos"][int(i)],
                            "stats": {"Rec yds": b64(sim["recv"]["yds"][:, i]), "Receptions": b64(sim["recv"]["rec"][:, i])}})
        g["teams"][team] = {"players": players, "team_attempts": b64(v["A"])}
    meta = {"season": season, "week": week, "draws": N, "sigma": S.sigma(), "fit": {k: v for k, v in vars(fit).items() if k != "scales"},
            "lines_fetched_utc": fetched.isoformat(timespec="minutes"),
            "result": json.loads((ROOT / "phase4" / "results.json").read_text())}
    (ROOT / "phase4" / "week_sims.json").write_text(json.dumps({"meta": meta, "games": list(out_games.values())}))
    print(f"week {week}: {len(out_games)} games, {sum(len(t['players']) for g in out_games.values() for t in g['teams'].values())} players")


if __name__ == "__main__":
    main()
