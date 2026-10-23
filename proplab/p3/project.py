"""Next-week projections for the Phase 3a markets with the locked configurations.

Same honesty rules as proplab/p1/project.py: projected actives are each team's actives
in its last completed game; game lines are the current consensus spread/total,
stamped with when they were fetched. Writes phase3/projections.json.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from proplab import db
from proplab.p3 import engine as E
from proplab.p3.run import load_chosen
from proplab.p8.recal import MAPS, apply_map

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase3" / "projections.json"
N_SIMS = 4000
RECAL = json.loads(MAPS.read_text())["maps"] if MAPS.exists() else {}
QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]   # 5th..95th percentiles, stored for live scoring
GRID = [(i + 0.5) / 100 for i in range(100)]              # 100-point grid for "your own line" on the dashboard


def grid(d: np.ndarray) -> list:
    return [round(float(x), 1) for x in np.quantile(d, GRID)]
SHOW_LINES = {
    "qb_pass_yds": [224.5, 249.5, 274.5], "qb_pass_att": [29.5, 32.5, 35.5], "qb_pass_tds": [0.5, 1.5, 2.5],
    "qb_ints": [0.5], "qb_rush_yds": [9.5, 19.5, 29.5], "rb_rush_yds": [39.5, 59.5, 79.5],
    "rb_rush_att": [9.5, 14.5, 19.5], "rb_rec": [1.5, 2.5, 3.5], "rb_rec_yds": [9.5, 19.5, 29.5],
    "anytime_td": [0.5],
    "rb_rush_rec_yds": [49.5, 69.5, 89.5], "qb_completions": [17.5, 20.5, 23.5], "rec_td": [0.5],
}


def main(season: int = 2026) -> None:
    conn = db.connect()
    D = E.load(conn)
    u, tg = D["u"], D["tg"]
    chosen = load_chosen()
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
    games = u.filter((pl.col("season") == season) & (pl.col("week") == week)).select(
        "game_id", "kickoff_utc", "team", "opp", "is_home", "season", "week").unique()
    zeros = [pl.lit(0).alias(c) for c in ("targets", "receptions", "rec_tds", "carries", "rush_tds", "attempts",
                                          "completions", "pass_tds", "ints")]
    fzeros = [pl.lit(0.0).alias(c) for c in ("rec_yds", "rush_yds", "pass_yds")]
    future = (acts.join(games, on="team").with_columns(*zeros, *fzeros, pl.lit(False).alias("played"))
                  .select(u.columns).cast(dict(u.schema)))
    target = (pl.col("season") == season) & (pl.col("week") == week)
    u2 = pl.concat([u.filter(~target), future]).sort("gsis_id", "kickoff_utc", "team")
    # Phase 9A: snap-share role signal, adopted after its pre-registered confirmation
    roles_path = ROOT / "phase9" / "roles_ablation.json"
    if roles_path.exists() and json.loads(roles_path.read_text()).get("adopted"):
        from dataclasses import replace as _replace
        from proplab.p9.roles import role_signal
        g_role = json.loads(roles_path.read_text())["best"]
        u2 = u2.join(role_signal(conn, u2), on=["gsis_id", "game_id"], how="left").with_columns(pl.col("role_r").fill_null(1.0))
        chosen = {vt: (_replace(c, g_role=g_role) if vt in ("tgt", "car") else c) for vt, c in chosen.items()}

    tf = E.team_features(tg)
    feats = {c.half_life: E.features(u2, tg, c.half_life) for c in chosen.values()}
    preps = {vt: E.prepare(feats[c.half_life], tf, D["pools"], season, c) for vt, c in chosen.items()}

    out: dict = {"meta": {"season": season, "week": week, "lines_fetched_utc": fetched.isoformat(timespec="minutes"),
                          "configs": {vt: c.label() for vt, c in chosen.items()},
                          "assumption": "Actives = each team's actives in its last completed game; "
                                        "this week's injury reports and inactives are not yet known."},
                 "markets": {}}
    rng = np.random.default_rng(606)
    for m, (pos, vt, qty, _) in {**E.MARKETS, **E.NEW_MARKETS}.items():
        base_vt = "car" if vt == "both" else vt
        params, k = preps[base_vt]
        rows = params.filter(target & E.market_eligible(m)).sort("kickoff_utc", "team", "gsis_id")
        if vt == "both":
            tp, tk = preps["tgt"]
            tcols = [c for c in tp.columns if c.startswith("tgt_")]
            rows = rows.drop(tcols).join(tp.select("game_id", "gsis_id", *tcols), on=["game_id", "gsis_id"]) \
                       .sort("kickoff_utc", "team", "gsis_id")
            pick = lambda d: {kk: v for kk, v in d.items() if kk[0] == "tgt"}
            k = E.Constants(prior=k.prior, pools={**k.pools, **tk.pools}, dispersion={**k.dispersion, **pick(tk.dispersion)},
                            exit_rate={**k.exit_rate, **pick(tk.exit_rate)})
        draws = E.market_draws(m, rows, k, N_SIMS, rng)
        if RECAL.get(m, {}).get("adopted") and draws.shape[0]:
            draws = apply_map(m, draws, RECAL[m])   # Phase 8 recalibration (adopted markets only)
        players = []
        for i, r in enumerate(rows.iter_rows(named=True)):
            d = draws[i]
            players.append({
                "player": r["name"], "gsis_id": r["gsis_id"], "pos": r["pos"], "team": r["team"], "opp": r["opp"],
                "home": r["is_home"], "kickoff": r["kickoff_utc"].isoformat(), "game_id": r["game_id"],
                "mean": float(d.mean()), "p10": float(np.quantile(d, 0.1)), "p90": float(np.quantile(d, 0.9)),
                "q": [float(x) for x in np.quantile(d, QUANTILES)], "g": grid(d),
                "over": {str(L): float((d > L).mean()) for L in SHOW_LINES[m]},
                "volume": float(r[f"{base_vt}_mu"]),
            })
        out["markets"][m] = {"label": E.MARKET_LABELS[m], "lines": SHOW_LINES[m], "players": players}
        print(f"{m:12s} {len(players):3d} players", flush=True)
    # kicker points (Phase 7 model K), each team's kicker from its last game
    from proplab.p7 import kicker as KK
    klines = lines.rename({"ts": "team_spread", "tl": "total_line"})
    kp = KK.project(conn, season, week, klines)
    out["markets"]["k_points"] = {"label": "Kicker points", "lines": KK.LINES, "players": kp}
    print(f"k_points     {len(kp):3d} players", flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
