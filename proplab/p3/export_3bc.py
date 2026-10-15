"""Collect 3b/3c results and this week's ratings and matchup tendencies for the dashboard.

    python -m proplab.p3.export_3bc     # -> phase3/dashboard_3bc.json
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from proplab import db
from proplab.p3 import engine as E
from proplab.p3 import matchup as MX
from proplab.ratings.team import CANON

ROOT = Path(__file__).resolve().parents[2]


def main(season: int = 2026) -> None:
    conn = db.connect()
    (week,) = conn.execute("""SELECT min(week) FROM games g WHERE season = %s AND game_type = 'REG'
                              AND NOT EXISTS (SELECT 1 FROM game_results r WHERE r.game_id = g.game_id)""",
                           (season,)).fetchone()
    tr = conn.execute("SELECT team, unit, off, def FROM team_ratings WHERE season = %s AND week = %s", (season, week)).fetchall()
    teams: dict = {}
    for team, unit, off, de in tr:
        teams.setdefault(team, {})[f"off_{unit}"] = off
        teams[team][f"def_{unit}"] = de

    D = E.load(conn)
    _, tend = MX.build(conn, D["u"], D["tg"], return_tendencies=True)
    games = conn.execute("""SELECT game_id, home_team, away_team, kickoff_utc FROM games
                            WHERE season = %s AND week = %s AND game_type = 'REG' ORDER BY kickoff_utc, game_id""",
                         (season, week)).fetchall()
    gids = [g[0] for g in games]
    pick = lambda df: df.filter(pl.col("game_id").is_in(gids))
    depth, gap, press = pick(tend["depth"]), pick(tend["gap"]), pick(tend["pressure"])
    matchups = []
    for gid, home, away, kick in games:
        sides = {}
        for team, opp in ((home, away), (away, home)):
            d = depth.filter((pl.col("game_id") == gid) & (pl.col("defense") == opp))
            g = gap.filter((pl.col("game_id") == gid) & (pl.col("defense") == opp))
            p = press.filter((pl.col("game_id") == gid) & (pl.col("defense") == opp))
            sides[team] = {  # how the OPPONENT's defense has played, faced by `team`'s offense
                "defense": opp,
                "depth": {r["depth"]: {"catch": r["r_cr"], "ypc": r["r_ypc"]} for r in d.iter_rows(named=True)},
                "gap": {r["gap"]: r["r"] for r in g.iter_rows(named=True)},
                "pressure": p["r_press"][0] if p.height else None,
            }
        matchups.append({"game_id": gid, "home": home, "away": away, "kickoff": kick.isoformat(), "sides": sides})

    pr = json.loads((ROOT / "phase3" / "player_ratings_current.json").read_text())
    players = [r for r in pr if r["qualified"]]
    out = {
        "season": season, "week": week,
        "team_ratings": [{"team": t, **v} for t, v in sorted(teams.items())],
        "team_results": json.loads((ROOT / "phase3" / "ratings_team_results.json").read_text()),
        "team_tuning": {u: v["chosen"] for u, v in json.loads((ROOT / "phase3" / "ratings_team_tuning.json").read_text()).items()},
        "player_results": json.loads((ROOT / "phase3" / "ratings_player_results.json").read_text()),
        "player_ratings": [{k: r[k] for k in ("name", "pos", "team", "post_mean", "lo80", "hi80", "N8", "rating",
                                             "rating_lo", "rating_hi")} for r in players],
        "ablation": json.loads((ROOT / "phase3" / "ablation.json").read_text()),
        "matchups": matchups,
    }
    out["ablation"].pop("final_configs", None)
    (ROOT / "phase3" / "dashboard_3bc.json").write_text(json.dumps(out, default=str))
    print(f"week {week}: {len(out['team_ratings'])} teams, {len(players)} rated players, {len(matchups)} games")


if __name__ == "__main__":
    main()
