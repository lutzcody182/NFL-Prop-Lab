"""Each defense's scheme profile going into the next game, plus the Phase 6 test results, for the dashboard.

    python -m proplab.p6.export     # -> phase6/dashboard_scheme.json
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from proplab import db
from proplab.p3 import engine as E
from proplab.p6 import scheme as SC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase6" / "dashboard_scheme.json"


def main(season: int = 2026) -> dict:
    conn = db.connect()
    (week,) = conn.execute("""SELECT min(week) FROM games g WHERE season = %s AND game_type = 'REG'
                              AND NOT EXISTS (SELECT 1 FROM game_results r WHERE r.game_id = g.game_id)""", (season,)).fetchone()
    D = E.load(conn)
    _, tend = SC.build(conn, D["u"], D["tg"], return_tendencies=True)
    # the row for each defense's next game holds everything known before it (prior-only)
    nxt = (D["tg"].filter((pl.col("season") == season) & ~pl.col("played")).select("game_id", pl.col("opp").alias("defense"), "kickoff_utc")
                 .sort("kickoff_utc").unique(subset="defense", keep="first"))
    t = nxt.join(tend, on=["game_id", "defense"], how="left")
    rows = [{"team": r["defense"], "blitz": r["h_def"], "blitz_lg": r["lg_bl"], "heavy_box": r["h_box"], "box_lg": r["lg_hv"],
             "pa_allowed": r["R_pa"], "nonpa_allowed": r["Q_np"], "dropbacks_weighted": r["P_d"]}
            for r in t.sort("defense").iter_rows(named=True)]
    abl = json.loads((ROOT / "phase6" / "ablation.json").read_text())
    payload = {"season": season, "week": week, "defenses": rows,
               "results": {"seasons": abl["seasons"], "kept": abl["kept"], "adopted": abl.get("adopted", False),
                           "groups": {k: {"best": v["best"], "survived": v["survived"], "markets": v["markets"],
                                          "trials": [{x: tr[x] for x in ("value", "change", "ci95", "by_market")} for tr in v["trials"]]}
                                      for k, v in abl["groups"].items()},
                           "confirmation": {k: v for k, v in abl.get("confirmation", {}).items() if k != "markets"}}}
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"week {week}: {len(rows)} defenses")
    return payload


if __name__ == "__main__":
    main()
