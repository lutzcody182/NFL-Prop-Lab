"""Tune every model on 2024 only, then run the M1 ablation. Writes phase1/tuning.json.

Objective: mean of (CRPS / B1-default CRPS) across receptions and yards, so both
markets count equally regardless of scale. 1,000 draws, fixed seed (common random
numbers across configs).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

from proplab import db
from proplab.p1 import data, model as M

TUNE_SEASON = 2024
OUT = Path(__file__).resolve().parents[2] / "phase1" / "tuning.json"


def main() -> None:
    conn = db.connect()
    u, tg, cy = data.load_universe(conn), data.load_team_games(conn), data.load_catch_yards(conn)
    tf = M.team_features(tg)
    feats = {h: M.assemble(u, tg, h) for h in (4.0, 8.0, 16.0)}

    def run(cfg: M.Config) -> dict:
        r = M.run_season(feats[cfg.half_life], tf, cy, TUNE_SEASON, cfg, n_sims=1000, seed=11)
        return {"config": cfg.label(), "rec": r["receptions"]["crps"], "yds": r["yards"]["crps"],
                "rec_ece": r["receptions"]["ece"], "yds_ece": r["yards"]["ece"],
                "rec_cov": r["receptions"]["coverage80"], "yds_cov": r["yards"]["coverage80"],
                "rec_bias": r["receptions"]["mean_pred"] - r["receptions"]["mean_actual"],
                "yds_bias": r["yards"]["mean_pred"] - r["yards"]["mean_actual"], "cfg": cfg}

    ref = run(M.Config(model="B1"))
    obj = lambda r: 0.5 * (r["rec"] / ref["rec"] + r["yds"] / ref["yds"])

    grids = {
        "B1": [M.Config(model="B1", half_life=h, a_tpg=g, a_catch=c, a_ypr=y)
               for h, g, c, y in itertools.product((4.0, 8.0, 16.0), (2.0, 4.0, 8.0), (10.0, 30.0), (10.0, 30.0))],
        "B2": [M.Config(model="B2", half_life=h, a_share=s, a_catch=c, a_ypr=y)
               for h, s, c, y in itertools.product((4.0, 8.0, 16.0), (10.0, 20.0, 40.0, 80.0), (10.0, 30.0), (10.0, 30.0))],
    }
    results, best = {}, {}
    for name, grid in grids.items():
        rows = [run(c) for c in grid]
        rows.sort(key=obj)
        best[name] = rows[0]["cfg"]
        results[name] = [{k: v for k, v in r.items() if k != "cfg"} | {"objective": obj(r)} for r in rows[:10]]
        print(name, "best", rows[0]["config"], round(obj(rows[0]), 4), flush=True)

    # M1 starts from B2's tuned shrinkage, then tunes its two components
    base = best["B2"]
    m1_rows = [run(replace(base, model="M1", redistribute=l, opponent=g))
               for l, g in itertools.product((0.0, 0.5, 1.0), (0.0, 0.5, 1.0))]
    m1_rows.sort(key=obj)
    best["M1"] = m1_rows[0]["cfg"]
    results["M1_grid"] = [{k: v for k, v in r.items() if k != "cfg"} | {"objective": obj(r)} for r in m1_rows]
    print("M1 best", m1_rows[0]["config"], round(obj(m1_rows[0]), 4), flush=True)

    # Ablation from the chosen M1: remove each component on its own
    b = best["M1"]
    ablation = {
        "M1 (chosen)": run(b),
        "without redistribution": run(replace(b, redistribute=0.0)),
        "without opponent adjustment": run(replace(b, opponent=0.0)),
        "B2 (neither)": run(replace(b, model="B2")),
    }
    results["ablation"] = {k: {kk: vv for kk, vv in v.items() if kk != "cfg"} | {"objective": obj(v)}
                           for k, v in ablation.items()}
    results["chosen"] = {k: vars(v) for k, v in best.items()}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    for k, v in results["ablation"].items():
        print(f"  {k:30s} rec {v['rec']:.4f}  yds {v['yds']:.3f}  obj {v['objective']:.4f}  "
              f"bias rec {v['rec_bias']:+.3f} yds {v['yds_bias']:+.2f}  ece {v['rec_ece']:.3f}/{v['yds_ece']:.3f}  "
              f"cov {v['rec_cov']:.3f}/{v['yds_cov']:.3f}")


if __name__ == "__main__":
    main()
