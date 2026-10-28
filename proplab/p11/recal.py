"""Phase 11 group R: recalibration on the CURRENT models (phase11/PREREGISTRATION.md).

    python -m proplab.p11.recal engine   # 14 Phase 8 markets refit with the Phase 9 role signal -> phase11/recal_engine.json
    python -m proplab.p11.recal wrte     # WR/TE receptions and yards on the final WR/TE model -> phase11/recal_wrte.json

Live maps go to phase11/recalibration.json (engine + kicker) and phase11/recal_wrte.json ("live_maps").
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.p1 import model as M
from proplab.p1.backtest import bootstrap_diff, load_configs
from proplab.p3 import engine as E
from proplab.p3.run import HALF_LIVES, load_chosen
from proplab.p8 import recal as R
from proplab.p9.roles import role_signal

ROOT = Path(__file__).resolve().parents[2]
ENGINE_RESULTS = ROOT / "phase11" / "recal_engine.json"
ENGINE_MAPS = ROOT / "phase11" / "recalibration.json"
WRTE = ROOT / "phase11" / "recal_wrte.json"
LIVE_WEEKS = (1, 7)


class RoleEngine(R.Engine):
    """Phase 8 engine runs, but with the adopted Phase 9 role signal, exactly as live projections use it."""

    def __init__(self):
        conn = db.connect()
        D = E.load(conn)
        D["u"] = D["u"].join(role_signal(conn, D["u"]), on=["gsis_id", "game_id"], how="left") \
                       .with_columns(pl.col("role_r").fill_null(1.0))
        self.D = D
        self.feat = {h: E.features(D["u"], D["tg"], h) for h in HALF_LIVES}
        self.tf = E.team_features(D["tg"])
        g = json.loads((ROOT / "phase9" / "roles_ablation.json").read_text())["best"]
        chosen = load_chosen()
        live = {vt: (replace(c, g_role=g) if vt in ("tgt", "car") else c) for vt, c in chosen.items()}
        self.variants = {"chosen": live,
                         "B2": {vt: replace(c, model="B2", redistribute=0.0, opponent=0.0) for vt, c in chosen.items()}}
        self.cache = {v: {} for v in self.variants}


def engine() -> None:
    R.main(eng=RoleEngine(), results_path=ENGINE_RESULTS, maps_path=ENGINE_MAPS, live_weeks=LIVE_WEEKS)


# ---------------------------------------------------------------- WR/TE

def wrte() -> None:
    from proplab.p10.ablation import setup
    S = setup()
    final_path = ROOT / "phase11" / "ablation.json"
    final = M.Config(**json.loads(final_path.read_text())["final"])
    b2 = load_configs()["B2"]
    feats = {}

    def run(cfg, season, weeks=None, n=4000):
        if cfg.half_life not in feats:
            feats[cfg.half_life] = M.assemble(S["u"], S["tg"], cfg.half_life)
        r = M.run_season(feats[cfg.half_life], S["tf"], S["cy"], season, cfg, n_sims=n, seed=R.SEED + season,
                         weeks=weeks, keep_draws=True)
        return {"rec": (r["rec_draws"], r["frame"]["receptions"].to_numpy().astype(float)),
                "yds": (r["yds_draws"], r["frame"]["rec_yds"].to_numpy().astype(float)), "games": r["game_ids"]}

    def score(d, y, lines, seed):
        return M.score_market(d, y, lines, np.random.default_rng(seed))

    out = {"model": final.label(), "baseline": b2.label(), "markets": {}}
    fits = {name: {s: run(cfg, s, n=R.N_FIT) for s in R.FIT_TEST} for name, cfg in (("model", final), ("B2", b2))}
    test = {name: run(cfg, 2025) for name, cfg in (("model", final), ("B2", b2))}
    live_fit = [run(final, s, n=R.N_FIT) for s in R.FIT_LIVE]
    live = run(final, 2026, weeks=LIVE_WEEKS)
    live_maps = {}
    for mk, key, lines in (("receptions", "rec", M.REC_LINES), ("yards", "yds", M.YDS_LINES)):
        maps = {name: R.fit_map("x", [fits[name][s][key] for s in R.FIT_TEST], np.random.default_rng(1)) for name in fits}
        ev = {}
        for name in ("model", "B2"):
            d, y = test[name][key]
            ev[name] = {"raw": score(d, y, lines, 2), "recal": score(R.apply_quantile_map(d, maps[name]["q"]), y, lines, 2)}
        verdict = {}
        for version in ("raw", "recal"):
            c, b = ev["model"][version], ev["B2"][version]
            diff, lo, hi = bootstrap_diff(c["crps_rows"] - b["crps_rows"], test["model"]["games"], np.random.default_rng(R.SEED))
            crit = {"beats_baseline": bool(hi < 0), "ece_ok": c["ece"] <= 0.02, "coverage_ok": 0.77 <= c["coverage80"] <= 0.83}
            verdict[version] = {"model": R.strip(c), "baseline": R.strip(b),
                                "vs_baseline": {"pct": diff / b["crps"], "ci95": [lo / b["crps"], hi / b["crps"]]},
                                "criteria": crit, "passed": all(crit.values())}
        raw_c, rec_c = ev["model"]["raw"], ev["model"]["recal"]
        adopted = bool(rec_c["ece"] < raw_c["ece"] and rec_c["crps"] <= raw_c["crps"])
        lm = R.fit_map("x", [p[key] for p in live_fit], np.random.default_rng(1))
        d, y = live[key]
        out["markets"][mk] = {"test_2025": verdict, "adopted": adopted, "maps_2022_2024": maps,
                              f"live_2026_wk{LIVE_WEEKS[0]}_{LIVE_WEEKS[1]}": {"raw": R.strip(score(d, y, lines, 3)),
                                                                              "recal": R.strip(score(R.apply_quantile_map(d, lm["q"]), y, lines, 3))}}
        live_maps[mk] = {**lm, "adopted": adopted}
        v = verdict["recal"]
        print(f"{mk:10s} raw {'PASS' if verdict['raw']['passed'] else 'fail'} -> recal {'PASS' if v['passed'] else 'fail'} | "
              f"ece {raw_c['ece']:.4f}->{rec_c['ece']:.4f} cov {raw_c['coverage80']:.3f}->{rec_c['coverage80']:.3f} "
              f"crps {raw_c['crps']:.4f}->{rec_c['crps']:.4f} vs B2 {v['vs_baseline']['pct']:+.2%} "
              f"[{v['vs_baseline']['ci95'][0]:+.2%},{v['vs_baseline']['ci95'][1]:+.2%}] adopt={adopted}", flush=True)
    out["live_maps"] = {"fit_seasons": R.FIT_LIVE, "maps": live_maps}
    WRTE.write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    {"engine": engine, "wrte": wrte}[sys.argv[1]]()
