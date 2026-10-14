"""Phase 3a: tune on 2024 (with ablation), then score 2024 / 2025 (once) / 2026 wks 1-5.

    python -m proplab.p3.run tune       # -> phase3/tuning.json   (2024 only)
    python -m proplab.p3.run backtest   # -> phase3/results.json  (locked configs)
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from proplab import db
from proplab.p3 import engine as E

ROOT = Path(__file__).resolve().parents[2]
TUNING = ROOT / "phase3" / "tuning.json"
RESULTS = ROOT / "phase3" / "results.json"
TUNE_SEASON = 2024
VT_MARKETS = {
    "att": ["qb_pass_att", "qb_pass_yds", "qb_pass_tds", "qb_ints"],
    "car": ["qb_rush_yds", "rb_rush_att", "rb_rush_yds"],
    "tgt": ["rb_rec", "rb_rec_yds"],
}
HALF_LIVES = (4.0, 8.0, 16.0)
N_TUNE, N_TEST, SEED = 1000, 4000, 33


def setup():
    D = E.load(db.connect())
    feat = {h: E.features(D["u"], D["tg"], h) for h in HALF_LIVES}
    return D, feat, E.team_features(D["tg"])


def tune() -> None:
    D, feat, tf = setup()
    cache: dict = {}

    def objective(vt: str, cfg: E.Config, ref: dict) -> tuple[float, dict]:
        cfgs = {v: cfg for v in E.VT}
        crps = {m: E.run_market(feat, tf, D["pools"], TUNE_SEASON, m, cfgs, N_TUNE, SEED, cache=cache)["crps"]
                for m in VT_MARKETS[vt]}
        return float(np.mean([crps[m] / ref[m] for m in crps])), crps

    out: dict = {"chosen": {}, "grids": {}, "ablation": {}}
    for vt, markets in VT_MARKETS.items():
        ref_cfg = {v: E.Config(model="B1") for v in E.VT}
        ref = {m: E.run_market(feat, tf, D["pools"], TUNE_SEASON, m, ref_cfg, N_TUNE, SEED, cache=cache)["crps"]
               for m in markets}
        grid = [E.Config(model="M2", half_life=h, a_share=s, a_eff=e, redistribute=lam, opponent=g)
                for h, s, e, lam, g in itertools.product(HALF_LIVES, (10.0, 40.0), (10.0, 30.0), (0.0, 1.0), (0.0, 0.5, 1.0))]
        rows = []
        for cfg in grid:
            obj, crps = objective(vt, cfg, ref)
            rows.append((obj, cfg, crps))
        rows.sort(key=lambda r: r[0])
        best_obj, best, _ = rows[0]
        # Ablation: keep a component only if removing it makes 2024 worse.
        keep_lam = best.redistribute > 0 and objective(vt, replace(best, redistribute=0.0), ref)[0] > best_obj
        keep_gam = best.opponent > 0 and objective(vt, replace(best, opponent=0.0), ref)[0] > best_obj
        model = "M2" if keep_gam else ("M1" if keep_lam else "B2")
        chosen = replace(best, model=model, redistribute=best.redistribute if keep_lam else 0.0,
                         opponent=best.opponent if keep_gam else 0.0)
        out["chosen"][vt] = asdict(chosen)
        out["grids"][vt] = [{"config": c.label(), "objective": o, **{f"crps_{m}": v for m, v in cr.items()}}
                            for o, c, cr in rows[:10]]
        out["ablation"][vt] = {
            "best": {"config": best.label(), "objective": best_obj},
            "without_redistribution": objective(vt, replace(best, redistribute=0.0), ref)[0],
            "without_opponent": objective(vt, replace(best, opponent=0.0), ref)[0],
            "kept": {"redistribution": keep_lam, "opponent": keep_gam}, "chosen": chosen.label()}
        print(f"{vt}: best {best.label()} obj {best_obj:.4f} -> chosen {chosen.label()}", flush=True)
    TUNING.parent.mkdir(exist_ok=True)
    TUNING.write_text(json.dumps(out, indent=2))


def load_chosen() -> dict:
    raw = json.loads(TUNING.read_text())["chosen"]
    return {vt: E.Config(**cfg) for vt, cfg in raw.items()}


def boot_diff(a: np.ndarray, b: np.ndarray, games: np.ndarray, rng, n: int = 1000):
    uniq, inv = np.unique(games, return_inverse=True)
    d = a - b
    sums, cnts = np.bincount(inv, weights=d), np.bincount(inv)
    stats = [(sums[p].sum() / cnts[p].sum()) for p in (rng.integers(0, len(uniq), len(uniq)) for _ in range(n))]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def backtest() -> None:
    D, feat, tf = setup()
    chosen = load_chosen()
    variants = {
        "chosen": chosen,
        "B2": {vt: replace(c, model="B2", redistribute=0.0, opponent=0.0) for vt, c in chosen.items()},
        "B1": {vt: replace(c, model="B1", redistribute=0.0, opponent=0.0) for vt, c in chosen.items()},
    }
    results: dict = {"configs": {vt: c.label() for vt, c in chosen.items()}, "splits": {}}
    rng = np.random.default_rng(SEED)
    for split, season, weeks in (("tuning_2024", 2024, None), ("test_2025", 2025, None), ("live_2026", 2026, (1, 5))):
        caches = {name: {} for name in variants}
        sres = {}
        for m in E.MARKETS:
            runs = {name: E.run_market(feat, tf, D["pools"], season, m, cfgs, N_TEST, SEED, weeks=weeks, cache=caches[name])
                    for name, cfgs in variants.items()}
            assert all((r["keys"] == runs["chosen"]["keys"]).all() for r in runs.values())
            games = runs["chosen"]["game_ids"]
            d, lo, hi = boot_diff(runs["chosen"]["crps_rows"], runs["B2"]["crps_rows"], games, rng)
            c = runs["chosen"]
            vt = E.MARKETS[m][1]
            components_kept = (vt == "both" and any(chosen[v].model != "B2" for v in ("car", "tgt"))) or \
                              (vt != "both" and chosen[vt].model != "B2")
            crit = {"beats_B2": bool(components_kept and hi < 0), "calibrated": bool(c["ece"] <= c["ece_null95"])}
            if "coverage80" in c:
                crit["coverage_ok"] = bool(abs(c["coverage80"] - 0.8) <= c["coverage_tol"])
            sres[m] = {
                "label": E.MARKET_LABELS[m], "rows": c["rows"], "games": int(len(np.unique(games))),
                "models": {name: {k: v for k, v in r.items() if k not in ("crps_rows", "game_ids", "keys")}
                           for name, r in runs.items()},
                "vs_B2": {"mean_diff": d, "ci95": [lo, hi], "pct": d / runs["B2"]["crps"]},
                "criteria": crit, "passed": all(crit.values()),
            }
            mark = "PASS" if sres[m]["passed"] else "fail"
            print(f"{split} {m:12s} n={c['rows']:5d} chosen {c['crps']:.4f} B2 {runs['B2']['crps']:.4f} "
                  f"diff {d / runs['B2']['crps']:+.2%} [{lo / runs['B2']['crps']:+.2%},{hi / runs['B2']['crps']:+.2%}] "
                  f"ece {c['ece']:.3f}/{c['ece_null95']:.3f} cov {c.get('coverage80', float('nan')):.3f} {crit} {mark}",
                  flush=True)
        results["splits"][split] = sres
    RESULTS.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    {"tune": tune, "backtest": backtest}[sys.argv[1]]()
