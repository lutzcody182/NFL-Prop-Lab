"""Phase 7C: score the new engine-derived markets once (same rules and code path as Phase 3a).

    python -m proplab.p7.markets     # -> phase7/markets_results.json
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from proplab.p3 import engine as E
from proplab.p3.run import N_TEST, SEED, boot_diff, load_chosen, setup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase7" / "markets_results.json"


def main() -> None:
    D, feat, tf = setup()
    chosen = load_chosen()
    variants = {
        "chosen": chosen,
        "B2": {vt: replace(c, model="B2", redistribute=0.0, opponent=0.0) for vt, c in chosen.items()},
        "B1": {vt: replace(c, model="B1", redistribute=0.0, opponent=0.0) for vt, c in chosen.items()},
    }
    results: dict = {"configs": {vt: c.label() for vt, c in chosen.items()}, "splits": {}}
    rng = np.random.default_rng(SEED + 7)
    for split, season, weeks in (("tuning_2024", 2024, None), ("test_2025", 2025, None), ("live_2026", 2026, (1, 5))):
        caches = {name: {} for name in variants}
        sres = {}
        for m in E.NEW_MARKETS:
            runs = {name: E.run_market(feat, tf, D["pools"], season, m, cfgs, N_TEST, SEED, weeks=weeks, cache=caches[name])
                    for name, cfgs in variants.items()}
            assert all((r["keys"] == runs["chosen"]["keys"]).all() for r in runs.values())
            games = runs["chosen"]["game_ids"]
            d, lo, hi = boot_diff(runs["chosen"]["crps_rows"], runs["B2"]["crps_rows"], games, rng)
            c = runs["chosen"]
            vt = E.NEW_MARKETS[m][1]
            kept = any(chosen[v].model != "B2" for v in (("car", "tgt") if vt == "both" else (vt,)))
            crit = {"beats_B2": bool(kept and hi < 0), "calibrated": bool(c["ece"] <= c["ece_null95"])}
            if "coverage80" in c:
                crit["coverage_ok"] = bool(abs(c["coverage80"] - 0.8) <= c["coverage_tol"])
            sres[m] = {"label": E.MARKET_LABELS[m], "rows": c["rows"], "games": int(len(np.unique(games))),
                       "models": {name: {k: v for k, v in r.items() if k not in ("crps_rows", "game_ids", "keys")} for name, r in runs.items()},
                       "vs_B2": {"mean_diff": d, "ci95": [lo, hi], "pct": d / runs["B2"]["crps"]},
                       "criteria": crit, "passed": all(crit.values())}
            print(f"{split} {m:16s} n={c['rows']:5d} chosen {c['crps']:.4f} B2 {runs['B2']['crps']:.4f} "
                  f"diff {d / runs['B2']['crps']:+.2%} [{lo / runs['B2']['crps']:+.2%},{hi / runs['B2']['crps']:+.2%}] "
                  f"ece {c['ece']:.3f}/{c['ece_null95']:.3f} cov {c.get('coverage80', float('nan')):.3f} {crit}", flush=True)
        results["splits"][split] = sres
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
