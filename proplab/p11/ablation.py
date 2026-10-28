"""Phase 11 groups D and P inside the live WR/TE model (phase11/PREREGISTRATION.md).

    python -m proplab.p11.ablation decide    # walk-forward 2019-2024 -> phase11/ablation.json
    python -m proplab.p11.ablation confirm   # 2025, 2026 weeks 1-7, pooled
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from proplab.p1 import model as M
from proplab.p10.ablation import Runner, compare, setup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase11" / "ablation.json"
BASE = ROOT / "phase10" / "ablation.json"
SEASONS = list(range(2019, 2025))
LIVE_WEEKS = (1, 7)
SEED = 1111
GROUPS = {"dispersion_bins": ("disp_bins", [True]), "position_redistribution": ("pos_redist", [0.5, 1.0])}


def base_config() -> M.Config:
    return M.Config(**json.loads(BASE.read_text())["final"])


def decide() -> None:
    runner = Runner(setup())
    rng = np.random.default_rng(SEED)
    base = base_config()
    sets = [(s, None) for s in SEASONS]
    out = {"seasons": SEASONS, "base": asdict(base), "groups": {}}
    surv = {}
    for name, (field, values) in GROUPS.items():
        trials = []
        for v in values:
            res = compare(runner, base, replace(base, **{field: v}), sets, rng)
            trials.append({"value": v, **res})
            print(f"{name:24s} {field}={v}: {res['change']:+.4%} CI [{res['ci95'][0]:+.4%}, {res['ci95'][1]:+.4%}] {res['by_market']}", flush=True)
        best = min(trials, key=lambda t: t["change"])
        ok = best["ci95"][1] < 0
        out["groups"][name] = {"trials": trials, "best": best["value"], "survived": ok}
        if ok:
            surv[name] = best["value"]
    final = replace(base, **{GROUPS[n][0]: v for n, v in surv.items()})
    combined = {}
    if len(surv) > 1:
        for n in list(surv):
            without = replace(final, **{GROUPS[n][0]: type(surv[n])(0)})
            res = compare(runner, without, final, sets, rng)
            combined[n] = {"removal_hurts": res["change"] < 0, **res}
            print(f"combined: removing {n} -> {res['change']:+.4%}", flush=True)
            if res["change"] >= 0:
                surv.pop(n)
        final = replace(base, **{GROUPS[n][0]: v for n, v in surv.items()})
    out.update({"combined_check": combined, "kept": surv, "final": asdict(final)})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("KEPT:", surv)


def confirm() -> None:
    out = json.loads(OUT.read_text())
    if not out["kept"]:
        out["adopted"] = False
        OUT.write_text(json.dumps(out, indent=2))
        print("nothing survived development")
        return
    runner = Runner(setup())
    rng = np.random.default_rng(SEED + 1)
    base, final = M.Config(**out["base"]), M.Config(**out["final"])
    live = compare(runner, base, final, [(2026, LIVE_WEEKS)], rng)
    second = compare(runner, base, final, [(2025, None)], rng)
    pooled = compare(runner, base, final, [(2025, None), (2026, LIVE_WEEKS)], rng)
    adopted = live["change"] < 0 and second["change"] < 0 and pooled["ci95"][1] < 0
    cov = {}
    for cfg_name, cfg in (("base", base), ("final", final)):
        r = runner.run(cfg, 2025)["full"]
        cov[cfg_name] = {m: {"coverage80": r[m]["coverage80"], "ece": r[m]["ece"]} for m in ("receptions", "yards")}
    out.update({"confirmation": {"live_2026_wk1_7": live, "second_look_2025": second, "pooled": pooled},
                "adopted": adopted, "calibration_2025": cov})
    OUT.write_text(json.dumps(out, indent=2, default=str))
    for k, v in out["confirmation"].items():
        print(f"{k}: {v['change']:+.4%} CI [{v['ci95'][0]:+.4%}, {v['ci95'][1]:+.4%}] {v['by_market']}")
    print("2025 calibration:", cov)
    print("ADOPTED" if adopted else "NOT ADOPTED")


if __name__ == "__main__":
    {"decide": decide, "confirm": confirm}[sys.argv[1]]()
