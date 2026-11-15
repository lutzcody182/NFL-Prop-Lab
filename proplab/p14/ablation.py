"""Phase 14 groups D and U (phase14/PREREGISTRATION.md): RB ranges, in the engine with actual actives.

    python -m proplab.p14.ablation decide    # walk-forward 2019-2024 -> phase14/ablation.json
    python -m proplab.p14.ablation confirm   # 2025, 2026 weeks 1-9, pooled
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from proplab.p3 import engine as E
from proplab.p3.ablation import Runner, base_configs, compare, with_group
from proplab.p9.roles import setup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase14" / "ablation.json"
SEASONS = list(range(2019, 2025))
LIVE_WEEKS = (1, 9)
MARKETS = ["rb_rush_att", "rb_rush_yds", "rb_rec", "rb_rec_yds", "rb_rush_rec_yds"]
KEYS = ["car", "tgt"]
SEED = 1414
GROUPS = {"dispersion_bins": ("rb_disp_bins", [True]), "efficiency_uncertainty": ("rb_eff_sd", [0.10, 0.20])}


def live_base() -> dict:
    """Current live engine: Phase 3a locked configs + adopted Phase 9 role signal on carries and targets."""
    g = json.loads((ROOT / "phase9" / "roles_ablation.json").read_text())["best"]
    return with_group(base_configs(), KEYS, "g_role", g)


class _Pooled:
    def __init__(self, runner):
        self.r = runner

    def pooled(self, market, cfgs, seasons, weeks=None):
        a = self.r.pooled(market, cfgs, [2025])
        b = self.r.pooled(market, cfgs, [2026], LIVE_WEEKS)
        return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


def fmt(r):
    return f"{r['change']:+.4%} CI [{r['ci95'][0]:+.4%}, {r['ci95'][1]:+.4%}] { {m: f'{x:+.2%}' for m, x in r['by_market'].items()} }"


def decide() -> None:
    runner = Runner(setup())
    rng = np.random.default_rng(SEED)
    base = live_base()
    out = {"seasons": SEASONS, "groups": {}}
    surv = {}
    for name, (field, values) in GROUPS.items():
        trials = []
        for v in values:
            res = compare(runner, MARKETS, base, with_group(base, KEYS, field, v), SEASONS, rng=rng)
            trials.append({"value": v, **res})
            print(f"{name} {field}={v}: {fmt(res)}", flush=True)
        best = min(trials, key=lambda t: t["change"])
        ok = best["ci95"][1] < 0
        out["groups"][name] = {"trials": trials, "best": best["value"], "survived": ok}
        if ok:
            surv[name] = best["value"]
    final = base
    for n, v in surv.items():
        final = with_group(final, KEYS, GROUPS[n][0], v)
    combined = {}
    if len(surv) > 1:
        for n in list(surv):
            without = with_group(final, KEYS, GROUPS[n][0], type(surv[n])(0))
            res = compare(runner, MARKETS, without, final, SEASONS, rng=rng)
            combined[n] = {"removal_hurts": res["change"] < 0, **res}
            print(f"combined: removing {n} -> {res['change']:+.4%}", flush=True)
            if res["change"] >= 0:
                surv.pop(n)
        final = base
        for n, v in surv.items():
            final = with_group(final, KEYS, GROUPS[n][0], v)
    out.update({"combined_check": combined, "kept": surv, "final_configs": {k: asdict(c) for k, c in final.items()}})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("KEPT:", surv)


def confirm() -> None:
    out = json.loads(OUT.read_text())
    if not out["kept"]:
        out["adopted"] = False
        OUT.write_text(json.dumps(out, indent=2, default=str))
        print("nothing survived development")
        return
    runner = Runner(setup())
    rng = np.random.default_rng(SEED + 1)
    base = live_base()
    final = {k: E.Config(**v) for k, v in out["final_configs"].items()}
    live = compare(runner, MARKETS, base, final, [2026], weeks=LIVE_WEEKS, rng=rng)
    second = compare(runner, MARKETS, base, final, [2025], rng=rng)
    pooled = compare(_Pooled(runner), MARKETS, base, final, ["pooled"], rng=rng)
    out["confirmation"] = {"live_2026_wk1_9": live, "second_look_2025": second, "pooled": pooled}
    out["adopted"] = live["change"] < 0 and second["change"] < 0 and pooled["ci95"][1] < 0
    # 2025 calibration and ranges, base vs final, for the record
    cal = {}
    for name, cfgs in (("base", base), ("final", final)):
        cal[name] = {m: {kk: runner.run(m, cfgs, 2025)[kk] for kk in ("ece", "ece_null95", "coverage80")} for m in MARKETS}
    out["calibration_2025"] = cal
    OUT.write_text(json.dumps(out, indent=2, default=str))
    for k, v in out["confirmation"].items():
        print(f"{k}: {fmt(v)}")
    for m in MARKETS:
        b, f = cal["base"][m], cal["final"][m]
        print(f"  {m:16s} ECE {b['ece']:.3f} -> {f['ece']:.3f} (limit {f['ece_null95']:.3f})  80% ranges {b['coverage80']} -> {f['coverage80']}")
    print("ADOPTED" if out["adopted"] else "NOT ADOPTED")


if __name__ == "__main__":
    {"decide": decide, "confirm": confirm}[sys.argv[1]]()
