"""Phase 6: do the scheme feature groups survive ablation? (see phase6/PREREGISTRATION.md)

    python -m proplab.p6.ablation decide    # walk-forward 2023-2024 -> phase6/ablation.json
    python -m proplab.p6.ablation confirm   # survivors vs base on 2026 wks 1-5, 2025, and pooled
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from proplab import db
from proplab.p3 import engine as E
from proplab.p3.ablation import BOOT, Runner, base_configs, compare, with_group
from proplab.p6 import scheme as SC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase6" / "ablation.json"
SEASONS = [2023, 2024]
SEED = 66

GROUPS = {
    # name: (markets, config keys it changes, field, values to try)
    "blitz": (["qb_pass_yds", "qb_pass_tds", "qb_ints"], ["att"], "g_blitz", [0.5, 1.0]),
    "box": (["rb_rush_yds"], ["car"], "g_box", [0.5, 1.0]),
    "play_action": (["qb_pass_yds"], ["att"], "g_pa", [0.5, 1.0]),
}
OFF = {"g_blitz": 0.0, "g_box": 0.0, "g_pa": 0.0}


def setup():
    conn = db.connect()
    D = E.load(conn)
    sc = SC.build(conn, D["u"], D["tg"])
    return {"u": D["u"].join(sc, on=["gsis_id", "game_id"], how="left"), "tg": D["tg"], "pools": D["pools"]}


def decide() -> None:
    runner = Runner(setup())
    rng = np.random.default_rng(SEED)
    base = base_configs()
    out = {"groups": {}, "seasons": SEASONS, "bootstrap": BOOT}
    survivors = {}
    for name, (markets, keys, field, values) in GROUPS.items():
        trials = []
        for v in values:
            res = compare(runner, markets, base, with_group(base, keys, field, v), SEASONS, rng=rng)
            trials.append({"value": v, **res})
            print(f"{name:12s} {field}={v}: change {res['change']:+.4%} CI [{res['ci95'][0]:+.4%}, {res['ci95'][1]:+.4%}] "
                  f"{ {m: f'{x:+.3%}' for m, x in res['by_market'].items()} }", flush=True)
        best = min(trials, key=lambda t: t["change"])
        survived = best["ci95"][1] < 0
        out["groups"][name] = {"markets": markets, "trials": trials, "best": best["value"], "survived": survived}
        if survived:
            survivors[name] = best["value"]
    final = base
    for name, v in survivors.items():
        final = with_group(final, GROUPS[name][1], GROUPS[name][2], v)
    combined = {}
    for name in list(survivors):
        without = with_group(final, GROUPS[name][1], GROUPS[name][2], OFF[GROUPS[name][2]])
        res = compare(runner, GROUPS[name][0], without, final, SEASONS, rng=rng)
        combined[name] = {"removal_hurts": res["change"] < 0, **res}
        print(f"combined: removing {name} -> final vs without {res['change']:+.4%}", flush=True)
    for name, r in combined.items():
        if not r["removal_hurts"]:
            final = with_group(final, GROUPS[name][1], GROUPS[name][2], OFF[GROUPS[name][2]])
            survivors.pop(name)
    out.update({"combined_check": combined, "kept": survivors, "final_configs": {k: asdict(c) for k, c in final.items()}})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("KEPT:", survivors)


def confirm() -> None:
    res = json.loads(OUT.read_text())
    if not res["kept"]:
        res["confirmation"] = {}
        res["adopted"] = False
        OUT.write_text(json.dumps(res, indent=2, default=str))
        print("No group survived development; nothing to confirm.")
        return
    runner = Runner(setup())
    rng = np.random.default_rng(SEED + 1)
    base = base_configs()
    final = {k: E.Config(**v) for k, v in res["final_configs"].items()}
    markets = sorted({m for name in res["kept"] for m in GROUPS[name][0]})
    live = compare(runner, markets, base, final, [2026], weeks=(1, 5), rng=rng)
    second = compare(runner, markets, base, final, [2025], rng=rng)
    # pooled: 2025 + 2026 weeks 1-5, one bootstrap over all their games
    pooled = compare(_Pooled(runner), markets, base, final, ["pooled"], rng=rng)
    adopted = live["change"] < 0 and second["change"] < 0 and pooled["ci95"][1] < 0
    res["confirmation"] = {"live_2026_wk1_5": live, "second_look_2025": second, "pooled": pooled, "markets": markets}
    res["adopted"] = adopted
    OUT.write_text(json.dumps(res, indent=2, default=str))
    for k in ("live_2026_wk1_5", "second_look_2025", "pooled"):
        v = res["confirmation"][k]
        print(f"{k}: {v['change']:+.4%} CI [{v['ci95'][0]:+.4%}, {v['ci95'][1]:+.4%}]")
    print("ADOPTED" if adopted else "NOT ADOPTED")


class _Pooled:
    """Runner adapter: the season 'pooled' = all of 2025 plus 2026 weeks 1-5."""
    def __init__(self, runner):
        self.r = runner

    def pooled(self, market, cfgs, seasons, weeks=None):
        a = self.r.pooled(market, cfgs, [2025])
        b = self.r.pooled(market, cfgs, [2026], (1, 5))
        return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


if __name__ == "__main__":
    {"decide": decide, "confirm": confirm}[sys.argv[1]]()
