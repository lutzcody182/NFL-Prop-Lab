"""Phase 10: do role, QB change and air yards improve the Phase 1 WR/TE model? (phase10/PREREGISTRATION.md)

    python -m proplab.p10.ablation decide    # walk-forward 2019-2024 -> phase10/ablation.json
    python -m proplab.p10.ablation confirm   # 2025, 2026 weeks 1-6, pooled; re-check the WR/TE yards gate on 2025
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.p1 import data
from proplab.p1 import model as M
from proplab.p1.backtest import load_configs
from proplab.p3 import engine as E
from proplab.p10 import features as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase10" / "ablation.json"
SEASONS = list(range(2019, 2025))
N_SIMS, SEED, BOOT = 2000, 1010, 1000
GROUPS = {"role": ("role", [1.0]), "qb_change": ("qb", [0.5, 1.0]), "air_yards": ("ay", [0.25, 0.5])}


def setup():
    conn = db.connect()
    D = E.load(conn)
    f = F.build(conn, D["u"])
    u = data.load_universe(conn).join(f, on=["gsis_id", "game_id"], how="left")
    tg, cy = data.load_team_games(conn), data.load_catch_yards(conn)
    return {"u": u, "tg": tg, "cy": cy, "tf": M.team_features(tg), "feat": {}}


class Runner:
    def __init__(self, S):
        self.S, self.memo = S, {}

    def run(self, cfg: M.Config, season: int, weeks=None):
        key = (cfg, season, weeks)
        if key not in self.memo:
            if cfg.half_life not in self.S["feat"]:
                self.S["feat"][cfg.half_life] = M.assemble(self.S["u"], self.S["tg"], cfg.half_life)
            r = M.run_season(self.S["feat"][cfg.half_life], self.S["tf"], self.S["cy"], season, cfg, n_sims=N_SIMS,
                             seed=SEED + season, weeks=weeks)
            self.memo[key] = {"rec": r["receptions"]["crps_rows"], "yds": r["yards"]["crps_rows"], "games": r["game_ids"],
                              "keys": r["keys"], "full": r}
        return self.memo[key]


def compare(runner, base, variant, sets, rng) -> dict:
    """sets: list of (season, weeks). Mean over the two markets of (variant / base - 1), bootstrap over games."""
    parts = [(runner.run(base, s, w), runner.run(variant, s, w)) for s, w in sets]
    for b, v in parts:
        assert (b["keys"] == v["keys"]).all()
    games = np.concatenate([b["games"] for b, _ in parts])
    ug, gi = np.unique(games, return_inverse=True)
    sums = {}
    for m in ("rec", "yds"):
        bm, vm = np.concatenate([b[m] for b, _ in parts]), np.concatenate([v[m] for _, v in parts])
        sums[m] = (np.bincount(gi, weights=bm), np.bincount(gi, weights=vm))
    by = {m: float(s[1].sum() / s[0].sum() - 1) for m, s in sums.items()}
    stats = []
    for _ in range(BOOT):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        stats.append(np.mean([(s[1] @ w) / (s[0] @ w) - 1 for s in sums.values()]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"change": float(np.mean(list(by.values()))), "ci95": [float(lo), float(hi)], "by_market": by, "rows": int(len(games))}


def decide() -> None:
    runner = Runner(setup())
    rng = np.random.default_rng(SEED)
    base = load_configs()["M1"]
    sets = [(s, None) for s in SEASONS]
    out = {"seasons": SEASONS, "base": asdict(base), "groups": {}}
    surv = {}
    for name, (field, values) in GROUPS.items():
        trials = []
        for v in values:
            res = compare(runner, base, replace(base, **{field: v}), sets, rng)
            trials.append({"value": v, **res})
            print(f"{name:10s} {field}={v}: {res['change']:+.4%} CI [{res['ci95'][0]:+.4%}, {res['ci95'][1]:+.4%}] {res['by_market']}", flush=True)
        best = min(trials, key=lambda t: t["change"])
        ok = best["ci95"][1] < 0
        out["groups"][name] = {"trials": trials, "best": best["value"], "survived": ok}
        if ok:
            surv[name] = best["value"]
    final = replace(base, **{GROUPS[n][0]: v for n, v in surv.items()})
    combined = {}
    for n in list(surv):
        without = replace(final, **{GROUPS[n][0]: 0.0})
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
    live = compare(runner, base, final, [(2026, (1, 6))], rng)
    second = compare(runner, base, final, [(2025, None)], rng)
    pooled = compare(runner, base, final, [(2025, None), (2026, (1, 6))], rng)
    adopted = live["change"] < 0 and second["change"] < 0 and pooled["ci95"][1] < 0
    # re-check the WR/TE yards Phase 1 gate on 2025 with the final model (beats B2, calibrated, ranges right)
    from proplab.p1.backtest import bootstrap_diff
    b2 = load_configs()["B2"]
    f, b = runner.run(final, 2025)["full"], runner.run(b2, 2025)["full"]
    gate = {}
    for mk, key in (("receptions", "rec"), ("yards", "yds")):
        d, lo, hi = bootstrap_diff(runner.run(final, 2025)[key] - runner.run(b2, 2025)[key], f["game_ids"], np.random.default_rng(SEED))
        r = f[mk]
        gate[mk] = {"vs_B2_pct": d / b[mk]["crps"], "ci95": [lo / b[mk]["crps"], hi / b[mk]["crps"]], "beats_B2": hi < 0,
                    "ece": r["ece"], "coverage80": r["coverage80"]}
    out.update({"confirmation": {"live_2026_wk1_6": live, "second_look_2025": second, "pooled": pooled},
                "adopted": adopted, "gate_2025_final": gate})
    OUT.write_text(json.dumps(out, indent=2, default=str))
    for k, v in out["confirmation"].items():
        print(f"{k}: {v['change']:+.4%} CI [{v['ci95'][0]:+.4%}, {v['ci95'][1]:+.4%}] {v['by_market']}")
    print("gate 2025:", gate)
    print("ADOPTED" if adopted else "NOT ADOPTED")


if __name__ == "__main__":
    {"decide": decide, "confirm": confirm}[sys.argv[1]]()
