"""3c: does each matchup feature group survive ablation? (see phase3/PREREGISTRATION_3bc.md)

    python -m proplab.p3.ablation decide    # walk-forward 2019-2024 -> phase3/ablation.json
    python -m proplab.p3.ablation confirm   # final vs base on 2026 wks 1-5 (+ 2025 second look)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.p3 import engine as E
from proplab.p3 import matchup as MX
from proplab.p3.run import load_chosen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase3" / "ablation.json"
SEASONS = list(range(2019, 2025))
N_SIMS, SEED, BOOT = 2000, 44, 1000
WRTE_BASE = E.Config(model="M1", half_life=4.0, a_share=10.0, a_eff=10.0, redistribute=1.0, opponent=0.0)

GROUPS = {
    # name: (markets, config keys it changes, field, values to try)
    "depth":        (["wrte_rec", "wrte_rec_yds", "rb_rec", "rb_rec_yds"], ["tgt", "wrte_rec", "wrte_rec_yds"], "g_depth", [0.5, 1.0]),
    "gap":          (["rb_rush_yds"], ["car"], "g_gap", [0.5, 1.0]),
    "pressure":     (["qb_pass_yds", "qb_pass_tds", "qb_ints"], ["att"], "g_press", [0.5, 1.0]),
    "archetype":    (["wrte_rec", "wrte_rec_yds"], ["wrte_rec", "wrte_rec_yds"], "arch", [True]),
    "team_ratings": (list(E.ALL_MARKETS), ["att", "car", "tgt", "wrte_rec", "wrte_rec_yds"], "team_ratings", [True]),
}


def base_configs() -> dict:
    c = load_chosen()
    return {**c, "wrte_rec": WRTE_BASE, "wrte_rec_yds": WRTE_BASE}


def setup():
    conn = db.connect()
    D = E.load(conn)
    mx = MX.build(conn, D["u"], D["tg"])
    arch, arch_priors = MX.archetypes(conn, D["u"])
    tg = D["tg"].join(MX.team_rating_covariates(conn, D["tg"]), on=["game_id", "team"], how="left")
    u = D["u"].join(mx, on=["gsis_id", "game_id"], how="left").join(
        arch.select("gsis_id", "game_id", "arch", "arch_rate", "arch_ypc"), on=["gsis_id", "game_id"], how="left")
    return {"u": u, "tg": tg, "pools": D["pools"], "arch_priors": arch_priors}


class Runner:
    def __init__(self, S):
        self.S = S
        self.tf = E.team_features(S["tg"])
        self.feat: dict = {}
        self.cache: dict = {}
        self.memo: dict = {}

    def run(self, market, cfgs, season, weeks=None):
        for c in cfgs.values():
            if c.half_life not in self.feat:
                self.feat[c.half_life] = E.features(self.S["u"], self.S["tg"], c.half_life)
        key = (market, season, weeks, tuple(sorted((k, v) for k, v in cfgs.items())))
        if key not in self.memo:
            r = E.run_market(self.feat, self.tf, self.S["pools"], season, market, cfgs, N_SIMS, SEED + season,
                             weeks=weeks, cache=self.cache)
            self.memo[key] = {"crps_rows": r["crps_rows"], "game_ids": r["game_ids"], "keys": r["keys"],
                              "crps": r["crps"], "ece": r["ece"], "ece_null95": r["ece_null95"],
                              "coverage80": r.get("coverage80"), "rows": r["rows"]}
        return self.memo[key]

    def pooled(self, market, cfgs, seasons, weeks=None):
        rs = [self.run(market, cfgs, s, weeks) for s in seasons]
        return (np.concatenate([r["crps_rows"] for r in rs]), np.concatenate([r["game_ids"] for r in rs]),
                np.concatenate([r["keys"] for r in rs]))


def compare(runner, markets, base, variant, seasons, weeks=None, rng=None):
    """Mean over markets of (variant CRPS / base CRPS - 1), with a bootstrap over games."""
    per = {}
    for m in markets:
        b, g, kb = runner.pooled(m, base, seasons, weeks)
        v, _, kv = runner.pooled(m, variant, seasons, weeks)
        assert (kb == kv).all(), "base and variant must score identical rows"
        per[m] = (b, v, g)
    point = float(np.mean([per[m][1].mean() / per[m][0].mean() - 1 for m in markets]))
    by_m = {m: float(per[m][1].mean() / per[m][0].mean() - 1) for m in markets}
    # bootstrap: resample games (shared across markets)
    all_games = np.unique(np.concatenate([per[m][2] for m in markets]))
    gi = {m: np.searchsorted(all_games, per[m][2]) for m in markets}
    sums = {m: (np.bincount(gi[m], weights=per[m][0], minlength=len(all_games)),
                np.bincount(gi[m], weights=per[m][1], minlength=len(all_games))) for m in markets}
    stats = []
    for _ in range(BOOT):
        w = np.bincount(rng.integers(0, len(all_games), len(all_games)), minlength=len(all_games))
        stats.append(np.mean([(sums[m][1] @ w) / (sums[m][0] @ w) - 1 for m in markets]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"change": point, "ci95": [float(lo), float(hi)], "by_market": by_m,
            "rows": {m: int(len(per[m][0])) for m in markets}}


def with_group(cfgs: dict, keys: list[str], field: str, value) -> dict:
    return {k: (replace(c, **{field: value}) if k in keys else c) for k, c in cfgs.items()}


def decide() -> None:
    S = setup()
    runner = Runner(S)
    rng = np.random.default_rng(SEED)
    base = base_configs()
    out = {"groups": {}, "seasons": SEASONS,
           "archetypes": S["arch_priors"].filter(pl.col("arch").str.starts_with("2026")).to_dicts()}
    survivors = {}
    for name, (markets, keys, field, values) in GROUPS.items():
        trials = []
        for v in values:
            res = compare(runner, markets, base, with_group(base, keys, field, v), SEASONS, rng=rng)
            trials.append({"value": v, **res})
            print(f"{name:13s} {field}={v}: change {res['change']:+.4%} CI [{res['ci95'][0]:+.4%}, {res['ci95'][1]:+.4%}] "
                  f"{ {m: f'{x:+.3%}' for m, x in res['by_market'].items()} }", flush=True)
        best = min(trials, key=lambda t: t["change"])
        survived = best["ci95"][1] < 0
        out["groups"][name] = {"markets": markets, "trials": trials, "best": best["value"], "survived": survived}
        if survived:
            survivors[name] = best["value"]

    # combine survivors; drop any whose removal no longer hurts
    final = base
    for name, v in survivors.items():
        final = with_group(final, GROUPS[name][1], GROUPS[name][2], v)
    combined = {}
    for name, v in list(survivors.items()):
        off = {"g_depth": 0.0, "g_gap": 0.0, "g_press": 0.0, "arch": False, "team_ratings": False}[GROUPS[name][2]]
        without = with_group(final, GROUPS[name][1], GROUPS[name][2], off)
        res = compare(runner, GROUPS[name][0], without, final, SEASONS, rng=rng)
        combined[name] = {"removal_hurts": res["change"] < 0, **res}
        print(f"combined: removing {name} -> final vs without {res['change']:+.4%}", flush=True)
    for name, r in combined.items():
        if not r["removal_hurts"]:
            final = with_group(final, GROUPS[name][1], GROUPS[name][2],
                               {"g_depth": 0.0, "g_gap": 0.0, "g_press": 0.0, "arch": False, "team_ratings": False}[GROUPS[name][2]])
            survivors.pop(name)
    out["combined_check"] = combined
    out["kept"] = survivors
    out["final_configs"] = {k: asdict(c) for k, c in final.items()}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("KEPT:", survivors)


def load_final() -> dict:
    raw = json.loads(OUT.read_text())["final_configs"]
    return {k: E.Config(**v) for k, v in raw.items()}


def confirm() -> None:
    S = setup()
    runner = Runner(S)
    rng = np.random.default_rng(SEED + 1)
    base, final = base_configs(), load_final()
    res = json.loads(OUT.read_text())
    conf = {}
    for name in res["kept"]:
        markets = GROUPS[name][0]
        conf[name] = {
            "live_2026_wk1_5": compare(runner, markets, base, final, [2026], weeks=(1, 5), rng=rng),
            "second_look_2025": compare(runner, markets, base, final, [2025], rng=rng),
        }
        print(name, {k: (round(v["change"], 5), [round(x, 5) for x in v["ci95"]]) for k, v in conf[name].items()}, flush=True)
    res["confirmation"] = conf
    OUT.write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    {"decide": decide, "confirm": confirm}[sys.argv[1]]()
