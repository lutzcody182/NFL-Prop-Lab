"""Phase 8: recalibrate ranges and over/under chances (see phase8/PREREGISTRATION.md).

    python -m proplab.p8.recal      # fit maps, score 2025 and 2026 wks 1-5 -> phase8/results.json, phase8/recalibration.json

Quantile maps for yards/counts, logistic maps for yes/no markets. Model and baseline each get their own map.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from proplab import db
from proplab.p1.model import randomized_pit
from proplab.p3 import engine as E
from proplab.p3.run import boot_diff, load_chosen, setup
from proplab.p7 import kicker as K

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "phase8" / "results.json"
MAPS = ROOT / "phase8" / "recalibration.json"
GRID = np.linspace(0, 1, 201)
N_FIT, N_EVAL, SEED = 2000, 4000, 88
FIT_TEST, FIT_LIVE = [2022, 2023, 2024], [2023, 2024, 2025]
MARKETS = list(E.MARKETS) + list(E.NEW_MARKETS)


# ---------------------------------------------------------------- maps

def fit_quantile_map(pits: np.ndarray) -> list:
    return [float(x) for x in np.quantile(pits, GRID)]


def apply_quantile_map(draws: np.ndarray, qmap) -> np.ndarray:
    s = np.sort(draws, axis=1)
    n = s.shape[1]
    lv = np.interp((np.arange(n) + 0.5) / n, GRID, np.asarray(qmap))
    return s[:, np.clip((lv * n).astype(int), 0, n - 1)]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_logistic_map(p: np.ndarray, y: np.ndarray) -> dict:
    x = _logit(p)
    def nll(b):
        z = np.clip(b[0] + b[1] * x, -30, 30)
        return float(np.sum(np.log1p(np.exp(z)) - y * z))
    b = minimize(nll, np.array([0.0, 1.0]), method="BFGS").x
    return {"a": float(b[0]), "b": float(b[1])}


def apply_logistic_map(draws: np.ndarray, m: dict) -> np.ndarray:
    p = 1 / (1 + np.exp(-(m["a"] + m["b"] * _logit(draws.mean(axis=1)))))
    n = draws.shape[1]
    k = np.rint(p * n).astype(int)
    return (np.arange(n)[None, :] < k[:, None]).astype(np.float32)


def apply_map(market: str, draws: np.ndarray, m: dict) -> np.ndarray:
    return apply_logistic_map(draws, m) if m["type"] == "logistic" else apply_quantile_map(draws, m["q"])


def fit_map(market: str, parts: list[tuple[np.ndarray, np.ndarray]], rng) -> dict:
    if market in E.BINARY:
        p = np.concatenate([d.mean(axis=1) for d, _ in parts])
        y = np.concatenate([y for _, y in parts])
        return {"type": "logistic", **fit_logistic_map(p, y)}
    pits = np.concatenate([randomized_pit(d, y, rng) for d, y in parts])
    return {"type": "quantile", "q": fit_quantile_map(pits)}


# ---------------------------------------------------------------- runs

class Engine:
    def __init__(self):
        self.D, self.feat, self.tf = setup()
        chosen = load_chosen()
        self.variants = {"chosen": chosen,
                         "B2": {vt: replace(c, model="B2", redistribute=0.0, opponent=0.0) for vt, c in chosen.items()}}
        self.cache = {v: {} for v in self.variants}

    def run(self, market, variant, season, n, weeks=None):
        r = E.run_market(self.feat, self.tf, self.D["pools"], season, market, self.variants[variant], n, SEED + season,
                         weeks=weeks, cache=self.cache[variant], keep=True)
        return r["draws"], E.outcome(market, r["frame"]), r["game_ids"], r["frame"]


class Kicker:
    variants = {"chosen": "K", "B2": "B1"}   # kicker's baseline is B1

    def __init__(self):
        self.df = K.load(db.connect())
        self.t = json.loads(K.TUNING.read_text())

    def run(self, market, variant, season, n, weeks=None):
        model = self.variants[variant]
        p = K.params(self.df, season, model, self.t[model]["half_life"], self.t[model]["a"])
        rows = K.scored_rows(p, season, weeks)
        draws = K.simulate(rows, n, np.random.default_rng(K.SEED + season))
        y = (3 * rows["fg"] + rows["pat"]).to_numpy().astype(float)
        return draws, y, rows["game_id"].to_numpy(), rows


def score(market, draws, y, rng):
    if market == "k_points":
        rows = pl.DataFrame({"fg": np.zeros(len(y)), "pat": y})   # evaluate() rebuilds y = 3*fg + pat
        return K.evaluate(rows, draws, rng)
    return E.score(market, draws, y, rng)


def strip(r):
    return {k: v for k, v in r.items() if k not in ("crps_rows",)}


def main() -> None:
    eng, kick = Engine(), Kicker()
    rng = np.random.default_rng(SEED)
    chosen = load_chosen()
    results, live_maps = {}, {}
    for market in MARKETS + ["k_points"]:
        src = kick if market == "k_points" else eng
        out = {"label": "Kicker points" if market == "k_points" else E.MARKET_LABELS[market]}
        maps = {}
        for variant in ("chosen", "B2"):
            parts = [src.run(market, variant, s, N_FIT)[:2] for s in FIT_TEST]
            maps[variant] = fit_map(market, parts, np.random.default_rng(1))
        ev = {}
        for variant in ("chosen", "B2"):
            d, y, g, _ = src.run(market, variant, 2025, N_EVAL)
            ev[variant] = {"raw": score(market, d, y, np.random.default_rng(2)),
                           "recal": score(market, apply_map(market, d, maps[variant]), y, np.random.default_rng(2)), "games": g}
        games = ev["chosen"]["games"]
        if market == "k_points":
            kept = True
        else:
            vt = E.ALL_MARKETS[market][1]
            kept = any(chosen[v].model != "B2" for v in (("car", "tgt") if vt == "both" else (vt,)))
        verdict = {}
        for version in ("raw", "recal"):
            c, b = ev["chosen"][version], ev["B2"][version]
            d, lo, hi = boot_diff(c["crps_rows"], b["crps_rows"], games, rng)
            crit = {"beats_baseline": bool(kept and hi < 0), "calibrated": bool(c["ece"] <= c["ece_null95"])}
            if "coverage80" in c:
                crit["coverage_ok"] = bool(abs(c["coverage80"] - 0.8) <= c["coverage_tol"])
            verdict[version] = {"model": strip(c), "baseline": strip(b), "vs_baseline": {"pct": d / b["crps"], "ci95": [lo / b["crps"], hi / b["crps"]]},
                                "criteria": crit, "passed": all(crit.values())}
        raw_c, rec_c = ev["chosen"]["raw"], ev["chosen"]["recal"]
        adopted = bool(rec_c["ece"] < raw_c["ece"] and rec_c["crps"] <= raw_c["crps"])
        out.update({"test_2025": verdict, "adopted": adopted, "maps_2022_2024": maps})
        # live maps (2023-2025) and the 2026 weeks 1-5 check, model only
        lm = fit_map(market, [src.run(market, "chosen", s, N_FIT)[:2] for s in FIT_LIVE], np.random.default_rng(1))
        d, y, _, _ = src.run(market, "chosen", 2026, N_EVAL, weeks=(1, 5))
        r_raw, r_rec = score(market, d, y, np.random.default_rng(3)), score(market, apply_map(market, d, lm), y, np.random.default_rng(3))
        out["live_2026_wk1_5"] = {"raw": strip(r_raw), "recal": strip(r_rec)}
        live_maps[market] = {**lm, "adopted": adopted}
        results[market] = out
        v = verdict["recal"]
        print(f"{market:16s} raw {'PASS' if verdict['raw']['passed'] else 'fail'} -> recal {'PASS' if v['passed'] else 'fail'} "
              f"| ece {raw_c['ece']:.3f}->{rec_c['ece']:.3f} (noise {rec_c['ece_null95']:.3f}) "
              f"cov {raw_c.get('coverage80', float('nan')):.3f}->{rec_c.get('coverage80', float('nan')):.3f} "
              f"crps {raw_c['crps']:.4f}->{rec_c['crps']:.4f} vs base {v['vs_baseline']['pct']:+.2%} {v['criteria']} adopt={adopted}", flush=True)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, default=float))
    MAPS.write_text(json.dumps({"fit_seasons": FIT_LIVE, "maps": live_maps}, indent=1))


if __name__ == "__main__":
    main()
