"""Score the locked configurations: 2024 (tuning, for reference), 2025 (locked test,
scored once), 2026 weeks 1-5 (live holdout). Applies the pre-registered gate and
writes phase1/results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.p1 import data, model as M

ROOT = Path(__file__).resolve().parents[2]
TUNING = ROOT / "phase1" / "tuning.json"
OUT = ROOT / "phase1" / "results.json"
N_SIMS = 2000
SEED = 2025
BOOT = 1000


def load_configs() -> dict[str, M.Config]:
    chosen = json.loads(TUNING.read_text())["chosen"]
    return {name: M.Config(**{k: (v if k == "model" else (bool(v) if k == "calibrate" else float(v)))
                              for k, v in cfg.items()}) for name, cfg in chosen.items()}


def bootstrap_diff(diff_rows: np.ndarray, games: np.ndarray, rng) -> tuple[float, float, float]:
    """Paired bootstrap over games of mean(per-row CRPS difference)."""
    uniq, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=diff_rows)
    counts = np.bincount(inv)
    stats = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats.append(sums[pick].sum() / counts[pick].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(diff_rows.mean()), float(lo), float(hi)


def reliability(draws: np.ndarray, y: np.ndarray, lines: list[float], bins: int = 10) -> list[dict]:
    p = np.concatenate([(draws > L).mean(axis=1) for L in lines])
    o = np.concatenate([(y > L).astype(float) for L in lines])
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            out.append({"pred": float(p[m].mean()), "obs": float(o[m].mean()), "n": int(m.sum())})
    return out


def score_split(feat, tf, cy, season, cfgs, weeks=None) -> dict:
    runs = {name: M.run_season(feat[cfg.half_life], tf, cy, season, cfg, n_sims=N_SIMS, seed=SEED,
                               weeks=weeks, keep_draws=True)
            for name, cfg in cfgs.items()}
    keys = runs["M1"]["keys"]
    for r in runs.values():
        assert (r["keys"] == keys).all(), "models must be scored on identical rows"
    games = runs["M1"]["game_ids"]
    rng = np.random.default_rng(SEED)
    split = {"season": season, "weeks": weeks, "rows": int(len(keys)), "games": int(len(np.unique(games))),
             "models": {}, "vs_B2": {}, "vs_B1": {}, "by_week": {}, "by_pos": {}, "reliability": {}}
    frame = runs["M1"]["frame"]
    for name, r in runs.items():
        split["models"][name] = {
            "config": r["config"], "calibration": r["calibration"], "dispersion": r["dispersion"],
            **{mk: {k: v for k, v in r[mk].items() if k != "crps_rows"} for mk in ("receptions", "yards")}}
    for mk, ycol, lines, dk in (("receptions", "receptions", M.REC_LINES, "rec_draws"),
                                ("yards", "rec_yds", M.YDS_LINES, "yds_draws")):
        m1 = runs["M1"][mk]["crps_rows"]
        for base in ("B2", "B1"):
            mean, lo, hi = bootstrap_diff(m1 - runs[base][mk]["crps_rows"], games, rng)
            split[f"vs_{base}"][mk] = {"mean_diff": mean, "ci95": [lo, hi],
                                       "pct": mean / runs[base][mk]["crps"]}
        weeks_arr = frame["week"].to_numpy()
        pos_arr = frame["pos"].to_numpy()
        split["by_week"][mk] = {name: {int(w): float(r[mk]["crps_rows"][weeks_arr == w].mean())
                                       for w in np.unique(weeks_arr)} for name, r in runs.items()}
        split["by_pos"][mk] = {name: {p: float(r[mk]["crps_rows"][pos_arr == p].mean()) for p in M.EVAL_POS}
                               for name, r in runs.items()}
        y = frame[ycol].to_numpy().astype(float)
        split["reliability"][mk] = {name: reliability(r[dk], y, lines) for name, r in runs.items()}
    return split


def gate(test: dict) -> dict:
    checks = {}
    for mk in ("receptions", "yards"):
        d = test["vs_B2"][mk]
        m1 = test["models"]["M1"][mk]
        checks[mk] = {
            "beats_B2": d["ci95"][1] < 0,
            "ece_ok": m1["ece"] <= 0.02,
            "coverage_ok": 0.77 <= m1["coverage80"] <= 0.83,
        }
        checks[mk]["passed"] = all(checks[mk].values())
    return {"criteria": checks, "passed": all(c["passed"] for c in checks.values())}


def main() -> None:
    conn = db.connect()
    u, tg, cy = data.load_universe(conn), data.load_team_games(conn), data.load_catch_yards(conn)
    tf = M.team_features(tg)
    cfgs = load_configs()
    feat = {h: M.assemble(u, tg, h) for h in {c.half_life for c in cfgs.values()}}

    results = {"configs": {k: v.label() for k, v in cfgs.items()}}
    results["tuning_2024"] = score_split(feat, tf, cy, 2024, cfgs)
    results["test_2025"] = score_split(feat, tf, cy, 2025, cfgs)
    results["live_2026"] = score_split(feat, tf, cy, 2026, cfgs, weeks=(1, 5))
    results["gate"] = gate(results["test_2025"])
    results["gate_live_2026_for_reference"] = gate(results["live_2026"])
    OUT.write_text(json.dumps(results, indent=2))

    for split in ("tuning_2024", "test_2025", "live_2026"):
        s = results[split]
        print(f"\n== {split}: {s['rows']:,} player-games, {s['games']} games")
        for mk in ("receptions", "yards"):
            for name in ("B1", "B2", "M1"):
                m = s["models"][name][mk]
                print(f"  {mk:10s} {name}: CRPS {m['crps']:.4f}  MAE {m['mae_median']:.3f}  "
                      f"cov80 {m['coverage80']:.3f}  ECE {m['ece']:.4f}  Brier {m['brier']:.4f}  "
                      f"bias {m['mean_pred'] - m['mean_actual']:+.3f}")
            d = s["vs_B2"][mk]
            print(f"  {mk:10s} M1-B2: {d['mean_diff']:+.4f} ({d['pct']:+.2%}) CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    print("\nGATE (2025):", json.dumps(results["gate"], indent=2))


if __name__ == "__main__":
    main()
