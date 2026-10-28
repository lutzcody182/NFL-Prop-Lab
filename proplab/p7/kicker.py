"""Phase 7C: kicker points (3 x field goals made + extra points made). See phase7/PREREGISTRATION.md.

  B1  field goals and extra points per game ~ Poisson, means = the kicker's shrunk recent averages
  K   means from a Poisson regression on the team's implied points and the kicker's shrunk recent rate,
      fit on the three seasons before the one being projected

Half-life and shrinkage are tuned on 2024 only; 2025 is scored once.

    python -m proplab.p7.kicker tune       # -> phase7/kicker_tuning.json
    python -m proplab.p7.kicker backtest   # -> phase7/kicker_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from proplab import db
from proplab.p1.data import decay_for, ewm_prior_sums
from proplab.p1.model import crps_samples, ece, randomized_pit
from proplab.p3.engine import coverage_tolerance, ece_null95
from proplab.p3.matchup import _q

ROOT = Path(__file__).resolve().parents[2]
TUNING = ROOT / "phase7" / "kicker_tuning.json"
RESULTS = ROOT / "phase7" / "kicker_results.json"
LINES = [5.5, 7.5, 9.5]
N_SIMS, SEED, BOOT = 4000, 909, 1000
GRID_H, GRID_A = [4.0, 8.0, 16.0], [4.0, 10.0, 20.0]


def load(conn) -> pl.DataFrame:
    """One row per (kicker, game) for every kicker active (ACT) in a regular-season game."""
    df = _q(conn, """
        SELECT r.gsis_id, p.display_name AS name, r.season, r.week, r.team, g.game_id, g.kickoff_utc,
               CASE WHEN g.home_team = r.team THEN g.away_team ELSE g.home_team END AS opp, (g.home_team = r.team) AS is_home,
               coalesce(pg.fg_made, 0) AS fg, coalesce(pg.pat_made, 0) AS pat, (res.game_id IS NOT NULL) AS played,
               CASE WHEN g.home_team = r.team THEN l.spread_line ELSE -l.spread_line END::float AS team_spread,
               l.total_line::float AS total_line
        FROM roster_status r
        JOIN games g ON g.season = r.season AND g.week = r.week AND r.team IN (g.home_team, g.away_team) AND g.game_type = 'REG'
        LEFT JOIN players p ON p.gsis_id = r.gsis_id
        LEFT JOIN player_game pg ON pg.gsis_id = r.gsis_id AND pg.game_id = g.game_id
        LEFT JOIN game_results res ON res.game_id = g.game_id
        LEFT JOIN game_lines l ON l.game_id = g.game_id AND l.book = 'nflverse_close'
        WHERE r.status = 'ACT' AND r.position = 'K'""")
    # a team occasionally lists two kickers; keep the one who kicked (or, before the game, the one listed first)
    df = df.with_columns((pl.col("fg") + pl.col("pat")).alias("_k")) \
           .sort("game_id", "team", "_k", "gsis_id", descending=[False, False, True, False]) \
           .unique(subset=["game_id", "team"], keep="first", maintain_order=True).drop("_k")
    return df.with_columns((pl.col("total_line") / 2 + pl.col("team_spread") / 2).clip(6, 45).alias("implied")) \
             .sort("gsis_id", "kickoff_utc")


def features(df: pl.DataFrame, h: float) -> pl.DataFrame:
    df = df.sort("gsis_id", "kickoff_utc")
    k = df["gsis_id"].to_numpy()
    p = df["played"].to_numpy().astype(float)
    d = decay_for(h)
    return df.with_columns(pl.Series("S_n", ewm_prior_sums(k, p, d)),
                           pl.Series("S_fg", ewm_prior_sums(k, df["fg"].to_numpy() * p, d)),
                           pl.Series("S_pat", ewm_prior_sums(k, df["pat"].to_numpy() * p, d)),
                           pl.Series("n_prev", ewm_prior_sums(k, p, 1.0)))


def league(df: pl.DataFrame, season: int) -> tuple[float, float]:
    h = df.filter(pl.col("played") & pl.col("season").is_between(season - 3, season - 1))
    return float(h["fg"].mean()), float(h["pat"].mean())


def shrunk(df: pl.DataFrame, lfg: float, lpat: float, a: float) -> pl.DataFrame:
    return df.with_columns(((pl.col("S_fg") + a * lfg) / (pl.col("S_n") + a)).alias("r_fg"),
                           ((pl.col("S_pat") + a * lpat) / (pl.col("S_n") + a)).alias("r_pat"))


def _poisson_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    def nll(b):
        eta = np.clip(X @ b, -10, 5)
        return float(np.sum(np.exp(eta) - y * eta))
    def grad(b):
        eta = np.clip(X @ b, -10, 5)
        return X.T @ (np.exp(eta) - y)
    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x


def design(df: pl.DataFrame, stat: str) -> np.ndarray:
    return np.column_stack([np.ones(df.height), np.log(df["implied"].fill_null(22.0).to_numpy()),
                            np.log(np.clip(df[f"r_{stat}"].to_numpy(), 0.05, None))])


def params(df: pl.DataFrame, season: int, model: str, h: float, a: float) -> pl.DataFrame:
    f = features(df, h)
    lfg, lpat = league(f, season)
    f = shrunk(f, lfg, lpat, a)
    if model == "B1":
        return f.with_columns(pl.col("r_fg").alias("mu_fg"), pl.col("r_pat").alias("mu_pat"))
    train = f.filter(pl.col("played") & pl.col("season").is_between(season - 3, season - 1) & pl.col("implied").is_not_null())
    out = f
    for stat in ("fg", "pat"):
        b = _poisson_fit(design(train, stat), train[stat].to_numpy().astype(float))
        out = out.with_columns(pl.Series(f"mu_{stat}", np.exp(np.clip(design(out, stat) @ b, -10, 5))))
    return out


def simulate(rows: pl.DataFrame, n: int, rng) -> np.ndarray:
    fg = rng.poisson(rows["mu_fg"].to_numpy()[:, None], size=(rows.height, n))
    pat = rng.poisson(rows["mu_pat"].to_numpy()[:, None], size=(rows.height, n))
    return (3 * fg + pat).astype(np.float32)


def scored_rows(p: pl.DataFrame, season: int, weeks=None) -> pl.DataFrame:
    r = p.filter((pl.col("season") == season) & pl.col("played") & (pl.col("n_prev") >= 3))
    if weeks:
        r = r.filter(pl.col("week").is_between(*weeks))
    return r.sort("game_id", "gsis_id")


def evaluate(rows: pl.DataFrame, draws: np.ndarray, rng) -> dict:
    y = (3 * rows["fg"] + rows["pat"]).to_numpy().astype(float)
    crps = crps_samples(draws, y)
    probs = np.concatenate([(draws > L).mean(axis=1) for L in LINES])
    outs = np.concatenate([(y > L).astype(float) for L in LINES])
    pit = randomized_pit(draws, y, rng)
    return {"rows": int(len(y)), "crps_rows": crps, "crps": float(crps.mean()), "ece": ece(probs, outs),
            "ece_null95": ece_null95(probs, rng), "coverage80": float(((pit >= 0.1) & (pit <= 0.9)).mean()),
            "coverage_tol": coverage_tolerance(len(y)), "brier": float(((probs - outs) ** 2).mean()),
            "mean_pred": float(draws.mean()), "mean_actual": float(y.mean())}


def run(df, season, model, h, a, weeks=None):
    p = params(df, season, model, h, a)
    rows = scored_rows(p, season, weeks)
    draws = simulate(rows, N_SIMS, np.random.default_rng(SEED + season))
    return rows, evaluate(rows, draws, np.random.default_rng(SEED))


def tune() -> None:
    df = load(db.connect())
    out = {}
    for model in ("B1", "K"):
        best = None
        for h in GRID_H:
            for a in GRID_A:
                _, r = run(df, 2024, model, h, a)
                print(f"{model} h={h} a={a}: crps {r['crps']:.4f}", flush=True)
                if best is None or r["crps"] < best[2]:
                    best = (h, a, r["crps"])
        out[model] = {"half_life": best[0], "a": best[1], "crps_2024": best[2]}
    TUNING.parent.mkdir(exist_ok=True)
    TUNING.write_text(json.dumps(out, indent=2))
    print(out)


def backtest() -> None:
    df = load(db.connect())
    t = json.loads(TUNING.read_text())
    res = {"tuning": t, "splits": {}}
    rng = np.random.default_rng(SEED)
    for split, season, weeks in (("tuning_2024", 2024, None), ("test_2025", 2025, None), ("live_2026", 2026, (1, 5))):
        rb, b = run(df, season, "B1", t["B1"]["half_life"], t["B1"]["a"], weeks)
        rk, k = run(df, season, "K", t["K"]["half_life"], t["K"]["a"], weeks)
        assert (rb["game_id"] == rk["game_id"]).all() and (rb["gsis_id"] == rk["gsis_id"]).all()
        games = rk["game_id"].to_numpy()
        ug, gi = np.unique(games, return_inverse=True)
        sb = np.bincount(gi, weights=b["crps_rows"]); sk = np.bincount(gi, weights=k["crps_rows"])
        stats = []
        for _ in range(BOOT):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
            stats.append((sk @ w - sb @ w) / (w @ np.bincount(gi)))
        lo, hi = np.percentile(stats, [2.5, 97.5])
        d = k["crps"] - b["crps"]
        crit = {"beats_B1": bool(hi < 0), "calibrated": bool(k["ece"] <= k["ece_null95"]),
                "coverage_ok": bool(abs(k["coverage80"] - 0.8) <= k["coverage_tol"])}
        strip = lambda r: {x: v for x, v in r.items() if x != "crps_rows"}
        res["splits"][split] = {"models": {"B1": strip(b), "K": strip(k)}, "vs_B1": {"mean_diff": float(d), "ci95": [float(lo), float(hi)],
                                "pct": float(d / b["crps"])}, "criteria": crit, "passed": all(crit.values())}
        print(f"{split}: K {k['crps']:.4f} B1 {b['crps']:.4f} diff {d / b['crps']:+.2%} [{lo / b['crps']:+.2%}, {hi / b['crps']:+.2%}] "
              f"ece {k['ece']:.3f}/{k['ece_null95']:.3f} cov {k['coverage80']:.3f}±{k['coverage_tol']:.3f} {crit}", flush=True)
    RESULTS.write_text(json.dumps(res, indent=2))


def project(conn, season: int, week: int, lines: pl.DataFrame) -> list[dict]:
    """Next-week kicker points: each team's kicker from its last game; current consensus lines."""
    df = load(conn)
    t = json.loads(TUNING.read_text())["K"]
    played = df.filter(pl.col("played"))
    last = played.sort("kickoff_utc").group_by("team").agg(pl.col("gsis_id").last(), pl.col("name").last())
    games = _q(conn, f"""SELECT g.game_id, g.season, g.week, g.kickoff_utc, s.team, s.opp, (s.team = g.home_team) AS is_home
                          FROM games g CROSS JOIN LATERAL (VALUES (g.home_team, g.away_team), (g.away_team, g.home_team)) s(team, opp)
                          WHERE g.season = {season} AND g.week = {week} AND g.game_type = 'REG'""")
    fut = games.join(last, on="team").join(lines, on=["game_id", "team"], how="left").with_columns(
        pl.lit(0).alias("fg"), pl.lit(0).alias("pat"), pl.lit(False).alias("played"),
        (pl.col("total_line") / 2 + pl.col("team_spread") / 2).clip(6, 45).alias("implied"))
    cur = df.filter(~((pl.col("season") == season) & (pl.col("week") == week)))
    allr = pl.concat([cur, fut.select(cur.columns).cast(dict(cur.schema))])
    p = params(allr, season, "K", t["half_life"], t["a"]).filter((pl.col("season") == season) & (pl.col("week") == week)
                                                                 & (pl.col("n_prev") >= 3)).sort("kickoff_utc", "team")
    draws = simulate(p, N_SIMS, np.random.default_rng(606))
    from proplab.p8.recal import MAPS, apply_map   # Phase 8 recalibration, if adopted for this market
    p11 = MAPS.parents[1] / "phase11" / "recalibration.json"
    src = p11 if p11.exists() else MAPS
    recal = json.loads(src.read_text())["maps"].get("k_points", {}) if src.exists() else {}
    if recal.get("adopted") and draws.shape[0]:
        draws = apply_map("k_points", draws, recal)
    grid_lv = [(i + 0.5) / 100 for i in range(100)]
    qs = [round(0.05 * i, 2) for i in range(1, 20)]
    out = []
    for i, r in enumerate(p.iter_rows(named=True)):
        d = draws[i]
        out.append({"player": r["name"], "gsis_id": r["gsis_id"], "pos": "K", "team": r["team"], "opp": r["opp"],
                    "home": r["is_home"], "kickoff": r["kickoff_utc"].isoformat(), "game_id": r["game_id"],
                    "mean": float(d.mean()), "p10": float(np.quantile(d, 0.1)), "p90": float(np.quantile(d, 0.9)),
                    "q": [float(x) for x in np.quantile(d, qs)], "g": [round(float(x), 1) for x in np.quantile(d, grid_lv)],
                    "over": {str(L): float((d > L).mean()) for L in LINES}, "volume": float(r["implied"])})
    return out


if __name__ == "__main__":
    {"tune": tune, "backtest": backtest}[sys.argv[1]]()
