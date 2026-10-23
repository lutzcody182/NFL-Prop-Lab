"""Phase 9B: game score simulation (see phase9/PREREGISTRATION.md, section B).

S: the 200 games from the previous five seasons nearest in (home implied points, away implied points); their actual
   final scores are the simulated outcomes.  N: margin ~ Normal(spread, 13.5), total ~ Normal(total, 13.5).

    python -m proplab.p9.scores test     # 2025 (+ 2026 wks 1-6) -> phase9/scores_results.json
    python -m proplab.p9.scores export   # next week's score draws, linked to the Phase 4 player draws
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm, spearmanr

from proplab import db
from proplab.p1.model import crps_samples
from proplab.p3.matchup import _q

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "phase9" / "scores_results.json"
K_NEIGHBORS, LOOKBACK, SD = 200, 5, 13.5
N_DRAWS, SEED, BOOT = 2000, 909, 1000


def games(conn) -> pl.DataFrame:
    g = _q(conn, """
        SELECT g.game_id, g.season, g.week, g.kickoff_utc, g.home_team, g.away_team,
               l.spread_line::float AS spread, l.total_line::float AS total, r.home_score, r.away_score
        FROM games g LEFT JOIN game_lines l ON l.game_id = g.game_id AND l.book = 'nflverse_close'
        LEFT JOIN game_results r ON r.game_id = g.game_id
        WHERE g.game_type = 'REG' ORDER BY g.kickoff_utc, g.game_id""")
    return g.with_columns((pl.col("total") / 2 + pl.col("spread") / 2).alias("imp_h"), (pl.col("total") / 2 - pl.col("spread") / 2).alias("imp_a"))


def library(g: pl.DataFrame, season: int) -> np.ndarray:
    h = g.filter(pl.col("season").is_between(season - LOOKBACK, season - 1) & pl.col("home_score").is_not_null() & pl.col("spread").is_not_null())
    return h.select("imp_h", "imp_a", "home_score", "away_score").to_numpy().astype(float)


def draws_S(lib: np.ndarray, imp_h: float, imp_a: float, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    d = (lib[:, 0] - imp_h) ** 2 + (lib[:, 1] - imp_a) ** 2
    nb = lib[np.argsort(d, kind="stable")[:K_NEIGHBORS]]
    pick = rng.integers(0, len(nb), n)
    return nb[pick, 2], nb[pick, 3]


def draws_N(spread: float, total: float, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    m, t = rng.normal(spread, SD, n), rng.normal(total, SD, n)
    return (t + m) / 2, (t - m) / 2


def events(h: np.ndarray, a: np.ndarray, spread: float, total: float) -> dict:
    m, t = h - a, h + a
    win = float((m > 0).mean() + 0.5 * (m == 0).mean())
    cov_push, over_push = (m == spread).mean(), (t == total).mean()
    cover = float((m > spread).mean() / max(1 - cov_push, 1e-9))
    over = float((t > total).mean() / max(1 - over_push, 1e-9))
    return {"win": win, "cover": cover, "over": over}


def evaluate(g: pl.DataFrame, season: int, weeks=None) -> dict:
    lib = library(g, season)
    rows = g.filter((pl.col("season") == season) & pl.col("home_score").is_not_null() & pl.col("spread").is_not_null())
    if weeks:
        rows = rows.filter(pl.col("week").is_between(*weeks))
    rng = np.random.default_rng(SEED + season)
    out = {k: {"brier": [], "crps_margin": [], "crps_total": []} for k in ("S", "N")}
    yv = {"win": [], "cover": [], "over": []}
    masks = {"cover": [], "over": []}
    for r in rows.iter_rows(named=True):
        m, t = r["home_score"] - r["away_score"], r["home_score"] + r["away_score"]
        y = {"win": 1.0 if m > 0 else (0.5 if m == 0 else 0.0), "cover": float(m > r["spread"]), "over": float(t > r["total"])}
        masks["cover"].append(m != r["spread"]); masks["over"].append(t != r["total"])
        for k in yv:
            yv[k].append(y[k])
        for name, (h, a) in (("S", draws_S(lib, r["imp_h"], r["imp_a"], N_DRAWS, rng)), ("N", draws_N(r["spread"], r["total"], N_DRAWS, rng))):
            e = events(h, a, r["spread"], r["total"])
            out[name]["brier"].append([e["win"], e["cover"], e["over"]])
            out[name]["crps_margin"].append(float(crps_samples((h - a)[None, :], np.array([float(m)]))[0]))
            out[name]["crps_total"].append(float(crps_samples((h + a)[None, :], np.array([float(t)]))[0]))
    Y = np.array([yv["win"], yv["cover"], yv["over"]]).T
    M = np.array([np.ones(len(Y), bool), masks["cover"], masks["over"]]).T
    res = {"games": int(len(Y))}
    per_game = {}
    for name in ("S", "N"):
        P = np.array(out[name]["brier"])
        sq = np.where(M, (P - Y) ** 2, np.nan)
        res[name] = {"brier_win": float(np.nanmean(sq[:, 0])), "brier_cover": float(np.nanmean(sq[:, 1])), "brier_over": float(np.nanmean(sq[:, 2])),
                     "brier_mean": float(np.nanmean(np.nanmean(sq, axis=0))), "crps_margin": float(np.mean(out[name]["crps_margin"])),
                     "crps_total": float(np.mean(out[name]["crps_total"]))}
        per_game[name] = np.nanmean(sq, axis=1)
    diff = per_game["S"] - per_game["N"]
    brng = np.random.default_rng(SEED)
    boots = [np.nanmean(diff[brng.integers(0, len(diff), len(diff))]) for _ in range(BOOT)]
    res["S_minus_N_brier"] = {"mean": float(np.nanmean(diff)), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
    return res


def test() -> None:
    g = games(db.connect())
    res = {"test_2025": evaluate(g, 2025), "live_2026_wk1_6": evaluate(g, 2026, (1, 6))}
    t = res["test_2025"]["S_minus_N_brier"]
    res["use_S"] = bool(t["mean"] < 0 and t["ci95"][1] < 0)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2))
    for k in ("test_2025", "live_2026_wk1_6"):
        r = res[k]
        print(k, r["games"], "S", {x: round(v, 4) for x, v in r["S"].items()}, "N", {x: round(v, 4) for x, v in r["N"].items()}, r["S_minus_N_brier"])
    print("use S" if res["use_S"] else "use N")


def attempts_margin_rho(conn, season: int) -> float:
    """Rank correlation between (home - away team pass attempts) and home margin, previous three seasons."""
    d = _q(conn, f"""
        WITH t AS (SELECT game_id, team, sum(coalesce(attempts, 0)) a FROM player_game GROUP BY 1, 2)
        SELECT th.a - ta.a AS da, r.home_score - r.away_score AS m
        FROM games g JOIN game_results r USING (game_id)
        JOIN t th ON th.game_id = g.game_id AND th.team = g.home_team JOIN t ta ON ta.game_id = g.game_id AND ta.team = g.away_team
        WHERE g.game_type = 'REG' AND g.season BETWEEN {season - 3} AND {season - 1}""")
    return float(spearmanr(d["da"], d["m"]).statistic)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.rint(a).astype("<i2").tobytes()).decode()


def dec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<i2").astype(float)


def export() -> None:
    """Add score draws to phase4/week_sims.json, ordered so the margin moves with the attempt gap (Gaussian copula)."""
    conn = db.connect()
    path = ROOT / "phase4" / "week_sims.json"
    sims = json.loads(path.read_text())
    season = sims["meta"]["season"]
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"use_S": False}
    g = games(conn)
    lib = library(g, season)
    rho = attempts_margin_rho(conn, season)
    import nflreadpy as nfl
    sched = {r["game_id"]: r for r in nfl.load_schedules([season]).select("game_id", "spread_line", "total_line").iter_rows(named=True)}
    rng = np.random.default_rng(SEED)
    for gm in sims["games"]:
        ln = sched.get(gm["game_id"], {})
        spread, total = ln.get("spread_line"), ln.get("total_line")
        if spread is None or total is None:
            continue
        _, _, away, home = gm["game_id"].split("_")
        n = len(dec(next(iter(gm["teams"].values()))["team_attempts"]))
        imp_h, imp_a = total / 2 + spread / 2, total / 2 - spread / 2
        h, a = draws_S(lib, imp_h, imp_a, n, rng) if res["use_S"] else draws_N(spread, total, n, rng)
        if not res["use_S"]:
            h, a = np.maximum(np.rint(h), 0), np.maximum(np.rint(a), 0)
        if home in gm["teams"] and away in gm["teams"]:
            da = dec(gm["teams"][home]["team_attempts"]) - dec(gm["teams"][away]["team_attempts"])
            z = norm.ppf((np.argsort(np.argsort(da + rng.uniform(0, 1e-3, n))) + 0.5) / n)
            latent = rho * z + np.sqrt(max(1 - rho ** 2, 0)) * rng.normal(size=n)
            order = np.argsort(np.argsort(latent))          # rank of each draw's latent value
            by_margin = np.argsort(h - a, kind="stable")    # score draws sorted by margin
            h, a = h[by_margin][order], a[by_margin][order]
        e = events(h, a, spread, total)
        gm["score"] = {"home": home, "away": away, "home_pts": b64(h), "away_pts": b64(a), "spread": spread, "total": total,
                       "method": "S" if res["use_S"] else "N", "rho_attempts_margin": rho, **e,
                       "median": [float(np.median(h)), float(np.median(a))]}
    sims["meta"]["scores"] = {"method": "S" if res["use_S"] else "N", "result": res}
    path.write_text(json.dumps(sims))
    print(f"scores added ({'S' if res['use_S'] else 'N'}), rho={rho:.2f}")


if __name__ == "__main__":
    {"test": test, "export": export}[sys.argv[1]]()
