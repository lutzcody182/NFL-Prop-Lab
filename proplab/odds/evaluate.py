"""Phase 2: compare locked M1 probabilities with sportsbook prices (see phase2/PREREGISTRATION.md).

    python -m proplab.odds.evaluate            # -> phase2/results.json

Close comparison: M1 with kickoff information vs the closing no-vig market.
Bet-time comparison (CLV): M1 with only what was known 36 hours before kickoff
(previous game's actives, bet-snapshot spread/total) vs the bet-time market.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.odds import oddsmath as om
from proplab.p1 import backtest, data, model as M

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase2" / "results.json"
MARKETS = {"player_receptions": ("receptions", "receptions"), "player_reception_yds": ("yards", "rec_yds")}
N_SIMS = 4000
SEED = 2026
BOOT = 1000
DISAGREE = 0.05
POOLED = (2023, 2024, 2025)


# ---------------------------------------------------------------- market side

def _q(conn, sql, params=()):
    cur = conn.execute(sql, params)
    return pl.DataFrame(cur.fetchall(), schema=[d.name for d in cur.description], orient="row", infer_schema_length=None)


def consensus(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, player, market, snapshot): modal line across books and the mean
    no-vig over probability of books posting that line."""
    d = rows.filter(pl.col("over_price").is_not_null() & pl.col("under_price").is_not_null()
                    & (pl.col("over_price") > 1) & (pl.col("under_price") > 1))
    if d.is_empty():
        return d
    o, u = d["over_price"].to_list(), d["under_price"].to_list()
    d = d.with_columns(
        pl.Series("p_mult", [om.novig_multiplicative(a, b) for a, b in zip(o, u)]),
        pl.Series("p_power", [om.novig_power(a, b) for a, b in zip(o, u)]),
        pl.Series("p_shin", [om.novig_shin(a, b) for a, b in zip(o, u)]),
    )
    key = ["game_id", "gsis_id", "market", "snapshot_tag"]
    modal = (d.group_by(key + ["line"]).agg(pl.len().alias("n"))
               .sort(key + ["n", "line"], descending=[False] * len(key) + [True, False])
               .group_by(key, maintain_order=True).first().select(key + ["line"]))
    return (d.join(modal, on=key + ["line"])
              .group_by(key + ["line"])
              .agg(pl.len().alias("n_books"), pl.col("p_mult").mean(), pl.col("p_power").mean(),
                   pl.col("p_shin").mean(), pl.col("over_price").max().alias("best_over"),
                   pl.col("under_price").max().alias("best_under"), pl.col("snapshot_ts").max()))


def bet_game_lines(conn) -> pl.DataFrame:
    """Median bet-snapshot spread/total per game across books (home perspective)."""
    return _q(conn, """
        SELECT game_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY spread_line)::float AS bet_spread,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY total_line)::float AS bet_total
        FROM game_lines WHERE book <> 'nflverse_close' AND NOT is_closing GROUP BY game_id""")


# ---------------------------------------------------------------- model side

def attach_model(cons: pl.DataFrame, rows: pl.DataFrame, rec: np.ndarray, yds: np.ndarray) -> pl.DataFrame:
    """For each consensus row matched to a simulated player-game, add P(over | no push)."""
    idx = {(g, p): i for i, (g, p) in enumerate(zip(rows["game_id"].to_list(), rows["gsis_id"].to_list()))}
    out = []
    for r in cons.iter_rows(named=True):
        i = idx.get((r["game_id"], r["gsis_id"]))
        if i is None:
            out.append(None)
            continue
        draws = rec[i] if r["market"] == "player_receptions" else yds[i]
        x = draws if r["market"] == "player_receptions" else np.rint(draws)
        over, push = (x > r["line"]).mean(), (x == r["line"]).mean()
        out.append(float(over / (1 - push)) if push < 1 else None)
    return cons.with_columns(pl.Series("p_model", out, dtype=pl.Float64))


def model_close(u, tg, cy, cfg, keys: pl.DataFrame) -> dict:
    """Simulated draws per season for the (game_id, gsis_id) keys, kickoff information."""
    feat = M.assemble(u, tg, cfg.half_life)
    tf = M.team_features(tg)
    out = {}
    for season in sorted(keys["season"].unique().to_list()):
        params, k = M.prepare_season(feat, tf, cy, season, cfg)
        rows = (params.join(keys.filter(pl.col("season") == season).select("game_id", "gsis_id").unique(),
                            on=["game_id", "gsis_id"])
                      .filter(pl.col("pos").is_in(M.EVAL_POS)).sort("game_id", "gsis_id"))
        if rows.is_empty():
            continue
        rec, yds = M.simulate(rows, k, cfg.model, N_SIMS, SEED + season)
        out[season] = (rows, rec, yds)
    return out


def model_bet(u, tg, cy, cfg, keys: pl.DataFrame, bet_lines: pl.DataFrame, home: dict) -> dict:
    """Draws per (season, week) using only information available 36h before kickoff."""
    out = {}
    for (season, week), wk in keys.group_by(["season", "week"], maintain_order=True):
        target = (pl.col("season") == season) & (pl.col("week") == week)
        week_games = u.filter(target).select("game_id", "kickoff_utc", "team", "opp", "is_home",
                                             "season", "week").unique()
        first_kick = week_games["kickoff_utc"].min()
        prior = u.filter(pl.col("played") & (pl.col("kickoff_utc") < first_kick))
        last = prior.group_by("team").agg(pl.col("kickoff_utc").max().alias("last_kick"))
        acts = (prior.join(last, on="team").filter(pl.col("kickoff_utc") == pl.col("last_kick"))
                     .select("gsis_id", "name", "pos", "team"))
        proj = (acts.join(week_games, on="team")
                    .with_columns(pl.lit(0).alias("targets"), pl.lit(0).alias("receptions"),
                                  pl.lit(0.0).alias("rec_yds"), pl.lit(False).alias("played"))
                    .select(u.columns).cast(dict(u.schema)))
        u_w = pl.concat([u.filter(~target), proj]).sort("gsis_id", "kickoff_utc", "team")

        bl = bet_lines.with_columns(pl.col("game_id").replace_strict(home, default=None).alias("home_team"))
        tg_w = (tg.join(bl, on="game_id", how="left")
                  .with_columns(
                      pl.when(target).then(
                          pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("bet_spread"))
                            .otherwise(-pl.col("bet_spread")))
                        .otherwise(pl.col("team_spread")).alias("team_spread"),
                      pl.when(target).then(pl.col("bet_total")).otherwise(pl.col("total_line")).alias("total_line"))
                  .drop("bet_spread", "bet_total", "home_team"))
        feat = M.assemble(u_w, tg_w, cfg.half_life)
        params, k = M.prepare_season(feat, M.team_features(tg_w), cy, season, cfg)
        rows = (params.filter(target & pl.col("pos").is_in(M.EVAL_POS) & pl.col("team_spread").is_not_null()
                              & pl.col("total_line").is_not_null())
                      .join(wk.select("game_id", "gsis_id").unique(), on=["game_id", "gsis_id"])
                      .sort("game_id", "gsis_id"))
        if rows.is_empty():
            continue
        rec, yds = M.simulate(rows, k, cfg.model, N_SIMS, SEED + season * 100 + week)
        out[(season, week)] = (rows, rec, yds)
    return out


# ---------------------------------------------------------------- statistics

def boot_mean(values: np.ndarray, games: np.ndarray, rng, n: int = BOOT) -> tuple[float, float, float]:
    if len(values) == 0:
        return (float("nan"),) * 3
    uniq, inv = np.unique(games, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=values), np.bincount(inv)
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats.append(sums[pick].sum() / cnts[pick].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def reliability(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [{"pred": float(p[idx == b].mean()), "obs": float(y[idx == b].mean()), "n": int((idx == b).sum())}
            for b in range(bins) if (idx == b).sum() >= 20]


def compare(df: pl.DataFrame, rng) -> dict:
    """Model vs market at close for one subset."""
    p_m, p_k = df["p_model"].to_numpy(), df["p_mult"].to_numpy()
    y, games = df["over"].to_numpy().astype(float), df["game_id"].to_numpy()
    ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    ll_m, ll_k = ll(p_m), ll(p_k)
    diff = boot_mean(ll_m - ll_k, games, rng)
    alt = {name: float(ll(df[col].to_numpy()).mean()) for name, col in (("power", "p_power"), ("shin", "p_shin"))}
    return {
        "rows": int(len(y)), "games": int(len(np.unique(games))), "over_rate": float(y.mean()),
        "log_loss": {"model": float(ll_m.mean()), "market": float(ll_k.mean()), "market_power": alt["power"],
                     "market_shin": alt["shin"], "diff_model_minus_market": diff[0], "diff_ci95": [diff[1], diff[2]]},
        "brier": {"model": float(((p_m - y) ** 2).mean()), "market": float(((p_k - y) ** 2).mean())},
        "ece": {"model": ece(p_m, y), "market": ece(p_k, y)},
        "reliability": {"model": reliability(p_m, y), "market": reliability(p_k, y)},
        "mean_abs_gap": float(np.abs(p_m - p_k).mean()),
    }


def blend_test(train: pl.DataFrame, test: pl.DataFrame, rng) -> dict:
    def X(df):
        return np.column_stack([np.ones(df.height), logit(df["p_mult"].to_numpy()), logit(df["p_model"].to_numpy())])
    y_tr, y_te = train["over"].to_numpy().astype(float), test["over"].to_numpy().astype(float)
    beta = fit_logistic(X(train), y_tr)
    games = train["game_id"].to_numpy()
    uniq, inv = np.unique(games, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    cs = []
    Xtr = X(train)
    for _ in range(BOOT):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])
        cs.append(fit_logistic(Xtr[pick], y_tr[pick])[2])
    p_te = 1 / (1 + np.exp(-X(test) @ beta))
    ll = lambda p, y: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    d = boot_mean(ll(p_te, y_te) - ll(test["p_mult"].to_numpy(), y_te), test["game_id"].to_numpy(), rng)
    lo, hi = np.percentile(cs, [2.5, 97.5])
    return {"train_rows": train.height, "test_rows": test.height,
            "coef": {"intercept": float(beta[0]), "market": float(beta[1]), "model": float(beta[2])},
            "model_coef_ci95": [float(lo), float(hi)], "model_adds_information": bool(lo > 0),
            "test_log_loss_blend_minus_market": d[0], "test_diff_ci95": [d[1], d[2]]}


def clv(bet: pl.DataFrame, close: pl.DataFrame, rng) -> dict:
    key = ["game_id", "gsis_id", "market"]
    j = bet.join(close.select(key + [pl.col("line").alias("line_close"), pl.col("p_mult").alias("p_close")]),
                 on=key, how="inner")
    j = j.with_columns((pl.col("p_model") - pl.col("p_mult")).alias("edge")) \
         .filter(pl.col("edge").abs() >= DISAGREE)
    if j.is_empty():
        return {"flagged": 0}
    side_over = j["edge"].to_numpy() > 0
    same = j["line"].to_numpy() == j["line_close"].to_numpy()
    games = j["game_id"].to_numpy()
    move = np.where(side_over, j["p_close"].to_numpy() - j["p_mult"].to_numpy(),
                    j["p_mult"].to_numpy() - j["p_close"].to_numpy())
    line_dir = np.sign(j["line_close"].to_numpy() - j["line"].to_numpy())
    agree = np.where(side_over, line_dir > 0, line_dir < 0).astype(float)
    # hypothetical flat 1-unit stake at the best bet-time price on the flagged side
    y = j["stat"].to_numpy().astype(float)
    L = j["line"].to_numpy().astype(float)
    win = np.where(side_over, y > L, y < L)
    push = y == L
    price = np.where(side_over, j["best_over"].to_numpy(), j["best_under"].to_numpy()).astype(float)
    profit = np.where(push, 0.0, np.where(win, price - 1, -1.0))
    a, b, c = boot_mean(move[same], games[same], rng)
    d, e, f = boot_mean(agree[~same], games[~same], rng)
    g, h, i = boot_mean(profit, games, rng)
    return {"flagged": int(j.height), "same_line": int(same.sum()), "line_moved": int((~same).sum()),
            "prob_move_toward_model": {"mean": a, "ci95": [b, c]},
            "line_moved_toward_model_share": {"mean": d, "ci95": [e, f]},
            "hypothetical_flat_return_per_unit": {"mean": g, "ci95": [h, i], "note": "secondary; noisy"}}


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    rng = np.random.default_rng(SEED)
    conn = db.connect()

    props = _q(conn, """
        SELECT p.*, g.season, g.week FROM prop_lines p JOIN games g USING (game_id)
        WHERE p.match_status = 'matched' AND g.game_type = 'REG'""")
    if props.is_empty():
        raise SystemExit("No matched odds in prop_lines. Run the backfill and proplab.odds.load first.")
    props = props.with_columns(pl.col("line").cast(pl.Float64), pl.col("over_price").cast(pl.Float64),
                               pl.col("under_price").cast(pl.Float64))
    u, tg, cy = data.load_universe(conn), data.load_team_games(conn), data.load_catch_yards(conn)
    home = dict(conn.execute("SELECT game_id, home_team FROM games WHERE season >= 2023").fetchall())
    blines = bet_game_lines(conn)
    conn.close()
    cfg = backtest.load_configs()["M1"]

    # outcomes: only for players active in that game (books void inactive players)
    act = u.select("game_id", "gsis_id", "pos", "played", "receptions", "rec_yds")
    cons = {}
    counts = {}
    for tag in ("close", "bet"):
        c = consensus(props.filter(pl.col("snapshot_tag") == tag)).join(
            props.select("game_id", "season", "week").unique(), on="game_id")
        c = c.join(act, on=["game_id", "gsis_id"], how="left")
        counts[tag] = {"consensus_rows": c.height,
                       "not_active_or_void": int(c["pos"].is_null().sum()),
                       "non_wr_te": int((c["pos"].is_not_null() & ~c["pos"].is_in(M.EVAL_POS)).sum())}
        c = c.filter(pl.col("pos").is_in(M.EVAL_POS) & pl.col("played"))
        c = c.with_columns(pl.when(pl.col("market") == "player_receptions").then(pl.col("receptions").cast(pl.Float64))
                             .otherwise(pl.col("rec_yds")).alias("stat"))
        cons[tag] = c

    close_all = cons["close"]  # before push filtering: CLV needs every closing price
    keys = pl.concat([cons[t].select("game_id", "gsis_id", "season", "week") for t in cons]).unique()
    close_sims = model_close(u, tg, cy, cfg, keys)
    close_parts = []
    for season, (rows, rec, yds) in close_sims.items():
        close_parts.append(attach_model(cons["close"].filter(pl.col("season") == season), rows, rec, yds))
    close = pl.concat(close_parts) if close_parts else cons["close"].head(0)
    counts["close"]["no_model_prediction"] = int(close["p_model"].is_null().sum())
    close = close.filter(pl.col("p_model").is_not_null() & (pl.col("stat") != pl.col("line")))
    close = close.with_columns((pl.col("stat") > pl.col("line")).cast(pl.Int8).alias("over"))

    bet_keys = cons["bet"].select("game_id", "gsis_id", "season", "week").unique()
    bet_sims = model_bet(u, tg, cy, cfg, bet_keys, blines, home) if not bet_keys.is_empty() else {}
    bet_parts = []
    for (season, week), (rows, rec, yds) in bet_sims.items():
        sub = cons["bet"].filter((pl.col("season") == season) & (pl.col("week") == week))
        bet_parts.append(attach_model(sub, rows, rec, yds))
    bet = pl.concat(bet_parts) if bet_parts else cons["bet"].with_columns(pl.lit(None, dtype=pl.Float64).alias("p_model")).head(0)
    counts["bet"]["no_model_prediction"] = int(cons["bet"].height - bet.filter(pl.col("p_model").is_not_null()).height)
    bet = bet.filter(pl.col("p_model").is_not_null())

    results: dict = {"config": cfg.label(), "counts": counts, "markets": {}}
    for mk, (label, _) in MARKETS.items():
        cm, bm = close.filter(pl.col("market") == mk), bet.filter(pl.col("market") == mk)
        r: dict = {"close": {}, "clv": {}}
        pooled = cm.filter(pl.col("season").is_in(POOLED))
        if pooled.height:
            r["close"]["2023-2025"] = compare(pooled, rng)
        for s in sorted(cm["season"].unique().to_list()):
            r["close"][str(s)] = compare(cm.filter(pl.col("season") == s), rng)
        tr = cm.filter(pl.col("season").is_in([2023, 2024]))
        te = cm.filter(pl.col("season") == 2025)
        if tr.height > 50 and te.height > 50:
            r["blend"] = blend_test(tr, te, rng)
        for name, seasons in (("2023-2025", POOLED), ("2026", (2026,))):
            sub = bm.filter(pl.col("season").is_in(seasons))
            if sub.height:
                r["clv"][name] = clv(sub, close_all.filter(pl.col("market") == mk), rng)
        results["markets"][label] = r

    results["ev_unlocked"] = {lbl: bool(results["markets"][lbl].get("blend", {}).get("model_adds_information", False))
                              for lbl in results["markets"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    summarize(results)
    return results


def summarize(r: dict) -> None:
    print(json.dumps(r["counts"], indent=1))
    for lbl, m in r["markets"].items():
        for k, v in m["close"].items():
            ll = v["log_loss"]
            print(f"{lbl:10s} {k:9s} n={v['rows']:5d}  LL model {ll['model']:.4f}  market {ll['market']:.4f}  "
                  f"diff {ll['diff_model_minus_market']:+.4f} [{ll['diff_ci95'][0]:+.4f}, {ll['diff_ci95'][1]:+.4f}]  "
                  f"ECE {v['ece']['model']:.3f}/{v['ece']['market']:.3f}")
        if "blend" in m:
            b = m["blend"]
            print(f"{lbl:10s} blend model coef {b['coef']['model']:+.3f} CI {b['model_coef_ci95']}  "
                  f"2025 blend-market {b['test_log_loss_blend_minus_market']:+.4f}")
        for k, v in m["clv"].items():
            print(f"{lbl:10s} CLV {k}: {v}")
    print("EV unlocked:", r["ev_unlocked"])


if __name__ == "__main__":
    main()
