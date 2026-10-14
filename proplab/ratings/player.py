"""3b-2: player efficiency ratings (empirical Bayes, opponent-adjusted), weekly and point in time.

QB: EPA per dropback (passes, sacks, scrambles). RB: EPA per designed carry. WR/TE: EPA per target.
Each play's EPA minus the defense's unit rating as of that week (team_ratings, fit on earlier
games only). For a player's game g, the rating uses only his earlier games:

    xbar = sum_k w_k * adjEPA_k / sum_k w_k * n_k,   w_k = 0.5 ** (games_since_k / h)
    posterior mean = mu + (xbar - mu) * n_eff / (n_eff + sigma2 / tau2)
    posterior sd   = sqrt(1 / (n_eff / sigma2 + 1 / tau2))

mu, sigma2 (per-play variance) and tau2 (true between-player variance) come from the three
seasons before the one being rated.

    python -m proplab.ratings.player tune | test | publish
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.p1.data import decay_for, ewm_prior_sums
from proplab.ratings.team import CANON

ROOT = Path(__file__).resolve().parents[2]
POS = ("QB", "RB", "WR", "TE")
HALF_LIVES = (8.0, 16.0, 32.0)
QUALIFY = {"QB": 50, "RB": 30, "WR": 20, "TE": 20}   # opportunities in the last 8 games
NEXT_GAMES = 4
Z80 = 1.2816


def _q(conn, sql):
    cur = conn.execute(sql)
    return pl.DataFrame(cur.fetchall(), schema=[d.name for d in cur.description], orient="row", infer_schema_length=None)


def player_games(conn) -> pl.DataFrame:
    """One row per player-game: opportunities, sum of raw EPA, sum of opponent-adjusted EPA."""
    plays = _q(conn, """
        SELECT p.game_id, g.season, g.week, g.kickoff_utc, p.defteam, p.epa::float AS epa, p.qb_dropback, p.rush,
               p.passer_player_id, p.rusher_player_id, p.receiver_player_id
        FROM plays p JOIN games g USING (game_id)
        WHERE g.game_type = 'REG' AND p.play_type IN ('pass', 'run') AND p.down IS NOT NULL AND p.epa IS NOT NULL""")
    pos = _q(conn, "SELECT gsis_id, position AS pos, display_name AS name FROM players")
    tr = _q(conn, "SELECT season, week, team, unit, def FROM team_ratings")
    plays = plays.with_columns(pl.col("defteam").replace(CANON))
    qb = plays.filter(pl.col("qb_dropback")).with_columns(
        pl.coalesce("passer_player_id", "rusher_player_id").alias("gsis_id"), pl.lit("pass").alias("unit"))
    rb = plays.filter(pl.col("rush") & ~pl.col("qb_dropback")).with_columns(
        pl.col("rusher_player_id").alias("gsis_id"), pl.lit("run").alias("unit"))
    rc = plays.filter(pl.col("qb_dropback") & pl.col("receiver_player_id").is_not_null()).with_columns(
        pl.col("receiver_player_id").alias("gsis_id"), pl.lit("pass").alias("unit"))
    parts = []
    for df, want in ((qb, ("QB",)), (rb, ("RB",)), (rc, ("WR", "TE"))):
        d = (df.join(pos, on="gsis_id").filter(pl.col("pos").is_in(want))
               .join(tr.rename({"team": "defteam"}), on=["season", "week", "defteam", "unit"], how="left")
               .with_columns((pl.col("epa") - pl.col("def").fill_null(0.0)).alias("adj")))
        parts.append(d.group_by("gsis_id", "name", "pos", "game_id", "season", "week", "kickoff_utc")
                      .agg(pl.len().alias("n"), pl.col("epa").sum().alias("raw_sum"), pl.col("adj").sum().alias("adj_sum"),
                           (pl.col("adj") ** 2).sum().alias("adj_sq")))
    return pl.concat(parts).sort("gsis_id", "kickoff_utc")


def priors(pg: pl.DataFrame, season: int) -> dict:
    """Per position: mu, sigma2 (per play), tau2 (between players), from the 3 prior seasons."""
    h = pg.filter(pl.col("season").is_between(season - 3, season - 1))
    out = {}
    for p in POS:
        d = h.filter(pl.col("pos") == p)
        if d.is_empty():  # callers may pass a single position
            continue
        n, s = d["n"].sum(), d["adj_sum"].sum()
        mu = s / n
        ps = (d.group_by("gsis_id", "season").agg(pl.col("n").sum(), pl.col("adj_sum").sum(), pl.col("adj_sq").sum())
               .filter(pl.col("n") >= QUALIFY[p]))
        within = float(((ps["adj_sq"] - ps["adj_sum"] ** 2 / ps["n"]).sum()) / (ps["n"] - 1).sum())
        xbar = (ps["adj_sum"] / ps["n"]).to_numpy()
        w = ps["n"].to_numpy().astype(float)
        between = float(np.average((xbar - np.average(xbar, weights=w)) ** 2, weights=w) - within * (1 / w).mean())
        out[p] = {"mu": float(mu), "sigma2": within, "tau2": max(between, 1e-5)}
    return out


def with_prior_sums(pg: pl.DataFrame, h: float) -> pl.DataFrame:
    keys = pg["gsis_id"].to_numpy()
    d = decay_for(h)
    out = pg.with_columns(
        pl.Series("W_n", ewm_prior_sums(keys, pg["n"].to_numpy().astype(float), d)),
        pl.Series("W_adj", ewm_prior_sums(keys, pg["adj_sum"].to_numpy(), d)),
        pl.Series("W_raw", ewm_prior_sums(keys, pg["raw_sum"].to_numpy(), d)),
        pl.col("n").cast(pl.Float64).shift(1).over("gsis_id").alias("_prev_n"),
    )
    # unweighted opportunities in the previous 8 games, for "qualified"
    return out.with_columns(
        pl.col("_prev_n").rolling_sum(8, min_samples=1).over("gsis_id").fill_null(0).alias("N8")).drop("_prev_n")


def next_target(pg: pl.DataFrame) -> pl.DataFrame:
    """Opponent-adjusted EPA per opportunity over the player's next up-to-4 games."""
    s, n = pl.col("adj_sum"), pl.col("n")
    fut_s = sum(s.shift(-k).over("gsis_id").fill_null(0) for k in range(NEXT_GAMES))
    fut_n = sum(n.shift(-k).over("gsis_id").fill_null(0) for k in range(NEXT_GAMES))
    return pg.with_columns(fut_s.alias("T_sum"), fut_n.alias("T_n"))


def rate(df: pl.DataFrame, pri: dict) -> pl.DataFrame:
    mu = pl.col("pos").replace_strict({p: v["mu"] for p, v in pri.items()}, return_dtype=pl.Float64)
    s2 = pl.col("pos").replace_strict({p: v["sigma2"] for p, v in pri.items()}, return_dtype=pl.Float64)
    t2 = pl.col("pos").replace_strict({p: v["tau2"] for p, v in pri.items()}, return_dtype=pl.Float64)
    xbar = pl.when(pl.col("W_n") > 0).then(pl.col("W_adj") / pl.col("W_n")).otherwise(mu)
    return df.with_columns(
        (mu + (xbar - mu) * pl.col("W_n") / (pl.col("W_n") + s2 / t2)).alias("post_mean"),
        (1 / (pl.col("W_n") / s2 + 1 / t2)).sqrt().alias("post_sd"),
        pl.when(pl.col("W_n") > 0).then(pl.col("W_raw") / pl.col("W_n")).otherwise(None).alias("raw_mean"),
    )


def scored(pg: pl.DataFrame, h: float, seasons: list[int]) -> pl.DataFrame:
    """Rows to score: player-games in `seasons` with prior opportunities and >=1 future opportunity."""
    d = next_target(with_prior_sums(pg, h))
    out = []
    for s in seasons:
        sub = d.filter((pl.col("season") == s) & (pl.col("W_n") > 0) & (pl.col("T_n") > 0))
        out.append(rate(sub, priors(pg, s)))
    df = pl.concat(out)
    tgt = pl.col("T_sum") / pl.col("T_n")
    return df.with_columns(((pl.col("post_mean") - tgt) ** 2).alias("se_eb"),
                           ((pl.col("raw_mean") - tgt) ** 2).alias("se_raw"))


def wmse(df, col):
    return float((df[col] * df["T_n"]).sum() / df["T_n"].sum())


def boot(df: pl.DataFrame, rng, n=1000):
    g = (df.with_columns(((pl.col("se_eb") - pl.col("se_raw")) * pl.col("T_n")).alias("d"))
           .group_by("gsis_id").agg(pl.col("d").sum(), pl.col("T_n").sum()))
    d, w = g["d"].to_numpy(), g["T_n"].to_numpy()
    stats = [d[p].sum() / w[p].sum() for p in (rng.integers(0, len(d), len(d)) for _ in range(n))]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(d.sum() / w.sum()), float(lo), float(hi)


def tune() -> None:
    pg = player_games(db.connect())
    out = {}
    for p in POS:
        sub = pg.filter(pl.col("pos") == p)
        eb = {h: wmse(scored(sub, h, list(range(2018, 2025))), "se_eb") for h in HALF_LIVES}
        raw = {h: wmse(scored(sub, h, list(range(2018, 2025))), "se_raw") for h in HALF_LIVES}
        out[p] = {"h": min(eb, key=eb.get), "h_raw": min(raw, key=raw.get), "grid_eb": eb, "grid_raw": raw}
        print(p, out[p], flush=True)
    (ROOT / "phase3" / "ratings_player_tuning.json").write_text(json.dumps(out, indent=2))


def test() -> None:
    pg = player_games(db.connect())
    ch = json.loads((ROOT / "phase3" / "ratings_player_tuning.json").read_text())
    rng = np.random.default_rng(26)
    res = {}
    for p in POS:
        sub = pg.filter(pl.col("pos") == p)
        res[p] = {}
        for name, seasons in (("tuning_2018_2024", list(range(2018, 2025))), ("test_2025", [2025]), ("live_2026", [2026])):
            eb = scored(sub, ch[p]["h"], seasons)
            raw = scored(sub, ch[p]["h_raw"], seasons).select("gsis_id", "game_id", "se_raw")
            j = eb.drop("se_raw").join(raw, on=["gsis_id", "game_id"])
            d, lo, hi = boot(j, rng)
            res[p][name] = {"rows": j.height, "players": j["gsis_id"].n_unique(), "wmse_eb": wmse(j, "se_eb"),
                            "wmse_raw": wmse(j, "se_raw"), "diff": d, "ci95": [lo, hi], "pct": d / wmse(j, "se_raw")}
            print(p, name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[p][name].items()}, flush=True)
    res["gate"] = {p: bool(res[p]["test_2025"]["ci95"][1] < 0) for p in POS}
    res["gate"]["passed"] = all(res["gate"][p] for p in POS)
    (ROOT / "phase3" / "ratings_player_results.json").write_text(json.dumps(res, indent=2))
    print("GATE", res["gate"])


def current(conn, season: int = 2026) -> pl.DataFrame:
    """Ratings as of the next unplayed week, for every player with prior opportunities."""
    pg = player_games(conn)
    ch = json.loads((ROOT / "phase3" / "ratings_player_tuning.json").read_text())
    (week,) = conn.execute("""SELECT min(week) FROM games g WHERE season = %s AND game_type = 'REG'
                              AND NOT EXISTS (SELECT 1 FROM game_results r WHERE r.game_id = g.game_id)""",
                           (season,)).fetchone()
    team = _q(conn, f"""SELECT DISTINCT ON (gsis_id) gsis_id, team FROM roster_status
                        WHERE season = {season} AND week < {week} ORDER BY gsis_id, week DESC""")
    out = []
    pri = priors(pg, season)
    for p in POS:
        sub = pg.filter(pl.col("pos") == p)
        # a dummy "next game" row per player so the prior sums cover all his played games
        nxt = (sub.group_by("gsis_id", "name", "pos").agg(pl.col("kickoff_utc").max())
                  .with_columns(pl.lit(None, dtype=pl.Utf8).alias("game_id"), pl.lit(season).alias("season"),
                                pl.lit(week).alias("week"), pl.col("kickoff_utc") + pl.duration(days=7),
                                pl.lit(0, dtype=pl.UInt32).alias("n"), pl.lit(0.0).alias("raw_sum"),
                                pl.lit(0.0).alias("adj_sum"), pl.lit(0.0).alias("adj_sq"))
                  .select(sub.columns).cast(dict(sub.schema)))
        d = with_prior_sums(pl.concat([sub, nxt]).sort("gsis_id", "kickoff_utc"), ch[p]["h"])
        d = rate(d.filter(pl.col("game_id").is_null()), pri)
        d = d.with_columns((pl.col("post_mean") - Z80 * pl.col("post_sd")).alias("lo80"),
                           (pl.col("post_mean") + Z80 * pl.col("post_sd")).alias("hi80"))
        q = d.filter(pl.col("N8") >= QUALIFY[p])
        ref = np.sort(q["post_mean"].to_numpy())
        pct = lambda v: (np.searchsorted(ref, v.to_numpy(), side="right") / max(len(ref), 1) * 100)
        d = d.with_columns(pl.Series("rating", pct(d["post_mean"])), pl.Series("rating_lo", pct(d["lo80"])),
                           pl.Series("rating_hi", pct(d["hi80"])), (pl.col("N8") >= QUALIFY[p]).alias("qualified"))
        out.append(d.join(team, on="gsis_id", how="left"))
    return pl.concat(out).select("gsis_id", "name", "pos", "team", "season", "week", "post_mean", "post_sd", "lo80", "hi80",
                                 "W_n", "N8", "qualified", "rating", "rating_lo", "rating_hi", "raw_mean")


def publish() -> None:
    conn = db.connect()
    cur = current(conn)
    cur.write_json(ROOT / "phase3" / "player_ratings_current.json")
    print(cur.filter(pl.col("qualified")).group_by("pos").len())
    for p in POS:
        print(p, cur.filter(pl.col("qualified") & (pl.col("pos") == p)).sort("rating", descending=True)
              .select("name", "team", "rating", "rating_lo", "rating_hi", "W_n").head(5).rows())


if __name__ == "__main__":
    {"tune": tune, "test": test, "publish": publish}[sys.argv[1]]()
