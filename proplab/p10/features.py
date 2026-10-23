"""Phase 10 features (see phase10/PREREGISTRATION.md), all from EARLIER games only.

Per (gsis_id, game_id) of the engine universe: role_r (Phase 9A), qb_cr, qb_ypc (quarterback change, per team-game),
ay_ypc (yards per catch implied by the receiver's depth of target).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from proplab.p1.data import decay_for, ewm_prior_sums
from proplab.p3.matchup import _q
from proplab.p9.roles import role_signal

QB_HL, TEAM_HL, AY_HL = 16.0, 4.0, 8.0
PSEUDO_ATT, PSEUDO_CMP, PSEUDO_TGT = 150.0, 90.0, 20.0


def _prior_seasons(df: pl.DataFrame, season_col: str = "season") -> list[int]:
    return sorted(df[season_col].unique().to_list())


def qb_change(u: pl.DataFrame, out_qbs: pl.DataFrame | None = None) -> pl.DataFrame:
    """(game_id, team) -> qb_cr, qb_ypc. out_qbs (gsis_id, game_id) are skipped when picking the projected starter."""
    q = u.filter(pl.col("pos") == "QB").sort("gsis_id", "kickoff_utc", "team")
    k = q["gsis_id"].to_numpy()
    pl_ = q["played"].to_numpy().astype(float)
    a, c, y = (q[x].to_numpy().astype(float) * pl_ for x in ("attempts", "completions", "pass_yds"))
    d16, d4 = decay_for(QB_HL), decay_for(TEAM_HL)
    q = q.with_columns(pl.Series("Sa", ewm_prior_sums(k, a, d16)), pl.Series("Sc", ewm_prior_sums(k, c, d16)),
                       pl.Series("Sy", ewm_prior_sums(k, y, d16)), pl.Series("A4", ewm_prior_sums(k, a, d4)),
                       pl.Series("N4", ewm_prior_sums(k, pl_, d4)))
    # league QB averages from the three prior seasons
    per = q.filter(pl.col("played")).group_by("season").agg(pl.col("attempts").sum(), pl.col("completions").sum(), pl.col("pass_yds").sum())
    lg = []
    for s in _prior_seasons(q):
        h = per.filter(pl.col("season").is_between(s - 3, s - 1))
        if h.height:
            lg.append({"season": s, "L_cr": h["completions"].sum() / h["attempts"].sum(), "L_ypc": h["pass_yds"].sum() / h["completions"].sum()})
    q = q.join(pl.DataFrame(lg), on="season", how="left").with_columns(
        ((pl.col("Sc") + PSEUDO_ATT * pl.col("L_cr")) / (pl.col("Sa") + PSEUDO_ATT)).alias("Q_cr"),
        ((pl.col("Sy") + PSEUDO_CMP * pl.col("L_ypc")) / (pl.col("Sc") + PSEUDO_CMP)).alias("Q_ypc"),
        (pl.col("A4") / pl.col("N4")).fill_nan(0.0).fill_null(0.0).alias("att_pg"))
    # projected starter: active QB with the highest recent attempts per game (skipping listed Out/Doubtful QBs)
    cand = q
    if out_qbs is not None and out_qbs.height:
        cand = q.join(out_qbs, on=["gsis_id", "game_id"], how="anti")
    proj = cand.sort("att_pg", "gsis_id", descending=[True, False]).unique(subset=["game_id", "team"], keep="first") \
               .select("game_id", "team", pl.col("Q_cr").alias("P_cr"), pl.col("Q_ypc").alias("P_ypc"))
    # actual starter (most attempts) of each played game, and the team's recent context over previous games
    act = q.filter(pl.col("played")).sort("attempts", "gsis_id", descending=[True, False]).unique(subset=["game_id", "team"], keep="first") \
           .select("game_id", "team", pl.col("Q_cr").alias("A_cr"), pl.col("Q_ypc").alias("A_ypc"))
    tg = u.select("game_id", "team", "kickoff_utc", "played").unique(subset=["game_id", "team"]).sort("team", "kickoff_utc") \
          .join(act, on=["game_id", "team"], how="left")
    tk = tg["team"].to_numpy()
    w = (tg["played"].to_numpy() & tg["A_cr"].is_not_null().to_numpy()).astype(float)
    sw = ewm_prior_sums(tk, w, d4)
    ccr = ewm_prior_sums(tk, tg["A_cr"].fill_null(0).to_numpy() * w, d4)
    cyp = ewm_prior_sums(tk, tg["A_ypc"].fill_null(0).to_numpy() * w, d4)
    tg = tg.with_columns(pl.Series("C_cr", np.divide(ccr, sw, out=np.full_like(ccr, np.nan), where=sw > 0)),
                         pl.Series("C_ypc", np.divide(cyp, sw, out=np.full_like(cyp, np.nan), where=sw > 0)))
    out = tg.join(proj, on=["game_id", "team"], how="left").with_columns(
        (pl.col("P_cr") / pl.col("C_cr")).alias("qb_cr"), (pl.col("P_ypc") / pl.col("C_ypc")).alias("qb_ypc"))
    return out.select("game_id", "team", *[pl.col(c).fill_nan(1.0).fill_null(1.0) for c in ("qb_cr", "qb_ypc")])


def air_yards(conn, u: pl.DataFrame) -> pl.DataFrame:
    """(gsis_id, game_id) -> ay_ypc: yards per catch implied by the receiver's earlier depth of target."""
    pg = _q(conn, """SELECT gsis_id, game_id, coalesce(receiving_air_yards, 0)::float AS ay, coalesce(targets, 0) AS t,
                            coalesce(receptions, 0) AS r, coalesce(receiving_yards, 0)::float AS y FROM player_game""")
    r = u.filter(pl.col("pos").is_in(["WR", "TE", "RB"])).select("gsis_id", "game_id", "season", "pos", "kickoff_utc", "team", "played") \
         .join(pg, on=["gsis_id", "game_id"], how="left").fill_null(0).sort("gsis_id", "kickoff_utc", "team")
    k = r["gsis_id"].to_numpy()
    p = r["played"].to_numpy().astype(float)
    d = decay_for(AY_HL)
    r = r.with_columns(pl.Series("S_ay", ewm_prior_sums(k, r["ay"].to_numpy() * p, d)), pl.Series("S_t", ewm_prior_sums(k, r["t"].to_numpy() * p, d)))
    # league depth-of-target by position and the yards-per-catch fit, from the three prior seasons (player-seasons)
    ps = r.filter(pl.col("played")).group_by("gsis_id", "season", "pos").agg(pl.col("ay").sum(), pl.col("t").sum(), pl.col("r").sum(), pl.col("y").sum()) \
          .filter((pl.col("t") >= 20) & (pl.col("r") >= 10))
    fits = []
    for s in _prior_seasons(r):
        h = ps.filter(pl.col("season").is_between(s - 3, s - 1))
        if h.height < 50:
            continue
        x, yv, wt = (h["ay"] / h["t"]).to_numpy(), (h["y"] / h["r"]).to_numpy(), h["r"].to_numpy().astype(float)
        X = np.column_stack([np.ones_like(x), x]) * np.sqrt(wt)[:, None]
        b, *_ = np.linalg.lstsq(X, yv * np.sqrt(wt), rcond=None)
        for pos in ("WR", "TE", "RB"):
            hp = h.filter(pl.col("pos") == pos)
            fits.append({"season": s, "pos": pos, "b0": float(b[0]), "b1": float(b[1]),
                         "L_adot": float(hp["ay"].sum() / max(hp["t"].sum(), 1))})
    r = r.join(pl.DataFrame(fits), on=["season", "pos"], how="left").with_columns(
        ((pl.col("S_ay") + PSEUDO_TGT * pl.col("L_adot")) / (pl.col("S_t") + PSEUDO_TGT)).alias("adot"))
    r = r.with_columns((pl.col("b0") + pl.col("b1") * pl.col("adot")).clip(3.0, 30.0).alias("ay_ypc"))
    return r.select("gsis_id", "game_id", "ay_ypc")


def build(conn, u: pl.DataFrame, out_qbs: pl.DataFrame | None = None) -> pl.DataFrame:
    """All Phase 10 columns for every (gsis_id, game_id) of the engine universe u (QB/RB/WR/TE)."""
    base = u.select("gsis_id", "game_id", "team")
    return (base.join(role_signal(conn, u), on=["gsis_id", "game_id"], how="left")
                .join(qb_change(u, out_qbs), on=["game_id", "team"], how="left")
                .join(air_yards(conn, u), on=["gsis_id", "game_id"], how="left")
                .with_columns(pl.col("role_r").fill_null(1.0), pl.col("qb_cr").fill_null(1.0), pl.col("qb_ypc").fill_null(1.0))
                .select("gsis_id", "game_id", "role_r", "qb_cr", "qb_ypc", "ay_ypc"))
