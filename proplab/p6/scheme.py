"""Phase 6: defensive scheme features from FTN charting, all from EARLIER games only.

For every (player, game) row of the Phase 3a universe:
  blitz_comp, blitz_ypc, blitz_td, blitz_int :
        [(1-h_j) + h_j*s_i] / [(1-hbar) + hbar*s_i], h_j = defense blitz rate on dropbacks,
        s_i = QB's blitzed / not-blitzed ratio for that rate (shrunk toward the league ratio)
  box_ypc : same form, h_j = defense's 8+-in-the-box rate on designed runs, s_i = RB's ypc heavy / light
  pa_ypc  : [p_i R_j + (1-p_i) Q_j] / [pbar R_j + (1-pbar) Q_j], p_i = QB play-action share,
            R_j / Q_j = defense yards per attempt allowed on play action / not, relative to league

Defense tendencies: half-life 16 of the defense's games, shrunk toward league. League rates come from
the prior seasons only (2023 -> 2022; later -> three prior seasons). Rows with no history get 1.0.
See phase6/PREREGISTRATION.md.
"""
from __future__ import annotations

import polars as pl

from proplab.p1.data import decay_for, ewm_prior_sums
from proplab.p3.matchup import _def_prior, _league, _q

HEAVY_BOX = 8
DEF_PSEUDO = 150.0      # defense dropbacks / runs for blitz and box rates
DEF_PSEUDO_PA = 100.0   # defense attempts in each of play action / not
QB_PSEUDO = 40.0        # QB blitzed attempts; RB heavy-box carries
QB_PSEUDO_RARE = 200.0  # QB blitzed attempts for TD / INT ratios
PA_PSEUDO = 100.0       # QB attempts for his play-action share
MIX_HL = 16


def load_plays(conn) -> pl.DataFrame:
    return _q(conn, """
        SELECT p.game_id, g.season, g.kickoff_utc, p.posteam, p.defteam AS defense, p.qb_dropback, p.rush, p.sack,
               p.complete_pass, p.interception, p.touchdown, p.yards_gained::float AS yds,
               p.passer_player_id, p.rusher_player_id,
               f.n_blitzers, f.n_defense_box, f.is_play_action
        FROM plays p JOIN games g USING (game_id)
        JOIN ftn_charting f ON f.game_id = p.game_id AND f.play_id = p.play_id
        WHERE g.game_type = 'REG' AND p.play_type IN ('pass', 'run') AND p.down IS NOT NULL""")


def _player_prior(u: pl.DataFrame, per_game: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Each player's EWMA sums over his PREVIOUS games in the universe (prior-only)."""
    j = u.select("gsis_id", "game_id", "kickoff_utc", "team", "season").join(per_game, on=["gsis_id", "game_id"], how="left") \
         .fill_null(0).sort("gsis_id", "kickoff_utc", "team")
    keys = j["gsis_id"].to_numpy()
    d = decay_for(MIX_HL)
    return j.with_columns(*[pl.Series(f"Q_{c}", ewm_prior_sums(keys, j[c].to_numpy().astype(float), d)) for c in cols]) \
            .select("gsis_id", "game_id", "season", *[f"Q_{c}" for c in cols])


def _shrunk_ratio(num_a, den_a, num_b, den_b, la, lb, k):
    ra = (pl.col(num_a) + k * pl.col(la)) / (pl.col(den_a) + k)
    rb = (pl.col(num_b) + k * pl.col(lb)) / (pl.col(den_b) + k)
    return ra / rb


def build(conn, u: pl.DataFrame, tg: pl.DataFrame, return_tendencies: bool = False, plays: pl.DataFrame | None = None):
    plays = load_plays(conn) if plays is None else plays
    games = tg.select("game_id", pl.col("opp").alias("defense"), "kickoff_utc", "played", "season")
    base = u.select("gsis_id", "game_id", "season", "pos", "opp")

    # ---------------- blitz
    db_ = plays.filter(pl.col("qb_dropback")).with_columns((pl.col("n_blitzers").fill_null(0) > 0).cast(pl.Int32).alias("bl"))
    att = db_.filter(~pl.col("sack") & ~pl.col("rush") & pl.col("complete_pass").is_not_null()).with_columns(
        pl.col("complete_pass").cast(pl.Int32).alias("c"),
        pl.when(pl.col("complete_pass")).then(pl.col("yds")).otherwise(0.0).alias("cy"),
        pl.col("interception").cast(pl.Int32).alias("i"),
        (pl.col("touchdown") & pl.col("complete_pass")).cast(pl.Int32).alias("td"))
    lgb = _league(db_, [], {"d": pl.len(), "bl": pl.col("bl").sum()}).with_columns((pl.col("bl") / pl.col("d")).alias("lg_bl"))
    bg = db_.group_by("game_id", "defense").agg(pl.len().alias("d"), pl.col("bl").sum())
    bprior = _def_prior(games, bg, [], ["d", "bl"]).join(games.select("game_id", "defense", "season"), on=["game_id", "defense"]) \
        .join(lgb.select("season", "lg_bl"), on="season", how="left") \
        .with_columns(((pl.col("P_bl") + DEF_PSEUDO * pl.col("lg_bl")) / (pl.col("P_d") + DEF_PSEUDO)).alias("h_def"))
    b, nb = pl.col("bl"), 1 - pl.col("bl")
    stats = {"a": pl.lit(1), "c": pl.col("c"), "cy": pl.col("cy"), "i": pl.col("i"), "td": pl.col("td")}
    qg = att.rename({"passer_player_id": "gsis_id"}).filter(pl.col("gsis_id").is_not_null()).group_by("gsis_id", "game_id").agg(
        *[(v * b).sum().alias(f"{k}_b") for k, v in stats.items()], *[(v * nb).sum().alias(f"{k}_n") for k, v in stats.items()])
    lr = _league(att, [], {**{f"{k}_b": (v * b).sum() for k, v in stats.items()}, **{f"{k}_n": (v * nb).sum() for k, v in stats.items()}})
    lr = lr.select("season",
                   (pl.col("c_b") / pl.col("a_b")).alias("L_cr_b"), (pl.col("c_n") / pl.col("a_n")).alias("L_cr_n"),
                   (pl.col("cy_b") / pl.col("c_b")).alias("L_ypc_b"), (pl.col("cy_n") / pl.col("c_n")).alias("L_ypc_n"),
                   (pl.col("i_b") / pl.col("a_b")).alias("L_ir_b"), (pl.col("i_n") / pl.col("a_n")).alias("L_ir_n"),
                   (pl.col("td_b") / pl.col("a_b")).alias("L_tr_b"), (pl.col("td_n") / pl.col("a_n")).alias("L_tr_n"))
    qcols = [f"{k}_{s}" for k in stats for s in ("b", "n")]
    j = _player_prior(u, qg, qcols).join(lr, on="season", how="left").with_columns(
        _shrunk_ratio("Q_c_b", "Q_a_b", "Q_c_n", "Q_a_n", "L_cr_b", "L_cr_n", QB_PSEUDO).alias("s_comp"),
        _shrunk_ratio("Q_cy_b", "Q_c_b", "Q_cy_n", "Q_c_n", "L_ypc_b", "L_ypc_n", QB_PSEUDO).alias("s_ypc"),
        _shrunk_ratio("Q_i_b", "Q_a_b", "Q_i_n", "Q_a_n", "L_ir_b", "L_ir_n", QB_PSEUDO_RARE).alias("s_int"),
        _shrunk_ratio("Q_td_b", "Q_a_b", "Q_td_n", "Q_a_n", "L_tr_b", "L_tr_n", QB_PSEUDO_RARE).alias("s_td"))
    rows = base.join(j.select("gsis_id", "game_id", "s_comp", "s_ypc", "s_int", "s_td"), on=["gsis_id", "game_id"], how="left") \
               .join(bprior.select("game_id", "defense", "h_def", "lg_bl"), left_on=["game_id", "opp"], right_on=["game_id", "defense"], how="left")
    h, hb = pl.col("h_def"), pl.col("lg_bl")
    mixf = lambda s: ((1 - h) + h * pl.col(s)) / ((1 - hb) + hb * pl.col(s))
    rows = rows.with_columns(*[mixf(s).alias(f"blitz_{s[2:]}") for s in ("s_comp", "s_ypc", "s_int", "s_td")])
    out = rows.select("gsis_id", "game_id", "blitz_comp", "blitz_ypc", "blitz_int", "blitz_td")

    # ---------------- box (designed runs)
    runs = plays.filter(pl.col("rush") & ~pl.col("qb_dropback") & pl.col("yds").is_not_null() & pl.col("n_defense_box").is_not_null()) \
                .with_columns((pl.col("n_defense_box") >= HEAVY_BOX).cast(pl.Int32).alias("hv"))
    lgx = _league(runs, [], {"n": pl.len(), "hv": pl.col("hv").sum(),
                             "y_h": (pl.col("yds") * pl.col("hv")).sum(), "y_l": (pl.col("yds") * (1 - pl.col("hv"))).sum()}) \
        .with_columns((pl.col("hv") / pl.col("n")).alias("lg_hv"), (pl.col("y_h") / pl.col("hv")).alias("L_h"),
                      (pl.col("y_l") / (pl.col("n") - pl.col("hv"))).alias("L_l"))
    xg = runs.group_by("game_id", "defense").agg(pl.len().alias("n"), pl.col("hv").sum())
    xprior = _def_prior(games, xg, [], ["n", "hv"]).join(games.select("game_id", "defense", "season"), on=["game_id", "defense"]) \
        .join(lgx.select("season", "lg_hv"), on="season", how="left") \
        .with_columns(((pl.col("P_hv") + DEF_PSEUDO * pl.col("lg_hv")) / (pl.col("P_n") + DEF_PSEUDO)).alias("h_box"))
    rg = runs.rename({"rusher_player_id": "gsis_id"}).filter(pl.col("gsis_id").is_not_null()).group_by("gsis_id", "game_id").agg(
        pl.col("hv").sum().alias("n_h"), (1 - pl.col("hv")).sum().alias("n_l"),
        (pl.col("yds") * pl.col("hv")).sum().alias("y_h"), (pl.col("yds") * (1 - pl.col("hv"))).sum().alias("y_l"))
    j = _player_prior(u, rg, ["n_h", "n_l", "y_h", "y_l"]).join(lgx.select("season", "L_h", "L_l"), on="season", how="left") \
        .with_columns(_shrunk_ratio("Q_y_h", "Q_n_h", "Q_y_l", "Q_n_l", "L_h", "L_l", QB_PSEUDO).alias("s_box"))
    rows = base.join(j.select("gsis_id", "game_id", "s_box"), on=["gsis_id", "game_id"], how="left") \
               .join(xprior.select("game_id", "defense", "h_box", "lg_hv"), left_on=["game_id", "opp"], right_on=["game_id", "defense"], how="left")
    h, hb, s = pl.col("h_box"), pl.col("lg_hv"), pl.col("s_box")
    rows = rows.with_columns((((1 - h) + h * s) / ((1 - hb) + hb * s)).alias("box_ypc"))
    out = out.join(rows.select("gsis_id", "game_id", "box_ypc"), on=["gsis_id", "game_id"], how="left")

    # ---------------- play action
    pa = att.filter(pl.col("is_play_action").is_not_null()).with_columns(pl.col("is_play_action").cast(pl.Int32).alias("pa"))
    lgp = _league(pa, [], {"a": pl.len(), "pa": pl.col("pa").sum(), "y_pa": (pl.col("cy") * pl.col("pa")).sum(),
                           "y_np": (pl.col("cy") * (1 - pl.col("pa"))).sum()}) \
        .with_columns((pl.col("pa") / pl.col("a")).alias("lg_pa"), (pl.col("y_pa") / pl.col("pa")).alias("L_ypa_pa"),
                      (pl.col("y_np") / (pl.col("a") - pl.col("pa"))).alias("L_ypa_np"))
    pg = pa.group_by("game_id", "defense").agg(pl.col("pa").sum().alias("n_pa"), (1 - pl.col("pa")).sum().alias("n_np"),
                                               (pl.col("cy") * pl.col("pa")).sum().alias("y_pa"), (pl.col("cy") * (1 - pl.col("pa"))).sum().alias("y_np"))
    pprior = _def_prior(games, pg, [], ["n_pa", "n_np", "y_pa", "y_np"]).join(games.select("game_id", "defense", "season"), on=["game_id", "defense"]) \
        .join(lgp.select("season", "lg_pa", "L_ypa_pa", "L_ypa_np"), on="season", how="left") \
        .with_columns(((pl.col("P_y_pa") + DEF_PSEUDO_PA * pl.col("L_ypa_pa")) / (pl.col("P_n_pa") + DEF_PSEUDO_PA) / pl.col("L_ypa_pa")).alias("R_pa"),
                      ((pl.col("P_y_np") + DEF_PSEUDO_PA * pl.col("L_ypa_np")) / (pl.col("P_n_np") + DEF_PSEUDO_PA) / pl.col("L_ypa_np")).alias("Q_np"))
    qp = pa.rename({"passer_player_id": "gsis_id"}).filter(pl.col("gsis_id").is_not_null()).group_by("gsis_id", "game_id") \
           .agg(pl.len().alias("a"), pl.col("pa").sum())
    j = _player_prior(u, qp, ["a", "pa"]).join(lgp.select("season", "lg_pa"), on="season", how="left") \
        .with_columns(((pl.col("Q_pa") + PA_PSEUDO * pl.col("lg_pa")) / (pl.col("Q_a") + PA_PSEUDO)).alias("p_i"))
    rows = base.join(j.select("gsis_id", "game_id", "p_i"), on=["gsis_id", "game_id"], how="left") \
               .join(pprior.select("game_id", "defense", "R_pa", "Q_np", "lg_pa"), left_on=["game_id", "opp"], right_on=["game_id", "defense"], how="left")
    p, pb, R, Q = pl.col("p_i"), pl.col("lg_pa"), pl.col("R_pa"), pl.col("Q_np")
    rows = rows.with_columns(((p * R + (1 - p) * Q) / (pb * R + (1 - pb) * Q)).alias("pa_ypc"))
    out = out.join(rows.select("gsis_id", "game_id", "pa_ypc"), on=["gsis_id", "game_id"], how="left")

    out = out.with_columns(*[pl.col(c).fill_null(1.0).fill_nan(1.0) for c in out.columns if c not in ("gsis_id", "game_id")])
    if not return_tendencies:
        return out
    tend = (games.select("game_id", "defense", "season")
            .join(bprior.select("game_id", "defense", "h_def", "lg_bl", "P_d"), on=["game_id", "defense"], how="left")
            .join(xprior.select("game_id", "defense", "h_box", "lg_hv", "P_n"), on=["game_id", "defense"], how="left")
            .join(pprior.select("game_id", "defense", "R_pa", "Q_np", "P_n_pa"), on=["game_id", "defense"], how="left"))
    return out, tend
