"""3c: matchup features, all computed from EARLIER games only.

For every (player, game) row of the Phase 3a universe:
  depth_catch, depth_ypc : sum_k mix_ik * (defense catch rate / yards per catch allowed in depth bucket k
                           / league rate in k);  mix_ik = receiver's own target mix over depth buckets
  gap_ypc                : sum_k mix_ik * (defense yards per carry allowed in gap bucket k / league)
  press_comp, press_ypc, press_td, press_int :
                           [(1-h_j) + h_j*s_i] / [(1-hbar) + hbar*s_i], h_j = defense pressure
                           on pass attempts, s_i = QB's pressured/clean ratio for that rate
  arch                   : receiver archetype (Gaussian mixture fit on earlier seasons), or null
and for every (game, team): own offense and opponent defense ratings as of that week.

Defense tendencies: half-life 16 of the defense's games, shrunk toward the league rate.
League rates: the three seasons before the game's season.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from proplab.p1.data import decay_for, ewm_prior_sums
from proplab.ratings.team import CANON

DEF_HL = 16
MIX_HL = 16
DEPTH = ("behind", "short", "mid", "deep")
GAPS = ("inside", "tackle", "outside")
MIX_PSEUDO = 20.0       # receiver/rusher mix shrunk toward position mix
DEF_PSEUDO_T = 60.0     # defense per-bucket targets / carries
DEF_PSEUDO_P = 150.0    # defense dropbacks for pressure rate
QB_PSEUDO = 40.0        # QB pressured attempts
QB_PSEUDO_RARE = 200.0  # QB pressured attempts for TD / INT ratios


def _q(conn, sql):
    cur = conn.execute(sql)
    return pl.DataFrame(cur.fetchall(), schema=[d.name for d in cur.description], orient="row", infer_schema_length=None)


def load_plays(conn) -> pl.DataFrame:
    return _q(conn, """
        SELECT p.game_id, g.season, g.kickoff_utc, p.posteam, p.defteam, p.qb_dropback, p.rush, p.sack, p.qb_hit,
               p.complete_pass, p.interception, p.touchdown, p.air_yards::float AS air_yards,
               p.yards_gained::float AS yds, p.run_location, p.run_gap,
               p.passer_player_id, p.receiver_player_id, p.rusher_player_id
        FROM plays p JOIN games g USING (game_id)
        WHERE g.game_type = 'REG' AND p.play_type IN ('pass', 'run') AND p.down IS NOT NULL""")


def _league(df: pl.DataFrame, by: list[str], num: dict) -> pl.DataFrame:
    """League rates per season from the three PRIOR seasons."""
    per = df.group_by(["season", *by]).agg(**{k: v for k, v in num.items()})
    seasons = sorted(per["season"].unique().to_list())
    rows = []
    for s in seasons + [max(seasons) + 1]:
        h = per.filter(pl.col("season").is_between(s - 3, s - 1))
        if h.is_empty():
            continue
        agg = h.group_by(by).agg(*[pl.col(k).sum() for k in num]) if by else h.select(*[pl.col(k).sum() for k in num])
        rows.append(agg.with_columns(pl.lit(int(s), dtype=pl.Int64).alias("season")))
    return pl.concat(rows)


def _prior_share(df: pl.DataFrame, group: list[str], bucket: str) -> pl.DataFrame:
    """Share of each bucket within `group`, per season, from the three PRIOR seasons."""
    lg = _league(df, [*group, bucket], {"n": pl.len()})
    return lg.with_columns((pl.col("n") / pl.col("n").sum().over(["season", *group])).alias("share")) \
             .select("season", *group, bucket, "share")


def _def_prior(games: pl.DataFrame, per_game: pl.DataFrame, keys: list[str], cols: list[str]) -> pl.DataFrame:
    """EWMA over each defense's PREVIOUS games (prior-only), per bucket."""
    grid = (games.join(per_game, on=["game_id", "defense", *keys], how="left").fill_null(0)
                 .sort(["defense", *keys, "kickoff_utc"]))
    k = grid.select(pl.concat_str(["defense", *keys], separator="|")).to_series().to_numpy()
    played = grid["played"].to_numpy().astype(float)
    d = decay_for(DEF_HL)
    return grid.with_columns(*[pl.Series(f"P_{c}", ewm_prior_sums(k, grid[c].to_numpy().astype(float) * played, d))
                               for c in cols]).select("game_id", "defense", *keys, *[f"P_{c}" for c in cols])


def _mix_prior(u: pl.DataFrame, per_game: pl.DataFrame, bucket_col: str, buckets: tuple, count: str) -> pl.DataFrame:
    """Each player's prior bucket counts (EWMA over his previous active games)."""
    wide = (per_game.pivot(on=bucket_col, index=["gsis_id", "game_id"], values=count, aggregate_function="sum")
                    .fill_null(0))
    for b in buckets:
        if b not in wide.columns:
            wide = wide.with_columns(pl.lit(0).alias(b))
    j = u.select("gsis_id", "game_id", "kickoff_utc", "team").join(wide, on=["gsis_id", "game_id"], how="left") \
         .fill_null(0).sort("gsis_id", "kickoff_utc", "team")
    keys = j["gsis_id"].to_numpy()
    d = decay_for(MIX_HL)
    return j.with_columns(*[pl.Series(f"M_{b}", ewm_prior_sums(keys, j[b].to_numpy().astype(float), d)) for b in buckets]) \
            .select("gsis_id", "game_id", *[f"M_{b}" for b in buckets])


def build(conn, u: pl.DataFrame, tg: pl.DataFrame, return_tendencies: bool = False):
    """Matchup columns keyed by (gsis_id, game_id). With return_tendencies, also the
    defense-side tables (per game_id and defense, prior-only) for the matchup explorer."""
    plays = load_plays(conn).with_columns(pl.col("defteam").alias("defense"))
    games = tg.select("game_id", pl.col("opp").alias("defense"), "kickoff_utc", "played", "season")
    base = u.select("gsis_id", "game_id", "season", "pos", "opp")

    # ---------------- depth of target
    tgt = plays.filter(pl.col("qb_dropback") & pl.col("receiver_player_id").is_not_null() & ~pl.col("sack")
                       & pl.col("air_yards").is_not_null())
    tgt = tgt.with_columns(
        pl.when(pl.col("air_yards") < 0).then(pl.lit("behind")).when(pl.col("air_yards") < 10).then(pl.lit("short"))
          .when(pl.col("air_yards") < 20).then(pl.lit("mid")).otherwise(pl.lit("deep")).alias("depth"),
        pl.col("complete_pass").cast(pl.Int32).alias("c"),
        pl.when(pl.col("complete_pass")).then(pl.col("yds")).otherwise(0.0).alias("cy"))
    lg = _league(tgt, ["depth"], {"t": pl.len(), "c": pl.col("c").sum(), "cy": pl.col("cy").sum()}) \
        .with_columns((pl.col("c") / pl.col("t")).alias("lg_cr"), (pl.col("cy") / pl.col("c")).alias("lg_ypc"))
    dg = tgt.group_by("game_id", "defense", "depth").agg(pl.len().alias("t"), pl.col("c").sum(), pl.col("cy").sum())
    dgrid = games.join(pl.DataFrame({"depth": list(DEPTH)}), how="cross")
    dprior = _def_prior(dgrid, dg, ["depth"], ["t", "c", "cy"]).join(
        games.select("game_id", "defense", "season"), on=["game_id", "defense"]).join(
        lg.select("season", "depth", "lg_cr", "lg_ypc"), on=["season", "depth"], how="left")
    dprior = dprior.with_columns(
        (((pl.col("P_c") + DEF_PSEUDO_T * pl.col("lg_cr")) / (pl.col("P_t") + DEF_PSEUDO_T)) / pl.col("lg_cr")).alias("r_cr"),
        (((pl.col("P_cy") + DEF_PSEUDO_T * pl.col("lg_cr") * pl.col("lg_ypc"))
          / (pl.col("P_c") + DEF_PSEUDO_T * pl.col("lg_cr"))) / pl.col("lg_ypc")).alias("r_ypc"))
    pmix = _mix_prior(u, tgt.rename({"receiver_player_id": "gsis_id"}).group_by("gsis_id", "game_id", "depth").agg(pl.len().alias("n")),
                      "depth", DEPTH, "n")
    # position depth mix from the three prior seasons only (shrinkage target for each receiver's mix)
    pos_mix = _prior_share(tgt.join(u.select("gsis_id", "pos").unique(subset="gsis_id"),
                                    left_on="receiver_player_id", right_on="gsis_id"), ["pos"], "depth") \
        .pivot(on="depth", index=["season", "pos"], values="share")
    rows = base.join(pmix, on=["gsis_id", "game_id"], how="left").fill_null(0) \
               .join(pos_mix.rename({b: f"pm_{b}" for b in DEPTH}), on=["season", "pos"], how="left")
    tot = sum(pl.col(f"M_{b}") for b in DEPTH)
    for b in DEPTH:
        rows = rows.with_columns(((pl.col(f"M_{b}") + MIX_PSEUDO * pl.col(f"pm_{b}").fill_null(0.25))
                                  / (tot + MIX_PSEUDO)).alias(f"mix_{b}"))
    dw = dprior.pivot(on="depth", index=["game_id", "defense"], values=["r_cr", "r_ypc"])
    rows = rows.join(dw, left_on=["game_id", "opp"], right_on=["game_id", "defense"], how="left")
    rows = rows.with_columns(
        sum(pl.col(f"mix_{b}") * pl.col(f"r_cr_{b}").fill_null(1.0) for b in DEPTH).alias("depth_catch"),
        sum(pl.col(f"mix_{b}") * pl.col(f"r_ypc_{b}").fill_null(1.0) for b in DEPTH).alias("depth_ypc"))
    out = rows.select("gsis_id", "game_id", "depth_catch", "depth_ypc")

    # ---------------- run gap
    runs = plays.filter(pl.col("rush") & ~pl.col("qb_dropback") & pl.col("run_location").is_not_null()
                        & pl.col("yds").is_not_null())
    runs = runs.with_columns(
        pl.when((pl.col("run_location") == "middle") | (pl.col("run_gap") == "guard")).then(pl.lit("inside"))
          .when(pl.col("run_gap") == "tackle").then(pl.lit("tackle")).otherwise(pl.lit("outside")).alias("gap"))
    lg = _league(runs, ["gap"], {"n": pl.len(), "y": pl.col("yds").sum()}).with_columns((pl.col("y") / pl.col("n")).alias("lg_ypc"))
    rg = runs.group_by("game_id", "defense", "gap").agg(pl.len().alias("n"), pl.col("yds").sum().alias("y"))
    ggrid = games.join(pl.DataFrame({"gap": list(GAPS)}), how="cross")
    gprior = _def_prior(ggrid, rg, ["gap"], ["n", "y"]).join(games.select("game_id", "defense", "season"), on=["game_id", "defense"]) \
        .join(lg.select("season", "gap", "lg_ypc"), on=["season", "gap"], how="left") \
        .with_columns((((pl.col("P_y") + DEF_PSEUDO_T * pl.col("lg_ypc")) / (pl.col("P_n") + DEF_PSEUDO_T)) / pl.col("lg_ypc")).alias("r"))
    gmix = _mix_prior(u, runs.rename({"rusher_player_id": "gsis_id"}).group_by("gsis_id", "game_id", "gap").agg(pl.len().alias("n")),
                      "gap", GAPS, "n")
    # league gap mix from the three prior seasons only
    lg_mix = _prior_share(runs, [], "gap").pivot(on="gap", index="season", values="share")
    rows = base.join(gmix, on=["gsis_id", "game_id"], how="left").fill_null(0) \
               .join(lg_mix.rename({b: f"lm_{b}" for b in GAPS}), on="season", how="left")
    tot = sum(pl.col(f"M_{b}") for b in GAPS)
    rows = rows.with_columns(*[((pl.col(f"M_{b}") + MIX_PSEUDO * pl.col(f"lm_{b}").fill_null(1 / 3)) / (tot + MIX_PSEUDO))
                               .alias(f"gmix_{b}") for b in GAPS])
    gw = gprior.pivot(on="gap", index=["game_id", "defense"], values="r")
    rows = rows.join(gw, left_on=["game_id", "opp"], right_on=["game_id", "defense"], how="left")
    rows = rows.with_columns(sum(pl.col(f"gmix_{b}") * pl.col(b).fill_null(1.0) for b in GAPS).alias("gap_ypc"))
    out = out.join(rows.select("gsis_id", "game_id", "gap_ypc"), on=["gsis_id", "game_id"], how="left")

    # ---------------- pressure
    db_ = plays.filter(pl.col("qb_dropback"))
    # pass attempts: dropbacks that were neither sacks nor scrambles
    att = db_.filter(~pl.col("sack") & ~pl.col("rush") & pl.col("complete_pass").is_not_null())
    att = att.with_columns(pl.col("qb_hit").cast(pl.Int32).alias("hit"), pl.col("complete_pass").cast(pl.Int32).alias("c"),
                           pl.when(pl.col("complete_pass")).then(pl.col("yds")).otherwise(0.0).alias("cy"),
                           pl.col("interception").cast(pl.Int32).alias("i"),
                           (pl.col("touchdown") & pl.col("complete_pass")).cast(pl.Int32).alias("td"))
    press = db_.with_columns((pl.col("sack") | pl.col("qb_hit")).cast(pl.Int32).alias("pr"))
    lgp = _league(press, [], {"d": pl.len(), "pr": pl.col("pr").sum()}).with_columns((pl.col("pr") / pl.col("d")).alias("lg_pr"))
    lgh = _league(att, [], {"a": pl.len(), "h": pl.col("hit").sum()}).with_columns((pl.col("h") / pl.col("a")).alias("lg_h"))
    pg = press.group_by("game_id", "defense").agg(pl.len().alias("d"), pl.col("pr").sum())
    pprior = _def_prior(games, pg, [], ["d", "pr"]).join(games.select("game_id", "defense", "season"), on=["game_id", "defense"]) \
        .join(lgp.select("season", "lg_pr"), on="season", how="left").join(lgh.select("season", "lg_h"), on="season", how="left") \
        .with_columns((((pl.col("P_pr") + DEF_PSEUDO_P * pl.col("lg_pr")) / (pl.col("P_d") + DEF_PSEUDO_P)) / pl.col("lg_pr")
                       * pl.col("lg_h")).alias("h_def"))
    # QB pressured vs clean, prior-only over his previous games
    hit, clean = pl.col("hit"), 1 - pl.col("hit")
    qg = (att.rename({"passer_player_id": "gsis_id"}).filter(pl.col("gsis_id").is_not_null())
             .group_by("gsis_id", "game_id")
             .agg(hit.sum().alias("a_p"), clean.sum().alias("a_c"),
                  (pl.col("c") * hit).sum().alias("c_p"), (pl.col("c") * clean).sum().alias("c_c"),
                  (pl.col("cy") * hit).sum().alias("cy_p"), (pl.col("cy") * clean).sum().alias("cy_c"),
                  (pl.col("i") * hit).sum().alias("i_p"), (pl.col("i") * clean).sum().alias("i_c"),
                  (pl.col("td") * hit).sum().alias("td_p"), (pl.col("td") * clean).sum().alias("td_c")))
    # league pressured / clean rates per season, from the three prior seasons only
    lr = _league(att, [], {
        "a_p": pl.col("hit").sum(), "a_c": (1 - pl.col("hit")).sum(),
        "c_p": (pl.col("c") * pl.col("hit")).sum(), "c_c": (pl.col("c") * (1 - pl.col("hit"))).sum(),
        "cy_p": (pl.col("cy") * pl.col("hit")).sum(), "cy_c": (pl.col("cy") * (1 - pl.col("hit"))).sum(),
        "i_p": (pl.col("i") * pl.col("hit")).sum(), "i_c": (pl.col("i") * (1 - pl.col("hit"))).sum(),
        "td_p": (pl.col("td") * pl.col("hit")).sum(), "td_c": (pl.col("td") * (1 - pl.col("hit"))).sum()})
    lr = lr.select("season",
                   (pl.col("c_p") / pl.col("a_p")).alias("L_cr_p"), (pl.col("c_c") / pl.col("a_c")).alias("L_cr_c"),
                   (pl.col("cy_p") / pl.col("c_p")).alias("L_ypc_p"), (pl.col("cy_c") / pl.col("c_c")).alias("L_ypc_c"),
                   (pl.col("i_p") / pl.col("a_p")).alias("L_ir_p"), (pl.col("i_c") / pl.col("a_c")).alias("L_ir_c"),
                   (pl.col("td_p") / pl.col("a_p")).alias("L_tr_p"), (pl.col("td_c") / pl.col("a_c")).alias("L_tr_c"))
    j = u.select("gsis_id", "game_id", "kickoff_utc", "team", "season").join(qg, on=["gsis_id", "game_id"], how="left") \
         .fill_null(0).sort("gsis_id", "kickoff_utc", "team")
    keys = j["gsis_id"].to_numpy()
    dd = decay_for(MIX_HL)
    qcols = ["a_p", "a_c", "c_p", "c_c", "cy_p", "cy_c", "i_p", "i_c", "td_p", "td_c"]
    j = j.with_columns(*[pl.Series(f"Q_{c}", ewm_prior_sums(keys, j[c].to_numpy().astype(float), dd)) for c in qcols])
    j = j.join(lr, on="season", how="left")

    def ratio(num_p, den_p, num_c, den_c, lp, lc, k):
        # shrunk pressured rate / shrunk clean rate
        rp = (pl.col(num_p) + k * pl.col(lp)) / (pl.col(den_p) + k)
        rc = (pl.col(num_c) + k * pl.col(lc)) / (pl.col(den_c) + k)
        return rp / rc

    j = j.with_columns(
        ratio("Q_c_p", "Q_a_p", "Q_c_c", "Q_a_c", "L_cr_p", "L_cr_c", QB_PSEUDO).alias("s_comp"),
        ratio("Q_cy_p", "Q_c_p", "Q_cy_c", "Q_c_c", "L_ypc_p", "L_ypc_c", QB_PSEUDO).alias("s_ypc"),
        ratio("Q_i_p", "Q_a_p", "Q_i_c", "Q_a_c", "L_ir_p", "L_ir_c", QB_PSEUDO_RARE).alias("s_int"),
        ratio("Q_td_p", "Q_a_p", "Q_td_c", "Q_a_c", "L_tr_p", "L_tr_c", QB_PSEUDO_RARE).alias("s_td"),
    ).select("gsis_id", "game_id", "s_comp", "s_ypc", "s_int", "s_td")
    rows = base.join(j, on=["gsis_id", "game_id"], how="left").join(
        pprior.select("game_id", "defense", "h_def", "lg_h"), left_on=["game_id", "opp"], right_on=["game_id", "defense"], how="left")
    h, hb = pl.col("h_def").fill_null(pl.col("lg_h")), pl.col("lg_h")
    mixf = lambda s: ((1 - h) + h * pl.col(s)) / ((1 - hb) + hb * pl.col(s))
    rows = rows.with_columns(*[mixf(s).fill_null(1.0).alias(f"press_{s[2:]}") for s in ("s_comp", "s_ypc", "s_int", "s_td")])
    out = out.join(rows.select("gsis_id", "game_id", "press_comp", "press_ypc", "press_int", "press_td"),
                   on=["gsis_id", "game_id"], how="left")
    out = out.with_columns(*[pl.col(c).fill_null(1.0) for c in out.columns if c not in ("gsis_id", "game_id")])
    if not return_tendencies:
        return out
    tend = {
        "depth": dprior.select("game_id", "defense", "depth", "r_cr", "r_ypc", "P_t"),
        "gap": gprior.select("game_id", "defense", "gap", "r", "P_n"),
        "pressure": pprior.select("game_id", "defense", (pl.col("h_def") / pl.col("lg_h")).alias("r_press"), "P_d"),
    }
    return out, tend


def team_rating_covariates(conn, tg: pl.DataFrame) -> pl.DataFrame:
    """Own offense and opponent defense ratings (as of that week) per (game_id, team)."""
    tr = _q(conn, "SELECT season, week, team, unit, off, def FROM team_ratings")
    w = tr.pivot(on="unit", index=["season", "week", "team"], values=["off", "def"])
    t = tg.select("game_id", "season", "week", "team", "opp").with_columns(
        pl.col("team").replace(CANON).alias("tc"), pl.col("opp").replace(CANON).alias("oc"))
    t = t.join(w.select("season", "week", "team", "off_pass", "off_run"), left_on=["season", "week", "tc"],
               right_on=["season", "week", "team"], how="left")
    t = t.join(w.select("season", "week", pl.col("team").alias("oc"), pl.col("def_pass").alias("opp_def_pass"),
                        pl.col("def_run").alias("opp_def_run")), on=["season", "week", "oc"], how="left")
    return t.select("game_id", "team", *[pl.col(c).fill_null(0.0) for c in ("off_pass", "off_run", "opp_def_pass", "opp_def_run")])


def archetypes(conn, u: pl.DataFrame) -> pl.DataFrame:
    """WR/TE archetype per (gsis_id, game_id), from a Gaussian mixture fit on EARLIER seasons'
    player-seasons; the player's profile uses only his previous games (prior-only EWMA)."""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    pg = _q(conn, """
        SELECT gsis_id, game_id, season, targets, receptions, receiving_air_yards::float AS ay,
               receiving_yards_after_catch::float AS yac, target_share::float AS ts, receiving_yards::float AS yds
        FROM player_game WHERE position IN ('WR', 'TE')""").fill_null(0)
    feats = ["adot", "yac_pc", "share", "catch"]

    def profile(df):
        return df.with_columns((pl.col("ay") / pl.col("targets")).alias("adot"),
                               (pl.col("yac") / pl.col("receptions").clip(1)).alias("yac_pc"),
                               (pl.col("ts_w") / pl.col("g")).alias("share"),
                               (pl.col("receptions") / pl.col("targets")).alias("catch"))

    seasonal = profile(pg.group_by("gsis_id", "season").agg(
        pl.col("targets").sum(), pl.col("receptions").sum(), pl.col("ay").sum(), pl.col("yac").sum(),
        pl.col("yds").sum(), pl.col("ts").sum().alias("ts_w"), pl.len().alias("g"))).filter(pl.col("targets") >= 40)

    # prior-only rolling profile for every u row (WR/TE)
    j = u.filter(pl.col("pos").is_in(["WR", "TE"])).select("gsis_id", "game_id", "kickoff_utc", "team", "season") \
         .join(pg.select("gsis_id", "game_id", "targets", "receptions", "ay", "yac", "ts"), on=["gsis_id", "game_id"], how="left") \
         .fill_null(0).sort("gsis_id", "kickoff_utc", "team")
    keys = j["gsis_id"].to_numpy()
    d = decay_for(32)
    played_t = (j["targets"].to_numpy() > 0).astype(float)
    j = j.with_columns(*[pl.Series(c, ewm_prior_sums(keys, j[c].to_numpy().astype(float), d)) for c in ("targets", "receptions", "ay", "yac")],
                       pl.Series("ts_w", ewm_prior_sums(keys, j["ts"].to_numpy().astype(float), d)),
                       pl.Series("g", ewm_prior_sums(keys, played_t, d)))
    j = profile(j).filter(pl.col("targets") >= 30)

    out, pri = [], []
    for s in sorted(j["season"].unique().to_list()):
        train = seasonal.filter(pl.col("season").is_between(s - 3, s - 1))
        if train.height < 100:
            continue
        X = train.select(feats).to_numpy()
        mu, sd = X.mean(0), X.std(0)
        Z = (X - mu) / sd
        rng = np.random.default_rng(s)
        best_k, best_ari = None, -1.0
        for kk in range(2, 7):
            ref = GaussianMixture(kk, random_state=0, n_init=3).fit(Z).predict(Z)
            aris = [adjusted_rand_score(ref, GaussianMixture(kk, random_state=b + 1, n_init=1)
                                        .fit(Z[rng.integers(0, len(Z), len(Z))]).predict(Z)) for b in range(10)]
            if np.mean(aris) > best_ari + 1e-9:
                best_k, best_ari = kk, float(np.mean(aris))
        gm = GaussianMixture(best_k, random_state=0, n_init=3).fit(Z)
        # archetype shrinkage targets, from the same earlier seasons the mixture was fit on
        tl = train.with_columns(pl.Series("arch", [f"{s}-{x}" for x in gm.predict(Z)]))
        pri.append(tl.group_by("arch").agg((pl.col("receptions").sum() / pl.col("targets").sum()).alias("arch_rate"),
                                           (pl.col("yds").sum() / pl.col("receptions").sum()).alias("arch_ypc"),
                                           pl.len().alias("arch_n")).with_columns(pl.lit(best_k).alias("arch_k"),
                                                                                  pl.lit(best_ari).alias("arch_stability")))
        rows = j.filter(pl.col("season") == s)
        if rows.is_empty():
            continue
        lab = gm.predict((rows.select(feats).to_numpy() - mu) / sd)
        out.append(rows.select("gsis_id", "game_id").with_columns(pl.Series("arch", [f"{s}-{x}" for x in lab])))
    priors = pl.concat(pri)
    return pl.concat(out).join(priors.select("arch", "arch_rate", "arch_ypc"), on="arch", how="left"), priors
