"""Phase 3a: one engine for every volume -> success -> yards prop market.

Volume types:
  tgt  targets      -> catches (catch rate)       -> receiving yards per catch, rec TD per catch
  car  carries      -> every carry                -> rushing yards per carry,  rush TD per carry
  att  pass attempts-> completions (comp. rate)   -> passing yards per compl., pass TD and INT per attempt

Point-in-time rules are the same as Phase 1: player/team/defense features are
weighted sums of PREVIOUS games only; constants, calibration and dispersion are
fit on the three seasons before the season being predicted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from proplab.p1.data import decay_for, ewm_prior_sums
from proplab.p1.model import crps_samples, ece, randomized_pit

POSITIONS = ("QB", "RB", "WR", "TE")
ELIG_HALF_LIFE = 8
ELIG_MIN_GAMES = 3
TEAM_HALF_LIFE = 8
TEAM_PSEUDO_GAMES = 4.0
DEF_HALF_LIFE = 16
DEF_PSEUDO = 60.0
TD_SHRINK_MULT = 5.0
EXIT_CUTOFF = 0.35          # a game below 35% of expected volume counts as an early exit
EXIT_MEAN_FRACTION = EXIT_CUTOFF / 2   # exits keep a uniform 0-35% of a full game's volume

VT = {
    "tgt": {"vol": "targets", "team": "team_targets", "succ": "receptions", "yds": "rec_yds",
            "td": "rec_tds", "td_units": "succ", "int": None, "pool": "catch"},
    "car": {"vol": "carries", "team": "team_carries", "succ": None, "yds": "rush_yds",
            "td": "rush_tds", "td_units": "vol", "int": None, "pool": "carry"},
    "att": {"vol": "attempts", "team": "team_attempts", "succ": "completions", "yds": "pass_yds",
            "td": "pass_tds", "td_units": "vol", "int": "ints", "pool": "completion"},
}

# (market, position(s), volume type, quantity, fixed lines)
MARKETS = {
    "qb_pass_yds":  (("QB",), "att", "yds", [199.5, 224.5, 249.5, 274.5]),
    "qb_pass_att":  (("QB",), "att", "vol", [29.5, 32.5, 35.5]),
    "qb_pass_tds":  (("QB",), "att", "td", [0.5, 1.5, 2.5]),
    "qb_ints":      (("QB",), "att", "int", [0.5]),
    "qb_rush_yds":  (("QB",), "car", "yds", [9.5, 19.5, 29.5]),
    "rb_rush_yds":  (("RB",), "car", "yds", [39.5, 59.5, 79.5]),
    "rb_rush_att":  (("RB",), "car", "vol", [9.5, 14.5, 19.5]),
    "rb_rec":       (("RB",), "tgt", "succ", [1.5, 2.5, 3.5]),
    "rb_rec_yds":   (("RB",), "tgt", "yds", [9.5, 19.5, 29.5]),
    "anytime_td":   (("RB", "WR", "TE"), "both", "atd", [0.5]),
}
# Phase 1 markets, rebuilt inside this engine as the base for 3c (not part of the 3a gate)
WRTE_MARKETS = {
    "wrte_rec":     (("WR", "TE"), "tgt", "succ", [2.5, 3.5, 4.5, 5.5, 6.5]),
    "wrte_rec_yds": (("WR", "TE"), "tgt", "yds", [24.5, 39.5, 54.5, 69.5, 84.5]),
}
ALL_MARKETS = {**WRTE_MARKETS, **MARKETS}
TEAM_RT_COLS = ("off_pass", "off_run", "opp_def_pass", "opp_def_run")
MARKET_LABELS = {
    "qb_pass_yds": "QB passing yards", "qb_pass_att": "QB pass attempts", "qb_pass_tds": "QB passing TDs",
    "qb_ints": "QB interceptions", "qb_rush_yds": "QB rushing yards", "rb_rush_yds": "RB rushing yards",
    "rb_rush_att": "RB rushing attempts", "rb_rec": "RB receptions", "rb_rec_yds": "RB receiving yards",
    "anytime_td": "Anytime TD (RB/WR/TE)",
    "wrte_rec": "WR/TE receptions", "wrte_rec_yds": "WR/TE receiving yards",
}
OUTCOME = {  # market -> outcome column(s)
    "wrte_rec": "receptions", "wrte_rec_yds": "rec_yds",
    "qb_pass_yds": "pass_yds", "qb_pass_att": "attempts", "qb_pass_tds": "pass_tds", "qb_ints": "ints",
    "qb_rush_yds": "rush_yds", "rb_rush_yds": "rush_yds", "rb_rush_att": "carries",
    "rb_rec": "receptions", "rb_rec_yds": "rec_yds",
}


@dataclass(frozen=True)
class Config:
    model: str = "M2"          # B1 | B2 | M1 | M2
    half_life: float = 8.0
    a_share: float = 40.0
    a_eff: float = 10.0
    a_tpg: float = 4.0
    redistribute: float = 1.0  # used by M1, M2
    opponent: float = 0.5      # used by M2
    # 3c matchup feature groups (all off by default, so Phase 3a configs reproduce exactly)
    g_depth: float = 0.0       # depth-of-target multipliers on catch rate and yards per catch
    g_gap: float = 0.0         # run-gap multiplier on RB yards per carry
    g_press: float = 0.0       # pressure multipliers on QB completion, yards/compl., TD and INT rates
    arch: bool = False         # receiver archetype as shrinkage target (WR/TE)
    team_ratings: bool = False # team ratings in the team-volume regressions
    # Phase 6 scheme feature groups (off by default)
    g_blitz: float = 0.0       # blitz multipliers on QB completion, yards/compl., TD and INT rates
    g_box: float = 0.0         # stacked-box multiplier on RB yards per carry
    g_pa: float = 0.0          # play-action multiplier on QB yards per completion
    # Phase 7 weather (off by default)
    g_wx: float = 0.0          # weather multiplier on completion rate and yards per completion (QB attempts, targets)

    def label(self) -> str:
        s = f"{self.model}(h={self.half_life:g},as={self.a_share:g},ae={self.a_eff:g}"
        if self.model in ("M1", "M2"):
            s += f",lam={self.redistribute:g}"
        if self.model == "M2":
            s += f",gam={self.opponent:g}"
        for name, v in (("depth", self.g_depth), ("gap", self.g_gap), ("press", self.g_press),
                        ("blitz", self.g_blitz), ("box", self.g_box), ("pa", self.g_pa), ("wx", self.g_wx)):
            if v:
                s += f",{name}={v:g}"
        if self.arch:
            s += ",arch"
        if self.team_ratings:
            s += ",teamrt"
        return s + ")"

    @property
    def lam(self) -> float:
        return self.redistribute if self.model in ("M1", "M2") else 0.0

    @property
    def gam(self) -> float:
        return self.opponent if self.model == "M2" else 0.0


# ---------------------------------------------------------------- data

def _q(conn, sql: str) -> pl.DataFrame:
    cur = conn.execute(sql)
    return pl.DataFrame(cur.fetchall(), schema=[d.name for d in cur.description], orient="row",
                        infer_schema_length=None)


def load(conn) -> dict:
    u = _q(conn, """
        SELECT r.gsis_id, p.display_name AS name, r.position AS pos, r.season, r.week, r.team, g.game_id,
               g.kickoff_utc, CASE WHEN g.home_team = r.team THEN g.away_team ELSE g.home_team END AS opp,
               (g.home_team = r.team) AS is_home,
               coalesce(pg.targets, 0) AS targets, coalesce(pg.receptions, 0) AS receptions,
               coalesce(pg.receiving_yards, 0)::float AS rec_yds, coalesce(pg.receiving_tds, 0) AS rec_tds,
               coalesce(pg.carries, 0) AS carries, coalesce(pg.rushing_yards, 0)::float AS rush_yds,
               coalesce(pg.rushing_tds, 0) AS rush_tds, coalesce(pg.attempts, 0) AS attempts,
               coalesce(pg.completions, 0) AS completions, coalesce(pg.passing_yards, 0)::float AS pass_yds,
               coalesce(pg.passing_tds, 0) AS pass_tds, coalesce(pg.passing_interceptions, 0) AS ints,
               (res.game_id IS NOT NULL) AS played
        FROM roster_status r
        JOIN games g ON g.season = r.season AND g.week = r.week
                    AND r.team IN (g.home_team, g.away_team) AND g.game_type = 'REG'
        LEFT JOIN players p ON p.gsis_id = r.gsis_id
        LEFT JOIN player_game pg ON pg.gsis_id = r.gsis_id AND pg.game_id = g.game_id
        LEFT JOIN game_results res ON res.game_id = g.game_id
        WHERE r.status = 'ACT' AND r.position IN ('QB', 'RB', 'WR', 'TE')""").sort("gsis_id", "kickoff_utc", "team")
    tg = _q(conn, """
        WITH t AS (SELECT game_id, team, sum(coalesce(targets,0)) tt, sum(coalesce(carries,0)) tc,
                          sum(coalesce(attempts,0)) ta FROM player_game GROUP BY game_id, team)
        SELECT g.game_id, g.season, g.week, g.kickoff_utc, side.team, side.opp,
               coalesce(t.tt, 0) AS team_targets, coalesce(t.tc, 0) AS team_carries, coalesce(t.ta, 0) AS team_attempts,
               (res.game_id IS NOT NULL) AS played,
               CASE WHEN side.team = g.home_team THEN l.spread_line ELSE -l.spread_line END::float AS team_spread,
               l.total_line::float AS total_line
        FROM games g
        CROSS JOIN LATERAL (VALUES (g.home_team, g.away_team), (g.away_team, g.home_team)) AS side(team, opp)
        LEFT JOIN t ON t.game_id = g.game_id AND t.team = side.team
        LEFT JOIN game_results res ON res.game_id = g.game_id
        LEFT JOIN game_lines l ON l.game_id = g.game_id AND l.book = 'nflverse_close'
        WHERE g.game_type = 'REG' ORDER BY side.team, g.kickoff_utc""")
    pools = _q(conn, """
        SELECT 'catch' AS pool, pl.season, p.position AS pos, pl.yards_gained::float AS yds, pl.game_id, pl.play_id
        FROM plays pl JOIN players p ON p.gsis_id = pl.receiver_player_id
        WHERE pl.complete_pass AND pl.yards_gained IS NOT NULL
        UNION ALL
        SELECT 'carry', pl.season, p.position, pl.yards_gained::float, pl.game_id, pl.play_id
        FROM plays pl JOIN players p ON p.gsis_id = pl.rusher_player_id
        WHERE pl.rush AND pl.yards_gained IS NOT NULL
        UNION ALL
        SELECT 'completion', pl.season, 'QB', pl.yards_gained::float, pl.game_id, pl.play_id
        FROM plays pl WHERE pl.complete_pass AND pl.yards_gained IS NOT NULL
        ORDER BY 1, 5, 6""")
    return {"u": u, "tg": tg, "pools": pools}


# ---------------------------------------------------------------- features

def features(u: pl.DataFrame, tg: pl.DataFrame, half_life: float) -> pl.DataFrame:
    u = u.join(tg.select("game_id", "team", "team_targets", "team_carries", "team_attempts"),
               on=["game_id", "team"], how="left").sort("gsis_id", "kickoff_utc", "team")
    keys = u["gsis_id"].to_numpy()
    played = u["played"].to_numpy().astype(float)
    d, d8 = decay_for(half_life), decay_for(ELIG_HALF_LIFE)
    cols = {"S_n": ewm_prior_sums(keys, played, d), "n_prev": ewm_prior_sums(keys, played, 1.0)}
    for vt, spec in VT.items():
        vol = u[spec["vol"]].to_numpy() * played
        team = u[spec["team"]].fill_null(0).to_numpy() * played
        succ = (u[spec["succ"]].to_numpy() if spec["succ"] else u[spec["vol"]].to_numpy()) * played
        cols[f"{vt}_S_v"] = ewm_prior_sums(keys, vol.astype(float), d)
        cols[f"{vt}_S_tv"] = ewm_prior_sums(keys, team.astype(float), d)
        cols[f"{vt}_S_s"] = ewm_prior_sums(keys, succ.astype(float), d)
        cols[f"{vt}_S_y"] = ewm_prior_sums(keys, (u[spec["yds"]].to_numpy() * played).astype(float), d)
        cols[f"{vt}_S_td"] = ewm_prior_sums(keys, (u[spec["td"]].to_numpy() * played).astype(float), d)
        if spec["int"]:
            cols[f"{vt}_S_int"] = ewm_prior_sums(keys, (u[spec["int"]].to_numpy() * played).astype(float), d)
        ev, et = ewm_prior_sums(keys, vol.astype(float), d8), ewm_prior_sums(keys, team.astype(float), d8)
        cols[f"{vt}_elig"] = np.divide(ev, et, out=np.zeros_like(ev), where=et > 0)
    out = u.with_columns(*[pl.Series(k, v) for k, v in cols.items()])
    # Projected starting QB: the active QB with the highest recent share of team pass
    # attempts in this game (pre-game information only).
    out = out.with_columns(
        ((pl.col("pos") == "QB") & (pl.col("att_elig") == pl.col("att_elig").filter(pl.col("pos") == "QB")
                                    .max().over("game_id", "team"))).alias("qb_top"))
    rt = [c for c in TEAM_RT_COLS if c in tg.columns]
    out = out.join(team_features(tg).select("game_id", "team", *[f"{vt}_T_n" for vt in VT],
                                            *[f"{vt}_T_v" for vt in VT], "team_spread", "total_line", *rt),
                   on=["game_id", "team"], how="left")
    return out.join(defense_features(u, tg), left_on=["game_id", "opp", "pos"],
                    right_on=["game_id", "defense", "pos"], how="left")


def team_features(tg: pl.DataFrame) -> pl.DataFrame:
    t = tg.sort("team", "kickoff_utc")
    keys = t["team"].to_numpy()
    played = t["played"].to_numpy().astype(float)
    d = decay_for(TEAM_HALF_LIFE)
    n = ewm_prior_sums(keys, played, d)
    cols = []
    for vt, spec in VT.items():
        cols += [pl.Series(f"{vt}_T_n", n),
                 pl.Series(f"{vt}_T_v", ewm_prior_sums(keys, t[spec["team"]].to_numpy() * played, d))]
    return t.with_columns(*cols)


def defense_features(u: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    aggs = []
    for vt, spec in VT.items():
        succ = pl.col(spec["succ"]) if spec["succ"] else pl.col(spec["vol"])
        units = succ if spec["td_units"] == "succ" else pl.col(spec["vol"])
        aggs += [pl.col(spec["vol"]).sum().alias(f"{vt}_v"), succ.sum().alias(f"{vt}_s"),
                 pl.col(spec["yds"]).sum().alias(f"{vt}_y"), pl.col(spec["td"]).sum().alias(f"{vt}_td"),
                 units.sum().alias(f"{vt}_u")]
    allowed = u.filter(pl.col("played")).group_by("game_id", "opp", "pos").agg(aggs).rename({"opp": "defense"})
    grid = (tg.select("game_id", pl.col("opp").alias("defense"), "kickoff_utc", "played")
              .join(pl.DataFrame({"pos": list(POSITIONS)}), how="cross")
              .join(allowed, on=["game_id", "defense", "pos"], how="left").fill_null(0)
              .sort("defense", "pos", "kickoff_utc"))
    keys = (grid["defense"] + "|" + grid["pos"]).to_numpy()
    played = grid["played"].to_numpy().astype(float)
    d = decay_for(DEF_HALF_LIFE)
    stat_cols = [c for c in grid.columns if c.split("_")[0] in VT and c not in ("defense",)]
    return grid.with_columns(*[pl.Series(f"D_{c}", ewm_prior_sums(keys, grid[c].to_numpy().astype(float) * played, d))
                               for c in stat_cols]).select("game_id", "defense", "pos", *[f"D_{c}" for c in stat_cols])


# ---------------------------------------------------------------- constants

ELIG_RULES = {  # (pos, vt) -> (share column vt, threshold)
    ("QB", "att"): ("att", 0.50), ("QB", "car"): ("att", 0.50),
    ("RB", "car"): ("car", 0.15), ("RB", "tgt"): ("tgt", 0.05),
    ("WR", "tgt"): ("tgt", 0.05), ("TE", "tgt"): ("tgt", 0.05),
    ("WR", "car"): ("car", 0.15), ("TE", "car"): ("car", 0.15),
}


def eligible_expr(pos: str, vt: str) -> pl.Expr:
    share_vt, thr = ELIG_RULES[(pos, vt)]
    e = (pl.col("pos") == pos) & (pl.col("n_prev") >= ELIG_MIN_GAMES) & (pl.col(f"{share_vt}_elig") >= thr)
    return e & pl.col("qb_top") if pos == "QB" else e


def market_eligible(market: str) -> pl.Expr:
    if market == "anytime_td":
        return pl.col("pos").is_in(["RB", "WR", "TE"]) & (pl.col("n_prev") >= ELIG_MIN_GAMES) & (
            (pl.col("car_elig") >= 0.15) | (pl.col("tgt_elig") >= 0.05))
    pos, vt, _, _ = ALL_MARKETS[market]
    e = eligible_expr(pos[0], vt)
    for p in pos[1:]:
        e = e | eligible_expr(p, vt)
    return e


@dataclass
class Constants:
    prior: dict = field(default_factory=dict)       # (vt, pos, stat) -> value
    league_team: dict = field(default_factory=dict)  # vt -> mean team volume
    covered: dict = field(default_factory=dict)      # vt -> share of team volume covered by QB/RB/WR/TE
    volume_coef: dict = field(default_factory=dict)  # vt -> [c0, c_ewma, c_spread, c_total]
    calibration: dict = field(default_factory=dict)  # (vt, pos) -> factor
    dispersion: dict = field(default_factory=dict)   # (vt, pos) -> r, fit on full games
    exit_rate: dict = field(default_factory=dict)    # (vt, pos) -> share of games far below expectation
    pools: dict = field(default_factory=dict)        # (pool, pos) -> draws normalized to mean 1


def volume_design(df: pl.DataFrame, vt: str, k: "Constants", team_ratings: bool) -> np.ndarray:
    ew = ((df[f"{vt}_T_v"] + TEAM_PSEUDO_GAMES * k.league_team[vt]) / (df[f"{vt}_T_n"] + TEAM_PSEUDO_GAMES)).to_numpy()
    cols = [np.ones(df.height), ew, df["team_spread"].to_numpy(), df["total_line"].to_numpy()]
    if team_ratings:
        cols += [df[c].fill_null(0.0).to_numpy() for c in TEAM_RT_COLS]
    return np.column_stack(cols)


def fit_constants(feat: pl.DataFrame, tf: pl.DataFrame, pools: pl.DataFrame, season: int,
                  team_ratings: bool = False) -> Constants:
    k = Constants()
    hist = feat.filter(pl.col("played") & pl.col("season").is_between(season - 3, season - 1))
    for vt, spec in VT.items():
        succ = pl.col(spec["succ"]) if spec["succ"] else pl.col(spec["vol"])
        units = succ if spec["td_units"] == "succ" else pl.col(spec["vol"])
        by = hist.group_by("pos").agg(
            (pl.col(spec["vol"]).sum() / pl.col(spec["team"]).sum()).alias("share"),
            (succ.sum() / pl.col(spec["vol"]).sum()).alias("rate"),
            (pl.col(spec["yds"]).sum() / succ.sum()).alias("ypu"),
            (pl.col(spec["td"]).sum() / units.sum()).alias("tdr"),
            ((pl.col(spec["int"]).sum() if spec["int"] else pl.lit(0)) / pl.col(spec["vol"]).sum()).alias("intr"),
            pl.col(spec["vol"]).mean().alias("tpg"))
        for r in by.iter_rows(named=True):
            for stat in ("share", "rate", "ypu", "tdr", "intr", "tpg"):
                v = r[stat]
                k.prior[(vt, r["pos"], stat)] = float(v) if v is not None and np.isfinite(v) else 0.0
        th = tf.filter(pl.col("played") & pl.col("season").is_between(season - 3, season - 1))
        k.league_team[vt] = float(th[spec["team"]].mean())
        sums = hist.group_by("game_id", "team").agg(pl.col(spec["vol"]).sum().alias("v"),
                                                    pl.col(spec["team"]).first().alias("t"))
        k.covered[vt] = float(sums["v"].sum() / sums["t"].sum())
        tv = tf.filter(pl.col("played") & pl.col("season").is_between(season - 4, season - 1))
        X = volume_design(tv, vt, k, team_ratings)
        k.volume_coef[vt], *_ = np.linalg.lstsq(X, tv[spec["team"]].to_numpy().astype(float), rcond=None)
    pp = pools.filter(pl.col("season").is_between(season - 3, season - 1))
    for (pool, pos), grp in pp.group_by(["pool", "pos"]):
        y = grp["yds"].to_numpy()
        if len(y) > 200 and y.mean() > 0:
            k.pools[(pool, pos)] = (y / y.mean()).astype(np.float32)
    return k


def prior(k: Constants, vt: str, stat: str) -> pl.Expr:
    m = {pos: k.prior.get((vt, pos, stat), 0.0) for pos in POSITIONS}
    return pl.col("pos").replace_strict(m, default=0.0, return_dtype=pl.Float64)


def parameters(feat: pl.DataFrame, k: Constants, cfg: Config) -> pl.DataFrame:
    df = feat
    a_td = cfg.a_eff * TD_SHRINK_MULT
    for vt, spec in VT.items():
        c = k.volume_coef[vt]
        units = pl.col(f"{vt}_S_s") if spec["td_units"] == "succ" else pl.col(f"{vt}_S_v")
        team_hat = (c[0] + c[1] * (pl.col(f"{vt}_T_v") + TEAM_PSEUDO_GAMES * k.league_team[vt]) / (pl.col(f"{vt}_T_n") + TEAM_PSEUDO_GAMES)
                    + c[2] * pl.col("team_spread") + c[3] * pl.col("total_line"))
        if cfg.team_ratings:
            for i, col in enumerate(TEAM_RT_COLS):
                team_hat = team_hat + c[4 + i] * pl.col(col).fill_null(0.0)
        rate_prior, ypu_prior = prior(k, vt, "rate"), prior(k, vt, "ypu")
        if cfg.arch and vt == "tgt" and "arch_rate" in df.columns:
            # receiver archetype replaces position as the shrinkage target (WR/TE with an archetype)
            rate_prior = pl.coalesce(pl.col("arch_rate"), rate_prior)
            ypu_prior = pl.coalesce(pl.col("arch_ypc"), ypu_prior)
        df = df.with_columns(
            ((pl.col(f"{vt}_S_v") + cfg.a_share * prior(k, vt, "share")) / (pl.col(f"{vt}_S_tv") + cfg.a_share)).alias(f"{vt}_share"),
            ((pl.col(f"{vt}_S_v") + cfg.a_tpg * prior(k, vt, "tpg")) / (pl.col("S_n") + cfg.a_tpg)).alias(f"{vt}_tpg"),
            team_hat.clip(5, 90).alias(f"{vt}_team_hat"),
            ((pl.col(f"{vt}_S_s") + cfg.a_eff * rate_prior) / (pl.col(f"{vt}_S_v") + cfg.a_eff)).alias(f"{vt}_rate"),
            ((pl.col(f"{vt}_S_y") + cfg.a_eff * ypu_prior) / (pl.col(f"{vt}_S_s") + cfg.a_eff)).alias(f"{vt}_ypu"),
            ((pl.col(f"{vt}_S_td") + a_td * prior(k, vt, "tdr")) / (units + a_td)).alias(f"{vt}_tdr"),
            (((pl.col(f"{vt}_S_int") if spec["int"] else pl.lit(0.0)) + a_td * prior(k, vt, "intr"))
             / (pl.col(f"{vt}_S_v") + a_td)).alias(f"{vt}_intr"),
        )
        share = pl.col(f"{vt}_share")
        if cfg.lam > 0:
            df = df.with_columns(pl.col(f"{vt}_share").sum().over("game_id", "team").alias(f"{vt}_share_sum"))
            share = share * (k.covered[vt] / pl.col(f"{vt}_share_sum")) ** cfg.lam
        mu = pl.col(f"{vt}_tpg") if cfg.model == "B1" else share * pl.col(f"{vt}_team_hat")
        df = df.with_columns(mu.alias(f"{vt}_mu"))
        if cfg.gam > 0:
            g = cfg.gam
            dsucc = pl.col(f"D_{vt}_s")
            d_rate = (dsucc + DEF_PSEUDO * prior(k, vt, "rate")) / (pl.col(f"D_{vt}_v") + DEF_PSEUDO)
            d_ypu = (pl.col(f"D_{vt}_y") + DEF_PSEUDO * prior(k, vt, "ypu")) / (dsucc + DEF_PSEUDO)
            d_tdr = (pl.col(f"D_{vt}_td") + DEF_PSEUDO * TD_SHRINK_MULT * prior(k, vt, "tdr")) / (
                pl.col(f"D_{vt}_u") + DEF_PSEUDO * TD_SHRINK_MULT)
            safe = lambda e, base: pl.when(base > 0).then((e / base) ** g).otherwise(1.0)
            df = df.with_columns(
                (pl.col(f"{vt}_rate") * safe(d_rate, prior(k, vt, "rate"))).clip(0.02, 0.98).alias(f"{vt}_rate"),
                (pl.col(f"{vt}_ypu") * safe(d_ypu, prior(k, vt, "ypu"))).alias(f"{vt}_ypu"),
                (pl.col(f"{vt}_tdr") * safe(d_tdr, prior(k, vt, "tdr"))).alias(f"{vt}_tdr"),
            )
    # ---- 3c matchup multipliers (columns from proplab.p3.matchup, prior-only)
    if cfg.g_depth and "depth_catch" in df.columns:
        rec = pl.col("pos").is_in(["WR", "TE", "RB"])
        df = df.with_columns(
            pl.when(rec).then(pl.col("tgt_rate") * pl.col("depth_catch") ** cfg.g_depth).otherwise(pl.col("tgt_rate"))
              .clip(0.02, 0.98).alias("tgt_rate"),
            pl.when(rec).then(pl.col("tgt_ypu") * pl.col("depth_ypc") ** cfg.g_depth).otherwise(pl.col("tgt_ypu")).alias("tgt_ypu"))
    if cfg.g_gap and "gap_ypc" in df.columns:
        df = df.with_columns(pl.when(pl.col("pos") == "RB").then(pl.col("car_ypu") * pl.col("gap_ypc") ** cfg.g_gap)
                               .otherwise(pl.col("car_ypu")).alias("car_ypu"))
    if cfg.g_press and "press_comp" in df.columns:
        qb, g = pl.col("pos") == "QB", cfg.g_press
        df = df.with_columns(
            pl.when(qb).then(pl.col("att_rate") * pl.col("press_comp") ** g).otherwise(pl.col("att_rate")).clip(0.02, 0.98).alias("att_rate"),
            pl.when(qb).then(pl.col("att_ypu") * pl.col("press_ypc") ** g).otherwise(pl.col("att_ypu")).alias("att_ypu"),
            pl.when(qb).then(pl.col("att_tdr") * pl.col("press_td") ** g).otherwise(pl.col("att_tdr")).alias("att_tdr"),
            pl.when(qb).then(pl.col("att_intr") * pl.col("press_int") ** g).otherwise(pl.col("att_intr")).alias("att_intr"))
    # ---- Phase 6 scheme multipliers (columns from proplab.p6.scheme, prior-only)
    if cfg.g_blitz and "blitz_comp" in df.columns:
        qb, g = pl.col("pos") == "QB", cfg.g_blitz
        df = df.with_columns(
            pl.when(qb).then(pl.col("att_rate") * pl.col("blitz_comp") ** g).otherwise(pl.col("att_rate")).clip(0.02, 0.98).alias("att_rate"),
            pl.when(qb).then(pl.col("att_ypu") * pl.col("blitz_ypc") ** g).otherwise(pl.col("att_ypu")).alias("att_ypu"),
            pl.when(qb).then(pl.col("att_tdr") * pl.col("blitz_td") ** g).otherwise(pl.col("att_tdr")).alias("att_tdr"),
            pl.when(qb).then(pl.col("att_intr") * pl.col("blitz_int") ** g).otherwise(pl.col("att_intr")).alias("att_intr"))
    if cfg.g_box and "box_ypc" in df.columns:
        df = df.with_columns(pl.when(pl.col("pos") == "RB").then(pl.col("car_ypu") * pl.col("box_ypc") ** cfg.g_box)
                               .otherwise(pl.col("car_ypu")).alias("car_ypu"))
    if cfg.g_wx and "wx_m" in df.columns:
        f = pl.col("wx_m") ** (0.5 * cfg.g_wx)
        df = df.with_columns((pl.col("att_rate") * f).clip(0.02, 0.98).alias("att_rate"), (pl.col("att_ypu") * f).alias("att_ypu"),
                             (pl.col("tgt_rate") * f).clip(0.02, 0.98).alias("tgt_rate"), (pl.col("tgt_ypu") * f).alias("tgt_ypu"))
    if cfg.g_pa and "pa_ypc" in df.columns:
        df = df.with_columns(pl.when(pl.col("pos") == "QB").then(pl.col("att_ypu") * pl.col("pa_ypc") ** cfg.g_pa)
                               .otherwise(pl.col("att_ypu")).alias("att_ypu"))
    return df


def prepare(feat: pl.DataFrame, tf: pl.DataFrame, pools: pl.DataFrame, season: int, cfg: Config):
    """Parameters for every row plus calibration/dispersion fit on earlier seasons."""
    k = fit_constants(feat, tf, pools, season, team_ratings=cfg.team_ratings)
    params = parameters(feat, k, cfg)
    train = params.filter(pl.col("played") & pl.col("season").is_between(season - 3, season - 1))
    from proplab.p1.model import fit_dispersion
    for (pos, vt) in ELIG_RULES:
        t = train.filter(eligible_expr(pos, vt))
        vol = VT[vt]["vol"]
        if t.height < 50:
            k.calibration[(vt, pos)], k.dispersion[(vt, pos)], k.exit_rate[(vt, pos)] = 1.0, 10.0, 0.0
            continue
        k.calibration[(vt, pos)] = float(t[vol].sum() / t[f"{vt}_mu"].sum())
        mu = t[f"{vt}_mu"].to_numpy() * k.calibration[(vt, pos)]
        y = t[vol].to_numpy().astype(float)
        # Early-exit component (QB only): starts far below expectation (injury, benching).
        # For RB/WR/TE, low-volume games are ordinary committee variance, which the
        # negative binomial already covers; on 2024 the component made their ranges too wide.
        exited = (y < EXIT_CUTOFF * mu) if pos == "QB" else np.zeros(len(y), dtype=bool)
        pi = float(exited.mean())
        k.exit_rate[(vt, pos)] = pi
        full_mu = mu / ((1 - pi) + EXIT_MEAN_FRACTION * pi)
        k.dispersion[(vt, pos)] = fit_dispersion(full_mu[~exited], y[~exited])
    for vt in VT:
        f = pl.col("pos").replace_strict({p: k.calibration.get((vt, p), 1.0) for p in POSITIONS},
                                         default=1.0, return_dtype=pl.Float64)
        params = params.with_columns(pl.col(f"{vt}_mu") * f)
    return params, k


# ---------------------------------------------------------------- simulation

def simulate_vt(rows: pl.DataFrame, vt: str, k: Constants, n: int, rng) -> dict:
    spec = VT[vt]
    m = rows.height
    pos = rows["pos"].to_numpy()
    r = np.array([k.dispersion.get((vt, p), 10.0) for p in pos])
    pi = np.array([k.exit_rate.get((vt, p), 0.0) for p in pos])
    # mixture: full game with prob 1-pi, early exit with prob pi; the overall mean stays mu
    mu = np.clip(rows[f"{vt}_mu"].to_numpy(), 0.02, None) / ((1 - pi) + EXIT_MEAN_FRACTION * pi)
    vol = rng.negative_binomial(r[:, None], (r / (r + mu))[:, None], size=(m, n))
    exits = rng.uniform(size=(m, n)) < pi[:, None]
    vol = np.where(exits, np.floor(vol * rng.uniform(0, EXIT_CUTOFF, size=(m, n))), vol).astype(np.int64)
    succ = rng.binomial(vol, np.clip(rows[f"{vt}_rate"].to_numpy(), 0, 1)[:, None]) if spec["succ"] else vol
    yds = np.zeros((m, n), dtype=np.float32)
    ypu = rows[f"{vt}_ypu"].to_numpy()
    for p in np.unique(pos):
        key = (spec["pool"], "QB" if spec["pool"] == "completion" else p)
        pool = k.pools.get(key)
        if pool is None:
            continue
        idx = np.where(pos == p)[0]
        counts = succ[idx].ravel()
        draws = rng.choice(pool, size=int(counts.sum()))
        scale = np.repeat(np.repeat(ypu[idx], n), counts).astype(np.float32)
        cs = np.concatenate([[0.0], np.cumsum(draws * scale, dtype=np.float64)])
        ends = np.cumsum(counts)
        yds[idx] = (cs[ends] - cs[ends - counts]).reshape(len(idx), n)
    units = succ if spec["td_units"] == "succ" else vol
    td = rng.binomial(units, np.clip(rows[f"{vt}_tdr"].to_numpy(), 0, 1)[:, None])
    out = {"vol": vol, "succ": succ, "yds": yds, "td": td}
    if spec["int"]:
        out["int"] = rng.binomial(vol, np.clip(rows[f"{vt}_intr"].to_numpy(), 0, 1)[:, None])
    return out


def market_draws(market: str, rows: pl.DataFrame, k: Constants, n: int, rng) -> np.ndarray:
    pos, vt, qty, _ = ALL_MARKETS[market]
    if market == "anytime_td":
        car = simulate_vt(rows, "car", k, n, rng)
        tgt = simulate_vt(rows, "tgt", k, n, rng)
        return ((car["td"] + tgt["td"]) >= 1).astype(np.float32)
    return simulate_vt(rows, vt, k, n, rng)[qty].astype(np.float32)


def outcome(market: str, rows: pl.DataFrame) -> np.ndarray:
    if market == "anytime_td":
        return ((rows["rush_tds"] + rows["rec_tds"]) >= 1).cast(pl.Float64).to_numpy()
    return rows[OUTCOME[market]].cast(pl.Float64).to_numpy()


def ece_null95(probs: np.ndarray, rng, n: int = 300) -> float:
    """95th percentile of ECE if outcomes really were drawn from these probabilities:
    the most miscalibration that noise alone produces at this sample size."""
    return float(np.percentile([ece(probs, (rng.uniform(size=len(probs)) < probs).astype(float))
                                for _ in range(n)], 95))


def coverage_tolerance(n_rows: int) -> float:
    return max(0.03, 1.96 * np.sqrt(0.16 / max(n_rows, 1)))


def score(market: str, draws: np.ndarray, y: np.ndarray, rng) -> dict:
    lines = ALL_MARKETS[market][3]
    probs = np.concatenate([(draws > L).mean(axis=1) for L in lines])
    outs = np.concatenate([(y > L).astype(float) for L in lines])
    brier_rows = ((draws > lines[0]).mean(axis=1) - (y > lines[0])) ** 2
    out = {"rows": int(len(y)), "brier": float(((probs - outs) ** 2).mean()), "ece": ece(probs, outs),
           "ece_null95": ece_null95(probs, rng), "coverage_tol": coverage_tolerance(len(y)),
           "mean_pred": float(draws.mean()), "mean_actual": float(y.mean())}
    if market == "anytime_td":
        p = np.clip(draws.mean(axis=1), 1e-6, 1 - 1e-6)
        out["crps_rows"] = brier_rows  # proper score used for the gate
        out["crps"] = float(brier_rows.mean())
        out["log_loss"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        return out
    crps = crps_samples(draws, y)
    pit = randomized_pit(draws, y, rng)
    out.update({"crps_rows": crps, "crps": float(crps.mean()),
                "mae_median": float(np.abs(np.median(draws, axis=1) - y).mean()),
                "coverage80": float(((pit >= 0.1) & (pit <= 0.9)).mean())})
    return out


def run_market(feat, tf, pools, season: int, market: str, cfg_by_vt: dict, n_sims: int, seed: int,
               weeks=None, cache: dict | None = None, keep: bool = False) -> dict:
    """Score one market for one season. cfg_by_vt maps 'tgt'/'car'/'att' -> Config; a key equal to
    the market name overrides its volume type's config (used for the WR/TE markets)."""
    vt = ALL_MARKETS[market][1]
    cfg_key = ("car", "tgt") if vt == "both" else (vt,)
    preps = {}
    for v in cfg_key:
        c = cfg_by_vt.get(market, cfg_by_vt[v]) if vt != "both" else cfg_by_vt[v]
        ck = (season, c)
        if cache is not None and ck in cache:
            preps[v] = cache[ck]
        else:
            preps[v] = prepare(feat[c.half_life], tf, pools, season, c)
            if cache is not None:
                cache[ck] = preps[v]
    base_params, k = preps[cfg_key[0]]
    rows = base_params.filter((pl.col("season") == season) & pl.col("played") & market_eligible(market))
    if weeks:
        rows = rows.filter(pl.col("week").is_between(*weeks))
    rows = rows.sort("game_id", "gsis_id")
    if vt == "both":  # take tgt parameters from the tgt config
        tp, tk = preps["tgt"]
        tcols = [c for c in tp.columns if c.startswith("tgt_")]
        rows = rows.drop(tcols).join(tp.select("game_id", "gsis_id", *tcols), on=["game_id", "gsis_id"])
        pick = lambda d: {kk: v for kk, v in d.items() if kk[0] == "tgt"}
        k = Constants(prior=k.prior, pools={**k.pools, **tk.pools},
                      calibration={**k.calibration, **pick(tk.calibration)},
                      dispersion={**k.dispersion, **pick(tk.dispersion)},
                      exit_rate={**k.exit_rate, **pick(tk.exit_rate)})
        rows = rows.sort("game_id", "gsis_id")
    rng = np.random.default_rng(seed)
    draws = market_draws(market, rows, k, n_sims, rng)
    y = outcome(market, rows)
    res = score(market, draws, y, np.random.default_rng(seed + 1))
    res["game_ids"] = rows["game_id"].to_numpy()
    res["keys"] = (rows["gsis_id"] + "|" + rows["game_id"]).to_numpy()
    if keep:
        res.update(frame=rows, draws=draws)
    return res
