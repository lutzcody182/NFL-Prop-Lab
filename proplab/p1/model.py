"""Phase 1 models (B1, B2, M1) and the shared simulation + scoring.

Pipeline for one evaluation season S:
  1. features(): weighted sums of each player's / team's / defense's PREVIOUS games
  2. fit_constants(): priors, team-volume regression, dispersion, catch-yard shapes,
     using seasons before S only
  3. parameters(): per player-game target mean, catch rate, yards per catch
  4. simulate(): 2,000 draws of (receptions, yards) per player-game
  5. score(): CRPS, MAE, randomized-PIT coverage, Brier/ECE at fixed lines
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from proplab.p1.data import decay_for, ewm_prior_sums

REC_LINES = [2.5, 3.5, 4.5, 5.5, 6.5]
YDS_LINES = [24.5, 39.5, 54.5, 69.5, 84.5]
EVAL_POS = ("WR", "TE")
TEAM_HALF_LIFE = 8
DEF_HALF_LIFE = 16
TEAM_PSEUDO_GAMES = 4.0
DEF_PSEUDO = 60.0
ELIG_HALF_LIFE = 8          # fixed by pre-registration
ELIG_MIN_GAMES = 3
ELIG_MIN_SHARE = 0.05


@dataclass(frozen=True)
class Config:
    model: str = "M1"            # B1 | B2 | M1
    half_life: float = 8.0       # player history, in games
    a_share: float = 40.0        # pseudo team-targets for target share
    a_catch: float = 20.0        # pseudo targets for catch rate
    a_ypr: float = 20.0          # pseudo receptions for yards per catch
    a_tpg: float = 4.0           # pseudo games for B1 targets per game
    redistribute: float = 1.0    # M1 lambda: 0 = off, 1 = full renormalization
    opponent: float = 1.0        # M1 gamma: 0 = off, 1 = full defensive factor
    calibrate: bool = True       # per-position mean calibration fit on earlier seasons (all models)

    def label(self) -> str:
        if self.model == "M1":
            return f"M1(h={self.half_life:g},as={self.a_share:g},ac={self.a_catch:g},ay={self.a_ypr:g},lam={self.redistribute:g},gam={self.opponent:g})"
        if self.model == "B2":
            return f"B2(h={self.half_life:g},as={self.a_share:g},ac={self.a_catch:g},ay={self.a_ypr:g})"
        return f"B1(h={self.half_life:g},ag={self.a_tpg:g},ac={self.a_catch:g},ay={self.a_ypr:g})"


# ---------------------------------------------------------------- features

def player_features(universe: pl.DataFrame, team_games: pl.DataFrame, half_life: float) -> pl.DataFrame:
    """Weighted sums over each player's previous *played* active games."""
    u = universe.join(team_games.select("game_id", "team", "team_targets"), on=["game_id", "team"], how="left")
    u = u.sort("gsis_id", "kickoff_utc")
    keys = u["gsis_id"].to_numpy()
    played = u["played"].to_numpy().astype(float)
    d = decay_for(half_life)
    cols = {
        "S_n": played,
        "S_tgt": u["targets"].to_numpy() * played,
        "S_rec": u["receptions"].to_numpy() * played,
        "S_yds": u["rec_yds"].to_numpy() * played,
        "S_tt": u["team_targets"].fill_null(0).to_numpy() * played,
    }
    out = {k: ewm_prior_sums(keys, v.astype(float), d) for k, v in cols.items()}
    # eligibility uses a fixed half-life and an unweighted game count (pre-registered)
    d8 = decay_for(ELIG_HALF_LIFE)
    e_tgt = ewm_prior_sums(keys, cols["S_tgt"].astype(float), d8)
    e_tt = ewm_prior_sums(keys, cols["S_tt"].astype(float), d8)
    n_prev = ewm_prior_sums(keys, played, 1.0)
    return u.with_columns(
        *[pl.Series(k, v) for k, v in out.items()],
        pl.Series("n_prev", n_prev),
        pl.Series("elig_share", np.divide(e_tgt, e_tt, out=np.zeros_like(e_tgt), where=e_tt > 0)),
    )


def team_features(team_games: pl.DataFrame) -> pl.DataFrame:
    t = team_games.sort("team", "kickoff_utc")
    keys = t["team"].to_numpy()
    played = t["played"].to_numpy().astype(float)
    d = decay_for(TEAM_HALF_LIFE)
    return t.with_columns(
        pl.Series("T_n", ewm_prior_sums(keys, played, d)),
        pl.Series("T_tt", ewm_prior_sums(keys, t["team_targets"].to_numpy() * played, d)),
    )


def defense_features(universe: pl.DataFrame, team_games: pl.DataFrame) -> pl.DataFrame:
    """What each defense allowed to WR/TE/RB in its previous games."""
    allowed = (universe.filter(pl.col("played"))
               .group_by("game_id", "opp", "pos")
               .agg(pl.col("targets").sum().alias("tgt"), pl.col("receptions").sum().alias("rec"),
                    pl.col("rec_yds").sum().alias("yds")))
    # one row per defense-game-position, including future games (zeros), sorted in time
    grid = (team_games.select("game_id", pl.col("opp").alias("defense"), "kickoff_utc", "played")
            .join(pl.DataFrame({"pos": ["WR", "TE", "RB"]}), how="cross")
            .join(allowed.rename({"opp": "defense"}), on=["game_id", "defense", "pos"], how="left")
            .fill_null(0)
            .sort("defense", "pos", "kickoff_utc"))
    keys = (grid["defense"] + "|" + grid["pos"]).to_numpy()
    played = grid["played"].to_numpy().astype(float)
    d = decay_for(DEF_HALF_LIFE)
    return grid.with_columns(
        *[pl.Series(f"D_{c}", ewm_prior_sums(keys, grid[c].to_numpy().astype(float) * played, d))
          for c in ("tgt", "rec", "yds")]
    ).select("game_id", "defense", "pos", "D_tgt", "D_rec", "D_yds")


def assemble(universe, team_games, half_life: float) -> pl.DataFrame:
    pf = player_features(universe, team_games, half_life)
    tf = team_features(team_games).select("game_id", "team", "T_n", "T_tt", "team_spread", "total_line")
    out =(pf.join(tf, on=["game_id", "team"], how="left")
             .join(defense_features(universe, team_games),
                   left_on=["game_id", "opp", "pos"], right_on=["game_id", "defense", "pos"], how="left"))
    return out


# ---------------------------------------------------------------- constants

@dataclass
class Constants:
    prior_share: dict
    prior_catch: dict
    prior_ypr: dict
    prior_tpg: dict
    league_team_targets: float
    covered_share: float          # share of team targets going to WR/TE/RB
    volume_coef: np.ndarray       # [intercept, ewma, team_spread, total]
    season_targets: dict = field(default_factory=dict)   # pos -> targets per season (3-season mean)
    calibration: dict = field(default_factory=dict)      # pos -> mean-calibration factor
    dispersion: dict = field(default_factory=dict)       # (model, pos) -> r
    catch_yards: dict = field(default_factory=dict)      # pos -> normalized draws (mean 1)


def fit_constants(feat: pl.DataFrame, team_feat: pl.DataFrame, catch_yards: pl.DataFrame, season: int) -> Constants:
    hist = feat.filter((pl.col("season") < season) & pl.col("played"))
    recent = hist.filter(pl.col("season") >= season - 3)
    by = recent.group_by("pos").agg(
        (pl.col("targets").sum() / pl.col("team_targets").sum()).alias("share"),
        (pl.col("receptions").sum() / pl.col("targets").sum()).alias("catch"),
        (pl.col("rec_yds").sum() / pl.col("receptions").sum()).alias("ypr"),
        pl.col("targets").mean().alias("tpg"),
        (pl.col("targets").sum() / pl.col("season").n_unique()).alias("season_targets"))
    pick = lambda c: dict(zip(by["pos"].to_list(), by[c].to_list()))

    th = team_feat.filter((pl.col("season") < season) & (pl.col("season") >= season - 3) & pl.col("played"))
    league_tt = float(th["team_targets"].mean())
    tg_sums = recent.group_by("game_id", "team").agg(pl.col("targets").sum().alias("t"),
                                                       pl.col("team_targets").first().alias("tt"))
    covered = float(tg_sums["t"].sum() / tg_sums["tt"].sum())  # ratio of totals: safe if a team-game has 0

    # team volume regression on prior seasons (features are as-of each game)
    tv = team_feat.filter((pl.col("season") < season) & (pl.col("season") >= season - 4) & pl.col("played"))
    ew = ((tv["T_tt"] + TEAM_PSEUDO_GAMES * league_tt) / (tv["T_n"] + TEAM_PSEUDO_GAMES)).to_numpy()
    X = np.column_stack([np.ones(len(tv)), ew, tv["team_spread"].to_numpy(), tv["total_line"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, tv["team_targets"].to_numpy().astype(float), rcond=None)

    cy = {}
    for pos in ("WR", "TE", "RB"):
        y = catch_yards.filter((pl.col("pos") == pos) & (pl.col("season") < season)
                               & (pl.col("season") >= season - 3))["yds"].to_numpy()
        cy[pos] = (y / y.mean()).astype(np.float32)

    return Constants(pick("share"), pick("catch"), pick("ypr"), pick("tpg"), league_tt, covered, coef,
                     season_targets=pick("season_targets"), catch_yards=cy)


# ---------------------------------------------------------------- parameters

def parameters(feat: pl.DataFrame, k: Constants, cfg: Config) -> pl.DataFrame:
    """Adds mu_t (expected targets), p_catch, ypr for every row."""
    pos = pl.col("pos")
    prior = lambda d: pos.replace_strict(d, default=None, return_dtype=pl.Float64)

    # Catch-rate prior that tracks the current season (prior weeks only). Earlier
    # seasons count as one season's worth of targets at that position.
    base_t = prior(k.season_targets)
    feat =(feat.join(in_season_catch_prior(feat, k), on=["season", "week", "pos"], how="left")
                .with_columns(pl.col("r_sofar").fill_null(0), pl.col("t_sofar").fill_null(0)))
    catch_prior = ((prior(k.prior_catch) * base_t + pl.col("r_sofar")) / (base_t + pl.col("t_sofar")))

    df = feat.with_columns(
        ((pl.col("S_tgt") + cfg.a_share * prior(k.prior_share)) / (pl.col("S_tt") + cfg.a_share)).alias("share"),
        ((pl.col("S_rec") + cfg.a_catch * catch_prior) / (pl.col("S_tgt") + cfg.a_catch)).alias("p_catch"),
        ((pl.col("S_yds") + cfg.a_ypr * prior(k.prior_ypr)) / (pl.col("S_rec") + cfg.a_ypr)).alias("ypr"),
        ((pl.col("S_tgt") + cfg.a_tpg * prior(k.prior_tpg)) / (pl.col("S_n") + cfg.a_tpg)).alias("tpg"),
        ((pl.col("T_tt") + TEAM_PSEUDO_GAMES * k.league_team_targets) / (pl.col("T_n") + TEAM_PSEUDO_GAMES)).alias("team_ewma"),
    )
    c = k.volume_coef
    df = df.with_columns(
        (c[0] + c[1] * pl.col("team_ewma") + c[2] * pl.col("team_spread") + c[3] * pl.col("total_line"))
        .clip(15, 60).alias("team_hat"))

    if cfg.model == "B1":
        return df.with_columns(pl.col("tpg").alias("mu_t"))

    share = pl.col("share")
    if cfg.model == "M1" and cfg.redistribute > 0:
        # Renormalize shares across the players actually active for this team-game.
        df = df.with_columns(pl.col("share").sum().over("game_id", "team").alias("share_sum"))
        share = share * (k.covered_share / pl.col("share_sum")) ** cfg.redistribute
    df = df.with_columns((share * pl.col("team_hat")).alias("mu_t"))

    if cfg.model == "M1" and cfg.opponent > 0:
        def_catch = (pl.col("D_rec") + DEF_PSEUDO * prior(k.prior_catch)) / (pl.col("D_tgt") + DEF_PSEUDO)
        def_ypr = (pl.col("D_yds") + DEF_PSEUDO * prior(k.prior_ypr)) / (pl.col("D_rec") + DEF_PSEUDO)
        g = cfg.opponent
        df = df.with_columns(
            (pl.col("p_catch") * (def_catch / prior(k.prior_catch)) ** g).clip(0.02, 0.95).alias("p_catch"),
            (pl.col("ypr") * (def_ypr / prior(k.prior_ypr)) ** g).alias("ypr"),
        )
    return df


# ---------------------------------------------------------------- simulation

def fit_dispersion(mu: np.ndarray, y: np.ndarray) -> float:
    """MLE of the negative binomial size parameter r on a grid."""
    from scipy.special import gammaln
    best_r, best_ll = None, -np.inf
    mu = np.clip(mu, 0.05, None)
    for r in np.exp(np.linspace(np.log(0.5), np.log(200), 60)):
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
              + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))).sum()
        if ll > best_ll:
            best_r, best_ll = r, ll
    return float(best_r)


def simulate(params: pl.DataFrame, k: Constants, model: str, n_sims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (receptions, yards) draws, each shape (rows, n_sims)."""
    rng = np.random.default_rng(seed)
    n = params.height
    mu = np.clip(params["mu_t"].to_numpy(), 0.05, None)
    r = params["pos"].replace_strict({p: k.dispersion[(model, p)] for p in EVAL_POS}, default=10.0).to_numpy()
    p = r / (r + mu)
    tgt = rng.negative_binomial(r[:, None], p[:, None], size=(n, n_sims))
    rec = rng.binomial(tgt, params["p_catch"].to_numpy()[:, None])

    yds = np.zeros((n, n_sims), dtype=np.float32)
    ypr = params["ypr"].to_numpy()
    pos = params["pos"].to_numpy()
    for ps in EVAL_POS:
        rows = np.where(pos == ps)[0]
        if not len(rows):
            continue
        counts = rec[rows].ravel()
        total = int(counts.sum())
        draws = rng.choice(k.catch_yards[ps], size=total)
        scale = np.repeat(np.repeat(ypr[rows], n_sims), counts).astype(np.float32)
        cs = np.concatenate([[0.0], np.cumsum(draws * scale, dtype=np.float64)])
        ends = np.cumsum(counts)
        sums = cs[ends] - cs[ends - counts]
        yds[rows] = sums.reshape(len(rows), n_sims)
    return rec.astype(np.float32), yds


# ---------------------------------------------------------------- scoring

def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample-based CRPS per row: E|X - y| - 0.5 E|X - X'|."""
    x = np.sort(samples, axis=1)
    n = x.shape[1]
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    i = np.arange(1, n + 1)
    term2 = ((2 * i - n - 1) * x).sum(axis=1) / (n * n)
    return term1 - term2


def randomized_pit(samples: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    below = (samples < y[:, None]).mean(axis=1)
    at_or_below = (samples <= y[:, None]).mean(axis=1)
    return below + rng.uniform(size=len(y)) * (at_or_below - below)


def ece(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(prob[m].mean() - outcome[m].mean())
    return float(total)


def score_market(samples: np.ndarray, y: np.ndarray, lines: list[float], rng) -> dict:
    crps = crps_samples(samples, y)
    pit = randomized_pit(samples, y, rng)
    probs = np.concatenate([(samples > L).mean(axis=1) for L in lines])
    outs = np.concatenate([(y > L).astype(float) for L in lines])
    return {
        "crps_rows": crps,
        "crps": float(crps.mean()),
        "mae_median": float(np.abs(np.median(samples, axis=1) - y).mean()),
        "coverage80": float(((pit >= 0.1) & (pit <= 0.9)).mean()),
        "brier": float(((probs - outs) ** 2).mean()),
        "ece": ece(probs, outs),
        "mean_pred": float(samples.mean()),
        "mean_actual": float(y.mean()),
    }


# ---------------------------------------------------------------- one season

def eligible(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("pos").is_in(EVAL_POS) & (pl.col("n_prev") >= ELIG_MIN_GAMES)
                     & (pl.col("elig_share") >= ELIG_MIN_SHARE))


def run_season(feat: pl.DataFrame, team_feat: pl.DataFrame, catch_yards: pl.DataFrame, season: int,
               cfg: Config, n_sims: int = 2000, seed: int = 7, weeks: tuple[int, int] | None = None,
               keep_draws: bool = False) -> dict:
    k = fit_constants(feat, team_feat, catch_yards, season)
    params = parameters(feat, k, cfg)

    # Mean calibration + dispersion, both from earlier seasons' eligible rows,
    # using this model's own predictions.
    train = eligible(params.filter((pl.col("season") < season) & (pl.col("season") >= season - 3) & pl.col("played")))
    k.calibration = {}
    if cfg.calibrate:
        for ps in EVAL_POS:
            t = train.filter(pl.col("pos") == ps)
            k.calibration[ps] = float(t["targets"].sum() / t["mu_t"].sum())
        factor = pl.col("pos").replace_strict(k.calibration, default=1.0, return_dtype=pl.Float64)
        params = params.with_columns(pl.col("mu_t") * factor)
        train = train.with_columns(pl.col("mu_t") * factor)
    for ps in EVAL_POS:
        t = train.filter(pl.col("pos") == ps)
        k.dispersion[(cfg.model, ps)] = fit_dispersion(t["mu_t"].to_numpy(), t["targets"].to_numpy().astype(float))

    test = eligible(params.filter((pl.col("season") == season) & pl.col("played")))
    if weeks:
        test = test.filter(pl.col("week").is_between(*weeks))
    test = test.sort("game_id", "gsis_id")
    rec, yds = simulate(test, k, cfg.model, n_sims, seed)
    rng = np.random.default_rng(seed + 1)
    out = {
        "season": season, "config": cfg.label(), "rows": test.height,
        "receptions": score_market(rec, test["receptions"].to_numpy().astype(float), REC_LINES, rng),
        "yards": score_market(yds, test["rec_yds"].to_numpy().astype(float), YDS_LINES, rng),
        "game_ids": test["game_id"].to_numpy(),
        "keys": (test["gsis_id"] + "|" + test["game_id"]).to_numpy(),
        "dispersion": {p: k.dispersion[(cfg.model, p)] for p in EVAL_POS},
        "calibration": k.calibration,
    }
    if keep_draws:
        out.update(frame=test, rec_draws=rec, yds_draws=yds, constants=k)
    return out
