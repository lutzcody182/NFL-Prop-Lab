"""Phase 1 tests: features can't see the game they predict, scoring math is right,
and results are reproducible."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from proplab import db
from proplab.p1 import data, model as M

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def loaded():
    c = db.connect()
    u, tg, cy = data.load_universe(c), data.load_team_games(c), data.load_catch_yards(c)
    c.close()
    return u, tg, cy


def test_ewm_prior_sums_excludes_current_row():
    keys = np.array(["a", "a", "a", "b", "b"])
    vals = np.array([1.0, 2.0, 4.0, 10.0, 20.0])
    out = data.ewm_prior_sums(keys, vals, 0.5)
    assert out.tolist() == [0.0, 1.0, 2.0 + 0.5 * 1.0, 0.0, 10.0]


def test_prior_week_games_known_before_each_week_starts():
    """Features use prior weeks only; every prior-week game must be knowable
    before the first kickoff of the target week."""
    c = db.connect()
    (bad,) = c.execute("""
        WITH wk AS (SELECT season, week, min(kickoff_utc) AS first_kick FROM games
                    WHERE game_type = 'REG' GROUP BY season, week)
        SELECT count(*) FROM wk
        JOIN games g ON g.game_type = 'REG' AND (g.season < wk.season OR (g.season = wk.season AND g.week < wk.week))
        JOIN player_game pg ON pg.game_id = g.game_id
        WHERE pg.known_at > wk.first_kick""").fetchone()
    c.close()
    assert bad == 0


def test_features_unchanged_when_target_week_outcomes_change(loaded):
    """Perturbation test: scramble every outcome in one week. Features (and so
    predictions) for that week must not move; later weeks may."""
    u, tg, _ = loaded
    target = (pl.col("season") == 2025) & (pl.col("week") == 9)
    u2 = u.with_columns(
        pl.when(target).then(pl.col("targets") + 7).otherwise(pl.col("targets")).alias("targets"),
        pl.when(target).then(pl.col("receptions") + 5).otherwise(pl.col("receptions")).alias("receptions"),
        pl.when(target).then(pl.col("rec_yds") + 99).otherwise(pl.col("rec_yds")).alias("rec_yds"))
    tg2 = tg.with_columns(pl.when((pl.col("season") == 2025) & (pl.col("week") == 9))
                          .then(pl.col("team_targets") + 30).otherwise(pl.col("team_targets")).alias("team_targets"))
    cols = ["S_n", "S_tgt", "S_rec", "S_yds", "S_tt", "D_tgt", "D_rec", "D_yds", "n_prev", "elig_share"]
    key = ["gsis_id", "game_id"]
    f1 = M.assemble(u, tg, 4.0).filter(target).sort(key).select(key + cols)
    f2 = M.assemble(u2, tg2, 4.0).filter(target).sort(key).select(key + cols)
    assert f1.height > 300
    assert f1.equals(f2), "a feature for week 9 changed when week 9 outcomes changed: leakage"
    t1 = M.team_features(tg).filter((pl.col("season") == 2025) & (pl.col("week") == 9)).sort("game_id", "team")
    t2 = M.team_features(tg2).filter((pl.col("season") == 2025) & (pl.col("week") == 9)).sort("game_id", "team")
    assert t1.select("T_n", "T_tt").equals(t2.select("T_n", "T_tt"))
    # sanity: week 10 features DO move, so the test is able to detect a change
    later = (pl.col("season") == 2025) & (pl.col("week") == 10)
    g1 = M.assemble(u, tg, 4.0).filter(later).sort(key)["S_tgt"]
    g2 = M.assemble(u2, tg2, 4.0).filter(later).sort(key)["S_tgt"]
    assert not g1.equals(g2)


def test_constants_ignore_target_season(loaded):
    """Constants for season S must not change when season S outcomes change."""
    u, tg, cy = loaded
    u2 = u.with_columns(pl.when(pl.col("season") == 2025).then(pl.col("targets") * 3)
                        .otherwise(pl.col("targets")).alias("targets"))
    tf = M.team_features(tg)
    k1 = M.fit_constants(M.assemble(u, tg, 4.0), tf, cy, 2025)
    k2 = M.fit_constants(M.assemble(u2, tg, 4.0), tf, cy, 2025)
    assert k1.prior_share == k2.prior_share and k1.prior_tpg == k2.prior_tpg
    assert np.allclose(k1.volume_coef, k2.volume_coef)


def test_crps_matches_closed_form_for_normal():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, size=(1, 200_000))
    y = np.array([0.7])
    # closed form CRPS for N(0,1) at y: y(2Phi(y)-1) + 2phi(y) - 1/sqrt(pi)
    from scipy.stats import norm
    exact = y * (2 * norm.cdf(y) - 1) + 2 * norm.pdf(y) - 1 / np.sqrt(np.pi)
    assert abs(M.crps_samples(s, y)[0] - exact[0]) < 0.005


def test_randomized_pit_is_uniform_when_calibrated():
    rng = np.random.default_rng(1)
    mu = rng.uniform(1, 6, 4000)
    samples = rng.poisson(mu[:, None], size=(4000, 2000)).astype(float)
    y = rng.poisson(mu).astype(float)
    pit = M.randomized_pit(samples, y, rng)
    cover = ((pit >= 0.1) & (pit <= 0.9)).mean()
    assert 0.78 < cover < 0.82


def test_saved_results_reproduce():
    """Re-scoring 2025 with the locked configs reproduces the committed results."""
    from proplab.p1 import backtest
    saved = json.loads((ROOT / "phase1" / "results.json").read_text())["test_2025"]["models"]["M1"]
    c = db.connect()
    u, tg, cy = data.load_universe(c), data.load_team_games(c), data.load_catch_yards(c)
    cfg = backtest.load_configs()["M1"]
    r = M.run_season(M.assemble(u, tg, cfg.half_life), M.team_features(tg), cy, 2025, cfg,
                     n_sims=backtest.N_SIMS, seed=backtest.SEED)
    assert abs(r["receptions"]["crps"] - saved["receptions"]["crps"]) < 1e-9
    assert abs(r["yards"]["crps"] - saved["yards"]["crps"]) < 1e-9
