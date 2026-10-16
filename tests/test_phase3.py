"""Phase 3a tests: point-in-time features, constants from earlier seasons only,
the early-exit mixture keeps the mean, and the committed 2025 results reproduce."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from proplab import db
from proplab.p3 import engine as E
from proplab.p3 import run as R

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def D():
    return E.load(db.connect())


def test_features_ignore_target_week_outcomes(D):
    u, tg = D["u"], D["tg"]
    wk = (pl.col("season") == 2025) & (pl.col("week") == 9)
    stats = ["targets", "receptions", "rec_tds", "carries", "rush_tds", "attempts", "completions", "pass_tds", "ints"]
    u2 = u.with_columns(*[pl.when(wk).then(pl.col(c) + 3).otherwise(pl.col(c)).alias(c) for c in stats],
                        *[pl.when(wk).then(pl.col(c) + 50).otherwise(pl.col(c)).alias(c) for c in ("rec_yds", "rush_yds", "pass_yds")])
    tg2 = tg.with_columns(*[pl.when((pl.col("season") == 2025) & (pl.col("week") == 9)).then(pl.col(c) + 20)
                            .otherwise(pl.col(c)).alias(c) for c in ("team_targets", "team_carries", "team_attempts")])
    key = ["gsis_id", "game_id"]
    cols = [c for c in E.features(u, tg, 8.0).columns if c.split("_")[0] in ("tgt", "car", "att", "D", "S")
            or c in ("n_prev", "qb_top")]
    f1 = E.features(u, tg, 8.0).filter(wk).sort(key).select(key + cols)
    f2 = E.features(u2, tg2, 8.0).filter(wk).sort(key).select(key + cols)
    assert f1.height > 300  # 2025 week 9 has 387 active QB/RB/WR/TE rows (bye weeks)
    assert f1.equals(f2), "a week-9 feature changed when week-9 outcomes changed: leakage"
    later = (pl.col("season") == 2025) & (pl.col("week") == 10)
    assert not E.features(u, tg, 8.0).filter(later).sort(key)["car_S_v"].equals(
        E.features(u2, tg2, 8.0).filter(later).sort(key)["car_S_v"]), "test must be able to detect a change"


def test_constants_ignore_target_season(D):
    u, tg = D["u"], D["tg"]
    u2 = u.with_columns(pl.when(pl.col("season") == 2025).then(pl.col("carries") * 3).otherwise(pl.col("carries")).alias("carries"))
    tf = E.team_features(tg)
    k1 = E.fit_constants(E.features(u, tg, 8.0), tf, D["pools"], 2025)
    k2 = E.fit_constants(E.features(u2, tg, 8.0), tf, D["pools"], 2025)
    assert k1.prior == k2.prior and k1.covered == k2.covered


def test_exit_mixture_preserves_mean():
    rows = pl.DataFrame({"pos": ["QB"] * 400, "att_mu": [32.0] * 400, "att_rate": [0.65] * 400,
                         "att_ypu": [11.0] * 400, "att_tdr": [0.045] * 400, "att_intr": [0.022] * 400})
    k = E.Constants(dispersion={("att", "QB"): 12.0}, exit_rate={("att", "QB"): 0.06},
                    pools={("completion", "QB"): np.ones(1000, dtype=np.float32)})
    out = E.simulate_vt(rows, "att", k, 2000, np.random.default_rng(0))
    assert out["vol"].mean() == pytest.approx(32.0, rel=0.01)
    assert (out["vol"] < 0.35 * 32).mean() == pytest.approx(0.06, abs=0.01)


def test_saved_2025_results_reproduce(D):
    saved = json.loads((ROOT / "phase3" / "results.json").read_text())["splits"]["test_2025"]
    chosen = R.load_chosen()
    feat = {c.half_life: E.features(D["u"], D["tg"], c.half_life) for c in chosen.values()}
    tf = E.team_features(D["tg"])
    for m in ("rb_rush_yds", "qb_pass_yds"):
        r = E.run_market(feat, tf, D["pools"], 2025, m, chosen, R.N_TEST, R.SEED, cache={})
        assert r["crps"] == pytest.approx(saved[m]["models"]["chosen"]["crps"], rel=1e-3)
