"""Phase 7: injury actives, weather multiplier, new markets and kicker model behave as specified."""
import numpy as np
import polars as pl
import pytest

from proplab import db
from proplab.p3 import engine as E
from proplab.p7 import actives as A
from proplab.p7 import kicker as K
from proplab.p7 import weather as W


@pytest.fixture(scope="module")
def conn():
    c = db.connect()
    yield c
    c.close()


def test_new_market_outcomes():
    rows = pl.DataFrame({"rush_yds": [10.0, 0.0], "rec_yds": [25.0, 3.0], "rec_tds": [0, 2], "rush_tds": [1, 0], "completions": [20, 0]})
    assert E.outcome("rb_rush_rec_yds", rows).tolist() == [35.0, 3.0]
    assert E.outcome("rec_td", rows).tolist() == [0.0, 1.0]
    assert E.outcome("qb_completions", rows).tolist() == [20.0, 0.0]
    assert "rec_td" in E.BINARY and set(E.NEW_MARKETS) <= set(E.ALL_MARKETS)


def test_weather_multiplier_neutral_indoors_and_fit_uses_earlier_seasons(conn):
    t = W.with_priors(W.weather_terms(W.team_games(conn)))
    indoor = t.filter(~pl.col("roof").is_in(W.OUTDOOR))
    assert (indoor["w_ex"] == 0).all() and (indoor["cold"] == 0).all()
    b1 = W.fit(t, 2025)
    changed = t.with_columns(pl.when(pl.col("season") >= 2025).then(pl.col("yds") * 3).otherwise(pl.col("yds")).alias("yds"))
    assert W.fit(changed, 2025) == pytest.approx(b1), "the 2025 fit must not see 2025 games"
    assert b1[0] < 0 and b1[1] < 0


def test_kicker_rows_unique_and_deterministic(conn):
    a, b = K.load(conn), K.load(conn)
    assert a.equals(b)
    assert a.group_by("game_id", "team").len()["len"].max() == 1


def test_injury_removal_variant_is_subset_of_last_game(conn):
    D = E.load(conn)
    tg = D["tg"]
    targets = tg.filter((pl.col("season") == 2025) & (pl.col("week") == 10) & pl.col("played")).select(
        "game_id", "team", "season", "week", "kickoff_utc")
    inj = A.injury_out(conn, 2025)
    L = A.expected_actives(D["u"], tg, targets, inj, "L")
    R = A.expected_actives(D["u"], tg, targets, inj, "R")
    assert R.height <= L.height and R.join(L, on=["game_id", "team", "gsis_id"], how="anti").is_empty()
    # nothing about week 10 itself is used by L: last-game actives come from earlier games
    assert L.join(D["u"].filter((pl.col("season") == 2025) & (pl.col("week") == 10)).select("gsis_id", "game_id"),
                  on=["gsis_id", "game_id"]).height <= L.height


def test_kicker_points_simulation_matches_means():
    rows = pl.DataFrame({"mu_fg": [1.6, 2.0], "mu_pat": [2.5, 3.0]})
    d = K.simulate(rows, 20000, np.random.default_rng(0))
    assert d.mean(axis=1) == pytest.approx([3 * 1.6 + 2.5, 3 * 2.0 + 3.0], rel=0.02)
