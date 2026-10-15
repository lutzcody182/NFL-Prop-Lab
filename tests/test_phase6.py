"""Phase 6: scheme features use only earlier games (incl. league rates), and are neutral without history."""
import polars as pl
import pytest

from proplab import db
from proplab.p3 import engine as E
from proplab.p6 import scheme as SC

COLS = ["blitz_comp", "blitz_ypc", "blitz_td", "blitz_int", "box_ypc", "pa_ypc"]


@pytest.fixture(scope="module")
def data():
    conn = db.connect()
    D = E.load(conn)
    plays = SC.load_plays(conn)
    yield conn, D, plays
    conn.close()


def test_scheme_features_ignore_target_week_and_later(data):
    conn, D, plays = data
    u, tg = D["u"], D["tg"]
    base = SC.build(conn, u, tg, plays=plays).sort("gsis_id", "game_id")
    w9 = pl.col("game_id").str.starts_with("2025_09")
    changed_plays = plays.with_columns(
        pl.when(w9).then(pl.col("yds") + 25).otherwise(pl.col("yds")).alias("yds"),
        pl.when(w9).then(~pl.col("complete_pass")).otherwise(pl.col("complete_pass")).alias("complete_pass"),
        pl.when(w9).then(pl.col("n_blitzers").fill_null(0) + 3).otherwise(pl.col("n_blitzers")).alias("n_blitzers"),
        pl.when(w9).then(pl.lit(9)).otherwise(pl.col("n_defense_box")).alias("n_defense_box"),
        pl.when(w9).then(~pl.col("is_play_action")).otherwise(pl.col("is_play_action")).alias("is_play_action"))
    changed = SC.build(conn, u, tg, plays=changed_plays).sort("gsis_id", "game_id")
    keys = u.select("gsis_id", "game_id", "season", "week").unique()
    upto = (pl.col("season") < 2025) | ((pl.col("season") == 2025) & (pl.col("week") <= 9))
    a = base.join(keys, on=["gsis_id", "game_id"]).filter(upto).select("gsis_id", "game_id", *COLS)
    b = changed.join(keys, on=["gsis_id", "game_id"]).filter(upto).select("gsis_id", "game_id", *COLS)
    assert a.height > 10_000 and a.equals(b), "a feature for week 9 or earlier changed when week-9 plays changed"
    wk10 = pl.col("game_id").str.starts_with("2025_10")
    for c in ("blitz_ypc", "box_ypc", "pa_ypc"):
        assert not base.filter(wk10)[c].equals(changed.filter(wk10)[c]), f"test must be able to detect a change in {c}"


def test_scheme_features_neutral_before_charting(data):
    conn, D, plays = data
    out = SC.build(conn, D["u"], D["tg"], plays=plays).join(D["u"].select("gsis_id", "game_id", "season"), on=["gsis_id", "game_id"])
    early = out.filter(pl.col("season") <= 2022)
    assert early.height > 1000
    for c in COLS:
        assert (early[c] == 1.0).all(), f"{c} should be 1.0 when no prior-season league rates exist"


def test_config_defaults_reproduce_phase3a():
    c = E.Config()
    assert c.g_blitz == c.g_box == c.g_pa == 0.0 and "blitz" not in c.label()
