"""Phase 10: new WR/TE features use only earlier games; defaults keep the locked model unchanged."""
import polars as pl
import pytest

from proplab import db
from proplab.p1 import model as M
from proplab.p3 import engine as E
from proplab.p10 import features as F


@pytest.fixture(scope="module")
def data():
    c = db.connect()
    D = E.load(c)
    yield c, D
    c.close()


def test_defaults_are_off():
    c = M.Config()
    assert c.role == c.qb == c.ay == 0.0 and "role" not in c.label()


def test_qb_change_ignores_target_week(data):
    _, D = data
    u = D["u"]
    base = F.qb_change(u).sort("game_id", "team")
    w9 = pl.col("game_id").str.starts_with("2025_09")
    changed_u = u.with_columns(pl.when(w9).then(pl.col("pass_yds") * 3).otherwise(pl.col("pass_yds")).alias("pass_yds"),
                               pl.when(w9).then(pl.col("completions") // 2).otherwise(pl.col("completions")).alias("completions"))
    changed = F.qb_change(changed_u).sort("game_id", "team")
    assert base.filter(w9).equals(changed.filter(w9))
    assert not base.filter(pl.col("game_id").str.starts_with("2025_10")).equals(changed.filter(pl.col("game_id").str.starts_with("2025_10")))


def test_air_yards_ignores_target_week(data, monkeypatch):
    conn, D = data
    base = F.air_yards(conn, D["u"]).sort("gsis_id", "game_id")
    orig = F._q
    monkeypatch.setattr(F, "_q", lambda c, sql: orig(c, sql).with_columns(
        pl.when(pl.col("game_id").str.starts_with("2025_09")).then(pl.col("ay") + 200).otherwise(pl.col("ay")).alias("ay")))
    changed = F.air_yards(conn, D["u"]).sort("gsis_id", "game_id")
    j = base.join(changed, on=["gsis_id", "game_id"], suffix="_c").with_columns(
        ((pl.col("ay_ypc") - pl.col("ay_ypc_c")).abs() > 1e-9).fill_null(False).alias("diff"))
    w9 = pl.col("game_id").str.starts_with("2025_09")
    assert j.filter(w9).height > 100 and not j.filter(w9)["diff"].any(), "week-9 feature saw week-9 air yards"
    assert j.filter(pl.col("game_id").str.starts_with("2025_10"))["diff"].any()
