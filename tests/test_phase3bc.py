"""3b/3c tests: ratings and matchup features use only earlier games; saved results reproduce."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from proplab import db
from proplab.p3 import engine as E
from proplab.p3 import matchup as MX
from proplab.ratings import player as PR
from proplab.ratings import team as TR

ROOT = Path(__file__).resolve().parents[1]
WK = (pl.col("season") == 2025) & (pl.col("week") == 9)


@pytest.fixture(scope="module")
def conn():
    c = db.connect()
    yield c
    c.close()


def test_team_ratings_ignore_the_week_being_rated(conn):
    plays, teams, _ = TR.setup(conn)
    changed = plays.with_columns(pl.when(WK).then(pl.col("epa") + 3.0).otherwise(pl.col("epa")).alias("epa"))
    (cut9,) = conn.execute("SELECT min(kickoff_utc) FROM games WHERE season = 2025 AND week = 9 AND game_type = 'REG'").fetchone()
    (cut10,) = conn.execute("SELECT min(kickoff_utc) FROM games WHERE season = 2025 AND week = 10 AND game_type = 'REG'").fetchone()
    for unit in TR.UNITS:
        a, b = TR.UnitData(plays, unit, teams), TR.UnitData(changed, unit, teams)
        f1, f2 = a.fit(TR.days(cut9), 32, 200), b.fit(TR.days(cut9), 32, 200)
        assert np.allclose(f1["off"], f2["off"]) and np.allclose(f1["def"], f2["def"]), "week-9 rating saw week-9 plays"
        g1, g2 = a.fit(TR.days(cut10), 32, 200), b.fit(TR.days(cut10), 32, 200)
        assert not np.allclose(g1["off"], g2["off"]), "test must be able to detect a change"


def test_player_ratings_ignore_the_game_being_rated(conn):
    pg = PR.player_games(conn)
    changed = pg.with_columns(pl.when(WK).then(pl.col("adj_sum") + 50).otherwise(pl.col("adj_sum")).alias("adj_sum"))
    key = ["gsis_id", "game_id"]
    a = PR.with_prior_sums(pg, 16).filter(WK).sort(key).select(key + ["W_n", "W_adj"])
    b = PR.with_prior_sums(changed, 16).filter(WK).sort(key).select(key + ["W_n", "W_adj"])
    assert a.height > 100 and a.equals(b)


def test_matchup_features_ignore_target_week(conn):
    D = E.load(conn)
    u, tg = D["u"], D["tg"]
    base = MX.build(conn, u, tg).sort("gsis_id", "game_id")

    # perturb week-9 plays inside the module by changing its play loader output
    orig = MX.load_plays
    try:
        w9 = pl.col("game_id").str.starts_with("2025_09")
        MX.load_plays = lambda c: orig(c).with_columns(  # change outcomes AND the bucket mixes / pressure in week 9
            pl.when(w9).then(pl.col("yds") + 25).otherwise(pl.col("yds")).alias("yds"),
            pl.when(w9).then(~pl.col("complete_pass")).otherwise(pl.col("complete_pass")).alias("complete_pass"),
            pl.when(w9).then(pl.col("air_yards") + 15).otherwise(pl.col("air_yards")).alias("air_yards"),
            pl.when(w9).then(pl.lit("end")).otherwise(pl.col("run_gap")).alias("run_gap"),
            pl.when(w9).then(~pl.col("qb_hit")).otherwise(pl.col("qb_hit")).alias("qb_hit"))
        changed = MX.build(conn, u, tg).sort("gsis_id", "game_id")
    finally:
        MX.load_plays = orig
    wk9 = pl.col("game_id").str.starts_with("2025_09")
    assert base.filter(wk9).equals(changed.filter(wk9)), "a week-9 matchup feature changed when week-9 plays changed"
    wk10 = pl.col("game_id").str.starts_with("2025_10")
    assert not base.filter(wk10)["depth_ypc"].equals(changed.filter(wk10)["depth_ypc"])


def test_team_rating_results_reproduce(conn):
    saved = json.loads((ROOT / "phase3" / "ratings_team_results.json").read_text())
    ch = TR.chosen()
    plays, teams, games = TR.setup(conn)
    ud = TR.UnitData(plays, "run", teams)
    g = TR.games_for(conn, games, [2025])
    adj = TR.evaluate(conn, ud, g, ch["run"]["chosen"]["H"], ch["run"]["chosen"]["lam"])
    assert TR.wmse(adj) == pytest.approx(saved["run"]["test_2025"]["wmse_adjusted"], rel=1e-3)
