"""Phase 2 tests: fetch script (against a local fake server serving SYNTHETIC odds),
loader, player matching, market consensus, and bet-time leakage."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests" / "fixtures")]

import mock_odds_api as mock  # noqa: E402
from odds_backfill import TEAM_NAMES  # noqa: E402

from proplab import db, names  # noqa: E402
from proplab.odds import evaluate as ev, load  # noqa: E402
from proplab.p1 import backtest, data, model as M  # noqa: E402

KEY = "test-key"


@pytest.fixture(scope="module")
def server():
    world = mock.build_world(db.connect(), TEAM_NAMES)
    srv, state = mock.serve(world, KEY)
    yield srv, state
    srv.shutdown()


def backfill(server, raw: Path, *extra: str) -> subprocess.CompletedProcess:
    srv, _ = server
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "odds_backfill.py"), "--run", "--seasons", "2025", "--only-weeks", "9",
         "--base-url", f"http://127.0.0.1:{srv.server_port}", "--raw-dir", str(raw),
         "--now", "2026-10-14T00:00:00Z", *extra],
        env={**os.environ, "ODDS_API_KEY": KEY}, capture_output=True, text=True, check=True)


@pytest.fixture
def clean_odds_tables():
    yield
    load.main(["--raw-dir", str(ROOT / "data" / "odds" / "raw")])  # rebuild from the real (possibly empty) folder


# ---------------------------------------------------------------- fetch script

def test_budget_cap_stops_and_resume_completes(server, tmp_path):
    _, state = server
    start = state["used"]
    out = backfill(server, tmp_path, "--max-credits", "200")
    assert "Credit limit reached" in out.stdout
    assert state["used"] - start <= 200
    n_partial = len(list(tmp_path.glob("2025/*.json")))
    backfill(server, tmp_path, "--max-credits", "100000")
    n_full = len(list(tmp_path.glob("2025/*.json")))
    assert n_partial < n_full
    before = state["used"]
    out = backfill(server, tmp_path)
    assert state["used"] == before, "a re-run must not re-download anything"
    assert "0 credits used" in out.stdout


def test_key_is_never_written_to_disk(server, tmp_path):
    backfill(server, tmp_path)
    for f in tmp_path.rglob("*.json"):
        assert KEY not in f.read_text()


# ---------------------------------------------------------------- loader + evaluation

def test_pipeline_end_to_end_on_synthetic_odds(server, tmp_path, clean_odds_tables):
    backfill(server, tmp_path)
    out = load.main(["--raw-dir", str(tmp_path)])
    checks = {c["name"]: c for c in out["checks"]}
    assert checks["snapshots_before_cutoff"]["passed"]
    assert out["match"]["matched"] > 0
    # every unmatched row is the one invented name the fake server adds
    unmatched = (tmp_path.parent / "unmatched.csv").read_text().splitlines()[1:]
    assert [r.split(",")[0] for r in unmatched] == ["Synthetic Nobody"]

    c = db.connect()
    (nl,) = c.execute("SELECT count(*) FROM prop_lines").fetchone()
    assert nl == out["prop_rows"] - 0, "rows must be committed, not rolled back"
    c.close()

    res = ev.main(["--out", str(tmp_path / "results.json")])
    rec = res["markets"]["receptions"]["close"]["2025"]
    assert rec["rows"] > 100
    assert 0 < rec["log_loss"]["market"] < 1 and 0 < rec["log_loss"]["model"] < 1


def test_spread_sign_home_perspective(tmp_path):
    f = tmp_path / "2025" / "X_bet.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"request": {"game_id": "X", "event_id": "e"}, "response": {
        "timestamp": "2025-01-01T00:00:00Z",
        "data": {"home_team": "Buffalo Bills", "bookmakers": [{"key": "b", "markets": [
            {"key": "spreads", "outcomes": [{"name": "Buffalo Bills", "point": -3.5, "price": 1.9},
                                            {"name": "Miami Dolphins", "point": 3.5, "price": 1.9}]}]}]}}}))
    _, _, games = load.parse_odds_file(f)
    home = [g for g in games if g["name"] == "Buffalo Bills"][0]
    assert -float(home["point"]) == 3.5  # the loader stores -point: home favored by 3.5 -> +3.5


# ---------------------------------------------------------------- matching + consensus

@pytest.mark.parametrize("raw,expected", [
    ("D.J. Moore", "dj moore"), ("Amon-Ra St. Brown", "amon ra st brown"),
    ("Marvin Harrison Jr.", "marvin harrison"), ("Ja'Marr Chase", "jamarr chase"), ("Kenneth Walker III", "kenneth walker"),
])
def test_name_normalization(raw, expected):
    assert names.normalize(raw) == expected


def test_consensus_uses_modal_line_and_lower_tie():
    rows = pl.DataFrame({
        "game_id": ["g"] * 5, "gsis_id": ["p"] * 5, "market": ["player_receptions"] * 5,
        "snapshot_tag": ["close"] * 5, "snapshot_ts": [None] * 5,
        "book": ["a", "b", "c", "d", "e"], "line": [4.5, 4.5, 5.5, 5.5, 3.5],
        "over_price": [1.80, 2.00, 2.10, 2.20, 1.50], "under_price": [2.00, 1.80, 1.70, 1.65, 2.50]})
    c = ev.consensus(rows)
    assert c.height == 1 and c["line"][0] == 4.5 and c["n_books"][0] == 2
    from proplab.odds import oddsmath as om
    expected = (om.novig_multiplicative(1.80, 2.00) + om.novig_multiplicative(2.00, 1.80)) / 2
    assert c["p_mult"][0] == pytest.approx(expected)
    assert c["best_over"][0] == 2.00


# ---------------------------------------------------------------- bet-time leakage

def test_bet_time_model_ignores_gameday_and_closing_information():
    """Changing week-9 actives, outcomes, or closing game lines must not change the
    bet-time predictions for week 9; changing the bet-time lines must."""
    conn = db.connect()
    u, tg, cy = data.load_universe(conn), data.load_team_games(conn), data.load_catch_yards(conn)
    home = dict(conn.execute("SELECT game_id, home_team FROM games WHERE season >= 2023").fetchall())
    conn.close()
    cfg = backtest.load_configs()["M1"]
    target = (pl.col("season") == 2025) & (pl.col("week") == 9)
    games9 = u.filter(target)["game_id"].unique().to_list()
    keys = u.filter(target & pl.col("pos").is_in(["WR", "TE"])).select("game_id", "gsis_id", "season", "week")
    bl = pl.DataFrame({"game_id": games9, "bet_spread": [2.5] * len(games9), "bet_total": [44.5] * len(games9)})

    def mu(u_, tg_, bl_):
        out = ev.model_bet(u_, tg_, cy, cfg, keys, bl_, home)
        rows, rec, _ = out[(2025, 9)]
        return rows.select("game_id", "gsis_id", "mu_t", "p_catch", "ypr").sort("game_id", "gsis_id"), rec

    base, rec0 = mu(u, tg, bl)
    assert base.height > 50

    u_changed = (u.filter(~(target & (pl.col("pos") == "WR") & (pl.col("targets") == 0)))  # drop some game-day actives
                  .with_columns(pl.when(target).then(pl.col("targets") + 9).otherwise(pl.col("targets")).alias("targets")))
    tg_changed = tg.with_columns(pl.when(pl.col("game_id").is_in(games9)).then(pl.col("team_spread") + 10)
                                 .otherwise(pl.col("team_spread")).alias("team_spread"))
    other, rec1 = mu(u_changed, tg_changed, bl)
    assert base.equals(other), "bet-time prediction moved when game-day or closing information changed"
    assert np.array_equal(rec0, rec1)

    moved, _ = mu(u, tg, bl.with_columns(pl.col("bet_total") + 10))
    assert not base["mu_t"].equals(moved["mu_t"]), "bet-time lines should affect the bet-time prediction"
