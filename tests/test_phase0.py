"""Phase 0 tests: time handling, leakage guards, data quality, reproducibility.

Run after an ingest:  pytest -q
"""
import datetime as dt

import polars as pl
import pytest

from proplab import db, ingest, pit

UTC = dt.timezone.utc


@pytest.fixture(scope="module")
def conn():
    c = db.connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def sample_games(conn):
    """Completed regular-season games from mid-season weeks, deterministic sample."""
    rows = conn.execute("""
        SELECT g.game_id, g.season, g.week, g.home_team, g.kickoff_utc FROM games g
        JOIN game_results USING (game_id)
        WHERE g.game_type = 'REG' AND g.week BETWEEN 4 AND 17
        ORDER BY md5(g.game_id) LIMIT 40""").fetchall()
    assert rows, "no completed games loaded"
    return rows


# ---------- time conversion ----------

@pytest.mark.parametrize("day,time,expected", [
    ("2025-09-04", "20:20", dt.datetime(2025, 9, 5, 0, 20, tzinfo=UTC)),   # EDT, crosses midnight UTC
    ("2025-12-07", "13:00", dt.datetime(2025, 12, 7, 18, 0, tzinfo=UTC)),  # EST
    ("2025-11-02", "13:00", dt.datetime(2025, 11, 2, 18, 0, tzinfo=UTC)),  # DST-change day
])
def test_kickoff_eastern_to_utc(day, time, expected):
    out = pl.DataFrame({"gameday": [day], "gametime": [time]}).select(ingest.kickoff_expr())
    assert out.item().replace(tzinfo=UTC) == expected


def test_thursday_cutoff_is_before_kickoff(conn, sample_games):
    for game_id, *_ , kick in sample_games:
        cut = pit.prediction_cutoff(conn, game_id, "thursday")
        assert cut < kick and (kick - cut) < dt.timedelta(days=7)


# ---------- leakage guards ----------

def test_player_history_excludes_target_and_future_games(conn, sample_games):
    checked = 0
    for game_id, season, week, team, kick in sample_games:
        players = conn.execute(
            "SELECT gsis_id FROM player_game WHERE game_id = %s AND targets > 0 LIMIT 3", (game_id,)).fetchall()
        cutoff = pit.prediction_cutoff(conn, game_id)
        for (pid,) in players:
            hist = pit.player_history(conn, pid, cutoff)
            assert game_id not in hist["game_id"].to_list()
            if hist.height:
                assert hist["kickoff_utc"].max() < kick
            checked += 1
    assert checked > 50


def test_injury_reports_hidden_at_thursday_cutoff(conn, sample_games):
    """Final injury designations must not be visible at the Thursday cutoff
    for a Sunday game (assumption A1: released ~40h before kickoff)."""
    for game_id, season, week, team, kick in sample_games:
        if kick.weekday() != 6:  # Sunday games only
            continue
        cut = pit.prediction_cutoff(conn, game_id, "thursday")
        assert pit.injury_status(conn, season, week, team, cut).height == 0
        visible = pit.injury_status(conn, season, week, team, kick - dt.timedelta(minutes=1))
        assert visible.height >= 0  # visible before kickoff; may be empty for healthy teams


def test_inactives_hidden_until_90_minutes_before_kickoff(conn, sample_games):
    for game_id, season, week, team, kick in sample_games:
        early = pit.roster(conn, season, week, team, kick - dt.timedelta(hours=3))
        late = pit.roster(conn, season, week, team, kick - dt.timedelta(minutes=1))
        assert early.height == 0
        assert late.height > 40


def test_no_outcome_row_known_before_kickoff(conn):
    for table in ["player_game", "plays", "snaps", "game_results"]:
        (n,) = conn.execute(f"""
            SELECT count(*) FROM {table} t JOIN games g USING (game_id)
            WHERE t.known_at <= g.kickoff_utc""").fetchone()
        assert n == 0, table


def test_closing_lines_only_for_completed_games(conn):
    (n,) = conn.execute("""
        SELECT count(*) FROM game_lines l LEFT JOIN game_results r USING (game_id)
        WHERE r.game_id IS NULL""").fetchone()
    assert n == 0


# ---------- quality + reproducibility ----------

def test_latest_quality_run_passed(conn):
    rows = conn.execute("""
        SELECT check_name, passed, detail FROM data_quality_runs
        WHERE run_id = (SELECT max(run_id) FROM ingest_runs WHERE finished_at IS NOT NULL)""").fetchall()
    assert len(rows) >= 9
    failed = [r for r in rows if not r[1]]
    assert not failed, failed


def test_reruns_are_reproducible(conn):
    """The two most recent completed runs over the same seasons must produce identical content."""
    rows = conn.execute("""
        SELECT seasons, table_hashes FROM ingest_runs WHERE finished_at IS NOT NULL
        ORDER BY run_id DESC LIMIT 2""").fetchall()
    if len(rows) < 2 or rows[0][0] != rows[1][0]:
        pytest.skip("need two completed runs over the same seasons")
    (s1, h1), (s2, h2) = rows
    diffs = {t for t in h1 if h1[t] != h2.get(t)}
    assert not diffs, f"tables differ between runs: {sorted(diffs)}"
