"""Point-in-time readers. Every model input must come through functions like
these, which only return rows whose known_at is at or before the cutoff."""
from __future__ import annotations

import datetime as dt

import polars as pl
import psycopg


def _frame(conn: psycopg.Connection, sql: str, params: tuple) -> pl.DataFrame:
    cur = conn.execute(sql, params)
    cols = [d.name for d in cur.description]
    return pl.DataFrame(cur.fetchall(), schema=cols, orient="row")


def prediction_cutoff(conn: psycopg.Connection, game_id: str, mode: str = "kickoff") -> dt.datetime:
    """Cutoff for predicting a game.

    mode="kickoff"  : just before kickoff (after inactives are known)
    mode="thursday" : Thursday 12:00 UTC of game week, before final injury reports
    """
    (kick,) = conn.execute("SELECT kickoff_utc FROM games WHERE game_id = %s", (game_id,)).fetchone()
    if mode == "kickoff":
        return kick - dt.timedelta(minutes=1)
    if mode == "thursday":
        days_back = (kick.weekday() - 3) % 7  # Thursday = 3
        thu = (kick - dt.timedelta(days=days_back)).replace(hour=12, minute=0, second=0, microsecond=0)
        return min(thu, kick - dt.timedelta(minutes=1))
    raise ValueError(mode)


def player_history(conn: psycopg.Connection, gsis_id: str, as_of: dt.datetime) -> pl.DataFrame:
    return _frame(conn, """
        SELECT pg.*, g.kickoff_utc FROM player_game pg JOIN games g USING (game_id)
        WHERE pg.gsis_id = %s AND pg.known_at <= %s ORDER BY g.kickoff_utc""", (gsis_id, as_of))


def team_plays(conn: psycopg.Connection, team: str, as_of: dt.datetime) -> pl.DataFrame:
    return _frame(conn, """
        SELECT * FROM plays WHERE (posteam = %s OR defteam = %s) AND known_at <= %s""", (team, team, as_of))


def injury_status(conn: psycopg.Connection, season: int, week: int, team: str, as_of: dt.datetime) -> pl.DataFrame:
    return _frame(conn, """
        SELECT gsis_id, position, report_status, practice_status FROM injury_reports
        WHERE season = %s AND week = %s AND team = %s AND known_at <= %s""", (season, week, team, as_of))


def roster(conn: psycopg.Connection, season: int, week: int, team: str, as_of: dt.datetime) -> pl.DataFrame:
    return _frame(conn, """
        SELECT gsis_id, position, depth_chart_position, status, status_abbr FROM roster_status
        WHERE season = %s AND week = %s AND team = %s AND known_at <= %s""", (season, week, team, as_of))
