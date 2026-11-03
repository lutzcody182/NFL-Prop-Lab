"""The next unplayed week and its games, from the schedule (games table).

The projection steps used to read the upcoming week from nflverse's weekly rosters, which are posted a day or
two into the week. Between Monday night and that posting there were no unplayed roster rows, and the run failed.
The schedule is known all season, so it is the source of truth for which games come next.
"""
from __future__ import annotations

import polars as pl


def next_week(conn, season: int) -> int:
    (w,) = conn.execute("""SELECT min(g.week) FROM games g LEFT JOIN game_results r USING (game_id)
                           WHERE g.season = %s AND g.game_type = 'REG' AND r.game_id IS NULL""", (season,)).fetchone()
    if w is None:
        raise SystemExit(f"No unplayed regular-season games left in {season}.")
    return int(w)


def week_games(conn, season: int, week: int) -> pl.DataFrame:
    """One row per team per game: game_id, kickoff_utc, team, opp, is_home, season, week."""
    cur = conn.execute("""
        SELECT g.game_id, g.kickoff_utc, s.team, s.opp, s.is_home, g.season, g.week
        FROM games g CROSS JOIN LATERAL (VALUES (g.home_team, g.away_team, true), (g.away_team, g.home_team, false)) AS s(team, opp, is_home)
        WHERE g.season = %s AND g.week = %s AND g.game_type = 'REG'""", (season, week))
    return pl.DataFrame(cur.fetchall(), schema=[d.name for d in cur.description], orient="row")
