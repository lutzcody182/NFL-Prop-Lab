"""Load the Phase 1 universe from Postgres.

One row per active (ACT) WR/TE/RB per regular-season team-game, with the outcome
(0 when the player had no recorded stat), the team's total targets in that game,
the game's closing spread/total, and what the opposing defense allowed by position.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from proplab import db


def _q(conn, sql: str, params: tuple = ()) -> pl.DataFrame:
    cur = conn.execute(sql, params)
    return pl.DataFrame(cur.fetchall(), schema=[d.name for d in cur.description], orient="row", infer_schema_length=None)


def load_universe(conn) -> pl.DataFrame:
    df = _q(conn, """
        SELECT r.gsis_id, p.display_name AS name, r.position AS pos, r.season, r.week, r.team,
               g.game_id, g.kickoff_utc,
               CASE WHEN g.home_team = r.team THEN g.away_team ELSE g.home_team END AS opp,
               (g.home_team = r.team) AS is_home,
               coalesce(pg.targets, 0) AS targets,
               coalesce(pg.receptions, 0) AS receptions,
               coalesce(pg.receiving_yards, 0)::float AS rec_yds,
               (res.game_id IS NOT NULL) AS played
        FROM roster_status r
        JOIN games g ON g.season = r.season AND g.week = r.week
                    AND r.team IN (g.home_team, g.away_team) AND g.game_type = 'REG'
        LEFT JOIN players p ON p.gsis_id = r.gsis_id
        LEFT JOIN player_game pg ON pg.gsis_id = r.gsis_id AND pg.game_id = g.game_id
        LEFT JOIN game_results res ON res.game_id = g.game_id
        WHERE r.status = 'ACT' AND r.position IN ('WR', 'TE', 'RB')""")
    return df.sort("gsis_id", "kickoff_utc")


def load_team_games(conn) -> pl.DataFrame:
    """Team-level targets per game plus that game's closing lines (team perspective)."""
    return _q(conn, """
        WITH t AS (
            SELECT game_id, team, sum(coalesce(targets, 0)) AS team_targets
            FROM player_game GROUP BY game_id, team)
        SELECT g.game_id, g.season, g.week, g.kickoff_utc, side.team, side.opp,
               coalesce(t.team_targets, 0) AS team_targets,
               (res.game_id IS NOT NULL) AS played,
               CASE WHEN side.team = g.home_team THEN l.spread_line ELSE -l.spread_line END::float AS team_spread,
               l.total_line::float AS total_line
        FROM games g
        CROSS JOIN LATERAL (VALUES (g.home_team, g.away_team), (g.away_team, g.home_team)) AS side(team, opp)
        LEFT JOIN t ON t.game_id = g.game_id AND t.team = side.team
        LEFT JOIN game_results res ON res.game_id = g.game_id
        LEFT JOIN game_lines l ON l.game_id = g.game_id AND l.book = 'nflverse_close'
        WHERE g.game_type = 'REG'
        ORDER BY side.team, g.kickoff_utc""")


def load_catch_yards(conn) -> pl.DataFrame:
    """Every completed pass to a WR/TE/RB: yards gained, season, receiver position."""
    return _q(conn, """
        SELECT pl.season, p.position AS pos, pl.yards_gained::float AS yds
        FROM plays pl JOIN players p ON p.gsis_id = pl.receiver_player_id
        WHERE pl.complete_pass AND p.position IN ('WR', 'TE', 'RB')
          AND pl.yards_gained IS NOT NULL""")


def ewm_prior_sums(keys: np.ndarray, values: np.ndarray, decay: float) -> np.ndarray:
    """For rows sorted by (key, time): exponentially weighted sum of PREVIOUS rows'
    values within the same key. The row itself is never included, so a feature
    built from it cannot see its own game.

    S_t = x_{t-1} + decay * S_{t-1}, reset to 0 at each new key.
    """
    out = np.zeros(len(values), dtype=float)
    s = 0.0
    prev_val = 0.0
    for i in range(len(values)):
        if i == 0 or keys[i] != keys[i - 1]:
            s = 0.0
        else:
            s = prev_val + decay * s
        out[i] = s
        prev_val = values[i]
    return out


def decay_for(half_life: float) -> float:
    return 0.5 ** (1.0 / half_life)
