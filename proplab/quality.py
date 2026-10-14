"""Data quality checks run after every ingest. Results go to data_quality_runs.

A failed check makes the ingest exit non-zero, so a scheduled run that loads
bad data shows up as a failed job instead of silently feeding the models.
"""
from __future__ import annotations

import psycopg

# table -> ordering used for the reproducibility hash
TABLE_KEYS = {
    "players": "gsis_id",
    "player_id_map": "source, source_id, gsis_id",
    "games": "game_id",
    "game_results": "game_id",
    "game_lines": "game_id, book, snapshot_ts",
    "plays": "game_id, play_id",
    "player_game": "gsis_id, game_id",
    "snaps": "pfr_player_id, game_id",
    "roster_status": "gsis_id, season, week, team",
    "injury_reports": "gsis_id, season, week, game_type, team",
}

SKILL = ("QB", "RB", "WR", "TE", "FB")

# Real-world cancellations, so game-count checks stay exact.
CANCELLED_REG_GAMES = {
    2022: 1,  # Week 17 BUF @ CIN, cancelled after Damar Hamlin's cardiac arrest
}


def _one(conn: psycopg.Connection, sql: str, params: tuple = ()):
    return conn.execute(sql, params).fetchone()


def run_checks(conn: psycopg.Connection, run_id: int, seasons: list[int], report: dict) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # 1. Regular-season game counts: 256 per season before 2021, 272 from 2021,
    #    minus documented cancellations.
    rows = conn.execute(
        "SELECT season, count(*) FROM games WHERE game_type = 'REG' AND season = ANY(%s) GROUP BY season ORDER BY season",
        (seasons,),
    ).fetchall()
    bad = [(s, n) for s, n in rows
           if n != (256 if s < 2021 else 272) - CANCELLED_REG_GAMES.get(s, 0)]
    add("regular_season_game_counts", not bad and len(rows) == len(seasons),
        f"{len(rows)} seasons checked; mismatches: {bad or 'none'}")

    # 2. Duplicate keys in the source (dropped before load). Allow < 0.1%.
    dup_detail, dup_ok = [], True
    for name, r in report.items():
        raw, dups = r.get("raw_rows", 0), r.get("duplicates_dropped", 0)
        if raw and dups / raw >= 0.001:
            dup_ok = False
        if dups:
            dup_detail.append(f"{name}={dups}")
    add("source_duplicate_keys", dup_ok, ", ".join(dup_detail) or "none")

    # 3. Every completed game has a result and a plausible number of plays.
    n_missing = _one(conn, """
        SELECT count(*) FROM games g
        WHERE g.season = ANY(%s) AND EXISTS (SELECT 1 FROM plays p WHERE p.game_id = g.game_id)
          AND NOT EXISTS (SELECT 1 FROM game_results r WHERE r.game_id = g.game_id)""", (seasons,))[0]
    thin = _one(conn, """
        SELECT count(*) FROM (SELECT p.game_id, count(*) n FROM plays p JOIN game_results r USING (game_id)
                              WHERE p.season = ANY(%s) GROUP BY p.game_id) t WHERE n < 100""", (seasons,))[0]
    add("completed_games_have_results_and_plays", n_missing == 0 and thin == 0,
        f"games with plays but no result: {n_missing}; completed games with <100 plays: {thin}")

    # 4. EPA present on real offensive plays.
    tot, nulls = _one(conn, """
        SELECT count(*), count(*) FILTER (WHERE epa IS NULL) FROM plays
        WHERE season = ANY(%s) AND play_type IN ('pass','run')""", (seasons,))
    add("epa_null_rate_offense", tot > 0 and nulls / tot < 0.005, f"{nulls:,} of {tot:,} pass/run plays lack EPA")

    # 5. Box score consistency: team passing yards (QBs) vs team receiving yards.
    n, ok = _one(conn, """
        SELECT count(*), count(*) FILTER (WHERE abs(pass_yds - rec_yds) <= 1) FROM (
            SELECT game_id, team, sum(coalesce(passing_yards,0)) pass_yds, sum(coalesce(receiving_yards,0)) rec_yds
            FROM player_game WHERE season = ANY(%s) GROUP BY game_id, team) t""", (seasons,))
    add("team_pass_vs_receiving_yards", n > 0 and ok / n >= 0.98,
        f"{ok:,} of {n:,} team-games within 1 yard ({ok / max(n, 1):.1%})")

    # 6. Box score vs play-by-play: completions per team-game.
    n, ok = _one(conn, """
        SELECT count(*), count(*) FILTER (WHERE box = pbp) FROM (
            SELECT b.game_id, b.team, b.box, coalesce(p.pbp, 0) pbp FROM
              (SELECT game_id, team, sum(coalesce(completions,0)) box FROM player_game
               WHERE season = ANY(%s) GROUP BY game_id, team) b
            LEFT JOIN
              (SELECT game_id, posteam team, count(*) pbp FROM plays
               WHERE season = ANY(%s) AND complete_pass GROUP BY game_id, posteam) p
            USING (game_id, team)) t""", (seasons, seasons))
    add("completions_box_vs_pbp", n > 0 and ok / n >= 0.97,
        f"{ok:,} of {n:,} team-games match exactly ({ok / max(n, 1):.1%})")

    # 7. Point-in-time stamps: outcomes after kickoff, pre-game info before it.
    late = _one(conn, """
        SELECT
          (SELECT count(*) FROM player_game pg JOIN games g USING (game_id) WHERE pg.known_at <= g.kickoff_utc),
          (SELECT count(*) FROM plays p JOIN games g USING (game_id) WHERE p.known_at <= g.kickoff_utc),
          (SELECT count(*) FROM game_results r JOIN games g USING (game_id) WHERE r.known_at <= g.kickoff_utc)""")
    early = _one(conn, """
        SELECT
          (SELECT count(*) FROM roster_status r JOIN games g
             ON g.season = r.season AND g.week = r.week AND r.team IN (g.home_team, g.away_team)
           WHERE r.known_at >= g.kickoff_utc),
          (SELECT count(*) FROM injury_reports i JOIN games g
             ON g.season = i.season AND g.week = i.week AND i.team IN (g.home_team, g.away_team)
           WHERE i.known_at >= g.kickoff_utc)""")
    add("known_at_ordering", sum(late) == 0 and sum(early) == 0,
        f"outcome rows stamped at/before kickoff: {sum(late)}; pre-game rows stamped at/after kickoff: {sum(early)}")

    # 8. Identity resolution for skill players.
    tot, unmatched = _one(conn, """
        SELECT count(*), count(*) FILTER (WHERE gsis_id IS NULL) FROM snaps s JOIN games g USING (game_id)
        WHERE g.season = ANY(%s) AND s.position = ANY(%s) AND s.offense_snaps > 0""", (seasons, list(SKILL)))
    add("snap_ids_resolved_skill_positions", tot > 0 and unmatched / tot < 0.02,
        f"{unmatched:,} of {tot:,} skill-player snap rows unmatched ({unmatched / max(tot, 1):.2%})")
    tot, unknown = _one(conn, """
        SELECT count(*), count(*) FILTER (WHERE p.gsis_id IS NULL) FROM player_game pg
        LEFT JOIN players p USING (gsis_id) WHERE pg.season = ANY(%s)""", (seasons,))
    add("player_game_ids_in_players", tot > 0 and unknown / tot < 0.005,
        f"{unknown:,} of {tot:,} player-game rows have no players record")

    # 9. Team codes in game-level tables must be one of that game's two teams
    #    (catches relocation codes like LV vs OAK not lining up with the schedule).
    mism = _one(conn, """
        SELECT
          (SELECT count(*) FROM player_game t JOIN games g USING (game_id)
            WHERE t.team NOT IN (g.home_team, g.away_team) OR t.opponent NOT IN (g.home_team, g.away_team)),
          (SELECT count(*) FROM plays t JOIN games g USING (game_id)
            WHERE (t.posteam IS NOT NULL AND t.posteam NOT IN (g.home_team, g.away_team))
               OR (t.defteam IS NOT NULL AND t.defteam NOT IN (g.home_team, g.away_team))),
          (SELECT count(*) FROM snaps t JOIN games g USING (game_id) WHERE t.team NOT IN (g.home_team, g.away_team))""")
    add("team_codes_match_schedule", sum(mism) == 0,
        f"mismatched rows: player_game {mism[0]}, plays {mism[1]}, snaps {mism[2]}")

    # 10. Every completed game has receiving targets recorded for both teams.
    (no_tgt,) = _one(conn, """
        SELECT count(*) FROM games g JOIN game_results r USING (game_id)
        CROSS JOIN LATERAL (VALUES (g.home_team), (g.away_team)) s(team)
        WHERE g.season = ANY(%s) AND NOT EXISTS (
            SELECT 1 FROM player_game pg WHERE pg.game_id = g.game_id AND pg.team = s.team AND pg.targets > 0)""",
        (seasons,))
    add("every_team_game_has_targets", no_tgt == 0, f"{no_tgt} completed team-games with no recorded targets")

    # 11. Pre-game rows that could not be placed on a kickoff (dropped).
    dropped = sum(r.get("no_week_kickoff", 0) for r in report.values())
    add("pregame_rows_without_kickoff", dropped == 0 or dropped < 50,
        f"{dropped} roster/injury rows had no matching week and were dropped")

    with conn.cursor() as cur:
        for c in checks:
            cur.execute(
                "INSERT INTO data_quality_runs (run_id, check_name, passed, detail) VALUES (%s, %s, %s, %s)",
                (run_id, c["name"], c["passed"], c["detail"]),
            )
    conn.commit()
    return checks
