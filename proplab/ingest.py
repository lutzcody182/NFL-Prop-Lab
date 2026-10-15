"""Phase 0 ingest: nflverse -> PostgreSQL with point-in-time known_at stamps.

Usage:
    python -m proplab.ingest                 # default seasons 2016..current
    python -m proplab.ingest --seasons 2024 2025

Every known_at is derived from the game's kickoff (never from the clock at load
time), so two runs over the same source data produce identical tables.

known_at rules (see README "Point-in-time assumptions"):
    plays, player_game, snaps : kickoff + 1 day   (nflverse nightly update)
    game_results              : kickoff + 4 hours
    roster_status             : kickoff - 90 minutes (game-day inactives)
    injury_reports            : kickoff - 40 hours  (final report, assumption A1)
    ftn_charting              : kickoff + 2 days  (FTN charting lag, assumption A6)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import nflreadpy as nfl
import polars as pl

from proplab import db, names, quality

DEFAULT_FIRST_SEASON = 2016  # Next Gen Stats era; enough history for priors
CURRENT_SEASON = 2026

PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "posteam", "defteam", "qtr", "down",
    "ydstogo", "yardline_100", "game_seconds_remaining", "play_type", "pass", "rush",
    "qb_dropback", "sack", "qb_hit", "qb_scramble", "complete_pass", "interception",
    "touchdown", "shotgun", "no_huddle", "air_yards", "yards_after_catch",
    "yards_gained", "epa", "success", "wp", "xpass", "pass_oe", "cpoe",
    "pass_length", "pass_location", "run_location", "run_gap", "passer_player_id",
    "receiver_player_id", "rusher_player_id", "td_player_id",
]
PBP_BOOL = ["pass", "rush", "qb_dropback", "sack", "qb_hit", "qb_scramble",
            "complete_pass", "interception", "touchdown", "shotgun", "no_huddle", "success"]
PBP_INT = ["play_id", "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining"]

STAT_COLUMNS = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "passing_air_yards", "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
    "receiving_yards_after_catch", "target_share", "air_yards_share",
]

ONE_DAY = dt.timedelta(days=1)
FTN_FIRST_SEASON = 2022
FTN_LAG = dt.timedelta(days=2)
FTN_BOOL = ["is_play_action", "is_screen_pass", "is_rpo", "is_motion", "is_no_huddle", "is_qb_out_of_pocket",
            "is_throw_away", "is_catchable_ball", "is_drop"]
FTN_INT = ["n_defense_box", "n_blitzers", "n_pass_rushers", "n_offense_backfield"]


# ---------- helpers ----------

def kickoff_expr() -> pl.Expr:
    """gameday + gametime are US Eastern local time; convert to UTC."""
    return (
        pl.concat_str([pl.col("gameday"), pl.col("gametime").fill_null("13:00")], separator=" ")
        .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M")
        .dt.replace_time_zone("America/New_York", ambiguous="earliest")
        .dt.convert_time_zone("UTC")
        .alias("kickoff_utc")
    )


def dedupe(df: pl.DataFrame, key: list[str], name: str, report: dict) -> pl.DataFrame:
    out = df.unique(subset=key, keep="last", maintain_order=True)
    report[name] = {"raw_rows": df.height, "duplicates_dropped": df.height - out.height}
    return out


# Franchise relocations / alternate codes. nflverse player stats and play-by-play
# use current codes (LV, LAC) for every season, while schedules, rosters and snap
# counts use the code of the time (OAK, SD). We rewrite every team column to the
# schedule's code for that game so all tables join.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}


def align_teams(df: pl.DataFrame, sched: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    canon = lambda e: e.replace(FRANCHISE)
    sides = pl.concat([
        sched.select("game_id", pl.col("home_team").alias("_code")),
        sched.select("game_id", pl.col("away_team").alias("_code")),
    ]).with_columns(canon(pl.col("_code")).alias("_canon"))
    for c in cols:
        df = (df.with_columns(canon(pl.col(c)).alias("_canon"))
                .join(sides, on=["game_id", "_canon"], how="left")
                .with_columns(pl.coalesce("_code", c).alias(c))
                .drop("_canon", "_code"))
    return df


def to_bool(cols: list[str]) -> list[pl.Expr]:
    return [pl.when(pl.col(c).is_null()).then(None).otherwise(pl.col(c) != 0).alias(c) for c in cols]


# ---------- loaders ----------

def load_games(seasons: list[int]) -> pl.DataFrame:
    sched = nfl.load_schedules(seasons).with_columns(kickoff_expr())
    return sched


def build_team_weeks(sched: pl.DataFrame) -> pl.DataFrame:
    """One row per team per season-week with that team's kickoff."""
    home = sched.select("season", "week", pl.col("home_team").alias("team"), "kickoff_utc")
    away = sched.select("season", "week", pl.col("away_team").alias("team"), "kickoff_utc")
    team_weeks = pl.concat([home, away]).group_by("season", "week", "team").agg(pl.col("kickoff_utc").min())
    # Bye weeks: fall back to the earliest kickoff of that season-week.
    week_first = sched.group_by("season", "week").agg(pl.col("kickoff_utc").min().alias("week_kickoff"))
    return team_weeks, week_first


def attach_kickoff(df: pl.DataFrame, team_weeks: pl.DataFrame, week_first: pl.DataFrame) -> pl.DataFrame:
    # Older nflverse files store season/week as floats; normalize before joining.
    df = df.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    team_weeks = team_weeks.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    week_first = week_first.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    return (
        df.join(team_weeks, on=["season", "week", "team"], how="left")
        .join(week_first, on=["season", "week"], how="left")
        .with_columns(pl.coalesce("kickoff_utc", "week_kickoff").alias("kickoff_utc"))
        .drop("week_kickoff")
    )


def run(seasons: list[int]) -> dict:
    conn = db.connect()
    db.apply_schema(conn)
    report: dict = {}

    with conn.cursor() as cur:
        cur.execute("INSERT INTO ingest_runs (seasons) VALUES (%s) RETURNING run_id", (seasons,))
        run_id = cur.fetchone()[0]
    conn.commit()
    season_list = list(seasons)

    # --- players + ID map (full reload; players file is cumulative) ---
    players = nfl.load_players().filter(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != ""))
    players = dedupe(players, ["gsis_id"], "players", report)
    players_tbl = players.select(
        "gsis_id", "display_name", "position", "position_group",
        pl.col("birth_date").str.to_date(strict=False),
        pl.col("height").cast(pl.Int32, strict=False).alias("height_in"),
        pl.col("weight").cast(pl.Int32, strict=False).alias("weight_lb"),
        pl.col("rookie_season").cast(pl.Int32, strict=False),
        pl.col("draft_year").cast(pl.Int32, strict=False),
        pl.col("draft_round").cast(pl.Int32, strict=False),
        pl.col("draft_pick").cast(pl.Int32, strict=False),
    )
    id_map = pl.concat([
        players.select("gsis_id", pl.lit(src).alias("source"), pl.col(col).cast(pl.Utf8).alias("source_id"))
        for src, col in [("pfr", "pfr_id"), ("espn", "espn_id"), ("pff", "pff_id")]
    ]).filter(pl.col("source_id").is_not_null() & (pl.col("source_id") != ""))
    # Normalized name variants (source='name'), used to match sportsbook player names.
    name_rows = [
        (r["gsis_id"], "name", v)
        for r in players.select("gsis_id", "display_name", "first_name", "last_name",
                                "football_name", "common_first_name").iter_rows(named=True)
        for v in names.variants(r["display_name"], r["first_name"], r["last_name"],
                                r["football_name"], r["common_first_name"])
    ]
    id_map = pl.concat([id_map, pl.DataFrame(name_rows, schema=["gsis_id", "source", "source_id"], orient="row")])
    id_map = dedupe(id_map, ["source", "source_id", "gsis_id"], "player_id_map", report)

    with conn.transaction():
        db.replace_rows(conn, "player_id_map", id_map.head(0))  # clear child before parent
        db.replace_rows(conn, "players", players_tbl)
        db.replace_rows(conn, "player_id_map", id_map)
    known_players = set(players_tbl["gsis_id"].to_list())

    # --- games, results, closing lines ---
    sched = load_games(season_list)
    sched = dedupe(sched, ["game_id"], "games", report)
    games = sched.select(
        "game_id", "season", "game_type", "week", "kickoff_utc", "home_team", "away_team",
        "home_rest", "away_rest", pl.col("div_game").cast(pl.Boolean),
        "roof", "surface", "stadium_id", "stadium", "location",
    )
    played = sched.filter(pl.col("home_score").is_not_null())
    results = played.select(
        "game_id", "home_score", "away_score", pl.col("overtime").cast(pl.Boolean),
        pl.col("temp").cast(pl.Int32, strict=False).alias("temp_f"),
        pl.col("wind").cast(pl.Int32, strict=False).alias("wind_mph"),
        (pl.col("kickoff_utc") + dt.timedelta(hours=4)).alias("known_at"),
    )
    # Schedule lines are only treated as closing lines once the game is played;
    # live/opening lines arrive from the odds provider in Phase 2.
    lines = played.filter(pl.col("spread_line").is_not_null()).select(
        "game_id", pl.lit("nflverse_close").alias("book"),
        pl.col("kickoff_utc").alias("snapshot_ts"), pl.lit(True).alias("is_closing"),
        "spread_line", "total_line",
        pl.col("home_moneyline").cast(pl.Int32, strict=False),
        pl.col("away_moneyline").cast(pl.Int32, strict=False),
    )
    team_weeks, week_first = build_team_weeks(sched)
    kick = sched.select("game_id", "kickoff_utc")
    game_ids = set(games["game_id"].to_list())

    season_filter = "game_id IN (SELECT game_id FROM games WHERE season = ANY(%s))"
    with conn.transaction():
        for child in ["game_lines", "game_results", "snaps", "player_game", "plays"]:
            conn.execute(f"DELETE FROM {child} WHERE {season_filter}", (season_list,))
        db.replace_rows(conn, "games", games, "season = ANY(%s)", (season_list,))
        db.replace_rows(conn, "game_results", results, season_filter, (season_list,))
        db.replace_rows(conn, "game_lines", lines, season_filter, (season_list,))

    # --- weekly player stats (prop outcomes) ---
    ps = nfl.load_player_stats(season_list, summary_level="week")
    ps = ps.filter(pl.col("player_id").is_not_null() & pl.col("game_id").is_in(list(game_ids)))
    ps = ps.rename({"player_id": "gsis_id", "opponent_team": "opponent"})
    ps = align_teams(ps, sched, ["team", "opponent"])
    ps = dedupe(ps, ["gsis_id", "game_id"], "player_game", report)
    player_game = ps.join(kick, on="game_id").select(
        "gsis_id", "game_id", "season", "week", "team", "opponent", "position",
        *[pl.col(c) for c in STAT_COLUMNS],
        (pl.col("kickoff_utc") + ONE_DAY).alias("known_at"),
    ).with_columns([pl.col(c).cast(pl.Int32, strict=False) for c in
                    ["completions", "attempts", "passing_tds", "passing_interceptions", "sacks_suffered",
                     "carries", "rushing_tds", "targets", "receptions", "receiving_tds"]])
    report["player_game"]["unknown_players"] = int((~player_game["gsis_id"].is_in(list(known_players))).sum())
    with conn.transaction():
        db.replace_rows(conn, "player_game", player_game, "season = ANY(%s)", (season_list,))

    # --- play-by-play, one season at a time to bound memory ---
    report["plays"] = {"raw_rows": 0, "duplicates_dropped": 0}
    for season in season_list:
        pbp = nfl.load_pbp([season])
        cols = [c for c in PBP_COLUMNS if c in pbp.columns]
        pbp = pbp.select(cols).filter(pl.col("play_id").is_not_null() & pl.col("game_id").is_in(list(game_ids)))
        pbp = align_teams(pbp, sched, ["posteam", "defteam"])
        pbp = pbp.with_columns(to_bool([c for c in PBP_BOOL if c in cols]))
        pbp = pbp.with_columns([pl.col(c).cast(pl.Int32, strict=False) for c in PBP_INT if c in cols])
        part: dict = {}
        pbp = dedupe(pbp, ["game_id", "play_id"], "p", part)
        report["plays"]["raw_rows"] += part["p"]["raw_rows"]
        report["plays"]["duplicates_dropped"] += part["p"]["duplicates_dropped"]
        pbp = pbp.join(kick, on="game_id").with_columns((pl.col("kickoff_utc") + ONE_DAY).alias("known_at")).drop("kickoff_utc")
        with conn.transaction():
            db.replace_rows(conn, "plays", pbp, "season = %s", (season,))
        print(f"  plays {season}: {pbp.height:,}", flush=True)

    # --- FTN charting (blitzers, box count, play action...), 2022 onward ---
    ftn_seasons = [s for s in season_list if s >= FTN_FIRST_SEASON]
    report["ftn_charting"] = {"raw_rows": 0, "duplicates_dropped": 0, "unmatched_plays": 0}
    if ftn_seasons:
        ftn = nfl.load_ftn_charting(ftn_seasons).rename({"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        ftn = ftn.with_columns(pl.col("play_id").cast(pl.Int32, strict=False)) \
                 .filter(pl.col("play_id").is_not_null() & pl.col("game_id").is_in(list(game_ids)))
        ftn = dedupe(ftn, ["game_id", "play_id"], "ftn_charting", report)
        ftn = ftn.select("game_id", "play_id", pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
                         *[pl.col(c).cast(pl.Int32, strict=False) for c in FTN_INT], "qb_location",
                         *[pl.col(c).cast(pl.Boolean, strict=False) for c in FTN_BOOL])
        ftn = ftn.join(kick, on="game_id").with_columns((pl.col("kickoff_utc") + FTN_LAG).alias("known_at")).drop("kickoff_utc")
        with conn.transaction():
            db.replace_rows(conn, "ftn_charting", ftn, "season = ANY(%s)", (ftn_seasons,))
        (n_un,) = conn.execute("""SELECT count(*) FROM ftn_charting f WHERE f.season = ANY(%s)
                                  AND NOT EXISTS (SELECT 1 FROM plays p WHERE p.game_id = f.game_id AND p.play_id = f.play_id)""",
                               (ftn_seasons,)).fetchone()
        report["ftn_charting"]["unmatched_plays"] = int(n_un)
        print(f"  ftn_charting: {ftn.height:,}", flush=True)

    # --- snap counts (PFR ids -> gsis through the ID map) ---
    pfr_map = id_map.filter(pl.col("source") == "pfr").select(
        pl.col("source_id").alias("pfr_player_id"), "gsis_id").unique(subset=["pfr_player_id"], keep="first")
    snaps_raw = nfl.load_snap_counts([s for s in season_list if s >= 2012])
    snaps_raw = snaps_raw.filter(pl.col("game_id").is_in(list(game_ids)) & pl.col("pfr_player_id").is_not_null())
    snaps_raw = dedupe(snaps_raw, ["pfr_player_id", "game_id"], "snaps", report)
    snaps_raw = align_teams(snaps_raw, sched, ["team"])
    snaps = snaps_raw.join(pfr_map, on="pfr_player_id", how="left").join(kick, on="game_id").select(
        "pfr_player_id", "gsis_id", "game_id", "team", "position",
        *[pl.col(c).cast(pl.Int32, strict=False) for c in ["offense_snaps", "defense_snaps", "st_snaps"]],
        "offense_pct", "defense_pct", "st_pct",
        (pl.col("kickoff_utc") + ONE_DAY).alias("known_at"),
    )
    report["snaps"]["unmatched_pfr_ids"] = int(snaps["gsis_id"].is_null().sum())
    with conn.transaction():
        db.replace_rows(conn, "snaps", snaps, season_filter, (season_list,))

    # --- weekly roster status ---
    rw = nfl.load_rosters_weekly(season_list).filter(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != ""))
    rw = dedupe(rw, ["gsis_id", "season", "week", "team"], "roster_status", report)
    rw = attach_kickoff(rw, team_weeks, week_first)
    report["roster_status"]["no_week_kickoff"] = int(rw["kickoff_utc"].is_null().sum())
    roster = rw.filter(pl.col("kickoff_utc").is_not_null()).select(
        "gsis_id", "season", "week", "team", "position", "depth_chart_position", "status",
        pl.col("status_description_abbr").alias("status_abbr"),
        (pl.col("kickoff_utc") - dt.timedelta(minutes=90)).alias("known_at"),
    )
    with conn.transaction():
        db.replace_rows(conn, "roster_status", roster, "season = ANY(%s)", (season_list,))

    # --- injury reports ---
    inj = nfl.load_injuries([s for s in season_list if s >= 2009])
    inj = inj.filter(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != ""))
    inj = dedupe(inj, ["gsis_id", "season", "week", "game_type", "team"], "injury_reports", report)
    inj = attach_kickoff(inj, team_weeks, week_first)
    report["injury_reports"]["no_week_kickoff"] = int(inj["kickoff_utc"].is_null().sum())
    injuries = inj.filter(pl.col("kickoff_utc").is_not_null()).select(
        "gsis_id", "season", "week", "game_type", "team", "position",
        "report_primary_injury", "report_status", "practice_status",
        (pl.col("kickoff_utc") - dt.timedelta(hours=40)).alias("known_at"),
    )
    with conn.transaction():
        db.replace_rows(conn, "injury_reports", injuries, "season = ANY(%s)", (season_list,))

    # --- quality checks + reproducibility fingerprint ---
    results_qc = quality.run_checks(conn, run_id, season_list, report)
    hashes = {t: db.table_hash(conn, t, k) for t, k in quality.TABLE_KEYS.items()}
    counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in quality.TABLE_KEYS}
    conn.execute(
        "UPDATE ingest_runs SET finished_at = now(), table_hashes = %s, row_counts = %s WHERE run_id = %s",
        (json.dumps(hashes), json.dumps(counts), run_id),
    )
    conn.commit()
    conn.close()
    return {"run_id": run_id, "report": report, "checks": results_qc, "hashes": hashes, "counts": counts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=list(range(DEFAULT_FIRST_SEASON, CURRENT_SEASON + 1)))
    args = ap.parse_args(argv)
    out = run(args.seasons)
    failed = [c for c in out["checks"] if not c["passed"]]
    print(json.dumps({"run_id": out["run_id"], "counts": out["counts"]}, indent=2))
    for c in out["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
