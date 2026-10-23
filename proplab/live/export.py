"""Collect what the Week overview, Player profiles and Model health pages need.

    python -m proplab.live.export     # -> phase5/dashboard_live.json
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from proplab import db

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase5" / "dashboard_live.json"
LOG_GAMES = 8


def projected_ids(conn) -> tuple[set[str], dict]:
    """gsis_ids of everyone projected this week; older exports without ids are matched by name within the team."""
    players = json.loads((ROOT / "phase1" / "projections.json").read_text())["players"]
    for m in json.loads((ROOT / "phase3" / "projections.json").read_text())["markets"].values():
        players += m["players"]
    ids = {p["gsis_id"] for p in players if p.get("gsis_id")}
    missing = {(p["player"], p["team"]) for p in players if not p.get("gsis_id")}
    by_name = {}
    if missing:
        rows = conn.execute("""SELECT DISTINCT ON (p.display_name, pg.team) p.display_name, pg.team, p.gsis_id
                               FROM player_game pg JOIN players p USING (gsis_id) WHERE pg.season >= %s - 1
                               ORDER BY p.display_name, pg.team, pg.season DESC, pg.week DESC""", (2026,)).fetchall()
        look = {(n, t): g for n, t, g in rows}
        by_name = {f"{n}|{t}": look[(n, t)] for n, t in missing if (n, t) in look}
        ids |= set(by_name.values())
    return ids, by_name


def market_lines(season: int, week: int) -> dict:
    """Latest sportsbook consensus per (player, market) for this week, if any were collected (Phase 9C)."""
    from proplab.p9.lines import OUT as LINES, latest_for_week
    book = latest_for_week(season, week)
    meta = json.loads(LINES.read_text()).get("last_run") if LINES.exists() else None
    return {"last_run": meta, "lines": {f"{g}|{m}": {k: v[k] for k in ("line", "p_over_market", "books", "snapshot_utc")}
                                        for (g, m), v in book.items()}}


def main(season: int = 2026) -> dict:
    conn = db.connect()
    now = dt.datetime.now(dt.timezone.utc)
    (week,) = conn.execute("""SELECT min(week) FROM games g WHERE season = %s AND game_type = 'REG'
                              AND NOT EXISTS (SELECT 1 FROM game_results r WHERE r.game_id = g.game_id)""", (season,)).fetchone()

    # --- week overview: games with the current consensus lines (nflverse schedule, fetched now)
    games = [dict(zip(("game_id", "kickoff", "home", "away", "roof", "stadium"), r)) for r in conn.execute("""
        SELECT game_id, kickoff_utc, home_team, away_team, roof, stadium FROM games
        WHERE season = %s AND week = %s AND game_type = 'REG' ORDER BY kickoff_utc, game_id""", (season, week))]
    try:
        sched = nfl.load_schedules([season]).filter(pl.col("week") == week)
        lines = {r["game_id"]: r for r in sched.select("game_id", "spread_line", "total_line").iter_rows(named=True)}
    except Exception:          # offline: show the games without lines
        lines = {}
    for g in games:
        g["kickoff"] = g["kickoff"].astimezone(dt.timezone.utc).isoformat()
        g["spread_home"] = lines.get(g["game_id"], {}).get("spread_line")
        g["total"] = lines.get(g["game_id"], {}).get("total_line")

    # --- player profiles: last 8 games for everyone with a projection this week
    idset, name_ids = projected_ids(conn)
    ids = sorted(idset)
    logs: dict = {}
    for r in conn.execute("""
        SELECT gsis_id, season, week, team, opponent, targets, receptions, receiving_yards::float, carries,
               rushing_yards::float, attempts, completions, passing_yards::float, passing_tds, passing_interceptions,
               coalesce(rushing_tds, 0) + coalesce(receiving_tds, 0), fg_made, pat_made,
               3 * coalesce(fg_made, 0) + coalesce(pat_made, 0)
        FROM (SELECT pg.*, row_number() OVER (PARTITION BY gsis_id ORDER BY season DESC, week DESC) AS rn
              FROM player_game pg WHERE gsis_id = ANY(%s) AND known_at <= %s) x
        WHERE rn <= %s ORDER BY gsis_id, season, week""", (ids, now, LOG_GAMES)):
        logs.setdefault(r[0], []).append(list(r[1:]))
    log_cols = ["season", "week", "team", "opp", "tgt", "rec", "rec_yds", "car", "rush_yds", "att", "cmp", "pass_yds",
                "pass_td", "int", "td", "fg", "pat", "k_pts"]

    # --- model health: latest ingest, its checks, data freshness, live record
    run = conn.execute("SELECT run_id, started_at, finished_at, row_counts FROM ingest_runs WHERE finished_at IS NOT NULL "
                       "ORDER BY run_id DESC LIMIT 1").fetchone()
    checks = [{"name": n, "passed": p, "detail": d} for n, p, d in conn.execute(
        "SELECT check_name, passed, detail FROM data_quality_runs WHERE run_id = %s ORDER BY check_name", (run[0],))] if run else []
    fresh = conn.execute("""
        SELECT max(g.season * 100 + g.week) FILTER (WHERE r.game_id IS NOT NULL),
               (SELECT max(known_at) FROM player_game WHERE known_at <= %s),
               (SELECT max(known_at) FROM roster_status WHERE known_at <= %s),
               (SELECT max(known_at) FROM injury_reports WHERE known_at <= %s)
        FROM games g LEFT JOIN game_results r USING (game_id) WHERE g.game_type = 'REG'""", (now, now, now)).fetchone()
    live_path = ROOT / "phase5" / "live_record.json"
    # final injury designations for this week, once published (Out / Doubtful / Questionable)
    inj = {g: st for g, st in conn.execute("""
        SELECT gsis_id, report_status FROM injury_reports
        WHERE season = %s AND week = %s AND game_type = 'REG' AND report_status IS NOT NULL AND known_at <= %s""",
        (season, week, now)).fetchall()}
    from proplab.live.leaders import build as build_leaders
    payload = {
        "leaders": build_leaders(conn, season, now),
        "injuries": inj,
        "market_lines": market_lines(season, week),
        "generated_utc": now.isoformat(timespec="minutes"), "season": season, "week": week, "games": games,
        "log_cols": log_cols, "logs": logs, "name_ids": name_ids,
        "health": {
            "ingest": None if not run else {"run_id": run[0], "started": run[1].astimezone(dt.timezone.utc).isoformat(timespec="minutes"),
                                           "finished": run[2].astimezone(dt.timezone.utc).isoformat(timespec="minutes"), "rows": run[3]},
            "checks": checks,
            "freshness": {"last_final_week": None if fresh[0] is None else f"{fresh[0] // 100} week {fresh[0] % 100}",
                          "player_stats_known_at": fresh[1] and fresh[1].astimezone(dt.timezone.utc).isoformat(timespec="minutes"),
                          "rosters_known_at": fresh[2] and fresh[2].astimezone(dt.timezone.utc).isoformat(timespec="minutes"),
                          "injuries_known_at": fresh[3] and fresh[3].astimezone(dt.timezone.utc).isoformat(timespec="minutes")},
            "ledger": [json.loads(p.read_text()) for p in sorted((ROOT / "ledger").glob("*_w*/manifest.json"))],
            "live": json.loads(live_path.read_text()) if live_path.exists() else None,
        },
    }
    for m in payload["health"]["ledger"]:
        m.pop("models", None)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"week {week}: {len(games)} games, {len(logs)} player logs, {len(checks)} checks")
    return payload


if __name__ == "__main__":
    main()
