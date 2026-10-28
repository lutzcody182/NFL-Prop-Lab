"""Phase 9C: snapshot live player prop lines from The Odds API and turn them into consensus lines.

    python -m proplab.p9.lines            # in the weekly run (needs ODDS_API_KEY); skips quietly without a key

Budget-safe (the same key also serves another dashboard):
  - at most MAX_PER_RUN credits per run, and never below RESERVE remaining credits on the account
  - one snapshot per game per 24 hours, only for games kicking off within the next 4 days
Raw responses are NOT stored in the repository. Only our derived consensus (modal line across books and the
median no-vig over chance) is appended to phase9/market_lines.json.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path

from proplab import db, names
from proplab.odds.oddsmath import novig_multiplicative
from proplab.p12.books import allowed as licensed_books

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase9" / "market_lines.json"
sys.path.insert(0, str(ROOT / "tools"))
from odds_backfill import TEAM_NAMES, Client  # noqa: E402

BASE_URL = os.environ.get("ODDS_API_BASE", "https://api.the-odds-api.com")
SPORT = "americanfootball_nfl"
MAX_PER_RUN = int(os.environ.get("ODDS_MAX_PER_RUN", "400"))
RESERVE = int(os.environ.get("ODDS_RESERVE", "2000"))
WINDOW = dt.timedelta(days=4)
DEDUPE = dt.timedelta(hours=24)
# Odds API market -> our market(s), chosen by the player's position
API_MARKETS = {
    "player_pass_yds": {"QB": "qb_pass_yds"}, "player_pass_tds": {"QB": "qb_pass_tds"},
    "player_pass_attempts": {"QB": "qb_pass_att"}, "player_pass_completions": {"QB": "qb_completions"},
    "player_rush_yds": {"QB": "qb_rush_yds", "RB": "rb_rush_yds"}, "player_rush_attempts": {"RB": "rb_rush_att"},
    "player_receptions": {"WR": "wrte_rec", "TE": "wrte_rec", "RB": "rb_rec"},
    "player_reception_yds": {"WR": "wrte_rec_yds", "TE": "wrte_rec_yds", "RB": "rb_rec_yds"},
}


def candidates(conn, games: dict) -> dict:
    """game_id -> {normalized name -> set(gsis_id)}: players on either team's latest posted roster (this week or earlier)."""
    out: dict = {}
    for gid, g in games.items():
        rows = conn.execute("""
            SELECT m.source_id, m.gsis_id FROM roster_status r
            JOIN player_id_map m ON m.gsis_id = r.gsis_id AND m.source = 'name'
            WHERE r.season = %s AND r.team = ANY(%s) AND r.status = 'ACT'
              AND r.week = (SELECT max(week) FROM roster_status x WHERE x.season = %s AND x.team = r.team AND x.week <= %s)""",
            (g["season"], [g["home"], g["away"]], g["season"], g["week"])).fetchall()
        d: dict = {}
        for nm, gs in rows:
            d.setdefault(nm, set()).add(gs)
        out[gid] = d
    return out


def parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def consensus(event_odds: dict) -> list[dict]:
    """(api market, player, line) -> modal line across books and median no-vig over chance at that line."""
    by: dict = {}
    for bk in event_odds.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] not in API_MARKETS:
                continue
            pairs: dict = {}
            for o in mk.get("outcomes", []):
                side = (o.get("name") or "").lower()
                if side in ("over", "under") and o.get("point") is not None:
                    pairs.setdefault((o.get("description", ""), float(o["point"])), {})[side] = float(o["price"])
            for (player, line), pr in pairs.items():
                if "over" in pr and "under" in pr and pr["over"] > 1 and pr["under"] > 1:
                    by.setdefault((mk["key"], player), []).append((bk["key"], line, novig_multiplicative(pr["over"], pr["under"]),
                                                                    pr["over"], pr["under"]))
    out = []
    for (mk, player), rows in by.items():
        lines = [r[1] for r in rows]
        modal = max(set(lines), key=lambda L: (lines.count(L), -abs(L - statistics.median(lines))))
        ps = [r[2] for r in rows if r[1] == modal]
        row = {"api_market": mk, "player_name": player, "line": modal, "p_over_market": statistics.median(ps),
               "books": len(ps), "books_any_line": len(rows)}
        # Phase 12: best price on each side at the consensus line among state-licensed US books only
        ok = licensed_books()
        lic = [r for r in rows if r[1] == modal and r[0] in ok]
        if lic:
            bo, bu = max(lic, key=lambda r: r[3]), max(lic, key=lambda r: r[4])
            row["licensed"] = {"books": sorted(r[0] for r in lic), "best_over": [bo[3], bo[0]], "best_under": [bu[4], bu[0]]}
        out.append(row)
    return out


def match_game(events_game: dict, games: dict) -> str | None:
    h, a = TEAM_NAMES.get(events_game["home_team"]), TEAM_NAMES.get(events_game["away_team"])
    k = parse_ts(events_game["commence_time"])
    for gid, g in games.items():
        if g["home"] == h and g["away"] == a and abs((g["kickoff"] - k).total_seconds()) < 12 * 3600:
            return gid
    return None


def main(now: dt.datetime | None = None, base_url: str = BASE_URL) -> dict:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY not set: skipping live lines.")
        return {"skipped": True}
    now = now or dt.datetime.now(dt.timezone.utc)
    conn = db.connect()
    games = {r[0]: {"home": r[1], "away": r[2], "kickoff": r[3], "season": r[4], "week": r[5]} for r in conn.execute(
        "SELECT game_id, home_team, away_team, kickoff_utc, season, week FROM games WHERE game_type = 'REG' AND kickoff_utc BETWEEN %s AND %s",
        (now, now + WINDOW)).fetchall()}
    cand = candidates(conn, games)
    pos = dict(conn.execute("SELECT gsis_id, position FROM players").fetchall())
    store = json.loads(OUT.read_text()) if OUT.exists() else {"snapshots": []}
    last = {}
    for s in store["snapshots"]:
        last[s["game_id"]] = max(last.get(s["game_id"], s["snapshot_utc"]), s["snapshot_utc"])
    client = Client(key, base_url, MAX_PER_RUN)
    events, _ = client.get(f"/v4/sports/{SPORT}/events", {})          # free: no credits
    cost = len(API_MARKETS)                                            # 1 credit per market per region per event
    taken = 0
    for ev in sorted(events, key=lambda e: e["commence_time"]):
        gid = match_game(ev, games)
        if gid is None or now >= games[gid]["kickoff"]:
            continue
        if gid in last and now - parse_ts(last[gid]) < DEDUPE:
            continue
        if not client.can_afford(cost) or (client.remaining is not None and client.remaining - cost < RESERVE):
            print(f"Budget reached (spent {client.spent}, remaining {client.remaining}); stopping.")
            break
        body, hdr = client.get(f"/v4/sports/{SPORT}/events/{ev['id']}/odds",
                               {"regions": "us", "markets": ",".join(API_MARKETS), "oddsFormat": "decimal"})
        if isinstance(body, dict) and body.get("status") == 422:
            continue
        rows = []
        for c in consensus(body):
            nm = names.normalize(c["player_name"])
            ids = cand.get(gid, {}).get(nm, set())
            gsis = next(iter(ids)) if len(ids) == 1 else None
            ours = API_MARKETS[c["api_market"]].get(pos.get(gsis, ""), None) if gsis else None
            rows.append({**c, "gsis_id": gsis, "market": ours})
        store["snapshots"].append({"game_id": gid, "season": games[gid]["season"], "week": games[gid]["week"],
                                   "snapshot_utc": now.isoformat(timespec="seconds"), "lines": rows})
        taken += 1
    store["last_run"] = {"utc": now.isoformat(timespec="seconds"), "games": taken, "credits": client.spent,
                         "remaining": client.remaining}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(store, separators=(",", ":")))
    print(f"live lines: {taken} games, {client.spent} credits, remaining {client.remaining}")
    return store


def latest_for_week(season: int, week: int) -> dict:
    """(gsis_id, market) -> latest consensus line for that week (for the dashboard and scoring)."""
    if not OUT.exists():
        return {}
    out = {}
    for s in sorted(json.loads(OUT.read_text())["snapshots"], key=lambda s: s["snapshot_utc"]):
        if s["season"] == season and s["week"] == week:
            for r in s["lines"]:
                if r["gsis_id"] and r["market"]:
                    out[(r["gsis_id"], r["market"])] = {**r, "game_id": s["game_id"], "snapshot_utc": s["snapshot_utc"]}
    return out


if __name__ == "__main__":
    main()
