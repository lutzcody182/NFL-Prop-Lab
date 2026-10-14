"""Load raw Odds API files (data/odds/raw) into odds_events, prop_lines and game_lines.

    python -m proplab.odds.load                 # default raw folder
    python -m proplab.odds.load --raw-dir PATH

Idempotent: rebuilds the odds tables from the raw files every time.
Bet-snapshot spreads/totals go to game_lines with book = '<bookmaker>' and
is_closing = false; they are the only game lines a bet-time model may use.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import polars as pl

from proplab import db, names

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "odds" / "raw"
sys.path.insert(0, str(ROOT / "tools"))
from odds_backfill import TEAM_NAMES  # noqa: E402

PROP_MARKETS = {"player_receptions", "player_reception_yds"}


def ts(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def roster_candidates(conn) -> dict:
    """(game_id) -> {normalized name -> set(gsis_id)} for players on either team that week."""
    rows = conn.execute("""
        SELECT g.game_id, m.source_id, m.gsis_id
        FROM games g
        JOIN roster_status r ON r.season = g.season AND r.week = g.week AND r.team IN (g.home_team, g.away_team)
        JOIN player_id_map m ON m.gsis_id = r.gsis_id AND m.source = 'name'
        WHERE g.season >= 2023""").fetchall()
    out: dict = collections.defaultdict(lambda: collections.defaultdict(set))
    for game_id, name, gsis in rows:
        out[game_id][name].add(gsis)
    return out


def parse_odds_file(path: Path) -> tuple[dict, list[dict], list[dict]]:
    blob = json.loads(path.read_text())
    req, resp = blob["request"], blob["response"]
    if not isinstance(resp, dict) or "data" not in resp or not resp["data"]:
        return req, [], []
    snap_ts = ts(resp.get("timestamp"))
    ev = resp["data"]
    props, games = [], []
    tag = path.stem.rsplit("_", 1)[1]
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            key = mk["key"]
            upd = ts(mk.get("last_update") or bk.get("last_update"))
            if key in PROP_MARKETS:
                pairs: dict = {}
                for o in mk.get("outcomes", []):
                    side = o.get("name", "").lower()
                    if side not in ("over", "under") or o.get("point") is None:
                        continue
                    k = (o.get("description", ""), float(o["point"]))
                    pairs.setdefault(k, {})[side] = float(o["price"])
                for (player, line), pr in pairs.items():
                    props.append({"game_id": req["game_id"], "snapshot_tag": tag, "snapshot_ts": snap_ts,
                                  "book": bk["key"], "market": key, "player_name": player, "line": line,
                                  "over_price": pr.get("over"), "under_price": pr.get("under"),
                                  "book_update_ts": upd})
            elif key in ("spreads", "totals"):
                for o in mk.get("outcomes", []):
                    games.append({"game_id": req["game_id"], "book": bk["key"], "snapshot_ts": snap_ts,
                                  "market": key, "name": o.get("name"), "point": o.get("point"),
                                  "home_team": ev.get("home_team")})
    return req, props, games


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    a = ap.parse_args(argv)
    raw = Path(a.raw_dir)

    conn = db.connect()
    db.apply_schema(conn)
    kick = dict(conn.execute("SELECT game_id, kickoff_utc FROM games WHERE season >= 2023").fetchall())
    cands = roster_candidates(conn)

    events, props, glines = {}, [], []
    files = sorted(p for p in raw.glob("*/*.json") if p.parent.name != "events")
    for f in files:
        req, pr, gl = parse_odds_file(f)
        if "event_id" in req:
            events[req["event_id"]] = req["game_id"]
        props += pr
        glines += gl

    # player matching, restricted to the game's two rosters
    status_counts = collections.Counter()
    unmatched = collections.Counter()
    for p in props:
        hits = cands.get(p["game_id"], {}).get(names.normalize(p["player_name"]), set())
        if len(hits) == 1:
            p["gsis_id"], p["match_status"] = next(iter(hits)), "matched"
        else:
            p["gsis_id"] = None
            p["match_status"] = "ambiguous" if hits else "unmatched"
            unmatched[(p["player_name"], p["match_status"])] += 1
        p["known_at"] = p["snapshot_ts"]
        status_counts[p["match_status"]] += 1

    # consensus game lines from the bet snapshot, home-team perspective, one row per book
    gl_rows = {}
    for g in glines:
        k = (g["game_id"], g["book"], g["snapshot_ts"])
        row = gl_rows.setdefault(k, {"game_id": g["game_id"], "book": g["book"], "snapshot_ts": g["snapshot_ts"],
                                     "is_closing": False, "spread_line": None, "total_line": None})
        if g["market"] == "spreads" and g["name"] == g["home_team"] and g["point"] is not None:
            row["spread_line"] = -float(g["point"])  # nflverse: positive = home favored
        if g["market"] == "totals" and g["point"] is not None:
            row["total_line"] = float(g["point"])

    with conn.transaction():
        conn.execute("DELETE FROM prop_lines")
        conn.execute("DELETE FROM odds_events")
        conn.execute("DELETE FROM game_lines WHERE book <> 'nflverse_close'")
        if events:
            db.replace_rows(conn, "odds_events",
                            pl.DataFrame({"event_id": list(events), "game_id": list(events.values())}), "false")
        if props:
            cols = ["game_id", "snapshot_tag", "snapshot_ts", "book", "market", "player_name", "gsis_id",
                    "match_status", "line", "over_price", "under_price", "book_update_ts", "known_at"]
            df = pl.DataFrame([{c: p[c] for c in cols} for p in props], infer_schema_length=None)
            df = df.unique(subset=["game_id", "snapshot_tag", "book", "market", "player_name", "line"], keep="last")
            db.replace_rows(conn, "prop_lines", df, "false")
        if gl_rows:
            gdf = pl.DataFrame(list(gl_rows.values()), infer_schema_length=None) \
                    .filter(pl.col("spread_line").is_not_null() | pl.col("total_line").is_not_null())
            db.replace_rows(conn, "game_lines", gdf.with_columns(
                pl.lit(None, dtype=pl.Int32).alias("home_moneyline"),
                pl.lit(None, dtype=pl.Int32).alias("away_moneyline")), "false")
    # Earlier reads opened an implicit transaction, so the block above was only a
    # savepoint inside it. Commit explicitly or closing the connection rolls it back.
    conn.commit()

    unmatched_path = raw.parent / "unmatched.csv"   # data/odds/unmatched.csv by default
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    with unmatched_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_name", "status", "rows"])
        for (name, st), n in unmatched.most_common():
            w.writerow([name, st, n])

    checks = run_checks(conn, kick)
    conn.close()
    out = {"files": len(files), "events": len(events), "prop_rows": len(props),
           "match": dict(status_counts), "game_line_rows": len(gl_rows), "checks": checks}
    print(json.dumps(out, indent=2, default=str))
    return out


def run_checks(conn, kick) -> list[dict]:
    res = []
    (n_bad_bet,) = conn.execute("""
        SELECT count(*) FROM prop_lines p JOIN games g USING (game_id)
        WHERE p.snapshot_tag = 'bet' AND p.known_at > g.kickoff_utc - interval '36 hours'""").fetchone()
    (n_bad_close,) = conn.execute("""
        SELECT count(*) FROM prop_lines p JOIN games g USING (game_id)
        WHERE p.snapshot_tag = 'close' AND p.known_at > g.kickoff_utc""").fetchone()
    res.append({"name": "snapshots_before_cutoff", "passed": n_bad_bet == 0 and n_bad_close == 0,
                "detail": f"bet rows after kickoff-36h: {n_bad_bet}; close rows after kickoff: {n_bad_close}"})
    tot, matched = conn.execute("""
        SELECT count(*), count(*) FILTER (WHERE match_status = 'matched') FROM prop_lines p
        WHERE snapshot_tag = 'close'""").fetchone()
    res.append({"name": "close_rows_matched_90pct", "passed": tot > 0 and matched / tot >= 0.9,
                "detail": f"{matched:,} of {tot:,} close rows matched ({matched / max(tot, 1):.1%})"})
    (bad_price,) = conn.execute("""
        SELECT count(*) FROM prop_lines WHERE over_price <= 1 OR under_price <= 1""").fetchone()
    res.append({"name": "prices_valid", "passed": bad_price == 0, "detail": f"{bad_price} rows with price <= 1"})
    return res


if __name__ == "__main__":
    main()
