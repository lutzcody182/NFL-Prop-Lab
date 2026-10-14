#!/usr/bin/env python3
"""Download historical NFL prop odds from The Odds API into data/odds/raw/.

Runs with plain Python 3.9+ and nothing to install. Run it from the project folder:

    python3 tools/odds_backfill.py --estimate              # shows the cost, uses no credits
    python3 tools/odds_backfill.py --run --max-credits 60000

Your API key is read from the ODDS_API_KEY environment variable, or typed at a
hidden prompt. It is never printed or saved to a file.

Safe to stop and restart: files already downloaded are skipped. Re-run it each week
to add new games. Raw responses are saved unchanged; loading into the database
happens later (proplab/odds/load.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "schedule.json"
RAW = ROOT / "data" / "odds" / "raw"
BASE_URL = "https://api.the-odds-api.com"
SPORT = "americanfootball_nfl"
PROP_MARKETS = ["player_receptions", "player_reception_yds"]
GAME_MARKETS = ["spreads", "totals"]
SNAPSHOTS = {
    # tag: (time before kickoff, markets)
    "bet": (dt.timedelta(hours=36), PROP_MARKETS + GAME_MARKETS),
    "close": (dt.timedelta(minutes=5), PROP_MARKETS),
}
COST_PER_MARKET = 10  # per region per market per event (historical event odds)
EVENT_MATCH_HOURS = 12

TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


class Client:
    def __init__(self, key: str, base_url: str, max_credits: int):
        self.key, self.base_url, self.max_credits = key, base_url.rstrip("/"), max_credits
        self.spent = 0
        self.remaining: int | None = None
        self.ctx = ssl.create_default_context()

    def get(self, path: str, params: dict) -> tuple[object, dict]:
        query = urllib.parse.urlencode({**params, "apiKey": self.key})
        url = f"{self.base_url}{path}?{query}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60, context=self.ctx) as r:
                    headers = {k.lower(): v for k, v in r.headers.items()}
                    body = json.loads(r.read().decode())
                last = int(headers.get("x-requests-last", "0") or 0)
                self.spent += last
                if "x-requests-remaining" in headers:
                    self.remaining = int(float(headers["x-requests-remaining"]))
                return body, {k: headers[k] for k in headers if k.startswith("x-requests")}
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code == 401:
                    sys.exit("The API key was rejected (401). Check it and try again.")
                if e.code == 422:
                    return {"error": msg, "status": 422}, {}
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** attempt * 2)
                    continue
                sys.exit(f"HTTP {e.code} from {path}: {msg}")
            except urllib.error.URLError as e:
                if "CERTIFICATE_VERIFY_FAILED" in str(e):
                    sys.exit("Python can't verify HTTPS certificates. On a Mac with python.org Python, open "
                             "Applications > Python 3.x and double-click 'Install Certificates.command', then retry.")
                if attempt < 4:
                    time.sleep(2 ** attempt * 2)
                    continue
                sys.exit(f"Network error: {e}")
        sys.exit("Gave up after repeated errors.")

    def can_afford(self, cost: int) -> bool:
        if self.spent + cost > self.max_credits:
            return False
        if self.remaining is not None and self.remaining - cost < 50:
            return False
        return True


def load_schedule(now: dt.datetime, seasons: list[int]) -> list[dict]:
    games = json.loads(SCHEDULE.read_text())
    out = [g for g in games if g["season"] in seasons and g["game_type"] == "REG"
           and parse_ts(g["kickoff_utc"]) - SNAPSHOTS["close"][0] < now]
    return sorted(out, key=lambda g: g["kickoff_utc"])


def weeks_of(games: list[dict]) -> dict:
    wk: dict = {}
    for g in games:
        wk.setdefault((g["season"], g["week"]), []).append(g)
    return wk


def event_file(season: int, week: int) -> Path:
    return RAW / "events" / f"{season}_{week:02d}.json"


def odds_file(g: dict, tag: str) -> Path:
    return RAW / str(g["season"]) / f"{g['game_id']}_{tag}.json"


def match_events(events: list[dict], games: list[dict]) -> dict:
    """game_id -> event id, by teams and kickoff within EVENT_MATCH_HOURS."""
    out = {}
    for g in games:
        k = parse_ts(g["kickoff_utc"])
        for e in events:
            if (TEAM_NAMES.get(e.get("home_team")) == g["home_team"]
                    and TEAM_NAMES.get(e.get("away_team")) == g["away_team"]
                    and abs((parse_ts(e["commence_time"]) - k).total_seconds()) <= EVENT_MATCH_HOURS * 3600):
                out[g["game_id"]] = e["id"]
                break
    return out


def estimate(games: list[dict]) -> dict:
    weeks = weeks_of(games)
    todo_events = sum(1 for (s, w) in weeks if not event_file(s, w).exists())
    per_game = {t: COST_PER_MARKET * len(m) for t, (_, m) in SNAPSHOTS.items()}
    todo = {t: sum(1 for g in games if not odds_file(g, t).exists()) for t in SNAPSHOTS}
    credits = todo_events + sum(todo[t] * per_game[t] for t in SNAPSHOTS)
    return {"games": len(games), "weeks": len(weeks), "event_lookups_todo": todo_events,
            "snapshots_todo": todo, "max_credits_needed": credits}


def save(path: Path, request: dict, body: object, headers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_utc": iso(dt.datetime.now(dt.timezone.utc)), "request": request,
               "quota_headers": headers, "response": body}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def run(client: Client, games: list[dict]) -> None:
    weeks = weeks_of(games)
    unmatched = []
    for (season, week), wgames in sorted(weeks.items()):
        ef = event_file(season, week)
        if not ef.exists():
            if not client.can_afford(1):
                print("Credit limit reached; stopping. Re-run later to continue.")
                return
            first = min(parse_ts(g["kickoff_utc"]) for g in wgames)
            last = max(parse_ts(g["kickoff_utc"]) for g in wgames)
            req = {"date": iso(first - dt.timedelta(hours=40)),
                   "commenceTimeFrom": iso(first - dt.timedelta(hours=2)),
                   "commenceTimeTo": iso(last + dt.timedelta(hours=2))}
            body, headers = client.get(f"/v4/historical/sports/{SPORT}/events", req)
            save(ef, req, body, headers)
        events = json.loads(ef.read_text())["response"]
        events = events.get("data", []) if isinstance(events, dict) else []
        ids = match_events(events, wgames)

        for g in wgames:
            if g["game_id"] not in ids:
                unmatched.append(g["game_id"])
                continue
            kick = parse_ts(g["kickoff_utc"])
            for tag, (before, markets) in SNAPSHOTS.items():
                of = odds_file(g, tag)
                if of.exists():
                    continue
                cost = COST_PER_MARKET * len(markets)
                if not client.can_afford(cost):
                    print(f"Credit limit reached at {g['game_id']}; stopping. Re-run later to continue.")
                    return
                req = {"regions": "us", "markets": ",".join(markets), "oddsFormat": "decimal",
                       "dateFormat": "iso", "date": iso(kick - before)}
                body, headers = client.get(f"/v4/historical/sports/{SPORT}/events/{ids[g['game_id']]}/odds", req)
                save(of, {**req, "event_id": ids[g["game_id"]], "game_id": g["game_id"]}, body, headers)
                time.sleep(0.2)
        left = f", {client.remaining:,} left on plan" if client.remaining is not None else ""
        print(f"  {season} week {week:2d}: done ({client.spent:,} credits used this run{left})", flush=True)
    if unmatched:
        print(f"{len(unmatched)} games had no matching odds event: {', '.join(unmatched[:10])}"
              + (" ..." if len(unmatched) > 10 else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download historical NFL prop odds (The Odds API).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate", action="store_true", help="show how many credits are needed; uses none")
    mode.add_argument("--run", action="store_true", help="download")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    ap.add_argument("--max-credits", type=int, default=60000, help="stop before spending more than this")
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)  # for tests
    ap.add_argument("--now", help=argparse.SUPPRESS)                        # for tests
    ap.add_argument("--raw-dir", help=argparse.SUPPRESS)                    # for tests
    ap.add_argument("--only-weeks", type=int, nargs="+", help=argparse.SUPPRESS)  # for tests
    a = ap.parse_args(argv)
    if a.raw_dir:
        global RAW
        RAW = Path(a.raw_dir)

    now = parse_ts(a.now) if a.now else dt.datetime.now(dt.timezone.utc)
    games = load_schedule(now, a.seasons)
    if a.only_weeks:
        games = [g for g in games if g["week"] in a.only_weeks]
    est = estimate(games)
    print(f"{est['games']} games already played, in {est['weeks']} weeks. Still to download: "
          f"{est['event_lookups_todo']} week lookups, {est['snapshots_todo']['bet']} bet-time and "
          f"{est['snapshots_todo']['close']} closing snapshots.")
    print(f"At most {est['max_credits_needed']:,} credits (games with no posted props cost less).")
    if a.estimate:
        return 0

    key = os.environ.get("ODDS_API_KEY") or getpass.getpass("Paste your Odds API key (hidden): ").strip()
    if not key:
        sys.exit("No API key given.")
    client = Client(key, a.base_url, a.max_credits)
    run(client, games)
    left = f"; {client.remaining:,} left on your plan" if client.remaining is not None else ""
    print(f"Finished. {client.spent:,} credits used this run{left}.")
    print("Next: tell Claude the download is done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
