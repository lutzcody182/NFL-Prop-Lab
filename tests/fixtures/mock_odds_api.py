"""A local stand-in for The Odds API, for testing only.

!!! EVERY PRICE AND LINE THIS SERVES IS SYNTHETIC (made up by a random number
generator). It exists to exercise the download -> load -> evaluate pipeline end
to end. Results computed from it mean nothing about real markets. !!!

Players, teams and kickoff times come from the local database so that player
matching is exercised on real names; one invented name is added per game to
check that unmatched names are reported, not guessed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import http.server
import json
import threading
import urllib.parse

INVERSE_TEAMS = None


def _rng_unit(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _floor5(t: dt.datetime) -> dt.datetime:
    return t.replace(minute=t.minute - t.minute % 5, second=0, microsecond=0)


def build_world(conn, team_names: dict) -> dict:
    inv = {v: k for k, v in team_names.items()}
    games = conn.execute("""
        SELECT game_id, home_team, away_team, kickoff_utc FROM games WHERE season >= 2023 AND game_type = 'REG'""").fetchall()
    players = conn.execute("""
        SELECT g.game_id, p.display_name, r.position
        FROM games g JOIN roster_status r ON r.season = g.season AND r.week = g.week
             AND r.team IN (g.home_team, g.away_team) AND r.status = 'ACT' AND r.position IN ('WR', 'TE')
        JOIN players p ON p.gsis_id = r.gsis_id
        JOIN player_game pg ON pg.gsis_id = r.gsis_id AND pg.game_id = g.game_id AND pg.targets >= 2
        WHERE g.season >= 2023""").fetchall()
    by_game: dict = {}
    for gid, name, pos in players:
        by_game.setdefault(gid, []).append(name)
    return {"games": {g[0]: {"home": inv[g[1]], "away": inv[g[2]], "kickoff": g[3]} for g in games},
            "players": by_game}


def make_handler(world: dict, state: dict):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, body: object, cost: int) -> None:
            state["used"] += cost
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-requests-last", str(cost))
            self.send_header("x-requests-used", str(state["used"]))
            self.send_header("x-requests-remaining", str(state["quota"] - state["used"]))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            if q.get("apiKey") != state["key"]:
                self.send_response(401)
                self.end_headers()
                return
            date = dt.datetime.fromisoformat(q["date"].replace("Z", "+00:00"))
            snap = _floor5(date)
            parts = url.path.strip("/").split("/")
            if parts[-1] == "events":
                lo = dt.datetime.fromisoformat(q["commenceTimeFrom"].replace("Z", "+00:00"))
                hi = dt.datetime.fromisoformat(q["commenceTimeTo"].replace("Z", "+00:00"))
                data = [{"id": f"synth-{gid}", "sport_key": "americanfootball_nfl", "home_team": g["home"],
                         "away_team": g["away"], "commence_time": g["kickoff"].strftime("%Y-%m-%dT%H:%M:%SZ")}
                        for gid, g in world["games"].items() if lo <= g["kickoff"] <= hi]
                return self._send({"timestamp": snap.strftime("%Y-%m-%dT%H:%M:%SZ"), "data": data}, 1 if data else 0)
            gid = parts[-2].removeprefix("synth-")
            g = world["games"][gid]
            markets = q["markets"].split(",")
            tag = "close" if (g["kickoff"] - date) < dt.timedelta(hours=1) else "bet"
            books = []
            for book in ("draftkings", "fanduel", "betmgm"):
                mks = []
                for mk in markets:
                    outs = []
                    if mk.startswith("player_"):
                        for name in world["players"].get(gid, []) + ["Synthetic Nobody"]:
                            base = 2.5 + round(_rng_unit(gid, name, mk) * 4) if mk == "player_receptions" \
                                else 24.5 + 10 * round(_rng_unit(gid, name, mk) * 5)
                            shift = 1 if (tag == "close" and _rng_unit(gid, name, mk, "mv") > 0.8) else 0
                            line = base + shift * (1 if mk == "player_receptions" else 5)
                            tilt = (_rng_unit(gid, name, mk, book, tag) - 0.5) * 0.3
                            outs += [{"name": "Over", "description": name, "price": round(1.91 + tilt, 2), "point": line},
                                     {"name": "Under", "description": name, "price": round(1.91 - tilt, 2), "point": line}]
                    elif mk == "spreads":
                        sp = round((_rng_unit(gid, "sp") - 0.5) * 14 * 2) / 2
                        outs = [{"name": g["home"], "price": 1.91, "point": sp}, {"name": g["away"], "price": 1.91, "point": -sp}]
                    elif mk == "totals":
                        tot = 38.5 + round(_rng_unit(gid, "tot") * 12)
                        outs = [{"name": "Over", "price": 1.91, "point": tot}, {"name": "Under", "price": 1.91, "point": tot}]
                    mks.append({"key": mk, "last_update": snap.strftime("%Y-%m-%dT%H:%M:%SZ"), "outcomes": outs})
                books.append({"key": book, "title": book, "last_update": snap.strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": mks})
            body = {"timestamp": snap.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "data": {"id": f"synth-{gid}", "home_team": g["home"], "away_team": g["away"],
                             "commence_time": g["kickoff"].strftime("%Y-%m-%dT%H:%M:%SZ"), "bookmakers": books}}
            return self._send(body, 10 * len(markets))
    return H


def serve(world: dict, key: str, quota: int = 1_000_000):
    state = {"used": 0, "quota": quota, "key": key}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(world, state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state
