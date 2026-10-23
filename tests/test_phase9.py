"""Phase 9: role signal is prior-only; score simulation basics; live-lines collector against a fake Odds API."""
import datetime as dt
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import polars as pl
import pytest

from proplab import db
from proplab.p3 import engine as E
from proplab.p9 import lines as LN
from proplab.p9 import roles as R
from proplab.p9 import scores as S

UTC = dt.timezone.utc


@pytest.fixture(scope="module")
def conn():
    c = db.connect()
    yield c
    c.close()


def test_role_signal_ignores_the_game_itself(conn, monkeypatch):
    D = E.load(conn)
    base = R.role_signal(conn, D["u"]).sort("gsis_id", "game_id")
    orig = R._q
    w9 = "2025_09"
    monkeypatch.setattr(R, "_q", lambda c, sql: orig(c, sql).with_columns(
        pl.when(pl.col("game_id").str.starts_with(w9)).then(pl.lit(1.0)).otherwise(pl.col("pct")).alias("pct")))
    changed = R.role_signal(conn, D["u"]).sort("gsis_id", "game_id")
    wk9 = pl.col("game_id").str.starts_with(w9)
    assert base.filter(wk9).equals(changed.filter(wk9)), "week-9 role signal saw week-9 snaps"
    assert not base.filter(pl.col("game_id").str.starts_with("2025_10"))["role_r"].equals(
        changed.filter(pl.col("game_id").str.starts_with("2025_10"))["role_r"])


def test_score_events_and_normal_baseline():
    rng = np.random.default_rng(0)
    h, a = S.draws_N(3.0, 45.0, 200000, rng)
    e = S.events(h, a, 3.0, 45.0)
    assert e["cover"] == pytest.approx(0.5, abs=0.01) and e["over"] == pytest.approx(0.5, abs=0.01)
    assert e["win"] == pytest.approx(0.587, abs=0.01)   # Normal(3, 13.5) above zero


def test_score_library_uses_only_earlier_seasons(conn):
    g = S.games(conn)
    lib = S.library(g, 2025)
    seasons = g.filter(pl.col("home_score").is_not_null() & pl.col("spread").is_not_null() & pl.col("season").is_between(2020, 2024))
    assert len(lib) == seasons.height


class _Fake(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Fake.calls.append(self.path)
        if "/events/" in self.path and "/odds" in self.path:
            body = {"id": "ev1", "home_team": "Green Bay Packers", "away_team": "Carolina Panthers", "bookmakers": [
                {"key": b, "markets": [{"key": "player_receptions", "outcomes": [
                    {"name": "Over", "description": "Christian Watson", "point": 3.5, "price": po},
                    {"name": "Under", "description": "Christian Watson", "point": 3.5, "price": pu}]}]}
                for b, po, pu in (("bookA", 1.91, 1.91), ("bookB", 2.0, 1.83))]}
            hdr = {"x-requests-last": "8", "x-requests-remaining": "15000"}
        else:
            body = [{"id": "ev1", "home_team": "Green Bay Packers", "away_team": "Carolina Panthers",
                     "commence_time": "2026-10-30T00:15:00Z"}]
            hdr = {"x-requests-last": "0", "x-requests-remaining": "15008"}
        data = json.dumps(body).encode()
        self.send_response(200)
        for k, v in hdr.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


def test_live_lines_collector(tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(LN, "OUT", tmp_path / "market_lines.json")
    now = dt.datetime(2026, 10, 27, 12, tzinfo=UTC)
    store = LN.main(now=now, base_url=f"http://127.0.0.1:{srv.server_port}")
    srv.shutdown()
    snap = [s for s in store["snapshots"] if s["game_id"] == "2026_08_CAR_GB"]
    assert len(snap) == 1
    row = snap[0]["lines"][0]
    assert row["line"] == 3.5 and row["books"] == 2 and row["market"] == "wrte_rec" and row["gsis_id"]
    assert 0.47 < row["p_over_market"] < 0.5          # median of 0.5 and the no-vig chance of the 2.0/1.83 pair
    assert store["last_run"]["credits"] == 8
    assert not any("apiKey=test-key" in json.dumps(store) for _ in [0])   # the key is never stored
    # a second run within 24 hours does not buy the same game again
    _Fake.calls.clear()
    srv2 = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    LN.main(now=now + dt.timedelta(hours=2), base_url=f"http://127.0.0.1:{srv2.server_port}")
    srv2.shutdown()
    assert not any("/odds" in c for c in _Fake.calls)


def test_live_lines_skip_without_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert LN.main() == {"skipped": True}
