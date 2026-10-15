"""Phase 5: freeze guard and live scoring plumbing (synthetic ledger built from a finished week)."""
import base64
import datetime as dt
import hashlib
import json

import numpy as np
import pytest

from proplab import db
from proplab.live import freeze, score

UTC = dt.timezone.utc


def _proj(season, week, extra=None):
    return {"meta": {"season": season, "week": week, "lines_fetched_utc": "x", **(extra or {})}, "players": [], "markets": {}, "games": []}


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "schedule.json").write_text(json.dumps(
        [{"season": 2030, "week": 3, "game_type": "REG", "kickoff_utc": "2030-09-20T00:15:00Z"}]))
    srcs = {}
    for name in ("wrte.json", "markets.json", "sims.json"):
        p = tmp_path / f"src_{name}"
        p.write_text(json.dumps(_proj(2030, 3)))
        srcs[name] = p
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(freeze, "SOURCES", srcs)
    return tmp_path


def test_freeze_refuses_after_kickoff(fake_root):
    with pytest.raises(SystemExit, match="Refusing to freeze"):
        freeze.main(now=dt.datetime(2030, 9, 20, 0, 16, tzinfo=UTC))
    assert not (fake_root / "ledger" / "2030_w03").exists()


def test_freeze_writes_once_with_fingerprints(fake_root):
    out = freeze.main(now=dt.datetime(2030, 9, 19, 22, 0, tzinfo=UTC))
    man = json.loads((out / "manifest.json").read_text())
    assert man["minutes_before_kickoff"] == 135
    for name, info in man["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == info["sha256"]
    with pytest.raises(SystemExit, match="never overwritten"):
        freeze.main(now=dt.datetime(2030, 9, 19, 23, 0, tzinfo=UTC))


def test_freeze_refuses_mixed_weeks(fake_root):
    freeze.SOURCES["sims.json"].write_text(json.dumps(_proj(2030, 4)))
    with pytest.raises(SystemExit, match="different weeks"):
        freeze.main(now=dt.datetime(2030, 9, 19, 22, 0, tzinfo=UTC))


def test_pinball_matches_crps_for_normal():
    rng = np.random.default_rng(0)
    levels = np.array([round(0.05 * i, 2) for i in range(1, 20)])
    from scipy.stats import norm
    q = norm.ppf(levels)[None, :].repeat(4000, 0)
    y = rng.normal(size=4000)
    # CRPS of N(0,1) at its own draws averages 1/sqrt(pi) ~ 0.564; a 19-quantile grid lands within ~5% of it
    assert 0.53 < score.pinball(q, levels, y).mean() < 0.61


@pytest.fixture(scope="module")
def conn():
    return db.connect()


def _synthetic_week(tmp_path, conn, season=2026, week=5, tamper=False):
    """Build a 'frozen' week from real outcomes with deliberately wide, calibrated-ish forecasts."""
    out = score.outcomes(conn, season, week)
    rng = np.random.default_rng(1)
    wr = out.sort("receiving_yards", descending=True).head(60)
    levels = [round(0.05 * i, 2) for i in range(1, 20)]
    players = []
    for r in wr.iter_rows(named=True):
        draws_y = np.maximum(0, r["receiving_yards"] + rng.normal(0, 25, 2000))
        draws_r = np.maximum(0, r["receptions"] + rng.normal(0, 2, 2000))
        mk = lambda d, lines: {"mean": float(d.mean()), "p10": float(np.quantile(d, .1)), "p90": float(np.quantile(d, .9)),
                               "q": [float(x) for x in np.quantile(d, levels)], "over": {str(L): float((d > L).mean()) for L in lines}}
        players.append({"player": r["name"], "gsis_id": r["gsis_id"], "team": r["team"], "game_id": r["game_id"],
                        "rec": mk(draws_r, [3.5, 4.5]), "yds": mk(draws_y, [49.5])})
    players.append({"player": "Nobody Inactive", "gsis_id": "00-XXXXXXX", "team": "GB", "game_id": wr["game_id"][0],
                    "rec": players[0]["rec"], "yds": players[0]["yds"]})
    td = [{"player": r["name"], "gsis_id": r["gsis_id"], "team": r["team"], "game_id": r["game_id"], "mean": 0.4,
           "p10": 0, "p90": 1, "over": {"0.5": 0.4}} for r in wr.iter_rows(named=True)]
    g0 = wr.row(0, named=True)
    enc = lambda a: base64.b64encode(np.rint(a).astype("<i2").tobytes()).decode()
    qb = out.filter((out["game_id"] == g0["game_id"]) & (out["team"] == g0["team"])).sort("attempts", descending=True).row(0, named=True)
    sims = {"meta": {}, "games": [{"game_id": g0["game_id"], "teams": {g0["team"]: {"players": [
        {"name": qb["name"], "pos": "QB", "stats": {"Pass yds": enc(rng.normal(230, 60, 1000))}},
        {"name": g0["name"], "pos": "WR", "stats": {"Rec yds": enc(rng.normal(60, 30, 1000))}}]}}}]}
    wd = tmp_path / "ledger" / f"{season}_w{week:02d}"
    wd.mkdir(parents=True)
    files = {"wrte.json": {"meta": {"rec_lines": [3.5, 4.5], "yds_lines": [49.5]}, "players": players},
             "markets.json": {"markets": {"anytime_td": {"players": td}}}, "sims.json": sims}
    man = {"season": season, "week": week, "frozen_utc": "t", "minutes_before_kickoff": 60, "files": {}}
    for n, obj in files.items():
        (wd / n).write_text(json.dumps(obj))
        man["files"][n] = {"sha256": hashlib.sha256((wd / n).read_bytes()).hexdigest()}
    (wd / "manifest.json").write_text(json.dumps(man))
    if tamper:
        (wd / "wrte.json").write_text(json.dumps(files["wrte.json"]) + " ")
    return tmp_path / "ledger"


def test_scoring_waits_for_corrections(tmp_path, conn):
    led = _synthetic_week(tmp_path, conn)
    kick = conn.execute("SELECT max(kickoff_utc) FROM games WHERE season=2026 AND week=5 AND game_type='REG'").fetchone()[0]
    rec = score.main(now=kick + dt.timedelta(hours=24), ledger=led, out_path=tmp_path / "r.json")
    assert rec["weeks"][0]["status"].startswith("waiting for stat corrections")
    assert rec["cumulative"] == {}


def test_scoring_end_to_end(tmp_path, conn):
    led = _synthetic_week(tmp_path, conn)
    rec = score.main(now=dt.datetime(2026, 12, 1, tzinfo=UTC), ledger=led, out_path=tmp_path / "r.json")
    wk = rec["weeks"][0]
    assert wk["status"] == "scored"
    m = wk["markets"]["wrte_rec_yds"]
    assert m["projected"] == 61 and m["scored"] == 60 and m["void_inactive"] == 1
    assert 0.6 < m["coverage80"] <= 1.0 and m["quantile_score"] > 0
    assert 0 <= wk["markets"]["anytime_td"]["brier"] <= 1
    assert wk["sims"]["players_scored"] == 2 and wk["sims"]["two_leg_teams"] == 1
    assert json.loads((tmp_path / "r.json").read_text())["weeks"][0]["status"] == "scored"


def test_scoring_refuses_tampered_ledger(tmp_path, conn):
    led = _synthetic_week(tmp_path, conn, tamper=True)
    with pytest.raises(SystemExit, match="frozen fingerprint"):
        score.main(now=dt.datetime(2026, 12, 1, tzinfo=UTC), ledger=led, out_path=tmp_path / "r.json")
