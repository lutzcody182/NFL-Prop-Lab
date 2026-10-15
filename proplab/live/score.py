"""Score frozen ledger weeks once their games are final (see phase5/PREREGISTRATION.md).

    python -m proplab.live.score     # -> phase5/live_record.json

A week is scored only when every game has a result AND 72 hours have passed since its last
kickoff (the NFL's Monday-Wednesday stat corrections). Frozen files are verified against their
manifest fingerprints first; a mismatch stops the scoring.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.p1.model import crps_samples, ece
from proplab.p3.engine import coverage_tolerance, ece_null95

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "ledger"
OUT = ROOT / "phase5" / "live_record.json"
CORRECTIONS_WAIT = dt.timedelta(hours=72)

STAT = {  # market -> (outcome column(s) in player_game)
    "wrte_rec": "receptions", "wrte_rec_yds": "receiving_yards",
    "qb_pass_yds": "passing_yards", "qb_pass_att": "attempts", "qb_pass_tds": "passing_tds",
    "qb_ints": "passing_interceptions", "qb_rush_yds": "rushing_yards", "rb_rush_yds": "rushing_yards",
    "rb_rush_att": "carries", "rb_rec": "receptions", "rb_rec_yds": "receiving_yards", "anytime_td": "anytime_td",
}
SIM_STAT = {"Pass yds": "passing_yards", "Completions": "completions", "Pass TDs": "passing_tds",
            "Pass attempts": "attempts", "Rec yds": "receiving_yards", "Receptions": "receptions"}


def _q(conn, sql, params=()):
    cur = conn.execute(sql, params)
    return pl.DataFrame(cur.fetchall(), schema=[d.name for d in cur.description], orient="row", infer_schema_length=None)


def verify(week_dir: Path) -> dict:
    man = json.loads((week_dir / "manifest.json").read_text())
    for name, info in man["files"].items():
        h = hashlib.sha256((week_dir / name).read_bytes()).hexdigest()
        if h != info["sha256"]:
            raise SystemExit(f"{week_dir.name}/{name} does not match its frozen fingerprint; refusing to score.")
    return man


def outcomes(conn, season: int, week: int) -> pl.DataFrame:
    """Every player active (ACT) in a game that week, with his stats (0 when he has no stat row)."""
    return _q(conn, """
        SELECT r.gsis_id, p.display_name AS name, r.team, g.game_id,
               coalesce(pg.receptions, 0) AS receptions, coalesce(pg.receiving_yards, 0)::float AS receiving_yards,
               coalesce(pg.passing_yards, 0)::float AS passing_yards, coalesce(pg.attempts, 0) AS attempts,
               coalesce(pg.completions, 0) AS completions, coalesce(pg.passing_tds, 0) AS passing_tds,
               coalesce(pg.passing_interceptions, 0) AS passing_interceptions,
               coalesce(pg.rushing_yards, 0)::float AS rushing_yards, coalesce(pg.carries, 0) AS carries,
               (coalesce(pg.rushing_tds, 0) + coalesce(pg.receiving_tds, 0) > 0)::int AS anytime_td
        FROM roster_status r
        JOIN games g ON g.season = r.season AND g.week = r.week AND r.team IN (g.home_team, g.away_team)
        JOIN players p ON p.gsis_id = r.gsis_id
        LEFT JOIN player_game pg ON pg.gsis_id = r.gsis_id AND pg.game_id = g.game_id
        WHERE r.status = 'ACT' AND g.season = %s AND g.week = %s AND g.game_type = 'REG'""", (season, week))


def week_ready(conn, season: int, week: int, now: dt.datetime) -> tuple[bool, str]:
    n, done, last = conn.execute("""
        SELECT count(*), count(r.game_id), max(g.kickoff_utc) FROM games g LEFT JOIN game_results r USING (game_id)
        WHERE g.season = %s AND g.week = %s AND g.game_type = 'REG'""", (season, week)).fetchone()
    if done < n:
        return False, f"{done} of {n} games final"
    if now < last + CORRECTIONS_WAIT:
        return False, f"waiting for stat corrections until {(last + CORRECTIONS_WAIT).isoformat(timespec='minutes')}"
    return True, "ready"


def pinball(q: np.ndarray, levels: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Average quantile (pinball) loss; 2x this approximates CRPS."""
    diff = y[:, None] - q
    return (np.maximum(levels * diff, (levels - 1) * diff)).mean(axis=1) * 2


def ledger_rows(week_dir: Path) -> pl.DataFrame:
    """Flatten a frozen week's projections into one row per (market, player)."""
    rows = []
    w = json.loads((week_dir / "wrte.json").read_text())
    for p in w["players"]:
        for mk, key, lines in (("wrte_rec", "rec", w["meta"]["rec_lines"]), ("wrte_rec_yds", "yds", w["meta"]["yds_lines"])):
            d = p[key]
            rows.append({"market": mk, "player": p["player"], "gsis_id": p.get("gsis_id"), "team": p["team"],
                         "game_id": p["game_id"], "mean": d["mean"], "p10": d["p10"], "p90": d["p90"],
                         "q": d.get("q"), "over": {str(L): d["over"][str(L)] for L in lines}})
    m = json.loads((week_dir / "markets.json").read_text())
    for mk, v in m["markets"].items():
        for p in v["players"]:
            rows.append({"market": mk, "player": p["player"], "gsis_id": p.get("gsis_id"), "team": p["team"],
                         "game_id": p["game_id"], "mean": p["mean"], "p10": p["p10"], "p90": p["p90"],
                         "q": p.get("q"), "over": p["over"]})
    return rows


def match(rows: list[dict], out: pl.DataFrame) -> list[dict]:
    """Attach outcomes; match by gsis_id when stored, else by (game, team, name)."""
    by_id = {(r["gsis_id"], r["game_id"]): r for r in out.iter_rows(named=True)}
    by_name = {(r["game_id"], r["team"], r["name"]): r for r in out.iter_rows(named=True)}
    for r in rows:
        o = by_id.get((r["gsis_id"], r["game_id"])) if r["gsis_id"] else by_name.get((r["game_id"], r["team"], r["player"]))
        r["outcome"] = None if o is None else float(o[STAT[r["market"]]])
    return rows


def market_metrics(rows: list[dict], rng) -> dict:
    played = [r for r in rows if r["outcome"] is not None]
    res = {"projected": len(rows), "scored": len(played), "void_inactive": len(rows) - len(played)}
    if not played:
        return res
    y = np.array([r["outcome"] for r in played])
    if played[0]["market"] == "anytime_td":
        p = np.array([r["mean"] for r in played])
        res.update({"brier": float(((p - y) ** 2).mean()), "ece": ece(p, y), "ece_null95": ece_null95(p, rng),
                    "pred_rate": float(p.mean()), "actual_rate": float(y.mean())})
        return res
    lo, hi = np.array([r["p10"] for r in played]), np.array([r["p90"] for r in played])
    probs, outs = [], []
    for r in played:
        for L, pr in r["over"].items():
            probs.append(pr)
            outs.append(float(r["outcome"] > float(L)))
    probs, outs = np.array(probs), np.array(outs)
    res.update({"coverage80": float(((y >= lo) & (y <= hi)).mean()), "coverage_tol": coverage_tolerance(len(y)),
                "brier": float(((probs - outs) ** 2).mean()), "ece": ece(probs, outs), "ece_null95": ece_null95(probs, rng),
                "mean_pred": float(np.mean([r["mean"] for r in played])), "mean_actual": float(y.mean())})
    if all(r["q"] for r in played):
        levels = np.array([round(0.05 * i, 2) for i in range(1, 20)])
        res["quantile_score"] = float(pinball(np.array([r["q"] for r in played]), levels, y).mean())
    return res


def sim_metrics(week_dir: Path, out: pl.DataFrame) -> dict:
    s = json.loads((week_dir / "sims.json").read_text())
    by_name = {(r["game_id"], r["team"], r["name"]): r for r in out.iter_rows(named=True)}
    crps, legs = [], []
    for g in s["games"]:
        for team, t in g["teams"].items():
            vals = {}
            for p in t["players"]:
                o = by_name.get((g["game_id"], team, p["name"]))
                if o is None:
                    continue
                for stat, b in p["stats"].items():
                    draws = np.frombuffer(base64.b64decode(b), dtype="<i2").astype(float)
                    y = float(o[SIM_STAT[stat]])
                    crps.append({"stat": stat, "crps": float(crps_samples(draws[None, :], np.array([y]))[0])})
                    vals[(p["pos"], p["name"], stat)] = (draws, y)
            qb = [k for k in vals if k[0] == "QB" and k[2] == "Pass yds"]
            wr = [k for k in vals if k[0] != "QB" and k[2] == "Rec yds"]
            if qb and wr:
                (qd, qy), (wd, wy) = vals[qb[0]], vals[wr[0]]
                mq, mw = np.median(qd), np.median(wd)
                legs.append({"p_joint": float(np.mean((qd > mq) & (wd > mw))), "p_ind": float(np.mean(qd > mq) * np.mean(wd > mw)),
                             "hit": float(qy > mq and wy > mw)})
    df = pl.DataFrame(crps) if crps else None
    res = {"players_scored": 0 if df is None else df.height}
    if df is not None:
        res["crps_by_stat"] = {k: float(v) for k, v in df.group_by("stat").agg(pl.col("crps").mean()).iter_rows()}
    if legs:
        L = pl.DataFrame(legs)
        res.update({"two_leg_teams": L.height, "two_leg_actual": float(L["hit"].mean()),
                    "two_leg_pred_joint": float(L["p_joint"].mean()), "two_leg_pred_ind": float(L["p_ind"].mean()),
                    "brier_joint": float(((L["p_joint"] - L["hit"]) ** 2).mean()),
                    "brier_ind": float(((L["p_ind"] - L["hit"]) ** 2).mean())})
    return res


def main(now: dt.datetime | None = None, ledger: Path = LEDGER, out_path: Path = OUT) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    conn = db.connect()
    rng = np.random.default_rng(5)
    record = {"updated_utc": now.isoformat(timespec="minutes"), "weeks": [], "cumulative": {}, "flags": []}
    all_rows: dict = {}
    for wd in sorted(p for p in ledger.glob("*_w*") if (p / "manifest.json").exists()):
        man = verify(wd)
        ready, why = week_ready(conn, man["season"], man["week"], now)
        entry = {"season": man["season"], "week": man["week"], "frozen_utc": man["frozen_utc"],
                 "minutes_before_kickoff": man["minutes_before_kickoff"], "status": "scored" if ready else why}
        if ready:
            out = outcomes(conn, man["season"], man["week"])
            rows = match(ledger_rows(wd), out)
            by_m: dict = {}
            for r in rows:
                by_m.setdefault(r["market"], []).append(r)
            entry["markets"] = {m: market_metrics(v, rng) for m, v in by_m.items()}
            entry["sims"] = sim_metrics(wd, out)
            for m, v in by_m.items():
                all_rows.setdefault(m, []).extend(v)
        record["weeks"].append(entry)
    for m, rows in all_rows.items():
        c = market_metrics(rows, rng)
        record["cumulative"][m] = c
        if "coverage80" in c and abs(c["coverage80"] - 0.8) > c["coverage_tol"]:
            record["flags"].append({"market": m, "issue": f"80% ranges covered {c['coverage80']:.0%} of outcomes"})
        if c.get("scored") and c["ece"] > c["ece_null95"]:
            record["flags"].append({"market": m, "issue": f"over/under probabilities off (ECE {c['ece']:.3f} > noise {c['ece_null95']:.3f})"})
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps({"weeks": [(w["season"], w["week"], w["status"]) for w in record["weeks"]], "flags": record["flags"]}, indent=1))
    return record


if __name__ == "__main__":
    main()
