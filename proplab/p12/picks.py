"""Phase 12: tracked picks at US-licensed books (phase12/PREREGISTRATION.md). Not betting advice.

    python -m proplab.p12.picks     # in the weekly run: log picks for games with no later run before kickoff, grade old ones

Picks are written once per game and never changed; grading only fills in the result fields.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from proplab import db
from proplab.p12.books import LICENSED, STATE, allowed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase12" / "picks.json"
LINES = ROOT / "phase9" / "market_lines.json"
LEDGER = ROOT / "ledger"
FIRST = (2026, 9)
MARKETS = {"wrte_rec": ("rec", "receptions"), "wrte_rec_yds": ("yds", "receiving_yards")}   # ledger key, outcome column
MIN_EV, MAX_GAP = 0.05, 0.20
UNIT_TIERS = [(0.12, 2.0), (0.08, 1.5), (0.05, 1.0)]
MAX_PER_GAME = 3
SNAPSHOT_MAX_AGE = dt.timedelta(hours=48)
GRADE_AFTER = dt.timedelta(days=3)
RUNS = [(1, 13, 41), (2, 13, 41), (5, 13, 41)]       # scheduled weekly runs (Mon=0): Tue, Wed, Sat 13:41 UTC
RUN_SLACK = dt.timedelta(hours=1)                    # a run within the next hour is this run, not a later one
GRADUATE_WEEKS = 8


def next_run(now: dt.datetime) -> dt.datetime:
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(8):
        d = day + dt.timedelta(days=i)
        for wd, h, m in RUNS:
            t = d.replace(hour=h, minute=m)
            if d.weekday() == wd and t > now + RUN_SLACK:
                return t
    raise AssertionError("unreachable")


def units_for(ev: float) -> float:
    return next(u for lo, u in UNIT_TIERS if ev >= lo)


def chances(grid: list, line: float) -> tuple[float, float, float]:
    g = np.asarray(grid, dtype=float)
    over, under = float((g > line).mean()), float((g < line).mean())
    return over, under, 1.0 - over - under


def candidates(snapshot: dict, ledger_players: dict) -> list[dict]:
    """Every eligible side for one game snapshot, with EV at the best licensed price."""
    out = []
    for ln in snapshot["lines"]:
        mk = ln.get("market")
        lic = ln.get("licensed")
        if mk not in MARKETS or not lic or not ln.get("gsis_id"):
            continue
        p = ledger_players.get(ln["gsis_id"])
        if p is None or p["game_id"] != snapshot["game_id"]:
            continue
        key = MARKETS[mk][0]
        over, under, push = chances(p[key]["g"], ln["line"])
        if abs(over - ln["p_over_market"]) > MAX_GAP:
            continue
        sides = [("over", over, *lic["best_over"]), ("under", under, *lic["best_under"])]
        best = max(({"side": s, "p_win": pw, "price": pr, "book": bk, "ev": pw * pr + push - 1} for s, pw, pr, bk in sides),
                   key=lambda r: r["ev"])
        if best["ev"] < MIN_EV:
            continue
        out.append({"player": p["player"], "gsis_id": p["gsis_id"], "pos": p["pos"], "team": p["team"], "opp": p["opp"],
                    "market": mk, "line": ln["line"], **best, "book_name": LICENSED.get(best["book"], best["book"]),
                    "p_push": push, "p_model_over": over, "p_market_over": ln["p_over_market"],
                    "licensed_books": len(lic["books"]), "units": units_for(best["ev"])})
    # one pick per player per game, then the top MAX_PER_GAME by EV
    out.sort(key=lambda r: -r["ev"])
    seen, keep = set(), []
    for r in out:
        if r["gsis_id"] not in seen:
            seen.add(r["gsis_id"])
            keep.append(r)
    return keep[:MAX_PER_GAME]


def log(store: dict, now: dt.datetime, conn) -> int:
    lines = json.loads(LINES.read_text())["snapshots"] if LINES.exists() else []
    nxt = next_run(now)
    games = {r[0]: {"kickoff": r[1], "season": r[2], "week": r[3]} for r in conn.execute(
        "SELECT game_id, kickoff_utc, season, week FROM games WHERE game_type = 'REG' AND kickoff_utc > %s AND kickoff_utc <= %s",
        (now, nxt)).fetchall()}
    done = {p["game_id"] for p in store["picks"]} | set(store.get("games_logged", {}))
    added = 0
    for gid, g in sorted(games.items(), key=lambda kv: kv[1]["kickoff"]):
        if gid in done or (g["season"], g["week"]) < FIRST:
            continue
        wdir = LEDGER / f"{g['season']}_w{g['week']:02d}"
        if not (wdir / "wrte.json").exists():
            continue
        snaps = [s for s in lines if s["game_id"] == gid]
        if not snaps:
            continue
        snap = max(snaps, key=lambda s: s["snapshot_utc"])
        if now - dt.datetime.fromisoformat(snap["snapshot_utc"]) > SNAPSHOT_MAX_AGE:
            continue
        players = {p["gsis_id"]: p for p in json.loads((wdir / "wrte.json").read_text())["players"]}
        picks = candidates(snap, players)
        for r in picks:
            store["picks"].append({"id": f"{gid}|{r['gsis_id']}|{r['market']}", "season": g["season"], "week": g["week"],
                                   "game_id": gid, "kickoff": g["kickoff"].isoformat(), "logged_utc": now.isoformat(timespec="seconds"),
                                   "snapshot_utc": snap["snapshot_utc"], "state": STATE, **r,
                                   "result": None, "actual": None, "net": None, "graded_utc": None})
        # record that this game was considered, even with zero picks, so it is never revisited
        store.setdefault("games_logged", {})[gid] = {"logged_utc": now.isoformat(timespec="seconds"), "picks": len(picks),
                                                     "snapshot_utc": snap["snapshot_utc"]}
        added += len(picks)
    return added


def settle(pick: dict, actual: float | None) -> tuple[str, float]:
    if actual is None:
        return "void", 0.0
    if actual == pick["line"]:
        return "push", 0.0
    won = actual > pick["line"] if pick["side"] == "over" else actual < pick["line"]
    return ("win", pick["units"] * (pick["price"] - 1)) if won else ("loss", -pick["units"])


def grade(store: dict, now: dt.datetime, conn) -> int:
    from proplab.live.score import outcomes
    todo = [p for p in store["picks"] if p["result"] is None and now - dt.datetime.fromisoformat(p["kickoff"]) >= GRADE_AFTER]
    n = 0
    for sw in sorted({(p["season"], p["week"]) for p in todo}):
        out = outcomes(conn, *sw)
        finals = {r[0] for r in conn.execute("SELECT game_id FROM game_results").fetchall()}
        acts = {(r["gsis_id"], r["game_id"]): r for r in out.iter_rows(named=True)}
        for p in todo:
            if (p["season"], p["week"]) != sw or p["game_id"] not in finals:
                continue
            row = acts.get((p["gsis_id"], p["game_id"]))
            actual = float(row[MARKETS[p["market"]][1]]) if row else None
            p["result"], p["net"] = settle(p, actual)
            p["actual"], p["graded_utc"] = actual, now.isoformat(timespec="seconds")
            n += 1
    return n


def summarize(picks: list[dict], seed: int = 12) -> dict:
    g = [p for p in picks if p["result"] in ("win", "loss", "push")]
    res = {k: sum(p["result"] == k for p in picks) for k in ("win", "loss", "push", "void")}
    res["pending"] = sum(p["result"] is None for p in picks)
    staked = sum(p["units"] for p in g)
    net = sum(p["net"] for p in g)
    out = {**res, "units_staked": staked, "net_units": net, "roi": net / staked if staked else None,
           "weeks_graded": len({(p["season"], p["week"]) for p in g})}
    if len(g) >= 10:
        games = sorted({p["game_id"] for p in g})
        idx = {gid: i for i, gid in enumerate(games)}
        gi = np.array([idx[p["game_id"]] for p in g])
        s = np.bincount(gi, weights=[p["units"] for p in g], minlength=len(games))
        w = np.bincount(gi, weights=[p["net"] for p in g], minlength=len(games))
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(2000):
            c = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))
            if (s @ c) > 0:
                boots.append((w @ c) / (s @ c))
        out["roi_ci95"] = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
    out["graduated"] = bool(out["weeks_graded"] >= GRADUATE_WEEKS and out.get("roi_ci95", [0])[0] > 0)
    return out


def main(now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    conn = db.connect()
    store = json.loads(OUT.read_text()) if OUT.exists() else {"picks": [], "games_logged": {}}
    added, graded = log(store, now, conn), grade(store, now, conn)
    weeks = sorted({(p["season"], p["week"]) for p in store["picks"]})
    store["summary"] = {"overall": summarize(store["picks"]),
                        "by_week": [{"season": s, "week": w, **summarize([p for p in store["picks"] if (p["season"], p["week"]) == (s, w)])}
                                    for s, w in weeks]}
    store["rules"] = {"markets": list(MARKETS), "min_ev": MIN_EV, "max_gap": MAX_GAP, "unit_tiers": UNIT_TIERS,
                      "max_per_game": MAX_PER_GAME, "books": [LICENSED[b] for b in LICENSED if b in allowed()], "state": STATE,
                      "first_week": list(FIRST), "graduate_weeks": GRADUATE_WEEKS}
    store["updated_utc"] = now.isoformat(timespec="seconds")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(store, indent=1, default=str))
    o = store["summary"]["overall"]
    print(f"picks: {added} logged, {graded} graded | record {o['win']}-{o['loss']}-{o['push']} ({o['void']} void, {o['pending']} pending), "
          f"net {o['net_units']:+.2f}u on {o['units_staked']:.1f}u")
    return store


if __name__ == "__main__":
    main()
