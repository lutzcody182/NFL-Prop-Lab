"""Phase 12: tracked picks at US-licensed books (phase12/PREREGISTRATION.md). Not betting advice.

    python -m proplab.p12.picks     # in the weekly run: log picks for games with no later run before kickoff, grade old ones

Picks are written once per game and never changed; grading only fills in the result fields.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from proplab import db
from proplab.p12.books import LICENSED, STATE, allowed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase12" / "picks.json"
LINES = ROOT / "phase9" / "market_lines.json"
LEDGER = ROOT / "ledger"
FIRST = (2026, 9)
# ledger key, outcome column. anytime_td (added 2026-11-11, passed every test after Phase 8 recalibration) is a Yes/No
# market stored as line 0.5: Yes = over. Its chance is the frozen projection's recalibrated probability.
MARKETS = {"wrte_rec": ("rec", "receptions"), "wrte_rec_yds": ("yds", "receiving_yards"), "anytime_td": ("td", "anytime_td"),
           # passed in the Phase 14 re-check (2026-11-14); engine markets read their grid from markets.json
           "rb_rush_yds": ("eng", "rushing_yards"), "rb_rush_att": ("eng", "carries")}
MARKET_FIRST = {"rb_rush_yds": (2026, 11), "rb_rush_att": (2026, 11)}   # first week frozen with the engine that passed
MIN_EV, MAX_GAP = 0.05, 0.20
# EV method (changed 2026-11-14, deviation 3): EV = blended chance x decimal price + push chance - 1, where the blended
# chance mixes the model with the market's fair chance on the log-odds scale; stake = quarter Kelly, 1 unit = 1% of
# bankroll, capped at MAX_UNITS. Version 1 weights the model 75%, the blended test version 50%.
MAIN_WEIGHT = 0.75
KELLY_FRACTION, MAX_UNITS = 0.25, 2.0
TD_SCORERS = (-0.492, 0.1012)    # distinct rush/rec TD scorers per game = a + b x over/under (2021-2025, 1,359 games)

UNIT_TIERS = [(0.12, 2.0), (0.08, 1.5), (0.05, 1.0)]
MAX_PER_GAME = 3
SNAPSHOT_MAX_AGE = dt.timedelta(hours=48)
STATUS_WAIT = dt.timedelta(days=3)       # only for players with no stat line (sat out, or no catches)
RUNS = [(1, 13, 41), (2, 13, 41), (5, 13, 41)]       # pick-logging runs (Mon=0): Tue, Wed, Sat 13:41 UTC
# (the Fri/Mon/Tue early-morning runs grade results; they are not counted here, so picks are still logged
#  Wednesday for Thursday games and Saturday for Sunday/Monday games, as pre-registered)
RUN_SLACK = dt.timedelta(hours=1)                    # a run within the next hour is this run, not a later one
GRADUATE_WEEKS = 8
# Phase 12 v2 (test version, tracked alongside v1 from week 10): same rules with a more market-heavy blend.
# Pre-registered 2026-11-11 as a 75/25 linear blend; from 2026-11-14 (deviation 3) a 50/50 log-odds blend, because v1
# itself now uses the 75/25 blend.
BLEND_WEIGHT = 0.50
BLEND_FIRST = (2026, 10)


def next_run(now: dt.datetime) -> dt.datetime:
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(8):
        d = day + dt.timedelta(days=i)
        for wd, h, m in RUNS:
            t = d.replace(hour=h, minute=m)
            if d.weekday() == wd and t > now + RUN_SLACK:
                return t
    raise AssertionError("unreachable")


def logit_gap(a: float, b: float) -> float:
    lg = lambda x: math.log(min(max(x, 1e-4), 1 - 1e-4) / (1 - min(max(x, 1e-4), 1 - 1e-4)))
    return abs(lg(a) - lg(b))


def units_for(ev: float) -> float:
    return next(u for lo, u in UNIT_TIERS if ev >= lo)


def chances(grid: list, line: float) -> tuple[float, float, float]:
    g = np.asarray(grid, dtype=float)
    over, under = float((g > line).mean()), float((g < line).mean())
    return over, under, 1.0 - over - under


def fair_td(lines: list[dict], total: float | None) -> dict:
    """gsis_id -> fair anytime-TD chance for one game's Yes-only lines (book's cut included in the consensus price).

    Power de-vig across the game's scorer list: find k >= 1 so that the sum of q_i^k equals the expected number of
    distinct rush/receiving TD scorers for the game total (fit on 2021-2025: TD_SCORERS[0] + TD_SCORERS[1] x total).
    Long shots lose proportionally more of their implied chance, as with real sportsbook margins."""
    td = [(ln["gsis_id"], ln["p_over_market"]) for ln in lines
          if ln.get("market") == "anytime_td" and ln.get("vig_included") and ln.get("gsis_id")]
    if not td or total is None:
        return {}
    target = TD_SCORERS[0] + TD_SCORERS[1] * total
    q = np.clip(np.array([x for _, x in td]), 1e-4, 0.999)
    if q.sum() <= target:
        return {g: float(x) for (g, _), x in zip(td, q)}
    lo, hi = 1.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (q ** mid).sum() > target else (lo, mid)
    return {g: float(x) for (g, _), x in zip(td, q ** hi)}


def blend_chance(model: float, market: float, w: float) -> float:
    """w x model + (1 - w) x market, on the log-odds scale."""
    lg = lambda x: math.log(min(max(x, 1e-4), 1 - 1e-4) / (1 - min(max(x, 1e-4), 1 - 1e-4)))
    z = w * lg(model) + (1 - w) * lg(market)
    return 1 / (1 + math.exp(-z))


def kelly_units(p_win: float, push: float, price: float) -> float:
    """Quarter-Kelly stake in units (1 unit = 1% of bankroll), rounded to 0.1, between 0.1 and MAX_UNITS."""
    b = price - 1
    f = (p_win * b - (1 - p_win - push)) / b
    return float(min(MAX_UNITS, max(0.1, round(KELLY_FRACTION * f * 100, 1))))


def candidates(snapshot: dict, ledger_players: dict, blend: float | None = None, total: float | None = None,
               select: bool = True) -> list[dict]:
    """Every eligible side for one game snapshot, with EV at the best licensed price.

    blend: weight on the model when mixing its chance with the market's fair chance on the log-odds scale
    (None = model only). total: the game's over/under, used to remove the book's cut from Yes-only TD prices.
    select=False returns every prop's best side (no EV bar, no per-game limit), for recomputing logged picks."""
    fair = fair_td(snapshot["lines"], total)
    out = []
    for ln in snapshot["lines"]:
        mk = ln.get("market")
        lic = ln.get("licensed")
        if mk not in MARKETS or not lic or not ln.get("gsis_id"):
            continue
        key = MARKETS[mk][0]
        p = (ledger_players.get((key, ln["gsis_id"])) if key == "td" else ledger_players.get((mk, ln["gsis_id"])) if key == "eng"
             else ledger_players.get(ln["gsis_id"]))
        if p is None or p["game_id"] != snapshot["game_id"]:
            continue
        if key == "td":
            over = float(p["mean"]); under, push = 1.0 - over, 0.0
        elif key == "eng":
            over, under, push = chances(p["g"], ln["line"])
        else:
            over, under, push = chances(p[key]["g"], ln["line"])
        pk = fair.get(ln["gsis_id"], ln["p_over_market"]) if mk == "anytime_td" else ln["p_over_market"]
        if select and abs(over - pk) > MAX_GAP:
            continue
        m_over, m_under = over, under
        if blend is not None:   # pushes stay as the model has them; the rest is blended with the market's fair chance
            rest = 1 - push
            over = blend_chance(m_over / rest, pk, blend) * rest if rest > 0 else 0.0
            under = rest - over
        sides = [("over", over, *lic["best_over"])] + ([("under", under, *lic["best_under"])] if lic.get("best_under") else [])
        best = max(({"side": s, "p_win": pw, "price": pr, "book": bk, "ev": pw * pr + push - 1} for s, pw, pr, bk in sides),
                   key=lambda r: r["ev"])
        if select and best["ev"] < MIN_EV:
            continue
        out.append({"player": p["player"], "gsis_id": p["gsis_id"], "pos": p["pos"], "team": p["team"], "opp": p["opp"],
                    "market": mk, "line": ln["line"], **best, "book_name": LICENSED.get(best["book"], best["book"]),
                    "p_push": push, "p_model_over": m_over, "p_market_over": ln["p_over_market"], "p_market_fair_over": pk,
                    "p_model_win": m_over if best["side"] == "over" else m_under,
                    "licensed_books": len(lic["books"]), "units": kelly_units(best["p_win"], push, best["price"])})
    if not select:
        return out
    # one pick per player per game, then the top MAX_PER_GAME by EV
    out.sort(key=lambda r: -r["ev"])
    seen, keep = set(), []
    for r in out:
        if r["gsis_id"] not in seen:
            seen.add(r["gsis_id"])
            keep.append(r)
    return keep[:MAX_PER_GAME]


def game_total(players: dict, gid: str) -> float | None:
    """The game's over/under as frozen with the week's WR/TE projections (stored per player as drivers.total)."""
    for k, p in players.items():
        if isinstance(k, str) and p.get("game_id") == gid and (p.get("drivers") or {}).get("total") is not None:
            return float(p["drivers"]["total"])
    return None


def log(store: dict, now: dt.datetime, conn, blend: float | None = MAIN_WEIGHT, first: tuple = FIRST) -> int:
    lines = json.loads(LINES.read_text())["snapshots"] if LINES.exists() else []
    nxt = next_run(now)
    games = {r[0]: {"kickoff": r[1], "season": r[2], "week": r[3]} for r in conn.execute(
        "SELECT game_id, kickoff_utc, season, week FROM games WHERE game_type = 'REG' AND kickoff_utc > %s AND kickoff_utc <= %s",
        (now, nxt)).fetchall()}
    done = {p["game_id"] for p in store["picks"]} | set(store.get("games_logged", {}))
    added = 0
    for gid, g in sorted(games.items(), key=lambda kv: kv[1]["kickoff"]):
        if gid in done or (g["season"], g["week"]) < first:
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
        if (wdir / "markets.json").exists():     # anytime TD chances, keyed ("td", gsis_id)
            mj = json.loads((wdir / "markets.json").read_text())["markets"]
            for q in mj.get("anytime_td", {}).get("players", []):
                players[("td", q["gsis_id"])] = q
            for mk in (m for m, (k_, _) in MARKETS.items() if k_ == "eng"):   # RB engine markets, keyed (market, gsis_id)
                for q in mj.get(mk, {}).get("players", []):
                    players[(mk, q["gsis_id"])] = q
        wk = (g["season"], g["week"])
        snap = {**snap, "lines": [ln for ln in snap["lines"] if wk >= MARKET_FIRST.get(ln.get("market"), (0, 0))]}
        picks = candidates(snap, players, blend, total=game_total(players, gid))
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
    """Settle like a sportsbook: as soon as the game is final and the player's box-score line is loaded.

    A player with no stat line either sat out (void) or played without recording a stat (e.g. 0 catches = a loss
    for an over). Game-day inactives reach the roster data only after games, so those picks wait for
    STATUS_WAIT and are then settled from the final active/inactive list.
    """
    from proplab.live.score import outcomes
    todo = [p for p in store["picks"] if p["result"] is None and not p.get("excluded")]
    n = 0
    for sw in sorted({(p["season"], p["week"]) for p in todo}):
        finals = {r[0] for r in conn.execute("SELECT game_id FROM game_results").fetchall()}
        loaded = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM player_game WHERE game_id LIKE %s",
                                             (f"{sw[0]}_{sw[1]:02d}_%",)).fetchall()}
        cols = {mk: col for mk, (_, col) in MARKETS.items()}
        stat = {(r[0], r[1]): {"receptions": float(r[2]), "receiving_yards": float(r[3]), "anytime_td": float(r[4] > 0),
                               "rushing_yards": float(r[5]), "carries": float(r[6])}
                for r in conn.execute(
            """SELECT gsis_id, game_id, coalesce(receptions, 0), coalesce(receiving_yards, 0),
                      coalesce(rushing_tds, 0) + coalesce(receiving_tds, 0), coalesce(rushing_yards, 0), coalesce(carries, 0)
               FROM player_game
               WHERE game_id LIKE %s""", (f"{sw[0]}_{sw[1]:02d}_%",)).fetchall()}
        acts = None
        for p in todo:
            if (p["season"], p["week"]) != sw or p["game_id"] not in finals or p["game_id"] not in loaded:
                continue
            key = (p["gsis_id"], p["game_id"])
            if key in stat:
                actual = stat[key][cols[p["market"]]]
            elif now - dt.datetime.fromisoformat(p["kickoff"]) >= STATUS_WAIT:
                if acts is None:
                    acts = {(r["gsis_id"], r["game_id"]) for r in outcomes(conn, *sw).iter_rows(named=True)}
                actual = 0.0 if key in acts else None      # active with no stat line = 0; not active = void
            else:
                continue                                    # wait for the final active/inactive list
            p["result"], p["net"] = settle(p, actual)
            p["actual"], p["graded_utc"] = actual, now.isoformat(timespec="seconds")
            n += 1
    return n


def summarize(picks: list[dict], seed: int = 12) -> dict:
    picks = [p for p in picks if not p.get("excluded")]     # removed before kickoff (deviation 2): not in the record
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
    bl = store.setdefault("blend", {"picks": [], "games_logged": {}})
    b_added, b_graded = log(bl, now, conn, blend=BLEND_WEIGHT, first=BLEND_FIRST), grade(bl, now, conn)
    for st in (store, bl):
        weeks = sorted({(p["season"], p["week"]) for p in st["picks"]})
        st["summary"] = {"overall": summarize(st["picks"]),
                         "by_week": [{"season": s, "week": w, **summarize([p for p in st["picks"] if (p["season"], p["week"]) == (s, w)])}
                                     for s, w in weeks]}
    bl["rules"] = {"model_weight": BLEND_WEIGHT, "first_week": list(BLEND_FIRST)}
    store["rules"] = {"markets": list(MARKETS), "min_ev": MIN_EV, "max_gap": MAX_GAP, "model_weight": MAIN_WEIGHT,
                      "kelly_fraction": KELLY_FRACTION, "max_units": MAX_UNITS,
                      "max_per_game": MAX_PER_GAME, "books": [LICENSED[b] for b in LICENSED if b in allowed()], "state": STATE,
                      "first_week": list(FIRST), "graduate_weeks": GRADUATE_WEEKS}
    store["updated_utc"] = now.isoformat(timespec="seconds")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(store, indent=1, default=str))
    o = store["summary"]["overall"]
    print(f"picks: {added} logged, {graded} graded | record {o['win']}-{o['loss']}-{o['push']} ({o['void']} void, {o['pending']} pending), "
          f"net {o['net_units']:+.2f}u on {o['units_staked']:.1f}u")
    b = bl["summary"]["overall"]
    print(f"blended picks (v2 shadow): {b_added} logged, {b_graded} graded | record {b['win']}-{b['loss']}-{b['push']}, "
          f"net {b['net_units']:+.2f}u on {b['units_staked']:.1f}u")
    return store


if __name__ == "__main__":
    main()
