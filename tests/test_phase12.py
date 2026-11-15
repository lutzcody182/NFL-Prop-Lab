"""Phase 12: licensed-book prices only, pick selection rules, timing and grading."""
import datetime as dt

import numpy as np

from proplab.p9 import lines as LN
from proplab.p12 import picks as P

UTC = dt.timezone.utc


def _ev(books):
    return {"bookmakers": [{"key": b, "markets": [{"key": "player_receptions", "outcomes": [
        {"name": "Over", "description": "A Receiver", "point": 4.5, "price": po},
        {"name": "Under", "description": "A Receiver", "point": 4.5, "price": pu}]}]} for b, po, pu in books]}


def test_best_prices_ignore_offshore_books():
    rows = LN.consensus(_ev([("draftkings", 1.87, 1.95), ("fanduel", 1.95, 1.87), ("bovada", 2.30, 1.70)]))
    lic = rows[0]["licensed"]
    assert lic["best_over"] == [1.95, "fanduel"] and lic["best_under"] == [1.95, "draftkings"]
    assert "bovada" not in lic["books"] and rows[0]["books"] == 3      # consensus itself still uses every book
    assert "licensed" not in LN.consensus(_ev([("bovada", 1.9, 1.9)]))[0]


def _snap(p_over_market, best_over, best_under, line=4.5):
    return {"game_id": "G", "lines": [{"market": "wrte_rec", "gsis_id": "X", "line": line, "p_over_market": p_over_market,
                                       "licensed": {"books": ["draftkings", "fanduel"], "best_over": best_over, "best_under": best_under}}]}


def _ledger(over_share):
    g = [6.0] * int(over_share * 100) + [3.0] * (100 - int(over_share * 100))
    return {"X": {"player": "A Receiver", "gsis_id": "X", "pos": "WR", "team": "T", "opp": "O", "game_id": "G", "rec": {"g": sorted(g)}}}


def test_selection_rules():
    # model 60% over, best over price 1.95 -> EV 17% -> 2 units
    c = P.candidates(_snap(0.50, [1.95, "fanduel"], [1.87, "draftkings"]), _ledger(0.60))
    assert len(c) == 1 and c[0]["side"] == "over" and c[0]["units"] == 2.0 and c[0]["book_name"] == "FanDuel"   # Kelly 4.6u, capped
    assert abs(c[0]["ev"] - (0.60 * 1.95 - 1)) < 1e-9
    # same edge but the model is 0.25 away from the market -> skipped (likely missing news)
    assert P.candidates(_snap(0.35, [1.95, "fanduel"], [1.87, "draftkings"]), _ledger(0.60)) == []
    # small edge -> skipped
    assert P.candidates(_snap(0.50, [1.80, "fanduel"], [1.80, "draftkings"]), _ledger(0.55)) == []
    # under side
    c = P.candidates(_snap(0.50, [1.87, "x"], [1.95, "betmgm"]), _ledger(0.40))
    assert c[0]["side"] == "under" and c[0]["units"] == 2.0


def test_units_and_settlement():
    # quarter Kelly, 1 unit = 1% of bankroll: 55% at -110 -> f = 5.5% -> 1.4u; long shot 5.4% at +3500 -> 0.7u
    assert P.kelly_units(0.55, 0.0, 1.909) == 1.4 and P.kelly_units(0.054, 0.0, 36.0) == 0.7 and P.kelly_units(0.7, 0, 1.9) == 2.0
    pk = {"line": 4.5, "side": "over", "units": 1.5, "price": 2.0}
    assert P.settle(pk, 5) == ("win", 1.5) and P.settle(pk, 4) == ("loss", -1.5) and P.settle(pk, None) == ("void", 0.0)
    assert P.settle({**pk, "line": 5.0}, 5) == ("push", 0.0)


def test_picks_logged_at_last_run_before_kickoff():
    wed = dt.datetime(2026, 11, 4, 13, 45, tzinfo=UTC)          # Wednesday run
    assert P.next_run(wed) == dt.datetime(2026, 11, 7, 13, 41, tzinfo=UTC)   # next is Saturday
    tue = dt.datetime(2026, 11, 3, 13, 45, tzinfo=UTC)
    assert P.next_run(tue) == dt.datetime(2026, 11, 4, 13, 41, tzinfo=UTC)


def test_summary_and_graduation():
    rng = np.random.default_rng(0)
    picks = [{"season": 2026, "week": 9 + i // 10, "game_id": f"g{i // 2}", "units": 1.0,
              "result": r, "net": 0.9 if r == "win" else -1.0} for i, r in enumerate(rng.choice(["win", "loss"], 40))]
    s = P.summarize(picks)
    assert s["win"] + s["loss"] == 40 and "roi_ci95" in s and s["graduated"] is False   # only 4 weeks


def test_blend_pulls_chances_toward_market():
    # model 60% over vs market 50%: log-odds blend 0.75 -> 57.6%; EV at 1.95 = 12.3% (model-only would be 17%)
    c = P.candidates(_snap(0.50, [1.95, "fanduel"], [1.87, "draftkings"]), _ledger(0.60), blend=0.75)
    pb = P.blend_chance(0.60, 0.50, 0.75)
    assert abs(pb - 0.5757) < 1e-3 and c[0]["side"] == "over" and abs(c[0]["p_win"] - pb) < 1e-9 and abs(c[0]["p_model_win"] - 0.60) < 1e-9
    assert abs(c[0]["ev"] - (pb * 1.95 - 1)) < 1e-9
    # a 5-point model edge at 1.95 is a pick model-only (EV 7.25%) but not after a 50/50 blend (EV ~4.9%)
    assert len(P.candidates(_snap(0.50, [1.95, "fanduel"], [1.87, "draftkings"]), _ledger(0.55))) == 1
    assert P.candidates(_snap(0.50, [1.95, "fanduel"], [1.87, "draftkings"]), _ledger(0.55), blend=0.5) == []


def test_anytime_td_yes_only_and_yes_no():
    ev = {"bookmakers": [
        {"key": "draftkings", "markets": [{"key": "player_anytime_td", "outcomes": [{"name": "Yes", "description": "A Back", "price": 2.6}]}]},
        {"key": "fanduel", "markets": [{"key": "player_anytime_td", "outcomes": [{"name": "Yes", "description": "A Back", "price": 2.8}]}]}]}
    r = LN.consensus(ev)[0]
    assert r["line"] == 0.5 and r.get("vig_included") and abs(r["p_over_market"] - (1 / 2.6 + 1 / 2.8) / 2) < 1e-9
    assert r["licensed"]["best_over"] == [2.8, "fanduel"] and r["licensed"]["best_under"] is None
    ev["bookmakers"][0]["markets"][0]["outcomes"].append({"name": "No", "description": "A Back", "price": 1.45})
    r = LN.consensus(ev)[0]
    assert not r.get("vig_included") and r["books"] == 1 and r["licensed"]["best_under"] == [1.45, "draftkings"]
    # a 45% model chance at +180 (2.8) is EV 26% -> Kelly 3.6u, capped at 2, Yes side
    snap = {"game_id": "G", "lines": [{"market": "anytime_td", "gsis_id": "X", "line": 0.5, "p_over_market": 0.37,
                                      "licensed": {"books": ["fanduel"], "best_over": [2.8, "fanduel"], "best_under": None}}]}
    c = P.candidates(snap, {("td", "X"): {"player": "A Back", "gsis_id": "X", "pos": "RB", "team": "T", "opp": "O", "game_id": "G", "mean": 0.45}})
    assert len(c) == 1 and c[0]["side"] == "over" and c[0]["units"] == 2.0 and abs(c[0]["ev"] - (0.45 * 2.8 - 1)) < 1e-9


def test_rb_engine_markets_use_markets_grid():
    snap = {"game_id": "G", "lines": [{"market": "rb_rush_yds", "gsis_id": "R", "line": 60.5, "p_over_market": 0.5,
                                      "licensed": {"books": ["fanduel"], "best_over": [1.91, "fanduel"], "best_under": [1.91, "draftkings"]}}]}
    g = sorted([80.0] * 62 + [40.0] * 38)
    pl_ = {("rb_rush_yds", "R"): {"player": "A Back", "gsis_id": "R", "pos": "RB", "team": "T", "opp": "O", "game_id": "G", "g": g}}
    c = P.candidates(snap, pl_)
    assert len(c) == 1 and c[0]["side"] == "over" and abs(c[0]["p_win"] - 0.62) < 1e-9
    assert P.MARKET_FIRST["rb_rush_yds"] == (2026, 11)


def test_fair_td_power_devig():
    # three players priced 50%, 30%, 5% (sum 0.85) but only 0.6 scorers expected -> each shrinks, long shot the most
    lines = [{"market": "anytime_td", "vig_included": True, "gsis_id": g, "p_over_market": q} for g, q in (("a", .5), ("b", .3), ("c", .05))]
    import proplab.p12.picks as PP
    old = PP.TD_SCORERS; PP.TD_SCORERS = (0.6, 0.0)
    try:
        f = PP.fair_td(lines, 44.5)
    finally:
        PP.TD_SCORERS = old
    assert abs(sum(f.values()) - 0.6) < 1e-6
    assert f["c"] / 0.05 < f["b"] / 0.3 < f["a"] / 0.5 < 1
