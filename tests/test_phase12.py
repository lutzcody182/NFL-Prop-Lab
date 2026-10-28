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
    assert len(c) == 1 and c[0]["side"] == "over" and c[0]["units"] == 2.0 and c[0]["book_name"] == "FanDuel"
    assert abs(c[0]["ev"] - (0.60 * 1.95 - 1)) < 1e-9
    # same edge but the model is 0.25 away from the market -> skipped (likely missing news)
    assert P.candidates(_snap(0.35, [1.95, "fanduel"], [1.87, "draftkings"]), _ledger(0.60)) == []
    # small edge -> skipped
    assert P.candidates(_snap(0.50, [1.80, "fanduel"], [1.80, "draftkings"]), _ledger(0.55)) == []
    # under side
    c = P.candidates(_snap(0.50, [1.87, "x"], [1.95, "betmgm"]), _ledger(0.40))
    assert c[0]["side"] == "under" and c[0]["units"] == 2.0


def test_units_and_settlement():
    assert P.units_for(0.05) == 1.0 and P.units_for(0.09) == 1.5 and P.units_for(0.2) == 2.0
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
