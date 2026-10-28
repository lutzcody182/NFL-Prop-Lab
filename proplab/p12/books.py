"""State-licensed US sportsbooks (The Odds API bookmaker keys) whose prices count for tracked picks (Phase 12).

Offshore books (bovada, betonlineag, mybookieag, lowvig, betus), sweepstakes apps (fliff) and pick'em/DFS apps are
deliberately absent. To limit picks to one state's books, set STATE_BOOKS to a subset of these keys.
"""
LICENSED = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
    "betrivers": "BetRivers", "espnbet": "theScore Bet", "fanatics": "Fanatics", "hardrockbet": "Hard Rock Bet",
    "ballybet": "Bally Bet", "betparx": "betPARX",
}
# Pennsylvania (PGCB-licensed online books available in our odds feed; checked 2026-10-28).
# Hard Rock Bet and Bally Bet are not licensed in PA. bet365, PlaySugarHouse and Mohegan are licensed but not in the feed.
STATE = "PA"
STATE_BOOKS = {"draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "betparx"}


def allowed() -> set[str]:
    return set(STATE_BOOKS) if STATE_BOOKS else set(LICENSED)
