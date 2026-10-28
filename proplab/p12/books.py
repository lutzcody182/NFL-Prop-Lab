"""State-licensed US sportsbooks (The Odds API bookmaker keys) whose prices count for tracked picks (Phase 12).

Offshore books (bovada, betonlineag, mybookieag, lowvig, betus), sweepstakes apps (fliff) and pick'em/DFS apps are
deliberately absent. To limit picks to one state's books, set STATE_BOOKS to a subset of these keys.
"""
LICENSED = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
    "betrivers": "BetRivers", "espnbet": "ESPN BET", "fanatics": "Fanatics", "hardrockbet": "Hard Rock Bet",
    "ballybet": "Bally Bet", "betparx": "betPARX",
}
STATE = None          # e.g. "NJ" once chosen
STATE_BOOKS = None    # e.g. {"draftkings", "fanduel", ...}; None = every licensed book above


def allowed() -> set[str]:
    return set(STATE_BOOKS) if STATE_BOOKS else set(LICENSED)
