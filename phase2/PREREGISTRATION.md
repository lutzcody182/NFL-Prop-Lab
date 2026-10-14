# Phase 2 pre-registration: model vs sportsbook market

Written 2026-10-14, **before any odds data was downloaded**. Changes after the
first odds file arrives must be listed under "Deviations" with a reason.

## Question

How do the locked Phase 1 M1 probabilities compare with sportsbook prices for WR/TE
receptions and receiving yards, and does the model add any information the market lacks?

Expectation stated up front: closing prop prices are usually hard to beat, so the
most likely result is that the market is better calibrated than the model.

## Data

- Source: The Odds API, historical event odds, region `us`.
- Markets: `player_receptions`, `player_reception_yds` (main lines only, no alternates).
- Games: regular season 2023, 2024, 2025, and 2026 weeks 1–5. Later 2026 weeks are added as they finish.
- Two snapshots per game (the API returns the latest snapshot at or before the requested time; snapshots are 5 minutes apart):
  - **bet** = kickoff − 36 hours: both prop markets plus `spreads` and `totals`
  - **close** = kickoff − 5 minutes: both prop markets
- Raw responses are stored unchanged in `data/odds/raw/` and are never edited.

## Matching

- Odds events map to nflverse games by home team, away team, and kickoff within 12 hours.
- Player names map to gsis_id **only among players on one of the game's two rosters that week**,
  after normalizing (lowercase, strip punctuation, drop Jr/Sr/II/III/IV). Ambiguous or unmatched
  names are listed in `data/odds/unmatched.csv` and excluded, not guessed.
- Phase 2 counts as complete only if at least 90% of close-snapshot WR/TE prop rows are matched.

## Market probability

For one player-game-snapshot:

1. Each bookmaker's over/under pair gives a no-vig over probability (multiplicative method:
   p_over / (p_over + p_under)). Power and Shin methods are computed and reported as alternatives.
2. The **consensus line** is the most common line across books (ties: the lower line).
3. The market probability is the mean no-vig over probability of the books posting the consensus line.

Integer lines: a push (stat equals the line) is excluded from scoring; model probability is
P(over) / (1 − P(push)).

## Model probability

- **At close:** M1 exactly as locked in Phase 1 (`phase1/tuning.json`), 4,000 draws, actives from
  game-day rosters, closing game spread/total. Same information time as the closing market.
- **At bet time (kickoff − 36h):** M1 with projected actives (each team's actives in its previous
  game, as in `proplab/p1/project.py`) and the consensus spread/total from the **bet** snapshot.
  No closing game lines, no game-day actives, no injury report released after the cutoff.
- No refitting of Phase 1 parameters with odds data.
- Population: matched WR/TE props where M1 has a prediction (no Phase 1 eligibility filter;
  the market already chose which players to post). Players with RB position are excluded and counted.

## Metrics

Primary, per market, on 2023–2025 pooled and on each season:

- **Log loss** of M1 and of the no-vig close market on the over/under outcome, and the paired
  difference (M1 − market) with a 95% bootstrap interval over games (1,000 resamples).

Secondary:

- Brier score and ECE for both; reliability curves.
- **Blend test:** logit(p) = a + b·logit(market) + c·logit(M1), fit on 2023–2024, scored on 2025.
  Report c with a 95% bootstrap interval and the 2025 log loss of the blend vs the market.
  c > 0 with an interval clear of zero means the model carries information the closing market lacks.
- **Closing-line value (CLV):** for props where |M1_bet − market_bet| ≥ 5 percentage points:
  - same line at bet and close: mean change in the market's no-vig probability toward the model's side;
  - line moved: share of line moves in the model's direction (model says over → line moved up).
  Both with 95% bootstrap intervals over games.
- Hypothetical flat-stake return at the best bet-time price on those props, reported as secondary
  and labelled as noisy. No betting recommendation is made from it.

2026 weeks are reported separately as a live holdout.

## What Phase 2 unlocks

- EV numbers appear on the dashboard **only** for a market where, on 2025, the blend coefficient c
  has a 95% interval above zero. Otherwise the dashboard shows the market comparison and says the
  model did not add information beyond the market.

## Deviations

(none yet)
