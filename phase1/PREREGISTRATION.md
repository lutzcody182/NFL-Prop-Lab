# Phase 1 pre-registration

Written 2026-10-14, **before** any model was scored on the 2025 test season.
Changes after that point must be listed under "Deviations" with a reason.

## Question

Can a usage-based model forecast WR/TE **receptions** and **receiving yards**
distributions better than simple baselines, out of sample?

## Population

- Regular-season games, 2017–2026 (2016 is history only, used to warm up features).
- Player-games: WR or TE, roster status `ACT` (dressed for the game) at the prediction cutoff.
- **Evaluated** player-games additionally need, from pre-game data only:
  at least 3 prior active games and a recent target share (half-life 8 games) of at least 5%.
  This approximates "players a sportsbook would post a line for" without looking at outcomes.
- Outcome = the player's receptions / receiving yards; 0 if active with no recorded stat.

## Prediction cutoff

Kickoff. Inputs allowed: anything with `known_at <= kickoff` (game-day actives and the
closing game spread/total). Prior-week games only, verified by a test that every
prior-week game is known before the earliest kickoff of the target week.
Phase 2 bet-time simulations must replace the closing game lines with bet-time lines.

## Splits

| Use | Seasons |
| --- | --- |
| Feature history and distribution fitting | everything before the season being predicted |
| Tuning and ablation | 2024 (walk-forward by week) |
| Locked test, scored once | 2025 |
| Live holdout, reported, never tuned on | 2026 weeks 1–6 |

## Models

- **B1 — Shrunk recent average.** Player's own exponentially weighted receptions/targets per game, shrunk to the position mean.
- **B2 — Usage × volume.** Shrunk target share × projected team targets (from team history, spread and total) × shrunk catch rate and yards per catch.
- **M1 — Candidate.** B2 plus (a) teammate-out redistribution of target share among the players actually active, and (b) opponent adjustment: the defense's shrunk catch-rate and yards-per-target allowed by position. Each component kept only if it improves 2024 CRPS (ablation).

All three share the same simulation: targets ~ negative binomial, catches ~ binomial(targets, catch rate),
yards = sum of per-catch yards drawn from the position's empirical distribution scaled to the player's
yards per catch. 2,000 draws per player-game.

## Metrics

- **Primary:** mean CRPS (lower is better), separately for receptions and yards.
- Secondary: MAE of the median, 80% interval coverage, Brier score and expected calibration error (ECE)
  of P(over) at fixed lines: receptions 2.5/3.5/4.5/5.5/6.5; yards 24.5/39.5/54.5/69.5/84.5.
- Uncertainty: paired bootstrap over games (1,000 resamples), 95% CI for the CRPS difference.

## Gate (both markets must pass, on 2025)

1. M1 mean CRPS lower than B2, with the 95% CI of (M1 − B2) entirely below 0.
2. M1 calibration: ECE ≤ 0.02 and 80% interval coverage between 77% and 83%.

If M1 fails, Phase 1 does not pass and the report says so. B2 is not swapped in as the "winner" after the fact.

## Deviations

(none yet)
