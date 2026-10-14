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

Clarifications made before any 2025 score was computed:

1. **Live holdout is 2026 weeks 1–5**, not 1–6: week 6 kicks off 2026-10-15 and has no results yet. Week 6 gets projections only.
2. **Interval coverage uses the randomized PIT** (u = F(y−) + V·[F(y) − F(y−)], V ~ Uniform(0,1); covered if 0.1 ≤ u ≤ 0.9). Plain 10th–90th percentile intervals over-cover for counts like receptions purely from discreteness, which would make the 77–83% band meaningless.
3. **"Walk-forward" means:** every feature updates week by week from prior weeks only; fitted constants (position priors, team-volume regression, dispersion, per-catch yard shapes) are refit once per season from earlier seasons only. Hyperparameters are chosen on 2024 alone.
4. **Mean calibration added to all three models** after 2024 tuning showed teammate redistribution under-predicting established players' targets by about 2.6% (ECE 0.028). Each model's expected targets are multiplied by a per-position factor (actual ÷ predicted targets) fit on the three seasons before the one being predicted. Hyperparameters were then re-tuned on 2024. Still no 2025 score had been computed.
5. **Opponent adjustment removed from M1**, as the pre-registered ablation rule requires: on 2024 the tuned weight was 0 (no improvement in CRPS).
6. **Tried and reverted:** an in-season league catch-rate prior (blending earlier seasons with the current season's prior weeks). On 2024 it moved receptions ECE only from 0.025 to 0.024 and made the tuning objective slightly worse (0.9476 to 0.9481), so it was removed.
7. **Data fix during Phase 1 (Phase 0 bug):** 2016 SD and 2016–2019 OAK player stats and plays used relocated team codes and did not join to their games. Fixed at ingest, with two new quality checks; all tuning above was re-run on the corrected data.
8. **Tuning stopped here.** On 2024, M1 receptions ECE is 0.025, above the 0.02 gate. No further changes were made to chase that number; 2025 is scored once with this configuration.
