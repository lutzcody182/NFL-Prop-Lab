# Phase 3a pre-registration: the remaining prop markets

Written 2026-10-14, before any Phase 3a model was scored on 2025. Changes after the
first 2025 score must go under "Deviations" with a reason.

Phase 2 (market comparison) is on hold until historical odds are downloaded; nothing
here depends on it. Phase 3 is split: **3a** = the remaining markets (this file),
**3b** = team and player ratings, **3c** = matchup engine.

## Markets

| Market | Position | Fixed lines for Brier/ECE |
| --- | --- | --- |
| Passing yards | QB | 199.5, 224.5, 249.5, 274.5 |
| Passing attempts | QB | 29.5, 32.5, 35.5 |
| Passing TDs | QB | 0.5, 1.5, 2.5 |
| Interceptions | QB | 0.5 |
| QB rushing yards | QB | 9.5, 19.5, 29.5 |
| Rushing yards | RB | 39.5, 59.5, 79.5 |
| Rushing attempts | RB | 9.5, 14.5, 19.5 |
| Receptions | RB | 1.5, 2.5, 3.5 |
| Receiving yards | RB | 9.5, 19.5, 29.5 |
| Anytime touchdown | RB, WR, TE | yes/no |

WR/TE receptions and receiving yards are Phase 1 and stay locked: same model, same results.

## Population (pre-game information only)

Regular season; active (ACT) at kickoff; at least 3 prior active games; and a recent share
(half-life 8 games) of at least: QB 50% of team pass attempts; RB rushing markets 15% of team
carries; RB receiving markets 5% of team targets; anytime TD (RB/WR/TE) 15% of carries or 5% of targets.

## Simulation (shared)

Each player-game is simulated 4,000 times:

- **Volume:** targets, carries, pass attempts ~ negative binomial, mean = player share × projected team volume
  (B1: the player's own recent per-game average instead).
- **Successes:** catches ~ binomial(targets, catch rate); completions ~ binomial(attempts, completion rate); every carry counts.
- **Yards:** sum of per-unit yards drawn from that position's real per-catch / per-carry / per-completion
  yards in the three prior seasons, scaled to the player's shrunk yards per unit.
- **TDs and INTs:** binomial per unit (rushing TD per carry, receiving TD per catch, passing TD and INT per attempt).
- **Anytime TD:** any rushing or receiving TD in the same draw.
- Per-position mean calibration and dispersion fit on the three prior seasons, as in Phase 1.

## Models (per market)

- **B1:** player's own recent averages, shrunk to position.
- **B2:** share × team volume (team history, spread, total), shrunk efficiency.
- **M1:** B2 + renormalizing shares across the players actually active (redistribution).
- **M2:** M1 + opponent efficiency adjustment (what the defense allowed per unit to that position).

## Tuning and ablation (2024 only)

Per volume type (targets, carries, pass attempts), grid: half-life {4, 8, 16} games, share shrinkage {10, 40},
efficiency shrinkage {10, 30}, redistribution {0, 1}, opponent {0, 0.5, 1}. Objective: mean CRPS relative to B1
across that volume type's markets (Brier for anytime TD). **A component is kept only if removing it makes the
2024 objective worse.** The kept configuration is the "chosen model" for every market of that volume type.

## Test (2025, scored once) and gate per market

1. Chosen model better than B2: 95% bootstrap interval (over games) of the CRPS difference entirely below 0
   (Brier for anytime TD). If ablation removed both components, the chosen model *is* B2, and this criterion fails.
2. ECE of over/under probabilities at the fixed lines ≤ 0.02.
3. 80% interval coverage (randomized PIT) between 77% and 83% (not applied to anytime TD).

A market passes if all apply-able criteria pass. 2026 weeks 1–5 are reported as a live holdout.

## Deviations

Clarifications made before any 2025 score was computed:

1. **TD and INT rates use 5× the efficiency shrinkage** (50 or 150 pseudo-units instead of 10 or 30). A
   single TD moves a per-catch TD rate far more than one catch moves a catch rate, so the same
   shrinkage would let one-game flukes dominate. The opponent adjustment for TD rates likewise uses 5×
   the pseudo-count of the other defensive rates.
2. **QB rushing yards uses the QB pass-attempt eligibility** (≥ 50% of team pass attempts, i.e. starters);
   the pre-registered table names the market but not its population rule.
3. **QB population = the projected starter.** On 2021–2024, 10% of QB rows meeting the ≥ 50% share rule were
   benched former starters with 0–4 attempts (their recent share lingers after losing the job). The rule now
   also requires being the active QB with the highest recent share in that game, which is known before kickoff.
4. **Calibration and coverage criteria now scale with sample size.** At QB sample sizes (~500 games), noise
   alone produces an ECE around 0.03, so the fixed "ECE ≤ 0.02" would fail a perfectly calibrated model; Phase 1
   had ~13,000 line-pairs per market, where the noise floor was ~0.01. New criteria 2 and 3:
   - **2.** Observed ECE ≤ the 95th percentile of ECE when outcomes are simulated from the model's own
     probabilities (300 simulations), i.e. no detectable miscalibration at this sample size.
   - **3.** |coverage − 80%| ≤ max(3 points, 1.96·√(0.16/n)).
   Phase 1's gate is unchanged. These were decided after looking at 2024 diagnostics and before any 2025 score.
5. **QB volume = negative binomial + early-exit mixture.** Even projected starters leave about 5% of games early
   (injury, benching). A single negative binomial stretched to cover those had 80% intervals covering 87–89%
   on 2024. Now: with probability π the game is an early exit and keeps a uniform 0–35% of a full game's volume;
   π and the full-game dispersion are fit on the three prior seasons (a start below 35% of expectation counts as an
   exit). 2024 QB coverage became 78–84%. Applied to QBs only: for RBs it widened intervals (coverage up to 86%).
