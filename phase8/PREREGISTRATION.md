# Phase 8 pre-registration: recalibrating ranges and over/under chances

Written 2026-10-16, before any recalibration was fit or scored.

## Why

Most markets fail on calibration or range coverage, not on the average projection: ranges that hold 75% or 85% of
outcomes instead of 80%, or over/under chances that run slightly hot or cold. A recalibration step learns that
pattern from past seasons and corrects it.

## Method (per market, separately for the model and for its baseline)

- **Yards and counts:** quantile recalibration. Collect the PIT (where the actual result fell in its simulated
  distribution, randomized for whole numbers) for every player-game in the fitting seasons. The recalibrated
  distribution takes the original simulation's quantile at level Q(τ), where Q is the quantile function of those PITs.
  A perfectly calibrated model has Q(τ) = τ and is unchanged.
- **Yes/no markets (anytime TD, receiving TD):** logistic recalibration, p' = sigmoid(a + b·logit p).
- **Fitting seasons:** for the 2025 test, maps are fit on model output for 2022–2024 only. For live 2026 projections
  (and for the 2026 weeks 1–5 check), maps are fit on 2023–2025. The model itself is not changed or re-tuned.
- **Scope:** the 10 Phase 3a markets, the 3 Phase 7 engine markets and kicker points (14). The two Phase 1 WR/TE markets
  are out of scope: receptions failed only on "beats baseline", which recalibration cannot fix, and yards passed.

## Rules

- Each market is re-scored once on 2025 with the same pass rules as before (beats its baseline with a 95% interval below
  zero, calibrated, ranges within tolerance), using the **recalibrated model against the recalibrated baseline**, so a
  pass cannot come from calibrating only one side.
- **Adoption, per market:** published projections use the recalibrated version if, on 2025, it lowers the calibration
  error (ECE) and does not raise CRPS (or Brier). Otherwise the raw version stays. The dashboard's pass/fail then reflects
  whichever version is published.
- 2026 weeks 1–5 are reported as a second check with the live maps. Adopted changes start a new model version in the
  ledger manifest from the next frozen week; frozen weeks are never re-scored.

## Deviations

(none yet)

## Results and decision (added after scoring)

2025, recalibrated model vs recalibrated baseline (`phase8/results.json`). Adopted = 2025 calibration error went down
and CRPS (Brier for yes/no) did not go up.

| Market | Calibration error raw → recal (noise limit) | 80% ranges raw → recal | Passes after? | Adopted |
| --- | --- | --- | --- | --- |
| QB passing yards | 0.028 → 0.025 (0.033) | 82% → 80% | no (baseline kept, so cannot beat it) | yes |
| QB pass attempts | 0.050 → 0.058 (0.038) | 79% → 79% | no | no |
| QB passing TDs | 0.036 → 0.033 (0.039) | 82% → 79% | no (baseline kept) | yes |
| QB interceptions | 0.023 → 0.025 (0.059) | 82% → 82% | no (baseline kept) | no |
| QB rushing yards | 0.030 → 0.024 (0.037) | 79% → 83% | no (worse than baseline) | no (CRPS up) |
| RB rushing yards | 0.025 → 0.036 (0.025) | 77% → 79% | no | no |
| RB rushing attempts | 0.038 → 0.039 (0.022) | 80% → 79% | no | no |
| RB receptions | 0.027 → 0.024 (0.029) | 82% → 82% | no (gain could be chance) | no (CRPS up) |
| RB receiving yards | 0.022 → 0.019 (0.031) | 81% → 81% | no (gain could be chance) | no (CRPS up) |
| **Anytime TD** | 0.023 → 0.016 (0.020) | — | **yes** (−0.46% vs baseline) | **yes** |
| RB rushing + receiving yards | 0.028 → 0.038 (0.025) | 75% → 78% | no | no |
| QB completions | 0.031 → 0.038 (0.038) | 81% → 81% | no | no |
| Receiving TD | 0.015 → 0.011 (0.017) | — | no (gain could be chance) | yes |
| Kicker points | 0.042 → 0.041 (0.037) | 84% → 80% | no (chances still slightly off) | yes |

- **Anytime TD now passes every rule.** Together with WR/TE receiving yards, 2 of 16 markets pass.
- Quantile recalibration fixes range coverage almost everywhere, but for count markets with few distinct values (carries,
  attempts, completions) it can make the over/under chances at the fixed lines worse, so it is not adopted there.
- Live projections use the 2023–2025 maps for the five adopted markets from the next frozen week (week 7 of 2026);
  2026 weeks 1–5 results with those maps are in `phase8/results.json` (`live_2026_wk1_5`).
