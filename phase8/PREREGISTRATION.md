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
