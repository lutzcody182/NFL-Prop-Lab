# Phase 14 pre-registration: RB markets (ranges, efficiency uncertainty, chance of playing)

Written 2026-11-14, before any of these was built or scored. The base is the current live engine: the Phase 3a
locked configs plus the adopted Phase 9 role signal on carries and targets.

## Why

- Four RB markets beat their baseline on 2025 but fail calibration (and RB rushing yards and rush+rec yards also
  fail on ranges: 77% and 75% of results inside the 80% range). Those tests use the players who actually played, so
  this is not the actives problem from Phase 13.
- The engine fits one target-count and one carry-count spread per position for every player, whatever his volume.
  The same fix for WR/TE (Phase 11) was the largest WR/TE gain so far.
- The simulation treats a back's yards per carry or per catch as known exactly. Only the play-by-play outcome
  varies, so ranges for yardage markets come out too narrow.
- Live (Phase 13): when a projected teammate RB sits, the RBs who play are projected too low, because the missing
  player's share stays in the split. The injury report catches few of these cases.

## Groups 1–2: tested in the engine with the players who actually played (the setting of the pass/fail tests)

1. **Volume-dependent dispersion (D).** For RB rows only, the negative-binomial size for carries and for targets is
   fit separately in three bins of expected volume (cut at the 1/3 and 2/3 quantiles of the fitting rows), on the
   same earlier seasons as now. Nothing is tuned.
2. **Efficiency uncertainty (U).** For RB rows only, each simulated game multiplies the back's yards per carry and
   yards per catch by a lognormal factor with mean 1 and log-SD σ (one draw per player per simulated game, shared by
   all of that game's carries or catches). σ tried: 0.10, 0.20.

**Markets:** RB rush attempts, RB rushing yards, RB receptions, RB receiving yards, RB rush + rec yards.

**Rules (as Phases 9–11):**
- **Development:** walk-forward 2019–2024. A group survives if its better setting improves mean CRPS over the five
  markets with a 95% game-bootstrap interval below zero. Survivors are combined, and any whose removal no longer hurts
  is dropped.
- **Confirmation:** 2025 and 2026 weeks 1–9 against the base. Adopted if both changes are below zero and the pooled
  interval is below zero.
- **Pass/fail re-check:** if adopted, the Phase 8 recalibration and pass/fail check is rerun on 2025 for all engine
  markets with the new engine, under the Phase 8 rule unchanged. A market's pass/fail status changes only through
  that re-check.

## Group 3: chance of playing (A), tested in the live setting (method L: last game's actives)

- **P(play):** for a player who was active in his team's previous game, the chance he is active in the next one.
  It is estimated by position and by his offensive snap share in that previous game (bins: 0, 0–10%, 10–25%, 25–50%,
  50–75%, 75%+). It is fit on the three seasons before the target season, from consecutive team games.
- **Split:** when shares are scaled so the team's total is covered, each teammate's share counts × P(play)^w. A
  player's own projection still assumes he plays. w tried: 0.5, 1.0. Applied to carries and targets for all positions
  (the split is shared).
- **Development:** 2024 in the Phase 7 live-actives backtest (method L vs L + A). The pass rule for development is the
  same: mean CRPS change over the five RB markets with a 95% interval below zero.
- **Confirmation:** 2025 and 2026 weeks 1–9. Adopted if both are below zero, the pooled interval is below zero, and
  no other market (QB, WR/TE, anytime TD) gets worse by more than 0.5%.
- **If adopted:** live projections use it, starting with the next frozen week. It needs no injury information, so it
  works at the Tuesday freeze.

Adopted changes start a new model version in the ledger manifest. Frozen weeks are never re-scored.

## Deviations

1. The first chance-of-playing development run stopped on a code error (the backtest also needs the perfect-actives
   comparison) before it produced any result. It was fixed and re-run unchanged.

## Results and decision (added after scoring)

**A. Development, walk-forward 2019–2024** (`phase14/ablation.json`). Mean CRPS change over the five RB markets:

| Group | Change | 95% interval | Result |
| --- | --- | --- | --- |
| Volume-dependent dispersion (D) | −1.14% | −1.32% to −0.98% | survived (rush att −3.2%, rush yds −1.3%) |
| Efficiency uncertainty, σ = 0.10 | +0.02% | +0.01% to +0.03% | worse |
| Efficiency uncertainty, σ = 0.20 | +0.13% | +0.10% to +0.16% | worse |

**B. Confirmation of D** (vs the live engine): 2025 −0.70% (−1.07% to −0.32%), 2026 weeks 1–9 −0.62% (−1.07% to
−0.22%), pooled −0.67% (−0.95% to −0.38%). **Adopted.** 2025 calibration error went from 0.024 to 0.016 for RB rushing
yards, 0.026 to 0.019 for rush + rec yards, and 0.035 to 0.030 for rush attempts.

**C. Pass/fail re-check, Phase 8 rule unchanged** (`phase14/recal_engine.json`):

- **RB rushing yards now passes.** Raw version: −4.2% vs baseline, calibrated (0.018 vs limit 0.024), 80% ranges
  78.3%. Recalibration is not adopted (CRPS went up).
- **RB rush attempts now passes.** Recalibrated version: −8.3% vs baseline, calibration error 0.019 (limit 0.024),
  ranges 80.8%.
- **RB rush + rec yards still fails.** Its raw ranges are 75.5%. Recalibration fixes them but raises calibration error
  (0.018 → 0.022), so under the rule it is not adopted.
- **Status of the others is unchanged.** Recalibration is adopted for QB passing yards, QB passing TDs, RB rush
  attempts, anytime TD, receiving TD and kicker points.
- **5 of 16 markets now pass:** WR/TE receptions, WR/TE receiving yards, anytime TD, RB rushing yards and RB rush
  attempts.

**D. Chance of playing, development 2024** (`phase14/avail_dev.json`, live-actives backtest): w = 0.5 −0.07%, w = 1.0
−0.05% over the RB markets. The 95% intervals include zero. **Did not survive; not used.** Average participation
rates by snap share do not identify which backup sits.

Live projections use D and the refit maps from the week 11 freeze. Under the Phase 12 rule, RB rushing yards and
RB rush attempts join the tracked picks from week 11.
