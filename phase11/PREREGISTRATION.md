# Phase 11 pre-registration: sharper ranges and smarter redistribution (free data only)

Written 2026-10-28, before any of these was built or scored. No sportsbook data is used.

## Why

- In the first live scored week (2026 week 6), WR/TE receptions landed inside the 80% range 90% of the time and yards 86%:
  ranges may be too wide. One week is too small to act on by itself, so this is tested on history.
- The WR/TE model uses one target-count spread shape per position for every player, whatever his volume.
- When a regular misses a game, his share is spread over all active teammates in proportion to their shares, so a missing
  WR1 boosts the TE and RBs as much as the other WRs.
- The Phase 8 recalibration maps were fit before the Phase 9 role signal was added to the engine markets, and the WR/TE
  markets were never recalibrated.

## Groups

All features use earlier games only. Tested inside the live WR/TE model (Phase 1 M1 + Phase 10 role and air yards).

1. **Volume-dependent dispersion (D).** The negative-binomial size parameter for targets is fit separately for each
   position in three bins of the model's expected targets (cut at the 1/3 and 2/3 quantiles of the fitting rows), on the
   same earlier seasons as now. Nothing is tuned.
2. **Position-aware redistribution (P).** Each team's recent share of targets by position group (WR, TE, RB; half-life
   8 games, prior games only) is compared with the summed shares of that group's active players. Before the existing
   team-wide rescaling, each player's share × (team's recent group share / active group share)^w, so a missing WR's
   targets go mainly to the other WRs. w tried: 0.5, 1.0.

**Rules for groups 1–2 (same as Phases 9 and 10).** Development is walk-forward 2019–2024 on both WR/TE markets. A group
survives if its better setting improves mean CRPS with a 95% game-bootstrap interval below zero. Survivors are combined,
and any whose removal no longer hurts is dropped. Confirmation compares the survivors with the current live model on 2025
and on 2026 weeks 1–7. They are adopted if both changes are below zero and the pooled interval is below zero.

3. **Recalibration (R).** This is the Phase 8 method and rule, unchanged.
   - **WR/TE receptions and yards.** Quantile maps are fit on the final WR/TE model (after groups 1–2 are decided) for
     2022–2024 and scored on 2025. A map is adopted if the 2025 calibration error goes down and CRPS does not go up. Live
     maps are fit on 2023–2025. The WR/TE baseline is recalibrated the same way, so the pass/fail check stays fair.
   - **Engine markets.** All 14 Phase 8 markets are refit on the current engine (with the adopted Phase 9 role signal).
     Each market's adoption is re-decided by the same rule.

Adopted changes start a new model version in the ledger manifest from the next frozen week. Frozen weeks are never
re-scored.

## Deviations

1. The first development run stopped on a code error (a read-only array) before it produced any result. It was fixed
   and re-run unchanged.

## Results and decision (added after scoring)

**A. Development, walk-forward 2019–2024** (`phase11/ablation.json`). Mean CRPS change over WR/TE receptions and yards
compared with the live model:

| Group | Change | 95% interval | Result |
| --- | --- | --- | --- |
| Volume-dependent dispersion | −0.42% | −0.53% to −0.32% | survived |
| Position-aware redistribution, w = 0.5 | −0.24% | −0.41% to −0.06% | survived |
| Position-aware redistribution, w = 1.0 | +0.70% | +0.35% to +1.05% | worse |

Both still helped when combined: removing dispersion costs +0.42% and removing redistribution costs +0.23%.

**B. Confirmation** (both together vs the live model):

| Set | Change | 95% interval | Receptions | Yards |
| --- | --- | --- | --- | --- |
| 2025 | −0.81% | −1.34% to −0.28% | −0.92% | −0.70% |
| 2026 weeks 1–7 | −0.61% | −1.31% to +0.09% | −0.56% | −0.66% |
| Pooled | −0.75% | −1.16% to −0.30% | −0.81% | −0.69% |

**Adopted.** This is the largest WR/TE gain of any phase.

**C. WR/TE recalibration and pass check, 2025** (`phase11/recal_wrte.json`):

| Market | Version | Vs baseline (B2) | Calibration error | 80% ranges | Passes | Recalibration adopted |
| --- | --- | --- | --- | --- | --- | --- |
| Receptions | raw | −1.81% (−2.89% to −0.71%) | 0.014 | 81.0% | **yes** | no (error went up) |
| Receiving yards | recalibrated | −1.74% (−2.72% to −0.81%) | 0.010 → 0.008 | 79.5% | yes | yes |

- **WR/TE receptions now passes every rule.** Before Phase 11 it missed only "beats baseline" (−0.84%, interval
  reaching +0.17%).
- On 2026 weeks 1–7, WR/TE 80% ranges held 78.5% (receptions) and 76.9% (yards) of results. The 90% in week 6 alone was
  noise, not a sign that ranges are too wide. With the yards map, the 2026 calibration error was slightly higher
  (0.018 → 0.023). It is adopted under the pre-registered 2025 rule, and the live record will show whether that holds.

**D. Engine recalibration refit with the role signal** (`phase11/recal_engine.json`). Same rule as Phase 8:

- **Recalibration adopted:** QB passing yards, QB passing TDs, RB rushing attempts (new), anytime TD and kicker points.
- **Dropped:** receiving TD. With the role signal its error no longer improves.
- **Pass check:** anytime TD still passes. No other engine market changes status.

**Overall:** 3 of 16 markets now pass (WR/TE receptions, WR/TE receiving yards, anytime TD). Live projections use
all of this from the week 9 freeze. The week 8 freeze stays as it is.
