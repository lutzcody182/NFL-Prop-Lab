# Phase 6 pre-registration: defensive scheme engine

Written 2026-10-15, before any scheme feature was built or scored.

## Data, and what it limits

- **FTN charting** (via nflverse, CC-BY-SA 4.0, credit FTN Data) charts every play from 2022 on: blitzers, pass
  rushers, defenders in the box, play action, screens, motion. It is updated during the season, so it can feed live
  projections. Stored in `ftn_charting` with `known_at` = kickoff + 2 days (assumption A6: FTN's publishing lag; the
  weekly run on Tuesday only uses what is actually loaded).
- **Coverage (man/zone, shell) and defensive personnel** exist only in the NGS participation file, which stops at 2025.
  A feature that cannot be computed for 2026 cannot be used in live projections, so coverage is **out of scope**.
- History is short: 2022 is used only as history, development is **2023–2024**, confirmation is 2026 weeks 1–5 and 2025.

## Feature groups (all prior-only: earlier games only; league rates from prior seasons only)

Defense tendencies use an exponentially weighted average over the defense's previous games (half-life 16 games),
shrunk toward the league rate of the prior seasons (2023 uses 2022 alone; later seasons the three prior seasons).

1. **Blitz** (QB pass yards, pass TDs, interceptions). `h_j` = defense's share of charted dropbacks with at least one
   blitzer (pseudo-count 150 dropbacks). `s_i` = the QB's blitzed / not-blitzed ratio for completion rate, yards per
   completion, TD rate and INT rate over his earlier games, shrunk toward the league ratio (pseudo 40 blitzed
   attempts; 200 for TD and INT). Multiplier = [(1−h_j) + h_j·s_i] / [(1−h̄) + h̄·s_i], raised to γ_blitz.
2. **Box** (RB rushing yards). `h_j` = defense's share of faced designed runs with 8+ in the box (pseudo 150 runs).
   `s_i` = the RB's yards per carry against 8+ / fewer, shrunk toward the league ratio (pseudo 40 heavy-box carries).
   Same multiplier form on yards per carry, raised to γ_box. (Box count is partly chosen by the offense's personnel;
   that is why it is tested rather than assumed.)
3. **Play action** (QB pass yards). `p_i` = the QB's play-action share of attempts (shrunk to league, pseudo 100).
   `R_j`, `Q_j` = defense's yards per attempt allowed on play-action / non-play-action passes relative to league
   (pseudo 100 attempts each). Multiplier on yards per completion = [p_i·R_j + (1−p_i)·Q_j] / [p̄·R_j + (1−p̄)·Q_j],
   raised to γ_pa: the QB's play-action habit against this defense's play-action weakness, compared with an average QB.

Each γ is tried at 0.5 and 1.0. The base is the locked Phase 3a models (Phase 1 M1 settings for WR/TE), as in 3c.

## Rules

- **Leakage test (must pass before scoring):** changing one week's charted plays must leave every feature for that week
  and earlier unchanged.
- **Survival (development, 2023–2024 walk-forward):** a group survives if its better γ improves the mean CRPS of its
  markets relative to base (Brier for binary) with a 95% bootstrap interval over games entirely below zero. Survivors
  are combined; any whose removal no longer hurts is dropped.
- **Confirmation:** the combined survivors vs base on 2026 weeks 1–5 and on 2025, reported with intervals.
- **Adoption into published projections** only if the change is below zero on both confirmation sets AND the pooled
  (2025 + 2026 weeks 1–5) 95% interval is entirely below zero. Otherwise the scheme profiles are shown on the dashboard
  as descriptive only, as with 3c.
- Any adopted change becomes a new model version in the Phase 5 ledger manifest; frozen weeks are never re-scored.

## Deviations

(none yet)

## Results and decision (added after scoring)

- **Leakage test passed** before scoring (`tests/test_phase6.py`): changing week 9 of 2025 leaves every feature for
  that week and earlier unchanged, and does change week 10.
- **Development (2023–2024):** blitz did not help (best γ = 1.0: +0.01%, interval −0.14% to +0.16%); play action did
  not help (−0.02%, interval −0.05% to +0.02%); **stacked box survived** (γ = 1.0: −0.105% CRPS on RB rushing yards,
  interval −0.17% to −0.04%). γ was picked from two values on the same seasons, so that interval is slightly optimistic.
- **Confirmation failed:** 2026 weeks 1–5 +0.03% (interval −0.13% to +0.18%); 2025 −0.04% (−0.15% to +0.06%);
  pooled −0.03% (−0.12% to +0.06%). The adoption rule needed both below zero and the pooled interval below zero.
- **Decision:** published projections stay on the locked Phase 1 / 3a models. Scheme profiles are shown on the
  Matchups tab as descriptive only. Details: `phase6/ablation.json`.
