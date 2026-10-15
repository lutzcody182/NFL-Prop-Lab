# Phase 5 pre-registration: live weekly tracking ("paper trading") and the full dashboard

Written 2026-10-15, after week 6's predictions were frozen and before any week 6 game.

## What gets frozen, and when

- Every week, the projections for every market (WR/TE from Phase 1; QB, RB and anytime TD from Phase 3a) and the
  Phase 4 game simulations are copied into `ledger/<season>_w<week>/` with a manifest of SHA-256 fingerprints,
  the freeze time, the first kickoff and the code version.
- The freeze **refuses to run after the week's first kickoff** and **never overwrites** an earlier freeze.
- Models are not changed during the season. Any model change starts a new version, recorded in the manifest.
- Week 6 of 2026 (frozen 113 minutes before kickoff) is the first live week. Weeks 1–5 are backtests, never live.

## How a frozen week is scored (once all its games are final and stat corrections are in)

Per market:
- **80% range coverage:** share of outcomes between the stored 10th and 90th percentiles. Target 80%.
- **Over/under probabilities** at the stored lines: Brier score and calibration (ECE vs its noise limit, as in Phase 3a).
- **Quantile score** (average pinball loss over the stored 5th–95th percentiles, an approximation of CRPS) from
  week 7 on; week 6 was frozen before quantiles were added to the export, so it is scored on the two items above.
- Phase 4 simulations: CRPS for each simulated player and the two-leg event (QB and top receiver above their medians).
- Players who did not play (inactive) are dropped and counted, as sportsbooks void those props.

## Drift flags (cumulative over live weeks)

A market is flagged on the Model Health page when, over all scored live weeks:
- coverage is outside 80% ± max(3 points, 1.96·√(0.16/n)), or
- ECE of its over/under probabilities is above its noise limit.

A flag means "look at this market", not an automatic model change; any fix becomes a new model version.

## What Phase 5 can and cannot claim

- **Can:** an honest, timestamped weekly record of forecast accuracy and calibration.
- **Cannot yet:** anything about betting value. The plan's gate ("a live season of closing-line value before any value
  claims") needs sportsbook prices (Phase 2, waiting on an odds API key). Until then, no expected-value numbers are shown.

## Deviations

(none yet)
