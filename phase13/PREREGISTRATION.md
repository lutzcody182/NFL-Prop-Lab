# Phase 13 pre-registration: fixing the RB low bias (drop RBs listed Out/Doubtful)

Written 2026-11-11, before the confirmation below was run.

## What was found (weeks 6–9 of 2026, live ledger)

- Over 2026 weeks 7–9, the model's over chances for RB carries were lower than the market's and than how often overs
  actually hit. RB rush attempts was the one market where the model was clearly worse than the market.
- The bias comes from the actives assumption (each team's actives in its last game). For RBs whose projected teammate
  RBs all played, projected and actual carries matched (11.6 vs 11.6). In the 15% of cases where a projected teammate
  RB did not play, the RBs who did play were projected 8.4 carries and 35 yards but got 12.4 and 47. The missing RB's
  share went unused. WR/TE markets showed almost no such effect.
- In the Phase 7 backtest (2025 and 2026 weeks 1–5), method R (drop players listed Out or Doubtful) improved RB rush
  attempts (−1.4% CRPS) and RB rushing yards (−0.7%), but worsened WR/TE receptions (+0.7%) and QB rushing yards (+2.2%).
  Limiting removals to RBs is therefore a choice made after seeing those results, and it needs data that backtest never
  used.

## Test

- **Method B:** each team's actives in its last game, minus **RBs** listed Out or Doubtful on that week's final injury
  report. Shares are redistributed among the remaining players exactly as now.
- **Confirmation data:** 2026 weeks 6–9, scored with the Phase 7 backtest code (same models, same simulation), against
  method L (current). Primary markets: RB rush attempts and RB rushing yards.
- **Adopt if:** B improves mean CRPS on both primary markets, AND the mean change over the two has a 95% game-bootstrap
  interval below zero, AND no other market gets worse by more than 0.5%.
- **If adopted:** the projection runs apply B whenever that week's final injury report is in the data (the
  Saturday refresh and any later run). The Tuesday freeze is unchanged, because the final report does not exist yet. So
  this improves the dashboard's Saturday RB projections and the live record compares both. Tracked picks are
  unaffected (WR/TE only).
- **If not adopted:** keep method L, and show a note on RB projections when a teammate RB is listed Out or Doubtful.

## Deviations

(none yet)

## Results and decision (added after scoring)

Confirmation on 2026 weeks 6–9 (`phase13/rb_actives_confirm.json`), CRPS change vs method L:

| Method | RB rush attempts | RB rushing yards | QB rushing yards | All markets (95% interval) |
| --- | --- | --- | --- | --- |
| **B (drop RBs Out/Doubtful)** | **−2.3%** | **+0.4%** | +2.4% | +0.02% (−0.44% to +0.53%) |
| R (drop everyone Out/Doubtful) | −1.7% | +1.3% | +2.5% | +0.06% (−0.44% to +0.54%) |
| O (perfect knowledge of who plays) | −5.4% | −3.1% | +2.8% | −0.73% (−1.29% to −0.12%) |

**Not adopted:** RB rushing yards did not improve, and QB rushing yards got worse by more than 0.5%. Projections keep
method L.

Why: of the 362 projected player-games (all positions) where the player did not play, removing RBs on the injury
report caught only 41, and removing everyone on it (R) caught 141. Most players who sit, RBs included, are healthy
scratches who never appear on the report. Perfect knowledge (O) would remove most of the bias, but that
information only exists after inactives are announced, 90 minutes before kickoff.

Fallback, as planned: on the Projections tab, an RB whose teammate RB is listed Out or Doubtful gets an "RB ↑" tag. It
says he is probably projected too low.
