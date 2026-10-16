# Phase 7 pre-registration: injuries, weather, new markets

Written 2026-10-15 (evening), before any of these was built or scored. Same discipline as earlier phases: rules first,
one look at the test data, deviations logged below.

## A. Injury-aware expected actives

Live projections currently assume each team's actives from its last game (method **L**). Two alternatives:

- **I (injury report):** L, minus players whose final injury report says Out or Doubtful, plus players who missed the
  last game but were active within the team's previous 4 games and are not listed Out/Doubtful this week.
  Uses only injury reports (known_at = kickoff − 40 h) and earlier games.
- **O (actual actives, the ceiling):** the game-day active list (known_at = kickoff − 90 min). Used live only by a
  refresh that runs after inactives are posted; reported in the backtest as the best achievable.

**Test:** 2025 weeks 1–18 and 2026 weeks 1–5, every market, locked models, projections rebuilt per week exactly as the
live run builds them. Scored on the rows every method projected for players who actually played (so the sets match);
the effect comes through teammates' shares being redistributed. Also reported: share of projected players who did
not play (voided) and share of players who played but were not projected (missed).

**Adopt I** if the mean CRPS change vs L across markets (Brier for anytime TD) has a 95% game-bootstrap interval
entirely below zero, **or** if that interval includes zero and I lowers both the voided and the missed share.
Otherwise L stays. (I is rule-based: no parameters are fit, so there is no separate development set.)

**Live use:** the Tuesday run freezes the official week as now. A Saturday run (after Friday's final reports) and a
Sunday run (after inactives) re-project games that have not kicked off and freeze them as separate `late` snapshots;
both are scored. The official record stays the Tuesday freeze.

## B. Weather

Outdoor games only (roofs closed/dome = no effect). One pass-game multiplier per team-game,
`m = exp(b_w·max(wind − 10 mph, 0) + b_t·max(40 °F − temp, 0))`, applied to completion rate and yards per completion
(QB attempts, WR/TE/RB targets). `b_w`, `b_t` are fit on earlier seasons only (team pass yards per attempt relative to
its model expectation, least squares), refit each season.

- **Development:** walk-forward 2019–2024 on QB passing yards, WR/TE receptions and receiving yards; survives if the
  mean CRPS change has a 95% interval below zero (same rule as 3c / 6).
- **Confirmation and adoption:** as Phase 6 — below zero on 2025 and on 2026 weeks 1–5, and pooled interval below zero.
- **Caveat:** history uses the weather recorded at the game, not the forecast made days earlier, so the backtest is an
  upper bound on what a forecast can deliver. Live forecasts come from Open-Meteo (free, no key) in the GitHub run;
  if the fetch fails, no adjustment is applied.

## C. New markets (each scored once on 2025, same pass rules as Phase 3a)

- **RB rushing + receiving yards** and **QB completions:** from the locked engine's joint draws (no new tuning).
- **Receiving TD (RB/WR/TE, yes/no):** from the locked engine's target draws (no new tuning).
- **Kicker points:** new model. Field goals made and extra points made per game are Poisson with means from the kicker's
  shrunk recent rates (B1), or from the team's implied points and the kicker's share of team scoring (model K).
  Half-life and shrinkage are tuned on 2024 only; scored on 2025 vs B1.

Pass rule (as Phase 3a): beats its baseline with a 95% interval below zero, calibrated (ECE under its noise limit),
and 80% ranges within tolerance. Markets that fail are still shown, marked "did not pass", like the others.

## Deviations

(none yet)
