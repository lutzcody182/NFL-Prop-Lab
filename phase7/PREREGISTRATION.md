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

1. **No weekend refresh.** While building it we found that nflverse's weekly rosters do not show game-day inactives
   before kickoff (week 6 of 2026 had no inactive rows the day before its games; they appear after). So method O cannot
   run live from our data, and the Saturday/Sunday refresh would add nothing for actives. The weekly run gains a Saturday
   slot instead, which only refreshes data, lines and the dashboard (shows each player's final injury designation);
   the Tuesday freeze stays the official record and there is no `late` snapshot.
2. **Weather multiplier split:** "applied to completion rate and yards per completion" is implemented as √m on each,
   so pass yards per attempt move by m (applying the full m to both would square the effect). Only γ = 1 was tried.
3. **Kicker scoring re-run twice before any decision:** the first 2025 run had a bracket bug in the bootstrap interval
   (printed NaN) and a non-deterministic choice between two listed kickers when neither kicked. Both were fixed, 2024
   tuning was re-run (same choice: B1 h=4, a=10; K h=8, a=20) and 2025 re-scored. Point estimates moved by under 0.05%.
4. **One exploratory variant after seeing the result:** R = L minus Out/Doubtful only (no add-backs). It is reported
   below as exploratory and is not adopted; it can only be adopted after it holds up on live weeks not yet played.

## Results and decision (added after scoring)

Backtest on 2025 weeks 1–18 and 2026 weeks 1–5 (`phase7/actives_backtest.json`; 19,443 played, eligible player-markets):

| Method | Projected | Void (did not play) | Missed (played, not projected) | CRPS vs L on common rows |
| --- | --- | --- | --- | --- |
| L, last game (current) | 19,639 | 1,934 | 1,768 | — |
| I, injury report (pre-registered) | 22,877 | 4,298 | 884 | +1.77% (worse; 95% +1.26% to +2.28%) |
| R, removals only (exploratory) | 19,013 | 1,293 | 1,753 | +0.06% (−0.20% to +0.30%) |
| O, actual actives (ceiling) | — | 0 | 0 | −0.43% (−0.75% to −0.11%) |

- **I fails both adoption conditions:** it is worse on accuracy, and its add-back rule more than doubles the voided
  projections (players who were recently active but are cut, on injured reserve or benched).
- Even perfect knowledge of who plays (O) improves accuracy for players who do play by only 0.4%: the value of injury
  information is mainly in *who* gets projected, not in the numbers.
- **Decision:** projections keep method L. The dashboard shows each player's final injury designation (Out, Doubtful,
  Questionable) once it is published, so Out players can be skipped by eye; R will be checked on live weeks.

**B. Weather** (`phase7/weather_ablation.json`): fitted effects are sensible (each mph of wind above 10 costs about
0.6% of pass yards per attempt; each degree below 40 °F about 0.4%), but on 2019–2024 the multiplier improved CRPS by
only 0.08% (95% −0.20% to +0.04%). **Did not survive; not used.** No live weather fetch was added.

**C. New markets, 2025** (`phase7/markets_results.json`, `phase7/kicker_results.json`):

| Market | vs baseline | 95% interval | Calibrated | 80% ranges | Result |
| --- | --- | --- | --- | --- | --- |
| RB rushing + receiving yards | −3.1% | −4.4% to −1.9% | no (ECE 0.028 vs 0.024) | 75% (too narrow) | fail |
| QB completions | 0.0% (baseline kept) | — | no (0.039 vs 0.037) | 82% | fail |
| Receiving TD | −0.0% | −0.3% to +0.3% | yes | — | fail |
| Kicker points (K vs B1) | −1.7% | −2.9% to −0.4% | no (0.042 vs 0.036) | 85% (too wide) | fail |

All four are shown on the dashboard marked "did not pass", like the Phase 3a markets.
