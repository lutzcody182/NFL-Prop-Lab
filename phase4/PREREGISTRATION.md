# Phase 4 pre-registration: correlated game simulation (passing game)

Written 2026-10-14, before the simulator was built or scored.

## What changes

Until now each player was simulated on his own, so a QB's passing yards and his receivers' yards could not
be added up and never moved together. Phase 4 simulates each team's passing game as one unit:

1. **Team pass attempts** ~ negative binomial around the team-volume prediction (Phase 3a engine), with the two
   teams in a game linked by a Gaussian copula (correlation ρ: positive = shootouts, negative = game script).
2. **Starting QB early exit** with the Phase 3a probability; the starter throws a uniform 0–35% of attempts when he exits.
3. **Targets** = binomial(attempts, τ); split among the active receivers (WR, TE, RB, plus "everyone else") by a
   Dirichlet-multinomial whose mean shares reproduce each player's expected targets from the locked models.
4. **One efficiency shock per team per simulated game** (normal, sd σ): shifts every receiver's catch probability
   (logit scale) and yards per catch together, so a good passing day lifts the QB and his receivers at once.
5. Catches, yards per catch and receiving TDs per player as in the locked models. **QB completions, passing yards and
   passing TDs are the sums over his throws** (each catch thrown by the starter with probability = his share of attempts).

Rushing is not changed in this phase.

## Fitting (each season uses only the three seasons before it)

- Team-attempt dispersion and calibration: maximum likelihood on team-games.
- ρ: correlation of the two teams' standardized attempt residuals in the same game.
- τ: league targets per pass attempt.
- Dirichlet concentration κ: maximum likelihood over κ ∈ {5, 10, 20, 40, 80, 160, 320}.
- **Tuned:** σ ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30}, chosen on 2022–2024 by the primary joint score below.

## Scoring (per team-game)

Vector y = [starting QB passing yards, QB completions, receiving yards and receptions of the team's three receivers
with the most expected targets]. Each component is divided by its simulated standard deviation.

- **Primary — variogram score** (order 0.5, all pairs). Proper and sensitive to the dependence between components.
- **Baseline — same simulation with each component's draws shuffled independently.** Identical marginals,
  no dependence, so the comparison isolates the dependence structure.
- **Two-leg events:** P(QB passing yards above its own simulated median AND top receiver's receiving yards above
  his own median). Brier score vs the independent product of the two marginals, and ECE vs the noise limit
  (95th percentile under the model's own probabilities, as in Phase 3a).
- **Marginal non-inferiority:** QB passing yards, WR/TE receptions and receiving yards, RB receptions and receiving
  yards, scored with CRPS against the locked models on the same rows.

## Splits and gate

Joint outcomes were never scored before, so **2025 is a fresh test** for the joint criteria (scored once);
tuning uses 2022–2024; 2026 weeks 1–5 are reported as live.

The gate passes on 2025 when all of:
1. Variogram score lower than the shuffled baseline, 95% bootstrap interval over games below zero.
2. Two-leg Brier lower than the independent product, 95% interval below zero; and ECE ≤ its noise limit.
3. Every marginal market listed above: CRPS ratio vs the locked model with 95% interval upper end below +1%.

## Deviations

1. **Fixed scales for the variogram score** (decided on 2022–2024 tuning output, before any 2025 score). Dividing each
   component by its *simulated* standard deviation makes the score depend on the model's own spread: a larger
   efficiency shock widens the spread and lowers the score mechanically, so the σ grid simply picked its largest value.
   Each component is now divided by its standard deviation in the three seasons before the one scored (the same scales
   for every setting). The joint-vs-shuffled comparison at a given σ was unaffected; σ was re-tuned with the fixed scales.

## Results (added after scoring 2025 once)

Gate **not passed** (2 of 4 criteria).

| Criterion (2025, 544 team-games) | Result |
| --- | --- |
| Variogram score vs shuffled | ✓ 12.1% lower, interval clear of zero |
| Two-leg Brier vs independent product | ✕ 0.2104 vs 0.2116, interval crosses zero |
| Two-leg calibration | ✓ ECE 0.030, noise limit 0.042 |
| Marginal non-inferiority | ✕ QB passing yards +1.6% (interval −2.3% to +6.2%); WR/TE and RB markets passed |

- Two-leg event happened 29.8% of the time; the simulator said 32.7%, independence said 24.8%.
- 2026 weeks 1–5 (158 team-games): variogram ✓ (−8.9%), two-leg Brier ✓ (interval below zero), calibrated ✓,
  marginals ✕ (QB interval too wide; WR/TE receptions upper end +1.01%).
- Lesson for future rules: a "+1% upper bound" is not reachable with ~500 QB games (the interval alone is about ±4%),
  the same sample-size mistake as Phase 3a's original ECE rule. The verdict stands as pre-registered.
- Decision: player projections stay on the locked marginal models. The simulator is shown on the dashboard for
  same-game combinations, labelled with this result.

## Bug found after scoring, and corrected re-run

A coherence test written after the 2025 score found that when the starting QB exits early, his attempts and his
completions were thinned independently, so a few simulated games (only exit draws, ~5%) had more completions than
attempts. Fixed: the starter's throws are a random subset of the team's attempts (multivariate hypergeometric).

- The **official verdict is the first run** (saved as `phase4/results_official_first_run.json`): not passed.
- The corrected simulator was re-tuned on 2022–2024 (σ = 0.05 instead of 0) and re-scored, as a second look at 2025:
  variogram −12.3% ✓, two-leg Brier 0.2090 vs 0.2096 (interval crosses zero) ✕, calibrated ✓, QB marginal +1.8%
  (−2.2% to +6.4%) ✕. Same criteria pass and fail. 2026 weeks 1–5: variogram ✓, two-leg ✓, calibrated ✓, QB ✕.
- The dashboard and `phase4/results.json` use the corrected simulator.
