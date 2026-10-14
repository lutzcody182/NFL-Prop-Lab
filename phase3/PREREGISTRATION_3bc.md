# Phase 3b and 3c pre-registration: ratings and matchup engine

Written 2026-10-14, before any rating or matchup feature was fit or scored.
Changes after the first score must go under "Deviations" with a reason.

## Which seasons are still unused

- **Ratings (3b)** predict new targets (a team's next-game EPA per play, a player's future efficiency).
  No earlier phase scored these targets, so **2025 is a fresh, locked test**: tune on 2018–2024, score 2025 once.
- **Matchup features (3c)** change the prop markets, and **2025 has already been used for all 12 markets**.
  Their keep/drop decisions use walk-forward results on **2019–2024**; **2026 weeks 1–5 are the confirmatory test**
  (small; it grows as weeks finish). 2025 is also reported, labelled as a second look at a used season.

## 3b-1 Team ratings

- Separate models for **dropbacks** (incl. sacks and scrambles) and **designed runs**.
- Weighted ridge regression on each play's EPA: intercept + offense effect + defense effect + home + situation controls
  (down, distance, field position, garbage time = win probability below 10% or above 90%, 4th quarter).
  Ridge penalty only on team effects, which makes them shrunk random-effect estimates (empirical Bayes).
- Point in time: ratings for week W use only games before week W; plays are down-weighted by age with half-life H weeks.
  The previous season is included (it carries the most information early in a season). Franchise codes are unified
  across relocations (OAK→LV, SD→LAC, STL→LA).
- Tune penalty λ ∈ {50, 200, 800, 3200} and H ∈ {8, 16, 32} weeks on 2018–2024.
- **Target:** a team's actual EPA per play in its next game, by unit (offense dropbacks, offense runs).
- **Baseline:** raw (unadjusted) recency-weighted EPA/play: own offense + opponent defense allowed − league mean.
- **Gate:** play-weighted squared error of the adjusted ratings lower than the baseline on 2025, 95% bootstrap interval
  over games entirely below zero, for both units.

## 3b-2 Player efficiency ratings

- QB: EPA per dropback (passes, sacks, scrambles). RB: EPA per designed carry. WR/TE: EPA per target.
- Each play's EPA is opponent-adjusted by subtracting the defense's rating for that unit **as of that week** (3b-1).
- Empirical Bayes per position: recency-weighted mean (half-life in games, tuned in {8, 16, 32} on 2018–2024),
  shrunk toward the position mean with between-player variance estimated from earlier seasons.
  Stored: posterior mean, 80% interval, opportunities.
- Display: 0–100 = percentile of the posterior mean among qualified players at the position that week
  (qualified: at least 50 dropbacks / 30 carries / 20 targets in the window).
- **Target:** the player's opponent-adjusted EPA per opportunity over his next 4 games.
- **Baseline:** his raw (unshrunk, unadjusted) recency-weighted EPA per opportunity.
- **Gate:** opportunity-weighted squared error lower than the baseline on 2025 with a 95% interval below zero, every position.

## 3c Matchup feature groups

Each group is a multiplier on an efficiency (or volume) parameter of the locked prop models, with strength γ ∈ {0.5, 1}.
All defensive tendencies use only earlier games (half-life 16 games) and are shrunk toward league rates.

| Group | Markets | What it does |
| --- | --- | --- |
| **Depth of target** | WR/TE and RB receptions, receiving yards | Defense's catch rate and yards per target allowed in 4 depth buckets (behind line, 0–9, 10–19, 20+ air yards) vs league, weighted by the receiver's own depth mix |
| **Run gap** | RB rushing yards | Defense's yards per carry allowed inside / off-tackle / outside vs league, weighted by the back's own gap mix |
| **Pressure** | QB passing yards, passing TDs, interceptions | Defense pressure rate (sacks + QB hits per dropback) combined with the QB's own pressured-vs-clean yards per attempt and completion rate |
| **Receiver archetypes** | WR/TE receptions, receiving yards | Players clustered on earlier-season profiles (aDOT, YAC per catch, target share, catch rate; Gaussian mixture, cluster count chosen by bootstrap stability); the archetype replaces position as the shrinkage target for catch rate and yards per catch |
| **Team ratings in volume** | all 12 markets | Adds own offense and opponent defense ratings (3b-1) to the team volume regressions |

WR/TE markets use the Phase 1 M1 settings inside the Phase 3a engine as their base.

**Survival rule (the Phase 3 gate):** a group is kept only if adding it to the locked base improves the walk-forward
2019–2024 objective (mean CRPS relative to base, Brier for anytime TD) with a 95% bootstrap interval over games below
zero. Kept groups are then combined; any group whose removal from the combination no longer hurts is dropped.
Results on 2026 weeks 1–5 are reported as the confirmatory test, with intervals, whether they agree or not.

Age curves (plan item 8) are deferred: they need multi-season targets this phase does not test.

## Deviations

(none yet)
