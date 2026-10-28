# NFL Prop Lab

A point-in-time NFL player prop modeling platform. Architecture and roadmap:
see the "NFL Prop Lab — Architecture & Phased Plan" doc.

**Status:**
- Phase 0 (Foundations): gate passed.
- Phase 2 (market comparison): built and tested; on hold until historical odds are downloaded (see below).
- Phase 3a (10 more markets: QB passing/rushing, RB rushing/receiving, anytime TD): **0 of 10 passed** on 2025.
  RB rushing yards (−3.5% CRPS vs baseline) and attempts (−5.1%) and anytime TD (−0.5% Brier) beat the baseline
  with intervals clear of zero but are slightly miscalibrated; QB passing kept no improvement over the baseline;
  the shared carries setup made QB rushing 2.6% worse. Details: `phase3/PREREGISTRATION.md`, `phase3/results.json`.
- Phase 3b (ratings): **player ratings pass** on 2025 at every position (11–34% less error than raw averages);
  **team run ratings pass** (−4.5%), team pass ratings do not (−0.4%, interval crosses zero).
- Phase 3c (matchup engine): run gap and team ratings in volume survived 2019–2024 ablation by under 0.2%;
  depth of target, pressure and receiver archetypes did not. Neither survivor confirmed on 2026 weeks 1–5 or
  2025, so **projections stay on the locked Phase 1/3a models**. A leak (league constants using future seasons)
  was found by a perturbation test, fixed, and 3c re-run. Details: `phase3/PREREGISTRATION_3bc.md`.

- Phase 8 (recalibration): ranges and over/under chances corrected from 2022–2024 and re-tested on 2025; used for
  QB passing yards and TDs, anytime TD (now passes), receiving TD and kicker points. Details: `phase8/`.
- Phase 10 (WR/TE upgrades): the role signal and an air-yards catch-up term passed every step in the Phase 1 WR/TE
  model (pooled −0.37% CRPS) and are live for WR/TE; a quarterback-change adjustment did not help and is not used.
  WR/TE yards still pass their gate on 2025. Details: `phase10/PREREGISTRATION.md`.
- Phase 11 (sharper WR/TE ranges, free data only): volume-dependent target dispersion and position-aware
  redistribution (w = 0.5) improved WR/TE CRPS by 0.75% pooled over 2025 and 2026 weeks 1–7, and were adopted. WR/TE receptions
  now passes (3 of 16 markets pass). Recalibration was refit on the role-signal engine, and WR/TE yards is now recalibrated.
  See `phase11/PREREGISTRATION.md`.
- Phase 9 (role changes, game scores, live lines): the **snap-share role signal passed every step** (−0.6% CRPS on
  2019–2024, −0.5% on 2025, pooled interval below zero) and is live for RB and TD markets. A nearest-games score model
  lost to a simple normal model, which now drives the Game simulator's score, win, cover and total chances (from the
  betting line). The weekly run snapshots sportsbook prop lines (key from the `ODDS_API_KEY` secret, budget-capped) and
  shows them next to the model; a model-vs-market record builds on the Weekly results tab. Details: `phase9/`.

- Phase 7 (injuries, weather, more markets): the pre-registered injury-report lineup made projections **worse**
  (+1.8% CRPS; it added back players who then did not play), so projections keep last game's lineup and the dashboard
  shows each player's injury designation instead. Weather helped 0.08% (within noise) and is not used. New markets
  (RB rushing + receiving yards, QB completions, receiving TD, kicker points) are live on the dashboard; none passed
  every rule. Also new: "Your line" calculator and the Weekly results tab. Details: `phase7/PREREGISTRATION.md`.

```bash
python -m proplab.p7.actives backtest              # who-plays methods -> phase7/actives_backtest.json
python -m proplab.p7.weather decide | confirm      # weather -> phase7/weather_ablation.json
python -m proplab.p7.markets                       # new engine markets -> phase7/markets_results.json
python -m proplab.p7.kicker tune | backtest        # kicker points -> phase7/kicker_*.json
```

- Phase 6 (defensive scheme engine): built from FTN play charting (2022+): blitz rate, stacked boxes, play-action
  yards allowed, each matched to the player's own split. Only stacked box helped on 2023–2024 (−0.1% CRPS, RB rushing
  yards) and it **did not confirm** on 2026 weeks 1–5 or 2025, so projections don't use it; profiles are shown as
  descriptive. Coverage (man/zone) is out of scope: that data stops after 2025. Details: `phase6/PREREGISTRATION.md`.

```bash
python -m proplab.p6.ablation decide | confirm   # walk-forward 2023-2024; confirmation -> phase6/ablation.json
python -m proplab.p6.export                      # scheme profiles for the dashboard
```

- Phase 5 (live tracking + full dashboard): **running.** Every week's projections are frozen in `ledger/` before
  the first kickoff (SHA-256 fingerprints; the freeze refuses after kickoff and never overwrites) and scored once
  games are final plus 72 hours for stat corrections: 80% range coverage, over/under Brier and calibration,
  quantile score, simulator CRPS and two-leg events, with drift flags. Week 6 of 2026 was the first frozen week
  (113 minutes before kickoff). No betting-value claims until sportsbook prices are available (Phase 2).
  Rules: `phase5/PREREGISTRATION.md`. Dashboard pages: Week overview, Projections, Game simulator, Ratings,
  Matchups, Player profiles, Model performance, Model health, Method and notes.

```bash
python -m proplab.live.weekly           # whole weekly run: ingest -> ratings -> projections -> sims -> freeze -> score -> dashboard
python -m proplab.live.freeze           # freeze the next week (before its first kickoff)
python -m proplab.live.score            # score frozen weeks that are final -> phase5/live_record.json
```

The `weekly` GitHub Actions workflow runs this every Tuesday and Wednesday and commits the ledger and
dashboard back to the repo (needs the repo on GitHub with Actions enabled).

- Phase 4 (correlated game simulation, passing game): **gate not passed.** Linking players beat treating them as
  independent by 12% (variogram score, interval clear of zero), but two-outcome probabilities were not clearly
  better than multiplying on 2025 (they were on 2026 weeks 1–5), and the QB passing-yards non-inferiority check
  was too uncertain at ~500 games. Single-player projections stay on the earlier models; the simulator powers the
  dashboard's same-game calculator. A coherence bug was found after scoring, fixed, and re-scored (same verdict).
  Details: `phase4/PREREGISTRATION.md`.

```bash
python -m proplab.p4.sim tune | test    # sigma on 2022-2024; 2025 + 2026 scoring
python -m proplab.p4.export             # next week's simulated draws for the dashboard
```

```bash
python -m proplab.ratings.team tune | test | publish     # team ratings -> DB table team_ratings
python -m proplab.ratings.player tune | test | publish   # player ratings -> phase3/player_ratings_current.json
python -m proplab.p3.ablation decide | confirm           # matchup feature groups -> phase3/ablation.json
python -m proplab.p3.export_3bc                          # ratings + matchup data for the dashboard
```

```bash
python -m proplab.p3.run tune       # 2024 only -> phase3/tuning.json
python -m proplab.p3.run backtest   # -> phase3/results.json (2025 already used; do not re-tune on it)
python -m proplab.p3.project        # next week -> phase3/projections.json
python -m proplab.p1.publish        # rebuild index.html data
```
- Phase 1 (WR/TE receptions and receiving yards): **gate not passed.** M1 beat both baselines in every season, and the yards market passed. But the 2025 receptions gain over B2 (−0.63% CRPS, 95% interval −1.52% to +0.29%) is not distinguishable from zero. Details: `phase1/PREREGISTRATION.md`, `phase1/results.json`, and the dashboard (`index.html`).

## Phase 2: model vs sportsbook prices

Status: **code built and tested on synthetic data; waiting for real odds.** Plan and
pass rules: `phase2/PREREGISTRATION.md` (written before any odds were downloaded).

**Step 1: download odds (on your Mac, in Terminal).** Needs a paid The Odds API key.
Uses only built-in Python; nothing to install.

```bash
cd ~/Desktop/NFL-Prop-Lab
python3 tools/odds_backfill.py --estimate                  # cost check, uses no credits
python3 tools/odds_backfill.py --run --max-credits 60000   # asks for your key (hidden)
```

It saves raw responses to `data/odds/raw/`, skips anything already downloaded, and stops
before going over `--max-credits`. Re-run it weekly to add new games. `data/odds/` is in
`.gitignore`: sportsbook prices may not be redistributable, so they stay out of the public repo.

**Step 2: load and evaluate** (where the database runs):

```bash
python -m proplab.odds.load        # -> prop_lines, odds_events, bet-time game_lines; data/odds/unmatched.csv
python -m proplab.odds.evaluate    # -> phase2/results.json
```

`tests/fixtures/mock_odds_api.py` is a fake Odds API that serves **synthetic** prices so the
whole chain can be tested without spending credits. Its numbers mean nothing about real markets.

## Phase 1

```bash
python -m proplab.p1.tune       # tunes on 2024 only -> phase1/tuning.json
python -m proplab.p1.backtest   # scores 2024, 2025 (locked test), 2026 wks 1-5 -> phase1/results.json
python -m proplab.p1.project    # next week's projections -> phase1/projections.json
python -m proplab.p1.publish    # writes results + projections into index.html
```

Re-running `backtest` reproduces the committed 2025 numbers exactly (a test checks this).
Do not re-tune against 2025: it has been used. New ideas get judged on 2026 live weeks.

## Phase 0 gate

| Gate criterion | Result |
| --- | --- |
| Ingest is reproducible | Two back-to-back full runs (2016–2026) produce identical content hashes for all 10 tables |
| Data quality checks pass | 10 of 10 checks pass (see `proplab/quality.py`) |
| Leakage guards tested | 11 of 11 tests pass (see `tests/test_phase0.py`) |

Loaded (2016 through 2026 week 6): 3,033 games, 498,096 plays, 187,541
player-games, 260,477 snap rows, 481,450 weekly roster rows, 56,982 injury
report rows, 24,854 players with 50,913 cross-source IDs.

## Run it

```bash
pip install -r requirements.txt
# needs a PostgreSQL 16 database; default URL is postgresql://postgres@/nflprop?host=/tmp
export DATABASE_URL=postgresql://user:pass@host:5432/nflprop
python -m proplab.ingest                 # all seasons 2016..2026
python -m proplab.ingest --seasons 2026  # just the current season
pytest -q
```

The ingest exits non-zero if any quality check fails.

## Layout

```
db/schema.sql            tables, with known_at on everything time-dependent
proplab/ingest.py        nflverse -> Postgres loader
proplab/quality.py       data quality checks (results in data_quality_runs)
proplab/pit.py           point-in-time readers; all model inputs go through these
proplab/db.py            connection, bulk load, content hashing
tests/test_phase0.py     time conversion, leakage, quality, reproducibility tests
proplab/live/            Phase 5: freeze, score, weekly pipeline, dashboard export
ledger/                  frozen weekly projections (never edited)
.github/workflows/       nightly ingest + checks + tests; weekly pipeline
```

The dashboard (Phase 1 onward) will be `index.html` at the repo root for GitHub Pages.

## Point-in-time assumptions

Every row carries `known_at`, derived from kickoff so reruns are deterministic:

| Table | known_at | Why |
| --- | --- | --- |
| plays, player_game, snaps | kickoff + 1 day | nflverse nightly update |
| game_results | kickoff + 4 hours | final score |
| game_lines (`nflverse_close`) | kickoff, completed games only | closing consensus; live lines come from the odds provider in Phase 2 |
| roster_status | kickoff − 90 minutes | includes game-day inactives |
| injury_reports | kickoff − 40 hours | **A1:** nflverse has no report timestamp; final reports usually come out the day before the eve of game day |

Known approximations:

- **A1** (above): injury `known_at` is estimated, not observed. A Thursday-morning prediction never sees final designations; a Sunday-morning prediction does.
- **A2:** stat corrections (Mon–Wed) overwrite values but keep the kickoff + 1 day stamp. The effect is a few yards on a few players per week.
- **A3:** observed game weather in `game_results` is post-game only; models must use forecasts.

## Data notes

- nflverse player stats and play-by-play use current team codes (LV, LAC) for every season; schedules, rosters and snaps use the code of the time (OAK, SD). Ingest rewrites them to the schedule's code (found and fixed in Phase 1).
- The 2022 season has 271 regular-season games: Week 17 BUF @ CIN was cancelled.
- 78 of 68,528 skill-player snap rows (0.11%) have PFR IDs with no gsis match; they are kept with `gsis_id = NULL`.
- The dynastyprocess fantasy ID crosswalk is not used; the nflverse players file already carries PFR, ESPN and PFF IDs.

## Data credits

Data from [nflverse](https://github.com/nflverse) (CC-BY 4.0); FTN charting data CC-BY-SA 4.0.
