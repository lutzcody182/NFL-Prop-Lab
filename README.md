# NFL Prop Lab

A point-in-time NFL player prop modeling platform. Architecture and roadmap:
see the "NFL Prop Lab — Architecture & Phased Plan" doc.

**Status:**
- Phase 0 (Foundations): gate passed.
- Phase 1 (WR/TE receptions and receiving yards): **gate not passed.** M1 beat both baselines in every season, and the yards market passed. But the 2025 receptions gain over B2 (−0.63% CRPS, 95% interval −1.52% to +0.29%) is not distinguishable from zero. Details: `phase1/PREREGISTRATION.md`, `phase1/results.json`, and the dashboard (`index.html`).

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
.github/workflows/       nightly ingest + checks + tests
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
