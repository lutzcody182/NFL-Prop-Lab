-- NFL Prop Lab — Phase 0 schema
-- Rule: every row that describes something learned over time carries known_at
-- (when it became knowable). Feature queries must filter known_at <= prediction_ts.
-- loaded_at is bookkeeping only and never used for modeling.

CREATE TABLE IF NOT EXISTS players (
    gsis_id        TEXT PRIMARY KEY,
    display_name   TEXT NOT NULL,
    position       TEXT,
    position_group TEXT,
    birth_date     DATE,
    height_in      INT,
    weight_lb      INT,
    rookie_season  INT,
    draft_year     INT,
    draft_round    INT,
    draft_pick     INT,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS player_id_map (
    gsis_id        TEXT NOT NULL REFERENCES players(gsis_id),
    source         TEXT NOT NULL,          -- pfr | espn | pff | odds_api
    source_id      TEXT NOT NULL,
    valid_from     TIMESTAMPTZ,
    valid_to       TIMESTAMPTZ,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, source_id, gsis_id)
);

CREATE TABLE IF NOT EXISTS games (
    game_id        TEXT PRIMARY KEY,
    season         INT NOT NULL,
    game_type      TEXT NOT NULL,          -- REG, WC, DIV, CON, SB
    week           INT NOT NULL,
    kickoff_utc    TIMESTAMPTZ NOT NULL,
    home_team      TEXT NOT NULL,
    away_team      TEXT NOT NULL,
    home_rest      INT,
    away_rest      INT,
    div_game       BOOLEAN,
    roof           TEXT,
    surface        TEXT,
    stadium_id     TEXT,
    stadium        TEXT,
    location       TEXT,                   -- Home | Neutral
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS games_season_week ON games(season, week);

-- Outcomes are separate from the schedule: they are only known after the game.
CREATE TABLE IF NOT EXISTS game_results (
    game_id        TEXT PRIMARY KEY REFERENCES games(game_id),
    home_score     INT,
    away_score     INT,
    overtime       BOOLEAN,
    temp_f         INT,                    -- observed weather: post-game only
    wind_mph       INT,
    known_at       TIMESTAMPTZ NOT NULL,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS game_lines (
    game_id        TEXT NOT NULL REFERENCES games(game_id),
    book           TEXT NOT NULL,          -- 'nflverse_close' = closing consensus from schedules
    snapshot_ts    TIMESTAMPTZ NOT NULL,
    is_closing     BOOLEAN NOT NULL,
    spread_line    NUMERIC,                -- positive = home favored (nflverse convention)
    total_line     NUMERIC,
    home_moneyline INT,
    away_moneyline INT,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, book, snapshot_ts)
);

CREATE TABLE IF NOT EXISTS plays (
    game_id        TEXT NOT NULL REFERENCES games(game_id),
    play_id        INT  NOT NULL,
    season         INT  NOT NULL,
    week           INT  NOT NULL,
    posteam        TEXT,
    defteam        TEXT,
    qtr            INT,
    down           INT,
    ydstogo        INT,
    yardline_100   INT,
    game_seconds_remaining INT,
    play_type      TEXT,
    pass           BOOLEAN,
    rush           BOOLEAN,
    qb_dropback    BOOLEAN,
    sack           BOOLEAN,
    qb_hit         BOOLEAN,
    qb_scramble    BOOLEAN,
    complete_pass  BOOLEAN,
    interception   BOOLEAN,
    touchdown      BOOLEAN,
    shotgun        BOOLEAN,
    no_huddle      BOOLEAN,
    air_yards      NUMERIC,
    yards_after_catch NUMERIC,
    yards_gained   NUMERIC,
    epa            NUMERIC,
    success        BOOLEAN,
    wp             NUMERIC,
    xpass          NUMERIC,
    pass_oe        NUMERIC,
    cpoe           NUMERIC,
    pass_length    TEXT,
    pass_location  TEXT,
    run_location   TEXT,
    run_gap        TEXT,
    passer_player_id   TEXT,
    receiver_player_id TEXT,
    rusher_player_id   TEXT,
    td_player_id       TEXT,
    known_at       TIMESTAMPTZ NOT NULL,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, play_id)
);
CREATE INDEX IF NOT EXISTS plays_season_week ON plays(season, week);

CREATE TABLE IF NOT EXISTS player_game (
    gsis_id        TEXT NOT NULL,
    game_id        TEXT NOT NULL REFERENCES games(game_id),
    season         INT  NOT NULL,
    week           INT  NOT NULL,
    team           TEXT NOT NULL,
    opponent       TEXT NOT NULL,
    position       TEXT,
    completions    INT, attempts INT, passing_yards NUMERIC, passing_tds INT,
    passing_interceptions INT, sacks_suffered INT, passing_air_yards NUMERIC,
    carries        INT, rushing_yards NUMERIC, rushing_tds INT,
    targets        INT, receptions INT, receiving_yards NUMERIC, receiving_tds INT,
    receiving_air_yards NUMERIC, receiving_yards_after_catch NUMERIC,
    target_share   NUMERIC, air_yards_share NUMERIC,
    known_at       TIMESTAMPTZ NOT NULL,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (gsis_id, game_id)
);
CREATE INDEX IF NOT EXISTS player_game_season_week ON player_game(season, week);

CREATE TABLE IF NOT EXISTS snaps (
    pfr_player_id  TEXT NOT NULL,
    gsis_id        TEXT,                   -- resolved through player_id_map; NULL = unmatched
    game_id        TEXT NOT NULL REFERENCES games(game_id),
    team           TEXT NOT NULL,
    position       TEXT,
    offense_snaps  INT, offense_pct NUMERIC,
    defense_snaps  INT, defense_pct NUMERIC,
    st_snaps       INT, st_pct NUMERIC,
    known_at       TIMESTAMPTZ NOT NULL,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (pfr_player_id, game_id)
);

-- Weekly roster status includes INA (game-day inactives), which is announced
-- ~90 minutes before kickoff, so known_at = kickoff - 90 min.
CREATE TABLE IF NOT EXISTS roster_status (
    gsis_id        TEXT NOT NULL,
    season         INT  NOT NULL,
    week           INT  NOT NULL,
    team           TEXT NOT NULL,
    position       TEXT,
    depth_chart_position TEXT,
    status         TEXT,
    status_abbr    TEXT,
    known_at       TIMESTAMPTZ NOT NULL,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (gsis_id, season, week, team)
);

-- nflverse injury rows carry no report timestamp. The final report for most games
-- is released the day before the eve of game day; known_at = kickoff - 40 hours
-- is a conservative approximation (documented assumption A1).
CREATE TABLE IF NOT EXISTS injury_reports (
    gsis_id        TEXT NOT NULL,
    season         INT  NOT NULL,
    week           INT  NOT NULL,
    game_type      TEXT NOT NULL,
    team           TEXT NOT NULL,
    position       TEXT,
    report_primary_injury TEXT,
    report_status  TEXT,                   -- Out | Doubtful | Questionable | NULL
    practice_status TEXT,
    known_at       TIMESTAMPTZ NOT NULL,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (gsis_id, season, week, game_type, team)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id         BIGSERIAL PRIMARY KEY,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    seasons        INT[] NOT NULL,
    table_hashes   JSONB,                  -- md5 of content excluding loaded_at
    row_counts     JSONB
);

CREATE TABLE IF NOT EXISTS data_quality_runs (
    run_id         BIGINT REFERENCES ingest_runs(run_id),
    check_name     TEXT NOT NULL,
    passed         BOOLEAN NOT NULL,
    detail         TEXT,
    checked_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- Phase 2: odds
-- Raw API responses stay as files in data/odds/raw/; these tables are rebuilt from them.

CREATE TABLE IF NOT EXISTS odds_events (
    event_id       TEXT PRIMARY KEY,
    game_id        TEXT NOT NULL UNIQUE REFERENCES games(game_id),
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per bookmaker x player x market x line at one snapshot.
-- known_at = the API's snapshot timestamp (when these prices were on the board).
CREATE TABLE IF NOT EXISTS prop_lines (
    game_id        TEXT NOT NULL REFERENCES games(game_id),
    snapshot_tag   TEXT NOT NULL,          -- bet | close
    snapshot_ts    TIMESTAMPTZ NOT NULL,
    book           TEXT NOT NULL,
    market         TEXT NOT NULL,          -- player_receptions | player_reception_yds
    player_name    TEXT NOT NULL,          -- as the book wrote it
    gsis_id        TEXT,                   -- NULL = unmatched / ambiguous
    match_status   TEXT NOT NULL,          -- matched | unmatched | ambiguous
    line           NUMERIC NOT NULL,
    over_price     NUMERIC,                -- decimal odds
    under_price    NUMERIC,
    book_update_ts TIMESTAMPTZ,
    known_at       TIMESTAMPTZ NOT NULL,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, snapshot_tag, book, market, player_name, line)
);
CREATE INDEX IF NOT EXISTS prop_lines_player ON prop_lines(gsis_id, game_id);
