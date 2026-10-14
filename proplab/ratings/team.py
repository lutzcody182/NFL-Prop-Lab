"""3b-1: opponent-adjusted team ratings (weighted ridge on play EPA), weekly and point in time.

For each unit (dropbacks, designed runs) and each rating date t (first kickoff of a week):

    EPA_p = a + off[posteam] + def[defteam] + h*home + controls_p . c + e_p
    weights w_p = 0.5 ** (age_weeks / H),  only plays from games that kicked off before t
    ridge penalty lam on the 64 team effects (shrinks them like random effects)

Speed: the weighted normal equations are a time-weighted sum of per-game blocks
(X_g' X_g, X_g' y_g), so each fit sums ~1,000 small matrices instead of re-reading plays.

    python -m proplab.ratings.team tune      # 2018-2024 -> phase3/ratings_team_tuning.json
    python -m proplab.ratings.team test      # 2025 once, 2026 live -> phase3/ratings_team_results.json
    python -m proplab.ratings.team publish   # weekly ratings -> DB table team_ratings
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db

ROOT = Path(__file__).resolve().parents[2]
CANON = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
UNITS = ("pass", "run")
LAMS = (50.0, 200.0, 800.0, 3200.0)
HALF_LIVES = (8.0, 16.0, 32.0)
WINDOW_DAYS = 2 * 365
N_CTRL = 9  # intercept, home, down2, down3, down4, ydstogo, yardline, yardline^2, garbage, q4 -> see controls()


def _q(conn, sql):
    cur = conn.execute(sql)
    return pl.DataFrame(cur.fetchall(), schema=[d.name for d in cur.description], orient="row", infer_schema_length=None)


def load_plays(conn) -> pl.DataFrame:
    df = _q(conn, """
        SELECT p.game_id, g.season, g.week, g.game_type, g.kickoff_utc, g.home_team, g.location,
               p.posteam, p.defteam, p.down, p.ydstogo, p.yardline_100, p.wp, p.qtr,
               p.epa::float AS epa, p.qb_dropback, p.rush
        FROM plays p JOIN games g USING (game_id)
        WHERE p.play_type IN ('pass', 'run') AND p.down IS NOT NULL AND p.epa IS NOT NULL
          AND p.posteam IS NOT NULL AND p.defteam IS NOT NULL""")
    canon = lambda c: pl.col(c).replace(CANON)
    df = df.with_columns(canon("posteam"), canon("defteam"), canon("home_team"),
                         pl.when(pl.col("qb_dropback")).then(pl.lit("pass"))
                           .when(pl.col("rush")).then(pl.lit("run")).otherwise(None).alias("unit"))
    return df.filter(pl.col("unit").is_not_null())


def controls(df: pl.DataFrame) -> np.ndarray:
    home = ((df["posteam"] == df["home_team"]) & (df["location"] != "Neutral")).cast(pl.Float64).to_numpy()
    down = df["down"].to_numpy()
    yl = df["yardline_100"].to_numpy().astype(float) / 100
    wp = df["wp"].fill_null(0.5).to_numpy()
    return np.column_stack([
        np.ones(len(df)), home, (down == 2), (down == 3), (down == 4),
        np.minimum(df["ydstogo"].to_numpy(), 20) / 10, yl, yl ** 2,
        ((wp < 0.1) | (wp > 0.9)), (df["qtr"].to_numpy() >= 4),
    ]).astype(float)


class UnitData:
    """Per-game sufficient statistics for one unit."""

    def __init__(self, plays: pl.DataFrame, unit: str, teams: list[str]):
        self.teams = teams
        idx = {t: i for i, t in enumerate(teams)}
        d = plays.filter(pl.col("unit") == unit).sort("kickoff_utc", "game_id")
        C = controls(d)
        self.n_ctrl = C.shape[1]
        T = len(teams)
        p = self.n_ctrl + 2 * T
        off = np.array([idx[t] for t in d["posteam"].to_list()])
        de = np.array([idx[t] for t in d["defteam"].to_list()])
        y = d["epa"].to_numpy()
        games, starts = np.unique(d["game_id"].to_numpy(), return_index=True)
        order = np.argsort(starts)
        games, starts = games[order], starts[order]
        ends = np.append(starts[1:], len(d))
        self.game_ids = games
        kick = d["kickoff_utc"].to_numpy()
        self.kick = kick[starts].astype("datetime64[s]").astype(np.int64) / 86400.0  # days
        self.A = np.zeros((len(games), p, p))
        self.b = np.zeros((len(games), p))
        self.cbar = np.zeros((len(games), self.n_ctrl))
        self.n = np.zeros(len(games))
        for g, (s, e) in enumerate(zip(starts, ends)):
            X = np.zeros((e - s, p))
            X[:, :self.n_ctrl] = C[s:e]
            X[np.arange(e - s), self.n_ctrl + off[s:e]] = 1
            X[np.arange(e - s), self.n_ctrl + T + de[s:e]] = 1
            self.A[g] = X.T @ X
            self.b[g] = X.T @ y[s:e]
            self.cbar[g] = C[s:e].sum(axis=0)
            self.n[g] = e - s
        # per game & side: mean EPA and plays, for evaluation targets
        gd = (d.group_by("game_id", "posteam", "defteam", "home_team", "location")
               .agg(pl.col("epa").mean().alias("epa"), pl.len().alias("plays")))
        self.targets = gd

    def fit(self, t_days: float, half_life: float, lam: float) -> dict:
        age_weeks = (t_days - self.kick) / 7.0
        m = (age_weeks > 0) & (age_weeks * 7 <= WINDOW_DAYS)
        w = np.where(m, 0.5 ** (age_weeks / half_life), 0.0)
        A = np.tensordot(w, self.A, axes=1)
        b = w @ self.b
        pen = np.full(A.shape[0], 1e-6)
        pen[self.n_ctrl:] = lam
        beta = np.linalg.solve(A + np.diag(pen), b)
        T = len(self.teams)
        ctrl_mean = (w @ self.cbar) / max((w @ self.n), 1e-9)
        return {"beta": beta, "off": beta[self.n_ctrl:self.n_ctrl + T], "def": beta[self.n_ctrl + T:],
                "base": float(ctrl_mean @ beta[:self.n_ctrl]) - float(beta[1] * ctrl_mean[1]),  # league avg, no home
                "home": float(beta[1]), "plays_weight": float(w @ self.n)}

    def raw(self, t_days: float, half_life: float) -> dict:
        """Baseline: recency-weighted raw EPA/play for and against each team."""
        age_weeks = (t_days - self.kick) / 7.0
        m = (age_weeks > 0) & (age_weeks * 7 <= WINDOW_DAYS)
        w = np.where(m, 0.5 ** (age_weeks / half_life), 0.0)
        T = len(self.teams)
        nc = self.n_ctrl
        # b[:, nc+i] = sum EPA with offense i; A[:, nc+i, nc+i] = plays with offense i
        off_sum, off_n = w @ self.b[:, nc:nc + T], w @ self.A[:, np.arange(nc, nc + T), np.arange(nc, nc + T)]
        def_sum, def_n = w @ self.b[:, nc + T:], w @ self.A[:, np.arange(nc + T, nc + 2 * T), np.arange(nc + T, nc + 2 * T)]
        league = (w @ self.b[:, 0]) / max(w @ self.n, 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            off = np.where(off_n > 0, off_sum / off_n, league)
            de = np.where(def_n > 0, def_sum / def_n, league)
        return {"off": off - league, "def": de - league, "league": float(league)}


def rating_dates(conn, seasons) -> pl.DataFrame:
    return _q(conn, f"""
        SELECT season, week, min(kickoff_utc) AS cutoff FROM games
        WHERE game_type = 'REG' AND season IN ({",".join(map(str, seasons))}) GROUP BY season, week ORDER BY 1, 2""")


def days(ts) -> float:
    return np.datetime64(ts.replace(tzinfo=None), "s").astype(np.int64) / 86400.0


def evaluate(conn, ud: UnitData, games: pl.DataFrame, half_life: float, lam: float | None) -> pl.DataFrame:
    """Predict each (game, offense) unit EPA/play; lam=None -> raw baseline."""
    idx = {t: i for i, t in enumerate(ud.teams)}
    tg = games.join(ud.targets, on="game_id")
    out = []
    for (season, week, cutoff), grp in tg.group_by(["season", "week", "cutoff"], maintain_order=True):
        t = days(cutoff)
        if lam is None:
            r = ud.raw(t, half_life)
            pred = [r["league"] + r["off"][idx[o]] + r["def"][idx[d]] for o, d in zip(grp["posteam"], grp["defteam"])]
        else:
            f = ud.fit(t, half_life, lam)
            pred = [f["base"] + f["off"][idx[o]] + f["def"][idx[d]] + f["home"] * float(o == h and loc != "Neutral")
                    for o, d, h, loc in zip(grp["posteam"], grp["defteam"], grp["home_team"], grp["location"])]
        out.append(grp.with_columns(pl.Series("pred", pred, dtype=pl.Float64)))
    return pl.concat(out).with_columns(((pl.col("epa") - pl.col("pred")) ** 2).alias("se"))


def wmse(df: pl.DataFrame) -> float:
    return float((df["se"] * df["plays"]).sum() / df["plays"].sum())


def boot_diff(a: pl.DataFrame, b: pl.DataFrame, rng, n=1000):
    j = a.select("game_id", "posteam", "plays", pl.col("se").alias("sa")).join(
        b.select("game_id", "posteam", pl.col("se").alias("sb")), on=["game_id", "posteam"])
    g = j.group_by("game_id").agg(((pl.col("sa") - pl.col("sb")) * pl.col("plays")).sum().alias("d"), pl.col("plays").sum().alias("w"))
    d, w = g["d"].to_numpy(), g["w"].to_numpy()
    stats = [d[p].sum() / w[p].sum() for p in (rng.integers(0, len(d), len(d)) for _ in range(n))]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(d.sum() / w.sum()), float(lo), float(hi)


def setup(conn):
    plays = load_plays(conn)
    teams = sorted(set(plays["posteam"].unique().to_list()) | set(plays["defteam"].unique().to_list()))
    games = _q(conn, """SELECT game_id, season, week FROM games WHERE game_type = 'REG'""")
    return plays, teams, games


def games_for(conn, games, seasons):
    dates = rating_dates(conn, seasons)
    played = _q(conn, "SELECT game_id FROM game_results")
    return games.filter(pl.col("season").is_in(seasons)).join(dates, on=["season", "week"]).join(played, on="game_id")


def tune() -> None:
    conn = db.connect()
    plays, teams, games = setup(conn)
    g = games_for(conn, games, list(range(2018, 2025)))
    out = {}
    for unit in UNITS:
        ud = UnitData(plays, unit, teams)
        grid = []
        for H, lam in itertools.product(HALF_LIVES, LAMS):
            grid.append({"H": H, "lam": lam, "wmse": wmse(evaluate(conn, ud, g, H, lam))})
        base = [{"H": H, "wmse": wmse(evaluate(conn, ud, g, H, None))} for H in HALF_LIVES]
        best = min(grid, key=lambda r: r["wmse"])
        best_raw = min(base, key=lambda r: r["wmse"])
        out[unit] = {"chosen": best, "baseline_chosen": best_raw, "grid": grid, "baseline_grid": base}
        print(unit, "adjusted", best, "raw", best_raw, flush=True)
    (ROOT / "phase3" / "ratings_team_tuning.json").write_text(json.dumps(out, indent=2))


def chosen() -> dict:
    return json.loads((ROOT / "phase3" / "ratings_team_tuning.json").read_text())


def test() -> None:
    conn = db.connect()
    plays, teams, games = setup(conn)
    ch = chosen()
    rng = np.random.default_rng(25)
    res = {}
    for unit in UNITS:
        ud = UnitData(plays, unit, teams)
        c, b = ch[unit]["chosen"], ch[unit]["baseline_chosen"]
        res[unit] = {}
        for name, seasons in (("tuning_2018_2024", list(range(2018, 2025))), ("test_2025", [2025]), ("live_2026", [2026])):
            g = games_for(conn, games, seasons)
            adj, raw = evaluate(conn, ud, g, c["H"], c["lam"]), evaluate(conn, ud, g, b["H"], None)
            d, lo, hi = boot_diff(adj, raw, rng)
            res[unit][name] = {"team_games": adj.height, "wmse_adjusted": wmse(adj), "wmse_raw": wmse(raw),
                               "diff": d, "ci95": [lo, hi], "pct": d / wmse(raw)}
            print(unit, name, res[unit][name], flush=True)
    res["gate"] = {u: bool(res[u]["test_2025"]["ci95"][1] < 0) for u in UNITS}
    res["gate"]["passed"] = all(res["gate"][u] for u in UNITS)
    (ROOT / "phase3" / "ratings_team_results.json").write_text(json.dumps(res, indent=2))
    print("GATE", res["gate"])


def weekly_ratings(conn, seasons) -> pl.DataFrame:
    """Ratings for every REG week of the seasons (including the next unplayed week)."""
    plays, teams, _ = setup(conn)
    ch = chosen()
    rows = []
    dates = rating_dates(conn, seasons)
    for unit in UNITS:
        ud = UnitData(plays, unit, teams)
        c = ch[unit]["chosen"]
        for season, week, cutoff in dates.iter_rows():
            f = ud.fit(days(cutoff), c["H"], c["lam"])
            for i, t in enumerate(teams):
                rows.append((season, week, t, unit, float(f["off"][i]), float(f["def"][i]), cutoff))
    return pl.DataFrame(rows, schema=["season", "week", "team", "unit", "off", "def", "as_of"], orient="row")


def publish() -> None:
    conn = db.connect()
    r = weekly_ratings(conn, list(range(2017, 2027)))
    conn.execute("""CREATE TABLE IF NOT EXISTS team_ratings (
        season INT, week INT, team TEXT, unit TEXT, off DOUBLE PRECISION, def DOUBLE PRECISION,
        as_of TIMESTAMPTZ, PRIMARY KEY (season, week, team, unit))""")
    with conn.transaction():
        conn.execute("DELETE FROM team_ratings")
        with conn.cursor().copy("COPY team_ratings (season, week, team, unit, off, def, as_of) FROM STDIN") as cp:
            for row in r.iter_rows():
                cp.write_row(row)
    conn.commit()
    print("team_ratings rows:", r.height)


if __name__ == "__main__":
    {"tune": tune, "test": test, "publish": publish}[sys.argv[1]]()
