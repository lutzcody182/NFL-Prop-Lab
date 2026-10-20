"""Weekly and season-to-date leaders, offense and defense, players and teams.

    python -m proplab.live.leaders     # -> phase5/leaders.json (also built by proplab.live.export)

Only games whose stats are published (known_at <= now) are counted.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl

from proplab import db
from proplab.p3.matchup import _q

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase5" / "leaders.json"
TOP = 10

# key -> (label, expression over player_game sums, decimals, positions or None)
PLAYER_OFF = {
    "pass_yds": ("Passing yards", pl.col("passing_yards"), 0, None),
    "pass_td": ("Passing TDs", pl.col("passing_tds"), 0, None),
    "rush_yds": ("Rushing yards", pl.col("rushing_yards"), 0, None),
    "rush_td": ("Rushing TDs", pl.col("rushing_tds"), 0, None),
    "rec_yds": ("Receiving yards", pl.col("receiving_yards"), 0, None),
    "rec": ("Receptions", pl.col("receptions"), 0, None),
    "rec_td": ("Receiving TDs", pl.col("receiving_tds"), 0, None),
    "scrim": ("Scrimmage yards", pl.col("rushing_yards") + pl.col("receiving_yards"), 0, None),
    "tds": ("Total TDs", pl.col("rushing_tds") + pl.col("receiving_tds"), 0, None),
    "k_pts": ("Kicker points", 3 * pl.col("fg_made") + pl.col("pat_made"), 0, None),
}
PLAYER_DEF = {
    "tackles": ("Tackles", pl.col("def_tackles_solo") + pl.col("def_tackle_assists"), 0, None),
    "sacks": ("Sacks", pl.col("def_sacks"), 1, None),
    "qb_hits": ("QB hits", pl.col("def_qb_hits"), 0, None),
    "tfl": ("Tackles for loss", pl.col("def_tackles_for_loss"), 1, None),
    "ints": ("Interceptions", pl.col("def_interceptions"), 0, None),
    "pd": ("Passes defended", pl.col("def_pass_defended"), 0, None),
    "ff": ("Forced fumbles", pl.col("def_fumbles_forced"), 0, None),
}
NUM = ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds", "receiving_yards", "receptions", "receiving_tds",
       "fg_made", "pat_made", "def_tackles_solo", "def_tackle_assists", "def_sacks", "def_qb_hits", "def_tackles_for_loss",
       "def_interceptions", "def_pass_defended", "def_fumbles_forced", "fumble_recovery_opp"]


def player_games(conn, season: int, now) -> pl.DataFrame:
    cols = ", ".join(f"coalesce(pg.{c}, 0)::float AS {c}" for c in NUM)
    return _q(conn, f"""
        SELECT pg.gsis_id, coalesce(p.display_name, pg.gsis_id) AS name, pg.position AS pos, pg.team, pg.opponent AS opp,
               g.week, g.game_id, {cols}
        FROM player_game pg JOIN games g USING (game_id) LEFT JOIN players p ON p.gsis_id = pg.gsis_id
        WHERE g.season = {season} AND g.game_type = 'REG' AND pg.known_at <= '{now.isoformat()}'""")


def team_games(conn, season: int, now) -> pl.DataFrame:
    t = _q(conn, f"""
        SELECT g.game_id, g.week, s.team, s.opp,
               CASE WHEN s.team = g.home_team THEN r.home_score ELSE r.away_score END AS pts,
               CASE WHEN s.team = g.home_team THEN r.away_score ELSE r.home_score END AS pts_allowed
        FROM games g CROSS JOIN LATERAL (VALUES (g.home_team, g.away_team), (g.away_team, g.home_team)) s(team, opp)
        JOIN game_results r USING (game_id)
        WHERE g.season = {season} AND g.game_type = 'REG' AND r.known_at <= '{now.isoformat()}'""")
    pl_ = _q(conn, f"""
        SELECT game_id, posteam AS team, sum(yards_gained)::float AS yds, avg(epa)::float AS epa, count(*) AS plays,
               sum(CASE WHEN sack THEN 1 ELSE 0 END) AS sacks_taken, sum(CASE WHEN interception THEN 1 ELSE 0 END) AS ints_thrown
        FROM plays WHERE season = {season} AND play_type IN ('pass', 'run') AND known_at <= '{now.isoformat()}'
        GROUP BY 1, 2""")
    fr = _q(conn, f"""SELECT game_id, team, sum(coalesce(fumble_recovery_opp, 0)) AS fum_rec
                      FROM player_game WHERE season = {season} AND known_at <= '{now.isoformat()}' GROUP BY 1, 2""")
    t = t.join(pl_, on=["game_id", "team"], how="inner")   # only games whose play-by-play is published
    opp = pl_.rename({"team": "opp", "yds": "yds_allowed", "epa": "epa_allowed", "plays": "plays_faced",
                      "sacks_taken": "sacks", "ints_thrown": "ints"})
    return t.join(opp, on=["game_id", "opp"], how="inner").join(fr, on=["game_id", "team"], how="left").fill_null(0) \
            .with_columns((pl.col("ints") + pl.col("fum_rec")).alias("takeaways"))


TEAM_OFF = {"pts": ("Points scored", "pts", 0, False), "yds": ("Total yards", "yds", 0, False),
            "epa": ("EPA per play", "epa", 3, False)}
TEAM_DEF = {"pts_allowed": ("Fewest points allowed", "pts_allowed", 0, True), "yds_allowed": ("Fewest yards allowed", "yds_allowed", 0, True),
            "epa_allowed": ("Lowest EPA per play allowed", "epa_allowed", 3, True), "sacks": ("Sacks", "sacks", 0, False),
            "takeaways": ("Takeaways", "takeaways", 0, False)}


def board_players(df: pl.DataFrame, cats: dict, season_total: bool) -> dict:
    out = {}
    if season_total:
        df = df.group_by("gsis_id").agg(pl.col("name").last(), pl.col("pos").last(), pl.col("team").last(),
                                        pl.len().alias("g"), *[pl.col(c).sum() for c in NUM])
    for key, (label, expr, dec, _) in cats.items():
        r = df.with_columns(expr.alias("v")).filter(pl.col("v") > 0).sort("v", "name", descending=[True, False]).head(TOP)
        out[key] = {"label": label, "dec": dec, "rows": [
            {"id": x["gsis_id"], "name": x["name"], "pos": x["pos"], "team": x["team"], "v": round(float(x["v"]), dec),
             **({"g": x["g"]} if season_total else {"opp": x["opp"]})} for x in r.iter_rows(named=True)]}
    return out


def board_teams(df: pl.DataFrame, cats: dict, season_total: bool) -> dict:
    out = {}
    if season_total:
        df = df.group_by("team").agg(pl.len().alias("g"), pl.col("pts", "pts_allowed", "yds", "yds_allowed", "sacks", "takeaways").sum(),
                                     ((pl.col("epa") * pl.col("plays")).sum() / pl.col("plays").sum()).alias("epa"),
                                     ((pl.col("epa_allowed") * pl.col("plays_faced")).sum() / pl.col("plays_faced").sum()).alias("epa_allowed"))
        per_game = {"pts", "pts_allowed", "yds", "yds_allowed"}
    else:
        per_game = set()
    for key, (label, col, dec, asc) in cats.items():
        expr = (pl.col(col) / pl.col("g")) if col in per_game else pl.col(col)
        r = df.with_columns(expr.alias("v")).sort("v", "team", descending=[not asc, False]).head(TOP)
        lab = label + (" (per game)" if col in per_game else "")
        out[key] = {"label": lab, "dec": 1 if col in per_game else dec, "rows": [
            {"team": x["team"], "v": round(float(x["v"]), 1 if col in per_game else dec), **({"g": x["g"]} if season_total else {"opp": x["opp"]})}
            for x in r.iter_rows(named=True)]}
    return out


def build(conn, season: int = 2026, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    pg, tg = player_games(conn, season, now), team_games(conn, season, now)
    weeks = sorted(set(pg["week"].unique().to_list()) & set(tg["week"].unique().to_list()))
    sched = dict(conn.execute("SELECT week, count(*) FROM games WHERE season = %s AND game_type = 'REG' GROUP BY 1", (season,)).fetchall())
    views = {}
    for w in weeks:
        p, t = pg.filter(pl.col("week") == w), tg.filter(pl.col("week") == w)
        views[str(w)] = {"label": f"Week {w}", "games": int(t["game_id"].n_unique()), "scheduled": int(sched.get(w, 0)),
                         "offense": {**board_players(p, PLAYER_OFF, False), **{f"team_{k}": v for k, v in board_teams(t, TEAM_OFF, False).items()}},
                         "defense": {**board_players(p, PLAYER_DEF, False), **{f"team_{k}": v for k, v in board_teams(t, TEAM_DEF, False).items()}}}
    if weeks:
        views["season"] = {"label": f"{season} season to date", "games": int(tg["game_id"].n_unique()),
                           "offense": {**board_players(pg, PLAYER_OFF, True), **{f"team_{k}": v for k, v in board_teams(tg, TEAM_OFF, True).items()}},
                           "defense": {**board_players(pg, PLAYER_DEF, True), **{f"team_{k}": v for k, v in board_teams(tg, TEAM_DEF, True).items()}}}
    return {"season": season, "weeks": weeks, "views": views, "updated_utc": now.isoformat(timespec="minutes")}


def main() -> None:
    out = build(db.connect())
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"leaders: weeks {out['weeks']}")


if __name__ == "__main__":
    main()
