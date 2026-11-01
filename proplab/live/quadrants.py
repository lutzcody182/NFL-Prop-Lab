"""Team and player numbers for the dashboard's Quadrants tab (descriptive only; nothing here feeds the models).

    python -m proplab.live.quadrants      # -> phase5/quadrants.json (season to date and each team's last 4 games)

Regular-season games with a final score only. Play-level numbers use nflverse play-by-play: plays are runs and
dropbacks (passes, sacks, scrambles) with an EPA value; "neutral" = win probability 20-80% in quarters 1-3.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from proplab import db

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase5" / "quadrants.json"
WINDOWS = {"season": None, "last4": 4}


def rows(conn, sql: str, params: dict) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [d.name for d in cur.description]
    return [{c: (float(v) if hasattr(v, "as_integer_ratio") and not isinstance(v, (int, bool)) else v) for c, v in zip(cols, r)}
            for r in cur.fetchall()]


# games in the window: per team, its last N final regular-season games of the season (or all of them)
GAMES_CTE = """
WITH fin AS (
  SELECT g.game_id, g.kickoff_utc, g.week, t.team, t.opp, t.pf, t.pa
  FROM games g JOIN game_results r USING (game_id)
  CROSS JOIN LATERAL (VALUES (g.home_team, g.away_team, r.home_score, r.away_score),
                             (g.away_team, g.home_team, r.away_score, r.home_score)) AS t(team, opp, pf, pa)
  WHERE g.season = %(season)s AND g.game_type = 'REG'),
win AS (
  SELECT * FROM (SELECT f.*, row_number() OVER (PARTITION BY team ORDER BY kickoff_utc DESC) AS rn FROM fin f) x
  WHERE %(n)s::int IS NULL OR rn <= %(n)s::int),
p AS (
  SELECT pl.*, coalesce(pl.qb_dropback, false) AS db, (coalesce(pl.rush, false) AND NOT coalesce(pl.qb_dropback, false)) AS run, pl.success::int::float AS sr1,
         (pl.wp BETWEEN 0.2 AND 0.8 AND pl.qtr <= 3) AS neutral
  FROM plays pl
  WHERE pl.season = %(season)s AND pl.epa IS NOT NULL AND (pl.qb_dropback OR pl.rush))
"""

TEAM_SQL = GAMES_CTE + """
, off AS (
  SELECT p.posteam AS team, count(*) AS plays, avg(p.epa) AS epa, avg(p.sr1) AS sr,
         avg(p.epa) FILTER (WHERE db) AS pass_epa, avg(p.epa) FILTER (WHERE run) AS rush_epa,
         avg(p.sr1) FILTER (WHERE p.down IN (1, 2)) AS early_sr,
         avg(CASE WHEN (db AND p.yards_gained >= 20) OR (run AND p.yards_gained >= 10) THEN 1.0 ELSE 0.0 END) AS explosive,
         avg(p.yards_gained) AS ypp, avg(p.pass_oe) FILTER (WHERE neutral AND p.pass_oe IS NOT NULL) AS proe,
         avg(CASE WHEN db THEN 1.0 ELSE 0.0 END) FILTER (WHERE neutral) AS neutral_pass_rate,
         avg(CASE WHEN p.sack THEN 1.0 ELSE 0.0 END) FILTER (WHERE db) AS sacked_rate
  FROM p JOIN win w ON w.game_id = p.game_id AND w.team = p.posteam GROUP BY 1),
def AS (
  SELECT p.defteam AS team, avg(p.epa) AS epa, avg(p.sr1) AS sr,
         avg(p.epa) FILTER (WHERE db) AS pass_epa, avg(p.epa) FILTER (WHERE run) AS rush_epa,
         avg(CASE WHEN (db AND p.yards_gained >= 20) OR (run AND p.yards_gained >= 10) THEN 1.0 ELSE 0.0 END) AS explosive,
         avg(p.yards_gained) AS ypp,
         avg(CASE WHEN p.sack OR p.qb_hit THEN 1.0 ELSE 0.0 END) FILTER (WHERE db) AS pressure,
         avg(CASE WHEN f.n_blitzers > 0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE db AND f.n_blitzers IS NOT NULL) AS blitz,
         avg(f.n_defense_box::float) FILTER (WHERE run AND f.n_defense_box IS NOT NULL) AS box_vs_run
  FROM p JOIN win w ON w.game_id = p.game_id AND w.team = p.defteam
  LEFT JOIN ftn_charting f ON f.game_id = p.game_id AND f.play_id = p.play_id GROUP BY 1),
res AS (
  SELECT team, count(*) AS gp, avg(pf) AS pf, avg(pa) AS pa,
         avg(CASE WHEN pf > pa THEN 1.0 WHEN pf = pa THEN 0.5 ELSE 0.0 END) AS win_pct,
         sum(CASE WHEN abs(pf - pa) <= 8 THEN 1 ELSE 0 END) AS close_games,
         avg(CASE WHEN pf > pa THEN 1.0 WHEN pf = pa THEN 0.5 ELSE 0.0 END) FILTER (WHERE abs(pf - pa) <= 8) AS close_win_pct
  FROM win GROUP BY 1)
SELECT res.*, off.plays::float / res.gp AS plays_pg,
       off.epa AS off_epa, off.sr AS off_sr, off.pass_epa AS off_pass_epa, off.rush_epa AS off_rush_epa,
       off.early_sr AS off_early_sr, off.explosive AS off_explosive, off.ypp AS off_ypp, off.proe, off.neutral_pass_rate,
       off.sacked_rate,
       def.epa AS def_epa, def.sr AS def_sr, def.pass_epa AS def_pass_epa, def.rush_epa AS def_rush_epa,
       def.explosive AS def_explosive, def.ypp AS def_ypp, def.pressure, def.blitz, def.box_vs_run
FROM res JOIN off USING (team) JOIN def USING (team) ORDER BY team
"""

# opponents faced in the window (for schedule strength)
OPP_SQL = GAMES_CTE + "SELECT team, opp FROM win"

PLAYER_SQL = GAMES_CTE + """
, pg AS (
  SELECT x.*, s.offense_pct, s.defense_pct, w.kickoff_utc,
         row_number() OVER (PARTITION BY x.gsis_id ORDER BY w.kickoff_utc DESC) AS recent
  FROM player_game x JOIN win w ON w.game_id = x.game_id AND w.team = x.team
  LEFT JOIN snaps s ON s.gsis_id = x.gsis_id AND s.game_id = x.game_id),
tm AS (
  SELECT x.game_id, x.team, sum(coalesce(x.targets, 0)) AS tt, sum(coalesce(x.carries, 0)) AS tc,
         sum(coalesce(x.receiving_air_yards, 0)) FILTER (WHERE coalesce(x.receiving_air_yards, 0) > 0) AS tay
  FROM player_game x JOIN win w ON w.game_id = x.game_id AND w.team = x.team GROUP BY 1, 2),
rec AS (
  SELECT p.receiver_player_id AS gsis_id, avg(p.epa) AS epa_tgt FROM p JOIN win w ON w.game_id = p.game_id AND w.team = p.posteam
  WHERE p.receiver_player_id IS NOT NULL AND p.db GROUP BY 1),
rush AS (
  SELECT p.rusher_player_id AS gsis_id, avg(p.epa) AS epa_car, avg(p.sr1) AS sr_car
  FROM p JOIN win w ON w.game_id = p.game_id AND w.team = p.posteam
  WHERE p.rusher_player_id IS NOT NULL AND p.run GROUP BY 1),
qb AS (
  SELECT coalesce(p.passer_player_id, p.rusher_player_id) AS gsis_id, count(*) AS dropbacks, avg(p.epa) AS epa_db,
         avg(p.cpoe) AS cpoe, avg(p.air_yards) FILTER (WHERE p.air_yards IS NOT NULL AND NOT p.sack) AS adot,
         avg(CASE WHEN p.sack THEN 1.0 ELSE 0.0 END) AS sack_rate, avg(p.sr1) AS sr_db
  FROM p JOIN win w ON w.game_id = p.game_id AND w.team = p.posteam
  WHERE p.db AND coalesce(p.passer_player_id, p.rusher_player_id) IS NOT NULL GROUP BY 1)
SELECT pg.gsis_id, max(pl.display_name) AS name, (array_agg(pg.team ORDER BY pg.kickoff_utc DESC))[1] AS team,
       max(coalesce(pl.position, pg.position)) AS pos,
       count(*) FILTER (WHERE coalesce(pg.offense_pct, 0) > 0 OR coalesce(pg.defense_pct, 0) > 0
                         OR coalesce(pg.attempts, 0) + coalesce(pg.carries, 0) + coalesce(pg.targets, 0) > 0) AS gp,
       sum(coalesce(pg.targets, 0)) AS targets, sum(coalesce(pg.receptions, 0)) AS receptions,
       sum(coalesce(pg.receiving_yards, 0)) AS rec_yds, sum(coalesce(pg.receiving_air_yards, 0)) AS rec_air,
       sum(coalesce(pg.receiving_yards_after_catch, 0)) AS yac, sum(coalesce(pg.receiving_tds, 0)) AS rec_td,
       sum(coalesce(pg.carries, 0)) AS carries, sum(coalesce(pg.rushing_yards, 0)) AS rush_yds, sum(coalesce(pg.rushing_tds, 0)) AS rush_td,
       sum(coalesce(pg.attempts, 0)) AS att, sum(coalesce(pg.completions, 0)) AS cmp, sum(coalesce(pg.passing_yards, 0)) AS pass_yds,
       sum(coalesce(pg.passing_tds, 0)) AS pass_td, sum(coalesce(pg.passing_interceptions, 0)) AS ints,
       sum(tm.tt) AS team_targets, sum(tm.tc) AS team_carries, sum(tm.tay) AS team_air,
       avg(pg.offense_pct) FILTER (WHERE pg.offense_pct > 0) AS snap_pct,
       avg(pg.offense_pct) FILTER (WHERE pg.offense_pct > 0 AND pg.recent <= 3) AS snap_pct_last3,
       avg(pg.defense_pct) FILTER (WHERE pg.defense_pct > 0) AS def_snap_pct,
       sum(coalesce(pg.def_sacks, 0)) AS sacks, sum(coalesce(pg.def_qb_hits, 0)) AS qb_hits,
       sum(coalesce(pg.def_tackles_solo, 0) + coalesce(pg.def_tackle_assists, 0)) AS tackles,
       sum(coalesce(pg.def_tackles_for_loss, 0)) AS tfl, sum(coalesce(pg.def_pass_defended, 0)) AS pd,
       sum(coalesce(pg.def_interceptions, 0)) AS def_int,
       max(rec.epa_tgt) AS epa_tgt, max(rush.epa_car) AS epa_car, max(rush.sr_car) AS sr_car,
       max(qb.dropbacks) AS dropbacks, max(qb.epa_db) AS epa_db, max(qb.cpoe) AS cpoe, max(qb.adot) AS qb_adot,
       max(qb.sack_rate) AS sack_rate, max(qb.sr_db) AS sr_db
FROM pg JOIN tm ON tm.game_id = pg.game_id AND tm.team = pg.team
LEFT JOIN players pl ON pl.gsis_id = pg.gsis_id
LEFT JOIN rec ON rec.gsis_id = pg.gsis_id LEFT JOIN rush ON rush.gsis_id = pg.gsis_id LEFT JOIN qb ON qb.gsis_id = pg.gsis_id
GROUP BY pg.gsis_id
"""

GROUPS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
          "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
          "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB"}


BASE = ["gsis_id", "name", "team", "pos", "gp"]
REC = ["targets", "receptions", "rec_yds", "rec_air", "yac", "rec_td", "team_targets", "team_air", "epa_tgt", "snap_pct", "snap_pct_last3"]
KEEP = {  # (who is worth plotting, fields the charts use)
    "QB": (lambda p: (p["dropbacks"] or 0) >= 30, ["att", "cmp", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "dropbacks",
                                                  "epa_db", "cpoe", "qb_adot", "sack_rate", "sr_db"]),
    "RB": (lambda p: p["carries"] >= 10 or p["targets"] >= 6, ["carries", "rush_yds", "rush_td", "team_carries", "epa_car", "sr_car"] + REC),
    "WR": (lambda p: p["targets"] >= 6, REC), "TE": (lambda p: p["targets"] >= 6, REC),
    **{g: (lambda p: p["tackles"] >= 6 or p["sacks"] + p["qb_hits"] >= 2 or p["pd"] >= 2,
           ["def_snap_pct", "sacks", "qb_hits", "tackles", "tfl", "pd", "def_int"]) for g in ("DL", "LB", "DB")},
}


def r4(x):
    return None if x is None else round(float(x), 4)


def build(conn, season: int, n: int | None) -> dict:
    prm = {"season": season, "n": n}
    teams = rows(conn, TEAM_SQL, prm)
    net = {t["team"]: t["off_epa"] - t["def_epa"] for t in teams}
    opps: dict = {}
    for r in rows(conn, OPP_SQL, prm):
        opps.setdefault(r["team"], []).append(net.get(r["opp"]))
    for t in teams:
        v = [x for x in opps.get(t["team"], []) if x is not None]
        t["sos"] = sum(v) / len(v) if v else None
    players = []
    for p in rows(conn, PLAYER_SQL, prm):
        grp = GROUPS.get((p["pos"] or "").upper())
        if grp is None or not p["gp"] or not KEEP[grp][0](p):
            continue
        players.append({"grp": grp, **{k: p[k] for k in BASE + KEEP[grp][1]}})
    clean = lambda d: {k: (r4(v) if isinstance(v, float) else v) for k, v in d.items()}
    return {"teams": [clean(t) for t in teams], "players": [clean(p) for p in players]}


def main(season: int | None = None) -> dict:
    conn = db.connect()
    if season is None:
        (season,) = conn.execute("""SELECT max(g.season) FROM games g JOIN game_results r USING (game_id)
                                    WHERE g.game_type = 'REG'""").fetchone()
    (week,) = conn.execute("""SELECT max(g.week) FROM games g JOIN game_results r USING (game_id)
                              WHERE g.game_type = 'REG' AND g.season = %s""", (season,)).fetchone()
    out = {"season": season, "through_week": week, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
           "windows": {name: build(conn, season, n) for name, n in WINDOWS.items()}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    s = out["windows"]["season"]
    print(f"quadrants: {season} through week {week}: {len(s['teams'])} teams, {len(s['players'])} players, "
          f"{OUT.stat().st_size / 1024:.0f} KB")
    return out


if __name__ == "__main__":
    main()
