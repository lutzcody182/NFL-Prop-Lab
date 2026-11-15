"""Phase 14 group A: chance of playing (phase14/PREREGISTRATION.md).

For a player active in his team's previous game: P(active in the next game), by position and by his offensive
snap share in that previous game, estimated from consecutive team games in the three seasons before the target
season. Used to weight teammates' shares when a team's targets/carries are split (Config.avail).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from proplab.p3.matchup import _q

BINS = [0.0, 0.10, 0.25, 0.50, 0.75]          # snap share 0 | (0,10%] | (10,25%] | (25,50%] | (50,75%] | >75%
LABELS = ["0", "0-10", "10-25", "25-50", "50-75", "75+"]


def _bin(col: pl.Expr) -> pl.Expr:
    e = pl.when(col <= 0).then(pl.lit("0"))
    for lo, hi, lab in zip(BINS[:-1], BINS[1:], LABELS[1:-1]):
        e = e.when((col > lo) & (col <= hi)).then(pl.lit(lab))
    return e.otherwise(pl.lit("75+"))


def pairs(conn, u: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """(gsis_id, team, prev game, next game) for every player active in a team game that has a next game, with his
    snap share in the previous game, the bin, and whether he was active in the next game (played_next)."""
    t = tg.select("game_id", "team", "season", "kickoff_utc").unique(subset=["game_id", "team"]).sort("team", "kickoff_utc")
    t = t.with_columns(pl.col("game_id").shift(-1).over("team").alias("next_game"),
                       pl.col("season").shift(-1).over("team").alias("next_season"))
    act = u.select("gsis_id", "game_id", "team", "pos").unique(subset=["gsis_id", "game_id", "team"])
    sn = _q(conn, "SELECT gsis_id, game_id, offense_pct::float AS pct FROM snaps WHERE gsis_id IS NOT NULL") \
        .unique(subset=["gsis_id", "game_id"])
    p = (act.join(t, on=["game_id", "team"]).filter(pl.col("next_game").is_not_null())
            .join(sn, on=["gsis_id", "game_id"], how="left").with_columns(pl.col("pct").fill_null(0.0)))
    p = p.with_columns(_bin(pl.col("pct")).alias("bin"))
    nxt = act.select("gsis_id", pl.col("game_id").alias("next_game"), "team").with_columns(pl.lit(True).alias("played_next"))
    return p.join(nxt, on=["gsis_id", "next_game", "team"], how="left").with_columns(pl.col("played_next").fill_null(False))


def rates(pr: pl.DataFrame, season: int) -> pl.DataFrame:
    """P(active next game) by position and bin, from the three seasons before `season` (next games in those seasons)."""
    h = pr.filter(pl.col("next_season").is_between(season - 3, season - 1))
    return h.group_by("pos", "bin").agg(((pl.col("played_next").sum() + 1) / (pl.len() + 2)).alias("p_play"), pl.len().alias("n"))


def p_play(pr: pl.DataFrame, season: int) -> pl.DataFrame:
    """(gsis_id, next_game, team) -> p_play for target games in `season` (keyed by the game being projected)."""
    r = rates(pr, season)
    return (pr.filter(pl.col("next_season") == season).join(r.select("pos", "bin", "p_play"), on=["pos", "bin"], how="left")
              .select("gsis_id", pl.col("next_game").alias("game_id"), "team", pl.col("p_play").fill_null(1.0)))
