"""Phase 7A: who will play? Expected actives for an unplayed game, and the pre-registered backtest.

Methods (see phase7/PREREGISTRATION.md):
  L  last-game actives (what live projections used through week 6 of 2026)
  I  L minus players listed Out/Doubtful on this week's final injury report, plus players who missed the
     last game but were active within the team's previous 4 games and are not listed Out/Doubtful
  O  the actual game-day actives (posted 90 minutes before kickoff): the ceiling

    python -m proplab.p7.actives backtest     # -> phase7/actives_backtest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.p3 import engine as E
from proplab.p1.model import crps_samples
from proplab.p3.ablation import base_configs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase7" / "actives_backtest.json"
OUT_STATUSES = ("Out", "Doubtful")
RECENT_GAMES = 4
N_SIMS, SEED, BOOT = 1000, 77, 1000
MARKETS = list(E.ALL_MARKETS)


def team_games(tg: pl.DataFrame) -> pl.DataFrame:
    return tg.select("game_id", "season", "week", "kickoff_utc", "team", "played").sort("team", "kickoff_utc")


def injury_out(conn, season: int | None = None) -> pl.DataFrame:
    """(season, week, team, gsis_id) listed Out or Doubtful on the final regular-season report."""
    cur = conn.execute("""SELECT season, week, team, gsis_id FROM injury_reports
                          WHERE game_type = 'REG' AND report_status = ANY(%s)""" + (" AND season = %s" if season else ""),
                       (list(OUT_STATUSES), season) if season else (list(OUT_STATUSES),))
    return pl.DataFrame(cur.fetchall(), schema=["season", "week", "team", "gsis_id"], orient="row")


def expected_actives(u: pl.DataFrame, tg: pl.DataFrame, targets: pl.DataFrame, inj: pl.DataFrame, method: str) -> pl.DataFrame:
    """For each target (game_id, team) -> the players expected active, by method L or I.
    Uses only games before the target's kickoff and (for I) that week's injury report."""
    tgp = team_games(tg).filter(pl.col("played"))
    rows = []
    for t in targets.iter_rows(named=True):
        prev = tgp.filter((pl.col("team") == t["team"]) & (pl.col("kickoff_utc") < t["kickoff_utc"])) \
                  .sort("kickoff_utc", descending=True).head(RECENT_GAMES)["game_id"].to_list()
        if not prev:
            continue
        last = set(u.filter((pl.col("game_id") == prev[0]) & (pl.col("team") == t["team"]))["gsis_id"].to_list())
        chosen = set(last)
        if method == "R":   # exploratory, added after the pre-registered test: removals only
            out = set(inj.filter((pl.col("season") == t["season"]) & (pl.col("week") == t["week"]) & (pl.col("team") == t["team"]))["gsis_id"].to_list())
            chosen = last - out
        if method == "I":
            out = set(inj.filter((pl.col("season") == t["season"]) & (pl.col("week") == t["week"]) & (pl.col("team") == t["team"]))["gsis_id"].to_list())
            recent = set(u.filter(pl.col("game_id").is_in(prev) & (pl.col("team") == t["team"]))["gsis_id"].to_list())
            chosen = (last - out) | (recent - last - out)
        rows += [{"game_id": t["game_id"], "team": t["team"], "gsis_id": g} for g in chosen]
    return pl.DataFrame(rows, schema={"game_id": pl.String, "team": pl.String, "gsis_id": pl.String})


def phantom_features(u: pl.DataFrame, tg: pl.DataFrame, need: pl.DataFrame, half_life: float, dfeat: pl.DataFrame) -> pl.DataFrame:
    """Feature rows for players expected active in a game they did not actually play.
    Each phantom gets its own copy of the player's earlier rows, so it never shifts anyone's later history."""
    info = u.sort("kickoff_utc").group_by("gsis_id").agg(pl.col("name").last(), pl.col("pos").last())
    games = tg.select("game_id", "team", "season", "week", "kickoff_utc", "opp").unique(subset=["game_id", "team"])
    ph = need.join(games, on=["game_id", "team"]).join(info, on="gsis_id", how="left") \
             .with_columns(pl.col("team").alias("_t"))
    if ph.is_empty():
        return ph
    hist = u.join(ph.select(pl.col("gsis_id"), pl.col("game_id").alias("_g"), pl.col("kickoff_utc").alias("_k")), on="gsis_id") \
            .filter(pl.col("kickoff_utc") < pl.col("_k"))
    hist = hist.with_columns((pl.col("gsis_id") + "#" + pl.col("_g")).alias("gsis_id")).drop("_g", "_k")
    zeros = {c: 0 for c in ("targets", "receptions", "rec_tds", "carries", "rush_tds", "attempts", "completions", "pass_tds", "ints")}
    fut = ph.with_columns((pl.col("gsis_id") + "#" + pl.col("game_id")).alias("gsis_id"),
                          *[pl.lit(v).alias(c) for c, v in zeros.items()],
                          *[pl.lit(0.0).alias(c) for c in ("rec_yds", "rush_yds", "pass_yds")],
                          pl.lit(False).alias("played"))
    home = tg.select("game_id", "team").join(
        u.select("game_id", "team", "is_home").unique(subset=["game_id", "team"]), on=["game_id", "team"], how="left")
    fut = fut.join(home, on=["game_id", "team"], how="left").with_columns(pl.col("is_home").fill_null(False))
    chains = pl.concat([hist.select(u.columns), fut.select(u.columns).cast(dict(u.schema))]).sort("gsis_id", "kickoff_utc", "team")
    f = E.features(chains, tg, half_life)
    f = f.filter(~pl.col("played") & pl.col("gsis_id").str.contains("#"))
    dcols = [c for c in f.columns if c.startswith("D_")]
    f = f.drop(dcols).join(dfeat, left_on=["game_id", "opp", "pos"], right_on=["game_id", "defense", "pos"], how="left")
    return f.with_columns(pl.col("gsis_id").str.split("#").list.first().alias("gsis_id"))


def finish(rows: pl.DataFrame, k: E.Constants, cfg: E.Config) -> pl.DataFrame:
    """Recompute the roster-dependent parts (projected starting QB, redistribution) and apply calibration."""
    rows = rows.drop("qb_top", strict=False).with_columns(
        ((pl.col("pos") == "QB") & (pl.col("att_elig") == pl.col("att_elig").filter(pl.col("pos") == "QB")
                                    .max().over("game_id", "team"))).alias("qb_top"))
    p = E.parameters(rows, k, cfg)
    for vt in E.VT:
        fct = pl.col("pos").replace_strict({q: k.calibration.get((vt, q), 1.0) for q in E.POSITIONS},
                                           default=1.0, return_dtype=pl.Float64)
        p = p.with_columns(pl.col(f"{vt}_mu") * fct)
    return p


def draws_for(market, params_by_vt, k_by_vt, rng):
    vt = E.ALL_MARKETS[market][1]
    base_vt = "car" if vt == "both" else vt
    rows, k = params_by_vt[base_vt], k_by_vt[base_vt]
    rows = rows.filter(E.market_eligible(market)).sort("game_id", "gsis_id")
    if vt == "both":
        tp, tk = params_by_vt["tgt"], k_by_vt["tgt"]
        tcols = [c for c in tp.columns if c.startswith("tgt_")]
        rows = rows.drop(tcols).join(tp.select("game_id", "gsis_id", *tcols), on=["game_id", "gsis_id"]).sort("game_id", "gsis_id")
        pick = lambda d: {kk: v for kk, v in d.items() if kk[0] == "tgt"}
        k = E.Constants(prior=k.prior, pools={**k.pools, **tk.pools}, calibration={**k.calibration, **pick(tk.calibration)},
                        dispersion={**k.dispersion, **pick(tk.dispersion)}, exit_rate={**k.exit_rate, **pick(tk.exit_rate)})
    if rows.is_empty():
        return rows, np.zeros((0, N_SIMS))
    return rows, E.market_draws(market, rows, k, N_SIMS, rng)


def backtest() -> None:
    conn = db.connect()
    D = E.load(conn)
    u, tg, pools = D["u"], D["tg"], D["pools"]
    tf = E.team_features(tg)
    inj = injury_out(conn)
    cfgs = base_configs()
    cfg_vt = {vt: cfgs[vt] for vt in ("tgt", "car", "att")}
    wr = cfgs["wrte_rec"]
    targets_all = tg.filter(pl.col("played") & (((pl.col("season") == 2025) & (pl.col("week") <= 18))
                                                | ((pl.col("season") == 2026) & (pl.col("week") <= 5)))) \
                    .select("game_id", "team", "season", "week", "kickoff_utc")
    real = u.select("gsis_id", "game_id", "team")
    per = {m: {meth: [] for meth in "LIOR"} for m in MARKETS}   # rows: (key, game, crps)
    counts = {meth: {"projected": 0, "void": 0, "missed": 0, "played_eligible": 0} for meth in "LIR"}
    feats, dfeats, preps = {}, {}, {}
    for season in (2025, 2026):
        targets = targets_all.filter(pl.col("season") == season)
        sets = {"L": expected_actives(u, tg, targets, inj, "L"), "I": expected_actives(u, tg, targets, inj, "I"),
                "R": expected_actives(u, tg, targets, inj, "R"),
                "O": real.join(targets.select("game_id", "team"), on=["game_id", "team"])}
        need = pl.concat([sets["L"], sets["I"]]).unique().join(real, on=["gsis_id", "game_id", "team"], how="anti")
        for c in {cfg_vt["tgt"], cfg_vt["car"], cfg_vt["att"], wr}:
            if c.half_life not in feats:
                feats[c.half_life] = E.features(u, tg, c.half_life)
                dfeats[c.half_life] = E.defense_features(u, tg)
            key = (season, c)
            if key not in preps:
                preps[key] = E.prepare(feats[c.half_life], tf, pools, season, c)
        ph = {h: phantom_features(u, tg, need, h, dfeats[h]) for h in feats}
        for week in sorted(targets["week"].unique().to_list()):
            wk_games = targets.filter(pl.col("week") == week)
            for meth in "LIOR":
                s = sets[meth].join(wk_games.select("game_id", "team"), on=["game_id", "team"])
                by_vt, k_by_vt = {}, {}
                for vt in ("tgt", "car", "att", "wrte"):
                    c = wr if vt == "wrte" else cfg_vt[vt]
                    f = feats[c.half_life]
                    realrows = f.join(s, on=["gsis_id", "game_id", "team"])
                    phr = ph[c.half_life].join(s, on=["gsis_id", "game_id", "team"]) if not ph[c.half_life].is_empty() else ph[c.half_life]
                    rows = pl.concat([realrows, phr.select(realrows.columns).cast(dict(realrows.schema))]) if phr.height else realrows
                    _, k = preps[(season, c)]
                    by_vt[vt], k_by_vt[vt] = finish(rows, k, c), k
                rng = np.random.default_rng(SEED + season * 100 + week)
                for m in MARKETS:
                    use = dict(by_vt)
                    kk = dict(k_by_vt)
                    if m.startswith("wrte"):
                        use["tgt"], kk["tgt"] = by_vt["wrte"], k_by_vt["wrte"]
                    rows, draws = draws_for(m, use, kk, rng)
                    if rows.is_empty():
                        continue
                    playedmask = rows["played"].to_numpy()
                    y = E.outcome(m, rows)
                    keys = (rows["gsis_id"] + "|" + rows["game_id"]).to_numpy()
                    if playedmask.any():
                        dm, ym = draws[playedmask], y[playedmask]
                        if m == "anytime_td":
                            cr = ((dm > 0.5).mean(axis=1) - (ym > 0.5)) ** 2
                        else:
                            cr = crps_samples(dm, ym)
                        per[m][meth].append(pl.DataFrame({"key": keys[playedmask], "game": rows["game_id"].to_numpy()[playedmask],
                                                          "crps": cr}))
                    if meth in "LIR":
                        counts[meth]["projected"] += len(rows)
                        counts[meth]["void"] += int((~playedmask).sum())
            print(f"{season} week {week} done", flush=True)
    # missed = played-and-eligible under O but not projected by the method
    res = {"markets": {}, "counts": counts}
    rng = np.random.default_rng(SEED)
    for meth in "LIR":
        miss = tot = 0
        for m in MARKETS:
            o = pl.concat(per[m]["O"]) if per[m]["O"] else None
            x = pl.concat(per[m][meth]) if per[m][meth] else None
            if o is None:
                continue
            tot += o.height
            miss += o.height - (o.join(x, on="key").height if x is not None else 0)
        counts[meth]["missed"], counts[meth]["played_eligible"] = miss, tot
    for comp in ("I", "O", "R"):
        stats_in = {}
        for m in MARKETS:
            a, b = pl.concat(per[m]["L"]), pl.concat(per[m][comp])
            j = a.join(b, on="key", suffix="_b")
            stats_in[m] = j
        res[f"{comp}_vs_L"] = compare_rows(stats_in, rng)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "markets"}, indent=1, default=str))


def compare_rows(joined: dict, rng) -> dict:
    """Mean over markets of (method CRPS / L CRPS - 1) on common rows, bootstrap over games."""
    by_m = {m: float(j["crps_b"].mean() / j["crps"].mean() - 1) for m, j in joined.items() if j.height}
    games = np.unique(np.concatenate([j["game"].to_numpy() for j in joined.values() if j.height]))
    sums = {}
    for m, j in joined.items():
        if not j.height:
            continue
        gi = np.searchsorted(games, j["game"].to_numpy())
        sums[m] = (np.bincount(gi, weights=j["crps"].to_numpy(), minlength=len(games)),
                   np.bincount(gi, weights=j["crps_b"].to_numpy(), minlength=len(games)))
    stats = []
    for _ in range(BOOT):
        w = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))
        stats.append(np.mean([(s[1] @ w) / (s[0] @ w) - 1 for s in sums.values()]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"change": float(np.mean(list(by_m.values()))), "ci95": [float(lo), float(hi)], "by_market": by_m,
            "rows": {m: int(j.height) for m, j in joined.items()}}


if __name__ == "__main__":
    {"backtest": backtest}[sys.argv[1]]()
