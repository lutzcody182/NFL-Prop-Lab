"""Phase 7B: weather multiplier on the passing game (see phase7/PREREGISTRATION.md, section B).

m = exp(b_w * max(wind - 10, 0) + b_t * max(40 - temp, 0)) for outdoor games (1.0 indoors or unknown),
applied as sqrt(m) to completion rate and sqrt(m) to yards per completion, so pass yards per attempt move by m.
b_w, b_t for season s are fit on seasons before s: team-game log(pass yards per attempt) regressed on the two
weather terms plus the offense's and the defense's prior yards per attempt (controls), weighted by attempts.

    python -m proplab.p7.weather decide    # walk-forward 2019-2024 -> phase7/weather_ablation.json
    python -m proplab.p7.weather confirm   # 2025, 2026 weeks 1-5, pooled
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from proplab import db
from proplab.p1.data import decay_for, ewm_prior_sums
from proplab.p3 import engine as E
from proplab.p3.ablation import Runner, base_configs, compare, with_group
from proplab.p3.matchup import _q
from proplab.p6.ablation import _Pooled

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase7" / "weather_ablation.json"
SEASONS = list(range(2019, 2025))
SEED = 71
MARKETS = ["qb_pass_yds", "wrte_rec", "wrte_rec_yds"]
KEYS = ["att", "tgt", "wrte_rec", "wrte_rec_yds"]
OUTDOOR = ("outdoors", "open")
HL = 16


def team_games(conn) -> pl.DataFrame:
    return _q(conn, """
        SELECT g.game_id, g.season, g.week, g.kickoff_utc, s.team, s.opp, g.roof,
               r.wind_mph::float AS wind, r.temp_f::float AS temp, (r.game_id IS NOT NULL) AS played,
               coalesce(t.att, 0)::float AS att, coalesce(t.yds, 0)::float AS yds
        FROM games g
        CROSS JOIN LATERAL (VALUES (g.home_team, g.away_team), (g.away_team, g.home_team)) AS s(team, opp)
        LEFT JOIN game_results r USING (game_id)
        LEFT JOIN (SELECT game_id, team, sum(attempts) att, sum(passing_yards) yds FROM player_game GROUP BY 1, 2) t
               ON t.game_id = g.game_id AND t.team = s.team
        WHERE g.game_type = 'REG'""")


def weather_terms(df: pl.DataFrame) -> pl.DataFrame:
    out = pl.col("roof").is_in(OUTDOOR)
    return df.with_columns(
        pl.when(out & pl.col("wind").is_not_null()).then((pl.col("wind") - 10).clip(0)).otherwise(0.0).alias("w_ex"),
        pl.when(out & pl.col("temp").is_not_null()).then((40 - pl.col("temp")).clip(0)).otherwise(0.0).alias("cold"))


def with_priors(t: pl.DataFrame) -> pl.DataFrame:
    """Prior-only EWMA yards and attempts for each offense and each defense (earlier games only)."""
    d = decay_for(HL)
    t = t.sort("team", "kickoff_utc")
    pl_ = t["played"].to_numpy().astype(float)
    k = t["team"].to_numpy()
    t = t.with_columns(pl.Series("o_y", ewm_prior_sums(k, t["yds"].to_numpy() * pl_, d)),
                       pl.Series("o_a", ewm_prior_sums(k, t["att"].to_numpy() * pl_, d)))
    t = t.sort("opp", "kickoff_utc")
    k = t["opp"].to_numpy()
    pl_ = t["played"].to_numpy().astype(float)
    return t.with_columns(pl.Series("d_y", ewm_prior_sums(k, t["yds"].to_numpy() * pl_, d)),
                          pl.Series("d_a", ewm_prior_sums(k, t["att"].to_numpy() * pl_, d)))


def fit(t: pl.DataFrame, season: int) -> tuple[float, float]:
    h = t.filter(pl.col("played") & (pl.col("season") < season) & (pl.col("att") >= 10) & (pl.col("o_a") > 50) & (pl.col("d_a") > 50))
    y = np.log((h["yds"] / h["att"]).clip(0.5).to_numpy())
    X = np.column_stack([np.ones(h.height), np.log((h["o_y"] / h["o_a"]).to_numpy()), np.log((h["d_y"] / h["d_a"]).to_numpy()),
                         h["w_ex"].to_numpy(), h["cold"].to_numpy()])
    w = np.sqrt(h["att"].to_numpy())
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return float(beta[3]), float(beta[4])


def multipliers(conn) -> tuple[pl.DataFrame, dict]:
    t = with_priors(weather_terms(team_games(conn)))
    coefs, parts = {}, []
    for s in sorted(t["season"].unique().to_list()):
        if s < 2017:
            continue
        bw, bt = fit(t, s)
        coefs[s] = {"b_wind": bw, "b_cold": bt}
        parts.append(t.filter(pl.col("season") == s).with_columns(
            (bw * pl.col("w_ex") + bt * pl.col("cold")).exp().alias("wx_m")))
    return pl.concat(parts).select("game_id", "team", "wx_m"), coefs


def setup():
    conn = db.connect()
    D = E.load(conn)
    m, coefs = multipliers(conn)
    u = D["u"].join(m, on=["game_id", "team"], how="left").with_columns(pl.col("wx_m").fill_null(1.0))
    return {"u": u, "tg": D["tg"], "pools": D["pools"]}, coefs


def decide() -> None:
    S, coefs = setup()
    runner = Runner(S)
    rng = np.random.default_rng(SEED)
    base = base_configs()
    res = compare(runner, MARKETS, base, with_group(base, KEYS, "g_wx", 1.0), SEASONS, rng=rng)
    survived = res["ci95"][1] < 0
    print(f"weather: change {res['change']:+.4%} CI [{res['ci95'][0]:+.4%}, {res['ci95'][1]:+.4%}] {res['by_market']}")
    final = with_group(base, KEYS, "g_wx", 1.0) if survived else base
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"seasons": SEASONS, "coefficients": coefs, "development": res, "survived": survived,
                               "final_configs": {k: asdict(c) for k, c in final.items()}}, indent=2, default=str))


def confirm() -> None:
    out = json.loads(OUT.read_text())
    if not out["survived"]:
        out["adopted"] = False
        OUT.write_text(json.dumps(out, indent=2, default=str))
        print("did not survive development; nothing to confirm")
        return
    S, _ = setup()
    runner = Runner(S)
    rng = np.random.default_rng(SEED + 1)
    base = base_configs()
    final = {k: E.Config(**v) for k, v in out["final_configs"].items()}
    live = compare(runner, MARKETS, base, final, [2026], weeks=(1, 5), rng=rng)
    second = compare(runner, MARKETS, base, final, [2025], rng=rng)
    pooled = compare(_Pooled(runner), MARKETS, base, final, ["pooled"], rng=rng)
    out["confirmation"] = {"live_2026_wk1_5": live, "second_look_2025": second, "pooled": pooled}
    out["adopted"] = live["change"] < 0 and second["change"] < 0 and pooled["ci95"][1] < 0
    OUT.write_text(json.dumps(out, indent=2, default=str))
    for k in ("live_2026_wk1_5", "second_look_2025", "pooled"):
        v = out["confirmation"][k]
        print(f"{k}: {v['change']:+.4%} CI [{v['ci95'][0]:+.4%}, {v['ci95'][1]:+.4%}]")
    print("ADOPTED" if out["adopted"] else "NOT ADOPTED")


if __name__ == "__main__":
    {"decide": decide, "confirm": confirm}[sys.argv[1]]()
