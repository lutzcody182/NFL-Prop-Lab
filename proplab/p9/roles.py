"""Phase 9A: snap-share role signal (see phase9/PREREGISTRATION.md, section A).

role_r = shrunk short-term offensive snap share (half-life 2 games) / long-term snap share (half-life 8),
over the player's EARLIER games only; 2 pseudo-games at the long-term share; 1.0 without history.

    python -m proplab.p9.roles decide    # walk-forward 2019-2024 -> phase9/roles_ablation.json
    python -m proplab.p9.roles confirm   # 2025, 2026 weeks 1-6, pooled
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase9" / "roles_ablation.json"
SEASONS = list(range(2019, 2025))
MARKETS = ["wrte_rec", "wrte_rec_yds", "rb_rush_att", "rb_rush_yds", "rb_rec", "rb_rec_yds", "anytime_td"]
KEYS = ["tgt", "car", "wrte_rec", "wrte_rec_yds"]
SEED = 99
PSEUDO = 2.0
R_CLIP = (0.2, 5.0)   # numerical guard only


def role_signal(conn, u: pl.DataFrame) -> pl.DataFrame:
    sn = _q(conn, "SELECT gsis_id, game_id, offense_pct::float AS pct FROM snaps WHERE gsis_id IS NOT NULL AND offense_pct IS NOT NULL") \
        .unique(subset=["gsis_id", "game_id"])
    j = u.select("gsis_id", "game_id", "kickoff_utc", "team").join(sn, on=["gsis_id", "game_id"], how="left") \
         .sort("gsis_id", "kickoff_utc", "team")
    has = j["pct"].is_not_null().to_numpy().astype(float)
    pct = j["pct"].fill_null(0.0).to_numpy()
    k = j["gsis_id"].to_numpy()
    s2, n2 = ewm_prior_sums(k, pct * has, decay_for(2.0)), ewm_prior_sums(k, has, decay_for(2.0))
    s8, n8 = ewm_prior_sums(k, pct * has, decay_for(8.0)), ewm_prior_sums(k, has, decay_for(8.0))
    long = np.divide(s8, n8, out=np.zeros_like(s8), where=n8 > 0)
    short = (s2 + PSEUDO * long) / (n2 + PSEUDO)
    r = np.divide(short, long, out=np.ones_like(short), where=long > 0.01)
    return j.select("gsis_id", "game_id").with_columns(pl.Series("role_r", np.clip(r, *R_CLIP)))


def setup():
    conn = db.connect()
    D = E.load(conn)
    u = D["u"].join(role_signal(conn, D["u"]), on=["gsis_id", "game_id"], how="left").with_columns(pl.col("role_r").fill_null(1.0))
    return {"u": u, "tg": D["tg"], "pools": D["pools"]}


class _Pooled:
    def __init__(self, runner):
        self.r = runner

    def pooled(self, market, cfgs, seasons, weeks=None):
        a = self.r.pooled(market, cfgs, [2025])
        b = self.r.pooled(market, cfgs, [2026], (1, 6))
        return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


def decide() -> None:
    runner = Runner(setup())
    rng = np.random.default_rng(SEED)
    base = base_configs()
    trials = []
    for g in (0.5, 1.0):
        res = compare(runner, MARKETS, base, with_group(base, KEYS, "g_role", g), SEASONS, rng=rng)
        trials.append({"value": g, **res})
        print(f"role g={g}: {res['change']:+.4%} CI [{res['ci95'][0]:+.4%}, {res['ci95'][1]:+.4%}] "
              f"{ {m: f'{x:+.3%}' for m, x in res['by_market'].items()} }", flush=True)
    best = min(trials, key=lambda t: t["change"])
    survived = best["ci95"][1] < 0
    final = with_group(base, KEYS, "g_role", best["value"]) if survived else base
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"seasons": SEASONS, "trials": trials, "best": best["value"], "survived": survived,
                               "final_configs": {k: asdict(c) for k, c in final.items()}}, indent=2, default=str))
    print("SURVIVED" if survived else "did not survive")


def confirm() -> None:
    out = json.loads(OUT.read_text())
    if not out["survived"]:
        out["adopted"] = False
        OUT.write_text(json.dumps(out, indent=2, default=str))
        print("did not survive development; nothing to confirm")
        return
    runner = Runner(setup())
    rng = np.random.default_rng(SEED + 1)
    base = base_configs()
    final = {k: E.Config(**v) for k, v in out["final_configs"].items()}
    live = compare(runner, MARKETS, base, final, [2026], weeks=(1, 6), rng=rng)
    second = compare(runner, MARKETS, base, final, [2025], rng=rng)
    pooled = compare(_Pooled(runner), MARKETS, base, final, ["pooled"], rng=rng)
    out["confirmation"] = {"live_2026_wk1_6": live, "second_look_2025": second, "pooled": pooled}
    out["adopted"] = live["change"] < 0 and second["change"] < 0 and pooled["ci95"][1] < 0
    OUT.write_text(json.dumps(out, indent=2, default=str))
    for k, v in out["confirmation"].items():
        print(f"{k}: {v['change']:+.4%} CI [{v['ci95'][0]:+.4%}, {v['ci95'][1]:+.4%}] { {m: f'{x:+.3%}' for m, x in v['by_market'].items()} }")
    print("ADOPTED" if out["adopted"] else "NOT ADOPTED")


if __name__ == "__main__":
    {"decide": decide, "confirm": confirm}[sys.argv[1]]()
