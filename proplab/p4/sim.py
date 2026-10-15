"""Phase 4: correlated simulation of each team's passing game (see phase4/PREREGISTRATION.md).

    python -m proplab.p4.sim tune       # sigma grid on 2022-2024 -> phase4/tuning.json
    python -m proplab.p4.sim test       # 2025 once + 2026 wks 1-5 -> phase4/results.json
"""
from __future__ import annotations

import json
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import gammaln
from scipy.stats import nbinom, norm

from proplab import db
from proplab.p1.model import fit_dispersion
from proplab.p3 import engine as E
from proplab.p3.ablation import WRTE_BASE
from proplab.p3.run import load_chosen

ROOT = Path(__file__).resolve().parents[2]
SIGMAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
KAPPAS = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
EXIT_MAX = E.EXIT_CUTOFF
N_SIMS = 2000
SEED = 404
TOP_RECEIVERS = 3


# ---------------------------------------------------------------- inputs per season

def season_inputs(D, feat, tf, season: int) -> dict:
    """Player parameters for every team-game of `season` from the LOCKED models, fit on earlier seasons."""
    chosen = load_chosen()
    cfgs = {"att": chosen["att"], "rb": chosen["tgt"], "wr": WRTE_BASE}
    preps = {k: E.prepare(feat[c.half_life], tf, D["pools"], season, c) for k, c in cfgs.items()}
    s = pl.col("season") == season
    pa, ka = preps["att"]
    team = (pa.filter(s).group_by("game_id", "team").agg(
        pl.col("opp").first(), pl.col("is_home").first(), pl.col("week").first(), pl.col("played").first(),
        pl.col("att_team_hat").first().alias("att_hat"), pl.col("team_attempts").first(),
        pl.col("team_targets").first()))
    cols = ["game_id", "team", "gsis_id", "name", "pos", "week", "played", "tgt_mu", "tgt_rate", "tgt_ypu", "tgt_tdr",
            "targets", "receptions", "rec_yds", "rec_tds"]
    wr = preps["wr"][0].filter(s & pl.col("pos").is_in(["WR", "TE"])).select(cols)
    rb = preps["rb"][0].filter(s & (pl.col("pos") == "RB")).select(cols)
    qb = pa.filter(s & pl.col("qb_top")).select("game_id", "team", "gsis_id", "name", "week", "played", "att_intr",
                                                 "attempts", "completions", "pass_yds", "pass_tds", "ints")
    kw = preps["wr"][1]
    return {"season": season, "team": team, "recv": pl.concat([wr, rb]).sort("game_id", "team", "gsis_id"), "qb": qb,
            "pools": kw.pools, "exit": ka.exit_rate.get(("att", "QB"), 0.0),
            "other": {"rate": kw.prior.get(("tgt", "WR", "rate"), 0.62), "ypu": kw.prior.get(("tgt", "WR", "ypu"), 12.0),
                      "tdr": kw.prior.get(("tgt", "WR", "tdr"), 0.06)}}


@dataclass
class Fit:
    calib: float
    r: float
    rho: float
    tau: float
    kappa: float
    scales: list  # fixed per-component scales for the variogram score (from earlier seasons)


def component_outcomes(recv: pl.DataFrame, qb: dict) -> list[float]:
    top = np.argsort(-recv["tgt_mu"].to_numpy())[:TOP_RECEIVERS]
    return ([qb["pass_yds"], qb["completions"]] + [recv["rec_yds"][int(i)] for i in top]
            + [recv["receptions"][int(i)] for i in top])


def alloc_probs(recv_tg: pl.DataFrame, att_mean: float, tau: float) -> np.ndarray:
    """Mean target shares (receivers..., other) that reproduce each receiver's expected targets."""
    mu = recv_tg["tgt_mu"].to_numpy().astype(float)
    p = mu / max(tau * att_mean, 1e-9)
    if p.sum() > 0.995:
        p = p * 0.995 / p.sum()
    return np.append(p, 1 - p.sum())


def fit_params(inputs: list[dict]) -> Fit:
    """Fit the simulator's pieces on earlier seasons (each season's rows predicted out of sample)."""
    team = pl.concat([i["team"] for i in inputs]).filter(pl.col("played") & (pl.col("team_attempts") > 0))
    calib = float(team["team_attempts"].sum() / team["att_hat"].sum())
    mu = team["att_hat"].to_numpy() * calib
    y = team["team_attempts"].to_numpy().astype(float)
    r = fit_dispersion(mu, y)
    z = team.with_columns(pl.Series("z", (y - mu) / np.sqrt(mu + mu ** 2 / r)))
    pairs = z.filter(pl.col("is_home")).select("game_id", pl.col("z").alias("zh")).join(
        z.filter(~pl.col("is_home")).select("game_id", pl.col("z").alias("za")), on="game_id")
    rho = float(np.corrcoef(pairs["zh"], pairs["za"])[0, 1])
    tau = float(team["team_targets"].sum() / team["team_attempts"].sum())
    # Dirichlet-multinomial concentration by maximum likelihood
    ll = {k: 0.0 for k in KAPPAS}
    for inp in inputs:
        recv = inp["recv"].filter(pl.col("played"))
        tm = inp["team"].filter(pl.col("played") & (pl.col("team_targets") > 0))
        att = dict(zip(zip(tm["game_id"], tm["team"]), tm["att_hat"].to_numpy() * calib))
        tt = dict(zip(zip(tm["game_id"], tm["team"]), tm["team_targets"].to_numpy()))
        for (g, t), grp in recv.group_by(["game_id", "team"]):
            if (g, t) not in att:
                continue
            p = alloc_probs(grp, att[(g, t)], tau)
            n = grp["targets"].to_numpy().astype(float)
            T = float(tt[(g, t)])
            counts = np.append(n, max(T - n.sum(), 0.0))
            T = counts.sum()
            for k in KAPPAS:
                a = k * p
                ll[k] += gammaln(k) - gammaln(T + k) + float(np.sum(gammaln(counts + a) - gammaln(a)))
    kappa = max(ll, key=ll.get)
    # fixed scales: standard deviation of each scored component in the earlier seasons
    vecs = []
    for inp in inputs:
        recv, qbs = inp["recv"].filter(pl.col("played")), inp["qb"].filter(pl.col("played"))
        for q in qbs.iter_rows(named=True):
            grp = recv.filter((pl.col("game_id") == q["game_id"]) & (pl.col("team") == q["team"]))
            if grp.height >= TOP_RECEIVERS:
                vecs.append(component_outcomes(grp, q))
    scales = np.array(vecs, dtype=float).std(axis=0).tolist()
    return Fit(calib, r, rho, tau, kappa, scales)


# ---------------------------------------------------------------- simulation

def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def _sum_yards(counts: np.ndarray, pool: np.ndarray, scale: np.ndarray, rng) -> np.ndarray:
    total = int(counts.sum())
    if total == 0:
        return np.zeros(len(counts), dtype=np.float32)
    draws = rng.choice(pool, size=total) * np.repeat(scale, counts)
    cs = np.concatenate([[0.0], np.cumsum(draws, dtype=np.float64)])
    ends = np.cumsum(counts)
    return (cs[ends] - cs[ends - counts]).astype(np.float32)


def simulate_team(rng, A: np.ndarray, fit: Fit, sigma: float, recv: pl.DataFrame, qb: dict | None,
                  inp: dict, att_mean: float) -> dict:
    n = len(A)
    exit_ = rng.uniform(size=n) < inp["exit"]
    f = np.where(exit_, rng.uniform(0, EXIT_MAX, size=n), 1.0)
    T = rng.binomial(A, fit.tau)
    p = alloc_probs(recv, att_mean, fit.tau)
    g = rng.gamma(np.maximum(fit.kappa * p, 1e-6)[None, :], size=(n, len(p)))
    q = g / g.sum(axis=1, keepdims=True)
    # multinomial(T, q) row by row via conditional binomials
    targets = np.zeros((n, len(p)), dtype=np.int64)
    rem, qrem = T.copy(), np.ones(n)
    for i in range(len(p) - 1):
        pi = np.clip(q[:, i] / np.maximum(qrem, 1e-12), 0, 1)
        targets[:, i] = rng.binomial(rem, pi)
        rem -= targets[:, i]
        qrem -= q[:, i]
    targets[:, -1] = rem
    eps = rng.normal(0, sigma, size=n) if sigma > 0 else np.zeros(n)
    rates = np.append(recv["tgt_rate"].to_numpy(), inp["other"]["rate"])
    ypus = np.append(recv["tgt_ypu"].to_numpy(), inp["other"]["ypu"])
    tdrs = np.append(recv["tgt_tdr"].to_numpy(), inp["other"]["tdr"])
    pos = recv["pos"].to_list() + ["WR"]
    logit = np.log(np.clip(rates, 1e-4, 1 - 1e-4) / (1 - np.clip(rates, 1e-4, 1 - 1e-4)))
    catches = np.zeros_like(targets)
    yards = np.zeros(targets.shape, dtype=np.float32)
    tds = np.zeros_like(targets)
    for i in range(len(p)):
        pc = 1 / (1 + np.exp(-(logit[i] + eps)))
        catches[:, i] = rng.binomial(targets[:, i], pc)
        pool = inp["pools"].get(("catch", pos[i]), inp["pools"].get(("catch", "WR")))
        yards[:, i] = _sum_yards(catches[:, i], pool, ypus[i] * np.exp(eps), rng)
        tds[:, i] = rng.binomial(catches[:, i], np.clip(tdrs[i], 0, 1))
    out = {"recv": {"targets": targets[:, :-1], "rec": catches[:, :-1], "yds": yards[:, :-1], "td": tds[:, :-1]}}
    if qb is not None:
        # The starter throws a random subset of the team's attempts (all of them unless he exits).
        # Which catches and TDs came from his throws follows a multivariate hypergeometric draw,
        # so his completions can never exceed his attempts.
        qatt = rng.binomial(A, f)
        thrown = np.zeros_like(catches)
        a_rem, s_rem = A.astype(np.int64).copy(), qatt.astype(np.int64).copy()
        for i in range(catches.shape[1]):
            good = catches[:, i]
            thrown[:, i] = rng.hypergeometric(good, np.maximum(a_rem - good, 0), np.minimum(s_rem, a_rem))
            a_rem -= good
            s_rem -= thrown[:, i]
        td_thrown = rng.hypergeometric(tds, np.maximum(catches - tds, 0), thrown)
        frac = np.divide(thrown, catches, out=np.zeros(catches.shape), where=catches > 0)
        out["qb"] = {"att": qatt, "comp": thrown.sum(axis=1), "yds": (yards * frac).sum(axis=1),
                     "td": td_thrown.sum(axis=1), "int": rng.binomial(qatt, np.clip(qb["att_intr"], 0, 1))}
    return out


def simulate_season(inp: dict, fit: Fit, sigma: float, n: int = N_SIMS, weeks=None, only_played=True) -> dict:
    team = inp["team"]
    if only_played:
        team = team.filter(pl.col("played"))
    if weeks:
        team = team.filter(pl.col("week").is_between(*weeks))
    recv_all, qb_all = inp["recv"], inp["qb"]
    out = {}
    for gid in sorted(team["game_id"].unique().to_list()):
        gt = team.filter(pl.col("game_id") == gid).sort("is_home")
        rng = np.random.default_rng(_seed(SEED, gid))
        z1 = rng.normal(size=n)
        z2 = fit.rho * z1 + np.sqrt(max(1 - fit.rho ** 2, 0)) * rng.normal(size=n)
        for zi, row in zip((z1, z2), gt.iter_rows(named=True)):
            mu = max(row["att_hat"] * fit.calib, 1.0)
            A = nbinom.ppf(norm.cdf(zi), fit.r, fit.r / (fit.r + mu)).astype(np.int64)
            recv = recv_all.filter((pl.col("game_id") == gid) & (pl.col("team") == row["team"])).sort("gsis_id")
            qrow = qb_all.filter((pl.col("game_id") == gid) & (pl.col("team") == row["team"]))
            qb = qrow.row(0, named=True) if qrow.height else None
            sim = simulate_team(rng, A, fit, sigma, recv, qb, inp, mu)
            out[(gid, row["team"])] = {"sim": sim, "recv": recv, "qb": qb, "A": A}
    return out


# ---------------------------------------------------------------- scoring

def vs05(draws: np.ndarray, y: np.ndarray) -> float:
    """Variogram score of order 0.5 for one vector (draws: n x d, y: d)."""
    d = len(y)
    s = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            s += (abs(y[i] - y[j]) ** 0.5 - np.mean(np.abs(draws[:, i] - draws[:, j]) ** 0.5)) ** 2
    return float(s)


def joint_rows(sims: dict, rng, scales: list) -> list[dict]:
    """Per team-game: variogram score (joint vs shuffled) and the two-leg event.
    Components are divided by FIXED scales from earlier seasons, so scores are
    comparable across simulator settings."""
    rows = []
    sd = np.array(scales, dtype=float)
    for (gid, team), v in sims.items():
        qb, recv, sim = v["qb"], v["recv"], v["sim"]
        if qb is None or recv.height < TOP_RECEIVERS:
            continue
        top = np.argsort(-recv["tgt_mu"].to_numpy())[:TOP_RECEIVERS]
        X = [sim["qb"]["yds"], sim["qb"]["comp"]] + [sim["recv"]["yds"][:, i] for i in top] + [sim["recv"]["rec"][:, i] for i in top]
        X = np.column_stack(X).astype(float)
        y = np.array(component_outcomes(recv, qb), dtype=float)
        Xs, ys = X / sd, y / sd
        Xsh = np.column_stack([rng.permutation(Xs[:, j]) for j in range(Xs.shape[1])])
        # two-leg event: QB yards and top receiver yards both above their own simulated medians
        q, w = X[:, 0], X[:, 2]
        mq, mw = np.median(q), np.median(w)
        p_joint = float(np.mean((q > mq) & (w > mw)))
        p_ind = float(np.mean(q > mq) * np.mean(w > mw))
        hit = float((y[0] > mq) and (y[2] > mw))
        rows.append({"game_id": gid, "team": team, "vs_joint": vs05(Xs, ys), "vs_shuffled": vs05(Xsh, ys),
                     "p_joint": p_joint, "p_ind": p_ind, "hit": hit})
    return rows


def boot(diff: np.ndarray, games: np.ndarray, rng, n=1000):
    u, inv = np.unique(games, return_inverse=True)
    s, c = np.bincount(inv, weights=diff), np.bincount(inv)
    st = [s[p].sum() / c[p].sum() for p in (rng.integers(0, len(u), len(u)) for _ in range(n))]
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def summarize(rows: list[dict], rng) -> dict:
    df = pl.DataFrame(rows)
    g = df["game_id"].to_numpy()
    vs = boot(df["vs_joint"].to_numpy() - df["vs_shuffled"].to_numpy(), g, rng)
    bj = (df["p_joint"] - df["hit"]) ** 2
    bi = (df["p_ind"] - df["hit"]) ** 2
    br = boot((bj - bi).to_numpy(), g, rng)
    from proplab.p3.engine import ece_null95
    from proplab.p1.model import ece
    pj, hit = df["p_joint"].to_numpy(), df["hit"].to_numpy()
    return {"team_games": df.height, "vs_joint": float(df["vs_joint"].mean()), "vs_shuffled": float(df["vs_shuffled"].mean()),
            "vs_diff": vs[0], "vs_ci95": [vs[1], vs[2]], "vs_pct": vs[0] / float(df["vs_shuffled"].mean()),
            "brier_joint": float(bj.mean()), "brier_ind": float(bi.mean()), "brier_diff": br[0], "brier_ci95": [br[1], br[2]],
            "two_leg_rate_actual": float(hit.mean()), "two_leg_pred_joint": float(pj.mean()),
            "two_leg_pred_ind": float(df["p_ind"].mean()), "ece_joint": ece(pj, hit), "ece_null95": ece_null95(pj, rng)}


# ---------------------------------------------------------------- runners

def setup():
    conn = db.connect()
    D = E.load(conn)
    chosen = load_chosen()
    feat = {h: E.features(D["u"], D["tg"], h) for h in {c.half_life for c in [*chosen.values(), WRTE_BASE]}}
    return D, feat, E.team_features(D["tg"])


def inputs_for(D, feat, tf, seasons) -> dict:
    return {s: season_inputs(D, feat, tf, s) for s in seasons}


def tune() -> None:
    D, feat, tf = setup()
    inps = inputs_for(D, feat, tf, range(2019, 2025))
    fits = {s: fit_params([inps[s - 3], inps[s - 2], inps[s - 1]]) for s in (2022, 2023, 2024)}
    rng = np.random.default_rng(SEED)
    grid = []
    for sigma in SIGMAS:
        rows = []
        for s in (2022, 2023, 2024):
            rows += joint_rows(simulate_season(inps[s], fits[s], sigma, n=1000), rng, fits[s].scales)
        r = summarize(rows, rng)
        grid.append({"sigma": sigma, **{k: r[k] for k in ("vs_joint", "vs_shuffled", "vs_pct", "brier_joint", "brier_ind", "ece_joint")}})
        print(grid[-1], flush=True)
    best = min(grid, key=lambda g: g["vs_joint"])
    out = {"sigma": best["sigma"], "grid": grid, "fits": {s: vars(f) for s, f in fits.items()}}
    (ROOT / "phase4").mkdir(exist_ok=True)
    (ROOT / "phase4" / "tuning.json").write_text(json.dumps(out, indent=2))
    print("chosen sigma", best["sigma"])


def sigma() -> float:
    return json.loads((ROOT / "phase4" / "tuning.json").read_text())["sigma"]


def marginal_check(D, feat, tf, inp: dict, sims: dict, season: int, rng, weeks=None) -> dict:
    """CRPS of Phase 4 marginals vs the locked models on the same player-games."""
    from proplab.p1.model import crps_samples
    chosen = load_chosen()
    cfgs = {**chosen, "wrte_rec": WRTE_BASE, "wrte_rec_yds": WRTE_BASE}
    out = {}
    specs = {"qb_pass_yds": ("qb", "yds", "pass_yds"), "wrte_rec": ("recv", "rec", "receptions"),
             "wrte_rec_yds": ("recv", "yds", "rec_yds"), "rb_rec": ("recv", "rec", "receptions"),
             "rb_rec_yds": ("recv", "yds", "rec_yds")}
    for m, (side, stat, ycol) in specs.items():
        r = E.run_market(feat, tf, D["pools"], season, m, cfgs, N_SIMS, SEED, weeks=weeks, keep=True, cache={})
        frame = r["frame"]
        base_crps = dict(zip(r["keys"], r["crps_rows"]))
        mine, base, games = [], [], []
        for (gid, team), v in sims.items():
            if side == "qb":
                if v["qb"] is None:
                    continue
                key = f"{v['qb']['gsis_id']}|{gid}"
                if key in base_crps:
                    mine.append(crps_samples(v["sim"]["qb"][stat][None, :].astype(float), np.array([float(v["qb"][ycol])]))[0])
                    base.append(base_crps[key]); games.append(gid)
            else:
                for i, pid in enumerate(v["recv"]["gsis_id"].to_list()):
                    key = f"{pid}|{gid}"
                    if key in base_crps:
                        yv = float(v["recv"][ycol][i])
                        mine.append(crps_samples(v["sim"]["recv"][stat][:, i][None, :].astype(float), np.array([yv]))[0])
                        base.append(base_crps[key]); games.append(gid)
        mine, base, games = np.array(mine), np.array(base), np.array(games)
        u, inv = np.unique(games, return_inverse=True)
        sm, sb = np.bincount(inv, weights=mine), np.bincount(inv, weights=base)
        st = [sm[p].sum() / sb[p].sum() - 1 for p in (rng.integers(0, len(u), len(u)) for _ in range(1000))]
        lo, hi = np.percentile(st, [2.5, 97.5])
        out[m] = {"rows": int(len(mine)), "crps_p4": float(mine.mean()), "crps_locked": float(base.mean()),
                  "ratio_minus_1": float(mine.mean() / base.mean() - 1), "ci95": [float(lo), float(hi)],
                  "non_inferior": bool(hi < 0.01)}
    return out


def test() -> None:
    D, feat, tf = setup()
    sg = sigma()
    inps = inputs_for(D, feat, tf, range(2022, 2027))
    rng = np.random.default_rng(SEED + 1)
    res = {"sigma": sg, "splits": {}}
    for name, season, weeks in (("test_2025", 2025, None), ("live_2026", 2026, (1, 5))):
        fit = fit_params([inps[season - 3], inps[season - 2], inps[season - 1]])
        sims = simulate_season(inps[season], fit, sg, weeks=weeks)
        joint = summarize(joint_rows(sims, rng, fit.scales), rng)
        marg = marginal_check(D, feat, tf, inps[season], sims, season, rng, weeks=weeks)
        crit = {"joint_beats_shuffled": joint["vs_ci95"][1] < 0,
                "two_leg_beats_independent": joint["brier_ci95"][1] < 0,
                "two_leg_calibrated": joint["ece_joint"] <= joint["ece_null95"],
                "marginals_non_inferior": all(v["non_inferior"] for v in marg.values())}
        res["splits"][name] = {"fit": vars(fit), "joint": joint, "marginals": marg, "criteria": crit,
                               "passed": all(crit.values())}
        print(name, json.dumps({"fit": vars(fit), "joint": joint, "criteria": crit}, indent=1), flush=True)
        for m, v in marg.items():
            print(f"  {m:13s} n={v['rows']:5d} p4/locked-1 = {v['ratio_minus_1']:+.3%} CI [{v['ci95'][0]:+.3%}, {v['ci95'][1]:+.3%}]")
    (ROOT / "phase4" / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"tune": tune, "test": test}[sys.argv[1]]()
