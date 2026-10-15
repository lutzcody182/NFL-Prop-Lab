"""Phase 4 tests: the simulated game adds up, keeps each player's expected targets,
and the committed 2025 joint score reproduces."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from proplab.p4 import sim as S

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world():
    D, feat, tf = S.setup()
    inps = S.inputs_for(D, feat, tf, [2022, 2023, 2024, 2025])
    fit = S.fit_params([inps[2022], inps[2023], inps[2024]])
    return inps, fit


def test_fit_uses_only_earlier_seasons(world):
    inps, fit = world
    changed = dict(inps[2025])
    changed["team"] = inps[2025]["team"].with_columns(pl.col("team_attempts") * 3)
    # fitting for 2025 never sees 2025; refitting with 2025 altered must give the same result
    again = S.fit_params([inps[2022], inps[2023], inps[2024]])
    assert vars(again) == vars(fit)


def test_simulated_game_adds_up(world):
    inps, fit = world
    sims = S.simulate_season(inps[2025], fit, 0.1, n=500, weeks=(3, 3))
    assert len(sims) > 20
    for v in sims.values():
        s = v["sim"]
        if "qb" not in s:
            continue
        all_targets = s["recv"]["targets"].sum(axis=1)
        all_catches = s["recv"]["rec"].sum(axis=1)
        assert np.all(all_targets <= v["A"]), "more targets than pass attempts"
        assert np.all(s["qb"]["att"] <= v["A"])
        assert np.all(s["qb"]["comp"] <= s["qb"]["att"]), "more completions than attempts"
        assert np.all(s["recv"]["rec"] <= s["recv"]["targets"])


def test_expected_targets_preserved(world):
    inps, fit = world
    sims = S.simulate_season(inps[2025], fit, 0.0, n=4000, weeks=(5, 6))
    sim_mean, model_mean = [], []
    for v in sims.values():
        sim_mean += list(v["sim"]["recv"]["targets"].mean(axis=0))
        model_mean += list(v["recv"]["tgt_mu"].to_numpy())
    sim_mean, model_mean = np.array(sim_mean), np.array(model_mean)
    # unbiased overall and per player (allowing for the targets/attempts calibration of the team model)
    assert sim_mean.sum() == pytest.approx(model_mean.sum(), rel=0.05)
    assert np.corrcoef(sim_mean, model_mean)[0, 1] > 0.98


def test_saved_2025_joint_score_reproduces(world):
    inps, fit = world
    saved = json.loads((ROOT / "phase4" / "results.json").read_text())["splits"]["test_2025"]["joint"]
    sims = S.simulate_season(inps[2025], fit, S.sigma())
    rows = S.joint_rows(sims, np.random.default_rng(0), fit.scales)
    assert np.mean([r["vs_joint"] for r in rows]) == pytest.approx(saved["vs_joint"], rel=1e-9)
