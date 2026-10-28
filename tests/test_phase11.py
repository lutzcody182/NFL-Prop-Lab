"""Phase 11: team position-group shares use only earlier games; defaults keep the earlier model unchanged;
recalibration maps load and the dispersion bins are used."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from proplab import db
from proplab.p1 import data as P1D
from proplab.p1 import model as M

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    c = db.connect()
    yield P1D.load_universe(c), P1D.load_team_games(c)
    c.close()


def test_defaults_are_off():
    c = M.Config()
    assert c.disp_bins is False and c.pos_redist == 0.0 and "disp3" not in c.label() and "posw" not in c.label()


def test_group_shares_ignore_target_week(data):
    u, tg = data
    base = M.group_features(u, tg).sort("game_id", "team", "pos")
    w9 = pl.col("game_id").str.starts_with("2025_09")
    changed = M.group_features(u.with_columns(pl.when(w9).then(pl.col("targets") * 5).otherwise(pl.col("targets")).alias("targets")),
                               tg).sort("game_id", "team", "pos")
    assert base.filter(w9).height > 0 and base.filter(w9).equals(changed.filter(w9))
    w10 = pl.col("game_id").str.starts_with("2025_10")
    assert not base.filter(w10).equals(changed.filter(w10))


def test_dispersion_bins_lookup():
    k = M.Constants({}, {"WR": 0.6}, {"WR": 12.0}, {}, 35.0, 0.9, np.zeros(4),
                    catch_yards={"WR": np.ones(10, dtype=np.float32), "TE": np.ones(10, dtype=np.float32)})
    k.dispersion = {("M1", "WR"): 5.0, ("M1", "TE"): 5.0, ("M1", "WR", 0): 1e6, ("M1", "WR", 1): 1e6, ("M1", "WR", 2): 0.5}
    k.disp_cuts = {"WR": (3.0, 6.0)}
    rows = pl.DataFrame({"pos": ["WR", "WR"], "mu_t": [2.0, 9.0], "p_catch": [1.0, 1.0], "ypr": [10.0, 10.0]})
    rec, _ = M.simulate(rows, k, "M1", 20000, seed=1)
    # bin 0 is nearly Poisson (variance ~ mean); bin 2 is very overdispersed
    assert rec[0].var() == pytest.approx(2.0, rel=0.1)
    assert rec[1].var() > 9.0 * 5


def test_live_files_consistent():
    a = json.loads((ROOT / "phase11" / "ablation.json").read_text())
    if a.get("adopted"):
        cfg = M.Config(**a["final"])
        assert cfg.disp_bins or cfg.pos_redist
    r = json.loads((ROOT / "phase11" / "recal_wrte.json").read_text())["live_maps"]["maps"]
    for mk in ("receptions", "yards"):
        q = np.asarray(r[mk]["q"])
        assert len(q) == 201 and np.all(np.diff(q) >= -1e-12) and 0 <= q[0] <= q[-1] <= 1
