"""Phase 14 pass/fail re-check: Phase 8 recalibration and gate on 2025 for all engine markets, with the engine
including the adopted Phase 14 changes (phase14/PREREGISTRATION.md).

    python -m proplab.p14.recal     # -> phase14/recal_engine.json, phase14/recalibration.json (live maps)
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from proplab.p11.recal import RoleEngine
from proplab.p8 import recal as R

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "phase14" / "recal_engine.json"
MAPS = ROOT / "phase14" / "recalibration.json"


def adopted_fields() -> dict:
    out = json.loads((ROOT / "phase14" / "ablation.json").read_text())
    if not out.get("adopted"):
        return {}
    car = out["final_configs"]["car"]
    return {f: car[f] for f in ("rb_disp_bins", "rb_eff_sd") if car.get(f)}


class Phase14Engine(RoleEngine):
    def __init__(self):
        super().__init__()
        extra = adopted_fields()
        self.variants["chosen"] = {vt: (replace(c, **extra) if vt in ("tgt", "car") else c)
                                   for vt, c in self.variants["chosen"].items()}
        self.cache = {v: {} for v in self.variants}


if __name__ == "__main__":
    print("adopted Phase 14 fields:", adopted_fields())
    R.main(eng=Phase14Engine(), results_path=RESULTS, maps_path=MAPS, live_weeks=(1, 9))
