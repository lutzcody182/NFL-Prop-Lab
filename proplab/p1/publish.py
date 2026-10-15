"""Write the latest results and projections (Phase 1 and Phase 3a) into index.html.

index.html is the whole dashboard: this script only replaces the JSON between the
DATA markers, so the page stays one self-contained file for GitHub Pages.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "index.html"
START, END = "<!-- DATA:START -->", "<!-- DATA:END -->"
KEEP = ("crps", "mae_median", "coverage80", "brier", "ece", "mean_pred", "mean_actual")


def slim_p1(s: dict) -> dict:
    return {
        "rows": s["rows"], "games": s["games"], "weeks": s["weeks"],
        "models": {m: {mk: {k: v[mk][k] for k in KEEP} for mk in ("receptions", "yards")}
                   for m, v in s["models"].items()},
        "vs_B2": s["vs_B2"], "vs_B1": s["vs_B1"],
        "reliability": {mk: s["reliability"][mk]["M1"] for mk in ("receptions", "yards")},
    }


def slim_p3(split: dict) -> dict:
    keep = KEEP + ("ece_null95", "coverage_tol", "log_loss", "rows")
    return {m: {"label": v["label"], "rows": v["rows"], "games": v["games"], "vs_B2": v["vs_B2"],
                "criteria": v["criteria"], "passed": v["passed"],
                "models": {name: {k: r[k] for k in keep if k in r} for name, r in v["models"].items()}}
            for m, v in split.items()}


def main() -> None:
    p1 = json.loads((ROOT / "phase1" / "results.json").read_text())
    p1proj = json.loads((ROOT / "phase1" / "projections.json").read_text())
    p3 = json.loads((ROOT / "phase3" / "results.json").read_text())
    p3proj = json.loads((ROOT / "phase3" / "projections.json").read_text())
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "p1": {"configs": p1["configs"], "gate": p1["gate"],
               "splits": {k: slim_p1(p1[k]) for k in ("tuning_2024", "test_2025", "live_2026")}},
        "p1proj": p1proj,
        "p3": {"configs": p3["configs"], "splits": {k: slim_p3(v) for k, v in p3["splits"].items()}},
        "p3proj": p3proj,
        "r3": json.loads((ROOT / "phase3" / "dashboard_3bc.json").read_text()),
        "p4": json.loads((ROOT / "phase4" / "week_sims.json").read_text()),
        "live": json.loads(live.read_text()) if (live := ROOT / "phase5" / "dashboard_live.json").exists() else None,
    }
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    block = f'{START}\n<script id="data" type="application/json">{blob}</script>\n{END}'
    html = PAGE.read_text()
    html, n = re.subn(re.escape(START) + r".*?" + re.escape(END), lambda _: block, html, flags=re.S)
    assert n == 1, "DATA markers not found in index.html"
    PAGE.write_text(html)
    n3 = sum(len(m["players"]) for m in p3proj["markets"].values())
    print(f"index.html updated: {len(p1proj['players'])} WR/TE + {n3} other projections, {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
