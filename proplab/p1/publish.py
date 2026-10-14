"""Write the latest Phase 1 results and projections into index.html.

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


def slim_split(s: dict) -> dict:
    keep = ("crps", "mae_median", "coverage80", "brier", "ece", "mean_pred", "mean_actual")
    return {
        "rows": s["rows"], "games": s["games"], "weeks": s["weeks"],
        "models": {m: {mk: {k: v[mk][k] for k in keep} for mk in ("receptions", "yards")}
                   for m, v in s["models"].items()},
        "vs_B2": s["vs_B2"], "vs_B1": s["vs_B1"],
        "reliability": {mk: s["reliability"][mk]["M1"] for mk in ("receptions", "yards")},
    }


def main() -> None:
    res = json.loads((ROOT / "phase1" / "results.json").read_text())
    proj = json.loads((ROOT / "phase1" / "projections.json").read_text())
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "configs": res["configs"],
        "gate": res["gate"],
        "splits": {k: slim_split(res[k]) for k in ("tuning_2024", "test_2025", "live_2026")},
        "projections": proj,
    }
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    block = f'{START}\n<script id="data" type="application/json">{blob}</script>\n{END}'
    html = PAGE.read_text()
    html, n = re.subn(re.escape(START) + r".*?" + re.escape(END), lambda _: block, html, flags=re.S)
    assert n == 1, "DATA markers not found in index.html"
    PAGE.write_text(html)
    print(f"index.html updated: {len(proj['players'])} projections, {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
