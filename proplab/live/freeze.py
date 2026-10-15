"""Freeze this week's predictions into the ledger BEFORE the first kickoff.

    python -m proplab.live.freeze            # next unplayed week (from the projection files)

Copies the projection files into ledger/<season>_w<week>/ unchanged and writes a manifest with
each file's SHA-256, the freeze time, the week's first kickoff and the code version. Refuses to
run once any game of that week has kicked off, so the ledger only ever holds real forecasts.
Never overwrites an existing freeze.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "wrte.json": ROOT / "phase1" / "projections.json",      # WR/TE receptions & yards (Phase 1 M1)
    "markets.json": ROOT / "phase3" / "projections.json",   # QB/RB/anytime TD (Phase 3a locked)
    "sims.json": ROOT / "phase4" / "week_sims.json",        # joint game simulation draws (Phase 4)
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def first_kickoff(season: int, week: int) -> dt.datetime:
    """From the database when it is reachable (picks up flexed/moved games), else the exported schedule."""
    try:
        from proplab import db
        (k,) = db.connect().execute("SELECT min(kickoff_utc) FROM games WHERE season = %s AND week = %s AND game_type = 'REG'",
                                    (season, week)).fetchone()
        if k is not None:
            return k
    except Exception:
        pass
    games = json.loads((ROOT / "data" / "schedule.json").read_text())
    ks = [g["kickoff_utc"] for g in games if g["season"] == season and g["week"] == week and g["game_type"] == "REG"]
    return dt.datetime.fromisoformat(min(ks).replace("Z", "+00:00"))


def main(now: dt.datetime | None = None) -> Path:
    now = now or dt.datetime.now(dt.timezone.utc)
    weeks = {(json.loads(p.read_text())["meta"]["season"], json.loads(p.read_text())["meta"]["week"]) for p in SOURCES.values()}
    if len(weeks) != 1:
        sys.exit(f"Projection files are for different weeks: {weeks}. Regenerate them first.")
    season, week = weeks.pop()
    kick = first_kickoff(season, week)
    if now >= kick:
        sys.exit(f"Refusing to freeze: week {week} kicked off at {kick.isoformat()} (now {now.isoformat(timespec='seconds')}).")
    out = ROOT / "ledger" / f"{season}_w{week:02d}"
    if (out / "manifest.json").exists():
        sys.exit(f"{out.relative_to(ROOT)} is already frozen; a freeze is never overwritten.")
    out.mkdir(parents=True)
    files = {}
    for name, src in SOURCES.items():
        shutil.copyfile(src, out / name)
        files[name] = {"sha256": sha256(out / name), "generated": json.loads(src.read_text())["meta"].get("lines_fetched_utc")}
    manifest = {
        "season": season, "week": week, "frozen_utc": now.isoformat(timespec="seconds"),
        "first_kickoff_utc": kick.isoformat(), "minutes_before_kickoff": round((kick - now).total_seconds() / 60),
        "code_git_sha": git_sha(), "files": files,
        "models": {"wrte": "Phase 1 M1 (phase1/tuning.json)", "markets": "Phase 3a locked (phase3/tuning.json)",
                   "sims": "Phase 4 simulator (phase4/tuning.json)"},
        "assumption": "Actives = each team's actives in its last completed game; injuries for this week not used.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Frozen {out.relative_to(ROOT)}: {manifest['minutes_before_kickoff']} minutes before the first kickoff.")
    return out


if __name__ == "__main__":
    main()
