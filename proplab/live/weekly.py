"""The weekly pipeline: refresh data, re-project the next week, freeze it, score finished weeks, rebuild the dashboard.

    python -m proplab.live.weekly            # everything
    python -m proplab.live.weekly --dry-run  # print the steps only

Run it on Tuesday/Wednesday (after Monday night, before Thursday's kickoff). The freeze step is
skipped, not failed, when the week has already started or is already frozen. Models are never
re-tuned here: every step uses the locked configurations in phase*/tuning.json.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("ingest nflverse + FTN data + 13 quality checks", ["proplab.ingest"], True),
    ("team ratings (Phase 3b)", ["proplab.ratings.team", "publish"], True),
    ("player ratings (Phase 3b)", ["proplab.ratings.player", "publish"], True),
    ("WR/TE projections (Phase 1)", ["proplab.p1.project"], True),
    ("QB/RB/TD projections (Phase 3a)", ["proplab.p3.project"], True),
    ("ratings + matchups export (Phase 3b/3c)", ["proplab.p3.export_3bc"], True),
    ("defensive scheme profiles (Phase 6)", ["proplab.p6.export"], True),
    ("game simulations (Phase 4)", ["proplab.p4.export"], True),
    ("freeze the week in the ledger", ["proplab.live.freeze"], False),   # exits non-zero after kickoff: OK
    ("score finished ledger weeks", ["proplab.live.score"], True),
    ("week / players / health export", ["proplab.live.export"], True),
    ("rebuild index.html", ["proplab.p1.publish"], True),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true", help="use the database as it is")
    args = ap.parse_args(argv)
    for i, (label, mod, required) in enumerate(STEPS, 1):
        if args.skip_ingest and mod[0] == "proplab.ingest":
            continue
        cmd = [sys.executable, "-m", *mod]
        print(f"[{i}/{len(STEPS)}] {label}: {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        t = time.time()
        r = subprocess.run(cmd)
        print(f"      exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
        if r.returncode != 0 and required:
            print(f"Stopping: '{label}' failed. Nothing after it ran; the ledger is unchanged.", flush=True)
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
