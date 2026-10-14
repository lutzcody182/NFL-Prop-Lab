"""Write data/schedule.json for tools/odds_backfill.py (which runs without a database)."""
from __future__ import annotations

import json
from pathlib import Path

from proplab import db

OUT = Path(__file__).resolve().parents[2] / "data" / "schedule.json"


def main() -> None:
    conn = db.connect()
    rows = conn.execute("""
        SELECT game_id, season, week, game_type, home_team, away_team,
               to_char(kickoff_utc AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
        FROM games WHERE season >= 2023 ORDER BY kickoff_utc, game_id""").fetchall()
    keys = ["game_id", "season", "week", "game_type", "home_team", "away_team", "kickoff_utc"]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps([dict(zip(keys, r)) for r in rows], indent=1))
    print(f"{OUT.relative_to(OUT.parents[1])}: {len(rows)} games")


if __name__ == "__main__":
    main()
