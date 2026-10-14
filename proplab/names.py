"""Player-name normalization shared by ingest (name variants) and odds matching."""
from __future__ import annotations

import re
import unicodedata

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize(name: str | None) -> str:
    """'D.J. Moore' -> 'dj moore'; 'Amon-Ra St. Brown' -> 'amon ra st brown';
    'Marvin Harrison Jr.' -> 'marvin harrison'."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("’", "").replace("'", "").replace(".", "")
    s = re.sub(r"[-_/]", " ", s)
    tokens = [t for t in re.sub(r"[^a-z0-9 ]", " ", s).split() if t not in SUFFIXES]
    return " ".join(tokens)


def variants(display: str | None, first: str | None, last: str | None,
             football: str | None, common_first: str | None) -> set[str]:
    out = {normalize(display)}
    for f in (first, football, common_first):
        if f and last:
            out.add(normalize(f"{f} {last}"))
    out.discard("")
    return out
