"""Where a track's analysis file (audio_catalogue/<track_id>.json) is on this machine.

A name with an accent can be stored two ways — "Ã" as one character, or "A" plus a combining
tilde (iPhones and Macs often save the second) — and Linux treats them as different file
names. A file renamed on Google Drive arrives in whichever form Drive has, while the
track_id inside it (and in plans, story maps, queues) may be in the other: the app then
couldn't open "01 NO BATIDÃO.json" although it was right there (22 Oct 2026).
track_file() finds it either way. No Streamlit in here."""

from __future__ import annotations

import unicodedata
from pathlib import Path

_LISTINGS: dict = {}       # folder → (folder mtime, {NFC stem: path})


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s))


def track_file(track_id: str, audio_dir=None) -> Path:
    """audio_dir/<track_id>.json — or the file whose name is the same written the other way.
    Returns the plain path when there's no such file (so callers can still create it)."""
    if audio_dir is None:
        from config import AUDIO_DIR as audio_dir
    d = Path(audio_dir)
    p = d / f"{track_id}.json"
    if p.exists():
        return p
    try:
        mtime = d.stat().st_mtime
    except OSError:
        return p
    got = _LISTINGS.get(str(d))
    if got is None or got[0] != mtime:
        names = {}
        for f in d.glob("*.json"):
            names.setdefault(_nfc(f.stem), f)
        got = _LISTINGS[str(d)] = (mtime, names)
    return got[1].get(_nfc(track_id), p)
