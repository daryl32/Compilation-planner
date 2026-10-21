"""Musical key: estimate from chroma, Camelot codes and mixing compatibility.

No audio decoding here — audio_analysis.py turns a track into a 12-bin
chroma profile and calls estimate_key(); the 🎚️ Mix screen and the mashup
partner search use the rest.

Key dict (stored as track["key"]):
    {"tonic": "A", "mode": "minor", "pc": 9, "camelot": "8A",
     "name": "A minor", "confidence": 0.71}
"""

from __future__ import annotations

import numpy as np

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl–Kessler key profiles (C major / C minor).
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def camelot(pc: int, mode: str) -> str:
    """Camelot wheel code: C major = 8B, A minor = 8A."""
    if mode == "minor":
        pc = (pc + 3) % 12              # its relative major shares the number
    n = (7 * pc + 8) % 12 or 12
    return f"{n}{'A' if mode == 'minor' else 'B'}"


def key_from(pc: int, mode: str, confidence: float = 1.0) -> dict:
    pc = int(pc) % 12
    return {"tonic": NAMES[pc], "mode": mode, "pc": pc, "camelot": camelot(pc, mode),
            "name": f"{NAMES[pc]} {mode}", "confidence": round(float(confidence), 2)}


def estimate_key(profile) -> dict | None:
    """profile: 12 pitch-class weights (C..B), e.g. the track's average chroma.
    Correlates it with all 24 rotated key profiles. confidence 0–1 = how far the
    best key stands out from the runner-up that isn't its relative key."""
    p = np.asarray(profile, dtype=float)
    if p.shape != (12,) or not np.isfinite(p).all() or p.sum() <= 0:
        return None
    p = (p - p.mean()) / (p.std() + 1e-12)
    scores = []
    for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
        z = (prof - prof.mean()) / prof.std()
        for pc in range(12):
            scores.append((float(np.dot(p, np.roll(z, pc)) / 12.0), pc, mode))
    scores.sort(reverse=True)
    best = scores[0]
    rel_cam = camelot(best[1], best[2])[:-1]
    rival = next(s for s in scores[1:] if camelot(s[1], s[2])[:-1] != rel_cam)
    conf = np.clip((best[0] - rival[0]) / 0.25, 0.0, 1.0) * np.clip(best[0] / 0.6, 0.0, 1.0)
    return key_from(best[1], best[2], conf)


def parse_camelot(code: str) -> tuple | None:
    try:
        return int(code[:-1]), code[-1].upper()
    except (ValueError, TypeError, IndexError):
        return None


def compatibility(a: dict | None, b: dict | None) -> tuple:
    """(score 0–1, label) for playing key b next to / over key a."""
    if not a or not b:
        return 0.5, "key unknown"
    ca, cb = parse_camelot(a.get("camelot", "")), parse_camelot(b.get("camelot", ""))
    if not ca or not cb:
        return 0.5, "key unknown"
    step = min((ca[0] - cb[0]) % 12, (cb[0] - ca[0]) % 12)
    if step == 0 and ca[1] == cb[1]:
        return 1.0, "same key"
    if step == 0:
        return 0.9, "relative major/minor"
    if step == 1 and ca[1] == cb[1]:
        return 0.85, "next door on the wheel"
    if step == 2 and ca[1] == cb[1]:
        return 0.55, "two steps — energy lift"
    if step == 1:
        return 0.45, "diagonal — use with care"
    return 0.15, "clashes"


def shifted(key: dict | None, semitones: int) -> dict | None:
    if not key:
        return None
    return key_from(key["pc"] + int(semitones), key["mode"], key.get("confidence", 1.0))


def best_shift(target: dict | None, key: dict | None, max_semitones: int = 3) -> int:
    """Smallest pitch shift (semitones, ±max) that makes key sit well with target;
    0 when it already does, or when nothing within reach is better."""
    if not target or not key:
        return 0
    base = compatibility(target, key)[0]
    best, best_score = 0, base
    for s in sorted(range(-max_semitones, max_semitones + 1), key=abs):
        if s == 0:
            continue
        score = compatibility(target, shifted(key, s))[0]
        if score > best_score + 0.05:
            best, best_score = s, score
    return best if best_score >= 0.85 else 0
