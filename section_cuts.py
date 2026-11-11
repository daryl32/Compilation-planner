"""
✂️ Section cuts — the 🤖 new matcher's blocks, without the Audio Settings step. The track's 🧩 section map gives the
boundaries and each part's type; the type's 🍳 recipe gives the clip length range (lo–hi seconds). Each section is
divided as a whole: every clip inside the range, cuts on the beats the music accents (drum hits, changes, bar and
phrase lines — or vocal phrase starts where a voice drives the bar; audio_drive.py), clips a musical number of
beats long. A "rise" trend shortens the clips from hi to lo through the section, a "fall" lengthens them, a steady
section aims a little above the middle. Section boundaries are always cuts. Each block carries the same fields Classic's segments do (start, end, energy, intensity) plus its
section type.

No Streamlit in here.
"""

from __future__ import annotations

import numpy as np

import cut_scoring as CS
import section_types as ST

MIN_BLOCK_SEC = 0.5        # a sliver shorter than this at a section's end joins the block before it
MIN_CLIP_FLOOR = 1.4       # ✂️ Shortest clip, Auto: the shortest musical length (1, 2, 4, 8 beats) at least this long —
#                            a bar at 90 BPM (2.7 s), a bar at 128 (1.9 s), a bar at 170 (1.4 s), two beats at 70 (1.7 s).
#                            Daryl, 11 Nov (01 AURA, 90 BPM): half-bar clips of 1.3 s were too short, however good.


def auto_min_clip(beat: float) -> float:
    """The shortest clip for a track whose beats are `beat` seconds apart (see MIN_CLIP_FLOOR)."""
    beat = float(beat) if beat and beat > 0 else 0.5
    for n in (1, 2, 4, 8, 16):
        if n * beat >= MIN_CLIP_FLOOR - 1e-6:
            return n * beat
    return 16 * beat


def _beats(track: dict) -> np.ndarray:
    try:
        beats, _, _ = CS.beat_grid(track)
        b = np.asarray(beats, float)
        if b.size >= 2:
            return b
    except Exception:          # noqa: BLE001 — a track without a usable grid
        pass
    dur = float(track.get("duration_sec") or 0.0)
    bpm = float(track.get("bpm") or 120.0)
    return np.arange(0.0, dur + 1e-6, 60.0 / max(bpm, 30.0))


def _energy_curve(track: dict):
    hires = track.get("hires") or {}
    if hires.get("rms"):
        rms = np.asarray(hires["rms"], float)
        return rms / max(float(rms.max()), 1e-9), float(hires["rate"])
    env = track.get("energy_envelope") or []
    if env:
        t = np.array([e["time"] for e in env], float)
        e = np.array([e["energy"] for e in env], float)
        rate = 1.0 / max(float(np.median(np.diff(t))) if t.size > 1 else 1.0, 1e-3)
        return e / max(float(e.max()), 1e-9), rate
    return np.zeros(0), 1.0


def _mean_energy(curve: np.ndarray, rate: float, a: float, b: float) -> float:
    if curve.size == 0:
        return 0.5
    lo, hi = int(a * rate), max(int(a * rate) + 1, int(b * rate))
    seg = curve[lo:hi]
    return float(seg.mean()) if seg.size else 0.5


def pace_at(recipe: dict, share: float) -> float:
    """The wanted clip length at `share` (0–1) of the way through the section. A steady section aims a little
    above the middle of the range (Daryl, 10 Nov: rather slightly longer clips, when the footage holds)."""
    pc = recipe.get("pace") or {"lo": 2.0, "hi": 4.0}
    lo, hi = float(pc["lo"]), float(pc["hi"])
    trend = (recipe.get("movement") or {}).get("trend", "flat")
    if trend == "rise":
        return hi + (lo - hi) * share
    if trend == "fall":
        return lo + (hi - lo) * share
    return lo + FLAT_AIM * (hi - lo)


FLAT_AIM = 0.65            # where in the range a steady section aims
RANGE_SLACK = 0.15         # a clip may be this far outside the recipe's range (cuts sit on beats)
MUSICAL_BEATS = (1, 2, 4, 8, 16, 32)             # a beat, half a bar, 1, 2, 4, 8 bars
W_ODD_BARS = 0.15          # cost of a whole number of bars that isn't one of those (3, 5, 6 bars)
W_LENGTH = 0.5             # cost of a clip's distance from the wanted length (per its own length)
W_UNMUSICAL = 0.4          # cost of a clip that isn't whole bars or half a bar long (3, 5, 7 beats…)


def cut_section(beats: np.ndarray, start: float, end: float, recipe: dict, accent: np.ndarray = None,
                min_clip: float = 0.0) -> list:
    """Block starts inside [start, end) (incl. start). The section is divided as a whole: every clip within the
    recipe's range (so no short leftover at the end), cuts on the beats the music accents (`accent`, one value per
    beat — audio_drive.accents), clips a musical number of beats long, each as near the wanted length as that
    allows. Without accents: the most even musical division. min_clip: no clip shorter than this (seconds), whatever
    the recipe's range says — a section too short for two such clips stays one clip."""
    beats = np.asarray(beats, float)
    sel = (beats > start + 1e-6) & (beats < end - 1e-6)
    inner = beats[sel]
    if inner.size == 0:
        return [start]
    acc = np.asarray(accent, float)[sel] if accent is not None and len(accent) == len(beats) else np.zeros(inner.size)
    acc = acc - float(np.median(acc)) if acc.size else acc          # only beats that stand out attract a cut
    beat = float(np.median(np.diff(beats))) if beats.size > 1 else 0.5
    pc = recipe.get("pace") or {"lo": 2.0, "hi": 4.0}
    lo, hi = float(pc["lo"]), float(pc["hi"])
    m = max(float(min_clip or 0.0) - 0.25 * beat, 0.0) if min_clip else 0.0      # cuts sit on beats that wobble a little
    if m > 0:
        if end - start < 2 * m:
            return [start]
        lo, hi = max(lo, m / (1 - RANGE_SLACK)), max(hi, m / (1 - RANGE_SLACK))
        recipe = {**recipe, "pace": {"lo": lo, "hi": hi}}
    T = np.concatenate([[start], inner, [end]])
    A = np.concatenate([[0.0], acc, [0.0]])
    K = len(T)
    a_edge = float(max(acc.max(), 0.0)) if acc.size else 0.0      # a section edge is as good a cut as the best beat

    def solve(lo_, hi_):
        best = np.full(K, -np.inf)
        prev = np.full(K, -1, int)
        best[0] = 0.0
        for j in range(1, K):
            for i in range(j - 1, -1, -1):
                d = T[j] - T[i]
                if d > hi_ + 1e-6:
                    break
                if d < lo_ - 1e-6 or best[i] == -np.inf:
                    continue
                want = pace_at(recipe, (T[i] - start) / max(end - start, 1e-9))
                n_beats = int(round(d / beat))
                cost = W_LENGTH * abs(d - want) / max(want, 1e-6)
                if abs(d - n_beats * beat) <= 0.3 * beat:        # (a clip ending on a section edge off the beat isn't judged)
                    cost += 0.0 if n_beats in MUSICAL_BEATS else (W_ODD_BARS if n_beats % 4 == 0 else W_UNMUSICAL)
                # each clip is judged by how well it ends and how well its length suits — weighted by its length, so
                # many short clips can't out-score a few long ones just by collecting more accents
                v = best[i] + d * ((A[j] if j < K - 1 else a_edge) - cost)
                if v > best[j]:
                    best[j], prev[j] = v, i
        if best[K - 1] == -np.inf:
            return None
        cuts, j = [], K - 1
        while j > 0:
            j = int(prev[j])
            cuts.append(float(T[j]))
        return cuts[::-1]
    got = solve(max(lo * (1 - RANGE_SLACK), MIN_BLOCK_SEC), hi * (1 + RANGE_SLACK))
    if got is None and m > 0:                           # the range can't tile it: any length from the shortest clip up
        got = solve(m, max(hi * 2, end - start))
    if got is None:                                     # a section shorter than a clip, or one the range can't tile
        got = solve(MIN_BLOCK_SEC, max(hi * 2, end - start))
    return got or [start]


def story_segments(track: dict, regions: list, recipes: dict, min_clip: float = None) -> list:
    """Blocks for the whole track from its section map + the recipes' clip lengths, cut where the music accents
    (audio_drive). Each block: start, end, energy, intensity, section_type, driver (what drives the music there).
    min_clip: the shortest clip in seconds — None: worked out from the tempo (auto_min_clip), 0: no minimum."""
    import audio_drive as AD
    duration = float(track["duration_sec"])
    beats = _beats(track)
    curve, rate = _energy_curve(track)
    if min_clip is None:
        min_clip = auto_min_clip(float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5)
    try:
        drv = AD.drivers(track, beats)
        accent = AD.accents(track, beats, drv)
    except Exception:          # noqa: BLE001 — a track the extra curves can't be read from: even musical cuts
        drv, accent = None, None
    bounds = []
    for r in regions or []:
        a, b = float(r["start"]), float(min(r["end"], duration))
        if b <= a:
            continue
        recipe = recipes.get(r.get("type")) or {}
        for s in cut_section(beats, a, b, recipe, accent, min_clip):
            bounds.append((s, r.get("type")))
    if not bounds:
        bounds = [(0.0, None)]
    bounds.sort()
    out = []
    for k, (s, t) in enumerate(bounds):
        e = bounds[k + 1][0] if k + 1 < len(bounds) else duration
        if e - s < MIN_BLOCK_SEC and out:
            out[-1]["end"] = round(e, 3)
            continue
        energy = _mean_energy(curve, rate, s, e)
        out.append({"start": round(s, 3), "end": round(e, 3), "energy": round(energy, 4),
                    "intensity": round(energy, 4), "section_type": t,
                    "driver": AD.driver_at(drv, (s + e) / 2) if drv else "loudness"})
    if out:
        out[-1]["end"] = round(duration, 3)
    return out


def describe(segments: list) -> str:
    """One line: blocks, clip length range, per type."""
    if not segments:
        return "no blocks"
    lens = [s["end"] - s["start"] for s in segments]
    per = {}
    for s in segments:
        per[s.get("section_type")] = per.get(s.get("section_type"), 0) + 1
    types = " · ".join(f"{ST.label(t)} {n}" for t, n in sorted(per.items(), key=lambda kv: -kv[1]) if t)
    return f"{len(segments)} blocks · clips {min(lens):.1f}–{max(lens):.1f} s" + (f" · {types}" if types else "")
