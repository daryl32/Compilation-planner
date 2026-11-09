"""
✂️ Section cuts — the 🤖 new matcher's blocks, without the Audio Settings step. The track's 🧩 section map gives the
boundaries and each part's type; the type's 🍳 recipe gives the pace (clip length lo–hi seconds). Inside each
section the cuts land on beats, at the pace: a "rise" trend shortens the clips from hi to lo through the section
(tension building), a "fall" lengthens them lo → hi, "flat" keeps them around the middle. Section boundaries are
always cuts. Each block carries the same fields Classic's segments do (start, end, energy, intensity) plus its
section type.

No Streamlit in here.
"""

from __future__ import annotations

import numpy as np

import cut_scoring as CS
import section_types as ST

MIN_BLOCK_SEC = 0.5        # a sliver shorter than this at a section's end joins the block before it


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
    """The wanted clip length at `share` (0–1) of the way through the section."""
    pc = recipe.get("pace") or {"lo": 2.0, "hi": 4.0}
    lo, hi = float(pc["lo"]), float(pc["hi"])
    trend = (recipe.get("movement") or {}).get("trend", "flat")
    if trend == "rise":
        return hi + (lo - hi) * share
    if trend == "fall":
        return lo + (hi - lo) * share
    return (lo + hi) / 2


def cut_section(beats: np.ndarray, start: float, end: float, recipe: dict) -> list:
    """Cut times inside [start, end): on beats, at the recipe's pace. Returns the block starts (incl. start)."""
    starts = [start]
    t = start
    inner = beats[(beats > start + 1e-6) & (beats < end - 1e-6)]
    if inner.size == 0:
        return starts
    while True:
        want = pace_at(recipe, (t - start) / max(end - start, 1e-9))
        target = t + want
        if target >= end - MIN_BLOCK_SEC:
            break
        later = inner[inner > t + 1e-6]
        if later.size == 0:
            break
        nxt = float(later[int(np.argmin(np.abs(later - target)))])
        if nxt <= t + 1e-6:
            break
        if end - nxt < MIN_BLOCK_SEC:
            break
        starts.append(nxt)
        t = nxt
    return starts


def story_segments(track: dict, regions: list, recipes: dict) -> list:
    """Blocks for the whole track from its section map + the recipes' paces."""
    duration = float(track["duration_sec"])
    beats = _beats(track)
    curve, rate = _energy_curve(track)
    bounds = []
    for r in regions or []:
        a, b = float(r["start"]), float(min(r["end"], duration))
        if b <= a:
            continue
        recipe = recipes.get(r.get("type")) or {}
        for s in cut_section(beats, a, b, recipe):
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
                    "intensity": round(energy, 4), "section_type": t})
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
