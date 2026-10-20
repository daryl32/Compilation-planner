"""✂️ Hand edits to a track's cuts.

The cutting methods place cuts automatically; the cut editor (Step 0) lets
you add, move and remove them while listening. An edit is stored as the full
list of cut times you ended up with (per project), plus a signature of the
settings the cuts were made from — so if those settings change later the
planner can ask whether to keep your cuts or re-cut.

segments_from_cuts() turns that list back into the same segment dicts every
method produces ({"start", "end", "energy", "cut", "cut_strength"}), so the
story map, matching and export work unchanged. A cut that is still where the
method put it keeps its original kind (and colour on the chart); added or
moved ones are kind "edit".

Pure functions, no Streamlit.
"""

from __future__ import annotations

import json

import numpy as np

MIN_GAP = 0.15        # seconds — two cuts can't be closer than this
EDGE = 0.05           # no cut within this of the start / end of the track
SAME_CUT = 0.02       # a cut this close to one of the method's counts as unchanged


def base_cut_times(segments: list) -> list:
    """The cut times (segment starts, without the first) of a method's segments."""
    return [round(float(s["start"]), 3) for s in segments[1:]]


def clean_cuts(cuts, duration: float, min_gap: float = MIN_GAP) -> list:
    """Sorted, rounded to ms, inside the track, and never closer than min_gap
    (of two cuts too close together the earlier one is kept)."""
    out = []
    for t in sorted(float(c) for c in cuts if c is not None):
        if not (EDGE < t < duration - EDGE):
            continue
        if out and t - out[-1] < min_gap:
            continue
        out.append(round(t, 3))
    return out


def _near(t: float, ts: list, tol: float) -> int | None:
    if not ts:
        return None
    arr = np.asarray(ts, dtype=float)
    i = int(np.argmin(np.abs(arr - t)))
    return i if abs(arr[i] - t) <= tol else None


def diff(base_cuts: list, cuts: list, tol: float = SAME_CUT) -> dict:
    """{"added": [...], "removed": [...], "kept": n} between the method's cuts and edited ones."""
    added = [t for t in cuts if _near(t, base_cuts, tol) is None]
    removed = [t for t in base_cuts if _near(t, cuts, tol) is None]
    return {"added": added, "removed": removed, "kept": len(cuts) - len(added)}


def _energy(track: dict, s: float, e: float) -> float:
    hires = track.get("hires")
    if hires:
        rate = float(hires.get("rate", 30))
        rms = hires.get("rms") or []
        a, b = int(s * rate), int(e * rate)
        if b > a and len(rms):
            return round(float(np.mean(rms[a:b])), 3)
        return 0.0
    env = track.get("energy_envelope") or []
    vals = [x["energy"] for x in env if s <= x["time"] < e]
    return round(float(np.mean(vals)), 3) if vals else 0.0


def _strength(track: dict, t: float) -> float:
    hires = track.get("hires")
    if not hires or not hires.get("onset"):
        return 0.0
    i = int(round(t * float(hires.get("rate", 30))))
    onset = hires["onset"]
    return round(float(onset[i]), 3) if 0 <= i < len(onset) else 0.0


def segments_from_cuts(track: dict, cuts: list, base_segments: list, tol: float = SAME_CUT) -> list:
    """Segments for the edited cut list (see module docstring)."""
    duration = float(track["duration_sec"])
    cuts = clean_cuts(cuts, duration)
    base_starts = [float(s["start"]) for s in base_segments]
    bounds = [0.0] + cuts + [duration]
    out = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if k == 0:
            kind, strength = (base_segments[0].get("cut", "start") if base_segments else "start"), 0.0
        else:
            j = _near(s, base_starts, tol)
            if j is not None and j > 0:
                kind = base_segments[j].get("cut", "edit")
                strength = base_segments[j].get("cut_strength", _strength(track, s))
            else:
                kind, strength = "edit", _strength(track, s)
        out.append({"start": round(s, 3), "end": round(e, 3), "energy": _energy(track, s, e),
                    "cut": kind, "cut_strength": strength})
    return out


def settings_signature(cfg: dict) -> str:
    """A stable string for a segmentation_settings() dict — changes when any cutting
    setting does. Numbers are compared by value (a widget can turn 0.0 into 0)."""
    def norm(v):
        if isinstance(v, bool) or v is None or isinstance(v, str):
            return v
        if isinstance(v, (int, float)):
            return round(float(v), 6)
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return str(v)
    return json.dumps({k: norm(v) for k, v in cfg.items()}, sort_keys=True)


# ---------------------------------------------------------------------------
# 🔒 Locked cuts, 🎨 re-cutting a region, 👻 ideas, ⚖️ A/B — the cut editor extras
# ---------------------------------------------------------------------------

LOCK_CLEAR = 0.35     # seconds — a new method cut this close to a locked one gives way to it


def clean_locked(locked, cuts: list, tol: float = SAME_CUT) -> list:
    """The locked times that are still cuts (snapped to the cut they belong to)."""
    out = []
    for t in locked or []:
        j = _near(float(t), cuts, tol)
        if j is not None and cuts[j] not in out:
            out.append(cuts[j])
    return sorted(out)


def merge_locked(new_cuts: list, locked: list, duration: float, clear: float = LOCK_CLEAR) -> list:
    """A fresh set of cuts (e.g. the method's after the settings changed) with the 🔒 locked
    cuts kept exactly where they are; any new cut within `clear` seconds of one gives way."""
    locked = [float(t) for t in locked or []]
    kept = [t for t in new_cuts if all(abs(t - l) > clear for l in locked)]
    return clean_cuts(locked + kept, duration)


def recut_region(cuts: list, region_cuts: list, start: float, end: float, duration: float,
                 locked: list = ()) -> list:
    """🎨 Re-cut a part of the track: the cuts inside [start, end) are replaced by
    `region_cuts` that fall inside it; cuts outside the region and 🔒 locked cuts stay."""
    start, end = float(min(start, end)), float(max(start, end))
    inside = lambda t: start - 1e-6 <= t < end - 1e-6          # noqa: E731
    outside = [t for t in cuts if not inside(t)]
    new = [t for t in region_cuts if inside(t)]
    keep_locked = [t for t in locked or [] if inside(t)]
    new = [t for t in new if all(abs(t - l) > LOCK_CLEAR for l in keep_locked)]
    return clean_cuts(outside + keep_locked + new, duration)


def ghost_cuts(candidates, cuts: list, duration: float, min_gap: float = MIN_GAP) -> list:
    """👻 Ideas: candidate cut times (from other styles / a busier pace) that aren't already
    cuts — tap one in the editor to accept it."""
    out = []
    for t in sorted({round(float(c), 3) for c in candidates or []}):
        if not (EDGE < t < duration - EDGE):
            continue
        if _near(t, cuts, min_gap) is not None or (out and t - out[-1] < min_gap):
            continue
        out.append(t)
    return out


def cut_stats(cuts: list, duration: float) -> dict:
    """⚖️ For comparing two cut sets: segment count and lengths."""
    bounds = [0.0] + sorted(cuts) + [float(duration)]
    lens = np.diff(bounds)
    return {"segments": len(lens), "average": float(lens.mean()) if len(lens) else 0.0,
            "shortest": float(lens.min()) if len(lens) else 0.0, "longest": float(lens.max()) if len(lens) else 0.0}
