"""🎯 Aim for a result, not a threshold.

The ⚡ Energy changes & big hits (Adaptive) method has dials that are hard to
reason about — an energy-change threshold, a big-hit threshold, shortest and
longest segment — and they interact. Here you say what you want instead —
"about 40 segments" or "about 2.5 s per clip" — and adaptive_for_target()
finds dial values that give it.

How: the shortest / longest segment are set from the target clip length
(shortest ≈ 0.45×, longest ≈ 2.2× — so the longest-segment splits alone
already give about half the cuts, spread evenly), then one "sensitivity"
moves both thresholds together, from strict (only the biggest changes and
hits) to loose. Every threshold value the sliders can show is tried (they're
in 0.01 steps), so the dials it sets give exactly the result it reports. If
the target can't be reached that way, the length limits are widened.

Pure functions, no Streamlit.
"""

from __future__ import annotations

import math

from core.segmentation import build_track_segments

# Slider ranges on the planner page (values set here must fit them).
CHANGE_MIN, CHANGE_MAX = 0.02, 0.5
HIT_MIN, HIT_MAX = 0.5, 1.0
MIN_SEG_RANGE = (0.3, 8.0)
MAX_SEG_RANGE = (0.0, 30.0)

# (shortest ×, longest ×) the target clip length — tried in order until the target is met.
LENGTH_SHAPES = [(0.45, 2.2), (0.3, 2.2), (0.45, 3.5), (0.2, 1.6), (0.6, 5.0)]
GOOD_ENOUGH = 0.05     # within 5% of the target segment count


def _round_to(x: float, step: float) -> float:
    return round(round(x / step) * step, 4)


def _threshold_grid() -> list:
    """(change_threshold, hit_threshold) from strict to loose, on the sliders' 0.01 steps.
    Both move together: change from 0.5 down to 0.02 (log scale), hit from 1.0 to 0.5."""
    out = []
    n = int(round((CHANGE_MAX - CHANGE_MIN) / 0.01))
    for k in range(n + 1):
        ct = round(CHANGE_MAX - 0.01 * k, 2)
        s = math.log(ct / CHANGE_MAX) / math.log(CHANGE_MIN / CHANGE_MAX)       # 0 → 1
        hit = round(HIT_MAX - (HIT_MAX - HIT_MIN) * s, 2)
        out.append((ct, hit))
    return out


def _count(track, settings, bpb) -> int:
    return len(build_track_segments(
        track, float(settings["change_window_secs"]), float(settings["change_threshold"]),
        float(settings["min_segment_sec"]), bool(settings["react_to_hits"]), float(settings["hit_threshold"]),
        float(settings["snap_secs"]), float(settings["max_segment_sec"]), bpb))


def target_count(track: dict, segments: int = None, avg_sec: float = None) -> int:
    duration = float(track["duration_sec"])
    if segments:
        return max(1, int(round(segments)))
    if avg_sec and avg_sec > 0:
        return max(1, int(round(duration / avg_sec)))
    raise ValueError("give segments or avg_sec")


def adaptive_for_target(track: dict, *, segments: int = None, avg_sec: float = None, base: dict,
                        beats_per_bar: int = 4) -> dict:
    """Adaptive dial values that cut `track` into about `segments` segments
    (or segments about `avg_sec` long on average).

    base: the current Adaptive settings (change_window_secs, snap_secs and
    react_to_hits are kept as they are; the rest are set).

    Returns {"settings": {...}, "target": n, "segments": got, "avg_sec": …,
    "exact": bool (within 5%), "notes": [plain-language lines]}."""
    duration = float(track["duration_sec"])
    want = target_count(track, segments, avg_sec)
    want_len = duration / want
    keep = {k: base[k] for k in ("change_window_secs", "snap_secs", "react_to_hits")}
    hires = bool(track.get("hires"))
    grid = _threshold_grid() if hires else [(base.get("change_threshold", 0.12), base.get("hit_threshold", 0.9))]

    best = None          # (error, settings, count)
    for lo_x, hi_x in LENGTH_SHAPES:
        mn = min(max(_round_to(lo_x * want_len, 0.1), MIN_SEG_RANGE[0]), MIN_SEG_RANGE[1])
        mx = min(max(_round_to(hi_x * want_len, 0.5), _round_to(2 * mn, 0.5), 0.5), MAX_SEG_RANGE[1])
        for ct, hit in grid:
            s = {**keep, "change_threshold": ct, "hit_threshold": hit,
                 "min_segment_sec": mn, "max_segment_sec": mx}
            got = _count(track, s, beats_per_bar)
            err = abs(got - want) / want
            # ties: prefer the stricter thresholds (fewer, stronger cuts) and the first length shape
            if best is None or err < best[0] - 1e-12:
                best = (err, s, got)
            if got > want and err > best[0]:
                break           # looser only adds cuts from here (near enough) — next shape
        if best[0] <= GOOD_ENOUGH:
            break

    err, settings, got = best
    notes = [f"Aimed for {want} segments (about {want_len:.1f}s each) — got {got} "
             f"(about {duration / max(got, 1):.1f}s each)."]
    if err > GOOD_ENOUGH:
        notes.append("That's as close as this method gets on this track — "
                     + ("its cuts come from loudness changes and hits, and there aren't enough clear ones."
                        if got < want else "the track has more strong moments than that, even at the strictest "
                                           "settings."))
    if not hires:
        notes.append("This track only has the older coarse analysis, so only the segment lengths were set.")
    notes.append(f"Dials: energy-change threshold {settings['change_threshold']:.2f}, "
                 + (f"big-hit threshold {settings['hit_threshold']:.2f}, " if settings["react_to_hits"] else "")
                 + f"shortest {settings['min_segment_sec']:g}s, longest {settings['max_segment_sec']:g}s.")
    return {"settings": settings, "target": want, "segments": got, "avg_sec": duration / max(got, 1),
            "exact": err <= GOOD_ENOUGH, "notes": notes}
