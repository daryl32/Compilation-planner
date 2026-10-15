"""
Auto-fill scoring, version 2 — used by every mode through the planner's
rank_candidates_weighted when "Improved auto-fill" is on.

What changed from version 1 (and why):

1. Every score is put on the same scale before the sliders weigh it. v1 added
   raw numbers that spread very differently (randomness 0–1 evenly; motion match
   often within 0.1 between candidates), so a small randomness weight could
   outvote a large motion weight. Here each score is standardised across the
   candidates for this decision, and the sliders are relative SHARES: two
   sliders at the same value have the same say, whatever their absolute value.
2. Motion is compared on a fairer scale. v1 scaled motion 0–1 within each video,
   so a calm video's busiest moment counted as 1.0, like an action video's peak.
   Here a clip's motion level is half its percentile within its own video and
   half its percentile across every video in play (percentiles, so one outlier
   scene doesn't squash the rest).
3. Motion is measured for the part of the scene actually used (from the per-
   second motion summary video_ranking keeps), not the whole scene's average.
4. Motion match falls off as a bell curve with a tolerance, so near matches
   stand apart from mediocre ones (v1: a clip 0.3 off still scored 0.7).
5. Balance between videos compares each video's use with its FAIR SHARE of the
   plan (by how much usable footage it has), so it works from the first block.
6. Variety picks randomly among the best few (a seeded Gumbel draw scaled by the
   Variety slider) instead of adding noise to every score — a poor clip can't
   win just by luck. Same seed, same picks.

The "avoid repeating previous block" rule (C) stays a hard rule, applied first.

Pure functions: the planner passes in its own helpers for the parts that need
its state (position targets, stable random numbers).
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right

import numpy as np

# Term ids, their slider and their plain name.
TERMS = {
    "shape": ("weight_shape", "Match the music's shape"),
    "motion": ("weight_motion", "Match the energy level"),
    "balance": ("weight_spread", "Use all videos evenly"),
    "position": ("weight_position", "Move through each video in order"),
}
# Smallest spread used when standardising a score, in that score's own units —
# so tiny differences between near-identical candidates aren't blown up.
SPREAD_FLOOR = {"shape": 0.05, "motion": 0.05, "balance": 0.25, "position": 0.05}
MOTION_TOLERANCE = 0.2      # bell-curve width for the motion match (0–1 motion scale)
VARIETY_SCALE = 0.08        # Variety slider × this = how widely it picks among the best

PRESETS = {
    "Balanced": {"weight_shape": 2.0, "weight_motion": 2.0, "weight_spread": 1.5,
                 "weight_position": 1.0, "weight_random": 0.5},
    "Follow the music": {"weight_shape": 3.0, "weight_motion": 3.5, "weight_spread": 0.8,
                         "weight_position": 0.5, "weight_random": 0.2},
    "Maximum variety": {"weight_shape": 1.0, "weight_motion": 1.0, "weight_spread": 3.0,
                        "weight_position": 0.5, "weight_random": 3.0},
    "Tell each video's story": {"weight_shape": 1.5, "weight_motion": 1.5, "weight_spread": 1.5,
                                "weight_position": 4.0, "weight_random": 0.3},
}


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

class MotionScale:
    """Percentiles of scene motion — across every video in play and within each."""

    def __init__(self, queues: dict):
        self.per_video = {}
        everything = []
        for vid, spans in queues.items():
            vals = sorted(float(s["motion_raw"]) for s in spans if s.get("motion_raw") is not None)
            if vals:
                self.per_video[vid] = vals
                everything += vals
        self.library = sorted(everything)

    @staticmethod
    def _pct(sorted_vals: list, v: float) -> float:
        if not sorted_vals:
            return 0.5
        lo, hi = bisect_left(sorted_vals, v), bisect_right(sorted_vals, v)
        return ((lo + hi) / 2) / len(sorted_vals)

    def level(self, video_id: str, raw: float) -> float:
        """0–1: half within this video, half across every video in play."""
        return 0.5 * self._pct(self.per_video.get(video_id, []), raw) + 0.5 * self._pct(self.library, raw)


def window_motion(summary: dict | None, start: float, end: float) -> float | None:
    """Mean motion over [start, end] (seconds in the video) from a video_ranking
    summary, or None when it has no data there."""
    if not summary or end <= start:
        return None
    starts, ends = summary["starts"], summary["ends"]
    offsets, rates, values = summary["offsets"], summary["rates"], summary["values"]
    total = weight = 0.0
    for k in range(len(starts)):
        s, e = float(starts[k]), float(ends[k])
        a, b = max(start, s), min(end, e)
        if b <= a:
            continue
        rate = float(rates[k])
        i0 = int((a - s) * rate)
        i1 = max(i0 + 1, int(math.ceil((b - s) * rate)))
        lo, hi = int(offsets[k]), int(offsets[k + 1])
        chunk = values[min(lo + i0, hi):min(lo + i1, hi)]
        if len(chunk):
            total += float(np.mean(chunk)) * (b - a)
            weight += b - a
    return total / weight if weight else None


_SUMMARIES = {}   # (video_id, catalogue mtime) -> summary, shared by every session


def summary_for(catalogue_dir, video_id: str):
    """video_ranking's per-second motion summary for a video, cached in memory
    until its catalogue changes. None if there isn't one."""
    try:
        mtime = (catalogue_dir / f"{video_id}.json").stat().st_mtime
    except OSError:
        return None
    key = (video_id, mtime)
    if key not in _SUMMARIES:
        try:
            import video_ranking as VR
            summ = VR.load_summary(catalogue_dir, video_id)
        except Exception:
            summ = None
        if len(_SUMMARIES) > 400:
            _SUMMARIES.clear()
        _SUMMARIES[key] = summ
    return _SUMMARIES[key]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def candidate_window(c: dict, default_duration: float) -> tuple:
    """(start, end) in seconds into the video of the footage this candidate would use."""
    start = c.get("scene_start_sec", 0.0) + c.get("window_offset_sec", c.get("offset_sec", 0.0))
    dur = c.get("window_duration_sec") or default_duration
    if c.get("remaining_sec") is not None and "window_duration_sec" not in c:
        dur = min(dur, c["remaining_sec"])
    return start, start + max(dur, 0.0)


def _standardise(values: list, floor: float) -> list:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    return list((arr - arr.mean()) / max(float(arr.std()), floor))


def rank(candidates: list, *, current_block: int, segments: list, confirmed: dict, queues: dict,
         prev_block_videos: set, weights: dict, avoid_repeat: bool, seed: int,
         position_targets: dict | None, position_score, random_unit, motion_scale: MotionScale,
         window_motion_of) -> tuple:
    """Rank candidates best first. Returns (ranked, why) where why[i] explains
    ranked[i]: {"shares": {term: contribution}, "score", "variety", "repeat",
    "motion_level", "start"}.

    weights: {"weight_shape", "weight_motion", "weight_spread", "weight_position",
    "weight_random"}. position_score(c) -> 0..1 (weight F's closeness);
    random_unit(*parts) -> stable 0..1; window_motion_of(video_id, start, end) ->
    raw motion or None."""
    if not candidates:
        return [], []
    seg = segments[current_block]
    block_dur = seg["end"] - seg["start"]
    target = float(seg.get("intensity", seg.get("energy", 0.5)))

    # Fair share per video: share of the plan so far that matches its share of footage.
    in_play = set(queues) | {c["video_id"] for c in candidates}
    used = {v: 0.0 for v in in_play}
    for b, picks in confirmed.items():
        if b == current_block:
            continue
        for p in picks:
            used[p["video_id"]] = used.get(p["video_id"], 0.0) + p.get("clip_duration_sec", 0.0)
    remaining = {v: sum(s.get("remaining_sec", 0.0) for s in queues.get(v, [])) for v in in_play}
    footage = {v: used.get(v, 0.0) + remaining.get(v, 0.0) for v in in_play}
    total_used = sum(used.values())
    total_footage = sum(footage.values()) or 1.0
    unit = max(block_dur, 1.0)

    raw = {t: [] for t in TERMS}
    levels, windows = [], []
    for c in candidates:
        start, end = candidate_window(c, block_dur)
        windows.append(start)
        m_raw = window_motion_of(c["video_id"], start, end)
        if m_raw is None:
            m_raw = c.get("motion_raw")
        if m_raw is None:
            level = float(c.get("motion_norm", 0.5))      # nothing better known
        else:
            level = motion_scale.level(c["video_id"], float(m_raw))
        levels.append(level)
        raw["shape"].append(max(0.0, float(c.get("score", 0.0))))
        raw["motion"].append(math.exp(-((level - target) / MOTION_TOLERANCE) ** 2))
        fair = (total_used + block_dur) * footage.get(c["video_id"], 0.0) / total_footage
        raw["balance"].append((fair - used.get(c["video_id"], 0.0)) / unit)
        raw["position"].append(position_score(c) if position_targets else 0.0)

    active = {t: float(weights.get(TERMS[t][0], 0.0)) for t in TERMS}
    active = {t: w for t, w in active.items() if w > 0 and max(raw[t]) - min(raw[t]) > 1e-9}
    wsum = sum(active.values())
    z = {t: _standardise(raw[t], SPREAD_FLOOR[t]) for t in active}
    tau = VARIETY_SCALE * float(weights.get("weight_random", 0.0))

    scored = []
    for i, c in enumerate(candidates):
        shares = {t: (active[t] * z[t][i] / wsum) for t in active} if wsum else {}
        score = sum(shares.values())
        variety = 0.0
        if tau > 0:
            u = min(max(random_unit(seed, current_block, c["video_id"], c["scene_id"], round(windows[i], 1)),
                        1e-9), 1 - 1e-9)
            variety = tau * -math.log(-math.log(u))
        is_repeat = bool(avoid_repeat and c["video_id"] in prev_block_videos)
        scored.append((is_repeat, score + variety, i, {
            "shares": shares, "score": score, "variety": variety, "repeat": is_repeat,
            "motion_level": levels[i], "target": target, "start": windows[i],
        }))
    scored.sort(key=lambda t: (t[0], -t[1], t[2]))
    return [candidates[t[2]] for t in scored], [t[3] for t in scored]


def fill_order(n_blocks: int, intensities: list, role_blocks: set) -> list:
    """The order to fill blocks in when they don't depend on each other (non-
    sequential Auto mode): intro/outro blocks first (their tagged clips are
    scarce), then the most intense blocks (high-motion footage is scarce),
    then the rest. Ties keep track order."""
    return sorted(range(n_blocks), key=lambda i: (i not in role_blocks, -float(intensities[i]), i))
