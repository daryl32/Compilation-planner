"""Helpers for automatic picks: how far a sequential pick may skip ahead,
where in each video its next use should come from (weight F), stable
pseudo-random draws, and the hard pick rules.

Pure functions (no Streamlit). Moved unchanged from the planner page except
that clips-per-block, time ranges and video durations come in as arguments."""

from __future__ import annotations

import hashlib
import math


SKIP_SAFETY = 1.5   # keep this many times a video's expected remaining uses in reserve


def compute_skip_budget(available_windows: int, remaining_blocks: int, weight_position: float,
                        n_videos: int = 1, clips_per_block: float = 1.0) -> int:
    """How many windows past the frontmost an automatic sequential-mode pick may
    consider, instead of always being pinned to the very front.

    A video only appears in its SHARE of the remaining blocks — about
    remaining blocks × clips per block ÷ videos in play. That many windows
    (× SKIP_SAFETY, +1) are kept in reserve so later blocks never run short;
    everything beyond is surplus the pick may skip ahead into. (This used to
    reserve one window for EVERY remaining block of the plan, as if one video
    had to fill them all — so the surplus was almost always 0 and each video
    was used only from its start.)

    weight F (Move through each video in order) > 0 opens the whole surplus;
    the ranking's position term (compute_position_targets) then pulls each pick
    towards the part of the video it should come from — beginning → middle →
    end across the plan. F = 0 keeps the strict front-of-queue behaviour.

    clips_per_block: average clips per block in the plan (the planner keeps it in
    session state as _avg_clips_per_block); never counted below 1.

    Returns the number of EXTRA windows beyond the frontmost — slice with [: this + 1]."""
    if weight_position <= 0:
        return 0
    share = min(1.0, max(1.0, float(clips_per_block)) / max(1, n_videos))
    reserve = int(math.ceil(max(1, remaining_blocks) * share * SKIP_SAFETY)) + 1
    return max(0, int(available_windows) - reserve)


def compute_position_targets(current_block: int, n_blocks: int, confirmed: dict, queues: dict,
                             videos_per_block: float = None, ranges: dict = None,
                             duration_of=None) -> dict:
    """Weight F (sequential mode): where in each video its NEXT use should
    come from, so a video's uses spread beginning → middle → end across the
    plan instead of bunching at its front.

    Each video's usable footage (its effective time range — project override,
    else Media Library range, else the whole video) is split into as many
    equal parts as the video is expected to be used; use #k aims at the
    centre of part k. Expected uses = clips per block (from blocks confirmed
    so far, or videos_per_block before any are) × blocks ÷ videos in play,
    never fewer than the uses already made + 1, so it self-corrects as the
    plan fills in. A use = a block the video appears in (chain links within
    one block count once).

    ranges: {video_id: (start, end)} effective time ranges (the planner's
    effective_time_ranges()); duration_of(video_id) -> seconds, used for videos
    without a range.

    Returns {video_id: (target_abs_sec, part_len_sec)}."""
    used_blocks = {}
    video_block_count = 0
    for i in range(current_block):
        vids_i = {p["video_id"] for p in confirmed.get(i, [])}
        video_block_count += len(vids_i)
        for v in vids_i:
            used_blocks[v] = used_blocks.get(v, 0) + 1
    if current_block > 0 and video_block_count:
        per_block = video_block_count / current_block
    else:
        per_block = max(1.0, float(videos_per_block or 1))
    n_videos = max(1, len(queues))
    expected_uses = per_block * n_blocks / n_videos

    ranges = ranges or {}
    duration_of = duration_of or (lambda _vid: 0.0)
    targets = {}
    for vid in queues:
        r0, r1 = ranges.get(vid) or (0.0, duration_of(vid))
        length = max(r1 - r0, 1e-6)
        used = used_blocks.get(vid, 0)
        n_parts = max(used + 1, int(round(expected_uses)), 1)
        part_len = length / n_parts
        targets[vid] = (r0 + (used + 0.5) * part_len, part_len)
    return targets


def position_closeness(c: dict, targets: dict, default_duration: float) -> float:
    """Weight F's score for one candidate: 1.0 when the clip's midpoint is at
    its video's target, falling to 0 two parts away (see
    compute_position_targets). 0 when the video has no target."""
    target = targets.get(c["video_id"])
    if not target:
        return 0.0
    start = c.get("scene_start_sec", 0.0) + c.get("window_offset_sec", c.get("offset_sec", 0.0))
    duration = c.get("window_duration_sec") or default_duration
    mid = start + duration / 2
    return max(0.0, 1.0 - abs(mid - target[0]) / (2 * target[1]))


def select_autofill_picks(ranked: list, count: int, allow_same_video: bool) -> list:
    """The sidebar's hard rules for an automatic pick: at most `count` clips
    (1 when split-screen is off), and — unless 'Allow multiple clips from the
    same video' is on — no two from the same video. Walks the weighted ranking
    best-first, skipping a clip only when its video is already used."""
    chosen, used_videos = [], set()
    for c in ranked:
        if len(chosen) >= count:
            break
        if not allow_same_video and c["video_id"] in used_videos:
            continue
        chosen.append(c)
        used_videos.add(c["video_id"])
    return chosen


def _stable_random_unit(seed: int, *parts) -> float:
    """Deterministic pseudo-random value in [0,1), derived from seed + parts.
    Stable across reruns for the SAME inputs (so an unrelated widget click
    never reshuffles already-made auto-fill picks) — changes only if the
    seed itself changes. Not Python's hash(), which is randomized per
    process for strings."""
    h = hashlib.sha256(f"{seed}:{':'.join(str(p) for p in parts)}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def repeat_fallback_videos(picks: list, prev_block_videos: set, avoid_repeat: bool) -> list:
    """Videos among an automatic block's picks that repeat the previous block
    despite 'Avoid repeating previous block' being on — i.e. the fallback was
    used because nothing else could fill the spot. Empty when the rule held."""
    if not avoid_repeat or not prev_block_videos:
        return []
    return sorted({p["video_id"] for p in picks if p["video_id"] in prev_block_videos})
