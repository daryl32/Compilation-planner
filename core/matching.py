"""The matcher: finding, ranking and placing clips for each block — Auto mode,
Manual step-through's candidate search and Auto-fill, Choreography's windows.

Moved unchanged from pages/1_Compilation_Planner.py (Phase 1b, Oct 2026)
except that everything the page used to read from session state comes in a
PlanContext — the page builds one from its settings (plan_context()), and
scripts or background jobs (recommendations) can build their own. Same
results: tests/test_golden.py checks every case against the snapshot taken
before the move.

No Streamlit in here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import autofill as AF
import core.picking as CPICK
from core.footage import carve_span, consume_span, get_candidate_spans, has_role, role_scarcity
from core.picking import _stable_random_unit, position_closeness, repeat_fallback_videos, select_autofill_picks
from core.segmentation import clips_for_intensity
from core.shape import find_best_shape_window, get_block_audio_curve


@dataclass
class PlanContext:
    """Everything the matcher needs to know about the plan besides its direct
    arguments. Defaults = a fresh planner session."""
    clips_per_block: float = 1.0                 # average clips per block in the plan (skip budget)
    autofill_v2: bool = True                     # ✨ Improved auto-fill ranking
    reserved_roles: frozenset = frozenset()      # roles kept for their own parts (story map)
    locked_roles: frozenset = frozenset()        # roles whose clips may only go in their blocks…
    intro_outro_exclusive: bool = False          # …when 'Intro/outro clips only in their blocks' is on
    story_in_use: bool = False                   # the story map decides block roles
    clip_count_skip: tuple = ()                  # 'Skip these clip counts'
    time_ranges: dict | Callable = field(default_factory=dict)  # {video_id: (start, end)} (or a function giving it)
    duration_of: Callable = lambda _vid: 0.0                  # video length in seconds
    curves_of: Callable = lambda _vid: {}                     # {scene_id: {"fps", "values"}} per video
    catalogue_dir: Path | None = None            # per-second motion summaries (autofill); None = without them
    why_store: dict | Callable | None = None     # 🔎 Why these picks? — filled in when given (dict or function)
    role_chips: Callable = lambda _c: ""         # how a clip's roles are written in labels


def _skip_budget(c: PlanContext, available_windows, remaining_blocks, weight_position, n_videos=1) -> int:
    return CPICK.compute_skip_budget(available_windows, remaining_blocks, weight_position, n_videos,
                                     clips_per_block=c.clips_per_block)


def _targets(c: PlanContext, current_block, n_blocks, confirmed, queues, videos_per_block=None) -> dict:
    ranges = c.time_ranges() if callable(c.time_ranges) else c.time_ranges
    return CPICK.compute_position_targets(current_block, n_blocks, confirmed, queues, videos_per_block,
                                          ranges=ranges, duration_of=c.duration_of)


def _clips_for(c: PlanContext, intensity, min_clips, max_clips, ramp_start, ramp_end, vary, rng) -> int:
    return clips_for_intensity(intensity, min_clips, max_clips, ramp_start, ramp_end, vary, rng,
                               skip=c.clip_count_skip)


def format_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def soft_reserve(items: list, role, min_keep: int = 1, ctx: "PlanContext" = None) -> list:
    """Keep role clips for their parts (a preference, not a rule): drop clips
    carrying a reserved role this block isn't for — unless that would leave
    fewer than min_keep to choose from."""
    pctx = ctx or PlanContext()
    reserved = set(pctx.reserved_roles or ())
    if not reserved:
        return items
    keep = [c for c in items if has_role(c, role) or not (set(c.get("roles") or []) & reserved)]
    return keep if len(keep) >= max(1, min_keep) else items


def prefer_role(items: list, role, sequential: bool = False, reach: int = None) -> list:
    """The items with this block's role if there are any, else all of them.
    reach (sequential mode): only role items among the first reach+1 count —
    a role clip is never taken out of order beyond the normal skip allowance."""
    if not role:
        return items
    pool = items[:reach + 1] if (sequential and reach is not None) else items
    matches = [c for c in pool if has_role(c, role)]
    return matches or items


def role_locked_keys(queues: dict, role_filter, ctx: "PlanContext" = None) -> set:
    """'Intro/outro clips only in their blocks' (weights panel; needs Prefer
    Intro/Outro on): the (video, scene) keys a block may NOT use — intro-marked
    clips outside intro blocks, outro-marked clips outside outro blocks. A hard
    rule; videos with no marked clips are unaffected. In sequential mode these are
    skipped over (not consumed), like rejected clips, so they stay for their blocks."""
    pctx = ctx or PlanContext()
    locked = set(pctx.locked_roles or ())
    if not (pctx.intro_outro_exclusive and locked):
        return set()
    out = set()
    for spans in queues.values():
        for sp in spans:
            roles = set(sp.get("roles") or []) & locked
            if roles and role_filter not in roles:
                out.add((sp["video_id"], sp["scene_id"]))
    return out


def excluded_for_block(queues: dict, global_excluded: set, role_filter, ctx: "PlanContext" = None) -> set:
    """Rejected clips plus the ones role_locked_keys keeps out of this block."""
    pctx = ctx or PlanContext()
    locked = role_locked_keys(queues, role_filter, ctx=pctx)
    return (set(global_excluded) | locked) if locked else global_excluded


def find_shape_candidates(
    segment: dict, queues: dict, sequential: bool, excluded_here: set, global_excluded: set,
    audio_curve: np.ndarray, max_matches: int, restrict_to_front: bool = False,
    remaining_blocks: int = 1, weight_position: float = 0.0,
    position_targets: dict = None, weight_shape: float = 1.0, exclude_videos: set = None,
    only_role: str = None,
    ctx: "PlanContext" = None,
) -> list[dict]:
    """For every currently-available span at least as long as this block,
    slide a window of the block's exact duration across EVERY frame position
    (see find_best_shape_window — vectorized, so checking every position
    costs less than the old stepped search did), and return the best-scoring
    window PER SPAN, sorted best-first and capped at max_matches overall.
    Scenes shorter than
    the block are excluded entirely (per spec — no chaining in this mode).
    Scenes with no motion_curve data are skipped with a note, not silently
    dropped, so missing-backfill is visible rather than just "fewer candidates".

    restrict_to_front: when True AND sequential, the search is confined to a
    budgeted range near the span's own front instead of sliding across the
    whole span — used by Auto-fill ALL, which is fully automatic. Sequential
    mode already restricts each video to a single span via get_candidate_spans,
    but the shape search within that span can otherwise land on a non-front
    position; carving it then discards everything before it, and doing that
    automatically across every remaining block can silently leave later
    blocks with no footage left to pick from.

    remaining_blocks / weight_position: feed compute_skip_budget (see there) to
    size that range — how many more block-length windows past the front this
    span's own surplus footage (relative to blocks still left overall) can
    safely spare, scaled by weight F.

    position_targets (from compute_position_targets) / weight_shape: when given
    with weight_position > 0, the window chosen WITHIN that budgeted range is
    pulled toward the video's target part (weight F) as well as by shape —
    F and A traded off in the same proportion the ranking itself uses. 0 (the defaults) collapses to exactly
    the front window, same as a hard restriction. The live candidate grid
    (manual review) always calls this with restrict_to_front's default
    False — skipping ahead there is a deliberate, informed choice, not
    something this budget needs to manage."""
    pctx = ctx or PlanContext()
    block_duration = segment["end"] - segment["start"]
    results = []
    missing_curve_videos = set()
    video_curves_cache = {}  # video_id -> {scene_id: curve}, fetched at most once per call, not per span

    search_spans = get_candidate_spans(queues, sequential, global_excluded, min_duration=block_duration)
    span_limit = {}   # (video, scene) -> seconds of this span the search may reach (restrict_to_front)
    if restrict_to_front and sequential:
        # The skip budget counts windows across the WHOLE video (not just its front
        # scene), and later scenes within it are searched too — picking one discards
        # what's before it, as any sequential skip does.
        extra = []
        for front in list(search_spans):
            vid = front["video_id"]
            spans_v = [sp for sp in queues.get(vid, [])
                       if sp["remaining_sec"] > 0.05 and (vid, sp["scene_id"]) not in (global_excluded or set())]
            windows = sum(int(sp["remaining_sec"] // block_duration) for sp in spans_v)
            budget_sec = (_skip_budget(pctx, max(1, windows), remaining_blocks, weight_position, len(queues)) + 1) \
                * block_duration
            reach = budget_sec
            # Never further than one part past where this use should land (weight F's target).
            tgt = (position_targets or {}).get(vid)
            if tgt:
                front_abs = front["scene_start_sec"] + front["offset_sec"]
                reach = min(reach, max(block_duration, tgt[0] + tgt[1] - front_abs))
            started = False
            for sp in spans_v:
                if sp is front:
                    started = True
                if not started:
                    continue
                if reach <= 1e-6:
                    break
                span_limit[(vid, sp["scene_id"])] = reach
                if sp is not front and sp["remaining_sec"] >= block_duration - 1e-6:
                    extra.append(sp)
                reach -= sp["remaining_sec"]
        search_spans = list(search_spans) + extra

    for span in search_spans:
        key = (span["video_id"], span["scene_id"])
        if key in excluded_here or span["remaining_sec"] < block_duration - 1e-6:
            continue
        if only_role and not has_role(span, only_role):
            continue
        if exclude_videos and span["video_id"] in exclude_videos:
            continue

        if span["video_id"] not in video_curves_cache:
            video_curves_cache[span["video_id"]] = pctx.curves_of(span["video_id"])
        curve = video_curves_cache[span["video_id"]].get(span["scene_id"])
        if not curve or curve.get("values") is None or len(curve["values"]) == 0:
            missing_curve_videos.add(span["video_id"])
            continue

        fps = curve["fps"]
        values = np.asarray(curve["values"], dtype=float)
        window_frames = max(1, round(block_duration * fps))

        span_start_frame = round(span["offset_sec"] * fps)
        span_end_frame = round((span["offset_sec"] + span["remaining_sec"]) * fps)

        if restrict_to_front and sequential:
            # Confine the search to a budgeted range near the front rather than
            # either the single front window or the whole span — still scores
            # the actual best-matching position WITHIN that budgeted range
            # properly (needed for weight A / reporting).
            reach = span_limit.get(key, block_duration)
            search_end_frame = min(span_end_frame,
                                   span_start_frame + max(window_frames, round(max(reach, block_duration) * fps)))
        else:
            search_end_frame = span_end_frame

        bias = None
        target = (position_targets or {}).get(span["video_id"])
        if restrict_to_front and sequential and weight_position > 0 and target:
            n_pos = (search_end_frame - span_start_frame) - window_frames + 1
            if n_pos > 1:
                mids = (span["scene_start_sec"]
                        + (span_start_frame + np.arange(n_pos)) / fps + block_duration / 2)
                closeness = np.clip(1.0 - np.abs(mids - target[0]) / (2 * target[1]), 0.0, 1.0)
                bias = (weight_position / max(weight_shape, 0.1)) * closeness

        # Vectorized: scores every frame position in the search range at once (see
        # find_best_shape_window), not just every SHAPE_SEARCH_STEP_SEC —
        # faster AND exhaustive rather than a speed/precision compromise.
        best_score, best_offset = find_best_shape_window(
            values[span_start_frame:search_end_frame], window_frames, audio_curve, bias=bias
        )
        best_start_frame = span_start_frame + best_offset
        # Frame rounding can put the window a fraction outside the span; keep it inside.
        win_start = min(max(best_start_frame / fps, span["offset_sec"]),
                        span["offset_sec"] + span["remaining_sec"] - block_duration)

        results.append({
            "video_id": span["video_id"],
            "scene_id": span["scene_id"],
            "scene_start_sec": span["scene_start_sec"],  # + window_offset_sec = absolute position in the video
            "window_offset_sec": win_start,
            "window_duration_sec": block_duration,
            "score": round(best_score, 3),
            "motion_norm": span["motion_norm"],
            "curve_slice": np.array(values[best_start_frame:best_start_frame + window_frames]),  # own copy — not tied to the curve cache's file
            "fps": fps,
            "tags": span["tags"],
            "thumbnail": span["thumbnail"],
            "intro_candidate": span.get("intro_candidate", False),
            "outro_candidate": span.get("outro_candidate", False),
            "roles": span.get("roles", []),
        })

    results.sort(key=lambda r: -r["score"])
    # A scene whose middle was used by an earlier block (non-sequential mode) is
    # left as TWO pieces with the same (video, scene) id. Picks and checkboxes
    # are keyed by that id, so offering both made one Auto-fill pick confirm
    # two clips. Keep only the better-scoring piece of each scene.
    seen, unique = set(), []
    for r in results:
        key = (r["video_id"], r["scene_id"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:max_matches], missing_curve_videos


def widen_for_repeat_rule(cands: list, prev_block_videos: set, need: int, search) -> list:
    """The shape search keeps only its top matches, which can all come from the
    previous block's video(s) even when other videos have footage. When fewer
    than `need` non-repeating videos are among cands, search again WITHOUT the
    previous block's videos and add the best clip of each new video until there
    are enough. search(exclude_videos) -> candidate list, best first."""
    if not prev_block_videos or need <= 0:
        return cands
    have = {c["video_id"] for c in cands if c["video_id"] not in prev_block_videos}
    if len(have) >= need:
        return cands
    existing = {cand_key(c) for c in cands}
    extra = []
    for c in search(set(prev_block_videos) | have):
        if len(have) >= need:
            break
        if c["video_id"] in have or cand_key(c) in existing:
            continue
        extra.append(c)
        have.add(c["video_id"])
    return cands + extra


def cand_key(c: dict) -> str:
    """Widget/pick key for a Manual step-through candidate. Shape-search
    candidates are one per scene; ✨ results carry their own _key so they can
    sit next to a candidate from the same scene."""
    return c.get("_key") or f"{c['video_id']}_{c['scene_id']}"


def rank_candidates_weighted(
    candidates: list, current_block: int, segments: list, confirmed: dict, queues: dict,
    prev_block_videos: set, weight_shape: float, weight_random: float,
    avoid_repeat: bool, weight_spread: float, weight_motion: float, random_seed: int,
    weight_position: float = 0.0, sequential: bool = False, videos_per_block: float = None,
    ctx: "PlanContext" = None,
) -> list:
    """Re-ranks candidates by a blended score. Used by Manual step-through's
    Auto-fill button (candidates already found by the shape-score search; the
    grid itself always stays sorted by raw shape score for manual browsing)
    and by Auto mode's own automatic picking in fill_slot (candidates are
    raw spans with no shape score at all — see A below).

    A: shape match, clamped to >= 0 (an anti-correlated clip is never
       preferred over "no match" just because randomness happens to favour
       it). Candidates with no "score" field (Auto mode's spans) contribute
       0 here regardless of weight_shape — there's no per-frame curve
       comparison in Auto mode for this weight to act on.
    B: a STABLE pseudo-random value — same seed+block+candidate always draws
       the same number, so a rerun from an unrelated click never reshuffles
       picks already made; change the seed to deliberately reroll.
    C (avoid_repeat, on/off — a hard rule, not a weight): when on, every
       candidate from a video NOT used in the immediately preceding block
       ranks above every candidate from one that was; the weights below only
       order clips within each of those two groups. A repeat therefore only
       comes out on top when no other video is available (the fallback —
       callers check for it with repeat_fallback_videos and warn).
    D: blends "hasn't been used much yet" with "still has a lot of footage
       remaining" — pulls long, under-used videos back into rotation across
       many block decisions, rather than a long source being used once early
       and then left untouched.
    E: motion intensity match — how closely the clip's normalised motion level
       matches this block's intensity. 1.0 = perfect match, 0.0 = opposite
       end of the scale. Complementary to A (shape): A rewards the same
       PATTERN of rises/falls; E rewards the right overall LEVEL of activity.
    F: sequential mode only — how close the clip sits to where this video's
       next use should come from (see compute_position_targets), so each
       video gets a beginning, middle and end. 0 outside sequential mode.

    Returns candidates sorted by combined score, descending. Does not
    mutate or reorder the input list itself.

    With "✨ Improved auto-fill" on (the default) the ranking is autofill.rank
    instead — see autofill.py for what it changes and why."""
    pctx = ctx or PlanContext()
    if not candidates:
        return []
    if pctx.autofill_v2:
        return _rank_v2(candidates, current_block, segments, confirmed, queues, prev_block_videos,
                        weight_shape, weight_random, avoid_repeat, weight_spread, weight_motion,
                        random_seed, weight_position, sequential, videos_per_block, ctx=pctx)

    curr_duration = segments[current_block]["end"] - segments[current_block]["start"]
    block_intensity = segments[current_block].get("intensity", segments[current_block].get("energy", 0.5))

    video_ids = {c["video_id"] for c in candidates}
    used_by_video = {
        vid: sum(p["clip_duration_sec"] for i in range(current_block) for p in confirmed.get(i, [])
                if p["video_id"] == vid)
        for vid in video_ids
    }
    remaining_by_video = {vid: sum(span["remaining_sec"] for span in queues.get(vid, [])) for vid in video_ids}
    max_used = max(used_by_video.values(), default=0.0) or 1.0
    max_remaining = max(remaining_by_video.values(), default=0.0) or 1.0
    position_targets = (
        _targets(pctx, current_block, len(segments), confirmed, queues, videos_per_block)
        if sequential and weight_position > 0 else {}
    )

    scored = []
    for c in candidates:
        A = max(0.0, c.get("score", 0.0))  # Auto mode's spans carry no shape score — A is simply 0 there
        B = _stable_random_unit(random_seed, current_block, c["video_id"], c["scene_id"])

        is_repeat = bool(avoid_repeat and current_block > 0 and c["video_id"] in prev_block_videos)

        norm_used = used_by_video[c["video_id"]] / max_used
        norm_remaining = remaining_by_video[c["video_id"]] / max_remaining
        D = (1 - norm_used) * 0.5 + norm_remaining * 0.5

        # E: 1 - |clip motion_norm - block intensity|, clamped [0, 1]
        E = max(0.0, 1.0 - abs(c.get("motion_norm", 0.5) - block_intensity))

        F = position_closeness(c, position_targets, curr_duration) if position_targets else 0.0

        combined = (weight_shape * A + weight_random * B
                    + weight_spread * D + weight_motion * E + weight_position * F)
        scored.append((is_repeat, combined, c))

    # Hard rule first (non-repeats before repeats), then the blended score.
    scored.sort(key=lambda t: (t[0], -t[1]))
    return [c for _, _, c in scored]


WHY_MAX = 5000


def why_key(block: int, video_id: str, scene_id, start: float) -> tuple:
    return (int(block), str(video_id), scene_id, round(float(start), 1))


def _rank_v2(candidates, current_block, segments, confirmed, queues, prev_block_videos,
             weight_shape, weight_random, avoid_repeat, weight_spread, weight_motion,
             random_seed, weight_position, sequential, videos_per_block, ctx: "PlanContext" = None) -> list:
    """autofill.rank with this page's helpers; remembers why each of the top
    candidates scored as it did, for 🔎 Why these picks?."""
    pctx = ctx or PlanContext()
    position_targets = (
        _targets(pctx, current_block, len(segments), confirmed, queues, videos_per_block)
        if sequential and weight_position > 0 else None
    )
    seg_dur = segments[current_block]["end"] - segments[current_block]["start"]
    ranked, why = AF.rank(
        candidates, current_block=current_block, segments=segments, confirmed=confirmed, queues=queues,
        prev_block_videos=prev_block_videos or set(),
        weights={"weight_shape": weight_shape, "weight_motion": weight_motion,
                 "weight_spread": weight_spread, "weight_position": weight_position,
                 "weight_random": weight_random},
        avoid_repeat=avoid_repeat, seed=random_seed, position_targets=position_targets,
        position_score=lambda c: position_closeness(c, position_targets or {}, seg_dur),
        random_unit=_stable_random_unit, motion_scale=AF.MotionScale(queues),
        window_motion_of=lambda vid, a, b: (AF.window_motion(AF.summary_for(pctx.catalogue_dir, vid), a, b)
                                         if pctx.catalogue_dir is not None else None),
    )
    store = pctx.why_store() if callable(pctx.why_store) else pctx.why_store
    if store is None:
        store = {}
    if len(store) > WHY_MAX:
        store.clear()
    for i, (c, w) in enumerate(zip(ranked[:8], why[:8])):
        store[why_key(current_block, c["video_id"], c["scene_id"], w["start"])] = {
            **w, "rank": i + 1, "of": len(ranked), "video_id": c["video_id"], "scene_id": c["scene_id"],
        }
    return ranked


CHOR_MAX_OPTIONS_PER_VIDEO = 300  # safety cap on tiled dropdown entries — see docstring below


def get_available_clips_for_video(queues: dict, video_id: str, block_duration: float,
                                   sequential: bool, global_excluded: set,
                                   max_options: int = CHOR_MAX_OPTIONS_PER_VIDEO,
                                   role_filter: str = None, role_reach=None, ctx: "PlanContext" = None) -> list[dict]:
    """Every selectable block_duration-long window across ALL of video_id's still-
    available footage — not just the front of each span.  Each currently-available
    span (one per not-yet-fully-consumed scene) is pre-trimmed into consecutive,
    non-overlapping block_duration-sized windows: offset, offset+D, offset+2D, ...
    for as many whole windows as fit in that span.  Every window becomes its own
    selectable entry, so the WHOLE of a long, untouched scene is reachable from the
    dropdown — not only its current front position.

    A trailing remainder shorter than block_duration is left out (not enough
    footage for one more whole window); it stays in the span for a future block
    of a different duration to use.

    role_filter: a role id ("intro", "slow", "action", "outro" or your own) — if given, the result is
    narrowed to windows whose scene carries that flag, but ONLY when at least one
    such window exists for this video; otherwise every window is returned
    unfiltered. This is deliberately opportunistic rather than a hard requirement,
    so a video with no tagged intro/outro clips is never silently locked out of
    a block it could otherwise fill.

    Returns a list of window dicts (span fields plus this window's own 'offset_sec',
    'abs_start', and a timecode 'label'), sorted by absolute start time.  Capped at
    max_options entries total (across all spans) purely for UI/performance safety on
    a very long, uncut scene — raise CHOR_MAX_OPTIONS_PER_VIDEO if you need more.

    Sequential mode: the queue already contains only forward spans after replay
    (carve_span and skip_span enforce the ordering invariant), so no additional
    filtering happens here — every span present is a valid future choice, and
    clips already used are simply absent from the queue."""
    pctx = ctx or PlanContext()
    spans = queues.get(video_id, [])
    result = []
    for span in spans:
        key = (span["video_id"], span["scene_id"])
        if key in global_excluded:
            continue
        if span["remaining_sec"] < block_duration - 1e-6:
            continue
        span_start = span["offset_sec"]
        span_end = span["offset_sec"] + span["remaining_sec"]
        n_windows = int((span_end - span_start + 1e-6) // block_duration)
        for w in range(n_windows):
            if len(result) >= max_options:
                break
            win_offset = span_start + w * block_duration
            abs_start = span["scene_start_sec"] + win_offset
            abs_end = abs_start + block_duration
            role_tag = f"  {pctx.role_chips(span)}" if span.get("roles") else ""
            label = (f"Scene {span['scene_id']}  {format_mmss(abs_start)} – {format_mmss(abs_end)}"
                     f"  motion {span['motion_norm']:.2f}{role_tag}")
            result.append({**span, "offset_sec": win_offset, "abs_start": abs_start, "label": label})
        if len(result) >= max_options:
            break
    result = sorted(result, key=lambda s: s["abs_start"])

    if role_filter:
        reach = None
        if sequential and role_reach is not None and not pctx.intro_outro_exclusive:
            reach = role_reach(len(result))
        return prefer_role(result, role_filter, sequential, reach)
    return result


MAX_CHAIN_LINKS = 20  # safety cap against pathological tiny-leftover loops


def fill_slot(seg_energy: float, needed_sec: float, queues: dict, sequential: bool,
              excluded_here: set, global_excluded: set, avoid_video_ids: set,
              min_clip_len_sec: float = 1.0, role_filter: str = None,
              weighting: dict = None, ctx: "PlanContext" = None) -> tuple:
    """Fill one slot's full duration with a CHAIN of clips rather than a
    single clip — the moment one clip's footage runs out, immediately cut to
    the next best-matching available clip for the remaining time, so
    playback never freezes or holds a frame. Returns (chain, unfilled_sec).

    excluded_here: scenes excluded from THIS segment only (Swap / Remove).
    global_excluded: scenes permanently banned from the whole plan (Reject).

    min_clip_len_sec: ported from the VSE add-on's minimum-clip-size rule —
    once only a sliver shorter than this remains, don't cut to a new clip
    just to cover it. Instead extend the PREVIOUS clip in the chain to cover
    the gap (reading slightly past where scene detection drew the boundary,
    into footage from the same source that's already playing) — this is a
    seamless continuation of the same shot, not a new cut, so it's never
    itself a sub-threshold fragment on screen.

    role_filter: a role id ("intro", "slow", "action", "outro" or your own) when this segment is
    a marked intro/outro block — narrows the candidate pool to scenes
    carrying that flag, but ONLY when at least one such scene is actually
    available at this point in the chain; otherwise every candidate is
    considered as usual. Checked fresh on every link of the chain (not just
    the first), so a role-tagged clip that runs out partway through still
    lets the rest of the chain fall back to normal matching rather than
    failing to fill the slot.

    weighting: {"seg_idx", "segments", "confirmed", "prev_block_videos",
    "weight_shape", "weight_random", "avoid_repeat", "weight_spread",
    "weight_motion", "seed"} — when given AND this isn't a role_filter block,
    the pick is the top result of rank_candidates_weighted instead of the
    plain closest-motion-match. None (or a role_filter block) keeps the
    original simple behaviour — this is the "intro/outro blocks aren't
    included" rule: those stay on the plain motion match regardless of
    whether weighting is otherwise in effect."""
    pctx = ctx or PlanContext()
    chain = []
    remaining = needed_sec
    last_used = None  # avoid an immediate cut back into the same scene it just came from

    while remaining > 0.05 and len(chain) < MAX_CHAIN_LINKS:
        if chain and remaining < min_clip_len_sec:
            chain[-1]["clip_duration_sec"] = round(chain[-1]["clip_duration_sec"] + remaining, 3)
            remaining = 0.0
            break

        pool = [
            s for s in get_candidate_spans(queues, sequential, global_excluded)
            if (s["video_id"], s["scene_id"]) not in excluded_here
        ]
        if not pool:
            break

        pool = soft_reserve(prefer_role(pool, role_filter), role_filter, ctx=pctx)

        preferred = [s for s in pool if s["video_id"] not in avoid_video_ids
                     and (s["video_id"], s["scene_id"]) != last_used]
        candidates = preferred or [s for s in pool if (s["video_id"], s["scene_id"]) != last_used] or pool

        if weighting and (not role_filter or pctx.story_in_use):
            weight_position = weighting.get("weight_position", 0.0)
            ranked_pool = candidates
            if sequential and weight_position > 0:
                # Weight F: also offer windows further into each video, as far as
                # the same skip budget allows (compute_skip_budget) — picking one
                # discards the footage before it, like any sequential skip.
                remaining_blocks = len(weighting["segments"]) - weighting["seg_idx"]
                ranked_pool = list(candidates)
                for vid in {c["video_id"] for c in candidates}:
                    windows = get_available_clips_for_video(queues, vid, remaining, sequential, global_excluded, ctx=pctx)
                    skip = _skip_budget(pctx, len(windows), remaining_blocks, weight_position, len(queues))
                    for w in windows[1:skip + 1]:
                        ranked_pool.append({**w, "_skip_window": True})
                # A role block: role clips anywhere within that in-order reach count too.
                ranked_pool = soft_reserve(prefer_role(ranked_pool, role_filter), role_filter, ctx=pctx)
            ranked = rank_candidates_weighted(
                ranked_pool, weighting["seg_idx"], weighting["segments"], weighting["confirmed"],
                queues, weighting["prev_block_videos"], weighting["weight_shape"],
                weighting["weight_random"], weighting["avoid_repeat"],
                weighting["weight_spread"], weighting["weight_motion"], weighting["seed"],
                weight_position=weight_position, sequential=sequential,
                videos_per_block=weighting.get("videos_per_block"),
            ctx=pctx)
            best_span = ranked[0]
        else:
            if weighting and weighting.get("avoid_repeat"):
                _fresh = [s for s in candidates if s["video_id"] not in weighting["prev_block_videos"]]
                candidates = _fresh or candidates
            best_span = min(candidates, key=lambda s: abs(s["motion_norm"] - seg_energy))
        if weighting is not None and repeat_fallback_videos(
                [best_span], weighting.get("prev_block_videos", set()), weighting.get("avoid_repeat")):
            weighting.setdefault("repeat_fallbacks", set()).add(weighting["seg_idx"])
        if best_span.get("_skip_window"):
            pick = carve_span(queues, best_span["video_id"], best_span["scene_id"],
                              best_span["offset_sec"], remaining, sequential=True)
        else:
            pick = consume_span(best_span, remaining)
        chain.append(pick)
        remaining -= pick["clip_duration_sec"]
        last_used = (pick["video_id"], pick["scene_id"])

        # The candidate's own available footage can itself be shorter than
        # min_clip_len_sec even when plenty of time is still needed overall —
        # fold any such fragment into the previous link rather than showing
        # it as its own brief cut. Repeats in case that now leaves the
        # (extended) previous link short too.
        while len(chain) > 1 and chain[-1]["clip_duration_sec"] < min_clip_len_sec:
            tail = chain.pop()
            chain[-1]["clip_duration_sec"] = round(chain[-1]["clip_duration_sec"] + tail["clip_duration_sec"], 3)

    # Post-chain cleanup: if the LAST link is still shorter than min_clip_len_sec
    # after the loop exits (e.g. a final span that was just barely under the threshold
    # and wasn't caught mid-loop), fold it into the preceding link rather than
    # leaving a very short trailing clip at the end of the block.
    while len(chain) > 1 and chain[-1]["clip_duration_sec"] < min_clip_len_sec:
        tail = chain.pop()
        chain[-1]["clip_duration_sec"] = round(chain[-1]["clip_duration_sec"] + tail["clip_duration_sec"], 3)

    return chain, max(0.0, remaining)


def _auto_shape_picks(i: int, seg: dict, n_picks: int, queues: dict, sequential: bool, excluded_here: set,
                      global_excluded: set, role_filter, segments: list, confirmed: dict,
                      prev_block_videos: set, allow_same_video: bool, weighting: dict, track: dict,
                      max_matches: int, repeat_fallbacks, ctx: "PlanContext" = None) -> list:
    """Auto mode's picks for one block — Manual step-through's Auto-fill ALL, per block:
    shape-scored windows (only within the in-order reach in sequential mode), the
    block's story role preferred, the never-repeat rule widened for, ranked by the
    Auto-fill weights, and carved out of the queues. Returns the placed picks
    (possibly fewer than n_picks)."""
    pctx = ctx or PlanContext()
    w = weighting
    wp = float(w.get("weight_position", 0.0)) if sequential else 0.0
    audio_curve = get_block_audio_curve(track, seg["start"], seg["end"])
    pos_targets = (_targets(pctx, i, len(segments), confirmed, queues, n_picks)
                   if sequential and wp > 0 else None)
    search = dict(restrict_to_front=sequential, remaining_blocks=len(segments) - i, weight_position=wp,
                  position_targets=pos_targets, weight_shape=w["weight_shape"])
    cands, _ = find_shape_candidates(seg, queues, sequential, excluded_here, global_excluded, audio_curve,
                                     max(int(max_matches), n_picks * 3), **search, ctx=pctx)
    if role_filter and not any(has_role(c, role_filter) for c in cands):
        role_c, _ = find_shape_candidates(seg, queues, sequential, excluded_here, global_excluded, audio_curve,
                                          max(int(max_matches), n_picks * 3), only_role=role_filter, **search, ctx=pctx)
        cands = role_c or cands
    cands = soft_reserve(prefer_role(cands, role_filter), role_filter, ctx=pctx)
    if not cands:
        return []
    if w.get("avoid_repeat"):
        cands = widen_for_repeat_rule(
            cands, prev_block_videos, 1 if allow_same_video else n_picks,
            lambda ex: find_shape_candidates(seg, queues, sequential, excluded_here, global_excluded, audio_curve,
                                             max(n_picks, 1), exclude_videos=ex, **search, ctx=pctx)[0])
    ranked = rank_candidates_weighted(
        cands, i, segments, confirmed, queues, prev_block_videos,
        w["weight_shape"], w["weight_random"], w["avoid_repeat"], w["weight_spread"],
        w["weight_motion"], w["seed"], weight_position=wp, sequential=sequential, videos_per_block=n_picks, ctx=pctx)
    chosen = select_autofill_picks(ranked, n_picks, allow_same_video)
    order = chosen + [c for c in ranked if all(c is not x for x in chosen)]
    picks, used = [], set()
    for c in order:
        if len(picks) >= n_picks:
            break
        if not allow_same_video and c["video_id"] in used:
            continue
        try:
            picks.append(carve_span(queues, c["video_id"], c["scene_id"], c["window_offset_sec"],
                                    c["window_duration_sec"], sequential))
        except ValueError:
            continue
        used.add(c["video_id"])
    if repeat_fallbacks is not None and repeat_fallback_videos(picks, prev_block_videos, w.get("avoid_repeat")):
        repeat_fallbacks.add(i)
    return picks


def match_scenes_to_track(
    segments: list[dict],
    queues: dict,
    min_clips: int,
    max_clips: int,
    ramp_start: float,
    ramp_end: float,
    allow_same_video: bool,
    sequential: bool,
    segment_exclusions: dict = None,
    segment_clip_reduction: dict = None,
    global_excluded: set = None,
    vary_count: bool = False,
    seed: int = 0,
    min_clip_len_sec: float = 1.0,
    intro_blocks: set = None,
    outro_blocks: set = None,
    weight_shape: float = 0.0,
    weight_random: float = 0.0,
    avoid_repeat: bool = False,
    weight_spread: float = 0.0,
    weight_motion: float = 0.0,
    autofill_seed: int = 42,
    weight_position: float = 0.0,
    repeat_fallbacks: set = None,

    block_roles: list = None,
    track: dict = None,
    shape_matches: int = 12,
    ctx: "PlanContext" = None,
) -> tuple:
    """repeat_fallbacks: if a set is given, 0-based indices of blocks where
    'Avoid repeating previous block' had to fall back to a repeat are added to it.

    Each slot in a segment is filled to its FULL duration with a chain of
    one or more clips (see fill_slot) — footage running out mid-slot never
    produces a freeze or gap, it just cuts to the next best match. Leftover
    footage from any trimmed clip goes back into its queue for later reuse.
    In sequential mode, a video's own scenes are only ever drawn in their
    original chronological order.

    segment_exclusions (Swap / Remove) apply only to the specific segment
    they were clicked on; global_excluded (Reject) applies to every segment,
    permanently removing that scene from the whole plan.

    intro_blocks / outro_blocks: 0-based segment indices marked to prefer
    intro/outro-tagged clips (see fill_slot's role_filter) — a block in both
    sets is treated as intro. Empty/None means no preference anywhere,
    identical to this feature not existing.

    weight_*/autofill_seed: same Auto-fill weighting used elsewhere in the
    app, reused here via rank_candidates_weighted instead of the plain
    closest-motion-match (see fill_slot's weighting param) — EXCEPT for
    intro/outro blocks, which always use the plain match regardless of these
    weights, same as before this existed. Note these do NOT cleanly degrade
    to the old plain-motion-match behaviour if every weight is manually set
    to 0 — every candidate would score an exact tie, and sort() breaks ties
    by candidate order rather than by motion closeness. The practical
    default (weight_motion=1.0) avoids this in normal use."""
    pctx = ctx or PlanContext()
    segment_exclusions = segment_exclusions or {}
    segment_clip_reduction = segment_clip_reduction or {}
    global_excluded = global_excluded or set()
    intro_blocks = intro_blocks or set()
    outro_blocks = outro_blocks or set()
    rng = random.Random(seed)
    timeline = []
    clip_count_shortfall = 0
    duration_shortfall_sec = 0.0
    confirmed_so_far: dict = {}  # seg_idx -> flat list of every chain-link pick made so far

    # Clip counts are drawn in track order whatever order blocks are filled in,
    # so the same seed gives the same counts.
    base_counts = [_clips_for(pctx, seg.get("intensity", seg["energy"]), min_clips, max_clips,
                                       ramp_start, ramp_end, vary_count, rng) for seg in segments]
    # ✨ Improved auto-fill (not sequential mode, where each video must be used in
    # order): fill intro/outro blocks, then the most intense blocks, first — so the
    # scarce tagged and high-motion footage goes where it's needed most.
    if pctx.autofill_v2 and not sequential:
        _role_blocks = ({i for i, r in enumerate(block_roles) if r} if block_roles is not None
                        else (intro_blocks | outro_blocks))
        order = AF.fill_order(len(segments), [s.get("intensity", s["energy"]) for s in segments],
                              _role_blocks, block_priority=role_scarcity(block_roles, segments, queues))
    else:
        order = range(len(segments))
    by_index = {}

    for i in order:
        seg = segments[i]
        seg_duration = seg["end"] - seg["start"]
        target = seg.get("intensity", seg["energy"])
        base_n_picks = base_counts[i]
        n_picks = max(0, base_n_picks - segment_clip_reduction.get(i, 0))
        excluded_here = segment_exclusions.get(i, set())
        if block_roles is not None:
            role_filter = block_roles[i]
        else:
            role_filter = "intro" if i in intro_blocks else ("outro" if i in outro_blocks else None)
        # Neighbours already filled (with out-of-order filling that can be the block after, too).
        prev_block_videos = {p["video_id"] for j in (i - 1, i + 1) if 0 <= j < len(segments)
                             for p in confirmed_so_far.get(j, [])}
        weighting = {
            "seg_idx": i, "segments": segments, "confirmed": confirmed_so_far,
            "prev_block_videos": prev_block_videos,
            "weight_shape": weight_shape, "weight_random": weight_random,
            "avoid_repeat": avoid_repeat, "weight_spread": weight_spread,
            "weight_motion": weight_motion, "seed": autofill_seed,
            "weight_position": weight_position, "videos_per_block": n_picks,
        }
        if repeat_fallbacks is not None:
            weighting["repeat_fallbacks"] = repeat_fallbacks

        slots = []
        used_videos = set()
        if track is not None and n_picks > 0:
            # 🎵 Shape matching (the same search and weighting as Manual step-through's
            # ⚡⚡ Auto-fill ALL): every block-length window of every available scene is
            # scored against this block's music shape, then ranked by the Auto-fill weights.
            for pick in _auto_shape_picks(i, seg, n_picks, queues, sequential, excluded_here,
                                          excluded_for_block(queues, global_excluded, role_filter, ctx=pctx),
                                          role_filter, segments, confirmed_so_far, prev_block_videos,
                                          allow_same_video, weighting, track, shape_matches,
                                          repeat_fallbacks, ctx=pctx):
                slots.append({
                    "chain": [pick], "video_id": pick["video_id"], "scene_id": pick["scene_id"],
                    "tags": sorted(pick["tags"]), "motion_norm": pick["motion_norm"],
                    "thumbnail": pick["thumbnail"], "clip_duration_sec": pick["clip_duration_sec"],
                    "offset_into_scene_sec": pick["offset_into_scene_sec"],
                    "clip_start_sec": pick["clip_start_sec"],
                })
                used_videos.add(pick["video_id"])
        # Any slot shape matching couldn't fill (no window long enough, or no motion
        # curves) is filled the original way — a chain of clips by motion.
        for _ in range(n_picks - len(slots)):
            avoid = used_videos if not allow_same_video else set()
            chain, unfilled = fill_slot(target, seg_duration, queues, sequential,
                                         excluded_here, excluded_for_block(queues, global_excluded, role_filter, ctx=pctx),
                                         avoid, min_clip_len_sec, role_filter, weighting, ctx=pctx)
            if not chain:
                break
            slots.append({
                "chain": chain,
                # convenience top-level fields mirror the chain's first clip, for
                # anything that only cares about "the" clip (visualization, etc.)
                "video_id": chain[0]["video_id"],
                "scene_id": chain[0]["scene_id"],
                "tags": sorted({t for c in chain for t in c["tags"]}),
                "motion_norm": chain[0]["motion_norm"],
                "thumbnail": chain[0]["thumbnail"],
                "clip_duration_sec": round(sum(c["clip_duration_sec"] for c in chain), 3),
                "offset_into_scene_sec": chain[0]["offset_into_scene_sec"],
                "clip_start_sec": chain[0]["clip_start_sec"],
            })
            for c in chain:
                used_videos.add(c["video_id"])
            if unfilled > 0.05:
                duration_shortfall_sec += unfilled

        if len(slots) < n_picks:
            clip_count_shortfall += (n_picks - len(slots))

        confirmed_so_far[i] = [c for slot in slots for c in slot["chain"]]

        by_index[i] = {
            "segment_index": i,
            "track_time": [round(seg["start"], 2), round(seg["end"], 2)],
            "segment_energy": seg["energy"],
            "segment_intensity": target,
            "clip_count": len(slots),
            "split_screen": len(slots) > 1,
            "manually_edited": i in segment_exclusions or i in segment_clip_reduction,
            "scenes": slots,
        }

    timeline = [by_index[i] for i in sorted(by_index)]
    return timeline, clip_count_shortfall, duration_shortfall_sec
