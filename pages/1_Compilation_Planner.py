"""
Compilation Planner — interactive matching of video scenes to a music track's
energy/beat timeline, with a live visualization and JSON export for the VSE add-on.
"""

import hashlib
import json
import math
import random
import re
import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from render_preview import render_plan_dict, colab_to_local

# st.fragment (Streamlit 1.37+; was st.experimental_fragment in 1.33-1.36) lets
# part of the page rerun on its own instead of the whole script re-executing on
# every click — used below so "Use this clip" / "View fit" don't have to redo
# everything above them (sidebar, chart, track/catalogue loading) just to
# update the candidate grid. Falls back to a no-op on an older Streamlit —
# the page still works, it just won't get this speedup until you upgrade.
if hasattr(st, "fragment"):
    fragment = st.fragment
elif hasattr(st, "experimental_fragment"):
    fragment = st.experimental_fragment
else:
    fragment = lambda f: f


def rerun_full():
    """A rerun called from INSIDE a fragment normally only reruns that
    fragment, not the whole page — right for View Fit / checkboxes, which
    only affect what's shown, but wrong for Skip Ahead, which changes which
    footage is available and needs the search above the fragment to redo.
    scope="app" forces a full rerun even from inside a fragment; older
    Streamlit versions don't have that parameter, so fall back to a plain
    rerun (harmless there since such versions don't have fragments to
    escape from anyway — the fallback IS a full rerun already)."""
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()
APP_VERSION = "1.6.0"

from config import CATALOGUE_DIR, AUDIO_DIR, PLANS_DIR, PREVIEW_DIR, PROJECTS_DIR, WHITELISTED_EMAILS

# Google Drive sync — graceful fallback if credentials not present
try:
    from drive_sync import sync_pull, push_file_to_drive, push_directory_to_drive, credentials_available
    _DRIVE_SYNC_AVAILABLE = credentials_available()
except ImportError:
    _DRIVE_SYNC_AVAILABLE = False

# Google Drive OAuth — for writing plans/projects/previews to personal Drive
try:
    from drive_oauth import (
        get_auth_url, exchange_code_for_token,
        push_file_with_oauth, is_authenticated, SESSION_KEY as _OAUTH_SESSION_KEY,
    )
    _OAUTH_AVAILABLE = True
except ImportError:
    _OAUTH_AVAILABLE = False

st.set_page_config(page_title="Compilation Planner", layout="wide")

# --- Whitelist gate ---
if st.user.email not in WHITELISTED_EMAILS:
    st.title("Access Denied")
    st.error(f"**{st.user.email}** is not authorised to use this app.")
    st.caption("Contact the administrator to request access.")
    if st.button("Sign out"):
        st.logout()
    st.stop()

st.title("Compilation Planner")
st.caption(f"v{APP_VERSION}  ·  {st.user.name}")
if st.sidebar.button("Sign out", key="signout_btn"):
    st.logout()


def sanitize_filename(name: str) -> str:
    """Strip characters that aren't valid in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def _tc_to_seconds(tc: str) -> float:
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def resolve_thumbnail(stored_path: str, video_id: str) -> Path:
    filename = Path(stored_path).name
    return CATALOGUE_DIR / "thumbnails" / video_id / filename


# ---------------------------------------------------------------------------
# Data loading (cached — only recomputes when inputs actually change)
# ---------------------------------------------------------------------------

@st.cache_data
def list_tracks():
    return sorted(f.stem for f in AUDIO_DIR.glob("*.json") if f.name != "audio_index.json")


@st.cache_data
def load_track(track_id: str) -> dict:
    with open(AUDIO_DIR / f"{track_id}.json") as f:
        return json.load(f)


@st.cache_data
def load_all_catalogues() -> dict:
    """video_id -> catalogue dict, WITHOUT motion_curve. st.cache_data
    deep-copies its return value on every single call, and this function is
    called many times per script rerun — and every widget interaction on the
    page triggers a full rerun. motion_curve holds a frame-by-frame array per
    scene (added by the motion-curve backfill) and can be large across a
    whole library; keeping it out of the shared cache is what stops every
    click, anywhere on the page, from re-copying tens of MB it doesn't need.
    Advanced Shape Matching gets curve data separately, per video, from
    load_motion_curves_for_video — so only an actual search pays that cost,
    and only for the videos it's actually searching."""
    catalogues = {}
    for cat_file in CATALOGUE_DIR.glob("*.json"):
        if cat_file.name in ("training_data.jsonl", "video_index.json"):
            continue
        with open(cat_file) as f:
            cat = json.load(f)
        for scene in cat["scenes"]:
            scene.pop("motion_curve", None)
        catalogues[cat["video_id"]] = cat
    return catalogues


@st.cache_data(max_entries=20)  # bounded: a long session touching many videos won't grow this forever —
                                # oldest-used video's curves get evicted once the cap is hit
def load_motion_curves_for_video(video_id: str) -> dict:
    """scene_id -> motion_curve, for ONE video, loaded and cached separately
    from load_all_catalogues (see that function's docstring for why)."""
    cat_file = CATALOGUE_DIR / f"{video_id}.json"
    if not cat_file.exists():
        return {}
    with open(cat_file) as f:
        cat = json.load(f)
    return {s["scene_id"]: s["motion_curve"] for s in cat["scenes"] if s.get("motion_curve")}


def seconds_to_time(sec: float) -> datetime.time:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return datetime.time(hour=min(h, 23), minute=m, second=s)


def time_to_seconds(t: datetime.time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second


def format_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def find_best_motion_window(video_id: str, window_len: float) -> tuple:
    """Slide a window of window_len seconds across this video's scenes and
    return the (start, end) with the highest duration-weighted average motion
    intensity. Candidate start positions are each scene's own start time —
    a natural, efficient heuristic rather than scanning every possible offset."""
    cat = load_all_catalogues()[video_id]
    scenes = cat["scenes"]
    duration = get_video_duration(video_id)
    if window_len >= duration:
        return 0.0, duration

    candidates = sorted({0.0, max(0.0, duration - window_len)} |
                         {_tc_to_seconds(s["start_tc"]) for s in scenes})

    best_start, best_score = 0.0, -1.0
    for t0 in candidates:
        t0 = min(t0, max(0.0, duration - window_len))
        t1 = t0 + window_len
        total_overlap = weighted = 0.0
        for s in scenes:
            s_start, s_end = _tc_to_seconds(s["start_tc"]), _tc_to_seconds(s["end_tc"])
            overlap = max(0.0, min(s_end, t1) - max(s_start, t0))
            if overlap > 0:
                total_overlap += overlap
                weighted += overlap * s.get("motion_intensity", 0.0)
        score = (weighted / total_overlap) if total_overlap > 0 else 0.0
        if score > best_score:
            best_score, best_start = score, t0

    return best_start, min(best_start + window_len, duration)


def get_video_duration(video_id: str) -> float:
    cat = load_all_catalogues().get(video_id)
    if not cat or not cat["scenes"]:
        return 0.0
    return _tc_to_seconds(cat["scenes"][-1]["end_tc"])


def _overlap_with_range(scene_start: float, scene_end: float, time_range) -> tuple:
    """Intersect a scene's [start, end] with an optional (start, end) restriction.
    Returns (effective_start, effective_end) or (None, None) if no overlap."""
    if not time_range:
        return scene_start, scene_end
    r_start, r_end = time_range
    eff_start, eff_end = max(scene_start, r_start), min(scene_end, r_end)
    if eff_end <= eff_start:
        return None, None
    return eff_start, eff_end


def scene_is_usable(scene: dict, tag_filter: tuple) -> bool:
    """False if the scene should never be used by the planner — either it's
    been manually excluded (e.g. an intro/outro clip) or it doesn't match the
    current tag filter."""
    if scene.get("excluded"):
        return False
    if not tag_filter:
        return True
    wanted = set(t.lower() for t in tag_filter)
    return bool(wanted & set(t.lower() for t in scene.get("tags", [])))


# Leftover footage shorter than this after a trim is discarded rather than
# kept in the pool — not worth tracking a sliver nobody could use anyway.
MIN_LEFTOVER_SEC = 0.5


def build_footage_queues(candidate_video_ids: tuple, tag_filter: tuple, video_time_ranges: dict = None) -> dict:
    """video_id -> list of mutable 'span' dicts, one per usable scene, sorted
    by scene_id (chronological order within that video). Each span tracks how
    much of that scene's footage is still unused — segments consume from the
    front of a span, and any leftover stays available for a later segment to
    pick up. motion_norm is normalized PER VIDEO, matching the previous
    behaviour (shape/rhythm matching within that video's own scenes).

    video_time_ranges: optional {video_id: (start_sec, end_sec)} — restricts
    that video to only the footage overlapping the given range. A scene that
    straddles the boundary is trimmed to the overlapping portion, not
    dropped or included whole."""
    video_time_ranges = video_time_ranges or {}
    catalogues = load_all_catalogues()
    queues = {}

    for video_id, cat in catalogues.items():
        if candidate_video_ids and video_id not in candidate_video_ids:
            continue

        filtered = [s for s in cat["scenes"] if scene_is_usable(s, tag_filter)]
        motions = [s.get("motion_intensity", 0.0) for s in filtered]
        if not motions:
            continue
        m_min, m_max = min(motions), max(motions)
        m_span = (m_max - m_min) or 1.0
        time_range = video_time_ranges.get(video_id)

        spans = []
        for scene in sorted(filtered, key=lambda s: s["scene_id"]):
            scene_start_sec = _tc_to_seconds(scene["start_tc"])
            scene_end_sec = _tc_to_seconds(scene["end_tc"])
            eff_start, eff_end = _overlap_with_range(scene_start_sec, scene_end_sec, time_range)
            if eff_start is None:
                continue  # entirely outside the selected range
            duration = eff_end - eff_start
            if duration <= 0:
                continue
            spans.append({
                "video_id": video_id,
                "scene_id": scene["scene_id"],
                "tags": scene.get("tags", []),
                "thumbnail": str(resolve_thumbnail(scene["thumbnail_paths"][0], video_id))
                             if scene["thumbnail_paths"] else None,
                "motion_norm": round((scene.get("motion_intensity", 0.0) - m_min) / m_span, 3),
                "scene_start_sec": scene_start_sec,
                "offset_sec": eff_start - scene_start_sec,  # starts partway in if the range clipped the front
                "remaining_sec": duration,
                "intro_candidate": bool(scene.get("intro_candidate", False)),
                "outro_candidate": bool(scene.get("outro_candidate", False)),
            })
        if spans:
            queues[video_id] = spans

    return queues


def get_candidate_spans(queues: dict, sequential: bool, global_excluded: set = None,
                        min_duration: float = 0.0) -> list[dict]:
    """Currently-pickable spans. In sequential mode, only the earliest
    not-yet-exhausted, non-rejected span of each video is eligible —
    enforcing that a video's own scenes are only ever used in their original
    order. Globally-rejected spans (see global_excluded) are skipped over
    entirely here, not just filtered out afterward — a rejected clip is never
    consumed, so without this it would permanently block a sequential
    queue's front position, effectively banning every scene after it too.

    min_duration: in sequential mode, spans SHORTER than this are also
    skipped over (not just rejected ones) when looking for "the front" —
    used by Advanced Shape Matching so a too-short leftover doesn't lock a
    video out of a block entirely. This only affects which span is offered
    as a SEARCH candidate; nothing is discarded here — a video's shorter
    earlier footage is still there and still gets discarded the normal way
    (via carve_span's sequential rule) only if something later actually gets
    picked. Non-sequential mode and auto mode's fill_slot (min_duration=0.0,
    the default) are unaffected."""
    global_excluded = global_excluded or set()
    candidates = []
    for spans in queues.values():
        if sequential:
            for span in spans:
                key = (span["video_id"], span["scene_id"])
                if span["remaining_sec"] > 0.05 and span["remaining_sec"] >= min_duration and key not in global_excluded:
                    candidates.append(span)
                    break
        else:
            candidates.extend(
                s for s in spans
                if s["remaining_sec"] > 0.05 and (s["video_id"], s["scene_id"]) not in global_excluded
            )
    return candidates


def consume_span(span: dict, needed_sec: float) -> dict:
    """Trim needed_sec off the front of a span (in place) and return the
    picked sub-clip. If less than needed_sec remains, the whole remainder is
    used — the resulting clip is simply shorter than requested."""
    used_sec = min(needed_sec, span["remaining_sec"])
    pick = {
        "video_id": span["video_id"],
        "scene_id": span["scene_id"],
        "tags": span["tags"],
        "thumbnail": span["thumbnail"],
        "motion_norm": span["motion_norm"],
        "clip_start_sec": round(span["scene_start_sec"] + span["offset_sec"], 3),
        "clip_duration_sec": round(used_sec, 3),
        "offset_into_scene_sec": round(span["offset_sec"], 2),
    }
    span["offset_sec"] += used_sec
    span["remaining_sec"] -= used_sec
    if span["remaining_sec"] < MIN_LEFTOVER_SEC:
        span["remaining_sec"] = 0.0
    return pick


# ---------------------------------------------------------------------------
# Advanced Shape Matching: compare a candidate clip's frame-by-frame motion
# curve to a block's audio energy curve, point for point, rather than a
# single motion-vs-intensity average. Needs the per-scene motion_curve field
# from cell3_updated.py's compute_motion_curve (run backfill_motion_curve.py
# in Colab to add it to catalogues processed before this feature existed).
# ---------------------------------------------------------------------------

SHAPE_MATCH_POINTS = 40      # both curves are resampled to this many points before comparing
# (A per-position search-step constant used to live here — the search is now vectorized and
#  exhaustive, checking every frame position at once, so the speed/precision compromise it existed
#  for is gone; see find_best_shape_window.)


def resample_curve(values, n_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.full(n_points, float(values[0]) if len(values) else 0.0)
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_old, values)


def shape_score(curve_a, curve_b, n_points: int = SHAPE_MATCH_POINTS) -> float:
    """Pearson correlation between two curves of possibly different length /
    sample rate, after resampling both to n_points and z-scoring — so this
    measures PATTERN similarity (rises and falls in the same places),
    independent of either curve's absolute scale. Range roughly [-1, 1];
    higher is a better shape match. Returns 0.0 for a degenerate (flat)
    curve, since correlation is undefined there."""
    a = resample_curve(curve_a, n_points)
    b = resample_curve(curve_b, n_points)
    a_std, b_std = a.std(), b.std()
    if a_std < 1e-9 or b_std < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _make_interp_matrix(src_len: int, n_points: int) -> np.ndarray:
    """Precomputed linear-interpolation weights: resampling any uniformly-
    sampled src_len-point signal to n_points is a FIXED linear combination —
    it depends only on src_len/n_points, never on the values themselves. So
    this matrix, built once, lets many candidate windows get resampled at
    once via a single matrix multiply (W @ values, or windows @ W.T for many
    windows stacked as rows) instead of calling np.interp per window.
    Verified to reproduce np.interp's own output exactly."""
    if src_len < 2:
        return np.ones((n_points, max(src_len, 1)))
    x_old = np.linspace(0, 1, src_len)
    x_new = np.linspace(0, 1, n_points)
    idx = np.clip(np.searchsorted(x_old, x_new, side="right") - 1, 0, src_len - 2)
    x0, x1 = x_old[idx], x_old[idx + 1]
    frac = np.where(x1 > x0, (x_new - x0) / (x1 - x0), 0.0)
    W = np.zeros((n_points, src_len))
    rows = np.arange(n_points)
    W[rows, idx] += 1 - frac
    W[rows, idx + 1] += frac
    return W


def find_best_shape_window(values: np.ndarray, window_frames: int, audio_curve: np.ndarray,
                           n_points: int = SHAPE_MATCH_POINTS) -> tuple:
    """Vectorized replacement for sliding shape_score() across every
    candidate start position one at a time: resamples EVERY window in a
    single matrix multiply and scores them all in one batched correlation,
    instead of a Python loop calling np.interp + np.corrcoef per position.
    Mathematically identical to shape_score() at each position (verified
    directly against it) — the only behavioural change is that it's cheap
    enough to check every single frame position, not just every
    SHAPE_SEARCH_STEP_SEC, so it can also find a better-aligned match than
    the old stepped search could. Returns (best_score, best_start_frame)."""
    n_positions = len(values) - window_frames + 1
    if n_positions <= 0:
        return -2.0, 0

    windows = np.lib.stride_tricks.sliding_window_view(values, window_frames)  # (n_positions, window_frames), no copy
    W = _make_interp_matrix(window_frames, n_points)
    resampled = windows @ W.T  # (n_positions, n_points)

    audio_r = resample_curve(audio_curve, n_points)
    audio_c = audio_r - audio_r.mean()
    audio_std = audio_r.std()

    row_mean = resampled.mean(axis=1, keepdims=True)
    row_std = resampled.std(axis=1)
    numerator = (resampled - row_mean) @ audio_c
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where((row_std > 1e-9) & (audio_std > 1e-9),
                          numerator / (n_points * row_std * audio_std), 0.0)

    best_idx = int(np.argmax(scores))
    return float(scores[best_idx]), best_idx


def get_block_audio_curve(track: dict, start_sec: float, end_sec: float) -> np.ndarray:
    """The audio energy curve for one block's time range, at whatever
    resolution the track has (30 Hz hires if available, else the coarse
    envelope)."""
    hires = track.get("hires")
    if hires:
        rate = float(hires["rate"])
        rms = np.asarray(hires["rms"], dtype=float)
        lo, hi = int(start_sec * rate), max(int(start_sec * rate) + 1, int(end_sec * rate))
        return rms[lo:hi]
    env = track["energy_envelope"]
    slice_ = np.array([e["energy"] for e in env if start_sec <= e["time"] < end_sec])
    return slice_ if slice_.size > 0 else np.array([e["energy"] for e in env])


def find_shape_candidates(
    segment: dict, queues: dict, sequential: bool, excluded_here: set, global_excluded: set,
    audio_curve: np.ndarray, max_matches: int, restrict_to_front: bool = False,
    remaining_blocks: int = 1, weight_spread: float = 0.0,
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

    remaining_blocks / weight_spread: feed compute_skip_budget (see there) to
    size that range — how many more block-length windows past the front this
    span's own surplus footage (relative to blocks still left overall) can
    safely spare, scaled by weight D. 0 (the defaults) collapses to exactly
    the front window, same as a hard restriction. The live candidate grid
    (manual review) always calls this with restrict_to_front's default
    False — skipping ahead there is a deliberate, informed choice, not
    something this budget needs to manage."""
    block_duration = segment["end"] - segment["start"]
    results = []
    missing_curve_videos = set()
    video_curves_cache = {}  # video_id -> {scene_id: curve}, fetched at most once per call, not per span

    for span in get_candidate_spans(queues, sequential, global_excluded, min_duration=block_duration):
        key = (span["video_id"], span["scene_id"])
        if key in excluded_here or span["remaining_sec"] < block_duration - 1e-6:
            continue

        if span["video_id"] not in video_curves_cache:
            video_curves_cache[span["video_id"]] = load_motion_curves_for_video(span["video_id"])
        curve = video_curves_cache[span["video_id"]].get(span["scene_id"])
        if not curve or not curve.get("values"):
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
            available_windows = max(1, int(span["remaining_sec"] // block_duration))
            skip_budget = compute_skip_budget(available_windows, remaining_blocks, weight_spread)
            search_end_frame = min(span_end_frame, span_start_frame + (skip_budget + 1) * window_frames)
        else:
            search_end_frame = span_end_frame

        # Vectorized: scores every frame position in the search range at once (see
        # find_best_shape_window), not just every SHAPE_SEARCH_STEP_SEC —
        # faster AND exhaustive rather than a speed/precision compromise.
        best_score, best_offset = find_best_shape_window(
            values[span_start_frame:search_end_frame], window_frames, audio_curve
        )
        best_start_frame = span_start_frame + best_offset

        results.append({
            "video_id": span["video_id"],
            "scene_id": span["scene_id"],
            "scene_start_sec": span["scene_start_sec"],  # + window_offset_sec = absolute position in the video
            "window_offset_sec": round(best_start_frame / fps, 3),
            "window_duration_sec": round(block_duration, 3),
            "score": round(best_score, 3),
            "motion_norm": span["motion_norm"],
            "curve_slice": values[best_start_frame:best_start_frame + window_frames],
            "fps": fps,
            "tags": span["tags"],
            "thumbnail": span["thumbnail"],
            "intro_candidate": span.get("intro_candidate", False),
            "outro_candidate": span.get("outro_candidate", False),
        })

    results.sort(key=lambda r: -r["score"])
    return results[:max_matches], missing_curve_videos


def compute_skip_budget(available_windows: int, remaining_blocks: int, weight_spread: float) -> int:
    """How many windows past the frontmost an automatic sequential-mode pick
    is allowed to consider, instead of always being pinned to the very front.

    surplus = how many MORE windows this video has available right now than
    one-per-remaining-block — a conservative estimate, since not every
    remaining block will necessarily draw from this particular video, so it
    never overcommits footage that might genuinely be needed later. Never
    negative: a video with barely enough (or less than enough) footage for
    what's left gets a surplus of 0, so it's pinned to the front exactly like
    before this existed.

    weight D (spread) controls how much of that surplus is actually used —
    0 means never skip ahead (the original, fully conservative behaviour);
    the slider's own max means use the entire safety margin. This reuses D
    rather than introducing a new weight because "how freely to roam across
    a video's own footage" is the same underlying idea as "spread usage
    across videos" — both are about not over-committing to the first option
    directly in front of you.

    Returns the number of EXTRA windows beyond the frontmost that may be
    considered — 0 means restricted to just the front window, same as
    before; slice candidates with [: this + 1]."""
    surplus = max(0, available_windows - max(1, remaining_blocks))
    power = max(0.0, min(1.0, weight_spread / 5.0))  # D's slider range is 0..5
    return int(surplus * power)


def _stable_random_unit(seed: int, *parts) -> float:
    """Deterministic pseudo-random value in [0,1), derived from seed + parts.
    Stable across reruns for the SAME inputs (so an unrelated widget click
    never reshuffles already-made auto-fill picks) — changes only if the
    seed itself changes. Not Python's hash(), which is randomized per
    process for strings."""
    h = hashlib.sha256(f"{seed}:{':'.join(str(p) for p in parts)}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def rank_candidates_weighted(
    candidates: list, current_block: int, segments: list, confirmed: dict, queues: dict,
    prev_block_videos: set, weight_shape: float, weight_random: float,
    weight_repeat_penalty: float, weight_spread: float, weight_motion: float, random_seed: int,
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
    C: a penalty if this candidate's video was used in the immediately
       preceding block, scaled by how long the adjacent blocks are relative
       to the plan's average segment length — repeating right after a long,
       lingering shot is penalized hard; repeating during a run of short,
       rapid cuts barely matters. This weight is also the control for "too
       many rapid changes": raise it to settle switching down.
    D: blends "hasn't been used much yet" with "still has a lot of footage
       remaining" — pulls long, under-used videos back into rotation across
       many block decisions, rather than a long source being used once early
       and then left untouched.
    E: motion intensity match — how closely the clip's normalised motion level
       matches this block's intensity. 1.0 = perfect match, 0.0 = opposite
       end of the scale. Complementary to A (shape): A rewards the same
       PATTERN of rises/falls; E rewards the right overall LEVEL of activity.

    Returns candidates sorted by combined score, descending. Does not
    mutate or reorder the input list itself."""
    if not candidates:
        return []

    seg_durations = [sg["end"] - sg["start"] for sg in segments]
    avg_duration = max((sum(seg_durations) / len(seg_durations)) if seg_durations else 1.0, 1e-6)
    curr_duration = segments[current_block]["end"] - segments[current_block]["start"]
    prev_duration = (segments[current_block - 1]["end"] - segments[current_block - 1]["start"]
                     if current_block > 0 else 0.0)
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

    scored = []
    for c in candidates:
        A = max(0.0, c.get("score", 0.0))  # Auto mode's spans carry no shape score — A is simply 0 there
        B = _stable_random_unit(random_seed, current_block, c["video_id"], c["scene_id"])

        if current_block > 0 and c["video_id"] in prev_block_videos:
            length_factor = min(3.0, max(0.2, (prev_duration + curr_duration) / (2 * avg_duration)))
            C = -length_factor / 3.0
        else:
            C = 0.0

        norm_used = used_by_video[c["video_id"]] / max_used
        norm_remaining = remaining_by_video[c["video_id"]] / max_remaining
        D = (1 - norm_used) * 0.5 + norm_remaining * 0.5

        # E: 1 - |clip motion_norm - block intensity|, clamped [0, 1]
        E = max(0.0, 1.0 - abs(c.get("motion_norm", 0.5) - block_intensity))

        combined = (weight_shape * A + weight_random * B + weight_repeat_penalty * C
                    + weight_spread * D + weight_motion * E)
        scored.append((combined, c))

    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored]


CHOR_MAX_OPTIONS_PER_VIDEO = 300  # safety cap on tiled dropdown entries — see docstring below


def get_available_clips_for_video(queues: dict, video_id: str, block_duration: float,
                                   sequential: bool, global_excluded: set,
                                   max_options: int = CHOR_MAX_OPTIONS_PER_VIDEO,
                                   role_filter: str = None) -> list[dict]:
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

    role_filter: "intro_candidate" or "outro_candidate" — if given, the result is
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
            role_tag = ""
            if span.get("intro_candidate"):
                role_tag += "  🎬intro"
            if span.get("outro_candidate"):
                role_tag += "  🎬outro"
            label = (f"Scene {span['scene_id']}  {format_mmss(abs_start)} – {format_mmss(abs_end)}"
                     f"  motion {span['motion_norm']:.2f}{role_tag}")
            result.append({**span, "offset_sec": win_offset, "abs_start": abs_start, "label": label})
        if len(result) >= max_options:
            break
    result = sorted(result, key=lambda s: s["abs_start"])

    if role_filter:
        role_matches = [c for c in result if c.get(role_filter)]
        if role_matches:
            return role_matches
    return result


@fragment
def render_choreography_block(
    seg_idx: int, seg: dict, selected_videos: list, max_chor_videos: int,
    rec_count: int, queues: dict, sequential_mode: bool, global_excluded: set,
    catalogues: dict, prev_block_videos: set = None, video_stats: dict = None,
    role_filter: str = None, weighting: dict = None,
) -> None:
    """One block's choreography grid: one column per source video (up to
    max_chor_videos).  Each column has:
      • a thumbnail of the selected clip
      • a dropdown of every available block_duration-sized window across
        that video's whole remaining footage (past clips already removed
        by the queue, so the list is always forward-only)
      • a "Use this clip" checkbox — only ticked columns get confirmed

    rec_count is the recommended simultaneous-clip count from the split-
    screen intensity settings — shown as a guide only, same as Manual
    step-through: you can tick any number of columns, more or fewer.

    prev_block_videos: video_ids used anywhere in the immediately preceding
    CONFIRMED block — these columns get a blue box, the same visual nudge
    Manual step-through's candidate grid gives (purely informational,
    doesn't affect what can be picked).

    video_stats: video_id -> {"used": seconds confirmed so far, "remaining":
    seconds still available in queues} — shown on each card, same as Manual
    step-through's candidate grid.

    role_filter: "intro_candidate" or "outro_candidate" when this is the
    first/last block and the Prefer Intro/Outro sidebar toggle is on —
    narrows each column's dropdown to that video's tagged clips, when it
    has any (see get_available_clips_for_video for the opportunistic
    fallback). None for every other block.

    weighting: {"segments", "confirmed", "weight_shape", "weight_random",
    "weight_repeat_penalty", "weight_spread", "weight_motion", "seed"} — when
    given, the "⭐ Auto pick" preview shown for a column left on Auto uses
    rank_candidates_weighted instead of plain closest-motion-match, matching
    exactly what Confirm & Next will actually do. None falls back to plain
    closest-motion-match.

    Overrides are stored in session_state["chor_overrides"][seg_idx][video_id]
    as (scene_id, offset_sec).  Pick state lives in
    session_state["chor_pick_{seg_idx}_{video_id}"] as a bool, same pattern
    as adv_pick_ in Manual step-through."""

    prev_block_videos = prev_block_videos or set()
    video_stats = video_stats or {}
    block_duration = seg["end"] - seg["start"]
    seg_target = seg.get("intensity", seg["energy"])
    eligible = [vid for vid in selected_videos
                if get_available_clips_for_video(queues, vid, block_duration, sequential_mode,
                                                 global_excluded, role_filter=role_filter)]
    shown = eligible[:max_chor_videos]
    hidden = len(eligible) - len(shown)

    if not shown:
        st.info("No source video has a clip long enough for this block's duration. "
                "Try adjusting segmentation or video selection.")
        return
    if hidden:
        st.caption(f"{hidden} more eligible video(s) not shown — increase 'Max videos per block' in the sidebar.")
    if role_filter:
        role_label = "intro" if role_filter == "intro_candidate" else "outro"
        st.caption(f"🎬 Dropdowns below are narrowed to each video's tagged **{role_label}** clips, "
                   f"where it has any — videos with none show their normal full availability.")

    ticked_count = sum(1 for vid in shown if st.session_state.get(f"chor_pick_{seg_idx}_{vid}", False))
    st.caption(
        f"Recommended simultaneous clips for this block: **{rec_count}** (from your split-screen settings) "
        f"— this is a guide only; tick any number of columns below.  Currently ticked: **{ticked_count}**."
    )

    overrides = st.session_state.setdefault("chor_overrides", {}).setdefault(seg_idx, {})
    chor_cols = st.columns(len(shown))
    highlight_keys = []

    for col, video_id in zip(chor_cols, shown):
        clips = get_available_clips_for_video(queues, video_id, block_duration, sequential_mode,
                                               global_excluded, role_filter=role_filter)
        pick_key = f"chor_pick_{seg_idx}_{video_id}"
        is_repeat = video_id in prev_block_videos

        with col:
            box_key = f"chor_box_{seg_idx}_{video_id}"
            box = st.container(border=True, key=box_key) if _SUPPORTS_KEYED_CONTAINER else st.container(border=True)
            if is_repeat and _SUPPORTS_KEYED_CONTAINER:
                highlight_keys.append(box_key)
            with box:
                if is_repeat:
                    st.caption("🔵 same video as previous block")
                st.caption(f"**{video_id}**")
                stats = video_stats.get(video_id, {})
                st.caption(f"used so far: {stats.get('used', 0.0):.0f}s  ·  "
                          f"remaining: {stats.get('remaining', 0.0):.0f}s")
                if role_filter and clips and not any(c.get(role_filter) for c in clips):
                    role_label = "intro" if role_filter == "intro_candidate" else "outro"
                    st.caption(f"⚠️ no {role_label}-tagged clips for this video — showing full availability")

                # Thumbnail for the currently selected clip
                current_key = overrides.get(video_id)
                display_clip = None
                if current_key is not None:
                    display_clip = next((c for c in clips
                                        if c["scene_id"] == current_key[0]
                                        and abs(c["offset_sec"] - current_key[1]) < 0.1), None)
                if display_clip is None and clips:
                    # "Auto" is selected: show what it would actually pick on Confirm,
                    # not just the chronologically-first clip (which could be a
                    # completely different one and was previously shown here
                    # misleadingly). Mirrors Confirm & Next's own Auto logic exactly,
                    # INCLUDING the sequential-mode skip budget below.
                    #
                    # In sequential mode, clips (plural) can include windows well past
                    # the front of this video's remaining footage — fine for a deliberate
                    # manual pick via the dropdown, but automatic selection shouldn't
                    # freely roam the whole thing: carving a non-front window discards
                    # everything before it. compute_skip_budget (weight D-controlled)
                    # allows a FEW windows ahead when this video has spare footage
                    # relative to how many blocks are left, 0 extra when it doesn't.
                    if sequential_mode and weighting:
                        _remaining_blocks = len(weighting["segments"]) - seg_idx
                        _skip = compute_skip_budget(len(clips), _remaining_blocks, weighting["weight_spread"])
                        auto_candidates = clips[:_skip + 1]
                    elif sequential_mode:
                        auto_candidates = clips[:1]
                    else:
                        auto_candidates = clips
                    if weighting:
                        ranked = rank_candidates_weighted(
                            auto_candidates, seg_idx, weighting["segments"], weighting["confirmed"], queues,
                            prev_block_videos, weighting["weight_shape"], weighting["weight_random"],
                            weighting["weight_repeat_penalty"], weighting["weight_spread"],
                            weighting["weight_motion"], weighting["seed"],
                        )
                        display_clip = ranked[0]
                    else:
                        display_clip = min(auto_candidates, key=lambda c: abs(c["motion_norm"] - seg_target))

                if display_clip is not None:
                    clip_abs_start = display_clip["scene_start_sec"] + display_clip["offset_sec"]
                    nearest_thumb = get_nearest_thumbnail(catalogues, video_id, clip_abs_start)
                    if nearest_thumb is not None:
                        st.image(nearest_thumb, width=180)
                    elif display_clip.get("thumbnail") and Path(display_clip["thumbnail"]).exists():
                        st.image(display_clip["thumbnail"], width=180)

                # Dropdown: which block_duration-sized window to use from this video
                options = ["Auto (algorithm picks)"] + [c["label"] for c in clips]
                if current_key is None:
                    sel_idx = 0
                else:
                    matched = next((i + 1 for i, c in enumerate(clips)
                                    if c["scene_id"] == current_key[0]
                                    and abs(c["offset_sec"] - current_key[1]) < 0.1), 0)
                    sel_idx = matched

                chosen = st.selectbox(
                    "Clip", options, index=sel_idx,
                    key=f"chor_sel_{seg_idx}_{video_id}",
                    label_visibility="collapsed",
                )
                if chosen == "Auto (algorithm picks)":
                    overrides.pop(video_id, None)
                else:
                    chosen_span = clips[options.index(chosen) - 1]
                    overrides[video_id] = (chosen_span["scene_id"], chosen_span["offset_sec"])

                if display_clip is not None:
                    auto_tag = "⭐ Auto pick — " if current_key is None else ""
                    role_bits = []
                    if display_clip.get("intro_candidate"):
                        role_bits.append("🎬intro")
                    if display_clip.get("outro_candidate"):
                        role_bits.append("🎬outro")
                    role_suffix = f"  {' '.join(role_bits)}" if role_bits else ""
                    st.caption(
                        f"{auto_tag}Scene {display_clip['scene_id']}  "
                        f"{format_mmss(display_clip['scene_start_sec'] + display_clip['offset_sec'])}  "
                        f"motion {display_clip['motion_norm']:.2f}{role_suffix}"
                    )

                st.checkbox("Use this clip", key=pick_key)

    if highlight_keys:
        css = "\n".join(f'div[class*="st-key-{k}"] {{ border: 3px solid #1c6fea !important; }}' for k in highlight_keys)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# Streamlit has no simple "coloured border" parameter, so getting an actual
# blue box (not just a text label) relies on a version-dependent technique:
# st.container(key=...) generates a CSS class named "st-key-<key>" (added
# around Streamlit 1.32-1.34, the same era as the fragment feature used
# elsewhere in this file), which a small injected <style> block can then
# target. Detected once; if unsupported, candidates still get a plain
# bordered container plus a text badge, so the INFORMATION survives even
# without the colour on an older Streamlit.
def _container_supports_key():
    import inspect
    try:
        return "key" in inspect.signature(st.container).parameters
    except (TypeError, ValueError):
        return False


_SUPPORTS_KEYED_CONTAINER = _container_supports_key()


@fragment
def render_candidate_grid(candidates: list, current_block: int, already_keys: set, seg: dict,
                          block_audio_curve, prev_block_videos: set = None,
                          video_stats: dict = None, recommended_count: int = 0,
                          autofill_weights: dict = None, catalogues: dict = None) -> None:
    """The candidate thumbnails, 'Use this clip' checkboxes, 'View fit'
    buttons, and the overlay chart — as a fragment, so ticking a box or
    switching which fit you're viewing only reruns THIS, not the whole page
    (sidebar, chart, track/catalogue loading, etc. all stay put).

    Each candidate's thumbnail is the single frame closest to where that
    specific trimmed clip actually STARTS, cropped from its video's timeline
    sprite (see backfill_timeline_thumbnails.py) — never a frame from before
    the trim point, so what's shown is representative of what the clip opens
    on, not a static frame from the scene in general or a moment that gets
    trimmed away. Falls back to the scene's static thumbnail for any video
    that hasn't been backfilled yet.

    prev_block_videos: video_ids used anywhere in the immediately PREVIOUS
    block (any of its simultaneous slots) — candidates from one of these
    videos get a blue box, as a visual nudge while deciding this block's
    picks. Purely informational; doesn't affect what can be selected.

    video_stats: video_id -> {"used": seconds confirmed so far, "remaining":
    seconds still available in queues}, shown on each candidate so you can
    see at a glance how much of that source is already spoken for.

    recommended_count: the Auto-fill button ticks this many candidates,
    chosen by rank_candidates_weighted (see autofill_weights) rather than
    simply the top-scoring ones — un-ticking everything else. A starting
    point you can still freely adjust with the checkboxes, View fit, and
    Skip ahead below, exactly as before.

    autofill_weights: {"segments", "confirmed", "queues", "weight_shape",
    "weight_random", "weight_repeat_penalty", "weight_spread", "weight_motion", "seed"} —
    everything rank_candidates_weighted needs. The candidate GRID below
    always stays sorted by raw shape score regardless; only the auto-fill
    button's own picks use the weighted ranking."""
    prev_block_videos = prev_block_videos or set()
    video_stats = video_stats or {}
    autofill_weights = autofill_weights or {}

    if candidates and recommended_count > 0:
        if st.button(f"⚡ Auto-fill top {min(recommended_count, len(candidates))}",
                     key=f"adv_autofill_{current_block}",
                     help="Picks the best candidates by the Auto-fill weighting above (shape match, "
                          "motion intensity, randomness, avoiding repeats, spreading usage) — not just "
                          "the highest raw score. Unticks anything else. A starting point; adjust freely afterward."):
            ranked = rank_candidates_weighted(
                candidates, current_block, autofill_weights["segments"], autofill_weights["confirmed"],
                autofill_weights["queues"], prev_block_videos, autofill_weights["weight_shape"],
                autofill_weights["weight_random"], autofill_weights["weight_repeat_penalty"],
                autofill_weights["weight_spread"], autofill_weights["weight_motion"],
                autofill_weights["seed"],
            )
            chosen_keys = {(c["video_id"], c["scene_id"]) for c in ranked[:recommended_count]}
            for c in candidates:
                st.session_state[f"adv_pick_{current_block}_{c['video_id']}_{c['scene_id']}"] = \
                    (c["video_id"], c["scene_id"]) in chosen_keys
            st.rerun()

    # Sort candidates alphabetically by video_id then scene_id for a stable
    # grid layout — cards don't jump around when skipping ahead changes what's
    # available. Shape score order is preserved separately (for View fit default
    # and auto-fill), and shown on each card so ranking is still visible.
    display_candidates = sorted(candidates, key=lambda c: (c["video_id"], c["scene_id"]))

    highlight_keys = []
    cand_cols = st.columns(min(3, max(1, len(display_candidates))))
    for i, c in enumerate(display_candidates):
        key_id = (c["video_id"], c["scene_id"])
        is_repeat = c["video_id"] in prev_block_videos
        stats = video_stats.get(c["video_id"], {})
        with cand_cols[i % len(cand_cols)]:
            box_key = f"cand_box_{current_block}_{c['video_id']}_{c['scene_id']}"
            box = st.container(border=True, key=box_key) if _SUPPORTS_KEYED_CONTAINER else st.container(border=True)
            if is_repeat and _SUPPORTS_KEYED_CONTAINER:
                highlight_keys.append(box_key)
            with box:
                if is_repeat:
                    st.caption("🔵 same video as previous block")
                if c.get("intro_candidate"):
                    st.caption("🎬 intro candidate")
                if c.get("outro_candidate"):
                    st.caption("🎬 outro candidate")
                clip_start = c.get("scene_start_sec", 0.0) + c["window_offset_sec"]
                nearest_thumb = get_nearest_thumbnail(catalogues or {}, c["video_id"], clip_start) if catalogues else None
                if nearest_thumb is not None:
                    st.image(nearest_thumb, width=200, caption="at the clip's start")
                elif c.get("thumbnail") and Path(c["thumbnail"]).exists():
                    st.image(c["thumbnail"], width=200, caption="whole scene (run the timeline-thumbnail "
                             "backfill for a preview of just this clip)")
                _score_rank = next((r + 1 for r, sc in enumerate(candidates) if sc["video_id"] == c["video_id"] and sc["scene_id"] == c["scene_id"]), "?")
                st.caption(f"`{c['video_id']}` #{c['scene_id']}  ·  rank #{_score_rank}  ·  shape {c['score']:.2f}  ·  motion {c['motion_norm']:.2f}")
                st.caption(f"trimmed to {c['window_offset_sec']:.1f}s–"
                          f"{c['window_offset_sec'] + c['window_duration_sec']:.1f}s in the scene")
                st.caption(f"used so far: {stats.get('used', 0.0):.0f}s  ·  remaining: {stats.get('remaining', 0.0):.0f}s")
                st.checkbox("Use this clip", key=f"adv_pick_{current_block}_{c['video_id']}_{c['scene_id']}",
                           value=key_id in already_keys)
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if st.button("🔍 View fit", key=f"adv_view_{current_block}_{c['video_id']}_{c['scene_id']}"):
                        st.session_state["adv_viewing"] = i
                        st.rerun()
                with btn_cols[1]:
                    # Skip back: remove the most recent forward-skip for this video/scene
                    # from the current block's skip list. Because queues are rebuilt from
                    # scratch on every rerun, simply removing the skip record is enough to
                    # make that footage reappear — no separate "restore" step needed.
                    # After a skip ahead the scene_id changes (new footage appears), so check
                    # by video_id only — the skip was against this video, not this specific scene.
                    _live_skips = st.session_state.get("adv_skips", {}).get(current_block, [])
                    skip_back_possible = any(sk["video_id"] == c["video_id"] for sk in _live_skips)
                    if st.button("⏮️ Skip back", key=f"adv_skipback_{current_block}_{c['video_id']}_{c['scene_id']}",
                               disabled=not skip_back_possible,
                               help="Undo the last Skip Ahead for this video, restoring its earlier footage "
                                    "as a candidate again."):
                        _skips_dict = st.session_state.setdefault("adv_skips", {})
                        _block_skips = _skips_dict.get(current_block, [])
                        # Remove the LAST skip for this video (any scene_id)
                        for idx in range(len(_block_skips) - 1, -1, -1):
                            if _block_skips[idx]["video_id"] == c["video_id"]:
                                _block_skips.pop(idx)
                                break
                        _skips_dict[current_block] = _block_skips
                        st.session_state["adv_skips"] = _skips_dict
                        st.session_state["adv_viewing"] = None
                        rerun_full()
                with btn_cols[2]:
                    if st.button("⏭️ Skip ahead", key=f"adv_skip_{current_block}_{c['video_id']}_{c['scene_id']}",
                               help="Give up this video's current footage without using it, so a later, "
                                    "possibly better-matching part of the same video becomes reachable."):
                        st.session_state["adv_skips"].setdefault(current_block, []).append(
                            {"video_id": c["video_id"], "scene_id": c["scene_id"]}
                        )
                        st.session_state["adv_viewing"] = None
                        rerun_full()

    if highlight_keys:
        css = "\n".join(f'div[class*="st-key-{k}"] {{ border: 3px solid #1c6fea !important; }}' for k in highlight_keys)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Default to showing the top candidate's fit immediately — no click needed
    # for the common case of just checking the best match. Explicitly clicking
    # a different candidate still switches the view as before.
    viewing = st.session_state.get("adv_viewing")
    if viewing is None and candidates:
        viewing = 0
    if viewing is not None and viewing < len(candidates):
        vc = candidates[viewing]
        vfig = go.Figure()
        audio_n = resample_curve(block_audio_curve, SHAPE_MATCH_POINTS)
        clip_n = resample_curve(vc["curve_slice"], SHAPE_MATCH_POINTS)
        a_lo, a_hi = audio_n.min(), audio_n.max()
        c_lo, c_hi = clip_n.min(), clip_n.max()
        audio_norm = (audio_n - a_lo) / (a_hi - a_lo) if a_hi > a_lo else audio_n * 0 + 0.5
        clip_norm = (clip_n - c_lo) / (c_hi - c_lo) if c_hi > c_lo else clip_n * 0 + 0.5
        x_axis = np.linspace(seg["start"], seg["end"], SHAPE_MATCH_POINTS)
        vfig.add_trace(go.Scatter(x=x_axis, y=audio_norm, mode="lines", name="Block audio curve",
                                  line=dict(color="rgba(100,150,255,0.8)", width=2)))
        vfig.add_trace(go.Scatter(x=x_axis, y=clip_norm, mode="lines",
                                  name=f"{vc['video_id']} #{vc['scene_id']} motion curve",
                                  line=dict(color="rgba(230,120,20,0.9)", width=2)))
        vfig.update_layout(height=250, margin=dict(t=30, b=20), title=f"Shape score: {vc['score']:.2f}",
                           legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(vfig, use_container_width=True)


def skip_span(queues: dict, video_id: str, scene_id: int, sequential: bool = False) -> None:
    """Manual escape hatch for sequential mode: discard a video's current
    front span ENTIRELY without using any of it, so the search can reach
    whatever comes after it in that video. Unlike carve_span, this produces
    no pick — the footage is simply given up. Raises ValueError if that
    scene's span no longer exists (already used or already skipped).

    sequential=True also discards every OTHER remaining span in this video
    with a smaller scene_id, not just the exact one skipped. Without this,
    skipping straight to a much later scene_id (jumping over several scenes
    the search never individually touched) left those in-between scenes
    sitting untouched in the queue — free to resurface as a candidate on a
    LATER block, stepping backward past something already used. See
    carve_span's matching fix for the same invariant when a scene is used
    rather than skipped."""
    spans = queues.get(video_id, [])
    if not any(s["scene_id"] == scene_id for s in spans):
        raise ValueError(f"No available span for {video_id} scene {scene_id} to skip — it may already be gone.")
    if sequential:
        spans[:] = [s for s in spans if s["scene_id"] > scene_id]
    else:
        spans[:] = [s for s in spans if s["scene_id"] != scene_id]


def carve_span(queues: dict, video_id: str, scene_id: int, window_offset_sec: float,
               window_duration_sec: float, sequential: bool = False) -> dict:
    """Generalization of consume_span for Advanced Shape Matching: consumes
    an ARBITRARY window from within a scene's still-available region, not
    just the front. consume_span always eats from span["offset_sec"] forward,
    which is right for auto mode's chain-filling — but a shape-matched window
    can be the best fit ANYWHERE in a scene, including its middle, which can
    leave usable footage on BOTH sides. This finds the span containing the
    window and replaces it in the queue with the remainder(s), returning a
    pick dict in the exact same shape consume_span does, so everything
    downstream (render, export, Blender import) is unaffected by which
    function produced a pick.

    sequential=True (mirrors "keep sequential order"): the LEADING remainder
    — footage before the carved window, WITHIN THIS SAME SCENE — is discarded,
    not kept for later. E.g. using 6.7s-9.7s of a scene means nothing before
    6.7s in that scene is offered to any later block for this video; only
    what comes after 9.7s can still be picked. ANY OTHER remaining scene in
    this video with a smaller scene_id is discarded too — using scene 36
    must also give up scenes 6 through 35, even ones the search never
    individually touched, or a later block could still find one of them and
    step backward past something already used. Non-sequential mode keeps
    everything else, as before.

    Raises ValueError if no current span for this scene covers the requested
    window — this would mean the window was already (partly) consumed by an
    earlier confirmed block since the candidate was generated, which the
    caller should treat as "re-search this block", not silently paper over."""
    spans = queues.get(video_id, [])
    win_start = window_offset_sec
    win_end = window_offset_sec + window_duration_sec

    for i, span in enumerate(spans):
        if span["scene_id"] != scene_id:
            continue
        span_start = span["offset_sec"]
        span_end = span["offset_sec"] + span["remaining_sec"]
        if span_start - 1e-6 <= win_start and win_end <= span_end + 1e-6:
            pick = {
                "video_id": span["video_id"],
                "scene_id": span["scene_id"],
                "tags": span["tags"],
                "thumbnail": span["thumbnail"],
                "motion_norm": span["motion_norm"],
                "clip_start_sec": round(span["scene_start_sec"] + win_start, 3),
                "clip_duration_sec": round(window_duration_sec, 3),
                "offset_into_scene_sec": round(win_start, 2),
            }

            replacements = []
            leading = win_start - span_start
            if leading >= MIN_LEFTOVER_SEC and not sequential:
                replacements.append({**span, "offset_sec": span_start, "remaining_sec": leading})
            trailing = span_end - win_end
            if trailing >= MIN_LEFTOVER_SEC:
                replacements.append({**span, "offset_sec": win_end, "remaining_sec": trailing})

            spans[i:i + 1] = replacements
            if sequential:
                spans[:] = [s for s in spans if s["scene_id"] >= scene_id]
            return pick

    raise ValueError(
        f"No available span for {video_id} scene {scene_id} covers "
        f"[{win_start:.2f}, {win_end:.2f}] — it may have already been used by an earlier confirmed block."
    )


@st.cache_data
def load_all_audio_tracks() -> dict:
    """track_id -> full track dict, loaded once and cached (mirrors load_all_catalogues)."""
    tracks = {}
    for f in AUDIO_DIR.glob("*.json"):
        if f.name == "audio_index.json":
            continue
        with open(f) as fh:
            t = json.load(fh)
        tracks[t.get("track_id", f.stem)] = t
    return tracks


def compute_track_features(track: dict) -> dict:
    """A track's ENERGY SHAPE, not its level — level is destroyed by the
    per-track 0-1 normalisation in audio_pipeline.py (every track's mean ends
    up in a similar band regardless of how it actually sounds), but variance
    and hit density survive that normalisation because they describe the
    pattern, not the absolute scale."""
    hires = track.get("hires")
    if hires:
        rms = np.asarray(hires["rms"], dtype=float)
        onset = np.asarray(hires.get("onset", []), dtype=float)
    else:
        rms = np.asarray([e["energy"] for e in track["energy_envelope"]], dtype=float)
        onset = np.array([])
    duration_min = max(track["duration_sec"] / 60.0, 0.1)
    hit_rate = float((onset >= 0.8).sum()) / duration_min if len(onset) else 0.0
    return {"variance": float(rms.std()), "hit_rate": hit_rate, "level": float(rms.mean())}


def compute_video_features(video_id: str, tag_filter: tuple, video_time_ranges: dict = None) -> dict:
    """A video's motion SHAPE — variance across scenes and how often it has a
    standout burst (motion_intensity_max notably above motion_intensity) —
    the video-side counterpart to compute_track_features. Unlike audio,
    video motion is never per-video normalised at storage time, so its level
    stays genuinely comparable across videos — level is kept as a fallback
    when duration/scene count is too small to trust variance/hit_rate."""
    video_time_ranges = video_time_ranges or {}
    cat = load_all_catalogues().get(video_id)
    if not cat:
        return {"variance": 0.0, "hit_rate": 0.0, "level": 0.0}
    filtered = [s for s in cat["scenes"] if scene_is_usable(s, tag_filter)]
    time_range = video_time_ranges.get(video_id)
    if time_range:
        filtered = [
            s for s in filtered
            if _overlap_with_range(_tc_to_seconds(s["start_tc"]), _tc_to_seconds(s["end_tc"]), time_range)[0] is not None
        ]
    if not filtered:
        return {"variance": 0.0, "hit_rate": 0.0, "level": 0.0}
    levels = np.array([s.get("motion_intensity", 0.0) for s in filtered])
    peaks = np.array([s.get("motion_intensity_max", 0.0) for s in filtered])
    duration_min = sum(_tc_to_seconds(s["end_tc"]) - _tc_to_seconds(s["start_tc"]) for s in filtered) / 60.0
    bursts = float((peaks > levels * 1.5).sum()) / max(duration_min, 0.1)
    return {"variance": float(levels.std()), "hit_rate": bursts, "level": float(levels.mean())}


def _population_z(values: dict) -> dict:
    """z-score a {key: value} dict against its own population (mean/std across
    all its values) — 0.0 for every entry if there's no spread to measure."""
    arr = np.array(list(values.values()), dtype=float)
    mean, std = arr.mean(), arr.std()
    if std < 1e-9:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / std for k, v in values.items()}


def compute_video_energy_matches(track: dict, tag_filter: tuple, video_time_ranges: dict = None) -> list[dict]:
    """Rank videos by SHAPE similarity to the track: variance (dynamic swings)
    and hit density (frequent standout moments), each independently z-scored
    against its own library (all candidate videos / all tracks in your audio
    folder) so the two domains sit on a comparable "how unusual for its own
    library" scale before being compared. See compute_track_features for why
    a simple level-to-level comparison doesn't differentiate tracks."""
    video_time_ranges = video_time_ranges or {}
    catalogues = load_all_catalogues()

    raw_video = {}
    scene_counts = {}
    for video_id, cat in catalogues.items():
        filtered = [s for s in cat["scenes"] if scene_is_usable(s, tag_filter)]
        time_range = video_time_ranges.get(video_id)
        if time_range:
            filtered = [
                s for s in filtered
                if _overlap_with_range(_tc_to_seconds(s["start_tc"]), _tc_to_seconds(s["end_tc"]), time_range)[0] is not None
            ]
        if not filtered:
            continue
        raw_video[video_id] = compute_video_features(video_id, tag_filter, video_time_ranges)
        scene_counts[video_id] = len(filtered)

    if not raw_video:
        return []

    video_var_z = _population_z({k: v["variance"] for k, v in raw_video.items()})
    video_hit_z = _population_z({k: v["hit_rate"] for k, v in raw_video.items()})
    video_lvl_z = _population_z({k: v["level"] for k, v in raw_video.items()})

    all_tracks = load_all_audio_tracks()
    track_feats = {tid: compute_track_features(t) for tid, t in all_tracks.items()}
    if track.get("track_id") not in track_feats:
        track_feats[track.get("track_id", "_current")] = compute_track_features(track)
    tvar_z = _population_z({k: v["variance"] for k, v in track_feats.items()})
    thit_z = _population_z({k: v["hit_rate"] for k, v in track_feats.items()})
    this_track_id = track.get("track_id") or "_current"
    t_var, t_hit = tvar_z[this_track_id], thit_z[this_track_id]

    results = []
    for video_id, feat in raw_video.items():
        diff = round(abs(video_var_z[video_id] - t_var) + abs(video_hit_z[video_id] - t_hit), 3)
        results.append({
            "video_id": video_id,
            "avg_motion_norm": round((video_lvl_z[video_id] + 3) / 6, 3),  # kept only for the on-screen label
            "variance_z": round(video_var_z[video_id], 2),
            "hit_rate_z": round(video_hit_z[video_id], 2),
            "matching_scene_count": scene_counts[video_id],
            "diff_from_track": diff,
        })
    return sorted(results, key=lambda r: r["diff_from_track"])


RECOMMENDED_COUNT = 10   # how many best-matching videos are listed by default


@st.cache_data(max_entries=20)  # bounded, same pattern as load_motion_curves_for_video
def load_timeline_sprite(sprite_path: str):
    """The whole per-video sprite sheet, loaded and cached once per video —
    cropping a specific tile from it (get_nearest_thumbnail) is cheap and
    done fresh on every call, but re-reading the sprite FILE every time
    would not be."""
    img = Image.open(sprite_path)
    img.load()  # force full read now, so the cached object doesn't hold an open file handle
    return img


def get_nearest_thumbnail(catalogues: dict, video_id: str, clip_start_sec: float):
    """The single thumbnail closest to clip_start_sec, constrained to never be
    from BEFORE it — so what's shown is representative of what the clip
    actually opens on, never a moment that gets trimmed away before the clip
    even starts. Crops one tile out of the video's timeline sprite sheet (see
    backfill_timeline_thumbnails.py). Returns a PIL Image, or None if this
    video has no sprite yet (backfill not run) or the sprite file is missing
    locally. If clip_start_sec falls past the sprite's last sampled second
    (an edge case — the clip starts later than any thumbnail was taken), the
    last available tile is used, since no at-or-after tile exists to pick."""
    meta = catalogues.get(video_id, {}).get("timeline_thumbnails")
    if not meta:
        return None
    # Derived from the LOCAL CATALOGUE_DIR, never from meta["sprite_path"] — that field
    # was written by whichever environment ran the timeline-thumbnail backfill (Colab,
    # the Windows laptop, or the Hetzner server), so it can point somewhere that doesn't
    # exist, or worse, somewhere unrelated, on whichever environment reads it back.
    sprite_path = CATALOGUE_DIR / "timeline_sprites" / f"{video_id}.jpg"
    if not Path(sprite_path).exists():
        return None
    sprite = load_timeline_sprite(str(sprite_path))

    tile_w, tile_h, columns, count, interval = (
        meta["tile_width"], meta["tile_height"], meta["columns"], meta["count"], meta["interval_sec"]
    )
    # ceil, not round: the nearest tile at-or-after clip_start_sec, never one before it.
    idx = min(count - 1, max(0, math.ceil(clip_start_sec / interval)))
    r, c = divmod(idx, columns)
    return sprite.crop((c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h))


def resolve_video_selection(track: dict, tag_filter: tuple, all_video_ids: list) -> dict:
    """Which videos are recommended (the top-10 ranking) and which are
    currently selected, read entirely from session_state — called by the
    Video Selection fragment on every one of its own reruns.

    The RANKING itself (video_matches/top_ids/other_options) is cached and
    only recomputed when the track or tag filter actually changes — ticking
    a checkbox, adding something from the dropdown, or setting a time range
    doesn't touch either, so the top-10 list stays stable through all of
    that instead of recomputing (and visually redrawing) on every click.
    One consequence worth knowing: a video's displayed match stats (shape
    distance, variance/hit-rate z, matching scene count) reflect whatever
    they were when the track/tag filter was last (re)selected — applying a
    time range won't immediately update those numbers, by the same design.

    SELECTION (who's ticked) is never cached — always computed fresh, since
    it's cheap and must reflect the current checkboxes exactly."""
    cache_key = (track.get("track_id"), tag_filter)
    cache = st.session_state.get("video_ranking_cache")
    if cache and cache["key"] == cache_key:
        ranking = cache["ranking"]
    else:
        video_matches = compute_video_energy_matches(track, tag_filter, st.session_state["video_time_ranges"])
        top_matches = video_matches[:RECOMMENDED_COUNT]
        top_ids = [v["video_id"] for v in top_matches]
        match_by_id = {v["video_id"]: v for v in video_matches}
        rank_order = {v["video_id"]: i for i, v in enumerate(video_matches)}
        other_options = sorted((vid for vid in all_video_ids if vid not in top_ids),
                               key=lambda vid: (rank_order.get(vid, 10**6), vid))
        ranking = {
            "video_matches": video_matches, "top_matches": top_matches, "top_ids": top_ids,
            "match_by_id": match_by_id, "rank_order": rank_order, "other_options": other_options,
        }
        st.session_state["video_ranking_cache"] = {"key": cache_key, "ranking": ranking}

    # A stored pick can drop out of the options (e.g. it entered the top 10 after a tag change) —
    # filtered here regardless of whether the fragment's own widget-safety sanitizing has run.
    extra_videos_stored = [v for v in st.session_state.get("extra_videos", []) if v in ranking["other_options"]]
    extra_rows = list(extra_videos_stored)
    # A video with a custom time range must stay listed even if the range left it with no
    # matching scenes — otherwise its 🎚️ controls would vanish and the range couldn't be cleared.
    extra_rows += [vid for vid in st.session_state["video_time_ranges"]
                  if vid not in ranking["top_ids"] and vid not in extra_rows]
    # A previously committed selection must also stay listed even if the "extra_videos" multiselect's
    # OWN widget state was discarded by Streamlit for having gone unrendered while the user viewed a
    # different section (this fragment's widgets only run while this section is the active one).
    # committed_selected_videos is a plain session_state entry, not tied to any widget, so it survives
    # that and is the right thing to self-heal from.
    extra_rows += [vid for vid in st.session_state.get("committed_selected_videos", [])
                  if vid not in ranking["top_ids"] and vid not in extra_rows and vid in ranking["other_options"]]
    selected_videos = [vid for vid in ranking["top_ids"] + extra_rows if st.session_state.get(f"select_video_{vid}", False)]
    return {
        **ranking,
        "extra_videos_stored": extra_videos_stored, "extra_rows": extra_rows,
        "selected_videos": selected_videos,
    }


AUDIO_SETTINGS_KEYS = [
    "segmentation_method", "beats_per_bar",
    "change_window_secs", "change_threshold", "min_segment_sec", "max_segment_sec",
    "use_bar_snapping", "react_to_hits", "hit_threshold", "snap_secs",
    "hyper_delta_thresh", "fast_energy_thresh", "hyper_cooldown_bars",
    "phrase_lock_bars", "cinematic_bars", "dynamic_priority_override", "override_delta_thresh",
]


@fragment
def render_audio_settings_section(track: dict) -> None:
    """Step 0: all segmentation dials plus a live audio chart — so you can
    tune the cuts while watching exactly where they land on the waveform.
    Runs as a fragment so adjusting any dial only reruns this section, not
    the full page.

    This section's widgets only ever run while THIS section is the active
    one (the other sections use st.stop() before reaching here) — and
    Streamlit discards session_state for widgets that weren't instantiated
    on a given run. So every widget below is given an explicit value=/index=
    sourced from a plain, non-widget "shadow" copy (audio_settings_shadow),
    saved at the end of every call here — rather than relying on key= alone
    to survive the round trip back from another section. key= still wins
    whenever its own state DOES exist (i.e. on every normal interaction
    while staying on this section), so nothing about dragging a slider
    changes; value= only matters the moment this section is re-entered."""
    _shadow = st.session_state.get("audio_settings_shadow", {})

    def _v(key):
        return st.session_state.get(key, _shadow.get(key, DIAL_DEFAULTS.get(key)))

    st.subheader("Segmentation method")
    _seg_options = ["Adaptive (energy-change + hits)", "Rhythm Engine (bar cutting styles)"]
    _seg_default = _v("segmentation_method")
    segmentation_method = st.radio(
        "Method", _seg_options,
        index=_seg_options.index(_seg_default) if _seg_default in _seg_options else 0,
        key="segmentation_method",
        help="Adaptive: detects individual cut points from energy changes and hits, ported from the VSE "
             "add-on's block detector. Rhythm Engine: assigns a named cutting STYLE to each bar based on its "
             "energy and momentum — CINEMATIC (slow, multi-bar), FAST (one segment per bar), or HYPER (one "
             "segment per beat) — a different feel driven by musical structure rather than individual events.",
    )
    beats_per_bar = st.number_input(
        "Beats per bar", min_value=2, max_value=12, value=int(_v("beats_per_bar")), key="beats_per_bar",
        help="4 = common time (most pop/rock/EDM). Use 3 for a waltz, 6 for 6/8, etc.",
    )

    if segmentation_method == "Adaptive (energy-change + hits)":
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Energy-change window (s)", 0.5, 6.0, value=float(_v("change_window_secs")), step=0.1,
                      key="change_window_secs",
                      help="Compares average energy over this long just after each moment with just before it.")
            st.slider("Energy-change threshold", 0.02, 0.5, value=float(_v("change_threshold")), step=0.01,
                      key="change_threshold",
                      help="How big a jump in average energy counts as a structural change. Lower = more cuts.")
            st.slider("Min segment length (s)", 0.3, 8.0, value=float(_v("min_segment_sec")), step=0.1,
                      key="min_segment_sec",
                      help="No two cuts can be closer than this.")
            st.slider("Max segment length (s)", 0.0, 30.0, value=float(_v("max_segment_sec")), step=0.5,
                      key="max_segment_sec",
                      help="Segments longer than this are split at their strongest beat. 0 = no limit.")
        with col2:
            st.checkbox("Snap structural cuts to bar lines", value=bool(_v("use_bar_snapping")),
                        key="use_bar_snapping",
                        help="Structural cuts land on the nearest musical bar line instead of raw signal crossings.")
            react = st.checkbox("Cut on big hits", value=bool(_v("react_to_hits")), key="react_to_hits",
                                help="Adds segment boundaries at the strongest onsets.")
            st.slider("Big-hit threshold", 0.5, 1.0, value=float(_v("hit_threshold")), step=0.01,
                      key="hit_threshold", disabled=not react,
                      help="Onset strength (0-1) a hit needs to become a cut.")
            st.slider("Snap radius (s)", 0.0, 1.0, value=float(_v("snap_secs")), step=0.05, key="snap_secs",
                      help="Energy-change cuts move onto the strongest hit within this distance.")
        # local variables for the preview chart below
        _seg_method = "adaptive"
        _hyper_delta = _fast_thresh = 0.0
        _hyper_cd = _phrase_lock = _cine_bars = 0
        _dpo = False
        _override_thresh = 0.0
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Hyper delta threshold", 0.05, 1.0, value=float(_v("hyper_delta_thresh")), step=0.01,
                      key="hyper_delta_thresh",
                      help="How sharp a bar-to-bar energy jump triggers HYPER.")
            st.slider("Fast energy threshold", 0.1, 1.0, value=float(_v("fast_energy_thresh")), step=0.01,
                      key="fast_energy_thresh",
                      help="Bars above this (and no sharp jump) become FAST instead of CINEMATIC.")
            st.number_input("Hyper cooldown (bars)", min_value=1, max_value=8,
                            value=int(_v("hyper_cooldown_bars")), key="hyper_cooldown_bars",
                            help="HYPER is forced to exit after this many bars.")
        with col2:
            st.number_input("Phrase lock (bars)", min_value=1, max_value=8,
                            value=int(_v("phrase_lock_bars")), key="phrase_lock_bars",
                            help="FAST or CINEMATIC is held for this many bars before re-evaluating.")
            st.number_input("Cinematic segment length (bars)", min_value=1, max_value=8,
                            value=int(_v("cinematic_bars")), key="cinematic_bars",
                            help="How many bars a CINEMATIC segment spans.")
            dpo = st.checkbox("Dynamic Priority Override", value=bool(_v("dynamic_priority_override")),
                              key="dynamic_priority_override",
                              help="Adds extra cuts at instantaneous RMS jumps the bar-averaged state machine smooths away.")
            st.slider("Override sensitivity", 0.02, 1.0, value=float(_v("override_delta_thresh")), step=0.01,
                      key="override_delta_thresh", disabled=not dpo,
                      help="How large a single-frame RMS jump triggers a forced cut.")
        _seg_method = "rhythm"
        _hyper_delta = st.session_state.get("hyper_delta_thresh", 0.35)
        _fast_thresh = st.session_state.get("fast_energy_thresh", 0.75)
        _hyper_cd = int(st.session_state.get("hyper_cooldown_bars", 2))
        _phrase_lock = int(st.session_state.get("phrase_lock_bars", 4))
        _cine_bars = int(st.session_state.get("cinematic_bars", 4))
        _dpo = st.session_state.get("dynamic_priority_override", False)
        _override_thresh = st.session_state.get("override_delta_thresh", 0.15)

    # ---- Live preview chart ------------------------------------------------
    st.divider()
    st.subheader("Preview: cut points on the audio signal")
    st.caption("Adjust dials above and the chart updates immediately. "
               "Scene match dots appear on the Matching & Export step once clips are assigned.")

    # Build segments from current widget values for the preview
    _cws = st.session_state.get("change_window_secs", 2.0)
    _ct = st.session_state.get("change_threshold", 0.12)
    _mins = st.session_state.get("min_segment_sec", 2.0)
    _maxs = st.session_state.get("max_segment_sec", 8.0)
    _rth = st.session_state.get("react_to_hits", True)
    _ht = st.session_state.get("hit_threshold", 0.9)
    _snap = st.session_state.get("snap_secs", 0.3)
    _ubs = st.session_state.get("use_bar_snapping", True)
    _bpb = int(beats_per_bar)
    _dc = st.session_state.get("density_contrast", 1.0)
    _min_c = st.session_state.get("min_clips", 1)
    _max_c = st.session_state.get("max_clips", 4) if st.session_state.get("split_screen_enabled") else 1
    _rs, _re = st.session_state.get("ramp_range", (0.2, 0.85))

    if segmentation_method == "Adaptive (energy-change + hits)":
        _segs = build_track_segments(track, _cws, _ct, _mins, _rth, _ht, _snap, _maxs, _bpb, _ubs)
    else:
        _segs = build_rhythm_engine_segments(track, _bpb, _hyper_delta, _fast_thresh,
                                             _hyper_cd, _phrase_lock, _cine_bars, _dpo, _override_thresh)
    apply_segment_intensity(_segs, track, _dc)
    _rec_counts = [
        clips_for_intensity(sg["intensity"], _min_c, _max_c, _rs, _re, False, None)
        for sg in _segs
    ]

    afig = go.Figure()
    if track.get("hires"):
        _rate = float(track["hires"]["rate"])
        _t = [i / _rate for i in range(len(track["hires"]["rms"]))]
        afig.add_trace(go.Scatter(x=_t, y=track["hires"]["rms"], mode="lines",
                                  name="Track energy (30 Hz)",
                                  line=dict(color="rgba(100,150,255,0.5)", width=1)))
        afig.add_trace(go.Scatter(x=_t, y=track["hires"]["onset"], mode="lines",
                                  name="Onset strength",
                                  line=dict(color="rgba(150,150,150,0.35)", width=1),
                                  visible="legendonly"))
    else:
        afig.add_trace(go.Scatter(x=[e["time"] for e in track["energy_envelope"]],
                                  y=[e["energy"] for e in track["energy_envelope"]],
                                  mode="lines", name="Track energy",
                                  line=dict(color="rgba(100,150,255,0.5)")))

    for _kind, _colour, _label in (
        ("change", "rgba(0,160,120,0.7)", "Cut: energy change (snapped)"),
        ("hit",    "rgba(220,60,60,0.55)", "Cut: big hit"),
        ("fill",   "rgba(230,150,30,0.6)", "Cut: max-length split"),
        ("cinematic", "rgba(80,120,220,0.5)", "Rhythm Engine: CINEMATIC"),
        ("fast",   "rgba(230,150,30,0.6)", "Rhythm Engine: FAST"),
        ("hyper",  "rgba(220,60,60,0.55)", "Rhythm Engine: HYPER"),
        ("override", "rgba(230,30,200,0.75)", "Rhythm Engine: Dynamic Priority Override"),
    ):
        _xs, _ys = [], []
        for _sg in _segs:
            if _sg.get("cut") == _kind:
                _xs += [_sg["start"], _sg["start"], None]
                _ys += [0, 1, None]
        if _xs:
            afig.add_trace(go.Scatter(x=_xs, y=_ys, mode="lines", name=_label,
                                      line=dict(color=_colour, width=1, dash="dot"),
                                      hoverinfo="skip"))

    _dt, _dens = compute_density_curve(track)
    afig.add_trace(go.Scatter(x=_dt, y=_dens, mode="lines", name="Density (1.5 s smoothed)",
                              line=dict(color="rgba(140,90,220,0.8)", width=2)))
    _cx, _cy = [], []
    for _sg, _rc in zip(_segs, _rec_counts):
        _cx += [_sg["start"], _sg["end"]]
        _cy += [_rc, _rc]
    afig.add_trace(go.Scatter(x=_cx, y=_cy, mode="lines", name="Recommended clips", yaxis="y2",
                              line=dict(color="rgba(30,30,30,0.85)", width=2, shape="hv")))

    afig.update_layout(
        height=400, margin=dict(t=20, b=20),
        xaxis_title="Time (s)", yaxis_title="Normalised level",
        yaxis2=dict(title="Clips", overlaying="y", side="right",
                    range=[0, _max_c + 0.5], dtick=1, showgrid=False),
        legend=dict(orientation="h", y=-0.28),
    )
    st.plotly_chart(afig, use_container_width=True)
    st.caption(f"{len(_segs)} segments from {format_mmss(0.0)} – {format_mmss(track['duration_sec'])}  "
               f"|  {track.get('bpm', '?')} BPM  |  switch to **Videos & Time Ranges** when happy with the cuts.")

    # Save the shadow copy every call — this is what actually survives the round trip to
    # another section (plain session_state, not tied to any widget), and is what the explicit
    # value=/index= above restore from on re-entry.
    st.session_state["audio_settings_shadow"] = {
        k: st.session_state[k] for k in AUDIO_SETTINGS_KEYS if k in st.session_state
    }


def _select_all_recommended():
    """Tick every video in the current top-RECOMMENDED_COUNT list. Runs as a button
    callback (before the page redraws), using the list saved by the previous run."""
    for vid in st.session_state.get("recommended_ids", []):
        st.session_state[f"select_video_{vid}"] = True


def _clear_video_selection():
    for key in [k for k in st.session_state if k.startswith("select_video_")]:
        st.session_state[key] = False


@fragment
def render_video_selection_section(track: dict, all_video_ids: list, all_tag_options: list, catalogues: dict) -> None:
    """Tag filter, Select all recommended / Clear, and the whole video list
    with checkboxes and time-range pickers — as a fragment, so ticking a
    checkbox, opening/closing a range picker, dragging its slider, or
    clicking Apply/Clear range only reruns THIS, not the rest of the page.
    Nothing here needs to force a full app rerun (unlike Skip Ahead
    elsewhere): the Matching & Export section only reads this selection when
    the user switches to it, which is itself a normal full rerun — by then
    session_state already holds whatever was last set here."""
    top_cols = st.columns([5, 2, 2])
    with top_cols[0]:
        tag_filter = st.multiselect(
            "Restrict to tags (optional)", all_tag_options, key="tag_filter",
            default=[t for t in st.session_state.get("committed_tag_filter", []) if t in all_tag_options],
            help="Only scenes carrying at least one of these tags are used — this also changes which videos "
                 "are recommended below.",
        )
    with top_cols[1]:
        st.write("")
        st.button("✅ Select all recommended", on_click=_select_all_recommended, use_container_width=True,
                  help=f"Tick the top {RECOMMENDED_COUNT} best-matching videos listed below.")
    with top_cols[2]:
        st.write("")
        st.button("Clear selection", on_click=_clear_video_selection, use_container_width=True)

    info = resolve_video_selection(track, tuple(tag_filter), all_video_ids)
    video_matches = info["video_matches"]
    st.session_state["recommended_ids"] = info["top_ids"]

    if not video_matches:
        st.warning("No videos have scenes matching the current tag filter.")
        return

    _SORT_OPTIONS = {
        "Shape distance (best match first)": lambda v: v["diff_from_track"],
        "Matching scenes (most first)":      lambda v: -v["matching_scene_count"],
        "Overall energy (highest first)":    lambda v: -v["avg_motion_norm"],
        "Total video length (longest first)": lambda v: -get_video_duration(v["video_id"]),
    }
    sort_by = st.selectbox(
        "Sort top videos by",
        list(_SORT_OPTIONS.keys()),
        index=0,
        key="video_sort_by",
        label_visibility="collapsed",
    )
    sorted_top_matches = sorted(info["top_matches"], key=_SORT_OPTIONS[sort_by])

    st.subheader(f"Best-matching videos (top {RECOMMENDED_COUNT})")
    st.caption(f"Matched on energy shape (variance + hit density), not average level" +
               (f"  |  filtered to tags: {', '.join(tag_filter)}" if tag_filter else ""))
    st.caption(
        "✅ marks a recommended video. ☑ is your actual selection — nothing is selected until you tick it "
        "(or use Select all recommended above), and it stays as you set it even if re-sorting moves the "
        "video. 🎚️ opens a range picker to restrict a video to just a portion of its footage."
    )

    def render_video_row(video_id: str, recommended: bool, default_checked: bool):
        match = info["match_by_id"].get(video_id)
        duration = get_video_duration(video_id)
        current_range = st.session_state["video_time_ranges"].get(video_id)
        range_tag = f"  🎚️ {seconds_to_time(current_range[0]):%M:%S}–{seconds_to_time(current_range[1]):%M:%S}" if current_range else ""

        row_cols = st.columns([1, 7, 1])
        with row_cols[0]:
            st.checkbox("select", value=default_checked, key=f"select_video_{video_id}",
                        label_visibility="collapsed")
        with row_cols[1]:
            rank_mark = "✅ " if recommended else ""
            title = f"{rank_mark}**{video_id}** · ⏱ {format_mmss(duration)}"
            if match:
                st.write(f"{title} — shape distance: {match['diff_from_track']}  |  "
                         f"variance z: {match['variance_z']}  |  hit-rate z: {match['hit_rate_z']}  |  "
                         f"{match['matching_scene_count']} matching scenes{range_tag}")
            else:
                st.write(f"{title} — no scenes match the current tag filter / time range{range_tag}")
        with row_cols[2]:
            if st.button("🎚️", key=f"range_toggle_{video_id}", help="Set a time range for this video"):
                st.session_state["show_range_picker"][video_id] = not st.session_state["show_range_picker"].get(video_id, False)
                st.rerun()

        if st.session_state["show_range_picker"].get(video_id):
            with st.container(border=True):
                source_path = colab_to_local(catalogues[video_id]["source_path"])
                if Path(source_path).exists():
                    st.video(source_path)
                else:
                    st.warning(f"Source video not found locally: {source_path}")

                slider_key = f"range_slider_{video_id}"
                if st.button("🎯 Auto Range (match audio length, highest motion)", key=f"auto_range_{video_id}"):
                    best_start, best_end = find_best_motion_window(video_id, track["duration_sec"])
                    st.session_state[slider_key] = (seconds_to_time(best_start), seconds_to_time(best_end))
                    st.rerun()

                default_range = current_range or (0.0, duration)
                new_range_t = st.slider(
                    f"Usable range for {video_id}",
                    min_value=seconds_to_time(0), max_value=seconds_to_time(max(duration, 1.0)),
                    value=(seconds_to_time(default_range[0]), seconds_to_time(default_range[1])),
                    step=datetime.timedelta(seconds=1), format="mm:ss",
                    key=slider_key,
                )

                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("Apply range", key=f"apply_range_{video_id}"):
                        st.session_state["video_time_ranges"][video_id] = (
                            time_to_seconds(new_range_t[0]), time_to_seconds(new_range_t[1])
                        )
                        st.rerun()
                with btn_cols[1]:
                    if st.button("Clear (use full video)", key=f"clear_range_{video_id}"):
                        st.session_state["video_time_ranges"].pop(video_id, None)
                        st.rerun()

    # committed_selected_videos is what actually survives a round trip to another section
    # (plain session_state, not a widget), so it's what default_checked restores from —
    # NOT a static True/False by row type, which would silently reset every tick the moment
    # the underlying select_video_* widget state gets discarded for having gone unrendered.
    _committed = st.session_state.get("committed_selected_videos", [])
    for _v in sorted_top_matches:
        render_video_row(_v["video_id"], recommended=True,
                         default_checked=(_v["video_id"] in _committed))

    st.session_state["extra_videos"] = info["extra_rows"]
    st.multiselect(
        "➕ Add other videos from the library", info["other_options"], key="extra_videos",
        format_func=lambda vid: f"{vid}  ·  {format_mmss(get_video_duration(vid))}",
        placeholder="Choose a video to add it to the list above…",
        help="Everything in your video folder that isn't in the top list. Picking one adds it below, ticked.",
    )
    extra_rows = list(st.session_state["extra_videos"])
    extra_rows += [vid for vid in st.session_state["video_time_ranges"]
                  if vid not in info["top_ids"] and vid not in extra_rows]
    # Distinguishes "just added from the dropdown this run" (defaults to ticked, the existing
    # convenience of adding a video) from "already in the extra list before" (respects whatever
    # was last committed, ticked or not) — both read from plain session_state, so both survive
    # a round trip to another section even if the underlying checkbox widgets don't.
    _known_extra = set(st.session_state.get("known_extra_videos", []))
    for _vid in extra_rows:
        default_checked = (_vid in _committed) if _vid in _known_extra else True
        render_video_row(_vid, recommended=False, default_checked=default_checked)
    st.session_state["known_extra_videos"] = list(set(extra_rows) | _known_extra)

    selected_now = [vid for vid in info["top_ids"] + extra_rows if st.session_state.get(f"select_video_{vid}", False)]
    # Committed to plain (non-widget) session_state keys every time this fragment runs — the
    # Matching section reads these directly rather than re-deriving them from individual widget
    # keys (select_video_*, extra_videos) from outside the fragment that owns them.
    st.session_state["committed_selected_videos"] = selected_now
    st.session_state["committed_tag_filter"] = tag_filter
    if not selected_now:
        st.info("No videos selected yet. Tick the videos you want above, use **Select all recommended**, or add "
                "others from the dropdown — switch to the Matching & Export tab once you're ready.")
    else:
        st.success(f"{len(selected_now)} video(s) selected. Switch to the **Matching & Export** tab above when ready.")


def _moving_average(values, half_window: int):
    """Symmetric moving average with edge clamping (cumulative-sum version,
    fast enough for 30 Hz signals)."""
    values = np.asarray(values, dtype=float)
    if half_window <= 0:
        return values.copy()
    n = len(values)
    c = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(n, idx + half_window + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def compute_bar_times(track: dict, beats_per_bar: int = 4) -> list:
    """Group detected beats into bars, assuming a constant beats_per_bar
    (4 = common/4-4 time — most pop, rock, EDM; use 3 for a waltz, etc.).
    Anchored at the first detected beat, which is a reasonable default for
    most consistent-tempo tracks without a pickup beat. Purely derived from
    the beat_times audio_pipeline.py already stores in Colab — no new audio
    processing or re-running the backfill needed for this."""
    beats = track.get("beat_times") or []
    if len(beats) < beats_per_bar:
        return []
    return beats[0::beats_per_bar]


def _snap_to_bar_or_onset(i: int, bar_frames: np.ndarray, onset: np.ndarray, snap: int, n: int) -> int:
    """Snap position i onto the nearest bar line if one is close enough to be
    plausibly the SAME musical moment (within 1.5 bars, or 3x the onset-snap
    radius if bars are sparse/unavailable) — otherwise fall back to the
    original onset-peak snap, same as before bar-awareness existed."""
    if len(bar_frames) > 0:
        nearest = int(np.argmin(np.abs(bar_frames - i)))
        bar_pos = int(bar_frames[nearest])
        bar_spacing = float(np.median(np.diff(bar_frames))) if len(bar_frames) > 1 else float(n)
        if abs(bar_pos - i) <= max(bar_spacing * 1.5, snap * 3):
            return bar_pos
    lo, hi = max(1, i - snap), min(n - 1, i + snap + 1)
    return lo + int(np.argmax(onset[lo:hi])) if hi > lo else int(i)


# ---------------------------------------------------------------------------
# Rhythm Engine: an alternative segmentation method, ported from a reference
# state-machine design. Instead of detecting individual cut POINTS, it works
# at bar granularity — each bar gets assigned a cutting STYLE (CINEMATIC =
# slow, multi-bar segments; FAST = one segment per bar; HYPER = one segment
# per BEAT) based on that bar's smoothed energy and how sharply it jumped
# from before. Phrase-locking commits to a style for several bars at a time
# so it doesn't flicker; a cooldown forces an exit from HYPER after a few
# bars so rapid cutting can't run away. This is a genuinely different feel
# from the Adaptive method above — named "styles" driven by musical momentum,
# rather than reacting to individual detected events.
# ---------------------------------------------------------------------------

RHYTHM_STATE_CINEMATIC = "CINEMATIC"
RHYTHM_STATE_FAST = "FAST"
RHYTHM_STATE_HYPER = "HYPER"


def compute_cutting_states(
    track: dict,
    beats_per_bar: int = 4,
    hyper_delta_thresh: float = 0.35,
    fast_energy_thresh: float = 0.75,
    hyper_cooldown_bars: int = 2,
    phrase_lock_bars: int = 4,
) -> list:
    """Assigns a cutting style to every bar. Returns a list of state strings,
    one per bar (same length/order as compute_bar_times' output), or an
    empty list if there isn't enough beat/hires data to form bars."""
    bar_times = compute_bar_times(track, beats_per_bar)
    hires = track.get("hires")
    if len(bar_times) < 2 or not hires:
        return []

    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    bar_frames = [round(bt * rate) for bt in bar_times] + [len(rms)]

    bar_energies = np.array([
        float(rms[bar_frames[i]:bar_frames[i + 1]].mean()) if bar_frames[i + 1] > bar_frames[i] else 0.0
        for i in range(len(bar_times))
    ])
    max_e = bar_energies.max() if bar_energies.max() > 0 else 1.0
    norm = bar_energies / max_e
    local_smooth = np.convolve(norm, np.ones(2) / 2, mode="same")
    deltas = np.diff(norm, prepend=norm[0])

    states = []
    current_state = RHYTHM_STATE_CINEMATIC
    hyper_cooldown = 0
    phrase_lock = 0

    for i in range(len(bar_times)):
        energy, delta = local_smooth[i], deltas[i]
        if hyper_cooldown > 0:
            hyper_cooldown -= 1
        if phrase_lock > 0:
            phrase_lock -= 1
            states.append(current_state)
            continue

        if current_state == RHYTHM_STATE_HYPER and hyper_cooldown == 0:
            current_state = RHYTHM_STATE_FAST if energy > fast_energy_thresh else RHYTHM_STATE_CINEMATIC

        if delta > hyper_delta_thresh and hyper_cooldown == 0:
            current_state = RHYTHM_STATE_HYPER
            hyper_cooldown = hyper_cooldown_bars
            phrase_lock = hyper_cooldown_bars
        elif energy > fast_energy_thresh:
            if current_state != RHYTHM_STATE_FAST:
                current_state = RHYTHM_STATE_FAST
                phrase_lock = phrase_lock_bars
        else:
            if current_state != RHYTHM_STATE_CINEMATIC:
                current_state = RHYTHM_STATE_CINEMATIC
                phrase_lock = phrase_lock_bars

        states.append(current_state)

    return states


MIN_OVERRIDE_GAP_SEC = 0.15  # an override this close to an existing boundary is skipped, not a degenerate sliver


def find_dynamic_priority_cuts(track: dict, bar_times: list, override_delta_thresh: float) -> list:
    """Secondary pass, run right after the main state-machine loop: scans the
    RAW, frame-by-frame derivative of RMS — NOT the bar-averaged, 2-bar-
    smoothed energy the state machine itself uses — for instantaneous jumps
    that view can miss entirely. The state machine's smoothing is
    deliberately built to filter out brief spikes (right for overall
    pacing), but that can also wash out a single dramatic hit or drop that
    happens to land mid-bar. For each bar where the raw derivative's peak
    exceeds override_delta_thresh, returns the EXACT time (in seconds) of
    that peak frame — a forced cut point, independent of whatever state the
    state machine assigned that bar."""
    hires = track.get("hires")
    if not hires or len(bar_times) < 2:
        return []
    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    raw_deltas = np.abs(np.diff(rms, prepend=rms[0]))

    bar_frames = [round(bt * rate) for bt in bar_times] + [len(rms)]
    override_times = []
    for i in range(len(bar_times)):
        s, e = bar_frames[i], bar_frames[i + 1]
        if e <= s:
            continue
        window = raw_deltas[s:e]
        peak_idx = int(np.argmax(window))
        if window[peak_idx] > override_delta_thresh:
            override_times.append((s + peak_idx) / rate)
    return override_times


def apply_dynamic_priority_overrides(raw_segments: list, override_times: list) -> list:
    """Splits whichever segment each override time falls inside, at that
    EXACT point — independent of the segment's original kind or boundaries.
    The piece AFTER the split is tagged "override", matching the existing
    convention that a segment's "cut" field names why ITS boundary exists.
    An override too close to an already-existing boundary is skipped rather
    than creating a near-zero-length sliver segment."""
    result = list(raw_segments)
    for ot in sorted(override_times):
        for idx, (s, e, kind) in enumerate(result):
            if s + MIN_OVERRIDE_GAP_SEC < ot < e - MIN_OVERRIDE_GAP_SEC:
                result[idx:idx + 1] = [(s, ot, kind), (ot, e, "override")]
                break
    return result


def build_rhythm_engine_segments(
    track: dict,
    beats_per_bar: int = 4,
    hyper_delta_thresh: float = 0.35,
    fast_energy_thresh: float = 0.75,
    hyper_cooldown_bars: int = 2,
    phrase_lock_bars: int = 4,
    cinematic_bars: int = 4,
    dynamic_priority_override: bool = False,
    override_delta_thresh: float = 0.15,
) -> list:
    """Converts the per-bar cutting states into actual segments, in the same
    {"start","end","energy","cut","cut_strength"} schema build_track_segments
    produces — so everything downstream (matching, the intensity/clip-count
    ramp, Advanced mode, the chart) works with either segmentation method
    unchanged. CINEMATIC runs are grouped into cinematic_bars-bar chunks;
    FAST bars each become their own segment; HYPER runs are subdivided by
    BEAT (the fastest cut rate available). Falls back to the whole track as
    one segment if there isn't enough data to form even one bar."""
    duration = float(track["duration_sec"])
    hires = track.get("hires")
    bar_times = compute_bar_times(track, beats_per_bar)
    states = compute_cutting_states(track, beats_per_bar, hyper_delta_thresh,
                                    fast_energy_thresh, hyper_cooldown_bars, phrase_lock_bars)

    if len(bar_times) < 2 or not states:
        energy = float(np.mean(hires["rms"])) if hires else 0.0
        return [{"start": 0.0, "end": duration, "energy": round(energy, 3), "cut": "start", "cut_strength": 0.0}]

    bar_bounds = list(bar_times) + [duration]
    beat_times = track.get("beat_times") or []

    raw_segments = []  # (start, end, cut_kind)
    i = 0
    while i < len(states):
        state = states[i]
        if state == RHYTHM_STATE_HYPER:
            j = i
            while j < len(states) and states[j] == RHYTHM_STATE_HYPER:
                j += 1
            bar_start, bar_end = bar_bounds[i], bar_bounds[j]
            beats_in_range = [b for b in beat_times if bar_start < b < bar_end]
            bounds = sorted({bar_start, *beats_in_range, bar_end})
            for k in range(len(bounds) - 1):
                raw_segments.append((bounds[k], bounds[k + 1], "hyper"))
            i = j
        elif state == RHYTHM_STATE_FAST:
            j = i
            while j < len(states) and states[j] == RHYTHM_STATE_FAST:
                raw_segments.append((bar_bounds[j], bar_bounds[j + 1], "fast"))
                j += 1
            i = j
        else:
            j = i
            while j < len(states) and states[j] == RHYTHM_STATE_CINEMATIC:
                j += 1
            k = i
            while k < j:
                chunk_end = min(k + max(1, cinematic_bars), j)
                raw_segments.append((bar_bounds[k], bar_bounds[chunk_end], "cinematic"))
                k = chunk_end
            i = j

    raw_segments.sort()
    if dynamic_priority_override:
        override_times = find_dynamic_priority_cuts(track, bar_times, override_delta_thresh)
        raw_segments = apply_dynamic_priority_overrides(raw_segments, override_times)

    rate = float(hires["rate"]) if hires else 0.0
    rms = np.asarray(hires["rms"], dtype=float) if hires else np.array([])
    onset = np.asarray(hires["onset"], dtype=float) if hires else np.array([])

    segments = []
    for start, end, kind in raw_segments:
        if end <= start:
            continue
        s_f, e_f = int(round(start * rate)), int(round(end * rate))
        energy = float(rms[s_f:e_f].mean()) if e_f > s_f and len(rms) else 0.0
        strength = float(onset[s_f]) if 0 <= s_f < len(onset) else 0.0
        segments.append({
            "start": round(start, 3),
            "end": round(min(end, duration), 3),
            "energy": round(energy, 3),
            "cut": kind,
            "cut_strength": round(strength, 3),
        })
    return segments


def build_track_segments(
    track: dict,
    change_window_secs: float,
    change_threshold: float,
    min_block_secs: float,
    react_to_hits: bool,
    hit_threshold: float,
    snap_secs: float,
    max_block_secs: float = 0.0,
    beats_per_bar: int = 4,
    use_bar_snapping: bool = True,
) -> list[dict]:
    """Tight segmentation on the 30 Hz signals from audio_pipeline.py.

    1. ENERGY CHANGES — a step detector: at every sample, compare the mean
       energy over the next `change_window_secs` with the mean over the
       previous `change_window_secs`. Peaks in that difference above
       `change_threshold` are genuine level changes (drops, build-ups,
       chorus in/out). Steady beat ripple averages out, so it can't slowly
       accumulate into false cuts.
    2. SNAP — each energy-change cut moves onto the nearest BAR LINE (derived
       from the track's own beat grid, assuming beats_per_bar beats per bar)
       if one is close enough to plausibly be the same musical moment —
       otherwise it falls back to the strongest nearby onset (the add-on's
       "Clip Change" score) within ±snap_secs, same as before bar-awareness
       existed. This is what keeps segment LENGTHS feeling musically
       consistent — a structural change lands on a real phrase boundary
       (start of a bar) instead of wherever the raw energy signal happened
       to cross a threshold. use_bar_snapping=False restores the pure
       onset-snap behaviour if you'd rather not quantize to bars.
    3. BIG HITS — onset peaks at or above hit_threshold become extra cuts,
       landing exactly on the transient, NOT snapped to a bar line — a hit
       can legitimately fall mid-bar, and forcing it onto the bar grid would
       undo the point of reacting to it individually.

    Candidates are accepted strongest-first (energy changes rank above hits),
    skipping any closer than min_block_secs to an accepted cut, so the most
    important moments always win.

    4. MAX LENGTH — like the add-on's Build Signal Grid Plan (which chunks long
       blocks and snaps chunk boundaries to hits), any segment longer than
       max_block_secs is split at its strongest internal onset, repeatedly,
       so evenly-busy passages with no standout hit still get cut on a beat.
       0 = off.

    Tracks analysed before the 30 Hz signals existed fall back to the older
    coarse method (re-run audio_pipeline.py in Colab to upgrade them)."""
    hires = track.get("hires")
    if not hires:
        return _legacy_track_segments(track, change_window_secs, 0.5, 0.5, min_block_secs, react_to_hits)

    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    onset = np.asarray(hires["onset"], dtype=float)
    n = min(len(rms), len(onset))
    rms, onset = rms[:n], onset[:n]
    duration = float(track["duration_sec"])
    if n < 3:
        return [{"start": 0.0, "end": duration, "energy": float(rms.mean()) if n else 0.0,
                 "cut": "start", "cut_strength": 0.0}]

    w = max(1, round(rate * change_window_secs))
    min_len = max(1, round(rate * min_block_secs))
    snap = max(0, round(rate * snap_secs))

    bar_times = compute_bar_times(track, beats_per_bar) if use_bar_snapping else []
    bar_frames = np.array([round(bt * rate) for bt in bar_times]) if bar_times else np.array([])

    # 1. step detector: |mean(after) - mean(before)|
    c = np.concatenate([[0.0], np.cumsum(rms)])
    idx = np.arange(n)
    b_lo, a_hi = np.maximum(0, idx - w), np.minimum(n, idx + w)
    before = (c[idx] - c[b_lo]) / np.maximum(1, idx - b_lo)
    after = (c[a_hi] - c[idx]) / np.maximum(1, a_hi - idx)
    change = np.abs(after - before)
    change[:1] = 0.0
    change_peaks = np.flatnonzero((change[1:-1] >= change[:-2]) & (change[1:-1] > change[2:])) + 1

    candidates = []  # (score, index, kind)
    for i in change_peaks:
        if change[i] < change_threshold:
            continue
        # 2. snap onto the nearest bar line, or the strongest nearby onset if none is close
        pos = _snap_to_bar_or_onset(i, bar_frames, onset, snap, n)
        candidates.append((1.0 + float(change[i]), int(pos), "change"))

    # 3. big hits
    if react_to_hits:
        peaks = np.flatnonzero((onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:])) + 1
        for pos in peaks:
            if onset[pos] >= hit_threshold:
                candidates.append((float(onset[pos]), int(pos), "hit"))

    accepted = []
    for score, pos, kind in sorted(candidates, key=lambda x: -x[0]):
        if pos < min_len or n - pos < min_len:
            continue
        if any(abs(pos - a) < min_len for a, _ in accepted):
            continue
        accepted.append((pos, kind))
    accepted.sort()

    # 4. split over-long segments at their strongest internal onset
    if max_block_secs and max_block_secs > 0:
        max_len = max(2 * min_len, round(rate * max_block_secs))
        onset_peak = np.zeros(n, dtype=bool)
        onset_peak[1:-1] = (onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:])
        changed = True
        while changed:
            changed = False
            edges = [0] + [a for a, _ in accepted] + [n]
            for s0, e0 in zip(edges[:-1], edges[1:]):
                if e0 - s0 <= max_len:
                    continue
                lo, hi = s0 + min_len, e0 - min_len
                if hi <= lo:
                    continue
                bars_in_range = bar_frames[(bar_frames > lo) & (bar_frames < hi)] if len(bar_frames) else np.array([])
                if len(bars_in_range):
                    mid = (lo + hi) / 2
                    fill_pos = int(bars_in_range[np.argmin(np.abs(bars_in_range - mid))])
                else:
                    window = np.where(onset_peak[lo:hi], onset[lo:hi], -1.0)
                    if window.max() < 0:
                        window = onset[lo:hi]
                    fill_pos = lo + int(np.argmax(window))
                accepted.append((fill_pos, "fill"))
                changed = True
            accepted.sort()

    bounds = [0] + [a for a, _ in accepted] + [n]
    kinds = ["start"] + [k for _, k in accepted]
    segments = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        segments.append({
            "start": round(s / rate, 3),
            "end": round(min(e / rate, duration), 3) if e < n else duration,
            "energy": round(float(rms[s:e].mean()), 3),
            "cut": kinds[k],
            "cut_strength": round(float(onset[s]), 3),
        })
    return segments


def _legacy_track_segments(
    track: dict,
    smooth_secs: float,
    lookback_secs: float,
    drift_threshold: float,
    min_block_secs: float,
    react_to_beats: bool = True,
) -> list[dict]:
    """Adaptive segmentation ported from the VSE add-on's block-detection
    algorithm: cut a new segment wherever the smoothed energy level has
    drifted from where it was `lookback_secs` ago by more than
    `drift_threshold` (accumulated). This alone only reacts to broad energy
    LEVEL shifts, which is too slow for sharp hits — so when react_to_beats
    is on, every detected beat is ALSO forced in as a cut candidate, keeping
    blocks reactive to individual hits, not just plateau changes. Either way,
    any resulting span shorter than `min_block_secs` is merged into whichever
    neighbour is more energetic, so beat-forcing can't fragment things below
    your chosen floor. Operates on the track's energy envelope (sampled every
    ENERGY_WINDOW_SEC seconds in audio_pipeline.py)."""
    envelope = track["energy_envelope"]
    times = np.array([e["time"] for e in envelope])
    energies = np.array([e["energy"] for e in envelope])
    n = len(energies)
    if n < 2:
        return [{"start": 0.0, "end": track["duration_sec"], "energy": float(energies[0]) if n else 0.0}]

    sample_rate = 1.0 / max(times[1] - times[0], 1e-6)  # samples per second
    hw_smooth = max(1, round(sample_rate * smooth_secs / 2.0))
    lb = max(1, round(sample_rate * lookback_secs))
    min_samples = max(1, round(sample_rate * min_block_secs))

    smoothed = _moving_average(energies, hw_smooth)

    # --- cut boundaries wherever accumulated drift crosses the threshold ---
    drift_boundaries = {0, n}
    accumulator = 0.0
    since_last = 0
    for i in range(n):
        since_last += 1
        prev_idx = max(0, i - lb)
        accumulator += abs(smoothed[i] - smoothed[prev_idx])
        if accumulator >= drift_threshold and since_last >= min_samples and i > 0:
            drift_boundaries.add(i)
            accumulator = 0.0
            since_last = 0

    # --- fold in every beat as a forced cut candidate, so blocks react to hits too ---
    if react_to_beats and track.get("beat_times"):
        for bt in track["beat_times"]:
            idx = int(round(bt * sample_rate))
            if 0 < idx < n:
                drift_boundaries.add(idx)

    raw_boundaries = sorted(drift_boundaries)

    spans = [(raw_boundaries[k], raw_boundaries[k + 1]) for k in range(len(raw_boundaries) - 1)
             if raw_boundaries[k] < raw_boundaries[k + 1]]

    # --- merge pass: absorb spans shorter than min_samples into the more energetic neighbour ---
    def avg_smoothed(span):
        lo, hi = span
        return float(smoothed[lo:hi].mean())

    changed = True
    while changed and len(spans) > 1:
        changed = False
        for k in range(len(spans)):
            s, e = spans[k]
            if (e - s) < min_samples:
                if k == 0:
                    ns, ne = spans[k + 1]; spans[k + 1] = (s, ne)
                elif k == len(spans) - 1:
                    ps, pe = spans[k - 1]; spans[k - 1] = (ps, e)
                else:
                    if avg_smoothed(spans[k + 1]) >= avg_smoothed(spans[k - 1]):
                        ns, ne = spans[k + 1]; spans[k + 1] = (s, ne)
                    else:
                        ps, pe = spans[k - 1]; spans[k - 1] = (ps, e)
                del spans[k]
                changed = True
                break

    # --- convert sample-index spans to time ranges with an attached energy value ---
    segments = []
    for s, e in spans:
        start_t = float(times[s])
        end_t = float(times[e]) if e < n else track["duration_sec"]
        segments.append({
            "start": start_t,
            "end": end_t,
            "energy": round(float(energies[s:e].mean()), 3) if e > s else 0.0,
        })

    return segments


DENSITY_SMOOTH_SECS = 1.5  # same as the VSE add-on's density signal


def compute_density_curve(track: dict) -> tuple:
    """Port of the add-on's density signal: smooth the raw energy over
    DENSITY_SMOOTH_SECS FIRST, then normalise 0-1 — so the scale reflects
    sustained loudness rather than being set by a single transient peak.
    Returns (times, density)."""
    hires = track.get("hires")
    if hires:
        rate = float(hires["rate"])
        raw = np.asarray(hires["rms"], dtype=float)
    else:
        env = track["energy_envelope"]
        raw = np.asarray([e["energy"] for e in env], dtype=float)
        rate = 1.0 / max(env[1]["time"] - env[0]["time"], 1e-6) if len(env) > 1 else 2.0
    dens = _moving_average(raw, max(1, round(rate * DENSITY_SMOOTH_SECS / 2.0)))
    lo, hi = float(dens.min()), float(dens.max())
    dens = (dens - lo) / (hi - lo) if hi - lo > 1e-9 else np.full_like(dens, 0.5)
    return np.arange(len(dens)) / rate, dens


def apply_segment_intensity(segments: list[dict], track: dict, contrast: float) -> None:
    """Port of the add-on's block density (compute_block_density_values):
    average the density curve over each segment, then RE-NORMALISE across
    segments so the quietest segment is 0 and the busiest is 1 — the full
    clip-count range is always used, whatever the track's absolute level.
    `contrast` is the add-on's Density Power (>1 spreads values apart,
    <1 flattens them). Writes seg["intensity"] in place."""
    times, dens = compute_density_curve(track)
    means = []
    for seg in segments:
        mask = (times >= seg["start"]) & (times < seg["end"])
        means.append(float(dens[mask].mean()) if mask.any() else 0.0)
    lo, hi = min(means), max(means)
    for seg, m in zip(segments, means):
        norm = (m - lo) / (hi - lo) if hi - lo > 1e-9 else 0.5
        seg["intensity"] = round(norm ** max(0.05, contrast), 3)


def clips_for_intensity(intensity: float, min_clips: int, max_clips: int,
                        ramp_start: float, ramp_end: float, vary: bool, rng) -> int:
    """Map a segment's 0-1 intensity to a clip count.

    The ramp range rescales intensity first (at/below ramp_start = 0, at/above
    ramp_end = 1). Then, following the add-on's _overlap_count_from_intensity:
    intensity sets both a ceiling (min + i*span) and a floor that only starts
    rising above the midpoint (min + max(0, (i-0.5)*2)*span).
      vary off -> always the ceiling (the add-on's FIXED mode, smooth & predictable)
      vary on  -> a seeded pick between floor and ceiling (the add-on's RANDOM mode)
    Either way intensity 0 guarantees min_clips and 1 guarantees max_clips."""
    if max_clips <= min_clips:
        return min_clips
    if ramp_end <= ramp_start:
        pos = 1.0 if intensity >= ramp_start else 0.0
    else:
        pos = min(1.0, max(0.0, (intensity - ramp_start) / (ramp_end - ramp_start)))
    span = max_clips - min_clips
    ceiling = int(round(min_clips + pos * span))
    if not vary:
        return ceiling
    floor = min(ceiling, int(round(min_clips + max(0.0, (pos - 0.5) * 2) * span)))
    return rng.randint(floor, ceiling)


MAX_CHAIN_LINKS = 20  # safety cap against pathological tiny-leftover loops


def fill_slot(seg_energy: float, needed_sec: float, queues: dict, sequential: bool,
              excluded_here: set, global_excluded: set, avoid_video_ids: set,
              min_clip_len_sec: float = 1.0, role_filter: str = None,
              weighting: dict = None) -> tuple:
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

    role_filter: "intro_candidate" or "outro_candidate" when this segment is
    a marked intro/outro block — narrows the candidate pool to scenes
    carrying that flag, but ONLY when at least one such scene is actually
    available at this point in the chain; otherwise every candidate is
    considered as usual. Checked fresh on every link of the chain (not just
    the first), so a role-tagged clip that runs out partway through still
    lets the rest of the chain fall back to normal matching rather than
    failing to fill the slot.

    weighting: {"seg_idx", "segments", "confirmed", "prev_block_videos",
    "weight_shape", "weight_random", "weight_repeat_penalty", "weight_spread",
    "weight_motion", "seed"} — when given AND this isn't a role_filter block,
    the pick is the top result of rank_candidates_weighted instead of the
    plain closest-motion-match. None (or a role_filter block) keeps the
    original simple behaviour — this is the "intro/outro blocks aren't
    included" rule: those stay on the plain motion match regardless of
    whether weighting is otherwise in effect."""
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

        if role_filter:
            role_matches = [s for s in pool if s.get(role_filter)]
            if role_matches:
                pool = role_matches

        preferred = [s for s in pool if s["video_id"] not in avoid_video_ids
                     and (s["video_id"], s["scene_id"]) != last_used]
        candidates = preferred or [s for s in pool if (s["video_id"], s["scene_id"]) != last_used] or pool

        if weighting and not role_filter:
            ranked = rank_candidates_weighted(
                candidates, weighting["seg_idx"], weighting["segments"], weighting["confirmed"],
                queues, weighting["prev_block_videos"], weighting["weight_shape"],
                weighting["weight_random"], weighting["weight_repeat_penalty"],
                weighting["weight_spread"], weighting["weight_motion"], weighting["seed"],
            )
            best_span = ranked[0]
        else:
            best_span = min(candidates, key=lambda s: abs(s["motion_norm"] - seg_energy))
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
    weight_repeat_penalty: float = 0.0,
    weight_spread: float = 0.0,
    weight_motion: float = 0.0,
    autofill_seed: int = 42,
) -> tuple:
    """Each slot in a segment is filled to its FULL duration with a chain of
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

    for i, seg in enumerate(segments):
        seg_duration = seg["end"] - seg["start"]
        target = seg.get("intensity", seg["energy"])
        base_n_picks = clips_for_intensity(target, min_clips, max_clips, ramp_start, ramp_end, vary_count, rng)
        n_picks = max(0, base_n_picks - segment_clip_reduction.get(i, 0))
        excluded_here = segment_exclusions.get(i, set())
        role_filter = "intro_candidate" if i in intro_blocks else ("outro_candidate" if i in outro_blocks else None)
        prev_block_videos = {p["video_id"] for p in confirmed_so_far.get(i - 1, [])} if i > 0 else set()
        weighting = {
            "seg_idx": i, "segments": segments, "confirmed": confirmed_so_far,
            "prev_block_videos": prev_block_videos,
            "weight_shape": weight_shape, "weight_random": weight_random,
            "weight_repeat_penalty": weight_repeat_penalty, "weight_spread": weight_spread,
            "weight_motion": weight_motion, "seed": autofill_seed,
        }

        slots = []
        used_videos = set()
        for _ in range(n_picks):
            avoid = used_videos if not allow_same_video else set()
            chain, unfilled = fill_slot(target, seg_duration, queues, sequential,
                                         excluded_here, global_excluded, avoid, min_clip_len_sec,
                                         role_filter, weighting)
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

        timeline.append({
            "segment_index": i,
            "track_time": [round(seg["start"], 2), round(seg["end"], 2)],
            "segment_energy": seg["energy"],
            "segment_intensity": target,
            "clip_count": len(slots),
            "split_screen": len(slots) > 1,
            "manually_edited": i in segment_exclusions or i in segment_clip_reduction,
            "scenes": slots,
        })

    return timeline, clip_count_shortfall, duration_shortfall_sec


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

tracks = list_tracks()
if not tracks:
    st.error(f"No audio catalogues found in {AUDIO_DIR}")
    st.stop()

st.session_state.setdefault("video_time_ranges", {})
st.session_state.setdefault("show_range_picker", {})
st.session_state.setdefault("segment_exclusions", {})
st.session_state.setdefault("segment_clip_reduction", {})
st.session_state.setdefault("global_excluded_scenes", set())

catalogues = load_all_catalogues()
all_video_ids = sorted(catalogues.keys())
all_tag_options = sorted({t for cat in catalogues.values() for s in cat["scenes"] for t in s.get("tags", [])})
total_excluded = sum(1 for cat in catalogues.values() for s in cat["scenes"] if s.get("excluded"))
if total_excluded:
    st.caption(f"{total_excluded} scene(s) excluded from planning (e.g. intros) and ignored throughout.")

# (The tag filter, Select all recommended / Clear buttons, and video list used to render here —
#  they now live in render_video_selection_section, a fragment, called once the track below is
#  loaded. See that function and resolve_video_selection for why.)

# ---------------------------------------------------------------------------
# Dial defaults + Auto-detect
# Every tunable dial is keyed in session_state, so Auto-detect can set them
# all at once (via a button callback, which runs before the widgets redraw).
# ---------------------------------------------------------------------------

MAX_SIMULTANEOUS_CLIPS = 8   # matches the VSE add-on's layouts (up to SEVEN_UP / 4x2 grid)

DIAL_DEFAULTS = {
    # Segmentation — Adaptive
    "segmentation_method": "Adaptive (energy-change + hits)",
    "beats_per_bar": 4,
    "change_window_secs": 2.0,
    "change_threshold": 0.12,
    "min_segment_sec": 2.0,
    "max_segment_sec": 8.0,
    "react_to_hits": True,
    "hit_threshold": 0.9,
    "snap_secs": 0.3,
    "use_bar_snapping": True,
    # Segmentation — Rhythm Engine
    "hyper_delta_thresh": 0.35,
    "fast_energy_thresh": 0.75,
    "hyper_cooldown_bars": 2,
    "phrase_lock_bars": 4,
    "cinematic_bars": 4,
    "dynamic_priority_override": False,
    "override_delta_thresh": 0.15,
    # Split screen & intensity
    "split_screen_enabled": False,
    "min_clips": 1,
    "max_clips": 4,
    "ramp_range": (0.2, 0.85),
    "density_contrast": 1.0,
    "vary_count": False,
    "count_seed": 0,
    "min_clip_len_sec": 1.0,
    # Choreography mode
    "max_chor_videos": 4,
    "max_shape_matches": 6,
    "allow_same_video": False,
    "prefer_intro_outro": False,
    # Auto-fill weights (used in Matching & Export, stored here for persistence)
    "weight_shape": 2.0,
    "weight_random": 0.5,
    "weight_repeat_penalty": 1.5,
    "weight_spread": 1.0,
    "weight_motion": 1.0,
    "autofill_seed": 42,
}

# Weight keys that live on the Matching & Export page — Streamlit can purge
# these when the page reruns while that section isn't active (the widgets
# don't render, so their session_state entries can disappear). We shadow them
# exactly like audio_settings_shadow so they survive round-trips to other
# sections and come back at the user's last-set values rather than defaults.
WEIGHT_KEYS = [
    "weight_shape", "weight_random", "weight_repeat_penalty",
    "weight_spread", "weight_motion", "autofill_seed",
]

# Restore audio-settings keys from their shadow copy BEFORE the generic setdefault below
# fills any gap with a static default. This matters because render_audio_settings_section's
# own value=/index= restoration only runs while that section is the active one — by then it
# would be too late, since the setdefault loop runs unconditionally, every rerun, regardless
# of section, and setdefault only acts when the key is ABSENT. If a purged key were left
# absent until here, this loop would correctly see the gap and fill it from the shadow; but
# running it only AFTER the generic setdefault had already filled that same gap with the
# static default would mean the key is no longer absent, so the shadow value would never be
# reached — hence doing this restoration first.
# Audio settings shadow — kept for backward compatibility with saved sessions
_audio_shadow = st.session_state.get("audio_settings_shadow", {})
_audio_shadow.update({k: st.session_state[k] for k in AUDIO_SETTINGS_KEYS if k in st.session_state})
st.session_state["audio_settings_shadow"] = _audio_shadow

# Weight shadow
_weight_shadow = st.session_state.get("weight_settings_shadow", {})
_weight_shadow.update({k: st.session_state[k] for k in WEIGHT_KEYS if k in st.session_state})
st.session_state["weight_settings_shadow"] = _weight_shadow

for _k, _v in DIAL_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


# ---------------------------------------------------------------------------
# Project save / load — captures every setting plus matching progress (for
# all three matching modes at once, regardless of which is currently active)
# into a single JSON file, so a session can be closed and resumed later.
# ---------------------------------------------------------------------------

PROJECT_FILE_VERSION = 1

# Plain scalar / list-of-strings keys — copied straight across, no conversion needed.
PROJECT_SIMPLE_KEYS = [
    "track_id", "matching_mode", "max_shape_matches", "max_chor_videos", "allow_same_video",
    "sequential_mode", "prefer_intro_outro",
    *AUDIO_SETTINGS_KEYS,
    "split_screen_enabled", "min_clips", "max_clips", "density_contrast",
    "vary_count", "count_seed", "min_clip_len_sec",
    "weight_shape", "weight_random", "weight_repeat_penalty", "weight_spread", "weight_motion", "autofill_seed",
    "committed_selected_videos", "committed_tag_filter", "known_extra_videos", "extra_videos",
    "adv_current_block", "chor_current_block",
    "intro_block_indices", "outro_block_indices",
]


def _build_project_save_dict() -> dict:
    """Gathers every setting and all matching progress (Auto/Manual/Choreography
    alike, whichever is or isn't currently active) into one JSON-safe dict."""
    data = {"_version": PROJECT_FILE_VERSION, "_saved_at": datetime.datetime.now().isoformat(timespec="seconds")}

    for k in PROJECT_SIMPLE_KEYS:
        if k in st.session_state:
            data[k] = st.session_state[k]

    # Tuple -> list
    if "ramp_range" in st.session_state:
        data["ramp_range"] = list(st.session_state["ramp_range"])

    # {video_id: (start, end)} -> {video_id: [start, end]}
    data["video_time_ranges"] = {
        vid: list(rng) for vid, rng in st.session_state.get("video_time_ranges", {}).items()
    }

    # {int: [pick dict, ...]} -> {str(int): [...]} (pick dicts are already JSON-safe)
    data["adv_confirmed"] = {str(k): v for k, v in st.session_state.get("adv_confirmed", {}).items()}
    data["chor_confirmed"] = {str(k): v for k, v in st.session_state.get("chor_confirmed", {}).items()}

    # {int: [{"video_id":.., "scene_id":..}, ...]} -> {str(int): [...]}
    data["adv_skips"] = {str(k): v for k, v in st.session_state.get("adv_skips", {}).items()}

    # {int: {video_id: (scene_id, offset_sec)}} -> {str(int): {video_id: [scene_id, offset_sec]}}
    data["chor_overrides"] = {
        str(seg_idx): {vid: list(pair) for vid, pair in overrides.items()}
        for seg_idx, overrides in st.session_state.get("chor_overrides", {}).items()
    }

    # {int: set((video_id, scene_id))} -> {str(int): [[video_id, scene_id], ...]}
    data["segment_exclusions"] = {
        str(seg_idx): [list(pair) for pair in excl]
        for seg_idx, excl in st.session_state.get("segment_exclusions", {}).items()
    }

    # {int: int} -> {str(int): int}
    data["segment_clip_reduction"] = {
        str(k): v for k, v in st.session_state.get("segment_clip_reduction", {}).items()
    }

    # set((video_id, scene_id)) -> [[video_id, scene_id], ...]
    data["global_excluded_scenes"] = [list(pair) for pair in st.session_state.get("global_excluded_scenes", set())]

    # Individual tick checkboxes for both matching grids — plain bools, scanned by prefix.
    # Restoring these (rather than only the confirmed/override dicts) preserves ticks made
    # for the CURRENT, not-yet-confirmed block too, not just already-confirmed ones.
    data["adv_pick_state"] = {
        k: v for k, v in st.session_state.items() if k.startswith("adv_pick_") and isinstance(v, bool)
    }
    data["chor_pick_state"] = {
        k: v for k, v in st.session_state.items() if k.startswith("chor_pick_") and isinstance(v, bool)
    }

    return data


def _apply_project_load_dict(data: dict) -> list:
    """Writes a loaded project dict back into session_state, converting JSON-safe
    shapes back to their Python originals. Returns a list of warning strings for
    anything that couldn't be applied cleanly (never raises)."""
    warnings = []

    for k in PROJECT_SIMPLE_KEYS:
        if k in data:
            st.session_state[k] = data[k]

    if "ramp_range" in data:
        try:
            st.session_state["ramp_range"] = tuple(data["ramp_range"])
        except (TypeError, ValueError):
            warnings.append("Couldn't restore the intensity ramp range — left at its current value.")

    try:
        st.session_state["video_time_ranges"] = {
            vid: tuple(rng) for vid, rng in data.get("video_time_ranges", {}).items()
        }
    except (TypeError, ValueError):
        warnings.append("Couldn't restore video time ranges — left at their current values.")

    try:
        st.session_state["adv_confirmed"] = {int(k): v for k, v in data.get("adv_confirmed", {}).items()}
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Manual step-through's confirmed blocks.")
        st.session_state["adv_confirmed"] = {}

    try:
        st.session_state["chor_confirmed"] = {int(k): v for k, v in data.get("chor_confirmed", {}).items()}
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Choreography's confirmed blocks.")
        st.session_state["chor_confirmed"] = {}

    try:
        st.session_state["adv_skips"] = {int(k): v for k, v in data.get("adv_skips", {}).items()}
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Manual step-through's skip history.")
        st.session_state["adv_skips"] = {}

    try:
        st.session_state["chor_overrides"] = {
            int(seg_idx): {vid: tuple(pair) for vid, pair in overrides.items()}
            for seg_idx, overrides in data.get("chor_overrides", {}).items()
        }
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Choreography's clip overrides.")
        st.session_state["chor_overrides"] = {}

    try:
        st.session_state["segment_exclusions"] = {
            int(seg_idx): {tuple(pair) for pair in excl}
            for seg_idx, excl in data.get("segment_exclusions", {}).items()
        }
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Auto mode's per-segment Swap/Reject exclusions.")
        st.session_state["segment_exclusions"] = {}

    try:
        st.session_state["segment_clip_reduction"] = {
            int(k): v for k, v in data.get("segment_clip_reduction", {}).items()
        }
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Auto mode's per-segment Remove counts.")
        st.session_state["segment_clip_reduction"] = {}

    try:
        st.session_state["global_excluded_scenes"] = {tuple(pair) for pair in data.get("global_excluded_scenes", [])}
    except (TypeError, ValueError):
        warnings.append("Couldn't restore globally rejected clips.")
        st.session_state["global_excluded_scenes"] = set()

    for k, v in data.get("adv_pick_state", {}).items():
        st.session_state[k] = v
    for k, v in data.get("chor_pick_state", {}).items():
        st.session_state[k] = v

    # Re-seed the audio-settings shadow too, so render_audio_settings_section's own
    # restoration (which reads from the shadow, not these keys directly) stays in sync
    # the next time that section is visited.
    st.session_state["audio_settings_shadow"] = {
        k: st.session_state[k] for k in AUDIO_SETTINGS_KEYS if k in st.session_state
    }

    # One-shot, per-mode suppression so the Auto/Manual/Choreography "settings changed,
    # reset progress" guards don't immediately wipe out what was just loaded the first
    # time each mode happens to be visited after this. Each mode consumes its own entry
    # from this set the first time its own guard runs, then behaves normally again —
    # so a later, genuine settings change still resets progress as expected.
    st.session_state["_project_load_pending_modes"] = {"Auto", "Manual step-through", "Choreography"}

    return warnings


def save_project(name: str) -> tuple[Path, str | None]:
    """Writes the current session's full settings + progress to
    PROJECTS_DIR/<sanitized name>.json and uploads to Drive if authenticated.
    Returns (path, drive_error_or_None)."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROJECTS_DIR / f"{sanitize_filename(name)}.json"
    out_path.write_text(json.dumps(_build_project_save_dict(), indent=2))
    drive_err = None
    if _OAUTH_AVAILABLE and st.session_state.get(_OAUTH_SESSION_KEY):
        drive_err = push_file_with_oauth(
            st.session_state[_OAUTH_SESSION_KEY], out_path, "scene-labeling/projects"
        )
    return out_path, drive_err


def load_project(path: Path) -> list:
    """Reads a project file and applies it to session_state. Returns a list
    of warning strings (empty if everything restored cleanly)."""
    data = json.loads(path.read_text())
    return _apply_project_load_dict(data)


def list_saved_projects() -> list:
    if not PROJECTS_DIR.exists():
        return []
    return sorted((p.stem for p in PROJECTS_DIR.glob("*.json")), key=str.lower)


def _consume_load_suppression(mode_name: str) -> bool:
    """True if this mode's progress-reset guard should be suppressed because a
    project was just loaded and this mode hasn't had a chance to "see" the
    restored state yet. Consumes (removes) this mode's entry so the suppression
    only applies once — a later, genuine settings change still resets that
    mode's progress normally, same as before any project was ever loaded."""
    pending = st.session_state.get("_project_load_pending_modes")
    if pending and mode_name in pending:
        pending.discard(mode_name)
        st.session_state["_project_load_pending_modes"] = pending
        return True
    return False


def _weighted_percentile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights, dtype=float)[order]
    cum = np.cumsum(w) / max(w.sum(), 1e-9)
    return float(v[min(len(v) - 1, int(np.searchsorted(cum, q)))])


def auto_tune_parameters(track: dict, min_clips: int, max_clips: int) -> tuple:
    """Pick segmentation + split-screen dials from the track's own tempo,
    energy and onset signals. Returns (settings dict, list of explanation lines)."""
    bpm = float(track.get("bpm") or 120.0)
    while bpm > 190:
        bpm /= 2        # librosa sometimes reports double-time
    while bpm < 70:
        bpm *= 2        # ... or half-time
    beat = 60.0 / bpm
    bar = 4 * beat
    duration = float(track["duration_sec"])
    why = [f"Tempo {bpm:.0f} BPM → 1 beat = {beat:.2f}s, 1 bar = {bar:.2f}s"]
    s = {}

    # Timing grid: one bar is the natural minimum phrase length; snap by half a beat.
    s["min_segment_sec"] = round(min(4.0, max(1.0, bar)), 1)
    s["change_window_secs"] = round(min(4.0, max(1.0, bar)), 1)
    s["snap_secs"] = round(min(0.5, max(0.1, beat / 2)), 2)
    s["max_segment_sec"] = round(min(16.0, max(4.0, 4 * bar)), 1)
    s["react_to_hits"] = True
    why.append(f"Min segment & energy-change window = 1 bar ({s['min_segment_sec']}s); "
               f"max segment = 4 bars ({s['max_segment_sec']}s); snap radius = ½ beat ({s['snap_secs']}s)")

    hires = track.get("hires")
    if not hires:
        why.append("⚠️ No 30 Hz signals for this track — thresholds left unchanged. "
                   "Re-run audio_pipeline.py in Colab for full auto-detect.")
        return s, why

    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    onset = np.asarray(hires["onset"], dtype=float)
    n = min(len(rms), len(onset))
    rms, onset = rms[:n], onset[:n]
    min_len = max(1, round(rate * s["min_segment_sec"]))

    # Energy-change threshold: aim for roughly one structural change per 8 bars
    # (a typical phrase), but never below 3x the typical ripple (noise floor).
    w = max(1, round(rate * s["change_window_secs"]))
    c = np.concatenate([[0.0], np.cumsum(rms)])
    idx = np.arange(n)
    b_lo, a_hi = np.maximum(0, idx - w), np.minimum(n, idx + w)
    change = np.abs((c[a_hi] - c[idx]) / np.maximum(1, a_hi - idx) - (c[idx] - c[b_lo]) / np.maximum(1, idx - b_lo))
    cp = np.flatnonzero((change[1:-1] >= change[:-2]) & (change[1:-1] > change[2:])) + 1
    target_changes = max(2, int(duration / (8 * bar)))
    noise_floor = 3.0 * float(np.median(change))
    peak_vals = np.sort(change[cp])[::-1] if len(cp) else np.array([0.12])
    thr = peak_vals[min(len(peak_vals) - 1, target_changes - 1)]
    s["change_threshold"] = round(float(np.clip(max(thr, noise_floor), 0.02, 0.5)), 2)
    why.append(f"Energy-change threshold {s['change_threshold']} → aims for ~{target_changes} "
               f"structural changes (1 per 8 bars)")

    # Big-hit threshold: aim for an average of one cut per 2 bars overall, taking the
    # strongest hits first with the same minimum spacing the segmenter uses.
    op = np.flatnonzero((onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:])) + 1
    accepted = []
    for pos in sorted(op, key=lambda i: -onset[i]):
        if pos >= min_len and n - pos >= min_len and all(abs(pos - a) >= min_len for a in accepted):
            accepted.append(pos)
    target_hits = max(1, int(duration / (2 * bar)) - target_changes)
    if accepted:
        hit_thr = onset[accepted[min(len(accepted) - 1, target_hits - 1)]]
        typical = float(np.percentile(onset[op], 75)) if len(op) else 0.5
        s["hit_threshold"] = round(float(np.clip(max(hit_thr, typical), 0.5, 0.99)), 2)
    why.append(f"Big-hit threshold {s.get('hit_threshold', st.session_state['hit_threshold'])} → aims for "
               f"~{target_hits} hit cuts (≈1 cut per 2 bars overall), never below the typical beat level")

    # Split screen: run the segmenter with these settings, then shape intensity.
    segs = build_track_segments(track, s["change_window_secs"], s["change_threshold"], s["min_segment_sec"],
                                True, s.get("hit_threshold", 0.9), s["snap_secs"], s["max_segment_sec"])
    apply_segment_intensity(segs, track, 1.0)
    base = np.array([sg["intensity"] for sg in segs])
    lengths = np.array([sg["end"] - sg["start"] for sg in segs])
    med = _weighted_percentile(base, lengths, 0.5)
    contrast = np.log(0.5) / np.log(med) if 0.05 < med < 0.95 else 1.0
    s["density_contrast"] = round(float(np.clip(contrast, 0.5, 2.5)), 2)
    shaped = base ** s["density_contrast"]
    lo = _weighted_percentile(shaped, lengths, 0.2)
    hi = _weighted_percentile(shaped, lengths, 0.9)
    if hi - lo < 0.2:
        lo, hi = max(0.0, (lo + hi) / 2 - 0.1), min(1.0, (lo + hi) / 2 + 0.1)
    s["ramp_range"] = (round(lo * 20) / 20, round(max(hi, lo + 0.1) * 20) / 20)
    s["split_screen_enabled"] = True
    s["vary_count"] = False
    why.append(f"Density contrast {s['density_contrast']} → the typical (median) moment sits at mid intensity")
    why.append(f"Intensity range {s['ramp_range'][0]:.2f}–{s['ramp_range'][1]:.2f} → quietest 20% of the track "
               f"gets {min_clips} clip(s), loudest 10% gets {max_clips}")
    why.append(f"Result: {len(segs)} segments")
    return s, why


def _run_auto_tune():
    t = load_track(st.session_state["track_id"])
    mn = int(st.session_state.get("min_clips", 1))
    mx = max(int(st.session_state.get("max_clips", 4)), mn + 1, 2)
    settings, why = auto_tune_parameters(t, mn, mx)
    for k, v in settings.items():
        st.session_state[k] = v
    st.session_state["max_clips"] = min(MAX_SIMULTANEOUS_CLIPS, mx)
    st.session_state["auto_tune_report"] = why


st.sidebar.header("Project")
with st.sidebar.expander("💾 Save / 📂 Load", expanded=False):
    st.caption("Saves every setting plus matching progress (Auto, Manual step-through, and "
               "Choreography all at once) so you can close this and pick up again later.")
    _save_name = st.text_input("Project name", key="project_save_name", placeholder="my-compilation")
    if st.button("💾 Save Project", disabled=not _save_name.strip(), use_container_width=True):
        _out_path, _drive_err = save_project(_save_name.strip())
        if _drive_err:
            st.warning(f"Saved locally but Drive upload failed: {_drive_err}")
        elif _DRIVE_SYNC_AVAILABLE:
            st.success(f"Saved and uploaded to Drive: {_out_path.name}")
        else:
            st.success(f"Saved as {_out_path.name}")

    st.divider()
    _existing_projects = list_saved_projects()
    if _existing_projects:
        _load_choice = st.selectbox("Load project", ["—"] + _existing_projects, key="project_load_choice")
        if _load_choice != "—" and st.button("📂 Load Project", use_container_width=True):
            _load_warnings = load_project(PROJECTS_DIR / f"{_load_choice}.json")
            for _w in _load_warnings:
                st.warning(_w)
            st.success(f"Loaded '{_load_choice}'.")
            st.rerun()
    else:
        st.caption("No saved projects yet.")

# A loaded project's track_id can reference a track that no longer exists (renamed/deleted
# audio file) — the Track selectbox below would error if its backing value isn't among the
# current options, so drop it here and let the widget fall back to its own first option.
if st.session_state.get("track_id") not in tracks:
    st.session_state.pop("track_id", None)
    if st.session_state.get("_project_load_pending_modes"):
        st.sidebar.warning("The loaded project's track is no longer available — defaulted to the first track.")


if _DRIVE_SYNC_AVAILABLE:
    with st.sidebar.expander("☁️ Sync from Google Drive", expanded=False):
        st.caption("Pull latest catalogues, audio, thumbnails and sprites from Drive to the server. "
                   "Run after processing new videos or audio in Colab.")
        if st.button("🔄 Sync now", key="drive_sync_btn", use_container_width=True):
            sync_progress = st.progress(0.0, text="Starting sync…")
            sync_errors = []

            def _sync_progress(idx, total, name):
                sync_progress.progress(
                    min(0.99, idx / max(total, 1)),
                    text=f"Syncing {name}…"
                )

            result = sync_pull(CATALOGUE_DIR, AUDIO_DIR, _sync_progress)
            sync_progress.progress(1.0, text="Done.")
            if result["errors"]:
                for err in result["errors"][:5]:
                    st.error(err)
                st.warning("Some files failed — check Drive sharing if errors persist.")
            else:
                st.success(f"Synced {result['synced']} file(s), "
                           f"skipped {result['skipped']} unchanged.")
                st.cache_data.clear()
else:
    with st.sidebar.expander("☁️ Drive sync unavailable", expanded=False):
        st.caption("Service account credentials not found at /root/drive-credentials.json")

if _OAUTH_AVAILABLE:
    with st.sidebar.expander("📤 Save to Google Drive", expanded=False):
        if st.session_state.get(_OAUTH_SESSION_KEY):
            st.success("Connected — saves will upload to Drive.")
            if st.button("Disconnect Drive", key="drive_disconnect"):
                st.session_state.pop(_OAUTH_SESSION_KEY, None)
                st.rerun()
        else:
            st.caption("Connect to save plans, projects and previews to your Google Drive.")
            _auth_url = get_auth_url(state="pages/1_Compilation_Planner.py")
            st.markdown(f'<a href="{_auth_url}" target="_self">🔗 Connect Google Drive</a>', unsafe_allow_html=True)

st.sidebar.header("Track")
track_id = st.sidebar.selectbox("Track", tracks, key="track_id")

st.sidebar.button("🎯 Auto-detect settings for this track", on_click=_run_auto_tune, type="primary",
                  help="Analyses this track's tempo, energy and hits, then sets the segmentation and "
                       "split-screen dials below. Adjust any dial afterwards to fine-tune.")
if st.session_state.get("auto_tune_report"):
    with st.sidebar.expander("What auto-detect chose, and why"):
        for line in st.session_state["auto_tune_report"]:
            st.write("• " + line)

st.sidebar.header("Matching mode")
matching_mode = st.sidebar.radio(
    "How clips are chosen", ["Auto", "Manual step-through", "Choreography"], key="matching_mode",
    help="Auto: the algorithm picks the best-matching clip per segment automatically.  "
         "Manual step-through: pick from the best shape-matched candidates one block at a time.  "
         "Choreography: one column per source video — see the auto pick for each, then override with "
         "any available clip from that video via a timecode dropdown.",
)
max_shape_matches = st.sidebar.number_input(
    "Max candidates shown per block", min_value=1, max_value=30, key="max_shape_matches",
    disabled=(matching_mode != "Manual step-through"),
    help="How many of the best-scoring clips are offered per block in Manual step-through mode.",
)
max_chor_videos = st.sidebar.number_input(
    "Max videos shown per block (Choreography)", min_value=2, max_value=6, key="max_chor_videos",
    disabled=(matching_mode != "Choreography"),
    help="How many source-video columns to show per block. Keep it small enough that the grid "
         "stays readable — 4 is comfortable, 6 is the practical limit.",
)
prefer_intro_outro = st.sidebar.checkbox(
    "Prefer Intro/Outro-tagged clips for marked blocks",
    key="prefer_intro_outro",
    help="Mark scenes as intro/outro candidates in the review app (app.py) — 'Intro candidate' / "
         "'Outro candidate' checkboxes alongside Exclude. Works in all three matching modes. With "
         "this on, an 'Intro/Outro blocks' picker appears on the Matching & Export page (default: "
         "first block / last block, freely changeable) — those blocks draw from a video's tagged "
         "clips, but only for videos that actually have any tagged that way; a video with none "
         "shows its normal full availability there, same as any other block.",
)

st.sidebar.header("Video selection")
sequential_mode = st.sidebar.checkbox(
    "Keep each video's own clips in original sequential order",
    value=False, key="sequential_mode",
    help="When on, a video's scenes are only ever used in their original chronological order "
         "(matches the VSE add-on's 'Keep Sequential Order' behaviour) — other videos can still "
         "interleave freely around them.",
)

if st.session_state["video_time_ranges"]:
    st.sidebar.caption(f"{len(st.session_state['video_time_ranges'])} video(s) restricted to a custom time range.")
    if st.sidebar.button("Reset all video ranges"):
        st.session_state["video_time_ranges"] = {}
        st.rerun()

allow_same_video = st.sidebar.checkbox(
    "Allow multiple clips from the same video in one split-screen segment",
    key="allow_same_video",
)

st.sidebar.header("Split screen & intensity")
split_screen_enabled = st.sidebar.checkbox(
    "Enable split-screen for high-energy segments", key="split_screen_enabled",
)
st.sidebar.slider(
    "Min simultaneous clips", 1, MAX_SIMULTANEOUS_CLIPS, key="min_clips",
    disabled=not split_screen_enabled,
)
if st.session_state["max_clips"] < st.session_state["min_clips"]:
    st.session_state["max_clips"] = st.session_state["min_clips"]
st.sidebar.slider(
    "Max simultaneous clips",
    min_value=st.session_state["min_clips"], max_value=MAX_SIMULTANEOUS_CLIPS,
    key="max_clips", disabled=not split_screen_enabled,
    help="Up to 8, using the same layouts as the VSE add-on.",
)
st.sidebar.slider(
    "Intensity ramp range", 0.0, 1.0, step=0.05, key="ramp_range",
    disabled=not split_screen_enabled,
    help="At or below left value: min clips. At or above right: max clips.",
)
st.sidebar.slider(
    "Density contrast", 0.25, 3.0, step=0.05, key="density_contrast",
    disabled=not split_screen_enabled,
    help="Above 1 = only the loudest sections reach high clip counts; below 1 = rises more evenly.",
)
_vary = st.sidebar.checkbox(
    "Vary clip count within range (seeded)", key="vary_count",
    disabled=not split_screen_enabled,
)
st.sidebar.number_input(
    "Clip count seed", 0, 99999, key="count_seed",
    disabled=not (split_screen_enabled and _vary),
)
st.sidebar.slider(
    "Minimum clip length (s)", 0.2, 3.0, step=0.1, key="min_clip_len_sec",
    help="If footage left to fill a slot is shorter than this, extend the previous clip instead.",
)

# ---------------------------------------------------------------------------
# Compute: track + video-level energy matching
# ---------------------------------------------------------------------------

track = load_track(track_id)
st.sidebar.caption(f"🎵 Track length: {format_mmss(track['duration_sec'])}")

# ---------------------------------------------------------------------------
# App-scope audio settings preservation
# ---------------------------------------------------------------------------
# Streamlit @fragment has its own scoped session state — keys written inside
# a fragment during a fragment rerun are NOT guaranteed to survive into the
# next full app rerun. This means shadow dicts and plain keys set inside
# render_audio_settings_section (a @fragment) can be wiped between reruns.
#
# _audio_appscope is written here — in the main script body, outside any
# fragment — on every full rerun. If audio keys are currently present
# (fragment ran this rerun), capture them. If absent (wiped by Streamlit),
# restore from the last captured snapshot. This dict is never touched by
# any fragment and survives all rerun types.
_appscope = st.session_state.get("_audio_appscope", {})
_captured = {k: st.session_state[k] for k in AUDIO_SETTINGS_KEYS if k in st.session_state}
if _captured:
    _appscope.update(_captured)
    st.session_state["_audio_appscope"] = _appscope
for _k in AUDIO_SETTINGS_KEYS:
    if _k in _appscope:
        st.session_state[_k] = _appscope[_k]

# ---------------------------------------------------------------------------
# Three free-navigation sections (not a locked wizard — jump between them
# anytime). Each section body only runs when selected — using a plain radio
# selector plus explicit st.stop() calls rather than st.tabs(), which
# executes every tab's body regardless of which tab is visually active.
# Sections 0 and 1 are fragments: widget interactions inside them only
# rerun that section, not the whole page.
# ---------------------------------------------------------------------------

ui_section = st.radio(
    "Section",
    ["🎵 0. Audio Settings", "📼 1. Videos & Time Ranges", "🎬 2. Matching & Export"],
    horizontal=True, key="ui_section", label_visibility="collapsed",
)

if ui_section == "🎵 0. Audio Settings":
    render_audio_settings_section(track)
    st.stop()

if ui_section == "📼 1. Videos & Time Ranges":
    render_video_selection_section(track, all_video_ids, all_tag_options, catalogues)
    st.stop()

tag_filter = list(st.session_state.get("committed_tag_filter", []))
selected_videos = list(st.session_state.get("committed_selected_videos", []))

# Segmentation + split-screen values are set in the Audio Settings section
# and stored in session_state; read them here for the Matching computation.
segmentation_method = st.session_state.get("segmentation_method", "Adaptive (energy-change + hits)")
beats_per_bar = int(st.session_state.get("beats_per_bar", 4))

if segmentation_method == "Adaptive (energy-change + hits)":
    change_window_secs = st.session_state.get("change_window_secs", 2.0)
    change_threshold = st.session_state.get("change_threshold", 0.12)
    min_segment_sec = st.session_state.get("min_segment_sec", 2.0)
    max_segment_sec = st.session_state.get("max_segment_sec", 8.0)
    react_to_hits = st.session_state.get("react_to_hits", True)
    hit_threshold = st.session_state.get("hit_threshold", 0.9)
    snap_secs = st.session_state.get("snap_secs", 0.3)
    use_bar_snapping = st.session_state.get("use_bar_snapping", True)
    hyper_delta_thresh = fast_energy_thresh = 0.0
    hyper_cooldown_bars = phrase_lock_bars = cinematic_bars = 0
    dynamic_priority_override = False
    override_delta_thresh = 0.0
else:
    hyper_delta_thresh = st.session_state.get("hyper_delta_thresh", 0.35)
    fast_energy_thresh = st.session_state.get("fast_energy_thresh", 0.75)
    hyper_cooldown_bars = int(st.session_state.get("hyper_cooldown_bars", 2))
    phrase_lock_bars = int(st.session_state.get("phrase_lock_bars", 4))
    cinematic_bars = int(st.session_state.get("cinematic_bars", 4))
    dynamic_priority_override = st.session_state.get("dynamic_priority_override", False)
    override_delta_thresh = st.session_state.get("override_delta_thresh", 0.15)
    change_window_secs = change_threshold = min_segment_sec = max_segment_sec = 0.0
    use_bar_snapping = True
    react_to_hits = False
    hit_threshold = snap_secs = 0.0

# Split-screen values are set by the sidebar widgets (always live regardless of active section)
min_clips = st.session_state["min_clips"]
max_clips = st.session_state["max_clips"]
ramp_start, ramp_end = st.session_state["ramp_range"]
density_contrast = st.session_state["density_contrast"]
vary_count = st.session_state["vary_count"]
count_seed = st.session_state["count_seed"]
min_clip_len_sec = st.session_state["min_clip_len_sec"]
if not split_screen_enabled:
    min_clips = max_clips = 1
    ramp_start, ramp_end = 0.0, 1.0
    density_contrast, vary_count, count_seed = 1.0, False, 0

if not selected_videos:
    st.info("No videos selected yet. Switch to the **Videos & Time Ranges** tab above to tick some — the "
            "plan builds as soon as at least one is selected.")
    st.stop()

# ---------------------------------------------------------------------------
# Auto-fill weighting — shown inline on this page, not in the sidebar
# ---------------------------------------------------------------------------

_autofill_disabled = matching_mode not in ("Manual step-through", "Auto", "Choreography")
_is_auto_mode = matching_mode == "Auto"
_no_shape_score_mode = matching_mode in ("Auto", "Choreography")  # neither computes a per-frame shape score
with st.expander("⚙️ Auto-fill weighting", expanded=False):
    if _is_auto_mode:
        st.caption("In Auto mode these weights drive EVERY pick directly (there's no separate "
                   "manual grid to Auto-fill here). On a marked intro/outro block, candidates whose "
                   "clip is actually tagged are still preferred first; these weights rank among "
                   "whichever pool (tagged or, failing that, untagged) ends up available.")
    elif matching_mode == "Choreography":
        st.caption("Drives a column's '⭐ Auto pick' preview, Confirm & Next's Auto columns, and "
                   "Auto-fill ALL — ticking a clip yourself always overrides these. On a marked "
                   "intro/outro block, videos whose clip is actually tagged are still preferred over "
                   "ones that fell back to untagged footage; these weights rank within each group.")
    else:
        st.caption("Controls Auto-fill's automatic picks only — the candidate grid always stays sorted by "
                   "raw shape score for manual browsing, and these weights never restrict what you can pick yourself.")
    _wt_cols = st.columns(3)
    with _wt_cols[0]:
        st.slider("A — Shape match", 0.0, 5.0, step=0.1, key="weight_shape",
                  disabled=(_autofill_disabled or _no_shape_score_mode),
                  help="How much the clip's frame-by-frame shape-match score drives the pick. Has no "
                       "effect in Auto or Choreography mode — neither computes a per-frame curve "
                       "comparison for this weight to act on." if _no_shape_score_mode else
                       "How much the clip's frame-by-frame shape-match score drives the pick.")
        st.slider("B — Randomness", 0.0, 5.0, step=0.1, key="weight_random",
                  disabled=_autofill_disabled,
                  help="Adds variety. Reproducible — same seed always gives same picks.")
    with _wt_cols[1]:
        st.slider("C — Avoid repeating previous block", 0.0, 5.0, step=0.1,
                  key="weight_repeat_penalty", disabled=_autofill_disabled,
                  help="Penalizes reusing a video from the immediately preceding block. "
                       "Also your control for 'too many rapid changes' — raise to settle.")
        st.slider("D — Spread usage across videos", 0.0, 5.0, step=0.1, key="weight_spread",
                  disabled=_autofill_disabled,
                  help="Favors under-used videos and those with more footage remaining. In "
                       "sequential mode, also controls how far an automatic pick may roam past "
                       "a video's frontmost available clip — 0 stays pinned to the front, higher "
                       "allows skipping ahead when that video has spare footage relative to how "
                       "many blocks are left (never enough to risk running out later).")
    with _wt_cols[2]:
        st.slider("E — Motion intensity match", 0.0, 5.0, step=0.1, key="weight_motion",
                  disabled=_autofill_disabled,
                  help="Prefers clips whose overall motion level matches this block's intensity. "
                       "Complementary to A (shape pattern) — A rewards the same rises/falls; "
                       "E rewards the right activity level.")
        st.number_input("Random seed", 0, 99999, key="autofill_seed",
                        disabled=_autofill_disabled,
                        help="Change to reroll randomness (B) without touching anything else.")

# Save weight values to their shadow copy every time this section renders —
# mirrors audio_settings_shadow so values survive switching to another section.
st.session_state["weight_settings_shadow"] = {
    k: st.session_state[k] for k in WEIGHT_KEYS if k in st.session_state
}

# Always read weight values from session_state (widgets above have already written them)
weight_shape = st.session_state.get("weight_shape", 2.0)
weight_random = st.session_state.get("weight_random", 0.5)
weight_repeat_penalty = st.session_state.get("weight_repeat_penalty", 1.5)
weight_spread = st.session_state.get("weight_spread", 1.0)
weight_motion = st.session_state.get("weight_motion", 1.0)
autofill_seed = st.session_state.get("autofill_seed", 42)

# ---------------------------------------------------------------------------
# Per-scene matching against the beat timeline
# ---------------------------------------------------------------------------

# Manual overrides (reject / swap) are keyed by segment index, which only stays
# meaningful for a given track + segment length — reset them if either changes.
state_key = (track_id, segmentation_method, change_window_secs, change_threshold, min_segment_sec,
             max_segment_sec, react_to_hits, hit_threshold, snap_secs, use_bar_snapping, beats_per_bar,
             hyper_delta_thresh, fast_energy_thresh, hyper_cooldown_bars, phrase_lock_bars, cinematic_bars,
             dynamic_priority_override, override_delta_thresh)
if st.session_state.get("planner_state_key") != state_key:
    st.session_state["planner_state_key"] = state_key
    if not _consume_load_suppression("Auto"):
        st.session_state["segment_exclusions"] = {}
        st.session_state["segment_clip_reduction"] = {}
        st.session_state["global_excluded_scenes"] = set()

if st.sidebar.button("Reset manual clip overrides"):
    st.session_state["segment_exclusions"] = {}
    st.session_state["segment_clip_reduction"] = {}
    st.session_state["global_excluded_scenes"] = set()
    st.rerun()

if st.session_state["global_excluded_scenes"]:
    st.sidebar.caption(f"{len(st.session_state['global_excluded_scenes'])} clip(s) permanently rejected from this plan.")

queues = build_footage_queues(tuple(selected_videos), tuple(tag_filter), st.session_state["video_time_ranges"])
if not queues:
    st.warning("No candidate scenes available for the selected videos/tags.")
    st.stop()

total_footage_sec = sum(span["remaining_sec"] for spans in queues.values() for span in spans)

if segmentation_method == "Adaptive (energy-change + hits)":
    segments = build_track_segments(track, change_window_secs, change_threshold, min_segment_sec,
                                    react_to_hits, hit_threshold, snap_secs, max_segment_sec,
                                    beats_per_bar, use_bar_snapping)
else:
    segments = build_rhythm_engine_segments(track, beats_per_bar, hyper_delta_thresh, fast_energy_thresh,
                                            hyper_cooldown_bars, phrase_lock_bars, cinematic_bars,
                                            dynamic_priority_override, override_delta_thresh)
apply_segment_intensity(segments, track, density_contrast)
if not track.get("hires"):
    if segmentation_method == "Adaptive (energy-change + hits)":
        st.warning("This track was analysed before the 30 Hz onset signals existed, so cuts use the older, "
                   "looser method. Re-run audio_pipeline.py in Colab to upgrade it.")
    else:
        st.warning("Rhythm Engine needs the 30 Hz onset signals to compute bar energies, which this track "
                   "doesn't have — it's being treated as one single segment. Re-run audio_pipeline.py in "
                   "Colab to upgrade it, or switch to Adaptive segmentation for now.")
recommended_counts = [
    clips_for_intensity(sg["intensity"], min_clips, max_clips, ramp_start, ramp_end, False, None)
    for sg in segments
]

# ---------------------------------------------------------------------------
# Intro/Outro block picker — shared across all three matching modes. Defaults
# to the first block for intro and the last for outro, but every block is
# freely selectable, so you can add extra blocks alongside the defaults or
# replace them with different ones entirely.
#
# The underlying widgets live inside Matching & Export, so (same as every
# other widget confined to one of the three ui_sections) they'd lose their
# session_state the moment you visit Audio Settings or Videos & Time Ranges
# and come back — Streamlit discards state for widgets that didn't render on
# a given rerun. So the actual selections are kept in plain, non-widget keys
# (intro_block_indices / outro_block_indices) and explicitly fed back in as
# each multiselect's default= on every call, the same fix used for Audio
# Settings' dials.
# ---------------------------------------------------------------------------
intro_blocks_0based: set = set()
outro_blocks_0based: set = set()
if st.session_state.get("prefer_intro_outro"):
    with st.expander("🎬 Intro/Outro blocks", expanded=False):
        st.caption("Which block(s) draw from each video's intro/outro-tagged clips (tag scenes in "
                   "app.py). Add blocks alongside the defaults, or remove a default and pick others "
                   "instead — a video with no tagged clips for a block just shows its normal full "
                   "availability there, same as everywhere else.")
        block_options = list(range(1, len(segments) + 1))

        _intro_shadow = [b for b in (st.session_state.get("intro_block_indices") or [1]) if b in block_options] or [1]
        intro_blocks_1based = st.multiselect(
            "Intro blocks", block_options, default=_intro_shadow, key="intro_block_picker",
        )
        st.session_state["intro_block_indices"] = intro_blocks_1based
        intro_blocks_0based = {b - 1 for b in intro_blocks_1based}

        _outro_default = [len(segments)]
        _outro_shadow = [b for b in (st.session_state.get("outro_block_indices") or _outro_default)
                         if b in block_options] or _outro_default
        outro_blocks_1based = st.multiselect(
            "Outro blocks", block_options, default=_outro_shadow, key="outro_block_picker",
        )
        st.session_state["outro_block_indices"] = outro_blocks_1based
        outro_blocks_0based = {b - 1 for b in outro_blocks_1based}

        _both = intro_blocks_0based & outro_blocks_0based
        if _both:
            st.caption(f"Block(s) {', '.join(str(b + 1) for b in sorted(_both))} are marked as both — "
                      f"intro takes priority there.")

if matching_mode == "Auto":
    timeline, clip_count_shortfall, duration_shortfall_sec = match_scenes_to_track(
        segments, queues,
        min_clips, max_clips, ramp_start, ramp_end, allow_same_video, sequential_mode,
        st.session_state["segment_exclusions"], st.session_state["segment_clip_reduction"],
        st.session_state["global_excluded_scenes"],
        vary_count, int(count_seed), min_clip_len_sec,
        intro_blocks_0based, outro_blocks_0based,
        weight_shape, weight_random, weight_repeat_penalty, weight_spread, weight_motion, autofill_seed,
    )

    split_count = sum(1 for e in timeline if e["split_screen"])
    total_picks = sum(len(e["scenes"]) for e in timeline)
    st.caption(f"{track_id}  |  {track['bpm']} BPM  |  {format_mmss(track['duration_sec'])}  |  "
               f"{len(timeline)} segments ({split_count} split-screen)  |  "
               f"{total_footage_sec:.0f}s of usable footage across {len(selected_videos)} videos" +
               ("  |  sequential order" if sequential_mode else ""))

    if clip_count_shortfall:
        st.warning(
            f"Ran out of usable footage {clip_count_shortfall} time(s) — some segments have fewer "
            f"clips than requested. Widen your video/tag selection for a fuller plan."
        )
    if duration_shortfall_sec > 0.5:
        st.warning(
            f"{duration_shortfall_sec:.1f}s total across the plan could NOT be filled at all — every "
            f"scene in the library was exhausted at that point. Widen your video/tag selection."
        )
    if not clip_count_shortfall and duration_shortfall_sec <= 0.5:
        st.caption(f"Every segment fully and continuously filled, no freezes or gaps ({total_picks} total slots).")

elif matching_mode == "Choreography":
    # ---------------------------------------------------------------------------
    # Choreography mode: one column per source video, one block at a time.
    # The user sees the auto pick for each video and can override with any
    # available clip from that video via a timecode dropdown.
    # Uses the same replay-from-scratch queue architecture as Manual step-through.
    # ---------------------------------------------------------------------------
    chor_key = (track_id, tuple(tag_filter), len(segments))
    if st.session_state.get("chor_state_key") != chor_key:
        st.session_state["chor_state_key"] = chor_key
        if not _consume_load_suppression("Choreography"):
            st.session_state["chor_current_block"] = 0
            st.session_state["chor_confirmed"] = {}   # seg_idx -> list of pick dicts
            st.session_state["chor_overrides"] = {}   # seg_idx -> {video_id: (scene_id, offset_sec)}

    # Guaranteed fallback regardless of which path above ran: a suppressed reset
    # (project load) doesn't guarantee these keys were actually in the loaded
    # data — a project saved without ever visiting Choreography mode has none of
    # them, since the save only includes keys that already existed at save time.
    st.session_state.setdefault("chor_current_block", 0)
    st.session_state.setdefault("chor_confirmed", {})
    st.session_state.setdefault("chor_overrides", {})

    chor_block = min(st.session_state["chor_current_block"], len(segments))
    chor_confirmed = st.session_state["chor_confirmed"]

    # Replay confirmed blocks to keep queues accurate for the current block
    chor_invalidated = []
    for i in range(chor_block):
        still_valid = []
        for pick in chor_confirmed.get(i, []):
            try:
                carve_span(queues, pick["video_id"], pick["scene_id"],
                           pick["offset_into_scene_sec"], pick["clip_duration_sec"], sequential_mode)
                still_valid.append(pick)
            except ValueError:
                chor_invalidated.append((i, pick["video_id"], pick["scene_id"]))
        chor_confirmed[i] = still_valid
    if chor_invalidated:
        st.warning(f"{len(chor_invalidated)} previously confirmed clip(s) are no longer available "
                   f"(a setting changed) and were removed — revisit those blocks.")

    # Build timeline for the chart (real picks for confirmed, empty for the rest)
    timeline = []
    for i, sg in enumerate(segments):
        picks = chor_confirmed.get(i, []) if i < chor_block else []
        timeline.append({
            "segment_index": i, "track_time": [round(sg["start"], 2), round(sg["end"], 2)],
            "segment_energy": sg["energy"], "segment_intensity": sg["intensity"],
            "clip_count": len(picks), "split_screen": len(picks) > 1, "manually_edited": True,
            "scenes": [{"chain": [p], **{k: p[k] for k in ("video_id", "scene_id", "tags", "motion_norm",
                                                             "thumbnail", "clip_duration_sec",
                                                             "offset_into_scene_sec", "clip_start_sec")}}
                       for p in picks],
        })

    clip_count_shortfall = 0
    duration_shortfall_sec = 0.0
    st.caption(f"{track_id}  |  {track['bpm']} BPM  |  {format_mmss(track['duration_sec'])}  |  "
               f"{len(segments)} segments  |  {total_footage_sec:.0f}s of usable footage across "
               f"{len(selected_videos)} videos  |  Choreography")

elif matching_mode == "Manual step-through":
    # ---------------------------------------------------------------------------
    # Manual step-through: block-by-block candidate grid. Progress is replayed
    # against freshly-built queues every rerun.
    # ---------------------------------------------------------------------------

    # selected_videos is deliberately NOT part of this key: adding a video to
    # the pool mid-stepping should just widen what's searchable from here on,
    # not wipe everything already confirmed.
    adv_key = (track_id, tuple(tag_filter), len(segments))
    if st.session_state.get("adv_state_key") != adv_key:
        st.session_state["adv_state_key"] = adv_key
        if not _consume_load_suppression("Manual step-through"):
            st.session_state["adv_current_block"] = 0
            st.session_state["adv_confirmed"] = {}
            st.session_state["adv_skips"] = {}
            st.session_state["adv_viewing"] = None

    # Guaranteed fallback regardless of which path above ran — same reasoning as
    # Choreography's equivalent: a suppressed reset (project load) doesn't
    # guarantee adv_current_block was actually in the loaded data, since the
    # save only includes it if Manual step-through had already been visited
    # in the session that produced the save file.
    st.session_state.setdefault("adv_current_block", 0)
    st.session_state.setdefault("adv_confirmed", {})
    st.session_state.setdefault("adv_skips", {})

    current_block = min(st.session_state["adv_current_block"], len(segments))
    confirmed = st.session_state["adv_confirmed"]
    skips = st.session_state["adv_skips"]

    # Replay every already-confirmed block's skips-then-picks against the fresh
    # queues, in order, so the candidate search for the CURRENT block sees
    # accurate remaining availability. Skips are replayed before picks within
    # each block, matching normal usage (skip past a bad option, then confirm
    # something better). A skip or pick that can no longer be applied (its
    # footage was invalidated by a setting change since) is dropped with a
    # warning rather than crashing the page.
    invalidated = []
    for i in range(current_block):
        still_valid_skips = []
        for sk in skips.get(i, []):
            try:
                skip_span(queues, sk["video_id"], sk["scene_id"], sequential_mode)
                still_valid_skips.append(sk)
            except ValueError:
                invalidated.append((i, sk["video_id"], sk["scene_id"]))
        skips[i] = still_valid_skips

        still_valid = []
        for pick in confirmed.get(i, []):
            try:
                carve_span(queues, pick["video_id"], pick["scene_id"],
                          pick["offset_into_scene_sec"], pick["clip_duration_sec"], sequential_mode)
                still_valid.append(pick)
            except ValueError:
                invalidated.append((i, pick["video_id"], pick["scene_id"]))
        confirmed[i] = still_valid
    if invalidated:
        st.warning(f"{len(invalidated)} previously-confirmed/skipped item(s) are no longer available "
                   f"(a setting changed since) and were removed — revisit those blocks with Previous.")

    # Also replay any skips already made for the CURRENT block (from earlier
    # reruns while still on it — e.g. you skipped, then ticked a checkbox,
    # which reran the page and rebuilt queues from scratch).
    still_valid_current_skips = []
    for sk in skips.get(current_block, []):
        try:
            skip_span(queues, sk["video_id"], sk["scene_id"], sequential_mode)
            still_valid_current_skips.append(sk)
        except ValueError:
            pass
    skips[current_block] = still_valid_current_skips

    clip_count_shortfall = sum(1 for i in range(current_block) if len(confirmed.get(i, [])) < recommended_counts[i])
    duration_shortfall_sec = 0.0  # Advanced mode never partially fills — a slot is either a full-length match or absent

    # Build the timeline for the chart's markers: real picks for confirmed
    # blocks, empty for anything not yet reached.
    timeline = []
    for i, sg in enumerate(segments):
        picks = confirmed.get(i, []) if i < current_block else []
        timeline.append({
            "segment_index": i, "track_time": [round(sg["start"], 2), round(sg["end"], 2)],
            "segment_energy": sg["energy"], "segment_intensity": sg["intensity"],
            "clip_count": len(picks), "split_screen": len(picks) > 1, "manually_edited": True,
            "scenes": [{"chain": [p], **{k: p[k] for k in ("video_id", "scene_id", "tags", "motion_norm", "thumbnail", "clip_duration_sec", "offset_into_scene_sec", "clip_start_sec")}}
                      for p in picks],
        })

    st.caption(f"{track_id}  |  {track['bpm']} BPM  |  {format_mmss(track['duration_sec'])}  |  "
               f"{len(segments)} segments  |  {total_footage_sec:.0f}s of usable footage across "
               f"{len(selected_videos)} videos  |  Manual step-through")

# ---------------------------------------------------------------------------
# Live visualization
# ---------------------------------------------------------------------------
# current_block / chor_block are defined inside their respective mode branches;
# provide safe defaults here so the chart vline code below never NameErrors
# when running in a different mode.
if matching_mode != "Manual step-through":
    current_block = 0
if matching_mode != "Choreography":
    chor_block = 0

fig = go.Figure()
if track.get("hires"):
    _rate = float(track["hires"]["rate"])
    _t = [i / _rate for i in range(len(track["hires"]["rms"]))]
    fig.add_trace(go.Scatter(x=_t, y=track["hires"]["rms"], mode="lines", name="Track energy (30 Hz)",
                              line=dict(color="rgba(100,150,255,0.5)", width=1)))
    fig.add_trace(go.Scatter(x=_t, y=track["hires"]["onset"], mode="lines", name="Onset strength",
                              line=dict(color="rgba(150,150,150,0.35)", width=1), visible="legendonly"))
else:
    fig.add_trace(go.Scatter(x=[e["time"] for e in track["energy_envelope"]],
                              y=[e["energy"] for e in track["energy_envelope"]],
                              mode="lines", name="Track energy", line=dict(color="rgba(100,150,255,0.5)")))

# Vertical markers at every cut, coloured by why the cut happened
for _kind, _colour, _label in (("change", "rgba(0,160,120,0.7)", "Cut: energy change (snapped)"),
                                ("hit", "rgba(220,60,60,0.55)", "Cut: big hit"),
                                ("fill", "rgba(230,150,30,0.6)", "Cut: max-length split (on strongest beat)"),
                                ("cinematic", "rgba(80,120,220,0.5)", "Rhythm Engine: CINEMATIC"),
                                ("fast", "rgba(230,150,30,0.6)", "Rhythm Engine: FAST"),
                                ("hyper", "rgba(220,60,60,0.55)", "Rhythm Engine: HYPER"),
                                ("override", "rgba(230,30,200,0.75)", "Rhythm Engine: Dynamic Priority Override")):
    _xs, _ys = [], []
    for _seg in segments:
        if _seg.get("cut") == _kind:
            _xs += [_seg["start"], _seg["start"], None]
            _ys += [0, 1, None]
    if _xs:
        fig.add_trace(go.Scatter(x=_xs, y=_ys, mode="lines", name=_label,
                                  line=dict(color=_colour, width=1, dash="dot"), hoverinfo="skip"))

single_x, single_y, single_text = [], [], []
split_x, split_y, split_text = [], [], []

for entry in timeline:
    mid = (entry["track_time"][0] + entry["track_time"][1]) / 2
    for s in entry["scenes"]:
        label = f"{s['video_id']} #{s['scene_id']}"
        if entry["split_screen"]:
            split_x.append(mid); split_y.append(s["motion_norm"]); split_text.append(label)
        else:
            single_x.append(mid); single_y.append(s["motion_norm"]); single_text.append(label)

fig.add_trace(go.Scatter(x=single_x, y=single_y, mode="markers", name="Matched scene",
                          marker=dict(color="orange", size=6), text=single_text,
                          hovertemplate="%{text}<br>motion=%{y:.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=split_x, y=split_y, mode="markers", name="Split-screen pick",
                          marker=dict(color="red", size=7, symbol="diamond"), text=split_text,
                          hovertemplate="%{text}<br>motion=%{y:.2f}<extra></extra>"))

# Density curve (what drives intensity) and the resulting clip count per segment
_dt, _dens = compute_density_curve(track)
fig.add_trace(go.Scatter(x=_dt, y=_dens, mode="lines", name="Density (1.5 s smoothed)",
                          line=dict(color="rgba(140,90,220,0.8)", width=2)))
_cx, _cy = [], []
for _e, _rec in zip(timeline, recommended_counts):
    _count = _rec if matching_mode == "Manual step-through" else _e["clip_count"]
    _cx += [_e["track_time"][0], _e["track_time"][1]]
    _cy += [_count, _count]
_line_name = "Recommended clips" if matching_mode == "Manual step-through" else "Simultaneous clips"
fig.add_trace(go.Scatter(x=_cx, y=_cy, mode="lines", name=_line_name, yaxis="y2",
                          line=dict(color="rgba(30,30,30,0.85)", width=2, shape="hv")))
if matching_mode == "Manual step-through":
    fig.add_vline(x=segments[min(current_block, len(segments) - 1)]["start"],
                  line=dict(color="rgba(200,30,180,0.6)", width=2, dash="dash"))
elif matching_mode == "Choreography":
    fig.add_vline(x=segments[min(chor_block, len(segments) - 1)]["start"],
                  line=dict(color="rgba(200,30,180,0.6)", width=2, dash="dash"))

fig.update_layout(
    height=380, margin=dict(t=20, b=20), xaxis_title="Time (s)", yaxis_title="Normalized level",
    yaxis2=dict(title="Clips", overlaying="y", side="right", range=[0, max_clips + 0.5], dtick=1, showgrid=False),
    legend=dict(orientation="h", y=-0.25),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Timeline table + thumbnails preview
# ---------------------------------------------------------------------------

if matching_mode == "Auto":
    st.subheader("Matched timeline")
    st.caption(
        "🔄 **Swap** — replace with a different clip here; the original can still be used elsewhere.  "
        "🚫 **Reject** — replace it here too, but permanently ban the original from the whole plan.  "
        "✕ **Remove** — delete this slot (fewer simultaneous clips); the original can still be used elsewhere."
    )
    preview_count = st.slider("Segments to preview below", 5, 50, 15)

    for entry in timeline[:preview_count]:
        t0, t1 = entry["track_time"]
        seg_idx = entry["segment_index"]
        split_tag = f"  🔲 {entry['clip_count']} CLIPS" if entry["split_screen"] else ""
        edited_tag = "  ✏️ edited" if entry["manually_edited"] else ""
        role_tag = ("  🎬 INTRO BLOCK" if seg_idx in intro_blocks_0based
                   else "  🎬 OUTRO BLOCK" if seg_idx in outro_blocks_0based else "")
        st.markdown(f"**Block {seg_idx + 1}**  {t0:.1f}s – {t1:.1f}s  (intensity {entry['segment_intensity']:.2f})"
                   f"{split_tag}{edited_tag}{role_tag}")

        if not entry["scenes"]:
            st.caption("No clip selected for this segment (all rejected).")
        else:
            cols = st.columns(len(entry["scenes"]))
            for slot_idx, (col, s) in enumerate(zip(cols, entry["scenes"])):
                with col:
                    # Sprite-sheet crop at the clip's ACTUAL (trimmed) start, same as
                    # Choreography/Manual step-through — s["thumbnail"] alone is just the
                    # generic scene keyframe taken at detection time, which ignores
                    # offset_into_scene_sec entirely and can show the wrong moment.
                    nearest_thumb = get_nearest_thumbnail(catalogues, s["video_id"], s.get("clip_start_sec", 0.0))
                    if nearest_thumb is not None:
                        st.image(nearest_thumb, width=180)
                    elif s.get("thumbnail") and Path(s["thumbnail"]).exists():
                        st.image(s["thumbnail"], width=180)
                    chain = s["chain"]
                    if len(chain) == 1:
                        st.caption(f"`{s['video_id']}` #{s['scene_id']}  (motion {s['motion_norm']:.2f})  "
                                   f"— {s['clip_duration_sec']:.1f}s")
                    else:
                        chain_desc = " → ".join(f"`{c['video_id']}`#{c['scene_id']} ({c['clip_duration_sec']:.1f}s)"
                                                 for c in chain)
                        st.caption(f"Chain of {len(chain)}: {chain_desc}  (total {s['clip_duration_sec']:.1f}s)")
                    st.caption(", ".join(s.get("tags", [])))

                    scene_key = f"{seg_idx}_{slot_idx}_{s['video_id']}_{s['scene_id']}"
                    btn_cols = st.columns(3)
                    with btn_cols[0]:
                        if st.button("🔄 Swap", key=f"swap_{scene_key}",
                                     help="Replace with a different clip here. Original stays usable elsewhere."):
                            excl = st.session_state["segment_exclusions"].setdefault(seg_idx, set())
                            for c in chain:
                                excl.add((c["video_id"], c["scene_id"]))
                            st.rerun()
                    with btn_cols[1]:
                        if st.button("🚫 Reject", key=f"reject_{scene_key}",
                                     help="Replace it here AND permanently ban it from the whole plan."):
                            for c in chain:
                                st.session_state["global_excluded_scenes"].add((c["video_id"], c["scene_id"]))
                            excl = st.session_state["segment_exclusions"].setdefault(seg_idx, set())
                            for c in chain:
                                excl.add((c["video_id"], c["scene_id"]))
                            st.rerun()
                    with btn_cols[2]:
                        if st.button("✕ Remove", key=f"remove_{scene_key}",
                                     help="Delete this slot entirely. Original stays usable elsewhere."):
                            excl = st.session_state["segment_exclusions"].setdefault(seg_idx, set())
                            for c in chain:
                                excl.add((c["video_id"], c["scene_id"]))
                            st.session_state["segment_clip_reduction"][seg_idx] = (
                                st.session_state["segment_clip_reduction"].get(seg_idx, 0) + 1
                            )
                            st.rerun()
        st.divider()

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------

elif matching_mode == "Choreography":
    # -----------------------------------------------------------------------
    # Choreography mode: block-by-block grid, one column per source video
    # -----------------------------------------------------------------------

    if chor_block >= len(segments):
        st.success(f"✅ All {len(segments)} blocks confirmed. Review below — click ✏️ Edit on any block "
                   f"to jump straight to it, or use Restart to redo.")

        chor_rev_cols = st.columns([2, 2, 3])
        with chor_rev_cols[0]:
            if st.button("↺ Restart from block 1", key="chor_restart"):
                st.session_state["chor_current_block"] = 0
                st.session_state["chor_confirmed"] = {}
                st.session_state["chor_overrides"] = {}
                st.rerun()
        with chor_rev_cols[2]:
            chor_rev_jump = st.number_input(
                "Jump to block", min_value=1, max_value=len(segments), value=1,
                key="chor_review_jump_input", label_visibility="collapsed",
            )
            if st.button("↩ Go to block", key="chor_review_jump_go"):
                st.session_state["chor_current_block"] = int(chor_rev_jump) - 1
                st.rerun()

        st.subheader("Confirmed timeline")
        chor_preview = st.slider("Blocks to show below", 5, 50, min(15, len(segments)), key="chor_preview_count")
        for entry in timeline[:chor_preview]:
            seg_idx = entry["segment_index"]
            t0, t1 = entry["track_time"]
            chor_review_role_tag = ("  🎬 INTRO" if seg_idx in intro_blocks_0based
                                    else "  🎬 OUTRO" if seg_idx in outro_blocks_0based else "")
            hdr_cols = st.columns([6, 1])
            with hdr_cols[0]:
                st.markdown(f"**Block {seg_idx + 1}** &nbsp; {t0:.1f}s – {t1:.1f}s &nbsp; "
                            f"intensity {entry['segment_intensity']:.2f}{chor_review_role_tag}")
            with hdr_cols[1]:
                if st.button("✏️ Edit", key=f"chor_edit_{seg_idx}",
                             help=f"Jump to block {seg_idx + 1} to re-pick its clips."):
                    st.session_state["chor_current_block"] = seg_idx
                    st.rerun()
            if not entry["scenes"]:
                st.caption("No clip selected for this block.")
            else:
                cols = st.columns(len(entry["scenes"]))
                for col, s in zip(cols, entry["scenes"]):
                    with col:
                        nearest_thumb = get_nearest_thumbnail(catalogues, s["video_id"], s["clip_start_sec"])
                        if nearest_thumb is not None:
                            st.image(nearest_thumb, width=180)
                        elif s.get("thumbnail") and Path(s["thumbnail"]).exists():
                            st.image(s["thumbnail"], width=180)
                        st.caption(f"`{s['video_id']}` #{s['scene_id']}  — {s['clip_duration_sec']:.1f}s")
                        st.caption(", ".join(s.get("tags", [])))
            st.divider()

    else:
        # Active block
        seg = segments[chor_block]
        block_duration = seg["end"] - seg["start"]
        chor_rec_count = recommended_counts[chor_block]
        chor_role_filter = None
        if chor_block in intro_blocks_0based:
            chor_role_filter = "intro_candidate"
        elif chor_block in outro_blocks_0based:
            chor_role_filter = "outro_candidate"
        chor_role_tag = ("  🎬 INTRO BLOCK" if chor_role_filter == "intro_candidate"
                        else "  🎬 OUTRO BLOCK" if chor_role_filter == "outro_candidate" else "")
        st.subheader(f"Block {chor_block + 1} of {len(segments)}  "
                     f"({seg['start']:.1f}s – {seg['end']:.1f}s, intensity {seg['intensity']:.2f}){chor_role_tag}")
        st.caption("Each column is one source video. The dropdown shows every available block-length window "
                   "across that video's whole remaining footage (past clips are already gone from the list). "
                   "Pick any, or leave on Auto. Tick 'Use this clip' on the columns you want, then confirm.")

        # Videos used anywhere in the immediately preceding CONFIRMED block — needed by
        # the weighted auto-pick (repeat-penalty term) below, and by render_choreography_block
        # for its blue-box "same video as previous block" highlight.
        chor_prev_block_videos = (
            {p["video_id"] for p in chor_confirmed.get(chor_block - 1, [])} if chor_block > 0 else set()
        )

        # In sequential mode, a video's footage only narrows down AFTER a block confirming
        # that video is actually Confirmed — navigating ahead (Jump to block / Next) without
        # confirming earlier blocks leaves their picks unrecorded, so forward-only narrowing
        # won't have taken effect yet for those blocks. Surface that plainly so it isn't
        # mistaken for sequential mode not working.
        if sequential_mode:
            unconfirmed_earlier = [i for i in range(chor_block) if not chor_confirmed.get(i)]
            if unconfirmed_earlier:
                nums = ", ".join(str(i + 1) for i in unconfirmed_earlier)
                st.warning(
                    f"Block{'s' if len(unconfirmed_earlier) != 1 else ''} {nums} earlier in the plan "
                    f"{'have' if len(unconfirmed_earlier) != 1 else 'has'} no confirmed clips yet. "
                    f"Sequential narrowing for a video only takes effect once a block using it is "
                    f"confirmed — if you skipped ahead without clicking Confirm & Next, go back and "
                    f"confirm those blocks first."
                )

        if st.button(f"⚡⚡ Auto-fill ALL remaining blocks ({len(segments) - chor_block} left)",
                     key="chor_autofill_all",
                     help="Automatically fills every block from here to the end using each block's own "
                          "recommended clip count, picking the best motion-matching window per chosen "
                          "video — same as ticking videos and confirming repeatedly. You can still revisit "
                          "and adjust any block afterward with Previous or ✏️ Edit."):
            for b in range(chor_block, len(segments)):
                seg_b = segments[b]
                block_duration_b = seg_b["end"] - seg_b["start"]
                rec_b = recommended_counts[b]
                seg_target_b = seg_b.get("intensity", seg_b["energy"])

                role_filter_b = None
                if b in intro_blocks_0based:
                    role_filter_b = "intro_candidate"
                elif b in outro_blocks_0based:
                    role_filter_b = "outro_candidate"

                prev_block_videos_b = (
                    {p["video_id"] for p in chor_confirmed.get(b - 1, [])} if b > 0 else set()
                )

                eligible_b = [vid for vid in selected_videos
                              if get_available_clips_for_video(
                                  queues, vid, block_duration_b, sequential_mode,
                                  st.session_state["global_excluded_scenes"], role_filter=role_filter_b)]
                shown_b = eligible_b[:int(max_chor_videos)]

                # For each eligible video, the WEIGHTED best clip from its own (already
                # role-narrowed, if tagged clips exist) pool — same Auto-fill weighting
                # used everywhere else (A has no effect: Choreography has no shape score).
                candidates_b = []
                for vid in shown_b:
                    clips_b = get_available_clips_for_video(
                        queues, vid, block_duration_b, sequential_mode,
                        st.session_state["global_excluded_scenes"], role_filter=role_filter_b)
                    if not clips_b:
                        continue
                    # Sequential mode: skip budget (weight D-controlled) rather than a hard
                    # frontmost-only restriction — see compute_skip_budget. Picking a window
                    # past the budget discards everything before it, so the budget caps how
                    # far ahead this video's own spare footage safely allows roaming, given
                    # how many blocks are still left overall.
                    if sequential_mode:
                        _remaining_blocks_b = len(segments) - b
                        _skip_b = compute_skip_budget(len(clips_b), _remaining_blocks_b, weight_spread)
                        auto_candidates_b = clips_b[:_skip_b + 1]
                    else:
                        auto_candidates_b = clips_b
                    ranked_b = rank_candidates_weighted(
                        auto_candidates_b, b, segments, chor_confirmed, queues, prev_block_videos_b,
                        weight_shape, weight_random, weight_repeat_penalty, weight_spread,
                        weight_motion, autofill_seed,
                    )
                    best_b = ranked_b[0]
                    is_tagged_b = bool(role_filter_b and best_b.get(role_filter_b))
                    candidates_b.append((vid, best_b, is_tagged_b, abs(best_b["motion_norm"] - seg_target_b)))

                # Marked block: prefer candidates whose clip actually IS tagged over ones
                # that fell back to untagged footage. If there aren't enough tagged
                # candidates to fill every slot, the REMAINING slots naturally fall back
                # to the best untagged ones — this IS the fallback-to-original-behaviour
                # rule, expressed as "run out of tagged options, keep going down the list."
                if role_filter_b:
                    candidates_b.sort(key=lambda t: (not t[2], t[3]))
                else:
                    candidates_b.sort(key=lambda t: t[3])
                chosen_b = [(vid, best) for vid, best, _, _ in candidates_b[:rec_b]]
                chosen_vids_b = {vid for vid, _ in chosen_b}

                new_picks_b = []
                for vid, clip in chosen_b:
                    try:
                        pick = carve_span(queues, vid, clip["scene_id"], clip["offset_sec"],
                                          block_duration_b, sequential_mode)
                        new_picks_b.append(pick)
                        st.session_state[f"chor_pick_{b}_{vid}"] = True
                        st.session_state.setdefault("chor_overrides", {}).setdefault(b, {})[vid] = \
                            (clip["scene_id"], clip["offset_sec"])
                    except ValueError:
                        st.session_state[f"chor_pick_{b}_{vid}"] = False

                for vid in shown_b:
                    if vid not in chosen_vids_b:
                        st.session_state[f"chor_pick_{b}_{vid}"] = False

                chor_confirmed[b] = new_picks_b

            st.session_state["chor_current_block"] = len(segments)
            st.rerun()

        # Navigation + block jump
        chor_nav_cols = st.columns([2, 2, 3])
        with chor_nav_cols[0]:
            if st.button("◀ Previous", disabled=(chor_block == 0), key="chor_prev"):
                st.session_state["chor_current_block"] = chor_block - 1
                st.rerun()
        with chor_nav_cols[2]:
            chor_jump = st.number_input(
                "Jump to block", min_value=1, max_value=len(segments),
                value=chor_block + 1, key="chor_jump_input", label_visibility="collapsed",
            )
            if st.button("↩ Go to block", key="chor_jump_go"):
                st.session_state["chor_current_block"] = int(chor_jump) - 1
                st.rerun()
        with chor_nav_cols[1]:
            if st.button("Confirm & Next ▶", type="primary", key="chor_next"):
                new_picks = []
                overrides = st.session_state.get("chor_overrides", {}).get(chor_block, {})
                eligible_videos = [vid for vid in selected_videos
                                   if get_available_clips_for_video(
                                       queues, vid, block_duration, sequential_mode,
                                       st.session_state["global_excluded_scenes"],
                                       role_filter=chor_role_filter)]
                shown_videos = eligible_videos[:int(max_chor_videos)]
                seg_target = seg.get("intensity", seg["energy"])

                # Confirm only the ticked columns (checkbox state drives this, not position)
                chosen_videos = [vid for vid in shown_videos
                                 if st.session_state.get(f"chor_pick_{chor_block}_{vid}", False)]

                for video_id in chosen_videos:
                    override = overrides.get(video_id)
                    clips = get_available_clips_for_video(
                        queues, video_id, block_duration, sequential_mode,
                        st.session_state["global_excluded_scenes"], role_filter=chor_role_filter)
                    if not clips:
                        continue

                    if override is not None:
                        target = next((c for c in clips
                                       if c["scene_id"] == override[0]
                                       and abs(c["offset_sec"] - override[1]) < 0.1), None)
                        if target is None:
                            st.warning(f"Override clip for {video_id} is no longer available — skipped.")
                            continue
                        use_scene_id = target["scene_id"]
                        use_offset = target["offset_sec"]
                    else:
                        # Auto: weighted pick, same Auto-fill weighting used everywhere else
                        # (A has no effect here — Choreography has no shape score). Sequential
                        # mode: skip budget (weight D-controlled) instead of a hard frontmost-
                        # only restriction — see compute_skip_budget and the matching comment
                        # in render_choreography_block's preview above.
                        if sequential_mode:
                            _remaining_blocks = len(segments) - chor_block
                            _skip = compute_skip_budget(len(clips), _remaining_blocks, weight_spread)
                            auto_candidates = clips[:_skip + 1]
                        else:
                            auto_candidates = clips
                        ranked = rank_candidates_weighted(
                            auto_candidates, chor_block, segments, chor_confirmed, queues,
                            chor_prev_block_videos, weight_shape, weight_random,
                            weight_repeat_penalty, weight_spread, weight_motion, autofill_seed,
                        )
                        best = ranked[0]
                        use_scene_id = best["scene_id"]
                        use_offset = best["offset_sec"]

                    try:
                        pick = carve_span(queues, video_id, use_scene_id,
                                          use_offset, block_duration, sequential_mode)
                        new_picks.append(pick)
                    except ValueError as e:
                        st.warning(f"Couldn't confirm clip for {video_id}: {e}")

                chor_confirmed[chor_block] = new_picks
                st.session_state["chor_current_block"] = chor_block + 1
                st.rerun()

        # Used-so-far / remaining footage per video, same computation Manual step-through
        # uses for its candidate cards.
        chor_video_stats = {}
        for vid in selected_videos:
            used = sum(p["clip_duration_sec"] for i in range(chor_block) for p in chor_confirmed.get(i, [])
                      if p["video_id"] == vid)
            remaining = sum(span["remaining_sec"] for span in queues.get(vid, []))
            chor_video_stats[vid] = {"used": used, "remaining": remaining}

        chor_weighting = {
            "segments": segments, "confirmed": chor_confirmed,
            "weight_shape": weight_shape, "weight_random": weight_random,
            "weight_repeat_penalty": weight_repeat_penalty, "weight_spread": weight_spread,
            "weight_motion": weight_motion, "seed": autofill_seed,
        }

        # The choreography grid fragment — dropdowns update live without a full rerun
        render_choreography_block(
            chor_block, seg, selected_videos, int(max_chor_videos),
            chor_rec_count, queues, sequential_mode,
            st.session_state["global_excluded_scenes"], catalogues,
            chor_prev_block_videos, chor_video_stats, chor_role_filter, chor_weighting,
        )

else:
    # -----------------------------------------------------------------------
    # Manual step-through UI
    # -----------------------------------------------------------------------

    if current_block >= len(segments):
        st.success(f"✅ All {len(segments)} blocks confirmed. Review below — click ✏️ Edit on any block "
                   f"to jump straight to it, or use Restart to start over.")

        rev_top_cols = st.columns([2, 2, 3])
        with rev_top_cols[0]:
            if st.button("↺ Restart from block 1", key="adv_restart"):
                st.session_state["adv_current_block"] = 0
                st.session_state["adv_viewing"] = None
                st.rerun()
        with rev_top_cols[2]:
            rev_jump = st.number_input(
                "Jump to block", min_value=1, max_value=len(segments), value=1,
                key="adv_review_jump_input",
                help="Type a block number and press Enter or click Go.",
                label_visibility="collapsed",
            )
            if st.button("↩ Go to block", key="adv_review_jump_go"):
                st.session_state["adv_current_block"] = int(rev_jump) - 1
                st.session_state["adv_viewing"] = None
                st.rerun()

        st.subheader("Confirmed timeline")
        preview_count = st.slider("Blocks to show below", 5, 50, min(15, len(segments)), key="adv_preview_count")
        for entry in timeline[:preview_count]:
            seg_idx = entry["segment_index"]
            t0, t1 = entry["track_time"]
            split_tag = f"  🔲 {entry['clip_count']} CLIPS" if entry["split_screen"] else ""
            adv_review_role_tag = ("  🎬 INTRO" if seg_idx in intro_blocks_0based
                                   else "  🎬 OUTRO" if seg_idx in outro_blocks_0based else "")

            hdr_cols = st.columns([6, 1])
            with hdr_cols[0]:
                st.markdown(
                    f"**Block {seg_idx + 1}** &nbsp; {t0:.1f}s – {t1:.1f}s &nbsp; "
                    f"intensity {entry['segment_intensity']:.2f}{split_tag}{adv_review_role_tag}"
                )
            with hdr_cols[1]:
                if st.button("✏️ Edit", key=f"adv_edit_block_{seg_idx}",
                             help=f"Jump to block {seg_idx + 1} to re-pick its clips."):
                    st.session_state["adv_current_block"] = seg_idx
                    st.session_state["adv_viewing"] = None
                    st.rerun()

            if not entry["scenes"]:
                st.caption("No clip selected for this block.")
            else:
                cols = st.columns(len(entry["scenes"]))
                for col, s in zip(cols, entry["scenes"]):
                    with col:
                        nearest_thumb = get_nearest_thumbnail(catalogues, s["video_id"], s["clip_start_sec"])
                        if nearest_thumb is not None:
                            st.image(nearest_thumb, width=180)
                        elif s.get("thumbnail") and Path(s["thumbnail"]).exists():
                            st.image(s["thumbnail"], width=180)
                        st.caption(f"`{s['video_id']}` #{s['scene_id']}  — {s['clip_duration_sec']:.1f}s")
                        st.caption(", ".join(s.get("tags", [])))
            st.divider()

    else:
        seg = segments[current_block]
        rec_count = recommended_counts[current_block]
        already = confirmed.get(current_block, [])
        already_keys = {(p["video_id"], p["scene_id"]) for p in already}

        adv_role_tag = ("  🎬 INTRO BLOCK" if current_block in intro_blocks_0based
                       else "  🎬 OUTRO BLOCK" if current_block in outro_blocks_0based else "")
        st.subheader(f"Block {current_block + 1} of {len(segments)}  "
                     f"({seg['start']:.1f}s – {seg['end']:.1f}s, intensity {seg['intensity']:.2f}){adv_role_tag}")
        st.caption(f"Recommended simultaneous clips for this block: {rec_count} "
                   f"(from your split-screen settings) — this is a guide only; select any number below.")

        if st.button(f"⚡⚡ Auto-fill ALL remaining blocks ({len(segments) - current_block} left)",
                     key="adv_autofill_all",
                     help="Automatically picks the best-scoring clip(s) for every block from here to the "
                          "end, using each block's own recommended count — same as clicking Auto-fill and "
                          "Confirm & Next repeatedly. You can still revisit and adjust any block afterward "
                          "with Previous."):
            for b in range(current_block, len(segments)):
                seg_b = segments[b]
                audio_curve_b = get_block_audio_curve(track, seg_b["start"], seg_b["end"])
                _excl_b = st.session_state["segment_exclusions"].get(b, set())
                _global_excl = st.session_state["global_excluded_scenes"]
                cands_b, _ = find_shape_candidates(
                    seg_b, queues, sequential_mode, _excl_b, _global_excl, audio_curve_b,
                    int(max_shape_matches), restrict_to_front=sequential_mode,
                    remaining_blocks=len(segments) - b, weight_spread=weight_spread,
                )
                # Auto-fill ALL is a fully automatic process (like Auto mode and
                # Choreography's own Auto-fill ALL), so marked blocks opportunistically
                # hard-filter here, unlike the manual grid above which only highlights.
                if b in intro_blocks_0based:
                    role_matches_b = [c for c in cands_b if c.get("intro_candidate")]
                    if role_matches_b:
                        cands_b = role_matches_b
                elif b in outro_blocks_0based:
                    role_matches_b = [c for c in cands_b if c.get("outro_candidate")]
                    if role_matches_b:
                        cands_b = role_matches_b

                # --- Fallback chain: relax constraints progressively ---
                # Level 1: shape search found nothing → drop role filter and retry
                if not cands_b and (b in intro_blocks_0based or b in outro_blocks_0based):
                    cands_b, _ = find_shape_candidates(
                        seg_b, queues, sequential_mode, _excl_b, _global_excl, audio_curve_b,
                        int(max_shape_matches), restrict_to_front=sequential_mode,
                        remaining_blocks=len(segments) - b, weight_spread=weight_spread,
                    )

                # Level 2: still nothing in sequential mode → retry non-sequentially
                # (footage may be available further along a video's queue)
                if not cands_b and sequential_mode:
                    cands_b, _ = find_shape_candidates(
                        seg_b, queues, False, _excl_b, _global_excl, audio_curve_b,
                        int(max_shape_matches),
                    )

                # Level 3: shape search still empty (no motion curve data, or all excluded)
                # → take the best raw span by motion match, ignoring shape scoring entirely
                if not cands_b:
                    block_intensity_b = seg_b.get("intensity", seg_b.get("energy", 0.5))
                    raw_spans = get_candidate_spans(queues, False, _global_excl, min_duration=0.0)
                    raw_spans = [s for s in raw_spans if (s["video_id"], s["scene_id"]) not in _excl_b
                                 and s["remaining_sec"] >= (seg_b["end"] - seg_b["start"]) - 1e-6]
                    if raw_spans:
                        best_span = min(raw_spans, key=lambda s: abs(s["motion_norm"] - block_intensity_b))
                        cands_b = [{
                            "video_id": best_span["video_id"],
                            "scene_id": best_span["scene_id"],
                            "scene_start_sec": best_span["scene_start_sec"],
                            "window_offset_sec": round(best_span["offset_sec"], 3),
                            "window_duration_sec": round(seg_b["end"] - seg_b["start"], 3),
                            "score": 0.0,
                            "motion_norm": best_span["motion_norm"],
                            "curve_slice": np.array([]),
                            "fps": 25.0,
                            "tags": best_span["tags"],
                            "thumbnail": best_span["thumbnail"],
                            "intro_candidate": best_span.get("intro_candidate", False),
                            "outro_candidate": best_span.get("outro_candidate", False),
                        }]

                rec_b = recommended_counts[b]
                # Each block's "previous block" is whatever was JUST decided for b-1 in this same
                # loop (or already-confirmed history if b == current_block) — cascades correctly,
                # same as the repetition penalty would see stepping through manually one at a time.
                prev_videos_b = {p["video_id"] for p in confirmed.get(b - 1, [])} if b > 0 else set()
                ranked_b = rank_candidates_weighted(
                    cands_b, b, segments, confirmed, queues, prev_videos_b,
                    weight_shape, weight_random, weight_repeat_penalty, weight_spread,
                    weight_motion, autofill_seed,
                )
                chosen_keys_b = {(c["video_id"], c["scene_id"]) for c in ranked_b[:rec_b]}
                new_picks = []
                for c in cands_b:
                    pick_key = f"adv_pick_{b}_{c['video_id']}_{c['scene_id']}"
                    if (c["video_id"], c["scene_id"]) in chosen_keys_b:
                        try:
                            pick = carve_span(queues, c["video_id"], c["scene_id"],
                                              c["window_offset_sec"], c["window_duration_sec"], sequential_mode)
                            new_picks.append(pick)
                            st.session_state[pick_key] = True
                        except ValueError:
                            st.session_state[pick_key] = False
                    else:
                        st.session_state[pick_key] = False
                confirmed[b] = new_picks
            st.session_state["adv_current_block"] = len(segments)
            st.session_state["adv_viewing"] = None
            st.rerun()

        nav_cols = st.columns([2, 2, 3])
        with nav_cols[0]:
            if st.button("◀ Previous", disabled=(current_block == 0), key="adv_prev"):
                st.session_state["adv_current_block"] = current_block - 1
                st.session_state["adv_viewing"] = None
                st.rerun()
        with nav_cols[2]:
            jump_target = st.number_input(
                "Jump to block", min_value=1, max_value=len(segments),
                value=current_block + 1, key="adv_jump_input",
                help="Type a block number and press Enter to jump directly to it.",
                label_visibility="collapsed",
            )
            if st.button("↩ Go to block", key="adv_jump_go"):
                st.session_state["adv_current_block"] = int(jump_target) - 1
                st.session_state["adv_viewing"] = None
                st.rerun()
        with nav_cols[1]:
            if st.button("Confirm & Next ▶", type="primary", key="adv_next"):
                chosen = [c for c in st.session_state.get(f"adv_candidates_{current_block}", [])
                         if st.session_state.get(f"adv_pick_{current_block}_{c['video_id']}_{c['scene_id']}")]
                new_confirmed = []
                for c in chosen:
                    try:
                        pick = carve_span(queues, c["video_id"], c["scene_id"],
                                          c["window_offset_sec"], c["window_duration_sec"], sequential_mode)
                        new_confirmed.append(pick)
                    except ValueError as e:
                        st.warning(f"Couldn't use {c['video_id']} #{c['scene_id']}: {e}")
                confirmed[current_block] = new_confirmed
                st.session_state["adv_current_block"] = current_block + 1
                st.session_state["adv_viewing"] = None
                st.rerun()

        # The search is real work (a sliding window across every eligible scene) —
        # cache it per block so a checkbox tick or a View Fit click (both just
        # normal Streamlit reruns) reuses the result instead of re-searching.
        # Only recompute if something that could actually change the answer has:
        # which block, the settings, or which picks were confirmed for EARLIER
        # blocks (that changes what footage is still available here via replay).
        prior_picks_fingerprint = tuple(sorted(
            (p["video_id"], p["scene_id"], p["offset_into_scene_sec"], p["clip_duration_sec"])
            for i in range(current_block) for p in confirmed.get(i, [])
        ))
        current_skips_fingerprint = tuple(sorted(
            (sk["video_id"], sk["scene_id"]) for sk in skips.get(current_block, [])
        ))
        # selected_videos is deliberately NOT part of adv_key (so toggling a video doesn't wipe
        # confirmed progress — see adv_key's own comment), but the SEARCH result genuinely does
        # depend on it: deselecting a video removes it from queues, and without this the cached
        # candidate list would still show that video's clips until something else invalidated it.
        search_key = (adv_key, current_block, int(max_shape_matches), sequential_mode,
                     prior_picks_fingerprint, current_skips_fingerprint, tuple(sorted(selected_videos)))
        cache = st.session_state.get(f"adv_search_cache_{current_block}")
        block_audio_curve = get_block_audio_curve(track, seg["start"], seg["end"])  # cheap — always needed, cache or not

        if cache and cache["key"] == search_key:
            candidates, missing_curve_videos = cache["candidates"], cache["missing"]
        else:
            candidates, missing_curve_videos = find_shape_candidates(
                seg, queues, sequential_mode, st.session_state["segment_exclusions"].get(current_block, set()),
                st.session_state["global_excluded_scenes"], block_audio_curve, int(max_shape_matches),
            )
            st.session_state[f"adv_search_cache_{current_block}"] = {
                "key": search_key, "candidates": candidates, "missing": missing_curve_videos,
            }

        # Marked intro/outro blocks surface their tagged candidates first rather than
        # hiding the rest — you're already choosing by hand here, so a highlight that
        # preserves every legitimate shape match felt like the better fit than a hard
        # filter (unlike Auto/Choreography, which do filter). Stable sort: within each
        # group (tagged vs not), the existing best-score-first order is untouched.
        adv_role_filter = None
        if current_block in intro_blocks_0based:
            adv_role_filter = "intro_candidate"
        elif current_block in outro_blocks_0based:
            adv_role_filter = "outro_candidate"
        if adv_role_filter:
            candidates = sorted(candidates, key=lambda c: not c.get(adv_role_filter, False))

        st.session_state[f"adv_candidates_{current_block}"] = candidates

        if missing_curve_videos:
            st.warning(f"{len(missing_curve_videos)} video(s) have no frame-by-frame motion data yet and were "
                      f"skipped: {', '.join(sorted(missing_curve_videos))}. Run backfill_motion_curve.py in "
                      f"Colab to include them.")
        if not candidates:
            st.info("No candidate clips at least this block's length are currently available. "
                    "Try a shorter block (adjust segmentation), or widen your video/tag selection.")

        prev_block_videos = {p["video_id"] for p in confirmed.get(current_block - 1, [])} if current_block > 0 else set()

        video_stats = {}
        for c in candidates:
            vid = c["video_id"]
            if vid in video_stats:
                continue
            used = sum(p["clip_duration_sec"] for i in range(current_block) for p in confirmed.get(i, [])
                      if p["video_id"] == vid)
            remaining = sum(span["remaining_sec"] for span in queues.get(vid, []))
            video_stats[vid] = {"used": used, "remaining": remaining}

        autofill_weights = {
            "segments": segments, "confirmed": confirmed, "queues": queues,
            "weight_shape": weight_shape, "weight_random": weight_random,
            "weight_repeat_penalty": weight_repeat_penalty, "weight_spread": weight_spread,
            "weight_motion": weight_motion, "seed": autofill_seed,
        }
        render_candidate_grid(candidates, current_block, already_keys, seg, block_audio_curve,
                              prev_block_videos, video_stats, recommended_counts[current_block],
                              autofill_weights, catalogues)


if matching_mode == "Manual step-through" and current_block < len(segments):
    st.warning(
        f"Only {current_block} of {len(segments)} blocks confirmed — exporting/rendering now will "
        f"leave the rest empty. Finish stepping through above first, or export anyway if intentional."
    )
if matching_mode == "Choreography" and chor_block < len(segments):
    st.warning(
        f"Only {chor_block} of {len(segments)} blocks confirmed — exporting/rendering now will "
        f"leave the rest empty. Finish stepping through above first, or export anyway if intentional."
    )

st.subheader("Export")

# Attach resolved local file paths so the Blender add-on can load footage
# and audio directly, without re-deriving them from the catalogues.
for _entry in timeline:
    for _slot in _entry["scenes"]:
        for _link in _slot["chain"]:
            _link["source_path"] = colab_to_local(catalogues[_link["video_id"]]["source_path"])

export_plan = {
    "track_id": track_id,
    "matching_mode": matching_mode,
    "track_source_path": colab_to_local(track["source_path"]),
    "track_duration_sec": track["duration_sec"],
    "bpm": track["bpm"],
    "segmentation": {
        "method": segmentation_method if track.get("hires") else "legacy",
        "change_window_secs": change_window_secs,
        "change_threshold": change_threshold,
        "min_segment_sec": min_segment_sec,
        "max_segment_sec": max_segment_sec,
        "hyper_delta_thresh": hyper_delta_thresh,
        "fast_energy_thresh": fast_energy_thresh,
        "hyper_cooldown_bars": hyper_cooldown_bars,
        "phrase_lock_bars": phrase_lock_bars,
        "cinematic_bars": cinematic_bars,
        "dynamic_priority_override": dynamic_priority_override,
        "override_delta_thresh": override_delta_thresh,
        "use_bar_snapping": use_bar_snapping,
        "beats_per_bar": beats_per_bar,
        "react_to_hits": react_to_hits,
        "hit_threshold": hit_threshold,
        "snap_secs": snap_secs,
    },
    "sequential_video_order": sequential_mode,
    "globally_rejected_clips": [list(k) for k in st.session_state["global_excluded_scenes"]],
    "video_time_ranges": {k: list(v) for k, v in st.session_state["video_time_ranges"].items()},
    "tag_filter": tag_filter,
    "selected_videos": selected_videos,
    "split_screen_enabled": split_screen_enabled,
    "min_clips": min_clips,
    "max_clips": max_clips,
    "ramp_start": ramp_start,
    "ramp_end": ramp_end,
    "density_contrast": density_contrast,
    "vary_clip_count": vary_count,
    "clip_count_seed": int(count_seed),
    "min_clip_len_sec": min_clip_len_sec,
    "allow_same_video": allow_same_video,
    "timeline": timeline,
}

if st.button("💾 Save plan"):
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLANS_DIR / f"{track_id}_plan.json"
    out_path.write_text(json.dumps(export_plan, indent=2))
    if _OAUTH_AVAILABLE and st.session_state.get(_OAUTH_SESSION_KEY):
        _err = push_file_with_oauth(
            st.session_state[_OAUTH_SESSION_KEY], out_path, "scene-labeling/compilation_plans"
        )
        if _err:
            st.warning(f"Saved locally but Drive upload failed: {_err}")
        else:
            st.success(f"Saved and uploaded to Drive: {out_path.name}")
    else:
        st.success(f"Saved locally: {out_path.name}")

st.download_button(
    "Download plan as JSON",
    data=json.dumps(export_plan, indent=2),
    file_name=f"{track_id}_plan.json",
    mime="application/json",
)

# ---------------------------------------------------------------------------
# Render preview — versioned, stored in Drive, viewable side-by-side
# ---------------------------------------------------------------------------

st.subheader("Preview")

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
safe_track_id = sanitize_filename(track_id)
preview_pattern = f"temp-Video-preview-{safe_track_id}.*.mp4"


def existing_preview_versions() -> list[Path]:
    files = list(PREVIEW_DIR.glob(preview_pattern))
    return sorted(files, key=lambda p: p.name)


def next_preview_path() -> Path:
    existing = existing_preview_versions()
    used_numbers = []
    for p in existing:
        m = re.search(r"\.(\d{3})\.mp4$", p.name)
        if m:
            used_numbers.append(int(m.group(1)))
    next_num = (max(used_numbers) + 1) if used_numbers else 1
    return PREVIEW_DIR / f"temp-Video-preview-{safe_track_id}.{next_num:03d}.mp4"


if st.button("🎬 Render Preview", type="primary"):
    out_path = next_preview_path()
    progress_bar = st.progress(0.0, text="Rendering...")

    def _progress(done, total):
        progress_bar.progress(done / total, text=f"Rendering segment {done}/{total}...")

    try:
        render_plan_dict(export_plan, str(out_path), progress_callback=_progress)
        progress_bar.progress(1.0, text="Done.")
        if _OAUTH_AVAILABLE and st.session_state.get(_OAUTH_SESSION_KEY):
            progress_bar.progress(1.0, text="Uploading to Drive…")
            _err = push_file_with_oauth(
                st.session_state[_OAUTH_SESSION_KEY], out_path, "scene-labeling/previews"
            )
            if _err:
                st.warning(f"Rendered but Drive upload failed: {_err}")
            else:
                st.success(f"Rendered and uploaded to Drive: {out_path.name}")
        else:
            st.success(f"Rendered {out_path.name}")
        st.rerun()
    except Exception as e:
        st.error(f"Render failed: {e}")

versions = existing_preview_versions()
if versions:
    st.caption(f"{len(versions)} preview(s) for this track — compare side by side below.")
    cols_per_row = 2
    for row_start in range(0, len(versions), cols_per_row):
        row = versions[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, path in zip(cols, row):
            with col:
                st.video(str(path))
                st.caption(path.name)
                if st.button("🗑️ Delete", key=f"delete_preview_{path.name}"):
                    path.unlink(missing_ok=True)
                    st.rerun()
else:
    st.caption("No previews rendered yet for this track.")
