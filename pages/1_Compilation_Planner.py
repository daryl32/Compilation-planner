"""
Compilation Planner — interactive matching of video scenes to a music track's
energy/beat timeline, with a live visualization and JSON export for the VSE add-on.
"""

import hashlib
import inspect
import json
import math
import os
import random
import re
import datetime
import time
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from render_preview import render_plan_dict, colab_to_local
from renders import (delete_render, load_render, render_files, upload_render, clear_local,
                     auto_clear_enabled)
import video_ranking as VR
import catalogue_store as CStore
import autofill as AF
import similarity as SIM
import cut_scoring as CS
import smart_cuts as SM
import story_map as STM
import scene_roles as SR
from audio_charts import add_analysis_overlays
import preview_snippets
from library_common import (safe_image, video_details, all_actors, all_video_tags, matches_details, scene_tags, render_range_picker, library_ranges, master_thumbnail_path, proxy_path,
                            refresh_caches_after_sync, show_render_player, shared_cache,
                            clear_shared_data_caches, memory_check)

# Cutting, shape matching and the footage pool live in core/ (no Streamlit) so
# scripts and background jobs can use them too. The few that read this page's
# settings have thin wrappers below that pass those settings in.
import core.segmentation as CSEG
import core.autotune as CAT
import core.footage as CFOOT
import core.picking as CPICK
from core.segmentation import (SEG_ADAPTIVE, SEG_RHYTHM, SEG_SCORED, SEG_SMART, SEG_METHOD_KEYS,
                               build_segments, compute_density_curve, apply_segment_intensity)
from core.autotune import AUDIO_STYLES, AUDIO_STYLE_CUSTOM
import core.targets as CTGT
import core.cut_edits as CED
import core.matching as CMATCH
import core.styles as CSTY
from core.matching import prefer_role, widen_for_repeat_rule, cand_key, why_key, format_mmss
from cut_editor import cut_editor as _cut_editor_widget, media_url as _media_url
from core.shape import SHAPE_MATCH_POINTS, resample_curve, shape_score, get_block_audio_curve
from core.footage import (_tc_to_seconds, _overlap_with_range, get_candidate_spans, has_role, role_supply,
                          role_needs, consume_span, limit_queue_before, split_span_at,
                          skip_span, carve_span)
from core.picking import select_autofill_picks, repeat_fallback_videos
from diagnostics import timed as _timed   # 🩺 step timings for the Server page

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


def rerun_fragment():
    """Rerun only the fragment this is called from. A bare st.rerun() defaults to
    scope="app" — a FULL page rerun even when called inside a fragment — which is
    what made every range-picker click reload the whole page. Falls back to a plain
    rerun on a Streamlit too old for scope=, or when not actually inside a fragment
    (fragment is a no-op decorator on old versions)."""
    from streamlit.errors import StreamlitAPIException
    try:
        st.rerun(scope="fragment")
    except (TypeError, StreamlitAPIException):
        st.rerun()


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

from config import CATALOGUE_DIR, AUDIO_DIR, PLANS_DIR, PREVIEW_DIR, PROJECTS_DIR

# Google Drive sync — graceful fallback if credentials not present
try:
    from drive_sync import (sync_pull, push_file_to_drive, push_directory_to_drive,
                            credentials_available, download_source_video)
    _DRIVE_SYNC_AVAILABLE = credentials_available()
except ImportError:
    _DRIVE_SYNC_AVAILABLE = False

# Sign-in, whitelist, Drive connect, title and the shared sidebar — see page_setup.py
from page_setup import (page_setup, push_file_with_oauth, APP_VERSION, drive_token,
                        OAUTH_AVAILABLE as _OAUTH_AVAILABLE, DRIVE_SESSION_KEY as _OAUTH_SESSION_KEY)
page_setup("Compilation Planner", "pages/1_Compilation_Planner.py")
_PAGE_T0 = time.perf_counter()   # 🩺 whole-page run time — see _record_page_run


def _record_page_run() -> None:
    """🩺 How long this full run of the page took, for the Server page's timings
    (fragment reruns and runs that stop early — e.g. no track yet — aren't counted)."""
    import diagnostics as _diag
    _diag.record("Planner page (full run)", time.perf_counter() - _PAGE_T0,
                 str(st.session_state.get("ui_section", ""))[:2].strip())
# The background Drive sync timer may have pulled new catalogues/audio since
# the caches were filled — reload them once if so.
refresh_caches_after_sync()


def sanitize_filename(name: str) -> str:
    """Strip characters that aren't valid in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def resolve_thumbnail(stored_path: str, video_id: str) -> Path:
    filename = Path(stored_path).name
    return CATALOGUE_DIR / "thumbnails" / video_id / filename


# ---------------------------------------------------------------------------
# Data loading (cached — only recomputes when inputs actually change)
# ---------------------------------------------------------------------------

def list_tracks():
    # Not cached: a folder listing is cheap, and new tracks (synced, or analysed
    # on the server from the Media Library) should appear without a restart.
    return sorted(f.stem for f in AUDIO_DIR.glob("*.json") if f.name != "audio_index.json")


@st.cache_data(max_entries=4)
def _load_track_file(track_id: str, mtime: float) -> dict:
    with open(AUDIO_DIR / f"{track_id}.json") as f:
        return json.load(f)


def load_track(track_id: str) -> dict:
    """The track's catalogue — re-read whenever the file changes (e.g. after the
    server adds its extra analysis)."""
    try:
        mtime = (AUDIO_DIR / f"{track_id}.json").stat().st_mtime
    except OSError:
        mtime = 0.0
    return _load_track_file(track_id, mtime)


def load_all_catalogues() -> dict:
    """video_id -> catalogue dict, WITHOUT motion_curve, tags corrected. ONE
    SHARED COPY — read it, never change it. Comes from catalogue_store, which
    re-reads only the files that changed, so Reviewer edits and Drive syncs
    show up on the next click. (History: st.cache_data deep-copied the whole
    library on every call and ran the 4 GB server out of memory; motion curves
    stay out of it — Advanced Shape Matching loads them per video below.)"""
    return CStore.all_catalogues()


def load_motion_curves_for_video(video_id: str) -> dict:
    """scene_id -> motion_curve, for ONE video — reloaded when that video's
    catalogue changes (e.g. a Reviewer split). SHARED — read it, never change it."""
    return CStore.motion_curves(video_id)


def seconds_to_time(sec: float) -> datetime.time:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return datetime.time(hour=min(h, 23), minute=m, second=s)


def time_to_seconds(t: datetime.time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second


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


@st.cache_data
def _all_video_durations() -> dict:
    """video_id -> duration in seconds. Small, so cheap for st.cache_data to copy on
    every call — unlike load_all_catalogues(), which deep-copies the whole library each
    time and used to be called once per video row (and again per row in the sort)."""
    out = {}
    for vid, cat in load_all_catalogues().items():
        out[vid] = _tc_to_seconds(cat["scenes"][-1]["end_tc"]) if cat.get("scenes") else 0.0
    return out


def get_video_duration(video_id: str) -> float:
    return _all_video_durations().get(video_id, 0.0)


def effective_time_ranges() -> dict:
    """{video_id: (start, end)} each video actually uses: this project's override
    (🎚️, session_state["video_time_ranges"]) if there is one, otherwise its Media
    Library range (📚), otherwise the full video (not listed). An override of None
    means "full video" even when the library has a range."""
    merged = dict(library_ranges())
    merged.update(st.session_state.get("video_time_ranges", {}))
    return {vid: tuple(rng) for vid, rng in merged.items() if rng}


def build_footage_queues(candidate_video_ids: tuple, tag_filter: tuple, video_time_ranges: dict = None) -> dict:
    """core.footage.build_footage_queues on this library's catalogues."""
    return CFOOT.build_footage_queues(load_all_catalogues(), candidate_video_ids, tag_filter, video_time_ranges,
                                      thumbnail_for=resolve_thumbnail, roles_of=SR.scene_roles)


# ---------------------------------------------------------------------------
# 🎭 Scene roles in the plan. A block's role (a role id — "intro", "slow",
# "action", "outro" or one of your own) comes from the story map, or from the
# Intro/Outro blocks picker when there's no story map in use. Roles are
# PREFERRED: a block draws from clips with its role where there are any, and
# falls back to normal picks otherwise. Sequential mode keeps each video in order.
# ---------------------------------------------------------------------------


def role_chips(c: dict) -> str:
    """'🎬 Intro  💥 Action' for a clip's roles ('' if it has none)."""
    ids = c.get("roles") or []
    if not ids:
        return ""
    rb = SR.roles_by_id()
    return "  ".join(SR.label(r, rb) for r in ids)


# ---------------------------------------------------------------------------
# The matcher lives in core/matching.py. These wrappers keep the names this page
# has always used and hand the matcher this plan's settings (plan_context()).
# ---------------------------------------------------------------------------

def plan_context() -> CMATCH.PlanContext:
    """This plan's settings, as the matcher needs them (read fresh on every call)."""
    ss = st.session_state
    return CMATCH.PlanContext(
        clips_per_block=float(ss.get("_avg_clips_per_block", 1.0)),
        autofill_v2=bool(ss.get("autofill_v2", True)),
        reserved_roles=frozenset(ss.get("_plan_reserved_roles") or ()),
        locked_roles=frozenset(ss.get("_plan_locked_roles") or ()),
        intro_outro_exclusive=bool(ss.get("intro_outro_exclusive")),
        story_in_use=bool(ss.get("_plan_story_in_use")),
        clip_count_skip=tuple(ss.get("clip_count_skip") or ()),
        time_ranges=effective_time_ranges, duration_of=get_video_duration,
        curves_of=load_motion_curves_for_video, catalogue_dir=CATALOGUE_DIR,
        why_store=lambda: st.session_state.setdefault(WHY_KEY, {}), role_chips=role_chips,
    )


def plan_clips_per_block() -> float:
    """Average clips per block in the current plan (set where the clip counts are worked out)."""
    return max(1.0, float(st.session_state.get("_avg_clips_per_block", 1.0)))


def soft_reserve(items: list, role, min_keep: int = 1) -> list:
    return CMATCH.soft_reserve(items, role, min_keep, ctx=plan_context())


def role_locked_keys(queues: dict, role_filter) -> set:
    return CMATCH.role_locked_keys(queues, role_filter, ctx=plan_context())


def excluded_for_block(queues: dict, global_excluded: set, role_filter) -> set:
    return CMATCH.excluded_for_block(queues, global_excluded, role_filter, ctx=plan_context())


def find_shape_candidates(*args, **kwargs):
    """core.matching.find_shape_candidates with this plan's settings."""
    return CMATCH.find_shape_candidates(*args, ctx=plan_context(), **kwargs)


def compute_skip_budget(available_windows: int, remaining_blocks: int, weight_position: float,
                        n_videos: int = 1) -> int:
    """core.picking.compute_skip_budget with this plan's clips per block."""
    return CPICK.compute_skip_budget(available_windows, remaining_blocks, weight_position, n_videos,
                                     clips_per_block=plan_clips_per_block())


def compute_position_targets(current_block: int, n_blocks: int, confirmed: dict, queues: dict,
                             videos_per_block: float = None) -> dict:
    """core.picking.compute_position_targets with this project's time ranges."""
    return CPICK.compute_position_targets(current_block, n_blocks, confirmed, queues, videos_per_block,
                                          ranges=effective_time_ranges(), duration_of=get_video_duration)


def rank_candidates_weighted(*args, **kwargs) -> list:
    """core.matching.rank_candidates_weighted with this plan's settings."""
    return CMATCH.rank_candidates_weighted(*args, ctx=plan_context(), **kwargs)


def get_available_clips_for_video(*args, **kwargs) -> list:
    """core.matching.get_available_clips_for_video with this plan's settings."""
    return CMATCH.get_available_clips_for_video(*args, ctx=plan_context(), **kwargs)


def fill_slot(*args, **kwargs) -> tuple:
    """core.matching.fill_slot with this plan's settings."""
    return CMATCH.fill_slot(*args, ctx=plan_context(), **kwargs)


def match_scenes_to_track(*args, **kwargs) -> tuple:
    """core.matching.match_scenes_to_track (Auto mode) with this plan's settings."""
    return CMATCH.match_scenes_to_track(*args, ctx=plan_context(), **kwargs)


WHY_KEY = "autofill_why"


def flash_repeat_fallback(blocks: list) -> None:
    """Queue the fallback warning to show after the rerun that follows an
    Auto-fill. blocks: 1-based block numbers where a repeat had to be used."""
    if blocks:
        st.session_state["repeat_fallback_flash"] = sorted(set(blocks))


def repeat_fallback_message(blocks: list) -> str:
    nums = ", ".join(str(b) for b in blocks[:15]) + (f" (+{len(blocks) - 15} more)" if len(blocks) > 15 else "")
    return (f"🔁 'Avoid repeating previous block' couldn't be kept in block{'s' if len(blocks) != 1 else ''} "
            f"{nums} — no other video had footage for {'those spots' if len(blocks) != 1 else 'that spot'}, "
            f"so a video from the block before was reused. Add videos or widen time ranges/tags to avoid it.")


def show_repeat_fallback_flash() -> None:
    blocks = st.session_state.pop("repeat_fallback_flash", None)
    if blocks:
        msg = repeat_fallback_message(blocks)
        st.warning(msg)
        if hasattr(st, "toast"):
            st.toast(msg, icon="🔁")


@fragment
def render_choreography_block(
    seg_idx: int, seg: dict, selected_videos: list, max_chor_videos: int,
    rec_count: int, queues: dict, sequential_mode: bool, global_excluded: set,
    catalogues: dict, prev_block_videos: set = None, video_stats: dict = None,
    role_filter: str = None, weighting: dict = None, track: dict = None,
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

    role_filter: a role id ("intro", "slow", "action", "outro" or your own) when this is the
    first/last block and the Prefer Intro/Outro sidebar toggle is on —
    narrows each column's dropdown to that video's tagged clips, when it
    has any (see get_available_clips_for_video for the opportunistic
    fallback). None for every other block.

    weighting: {"segments", "confirmed", "weight_shape", "weight_random",
    "avoid_repeat", "weight_spread", "weight_motion", "seed"} — when
    given, the "⭐ Auto pick" preview shown for a column left on Auto uses
    rank_candidates_weighted instead of plain closest-motion-match, matching
    exactly what Confirm & Next will actually do. None falls back to plain
    closest-motion-match.

    Overrides are stored in session_state["chor_overrides"][seg_idx][video_id]
    as (scene_id, offset_sec).  Pick state lives in
    session_state["chor_pick_{seg_idx}_{video_id}"] as a bool, same pattern
    as adv_pick_ in Manual step-through."""

    memory_check(f"Choreography block {seg_idx + 1}")
    autosave_now()
    prev_block_videos = prev_block_videos or set()
    video_stats = video_stats or {}
    block_duration = seg["end"] - seg["start"]
    seg_target = seg.get("intensity", seg["energy"])
    _reach = ((lambda n: compute_skip_budget(n, len(weighting["segments"]) - seg_idx,
                                             weighting.get("weight_position", 0.0), len(queues)))
              if weighting else (lambda n: 0))
    _clips_by_vid = {vid: get_available_clips_for_video(queues, vid, block_duration, sequential_mode,
                                                         global_excluded, role_filter=role_filter,
                                                         role_reach=_reach)
                     for vid in selected_videos}
    eligible = [vid for vid in selected_videos if _clips_by_vid[vid]]
    shown = eligible[:max_chor_videos]
    hidden = len(eligible) - len(shown)

    if not shown:
        st.info("No source video has a clip long enough for this block's duration. "
                "Try adjusting segmentation or video selection.")
        return
    if hidden:
        st.caption(f"{hidden} more eligible video(s) not shown — increase 'Max videos per block' in the sidebar.")
    if role_filter:
        st.caption(f"🎭 Dropdowns below are narrowed to each video's **{SR.label(role_filter)}** clips, "
                   f"where it has any" + (" within its in-order reach" if sequential_mode else "")
                   + " — videos with none show their normal full availability.")

    ticked_count = sum(1 for vid in shown if st.session_state.get(f"chor_pick_{seg_idx}_{vid}", False))
    st.caption(
        f"Recommended simultaneous clips for this block: **{rec_count}** (from your split-screen settings) "
        f"— this is a guide only; tick any number of columns below.  Currently ticked: **{ticked_count}**."
    )

    overrides = st.session_state.setdefault("chor_overrides", {}).setdefault(seg_idx, {})
    chor_cols = st.columns(len(shown))
    highlight_keys = []
    _show_clip = {}  # video_id -> (clip shown, all its clips, index) for the preview panel

    for col, video_id in zip(chor_cols, shown):
        clips = _clips_by_vid[video_id]
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
                if role_filter and clips and not any(has_role(c, role_filter) for c in clips):
                    st.caption(f"⚠️ no {SR.label(role_filter)} clips here — showing full availability")

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
                    # everything before it. compute_skip_budget (weight F-controlled)
                    # allows a FEW windows ahead when this video has spare footage
                    # relative to how many blocks are left, 0 extra when it doesn't.
                    if sequential_mode and weighting:
                        _remaining_blocks = len(weighting["segments"]) - seg_idx
                        _skip = compute_skip_budget(len(clips), _remaining_blocks,
                                                    weighting.get("weight_position", 0.0), len(queues))
                        auto_candidates = clips[:_skip + 1]
                    elif sequential_mode:
                        auto_candidates = clips[:1]
                    else:
                        auto_candidates = clips
                    auto_candidates = soft_reserve(auto_candidates, role_filter)
                    if weighting:
                        ranked = rank_candidates_weighted(
                            auto_candidates, seg_idx, weighting["segments"], weighting["confirmed"], queues,
                            prev_block_videos, weighting["weight_shape"], weighting["weight_random"],
                            weighting["avoid_repeat"], weighting["weight_spread"],
                            weighting["weight_motion"], weighting["seed"],
                            weight_position=weighting.get("weight_position", 0.0),
                            sequential=sequential_mode, videos_per_block=rec_count,
                        )
                        display_clip = ranked[0]
                    else:
                        display_clip = min(auto_candidates, key=lambda c: abs(c["motion_norm"] - seg_target))

                if display_clip is not None:
                    clip_abs_start = display_clip["scene_start_sec"] + display_clip["offset_sec"]
                    nearest_thumb = get_nearest_thumbnail(catalogues, video_id, clip_abs_start)
                    if nearest_thumb is not None:
                        safe_image(nearest_thumb, width=180)
                    elif display_clip.get("thumbnail") and Path(display_clip["thumbnail"]).exists():
                        safe_image(display_clip["thumbnail"], width=180)

                # Which trimmed clip (block-length window) this column uses: ⏪ ◀ ▶ ⏩ step
                # back/forward 5 or 1 through this video's trimmed clips, the slider jumps
                # anywhere in the list, ↺ goes back to the automatic pick, 🎵 Preview plays
                # it with the music.
                sel_idx = _clip_index(clips, display_clip)
                _show_clip[video_id] = (display_clip, clips, sel_idx)
                _at_start = sel_idx is None or sel_idx <= 0
                _at_end = sel_idx is None or sel_idx >= len(clips) - 1
                nav = st.columns(5)
                for _col, (_lbl, _step, _off, _tip) in zip(
                        [nav[0], nav[1], nav[3], nav[4]],
                        [("⏪", -CHOR_SKIP, _at_start, f"Back {CHOR_SKIP} trimmed clips"),
                         ("◀", -1, _at_start, "Previous trimmed clip"),
                         ("▶", +1, _at_end, "Next trimmed clip"),
                         ("⏩", +CHOR_SKIP, _at_end, f"Forward {CHOR_SKIP} trimmed clips")]):
                    with _col:
                        if st.button(_lbl, key=f"chor_step{_step:+d}_{seg_idx}_{video_id}",
                                     use_container_width=True, disabled=_off, help=_tip):
                            _chor_step(overrides, video_id, clips, sel_idx, _step)
                            rerun_fragment()
                with nav[2]:
                    if st.button("↺", key=f"chor_auto_{seg_idx}_{video_id}", use_container_width=True,
                                 disabled=current_key is None, help="Back to the automatic pick"):
                        overrides.pop(video_id, None)
                        rerun_fragment()
                if sel_idx is not None and len(clips) > 1:
                    # Kept in step with ◀ ▶ / Auto: set to the current position before it's drawn;
                    # dragging it selects that clip (on_change runs before the rerun).
                    _slider_key = f"chor_slider_{seg_idx}_{video_id}"
                    st.session_state[_slider_key] = sel_idx + 1
                    st.slider("Trimmed clip", 1, len(clips), key=_slider_key, label_visibility="collapsed",
                              on_change=_chor_slider_changed, args=(overrides, video_id, clips, _slider_key),
                              help="Drag to skip through this video's trimmed clips")

                if display_clip is not None:
                    auto_tag = "⭐ Auto pick — " if current_key is None else ""
                    role_suffix = f"  {role_chips(display_clip)}" if display_clip.get("roles") else ""
                    pos = f"  ·  {sel_idx + 1}/{len(clips)}" if sel_idx is not None else ""
                    st.caption(
                        f"{auto_tag}Scene {display_clip['scene_id']}  "
                        f"{format_mmss(display_clip['scene_start_sec'] + display_clip['offset_sec'])}  "
                        f"motion {display_clip['motion_norm']:.2f}{role_suffix}{pos}"
                    )

                _is_open = st.session_state.get("chor_preview") == (seg_idx, video_id)
                if st.button("⏹ Close preview" if _is_open else "🎵 Preview with music",
                             key=f"chor_prevw_{seg_idx}_{video_id}", use_container_width=True,
                             disabled=display_clip is None):
                    st.session_state["chor_preview"] = None if _is_open else (seg_idx, video_id)
                    rerun_fragment()

                st.checkbox("Use this clip", key=pick_key)

    if highlight_keys:
        css = "\n".join(f'div[class*="st-key-{k}"] {{ border: 3px solid #1c6fea !important; }}' for k in highlight_keys)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    target = st.session_state.get("chor_preview")
    if target and target[0] == seg_idx and target[1] in _show_clip:
        _render_chor_preview_panel(seg_idx, seg, target[1], _show_clip[target[1]], overrides,
                                   catalogues, track, f"chor_pick_{seg_idx}_{target[1]}")


def _clip_index(clips: list, clip) -> int | None:
    """Position of `clip` in this video's list of trimmed clips (matched by scene and offset)."""
    if clip is None:
        return None
    return next((i for i, c in enumerate(clips)
                 if c["scene_id"] == clip["scene_id"] and abs(c["offset_sec"] - clip["offset_sec"]) < 0.1), None)


CHOR_SKIP = 5   # trimmed clips skipped by ⏪ / ⏩ in Choreography


def _chor_step(overrides: dict, video_id: str, clips: list, idx, step: int) -> None:
    """Move this column's trimmed clip by `step` (±1 or ±CHOR_SKIP), stopping at
    the first / last clip rather than refusing a jump that would overshoot."""
    if idx is None or not clips:
        return
    j = max(0, min(len(clips) - 1, idx + step))
    if j != idx:
        overrides[video_id] = (clips[j]["scene_id"], clips[j]["offset_sec"])


def _chor_slider_changed(overrides: dict, video_id: str, clips: list, slider_key: str) -> None:
    """Choreography card slider moved: select that trimmed clip (1-based position)."""
    j = int(st.session_state.get(slider_key, 1)) - 1
    if 0 <= j < len(clips):
        overrides[video_id] = (clips[j]["scene_id"], clips[j]["offset_sec"])


def _local_media(raw_source: str, prefer: Path = None):
    """(local path, None) for a media file, or (None, raw_source) if it isn't on this machine
    (or can't be checked — e.g. /root isn't readable by this user)."""
    def _there(p):
        try:
            return p.exists()
        except OSError:
            return False
    if prefer is not None and _there(Path(prefer)):
        return Path(prefer), None
    local = Path(colab_to_local(raw_source)) if raw_source else None
    if local is not None and _there(local):
        return local, None
    return None, raw_source


def _render_chor_preview_panel(seg_idx: int, seg: dict, video_id: str, shown: tuple, overrides: dict,
                               catalogues: dict, track: dict, pick_key: str) -> None:
    """🎵 Focus panel: the column's current clip playing with this block's music
    (a synced snippet made on the server), ◀ ▶ to step through this video's
    trimmed clips, and ✔ Use this clip."""
    clip, clips, idx = shown
    duration = seg["end"] - seg["start"]
    with st.container(border=True):
        st.markdown(f"**🎵 `{video_id}` with the music for block {seg_idx + 1}** "
                    f"({seg['start']:.1f}s – {seg['end']:.1f}s)")
        video_path, video_missing = _local_media(catalogues[video_id].get("source_path", ""),
                                                 prefer=proxy_path(video_id))
        audio_path, audio_missing = _local_media((track or {}).get("source_path", ""))
        missing = [m for m in (video_missing, audio_missing) if m]
        if missing:
            st.warning("Needs " + " and ".join(Path(colab_to_local(m)).name for m in missing)
                       + " on the server first.")
            if _DRIVE_SYNC_AVAILABLE and all("MyDrive/" in m for m in missing):
                if st.button("⬇️ Download from Google Drive", key=f"chor_prev_dl_{seg_idx}_{video_id}"):
                    for m in missing:
                        bar = st.progress(0.0, text=f"Downloading {Path(colab_to_local(m)).name}…")
                        err = download_source_video(m.split("MyDrive/", 1)[1], Path(colab_to_local(m)),
                                                    progress_callback=lambda p, b=bar: b.progress(min(p, 1.0)))
                        bar.empty()
                        if err:
                            st.error(f"Download failed: {err}")
                            return
                    rerun_fragment()
        elif clip is not None:
            with st.spinner("Making the preview…"):
                snippet, err = preview_snippets.make_synced_snippet(
                    PREVIEW_DIR / "snippets", video_path,
                    clip["scene_start_sec"] + clip["offset_sec"], duration,
                    audio_path, seg["start"],
                )
            if err:
                st.error(f"Couldn't make the preview: {err}")
            else:
                _video_kwargs = {"loop": True} if "loop" in inspect.signature(st.video).parameters else {}
                st.video(str(snippet), **_video_kwargs)

        pos = f"{idx + 1} of {len(clips)}" if idx is not None else "—"
        if clip is not None:
            st.caption(f"Scene {clip['scene_id']}  ·  "
                       f"{format_mmss(clip['scene_start_sec'] + clip['offset_sec'])}  ·  "
                       f"motion {clip['motion_norm']:.2f}  ·  trimmed clip {pos}"
                       + (f"  ·  {', '.join(clip.get('tags', []))}" if clip.get("tags") else ""))
        cols = st.columns([1, 1, 1, 1, 2, 1])
        _at_start = idx is None or idx <= 0
        _at_end = idx is None or idx >= len(clips) - 1
        for _col, (_lbl, _step, _off) in zip(cols[:4], [(f"⏪ {CHOR_SKIP}", -CHOR_SKIP, _at_start),
                                                       ("◀ Previous", -1, _at_start),
                                                       ("Next ▶", +1, _at_end),
                                                       (f"{CHOR_SKIP} ⏩", +CHOR_SKIP, _at_end)]):
            with _col:
                if st.button(_lbl, key=f"chor_p{_step:+d}_{seg_idx}_{video_id}", use_container_width=True,
                             disabled=_off):
                    _chor_step(overrides, video_id, clips, idx, _step)
                    rerun_fragment()
        with cols[4]:
            def _use_and_close():
                if clip is not None and video_id not in overrides:
                    overrides[video_id] = (clip["scene_id"], clip["offset_sec"])  # lock in what you heard
                st.session_state[pick_key] = True
                st.session_state["chor_preview"] = None
            st.button("✔ Use this clip", key=f"chor_puse_{seg_idx}_{video_id}", type="primary",
                      use_container_width=True, on_click=_use_and_close, disabled=clip is None)
        with cols[5]:
            if st.button("Close", key=f"chor_pclose_{seg_idx}_{video_id}", use_container_width=True):
                st.session_state["chor_preview"] = None
                rerun_fragment()


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
                          autofill_weights: dict = None, catalogues: dict = None,
                          similar_ctx: dict = None) -> None:
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
    "weight_random", "avoid_repeat", "weight_spread", "weight_motion", "seed"} —
    everything rank_candidates_weighted needs. The candidate GRID below
    always stays sorted by raw shape score regardless; only the auto-fill
    button's own picks use the weighted ranking."""
    memory_check(f"Manual step-through block {current_block + 1}")
    autosave_now()
    prev_block_videos = prev_block_videos or set()
    video_stats = video_stats or {}
    autofill_weights = autofill_weights or {}

    if candidates and recommended_count > 0:
        if st.button(f"⚡ Auto-fill top {min(recommended_count, len(candidates))}",
                     key=f"adv_autofill_{current_block}",
                     help="Picks the best candidates by the Auto-fill weighting above (shape match, "
                          "motion intensity, randomness, avoiding repeats, spreading usage) — not just "
                          "the highest raw score. Unticks anything else. A starting point; adjust freely afterward."):
            _brole = autofill_weights.get("block_role")
            _pool = soft_reserve(prefer_role(candidates, _brole), _brole)
            ranked = rank_candidates_weighted(
                _pool, current_block, autofill_weights["segments"], autofill_weights["confirmed"],
                autofill_weights["queues"], prev_block_videos, autofill_weights["weight_shape"],
                autofill_weights["weight_random"], autofill_weights["avoid_repeat"],
                autofill_weights["weight_spread"], autofill_weights["weight_motion"],
                autofill_weights["seed"],
                weight_position=autofill_weights.get("weight_position", 0.0),
                sequential=autofill_weights.get("sequential", False),
                videos_per_block=recommended_count,
            )
            chosen = select_autofill_picks(ranked, recommended_count,
                                           autofill_weights.get("allow_same_video", False))
            chosen_keys = {cand_key(c) for c in chosen}
            for c in candidates:
                st.session_state[f"adv_pick_{current_block}_{cand_key(c)}"] = cand_key(c) in chosen_keys
            if repeat_fallback_videos(chosen, prev_block_videos, autofill_weights["avoid_repeat"]):
                flash_repeat_fallback([current_block + 1])
            st.rerun()

    if any(c.get("similar") for c in candidates):
        if st.button("✖ Clear ✨ results", key=f"adv_sim_clear_{current_block}",
                     help="Remove the ✨ More-like-this clips from the grid (ticked ones are unticked too)."):
            for c in candidates:
                if c.get("similar"):
                    st.session_state.pop(f"adv_pick_{current_block}_{cand_key(c)}", None)
            st.session_state.pop(f"adv_similar_{current_block}", None)
            rerun_full()

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
            box_key = f"cand_box_{current_block}_{cand_key(c)}"
            box = st.container(border=True, key=box_key) if _SUPPORTS_KEYED_CONTAINER else st.container(border=True)
            if is_repeat and _SUPPORTS_KEYED_CONTAINER:
                highlight_keys.append(box_key)
            with box:
                if is_repeat:
                    st.caption("🔵 same video as previous block")
                if c.get("similar"):
                    st.caption(f"✨ {c['similar']}% like {c['similar_to']} — {c['similar_reason']}")
                if c.get("roles"):
                    st.caption(f"🎭 {role_chips(c)}")
                clip_start = c.get("scene_start_sec", 0.0) + c["window_offset_sec"]
                nearest_thumb = get_nearest_thumbnail(catalogues or {}, c["video_id"], clip_start) if catalogues else None
                if nearest_thumb is not None:
                    safe_image(nearest_thumb, width=200, caption="at the clip's start")
                elif c.get("thumbnail") and Path(c["thumbnail"]).exists():
                    safe_image(c["thumbnail"], width=200, caption="whole scene (run the timeline-thumbnail "
                             "backfill for a preview of just this clip)")
                _score_rank = next((r + 1 for r, sc in enumerate(candidates) if sc["video_id"] == c["video_id"] and sc["scene_id"] == c["scene_id"]), "?")
                st.caption(f"`{c['video_id']}` #{c['scene_id']}  ·  rank #{_score_rank}  ·  shape {c['score']:.2f}  ·  motion {c['motion_norm']:.2f}")
                st.caption(f"trimmed to {c['window_offset_sec']:.1f}s–"
                          f"{c['window_offset_sec'] + c['window_duration_sec']:.1f}s in the scene")
                st.caption(f"used so far: {stats.get('used', 0.0):.0f}s  ·  remaining: {stats.get('remaining', 0.0):.0f}s")
                st.checkbox("Use this clip", key=f"adv_pick_{current_block}_{cand_key(c)}",
                           value=(key_id in already_keys) and not c.get("similar"))
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if st.button("🔍 View fit", key=f"adv_view_{current_block}_{cand_key(c)}"):
                        st.session_state["adv_viewing"] = cand_key(c)
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
                    if st.button("⏮️ Skip back", key=f"adv_skipback_{current_block}_{cand_key(c)}",
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
                    if st.button("⏭️ Skip ahead", key=f"adv_skip_{current_block}_{cand_key(c)}",
                               help="Give up this video's current footage without using it, so a later, "
                                    "possibly better-matching part of the same video becomes reachable."):
                        st.session_state["adv_skips"].setdefault(current_block, []).append(
                            {"video_id": c["video_id"], "scene_id": c["scene_id"]}
                        )
                        st.session_state["adv_viewing"] = None
                        rerun_full()
                if similar_ctx and st.button("✨ More like this", key=f"adv_more_{current_block}_{cand_key(c)}",
                                             use_container_width=True,
                                             help="Find clips like this one in your selected videos' still-available "
                                                  "footage — they're added to the grid marked ✨."):
                    found = similar_ctx["find"](c)
                    if found is None:
                        st.warning("This clip has no frame-by-frame motion data to compare with.")
                    else:
                        st.session_state[f"adv_similar_{current_block}"] = {
                            "key": similar_ctx["search_key"], "results": found}
                        if not found:
                            st.info("Nothing similar is available for this block.")
                        else:
                            rerun_full()

    if highlight_keys:
        css = "\n".join(f'div[class*="st-key-{k}"] {{ border: 3px solid #1c6fea !important; }}' for k in highlight_keys)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Default to showing the top candidate's fit immediately — no click needed
    # for the common case of just checking the best match. Explicitly clicking
    # a different candidate still switches the view as before.
    # adv_viewing holds the candidate's key (an index used to point into a
    # differently-sorted list, so View fit could show the wrong clip).
    viewing = st.session_state.get("adv_viewing")
    vc = next((x for x in candidates if cand_key(x) == viewing), None) if viewing is not None else None
    if vc is None and candidates:
        vc = candidates[0]
    if vc is not None:
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


# ---------------------------------------------------------------------------
# ✨ More like this — similar videos, similar clips, swap for similar
# (similarity.py does the scoring: tags + movement, no AI model)
# ---------------------------------------------------------------------------


def _clip_curve(video_id: str, scene_id: int, offset_sec: float, duration_sec: float):
    """(values, fps) of a clip's slice of its scene's motion curve, or (None, None)."""
    curve = load_motion_curves_for_video(video_id).get(scene_id)
    if not curve or curve.get("values") is None or len(curve["values"]) == 0:
        return None, None
    fps = float(curve["fps"])
    vals = np.asarray(curve["values"], dtype=float)
    a = int(round(offset_sec * fps))
    b = a + max(1, int(round(duration_sec * fps)))
    return np.array(vals[a:b]), fps      # own copy — not tied to the curve cache's file


@shared_cache
@st.cache_resource(show_spinner=False)
def library_tag_idf() -> dict:
    """{tag: weight} across the whole library — computed once, not per search."""
    return SIM.tag_idf(load_all_catalogues(), lambda sc: sc.get("tags", []))


def find_similar_clips(ref: dict, spans: list, duration: float, exclude=None, top: int = 6,
                       max_per_video: int = 2):
    """Clips of exactly `duration` from `spans` (available footage) most like
    ref = {"video_id", "scene_id", "offset", "tags"}. Windows step by half a
    clip (coarser on very large pools, to stay quick); results never overlap
    each other and come at most max_per_video from one video. Returns
    [(similarity, reason, (span, offset, values, fps))], or None if the
    reference clip has no motion curve to compare."""
    ref_vals, ref_fps = _clip_curve(ref["video_id"], ref["scene_id"], ref["offset"], duration)
    if ref_vals is None or ref_vals.size == 0:
        return None
    total = sum(sp["remaining_sec"] for sp in spans)
    step = max(0.5, duration / 2, total / 4000.0)
    pool = []
    for span in spans:
        off = span["offset_sec"]
        end = span["offset_sec"] + span["remaining_sec"]
        while off + duration <= end + 1e-6:
            if not (exclude and exclude(span, off)):
                vals, fps = _clip_curve(span["video_id"], span["scene_id"], off, duration)
                if vals is not None and vals.size:
                    pool.append((SIM.clip_descriptor(vals, fps, span.get("tags", [])), (span, off, vals, fps)))
            off += step
    idf = library_tag_idf()
    ranked = SIM.rank_similar_clips(SIM.clip_descriptor(ref_vals, ref_fps, ref.get("tags", [])), pool, idf,
                                    top=len(pool))
    # Keep the results varied: no two overlapping clips, and at most
    # max_per_video from any one video — otherwise near-copies of the same
    # stretch of footage crowd everything else out.
    chosen, per_video = [], {}
    for sim, reason, payload in ranked:
        span, off = payload[0], payload[1]
        vid = span["video_id"]
        if per_video.get(vid, 0) >= max_per_video:
            continue
        if any(o[0]["video_id"] == vid and o[0]["scene_id"] == span["scene_id"] and abs(o[1] - off) < duration
               for _, _, o in chosen):
            continue
        chosen.append((sim, reason, payload))
        per_video[vid] = per_video.get(vid, 0) + 1
        if len(chosen) >= top:
            break
    return chosen


def more_like_this_for_grid(c: dict, seg: dict, queues: dict, excluded_here: set, global_excluded: set,
                            sequential: bool, audio_curve, existing: list) -> list | None:
    """✨ for a Manual step-through card: block-length clips from the SELECTED
    videos' still-available footage (same spans the grid searches — sequential
    rule, exclusions and tag filter already applied), most like candidate c.
    Returned in the same shape as shape-search candidates, so View fit,
    ticking and Confirm work on them unchanged."""
    duration = seg["end"] - seg["start"]
    spans = [sp for sp in get_candidate_spans(queues, sequential, global_excluded, min_duration=duration)
             if (sp["video_id"], sp["scene_id"]) not in excluded_here]

    def exclude(span, off):
        if (span["video_id"], span["scene_id"]) == (c["video_id"], c["scene_id"]) \
                and abs(off - c["window_offset_sec"]) < duration:
            return True  # overlaps the reference clip itself
        return any((e["video_id"], e["scene_id"]) == (span["video_id"], span["scene_id"])
                   and abs(e["window_offset_sec"] - off) < 0.25 for e in existing)

    found = find_similar_clips({"video_id": c["video_id"], "scene_id": c["scene_id"],
                                "offset": c["window_offset_sec"], "tags": c.get("tags", [])},
                               spans, duration, exclude, top=6)
    if found is None:
        return None
    out = []
    for sim, reason, (span, off, vals, fps) in found:
        out.append({
            "video_id": span["video_id"], "scene_id": span["scene_id"],
            "scene_start_sec": span["scene_start_sec"],
            "window_offset_sec": round(off, 3), "window_duration_sec": round(duration, 3),
            "score": round(shape_score(vals, audio_curve), 3),
            "motion_norm": span["motion_norm"], "curve_slice": vals, "fps": fps,
            "tags": span["tags"], "thumbnail": span["thumbnail"],
            "intro_candidate": span.get("intro_candidate", False),
            "outro_candidate": span.get("outro_candidate", False),
            "roles": span.get("roles", []),
            "similar": int(round(100 * sim)), "similar_reason": reason,
            "similar_to": f"{c['video_id']} #{c['scene_id']}",
            "_key": f"{span['video_id']}_{span['scene_id']}_s{int(round(off * 10))}",
        })
    return out


def compute_similar_videos(ref_id: str, tag_filter: tuple, time_ranges: dict) -> list:
    """✨ Similar videos: [(similarity, reason, video_id)] — top 5 from the whole
    library, each judged on its usable footage (range + tag filter applied)."""
    catalogues = load_all_catalogues()
    wanted = {t.lower() for t in tag_filter}
    videos = {}
    for vid, cat in catalogues.items():
        summary = VR.load_summary(CATALOGUE_DIR, vid)
        if summary is None:
            continue
        rng = time_ranges.get(vid)
        feats = VR.video_features(summary, {sc["scene_id"]: sc for sc in cat["scenes"]}, rng, tuple(tag_filter))
        if not feats:
            continue
        tag_seconds = {}
        for sc in cat["scenes"]:
            if sc.get("excluded"):
                continue
            a, b = _tc_to_seconds(sc["start_tc"]), _tc_to_seconds(sc["end_tc"])
            a, b = _overlap_with_range(a, b, rng)
            if a is None:
                continue
            tags = sc.get("tags", [])
            if wanted and not (wanted & {t.lower() for t in tags}):
                continue
            for t in tags:
                tag_seconds[t] = tag_seconds.get(t, 0.0) + (b - a)
        videos[vid] = {"tag_seconds": tag_seconds, "level": feats["level"],
                       "dynamics": feats["dynamics"], "punch": feats["punch"]}
    idf = SIM.tag_idf(catalogues, lambda sc: sc.get("tags", []))
    return SIM.rank_similar_videos(ref_id, videos, idf, top=5)


def _add_video_to_selection(video_id: str) -> None:
    """➕ from the ✨ Similar videos panel — runs as a button callback (before
    widgets redraw), so it can tick the video and add it to the extra list."""
    if video_id not in st.session_state.get("recommended_ids", []):
        extras = list(st.session_state.get("extra_videos", []))
        if video_id not in extras:
            extras.append(video_id)
        st.session_state["extra_videos"] = extras
    st.session_state[f"select_video_{video_id}"] = True


def replacement_pool(confirmed_map: dict, seg_idx: int, pick_idx: int, n_blocks: int) -> dict:
    """Footage still available for ONE placed clip if every other confirmed clip
    stays where it is: earlier blocks are replayed as usual, every other clip
    is taken out, and in sequential mode each video is also cut off before its
    clip in any later block — so a replacement keeps the order intact."""
    q = build_footage_queues(tuple(selected_videos), tuple(tag_filter), effective_time_ranges())
    for i in range(seg_idx):
        for p in confirmed_map.get(i, []):
            try:
                carve_span(q, p["video_id"], p["scene_id"], p["offset_into_scene_sec"],
                           p["clip_duration_sec"], sequential_mode)
            except ValueError:
                pass
    earliest_later = {}
    for i in range(seg_idx, n_blocks):
        for j, p in enumerate(confirmed_map.get(i, [])):
            if i == seg_idx and j == pick_idx:
                continue
            try:
                carve_span(q, p["video_id"], p["scene_id"], p["offset_into_scene_sec"],
                           p["clip_duration_sec"], False)
            except ValueError:
                pass
            if i > seg_idx:
                earliest_later[p["video_id"]] = min(earliest_later.get(p["video_id"], float("inf")),
                                                    p["clip_start_sec"])
    if sequential_mode:
        for vid, t in earliest_later.items():
            limit_queue_before(q, vid, t)
    return q


def _swap_candidates(mode: str, confirmed_map: dict, seg_idx: int, pick_idx: int, n_blocks: int) -> list | None:
    """Top 3 replacements for one placed clip, from the selected videos'
    available pool (see replacement_pool). Videos already in this block are
    left out (Choreography always; Manual unless 'Allow multiple clips from the
    same video' is on), apart from the clip's own video."""
    pick = confirmed_map[seg_idx][pick_idx]
    duration = pick["clip_duration_sec"]
    q = replacement_pool(confirmed_map, seg_idx, pick_idx, n_blocks)
    others = {p["video_id"] for j, p in enumerate(confirmed_map.get(seg_idx, [])) if j != pick_idx}
    if mode == "manual" and allow_same_video:
        others = set()
    others.discard(pick["video_id"])
    gex = st.session_state.get("global_excluded_scenes", set())
    spans = [sp for vid, sps in q.items() if vid not in others for sp in sps
             if sp["remaining_sec"] >= duration - 1e-6 and (vid, sp["scene_id"]) not in gex]

    def exclude(span, off):
        return (span["video_id"], span["scene_id"]) == (pick["video_id"], pick["scene_id"]) \
            and abs(off - pick["offset_into_scene_sec"]) < duration

    return find_similar_clips({"video_id": pick["video_id"], "scene_id": pick["scene_id"],
                               "offset": pick["offset_into_scene_sec"], "tags": pick.get("tags", [])},
                              spans, duration, exclude, top=3)


def render_swap_for_similar(mode: str, confirmed_map: dict, seg_idx: int, n_blocks: int) -> None:
    """🔁 Swap for similar panel under a review-list block, when one of its
    clips was picked. mode: "manual" or "chor"."""
    target = st.session_state.get("swap_target")
    if not target or target[0] != mode or target[1] != seg_idx:
        return
    pick_idx = target[2]
    picks = confirmed_map.get(seg_idx, [])
    if pick_idx >= len(picks):
        st.session_state.pop("swap_target", None)
        return
    pick = picks[pick_idx]
    with st.container(border=True):
        st.markdown(f"**🔁 Swap `{pick['video_id']}` #{pick['scene_id']} for something similar** "
                    f"— from your selected videos' available footage; every other block stays as it is.")
        cache_key = (mode, seg_idx, pick_idx, pick["video_id"], pick["scene_id"], pick["offset_into_scene_sec"])
        cached = st.session_state.get("swap_results")
        if not cached or cached["key"] != cache_key:
            with st.spinner("Finding similar clips…"):
                found = _swap_candidates(mode, confirmed_map, seg_idx, pick_idx, n_blocks)
            cached = {"key": cache_key, "found": found}
            st.session_state["swap_results"] = cached
        found = cached["found"]
        if found is None:
            st.warning("This clip has no frame-by-frame motion data, so there's nothing to compare it with. "
                       "Run the motion-curve backfill for its video.")
        elif not found:
            st.info("No other footage of this length is available for this block.")
        else:
            cols = st.columns(len(found))
            for k, (col, (sim, reason, (span, off, _vals, _fps))) in enumerate(zip(cols, found)):
                with col:
                    thumb = get_nearest_thumbnail(catalogues, span["video_id"], span["scene_start_sec"] + off)
                    if thumb is not None:
                        safe_image(thumb, width=180)
                    st.caption(f"`{span['video_id']}` #{span['scene_id']} at "
                               f"{format_mmss(span['scene_start_sec'] + off)}  ·  **{int(round(100 * sim))}% similar**")
                    st.caption(reason)
                    if st.button("Use this", key=f"swap_use_{mode}_{seg_idx}_{k}", type="primary"):
                        q = replacement_pool(confirmed_map, seg_idx, pick_idx, n_blocks)
                        try:
                            new_pick = carve_span(q, span["video_id"], span["scene_id"], off,
                                                  pick["clip_duration_sec"], sequential_mode)
                        except ValueError as e:
                            st.error(f"Couldn't use that clip: {e}")
                        else:
                            confirmed_map[seg_idx][pick_idx] = new_pick
                            if mode == "chor":
                                ov = st.session_state.setdefault("chor_overrides", {}).setdefault(seg_idx, {})
                                ov.pop(pick["video_id"], None)
                                ov[new_pick["video_id"]] = (new_pick["scene_id"], off)
                                if new_pick["video_id"] != pick["video_id"]:
                                    st.session_state[f"chor_pick_{seg_idx}_{pick['video_id']}"] = False
                                st.session_state[f"chor_pick_{seg_idx}_{new_pick['video_id']}"] = True
                            st.session_state.pop("swap_target", None)
                            st.session_state.pop("swap_results", None)
                            st.rerun()
        if st.button("Cancel", key=f"swap_cancel_{mode}_{seg_idx}"):
            st.session_state.pop("swap_target", None)
            st.session_state.pop("swap_results", None)
            st.rerun()


def blocks_to_show(label: str, key: str, n_blocks: int) -> int:
    """'Blocks to show' slider for the review lists: its range is 1…n_blocks and
    it defaults to every block (capped at 50 for long tracks). When the number
    of blocks changes (new track, different segmentation), it resets to that
    default instead of keeping a value from the old range."""
    if n_blocks <= 1:
        return n_blocks
    default = min(n_blocks, 50)
    if st.session_state.get(f"{key}__n") != n_blocks:
        st.session_state.pop(key, None)
        st.session_state[f"{key}__n"] = n_blocks
    return st.slider(label, 1, n_blocks, default, key=key)


@shared_cache
@st.cache_resource(ttl=1800, show_spinner=False)   # shared, not copied per call — read-only
def load_all_audio_tracks() -> dict:
    """track_id -> full track dict, loaded once and cached. SHARED — read it, never change it."""
    tracks = {}
    for f in AUDIO_DIR.glob("*.json"):
        if f.name == "audio_index.json":
            continue
        with open(f) as fh:
            t = json.load(fh)
        tracks[t.get("track_id", f.stem)] = t
    return tracks


def video_role_seconds(time_ranges: dict) -> dict:
    """{video_id: {role id: seconds of usable (not excluded, in range) footage carrying it}}."""
    out = {}
    for vid, cat in load_all_catalogues().items():
        rng = time_ranges.get(vid)
        d = {}
        for sc in cat["scenes"]:
            if sc.get("excluded"):
                continue
            roles = SR.scene_roles(sc)
            if not roles:
                continue
            a, b = _overlap_with_range(_tc_to_seconds(sc["start_tc"]), _tc_to_seconds(sc["end_tc"]), rng)
            if a is None:
                continue
            for r in roles:
                d[r] = d.get(r, 0.0) + (b - a)
        if d:
            out[vid] = d
    return out


def story_role_need(track_id: str) -> dict:
    """{role id: seconds of the song its story map gives that role} ({} without a map)."""
    out = {}
    for r in current_story_map(track_id) or []:
        if r["role"]:
            out[r["role"]] = out.get(r["role"], 0.0) + (r["end"] - r["start"])
    return out


def compute_video_rankings(track: dict, tag_filter: tuple, time_ranges: dict,
                           weights: dict, experimental: bool) -> list[dict]:
    """Match score (0–100) for every video with usable footage, best first — see
    video_ranking.py for the metrics. Each video is compared against the WHOLE
    library (ranges applied, no tag filter), so the tag filter never shifts the
    scale. Per-video motion summaries are cached on disk; any missing ones are
    built here with a progress bar (first run, or after new videos sync in)."""
    catalogues = load_all_catalogues()
    video_ids = sorted(catalogues)
    todo = VR.missing_summaries(CATALOGUE_DIR, video_ids)
    if todo:
        bar = st.progress(0.0, text=f"Analysing motion curves for {len(todo)} video(s) (one-off)…")
        for i, vid in enumerate(todo):
            VR.load_summary(CATALOGUE_DIR, vid)
            bar.progress((i + 1) / len(todo), text=f"Analysing motion curves… {vid}")
        bar.empty()

    feats, reference = {}, {}
    for vid in video_ids:
        summary = VR.load_summary(CATALOGUE_DIR, vid)
        if summary is None:
            continue
        scenes_by_id = {sc["scene_id"]: sc for sc in catalogues[vid]["scenes"]}
        rng = time_ranges.get(vid)
        reference[vid] = VR.video_features(summary, scenes_by_id, rng, ())
        feats[vid] = (VR.video_features(summary, scenes_by_id, rng, tuple(tag_filter))
                      if tag_filter else reference[vid])
    return VR.score_videos(
        track, load_all_audio_tracks(), feats, reference, weights, tuple(tag_filter),
        usage=VR.combined_usage(PROJECTS_DIR, PREVIEW_DIR), experimental=experimental,
        role_feats=video_role_seconds(time_ranges), role_need=story_role_need(track.get("track_id")),
    )


def _ranking_weights() -> dict:
    stored = st.session_state.setdefault("ranking_weights", dict(VR.DEFAULT_WEIGHTS))
    return {m: float(stored.get(m, VR.DEFAULT_WEIGHTS[m])) for m in VR.METRICS}


def render_ranking_weights(tag_filter: list) -> None:
    """⚙️ Ranking weights — one slider per metric. Values live in the plain
    session_state dict "ranking_weights" (saved with projects), so they
    survive this section's widgets being unrendered."""
    weights = _ranking_weights()
    with st.expander("⚙️ Ranking weights — how the top videos are chosen", expanded=False):
        st.caption("Each video gets a 0–100 match score from these metrics. 0 switches a metric off; "
                   "higher makes it count more. The list re-ranks as you change them.")
        experimental = st.toggle("Show experimental metrics (movement tempo, energy arc)",
                                 value=bool(st.session_state.get("ranking_experimental", False)),
                                 key="ranking_experimental_toggle")
        st.session_state["ranking_experimental"] = experimental
        shown = [m for m in VR.METRICS if experimental or m not in VR.EXPERIMENTAL]
        cols = st.columns(3)
        for i, m in enumerate(shown):
            with cols[i % 3]:
                disabled = (m == "relevance" and not tag_filter)
                weights[m] = st.slider(
                    VR.LABELS[m], 0.0, 5.0, value=weights[m], step=0.1, key=f"rank_w_{m}",
                    disabled=disabled,
                    help=VR.HELP[m] + (" (Set a tag filter above to use it.)" if disabled else ""),
                )
        if st.button("Reset to defaults", key="rank_w_reset"):
            weights = dict(VR.DEFAULT_WEIGHTS)
            for m in VR.METRICS:
                st.session_state.pop(f"rank_w_{m}", None)
            st.session_state["ranking_weights"] = weights
            rerun_fragment()
        st.session_state["ranking_weights"] = weights


def format_match_breakdown(match: dict, experimental: bool, tag_filter) -> str:
    short = {"dynamics": "dyn", "punch": "punch", "footage": "footage", "pace": "pace",
             "spread": "spread", "relevance": "tags", "freshness": "fresh", "story": "story",
             "tempo": "tempo", "arc": "arc"}
    parts = []
    for m in VR.METRICS:
        if m in VR.EXPERIMENTAL and not experimental:
            continue
        if m == "relevance" and not tag_filter:
            continue
        if m == "story" and not story_role_need(st.session_state.get("track_id")):
            continue
        v = match["parts"].get(m, 0.0)
        dots = "●" * int(round(v * 4)) + "○" * (4 - int(round(v * 4)))
        label = short[m]
        if m == "tempo" and match.get("tempo_bpm"):
            label += f" ({match['tempo_bpm']:.0f} BPM)"
        parts.append(f"{label} {dots}")
    text = "  ·  ".join(parts)
    if match.get("curve_coverage", 1.0) < 0.5:
        text += "  ·  ⚠️ no motion curves for most scenes (run the motion-curve backfill)"
    return text


def _columns_support_valign() -> bool:
    import inspect
    try:
        return "vertical_alignment" in inspect.signature(st.columns).parameters
    except (TypeError, ValueError):
        return False


_COLUMNS_SUPPORT_VALIGN = _columns_support_valign()

RECOMMENDED_COUNT = 10   # how many best-matching videos are listed by default


# Shared, not copied per call (cache_data handed every thumbnail crop its own full
# copy of the sprite sheet). Only ever cropped, which makes a new small image.
@shared_cache
@st.cache_resource(max_entries=16, ttl=1800, show_spinner=False)
def load_timeline_sprite(sprite_path: str):
    """The whole per-video sprite sheet, loaded and cached once per video —
    cropping a specific tile from it (get_nearest_thumbnail) is cheap and
    done fresh on every call, but re-reading the sprite FILE every time
    would not be."""
    img = Image.open(sprite_path)
    img.load()  # force full read now, so the cached object doesn't hold an open file handle
    return img


def video_list_thumbnail(catalogues: dict, video_id: str, time_range=None):
    """Small preview for the video list: the ⭐ master thumbnail chosen in the
    Reviewer if there is one, otherwise the timeline-sprite tile from the middle
    of the video's usable range (a 120-px crop from an already-cached sprite, so
    it's cheap), falling back to that scene's own thumbnail file. None if
    neither exists."""
    master = master_thumbnail_path(video_id, catalogues.get(video_id, {}).get("scenes", []))
    if master is not None:  # ⭐ chosen in the Reviewer
        return str(master)
    duration = get_video_duration(video_id)
    a, b = time_range or (0.0, duration)
    mid = (a + b) / 2
    tile = get_nearest_thumbnail(catalogues, video_id, mid)
    if tile is not None:
        return tile
    for sc in catalogues.get(video_id, {}).get("scenes", []):
        try:
            inside = _tc_to_seconds(sc["start_tc"]) <= mid <= _tc_to_seconds(sc["end_tc"])
        except (KeyError, ValueError):
            continue
        if inside and sc.get("thumbnail_paths"):
            path = resolve_thumbnail(sc["thumbnail_paths"][0], video_id)
            return str(path) if path.exists() else None
    return None


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


def _usage_signature() -> tuple:
    """Changes whenever a project is saved or a render is made/deleted, so the
    ranking (its freshness metric) is recomputed then — and only then."""
    sig = []
    for d in (PROJECTS_DIR, PREVIEW_DIR):
        try:
            entries = [e for e in os.scandir(d) if e.name.endswith(".json") and not e.name.startswith(".")]
            sig.append((len(entries), max((e.stat().st_mtime for e in entries), default=0.0)))
        except OSError:
            sig.append((0, 0.0))
    return tuple(sig)


def resolve_video_selection(track: dict, tag_filter: tuple, all_video_ids: list, allowed: frozenset = None) -> dict:
    """Which videos are recommended (the top-10 ranking) and which are
    currently selected, read entirely from session_state — called by the
    Video Selection fragment on every one of its own reruns.

    The RANKING itself (video_matches/top_ids/other_options) is cached and
    only recomputed when something it depends on changes: the track, the tag
    filter, any video's effective time range, or the ranking weights.
    Ticking a checkbox or adding something from the dropdown doesn't, so the
    list doesn't redraw on every click.

    SELECTION (who's ticked) is never cached — always computed fresh, since
    it's cheap and must reflect the current checkboxes exactly."""
    _ranges = effective_time_ranges()
    _weights = _ranking_weights()
    _experimental = bool(st.session_state.get("ranking_experimental", False))
    cache_key = (track.get("track_id"), tag_filter,
                 tuple(sorted((v, tuple(r)) for v, r in _ranges.items())),
                 tuple(sorted(_weights.items())), _experimental, _usage_signature(),
                 tuple(sorted(allowed)) if allowed is not None else None,
                 tuple(sorted(story_role_need(track.get("track_id")).items())))
    random_seed = st.session_state.get("random_top_seed")
    cache = st.session_state.get("video_ranking_cache")
    if cache and cache["key"] == cache_key and cache.get("seed") == random_seed:
        ranking = cache["ranking"]
    else:
        if cache and cache["key"] == cache_key:
            video_matches = cache["ranking"]["video_matches"]   # only the seed changed — no re-scoring
        else:
            with _timed("Video ranking", f"{len(_ranges)} ranges"):
                video_matches = compute_video_rankings(track, tag_filter, _ranges, _weights, _experimental)
            if allowed is not None:   # 👤 actor / 🏷️ video tag filter
                video_matches = [v for v in video_matches if v["video_id"] in allowed]
        if random_seed is not None:
            # 🎲 Randomize: RECOMMENDED_COUNT videos drawn purely at random from every video that
            # has usable scenes — the ranking weights play no part. Same seed → same draw.
            top_matches = random.Random(random_seed).sample(video_matches, min(RECOMMENDED_COUNT, len(video_matches)))
        else:
            top_matches = video_matches[:RECOMMENDED_COUNT]
        top_ids = [v["video_id"] for v in top_matches]
        match_by_id = {v["video_id"]: v for v in video_matches}
        rank_order = {v["video_id"]: i for i, v in enumerate(video_matches)}
        other_options = sorted((vid for vid in all_video_ids if vid not in top_ids
                                and (allowed is None or vid in allowed)),
                               key=lambda vid: (rank_order.get(vid, 10**6), vid))
        ranking = {
            "video_matches": video_matches, "top_matches": top_matches, "top_ids": top_ids,
            "match_by_id": match_by_id, "rank_order": rank_order, "other_options": other_options,
        }
        st.session_state["video_ranking_cache"] = {"key": cache_key, "seed": random_seed, "ranking": ranking}

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
    "segmentation_method", "beats_per_bar", "audio_style", "sc_focus", "cut_style", "cut_style_pace",
    "change_window_secs", "change_threshold", "min_segment_sec", "max_segment_sec",
    "react_to_hits", "hit_threshold", "snap_secs",
    "fast_energy_thresh", "fast_max_bars", "fast_cooldown_bars",
    "phrase_lock_bars", "cinematic_bars", "dynamic_priority_override", "override_delta_thresh",
    "override_clear_secs",
    *CS.DEFAULTS.keys(),
    *SM.DEFAULTS.keys(),
]


def segmentation_settings() -> dict:
    """The current segmentation method plus exactly the settings it uses —
    the one place the preview chart, the matching and the export read them from."""
    method = st.session_state.get("segmentation_method", SEG_ADAPTIVE)
    if method not in SEG_METHOD_KEYS:
        method = SEG_ADAPTIVE
    out = {"method": method, "beats_per_bar": int(st.session_state.get("beats_per_bar", 4))}
    for k in SEG_METHOD_KEYS[method]:
        out[k] = st.session_state.get(k, DIAL_DEFAULTS.get(k))
    return out


SEG_LABELS = {
    SEG_SMART: "🧠 Smart — tunes itself to the song (new)",
    SEG_SCORED: "🎯 Best musical moments",
    SEG_ADAPTIVE: "⚡ Energy changes & big hits",
    SEG_RHYTHM: "🎬 Bar styles — cinematic / fast",
}

# "What to cut on" mixes for the Scored method's reason weights.
CUT_FOCUS = {
    "Balanced": {},   # = cut_scoring.DEFAULTS
    "Song structure": {"sc_w_bar": 1.5, "sc_w_phrase": 2.5, "sc_w_section": 4.0, "sc_w_drop": 4.0,
                       "sc_w_hit": 0.5, "sc_w_kick": 0.5, "sc_w_snare": 0.3, "sc_w_harmony": 1.0},
    "Drums & hits": {"sc_w_hit": 2.0, "sc_w_kick": 2.0, "sc_w_snare": 1.5, "sc_w_bar": 0.7,
                     "sc_w_phrase": 1.0, "sc_w_harmony": 0.4},
    "Vocals": {"sc_w_vocal": 3.0, "sc_w_vocal_mid": 2.5, "sc_w_harmony": 1.0},
}


def _mark_audio_custom() -> None:
    """A dial was moved by hand: stop auto-tuning over it."""
    st.session_state["audio_style"] = AUDIO_STYLE_CUSTOM
    st.session_state["cut_style"] = "custom"       # 🎨 no longer exactly the style's recipe
    st.session_state.pop("aim_report", None)     # 🎯 its dial values no longer apply


def _mark_style_custom() -> None:
    """A Smart dial was moved by hand: the 🎨 style no longer describes the settings."""
    st.session_state["cut_style"] = "custom"


def _method_changed() -> None:
    _mark_style_custom()
    _audio_retune_if_paced()


def _beats_per_bar_changed() -> None:
    """A style's recipe depends on the bar length, so it's worked out again; otherwise as before."""
    if st.session_state.get("cut_style") in CSTY.STYLES:
        _apply_cut_style()
    else:
        _audio_retune_if_paced()


def _apply_cut_style() -> None:
    """🎨 Style or its pace changed: set the method and every dial from the style's recipe
    for this track (core/styles.py)."""
    style = st.session_state.get("cut_style", "custom")
    if style not in CSTY.STYLES:
        st.session_state.pop("style_report", None)
        return
    track = load_track(st.session_state["track_id"])
    method, settings, notes = CSTY.style_settings(track, style, int(st.session_state.get("cut_style_pace", 0)),
                                                  int(st.session_state.get("beats_per_bar", 4)))
    st.session_state["segmentation_method"] = method
    for k, v in settings.items():
        st.session_state[k] = v
    st.session_state["audio_style"] = AUDIO_STYLE_CUSTOM     # the style owns the dials — no pace re-tune over them
    appscope = st.session_state.get("_audio_appscope", {})
    appscope.update({k: st.session_state[k] for k in ["segmentation_method", "audio_style", "cut_style",
                                                      "cut_style_pace", *settings] if k in AUDIO_SETTINGS_KEYS})
    st.session_state["_audio_appscope"] = appscope
    st.session_state["style_report"] = {"track_id": st.session_state["track_id"], "notes": notes}
    st.session_state.pop("aim_report", None)


def _render_style_row(_v) -> None:
    """🎨 Cutting style + Pace — the quick way in; the method and dials below stay editable."""
    opts = ["custom"] + list(CSTY.STYLES)
    cur = _v("cut_style") if _v("cut_style") in opts else "custom"
    with st.container(border=True):
        c = st.columns([3, 2])
        style = c[0].selectbox(
            "🎨 Cutting style", opts, index=opts.index(cur), key="cut_style", on_change=_apply_cut_style,
            format_func=lambda k: "✏️ My own settings" if k == "custom"
            else CSTY.STYLES[k]["label"],
            help="Pick the feel and the app sets the cutting method and all its dials for this track. You can "
                 "still change anything below (it then shows ✏️ My own settings).")
        c[1].select_slider("Pace", options=[-2, -1, 0, 1, 2], value=int(_v("cut_style_pace") or 0),
                           key="cut_style_pace", format_func=CSTY.PACE_LABELS.get, on_change=_apply_cut_style,
                           disabled=style == "custom",
                           help="Each step halves or doubles how often the style cuts.")
        if style != "custom":
            st.caption(CSTY.STYLES[style]["help"])
            rep = st.session_state.get("style_report")
            if rep and rep.get("track_id") == st.session_state.get("track_id"):
                for n in rep["notes"][1:]:
                    st.caption("• " + n)


def _audio_retune_if_paced() -> None:
    """Method, pace or beats per bar changed: re-tune now (unless Custom)."""
    if st.session_state.get("audio_style") in AUDIO_STYLES:
        apply_audio_profile(load_track(st.session_state["track_id"]), st.session_state["track_id"], report=True)


def _apply_cut_focus() -> None:
    mix = {**{k: CS.DEFAULTS[k] for k in CS.REASONS}, "sc_w_vocal_mid": CS.DEFAULTS["sc_w_vocal_mid"],
           **CUT_FOCUS.get(st.session_state.get("sc_focus"), {})}
    for k, v in mix.items():
        st.session_state[k] = v   # the cutting pace doesn't touch these, so it stays as it is


def _pace_row(_pc, _styles, _style_now) -> None:
    """Cutting pace preset + Re-tune (every method except Smart)."""
    _pc[0].selectbox(
        "Cutting pace", _styles, index=_styles.index(_style_now), key="audio_style",
        on_change=_method_changed,
        format_func=lambda x: {"Calm": "🌊 Calm — a cut every ~4 bars",
                               "Balanced": "⚖️ Balanced — every ~2 bars",
                               "Energetic": "🔥 Energetic — every bar",
                               AUDIO_STYLE_CUSTOM: "✏️ Custom — your own settings"}.get(x, x),
        help="Sets every dial below for THIS track — its tempo, loudness and hits — and keeps them "
             "tuned if you change the track or method. Moving any dial switches to Custom.")
    _pc[1].button("🎯 Re-tune for this track", on_click=_run_auto_tune, use_container_width=True,
                  help="Re-runs auto-detect for the current method at the chosen pace (Balanced if Custom). "
                       "Same as the sidebar's Auto-detect.")


AIM_MODES = {"avg": "Average clip length", "count": "Number of segments"}
ADAPTIVE_DIAL_KEYS = ["change_window_secs", "change_threshold", "min_segment_sec", "max_segment_sec",
                      "react_to_hits", "hit_threshold", "snap_secs"]


def _aim_adaptive() -> None:
    """🎯 Set the Adaptive dials to give the segment count / clip length asked for
    (core/targets.py). Runs as a button callback, so the dials redraw with the new values."""
    track = load_track(st.session_state["track_id"])
    mode = st.session_state.get("aim_mode", "avg")
    base = {k: st.session_state.get(k, DIAL_DEFAULTS.get(k)) for k in ADAPTIVE_DIAL_KEYS}
    kw = ({"avg_sec": float(st.session_state.get("aim_avg_sec", 2.5))} if mode == "avg"
          else {"segments": int(st.session_state.get("aim_segments", 40))})
    res = CTGT.adaptive_for_target(track, base=base, beats_per_bar=int(st.session_state.get("beats_per_bar", 4)),
                                   **kw)
    for k, v in res["settings"].items():
        st.session_state[k] = v
    st.session_state["audio_style"] = AUDIO_STYLE_CUSTOM      # your own settings now — don't re-tune over them
    st.session_state["cut_style"] = "custom"
    appscope = st.session_state.get("_audio_appscope", {})
    appscope.update({k: v for k, v in res["settings"].items() if k in AUDIO_SETTINGS_KEYS})
    appscope["audio_style"] = AUDIO_STYLE_CUSTOM
    st.session_state["_audio_appscope"] = appscope
    st.session_state["aim_report"] = {"track_id": st.session_state["track_id"], "exact": res["exact"],
                                      "notes": res["notes"]}


def _render_aim_adaptive(track: dict, current_segments: int) -> None:
    """🎯 Aim for: say how many cuts you want and let the app set the dials."""
    dur = float(track["duration_sec"])
    st.session_state.setdefault("aim_avg_sec", round(min(max(dur / max(current_segments, 1), 0.5), 30.0), 1))
    st.session_state.setdefault("aim_segments", int(min(max(current_segments, 1), 1000)))
    with st.container(border=True):
        c = st.columns([2, 2, 2])
        mode = c[0].selectbox("🎯 Aim for", list(AIM_MODES), format_func=AIM_MODES.get, key="aim_mode",
                              help="Say what you want and the app sets the dials below to get it. "
                                   "You can still fine-tune them afterwards.")
        if mode == "avg":
            c[1].number_input("Seconds per clip (average)", 0.5, 30.0, step=0.5, key="aim_avg_sec",
                              help=f"This track is {dur:.0f}s long, so 2.5s ≈ {dur / 2.5:.0f} segments.")
        else:
            c[1].number_input("Segments", 1, 1000, step=1, key="aim_segments",
                              help=f"This track is {dur:.0f}s long.")
        c[2].button("Set the dials", on_click=_aim_adaptive, type="primary", use_container_width=True,
                    help="Finds energy-change and big-hit thresholds plus shortest / longest segment that "
                         "give that many cuts on this track.")
        rep = st.session_state.get("aim_report")
        if rep and rep.get("track_id") == st.session_state.get("track_id"):
            (st.caption if rep["exact"] else st.warning)("  \n".join(rep["notes"]))


# ---------------------------------------------------------------------------
# ✂️ Hand-edited cuts (per project): st.session_state["cut_edits"] =
#   {"track_id", "cuts": [seconds…], "base_sig": settings the cuts were edited from}
# planned_segments() is what everything downstream uses — the method's segments,
# or the edited ones when this project has edits for this track.
# ---------------------------------------------------------------------------

def _bump_cut_edit_rev() -> None:
    """Tell the editor the cut list changed from here (load, reset…), not from its own edits."""
    st.session_state["cut_edit_rev"] = int(time.time() * 1000) % 2_000_000_000


def _cut_edits(track_id: str = None):
    e = st.session_state.get("cut_edits")
    tid = track_id or st.session_state.get("track_id")
    return e if isinstance(e, dict) and e.get("track_id") == tid and isinstance(e.get("cuts"), list) else None


def planned_segments(track: dict, cfg: dict) -> list:
    """The segments the plan uses: the method's, or this project's hand-edited cuts."""
    base = build_segments(track, cfg)
    e = _cut_edits()
    return CED.segments_from_cuts(track, e["cuts"], base) if e else base


def _drop_cut_edits() -> None:
    st.session_state.pop("cut_edits", None)
    _bump_cut_edit_rev()


def _keep_cut_edits(sig: str) -> None:
    e = _cut_edits()
    if e:
        e["base_sig"] = sig


def _render_cut_editor(track: dict) -> None:
    """✂️ Listen and edit the cuts — see cut_editor.py."""
    tid = st.session_state["track_id"]
    cfg = segmentation_settings()
    sig = CED.settings_signature(cfg)
    bpb = int(st.session_state.get("beats_per_bar", 4))
    base_cuts = CED.base_cut_times(build_segments(track, cfg))
    e = _cut_edits(tid)
    st.subheader("✂️ Listen and edit the cuts")
    if e and e.get("base_sig") != sig:
        d = CED.diff(base_cuts, e["cuts"])
        with st.container(border=True):
            st.warning(f"You edited these cuts by hand, then changed the cutting settings. The plan still uses "
                       f"your edited cuts ({len(e['cuts']) + 1} segments) — the new settings would give "
                       f"{len(base_cuts) + 1}, with {len(d['added'])} of your cuts not among them.")
            bc = st.columns(2)
            bc[0].button("✋ Keep my edited cuts", on_click=_keep_cut_edits, args=(sig,), use_container_width=True)
            bc[1].button("♻️ Use the new settings (drop my edits)", on_click=_drop_cut_edits,
                         use_container_width=True)

    audio_url = ""
    audio_path, missing = _local_media(track.get("source_path", ""))
    if audio_path is not None:
        clip, err = preview_snippets.make_editor_audio(PREVIEW_DIR / "snippets", audio_path)
        if err:
            st.warning(f"Couldn't prepare the audio for playback: {err}")
        else:
            audio_url = _media_url(clip, "audio/mpeg", f"cut_editor_audio_{tid}")
    elif missing:
        _dc = st.columns([3, 2])
        _dc[0].caption(f"▶ Playback needs {Path(colab_to_local(missing)).name} on the server — you can still "
                       f"edit the cuts without it.")
        if _DRIVE_SYNC_AVAILABLE and "MyDrive/" in missing and _dc[1].button(
                "⬇️ Download the track from Google Drive", key="ced_dl", use_container_width=True):
            bar = st.progress(0.0, text="Downloading…")
            dl_err = download_source_video(missing.split("MyDrive/", 1)[1], Path(colab_to_local(missing)),
                                           progress_callback=lambda pr: bar.progress(min(pr, 1.0)))
            bar.empty()
            if dl_err:
                st.error(f"Download failed: {dl_err}")
            else:
                rerun_fragment()

    if "cut_edit_rev" not in st.session_state:
        _bump_cut_edit_rev()
    rev = int(st.session_state["cut_edit_rev"])
    cuts_now = e["cuts"] if e else base_cuts
    val = _cut_editor_widget(track, tid, cuts=cuts_now, base_cuts=base_cuts,
                             beats=CS.beat_grid(track)[0], bars=CS.bar_lines(track, bpb), doc_rev=rev,
                             audio_url=audio_url)
    if isinstance(val, dict) and val.get("edited") and int(val.get("doc_rev", -1)) == rev:
        new = CED.clean_cuts(val.get("cuts") or [], float(track["duration_sec"]))
        if new != cuts_now:
            if not CED.diff(base_cuts, new)["added"] and len(new) == len(base_cuts):
                st.session_state.pop("cut_edits", None)       # back to exactly the method's cuts
            else:
                st.session_state["cut_edits"] = {"track_id": tid, "cuts": new,
                                                 "base_sig": e.get("base_sig", sig) if e else sig}
            e = _cut_edits(tid)
    if e:
        d = CED.diff(base_cuts, e["cuts"])
        ec = st.columns([4, 2])
        ec[0].caption(f"✏️ **Edited by hand:** {len(e['cuts']) + 1} segments — {len(d['added'])} cut(s) added or "
                      f"moved, {len(d['removed'])} of the method's removed. Saved with this project. "
                      f"🟠 = your cuts · 🔵 = the method's · dashed = removed.")
        ec[1].button("↺ Undo all edits", on_click=_drop_cut_edits, use_container_width=True,
                     help="Drops your hand edits for this track (Undo in the editor can't bring them back).")
    else:
        st.caption("▶ Play and listen — tap ✂️ *Cut here* on the beat, drag a cut to move it, or select one and "
                   "🗑 remove it. Your edits are saved with this project.")


def _smart_reset() -> None:
    for k, v in SM.DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["cut_style"] = "smart"        # = the 🧠 Smart style at normal pace
    st.session_state["cut_style_pace"] = 0


_PACE_LABELS = {-2: "much calmer", -1: "calmer", 0: "auto", 1: "busier", 2: "much busier"}


def _render_smart_cut_settings(track: dict, _v) -> None:
    """Controls and the per-section breakdown for the Smart method. Smart tunes
    itself to the track; these only dial it in (the cutting-pace presets don't apply)."""
    _custom = dict(on_change=_mark_style_custom)
    a = track.get("analysis") or {}
    if not a:
        st.info("This track only has the basic analysis — Smart works from its beats and loudness. Add the "
                "extra analysis (bars, sections, drops, build-ups, drums, vocals) from **Media Library → Audio → "
                "🔬 Audio analysis** for its full picture of the song.")
    c = st.columns(2)
    c[0].select_slider("Pace", options=[-2, -1, 0, 1, 2], value=int(round(float(_v("sm_pace")))), key="sm_pace",
                       format_func=lambda x: _PACE_LABELS[x], **_custom,
                       help="Moves the whole edit one grid level calmer or busier than the pace Smart picked "
                            "for this song (e.g. every bar → every 2 bars).")
    c[1].slider("Contrast", 0.0, 2.0, value=float(_v("sm_contrast")), step=0.25, key="sm_contrast", **_custom,
                help="How much the pace changes between quiet and intense sections. 1 = what this song's own "
                     "dynamics suggest; 0 = the same pace throughout; 2 = exaggerated.")
    t = st.columns(4)
    _bld = t[0].toggle("Speed up into drops", value=bool(_v("sm_builds")), key="sm_builds", **_custom,
                       help="Through a build-up the cuts get quicker (e.g. every 2 bars → bar → half bar), "
                            "landing on the drop. How long the speed-up runs and how fast it gets follow how "
                            "big the rise into the drop is.")
    c[1].slider("Speed-up strength", 0.0, 2.0, value=float(_v("sm_build_strength")), step=0.25,
                key="sm_build_strength", disabled=not _bld, **_custom,
                help="1 = follows the song: a small rise gets a short nudge (2 bars, a step or two quicker), a "
                     "huge rise a long, steep run (up to 16 bars). Lower = gentler and shorter; 2 = longer and "
                     "faster; 0 = no speed-up.")
    t[1].toggle("Accents", value=bool(_v("sm_accents")), key="sm_accents", **_custom,
                help="An extra cut on hits that stand out from the rest of the song, and on vocal entries — "
                     "only where the section's rhythm leaves room.")
    t[2].toggle("Respect vocals", value=bool(_v("sm_vocals")), key="sm_vocals", **_custom,
                help="Moves a cut that would land mid-line to the gap between sung lines (needs vocal analysis).")
    t[3].toggle("Mirror repeats", value=bool(_v("sm_mirror")), key="sm_mirror", **_custom,
                help="Repeated sections (chorus 1 and 2) get the same cutting pattern.")
    _, rep = SM.plan_cuts(track, segmentation_settings())
    with st.expander("🧠 How Smart read this song", expanded=True):
        for note in rep.get("notes", []):
            st.caption("• " + note)
        if rep.get("features"):
            st.caption("• Intensity from: " + ", ".join(rep["features"]))
        rows = [{"From": format_mmss(s["start"]), "To": format_mmss(s["end"]), "Section": s["label"],
                 "Intensity": s["intensity"], "A cut every": s["every"], "Why": s["reason"]}
                for s in rep.get("sections", [])]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True,
                         column_config={"Intensity": st.column_config.ProgressColumn(
                             "Intensity", min_value=0.0, max_value=1.0, format="%.2f")})


def _render_scored_cut_settings(track: dict, _v) -> None:
    """Dials for the Best musical moments (Scored cuts) method."""
    a = track.get("analysis") or {}
    if not a:
        st.info("This track only has the basic analysis, so cuts are scored from hits and beats alone. "
                "Add the extra analysis (bars, drums, sections, drops, vocals) from **Media Library → "
                "Audio → 🔬 Audio analysis** for the full set of signals.")
    _custom = dict(on_change=_mark_audio_custom)
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Average bars per cut", 0.5, 8.0, value=float(_v("sc_bars_per_cut")), step=0.25,
                  key="sc_bars_per_cut", **_custom,
                  help="Overall cut rate: 1 = about one cut per bar, 4 = one per 4 bars.")
        st.slider("Follow the energy", 0.0, 1.0, value=float(_v("sc_follow_energy")), step=0.05,
                  key="sc_follow_energy", **_custom,
                  help="How much faster loud, busy passages cut than quiet ones. 0 = the same rate everywhere.")
    with col2:
        st.slider("Shortest segment (s)", 0.3, 8.0, value=float(_v("sc_min_sec")), step=0.1, key="sc_min_sec",
                  **_custom, help="No two cuts closer than this.")
        st.slider("Longest segment (s)", 0.0, 30.0, value=float(_v("sc_max_sec")), step=0.5, key="sc_max_sec",
                  **_custom, help="No segment longer than this, if there's any reasonable cut in range. 0 = no limit.")
    with st.expander("What to cut on", expanded=False):
        _fl = list(CUT_FOCUS)
        st.selectbox("Mix", _fl, index=_fl.index(_v("sc_focus")) if _v("sc_focus") in _fl else 0,
                     key="sc_focus", on_change=_apply_cut_focus,
                     help="Sets the weights below: Song structure favours bar lines, phrases, sections and "
                          "drops; Drums & hits favours kicks, snares and big hits; Vocals cuts between sung "
                          "lines and avoids cutting through them.")
        if a and a.get("vocals") is None:
            st.caption("No vocal analysis for this track — the vocal weights have no effect.")
        wcols = st.columns(3)
        weight_help = {
            "sc_w_bar": ("Bar line", "The first beat of a bar."),
            "sc_w_phrase": ("Phrase line", "Every 4 / 8 / 16 bars from a section start — 16 counts most."),
            "sc_w_section": ("Section change", "Verse → chorus etc., found from a change in the sound."),
            "sc_w_drop": ("Drop", "The moment a build-up lands."),
            "sc_w_hit": ("Big hit", "Any sharp new sound (onset strength)."),
            "sc_w_kick": ("Kick", "Low-frequency drum hits."),
            "sc_w_snare": ("Snare / clap", "Mid-frequency drum hits."),
            "sc_w_harmony": ("Chord change", "The notes / chords change."),
            "sc_w_vocal": ("Vocal line start / end", "Where a sung phrase begins or ends."),
            "sc_w_vocal_mid": ("Avoid cutting mid-vocal", "Penalty for cutting while a vocal line is being sung."),
        }
        for i, (key, (label, hlp)) in enumerate(weight_help.items()):
            with wcols[i % 3]:
                st.slider(label, 0.0, 5.0, value=float(_v(key)), step=0.1, key=key, help=hlp)


CUT_KIND_STYLES = (
    ("change", "rgba(0,160,120,0.7)", "Cut: energy change (snapped)"),
    ("hit", "rgba(220,60,60,0.55)", "Cut: big hit"),
    ("fill", "rgba(230,150,30,0.6)", "Cut: max-length split (on strongest beat)"),
    ("cinematic", "rgba(80,120,220,0.5)", "Rhythm Engine: CINEMATIC"),
    ("fast", "rgba(230,150,30,0.6)", "Rhythm Engine: FAST"),
    ("override", "rgba(230,30,200,0.75)", "Rhythm Engine: Dynamic Priority Override"),
    ("sc_section", "rgba(150,60,200,0.9)", "Scored cut: section change"),
    ("sc_drop", "rgba(220,40,40,0.9)", "Scored cut: drop"),
    ("sc_phrase", "rgba(60,90,200,0.8)", "Scored cut: phrase line"),
    ("sc_bar", "rgba(90,90,90,0.6)", "Scored cut: bar line"),
    ("sc_kick", "rgba(200,60,40,0.6)", "Scored cut: kick"),
    ("sc_snare", "rgba(40,140,200,0.6)", "Scored cut: snare"),
    ("sc_hit", "rgba(220,60,60,0.55)", "Scored cut: big hit"),
    ("sc_harmony", "rgba(60,170,90,0.7)", "Scored cut: chord change"),
    ("sc_vocal", "rgba(230,120,0,0.8)", "Scored cut: vocal line"),
    ("sc_beat", "rgba(140,140,140,0.5)", "Scored cut: beat"),
    ("sm_section", "rgba(150,60,200,0.9)", "Smart: section change"),
    ("sm_drop", "rgba(220,40,40,0.9)", "Smart: drop"),
    ("sm_build", "rgba(230,150,30,0.8)", "Smart: build-up (speeding up)"),
    ("sm_grid", "rgba(60,90,200,0.6)", "Smart: section's cutting rhythm"),
    ("sm_accent", "rgba(220,60,60,0.7)", "Smart: accent (standout hit)"),
    ("sm_vocal", "rgba(230,120,0,0.8)", "Smart: vocal entry"),
)


def add_cut_markers(fig, segments: list) -> None:
    """Vertical markers at every cut, coloured by why the cut happened."""
    for _kind, _colour, _label in CUT_KIND_STYLES:
        _xs, _ys = [], []
        for _sg in segments:
            if _sg.get("cut") == _kind:
                _xs += [_sg["start"], _sg["start"], None]
                _ys += [0, 1, None]
        if _xs:
            fig.add_trace(go.Scatter(x=_xs, y=_ys, mode="lines", name=_label,
                                     line=dict(color=_colour, width=1, dash="dot"), hoverinfo="skip"))


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
    autosave_now()
    _shadow = st.session_state.get("audio_settings_shadow", {})

    def _v(key):
        return st.session_state.get(key, _shadow.get(key, DIAL_DEFAULTS.get(key)))

    st.subheader("How the music is cut")
    _render_style_row(_v)
    _seg_options = [SEG_SMART, SEG_SCORED, SEG_ADAPTIVE, SEG_RHYTHM]
    _seg_default = _v("segmentation_method")
    segmentation_method = st.radio(
        "Method", _seg_options,
        index=_seg_options.index(_seg_default) if _seg_default in _seg_options else 0,
        key="segmentation_method", format_func=lambda m: SEG_LABELS.get(m, m), horizontal=True,
        on_change=_method_changed,
        help="🧠 Smart: reads the song's structure and intensity and sets its own cutting rhythm — a "
             "steady pace per section on the bar / beat grid, faster when the music is bigger, speeding up "
             "into drops, with cuts at every section change. 🎯 Best musical moments: scores every beat for how good a cut it would be (bar lines, "
             "phrases, section changes, drops, drums, chords, vocals) and picks the best set together. "
             "⚡ Energy changes & big hits: cuts where the loudness shifts and on the strongest hits, "
             "wherever they fall. 🎬 Bar styles: each stretch of bars is either CINEMATIC (a cut every "
             "few bars) or FAST (a cut every bar), from how loud it is.",
    )
    _pc = st.columns([2, 2, 1])
    _styles = list(AUDIO_STYLES) + [AUDIO_STYLE_CUSTOM]
    _style_now = _v("audio_style") if _v("audio_style") in _styles else AUDIO_STYLE_CUSTOM
    if segmentation_method == SEG_SMART:
        # Smart sets its own pace from the track; the Pace slider below dials it in.
        _pc[0].caption("🧠 Smart sets its pace from the track itself — dial it in with **Pace** below.")
        _pc[1].button("↺ Back to auto", on_click=_smart_reset, use_container_width=True,
                      help="Pace back to auto, contrast back to the song's own, all switches on.")
    else:
        _pace_row(_pc, _styles, _style_now)
    _pc[2].number_input(
        "Beats per bar", min_value=2, max_value=12, value=int(_v("beats_per_bar")), key="beats_per_bar",
        on_change=_beats_per_bar_changed,
        help="4 = common time (most pop/rock/EDM). Use 3 for a waltz, 6 for 6/8, etc.",
    )
    if st.session_state.get("auto_tune_report"):
        with st.expander("What auto-detect chose, and why", expanded=False):
            for line in st.session_state["auto_tune_report"]:
                st.caption(line)

    _custom = dict(on_change=_mark_audio_custom)
    if segmentation_method == SEG_SMART:
        _render_smart_cut_settings(track, _v)
    elif segmentation_method == SEG_SCORED:
        _render_scored_cut_settings(track, _v)
    elif segmentation_method == SEG_ADAPTIVE:
        _render_aim_adaptive(track, len(build_segments(track, segmentation_settings())))
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Cut at energy changes — threshold", 0.02, 0.5, value=float(_v("change_threshold")),
                      step=0.01, key="change_threshold", **_custom,
                      help="How big a jump in loudness counts as a change worth cutting on. Lower = more cuts.")
            react = st.toggle("Cut on big hits", value=bool(_v("react_to_hits")), key="react_to_hits", **_custom,
                              help="Also cut exactly on the strongest hits, wherever they fall.")
            st.slider("Big-hit threshold", 0.5, 1.0, value=float(_v("hit_threshold")), step=0.01,
                      key="hit_threshold", disabled=not react, **_custom,
                      help="How strong a hit must be (0–1) to become a cut. Lower = more cuts.")
        with col2:
            st.slider("Shortest segment (s)", 0.3, 8.0, value=float(_v("min_segment_sec")), step=0.1,
                      key="min_segment_sec", **_custom, help="No two cuts closer than this.")
            st.slider("Longest segment (s)", 0.0, 30.0, value=float(_v("max_segment_sec")), step=0.5,
                      key="max_segment_sec", **_custom,
                      help="Longer segments are split at their strongest beat. 0 = no limit.")
        with st.expander("Advanced", expanded=False):
            ac = st.columns(2)
            ac[0].slider("Energy-change window (s)", 0.5, 6.0, value=float(_v("change_window_secs")), step=0.1,
                         key="change_window_secs", **_custom,
                         help="Compares the loudness this long after each moment with this long before it.")
            ac[1].slider("Snap radius (s)", 0.0, 1.0, value=float(_v("snap_secs")), step=0.05, key="snap_secs",
                         **_custom, help="Energy-change cuts move onto the strongest hit within this distance.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Fast-cutting threshold", 0.1, 1.0, value=float(_v("fast_energy_thresh")), step=0.01,
                      key="fast_energy_thresh", **_custom,
                      help="Bars louder than this (0–1 of the loudest) cut every bar (FAST); quieter bars "
                           "are CINEMATIC. Lower = more fast cutting.")
            st.number_input("Cinematic segment length (bars)", min_value=1, max_value=8,
                            value=int(_v("cinematic_bars")), key="cinematic_bars", **_custom,
                            help="How many bars a CINEMATIC segment spans. 2, 4 or 8 follow musical phrases.")
        with col2:
            dpo = st.toggle("Extra cuts at sudden jumps", value=bool(_v("dynamic_priority_override")),
                            key="dynamic_priority_override", **_custom,
                            help="Adds a cut at a sudden mid-bar jump in loudness (a hit or drop) that the "
                                 "bar-by-bar styles would smooth away.")
            st.slider("Sudden-jump sensitivity", 0.02, 1.0, value=float(_v("override_delta_thresh")), step=0.01,
                      key="override_delta_thresh", disabled=not dpo, **_custom,
                      help="How large a jump triggers an extra cut. Lower = more extra cuts.")
        with st.expander("Advanced", expanded=False):
            ac = st.columns(2)
            with ac[0]:
                st.number_input("FAST max run (bars)", min_value=0, max_value=64,
                                value=int(_v("fast_max_bars")), key="fast_max_bars", **_custom,
                                help="After this many FAST bars in a row, FAST stops for a while. 0 = no limit.")
                st.number_input("FAST pause after a run (bars)", min_value=0, max_value=32,
                                value=int(_v("fast_cooldown_bars")), key="fast_cooldown_bars", **_custom,
                                help="Bars that must pass after a FAST run before FAST can start again.")
            with ac[1]:
                st.number_input("Hold a style for (bars)", min_value=1, max_value=8,
                                value=int(_v("phrase_lock_bars")), key="phrase_lock_bars", **_custom,
                                help="FAST or CINEMATIC is kept this many bars before it's re-decided.")
                st.slider("Clear zone around a jump cut (s)", 0.0, 4.0, value=float(_v("override_clear_secs")),
                          step=0.1, key="override_clear_secs", disabled=not dpo, **_custom,
                          help="Other cuts closer than this to a jump cut are removed.")

    st.divider()
    _render_cut_editor(track)

    # ---- Live preview chart ------------------------------------------------
    st.divider()
    st.subheader("Preview: cut points on the audio signal")
    st.caption("Adjust dials above and the chart updates immediately. "
               "Scene match dots appear on the Matching & Export step once clips are assigned.")

    # Build segments from current widget values for the preview
    _dc = st.session_state.get("density_contrast", 1.0)
    _min_c = st.session_state.get("min_clips", 1)
    _max_c = st.session_state.get("max_clips", 4) if st.session_state.get("split_screen_enabled") else 1
    _rs, _re = st.session_state.get("ramp_range", (0.2, 0.85))

    _segs = planned_segments(track, segmentation_settings())
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

    add_analysis_overlays(afig, track)
    add_cut_markers(afig, _segs)

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
    _bar = CS.bar_sec(track, int(st.session_state.get("beats_per_bar", 4)))
    _bars = np.asarray(CS.bar_lines(track, int(st.session_state.get("beats_per_bar", 4))), dtype=float)
    _cuts = [sg["start"] for sg in _segs[1:]]
    _on_bar = (sum(1 for c in _cuts if len(_bars) and np.min(np.abs(_bars - c)) < 0.12) / len(_cuts)) if _cuts else 0.0
    st.caption(f"**{len(_segs)} segments** · a cut every {track['duration_sec'] / max(len(_segs), 1) / _bar:.1f} bars "
               f"on average ({track['duration_sec'] / max(len(_segs), 1):.1f}s) · {_on_bar:.0%} of cuts on a bar line  "
               f"|  {CS.beat_grid(track)[1]:.0f} BPM  |  next: **🎭 Story map**, when you're happy with the cuts.")

    # Save the shadow copy every call — this is what actually survives the round trip to
    # another section (plain session_state, not tied to any widget), and is what the explicit
    # value=/index= above restore from on re-entry.
    st.session_state["audio_settings_shadow"] = {
        k: st.session_state[k] for k in AUDIO_SETTINGS_KEYS if k in st.session_state
    }


# ---------------------------------------------------------------------------
# Step 1: Story map — which parts of the song call for which scene role
# (story_map.py for the logic; scene_roles.py for the roles)
# ---------------------------------------------------------------------------
STORY_KEY = "story_map"   # {"track_id", "regions": [...], "v": edit counter, "edited": bool}
SECTION_AUDIO = "🎵 0. Audio Settings"
SECTION_STORY = "🎭 1. Story map"
SECTION_VIDEOS = "📼 2. Videos & Time Ranges"
SECTION_MATCH = "🎬 3. Matching & Export"
SECTIONS = [SECTION_AUDIO, SECTION_STORY, SECTION_VIDEOS, SECTION_MATCH]


def _track_mtime(track_id: str) -> float:
    try:
        return (AUDIO_DIR / f"{track_id}.json").stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(max_entries=8, show_spinner="Reading the song's structure…")
def _story_analysis(track_id: str, beats_per_bar: int, mtime: float) -> dict:
    return STM.analyse(load_track(track_id), beats_per_bar)


def current_story_map(track_id: str):
    """This track's story map regions, or None if it hasn't got one. The one in
    this session/project first; otherwise the track's saved map (Media Library →
    Audio, or saved from another project), which this session then adopts."""
    sm = st.session_state.get(STORY_KEY)
    if isinstance(sm, dict) and sm.get("track_id") == track_id:
        if sm.get("regions"):
            return sm["regions"]
        if sm.get("cleared"):
            return None
    saved = STM.saved_map(track_id) if track_id else None
    if saved:
        st.session_state[STORY_KEY] = {"track_id": track_id, "regions": saved, "v": 1, "edited": False}
        return saved
    return None


def _story_set(regions: list, edited: bool = True) -> None:
    sm = st.session_state.get(STORY_KEY) if isinstance(st.session_state.get(STORY_KEY), dict) else {}
    st.session_state[STORY_KEY] = {
        "track_id": st.session_state.get("track_id"),
        "regions": regions,
        "v": int(sm.get("v", 0)) + 1 if sm.get("track_id") == st.session_state.get("track_id") else 1,
        "edited": edited,
    }
    st.session_state.pop("sm_play", None)
    try:
        STM.save_map(st.session_state.get("track_id"), regions)   # the track's saved map, for every page
    except OSError:
        pass


def _story_suggest(analysis: dict, cut_times: list) -> None:
    _story_set(STM.snap(STM.suggest(analysis, SR.all_roles()), cut_times), edited=False)


def _story_role_changed(i: int, key: str) -> None:
    _story_set(STM.set_role(current_story_map(st.session_state["track_id"]), i, st.session_state.get(key)))


def _story_edit(fn, *args) -> None:
    _story_set(fn(current_story_map(st.session_state["track_id"]), *args))


def _story_boundary_changed(i: int, key: str) -> None:
    _story_edit(STM.move_boundary, i, float(st.session_state[key]))


def _story_snap_one(i: int, cut_times: list) -> None:
    """📐 on one line: move it onto the nearest cut."""
    regions = current_story_map(st.session_state["track_id"])
    t = regions[i]["end"]
    inner = [c for c in cut_times if regions[i]["start"] < c < regions[i + 1]["end"]]
    if inner:
        _story_edit(STM.move_boundary, i, min(inner, key=lambda c: abs(c - t)))


def _story_play(i: int) -> None:
    st.session_state["sm_play"] = None if st.session_state.get("sm_play") == i else i


def _story_chart(track: dict, regions: list, segs: list, roles: dict, playing) -> go.Figure:
    fig = go.Figure()
    if track.get("hires"):
        rate = float(track["hires"]["rate"])
        rms = np.asarray(track["hires"]["rms"], dtype=float)
        k = max(1, int(rate * 0.5))
        sm = np.convolve(rms, np.ones(k) / k, mode="same")
        sm = sm / max(float(sm.max()), 1e-9)
        fig.add_trace(go.Scatter(x=np.arange(len(sm)) / rate, y=sm, mode="lines", name="Loudness",
                                 line=dict(color="rgba(80,120,220,0.85)", width=1.5), hoverinfo="skip"))
    xs, ys = [], []
    for sg in segs[1:]:
        xs += [sg["start"], sg["start"], None]
        ys += [0, 1.05, None]
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Cuts (blocks)", hoverinfo="skip",
                                 line=dict(color="rgba(128,128,128,0.35)", width=1)))
    for i, r in enumerate(regions):
        role = roles.get(r["role"]) if r["role"] else None
        colour = role["colour"] if role else "#bbbbbb"
        fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=colour,
                      opacity=0.32 if role else 0.06, line_width=2 if i == playing else 0,
                      line_color="#222222", layer="below")
        if role:
            fig.add_annotation(x=(r["start"] + r["end"]) / 2, y=1.12, text=f"{role['emoji']} {role['name']}",
                               showarrow=False, font=dict(size=12))
    fig.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10), showlegend=False,
                      xaxis=dict(title="Time (s)", range=[0, float(track["duration_sec"])]),
                      yaxis=dict(visible=False, range=[0, 1.2]))
    return fig


def render_story_map_section(track: dict) -> None:
    """🎭 Story map: suggest which parts of the song suit Intro / Slow / Action /
    Outro (and your own roles), then adjust them — role per region, handles on
    the cut points, ▶ to hear each part."""
    autosave_now()
    track_id = st.session_state["track_id"]
    bpb = int(st.session_state.get("beats_per_bar", 4))
    duration = float(track["duration_sec"])
    segs = planned_segments(track, segmentation_settings())
    cut_times = [sg["start"] for sg in segs] + [duration]
    roles = SR.roles_by_id()
    analysis = _story_analysis(track_id, bpb, _track_mtime(track_id))

    st.subheader("🎭 Story map")
    st.caption("Which parts of the song call for which kind of scene — 🎬 Intro at the start, 🌙 Slow for "
               "build-ups and breakdowns, 💥 Action for the high points, 🏁 Outro at the end, and your own "
               "roles (Reviewer → 🎭 Scene roles). Mark scenes with these roles in the Reviewer; the planner "
               "uses the role clips in these parts of the song, and normal picks everywhere else.")
    regions = current_story_map(track_id)
    if regions is None:
        old = st.session_state.get(STORY_KEY)
        if isinstance(old, dict) and old.get("regions"):
            st.info(f"The story map you had was for **{old.get('track_id')}** — this track needs its own.")
        c = st.columns([2, 2, 3])
        c[0].button("✨ Suggest a story map from the song", type="primary", use_container_width=True,
                    on_click=_story_suggest, args=(analysis, cut_times))
        c[1].button("Start with an empty map", use_container_width=True,
                    on_click=lambda: _story_set(STM.blank(duration)))
        c[2].caption("Or skip this step — without a story map the planner works as before.")
        return

    tc = st.columns([2, 2, 2, 3])
    tc[0].button("✨ Suggest again", use_container_width=True, on_click=_story_suggest, args=(analysis, cut_times),
                 help="Replaces the map with a fresh suggestion from the song (your edits are lost).")
    tc[1].button("🧹 Clear the map", use_container_width=True, on_click=lambda: _story_set(STM.blank(duration)),
                 help="One region, no role — then split and set roles yourself.")
    tc[2].button("📐 Line up with the cuts", use_container_width=True,
                 on_click=lambda: _story_edit(STM.snap, cut_times),
                 help="Moves every boundary onto the nearest cut, so no block straddles two parts. "
                      "Useful after changing the audio settings.")
    lead = tc[3].toggle("▶ starts 2s early", key="sm_lead_in",
                        help="Hear what leads into each part, not just the part itself.")
    snap = tc[3].toggle("Handles snap to cuts", key="sm_snap",
                        help="On: the lines between parts stop only on the cut points. Off: they move freely "
                             "(0.1s steps) to line up with the audio curve — 📐 next to a line snaps just that "
                             "one to its nearest cut.")

    playing = st.session_state.get("sm_play")
    st.plotly_chart(_story_chart(track, regions, segs, roles, playing), use_container_width=True,
                    key=f"sm_chart_{st.session_state[STORY_KEY].get('v', 0)}")

    # Totals per role
    tot = STM.totals(regions, segs)
    if tot:
        st.markdown("  ·  ".join(f"{SR.label(r, roles)} **{format_mmss(v['sec'])}** ({v['blocks']} block"
                                 f"{'s' if v['blocks'] != 1 else ''})" for r, v in tot.items()))
    unknown = sorted({r["role"] for r in regions if r["role"] and r["role"] not in roles})
    if unknown:
        st.warning("Some parts use a role that's been deleted: " + ", ".join(unknown))

    v = st.session_state[STORY_KEY].get("v", 0)
    role_opts = [None, *roles, *unknown]
    block_role = STM.block_roles(regions, segs)
    for i, r in enumerate(regions):
        role = roles.get(r["role"]) if r["role"] else None
        n_blocks = sum(1 for sg in segs if r["start"] <= (sg["start"] + sg["end"]) / 2 < r["end"])
        with st.container(border=True):
            rc = st.columns([0.12, 1.5, 1.8, 3.2, 0.9, 0.9, 0.9])
            rc[0].markdown(f'<div style="width:18px;height:46px;border-radius:4px;background:'
                           f'{role["colour"] if role else "#cccccc"};"></div>', unsafe_allow_html=True)
            rc[1].markdown(f"**{format_mmss(r['start'])} – {format_mmss(r['end'])}**  \n"
                           f":gray[{r['end'] - r['start']:.0f}s · {n_blocks} block{'s' if n_blocks != 1 else ''}]")
            rk = f"sm_role_{v}_{i}"
            rc[2].selectbox("Role", role_opts, index=role_opts.index(r["role"]) if r["role"] in role_opts else 0,
                            key=rk, format_func=lambda x: "— no role" if x is None else SR.label(x, roles),
                            label_visibility="collapsed", on_change=_story_role_changed, args=(i, rk))
            with rc[3]:
                if r.get("why"):
                    st.caption(("✏️ " if r["why"] == "set by you" else "💡 ") + r["why"])
                alt = r.get("alt")
                if alt and alt.get("role") in roles and alt["role"] != r["role"]:
                    st.button(f"↔ {SR.label(alt['role'], roles)} instead", key=f"sm_alt_{v}_{i}",
                              on_click=_story_edit, args=(STM.set_role, i, alt["role"]),
                              help="The runner-up for this part of the song.")
            rc[4].button("⏹ Stop" if playing == i else "▶ Play", key=f"sm_play_{v}_{i}", use_container_width=True,
                         on_click=_story_play, args=(i,))
            inner = [t for t in cut_times if r["start"] < t < r["end"]]
            mid = min(inner, key=lambda t: abs(t - (r["start"] + r["end"]) / 2)) if inner else None
            rc[5].button("✂️ Split", key=f"sm_split_{v}_{i}", use_container_width=True, disabled=mid is None,
                         on_click=_story_edit, args=(STM.split, i, mid),
                         help="Split into two at the cut nearest the middle (then move the line below).")
            rc[6].button("🔗 Join ↓", key=f"sm_merge_{v}_{i}", use_container_width=True,
                         disabled=i == len(regions) - 1, on_click=_story_edit, args=(STM.merge_next, i),
                         help="Join with the next part (keeps this part's role).")

            if playing == i:
                audio_path, missing = _local_media(track.get("source_path", ""))
                if missing:
                    st.warning(f"Needs {Path(colab_to_local(missing)).name} on the server first.")
                    if _DRIVE_SYNC_AVAILABLE and "MyDrive/" in missing and st.button(
                            "⬇️ Download the track from Google Drive", key=f"sm_dl_{v}_{i}"):
                        bar = st.progress(0.0, text="Downloading…")
                        err = download_source_video(missing.split("MyDrive/", 1)[1], Path(colab_to_local(missing)),
                                                    progress_callback=lambda p: bar.progress(min(p, 1.0)))
                        bar.empty()
                        if err:
                            st.error(f"Download failed: {err}")
                        else:
                            st.rerun()
                else:
                    with st.spinner("Cutting the music…"):
                        clip, err = preview_snippets.make_audio_snippet(
                            PREVIEW_DIR / "snippets", audio_path, r["start"], r["end"] - r["start"],
                            lead_in=2.0 if lead else 0.0)
                    if err:
                        st.error(f"Couldn't cut the music: {err}")
                    else:
                        st.audio(str(clip), autoplay=True)

        # The handle between this part and the next: free (0.1s steps) or on the cuts.
        if i + 1 < len(regions):
            nxt = regions[i + 1]
            bk = f"sm_line_{v}_{i}_{'s' if snap else 'f'}"
            _lab = lambda x: SR.label(x, roles) if x else "no role"
            on_cut = any(abs(c - r["end"]) < 0.05 for c in cut_times)
            lc = st.columns([1.3, 3.6, 0.4])
            lc[0].caption(f"↕ Where {_lab(r['role'])} ends and {_lab(nxt['role'])} begins"
                          + ("" if on_cut else "  ·  :orange[between cuts]"))
            if snap:
                opts = sorted({t for t in cut_times if r["start"] < t < nxt["end"]} | {r["end"]})
                if len(opts) > 1:
                    lc[1].select_slider(
                        "Line between these parts", opts, value=r["end"], key=bk,
                        format_func=lambda t: f"{format_mmss(t)} ({t:.1f}s)", label_visibility="collapsed",
                        on_change=_story_boundary_changed, args=(i, bk),
                        help="Drag to move where one part ends and the next begins — it stops on the cuts.")
            else:
                lo = math.ceil((r["start"] + 0.1) * 10) / 10
                hi = math.floor((nxt["end"] - 0.1) * 10) / 10
                if hi > lo:
                    lc[1].slider(
                        "Line between these parts", min_value=float(lo), max_value=float(hi),
                        value=float(min(max(round(r["end"], 1), lo), hi)), step=0.1, format="%.1fs", key=bk,
                        label_visibility="collapsed", on_change=_story_boundary_changed, args=(i, bk),
                        help="Drag to line the change up with the audio curve (arrow keys move it 0.1s). "
                             "📐 snaps it to the nearest cut.")
            lc[2].button("📐", key=f"sm_snap1_{v}_{i}", disabled=on_cut, on_click=_story_snap_one,
                         args=(i, cut_times), help="Snap this line to the nearest cut.")

    st.divider()
    n_role = sum(1 for x in block_role if x)
    st.caption(f"{n_role} of {len(segs)} blocks have a role. Times are kept in seconds, so changing the "
               "audio settings later keeps the map where it is (📐 lines it up with the new cuts).")


def _select_all_recommended():
    """Tick every video in the current top-RECOMMENDED_COUNT list. Runs as a button
    callback (before the page redraws), using the list saved by the previous run."""
    for vid in st.session_state.get("recommended_ids", []):
        st.session_state[f"select_video_{vid}"] = True


def _randomize_top_videos():
    """🎲 A fresh random seed on every press → a different random top list."""
    st.session_state["random_top_seed"] = random.SystemRandom().randrange(1, 1_000_000)


def _back_to_best_matches():
    st.session_state["random_top_seed"] = None


def _add_new_videos(video_ids: list) -> None:
    """🆕 Add new videos — tick every recently added video (adding it to the list if needed)."""
    for vid in video_ids:
        _add_video_to_selection(vid)


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
    memory_check("Video selection")
    autosave_now()
    top_cols = st.columns([5, 2, 2, 2])
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
    with top_cols[3]:
        st.write("")
        st.button("🎲 Randomize", on_click=_randomize_top_videos, use_container_width=True,
                  help=f"Replace the top list with {RECOMMENDED_COUNT} videos picked purely at random from the "
                       f"whole library (ignores the ranking weights). Press again for a different set.")

    _details = video_details()
    _role_secs_now = video_role_seconds(effective_time_ranges())
    _dc = st.columns([5, 5, 2])
    _f_actors = _dc[0].multiselect("👤 Actors", all_actors(), key="vid_actor_filter", placeholder="anyone",
                                   help="Only videos with these actors (set in the Reviewer's 🎬 Video details).")
    _f_vtags = _dc[1].multiselect("🏷️ Video tags", all_video_tags(), key="vid_vtag_filter", placeholder="any",
                                  help="Only videos with these video tags (set in the Reviewer's 🎬 Video details).")
    _f_all = _dc[2].toggle("Must have all", key="vid_details_all", disabled=not (_f_actors or _f_vtags),
                           help="Off: any of them. On: every one.")
    _allowed = (frozenset(v for v in all_video_ids if matches_details(v, _details, _f_actors, _f_vtags, _f_all))
                if (_f_actors or _f_vtags) else None)
    if _allowed is not None:
        st.caption(f"👤🏷️ {len(_allowed)} of {len(all_video_ids)} videos match — the recommendations and the "
                   f"'add another video' list only include those. Videos you've already ticked stay ticked.")
    render_ranking_weights(tag_filter)
    info = resolve_video_selection(track, tuple(tag_filter), all_video_ids, _allowed)
    video_matches = info["video_matches"]
    st.session_state["recommended_ids"] = info["top_ids"]

    if not video_matches:
        st.warning("No videos have scenes matching the current tag filter.")
        return

    _SORT_OPTIONS = {
        "Match score (best first)":          lambda v: -v["score"],
        "Matching scenes (most first)":      lambda v: -v["matching_scene_count"],
        "Usable footage (most first)":       lambda v: -v["usable_sec"],
        "Overall energy (highest first)":    lambda v: -v["level_pct"],
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

    _seed = st.session_state.get("random_top_seed")
    if _seed is not None:
        _hc = st.columns([6, 2])
        _hc[0].subheader(f"🎲 Random pick — {len(info['top_matches'])} videos")
        _hc[1].button("↩️ Back to best matches", on_click=_back_to_best_matches, use_container_width=True)
        st.caption(f"Picked at random from all {len(video_matches)} videos with usable scenes (seed {_seed}) — "
                   f"the ranking weights aren't used. Match scores are still shown for reference." +
                   (f"  |  filtered to tags: {', '.join(tag_filter)}" if tag_filter else ""))
    else:
        st.subheader(f"Best-matching videos (top {RECOMMENDED_COUNT})")
        st.caption(f"Ranked by match score — see ⚙️ Ranking weights above for what counts" +
                   (f"  |  filtered to tags: {', '.join(tag_filter)}" if tag_filter else ""))
    st.caption(
        "✅ marks a recommended video. ☑ is your actual selection — nothing is selected until you tick it "
        "(or use Select all recommended above), and it stays as you set it even if re-sorting moves the "
        "video. 🎚️ opens a range picker to restrict a video to just a portion of its footage. "
        "📚 is the video's Media Library range (used by default); 🎚️ next to a time is a range "
        "set here, which overrides the library range for this project only."
    )
    _lib_ranges = library_ranges()
    _added = CStore.added_times()
    _new_ids = sorted((v for v in all_video_ids if v in _added and CStore.is_new_video(_added[v])),
                      key=lambda v: -_added[v])

    def render_video_row(video_id: str, recommended: bool, default_checked: bool):
        match = info["match_by_id"].get(video_id)
        duration = get_video_duration(video_id)
        _overrides = st.session_state["video_time_ranges"]
        lib_range = _lib_ranges.get(video_id)
        has_override = video_id in _overrides
        current_range = _overrides[video_id] if has_override else lib_range
        if current_range:
            range_tag = (f"  {'🎚️' if has_override else '📚'} "
                         f"{seconds_to_time(current_range[0]):%M:%S}–{seconds_to_time(current_range[1]):%M:%S}")
        elif has_override and lib_range:
            range_tag = "  🎚️ full video (library range off for this project)"
        else:
            range_tag = ""

        row_cols = st.columns([0.5, 1.5, 7, 1], vertical_alignment="center") \
            if _COLUMNS_SUPPORT_VALIGN else st.columns([0.5, 1.5, 7, 1])
        with row_cols[0]:
            st.checkbox("select", value=default_checked, key=f"select_video_{video_id}",
                        label_visibility="collapsed")
        with row_cols[1]:
            _thumb = video_list_thumbnail(catalogues, video_id, current_range)
            if _thumb is not None:
                safe_image(_thumb, width=120)
        with row_cols[2]:
            rank_mark = "✅ " if recommended else ""
            new_mark = f" · 🆕 added {CStore.added_ago(_added[video_id])}" if video_id in _new_ids else ""
            title = f"{rank_mark}**{video_id}** · ⏱ {format_mmss(duration)}{new_mark}"
            if match:
                st.write(f"{title} — match **{match['score']:.0f}%**  |  "
                         f"{match['matching_scene_count']} scenes · {format_mmss(match['usable_sec'])} usable{range_tag}")
                st.caption(format_match_breakdown(match, bool(st.session_state.get("ranking_experimental")),
                                                  tag_filter))
            else:
                st.write(f"{title} — no scenes match the current tag filter / time range{range_tag}")
            _rs = (match or {}).get("role_sec") or _role_secs_now.get(video_id) or {}
            if _rs:
                _rb = SR.roles_by_id()
                st.caption("🎭 " + "  ·  ".join(f"{SR.label(r, _rb)} {format_mmss(v)}"
                                                for r, v in sorted(_rs.items(), key=lambda kv: -kv[1])))
            _vd = _details.get(video_id) or {}
            if _vd.get("actors") or _vd.get("tags"):
                st.caption(("👤 " + ", ".join(_vd.get("actors", [])) if _vd.get("actors") else "")
                           + ("  ·  " if _vd.get("actors") and _vd.get("tags") else "")
                           + ("🏷️ " + ", ".join(_vd.get("tags", [])) if _vd.get("tags") else ""))
        with row_cols[3]:
            if st.button("✨", key=f"sim_toggle_{video_id}", help="Find videos similar to this one"):
                st.session_state["show_similar_videos"] = (
                    None if st.session_state.get("show_similar_videos") == video_id else video_id)
                rerun_fragment()
            if st.button("🎚️", key=f"range_toggle_{video_id}", help="Set a time range for this video"):
                # One picker open at a time: each open picker embeds a full video player,
                # which is the heaviest thing on this page to re-render.
                _was_open = st.session_state["show_range_picker"].get(video_id, False)
                st.session_state["show_range_picker"] = {video_id: not _was_open}
                rerun_fragment()

        if st.session_state.get("show_similar_videos") == video_id:
            with st.container(border=True):
                st.markdown(f"**✨ Videos similar to `{video_id}`** — by tags and movement, "
                            f"within each video's time range" + (" and tag filter" if tag_filter else ""))
                _sk = (video_id, tuple(tag_filter), tuple(sorted((v, tuple(r)) for v, r in effective_time_ranges().items())))
                _cached = st.session_state.get("similar_videos_cache")
                if not _cached or _cached["key"] != _sk:
                    _cached = {"key": _sk, "found": compute_similar_videos(video_id, tuple(tag_filter),
                                                                          effective_time_ranges())}
                    st.session_state["similar_videos_cache"] = _cached
                if not _cached["found"]:
                    st.caption("No other videos to compare with.")
                _selected_now = set(st.session_state.get("committed_selected_videos", []))
                for _sim, _reason, _vid in _cached["found"]:
                    sc = st.columns([1.5, 7, 1.5])
                    with sc[0]:
                        _t = video_list_thumbnail(catalogues, _vid, effective_time_ranges().get(_vid))
                        if _t is not None:
                            safe_image(_t, width=110)
                    with sc[1]:
                        st.markdown(f"**{_vid}** · ⏱ {format_mmss(get_video_duration(_vid))} · "
                                    f"**{int(round(100 * _sim))}% similar**")
                        st.caption(_reason)
                    with sc[2]:
                        _is_sel = _vid in _selected_now or st.session_state.get(f"select_video_{_vid}", False)
                        st.button("✓ Selected" if _is_sel else "➕ Add", key=f"sim_add_{video_id}_{_vid}",
                                  disabled=_is_sel, on_click=_add_video_to_selection, args=(_vid,),
                                  use_container_width=True)

        if st.session_state["show_range_picker"].get(video_id):
            with st.container(border=True):
                _action = render_range_picker(
                    video_id, catalogues[video_id]["source_path"], duration, current_range,
                    key="range", rerun=rerun_fragment,
                    suggestions=[("🎯 Auto Range (match audio length, highest motion)",
                                  lambda: find_best_motion_window(video_id, track["duration_sec"]))],
                )
                if _action:
                    _kind, _rng = _action
                    if _kind == "apply":
                        st.session_state["video_time_ranges"][video_id] = _rng
                    elif lib_range:
                        # Explicit "full video" override, so the library range isn't used either.
                        st.session_state["video_time_ranges"][video_id] = None
                    else:
                        st.session_state["video_time_ranges"].pop(video_id, None)
                    st.session_state.pop(f"range_slider_{video_id}", None)
                    rerun_fragment()
                if has_override and lib_range:
                    if st.button(f"📚 Reset to library range ({format_mmss(lib_range[0])}–{format_mmss(lib_range[1])})",
                                 key=f"range_reset_lib_{video_id}"):
                        st.session_state["video_time_ranges"].pop(video_id, None)
                        st.session_state.pop(f"range_slider_{video_id}", None)
                        rerun_fragment()

    # committed_selected_videos is what actually survives a round trip to another section
    # (plain session_state, not a widget), so it's what default_checked restores from —
    # NOT a static True/False by row type, which would silently reset every tick the moment
    # the underlying select_video_* widget state gets discarded for having gone unrendered.
    _committed = st.session_state.get("committed_selected_videos", [])
    for _v in sorted_top_matches:
        render_video_row(_v["video_id"], recommended=True,
                         default_checked=(_v["video_id"] in _committed))

    if _new_ids:
        _not_ticked = [v for v in _new_ids if not st.session_state.get(f"select_video_{v}", False)]
        _nc = st.columns([6, 2])
        _nc[0].caption(f"🆕 **{len(_new_ids)} video(s) added in the last {CStore.NEW_VIDEO_DAYS} days:** "
                       + ", ".join(_new_ids[:6]) + ("…" if len(_new_ids) > 6 else "")
                       + ("  — all selected" if not _not_ticked else ""))
        _nc[1].button(f"🆕 Add new videos ({len(_not_ticked)})", on_click=_add_new_videos, args=(_not_ticked,),
                      disabled=not _not_ticked, use_container_width=True, key="add_new_videos",
                      help="Tick every video added recently, adding any that aren't in the top list. "
                           "New videos are also listed first, marked 🆕, in the dropdown below.")

    st.session_state["extra_videos"] = info["extra_rows"]
    _other_options = sorted(info["other_options"], key=lambda v: v not in _new_ids)   # 🆕 first, order kept
    st.multiselect(
        "➕ Add other videos from the library", _other_options, key="extra_videos",
        format_func=lambda vid: (f"🆕 " if vid in _new_ids else "") + f"{vid}  ·  {format_mmss(get_video_duration(vid))}",
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
        st.success(f"{len(selected_now)} video(s) selected. Go on to **🎬 Matching & Export** (below) when ready.")


# ---------------------------------------------------------------------------
# Rhythm Engine: an alternative segmentation method, ported from a reference
# state-machine design. Instead of detecting individual cut POINTS, it works
# at bar granularity — each bar gets a cutting STYLE: CINEMATIC (slow,
# multi-bar segments) or FAST (one segment per bar), from that bar's
# smoothed energy. Phrase-locking commits to a style for several bars so it
# doesn't flicker; FAST has a maximum run length and a cooldown so rapid
# cutting can't run on for too long or restart straight away.
# ---------------------------------------------------------------------------


def clips_for_intensity(intensity: float, min_clips: int, max_clips: int,
                        ramp_start: float, ramp_end: float, vary: bool, rng) -> int:
    """core.segmentation.clips_for_intensity with the 'Skip these counts' setting."""
    return CSEG.clips_for_intensity(intensity, min_clips, max_clips, ramp_start, ramp_end, vary, rng,
                                    skip=st.session_state.get("clip_count_skip"))


def allowed_clip_count(n: int, min_clips: int, max_clips: int) -> int:
    """core.segmentation.allowed_clip_count with the 'Skip these counts' setting."""
    return CSEG.allowed_clip_count(n, min_clips, max_clips, skip=st.session_state.get("clip_count_skip"))


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
    "segmentation_method": "Smart (self-tuning)",   # = SEG_SMART
    "audio_style": "Balanced",    # cutting pace preset; dials follow the track unless Custom
    "cut_style": "custom",        # 🎨 cutting style (core/styles.py); "custom" = choose the method yourself
    "cut_style_pace": 0,
    "sc_focus": "Balanced",       # Scored cuts: "What to cut on" mix
    "beats_per_bar": 4,
    "change_window_secs": 2.0,
    "change_threshold": 0.12,
    "min_segment_sec": 2.0,
    "max_segment_sec": 8.0,
    "react_to_hits": True,
    "hit_threshold": 0.9,
    "snap_secs": 0.3,
    # Segmentation — Rhythm Engine
    "fast_energy_thresh": 0.75,
    "fast_max_bars": 8,
    "fast_cooldown_bars": 4,
    "phrase_lock_bars": 4,
    "cinematic_bars": 4,
    "dynamic_priority_override": False,
    "override_delta_thresh": 0.15,
    "override_clear_secs": 1.0,
    # Segmentation — Scored cuts
    **CS.DEFAULTS,
    # Segmentation — Smart
    **SM.DEFAULTS,
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
    "matching_mode": "Auto",
    "sequential_mode": False,
    "use_story_map": True,
    "story_reserve": True,
    "intro_outro_exclusive": False,
    "clip_count_skip": [],
    # Auto-fill weights (used in Matching & Export, stored here for persistence)
    # Defaults = autofill.PRESETS["Balanced"] (projects keep the values they were saved with).
    "weight_shape": 2.0,
    "weight_random": 0.5,
    "avoid_repeat": True,
    "weight_spread": 1.5,
    "weight_motion": 2.0,
    "weight_position": 1.0,
    "autofill_seed": 42,
    "autofill_v2": True,          # ✨ Improved auto-fill (autofill.py); off = the original scoring
    "autofill_preset": "Balanced",
}

# Weight keys that live on the Matching & Export page — Streamlit can purge
# these when the page reruns while that section isn't active (the widgets
# don't render, so their session_state entries can disappear). We shadow them
# exactly like audio_settings_shadow so they survive round-trips to other
# sections and come back at the user's last-set values rather than defaults.
WEIGHT_KEYS = [
    "weight_shape", "weight_random", "avoid_repeat",
    "weight_spread", "weight_motion", "weight_position", "autofill_seed",
    "autofill_v2", "autofill_preset",
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
    "sequential_mode", "prefer_intro_outro", "intro_outro_exclusive", "use_story_map", "story_reserve",
    *AUDIO_SETTINGS_KEYS,
    "split_screen_enabled", "min_clips", "max_clips", "clip_count_skip", "density_contrast",
    "vary_count", "count_seed", "min_clip_len_sec",
    "weight_shape", "weight_random", "avoid_repeat", "weight_spread", "weight_motion",
    "weight_position", "autofill_seed", "autofill_v2", "autofill_preset",
    "ranking_weights", "ranking_experimental", "random_top_seed",
    "committed_selected_videos", "committed_tag_filter", "known_extra_videos", "extra_videos",
    "adv_current_block", "chor_current_block",
    "intro_block_indices", "outro_block_indices",
    "story_map",
    "cut_edits",
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
    # {video_id: (start, end) | None} -> {video_id: [start, end] | None}
    # Overrides only — library ranges live in library_meta.json. None = "full video".
    data["video_time_ranges"] = {
        vid: (list(rng) if rng else None)
        for vid, rng in st.session_state.get("video_time_ranges", {}).items()
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
    if "cut_edits" not in data:
        st.session_state.pop("cut_edits", None)       # a project without hand edits
    _bump_cut_edit_rev()                               # the ✂️ editor reloads its cuts
    # Projects saved before cutting paces existed keep their exact dials.
    if "audio_style" not in data:
        st.session_state["audio_style"] = AUDIO_STYLE_CUSTOM
    st.session_state["_audio_profile_for"] = _audio_profile_key(st.session_state.get("track_id"))

    if "ramp_range" in data:
        try:
            st.session_state["ramp_range"] = tuple(data["ramp_range"])
        except (TypeError, ValueError):
            warnings.append("Couldn't restore the intensity ramp range — left at its current value.")

    try:
        st.session_state["video_time_ranges"] = {
            vid: (tuple(rng) if rng else None) for vid, rng in data.get("video_time_ranges", {}).items()
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


def _canon(value):
    """Lists of pairs (saved sets, e.g. excluded clips) in a fixed order — sets have none."""
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [_canon(v) for v in value]
        if items and all(isinstance(v, list) for v in items):
            items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return items
    return value


def _normalise(data: dict) -> dict:
    """JSON round-trip (tuples → lists, int keys → str, sets → sorted lists) so saved and
    current compare like for like."""
    return _canon(json.loads(json.dumps(data, default=list)))


def _remember_project(name: str, data: dict) -> None:
    """Track this project as the one being worked on (for prefill and 'unsaved changes')."""
    st.session_state["_loaded_project"] = {"name": name, "data": _normalise(data)}


def _project_changes(loaded: dict) -> int:
    """How many saved areas of the remembered project differ from the current session.
    Only what the file holds is compared — settings it never had (e.g. defaults added
    since) don't count — and for tick boxes only the ticked ones matter."""
    saved = loaded["data"]
    current = _normalise(_build_project_save_dict())
    changed = 0
    for key, value in saved.items():
        if key.startswith("_"):
            continue
        now = current.get(key)
        if key.endswith("_pick_state"):
            value = {k for k, v in (value or {}).items() if v}
            now = {k for k, v in (now or {}).items() if v}
        if now != value:
            changed += 1
    return changed


def _do_save_project(name: str) -> None:
    out_path, drive_err = save_project(name)
    _remember_project(out_path.stem, _build_project_save_dict())
    st.session_state["_prefill_project_name"] = out_path.stem
    if drive_err:
        notice = ("warning", f"Saved locally but Drive upload failed: {drive_err}")
    elif _DRIVE_SYNC_AVAILABLE:
        notice = ("success", f"Saved and uploaded to Drive: {out_path.name}")
    else:
        notice = ("success", f"Saved as {out_path.name}")
    st.session_state["_project_notices"] = [notice]
    st.rerun()


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


def _dial_value(k):
    """A dial's current value (its default if it hasn't been set)."""
    return st.session_state.get(k, DIAL_DEFAULTS.get(k))


def _method_cfg(method: str, settings: dict, bpb: int) -> dict:
    return CAT._method_cfg(method, settings, bpb, lookup=_dial_value)


def auto_tune_parameters(track: dict, min_clips: int, max_clips: int, method: str = None,
                         style_name: str = "Balanced", beats_per_bar: int = None) -> tuple:
    """core.autotune.auto_tune_parameters for the current method / beats per bar."""
    method = method or st.session_state.get("segmentation_method", SEG_ADAPTIVE)
    bpb = int(beats_per_bar or st.session_state.get("beats_per_bar", 4))
    return CAT.auto_tune_parameters(track, min_clips, max_clips, method=method, style_name=style_name,
                                    beats_per_bar=bpb, lookup=_dial_value)


def _audio_profile_key(track_id) -> tuple:
    return (track_id, st.session_state.get("segmentation_method"), st.session_state.get("audio_style"),
            int(st.session_state.get("beats_per_bar", 4)))


def apply_audio_profile(track: dict, track_id: str, report: bool = False) -> None:
    """Set this track's segmentation dials for the current method and cutting
    pace (and split-screen shape). Keeps the app-scope copy in step."""
    style = st.session_state.get("audio_style", "Balanced")
    if style not in AUDIO_STYLES:
        style = "Balanced"
    mn = int(st.session_state.get("min_clips", 1))
    mx = max(int(st.session_state.get("max_clips", 4)), mn + 1, 2)
    settings, why = auto_tune_parameters(track, mn, mx, style_name=style)
    for k, v in settings.items():
        try:
            st.session_state[k] = v
        except Exception:
            pass   # a widget already drawn this run keeps its value until the next run
    appscope = st.session_state.get("_audio_appscope", {})
    appscope.update({k: v for k, v in settings.items() if k in AUDIO_SETTINGS_KEYS})
    appscope["audio_style"] = st.session_state.get("audio_style")
    st.session_state["_audio_appscope"] = appscope
    st.session_state["_audio_profile_for"] = _audio_profile_key(track_id)
    if report:
        st.session_state["auto_tune_report"] = why


def ensure_audio_profile(track: dict, track_id: str) -> None:
    """With a cutting pace chosen (not Custom), the dials follow the track,
    method and beats per bar: re-tuned whenever any of them changes."""
    if st.session_state.get("audio_style", "Balanced") == AUDIO_STYLE_CUSTOM:
        return
    if st.session_state.get("_audio_profile_for") != _audio_profile_key(track_id):
        apply_audio_profile(track, track_id)


def _run_auto_tune():
    """🎯 Auto-detect: re-tune the current method for this track at the chosen
    pace (Balanced if the pace is Custom)."""
    if st.session_state.get("audio_style") not in AUDIO_STYLES:
        st.session_state["audio_style"] = "Balanced"
    t = load_track(st.session_state["track_id"])
    st.session_state["split_screen_enabled"] = True
    st.session_state["max_clips"] = min(MAX_SIMULTANEOUS_CLIPS, max(int(st.session_state.get("max_clips", 4)),
                                                                   int(st.session_state.get("min_clips", 1)) + 1, 2))
    apply_audio_profile(t, st.session_state["track_id"], report=True)


# ---------------------------------------------------------------------------
# Autosave & restore — your work survives the app restarting (a crash, running
# out of memory, a deploy). Every change is written to a recovery file for you
# (PROJECTS_DIR/.autosave/<email>.json — separate from your saved projects).
# A new session moves that file aside and offers to restore it.
# ---------------------------------------------------------------------------

AUTOSAVE_DIR = PROJECTS_DIR / ".autosave"


def _autosave_path(kind: str = "") -> Path:
    who = re.sub(r"[^A-Za-z0-9._-]", "_", str(getattr(st.user, "email", None) or "user"))
    return AUTOSAVE_DIR / f"{who}{kind}.json"


def _progress_summary(state: dict) -> dict:
    def blocks(key):
        return sum(1 for v in (state.get(key) or {}).values() if v)
    return {"track_id": state.get("track_id"), "mode": state.get("matching_mode"),
            "videos": len(state.get("committed_selected_videos") or []),
            "auto_blocks": len(state.get("segment_exclusions") or {}),
            "manual_blocks": blocks("adv_confirmed"), "chor_blocks": blocks("chor_confirmed")}


def autosave_now() -> None:
    """Write this session's state to the recovery file — only when it changed."""
    if not st.session_state.get("_autosave_started"):
        return
    try:
        state = _normalise(_build_project_save_dict())
        state.pop("_saved_at", None)
        blob = json.dumps(state, sort_keys=True, default=str)
    except Exception:
        return
    digest = hashlib.sha1(blob.encode()).hexdigest()
    if digest == st.session_state.get("_autosave_hash"):
        return
    payload = {
        "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": _progress_summary(state),
        "loaded_project": st.session_state.get("_loaded_project"),
        "project_name": (st.session_state.get("project_save_name") or "").strip() or None,
        "state": state,
    }
    try:
        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = _autosave_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(path)
        st.session_state["_autosave_hash"] = digest
    except OSError:
        pass


def _start_autosave_session() -> None:
    """First run of a new session: the recovery file left by the last session is
    moved aside (so this session's autosave can't overwrite it) and offered."""
    if st.session_state.get("_autosave_started"):
        return
    st.session_state["_autosave_started"] = True
    path, recover = _autosave_path(), _autosave_path(".recover")
    try:
        if path.exists():
            data = json.loads(path.read_text())
            s = data.get("summary") or {}
            worth_it = s.get("track_id") and (s.get("videos") or s.get("manual_blocks") or s.get("chor_blocks")
                                              or s.get("auto_blocks"))
            if worth_it:
                path.replace(recover)
            else:
                path.unlink()
    except (OSError, ValueError):
        pass


def _restore_choice(choice: str) -> None:
    recover = _autosave_path(".recover")
    if choice == "restore":
        try:
            st.session_state["_restore_autosave"] = json.loads(recover.read_text())
        except (OSError, ValueError):
            st.session_state["_project_notices"] = [("error", "Couldn't read the unsaved work.")]
            return
    try:
        recover.replace(_autosave_path(".discarded"))   # one step of undo, just in case
    except OSError:
        pass


_start_autosave_session()
st.session_state["_before_reset"] = autosave_now   # used by the sidebar's 🧹 Free memory

_restore = st.session_state.pop("_restore_autosave", None)
if _restore:
    _restore_warnings = _apply_project_load_dict(_restore.get("state") or {})
    _appscope = st.session_state.get("_audio_appscope", {})
    _appscope.update({k: st.session_state[k] for k in AUDIO_SETTINGS_KEYS if k in st.session_state})
    st.session_state["_audio_appscope"] = _appscope
    if _restore.get("loaded_project"):
        st.session_state["_loaded_project"] = _restore["loaded_project"]
    if _restore.get("project_name"):
        st.session_state["_prefill_project_name"] = _restore["project_name"]
    st.session_state["_project_notices"] = ([("warning", _w) for _w in _restore_warnings]
                                            + [("success", "♻️ Unsaved work restored.")])
    st.rerun()

_recover_file = _autosave_path(".recover")
if _recover_file.exists():
    try:
        _rec = json.loads(_recover_file.read_text())
        _rs = _rec.get("summary") or {}
        _when = datetime.datetime.fromisoformat(_rec["saved_at"]).strftime("%a %d %b, %H:%M")
    except (OSError, ValueError, KeyError):
        _rec, _rs, _when = None, {}, "?"
    if _rec:
        _bits = [f"track **{_rs.get('track_id')}**", str(_rs.get("mode") or "")]
        if _rs.get("chor_blocks"):
            _bits.append(f"{_rs['chor_blocks']} Choreography block(s) confirmed")
        if _rs.get("manual_blocks"):
            _bits.append(f"{_rs['manual_blocks']} Manual block(s) confirmed")
        if _rs.get("videos"):
            _bits.append(f"{_rs['videos']} video(s) selected")
        if _rec.get("loaded_project"):
            _bits.append(f"project **{_rec['loaded_project'].get('name')}**")
        with st.container(border=True):
            st.markdown(f"♻️ **Unsaved work from {_when}** — the app restarted before it was saved.  \n"
                        + " · ".join(b for b in _bits if b))
            _rc = st.columns([1, 1, 4])
            _rc[0].button("♻️ Restore it", type="primary", on_click=_restore_choice, args=("restore",),
                          use_container_width=True, key="autosave_restore")
            _rc[1].button("Discard", on_click=_restore_choice, args=("discard",),
                          use_container_width=True, key="autosave_discard")

# Media Library → Renders → "Reopen in Planner": load that render's saved project
# state (or the settings recovered from its plan) exactly like 📂 Load Project.
_reopen = st.session_state.pop("_reopen_project", None)
if _reopen:
    _reopen_warnings = _apply_project_load_dict(_reopen["data"])
    # Make the app-scope audio snapshot agree with what was just loaded.
    _appscope = st.session_state.get("_audio_appscope", {})
    _appscope.update({k: st.session_state[k] for k in AUDIO_SETTINGS_KEYS if k in st.session_state})
    st.session_state["_audio_appscope"] = _appscope
    st.session_state["_reopen_notice"] = {**{k: v for k, v in _reopen.items() if k != "data"},
                                          "warnings": _reopen_warnings}
    st.session_state.pop("_loaded_project", None)   # a render's state isn't that saved project's
    if _reopen.get("project_name"):
        st.session_state["_prefill_project_name"] = _reopen["project_name"]
    st.rerun()
_reopen_notice = st.session_state.pop("_reopen_notice", None)
if _reopen_notice:
    if _reopen_notice.get("kind") == "full":
        st.success(f"↩️ Reopened from render **{_reopen_notice['label']}** — settings and matches "
                   f"are as they were when it was rendered.")
    else:
        st.info(f"↩️ Reopened from render **{_reopen_notice['label']}** — this render was made before full "
                f"project state was saved, so only its settings were restored. Matches are rebuilt from "
                f"them and may differ from the render.")
    for _w in _reopen_notice.get("warnings") or []:
        st.warning(_w)

# The project you loaded (or last saved) is remembered: its name is filled in for
# saving back, and the sidebar shows when there are changes since its last save.
if "_prefill_project_name" in st.session_state:   # set by a load/reopen — applied before the box is drawn
    st.session_state["project_save_name"] = st.session_state.pop("_prefill_project_name")

autosave_now()


def _new_project() -> None:
    """🆕 New project: back to a blank planner (every setting at its default, no
    track work, no project loaded). What was on screen is kept as a one-step
    backup so it can be brought back."""
    autosave_now()
    try:
        cur = _autosave_path()
        if cur.exists():
            cur.replace(_autosave_path(".before_new"))
    except OSError:
        pass
    from page_setup import session_keep_keys
    keep = session_keep_keys() | {"_autosave_started"}
    name = (st.session_state.get("new_project_name") or "").strip()
    for k in list(st.session_state.keys()):
        if k not in keep:
            try:
                del st.session_state[k]
            except Exception:
                pass
    st.cache_data.clear()
    if name:
        st.session_state["_prefill_project_name"] = sanitize_filename(name)
    st.session_state["_new_project_notice"] = name or True


def _undo_new_project() -> None:
    backup = _autosave_path(".before_new")
    try:
        st.session_state["_restore_autosave"] = json.loads(backup.read_text())
        backup.unlink()
    except (OSError, ValueError):
        st.session_state["_project_notices"] = [("error", "Couldn't find what you had before.")]


# 📁 Project bar — at the top of the page, above the steps (it used to be in the sidebar).
_pbar = st.container(border=True)
_pb = _pbar.columns([3, 1.3, 1, 1.2])
_new_notice = st.session_state.pop("_new_project_notice", None)
_new_loaded = st.session_state.get("_loaded_project")
_new_loaded_changes = _project_changes(_new_loaded) if _new_loaded else 0
_loaded = _new_loaded
with _pb[0]:
    if _loaded:
        _n_changed = _new_loaded_changes
        if _n_changed:
            st.markdown(f"📁 **{_loaded['name']}**  ·  :orange[unsaved changes ({_n_changed} area"
                        f"{'' if _n_changed == 1 else 's'})]")
        else:
            st.markdown(f"📁 **{_loaded['name']}**  ·  :green[all changes saved]")
    else:
        st.markdown("📁 **Unsaved project**  ·  :gray[save it to keep it and come back later]")
if _loaded and _new_loaded_changes:
    if _pb[1].button(f"💾 Save changes", type="primary", use_container_width=True, key="project_quick_save",
                     help=f"Save to '{_loaded['name']}'."):
        _do_save_project(_loaded["name"])
with _pb[2].popover("🆕 New", use_container_width=True):
    st.caption("Start again from a blank planner — default settings, no track work, no project "
               "loaded. Saved projects aren't touched, and you can bring back what's on screen "
               "now straight afterwards.")
    if _new_loaded and _new_loaded_changes:
        st.warning(f"**{_new_loaded['name']}** has unsaved changes — save it first if you want to keep them.")
    st.text_input("Name (optional)", key="new_project_name", placeholder="my-new-compilation",
                  help="Filled in as the project name, ready for 💾 Save.")
    st.button("🆕 Start new project", type="primary", on_click=_new_project, use_container_width=True,
              key="new_project_go")
with _pb[3].popover("💾 Save / 📂 Open", use_container_width=True):
    st.caption("Saves every setting plus matching progress (Auto, Manual step-through, and "
               "Choreography all at once) so you can close this and pick up again later.")
    _save_name = st.text_input("Project name", key="project_save_name", placeholder="my-compilation",
                               help="Filled in with the project you loaded — save to update it, or type a "
                                    "new name to save a copy.")
    _overwrites = _save_name.strip() and (PROJECTS_DIR / f"{sanitize_filename(_save_name.strip())}.json").exists()
    if st.button("💾 Save Project" + (" (update)" if _overwrites else ""), disabled=not _save_name.strip(),
                 use_container_width=True):
        _do_save_project(_save_name.strip())

    st.divider()
    _existing_projects = list_saved_projects()
    if _existing_projects:
        _load_choice = st.selectbox("Open project", ["—"] + _existing_projects, key="project_load_choice")
        if _load_choice != "—" and st.button("📂 Open Project", use_container_width=True):
            _path = PROJECTS_DIR / f"{_load_choice}.json"
            _load_warnings = load_project(_path)
            _remember_project(_load_choice, json.loads(_path.read_text()))
            st.session_state["_prefill_project_name"] = _load_choice
            st.session_state["_project_notices"] = ([("warning", _w) for _w in _load_warnings]
                                                    + [("success", f"Opened '{_load_choice}'.")])
            st.rerun()
    else:
        st.caption("No saved projects yet.")
if _new_notice:
    _pbar.success("🆕 New project started" + (f" — **{_new_notice}**" if isinstance(_new_notice, str) else "")
                  + ". Pick a track and videos to begin.")
    if _autosave_path(".before_new").exists():
        _pbar.button("↩️ Bring back what I had", on_click=_undo_new_project, key="new_project_undo")
for _kind, _msg in st.session_state.pop("_project_notices", []):
    getattr(_pbar, _kind)(_msg)

# A loaded project's track_id can reference a track that no longer exists (renamed/deleted
# audio file) — the Track selectbox below would error if its backing value isn't among the
# current options, so drop it here and let the widget fall back to its own first option.
if st.session_state.get("track_id") not in tracks:
    st.session_state.pop("track_id", None)
    if st.session_state.get("_project_load_pending_modes"):
        st.warning("The loaded project's track is no longer available — defaulted to the first track.")


# 🔄 Sync now lives in the shared sidebar (page_setup → ☁️ Google Drive).

# Media Library → Audio → 🎭 Edit story map: open that track on the Story map step.
_open_story = st.session_state.pop("_open_story_for", None)
if _open_story in tracks:
    st.session_state["track_id"] = _open_story
    st.session_state["ui_section"] = "🎭 1. Story map"
# The track is picked at the top of the 🎵 Audio step (render_track_picker); it's
# kept like the other step settings (STEP_WIDGET_KEYS) so every step can use it.
if tracks and st.session_state.get("track_id") not in tracks:
    st.session_state["track_id"] = tracks[0]
st.session_state["track_id"] = st.session_state["track_id"]
track_id = st.session_state["track_id"]
# A chosen cutting pace keeps the dials tuned to this track (see ensure_audio_profile) —
# done here, before the split-screen widgets (🎵 Audio step) are drawn, so it may set them.
ensure_audio_profile(load_track(track_id), track_id)

# ---------------------------------------------------------------------------
# Settings that live inside ONE step (split screen on 🎵 Audio; matching mode,
# order and intro/outro on 🎬 Matching). Their widgets are only drawn on that
# step, and Streamlit drops a widget's value on any run where the widget isn't
# drawn — so every run re-stores each one here, which keeps it (and keeps it in
# saved projects / the recovery file) whichever step you're on. Everything else
# reads them from session_state, never from the widget's return value.
# ---------------------------------------------------------------------------
STEP_WIDGET_KEYS = [
    "matching_mode", "max_shape_matches", "max_chor_videos", "prefer_intro_outro", "intro_outro_exclusive",
    "sequential_mode", "allow_same_video", "use_story_map", "story_reserve",
    "split_screen_enabled", "min_clips", "max_clips", "clip_count_skip", "ramp_range", "density_contrast",
    "vary_count", "count_seed", "min_clip_len_sec",
]
for _k in STEP_WIDGET_KEYS:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]
if st.session_state["max_clips"] < st.session_state["min_clips"]:
    st.session_state["max_clips"] = st.session_state["min_clips"]

matching_mode = st.session_state["matching_mode"]
max_shape_matches = st.session_state["max_shape_matches"]
max_chor_videos = st.session_state["max_chor_videos"]
prefer_intro_outro = bool(st.session_state["prefer_intro_outro"])
sequential_mode = bool(st.session_state["sequential_mode"])
allow_same_video = bool(st.session_state["allow_same_video"])
split_screen_enabled = bool(st.session_state["split_screen_enabled"])


def render_split_screen_settings() -> None:
    """🔀 Split screen & clip length — on the 🎵 Audio step (they set how many clips
    each block gets, which the audio preview chart shows)."""
    on = bool(st.session_state.get("split_screen_enabled"))
    summary = (f"{st.session_state['min_clips']}–{st.session_state['max_clips']} clips per block"
               if on else "off — one clip per block")
    with st.expander(f"🔀 Split screen & clip length  ·  {summary}", expanded=False):
        c = st.columns([1, 1])
        with c[0]:
            st.toggle("Split-screen for high-energy blocks", key="split_screen_enabled",
                      help="Louder blocks show several clips at once (the VSE add-on's split-screen layouts).")
            st.slider("Min simultaneous clips", 1, MAX_SIMULTANEOUS_CLIPS, key="min_clips", disabled=not on)
            st.slider("Max simultaneous clips", min_value=st.session_state["min_clips"],
                      max_value=MAX_SIMULTANEOUS_CLIPS, key="max_clips", disabled=not on,
                      help="Up to 8, using the same layouts as the VSE add-on.")
            between = list(range(int(st.session_state["min_clips"]) + 1, int(st.session_state["max_clips"])))
            if between:
                # Keep only counts still between min and max (the widget errors on a value it can't offer).
                if "clip_count_skip" in st.session_state:
                    st.session_state["clip_count_skip"] = [x for x in (st.session_state["clip_count_skip"] or [])
                                                           if x in between]
                st.pills(
                    "Skip these clip counts", between, selection_mode="multi", key="clip_count_skip",
                    disabled=not on, format_func=lambda n: f"{n} clips",
                    help="Blocks never use a count you pick here — e.g. min 1, max 4, skip 3 → blocks use 1, 2 "
                         "or 4 clips. A block that would have used a skipped count takes the nearest allowed "
                         "one (the smaller when it's halfway).")
        with c[1]:
            st.slider("Intensity ramp range", 0.0, 1.0, step=0.05, key="ramp_range", disabled=not on,
                      help="At or below left value: min clips. At or above right: max clips.")
            st.slider("Density contrast", 0.25, 3.0, step=0.05, key="density_contrast", disabled=not on,
                      help="Above 1 = only the loudest sections reach high clip counts; below 1 = rises more evenly.")
            vary = st.checkbox("Vary clip count within range (seeded)", key="vary_count", disabled=not on)
            st.number_input("Clip count seed", 0, 99999, key="count_seed", disabled=not (on and vary))
        st.slider("Minimum clip length (s)", 0.2, 3.0, step=0.1, key="min_clip_len_sec",
                  help="If footage left to fill a slot is shorter than this, extend the previous clip instead.")


def render_matching_mode_settings() -> None:
    """🎛️ Mode tab of the 🎬 Matching step's settings."""
    c = st.columns([2, 1])
    with c[0]:
        mode = st.radio(
            "How clips are chosen", ["Auto", "Manual step-through", "Choreography"], key="matching_mode",
            horizontal=True,
            help="Auto: the algorithm picks the best-matching clip per segment automatically.  "
                 "Manual step-through: pick from the best shape-matched candidates one block at a time.  "
                 "Choreography: one column per source video — see the auto pick for each, then override with "
                 "any available clip from that video via a timecode dropdown.")
    with c[1]:
        if mode == "Manual step-through":
            st.number_input("Candidates shown per block", min_value=1, max_value=30, key="max_shape_matches",
                            help="How many of the best-scoring clips are offered per block.")
        elif mode == "Choreography":
            st.number_input("Videos shown per block", min_value=2, max_value=6, key="max_chor_videos",
                            help="How many source-video columns to show per block. 4 is comfortable, 6 is "
                                 "the practical limit.")
    c = st.columns(3)
    c[0].toggle("Keep each video's clips in order", key="sequential_mode",
                help="A video's scenes are only ever used in their original chronological order (the VSE "
                     "add-on's 'Keep Sequential Order') — other videos can still interleave around them.")
    c[1].toggle("Allow one video twice in a split-screen block", key="allow_same_video",
                help="Lets a split-screen block use more than one clip from the same video.")
    has_map = bool(current_story_map(st.session_state.get("track_id")))
    sc = st.columns(3)
    sc[0].toggle("🎭 Use the story map", key="use_story_map", disabled=not has_map,
                 help="Each block prefers clips marked with its story map part's role (🎬 Intro, 🌙 Slow, "
                      "💥 Action, 🏁 Outro, your own) — falling back to normal picks where a video has none. "
                      "In order (sequential) mode a role clip is only taken within the normal skip allowance."
                      + ("" if has_map else " This track has no story map yet (🎭 1. Story map)."))
    sc[1].toggle("Keep role clips for their parts", key="story_reserve", disabled=not has_map,
                 help="Automatic picks outside a role's parts avoid clips marked with that role, so they're "
                      "still there when their part comes — unless nothing else is left.")
    if has_map and st.session_state.get("use_story_map", True):
        sc[2].caption("🎭 The story map sets each block's role — the Intro / Outro tab isn't used.")
    c[2].toggle("Prefer Intro/Outro-tagged clips", key="prefer_intro_outro",
                help="Intro/Outro blocks (🎬 Intro/Outro tab — default: first and last block) draw from a "
                     "video's clips marked Intro / Outro in the Reviewer, for videos that have any; a video "
                     "with none offers its normal clips there.")


# ---------------------------------------------------------------------------
# Compute: track + video-level energy matching
# ---------------------------------------------------------------------------

track = load_track(track_id)


def render_track_picker() -> None:
    """🎵 Track — first thing on the Audio step."""
    c = st.columns([2, 3])
    c[0].selectbox("🎵 Track", tracks, key="track_id",
                   help="The music this compilation is cut to. Changing it re-tunes the cutting settings "
                        "if a cutting pace is chosen; each track has its own story map.")
    try:
        info = f"{format_mmss(track['duration_sec'])}  ·  {float(track.get('bpm') or 0):.0f} BPM"
    except (TypeError, ValueError):
        info = format_mmss(track["duration_sec"])
    c[1].markdown(f"<div style='padding-top:2.1rem;opacity:.7'>{info}</div>", unsafe_allow_html=True)

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

# Sessions / projects from before the 🎭 Story map step: same section, new name.
_old_section = st.session_state.get("ui_section")
if _old_section is not None and _old_section not in SECTIONS:
    st.session_state["ui_section"] = next((x for x in SECTIONS if x[:1] == str(_old_section)[:1]), SECTION_AUDIO)
def _goto(section: str) -> None:
    st.session_state["ui_section"] = section


def _step_status() -> list:
    """One short line per step, shown under the step bar."""
    method = st.session_state.get("segmentation_method", SEG_SMART)
    audio = f"{track_id} · " + SEG_LABELS.get(method, method).split("—")[0].strip()
    if st.session_state.get("split_screen_enabled"):
        audio += f" · split {st.session_state['min_clips']}–{st.session_state['max_clips']}"
    regions = current_story_map(track_id)
    story = (f"{sum(1 for r in regions if r['role'])} part(s) with a role" if regions else "optional — not set")
    n_vid = len(st.session_state.get("committed_selected_videos", []))
    videos = f"{n_vid} video(s) selected" if n_vid else "none selected yet"
    mode = st.session_state.get("matching_mode", "Auto")
    done = {"Manual step-through": len(st.session_state.get("adv_confirmed", {}) or {}),
            "Choreography": len(st.session_state.get("chor_confirmed", {}) or {})}.get(mode)
    match = mode + (f" · {done} block(s) confirmed" if done else "")
    if sequential_mode:
        match += " · in order"
    return [audio, story, videos, match]


def step_nav(back: str = None, nxt: str = None) -> None:
    """⬅ / ➡ buttons at the foot of a step."""
    st.divider()
    c = st.columns([1, 2, 1])
    if back:
        c[0].button(f"⬅ {back.split(' ', 1)[1]}", on_click=_goto, args=(back,), use_container_width=True,
                    key=f"nav_back_{back}")
    if nxt:
        c[2].button(f"{nxt.split(' ', 1)[1]} ➡", on_click=_goto, args=(nxt,), type="primary",
                    use_container_width=True, key=f"nav_next_{nxt}")


# The step bar: the radio (same key and values as always — saved sessions and the
# recovery file keep working) styled as steps, with a status line under each.
st.markdown("""<style>
.st-key-stepnav .stElementContainer, .st-key-stepnav .stRadio { width: 100% !important; }
.st-key-stepnav [role="radiogroup"] { display: grid !important; grid-template-columns: repeat(4, 1fr); gap: .5rem;
  width: 100%; }
.st-key-stepnav [role="radiogroup"] > div { margin: 0 !important; border-radius: .6rem;
  border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.06); }
.st-key-stepnav [role="radiogroup"] > div:hover { border-color: rgba(255,75,75,.6); }
.st-key-stepnav [role="radiogroup"] > div[data-selected="true"] { border-color: #ff4b4b;
  background: rgba(255,75,75,.10); box-shadow: inset 0 -3px 0 #ff4b4b; }
.st-key-stepnav label[data-testid="stRadioOption"] { width: 100%; padding: .6rem .9rem; margin: 0; cursor: pointer; }
.st-key-stepnav label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-stepnav [data-testid="stMarkdownContainer"] p { font-weight: 600; margin: 0; }
.st-key-stepstatus { margin-top: -.6rem; }
.st-key-stepstatus [data-testid="stCaptionContainer"] { padding: 0 .9rem; }
</style>""", unsafe_allow_html=True)
with st.container(key="stepnav"):
    ui_section = st.radio(
        "Section", SECTIONS,
        horizontal=True, key="ui_section", label_visibility="collapsed",
    )
with st.container(key="stepstatus"):
    for _col, _txt in zip(st.columns(4), _step_status()):
        _col.caption(_txt)

if ui_section == SECTION_AUDIO:
    render_track_picker()
    _ac = st.columns([1, 2])
    _ac[0].button("🎯 Auto-detect settings for this track", on_click=_run_auto_tune, type="primary",
                  use_container_width=True,
                  help="Analyses this track's tempo, energy and hits, then sets the cutting and split-screen "
                       "settings. Adjust anything afterwards to fine-tune.")
    _ac[1].caption("Pick how the music is cut into blocks and check where the cuts land. Split screen "
                   "(below) sets how many clips each block shows.")
    render_split_screen_settings()
    render_audio_settings_section(track)
    step_nav(nxt=SECTION_STORY)
    _record_page_run()
    st.stop()

if ui_section == SECTION_STORY:
    render_story_map_section(track)
    step_nav(back=SECTION_AUDIO, nxt=SECTION_VIDEOS)
    _record_page_run()
    st.stop()

if ui_section == SECTION_VIDEOS:
    _n_lib_ranges = len(library_ranges())
    if _n_lib_ranges or st.session_state["video_time_ranges"]:
        _rc = st.columns([3, 1])
        _rc[0].caption(" · ".join(x for x in (
            f"📚 {_n_lib_ranges} video(s) have a Media Library range" if _n_lib_ranges else "",
            f"🎚️ {len(st.session_state['video_time_ranges'])} have a range set for this project"
            if st.session_state["video_time_ranges"] else "") if x))
        if st.session_state["video_time_ranges"] and _rc[1].button("Reset project ranges",
                                                                    use_container_width=True,
                                                                    help="Back to the Media Library ranges."):
            st.session_state["video_time_ranges"] = {}
            st.rerun()
    render_video_selection_section(track, all_video_ids, all_tag_options, catalogues)
    step_nav(back=SECTION_STORY, nxt=SECTION_MATCH)
    _record_page_run()
    st.stop()

tag_filter = list(st.session_state.get("committed_tag_filter", []))
selected_videos = list(st.session_state.get("committed_selected_videos", []))

# Segmentation + split-screen values are set in the Audio Settings section
# and stored in session_state; read them here for the Matching computation.

seg_cfg = segmentation_settings()

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
    st.info("No videos selected yet. Go to **📼 Videos & Time Ranges** to tick some — the plan builds as soon "
            "as at least one is selected.")
    step_nav(back=SECTION_VIDEOS)
    st.stop()

# ⚙️ Matching settings — every setting for this step in one panel. All three tabs
# are drawn on every run (st.tabs runs each body), so nothing is dropped.
with st.container(border=True):
    _mt = st.tabs(["🎛️ Mode & order", "⚖️ Auto-fill weights", "🎬 Intro / Outro", "🧹 Resets"])
with _mt[0]:
    render_matching_mode_settings()

# ---------------------------------------------------------------------------
# Auto-fill weighting — shown inline on this page, not in the sidebar
# ---------------------------------------------------------------------------

_autofill_disabled = matching_mode not in ("Manual step-through", "Auto", "Choreography")
_is_auto_mode = matching_mode == "Auto"
_no_shape_score_mode = matching_mode == "Choreography"  # it doesn't compute a per-frame shape score
_PRESET_CUSTOM = "Custom"


def _apply_preset() -> None:
    """Preset picked: set its sliders (Custom leaves them as they are)."""
    for k, v in AF.PRESETS.get(st.session_state.get("autofill_preset"), {}).items():
        st.session_state[k] = v


def _mark_custom() -> None:
    st.session_state["autofill_preset"] = _PRESET_CUSTOM


def _reroll() -> None:
    st.session_state["autofill_seed"] = (int(st.session_state.get("autofill_seed", 42)) + 1) % 100000


def _weight_shares(v2: bool) -> str:
    """'Energy 40% · Shape 30% · …' — each slider's share of the decision."""
    parts = []
    for term, (key, label) in AF.TERMS.items():
        if key == "weight_shape" and _no_shape_score_mode:
            continue
        if key == "weight_position" and not sequential_mode:
            continue
        parts.append((label, float(st.session_state.get(key, 0.0))))
    total = sum(w for _, w in parts)
    if not total:
        return "All weights are at 0 — picks follow the variety draw only."
    return "  ·  ".join(f"{label} **{100 * w / total:.0f}%**" for label, w in parts if w > 0)


# The preset name only stays if the sliders still hold its values.
_preset_vals = AF.PRESETS.get(st.session_state.get("autofill_preset"))
if _preset_vals and any(abs(float(st.session_state.get(k, v)) - v) > 1e-6 for k, v in _preset_vals.items()):
    st.session_state["autofill_preset"] = _PRESET_CUSTOM

with _mt[1]:
    if _is_auto_mode:
        st.caption("In Auto mode these weights drive EVERY pick directly. On a marked intro/outro "
                   "block, clips actually tagged are still preferred first.")
    elif matching_mode == "Choreography":
        st.caption("Drives a column's '⭐ Auto pick', Confirm & Next's Auto columns and Auto-fill ALL — "
                   "ticking a clip yourself always overrides these.")
    else:
        st.caption("Controls Auto-fill's automatic picks only — the candidate grid stays sorted by raw "
                   "shape score for manual browsing.")
    _hc = st.columns([2, 3])
    _v2 = _hc[0].toggle(
        "✨ Improved auto-fill", key="autofill_v2", disabled=_autofill_disabled,
        help="On: weights are compared on the same scale (each slider is a share of the decision), "
             "motion is matched for the exact part of the scene used on a fairer scale, balance "
             "uses each video's fair share, Variety picks among the best few, and Auto mode fills "
             "intro/outro and the most intense blocks first (except in sequential mode). "
             "Off: the original scoring, to compare.")
    _hc[1].selectbox(
        "Preset", list(AF.PRESETS) + [_PRESET_CUSTOM], key="autofill_preset", on_change=_apply_preset,
        disabled=_autofill_disabled,
        help="A starting point for the sliders below. Moving a slider switches this to Custom.")
    st.caption("Share of each pick: " + _weight_shares(_v2))

    _wt_cols = st.columns(3)
    with _wt_cols[0]:
        if not _no_shape_score_mode:
            st.slider("Match the music's shape", 0.0, 5.0, step=0.1, key="weight_shape",
                      disabled=_autofill_disabled, on_change=_mark_custom,
                      help="(A) How much the clip's frame-by-frame shape match with the music drives the pick.")
        st.slider("Match the energy level", 0.0, 5.0, step=0.1, key="weight_motion",
                  disabled=_autofill_disabled, on_change=_mark_custom,
                  help="(E) Prefers clips whose motion suits this block's intensity — calm footage for "
                       "quiet parts, busy footage for the drops.")
    with _wt_cols[1]:
        st.slider("Use all videos evenly", 0.0, 5.0, step=0.1, key="weight_spread",
                  disabled=_autofill_disabled, on_change=_mark_custom,
                  help="(D) Favours videos that are behind their fair share of the plan (by how much "
                       "usable footage they have), so no single video dominates.")
        st.toggle("Never repeat the previous block's videos", key="avoid_repeat", disabled=_autofill_disabled,
                  help="(C) A hard rule: automatic picks never reuse a video from the neighbouring "
                       "block unless no other video has footage for that spot (you'll be warned).")
        st.toggle("Role clips only in their blocks", key="intro_outro_exclusive",
                  disabled=not (prefer_intro_outro or (_story_regions_now := bool(
                      current_story_map(track_id))) and st.session_state.get("use_story_map", True)),
                  help="A hard rule: clips marked with a role in use (the story map's roles, or Intro / Outro) "
                       "are only used in blocks of that role — in every mode, including what you pick by hand. "
                       "In such a block, a video that has clips with that role offers only those (even out of "
                       "order in sequential mode). Videos with no marked clips aren't affected. Needs the story "
                       "map or 'Prefer Intro/Outro-tagged clips' on (🎛️ Mode & order).")
    with _wt_cols[2]:
        if sequential_mode:
            st.slider("Move through each video in order", 0.0, 5.0, step=0.1, key="weight_position",
                      disabled=_autofill_disabled, on_change=_mark_custom,
                      help="(F) Sequential mode: gives each video a beginning, middle and end — each use "
                           "prefers its own part of the video. Also sets how far a pick may skip ahead "
                           "within a video, never so far that it could run short before the plan ends.")
        st.slider("Variety", 0.0, 5.0, step=0.1, key="weight_random",
                  disabled=_autofill_disabled, on_change=_mark_custom,
                  help="(B) 0 = always the top-scoring clip. Higher = picks more often from the next "
                       "best few instead, for variety (Improved auto-fill), or adds random noise "
                       "(original scoring). Reproducible — the same seed gives the same picks.")
        _rc = st.columns([3, 2])
        _rc[0].number_input("Seed", 0, 99999, key="autofill_seed", disabled=_autofill_disabled,
                            help="Change to reroll Variety without touching anything else.")
        _rc[1].button("🎲 Reroll", on_click=_reroll, disabled=_autofill_disabled, use_container_width=True,
                      help="Next seed — a different draw with the same settings.")

# Save weight values to their shadow copy every time this section renders —
# mirrors audio_settings_shadow so values survive switching to another section.
st.session_state["weight_settings_shadow"] = {
    k: st.session_state[k] for k in WEIGHT_KEYS if k in st.session_state
}

# Always read weight values from session_state (widgets above have already written them)
weight_shape = st.session_state.get("weight_shape", 2.0)
weight_random = st.session_state.get("weight_random", 0.5)
avoid_repeat = bool(st.session_state.get("avoid_repeat", True))
weight_spread = st.session_state.get("weight_spread", 1.0)
weight_motion = st.session_state.get("weight_motion", 1.0)
autofill_seed = st.session_state.get("autofill_seed", 42)
# F only exists in sequential mode — outside it, it has no effect at all.
weight_position = st.session_state.get("weight_position", 0.0) if sequential_mode else 0.0
show_repeat_fallback_flash()  # left by an Auto-fill on the previous run, if it had to repeat

# ---------------------------------------------------------------------------
# Per-scene matching against the beat timeline
# ---------------------------------------------------------------------------

# Manual overrides (reject / swap) are keyed by segment index, which only stays
# meaningful for a given track + segment length — reset them if either changes.
state_key = (track_id, tuple(sorted(seg_cfg.items())))
if st.session_state.get("planner_state_key") != state_key:
    st.session_state["planner_state_key"] = state_key
    if not _consume_load_suppression("Auto"):
        st.session_state["segment_exclusions"] = {}
        st.session_state["segment_clip_reduction"] = {}
        st.session_state["global_excluded_scenes"] = set()

if _mt[3].button("Reset manual clip overrides",
                 help="Clears Auto mode's per-block Swap / Reject / Remove changes and every clip rejected "
                      "from this plan."):
    st.session_state["segment_exclusions"] = {}
    st.session_state["segment_clip_reduction"] = {}
    st.session_state["global_excluded_scenes"] = set()
    st.rerun()

if st.session_state["global_excluded_scenes"]:
    _mt[3].caption(f"{len(st.session_state['global_excluded_scenes'])} clip(s) permanently rejected from this plan.")

queues = build_footage_queues(tuple(selected_videos), tuple(tag_filter), effective_time_ranges())
if not queues:
    st.warning("No candidate scenes available for the selected videos/tags.")
    st.stop()

total_footage_sec = sum(span["remaining_sec"] for spans in queues.values() for span in spans)

with _timed("Cutting the track", seg_cfg["method"]):
    segments = planned_segments(track, seg_cfg)
    apply_segment_intensity(segments, track, density_contrast)
if not track.get("hires"):
    if seg_cfg["method"] == SEG_ADAPTIVE:
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
# For the sequential skip allowance (compute_skip_budget): each video's share of the blocks.
st.session_state["_avg_clips_per_block"] = (sum(recommended_counts) / len(recommended_counts)
                                            if recommended_counts else 1.0)

# 🎭 The story map's part for each block (by the block's middle) — shown on every
# block card and over the chart, as a guide while picking clips.
_story_regions = current_story_map(track_id)
_story_roles = SR.roles_by_id()
story_block_roles = STM.block_roles(_story_regions, segments) if _story_regions else [None] * len(segments)


def story_tag(i: int) -> str:
    """'  ·  🎭 💥 Action part' for block i's story map part, '' if none."""
    r = story_block_roles[i] if 0 <= i < len(story_block_roles) else None
    return f"  ·  🎭 **{SR.label(r, _story_roles)}** part" if r else ""

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
if _story_regions and st.session_state.get("use_story_map", True):
    _mt[2].caption("🎭 The story map is setting each block's role (🎛️ Mode & order → Use the story map), so "
                   "these Intro / Outro blocks aren't used.")
elif not st.session_state.get("prefer_intro_outro"):
    _mt[2].caption("Turn on **Prefer Intro/Outro-tagged clips** (🎛️ Mode & order) to choose which blocks use the "
                   "clips marked Intro / Outro in the Reviewer.")
if st.session_state.get("prefer_intro_outro"):
    with _mt[2]:
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

# 🎭 Each block's role for this plan: from the story map when it's in use, else the
# Intro/Outro blocks above (exactly as before), else none.
_story_on = bool(_story_regions) and bool(st.session_state.get("use_story_map", True))
if _story_on:
    plan_block_roles = list(story_block_roles)
elif prefer_intro_outro:
    plan_block_roles = ["intro" if i in intro_blocks_0based else "outro" if i in outro_blocks_0based else None
                        for i in range(len(segments))]
else:
    plan_block_roles = [None] * len(segments)
_plan_roles_used = sorted({r for r in plan_block_roles if r})
st.session_state["_plan_story_in_use"] = _story_on
st.session_state["_plan_reserved_roles"] = (_plan_roles_used if _story_on and st.session_state.get("story_reserve", True)
                                            else [])
st.session_state["_plan_locked_roles"] = _plan_roles_used if (_story_on or prefer_intro_outro) else []
if _story_on:
    _supply = role_supply(queues, st.session_state["global_excluded_scenes"])
    _needs = role_needs(plan_block_roles, segments)
    _missing = [r for r in _plan_roles_used if _supply.get(r, 0.0) < 0.05]
    if _missing:
        _blk = {r: [i + 1 for i, x in enumerate(plan_block_roles) if x == r] for r in _missing}
        st.warning("🎭 No clips marked " + " / ".join(SR.label(r, _story_roles) for r in _missing)
                   + " in the selected videos — those parts use normal picks (blocks "
                   + "; ".join(f"{SR.label(r, _story_roles)}: {', '.join(map(str, b[:12]))}"
                               + ("…" if len(b) > 12 else "") for r, b in _blk.items())
                   + "). Mark some in the Reviewer, or add videos that have them.")
    with st.expander("🎭 Story map: role clips needed vs available", expanded=False):
        st.caption("Seconds of the song each role covers, and seconds of footage marked with it in the "
                   "selected videos (not yet used). Short on footage → that part repeats fewer clips' worth "
                   "of role footage and falls back to normal picks.")
        st.dataframe([{"Role": SR.label(r, _story_roles),
                       "Needed": f"{_needs.get(r, 0):.0f}s",
                       "Available": f"{_supply.get(r, 0):.0f}s",
                       "": ("✅" if _supply.get(r, 0) >= _needs.get(r, 0)
                            else "⚠️ short" if _supply.get(r, 0) > 0 else "❌ none")}
                      for r in _plan_roles_used], hide_index=True, use_container_width=True)

if matching_mode == "Auto":
    # The shape search is real work, so the plan is remembered and only worked out
    # again when something that changes it changes (settings, videos, footage
    # edits in the Reviewer, Swap / Reject / Remove…).
    def _cat_mtime(v):
        try:
            return CStore.catalogue_path(v).stat().st_mtime
        except OSError:
            return 0.0
    _auto_sig = hashlib.sha1(json.dumps([
        track_id, [(round(sg["start"], 3), round(sg["end"], 3), round(sg.get("intensity", 0), 4)) for sg in segments],
        sorted(selected_videos), sorted(tag_filter), sorted((v, list(r) if r else None) for v, r in effective_time_ranges().items()),
        min_clips, max_clips, ramp_start, ramp_end, allow_same_video, sequential_mode,
        sorted((k, sorted(map(list, v))) for k, v in st.session_state["segment_exclusions"].items()),
        sorted(st.session_state["segment_clip_reduction"].items()),
        sorted(map(list, st.session_state["global_excluded_scenes"])),
        vary_count, int(count_seed), min_clip_len_sec, sorted(intro_blocks_0based), sorted(outro_blocks_0based),
        weight_shape, weight_random, avoid_repeat, weight_spread, weight_motion, autofill_seed, weight_position,
        plan_block_roles, sorted(st.session_state.get("clip_count_skip") or []),
        bool(st.session_state.get("autofill_v2", True)), bool(st.session_state.get("intro_outro_exclusive")),
        bool(st.session_state.get("story_reserve", True)), [(v, _cat_mtime(v)) for v in sorted(selected_videos)],
    ], default=str).encode()).hexdigest()
    _auto_cache = st.session_state.get("_auto_plan_cache")
    if _auto_cache and _auto_cache["sig"] == _auto_sig:
        timeline, clip_count_shortfall, duration_shortfall_sec = _auto_cache["result"]
        _auto_repeat_fallbacks = set(_auto_cache["fallbacks"])
    else:
        with st.spinner("Matching clips to the music's shape…"), \
                _timed("Auto matching", f"{len(segments)} blocks, {len(selected_videos)} videos"):
            timeline, clip_count_shortfall, duration_shortfall_sec = match_scenes_to_track(
                segments, queues,
                min_clips, max_clips, ramp_start, ramp_end, allow_same_video, sequential_mode,
                st.session_state["segment_exclusions"], st.session_state["segment_clip_reduction"],
                st.session_state["global_excluded_scenes"],
                vary_count, int(count_seed), min_clip_len_sec,
                intro_blocks_0based, outro_blocks_0based,
                weight_shape, weight_random, avoid_repeat, weight_spread, weight_motion, autofill_seed,
                weight_position=weight_position, repeat_fallbacks=(_auto_repeat_fallbacks := set()),
                block_roles=plan_block_roles, track=track,
            )
        st.session_state["_auto_plan_cache"] = {
            "sig": _auto_sig, "result": (timeline, clip_count_shortfall, duration_shortfall_sec),
            "fallbacks": sorted(_auto_repeat_fallbacks)}
    if _auto_repeat_fallbacks:
        st.warning(repeat_fallback_message([i + 1 for i in sorted(_auto_repeat_fallbacks)]))

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

    # Editing one already-committed block (✏️ Edit): every OTHER committed block
    # stays committed, and the edited block may only use footage none of them use —
    # in sequential mode, also only footage between that video's clips in the
    # nearest earlier and later committed blocks.
    chor_editing = st.session_state.get("chor_editing_block")
    if chor_editing is not None and not (0 <= chor_editing < chor_block):
        st.session_state.pop("chor_editing_block", None)
        chor_editing = None

    chor_invalidated = []
    if chor_editing is not None:
        # Earlier blocks in order (sequential discarding applies as usual)...
        for i in range(chor_editing):
            for pick in chor_confirmed.get(i, []):
                try:
                    carve_span(queues, pick["video_id"], pick["scene_id"],
                               pick["offset_into_scene_sec"], pick["clip_duration_sec"], sequential_mode)
                except ValueError:
                    chor_invalidated.append((i, pick["video_id"], pick["scene_id"]))
        # ...later blocks only remove their exact clips (they stay committed), and in
        # sequential mode cap each video before its earliest later clip.
        earliest_later = {}
        for i in range(chor_editing + 1, chor_block):
            for pick in chor_confirmed.get(i, []):
                try:
                    carve_span(queues, pick["video_id"], pick["scene_id"],
                               pick["offset_into_scene_sec"], pick["clip_duration_sec"], False)
                except ValueError:
                    chor_invalidated.append((i, pick["video_id"], pick["scene_id"]))
                earliest_later[pick["video_id"]] = min(
                    earliest_later.get(pick["video_id"], float("inf")), pick["clip_start_sec"])
        if sequential_mode:
            for vid, t in earliest_later.items():
                limit_queue_before(queues, vid, t)
        # The edited block's own current clips are free again — make each one an exact
        # dropdown choice.
        for pick in chor_confirmed.get(chor_editing, []):
            split_span_at(queues, pick["video_id"], pick["scene_id"], pick["offset_into_scene_sec"])
        if chor_invalidated:
            st.warning(f"{len(chor_invalidated)} confirmed clip(s) in other blocks no longer match the "
                       f"available footage (a setting changed) — check those blocks after this edit.")

    # Replay confirmed blocks to keep queues accurate for the current block
    for i in (range(chor_block) if chor_editing is None else ()):
        still_valid = []
        for pick in chor_confirmed.get(i, []):
            try:
                carve_span(queues, pick["video_id"], pick["scene_id"],
                           pick["offset_into_scene_sec"], pick["clip_duration_sec"], sequential_mode)
                still_valid.append(pick)
            except ValueError:
                chor_invalidated.append((i, pick["video_id"], pick["scene_id"]))
        chor_confirmed[i] = still_valid
    if chor_invalidated and chor_editing is None:
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

    # ✏️ Editing one already-confirmed block: every OTHER confirmed block stays
    # confirmed (same model as Choreography). adv_frontier is how far you've got;
    # current_block becomes the block being edited, so the grid below shows it.
    adv_frontier = current_block
    adv_editing = st.session_state.get("adv_editing_block")
    if adv_editing is not None and not (0 <= adv_editing < adv_frontier):
        st.session_state.pop("adv_editing_block", None)
        adv_editing = None
    if adv_editing is not None:
        current_block = adv_editing

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

    # Editing: the blocks AFTER the edited one stay confirmed — take their exact
    # clips out of the pool, and in sequential mode cut each video off before
    # its clip in the next later block, so the order still holds.
    _edit_fp = None
    if adv_editing is not None:
        earliest_later = {}
        for i in range(adv_editing + 1, adv_frontier):
            for pick in confirmed.get(i, []):
                try:
                    carve_span(queues, pick["video_id"], pick["scene_id"],
                               pick["offset_into_scene_sec"], pick["clip_duration_sec"], False)
                except ValueError:
                    invalidated.append((i, pick["video_id"], pick["scene_id"]))
                earliest_later[pick["video_id"]] = min(earliest_later.get(pick["video_id"], float("inf")),
                                                       pick["clip_start_sec"])
        if sequential_mode:
            for vid, t in earliest_later.items():
                limit_queue_before(queues, vid, t)
        _edit_fp = tuple(sorted(
            (i, p["video_id"], p["scene_id"], p["offset_into_scene_sec"])
            for i in range(adv_editing + 1, adv_frontier) for p in confirmed.get(i, [])
        ))
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

    clip_count_shortfall = sum(1 for i in range(adv_frontier) if len(confirmed.get(i, [])) < recommended_counts[i])
    duration_shortfall_sec = 0.0  # Advanced mode never partially fills — a slot is either a full-length match or absent

    # Build the timeline for the chart's markers: real picks for confirmed
    # blocks, empty for anything not yet reached.
    timeline = []
    for i, sg in enumerate(segments):
        picks = confirmed.get(i, []) if i < adv_frontier else []
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

# Sections / drops / extra curves, then a marker at every cut coloured by why it happened
add_analysis_overlays(fig, track)
add_cut_markers(fig, segments)

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
    # Marks the block being worked on: the one being edited, else the next unconfirmed one.
    _chor_mark = chor_editing if chor_editing is not None else chor_block
    fig.add_vline(x=segments[min(_chor_mark, len(segments) - 1)]["start"],
                  line=dict(color="rgba(200,30,180,0.6)", width=2, dash="dash"))

fig.update_layout(
    height=380, margin=dict(t=20, b=20), xaxis_title="Time (s)", yaxis_title="Normalized level",
    yaxis2=dict(title="Clips", overlaying="y", side="right", range=[0, max_clips + 0.5], dtick=1, showgrid=False),
    legend=dict(orientation="h", y=-0.25),
)
# 🎭 Story map parts behind the chart, labelled along the top.
for _r in (_story_regions or []):
    _role = _story_roles.get(_r["role"]) if _r["role"] else None
    if _role:
        fig.add_vrect(x0=_r["start"], x1=_r["end"], fillcolor=_role["colour"], opacity=0.16, line_width=0,
                      layer="below", annotation_text=f"{_role['emoji']} {_role['name']}",
                      annotation_position="top left", annotation_font_size=11)
st.plotly_chart(fig, use_container_width=True)
if _story_regions:
    st.caption("🎭 Shaded: the story map's parts (🎭 1. Story map). Each block card says which part it's in.")

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
    preview_count = blocks_to_show("Segments to preview below", "auto_preview_count", len(timeline))

    for entry in timeline[:preview_count]:
        t0, t1 = entry["track_time"]
        seg_idx = entry["segment_index"]
        split_tag = f"  🔲 {entry['clip_count']} CLIPS" if entry["split_screen"] else ""
        edited_tag = "  ✏️ edited" if entry["manually_edited"] else ""
        role_tag = ("" if _story_on else "  🎬 INTRO BLOCK" if seg_idx in intro_blocks_0based
                   else "  🎬 OUTRO BLOCK" if seg_idx in outro_blocks_0based else "")
        st.markdown(f"**Block {seg_idx + 1}**  {t0:.1f}s – {t1:.1f}s  (intensity {entry['segment_intensity']:.2f})"
                   f"{split_tag}{edited_tag}{role_tag}{story_tag(seg_idx)}")

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
                        safe_image(nearest_thumb, width=180)
                    elif s.get("thumbnail") and Path(s["thumbnail"]).exists():
                        safe_image(s["thumbnail"], width=180)
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

    def _chor_start_edit(k: int) -> None:
        """Open committed block k for editing, pre-filled with its current clips."""
        st.session_state["chor_editing_block"] = k
        picks_k = chor_confirmed.get(k, [])
        overrides_k = st.session_state.setdefault("chor_overrides", {}).setdefault(k, {})
        overrides_k.clear()
        for vid in selected_videos:
            st.session_state[f"chor_pick_{k}_{vid}"] = False
        for p in picks_k:
            st.session_state[f"chor_pick_{k}_{p['video_id']}"] = True
            overrides_k[p["video_id"]] = (p["scene_id"], p["offset_into_scene_sec"])

    chor_frontier = chor_block
    if chor_editing is not None:
        chor_block = chor_editing  # the section below renders and saves THIS block

    if chor_block >= len(segments):
        st.success(f"✅ All {len(segments)} blocks confirmed. Review below — click ✏️ Edit on any block "
                   f"to change just that block (the others stay as they are), or use Restart to redo.")

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
            if st.button("✏️ Edit block", key="chor_review_jump_go",
                         help="Re-pick this block's clips. Every other block stays as it is."):
                _chor_start_edit(int(chor_rev_jump) - 1)
                st.rerun()

        st.subheader("Confirmed timeline")
        chor_preview = blocks_to_show("Blocks to show below", "chor_preview_count", len(segments))
        for entry in timeline[:chor_preview]:
            seg_idx = entry["segment_index"]
            t0, t1 = entry["track_time"]
            chor_review_role_tag = ("" if _story_on else "  🎬 INTRO" if seg_idx in intro_blocks_0based
                                    else "  🎬 OUTRO" if seg_idx in outro_blocks_0based else "")
            hdr_cols = st.columns([6, 1])
            with hdr_cols[0]:
                st.markdown(f"**Block {seg_idx + 1}** &nbsp; {t0:.1f}s – {t1:.1f}s &nbsp; "
                            f"intensity {entry['segment_intensity']:.2f}{chor_review_role_tag}{story_tag(seg_idx)}")
            with hdr_cols[1]:
                if st.button("✏️ Edit", key=f"chor_edit_{seg_idx}",
                             help=f"Re-pick block {seg_idx + 1}'s clips. Every other block stays as it is."):
                    _chor_start_edit(seg_idx)
                    st.rerun()
            if not entry["scenes"]:
                st.caption("No clip selected for this block.")
            else:
                cols = st.columns(len(entry["scenes"]))
                for _j, (col, s) in enumerate(zip(cols, entry["scenes"])):
                    with col:
                        nearest_thumb = get_nearest_thumbnail(catalogues, s["video_id"], s["clip_start_sec"])
                        if nearest_thumb is not None:
                            safe_image(nearest_thumb, width=180)
                        elif s.get("thumbnail") and Path(s["thumbnail"]).exists():
                            safe_image(s["thumbnail"], width=180)
                        st.caption(f"`{s['video_id']}` #{s['scene_id']}  — {s['clip_duration_sec']:.1f}s")
                        st.caption(", ".join(s.get("tags", [])))
                        if st.button("🔁 Swap for similar", key=f"swap_chor_{seg_idx}_{_j}",
                                     help="Show the 3 most similar clips available for this spot."):
                            st.session_state["swap_target"] = ("chor", seg_idx, _j)
                            st.rerun()
                render_swap_for_similar("chor", chor_confirmed, seg_idx, len(segments))
            st.divider()

    else:
        # Active block
        seg = segments[chor_block]
        block_duration = seg["end"] - seg["start"]
        chor_rec_count = recommended_counts[chor_block]
        chor_role_filter = plan_block_roles[chor_block]
        chor_role_tag = ("" if _story_on else "  🎬 INTRO BLOCK" if chor_role_filter == "intro"
                         else "  🎬 OUTRO BLOCK" if chor_role_filter == "outro" else "")
        st.subheader(f"{'✏️ Editing block' if chor_editing is not None else 'Block'} "
                     f"{chor_block + 1} of {len(segments)}  "
                     f"({seg['start']:.1f}s – {seg['end']:.1f}s, intensity {seg['intensity']:.2f}){chor_role_tag}")
        if story_tag(chor_block):
            st.markdown(f"This block is in the story map's{story_tag(chor_block)[4:]} — "
                        f"{(SR.roles_by_id().get(story_block_roles[chor_block]) or {}).get('help', '')}")
        if chor_editing is not None:
            st.info("Every other block stays committed. The dropdowns only offer footage no other block uses"
                    + (" — and, with sequential order on, only footage between each video's clips in the "
                       "blocks before and after this one." if sequential_mode else ".")
                    + " This block's current clips are pre-selected.")
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
        if sequential_mode and chor_editing is None:
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

        if chor_editing is None and st.button(f"⚡⚡ Auto-fill ALL remaining blocks ({len(segments) - chor_block} left)",
                     key="chor_autofill_all",
                     help="Automatically fills every block from here to the end using each block's own "
                          "recommended clip count, picking the best motion-matching window per chosen "
                          "video — same as ticking videos and confirming repeatedly. You can still revisit "
                          "and adjust any block afterward with Previous or ✏️ Edit."):
            _chor_rep_fallbacks = []
            for b in range(chor_block, len(segments)):
                seg_b = segments[b]
                block_duration_b = seg_b["end"] - seg_b["start"]
                rec_b = recommended_counts[b]
                seg_target_b = seg_b.get("intensity", seg_b["energy"])

                role_filter_b = plan_block_roles[b]
                _reach_b = (lambda n, _b=b: compute_skip_budget(n, len(segments) - _b, weight_position, len(queues)))

                prev_block_videos_b = (
                    {p["video_id"] for p in chor_confirmed.get(b - 1, [])} if b > 0 else set()
                )

                eligible_b = [vid for vid in selected_videos
                              if get_available_clips_for_video(
                                  queues, vid, block_duration_b, sequential_mode,
                                  excluded_for_block(queues, st.session_state["global_excluded_scenes"],
                                                     role_filter_b), role_filter=role_filter_b,
                                  role_reach=_reach_b)]
                shown_b = eligible_b[:int(max_chor_videos)]

                # For each eligible video, the WEIGHTED best clip from its own (already
                # role-narrowed, if tagged clips exist) pool — same Auto-fill weighting
                # used everywhere else (A has no effect: Choreography has no shape score).
                candidates_b = []
                for vid in shown_b:
                    clips_b = get_available_clips_for_video(
                        queues, vid, block_duration_b, sequential_mode,
                        excluded_for_block(queues, st.session_state["global_excluded_scenes"], role_filter_b),
                        role_filter=role_filter_b, role_reach=_reach_b)
                    if not clips_b:
                        continue
                    # Sequential mode: skip budget (weight F-controlled) rather than a hard
                    # frontmost-only restriction — see compute_skip_budget. Picking a window
                    # past the budget discards everything before it, so the budget caps how
                    # far ahead this video's own spare footage safely allows roaming, given
                    # how many blocks are still left overall.
                    if sequential_mode:
                        _remaining_blocks_b = len(segments) - b
                        _skip_b = compute_skip_budget(len(clips_b), _remaining_blocks_b, weight_position, len(queues))
                        auto_candidates_b = clips_b[:_skip_b + 1]
                    else:
                        auto_candidates_b = clips_b
                    auto_candidates_b = soft_reserve(auto_candidates_b, role_filter_b)
                    ranked_b = rank_candidates_weighted(
                        auto_candidates_b, b, segments, chor_confirmed, queues, prev_block_videos_b,
                        weight_shape, weight_random, avoid_repeat, weight_spread,
                        weight_motion, autofill_seed,
                        weight_position=weight_position, sequential=sequential_mode,
                        videos_per_block=rec_b,
                    )
                    best_b = ranked_b[0]
                    is_tagged_b = has_role(best_b, role_filter_b)
                    candidates_b.append((vid, best_b, is_tagged_b, abs(best_b["motion_norm"] - seg_target_b)))

                # Marked block: prefer candidates whose clip actually IS tagged over ones
                # that fell back to untagged footage. If there aren't enough tagged
                # candidates to fill every slot, the REMAINING slots naturally fall back
                # to the best untagged ones — this IS the fallback-to-original-behaviour
                # rule, expressed as "run out of tagged options, keep going down the list."
                # 'Avoid repeating previous block' (C, when on) outranks everything else.
                _rep_b = lambda t: bool(avoid_repeat and t[0] in prev_block_videos_b)
                if role_filter_b:
                    candidates_b.sort(key=lambda t: (_rep_b(t), not t[2], t[3]))
                else:
                    candidates_b.sort(key=lambda t: (_rep_b(t), t[3]))
                chosen_b = [(vid, best) for vid, best, _, _ in candidates_b[:rec_b]]
                chosen_vids_b = {vid for vid, _ in chosen_b}
                if repeat_fallback_videos([{"video_id": v} for v in chosen_vids_b],
                                          prev_block_videos_b, avoid_repeat):
                    _chor_rep_fallbacks.append(b + 1)

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

            flash_repeat_fallback(_chor_rep_fallbacks)
            st.session_state["chor_current_block"] = len(segments)
            st.rerun()

        # Navigation + block jump
        chor_nav_cols = st.columns([2, 2, 3])
        with chor_nav_cols[0]:
            if chor_editing is not None:
                if st.button("✖ Cancel edit", key="chor_cancel_edit",
                             help="Leave this block exactly as it was committed."):
                    st.session_state.pop("chor_editing_block", None)
                    st.rerun()
            elif st.button("◀ Previous", disabled=(chor_block == 0), key="chor_prev",
                           help="Edit the previous block — every other block stays committed."):
                _chor_start_edit(chor_block - 1)
                st.rerun()
        with chor_nav_cols[2]:
            chor_jump = st.number_input(
                "Jump to block", min_value=1, max_value=len(segments),
                value=chor_block + 1, key="chor_jump_input", label_visibility="collapsed",
                disabled=chor_editing is not None,
            )
            if st.button("↩ Go to block", key="chor_jump_go", disabled=chor_editing is not None,
                         help="A block before your progress opens it for editing (the others stay "
                              "committed); a later one moves ahead to it."):
                _target = int(chor_jump) - 1
                if _target < chor_frontier:  # even an empty block: never move progress back
                    _chor_start_edit(_target)
                else:
                    st.session_state["chor_current_block"] = _target
                st.rerun()
        with chor_nav_cols[1]:
            if st.button("💾 Save block" if chor_editing is not None else "Confirm & Next ▶",
                         type="primary", key="chor_next"):
                new_picks = []
                _chor_reach = (lambda n: compute_skip_budget(n, len(segments) - chor_block, weight_position,
                                                            len(queues)))
                overrides = st.session_state.get("chor_overrides", {}).get(chor_block, {})
                eligible_videos = [vid for vid in selected_videos
                                   if get_available_clips_for_video(
                                       queues, vid, block_duration, sequential_mode,
                                       excluded_for_block(queues, st.session_state["global_excluded_scenes"],
                                                          chor_role_filter),
                                       role_filter=chor_role_filter, role_reach=_chor_reach)]
                shown_videos = eligible_videos[:int(max_chor_videos)]
                seg_target = seg.get("intensity", seg["energy"])

                # Confirm only the ticked columns (checkbox state drives this, not position)
                chosen_videos = [vid for vid in shown_videos
                                 if st.session_state.get(f"chor_pick_{chor_block}_{vid}", False)]

                for video_id in chosen_videos:
                    override = overrides.get(video_id)
                    clips = get_available_clips_for_video(
                        queues, video_id, block_duration, sequential_mode,
                        excluded_for_block(queues, st.session_state["global_excluded_scenes"], chor_role_filter),
                        role_filter=chor_role_filter, role_reach=_chor_reach)
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
                        # mode: skip budget (weight F-controlled) instead of a hard frontmost-
                        # only restriction — see compute_skip_budget and the matching comment
                        # in render_choreography_block's preview above.
                        if sequential_mode:
                            _remaining_blocks = len(segments) - chor_block
                            _skip = compute_skip_budget(len(clips), _remaining_blocks, weight_position, len(queues))
                            auto_candidates = clips[:_skip + 1]
                        else:
                            auto_candidates = clips
                        auto_candidates = soft_reserve(auto_candidates, chor_role_filter)
                        ranked = rank_candidates_weighted(
                            auto_candidates, chor_block, segments, chor_confirmed, queues,
                            chor_prev_block_videos, weight_shape, weight_random,
                            avoid_repeat, weight_spread, weight_motion, autofill_seed,
                            weight_position=weight_position, sequential=sequential_mode,
                            videos_per_block=max(1, len(chosen_videos)),
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
                if chor_editing is not None:
                    st.session_state.pop("chor_editing_block", None)  # back to where you were
                else:
                    st.session_state["chor_current_block"] = chor_block + 1
                st.rerun()

        # Used-so-far / remaining footage per video, same computation Manual step-through
        # uses for its candidate cards.
        chor_video_stats = {}
        for vid in selected_videos:
            used = sum(p["clip_duration_sec"] for i in range(chor_frontier) if i != chor_editing
                       for p in chor_confirmed.get(i, []) if p["video_id"] == vid)
            remaining = sum(span["remaining_sec"] for span in queues.get(vid, []))
            chor_video_stats[vid] = {"used": used, "remaining": remaining}

        chor_weighting = {
            "segments": segments, "confirmed": chor_confirmed,
            "weight_shape": weight_shape, "weight_random": weight_random,
            "avoid_repeat": avoid_repeat, "weight_spread": weight_spread,
            "weight_motion": weight_motion, "seed": autofill_seed,
            "weight_position": weight_position,
        }

        # The choreography grid fragment — dropdowns update live without a full rerun
        render_choreography_block(
            chor_block, seg, selected_videos, int(max_chor_videos),
            chor_rec_count, queues, sequential_mode,
            excluded_for_block(queues, st.session_state["global_excluded_scenes"], chor_role_filter), catalogues,
            chor_prev_block_videos, chor_video_stats, chor_role_filter, chor_weighting,
            track=track,
        )

else:
    # -----------------------------------------------------------------------
    # Manual step-through UI
    # -----------------------------------------------------------------------

    def _adv_start_edit(k: int) -> None:
        """Open confirmed block k for editing; its current clips start ticked."""
        st.session_state["adv_editing_block"] = k
        st.session_state["adv_viewing"] = None
        for key in [x for x in st.session_state if x.startswith(f"adv_pick_{k}_")]:
            st.session_state.pop(key, None)
        st.session_state.pop(f"adv_similar_{k}", None)

    if current_block >= len(segments):
        st.success(f"✅ All {len(segments)} blocks confirmed. Review below — click ✏️ Edit on any block "
                   f"to change just that block (the others stay as they are), or use Restart to start over.")

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
            if st.button("✏️ Edit block", key="adv_review_jump_go",
                         help="Re-pick this block's clips. Every other block stays as it is."):
                _adv_start_edit(int(rev_jump) - 1)
                st.rerun()

        st.subheader("Confirmed timeline")
        preview_count = blocks_to_show("Blocks to show below", "adv_preview_count", len(segments))
        for entry in timeline[:preview_count]:
            seg_idx = entry["segment_index"]
            t0, t1 = entry["track_time"]
            split_tag = f"  🔲 {entry['clip_count']} CLIPS" if entry["split_screen"] else ""
            adv_review_role_tag = ("" if _story_on else "  🎬 INTRO" if seg_idx in intro_blocks_0based
                                   else "  🎬 OUTRO" if seg_idx in outro_blocks_0based else "")

            hdr_cols = st.columns([6, 1])
            with hdr_cols[0]:
                st.markdown(
                    f"**Block {seg_idx + 1}** &nbsp; {t0:.1f}s – {t1:.1f}s &nbsp; "
                    f"intensity {entry['segment_intensity']:.2f}{split_tag}{adv_review_role_tag}{story_tag(seg_idx)}"
                )
            with hdr_cols[1]:
                if st.button("✏️ Edit", key=f"adv_edit_block_{seg_idx}",
                             help=f"Re-pick block {seg_idx + 1}'s clips. Every other block stays as it is."):
                    _adv_start_edit(seg_idx)
                    st.rerun()

            if not entry["scenes"]:
                st.caption("No clip selected for this block.")
            else:
                cols = st.columns(len(entry["scenes"]))
                for _j, (col, s) in enumerate(zip(cols, entry["scenes"])):
                    with col:
                        nearest_thumb = get_nearest_thumbnail(catalogues, s["video_id"], s["clip_start_sec"])
                        if nearest_thumb is not None:
                            safe_image(nearest_thumb, width=180)
                        elif s.get("thumbnail") and Path(s["thumbnail"]).exists():
                            safe_image(s["thumbnail"], width=180)
                        st.caption(f"`{s['video_id']}` #{s['scene_id']}  — {s['clip_duration_sec']:.1f}s")
                        st.caption(", ".join(s.get("tags", [])))
                        if st.button("🔁 Swap for similar", key=f"swap_manual_{seg_idx}_{_j}",
                                     help="Show the 3 most similar clips available for this spot."):
                            st.session_state["swap_target"] = ("manual", seg_idx, _j)
                            st.rerun()
                render_swap_for_similar("manual", confirmed, seg_idx, len(segments))
            st.divider()

    else:
        seg = segments[current_block]
        rec_count = recommended_counts[current_block]
        already = confirmed.get(current_block, [])
        already_keys = {(p["video_id"], p["scene_id"]) for p in already}

        adv_role_tag = ("" if _story_on else "  🎬 INTRO BLOCK" if current_block in intro_blocks_0based
                       else "  🎬 OUTRO BLOCK" if current_block in outro_blocks_0based else "")
        st.subheader(f"{'✏️ Editing block' if adv_editing is not None else 'Block'} "
                     f"{current_block + 1} of {len(segments)}  "
                     f"({seg['start']:.1f}s – {seg['end']:.1f}s, intensity {seg['intensity']:.2f}){adv_role_tag}")
        if story_tag(current_block):
            st.markdown(f"This block is in the story map's{story_tag(current_block)[4:]} — "
                        f"{(SR.roles_by_id().get(story_block_roles[current_block]) or {}).get('help', '')}")
        if adv_editing is not None:
            st.info("Every other block stays confirmed. The candidates only use footage no other block uses"
                    + (" — and, with sequential order on, only footage between each video's clips in the "
                       "blocks before and after this one." if sequential_mode else ".")
                    + " This block's current clips start ticked; 💾 Save block keeps your changes.")
        st.caption(f"Recommended simultaneous clips for this block: {rec_count} "
                   f"(from your split-screen settings) — this is a guide only; select any number below.")

        if adv_editing is None and st.button(f"⚡⚡ Auto-fill ALL remaining blocks ({len(segments) - current_block} left)",
                     key="adv_autofill_all",
                     help="Automatically picks the best-scoring clip(s) for every block from here to the "
                          "end, using each block's own recommended count — same as clicking Auto-fill and "
                          "Confirm & Next repeatedly. You can still revisit and adjust any block afterward "
                          "with Previous."):
            _adv_rep_fallbacks = []
            for b in range(current_block, len(segments)):
                seg_b = segments[b]
                audio_curve_b = get_block_audio_curve(track, seg_b["start"], seg_b["end"])
                _excl_b = st.session_state["segment_exclusions"].get(b, set())
                _rf_b = plan_block_roles[b]
                _global_excl = excluded_for_block(queues, st.session_state["global_excluded_scenes"], _rf_b)
                _pos_targets_b = (
                    compute_position_targets(b, len(segments), confirmed, queues, recommended_counts[b])
                    if sequential_mode and weight_position > 0 else None
                )
                cands_b, _ = find_shape_candidates(
                    seg_b, queues, sequential_mode, _excl_b, _global_excl, audio_curve_b,
                    int(max_shape_matches), restrict_to_front=sequential_mode,
                    remaining_blocks=len(segments) - b, weight_position=weight_position,
                    position_targets=_pos_targets_b, weight_shape=weight_shape,
                )
                # Auto-fill ALL is a fully automatic process (like Auto mode and
                # Choreography's own Auto-fill ALL), so marked blocks opportunistically
                # hard-filter here, unlike the manual grid above which only highlights.
                if _rf_b and not any(has_role(c, _rf_b) for c in cands_b):
                    # The best shape matches had none of this block's role — search the
                    # role clips on their own (same in-order reach in sequential mode).
                    _role_c, _ = find_shape_candidates(
                        seg_b, queues, sequential_mode, _excl_b, _global_excl, audio_curve_b,
                        int(max_shape_matches), restrict_to_front=sequential_mode,
                        remaining_blocks=len(segments) - b, weight_position=weight_position,
                        position_targets=_pos_targets_b, weight_shape=weight_shape, only_role=_rf_b,
                    )
                    cands_b = _role_c or cands_b
                cands_b = soft_reserve(prefer_role(cands_b, _rf_b), _rf_b)

                # --- Fallback chain: relax constraints progressively ---
                # Level 1: shape search found nothing → drop role filter and retry
                if not cands_b and _rf_b:
                    cands_b, _ = find_shape_candidates(
                        seg_b, queues, sequential_mode, _excl_b, _global_excl, audio_curve_b,
                        int(max_shape_matches), restrict_to_front=sequential_mode,
                        remaining_blocks=len(segments) - b, weight_position=weight_position,
                        position_targets=_pos_targets_b, weight_shape=weight_shape,
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
                            "window_offset_sec": best_span["offset_sec"],
                            "window_duration_sec": round(seg_b["end"] - seg_b["start"], 3),
                            "score": 0.0,
                            "motion_norm": best_span["motion_norm"],
                            "curve_slice": np.array([]),
                            "fps": 25.0,
                            "tags": best_span["tags"],
                            "thumbnail": best_span["thumbnail"],
                            "intro_candidate": best_span.get("intro_candidate", False),
                            "outro_candidate": best_span.get("outro_candidate", False),
                            "roles": best_span.get("roles", []),
                        }]

                rec_b = recommended_counts[b]
                # Each block's "previous block" is whatever was JUST decided for b-1 in this same
                # loop (or already-confirmed history if b == current_block) — cascades correctly,
                # same as the repetition penalty would see stepping through manually one at a time.
                prev_videos_b = {p["video_id"] for p in confirmed.get(b - 1, [])} if b > 0 else set()
                if avoid_repeat:
                    cands_b = widen_for_repeat_rule(
                        cands_b, prev_videos_b, 1 if allow_same_video else rec_b,
                        lambda ex: find_shape_candidates(
                            seg_b, queues, sequential_mode, _excl_b, _global_excl, audio_curve_b,
                            max(rec_b, 1), restrict_to_front=sequential_mode,
                            remaining_blocks=len(segments) - b, weight_position=weight_position,
                            position_targets=_pos_targets_b, weight_shape=weight_shape,
                            exclude_videos=ex)[0])
                ranked_b = rank_candidates_weighted(
                    cands_b, b, segments, confirmed, queues, prev_videos_b,
                    weight_shape, weight_random, avoid_repeat, weight_spread,
                    weight_motion, autofill_seed,
                    weight_position=weight_position, sequential=sequential_mode, videos_per_block=rec_b,
                )
                # Place the weighted picks; if one can't be placed, fall through to the next
                # best candidate (same hard rules), so a block is only ever left empty when
                # there is genuinely no footage left for it.
                chosen_b = select_autofill_picks(ranked_b, rec_b, allow_same_video)
                order_b = chosen_b + [c for c in ranked_b if all(c is not x for x in chosen_b)]
                new_picks, placed_ids, used_vids_b = [], set(), set()
                for c in order_b:
                    if len(new_picks) >= rec_b:
                        break
                    if not allow_same_video and c["video_id"] in used_vids_b:
                        continue
                    try:
                        pick = carve_span(queues, c["video_id"], c["scene_id"],
                                          c["window_offset_sec"], c["window_duration_sec"], sequential_mode)
                    except ValueError:
                        continue
                    new_picks.append(pick)
                    placed_ids.add(id(c))
                    used_vids_b.add(c["video_id"])
                for c in cands_b:
                    st.session_state[f"adv_pick_{b}_{cand_key(c)}"] = id(c) in placed_ids
                if not new_picks:
                    # Last resort: front of whichever available footage best matches the
                    # block's intensity — prefer pieces long enough for the whole block.
                    _need = seg_b["end"] - seg_b["start"]
                    _spans = [sp for sp in get_candidate_spans(queues, sequential_mode, _global_excl)
                              if (sp["video_id"], sp["scene_id"]) not in _excl_b]
                    if _spans:
                        _target = seg_b.get("intensity", seg_b.get("energy", 0.5))
                        _best = min(_spans, key=lambda sp: (sp["remaining_sec"] < _need - 1e-6,
                                                            abs(sp["motion_norm"] - _target)))
                        new_picks.append(consume_span(_best, _need))
                if repeat_fallback_videos(new_picks, prev_videos_b, avoid_repeat):
                    _adv_rep_fallbacks.append(b + 1)
                confirmed[b] = new_picks
            flash_repeat_fallback(_adv_rep_fallbacks)
            st.session_state["adv_current_block"] = len(segments)
            st.session_state["adv_viewing"] = None
            st.rerun()

        nav_cols = st.columns([2, 2, 3])
        with nav_cols[0]:
            if adv_editing is not None:
                if st.button("✖ Cancel edit", key="adv_cancel_edit",
                             help="Leave this block exactly as it was confirmed."):
                    st.session_state.pop("adv_editing_block", None)
                    st.session_state["adv_viewing"] = None
                    st.rerun()
            elif st.button("◀ Previous", disabled=(current_block == 0), key="adv_prev",
                           help="Edit the previous block — every other block stays confirmed."):
                _adv_start_edit(current_block - 1)
                st.rerun()
        with nav_cols[2]:
            jump_target = st.number_input(
                "Jump to block", min_value=1, max_value=len(segments),
                value=current_block + 1, key="adv_jump_input",
                help="Type a block number and press Enter to jump directly to it.",
                label_visibility="collapsed",
            )
            if st.button("↩ Go to block", key="adv_jump_go", disabled=adv_editing is not None,
                         help="A block before your progress opens it for editing (the others stay "
                              "confirmed); a later one moves ahead to it."):
                _target = int(jump_target) - 1
                if _target < adv_frontier:  # even an empty block: never move progress back
                    _adv_start_edit(_target)
                else:
                    st.session_state["adv_current_block"] = _target
                    st.session_state["adv_viewing"] = None
                st.rerun()
        with nav_cols[1]:
            if st.button("💾 Save block" if adv_editing is not None else "Confirm & Next ▶",
                         type="primary", key="adv_next"):
                chosen = [c for c in st.session_state.get(f"adv_candidates_{current_block}", [])
                         if st.session_state.get(f"adv_pick_{current_block}_{cand_key(c)}")]
                new_confirmed = []
                _original = {(p["video_id"], p["scene_id"]): p for p in confirmed.get(current_block, [])}
                for c in chosen:
                    # Editing and kept a clip that was already here: keep its exact window.
                    if adv_editing is not None and not c.get("similar") \
                            and (c["video_id"], c["scene_id"]) in _original:
                        _p = _original.pop((c["video_id"], c["scene_id"]))
                        try:
                            new_confirmed.append(carve_span(queues, _p["video_id"], _p["scene_id"],
                                                            _p["offset_into_scene_sec"], _p["clip_duration_sec"],
                                                            sequential_mode))
                            continue
                        except ValueError:
                            pass
                    try:
                        pick = carve_span(queues, c["video_id"], c["scene_id"],
                                          c["window_offset_sec"], c["window_duration_sec"], sequential_mode)
                        new_confirmed.append(pick)
                    except ValueError as e:
                        st.warning(f"Couldn't use {c['video_id']} #{c['scene_id']}: {e}")
                confirmed[current_block] = new_confirmed
                if adv_editing is not None:
                    st.session_state.pop("adv_editing_block", None)  # back to where you were
                else:
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
        prev_block_videos = {p["video_id"] for p in confirmed.get(current_block - 1, [])} if current_block > 0 else set()
        adv_role_filter = plan_block_roles[current_block]
        _adv_excl = excluded_for_block(queues, st.session_state["global_excluded_scenes"], adv_role_filter)
        _io_exclusive = bool(st.session_state.get("intro_outro_exclusive") and (prefer_intro_outro or _story_on))
        search_key = (adv_key, current_block, int(max_shape_matches), sequential_mode,
                     prior_picks_fingerprint, current_skips_fingerprint, tuple(sorted(selected_videos)),
                     _edit_fp, avoid_repeat, tuple(sorted(prev_block_videos)), _io_exclusive, adv_role_filter)
        cache = st.session_state.get(f"adv_search_cache_{current_block}")
        block_audio_curve = get_block_audio_curve(track, seg["start"], seg["end"])  # cheap — always needed, cache or not

        if cache and cache["key"] == search_key:
            candidates, missing_curve_videos = cache["candidates"], cache["missing"]
        else:
            with _timed("Manual: block shape search", f"{len(selected_videos)} videos"):
                candidates, missing_curve_videos = find_shape_candidates(
                    seg, queues, sequential_mode, st.session_state["segment_exclusions"].get(current_block, set()),
                    _adv_excl, block_audio_curve, int(max_shape_matches),
                )
            if adv_role_filter:
                # Also offer this block's best role clips (they may not be among the top shape matches).
                _role_c, _ = find_shape_candidates(
                    seg, queues, sequential_mode, st.session_state["segment_exclusions"].get(current_block, set()),
                    _adv_excl, block_audio_curve, int(max_shape_matches), only_role=adv_role_filter,
                )
                candidates = candidates + [r for r in _role_c if all(cand_key(r) != cand_key(c) for c in candidates)]
            if avoid_repeat:
                # Make sure Auto-fill has enough non-repeating videos to choose from
                # (C's hard rule) — adds their best clip to the grid if the top
                # matches all came from the previous block's videos.
                _rc = recommended_counts[current_block]
                candidates = widen_for_repeat_rule(
                    candidates, prev_block_videos, 1 if allow_same_video else _rc,
                    lambda ex: find_shape_candidates(
                        seg, queues, sequential_mode,
                        st.session_state["segment_exclusions"].get(current_block, set()),
                        _adv_excl, block_audio_curve,
                        max(_rc, 1), exclude_videos=ex)[0])
            for _k in [k for k in st.session_state
                       if k.startswith("adv_search_cache_") and k != f"adv_search_cache_{current_block}"]:
                del st.session_state[_k]
            st.session_state[f"adv_search_cache_{current_block}"] = {
                "key": search_key, "candidates": candidates, "missing": missing_curve_videos,
            }

        # Marked intro/outro blocks surface their tagged candidates first rather than
        # hiding the rest — you're already choosing by hand here, so a highlight that
        # preserves every legitimate shape match felt like the better fit than a hard
        # filter (unlike Auto/Choreography, which do filter). Stable sort: within each
        # group (tagged vs not), the existing best-score-first order is untouched.
        if adv_role_filter and _io_exclusive:
            # The hard rule: a video with clips marked for this block offers only those.
            _has_role = {c["video_id"] for c in candidates if has_role(c, adv_role_filter)}
            candidates = [c for c in candidates if c["video_id"] not in _has_role or has_role(c, adv_role_filter)]
        if adv_role_filter:
            candidates = sorted(candidates, key=lambda c: not has_role(c, adv_role_filter))

        # ✨ More-like-this results for this block, kept until the search itself changes.
        _sim_state = st.session_state.get(f"adv_similar_{current_block}")
        similar_results = (_sim_state["results"] if _sim_state and _sim_state.get("key") == search_key else [])
        if _sim_state and _sim_state.get("key") != search_key:
            st.session_state.pop(f"adv_similar_{current_block}", None)
        candidates = candidates + [r for r in similar_results
                                   if all(cand_key(r) != cand_key(c) for c in candidates)]

        st.session_state[f"adv_candidates_{current_block}"] = candidates

        if missing_curve_videos:
            st.warning(f"{len(missing_curve_videos)} video(s) have no frame-by-frame motion data yet and were "
                      f"skipped: {', '.join(sorted(missing_curve_videos))}. Run backfill_motion_curve.py in "
                      f"Colab to include them.")
        if not candidates:
            st.info("No candidate clips at least this block's length are currently available. "
                    "Try a shorter block (adjust segmentation), or widen your video/tag selection.")

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
            "avoid_repeat": avoid_repeat, "weight_spread": weight_spread,
            "weight_motion": weight_motion, "seed": autofill_seed,
            "weight_position": weight_position, "sequential": sequential_mode,
            "allow_same_video": allow_same_video, "block_role": adv_role_filter,
        }
        _excl_here = st.session_state["segment_exclusions"].get(current_block, set())
        _gex = st.session_state["global_excluded_scenes"]
        similar_ctx = {
            "search_key": search_key,
            "find": lambda c: more_like_this_for_grid(c, seg, queues, _excl_here, _gex, sequential_mode,
                                                      block_audio_curve, candidates),
        }
        render_candidate_grid(candidates, current_block, already_keys, seg, block_audio_curve,
                              prev_block_videos, video_stats, recommended_counts[current_block],
                              autofill_weights, catalogues, similar_ctx)


if matching_mode == "Manual step-through" and adv_frontier < len(segments):
    st.warning(
        f"Only {current_block} of {len(segments)} blocks confirmed — exporting/rendering now will "
        f"leave the rest empty. Finish stepping through above first, or export anyway if intentional."
    )
if matching_mode == "Choreography" and chor_frontier < len(segments):
    st.warning(
        f"Only {chor_frontier} of {len(segments)} blocks confirmed — exporting/rendering now will "
        f"leave the rest empty. Finish stepping through above first, or export anyway if intentional."
    )

def render_why_panel(timeline: list, segments: list) -> None:
    """🔎 Why these picks? — for one block: each automatic pick's score broken
    down by slider, its motion against the block's, and the runners-up."""
    if not st.session_state.get("autofill_v2", True):
        return
    store = st.session_state.get(WHY_KEY) or {}
    filled = [e for e in timeline if e.get("scenes")]
    if not filled or not store:
        return
    with st.expander("🔎 Why these picks?", expanded=False):
        st.caption("How Improved auto-fill scored each automatic pick: each slider's part of the score "
                   "(+ helped, − held it back, relative to the other candidates), the clip's motion "
                   "against the block's intensity, and the clips it beat. Picks you made yourself "
                   "aren't scored.")
        idx = st.selectbox(
            "Block", [e["segment_index"] for e in filled], key="why_block",
            format_func=lambda i: f"Block {i + 1}  ·  intensity {segments[i].get('intensity', 0):.2f}")
        entry = next(e for e in filled if e["segment_index"] == idx)
        picks = [c for slot in entry["scenes"] for c in (slot.get("chain") or [slot])]
        _brole = plan_block_roles[idx] if idx < len(plan_block_roles) else None
        if _brole:
            st.markdown(f"🎭 This block's role: **{SR.label(_brole)}**")

        def role_line(p: dict) -> str:
            if not _brole:
                return ""
            sc = next((x for x in (catalogues.get(p["video_id"]) or {}).get("scenes", [])
                       if x.get("scene_id") == p["scene_id"]), None)
            return (f"  ·  🎭 ✅ {SR.label(_brole)} clip" if sc and _brole in SR.scene_roles(sc)
                    else "  ·  🎭 not a role clip (none left / fallback)")
        picked = set()
        labels = {k: AF.TERMS[k][1] for k in AF.TERMS}

        def bar(v: float) -> str:
            n = min(10, int(round(abs(v) * 8)))
            return ("🟩" if v >= 0 else "🟥") * max(n, 1)

        for p in picks:
            k = why_key(idx, p["video_id"], p["scene_id"], p.get("clip_start_sec", 0.0))
            w = store.get(k)
            picked.add(k)
            head = (f"**{p['video_id']}** · scene {p['scene_id']} · "
                    f"{format_mmss(p.get('clip_start_sec', 0.0))}")
            if not w:
                st.markdown(head + "  — picked by you (not scored)" + role_line(p))
                continue
            st.markdown(head + f"  — ranked **{w['rank']} of {w['of']}**"
                        + ("  ·  🔁 repeat (nothing else available)" if w.get("repeat") else "") + role_line(p))
            lines = [f"{bar(v)} {labels[t]} {v:+.2f}" for t, v in
                     sorted(w["shares"].items(), key=lambda kv: -abs(kv[1]))]
            if abs(w.get("variety", 0)) > 1e-3:
                lines.append(f"🎲 Variety {w['variety']:+.2f}")
            lines.append(f"🏃 motion {w['motion_level']:.2f} for a block at {w['target']:.2f}")
            st.caption("  \n".join(lines))
        others = sorted((v for k, v in store.items() if k[0] == idx and k not in picked),
                        key=lambda v: (v["rank"], -v["score"]))[:3]
        if others:
            st.markdown("**Runners-up**")
            st.caption("  \n".join(
                f"{o['video_id']} · scene {o['scene_id']} · {format_mmss(o['start'])} — score "
                f"{o['score']:+.2f}, motion {o['motion_level']:.2f}" for o in others))


render_why_panel(timeline, segments)

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
    "segmentation": {**seg_cfg, "method": seg_cfg["method"] if track.get("hires") else "legacy"},
    "sequential_video_order": sequential_mode,
    "globally_rejected_clips": [list(k) for k in st.session_state["global_excluded_scenes"]],
    "video_time_ranges": {k: list(v) for k, v in effective_time_ranges().items()},
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
if _cut_edits():
    export_plan["segmentation"]["hand_edited_cuts"] = True    # ✂️ cuts edited in the cut editor

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
    """This track's renders — including ones cleared from the server after
    uploading to Google Drive (their details stay here), so numbering carries on."""
    import fnmatch
    return sorted((p for p in render_files() if fnmatch.fnmatch(p.name, preview_pattern)),
                  key=lambda p: p.name)


def next_preview_path() -> Path:
    existing = existing_preview_versions()
    used_numbers = []
    for p in existing:
        m = re.search(r"\.(\d{3})\.mp4$", p.name)
        if m:
            used_numbers.append(int(m.group(1)))
    next_num = (max(used_numbers) + 1) if used_numbers else 1
    return PREVIEW_DIR / f"temp-Video-preview-{safe_track_id}.{next_num:03d}.mp4"


def ensure_render_sources_local() -> list:
    """Make sure the track's audio and every video the plan uses exist on this machine,
    downloading any that are missing from Google Drive (sync_pull only brings over the
    catalogue/audio JSONs, never the media itself). Returns a list of error strings."""
    needed = {track["source_path"]}
    for _e in timeline:
        for _slot in _e["scenes"]:
            for _lnk in _slot["chain"]:
                needed.add(catalogues[_lnk["video_id"]]["source_path"])
    missing = [(raw, Path(colab_to_local(raw))) for raw in sorted(needed)
               if not Path(colab_to_local(raw)).exists()]
    errors = []
    if not missing:
        return errors
    for n, (raw, local) in enumerate(missing, 1):
        if not (_DRIVE_SYNC_AVAILABLE and "MyDrive/" in raw):
            errors.append(f"{local.name}: not found at {local} and Drive download isn't available")
            continue
        _bar = st.progress(0.0, text=f"Downloading {local.name} from Google Drive ({n}/{len(missing)})...")
        _err = download_source_video(
            raw.split("MyDrive/", 1)[1], local,
            progress_callback=lambda p, _b=_bar, _n=local.name, _i=n: _b.progress(
                min(p, 1.0), text=f"Downloading {_n} from Google Drive ({_i}/{len(missing)})..."),
        )
        _bar.empty()
        if _err:
            errors.append(f"{local.name}: {_err}")
    return errors


_stream_sources = False
if _DRIVE_SYNC_AVAILABLE:
    _stream_sources = st.toggle(
        "Stream original videos from Google Drive", value=True, key="render_stream_sources",
        help="On: videos and the track that aren't on the server are read straight from Google Drive "
             "while rendering — only the seconds each clip uses are fetched, and nothing is stored. "
             "If a stream fails, that file is downloaded instead. Off: every missing file is "
             "downloaded to the server first (the old way).",
    )

if st.button("🎬 Render Preview", type="primary"):
    if not _stream_sources:
        _src_errors = ensure_render_sources_local()
        if _src_errors:
            st.error("Can't render — some source files are missing and couldn't be downloaded:\n\n"
                     + "\n".join(f"- {e}" for e in _src_errors))
            st.stop()
    out_path = next_preview_path()
    progress_bar = st.progress(0.0, text="Rendering...")
    _render_started = time.time()

    def _progress(done, total):
        progress_bar.progress(done / total, text=f"Rendering segment {done}/{total}...")

    def _notice(text):
        progress_bar.progress(progress_bar_value[0], text=text)

    progress_bar_value = [0.0]

    def _progress_tracked(done, total):
        progress_bar_value[0] = done / total
        _progress(done, total)

    try:
        _src_stats = render_plan_dict(export_plan, str(out_path), progress_callback=_progress_tracked,
                                      stream=_stream_sources, notice=_notice,
                                      app_version=APP_VERSION,
                                      extra_meta={"project": _build_project_save_dict(),
                                                  "project_name": (st.session_state.get("project_save_name") or "").strip() or None},
                                      ) or {}
        _secs = time.time() - _render_started
        _how = []
        if _src_stats.get("streamed"):
            _how.append(f"{_src_stats['streamed']} streamed from Drive")
        if _src_stats.get("downloaded"):
            _how.append(f"{_src_stats['downloaded']} downloaded after a stream failed")
        if _src_stats.get("local"):
            _how.append(f"{_src_stats['local']} already on the server")
        st.session_state["render_last_stats"] = (
            f"Last render took {int(_secs // 60)}:{int(_secs % 60):02d}"
            + (f" — source files: {', '.join(_how)}." if _how else ".")
        )
        progress_bar.progress(1.0, text="Done.")
        _token = drive_token()
        if _token:
            progress_bar.progress(1.0, text="Uploading to Google Drive…")
            _err = upload_render(out_path, _token)
            if _err:
                st.session_state["render_drive_note"] = ("warning", f"Rendered, but the Drive upload failed: {_err} "
                                                         f"— upload it later from Media Library → Renders.")
            elif auto_clear_enabled():
                clear_local(out_path)
                st.session_state["render_drive_note"] = ("success", f"Rendered {out_path.name}, saved to Google Drive "
                                                         f"and cleared from the server.")
            else:
                st.session_state["render_drive_note"] = ("success", f"Rendered {out_path.name} and saved to Google Drive.")
        else:
            st.session_state["render_drive_note"] = ("info", f"Rendered {out_path.name}. Connect Google Drive (sidebar) "
                                                     f"to save renders there — then upload it from Media Library → Renders.")
        st.rerun()
    except Exception as e:
        st.error(f"Render failed: {e}")

if st.session_state.get("render_last_stats"):
    st.caption("⏱️ " + st.session_state["render_last_stats"])
_drive_note = st.session_state.pop("render_drive_note", None)
if _drive_note:
    getattr(st, _drive_note[0])(_drive_note[1])

versions = existing_preview_versions()
if versions:
    st.caption(f"{len(versions)} preview(s) for this track — compare side by side below.")
    cols_per_row = 2
    for row_start in range(0, len(versions), cols_per_row):
        row = versions[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, path in zip(cols, row):
            with col:
                _meta = load_render(path)
                if _meta is None:
                    continue
                show_render_player(_meta, height=300)
                st.caption(path.name + ("  ·  ☁️ on Drive" if _meta.get("drive") else ""))
                if st.button("🗑️ Delete", key=f"delete_preview_{path.name}",
                             help="Deletes it from the server and moves the Google Drive copy to the Drive bin."):
                    _del_err = delete_render(path, drive_token())   # also its metadata and thumbnail
                    if _del_err:
                        st.error(_del_err)
                    else:
                        st.rerun()
else:
    st.caption("No previews rendered yet for this track.")

_record_page_run()
