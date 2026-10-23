"""
Compilation Planner — interactive matching of video scenes to a music track's
energy/beat timeline, with a live visualization and JSON export for the VSE add-on.
"""

import hashlib
import inspect
import io
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
import story_map_editor as SME
import scene_roles as SR
from audio_charts import add_analysis_overlays, add_vrects
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
import core.sequential as SEQ
import settings_store as SSTORE
from core.matching import prefer_role, widen_for_repeat_rule, cand_key, why_key, format_mmss
from cut_editor import cut_editor as _cut_editor_widget, media_url as _media_url
from core.shape import SHAPE_MATCH_POINTS, resample_curve, shape_score, get_block_audio_curve
from core.footage import (_tc_to_seconds, _overlap_with_range, get_candidate_spans, has_role, role_supply,
                          role_needs, consume_span, limit_queue_before, split_span_at,
                          skip_span, carve_span)
from core.picking import select_autofill_picks, repeat_fallback_videos
from diagnostics import timed as _timed   # 🩺 step timings for the Server page

from config import CATALOGUE_DIR, AUDIO_DIR, PLANS_DIR, PREVIEW_DIR, PROJECTS_DIR
from track_files import track_file

# Sign-in, whitelist, Drive connect, title and the shared sidebar — see page_setup.py
from page_setup import (page_setup, push_file_with_oauth, APP_VERSION, drive_token,
                        OAUTH_AVAILABLE as _OAUTH_AVAILABLE, DRIVE_SESSION_KEY as _OAUTH_SESSION_KEY)

# The page's building blocks (planner/ — see planner/__init__.py)
from planner.state import (
    AUDIO_SETTINGS_KEYS, _bump_cut_edit_rev, credentials_available, DIAL_DEFAULTS, download_source_video,
    _DRIVE_SYNC_AVAILABLE, fragment, MAX_SIMULTANEOUS_CLIPS, push_directory_to_drive, push_file_to_drive,
    rerun_fragment, rerun_full, sanitize_filename, SECTION_AUDIO, SECTION_MATCH, SECTION_STORY, SECTION_VIDEOS,
    SECTIONS, STORY_KEY, sync_pull
)
from planner.data import (
    _add_video_to_selection, _all_video_durations, build_footage_queues, effective_time_ranges,
    find_best_motion_window, get_nearest_thumbnail, get_video_duration, list_tracks, load_all_catalogues,
    load_motion_curves_for_video, load_timeline_sprite, load_track, _load_track_file, _local_media,
    resolve_thumbnail, seconds_to_time, _tile_bytes, TILE_CACHE_MAX, time_to_seconds, video_list_thumbnail
)
from planner.projects import (
    apply_audio_profile, _apply_project_load_dict, _audio_profile_key, auto_tune_parameters, AUTOSAVE_DIR,
    autosave_now, _autosave_path, _build_project_save_dict, _canon, _consume_load_suppression, _dial_value,
    _do_save_project, ensure_audio_profile, list_saved_projects, load_project, _method_cfg, _new_project,
    _normalise, _open_recommended_plan, _progress_summary, _project_changes, PROJECT_FILE_VERSION,
    PROJECT_SIMPLE_KEYS, _remember_project, _restore_choice, _run_auto_tune, save_project,
    _start_autosave_session, _undo_new_project
)
from planner.audio import (
    _ab_keep, _ab_swap, ADAPTIVE_DIAL_KEYS, add_cut_markers, _aim_adaptive, AIM_MODES, allowed_clip_count,
    _apply_cut_focus, _apply_cut_style, _audio_retune_if_paced, _beats_per_bar_changed, clips_for_intensity,
    _cut_edits, CUT_FOCUS, CUT_KIND_STYLES, _drop_cut_edits, _idea_cuts, _keep_cut_edits, _keep_new_with_locks,
    _mark_audio_custom, _mark_style_custom, _method_changed, _PACE_LABELS, _pace_row, planned_segments,
    _recut_region, _region_lines, _render_aim_adaptive, _render_cut_editor, _render_scored_cut_settings,
    _render_smart_cut_settings, _render_style_row, SEG_LABELS, segmentation_settings, _set_cut_edits, _setting,
    _smart_reset, _style_cuts, _STYLE_CUTS_CACHE
)
from planner.story import (
    _add_new_videos, current_story_map, render_story_map_picker, render_story_map_section, _story_analysis,
    _story_play_audio, _story_save_default, _story_save_named, _story_set, _story_source, _story_update_named,
    _story_use, _track_mtime
)
from planner.matching import (
    chor_shown, CHOR_SKIP, _chor_slider_changed, _chor_step, _clip_index, clips_with_planned,
    compute_position_targets, compute_skip_budget, _container_supports_key, excluded_for_block, fill_slot,
    find_shape_candidates, flash_repeat_fallback, get_available_clips_for_video, match_scenes_to_track,
    plan_clips_per_block, plan_context, planned_clip, _planned_span, rank_candidates_weighted,
    render_candidate_grid, _render_chor_preview_panel, render_choreography_block, repeat_fallback_message,
    role_chips, role_locked_keys, _shape_memo, show_repeat_fallback_flash, soft_reserve,
    _SUPPORTS_KEYED_CONTAINER, WHY_KEY
)
from planner.similar import (
    blocks_to_show, _clip_curve, _COLUMNS_SUPPORT_VALIGN, _columns_support_valign, compute_similar_videos,
    compute_video_rankings, find_similar_clips, format_match_breakdown, library_tag_idf, load_all_audio_tracks,
    more_like_this_for_grid, _ranking_weights, RECOMMENDED_COUNT, render_ranking_weights,
    render_swap_for_similar, replacement_pool, resolve_video_selection, story_role_need, _swap_candidates,
    _usage_signature, video_role_seconds
)
from planner.videos import (
    _back_to_best_matches, _clear_video_selection, _randomize_top_videos, render_video_selection_section,
    _select_all_recommended
)
from planner.audio_step import (
    _audio_status, block_chart, render_audio_settings_section, render_audio_simple
)

page_setup("Compilation Planner", "pages/1_Compilation_Planner.py")
_PAGE_T0 = time.perf_counter()   # 🩺 whole-page run time — see _record_page_run
_PAGE_MARKS = [("start", _PAGE_T0)]
SLOW_RUN_SEC = 0.5               # a full run slower than this also logs which parts took the time
SLOW_PART_SEC = 0.1


def _mark(part: str) -> None:
    """🩺 The time since the previous mark went on `part` (see _record_page_run)."""
    _PAGE_MARKS.append((part, time.perf_counter()))


def _record_page_run() -> None:
    """🩺 How long this full run of the page took, for the Server page's timings
    (fragment reruns and runs that stop early — e.g. no track yet — aren't counted).
    Logged per step ("Planner page: 3. Matching") as well as overall; a slow run also logs its
    slow parts ("Planner part: block cards") so the Server page shows where the time went."""
    import diagnostics as _diag
    now = time.perf_counter()
    total = now - _PAGE_T0
    step = str(st.session_state.get("ui_section", ""))
    _diag.record("Planner page (full run)", total, step[:2].strip())
    _diag.record(f"Planner page: {step[2:].strip() or '?'}", total)
    if total >= SLOW_RUN_SEC:
        marks = _PAGE_MARKS + [("rest of the page", now)]
        for (_, t_prev), (part, t) in zip(marks[:-1], marks[1:]):
            if t - t_prev >= SLOW_PART_SEC:
                _diag.record(f"Planner part: {part}", t - t_prev, step[:2].strip())
# The background Drive sync timer may have pulled new catalogues/audio since
# the caches were filled — reload them once if so.
refresh_caches_after_sync()


shared_cache(VR.clear_memos)        # 🧠 memory reset clears the ranking's in-memory copies too


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
if total_excluded and st.session_state.get("ui_advanced"):
    st.caption(f"{total_excluded} scene(s) excluded from planning (e.g. intros) and ignored throughout.")

# ⚙️ Every setting above keeps its value from run to run, whichever step is showing (its widget may
# not be drawn) — see settings_store.py. Widgets for them use key= only; everything reads
# st.session_state. role_clips_mode is set from story_reserve / intro_outro_exclusive below.
SSTORE.keep(DIAL_DEFAULTS, extra_keys=("role_clips_mode",))


_start_autosave_session()
st.session_state["_before_reset"] = autosave_now   # used by the sidebar's 🧹 Free memory

_restore = st.session_state.pop("_restore_autosave", None)
if _restore:
    _restore_warnings = _apply_project_load_dict(_restore.get("state") or {})
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
                          width="stretch", key="autosave_restore")
            _rc[1].button("Discard", on_click=_restore_choice, args=("discard",),
                          width="stretch", key="autosave_discard")

# Media Library → Renders → "Reopen in Planner": load that render's saved project
# state (or the settings recovered from its plan) exactly like 📂 Load Project.
_reopen = st.session_state.pop("_reopen_project", None)
if _reopen:
    _reopen_warnings = _apply_project_load_dict(_reopen["data"])
    st.session_state["_match_pressed"] = True      # you asked for this plan — build it without ▶️ Match
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


if st.session_state.get("_open_recommended_plan"):
    _open_recommended_plan(st.session_state.pop("_open_recommended_plan"))
    st.rerun()


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
    if _pb[1].button(f"💾 Save changes", type="primary", width="stretch", key="project_quick_save",
                     help=f"Save to '{_loaded['name']}'."):
        _do_save_project(_loaded["name"])
# 🆕 New and 💾 Save / 📂 Open open a panel inside the project bar (not a pop-over: on a phone a pop-over
# re-positions itself while its dropdown is open, so it jumped around under your finger).
_PANEL_KEY = "_project_panel"


def _toggle_project_panel(which: str) -> None:
    st.session_state[_PANEL_KEY] = None if st.session_state.get(_PANEL_KEY) == which else which


def _start_new_project() -> None:
    st.session_state[_PANEL_KEY] = None
    _new_project()


_panel = st.session_state.get(_PANEL_KEY)
_pb[2].button("✕ Close" if _panel == "new" else "🆕 New", width="stretch", key="project_panel_new",
              type="primary" if _panel == "new" else "secondary", on_click=_toggle_project_panel, args=("new",))
_pb[3].button("✕ Close" if _panel == "file" else "💾 Save / 📂 Open", width="stretch", key="project_panel_file",
              type="primary" if _panel == "file" else "secondary", on_click=_toggle_project_panel, args=("file",))
if _panel == "new":
    with _pbar:
        st.caption("Start again from a blank planner — default settings, no track work, no project "
                   "loaded. Saved projects aren't touched, and you can bring back what's on screen "
                   "now straight afterwards.")
        if _new_loaded and _new_loaded_changes:
            st.warning(f"**{_new_loaded['name']}** has unsaved changes — save it first if you want to keep them.")
        _nc = st.columns([3, 2], vertical_alignment="bottom")
        _nc[0].text_input("Name (optional)", key="new_project_name", placeholder="my-new-compilation",
                          help="Filled in as the project name, ready for 💾 Save.")
        _nc[1].button("🆕 Start new project", type="primary", width="stretch", key="new_project_go",
                      on_click=_start_new_project)
elif _panel == "file":
    with _pbar:
        _fc = st.columns(2)
        with _fc[0]:
            st.markdown("**💾 Save**")
            st.caption("Every setting plus matching progress (Auto, Manual step-through and Choreography), so "
                       "you can pick up again later.")
            _save_name = st.text_input("Project name", key="project_save_name", placeholder="my-compilation",
                                       help="Filled in with the project you loaded — save to update it, or type a "
                                            "new name to save a copy.")
            _overwrites = _save_name.strip() and (PROJECTS_DIR / f"{sanitize_filename(_save_name.strip())}.json").exists()
            if st.button("💾 Save Project" + (" (update)" if _overwrites else ""), disabled=not _save_name.strip(),
                         width="stretch", type="primary"):
                st.session_state[_PANEL_KEY] = None
                _do_save_project(_save_name.strip())
        with _fc[1]:
            st.markdown("**📂 Open**")
            _existing_projects = list_saved_projects()
            if _existing_projects:
                _load_choice = st.selectbox("Open project", ["—"] + _existing_projects, key="project_load_choice",
                                            label_visibility="collapsed")
                if st.button("📂 Open Project", width="stretch", disabled=_load_choice == "—"):
                    _path = PROJECTS_DIR / f"{_load_choice}.json"
                    _load_warnings = load_project(_path)
                    _remember_project(_load_choice, json.loads(_path.read_text()))
                    st.session_state["_prefill_project_name"] = _load_choice
                    st.session_state["_project_notices"] = ([("warning", _w) for _w in _load_warnings]
                                                            + [("success", f"Opened '{_load_choice}'.")])
                    st.session_state[_PANEL_KEY] = None
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
if st.session_state.pop("_rec_opened", False) and _autosave_path(".before_new").exists():
    _pbar.button("↩️ Bring back what I had", on_click=_undo_new_project, key="rec_open_undo")

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
# kept like every other setting (settings_store.keep) so every step can use it.
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
# 🎭 Role clips: one 3-way control for the two settings underneath (they stay in projects as they were).
ROLE_CLIP_MODES = {"anywhere": "Anywhere", "saved": "Saved for their parts", "only": "Only in their parts"}
_ROLE_CLIP_SETTINGS = {"anywhere": (False, False), "saved": (True, False), "only": (True, True)}
if st.session_state.get("role_clips_mode") in _ROLE_CLIP_SETTINGS:
    st.session_state["story_reserve"], st.session_state["intro_outro_exclusive"] = \
        _ROLE_CLIP_SETTINGS[st.session_state["role_clips_mode"]]
else:
    st.session_state["role_clips_mode"] = (
        "only" if st.session_state.get("intro_outro_exclusive") else
        "saved" if st.session_state.get("story_reserve", True) else "anywhere")

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


def render_mode_picker() -> str:
    """How clips are chosen — at the top of the 🎬 Matching step."""
    return st.radio(
        "How clips are chosen", ["Auto", "Manual step-through", "Choreography"], key="matching_mode",
        horizontal=True,
        help="Auto: the algorithm picks the best-matching clip per segment automatically.  "
             "Manual step-through: pick from the best shape-matched candidates one block at a time.  "
             "Choreography: one column per source video — see the auto pick for each, then override with "
             "any available clip from that video via a timecode dropdown.")


def render_matching_mode_settings() -> None:
    """🎛️ Mode tab of the 🎬 Matching step's settings."""
    mode = st.session_state.get("matching_mode", "Auto")
    c = st.columns([1, 2])
    with c[0]:
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
                     "add-on's 'Keep Sequential Order') — other videos can still interleave around them. "
                     "How far a pick may move through each video: ⚖️ Auto-fill weights → Move through each "
                     "video in order (shown once this is applied).")
    if st.session_state.get("split_screen_enabled"):
        c[1].toggle("Allow one video twice in a split-screen block", key="allow_same_video",
                    help="Lets a split-screen block use more than one clip from the same video.")
    has_map = bool(current_story_map(st.session_state.get("track_id")))
    roles_in_play = (has_map and st.session_state.get("use_story_map", True)) or st.session_state.get("prefer_intro_outro")
    sc = st.columns([1, 2])
    sc[0].toggle("🎭 Use the story map", key="use_story_map", disabled=not has_map,
                 help="Each block prefers clips marked with its story map part's role (🎬 Intro, 🌙 Slow, "
                      "💥 Action, 🏁 Outro, your own) — falling back to normal picks where a video has none. "
                      "While it's on it decides every block's role, so the 🎬 Intro / Outro tab isn't used."
                      + ("" if has_map else " This track has no story map yet (🎭 1. Story map)."))
    sc[1].radio("Role clips", list(ROLE_CLIP_MODES), key="role_clips_mode", horizontal=True,
                format_func=ROLE_CLIP_MODES.get, disabled=not roles_in_play,
                help="How strictly clips marked with a role (by the story map, or Intro / Outro) stay in their "
                     "own parts.  Anywhere: they're preferred in their parts but free to be used elsewhere.  "
                     "Saved for their parts: automatic picks elsewhere avoid them, so they're still there when "
                     "their part comes — unless nothing else is left (story map only).  Only in their parts: a "
                     "hard rule in every mode, even what you pick by hand; in such a part a video with clips of "
                     "that role offers only those (even out of order in sequential mode).")


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
        c[0].button(f"⬅ {back.split(' ', 1)[1]}", on_click=_goto, args=(back,), width="stretch",
                    key=f"nav_back_{back}")
    if nxt:
        c[2].button(f"{nxt.split(' ', 1)[1]} ➡", on_click=_goto, args=(nxt,), type="primary",
                    width="stretch", key=f"nav_next_{nxt}")


# The step bar: the radio (same key and values as always — saved sessions and the
# recovery file keep working) styled as steps, with a status line under each.
st.markdown("""<style>
.st-key-stepnav .stElementContainer, .st-key-stepnav .stRadio { width: 100% !important; }
.st-key-stepnav [role="radiogroup"] { display: grid !important; grid-auto-columns: 1fr; grid-auto-flow: column; gap: .5rem;
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
import ui_mode
_sw = st.columns([6, 1])
_advanced = ui_mode.switch(_sw[1])
with st.container(key="stepnav"):
    # 🛠️ Simple uses the track's saved story map without showing its step (it still opens when
    # something sends you to it, e.g. Media Library → 🎭 Edit story map).
    _steps = (SECTIONS if _advanced or st.session_state.get("ui_section") == SECTION_STORY
              else [x for x in SECTIONS if x != SECTION_STORY])
    ui_section = st.radio(
        "Section", _steps,
        horizontal=True, key="ui_section", label_visibility="collapsed",
    )
with st.container(key="stepstatus"):
    _status = [t for x, t in zip(SECTIONS, _step_status()) if x in _steps]
    for _col, _txt in zip(st.columns(len(_steps)), _status):
        _col.caption(_txt)
_BACK_FROM_VIDEOS = SECTION_STORY if _advanced else SECTION_AUDIO

_mark("setup, track and settings")
if ui_section == SECTION_AUDIO:
    render_track_picker()
    if not _advanced:
        render_audio_simple(track)
        step_nav(nxt=SECTION_VIDEOS)
        _record_page_run()
        st.stop()
    _ac = st.columns([1, 2])
    _ac[0].button("🎯 Auto-detect settings", on_click=_run_auto_tune, width="stretch",
                  help="Analyses this track's tempo, energy and hits, then sets the cutting and split-screen "
                       "settings. Adjust anything afterwards to fine-tune.")
    with _ac[1]:
        render_split_screen_settings()
    render_audio_settings_section(track)
    _mark("cutting dials and chart")
    step_nav(nxt=SECTION_STORY)
    _record_page_run()
    st.stop()

if ui_section == SECTION_STORY:
    render_story_map_section(track)
    _mark("story map")
    step_nav(back=SECTION_AUDIO, nxt=SECTION_VIDEOS)
    _record_page_run()
    st.stop()

if ui_section == SECTION_VIDEOS:
    _n_lib_ranges = len(library_ranges())
    if _advanced and (_n_lib_ranges or st.session_state["video_time_ranges"]):
        _rc = st.columns([3, 1])
        _rc[0].caption(" · ".join(x for x in (
            f"📚 {_n_lib_ranges} video(s) have a Media Library range" if _n_lib_ranges else "",
            f"🎚️ {len(st.session_state['video_time_ranges'])} have a range set for this project"
            if st.session_state["video_time_ranges"] else "") if x))
        if st.session_state["video_time_ranges"] and _rc[1].button("Reset project ranges",
                                                                    width="stretch",
                                                                    help="Back to the Media Library ranges."):
            st.session_state["video_time_ranges"] = {}
            st.rerun()
    render_video_selection_section(track, all_video_ids, all_tag_options, catalogues, simple=not _advanced)
    _mark("video list")
    step_nav(back=_BACK_FROM_VIDEOS, nxt=SECTION_MATCH)
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

# How clips are chosen, then one status line (filled in once the plan is worked out), then
# ⚙️ Matching settings — every setting for this step, folded away with a summary. All the
# tabs are drawn on every run (st.tabs runs each body, an expander's content always runs),
# so nothing is dropped.
_notes = []     # things worth knowing about this plan — shown together in one box under the status line


def _note(text: str) -> None:
    _notes.append(text)


render_mode_picker()
_status_slot = st.container()
_has_map_now = bool(current_story_map(track_id))
_summary = " · ".join(x for x in (
    f"{st.session_state.get('autofill_preset', 'Balanced')} weights",
    ("🎭 story map on" if _has_map_now and st.session_state.get("use_story_map", True) else
     "story map off" if _has_map_now else ""),
    "in order" if sequential_mode else "",
    "intro / outro clips" if prefer_intro_outro and not (_has_map_now and st.session_state.get("use_story_map", True))
    else "",
    f"{len(st.session_state['global_excluded_scenes'])} rejected" if st.session_state["global_excluded_scenes"] else "",
) if x)
# ⚙️ The settings are a form: changing them reloads nothing until ▶️ Match / Apply & match. In Auto
# mode nothing is matched until you press it the first time, nor after the videos or cuts change
# (see _match_allowed below); Swap / Reject / Remove on a confirmed plan still apply straight away.


def _press_match() -> None:
    st.session_state["_match_pressed"] = True


def _reroll_and_match() -> None:
    _reroll()
    _press_match()


# 🎛️ Tune mode: Matching starts with the settings the tune was scored with (planner/tune.py).
import planner.tune as TUNE
if st.session_state.get("cut_style") == TUNE.TUNE_STYLE and TUNE.result() and not TUNE.stale():
    _tchanged = TUNE.changed_keys()
    _tc = st.columns([4, 1.4], vertical_alignment="center")
    if not _tchanged:
        _tc[0].caption("🎛️ **Tuned settings in use** — the ones the tune was scored with (Balanced weights, one "
                       "clip a block). Change anything in ⚙️ Matching settings to override.")
    else:
        _tc[0].caption(f"🎛️ ✏️ You've changed {len(_tchanged)} of the tuned settings.")
        _tc[1].button("↩️ Back to tuned", key="pl_tune_restore", width="stretch",
                      help="Puts back every matching setting the tune was scored with.",
                      on_click=lambda: (TUNE.apply_matching(), st.session_state.__setitem__("_match_pressed", True)))

_has_plan = bool(st.session_state.get("_match_confirmed_setup"))
with st.form("match_settings", border=False):
    with st.expander(f"⚙️ Matching settings  ·  {_summary}", expanded=False):
        _mt = st.tabs(["🎛️ Mode & order", "⚖️ Auto-fill weights", "🎬 Intro / Outro"])
    _fc = st.columns([1.4, 3])
    _fc[0].form_submit_button("▶️ Apply & match" if _has_plan else "▶️ Match", type="primary",
                              width="stretch", on_click=_press_match,
                              help="Builds the plan with these settings. Change as many settings as you like "
                                   "first — nothing reloads until you press this.")
    _fc[1].caption("Change any ⚙️ settings, then press ▶️ to update the plan." if _has_plan else
                   "Check the ⚙️ settings, then press ▶️ Match to build the plan.")
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


# A preset newly picked in the form: set its sliders (before they're drawn this run). Then the
# preset name only stays if the sliders still hold its values (a slider moved → Custom).
if st.session_state.get("autofill_preset") != st.session_state.get("_preset_seen"):
    if st.session_state.get("_preset_seen") is not None:
        _apply_preset()
    st.session_state["_preset_seen"] = st.session_state.get("autofill_preset")
_preset_vals = AF.PRESETS.get(st.session_state.get("autofill_preset"))
if _preset_vals and any(abs(float(st.session_state.get(k, v)) - v) > 1e-6 for k, v in _preset_vals.items()):
    st.session_state["autofill_preset"] = _PRESET_CUSTOM
    st.session_state["_preset_seen"] = _PRESET_CUSTOM

with _mt[1]:
    _what_weights_do = (
        "In Auto mode these weights drive EVERY pick directly. On a marked intro/outro block, clips actually "
        "tagged are still preferred first." if _is_auto_mode else
        "Drives a column's '⭐ Auto pick', Confirm & Next's Auto columns and Auto-fill ALL — ticking a clip "
        "yourself always overrides these." if matching_mode == "Choreography" else
        "Controls Auto-fill's automatic picks only — the candidate grid stays sorted by raw shape score for "
        "manual browsing.")
    _hc = st.columns([2, 3])
    _v2 = bool(st.session_state.get("autofill_v2", True))
    if not _v2:
        # Always on now; only a project saved with it off shows the switch, so it can be turned back on.
        _v2 = _hc[0].toggle(
            "✨ Improved auto-fill", key="autofill_v2", disabled=_autofill_disabled,
            help="This project was saved with the original scoring. On: weights are compared on the same "
                 "scale (each slider is a share of the decision), motion is matched for the exact part of the "
                 "scene used, balance uses each video's fair share, Variety picks among the best few, and Auto "
                 "mode fills intro/outro and the most intense blocks first (except in sequential mode).")
    _hc[1].selectbox(
        "Preset", list(AF.PRESETS) + [_PRESET_CUSTOM], key="autofill_preset",
        disabled=_autofill_disabled,
        help="A starting point for the sliders below. Moving a slider switches this to Custom. "
             + _what_weights_do)
    st.caption("Share of each pick: " + _weight_shares(_v2))

    _wt_cols = st.columns(3)
    with _wt_cols[0]:
        if not _no_shape_score_mode:
            st.slider("Match the music's shape", 0.0, 5.0, step=0.1, key="weight_shape",
                      disabled=_autofill_disabled,
                      help="(A) How much the clip's frame-by-frame shape match with the music drives the pick.")
        st.slider("Match the energy level", 0.0, 5.0, step=0.1, key="weight_motion",
                  disabled=_autofill_disabled,
                  help="(E) Prefers clips whose motion suits this block's intensity — calm footage for "
                       "quiet parts, busy footage for the drops.")
    with _wt_cols[1]:
        st.slider("Use all videos evenly", 0.0, 5.0, step=0.1, key="weight_spread",
                  disabled=_autofill_disabled,
                  help="(D) Favours videos that are behind their fair share of the plan (by how much "
                       "usable footage they have), so no single video dominates.")
        st.toggle("Never repeat the previous block's videos", key="avoid_repeat", disabled=_autofill_disabled,
                  help="(C) A hard rule: automatic picks never reuse a video from the neighbouring "
                       "block unless no other video has footage for that spot (you'll be warned).")
    with _wt_cols[2]:
        if sequential_mode:
            st.slider("Follow the music ↔ Follow the video", 0.0, 5.0, step=0.1, key="weight_position",
                      disabled=_autofill_disabled,
                      help="(F) In order: Auto plans each video's clips for all its blocks together, in time "
                           "order, so a video can never run short halfway. 0 = each clip is simply the best fit "
                           "to its block's music (still in order). Higher = the clips are also pulled into an "
                           "even walk through the whole video, beginning → middle → end. "
                           "Manual step-through and Choreography plan the same way from the block you're on — "
                           "the 📍 planned clips — and plan the rest again whenever you pick something else.")
        st.slider("Variety", 0.0, 5.0, step=0.1, key="weight_random",
                  disabled=_autofill_disabled,
                  help="(B) 0 = always the top-scoring clip. Higher = picks more often from the next "
                       "best few instead, for variety (Improved auto-fill), or adds random noise "
                       "(original scoring). Reproducible — the same seed gives the same picks.")
        st.form_submit_button("🎲 Reroll", on_click=_reroll_and_match, disabled=_autofill_disabled,
                              width="stretch",
                              help=f"A different Variety draw with the same settings — and matches straight away "
                                   f"(draw {int(st.session_state.get('autofill_seed', 42))}).")


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

def _reset_overrides() -> None:
    st.session_state["segment_exclusions"] = {}
    st.session_state["segment_clip_reduction"] = {}
    st.session_state["global_excluded_scenes"] = set()


_has_overrides = bool(st.session_state["segment_exclusions"] or st.session_state["segment_clip_reduction"]
                      or st.session_state["global_excluded_scenes"])

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
        _note("This track was analysed before the 30 Hz onset signals existed, so cuts use the older, "
              "looser method. Re-run audio_pipeline.py in Colab to upgrade it.")
    else:
        _note("Rhythm Engine needs the 30 Hz onset signals to compute bar energies, which this track "
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
_story_rules = bool(_story_regions) and bool(st.session_state.get("use_story_map", True))
with _mt[2]:
    st.toggle("Prefer Intro/Outro-tagged clips", key="prefer_intro_outro", disabled=_story_rules,
              help="For tracks without a story map (or with it off): the blocks you choose below draw from "
                   "each video's clips marked Intro / Outro in the Reviewer, where a video has any; a video "
                   "with none offers its normal clips there.")
    if _story_rules:
        st.caption("🎭 The story map is in charge of every block's role, its Intro and Outro parts included — "
                   "these settings aren't used while it's on (🎛️ Mode & order → Use the story map).")
if st.session_state.get("prefer_intro_outro") and not _story_rules:
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
        _io_roles = SR.roles_by_id()
        block_chart(track, segments, {**{i: ("outro", _io_roles["outro"]["colour"]) for i in outro_blocks_0based},
                                      **{i: ("intro", _io_roles["intro"]["colour"]) for i in intro_blocks_0based}})
        st.caption("Shaded: the intro and outro blocks as last applied (▶️ Apply & match to update).")

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
        _note("🎭 No clips marked " + " / ".join(SR.label(r, _story_roles) for r in _missing)
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
                      for r in _plan_roles_used], hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 📍 The planned path (in order, Manual step-through and Choreography): the same
# "plan each video's clips for all its blocks together" Auto uses, from the block
# you're on to the end — each clip inside one scene, as these modes pick. It's
# worked out again only when you pick something the plan didn't (or a setting
# changes): following it keeps the rest of it valid.
# ---------------------------------------------------------------------------

def _queues_fp(q: dict) -> str:
    """What footage is still free — a planned path is only good for the footage it was planned on."""
    return hashlib.sha1(json.dumps(sorted(
        (v, sp["scene_id"], round(float(sp["offset_sec"]), 1), round(float(sp["remaining_sec"]), 1))
        for v, spans in q.items() for sp in spans), default=str).encode()).hexdigest()


def planned_ahead(from_block: int, q: dict) -> dict:
    """{block: [pick, …]} for blocks from_block onward — the planned path on the footage in q
    (as it stands when from_block is reached). {} when not in order."""
    import copy
    import dataclasses
    if not sequential_mode or track is None or from_block >= len(segments):
        return {}
    excl = st.session_state["segment_exclusions"]
    base = hashlib.sha1(json.dumps([
        track_id, [(round(sg["start"], 3), round(sg["end"], 3), round(sg.get("intensity", 0), 4)) for sg in segments],
        sorted(q), min_clips, max_clips, ramp_start, ramp_end, allow_same_video, min_clip_len_sec,
        sorted((k, sorted(map(list, v))) for k, v in excl.items()),
        sorted(map(list, st.session_state["global_excluded_scenes"])),
        sorted(intro_blocks_0based), sorted(outro_blocks_0based), weight_shape, weight_random, avoid_repeat,
        weight_spread, weight_motion, autofill_seed, weight_position, plan_block_roles,
        bool(st.session_state.get("autofill_v2", True)), bool(st.session_state.get("intro_outro_exclusive")),
        bool(st.session_state.get("story_reserve", True)),
    ], default=str).encode()).hexdigest()
    fp = _queues_fp(q)
    cache = st.session_state.get("_planned_path")
    if cache and cache["base"] == base and cache["expect"].get(from_block) == fp:
        return {b: p for b, p in cache["picks"].items() if b >= from_block}
    segs = segments[from_block:]
    work = copy.deepcopy(q)
    ctx = dataclasses.replace(plan_context(), planned_sequential=True, plan_within_scenes=True)
    with st.spinner("📍 Planning each video's path through the rest of the track…"), \
            _timed("Planned path", f"from block {from_block + 1}, {len(segs)} blocks"):
        tl, _, _ = CMATCH.match_scenes_to_track(
            segs, work, min_clips, max_clips, ramp_start, ramp_end, allow_same_video, True,
            {k - from_block: v for k, v in excl.items() if k >= from_block}, {},
            st.session_state["global_excluded_scenes"], False, int(count_seed), min_clip_len_sec,
            {b - from_block for b in intro_blocks_0based if b >= from_block},
            {b - from_block for b in outro_blocks_0based if b >= from_block},
            weight_shape, weight_random, avoid_repeat, weight_spread, weight_motion, autofill_seed,
            weight_position=weight_position, repeat_fallbacks=set(),
            block_roles=list(plan_block_roles[from_block:]), track=track, ctx=ctx)
    picks = SEQ.planned_picks(tl, segs, offset=from_block)
    # The footage each later block will see if the plan is followed — so following it reuses it.
    sim, expect = copy.deepcopy(q), {from_block: fp}
    for b in range(from_block, len(segments)):
        dur = segments[b]["end"] - segments[b]["start"]
        kept = []
        for p in picks.get(b, []):
            try:
                carve_span(sim, p["video_id"], p["scene_id"], p["offset_into_scene_sec"], dur, True)
                kept.append(p)
            except ValueError:
                pass
        picks[b] = kept
        expect[b + 1] = _queues_fp(sim)
    st.session_state["_planned_path"] = {"base": base, "expect": expect, "picks": picks}
    return picks


def planned_candidates(picks: list, q: dict, seg: dict, audio_curve) -> list:
    """Manual step-through: the planned picks as candidate cards (marked 📍)."""
    dur = seg["end"] - seg["start"]
    out = []
    for p in picks:
        sp, off = _planned_span(q, p, dur)
        if sp is None:
            continue
        vals, fps = _clip_curve(p["video_id"], p["scene_id"], off, dur)
        score = 0.0
        if vals is not None and len(vals) and audio_curve is not None and len(audio_curve):
            score = float(CMATCH.find_best_shape_window(vals, len(vals), audio_curve)[0])
        out.append({
            "video_id": sp["video_id"], "scene_id": sp["scene_id"], "scene_start_sec": sp["scene_start_sec"],
            "window_offset_sec": off, "window_duration_sec": dur, "score": round(score, 3),
            "motion_norm": sp["motion_norm"],
            "curve_slice": vals if vals is not None else np.array([]), "fps": fps or 25.0,
            "tags": sp["tags"], "thumbnail": sp["thumbnail"],
            "intro_candidate": sp.get("intro_candidate", False),
            "outro_candidate": sp.get("outro_candidate", False),
            "roles": sp.get("roles", []), "planned": True,
        })
    return out


def with_planned(candidates: list, planned: list) -> list:
    """The grid's candidates with the planned ones in (a planned clip replaces the same scene's card)."""
    keys = {cand_key(c) for c in planned}
    rest = [c for c in candidates if not c.get("similar") and cand_key(c) not in keys]
    return (sorted(planned + rest, key=lambda c: -c["score"])            # rank stays the shape-score order
            + [c for c in candidates if c.get("similar")])


_mark("matching settings")
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
    # What the plan is built FROM (videos, cuts, ranges, story map…) — not the ⚙️ settings (they only
    # change through ▶️) nor Swap / Reject / Remove (they apply at once to a confirmed plan).
    _setup_sig = hashlib.sha1(json.dumps([
        track_id, [(round(sg["start"], 3), round(sg["end"], 3), round(sg.get("intensity", 0), 4)) for sg in segments],
        sorted(selected_videos), sorted(tag_filter), sorted((v, list(r) if r else None) for v, r in effective_time_ranges().items()),
        min_clips, max_clips, ramp_start, ramp_end, vary_count, int(count_seed), min_clip_len_sec,
        current_story_map(track_id), sorted(st.session_state.get("clip_count_skip") or []),
        [(v, _cat_mtime(v)) for v in sorted(selected_videos)],
    ], default=str).encode()).hexdigest()
    _bounds = [(round(sg["start"], 3), round(sg["end"], 3)) for sg in segments]
    if st.session_state.pop("_match_pressed", False):
        st.session_state["_match_confirmed_setup"] = _setup_sig
    _match_allowed = st.session_state.get("_match_confirmed_setup") == _setup_sig
    _auto_cache = st.session_state.get("_auto_plan_cache")
    _plan_stale = False
    if not _match_allowed:
        if _auto_cache and _auto_cache.get("bounds") == _bounds:
            # The last plan, kept for comparison until ▶️ Apply & match.
            timeline, clip_count_shortfall, duration_shortfall_sec = _auto_cache["result"]
            _auto_repeat_fallbacks = set(_auto_cache["fallbacks"])
            _plan_stale = True
        else:
            with _status_slot:
                if _auto_cache or st.session_state.get("_match_confirmed_setup"):
                    st.warning("⚠️ The videos or the cuts have changed since the last plan — press **▶️ Apply & "
                               "match** to build it again.")
                else:
                    st.info(f"**{len(segments)} segments** · {len(selected_videos)} videos · "
                            f"{total_footage_sec / 60:.0f} min of usable footage — press **▶️ Match** to build the "
                            f"plan.")
            _record_page_run()
            st.stop()
    elif _auto_cache and _auto_cache["sig"] == _auto_sig:
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
            "fallbacks": sorted(_auto_repeat_fallbacks), "bounds": _bounds}
    if _auto_repeat_fallbacks:
        _note(repeat_fallback_message([i + 1 for i in sorted(_auto_repeat_fallbacks)]))

    split_count = sum(1 for e in timeline if e["split_screen"])
    total_picks = sum(len(e["scenes"]) for e in timeline)
    _status_line = (f"**{len(timeline)} segments**" + (f" ({split_count} split-screen)" if split_count else "")
                    + f" · {len(selected_videos)} videos · {total_footage_sec / 60:.0f} min of usable footage"
                    + (" · in order" if sequential_mode else ""))
    if clip_count_shortfall:
        _note(f"Ran out of usable footage {clip_count_shortfall} time(s) — some segments have fewer "
              f"clips than requested. Widen your video/tag selection for a fuller plan.")
    if duration_shortfall_sec > 0.5:
        _note(f"{duration_shortfall_sec:.1f}s total across the plan could NOT be filled at all — every "
              f"scene in the library was exhausted at that point. Widen your video/tag selection.")
    if not clip_count_shortfall and duration_shortfall_sec <= 0.5:
        _status_line += f" · ✅ every segment filled, no gaps ({total_picks} slots)"
    if _plan_stale:
        _notes.insert(0, "**This is the last plan** — the videos have changed since. Press **▶️ Apply & match** "
                         "to update it (Save and Render wait until then).")

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
            _note(f"{len(chor_invalidated)} confirmed clip(s) in other blocks no longer match the "
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
        _note(f"{len(chor_invalidated)} previously confirmed clip(s) are no longer available "
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
    _status_line = (f"**{len(segments)} segments** · {len(selected_videos)} videos · "
                    f"{total_footage_sec / 60:.0f} min of usable footage · {len(chor_confirmed)} block(s) confirmed")

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
        _note(f"{len(invalidated)} previously-confirmed/skipped item(s) are no longer available "
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

    _status_line = (f"**{len(segments)} segments** · {len(selected_videos)} videos · "
                    f"{total_footage_sec / 60:.0f} min of usable footage · {len(confirmed)} block(s) confirmed")

with _status_slot:
    if _has_overrides:
        _sc = st.columns([4, 1.4])
        _sc[0].caption(_status_line)
        _n_rej = len(st.session_state["global_excluded_scenes"])
        _sc[1].button("🧹 Reset clip changes", on_click=_reset_overrides, width="stretch",
                      help="Clears Auto mode's per-block Swap / Reject / Remove changes"
                           + (f" and the {_n_rej} clip(s) rejected from this plan." if _n_rej else "."))
    else:
        st.caption(_status_line)
    if _notes:
        st.warning(_notes[0] if len(_notes) == 1 else
                   f"**{len(_notes)} things to check**\n\n" + "\n".join(f"- {n}" for n in _notes))

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

_mark("making the plan / mode screen")
fig = go.Figure()
if track.get("hires"):
    _rate = float(track["hires"]["rate"])
    _t = np.arange(len(track["hires"]["rms"])) / _rate
    fig.add_trace(go.Scatter(x=_t, y=np.asarray(track["hires"]["rms"], dtype=float), mode="lines", name="Track energy (30 Hz)",
                              line=dict(color="rgba(100,150,255,0.5)", width=1)))
    fig.add_trace(go.Scatter(x=_t, y=np.asarray(track["hires"]["onset"], dtype=float), mode="lines", name="Onset strength",
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
st.plotly_chart(fig, width="stretch")
_mark("chart")

# ---------------------------------------------------------------------------
# Timeline table + thumbnails preview
# ---------------------------------------------------------------------------

if matching_mode == "Auto":
    st.subheader("Matched timeline", help=(
        "🔄 Swap — replace with a different clip here; the original can still be used elsewhere.  "
        "🚫 Reject — replace it here too, but permanently ban the original from the whole plan.  "
        "✕ Remove — delete this slot (fewer simultaneous clips); the original can still be used elsewhere."
        + ("  🎭 The chart's shading is the story map's parts; each block card says which part it's in."
           if _story_regions else "")))
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

        # 📍 In order: the planned path's clip for each of its videos in this block.
        _chor_plan = ({p["video_id"]: p for p in planned_ahead(chor_block, queues).get(chor_block, [])}
                      if sequential_mode and chor_editing is None else {})

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
                if sequential_mode:
                    # 📍 Follow the planned path; a block it can't fill falls back to the picks below.
                    _placed_b = []
                    for _p in planned_ahead(b, queues).get(b, []):
                        _pc = planned_clip(queues, _p, block_duration_b)
                        if _pc is None:
                            continue
                        try:
                            _placed_b.append(carve_span(queues, _p["video_id"], _pc["scene_id"], _pc["offset_sec"],
                                                        block_duration_b, True))
                        except ValueError:
                            continue
                        st.session_state[f"chor_pick_{b}_{_p['video_id']}"] = True
                        st.session_state.setdefault("chor_overrides", {}).setdefault(b, {})[_p["video_id"]] = \
                            (_pc["scene_id"], _pc["offset_sec"])
                    if _placed_b:
                        for vid in selected_videos:
                            if all(vid != q["video_id"] for q in _placed_b):
                                st.session_state[f"chor_pick_{b}_{vid}"] = False
                        _prev_b = {p["video_id"] for p in chor_confirmed.get(b - 1, [])} if b > 0 else set()
                        if repeat_fallback_videos(_placed_b, _prev_b, avoid_repeat):
                            _chor_rep_fallbacks.append(b + 1)
                        chor_confirmed[b] = _placed_b
                        continue
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
                _pcs = {v: planned_clip(queues, p, block_duration) for v, p in _chor_plan.items()}
                _pcs = {v: c for v, c in _pcs.items() if c is not None}
                shown_videos = chor_shown(eligible_videos, _pcs, int(max_chor_videos))
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
                    clips = clips_with_planned(clips, _pcs.get(video_id))
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
                    elif video_id in _pcs:
                        # 📍 Auto on a planned video: its planned clip.
                        use_scene_id = _pcs[video_id]["scene_id"]
                        use_offset = _pcs[video_id]["offset_sec"]
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
            track=track, planned=_chor_plan,
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
                if sequential_mode:
                    # 📍 In order: follow the planned path; a block it can't fill falls back below.
                    _pl_b = planned_candidates(planned_ahead(b, queues).get(b, []), queues, seg_b, audio_curve_b)
                    _placed_b = []
                    for c in _pl_b:
                        try:
                            _placed_b.append(carve_span(queues, c["video_id"], c["scene_id"],
                                                        c["window_offset_sec"], c["window_duration_sec"], True))
                            st.session_state[f"adv_pick_{b}_{cand_key(c)}"] = True
                        except ValueError:
                            pass
                    if _placed_b:
                        _prev_b = {p["video_id"] for p in confirmed.get(b - 1, [])} if b > 0 else set()
                        if repeat_fallback_videos(_placed_b, _prev_b, avoid_repeat):
                            _adv_rep_fallbacks.append(b + 1)
                        confirmed[b] = _placed_b
                        continue
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

        # 📍 In order: the planned path's clips for this block join the grid (Auto-fill ticks them).
        _adv_planned = (planned_candidates(planned_ahead(current_block, queues).get(current_block, []),
                                           queues, seg, block_audio_curve)
                        if sequential_mode and adv_editing is None else [])
        if _adv_planned:
            candidates = with_planned(candidates, _adv_planned)
            st.caption(f"📍 **Planned path:** {len(_adv_planned)} clip(s) here — each video's clips are planned "
                       f"for all its remaining blocks together, so none runs short later. ⚡ Auto-fill ticks them; "
                       f"picking something else plans the rest again.")

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

# Attach resolved local file paths so the Blender add-on can load footage
# and audio directly, without re-deriving them from the catalogues.
for _entry in timeline:
    for _slot in _entry["scenes"]:
        for _link in _slot["chain"]:
            _link["source_path"] = colab_to_local(catalogues[_link["video_id"]]["source_path"])

_mark("block cards")
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

# 🏁 Finish: render a preview, save the plan, download it — together in one bar.
if matching_mode == "Auto" and _plan_stale:
    st.divider()
    st.caption("🏁 Save and Render come back once the plan is up to date (▶️ Apply & match above).")
    _record_page_run()
    st.stop()
st.divider()
st.subheader("Finish", help="🎬 Render a preview to watch the plan · 💾 Save the plan for the Blender add-on "
                            "(and Google Drive) · ⬇️ download the plan's JSON.")
_fin = st.columns([2, 1.5, 1.5, 0.7])
_render_clicked = _fin[0].button("🎬 Render Preview", type="primary", width="stretch")
_save_clicked = _fin[1].button("💾 Save plan", width="stretch")
_fin[2].download_button(
    "⬇️ Download plan", data=json.dumps(export_plan, indent=2), file_name=f"{track_id}_plan.json",
    mime="application/json", width="stretch", help="The plan as JSON, for the Blender add-on.")
_stream_sources = False
with _fin[3].popover("⚙️", width="stretch", help="Render options"):
    if _DRIVE_SYNC_AVAILABLE:
        _stream_sources = st.toggle(
            "Stream original videos from Google Drive", value=True, key="render_stream_sources",
            help="On: videos and the track that aren't on the server are read straight from Google Drive "
                 "while rendering — only the seconds each clip uses are fetched, and nothing is stored. "
                 "If a stream fails, that file is downloaded instead. Off: every missing file is "
                 "downloaded to the server first (the old way).",
        )
    else:
        st.caption("Rendering uses the files on this machine.")

if _save_clicked:
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

# ---------------------------------------------------------------------------
# Render preview — versioned, stored in Drive, viewable side-by-side
# ---------------------------------------------------------------------------

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


if _render_clicked:
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
        st.session_state["preview_just_rendered"] = out_path.name      # opens its player below
        st.rerun()
    except Exception as e:
        st.error(f"Render failed: {e}")

if st.session_state.get("render_last_stats"):
    st.caption("⏱️ " + st.session_state["render_last_stats"])
_drive_note = st.session_state.pop("render_drive_note", None)
if _drive_note:
    getattr(st, _drive_note[0])(_drive_note[1])

def _toggle_preview_player(name: str) -> None:
    """▶️ on a preview: show / hide its player — at most two at once (picking a third drops the oldest)."""
    shown = [n for n in st.session_state.get("preview_players", []) if n != name]
    if name not in st.session_state.get("preview_players", []):
        shown = (shown + [name])[-2:]
    st.session_state["preview_players"] = shown


_mark("save, render and download")
versions = existing_preview_versions()
if versions:
    # ⚡ A player is only made for the previews picked with ▶️: st.video given a file reads the whole
    # video into the app's memory and fingerprints it on every run — every preview, every click.
    if out_name := st.session_state.pop("preview_just_rendered", None):
        st.session_state["preview_players"] = [n for n in st.session_state.get("preview_players", [])
                                               if n != out_name][-1:] + [out_name]
    _shown = [n for n in st.session_state.get("preview_players", []) if any(p.name == n for p in versions)]
    st.session_state["preview_players"] = _shown
    st.caption(f"{len(versions)} preview(s) for this track — ▶️ play one, or two to compare side by side.")
    cols_per_row = 2
    for row_start in range(0, len(versions), cols_per_row):
        row = versions[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, path in zip(cols, row):
            with col:
                _meta = load_render(path)
                if _meta is None:
                    continue
                _on = path.name in _shown
                st.button(("⏹ Hide " if _on else "▶️ Play ") + path.name, key=f"play_preview_{path.name}",
                          on_click=_toggle_preview_player, args=(path.name,), width="stretch",
                          type="primary" if _on else "secondary")
                if _on:
                    show_render_player(_meta, height=300)
                st.caption(("☁️ on Drive" if _meta.get("drive") else "on the server only"))
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
