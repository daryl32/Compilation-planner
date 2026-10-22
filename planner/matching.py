"""🎬 Matching helpers: the matcher wrappers, candidate grids, Choreography blocks, swap for similar.

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

import inspect
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from render_preview import colab_to_local
import scene_roles as SR
import preview_snippets
from library_common import (safe_image, proxy_path, shared_cache, memory_check)
import core.picking as CPICK
import core.matching as CMATCH
from core.matching import prefer_role, cand_key, format_mmss
from core.shape import SHAPE_MATCH_POINTS, resample_curve
from core.footage import has_role
from core.picking import select_autofill_picks, repeat_fallback_videos
from config import CATALOGUE_DIR, PREVIEW_DIR

from planner.state import (
    download_source_video, _DRIVE_SYNC_AVAILABLE, fragment, rerun_fragment, rerun_full
)
from planner.data import (
    effective_time_ranges, get_nearest_thumbnail, get_video_duration, load_motion_curves_for_video,
    _local_media
)
from planner.projects import (
    autosave_now
)
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
        planned_sequential=True, shape_memo=_shape_memo(),
    )


@shared_cache
@st.cache_resource
def _shape_memo() -> dict:
    """⚡ Shape-window scores kept between matches, shared by every session: matching again with
    the same cuts and videos (after changing weights, order, F…) skips the heaviest work. Keyed on
    the curves' own numbers, so it can't go stale; capped in core.matching (SHAPE_MEMO_MAX)."""
    return {}


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
    role_filter: str = None, weighting: dict = None, track: dict = None, planned: dict = None,
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

    planned: {video_id: pick} — the 📍 planned path's clips for this block (in order); a planned
    video's column shows its planned clip as the Auto pick.

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
    planned = planned or {}
    _pclips = {vid: planned_clip(queues, p, block_duration) for vid, p in planned.items()}
    _pclips = {vid: c for vid, c in _pclips.items() if c is not None and vid in _clips_by_vid}
    for vid, c in _pclips.items():
        _clips_by_vid[vid] = clips_with_planned(_clips_by_vid[vid], c)
    eligible = [vid for vid in selected_videos if _clips_by_vid[vid]]
    shown = chor_shown(eligible, _pclips, max_chor_videos)
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
    if _pclips:
        _pl_cols = st.columns([2, 3])
        with _pl_cols[0]:
            if st.button(f"📍 Use the planned picks ({len(_pclips)})", key=f"chor_use_plan_{seg_idx}",
                         width="stretch",
                         help="Tick the planned path's videos (each on its planned clip) and untick the rest. "
                              "Each video's clips are planned for all its remaining blocks together, in order, "
                              "so none runs short later — picking something else plans the rest again."):
                for vid in shown:
                    st.session_state[f"chor_pick_{seg_idx}_{vid}"] = vid in _pclips
                    if vid in _pclips:
                        overrides.pop(vid, None)
                rerun_fragment()
        with _pl_cols[1]:
            st.caption("📍 **Planned path:** " + ", ".join(f"`{v}`" for v in _pclips)
                       + " — their Auto picks are the planned clips.")
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
                if display_clip is None and video_id in _pclips:
                    display_clip = next((c for c in clips if c.get("planned")), None)   # 📍
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
                                     width="stretch", disabled=_off, help=_tip):
                            _chor_step(overrides, video_id, clips, sel_idx, _step)
                            rerun_fragment()
                with nav[2]:
                    if st.button("↺", key=f"chor_auto_{seg_idx}_{video_id}", width="stretch",
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
                    auto_tag = (("📍 Planned pick — " if display_clip.get("planned") else "⭐ Auto pick — ")
                                if current_key is None else "📍 " if display_clip.get("planned") else "")
                    role_suffix = f"  {role_chips(display_clip)}" if display_clip.get("roles") else ""
                    pos = f"  ·  {sel_idx + 1}/{len(clips)}" if sel_idx is not None else ""
                    st.caption(
                        f"{auto_tag}Scene {display_clip['scene_id']}  "
                        f"{format_mmss(display_clip['scene_start_sec'] + display_clip['offset_sec'])}  "
                        f"motion {display_clip['motion_norm']:.2f}{role_suffix}{pos}"
                    )

                _is_open = st.session_state.get("chor_preview") == (seg_idx, video_id)
                if st.button("⏹ Close preview" if _is_open else "🎵 Preview with music",
                             key=f"chor_prevw_{seg_idx}_{video_id}", width="stretch",
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
                if st.button(_lbl, key=f"chor_p{_step:+d}_{seg_idx}_{video_id}", width="stretch",
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
                      width="stretch", on_click=_use_and_close, disabled=clip is None)
        with cols[5]:
            if st.button("Close", key=f"chor_pclose_{seg_idx}_{video_id}", width="stretch"):
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
            _plan = [c for c in candidates if c.get("planned")]
            if _plan:
                # 📍 In order: the planned path's clips (it already weighed everything above).
                chosen = _plan
            else:
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
                if c.get("planned"):
                    st.caption("📍 on the planned path")
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
                                             width="stretch",
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
        st.plotly_chart(vfig, width="stretch")


def _planned_span(q: dict, p: dict, dur: float):
    """(span, offset in scene) of the free footage holding planned pick p, or (None, None)."""
    for sp in q.get(p["video_id"], []):
        if sp["scene_id"] != p["scene_id"]:
            continue
        lo, hi = float(sp["offset_sec"]), float(sp["offset_sec"]) + float(sp["remaining_sec"])
        off = float(p["offset_into_scene_sec"])
        # (within carve_span's snap — a frame either way, as the plan's own carving allows)
        if lo - 0.05 <= off and off + dur <= hi + 0.05:
            return sp, max(lo, min(off, hi - dur))
    return None, None


def planned_clip(q: dict, p: dict, dur: float):
    """Choreography: planned pick p as one of its video's trimmed clips (marked 📍), or None."""
    sp, off = _planned_span(q, p, dur)
    if sp is None:
        return None
    abs_start = sp["scene_start_sec"] + off
    role_tag = f"  {role_chips(sp)}" if sp.get("roles") else ""
    return {**sp, "offset_sec": off, "abs_start": abs_start, "planned": True,
            "label": f"📍 Scene {sp['scene_id']}  {format_mmss(abs_start)} – {format_mmss(abs_start + dur)}"
                     f"  motion {sp['motion_norm']:.2f}{role_tag}"}


def clips_with_planned(clips: list, pclip) -> list:
    """A video's trimmed clips with its planned clip in its place in time (replacing one at the same spot)."""
    if pclip is None:
        return clips
    rest = [c for c in clips if not (c["scene_id"] == pclip["scene_id"]
                                     and abs(c["offset_sec"] - pclip["offset_sec"]) < 0.1)]
    return sorted(rest + [pclip], key=lambda c: c["abs_start"])


def chor_shown(eligible: list, planned_vids, n: int) -> list:
    """The columns shown: the planned path's videos first (so they always get one), then the rest."""
    first = [v for v in eligible if v in planned_vids]
    return (first + [v for v in eligible if v not in planned_vids])[:max(n, len(first))]

