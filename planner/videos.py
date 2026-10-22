"""📼 Step 2 — choosing the videos (and their time ranges).

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

import random
import streamlit as st
import catalogue_store as CStore
import scene_roles as SR
from library_common import (safe_image, video_details, all_actors, all_video_tags, matches_details, render_range_picker, library_ranges, memory_check)
from core.matching import format_mmss

from planner.state import (
    fragment, rerun_fragment
)
from planner.data import (
    _add_video_to_selection, effective_time_ranges, find_best_motion_window, get_video_duration,
    seconds_to_time, video_list_thumbnail
)
from planner.projects import (
    autosave_now
)
from planner.story import (
    _add_new_videos
)
from planner.similar import (
    _COLUMNS_SUPPORT_VALIGN, compute_similar_videos, format_match_breakdown, RECOMMENDED_COUNT,
    render_ranking_weights, resolve_video_selection, video_role_seconds
)


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


def _clear_video_selection():
    for key in [k for k in st.session_state if k.startswith("select_video_")]:
        st.session_state[key] = False


@fragment
def render_video_selection_section(track: dict, all_video_ids: list, all_tag_options: list, catalogues: dict,
                                   simple: bool = False) -> None:
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
    if simple:
        # 🛠️ Simple: no scene-tag filter widget — the one set in Advanced still applies (and the status says so).
        tag_filter = [t for t in st.session_state.get("committed_tag_filter", []) if t in all_tag_options]
        top_cols = st.columns([2, 2, 2])
        _bc = top_cols
    else:
        top_cols = st.columns([5, 2, 2, 2])
        with top_cols[0]:
            tag_filter = st.multiselect(
                "Restrict to tags (optional)", all_tag_options, key="tag_filter",
                default=[t for t in st.session_state.get("committed_tag_filter", []) if t in all_tag_options],
                help="Only scenes carrying at least one of these tags are used — this also changes which videos "
                     "are recommended below.",
            )
        _bc = top_cols[1:]
        for c in _bc:
            c.write("")
    _bc[0].button("✅ Select all recommended", on_click=_select_all_recommended, width="stretch",
                  help=f"Tick the top {RECOMMENDED_COUNT} best-matching videos listed below.")
    _bc[1].button("Clear selection", on_click=_clear_video_selection, width="stretch")
    _bc[2].button("🎲 Randomize", on_click=_randomize_top_videos, width="stretch",
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
    if not simple:
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
    sort_by = list(_SORT_OPTIONS)[0] if simple else st.selectbox(
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
        _hc[1].button("↩️ Back to best matches", on_click=_back_to_best_matches, width="stretch")
        st.caption(f"Picked at random from all {len(video_matches)} videos with usable scenes (seed {_seed}) — "
                   f"the ranking weights aren't used. Match scores are still shown for reference." +
                   (f"  |  filtered to tags: {', '.join(tag_filter)}" if tag_filter else ""))
    else:
        _legend = ("✅ marks a recommended video. ☑ is your actual selection — nothing is selected until you tick it "
                   "(or use Select all recommended above), and it stays as you set it even if re-sorting moves the "
                   "video. ✨ finds similar videos. 🎚️ opens a range picker to restrict a video to just a portion of "
                   "its footage (🛠️ Advanced). 📚 is the video's Media Library range (used by default); 🎚️ next to a "
                   "time is a range set here, which overrides the library range for this project only.")
        st.subheader(f"Best-matching videos (top {RECOMMENDED_COUNT})", help=_legend)
        if tag_filter:
            st.caption(f"Only scenes tagged: {', '.join(tag_filter)}"
                       + (" (set in 🛠️ Advanced)" if simple else ""))
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
                if not simple:
                    st.caption(format_match_breakdown(match, bool(st.session_state.get("ranking_experimental")),
                                                      tag_filter))
            else:
                st.write(f"{title} — no scenes match the current tag filter / time range{range_tag}")
            _rs = (match or {}).get("role_sec") or _role_secs_now.get(video_id) or {}
            _vd = _details.get(video_id) or {}
            _bits = []
            if _vd.get("actors"):
                _bits.append("👤 " + ", ".join(_vd["actors"]))
            if _vd.get("tags"):
                _bits.append("🏷️ " + ", ".join(_vd["tags"]))
            if _rs:
                _rb = SR.roles_by_id()
                _bits.append("🎭 " + "  ·  ".join(f"{SR.label(r, _rb)} {format_mmss(v)}"
                                                  for r, v in sorted(_rs.items(), key=lambda kv: -kv[1])))
            if _bits:
                st.caption("  ·  ".join(_bits))
        with row_cols[3]:
            if st.button("✨", key=f"sim_toggle_{video_id}", help="Find videos similar to this one"):
                st.session_state["show_similar_videos"] = (
                    None if st.session_state.get("show_similar_videos") == video_id else video_id)
                rerun_fragment()
            if not simple and st.button("🎚️", key=f"range_toggle_{video_id}", help="Set a time range for this video"):
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
                                  width="stretch")

        if not simple and st.session_state["show_range_picker"].get(video_id):
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
                      disabled=not _not_ticked, width="stretch", key="add_new_videos",
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
        st.caption("☑ No videos selected yet — tick them above, use **Select all recommended**, or add others from "
                   "the dropdown.")
    else:
        st.caption(f"✅ **{len(selected_now)} video(s) selected** — next: 🎬 Matching & Export.")

