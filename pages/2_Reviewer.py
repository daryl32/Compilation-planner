"""
Reviewer — browse and correct auto-generated scene tags.
"""

import json
import math
import tempfile
from pathlib import Path

import streamlit as st

from config import CATALOGUE_DIR
from library_common import (scene_tags, mark_pending, master_thumbnail_scenes, set_master_thumbnail,
                            scene_thumbnail_path, proxy_path,
                            library_ranges, tc_to_seconds, format_mmss, overlap_with_range)
import scene_edits as SE


def load_labels() -> list:
    labels_path = CATALOGUE_DIR / "labels.json"
    if labels_path.exists():
        return json.loads(labels_path.read_text())
    return []

CANDIDATE_LABELS = load_labels()

# Sign-in, whitelist, Drive connect, title and the shared sidebar — see page_setup.py
from page_setup import (page_setup, push_file_with_oauth,
                        OAUTH_AVAILABLE as _OAUTH_AVAILABLE, DRIVE_SESSION_KEY as _OAUTH_SESSION_KEY)

OUTPUT_DIR = CATALOGUE_DIR

def list_catalogue_files():
    """Only real video catalogues (dicts with video_id + scenes); skips labels.json etc."""
    files = []
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and "video_id" in d and "scenes" in d:
            files.append(f)
    return files


page_setup("Reviewer", "pages/2_Reviewer.py", anchor="reviewer-top")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Process a video")
st.sidebar.link_button("▶️ Open Google Colab", "https://colab.research.google.com", use_container_width=True)
st.sidebar.divider()
review_mode = st.sidebar.checkbox("Review / correction mode", value=False)
use_library_range = st.sidebar.toggle(
    "Use library range", value=True,
    help="Only show scenes inside the video's Media Library time range. "
         "Turn off to see every scene in the original video.",
)
st.sidebar.divider()

# Split / combine suggestions
st.sidebar.header("Split / combine suggestions")
show_suggestions = st.sidebar.toggle("Show suggestions", value=True)
suggest_long_sec = st.sidebar.number_input(
    "Suggest splitting clips longer than (seconds)", min_value=5.0, value=SE.DEFAULT_LONG_SEC, step=10.0,
    disabled=not show_suggestions,
)
suggest_short_sec = st.sidebar.number_input(
    "Suggest combining runs of clips shorter than (seconds)", min_value=0.1, value=SE.DEFAULT_SHORT_SEC, step=0.5,
    disabled=not show_suggestions,
    help="Flags two or more touching clips in a row that are each shorter than this.",
)
st.sidebar.divider()

# Add new label
st.sidebar.header("Manage Labels")
new_label = st.sidebar.text_input("Add new label", placeholder="e.g. slow-motion")
if st.sidebar.button("Add label") and new_label:
    new_label = new_label.strip().lower()
    if new_label in CANDIDATE_LABELS:
        st.sidebar.warning("Label already exists.")
    elif not st.session_state.get(_OAUTH_SESSION_KEY):
        st.sidebar.warning("Connect Google Drive first to save labels.")
    else:
        updated = CANDIDATE_LABELS + [new_label]
        labels_json = json.dumps(updated)
        tmp = Path("/tmp/labels.json")
        tmp.write_text(labels_json)
        error = push_file_with_oauth(
            st.session_state[_OAUTH_SESSION_KEY],
            tmp,
            "scene-labeling/catalogue",
        )
        if error:
            st.sidebar.error(f"Drive upload failed: {error}")
        else:
            try:
                from drive_sync import sync_pull
                from config import CATALOGUE_DIR, AUDIO_DIR
                sync_pull(CATALOGUE_DIR, AUDIO_DIR)
                CANDIDATE_LABELS.append(new_label)
                st.sidebar.success(f'Added "{new_label}" — label is now available.')
            except Exception as e:
                st.sidebar.warning(f'Label saved to Drive but sync failed: {e}. Restart the app to see it.')

# ---------------------------------------------------------------------------
# Export training data
# ---------------------------------------------------------------------------
def export_training_data():
    examples = []
    for cat_file in list_catalogue_files():
        data = json.loads(cat_file.read_text())
        for scene in data["scenes"]:
            if scene.get("reviewed") and not scene.get("excluded") and scene["thumbnail_paths"]:
                examples.append({
                    "image_path": str(OUTPUT_DIR / "thumbnails" / data["video_id"] / Path(scene["thumbnail_paths"][0]).name),
                    "tags": scene.get("corrected_tags") or scene.get("tags", []),
                })
    out_path = OUTPUT_DIR / "training_data.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples), out_path

if st.sidebar.button("Export corrections as training data"):
    count, out_path = export_training_data()
    if count == 0:
        st.sidebar.warning("No reviewed scenes yet — correct some scenes first.")
    else:
        st.sidebar.success(f"Exported {count} reviewed examples to {out_path}")

# ---------------------------------------------------------------------------
# Main: pick an existing catalogue and browse/correct it
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
catalogue_files = list_catalogue_files()

if not catalogue_files:
    st.info("No catalogues yet. Process a video from the sidebar to get started.")
else:
    names = [f.stem for f in catalogue_files]
    selected = st.selectbox("Catalogue", names)
    _saved_msg = st.session_state.pop("rev_saved_msg", None)
    if _saved_msg:
        st.success(_saved_msg)
    cat_path = OUTPUT_DIR / f"{selected}.json"
    data = json.loads(cat_path.read_text())

    reviewed_count = sum(1 for s in data["scenes"] if s.get("reviewed"))
    st.caption(f"Source: {data['source_path']}  |  {len(data['scenes'])} scenes  |  {reviewed_count} reviewed")

    # ⭐ Master thumbnail — the picture used for this video in the Media Library
    # and the planner's video list. Chosen from the scenes below; saved at once.
    video_id = data["video_id"]
    master_sid = master_thumbnail_scenes().get(video_id)
    _scene_by_id = {s["scene_id"]: s for s in data["scenes"]}
    with st.container(border=True):
        mt_cols = st.columns([1, 3])
        _mp = scene_thumbnail_path(video_id, _scene_by_id[master_sid]) if master_sid in _scene_by_id else None
        with mt_cols[0]:
            if _mp:
                st.image(str(_mp), width=160)
        with mt_cols[1]:
            if _mp:
                st.markdown(f"**⭐ Video thumbnail:** scene {master_sid}")
                if st.button("Use automatic instead", key=f"master_thumb_clear_{selected}"):
                    set_master_thumbnail(video_id, None)
                    st.session_state["rev_saved_msg"] = "Video thumbnail set to automatic."
                    st.rerun()
            else:
                st.markdown("**⭐ Video thumbnail:** automatic")
            st.caption("Pick one with ⭐ on any scene below. It's used for this video in the Media "
                       "Library and the Compilation Planner's video list.")

    lib_range = library_ranges().get(data["video_id"]) if use_library_range else None
    if lib_range:
        _outside = sum(
            1 for s in data["scenes"]
            if overlap_with_range(tc_to_seconds(s["start_tc"]), tc_to_seconds(s["end_tc"]), lib_range)[0] is None
        )
        st.info(f"📚 Showing the library range {format_mmss(lib_range[0])}–{format_mmss(lib_range[1])}"
                + (f" — {_outside} scene(s) outside it are hidden." if _outside else "."))
    elif use_library_range:
        st.caption("No library range set for this video — showing every scene.")

    # -----------------------------------------------------------------------
    # Split / combine (saved straight away, like the ⭐ button)
    # -----------------------------------------------------------------------
    thumbs_dir = OUTPUT_DIR / "thumbnails" / video_id
    sprite_path = OUTPUT_DIR / "timeline_sprites" / f"{video_id}.jpg"
    ordered_all = SE.sorted_scenes(data["scenes"])
    scene_by_id = {s["scene_id"]: s for s in data["scenes"]}
    next_of = {a["scene_id"]: b["scene_id"] for a, b in zip(ordered_all, ordered_all[1:])}

    def fmt_dur(sec: float) -> str:
        return f"{sec:.1f}s" if sec < 60 else format_mmss(sec)

    def fmt_precise(sec: float) -> str:
        m, s = divmod(max(0.0, sec), 60)
        h, m = divmod(int(m), 60)
        return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"

    def video_paths() -> list:
        """Preview copy first (fast to seek), then the source video — whichever exist here."""
        paths = [proxy_path(video_id)]
        try:
            from render_preview import colab_to_local
            paths.append(Path(colab_to_local(data.get("source_path", ""))))
        except Exception:
            pass
        return [p for p in paths if p and Path(p).exists()]

    def save_edit(message: str, new_thumbs=()) -> None:
        """Write the catalogue, queue it (and any new thumbnails) for Drive, rerun."""
        present = {s["scene_id"] for s in data["scenes"]}
        if master_sid is not None and master_sid not in present:
            # The ⭐ scene was combined into another — keep using that one.
            holder = next((s for s in data["scenes"]
                           if any(o.get("scene_id") == master_sid
                                  for o in (s.get("edit") or {}).get("originals", []))), None)
            set_master_thumbnail(video_id, holder["scene_id"] if holder else None)
        cat_path.write_text(json.dumps(data, indent=2))
        mark_pending(cat_path.stem)
        for t in new_thumbs:
            mark_pending(f"thumb:{video_id}/{Path(t).name}")
        st.cache_data.clear()  # so the Compilation Planner sees the change straight away
        st.session_state["rev_saved_msg"] = message
        st.rerun()

    def do_merge(ids: list) -> None:
        try:
            new_id = SE.merge_scenes(data, ids)
        except SE.EditError as e:
            st.error(str(e))
            return
        save_edit(f"Combined {len(ids)} scenes into scene {new_id}.")

    def do_undo(sid) -> None:
        try:
            msg = SE.undo_edit(data, sid)
        except SE.EditError as e:
            st.error(str(e))
            return
        save_edit(msg)

    def dismiss(key: str) -> None:
        lst = data.setdefault("dismissed_edit_suggestions", [])
        if key not in lst:
            lst.append(key)
        save_edit("Suggestion dismissed.")

    def frame_at(t: float):
        """A picture of the video at t: an exact frame if a video is here, else the
        nearest timeline-sprite tile. None if neither is available."""
        vids = video_paths()
        if vids:
            dest = Path(tempfile.gettempdir()) / "reviewer_frames" / f"{video_id}_{int(round(t * 10))}.jpg"
            if dest.exists() or SE.grab_frame(vids[0], t, dest, width=480):
                return str(dest)
        return SE.sprite_tile(data, sprite_path, t)

    def _set_state(key, value):
        st.session_state[key] = value

    def _add_cut(cuts_key, pos_key):
        cuts = st.session_state.setdefault(cuts_key, [])
        t = round(float(st.session_state[pos_key]), 1)
        if t not in cuts:
            cuts.append(t)
            cuts.sort()

    def _remove_cut(cuts_key, t):
        st.session_state[cuts_key] = [c for c in st.session_state.get(cuts_key, []) if c != t]

    @st.dialog("✂️ Split scene", width="large")
    def split_dialog(sid):
        scene = scene_by_id[sid]
        start, end = SE.scene_start_sec(scene), SE.scene_end_sec(scene)
        cuts_key, pos_key = f"split_cuts_{video_id}_{sid}", f"split_pos_{video_id}_{sid}"
        cuts = st.session_state.setdefault(cuts_key, [])
        st.markdown(f"**Scene {SE.scene_label(scene)}** · {fmt_precise(start)} → {fmt_precise(end)} "
                    f"({fmt_dur(end - start)})")

        lo = math.ceil((start + SE.MIN_PART_SEC) * 10) / 10
        hi = math.floor((end - SE.MIN_PART_SEC) * 10) / 10
        if hi <= lo:
            st.warning("This scene is too short to split.")
            return

        vids = video_paths()
        if vids:
            st.video(str(vids[0]), start_time=int(start), end_time=int(math.ceil(end)))
            st.caption("Play the scene to find the moment, then move the slider there.")

        if pos_key not in st.session_state:
            st.session_state[pos_key] = min(hi, max(lo, round((start + end) / 2, 1)))
        suggested = [t for t in SE.cut_suggestions_from_sprite(data, sprite_path, scene) if lo <= t <= hi]
        if suggested:
            st.caption("💡 Big picture changes found in the timeline thumbnails:")
            bcols = st.columns(len(suggested))
            for i, t in enumerate(suggested):
                bcols[i].button(f"Go to {fmt_precise(t)}", key=f"{pos_key}_go{i}",
                                on_click=_set_state, args=(pos_key, t))

        pos = st.slider("Cut at (seconds into the video)", min_value=float(lo), max_value=float(hi),
                        step=0.1, format="%.1f", key=pos_key)
        st.caption(f"{fmt_precise(pos)} — {pos - start:.1f}s into the scene")
        img = frame_at(pos)
        if img is not None:
            st.image(img, width=360, caption="Frame at the cut (first frame of the next part)")
        else:
            st.caption("No preview frame available (no video or timeline thumbnails here).")

        bc = st.columns(2)
        bc[0].button("➕ Add cut here", key=f"{pos_key}_add", on_click=_add_cut, args=(cuts_key, pos_key),
                     use_container_width=True)
        bc[1].button("Clear cuts", key=f"{pos_key}_clear", on_click=_set_state, args=(cuts_key, []),
                     disabled=not cuts, use_container_width=True)

        if cuts:
            bounds = [start] + cuts + [end]
            st.markdown("**Parts:**")
            for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
                pc = st.columns([4, 1])
                pc[0].markdown(f"{i + 1}. {fmt_precise(a)} → {fmt_precise(b)} ({fmt_dur(b - a)})")
                if i < len(cuts):
                    pc[1].button("✖ cut", key=f"{pos_key}_rm{i}", on_click=_remove_cut, args=(cuts_key, cuts[i]),
                                 help=f"Remove the cut at {fmt_precise(cuts[i])}")
            st.caption("Each part starts unreviewed with this scene's tags. Undo is available afterwards. "
                       "Compilation Planner projects that already picked clips from this scene may need "
                       "those picks redone.")
        if st.button(f"✂️ Split into {len(cuts) + 1} parts" if cuts else "✂️ Split",
                     type="primary", disabled=not cuts, use_container_width=True, key=f"{pos_key}_go"):
            try:
                ids = SE.split_scene(data, sid, cuts)
            except SE.EditError as e:
                st.error(str(e))
                return
            new_thumbs = []
            with st.spinner("Making thumbnails for the new parts…"):
                for s in data["scenes"]:
                    if s["scene_id"] in ids[1:]:
                        p = SE.make_part_thumbnail(data, s, thumbs_dir, vids, sprite_path)
                        if p:
                            new_thumbs.append(p)
            st.session_state.pop(cuts_key, None)
            st.session_state.pop(pos_key, None)
            save_edit(f"Scene {sid} split into {len(ids)} parts (scenes {', '.join(map(str, ids))}).", new_thumbs)

    # Suggestions — only for scenes inside the library range (when it's in use).
    in_range = [
        s for s in data["scenes"]
        if not lib_range
        or overlap_with_range(tc_to_seconds(s["start_tc"]), tc_to_seconds(s["end_tc"]), lib_range)[0] is not None
    ]
    if show_suggestions:
        sugg = SE.suggest_edits(in_range, suggest_long_sec, suggest_short_sec,
                                data.get("dismissed_edit_suggestions", []))
    else:
        sugg = {"split": [], "merge": []}
    split_sugg = set(sugg["split"])
    merge_group_of = {sid: grp for grp in sugg["merge"] for sid in grp}
    if sugg["split"] or sugg["merge"]:
        parts_txt = []
        if sugg["split"]:
            parts_txt.append(f"{len(sugg['split'])} long clip(s) could be split")
        if sugg["merge"]:
            parts_txt.append(f"{len(sugg['merge'])} run(s) of short clips could be combined")
        st.info("💡 " + " · ".join(parts_txt)
                + (" — turn off Review / correction mode to split or combine." if review_mode else "."))

    tag_filter = st.text_input("Filter by tag (optional)")
    filter_cols = st.columns(3)
    with filter_cols[0]:
        show_only_excluded = st.checkbox("Show only excluded scenes", value=False)
    with filter_cols[1]:
        show_only_intro_outro = st.checkbox("Show only intro/outro candidates", value=False)
    with filter_cols[2]:
        show_only_suggested = st.checkbox("Show only split/combine suggestions", value=False,
                                          disabled=not show_suggestions)

    # Which scenes to show (filters and the library range), in time order.
    shown = []
    for scene in ordered_all:
        current_tags = scene_tags(scene)
        trimmed_to = None
        if lib_range:
            _s, _e = tc_to_seconds(scene["start_tc"]), tc_to_seconds(scene["end_tc"])
            _eff_s, _eff_e = overlap_with_range(_s, _e, lib_range)
            if _eff_s is None:
                continue
            if (_eff_s, _eff_e) != (_s, _e):
                trimmed_to = (_eff_s, _eff_e)
        if tag_filter and tag_filter.lower() not in [t.lower() for t in current_tags]:
            continue
        if show_only_excluded and not scene.get("excluded"):
            continue
        if show_only_intro_outro and not (scene.get("intro_candidate") or scene.get("outro_candidate")):
            continue
        if show_only_suggested and show_suggestions and not (
                scene["scene_id"] in split_sugg or scene["scene_id"] in merge_group_of):
            continue
        shown.append((scene, current_tags, trimmed_to))

    def scene_header(scene, trimmed_to, star_button: bool = False):
        """Thumbnail and status badges (left column). star_button: offer
        '⭐ Use as video thumbnail' (not possible inside the review form)."""
        if scene["thumbnail_paths"]:
            thumb_path = OUTPUT_DIR / "thumbnails" / data["video_id"] / Path(scene["thumbnail_paths"][0]).name
            if thumb_path.exists():
                st.image(str(thumb_path), width=280)
            else:
                st.warning(f"Thumbnail not found: {thumb_path}")
        if scene["scene_id"] == master_sid:
            st.success("⭐ Video thumbnail")
        elif star_button and scene["thumbnail_paths"]:
            if st.button("⭐ Use as video thumbnail", key=f"star_{selected}_{scene['scene_id']}"):
                set_master_thumbnail(video_id, scene["scene_id"])
                st.session_state["rev_saved_msg"] = f"Video thumbnail set to scene {scene['scene_id']}."
                st.rerun()
        if trimmed_to:
            st.warning(f"✂️ Trimmed by library range — uses {format_mmss(trimmed_to[0])}–{format_mmss(trimmed_to[1])}")
        if scene.get("reviewed"):
            st.success("✓ Reviewed")
        if scene.get("excluded"):
            st.error("🚫 Excluded from planner")
        if scene.get("intro_candidate"):
            st.info("🎬 Intro candidate")
        if scene.get("outro_candidate"):
            st.info("🎬 Outro candidate")

    def scene_title(scene) -> str:
        dur = SE.scene_duration(scene)
        return (f"**Scene {SE.scene_label(scene)}**  ·  {scene['start_tc']} → {scene['end_tc']}"
                f"  ·  {fmt_dur(dur)}")

    def merge_banner(grp: list, buttons: bool) -> None:
        total = sum(SE.scene_duration(scene_by_id[i]) for i in grp)
        text = (f"🔗 **Possible over-split:** scenes {grp[0]}–{grp[-1]} are {len(grp)} touching clips, "
                f"{fmt_dur(total)} in total, each under {suggest_short_sec:g}s.")
        if not buttons:
            st.info(text)
            return
        with st.container(border=True):
            st.markdown(text)
            bc = st.columns([2, 1, 3])
            if bc[0].button(f"🔗 Combine these {len(grp)}", key=f"merge_run_{selected}_{grp[0]}", type="primary"):
                do_merge(grp)
            if bc[1].button("Dismiss", key=f"merge_dismiss_{selected}_{grp[0]}"):
                dismiss(SE.suggestion_key("merge", grp))

    def edit_buttons(scene) -> None:
        sid = scene["scene_id"]
        if sid in split_sugg:
            wc = st.columns([4, 1])
            wc[0].warning(f"✂️ {fmt_dur(SE.scene_duration(scene))} long — consider splitting.")
            if wc[1].button("Dismiss", key=f"split_dismiss_{selected}_{sid}"):
                dismiss(SE.suggestion_key("split", [sid]))
        bc = st.columns(3)
        if bc[0].button("✂️ Split…", key=f"split_{selected}_{sid}", use_container_width=True):
            split_dialog(sid)
        nxt = next_of.get(sid)
        if nxt is not None and bc[1].button("🔗 Combine with next", key=f"merge_next_{selected}_{sid}",
                                            use_container_width=True,
                                            help=f"Join this scene and scene {nxt} into one."):
            do_merge([sid, nxt])
        edit = scene.get("edit") or {}
        if edit.get("kind") == "merge" or edit.get("kind") == "split" or "split_from" in scene:
            label = "↩️ Undo combine" if edit.get("kind") == "merge" else "↩️ Undo split"
            if bc[2].button(label, key=f"undo_{selected}_{sid}", use_container_width=True):
                do_undo(sid)

    BACK_TO_TOP = (
        '<a href="#reviewer-top" target="_self" style="display:inline-block;padding:0.4rem 0.9rem;'
        'border:1px solid rgba(128,128,128,0.4);border-radius:0.5rem;text-decoration:none;">'
        '⬆️ Back to top</a>'
    )

    if not shown:
        st.info("No scenes match these filters.")
    elif not review_mode:
        for scene, current_tags, trimmed_to in shown:
            grp = merge_group_of.get(scene["scene_id"])
            if grp and grp[0] == scene["scene_id"]:
                merge_banner(grp, buttons=True)
            cols = st.columns([1, 2])
            with cols[0]:
                scene_header(scene, trimmed_to, star_button=True)
            with cols[1]:
                st.markdown(scene_title(scene))
                st.markdown("Tags: " + ", ".join(f"`{t}`" for t in current_tags))
                edit_buttons(scene)
            st.divider()
        st.markdown(BACK_TO_TOP, unsafe_allow_html=True)
    else:
        # Everything below is one form: ticking boxes and editing tags doesn't
        # rerun the page — nothing is sent to the server until a Save button.
        with st.form(f"review_form_{selected}", border=False):
            def save_row(where: str):
                c = st.columns([3, 2, 2])
                with c[0]:
                    st.caption(f"{len(shown)} scene(s) shown. Edits are kept on this page until you save — "
                               "changing the video or a filter first discards them.")
                with c[1]:
                    st.checkbox("Also mark unchanged scenes reviewed", key=f"rev_mark_all_{where}",
                                help="Off: only scenes you changed are marked ✓ Reviewed. "
                                     "On: every scene shown is marked reviewed (you've checked them all).")
                with c[2]:
                    # Labels differ top/bottom so the two buttons get distinct widget ids.
                    label = "💾 Save all changes for this video" if where == "top" else "💾 Save all changes"
                    return st.form_submit_button(label, type="primary", use_container_width=True)

            save_top = save_row("top")
            st.divider()

            for scene, current_tags, trimmed_to in shown:
                sid = scene["scene_id"]
                grp = merge_group_of.get(sid)
                if grp and grp[0] == sid:
                    merge_banner(grp, buttons=False)
                cols = st.columns([1, 2])
                with cols[0]:
                    scene_header(scene, trimmed_to)
                with cols[1]:
                    st.markdown(scene_title(scene))
                    if sid in split_sugg:
                        st.warning(f"✂️ {fmt_dur(SE.scene_duration(scene))} long — consider splitting "
                                   "(turn off Review mode to split).")
                    st.multiselect(
                        "Tags", options=sorted(set(CANDIDATE_LABELS) | set(current_tags)),
                        default=current_tags, key=f"tags_{selected}_{sid}",
                    )
                    st.checkbox("Exclude from planner (e.g. intro/outro clip)",
                                value=scene.get("excluded", False), key=f"excluded_{selected}_{sid}")
                    if scene.get("thumbnail_paths"):
                        st.checkbox("⭐ Use as video thumbnail", value=(sid == master_sid),
                                    key=f"thumb_{selected}_{sid}",
                                    help="Saved with the button. If you tick more than one, the last "
                                         "one ticked further down the page wins.")
                    role_cols = st.columns(2)
                    with role_cols[0]:
                        st.checkbox(
                            "Intro candidate", value=scene.get("intro_candidate", False),
                            key=f"intro_{selected}_{sid}",
                            help="Available to the Compilation Planner's first block when its "
                                 "'Prefer Intro/Outro-tagged clips' toggle is on. Independent of "
                                 "Exclude — this scene can still show up normally elsewhere too, "
                                 "unless you also exclude it above.",
                        )
                    with role_cols[1]:
                        st.checkbox(
                            "Outro candidate", value=scene.get("outro_candidate", False),
                            key=f"outro_{selected}_{sid}",
                            help="Available to the Compilation Planner's last block when its "
                                 "'Prefer Intro/Outro-tagged clips' toggle is on. Independent of "
                                 "Exclude — this scene can still show up normally elsewhere too, "
                                 "unless you also exclude it above.",
                        )
                st.divider()

            save_bottom = save_row("bottom")
            st.markdown(BACK_TO_TOP, unsafe_allow_html=True)

        if save_top or save_bottom:
            mark_all = st.session_state.get("rev_mark_all_top") or st.session_state.get("rev_mark_all_bottom")
            changed = 0
            for scene, current_tags, _ in shown:
                sid = scene["scene_id"]
                new = {
                    "tags": list(st.session_state.get(f"tags_{selected}_{sid}", current_tags)),
                    "excluded": bool(st.session_state.get(f"excluded_{selected}_{sid}", scene.get("excluded", False))),
                    "intro_candidate": bool(st.session_state.get(f"intro_{selected}_{sid}", scene.get("intro_candidate", False))),
                    "outro_candidate": bool(st.session_state.get(f"outro_{selected}_{sid}", scene.get("outro_candidate", False))),
                }
                is_changed = (
                    new["tags"] != current_tags
                    or new["excluded"] != bool(scene.get("excluded", False))
                    or new["intro_candidate"] != bool(scene.get("intro_candidate", False))
                    or new["outro_candidate"] != bool(scene.get("outro_candidate", False))
                )
                if is_changed:
                    changed += 1
                    scene["corrected_tags"] = new["tags"]
                    scene["excluded"] = new["excluded"]
                    scene["intro_candidate"] = new["intro_candidate"]
                    scene["outro_candidate"] = new["outro_candidate"]
                if is_changed or mark_all:
                    if mark_all and "corrected_tags" not in scene:
                        scene["corrected_tags"] = current_tags  # confirmed as correct
                    scene["reviewed"] = True

            # ⭐ thumbnail ticks: a newly ticked scene becomes the master (the lowest
            # one on the page if several); unticking the current master clears it.
            _ticked = [sc["scene_id"] for sc, _, _ in shown
                       if st.session_state.get(f"thumb_{selected}_{sc['scene_id']}")]
            _new_master = master_sid
            _newly = [sid for sid in _ticked if sid != master_sid]
            if _newly:
                _new_master = _newly[-1]
            elif master_sid is not None and any(sc["scene_id"] == master_sid for sc, _, _ in shown) \
                    and master_sid not in _ticked:
                _new_master = None
            thumb_msg = ""
            if _new_master != master_sid:
                set_master_thumbnail(video_id, _new_master)
                thumb_msg = (" Video thumbnail set to automatic." if _new_master is None
                             else f" Video thumbnail set to scene {_new_master}.")

            if changed or mark_all:
                cat_path.write_text(json.dumps(data, indent=2))
                mark_pending(cat_path.stem)
                st.cache_data.clear()  # so the Compilation Planner sees the change straight away
                st.session_state["rev_saved_msg"] = (
                    f"Saved {changed} changed scene(s)"
                    + (f"; all {len(shown)} shown scenes marked reviewed." if mark_all else ".")
                )
            elif not thumb_msg:
                st.session_state["rev_saved_msg"] = "Nothing had changed — nothing to save."
            if thumb_msg:
                st.session_state["rev_saved_msg"] = (st.session_state.get("rev_saved_msg", "") + thumb_msg).strip()
            # Fresh widgets next run, so they show what was just saved.
            for scene, _, _ in shown:
                for prefix in ("tags", "excluded", "intro", "outro", "thumb"):
                    st.session_state.pop(f"{prefix}_{selected}_{scene['scene_id']}", None)
            st.rerun()
