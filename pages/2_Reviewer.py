"""
Reviewer — browse and correct auto-generated scene tags.
"""

import json
from pathlib import Path

import streamlit as st

from config import CATALOGUE_DIR
from library_common import (scene_tags, mark_pending,
                            library_ranges, tc_to_seconds, format_mmss, overlap_with_range)


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

    tag_filter = st.text_input("Filter by tag (optional)")
    filter_cols = st.columns(2)
    with filter_cols[0]:
        show_only_excluded = st.checkbox("Show only excluded scenes", value=False)
    with filter_cols[1]:
        show_only_intro_outro = st.checkbox("Show only intro/outro candidates", value=False)

    # Which scenes to show (filters and the library range).
    shown = []
    for scene in data["scenes"]:
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
        shown.append((scene, current_tags, trimmed_to))

    def scene_header(scene, trimmed_to):
        """Thumbnail and status badges (left column)."""
        if scene["thumbnail_paths"]:
            thumb_path = OUTPUT_DIR / "thumbnails" / data["video_id"] / Path(scene["thumbnail_paths"][0]).name
            if thumb_path.exists():
                st.image(str(thumb_path), width=280)
            else:
                st.warning(f"Thumbnail not found: {thumb_path}")
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

    BACK_TO_TOP = (
        '<a href="#reviewer-top" target="_self" style="display:inline-block;padding:0.4rem 0.9rem;'
        'border:1px solid rgba(128,128,128,0.4);border-radius:0.5rem;text-decoration:none;">'
        '⬆️ Back to top</a>'
    )

    if not shown:
        st.info("No scenes match these filters.")
    elif not review_mode:
        for scene, current_tags, trimmed_to in shown:
            cols = st.columns([1, 2])
            with cols[0]:
                scene_header(scene, trimmed_to)
            with cols[1]:
                st.markdown(f"**Scene {scene['scene_id']}**  ·  {scene['start_tc']} → {scene['end_tc']}")
                st.markdown("Tags: " + ", ".join(f"`{t}`" for t in current_tags))
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
                cols = st.columns([1, 2])
                with cols[0]:
                    scene_header(scene, trimmed_to)
                with cols[1]:
                    st.markdown(f"**Scene {sid}**  ·  {scene['start_tc']} → {scene['end_tc']}")
                    st.multiselect(
                        "Tags", options=sorted(set(CANDIDATE_LABELS) | set(current_tags)),
                        default=current_tags, key=f"tags_{selected}_{sid}",
                    )
                    st.checkbox("Exclude from planner (e.g. intro/outro clip)",
                                value=scene.get("excluded", False), key=f"excluded_{selected}_{sid}")
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

            if changed or mark_all:
                cat_path.write_text(json.dumps(data, indent=2))
                mark_pending(cat_path.stem)
                st.cache_data.clear()  # so the Compilation Planner sees the change straight away
                st.session_state["rev_saved_msg"] = (
                    f"Saved {changed} changed scene(s)"
                    + (f"; all {len(shown)} shown scenes marked reviewed." if mark_all else ".")
                )
            else:
                st.session_state["rev_saved_msg"] = "Nothing had changed — nothing to save."
            # Fresh widgets next run, so they show what was just saved.
            for scene, _, _ in shown:
                for prefix in ("tags", "excluded", "intro", "outro"):
                    st.session_state.pop(f"{prefix}_{selected}_{scene['scene_id']}", None)
            st.rerun()
