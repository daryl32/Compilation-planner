"""
Reviewer — browse and correct auto-generated scene tags.
"""

import json
from pathlib import Path

import streamlit as st

from config import CATALOGUE_DIR
from labels import CANDIDATE_LABELS

# Google Drive OAuth — for writing labels to Drive
try:
    from drive_oauth import (
        get_auth_url, exchange_code_for_token,
        push_file_with_oauth, is_authenticated, SESSION_KEY as _OAUTH_SESSION_KEY,
    )
    _OAUTH_AVAILABLE = True
except ImportError:
    _OAUTH_AVAILABLE = False

OUTPUT_DIR = CATALOGUE_DIR

st.set_page_config(page_title="Reviewer", layout="wide")

# ---------------------------------------------------------------------------
# Drive OAuth callback
# ---------------------------------------------------------------------------
if _OAUTH_AVAILABLE and _OAUTH_SESSION_KEY not in st.session_state:
    _qp = st.query_params.to_dict()
    if "code" in _qp:
        _token = exchange_code_for_token(_qp["code"])
        if _token:
            st.session_state[_OAUTH_SESSION_KEY] = _token
        st.query_params.clear()
        st.rerun()

st.title("Reviewer")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Process a video")
st.sidebar.link_button("▶️ Open Google Colab", "https://colab.research.google.com", use_container_width=True)
st.sidebar.divider()
review_mode = st.sidebar.checkbox("Review / correction mode", value=False)
st.sidebar.divider()

# Drive connect
if _OAUTH_AVAILABLE:
    with st.sidebar.expander("☁️ Google Drive", expanded=False):
        if st.session_state.get(_OAUTH_SESSION_KEY):
            st.success("Connected")
            if st.button("Disconnect Drive", key="drive_disconnect"):
                st.session_state.pop(_OAUTH_SESSION_KEY, None)
                st.rerun()
        else:
            st.caption("Connect to save labels to your Google Drive.")
            _auth_url = get_auth_url()
            st.link_button("🔗 Connect Google Drive", _auth_url)

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
    "scene-labeling/catalogue/labels.json",
)
if error:
    st.sidebar.error(f"Drive upload failed: {error}")
else:
    st.sidebar.success(f'Added "{new_label}" — reload to see it.')
        st.sidebar.success(f'Added "{new_label}" — reload to see it.')

# ---------------------------------------------------------------------------
# Export training data
# ---------------------------------------------------------------------------
def export_training_data():
    examples = []
    for cat_file in sorted(OUTPUT_DIR.glob("*.json")):
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
catalogue_files = sorted(OUTPUT_DIR.glob("*.json"))

if not catalogue_files:
    st.info("No catalogues yet. Process a video from the sidebar to get started.")
else:
    names = [f.stem for f in catalogue_files]
    selected = st.selectbox("Catalogue", names)
    cat_path = OUTPUT_DIR / f"{selected}.json"
    data = json.loads(cat_path.read_text())

    reviewed_count = sum(1 for s in data["scenes"] if s.get("reviewed"))
    st.caption(f"Source: {data['source_path']}  |  {len(data['scenes'])} scenes  |  {reviewed_count} reviewed")

    tag_filter = st.text_input("Filter by tag (optional)")
    filter_cols = st.columns(2)
    with filter_cols[0]:
        show_only_excluded = st.checkbox("Show only excluded scenes", value=False)
    with filter_cols[1]:
        show_only_intro_outro = st.checkbox("Show only intro/outro candidates", value=False)

    for scene in data["scenes"]:
        current_tags = scene.get("tags", [])
        if tag_filter and tag_filter.lower() not in [t.lower() for t in current_tags]:
            continue
        if show_only_excluded and not scene.get("excluded"):
            continue
        if show_only_intro_outro and not (scene.get("intro_candidate") or scene.get("outro_candidate")):
            continue

        cols = st.columns([1, 2])
        with cols[0]:
            if scene["thumbnail_paths"]:
                thumb_path = OUTPUT_DIR / "thumbnails" / data["video_id"] / Path(scene["thumbnail_paths"][0]).name
                if thumb_path.exists():
                    st.image(str(thumb_path), width=280)
                else:
                    st.warning(f"Thumbnail not found: {thumb_path}")
            if scene.get("reviewed"):
                st.success("✓ Reviewed")
            if scene.get("excluded"):
                st.error("🚫 Excluded from planner")
            if scene.get("intro_candidate"):
                st.info("🎬 Intro candidate")
            if scene.get("outro_candidate"):
                st.info("🎬 Outro candidate")

        with cols[1]:
            st.markdown(f"**Scene {scene['scene_id']}**  ·  {scene['start_tc']} → {scene['end_tc']}")

            if not review_mode:
                st.markdown("Tags: " + ", ".join(f"`{t}`" for t in current_tags))
            else:
                tags_default = scene.get("corrected_tags") or current_tags
                tag_options = sorted(set(CANDIDATE_LABELS) | set(tags_default))

                new_tags = st.multiselect(
                    "Tags", options=tag_options, default=tags_default, key=f"tags_{selected}_{scene['scene_id']}",
                )
                excluded = st.checkbox(
                    "Exclude from planner (e.g. intro/outro clip)",
                    value=scene.get("excluded", False),
                    key=f"excluded_{selected}_{scene['scene_id']}",
                )

                role_cols = st.columns(2)
                with role_cols[0]:
                    intro_candidate = st.checkbox(
                        "Intro candidate",
                        value=scene.get("intro_candidate", False),
                        key=f"intro_{selected}_{scene['scene_id']}",
                        help="Available to the Compilation Planner's first block when its "
                             "'Prefer Intro/Outro-tagged clips' toggle is on. Independent of "
                             "Exclude — this scene can still show up normally elsewhere too, "
                             "unless you also exclude it above.",
                    )
                with role_cols[1]:
                    outro_candidate = st.checkbox(
                        "Outro candidate",
                        value=scene.get("outro_candidate", False),
                        key=f"outro_{selected}_{scene['scene_id']}",
                        help="Available to the Compilation Planner's last block when its "
                             "'Prefer Intro/Outro-tagged clips' toggle is on. Independent of "
                             "Exclude — this scene can still show up normally elsewhere too, "
                             "unless you also exclude it above.",
                    )

                if st.button("Save correction", key=f"save_{selected}_{scene['scene_id']}"):
                    scene["corrected_tags"] = new_tags
                    scene["reviewed"] = True
                    scene["excluded"] = excluded
                    scene["intro_candidate"] = intro_candidate
                    scene["outro_candidate"] = outro_candidate
                    cat_path.write_text(json.dumps(data, indent=2))
                    st.success("Saved.")
                    st.rerun()

        st.divider()
