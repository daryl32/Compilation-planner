"""
Reviewer — browse and correct auto-generated scene tags.
"""
 
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from pipeline import run_pipeline, OUTPUT_DIR, CANDIDATE_LABELS
 
st.set_page_config(page_title="Reviewer", layout="wide")
st.title("Reviewer")
 
 
def resolve_thumbnail(stored_path: str, video_id: str) -> Path:
    """Thumbnail paths in the JSON were saved relative to Colab's filesystem
    (e.g. 'catalogue/thumbnails/<video_id>/<file>.jpg'). Reconstruct the real
    local path using OUTPUT_DIR instead of trusting the stored path directly."""
    filename = Path(stored_path).name
    return OUTPUT_DIR / "thumbnails" / video_id / filename
 
# ---------------------------------------------------------------------------
# Sidebar: run the pipeline on a new video
# ---------------------------------------------------------------------------
 
st.sidebar.header("Process a video")
video_path = st.sidebar.text_input("Path to video file", placeholder="C:\\videos\\clip1.mp4")
 
if st.sidebar.button("Run pipeline") and video_path:
    if not Path(video_path).exists():
        st.sidebar.error("File not found.")
    else:
        with st.spinner("Processing... this can take a while for the first run (model downloads)."):
            run_pipeline(video_path)
        st.sidebar.success("Done.")
 
st.sidebar.divider()
review_mode = st.sidebar.checkbox("Review / correction mode", value=False)
 
# ---------------------------------------------------------------------------
# Export reviewed scenes as fine-tuning / few-shot training data
# ---------------------------------------------------------------------------
 
def export_training_data():
    examples = []
    for cat_file in sorted(OUTPUT_DIR.glob("*.json")):
        data = json.loads(cat_file.read_text())
        for scene in data["scenes"]:
            if scene.get("reviewed") and not scene.get("excluded") and scene["thumbnail_paths"]:
                examples.append({
                    "image_path": str(resolve_thumbnail(scene["thumbnail_paths"][0], data["video_id"])),
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
                thumb_path = resolve_thumbnail(scene["thumbnail_paths"][0], data["video_id"])
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
                # Read-only view
                st.markdown("Tags: " + ", ".join(f"`{t}`" for t in current_tags))
            else:
                # Correction view — tags only
                tags_default = scene.get("corrected_tags") or current_tags
                # Guard against tags that exist on the scene but aren't in the current
                # CANDIDATE_LABELS list (e.g. label list drift between Colab and local) —
                # multiselect requires every default value to also be a valid option.
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
