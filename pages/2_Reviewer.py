"""
Reviewer — browse and correct auto-generated scene tags, and split / combine scenes.

One view, always editable (there's no separate review mode any more):
  • Find a video: search by name and narrow the list to videos with no custom
    thumbnail, with split or combine suggestions, with no outro clip, or not
    fully reviewed.
  • Each scene is a card: tags, Exclude / Intro / Outro, ⭐ video thumbnail,
    ✂️ Split…, ⚡ Auto split, 🔗 Combine with next, ↩️ Undo.
  • Tag and checkbox edits stay on the page until 💾 Save (each card updates on
    its own, so ticking a box doesn't reload the whole page). Split, combine,
    undo and ⭐ save straight away — and save any unsaved edits with them.
  • ⚡ Auto split cuts long scenes into parts close to a target length (sidebar),
    nudging each cut to a sharp picture change nearby.
  • New parts get their own motion score, measured from the preview copy.
"""

import json
import math
import tempfile
from pathlib import Path

import streamlit as st

from config import CATALOGUE_DIR
from library_common import (safe_image, image_ok, mark_pending, master_thumbnail_scenes, set_master_thumbnail,
                            proxy_path, library_ranges, tc_to_seconds, format_mmss, overlap_with_range,
                            scene_tags)
import catalogue_store as CS
import scene_edits as SE

# Sign-in, whitelist, Drive connect, title and the shared sidebar — see page_setup.py
from page_setup import page_setup, push_file_with_oauth, drive_token

OUTPUT_DIR = CATALOGUE_DIR


def load_labels() -> list:
    labels_path = CATALOGUE_DIR / "labels.json"
    if labels_path.exists():
        return json.loads(labels_path.read_text())
    return []


CANDIDATE_LABELS = load_labels()

page_setup("Reviewer", "pages/2_Reviewer.py", anchor="reviewer-top")

# ---------------------------------------------------------------------------
# Sidebar (this page's controls — the shared ones are drawn by page_setup)
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Reviewer")
use_library_range = st.sidebar.toggle(
    "Use library range", value=True,
    help="Only show scenes inside the video's Media Library time range. "
         "Turn off to see every scene in the original video.",
)

with st.sidebar.expander("💡 Split / combine suggestions", expanded=False):
    show_suggestions = st.toggle("Show suggestions", value=True)
    suggest_long_sec = st.number_input(
        "Suggest splitting clips longer than (seconds)", min_value=5.0, value=SE.DEFAULT_LONG_SEC, step=10.0,
        disabled=not show_suggestions,
    )
    suggest_short_sec = st.number_input(
        "Suggest combining runs of clips shorter than (seconds)", min_value=0.1, value=SE.DEFAULT_SHORT_SEC,
        step=0.5, disabled=not show_suggestions,
        help="Flags two or more touching clips in a row that are each shorter than this.",
    )

with st.sidebar.expander("⚡ Auto split", expanded=False):
    auto_target_sec = st.number_input(
        "Target clip length (seconds)", min_value=2.0, value=30.0, step=5.0, key="rev_auto_target",
        help="⚡ Auto split cuts a scene into parts close to this length, adapted to the scene: "
             "a 95 s scene with a 30 s target becomes 3 parts of about 32 s. Each cut is moved to a "
             "sharp picture change nearby when the scene has a motion curve.",
    )
    st.caption(f"Offered on scenes longer than {1.5 * auto_target_sec:g} s.")

with st.sidebar.expander("🏷️ Labels", expanded=False):
    new_label = st.text_input("Add new label", placeholder="e.g. slow-motion")
    if st.button("Add label") and new_label:
        new_label = new_label.strip().lower()
        if new_label in CANDIDATE_LABELS:
            st.warning("Label already exists.")
        elif not drive_token():
            st.warning("Connect Google Drive first to save labels.")
        else:
            tmp = Path(tempfile.gettempdir()) / "labels.json"
            tmp.write_text(json.dumps(CANDIDATE_LABELS + [new_label]))
            error = push_file_with_oauth(drive_token(), tmp, "scene-labeling/catalogue")
            if error:
                st.error(f"Drive upload failed: {error}")
            else:
                (CATALOGUE_DIR / "labels.json").write_text(tmp.read_text())
                CANDIDATE_LABELS.append(new_label)
                st.success(f'Added "{new_label}" — label is now available.')


def export_training_data():
    examples = []
    for stem in CS.video_index():
        data = CS.load_catalogue(stem)
        for scene in data["scenes"]:
            if scene.get("reviewed") and not scene.get("excluded") and scene.get("thumbnail_paths"):
                examples.append({
                    "image_path": str(OUTPUT_DIR / "thumbnails" / data["video_id"]
                                      / Path(scene["thumbnail_paths"][0]).name),
                    "tags": scene.get("corrected_tags") or scene.get("tags", []),
                })
    out_path = OUTPUT_DIR / "training_data.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples), out_path


with st.sidebar.expander("🧪 Processing & training data", expanded=False):
    st.link_button("▶️ Open Google Colab", "https://colab.research.google.com", use_container_width=True)
    if st.button("Export corrections as training data", use_container_width=True):
        count, out_path = export_training_data()
        if count == 0:
            st.warning("No reviewed scenes yet — correct some scenes first.")
        else:
            st.success(f"Exported {count} reviewed examples to {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_dur(sec: float) -> str:
    return f"{sec:.1f}s" if sec < 60 else format_mmss(sec)


def fmt_precise(sec: float) -> str:
    m, s = divmod(max(0.0, sec), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def in_range_scenes(scenes: list, rng) -> list:
    if not rng:
        return list(scenes)
    return [s for s in scenes
            if overlap_with_range(tc_to_seconds(s["start_tc"]), tc_to_seconds(s["end_tc"]), rng)[0] is not None]


# ---------------------------------------------------------------------------
# Find a video
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
index = CS.video_index()

if not index:
    st.info("No catalogues yet. Process a video in Colab (sidebar) to get started.")
    st.stop()

ranges = library_ranges()
masters = master_thumbnail_scenes()

FIND_NO_THUMB = "⭐ No custom thumbnail"
FIND_SPLIT = "✂️ Has split suggestions"
FIND_MERGE = "🔗 Has combine suggestions"
FIND_NO_OUTRO = "🎬 No outro clip"
FIND_NO_INTRO = "🎬 No intro clip"
FIND_UNREVIEWED = "📝 Not fully reviewed"


def video_facts(v: dict) -> dict:
    rng = ranges.get(v["video_id"]) if use_library_range else None
    scenes = in_range_scenes(v["scenes"], rng)
    sugg = (SE.suggest_edits(scenes, suggest_long_sec, suggest_short_sec, v["dismissed_edit_suggestions"])
            if show_suggestions else {"split": [], "merge": []})
    return {
        "custom_thumb": v["video_id"] in masters,
        "n_split": len(sugg["split"]),
        "n_merge": len(sugg["merge"]),
        "has_outro": any(s["outro"] for s in v["scenes"]),
        "has_intro": any(s["intro"] for s in v["scenes"]),
        "reviewed": sum(s["reviewed"] for s in v["scenes"]),
        "n": len(v["scenes"]),
    }


facts = {stem: video_facts(v) for stem, v in index.items()}

with st.container(border=True):
    fc = st.columns([2, 3])
    name_query = fc[0].text_input("Find a video", placeholder="part of a video name", key="rev_find_name")
    find_flags = fc[1].multiselect(
        "Only videos that…",
        [FIND_NO_THUMB, FIND_SPLIT, FIND_MERGE, FIND_NO_OUTRO, FIND_NO_INTRO, FIND_UNREVIEWED],
        key="rev_find_flags", placeholder="any video",
        help="Pick several to narrow it down further (a video must match all of them). "
             "Split/combine suggestions use the sidebar settings and the library range.",
    )

    def matches(stem: str) -> bool:
        f = facts[stem]
        if name_query and name_query.lower() not in stem.lower():
            return False
        checks = {
            FIND_NO_THUMB: not f["custom_thumb"],
            FIND_SPLIT: f["n_split"] > 0,
            FIND_MERGE: f["n_merge"] > 0,
            FIND_NO_OUTRO: not f["has_outro"],
            FIND_NO_INTRO: not f["has_intro"],
            FIND_UNREVIEWED: f["reviewed"] < f["n"],
        }
        return all(checks[flag] for flag in find_flags)

    names = [stem for stem in index if matches(stem)]

    def video_label(stem: str) -> str:
        f = facts[stem]
        bits = [stem]
        if f["n_split"]:
            bits.append(f"✂️{f['n_split']}")
        if f["n_merge"]:
            bits.append(f"🔗{f['n_merge']}")
        if not f["custom_thumb"]:
            bits.append("no ⭐")
        if not f["has_outro"]:
            bits.append("no outro")
        bits.append(f"{f['reviewed']}/{f['n']} reviewed")
        return "  ·  ".join(bits)

    if not names:
        st.info(f"No videos match — {len(index)} video(s) in the library.")
        st.stop()
    if len(names) < len(index):
        st.caption(f"{len(names)} of {len(index)} videos match.")
    # The chosen video is kept outside the widget too (and in the address bar,
    # ?video=…), so a full reload — a card button, a reconnect, an app restart —
    # comes back to the same video instead of the first one in the list.
    if st.session_state.get("rev_video") not in names:
        for _cand in (st.session_state.get("_rev_video_keep"), st.query_params.get("video")):
            if _cand in names:
                st.session_state["rev_video"] = _cand
                break
    selected = st.selectbox("Video", names, format_func=video_label, key="rev_video")
    st.session_state["_rev_video_keep"] = selected
    if st.query_params.get("video") != selected:
        st.query_params["video"] = selected

_saved_msg = st.session_state.pop("rev_saved_msg", None)
if _saved_msg:
    st.success(_saved_msg)

cat_path = CS.catalogue_path(selected)
data = CS.load_catalogue(selected)
video_id = data["video_id"]
master_sid = masters.get(video_id)
scene_by_id = {s["scene_id"]: s for s in data["scenes"]}
ordered_all = SE.sorted_scenes(data["scenes"])
next_of = {a["scene_id"]: b["scene_id"] for a, b in zip(ordered_all, ordered_all[1:])}
thumbs_dir = OUTPUT_DIR / "thumbnails" / video_id
sprite_path = OUTPUT_DIR / "timeline_sprites" / f"{video_id}.jpg"

reviewed_count = sum(1 for s in data["scenes"] if s.get("reviewed"))
st.caption(f"Source: {data.get('source_path', '')}  |  {len(data['scenes'])} scenes  |  {reviewed_count} reviewed")


def video_paths() -> list:
    """Preview copy first (fast to seek), then the source video — whichever exist here."""
    paths = [proxy_path(video_id)]
    try:
        from render_preview import colab_to_local
        paths.append(Path(colab_to_local(data.get("source_path", ""))))
    except Exception:
        pass
    return [p for p in paths if p and Path(p).exists()]


# ⭐ Master thumbnail — the picture used for this video in the Media Library
# and the planner's video list. Chosen from the scenes below; saved at once.
with st.container(border=True):
    mt_cols = st.columns([1, 3])
    _mp = (CS.CATALOGUE_DIR / "thumbnails" / video_id / Path(scene_by_id[master_sid]["thumbnail_paths"][0]).name
           if master_sid in scene_by_id and scene_by_id[master_sid].get("thumbnail_paths") else None)
    with mt_cols[0]:
        if _mp and _mp.exists():
            safe_image(str(_mp), width=160)
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

lib_range = ranges.get(video_id) if use_library_range else None
if lib_range:
    _outside = len(data["scenes"]) - len(in_range_scenes(data["scenes"], lib_range))
    st.info(f"📚 Showing the library range {format_mmss(lib_range[0])}–{format_mmss(lib_range[1])}"
            + (f" — {_outside} scene(s) outside it are hidden." if _outside else "."))
elif use_library_range:
    st.caption("No library range set for this video — showing every scene.")


# ---------------------------------------------------------------------------
# Edits: unsaved field edits live in widget state until saved
# ---------------------------------------------------------------------------
FIELD_PREFIXES = ("tags", "excluded", "intro", "outro")


def wkey(prefix: str, sid) -> str:
    return f"{prefix}_{selected}_{sid}"


def pending_fields(scene: dict):
    """This scene's edited values from its widgets, or None if it has no widgets yet."""
    sid = scene["scene_id"]
    if wkey("tags", sid) not in st.session_state:
        return None
    return {
        "tags": list(st.session_state.get(wkey("tags", sid), scene_tags(scene))),
        "excluded": bool(st.session_state.get(wkey("excluded", sid), scene.get("excluded", False))),
        "intro_candidate": bool(st.session_state.get(wkey("intro", sid), scene.get("intro_candidate", False))),
        "outro_candidate": bool(st.session_state.get(wkey("outro", sid), scene.get("outro_candidate", False))),
    }


def is_changed(scene: dict, new: dict) -> bool:
    return (new["tags"] != scene_tags(scene)
            or new["excluded"] != bool(scene.get("excluded", False))
            or new["intro_candidate"] != bool(scene.get("intro_candidate", False))
            or new["outro_candidate"] != bool(scene.get("outro_candidate", False)))


def apply_pending(mark_all_ids=()) -> int:
    """Copy unsaved edits into `data` (changed scenes are marked reviewed; scenes
    in mark_all_ids are marked reviewed even if unchanged). Returns how many changed."""
    changed = 0
    mark_all_ids = set(mark_all_ids)
    for scene in data["scenes"]:
        new = pending_fields(scene)
        ch = new is not None and is_changed(scene, new)
        if ch:
            changed += 1
            scene["corrected_tags"] = new["tags"]
            scene["excluded"] = new["excluded"]
            scene["intro_candidate"] = new["intro_candidate"]
            scene["outro_candidate"] = new["outro_candidate"]
        if ch or scene["scene_id"] in mark_all_ids:
            if "corrected_tags" not in scene:
                scene["corrected_tags"] = scene_tags(scene)  # confirmed as correct
            scene["reviewed"] = True
    return changed


def forget_widgets() -> None:
    """Fresh widgets next run, so they show what was just saved."""
    for k in [k for k in st.session_state.keys() if isinstance(k, str)
              and any(k.startswith(f"{p}_{selected}_") for p in FIELD_PREFIXES)]:
        st.session_state.pop(k, None)


def write_catalogue(message: str, new_thumbs=()) -> None:
    present = {s["scene_id"] for s in data["scenes"]}
    if master_sid is not None and master_sid not in present:
        # The ⭐ scene was combined into another — keep using that one.
        holder = next((s for s in data["scenes"]
                       if any(o.get("scene_id") == master_sid
                              for o in (s.get("edit") or {}).get("originals", []))), None)
        set_master_thumbnail(video_id, holder["scene_id"] if holder else None)
    CS.save_catalogue(selected, data)
    for t in new_thumbs:
        mark_pending(f"thumb:{video_id}/{Path(t).name}")
    st.cache_data.clear()  # so the Compilation Planner sees the change straight away
    forget_widgets()
    st.session_state["rev_saved_msg"] = message
    st.rerun()


def structural_edit(fn, message_fn) -> None:
    """Split / combine / undo: save any unsaved field edits with it, so nothing
    typed on the page is lost when scene ids change. fn() returns
    (result, new thumbnail files); message_fn(result) is the saved message."""
    unsaved = apply_pending()
    try:
        result, new_thumbs = fn()
    except SE.EditError as e:
        st.error(str(e))
        return
    msg = message_fn(result)
    if unsaved:
        msg += f" Your unsaved changes to {unsaved} other scene(s) were saved too."
    write_catalogue(msg, new_thumbs)


def finish_new_parts(ids, progress=None) -> list:
    """Thumbnails and motion scores for new scene parts. Returns new thumbnail files."""
    vids = video_paths()
    new_thumbs = []
    for s in data["scenes"]:
        if s["scene_id"] in ids and not s.get("thumbnail_paths"):
            p = SE.make_part_thumbnail(data, s, thumbs_dir, vids, sprite_path)
            if p:
                new_thumbs.append(p)
    SE.fill_motion(data, ids, vids)
    return new_thumbs


def do_merge(ids: list) -> None:
    def run():
        new_id = SE.merge_scenes(data, ids)
        # A combine keeps a time-weighted motion score when every part had one;
        # only measure it when it couldn't.
        if not any(s["scene_id"] == new_id and "motion_intensity" in s for s in data["scenes"]):
            SE.fill_motion(data, [new_id], video_paths())
        return new_id, ()
    structural_edit(run, lambda new_id: f"Combined {len(ids)} scenes into scene {new_id}.")


def do_undo(sid) -> None:
    structural_edit(lambda: (SE.undo_edit(data, sid), ()), lambda msg: msg)


def dismiss(key: str) -> None:
    lst = data.setdefault("dismissed_edit_suggestions", [])
    if key not in lst:
        lst.append(key)
    apply_pending()
    write_catalogue("Suggestion dismissed.")


def set_star(sid) -> None:
    set_master_thumbnail(video_id, sid)
    st.session_state["rev_saved_msg"] = f"Video thumbnail set to scene {sid}."
    st.rerun()


def remake_thumbnail(scene: dict, dest: Path) -> None:
    if SE.write_thumbnail(data, scene, dest, video_paths(), sprite_path):
        mark_pending(f"thumb:{video_id}/{dest.name}")
        st.session_state["rev_saved_msg"] = f"Thumbnail remade for scene {scene['scene_id']}."
    else:
        st.session_state["rev_saved_msg"] = None
        st.error("Couldn't make a picture — no preview copy or timeline thumbnails for this video here.")
        return
    st.rerun()


def back_to(sid) -> None:
    """After the full page reload this button causes, scroll back to this scene."""
    st.session_state["rev_scroll_to"] = sid


def frame_at(t: float):
    """A picture of the video at t: an exact frame if a video is here, else the
    nearest timeline-sprite tile. None if neither is available."""
    vids = video_paths()
    if vids:
        dest = Path(tempfile.gettempdir()) / "reviewer_frames" / f"{video_id}_{int(round(t * 10))}.jpg"
        if dest.exists() or SE.grab_frame(vids[0], t, dest, width=480):
            return str(dest)
    return SE.sprite_tile(data, sprite_path, t)


# ---------------------------------------------------------------------------
# ✂️ Split dialog (manual cuts, or ⚡ auto cuts to review first)
# ---------------------------------------------------------------------------

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


def do_split(sid, cuts) -> None:
    def run():
        ids = SE.split_scene(data, sid, cuts)
        with st.spinner("Making thumbnails and motion scores for the new parts…"):
            return ids, finish_new_parts(ids)
    structural_edit(run, lambda ids: f"Scene {sid} split into {len(ids)} parts (scenes {', '.join(map(str, ids))}).")


@st.dialog("✂️ Split scene", width="large")
def split_dialog(sid, auto: bool = False):
    scene = scene_by_id[sid]
    start, end = SE.scene_start_sec(scene), SE.scene_end_sec(scene)
    cuts_key, pos_key = f"split_cuts_{video_id}_{sid}", f"split_pos_{video_id}_{sid}"
    if auto and not st.session_state.get(cuts_key):
        st.session_state[cuts_key] = SE.auto_split_points(scene, auto_target_sec)
    cuts = st.session_state.setdefault(cuts_key, [])
    st.markdown(f"**Scene {SE.scene_label(scene)}** · {fmt_precise(start)} → {fmt_precise(end)} "
                f"({fmt_dur(end - start)})")

    # Slider positions in tenths of a second (coarser for very long scenes, so
    # the slider never has more than ~20,000 stops).
    lo10 = math.ceil((start + SE.MIN_PART_SEC) * 10)
    hi10 = math.floor((end - SE.MIN_PART_SEC) * 10)
    lo, hi = lo10 / 10, hi10 / 10
    if hi <= lo:
        st.warning("This scene is too short to split.")
        return

    vids = video_paths()
    if vids:
        st.video(str(vids[0]), start_time=int(start), end_time=int(math.ceil(end)))
        st.caption("Play the scene to find the moment, then move the slider there.")

    step10 = max(1, math.ceil((hi10 - lo10) / 20000))
    positions = [x / 10 for x in range(lo10, hi10 + 1, step10)]
    if positions[-1] != hi:
        positions.append(hi)

    def snap(t: float) -> float:
        i = round((t * 10 - lo10) / step10)
        return positions[max(0, min(len(positions) - 1, i))]

    if st.session_state.get(pos_key) not in positions:
        st.session_state[pos_key] = snap(st.session_state.get(pos_key, (start + end) / 2))

    auto_cuts = SE.auto_split_points(scene, auto_target_sec)
    ac = st.columns([3, 2])
    ac[0].caption(f"⚡ Auto: {len(auto_cuts) + 1} parts of about {fmt_dur((end - start) / (len(auto_cuts) + 1))} "
                  f"(target {auto_target_sec:g}s, set in the sidebar)." if auto_cuts
                  else f"⚡ Auto: already close to the {auto_target_sec:g}s target — no cuts needed.")
    ac[1].button("⚡ Use auto cuts", key=f"{pos_key}_auto", on_click=_set_state, args=(cuts_key, auto_cuts),
                 disabled=not auto_cuts, use_container_width=True)

    suggested = [t for t in SE.cut_suggestions_from_sprite(data, sprite_path, scene) if lo <= t <= hi]
    if suggested:
        st.caption("💡 Big picture changes found in the timeline thumbnails:")
        bcols = st.columns(len(suggested))
        for i, t in enumerate(suggested):
            bcols[i].button(f"Go to {fmt_precise(t)}", key=f"{pos_key}_go{i}",
                            on_click=_set_state, args=(pos_key, snap(t)))

    pos = st.select_slider("Cut at (time in the video)", options=positions,
                           format_func=fmt_precise, key=pos_key)
    st.caption(f"{fmt_precise(pos)} in the video — {pos - start:.1f}s into the scene")
    img = frame_at(pos)
    if img is not None:
        safe_image(img, width=360, caption="Frame at the cut (first frame of the next part)")
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
            pc = st.columns([4, 1, 1])
            pc[0].markdown(f"{i + 1}. {fmt_precise(a)} → {fmt_precise(b)} ({fmt_dur(b - a)})")
            if i < len(cuts):
                pc[1].button("👁", key=f"{pos_key}_see{i}", on_click=_set_state, args=(pos_key, snap(cuts[i])),
                             help=f"Show the frame at the cut {fmt_precise(cuts[i])}")
                pc[2].button("✖ cut", key=f"{pos_key}_rm{i}", on_click=_remove_cut, args=(cuts_key, cuts[i]),
                             help=f"Remove the cut at {fmt_precise(cuts[i])}")
        st.caption("Each part starts unreviewed with this scene's tags and gets its own motion score. "
                   "Undo is available afterwards. Compilation Planner projects that already picked clips "
                   "from this scene may need those picks redone.")
    if st.button(f"✂️ Split into {len(cuts) + 1} parts" if cuts else "✂️ Split",
                 type="primary", disabled=not cuts, use_container_width=True, key=f"{pos_key}_go"):
        st.session_state.pop(pos_key, None)
        st.session_state.pop(cuts_key, None)
        back_to(sid)
        do_split(sid, cuts)


@st.dialog("⚡ Auto split long scenes", width="large")
def bulk_auto_split_dialog(candidate_ids: list):
    plans = []
    for sid in candidate_ids:
        s = scene_by_id[sid]
        cuts = SE.auto_split_points(s, auto_target_sec)
        if cuts:
            plans.append((sid, cuts, SE.scene_duration(s)))
    if not plans:
        st.info("Nothing to split at this target length.")
        return
    total_parts = sum(len(c) + 1 for _, c, _ in plans)
    st.markdown(f"Split **{len(plans)} scene(s)** into **{total_parts} parts** of about "
                f"**{auto_target_sec:g}s** each:")
    for sid, cuts, dur in plans:
        st.markdown(f"- Scene {sid} · {fmt_dur(dur)} → {len(cuts) + 1} parts of ~{fmt_dur(dur / (len(cuts) + 1))}")
    st.caption("Each split can be undone on its own afterwards. New parts get thumbnails and motion "
               "scores — this takes a few seconds per scene.")
    if st.button(f"⚡ Split {len(plans)} scene(s)", type="primary", use_container_width=True):
        unsaved = apply_pending()
        bar = st.progress(0.0, text="Splitting…")
        new_thumbs, done = [], 0
        for i, (sid, cuts, _) in enumerate(plans):
            bar.progress(i / len(plans), text=f"Splitting scene {sid}…")
            try:
                ids = SE.split_scene(data, sid, cuts)
            except SE.EditError:
                continue
            new_thumbs += finish_new_parts(ids)
            done += 1
        bar.progress(1.0, text="Saving…")
        msg = f"⚡ Auto split {done} scene(s) into parts of about {auto_target_sec:g}s."
        if unsaved:
            msg += f" Your unsaved changes to {unsaved} scene(s) were saved too."
        write_catalogue(msg, new_thumbs)


# ---------------------------------------------------------------------------
# Suggestions and scene filters
# ---------------------------------------------------------------------------
in_range = in_range_scenes(data["scenes"], lib_range)
if show_suggestions:
    sugg = SE.suggest_edits(in_range, suggest_long_sec, suggest_short_sec,
                            data.get("dismissed_edit_suggestions", []))
else:
    sugg = {"split": [], "merge": []}
split_sugg = set(sugg["split"])
merge_group_of = {sid: grp for grp in sugg["merge"] for sid in grp}
auto_ids = [s["scene_id"] for s in SE.auto_split_candidates(in_range, auto_target_sec)]

if sugg["split"] or sugg["merge"] or auto_ids:
    with st.container(border=True):
        hc = st.columns([3, 1])
        parts_txt = []
        if sugg["split"]:
            parts_txt.append(f"{len(sugg['split'])} long clip(s) could be split")
        if sugg["merge"]:
            parts_txt.append(f"{len(sugg['merge'])} run(s) of short clips could be combined")
        if auto_ids:
            parts_txt.append(f"{len(auto_ids)} scene(s) are over {1.5 * auto_target_sec:g}s "
                             f"(⚡ auto split target {auto_target_sec:g}s)")
        hc[0].markdown("💡 " + " · ".join(parts_txt))
        if auto_ids and hc[1].button(f"⚡ Auto split {len(auto_ids)}…", use_container_width=True,
                                     key=f"bulk_auto_{selected}"):
            bulk_auto_split_dialog(auto_ids)

fc = st.columns([2, 1, 1, 1, 1])
tag_filter = fc[0].text_input("Filter by tag (optional)")
show_only_excluded = fc[1].checkbox("Only excluded", value=False)
show_only_intro_outro = fc[2].checkbox("Only intro/outro", value=False)
show_only_suggested = fc[3].checkbox("Only suggestions", value=False, disabled=not show_suggestions)
show_only_unreviewed = fc[4].checkbox("Only not reviewed", value=False)

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
    if show_only_unreviewed and scene.get("reviewed"):
        continue
    shown.append((scene, trimmed_to))

# Motion scores of this video, to say whether a scene is calm or busy for it.
_motions = sorted(s["motion_intensity"] for s in data["scenes"]
                  if isinstance(s.get("motion_intensity"), (int, float)))


def motion_text(scene: dict) -> str:
    m = scene.get("motion_intensity")
    if not isinstance(m, (int, float)):
        return "🏃 motion —"
    rank = sum(v <= m for v in _motions) / len(_motions) if _motions else 0.5
    level = "calm" if rank < 0.34 else "busy" if rank > 0.66 else "medium"
    return f"🏃 motion {m:.1f} ({level})"


# ---------------------------------------------------------------------------
# Scene cards
# ---------------------------------------------------------------------------
BACK_TO_TOP = (
    '<a href="#reviewer-top" target="_self" style="display:inline-block;padding:0.4rem 0.9rem;'
    'border:1px solid rgba(128,128,128,0.4);border-radius:0.5rem;text-decoration:none;">'
    '⬆️ Back to top</a>'
)


def save_row(where: str) -> bool:
    c = st.columns([3, 2, 2])
    c[0].caption(f"{len(shown)} scene(s) shown. Tag and checkbox edits are kept until you save — "
                 "changing the video or a filter discards them. Split, combine and ⭐ save straight away.")
    c[1].checkbox("Also mark unchanged scenes reviewed", key=f"rev_mark_all_{where}",
                  help="Off: only scenes you changed are marked ✓ Reviewed. "
                       "On: every scene shown is marked reviewed (you've checked them all).")
    label = "💾 Save all changes for this video" if where == "top" else "💾 Save all changes"
    return c[2].button(label, type="primary", use_container_width=True, key=f"rev_save_{where}")


def merge_banner(grp: list) -> None:
    total = sum(SE.scene_duration(scene_by_id[i]) for i in grp)
    with st.container(border=True):
        st.markdown(f"🔗 **Possible over-split:** scenes {grp[0]}–{grp[-1]} are {len(grp)} touching clips, "
                    f"{fmt_dur(total)} in total, each under {suggest_short_sec:g}s.")
        bc = st.columns([2, 1, 3])
        if bc[0].button(f"🔗 Combine these {len(grp)}", key=f"merge_run_{selected}_{grp[0]}", type="primary"):
            back_to(grp[0])
            do_merge(grp)
        if bc[1].button("Dismiss", key=f"merge_dismiss_{selected}_{grp[0]}"):
            dismiss(SE.suggestion_key("merge", grp))


@st.fragment
def scene_card(sid, trimmed_to) -> None:
    """One scene. Ticking a box or editing tags reruns just this card."""
    scene = scene_by_id[sid]
    current_tags = scene_tags(scene)
    st.markdown(f'<div id="rev-scene-{sid}"></div>', unsafe_allow_html=True)
    cols = st.columns([1, 2])
    with cols[0]:
        if scene.get("thumbnail_paths"):
            thumb_path = thumbs_dir / Path(scene["thumbnail_paths"][0]).name
            if image_ok(thumb_path):
                safe_image(str(thumb_path), width=280)
            else:
                st.warning(f"Thumbnail {'unreadable' if thumb_path.exists() else 'missing'}: {thumb_path.name}")
                if st.button("🔄 Remake thumbnail", key=f"remake_thumb_{selected}_{sid}",
                             help="Takes a new picture from the video (or the timeline thumbnails) "
                                  "and queues it for Google Drive."):
                    back_to(sid)
                    remake_thumbnail(scene, thumb_path)
        if sid == master_sid:
            st.success("⭐ Video thumbnail")
        elif scene.get("thumbnail_paths"):
            if st.button("⭐ Use as video thumbnail", key=f"star_{selected}_{sid}"):
                back_to(sid)
                set_star(sid)
        if trimmed_to:
            st.warning(f"✂️ Trimmed by library range — uses {format_mmss(trimmed_to[0])}–{format_mmss(trimmed_to[1])}")
        if scene.get("reviewed"):
            st.success("✓ Reviewed")
    with cols[1]:
        dur = SE.scene_duration(scene)
        st.markdown(f"**Scene {SE.scene_label(scene)}**  ·  {scene['start_tc']} → {scene['end_tc']}"
                    f"  ·  {fmt_dur(dur)}  ·  {motion_text(scene)}")
        if sid in split_sugg:
            wc = st.columns([4, 1])
            wc[0].warning(f"✂️ {fmt_dur(dur)} long — consider splitting.")
            if wc[1].button("Dismiss", key=f"split_dismiss_{selected}_{sid}"):
                back_to(sid)
                dismiss(SE.suggestion_key("split", [sid]))
        st.multiselect("Tags", options=sorted(set(CANDIDATE_LABELS) | set(current_tags)),
                       default=current_tags, key=wkey("tags", sid))
        rc = st.columns(3)
        rc[0].checkbox("🚫 Exclude from planner", value=scene.get("excluded", False), key=wkey("excluded", sid))
        rc[1].checkbox("🎬 Intro candidate", value=scene.get("intro_candidate", False), key=wkey("intro", sid),
                       help="Available to the Compilation Planner's first block when its "
                            "'Prefer Intro/Outro-tagged clips' toggle is on. Independent of Exclude.")
        rc[2].checkbox("🎬 Outro candidate", value=scene.get("outro_candidate", False), key=wkey("outro", sid),
                       help="Available to the Compilation Planner's last block when its "
                            "'Prefer Intro/Outro-tagged clips' toggle is on. Independent of Exclude.")
        new = pending_fields(scene)
        if new is not None and is_changed(scene, new):
            st.caption("✏️ Unsaved changes")

        bc = st.columns(4)
        if bc[0].button("✂️ Split…", key=f"split_{selected}_{sid}", use_container_width=True):
            back_to(sid)
            st.session_state["rev_open_dialog"] = (selected, sid, False)
            st.rerun()
        if sid in auto_ids and bc[1].button("⚡ Auto split", key=f"auto_{selected}_{sid}", use_container_width=True,
                                            help=f"Cut into parts of about {auto_target_sec:g}s — "
                                                 "you'll see the cuts before anything changes."):
            back_to(sid)
            st.session_state["rev_open_dialog"] = (selected, sid, True)
            st.rerun()
        nxt = next_of.get(sid)
        if nxt is not None and bc[2].button("🔗 Combine with next", key=f"merge_next_{selected}_{sid}",
                                            use_container_width=True,
                                            help=f"Join this scene and scene {nxt} into one."):
            back_to(sid)
            do_merge([sid, nxt])
        edit = scene.get("edit") or {}
        if edit.get("kind") in ("merge", "split") or "split_from" in scene:
            label = "↩️ Undo combine" if edit.get("kind") == "merge" else "↩️ Undo split"
            if bc[3].button(label, key=f"undo_{selected}_{sid}", use_container_width=True):
                back_to(sid)
                do_undo(sid)


if not shown:
    st.info("No scenes match these filters.")
else:
    save_top = save_row("top")
    st.divider()
    for scene, trimmed_to in shown:
        grp = merge_group_of.get(scene["scene_id"])
        if grp and grp[0] == scene["scene_id"]:
            merge_banner(grp)
        scene_card(scene["scene_id"], trimmed_to)
        st.divider()
    save_bottom = save_row("bottom")
    st.markdown(BACK_TO_TOP, unsafe_allow_html=True)

    if save_top or save_bottom:
        mark_all = st.session_state.get("rev_mark_all_top") or st.session_state.get("rev_mark_all_bottom")
        shown_ids = [s["scene_id"] for s, _ in shown] if mark_all else []
        changed = apply_pending(shown_ids)
        if changed or mark_all:
            write_catalogue(f"Saved {changed} changed scene(s)"
                            + (f"; all {len(shown)} shown scenes marked reviewed." if mark_all else "."))
        else:
            st.session_state["rev_saved_msg"] = "Nothing had changed — nothing to save."
            st.rerun()

# Open a dialog asked for by a scene card (cards are fragments, so they ask the
# page to open it on its next full run instead of opening it themselves).
_dlg = st.session_state.pop("rev_open_dialog", None)
if _dlg and _dlg[0] == selected and _dlg[1] in scene_by_id:
    split_dialog(_dlg[1], auto=_dlg[2])

# Scroll back to the scene whose button reloaded the page (see back_to).
_scroll = st.session_state.pop("rev_scroll_to", None)
if _scroll is not None:
    import streamlit.components.v1 as _components
    _components.html(
        "<script>(function () {"
        f"  var id = 'rev-scene-{_scroll}', tries = 0;"
        "  function go() {"
        "    var el = window.parent.document.getElementById(id);"
        "    if (el) { el.scrollIntoView({block: 'start'}); }"
        "    else if (tries++ < 20) { setTimeout(go, 150); }"
        "  }"
        "  setTimeout(go, 200);"
        "})();</script>",
        height=0,
    )
