"""
Reviewer — browse and correct auto-generated scene tags, and split / combine scenes.

One view, always editable (there's no separate review mode any more):
  • Find a video: search by name and narrow the list to videos with no custom
    thumbnail, with split or combine suggestions, with no outro clip, added
    recently, or not fully tagged / not fully role marked.
  • Each scene is a card: tags, Exclude, roles (🎬 Intro · 🌙 Slow · 💥 Action ·
    🏁 Outro and your own — see scene_roles.py), ⭐ video thumbnail,
    ✂️ Split…, ⚡ Auto split, 🔗 Combine with next, ↩️ Undo.
  • Two separate review states per scene: 🏷️ tagged ("reviewed" — tags / exclude
    checked) and 🎭 role marked (has a role, or "roles_reviewed" — checked, no role).
  • Tag and checkbox edits stay on the page until 💾 Save (each card updates on
    its own, so ticking a box doesn't reload the whole page). Split, combine,
    undo and ⭐ save straight away — and save any unsaved edits with them.
  • One target clip length (sidebar → ✂️ Split & combine): clips longer than it are
    suggested for splitting and offered ⚡ Auto split, which cuts them into the fewest
    parts no longer than it, nudging each cut to a sharp picture change nearby.
    🔗 Auto combine joins runs of very short clips.
  • New parts get their own motion score, measured from the preview copy.
"""

import json
import math
import tempfile
from pathlib import Path

import streamlit as st

from config import CATALOGUE_DIR
from library_common import (safe_image, image_ok, video_details, set_video_details, all_actors,
                            all_video_tags, suggest_video_tags, matches_details, mark_pending, master_thumbnail_scenes, set_master_thumbnail,
                            proxy_path, library_ranges, tc_to_seconds, format_mmss, overlap_with_range,
                            scene_tags)
import catalogue_store as CS
import scene_edits as SE
import scene_roles as SR

# Sign-in, whitelist, Drive connect, title and the shared sidebar — see page_setup.py
from page_setup import page_setup, push_file_with_oauth, drive_token, top_button, wrap_css

OUTPUT_DIR = CATALOGUE_DIR


def load_labels() -> list:
    labels_path = CATALOGUE_DIR / "labels.json"
    if labels_path.exists():
        return json.loads(labels_path.read_text())
    return []


CANDIDATE_LABELS = load_labels()

page_setup("Reviewer", "pages/2_Reviewer.py", anchor="reviewer-top")
# Rows of controls wrap onto a second line when the page is narrow (iPad portrait with the
# sidebar open) instead of squashing every label to "…".
top_button("reviewer-top",
           wrap_css("rev_wrap", 135)
           + "[class*='st-key-rev_wrap_card'] [data-testid='stColumn']{min-width:min(200px,100%)}"
           "[class*='st-key-rev_wrap_card'] [class*='st-key-rev_wrap_act'] [data-testid='stColumn']"
           "{min-width:min(105px,100%)}")

# ---------------------------------------------------------------------------
# Sidebar (this page's controls — the shared ones are drawn by page_setup)
# ---------------------------------------------------------------------------
st.sidebar.header("Reviewer")
use_library_range = st.sidebar.toggle(
    "Use library range", value=True,
    help="Only show scenes inside the video's Media Library time range. "
         "Turn off to see every scene in the original video.",
)

with st.sidebar.expander("✂️ Split & combine", expanded=False):
    show_suggestions = st.toggle("Suggest splits and combines", value=True)
    auto_target_sec = st.number_input(
        "Target clip length (seconds)", min_value=2.0, value=30.0, step=5.0, key="rev_auto_target",
        help="Clips longer than this are suggested for splitting and get ⚡ Auto split, which cuts them "
             "into the fewest parts that are each no longer than it — a 95 s scene with a 30 s target "
             "becomes 4 parts of about 24 s. Each cut is moved to a sharp picture change nearby when the "
             "scene has a motion curve (never making a part longer than the target).",
    )
    # One value: split suggestions and ⚡ Auto split both use the target length.
    suggest_long_sec = auto_target_sec
    suggest_short_sec = st.number_input(
        "Suggest combining runs of clips shorter than (seconds)", min_value=0.1, value=SE.DEFAULT_SHORT_SEC,
        step=0.5, disabled=not show_suggestions, key="rev_short_sec",
        help="Flags two or more touching clips in a row that are each shorter than this. 🔗 Auto combine "
             "(in the To do bar) joins each run into one clip — a long run into several, each no longer "
             "than the target length.",
    )
    st.caption(f"✂️ Split suggestions and ⚡ Auto split: clips over {auto_target_sec:g} s.  \n"
               f"🔗 Combine suggestions and Auto combine: runs of clips under {suggest_short_sec:g} s.")

_labels_box = st.sidebar.expander("🏷️ Labels & training data", expanded=False)
with _labels_box:
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


with _labels_box:
    st.divider()
    st.link_button("▶️ Open Google Colab", "https://colab.research.google.com", width="stretch")
    if st.button("Export corrections as training data", width="stretch"):
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

FIND_NO_ACTORS = "👤 No actors yet"
FIND_NO_VTAGS = "🏷️ No video tags yet"
FIND_NO_THUMB = "⭐ No custom thumbnail"
FIND_SPLIT = "✂️ Has split suggestions"
FIND_MERGE = "🔗 Has combine suggestions"
FIND_NO_OUTRO = "🎬 No outro clip"
FIND_NO_INTRO = "🎬 No intro clip"
FIND_UNTAGGED = "🏷️ Not fully tagged"
FIND_UNROLED = "🎭 Not fully role marked"
FIND_3DAYS = "🆕 Added in the last 3 days"
FIND_NEW = f"🆕 Added in the last {CS.NEW_VIDEO_DAYS} days"
FIND_TODAY = "📅 Added today"

_added = CS.added_times()   # {video_id: unix time added}


def role_marked(scene: dict) -> bool:
    """🎭 Role marked: the scene has a role, or was checked and left without one."""
    return bool(scene.get("roles_reviewed")) or bool(SR.scene_roles(scene))


def video_facts(v: dict) -> dict:
    rng = ranges.get(v["video_id"]) if use_library_range else None
    scenes = in_range_scenes(v["scenes"], rng)
    sugg = (SE.suggest_edits(scenes, suggest_long_sec, suggest_short_sec, v["dismissed_edit_suggestions"])
            if show_suggestions else {"split": [], "merge": []})
    usable = [s for s in scenes if not s["excluded"]]   # excluded scenes don't need a role
    return {
        "custom_thumb": v["video_id"] in masters,
        "n_split": len(sugg["split"]),
        "n_merge": len(sugg["merge"]),
        "has_outro": any(s["outro"] for s in v["scenes"]),
        "has_intro": any(s["intro"] for s in v["scenes"]),
        "roles": {r for s in v["scenes"] if not s["excluded"] for r in s.get("roles", [])},
        # Counted over the scenes the Reviewer shows (inside the library range when that's on).
        "tagged": sum(s["reviewed"] for s in scenes),
        "n": len(scenes),
        "role_marked": sum(role_marked(s) for s in usable),
        "n_usable": len(usable),
        "added": _added.get(v["video_id"], v["added"]),
    }


facts = {stem: video_facts(v) for stem, v in index.items()}

with st.container(border=True, key="rev_wrap_find"):
    fc = st.columns([3, 4, 2], vertical_alignment="bottom")
    name_query = fc[0].text_input("Find a video", placeholder="part of a video name", key="rev_find_name")
    find_flags = fc[1].multiselect(
        "Only videos that…",
        [FIND_TODAY, FIND_3DAYS, FIND_NEW, FIND_UNTAGGED, FIND_UNROLED, FIND_NO_THUMB, FIND_SPLIT, FIND_MERGE,
         FIND_NO_OUTRO, FIND_NO_INTRO, FIND_NO_ACTORS, FIND_NO_VTAGS],
        key="rev_find_flags", placeholder="any video",
        help="Pick several to narrow it down further (a video must match all of them). "
             "Split/combine suggestions use the sidebar settings and the library range.",
    )

    _details = video_details()
    _roles = SR.roles_by_id()
    # Actors / video tags / roles: less used, so in a popover (its label counts the ones in use).
    _n_more = sum(bool(st.session_state.get(k)) for k in
                  ("rev_find_actors", "rev_find_vtags", "rev_find_roles", "rev_find_no_roles"))
    with fc[2].popover(f"More filters ({_n_more})" if _n_more else "More filters", width="stretch"):
        find_actors = st.multiselect("With actors", all_actors(), key="rev_find_actors", placeholder="anyone")
        find_vtags = st.multiselect("With video tags", all_video_tags(), key="rev_find_vtags", placeholder="any")
        find_roles = st.multiselect("With scenes marked", list(_roles), key="rev_find_roles", placeholder="any",
                                    format_func=lambda r: SR.label(r, _roles),
                                    help="Videos with at least one (not excluded) scene in that role.")
        find_no_roles = st.multiselect("Without scenes marked", list(_roles), key="rev_find_no_roles",
                                       placeholder="—", format_func=lambda r: SR.label(r, _roles),
                                       help="Videos with no (not excluded) scene in that role yet — "
                                            "handy for finding videos still to mark.")
        find_all = st.toggle("Must have all of them", key="rev_find_all",
                             disabled=not (find_actors or find_vtags or find_roles),
                             help="Off: any of the picked actors / tags / roles. On: every one of them.")

    import datetime as _dt
    _today_start = _dt.datetime.combine(_dt.date.today(), _dt.time.min).timestamp()

    def matches(stem: str) -> bool:
        f = facts[stem]
        if name_query and name_query.lower() not in stem.lower():
            return False
        vid = index[stem]["video_id"]
        if not matches_details(vid, _details, find_actors, find_vtags, find_all):
            return False
        if find_roles and not (all if find_all else any)(r in f["roles"] for r in find_roles):
            return False
        if any(r in f["roles"] for r in find_no_roles):
            return False
        checks = {
            FIND_NO_THUMB: not f["custom_thumb"],
            FIND_SPLIT: f["n_split"] > 0,
            FIND_MERGE: f["n_merge"] > 0,
            FIND_NO_OUTRO: not f["has_outro"],
            FIND_NO_INTRO: not f["has_intro"],
            FIND_UNTAGGED: f["tagged"] < f["n"],
            FIND_UNROLED: f["role_marked"] < f["n_usable"],
            FIND_3DAYS: CS.is_new_video(f["added"], days=3),
            FIND_NEW: CS.is_new_video(f["added"]),
            FIND_TODAY: f["added"] >= _today_start,
            FIND_NO_ACTORS: not (_details.get(vid) or {}).get("actors"),
            FIND_NO_VTAGS: not (_details.get(vid) or {}).get("tags"),
        }
        return all(checks[flag] for flag in find_flags)

    names = [stem for stem in index if matches(stem)]
    # Sent here for one video (Media Library → 📝 Review, or a link): open it,
    # even if another video was showing or the search above would hide it.
    _qp_video = st.query_params.get("video")
    if _qp_video in index and _qp_video != st.session_state.get("_rev_qp_seen"):
        st.session_state["_rev_qp_seen"] = _qp_video
        st.session_state["rev_video"] = _qp_video
        st.session_state["_rev_video_keep"] = _qp_video
        st.session_state["_rev_find_sig"] = (name_query, tuple(find_flags))   # keep it listed (see below)
    if {FIND_NEW, FIND_3DAYS, FIND_TODAY} & set(find_flags):
        names.sort(key=lambda stem: -facts[stem]["added"])   # newest first
    # A split / combine / review can make the video you're working on stop
    # matching "Only videos that…" (e.g. its last long clip was just split).
    # Keep it in the list until you change the search yourself, rather than
    # dropping you onto the first video.
    _find_sig = (name_query, tuple(find_flags))
    _keep = st.session_state.get("_rev_video_keep")
    if (st.session_state.get("_rev_find_sig") == _find_sig and _keep in index and _keep not in names):
        names = names + [_keep]
    st.session_state["_rev_find_sig"] = _find_sig

    def video_label(stem: str) -> str:
        f = facts[stem]
        bits = [stem]
        if CS.is_new_video(f["added"]):
            bits.append(f"🆕 added {CS.added_ago(f['added'])}")
        if f["n_split"]:
            bits.append(f"✂️{f['n_split']}")
        if f["n_merge"]:
            bits.append(f"🔗{f['n_merge']}")
        if not f["custom_thumb"]:
            bits.append("no ⭐")
        if not f["has_outro"]:
            bits.append("no outro")
        bits.append(f"🏷️ {f['tagged']}/{f['n']} tagged")
        bits.append(f"🎭 {f['role_marked']}/{f['n_usable']} role marked")
        return "  ·  ".join(bits)

    if not names:
        st.info(f"No videos match — {len(index)} video(s) in the library.")
        st.stop()
    # The chosen video is kept outside the widget too (and in the address bar,
    # ?video=…), so a full reload — a card button, a reconnect, an app restart —
    # comes back to the same video instead of the first one in the list.
    if st.session_state.get("rev_video") not in names:
        for _cand in (st.session_state.get("_rev_video_keep"), st.query_params.get("video")):
            if _cand in names:
                st.session_state["rev_video"] = _cand
                break
    selected = st.selectbox("Video", names, format_func=video_label, key="rev_video")
    if len(names) < len(index):
        st.caption(f"{len(names)} of {len(index)} videos match.")
    st.session_state["_rev_video_keep"] = selected
    if st.query_params.get("video") != selected:
        st.query_params["video"] = selected
    st.session_state["_rev_qp_seen"] = selected

_saved_msg = st.session_state.pop("rev_saved_msg", None)
if _saved_msg:
    st.toast(_saved_msg, icon="✅")

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
_usable_all = [s for s in data["scenes"] if not s.get("excluded")]
role_marked_count = sum(1 for s in _usable_all if role_marked(s))


def video_paths() -> list:
    """Preview copy first (fast to seek), then the source video — whichever exist here."""
    paths = [proxy_path(video_id)]
    try:
        from render_preview import colab_to_local
        paths.append(Path(colab_to_local(data.get("source_path", ""))))
    except Exception:
        pass
    def _here(p) -> bool:
        try:
            return bool(p) and Path(p).exists()
        except OSError:          # e.g. a folder this process can't read
            return False
    return [p for p in paths if _here(p)]


# One header row for the video: its ⭐ picture, then a single status line (scenes, reviewed,
# library range, thumbnail) — the details sit in ⋯ instead of boxes of their own.
lib_range = ranges.get(video_id) if use_library_range else None
_mp = (CS.CATALOGUE_DIR / "thumbnails" / video_id / Path(scene_by_id[master_sid]["thumbnail_paths"][0]).name
       if master_sid in scene_by_id and scene_by_id[master_sid].get("thumbnail_paths") else None)
with st.container(border=True):
    hc = st.columns([1, 6, 1], vertical_alignment="center")
    with hc[0]:
        if _mp and _mp.exists():
            safe_image(str(_mp), width=120)
        else:
            st.caption("🖼️ automatic picture")
    _bits = [f"**{len(data['scenes'])} scenes**", f"🏷️ {reviewed_count} tagged",
             f"🎭 {role_marked_count}/{len(_usable_all)} role marked"]
    if lib_range:
        _outside = len(data["scenes"]) - len(in_range_scenes(data["scenes"], lib_range))
        _bits.append(f"library range {format_mmss(lib_range[0])}–{format_mmss(lib_range[1])}"
                     + (f" ({_outside} scene(s) outside hidden)" if _outside else ""))
    elif use_library_range:
        _bits.append("no library range — every scene shown")
    else:
        _bits.append("library range off — every scene shown")
    _bits.append(f"⭐ thumbnail: scene {master_sid}" if _mp else "⭐ thumbnail: automatic")
    hc[1].markdown("  ·  ".join(_bits))
    with hc[2].popover("⋯", width="stretch"):
        st.caption(f"Source: {data.get('source_path', '')}")
        st.caption("⭐ The video thumbnail is used for this video in the Media Library and the planner's "
                   "video list — pick it with ⭐ on any scene below.")
        if _mp and st.button("Use the automatic thumbnail", key=f"master_thumb_clear_{selected}"):
            set_master_thumbnail(video_id, None)
            st.session_state["rev_saved_msg"] = "Video thumbnail set to automatic."
            st.rerun()

# 🎬 Video details — actors and video tags, for searching on every page.
def _vd_fill_from_clips(key_tags: str, suggested: list) -> None:
    cur = list(st.session_state.get(key_tags) or [])
    st.session_state[key_tags] = cur + [t for t, _ in suggested if t.lower() not in {c.lower() for c in cur}]


def _vd_add_actor(key_actors: str, key_new: str) -> None:
    name = " ".join((st.session_state.get(key_new) or "").split())
    if name:
        cur = list(st.session_state.get(key_actors) or [])
        if name.lower() not in {c.lower() for c in cur}:
            st.session_state[key_actors] = cur + [name]
            st.session_state[f"_extra_actors_{selected}"] = st.session_state.get(f"_extra_actors_{selected}", []) + [name]
    st.session_state[key_new] = ""


_vd = video_details().get(video_id) or {"actors": [], "tags": []}
_ka, _kt, _kn = f"vd_actors_{selected}", f"vd_tags_{selected}", f"vd_new_actor_{selected}"
_sugg = suggest_video_tags(data["scenes"], ranges.get(video_id))
with st.expander(f"🎬 Video details — {len(_vd['actors'])} actor(s), {len(_vd['tags'])} video tag(s)",
                 expanded=not (_vd["actors"] or _vd["tags"])):
    vc = st.columns(2)
    with vc[0]:
        _actor_opts = sorted(set(all_actors()) | set(_vd["actors"]) | set(st.session_state.get(_ka) or [])
                             | set(st.session_state.get(f"_extra_actors_{selected}", [])), key=str.lower)
        st.multiselect("👤 Actors", _actor_opts, default=_vd["actors"], key=_ka,
                       help="Who's in this video. Pick from names already used, or add a new one below.")
        ac = st.columns([3, 1])
        ac[0].text_input("New actor", key=_kn, placeholder="type a name", label_visibility="collapsed")
        ac[1].button("➕ Add", key=f"vd_add_{selected}", on_click=_vd_add_actor, args=(_ka, _kn),
                     width="stretch")
    with vc[1]:
        _tag_opts = sorted(set(CANDIDATE_LABELS) | set(all_video_tags()) | set(_vd["tags"])
                           | set(st.session_state.get(_kt) or []), key=str.lower)
        st.multiselect("🏷️ Video tags", _tag_opts, default=_vd["tags"], key=_kt,
                       help="Tags for the whole video — for finding it on every page. Separate from the clip tags.")
        if _sugg:
            st.caption("From its clips: " + ", ".join(f"`{tg}` {share:.0%}" for tg, share in _sugg[:10]))
            st.button("✨ Add the clip tags above", key=f"vd_fill_{selected}", on_click=_vd_fill_from_clips,
                      args=(_kt, _sugg),
                      help="Adds every tag that covers at least 10% of the usable footage (and 2+ clips), "
                           "inside the library range, excluded clips left out.")
        else:
            st.caption("No clip tag covers 10% of this video's footage yet.")
    _now_a, _now_t = list(st.session_state.get(_ka) or []), list(st.session_state.get(_kt) or [])
    _changed = (_now_a != _vd["actors"]) or (_now_t != _vd["tags"])
    sc = st.columns([1, 3])
    if sc[0].button("💾 Save video details", key=f"vd_save_{selected}", type="primary" if _changed else "secondary",
                    disabled=not _changed, width="stretch"):
        set_video_details(video_id, actors=_now_a, tags=_now_t)
        for k in (_ka, _kt):
            st.session_state.pop(k, None)
        st.session_state["rev_saved_msg"] = f"Video details saved: {len(_now_a)} actor(s), {len(_now_t)} video tag(s)."
        st.rerun()
    sc[1].caption("✏️ Unsaved changes" if _changed else "Saved with the library (synced to Google Drive like "
                                                         "library ranges).")



# ---------------------------------------------------------------------------
# Edits: unsaved field edits live in widget state until saved
# ---------------------------------------------------------------------------
FIELD_PREFIXES = ("tags", "excluded", "roles")


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
        "roles": list(st.session_state.get(wkey("roles", sid), SR.scene_roles(scene)) or []),
    }


def is_changed(scene: dict, new: dict) -> bool:
    return (new["tags"] != scene_tags(scene)
            or new["excluded"] != bool(scene.get("excluded", False))
            or set(new["roles"]) != set(SR.scene_roles(scene)))


def _qr_key() -> str:
    """Quick role marking's unsaved choices for this video: {scene_id: [role ids]}."""
    return f"rev_qr_{selected}"


def apply_pending(mark_tagged_ids=(), mark_roles_ids=()) -> int:
    """Copy unsaved edits (cards and 🎭 quick marking) into `data`. Two review states:
    a tag / exclude change marks the scene 🏷️ tagged ("reviewed"), a role change marks
    it 🎭 role marked ("roles_reviewed"). Scenes in mark_tagged_ids / mark_roles_ids are
    marked even if unchanged. Returns how many scenes changed."""
    changed = 0
    mark_tagged_ids, mark_roles_ids = set(mark_tagged_ids), set(mark_roles_ids)
    quick = st.session_state.get(_qr_key()) or {}
    for scene in data["scenes"]:
        sid = scene["scene_id"]
        new = pending_fields(scene)
        tag_ch = new is not None and (new["tags"] != scene_tags(scene)
                                      or new["excluded"] != bool(scene.get("excluded", False)))
        new_roles = new["roles"] if new is not None else None
        if sid in quick and (new_roles is None or set(new_roles) == set(SR.scene_roles(scene))):
            new_roles = list(quick[sid])
        role_ch = new_roles is not None and set(new_roles) != set(SR.scene_roles(scene))
        if tag_ch:
            scene["corrected_tags"] = new["tags"]
            scene["excluded"] = new["excluded"]
        if role_ch:
            SR.set_scene_roles(scene, new_roles)
        if tag_ch or role_ch:
            changed += 1
        if tag_ch or sid in mark_tagged_ids:
            if "corrected_tags" not in scene:
                scene["corrected_tags"] = scene_tags(scene)  # confirmed as correct
            scene["reviewed"] = True
        if role_ch or sid in mark_roles_ids:
            scene["roles_reviewed"] = True
    return changed


def forget_widgets() -> None:
    """Fresh widgets next run, so they show what was just saved (and no quick-marking
    choices left over — apply_pending has just saved them)."""
    for k in [k for k in st.session_state.keys() if isinstance(k, str)
              and any(k.startswith(f"{p}_{selected}_") for p in FIELD_PREFIXES)]:
        st.session_state.pop(k, None)
    st.session_state.pop(_qr_key(), None)


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
            else:
                # No picture could be made — the first part keeps the original's.
                orig = ((s.get("edit") or {}).get("original") or {}).get("thumbnail_paths")
                if orig:
                    s["thumbnail_paths"] = list(orig)
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
    """After the full page reload this button causes, scroll back to this scene —
    or, if it's no longer shown (a filter hides it now, or it was combined away),
    to the nearest shown scene at or after its start time."""
    scene = scene_by_id.get(sid)
    st.session_state["rev_scroll_to"] = (selected, sid, SE.scene_start_sec(scene) if scene else None)


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
        st.session_state[cuts_key] = SE.auto_split_points(scene, auto_target_sec, cap=True)
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

    auto_cuts = SE.auto_split_points(scene, auto_target_sec, cap=True)
    ac = st.columns([3, 2])
    ac[0].caption(f"⚡ Auto: {len(auto_cuts) + 1} parts of about {fmt_dur((end - start) / (len(auto_cuts) + 1))}, "
                  f"none over the {auto_target_sec:g}s target (set in the sidebar)." if auto_cuts
                  else f"⚡ Auto: already within the {auto_target_sec:g}s target — no cuts needed.")
    ac[1].button("⚡ Use auto cuts", key=f"{pos_key}_auto", on_click=_set_state, args=(cuts_key, auto_cuts),
                 disabled=not auto_cuts, width="stretch")

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
                 width="stretch")
    bc[1].button("Clear cuts", key=f"{pos_key}_clear", on_click=_set_state, args=(cuts_key, []),
                 disabled=not cuts, width="stretch")

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
                 type="primary", disabled=not cuts, width="stretch", key=f"{pos_key}_go"):
        st.session_state.pop(pos_key, None)
        st.session_state.pop(cuts_key, None)
        back_to(sid)
        do_split(sid, cuts)


@st.dialog("⚡ Auto split long scenes", width="large")
def bulk_auto_split_dialog(candidate_ids: list):
    plans = []
    for sid in candidate_ids:
        s = scene_by_id[sid]
        cuts = SE.auto_split_points(s, auto_target_sec, cap=True)
        if cuts:
            plans.append((sid, cuts, SE.scene_duration(s)))
    if not plans:
        st.info("Nothing to split at this target length.")
        return
    total_parts = sum(len(c) + 1 for _, c, _ in plans)
    st.markdown(f"Split **{len(plans)} scene(s)** into **{total_parts} parts**, each no longer than "
                f"**{auto_target_sec:g}s**:")
    for sid, cuts, dur in plans:
        st.markdown(f"- Scene {sid} · {fmt_dur(dur)} → {len(cuts) + 1} parts of ~{fmt_dur(dur / (len(cuts) + 1))}")
    st.caption("Each split can be undone on its own afterwards. New parts get thumbnails and motion "
               "scores — this takes a few seconds per scene.")
    if st.button(f"⚡ Split {len(plans)} scene(s)", type="primary", width="stretch"):
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
        msg = f"⚡ Auto split {done} scene(s) into parts of up to {auto_target_sec:g}s."
        if unsaved:
            msg += f" Your unsaved changes to {unsaved} scene(s) were saved too."
        write_catalogue(msg, new_thumbs)


@st.dialog("🔗 Auto combine short clips", width="large")
def bulk_auto_combine_dialog(groups: list):
    if not groups:
        st.info("Nothing to combine.")
        return
    n_clips = sum(len(g) for g in groups)
    st.markdown(f"Combine **{n_clips} short clips** into **{len(groups)} clip(s)**:")
    for g in groups:
        total = sum(SE.scene_duration(scene_by_id[i]) for i in g)
        st.markdown(f"- Scenes {g[0]}–{g[-1]} · {len(g)} clips → one clip of {fmt_dur(total)}")
    st.caption(f"Runs of touching clips that are each under {suggest_short_sec:g}s (sidebar). A long run is "
               f"combined into several clips, each no longer than the {auto_target_sec:g}s target. Each "
               "combine can be undone on its own afterwards (↩️ Undo combine on its card).")
    if st.button(f"🔗 Combine into {len(groups)} clip(s)", type="primary", width="stretch"):
        unsaved = apply_pending()
        done, measure = 0, []
        with st.spinner("Combining…"):
            for g in groups:
                try:
                    new_id = SE.merge_scenes(data, g)
                except SE.EditError:
                    continue
                done += 1
                # Measure motion only where a time-weighted score couldn't be kept.
                if not any(s["scene_id"] == new_id and "motion_intensity" in s for s in data["scenes"]):
                    measure.append(new_id)
            if measure:
                SE.fill_motion(data, measure, video_paths())
        msg = f"🔗 Auto combined {done} run(s) of short clips."
        if unsaved:
            msg += f" Your unsaved changes to {unsaved} scene(s) were saved too."
        write_catalogue(msg)


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
# ⚡ Auto split on a card: any (not excluded) scene longer than the target length. The To do bar's
# bulk ⚡ / 🔗 use the suggestions (so a dismissed suggestion is left alone).
auto_ids = [s["scene_id"] for s in SE.auto_split_candidates(in_range, auto_target_sec, factor=1.0, cap=True)]
bulk_split_ids = [i for i in sugg["split"] if i in set(auto_ids)]
combine_groups = SE.auto_combine_groups(in_range, sugg["merge"], max_sec=auto_target_sec)

# Scenes whose thumbnail is missing or unreadable — remake them all in one go.
def _thumb_dest(scene: dict) -> Path:
    paths = scene.get("thumbnail_paths") or []
    return thumbs_dir / (Path(paths[0]).name if paths else f"scene_{int(scene['scene_id']):04d}_manual.jpg")


_no_thumb = [s for s in ordered_all if not image_ok(_thumb_dest(s))]


def _remake_missing_thumbs() -> None:
    bar = st.progress(0.0, text="Remaking thumbnails…")
    made, failed = [], 0
    vpaths = video_paths()
    for n, scene in enumerate(_no_thumb, 1):
        dest = _thumb_dest(scene)
        if SE.write_thumbnail(data, scene, dest, vpaths, sprite_path):
            if not scene.get("thumbnail_paths"):
                scene["thumbnail_paths"] = [dest.name]
            made.append(dest)
        else:
            failed += 1
        bar.progress(n / len(_no_thumb), text=f"Remaking thumbnails… {n}/{len(_no_thumb)}")
    bar.empty()
    if not made:
        st.error("Couldn't make any pictures — no preview copy, source video or timeline thumbnails "
                 "for this video on the server.")
    else:
        write_catalogue(f"Remade {len(made)} thumbnail(s)"
                        + (f" — {failed} couldn't be made." if failed else "."), new_thumbs=made)


# 💡 One "to do" line for this video: missing pictures and split / combine suggestions,
# with their one-click fixes beside it.
_todo = []
if _no_thumb:
    _todo.append(f"🖼️ {len(_no_thumb)} missing thumbnail(s)")
if sugg["split"]:
    _todo.append(f"✂️ {len(sugg['split'])} clip(s) over {auto_target_sec:g}s to split")
if sugg["merge"]:
    _todo.append(f"🔗 {len(sugg['merge'])} run(s) of clips under {suggest_short_sec:g}s to combine")
if _todo:
    with st.container(border=True, key="rev_wrap_todo"):
        tc_ = st.columns([4, 1.3, 1.3, 1.3], vertical_alignment="center")
        tc_[0].markdown("**To do:** " + " · ".join(_todo),
                        help=f"Split: clips over the {auto_target_sec:g}s target length; combine: runs of clips "
                             f"each under {suggest_short_sec:g}s — set both in the sidebar (✂️ Split & combine).")
        if _no_thumb and tc_[1].button(f"🔄 Remake {len(_no_thumb)}", width="stretch",
                                       key=f"remake_all_thumbs_{selected}",
                                       help="Takes a new picture for each from the preview copy / video (or the "
                                            "timeline thumbnails) and queues them for Google Drive."):
            _remake_missing_thumbs()
        if bulk_split_ids and tc_[2].button(f"⚡ Auto split {len(bulk_split_ids)}…", width="stretch",
                                            key=f"bulk_auto_{selected}",
                                            help=f"Splits every suggested clip into parts no longer than "
                                                 f"{auto_target_sec:g}s (you see the list first)."):
            bulk_auto_split_dialog(bulk_split_ids)
        if combine_groups and tc_[3].button(f"🔗 Auto combine {len(sugg['merge'])}…", width="stretch",
                                            key=f"bulk_combine_{selected}",
                                            help=f"Joins every suggested run of clips under {suggest_short_sec:g}s "
                                                 "into one clip (you see the list first)."):
            bulk_auto_combine_dialog(combine_groups)

# Which scenes: tag search, role and "only …" chips; hide excluded sits with the role tools below.
_filters_box = st.container(key="rev_wrap_filters")
fc = _filters_box.columns([2, 2, 4], vertical_alignment="bottom")
tag_filter = fc[0].text_input("Scenes with tag", placeholder="any tag", key="rev_tag_filter")
ROLE_ANY, ROLE_SOME, ROLE_NONE = "Any role", "With a role", "No role"
role_filter = fc[1].selectbox("Role", [ROLE_ANY, ROLE_SOME, ROLE_NONE, *_roles], key="rev_role_filter",
                              format_func=lambda r: r if r in (ROLE_ANY, ROLE_SOME, ROLE_NONE) else SR.label(r, _roles))
ONLY_UNTAGGED, ONLY_UNROLED = "Not tagged", "Not role marked"
ONLY_SUGGESTED, ONLY_EXCLUDED = "Suggestions", "Excluded"
_only = fc[2].pills("Only", [ONLY_UNTAGGED, ONLY_UNROLED, ONLY_SUGGESTED, ONLY_EXCLUDED], selection_mode="multi",
                    key="rev_only",
                    help="Show only scenes that are… (pick several to narrow it further). Not tagged: tags not "
                         "checked yet. Not role marked: no role and not checked yet (excluded scenes left out).") or []
_tools = _filters_box.columns([1.2, 1.8, 1.4, 1.1], vertical_alignment="center")
hide_excluded = _tools[0].toggle("Hide excluded", key="rev_hide_excluded")
show_only_excluded = ONLY_EXCLUDED in _only
show_only_suggested = ONLY_SUGGESTED in _only and show_suggestions
show_only_untagged = ONLY_UNTAGGED in _only
show_only_unroled = ONLY_UNROLED in _only

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
    if hide_excluded and not show_only_excluded and scene.get("excluded"):
        continue
    if role_filter != ROLE_ANY:
        _sr = SR.scene_roles(scene)
        if ((role_filter == ROLE_SOME and not _sr) or (role_filter == ROLE_NONE and _sr)
                or (role_filter not in (ROLE_SOME, ROLE_NONE) and role_filter not in _sr)):
            continue
    if show_only_suggested and show_suggestions and not (
            scene["scene_id"] in split_sugg or scene["scene_id"] in merge_group_of):
        continue
    if show_only_untagged and scene.get("reviewed"):
        continue
    if show_only_unroled and (role_marked(scene) or scene.get("excluded")):
        continue
    shown.append((scene, trimmed_to))

# Motion scores of this video, to say whether a scene is calm or busy for it.
_motions = sorted(s["motion_intensity"] for s in data["scenes"]
                  if isinstance(s.get("motion_intensity"), (int, float)))


def motion_rank(scene: dict):
    """0 = calmest scene in this video, 1 = busiest (None if not measured)."""
    m = scene.get("motion_intensity")
    if not isinstance(m, (int, float)) or not _motions:
        return None
    return sum(v <= m for v in _motions) / len(_motions)


def motion_text(scene: dict) -> str:
    m = scene.get("motion_intensity")
    if not isinstance(m, (int, float)):
        return "🏃 motion —"
    rank = sum(v <= m for v in _motions) / len(_motions) if _motions else 0.5
    level = "calm" if rank < 0.34 else "busy" if rank > 0.66 else "medium"
    return f"🏃 motion {m:.1f} ({level})"


# ---------------------------------------------------------------------------
# Roles: mark every shown scene, and your own roles
# ---------------------------------------------------------------------------

def _bulk_role(role_id, add: bool, ids: list) -> None:
    """Add / remove a role on every shown scene — an unsaved edit like ticking each card.
    role_id None = the role picked in the popover (read now, not when the button was drawn)."""
    role_id = role_id or st.session_state.get("rev_bulk_role")
    for sid in ids:
        k = wkey("roles", sid)
        cur = (list(st.session_state.get(k) or []) if k in st.session_state
               else SR.scene_roles(scene_by_id[sid]))
        if add and role_id not in cur:
            cur.append(role_id)
        elif not add:
            cur = [r for r in cur if r != role_id]
        st.session_state[k] = cur


def _bulk_hints(ids: list) -> None:
    """Accept every 💡 motion hint among the shown scenes."""
    for sid in ids:
        hint = SR.motion_hint(motion_rank(scene_by_id[sid]))
        if hint in _roles:
            _bulk_role(hint, True, [sid])


def _role_fit_inputs(prefix: str, fit: dict) -> dict:
    c = st.columns(2)
    energy = c[0].selectbox("Energy in the music", list(SR.ENERGY), index=list(SR.ENERGY).index(fit["energy"]),
                            format_func=SR.ENERGY.get, key=f"{prefix}_energy")
    place = c[1].selectbox("Where in the track", list(SR.PLACES), index=list(SR.PLACES).index(fit["place"]),
                           format_func=SR.PLACES.get, key=f"{prefix}_place")
    moments = st.multiselect("Musical moments", list(SR.MOMENTS), default=fit["moments"],
                             format_func=SR.MOMENTS.get, key=f"{prefix}_moments")
    suggest = st.toggle("Let the story map wizard suggest it", value=fit["suggest"], key=f"{prefix}_suggest",
                        help="Off: the wizard never places it — you mark its parts of the track yourself.")
    return {"energy": energy, "place": place, "moments": moments, "suggest": suggest}


def role_manager() -> None:
    st.markdown("**Built in**")
    for r in SR.BUILTIN:
        st.markdown(f"{r['emoji']} **{r['name']}** — {r['help']}  \n"
                    f":gray[Music fit: {SR.describe_fit(r['fit'])}]")
    custom = [r for r in _roles.values() if not r["builtin"]]
    st.markdown("**Your own**")
    if not custom:
        st.caption("None yet.")
    for r in custom:
        st.markdown(f"{r['emoji']} **{r['name']}**" + (f" — {r['help']}" if r["help"] else "")
                    + f"  \n:gray[Music fit: {SR.describe_fit(r['fit'])}]")
    st.divider()
    choices = ["➕ New role", *[r["id"] for r in custom]]
    pick = st.selectbox("Add or change", choices, key="rev_role_edit_pick",
                        format_func=lambda r: r if r == choices[0] else f"✏️ {SR.label(r, _roles)}")
    r = _roles.get(pick) or {"name": "", "emoji": "🏷️", "colour": SR.next_colour(), "help": "",
                            "fit": {"energy": "any", "place": "anywhere", "moments": [], "suggest": True}}
    p = f"rev_role_{pick}"
    c = st.columns([3, 1, 1])
    name = c[0].text_input("Name", value=r["name"], key=f"{p}_name", placeholder="e.g. Reveal")
    emoji = c[1].text_input("Emoji", value=r["emoji"], key=f"{p}_emoji", max_chars=4)
    colour = c[2].color_picker("Colour", value=r["colour"], key=f"{p}_colour")
    help_text = st.text_input("What it's for (optional)", value=r["help"], key=f"{p}_help")
    st.caption("**Music fit** — where in a track this role belongs. The story map wizard uses it to "
               "suggest that role's parts of the track.")
    fit = _role_fit_inputs(p, r["fit"])
    bc = st.columns(2)
    try:
        if pick == choices[0]:
            if bc[0].button("➕ Add role", type="primary", width="stretch", key=f"{p}_add"):
                rid = SR.add_role(name, emoji, colour, fit, help_text)
                st.session_state["rev_saved_msg"] = f"Added {SR.label(rid)}."
                st.rerun()
        else:
            if bc[0].button("💾 Save role", type="primary", width="stretch", key=f"{p}_save"):
                SR.update_role(pick, name=name, emoji=emoji, colour=colour, help=help_text, fit=fit)
                st.session_state["rev_saved_msg"] = f"Saved {SR.label(pick)}."
                st.rerun()
            if bc[1].button("🗑️ Delete role", width="stretch", key=f"{p}_del",
                            help="Removes it from the list. Scenes keep the mark (shown as 'deleted') until "
                                 "they're next saved, so adding a role with the same name brings them back."):
                SR.delete_role(pick)
                st.session_state.pop("rev_role_edit_pick", None)
                st.session_state["rev_saved_msg"] = f"Deleted {r['name']}."
                st.rerun()
    except ValueError as e:
        st.error(str(e))


_shown_ids = [s["scene_id"] for s, _ in shown]
rc_ = [_tools[2], _tools[3], _tools[1]]
quick_mode = rc_[2].toggle("🎭 Quick role marking", key="rev_quick_roles",
                           help="A grid of the shown scenes' pictures: pick a role, then click scenes to "
                                "add or remove it — no cards to open. 💾 Save when done. Uses the filters "
                                "above (e.g. Role → No role to see just the unmarked ones).")
with rc_[0].popover(f"🎭 Mark all {len(_shown_ids)}…", width="stretch",
                    disabled=not _shown_ids):
    _brole = st.selectbox("Role", list(_roles), format_func=lambda r: SR.label(r, _roles), key="rev_bulk_role")
    _bc = st.columns(2)
    _bc[0].button("➕ Add to all", width="stretch", on_click=_bulk_role, args=(None, True, _shown_ids))
    _bc[1].button("➖ Remove from all", width="stretch", on_click=_bulk_role,
                  args=(None, False, _shown_ids))
    _n_hints = sum(1 for i in _shown_ids if SR.motion_hint(motion_rank(scene_by_id[i])) in _roles)
    st.button(f"💡 Accept the {_n_hints} motion hint(s)", width="stretch", disabled=not _n_hints,
              on_click=_bulk_hints, args=(_shown_ids,),
              help="Calmest fifth of this video's scenes → 🌙 Slow, busiest fifth → 💥 Action. Check them over.")
    st.caption("Unsaved until you 💾 save, like ticking each card.")
with rc_[1].popover("Roles…", width="stretch"):
    role_manager()


# ---------------------------------------------------------------------------
# Scene cards
# ---------------------------------------------------------------------------
MARK_TAGGED, MARK_ROLES = "🏷️ Tagged", "🎭 Role marked"


def save_row(where: str) -> bool:
    c = st.container(key=f"rev_wrap_save_{where}").columns([2, 3, 2], vertical_alignment="center")
    c[0].caption(f"{len(shown)} scene(s) shown")
    c[1].pills("Mark all shown as", [MARK_TAGGED, MARK_ROLES], selection_mode="multi",
               key=f"rev_mark_all_{where}",
               help="None picked: only scenes you changed are marked (a tag / exclude change → 🏷️ tagged, a role "
                    "change → 🎭 role marked). Pick one or both to mark every scene shown when you save "
                    "(you've checked them all).")
    return c[2].button("💾 Save changes", type="primary", width="stretch", key=f"rev_save_{where}",
                       help="Saves the tag, exclude and role edits on this video. They're kept until you save — "
                            "changing the video or a filter discards them. Split, combine and ⭐ save straight away.")


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
    cols = st.container(key=f"rev_wrap_card_{sid}").columns([1, 2])
    with cols[0]:
        if scene.get("thumbnail_paths"):
            thumb_path = thumbs_dir / Path(scene["thumbnail_paths"][0]).name
            if image_ok(thumb_path):
                safe_image(str(thumb_path), width=280)
            else:
                st.caption(f"🖼️ Picture {'unreadable' if thumb_path.exists() else 'missing'}")
                if st.button("🔄 Remake picture", key=f"remake_thumb_{selected}_{sid}",
                             help="Takes a new picture from the video (or the timeline thumbnails) "
                                  "and queues it for Google Drive."):
                    back_to(sid)
                    remake_thumbnail(scene, thumb_path)
        if sid != master_sid and scene.get("thumbnail_paths"):
            if st.button("⭐ Use as video thumbnail", key=f"star_{selected}_{sid}",
                         help="The picture shown for this video in the Media Library and the planner's video list."):
                back_to(sid)
                set_star(sid)
    with cols[1]:
        dur = SE.scene_duration(scene)
        # Header: scene, time, length, motion — then its state as small badges (not boxes).
        st.markdown(f"**Scene {SE.scene_label(scene)}**  ·  {format_mmss(SE.scene_start_sec(scene))}"
                    f"  ·  {fmt_dur(dur)}  ·  {motion_text(scene)}",
                    help=f"{scene['start_tc']} → {scene['end_tc']}")
        badges = []
        if scene.get("reviewed"):
            badges.append(":green-badge[🏷️ ✓ Tagged]")
        if role_marked(scene):
            badges.append(":violet-badge[🎭 ✓ Role marked]")
        if sid == master_sid:
            badges.append(":blue-badge[⭐ Video thumbnail]")
        if trimmed_to:
            badges.append(f":orange-badge[✂️ library range: uses {format_mmss(trimmed_to[0])}–"
                          f"{format_mmss(trimmed_to[1])}]")
        if sid in split_sugg:
            badges.append(f":orange-badge[✂️ {fmt_dur(dur)} long — consider splitting]")
        if badges or sid in split_sugg:
            bc_ = st.columns([5, 1], vertical_alignment="center")
            bc_[0].markdown(" ".join(badges))
            if sid in split_sugg and bc_[1].button("Dismiss", key=f"split_dismiss_{selected}_{sid}",
                                                   help="Hide the split suggestion for this scene."):
                back_to(sid)
                dismiss(SE.suggestion_key("split", [sid]))
        st.multiselect("Tags", options=sorted(set(CANDIDATE_LABELS) | set(current_tags)),
                       default=current_tags, key=wkey("tags", sid))
        _rk = wkey("roles", sid)
        _cur = SR.scene_roles(scene)
        _opts = list(_roles) + [r for r in _cur if r not in _roles]
        st.pills("Roles", _opts, selection_mode="multi", key=_rk,
                    default=None if _rk in st.session_state else _cur,
                    format_func=lambda r: SR.label(r, _roles), label_visibility="collapsed",
                    help="What this scene is for in the story. 🎬 Intro / 🏁 Outro: the start and end "
                         "(the planner's 'Prefer Intro/Outro' toggle). 🌙 Slow: build-ups and breakdowns. "
                         "💥 Action: the high points. Add your own under 🎭 Scene roles. Independent of Exclude.")
        _hint = SR.motion_hint(motion_rank(scene))
        if _hint and _hint in _roles and _hint not in (st.session_state.get(_rk) or []):
            st.caption(f":gray[*💡 one of the {'calmest' if _hint == 'slow' else 'busiest'} scenes here — "
                       f"maybe {SR.label(_hint, _roles)}?*]")
        new = pending_fields(scene)
        if new is not None and is_changed(scene, new):
            st.caption("✏️ Unsaved changes")

        # Only the buttons that apply, side by side (no gaps for missing ones).
        nxt = next_of.get(sid)
        edit = scene.get("edit") or {}
        can_undo = edit.get("kind") in ("merge", "split") or "split_from" in scene
        actions = ["split"] + (["auto"] if sid in auto_ids else []) + (["merge"] if nxt is not None else []) \
            + (["undo"] if can_undo else [])
        _row = st.container(key=f"rev_wrap_act_{sid}").columns(max(4, 1 + len(actions)),
                                                               vertical_alignment="center")
        _row[0].checkbox("🚫 Exclude", value=scene.get("excluded", False), key=wkey("excluded", sid))
        bc = dict(zip(actions, _row[1:]))
        if bc["split"].button("✂️ Split…", key=f"split_{selected}_{sid}", width="stretch"):
            back_to(sid)
            st.session_state["rev_open_dialog"] = (selected, sid, False)
            st.rerun()
        if "auto" in bc and bc["auto"].button("⚡ Auto split", key=f"auto_{selected}_{sid}", width="stretch",
                                              help=f"Splits it straight away into parts of about {auto_target_sec:g}s "
                                                   "(↩️ Undo split puts it back). ✂️ Split… lets you place cuts by hand."):
            back_to(sid)
            do_split(sid, SE.auto_split_points(scene_by_id[sid], auto_target_sec, cap=True))
        if "merge" in bc and bc["merge"].button("🔗 Combine", key=f"merge_next_{selected}_{sid}",
                                                width="stretch",
                                                help=f"Join this scene and the next one (scene {nxt}) into one."):
            back_to(sid)
            do_merge([sid, nxt])
        if "undo" in bc:
            label = "↩️ Undo combine" if edit.get("kind") == "merge" else "↩️ Undo split"
            if bc["undo"].button(label, key=f"undo_{selected}_{sid}", width="stretch"):
                back_to(sid)
                do_undo(sid)


QUICK_PAGE = 36   # pictures per page in quick role marking


def _qr_page_key() -> str:
    return f"rev_qr_page_{selected}"


def _qr_ver() -> int:
    """Bumped when the unsaved choices change outside the cards (➕ all, hints, undo), so
    every card's role buttons are drawn fresh from them."""
    return st.session_state.get(f"rev_qr_ver_{selected}", 0)


def _qr_bump() -> None:
    st.session_state[f"rev_qr_ver_{selected}"] = _qr_ver() + 1


def _qr_roles(sid) -> list:
    """A scene's roles in quick marking: the unsaved choice, else what's saved."""
    pend = st.session_state.get(_qr_key()) or {}
    return list(pend[sid]) if sid in pend else SR.scene_roles(scene_by_id[sid])


def _qr_pick(sid, widget_key) -> None:
    """A card's role buttons changed — keep the choice (unsaved) for this scene."""
    st.session_state.setdefault(_qr_key(), {})[sid] = list(st.session_state.get(widget_key) or [])


def _qr_all(ids: list, add: bool) -> None:
    role = st.session_state.get("rev_qr_role")
    if not role:
        return
    for sid in ids:
        cur = _qr_roles(sid)
        if add and role not in cur:
            cur = cur + [role]
        elif not add:
            cur = [r for r in cur if r != role]
        st.session_state.setdefault(_qr_key(), {})[sid] = cur
    _qr_bump()


def _qr_hints(ids: list) -> None:
    for sid in ids:
        hint = SR.motion_hint(motion_rank(scene_by_id[sid]))
        if hint in _roles and hint not in _qr_roles(sid):
            st.session_state.setdefault(_qr_key(), {})[sid] = _qr_roles(sid) + [hint]
    _qr_bump()


def _qr_discard() -> None:
    st.session_state.pop(_qr_key(), None)
    _qr_bump()


def _qr_goto(page: int) -> None:
    st.session_state[_qr_page_key()] = page
    st.session_state["_rev_qr_scroll"] = True


def _qr_changed() -> list:
    pend = st.session_state.get(_qr_key()) or {}
    return [sid for sid, roles in pend.items()
            if sid in scene_by_id and set(roles) != set(SR.scene_roles(scene_by_id[sid]))]


def _qr_save(changed: list) -> None:
    """Writes the quick-marking choices: each changed scene is marked 🎭 role marked
    (its 🏷️ tagged state is left alone)."""
    for sid in changed:
        sc = scene_by_id[sid]
        SR.set_scene_roles(sc, _qr_roles(sid))
        sc["roles_reviewed"] = True
    st.session_state.pop(_qr_key(), None)
    write_catalogue(f"Saved roles for {len(changed)} scene(s).")


@st.fragment
def quick_role_grid(ids: list) -> None:
    """🎭 Quick role marking — every role as a button on each picture: tap to add or
    remove it. Clicks only rerun this grid; 💾 Save writes them all at once."""
    st.markdown('<div id="rev-qr-top"></div>', unsafe_allow_html=True)
    changed = _qr_changed()
    n_pages = (len(ids) - 1) // QUICK_PAGE + 1
    pk = _qr_page_key()
    st.session_state[pk] = min(max(1, int(st.session_state.get(pk, 1) or 1)), n_pages)

    top = st.columns([4, 2, 2], vertical_alignment="center")
    top[0].caption("Tap a role under a picture to add or remove it:  "
                   + "  ·  ".join(SR.label(r, _roles) for r in _roles))
    if top[1].button(f"💾 Save {len(changed)} change(s)", type="primary", disabled=not changed,
                     width="stretch", key="rev_qr_save"):
        _qr_save(changed)
    top[2].button("↩️ Undo unsaved", disabled=not changed, width="stretch", key="rev_qr_discard",
                  on_click=_qr_discard)

    bc = st.columns([2, 2, 2, 2, 2], vertical_alignment="bottom")
    if n_pages > 1:
        bc[4].number_input(f"Page (of {n_pages})", 1, n_pages, key=pk)
    page = st.session_state[pk]
    page_ids = ids[(page - 1) * QUICK_PAGE: page * QUICK_PAGE]
    role = bc[0].selectbox("For every picture on this page", list(_roles), key="rev_qr_role",
                           index=list(_roles).index("action") if "action" in _roles else 0,
                           format_func=lambda r: SR.label(r, _roles))
    bc[1].button(f"➕ On all {len(page_ids)}", disabled=not role, on_click=_qr_all, args=(page_ids, True),
                 width="stretch", key="rev_qr_all_add")
    bc[2].button("➖ Off all of these", disabled=not role, on_click=_qr_all, args=(page_ids, False),
                 width="stretch", key="rev_qr_all_off")
    bc[3].button("💡 Accept motion hints", on_click=_qr_hints, args=(page_ids,), width="stretch",
                 key="rev_qr_hints",
                 help="Calmest fifth of this video's scenes → 🌙 Slow, busiest fifth → 💥 Action.")

    ver = _qr_ver()
    per_row = 6
    for i in range(0, len(page_ids), per_row):
        cols = st.columns(per_row)
        for col, sid in zip(cols, page_ids[i:i + per_row]):
            sc = scene_by_id[sid]
            cur = _qr_roles(sid)
            with col.container(border=True):
                paths = sc.get("thumbnail_paths") or []
                tp = thumbs_dir / Path(paths[0]).name if paths else None
                if tp is not None and image_ok(tp):
                    safe_image(str(tp), width="stretch")
                else:
                    st.caption("🖼️ no picture")
                hint = SR.motion_hint(motion_rank(sc))
                st.caption(f"**{SE.scene_label(sc)}** · {fmt_dur(SE.scene_duration(sc))}"
                           + (" · 🚫" if sc.get("excluded") else ""))
                # Motion score (re-measured for each part when a scene is split) and where it sits in
                # this video. A split part with no curve of its own still has the pre-split score.
                copied = sc.get("split_from") is not None and not (sc.get("motion_curve") or {}).get("values") \
                    and sc.get("motion_measured") != "reviewer"
                st.caption(motion_text(sc) + (" · from before the split" if copied else ""))
                # Marked roles as solid badges (green = saved, orange = not saved yet); a motion
                # suggestion only as faint grey italics, so the two can't be mixed up.
                saved = set(SR.scene_roles(sc))
                if cur:
                    st.markdown(" ".join(
                        (":green-badge[✅ " if r in saved else ":orange-badge[✏️ ")
                        + (SR.label(r, _roles) if r in _roles else f"🏷️ {r}") + "]" for r in cur))
                if hint in _roles and hint not in cur:
                    st.caption(f":gray[*💡 suggests {SR.label(hint, _roles)}*]")
                # Every role as a button (emoji — the names are in the line at the top).
                wk = f"rev_qr_p_{selected}_{sid}_{ver}"
                opts = list(_roles) + [r for r in cur if r not in _roles]
                st.pills("Roles", opts, selection_mode="multi", key=wk,
                         default=None if wk in st.session_state else cur,
                         format_func=lambda r: _roles[r]["emoji"] if r in _roles else "🏷️",
                         label_visibility="collapsed", on_change=_qr_pick, args=(sid, wk))
                # ✂️ / ⚡ without leaving quick marking (unsaved role choices are saved with the split;
                # the parts keep this scene's roles and each gets its own motion score).
                ec = st.columns(2)
                if ec[0].button("✂️", key=f"rev_qr_split_{selected}_{sid}", width="stretch",
                                help="Split… — place the cuts by hand."):
                    st.session_state["rev_open_dialog"] = (selected, sid, False)
                    st.rerun(scope="app")
                if ec[1].button("⚡", key=f"rev_qr_auto_{selected}_{sid}", width="stretch",
                                disabled=sid not in auto_ids,
                                help=(f"Auto split into parts no longer than {auto_target_sec:g}s (↩️ undo on its "
                                      "card outside quick marking)." if sid in auto_ids else
                                      f"Auto split is offered on scenes over the {auto_target_sec:g}s target "
                                      "length (sidebar → ✂️ Split & combine).")):
                    do_split(sid, SE.auto_split_points(scene_by_id[sid], auto_target_sec, cap=True))

    # Bottom: save and page buttons, so there's no scrolling back up after each page.
    changed = _qr_changed()
    st.divider()
    fb = st.columns([3, 2, 2, 2], vertical_alignment="center")
    fb[0].caption((f"✏️ {len(changed)} unsaved change(s)" if changed else "No unsaved changes")
                  + (f"  ·  page {page} of {n_pages}" if n_pages > 1 else ""))
    if fb[1].button(f"💾 Save {len(changed)} change(s)", type="primary", disabled=not changed,
                    width="stretch", key="rev_qr_save_bottom"):
        _qr_save(changed)
    if n_pages > 1:
        fb[2].button("◀ Previous page", disabled=page <= 1, width="stretch", key="rev_qr_prev",
                     on_click=_qr_goto, args=(page - 1,))
        fb[3].button("Next page ▶", disabled=page >= n_pages, width="stretch", key="rev_qr_next",
                     on_click=_qr_goto, args=(page + 1,),
                     help="Unsaved choices are kept while you move between pages — 💾 save when done.")
    if st.session_state.pop("_rev_qr_scroll", False):
        import embed as _embed
        _embed.html(
            "<script>(function () { var tries = 0; function go() {"
            "  var el = window.parent.document.getElementById('rev-qr-top');"
            "  if (el) { el.scrollIntoView({block: 'start'}); } else if (tries++ < 20) { setTimeout(go, 150); }"
            "} setTimeout(go, 150); })();</script>", height=0)


if quick_mode:
    if shown:
        quick_role_grid([sc["scene_id"] for sc, _ in shown])
    else:
        st.info("No scenes match these filters.")
elif not shown:
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

    if save_top or save_bottom:
        marks = set(st.session_state.get("rev_mark_all_top") or []) | set(st.session_state.get("rev_mark_all_bottom") or [])
        shown_ids = [s["scene_id"] for s, _ in shown]
        changed = apply_pending(shown_ids if MARK_TAGGED in marks else (),
                                shown_ids if MARK_ROLES in marks else ())
        if changed or marks:
            what = " and ".join(w for w, m in (("tagged", MARK_TAGGED), ("role marked", MARK_ROLES)) if m in marks)
            write_catalogue(f"Saved {changed} changed scene(s)"
                            + (f"; all {len(shown)} shown scenes marked {what}." if marks else "."))
        else:
            st.session_state["rev_saved_msg"] = "Nothing had changed — nothing to save."
            st.rerun()

# Open a dialog asked for by a scene card (cards are fragments, so they ask the
# page to open it on its next full run instead of opening it themselves).
_dlg = st.session_state.pop("rev_open_dialog", None)
if _dlg and _dlg[0] == selected and _dlg[1] in scene_by_id:
    split_dialog(_dlg[1], auto=_dlg[2])

# Scroll back to the scene whose button reloaded the page (see back_to).
_scroll = None
_want = st.session_state.pop("rev_scroll_to", None)
if _want and _want[0] == selected and shown:
    _shown_ids = [sc["scene_id"] for sc, _ in shown]
    if _want[1] in _shown_ids:
        _scroll = _want[1]
    elif _want[2] is not None:
        _after = [sc["scene_id"] for sc, _ in shown if SE.scene_start_sec(sc) >= _want[2] - 0.05]
        _scroll = _after[0] if _after else _shown_ids[-1]
if _scroll is not None:
    import embed as _embed
    _embed.html(
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
