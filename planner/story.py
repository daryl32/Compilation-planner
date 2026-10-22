"""🎭 Step 1 — the story map (which parts of the song call for which scene role).

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

from pathlib import Path
import streamlit as st
from render_preview import colab_to_local
import story_map as STM
import story_map_editor as SME
import preview_snippets
from config import AUDIO_DIR, PREVIEW_DIR
from track_files import track_file

from planner.state import (
    download_source_video, _DRIVE_SYNC_AVAILABLE, STORY_KEY
)
from planner.data import (
    _add_video_to_selection, load_track, _local_media
)
from planner.projects import (
    autosave_now
)
from planner.audio import (
    planned_segments, segmentation_settings
)

# ---------------------------------------------------------------------------
# Step 1: Story map — which parts of the song call for which scene role
# (story_map.py for the logic; scene_roles.py for the roles)
# ---------------------------------------------------------------------------

def _track_mtime(track_id: str) -> float:
    try:
        return track_file(track_id, AUDIO_DIR).stat().st_mtime
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
        st.session_state[STORY_KEY] = {"track_id": track_id, "regions": saved, "v": 1, "edited": False,
                                       "source": "default"}
        return saved
    return None


def _story_source() -> str:
    sm = st.session_state.get(STORY_KEY) if isinstance(st.session_state.get(STORY_KEY), dict) else {}
    return sm.get("source") or "project"


def _story_set(regions: list, edited: bool = True, source: str = None) -> None:
    """Change this project's story map. Edits stay with the project (source "project") until you save them
    as the track's default or a named map — nothing is written to the track's maps from here."""
    sm = st.session_state.get(STORY_KEY) if isinstance(st.session_state.get(STORY_KEY), dict) else {}
    same_track = sm.get("track_id") == st.session_state.get("track_id")
    st.session_state[STORY_KEY] = {
        "track_id": st.session_state.get("track_id"),
        "regions": regions,
        "v": int(sm.get("v", 0)) + 1 if same_track else 1,
        "edited": edited,
        "source": source or ("project" if edited else (sm.get("source") if same_track else None) or "project"),
        "from": sm.get("from") if same_track and not source else (source or sm.get("source")),
    }
    st.session_state.pop("sm_play", None)


def _story_use(choice: str) -> None:
    """🗺️ Map for this project: the track's default, a named map, or keep this project's own."""
    tid = st.session_state.get("track_id")
    maps = STM.track_maps(tid)
    if choice == "default" and maps["default"]:
        _story_set(maps["default"], edited=False, source="default")
    elif choice.startswith("named:") and choice[6:] in maps["named"]:
        _story_set(maps["named"][choice[6:]], edited=False, source=choice)


def _story_save_default() -> None:
    STM.save_map(st.session_state.get("track_id"), current_story_map(st.session_state.get("track_id")))
    _story_set(current_story_map(st.session_state.get("track_id")), edited=False, source="default")
    st.session_state["_story_saved_msg"] = "⭐ Saved as this track's default story map."


def _story_save_named(name_key: str) -> None:
    name = " ".join(str(st.session_state.get(name_key) or "").split())
    if not name:
        st.session_state["_story_saved_msg"] = "Give the new map a name first."
        return
    regions = current_story_map(st.session_state.get("track_id"))
    STM.save_named(st.session_state.get("track_id"), name, regions)
    _story_set(regions, edited=False, source=f"named:{name}")
    st.session_state[name_key] = ""
    st.session_state["_story_saved_msg"] = f"📁 Saved as “{name}” — pick it for other projects on this track."


def _story_update_named(name: str) -> None:
    regions = current_story_map(st.session_state.get("track_id"))
    STM.save_named(st.session_state.get("track_id"), name, regions)
    _story_set(regions, edited=False, source=f"named:{name}")
    st.session_state["_story_saved_msg"] = f"📁 “{name}” updated."


def render_story_map_picker(track_id: str) -> None:
    """🗺️ Which map this project uses, and — once you've changed it — where to keep the changes."""
    maps = STM.track_maps(track_id)
    src = _story_source()
    sm = st.session_state.get(STORY_KEY) or {}
    opts = (["default"] if maps["default"] else []) + [f"named:{n}" for n in sorted(maps["named"], key=str.lower)]
    if src == "project" or src not in opts:
        opts = ["project"] + opts
    label = {"default": "⭐ Track default", "project": "✏️ This project's own (not saved to the track)"}
    with st.container(border=True):
        c = st.columns([3, 4])
        c[0].selectbox("🗺️ Map for this project", opts, index=opts.index(src) if src in opts else 0,
                       format_func=lambda o: label.get(o) or f"📁 {o[6:]}", key=f"sm_pick_{sm.get('v', 0)}",
                       on_change=lambda k=f"sm_pick_{sm.get('v', 0)}": _story_use(st.session_state[k]),
                       help="The track's default map is made when the track is analysed, and is what the "
                            "recommender, the analytics and new projects use. Named maps are alternatives you've "
                            "saved for this track. Edits here stay with this project until you save them.")
        msg = st.session_state.pop("_story_saved_msg", None)
        with c[1]:
            if msg:
                st.success(msg)
            elif src == "project":
                st.caption("✏️ Changed for this project — save it for the track if you want it again:")
                bc = st.columns(2)
                bc[0].button("⭐ Save as the track's default", width="stretch", on_click=_story_save_default,
                             help="Replaces the track's default map — used by new projects, the recommender and "
                                  "the analytics.")
                with bc[1].popover("📁 Save as a new map", width="stretch"):
                    nk = f"sm_new_name_{sm.get('v', 0)}"
                    st.text_input("Name", key=nk, placeholder="e.g. Long intro")
                    st.button("📁 Save", key=f"{nk}_go", on_click=_story_save_named, args=(nk,), type="primary")
                    came_from = sm.get("from") or ""          # None for maps from a suggestion / older projects
                    if came_from.startswith("named:"):
                        st.button(f"💾 Update “{came_from[6:]}” instead", key=f"{nk}_upd",
                                  on_click=_story_update_named, args=(came_from[6:],))
            else:
                st.caption("Using the track's " + ("default map." if src == "default" else f"map “{src[6:]}”.")
                           + " Any change you make stays with this project until you save it.")


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
        SME.render_empty(duration, analysis, cut_times, set_regions=_story_set, ns="sm")
        return

    render_story_map_picker(track_id)
    SME.render_editor(track, segs, ns="sm", get_regions=lambda: current_story_map(st.session_state["track_id"]),
                      set_regions=_story_set, version=st.session_state[STORY_KEY].get("v", 0),
                      analysis=analysis, play=lambda r, lead, key: _story_play_audio(track, r, lead, key))


def _story_play_audio(track: dict, r: dict, lead: bool, key: str) -> None:
    """▶ on a story map part: cut that stretch of the song (2s early if asked) and play it."""
    audio_path, missing = _local_media(track.get("source_path", ""))
    if missing:
        st.warning(f"Needs {Path(colab_to_local(missing)).name} on the server first.")
        if _DRIVE_SYNC_AVAILABLE and "MyDrive/" in missing and st.button(
                "⬇️ Download the track from Google Drive", key=f"{key}_dl"):
            bar = st.progress(0.0, text="Downloading…")
            err = download_source_video(missing.split("MyDrive/", 1)[1], Path(colab_to_local(missing)),
                                        progress_callback=lambda p: bar.progress(min(p, 1.0)))
            bar.empty()
            if err:
                st.error(f"Download failed: {err}")
            else:
                st.rerun()
        return
    with st.spinner("Cutting the music…"):
        clip, err = preview_snippets.make_audio_snippet(PREVIEW_DIR / "snippets", audio_path, r["start"],
                                                        r["end"] - r["start"], lead_in=2.0 if lead else 0.0)
    if err:
        st.error(f"Couldn't cut the music: {err}")
    else:
        st.audio(str(clip), autoplay=True)


def _add_new_videos(video_ids: list) -> None:
    """🆕 Add new videos — tick every recently added video (adding it to the list if needed)."""
    for vid in video_ids:
        _add_video_to_selection(vid)

