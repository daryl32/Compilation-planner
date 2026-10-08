"""
Media Library — two libraries, picked at the top of the page:
  Videos: search, filter and sort, and set each video's library time range
          (the part of the video the Compilation Planner should use).
  Audio:  search, filter and sort the music tracks, see their energy shape
          and play them; analyse new tracks (and add the extra cut-finding
          analysis to existing ones) on the server.
"""

import json
import time
from pathlib import Path

import streamlit as st

import config
from config import CATALOGUE_DIR, AUDIO_DIR
from library_common import (
    scene_tags, tc_to_seconds, format_mmss, overlap_with_range,
    library_ranges, set_library_range, master_thumbnail_scenes,
    render_range_picker,
)

from page_setup import page_setup

PAGE_SIZE = 20

page_setup("Media Library", "pages/3_Media_Library.py", title="📁 Media Library")
library_kind = st.radio("Library", ["🎬 Videos", "🎵 Audio"], horizontal=True,
                        key="lib_kind", label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _catalogue_signature() -> tuple:
    """(name, mtime) of every catalogue file — the cache below reloads whenever
    any file is added, removed or edited (e.g. by the Reviewer or a Drive sync)."""
    sig = []
    for f in sorted(CATALOGUE_DIR.glob("*.json")):
        try:
            sig.append((f.name, f.stat().st_mtime))
        except OSError:
            continue
    return tuple(sig)


@st.cache_data(max_entries=2)
def load_library(signature: tuple) -> dict:
    """video_id -> slim summary of its catalogue (no motion curves)."""
    videos = {}
    for name, mtime in signature:
        try:
            cat = json.loads((CATALOGUE_DIR / name).read_text())
        except (ValueError, OSError):
            continue
        if not (isinstance(cat, dict) and "video_id" in cat and "scenes" in cat):
            continue
        scenes = []
        for s in sorted(cat["scenes"], key=lambda s: s["scene_id"]):
            try:
                start, end = tc_to_seconds(s["start_tc"]), tc_to_seconds(s["end_tc"])
            except (KeyError, ValueError):
                continue
            thumbs = s.get("thumbnail_paths") or []
            scenes.append({
                "scene_id": s["scene_id"], "start": start, "end": end,
                "tags": scene_tags(s),
                "excluded": bool(s.get("excluded")),
                "intro": bool(s.get("intro_candidate")),
                "outro": bool(s.get("outro_candidate")),
                "reviewed": bool(s.get("reviewed")),
                "thumb": Path(thumbs[0]).name if thumbs else None,
            })
        vid = cat["video_id"]
        videos[vid] = {
            "video_id": vid,
            "source_path": cat.get("source_path", ""),
            "duration": scenes[-1]["end"] if scenes else 0.0,
            "scenes": scenes,
            "added": mtime,
        }
    return videos


def trim_intro_outro_range(video: dict) -> tuple:
    """Range that drops leading intro/excluded scenes and trailing outro/excluded
    scenes. The whole video if nothing is flagged at either end."""
    scenes = video["scenes"]
    if not scenes:
        return 0.0, video["duration"]
    first, last = 0, len(scenes) - 1
    while first <= last and (scenes[first]["intro"] or scenes[first]["excluded"]):
        first += 1
    while last >= first and (scenes[last]["outro"] or scenes[last]["excluded"]):
        last -= 1
    if first > last:
        return 0.0, video["duration"]
    return scenes[first]["start"], scenes[last]["end"]


def scene_matches(scene: dict, wanted: set, match_all: bool) -> bool:
    tags = {t.lower() for t in scene["tags"]}
    return wanted <= tags if match_all else bool(wanted & tags)


def video_stats(video: dict, time_range, wanted: set, match_all: bool) -> dict:
    usable = tagged = 0.0
    tag_counts = {}
    first_thumb = None
    for s in video["scenes"]:
        eff_start, eff_end = overlap_with_range(s["start"], s["end"], time_range)
        if eff_start is None or s["excluded"]:
            continue
        secs = eff_end - eff_start
        usable += secs
        if first_thumb is None and s["thumb"]:
            first_thumb = s["thumb"]
        for t in s["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        if wanted and scene_matches(s, wanted, match_all):
            tagged += secs
    n = len(video["scenes"])
    return {
        "usable": usable,
        "tagged": tagged,
        "reviewed_pct": (100.0 * sum(s["reviewed"] for s in video["scenes"]) / n) if n else 0.0,
        "top_tags": sorted(tag_counts.items(), key=lambda kv: -kv[1])[:6],
        "thumb": first_thumb,
    }


# ---------------------------------------------------------------------------
# Sidebar: Drive (shared by both libraries)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

def render_videos():
    library = load_library(_catalogue_signature())
    ranges = library_ranges()
    masters = master_thumbnail_scenes()

    st.sidebar.header("Filter")
    name_query = st.sidebar.text_input("Search name", placeholder="part of a video name")
    all_tags = sorted({t for v in library.values() for s in v["scenes"] for t in s["tags"]}, key=str.lower)
    tag_filter = st.sidebar.multiselect("Tags", all_tags)
    match_all = st.sidebar.radio(
        "Scenes must have", ["Any of these tags", "All of these tags"], horizontal=True,
        disabled=not tag_filter,
    ) == "All of these tags"
    range_filter = st.sidebar.selectbox("Library range", ["Any", "Has a range", "No range"])
    review_filter = st.sidebar.selectbox("Review status", ["Any", "Fully reviewed", "Not fully reviewed"])
    max_minutes = max(1, int(max((v["duration"] for v in library.values()), default=60) // 60) + 1)
    duration_filter = st.sidebar.slider("Duration (minutes)", 0, max_minutes, (0, max_minutes))

    st.sidebar.header("Sort")
    SORTS = {
        "Tagged footage": lambda r: r["stats"]["tagged"],
        "Name": lambda r: r["video"]["video_id"].lower(),
        "Duration": lambda r: r["video"]["duration"],
        "Usable duration": lambda r: r["stats"]["usable"],
        "Scene count": lambda r: len(r["video"]["scenes"]),
        "% reviewed": lambda r: r["stats"]["reviewed_pct"],
        "Date added": lambda r: r["video"]["added"],
    }
    sort_options = list(SORTS) if tag_filter else [k for k in SORTS if k != "Tagged footage"]
    sort_by = st.sidebar.selectbox("Sort by", sort_options,
                                   help="Tagged footage = seconds of usable footage whose scenes match the tag filter.")
    descending = st.sidebar.toggle("Descending", value=(sort_by != "Name"))

    # ---------------------------------------------------------------------------
    # Apply filters
    # ---------------------------------------------------------------------------
    wanted = {t.lower() for t in tag_filter}
    rows = []
    for vid, video in library.items():
        if name_query and name_query.lower() not in vid.lower():
            continue
        rng = ranges.get(vid)
        if range_filter == "Has a range" and not rng:
            continue
        if range_filter == "No range" and rng:
            continue
        if not (duration_filter[0] * 60 <= video["duration"] <= duration_filter[1] * 60):
            continue
        stats = video_stats(video, rng, wanted, match_all)
        if review_filter == "Fully reviewed" and stats["reviewed_pct"] < 100:
            continue
        if review_filter == "Not fully reviewed" and stats["reviewed_pct"] >= 100:
            continue
        if wanted and stats["tagged"] <= 0:
            continue
        rows.append({"video": video, "stats": stats, "range": rng})

    rows.sort(key=SORTS[sort_by], reverse=descending)

    if not library:
        st.info("No video catalogues found. Process a video in Colab, then sync from Google Drive.")
        st.stop()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Videos shown", f"{len(rows)} / {len(library)}")
    metric_cols[1].metric("Usable footage", format_mmss(sum(r["stats"]["usable"] for r in rows)))
    if tag_filter:
        metric_cols[2].metric("Tagged footage", format_mmss(sum(r["stats"]["tagged"] for r in rows)))
    else:
        metric_cols[2].metric("With library range", sum(1 for r in rows if r["range"]))

    if not rows:
        st.info("No videos match these filters.")
        st.stop()

    # ---------------------------------------------------------------------------
    # Video list
    # ---------------------------------------------------------------------------
    n_pages = (len(rows) - 1) // PAGE_SIZE + 1
    page = st.number_input(f"Page (of {n_pages})", 1, n_pages, 1) if n_pages > 1 else 1
    st.session_state.setdefault("lib_open_picker", None)

    for row in rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        video, stats, rng = row["video"], row["stats"], row["range"]
        vid = video["video_id"]

        with st.container(border=True):
            cols = st.columns([2, 6, 1])
            with cols[0]:
                # ⭐ master thumbnail chosen in the Reviewer, else the first usable scene's
                _master = next((s["thumb"] for s in video["scenes"]
                                if s["scene_id"] == masters.get(vid) and s["thumb"]), None)
                _thumb_name = _master or stats["thumb"]
                if _thumb_name:
                    thumb_path = CATALOGUE_DIR / "thumbnails" / vid / _thumb_name
                    if thumb_path.exists():
                        st.image(str(thumb_path), width=220)
            with cols[1]:
                st.markdown(f"**{vid}**")
                line = (f"⏱ {format_mmss(video['duration'])}  ·  usable {format_mmss(stats['usable'])}"
                        f"  ·  {len(video['scenes'])} scenes  ·  {stats['reviewed_pct']:.0f}% reviewed")
                if tag_filter:
                    line += f"  ·  🏷️ {format_mmss(stats['tagged'])} tagged"
                st.caption(line)
                if rng:
                    st.caption(f"📚 Library range {format_mmss(rng[0])}–{format_mmss(rng[1])}")
                if stats["top_tags"]:
                    st.markdown(" ".join(f"`{t}` {n}" for t, n in stats["top_tags"]))
            with cols[2]:
                if st.button("🎚️", key=f"lib_toggle_{vid}", help="Set the library time range for this video"):
                    # One picker open at a time — each embeds a full video player.
                    st.session_state["lib_open_picker"] = None if st.session_state["lib_open_picker"] == vid else vid
                    st.rerun()

            if st.session_state["lib_open_picker"] == vid:
                st.caption("The Compilation Planner uses this range by default. "
                           "A range set inside the planner overrides it for that project only.")
                action = render_range_picker(
                    vid, video["source_path"], video["duration"], rng,
                    key="lib",
                    suggestions=[("✂️ Trim intro/outro", lambda v=video: trim_intro_outro_range(v))],
                    apply_label="Save library range",
                    clear_label="Remove library range",
                    slider_label=f"Library range for {vid}",
                )
                if action:
                    kind, new_rng = action
                    full = new_rng is not None and new_rng[0] <= 0 and new_rng[1] >= video["duration"] - 0.5
                    set_library_range(vid, None if (kind == "clear" or full) else new_rng)
                    st.session_state.pop(f"lib_slider_{vid}", None)
                    st.session_state["lib_open_picker"] = None
                    st.rerun()


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def _audio_signature() -> tuple:
    sig = []
    for f in sorted(AUDIO_DIR.glob("*.json")):
        if f.name == "audio_index.json":
            continue
        try:
            sig.append((f.name, f.stat().st_mtime))
        except OSError:
            continue
    return tuple(sig)


@st.cache_data(max_entries=2)
def load_audio_library(signature: tuple) -> dict:
    """track_id -> slim summary: duration, BPM, energy shape (downsampled for
    the mini chart) and the same variance / hits-per-minute the planner uses
    to match tracks to videos."""
    tracks = {}
    for name, mtime in signature:
        try:
            t = json.loads((AUDIO_DIR / name).read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(t, dict) or "duration_sec" not in t:
            continue
        hires = t.get("hires") or {}
        if hires.get("rms"):
            rms = [float(x) for x in hires["rms"]]
            onset = [float(x) for x in hires.get("onset", [])]
        else:
            rms = [float(e.get("energy", 0.0)) for e in t.get("energy_envelope", [])]
            onset = []
        duration = float(t["duration_sec"])
        mean = sum(rms) / len(rms) if rms else 0.0
        variance = (sum((x - mean) ** 2 for x in rms) / len(rms)) ** 0.5 if rms else 0.0
        hit_rate = sum(1 for x in onset if x >= 0.8) / max(duration / 60.0, 0.1) if onset else 0.0
        step = max(1, len(rms) // 200)
        track_id = t.get("track_id", Path(name).stem)
        tracks[track_id] = {
            "track_id": track_id,
            "source_path": t.get("source_path", ""),
            "duration": duration,
            "bpm": float(t["bpm"]) if t.get("bpm") else None,
            "beats": len(t.get("beat_times") or []),
            "variance": variance,
            "hit_rate": hit_rate,
            "energy": [round(sum(rms[i:i + step]) / len(rms[i:i + step]), 4)
                       for i in range(0, len(rms), step)],
            "added": mtime,
            "extra": int((t.get("analysis") or {}).get("version", 0)) >= EXTRA_ANALYSIS_VERSION,
            "extra_info": _extra_summary(t.get("analysis")),
        }
    return tracks


EXTRA_ANALYSIS_VERSION = 2   # same as audio_analysis.ANALYSIS_VERSION (kept here so this page loads without scipy)
AUDIO_LIBRARY_FOLDER = getattr(config, "AUDIO_LIBRARY_DRIVE_FOLDER", "Audio-Library")


def _extra_summary(a: dict) -> str:
    if not a:
        return ""
    bits = [f"{len(a.get('sections') or [])} sections",
            f"{len(a.get('downbeats') or [])} bars"]
    if a.get("drops"):
        bits.append(f"{len(a['drops'])} drop(s)")
    if a.get("vocals") is not None:
        bits.append("vocals")
    return ", ".join(bits)


def _audio_analysis_module():
    """audio_analysis needs scipy — import lazily so the page works without it."""
    try:
        import audio_analysis
        return audio_analysis, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600, show_spinner=False)
def _drive_audio_files(folder: str) -> tuple:
    from drive_sync import list_drive_audio
    return list_drive_audio(folder)


@st.fragment(run_every=4)
def _analysis_progress(aa) -> None:
    status = aa.read_status()
    if not status:
        return
    running = aa.worker_running()
    active = {k: v for k, v in status.items() if v.get("state") in ("queued", "running")}
    if active and not running:
        st.warning("Analysis jobs are waiting but the worker isn't running (the server may have restarted).")
        if st.button("▶️ Restart analysis", key="aa_restart"):
            aa.start_worker()
            st.rerun()
    icons = {"queued": "⏳", "running": "⚙️", "done": "✅", "error": "❌"}
    order = {"running": 0, "queued": 1, "error": 2, "done": 3}
    for tid, info in sorted(status.items(), key=lambda kv: (order.get(kv[1].get("state"), 9), kv[0].lower())):
        kind = "new track" if info.get("kind") == "new" else "extra analysis"
        st.caption(f"{icons.get(info.get('state'), '•')} **{tid}** — {kind}: {info.get('msg', '')}")
    if not active:
        if st.button("Clear finished", key="aa_clear"):
            aa.clear_finished_status()
            st.rerun()
    # Once everything is done, reload the page so new/updated tracks show.
    if st.session_state.get("_aa_was_active") and not active:
        st.session_state["_aa_was_active"] = False
        st.rerun(scope="app")
    st.session_state["_aa_was_active"] = bool(active)


def render_analysis_panel(tracks: dict) -> None:
    """Server-side analysis: new tracks in the Audio-Library, and the extra
    cut-finding analysis (bars, drums, sections, build-ups, vocals) for
    tracks that only have the original Colab analysis."""
    missing_extra = sorted(tid for tid, t in tracks.items() if not t["extra"])
    with st.expander(f"🔬 Audio analysis — {len(missing_extra)} track(s) without the extra analysis",
                     expanded=bool(missing_extra) or bool(st.session_state.get("_aa_was_active"))):
        aa, err = _audio_analysis_module()
        if aa is None:
            st.error(f"Audio analysis isn't available on this server: {err}. "
                     f"Install it with: `pip install scipy` (in the app's venv), then restart the app.")
            return
        st.caption("Runs on the server in the background (about a second or two per track, plus a few "
                   "minutes each for vocals when Demucs is installed) — you can leave this page.")
        extras = []
        extras.append("better downbeats: **beat_this** ✅" if aa.beat_this_available()
                      else "better downbeats: beat_this not installed (using an estimate)")
        extras.append("vocals: **Demucs** ✅" if aa.demucs_available() else "vocals: Demucs not installed (skipped)")
        st.caption(" · ".join(extras))

        cols = st.columns(2)
        with cols[0]:
            if st.button(f"➕ Add extra analysis to {len(missing_extra)} track(s)", key="aa_extra",
                         disabled=not missing_extra,
                         help="Keeps each track's existing beats and energy; adds bars, drums, "
                              "sections, phrases, build-ups/drops (and vocals if available)."):
                n = aa.enqueue([{"track_id": tid, "kind": "extra"} for tid in missing_extra])
                aa.start_worker()
                st.session_state["_aa_was_active"] = True
                st.toast(f"Queued {n} track(s).")
                st.rerun()
        with cols[1]:
            if st.button(f"🔍 Check {AUDIO_LIBRARY_FOLDER} for new tracks", key="aa_check",
                         help="Lists audio files in your Google Drive folder that have no catalogue yet."):
                _drive_audio_files.clear()
                st.session_state["aa_checked"] = True

        if st.session_state.get("aa_checked"):
            files, drive_err = _drive_audio_files(AUDIO_LIBRARY_FOLDER)
            if drive_err:
                st.error(f"Couldn't list {AUDIO_LIBRARY_FOLDER}: {drive_err}")
            else:
                known_names = {Path(t["source_path"]).name for t in tracks.values() if t["source_path"]}
                known_ids = set(tracks)
                new = [f for f in files if f["name"] not in known_names and Path(f["name"]).stem not in known_ids]
                if not new:
                    st.success(f"Every track in {AUDIO_LIBRARY_FOLDER} ({len(files)}) is already analysed.")
                else:
                    st.markdown(f"**{len(new)} new track(s)** in {AUDIO_LIBRARY_FOLDER}:")
                    st.caption("\n".join(f"• {f['rel']}" for f in new[:30])
                               + (f"\n… and {len(new) - 30} more" if len(new) > 30 else ""))
                    if st.button(f"🎵 Analyse {len(new)} new track(s)", type="primary", key="aa_new"):
                        n = aa.enqueue([{"track_id": Path(f["name"]).stem, "kind": "new", "drive_rel": f["rel"]}
                                        for f in new])
                        aa.start_worker()
                        st.session_state["_aa_was_active"] = True
                        st.session_state["aa_checked"] = False
                        st.toast(f"Queued {n} new track(s).")
                        st.rerun()

        if aa.read_status() or st.session_state.get("_aa_was_active"):
            _analysis_progress(aa)   # refreshes itself every few seconds


def render_audio():
    tracks = load_audio_library(_audio_signature())
    render_analysis_panel(tracks)

    st.sidebar.header("Filter")
    name_query = st.sidebar.text_input("Search name", placeholder="part of a track name", key="aud_name")
    max_minutes = max(1, int(max((t["duration"] for t in tracks.values()), default=600) // 60) + 1)
    duration_filter = st.sidebar.slider("Duration (minutes)", 0, max_minutes, (0, max_minutes), key="aud_dur")
    bpms = [t["bpm"] for t in tracks.values() if t["bpm"]]
    bpm_filter = None
    if bpms:
        lo, hi = int(min(bpms)), int(max(bpms)) + 1
        bpm_filter = st.sidebar.slider("BPM", lo, hi, (lo, hi), key="aud_bpm")

    st.sidebar.header("Sort")
    sorts = {
        "Name": lambda t: t["track_id"].lower(),
        "Duration": lambda t: t["duration"],
        "BPM": lambda t: t["bpm"] or 0.0,
        "Energy variation": lambda t: t["variance"],
        "Hits per minute": lambda t: t["hit_rate"],
        "Date added": lambda t: t["added"],
    }
    sort_by = st.sidebar.selectbox(
        "Sort by", list(sorts), key="aud_sort",
        help="Energy variation and hits per minute are what the planner matches videos on: "
             "high = dynamic, punchy tracks; low = steady, calm ones.",
    )
    descending = st.sidebar.toggle("Descending", value=(sort_by != "Name"), key="aud_desc")

    if not tracks:
        st.info("No audio catalogues yet — use **Check for new tracks** above, or sync from Google Drive.")
        return

    rows = []
    for t in tracks.values():
        if name_query and name_query.lower() not in t["track_id"].lower():
            continue
        if not (duration_filter[0] * 60 <= t["duration"] <= duration_filter[1] * 60):
            continue
        if bpm_filter and t["bpm"] and not (bpm_filter[0] <= t["bpm"] <= bpm_filter[1]):
            continue
        rows.append(t)
    rows.sort(key=sorts[sort_by], reverse=descending)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Tracks shown", f"{len(rows)} / {len(tracks)}")
    metric_cols[1].metric("Total length", format_mmss(sum(t["duration"] for t in rows)))
    if not rows:
        st.info("No tracks match these filters.")
        return

    n_pages = (len(rows) - 1) // PAGE_SIZE + 1
    page = st.number_input(f"Page (of {n_pages})", 1, n_pages, 1, key="aud_page") if n_pages > 1 else 1
    st.session_state.setdefault("aud_open_player", None)

    for t in rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        tid = t["track_id"]
        with st.container(border=True):
            cols = st.columns([6, 3, 1])
            with cols[0]:
                st.markdown(f"**{tid}**")
                info = [f"⏱ {format_mmss(t['duration'])}"]
                if t["bpm"]:
                    info.append(f"{t['bpm']:.0f} BPM")
                info.append(f"energy variation {t['variance']:.2f}")
                if t["hit_rate"]:
                    info.append(f"{t['hit_rate']:.1f} hits/min")
                info.append(f"🔬 {t['extra_info']}" if t["extra"] else "basic analysis only")
                st.caption("  ·  ".join(info))
            with cols[1]:
                if t["energy"]:
                    st.line_chart(t["energy"], height=70)
            with cols[2]:
                if st.button("▶️", key=f"aud_toggle_{tid}", help="Play this track"):
                    # One player open at a time.
                    st.session_state["aud_open_player"] = (
                        None if st.session_state["aud_open_player"] == tid else tid)
                    st.rerun()

            if st.session_state["aud_open_player"] == tid:
                _render_audio_player(tid, t["source_path"])


def _render_audio_player(track_id: str, source_path: str):
    from render_preview import colab_to_local
    local_path = colab_to_local(source_path)
    if Path(local_path).exists():
        st.audio(local_path)
        return
    st.warning(f"Audio file not found locally: {local_path}")
    try:
        from drive_sync import credentials_available, download_source_video
        can_download = credentials_available() and "MyDrive/" in source_path
    except ImportError:
        can_download = False
    if can_download and st.button("⬇️ Download this track from Google Drive", key=f"aud_dl_{track_id}"):
        bar = st.progress(0.0, text="Downloading from Google Drive...")
        err = download_source_video(source_path.split("MyDrive/", 1)[1], Path(local_path),
                                    progress_callback=lambda p: bar.progress(min(p, 1.0)))
        if err:
            st.error(f"Download failed: {err}")
        else:
            st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

if library_kind == "🎵 Audio":
    render_audio()
else:
    render_videos()
