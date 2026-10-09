"""
Media Library — two libraries, picked at the top of the page:
  Videos: search, filter and sort, and set each video's library time range
          (the part of the video the Compilation Planner should use).
  Audio:  search, filter and sort the music tracks, see their energy shape
          and play them; analyse new tracks (and add the extra cut-finding
          analysis to existing ones) on the server.
  Renders: the saved Render Preview MP4s — track, length, the videos and clips
          each one used, the planner settings behind it; play, rate, keep,
          download its plan, delete, and clean up old ones.
"""

import datetime
import json
import shutil
import time
from pathlib import Path

import streamlit as st

import config
from config import CATALOGUE_DIR, AUDIO_DIR, PREVIEW_DIR
from library_common import (
    scene_tags, tc_to_seconds, format_mmss, overlap_with_range,
    library_ranges, set_library_range, master_thumbnail_scenes,
    render_range_picker, show_render_player,
)

from page_setup import page_setup, drive_token

PAGE_SIZE = 20

page_setup("Media Library", "pages/3_Media_Library.py", title="📁 Media Library")
library_kind = st.radio("Library", ["🎬 Videos", "🎵 Audio", "🎞️ Renders"], horizontal=True,
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


def _scene_order(s: dict) -> tuple:
    """Time order — scene_id isn't chronological once the Reviewer has split a scene."""
    try:
        return (tc_to_seconds(s["start_tc"]), s["scene_id"])
    except (KeyError, ValueError):
        return (float("inf"), s.get("scene_id", 0))


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
        for s in sorted(cat["scenes"], key=_scene_order):
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
    vid_use, _ = render_usage()

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
        "Times rendered": lambda r: vid_use.get(r["video"]["video_id"], 0),
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
                if vid_use.get(vid):
                    line += f"  ·  🎞️ in {vid_use[vid]} render(s)"
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
            "sections": [{"start": x["start"], "end": x["end"], "label": x["label"]}
                         for x in (t.get("analysis") or {}).get("sections") or []],
            "drops": list((t.get("analysis") or {}).get("drops") or []),
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
    paused = aa.is_paused()
    active = {k: v for k, v in status.items() if v.get("state") in ("queued", "running")}
    queued_n = sum(1 for v in status.values() if v.get("state") == "queued")

    ctl = st.columns(4)
    if running and not paused:
        if ctl[0].button("⏸ Pause after this track", key="aa_pause",
                         help="Finish the track being analysed, then stop. The rest stay queued."):
            aa.pause()
            st.rerun()
        if ctl[1].button("⏹ Stop now", key="aa_stop",
                         help="Stop straight away and free the server. The current track goes back in the "
                              "queue and starts again from the beginning when you resume."):
            aa.stop_now()
            st.session_state["_aa_stop_at"] = time.time()
            st.rerun()
    elif running and paused:
        st.info("⏸ Stopping after the current track…")
        if ctl[1].button("⏹ Stop now", key="aa_stop2"):
            aa.stop_now()
            st.session_state["_aa_stop_at"] = time.time()
            st.rerun()
        if time.time() - st.session_state.get("_aa_stop_at", time.time()) > 15:
            if ctl[2].button("⚠️ Force stop", key="aa_force",
                             help="The worker hasn't stopped yet (a long step like vocal separation "
                                  "can't be interrupted) — end it immediately."):
                aa.stop_now(force=True)
                st.rerun()
    elif queued_n:
        st.info(f"⏸ Paused — {queued_n} track(s) waiting." if paused else
                f"{queued_n} track(s) waiting, but the analysis isn't running (the server may have restarted).")
        if ctl[0].button("▶️ Resume", key="aa_resume", type="primary"):
            aa.resume()
            st.rerun()
    if queued_n and ctl[3].button("🗑 Cancel waiting", key="aa_cancel",
                                  help="Remove every track that hasn't started yet from the queue."):
        aa.cancel_queued()
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

        targets = missing_extra
        cols = st.columns(2)
        with cols[0]:
            if st.button(f"➕ Add extra analysis to {len(targets)} track(s)", key="aa_extra",
                         disabled=not targets,
                         help="Keeps each track's existing beats and energy; adds bars, drums, "
                              "sections, phrases, build-ups/drops (and vocals if available)."):
                n = aa.enqueue([{"track_id": tid, "kind": "extra"} for tid in targets])
                aa.resume()
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
                    st.markdown(f"**{len(new)} new track(s)** in {AUDIO_LIBRARY_FOLDER}")
                    st.caption("\n".join(f"• {f['rel']}" for f in new[:30])
                               + (f"\n… and {len(new) - 30} more" if len(new) > 30 else ""))
                    chosen = new
                    if st.button(f"🎵 Analyse {len(chosen)} new track(s)", type="primary", key="aa_new"):
                        n = aa.enqueue([{"track_id": Path(f["name"]).stem, "kind": "new", "drive_rel": f["rel"]}
                                        for f in chosen])
                        aa.resume()
                        st.session_state["_aa_was_active"] = True
                        st.session_state["aa_checked"] = False
                        st.toast(f"Queued {n} new track(s).")
                        st.rerun()

        if aa.read_status() or st.session_state.get("_aa_was_active"):
            _analysis_progress(aa)   # refreshes itself every few seconds


def render_audio():
    tracks = load_audio_library(_audio_signature())
    _, track_use = render_usage()
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
        "Times rendered": lambda t: track_use.get(t["track_id"], 0),
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
    st.session_state.setdefault("aud_open_detail", None)

    for t in rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        tid = t["track_id"]
        with st.container(border=True):
            cols = st.columns([6, 3, 1, 1])
            with cols[0]:
                st.markdown(f"**{tid}**")
                info = [f"⏱ {format_mmss(t['duration'])}"]
                if t["bpm"]:
                    info.append(f"{t['bpm']:.0f} BPM")
                info.append(f"energy variation {t['variance']:.2f}")
                if t["hit_rate"]:
                    info.append(f"{t['hit_rate']:.1f} hits/min")
                info.append(f"🔬 {t['extra_info']}" if t["extra"] else "basic analysis only")
                if track_use.get(tid):
                    info.append(f"🎞️ in {track_use[tid]} render(s)")
                st.caption("  ·  ".join(info))
            with cols[1]:
                if t["energy"] and t["extra"]:
                    from audio_charts import mini_figure
                    st.plotly_chart(mini_figure(t["energy"], t["duration"], t["sections"], t["drops"]),
                                    use_container_width=True, config={"displayModeBar": False},
                                    key=f"aud_mini_{tid}")
                elif t["energy"]:
                    st.line_chart(t["energy"], height=70)
            with cols[3]:
                if st.button("🔬", key=f"aud_detail_{tid}", disabled=not t["extra"],
                             help="Show the extra analysis: sections, build-ups, drops, bar lines, drums, "
                                  "chord changes and vocals" if t["extra"]
                             else "Add the extra analysis (panel above) to see it here"):
                    st.session_state["aud_open_detail"] = (
                        None if st.session_state["aud_open_detail"] == tid else tid)
                    st.rerun()
            with cols[2]:
                if st.button("▶️", key=f"aud_toggle_{tid}", help="Play this track"):
                    # One player open at a time.
                    st.session_state["aud_open_player"] = (
                        None if st.session_state["aud_open_player"] == tid else tid)
                    st.rerun()

            if st.session_state["aud_open_detail"] == tid:
                _render_analysis_detail(tid)
            if st.session_state["aud_open_player"] == tid:
                _render_audio_player(tid, t["source_path"])


@st.cache_data(max_entries=3, show_spinner=False)
def _load_track(track_id: str, mtime: float) -> dict:
    return json.loads((AUDIO_DIR / f"{track_id}.json").read_text())


def _render_analysis_detail(track_id: str) -> None:
    from audio_charts import track_figure
    path = AUDIO_DIR / f"{track_id}.json"
    try:
        track = _load_track(track_id, path.stat().st_mtime)
    except (OSError, ValueError) as e:
        st.error(f"Couldn't read this track: {e}")
        return
    a = track.get("analysis") or {}
    st.plotly_chart(track_figure(track), use_container_width=True, key=f"aud_detail_chart_{track_id}")
    secs = a.get("sections") or []
    bits = []
    if secs:
        bits.append("Sections: " + " → ".join(f"{x['label']} ({format_mmss(x['start'])})" for x in secs)
                    + " — the same letter means they sound alike.")
    if a.get("drops"):
        bits.append("Drops at " + ", ".join(format_mmss(d) for d in a["drops"]) + " (red ▼); build-ups shaded yellow.")
    m = a.get("methods") or {}
    bits.append(f"Bar lines: {'beat_this model' if m.get('downbeats') == 'beat_this' else 'estimated'} · "
                f"vocals: {'Demucs' if m.get('vocals') else 'not analysed'} · "
                "click Kick, Snare, Hi-hat, Chord change, Section novelty, Vocals or Bar lines in the legend to show them.")
    st.caption("  \n".join(bits))


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
# Renders
# ---------------------------------------------------------------------------

def _renders_signature() -> tuple:
    """(name, mtime, size) of every render MP4 and sidecar — reloads on any change."""
    if not PREVIEW_DIR.exists():
        return ()
    sig = []
    for f in sorted([*PREVIEW_DIR.glob("*.mp4"), *PREVIEW_DIR.glob("*.json")]):
        try:
            st_ = f.stat()
            sig.append((f.name, st_.st_mtime, st_.st_size))
        except OSError:
            continue
    return tuple(sig)


@st.cache_data(max_entries=2, show_spinner="Reading renders…")
def load_renders(signature: tuple) -> list:
    from renders import list_renders
    return list_renders()


def render_usage() -> tuple:
    """(videos, tracks) Counters — how many saved renders used each one."""
    from renders import usage_counts
    return usage_counts(load_renders(_renders_signature()))


def _when(iso: str) -> str:
    try:
        return datetime.datetime.fromisoformat(iso).strftime("%d %b %Y %H:%M")
    except (TypeError, ValueError):
        return "?"


def _age_days(iso: str) -> float:
    try:
        then = datetime.datetime.fromisoformat(iso)
        return (datetime.datetime.now(then.tzinfo) - then).total_seconds() / 86400
    except (TypeError, ValueError):
        return 0.0


STARS = ["–", "★", "★★", "★★★", "★★★★", "★★★★★"]


def render_renders():
    from renders import (format_size, clips_outside_ranges, delete_render, make_thumbnail,
                         update_user_fields, reopen_state, on_drive)
    renders = load_renders(_renders_signature())
    ranges = library_ranges()
    vid_use, track_use = render_usage()

    st.sidebar.header("Filter")
    name_query = st.sidebar.text_input("Search track", placeholder="part of a track name", key="ren_name")
    video_filter = st.sidebar.multiselect("Uses video", sorted(vid_use, key=str.lower), key="ren_videos",
                                          help="Show renders that use any of these videos.")
    min_rating = st.sidebar.selectbox("Rating at least", STARS, key="ren_rating")
    kept_only = st.sidebar.toggle("Kept only", key="ren_kept")

    st.sidebar.header("Sort")
    sorts = {
        "Newest": lambda r: r.get("created") or "",
        "Track": lambda r: ((r.get("track") or {}).get("track_id") or "").lower(),
        "Length": lambda r: r.get("duration_sec") or 0,
        "Size": lambda r: r.get("size_bytes") or 0,
        "Rating": lambda r: r["user"]["rating"],
        "Videos used": lambda r: len(r.get("videos") or []),
    }
    sort_by = st.sidebar.selectbox("Sort by", list(sorts), key="ren_sort")
    descending = st.sidebar.toggle("Descending", value=(sort_by != "Track"), key="ren_desc")

    if not renders:
        st.info("No renders yet — use **🎬 Render Preview** in the Compilation Planner.")
        return

    # Storage
    total_size = sum(r.get("size_bytes") or 0 for r in renders if r.get("local"))
    try:
        disk = shutil.disk_usage(PREVIEW_DIR)
        st.progress(min(disk.used / disk.total, 1.0),
                    text=f"💾 Server disk: {format_size(disk.used)} of {format_size(disk.total)} used "
                         f"({format_size(disk.free)} free) — renders on the server take {format_size(total_size)}")
    except OSError:
        pass

    _render_drive_panel(renders)
    _render_usage_panel(renders, vid_use, track_use)
    _render_cleanup_panel(renders, delete_render, format_size)

    rows = []
    wanted_rating = STARS.index(min_rating)
    for r in renders:
        tid = (r.get("track") or {}).get("track_id") or ""
        if name_query and name_query.lower() not in tid.lower():
            continue
        if video_filter and not {v["video_id"] for v in r.get("videos") or []} & set(video_filter):
            continue
        if r["user"]["rating"] < wanted_rating:
            continue
        if kept_only and not r["user"]["keep"]:
            continue
        rows.append(r)
    rows.sort(key=sorts[sort_by], reverse=descending)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Renders shown", f"{len(rows)} / {len(renders)}")
    metric_cols[1].metric("Total length", format_mmss(sum(r.get("duration_sec") or 0 for r in rows)))
    metric_cols[2].metric("Size", format_size(sum(r.get("size_bytes") or 0 for r in rows)))
    if not rows:
        st.info("No renders match these filters.")
        return

    by_name = {Path(r["path"]).name: r for r in renders}
    st.session_state.setdefault("ren_compare", [])
    st.session_state["ren_compare"] = [n for n in st.session_state["ren_compare"] if n in by_name]
    _render_compare_panel(by_name, st.session_state["ren_compare"], ranges)

    n_pages = (len(rows) - 1) // PAGE_SIZE + 1
    page = st.number_input(f"Page (of {n_pages})", 1, n_pages, 1, key="ren_page") if n_pages > 1 else 1
    st.session_state.setdefault("ren_open_player", None)
    st.session_state.setdefault("ren_open_detail", None)
    st.session_state.setdefault("ren_confirm_delete", None)

    for r in rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        path = Path(r["path"])
        name = path.name
        track = r.get("track") or {}
        user = r["user"]
        with st.container(border=True):
            cols = st.columns([2, 6, 1, 1, 1, 1, 1])
            with cols[0]:
                thumb = make_thumbnail(path)
                if thumb:
                    st.image(str(thumb), width=220)
            with cols[1]:
                title = f"**{track.get('track_id') or 'Unknown track'}**"
                if user["keep"]:
                    title += "  📌"
                if user["rating"]:
                    title += f"  {STARS[user['rating']]}"
                st.markdown(title)
                info = [f"🗓 {_when(r.get('created'))}", f"⏱ {format_mmss(r.get('duration_sec') or 0)}"]
                if r.get("width"):
                    info.append(f"{r['width']}×{r['height']}")
                info.append(format_size(r.get("size_bytes")))
                if on_drive(r):
                    info.append("☁️ on Drive" + ("" if r.get("local") else " only (cleared from server)"))
                else:
                    info.append("💾 server only")
                if r.get("segments") is not None:
                    info.append(f"{r['segments']} segments ({r.get('split_segments') or 0} split-screen)")
                    info.append(f"{r.get('clip_count') or 0} clips from {len(r.get('videos') or [])} videos")
                if r.get("render_seconds"):
                    info.append(f"rendered in {format_mmss(r['render_seconds'])}")
                mode = (r.get("settings") or {}).get("matching_mode")
                if mode:
                    info.append(mode)
                st.caption("  ·  ".join(info))
                if r.get("videos"):
                    st.markdown(" ".join(f"`{v['video_id']}` ×{v['clips']}" for v in r["videos"][:12])
                                + (f" … +{len(r['videos']) - 12} more" if len(r["videos"]) > 12 else ""))
                if r.get("backfilled"):
                    st.caption("ℹ️ Made before render details were saved — the videos it used aren't known.")
                stale = clips_outside_ranges(r, ranges)
                if stale:
                    st.caption(f"⚠️ {len(stale)} clip(s) are now outside their video's library range — "
                               f"re-rendering this track would pick different footage.")
                if user["notes"]:
                    st.caption(f"📝 {user['notes']}")
            with cols[2]:
                if st.button("▶️", key=f"ren_play_{name}", help="Play this render"):
                    st.session_state["ren_open_player"] = None if st.session_state["ren_open_player"] == name else name
                    st.rerun()
            with cols[3]:
                if st.button("ℹ️", key=f"ren_detail_{name}",
                             help="Clip list, settings, rating, notes, keep, and the plan to download"):
                    st.session_state["ren_open_detail"] = None if st.session_state["ren_open_detail"] == name else name
                    st.rerun()
            with cols[4]:
                comparing = name in st.session_state["ren_compare"]
                if st.button("✅" if comparing else "⚖️", key=f"ren_cmp_{name}",
                             help="Remove from the comparison" if comparing
                             else "Compare side by side — pick two renders"):
                    _toggle_compare(name)
                    st.rerun()
            with cols[5]:
                if st.button("↩️", key=f"ren_reopen_{name}", disabled=reopen_state(r)[0] is None,
                             help="Open the Compilation Planner with this render's settings and matches"
                             if r.get("project") else
                             "Open the Compilation Planner with this render's settings (matches are rebuilt)"
                             if r.get("plan") else "No plan was saved with this render"):
                    _reopen_in_planner(r)
            with cols[6]:
                if st.button("🗑️", key=f"ren_del_{name}", disabled=user["keep"],
                             help="Kept — unkeep it (ℹ️) to delete" if user["keep"] else "Delete this render"):
                    st.session_state["ren_confirm_delete"] = name
                    st.rerun()

            if st.session_state["ren_confirm_delete"] == name:
                c = st.columns([4, 1, 1])
                c[0].warning(f"Delete {name}?" + (" It's also moved to the Google Drive bin (recoverable for "
                                                  "30 days)." if on_drive(r) else ""))
                if c[1].button("Delete", key=f"ren_del_yes_{name}", type="primary"):
                    err = delete_render(path, drive_token())
                    if err:
                        st.error(err)
                    else:
                        st.session_state["ren_confirm_delete"] = None
                        st.toast(f"Deleted {name}")
                        st.rerun()
                if c[2].button("Cancel", key=f"ren_del_no_{name}"):
                    st.session_state["ren_confirm_delete"] = None
                    st.rerun()
            if st.session_state["ren_open_player"] == name:
                show_render_player(r, height=420)
            if st.session_state["ren_open_detail"] == name:
                _render_render_detail(r, update_user_fields)


def _toggle_compare(name: str) -> None:
    """Keep at most two renders picked; picking a third drops the oldest pick."""
    picked = list(st.session_state.get("ren_compare", []))
    if name in picked:
        picked.remove(name)
    else:
        picked = (picked + [name])[-2:]
    st.session_state["ren_compare"] = picked


def _reopen_in_planner(r: dict) -> None:
    from renders import reopen_state
    data, kind = reopen_state(r)
    if data is None:
        st.warning("This render has no saved plan, so it can't be reopened.")
        return
    track_id = data.get("track_id")
    if track_id and not (AUDIO_DIR / f"{track_id}.json").exists():
        st.warning(f"Track **{track_id}** isn't in the audio library any more — the planner will "
                   f"open on its first track instead.")
    st.session_state["_reopen_project"] = {"data": data, "kind": kind, "label": Path(r["path"]).name}
    st.switch_page("pages/1_Compilation_Planner.py")


def _fmt_setting(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v)) if v else "(none)"
    return str(v)


def _render_compare_panel(by_name: dict, picked: list, ranges: dict) -> None:
    from renders import compare_settings, compare_videos, format_size
    if not picked:
        return
    with st.container(border=True):
        head = st.columns([6, 1])
        if len(picked) < 2:
            head[0].markdown(f"**⚖️ Compare** — picked **{picked[0]}**. Pick one more with ⚖️.")
        else:
            head[0].markdown("**⚖️ Side-by-side compare**")
        if head[1].button("Clear", key="ren_cmp_clear"):
            st.session_state["ren_compare"] = []
            st.rerun()
        if len(picked) < 2:
            return
        a, b = by_name[picked[0]], by_name[picked[1]]
        info_cols = st.columns(2)
        for col, r, label in ((info_cols[0], a, "Left (A)"), (info_cols[1], b, "Right (B)")):
            with col:
                track = (r.get("track") or {}).get("track_id") or "Unknown track"
                st.markdown(f"**{label}: {track}**  \n{Path(r['path']).name}")
                bits = [f"⏱ {format_mmss(r.get('duration_sec') or 0)}"]
                if r.get("clip_count") is not None:
                    bits.append(f"{r['segments']} segments · {r['clip_count']} clips · {len(r.get('videos') or [])} videos")
                if r["user"]["rating"]:
                    bits.append(STARS[r["user"]["rating"]])
                st.caption("  ·  ".join(bits))
                if r["user"]["notes"]:
                    st.caption(f"📝 {r['user']['notes']}")
                if st.button("↩️ Reopen this one in Planner", key=f"ren_cmp_reopen_{label[-2]}",
                             disabled=not (r.get("project") or r.get("plan"))):
                    _reopen_in_planner(r)

        same_track = (a.get("track") or {}).get("track_id") == (b.get("track") or {}).get("track_id")
        audio_from = "A"
        if not same_track:
            audio_from = st.radio("Sound from", ["A", "B"], horizontal=True, key="ren_cmp_audio",
                                  format_func=lambda x: f"{x} ({'left' if x == 'A' else 'right'})",
                                  help="These are different tracks, so pick whose music to hear.")
        from renders import make_compare_video
        with st.spinner("Building the side-by-side video (only needed once per pair)…"):
            cmp_path, cmp_err = make_compare_video(a, b, audio_from)
        if cmp_err:
            st.error(f"Couldn't build the side-by-side video: {cmp_err}")
        else:
            st.video(str(cmp_path))
            st.caption("Both renders in one video — they play, pause and seek together. "
                       "Left is A, right is B" + ("; the music is the same track." if same_track else "."))

        if (a.get("track") or {}).get("track_id") != (b.get("track") or {}).get("track_id"):
            st.caption("ℹ️ These are renders of different tracks.")
        diff = compare_settings(a, b)
        vids = compare_videos(a, b)
        c = st.columns(2)
        with c[0]:
            st.markdown("**Settings that differ**")
            if diff:
                st.dataframe([{"Setting": k, "A": _fmt_setting(va), "B": _fmt_setting(vb)} for k, va, vb in diff],
                             hide_index=True, use_container_width=True)
            elif a.get("settings") or b.get("settings"):
                st.caption("Same planner settings — the difference is in the picks (seed, swaps, manual choices).")
            else:
                st.caption("Settings weren't saved for these renders.")
        with c[1]:
            st.markdown("**Footage per video (seconds)**")
            if vids:
                only_a = sum(1 for _, x, y in vids if x and not y)
                only_b = sum(1 for _, x, y in vids if y and not x)
                st.dataframe([{"Video": v, "A": round(x, 1), "B": round(y, 1)} for v, x, y in vids],
                             hide_index=True, use_container_width=True, height=min(320, 38 + 35 * len(vids)))
                st.caption(f"{len(vids) - only_a - only_b} video(s) in both · {only_a} only in A · {only_b} only in B")
            else:
                st.caption("Videos weren't recorded for these renders.")


def _render_render_detail(r: dict, update_user_fields) -> None:
    path = Path(r["path"])
    name = path.name
    user = r["user"]

    c = st.columns([2, 1, 4])
    rating = c[0].radio("Rating", STARS, index=user["rating"], horizontal=True, key=f"ren_rate_{name}")
    keep = c[1].toggle("📌 Keep", value=user["keep"], key=f"ren_keep_{name}",
                       help="Kept renders can't be deleted, and Clean up skips them.")
    notes = c[2].text_input("Notes", value=user["notes"], key=f"ren_notes_{name}",
                            placeholder="e.g. good energy in the drop, slow intro")
    new = {"rating": STARS.index(rating), "keep": keep, "notes": notes.strip()}
    if new != {"rating": user["rating"], "keep": user["keep"], "notes": user["notes"]}:
        update_user_fields(path, **new)
        from renders import on_drive, sync_sidecar
        if on_drive(r) and drive_token():
            sync_sidecar(path, drive_token())   # keep the Drive backup of these details current
        st.rerun()

    plan = r.get("plan")
    if not plan:
        st.caption("No plan was saved with this render, so there's no clip list.")
        return

    clip_rows = []
    for i, entry in enumerate(plan.get("timeline") or [], 1):
        t0 = (entry.get("track_time") or [0, 0])[0]
        slots = entry.get("scenes") or []
        for slot_i, slot in enumerate(slots, 1):
            for link in slot.get("chain") or []:
                clip_rows.append({
                    "Segment": i,
                    "At": format_mmss(t0),
                    "Cell": f"{slot_i}/{len(slots)}" if len(slots) > 1 else "",
                    "Video": link.get("video_id"),
                    "From": format_mmss(link.get("clip_start_sec") or 0),
                    "Length (s)": round(float(link.get("clip_duration_sec") or 0), 2),
                })
    st.markdown(f"**Clips** ({len(clip_rows)})")
    st.dataframe(clip_rows, hide_index=True, use_container_width=True, height=min(400, 38 + 35 * len(clip_rows)))

    per_video = [{"Video": v["video_id"], "Clips": v["clips"], "Seconds": round(v["seconds"], 1)}
                 for v in sorted(r.get("videos") or [], key=lambda v: -v["seconds"])]
    c = st.columns(2)
    with c[0]:
        st.markdown("**Footage per video**")
        st.dataframe(per_video, hide_index=True, use_container_width=True)
    with c[1]:
        st.markdown("**Planner settings**")
        st.json(r.get("settings") or {}, expanded=False)
        src = r.get("sources") or {}
        if src:
            st.caption(f"Sources: {src.get('local', 0)} on the server, {src.get('streamed', 0)} streamed, "
                       f"{src.get('downloaded', 0)} downloaded" + (f" · app v{r['app_version']}" if r.get("app_version") else ""))
    st.download_button("⬇️ Download this render's plan (JSON)", data=json.dumps(plan, indent=2),
                       file_name=f"{path.stem}_plan.json", mime="application/json", key=f"ren_plan_{name}",
                       help="The exact plan this render was made from — import it with the Blender add-on.")


def _render_drive_panel(renders: list) -> None:
    from renders import (on_drive, upload_render, clear_local, auto_clear_enabled, set_auto_clear,
                         format_size)
    for err in st.session_state.pop("ren_flash_errors", None) or []:
        st.error(err)
    not_uploaded = [r for r in renders if r.get("local") and not on_drive(r)]
    clearable = [r for r in renders if r.get("local") and on_drive(r)]
    drive_only = [r for r in renders if not r.get("local") and on_drive(r)]
    token = drive_token()
    title = (f"☁️ Google Drive — {len(drive_only) + len(clearable)} on Drive, "
             f"{len(not_uploaded)} only on the server")
    with st.expander(title, expanded=bool(not_uploaded or clearable)):
        st.caption("Renders are saved to **scene-labeling/previews** in your Google Drive as they're made "
                   "(when Drive is connected). Once there, the server copy can go — playback, thumbnails "
                   "and comparisons then read them from Drive.")
        auto = st.toggle("Clear renders from the server automatically once they're on Drive",
                         value=auto_clear_enabled(), key="ren_auto_clear")
        if auto != auto_clear_enabled():
            set_auto_clear(auto)
        c = st.columns(2)
        with c[0]:
            size = sum(r.get("size_bytes") or 0 for r in not_uploaded)
            if st.button(f"☁️ Upload {len(not_uploaded)} render(s) to Drive ({format_size(size)})",
                         key="ren_upload_all", disabled=not (not_uploaded and token),
                         help=None if token else "Connect Google Drive in the sidebar first."):
                errors = []
                bar = st.progress(0.0, text="Uploading…")
                for i, r in enumerate(not_uploaded):
                    name = Path(r["path"]).name
                    bar.progress(i / len(not_uploaded), text=f"Uploading {name} ({i + 1}/{len(not_uploaded)})…")
                    err = upload_render(Path(r["path"]), token)
                    if err:
                        errors.append(f"{name}: {err}")
                    elif auto:
                        clear_local(Path(r["path"]))
                bar.empty()
                if errors:
                    st.session_state["ren_flash_errors"] = errors[:10]
                st.toast(f"Uploaded {len(not_uploaded) - len(errors)} render(s).")
                st.rerun()
            if not_uploaded and not token:
                st.caption("Connect Google Drive (sidebar) to upload.")
        with c[1]:
            size = sum(r.get("size_bytes") or 0 for r in clearable)
            if st.button(f"🧹 Clear {len(clearable)} server cop{'y' if len(clearable) == 1 else 'ies'} "
                         f"({format_size(size)})", key="ren_clear_local", disabled=not clearable,
                         help="Deletes the server's copy of renders that are safely on Google Drive."):
                errors = [f"{Path(r['path']).name}: {e}" for r in clearable
                          if (e := clear_local(Path(r["path"])))]
                if errors:
                    st.session_state["ren_flash_errors"] = errors[:10]
                st.toast(f"Freed {format_size(size)} on the server.")
                st.rerun()


def _render_usage_panel(renders: list, vid_use, track_use) -> None:
    with st.expander("📊 Usage across renders"):
        known = [r for r in renders if not r.get("backfilled")]
        if not known:
            st.caption("No renders with saved details yet — new renders will show here.")
            return
        st.caption(f"From {len(known)} render(s) with saved details"
                   + (f" ({len(renders) - len(known)} older ones don't record their videos)."
                      if len(known) < len(renders) else "."))
        c = st.columns(2)
        with c[0]:
            st.markdown("**Most-used videos**")
            st.dataframe([{"Video": v, "Renders": n} for v, n in vid_use.most_common(15)],
                         hide_index=True, use_container_width=True)
        with c[1]:
            st.markdown("**Most-rendered tracks**")
            st.dataframe([{"Track": t, "Renders": n} for t, n in track_use.most_common(15)],
                         hide_index=True, use_container_width=True)
        all_videos = set(load_library(_catalogue_signature()))
        unused = sorted(all_videos - set(vid_use), key=str.lower)
        if all_videos:
            st.caption(f"🆕 {len(unused)} of {len(all_videos)} library videos haven't been in a render yet"
                       + (": " + ", ".join(unused[:20]) + (" …" if len(unused) > 20 else "") if unused else "."))


def _render_cleanup_panel(renders: list, delete_render, format_size) -> None:
    with st.expander("🧹 Clean up old renders"):
        c = st.columns(2)
        days = c[0].number_input("Older than (days)", 1, 3650, 30, key="ren_clean_days")
        unrated_only = c[1].toggle("Only unrated ones", value=True, key="ren_clean_unrated")
        targets = [r for r in renders
                   if not r["user"]["keep"]
                   and (not unrated_only or not r["user"]["rating"])
                   and _age_days(r.get("created")) > days]
        st.caption("📌 Kept renders are never included.")
        if not targets:
            st.caption("Nothing to clean up with these settings.")
            return
        size = sum(r.get("size_bytes") or 0 for r in targets)
        st.markdown(f"**{len(targets)} render(s)**, {format_size(size)}: "
                    + ", ".join(Path(r["path"]).name for r in targets[:10])
                    + (" …" if len(targets) > 10 else ""))
        sure = st.checkbox(f"Yes, delete these {len(targets)} render(s)", key="ren_clean_sure")
        if any(r.get("drive") for r in targets):
            st.caption("Renders on Google Drive are moved to the Drive bin too (recoverable for 30 days)"
                       + ("." if drive_token() else " — connect Google Drive to delete those."))
        if st.button("🗑️ Delete them", disabled=not sure, key="ren_clean_go"):
            errors, done = [], 0
            token = drive_token()
            for r in targets:
                err = delete_render(Path(r["path"]), token)
                if err:
                    errors.append(f"{Path(r['path']).name}: {err}")
                else:
                    done += 1
            st.session_state["ren_clean_sure"] = False
            st.toast(f"Deleted {done} render(s).")
            if errors:
                st.session_state["ren_flash_errors"] = errors[:10]
            st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

if library_kind == "🎵 Audio":
    render_audio()
elif library_kind == "🎞️ Renders":
    render_renders()
else:
    render_videos()
