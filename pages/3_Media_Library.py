"""
Media Library — two libraries, picked at the top of the page:
  Videos: search, filter and sort, and set each video's library time range
          (the part of the video the Compilation Planner should use).
  Audio:  search, filter and sort the music tracks, see their energy shape
          and play them; analyse new tracks (and add the extra cut-finding
          analysis to existing ones) on the server.
  Analytics: 📚 Library — progress (split pass, review, roles, actors), actors,
          role footage vs what your tracks need, clip lengths vs blocks
          (library_analytics.py); ⭐ Render ratings — what your rated renders say
          about the planner settings behind them (read-only).
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
    safe_image, image_ok, video_details, all_actors, all_video_tags, matches_details,
    scene_tags, tc_to_seconds, format_mmss, overlap_with_range,
    library_ranges, set_library_range, master_thumbnail_scenes,
    render_range_picker, show_render_player, file_download_button,
)

from page_setup import page_setup, drive_token
import catalogue_store as CS

PAGE_SIZE = 20

page_setup("Media Library", "pages/3_Media_Library.py", title="📁 Media Library")
library_kind = st.radio("Library", ["🎬 Videos", "🎵 Audio", "🎞️ Renders", "📈 Analytics"], horizontal=True,
                        key="lib_kind", label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_library() -> dict:
    """video_id -> slim summary of its catalogue (no motion curves) — the shared
    index in catalogue_store, refreshed file by file when catalogues change."""
    return {v["video_id"]: v for v in CS.video_index().values()}


def trim_excluded_range(video: dict) -> tuple:
    """Range that drops the excluded scenes at the start and end of the video
    (intro/outro flags are ignored — only Exclude decides). The whole video if
    neither end is excluded."""
    scenes = video["scenes"]
    if not scenes:
        return 0.0, video["duration"]
    first, last = 0, len(scenes) - 1
    while first <= last and scenes[first]["excluded"]:
        first += 1
    while last >= first and scenes[last]["excluded"]:
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

ADDED_CHOICES = {          # "Added" filter: key -> (label, days back; None = any time / own date)
    "any": ("Any time", None),
    "new": (f"🆕 Newly added (last {CS.NEW_VIDEO_DAYS} days)", CS.NEW_VIDEO_DAYS),
    "today": ("Today", 0),
    "3d": ("Last 3 days", 3),
    "30d": ("Last 30 days", 30),
    "since": ("Since a date…", None),
}


def _show_new_videos():
    """🆕 banner button: just the new videos, newest first."""
    st.session_state["lib_added"] = "new"
    st.session_state["lib_sort"] = "Date added"
    st.session_state["lib_desc"] = True
    st.session_state["lib_desc_for"] = "Date added"
    st.session_state["lib_name_query"] = ""


def render_videos():
    library = load_library()
    ranges = library_ranges()
    masters = master_thumbnail_scenes()
    vid_use, _ = render_usage()
    added = CS.added_times()
    new_ids = sorted((vid for vid, ts in added.items() if CS.is_new_video(ts)), key=lambda v: -added[v])

    st.sidebar.header("Filter")
    name_query = st.sidebar.text_input("Search name", placeholder="part of a video name", key="lib_name_query")
    added_choice = st.sidebar.selectbox(
        "Added", list(ADDED_CHOICES), key="lib_added",
        format_func=lambda k: ADDED_CHOICES[k][0] + (f" · {len(new_ids)}" if k == "new" else ""),
        help="When the video was added — the date its catalogue first appeared in Google Drive.",
    )
    added_since = None
    if added_choice == "since":
        since_day = st.sidebar.date_input("Added on or after",
                                          value=datetime.date.today() - datetime.timedelta(days=7),
                                          key="lib_added_since", format="DD/MM/YYYY")
        added_since = datetime.datetime.combine(since_day, datetime.time.min).timestamp()
    elif ADDED_CHOICES[added_choice][1] is not None:
        days = ADDED_CHOICES[added_choice][1]
        added_since = datetime.datetime.combine(
            datetime.date.today() - datetime.timedelta(days=days), datetime.time.min).timestamp()
    details = video_details()
    actor_filter = st.sidebar.multiselect("👤 Actors", all_actors(), key="lib_actors", placeholder="anyone",
                                          help="Set on each video in the Reviewer (🎬 Video details).")
    vtag_filter = st.sidebar.multiselect("🏷️ Video tags", all_video_tags(), key="lib_vtags", placeholder="any",
                                         help="Tags for the whole video, set in the Reviewer (🎬 Video details).")
    details_all = st.sidebar.toggle("Must have all of these", key="lib_details_all",
                                    disabled=not (actor_filter or vtag_filter),
                                    help="Off: any of the picked actors / video tags. On: every one of them.")
    all_tags = sorted({t for v in library.values() for s in v["scenes"] for t in s["tags"]}, key=str.lower)
    tag_filter = st.sidebar.multiselect("Clip tags", all_tags)
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
        "Date added": lambda r: added.get(r["video"]["video_id"], 0),
        "Times rendered": lambda r: vid_use.get(r["video"]["video_id"], 0),
    }
    sort_options = list(SORTS) if tag_filter else [k for k in SORTS if k != "Tagged footage"]
    if st.session_state.get("lib_sort") not in sort_options:
        # First visit (or the tag filter that allowed "Tagged footage" was cleared):
        # newest first while an "Added" filter is on, else the first option.
        st.session_state["lib_sort"] = "Date added" if added_choice != "any" else sort_options[0]
    sort_by = st.sidebar.selectbox("Sort by", sort_options, key="lib_sort",
                                   help="Tagged footage = seconds of usable footage whose scenes match the tag filter. "
                                        "Date added = when the video's catalogue first appeared in Google Drive.")
    if st.session_state.get("lib_desc_for") != sort_by:
        st.session_state["lib_desc"] = sort_by != "Name"   # sensible default per sort, still switchable
        st.session_state["lib_desc_for"] = sort_by
    descending = st.sidebar.toggle("Descending", key="lib_desc")

    # ---------------------------------------------------------------------------
    # Apply filters
    # ---------------------------------------------------------------------------
    wanted = {t.lower() for t in tag_filter}
    rows = []
    for vid, video in library.items():
        if name_query and name_query.lower() not in vid.lower():
            continue
        if not matches_details(vid, details, actor_filter, vtag_filter, details_all):
            continue
        if added_since is not None and added.get(vid, 0) < added_since:
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

    if new_ids and added_choice == "any":
        _nc = st.columns([5, 2])
        _names = ", ".join(new_ids[:4]) + ("…" if len(new_ids) > 4 else "")
        _nc[0].info(f"🆕 **{len(new_ids)} video(s) added in the last {CS.NEW_VIDEO_DAYS} days:** {_names}")
        _nc[1].button("🆕 Show new videos", on_click=_show_new_videos, use_container_width=True)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Videos shown", f"{len(rows)} / {len(library)}")
    metric_cols[1].metric("Usable footage", format_mmss(sum(r["stats"]["usable"] for r in rows)))
    if tag_filter:
        metric_cols[2].metric("Tagged footage", format_mmss(sum(r["stats"]["tagged"] for r in rows)))
    else:
        metric_cols[2].metric("With library range", sum(1 for r in rows if r["range"]))

    if not rows:
        if added_choice != "any":
            st.info("No videos match these filters in that period — try a longer one under **Added** in the "
                    "sidebar. (Dates come from Google Drive: press ☁️ Sync if you've just added some.)")
        else:
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
                        safe_image(str(thumb_path), width=220)
            with cols[1]:
                _added_ts = added.get(vid)
                _badge = "  🆕" if _added_ts and CS.is_new_video(_added_ts) else ""
                st.markdown(f"**{vid}**{_badge}")
                line = (f"⏱ {format_mmss(video['duration'])}  ·  usable {format_mmss(stats['usable'])}"
                        f"  ·  {len(video['scenes'])} scenes  ·  {stats['reviewed_pct']:.0f}% reviewed")
                if tag_filter:
                    line += f"  ·  🏷️ {format_mmss(stats['tagged'])} tagged"
                if vid_use.get(vid):
                    line += f"  ·  🎞️ in {vid_use[vid]} render(s)"
                if _added_ts:
                    line += f"  ·  🗓 added {CS.added_ago(_added_ts)}"
                st.caption(line)
                if rng:
                    st.caption(f"📚 Library range {format_mmss(rng[0])}–{format_mmss(rng[1])}")
                _d = details.get(vid) or {}
                if _d.get("actors") or _d.get("tags"):
                    st.markdown(("👤 " + ", ".join(f"**{a}**" for a in _d.get("actors", [])) if _d.get("actors") else "")
                                + ("  ·  " if _d.get("actors") and _d.get("tags") else "")
                                + ("🏷️ " + " ".join(f"`{x}`" for x in _d.get("tags", [])) if _d.get("tags") else ""))
                if stats["top_tags"]:
                    st.caption("Clip tags: " + " ".join(f"`{t}` {n}" for t, n in stats["top_tags"]))
            with cols[2]:
                try:
                    st.page_link("pages/2_Reviewer.py", label="📝 Review",
                                 query_params={"video": video.get("stem", vid)},
                                 help="Open this video in the Reviewer (tags, roles, splits, thumbnails).")
                except Exception:   # page list unavailable (e.g. run on its own) — a plain link instead
                    st.markdown(f"[📝 Review](Reviewer?video={video.get('stem', vid)})")
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
                    suggestions=[("✂️ Trim excluded ends", lambda v=video: trim_excluded_range(v))],
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
            "bars_model": ((t.get("analysis") or {}).get("methods") or {}).get("downbeats") == "beat_this",
            "has_vocals": (t.get("analysis") or {}).get("vocals") is not None,
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
        kind = "new track" if info.get("kind") == "new" else "analysis"
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

        _render_upgrade(aa, tracks)

        if aa.read_status() or st.session_state.get("_aa_was_active"):
            _analysis_progress(aa)   # refreshes itself every few seconds


def _render_upgrade(aa, tracks: dict) -> None:
    """⬆️ Re-run the extra analysis on tracks that already have it, but with
    estimated bar lines (no beat_this at the time) or without vocals (no
    Demucs). Beats and energy are kept, so existing plans still line up."""
    done = {tid: t for tid, t in tracks.items() if t["extra"]}
    if not done:
        return
    have_bt, have_dm = aa.beat_this_available(), aa.demucs_available()
    no_bars = sorted(tid for tid, t in done.items() if not t["bars_model"])
    no_vocals = sorted(tid for tid, t in done.items() if not t["has_vocals"])
    st.divider()
    st.markdown(f"**⬆️ Upgrade already-analysed tracks** — {len(no_bars)} with estimated bar lines, "
                f"{len(no_vocals)} without vocals (of {len(done)})")
    if not (have_bt or have_dm):
        st.caption("Needs beat_this and/or Demucs installed on the server — see the server commands in "
                   "the setup notes, then restart the app.")
        return
    if not no_bars and not no_vocals:
        st.caption("Every analysed track already has the enhanced analysis. ✅")
        return
    st.caption("Keeps each track's beats and energy (plans still line up) and redoes bar lines, sections, "
               "phrases and drops with beat_this — about 30–60 s a track on the server. Vocals use Demucs: "
               "a few minutes a track. Runs in the background; you can leave the page and pause any time.")
    c = st.columns(2)
    bars_only = no_bars if have_bt else []
    if c[0].button(f"🥁 Bar lines only — {len(bars_only)} track(s) (~{max(1, round(len(bars_only) * 0.75))} min)",
                   key="aa_up_bars", disabled=not bars_only, use_container_width=True,
                   help="Quick: beat_this bar lines and everything built on them. Vocals are kept as they are."):
        n = aa.enqueue([{"track_id": tid, "kind": "extra", "vocals": False} for tid in bars_only])
        aa.resume()
        st.session_state["_aa_was_active"] = True
        st.toast(f"Queued {n} track(s).")
        st.rerun()
    full = sorted(set(no_bars if have_bt else []) | set(no_vocals if have_dm else []))
    if c[1].button(f"🎤 Bar lines + vocals — {len(full)} track(s) (~{max(1, len(full) * 4)} min)",
                   key="aa_up_full", disabled=not full or not have_dm, use_container_width=True,
                   help="Everything, including Demucs vocal separation (slow on the server's processor). "
                        + ("" if have_dm else "Demucs isn't installed on the server.")):
        n = aa.enqueue([{"track_id": tid, "kind": "extra", "vocals": True} for tid in full])
        aa.resume()
        st.session_state["_aa_was_active"] = True
        st.toast(f"Queued {n} track(s).")
        st.rerun()
    if not have_bt:
        st.caption("beat_this isn't installed on the server, so bar lines can't be upgraded here.")


def render_audio():
    if st.session_state.get(SM_EDIT):
        render_story_editor()
        return
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
    import story_map as STM
    _maps = STM.saved_maps()
    story_filter = st.sidebar.selectbox("🎭 Story map", ["Any", "Has one", "Not yet"], key="aud_story",
                                        help="Tracks with / without a saved story map (which parts of the "
                                             "song suit 🎬 Intro, 🌙 Slow, 💥 Action, 🏁 Outro…).")

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
        if (story_filter == "Has one" and t["track_id"] not in _maps) or \
                (story_filter == "Not yet" and t["track_id"] in _maps):
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
                _story_strip(tid, _maps.get(tid), t["duration"])
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

            sc = st.columns([1, 1, 3])
            sc[0].button("🎭 Edit story map" if tid in _maps else "🎭 Make a story map", key=f"aud_story_edit_{tid}",
                         on_click=_open_story_editor, args=(tid,), use_container_width=True,
                         help="Edit this track's story maps here — the default (used by new projects, the "
                              "recommender and the analytics) and any named ones.")
            if tid not in _maps:
                sc[1].button("✨ Suggest one", key=f"aud_story_new_{tid}", on_click=_suggest_story_map,
                             args=(tid,), use_container_width=True, disabled=not t["extra"],
                             help="Saves the suggestion straight away as the track's default map: which parts of "
                                  "the song suit 🎬 Intro, 🌙 Slow, 💥 Action and 🏁 Outro."
                             if t["extra"] else "Needs the extra analysis (panel above) first.")
            _sm_msg = st.session_state.pop(f"aud_story_msg_{tid}", None)
            if _sm_msg is True:
                sc[2].success("✨ Story map suggested — 🎭 Edit story map to check and adjust it.")
            elif _sm_msg:
                sc[2].error(_sm_msg)

            if st.session_state["aud_open_detail"] == tid:
                _render_analysis_detail(tid)
            if st.session_state["aud_open_player"] == tid:
                _render_audio_player(tid, t["source_path"])


def _story_strip(track_id: str, regions, duration: float) -> None:
    """A thin coloured bar of the track's story map parts (or a note that it has none)."""
    if not regions:
        st.caption("🎭 No story map yet")
        return
    import scene_roles as SR
    roles = SR.roles_by_id()
    total = max(duration, regions[-1]["end"], 1e-6)
    bits, totals = [], {}
    for r in regions:
        role = roles.get(r["role"]) if r["role"] else None
        w = 100 * (r["end"] - r["start"]) / total
        colour = role["colour"] if role else "rgba(128,128,128,.18)"
        tip = f"{SR.label(r['role'], roles) if r['role'] else 'no role'} {format_mmss(r['start'])}–{format_mmss(r['end'])}"
        bits.append(f'<div title="{tip}" style="width:{w:.3f}%;background:{colour};height:100%"></div>')
        if r["role"]:
            totals[r["role"]] = totals.get(r["role"], 0.0) + r["end"] - r["start"]
    st.markdown('<div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin:.2rem 0 .3rem">'
                + "".join(bits) + "</div>", unsafe_allow_html=True)
    st.caption("🎭 " + "  ·  ".join(f"{SR.label(r, roles)} {format_mmss(v)}" for r, v in totals.items()))


def _suggest_story_map(track_id: str) -> None:
    """✨ Suggest: the same suggestion as the planner's Story map step, lined up with
    the Smart method's default cuts, saved as the track's story map."""
    import story_map as STM
    path = AUDIO_DIR / f"{track_id}.json"
    try:
        track = _load_track(track_id, path.stat().st_mtime)
    except (OSError, ValueError) as e:
        st.session_state[f"aud_story_msg_{track_id}"] = f"Couldn't read this track: {e}"
        return
    STM.save_map(track_id, STM.suggested_map(track))
    st.session_state[f"aud_story_msg_{track_id}"] = True


# ---------------------------------------------------------------------------
# 🎭 Story map editor (full page) — the track's saved maps, edited here
# ---------------------------------------------------------------------------
SM_EDIT = "aud_sm_edit"   # {"track_id", "target": "default" | "named:<name>", "regions", "v", "dirty"}


def _sm_state() -> dict:
    return st.session_state.get(SM_EDIT) or {}


def _sm_load(track_id: str, target: str) -> None:
    import story_map as STM
    maps = STM.track_maps(track_id)
    regions = maps["default"] if target == "default" else maps["named"].get(target[6:])
    old = _sm_state()
    st.session_state[SM_EDIT] = {"track_id": track_id, "target": target, "regions": regions or None,
                                 "v": int(old.get("v", 0)) + 1, "dirty": False}
    for k in ("aud_sm_play", "aud_sm_focus"):
        st.session_state.pop(k, None)


def _open_story_editor(track_id: str) -> None:
    _sm_load(track_id, "default")


def _close_story_editor() -> None:
    for k in (SM_EDIT, "aud_sm_play", "aud_sm_focus", "aud_sm_msg"):
        st.session_state.pop(k, None)


def _sm_set(regions: list) -> None:
    sm = _sm_state()
    st.session_state[SM_EDIT] = {**sm, "regions": regions, "v": int(sm.get("v", 0)) + 1, "dirty": True}
    st.session_state.pop("aud_sm_play", None)


def _sm_target_name(target: str) -> str:
    return "the track's default map" if target == "default" else f"“{target[6:]}”"


def _sm_save() -> None:
    import story_map as STM
    sm = _sm_state()
    if sm["target"] == "default":
        STM.save_map(sm["track_id"], sm["regions"])
    else:
        STM.save_named(sm["track_id"], sm["target"][6:], sm["regions"])
    st.session_state[SM_EDIT] = {**sm, "dirty": False}
    st.session_state["aud_sm_msg"] = f"💾 Saved {_sm_target_name(sm['target'])}."


def _sm_save_as(name_key: str) -> None:
    import story_map as STM
    name = " ".join(str(st.session_state.get(name_key) or "").split())
    if not name:
        st.session_state["aud_sm_msg"] = "Give the new map a name first."
        return
    sm = _sm_state()
    STM.save_named(sm["track_id"], name, sm["regions"])
    st.session_state[SM_EDIT] = {**sm, "target": f"named:{name}", "dirty": False, "v": int(sm.get("v", 0)) + 1}
    st.session_state[name_key] = ""
    st.session_state["aud_sm_msg"] = f"📁 Saved as “{name}” — projects on this track can pick it."


def _sm_make_default() -> None:
    import story_map as STM
    sm = _sm_state()
    STM.save_map(sm["track_id"], sm["regions"])
    st.session_state["aud_sm_msg"] = f"⭐ {_sm_target_name(sm['target'])} is now the track's default map too."


def _sm_delete() -> None:
    import story_map as STM
    sm = _sm_state()
    STM.delete_named(sm["track_id"], sm["target"][6:])
    st.session_state["aud_sm_msg"] = f"🗑️ Deleted {_sm_target_name(sm['target'])}."
    _sm_load(sm["track_id"], "default")


@st.cache_data(max_entries=4, show_spinner="Reading the song's structure…")
def _sm_track_bits(track_id: str, mtime: float):
    """(track, 🧠 Smart default cuts, the suggestion's analysis) — the same cuts the ✨ suggestion lines up with."""
    import smart_cuts as SMC
    import story_map as STM
    track = _load_track(track_id, mtime)
    try:
        analysis = STM.analyse(track, 4)
    except Exception:          # a track the suggestion can't read — you can still mark it out by hand
        analysis = None
    return track, SMC.build_smart_segments(track, {"beats_per_bar": 4}), analysis


def _sm_play(track: dict, r: dict, lead: bool, key: str) -> None:
    """▶ on a part: cut that stretch of the song and play it (download the track first if it isn't here)."""
    from render_preview import colab_to_local
    import preview_snippets
    path = Path(colab_to_local(track.get("source_path", "")))
    if not path.exists():
        _render_audio_player(st.session_state[SM_EDIT]["track_id"], track.get("source_path", ""))
        return
    with st.spinner("Cutting the music…"):
        clip, err = preview_snippets.make_audio_snippet(PREVIEW_DIR / "snippets", path, r["start"],
                                                        r["end"] - r["start"], lead_in=2.0 if lead else 0.0)
    if err:
        st.error(f"Couldn't cut the music: {err}")
    else:
        st.audio(str(clip), autoplay=True)


def render_story_editor() -> None:
    """The full-page editor: replaces the track list until ← Back to tracks."""
    import story_map as STM
    import story_map_editor as SME
    sm = _sm_state()
    tid = sm["track_id"]
    try:
        track, segs, analysis = _sm_track_bits(tid, (AUDIO_DIR / f"{tid}.json").stat().st_mtime)
    except (OSError, ValueError) as e:
        st.error(f"Couldn't read {tid}: {e}")
        st.button("← Back to tracks", on_click=_close_story_editor, key="aud_sm_back_err")
        return
    dirty = sm.get("dirty")
    top = st.columns([1.8, 5])
    top[0].button("← Back to tracks" if not dirty else "← Back without saving", key="aud_sm_back",
                  use_container_width=True, on_click=_close_story_editor,
                  help="Unsaved changes are dropped." if dirty else None)
    top[1].subheader(f"🎭 Story map — {tid}")

    maps = STM.track_maps(tid)
    opts = ["default"] + [f"named:{n}" for n in sorted(maps["named"], key=str.lower)]
    if sm["target"] not in opts:
        opts.append(sm["target"])
    with st.container(border=True):
        c = st.columns([3, 4])
        pk = f"aud_sm_pick_{sm.get('v', 0)}"
        c[0].selectbox("🗺️ Editing", opts, index=opts.index(sm["target"]), key=pk, disabled=dirty,
                       format_func=lambda o: ("⭐ Track default" + ("" if maps["default"] else " (none yet)"))
                       if o == "default" else f"📁 {o[6:]}",
                       on_change=lambda: _sm_load(tid, st.session_state[pk]),
                       help="Save or undo your changes before switching maps." if dirty else
                            "The default is what new projects, the recommender and the analytics use. Named maps "
                            "are alternatives a project can pick.")
        with c[1]:
            msg = st.session_state.pop("aud_sm_msg", None)
            if msg:
                (st.warning if msg.startswith("Give") else st.success)(msg)
            if dirty:
                st.caption(f"✏️ Unsaved changes to {_sm_target_name(sm['target'])}")
                bc = st.columns(3)
                bc[0].button("💾 Save", key="aud_sm_save", type="primary", use_container_width=True,
                             on_click=_sm_save)
                with bc[1].popover("📁 Save as…", use_container_width=True):
                    nk = f"aud_sm_new_{sm.get('v', 0)}"
                    st.text_input("Name", key=nk, placeholder="e.g. Long intro")
                    st.button("📁 Save", key=f"{nk}_go", type="primary", on_click=_sm_save_as, args=(nk,))
                bc[2].button("↩️ Undo", key="aud_sm_undo", use_container_width=True,
                             on_click=_sm_load, args=(tid, sm["target"]))
            elif sm["target"] != "default" and sm.get("regions"):
                bc = st.columns(2)
                bc[0].button("⭐ Make this the default", key="aud_sm_mkdef", use_container_width=True,
                             on_click=_sm_make_default,
                             help="Copies this map over the track's default map.")
                bc[1].button("🗑️ Delete this map", key="aud_sm_del", use_container_width=True,
                             on_click=_sm_delete)
            elif sm.get("regions"):
                st.caption("Changes are only kept when you 💾 Save.")

    duration = float(track["duration_sec"])
    if not sm.get("regions"):
        SME.render_empty(duration, analysis, [sg["start"] for sg in segs] + [duration], set_regions=_sm_set,
                         ns="aud_sm", note="Lined up with the 🧠 Smart method's normal cuts.")
        return
    st.caption("Block lines are the 🧠 Smart method's normal cuts. A project with other audio settings keeps "
               "the parts where they are in seconds (📐 in the planner lines them up with its own cuts).")
    SME.render_editor(track, segs, ns="aud_sm", get_regions=lambda: _sm_state().get("regions"),
                      set_regions=_sm_set, version=int(sm.get("v", 0)), analysis=analysis,
                      play=lambda r, lead, key: _sm_play(track, r, lead, key))



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
    _msg = st.session_state.pop("ren_fetch_msg", None)
    if _msg:
        st.success(_msg)
    _drive_only = [r for r in rows if not r.get("local") and on_drive(r)]
    if _drive_only:
        _sz = sum(int((r.get("drive") or {}).get("size") or 0) for r in _drive_only)
        fc = st.columns([3, 2])
        fc[0].caption(f"☁️ {len(_drive_only)} of the renders shown are only on Google Drive — they stream from "
                      f"Drive, which can buffer.")
        if fc[1].button(f"💾 Copy those {len(_drive_only)} to the server ({format_size(_sz)})",
                        key="ren_fetch_all", use_container_width=True):
            _fetch_renders(_drive_only)

    by_name = {Path(r["path"]).name: r for r in renders}
    st.session_state.setdefault("ren_compare", [])
    st.session_state["ren_compare"] = [n for n in st.session_state["ren_compare"] if n in by_name]
    _render_compare_panel(by_name, st.session_state["ren_compare"], ranges)
    st.session_state.setdefault("ren_playlist", [])
    st.session_state["ren_playlist"] = [n for n in st.session_state["ren_playlist"] if n in by_name]
    _render_playlist_panel(by_name, rows)

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
            cols = st.columns([2, 6, 1, 1, 1, 1, 1, 1, 1])
            with cols[0]:
                thumb = make_thumbnail(path)
                if thumb:
                    safe_image(str(thumb), width=220)
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
            with cols[8]:
                if st.button("💾" if not r.get("local") and on_drive(r) else "⬇️", key=f"ren_dl_{name}",
                             help="Only on Google Drive — copy it back to the server for smooth playback, or "
                                  "download it" if not r.get("local") and on_drive(r)
                             else "Download it to this device"):
                    st.session_state["ren_open_download"] = (
                        None if st.session_state.get("ren_open_download") == name else name)
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
                in_list = name in st.session_state["ren_playlist"]
                if st.button("☑️" if in_list else "🎞️", key=f"ren_pl_{name}",
                             help=f"In the playlist (#{st.session_state['ren_playlist'].index(name) + 1}) — "
                                  "click to remove" if in_list else "Add to the playlist"):
                    _toggle_playlist(name)
                    st.rerun()
            with cols[6]:
                if st.button("↩️", key=f"ren_reopen_{name}", disabled=reopen_state(r)[0] is None,
                             help="Open the Compilation Planner with this render's settings and matches"
                             if r.get("project") else
                             "Open the Compilation Planner with this render's settings (matches are rebuilt)"
                             if r.get("plan") else "No plan was saved with this render"):
                    _reopen_in_planner(r)
            with cols[7]:
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
            if st.session_state.get("ren_open_download") == name:
                _render_download_row(r)
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


def _fetch_renders(metas: list) -> None:
    """💾 Copy Drive-only renders back onto the server, with progress."""
    from renders import fetch_to_server
    bar = st.progress(0.0, text=f"Copying {len(metas)} render(s) to the server…")
    errors = []
    for i, m in enumerate(metas, 1):
        err = fetch_to_server(m)
        if err:
            errors.append(f"{Path(m['path']).name}: {err}")
        bar.progress(i / len(metas), text=f"Copied {i} of {len(metas)}…")
    bar.empty()
    for e in errors[:5]:
        st.error(e)
    if not errors:
        st.session_state["ren_fetch_msg"] = f"💾 {len(metas)} render(s) are back on the server — they play smoothly now."
        st.rerun()


def _render_download_row(r: dict) -> None:
    """💾 Bring a Drive-only render back to the server; ⬇️ download to this device."""
    from renders import on_drive, format_size
    path = Path(r["path"])
    dc = st.columns([2, 2, 3])
    if path.exists():
        dc[0].success("💾 On the server")
        with dc[1]:
            file_download_button(path, f"⬇️ Download to this device ({format_size(path.stat().st_size)})",
                                 key=f"ren_dl_file_{path.name}", use_container_width=True)
        dc[2].caption("Plays from the server. Download to watch it on this device.")
    elif on_drive(r):
        size = (r.get("drive") or {}).get("size")
        if dc[0].button(f"💾 Copy to the server" + (f" ({format_size(int(size))})" if size else ""),
                        key=f"ren_fetch_{path.name}", type="primary", use_container_width=True,
                        help="Downloads it from Google Drive onto the server, so it plays from the server "
                             "instead of streaming from Drive."):
            _fetch_renders([r])
        fid = r["drive"]["file_id"]
        dc[1].link_button("⬇️ Download to this device", f"https://drive.google.com/uc?export=download&id={fid}",
                          use_container_width=True)
        dc[2].caption("Only on Google Drive (cleared from the server), so it streams from Drive. Copy it to the "
                      "server for smooth playback. The download link needs a browser signed in to your Google account.")


def _toggle_playlist(name: str) -> None:
    lst = list(st.session_state.get("ren_playlist", []))
    if name in lst:
        lst.remove(name)
    else:
        lst.append(name)
    st.session_state["ren_playlist"] = lst
    st.session_state.pop("ren_playlist_built", None)


def _playlist_move(i: int, step: int) -> None:
    lst = list(st.session_state.get("ren_playlist", []))
    j = i + step
    if 0 <= j < len(lst):
        lst[i], lst[j] = lst[j], lst[i]
        st.session_state["ren_playlist"] = lst
        st.session_state.pop("ren_playlist_built", None)


def _playlist_set(names: list) -> None:
    st.session_state["ren_playlist"] = list(names)
    st.session_state.pop("ren_playlist_built", None)


def _render_playlist_panel(by_name: dict, rows: list) -> None:
    """🎞️ Playlist: pick renders with 🎞️ on their cards (or add every one shown),
    order them, and play them back to back as one video with a chapter list."""
    import random as _random
    from renders import make_playlist_video
    names = st.session_state["ren_playlist"]
    shown = [Path(r["path"]).name for r in rows]
    with st.container(border=True):
        total = sum(float(by_name[n].get("duration_sec") or 0) for n in names)
        head = st.columns([4, 2, 1, 1, 1])
        head[0].markdown(f"**🎞️ Playlist** — {len(names)} render(s) · {format_mmss(total)}" if names else
                         "**🎞️ Playlist** — add renders with 🎞️ on their cards, or add every one shown")
        head[1].button(f"➕ Add all shown ({len(shown)})", key="ren_pl_addall", use_container_width=True,
                       on_click=_playlist_set, args=(names + [n for n in shown if n not in names],),
                       help="Adds every render matching the filters, in the current sort order.")
        head[2].button("🔀", key="ren_pl_shuffle", disabled=len(names) < 2, use_container_width=True,
                       on_click=_playlist_set, args=(_random.sample(names, len(names)),), help="Shuffle the order")
        head[3].button("Clear", key="ren_pl_clear", disabled=not names, use_container_width=True,
                       on_click=_playlist_set, args=([],))
        play = head[4].button("▶️ Play", key="ren_pl_play", type="primary", disabled=not names,
                              use_container_width=True, help="Plays the renders back to back as one video.")
        if not names:
            return
        with st.expander("Order", expanded=not st.session_state.get("ren_playlist_built")):
            for i, n in enumerate(names):
                m = by_name[n]
                rc = st.columns([6, 1, 1, 1])
                rc[0].markdown(f"{i + 1}. **{(m.get('track') or {}).get('track_id') or n}** · "
                               f"{format_mmss(m.get('duration_sec') or 0)}"
                               + ("" if m.get("local") else "  ☁️"))
                rc[1].button("⬆️", key=f"ren_pl_up_{n}", disabled=i == 0, on_click=_playlist_move, args=(i, -1))
                rc[2].button("⬇️", key=f"ren_pl_dn_{n}", disabled=i == len(names) - 1,
                             on_click=_playlist_move, args=(i, 1))
                rc[3].button("✖", key=f"ren_pl_rm_{n}", on_click=_toggle_playlist, args=(n,))

        built = st.session_state.get("ren_playlist_built")
        if play:
            note = st.empty()
            with st.spinner(f"Putting {len(names)} render(s) together…"):
                path, chapters, err = make_playlist_video([by_name[n] for n in names],
                                                          progress=lambda t: note.caption(t))
            note.empty()
            if err:
                st.error(f"Couldn't make the playlist: {err}")
                return
            built = {"path": str(path), "names": list(names), "start": 0,
                     "chapters": [(t, Path(m["path"]).name) for t, m in chapters]}
            st.session_state["ren_playlist_built"] = built
        if not built or built["names"] != names:
            return
        kw = {"start_time": int(built["start"])}
        import inspect
        if "autoplay" in inspect.signature(st.video).parameters:
            kw["autoplay"] = bool(built.get("autoplay"))
        st.video(built["path"], **kw)
        st.caption("Jump to:")
        ccols = st.columns(min(4, len(built["chapters"])))
        for i, (t, n) in enumerate(built["chapters"]):
            m = by_name.get(n, {})
            label = f"{i + 1}. {(m.get('track') or {}).get('track_id') or n} ({format_mmss(t)})"
            if ccols[i % len(ccols)].button(label, key=f"ren_pl_ch_{i}", use_container_width=True):
                built["start"], built["autoplay"] = t, True
                st.rerun()
        bc = st.columns([1, 1, 3])
        with bc[1]:
            file_download_button(built["path"], "⬇️ Download playlist", key="ren_pl_dl", use_container_width=True)
        if bc[0].button("⏹ Close player", key="ren_pl_close"):
            st.session_state.pop("ren_playlist_built", None)
            st.rerun()
        bc[1].caption("The renders play one after another as a single video. Closing the player frees the "
                      "memory it uses.")


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
    st.session_state["_reopen_project"] = {"data": data, "kind": kind, "label": Path(r["path"]).name,
                                           "project_name": r.get("project_name")}
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
        if on_drive(r):
            # Keep the Drive backup of these details current — now, or once Drive is connected.
            if not (drive_token() and sync_sidecar(path, drive_token()) is None):
                from library_common import mark_pending
                mark_pending(f"render:{name}")
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
        try:
            from drive_sync import credentials_available
            can_check = credentials_available()
        except Exception:
            can_check = False
        if can_check and st.button("🔄 Check Drive for renders", key="ren_restore",
                                   help="Brings back the details of any render that's on Google Drive but "
                                        "missing here, and marks renders already on Drive as uploaded. "
                                        "Also runs with every Drive sync."):
            from renders import restore_from_drive
            with st.spinner("Checking Google Drive…"):
                res = restore_from_drive()
            st.session_state["ren_flash_errors"] = res["errors"][:10]
            st.toast(f"Restored {res['restored']} render(s), matched {res['linked']} already on Drive.")
            st.rerun()
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
        all_videos = set(load_library())
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
# Analytics
# ---------------------------------------------------------------------------

_SERIES = "#2a78d6"   # single-series mark colour (reads on light and dark backgrounds)


def _chart_layout(fig, height: int, x_title: str, y_title: str = None):
    fig.update_layout(
        height=height, margin=dict(l=4, r=12, t=6, b=4), showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(title_text=x_title, title_font_size=11, tickfont_size=10,
                     gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(title_text=y_title, title_font_size=11, tickfont_size=10,
                     gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def _draw_chart(chart: dict, key: str) -> None:
    import plotly.graph_objects as go
    kind = chart.get("kind")
    if kind == "bars":
        items = chart.get("bars") or []
        if not items:
            return
        counting = chart.get("count")
        items = list(reversed(items))   # best at the top
        xs = [g["n"] if counting else (g["mean"] or 0) for g in items]
        labels = [f"{g['n']}" if counting else
                  (f"{g['mean']:.1f}★  ·  {g['rated']} rated" if g["mean"] else f"no ratings  ·  {g['n']} renders")
                  for g in items]
        fig = go.Figure(go.Bar(
            x=xs, y=[g["group"] for g in items], orientation="h", marker_color=_SERIES,
            text=labels, textposition="outside", cliponaxis=False, width=0.6,
            hovertemplate="%{y}: %{text}<extra></extra>",
        ))
        _chart_layout(fig, 40 + 30 * len(items), chart.get("x_title", ""))
        if not counting:
            fig.update_xaxes(range=[0, 5.9], dtick=1)
        else:
            fig.update_xaxes(range=[0, max(xs + [1]) * 1.25])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
    elif kind == "scatter":
        pts = chart.get("points") or []
        if not pts:
            st.caption("No data to plot yet.")
            return
        rating_axis = "y_title" not in chart
        fig = go.Figure(go.Scatter(
            x=[p["x"] for p in pts], y=[p["y"] for p in pts], mode="markers",
            marker=dict(size=10, color=_SERIES, opacity=0.85, line=dict(width=2, color="rgba(255,255,255,0.9)")),
            text=[p["name"] for p in pts],
            hovertemplate="%{text}<br>%{x:.1f} → %{y}" + ("★" if rating_axis else " min") + "<extra></extra>",
        ))
        _chart_layout(fig, 230, chart.get("x_title", ""), chart.get("y_title", "Rating (★)"))
        if rating_axis:
            fig.update_yaxes(range=[0.5, 5.5], dtick=1)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


# ---------------------------------------------------------------------------
# 📚 Library analytics (library_analytics.py does the sums)
# ---------------------------------------------------------------------------

def _lib_signature() -> tuple:
    """Changes whenever a catalogue, the library details (ranges, actors, ⭐), a track or a story map does."""
    files = []
    for f in [*CS.catalogue_paths(), CATALOGUE_DIR / "library_meta.json", CATALOGUE_DIR / "story_maps.json",
              CATALOGUE_DIR / "scene_roles.json", *AUDIO_DIR.glob("*.json")]:
        try:
            files.append((f.name, f.stat().st_mtime))
        except OSError:
            continue
    return tuple(sorted(files))


@st.cache_data(max_entries=4, show_spinner="Reading your tracks…")
def _track_needs_and_blocks(signature: tuple) -> tuple:
    """({track_id: {role: seconds its story map gives the role}}, [block lengths]) — each track's saved
    story map (else the one the wizard suggests) and its 🧠 Smart (normal pace) cuts, as the
    recommender's top three plans use."""
    import rec_library as RL
    from core.segmentation import build_segments
    from core.styles import style_settings
    needs, blocks = {}, []
    for tid in RL.list_tracks():
        try:
            track = RL.load_track(tid)
        except (OSError, ValueError):
            continue
        regions, _ = RL.story_for(tid, track)
        need = {}
        for r in regions or []:
            if r.get("role"):
                need[r["role"]] = need.get(r["role"], 0.0) + float(r["end"]) - float(r["start"])
        needs[tid] = need
        try:
            method, settings, _ = style_settings(track, "smart", 0)
            blocks += [sg["end"] - sg["start"] for sg in build_segments(track, {"method": method, "beats_per_bar": 4,
                                                                                 **settings})]
        except Exception:      # a track the cutter can't read — leave its blocks out
            pass
    return needs, blocks


@st.cache_data(max_entries=4, show_spinner="Working out the library's numbers…")
def _library_numbers(signature: tuple, renders_sig: tuple) -> dict:
    import library_analytics as LA
    import rec_library as RL
    import scene_roles as SR
    from render_analytics import to_row
    cats = CS.all_catalogues()
    ranges, details, masters = library_ranges(), video_details(), master_thumbnail_scenes()
    usage = RL.video_usage()
    ratings = {}
    for r in load_renders(renders_sig):
        row = to_row(r)
        if row["rating"]:
            for vid in set(row["videos"]):
                ratings.setdefault(vid, []).append(row["rating"])
    prog = LA.progress(cats, ranges, details, masters, SR.scene_roles, usage)
    needs, blocks = _track_needs_and_blocks(signature)
    profile = LA.length_profile(cats, ranges, blocks)
    return {
        "progress": prog,
        "actors": LA.actor_stats(cats, ranges, details, SR.scene_roles, usage, ratings),
        "roles": LA.role_balance(prog["rows"], needs, details),
        "role_next": {r: LA.role_candidates(cats, ranges, prog["rows"], r) for r in LA.HINT_ROLES},
        "lengths": profile,
        "mixed": LA.mixed_motion_clips(cats, ranges, CS.motion_curves),
        "tracks": len(needs),
    }


def _hbars(labels: list, series: list, x_title: str, height: int = None, stacked: bool = False,
           percent: bool = False):
    """Horizontal bars, one row per label; series = [(name, values, colour)]."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, vals, colour in series:
        fig.add_trace(go.Bar(y=labels, x=vals, name=name, orientation="h", marker_color=colour,
                             hovertemplate="%{y} · " + name + ": %{x:.0f}" + ("%" if percent else "") + "<extra></extra>"))
    fig.update_layout(barmode="stack" if stacked else "group", showlegend=len(series) > 1,
                      legend=dict(orientation="h", y=-0.25, x=0))
    _chart_layout(fig, height or (60 + 34 * len(labels)), x_title)
    fig.update_layout(showlegend=len(series) > 1)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


_SECOND = "#e8a317"   # second series (reads on light and dark backgrounds)
_CALM, _MED, _BUSY = "#76b7b2", "#9c9c9c", "#e15759"


def _mins(sec: float) -> str:
    return f"{sec / 60:.1f} min" if sec >= 90 else f"{sec:.0f}s"


def _lib_progress(nums: dict) -> None:
    p = nums["progress"]
    t = p["totals"]
    n = max(t["videos"], 1)
    m = st.columns(4)
    m[0].metric("✂️ Split pass", f"{t['split_done']}/{t['videos']}",
                help="Videos with no usable clip over 45s left (split, or never needed it). "
                     f"{t['split_started']} more are started (some long clips still to go).")
    m[1].metric("🎭 Role marking", f"{t['roles_started']}/{t['videos']}",
                help="Videos with at least one scene marked with a role.")
    m[2].metric("👤 Actors", f"{t['actors']}/{t['videos']}")
    m[3].metric("👀 Reviewed", f"{t['reviewed_pct']}%", help="Usable clips marked reviewed, across the library.")
    st.progress(t["split_done"] / n, text=f"✂️ Split pass — {t['split_done']} of {t['videos']} videos done"
                                         + (f", {t['split_started']} started" if t["split_started"] else ""))
    st.progress(t["roles_started"] / n, text=f"🎭 Role marking — {t['roles_started']} of {t['videos']} videos started")
    st.caption(f"{t['minutes']:.0f} minutes of usable footage · library ranges on {t['ranges']} videos · "
               f"⭐ picture chosen on {t['masters']}")
    c = st.columns(2)
    with c[0], st.container(border=True):
        st.markdown("**✂️ Split next** — most long clips (over 45s) left")
        if p["next_split"]:
            for r in p["next_split"]:
                st.markdown(f"- **{r['video_id']}** — {r['long_left']} long clip(s) · {r['minutes']:.0f} min")
        else:
            st.caption("No usable clip over 45s anywhere. ✅")
    with c[1], st.container(border=True):
        st.markdown("**🎭 Role-mark next** — the most footage, and the most used lately")
        if p["next_roles"]:
            for r in p["next_roles"]:
                st.markdown(f"- **{r['video_id']}** — {r['clips']} clips · {r['minutes']:.0f} min")
        else:
            st.caption("Every video has some roles marked. ✅")
    icon = {"done": "✅", "not needed": "✅", "started": "🟡", "to do": "⬜"}
    st.dataframe([{"Video": r["video_id"], "Clips": r["clips"], "Usable min": r["minutes"],
                   "✂️ Split": f"{icon[r['split']]} {r['split']}", "Long left": r["long_left"],
                   "Reviewed": r["reviewed_pct"], "Roles": r["roles_pct"],
                   "👤 Actors": ", ".join(r["actors"]) or "—", "Range": "✅" if r["has_range"] else "—",
                   "⭐": "✅" if r["has_master"] else "—"} for r in p["rows"]],
                 hide_index=True, width="stretch",
                 column_config={"Reviewed": st.column_config.ProgressColumn("Reviewed", format="%d%%", min_value=0,
                                                                            max_value=100),
                                "Roles": st.column_config.ProgressColumn("Roles", format="%d%%", min_value=0,
                                                                         max_value=100,
                                                                         help="Usable clips with a role")})


def _lib_actors(nums: dict) -> None:
    import library_analytics as LA
    import scene_roles as SR
    stats = nums["actors"]
    if not stats:
        st.info("No actors yet — add them in the Reviewer (🎬 Video details).")
        return
    for line in LA.actor_insights(stats):
        st.markdown(f"- {line}")
    st.markdown("**Usable footage per actor**")
    _hbars([a["actor"] for a in stats], [("Usable minutes", [a["minutes"] for a in stats], _SERIES)], "Minutes")
    st.markdown("**Motion profile** — share of each actor's footage in the library's calmest, middle and busiest "
                "third")
    _hbars([a["actor"] for a in stats], [("Calm", [a["calm_pct"] for a in stats], _CALM),
                                         ("Medium", [a["medium_pct"] for a in stats], _MED),
                                         ("Busy", [a["busy_pct"] for a in stats], _BUSY)],
           "% of footage", stacked=True, percent=True)
    roles = SR.roles_by_id()
    used_roles = sorted({r for a in stats for r in a["roles"]})
    st.dataframe([{"Actor": a["actor"], "Videos": a["n_videos"], "Usable min": a["minutes"], "Clips": a["clips"],
                   **{SR.label(r, roles): _mins(a["roles"].get(r, 0.0)) if a["roles"].get(r) else "—"
                      for r in used_roles},
                   "Recent uses": a["uses"],
                   "Avg render ★": f"{a['rating']:.1f} ({a['rated']})" if a["rating"] is not None else "—"}
                  for a in stats], hide_index=True, width="stretch")
    st.caption("A video with two actors counts for both. Recent uses = appearances in your recent saved projects "
               "and renders; ★ = average rating of rated renders that used the actor's videos.")


def _lib_roles(nums: dict) -> None:
    import scene_roles as SR
    roles = SR.roles_by_id()
    rows = [r for r in nums["roles"] if r["status"] != "not needed" or r["have_sec"]]
    if not rows:
        st.info("No story parts in your tracks yet, and no roles marked.")
        return
    st.caption(f"What's marked in the Reviewer vs what your {nums['tracks']} tracks' story maps ask for (the saved "
               "story map, else the one the wizard suggests). A typical track's need is its median.")
    _hbars([SR.label(r["role"], roles) for r in rows],
           [("Marked footage (min)", [r["have_sec"] / 60 for r in rows], _SERIES),
            ("A typical track needs (min)", [r["need_sec"] / 60 for r in rows], _SECOND)], "Minutes")
    status = {"none": "❌ none marked", "short": "⚠️ short", "ok": "✅", "not needed": "— no track uses it"}
    st.dataframe([{"Role": SR.label(r["role"], roles), "Marked": _mins(r["have_sec"]), "From videos": r["videos"],
                   "Tracks using it": r["tracks"], "Typical need": _mins(r["need_sec"]),
                   "Tracks' worth": "—" if r["tracks_worth"] is None else f"{r['tracks_worth']:.1f}",
                   "": status[r["status"]]} for r in rows], hide_index=True, width="stretch")
    st.caption("Tracks' worth = marked footage ÷ a typical track's need. Under 1.5 the same clips keep coming back "
               "in those parts — the planner and recommender fall back to unmarked clips there.")
    with st.expander("👤 Marked footage by actor"):
        for r in rows:
            if r["by_actor"]:
                st.markdown(f"**{SR.label(r['role'], roles)}** — " + " · ".join(f"{a} {_mins(s)}"
                                                                              for a, s in r["by_actor"].items()))
    st.markdown("**Where to find more** — videos with no roles yet, with the most footage the motion hints "
                "would suggest (the same 💡 hints quick marking shows)")
    c = st.columns(len(nums["role_next"]) or 1)
    for col, (role, items) in zip(c, nums["role_next"].items()):
        with col, st.container(border=True):
            st.markdown(f"**{SR.label(role, roles)}**")
            if not items:
                st.caption("Nothing left to suggest.")
            for it in items:
                st.markdown(f"- **{it['video_id']}** — {it['clips']} clip(s), {_mins(it['seconds'])}")


def _lib_lengths(nums: dict) -> None:
    import library_analytics as LA
    prof = nums["lengths"]
    for line in LA.length_insights(prof):
        st.markdown(f"- {line}")
    st.markdown("**Clip lengths vs block lengths**")
    st.caption(f"Share of your usable footage in clips of each length ({prof['clips']} clips), against the share of "
               f"blocks of that length when your {nums['tracks']} tracks are cut with 🧠 Smart ({prof['blocks']} "
               "blocks).")
    bins = prof["bins"]
    _hbars([b["label"] for b in bins], [("Footage (%)", [b["footage_pct"] for b in bins], _SERIES),
                                        ("Blocks (%)", [b["blocks_pct"] for b in bins], _SECOND)], "%", percent=True)
    st.markdown("**🎢 Long clips with the biggest motion swings** — calm and busy footage in one clip, so one role "
                "or one exclude switch can't fit all of it. The best ones to ✂️ split next.")
    if not nums["mixed"]:
        st.caption(f"No usable clip of {LA.MIXED_MIN_SEC:g}s or more with a motion curve.")
        return
    st.dataframe([{"Video": m["video_id"], "Clip": m["scene_id"], "Starts at": format_mmss(m["start"]),
                   "Length": f"{m['seconds']:.0f}s", "Swing": m["swing"]} for m in nums["mixed"]],
                 hide_index=True, width="stretch",
                 column_config={"Swing": st.column_config.NumberColumn(
                     "Swing", format="%.1f×", help="Busiest second minus calmest second, as a multiple of the "
                                                   "library's typical motion.")})


def render_library_analytics():
    nums = _library_numbers(_lib_signature(), _renders_signature())
    tabs = st.tabs(["📋 Progress", "👤 Actors", "🎭 Roles", "📏 Clip lengths"])
    with tabs[0]:
        _lib_progress(nums)
    with tabs[1]:
        _lib_actors(nums)
    with tabs[2]:
        _lib_roles(nums)
    with tabs[3]:
        _lib_lengths(nums)


def render_analytics_tab():
    which = st.radio("Analytics", ["📚 Library", "⭐ Render ratings"], horizontal=True, key="ana_kind",
                     label_visibility="collapsed")
    if which == "📚 Library":
        render_library_analytics()
    else:
        render_rating_analytics()


def render_rating_analytics():
    from render_analytics import build_cards
    renders = load_renders(_renders_signature())
    c = st.columns([2, 5])
    period = c[0].selectbox("Renders from", ["All time", "Last 30 days", "Last 7 days"], key="ana_period")
    c[1].caption("Theoretical for now — these insights come from your star ratings on renders and **don't "
                 "change the planner**. Rate renders in 🎞️ Renders (ℹ️ on each one) to sharpen them.")
    days = {"All time": None, "Last 30 days": 30, "Last 7 days": 7}[period]
    cards = build_cards(renders, days)
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, card in zip(cols, cards[i:i + 2]):
            with col, st.container(border=True):
                st.markdown(f"#### {card['title']}")
                st.caption(card["subtitle"])
                _draw_chart(card["chart"], key=f"ana_{card['key']}")
                if card.get("extra_chart"):
                    _draw_chart(card["extra_chart"], key=f"ana_{card['key']}_2")
                st.markdown(f"**Insight:** {card['insight']}")
                st.markdown(f"**Try:** {card['recommendation']}")
                if card.get("table"):
                    with st.expander("Data"):
                        st.dataframe(card["table"], hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

if library_kind == "🎵 Audio":
    render_audio()
elif library_kind == "🎞️ Renders":
    render_renders()
elif library_kind == "📈 Analytics":
    render_analytics_tab()
else:
    render_videos()
