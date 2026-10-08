"""
Helpers shared by the Media Library, Reviewer and Compilation Planner pages:
scene tags, time formatting, the library's per-video time ranges
(library_meta.json), the "edits not yet saved to Drive" list, and the
range-picker widget.
"""

import datetime
import json
import os
import subprocess
from pathlib import Path

import streamlit as st

import config
from config import CATALOGUE_DIR

# Small 480p copies of the source videos, used by the range picker so scrubbing
# is fast and the server doesn't load multi-GB originals into memory. Same
# timeline as the source (nothing trimmed), so range times match exactly.
# Set PROXY_DIR in config.py to put them elsewhere.
from auto_sync import PROXY_DIR  # noqa: E402  (one definition, shared with the sync)
PROXY_HEIGHT = 480

# Lives next to the video catalogues so it syncs with them (drive_sync.sync_pull
# pulls everything in scene-labeling/catalogue, and never overwrites a local copy
# that's newer than Drive's). Shape: {video_id: {"range": [start_sec, end_sec]}}
LIBRARY_META_FILE = CATALOGUE_DIR / "library_meta.json"
LIBRARY_META_STEM = LIBRARY_META_FILE.stem

# Files in CATALOGUE_DIR edited here but not yet uploaded to Drive, by stem
# (persisted so a page reload or restart doesn't lose track of them).
PENDING_FILE = CATALOGUE_DIR / ".pending_push.txt"


# ---------------------------------------------------------------------------
# Production vs test copy (set APP_ENV = "test" in the test copy's config.py)
# ---------------------------------------------------------------------------

IS_TEST = str(getattr(config, "APP_ENV", "production")).lower() == "test"


def page_title(name: str) -> str:
    """Browser-tab title, marked in the test copy so the two are easy to tell apart."""
    return f"🧪 TEST · {name}" if IS_TEST else name


def env_banner() -> None:
    """A bright strip at the top of every page of the test copy. Call right
    after st.set_page_config. Does nothing in production."""
    if IS_TEST:
        st.markdown(
            '<div style="background:#f59e0b;color:#111;padding:6px 12px;border-radius:6px;'
            'font-weight:600;margin-bottom:8px;">🧪 TEST COPY — separate data; saving to '
            'Google Drive is switched off. Production is unaffected.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tags and time
# ---------------------------------------------------------------------------

def scene_tags(scene: dict) -> list:
    """The tags to use for a scene: the Reviewer's corrected_tags when the scene
    has been corrected (even if the correction removed every tag), otherwise the
    auto-generated tags."""
    if "corrected_tags" in scene and scene["corrected_tags"] is not None:
        return list(scene["corrected_tags"])
    return list(scene.get("tags", []))


def tc_to_seconds(tc: str) -> float:
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_time(sec: float) -> datetime.time:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return datetime.time(hour=min(h, 23), minute=m, second=s)


def time_to_seconds(t: datetime.time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second


def format_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def overlap_with_range(scene_start: float, scene_end: float, time_range) -> tuple:
    """Intersect a scene's [start, end] with an optional (start, end) restriction.
    Returns (effective_start, effective_end) or (None, None) if no overlap."""
    if not time_range:
        return scene_start, scene_end
    r_start, r_end = time_range
    eff_start, eff_end = max(scene_start, r_start), min(scene_end, r_end)
    if eff_end <= eff_start:
        return None, None
    return eff_start, eff_end


# ---------------------------------------------------------------------------
# Library metadata (library_meta.json)
# ---------------------------------------------------------------------------

def load_library_meta() -> dict:
    """Small file, read fresh each time so every page sees the latest edits."""
    try:
        data = json.loads(LIBRARY_META_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def save_library_meta(meta: dict) -> None:
    """Write atomically and remember it still needs pushing to Drive."""
    LIBRARY_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIBRARY_META_FILE.with_name(LIBRARY_META_FILE.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.replace(tmp, LIBRARY_META_FILE)
    mark_pending(LIBRARY_META_STEM)


def library_ranges() -> dict:
    """{video_id: (start_sec, end_sec)} for every video with a library range."""
    out = {}
    for vid, entry in load_library_meta().items():
        rng = entry.get("range") if isinstance(entry, dict) else None
        try:
            if rng and len(rng) == 2 and float(rng[1]) > float(rng[0]):
                out[vid] = (float(rng[0]), float(rng[1]))
        except (TypeError, ValueError):
            continue
    return out


def set_library_range(video_id: str, time_range) -> None:
    """Set (start, end) as this video's library range, or None to remove it."""
    meta = load_library_meta()
    entry = meta.get(video_id) if isinstance(meta.get(video_id), dict) else {}
    if time_range is None:
        entry.pop("range", None)
    else:
        entry["range"] = [round(float(time_range[0]), 3), round(float(time_range[1]), 3)]
    if entry:
        meta[video_id] = entry
    else:
        meta.pop(video_id, None)
    save_library_meta(meta)


def master_thumbnail_scenes() -> dict:
    """{video_id: scene_id} for every video with a chosen master thumbnail."""
    out = {}
    for vid, entry in load_library_meta().items():
        sid = entry.get("thumbnail_scene") if isinstance(entry, dict) else None
        if isinstance(sid, int):
            out[vid] = sid
    return out


def set_master_thumbnail(video_id: str, scene_id) -> None:
    """Use this scene's thumbnail as the video's master thumbnail (None = clear)."""
    meta = load_library_meta()
    entry = meta.get(video_id) if isinstance(meta.get(video_id), dict) else {}
    if scene_id is None:
        entry.pop("thumbnail_scene", None)
    else:
        entry["thumbnail_scene"] = int(scene_id)
    if entry:
        meta[video_id] = entry
    else:
        meta.pop(video_id, None)
    save_library_meta(meta)


def scene_thumbnail_path(video_id: str, scene: dict):
    """Local path of a scene's (first) thumbnail image, or None if it isn't here.
    Always built from CATALOGUE_DIR — never trust the stored path, which was
    written by whichever environment processed the video."""
    paths = scene.get("thumbnail_paths") or []
    if not paths:
        return None
    p = CATALOGUE_DIR / "thumbnails" / video_id / Path(paths[0]).name
    return p if p.exists() else None


def master_thumbnail_path(video_id: str, scenes: list, masters: dict = None):
    """The chosen master thumbnail's image path for this video, or None."""
    sid = (masters if masters is not None else master_thumbnail_scenes()).get(video_id)
    if sid is None:
        return None
    scene = next((s for s in scenes if s.get("scene_id") == sid), None)
    return scene_thumbnail_path(video_id, scene) if scene else None


# ---------------------------------------------------------------------------
# Edits not yet pushed to Drive
# ---------------------------------------------------------------------------

def read_pending() -> set:
    try:
        return {l.strip() for l in PENDING_FILE.read_text().splitlines() if l.strip()}
    except Exception:
        return set()


def write_pending(stems: set) -> None:
    if stems:
        PENDING_FILE.write_text("\n".join(sorted(stems)))
    elif PENDING_FILE.exists():
        PENDING_FILE.unlink()


def mark_pending(stem: str) -> None:
    write_pending(read_pending() | {stem})


def push_pending(token: dict) -> dict:
    """Upload every pending file in CATALOGUE_DIR to scene-labeling/catalogue.
    Returns {stem: error} for any that failed (those stay pending)."""
    from drive_oauth import push_file_with_oauth
    failed = {}
    for stem in sorted(read_pending()):
        path = CATALOGUE_DIR / f"{stem}.json"
        if not path.exists():
            continue  # nothing to push any more
        err = push_file_with_oauth(token, path, "scene-labeling/catalogue")
        if err:
            failed[stem] = err
    write_pending(set(failed))
    return failed


# ---------------------------------------------------------------------------
# Fast preview copies (proxies)
# ---------------------------------------------------------------------------

def proxy_path(video_id: str) -> Path:
    return PROXY_DIR / f"{video_id}.mp4"


def make_proxy(source: Path, dest: Path, height: int = PROXY_HEIGHT) -> str | None:
    """Encode a small H.264 copy of source at dest (never upscales).
    Returns None on success, an error string on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".part.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"scale=-2:'min({height},ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-ac", "2",
        "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return "ffmpeg not found on PATH"
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return (proc.stderr or "ffmpeg failed").strip()[-500:]
    os.replace(tmp, dest)
    return None


# ---------------------------------------------------------------------------
# Drive sync status
# ---------------------------------------------------------------------------

def _ago(iso: str) -> tuple:
    """('3 min ago', age_in_minutes) for an ISO timestamp."""
    then = datetime.datetime.fromisoformat(iso)
    if then.tzinfo is None:
        then = then.replace(tzinfo=datetime.timezone.utc)
    mins = (datetime.datetime.now(datetime.timezone.utc) - then).total_seconds() / 60
    if mins < 1:
        text = "just now"
    elif mins < 60:
        text = f"{int(mins)} min ago"
    elif mins < 48 * 60:
        text = f"{mins / 60:.0f} h ago"
    else:
        text = f"{mins / 1440:.0f} days ago"
    return text, mins


def render_sync_status() -> None:
    """One sidebar line: when Drive was last synced, and whether it had errors."""
    from auto_sync import read_sync_status
    status = read_sync_status()
    if not status.get("finished_at"):
        return
    try:
        text, mins = _ago(status["finished_at"])
    except ValueError:
        return
    errors = status.get("errors") or []
    if errors:
        st.sidebar.warning(f"☁️ Drive sync {text} had {len(errors)} error(s): {errors[0]}")
    elif mins > 13 * 60:  # timer runs every 12 h
        st.sidebar.warning(f"☁️ Drive last synced {text} — is the sync timer running?")
    else:
        st.sidebar.caption(f"☁️ Drive synced {text}")


@st.cache_resource
def _sync_seen() -> dict:
    """Process-wide (shared by every session): the last sync change already
    reflected in the caches."""
    return {"changed_at": None}


def refresh_caches_after_sync(*cached_functions) -> None:
    """Clear these st.cache_data functions (all of st.cache_data if none are given)
    once after each sync that actually downloaded something, so the background
    timer's changes show up without anyone pressing a button."""
    from auto_sync import read_sync_status
    changed = read_sync_status().get("changed_at")
    seen = _sync_seen()
    if changed and changed != seen["changed_at"]:
        if cached_functions:
            for fn in cached_functions:
                fn.clear()
        else:
            st.cache_data.clear()
        seen["changed_at"] = changed


# ---------------------------------------------------------------------------
# Range picker widget
# ---------------------------------------------------------------------------

def _drive_sync_available() -> bool:
    try:
        from drive_sync import credentials_available
        return credentials_available()
    except ImportError:
        return False


def render_range_picker(video_id: str, source_path: str, duration: float, current_range,
                        *, key: str, suggestions=(), rerun=None,
                        apply_label: str = "Apply range",
                        clear_label: str = "Clear (use full video)",
                        slider_label: str = None):
    """Video player + range slider + Apply/Clear. Used by the Media Library and
    the Compilation Planner so both work the same way.

    source_path: the catalogue's raw source_path (converted to a local path here).
    current_range: (start, end) the slider starts from, or None for the full video.
    key: prefix for widget keys, so two pickers for the same video never clash.
    suggestions: [(button_label, fn)] — fn() returns (start, end) to put on the slider.
    rerun: called after a suggestion or download (st.rerun by default; the planner
        passes its fragment-scoped rerun).

    Returns ("apply", (start, end)), ("clear", None) or None when nothing was clicked.
    """
    from render_preview import colab_to_local
    rerun = rerun or st.rerun

    local_path = colab_to_local(source_path)
    proxy = proxy_path(video_id)
    if proxy.exists():
        st.video(str(proxy))
        st.caption("⚡ Fast preview copy")
    elif Path(local_path).exists():
        st.video(local_path)
        if st.button("⚡ Make a fast preview copy", key=f"{key}_proxy_{video_id}",
                     help="Encodes a small 480p copy so this player loads and scrubs quickly. "
                          "Takes a few minutes for a long video."):
            with st.spinner("Encoding preview copy…"):
                err = make_proxy(Path(local_path), proxy)
            if err:
                st.error(f"Couldn't make the preview copy: {err}")
            else:
                rerun()
    else:
        st.warning(f"Source video not found locally: {local_path}")
        if _drive_sync_available() and "MyDrive/" in source_path:
            if st.button("⬇️ Download this video from Google Drive", key=f"{key}_dl_{video_id}"):
                from drive_sync import download_source_video
                bar = st.progress(0.0, text="Downloading from Google Drive...")
                err = download_source_video(
                    source_path.split("MyDrive/", 1)[1], Path(local_path),
                    progress_callback=lambda p: bar.progress(min(p, 1.0)),
                )
                if err:
                    st.error(f"Download failed: {err}")
                else:
                    rerun()

    slider_key = f"{key}_slider_{video_id}"
    for i, (label, fn) in enumerate(suggestions):
        if st.button(label, key=f"{key}_suggest{i}_{video_id}"):
            s, e = fn()
            st.session_state[slider_key] = (seconds_to_time(s), seconds_to_time(e))
            rerun()

    default_range = current_range or (0.0, duration)
    new_range_t = st.slider(
        slider_label or f"Usable range for {video_id}",
        min_value=seconds_to_time(0), max_value=seconds_to_time(max(duration, 1.0)),
        value=(seconds_to_time(default_range[0]), seconds_to_time(default_range[1])),
        step=datetime.timedelta(seconds=1), format="mm:ss",
        key=slider_key,
    )

    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button(apply_label, key=f"{key}_apply_{video_id}", type="primary"):
            return "apply", (time_to_seconds(new_range_t[0]), time_to_seconds(new_range_t[1]))
    with btn_cols[1]:
        if st.button(clear_label, key=f"{key}_clear_{video_id}"):
            return "clear", None
    return None
