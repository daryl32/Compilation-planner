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
from track_files import track_file
from config import CATALOGUE_DIR, AUDIO_DIR

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


def set_library_ranges(changes: dict) -> None:
    """Several videos at once, in one save: {video_id: (start, end) or None to remove}."""
    meta = load_library_meta()
    for video_id, time_range in changes.items():
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


# ---------------------------------------------------------------------------
# Video details: actors and video tags (library_meta.json, synced like ranges)
# ---------------------------------------------------------------------------

def _clean_list(values) -> list:
    seen, out = set(), []
    for v in values or []:
        v = " ".join(str(v).split())
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def video_details() -> dict:
    """{video_id: {"actors": [...], "tags": [...]}} for videos that have any."""
    out = {}
    for vid, entry in load_library_meta().items():
        if isinstance(entry, dict) and (entry.get("actors") or entry.get("video_tags")):
            out[vid] = {"actors": list(entry.get("actors") or []), "tags": list(entry.get("video_tags") or [])}
    return out


def set_video_details(video_id: str, actors=None, tags=None) -> None:
    """Save a video's actors and/or video tags (None leaves that one as it is)."""
    meta = load_library_meta()
    entry = meta.get(video_id) if isinstance(meta.get(video_id), dict) else {}
    for key, values in (("actors", actors), ("video_tags", tags)):
        if values is None:
            continue
        values = _clean_list(values)
        if values:
            entry[key] = values
        else:
            entry.pop(key, None)
    if entry:
        meta[video_id] = entry
    else:
        meta.pop(video_id, None)
    save_library_meta(meta)


def all_actors() -> list:
    names = {}
    for d in video_details().values():
        for a in d["actors"]:
            names.setdefault(a.lower(), a)
    return sorted(names.values(), key=str.lower)


def all_video_tags() -> list:
    tags = {}
    for d in video_details().values():
        for t in d["tags"]:
            tags.setdefault(t.lower(), t)
    return sorted(tags.values(), key=str.lower)


def suggest_video_tags(scenes: list, time_range=None, min_share: float = 0.10, min_scenes: int = 2) -> list:
    """Clip tags worth putting on the whole video: [(tag, share of footage)],
    biggest first — a tag on at least min_share of the usable footage (inside
    the time range, excluded scenes left out) and on at least min_scenes scenes.
    scenes: catalogue scenes (start_tc / end_tc) or slim ones (start / end)."""
    secs, counts, total = {}, {}, 0.0
    for s in scenes:
        if s.get("excluded"):
            continue
        try:
            a = s["start"] if "start" in s else tc_to_seconds(s["start_tc"])
            b = s["end"] if "end" in s else tc_to_seconds(s["end_tc"])
        except (KeyError, ValueError, TypeError):
            continue
        a, b = overlap_with_range(a, b, time_range)
        if a is None or b <= a:
            continue
        total += b - a
        for t in (s.get("tags") if "start" in s else scene_tags(s)) or []:
            secs[t] = secs.get(t, 0.0) + (b - a)
            counts[t] = counts.get(t, 0) + 1
    if total <= 0:
        return []
    out = [(t, secs[t] / total) for t in secs if secs[t] / total >= min_share and counts[t] >= min_scenes]
    return sorted(out, key=lambda x: -x[1])


def matches_details(video_id: str, details: dict, actors=(), tags=(), match_all: bool = False) -> bool:
    """Does this video have the wanted actors / video tags? Empty lists match
    everything. match_all: every wanted one; otherwise any (actors and tags are
    each checked on their own, and both must pass)."""
    d = details.get(video_id) or {"actors": [], "tags": []}
    for wanted, have in ((actors, d["actors"]), (tags, d["tags"])):
        if not wanted:
            continue
        want = {w.lower() for w in wanted}
        got = {h.lower() for h in have}
        if (match_all and not want <= got) or (not match_all and not want & got):
            return False
    return True


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


def _pending_target(stem: str):
    """(local path, Drive folder) for a pending entry, or None for a render
    entry / a file that's gone."""
    if stem.startswith("audio:"):
        path, folder = track_file(stem[6:], AUDIO_DIR), "scene-labeling/audio_catalogue"
    elif stem.startswith("thumb:"):
        # "thumb:<video_id>/<file>" — a thumbnail the Reviewer made for a split scene.
        vid, _, name = stem[6:].partition("/")
        path = CATALOGUE_DIR / "thumbnails" / vid / name
        folder = f"scene-labeling/catalogue/thumbnails/{vid}"
    elif stem.startswith("cat:"):
        # "cat:<file>" — any other file in the catalogue folder (e.g. tag_vocabulary.csv).
        path, folder = CATALOGUE_DIR / Path(stem[4:]).name, "scene-labeling/catalogue"
    else:
        path, folder = CATALOGUE_DIR / f"{stem}.json", "scene-labeling/catalogue"
    return (path, folder) if path.exists() else None


def push_pending(token: dict, progress=None) -> dict:
    """Upload every pending file to Drive: plain stems are video catalogues
    (CATALOGUE_DIR → scene-labeling/catalogue); "audio:<stem>" entries are
    track catalogues (AUDIO_DIR → scene-labeling/audio_catalogue), e.g. ones
    the server analysed; "thumb:<video_id>/<file>" entries are thumbnails
    (CATALOGUE_DIR/thumbnails → scene-labeling/catalogue/thumbnails); "cat:<file>" entries
    are other catalogue-folder files such as tag_vocabulary.csv;
    "render:<file>.mp4" entries are a render's details (rating, notes, keep)
    edited while Drive wasn't connected (PREVIEW_DIR → scene-labeling/previews).

    Files go up several at a time (drive_oauth.upload_many). Each one is taken
    off the pending list as soon as it's uploaded, so stopping part-way only
    leaves the rest to do. progress(done, total) is called as files finish.
    Returns {entry: error} for any that failed (those stay pending)."""
    from drive_oauth import upload_many
    pending = sorted(read_pending())
    failed, items, gone, mtimes = {}, [], set(), {}
    for stem in pending:
        if stem.startswith("render:"):
            continue
        target = _pending_target(stem)
        if target is None:
            gone.add(stem)          # nothing to push any more
        else:
            items.append((stem, target[0], target[1]))
            mtimes[stem] = target[0].stat().st_mtime
    renders = [s for s in pending if s.startswith("render:")]
    total = len(items) + len(renders)
    if gone:
        write_pending(read_pending() - gone)

    finished = set()
    items_by_key = [(k, path) for k, path, _ in items]

    def on_done(done, _total, key, err):
        # Edited again while it was uploading? Then it stays pending for next time.
        try:
            unchanged = dict(items_by_key)[key].stat().st_mtime == mtimes[key]
        except (KeyError, OSError):
            unchanged = False
        if not err and unchanged:
            finished.add(key)
        # Write progress to the pending list every few files (and at the end).
        if finished and (len(finished) % 10 == 0 or done == _total):
            write_pending(read_pending() - finished)
        if progress:
            progress(done, total)

    failed.update(upload_many(token, items, progress=on_done))
    write_pending(read_pending() - finished)

    for i, stem in enumerate(renders, 1):
        from config import PREVIEW_DIR
        from renders import sync_sidecar, load_render, on_drive
        mp4 = PREVIEW_DIR / stem[7:]
        meta = load_render(mp4)
        err = None
        if meta is not None and on_drive(meta):
            err = sync_sidecar(mp4, token)
        if err:
            failed[stem] = err
        else:
            write_pending(read_pending() - {stem})   # done, or deleted / not on Drive
        if progress:
            progress(len(items) + i, total)
    return failed


# ---------------------------------------------------------------------------
# Pictures that might be broken
# ---------------------------------------------------------------------------
_IMAGE_OK = {}   # (path, mtime, size) -> bool


def image_ok(path) -> bool:
    """True if the file exists and is a readable picture. A thumbnail can be
    empty or half-written (e.g. a download cut off part-way)."""
    try:
        p = Path(path)
        st_ = p.stat()
    except (OSError, TypeError):
        return False
    key = (str(p), st_.st_mtime, st_.st_size)
    if key not in _IMAGE_OK:
        ok = st_.st_size > 0
        if ok:
            try:
                from PIL import Image
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                ok = False
        if len(_IMAGE_OK) > 50_000:
            _IMAGE_OK.clear()
        _IMAGE_OK[key] = ok
    return _IMAGE_OK[key]


def safe_image(img, *args, **kwargs) -> bool:
    """st.image that never breaks the page: a file that isn't a readable picture
    shows a small note instead. Images that aren't files (PIL images, arrays)
    are passed straight through. Returns True if a picture was shown."""
    if isinstance(img, (str, Path)) and not image_ok(img):
        st.caption(f"🖼️ Picture unreadable: {Path(img).name}")
        return False
    try:
        st.image(img, *args, **kwargs)
        return True
    except Exception:
        st.caption("🖼️ Picture unreadable")
        return False


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


def sync_time_text(iso: str) -> str:
    """'Thu 22 Oct, 09:14 (45 min ago)' — UK time (the app's users), the server keeps UTC."""
    then = datetime.datetime.fromisoformat(iso)
    if then.tzinfo is None:
        then = then.replace(tzinfo=datetime.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        local = then.astimezone(ZoneInfo(getattr(config, "DISPLAY_TIMEZONE", "Europe/London")))
    except Exception:
        local = then
    return f"{local:%a} {local.day} {local:%b}, {local:%H:%M} ({_ago(iso)[0]})"


def render_sync_status(where=None) -> None:
    """One line: when Drive was last synced (it only syncs when 🔄 Sync now is
    pressed), and whether it had errors. where: the container (default the sidebar)."""
    from auto_sync import read_sync_status
    where = where or st.sidebar
    status = read_sync_status()
    if not status.get("finished_at"):
        where.caption("☁️ Drive: not synced yet — 🔄 Sync now")
        return
    try:
        text = sync_time_text(status["finished_at"])
    except ValueError:
        return
    errors = status.get("errors") or []
    if errors:
        where.warning(f"☁️ Last Drive sync {text} had {len(errors)} error(s): {errors[0]}")
    else:
        where.caption(f"☁️ Last Drive sync: {text}")


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
            clear_shared_data_caches()
        seen["changed_at"] = changed


# Shared (st.cache_resource) data caches registered by pages — cleared after a
# sync and by the memory safety valve. (_sync_seen is NOT one of these.)
_SHARED_CACHES = []


def shared_cache(fn):
    """Register an st.cache_resource-decorated function as a clearable data cache."""
    _SHARED_CACHES.append(fn)
    return fn


def clear_shared_data_caches() -> None:
    for fn in _SHARED_CACHES:
        try:
            fn.clear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Memory: log the app's memory use, and clear caches before it gets dangerous
# ---------------------------------------------------------------------------
# The server has 4 GB. On 10 Oct 2026 the live app grew to 3.2 GB during a long
# Choreography session and the kernel killed it. memory_check() runs on every
# page run (page_setup) and inside the planner's busiest fragments: it logs the
# app's memory to DATA/logs/memory.csv whenever it moves, and if it passes
# MEMORY_LIMIT_MB (config.py, default 1800) it clears the data caches and hands
# the freed memory back — the page just reloads its data on the next click.

MEMORY_LIMIT_MB = float(getattr(config, "MEMORY_LIMIT_MB", 1800))
# How a clear is written in the memory log. The Server page counts only clears written this way at
# today's limit — older ones (VmRSS, or the 1800 MB limit before the 8 GB upgrade) don't apply now.
CLEAR_NOTE = "own memory over {limit:.0f} MB"
MEMORY_LOG = CATALOGUE_DIR.parent / "logs" / "memory.csv"
_MEMORY_LOG_MAX_BYTES = 2_000_000


def _status_mb(text: str) -> dict:
    """{"VmRSS", "RssAnon", "RssFile", "RssShmem"} in MB from /proc/<pid>/status text."""
    out = {}
    for line in text.splitlines():
        key = line.split(":", 1)[0]
        if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
            try:
                out[key] = int(line.split()[1]) / 1024
            except (IndexError, ValueError):
                pass
    return out


def private_mb(fields: dict):
    """The memory this app really holds: its own (anonymous + shared) pages. Not VmRSS — that
    also counts files it has read through memory-mapping (the motion-curve cache), pages the
    system can drop at any moment; a long planner session pushed VmRSS past the limit and
    cleared the caches for nothing (21 Oct 2026). Falls back to VmRSS on older kernels."""
    if "RssAnon" in fields:
        return fields["RssAnon"] + fields.get("RssShmem", 0.0)
    return fields.get("VmRSS")


def app_memory_mb():
    """This app's own memory in MB (private_mb — what the limit is about), or None off Linux."""
    try:
        with open("/proc/self/status") as f:
            return private_mb(_status_mb(f.read()))
    except OSError:
        return None


def app_mapped_mb():
    """Files the app has read through memory-mapping, in MB (shown, never counted), or None."""
    try:
        with open("/proc/self/status") as f:
            return _status_mb(f.read()).get("RssFile")
    except OSError:
        return None


@st.cache_resource
def _memory_state() -> dict:
    return {"last_mb": 0.0, "last_time": 0.0, "peak_mb": 0.0}


def _release_memory() -> None:
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)   # hand freed memory back to the system
    except Exception:
        pass


def _log_memory(where: str, mb: float, note: str = "") -> None:
    import time as _time
    try:
        MEMORY_LOG.parent.mkdir(parents=True, exist_ok=True)
        if MEMORY_LOG.exists() and MEMORY_LOG.stat().st_size > _MEMORY_LOG_MAX_BYTES:
            MEMORY_LOG.replace(MEMORY_LOG.with_suffix(".old.csv"))
        new = not MEMORY_LOG.exists()
        with open(MEMORY_LOG, "a") as f:
            if new:
                f.write("time,where,memory_mb,note\n")
            stamp = datetime.datetime.now().isoformat(timespec="seconds")
            f.write(f"{stamp},{where.replace(',', ';')},{mb:.0f},{note.replace(',', ';')}\n")
    except OSError:
        pass


def free_shared_memory(release_media: bool = True) -> dict:
    """Clear what every page shares: cached data, the shared data caches, and
    (release_media) the videos / pictures / music Streamlit is holding for
    players on any page — a page that's still open simply loads its player
    again next time it updates. Then hands freed memory back to the system.
    Returns {"media_files", "media_mb"} released."""
    media_files, media_mb = 0, 0.0
    if release_media:
        try:
            from streamlit.runtime import Runtime
            mgr = Runtime.instance().media_file_mgr
            before = memory_breakdown().get("media", 0.0)
            with mgr._lock:
                sessions = list(mgr._files_by_session_and_coord.keys())
                count_before = len(mgr._file_metadata)
            for sid in sessions:
                mgr.clear_session_refs(sid)
            mgr.remove_orphaned_files()
            media_files = max(0, count_before - len(mgr._file_metadata))
            media_mb = max(0.0, before - memory_breakdown().get("media", 0.0))
        except Exception:
            pass
    st.cache_data.clear()
    clear_shared_data_caches()
    _release_memory()
    return {"media_files": media_files, "media_mb": media_mb}


def memory_breakdown(include_sessions: bool = False) -> dict:
    """MB held, by kind, from Streamlit's own accounting:
      media     — videos / music / pictures loaded for players (st.video etc. given a file)
      cache     — st.cache_data
      shared    — st.cache_resource (catalogues, motion curves, …)
      sessions  — every open page's working data (only when include_sessions:
                  measuring it walks every session and can take a few seconds)
    Kinds it can't see (libraries, Python itself, memory not yet handed back)
    are the rest of app_memory_mb()."""
    out = {"media": 0.0, "cache": 0.0, "shared": 0.0, "sessions": 0.0, "n_sessions": 0}
    try:
        from streamlit.runtime import Runtime
        rt = Runtime.instance()
        families = None if include_sessions else ["cache_memory_bytes"]
        try:
            raw = rt.stats_mgr.get_stats(families) if families else rt.stats_mgr.get_stats()
        except TypeError:
            raw = rt.stats_mgr.get_stats()
        stats = []
        for v in (raw.values() if isinstance(raw, dict) else [raw]):
            stats.extend(v if isinstance(v, (list, tuple)) else [v])
        for stat in stats:
            name = getattr(stat, "category_name", "") or ""
            size = float(getattr(stat, "byte_length", 0) or 0) / 2**20
            if "media_file" in name:
                out["media"] += size
            elif "cache_data" in name:
                out["cache"] += size
            elif "cache_resource" in name:
                out["shared"] += size
            elif "session_state" in name and include_sessions:
                out["sessions"] += size
        try:
            out["n_sessions"] = len(rt._session_mgr.list_active_sessions())
        except Exception:
            pass
    except Exception:
        pass
    return out


def memory_check(where: str):
    """Log memory if it changed (or every 5 min); clear caches past the limit.
    Returns the app's memory in MB (None if unknown)."""
    import time as _time
    try:
        from app_switch import touch_heartbeat
        touch_heartbeat()   # "someone is using this copy" — for the test copy's idle shut-down
    except Exception:
        pass
    mb = app_memory_mb()
    if mb is None:
        return None
    state = _memory_state()
    state["peak_mb"] = max(state["peak_mb"], mb)
    note = ""
    # Past the limit: quietly clear cached data and hand memory back (no pop-up).
    # At most once every 10 minutes — clearing again straight away gains little,
    # and doing it on every click made pages slow without bringing memory down.
    if mb > MEMORY_LIMIT_MB and _time.time() - state.get("last_clear", 0.0) > 600:
        state["last_clear"] = _time.time()
        freed = free_shared_memory()
        after = app_memory_mb() or mb
        note = (f"{CLEAR_NOTE.format(limit=MEMORY_LIMIT_MB)}: cleared caches and {freed['media_files']} held "
                f"media file(s) -> {after:.0f} MB")
        mb = after
    now = _time.time()
    if note or abs(mb - state["last_mb"]) >= 25 or now - state["last_time"] >= 300:
        _log_memory(where, mb, note)
        state["last_mb"], state["last_time"] = mb, now
    return mb


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


def file_download_button(path, label: str, key: str, mime: str = "video/mp4", **kwargs) -> None:
    """A download button for a file on the server that doesn't read the file
    until it's clicked (a page full of them would otherwise load every video
    into memory). Older Streamlit without that: a first click prepares it."""
    path = Path(path)
    if "callable" in (st.download_button.__doc__ or ""):
        st.download_button(label, data=lambda p=path: p.read_bytes(), file_name=path.name, mime=mime,
                           key=key, **kwargs)
        return
    ready_key = f"_dl_ready_{key}"
    if st.session_state.get(ready_key):
        st.download_button(f"💾 Save {path.name}", data=path.read_bytes(), file_name=path.name, mime=mime,
                           key=key, on_click=lambda: st.session_state.pop(ready_key, None), **kwargs)
    elif st.button(label, key=f"{key}_prep", **kwargs):
        st.session_state[ready_key] = True
        st.rerun()


def show_render_player(meta: dict, height: int = 360) -> None:
    """Play a saved render: the server's copy if it's still here, otherwise
    straight from Google Drive with Drive's own player (works in a browser
    signed in to the Google account that owns the file)."""
    from renders import drive_preview_url
    path = Path(meta["path"])
    if path.exists():
        st.video(str(path))
        return
    url = drive_preview_url(meta)
    if url:
        if st.button("💾 Copy to the server for smooth playback", key=f"fetch_play_{path.name}",
                     help="Streaming from Google Drive can buffer. This downloads it onto the server once."):
            from renders import fetch_to_server
            with st.spinner("Copying from Google Drive…"):
                err = fetch_to_server(meta)
            if err:
                st.error(err)
            else:
                st.rerun()
        import embed
        embed.url(url, height=height)
        st.caption("☁️ Playing from Google Drive. A video uploaded in the last minute or two may still be "
                   "processing — if so, wait a moment and reopen it.")
    else:
        st.warning("This render is no longer on the server or Google Drive.")
