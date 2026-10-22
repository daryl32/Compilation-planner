"""Loading tracks, catalogues, motion curves and thumbnails (cached), plus footage helpers.

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

import io
import json
import math
import datetime
from pathlib import Path
import streamlit as st
from PIL import Image
from render_preview import colab_to_local
import catalogue_store as CStore
import scene_roles as SR
from library_common import (library_ranges, master_thumbnail_path, shared_cache)
import core.footage as CFOOT
from core.footage import _tc_to_seconds
from config import CATALOGUE_DIR, AUDIO_DIR
from track_files import track_file

def resolve_thumbnail(stored_path: str, video_id: str) -> Path:
    filename = Path(stored_path).name
    return CATALOGUE_DIR / "thumbnails" / video_id / filename


# ---------------------------------------------------------------------------
# Data loading (cached — only recomputes when inputs actually change)
# ---------------------------------------------------------------------------

def list_tracks():
    # Not cached: a folder listing is cheap, and new tracks (synced, or analysed
    # on the server from the Media Library) should appear without a restart.
    return sorted(f.stem for f in AUDIO_DIR.glob("*.json") if f.name != "audio_index.json")


@st.cache_data(max_entries=4)
def _load_track_file(track_id: str, mtime: float) -> dict:
    with open(track_file(track_id, AUDIO_DIR)) as f:
        return json.load(f)


def load_track(track_id: str) -> dict:
    """The track's catalogue — re-read whenever the file changes (e.g. after the
    server adds its extra analysis)."""
    try:
        mtime = track_file(track_id, AUDIO_DIR).stat().st_mtime
    except OSError:
        mtime = 0.0
    return _load_track_file(track_id, mtime)


def load_all_catalogues() -> dict:
    """video_id -> catalogue dict, WITHOUT motion_curve, tags corrected. ONE
    SHARED COPY — read it, never change it. Comes from catalogue_store, which
    re-reads only the files that changed, so Reviewer edits and Drive syncs
    show up on the next click. (History: st.cache_data deep-copied the whole
    library on every call and ran the 4 GB server out of memory; motion curves
    stay out of it — Advanced Shape Matching loads them per video below.)"""
    return CStore.all_catalogues()


def load_motion_curves_for_video(video_id: str) -> dict:
    """scene_id -> motion_curve, for ONE video — reloaded when that video's
    catalogue changes (e.g. a Reviewer split). SHARED — read it, never change it."""
    return CStore.motion_curves(video_id)


def seconds_to_time(sec: float) -> datetime.time:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return datetime.time(hour=min(h, 23), minute=m, second=s)


def time_to_seconds(t: datetime.time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second


def find_best_motion_window(video_id: str, window_len: float) -> tuple:
    """Slide a window of window_len seconds across this video's scenes and
    return the (start, end) with the highest duration-weighted average motion
    intensity. Candidate start positions are each scene's own start time —
    a natural, efficient heuristic rather than scanning every possible offset."""
    cat = load_all_catalogues()[video_id]
    scenes = cat["scenes"]
    duration = get_video_duration(video_id)
    if window_len >= duration:
        return 0.0, duration

    candidates = sorted({0.0, max(0.0, duration - window_len)} |
                         {_tc_to_seconds(s["start_tc"]) for s in scenes})

    best_start, best_score = 0.0, -1.0
    for t0 in candidates:
        t0 = min(t0, max(0.0, duration - window_len))
        t1 = t0 + window_len
        total_overlap = weighted = 0.0
        for s in scenes:
            s_start, s_end = _tc_to_seconds(s["start_tc"]), _tc_to_seconds(s["end_tc"])
            overlap = max(0.0, min(s_end, t1) - max(s_start, t0))
            if overlap > 0:
                total_overlap += overlap
                weighted += overlap * s.get("motion_intensity", 0.0)
        score = (weighted / total_overlap) if total_overlap > 0 else 0.0
        if score > best_score:
            best_score, best_start = score, t0

    return best_start, min(best_start + window_len, duration)


@st.cache_data
def _all_video_durations() -> dict:
    """video_id -> duration in seconds. Small, so cheap for st.cache_data to copy on
    every call — unlike load_all_catalogues(), which deep-copies the whole library each
    time and used to be called once per video row (and again per row in the sort)."""
    out = {}
    for vid, cat in load_all_catalogues().items():
        out[vid] = _tc_to_seconds(cat["scenes"][-1]["end_tc"]) if cat.get("scenes") else 0.0
    return out


def get_video_duration(video_id: str) -> float:
    return _all_video_durations().get(video_id, 0.0)


def effective_time_ranges() -> dict:
    """{video_id: (start, end)} each video actually uses: this project's override
    (🎚️, session_state["video_time_ranges"]) if there is one, otherwise its Media
    Library range (📚), otherwise the full video (not listed). An override of None
    means "full video" even when the library has a range."""
    merged = dict(library_ranges())
    merged.update(st.session_state.get("video_time_ranges", {}))
    return {vid: tuple(rng) for vid, rng in merged.items() if rng}


def build_footage_queues(candidate_video_ids: tuple, tag_filter: tuple, video_time_ranges: dict = None) -> dict:
    """core.footage.build_footage_queues on this library's catalogues."""
    return CFOOT.build_footage_queues(load_all_catalogues(), candidate_video_ids, tag_filter, video_time_ranges,
                                      thumbnail_for=resolve_thumbnail, roles_of=SR.scene_roles)


def _local_media(raw_source: str, prefer: Path = None):
    """(local path, None) for a media file, or (None, raw_source) if it isn't on this machine
    (or can't be checked — e.g. /root isn't readable by this user)."""
    def _there(p):
        try:
            return p.exists()
        except OSError:
            return False
    if prefer is not None and _there(Path(prefer)):
        return Path(prefer), None
    local = Path(colab_to_local(raw_source)) if raw_source else None
    if local is not None and _there(local):
        return local, None
    return None, raw_source


def _add_video_to_selection(video_id: str) -> None:
    """➕ from the ✨ Similar videos panel — runs as a button callback (before
    widgets redraw), so it can tick the video and add it to the extra list."""
    if video_id not in st.session_state.get("recommended_ids", []):
        extras = list(st.session_state.get("extra_videos", []))
        if video_id not in extras:
            extras.append(video_id)
        st.session_state["extra_videos"] = extras
    st.session_state[f"select_video_{video_id}"] = True


# Shared, not copied per call (cache_data handed every thumbnail crop its own full
# copy of the sprite sheet). Only ever cropped, which makes a new small image.
@shared_cache
@st.cache_resource(max_entries=6, ttl=1800, show_spinner=False)
def load_timeline_sprite(sprite_path: str):
    """The whole per-video sprite sheet, loaded and cached once per video —
    cropping a specific tile from it (get_nearest_thumbnail) is cheap and
    done fresh on every call, but re-reading the sprite FILE every time
    would not be."""
    img = Image.open(sprite_path)
    img.load()  # force full read now, so the cached object doesn't hold an open file handle
    return img


def video_list_thumbnail(catalogues: dict, video_id: str, time_range=None):
    """Small preview for the video list: the ⭐ master thumbnail chosen in the
    Reviewer if there is one, otherwise the timeline-sprite tile from the middle
    of the video's usable range (a 120-px crop from an already-cached sprite, so
    it's cheap), falling back to that scene's own thumbnail file. None if
    neither exists."""
    master = master_thumbnail_path(video_id, catalogues.get(video_id, {}).get("scenes", []))
    if master is not None:  # ⭐ chosen in the Reviewer
        return str(master)
    duration = get_video_duration(video_id)
    a, b = time_range or (0.0, duration)
    mid = (a + b) / 2
    tile = get_nearest_thumbnail(catalogues, video_id, mid)
    if tile is not None:
        return tile
    for sc in catalogues.get(video_id, {}).get("scenes", []):
        try:
            inside = _tc_to_seconds(sc["start_tc"]) <= mid <= _tc_to_seconds(sc["end_tc"])
        except (KeyError, ValueError):
            continue
        if inside and sc.get("thumbnail_paths"):
            path = resolve_thumbnail(sc["thumbnail_paths"][0], video_id)
            return str(path) if path.exists() else None
    return None


def get_nearest_thumbnail(catalogues: dict, video_id: str, clip_start_sec: float):
    """The single thumbnail closest to clip_start_sec, constrained to never be
    from BEFORE it — so what's shown is representative of what the clip
    actually opens on, never a moment that gets trimmed away before the clip
    even starts. Crops one tile out of the video's timeline sprite sheet (see
    backfill_timeline_thumbnails.py). Returns a PIL Image, or None if this
    video has no sprite yet (backfill not run) or the sprite file is missing
    locally. If clip_start_sec falls past the sprite's last sampled second
    (an edge case — the clip starts later than any thumbnail was taken), the
    last available tile is used, since no at-or-after tile exists to pick."""
    meta = catalogues.get(video_id, {}).get("timeline_thumbnails")
    if not meta:
        return None
    # Derived from the LOCAL CATALOGUE_DIR, never from meta["sprite_path"] — that field
    # was written by whichever environment ran the timeline-thumbnail backfill (Colab,
    # the Windows laptop, or the Hetzner server), so it can point somewhere that doesn't
    # exist, or worse, somewhere unrelated, on whichever environment reads it back.
    sprite_path = CATALOGUE_DIR / "timeline_sprites" / f"{video_id}.jpg"
    try:
        mtime = sprite_path.stat().st_mtime
    except OSError:
        return None

    tile_w, tile_h, columns, count, interval = (
        meta["tile_width"], meta["tile_height"], meta["columns"], meta["count"], meta["interval_sec"]
    )
    # ceil, not round: the nearest tile at-or-after clip_start_sec, never one before it.
    idx = min(count - 1, max(0, math.ceil(clip_start_sec / interval)))
    # ⚡ The finished picture is kept (a few KB), so a page run never decodes a whole sprite sheet
    # (tens of MB for a long video) or encodes the same tile again. JPEG at quality 100 — the bytes
    # st.image itself makes from a PIL image, so what's shown is unchanged.
    tiles = _tile_bytes()
    key = (str(sprite_path), mtime, idx, tile_w, tile_h, columns)
    got = tiles.get(key)
    if got is None:
        sprite = load_timeline_sprite(str(sprite_path))
        r, c = divmod(idx, columns)
        buf = io.BytesIO()
        sprite.crop((c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)).save(buf, format="JPEG", quality=100)
        got = buf.getvalue()
        if len(tiles) >= TILE_CACHE_MAX:
            tiles.clear()
        tiles[key] = got
    return got


TILE_CACHE_MAX = 6000      # ~50 MB of finished thumbnails


@shared_cache
@st.cache_resource
def _tile_bytes() -> dict:
    """(sprite file, modified time, tile …) → JPEG bytes of that tile, shared by every session."""
    return {}

