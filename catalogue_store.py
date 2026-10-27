"""
One place to read and write the video catalogues (CATALOGUE_DIR/<video>.json).

Every page used to glob CATALOGUE_DIR and json-load every file (motion curves
and all) on its own. This module:

  • video_index()   — a slim summary of every video and its scenes (times, tags,
                      flags, motion score, thumbnail name — no motion curves),
                      shared by every session. Only files whose modified-time
                      changed since last time are re-read, so after a Reviewer
                      save or a Drive sync just those files are parsed again.
  • load_catalogue(stem) / save_catalogue(stem, data) — the full JSON of one
                      video; saving is atomic and queues the file for Drive.
  • added_times()   — when each video was added (Drive's creation date of its
                      catalogue, recorded by drive_sync), for "Date added"
                      sorting and the 🆕 newly-added filters.
  • catalogue_paths() — the catalogue files (labels.json, library_meta.json,
                      dot-files etc. are skipped).

New features that need "something about every video" should add a field to
_slim_video() rather than loading catalogues themselves.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import streamlit as st

from config import CATALOGUE_DIR
from library_common import mark_pending, scene_tags, shared_cache, tc_to_seconds

# JSON files in CATALOGUE_DIR that are never video catalogues.
NOT_CATALOGUES = {"labels.json", "library_meta.json", "video_index.json", "scene_roles.json",
                  "story_maps.json", "training_labels.json"}


def catalogue_paths() -> list:
    try:
        return sorted(p for p in CATALOGUE_DIR.glob("*.json")
                      if p.name not in NOT_CATALOGUES and not p.name.startswith("."))
    except OSError:
        return []


def catalogue_path(stem: str) -> Path:
    return CATALOGUE_DIR / f"{stem}.json"


def is_video_catalogue(data) -> bool:
    return isinstance(data, dict) and "video_id" in data and "scenes" in data


def load_catalogue(stem: str) -> dict:
    return json.loads(catalogue_path(stem).read_text())


def save_catalogue(stem: str, data: dict) -> None:
    """Write atomically (a crash mid-write can't leave half a file), queue the
    file for the next Drive save, and let every page see the change."""
    path = catalogue_path(stem)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)
    mark_pending(stem)


# ---------------------------------------------------------------------------
# The shared index
# ---------------------------------------------------------------------------

def _scene_order(s: dict) -> tuple:
    """Time order — scene_id isn't chronological once the Reviewer has split a scene."""
    try:
        return (tc_to_seconds(s["start_tc"]), s["scene_id"])
    except (KeyError, ValueError, TypeError):
        return (float("inf"), s.get("scene_id", 0))


def _roles(scene: dict) -> list:
    import scene_roles as SR
    return SR.scene_roles(scene)


def _slim_video(stem: str, cat: dict, mtime: float) -> dict:
    scenes = []
    for s in sorted(cat["scenes"], key=_scene_order):
        try:
            start, end = tc_to_seconds(s["start_tc"]), tc_to_seconds(s["end_tc"])
        except (KeyError, ValueError, TypeError):
            continue
        thumbs = s.get("thumbnail_paths") or []
        scenes.append({
            "scene_id": s["scene_id"],
            "start_tc": s["start_tc"], "end_tc": s["end_tc"],
            "start": start, "end": end,
            "tags": scene_tags(s),
            "excluded": bool(s.get("excluded")),
            "intro": bool(s.get("intro_candidate")),
            "outro": bool(s.get("outro_candidate")),
            "roles": _roles(s),
            "reviewed": bool(s.get("reviewed")),
            "roles_reviewed": bool(s.get("roles_reviewed")),
            "thumb": Path(thumbs[0]).name if thumbs else None,
            "motion": s.get("motion_intensity"),
            "has_curve": bool((s.get("motion_curve") or {}).get("values")),
        })
    return {
        "stem": stem,
        "video_id": cat["video_id"],
        "source_path": cat.get("source_path", ""),
        "duration": scenes[-1]["end"] if scenes else 0.0,
        "scenes": scenes,
        "dismissed_edit_suggestions": list(cat.get("dismissed_edit_suggestions", [])),
        "added": mtime,
    }


@shared_cache
@st.cache_resource
def _store() -> dict:
    """Process-wide: {"files": {name: (mtime, catalogue or None, slim or None)},
    "catalogues": {video_id: catalogue} or None, "lock": Lock}."""
    return {"files": {}, "catalogues": None, "lock": threading.Lock()}


def _strip(cat: dict) -> dict:
    """The planner's working copy: no motion curves (big, per frame — loaded per
    video by motion_curves()), and Reviewer-corrected tags as "tags"."""
    for scene in cat["scenes"]:
        scene.pop("motion_curve", None)
        scene["tags"] = scene_tags(scene)
    return cat


def _refresh() -> dict:
    """Re-read only the catalogue files that changed (or appeared) since last
    time; drop ones that went. Returns the store. Call with no lock held."""
    store = _store()
    with store["lock"]:
        files = store["files"]
        seen, changed = set(), False
        for p in catalogue_paths():
            seen.add(p.name)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            cached = files.get(p.name)
            if cached and cached[0] == mtime:
                continue
            changed = True
            try:
                cat = json.loads(p.read_text())
            except (ValueError, OSError):
                files[p.name] = (mtime, None, None)
                continue
            if not is_video_catalogue(cat):
                files[p.name] = (mtime, None, None)
                continue
            slim = _slim_video(p.stem, cat, mtime)
            files[p.name] = (mtime, _strip(cat), slim)
        for gone in set(files) - seen:
            files.pop(gone, None)
            changed = True
        if changed or store["catalogues"] is None:
            # A new dict each time something changed — callers may still be
            # reading the previous one.
            store["catalogues"] = {c["video_id"]: c for _, (_, c, _) in sorted(files.items()) if c}
        return store


def video_index() -> dict:
    """{stem: slim video} for every catalogue (see _slim_video). Shared — read only."""
    store = _refresh()
    return {name[:-5]: slim for name, (_, _, slim) in sorted(store["files"].items()) if slim}


def all_catalogues() -> dict:
    """{video_id: catalogue} for every video — full catalogues WITHOUT motion
    curves, tags already corrected. ONE SHARED COPY: read it, never change it.
    Always current (Reviewer edits and syncs show up on the next call)."""
    return _refresh()["catalogues"]


@shared_cache
@st.cache_resource(max_entries=20, ttl=1800, show_spinner=False)
def _motion_curves(video_id: str, mtime: float) -> dict:
    import curve_cache
    return curve_cache.load(catalogue_path(video_id), curve_cache.default_cache_dir(CATALOGUE_DIR))


def motion_curves(video_id: str) -> dict:
    """{scene_id: {"fps": fps, "values": numpy array}} for one video, reloaded
    when its file changes. The values are read-only numpy arrays from the local
    curve cache (curve_cache.py) — test emptiness with len(), not truthiness.
    Shared — read only."""
    try:
        mtime = catalogue_path(video_id).stat().st_mtime
    except OSError:
        return {}
    return _motion_curves(video_id, mtime)


# ---------------------------------------------------------------------------
# When each video was added
# ---------------------------------------------------------------------------
# drive_sync writes CATALOGUE_DIR/.added_dates.json ({stem: Drive createdTime})
# on every sync. A catalogue's own modified-time is NOT its added date — every
# Reviewer edit or re-download changes it — so it's only the fallback for videos
# the file doesn't know yet (e.g. on the laptop, which never runs a sync).
ADDED_DATES_FILE = CATALOGUE_DIR / ".added_dates.json"
NEW_VIDEO_DAYS = 7   # "🆕 new" badge / default "newly added" window


@st.cache_resource(max_entries=2, show_spinner=False)
def _added_dates(mtime: float) -> dict:
    from datetime import datetime
    try:
        raw = json.loads(ADDED_DATES_FILE.read_text())
    except (ValueError, OSError):
        return {}
    out = {}
    for stem, iso in (raw.items() if isinstance(raw, dict) else []):
        try:
            out[stem] = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return out


def added_times() -> dict:
    """{video_id: unix time the video was added} for every video."""
    try:
        known = _added_dates(ADDED_DATES_FILE.stat().st_mtime)
    except OSError:
        known = {}
    return {v["video_id"]: known.get(stem, v["added"]) for stem, v in video_index().items()}


def added_ago(ts: float) -> str:
    """'today', 'yesterday', '3 days ago', else the date — for a video's added time."""
    from datetime import date, datetime
    when = datetime.fromtimestamp(ts).date()
    days = (date.today() - when).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    return when.strftime("%d %b %Y")


def is_new_video(ts: float, days: int = NEW_VIDEO_DAYS) -> bool:
    import time
    return ts >= time.time() - days * 86400
