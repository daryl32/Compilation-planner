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
NOT_CATALOGUES = {"labels.json", "library_meta.json", "video_index.json"}


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
            "reviewed": bool(s.get("reviewed")),
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
def _index_store() -> dict:
    """Process-wide: {"files": {name: (mtime, slim or None)}, "lock": Lock}."""
    return {"files": {}, "lock": threading.Lock()}


def video_index() -> dict:
    """{stem: slim video} for every catalogue, re-reading only changed files."""
    store = _index_store()
    with store["lock"]:
        files = store["files"]
        seen = set()
        for p in catalogue_paths():
            seen.add(p.name)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            cached = files.get(p.name)
            if cached and cached[0] == mtime:
                continue
            try:
                cat = json.loads(p.read_text())
            except (ValueError, OSError):
                files[p.name] = (mtime, None)
                continue
            files[p.name] = (mtime, _slim_video(p.stem, cat, mtime) if is_video_catalogue(cat) else None)
        for gone in set(files) - seen:
            files.pop(gone, None)
        return {name[:-5]: v for name, (_, v) in sorted(files.items()) if v is not None}
