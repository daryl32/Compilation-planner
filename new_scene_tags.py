"""
🏷️ New scene tags — the teacher's moment tags (and your labels) put onto your scenes.
Stage 3, step 1 (spec: claude/stage3-step1-spec.md). No Streamlit in here.

Every processed video's scenes get a second set of tags, worked out from the moments the Modal trial tagged inside
each scene (catalogue/trial_results/results.json), with your 🎯 Training Labels winning over the teacher:

  • a moment belongs to the scene it falls in by TIME (start ≤ t < end), so Reviewer splits and merges made since
    the pack still line up;
  • per moment: your label replaces the teacher's tags; "unsure" tags are left out; the newest tagging run is used
    when a video was tagged by more than one teacher;
  • per scene: a tag goes on when it's on at least half of the scene's moments; its share is kept
    (e.g. cowgirl 0.83);
  • scenes with no moment inside keep their old tags (marked as the fallback).

Stored in one separate file, CATALOGUE_DIR/scene_tags_new.json — no catalogue is touched — and synced to Drive:
    {"built", "export", "rule", "videos": {video_id: {scene_id: {"tags", "share", "moments", "labelled"}}}}

The 🏷️ Scene tags switch (Old / New) is per copy of the app (live and test each have their own), in the dot-file
CATALOGUE_DIR/.app_settings.json. With New, library_common.scene_tags() returns a scene's new tags — except where
you've corrected the scene in the Reviewer: your tags always win. The catalogue loader (catalogue_store) adds the
new tags to each scene in memory as "new_tags" / "new_tag_share".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from config import CATALOGUE_DIR

FILE = CATALOGUE_DIR / "scene_tags_new.json"
FILE_STEM = FILE.stem
SETTINGS = CATALOGUE_DIR / ".app_settings.json"
SHARE_NEEDED = 0.5            # a tag goes on a scene when it's on at least half of the scene's moments
TEACHER_ORDER = ("32b", "32b-fp8", "8b")       # when a video's tagging runs carry no times, the first found wins

_CACHE = {}


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def _read_settings() -> dict:
    try:
        stamp = SETTINGS.stat().st_mtime_ns
    except OSError:
        return {}
    if _CACHE.get("settings_stamp") != stamp:
        try:
            data = json.loads(SETTINGS.read_text())
        except (OSError, ValueError):
            data = {}
        _CACHE.update(settings_stamp=stamp, settings=data if isinstance(data, dict) else {})
    return _CACHE.get("settings") or {}


def mode() -> str:
    """"new" when the app uses the new scene tags, else "old". Checked often (once per scene when the library
    is read), so the file is looked at no more than once a second."""
    now = time.monotonic()
    if now - _CACHE.get("mode_at", -9.0) > 1.0:
        _CACHE["mode"] = "new" if _read_settings().get("scene_tags") == "new" else "old"
        _CACHE["mode_at"] = now
    return _CACHE["mode"]


def set_mode(value: str) -> None:
    data = dict(_read_settings())
    data["scene_tags"] = "new" if value == "new" else "old"
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS.with_name(SETTINGS.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, SETTINGS)
    _CACHE.pop("mode_at", None)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

def load() -> dict:
    """The built new tags (re-read when the file changes), or {} if there are none."""
    try:
        stamp = FILE.stat().st_mtime_ns
    except OSError:
        return {}
    if _CACHE.get("file_stamp") != stamp:
        try:
            data = json.loads(FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        _CACHE.update(file_stamp=stamp, file=data if isinstance(data, dict) else {})
    return _CACHE.get("file") or {}


def signature() -> tuple:
    """What the library's tags depend on beyond the catalogue files: the switch and the file's version.
    The catalogue loader re-reads everything when this changes."""
    try:
        stamp = FILE.stat().st_mtime_ns
    except OSError:
        stamp = None
    return (mode(), stamp)


def video_part(video_id: str):
    """A video's new tags as the recommendation searches see them: its scenes' new tags when the switch is on
    New (so a change to them makes its searches stale), else None."""
    if mode() != "new":
        return None
    v = (load().get("videos") or {}).get(video_id)
    return {sid: e.get("tags") for sid, e in sorted(v.items())} if v else None


def annotate(video_id: str, cat: dict) -> dict:
    """Add each scene's new tags to the catalogue in memory ("new_tags", "new_tag_share", "new_tag_moments").
    Scenes without any keep none (they fall back to their old tags). Returns cat."""
    v = (load().get("videos") or {}).get(video_id) or {}
    for scene in cat.get("scenes") or []:
        e = v.get(str(scene.get("scene_id")))
        if e is None:
            for k in ("new_tags", "new_tag_share", "new_tag_moments"):
                scene.pop(k, None)
            continue
        scene["new_tags"] = list(e.get("tags") or [])
        scene["new_tag_share"] = dict(e.get("share") or {})
        scene["new_tag_moments"] = int(e.get("moments") or 0)
    return cat


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _tc(tc) -> float:
    h, m, s = str(tc).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _scene_times(cat: dict) -> list:
    """[(start, end, scene)] in time order (scenes without readable times are left out)."""
    out = []
    for s in cat.get("scenes") or []:
        try:
            out.append((_tc(s["start_tc"]), _tc(s["end_tc"]), s))
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda x: x[0])


def _teacher_slot(video: dict) -> str | None:
    """The video's newest tagging run (by when it was written), else the first of TEACHER_ORDER it has."""
    runs = ((video.get("run") or {}).get("tagging") or {})
    timed = [(r.get("at"), slot) for slot, r in runs.items() if isinstance(r, dict) and r.get("at")]
    if timed:
        return max(timed)[1]
    have = set()
    for m in video.get("moments") or []:
        have |= set((m.get("teachers") or {}).keys())
    return next((k for k in TEACHER_ORDER if k in have), next(iter(sorted(have)), None))


def _old_tags(scene: dict) -> list:
    """The tags a scene has with the switch on Old: your corrected tags, else its own tags ("auto_tags" is the
    file's own tags in the catalogue loader's copy, where "tags" follows the switch)."""
    if scene.get("corrected_tags") is not None:
        return list(scene["corrected_tags"])
    return list(scene["auto_tags"] if "auto_tags" in scene else scene.get("tags") or [])


def build(results: dict, label_items: list, catalogues: dict) -> tuple:
    """(data to save, summary). results: trial_results.load(); label_items: training_labels.load_labels();
    catalogues: {video_id: catalogue} as they are now (the raw files — times as the Reviewer left them)."""
    import trial_results as TR
    videos = TR.with_saved_labels((results or {}).get("videos") or {}, label_items or [])
    out, summary = {}, {"videos": 0, "scenes_new": 0, "scenes_fallback": 0, "scenes_changed": 0,
                        "moments_used": 0, "moments_labelled": 0, "videos_missing": 0}
    for vid, v in sorted(videos.items()):
        cat = catalogues.get(vid)
        if cat is None:
            summary["videos_missing"] += 1
            continue
        slot = _teacher_slot(v)
        scenes = _scene_times(cat)
        bucket = {}
        for m in v.get("moments") or []:
            t = float(m.get("t") or 0.0)
            if m.get("mine") is not None:
                tags, labelled = list(m["mine"]), True
            else:
                tch = (m.get("teachers") or {}).get(slot) if slot else None
                if not tch or tch.get("tags") is None:
                    continue                   # no reading for this moment
                tags, labelled = list(tch.get("tags") or []), False
            hit = next((s for a, b, s in scenes if a <= t < b), None)
            if hit is None and scenes and abs(t - scenes[-1][1]) < 1e-6:
                hit = scenes[-1][2]            # a moment on the very end of the last scene
            if hit is None:
                continue
            e = bucket.setdefault(str(hit["scene_id"]), {"counts": {}, "moments": 0, "labelled": 0})
            e["moments"] += 1
            e["labelled"] += int(labelled)
            for tag in dict.fromkeys(tags):
                e["counts"][tag] = e["counts"].get(tag, 0) + 1
            summary["moments_used"] += 1
            summary["moments_labelled"] += int(labelled)
        if not bucket:
            continue
        summary["videos"] += 1
        vout = {}
        for a, b, s in scenes:
            e = bucket.get(str(s["scene_id"]))
            if e is None:
                summary["scenes_fallback"] += 1
                continue
            share = {t: round(c / e["moments"], 2) for t, c in e["counts"].items()}
            tags = sorted((t for t, x in share.items() if x >= SHARE_NEEDED - 1e-9), key=lambda t: (-share[t], t))
            vout[str(s["scene_id"])] = {"tags": tags, "share": dict(sorted(share.items(), key=lambda kv: -kv[1])),
                                        "moments": e["moments"], "labelled": e["labelled"]}
            summary["scenes_new"] += 1
            if set(tags) != set(_old_tags(s)):
                summary["scenes_changed"] += 1
        out[vid] = vout
    data = {"built": time.strftime("%Y-%m-%d %H:%M"), "export": (results or {}).get("created"),
            "rule": {"share_needed": SHARE_NEEDED}, "summary": summary, "videos": out}
    return data, summary


def save(data: dict) -> None:
    """Write atomically and queue it for Drive."""
    from library_common import mark_pending
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_name(FILE.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, FILE)
    mark_pending(FILE_STEM)


def raw_catalogues() -> dict:
    """{video_id: catalogue} straight from the files (scene times as they are now)."""
    import catalogue_store as CS
    out = {}
    for p in CS.catalogue_paths():
        try:
            cat = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        if CS.is_video_catalogue(cat):
            out[cat["video_id"]] = cat
    return out


# ---------------------------------------------------------------------------
# 🆚 Comparing old and new
# ---------------------------------------------------------------------------

def compare_rows(catalogues: dict, data: dict = None) -> list:
    """One row per scene of every built video: {"video_id", "scene_id", "start", "end", "old", "new" (None =
    fallback), "added", "removed", "share", "moments", "labelled", "corrected"}."""
    data = data if data is not None else load()
    rows = []
    for vid, v in sorted((data.get("videos") or {}).items()):
        cat = catalogues.get(vid)
        if cat is None:
            continue
        for a, b, s in _scene_times(cat):
            e = v.get(str(s["scene_id"]))
            old = _old_tags(s)
            new = None if e is None else list(e.get("tags") or [])
            rows.append({"video_id": vid, "scene_id": s["scene_id"], "start": a, "end": b, "old": old, "new": new,
                         "added": [] if new is None else [t for t in new if t not in old],
                         "removed": [] if new is None else [t for t in old if t not in new],
                         "share": dict((e or {}).get("share") or {}), "moments": int((e or {}).get("moments") or 0),
                         "labelled": int((e or {}).get("labelled") or 0),
                         "corrected": s.get("corrected_tags") is not None})
    return rows


def tag_changes(rows: list) -> list:
    """[(tag, added on n scenes, removed from n scenes)] — the tags that changed most first."""
    add, rem = {}, {}
    for r in rows:
        for t in r["added"]:
            add[t] = add.get(t, 0) + 1
        for t in r["removed"]:
            rem[t] = rem.get(t, 0) + 1
    tags = set(add) | set(rem)
    return sorted(((t, add.get(t, 0), rem.get(t, 0)) for t in tags), key=lambda x: (-(x[1] + x[2]), x[0]))


def similar_scenes(ref: tuple, catalogues: dict, data: dict, which: str, top: int = 6) -> list:
    """🔎 More like this for one scene, with the old or the new tags: [(similarity 0–1, video_id, scene_id,
    tags, shared)] from other videos. Scenes are compared on their tags (rarer tags count more) and their
    movement level — the movement is the same both ways, so the two lists differ only by the tags."""
    import math
    import similarity as SIM
    videos = data.get("videos") or {}

    def tags_of(vid, s):
        if which == "new" and s.get("corrected_tags") is None:
            e = (videos.get(vid) or {}).get(str(s["scene_id"]))
            if e is not None:
                return list(e.get("tags") or [])
        return _old_tags(s)

    scenes = [(vid, s) for vid, cat in catalogues.items() for s in cat.get("scenes") or [] if not s.get("excluded")]
    df = {}
    for vid, s in scenes:
        for t in {t.lower() for t in tags_of(vid, s)}:
            df[t] = df.get(t, 0) + 1
    n = len(scenes)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    rvid, rsid = ref
    rs = next((s for vid, s in scenes if vid == rvid and s["scene_id"] == rsid), None)
    if rs is None:
        return []
    rtags = {t.lower() for t in tags_of(rvid, rs)}
    levels = [float(s.get("motion_intensity") or 0.0) for _, s in scenes]
    spread = (max(levels) - min(levels)) or 1.0
    rlev = float(rs.get("motion_intensity") or 0.0)
    out = []
    for vid, s in scenes:
        if vid == rvid:
            continue
        tsim, shared = SIM._tag_jaccard(rtags, {t.lower() for t in tags_of(vid, s)}, idf)
        move = math.exp(-abs(float(s.get("motion_intensity") or 0.0) - rlev) / (0.15 * spread))
        out.append((0.7 * tsim + 0.3 * move, vid, s["scene_id"], list(dict.fromkeys(tags_of(vid, s))), shared))
    out.sort(key=lambda r: -r[0])
    chosen, per = [], {}
    for r in out:
        if per.get(r[1], 0) >= 2:
            continue
        chosen.append(r)
        per[r[1]] = per.get(r[1], 0) + 1
        if len(chosen) >= top:
            break
    return chosen
