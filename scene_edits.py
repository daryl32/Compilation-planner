"""
Manual scene edits for the Reviewer: split a scene at chosen times, combine
consecutive scenes, undo either, and suggest which scenes might need it.

Edits change the video's catalogue JSON directly (like tag corrections do), so
the Compilation Planner, Media Library and ranking pick them up with no extra
code. Each edited scene keeps what it replaced under "edit", so it can be undone:

  split   — the first part keeps the original scene_id and carries
            {"kind": "split", "original": <scene>, "part_ids": [...]};
            every part has "split_from": <original scene_id>.
            New parts get fresh ids (one above the highest id in the catalogue),
            so scene_id is unique but NOT chronological any more — sort scenes
            by start time (scene_start_sec), never by scene_id.
  combine — the combined scene keeps the first part's scene_id and carries
            {"kind": "merge", "originals": [<scene>, ...]}.

Nothing here touches Streamlit; the thumbnail helpers shell out to ffmpeg
or crop the timeline sprite.
"""

from __future__ import annotations

import copy
import math
import subprocess
from pathlib import Path

# Scenes closer together than this count as touching (scene detection's
# boundaries can differ by a frame or two from rounding).
CONTIGUOUS_TOLERANCE_SEC = 0.5
# Parts shorter than this aren't allowed when splitting.
MIN_PART_SEC = 0.2

DEFAULT_LONG_SEC = 120.0
DEFAULT_SHORT_SEC = 2.0


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def tc_to_seconds(tc: str) -> float:
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_tc(sec: float, like: str | None = None) -> str:
    """HH:MM:SS.mmm — at least 3 decimals, more if `like` (an existing
    timecode from the same catalogue) has more."""
    decimals = 3
    if like and "." in like:
        decimals = max(3, len(like.rsplit(".", 1)[1]))
    sec = max(0.0, float(sec))
    sec = round(sec, decimals)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    width = 2 + (decimals + 1 if decimals else 0)
    return f"{int(h):02d}:{int(m):02d}:{s:0{width}.{decimals}f}"


def scene_start_sec(scene: dict) -> float:
    return tc_to_seconds(scene["start_tc"])


def scene_end_sec(scene: dict) -> float:
    return tc_to_seconds(scene["end_tc"])


def scene_duration(scene: dict) -> float:
    return scene_end_sec(scene) - scene_start_sec(scene)


def sorted_scenes(scenes: list) -> list:
    """Chronological order (scene_id stops being chronological after a split)."""
    return sorted(scenes, key=lambda s: (scene_start_sec(s), s["scene_id"]))


def _effective_tags(scene: dict) -> list:
    if scene.get("corrected_tags") is not None:
        return list(scene["corrected_tags"])
    return list(scene.get("tags", []))


def _union(lists) -> list:
    out = []
    for lst in lists:
        for t in lst:
            if t not in out:
                out.append(t)
    return out


def scene_label(scene: dict) -> str:
    """'12', '12 (part 2 of 3)' or '14 (6 combined)' — for display."""
    sid = scene["scene_id"]
    edit = scene.get("edit") or {}
    if edit.get("kind") == "merge":
        return f"{sid} ({len(edit.get('originals', []))} combined)"
    if "split_from" in scene:
        return f"{sid} (split from {scene['split_from']})"
    return str(sid)


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------

class EditError(ValueError):
    """An edit that can't be made; the message is shown to the user."""


def _index_by_id(data: dict, scene_id) -> int:
    for i, s in enumerate(data["scenes"]):
        if s["scene_id"] == scene_id:
            return i
    raise EditError(f"Scene {scene_id} isn't in this catalogue any more.")


def _next_id(data: dict) -> int:
    """One above every id in use, including ids held inside undo records, so an
    undo can never bring back an id that clashes with a newer scene."""
    biggest = -1

    def walk(scene):
        nonlocal biggest
        if isinstance(scene.get("scene_id"), int):
            biggest = max(biggest, scene["scene_id"])
        edit = scene.get("edit") or {}
        if edit.get("original"):
            walk(edit["original"])
        for o in edit.get("originals", []):
            walk(o)

    for s in data["scenes"]:
        walk(s)
    return biggest + 1


def _slice_curve(curve: dict | None, offset_sec: float, dur_sec: float) -> dict | None:
    if not curve or not curve.get("values") or not curve.get("fps"):
        return None
    fps = float(curve["fps"])
    a = max(0, int(round(offset_sec * fps)))
    b = max(a, int(round((offset_sec + dur_sec) * fps)))
    out = dict(curve)
    out["values"] = list(curve["values"][a:b])
    return out


def _mean(values) -> float | None:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


def split_scene(data: dict, scene_id: int, cut_secs: list) -> list:
    """Split scene_id at the given absolute times (seconds into the video).
    Returns the ids of the parts in order. The first part keeps scene_id."""
    idx = _index_by_id(data, scene_id)
    scene = data["scenes"][idx]
    start, end = scene_start_sec(scene), scene_end_sec(scene)
    cuts = sorted({round(float(c), 3) for c in cut_secs})
    if not cuts:
        raise EditError("Add at least one cut point.")
    bounds = [start] + cuts + [end]
    for a, b in zip(bounds, bounds[1:]):
        if b - a < MIN_PART_SEC:
            raise EditError(f"Each part must be at least {MIN_PART_SEC:g}s long — "
                            "a cut is too close to another cut or to the scene's ends.")

    original = copy.deepcopy(scene)
    next_id = _next_id(data)
    has_frames = isinstance(scene.get("start_frame"), (int, float)) and isinstance(scene.get("end_frame"), (int, float))
    frames_per_sec = ((scene["end_frame"] - scene["start_frame"]) / (end - start)) if has_frames and end > start else None

    parts = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        part = copy.deepcopy(scene)
        part.pop("edit", None)
        part["scene_id"] = scene_id if i == 0 else next_id + i - 1
        part["start_tc"] = seconds_to_tc(a, like=scene["start_tc"])
        part["end_tc"] = seconds_to_tc(b, like=scene["end_tc"])
        if frames_per_sec is not None:
            part["start_frame"] = int(round(scene["start_frame"] + (a - start) * frames_per_sec))
            part["end_frame"] = int(round(scene["start_frame"] + (b - start) * frames_per_sec))
        curve = _slice_curve(scene.get("motion_curve"), a - start, b - a)
        if curve is not None:
            part["motion_curve"] = curve
            m = _mean(curve["values"])
            if m is not None:
                part["motion_intensity"] = round(m, 6)
        elif "motion_curve" in part:
            part.pop("motion_curve")
        if i > 0:
            part["thumbnail_paths"] = []  # filled in by the caller (make_part_thumbnail)
        part["reviewed"] = False
        part["split_from"] = scene_id
        parts.append(part)

    parts[0]["edit"] = {"kind": "split", "original": original, "part_ids": [p["scene_id"] for p in parts]}
    data["scenes"][idx:idx + 1] = parts
    return [p["scene_id"] for p in parts]


def check_mergeable(data: dict, scene_ids: list) -> list:
    """The scenes to combine, in time order — or EditError if they aren't
    consecutive, touching scenes."""
    if len(scene_ids) < 2:
        raise EditError("Pick at least two scenes to combine.")
    wanted = set(scene_ids)
    ordered = sorted_scenes(data["scenes"])
    positions = [i for i, s in enumerate(ordered) if s["scene_id"] in wanted]
    if len(positions) != len(wanted):
        raise EditError("Some of those scenes aren't in this catalogue any more.")
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise EditError("Only consecutive scenes can be combined.")
    group = [ordered[i] for i in positions]
    for a, b in zip(group, group[1:]):
        if abs(scene_start_sec(b) - scene_end_sec(a)) > CONTIGUOUS_TOLERANCE_SEC:
            raise EditError(f"Scenes {a['scene_id']} and {b['scene_id']} don't touch "
                            f"(there's a gap between them), so they can't be combined.")
    return group


def merge_scenes(data: dict, scene_ids: list) -> int:
    """Combine consecutive scenes into one. Returns the combined scene's id
    (the first scene's id)."""
    group = check_mergeable(data, scene_ids)
    first, last = group[0], group[-1]
    merged = copy.deepcopy(first)
    merged.pop("edit", None)
    merged.pop("split_from", None)
    merged["end_tc"] = last["end_tc"]
    if "end_frame" in last:
        merged["end_frame"] = last["end_frame"]

    merged["tags"] = _union(s.get("tags", []) for s in group)
    if any(s.get("corrected_tags") is not None for s in group):
        merged["corrected_tags"] = _union(_effective_tags(s) for s in group)
    else:
        merged.pop("corrected_tags", None)
    merged["thumbnail_paths"] = _union(s.get("thumbnail_paths") or [] for s in group)

    durations = [max(scene_duration(s), 1e-6) for s in group]
    if all("motion_intensity" in s for s in group):
        merged["motion_intensity"] = round(
            sum(s["motion_intensity"] * d for s, d in zip(group, durations)) / sum(durations), 6)

    curves = [s.get("motion_curve") for s in group]
    if all(c and c.get("values") and c.get("fps") for c in curves) and len({float(c["fps"]) for c in curves}) == 1:
        merged["motion_curve"] = dict(curves[0])
        merged["motion_curve"]["values"] = [v for c in curves for v in c["values"]]
    else:
        merged.pop("motion_curve", None)

    for flag in ("excluded", "intro_candidate", "outro_candidate"):
        merged[flag] = all(bool(s.get(flag)) for s in group)
    merged["reviewed"] = False
    merged["edit"] = {"kind": "merge", "originals": [copy.deepcopy(s) for s in group]}

    ids = {s["scene_id"] for s in group}
    insert_at = _index_by_id(data, first["scene_id"])
    data["scenes"][insert_at] = merged
    data["scenes"] = [s for s in data["scenes"] if s is merged or s["scene_id"] not in ids]
    return merged["scene_id"]


def undo_target(data: dict, scene_id) -> dict | None:
    """The scene holding the undo record that `scene_id` belongs to (itself for
    a combine or a split's first part; the first part for other split parts)."""
    scene = data["scenes"][_index_by_id(data, scene_id)]
    if scene.get("edit"):
        return scene
    if "split_from" in scene:
        for s in data["scenes"]:
            edit = s.get("edit") or {}
            if edit.get("kind") == "split" and scene_id in edit.get("part_ids", []):
                return s
        raise EditError(f"The first part of this split (scene {scene['split_from']}) has been "
                        "edited again since — undo that first.")
    return None


def undo_edit(data: dict, scene_id) -> str:
    """Undo the split or combine that produced scene_id. Returns a message."""
    holder = undo_target(data, scene_id)
    if holder is None:
        raise EditError(f"Scene {scene_id} hasn't been split or combined.")
    edit = holder["edit"]
    if edit["kind"] == "merge":
        idx = _index_by_id(data, holder["scene_id"])
        originals = copy.deepcopy(edit["originals"])
        data["scenes"][idx:idx + 1] = originals
        return f"Scene {holder['scene_id']} un-combined back into {len(originals)} scenes."
    # split
    part_ids = edit["part_ids"]
    by_id = {s["scene_id"]: s for s in data["scenes"]}
    untouched = all(
        pid in by_id and by_id[pid].get("split_from") == holder["scene_id"]
        and (pid == holder["scene_id"] or not by_id[pid].get("edit"))
        for pid in part_ids
    )
    if not untouched:
        raise EditError("Some parts of this split have since been combined with other scenes — "
                        "undo that combine first.")
    idx = _index_by_id(data, holder["scene_id"])
    data["scenes"][idx] = copy.deepcopy(edit["original"])
    ids = set(part_ids) - {holder["scene_id"]}
    data["scenes"] = [s for s in data["scenes"] if s["scene_id"] not in ids]
    return f"Split of scene {holder['scene_id']} undone ({len(part_ids)} parts rejoined)."


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

def suggest_edits(scenes: list, long_sec: float = DEFAULT_LONG_SEC, short_sec: float = DEFAULT_SHORT_SEC,
                  dismissed=()) -> dict:
    """{"split": [scene_id, ...], "merge": [[scene_id, ...], ...]}.

    split — scenes longer than long_sec.
    merge — runs of 2+ consecutive, touching scenes each shorter than short_sec
            (i.e. a short scene whose neighbour is also short). A run stops at a
            gap or where the excluded setting changes.
    dismissed — keys from suggestion_key() the user has dismissed."""
    dismissed = set(dismissed)
    ordered = sorted_scenes(scenes)
    split = [s["scene_id"] for s in ordered
             if scene_duration(s) > long_sec and suggestion_key("split", [s["scene_id"]]) not in dismissed]

    merge, run = [], []

    def close_run():
        if len(run) >= 2:
            ids = [s["scene_id"] for s in run]
            if suggestion_key("merge", ids) not in dismissed:
                merge.append(ids)

    for s in ordered:
        if scene_duration(s) < short_sec:
            if run and (abs(scene_start_sec(s) - scene_end_sec(run[-1])) > CONTIGUOUS_TOLERANCE_SEC
                        or bool(s.get("excluded")) != bool(run[-1].get("excluded"))):
                close_run()
                run = []
            run.append(s)
        else:
            close_run()
            run = []
    close_run()
    return {"split": split, "merge": merge}


def suggestion_key(kind: str, scene_ids: list) -> str:
    return f"{kind}:{'-'.join(str(i) for i in scene_ids)}"


# ---------------------------------------------------------------------------
# Thumbnails for new parts
# ---------------------------------------------------------------------------

def sprite_tile(catalogue: dict, sprite_path: Path, at_sec: float):
    """PIL image of the timeline-sprite tile nearest at_sec, or None."""
    meta = catalogue.get("timeline_thumbnails")
    if not meta or not Path(sprite_path).exists():
        return None
    try:
        from PIL import Image
        with Image.open(sprite_path) as img:
            tile_w, tile_h = meta["tile_width"], meta["tile_height"]
            columns, count, interval = meta["columns"], meta["count"], meta["interval_sec"]
            idx = min(count - 1, max(0, int(round(at_sec / interval))))
            r, c = divmod(idx, columns)
            return img.crop((c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)).copy()
    except Exception:
        return None


def grab_frame(video_path: Path, at_sec: float, dest: Path, width: int = 640) -> bool:
    """Save one frame of video_path at at_sec as a JPEG. True on success."""
    if not Path(video_path).exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(0.0, at_sec):.3f}", "-i", str(video_path),
           "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(dest)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def make_part_thumbnail(catalogue: dict, scene: dict, thumbs_dir: Path, video_paths: list,
                        sprite_path: Path) -> Path | None:
    """Make a thumbnail for a new scene part, from the first video in
    video_paths that exists (preview copy, then source), else from the
    timeline sprite. Sets scene["thumbnail_paths"]; returns the file or None."""
    start, end = scene_start_sec(scene), scene_end_sec(scene)
    at = start + min(1.0, (end - start) / 2)
    dest = thumbs_dir / f"scene_{int(scene['scene_id']):04d}_manual.jpg"
    made = any(grab_frame(Path(p), at, dest) for p in video_paths if p)
    if not made:
        tile = sprite_tile(catalogue, sprite_path, at)
        if tile is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tile.convert("RGB").save(dest, "JPEG", quality=88)
            made = True
    if made:
        scene["thumbnail_paths"] = [dest.name]
        return dest
    return None


def cut_suggestions_from_sprite(catalogue: dict, sprite_path: Path, scene: dict, top: int = 3,
                                min_gap_sec: float = 5.0) -> list:
    """Times inside a long scene where the picture changes most between
    neighbouring timeline-sprite tiles — likely places for a cut."""
    meta = catalogue.get("timeline_thumbnails")
    if not meta or not Path(sprite_path).exists():
        return []
    try:
        import numpy as np
        from PIL import Image
        interval = float(meta["interval_sec"])
        tile_w, tile_h, columns, count = meta["tile_width"], meta["tile_height"], meta["columns"], meta["count"]
        start, end = scene_start_sec(scene), scene_end_sec(scene)
        first = max(0, math.ceil(start / interval))
        last = min(count - 1, math.floor(end / interval))
        if last - first < 2:
            return []
        with Image.open(sprite_path) as img:
            img = img.convert("RGB")
            hists = []
            for idx in range(first, last + 1):
                r, c = divmod(idx, columns)
                tile = img.crop((c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h))
                arr = np.asarray(tile.resize((32, 32)), dtype=np.float32) / 255.0
                h = np.concatenate([np.histogram(arr[..., ch], bins=16, range=(0, 1))[0] for ch in range(3)])
                hists.append(h / max(h.sum(), 1))
        diffs = [(float(np.abs(hists[i + 1] - hists[i]).sum()), (first + i + 0.5) * interval)
                 for i in range(len(hists) - 1)]
        diffs.sort(reverse=True)
        chosen = []
        for score, t in diffs:
            if score < 0.3:
                break
            if t - start < min_gap_sec or end - t < min_gap_sec:
                continue
            if all(abs(t - c) >= min_gap_sec for c in chosen):
                chosen.append(round(t, 1))
            if len(chosen) >= top:
                break
        return sorted(chosen)
    except Exception:
        return []
