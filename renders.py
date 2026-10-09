"""
Saved renders (the Render Preview MP4s in PREVIEW_DIR) and their metadata.

Every render gets a sidecar JSON next to it (same name, .json) written by
render_preview.render_plan_dict(): the track, length, the videos and clips it
used, the planner settings and the full plan. Renders made before this existed
get a smaller sidecar the first time the Media Library lists them (length,
resolution and size from ffprobe; track guessed from the file name; videos
unknown).

The sidecar also holds what you set in the Media Library's Renders tab:
rating, notes and "keep" (kept renders can't be deleted until unkept).

Renders made from the Compilation Planner also store the planner's full
project state ("project", the same dict as a saved project), so a render can be
reopened in the planner exactly as it was. Renders without it can still be
reopened with the settings recovered from their plan (project_from_plan).

No Streamlit here — render_preview.py (also run from the command line) uses it.
"""

import datetime
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from config import PREVIEW_DIR

SIDECAR_SCHEMA = 1
THUMB_DIR_NAME = ".thumbs"
_PREVIEW_NAME = re.compile(r"^temp-Video-preview-(?P<track>.+)\.(?P<num>\d{3})\.mp4$")

# Planner settings worth showing for a render (all come from the exported plan).
SETTING_KEYS = [
    "matching_mode", "split_screen_enabled", "min_clips", "max_clips", "ramp_start", "ramp_end",
    "density_contrast", "vary_clip_count", "clip_count_seed", "min_clip_len_sec",
    "allow_same_video", "sequential_video_order", "tag_filter", "segmentation",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def render_files() -> list:
    """Every render MP4 (top level of PREVIEW_DIR only — snippets/ is separate)."""
    if not PREVIEW_DIR.exists():
        return []
    return sorted(p for p in PREVIEW_DIR.glob("*.mp4") if p.is_file())


def sidecar_path(mp4: Path) -> Path:
    return Path(mp4).with_suffix(".json")


def thumb_path(mp4: Path) -> Path:
    mp4 = Path(mp4)
    return mp4.parent / THUMB_DIR_NAME / f"{mp4.stem}.jpg"


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def probe(mp4: Path) -> dict:
    """Length (s), width and height of a video file via ffprobe. {} if it can't be read."""
    if shutil.which("ffprobe") is None:
        return {}
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration", "-of", "json", str(mp4)],
            capture_output=True, text=True, timeout=30)
        info = json.loads(proc.stdout or "{}")
    except (subprocess.SubprocessError, ValueError, OSError):
        return {}
    stream = (info.get("streams") or [{}])[0]
    out = {}
    try:
        out["duration_sec"] = round(float(info.get("format", {}).get("duration")), 3)
    except (TypeError, ValueError):
        pass
    if stream.get("width"):
        out["width"], out["height"] = int(stream["width"]), int(stream["height"])
    return out


def summarise_plan(plan: dict) -> dict:
    """Videos and clips a plan's timeline uses, in timeline order of first use."""
    videos = {}
    clip_count = split_segments = 0
    timeline = plan.get("timeline") or []
    for entry in timeline:
        slots = entry.get("scenes") or []
        if len(slots) > 1:
            split_segments += 1
        for slot in slots:
            for link in slot.get("chain") or []:
                vid = link.get("video_id")
                if not vid:
                    continue
                v = videos.setdefault(vid, {"video_id": vid, "clips": 0, "seconds": 0.0})
                v["clips"] += 1
                v["seconds"] = round(v["seconds"] + float(link.get("clip_duration_sec") or 0), 3)
                clip_count += 1
    return {
        "segments": len(timeline),
        "split_segments": split_segments,
        "clip_count": clip_count,
        "videos": list(videos.values()),
    }


def _user_fields(old: dict) -> dict:
    u = (old or {}).get("user") or {}
    return {"rating": int(u.get("rating") or 0), "notes": u.get("notes") or "", "keep": bool(u.get("keep"))}


def write_render_metadata(output_path, plan: dict, render_seconds: float = None,
                          source_stats: dict = None, app_version: str = None,
                          extra: dict = None) -> Path:
    """Write the sidecar JSON for a finished render. Keeps any rating/notes/keep
    already on an older sidecar of the same name. extra: more keys to store
    (e.g. "project" — the planner state). Returns the sidecar path."""
    mp4 = Path(output_path)
    side = sidecar_path(mp4)
    old = {}
    if side.exists():
        try:
            old = json.loads(side.read_text())
        except (ValueError, OSError):
            old = {}
    meta = {
        "schema": SIDECAR_SCHEMA,
        "render_file": mp4.name,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "app_version": app_version,
        "render_seconds": round(render_seconds, 1) if render_seconds is not None else None,
        "sources": source_stats or {},
        "track": {
            "track_id": plan.get("track_id"),
            "duration_sec": plan.get("track_duration_sec"),
            "bpm": plan.get("bpm"),
        },
        **probe(mp4),
        "size_bytes": mp4.stat().st_size if mp4.exists() else None,
        **summarise_plan(plan),
        "settings": {k: plan[k] for k in SETTING_KEYS if k in plan},
        "plan": plan,
    }
    for k, v in (extra or {}).items():
        if k not in meta:
            meta[k] = v
    meta["user"] = _user_fields(old)
    _write_json(side, meta)
    return side


def _backfill(mp4: Path) -> dict:
    """Sidecar for a render made before sidecars existed."""
    m = _PREVIEW_NAME.match(mp4.name)
    st = mp4.stat()
    meta = {
        "schema": SIDECAR_SCHEMA,
        "render_file": mp4.name,
        "created": datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
        "backfilled": True,
        "track": {"track_id": m.group("track") if m else None},
        **probe(mp4),
        "size_bytes": st.st_size,
        "segments": None, "split_segments": None, "clip_count": None, "videos": [],
        "settings": {}, "plan": None,
        "user": _user_fields({}),
    }
    try:
        _write_json(sidecar_path(mp4), meta)
    except OSError:
        pass
    return meta


def load_render(mp4: Path) -> dict:
    """Sidecar for one render (backfilled on first read if missing), plus
    "path" (the MP4) and fresh "size_bytes"."""
    mp4 = Path(mp4)
    side = sidecar_path(mp4)
    meta = None
    if side.exists():
        try:
            meta = json.loads(side.read_text())
        except (ValueError, OSError):
            meta = None
    if not isinstance(meta, dict):
        meta = _backfill(mp4)
    meta["path"] = mp4
    try:
        meta["size_bytes"] = mp4.stat().st_size
    except OSError:
        pass
    meta.setdefault("user", _user_fields({}))
    return meta


def list_renders() -> list:
    return [load_render(p) for p in render_files()]


def update_user_fields(mp4: Path, **fields) -> None:
    """Set rating / notes / keep on a render's sidecar."""
    meta = load_render(mp4)
    meta.pop("path", None)
    user = _user_fields(meta)
    user.update({k: v for k, v in fields.items() if k in ("rating", "notes", "keep")})
    meta["user"] = user
    _write_json(sidecar_path(mp4), meta)


def delete_render(mp4: Path) -> None:
    """Remove a render, its sidecar and its thumbnail."""
    mp4 = Path(mp4)
    for p in (mp4, sidecar_path(mp4), thumb_path(mp4)):
        p.unlink(missing_ok=True)


def make_thumbnail(mp4: Path, at_fraction: float = 0.3) -> Path | None:
    """A JPEG frame from the render (cached in PREVIEW_DIR/.thumbs). None on failure."""
    mp4 = Path(mp4)
    out = thumb_path(mp4)
    if out.exists() and out.stat().st_mtime >= mp4.stat().st_mtime:
        return out
    if shutil.which("ffmpeg") is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = probe(mp4).get("duration_sec") or 0
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{dur * at_fraction:.2f}", "-i", str(mp4),
                        "-frames:v", "1", "-vf", "scale=320:-2", str(out)],
                       capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    return out if out.exists() else None


# ---------------------------------------------------------------------------
# Reopening in the planner
# ---------------------------------------------------------------------------

# Plan key -> planner session key, for renders that don't carry a project.
_PLAN_TO_PROJECT = {
    "track_id": "track_id", "matching_mode": "matching_mode",
    "split_screen_enabled": "split_screen_enabled", "min_clips": "min_clips", "max_clips": "max_clips",
    "density_contrast": "density_contrast", "vary_clip_count": "vary_count",
    "clip_count_seed": "count_seed", "min_clip_len_sec": "min_clip_len_sec",
    "allow_same_video": "allow_same_video", "sequential_video_order": "sequential_mode",
    "selected_videos": "committed_selected_videos", "tag_filter": "committed_tag_filter",
}
_SEG_METHOD_KEY = {"method": "segmentation_method"}


def project_from_plan(plan: dict) -> dict:
    """Planner settings recovered from an exported plan (no matching progress —
    the planner rebuilds the matches from these settings)."""
    data = {"_version": 1, "_from_plan": True}
    for src, dst in _PLAN_TO_PROJECT.items():
        if plan.get(src) is not None:
            data[dst] = plan[src]
    if plan.get("ramp_start") is not None and plan.get("ramp_end") is not None:
        data["ramp_range"] = [plan["ramp_start"], plan["ramp_end"]]
    seg = plan.get("segmentation") or {}
    for k, v in seg.items():
        if k == "method":
            if v and v != "legacy":
                data["segmentation_method"] = v
        else:
            data[k] = v
    return data


def reopen_state(meta: dict) -> tuple:
    """(project dict, "full" | "settings") to load into the planner for this
    render, or (None, None) if it has neither a project nor a plan."""
    if isinstance(meta.get("project"), dict):
        return meta["project"], "full"
    if isinstance(meta.get("plan"), dict) and meta["plan"].get("track_id"):
        return project_from_plan(meta["plan"]), "settings"
    return None, None


# ---------------------------------------------------------------------------
# Comparing renders
# ---------------------------------------------------------------------------

def compare_settings(a: dict, b: dict) -> list:
    """[(setting, value in a, value in b)] for planner settings that differ."""
    sa, sb = a.get("settings") or {}, b.get("settings") or {}
    rows = []
    for k in sorted(set(sa) | set(sb)):
        va, vb = sa.get(k), sb.get(k)
        if isinstance(va, dict) or isinstance(vb, dict):
            va, vb = va or {}, vb or {}
            for kk in sorted(set(va) | set(vb)):
                if va.get(kk) != vb.get(kk):
                    rows.append((f"{k}.{kk}", va.get(kk), vb.get(kk)))
        elif va != vb:
            rows.append((k, va, vb))
    return rows


def compare_videos(a: dict, b: dict) -> list:
    """[(video_id, seconds in a, seconds in b)] for every video in either render."""
    sa = {v["video_id"]: v["seconds"] for v in a.get("videos") or []}
    sb = {v["video_id"]: v["seconds"] for v in b.get("videos") or []}
    return sorted(((vid, sa.get(vid, 0.0), sb.get(vid, 0.0)) for vid in set(sa) | set(sb)),
                  key=lambda r: -(r[1] + r[2]))


# ---------------------------------------------------------------------------
# Across renders
# ---------------------------------------------------------------------------

def usage_counts(renders: list) -> tuple:
    """(videos, tracks): Counters of how many renders used each video / track."""
    videos, tracks = Counter(), Counter()
    for r in renders:
        for v in r.get("videos") or []:
            videos[v["video_id"]] += 1
        tid = (r.get("track") or {}).get("track_id")
        if tid and not r.get("backfilled"):
            tracks[tid] += 1
    return videos, tracks


def clips_outside_ranges(meta: dict, ranges: dict) -> list:
    """Clips in this render that fall outside their video's current library
    range — the render no longer matches what the planner would pick."""
    out = []
    for entry in (meta.get("plan") or {}).get("timeline") or []:
        for slot in entry.get("scenes") or []:
            for link in slot.get("chain") or []:
                rng = ranges.get(link.get("video_id"))
                if not rng:
                    continue
                start = float(link.get("clip_start_sec") or 0)
                end = start + float(link.get("clip_duration_sec") or 0)
                if start < rng[0] - 0.05 or end > rng[1] + 0.05:
                    out.append(link)
    return out


def format_size(n) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
