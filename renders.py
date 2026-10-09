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

Google Drive: renders are uploaded to scene-labeling/previews (with the
user's Drive connection — the service account can't write to personal Drive).
The sidecar records the Drive file id ("drive"). Once a render is on Drive its
MP4 can be cleared from the server; the sidecar and thumbnail stay here, and
playback, thumbnails and comparisons read the MP4 from Drive instead.

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
COMPARE_DIR_NAME = ".compare"
SETTINGS_FILE_NAME = ".render_settings.json"
DRIVE_FOLDER = "scene-labeling/previews"
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
    """Every render, as the path of its MP4 — which may no longer be on the
    server if it was cleared after uploading to Drive (its sidecar remains).
    Top level of PREVIEW_DIR only — snippets/ is separate."""
    if not PREVIEW_DIR.exists():
        return []
    names = {p.name for p in PREVIEW_DIR.glob("*.mp4") if p.is_file() and not p.name.startswith(".")}
    names |= {p.with_suffix(".mp4").name for p in PREVIEW_DIR.glob("*.json")
              if p.is_file() and not p.name.startswith(".")}
    return sorted(PREVIEW_DIR / n for n in names)


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

def probe(mp4: Path, input_args: list = None) -> dict:
    """Length (s), width and height of a video file via ffprobe (or of the
    ffmpeg-style input_args, e.g. a Drive stream). {} if it can't be read."""
    if shutil.which("ffprobe") is None:
        return {}
    src = [a for a in (input_args or []) if a != "-i"] if input_args else [str(mp4)]
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration", "-of", "json", *src],
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


def load_render(mp4: Path) -> dict | None:
    """Sidecar for one render (backfilled on first read if missing), plus
    "path" (the MP4), "local" (MP4 on the server) and fresh "size_bytes".
    None for a stray sidecar whose render is neither here nor on Drive."""
    mp4 = Path(mp4)
    side = sidecar_path(mp4)
    meta = None
    if side.exists():
        try:
            meta = json.loads(side.read_text())
        except (ValueError, OSError):
            meta = None
    local = mp4.exists()
    if not isinstance(meta, dict) or "render_file" not in meta:
        if not local:
            return None
        meta = _backfill(mp4)
    if not local and not on_drive(meta):
        return None
    meta["path"] = mp4
    meta["local"] = local
    if local:
        try:
            meta["size_bytes"] = mp4.stat().st_size
        except OSError:
            pass
    meta.setdefault("user", _user_fields({}))
    return meta


def list_renders() -> list:
    return [m for m in (load_render(p) for p in render_files()) if m]


def _save_meta(mp4: Path, meta: dict) -> None:
    meta = {k: v for k, v in meta.items() if k not in ("path", "local")}
    _write_json(sidecar_path(mp4), meta)


def update_user_fields(mp4: Path, **fields) -> None:
    """Set rating / notes / keep on a render's sidecar."""
    meta = load_render(mp4)
    if meta is None:
        return
    user = _user_fields(meta)
    user.update({k: v for k, v in fields.items() if k in ("rating", "notes", "keep")})
    meta["user"] = user
    _save_meta(mp4, meta)


# ---------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------

def on_drive(meta: dict) -> bool:
    return bool((meta or {}).get("drive", {}).get("file_id"))


def drive_preview_url(meta: dict) -> str | None:
    """Google Drive's own video player for this render (plays in the browser of
    anyone signed in to Google with access to the file — i.e. you)."""
    fid = (meta or {}).get("drive", {}).get("file_id")
    return f"https://drive.google.com/file/d/{fid}/preview" if fid else None


def load_settings() -> dict:
    try:
        data = json.loads((PREVIEW_DIR / SETTINGS_FILE_NAME).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def auto_clear_enabled() -> bool:
    """Clear a render's MP4 from the server as soon as it's safely on Drive (default on)."""
    return bool(load_settings().get("auto_clear", True))


def set_auto_clear(value: bool) -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    data = load_settings()
    data["auto_clear"] = bool(value)
    _write_json(PREVIEW_DIR / SETTINGS_FILE_NAME, data)


def upload_render(mp4: Path, token: dict) -> str | None:
    """Upload a render's MP4 (and its sidecar, as a backup) to Drive with the
    user's Drive connection. Checks the uploaded size matches before recording
    it. Returns None on success, error string on failure."""
    from drive_oauth import upload_file_with_oauth
    mp4 = Path(mp4)
    meta = load_render(mp4)
    if meta is None or not mp4.exists():
        return "The render isn't on the server."
    make_thumbnail(mp4)   # while the file is still here
    info, err = upload_file_with_oauth(token, mp4, DRIVE_FOLDER)
    if err:
        return err
    local_size = mp4.stat().st_size
    if info.get("size") and info["size"] != local_size:
        return f"Upload incomplete ({info['size']} of {local_size} bytes) — try again."
    meta["drive"] = {"file_id": info["id"], "size": local_size,
                     "uploaded": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    _save_meta(mp4, meta)
    sync_sidecar(mp4, token)
    return None


def sync_sidecar(mp4: Path, token: dict) -> str | None:
    """Copy the current sidecar (rating, notes, …) to Drive next to the MP4."""
    from drive_oauth import upload_file_with_oauth
    side = sidecar_path(mp4)
    if not side.exists():
        return None
    info, err = upload_file_with_oauth(token, side, DRIVE_FOLDER)
    if err:
        return err
    meta = load_render(mp4)
    if meta is not None and on_drive(meta) and meta["drive"].get("sidecar_id") != info["id"]:
        meta["drive"]["sidecar_id"] = info["id"]
        _save_meta(mp4, meta)
    return None


def clear_local(mp4: Path) -> str | None:
    """Delete the server's copy of a render that's on Drive (sidecar and
    thumbnail stay). Returns None on success, error string otherwise."""
    mp4 = Path(mp4)
    meta = load_render(mp4)
    if meta is None:
        return "Unknown render."
    if not on_drive(meta):
        return "Not on Google Drive yet — upload it first."
    if mp4.exists():
        if meta["drive"].get("size") and meta["drive"]["size"] != mp4.stat().st_size:
            return "The Drive copy is a different size from the server copy — upload it again first."
        make_thumbnail(mp4)
        mp4.unlink()
    return None


def delete_render(mp4: Path, token: dict = None) -> str | None:
    """Remove a render everywhere: its Drive copy (to the Drive bin, so it can
    be recovered for 30 days), and the server's MP4, sidecar and thumbnail.
    Needs the Drive connection (token) if the render is on Drive. Returns None
    on success, error string (and nothing deleted) on failure."""
    mp4 = Path(mp4)
    meta = load_render(mp4) or {}
    if on_drive(meta):
        if not token:
            return "This render is on Google Drive — connect Google Drive to delete it."
        from drive_oauth import trash_file_with_oauth
        for fid in (meta["drive"].get("file_id"), meta["drive"].get("sidecar_id")):
            if fid:
                err = trash_file_with_oauth(token, fid)
                if err:
                    return err
    for p in (mp4, sidecar_path(mp4), thumb_path(mp4)):
        p.unlink(missing_ok=True)
    return None


def restore_from_drive() -> dict:
    """Bring the server's list of renders in line with Google Drive
    (scene-labeling/previews), read with the service account:
      • a render on Drive whose details file is missing here gets it back —
        downloaded from Drive, or rebuilt from the file itself if Drive has none;
      • a render that's here and also on Drive (e.g. uploaded by older versions)
        is marked as on Drive, so its server copy can be cleared.
    Never deletes anything. Returns {"restored", "linked", "errors"}."""
    from drive_sync import list_drive_folder, download_drive_file
    files, err = list_drive_folder(DRIVE_FOLDER)
    if err:
        return {"restored": 0, "linked": 0, "errors": [err]}
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    by_name = {f["name"]: f for f in files}
    restored = linked = 0
    errors = []
    for name, f in sorted(by_name.items()):
        if not name.endswith(".mp4") or name.startswith("."):
            continue
        mp4 = PREVIEW_DIR / name
        side = sidecar_path(mp4)
        size = int(f.get("size") or 0)
        modified = (f.get("modifiedTime") or "").replace("Z", "+00:00") or None
        drive_info = {"file_id": f["id"], "size": size, "uploaded": modified}
        js = by_name.get(side.name)
        if js:
            drive_info["sidecar_id"] = js["id"]

        meta = None
        if side.exists():
            try:
                meta = json.loads(side.read_text())
            except (ValueError, OSError):
                meta = None
        if isinstance(meta, dict) and "render_file" in meta:
            if on_drive(meta):
                if js and not meta["drive"].get("sidecar_id"):
                    meta["drive"]["sidecar_id"] = js["id"]
                    _save_meta(mp4, meta)
                continue
            if mp4.exists() and size and mp4.stat().st_size != size:
                continue   # a different file with the same name — leave it alone
            meta["drive"] = drive_info
            _save_meta(mp4, meta)
            linked += 1
            continue

        meta = None
        if js:
            tmp = side.with_name(side.name + ".download")
            try:
                download_drive_file(js["id"], tmp)
                meta = json.loads(tmp.read_text())
            except Exception as e:
                errors.append(f"{js['name']}: {e}")
            finally:
                tmp.unlink(missing_ok=True)
        if not isinstance(meta, dict) or "render_file" not in meta:
            m = _PREVIEW_NAME.match(name)
            meta = {
                "schema": SIDECAR_SCHEMA, "render_file": name, "backfilled": True,
                "created": modified,
                "track": {"track_id": m.group("track") if m else None},
                "size_bytes": size,
                "segments": None, "split_segments": None, "clip_count": None, "videos": [],
                "settings": {}, "plan": None, "user": _user_fields({}),
            }
            try:
                from drive_sync import access_token, stream_url
                meta.update(probe(mp4, ["-headers", f"Authorization: Bearer {access_token()}\r\n",
                                        "-i", stream_url(f["id"])]))
            except Exception:
                pass
        meta["drive"] = drive_info
        meta["user"] = _user_fields(meta)
        _save_meta(mp4, meta)
        restored += 1
    return {"restored": restored, "linked": linked, "errors": errors}


def _input_args(meta: dict) -> list:
    """ffmpeg input arguments for a render: the server's file, or a Drive stream
    (read with the service account, like render_preview does for sources)."""
    mp4 = Path(meta["path"])
    if mp4.exists():
        return ["-i", str(mp4)]
    if not on_drive(meta):
        raise FileNotFoundError(f"{mp4.name} is neither on the server nor on Google Drive")
    from drive_sync import access_token, stream_url
    return ["-reconnect", "1", "-reconnect_delay_max", "5", "-rw_timeout", "60000000",
            "-headers", f"Authorization: Bearer {access_token()}\r\n",
            "-i", stream_url(meta["drive"]["file_id"])]


def make_thumbnail(mp4: Path, at_fraction: float = 0.3) -> Path | None:
    """A JPEG frame from the render (cached in PREVIEW_DIR/.thumbs; read from
    Drive if the server copy has been cleared). None on failure."""
    mp4 = Path(mp4)
    out = thumb_path(mp4)
    if out.exists() and (not mp4.exists() or out.stat().st_mtime >= mp4.stat().st_mtime):
        return out
    if shutil.which("ffmpeg") is None:
        return None
    meta = load_render(mp4) if not mp4.exists() else {"path": mp4}
    if meta is None:
        return None
    try:
        args = _input_args(meta)
    except Exception:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = (meta.get("duration_sec") if not mp4.exists() else probe(mp4).get("duration_sec")) or 0
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{dur * at_fraction:.2f}", *args,
                        "-frames:v", "1", "-vf", "scale=320:-2", str(out)],
                       capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return None
    return out if out.exists() else None


def make_compare_video(a: dict, b: dict, audio_from: str = "A") -> tuple:
    """One video with render a on the left and b on the right, so both play in
    sync from a single play button. Sound comes from a or b (audio_from). Cached
    in PREVIEW_DIR/.compare (the last few are kept). Returns (path, None) or
    (None, error string)."""
    out_dir = PREVIEW_DIR / COMPARE_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(a['path']).stem}__vs__{Path(b['path']).stem}__{audio_from}.mp4"
    if out.exists() and out.stat().st_size > 0:
        out.touch()
        return out, None
    if shutil.which("ffmpeg") is None:
        return None, "ffmpeg isn't installed on this server."
    try:
        args_a, args_b = _input_args(a), _input_args(b)
    except Exception as e:
        return None, str(e)
    w, h = 640, 360
    fit = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
           f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25")
    # The shorter one holds its last frame until the longer one ends.
    longest = max(float(a.get("duration_sec") or 0), float(b.get("duration_sec") or 0))
    pad = f",tpad=stop_mode=clone:stop_duration={longest:.2f}" if longest else ""
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", *args_a, *args_b,
           "-filter_complex",
           f"[0:v]{fit}{pad}[l];[1:v]{fit}{pad}[r];[l][r]hstack=inputs=2[v]",
           "-map", "[v]", "-map", f"{0 if audio_from == 'A' else 1}:a:0?",
           "-t", f"{longest:.2f}" if longest else "36000",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-c:a", "aac",
           "-movflags", "+faststart", str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.SubprocessError as e:
        tmp.unlink(missing_ok=True)
        return None, str(e)
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        import re as _re
        return None, _re.sub(r"Bearer [A-Za-z0-9._\-]+", "Bearer ***", (proc.stderr or "ffmpeg failed")[-400:])
    tmp.replace(out)
    # Keep the 4 most recent comparisons.
    olds = sorted((p for p in out_dir.glob("*.mp4") if not p.name.endswith(".part.mp4")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for p in olds[4:]:
        p.unlink(missing_ok=True)
    return out, None


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
