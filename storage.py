"""
Server disk space: what's using it, and clearing what can be fetched again.

What the app keeps on the server, and why:
  kept     catalogue JSONs, thumbnails, timeline sprites, audio JSONs, projects,
           plans — small, and every page reads them all the time.
  kept     480p preview copies (proxies) — scrubbing and Reviewer frames need
           them fast; ~1/10th the size of the originals.
  temporary  original videos / music downloaded from Google Drive (a render whose
           stream failed, the old "download first" render mode, Choreography
           previews, audio analysis). They land in /root/data/<Drive folder>/
           and can always be streamed or downloaded again, so they're cleared
           once they haven't been used for KEEP_DOWNLOADS_DAYS (config.py,
           default 3; 0 = never automatically) — by the 12-hourly Drive sync and
           from the Server page. A separately mounted disk under /root/data
           (e.g. a Hetzner Volume for the originals) is never touched.
  temporary  local copies of renders already uploaded to Drive (they play from
           Drive), Reviewer frame grabs, half-finished downloads (*.part).

Deleting only ever happens on the Linux server: on the laptop the same paths
are the Google Drive folder itself, so every clean-up here is switched off.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import config
from config import CATALOGUE_DIR, AUDIO_DIR, PLANS_DIR, PREVIEW_DIR, PROJECTS_DIR

try:
    from auto_sync import PROXY_DIR
except Exception:
    PROXY_DIR = CATALOGUE_DIR.parent / "proxies"

# render_preview.colab_to_local puts every Drive original under here on Linux.
ORIGINALS_ROOT = Path("/root/data")


def _exists(path: Path) -> bool:
    """Path.exists that is False (not an error) when the folder can't be read —
    e.g. /root for a user other than root."""
    try:
        return path.exists()
    except OSError:
        return False
DATA_ROOT = CATALOGUE_DIR.parent
KEEP_DOWNLOADS_DAYS = float(getattr(config, "KEEP_DOWNLOADS_DAYS", 3))

MEDIA_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts",
             ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus"}
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus"}


def cleanup_allowed() -> bool:
    """Only on the Linux server, never on the Windows laptop (Drive-mounted paths)."""
    return os.name != "nt" and str(CATALOGUE_DIR).startswith("/root/")


def _app_folders() -> list:
    """Folders the app manages itself — never treated as downloaded originals."""
    return [Path(p).resolve() for p in
            (CATALOGUE_DIR, AUDIO_DIR, PLANS_DIR, PREVIEW_DIR, PROJECTS_DIR, PROXY_DIR, DATA_ROOT / "logs")]


def _inside(path: Path, folders) -> bool:
    p = path.resolve()
    return any(p == f or f in p.parents for f in folders)


def fmt_bytes(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit in ("B", "KB") else f"{n:,.1f} {unit}"
        n /= 1024


# ---------------------------------------------------------------------------
# What's using the space
# ---------------------------------------------------------------------------

def disk_usage(path: Path = None) -> dict:
    u = shutil.disk_usage(str(path or (ORIGINALS_ROOT if _exists(ORIGINALS_ROOT) else DATA_ROOT)))
    return {"total": u.total, "used": u.used, "free": u.free, "pct": 100.0 * u.used / u.total}


def du(path: Path) -> int | None:
    """Bytes used by a file or folder (None if it isn't there)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        out = subprocess.run(["du", "-sxb", str(path)], capture_output=True, text=True, timeout=120)
        return int(out.stdout.split()[0])
    except Exception:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def usage_breakdown() -> list:
    """[(label, path, bytes, note)] for the places that matter, biggest first."""
    rows = [
        ("Video catalogues (JSON)", None, sum(p.stat().st_size for p in CATALOGUE_DIR.glob("*.json"))
         if CATALOGUE_DIR.exists() else None, "Kept — every page reads these"),
        ("Scene thumbnails", CATALOGUE_DIR / "thumbnails", None, "Kept"),
        ("Timeline sprites", CATALOGUE_DIR / "timeline_sprites", None, "Kept"),
        ("Audio catalogues (JSON)", AUDIO_DIR, None, "Kept"),
        ("Preview copies (480p proxies)", PROXY_DIR, None, "Kept — fast scrubbing and frames"),
        ("Renders", PREVIEW_DIR, None, "Copies already on Drive can be cleared"),
        ("Projects & plans", None, (du(PROJECTS_DIR) or 0) + (du(PLANS_DIR) or 0), "Kept"),
        ("Logs", DATA_ROOT / "logs", None, "Kept (trimmed automatically)"),
    ]
    orig = downloaded_originals()
    rows.append(("Downloaded originals (videos & music)", None, sum(f["size"] for f in orig),
                 "Temporary — streamed or downloaded again when needed"))
    try:
        _has_dev = Path("/root/data-dev").exists()
    except OSError:                 # /root not readable by this user
        _has_dev = False
    if _has_dev and Path("/root/data-dev").resolve() != DATA_ROOT.resolve():
        rows.append(("Test copy's data", Path("/root/data-dev"), None,
                     "Refreshed by deploy/refresh_dev_data.sh"))
    if Path("/var/log/journal").exists():
        rows.append(("System logs (journal)", Path("/var/log/journal"), None, "Can be shrunk"))
    home_cache = Path.home() / ".cache"
    if home_cache.exists():
        rows.append(("Download caches (pip, models)", home_cache, None, "pip's cache can be cleared"))
    rows.append(("Temporary files", Path(tempfile.gettempdir()), None, "Frame grabs, leftovers"))
    out = []
    for label, path, size, note in rows:
        if size is None and path is not None:
            size = du(path)
        if size is not None:
            out.append((label, path, size, note))
    return sorted(out, key=lambda r: -r[2])


# ---------------------------------------------------------------------------
# Downloaded originals
# ---------------------------------------------------------------------------

def _last_used(p: Path) -> float:
    st = p.stat()
    return max(st.st_atime, st.st_mtime)


def downloaded_originals() -> list:
    """Original videos / music under /root/data that came from Google Drive:
    [{"path", "size", "last_used", "kind"}], oldest-used first."""
    if not cleanup_allowed() or not _exists(ORIGINALS_ROOT):
        return []
    app = _app_folders()
    found = []
    for root, dirs, files in os.walk(ORIGINALS_ROOT):
        rp = Path(root)
        if _inside(rp, app):
            dirs[:] = []
            continue
        # A separately mounted disk (e.g. a Hetzner Volume holding the originals on
        # purpose) isn't on the server's own disk — leave it alone.
        dirs[:] = [d for d in dirs if not d.startswith(".") and not os.path.ismount(rp / d)]
        for name in files:
            p = rp / name
            if p.suffix.lower() not in MEDIA_EXT or name.startswith("."):
                continue
            try:
                found.append({"path": p, "size": p.stat().st_size, "last_used": _last_used(p),
                              "kind": "music" if p.suffix.lower() in AUDIO_EXT else "video"})
            except OSError:
                continue
    return sorted(found, key=lambda f: f["last_used"])


def clear_originals(older_than_days: float = 0, kinds=("video", "music")) -> tuple:
    """Delete downloaded originals not used for older_than_days. (count, bytes)."""
    if not cleanup_allowed():
        return 0, 0
    cutoff = time.time() - older_than_days * 86400
    n = freed = 0
    for f in downloaded_originals():
        if f["kind"] in kinds and f["last_used"] <= cutoff:
            try:
                f["path"].unlink()
                n += 1
                freed += f["size"]
            except OSError:
                continue
    _remove_empty_dirs()
    return n, freed


def _remove_empty_dirs() -> None:
    app = _app_folders()
    for root, dirs, files in os.walk(ORIGINALS_ROOT, topdown=False):
        rp = Path(root)
        if rp == ORIGINALS_ROOT or _inside(rp, app) or rp.name.startswith("."):
            continue
        try:
            if not any(rp.iterdir()):
                rp.rmdir()
        except OSError:
            pass


def auto_cleanup() -> str | None:
    """Run by the Drive sync: clear originals unused for KEEP_DOWNLOADS_DAYS and
    stale temporary files. Returns a one-line summary, or None if nothing went."""
    if not cleanup_allowed() or KEEP_DOWNLOADS_DAYS <= 0:
        return None
    n, freed = clear_originals(KEEP_DOWNLOADS_DAYS)
    tn, tfreed = clear_temp(older_than_days=1)
    if n or tn:
        return f"Cleared {n + tn} file(s), {fmt_bytes(freed + tfreed)}"
    return None


# ---------------------------------------------------------------------------
# Temporary files
# ---------------------------------------------------------------------------

def temp_files() -> list:
    """Leftovers that are safe to delete: Reviewer frame grabs, half-finished
    downloads (*.part), render temp folders older than a day."""
    if not cleanup_allowed():
        return []
    out = []
    tmp = Path(tempfile.gettempdir())
    frames = tmp / "reviewer_frames"
    if frames.exists():
        out += [p for p in frames.glob("*.jpg")]
    if _exists(ORIGINALS_ROOT):
        out += [p for p in ORIGINALS_ROOT.rglob("*.part")]
    day_ago = time.time() - 86400
    for p in tmp.glob("tmp*"):
        try:
            if p.is_dir() and p.stat().st_mtime < day_ago and (p / "concat.txt").exists():
                out.append(p)
        except OSError:
            continue
    return out


def clear_temp(older_than_days: float = 0) -> tuple:
    cutoff = time.time() - older_than_days * 86400
    n = freed = 0
    recent = time.time() - 600
    for p in temp_files():
        try:
            mtime = p.stat().st_mtime
            if mtime > cutoff or (p.suffix == ".part" and mtime > recent):   # a download in progress
                continue
            size = du(p) or 0
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            n += 1
            freed += size
        except OSError:
            continue
    return n, freed


# ---------------------------------------------------------------------------
# Renders already on Drive
# ---------------------------------------------------------------------------

def local_renders_on_drive() -> list:
    """[(mp4 path, size)] — renders with a server copy that's also on Drive."""
    try:
        from renders import render_files, load_render, on_drive
    except Exception:
        return []
    out = []
    for mp4 in render_files():
        if not mp4.exists():
            continue
        meta = load_render(mp4)
        if meta and on_drive(meta):
            out.append((mp4, mp4.stat().st_size))
    return out


def clear_local_renders() -> tuple:
    from renders import clear_local
    n = freed = 0
    errors = []
    for mp4, size in local_renders_on_drive():
        err = clear_local(mp4)
        if err:
            errors.append(f"{mp4.name}: {err}")
        else:
            n += 1
            freed += size
    return n, freed, errors


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

def shrink_journal(max_mb: int = 200) -> str:
    try:
        r = subprocess.run(["journalctl", f"--vacuum-size={max_mb}M"], capture_output=True, text=True,
                           timeout=120)
        return (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else "Done."
    except Exception as e:
        return f"Couldn't shrink the logs: {e}"


def clear_pip_cache() -> int:
    p = Path.home() / ".cache" / "pip"
    size = du(p) or 0
    shutil.rmtree(p, ignore_errors=True)
    return size
