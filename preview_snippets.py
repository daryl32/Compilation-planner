"""
Short synced preview clips: a few seconds of video with the track's music for
the same moment laid on top — used by Choreography's ▶ Preview panel so you
can judge a clip against the beat. Made with ffmpeg on the server (480p, fast
settings), cached by content, so revisiting a position plays straight away.

No Streamlit in here.
"""

import hashlib
import os
import subprocess
from pathlib import Path

MAX_CACHED = 150     # oldest snippets beyond this are deleted
PREVIEW_HEIGHT = 480


def snippet_path(cache_dir: Path, video_path: Path, video_start: float, duration: float,
                 audio_path: Path, audio_start: float) -> Path:
    """Where the snippet for exactly these inputs is cached."""
    sig = "|".join(str(x) for x in (
        video_path, round(video_start, 3), round(duration, 3), audio_path, round(audio_start, 3),
        _mtime(video_path), _mtime(audio_path),
    ))
    return Path(cache_dir) / f"{hashlib.sha1(sig.encode()).hexdigest()[:20]}.mp4"


def _mtime(p: Path) -> float:
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return 0.0


def make_synced_snippet(cache_dir: Path, video_path: Path, video_start: float, duration: float,
                        audio_path: Path, audio_start: float) -> tuple:
    """(path, None) on success, (None, error) on failure. Re-uses a cached copy."""
    out = snippet_path(cache_dir, video_path, video_start, duration, audio_path, audio_start)
    if out.exists():
        os.utime(out)  # mark as recently used
        return out, None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, video_start):.3f}", "-t", f"{duration:.3f}", "-i", str(video_path),
        "-ss", f"{max(0.0, audio_start):.3f}", "-t", f"{duration:.3f}", "-i", str(audio_path),
        # 1:a:0, not 1:a — .m4a files can carry album art as an extra stream
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", f"scale=-2:'min({PREVIEW_HEIGHT},ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-t", f"{duration:.3f}", "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return None, "ffmpeg not found on the server"
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return None, "making the preview took too long"
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return None, (proc.stderr or "ffmpeg failed").strip()[-400:]
    tmp.replace(out)
    _prune(Path(cache_dir))
    return out, None


def _prune(cache_dir: Path) -> None:
    files = sorted((p for p in cache_dir.glob("*.mp4") if ".part" not in p.name),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[MAX_CACHED:]:
        p.unlink(missing_ok=True)
