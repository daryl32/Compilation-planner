"""
Backfill: a fast 480p preview copy ("proxy") of every video, saved to
scene-labeling/proxies on Google Drive. The server's 5-minute Drive sync pulls
them down, and the range picker in the Media Library / Compilation Planner
plays them instead of the full-size original.

Run in Colab (Drive mounted) after processing new videos. Safe to re-run —
skips any video that already has a proxy. Uses the GPU encoder (NVENC) when
the runtime has a GPU, otherwise the CPU.

To make one inside your own processing code instead, call:
    make_proxy_for(video_id, source_path)
"""

import os
import json
import shutil
import subprocess
import time
from pathlib import Path

DRIVE_BASE = Path("/content/drive/MyDrive/scene-labeling")
CATALOGUE_DIR = DRIVE_BASE / "catalogue"
PROXY_DIR = DRIVE_BASE / "proxies"
PROXY_HEIGHT = 480
WORK_DIR = Path("/content/proxy_work")  # local disk: Drive is too slow to read/write frame by frame

PROXY_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)


def _ffmpeg(args: list) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, text=True)


def _nvenc_available() -> bool:
    """True if this runtime can actually encode with NVENC (GPU runtime)."""
    probe = _ffmpeg(["-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                     "-c:v", "h264_nvenc", "-f", "null", "-"])
    return probe.returncode == 0


USE_NVENC = _nvenc_available()
print(f"Encoder: {'GPU (h264_nvenc)' if USE_NVENC else 'CPU (libx264)'}")


def _encode(src: Path, dest: Path) -> str | None:
    """Same settings as the server's make_proxy: ≤480p H.264 + AAC, faststart,
    same timeline as the source (nothing trimmed). Returns an error or None."""
    video = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "30"] if USE_NVENC
             else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"])
    proc = _ffmpeg([
        "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"scale=-2:'min({PROXY_HEIGHT},ih)'",
        *video, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-ac", "2",
        "-movflags", "+faststart",
        str(dest),
    ])
    if proc.returncode != 0:
        return (proc.stderr or "ffmpeg failed").strip()[-500:]
    return None


def make_proxy_for(video_id: str, source_path: str, force: bool = False) -> str:
    """Make scene-labeling/proxies/<video_id>.mp4. Returns a one-line status."""
    out = PROXY_DIR / f"{video_id}.mp4"
    if out.exists() and not force:
        return "already has a proxy"
    if not os.path.exists(source_path):
        return f"source not found at {source_path}"

    local_src = WORK_DIR / f"src_{video_id}{Path(source_path).suffix}"
    local_out = WORK_DIR / f"{video_id}.mp4"
    try:
        shutil.copy(source_path, local_src)
        err = _encode(local_src, local_out)
        if err and USE_NVENC:  # some sources trip NVENC — retry on the CPU
            err = _encode_cpu_fallback(local_src, local_out)
        if err:
            return f"failed: {err}"
        # Copy under a temporary name, then rename, so the server sync never
        # pulls a half-uploaded file.
        tmp = PROXY_DIR / f"{video_id}.part.mp4"
        shutil.copy(local_out, tmp)
        os.replace(tmp, out)
        return f"made ({out.stat().st_size / 1e6:.1f} MB)"
    finally:
        for p in (local_src, local_out):
            if p.exists():
                p.unlink()


def _encode_cpu_fallback(src: Path, dest: Path) -> str | None:
    global USE_NVENC
    was, USE_NVENC = USE_NVENC, False
    try:
        return _encode(src, dest)
    finally:
        USE_NVENC = was


if __name__ == "__main__":
    catalogue_files = sorted(CATALOGUE_DIR.glob("*.json"))
    print(f"Found {len(catalogue_files)} JSON files in the catalogue folder.\n")
    made = skipped = problems = 0
    for cat_file in catalogue_files:
        try:
            catalogue = json.loads(cat_file.read_text())
        except (ValueError, OSError):
            continue
        if not (isinstance(catalogue, dict) and "video_id" in catalogue and "scenes" in catalogue):
            continue  # not a video catalogue (labels.json, library_meta.json, ...)

        video_id = catalogue["video_id"]
        t0 = time.time()
        status = make_proxy_for(video_id, catalogue.get("source_path", ""))
        if status.startswith("made"):
            made += 1
            print(f"✓  {video_id}: {status} in {time.time() - t0:.0f}s")
        elif status == "already has a proxy":
            skipped += 1
        else:
            problems += 1
            print(f"⚠️  {video_id}: {status}")

    print(f"\nMade {made}, already done {skipped}, problems {problems}.")
    print("The server picks new proxies up on its next Drive sync (within 5 minutes).")
