"""🧍 Avatar Studio previews (17 Nov 2026).

The Avatar Studio (a page on claude.ai) reads clips through the Google Drive connector, which refuses files over
10 MB, and an iPhone can't play the old transparent .webm clips. So the server makes, for every avatar clip in
Edge_voiceover (transitions, avatar_clips, sitting — and their subfolders):

  Edge_voiceover/_studio_previews/<drive id>.mp4   a small H.264 copy (≤540 px wide, sound kept) — plays anywhere
  Edge_voiceover/_studio_previews/<drive id>.jpg   a strip of three frames: first | middle | last

Named by the clip's Drive id, so renaming a clip in the Studio doesn't lose its preview. Clips that already have
both are skipped; run it again after adding clips.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = "scene-labeling/Edge_voiceover"
FOLDERS = ("transitions", "avatar_clips", "sitting")
PREVIEW_FOLDER = f"{ROOT}/_studio_previews"
VIDEO_EXT = (".mp4", ".mov", ".webm", ".m4v")
WIDTH = 540            # preview width (px)
THUMB_W = 160          # each of the three frames in the strip
MAX_DEPTH = 3          # e.g. sitting/avatar_clips/<sub>


def clips() -> tuple:
    """([{"id", "name", "size", "folder"}], error or None) — every video under the Studio's folders."""
    try:
        from drive_sync import _find_folder, _get_service, _list_files
        service = _get_service()
        root = None
        for part in ROOT.split("/"):
            root = _find_folder(service, part, root)
            if not root:
                return [], f"Folder '{part}' not found in Drive"
        out, todo = [], []
        for name in FOLDERS:
            fid = _find_folder(service, name, root)
            if fid:
                todo.append((fid, name))
        while todo:
            fid, rel = todo.pop(0)
            for f in _list_files(service, fid):
                if f.get("name", "").lower().endswith(VIDEO_EXT):
                    out.append({"id": f["id"], "name": f["name"], "size": int(f.get("size") or 0), "folder": rel})
            if rel.count("/") + 1 < MAX_DEPTH:
                resp = service.files().list(
                    q=f"'{fid}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                    fields="files(id, name)", pageSize=200).execute()
                todo += [(s["id"], f"{rel}/{s['name']}") for s in resp.get("files", []) if not s["name"].startswith("_")]
        return out, None
    except Exception as e:
        return [], str(e)


def done_ids() -> set:
    """Drive ids that already have both a preview and a thumbnail strip."""
    from drive_sync import list_drive_folder
    files, err = list_drive_folder(PREVIEW_FOLDER)
    if err:
        return set()
    names = {f["name"] for f in files}
    return {n[:-4] for n in names if n.endswith(".mp4") and n[:-4] + ".jpg" in names}


def _duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def make_preview(src: Path, mp4: Path) -> str | None:
    """The small copy. A .webm (the old transparent clips) is decoded with its alpha and laid on dark grey, so the
    cut-out shows the same on every phone. Returns None, or an error."""
    webm = src.suffix.lower() == ".webm"
    scale = f"scale='min({WIDTH},iw)':-2"
    vf = (f"{scale},format=rgba,split[a][b];[a]drawbox=c=0x2b2b2b@1:t=fill[bg];[bg][b]overlay=format=auto,format=yuv420p"
          if webm else f"{scale},format=yuv420p")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if webm:
        cmd += ["-c:v", "libvpx-vp9"]                # the decoder that keeps the alpha
    cmd += ["-i", str(src), "-filter_complex" if webm else "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-profile:v", "main",
            "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-movflags", "+faststart", str(mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode and webm:                      # not VP9 after all: try the plain decoder
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-filter_complex", vf,
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-c:a", "aac", "-b:a", "64k",
                            "-ac", "1", "-movflags", "+faststart", str(mp4)], capture_output=True, text=True)
    if r.returncode or not mp4.exists():
        return (r.stderr or "ffmpeg failed").strip().splitlines()[-1][:200] if (r.stderr or "").strip() else "ffmpeg failed"
    return None


def make_strip(mp4: Path, jpg: Path) -> str | None:
    """First | middle | last frame side by side, from the preview."""
    d = _duration(mp4)
    times = [0.0, d / 2, max(d - 0.15, 0.0)]
    ins = []
    for t in times:
        ins += ["-ss", f"{t:.3f}", "-i", str(mp4)]
    fc = "".join(f"[{k}:v]trim=end_frame=1,scale={THUMB_W}:-2,setsar=1[f{k}];" for k in range(3)) + "[f0][f1][f2]hstack=3"
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *ins, "-filter_complex", fc, "-frames:v", "1",
                        "-q:v", "5", str(jpg)], capture_output=True, text=True)
    if r.returncode or not jpg.exists():
        return (r.stderr or "ffmpeg failed").strip()[-200:]
    return None


def run(job, token: dict, redo: bool = False) -> dict:
    """The server job: preview + strip for every clip that hasn't got them, uploaded to Drive."""
    from drive_oauth import upload_file_with_oauth
    from drive_sync import download_drive_file
    job.progress(0.0, "Listing the clips")
    found, err = clips()
    if err:
        raise RuntimeError(err)
    have = set() if redo else done_ids()
    todo = [c for c in found if c["id"] not in have]
    made, errs = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, c in enumerate(todo):
            job.progress(k / max(len(todo), 1), f"{c['name']} ({k + 1}/{len(todo)})")
            src = tmp / ("src" + Path(c["name"]).suffix.lower())
            mp4, jpg = tmp / f"{c['id']}.mp4", tmp / f"{c['id']}.jpg"
            try:
                download_drive_file(c["id"], src)
                e = make_preview(src, mp4) or make_strip(mp4, jpg)
                if not e:
                    for f in (mp4, jpg):
                        _, e = upload_file_with_oauth(token, f, PREVIEW_FOLDER)
                        if e:
                            break
            except Exception as ex:
                e = str(ex)
            if e:
                errs.append(f"{c['name']}: {e}")
            else:
                made += 1
            for f in (src, mp4, jpg):
                f.unlink(missing_ok=True)
    job.progress(1.0, "Done")
    return {"clips": len(found), "made": made, "skipped": len(found) - len(todo), "errors": errs}
