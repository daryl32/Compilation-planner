r"""
🎤 Split your songs into stems (vocals + instrumental) on your own PC's GPU — for
🎚️ mashups in the Media Library that put one track's vocals over another's music.

For every analysed track (a track JSON in AUDIO_DIR whose music file is in your
Audio-Library folder) that has no stems yet, it:
  • splits the song with Demucs (htdemucs) on the GPU — seconds a song,
  • saves  <Drive>/scene-labeling/stems/<track>/vocals.m4a  and  instrumental.m4a
    (kept OUT of Audio-Library, so they're never mistaken for new tracks),
  • notes them in the track's JSON (track["stems"]).
Drive for desktop uploads both; the server fetches the stems from Google Drive
the first time a mix uses them.

Easiest: double-click make_stems.bat — it checks and installs everything first
(the same check as run_audio_analysis.bat), then runs this.

Run from the repo folder, in your activated venv:
    python make_stems_local.py                 every track without stems yet
    python make_stems_local.py --only "Song"   tracks whose name contains "Song"
    python make_stems_local.py --redo          make them again, even if a track has stems
    python make_stems_local.py --dry-run       just list what would be done
    python make_stems_local.py --cpu           don't use the GPU (slow: minutes a song)

Safe to re-run and to stop with Ctrl+C — each track's JSON is updated in one go
when its stems are saved.
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyse_audio_local as AL  # noqa: E402  (setup, Drive root, JSON helpers)
from config import AUDIO_DIR  # noqa: E402

AA = None                                  # audio_analysis, imported in main() once the device is set
STEMS_FOLDER = "scene-labeling/stems"      # under My Drive
PARTS = ("vocals", "instrumental")
LOCAL_TEMP_DIR = HERE / ".stems_tmp"


def stems_dir(tid: str) -> Path:
    return AL.drive_root() / STEMS_FOLDER / tid


def drive_path(tid: str, part: str) -> str:
    """How the app refers to a stem — like a track's source_path."""
    return f"/content/drive/MyDrive/{STEMS_FOLDER}/{tid}/{part}.m4a"


def has_stems(track: dict) -> bool:
    st = track.get("stems") or {}
    return all(st.get(p) for p in PARTS)


def encode(y, sr: int, dest: Path) -> None:
    import numpy as np
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL_TEMP_DIR / dest.name
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, input=np.ascontiguousarray(y, dtype=np.float32).tobytes(), check=True,
                   capture_output=True)
    shutil.copy2(tmp, dest)            # one write into the Drive folder
    tmp.unlink(missing_ok=True)


def plan(library: Path, only: str | None, redo: bool) -> list:
    """[(audio file, track id)] — analysed tracks whose music is in the library."""
    files = {}
    for p in sorted(library.rglob("*")):
        if p.is_file() and p.suffix.lower() in AA.AUDIO_EXTENSIONS:
            files.setdefault(p.stem, p)
    jobs = []
    for js in sorted(Path(AUDIO_DIR).glob("*.json")):
        t = AL.read_json(js)
        if not isinstance(t, dict) or "duration_sec" not in t:
            continue
        tid = t.get("track_id", js.stem)
        if only and only.lower() not in tid.lower():
            continue
        if has_stems(t) and not redo:
            continue
        src = files.get(tid) or files.get(Path(t.get("source_path", "")).stem)
        if src is None:
            print(f"  ⚠️ {tid}: its music file isn't in {library} — skipped.")
            continue
        jobs.append((src, tid))
    return jobs


def run(audio_file: Path, tid: str) -> str:
    LOCAL_TEMP_DIR.mkdir(exist_ok=True)
    local = LOCAL_TEMP_DIR / audio_file.name
    t0 = time.time()
    shutil.copy2(audio_file, local)     # one fetch through Drive Stream
    try:
        vocals, inst, sr = AA.separate_stems(local)
    finally:
        local.unlink(missing_ok=True)
    for part, y in (("vocals", vocals), ("instrumental", inst)):
        encode(y, sr, stems_dir(tid) / f"{part}.m4a")
    json_path = Path(AUDIO_DIR) / f"{tid}.json"
    track = AL.read_json(json_path) or {"track_id": tid}     # re-read: never clobber an edit made meanwhile
    track["stems"] = {**{p: drive_path(tid, p) for p in PARTS}, "model": "htdemucs",
                      "made_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    AL.write_json(json_path, track)
    return f"{time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", help=f"folder of music (default: <Drive>/{AL.LIBRARY_FOLDER})")
    ap.add_argument("--only", help="only tracks whose name contains this text")
    ap.add_argument("--redo", action="store_true", help="make stems again for tracks that have them")
    ap.add_argument("--cpu", action="store_true", help="don't use the GPU")
    ap.add_argument("--dry-run", action="store_true", help="list what would be done, then stop")
    ap.add_argument("--setup", action="store_true", help="check / install the packages and ffmpeg, then stop")
    args = ap.parse_args()
    if args.setup:
        sys.exit(AL.setup())

    global AA
    os.environ["AUDIO_ANALYSIS_DEVICE"] = AL.pick_device(args.cpu)
    import audio_analysis as AA            # reads AUDIO_ANALYSIS_DEVICE
    AL.AA = AA
    library = Path(args.library) if args.library else AL.drive_root() / AL.LIBRARY_FOLDER
    if not library.exists():
        sys.exit(f"Music folder not found: {library}")
    jobs = plan(library, args.only, args.redo)
    print(f"🎤 {len(jobs)} track(s) to split (device: {AA.TORCH_DEVICE}) → {AL.drive_root() / STEMS_FOLDER}")
    for f, tid in jobs:
        print(f"   • {tid}")
    if args.dry_run or not jobs:
        return
    done = failed = 0
    for i, (f, tid) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {tid}", flush=True)
        try:
            print(f"      ✅ {run(f, tid)}", flush=True)
            done += 1
        except KeyboardInterrupt:
            print("\nStopped — the tracks finished so far are saved.")
            break
        except Exception as e:
            traceback.print_exc()
            print(f"      ❌ {e}")
            failed += 1
    shutil.rmtree(LOCAL_TEMP_DIR, ignore_errors=True)
    print(f"\nDone: {done} track(s) split" + (f", {failed} failed" if failed else "") + ".")


if __name__ == "__main__":
    main()
