r"""
Analyse your music on your own PC — the Colab audio pipeline and the newer
server audio analysis in one script, using your GPU.

For every track in your Audio-Library folder it writes (or updates) the
track's JSON in AUDIO_DIR, the same file the app reads:

  • New track (no JSON yet) — everything:
      duration_sec, bpm, beat_times, energy_envelope, hires{rate, rms, onset}
      (what audio_pipeline.py made in Colab) PLUS track["analysis"]:
      downbeats, kick/snare/hat, harmony, brightness, novelty, sections,
      phrases, builds, drops and vocals (what the server's "Analyse" button adds).
  • Track already analysed in Colab — keeps its beats and energy exactly as
      they are (so existing plans still line up) and only adds the analysis.
  • Track already fully analysed — skipped (use --redo to do it again).

Same code as the server (audio_analysis.py), so results look the same in the
app — but here beat_this and Demucs run on your RTX GPU, so downbeats and
vocals take seconds instead of minutes. New tracks also take their beat
grid from beat_this when it's installed (usually tighter than the built-in
tracker).

Your Google Drive is in STREAM mode, so each track is copied to a local
temp folder first and analysed from there.

Run from the repo folder, in your activated venv:
    python analyse_audio_local.py                 every track that needs it
    python analyse_audio_local.py --only "Song"   tracks whose name contains "Song"
    python analyse_audio_local.py --redo          re-analyse everything
    python analyse_audio_local.py --fresh         redo beats too, even for Colab tracks
    python analyse_audio_local.py --dry-run       just list what would be done

Needs: numpy, scipy, ffmpeg on PATH (as for the app). For the full result:
    pip install torch --index-url https://download.pytorch.org/whl/cu121   (CUDA build)
    pip install beat_this demucs
Without them it still runs, with estimated downbeats and no vocals.

Safe to re-run and to stop with Ctrl+C — each track's JSON is written in one
go when that track finishes. Drive for desktop uploads the JSONs; the server
picks them up at its next Drive sync (or press Sync in the app).
"""

import argparse
import datetime
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from config import AUDIO_DIR  # noqa: E402

LOCAL_TEMP_DIR = HERE / "_tmp_audio"   # genuine local disk, next to this script
LIBRARY_FOLDER = getattr(config, "AUDIO_LIBRARY_DRIVE_FOLDER", "Audio-Library")


def drive_root() -> Path:
    root = getattr(config, "DRIVE_ROOT", None)
    if root:
        return Path(root)
    # config.py without DRIVE_ROOT: AUDIO_DIR is <drive>/scene-labeling/audio_catalogue
    return Path(AUDIO_DIR).parent.parent


def pick_device(force_cpu: bool) -> str:
    if force_cpu or importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def plan_jobs(library: Path, only: str | None, redo: bool, fresh: bool, needs_extra) -> list:
    """[(audio_file, track_id, kind)] — kind "new" (full), "extra" (keep beats) or "redo"."""
    files = sorted(p for p in library.rglob("*")
                   if p.is_file() and p.suffix.lower() in AA.AUDIO_EXTENSIONS)
    jobs, seen = [], {}
    for f in files:
        tid = f.stem
        if tid in seen:
            print(f"  ⚠️ Two files are both called '{tid}' — skipping {f} (keeping {seen[tid]}).")
            continue
        seen[tid] = f
        if only and only.lower() not in tid.lower():
            continue
        existing = read_json(AUDIO_DIR / f"{tid}.json")
        if existing is None:
            jobs.append((f, tid, "new"))
        elif fresh:
            jobs.append((f, tid, "fresh"))
        elif redo or needs_extra(existing):
            jobs.append((f, tid, "extra"))
    return jobs


def run_job(audio_file: Path, tid: str, kind: str, rel: str, with_vocals: bool) -> str:
    json_path = AUDIO_DIR / f"{tid}.json"
    existing = read_json(json_path) if kind == "extra" else None

    LOCAL_TEMP_DIR.mkdir(exist_ok=True)
    local = LOCAL_TEMP_DIR / audio_file.name
    t0 = time.time()
    shutil.copy2(audio_file, local)   # one fetch through Drive Stream
    try:
        result = AA.analyse_file(local, existing=existing, with_vocals=with_vocals,
                                 progress=lambda m: print(f"      … {m}", flush=True),
                                 model_beats=True)
    finally:
        local.unlink(missing_ok=True)

    current = read_json(json_path)   # re-read: never clobber an edit made meanwhile
    if current is not None and kind != "new":
        track = current
        track.update(result)
    else:
        track = {"track_id": tid, **(current or {}), **result}
    track.setdefault("track_id", tid)
    track["source_path"] = track.get("source_path") or f"/content/drive/MyDrive/{rel}"
    write_json(json_path, track)

    a = result["analysis"]
    bpm = track.get("bpm")
    return (f"{time.time() - t0:.0f}s · {bpm:.1f} BPM · {len(a['downbeats'])} bars "
            f"({a['methods']['downbeats']}) · {len(a['sections'])} sections · "
            f"{len(a['drops'])} drop(s) · vocals: {a['methods']['vocals'] or 'no'}"
            if bpm else f"{time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", help=f"folder of music (default: <Drive>/{LIBRARY_FOLDER})")
    ap.add_argument("--only", help="only tracks whose file name contains this text")
    ap.add_argument("--redo", action="store_true", help="re-run the analysis on already-analysed tracks "
                                                        "(their beats are still kept)")
    ap.add_argument("--fresh", action="store_true", help="re-analyse from scratch, replacing beats, bpm, "
                                                         "energy and hires too (can shift existing plans)")
    ap.add_argument("--no-vocals", action="store_true", help="skip Demucs vocal separation")
    ap.add_argument("--cpu", action="store_true", help="don't use the GPU")
    ap.add_argument("--dry-run", action="store_true", help="list what would be analysed, then stop")
    args = ap.parse_args()

    device = pick_device(args.cpu)
    os.environ["AUDIO_ANALYSIS_DEVICE"] = device       # read when audio_analysis is imported
    os.environ["AUDIO_ANALYSED_BY"] = "local"
    global AA
    import audio_analysis as AA

    root = drive_root()
    library = Path(args.library) if args.library else root / LIBRARY_FOLDER
    if not library.is_dir():
        sys.exit(f"Music folder not found: {library}\nPass --library \"path\\to\\your\\music\".")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg isn't on PATH — install it (winget install ffmpeg) and open a new terminal.")

    print(f"Music:      {library}")
    print(f"Track JSON: {AUDIO_DIR}")
    if device == "cuda":
        import torch
        print(f"Device:     GPU — {torch.cuda.get_device_name(0)}")
    else:
        print("Device:     CPU" + ("" if args.cpu else "  (no CUDA torch found — models will be slow)"))
    print(f"beat_this:  {'yes' if AA.beat_this_available() else 'not installed — downbeats will be estimated (pip install beat_this)'}")
    vocals_on = not args.no_vocals and AA.demucs_available()
    print(f"Demucs:     {'yes' if vocals_on else ('off (--no-vocals)' if args.no_vocals else 'not installed — no vocals (pip install demucs)')}")

    jobs = plan_jobs(library, args.only, args.redo, args.fresh, AA.needs_extra)
    labels = {"new": "new — full analysis", "extra": "add analysis, keep its beats",
              "fresh": "from scratch, replacing beats"}
    print(f"\n{len(jobs)} track(s) to analyse.")
    for f, tid, kind in jobs:
        print(f"  • {tid}  ({labels[kind]})")
    if args.dry_run or not jobs:
        return

    done, failed = 0, []
    started = time.time()
    try:
        for i, (f, tid, kind) in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] {tid}")
            try:
                rel = f.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                rel = f"{LIBRARY_FOLDER}/{f.relative_to(library).as_posix()}"
            try:
                summary = run_job(f, tid, "new" if kind == "fresh" else kind, rel, vocals_on)
                print(f"    ✓ {summary}")
                done += 1
            except Exception as e:
                traceback.print_exc()
                print(f"    ✗ failed: {e}")
                failed.append(tid)
    except KeyboardInterrupt:
        print("\nStopped — tracks finished so far are saved. Run again to carry on.")

    mins = (time.time() - started) / 60
    print(f"\nDone: {done} analysed, {len(failed)} failed, in {mins:.1f} min "
          f"({datetime.datetime.now():%H:%M}).")
    if failed:
        print("Failed: " + ", ".join(failed))
    shutil.rmtree(LOCAL_TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
