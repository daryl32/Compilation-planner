r"""
Analyse your music on your own PC — the Colab audio pipeline and the newer
server audio analysis in one script, using your GPU.

By default it only does NEWLY ADDED tracks — music files in your
Audio-Library folder that have no track JSON in AUDIO_DIR yet — and gives
each one the full analysis:

  • New track (no JSON yet) — everything:
      duration_sec, bpm, beat_times, energy_envelope, hires{rate, rms, onset}
      (what audio_pipeline.py made in Colab) PLUS track["analysis"]:
      downbeats, kick/snare/hat, harmony, brightness, novelty, sections,
      phrases, builds, drops and vocals (what the server's "Analyse" button adds).
  • Older track already analysed (Colab or server) — left alone. With
      --upgrade, older tracks missing the latest analysis get it added, keeping
      their beats and energy exactly as they are (so existing plans still line up).
  • Track already fully analysed — skipped (use --redo to do it again).

Same code as the server (audio_analysis.py), so results look the same in the
app — but here beat_this and Demucs run on your RTX GPU, so downbeats and
vocals take seconds instead of minutes. New tracks also take their beat
grid from beat_this when it's installed (usually tighter than the built-in
tracker).

Your Google Drive is in STREAM mode, so each track is copied to a local
temp folder first and analysed from there.

Easiest: double-click run_audio_analysis.bat — it checks and installs
everything first (python analyse_audio_local.py --setup), then runs this.

Run from the repo folder, in your activated venv:
    python analyse_audio_local.py                 newly added tracks only
    python analyse_audio_local.py --upgrade       also add the latest analysis to older tracks
    python analyse_audio_local.py --only "Song"   tracks whose name contains "Song"
    python analyse_audio_local.py --redo          re-analyse everything
    python analyse_audio_local.py --fresh         redo beats too, even for Colab tracks
    python analyse_audio_local.py --dry-run       just list what would be done
    python analyse_audio_local.py --setup         check / install packages and ffmpeg, then stop

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


def plan_jobs(library: Path, only: str | None, redo: bool, fresh: bool, needs_extra,
              upgrade: bool = False) -> tuple:
    """([(audio_file, track_id, kind)], n_older_needing_upgrade) — kind "new"
    (full), "extra" (keep beats) or "fresh". Without --upgrade/--redo/--fresh/--only
    only new tracks (no JSON yet) are planned."""
    files = sorted(p for p in library.rglob("*")
                   if p.is_file() and p.suffix.lower() in AA.AUDIO_EXTENSIONS)
    jobs, seen, older = [], {}, 0
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
        elif redo or (needs_extra(existing) and (upgrade or only)):
            jobs.append((f, tid, "extra"))
        elif needs_extra(existing):
            older += 1
    return jobs, older


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
    ap.add_argument("--upgrade", action="store_true", help="also add the latest analysis to older tracks "
                                                           "that were analysed before (their beats are kept)")
    ap.add_argument("--setup", action="store_true", help="check / install the packages and ffmpeg, then stop")
    ap.add_argument("--redo", action="store_true", help="re-run the analysis on already-analysed tracks "
                                                        "(their beats are still kept)")
    ap.add_argument("--fresh", action="store_true", help="re-analyse from scratch, replacing beats, bpm, "
                                                         "energy and hires too (can shift existing plans)")
    ap.add_argument("--no-vocals", action="store_true", help="skip Demucs vocal separation")
    ap.add_argument("--cpu", action="store_true", help="don't use the GPU")
    ap.add_argument("--dry-run", action="store_true", help="list what would be analysed, then stop")
    args = ap.parse_args()
    if args.setup:
        sys.exit(setup())

    device = pick_device(args.cpu)
    os.environ["AUDIO_ANALYSIS_DEVICE"] = device       # read when audio_analysis is imported
    os.environ["AUDIO_ANALYSED_BY"] = "local"
    global AA
    import audio_analysis as AA

    root = drive_root()
    library = Path(args.library) if args.library else root / LIBRARY_FOLDER
    if not library.is_dir():
        sys.exit(f"Music folder not found: {library}\nPass --library \"path\\to\\your\\music\".")
    if not _ensure_ffmpeg():
        sys.exit("ffmpeg isn't on PATH — install it (winget install Gyan.FFmpeg) and open a new terminal.")

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

    jobs, older = plan_jobs(library, args.only, args.redo, args.fresh, AA.needs_extra, args.upgrade)
    labels = {"new": "new — full analysis", "extra": "add analysis, keep its beats",
              "fresh": "from scratch, replacing beats"}
    n_new = sum(1 for _, _, k in jobs if k == "new")
    print(f"\n{len(jobs)} track(s) to analyse" + (f" ({n_new} newly added)." if n_new == len(jobs) else "."))
    if older:
        print(f"  ({older} older track(s) could get the latest analysis added — "
              f"run with --upgrade to include them.)")
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


# ---------------------------------------------------------------------------
# --setup: make sure everything the full analysis needs is installed
# ---------------------------------------------------------------------------
# Each check runs in a fresh Python, so a package installed a moment ago is
# seen, and a broken one (e.g. torchaudio built for another torch) shows up as
# broken rather than just "present".

TORCH_INDEX = "https://download.pytorch.org/whl/{}"
DEFAULT_CUDA_TAG = "cu126"   # GPU build used when torch isn't installed yet


def _py(code: str) -> tuple:
    import subprocess
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout or "").strip(), (r.stderr or "").strip().splitlines()[-1:] or [""]


def _pip(*args) -> bool:
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args]
    print("    > pip install " + " ".join(args), flush=True)
    return subprocess.run(cmd).returncode == 0


def _has_nvidia_gpu() -> bool:
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    except OSError:
        return False


def _torch_info() -> dict | None:
    ok, out, _ = _py("import json, torch; print(json.dumps({'v': torch.__version__, "
                     "'cuda': torch.version.cuda, 'gpu': torch.cuda.is_available()}))")
    return json.loads(out.splitlines()[-1]) if ok and out else None


def _ensure_ffmpeg() -> bool:
    if shutil.which("ffmpeg"):
        return True
    links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
    if (links / "ffmpeg.exe").exists():
        os.environ["PATH"] = f"{links}{os.pathsep}{os.environ['PATH']}"
        return True
    if os.name == "nt" and shutil.which("winget"):
        print("  ffmpeg: not found — installing with winget…", flush=True)
        import subprocess
        subprocess.run(["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--silent",
                        "--accept-package-agreements", "--accept-source-agreements"])
        if (links / "ffmpeg.exe").exists() or shutil.which("ffmpeg"):
            return True
    return False


def setup() -> int:
    """Check / install numpy, scipy, a GPU build of torch with a matching
    torchaudio, beat_this, Demucs and ffmpeg. 0 = ready for the full analysis,
    1 = something is still missing (the analysis can still run, with less)."""
    problems = []
    print("Checking what the full audio analysis needs…\n", flush=True)

    # numpy + scipy (must-haves)
    if not _py("import numpy, scipy.signal, scipy.ndimage")[0]:
        print("  numpy / scipy: missing or broken — installing", flush=True)
        _pip("--upgrade", "--force-reinstall", "numpy", "scipy")
    ok = _py("import numpy, scipy.signal, scipy.ndimage")[0]
    print(f"  numpy / scipy: {'OK' if ok else 'STILL MISSING'}")
    if not ok:
        problems.append("numpy/scipy")

    # torch — the GPU (CUDA) build when there's an NVIDIA GPU
    gpu_here = _has_nvidia_gpu()
    ti = _torch_info()
    if ti is None or (gpu_here and not ti["gpu"]):
        why = "not installed" if ti is None else f"{ti['v']} can't use the GPU"
        print(f"  torch: {why} — installing the GPU build (large download, one time only)", flush=True)
        if gpu_here:
            _pip("--upgrade", "torch", "torchaudio", "--index-url", TORCH_INDEX.format(DEFAULT_CUDA_TAG))
        else:
            _pip("--upgrade", "torch", "torchaudio")
        ti = _torch_info()
    if ti is None:
        print("  torch: STILL MISSING")
        problems.append("torch")
    else:
        print(f"  torch: {ti['v']}  ·  " + ("GPU ✓" if ti["gpu"] else "CPU only (slow)"))
        if gpu_here and not ti["gpu"]:
            problems.append("torch GPU build")

        # torchaudio must be built for this exact torch (a mismatched one won't load,
        # and then beat_this quietly falls back to estimated downbeats).
        base = ti["v"].split("+")[0]
        ok_ta, ta_v, _ = _py("import torchaudio; print(torchaudio.__version__)")
        if not ok_ta or ta_v.split("+")[0] != base:
            print(f"  torchaudio: {'broken' if not ok_ta else ta_v} — needs {base}; reinstalling", flush=True)
            extra = (["--index-url", TORCH_INDEX.format("cu" + ti["cuda"].replace(".", ""))]
                     if ti.get("cuda") else [])
            _pip("--force-reinstall", "--no-deps", f"torchaudio=={base}", *extra)
            ok_ta, ta_v, _ = _py("import torchaudio; print(torchaudio.__version__)")
        print(f"  torchaudio: {ta_v if ok_ta else 'STILL BROKEN'}")
        if not ok_ta:
            problems.append("torchaudio")

    # beat_this + Demucs (installed after torch so they reuse it, not replace it)
    checks = {"beat_this": "import beat_this.inference", "demucs": "import demucs.pretrained, demucs.apply"}
    missing = [name for name, code in checks.items() if not _py(code)[0]]
    if missing and ti is not None:
        print(f"  {' + '.join(missing)}: missing or broken — installing", flush=True)
        _pip("--upgrade", *missing)
    for name, code in checks.items():
        ok, _, err = _py(code)
        print(f"  {name}: {'OK' if ok else 'NOT WORKING — ' + err[0]}")
        if not ok:
            problems.append(name)

    # ffmpeg (decodes the music)
    ff = _ensure_ffmpeg()
    print(f"  ffmpeg: {'OK' if ff else 'NOT FOUND — install it (winget install Gyan.FFmpeg) and run again'}")
    if not ff:
        problems.append("ffmpeg")

    print()
    if problems:
        print("⚠️ Not everything is ready: " + ", ".join(problems) + ".")
        return 1
    print("✓ Everything for the full analysis is installed.")
    return 0


if __name__ == "__main__":
    main()
