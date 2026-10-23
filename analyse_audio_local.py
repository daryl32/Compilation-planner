r"""
Analyse your music on your own PC, using your GPU — one click does everything:
double-click run_audio_analysis.bat.

It goes through every music file in your Audio-Library folder and does whatever
that track is still missing:

  • New track (no JSON yet) — the full analysis:
      duration_sec, bpm, beat_times, energy_envelope, hires{rate, rms, onset}
      PLUS track["analysis"]: downbeats, kick/snare/hat, harmony, brightness,
      novelty, sections, phrases, builds, drops and vocals; the 🎹 key; and
      🎤 stems (vocals + instrumental, and drums / bass / other for 🧠 neural
      transitions) for the Media Library's Mix tab.
  • Older track — only the missing parts:
      no extra analysis yet (or an older version) → added, keeping its beats and
                                                    energy exactly (plans still line up)
      no vocals            → the vocal curve and sung phrases
      no key               → the key
      no stems (or only     → vocals, instrumental, drums, bass and other .m4a in
      vocals + instrumental)  <Drive>/scene-labeling/stems/<track>/ (track["stems"])
  • Track that has everything — skipped.

Demucs runs once per track and gives both the vocals for the analysis and the
stems. Same code as the server (audio_analysis.py), so results look the same in
the app — but here beat_this and Demucs run on your RTX GPU, so they take
seconds instead of minutes. New tracks take their beat grid from beat_this when
it's installed (usually tighter than the built-in tracker).

Your Google Drive is in STREAM mode, so each track is copied to a local temp
folder first and analysed from there.

Easiest: double-click run_audio_analysis.bat — it checks and installs
everything first (python analyse_audio_local.py --setup), then runs this.

Run from the repo folder, in your activated venv:
    python analyse_audio_local.py                 everything each track is missing
    python analyse_audio_local.py --report        list each track: bar lines, vocals, key, stems
    python analyse_audio_local.py --enhance       also redo tracks whose bar lines were estimated
                                                  (beats kept) — or double-click enhance_audio_analysis.bat
    python analyse_audio_local.py --only "Song"   tracks whose name contains "Song"
    python analyse_audio_local.py --redo          re-analyse everything (beats kept)
    python analyse_audio_local.py --fresh         redo beats too, even for Colab tracks
    python analyse_audio_local.py --no-stems      skip making stems
    python analyse_audio_local.py --no-vocals     skip Demucs altogether (no vocals, no stems)
    python analyse_audio_local.py --dry-run       just list what would be done
    python analyse_audio_local.py --setup         check / install packages and ffmpeg, then stop

Needs: numpy, scipy, ffmpeg on PATH (as for the app). For the full result:
    pip install torch --index-url https://download.pytorch.org/whl/cu121   (CUDA build)
    pip install beat_this demucs
Without them it still runs, with estimated downbeats and no vocals or stems.

Safe to re-run and to stop with Ctrl+C — each track's JSON is written in one
go when that track finishes. Drive for desktop uploads the JSONs and stems; the
server picks them up at its next Drive sync (or press Sync in the app).
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

def _quiet_dll_errors() -> None:
    """Windows: a broken package (e.g. torchaudio built for another torch) makes
    Windows pop up an "Entry Point Not Found" box that blocks until OK is
    clicked. Turn those boxes off — the import then just fails, which the checks
    report and fix. Child processes inherit this setting."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        except Exception:
            pass


_quiet_dll_errors()

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


STEMS_FOLDER = "scene-labeling/stems"      # under My Drive — kept out of Audio-Library
STEM_PARTS = ("vocals", "instrumental")
FOUR_STEMS = ("vocals", "drums", "bass", "other")     # 🧠 neural transitions (core/stem_mix.py)
ALL_STEMS = ("vocals", "instrumental", "drums", "bass", "other")


def track_status(track: dict) -> dict:
    """What a track JSON has: the extra analysis, bar lines from the model or estimated,
    vocals, key, stems."""
    a = track.get("analysis") or {}
    m = a.get("methods") or {}
    st = track.get("stems") or {}
    return {
        "analysed": bool(a),
        "bars_model": str(m.get("downbeats") or "").startswith("beat_this"),
        "vocals": a.get("vocals") is not None,
        "key": bool(track.get("key")),
        "stems": all(st.get(p) for p in STEM_PARTS),
        "stems4": all(st.get(p) for p in FOUR_STEMS),
        "by": track.get("analysed_by") or ("colab" if track.get("beat_times") else "?"),
        "when": (a.get("analysed_at") or "")[:10],
    }


def missing_parts(track: dict | None, *, demucs: bool, stems: bool, beat_this: bool = False,
                  enhance: bool = False, redo: bool = False) -> set:
    """What this track still needs: "new", "analysis" (redo the extra analysis, beats kept),
    "vocals", "key", "stems". Empty = nothing to do."""
    if track is None:
        return {"new", "key"} | ({"vocals"} if demucs else set()) | ({"stems"} if demucs and stems else set())
    st = track_status(track)
    need = set()
    if redo or not st["analysed"] or AA.needs_extra(track) or (enhance and beat_this and not st["bars_model"]):
        need.add("analysis")
    if demucs and not st["vocals"]:
        need.add("vocals")
    if not st["key"]:
        need.add("key")
    if demucs and stems and not (st["stems"] and st["stems4"]):
        need.add("stems")            # tracks with only vocals + instrumental get drums / bass / other too
    return need


def report() -> None:
    """Every track JSON in AUDIO_DIR: bar lines (model / estimated), vocals."""
    rows = []
    for p in sorted(Path(AUDIO_DIR).glob("*.json")):
        t = read_json(p)
        if not isinstance(t, dict) or "duration_sec" not in t:
            continue
        rows.append((t.get("track_id", p.stem), track_status(t)))
    if not rows:
        print(f"No track JSONs in {AUDIO_DIR}.")
        return
    w = min(60, max(len(tid) for tid, _ in rows))
    yn = lambda b: "yes" if b else "no"     # noqa: E731
    print(f"{'Track':<{w}}  {'Bar lines':<10} {'Vocals':<7} {'Key':<4} {'Stems':<6} {'Made by':<10} Analysed")
    print("-" * (w + 54))
    for tid, st in rows:
        bars = "model" if st["bars_model"] else ("estimated" if st["analysed"] else "—")
        print(f"{tid[:w]:<{w}}  {bars:<10} {yn(st['vocals']):<7} {yn(st['key']):<4} {('4' if st['stems4'] else 'yes') if st['stems'] else 'no':<6} "
              f"{st['by']:<10} {st['when']}")
    n = len(rows)
    est = sum(1 for _, st in rows if st["analysed"] and not st["bars_model"])
    none = sum(1 for _, st in rows if not st["analysed"])
    novoc = sum(1 for _, st in rows if not st["vocals"])
    nokey = sum(1 for _, st in rows if not st["key"])
    nostem = sum(1 for _, st in rows if not (st["stems"] and st["stems4"]))
    print(f"\n{n} track(s): {n - est - none} with model bar lines, {est} estimated, "
          f"{none} without the extra analysis; {novoc} without vocals, {nokey} without a key, "
          f"{nostem} without all the stems (vocals, instrumental, drums, bass, other).")
    if none or novoc or nokey or nostem:
        print("run_audio_analysis.bat fills in whatever is missing.")
    if est:
        print("Estimated bar lines: enhance_audio_analysis.bat (or --enhance) redoes them with beat_this.")


def plan_jobs(library: Path, only: str | None, *, fresh: bool = False, **need_kw) -> list:
    """[(audio_file, track_id, parts)] for every music file whose track is missing something
    (see missing_parts); parts includes "fresh" with --fresh (beats redone too)."""
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
        need = missing_parts(existing, **need_kw)
        if fresh and existing is not None:
            need |= {"fresh", "analysis"}
        if need:
            jobs.append((f, tid, need))
    return jobs


def describe(parts: set) -> str:
    if "new" in parts:
        return "new — full analysis" + (" + stems" if "stems" in parts else "")
    names = {"fresh": "from scratch (beats too)", "analysis": "extra analysis (beats kept)", "vocals": "vocals",
             "key": "key", "stems": "stems"}
    order = ["fresh", "analysis", "vocals", "key", "stems"]
    shown = [names[p] for p in order if p in parts and not (p == "analysis" and "fresh" in parts)]
    return "missing: " + ", ".join(shown)


def stems_dir(tid: str) -> Path:
    return drive_root() / STEMS_FOLDER / tid


def stem_ref(tid: str, part: str) -> str:
    """How the app refers to a stem — like a track's source_path."""
    return f"/content/drive/MyDrive/{STEMS_FOLDER}/{tid}/{part}.m4a"


def save_stem(y, sr: int, dest: Path) -> None:
    """Encode a stem (float32 (n, 2)) to AAC locally, then copy it into the Drive folder in one go."""
    import subprocess
    import numpy as np
    LOCAL_TEMP_DIR.mkdir(exist_ok=True)
    tmp = LOCAL_TEMP_DIR / f"{dest.parent.name}_{dest.name}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, input=np.ascontiguousarray(y, dtype=np.float32).tobytes(), check=True, capture_output=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp, dest)
    tmp.unlink(missing_ok=True)


def run_job(audio_file: Path, tid: str, parts: set, rel: str, with_vocals: bool) -> str:
    """Do the parts this track is missing, from one local copy of the song and one Demucs pass."""
    json_path = AUDIO_DIR / f"{tid}.json"
    say = lambda m: print(f"      … {m}", flush=True)     # noqa: E731
    LOCAL_TEMP_DIR.mkdir(exist_ok=True)
    local = LOCAL_TEMP_DIR / audio_file.name
    t0 = time.time()
    shutil.copy2(audio_file, local)   # one fetch through Drive Stream
    done = []
    try:
        new = "new" in parts or "fresh" in parts
        analysis = new or "analysis" in parts
        sep = None
        if with_vocals and ("stems" in parts or "vocals" in parts or analysis):
            say("separating vocals, drums, bass and other (Demucs)")
            four, sr = AA.separate_all(local)
            inst = sum(four[k] for k in four if k != "vocals").astype("float32")
            sep = (four["vocals"], inst, sr, four)
        existing = None if new else read_json(json_path)
        result = None
        if analysis:
            result = AA.analyse_file(local, existing=existing, with_vocals=with_vocals, progress=say,
                                     model_beats=True, vocal_stem=(sep[0], sep[2]) if sep else None)
            done.append("full analysis" if new else "extra analysis")
        elif "key" in parts:
            say("finding the key")
            result = {"key": AA.key_for_file(local)}
            done.append("key")
        stems = None
        if sep is not None and "stems" in parts:
            for part, y in (("vocals", sep[0]), ("instrumental", sep[1]),
                            *((k, sep[3][k]) for k in ("drums", "bass", "other") if k in sep[3])):
                say(f"saving the {part} stem")
                save_stem(y, sep[2], stems_dir(tid) / f"{part}.m4a")
            stems = {**{p: stem_ref(tid, p) for p in ALL_STEMS if p in STEM_PARTS or p in sep[3]},
                     "model": "htdemucs",
                     "made_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            done.append("stems")
    finally:
        local.unlink(missing_ok=True)

    current = read_json(json_path)   # re-read: never clobber an edit made meanwhile
    if current is not None and not new:
        track = current
        if result:
            track.update(result)
        if not analysis and sep is not None and "vocals" in parts and track.get("analysis"):
            # vocals only (the analysis is up to date): just the vocal curve and sung phrases
            a = track["analysis"]
            n_out = int(len(a.get("kick") or []) or round(float(track["duration_sec"]) * AA.OUT_RATE))
            v = AA.vocal_curve_from(sep[0], sep[2], n_out)
            a["vocals"] = [round(float(x), 2) for x in v]
            a["vocal_starts"], a["vocal_ends"] = AA.vocal_phrases(v)
            a.setdefault("methods", {})["vocals"] = "demucs"
            done.append("vocals")
    else:
        track = {"track_id": tid, **(current or {}), **(result or {})}
    if stems:
        track["stems"] = stems
    track.setdefault("track_id", tid)
    track["source_path"] = track.get("source_path") or f"/content/drive/MyDrive/{rel}"
    write_json(json_path, track)

    a = (result or {}).get("analysis")
    bits = [f"{time.time() - t0:.0f}s", " + ".join(done) or "nothing new"]
    if a and track.get("bpm"):
        bits += [f"{track['bpm']:.1f} BPM", f"{len(a['downbeats'])} bars ({a['methods']['downbeats']})",
                 f"{len(a['sections'])} sections", f"{len(a['drops'])} drop(s)"]
    if track.get("key"):
        bits.append(f"key {track['key']['name']} ({track['key']['camelot']})")
    return " · ".join(bits)


def main() -> None:
    global AA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", help=f"folder of music (default: <Drive>/{LIBRARY_FOLDER})")
    ap.add_argument("--only", help="only tracks whose file name contains this text")
    ap.add_argument("--upgrade", action="store_true", help=argparse.SUPPRESS)   # the default now
    ap.add_argument("--setup", action="store_true", help="check / install the packages and ffmpeg, then stop")
    ap.add_argument("--report", action="store_true", help="list every track: bar lines (model or estimated), "
                                                          "vocals, key and stems, then stop")
    ap.add_argument("--enhance", action="store_true", help="also redo tracks whose bar lines were estimated "
                                                           "(their beats are kept)")
    ap.add_argument("--redo", action="store_true", help="re-run the analysis on already-analysed tracks "
                                                        "(their beats are still kept)")
    ap.add_argument("--fresh", action="store_true", help="re-analyse from scratch, replacing beats, bpm, "
                                                         "energy and hires too (can shift existing plans)")
    ap.add_argument("--no-vocals", action="store_true", help="skip Demucs altogether (no vocals, no stems)")
    ap.add_argument("--no-stems", action="store_true", help="don't make stems (vocals + instrumental)")
    ap.add_argument("--cpu", action="store_true", help="don't use the GPU")
    ap.add_argument("--dry-run", action="store_true", help="list what would be analysed, then stop")
    args = ap.parse_args()
    if args.setup:
        sys.exit(setup())
    if args.report:
        import audio_analysis as AA
        report()
        return

    device = pick_device(args.cpu)
    os.environ["AUDIO_ANALYSIS_DEVICE"] = device       # read when audio_analysis is imported
    os.environ["AUDIO_ANALYSED_BY"] = "local"
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
    print(f"Demucs:     {'yes — vocals and stems' if vocals_on else ('off (--no-vocals)' if args.no_vocals else 'not installed — no vocals or stems (pip install demucs)')}")
    stems_on = vocals_on and not args.no_stems
    if stems_on:
        print(f"Stems:      {root / STEMS_FOLDER}")

    if args.enhance and not AA.beat_this_available():
        sys.exit("--enhance needs beat_this for the bar lines — run run_audio_analysis.bat once to install it.")
    jobs = plan_jobs(library, args.only, fresh=args.fresh, demucs=vocals_on, stems=stems_on,
                     beat_this=AA.beat_this_available(), enhance=args.enhance, redo=args.redo)
    n_new = sum(1 for _, _, p in jobs if "new" in p)
    print(f"\n{len(jobs)} track(s) to do" + (f" ({n_new} newly added)" if n_new else "") + ".")
    for f, tid, parts in jobs:
        print(f"  • {tid}  ({describe(parts)})")
    if args.dry_run or not jobs:
        if not jobs:
            print("Every track has everything — nothing to do.")
        return

    done, failed = 0, []
    started = time.time()
    try:
        for i, (f, tid, parts) in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] {tid}")
            try:
                rel = f.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                rel = f"{LIBRARY_FOLDER}/{f.relative_to(library).as_posix()}"
            try:
                summary = run_job(f, tid, parts, rel, vocals_on)
                print(f"    ✓ {summary}")
                done += 1
            except Exception as e:
                traceback.print_exc()
                print(f"    ✗ failed: {e}")
                failed.append(tid)
    except KeyboardInterrupt:
        print("\nStopped — tracks finished so far are saved. Run again to carry on.")

    mins = (time.time() - started) / 60
    print(f"\nDone: {done} track(s), {len(failed)} failed, in {mins:.1f} min "
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
    pre = ("import os\nif os.name == 'nt':\n    import ctypes; "
           "ctypes.windll.kernel32.SetErrorMode(0x8003)\n")
    r = subprocess.run([sys.executable, "-c", pre + code], capture_output=True, text=True)
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
