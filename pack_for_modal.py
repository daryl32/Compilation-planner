r"""
pack_for_modal.py — Phase 1 of the ML redesign (claude/ml-redesign-plan.md): gather the
480p preview copies and everything the app already knows about each video into one folder,
ready to upload once to a Modal Volume for the GPU trial / training run.

Runs on the laptop (paths from config.py, like the other local scripts). Nothing in Google
Drive is changed — it only reads.

Writes (default: _modal_pack next to this script, on genuine local disk):
    videos\<video_id>.mp4          the preview copy (same timeline as the original)
    manifests\<video_id>.json      per video: duration / fps / size of the preview copy, the
                                   library range, actors, and every scene (times, tags,
                                   corrected tags, reviewed / excluded, roles, motion number,
                                   motion curve, CLIP embedding, split_from)
    library\                       tag_vocabulary.csv, training_labels.json, labels.json,
                                   library_meta.json, scene_roles.json (copied as they are)
    pack.json                      what's in the pack (videos, hours, GB, problems)
    pack_report.txt                the same, readable

Usage (from the activated venv in the scene-labeling folder, or pack_for_modal.bat):
    python pack_for_modal.py                     the latest batch again (the first run: batch 1)
    python pack_for_modal.py --batch             the NEXT batch: 10 videos not done yet, labelled first
    python pack_for_modal.py --batch 2           batch 2 again
    python pack_for_modal.py --all               every video with a preview copy
    python pack_for_modal.py --only vid05 vid06  just these
    python pack_for_modal.py --dry-run           show what would be packed, copy nothing
    python pack_for_modal.py --upload            ... then upload the pack to the Modal Volume
                                                 (needs: pip install modal, then modal setup)
Safe to re-run: preview copies already in the pack (same size) aren't copied again, and --upload
only sends preview copies the Volume doesn't have. After a good upload, earlier batches' preview
copies are removed from the local pack folder (they're on Modal; their manifests stay).

Trial picks: videos you've already labelled in 🎯 Training Labels first (so the big model can
be scored against your answers), then a spread of lengths and actors.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config
from config import CATALOGUE_DIR

CATALOGUE_DIR = Path(CATALOGUE_DIR)
PROXY_DIR = Path(getattr(config, "PROXY_DIR", CATALOGUE_DIR.parent / "proxies"))
DEFAULT_OUT = Path(__file__).resolve().parent / "_modal_pack"
VOLUME = "media-train"
PACK_VERSION = 1

LIBRARY_FILES = ["tag_vocabulary.csv", "training_labels.json", "labels.json",
                 "library_meta.json", "scene_roles.json"]
NOT_CATALOGUES = {"labels.json", "library_meta.json", "video_index.json", "scene_roles.json",
                  "story_maps.json", "training_labels.json"}
SCENE_FIELDS = ["scene_id", "start_tc", "end_tc", "start_frame", "end_frame", "tags", "corrected_tags",
                "reviewed", "excluded", "roles", "roles_reviewed", "intro_candidate", "outro_candidate",
                "motion_intensity", "motion_curve", "clip_embedding", "split_from"]


# ---------------------------------------------------------------------------
# Reading the library
# ---------------------------------------------------------------------------

def tc_to_seconds(tc: str) -> float:
    h, m, s = str(tc).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_catalogues(progress: bool = False) -> dict:
    """{video_id: catalogue} for every video catalogue in CATALOGUE_DIR. Read 8 at a time:
    with Google Drive in Stream mode each file is downloaded the first time it's read."""
    from concurrent.futures import ThreadPoolExecutor
    paths = [p for p in sorted(CATALOGUE_DIR.glob("*.json"))
             if p.name not in NOT_CATALOGUES and not p.name.startswith(".")]
    out, started = {}, time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, cat in enumerate(pool.map(_read_json, paths), 1):
            if isinstance(cat, dict) and "video_id" in cat and "scenes" in cat:
                out[cat["video_id"]] = cat
            if progress and (i % 10 == 0 or i == len(paths)):
                print(f"\rReading catalogues from Google Drive: {i}/{len(paths)} "
                      f"({time.time() - started:.0f}s)", end="", flush=True)
    if progress:
        print()
    return out


def newest_copy(name: str):
    """The newest of name and Google Drive's duplicates of it ("name (1).csv", …), or None.
    Drive keeps a second file with the same name when the app saves one it can't overwrite;
    the server's sync always takes the newest, so this does too."""
    stem, suffix = Path(name).stem, Path(name).suffix
    found = [p for p in CATALOGUE_DIR.glob(f"{stem}*{suffix}")
             if p.name == name or re.fullmatch(re.escape(stem) + r" \(\d+\)" + re.escape(suffix), p.name)]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def library_meta() -> dict:
    src = newest_copy("library_meta.json")
    meta = _read_json(src, {}) if src else {}
    return meta if isinstance(meta, dict) else {}


def labelled_counts() -> dict:
    src = newest_copy("training_labels.json")
    data = _read_json(src, {}) if src else {}
    counts = {}
    for it in (data or {}).get("items") or []:
        if isinstance(it, dict) and not it.get("skipped"):
            counts[it.get("video_id")] = counts.get(it.get("video_id"), 0) + 1
    return counts


def catalogue_duration(cat: dict) -> float:
    try:
        return max(tc_to_seconds(s["end_tc"]) for s in cat["scenes"])
    except (ValueError, KeyError):
        return 0.0


# ---------------------------------------------------------------------------
# Choosing the trial videos
# ---------------------------------------------------------------------------

def pick_trial(cats: dict, n: int, labelled: dict, meta: dict, has_proxy) -> list:
    """Up to n video ids: labelled videos first (most labels first), then a spread of lengths,
    trying not to repeat an actor until every actor has had a turn."""
    ids = [v for v in sorted(cats) if has_proxy(v)]
    chosen = [v for v in sorted(ids, key=lambda v: (-labelled.get(v, 0), v)) if labelled.get(v)][: n // 2]

    def actors(v):
        e = meta.get(v) if isinstance(meta.get(v), dict) else {}
        return {a.lower() for a in e.get("actors") or []}

    rest = sorted((v for v in ids if v not in chosen), key=lambda v: (catalogue_duration(cats[v]), v))
    if rest and len(chosen) < n:
        k = n - len(chosen)
        # Evenly spaced through the lengths (shortest → longest)…
        idx = sorted({round(i * (len(rest) - 1) / max(k - 1, 1)) for i in range(k)}) if k > 1 else [len(rest) // 2]
        wanted = [rest[i] for i in idx]
        # …swapping a pick for its nearest neighbour when that adds an unseen actor.
        seen = set().union(*(actors(v) for v in chosen)) if chosen else set()
        for i, v in enumerate(wanted):
            if actors(v) and actors(v) <= seen:
                pos = rest.index(v)
                for d in (1, -1, 2, -2, 3, -3):
                    j = pos + d
                    if 0 <= j < len(rest) and rest[j] not in wanted and not actors(rest[j]) <= seen:
                        wanted[i] = rest[j]
                        break
            seen |= actors(wanted[i])
        chosen += wanted
    return chosen[:n]


# ---------------------------------------------------------------------------
# The preview copies
# ---------------------------------------------------------------------------

def probe(path: Path) -> dict:
    """Duration, fps, size and whether the frame rate is variable — from ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
           "-of", "json", str(path)]
    try:
        out = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {"error": f"ffprobe failed: {e}"}
    st = (out.get("streams") or [{}])[0]

    def rate(s):
        try:
            a, b = str(s).split("/")
            return float(a) / float(b) if float(b) else 0.0
        except ValueError:
            return 0.0

    r, avg = rate(st.get("r_frame_rate")), rate(st.get("avg_frame_rate"))
    return {
        "width": st.get("width"), "height": st.get("height"),
        "fps": round(avg or r, 3),
        "variable_fps": bool(r and avg and abs(r - avg) / r > 0.01),
        "duration": round(float((out.get("format") or {}).get("duration") or 0.0), 3),
    }


def copy_if_changed(src: Path, dest: Path) -> bool:
    """Copy src → dest unless dest already has the same size. True if it copied."""
    try:
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            return False
    except OSError:
        pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(dest)
    return True


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def manifest(cat: dict, info: dict, meta_entry: dict, proxy: Path) -> dict:
    scenes = []
    for s in cat["scenes"]:
        out = {k: s[k] for k in SCENE_FIELDS if k in s}
        try:
            out["start"] = round(tc_to_seconds(s["start_tc"]), 3)
            out["end"] = round(tc_to_seconds(s["end_tc"]), 3)
        except (KeyError, ValueError):
            pass
        scenes.append(out)
    e = meta_entry if isinstance(meta_entry, dict) else {}
    return {
        "pack_version": PACK_VERSION,
        "video_id": cat["video_id"],
        "proxy": {"file": f"videos/{proxy.name}", "bytes": proxy.stat().st_size, **info},
        "catalogue_duration": round(catalogue_duration(cat), 3),
        "library_range": e.get("range"),
        "actors": list(e.get("actors") or []),
        "video_tags": list(e.get("video_tags") or []),
        "scenes": scenes,
    }


def check(info: dict, cat_dur: float) -> list:
    problems = []
    if info.get("error"):
        return [info["error"]]
    if info.get("variable_fps"):
        problems.append("variable frame rate")
    if cat_dur and info.get("duration") and abs(info["duration"] - cat_dur) > max(2.0, 0.02 * cat_dur):
        problems.append(f"preview is {info['duration']:.0f}s but the scenes end at {cat_dur:.0f}s")
    if (info.get("height") or 0) > 720:
        problems.append(f"preview is {info['height']}p (expected 480p)")
    return problems


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def remote_videos(volume: str) -> dict | None:
    """{file name: size} of the preview copies already on the Volume, or None if unknown."""
    r = subprocess.run(["modal", "volume", "ls", "--json", volume, "/pack/videos"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        rows = json.loads(r.stdout)
    except ValueError:
        return None
    out = {}
    for row in rows if isinstance(rows, list) else []:
        raw = str(row.get("filename") or row.get("Filename") or "")
        try:        # the CLI escapes non-ASCII characters (é → \xe9) — undo that
            raw = raw.encode("ascii", "backslashreplace").decode("unicode_escape")
        except UnicodeError:
            pass
        name = Path(raw).name
        size = row.get("size", row.get("Size"))
        if name:
            out[name] = size if isinstance(size, int) else None
    return out


def upload(out: Path, volume: str) -> int:
    """Send the pack to the Volume (as /pack): manifests, library files and the summaries every
    time (they're small); preview copies only when they aren't there yet (or changed size)."""
    if not shutil.which("modal"):
        print("\nThe modal command isn't installed here. In this venv run:\n"
              "    pip install modal\n    modal setup\nthen run this again with --upload.")
        return 1
    subprocess.run(["modal", "volume", "create", volume], capture_output=True)   # fine if it exists
    have = remote_videos(volume)
    videos = sorted((out / "videos").glob("*.mp4"))
    todo = [v for v in videos if have is None or have.get(v.name) not in (v.stat().st_size, None)
            or v.name not in have]
    print(f"\nUploading to Modal Volume '{volume}': {len(todo)} preview cop{'y' if len(todo) == 1 else 'ies'} "
          f"({sum(v.stat().st_size for v in todo) / 1e9:.1f} GB; {len(videos) - len(todo)} already there), "
          f"plus manifests and library files…")
    for i, v in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {v.name}")
        rc = subprocess.run(["modal", "volume", "put", "--force", volume, str(v), f"/pack/videos/{v.name}"]).returncode
        if rc:
            return rc
    for folder in ("manifests", "library"):
        rc = subprocess.run(["modal", "volume", "put", "--force", volume, str(out / folder), f"/pack/{folder}"]).returncode
        if rc:
            return rc
    for name in ("pack.json", "batches.json"):
        if (out / name).exists():
            rc = subprocess.run(["modal", "volume", "put", "--force", volume, str(out / name), f"/pack/{name}"]).returncode
            if rc:
                return rc
    print("Upload done.")
    return 0


# ---------------------------------------------------------------------------
# Batches — the library goes through Modal 10 videos at a time, so you can correct each batch
# in the app (🧪 Trial results) before the next one runs. batches.json in the pack folder:
# {"1": [video ids], "2": [...]}.
# ---------------------------------------------------------------------------

BATCH_SIZE = 10


def load_batches(out: Path) -> dict:
    data = _read_json(out / "batches.json", None)
    if isinstance(data, dict) and data:
        return {str(k): list(v) for k, v in data.items()}
    # A pack made before batches existed: what's in it is batch 1.
    packed = sorted(p.stem for p in (out / "manifests").glob("*.json")) if (out / "manifests").exists() else []
    return {"1": packed} if packed else {}


def next_batch(cats: dict, batches: dict, labelled: dict, meta: dict, has_proxy, size: int = BATCH_SIZE) -> list:
    """The next `size` videos that aren't in any batch yet — labelled ones first, then a spread
    of lengths and actors (the same picking as the first trial)."""
    done = {v for ids in batches.values() for v in ids}
    rest = {v: c for v, c in cats.items() if v not in done}
    return pick_trial(rest, size, labelled, meta, has_proxy)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pack preview copies + catalogues for Modal.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--batch", nargs="?", const="next", default=None, metavar="N",
                   help="--batch: the next 10 new videos · --batch N: batch N again "
                        "(no option: the latest batch again, e.g. after editing tags)")
    g.add_argument("--trial", type=int, default=None, help=argparse.SUPPRESS)   # old name for batch 1
    g.add_argument("--all", action="store_true", help="pack every video with a preview copy")
    g.add_argument("--only", nargs="+", metavar="VIDEO_ID", help="pack just these videos")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"pack folder (default {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true", help="list what would be packed, copy nothing")
    ap.add_argument("--upload", action="store_true", help=f"upload the pack to Modal Volume '{VOLUME}'")
    ap.add_argument("--volume", default=VOLUME, help="Modal Volume name")
    args = ap.parse_args(argv)

    if not shutil.which("ffprobe"):
        print("ffprobe isn't on PATH — install ffmpeg (winget install ffmpeg) and open a new terminal.")
        return 1
    print(f"Catalogues:     {CATALOGUE_DIR}\nPreview copies: {PROXY_DIR}\nPack folder:    {args.out}\n")
    print("The first run downloads every catalogue from Drive (a few minutes); later runs are quicker.")
    cats = load_catalogues(progress=True)
    meta = library_meta()
    has_proxy = lambda v: (PROXY_DIR / f"{v}.mp4").exists()
    no_proxy = sorted(v for v in cats if not has_proxy(v))

    labelled = labelled_counts()
    batches = load_batches(args.out)
    batch_no = None
    if args.only:
        unknown = [v for v in args.only if v not in cats]
        if unknown:
            print("Not in the catalogue folder: " + ", ".join(unknown))
        ids = [v for v in args.only if v in cats and has_proxy(v)]
    elif args.all:
        ids = [v for v in sorted(cats) if has_proxy(v)]
    elif args.trial is not None and not batches:
        batch_no, ids = "1", pick_trial(cats, args.trial, labelled, meta, has_proxy)
    elif args.batch == "next":
        batch_no = str(max((int(k) for k in batches), default=0) + 1)
        ids = next_batch(cats, batches, labelled, meta, has_proxy)
        if not ids:
            print("Every video with a preview copy is already in a batch — nothing new to pack.")
            return 0
    elif args.batch is not None:
        if not str(args.batch).isdigit() or str(int(args.batch)) not in batches:
            print(f"There's no batch {args.batch} yet (batches so far: {', '.join(sorted(batches, key=int)) or 'none'}). "
                  f"Use --batch for the next one.")
            return 1
        batch_no = str(int(args.batch))
        ids = [v for v in batches[batch_no] if v in cats and has_proxy(v)]
    elif batches:                       # no option: the latest batch again
        batch_no = max(batches, key=int)
        ids = [v for v in batches[batch_no] if v in cats and has_proxy(v)]
    else:                               # first ever run: batch 1
        batch_no, ids = "1", pick_trial(cats, BATCH_SIZE, labelled, meta, has_proxy)
    in_batches = len({v for b in batches.values() for v in b} | (set(ids) if batch_no else set()))
    print(f"{len(cats)} videos in the catalogue, {len(cats) - len(no_proxy)} with a preview copy, "
          f"{in_batches} in batches so far.")
    print(f"Packing {'batch ' + batch_no + ' — ' if batch_no else ''}{len(ids)} video(s):")
    for v in ids:
        extra = f", {labelled[v]} labels" if labelled.get(v) else ""
        print(f"  {v}  ({catalogue_duration(cats[v]) / 60:.1f} min, {len(cats[v]['scenes'])} scenes{extra})")
    if args.dry_run:
        print("\nDry run — nothing copied.")
        return 0

    out = args.out
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "library").mkdir(parents=True, exist_ok=True)
    for name in LIBRARY_FILES:
        src = newest_copy(name)
        if src is not None:
            shutil.copyfile(src, out / "library" / name)

    started, rows, copied = time.time(), [], 0
    for i, v in enumerate(ids, 1):
        src = PROXY_DIR / f"{v}.mp4"
        dest = out / "videos" / src.name
        print(f"[{i}/{len(ids)}] {v} … ", end="", flush=True)
        try:
            copied += copy_if_changed(src, dest)
        except OSError as e:
            print(f"couldn't copy ({e})")
            rows.append({"video_id": v, "problems": [f"copy failed: {e}"]})
            continue
        info = probe(dest)
        m = manifest(cats[v], info, meta.get(v), dest)
        (out / "manifests" / f"{v}.json").write_text(json.dumps(m), encoding="utf-8")
        problems = check(info, m["catalogue_duration"])
        rows.append({"video_id": v, "duration": info.get("duration") or 0.0, "fps": info.get("fps"),
                     "bytes": dest.stat().st_size, "scenes": len(m["scenes"]), "problems": problems})
        print("ok" if not problems else "⚠ " + "; ".join(problems))

    # Videos packed earlier (e.g. the trial, then --only more) stay in the pack.
    packed = sorted(p.stem for p in (out / "manifests").glob("*.json"))
    total_bytes = sum((out / "videos" / f"{v}.mp4").stat().st_size
                      for v in packed if (out / "videos" / f"{v}.mp4").exists())
    total_sec = sum((_read_json(out / "manifests" / f"{v}.json", {}).get("proxy") or {}).get("duration") or 0
                    for v in packed)
    summary = {
        "pack_version": PACK_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "videos": packed, "hours": round(total_sec / 3600, 2), "gb": round(total_bytes / 1e9, 2),
        # The whole library (every catalogue) — the trial report scales its costs to this.
        "library_hours": round(sum(catalogue_duration(c) for c in cats.values()) / 3600, 2),
        "library_videos": len(cats),
        "library_files": [n for n in LIBRARY_FILES if (out / "library" / n).exists()],
        "this_run": rows, "no_preview_copy": no_proxy,
    }
    if batch_no:
        batches[batch_no] = list(ids)
        (out / "batches.json").write_text(json.dumps(batches, indent=1), encoding="utf-8")
        summary["batch"] = batch_no
        summary["batches"] = {k: len(v) for k, v in sorted(batches.items(), key=lambda kv: int(kv[0]))}
    (out / "pack.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [(f"Batch {batch_no}: {len(ids)} videos. " if batch_no else "")
             + f"Pack {summary['created']}: {len(packed)} videos, {summary['hours']} hours, {summary['gb']} GB "
             f"(whole library: {len(cats)} videos, {summary['library_hours']} hours)",
             f"This run: {len(rows)} videos, {copied} preview copies copied, {time.time() - started:.0f}s",
             "Library files: " + (", ".join(summary["library_files"]) or "none")]
    flagged = [r for r in rows if r["problems"]]
    if flagged:
        lines.append("Needs a look:")
        lines += [f"  {r['video_id']}: {'; '.join(r['problems'])}" for r in flagged]
    if "tag_vocabulary.csv" not in summary["library_files"]:
        lines.append("No tag_vocabulary.csv in the catalogue folder yet — the trial will use labels.json.")
    if no_proxy:
        lines.append(f"{len(no_proxy)} videos have no preview copy (run_proxies.bat makes them): "
                     + ", ".join(no_proxy[:10]) + (" …" if len(no_proxy) > 10 else ""))
    (out / "pack_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    if args.upload:
        rc = upload(out, args.volume)
        if rc == 0 and batch_no:
            freed = 0
            for f in (out / "videos").glob("*.mp4"):
                if f.stem not in ids:
                    freed += f.stat().st_size
                    f.unlink()
            if freed:
                print(f"Removed earlier batches' preview copies from {out / 'videos'} ({freed / 1e9:.1f} GB) "
                      f"— they're on Modal.")
        return rc
    print(f"\nNext: run again with --upload to send it to Modal (Volume '{args.volume}'), then "
          f"run_modal_trial.bat --full (it skips videos already done).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
