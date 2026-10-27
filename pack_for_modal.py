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
    python pack_for_modal.py                     the 10-video trial (default)
    python pack_for_modal.py --trial 5           a smaller trial
    python pack_for_modal.py --all               every video with a preview copy
    python pack_for_modal.py --only vid05 vid06  just these
    python pack_for_modal.py --dry-run           show what would be packed, copy nothing
    python pack_for_modal.py --upload            ... then upload the pack to the Modal Volume
                                                 (needs: pip install modal, then modal setup)
Safe to re-run: preview copies already in the pack (same size) aren't copied again.

Trial picks: videos you've already labelled in 🎯 Training Labels first (so the big model can
be scored against your answers), then a spread of lengths and actors.
"""

from __future__ import annotations

import argparse
import json
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


def load_catalogues() -> dict:
    """{video_id: catalogue} for every video catalogue in CATALOGUE_DIR."""
    out = {}
    for p in sorted(CATALOGUE_DIR.glob("*.json")):
        if p.name in NOT_CATALOGUES or p.name.startswith("."):
            continue
        cat = _read_json(p)
        if isinstance(cat, dict) and "video_id" in cat and "scenes" in cat:
            out[cat["video_id"]] = cat
    return out


def library_meta() -> dict:
    meta = _read_json(CATALOGUE_DIR / "library_meta.json", {})
    return meta if isinstance(meta, dict) else {}


def labelled_counts() -> dict:
    data = _read_json(CATALOGUE_DIR / "training_labels.json", {})
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

def upload(out: Path, volume: str) -> int:
    if not shutil.which("modal"):
        print("\nThe modal command isn't installed here. In this venv run:\n"
              "    pip install modal\n    modal setup\nthen run this again with --upload.")
        return 1
    subprocess.run(["modal", "volume", "create", volume], capture_output=True)   # fine if it exists
    print(f"\nUploading {out} to Modal Volume '{volume}' as /pack (the whole folder is sent again; "
          f"--force replaces files already there)…")
    return subprocess.run(["modal", "volume", "put", "--force", volume, str(out), "/pack"]).returncode


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pack preview copies + catalogues for Modal.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--trial", type=int, default=10, help="pack this many videos (default 10)")
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
    cats = load_catalogues()
    meta = library_meta()
    has_proxy = lambda v: (PROXY_DIR / f"{v}.mp4").exists()
    no_proxy = sorted(v for v in cats if not has_proxy(v))

    if args.only:
        unknown = [v for v in args.only if v not in cats]
        if unknown:
            print("Not in the catalogue folder: " + ", ".join(unknown))
        ids = [v for v in args.only if v in cats and has_proxy(v)]
    elif args.all:
        ids = [v for v in sorted(cats) if has_proxy(v)]
    else:
        ids = pick_trial(cats, args.trial, labelled_counts(), meta, has_proxy)
    labelled = labelled_counts()
    print(f"{len(cats)} videos in the catalogue, {len(cats) - len(no_proxy)} with a preview copy. "
          f"Packing {len(ids)}:")
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
        src = CATALOGUE_DIR / name
        if src.exists():
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
    (out / "pack.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [f"Pack {summary['created']}: {len(packed)} videos, {summary['hours']} hours, {summary['gb']} GB "
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
        return upload(out, args.volume)
    print(f"\nNext: run again with --upload to send it to Modal (Volume '{args.volume}').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
