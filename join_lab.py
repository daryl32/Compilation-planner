"""🧩 Join Lab (19 Nov 2026) — how well the avatar's clips join, and which bridges are really needed.

Daryl wants to plan idles and joins between her clips, see which tools are needed and whether bridging clips will be
needed at all. The Lab works on the server, from the small H.264 copies the Studio previews job keeps on Drive
(Edge_voiceover/_studio_previews/<drive id>.mp4 — quick to fetch, every clip the same codec), falling back to the
clip itself.

  📥 Sync      every avatar clip in Edge_voiceover (voice clips in "<Outfit> - <Pose>/avatar_clips", idles,
               transitions) + each look's home-pose picture (the .png in its folder) → <data>/edge_avatar/lab/
  🔍 Scores    each clip's first and last frame against the look's home picture and against each other:
               ✅ 90+  a short crossfade hides the join
               🟡 75–90 a quick AI in-between (frame interpolation, ~0.5–1 s) is needed
               🔴 <75  a bridge: an idle / transition clip, or a generated one
  🎞️ Try       a sequence of clips (trim, ping-pong idles to any length, crossfade) rendered as a small preview,
               with every join scored

Frames are compared small (64×36 grey for the shape, 16×9 colour for the light), so a slight shift of the
camera or her hair doesn't count; the background does, so a transparent stand-in never joins a background clip.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from config import CATALOGUE_DIR

ROOT = "scene-labeling/Edge_voiceover"
PREVIEW_FOLDER = f"{ROOT}/_studio_previews"
VIDEO_EXT = (".mp4", ".mov", ".webm", ".m4v")
GOOD, OK = 90.0, 75.0                       # ✅ crossfade / 🟡 AI in-between / 🔴 bridge
GREY, COLOUR = (64, 36), (16, 9)
FPS, WIDTH = 30, 540


def lab_dir() -> Path:
    return Path(CATALOGUE_DIR).parent / "edge_avatar" / "lab"


def index_path() -> Path:
    return lab_dir() / "index.json"


def load_index() -> dict:
    """{"clips": {drive id: {...}}, "homes": {look: file}, "synced": time}."""
    try:
        return json.loads(index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"clips": {}, "homes": {}, "synced": 0}


def save_index(idx: dict) -> None:
    lab_dir().mkdir(parents=True, exist_ok=True)
    index_path().write_text(json.dumps(idx, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# What a clip is
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def classify(name: str, folder: str) -> dict:
    """{"kind": voice | idle | transition | other, "look", "outfit", "pose", "to"} from where the clip sits and its
    name. Look = the "<Outfit> - <Pose>" folder it's in; Studio names: outfit_pose_idle_NN, outfit_from-to-to_transition_NN."""
    parts = [p for p in str(folder or "").split("/") if p]
    top = parts[0] if parts else ""
    look = top if " - " in top else ""
    stem = Path(name).stem
    low = stem.lower()
    if "avatar_clips" in parts:
        kind = "voice"
    elif "_idle" in low or "idle" in [p.lower() for p in parts] or low.startswith("idle"):
        kind = "idle"
    elif "_transition" in low or "transitions" in [p.lower() for p in parts]:
        kind = "transition"
    else:
        kind = "other"
    outfit, pose, to = "", "", ""
    if look:
        outfit, pose = (x.strip() for x in look.split(" - ", 1))
    m = re.match(r"(?P<outfit>[^_]+)_(?P<from>.+?)-to-(?P<to>.+?)_transition", low)
    if m:
        outfit = outfit or m["outfit"].replace("-", " ")
        pose, to = m["from"].replace("-", " "), m["to"].replace("-", " ")
    m = re.match(r"(?P<outfit>[^_]+)_(?P<pose>.+?)_idle", low)
    if m:
        outfit = outfit or m["outfit"].replace("-", " ")
        pose = pose or m["pose"].replace("-", " ")
    return {"kind": kind, "look": look, "outfit": outfit, "pose": pose, "to": to}


def look_key(outfit: str, pose: str) -> str:
    return _norm(outfit) + "|" + _norm(pose)


def clip_look(c: dict, looks: list) -> str:
    """The look a clip belongs to: its folder's, else the look whose outfit + pose match its name's."""
    if c.get("look"):
        return c["look"]
    want = look_key(c.get("outfit", ""), c.get("pose", ""))
    for lk in looks:
        o, p = (lk.split(" - ", 1) + [""])[:2]
        if look_key(o, p) == want:
            return lk
    return ""


# ---------------------------------------------------------------------------
# Frames and scores
# ---------------------------------------------------------------------------

def probe_seconds(path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _grab(path, t: float | None, size: tuple, grey: bool) -> np.ndarray | None:
    """One frame, small. t=None: the last frame."""
    w, h = size
    fmt = "gray" if grey else "rgb24"
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if t is None:
        cmd += ["-sseof", "-0.25"]
    elif t > 0:
        cmd += ["-ss", f"{t:.3f}"]
    cmd += ["-i", str(path), "-vf", f"scale={w}:{h}:flags=area,format={fmt}", "-f", "rawvideo", "-pix_fmt", fmt, "pipe:1"]
    if t is None:                                   # every frame of the last 0.25 s; keep the final one
        r = subprocess.run(cmd, capture_output=True)
    else:
        r = subprocess.run(cmd[:-1] + ["-frames:v", "1", "pipe:1"], capture_output=True)
    n = w * h * (1 if grey else 3)
    buf = r.stdout
    if len(buf) < n:
        return None
    arr = np.frombuffer(buf[-n:] if t is None else buf[:n], dtype=np.uint8).astype(np.float32)
    return arr.reshape(h, w) if grey else arr.reshape(h, w, 3)


def features(path, t: float | None = 0.0) -> dict | None:
    """{"g": 64×36 grey, "c": 16×9 colour} of the frame at t (None = the last frame)."""
    g = _grab(path, t, GREY, True)
    c = _grab(path, t, COLOUR, False)
    if g is None or c is None:
        return None
    return {"g": g.tolist(), "c": c.tolist()}


def image_features(path) -> dict | None:
    """The same for a still picture (the look's home pose)."""
    return features(path, 0.0)


def score(a: dict | None, b: dict | None) -> float | None:
    """How alike two frames are, 0–100 (100 = the same picture). Shape (grey, slightly blurred out by the small size)
    counts 70 %, colour and light 30 %."""
    if not a or not b:
        return None
    ga, gb = np.asarray(a["g"], np.float32), np.asarray(b["g"], np.float32)
    ca, cb = np.asarray(a["c"], np.float32), np.asarray(b["c"], np.float32)
    mad = 0.7 * float(np.mean(np.abs(ga - gb))) + 0.3 * float(np.mean(np.abs(ca - cb)))
    return round(max(0.0, min(100.0, 100.0 * (1.0 - mad / 255.0 * 4.0))), 1)


def band(s: float | None) -> str:
    if s is None:
        return "?"
    return "crossfade" if s >= GOOD else "in-between" if s >= OK else "bridge"


BAND_ICON = {"crossfade": "✅", "in-between": "🟡", "bridge": "🔴", "?": "⚪"}


# ---------------------------------------------------------------------------
# 📥 Sync (a server job)
# ---------------------------------------------------------------------------

def sync(job, redo: bool = False) -> dict:
    """Fetch every avatar clip (the small preview copy when there is one) and each look's home picture, and work out
    each clip's first / last frame."""
    import studio_previews as SP
    from drive_sync import download_drive_file, list_drive_folder
    job.progress(0.0, "Listing the clips")
    found, err = SP.clips()
    if err:
        raise RuntimeError(err)
    pre, _ = list_drive_folder(PREVIEW_FOLDER)
    pre = {f["name"]: f["id"] for f in pre or []}
    idx = {"clips": {}, "homes": {}, "synced": 0} if redo else load_index()
    old = idx.get("clips") or {}
    clips_dir = lab_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    errs, fetched = [], 0
    live = {c["id"] for c in found}
    for k, c in enumerate(found):
        job.progress(0.05 + 0.85 * k / max(len(found), 1), f"{c['name']} ({k + 1}/{len(found)})")
        info = {**classify(c["name"], c["folder"]), "name": c["name"], "folder": c["folder"], "size": c["size"]}
        dest = clips_dir / f"{c['id']}.mp4"
        prev = old.get(c["id"]) or {}
        if prev.get("size") == c["size"] and dest.exists() and prev.get("first") and prev.get("last"):
            old_info = {k_: prev[k_] for k_ in ("seconds", "first", "last", "src") if k_ in prev}
            idx["clips"][c["id"]] = {**info, **old_info}
            continue
        try:
            if f"{c['id']}.mp4" in pre:                   # the Studio's small copy
                download_drive_file(pre[f"{c['id']}.mp4"], dest)
                src = "preview"
            else:                                         # no preview yet: the clip itself, made small here
                raw = clips_dir / f"{c['id']}.src{Path(c['name']).suffix.lower()}"
                download_drive_file(c["id"], raw)
                e = SP.make_preview(raw, dest)
                raw.unlink(missing_ok=True)
                if e:
                    raise RuntimeError(e)
                src = "original"
            fetched += 1
            info.update(seconds=round(probe_seconds(dest), 3), first=features(dest, 0.0), last=features(dest, None), src=src)
            idx["clips"][c["id"]] = info
        except Exception as ex:                           # noqa: BLE001
            errs.append(f"{c['name']}: {ex}")
    for cid in list(idx["clips"]):                       # clips gone from Drive
        if cid not in live:
            idx["clips"].pop(cid, None)
    job.progress(0.92, "Home pictures")
    idx["homes"] = sync_homes(sorted({c["look"] for c in idx["clips"].values() if c.get("look")}), errs)
    idx["synced"] = time.time()
    save_index(idx)
    job.progress(1.0, "Done")
    return {"clips": len(idx["clips"]), "fetched": fetched, "homes": len(idx["homes"]), "errors": errs}


def sync_homes(looks: list, errs: list) -> dict:
    """{look: {"file", "f"}}: the first picture (.png / .jpg) in each look's folder — her home pose."""
    from drive_sync import download_drive_file, list_drive_folder
    out = {}
    home_dir = lab_dir() / "homes"
    home_dir.mkdir(parents=True, exist_ok=True)
    for look in looks:
        files, err = list_drive_folder(f"{ROOT}/{look}")
        pics = sorted((f for f in files or [] if f["name"].lower().endswith((".png", ".jpg", ".jpeg"))),
                      key=lambda f: (not f["name"].lower().startswith("avatar"), f["name"].lower()))
        if not pics:
            continue
        dest = home_dir / f"{_norm(look)}{Path(pics[0]['name']).suffix.lower()}"
        try:
            download_drive_file(pics[0]["id"], dest)
            out[look] = {"file": pics[0]["name"], "path": str(dest), "f": image_features(dest)}
        except Exception as ex:                           # noqa: BLE001
            errs.append(f"{look} home picture: {ex}")
    return out


# ---------------------------------------------------------------------------
# 🔍 The report: per look, how each clip starts and ends against her home pose
# ---------------------------------------------------------------------------

def looks_in(idx: dict) -> list:
    return sorted({c["look"] for c in idx["clips"].values() if c.get("look")} | set(idx.get("homes") or {}))


def look_report(idx: dict, look: str) -> list:
    """[{"id", "name", "kind", "seconds", "start", "end", "loop"}] for the look's clips — start / end = score against the
    home picture, loop = its last frame against its own first (an idle that loops cleanly scores high)."""
    looks = looks_in(idx)
    home = ((idx.get("homes") or {}).get(look) or {}).get("f")
    rows = []
    for cid, c in idx["clips"].items():
        if clip_look(c, looks) != look:
            continue
        rows.append({"id": cid, "name": c["name"], "kind": c["kind"], "seconds": c.get("seconds"), "to": c.get("to", ""),
                     "start": score(c.get("first"), home), "end": score(c.get("last"), home),
                     "loop": score(c.get("last"), c.get("first"))})
    order = {"voice": 0, "idle": 1, "transition": 2, "other": 3}
    return sorted(rows, key=lambda r: (order.get(r["kind"], 9), r["name"].lower()))


def summary(rows: list) -> dict:
    """What the look needs: voice clips that end away from home (and so need something after them), per band."""
    voice = [r for r in rows if r["kind"] == "voice"]
    out = {"voice": len(voice), "idles": sum(1 for r in rows if r["kind"] == "idle"),
           "transitions": sum(1 for r in rows if r["kind"] == "transition")}
    for b in ("crossfade", "in-between", "bridge", "?"):
        out["end_" + b] = sum(1 for r in voice if band(r["end"]) == b)
        out["start_" + b] = sum(1 for r in voice if band(r["start"]) == b)
    out["back_to_back"] = None
    return out


def pair_matrix(idx: dict, ids: list) -> list:
    """[[score of a's end → b's start]] for the clips given (rows = from, columns = to)."""
    cl = idx["clips"]
    return [[score(cl[a].get("last"), cl[b].get("first")) if a != b else None for b in ids] for a in ids]


def best_idles(idx: dict, look: str, clip_id: str, n: int = 3) -> list:
    """The look's idles that start closest to where this clip ends: [(score, idle id)]."""
    looks = looks_in(idx)
    c = idx["clips"][clip_id]
    got = [(score(c.get("last"), i.get("first")), iid) for iid, i in idx["clips"].items()
           if i["kind"] == "idle" and clip_look(i, looks) == look]
    return sorted((g for g in got if g[0] is not None), reverse=True)[:n]


# ---------------------------------------------------------------------------
# 🎞️ Try a sequence
# ---------------------------------------------------------------------------

def piece_length(p: dict, seconds: float) -> float:
    a = max(float(p.get("trim_in") or 0.0), 0.0)
    b = float(p.get("trim_out") or 0.0) or seconds
    b = min(b, seconds) if seconds else b
    hold = float(p.get("hold") or 0.0)
    return max(hold, 0.0) if p.get("pingpong") and hold > 0 else max(b - a, 0.05)


def sequence_joins(idx: dict, pieces: list) -> list:
    """Each join's score: the frame where a piece ends (its trim out; a ping-pong ends where it started or turned)
    against where the next begins (its trim in)."""
    out = []
    for a, b in zip(pieces, pieces[1:]):
        ca, cb = idx["clips"][a["id"]], idx["clips"][b["id"]]
        pa, pb = lab_dir() / "clips" / f"{a['id']}.mp4", lab_dir() / "clips" / f"{b['id']}.mp4"
        end_t = _end_time(a, ca.get("seconds") or 0.0)
        fa = ca.get("last") if end_t is None else features(pa, end_t)
        fb = cb.get("first") if not float(b.get("trim_in") or 0) else features(pb, float(b["trim_in"]))
        s = score(fa, fb)
        out.append({"from": ca["name"], "to": cb["name"], "score": s, "band": band(s)})
    return out


def _end_time(p: dict, seconds: float):
    """Where in its clip a piece's last frame is (None = the clip's own last frame)."""
    a = float(p.get("trim_in") or 0.0)
    b = float(p.get("trim_out") or 0.0) or seconds
    if p.get("pingpong") and float(p.get("hold") or 0) > 0:
        span = max(b - a, 0.05)
        k, r = divmod(float(p["hold"]), span)
        return a + (r if int(k) % 2 == 0 else span - r)       # forwards on even passes, backwards on odd
    if b >= seconds - 0.05:
        return None
    return max(b - 0.04, 0.0)


def render_sequence(job, idx: dict, pieces: list, xfade: float = 0.3) -> dict:
    """The pieces joined into one small preview: <lab>/tries/try_<time>.mp4. Each piece: {"id", "trim_in", "trim_out",
    "pingpong", "hold"} — a ping-pong piece plays forwards, backwards, forwards… for `hold` seconds. Joins crossfade over
    `xfade` seconds (0 = a cut). Voice clips keep their sound."""
    if not pieces:
        raise ValueError("No clips in the sequence.")
    out_dir = lab_dir() / "tries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"try_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    job.progress(0.05, "Building")
    inputs, chains, lens, audio = [], [], [], []
    first = lab_dir() / "clips" / f"{pieces[0]['id']}.mp4"
    w, h = _dims(first)
    h = int(round(WIDTH * h / max(w, 1) / 2) * 2) if w else 960
    norm = (f"fps={FPS},scale={WIDTH}:{h}:force_original_aspect_ratio=decrease,pad={WIDTH}:{h}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,format=yuv420p,settb=AVTB")
    for k, p in enumerate(pieces):
        c = idx["clips"][p["id"]]
        path = lab_dir() / "clips" / f"{p['id']}.mp4"
        secs = float(c.get("seconds") or probe_seconds(path))
        a = max(float(p.get("trim_in") or 0.0), 0.0)
        b = min(float(p.get("trim_out") or 0.0) or secs, secs)
        span = max(b - a, 0.05)
        inputs += ["-i", str(path)]
        base = f"[{k}:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS,{norm}"
        if p.get("pingpong") and float(p.get("hold") or 0) > 0:
            hold = float(p["hold"])
            passes = max(int(math.ceil(hold / span)), 1)
            labels = "".join(f"[p{k}_{j}]" for j in range(passes))
            parts = [f"{base},split={passes}{labels}"]
            seq = ""
            for j in range(passes):
                if j % 2:
                    parts.append(f"[p{k}_{j}]reverse[r{k}_{j}]")
                    seq += f"[r{k}_{j}]"
                else:
                    seq += f"[p{k}_{j}]"
            parts.append(f"{seq}concat=n={passes}:v=1:a=0,trim=0:{hold:.3f},setpts=PTS-STARTPTS,fps={FPS},settb=AVTB[v{k}]")
            chains.append(";".join(parts))
            lens.append(hold)
        else:
            chains.append(f"{base}[v{k}]")
            lens.append(span)
            if c["kind"] == "voice" and _has_audio(path):
                audio.append((k, a, b))
    xf = max(float(xfade), 0.0)
    xf = min(xf, min(lens) / 2 - 0.02) if len(lens) > 1 else 0.0
    starts, t = [], 0.0
    for k, L in enumerate(lens):
        starts.append(t)
        t += L - (xf if k < len(lens) - 1 else 0.0)
    total = t
    fc = list(chains)
    cur = "[v0]"
    for k in range(1, len(pieces)):
        nxt = f"[x{k}]" if k < len(pieces) - 1 else "[vout]"
        if xf > 0:
            fc.append(f"{cur}[v{k}]xfade=transition=fade:duration={xf:.3f}:offset={starts[k]:.3f}{nxt}")
        else:
            fc.append(f"{cur}[v{k}]concat=n=2:v=1:a=0{nxt}")
        cur = nxt
    if len(pieces) == 1:
        fc.append("[v0]null[vout]")
    if audio:
        for j, (k, a, b) in enumerate(audio):
            ms = int(round(starts[k] * 1000))
            fc.append(f"[{k}:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,aresample=48000,adelay={ms}|{ms}[a{j}]")
        fc.append("".join(f"[a{j}]" for j in range(len(audio)))
                  + f"amix=inputs={len(audio)}:normalize=0:duration=longest,apad,atrim=0:{total:.3f}[aout]")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", "[vout]"]
    if audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-t", f"{total:.3f}", "-movflags", "+faststart", str(out)]
    job.progress(0.2, "Rendering")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or not out.exists():
        raise RuntimeError((r.stderr or "ffmpeg failed").strip()[-400:])
    job.progress(0.9, "Scoring the joins")
    joins = sequence_joins(idx, pieces)
    job.progress(1.0, "Done")
    return {"file": str(out), "seconds": round(total, 2), "joins": joins, "xfade": xf}


def _dims(path) -> tuple:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        w, h = (int(x) for x in r.stdout.strip().split(",")[:2])
        return w, h
    except ValueError:
        return 0, 0


def _has_audio(path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def clear_tries(keep: int = 5) -> int:
    """Old tries off the server's disk (the newest `keep` stay)."""
    d = lab_dir() / "tries"
    if not d.is_dir():
        return 0
    files = sorted(d.glob("try_*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[keep:]:
        f.unlink(missing_ok=True)
    return max(len(files) - keep, 0)
