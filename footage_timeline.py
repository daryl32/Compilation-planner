r"""
footage_timeline.py — Stage 3 step 3 (claude/plan-nov2-2026.md, claude/new-matcher-design.md §B):
the 🧭 footage timeline — for every half second of every processed video, everything the new matcher
scores a window on, in one small file per video:

    catalogue/footage_timeline/<video_id>.npz
        times        float32 (T,)        the half seconds (the student export's grid, STEP = 0.5 s)
        move         float16 (T,)        person movement (mean of the 15 fps person curve) — raw; scale it
                                         with the library's move_scale in index.json (scaled_move())
        change       float16 (T,)        how fast movement is changing (|slope| over ±1 s, raw units/s)
        bpm          float16 (T,)        the scene's movement BPM at that point (NaN = none found)
        periodicity  float16 (T,)        how regular that movement is, 0–1 (0 = none)
        onset        float16 (T,)        seconds to the nearest push of movement (0 = one starts here)
        scores       uint8   (T, tags)   the student's score per tag 0–100 — your 🎯 / ✏️ / ▦ labels win
                                         within ±LABEL_REACH s of a labelled moment (100 / 0)
        labelled     bool    (T,)        a label of yours covers this half second
        shot         float16 (T,)        shot size 0–3 (extreme close-up → wide), NaN when no shot shows
        people       uint8   (T,)        people on screen (pose detections)
        emb          int8    (T, EMB_DIM) the SigLIP2 picture embedding, PCA → EMB_DIM numbers, unit
                                         length × 127 (dot products ≈ cosine similarity)
        cuts         float32 (C,)        hard cuts (TransNetV2, seconds) — walls a clip window can't cross
    catalogue/footage_timeline/pca.npz     mean (D,), components (EMB_DIM, D) — fitted once on a sample of
                                           the library, kept so every video (and later a text query)
                                           lands in the same space
    catalogue/footage_timeline/index.json  tags, cutoffs, the student model, move_scale, emb_dim, videos
                                           {id: {duration, built, model, labels}}

Scenes, the library range, exclusions, Reviewer scene tags, hair and 👥 cast are NOT baked in: they change
with every Reviewer edit, so the matcher reads them live from the catalogue (catalogue_store) and lays them
over this file. Derived scene tags (the share of a scene a moment tag is visible for) come from
derived_scene_tags() at read time for the same reason.

Inputs (per video):
  • vision data from the Modal / laptop vision stage: <vision root>/videos/<video_id>/ — person_curve.npy
    (15 fps), vision.json (hard cuts + per-scene rhythm), embeddings.npz (frame_times, frame_embs);
    the root is STUDENT_WORK_DIR (env, or config.py), else the Hetzner Volume's student folder, else
    config.py's WORK_DIR — the same places student_tags.py / deploy/student.sh use;
  • the student export catalogue/student_scores/<video_id>.npz (+ index.json) — student_tags.py export;
  • your labels: catalogue/training_labels.json.

Usage:
    python footage_timeline.py build                  every video with vision data + student scores
    python footage_timeline.py build --only <id> …    just these
    python footage_timeline.py build --redo           rebuild videos already built (e.g. after a new export)
    python footage_timeline.py build --refit          fit the PCA again first (then every video is rebuilt)
    python footage_timeline.py status                 what's built, what's missing and why
In the app: 📁 Media Library → 🧭 Timeline (build button + viewer). A build is cheap — numpy only, a few
seconds per video on the server's CPU — and runs at low priority in the background.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

STEP = 0.5                 # seconds between points — the student export's grid
POSE_FPS = 15.0            # the person curve's rate (trial_logic.POSE_FPS)
EMB_DIM = 256              # PCA size for the stored embeddings
PCA_SAMPLE = 20000         # embeddings sampled across the library to fit the PCA
PCA_MIN = 1000             # fewer than this → fit on everything there is
LABEL_REACH = 1.0          # a labelled moment speaks for ± this many seconds
SHOT_MIN = 0.5             # the best shot tag ≥ this share of its cut-off → a shot size is given
CHANGE_HALF = 2            # |slope| of movement over ± this many points (±1 s)
MOVE_PCTS = (5.0, 95.0)    # library scaling of movement: these percentiles → 0 and 1
FOLDER = "footage_timeline"
STATUS_FILE = "build_status.json"
VERSION = 1                # bump when the file layout changes (files with an older version are rebuilt)


# ---------------------------------------------------------------------------
# Where things are
# ---------------------------------------------------------------------------

def catalogue_dir() -> Path:
    from config import CATALOGUE_DIR
    return Path(CATALOGUE_DIR)


def out_dir(folder: Path = None) -> Path:
    return Path(folder) if folder else catalogue_dir() / FOLDER


def vision_root(root: Path = None) -> Path | None:
    """The vision stage's videos folder (…/_student/videos): `root` if given, else STUDENT_WORK_DIR
    (environment or config.py), else the Hetzner Volume's student folder, else config.py's WORK_DIR.
    None when nothing is set and nothing is found."""
    if root is not None:
        return Path(root)
    cands = []
    if os.environ.get("STUDENT_WORK_DIR"):
        cands.append(Path(os.environ["STUDENT_WORK_DIR"]))
    try:
        import config
        if getattr(config, "STUDENT_WORK_DIR", None):
            cands.append(Path(config.STUDENT_WORK_DIR))
    except Exception:          # noqa: BLE001
        config = None
    cands += sorted(Path("/mnt").glob("HC_Volume_*/student"))
    if config is not None and getattr(config, "WORK_DIR", None):
        cands.append(Path(config.WORK_DIR))
    for c in cands:
        v = c / "_student" / "videos"
        if v.is_dir():
            return v
    return None


def scores_dir(folder: Path = None) -> Path:
    return Path(folder) if folder else catalogue_dir() / "student_scores"


def _read_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_index(folder: Path = None) -> dict:
    return _read_json(out_dir(folder) / "index.json", {}) or {}


def signature(folder: Path = None) -> tuple:
    """Changes whenever a build finishes — for the app's caches."""
    try:
        return (out_dir(folder) / "index.json").stat().st_mtime_ns,
    except OSError:
        return (None,)


# ---------------------------------------------------------------------------
# The maths
# ---------------------------------------------------------------------------

def half_second_mean(curve: np.ndarray, times: np.ndarray, fps: float = POSE_FPS) -> np.ndarray:
    """The curve's mean over the half second starting at each time (NaN-aware; NaN where nothing)."""
    c = np.asarray(curve, dtype=np.float64)
    out = np.full(len(times), np.nan, np.float32)
    if c.size == 0:
        return out
    n = int(round(STEP * fps))
    for k, t in enumerate(times):
        a = int(round(float(t) * fps))
        w = c[max(0, a): max(0, a) + n]
        w = w[np.isfinite(w)]
        if w.size:
            out[k] = float(w.mean())
    return out


def change_rate(move: np.ndarray, half: int = CHANGE_HALF) -> np.ndarray:
    """|slope| of the movement over ±half points, per second (NaN-filled first)."""
    m = np.asarray(move, dtype=np.float64)
    if m.size < 2:
        return np.zeros(m.size, np.float32)
    filled = m.copy()
    bad = ~np.isfinite(filled)
    if bad.all():
        return np.zeros(m.size, np.float32)
    if bad.any():
        idx = np.arange(m.size)
        filled[bad] = np.interp(idx[bad], idx[~bad], filled[~bad])
    lo = np.clip(np.arange(m.size) - half, 0, m.size - 1)
    hi = np.clip(np.arange(m.size) + half, 0, m.size - 1)
    span = np.maximum(hi - lo, 1) * STEP
    return (np.abs(filled[hi] - filled[lo]) / span).astype(np.float32)


def rhythm_at(times: np.ndarray, scenes: list, duration: float) -> tuple:
    """(bpm, periodicity, onset) per point from the vision stage's per-scene rhythm
    (trial_logic.rhythm_features: movement_bpm, periodicity, onsets relative to the scene start)."""
    T = len(times)
    bpm = np.full(T, np.nan, np.float32)
    per = np.zeros(T, np.float32)
    onset = np.full(T, np.nan, np.float32)
    all_onsets = []
    for s in scenes or []:
        try:
            a, b = float(s["start"]), float(min(s["end"], duration or s["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        sel = (times >= a) & (times < b)
        if s.get("movement_bpm"):
            bpm[sel] = float(s["movement_bpm"])
        per[sel] = float(s.get("periodicity") or 0.0)
        all_onsets += [a + float(o) for o in (s.get("onsets") or [])]
    if all_onsets:
        on = np.sort(np.asarray(all_onsets, np.float64))
        pos = np.searchsorted(on, times)
        left = on[np.clip(pos - 1, 0, on.size - 1)]
        right = on[np.clip(pos, 0, on.size - 1)]
        d = np.minimum(np.abs(times - left), np.abs(right - times))
        onset = np.where(d < STEP / 2, 0.0, d).astype(np.float32)
    return bpm, per, onset


def shot_columns(tags: list) -> list:
    """[(column, step)] for the shot-size tags present (trial_logic.shot_step)."""
    import trial_logic as TL
    out = []
    for k, t in enumerate(tags):
        s = TL.shot_step(t)
        if s is not None:
            out.append((k, s))
    return out


def shot_value(scores: np.ndarray, cutoffs: np.ndarray, cols: list) -> np.ndarray:
    """Shot size 0–3 per point: the shot tags' steps weighted by their scores, NaN where the best shot tag
    is under SHOT_MIN of its cut-off (claude/new-matcher-design.md, Shot size)."""
    T = scores.shape[0]
    out = np.full(T, np.nan, np.float32)
    if not cols:
        return out
    idx = np.array([c for c, _ in cols])
    steps = np.array([s for _, s in cols], np.float32)
    s = scores[:, idx].astype(np.float32)
    rel = s / np.maximum(cutoffs[idx].astype(np.float32), 1.0)
    ok = rel.max(axis=1) >= SHOT_MIN
    s = np.where(rel >= SHOT_MIN, s, 0.0)         # only shot tags that show count towards the size
    tot = s.sum(axis=1)
    val = (s * steps[None, :]).sum(axis=1) / np.maximum(tot, 1e-6)
    out[ok & (tot > 0)] = val[ok & (tot > 0)]
    return out


def apply_labels(scores: np.ndarray, times: np.ndarray, tags: list, items: list, hidden=None) -> np.ndarray:
    """Your labels win: a full 🎯 / ✏️ label sets its tags to 100 and every other (non-hidden) tag to 0; a ▦ partial
    answer sets only its yes / no tags — within ±LABEL_REACH s of the moment. Returns the mask of labelled points.
    `scores` is changed in place."""
    col = {str(t).lower(): k for k, t in enumerate(tags)}
    hidden = hidden or (lambda t: False)
    labelled = np.zeros(len(times), bool)
    for it in items or []:
        t = it.get("t")
        if t is None or it.get("skipped") or it.get("type") not in ("tags", "partial"):
            continue
        sel = np.abs(times - float(t)) <= LABEL_REACH + 1e-6
        if not sel.any():
            continue
        if it["type"] == "tags":
            yes = {str(x).lower() for x in it.get("tags") or []}
            for name, k in col.items():
                if name in yes:
                    scores[sel, k] = 100
                elif not hidden(name):
                    scores[sel, k] = 0
        else:
            for x in it.get("yes") or []:
                k = col.get(str(x).lower())
                if k is not None:
                    scores[sel, k] = 100
            for x in it.get("no") or []:
                k = col.get(str(x).lower())
                if k is not None:
                    scores[sel, k] = 0
        labelled |= sel
    return labelled


# ---------------------------------------------------------------------------
# PCA for the embeddings
# ---------------------------------------------------------------------------

def fit_pca(sample: np.ndarray, dim: int = EMB_DIM) -> dict:
    """mean + the top `dim` principal directions of the sample (float32 SVD)."""
    X = np.asarray(sample, np.float32)
    mean = X.mean(axis=0)
    Xc = X - mean
    dim = int(min(dim, Xc.shape[1], max(Xc.shape[0] - 1, 1)))
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return {"mean": mean.astype(np.float32), "components": vt[:dim].astype(np.float32)}


def project(emb: np.ndarray, pca: dict) -> np.ndarray:
    """Embeddings → int8 unit vectors × 127 in the PCA space."""
    X = (np.asarray(emb, np.float32) - pca["mean"]) @ pca["components"].T
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(n, 1e-6)
    return np.clip(np.rint(X * 127), -127, 127).astype(np.int8)


def load_pca(folder: Path = None) -> dict | None:
    p = out_dir(folder) / "pca.npz"
    try:
        with np.load(p) as z:
            return {"mean": np.asarray(z["mean"], np.float32), "components": np.asarray(z["components"], np.float32)}
    except (OSError, KeyError, ValueError):
        return None


def _sample_embeddings(vdirs: list, n: int = PCA_SAMPLE, seed: int = 0) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    per = max(1, n // max(len(vdirs), 1))
    parts = []
    for vdir in vdirs:
        emb = _video_embeddings(vdir)
        if emb is None:
            continue
        _, e = emb
        if e.shape[0] > per:
            e = e[rng.choice(e.shape[0], per, replace=False)]
        parts.append(e.astype(np.float32))
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Reading the inputs
# ---------------------------------------------------------------------------

def _video_embeddings(vdir: Path):
    try:
        with np.load(vdir / "embeddings.npz") as z:
            return np.asarray(z["frame_times"], np.float32), np.asarray(z["frame_embs"])
    except (OSError, KeyError, ValueError):
        return None


def _student_scores(video_id: str, folder: Path):
    try:
        with np.load(folder / f"{video_id}.npz") as z:
            people = np.asarray(z["people"], np.uint8) if "people" in z.files else None
            return np.asarray(z["times"], np.float32), np.asarray(z["scores"], np.uint8), people
    except (OSError, KeyError, ValueError):
        return None


def _labels_for(video_id: str, items: list) -> list:
    return [i for i in items if i.get("video_id") == video_id]


def _load_labels() -> list:
    try:
        import training_labels as TLB
        return TLB.load_labels()
    except Exception:          # noqa: BLE001 — no app config (a bare run)
        return []


def _is_hidden():
    try:
        import training_labels as TLB
        return TLB.is_hidden
    except Exception:          # noqa: BLE001
        return lambda t: False


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build_video(video_id: str, vdir: Path, sc_index: dict, sc_folder: Path, pca: dict, items: list,
                hidden=None) -> dict | None:
    """One video's arrays (not yet written), or None with no reason when an input is missing."""
    sc = _student_scores(video_id, sc_folder)
    if sc is None:
        return None
    times, scores, people = sc
    scores = scores.copy()
    tags = list(sc_index.get("tags") or [])
    if scores.shape[1] != len(tags):
        return None
    cutoffs = np.asarray(sc_index.get("cutoffs") or [50] * len(tags), np.float32)
    vis = _read_json(vdir / "vision.json", {}) or {}
    duration = float(vis.get("duration") or (times[-1] + STEP if times.size else 0.0))
    try:
        curve = np.load(vdir / "person_curve.npy")
    except (OSError, ValueError):
        curve = np.zeros(0, np.float32)
    move = half_second_mean(curve, times)
    change = change_rate(move)
    bpm, per, onset = rhythm_at(times, vis.get("scenes") or [], duration)
    labelled = apply_labels(scores, times, tags, _labels_for(video_id, items), hidden)
    shot = shot_value(scores, cutoffs, shot_columns(tags))
    emb_t = _video_embeddings(vdir)
    emb = np.zeros((len(times), pca["components"].shape[0]), np.int8)
    if emb_t is not None:
        et, e = emb_t
        pos = np.searchsorted(et, times)
        pos = np.clip(pos, 0, max(len(et) - 1, 0))
        # the embedding nearest each half second (the vision grid is the same 2/s, so usually exact)
        prev = np.clip(pos - 1, 0, max(len(et) - 1, 0))
        pick = np.where(np.abs(et[prev] - times) < np.abs(et[pos] - times), prev, pos)
        emb = project(e[pick], pca)
    if people is None:
        people = np.zeros(len(times), np.uint8)
    return {"times": times.astype(np.float32), "move": move.astype(np.float16),
            "change": change.astype(np.float16), "bpm": bpm.astype(np.float16),
            "periodicity": per.astype(np.float16), "onset": onset.astype(np.float16),
            "scores": scores, "labelled": labelled, "shot": shot.astype(np.float16),
            "people": people.astype(np.uint8), "emb": emb,
            "cuts": np.asarray([float(c) for c in vis.get("cuts") or []], np.float32),
            "_duration": duration, "_labels": int(len(_labels_for(video_id, items)))}


def _write_video(folder: Path, video_id: str, arrays: dict) -> None:
    tmp = folder / f"{video_id}.tmp.npz"
    np.savez_compressed(tmp, **{k: v for k, v in arrays.items() if not k.startswith("_")})
    tmp.replace(folder / f"{video_id}.npz")


def _move_scale(folder: Path, videos: list) -> dict:
    vals = []
    for vid in videos:
        try:
            with np.load(folder / f"{vid}.npz") as z:
                m = np.asarray(z["move"], np.float32)
                p = np.asarray(z["people"], np.uint8) if "people" in z.files else None
        except (OSError, KeyError, ValueError):
            continue
        m = m[np.isfinite(m) & ((p > 0) if p is not None and p.size == m.size else True)]
        if m.size:
            vals.append(m)
    if not vals:
        return {"lo": 0.0, "hi": 1.0}
    allv = np.concatenate(vals)
    lo, hi = np.percentile(allv, MOVE_PCTS)
    return {"lo": round(float(lo), 6), "hi": round(float(max(hi, lo + 1e-6)), 6)}


def status_path(folder: Path = None) -> Path:
    return out_dir(folder) / STATUS_FILE


def read_status(folder: Path = None) -> dict:
    return _read_json(status_path(folder), {}) or {}


def _say_status(folder: Path, **kw) -> None:
    st = read_status(folder)
    st.update(kw)
    _write_json(status_path(folder), st)


def build(only: list = None, redo: bool = False, refit: bool = False, root: Path = None,
          folder: Path = None, sc_folder: Path = None, progress=print) -> dict:
    """Build (or update) the timeline files. Returns {"built": [...], "skipped": {id: reason}, "pca": "fitted"|"kept"}."""
    vroot = vision_root(root)
    dest = out_dir(folder)
    dest.mkdir(parents=True, exist_ok=True)
    scf = scores_dir(sc_folder)
    sc_index = _read_json(scf / "index.json", {}) or {}
    result = {"built": [], "skipped": {}, "pca": "kept"}
    _say_status(dest, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), done=0, total=0, current="",
                error=None, finished=None)
    try:
        if vroot is None:
            raise RuntimeError("No vision data folder found — set STUDENT_WORK_DIR in config.py (the folder that holds "
                               "_student/videos, the Hetzner Volume's student folder on the server).")
        if not sc_index.get("tags"):
            raise RuntimeError(f"No student export in {scf} — run `student_tags.py export` first.")
        index = load_index(dest)
        vdirs = sorted(p for p in vroot.iterdir() if p.is_dir())
        if only:
            want = set(only)
            vdirs = [p for p in vdirs if p.name in want]
        pca = None if refit else load_pca(dest)
        if pca is None:
            progress(f"Fitting the PCA ({EMB_DIM} numbers) on a sample of the library…")
            sample = _sample_embeddings(sorted(p for p in vroot.iterdir() if p.is_dir()))
            if sample is None or sample.shape[0] < 2:
                raise RuntimeError("No embeddings found in the vision data — nothing to fit the PCA on.")
            pca = fit_pca(sample)
            np.savez(dest / "pca.npz", **pca)
            result["pca"] = "fitted"
            redo = True                      # everything must be in the new space
        items = _load_labels()
        hidden = _is_hidden()
        old_videos = dict(index.get("videos") or {})
        same_model = index.get("model") == sc_index.get("model") and index.get("version") == VERSION
        todo = []
        for vdir in vdirs:
            vid = vdir.name
            if not (vdir / "vision.json").exists():
                result["skipped"][vid] = "no vision data yet (local_vision / Modal)"
                continue
            if not (scf / f"{vid}.npz").exists():
                result["skipped"][vid] = "no student scores (export)"
                continue
            rec = old_videos.get(vid)
            if rec and not redo and same_model and (dest / f"{vid}.npz").exists() \
                    and rec.get("labels") == len(_labels_for(vid, items)):
                continue
            todo.append(vdir)
        _say_status(dest, total=len(todo))
        t0 = time.time()
        for k, vdir in enumerate(todo, 1):
            vid = vdir.name
            _say_status(dest, current=vid, done=k - 1)
            arrays = build_video(vid, vdir, sc_index, scf, pca, items, hidden)
            if arrays is None:
                result["skipped"][vid] = "scores don't match the export's tags"
                continue
            _write_video(dest, vid, arrays)
            old_videos[vid] = {"duration": round(arrays["_duration"], 2), "built": time.strftime("%Y-%m-%d %H:%M"),
                               "model": sc_index.get("model"), "labels": arrays["_labels"],
                               "points": int(len(arrays["times"]))}
            result["built"].append(vid)
            if k % 10 == 0 or k == len(todo):
                progress(f"   {k} of {len(todo)} videos ({time.time() - t0:.0f} s)")
        # videos whose inputs vanished are dropped from the index (their files are left alone)
        keep = {p.name for p in vdirs} if not only else set(old_videos)
        old_videos = {v: r for v, r in old_videos.items() if v in keep or (dest / f"{v}.npz").exists()}
        new_index = {"version": VERSION, "model": sc_index.get("model"), "tags": sc_index.get("tags"),
                     "cutoffs": sc_index.get("cutoffs"), "step": STEP, "emb_dim": int(pca["components"].shape[0]),
                     "built": time.strftime("%Y-%m-%d %H:%M"),
                     "move_scale": _move_scale(dest, sorted(old_videos)), "videos": old_videos}
        _write_json(dest / "index.json", new_index)
        _say_status(dest, state="done", done=len(todo), current="", finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                    built=len(result["built"]), skipped=len(result["skipped"]))
        progress(f"Footage timeline: {len(result['built'])} built, {len(old_videos)} in the library, "
                 f"{len(result['skipped'])} skipped → {dest}")
    except Exception as e:     # noqa: BLE001 — the status file carries the reason to the app
        _say_status(dest, state="error", error=str(e), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        raise
    return result


def start_build(redo: bool = False, refit: bool = False, log: Path = None) -> str:
    """Start a build in the background (low priority), for the app's button. Returns a message."""
    if read_status().get("state") == "running":
        return "A build is already running."
    dest = out_dir()
    dest.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(HERE / "footage_timeline.py"), "build"] + (["--redo"] if redo else []) + \
           (["--refit"] if refit else [])
    if os.name != "nt":
        args = ["nice", "-n", "10"] + args
    log = log or dest / "build.log"
    env = dict(os.environ)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "2")
    with open(log, "ab") as fh:
        subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE), env=env,
                         start_new_session=True)
    _say_status(dest, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), done=0, total=0,
                current="starting…", error=None, finished=None)
    return "Build started."


def status_report(root: Path = None, folder: Path = None, sc_folder: Path = None) -> dict:
    """What's built and what's missing, for the app and `status`."""
    vroot = vision_root(root)
    index = load_index(folder)
    scf = scores_dir(sc_folder)
    have = set((index.get("videos") or {}).keys())
    vision, scores = set(), set()
    if vroot is not None and vroot.is_dir():
        vision = {p.name for p in vroot.iterdir() if p.is_dir() and (p / "vision.json").exists()}
    if scf.is_dir():
        scores = {p.stem for p in scf.glob("*.npz") if not p.name.endswith(".tmp.npz")}
    return {"vision_root": str(vroot) if vroot else None, "built": sorted(have),
            "ready": sorted((vision & scores) - have), "no_scores": sorted(vision - scores),
            "no_vision": sorted(scores - vision), "model": index.get("model"),
            "scores_model": (_read_json(scf / "index.json", {}) or {}).get("model"), "index": index}


# ---------------------------------------------------------------------------
# Reading a timeline
# ---------------------------------------------------------------------------

class Timeline:
    """One video's timeline, arrays loaded on first use. `tags` / `cutoffs` come from the index."""

    def __init__(self, video_id: str, path: Path, index: dict):
        self.video_id = video_id
        self.path = Path(path)
        self.index = index or {}
        self.tags = list(self.index.get("tags") or [])
        self.cutoffs = np.asarray(self.index.get("cutoffs") or [50] * len(self.tags), np.float32)
        self._data = None

    def _load(self) -> dict:
        if self._data is None:
            with np.load(self.path) as z:
                self._data = {k: z[k] for k in z.files}
        return self._data

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        d = self._load()
        if name in d:
            return d[name]
        raise AttributeError(name)

    def __len__(self) -> int:
        return int(len(self._load()["times"]))

    def column(self, tag: str) -> int | None:
        low = str(tag).lower()
        for k, t in enumerate(self.tags):
            if str(t).lower() == low:
                return k
        return None

    def tag_scores(self, tag: str) -> np.ndarray | None:
        k = self.column(tag)
        return None if k is None else self.scores[:, k].astype(np.float32) / 100.0

    def visible(self, tag: str) -> np.ndarray | None:
        """True where the tag is visible (score ≥ its cut-off)."""
        k = self.column(tag)
        return None if k is None else self.scores[:, k] >= self.cutoffs[k]

    def scaled_move(self) -> np.ndarray:
        return scaled_move(self.move, self.index.get("move_scale"))

    def select(self, t0: float, t1: float) -> np.ndarray:
        t = self.times
        return (t >= t0) & (t < t1)

    def embeddings(self) -> np.ndarray:
        """Unit-length float32 embeddings (from the stored int8)."""
        e = self.emb.astype(np.float32) / 127.0
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-6)


def scaled_move(move, scale: dict | None) -> np.ndarray:
    """Movement 0–1 within the library (5th → 0, 95th percentile → 1, clipped; NaN stays NaN)."""
    m = np.asarray(move, np.float32)
    s = scale or {"lo": 0.0, "hi": 1.0}
    lo, hi = float(s.get("lo", 0.0)), float(s.get("hi", 1.0))
    return np.clip((m - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def has_timeline(video_id: str, folder: Path = None) -> bool:
    return (out_dir(folder) / f"{video_id}.npz").exists()


def load(video_id: str, folder: Path = None, index: dict = None) -> Timeline | None:
    p = out_dir(folder) / f"{video_id}.npz"
    if not p.exists():
        return None
    return Timeline(video_id, p, index if index is not None else load_index(folder))


def derived_scene_tags(tl: Timeline, scenes: list, min_share: float = 0.5, time_range=None) -> dict:
    """{scene_id: {"tags": [...], "share": {tag: share}}} — a moment tag becomes a derived scene tag when it's
    visible for ≥ min_share of the scene (the rule leaned to on 4 Nov). Shot-size, hair and unused tags are
    left out (they have their own levels)."""
    import trial_logic as TL
    from library_common import overlap_with_range
    keep = [k for k, t in enumerate(tl.tags) if not TL.not_in_f1(t)]
    vis = tl.scores[:, keep] >= tl.cutoffs[keep][None, :]
    out = {}
    for s in scenes or []:
        try:
            a, b = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if time_range is not None:
            a, b = overlap_with_range(a, b, time_range)
            if b <= a:
                continue
        sel = tl.select(a, b)
        n = int(sel.sum())
        if n == 0:
            continue
        share = vis[sel].mean(axis=0)
        tags = {tl.tags[keep[i]]: round(float(share[i]), 3) for i in range(len(keep)) if share[i] > 0}
        out[s.get("scene_id")] = {"tags": [t for t, v in sorted(tags.items(), key=lambda kv: -kv[1]) if v >= min_share],
                                  "share": tags}
    return out


def tag_time(tl: Timeline, mask: np.ndarray = None) -> list:
    """[(tag, seconds visible)] sorted by time, over the whole video or `mask` (shot / hair / unused tags left out)."""
    import trial_logic as TL
    vis = tl.scores >= tl.cutoffs[None, :]
    if mask is not None:
        vis = vis[mask]
    secs = vis.sum(axis=0) * STEP
    rows = [(t, float(secs[k])) for k, t in enumerate(tl.tags) if secs[k] > 0 and not TL.not_in_f1(t)]
    return sorted(rows, key=lambda r: -r[1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="🧭 The footage timeline per video (Stage 3 step 3).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--only", nargs="*", default=None)
    b.add_argument("--redo", action="store_true")
    b.add_argument("--refit", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        r = status_report()
        print(f"Vision data: {r['vision_root']}")
        print(f"Built: {len(r['built'])} (student {r['model']}; export is {r['scores_model']})")
        print(f"Ready to build: {len(r['ready'])}" + (" — " + ", ".join(r["ready"][:10]) if r["ready"] else ""))
        print(f"No student scores yet: {len(r['no_scores'])} · no vision data: {len(r['no_vision'])}")
        return 0
    try:
        build(only=a.only, redo=a.redo, refit=a.refit)
    except Exception as e:     # noqa: BLE001
        print(f"Build failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
