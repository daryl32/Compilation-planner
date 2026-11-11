r"""
local_vision.py — Stage 6 (claude/plan-nov2-2026.md), first piece: the Modal vision stage run on
THIS laptop's GPU (RTX 4070), for new videos. No Modal credit, no teacher.

For each video it makes exactly what the Modal vision stage makes, with the same models, the same
decode and the same maths (the helpers are imported from modal_trial.py / trial_logic.py):
  • scene cuts with TransNetV2
  • body pose of every person (YOLO11m-pose, 15 fps, batched) → tracked on the CPU with
    ultralytics' default tracker → the camera-free person-movement curve, and per scene its rhythm
    (movement BPM, regularity, onsets) and people on screen
  • SigLIP2 picture embeddings, 2 per second
…written to _student/videos/<video_id>/ (in config.py's WORK_DIR) (embeddings.npz, person_curve.npy, detections.npz,
vision.json) — the same folder `student_tags.py fetch` puts the Modal videos in, so training,
export and 🎓 picks see laptop videos like any other (they just have no teacher answers). Then the
student scores every half second → catalogue/student_scores on Drive.

GPU use: one decode feeds pose (32 frames at a time) and embeddings (32 at a time); the tiny
TransNetV2 frames are decoded at the same time on another core; the tracking step for a video runs
on the CPU in the background while the GPU starts the next video.

Before: the video needs its catalogue and preview copy — run the usual local pipeline first.

Usage (from the activated .venv in the scene-labeling folder, or local_vision.bat):
    python local_vision.py list                 videos with a preview copy but no vision data yet
                                                (newest catalogue first)
    python local_vision.py run --newest 5       the 5 newest of those
    python local_vision.py run --only vidA vidB just these
    python local_vision.py run --newest 5 --model v5b    …and score them with student v5b
    python local_vision.py check                re-run one Modal-processed video here and compare
                                                (cuts, movement curve, embeddings) — parity + speed
Options: --redo (run again even if done), --no-scores (skip the student export).
Each video's timings go to _student/local_vision_log.json; `run` ends with minutes per 20-min video.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_logic as TL  # noqa: E402



def work_root() -> Path:
    """config.py's WORK_DIR (scene-labeling\\data), else next to this script — as student_tags.py."""
    try:
        import config
        return Path(getattr(config, "WORK_DIR", HERE))
    except Exception:      # noqa: BLE001
        return HERE


WORK = work_root() / "_student"
VIDEOS = WORK / "videos"
MODELS = work_root() / "_vision_models"   # pose weights (downloaded once by ultralytics)
LOG = WORK / "local_vision_log.json"
EMB_MODEL = "google/siglip2-so400m-patch14-384"     # same as modal_trial.EMB_MODEL
POSE_MODEL = "yolo11m-pose.pt"                       # same as modal_trial.POSE_MODEL
POSE_BATCH = 32
EMB_BATCH = 32
TARGET_MIN = 20.0                         # the plan's yardstick: minutes per 20-min video


# ---------------------------------------------------------------------------
# Which videos
# ---------------------------------------------------------------------------

def has_vision(video_id: str, root: Path = None) -> bool:
    d = (root or VIDEOS) / video_id
    return (d / "vision.json").exists() and (d / "embeddings.npz").exists()


def candidates(cats: dict, has_proxy, root: Path = None, mtimes: dict = None) -> list:
    """Videos with a catalogue and a preview copy but no vision data in _student yet (from Modal or
    from here), newest catalogue first."""
    mtimes = mtimes or {}
    ids = [v for v in cats if has_proxy(v) and not has_vision(v, root)]
    return sorted(ids, key=lambda v: (-mtimes.get(v, 0.0), v))


def catalogue_mtimes() -> dict:
    """{video_id: when it was added}: the server's .added_dates.json where it knows the video, else
    the catalogue file's creation time on Windows (its modified time changes with every Reviewer edit)."""
    from datetime import datetime
    from pack_for_modal import CATALOGUE_DIR, NOT_CATALOGUES
    try:
        known = json.loads((CATALOGUE_DIR / ".added_dates.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    out = {}
    for p in CATALOGUE_DIR.glob("*.json"):
        if p.name in NOT_CATALOGUES or p.name.startswith("."):
            continue
        try:
            out[p.stem] = datetime.fromisoformat(str(known[p.stem]).replace("Z", "+00:00")).timestamp()
            continue
        except (KeyError, ValueError, TypeError):
            pass
        try:
            st = p.stat()
            out[p.stem] = st.st_ctime if os.name == "nt" else st.st_mtime
        except OSError:
            pass
    return out


def _library():
    import pack_for_modal as PK
    print("Reading catalogues from Drive…", flush=True)
    cats = PK.load_catalogues(progress=True)
    has_proxy = lambda v: (PK.PROXY_DIR / f"{v}.mp4").exists()
    mt = catalogue_mtimes()           # catalogue files are <video_id>.json
    return PK, cats, has_proxy, {v: mt.get(v, 0.0) for v in cats}


def manifest_for(PK, cat: dict, meta: dict) -> tuple:
    """(manifest, preview path) — the same manifest pack_for_modal makes for Modal."""
    v = cat["video_id"]
    proxy = PK.PROXY_DIR / f"{v}.mp4"
    info = PK.probe(proxy)
    if info.get("error"):
        raise RuntimeError(info["error"])
    m = PK.manifest(cat, info, meta.get(v), proxy)
    return m, proxy, PK.check(info, m["catalogue_duration"])


# ---------------------------------------------------------------------------
# GPU step: one decode → pose detections + embeddings; TransNetV2 frames decoded alongside
# ---------------------------------------------------------------------------

class Models:
    """Loaded once for the whole run (loading them per video would waste ~20 s each)."""

    def __init__(self):
        import torch
        import modal_trial as MT                 # its plain helpers only — nothing runs on Modal
        from transformers import AutoModel, AutoProcessor
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU found — this needs the laptop's RTX 4070 (GPU torch in the venv).")
        t0 = time.time()
        MODELS.mkdir(exist_ok=True)
        cwd = os.getcwd()
        os.chdir(MODELS)                         # ultralytics downloads the weights into the cwd
        try:
            self.pose = YOLO(POSE_MODEL)
        finally:
            os.chdir(cwd)
        self.emb = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.float16).cuda().eval()
        self.proc = AutoProcessor.from_pretrained(EMB_MODEL)
        self.shots = MT._transnet_model()
        self.gpu = torch.cuda.get_device_name(0)
        self.load_seconds = round(time.time() - t0, 1)


def _shot_frames(path: Path, box: dict) -> None:
    """The 48×27 RGB frames TransNetV2 reads (the video's own frame rate) — run in a thread while
    the main decode feeds the GPU, like vision_prep on Modal."""
    t0 = time.time()
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", "48x27", "pipe:"], capture_output=True, check=True).stdout
        box["frames"] = np.frombuffer(bytearray(raw), np.uint8).reshape([-1, 27, 48, 3])
    except Exception as e:      # noqa: BLE001 — keep going without cuts, as Modal does
        box["error"] = f"{e.__class__.__name__}: {e}"[:300]
    box["seconds"] = time.time() - t0


def gpu_step(models: Models, video_id: str, man: dict, path: Path, out_dir: Path) -> dict:
    import cv2
    import torch
    import modal_trial as MT
    started = time.time()
    timings = {}
    duration = float(man["proxy"]["duration"])
    native_fps = float(man["proxy"].get("fps") or 25.0)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = {}
    shot_thread = threading.Thread(target=_shot_frames, args=(path, shots), daemon=True)
    shot_thread.start()

    cmd, w, h = MT._pose_cmd(str(path), man)
    det_frame, det_box, det_kxy, det_kconf = [], [], [], []
    emb_times, emb_vecs, ebatch, ebatch_t = [], [], [], []
    pbatch, pbatch_i = [], []
    every = int(round(TL.POSE_FPS / 2))
    t_pose = t_emb = 0.0

    def flush_emb():
        nonlocal t_emb
        if not ebatch:
            return
        t1 = time.time()
        with torch.no_grad():
            inp = models.proc(images=[cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in ebatch],
                              return_tensors="pt").to("cuda")
            f = models.emb.get_image_features(pixel_values=inp["pixel_values"].half())
            if not torch.is_tensor(f):
                f = getattr(f, "pooler_output", None) if getattr(f, "pooler_output", None) is not None else f[0]
            f = torch.nn.functional.normalize(f.float(), dim=-1)
        emb_vecs.append(f.cpu().numpy().astype(np.float16))
        emb_times.extend(ebatch_t)
        ebatch.clear()
        ebatch_t.clear()
        t_emb += time.time() - t1

    def flush_pose():
        nonlocal t_pose
        if not pbatch:
            return
        t1 = time.time()
        results = models.pose.predict(pbatch, verbose=False, imgsz=640, conf=0.3)       # FP32, as on Modal
        for i, res in zip(pbatch_i, results):
            b = res.boxes
            if b is None or not len(b):
                continue
            data = b.data.cpu().numpy().astype(np.float32)[:, :6]
            xy = res.keypoints.xy.cpu().numpy().astype(np.float32)
            kc = (res.keypoints.conf.cpu().numpy().astype(np.float32) if res.keypoints.conf is not None
                  else np.ones(xy.shape[:2], np.float32))
            det_frame.append(np.full(len(data), i, np.int32))
            det_box.append(data)
            det_kxy.append(xy)
            det_kconf.append(kc)
        pbatch.clear()
        pbatch_i.clear()
        t_pose += time.time() - t1

    idx = 0
    for frame in MT._frames(cmd, w, h):
        pbatch.append(frame)
        pbatch_i.append(idx)
        if len(pbatch) >= POSE_BATCH:
            flush_pose()
        if idx % every == 0:
            ebatch.append(frame.copy())
            ebatch_t.append(round(idx / TL.POSE_FPS, 2))
            if len(ebatch) >= EMB_BATCH:
                flush_emb()
        idx += 1
        if idx % int(TL.POSE_FPS * 120) == 0:
            done_s, el = idx / TL.POSE_FPS, time.time() - started
            print(f"   {done_s / 60:.0f} of {duration / 60:.0f} min · about "
                  f"{el / done_s * max(duration - done_s, 0) / 60:.1f} min left", flush=True)
    flush_pose()
    flush_emb()
    n_frames = idx
    timings["decode_pose_embed"] = time.time() - started
    timings["pose_gpu"], timings["embed_gpu"] = t_pose, t_emb

    # scene cuts from the frames the other thread decoded
    t0 = time.time()
    shot_thread.join()
    cuts, shot_error = [], shots.get("error")
    if "frames" in shots:
        try:
            with torch.no_grad():
                preds, _ = models.shots.predict_frames(torch.from_numpy(shots["frames"]).to(models.shots.device),
                                                       quiet=True)
            p = preds.cpu().numpy() if hasattr(preds, "cpu") else preds
            cuts = TL.cuts_from_predictions(p, native_fps)
        except Exception as e:      # noqa: BLE001
            shot_error = f"{e.__class__.__name__}: {e}"[:300]
    timings["shots_gpu"] = time.time() - t0
    timings["shots_decode"] = shots.get("seconds", 0.0)

    def cat(parts, shape, dtype):
        return np.concatenate(parts) if parts else np.zeros(shape, dtype)
    np.savez_compressed(out_dir / "detections.npz", frame=cat(det_frame, (0,), np.int32),
                        box=cat(det_box, (0, 6), np.float32), kxy=cat(det_kxy, (0, 17, 2), np.float32),
                        kconf=cat(det_kconf, (0, 17), np.float32), n_frames=np.int32(n_frames))
    np.savez_compressed(out_dir / "frame_embs.npz", frame_times=np.array(emb_times),
                        frame_embs=np.vstack(emb_vecs) if emb_vecs else np.zeros((0, 1), np.float16))
    timings["total"] = time.time() - started
    return {"video_id": video_id, "duration": round(duration, 2), "partial": False, "native_fps": native_fps,
            "pose_fps": TL.POSE_FPS, "cuts": cuts, "shot_error": shot_error, "frames": n_frames,
            "targets": [], "pipeline": 2, "pose_batch": POSE_BATCH, "source": "laptop",
            "timings": {k: round(v, 1) for k, v in timings.items()}, "gpu": models.gpu,
            "models": {"pose": POSE_MODEL, "embed": EMB_MODEL, "shots": "TransNetV2"}}


# ---------------------------------------------------------------------------
# CPU step (a background process): tracking → movement curve, rhythm, scene numbers
# ---------------------------------------------------------------------------

def motion_step(vg: dict, man: dict, path: str, out_dir: str) -> dict:
    """What vision_motion does on Modal. Writes embeddings.npz, person_curve.npy and vision.json
    (the "done" marker) into out_dir. Top-level so a worker process can run it."""
    import modal_trial as MT
    out = Path(out_dir)
    t0 = time.time()
    # `with` closes each .npz at once — on Windows an open npz can't be deleted (WinError 32).
    with np.load(out / "detections.npz") as z:
        det = {k: z[k] for k in ("frame", "box", "kxy", "kconf")}
        n_frames = int(z["n_frames"])
    cmd, w, h = MT._pose_cmd(path, man)
    tracks, people, tracker_name = MT._track_offline(MT._frames(cmd, w, h), det, n_frames)
    t_track = time.time() - t0
    t1 = time.time()
    curve = TL.movement_curve(tracks, n_frames)
    with np.load(out / "frame_embs.npz") as fe:
        et, emb = fe["frame_times"], fe["frame_embs"]
    scenes, scene_ids, scene_embs = TL.scene_features(man["scenes"], vg["duration"], curve, people, et, emb)
    np.savez_compressed(out / "embeddings.npz", frame_times=et, frame_embs=emb,
                        scene_ids=np.array(scene_ids), scene_embs=np.array(scene_embs))
    np.save(out / "person_curve.npy", curve.astype(np.float32))
    cpu = {"track": round(t_track, 1), "maths": round(time.time() - t1, 1),
           "prep": round(vg["timings"].get("shots_decode", 0.0), 1)}
    cpu["total"] = round(cpu["track"] + cpu["maths"], 1)
    result = {**vg, "tracks": len(tracks), "tracker": tracker_name, "scenes": scenes, "cpu_timings": cpu,
              "made": time.strftime("%Y-%m-%d %H:%M")}
    tmp = out / "vision.json.part"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(out / "vision.json")
    try:
        (out / "frame_embs.npz").unlink(missing_ok=True)    # only a working copy — embeddings.npz has it all
    except OSError:
        pass
    return {"video_id": vg["video_id"], "cpu": cpu, "tracks": len(tracks)}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def per_target(video_s: float, wall_s: float, target_min: float = TARGET_MIN) -> float | None:
    """Laptop minutes per `target_min` minutes of video."""
    return round(wall_s / video_s * target_min, 1) if video_s > 0 else None


def _log(rows: list) -> None:
    old = []
    try:
        old = json.loads(LOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps((old if isinstance(old, list) else []) + rows, indent=1), encoding="utf-8")


def process(ids: list, PK, cats: dict, root: Path = None, redo: bool = False, label: str = "run") -> list:
    """GPU step per video in order; each video's CPU step runs in a worker process while the GPU
    moves on. Returns one timing row per finished video."""
    from concurrent.futures import ProcessPoolExecutor
    root = root or VIDEOS
    meta = PK.library_meta()
    todo = [v for v in ids if redo or not has_vision(v, root)]
    if not todo:
        print("Nothing to do — every video asked for already has vision data (--redo to run again).")
        return []
    print("Loading the models (pose, SigLIP2, TransNetV2)…", flush=True)
    models = Models()
    print(f"GPU: {models.gpu} · models loaded in {models.load_seconds:.0f} s\n", flush=True)
    rows, pending, t_run = [], [], time.time()
    with ProcessPoolExecutor(max_workers=1) as pool:
        for k, v in enumerate(todo, 1):
            try:
                man, path, problems = manifest_for(PK, cats[v], meta)
            except Exception as e:      # noqa: BLE001
                print(f"[{k}/{len(todo)}] {v}: skipped — {e}")
                continue
            print(f"[{k}/{len(todo)}] {v}  ({man['proxy']['duration'] / 60:.1f} min, {len(man['scenes'])} scenes)"
                  + (f"  ⚠ {'; '.join(problems)}" if problems else ""), flush=True)
            out_dir = root / v
            if redo:
                (out_dir / "vision.json").unlink(missing_ok=True)
            t0 = time.time()
            vg = gpu_step(models, v, man, path, out_dir)
            print(f"   GPU step {vg['timings']['total']:.0f} s (pose {vg['timings']['pose_gpu']:.0f} s, "
                  f"embeddings {vg['timings']['embed_gpu']:.0f} s), {len(vg['cuts'])} cuts"
                  + (f" — cuts failed: {vg['shot_error']}" if vg["shot_error"] else ""), flush=True)
            pending.append((v, t0, vg, pool.submit(motion_step, vg, man, str(path), str(out_dir))))
            while len(pending) > 1:          # at most one CPU step waiting behind the GPU
                rows.append(_finish(pending.pop(0), label))
        while pending:
            rows.append(_finish(pending.pop(0), label))
    if rows:
        _log(rows)
        total = time.time() - t_run         # the steps overlap, so the whole run is the honest time
        mins = per_target(sum(r["video_s"] for r in rows), total)
        print(f"\n{len(rows)} video(s), {sum(r['video_s'] for r in rows) / 60:.0f} min of footage in "
              f"{total / 60:.1f} min (+{models.load_seconds:.0f} s loading models).")
        if mins is not None:
            verdict = "within" if mins <= 5 else "over"
            print(f"≈ {mins} min per {TARGET_MIN:.0f}-min video — {verdict} the 5-min goal "
                  f"(vision only; tagging adds a few seconds).")
    return rows


def _finish(item, label: str) -> dict:
    v, t0, vg, fut = item
    res = fut.result()
    wall = time.time() - t0
    print(f"   {v}: tracking + rhythm {res['cpu']['total']:.0f} s on the CPU, {res['tracks']} people tracked "
          f"→ done ({wall / 60:.1f} min for {vg['duration'] / 60:.1f} min of video)", flush=True)
    return {"video_id": v, "label": label, "at": time.strftime("%Y-%m-%d %H:%M"), "video_s": vg["duration"],
            "wall_s": round(wall, 1), "gpu_s": vg["timings"]["total"], "pose_s": vg["timings"]["pose_gpu"],
            "embed_s": vg["timings"]["embed_gpu"], "shots_s": vg["timings"]["shots_gpu"],
            "cpu_s": res["cpu"]["total"], "cuts": len(vg["cuts"]), "tracks": res["tracks"], "gpu": vg["gpu"]}


def scores_model(want: str = "") -> str | None:
    """The student to score with: --model, else the one the current student_scores export used."""
    if want:
        return want
    try:
        from config import CATALOGUE_DIR
        idx = json.loads((Path(CATALOGUE_DIR) / "student_scores" / "index.json").read_text(encoding="utf-8"))
        return idx.get("model")
    except Exception:      # noqa: BLE001
        return None


def run(ids: list, PK, cats: dict, redo: bool, model: str, scores: bool) -> int:
    rows = process(ids, PK, cats, redo=redo)
    if rows:                      # a tuned SigLIP2 encoder in use (siglip_tune.py): its embeddings for the new videos
        try:
            import siglip_tune as FT
            for enc in FT.encoders_in_use():
                print(f"\nThe tuned encoder '{enc}' is in use — its embeddings for the new videos…", flush=True)
                FT.embed(enc, send=False)        # only the videos without them; upload-local sends them
        except ImportError:
            pass                  # no siglip_tune.py here: nothing tuned to keep up with
    if not rows or not scores:
        return 0
    name = scores_model(model)
    if not name:
        print("\nNo student_scores export yet — run with --model v5b to score these videos.")
        return 0
    print(f"\nScoring every half second with student {name} (all videos, so the export stays one model)…")
    import student_tags as S
    return S.export(name)


# ---------------------------------------------------------------------------
# check — the same video here and on Modal
# ---------------------------------------------------------------------------

def compare(modal_dir: Path, local_dir: Path) -> dict:
    """Modal vs laptop for one video: cuts matched within 0.25 s, movement-curve correlation,
    mean cosine of the embeddings at the same times, people tracked."""
    out = {}
    mv = json.loads((modal_dir / "vision.json").read_text(encoding="utf-8"))
    lv = json.loads((local_dir / "vision.json").read_text(encoding="utf-8"))
    a, b = np.asarray(mv.get("cuts") or [], float), np.asarray(lv.get("cuts") or [], float)
    matched = int(sum(1 for c in a if b.size and np.min(np.abs(b - c)) <= 0.25))
    out["cuts"] = {"modal": int(a.size), "laptop": int(b.size), "matched": matched}
    ca, cb = np.load(modal_dir / "person_curve.npy"), np.load(local_dir / "person_curve.npy")
    n = min(len(ca), len(cb))
    ok = np.isfinite(ca[:n]) & np.isfinite(cb[:n])
    out["curve_corr"] = round(float(np.corrcoef(ca[:n][ok], cb[:n][ok])[0, 1]), 4) if ok.sum() > 10 else None
    out["curve_seen"] = {"modal": int(np.isfinite(ca).sum()), "laptop": int(np.isfinite(cb).sum())}
    za, zb = np.load(modal_dir / "embeddings.npz"), np.load(local_dir / "embeddings.npz")
    ta, tb = {round(float(t), 2): i for i, t in enumerate(za["frame_times"])}, za["frame_embs"]
    pairs = [(ta[round(float(t), 2)], j) for j, t in enumerate(zb["frame_times"]) if round(float(t), 2) in ta]
    if pairs:
        ia, ib = map(list, zip(*pairs))
        ea, eb = tb[ia].astype(np.float32), zb["frame_embs"][ib].astype(np.float32)
        cos = (ea * eb).sum(1) / np.maximum(np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1), 1e-6)
        out["emb_cosine"] = {"mean": round(float(cos.mean()), 4), "min": round(float(cos.min()), 4), "n": len(pairs)}
    out["tracks"] = {"modal": mv.get("tracks"), "laptop": lv.get("tracks")}
    return out


def check(video_id: str, PK, cats: dict) -> int:
    if not video_id:
        done = [p.name for p in sorted(VIDEOS.iterdir()) if has_vision(p.name)
                and json.loads((p / "vision.json").read_text(encoding="utf-8")).get("source") != "laptop"
                and p.name in cats and (PK.PROXY_DIR / f"{p.name}.mp4").exists()] if VIDEOS.exists() else []
        if not done:
            print("No Modal-processed video with a local preview copy — run `student_tags.py fetch` first.")
            return 1
        # a mid-length one: representative, and quick
        lens = sorted(done, key=lambda v: PK.catalogue_duration(cats[v]))
        video_id = lens[len(lens) // 2]
    check_root = WORK / "vision_check"
    shutil.rmtree(check_root / video_id, ignore_errors=True)
    print(f"Check: {video_id} — run here, then compared with its Modal results.\n")
    rows = process([video_id], PK, cats, root=check_root, redo=True, label="check")
    if not rows:
        return 1
    res = compare(VIDEOS / video_id, check_root / video_id)
    c = res["cuts"]
    print(f"\nCuts: Modal {c['modal']}, laptop {c['laptop']}, {c['matched']} the same (±0.25 s)")
    print(f"Movement curve: correlation {res['curve_corr']} (Modal's own vision-check was 0.9991)")
    if "emb_cosine" in res:
        e = res["emb_cosine"]
        print(f"Embeddings: mean cosine {e['mean']} (lowest {e['min']}) over {e['n']} frames — 1.0 = identical")
    print(f"People tracked: Modal {res['tracks']['modal']}, laptop {res['tracks']['laptop']}")
    (check_root / video_id / "compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The Modal vision stage on the laptop GPU, for new videos.")
    ap.add_argument("command", choices=["list", "run", "check"])
    ap.add_argument("--only", nargs="+", metavar="VIDEO_ID", help="run: just these videos")
    ap.add_argument("--newest", type=int, default=0, metavar="N", help="run: the N newest videos without vision data")
    ap.add_argument("--video", default="", help="check: this video (default: a mid-length Modal one)")
    ap.add_argument("--redo", action="store_true", help="run again even if a video is done")
    ap.add_argument("--model", default="", help="student to score with (default: the current export's)")
    ap.add_argument("--no-scores", action="store_true", help="skip the student scores")
    a = ap.parse_args(argv)
    if not shutil.which("ffmpeg"):
        print("ffmpeg isn't on PATH — install it (winget install ffmpeg) and open a new terminal.")
        return 1
    WORK.mkdir(exist_ok=True)
    PK, cats, has_proxy, mtimes = _library()
    if a.command == "check":
        return check(a.video, PK, cats)
    todo = candidates(cats, has_proxy, mtimes=mtimes)
    if a.command == "list" or not (a.only or a.newest):
        print(f"\n{len(todo)} video(s) with a preview copy and no vision data yet (newest first):")
        for v in todo[:40]:
            when = time.strftime("%d %b %H:%M", time.localtime(mtimes.get(v))) if mtimes.get(v) else "?"
            print(f"  {v}   {PK.catalogue_duration(cats[v]) / 60:5.1f} min   catalogue {when}")
        if len(todo) > 40:
            print(f"  … and {len(todo) - 40} more")
        if a.command == "run":
            print("\nSay which: run --newest N, or run --only VIDEO_ID …")
        return 0
    if a.only:
        missing = [v for v in a.only if v not in cats or not has_proxy(v)]
        if missing:
            print("No catalogue or preview copy for: " + ", ".join(missing) + " — run the local pipeline first.")
        ids = [v for v in a.only if v not in missing]
    else:
        ids = todo[:a.newest]
    return run(ids, PK, cats, a.redo, a.model, not a.no_scores)


if __name__ == "__main__":
    sys.exit(main())
