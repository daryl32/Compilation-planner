r"""
modal_trial.py — the Modal GPU trial (Phase 2 + 3 of claude/ml-redesign-plan.md).

Reads the pack that pack_for_modal.py uploaded (Modal Volume "media-train", /pack) and, for
each video:
  vision (L4 GPU)   • scene changes with TransNetV2 (a neural shot detector)
                    • body pose of every person (YOLO11-pose, 15 fps, tracked) → a person-motion
                      curve with the camera taken out, and per scene its rhythm: movement BPM,
                      how regular it is, and the onsets
                    • SigLIP2 picture embeddings, 2 per second (for the vector search later)
                    • the frames the tagger needs
  tag (H100 GPU)    • Qwen3-VL-30B-A3B (the "teacher") tags every scene and every 🎯 tag card
                      from 7 frames over 4 s, using your tag_vocabulary.csv
  report (CPU)      • teacher vs the old tagger vs your 🎯 Training Labels, per tag
                    • new vs old scene detection vs your scene-change answers
                    • what it cost, per hour of footage, and the full-library estimate
Results go to /trial on the same Volume; the report is also saved next to this file
(trial_report.md).

Run from the scene-labeling folder, in the venv (pip install modal; modal setup):
    modal run modal_trial.py --smoke          1 video, first 90 s — checks it all works (cents)
    modal run modal_trial.py                  the whole trial pack
    modal run modal_trial.py --stage report   just rebuild the report
    modal run modal_trial.py --stage tag --videos vid05,vid06
Stages: download, vision, tag, report, export, status, all (default). status lists every packed
video with its batch and when vision / tagging were done — nothing runs on a GPU. export puts the results in your
Drive catalogue folder (catalogue/trial_results) for the web app's 🧪 Trial results view. Each skips a video it has already done
unless you pass --redo.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

# The teachers to compare. Both dense (non mixture-of-experts): the 30B-A3B MoE needed a
# FlashInfer MoE kernel that vLLM compiles from scratch on start-up (20+ silent minutes).
TEACHERS = {"8b": "Qwen/Qwen3-VL-8B-Instruct", "32b": "Qwen/Qwen3-VL-32B-Instruct"}
TEACHER_GPUS = {"8b": ["L40S", "A100-40GB", "A100-80GB", "H100"],   # 17 GB of weights
                "32b": ["H100", "A100-80GB"]}                        # 66 GB of weights
VLM_MODEL = TEACHERS["8b"]
TEACHERS_DEFAULT = "32b"     # chosen after batch 1
EMB_MODEL = "google/siglip2-so400m-patch14-384"
# Medium pose model: about twice as fast as the large one, and closer to what the laptop's
# RTX 4070 can run in its 5-minute-per-video budget.
POSE_MODEL = "yolo11m-pose.pt"
# First choice first; Modal falls back down the list when that GPU type is busy.
VISION_GPUS = ["L4", "A10", "L40S", "A100-40GB"]
VISION_GPU = VISION_GPUS[0]

DATA, CACHE = "/data", "/cache"
PACK, OUT = f"{DATA}/pack", f"{DATA}/trial"

app = modal.App("media-trial")
pack_vol = modal.Volume.from_name("media-train", create_if_missing=True)
cache_vol = modal.Volume.from_name("media-models", create_if_missing=True)

base = (modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
        .env({"HF_HOME": f"{CACHE}/hf", "YOLO_CONFIG_DIR": f"{CACHE}/yolo"}))
vision_image = (base.pip_install("torch", "torchvision", "transformers>=4.57", "ultralytics", "lap",
                                 "transnetv2-pytorch", "opencv-python-headless", "numpy", "pillow",
                                 "huggingface_hub")
                .add_local_python_source("trial_logic"))
# vLLM compiles some GPU kernels when it starts, so it needs the CUDA compiler: the same base
# image and vLLM version as Modal's own vLLM example (a tested pair).
tag_image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
             .entrypoint([])
             .uv_pip_install("vllm==0.21.0", "numpy", "pillow", "huggingface_hub")
             .env({"HF_HOME": f"{CACHE}/hf", "HF_XET_HIGH_PERFORMANCE": "1",
                   # compiled code kept on the cache Volume, so only the first start pays for it
                   "VLLM_CACHE_ROOT": f"{CACHE}/vllm", "FLASHINFER_WORKSPACE_BASE": f"{CACHE}/flashinfer",
                   # plain PyTorch sampling — no FlashInfer kernel to compile
                   "VLLM_USE_FLASHINFER_SAMPLER": "0"})
             .add_local_python_source("trial_logic"))
report_image = base.pip_install("numpy").add_local_python_source("trial_logic")


# ---------------------------------------------------------------------------
# Shared helpers (run inside the containers)
# ---------------------------------------------------------------------------

def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _gold_items() -> list:
    return [i for i in (_read_json(f"{PACK}/library/training_labels.json", {}) or {}).get("items") or []
            if isinstance(i, dict)]


def _vocab() -> list:
    import trial_logic as TL
    p = Path(f"{PACK}/library/tag_vocabulary.csv")
    if p.exists():
        rows = TL.parse_vocabulary(p.read_text(encoding="utf-8-sig"))
        if rows:
            return rows
    return TL.vocabulary_from_labels(_read_json(f"{PACK}/library/labels.json", []))


def _vision_done(video_id: str) -> bool:
    """A complete vision result made with the current pose model (a smoke test's is partial)."""
    prev = _read_json(f"{OUT}/{video_id}/vision.json")
    return bool(prev) and not prev.get("partial") and (prev.get("models") or {}).get("pose") == POSE_MODEL


def _tags_done(video_id: str, key: str) -> bool:
    """A complete tagging by this teacher of every moment the current vision result has."""
    vis = _read_json(f"{OUT}/{video_id}/vision.json")
    prev = _read_json(f"{OUT}/{video_id}/tags_{key}.json")
    return (bool(vis) and bool(prev) and not prev.get("partial") and not vis.get("partial")
            and prev.get("targets") == len(vis.get("targets") or []))


def _gpu_name(default: str) -> str:
    """The GPU this container actually got (it may be a fallback), in trial_logic.GPU_RATE's names."""
    try:
        import subprocess
        name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.upper()
    except Exception:
        return default
    for key, label in (("H100", "H100"), ("L40S", "L40S"), ("A10G", "A10"), ("A10", "A10"), ("L4", "L4"),
                       ("T4", "T4")):
        if key in name:
            return label
    if "A100" in name:
        return "A100-80GB" if "81" in name or "80" in name else "A100-40GB"
    return default


def _video_path(video_id: str, max_seconds) -> str:
    """The preview copy — or, for a smoke run, its first max_seconds as a temporary file."""
    import subprocess
    src = f"{PACK}/videos/{video_id}.mp4"
    if not max_seconds:
        return src
    dst = f"/tmp/{video_id}_first{int(max_seconds)}.mp4"
    if not os.path.exists(dst):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-t", str(max_seconds), "-an",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", dst], check=True)
    return dst


# ---------------------------------------------------------------------------
# download — model weights into the "media-models" Volume, once (CPU only, so no GPU time is
# spent downloading)
# ---------------------------------------------------------------------------

@app.function(image=vision_image, volumes={CACHE: cache_vol}, timeout=3 * 3600, cpu=4, memory=8192)
def download_models(vlms: list = None) -> str:
    from huggingface_hub import snapshot_download
    done = []
    for repo in [EMB_MODEL, *(vlms or [VLM_MODEL]), "Sn4kehead/TransNetV2"]:
        try:
            snapshot_download(repo)
            done.append(repo)
        except Exception as e:   # TransNetV2 weights may also ship inside the pip package
            done.append(f"{repo} (skipped: {e.__class__.__name__})")
    os.makedirs(f"{CACHE}/yolo", exist_ok=True)
    os.chdir(f"{CACHE}/yolo")
    from ultralytics import YOLO
    YOLO(POSE_MODEL)
    done.append(POSE_MODEL)
    cache_vol.commit()
    return "Downloaded: " + ", ".join(done)


# ---------------------------------------------------------------------------
# vision — scene changes, pose → movement and rhythm, embeddings, frames to tag
# ---------------------------------------------------------------------------

def _transnet_cuts(path: str, threshold: float = 0.5) -> tuple:
    """(cut times in seconds, fps) from TransNetV2's per-frame transition probabilities."""
    import glob
    import numpy as np
    import torch
    import transnetv2_pytorch
    from transnetv2_pytorch import TransNetV2
    model = TransNetV2(device="cuda")
    weights = glob.glob(os.path.join(os.path.dirname(transnetv2_pytorch.__file__), "**", "*.pth"), recursive=True)
    if not weights:
        from huggingface_hub import hf_hub_download
        weights = [hf_hub_download("Sn4kehead/TransNetV2", "transnetv2-pytorch-weights.pth")]
    model.load_state_dict(torch.load(weights[0], map_location=model.device))
    model.eval()
    with torch.no_grad():
        try:
            res = model.analyze_video(path)
            fps, preds = float(res["fps"]), res["single_frame_predictions"]
        except (AttributeError, KeyError, TypeError):
            _, preds, _ = model.predict_video(path)
            import cv2
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
    p = np.asarray(preds.cpu().numpy() if hasattr(preds, "cpu") else preds, dtype=float).reshape(-1)
    above = p > threshold
    cuts, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            cuts.append(round(((i + j) / 2 + 0.5) / fps, 2))
            i = j + 1
        else:
            i += 1
    return [c for c in cuts if c > 0.2], fps


@app.function(image=vision_image, gpu=VISION_GPUS, volumes={DATA: pack_vol, CACHE: cache_vol},
              timeout=4 * 3600, cpu=4, memory=16384)
def vision(video_id: str, max_seconds: float = 0, redo: bool = False) -> dict:
    import subprocess
    import cv2
    import numpy as np
    import torch
    import trial_logic as TL

    out_dir = Path(f"{OUT}/{video_id}")
    if not redo and _vision_done(video_id):
        return {"video_id": video_id, "skipped": True}
    started = time.time()
    timings = {}
    man = _read_json(f"{PACK}/manifests/{video_id}.json")
    duration = float(man["proxy"]["duration"])
    if max_seconds:
        duration = min(duration, float(max_seconds))
    path = _video_path(video_id, max_seconds)

    # 1. Scene changes
    t0 = time.time()
    shot_error = None
    try:
        cuts, native_fps = _transnet_cuts(path)
    except Exception as e:      # keep going — pose, embeddings and frames still matter
        cuts, native_fps, shot_error = [], None, f"{e.__class__.__name__}: {e}"[:300]
    timings["shots"] = time.time() - t0

    # 2. One decode at 15 fps: pose (tracked), embeddings at 2 fps, frames for the tagger
    t0 = time.time()
    targets = TL.tag_targets(man, _gold_items(), max_seconds or None)
    need = {}
    for t in TL.frame_times(targets, duration):
        need.setdefault(int(round(t * TL.POSE_FPS)), []).append(t)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(f"{CACHE}/yolo")
    from ultralytics import YOLO
    pose = YOLO(POSE_MODEL)
    from transformers import AutoModel, AutoProcessor
    emb_model = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.float16).cuda().eval()
    emb_proc = AutoProcessor.from_pretrained(EMB_MODEL)
    timings["load_models"] = time.time() - t0

    src_w, src_h = int(man["proxy"].get("width") or 854), int(man["proxy"].get("height") or 480)
    w = 640
    h = int(round(w * src_h / src_w / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={TL.POSE_FPS},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 8)
    tracks, people, emb_times, emb_vecs, batch, batch_t = {}, [], [], [], [], []
    every = int(round(TL.POSE_FPS / 2))
    t_pose = t_emb = 0.0

    def flush_emb():
        nonlocal t_emb
        if not batch:
            return
        t1 = time.time()
        with torch.no_grad():
            inp = emb_proc(images=[cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in batch], return_tensors="pt").to("cuda")
            f = emb_model.get_image_features(pixel_values=inp["pixel_values"].half())
            if not torch.is_tensor(f):      # newer transformers return an output object
                f = getattr(f, "pooler_output", None) if getattr(f, "pooler_output", None) is not None else f[0]
            f = torch.nn.functional.normalize(f.float(), dim=-1)
        emb_vecs.append(f.cpu().numpy().astype(np.float16))
        emb_times.extend(batch_t)
        batch.clear()
        batch_t.clear()
        t_emb += time.time() - t1

    idx = 0
    while True:
        buf = proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        t1 = time.time()
        res = pose.track(frame, persist=True, verbose=False, imgsz=640, conf=0.3)[0]
        t_pose += time.time() - t1
        n = 0
        if res.keypoints is not None and res.boxes is not None and len(res.boxes):
            ids = res.boxes.id
            ids = ids.int().tolist() if ids is not None else [None] * len(res.boxes)
            xy = res.keypoints.xy.cpu().numpy()
            kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(xy.shape[:2])
            bh = res.boxes.xywh[:, 3].cpu().numpy()
            for k, tid in enumerate(ids):
                if tid is None:
                    continue
                tracks.setdefault(tid, {})[idx] = (xy[k], kc[k], float(bh[k]))
                n += 1
        people.append(n)
        if idx % every == 0:
            batch.append(frame.copy())
            batch_t.append(round(idx / TL.POSE_FPS, 2))
            if len(batch) >= 32:
                flush_emb()
        for t in need.get(idx, []):
            cv2.imwrite(str(frames_dir / TL.frame_name(t)), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        idx += 1
        if idx % int(TL.POSE_FPS * 120) == 0:          # every 2 minutes of video
            done_s, el = idx / TL.POSE_FPS, time.time() - started
            left = el / done_s * max(duration - done_s, 0)
            print(f"[vision] {video_id[:40]}: {done_s / 60:.0f} of {duration / 60:.0f} min of video · "
                  f"about {left / 60:.1f} min left", flush=True)
        if idx % int(TL.POSE_FPS * 120) == 0:          # every 2 minutes of video
            done_s, el = idx / TL.POSE_FPS, time.time() - started
            left = el / done_s * max(duration - done_s, 0)
            print(f"[vision] {video_id[:40]}: {done_s / 60:.0f} of {duration / 60:.0f} min of video · "
                  f"about {left / 60:.1f} min left", flush=True)
    flush_emb()
    proc.wait()
    n_frames = idx
    timings["decode_pose_embed"] = time.time() - t0 - timings["load_models"]
    timings["pose_gpu"], timings["embed_gpu"] = t_pose, t_emb

    # 3. Movement: per person (camera taken out), then the busiest two people per frame
    energies = []
    for tid, frames in tracks.items():
        if len(frames) < TL.POSE_FPS:          # under a second on screen — ignore
            continue
        a, b = min(frames), max(frames) + 1    # only the stretch this person is on screen
        kps = np.zeros((b - a, 17, 2))
        conf = np.zeros((b - a, 17))
        bh = np.ones(b - a)
        for i, (xy, kc, h_) in frames.items():
            kps[i - a], conf[i - a], bh[i - a] = xy, kc, h_
        full = np.full(n_frames, np.nan)
        full[a:b] = TL.person_energy(kps, conf, bh)
        energies.append(full)
    if energies:
        E = np.vstack(energies)
        top2 = np.sort(np.nan_to_num(E, nan=0.0), axis=0)[-2:].sum(axis=0)
        seen = np.isfinite(E).any(axis=0)
        curve = np.where(seen, top2, np.nan)
    else:
        curve = np.full(n_frames, np.nan)

    # 4. Per old scene: rhythm, people, mean embedding
    emb = np.vstack(emb_vecs) if emb_vecs else np.zeros((0, 1), np.float16)
    et = np.array(emb_times)
    scenes = []
    scene_embs, scene_ids = [], []
    for s in man["scenes"]:
        if "start" not in s or s["start"] >= duration:
            continue
        a, b = int(s["start"] * TL.POSE_FPS), int(min(s["end"], duration) * TL.POSE_FPS)
        seg = curve[a:b]
        row = {"scene_id": s["scene_id"], "start": s["start"], "end": min(s["end"], duration),
               "people": float(np.median(people[a:b])) if b > a else 0.0,
               **TL.rhythm_features(seg)}
        sel = (et >= s["start"]) & (et < min(s["end"], duration)) if et.size else np.array([], bool)
        if sel.any():
            v = emb[sel].astype(np.float32).mean(axis=0)
            scene_embs.append((v / (np.linalg.norm(v) or 1)).astype(np.float16))
            scene_ids.append(s["scene_id"])
        scenes.append(row)
    np.savez_compressed(out_dir / "embeddings.npz", frame_times=et, frame_embs=emb,
                        scene_ids=np.array(scene_ids), scene_embs=np.array(scene_embs))
    np.save(out_dir / "person_curve.npy", curve.astype(np.float32))

    timings["total"] = time.time() - started
    result = {"video_id": video_id, "duration": round(duration, 2), "partial": bool(max_seconds),
              "native_fps": native_fps,
              "pose_fps": TL.POSE_FPS, "cuts": cuts, "shot_error": shot_error, "frames": n_frames, "tracks": len(tracks),
              "targets": targets, "scenes": scenes, "timings": {k: round(v, 1) for k, v in timings.items()},
              "gpu": _gpu_name(VISION_GPU), "models": {"pose": POSE_MODEL, "embed": EMB_MODEL, "shots": "TransNetV2"}}
    (out_dir / "vision.json").write_text(json.dumps(result), encoding="utf-8")
    pack_vol.commit()
    return {"video_id": video_id, "seconds": round(timings["total"], 1), "cuts": len(cuts), "shot_error": shot_error,
            "targets": len(targets), "tracks": len(tracks)}


# ---------------------------------------------------------------------------
# tag — the teacher
# ---------------------------------------------------------------------------

def tags_file(video_id: str, key: str) -> Path:
    return Path(f"{OUT}/{video_id}/tags_{key}.json")


LOAD_LIMIT_MIN = 20     # a teacher that takes longer than this to start is stopped


def _gpu_status() -> str:
    """'GPU memory 18.2 / 46.1 GB · GPU busy 97%' from nvidia-smi, or '' if it can't be read."""
    import subprocess
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=10).stdout.strip().splitlines()[0]
        used, total, busy = (float(x) for x in line.split(","))
        return f"GPU memory {used / 1024:.1f} / {total / 1024:.0f} GB · GPU busy {busy:.0f}%"
    except Exception:
        return ""


def _start_heartbeat(key: str, t0: float, done):
    """Every 30 s while the teacher starts: elapsed time and what the GPU is doing, so a long
    start-up never looks like a hang. Memory climbing = loading the weights; memory full and the
    GPU idle = vLLM compiling / warming up on the CPU. Past LOAD_LIMIT_MIN it stops the container,
    leaving a note so the restart Modal makes doesn't try (and pay) again."""
    import threading

    def beat():
        while not done.wait(30):
            mins = (time.time() - t0) / 60
            print(f"[{key} teacher] starting… {int(mins)}:{int(mins % 1 * 60):02d} elapsed · "
                  f"{_gpu_status()}", flush=True)
            if mins > LOAD_LIMIT_MIN:
                print(f"[{key} teacher] gave up after {LOAD_LIMIT_MIN} minutes of starting.", flush=True)
                Path(f"{CACHE}/load_timeout_{key}.json").write_text(json.dumps({"at": time.time()}))
                try:
                    cache_vol.commit()
                finally:
                    os._exit(3)

    threading.Thread(target=beat, daemon=True).start()


def _load_teacher(self, key: str) -> None:
    # A failure here is caught, not raised: a container that fails to start is restarted by
    # Modal over and over (paying for the GPU each time). Instead every call reports the error.
    import threading
    t0 = time.time()
    self.llm, self.load_error, self.key = None, None, key
    self.load_seconds, self.first = 0.0, True
    note = Path(f"{CACHE}/load_timeout_{key}.json")
    stamp = (_read_json(note) or {}).get("at")
    if stamp and time.time() - stamp < 3600:
        self.load_error = f"it took over {LOAD_LIMIT_MIN} minutes to start last time — not trying again for an hour"
        print(f"[{key} teacher] {self.load_error}", flush=True)
        return
    print(f"[{key} teacher] starting {TEACHERS[key]} on {_gpu_name('?')} — progress every 30 s…", flush=True)
    done = threading.Event()
    _start_heartbeat(key, t0, done)
    try:
        from vllm import LLM
        self.llm = LLM(model=TEACHERS[key], max_model_len=8192, gpu_memory_utilization=0.90,
                       limit_mm_per_prompt={"image": 7, "video": 0}, max_num_seqs=32,
                       enable_prefix_caching=True)
        note.unlink(missing_ok=True)
        try:
            cache_vol.commit()          # keep what start-up compiled for next time
        except Exception:
            pass
        print(f"[{key} teacher] ready after {(time.time() - t0) / 60:.1f} min.", flush=True)
    except Exception as e:
        import traceback
        self.load_error = f"{e.__class__.__name__}: {e}"[:500]
        traceback.print_exc()
    finally:
        done.set()
    self.load_seconds = time.time() - t0


def _tag_video(self, video_id: str, max_prompts: int = 0, redo: bool = False) -> dict:
    import base64
    import trial_logic as TL
    from vllm import SamplingParams

    key = self.key
    if self.llm is None:
        return {"video_id": video_id, "teacher": key, "error": f"the {key} model didn't load — {self.load_error}"}
    pack_vol.reload()
    out = tags_file(video_id, key)
    vis = _read_json(f"{OUT}/{video_id}/vision.json")
    if not vis:
        return {"video_id": video_id, "teacher": key, "error": "no vision.json — run the vision stage first"}
    if not redo and _tags_done(video_id, key):
        return {"video_id": video_id, "teacher": key, "skipped": True}
    started = time.time()
    vocab = _vocab()
    before, after = TL.build_prompt(vocab)
    frames_dir = Path(f"{OUT}/{video_id}/frames")
    targets = vis["targets"][:max_prompts] if max_prompts else vis["targets"]
    convs, keep = [], []
    for tg in targets:
        parts = [{"type": "text", "text": before}]
        ok = True
        for t in TL.context_times(tg["t"], vis["duration"]):
            f = frames_dir / TL.frame_name(t)
            if not f.exists():
                ok = False
                break
            b64 = base64.b64encode(f.read_bytes()).decode()
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if ok:
            parts.append({"type": "text", "text": after})
            convs.append([{"role": "user", "content": parts}])
            keep.append(tg)
    outs, chunk, params = [], 64, SamplingParams(temperature=0.0, max_tokens=256)
    for i in range(0, len(convs), chunk):
        outs += self.llm.chat(convs[i:i + chunk], params, use_tqdm=False)
        el = time.time() - started
        left = el / len(outs) * (len(convs) - len(outs))
        print(f"[{key} teacher] {video_id[:40]}: {len(outs)}/{len(convs)} moments tagged · "
              f"{el / 60:.1f} min so far, about {left / 60:.1f} min left", flush=True)
    results = {}
    bad = 0
    for tg, o in zip(keep, outs):
        ans = TL.parse_answer(o.outputs[0].text, vocab)
        bad += not ans["ok"]
        results[tg["key"]] = {**ans, "t": tg["t"], "scene_id": tg.get("scene_id"),
                              "gold_id": tg.get("gold_id"), "raw": o.outputs[0].text[:400]}
    seconds = time.time() - started
    load = self.load_seconds if self.first else 0.0
    self.first = False
    out.write_text(json.dumps({"video_id": video_id, "teacher": key, "model": TEACHERS[key],
                               "gpu": _gpu_name(TEACHER_GPUS[key][0]),
                               "partial": bool(max_prompts) or bool(vis.get("partial")),
                               "targets": len(vis["targets"]),
                               "seconds": round(seconds, 1), "load_seconds": round(load, 1),
                               "prompts": len(convs), "unreadable": bad,
                               "vocab": [r["tag"] for r in vocab], "results": results}), encoding="utf-8")
    pack_vol.commit()
    return {"video_id": video_id, "teacher": key, "prompts": len(convs), "seconds": round(seconds, 1),
            "unreadable": bad}


@app.cls(image=tag_image, gpu=TEACHER_GPUS["8b"], volumes={DATA: pack_vol, CACHE: cache_vol},
         timeout=4 * 3600, cpu=8, memory=65536, max_containers=1)
class Tagger8B:
    @modal.enter()
    def load(self):
        _load_teacher(self, "8b")

    @modal.method()
    def tag_video(self, video_id: str, max_prompts: int = 0, redo: bool = False) -> dict:
        return _tag_video(self, video_id, max_prompts, redo)


@app.cls(image=tag_image, gpu=TEACHER_GPUS["32b"], volumes={DATA: pack_vol, CACHE: cache_vol},
         timeout=4 * 3600, cpu=8, memory=98304, max_containers=1)
class Tagger32B:
    @modal.enter()
    def load(self):
        _load_teacher(self, "32b")

    @modal.method()
    def tag_video(self, video_id: str, max_prompts: int = 0, redo: bool = False) -> dict:
        return _tag_video(self, video_id, max_prompts, redo)


TAGGERS = {"8b": Tagger8B, "32b": Tagger32B}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@app.function(image=report_image, volumes={DATA: pack_vol}, timeout=1800)
def report() -> str:
    import numpy as np
    import trial_logic as TL

    pack_vol.reload()
    pack = _read_json(f"{PACK}/pack.json", {}) or {}
    gold = _gold_items()
    vocab = _vocab()
    tags = [r["tag"] for r in vocab]
    done = sorted(p.name for p in Path(OUT).iterdir() if (p / "vision.json").exists()) if Path(OUT).exists() else []
    vis = {v: _read_json(f"{OUT}/{v}/vision.json") for v in done}
    mans = {v: _read_json(f"{PACK}/manifests/{v}.json") for v in done}
    # {teacher: {video: tags json}} — "tags.json" from before there were two teachers counts as 8b
    tagged_by = {}
    for v in done:
        for key in TEACHERS:
            tj = _read_json(tags_file(v, key)) or (_read_json(f"{OUT}/{v}/tags.json") if key == "8b" else None)
            if tj:
                tagged_by.setdefault(key, {})[v] = tj
    keys = [k for k in TEACHERS if k in tagged_by]

    # --- Tags: old tagger vs each teacher, on the same labelled frames (ones every teacher tagged)
    gold_tags = [i for i in gold if i.get("type") == "tags" and not i.get("skipped")]
    old_all = TL.score_tags([(i.get("suggested") or [], i.get("tags") or []) for i in gold_tags], tags)
    by_gold = {}
    for key in keys:
        for tj in tagged_by[key].values():
            for r in tj["results"].values():
                if r.get("gold_id"):
                    by_gold.setdefault(key, {})[r["gold_id"]] = r
    trial_gold = [i for i in gold_tags if keys and all(i["id"] in by_gold.get(k, {}) for k in keys)]
    old_trial = TL.score_tags([(i.get("suggested") or [], i.get("tags") or []) for i in trial_gold], tags)
    scores = {k: TL.score_tags([(by_gold[k][i["id"]]["tags"], i.get("tags") or []) for i in trial_gold], tags)
              for k in keys}
    # The better teacher overall is the one the verdicts (and the library run) go with.
    best = max(keys, key=lambda k: scores[k]["overall"]["f1"] or 0.0) if keys else None
    teacher = scores[best] if best else TL.score_tags([], tags)
    tagged = tagged_by.get(best, {})
    positives = {t: sum(t in (i.get("tags") or []) for i in gold_tags) for t in tags}
    kinds = {r["tag"]: r["kind"] for r in vocab}

    def verdict(t):
        tp = teacher["per_tag"][t]
        ot = old_trial["per_tag"][t]
        n = tp["tp"] + tp["fn"]
        if n < 3:
            return "need more labels"
        if tp["f1"] is not None and tp["f1"] >= 0.6 and (ot["f1"] is None or tp["f1"] >= ot["f1"]):
            return "teacher = yes"
        if tp["f1"] is not None and tp["f1"] < 0.4:
            return "teacher = no (learn from your labels)"
        return "borderline — label more"

    # --- Scene changes
    shot_errors = {v: d["shot_error"] for v, d in vis.items() if d.get("shot_error")}
    pred = {v: d["cuts"] for v, d in vis.items() if v not in shot_errors}
    old = {v: [s["start"] for s in mans[v]["scenes"] if s.get("start", 0) > 0.2 and s["start"] <= vis[v]["duration"]]
           for v in done}
    bounds = TL.score_boundaries(gold, pred, old)

    def lengths(cuts, dur):
        edges = [0.0] + sorted(cuts) + [dur]
        return np.diff(edges)

    shot_stats = {"old_scenes": 0, "new_shots": 0, "old_long": 0, "new_long": 0}
    for v in pred:
        d = vis[v]["duration"]
        lo, ln = lengths(old[v], d), lengths(pred[v], d)
        shot_stats["old_scenes"] += len(lo)
        shot_stats["new_shots"] += len(ln)
        shot_stats["old_long"] += int((lo > 60).sum())
        shot_stats["new_long"] += int((ln > 60).sum())

    # --- Movement
    rows = [s for d in vis.values() for s in d["scenes"] if s.get("level") is not None]
    regular = [s for s in rows if s["periodicity"] >= 0.5]
    old_m, new_m = [], []
    for v in done:
        mi = {s["scene_id"]: s.get("motion_intensity") for s in mans[v]["scenes"]}
        for s in vis[v]["scenes"]:
            if s.get("level") is not None and mi.get(s["scene_id"]) is not None:
                old_m.append(mi[s["scene_id"]])
                new_m.append(s["level"])
    corr = float(np.corrcoef(old_m, new_m)[0, 1]) if len(old_m) > 2 else None

    # --- Cost
    trial_hours = sum(d["duration"] for d in vis.values()) / 3600
    # Start-up is counted apart from the work: vision loads its models once per video, a
    # teacher once per run — so a short trial doesn't make the library look far dearer.
    lib_hours = float(pack.get("library_hours") or 0) or trial_hours
    lib_videos = int(pack.get("library_videos") or 0) or len(done)
    v_fixed = [d["timings"].get("load_models", 0.0) for d in vis.values()]
    stage_sec = {"vision": sum(d["timings"]["total"] for d in vis.values()) - sum(v_fixed)}
    fixed = {"vision": (sum(v_fixed) / max(len(v_fixed), 1), lib_videos)}
    for k in keys:
        stage_sec[f"tag {k}"] = sum(t["seconds"] for t in tagged_by[k].values())
        fixed[f"tag {k}"] = (sum(t.get("load_seconds", 0) for t in tagged_by[k].values()), 1)

    def used(rows, default):        # the GPU most of the work actually ran on
        names = [r.get("gpu") or default for r in rows]
        return max(set(names), key=names.count) if names else default

    gpus = {"vision": used(vis.values(), VISION_GPU)}
    gpus.update({f"tag {k}": used(tagged_by[k].values(), TEACHER_GPUS[k][0]) for k in keys})
    cost = TL.extrapolate(stage_sec, gpus, trial_hours, lib_hours, fixed)
    # The library only needs one teacher: the estimate counts vision + the better one.
    lib_total = cost["vision"]["library_$"] + (cost[f"tag {best}"]["library_$"] if best else 0.0)

    def f(x):
        return "–" if x is None else f"{x:.2f}"

    names = {k: TEACHERS[k].split("/")[-1] for k in keys}
    md = [f"# Modal trial report — {time.strftime('%d %b %Y %H:%M')}", "",
          f"**{len(done)} videos, {trial_hours:.2f} h of footage** processed. "
          f"Teachers: {', '.join(names.values()) or 'none yet'}. Library: {lib_hours:.1f} h.", "",
          "## Cost", "",
          "| Stage | GPU | Trial | Per footage hour | Full library (est.) | GPU hours (library) |",
          "|---|---|---|---|---|---|"]
    for k in stage_sec:
        c = cost[k]
        md.append(f"| {k} | {c['gpu']} | ${c['trial_$']:.2f} | ${c['per_footage_hour_$']:.3f} | "
                  f"${c['library_$']:.2f} | {c['library_gpu_hours']:.1f} |")
    # Where the vision time goes, in seconds per hour of footage.
    steps = {"Scene cuts (TransNetV2)": "shots", "Loading models": "load_models",
             "Pose + tracking (GPU)": "pose_gpu", "Embeddings (GPU)": "embed_gpu"}
    tot = {k: sum(d["timings"].get(v, 0.0) for d in vis.values()) for k, v in steps.items()}
    dpe = sum(d["timings"].get("decode_pose_embed", 0.0) for d in vis.values())
    tot["Decoding, tracker, saving frames (CPU)"] = max(dpe - tot["Pose + tracking (GPU)"] - tot["Embeddings (GPU)"], 0.0)
    tot["Motion + rhythm maths"] = max(sum(d["timings"].get("total", 0.0) for d in vis.values())
                                       - dpe - tot["Scene cuts (TransNetV2)"] - tot["Loading models"], 0.0)
    vision_breakdown = {k: round(v / trial_hours) if trial_hours else 0 for k, v in tot.items()}
    md_steps = ["", "Where the vision time goes (seconds per hour of footage):", "",
                "| Step | Seconds | Share |", "|---|---|---|"]
    all_s = sum(vision_breakdown.values()) or 1
    md_steps += [f"| {k} | {v} | {100 * v / all_s:.0f}% |" for k, v in vision_breakdown.items()]
    md += [f"| **total** | | **${cost['total']['trial_$']:.2f}** | | "
           f"**${lib_total:.2f}**{' (vision + ' + best + ' only)' if len(keys) > 1 else ''} | |", "",
           *md_steps, "",
           "## Tags", "",
           f"Your labels: {len(gold_tags)} frames ({len(trial_gold)} in the trial videos, tagged by every teacher).",
           "Overall F1 on those frames (1 = matches you exactly): old tagger "
           f"**{f(old_trial['overall']['f1'])}**"
           + "".join(f", {k} **{f(scores[k]['overall']['f1'])}**" for k in keys)
           + f". Old tagger on all your frames: {f(old_all['overall']['f1'])}.", ""]
    if len(keys) > 1 and trial_gold:
        md.append(f"**Better teacher: {best}** — the suggestions below use it.")
        md.append("")
    elif len(keys) > 1:
        md.append("**No better teacher yet** — none of your labelled frames are in these videos, so the "
                  "models can't be scored. The full run includes them.")
        md.append("")
    head = "| Tag | Kind | Your examples | Old F1 | " + " | ".join(f"{k} F1" for k in keys) + \
           f" | {best or 'teacher'} P / R | Suggestion |"
    md += [head, "|" + "---|" * (6 + len(keys))]
    for t in sorted(tags, key=lambda t: -positives[t]):
        tp, ot = teacher["per_tag"][t], old_trial["per_tag"][t]
        md.append(f"| {t} | {kinds[t]} | {positives[t]} | {f(ot['f1'])} | "
                  + " | ".join(f(scores[k]["per_tag"][t]["f1"]) for k in keys)
                  + f" | {f(tp['precision'])} / {f(tp['recall'])} | {verdict(t)} |")
    md.append("")
    for k in keys:
        unreadable = sum(t.get("unreadable", 0) for t in tagged_by[k].values())
        prompts = sum(t.get("prompts", 0) for t in tagged_by[k].values())
        md.append(f"{k}: replies that weren't valid JSON {unreadable} of {prompts}.")
    md += ["",
           "## Scene changes", "",
           f"Your answers in the trial videos: {bounds['n']}.", "",
           "| Detector | Accuracy | Precision | Recall |", "|---|---|---|---|",
           f"| Old (PySceneDetect) | {f(bounds['old']['accuracy'])} | {f(bounds['old']['precision'])} | {f(bounds['old']['recall'])} |",
           f"| New (TransNetV2) | {f(bounds['new']['accuracy'])} | {f(bounds['new']['precision'])} | {f(bounds['new']['recall'])} |",
           "", f"Scenes: old {shot_stats['old_scenes']} ({shot_stats['old_long']} over 60 s) → new "
               f"{shot_stats['new_shots']} ({shot_stats['new_long']} over 60 s).", "",
           "## Movement", "",
           f"{len(rows)} scenes with people tracked; {len(regular)} ({100 * len(regular) / max(len(rows), 1):.0f}%) "
           f"move with a clear regular rhythm (periodicity ≥ 0.5)."
           + (f" Median movement BPM of those: {np.median([s['movement_bpm'] for s in regular]):.0f}." if regular else ""),
           f"How much the old motion number agrees with the new person-motion level: "
           f"{'–' if corr is None else f'{corr:.2f}'} (1 = the same; low means the old number was mostly "
           f"camera / picture change).", ""]
    if shot_errors:
        md += ["## Scene detector errors", ""] + [f"- {v}: {e}" for v, e in shot_errors.items()] + [""]
    errs = [v for v in done if not any(v in tagged_by[k] for k in keys)]
    if errs:
        md += ["## Not tagged yet", "", ", ".join(errs), ""]
    text = "\n".join(md)
    Path(f"{OUT}/report.md").write_text(text, encoding="utf-8")
    Path(f"{OUT}/report.json").write_text(json.dumps({
        "cost": cost, "library_total": lib_total, "best_teacher": best, "vision_breakdown": vision_breakdown,
        "tags": {"old_all": old_all, "old_trial": old_trial, "teachers": scores,
                               "positives": positives, "verdicts": {t: verdict(t) for t in tags}},
        "boundaries": bounds, "shots": shot_stats, "movement_corr": corr}, indent=1), encoding="utf-8")
    pack_vol.commit()
    return text


# ---------------------------------------------------------------------------
# export — a compact bundle for the web app's 🧪 Trial results view: results.json, report.md and
# one frame per tagged moment. Written to /trial/export on the Volume; the laptop then downloads
# it into the Drive catalogue folder (catalogue/trial_results), and the server's Drive sync picks
# it up.
# ---------------------------------------------------------------------------

CURVE_FPS = 2.0         # motion curves in the bundle: 2 points a second


def _resample_curve(values, fps: float, out_fps: float = CURVE_FPS) -> list:
    """Mean of each 1/out_fps-second window (NaN-aware), rounded — for small charts."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    if v.size == 0 or fps <= 0:
        return []
    n = int(v.size / fps * out_fps)          # whole 1/out_fps-second windows
    if n == 0:
        return []
    edges = np.round(np.arange(n + 1) * fps / out_fps).astype(int)
    out = []
    with np.errstate(all="ignore"):
        for a, b in zip(edges[:-1], edges[1:]):
            x = np.nanmean(v[a:max(b, a + 1)]) if np.isfinite(v[a:max(b, a + 1)]).any() else np.nan
            out.append(None if not np.isfinite(x) else round(float(x), 5))
    return out


@app.function(image=report_image, volumes={DATA: pack_vol}, timeout=1800)
def export(smoke: bool = False) -> dict:
    import shutil
    import numpy as np
    import trial_logic as TL

    pack_vol.reload()
    EXPORT = f"{OUT}/export"      # (read here, so tests can point OUT elsewhere)
    if Path(EXPORT).exists():
        shutil.rmtree(EXPORT)
    Path(f"{EXPORT}/frames").mkdir(parents=True)
    gold = {i["id"]: i for i in _gold_items()}
    done = sorted(p.name for p in Path(OUT).iterdir() if (p / "vision.json").exists())
    videos, n_frames = {}, 0
    for v in done:
        vis = _read_json(f"{OUT}/{v}/vision.json")
        man = _read_json(f"{PACK}/manifests/{v}.json") or {"scenes": []}
        teachers = {}
        for key in TEACHERS:
            tj = _read_json(tags_file(v, key)) or (_read_json(f"{OUT}/{v}/tags.json") if key == "8b" else None)
            if tj:
                teachers[key] = tj["results"]
        old_scenes = {s["scene_id"]: s for s in man["scenes"]}
        moments = []
        for tg in vis["targets"]:
            src = Path(f"{OUT}/{v}/frames/{TL.frame_name(tg['t'])}")
            frame = None
            if src.exists():
                frame = f"frames/{v}/{tg['key']}.jpg"
                Path(f"{EXPORT}/frames/{v}").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, f"{EXPORT}/{frame}")
                n_frames += 1
            sc = old_scenes.get(tg.get("scene_id")) or {}
            g = gold.get(tg.get("gold_id")) if tg.get("gold_id") else None
            moments.append({
                "key": tg["key"], "t": tg["t"], "scene_id": tg.get("scene_id"), "frame": frame,
                "mine": None if g is None else list(g.get("tags") or []),
                "old": list(sc.get("corrected_tags") if sc.get("corrected_tags") is not None else sc.get("tags") or []),
                "teachers": {k: {f: r[tg["key"]].get(f) for f in ("tags", "unsure", "caption")}
                             for k, r in teachers.items() if tg["key"] in r},
            })
        old_curve, old_fps = [], 0.0
        for s in man["scenes"]:                 # the old pixel-change curve, stitched scene by scene
            c = s.get("motion_curve") or {}
            if c.get("values"):
                old_fps = float(c.get("fps") or old_fps or 25.0)
                start = int(round(s.get("start", 0) * old_fps))
                if len(old_curve) < start:
                    old_curve += [float("nan")] * (start - len(old_curve))
                old_curve += c["values"]
        pc = Path(f"{OUT}/{v}/person_curve.npy")
        person = _resample_curve(np.load(pc), vis.get("pose_fps", TL.POSE_FPS)) if pc.exists() else []
        boundaries = [{"t": i["t"], "verdict": i.get("verdict"), "source": i.get("source")}
                      for i in gold.values() if i.get("type") == "boundary" and i.get("video_id") == v
                      and not i.get("skipped") and i.get("t") is not None and i["t"] <= vis["duration"]]
        videos[v] = {
            "duration": vis["duration"], "partial": bool(vis.get("partial")),
            "cuts_new": vis["cuts"],
            "cuts_old": [s["start"] for s in man["scenes"] if 0.2 < s.get("start", 0) <= vis["duration"]],
            "boundaries": boundaries,
            "scenes": [{**{k: s.get(k) for k in ("scene_id", "start", "end", "level", "periodicity",
                                                    "movement_bpm", "people")},
                        "onsets": s.get("onsets", [])[:200],
                        "old_motion": (old_scenes.get(s["scene_id"]) or {}).get("motion_intensity")}
                       for s in vis["scenes"]],
            "curve_fps": CURVE_FPS, "person_curve": person,
            "old_curve": _resample_curve(old_curve, old_fps) if old_fps else [],
            "gpu": vis.get("gpu"), "tracks": vis.get("tracks"),
            "moments": moments,
        }
    report_json = _read_json(f"{OUT}/report.json", {}) or {}
    results = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M"), "smoke": smoke,
               "teachers": {k: TEACHERS[k] for k in TEACHERS
                            if any(k in m["teachers"] for vd in videos.values() for m in vd["moments"])},
               "report": report_json, "videos": videos}
    Path(f"{EXPORT}/results.json").write_text(json.dumps(results), encoding="utf-8")
    if Path(f"{OUT}/report.md").exists():
        shutil.copyfile(f"{OUT}/report.md", f"{EXPORT}/report.md")
    pack_vol.commit()
    return {"videos": len(videos), "frames": n_frames}


def _download_export(dest_name: str = "trial_results") -> str:
    """Laptop side: fetch /trial/export into the Drive catalogue folder (config.py's CATALOGUE_DIR),
    replacing the previous trial's results. Returns where it went (or why it couldn't)."""
    import shutil
    import subprocess
    import sys
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from config import CATALOGUE_DIR
    except Exception as e:
        return f"couldn't read CATALOGUE_DIR from config.py ({e}) — results stay on Modal"
    tmp = Path(__file__).resolve().parent / "_trial_download"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    r = subprocess.run(["modal", "volume", "get", "--force", "media-train", "/trial/export", str(tmp)],
                       capture_output=True, text=True)
    found = list(tmp.rglob("results.json"))
    if r.returncode != 0 or not found:
        return f"download failed: {(r.stderr or r.stdout).strip()[-400:]}"
    dest = Path(CATALOGUE_DIR) / dest_name
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(found[0].parent, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return str(dest)


@app.function(image=report_image, volumes={DATA: pack_vol})
def status(keys: list = None) -> dict:
    """Every packed video: its batch, and when vision and each teacher's tagging were done (and how
    long they took) — or that they're still to do. {"batches": {...}, "rows": [...]}."""
    pack_vol.reload()
    keys = keys or [TEACHERS_DEFAULT]
    batches = _read_json(f"{PACK}/batches.json", {}) or {}
    batch_of = {v: str(b) for b, ids in batches.items() for v in ids}
    rows = []
    for m in sorted(Path(f"{PACK}/manifests").glob("*.json")):
        v = m.stem
        vf = Path(f"{OUT}/{v}/vision.json")
        vis = _read_json(vf) or {}
        row = {"video_id": v, "batch": batch_of.get(v, "1" if not batches else "?"),
               "vision": _vision_done(v),
               "vision_at": vf.stat().st_mtime if vf.exists() else None,
               "vision_s": (vis.get("timings") or {}).get("total"),
               "minutes": round((vis.get("duration") or 0) / 60, 1)}
        for k in keys:
            tf = Path(f"{OUT}/{v}/tags_{k}.json")
            tj = _read_json(tf) or {}
            row[f"tags_{k}"] = _tags_done(v, k)
            row[f"tags_{k}_at"] = tf.stat().st_mtime if tf.exists() else None
            row[f"tags_{k}_s"] = tj.get("seconds")
        rows.append(row)
    return {"batches": batches, "rows": rows}


def _print_status(st: dict, keys: list) -> None:
    def when(ts):
        return time.strftime("%d %b %H:%M", time.gmtime(ts)) if ts else "—"
    rows = sorted(st["rows"], key=lambda r: (int(r["batch"]) if str(r["batch"]).isdigit() else 999, r["video_id"]))
    print(f"\n{'Batch':>5}  {'Video':<42} {'Vision (UTC)':<14} " + " ".join(f"{k + ' tags (UTC)':<15}" for k in keys))
    for r in rows:
        line = f"{r['batch']:>5}  {r['video_id'][:42]:<42} {(when(r['vision_at']) if r['vision'] else 'to do'):<14} "
        line += " ".join(f"{(when(r[f'tags_{k}_at']) if r[f'tags_{k}'] else 'to do'):<15}" for k in keys)
        print(line)
    done_v = sum(r["vision"] for r in rows)
    print(f"\n{len(rows)} videos packed · vision done {done_v} · "
          + " · ".join(f"{k} tags done {sum(r[f'tags_{k}'] for r in rows)}" for k in keys))


@app.function(image=report_image, volumes={DATA: pack_vol})
def list_pack() -> list:
    pack_vol.reload()
    return sorted(p.stem for p in Path(f"{PACK}/manifests").glob("*.json"))


# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "all", smoke: bool = False, videos: str = "", redo: bool = False, teacher: str = ""):
    """teacher: 8b, 32b or both (default: 8b for --smoke, 32b otherwise — chosen after batch 1)."""
    teacher = (teacher or ("8b" if smoke else "32b")).lower()
    keys = list(TEACHERS) if teacher == "both" else [teacher]
    if any(k not in TEACHERS for k in keys):
        print(f"--teacher must be one of: {', '.join(TEACHERS)}, both")
        return
    stages = ["download", "vision", "tag", "report", "export"] if stage == "all" else [stage]
    if stage == "status":
        _print_status(status.remote(keys), keys)
        return
    vids = [v.strip() for v in videos.split(",") if v.strip()] or list_pack.remote()
    if not vids:
        print("No videos in the pack — run pack_for_modal.bat --upload first.")
        return
    todo_vision, todo_tags = list(vids), {k: list(vids) for k in keys}
    if not smoke and not redo and ("vision" in stages or "tag" in stages):
        # Only send Modal the videos that still need work — finished ones (earlier batches)
        # aren't started at all, so they cost nothing.
        st = {r["video_id"]: r for r in status.remote(keys)["rows"]}
        todo_vision = [v for v in vids if not (st.get(v) or {}).get("vision")]
        todo_tags = {k: [v for v in vids if not (st.get(v) or {}).get(f"tags_{k}")] for k in keys}
        print(f"{len(vids)} videos packed. To do: vision {len(todo_vision)}, "
              + ", ".join(f"{k} tags {len(todo_tags[k])}" for k in keys)
              + f". Already done (skipped): vision {len(vids) - len(todo_vision)}.")
        for v in todo_vision:
            print(f"   new: {v}")
    max_seconds, max_prompts = (90, 12) if smoke else (0, 0)
    if smoke:
        vids = vids[:1]
        redo = True
        print(f"Smoke test on {vids[0]} (first {max_seconds} s, {max_prompts} prompts).")
    t_all = time.time()
    if "download" in stages:
        print(download_models.remote([TEACHERS[k] for k in keys]))
    if "vision" in stages and todo_vision:
        print(f"Vision on {len(todo_vision)} video(s) ({' / '.join(VISION_GPUS)}, first free)…")
        for r in vision.starmap([(v, max_seconds, redo) for v in todo_vision], return_exceptions=True):
            print("  ", r)
    if "tag" in stages:
        for k in keys:
            if "vision" in stages and not smoke and not redo:
                # vision just ran — anything it finished needs tagging too
                st = {r["video_id"]: r for r in status.remote([k])["rows"]}
                todo_tags[k] = [v for v in vids if not (st.get(v) or {}).get(f"tags_{k}")]
            if not todo_tags[k]:
                print(f"{k}: every video already tagged.")
                continue
            print(f"Tagging {len(todo_tags[k])} video(s) with {TEACHERS[k]} ({' / '.join(TEACHER_GPUS[k])}, first free)…")
            tagger = TAGGERS[k]()
            for r in tagger.tag_video.starmap([(v, max_prompts, redo) for v in todo_tags[k]], return_exceptions=True):
                print("  ", r)
                if isinstance(r, dict) and "didn't load" in str(r.get("error", "")):
                    print(f"Stopping {k}: the model couldn't start (see the error above).")
                    break
    if "report" in stages:
        text = report.remote()
        name = "trial_report_smoke.md" if smoke else "trial_report.md"
        Path(name).write_text(text, encoding="utf-8")
        print("\n" + text + f"\n\nSaved {Path(name).resolve()}")
    if "export" in stages:
        print("Exporting results for the web app…")
        print("  ", export.remote(smoke))
        where = _download_export()
        print(f"   Trial results → {where}\n   In the app: ☁️ Sync from Google Drive, then "
              f"🎯 Training Labels → 🧪 Trial results.")
    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min.")
