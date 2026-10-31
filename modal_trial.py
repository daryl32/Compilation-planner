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
  prep (CPU)        • one prompts.json per video: every moment's full prompt with its 7 frames
                      already read and encoded, so the GPU reads nothing but that file; near-
                      identical moments in a scene are sent once and copy the answer (--dedup)
  tag (H100 GPU)    • Qwen3-VL-32B (the "teacher"; --teacher 32b-fp8 for Qwen's FP8 build) tags
                      every scene and every 🎯 tag card from 7 frames over 4 s, using your
                      tag_vocabulary.csv; records tokens/s and GPU use (MFU)
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
    modal run modal_trial.py --stage prep                 CPU only: shows how many moments dedup saves
    modal run modal_trial.py --stage fp8-check            the FP8 check: re-tags the 32B video with
                                                          the most labelled frames, then the report
    modal run modal_trial.py --stage tag --batch 6 --examples 3 --redo     re-tag one batch (the
                                                          earlier tags are kept as tags_<teacher>_before)
    modal run modal_trial.py --stage compare --batch 6    before vs after that re-tag: F1, time, cost
    modal run modal_trial.py --stage vision-check         re-run one finished video through the batched
                                                          vision pipeline; compare movement, cuts, cost
Stages: download, vision, prep, tag, report, export, status, results (report + export, CPU only),
all (default); tag always runs prep first. status lists every packed video with its batch and when
vision / tagging were done — nothing runs on a GPU. export puts the results in your Drive catalogue
folder (catalogue/trial_results) for the web app's 🧪 Trial results view. Each skips a video it has
already done unless you pass --redo (32b and 32b-fp8 count as the same teacher).
Options: --examples N worked examples from your labels per moment (default 0), --dedup S copy the
answer between moments in a scene more alike than S (default 0.96, 0 = off).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

# The teachers to compare. Both dense (non mixture-of-experts): the 30B-A3B MoE needed a
# FlashInfer MoE kernel that vLLM compiles from scratch on start-up (20+ silent minutes).
TEACHERS = {"8b": "Qwen/Qwen3-VL-8B-Instruct", "32b": "Qwen/Qwen3-VL-32B-Instruct",
            "32b-fp8": "Qwen/Qwen3-VL-32B-Instruct-FP8"}     # Qwen's own FP8 release of the 32B
TEACHER_GPUS = {"8b": ["L40S", "A100-40GB", "A100-80GB", "H100"],   # 17 GB of weights
                "32b": ["H100", "A100-80GB"],                        # 66 GB of weights
                "32b-fp8": ["H100"]}                                 # 33 GB; FP8 maths needs Hopper
# 32b and 32b-fp8 are the same teacher at two precisions: a video tagged by either counts as
# tagged, and the app shows them under one name (both names only where a video has both).
SLOT = {"8b": "8b", "32b": "32b", "32b-fp8": "32b"}
VLM_MODEL = TEACHERS["8b"]
TEACHERS_DEFAULT = "32b-fp8"  # the 32B (chosen after batch 1), in FP8 since batch 6: same
                              # accuracy in the check, much faster on the H100
EMB_MODEL = "google/siglip2-so400m-patch14-384"
# Medium pose model: about twice as fast as the large one, and closer to what the laptop's
# RTX 4070 can run in its 5-minute-per-video budget.
POSE_MODEL = "yolo11m-pose.pt"
READ_AHEAD = 64         # decoded frames queued ahead of the pose model (~44 MB at 640×360)
# First choice first; Modal falls back down the list when that GPU type is busy.
VISION_GPUS = ["L4", "A10", "L40S", "A100-40GB"]
VISION_GPU = VISION_GPUS[0]

# Worked examples from your labels shown to the teacher before each moment (--examples N).
# Off by default since batch 5: they cost 30–40% more per moment for no clear gain.
EXAMPLES_DEFAULT, EXAMPLES_MAX = 0, 6
# 🧪 Test set: the same ~2 h of already-processed videos re-tagged after every settings change,
# so runs are compared on identical frames. Picked once (--stage test-pick --redo picks again).
TEST_HOURS = 2.0
TEST_MIN_LABELLED = 5           # a test video needs at least this many of your labelled frames
# Near-identical moments in a scene are tagged once and the answer copied (--dedup 0 = tag all).
DEDUP_DEFAULT = 0.96
TAG_MAX_TOKENS = 800        # the reply lists every tag that might apply, with a confidence — room for ~50

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
# LoRA training (QLoRA: the 32B in 4-bit, small trainable adapters on its language layers)
lora_image = (base.pip_install("torch", "torchvision", "transformers>=4.57", "peft>=0.17", "bitsandbytes>=0.46", "accelerate",
                               "numpy", "pillow", "huggingface_hub")
              .add_local_python_source("trial_logic"))
# Unsloth (hand-written GPU kernels + activation offloading): the default trainer from try2 on;
# --trainer hf is the plain transformers/PEFT path try1 was trained with.
lora_unsloth_image = (base.pip_install("unsloth", "unsloth_zoo", "torchvision", "numpy", "pillow", "huggingface_hub")
                      .add_local_python_source("trial_logic"))
LORA_RANK, LORA_ALPHA = 16, 32
LORA_SAVE_EVERY = 200          # examples between checkpoints (a multiple of the 8-example step)
LORA_MICRO = 8                 # examples through the GPU at once (8 ≈ 55 GB of the H100's 80; halves itself on OOM)
LORA_MARKER = "<|im_start|>assistant\n"   # the answer starts right after this in every example
# the language model's layers only (the vision part names its layers qkv / proj / linear_fc*)
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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


def _vocab_text() -> str:
    p = Path(f"{PACK}/library/tag_vocabulary.csv")
    try:
        return p.read_text(encoding="utf-8-sig")
    except OSError:
        return ""


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
    """A complete tagging of every moment the current vision result has, by this teacher or by the
    other precision of the same one (32b / 32b-fp8). Pass --redo to tag again regardless."""
    vis = _read_json(f"{OUT}/{video_id}/vision.json")
    if not vis or vis.get("partial"):
        return False
    for k in TEACHERS:
        if SLOT[k] != SLOT[key]:
            continue
        prev = _read_json(f"{OUT}/{video_id}/tags_{k}.json")
        if prev and not prev.get("partial") and prev.get("targets") == len(vis.get("targets") or []):
            return True
    return False


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
# vision — three steps, each on the right machine:
#   vision_prep (CPU)  decode the tiny 48×27 frames scene-cut detection needs
#   vision      (GPU)  scene cuts from those frames; pose in batches of POSE_BATCH (raw detections
#                      saved, no tracking); picture embeddings; the tagger's frames
#   vision_motion (CPU) the same BoT-SORT person tracker over the saved detections (it re-decodes
#                      the frames for its camera-motion step), then movement, rhythm and the
#                      per-scene numbers → vision.json, the "done" marker
# Tracking used to run between every pose frame on the GPU machine, so the GPU took one frame at
# a time and waited; now it only detects, 32 frames at once.
# ---------------------------------------------------------------------------

POSE_BATCH = 32
VISION_PIPELINE = 2         # 2 = batched pose + tracking on CPU (1 = tracked on the GPU machine)


def _pose_cmd(path: str, man: dict) -> tuple:
    """The ffmpeg command both the GPU step and the CPU tracking step decode with — the same
    frames bit for bit, so the tracker's camera-motion step sees what the pose model saw."""
    import trial_logic as TL
    src_w, src_h = int(man["proxy"].get("width") or 854), int(man["proxy"].get("height") or 480)
    w = 640
    h = int(round(w * src_h / src_w / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={TL.POSE_FPS},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return cmd, w, h


def _frames(cmd: list, w: int, h: int, ahead: int = READ_AHEAD):
    """Decoded frames, one at a time, with a reader thread keeping `ahead` frames queued."""
    import queue
    import subprocess
    import threading
    import numpy as np
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 8)
    q = queue.Queue(maxsize=ahead)
    n = w * h * 3

    def reader():
        try:
            while True:
                buf = proc.stdout.read(n)
                if len(buf) < n:
                    break
                q.put(buf)
        finally:
            q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    while True:
        buf = q.get()
        if buf is None:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


def _vision_gpu_done(video_id: str) -> bool:
    prev = _read_json(f"{OUT}/{video_id}/vision_gpu.json")
    return (bool(prev) and not prev.get("partial") and (prev.get("models") or {}).get("pose") == POSE_MODEL
            and Path(f"{OUT}/{video_id}/detections.npz").exists())


@app.function(image=vision_image, volumes={DATA: pack_vol}, timeout=3600, cpu=2, memory=4096)
def vision_prep(video_id: str, max_seconds: float = 0, redo: bool = False) -> dict:
    """CPU: the 48×27 RGB frames TransNetV2 reads, at the video's own frame rate, saved for the
    GPU step (decoding them was most of the scene-cut time on the GPU machine)."""
    import subprocess
    import cv2
    import numpy as np
    pack_vol.reload()
    if not redo and (_vision_done(video_id) or _vision_gpu_done(video_id)):
        return {"video_id": video_id, "skipped": True}
    t0 = time.time()
    path = _video_path(video_id, max_seconds)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", "48x27", "pipe:"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape([-1, 27, 48, 3])
    out_dir = Path(f"{OUT}/{video_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shots_frames.npy", frames)
    (out_dir / "shots_meta.json").write_text(json.dumps({"fps": float(fps), "frames": int(len(frames)),
                                                        "seconds": round(time.time() - t0, 1)}))
    pack_vol.commit()
    return {"video_id": video_id, "frames": int(len(frames)), "seconds": round(time.time() - t0, 1)}


def _transnet_model():
    import glob
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
    return model


def _transnet_cuts(video_id: str, path: str) -> tuple:
    """(cut times in seconds, fps): from the frames vision_prep decoded when they're there, else
    decoding here as before."""
    import numpy as np
    import torch
    import trial_logic as TL
    model = _transnet_model()
    fpath, meta = Path(f"{OUT}/{video_id}/shots_frames.npy"), _read_json(f"{OUT}/{video_id}/shots_meta.json")
    with torch.no_grad():
        if fpath.exists() and meta:
            frames = torch.from_numpy(np.load(fpath)).to(model.device)
            preds, _ = model.predict_frames(frames, quiet=True)
            fps = float(meta["fps"])
        else:
            _, preds, _ = model.predict_video(path, quiet=True)
            import cv2
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
    p = preds.cpu().numpy() if hasattr(preds, "cpu") else preds
    return TL.cuts_from_predictions(p, fps), fps


@app.function(image=vision_image, gpu=VISION_GPUS, volumes={DATA: pack_vol, CACHE: cache_vol},
              timeout=4 * 3600, cpu=4, memory=16384)
def vision(video_id: str, max_seconds: float = 0, redo: bool = False) -> dict:
    """GPU: scene cuts, pose detections in batches (no tracking), embeddings, the tagger's frames."""
    import cv2
    import numpy as np
    import torch
    import trial_logic as TL

    pack_vol.reload()
    out_dir = Path(f"{OUT}/{video_id}")
    if not redo and (_vision_done(video_id) or _vision_gpu_done(video_id)):
        return {"video_id": video_id, "skipped": True}
    started = time.time()
    timings = {}
    man = _read_json(f"{PACK}/manifests/{video_id}.json")
    duration = float(man["proxy"]["duration"])
    if max_seconds:
        duration = min(duration, float(max_seconds))
    path = _video_path(video_id, max_seconds)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scene changes (from vision_prep's frames)
    t0 = time.time()
    shot_error = None
    try:
        cuts, native_fps = _transnet_cuts(video_id, path)
    except Exception as e:      # keep going — pose, embeddings and frames still matter
        cuts, native_fps, shot_error = [], None, f"{e.__class__.__name__}: {e}"[:300]
    timings["shots"] = time.time() - t0
    Path(f"{OUT}/{video_id}/shots_frames.npy").unlink(missing_ok=True)

    # 2. One decode at 15 fps: pose in batches, embeddings at 2 fps, frames for the tagger
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

    cmd, w, h = _pose_cmd(path, man)
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
            inp = emb_proc(images=[cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in ebatch], return_tensors="pt").to("cuda")
            f = emb_model.get_image_features(pixel_values=inp["pixel_values"].half())
            if not torch.is_tensor(f):      # newer transformers return an output object
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
        results = pose.predict(pbatch, verbose=False, imgsz=640, conf=0.3)
        for i, res in zip(pbatch_i, results):
            b = res.boxes
            if b is None or not len(b):
                continue
            data = b.data.cpu().numpy().astype(np.float32)[:, :6]       # x1 y1 x2 y2 conf cls
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
    for frame in _frames(cmd, w, h):
        pbatch.append(frame)
        pbatch_i.append(idx)
        if len(pbatch) >= POSE_BATCH:
            flush_pose()
        if idx % every == 0:
            ebatch.append(frame.copy())
            ebatch_t.append(round(idx / TL.POSE_FPS, 2))
            if len(ebatch) >= 32:
                flush_emb()
        for t in need.get(idx, []):
            cv2.imwrite(str(frames_dir / TL.frame_name(t)), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        idx += 1
        if idx % int(TL.POSE_FPS * 120) == 0:          # every 2 minutes of video
            done_s, el = idx / TL.POSE_FPS, time.time() - started
            left = el / done_s * max(duration - done_s, 0)
            print(f"[vision] {video_id[:40]}: {done_s / 60:.0f} of {duration / 60:.0f} min of video · "
                  f"about {left / 60:.1f} min left", flush=True)
    flush_pose()
    flush_emb()
    n_frames = idx
    timings["decode_pose_embed"] = time.time() - t0 - timings["load_models"]
    timings["pose_gpu"], timings["embed_gpu"] = t_pose, t_emb

    def cat(parts, shape, dtype):
        return np.concatenate(parts) if parts else np.zeros(shape, dtype)
    np.savez_compressed(out_dir / "detections.npz", frame=cat(det_frame, (0,), np.int32),
                        box=cat(det_box, (0, 6), np.float32), kxy=cat(det_kxy, (0, 17, 2), np.float32),
                        kconf=cat(det_kconf, (0, 17), np.float32), n_frames=np.int32(n_frames))
    np.savez_compressed(out_dir / "frame_embs.npz", frame_times=np.array(emb_times),
                        frame_embs=np.vstack(emb_vecs) if emb_vecs else np.zeros((0, 1), np.float16))
    timings["total"] = time.time() - started
    result = {"video_id": video_id, "duration": round(duration, 2), "partial": bool(max_seconds),
              "native_fps": native_fps, "pose_fps": TL.POSE_FPS, "cuts": cuts, "shot_error": shot_error,
              "frames": n_frames, "targets": targets, "pipeline": VISION_PIPELINE, "pose_batch": POSE_BATCH,
              "timings": {k: round(v, 1) for k, v in timings.items()},
              "gpu": _gpu_name(VISION_GPU), "models": {"pose": POSE_MODEL, "embed": EMB_MODEL, "shots": "TransNetV2"}}
    (out_dir / "vision_gpu.json").write_text(json.dumps(result), encoding="utf-8")
    pack_vol.commit()
    return {"video_id": video_id, "seconds": round(timings["total"], 1), "cuts": len(cuts), "shot_error": shot_error,
            "targets": len(targets), "detections": int(sum(len(x) for x in det_box))}


def _tracker_cfg():
    """The tracker config model.track() uses by default in the installed ultralytics (its
    default.yaml `tracker:` — TrackTrack in 8.4, BoT-SORT in older releases), so tracking here is
    the same as it was on the GPU machine. The vision image isn't rebuilt, so the version is fixed."""
    from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    path = check_yaml(DEFAULT_CFG.tracker)
    try:
        from ultralytics.utils import YAML
        data = YAML.load(path)
    except ImportError:                       # older ultralytics
        from ultralytics.utils import yaml_load
        data = yaml_load(path)
    return IterableSimpleNamespace(**data)


def _track_offline(frames, det: dict, n_frames: int) -> tuple:
    """Ultralytics' own default tracker (the one model.track used) run over saved detections, frame
    by frame, with the frames for its camera-motion compensation — what model.track did on the GPU
    machine, minus the GPU. (For pose, model.track gives the tracker only the final detections, so
    nothing else needs saving.) Returns (tracks, people per frame, tracker name):
    tracks = {id: {frame: (keypoints xy, keypoint conf, tracked box height)}}."""
    import numpy as np
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    cfg = _tracker_cfg()
    cfg.device = "cpu"
    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
    fr = np.asarray(det["frame"])
    order = np.argsort(fr, kind="stable")
    starts = np.searchsorted(fr[order], np.arange(n_frames + 1))
    tracks, people = {}, []
    for idx, frame in enumerate(frames):
        if idx >= n_frames:
            break
        sel = order[starts[idx]:starts[idx + 1]]
        boxes = Boxes(np.asarray(det["box"][sel], dtype=np.float32).reshape(-1, 6), frame.shape[:2])
        out = tracker.update(boxes.cpu().numpy(), frame)
        n = 0
        for row in out if len(out) else []:
            j = sel[int(row[-1])]
            tracks.setdefault(int(row[4]), {})[idx] = (det["kxy"][j], det["kconf"][j], float(row[3] - row[1]))
            n += 1
        people.append(n)
    people += [0] * (n_frames - len(people))
    return tracks, people, cfg.tracker_type


@app.function(image=vision_image, volumes={DATA: pack_vol}, timeout=3 * 3600, cpu=4, memory=8192)
def vision_motion(video_id: str, max_seconds: float = 0) -> dict:
    """CPU, after the GPU step: person tracking, the camera-free movement curve, rhythm and the
    per-scene numbers; writes vision.json (the "done" marker), embeddings.npz, person_curve.npy."""
    import numpy as np
    import trial_logic as TL

    pack_vol.reload()
    out_dir = Path(f"{OUT}/{video_id}")
    vg = _read_json(out_dir / "vision_gpu.json")
    if not vg:
        return {"video_id": video_id, "error": "no vision_gpu.json — the GPU step didn't finish"}
    t0 = time.time()
    man = _read_json(f"{PACK}/manifests/{video_id}.json")
    z = np.load(out_dir / "detections.npz")
    det = {k: z[k] for k in ("frame", "box", "kxy", "kconf")}
    n_frames = int(z["n_frames"])
    cmd, w, h = _pose_cmd(_video_path(video_id, max_seconds), man)
    tracks, people, tracker_name = _track_offline(_frames(cmd, w, h), det, n_frames)
    t_track = time.time() - t0
    t1 = time.time()
    curve = TL.movement_curve(tracks, n_frames)
    fe = np.load(out_dir / "frame_embs.npz")
    et, emb = fe["frame_times"], fe["frame_embs"]
    scenes, scene_ids, scene_embs = TL.scene_features(man["scenes"], vg["duration"], curve, people, et, emb)
    np.savez_compressed(out_dir / "embeddings.npz", frame_times=et, frame_embs=emb,
                        scene_ids=np.array(scene_ids), scene_embs=np.array(scene_embs))
    np.save(out_dir / "person_curve.npy", curve.astype(np.float32))
    cpu = {"track": round(t_track, 1), "maths": round(time.time() - t1, 1),
           "prep": (_read_json(out_dir / "shots_meta.json") or {}).get("seconds", 0.0)}
    cpu["total"] = round(cpu["track"] + cpu["maths"] + cpu["prep"], 1)
    result = {**vg, "tracks": len(tracks), "tracker": tracker_name, "scenes": scenes, "cpu_timings": cpu}
    (out_dir / "vision.json").write_text(json.dumps(result), encoding="utf-8")
    (out_dir / "frame_embs.npz").unlink(missing_ok=True)
    pack_vol.commit()
    return {"video_id": video_id, "tracks": len(tracks), "cpu_seconds": cpu["total"]}


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


def _load_teacher(self, key: str, lora: bool = False) -> None:
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
        extra = {"enable_lora": True, "max_lora_rank": LORA_RANK, "max_loras": 1} if lora else {}
        self.llm = LLM(model=TEACHERS[key], max_model_len=8192, gpu_memory_utilization=0.90,
                       limit_mm_per_prompt={"image": 7 + EXAMPLES_MAX, "video": 0}, max_num_seqs=32,
                       enable_prefix_caching=True, **extra)
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


def _video_embeddings(video_id: str):
    """(frame_times, unit-length frame embeddings) from a video's vision run, or (None, None)."""
    import numpy as np
    try:
        z = np.load(f"{OUT}/{video_id}/embeddings.npz")
        t, e = np.asarray(z["frame_times"], dtype=float), np.asarray(z["frame_embs"], dtype=np.float32)
    except (OSError, KeyError, ValueError):
        return None, None
    return (t, e) if t.size and e.ndim == 2 and len(e) == t.size else (None, None)


def _example_pool() -> list:
    """Every 🎯 tag label (cards and ✏️ corrections) whose frame and embedding are already on the
    Volume from an earlier vision run: {"video_id", "t", "tags", "corrected", "emb", "frame"}."""
    import trial_logic as TL
    by_video = {}
    for it in _gold_items():
        if it.get("type") == "tags" and not it.get("skipped") and it.get("t") is not None and it.get("video_id"):
            by_video.setdefault(it["video_id"], []).append(it)
    pool = []
    print(f"Gathering your labelled frames from {len(by_video)} videos…", flush=True)
    for n_vid, (vid, items) in enumerate(by_video.items(), 1):
        if n_vid % 10 == 0:
            print(f"   {n_vid} of {len(by_video)} videos, {len(pool)} frames so far", flush=True)
        vis = _read_json(f"{OUT}/{vid}/vision.json")
        times, embs = _video_embeddings(vid)
        if not vis or times is None:
            continue
        for it in items:
            t = float(it["t"])
            frame = Path(f"{OUT}/{vid}/frames") / TL.frame_name(TL.context_times(t, vis["duration"])[3])
            j = TL.nearest_index(times, t)
            if j is None or abs(times[j] - t) > 1.0 or not frame.exists():
                continue
            pool.append({"video_id": vid, "t": t, "tags": list(it.get("tags") or []),
                         "corrected": set(it.get("tags") or []) != set(it.get("suggested") or []),
                         "emb": embs[j], "frame": str(frame)})
    return pool


POOL_FILE = f"{OUT}/example_pool.npz"


def _save_pool(pool: list) -> None:
    import numpy as np
    np.savez(POOL_FILE, video_ids=np.array([p["video_id"] for p in pool], dtype=object),
             ts=np.array([p["t"] for p in pool], dtype=float),
             embs=np.array([p["emb"] for p in pool], dtype=np.float16) if pool else np.zeros((0, 1), np.float16),
             corrected=np.array([p["corrected"] for p in pool], dtype=bool),
             frames=np.array([p["frame"] for p in pool], dtype=object),
             tags=np.array([json.dumps(p["tags"]) for p in pool], dtype=object))


def _load_pool():
    """The pool build_example_pool saved, or None if there isn't one."""
    import numpy as np
    try:
        z = np.load(POOL_FILE, allow_pickle=True)
        return [{"video_id": str(v), "t": float(t), "emb": e, "corrected": bool(c), "frame": str(f),
                 "tags": json.loads(str(g))}
                for v, t, e, c, f, g in zip(z["video_ids"], z["ts"], z["embs"], z["corrected"], z["frames"], z["tags"])]
    except (OSError, KeyError, ValueError):
        return None


@app.function(image=report_image, volumes={DATA: pack_vol}, timeout=1800, cpu=2)
def build_example_pool() -> int:
    """Gather your labelled frames for the teacher's worked examples into one file, on a cheap CPU
    container — reading them one by one inside the H100 container took minutes of GPU time."""
    pack_vol.reload()
    pool = _example_pool()
    _save_pool(pool)
    pack_vol.commit()
    return len(pool)


def prompts_file(video_id: str) -> Path:
    return Path(f"{OUT}/{video_id}/prompts.json")


@app.function(image=report_image, volumes={DATA: pack_vol}, timeout=1800, cpu=2, memory=6144)
def prepare_prompts(video_id: str, max_prompts: int = 0, examples: int = 0, dedup: float = DEDUP_DEFAULT,
                    only_labelled: bool = False) -> dict:
    """CPU, before the teacher starts: everything the GPU needs for one video in ONE file
    (prompts.json) — which moments to send, the full prompt for each with its frames already
    read and base64-encoded, and which near-identical moments just copy another's answer. The
    teacher container then reads one file and feeds the GPU; it opens no frames itself.
    only_labelled: just the moments you've labelled (🔍 label checks), none copied."""
    import base64
    import trial_logic as TL

    pack_vol.reload()
    vis = _read_json(f"{OUT}/{video_id}/vision.json")
    if not vis:
        return {"video_id": video_id, "error": "no vision.json — run the vision stage first"}
    t0 = time.time()
    vocab_csv = _vocab_text()
    vocab = TL.teacher_vocab((TL.parse_vocabulary(vocab_csv) if vocab_csv else []) or _vocab())
    # (tags switched off for the teacher aren't asked about)
    head, moment, tail = TL.build_prompt(vocab)
    targets = vis["targets"][:max_prompts] if max_prompts else vis["targets"]
    if only_labelled:
        gold = [i for i in _gold_items() if i.get("type") == "tags" and not i.get("skipped")
                and i.get("video_id") == video_id and i.get("t") is not None]
        ids, ts = {i["id"] for i in gold}, {round(float(i["t"]), 2) for i in gold}
        targets = [tg for tg in targets if tg.get("gold_id") in ids or round(float(tg["t"]), 2) in ts]
        dedup = 0
    by_key = {tg["key"]: tg for tg in targets}
    times, embs = _video_embeddings(video_id)
    keep, copies = TL.dedup_targets(targets, times, embs, dedup)
    for key, c in copies.items():
        tg = by_key[key]
        c.update({"t": tg["t"], "scene_id": tg.get("scene_id"), "gold_id": tg.get("gold_id")})

    examples = max(0, min(int(examples or 0), EXAMPLES_MAX))
    pool = (_load_pool() or _example_pool()) if examples else []
    frames_dir = Path(f"{OUT}/{video_id}/frames")
    cache = {}

    def img(path) -> dict:
        path = str(path)
        if path not in cache:
            cache[path] = base64.b64encode(Path(path).read_bytes()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cache[path]}"}}

    moments, used_examples, missing = [], 0, 0
    for tg in keep:
        parts = [{"type": "text", "text": head}]        # same for every moment → prefix-cached
        if pool and times is not None:
            j = TL.nearest_index(times, tg["t"])
            shots = TL.pick_examples(embs[j], pool, examples, exclude_video=video_id) if j is not None else []
            if shots:
                parts.append({"type": "text", "text": TL.examples_intro(len(shots))})
                for n, ex in enumerate(shots, 1):
                    parts += [img(ex["frame"]), {"type": "text", "text": TL.example_caption(n, ex["tags"], vocab)}]
                used_examples += len(shots)
        parts.append({"type": "text", "text": moment})
        frames = [frames_dir / TL.frame_name(t) for t in TL.context_times(tg["t"], vis["duration"])]
        if not all(f.exists() for f in frames):
            missing += 1
            continue
        parts += [img(f) for f in frames]
        parts.append({"type": "text", "text": tail})
        moments.append({"key": tg["key"], "t": tg["t"], "scene_id": tg.get("scene_id"),
                        "gold_id": tg.get("gold_id"), "conv": [{"role": "user", "content": parts}]})
    sent = {m["key"] for m in moments}
    copies = {k: c for k, c in copies.items() if c["of"] in sent}      # a copy needs its original
    data = {"video_id": video_id, "created": time.time(), "targets": len(vis["targets"]),
            "partial": bool(max_prompts) or bool(vis.get("partial")),
            "vocab": [r["tag"] for r in vocab], "vocab_csv": vocab_csv,
            "examples": examples, "examples_used": used_examples,
            "dedup": dedup, "copies": copies, "moments": moments}
    pf = prompts_file(video_id)
    pf.write_text(json.dumps(data), encoding="utf-8")
    pack_vol.commit()
    return {"video_id": video_id, "targets": len(targets), "prompts": len(moments), "copied": len(copies),
            "missing_frames": missing, "mb": round(pf.stat().st_size / 1e6, 1),
            "examples_per_moment": round(used_examples / max(len(moments), 1), 1),
            "seconds": round(time.time() - t0, 1)}


def _tag_video(self, video_id: str, redo: bool = False, out_path: str = "", lora_path: str = "") -> dict:
    """GPU: feed the prepared prompts.json through the teacher. Nothing is read but that one file.
    out_path: write the result there instead (a 🧪 test run) — the batch results stay untouched."""
    import trial_logic as TL
    from vllm import SamplingParams

    key = self.key
    if self.llm is None:
        return {"video_id": video_id, "teacher": key, "error": f"the {key} model didn't load — {self.load_error}"}
    pack_vol.reload()
    out = Path(out_path) if out_path else tags_file(video_id, key)
    if not out_path and not redo and _tags_done(video_id, key):
        return {"video_id": video_id, "teacher": key, "skipped": True}
    pf = prompts_file(video_id)
    data = _read_json(pf)
    if not data:
        return {"video_id": video_id, "teacher": key, "error": "no prompts.json — the prep step didn't run"}
    if out.exists() and not out_path:   # a --redo: keep the earlier run so --stage compare can set them side by side
        out.replace(out.with_name(f"tags_{key}_before.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    moments, copies, vocab_names = data["moments"], data.get("copies") or {}, data.get("vocab") or []
    # the exact vocabulary prep used (rules, thresholds, order) — so a run is consistent with itself
    full = (TL.parse_vocabulary(data["vocab_csv"]) if data.get("vocab_csv") else []) or _vocab()
    vocab = [r for r in TL.teacher_vocab(full) if r["tag"] in set(vocab_names)] or TL.teacher_vocab(full)
    convs = [m["conv"] for m in moments]
    # Chunks the size of vLLM's batch (max_num_seqs), so progress shows every batch at no cost.
    # Replies are held to your tag list and category order (vLLM structured output): no typos or
    # invented tags, required categories always answered. Older vLLM without it → plain replies.
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:
        StructuredOutputsParams = None

    def sampling(max_tokens, schema):
        if StructuredOutputsParams is None:
            return SamplingParams(temperature=0.0, max_tokens=max_tokens)
        return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True))

    outs, chunk, params = [], 32, sampling(TAG_MAX_TOKENS, TL.answer_schema(vocab))
    lora_kw = {}
    if lora_path:                         # 🧠 your fine-tuned adapter on top of the teacher
        from vllm.lora.request import LoRARequest
        lora_kw = {"lora_request": LoRARequest(Path(lora_path).name, 1, lora_path)}
    print(f"[{key} teacher] {video_id[:40]}: {len(convs)} moments to tag"
          + (f" (+{len(copies)} near-identical ones copy an answer)" if copies else "")
          + (f", ~{data['examples_used'] / max(len(convs), 1):.1f} worked examples each" if data.get("examples_used") else "")
          + f" — progress every {chunk}…", flush=True)
    # Second look (usually_with): straight after each chunk, so its frames are still in vLLM's
    # prefix cache — the follow-up only pays for the new question and the short answer.
    answers, look_p_tok, look_o_tok, looks = [], 0, 0, 0
    for i in range(0, len(convs), chunk):
        got = self.llm.chat(convs[i:i + chunk], params, use_tqdm=False, **lora_kw)
        outs += got
        firsts = [TL.parse_answer(o.outputs[0].text, vocab) for o in got]
        asks = [TL.look_again(a, vocab) if a["ok"] else [] for a in firsts]
        follow = [(j, convs[i + j] + [{"role": "assistant", "content": got[j].outputs[0].text},
                                      {"role": "user", "content": TL.look_again_prompt(asks[j])}])
                  for j in range(len(got)) if asks[j]]
        if follow:
            seconds_out = self.llm.chat([c for _, c in follow],
                                        [sampling(160, TL.look_again_schema(asks[j])) for j, _ in follow],
                                        use_tqdm=False, **lora_kw)
            looks += len(follow)
            for (j, _), o2 in zip(follow, seconds_out):
                firsts[j] = TL.apply_look_again(firsts[j], o2.outputs[0].text, asks[j], vocab)
                look_p_tok += len(getattr(o2, "prompt_token_ids", None) or [])
                look_o_tok += len(getattr(o2.outputs[0], "token_ids", None) or [])
        answers += firsts
        el = time.time() - started
        left = el / len(outs) * (len(convs) - len(outs))
        print(f"[{key} teacher] {video_id[:40]}: {len(outs)}/{len(convs)} moments tagged · "
              f"{el / 60:.1f} min so far, about {left / 60:.1f} min left", flush=True)
    seconds = time.time() - started
    results, bad, p_tok, o_tok, rule_fixes, rule_adds, rule_drops = {}, 0, look_p_tok, look_o_tok, 0, 0, 0
    look_added = look_removed = required = 0
    for m, o, ans in zip(moments, outs, answers):
        bad += not ans["ok"]
        look_added += len(ans.get("look_added") or [])
        look_removed += len(ans.get("look_removed") or [])
        required += len(ans.get("rule_required") or [])
        rule_fixes += len(ans.get("rule_breaks") or [])
        rule_adds += len(ans.get("rule_adds") or [])
        rule_drops += len(ans.get("rule_drops") or [])
        p_tok += len(getattr(o, "prompt_token_ids", None) or [])
        o_tok += len(getattr(o.outputs[0], "token_ids", None) or [])
        results[m["key"]] = {**ans, "t": m["t"], "scene_id": m.get("scene_id"),
                             "gold_id": m.get("gold_id"), "raw": o.outputs[0].text[:400]}
    for k, c in copies.items():
        src = results.get(c["of"])
        if src:
            results[k] = {**{f: src.get(f) for f in ("tags", "unsure", "caption", "ok", "conf")}, "t": c["t"],
                          "scene_id": c.get("scene_id"), "gold_id": c.get("gold_id"),
                          "copied_from": c["of"], "similarity": c.get("similarity")}
    gpu = _gpu_name(TEACHER_GPUS[key][0])
    use = TL.gpu_use(key, gpu, p_tok + o_tok, seconds)
    load = self.load_seconds if self.first else 0.0
    self.first = False
    out.write_text(json.dumps({"video_id": video_id, "teacher": key, "model": TEACHERS[key], "gpu": gpu,
                               "partial": bool(data.get("partial")), "targets": data["targets"],
                               "seconds": round(seconds, 1), "load_seconds": round(load, 1),
                               "prompts": len(convs), "copied": len(copies), "unreadable": bad,
                               "rule_fixes": rule_fixes, "rule_adds": rule_adds, "rule_drops": rule_drops,
                               "looks": looks, "look_added": look_added, "look_removed": look_removed,
                               "required_fills": required, "structured": StructuredOutputsParams is not None,
                               "prompt_tokens": p_tok, "output_tokens": o_tok,
                               "tokens_per_s": use["tokens_per_s"], "mfu": use["mfu"],
                               "moments_per_min": round(60 * len(convs) / max(seconds, 1e-6), 1),
                               "examples": data.get("examples", 0), "examples_used": data.get("examples_used", 0),
                               "dedup": data.get("dedup"), "vocab": vocab_names, "results": results,
                               **({"vocab_csv": data.get("vocab_csv") or ""} if out_path else {}),
                               **({"lora": Path(lora_path).name} if lora_path else {})}),
                   encoding="utf-8")
    pf.unlink(missing_ok=True)          # it's rebuilt by the prep step whenever needed
    pack_vol.commit()
    if rule_fixes or rule_adds or rule_drops:
        print(f"[{key} teacher] {video_id[:40]}: rules — {rule_fixes} tag(s) broke a never-with → unsure, "
              f"{rule_adds} added by always-with, {rule_drops} dropped by only-with → unsure.", flush=True)
    if looks:
        print(f"[{key} teacher] {video_id[:40]}: second look on {looks} moment(s) → {look_added} tag(s) "
              f"added (usually-with), {look_removed} taken out (double-check).", flush=True)
    print(f"[{key} teacher] {video_id[:40]}: {len(convs)} moments in {seconds / 60:.1f} min · "
          f"{use['tokens_per_s'] or 0:,} tokens/s · GPU maths used {100 * (use['mfu'] or 0):.0f}%", flush=True)
    return {"video_id": video_id, "teacher": key, "prompts": len(convs), "copied": len(copies),
            "seconds": round(seconds, 1), "unreadable": bad, "tokens_per_s": use["tokens_per_s"], "mfu": use["mfu"]}


@app.cls(image=tag_image, gpu=TEACHER_GPUS["8b"], volumes={DATA: pack_vol, CACHE: cache_vol},
         timeout=4 * 3600, cpu=8, memory=65536, max_containers=1)
class Tagger8B:
    @modal.enter()
    def load(self):
        _load_teacher(self, "8b")

    @modal.method()
    def tag_video(self, video_id: str, redo: bool = False, out_path: str = "") -> dict:
        return _tag_video(self, video_id, redo, out_path)


@app.cls(image=tag_image, gpu=TEACHER_GPUS["32b"], volumes={DATA: pack_vol, CACHE: cache_vol},
         timeout=4 * 3600, cpu=8, memory=98304, max_containers=1)
class Tagger32B:
    @modal.enter()
    def load(self):
        _load_teacher(self, "32b")

    @modal.method()
    def tag_video(self, video_id: str, redo: bool = False, out_path: str = "") -> dict:
        return _tag_video(self, video_id, redo, out_path)


@app.cls(image=tag_image, gpu=TEACHER_GPUS["32b-fp8"], volumes={DATA: pack_vol, CACHE: cache_vol},
         timeout=4 * 3600, cpu=8, memory=65536, max_containers=1)
class Tagger32BFP8:
    @modal.enter()
    def load(self):
        _load_teacher(self, "32b-fp8")

    @modal.method()
    def tag_video(self, video_id: str, redo: bool = False, out_path: str = "") -> dict:
        return _tag_video(self, video_id, redo, out_path)


@app.cls(image=tag_image, gpu=TEACHER_GPUS["32b-fp8"], volumes={DATA: pack_vol, CACHE: cache_vol},
         timeout=4 * 3600, cpu=8, memory=65536, max_containers=1,
         scaledown_window=300)          # stays warm between a check's plain and LoRA runs (one model load)
class Tagger32BFP8LoRA:
    """The FP8 teacher with LoRA switched on — tags with one of your trained adapters."""
    @modal.enter()
    def load(self):
        _load_teacher(self, "32b-fp8", lora=True)

    @modal.method()
    def tag_video(self, video_id: str, redo: bool = False, out_path: str = "", lora_path: str = "") -> dict:
        return _tag_video(self, video_id, redo, out_path, lora_path)


TAGGERS = {"8b": Tagger8B, "32b": Tagger32B, "32b-fp8": Tagger32BFP8}


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
    if any(k != "8b" for k in keys):    # the 8B was dropped after batch 1 — only when nothing else
        keys = [k for k in keys if k != "8b"]

    # --- Tags: each teacher against your labels — 🎯 cards AND ✏️ corrections, matched to the
    # teacher's moments by video and time — on every labelled frame IT tagged. The old tagger is
    # scored on the same frames as the teacher in use, from the scene's tags in the manifest.
    gold_tags = [i for i in gold if i.get("type") == "tags" and not i.get("skipped")]

    def old_tags(i):
        sc = next((x for x in (mans.get(i.get("video_id")) or {}).get("scenes") or []
                   if x.get("scene_id") == i.get("scene_id")), None)
        if sc is None:
            return list(i.get("suggested") or [])
        return list(sc.get("corrected_tags") if sc.get("corrected_tags") is not None else sc.get("tags") or [])

    old_all = TL.score_tags([(old_tags(i), i.get("tags") or []) for i in gold_tags], tags)
    by_gold = {}             # {teacher: {label id: the teacher's result for that frame}}
    for key in keys:
        idx, ids = {}, {}
        for v, tj in tagged_by[key].items():
            for r in tj["results"].values():
                if r.get("t") is not None:
                    idx[(v, round(float(r["t"]), 2))] = r
                if r.get("gold_id"):
                    ids[r["gold_id"]] = r
        hit = {}
        for i in gold_tags:
            r = ids.get(i["id"]) or (idx.get((i.get("video_id"), round(float(i["t"]), 2)))
                                     if i.get("t") is not None else None)
            if r is not None:
                hit[i["id"]] = r
        by_gold[key] = hit
    frames_of = {k: [i for i in gold_tags if i["id"] in by_gold[k]] for k in keys}
    scores = {k: TL.score_tags([(by_gold[k][i["id"]]["tags"], i.get("tags") or []) for i in frames_of[k]], tags)
              for k in keys}
    # The teacher the verdicts (and the library estimate) go with: the one in use, else the best.
    best = (TEACHERS_DEFAULT if TEACHERS_DEFAULT in keys
            else max(keys, key=lambda k: scores[k]["overall"]["f1"] or 0.0) if keys else None)
    trial_gold = frames_of.get(best, [])
    old_trial = TL.score_tags([(old_tags(i), i.get("tags") or []) for i in trial_gold], tags)
    teacher = scores[best] if best else TL.score_tags([], tags)
    tagged = tagged_by.get(best, {})
    positives = {t: sum(t in (i.get("tags") or []) for i in gold_tags) for t in tags}
    kinds = {r["tag"]: r["kind"] for r in vocab}

    teacher_on = {r["tag"]: r.get("teacher", True) for r in vocab}

    def verdict(t):
        if not teacher_on.get(t, True):
            return "teacher off — learned from your labels"
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
    # Vision is costed from the videos run with the newest pipeline (batched pose + tracking on
    # CPU) once there are any — the older ones would make the library estimate look dearer.
    newest = max((d.get("pipeline", 1) for d in vis.values()), default=1)
    vis_cost = {v: d for v, d in vis.items() if d.get("pipeline", 1) == newest}
    v_fixed = [d["timings"].get("load_models", 0.0) for d in vis_cost.values()]
    stage_sec = {"vision": sum(d["timings"]["total"] for d in vis_cost.values()) - sum(v_fixed)}
    fixed = {"vision": (sum(v_fixed) / max(len(v_fixed), 1), lib_videos)}
    vis_hours = sum(d["duration"] for d in vis_cost.values()) / 3600
    stage_hours = {"vision": vis_hours}         # each teacher per hour of the videos IT tagged
    cpu_sec = sum((d.get("cpu_timings") or {}).get("total", 0.0) for d in vis_cost.values())
    if cpu_sec:
        stage_sec["vision CPU"] = cpu_sec
        stage_hours["vision CPU"] = vis_hours
    for k in keys:
        stage_sec[f"tag {k}"] = sum(t["seconds"] for t in tagged_by[k].values())
        fixed[f"tag {k}"] = (sum(t.get("load_seconds", 0) for t in tagged_by[k].values()), 1)
        stage_hours[f"tag {k}"] = sum(vis[v]["duration"] for v in tagged_by[k] if v in vis) / 3600

    def used(rows, default):        # the GPU most of the work actually ran on
        names = [r.get("gpu") or default for r in rows]
        return max(set(names), key=names.count) if names else default

    gpus = {"vision": used(vis_cost.values(), VISION_GPU), "vision CPU": "CPU"}
    gpus.update({f"tag {k}": used(tagged_by[k].values(), TEACHER_GPUS[k][0]) for k in keys})
    cost = TL.extrapolate(stage_sec, gpus, stage_hours, lib_hours, fixed)
    # The library only needs one teacher: the estimate counts vision + the one in use (the
    # default when it has run, else the better scorer).
    lib_teacher = TEACHERS_DEFAULT if TEACHERS_DEFAULT in keys else best
    lib_total = (cost["vision"]["library_$"] + cost.get("vision CPU", {}).get("library_$", 0.0)
                 + (cost[f"tag {lib_teacher}"]["library_$"] if lib_teacher else 0.0))

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
    # Where the vision time goes, in seconds per hour of footage (newest pipeline's videos).
    pose_label = "Pose (GPU, batched)" if newest >= 2 else "Pose + tracking (GPU)"
    steps = {"Scene cuts (TransNetV2)": "shots", "Loading models": "load_models",
             pose_label: "pose_gpu", "Embeddings (GPU)": "embed_gpu"}
    tot = {k: sum(d["timings"].get(v, 0.0) for d in vis_cost.values()) for k, v in steps.items()}
    dpe = sum(d["timings"].get("decode_pose_embed", 0.0) for d in vis_cost.values())
    rest = "Decoding, saving frames (GPU machine)" if newest >= 2 else "Decoding, tracker, saving frames (CPU)"
    tot[rest] = max(dpe - tot[pose_label] - tot["Embeddings (GPU)"], 0.0)
    tot["Other"] = max(sum(d["timings"].get("total", 0.0) for d in vis_cost.values())
                       - dpe - tot["Scene cuts (TransNetV2)"] - tot["Loading models"], 0.0)
    vision_breakdown = {k: round(v / vis_hours) if vis_hours else 0 for k, v in tot.items()}
    md_steps = ["", f"Where the vision GPU time goes (seconds per hour of footage, {len(vis_cost)} video(s) "
                f"on the {'batched' if newest >= 2 else 'original'} pipeline):", "",
                "| Step | Seconds | Share |", "|---|---|---|"]
    all_s = sum(vision_breakdown.values()) or 1
    md_steps += [f"| {k} | {v} | {100 * v / all_s:.0f}% |" for k, v in vision_breakdown.items()]
    if cpu_sec and vis_hours:
        cs = {k: sum((d.get("cpu_timings") or {}).get(k, 0.0) for d in vis_cost.values()) / vis_hours
              for k in ("prep", "track", "maths")}
        md_steps += ["", f"On cheap CPU machines (not GPU time): scene-cut frames {cs['prep']:.0f} s, "
                         f"person tracking {cs['track']:.0f} s, movement maths {cs['maths']:.0f} s per hour of footage."]
    md += [f"| **total** | | **${cost['total']['trial_$']:.2f}** | | "
           f"**${lib_total:.2f}**{' (vision + ' + lib_teacher + ' only)' if len(keys) > 1 else ''} | |", "",
           *md_steps, "",
           "## Tags", "",
           f"Your labels: {len(gold_tags)} frames (🎯 cards and ✏️ corrections). Each teacher is scored on "
           "every one of them it tagged: " + ", ".join(f"{k} {len(frames_of[k])} frames" for k in keys) + ".",
           "Overall F1 (1 = matches you exactly): "
           + ", ".join(f"{k} **{f(scores[k]['overall']['f1'])}**" for k in keys)
           + (f"; old tagger on {best}'s frames **{f(old_trial['overall']['f1'])}**" if best else "")
           + f", on all your frames {f(old_all['overall']['f1'])}.", ""]
    if len(keys) > 1:
        md.append(f"**Teacher in use: {best}** — the suggestions below use it. The teachers tagged different "
                  "videos, so compare their F1 columns with care (only the FP8 check video has both).")
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
        fixes = sum(t.get("rule_fixes", 0) for t in tagged_by[k].values())
        adds = sum(t.get("rule_adds", 0) for t in tagged_by[k].values())
        if fixes:
            md.append(f"{k}: {fixes} tag(s) moved to unsure because they broke one of your never-with rules.")
        if adds:
            md.append(f"{k}: {adds} tag(s) added by your always-with rules.")
        drops = sum(t.get("rule_drops", 0) for t in tagged_by[k].values())
        if drops:
            md.append(f"{k}: {drops} tag(s) moved to unsure because the tag your only-with rule needs wasn't there.")
        looks = sum(t.get("looks", 0) for t in tagged_by[k].values())
        if looks:
            added = sum(t.get("look_added", 0) for t in tagged_by[k].values())
            removed = sum(t.get("look_removed", 0) for t in tagged_by[k].values())
            md.append(f"{k}: took a second look at {looks} moment(s) — {added} tag(s) added by your usually-with "
                      f"rules, {removed} taken out by double-check.")
        fills = sum(t.get("required_fills", 0) for t in tagged_by[k].values())
        if fills:
            md.append(f"{k}: {fills} tag(s) were below their threshold but taken because their category needs a tag.")
    # How hard the tagging GPU worked (only runs that recorded tokens — from v1.32 on).
    speed = []
    for k in keys:
        runs = [t for t in tagged_by[k].values() if t.get("prompt_tokens")]
        if not runs:
            continue
        sec = sum(t.get("seconds", 0.0) for t in runs)
        tok = sum(t.get("prompt_tokens", 0) + t.get("output_tokens", 0) for t in runs)
        sent = sum(t.get("prompts", 0) for t in runs)
        copied = sum(t.get("copied", 0) for t in runs)
        gpu = runs[0].get("gpu") or TEACHER_GPUS[k][0]
        use = TL.gpu_use(k, gpu, tok, sec)
        speed.append(f"| {k} | {gpu} | {len(runs)} | {sent} | {copied} | {60 * sent / max(sec, 1e-6):.0f} | "
                     f"{(use['tokens_per_s'] or 0):,} | {f(use['mfu'])} |")
    if speed:
        md += ["", "How hard the tagging GPU worked (1 MFU = flat out; under 0.3 = mostly waiting):", "",
               "| Teacher | GPU | Videos | Moments sent | Copied (near-identical) | Moments / min | Tokens / s | MFU |",
               "|---|---|---|---|---|---|---|---|", *speed]
    # Did the worked examples help? Your labelled frames split by whether their video was tagged
    # with examples (so this batch's videos vs earlier ones).
    if best and trial_gold:
        with_ex = {v for v, tj in tagged.items() if tj.get("examples_used")}
        split = {"with worked examples": [i for i in trial_gold if i.get("video_id") in with_ex],
                 "without": [i for i in trial_gold if i.get("video_id") not in with_ex]}
        if all(split.values()):
            md += ["", f"**Worked examples ({best}):** overall F1 on your labelled frames — "
                   + ", ".join(f"{name} **{f(TL.score_tags([(by_gold[best][i['id']]['tags'], i.get('tags') or []) for i in items], tags)['overall']['f1'])}**"
                               f" ({len(items)} frames, {len({i['video_id'] for i in items})} videos)"
                               for name, items in split.items())
                   + ". Different videos, so treat a small gap with caution."]
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


def _export_slots(found: dict) -> dict:
    """What the app shows as "the teacher" for one video: the 8B only when nothing better exists;
    32b and 32b-fp8 are one teacher at two precisions, so an FP8-only video shows under "32b" —
    both names only where a video has both (the FP8 check, side by side)."""
    if set(found) == {"8b"}:
        return dict(found)
    out = {}
    if "32b" in found:
        out["32b"] = found["32b"]
    if "32b-fp8" in found:
        out["32b-fp8" if "32b" in found else "32b"] = found["32b-fp8"]
    return out


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
    batches = _read_json(f"{PACK}/batches.json", {}) or {}
    batch_of = {v: str(b) for b, ids in batches.items() for v in ids}
    if not batches:                     # packs from before batches existed are batch 1
        batch_of = {v: "1" for v in done}
    videos, n_frames, models = {}, 0, {}
    for v in done:
        vis = _read_json(f"{OUT}/{v}/vision.json")
        man = _read_json(f"{PACK}/manifests/{v}.json") or {"scenes": []}
        found = {}
        for key in TEACHERS:
            tf = tags_file(v, key)
            tj = _read_json(tf) or (_read_json(f"{OUT}/{v}/tags.json") if key == "8b" else None)
            if tj:
                found[key] = (tj, tf)
        teachers, tag_runs = {}, {}
        for slot, (tj, tf) in _export_slots(found).items():
            teachers[slot] = tj["results"]
            tag_runs[slot] = {"seconds": tj.get("seconds"), "load_seconds": tj.get("load_seconds"),
                              "gpu": tj.get("gpu"), "prompts": tj.get("prompts"), "copied": tj.get("copied"),
                              "unreadable": tj.get("unreadable"), "model": tj.get("model"),
                              "examples_used": tj.get("examples_used"), "dedup": tj.get("dedup"),
                              "tokens_per_s": tj.get("tokens_per_s"), "mfu": tj.get("mfu"),
                              "at": tf.stat().st_mtime if tf.exists() else None}
            models.setdefault(slot, set()).add((tj.get("model") or TEACHERS.get(slot, slot)).split("/")[-1])
        vf = Path(f"{OUT}/{v}/vision.json")
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
                "teachers": {k: {f: r[tg["key"]].get(f) for f in ("tags", "unsure", "caption", "copied_from", "look_added",
                                                               "look_removed", "conf")}
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
            "batch": batch_of.get(v),
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
            # what each step took — for the per-batch cost / time report in the app
            "run": {"vision_seconds": (vis.get("timings") or {}).get("total"),
                    "vision_timings": vis.get("timings") or {}, "vision_gpu": vis.get("gpu"),
                    "vision_at": vf.stat().st_mtime if vf.exists() else None,
                    "pose_model": (vis.get("models") or {}).get("pose"),
                    "tagging": tag_runs},
            "moments": moments,
        }
    report_json = _read_json(f"{OUT}/report.json", {}) or {}
    results = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M"), "smoke": smoke,
               "teachers": {k: ", ".join(sorted(v)) for k, v in models.items()},
               "report": report_json, "videos": videos}
    Path(f"{EXPORT}/results.json").write_text(json.dumps(results), encoding="utf-8")
    if Path(f"{OUT}/testset/test_runs.json").exists():         # 🧪 test runs travel with the export
        shutil.copyfile(f"{OUT}/testset/test_runs.json", f"{EXPORT}/test_runs.json")
    if Path(f"{OUT}/checks/check_runs.json").exists():         # 🔍 and the label-check runs
        shutil.copyfile(f"{OUT}/checks/check_runs.json", f"{EXPORT}/check_runs.json")
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


def _download_test_runs(src: str = "/trial/testset/test_runs.json", name: str = "test_runs.json") -> str:
    """Laptop / server side: fetch testset/test_runs.json (or the check runs) into the catalogue's
    trial_results folder."""
    import subprocess
    import sys
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from config import CATALOGUE_DIR
    except Exception as e:
        return f"couldn't read CATALOGUE_DIR from config.py ({e}) — results stay on Modal"
    dest = Path(CATALOGUE_DIR) / "trial_results"
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", "--force", "media-train", src, str(dest / name)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"download failed: {(r.stderr or r.stdout).strip()[-400:]}"
    return str(dest / name)


@app.function(image=report_image, volumes={DATA: pack_vol})
def labelled_videos(exclude: list) -> list:
    """Finished videos with labelled frames (🎯 tags), minus `exclude` (the test set)."""
    pack_vol.reload()
    vids = {i["video_id"] for i in _gold_items() if i.get("type") == "tags" and not i.get("skipped")}
    return sorted(v for v in vids if v not in set(exclude) and _vision_done(v))


def _run_check(lora: str, dedup: float) -> None:
    """--stage check-labels: tag ONLY your labelled frames in the training videos (the test set is
    left out), once with the plain teacher and once with the adapter — one GPU container, model
    loaded once — so 🔍 Check my label can list the training frames both disagree with you on."""
    _push_labels()
    ts = pick_test_set.remote(TEST_HOURS, False)
    vids = labelled_videos.remote([x["video_id"] for x in ts.get("videos") or []])
    if not vids:
        print("No labelled frames outside the test set.")
        return
    print(f"🔍 Label check: your labelled frames in {len(vids)} training videos, plain teacher + LoRA '{lora}'.")
    tagger = Tagger32BFP8LoRA()
    stamp = time.strftime("%Y-%m-%d_%H%M")
    texts = []
    for label, lpath in (("check-plain", ""), (f"check-lora-{lora}", f"{OUT}/lora/{lora}")):
        t0 = time.time()
        res = list(prepare_prompts.starmap([(v, 0, 0, 0, True) for v in vids], return_exceptions=True))
        n = sum(r.get("prompts", 0) for r in res if isinstance(r, dict))
        print(f"   {label}: {n} labelled frames to tag…")
        run_id = f"{stamp}_{label}"
        out = f"{OUT}/checks/runs/{run_id}"
        args = [(v, True, f"{out}/{v}.json", lpath) for v in vids]
        bad = [r for r in tagger.tag_video.starmap(args, return_exceptions=True)
               if not isinstance(r, dict) or r.get("error")]
        for r in bad[:5]:
            print("  ", r)
        meta = {"id": run_id, "label": label, "teacher": "32b-fp8", "model": TEACHERS["32b-fp8"],
                "lora": lora if lpath else "", "created": time.strftime("%Y-%m-%d %H:%M"), "kind": "check",
                "minutes": round((time.time() - t0) / 60, 1)}
        texts.append(finish_test_run.remote(run_id, meta, "check"))
    print("\n" + texts[-1])
    print(f"   Check runs → {_download_test_runs('/trial/checks/check_runs.json', 'check_runs.json')}\n"
          "   In the app: 🧪 Trial results → 🖼️ Compare tags → Show: 🔍 Check my label.")


def _run_test(keys: list, label: str, keep_vocab: bool, dedup: float, lora: str = "") -> None:
    """--stage test: re-tag the fixed test set with today's settings (or, with --keep-vocab, the
    vocabulary already on Modal = what the last batch used) and compare with earlier test runs."""
    _push_labels(vocab=not keep_vocab)
    ts = pick_test_set.remote(TEST_HOURS, False)
    vids = [x["video_id"] for x in ts.get("videos") or []]
    if not vids:
        print("No test set: no finished video has enough of your labelled frames yet.")
        return
    k = "32b-fp8" if lora else keys[0]
    label = label or (f"lora-{lora}" if lora else "")
    run_id = time.strftime("%Y-%m-%d_%H%M") + (f"_{label}" if label else "")
    print(f"🧪 Test run {run_id}: {len(vids)} videos, {ts['hours']} h of footage, {ts['labelled']} of your "
          f"labelled frames, teacher {k}" + (" — using the vocabulary already on Modal (the last batch's)"
                                             if keep_vocab else "") + ".")
    t0 = time.time()
    for r in prepare_prompts.starmap([(v, 0, 0, dedup) for v in vids], return_exceptions=True):
        print("  ", r)
    tagger = Tagger32BFP8LoRA() if lora else TAGGERS[k]()
    out = f"{OUT}/testset/runs/{run_id}"
    args = [(v, True, f"{out}/{v}.json") + ((f"{OUT}/lora/{lora}",) if lora else ()) for v in vids]
    for r in tagger.tag_video.starmap(args, return_exceptions=True):
        print("  ", r)
    meta = {"id": run_id, "label": label, "teacher": k, "model": TEACHERS[k], "lora": lora,
            "created": time.strftime("%Y-%m-%d %H:%M"), "keep_vocab": keep_vocab, "dedup": dedup,
            "minutes": round((time.time() - t0) / 60, 1)}
    text = finish_test_run.remote(run_id, meta)
    Path("test_report.md").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n\nSaved {Path('test_report.md').resolve()}")
    print(f"   Test runs → {_download_test_runs()}\n   In the app: 🧪 Trial results → 📋 Summary → 🧪 Test set.")


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


def _match_labels(video_id: str, results: dict, gold_tags: list) -> list:
    """[(teacher tags, your tags)] for your labelled frames in this video that the run tagged —
    🎯 cards by their id, ✏️ corrections by time."""
    by_t = {round(float(r["t"]), 2): r for r in results.values() if r.get("t") is not None}
    by_id = {r["gold_id"]: r for r in results.values() if r.get("gold_id")}
    pairs = []
    for i in gold_tags:
        if i.get("video_id") != video_id:
            continue
        r = by_id.get(i["id"]) or (by_t.get(round(float(i["t"]), 2)) if i.get("t") is not None else None)
        if r is not None:
            pairs.append((r.get("tags") or [], i.get("tags") or []))
    return pairs


@app.function(image=report_image, volumes={DATA: pack_vol})
def compare_runs(video_ids: list, key: str) -> str:
    """Before vs after a --redo of these videos by this teacher (tags_<key>_before.json vs
    tags_<key>.json): accuracy on your labelled frames, speed and cost, as a small table."""
    import trial_logic as TL
    pack_vol.reload()
    tags = [r["tag"] for r in _vocab()]
    gold_tags = [i for i in _gold_items() if i.get("type") == "tags" and not i.get("skipped")]
    runs = {"before": [], "after": []}
    for v in video_ids:
        for name, fn in (("before", f"tags_{key}_before.json"), ("after", f"tags_{key}.json")):
            tj = _read_json(f"{OUT}/{v}/{fn}")
            if tj:
                runs[name].append((v, tj))
    common = {v for v, _ in runs["before"]} & {v for v, _ in runs["after"]}
    if not common:
        return ("Nothing to compare: no video has both an earlier run (tags_" + key + "_before.json) and a "
                "new one. Run the --redo first.")
    rows = []
    for name in ("before", "after"):
        pairs, sec, moments, ex, gpu = [], 0.0, 0, 0, None
        for v, tj in runs[name]:
            if v not in common:
                continue
            pairs += _match_labels(v, tj.get("results") or {}, gold_tags)
            sec += float(tj.get("seconds") or 0.0)
            moments += int(tj.get("prompts") or 0)
            ex += int(tj.get("examples_used") or 0)
            gpu = gpu or tj.get("gpu")
        sc = TL.score_tags(pairs, tags)["overall"]
        cost = TL.stage_cost(sec, gpu or "H100")
        rows.append((name, len(pairs), sc, sec, moments, ex, cost))

    def f(x):
        return "–" if x is None else f"{x:.2f}"
    md = [f"Before vs after on {len(common)} video(s), teacher {key}, scored on your labelled frames "
          "(🎯 cards and ✏️ corrections) in those videos:", "",
          "| Run | Your frames | F1 | Precision | Recall | Worked examples / moment | Moments sent | Tagging min | Tagging $ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, n, sc, sec, moments, ex, cost in rows:
        md.append(f"| {name} | {n} | {f(sc['f1'])} | {f(sc['precision'])} | {f(sc['recall'])} | "
                  f"{ex / max(moments, 1):.1f} | {moments} | {sec / 60:.1f} | ${cost:.2f} |")
    md += ["", "Your corrections in these videos started from the BEFORE run's tags, which can flatter it "
               "slightly. A gain of a few points of F1 is noise on under ~100 frames."]
    return "\n".join(md)


# ---------------------------------------------------------------------------
# 🧪 Test set — the same videos re-tagged after each settings change
# ---------------------------------------------------------------------------

def _test_dir() -> Path:
    return Path(f"{OUT}/testset")


@app.function(image=report_image, volumes={DATA: pack_vol})
def pick_test_set(hours: float = TEST_HOURS, redo: bool = False) -> dict:
    """The fixed test set (testset/testset.json), picked once: finished videos with plenty of your
    labelled frames, taken in turn from each batch (most labelled frames per minute first) until
    about `hours` of footage. These videos are held out of any future teacher fine-tune."""
    import trial_logic as TL
    pack_vol.reload()
    f = _test_dir() / "testset.json"
    old = _read_json(f)
    if old and not redo:
        return old
    gold = [i for i in _gold_items() if i.get("type") == "tags" and not i.get("skipped")]
    batches = _read_json(f"{PACK}/batches.json", {}) or {}
    batch_of = {v: str(b) for b, ids in batches.items() for v in ids}
    cands = {}
    for p in Path(OUT).iterdir():
        v = p.name
        if p.name in ("testset", "export") or not _vision_done(v):
            continue
        vis = _read_json(p / "vision.json")
        best = 0
        for k in TEACHERS:
            tj = _read_json(p / f"tags_{k}.json")
            if tj:
                best = max(best, len(_match_labels(v, tj.get("results") or {}, gold)))
        if best >= TEST_MIN_LABELLED and vis.get("duration"):
            cands[v] = {"minutes": vis["duration"] / 60, "labelled": best, "batch": batch_of.get(v, "?")}
    by_batch = {}
    for v, c in sorted(cands.items(), key=lambda kv: -kv[1]["labelled"] / kv[1]["minutes"]):
        by_batch.setdefault(c["batch"], []).append(v)
    picked, total = [], 0.0
    limit = hours * 60
    while any(by_batch.values()) and total < limit * 0.9:
        for b in sorted(by_batch, key=lambda b: (len(b), b)):
            for v in list(by_batch[b]):
                by_batch[b].remove(v)
                if total + cands[v]["minutes"] <= limit * 1.1:
                    picked.append(v)
                    total += cands[v]["minutes"]
                    break
            if total >= limit * 0.9:
                break
    out = {"created": time.strftime("%Y-%m-%d %H:%M"), "hours": round(total / 60, 2),
           "labelled": sum(cands[v]["labelled"] for v in picked),
           "videos": [{"video_id": v, **{k: (round(c, 1) if k == "minutes" else c) for k, c in cands[v].items()}}
                      for v in picked], "holdout": True}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    pack_vol.commit()
    return out


def _run_summary(run_dir: Path) -> dict | None:
    meta = _read_json(run_dir / "run.json")
    if not meta:
        return None
    frames, sec, moments, gpu, vocab_csv = [], 0.0, 0, None, ""
    counts = {"looks": 0, "look_added": 0, "look_removed": 0, "required_fills": 0, "unreadable": 0}
    for f in sorted(run_dir.glob("*.json")):
        if f.name == "run.json":
            continue
        tj = _read_json(f) or {}
        v = f.stem
        sec += float(tj.get("seconds") or 0.0)
        moments += int(tj.get("prompts") or 0)
        gpu = gpu or tj.get("gpu")
        for k in counts:
            counts[k] += int(tj.get(k) or 0)
        vocab_csv = vocab_csv or tj.get("vocab_csv") or ""
        frames += [{"v": v, "t": r.get("t"), "g": r.get("gold_id"), "tags": r.get("tags") or [],
                    "unsure": r.get("unsure") or []} for r in (tj.get("results") or {}).values()]
    import trial_logic as TL
    return {**meta, "seconds": round(sec, 1), "moments": moments, "gpu": gpu,
            "cost": round(TL.stage_cost(sec, gpu or "H100"), 2), **counts, "vocab_csv": vocab_csv,
            "videos": len([f for f in run_dir.glob("*.json") if f.name != "run.json"]), "frames": frames}


@app.function(image=report_image, volumes={DATA: pack_vol}, timeout=1800)
def finish_test_run(run_id: str, meta: dict, kind: str = "test") -> str:
    """After a test run (kind "test") or a 🔍 label-check run ("check"): save its details, rebuild
    testset/test_runs.json or checks/check_runs.json (every run, for the app) and return the table
    of runs scored on your latest labels, plus the per-tag changes since the run before."""
    import trial_logic as TL
    pack_vol.reload()
    d = _test_dir() if kind == "test" else Path(f"{OUT}/checks")
    (d / "runs" / run_id / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    runs = [r for r in (_run_summary(p) for p in sorted((d / "runs").iterdir()) if p.is_dir()) if r]
    ts = _read_json(d / "testset.json", {}) or {}
    name = "test_runs.json" if kind == "test" else "check_runs.json"
    (d / name).write_text(json.dumps({"set": ts, "runs": runs}), encoding="utf-8")
    pack_vol.commit()
    gold = [i for i in _gold_items() if i.get("type") == "tags" and not i.get("skipped")]
    tags = [r["tag"] for r in _vocab()]
    return TL.test_runs_markdown(runs, gold, tags)


# ---------------------------------------------------------------------------
# 🧠 LoRA — fine-tune the teacher on your labelled frames (test-set videos held out)
# ---------------------------------------------------------------------------

def _lora_dir() -> Path:
    return Path(f"{OUT}/lora")


@app.function(image=report_image, volumes={DATA: pack_vol}, timeout=1800, cpu=2)
def prepare_lora_data(val_share: float = 0.05, seed: int = 7) -> dict:
    """CPU: one training example per labelled frame whose 7 frames are on the Volume, in exactly the
    prompt the teacher sees (today's vocabulary), answered with YOUR tags (lora/data.json). The test
    set's videos are left out; a few other whole videos are kept back to watch the loss (val)."""
    import random
    import trial_logic as TL
    pack_vol.reload()
    ts = _read_json(_test_dir() / "testset.json", {}) or {}
    held = {x["video_id"] for x in ts.get("videos") or []}
    vocab_csv = _vocab_text()
    vocab = TL.teacher_vocab((TL.parse_vocabulary(vocab_csv) if vocab_csv else []) or _vocab())
    head, moment, tail = TL.build_prompt(vocab)
    gold = [i for i in _gold_items() if i.get("type") == "tags" and not i.get("skipped") and i.get("t") is not None]
    recs, skipped_test, no_frames, vis_cache, cap_cache = [], 0, 0, {}, {}
    for i in gold:
        v = i["video_id"]
        if v in held:
            skipped_test += 1
            continue
        if v not in vis_cache:
            vis_cache[v] = _read_json(f"{OUT}/{v}/vision.json") or {}
            caps = {}
            for k in TEACHERS:                       # the teacher's caption for the moment, if any
                for r in ((_read_json(f"{OUT}/{v}/tags_{k}.json") or {}).get("results") or {}).values():
                    if r.get("t") is not None and r.get("caption"):
                        caps[round(float(r["t"]), 2)] = r["caption"]
            cap_cache[v] = caps
        vis = vis_cache[v]
        if not vis.get("duration"):
            no_frames += 1
            continue
        frames = [f"{OUT}/{v}/frames/{TL.frame_name(t)}" for t in TL.context_times(float(i["t"]), vis["duration"])]
        if not all(Path(f).exists() for f in frames):
            no_frames += 1
            continue
        target = TL.lora_target(i.get("tags") or [], vocab, cap_cache[v].get(round(float(i["t"]), 2), ""))
        recs.append({"video_id": v, "t": i["t"], "frames": frames, "target": target})
    vids = sorted({r["video_id"] for r in recs})
    random.Random(seed).shuffle(vids)
    val_vids = set(vids[:max(1, int(len(vids) * val_share))]) if len(vids) > 4 else set()
    data = {"created": time.strftime("%Y-%m-%d %H:%M"), "head": head, "moment": moment, "tail": tail,
            "vocab_csv": vocab_csv, "test_videos": sorted(held), "val_videos": sorted(val_vids),
            "train": [r for r in recs if r["video_id"] not in val_vids],
            "val": [r for r in recs if r["video_id"] in val_vids]}
    d = _lora_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.json").write_text(json.dumps(data), encoding="utf-8")
    pack_vol.commit()
    return {"train": len(data["train"]), "val": len(data["val"]), "videos": len(vids),
            "left_out_test_set": skipped_test, "no_frames": no_frames}


def _answer_loss_labels(model, batch: dict, starts: list):
    """(summed loss, answer tokens) like _answer_loss, but by handing the model labels (prompt and
    padding masked) — for Unsloth, whose own fused loss never builds the full logits."""
    import torch
    ids = batch["input_ids"]
    labels = ids.clone()
    pos = torch.arange(ids.shape[1], device=ids.device)
    for i, s0 in enumerate(starts):
        labels[i, pos < s0] = -100
    labels[batch["attention_mask"] == 0] = -100
    out = model(**batch, labels=labels)
    n = (labels[:, 1:] != -100).sum()
    return out.loss * n, n


def _plain_target_modules(cfg_path: Path) -> None:
    """Write the adapter's target modules as the plain list of layer names (Unsloth stores a regex);
    the weights are unchanged — it only keeps vLLM's adapter loading on its simplest path."""
    cfg = _read_json(cfg_path)
    if cfg is not None:
        cfg["target_modules"] = list(LORA_TARGET)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _lora_resume_point(progress, data_created, total: int, epochs: float, lr: float) -> int:
    """Where a training run carries on from: the checkpoint's example count, if the checkpoint is for
    these same examples and settings and not already at the end; else 0 (start over)."""
    if not progress:
        return 0
    same = (progress.get("data_created") == data_created and progress.get("total") == total
            and progress.get("epochs") == epochs and progress.get("lr") == lr)
    n = int(progress.get("n") or 0)
    return n if same and 0 < n < total else 0


@app.function(image=report_image, volumes={DATA: pack_vol})
def lora_status(name: str) -> dict:
    """{"done": finished adapter there, "checkpoint": its progress.json or None}."""
    pack_vol.reload()
    d = _lora_dir() / name
    return {"done": (d / "adapter_model.safetensors").exists() and (d / "meta.json").exists(),
            "checkpoint": _read_json(d / "checkpoint" / "progress.json")}


def _answer_loss(model, batch: dict, starts: list):
    """(summed loss, answer tokens) for a LEFT-padded batch whose answers start at `starts` (index of
    each row's first answer token). Only the answer positions are scored: the model is asked for
    the last K positions' logits instead of all ~4,500 (the vocabulary is ~150k wide, so full logits
    cost ~4 GB per example — this is what lets 8 examples fit at once)."""
    import torch
    import torch.nn.functional as F
    ids = batch["input_ids"]
    T = ids.shape[1]
    K = min(T, max(T - s for s in starts) + 1)
    out = model(**{k: v for k, v in batch.items() if k != "labels"}, logits_to_keep=K)
    logits = out.logits[:, :-1].float()                     # positions T-K .. T-2 → tokens T-K+1 .. T-1
    target = ids[:, T - K + 1:].clone()
    pos = torch.arange(T - K + 1, T, device=ids.device)
    for i, s0 in enumerate(starts):
        target[i, pos < s0] = -100
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100,
                           reduction="sum")
    return loss, (target != -100).sum()


def _lora_content(head: str, moment: str, tail: str, images: list) -> list:
    """The user turn exactly as prepare_prompts builds it for vLLM (head, moment line, 7 frames,
    tail) — in the transformers form."""
    return ([{"type": "text", "text": head}, {"type": "text", "text": moment}]
            + [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": tail}])


@app.function(image=lora_image, gpu="H100", volumes={DATA: pack_vol, CACHE: cache_vol}, timeout=6 * 3600,
              cpu=8, memory=49152)          # (was 98 GB: ~$0.45/h of unused RAM)
def train_lora(name: str, epochs: float = 1.0, lr: float = 1e-4, accum: int = 8, max_examples: int = 0,
               fresh: bool = True, micro: int = LORA_MICRO) -> dict:
    return _train_lora(name, epochs, lr, accum, max_examples, fresh, micro, "hf")


@app.function(image=lora_unsloth_image, gpu="H100", volumes={DATA: pack_vol, CACHE: cache_vol}, timeout=6 * 3600,
              cpu=8, memory=65536)          # Unsloth parks activations in RAM — a little more headroom
def train_lora_unsloth(name: str, epochs: float = 1.0, lr: float = 1e-4, accum: int = 8, max_examples: int = 0,
                       fresh: bool = True, micro: int = LORA_MICRO) -> dict:
    return _train_lora(name, epochs, lr, accum, max_examples, fresh, micro, "unsloth")


def _train_lora(name: str, epochs: float, lr: float, accum: int, max_examples: int, fresh: bool, micro: int,
                trainer: str = "hf") -> dict:
    """GPU (H100): QLoRA on the 32B (4-bit, bf16 maths) — rank-16 adapters on the language layers
    only (the vision part is left alone). Loss only on the answer. Saves lora/<name>/ (PEFT format,
    which vLLM loads on top of the FP8 teacher). GPU kept busy: `micro` examples per forward pass
    (left-padded, answer-only logits), the next batch's images prepared on the CPU while the GPU
    trains, and one optimiser step per `accum` examples whatever the micro-batch (so halving it after
    an out-of-memory error doesn't change the training).
    Save and resume: the run keeps its own copy of the examples (lora/<name>/data.json) and saves a
    checkpoint (adapter + optimiser) every LORA_SAVE_EVERY examples; fresh=False carries on from it.
    trainer "unsloth": the same loop on Unsloth's model (its 4-bit kernels, offloaded checkpointing,
    and its own memory-light loss — so the loss is taken from labels, not sliced logits)."""
    if trainer == "unsloth":
        import unsloth  # noqa: F401 — must come before transformers / peft so it can patch them
    import math
    import random
    import torch
    from PIL import Image
    from transformers import get_cosine_schedule_with_warmup
    import trial_logic as TL

    import shutil
    pack_vol.reload()
    out_dir = _lora_dir() / name
    ck = out_dir / "checkpoint"
    if fresh:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "data.json").exists() and (_lora_dir() / "data.json").exists():
        shutil.copyfile(_lora_dir() / "data.json", out_dir / "data.json")      # this run's own copy
    data = _read_json(out_dir / "data.json")
    if not data or not data.get("train"):
        return {"error": "no training data — the prepare step found no labelled frames with pictures"}
    train = list(data["train"])
    random.Random(1).shuffle(train)
    if max_examples:
        train = train[:max_examples]
    t0 = time.time()
    base_id = TEACHERS["32b"]                  # train on the full-precision weights, 4-bit in memory
    print(f"[lora] loading {base_id} in 4-bit ({trainer})…", flush=True)
    if trainer == "unsloth":
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(base_id, load_in_4bit=True,
                                                           use_gradient_checkpointing="unsloth")
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
            finetune_mlp_modules=True, r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none",
            random_state=1)
        FastVisionModel.for_training(model)
        loss_fn = _answer_loss_labels
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(base_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_id, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                                   bnb_4bit_use_double_quant=True))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                                                 target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
        loss_fn = _answer_loss
    model.print_trainable_parameters()
    print(f"[lora] ready after {(time.time() - t0) / 60:.1f} min · {len(train)} examples, {epochs} epoch(s)",
          flush=True)

    processor.tokenizer.padding_side = "left"          # answers stay at the end of every row
    marker = processor.tokenizer.encode(LORA_MARKER, add_special_tokens=False)

    def encode(recs: list) -> tuple:
        """CPU: one left-padded batch (tensors on the CPU) + each row's answer start."""
        convs = []
        for rec in recs:
            imgs = [Image.open(f).convert("RGB") for f in rec["frames"]]
            convs.append([{"role": "user", "content": _lora_content(data["head"], data["moment"], data["tail"], imgs)},
                          {"role": "assistant", "content": [{"type": "text", "text": rec["target"]}]}])
        b = processor.apply_chat_template(convs, tokenize=True, return_dict=True, return_tensors="pt", padding=True)
        starts = [TL.answer_start(row.tolist(), marker) for row in b["input_ids"]]
        return dict(b), starts

    def encode_group(recs: list, mb: int) -> list:
        return [encode(recs[i:i + mb]) for i in range(0, len(recs), mb)]

    def to_gpu(b: dict) -> dict:
        return {k: (v.to(model.device, non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}

    steps = max(1, math.ceil(len(train) * epochs / accum))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), steps)
    order = [train[i % len(train)] for i in range(int(len(train) * epochs))]
    seen, losses, start, prior_s = 0, [], 0, 0.0
    progress = _read_json(ck / "progress.json")
    start = _lora_resume_point(progress, data.get("created"), len(order), epochs, lr)
    if start:                                   # carry on from the last checkpoint
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(ck / "adapter_model.safetensors")))
        st = torch.load(ck / "optim.pt", map_location="cpu")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        seen, losses, prior_s = progress["seen"], list(progress.get("losses") or []), float(progress.get("train_s") or 0)
        print(f"[lora] resuming from the checkpoint at {start}/{len(order)} examples", flush=True)
    elif progress:
        print("[lora] a checkpoint exists but was for different examples or settings — starting over", flush=True)

    def save_checkpoint(n: int, train_s: float) -> None:
        tmp = out_dir / "checkpoint_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(str(tmp))
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, tmp / "optim.pt")
        (tmp / "progress.json").write_text(json.dumps({
            "n": n, "total": len(order), "seen": seen, "losses": losses[-200:], "train_s": train_s,
            "data_created": data.get("created"), "epochs": epochs, "lr": lr,
            "saved": time.strftime("%Y-%m-%d %H:%M")}), encoding="utf-8")
        shutil.rmtree(ck, ignore_errors=True)
        tmp.rename(ck)
        pack_vol.commit()
        print(f"[lora] checkpoint saved at {n}/{len(order)}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [order[i:i + accum] for i in range(0, len(order), accum)]
    first = start // accum                      # checkpoints are on whole steps
    mb = max(1, min(int(micro), accum))
    pool = ThreadPoolExecutor(max_workers=1)    # prepares the next batch's images while the GPU works
    nxt = pool.submit(encode_group, groups[first], mb) if first < len(groups) else None
    model.train()
    t_train, tokens, reported = time.time(), 0, False
    n = start
    for gi in range(first, len(groups)):
        recs = groups[gi]
        enc, used_mb = nxt.result(), mb
        nxt = pool.submit(encode_group, groups[gi + 1], mb) if gi + 1 < len(groups) else None
        while True:
            if used_mb != mb:
                enc = encode_group(recs, mb)
            total = sum(sum(b["input_ids"].shape[1] - s0 for s0 in st) for b, st in enc)
            try:
                opt.zero_grad(set_to_none=True)
                step_loss = 0.0
                for b, st in enc:
                    loss_sum, ntok = loss_fn(model, to_gpu(b), st)
                    (loss_sum / max(1, total)).backward()
                    step_loss += float(loss_sum)
                    tokens += int(b["attention_mask"].sum())
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if mb == 1:
                    print(f"[lora] skipped a step (out of memory even one example at a time)", flush=True)
                    step_loss = None
                    break
                mb = max(1, mb // 2)
                used_mb = None
                print(f"[lora] out of GPU memory — now {mb} example(s) at a time", flush=True)
        n = min(len(order), (gi + 1) * accum)
        if step_loss is None:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(step_loss / max(1, total))
        seen += len(recs)
        if not reported:
            reported = True
            print(f"[lora] {mb} example(s) at a time · peak GPU memory "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.0f} GB of "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}", flush=True)
        if n % 48 == 0 or n == len(order):
            el = time.time() - t_train
            print(f"[lora] {n}/{len(order)} examples · loss {sum(losses[-6:]) / max(1, len(losses[-6:])):.3f} · "
                  f"{tokens / max(el, 1e-6):,.0f} tokens/s · {(el + prior_s) / 60:.0f} min so far, about "
                  f"{el / max(1, n - start) * (len(order) - n) / 60:.0f} min left", flush=True)
        if n % LORA_SAVE_EVERY == 0 and n < len(order):
            save_checkpoint(n, prior_s + time.time() - t_train)
    pool.shutdown(wait=False)
    model.eval()
    val_loss = None
    if data.get("val"):
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b, st in encode_group(data["val"][:(32 if max_examples else 150)], mb):
                ls, nt = loss_fn(model, to_gpu(b), st)
                vs, vn = vs + float(ls), vn + int(nt)
        val_loss = round(vs / max(1, vn), 4)
    model.save_pretrained(str(out_dir))
    _plain_target_modules(out_dir / "adapter_config.json")
    shutil.rmtree(ck, ignore_errors=True)             # finished — the checkpoint isn't needed any more
    sec = time.time() - t0 + prior_s
    meta = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "base": base_id, "rank": LORA_RANK,
            "epochs": epochs, "lr": lr, "examples": seen, "micro_batch": mb, "trainer": trainer,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "train_loss_end": round(sum(losses[-50:]) / max(1, len(losses[-50:])), 4),
            "val_loss": val_loss, "minutes": round(sec / 60, 1), "cost": round(TL.stage_cost(sec, "H100"), 2),
            "test_videos": data["test_videos"], "val_videos": data["val_videos"], "vocab_csv": data["vocab_csv"]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    pack_vol.commit()
    return {k: v for k, v in meta.items() if k not in ("vocab_csv", "test_videos", "val_videos")}


VISION_SNAPSHOT = ("vision.json", "person_curve.npy", "embeddings.npz")


@app.function(image=report_image, volumes={DATA: pack_vol})
def vision_snapshot(video_id: str = "") -> dict:
    """For --stage vision-check: pick (unless given) the shortest finished video with people tracked
    in it, from before the batched pipeline, and keep a copy of its vision results to compare."""
    import shutil
    pack_vol.reload()
    if not video_id:
        cands = []
        for p in Path(OUT).iterdir() if Path(OUT).exists() else []:
            vis = _read_json(p / "vision.json")
            if vis and not vis.get("partial") and vis.get("pipeline", 1) == 1 and (vis.get("tracks") or 0) >= 5:
                cands.append((vis["duration"], p.name))
        if not cands:
            return {"video_id": None}
        video_id = min(cands)[1]
    d = Path(f"{OUT}/{video_id}")
    for f in VISION_SNAPSHOT:
        if (d / f).exists():
            shutil.copyfile(d / f, d / f.replace(".", "_before.", 1))
    pack_vol.commit()
    return {"video_id": video_id, "minutes": round((_read_json(d / "vision.json") or {}).get("duration", 0) / 60, 1)}


@app.function(image=report_image, volumes={DATA: pack_vol})
def vision_compare(video_id: str, restore: bool = True) -> str:
    """Old vs new vision on one video: cuts, tracks, the movement curve, per-scene rhythm, and GPU
    time / cost per footage hour. Then (restore) puts the old results back, so the video's tags
    stay matched to the moments they were made for — the check changes nothing downstream."""
    import shutil
    import numpy as np
    import trial_logic as TL
    pack_vol.reload()
    d = Path(f"{OUT}/{video_id}")
    old, new = _read_json(d / "vision_before.json"), _read_json(d / "vision.json")
    if not old or not new or new.get("pipeline", 1) < 2:
        return "Nothing to compare — the new run didn't finish (see the errors above)."
    hrs = (new["duration"] or 1) / 3600
    co, cn = np.load(d / "person_curve_before.npy"), np.load(d / "person_curve.npy")
    n = min(len(co), len(cn))
    co, cn = co[:n], cn[:n]
    both = np.isfinite(co) & np.isfinite(cn)
    corr = float(np.corrcoef(co[both], cn[both])[0, 1]) if both.sum() > 2 else None
    mad = float(np.mean(np.abs(co[both] - cn[both]))) if both.any() else None
    scale = float(np.nanmean(np.abs(co))) if np.isfinite(co).any() else None
    so, sn = {s["scene_id"]: s for s in old["scenes"]}, {s["scene_id"]: s for s in new["scenes"]}
    common = [k for k in so if k in sn]

    def diff(key):
        xs = [abs((so[k].get(key) or 0) - (sn[k].get(key) or 0)) for k in common
              if so[k].get(key) is not None or sn[k].get(key) is not None]
        return (max(xs) if xs else 0.0), (sum(1 for x in xs if x > 1e-6))

    cuts_same = sum(1 for c in new["cuts"] if any(abs(c - x) <= 0.1 for x in old["cuts"]))
    cpu = new.get("cpu_timings") or {}
    gpu_old, gpu_new = old["timings"]["total"], new["timings"]["total"]
    cost_old = TL.stage_cost(gpu_old, old.get("gpu") or VISION_GPU) / hrs
    cost_new = (TL.stage_cost(gpu_new, new.get("gpu") or VISION_GPU) + TL.stage_cost(cpu.get("total", 0), "CPU")) / hrs
    lines = [f"Vision check on {video_id} ({new['duration'] / 60:.1f} min of video): old pipeline vs batched pose + "
             "tracking on CPU.", "",
             "| | Old | New |", "|---|---|---|",
             f"| Scene cuts | {len(old['cuts'])} | {len(new['cuts'])} ({cuts_same} within 0.1 s of an old one) |",
             f"| People tracked | {old.get('tracks')} | {new.get('tracks')} |",
             f"| Pose GPU seconds per footage hour | {old['timings'].get('pose_gpu', 0) / hrs:.0f} | "
             f"{new['timings'].get('pose_gpu', 0) / hrs:.0f} |",
             f"| All vision GPU seconds per footage hour | {gpu_old / hrs:.0f} | {gpu_new / hrs:.0f} |",
             f"| CPU seconds per footage hour (cheap) | – | {cpu.get('total', 0) / hrs:.0f} |",
             f"| Vision $ per footage hour | ${cost_old:.3f} ({old.get('gpu')}) | ${cost_new:.3f} ({new.get('gpu')} + CPU) |",
             "", "Movement (should match):",
             f"- Person-motion curve: correlation {('–' if corr is None else f'{corr:.4f}')} (1 = identical), "
             f"mean difference {('–' if mad is None else f'{mad:.5f}')} vs a typical value of "
             f"{('–' if scale is None else f'{scale:.5f}')}.",
             f"- Per scene ({len(common)} scenes): " + ", ".join(
                 f"{k} biggest difference {diff(k)[0]:.4g} ({diff(k)[1]} scenes differ)"
                 for k in ("level", "periodicity", "movement_bpm", "people")) + "."]
    if restore:
        for f in VISION_SNAPSHOT:
            b = d / f.replace(".", "_before.", 1)
            if b.exists():
                shutil.copyfile(b, d / f)
        lines += ["", "The old results are back in place, so nothing downstream changed. From the next batch, "
                      "new videos use the batched pipeline."]
    pack_vol.commit()
    return "\n".join(lines)


@app.function(image=report_image, volumes={DATA: pack_vol})
def pick_check_video() -> dict:
    """For --stage fp8-check: the video already tagged by the 32B with the most of your labelled
    frames (🎯 cards that the 32B was scored on), so the FP8 check has the most to compare."""
    pack_vol.reload()
    best, best_n = None, 0
    for p in sorted(Path(OUT).iterdir()) if Path(OUT).exists() else []:
        tj = _read_json(p / "tags_32b.json")
        if not tj:
            continue
        n = sum(1 for r in (tj.get("results") or {}).values() if r.get("gold_id"))
        if n > best_n:
            best, best_n = p.name, n
    return {"video_id": best, "labelled": best_n}


@app.function(image=report_image, volumes={DATA: pack_vol})
def list_pack() -> list:
    pack_vol.reload()
    return sorted(p.stem for p in Path(f"{PACK}/manifests").glob("*.json"))


# ---------------------------------------------------------------------------

def _push_labels(vocab: bool = True) -> None:
    """Send your latest labels and vocabulary from this machine's catalogue folder to the Volume,
    so prep (gold frames, worked examples), compare and the report use today's corrections — not
    the copy uploaded with the last batch. Quietly does nothing where there's no catalogue."""
    import shutil
    import subprocess
    try:
        from pack_for_modal import newest_copy, VOLUME
    except Exception:
        return
    if not shutil.which("modal"):
        return
    sent = []
    for name in ("training_labels.json",) + (("tag_vocabulary.csv",) if vocab else ()):
        src = newest_copy(name)
        if src is None:
            continue
        rc = subprocess.run(["modal", "volume", "put", "--force", VOLUME, str(src), f"/pack/library/{name}"],
                            capture_output=True).returncode
        if rc == 0:
            sent.append(f"{name} (saved {time.strftime('%d %b %H:%M', time.localtime(src.stat().st_mtime))})")
    if sent:
        print("Sent your latest " + " and ".join(sent) + " to Modal.")


def _run_vision(vids: list, max_seconds: float, redo: bool) -> None:
    """Vision in three steps: CPU (scene-cut frames) → GPU (cuts, pose, embeddings, frames) → CPU
    (tracking, movement, rhythm). The CPU steps run many videos at once."""
    t0 = time.time()
    print(f"Vision step 1/3 (CPU): decoding scene-cut frames for {len(vids)} video(s)…")
    for r in vision_prep.starmap([(v, max_seconds, redo) for v in vids], return_exceptions=True):
        print("  ", r)
    print(f"Vision step 2/3 (GPU: {' / '.join(VISION_GPUS)}, first free): cuts, pose in batches of "
          f"{POSE_BATCH}, embeddings…")
    for r in vision.starmap([(v, max_seconds, redo) for v in vids], return_exceptions=True):
        print("  ", r)
    print("Vision step 3/3 (CPU): person tracking, movement and rhythm…")
    for r in vision_motion.starmap([(v, max_seconds) for v in vids], return_exceptions=True):
        print("  ", r)
    print(f"   Vision done in {(time.time() - t0) / 60:.1f} min.")


@app.local_entrypoint()
def main(stage: str = "all", smoke: bool = False, videos: str = "", redo: bool = False, teacher: str = "",
         examples: int = -1, dedup: float = -1.0, batch: int = 0, label: str = "", keep_vocab: bool = False,
         lora: str = "", epochs: float = 1.0, max_examples: int = 0, trainer: str = "unsloth"):
    """teacher: 8b, 32b, 32b-fp8 or both (= 8b + 32b). Default: 8b for --smoke, 32b-fp8 otherwise.
    examples: worked examples from your labels before each moment (default 0 = off, at most 6).
    dedup: near-identical moments in a scene are tagged once and the answer copied when their
    picture similarity is above this (default 0.96; 0 = tag every moment).
    batch: only the videos of this batch (e.g. --batch 6), instead of typing their names.
    --stage test: re-tag the fixed 🧪 test set (~2 h) and compare with earlier test runs; --label names
    the run (e.g. --label positions-v2); --keep-vocab uses the vocabulary already on Modal (the last
    batch's) instead of sending today's — for a baseline. --stage test-pick [--redo]: show / re-pick the set.
    --stage lora-train --label <name> [--epochs 1]: fine-tune the teacher on your labels (test set held
    out) → lora/<name>. Then --stage test --lora <name> tags the test set with it.
    --stage check-labels --lora <name>: tag only your labelled training frames, plain + LoRA, for 🔍.
    --trainer unsloth (default) or hf; --max-examples 16 --label smoke = a quick ~$1 check that training runs."""
    # No trial_logic import here: this runs on the machine you start it from, whose venv may
    # only have modal (trial_logic needs numpy).
    examples = EXAMPLES_DEFAULT if examples < 0 else min(examples, EXAMPLES_MAX)
    dedup = DEDUP_DEFAULT if dedup < 0 else dedup
    teacher = (teacher or ("8b" if smoke else TEACHERS_DEFAULT)).lower()
    keys = ["8b", "32b"] if teacher == "both" else [teacher]
    if any(k not in TEACHERS for k in keys):
        print(f"--teacher must be one of: {', '.join(TEACHERS)}, both")
        return
    stages = (["download", "vision", "prep", "tag", "report", "export"] if stage == "all"
              else ["report", "export"] if stage == "results" else [stage])
    if "tag" in stages and "prep" not in stages:
        stages.insert(stages.index("tag"), "prep")      # the GPU only ever reads what prep wrote
    if stage == "status":
        _print_status(status.remote(keys), keys)
        return
    if stage == "test-pick":
        _push_labels(vocab=False)
        ts = pick_test_set.remote(TEST_HOURS, redo)
        print(f"🧪 Test set ({ts.get('created')}): {len(ts.get('videos') or [])} videos, {ts.get('hours')} h, "
              f"{ts.get('labelled')} of your labelled frames. Held out of any teacher fine-tune.")
        for x in ts.get("videos") or []:
            print(f"   batch {x['batch']:>3} · {x['minutes']:5.1f} min · {x['labelled']:3} labelled · {x['video_id']}")
        return
    if stage == "check-labels":
        if not lora.strip():
            print("--lora <name> is needed (e.g. --lora try1).")
            return
        _run_check(lora.strip(), dedup)
        return
    if stage == "test":
        _run_test(keys, label.strip().replace(" ", "-"), keep_vocab, dedup, lora.strip())
        return
    if stage == "lora-train":
        name = label.strip().replace(" ", "-") or time.strftime("lora-%Y%m%d-%H%M")
        st = lora_status.remote(name)
        ck = st.get("checkpoint")
        if st.get("done") and not redo:
            print(f"🧠 LoRA '{name}' is already trained. Test it: modal run modal_trial.py --stage test --lora {name}"
                  f"\n   (or --redo to train it again from scratch)")
            return
        resume = bool(ck) and not redo
        if resume:
            print(f"🧠 LoRA '{name}': carrying on from the checkpoint at {ck['n']}/{ck['total']} examples "
                  f"(saved {ck.get('saved')}) — same examples as before. (--redo starts over.)")
        else:
            _push_labels()
            ts = pick_test_set.remote(TEST_HOURS, False)
            print(f"🧠 LoRA '{name}': the {len(ts.get('videos') or [])} test-set videos are held out.")
            print("Preparing training examples (CPU)…")
            print("  ", prepare_lora_data.remote())
        print(f"Training on an H100 ({epochs} epoch(s)) — progress every 50 examples, a checkpoint every "
              f"{LORA_SAVE_EVERY}…")
        fn = train_lora_unsloth if trainer == "unsloth" else train_lora
        r = fn.remote(name, epochs, 1e-4, 8, max_examples, not resume)
        print("  ", r)
        if not r.get("error"):
            print(f"\nNext: modal run modal_trial.py --stage test --lora {name}")
        return
    if stage in ("all", "prep", "tag", "report", "results", "compare", "fp8-check"):
        _push_labels()
    if stage == "vision-check":
        # One command: re-run one finished video through the batched vision pipeline and compare.
        snap = vision_snapshot.remote(videos.strip())
        if not snap["video_id"]:
            print("No finished video with people tracked from the original pipeline to compare against.")
            return
        print(f"Vision check on {snap['video_id']} ({snap['minutes']} min of video).")
        _run_vision([snap["video_id"]], 0, True)
        print("\n" + vision_compare.remote(snap["video_id"]))
        return
    if stage == "fp8-check":
        # One command for the FP8 check: re-tag the 32B video with the most labelled frames with
        # the FP8 build, then rebuild the report so the two sit side by side.
        pick = pick_check_video.remote()
        if not pick["video_id"]:
            print("No video tagged by the 32B with labelled frames yet — nothing to compare.")
            return
        print(f"FP8 check on {pick['video_id']} ({pick['labelled']} of your labelled frames).")
        check_vid, keys, redo = pick["video_id"], ["32b-fp8"], True
        stages = ["download", "prep", "tag", "report", "export"]
    if batch:
        rows = status.remote(keys)["rows"]
        vids = [r["video_id"] for r in rows if str(r["batch"]) == str(batch)]
        if not vids:
            print(f"No videos in batch {batch}.")
            return
        print(f"Batch {batch}: {len(vids)} videos.")
    else:
        vids = ([check_vid] if stage == "fp8-check"          # (file names can contain commas)
                else [v.strip() for v in videos.split(",") if v.strip()] or list_pack.remote())
    if stage == "compare":
        print(compare_runs.remote(vids, keys[0]))
        return
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
        todo_vision, todo_tags = list(vids), {k: list(vids) for k in keys}
        redo = True
        print(f"Smoke test on {vids[0]} (first {max_seconds} s, {max_prompts} prompts).")
    t_all = time.time()
    if "download" in stages:
        print(download_models.remote([TEACHERS[k] for k in keys]))
    if "vision" in stages and todo_vision:
        _run_vision(todo_vision, max_seconds, redo)
    if "vision" in stages and "tag" in stages and not smoke and not redo:
        for k in keys:          # vision just ran — anything it finished needs tagging too
            st = {r["video_id"]: r for r in status.remote([k])["rows"]}
            todo_tags[k] = [v for v in vids if not (st.get(v) or {}).get(f"tags_{k}")]
    # prep (CPU): one ready-to-feed file per video, so the GPU reads nothing else.
    todo_prep = (vids if stage == "prep" else
                 sorted({v for k in keys for v in todo_tags[k]} if "tag" in stages else set(), key=vids.index))
    if "prep" in stages and todo_prep:
        if examples:
            t0 = time.time()
            print("Gathering your labelled frames for the worked examples (CPU)…")
            print(f"   {build_example_pool.remote()} frames ready in {time.time() - t0:.0f} s.")
        print(f"Preparing prompts for {len(todo_prep)} video(s) on CPU (near-identical moments copy an answer "
              f"above {dedup:.2f} similarity" + (f", {examples} worked examples" if examples else "") + ")…")
        tot_p = tot_c = 0
        for r in prepare_prompts.starmap([(v, max_prompts, examples, dedup) for v in todo_prep],
                                         return_exceptions=True):
            print("  ", r)
            if isinstance(r, dict) and not r.get("error"):
                tot_p += r["prompts"]
                tot_c += r["copied"]
        if tot_p + tot_c:
            print(f"   {tot_p} moments go to the GPU, {tot_c} copy an answer "
                  f"({100 * tot_c / (tot_p + tot_c):.0f}% fewer prompts).")
    if "tag" in stages:
        for k in keys:
            if not todo_tags[k]:
                print(f"{k}: every video already tagged.")
                continue
            print(f"Tagging {len(todo_tags[k])} video(s) with {TEACHERS[k]} ({' / '.join(TEACHER_GPUS[k])}, first free)…")
            tagger = TAGGERS[k]()
            for r in tagger.tag_video.starmap([(v, redo) for v in todo_tags[k]], return_exceptions=True):
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
