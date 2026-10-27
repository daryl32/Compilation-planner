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
Stages: download, vision, tag, report, all (default). Each skips a video it has already done
unless you pass --redo.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
EMB_MODEL = "google/siglip2-so400m-patch14-384"
POSE_MODEL = "yolo11l-pose.pt"
# First choice first; Modal falls back down the list when that GPU type is busy.
VISION_GPUS = ["L4", "A10", "L40S", "A100-40GB"]
TAG_GPUS = ["H100", "A100-80GB"]
VISION_GPU, TAG_GPU = VISION_GPUS[0], TAG_GPUS[0]

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
tag_image = (base.pip_install("vllm", "numpy", "pillow", "huggingface_hub")
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
def download_models(vlm: str = VLM_MODEL) -> str:
    from huggingface_hub import snapshot_download
    done = []
    for repo in (EMB_MODEL, vlm, "Sn4kehead/TransNetV2"):
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
    prev = _read_json(out_dir / "vision.json")
    if not redo and prev and not prev.get("partial"):
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
    emb_model = AutoModel.from_pretrained(EMB_MODEL, torch_dtype=torch.float16).cuda().eval()
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

@app.cls(image=tag_image, gpu=TAG_GPUS, volumes={DATA: pack_vol, CACHE: cache_vol},
         timeout=4 * 3600, cpu=8, memory=65536, max_containers=1)
class Tagger:
    @modal.enter()
    def load(self):
        t0 = time.time()
        from vllm import LLM
        self.llm = LLM(model=VLM_MODEL, max_model_len=8192, gpu_memory_utilization=0.92,
                       limit_mm_per_prompt={"image": 7, "video": 0}, max_num_seqs=32,
                       enable_prefix_caching=True)
        self.load_seconds = time.time() - t0
        self.first = True

    @modal.method()
    def tag_video(self, video_id: str, max_prompts: int = 0, redo: bool = False) -> dict:
        import base64
        import trial_logic as TL
        from vllm import SamplingParams

        pack_vol.reload()
        out = Path(f"{OUT}/{video_id}/tags.json")
        vis = _read_json(f"{OUT}/{video_id}/vision.json")
        if not vis:
            return {"video_id": video_id, "error": "no vision.json — run the vision stage first"}
        prev = _read_json(out)
        if (not redo and prev and not prev.get("partial") and not vis.get("partial")
                and prev.get("targets") == len(vis["targets"])):
            return {"video_id": video_id, "skipped": True}
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
        outs = self.llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=256), use_tqdm=False) if convs else []
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
        out.write_text(json.dumps({"video_id": video_id, "model": VLM_MODEL, "gpu": _gpu_name(TAG_GPU),
                                   "partial": bool(max_prompts) or bool(vis.get("partial")),
                                   "targets": len(vis["targets"]),
                                   "seconds": round(seconds, 1), "load_seconds": round(load, 1),
                                   "prompts": len(convs), "unreadable": bad,
                                   "vocab": [r["tag"] for r in vocab], "results": results}), encoding="utf-8")
        pack_vol.commit()
        return {"video_id": video_id, "prompts": len(convs), "seconds": round(seconds, 1), "unreadable": bad}


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
    tagged = {v: _read_json(f"{OUT}/{v}/tags.json") for v in done if Path(f"{OUT}/{v}/tags.json").exists()}
    mans = {v: _read_json(f"{PACK}/manifests/{v}.json") for v in done}

    # --- Tags: old tagger (all your answers, and just the trial videos) vs the teacher
    gold_tags = [i for i in gold if i.get("type") == "tags" and not i.get("skipped")]
    old_all = TL.score_tags([(i.get("suggested") or [], i.get("tags") or []) for i in gold_tags], tags)
    by_gold = {}
    for v, tj in tagged.items():
        for r in tj["results"].values():
            if r.get("gold_id"):
                by_gold[r["gold_id"]] = r
    trial_gold = [i for i in gold_tags if i["id"] in by_gold]
    old_trial = TL.score_tags([(i.get("suggested") or [], i.get("tags") or []) for i in trial_gold], tags)
    teacher = TL.score_tags([(by_gold[i["id"]]["tags"], i.get("tags") or []) for i in trial_gold], tags)
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
    stage_sec = {"vision": sum(d["timings"]["total"] for d in vis.values()),
                 "tag": sum(t["seconds"] + t.get("load_seconds", 0) for t in tagged.values())}
    lib_hours = float(pack.get("library_hours") or 0) or trial_hours

    def used(rows, default):        # the GPU most of the work actually ran on
        names = [r.get("gpu") or default for r in rows]
        return max(set(names), key=names.count) if names else default

    cost = TL.extrapolate(stage_sec, {"vision": used(vis.values(), VISION_GPU),
                                      "tag": used(tagged.values(), TAG_GPU)}, trial_hours, lib_hours)

    def f(x):
        return "–" if x is None else f"{x:.2f}"

    md = [f"# Modal trial report — {time.strftime('%d %b %Y %H:%M')}", "",
          f"**{len(done)} videos, {trial_hours:.2f} h of footage** processed "
          f"({len(tagged)} tagged). Library: {lib_hours:.1f} h.", "",
          "## Cost", "",
          "| Stage | GPU | Trial | Per footage hour | Full library (est.) | GPU hours (library) |",
          "|---|---|---|---|---|---|"]
    for k in ("vision", "tag"):
        c = cost[k]
        md.append(f"| {k} | {c['gpu']} | ${c['trial_$']:.2f} | ${c['per_footage_hour_$']:.3f} | "
                  f"${c['library_$']:.2f} | {c['library_gpu_hours']:.1f} |")
    md += [f"| **total** | | **${cost['total']['trial_$']:.2f}** | | **${cost['total']['library_$']:.2f}** | |", "",
           "## Tags", "",
           f"Your labels: {len(gold_tags)} frames ({len(trial_gold)} in the trial videos). Overall F1 on the "
           f"trial frames — old tagger **{f(old_trial['overall']['f1'])}**, teacher "
           f"**{f(teacher['overall']['f1'])}** (old tagger on all your frames: {f(old_all['overall']['f1'])}).", "",
           "| Tag | Kind | Your examples | Old F1 | Teacher F1 | Teacher P / R | Suggestion |",
           "|---|---|---|---|---|---|---|"]
    for t in sorted(tags, key=lambda t: -positives[t]):
        tp, ot = teacher["per_tag"][t], old_trial["per_tag"][t]
        md.append(f"| {t} | {kinds[t]} | {positives[t]} | {f(ot['f1'])} | {f(tp['f1'])} | "
                  f"{f(tp['precision'])} / {f(tp['recall'])} | {verdict(t)} |")
    unreadable = sum(t.get("unreadable", 0) for t in tagged.values())
    prompts = sum(t.get("prompts", 0) for t in tagged.values())
    md += ["", f"Teacher replies that weren't valid JSON: {unreadable} of {prompts}.", "",
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
    errs = [v for v in done if v not in tagged]
    if errs:
        md += ["## Not tagged yet", "", ", ".join(errs), ""]
    text = "\n".join(md)
    Path(f"{OUT}/report.md").write_text(text, encoding="utf-8")
    Path(f"{OUT}/report.json").write_text(json.dumps({
        "cost": cost, "tags": {"old_all": old_all, "old_trial": old_trial, "teacher": teacher,
                               "positives": positives, "verdicts": {t: verdict(t) for t in tags}},
        "boundaries": bounds, "shots": shot_stats, "movement_corr": corr}, indent=1), encoding="utf-8")
    pack_vol.commit()
    return text


@app.function(image=report_image, volumes={DATA: pack_vol})
def list_pack() -> list:
    pack_vol.reload()
    return sorted(p.stem for p in Path(f"{PACK}/manifests").glob("*.json"))


# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "all", smoke: bool = False, videos: str = "", redo: bool = False):
    stages = ["download", "vision", "tag", "report"] if stage == "all" else [stage]
    vids = [v.strip() for v in videos.split(",") if v.strip()] or list_pack.remote()
    if not vids:
        print("No videos in the pack — run pack_for_modal.bat --upload first.")
        return
    max_seconds, max_prompts = (90, 12) if smoke else (0, 0)
    if smoke:
        vids = vids[:1]
        redo = True
        print(f"Smoke test on {vids[0]} (first {max_seconds} s, {max_prompts} prompts).")
    t_all = time.time()
    if "download" in stages:
        print(download_models.remote())
    if "vision" in stages:
        print(f"Vision on {len(vids)} video(s) ({' / '.join(VISION_GPUS)}, first free)…")
        for r in vision.starmap([(v, max_seconds, redo) for v in vids], return_exceptions=True):
            print("  ", r)
    if "tag" in stages:
        print(f"Tagging with {VLM_MODEL} ({' / '.join(TAG_GPUS)}, first free)…")
        tagger = Tagger()
        for r in tagger.tag_video.starmap([(v, max_prompts, redo) for v in vids], return_exceptions=True):
            print("  ", r)
    if "report" in stages:
        text = report.remote()
        name = "trial_report_smoke.md" if smoke else "trial_report.md"
        Path(name).write_text(text, encoding="utf-8")
        print("\n" + text + f"\n\nSaved {Path(name).resolve()}")
    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min.")
