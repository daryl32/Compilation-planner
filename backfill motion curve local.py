r"""
Local Windows version of backfill_motion_curve_gpu.py.

Since your Google Drive is in STREAM mode, reading a video's frames one by
one straight from G:\\My Drive\... would re-trigger network fetches the
whole way through the decode — the same problem we solved in Colab with a
/dev/shm copy. Windows has no equivalent RAM disk built in, so this copies
each video to a real local folder first (one network fetch via Drive Stream,
same as opening the file once), then decodes entirely from that local copy.

Run from your activated venv:
    python backfill_motion_curve_local.py

Needs: opencv-python, torch, tqdm (torch: install the CUDA build from
pytorch.org for your GPU, same as earlier in this project — a CPU-only
torch install will still work here, just without the GPU speedup).

Safe to re-run — skips any catalogue where every scene already has a
motion_curve.
"""

import os
import json
import shutil
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CATALOGUE_DIR
from render_preview import colab_to_local

LOCAL_TEMP_DIR = Path(__file__).parent / "_tmp_motion"  # genuine local disk, next to this script

LOCAL_TEMP_DIR.mkdir(exist_ok=True)

catalogue_files = sorted(
    f for f in CATALOGUE_DIR.glob("*.json")
    if f.name not in ("training_data.jsonl", "video_index.json")
)
print(f"Found {len(catalogue_files)} catalogues.\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}" + ("" if device == "cuda" else "  (no GPU found — this will run at CPU speed)"))
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

TARGET_H, TARGET_W = 90, 160
BATCH_SIZE = 128  # lower this if you hit an out-of-memory error


def compute_motion_curve_gpu(video_path: str, scenes: list) -> None:
    """Identical logic to the Colab GPU version — sets scene['motion_curve']
    = {"fps": .., "values": [..]} for every scene dict (needs start_frame/
    end_frame), via one sequential decode pass with batched GPU math."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max(
        (s["end_frame"] for s in scenes), default=0
    )
    all_diffs = np.zeros(max(total_frames, 1), dtype=np.float32)

    prev_last_gray = None
    frame_idx = 0
    batch = []

    def flush(batch_frames, start_idx, prev_last):
        arr = np.stack(batch_frames)
        t = torch.from_numpy(arr).to(device)
        gray = (t[..., 2].float() * 0.299 + t[..., 1].float() * 0.587 + t[..., 0].float() * 0.114)
        gray = gray.unsqueeze(1)
        gray = F.interpolate(gray, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
        gray = gray.squeeze(1)

        if prev_last is not None:
            full = torch.cat([prev_last.unsqueeze(0), gray], dim=0)
            diffs = (full[1:] - full[:-1]).abs().mean(dim=(1, 2))
            all_diffs[start_idx:start_idx + len(diffs)] = diffs.cpu().numpy()
        else:
            diffs = (gray[1:] - gray[:-1]).abs().mean(dim=(1, 2))
            all_diffs[start_idx + 1:start_idx + 1 + len(diffs)] = diffs.cpu().numpy()

        return gray[-1]

    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append(frame)
        frame_idx += 1
        if len(batch) >= BATCH_SIZE or frame_idx >= total_frames:
            start_idx = frame_idx - len(batch)
            prev_last_gray = flush(batch, start_idx, prev_last_gray)
            batch = []
    cap.release()

    for scene in scenes:
        lo = max(0, min(scene["start_frame"], len(all_diffs)))
        hi = max(lo, min(scene["end_frame"], len(all_diffs)))
        scene["motion_curve"] = {
            "fps": round(float(fps), 3),
            "values": [round(float(v), 2) for v in all_diffs[lo:hi]],
        }


for cat_file in tqdm(catalogue_files, desc="Videos"):
    with open(cat_file) as f:
        catalogue = json.load(f)
    if not (isinstance(catalogue, dict) and "video_id" in catalogue and "scenes" in catalogue):
        continue  # not a video catalogue (labels.json, library_meta.json, ...)

    video_id = catalogue["video_id"]
    scenes = catalogue["scenes"]

    if all("motion_curve" in s and s["motion_curve"] for s in scenes):
        print(f"{video_id}: already has motion curves, skipping.")
        continue

    source_path = colab_to_local(catalogue["source_path"])
    if not os.path.exists(source_path):
        print(f"⚠️  {video_id}: source video not found at {source_path}, skipping.")
        continue

    print(f"Computing motion curves for {video_id} ({len(scenes)} scenes)...")
    local_video = LOCAL_TEMP_DIR / f"tmp_curve_{video_id}{Path(source_path).suffix}"
    try:
        shutil.copy(source_path, local_video)
        compute_motion_curve_gpu(str(local_video), scenes)
    except torch.cuda.OutOfMemoryError:
        print(f"❌ {video_id}: out of GPU memory at BATCH_SIZE={BATCH_SIZE}. "
              f"Lower BATCH_SIZE at the top of this script and retry.")
        torch.cuda.empty_cache()
        continue
    except Exception as e:
        print(f"❌ Error processing {video_id}: {e}")
        continue
    finally:
        if local_video.exists():
            local_video.unlink()

    with open(cat_file, "w") as f:
        json.dump(catalogue, f)
    print(f"  Saved.\n")

shutil.rmtree(LOCAL_TEMP_DIR, ignore_errors=True)
print("Motion curve backfill (local GPU) complete.")
