"""
Backfill: add per-scene frame-by-frame motion curves to catalogues processed
before this feature existed, for the planner's Advanced Shape Matching mode.

Run in Colab (CPU runtime is fine — this script doesn't use GPU).
Safe to re-run — skips any catalogue where every scene already has a
motion_curve.
"""

import os
import json
import shutil
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

CATALOGUE_DIR = Path("/content/drive/MyDrive/scene-labeling/catalogue")

catalogue_files = sorted(
    f for f in CATALOGUE_DIR.glob("*.json")
    if f.name not in ("training_data.jsonl", "video_index.json")
)
print(f"Found {len(catalogue_files)} catalogues.\n")

TARGET_H, TARGET_W = 90, 160


def compute_motion_curve(video_path: str, scenes: list) -> None:
    """Compute a frame-by-frame motion diff curve for the whole video in one
    sequential pass, then slice out each scene's portion. Frames are
    grayscale and downsized before diffing.

    Sets scene['motion_curve'] = {"fps": <float>, "values": [<float>, ...]}
    for every scene dict in `scenes` (each needs start_frame/end_frame)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max(
        (s["end_frame"] for s in scenes), default=0
    )

    all_diffs = np.zeros(max(total_frames, 1), dtype=np.float32)
    prev_gray = None
    frame_idx = 0
    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (TARGET_W, TARGET_H))
        if prev_gray is not None:
            all_diffs[frame_idx] = cv2.absdiff(gray, prev_gray).mean()
        prev_gray = gray
        frame_idx += 1
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

    video_id = catalogue["video_id"]
    scenes = catalogue["scenes"]

    if all("motion_curve" in s and s["motion_curve"] for s in scenes):
        print(f"{video_id}: already has motion curves, skipping.")
        continue

    source_path = catalogue["source_path"]
    if not os.path.exists(source_path):
        print(f"⚠️  {video_id}: source video not found at {source_path}, skipping.")
        continue

    print(f"Computing motion curves for {video_id} ({len(scenes)} scenes)...")
    local_video = f"/dev/shm/tmp_curve_{video_id}{Path(source_path).suffix}"
    try:
        shutil.copy(source_path, local_video)
        compute_motion_curve(local_video, scenes)
    except Exception as e:
        print(f"❌ Error processing {video_id}: {e}")
        continue
    finally:
        if os.path.exists(local_video):
            os.remove(local_video)

    with open(cat_file, "w") as f:
        json.dump(catalogue, f)
    print(f"  Saved.\n")

print("Motion curve backfill complete.")
