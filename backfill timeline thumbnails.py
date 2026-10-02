"""
Backfill: one thumbnail per second for every video, packed into a single
sprite sheet per video — not thousands of individual files.

Used by the planner's candidate-grid thumbnail display (the single frame
nearest-at-or-after the trimmed clip's actual start).

Run in Colab. Safe to re-run — skips any video that already has
timeline_thumbnails in its catalogue AND whose sprite file still exists.
"""

import os
import json
import shutil
import math
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

CATALOGUE_DIR = Path("/content/drive/MyDrive/scene-labeling/catalogue")
SPRITE_DIR = CATALOGUE_DIR / "timeline_sprites"

SPRITE_DIR.mkdir(parents=True, exist_ok=True)

catalogue_files = sorted(
    f for f in CATALOGUE_DIR.glob("*.json")
    if f.name not in ("training_data.jsonl", "video_index.json")
)
print(f"Found {len(catalogue_files)} catalogues.\n")

INTERVAL_SEC = 1.0
TILE_WIDTH = 120
COLUMNS = 20
JPEG_QUALITY = 82


def build_sprite(video_path: str, sprite_path: Path) -> dict:
    """Sequential grab()/retrieve() scan — avoids per-second seeking (which
    redecodes from the nearest keyframe every time). next_target_sec tracks
    TRUE elapsed time so thumbnails don't drift at non-integer frame rates
    (29.97/23.976fps). Returns the timeline_thumbnails metadata dict, or
    None if the video couldn't be read at all."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = frame_count / fps if fps else 0.0
    if duration <= 0:
        cap.release()
        return None

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        return None
    src_h, src_w = first_frame.shape[:2]
    tile_h = max(1, round(TILE_WIDTH * src_h / src_w))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind after aspect-ratio probe

    n_tiles = max(1, int(duration // INTERVAL_SEC) + 1)
    rows = math.ceil(n_tiles / COLUMNS)
    sprite = Image.new("RGB", (TILE_WIDTH * COLUMNS, tile_h * rows), (20, 20, 20))

    frame_idx = 0
    tile_idx = 0
    next_target_sec = 0.0
    while tile_idx < n_tiles:
        if not cap.grab():
            break
        current_sec = frame_idx / fps
        if current_sec >= next_target_sec:
            ok, frame = cap.retrieve()
            if ok:
                frame = cv2.resize(frame, (TILE_WIDTH, tile_h))
                tile_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                r, c = divmod(tile_idx, COLUMNS)
                sprite.paste(tile_img, (c * TILE_WIDTH, r * tile_h))
            tile_idx += 1
            next_target_sec = tile_idx * INTERVAL_SEC
        frame_idx += 1
    cap.release()

    sprite.save(sprite_path, "JPEG", quality=JPEG_QUALITY)
    return {
        "sprite_path": str(sprite_path), "tile_width": TILE_WIDTH,
        "tile_height": tile_h, "columns": COLUMNS,
        "count": n_tiles, "interval_sec": INTERVAL_SEC,
    }


for cat_file in tqdm(catalogue_files, desc="Videos"):
    with open(cat_file) as f:
        catalogue = json.load(f)

    video_id = catalogue["video_id"]
    sprite_path = SPRITE_DIR / f"{video_id}.jpg"

    if catalogue.get("timeline_thumbnails") and sprite_path.exists():
        print(f"{video_id}: already has timeline thumbnails, skipping.")
        continue

    source_path = catalogue["source_path"]
    if not os.path.exists(source_path):
        print(f"⚠️  {video_id}: source video not found at {source_path}, skipping.")
        continue

    print(f"Building timeline thumbnails for {video_id}...")
    local_video = f"/dev/shm/tmp_thumb_{video_id}{Path(source_path).suffix}"
    try:
        shutil.copy(source_path, local_video)
        meta = build_sprite(local_video, sprite_path)
        if meta is None:
            print(f"⚠️  {video_id}: couldn't read any frames, skipping.")
            continue
        catalogue["timeline_thumbnails"] = meta
    except Exception as e:
        print(f"❌ Error processing {video_id}: {e}")
        continue
    finally:
        if os.path.exists(local_video):
            os.remove(local_video)

    with open(cat_file, "w") as f:
        json.dump(catalogue, f)
    size_kb = sprite_path.stat().st_size // 1024
    print(f"  Saved ({meta['count']} thumbnails, {math.ceil(meta['count']/COLUMNS)}x{COLUMNS} grid, {size_kb}KB).\n")

print("Timeline thumbnail backfill complete.")
