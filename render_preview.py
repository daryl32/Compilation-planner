r"""
Render a quick, low-res preview MP4 from a saved compilation plan JSON.
Trims and (for split-screen segments) grid-combines the matched scenes,
concatenates them in order, and muxes in the track's audio.

Usage:
    python render_preview.py "path\to\plan.json"

Requires ffmpeg on PATH (winget install ffmpeg).
"""

import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

from config import DRIVE_ROOT, CATALOGUE_DIR, AUDIO_DIR

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360


def colab_to_local(colab_path: str) -> str:
    """Convert a Colab-side '/content/drive/MyDrive/...' path to the local
    Drive-synced Windows equivalent."""
    marker = "MyDrive/"
    idx = colab_path.find(marker)
    if idx == -1:
        return colab_path
    rel = colab_path[idx + len(marker):]
    return str(DRIVE_ROOT / rel.replace("/", "\\"))


_video_source_cache = {}


def get_video_source(video_id: str) -> str:
    if video_id not in _video_source_cache:
        with open(CATALOGUE_DIR / f"{video_id}.json") as f:
            cat = json.load(f)
        _video_source_cache[video_id] = colab_to_local(cat["source_path"])
    return _video_source_cache[video_id]


def get_track_source(track_id: str) -> str:
    with open(AUDIO_DIR / f"{track_id}.json") as f:
        track = json.load(f)
    return colab_to_local(track["source_path"])


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install it with: winget install ffmpeg")
        print("Then close and reopen your terminal before retrying.")
        sys.exit(1)


def run(cmd: list):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg command failed:")
        print(" ".join(cmd))
        print(result.stderr[-2000:])
        raise RuntimeError("ffmpeg failed")


# Cell layouts copied from the VSE add-on's _AUTO_LAYOUT, so the preview matches
# what Import Plan + Apply Split-Screen (Layout Mode: Auto) builds in Blender.
# Each cell is (x0, y0, x1, y1) as fractions of the frame, (0,0) = top-left.
def _grid(cols, rows):
    return [(c / cols, r / rows, (c + 1) / cols, (r + 1) / rows) for r in range(rows) for c in range(cols)]

AUTO_LAYOUTS = {
    1: _grid(1, 1),
    2: [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)],                                  # DIPTYCH
    3: [(0.0, 0.0, 1/3, 1.0), (1/3, 0.0, 2/3, 1.0), (2/3, 0.0, 1.0, 1.0)],             # TRIPTYCH
    4: _grid(2, 2),
    5: [(0.0, 0.0, 1/3, 0.5), (1/3, 0.0, 2/3, 0.5), (2/3, 0.0, 1.0, 0.5),              # FIVE_UP
        (0.0, 0.5, 0.5, 1.0), (0.5, 0.5, 1.0, 1.0)],
    6: _grid(3, 2),
    7: [(0.0, 0.0, 0.25, 0.5), (0.25, 0.0, 0.5, 0.5), (0.5, 0.0, 0.75, 0.5),           # SEVEN_UP
        (0.75, 0.0, 1.0, 0.5), (0.0, 0.5, 1/3, 1.0), (1/3, 0.5, 2/3, 1.0), (2/3, 0.5, 1.0, 1.0)],
    8: _grid(4, 2),
}


def _cell_pixels(cell):
    """Fractional cell -> integer (x, y, w, h), with even sizes for libx264."""
    x0, y0, x1, y1 = cell
    x, y = int(round(x0 * PREVIEW_WIDTH)), int(round(y0 * PREVIEW_HEIGHT))
    w = int(round(x1 * PREVIEW_WIDTH)) - x
    h = int(round(y1 * PREVIEW_HEIGHT)) - y
    return x, y, max(2, w - w % 2), max(2, h - h % 2)


def render_segment(entry: dict, idx: int, tmp_dir: Path) -> Path:
    """Render one timeline entry (1-8 simultaneous slots) to a temp clip of
    the segment's duration, no audio. Each slot's chain of clips is joined
    back-to-back, cropped to fill its cell (the add-on's default Crop fit),
    and overlaid onto a black canvas at that cell's position."""
    duration = round(entry["track_time"][1] - entry["track_time"][0], 3)
    slots = entry["scenes"]
    n = len(slots)
    out_path = tmp_dir / f"seg_{idx:04d}.mp4"

    if n == 0:
        run(["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"color=c=black:s={PREVIEW_WIDTH}x{PREVIEW_HEIGHT}:d={duration}:r=25",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", str(out_path)])
        return out_path

    cells = AUTO_LAYOUTS.get(n) or _grid(4, (n + 3) // 4)
    inputs = []
    filter_parts = [f"color=c=black:s={PREVIEW_WIDTH}x{PREVIEW_HEIGHT}:d={duration}:r=25[base0]"]
    input_idx = 0

    for slot_i, slot in enumerate(slots):
        x, y, w, h = _cell_pixels(cells[slot_i])
        link_labels = []
        for link in slot["chain"]:
            src = get_video_source(link["video_id"])
            inputs += ["-ss", str(link["clip_start_sec"]), "-t", str(link["clip_duration_sec"]), "-i", src]
            label = f"c{input_idx}"
            filter_parts.append(
                f"[{input_idx}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},setsar=1,fps=25[{label}]"
            )
            link_labels.append(f"[{label}]")
            input_idx += 1

        if len(link_labels) == 1:
            filter_parts.append(f"{link_labels[0]}null[s{slot_i}]")
        else:
            filter_parts.append(f"{''.join(link_labels)}concat=n={len(link_labels)}:v=1:a=0[s{slot_i}]")

        out_label = "outv" if slot_i == n - 1 else f"base{slot_i + 1}"
        filter_parts.append(f"[base{slot_i}][s{slot_i}]overlay={x}:{y}:eof_action=pass[{out_label}]")

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]", "-an", "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        str(out_path),
    ]
    run(cmd)
    return out_path


def render_plan_dict(plan: dict, output_path: str, progress_callback=None):
    """Core renderer, operating on an already-loaded plan dict. Used both by
    the CLI (render_plan below) and directly by the planner app on its
    in-memory plan, with no need to write/read a JSON file in between.
    progress_callback(done, total), if given, is called after each segment —
    lets a caller (e.g. Streamlit) show progress without depending on tqdm's
    console-only output."""
    check_ffmpeg()

    track_source = get_track_source(plan["track_id"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clip_paths = []
        total = len(plan["timeline"])

        for i, entry in enumerate(tqdm(plan["timeline"], desc="Rendering segments")):
            clip_paths.append(render_segment(entry, i, tmp_dir))
            if progress_callback:
                progress_callback(i + 1, total)

        # Concatenate
        concat_list = tmp_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p.as_posix()}'\n")

        combined_video = tmp_dir / "combined.mp4"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(combined_video)])

        # Mux in the track's audio
        print("Muxing audio...")
        run([
            "ffmpeg", "-y", "-i", str(combined_video), "-i", track_source,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            output_path,
        ])

    print(f"\nDone: {output_path}")


def render_plan(plan_path: str, output_path: str = "preview_output.mp4"):
    with open(plan_path) as f:
        plan = json.load(f)
    render_plan_dict(plan, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python render_preview.py "path\\to\\plan.json"')
        sys.exit(1)
    render_plan(sys.argv[1])
