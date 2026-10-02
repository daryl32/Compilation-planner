# Media Planner Webapp

A Streamlit app for building music-synced video compilations from a personal video library. Part of a three-stage pipeline:

1. **Google Colab** — ML processing: scene detection, CLIP/Qwen2.5-VL tagging, audio analytics. Outputs catalogue JSONs and audio track JSONs to Google Drive.
2. **This app (`planner_app.py`)** — interactive Compilation Planner. Matches video clips to a music track's beat/energy timeline and exports a plan JSON.
3. **Blender VSE add-on** (`vse_split_into_blocks.py`) — imports the plan and builds the edit in Blender's Video Sequence Editor.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure local paths

Copy the example config and fill in your Google Drive path:

```bash
cp config.example.py config.py
```

Then edit `config.py` — set `DRIVE_ROOT` to the path where Google Drive is mounted on your machine (e.g. `G:\My Drive` on Windows).

> **Note:** `config.py` is gitignored. Never commit it — it contains your local paths.

Also update `render_preview.py` to import `DRIVE_ROOT` from `config` (see note in that file).

### 3. Run the app

```bash
streamlit run planner_app.py
```

---

## Project structure

```
planner_app.py                  # Main Streamlit app — Compilation Planner
render_preview.py               # Renders a plan JSON to a video preview
backfill_motion_curve.py        # Colab: backfills per-scene motion curves
backfill_motion_curve_gpu.py    # Colab GPU-accelerated variant
backfill_motion_curve_local.py  # Local Windows variant
backfill_timeline_thumbnails.py # Colab: builds per-video sprite sheets
backfill_timeline_thumbnails_local.py
vse_split_into_blocks.py        # Blender VSE add-on
config.py                       # YOUR local config — gitignored, not committed
config.example.py               # Template — copy to config.py and fill in
requirements.txt
```

---

## Google Drive layout

The app reads from and writes to these folders under your Drive root:

```
scene-labeling/
  catalogue/              # Per-video JSON catalogues + thumbnails/ + timeline_sprites/
  audio_catalogue/        # Per-track audio analysis JSONs
  compilation_plans/      # Exported plan JSONs (input to the VSE add-on)
  previews/               # Rendered video previews (temp files)
  projects/               # Saved planner project states
```

---

## Requirements

- Python 3.10+
- Windows recommended (Drive paths and some backfill scripts are Windows-oriented)
- Google Drive for Desktop in **Stream mode**
- For GPU backfill: NVIDIA GPU with CUDA, PyTorch installed separately
