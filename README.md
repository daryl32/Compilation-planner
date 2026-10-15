# Media Planner Webapp

A Streamlit app for building music-synced video compilations from a personal video library. Part of a three-stage pipeline:

1. **Google Colab** — ML processing: scene detection, CLIP/Qwen2.5-VL tagging, audio analytics. Outputs catalogue JSONs and audio track JSONs to Google Drive.
2. **This app** (Streamlit, entry point `Home.py`) — Media Library, Reviewer, Compilation Planner and Server pages. The planner matches video clips to a music track's beat/energy timeline and exports a plan JSON.
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
streamlit run Home.py
```

---

## Project structure

```
Home.py                         # Entry point — links to the pages below
pages/1_Compilation_Planner.py  # Match clips to a track, render previews, export plans
pages/2_Reviewer.py             # Tags, exclude/intro/outro, ⭐ thumbnail, split / combine / ⚡ auto split
pages/3_Media_Library.py        # Videos (library ranges), Audio, Renders, Analytics
pages/4_Server.py               # Disk usage and clean-up

page_setup.py                   # Sign-in, whitelist, and the sidebar every page shares
catalogue_store.py              # THE way to read/write video catalogues (shared, refreshed per file)
library_common.py               # Library meta (ranges, ⭐), Drive "pending" list, memory guard, helpers
scene_edits.py                  # Split / combine / undo, suggestions, auto split, motion for new parts
storage.py                      # What's on the server's disk; clearing temporary files
auto_sync.py / drive_sync.py    # Drive → server sync (12-hourly timer + Sync now), then auto clean-up
drive_oauth.py                  # Personal Drive writes (save plans, projects, edits)
render_preview.py / renders.py  # Render a plan to MP4; saved renders and their details
video_ranking.py, similarity.py, cut_scoring.py, audio_analysis.py, audio_charts.py, preview_snippets.py
backfill *.py, make_proxies.py  # Colab / laptop processing scripts
deploy/                         # Server scripts (see deploy/README.md)
config.py                       # YOUR local config — gitignored, not committed
config_example.py               # Template — copy to config.py and fill in
```

### Data: what lives where

| Data | Where | On the server |
|---|---|---|
| Video catalogues, thumbnails, timeline sprites, audio JSONs | Drive `scene-labeling/…` | Synced copy (small, read constantly) |
| 480p preview copies (proxies) | Drive `scene-labeling/proxies` | Synced copy (fast scrubbing / frames) |
| Original videos and music | Drive (`Videos-PH/`, `Audio-Library/`) | **Temporary** — streamed for renders; anything downloaded is cleared after `KEEP_DOWNLOADS_DAYS` unused (default 3) |
| Renders | Drive `scene-labeling/previews` | Server copy can be cleared once uploaded |
| Projects, plans, library meta | Drive | Copy, pushed with Save |

Code that needs "something about every video" should use `catalogue_store.video_index()`
(slim: times, tags, flags, motion score) or `all_catalogues()` (full, no motion curves) —
never glob and json-load the catalogue folder itself.

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
