"""The Compilation Planner page's building blocks (pages/1_Compilation_Planner.py imports them).

  state       Shared constants, settings and small helpers (defaults, step names, reruns, Drive sync availability).
  data        Loading tracks, catalogues, motion curves and thumbnails (cached), plus footage helpers.
  projects    💾 Saved projects (save / load / list) and ♻️ autosave & restore — and applying a track's audio profile.
  audio       🎵 Cutting settings: style row, pace presets, 🎯 Aim for, Smart / Scored dials and the ✂️ cut editor.
  story       🎭 Step 1 — the story map (which parts of the song call for which scene role).
  matching    🎬 Matching helpers: the matcher wrappers, candidate grids, Choreography blocks, swap for similar.
  similar     ✨ More like this — similar videos and clips, video ranking and the video list's thumbnails.
  videos      📼 Step 2 — choosing the videos (and their time ranges).
  audio_step  🎵 Step 0 — the Audio Settings step (Advanced and Simple) with its live chart.

Each module imports only from the ones listed above it (no import cycles).
"""
