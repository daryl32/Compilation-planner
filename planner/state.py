"""Shared constants, settings and small helpers (defaults, step names, reruns, Drive sync availability).

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

import re
import time
import streamlit as st
import cut_scoring as CS
import smart_cuts as SM

# st.fragment (Streamlit 1.37+; was st.experimental_fragment in 1.33-1.36) lets
# part of the page rerun on its own instead of the whole script re-executing on
# every click — used below so "Use this clip" / "View fit" don't have to redo
# everything above them (sidebar, chart, track/catalogue loading) just to
# update the candidate grid. Falls back to a no-op on an older Streamlit —
# the page still works, it just won't get this speedup until you upgrade.
if hasattr(st, "fragment"):
    fragment = st.fragment
elif hasattr(st, "experimental_fragment"):
    fragment = st.experimental_fragment
else:
    fragment = lambda f: f


def rerun_fragment():
    """Rerun only the fragment this is called from. A bare st.rerun() defaults to
    scope="app" — a FULL page rerun even when called inside a fragment — which is
    what made every range-picker click reload the whole page. Falls back to a plain
    rerun on a Streamlit too old for scope=, or when not actually inside a fragment
    (fragment is a no-op decorator on old versions)."""
    from streamlit.errors import StreamlitAPIException
    try:
        st.rerun(scope="fragment")
    except (TypeError, StreamlitAPIException):
        st.rerun()


def rerun_full():
    """A rerun called from INSIDE a fragment normally only reruns that
    fragment, not the whole page — right for View Fit / checkboxes, which
    only affect what's shown, but wrong for Skip Ahead, which changes which
    footage is available and needs the search above the fragment to redo.
    scope="app" forces a full rerun even from inside a fragment; older
    Streamlit versions don't have that parameter, so fall back to a plain
    rerun (harmless there since such versions don't have fragments to
    escape from anyway — the fallback IS a full rerun already)."""
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()

# Google Drive sync — graceful fallback if credentials not present
try:
    from drive_sync import (sync_pull, push_file_to_drive, push_directory_to_drive,  # noqa: F401  (used by other modules)
                            credentials_available, download_source_video)
    _DRIVE_SYNC_AVAILABLE = credentials_available()
except ImportError:
    _DRIVE_SYNC_AVAILABLE = False


def sanitize_filename(name: str) -> str:
    """Strip characters that aren't valid in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


AUDIO_SETTINGS_KEYS = [
    "segmentation_method", "beats_per_bar", "audio_style", "sc_focus", "cut_style", "cut_style_pace",
    "change_window_secs", "change_threshold", "min_segment_sec", "max_segment_sec",
    "react_to_hits", "hit_threshold", "snap_secs",
    "fast_energy_thresh", "fast_max_bars", "fast_cooldown_bars",
    "phrase_lock_bars", "cinematic_bars", "dynamic_priority_override", "override_delta_thresh",
    "override_clear_secs", "on_beat",
    *CS.DEFAULTS.keys(),
    *SM.DEFAULTS.keys(),
]


# ✂️ Hand-edited cuts (planner/audio.py): bumped when the cut list changes from outside the editor.

def _bump_cut_edit_rev() -> None:
    """Tell the editor the cut list changed from here (load, reset…), not from its own edits."""
    st.session_state["cut_edit_rev"] = int(time.time() * 1000) % 2_000_000_000


# The planner's steps, and where the 🎭 story map is kept in session_state (planner/story.py).
STORY_KEY = "story_map"   # {"track_id", "regions": [...], "v": edit counter, "edited": bool,
                          #  "source": "default" | "named:<name>" | "project" — which map this project uses}
SECTION_AUDIO = "🎵 0. Audio Settings"
SECTION_STORY = "🎭 1. Story map"
SECTION_VIDEOS = "📼 2. Videos & Time Ranges"
SECTION_MATCH = "🎬 3. Matching & Export"
SECTIONS = [SECTION_AUDIO, SECTION_STORY, SECTION_VIDEOS, SECTION_MATCH]

# ---------------------------------------------------------------------------
# Dial defaults — every setting's starting value. Each is keyed in session_state (kept between
# runs by settings_store.keep), so Auto-detect / styles can set them all at once from a callback.
# ---------------------------------------------------------------------------

MAX_SIMULTANEOUS_CLIPS = 8   # matches the VSE add-on's layouts (up to SEVEN_UP / 4x2 grid)

DIAL_DEFAULTS = {
    # Segmentation — Adaptive
    "segmentation_method": "Smart (self-tuning)",   # = SEG_SMART
    "audio_style": "Balanced",    # cutting pace preset; dials follow the track unless Custom
    "cut_style": "custom",        # 🎨 cutting style (core/styles.py); "custom" = choose the method yourself
    "cut_style_pace": 0,
    "sc_focus": "Balanced",       # Scored cuts: "What to cut on" mix
    "beats_per_bar": 4,
    "change_window_secs": 2.0,
    "change_threshold": 0.12,
    "min_segment_sec": 2.0,
    "max_segment_sec": 8.0,
    "react_to_hits": True,
    "hit_threshold": 0.9,
    "snap_secs": 0.3,
    # Segmentation — Rhythm Engine
    "fast_energy_thresh": 0.75,
    "fast_max_bars": 8,
    "fast_cooldown_bars": 4,
    "phrase_lock_bars": 4,
    "cinematic_bars": 4,
    "dynamic_priority_override": False,
    "override_delta_thresh": 0.15,
    "override_clear_secs": 1.0,
    "on_beat": True,              # 🥁 Adaptive / Rhythm Engine / Scored: every cut on the beat
    # Segmentation — Scored cuts
    **CS.DEFAULTS,
    # Segmentation — Smart
    **SM.DEFAULTS,
    "sm_pace": int(SM.DEFAULTS["sm_pace"]),   # a step on the Pace slider (-2 … 2): kept as a whole number
    # Split screen & intensity
    "split_screen_enabled": False,
    "min_clips": 1,
    "max_clips": 4,
    "ramp_range": (0.2, 0.85),
    "density_contrast": 1.0,
    "vary_count": False,
    "count_seed": 0,
    "min_clip_len_sec": 1.0,
    # Choreography mode
    "max_chor_videos": 4,
    "max_shape_matches": 6,
    "allow_same_video": False,
    "prefer_intro_outro": False,
    "matching_mode": "Auto",
    "sequential_mode": False,
    "use_story_map": True,
    "story_reserve": True,
    "intro_outro_exclusive": False,
    "clip_count_skip": [],
    # Auto-fill weights (used in Matching & Export, stored here for persistence)
    # Defaults = autofill.PRESETS["Balanced"] (projects keep the values they were saved with).
    "weight_shape": 2.0,
    "weight_random": 0.5,
    "avoid_repeat": True,
    "weight_spread": 1.5,
    "weight_motion": 2.0,
    "weight_position": 1.0,
    "autofill_seed": 42,
    "autofill_v2": True,          # ✨ Improved auto-fill (autofill.py); off = the original scoring
    "autofill_preset": "Balanced",
}

