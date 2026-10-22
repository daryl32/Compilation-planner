"""💾 Saved projects (save / load / list) and ♻️ autosave & restore — and applying a track's audio profile.

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

import hashlib
import json
import re
import datetime
from pathlib import Path
import streamlit as st
import story_map as STM
import core.autotune as CAT
from core.segmentation import SEG_ADAPTIVE
from core.autotune import AUDIO_STYLES, AUDIO_STYLE_CUSTOM
from config import PROJECTS_DIR
from page_setup import (push_file_with_oauth, OAUTH_AVAILABLE as _OAUTH_AVAILABLE, DRIVE_SESSION_KEY as _OAUTH_SESSION_KEY)

from planner.state import (
    AUDIO_SETTINGS_KEYS, _bump_cut_edit_rev, DIAL_DEFAULTS, _DRIVE_SYNC_AVAILABLE, MAX_SIMULTANEOUS_CLIPS,
    sanitize_filename, SECTION_MATCH, STORY_KEY
)
from planner.data import (
    load_track
)


# ---------------------------------------------------------------------------
# Project save / load — captures every setting plus matching progress (for
# all three matching modes at once, regardless of which is currently active)
# into a single JSON file, so a session can be closed and resumed later.
# ---------------------------------------------------------------------------

PROJECT_FILE_VERSION = 1

# Plain scalar / list-of-strings keys — copied straight across, no conversion needed.
PROJECT_SIMPLE_KEYS = [
    "track_id", "matching_mode", "max_shape_matches", "max_chor_videos", "allow_same_video",
    "sequential_mode", "prefer_intro_outro", "intro_outro_exclusive", "use_story_map", "story_reserve",
    *AUDIO_SETTINGS_KEYS,
    "split_screen_enabled", "min_clips", "max_clips", "clip_count_skip", "density_contrast",
    "vary_count", "count_seed", "min_clip_len_sec",
    "weight_shape", "weight_random", "avoid_repeat", "weight_spread", "weight_motion",
    "weight_position", "autofill_seed", "autofill_v2", "autofill_preset",
    "ranking_weights", "ranking_experimental", "random_top_seed",
    "committed_selected_videos", "committed_tag_filter", "known_extra_videos", "extra_videos",
    "adv_current_block", "chor_current_block",
    "intro_block_indices", "outro_block_indices",
    "story_map",
    "cut_edits",
]


def _build_project_save_dict() -> dict:
    """Gathers every setting and all matching progress (Auto/Manual/Choreography
    alike, whichever is or isn't currently active) into one JSON-safe dict."""
    data = {"_version": PROJECT_FILE_VERSION, "_saved_at": datetime.datetime.now().isoformat(timespec="seconds")}

    for k in PROJECT_SIMPLE_KEYS:
        if k in st.session_state:
            data[k] = st.session_state[k]

    # Tuple -> list
    if "ramp_range" in st.session_state:
        data["ramp_range"] = list(st.session_state["ramp_range"])

    # {video_id: (start, end)} -> {video_id: [start, end]}
    # {video_id: (start, end) | None} -> {video_id: [start, end] | None}
    # Overrides only — library ranges live in library_meta.json. None = "full video".
    data["video_time_ranges"] = {
        vid: (list(rng) if rng else None)
        for vid, rng in st.session_state.get("video_time_ranges", {}).items()
    }

    # {int: [pick dict, ...]} -> {str(int): [...]} (pick dicts are already JSON-safe)
    data["adv_confirmed"] = {str(k): v for k, v in st.session_state.get("adv_confirmed", {}).items()}
    data["chor_confirmed"] = {str(k): v for k, v in st.session_state.get("chor_confirmed", {}).items()}

    # {int: [{"video_id":.., "scene_id":..}, ...]} -> {str(int): [...]}
    data["adv_skips"] = {str(k): v for k, v in st.session_state.get("adv_skips", {}).items()}

    # {int: {video_id: (scene_id, offset_sec)}} -> {str(int): {video_id: [scene_id, offset_sec]}}
    data["chor_overrides"] = {
        str(seg_idx): {vid: list(pair) for vid, pair in overrides.items()}
        for seg_idx, overrides in st.session_state.get("chor_overrides", {}).items()
    }

    # {int: set((video_id, scene_id))} -> {str(int): [[video_id, scene_id], ...]}
    data["segment_exclusions"] = {
        str(seg_idx): [list(pair) for pair in excl]
        for seg_idx, excl in st.session_state.get("segment_exclusions", {}).items()
    }

    # {int: int} -> {str(int): int}
    data["segment_clip_reduction"] = {
        str(k): v for k, v in st.session_state.get("segment_clip_reduction", {}).items()
    }

    # set((video_id, scene_id)) -> [[video_id, scene_id], ...]
    data["global_excluded_scenes"] = [list(pair) for pair in st.session_state.get("global_excluded_scenes", set())]

    # Individual tick checkboxes for both matching grids — plain bools, scanned by prefix.
    # Restoring these (rather than only the confirmed/override dicts) preserves ticks made
    # for the CURRENT, not-yet-confirmed block too, not just already-confirmed ones.
    data["adv_pick_state"] = {
        k: v for k, v in st.session_state.items() if k.startswith("adv_pick_") and isinstance(v, bool)
    }
    data["chor_pick_state"] = {
        k: v for k, v in st.session_state.items() if k.startswith("chor_pick_") and isinstance(v, bool)
    }

    return data


def _apply_project_load_dict(data: dict) -> list:
    """Writes a loaded project dict back into session_state, converting JSON-safe
    shapes back to their Python originals. Returns a list of warning strings for
    anything that couldn't be applied cleanly (never raises)."""
    warnings = []

    for k in PROJECT_SIMPLE_KEYS:
        if k in data:
            st.session_state[k] = data[k]
    st.session_state.pop("role_clips_mode", None)       # worked out again from the project's own settings
    if "cut_edits" not in data:
        st.session_state.pop("cut_edits", None)       # a project without hand edits
    _bump_cut_edit_rev()                               # the ✂️ editor reloads its cuts
    # Projects saved before cutting paces existed keep their exact dials.
    if "audio_style" not in data:
        st.session_state["audio_style"] = AUDIO_STYLE_CUSTOM
    st.session_state["_audio_profile_for"] = _audio_profile_key(st.session_state.get("track_id"))

    if "ramp_range" in data:
        try:
            st.session_state["ramp_range"] = tuple(data["ramp_range"])
        except (TypeError, ValueError):
            warnings.append("Couldn't restore the intensity ramp range — left at its current value.")

    try:
        st.session_state["video_time_ranges"] = {
            vid: (tuple(rng) if rng else None) for vid, rng in data.get("video_time_ranges", {}).items()
        }
    except (TypeError, ValueError):
        warnings.append("Couldn't restore video time ranges — left at their current values.")

    try:
        st.session_state["adv_confirmed"] = {int(k): v for k, v in data.get("adv_confirmed", {}).items()}
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Manual step-through's confirmed blocks.")
        st.session_state["adv_confirmed"] = {}

    try:
        st.session_state["chor_confirmed"] = {int(k): v for k, v in data.get("chor_confirmed", {}).items()}
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Choreography's confirmed blocks.")
        st.session_state["chor_confirmed"] = {}

    try:
        st.session_state["adv_skips"] = {int(k): v for k, v in data.get("adv_skips", {}).items()}
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Manual step-through's skip history.")
        st.session_state["adv_skips"] = {}

    try:
        st.session_state["chor_overrides"] = {
            int(seg_idx): {vid: tuple(pair) for vid, pair in overrides.items()}
            for seg_idx, overrides in data.get("chor_overrides", {}).items()
        }
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Choreography's clip overrides.")
        st.session_state["chor_overrides"] = {}

    try:
        st.session_state["segment_exclusions"] = {
            int(seg_idx): {tuple(pair) for pair in excl}
            for seg_idx, excl in data.get("segment_exclusions", {}).items()
        }
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Auto mode's per-segment Swap/Reject exclusions.")
        st.session_state["segment_exclusions"] = {}

    try:
        st.session_state["segment_clip_reduction"] = {
            int(k): v for k, v in data.get("segment_clip_reduction", {}).items()
        }
    except (TypeError, ValueError):
        warnings.append("Couldn't restore Auto mode's per-segment Remove counts.")
        st.session_state["segment_clip_reduction"] = {}

    try:
        st.session_state["global_excluded_scenes"] = {tuple(pair) for pair in data.get("global_excluded_scenes", [])}
    except (TypeError, ValueError):
        warnings.append("Couldn't restore globally rejected clips.")
        st.session_state["global_excluded_scenes"] = set()

    for k, v in data.get("adv_pick_state", {}).items():
        st.session_state[k] = v
    for k, v in data.get("chor_pick_state", {}).items():
        st.session_state[k] = v


    # One-shot, per-mode suppression so the Auto/Manual/Choreography "settings changed,
    # reset progress" guards don't immediately wipe out what was just loaded the first
    # time each mode happens to be visited after this. Each mode consumes its own entry
    # from this set the first time its own guard runs, then behaves normally again —
    # so a later, genuine settings change still resets progress as expected.
    st.session_state["_project_load_pending_modes"] = {"Auto", "Manual step-through", "Choreography"}

    return warnings


def save_project(name: str) -> tuple[Path, str | None]:
    """Writes the current session's full settings + progress to
    PROJECTS_DIR/<sanitized name>.json and uploads to Drive if authenticated.
    Returns (path, drive_error_or_None)."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROJECTS_DIR / f"{sanitize_filename(name)}.json"
    out_path.write_text(json.dumps(_build_project_save_dict(), indent=2))
    drive_err = None
    if _OAUTH_AVAILABLE and st.session_state.get(_OAUTH_SESSION_KEY):
        drive_err = push_file_with_oauth(
            st.session_state[_OAUTH_SESSION_KEY], out_path, "scene-labeling/projects"
        )
    return out_path, drive_err


def load_project(path: Path) -> list:
    """Reads a project file and applies it to session_state. Returns a list
    of warning strings (empty if everything restored cleanly)."""
    data = json.loads(path.read_text())
    return _apply_project_load_dict(data)


def _canon(value):
    """Lists of pairs (saved sets, e.g. excluded clips) in a fixed order — sets have none."""
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [_canon(v) for v in value]
        if items and all(isinstance(v, list) for v in items):
            items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return items
    return value


def _normalise(data: dict) -> dict:
    """JSON round-trip (tuples → lists, int keys → str, sets → sorted lists) so saved and
    current compare like for like."""
    return _canon(json.loads(json.dumps(data, default=list)))


def _remember_project(name: str, data: dict) -> None:
    """Track this project as the one being worked on (for prefill and 'unsaved changes')."""
    st.session_state["_loaded_project"] = {"name": name, "data": _normalise(data)}


def _project_changes(loaded: dict) -> int:
    """How many saved areas of the remembered project differ from the current session.
    Only what the file holds is compared — settings it never had (e.g. defaults added
    since) don't count — and for tick boxes only the ticked ones matter."""
    saved = loaded["data"]
    current = _normalise(_build_project_save_dict())
    changed = 0
    for key, value in saved.items():
        if key.startswith("_"):
            continue
        now = current.get(key)
        if key.endswith("_pick_state"):
            value = {k for k, v in (value or {}).items() if v}
            now = {k for k, v in (now or {}).items() if v}
        if now != value:
            changed += 1
    return changed


def _do_save_project(name: str) -> None:
    out_path, drive_err = save_project(name)
    _remember_project(out_path.stem, _build_project_save_dict())
    st.session_state["_prefill_project_name"] = out_path.stem
    if drive_err:
        notice = ("warning", f"Saved locally but Drive upload failed: {drive_err}")
    elif _DRIVE_SYNC_AVAILABLE:
        notice = ("success", f"Saved and uploaded to Drive: {out_path.name}")
    else:
        notice = ("success", f"Saved as {out_path.name}")
    st.session_state["_project_notices"] = [notice]
    st.rerun()


def list_saved_projects() -> list:
    if not PROJECTS_DIR.exists():
        return []
    return sorted((p.stem for p in PROJECTS_DIR.glob("*.json")), key=str.lower)


def _consume_load_suppression(mode_name: str) -> bool:
    """True if this mode's progress-reset guard should be suppressed because a
    project was just loaded and this mode hasn't had a chance to "see" the
    restored state yet. Consumes (removes) this mode's entry so the suppression
    only applies once — a later, genuine settings change still resets that
    mode's progress normally, same as before any project was ever loaded."""
    pending = st.session_state.get("_project_load_pending_modes")
    if pending and mode_name in pending:
        pending.discard(mode_name)
        st.session_state["_project_load_pending_modes"] = pending
        return True
    return False


def _dial_value(k):
    """A dial's current value (its default if it hasn't been set)."""
    return st.session_state.get(k, DIAL_DEFAULTS.get(k))


def _method_cfg(method: str, settings: dict, bpb: int) -> dict:
    return CAT._method_cfg(method, settings, bpb, lookup=_dial_value)


def auto_tune_parameters(track: dict, min_clips: int, max_clips: int, method: str = None,
                         style_name: str = "Balanced", beats_per_bar: int = None) -> tuple:
    """core.autotune.auto_tune_parameters for the current method / beats per bar."""
    method = method or st.session_state.get("segmentation_method", SEG_ADAPTIVE)
    bpb = int(beats_per_bar or st.session_state.get("beats_per_bar", 4))
    return CAT.auto_tune_parameters(track, min_clips, max_clips, method=method, style_name=style_name,
                                    beats_per_bar=bpb, lookup=_dial_value)


def _audio_profile_key(track_id) -> tuple:
    return (track_id, st.session_state.get("segmentation_method"), st.session_state.get("audio_style"),
            int(st.session_state.get("beats_per_bar", 4)))


def apply_audio_profile(track: dict, track_id: str, report: bool = False) -> None:
    """Set this track's segmentation dials for the current method and cutting
    pace (and split-screen shape). Keeps the app-scope copy in step."""
    style = st.session_state.get("audio_style", "Balanced")
    if style not in AUDIO_STYLES:
        style = "Balanced"
    mn = int(st.session_state.get("min_clips", 1))
    mx = max(int(st.session_state.get("max_clips", 4)), mn + 1, 2)
    settings, why = auto_tune_parameters(track, mn, mx, style_name=style)
    for k, v in settings.items():
        try:
            st.session_state[k] = v
        except Exception:
            pass   # a widget already drawn this run keeps its value until the next run
    st.session_state["_audio_profile_for"] = _audio_profile_key(track_id)
    if report:
        st.session_state["auto_tune_report"] = why


def ensure_audio_profile(track: dict, track_id: str) -> None:
    """With a cutting pace chosen (not Custom), the dials follow the track,
    method and beats per bar: re-tuned whenever any of them changes."""
    if st.session_state.get("audio_style", "Balanced") == AUDIO_STYLE_CUSTOM:
        return
    if st.session_state.get("_audio_profile_for") != _audio_profile_key(track_id):
        apply_audio_profile(track, track_id)


def _run_auto_tune():
    """🎯 Auto-detect: re-tune the current method for this track at the chosen
    pace (Balanced if the pace is Custom)."""
    if st.session_state.get("audio_style") not in AUDIO_STYLES:
        st.session_state["audio_style"] = "Balanced"
    t = load_track(st.session_state["track_id"])
    st.session_state["split_screen_enabled"] = True
    st.session_state["max_clips"] = min(MAX_SIMULTANEOUS_CLIPS, max(int(st.session_state.get("max_clips", 4)),
                                                                   int(st.session_state.get("min_clips", 1)) + 1, 2))
    apply_audio_profile(t, st.session_state["track_id"], report=True)


# ---------------------------------------------------------------------------
# Autosave & restore — your work survives the app restarting (a crash, running
# out of memory, a deploy). Every change is written to a recovery file for you
# (PROJECTS_DIR/.autosave/<email>.json — separate from your saved projects).
# A new session moves that file aside and offers to restore it.
# ---------------------------------------------------------------------------

AUTOSAVE_DIR = PROJECTS_DIR / ".autosave"


def _autosave_path(kind: str = "") -> Path:
    who = re.sub(r"[^A-Za-z0-9._-]", "_", str(getattr(st.user, "email", None) or "user"))
    return AUTOSAVE_DIR / f"{who}{kind}.json"


def _progress_summary(state: dict) -> dict:
    def blocks(key):
        return sum(1 for v in (state.get(key) or {}).values() if v)
    return {"track_id": state.get("track_id"), "mode": state.get("matching_mode"),
            "videos": len(state.get("committed_selected_videos") or []),
            "auto_blocks": len(state.get("segment_exclusions") or {}),
            "manual_blocks": blocks("adv_confirmed"), "chor_blocks": blocks("chor_confirmed")}


def autosave_now() -> None:
    """Write this session's state to the recovery file — only when it changed."""
    if not st.session_state.get("_autosave_started"):
        return
    try:
        state = _normalise(_build_project_save_dict())
        state.pop("_saved_at", None)
        blob = json.dumps(state, sort_keys=True, default=str)
    except Exception:
        return
    digest = hashlib.sha1(blob.encode()).hexdigest()
    if digest == st.session_state.get("_autosave_hash"):
        return
    payload = {
        "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": _progress_summary(state),
        "loaded_project": st.session_state.get("_loaded_project"),
        "project_name": (st.session_state.get("project_save_name") or "").strip() or None,
        "state": state,
    }
    try:
        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = _autosave_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(path)
        st.session_state["_autosave_hash"] = digest
    except OSError:
        pass


def _start_autosave_session() -> None:
    """First run of a new session: the recovery file left by the last session is
    moved aside (so this session's autosave can't overwrite it) and offered."""
    if st.session_state.get("_autosave_started"):
        return
    st.session_state["_autosave_started"] = True
    path, recover = _autosave_path(), _autosave_path(".recover")
    try:
        if path.exists():
            data = json.loads(path.read_text())
            s = data.get("summary") or {}
            worth_it = s.get("track_id") and (s.get("videos") or s.get("manual_blocks") or s.get("chor_blocks")
                                              or s.get("auto_blocks"))
            if worth_it:
                path.replace(recover)
            else:
                path.unlink()
    except (OSError, ValueError):
        pass


def _restore_choice(choice: str) -> None:
    recover = _autosave_path(".recover")
    if choice == "restore":
        try:
            st.session_state["_restore_autosave"] = json.loads(recover.read_text())
        except (OSError, ValueError):
            st.session_state["_project_notices"] = [("error", "Couldn't read the unsaved work.")]
            return
    try:
        recover.replace(_autosave_path(".discarded"))   # one step of undo, just in case
    except OSError:
        pass


def _new_project() -> None:
    """🆕 New project: back to a blank planner (every setting at its default, no
    track work, no project loaded). What was on screen is kept as a one-step
    backup so it can be brought back."""
    autosave_now()
    try:
        cur = _autosave_path()
        if cur.exists():
            cur.replace(_autosave_path(".before_new"))
    except OSError:
        pass
    from page_setup import session_keep_keys
    keep = session_keep_keys() | {"_autosave_started"}
    name = (st.session_state.get("new_project_name") or "").strip()
    for k in list(st.session_state.keys()):
        if k not in keep:
            try:
                del st.session_state[k]
            except Exception:
                pass
    st.cache_data.clear()
    if name:
        st.session_state["_prefill_project_name"] = sanitize_filename(name)
    st.session_state["_new_project_notice"] = name or True


def _undo_new_project() -> None:
    backup = _autosave_path(".before_new")
    try:
        st.session_state["_restore_autosave"] = json.loads(backup.read_text())
        backup.unlink()
    except (OSError, ValueError):
        st.session_state["_project_notices"] = [("error", "Couldn't find what you had before.")]


def _open_recommended_plan(plan: dict) -> None:
    """🔮 Recommendations → 🎬 Open as project: a new project (what was on screen is
    autosaved and kept as the 🆕 undo backup) with the plan's track, cut settings and
    videos, on the Matching step in Auto mode — the plan the Recommendations page scored."""
    _new_project()
    st.session_state.pop("_new_project_notice", None)
    ss = st.session_state
    ss["track_id"] = plan["track_id"]
    for k, v in plan["settings"].items():
        ss[k] = v
    ss["audio_style"] = AUDIO_STYLE_CUSTOM          # these settings exactly — don't re-tune over them
    ss["cut_style"] = "custom"                      # 🎨 style row shows Custom (the plan's own settings)
    ss["committed_selected_videos"] = list(plan["videos"])
    for vid in plan["videos"]:
        ss[f"select_video_{vid}"] = True
    ss["matching_mode"] = "Auto"
    ss["sequential_mode"] = bool(plan.get("sequential"))
    # The story map exactly as the plan was scored: on with the regions it used, or off.
    _story = bool(plan.get("story")) and bool(plan.get("story_regions"))
    ss["use_story_map"] = _story
    if _story:
        ss[STORY_KEY] = {"track_id": plan["track_id"], "regions": plan["story_regions"], "v": 1, "edited": False,
                         "source": "default" if plan["story_regions"] == STM.saved_map(plan["track_id"]) else "project"}
    ss["ui_section"] = SECTION_MATCH
    _story_note = (" It uses the 🎭 story map it was scored with." if _story else
                   " The 🎭 story map is off so the plan is exactly the one recommended — turn it on under "
                   "🎛️ Mode if you like.")
    ss["_project_notices"] = [("success", f"🔮 Opened {plan.get('label') or 'the recommended plan'} as a new "
                                          "project." + _story_note)]
    ss["_rec_opened"] = True
    ss["_match_pressed"] = True                     # you asked for this plan — build it without ▶️ Match

