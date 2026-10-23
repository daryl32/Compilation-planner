"""🎛️ Tune for my videos — a cutting mode (step 0) that tunes itself when you lock in your videos (step 2).

  • Step 0: 🎨 Cutting style → 🎛️ Tune for my videos. Until it's tuned the track is cut 🧠 Smart.
  • Step 2: 🔒 Lock in & tune — rec_library.tune_settings tries every style × pace with the ticked
    videos, checks the best with the matcher, and the best cut is used (the runners-up one tap away).
  • Matching: starts with exactly the settings the tune was scored with (TUNED_MATCHING — the Auto
    defaults, one clip a block); change any to override, ↩️ puts them back.

session_state["tune_result"] = {"track_id", "videos" (sorted), "sequential", "use_story",
                                "results": [best first], "pick": index in results}
"""

from __future__ import annotations

import streamlit as st

TUNE_STYLE = "tune"
TUNE_LABEL = "🎛️ Tune for my videos"

# What the tuner's matcher runs with (rec_library.make_matcher — the planner's Auto defaults, one clip
# a block); Matching starts from these after a tune.
TUNED_MATCHING = {
    "matching_mode": "Auto", "split_screen_enabled": False, "min_clips": 1, "max_clips": 1,
    "min_clip_len_sec": 1.0, "allow_same_video": False, "vary_count": False,
    "weight_shape": 2.0, "weight_random": 0.5, "avoid_repeat": True, "weight_spread": 1.5,
    "weight_motion": 2.0, "autofill_seed": 42, "autofill_v2": True, "autofill_preset": "Balanced",
}
TUNED_KEYS = list(TUNED_MATCHING) + ["use_story_map", "sequential_mode"]


def _videos() -> list:
    return sorted(st.session_state.get("committed_selected_videos") or [])


def result():
    """The tune for this track and these videos, or None (not tuned yet / videos changed)."""
    r = st.session_state.get("tune_result")
    if isinstance(r, dict) and r.get("track_id") == st.session_state.get("track_id") and r.get("results"):
        return r
    return None


def stale() -> bool:
    r = result()
    return bool(r) and r["videos"] != _videos()


def choice():
    """(style, pace) the tune picked for these videos, or None."""
    r = result()
    if not r or stale():
        return None
    p = r["results"][min(int(r.get("pick", 0)), len(r["results"]) - 1)]
    return p["style"], int(p["pace"])


def effective() -> tuple:
    """(style, pace) to cut with: the cut_style's own, or — in tune mode — the tuned one (Smart before)."""
    style = st.session_state.get("cut_style", "custom")
    pace = int(st.session_state.get("cut_style_pace", 0))
    if style == TUNE_STYLE:
        return choice() or ("smart", 0)
    return style, pace


def tuned_matching() -> dict:
    r = result() or {}
    return {**TUNED_MATCHING, "use_story_map": bool(r.get("use_story", True)),
            "sequential_mode": bool(r.get("sequential", False))}


def apply_matching() -> None:
    """Matching gets exactly the settings the tune was scored with."""
    for k, v in tuned_matching().items():
        st.session_state[k] = v


def changed_keys() -> list:
    """Tuned matching settings you've changed since (for the ✏️ line in Matching)."""
    if not result() or st.session_state.get("cut_style") != TUNE_STYLE:
        return []
    return [k for k, v in tuned_matching().items() if st.session_state.get(k) != v]


def run(progress=None) -> str | None:
    """Tune for the locked-in videos; apply the best cut and the tuned matching settings. Error or None."""
    import rec_library as RL
    tid, vids = st.session_state.get("track_id"), _videos()
    if not tid or not vids:
        return "Pick your videos first."
    seq = bool(st.session_state.get("sequential_mode"))
    use_story = bool(st.session_state.get("use_story_map", True))
    try:
        res = RL.tune_settings(tid, vids, seq, use_story=use_story, progress=progress)
    except Exception as e:      # noqa: BLE001 — shown to the user
        return str(e)
    if not res.get("results"):
        return "Nothing could be checked with these videos."
    st.session_state["tune_result"] = {"track_id": tid, "videos": vids, "sequential": seq, "use_story": use_story,
                                       "results": res["results"], "tried": res.get("tried"), "pick": 0}
    pick(0)
    apply_matching()
    return None


def pick(i: int) -> None:
    """Use the i-th tuned way of cutting (re-applies the style's dials)."""
    r = st.session_state.get("tune_result")
    if not r:
        return
    r["pick"] = int(i)
    from planner.audio import _apply_cut_style
    _apply_cut_style()


def label(res: dict) -> str:
    rk = res.get("ranked") or res.get("scores") or {}
    bits = [f"Overall {rk['overall']:.0f}%"] if rk.get("overall") is not None else []
    return f"{res['label']}" + (f" · {' · '.join(bits)}" if bits else "")
