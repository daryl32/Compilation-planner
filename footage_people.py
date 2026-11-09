"""
👥 People on screen, from the student export (catalogue/student_scores/<video>.npz "people", written by
`student_tags.py export` from the pose detections) — claude/tag-levels.md:

  • 🚫 "Nobody on screen" exclude suggestions in the Reviewer: title cards, logos, black frames and other junk
    footage have no people. This replaces the text-on-screen tag, which caught every caption.
  • the 👥 cast hint: the most people the pose model sees together (a hint only — it can't tell women from men,
    and misses the cameraman in POV).
"""

from __future__ import annotations

import numpy as np

from library_common import tc_to_seconds

NOBODY_SHARE = 0.8        # a scene where nobody is on screen for ≥ 80% of it → suggest excluding it
MIN_SAMPLES = 2           # at least 1 s of scores in the scene
HINT_MIN_SAMPLES = 8      # cast hint: only scenes with ≥ 4 s of scores count
DISMISS_PREFIX = "nobody:"


def load_people(video_id: str, folder=None) -> tuple:
    """(times, people) for one video, or (None, None) when there's no export with people for it."""
    import hair_vote as HV
    f = (folder or HV.scores_dir()) / f"{video_id}.npz"
    try:
        with np.load(f) as z:
            if "people" not in z.files:
                return None, None
            return np.asarray(z["times"], np.float32), np.asarray(z["people"], np.int32)
    except (OSError, ValueError, KeyError):
        return None, None


def _span(scene: dict) -> tuple:
    if "start" in scene:
        return float(scene["start"]), float(scene["end"])
    return tc_to_seconds(scene["start_tc"]), tc_to_seconds(scene["end_tc"])


def nobody_scenes(scenes: list, times, people, dismissed=()) -> list:
    """Scene ids (not excluded, not dismissed) where nobody is on screen for ≥ NOBODY_SHARE of the scene."""
    if times is None or people is None or not len(times):
        return []
    dismissed = set(dismissed or [])
    out = []
    for s in scenes:
        if s.get("excluded") or f"{DISMISS_PREFIX}{s['scene_id']}" in dismissed:
            continue
        try:
            a, b = _span(s)
        except (KeyError, ValueError, TypeError):
            continue
        m = (times >= a) & (times < b)
        if int(m.sum()) < MIN_SAMPLES:
            continue
        if float((people[m] == 0).mean()) >= NOBODY_SHARE:
            out.append(s["scene_id"])
    return out


def dismiss_key(scene_id) -> str:
    return f"{DISMISS_PREFIX}{scene_id}"


def apply_choice(catalogue: dict, scene_ids: list, exclude: list) -> int:
    """The 🚫 dialog's answer, in the catalogue: scenes in `exclude` are excluded (and count as reviewed); the
    rest of scene_ids aren't suggested again. Returns how many were excluded."""
    exclude = set(exclude)
    lst = catalogue.setdefault("dismissed_edit_suggestions", [])
    n = 0
    for s in catalogue.get("scenes") or []:
        sid = s["scene_id"]
        if sid not in scene_ids:
            continue
        if sid in exclude:
            s["excluded"] = True
            s["reviewed"] = True
            n += 1
        elif dismiss_key(sid) not in lst:
            lst.append(dismiss_key(sid))
    return n


def people_hint(scenes: list, times, people) -> int | None:
    """The most people seen together: the largest per-scene median (scenes with ≥ 4 s of scores, excluded
    scenes left out). None without data."""
    if times is None or people is None or not len(times):
        return None
    best = None
    for s in scenes or [{"start": float(times.min()), "end": float(times.max()) + 0.5}]:
        if s.get("excluded"):
            continue
        try:
            a, b = _span(s)
        except (KeyError, ValueError, TypeError):
            continue
        m = (times >= a) & (times < b)
        if int(m.sum()) < HINT_MIN_SAMPLES:
            continue
        med = int(np.median(people[m]))
        best = med if best is None else max(best, med)
    return best
