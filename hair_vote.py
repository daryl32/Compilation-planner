"""
💇 The model's hair-colour vote per video (claude/hair-colour-plan.md, step 2).

Hair colour belongs to the person, not the moment: lighting and angle flip one actress between blonde,
brunette and red from shot to shot. So the student's half-second scores (catalogue/student_scores/, written by
`student_tags.py export`) are pooled over the whole video:

  1. Each colour's score is read relative to its own cut-off (score / cut-off), so a colour the student is
     shy about (Redhead, few labels) isn't out-voted just for having lower raw scores.
  2. Only half seconds where someone's hair shows count: the best colour there ≥ VISIBLE of its cut-off.
     Usable footage only — inside the library range, excluded scenes left out.
  3. Main colour = the highest average over those half seconds.
  4. Second actress: another colour is kept only if it wins whole scenes adding up to ≥ SECOND_SHARE of the
     counted time, at least one of them ≥ SECOND_MIN_SEC. Lighting flips are short and scattered; a second
     actress comes in long blocks.
  5. ❓ Unsure (Daryl confirms instead of the vote guessing): too little hair seen, the top two averages
     within CLOSE of each other with no second actress found, or the main colour winning under half the
     counted time.

The vote is written with library_common.set_hair_votes — Daryl's own answers are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from library_common import HAIR_COLOURS, overlap_with_range

STEP = 0.5               # seconds between student scores (matches student_tags.STEP)
VISIBLE = 0.6            # best colour ≥ 60 % of its cut-off → hair shows here
MIN_SEEN_SEC = 10.0      # less hair than this in the whole video → ❓
SECOND_SHARE = 0.25      # a second colour must win scenes covering ≥ 25 % of the counted time …
SECOND_MIN_SEC = 20.0    # … with at least one such scene ≥ 20 s of visible hair
CLOSE = 0.08             # top two averages this close (in cut-off units) and no second actress → ❓
MAIN_MIN_SHARE = 0.5     # main colour winning scenes under half the counted time (and no second) → ❓


def scores_dir() -> Path:
    from config import CATALOGUE_DIR
    return Path(CATALOGUE_DIR) / "student_scores"


def load_index(folder: Path = None) -> dict:
    try:
        return json.loads(((folder or scores_dir()) / "index.json").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def hair_columns(index: dict) -> tuple:
    """(columns of Blonde / Brunette / Redhead in the score files, their cut-offs 0–1), or (None, None)."""
    tags = [str(t).lower() for t in index.get("tags") or []]
    cutoffs = index.get("cutoffs") or []
    cols, cuts = [], []
    for c in HAIR_COLOURS:
        if c.lower() not in tags:
            return None, None
        k = tags.index(c.lower())
        cols.append(k)
        cut = cutoffs[k] / 100.0 if k < len(cutoffs) and cutoffs[k] else 0.5
        cuts.append(min(max(cut, 0.05), 1.0))
    return cols, np.asarray(cuts, np.float32)


def usable_spans(scenes: list, time_range=None) -> list:
    """[(start, end)] of the scenes that count: not excluded, clipped to the library range.
    scenes: slim catalogue scenes (start / end / excluded)."""
    out = []
    for s in scenes or []:
        if s.get("excluded"):
            continue
        a, b = overlap_with_range(float(s["start"]), float(s["end"]), time_range)
        if a is not None and b > a:
            out.append((a, b))
    return out


def vote_video(times: np.ndarray, hair_scores: np.ndarray, cutoffs: np.ndarray, spans: list = None,
               step: float = STEP) -> dict:
    """One video's vote. hair_scores: (n × 3) 0–100 (or 0–1) in HAIR_COLOURS order; spans: [(start, end)]
    of usable scenes (None = the whole video is one scene).
    → {"colours", "unsure", "scores": {colour: average in cut-off units}, "shares": {colour: share of the
       counted time in scenes it wins}, "seen_sec"}"""
    times = np.asarray(times, np.float32)
    s = np.asarray(hair_scores, np.float32)
    if s.size and s.max() > 1.0:
        s = s / 100.0
    rel = s / cutoffs[None, :] if len(s) else s.reshape(0, len(HAIR_COLOURS))
    if spans is None:
        spans = [(float(times.min()), float(times.max()) + step)] if len(times) else []
    scene_of = np.full(len(times), -1, np.int32)
    for k, (a, b) in enumerate(spans):
        scene_of[(times >= a) & (times < b)] = k
    seen = (scene_of >= 0) & (rel.max(axis=1) >= VISIBLE) if len(rel) else np.zeros(0, bool)
    seen_sec = float(seen.sum()) * step
    out = {"colours": [], "unsure": True, "scores": {}, "shares": {}, "seen_sec": round(seen_sec, 1)}
    if seen_sec < MIN_SEEN_SEC:
        return out
    avg = rel[seen].mean(axis=0)
    out["scores"] = {c: round(float(v), 3) for c, v in zip(HAIR_COLOURS, avg)}
    main = int(np.argmax(avg))
    # Scene winners
    won_sec = np.zeros(len(HAIR_COLOURS))
    big_scene = np.zeros(len(HAIR_COLOURS), bool)
    for k in range(len(spans)):
        m = seen & (scene_of == k)
        if not m.any():
            continue
        w = int(np.argmax(rel[m].mean(axis=0)))
        sec = float(m.sum()) * step
        won_sec[w] += sec
        big_scene[w] |= sec >= SECOND_MIN_SEC
    shares = won_sec / seen_sec
    out["shares"] = {c: round(float(v), 3) for c, v in zip(HAIR_COLOURS, shares)}
    keep = [main] + [c for c in range(len(HAIR_COLOURS))
                     if c != main and shares[c] >= SECOND_SHARE and big_scene[c]]
    out["colours"] = [HAIR_COLOURS[c] for c in sorted(keep)]
    second_best = float(np.sort(avg)[-2])
    unsure = len(keep) == 1 and (float(avg[main]) - second_best < CLOSE or shares[main] < MAIN_MIN_SHARE)
    out["unsure"] = bool(unsure)
    return out


def vote_all(index_videos: dict, ranges: dict, folder: Path = None) -> tuple:
    """Votes for every video with a score file. index_videos: catalogue_store.video_index();
    ranges: library_common.library_ranges(). → ({video_id: vote}, problem or None)"""
    folder = folder or scores_dir()
    index = load_index(folder)
    if not index:
        return {}, "No student scores yet — run `student.sh export --name <model>` on the server."
    cols, cuts = hair_columns(index)
    if cols is None:
        return {}, f"The {index.get('model', '?')} scores have no Blonde / Brunette / Redhead tags."
    step = float(index.get("step") or STEP)
    votes = {}
    for v in index_videos.values():
        vid = v["video_id"]
        f = folder / f"{vid}.npz"
        if not f.exists():
            continue
        try:
            with np.load(f) as z:
                times, scores = z["times"], z["scores"][:, cols]
        except (OSError, ValueError, KeyError, IndexError):
            continue
        spans = usable_spans(v.get("scenes"), ranges.get(vid)) or None
        vote = vote_video(times, scores, cuts, spans, step)
        vote["model"] = index.get("model")
        vote["created"] = index.get("created")
        votes[vid] = vote
    return votes, None


def stale(index: dict, hair: dict) -> bool:
    """Is there a student export newer than the votes saved (or no votes at all)?"""
    if not index:
        return False
    made = {(h.get("vote_model"), h.get("vote_created")) for h in hair.values() if h.get("vote_model")}
    return (index.get("model"), index.get("created")) not in made
