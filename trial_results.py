"""
🧪 Trial results — reading the Modal trial's results bundle for the web app (no Streamlit here).

modal_trial.py's export stage writes CATALOGUE_DIR/trial_results/ (synced from Drive):
    results.json   {"created", "smoke", "teachers": {key: model}, "report": {...},
                    "videos": {video_id: {"duration", "cuts_new", "cuts_old", "boundaries",
                                          "scenes", "curve_fps", "person_curve", "old_curve",
                                          "moments": [{"key", "t", "scene_id", "frame",
                                                       "mine" (your tags or None), "old",
                                                       "teachers": {key: {"tags", "unsure",
                                                                          "caption"}}}]}}}
    report.md      the trial report
    frames/<video_id>/<key>.jpg
"""

from __future__ import annotations

import json
from pathlib import Path

from config import CATALOGUE_DIR

RESULTS_DIR = CATALOGUE_DIR / "trial_results"
_CACHE = {}

RIGHT, WRONG, MISSED, NEUTRAL = "right", "wrong", "missed", "neutral"


def load() -> dict | None:
    """The latest results (re-read when the file changes), or None if there are none."""
    path = RESULTS_DIR / "results.json"
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    if _CACHE.get("stamp") != stamp:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        _CACHE.update(stamp=stamp, data=data if isinstance(data, dict) else None)
    return _CACHE.get("data")


def report_markdown() -> str:
    try:
        return (RESULTS_DIR / "report.md").read_text(encoding="utf-8")
    except OSError:
        return ""


def moment_id(video_id: str, moment: dict) -> str:
    """The 🎯 Training Labels id for this frame — the same id a tag card at that moment gets, so
    a correction here and a card there are one label."""
    import training_labels as TLB
    return TLB.card_id("tags", video_id, float(moment["t"]))


def with_saved_labels(videos: dict, items: list) -> dict:
    """The results' moments with "mine" taken from your current training labels where you've
    labelled (or corrected) that frame since the trial ran. Returns a new videos dict."""
    saved = {i["id"]: i for i in items if i.get("type") == "tags" and not i.get("skipped")}
    out = {}
    for vid, v in videos.items():
        ms = []
        for m in v.get("moments") or []:
            mid = m.get("gold_id") or moment_id(vid, m)
            if mid in saved:
                m = {**m, "mine": list(saved[mid].get("tags") or []), "gold_id": mid}
            ms.append(m)
        out[vid] = {**v, "moments": ms}
    return out


def correction_card(video_id: str, moment: dict, prefill: list) -> dict:
    """A 🎯 tag card for this frame, for training_labels.save_label."""
    return {"id": moment.get("gold_id") or moment_id(video_id, moment), "type": "tags",
            "video_id": video_id, "scene_id": moment.get("scene_id"), "t": float(moment["t"]),
            "suggested": list(prefill)}


def frame_path(moment: dict) -> Path | None:
    return RESULTS_DIR / moment["frame"] if moment.get("frame") else None


def norm(tag) -> str:
    """Tags compare ignoring case and spaces."""
    return "".join(str(tag).split()).lower()


def chips(pred, mine) -> list:
    """[(tag, status)] for one row of tags. Against your tags (mine): right / wrong, then the
    ones you have that this row missed. Without your tags every chip is neutral."""
    pred = list(dict.fromkeys(pred or []))
    if mine is None:
        return [(t, NEUTRAL) for t in pred]
    mine_n = {norm(t) for t in mine}
    pred_n = {norm(t) for t in pred}
    out = [(t, RIGHT if norm(t) in mine_n else WRONG) for t in pred]
    out += [(t, MISSED) for t in mine if norm(t) not in pred_n]
    return out


def tag_sets(moment: dict) -> dict:
    """{"mine"?, "old", <teacher>...: set of normalised tags} for one moment."""
    out = {"old": {norm(t) for t in moment.get("old") or []}}
    if moment.get("mine") is not None:
        out["mine"] = {norm(t) for t in moment["mine"]}
    for k, r in (moment.get("teachers") or {}).items():
        out[k] = {norm(t) for t in r.get("tags") or []}
    return out


def teachers_disagree(moment: dict) -> bool:
    sets = [s for k, s in tag_sets(moment).items() if k not in ("mine", "old")]
    return len(sets) > 1 and any(s != sets[0] for s in sets[1:])


def any_wrong(moment: dict) -> bool:
    """A teacher's tags differ from yours (only for moments you labelled)."""
    sets = tag_sets(moment)
    return "mine" in sets and any(s != sets["mine"] for k, s in sets.items() if k not in ("mine", "old"))


FILTERS = {
    "all": "All moments",
    "mine": "Only frames I labelled",
    "wrong": "A model differs from me",
    "disagree": "The models disagree",
}


def filter_moments(moments: list, how: str = "all", tag: str | None = None) -> list:
    out = []
    for m in moments:
        if how == "mine" and m.get("mine") is None:
            continue
        if how == "wrong" and not any_wrong(m):
            continue
        if how == "disagree" and not teachers_disagree(m):
            continue
        if tag and not any(norm(tag) in s for s in tag_sets(m).values()):
            continue
        out.append(m)
    return out


def agreement(moments: list) -> dict:
    """Per teacher: how many of your labelled frames it got exactly right, and its share of
    your tags found (recall) and of its tags that were right (precision)."""
    out = {}
    for m in moments:
        if m.get("mine") is None:
            continue
        mine = {norm(t) for t in m["mine"]}
        for k, r in (m.get("teachers") or {}).items():
            pred = {norm(t) for t in r.get("tags") or []}
            a = out.setdefault(k, {"frames": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0})
            a["frames"] += 1
            a["exact"] += pred == mine
            a["tp"] += len(pred & mine)
            a["fp"] += len(pred - mine)
            a["fn"] += len(mine - pred)
    for a in out.values():
        a["precision"] = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else None
        a["recall"] = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else None
    return out


def normalised(curve) -> list:
    """Scaled to 0–1 by its own 95th percentile, so two curves on different scales can share a
    chart (None stays None)."""
    vals = sorted(v for v in curve or [] if v is not None)
    if not vals:
        return list(curve or [])
    top = vals[min(len(vals) - 1, int(0.95 * len(vals)))] or vals[-1] or 1.0
    return [None if v is None else min(v / top, 1.5) for v in curve]
