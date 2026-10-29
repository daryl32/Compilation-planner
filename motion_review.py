"""🕺 Motion review (🎯 Training Labels → 🧪 Trial results → Motion): the video's preview copy with
its motion curves and scene cuts underneath, a playhead that moves with the video, and buttons to
judge the new cuts (👍 / 👎) or add one the detector missed (✂️). Each judgement is saved as a
scene-change label (training_labels.json) — the same kind as the ✂️ cards — so it trains and
scores the scene detector.

The front end is components/motion_review/index.html (plain HTML/JS, no build step).
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

import training_labels as TL

_COMPONENT = components.declare_component(
    "motion_review", path=str(Path(__file__).resolve().parent / "components" / "motion_review"))

MATCH_SEC = 0.5       # a new mark this close to one of yours replaces it


def video_url(path: Path) -> str:
    """A URL the browser can play `path` from (Streamlit's media server, like st.video).
    One slot, so only the video being reviewed is held in memory."""
    try:
        from streamlit import runtime
        if runtime.exists():
            return runtime.get_instance().media_file_mgr.add(str(path), "video/mp4", "motion_review_video")
    except Exception:
        pass
    return ""


def marks_for(video_id: str, items: list) -> list:
    """Your scene-change answers for this video: [{"id", "t", "verdict", "source"}]."""
    return sorted(({"id": i["id"], "t": float(i["t"]), "verdict": i.get("verdict"), "source": i.get("source")}
                   for i in items if i.get("type") == "boundary" and i.get("video_id") == video_id
                   and not i.get("skipped") and i.get("verdict") in ("cut", "no_cut") and i.get("t") is not None),
                  key=lambda m: m["t"])


def apply(value: dict, items: list) -> str | None:
    """Save what the player sent: {"action": "verdict", "t", "verdict", "source", "video_id"} or
    {"action": "remove", "id"}. Returns a short note for the page (None if nothing to do)."""
    if not value or not value.get("video_id"):
        return None
    vid, t = value["video_id"], round(float(value.get("t") or 0.0), 2)
    if value.get("action") == "remove":
        return "Removed your mark." if value.get("id") and TL.remove_label(value["id"]) else None
    if value.get("action") != "verdict" or value.get("verdict") not in ("cut", "no_cut"):
        return None
    near = [m for m in marks_for(vid, items) if abs(m["t"] - t) <= MATCH_SEC]
    if near:                                 # judging the same point again replaces your mark
        TL.remove_label(near[0]["id"])
    card = {"id": TL.card_id("boundary", vid, t), "type": "boundary", "video_id": vid, "scene_id": None,
            "t": t, "source": value.get("source") or "manual"}
    TL.save_label(card, verdict=value["verdict"])
    what = {"manual": "Added a cut", "new_cut": "Marked the cut " + ("good" if value["verdict"] == "cut" else "bad")}
    return f"{what.get(card['source'], 'Saved')} at {int(t // 60)}:{t % 60:04.1f}."


def motion_review(video: dict, video_id: str, url: str, marks: list, *, key: str):
    """Draw the player. Returns the latest value it sent, or None."""
    return _COMPONENT(
        video_id=video_id, video_url=url, duration=float(video.get("duration") or 0.0),
        curve_fps=float(video.get("curve_fps") or 2.0),
        person_curve=video.get("person_curve_n") or [], old_curve=video.get("old_curve_n") or [],
        cuts_new=[round(float(t), 2) for t in video.get("cuts_new") or []],
        cuts_old=[round(float(t), 2) for t in video.get("cuts_old") or []],
        marks=marks, key=key, default=None)
