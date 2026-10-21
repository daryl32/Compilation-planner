"""✂️ The cut editor (Step 0 of the planner): play the track with a playhead
over its energy curve and beats, and add / move / remove cuts by ear.

The front end is components/cut_editor/index.html (plain HTML/JS, no build
step). This module serves the track's audio to it and passes the cut list
both ways. Which cuts apply, and how edits are stored per project, is up to
the page (pages/1_Compilation_Planner.py); the cut logic is core/cut_edits.py.
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "cut_editor", path=str(Path(__file__).resolve().parent / "components" / "cut_editor"))


def media_url(path: Path, mimetype: str = "audio/mp4", coordinates: str = "cut_editor_audio") -> str:
    """A URL the browser can load `path` from — Streamlit's own media server
    (what st.audio uses). Registered again on every run, as st.audio does."""
    try:
        from streamlit import runtime
        if runtime.exists():
            return runtime.get_instance().media_file_mgr.add(str(path), mimetype, coordinates)
    except Exception:
        pass
    return ""


def _envelope(track: dict) -> tuple:
    hires = track.get("hires")
    if hires and hires.get("rms"):
        vals, rate = hires["rms"], float(hires.get("rate", 30))
    else:
        env = track.get("energy_envelope") or []
        vals = [e["energy"] for e in env]
        rate = 1.0 / max(env[1]["time"] - env[0]["time"], 1e-6) if len(env) > 1 else 2.0
    top = max(vals) if vals else 1.0
    top = top or 1.0
    return [round(v / top, 3) for v in vals], rate


def cut_editor(track: dict, track_id: str, *, cuts: list, base_cuts: list, beats: list, bars: list,
               doc_rev: int, audio_url: str = "", key: str = None, locked: list = (), ghosts: list = (),
               compare: list = None, region: tuple = None):
    """Draw the editor. Returns the latest value it sent — {"cuts", "locked", "doc_rev",
    "edited": True} after an edit, or {"region": [start, end], "nonce", "edited": False} when
    ▭ picks the part to re-cut — or None before the first. Streamlit keeps returning the last
    value on later runs, so callers compare doc_rev / nonce with what they stored.
    locked: 🔒 cut times; ghosts: 👻 idea times to show (tap to accept); compare: the ⚖️ A
    cut set to show along the top; region: the part picked for re-cutting (shaded)."""
    env, rate = _envelope(track)
    return _COMPONENT(
        track_id=track_id, duration=float(track["duration_sec"]), env=env, env_rate=rate,
        beats=[round(float(b), 3) for b in beats], bars=[round(float(b), 3) for b in bars],
        cuts=list(cuts), base_cuts=list(base_cuts), doc_rev=int(doc_rev), audio_url=audio_url,
        locked=[round(float(t), 3) for t in locked or []], ghosts=[round(float(t), 3) for t in ghosts or []],
        compare=[round(float(t), 3) for t in compare] if compare is not None else None,
        region=[round(float(region[0]), 3), round(float(region[1]), 3)] if region else None,
        key=key or f"cut_editor_{track_id}", default=None)
