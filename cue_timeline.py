"""🕒 The Cue Map's timeline component (components/cue_timeline/index.html, 19 Nov 2026): rounds as rows with the
phases on top and every line as a bar where the app's engine plays it; intro / outro screens; 🔎 Pick a line. Tap a
line to open it, drag it to move its cue, drag a line from Pick a line onto the timeline (or tap an empty spot) to cue
it there. Returns the last action ({"act", ..., "nonce"}) or None — cue_map_view.timeline applies it."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "cue_timeline", path=str(Path(__file__).resolve().parent / "components" / "cue_timeline"))


def cue_timeline(data: dict, key: str = "cue_timeline"):
    return _COMPONENT(data=data, key=key, default=None)
