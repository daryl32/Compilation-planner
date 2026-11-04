"""🎚️ The Mashup's lanes view: each part on its own lane where it plays in the mix (overlapping at the joins).

On a dark stage: the whole mix as a strip at the top (tap / drag it to move around; pinch the lanes to zoom), each
lane's sections on a strip above it, its waveform coloured by sound (deep blue = bass-heavy, pale = bright, pink =
vocals, orange = drums) with 8 / 16-bar phrase lines, ▌ ▐ handles to trim it (clicking onto phrases), ▶ per part,
▶ Mix (the preview, with a cursor) and a pill on every join — tap it to edit the join, drag it to move the hand-over.

The front end is components/mix_lanes/index.html. It returns {"uid", "start", "end", "nonce"} after a trim,
{"join", "a_uid", "a_end", "b_start", "nonce"} after a join drag, {"join", "open": True, "nonce"} after a tap on a
join, or None; the caller compares nonces (mix_ui._mashup_lanes)."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "mix_lanes", path=str(Path(__file__).resolve().parent / "components" / "mix_lanes"))


def mix_lanes(*, duration: float, lanes: list, joins: list, mix_url: str = "", key: str = "mix_lanes"):
    return _COMPONENT(duration=float(duration), lanes=lanes, joins=joins, mix_url=mix_url, key=key, default=None)
