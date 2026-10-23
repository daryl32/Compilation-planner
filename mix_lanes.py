"""🎚️ The Mashup's lanes view: each part on its own lane where it plays in the mix (overlapping at the joins),
coloured by level / vocals / drums, with ▌ start and end ▐ handles to trim it on its bar lines, ▶ per part
(its song, with a cursor on its lane) and ▶ Mix (the preview, with a cursor across every lane).

The front end is components/mix_lanes/index.html. It returns {"uid", "start", "end", "nonce"} after a trim,
or None; the caller compares nonces (mix_ui._mashup_lanes)."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "mix_lanes", path=str(Path(__file__).resolve().parent / "components" / "mix_lanes"))


def mix_lanes(*, duration: float, lanes: list, joins: list, mix_url: str = "", key: str = "mix_lanes"):
    return _COMPONENT(duration=float(duration), lanes=lanes, joins=joins, mix_url=mix_url, key=key, default=None)
