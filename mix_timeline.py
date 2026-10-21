"""🎚️ The Mix timeline: the new track's waveform with each part in its song's colour, the
🔀 crossfades drawn where they happen (◆ edges you drag to lengthen or shorten them on either
side of the join, snapped to beats), 🎛️ bass-swap points, 🎤 layers, and a player with a
playhead once the mix has been previewed.

The front end is components/mix_timeline/index.html (plain HTML/JS, no build step, touch
friendly). It returns {"uid", "pre_beats", "post_beats", "nonce"} after a drag — the join into
the part `uid` — or None; Streamlit keeps returning the last value, so the caller compares
nonces (see mix_ui)."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "mix_timeline", path=str(Path(__file__).resolve().parent / "components" / "mix_timeline"))


def mix_timeline(*, duration: float, beat_len: float, pieces: list, layers: list, joins: list, env: list,
                 env_rate: float, beats: list, bars: list, trim_to: float = None, audio_url: str = "",
                 key: str = "mix_timeline"):
    return _COMPONENT(duration=float(duration), beat_len=float(beat_len), pieces=pieces, layers=layers,
                      joins=joins, env=[round(float(v), 3) for v in env], env_rate=float(env_rate),
                      beats=[round(float(b), 3) for b in beats], bars=[round(float(b), 3) for b in bars],
                      trim_to=None if trim_to is None else float(trim_to), audio_url=audio_url,
                      key=key, default=None)
