"""🎚️ The two-deck transition editor: track A over track B on one timeline (A's clock), each lane
coloured by level, vocals and drums (from the analysis — no decoding), with the blend shaded.

Drag lane B to choose where it comes in, the ◆ mix point to where A hands over, the blend's left edge
for 4 / 8 / 16 bars; tap a lane to set where ▶ A / ▶ B play from; ▶ Mix plays the rendered transition
with the cursor across both lanes.

The front end is components/deck_editor/index.html (plain HTML/JS, touch friendly). It returns
{"a_end", "b_full", "bars", "nonce"} after a drag, or None; Streamlit keeps returning the last value,
so the caller compares nonces (see mix_ui._transition)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "deck_editor", path=str(Path(__file__).resolve().parent / "components" / "deck_editor"))

RATE = 10.0          # points a second sent to the browser


def _resample(v, rate: float, duration: float) -> list:
    if v is None or not len(v):
        return []
    v = np.asarray(v, dtype=float)
    n = max(1, int(duration * RATE))
    idx = np.clip((np.arange(n) / RATE * rate).astype(int), 0, len(v) - 1)
    # the loudest point in each step, so short hits still show
    step = max(1, int(round(rate / RATE)))
    padded = np.pad(v, (0, step))
    out = np.array([padded[i:i + step].max() for i in idx])
    top = float(np.percentile(out, 98)) or 1.0
    return [round(float(x), 3) for x in np.clip(out / top, 0, 1)]


def lane_data(track: dict, track_id: str, g: dict, url: str = "", label: str = None) -> dict:
    """What the browser draws for one deck (curves at RATE a second, bar lines, the audio's URL)."""
    a = track.get("analysis") or {}
    dur = float(g["duration"])
    h = track.get("hires") or {}
    if h.get("rms"):
        env = _resample(h["rms"], float(h.get("rate", 30)), dur)
    else:
        e = track.get("energy_envelope") or []
        rate = 1.0 / max(e[1]["time"] - e[0]["time"], 1e-6) if len(e) > 1 else 2.0
        env = _resample([x["energy"] for x in e], rate, dur)
    arate = float(a.get("rate", 30))
    drums = None
    if a.get("kick") is not None:
        k = np.asarray(a["kick"], dtype=float)
        if a.get("snare") is not None and len(a["snare"]) == len(k):
            k = np.maximum(k, np.asarray(a["snare"], dtype=float))
        drums = _resample(k, arate, dur)
    vox = _resample(a.get("vocals"), arate, dur) if a.get("vocals") is not None else None
    return {"id": track_id, "label": label or track_id, "duration": dur, "rate": RATE, "env": env,
            "vox": vox, "drums": drums, "bars": [round(float(b), 3) for b in g["bars"]], "url": url}


def deck_editor(*, a: dict, b: dict, state: dict, bar_a: float, bar_b: float, speed_b: float,
                preview: dict = None, state_key: str = "", key: str = "deck_editor"):
    return _COMPONENT(a=a, b=b, state={k: float(v) if k != "bars" else int(v) for k, v in state.items()
                                      if k in ("a_end", "b_full", "bars")},
                      bar_a=float(bar_a), bar_b=float(bar_b), speed_b=float(speed_b), preview=preview,
                      state_key=state_key, key=key, default=None)
