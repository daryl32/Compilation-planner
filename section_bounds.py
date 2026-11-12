"""
📐 Section lines straight from the music — sections from scratch, step 1 (12 Nov 2026).

The 🧩 Sections suggester (section_types.py) starts from ten type names and rules about where each belongs. This
starts from the other end, with no names at all: a section ends wherever the music CHANGES —
    a layer comes in or drops out (drums, hats, bass, voice),
    the loudness or the brightness steps up or down,
    or the sound itself changes (the analysis' novelty curve) —
and every line sits on a bar line. Each line carries what changed, in words, and how big the change is; each
section between two lines carries what is playing in it. What a section is CALLED comes later (step 2: a profile per
section; step 3: a type is a named profile).

How a line is found. Every layer is read once per bar, 0–1 against the track's own strong level (so any track
compares: audio_drive._norm). At each bar line the bars after it are compared with the bars before it, per layer,
over two bars and over four; the size of a change is the weighted length of those steps, plus the novelty at the
line. Lines are the local peaks of that size above MIN_SIZE, strongest first, at least MIN_GAP_BARS apart — except
a one-bar break (the band stopping dead for a bar), which is found on its own and gets a line on each side.

No Streamlit in here.
"""

from __future__ import annotations

import numpy as np

import audio_drive as AD

# What is read per bar, and how much a step in it counts. Drums and hats are spikes, so a bar is read on its hits
# (90th percentile, as audio_drive does); the rest on their mean.
LAYERS = ("drums", "hats", "bass", "vocals", "loudness", "bright")
WEIGHT = {"drums": 1.0, "hats": 0.4, "bass": 0.8, "vocals": 0.8, "loudness": 0.9, "bright": 0.3}
W_CHANGE = 0.35            # the novelty at the line ("the sound changes"), on the same scale
EMOJI = {"drums": "🥁", "hats": "🎩", "bass": "🎸", "vocals": "🎤", "loudness": "🔊", "bright": "✨"}
NAME = {"drums": "drums", "hats": "hats", "bass": "bass", "vocals": "voice", "loudness": "loudness",
        "bright": "brightness"}

WINDOWS = (2, 4)           # bars before / after a line that are compared
MIN_SIZE = 0.22            # a change smaller than this is not a line
STRONG = 0.5               # a change this big is a main line (level 1); the rest are level 2
MIN_GAP_BARS = 2           # lines are at least this far apart (a one-bar break apart)
SAY_STEP = 0.15            # a layer's step is named in the line's reason from this size
OFF, ON = 0.15, 0.3        # a layer "comes in" from under OFF to at least ON, and "drops out" the other way


# ---------------------------------------------------------------------------
# What is playing, bar by bar
# ---------------------------------------------------------------------------

def bar_levels(track: dict, beats=None) -> dict:
    """{"bars": bar starts (from 0), "duration", "levels": {layer: one 0–1 value per bar}, "change": novelty at each
    bar line or None}. Layers the analysis doesn't have are left out; loudness is always there."""
    drv = AD.drivers(track, beats)
    bars, rate, curves, duration = np.asarray(drv["bars"], float), drv["rate"], dict(drv["curves"]), drv["duration"]
    n = len(curves["loudness"])
    a = track.get("analysis") or {}
    a_rate = float(a.get("rate", rate))

    def extra(key):
        v = a.get(key)
        if v is None or not len(v):
            return None
        v = np.asarray(v, float)
        if abs(a_rate - rate) > 1e-6 or len(v) != n:
            v = np.interp(np.arange(n) / rate, np.arange(len(v)) / a_rate, v)
        return AD._norm(v)
    for name, key in (("hats", "hat"), ("bright", "brightness")):
        c = extra(key)
        if c is not None:
            curves[name] = c
    ends = np.append(bars[1:], duration)
    levels = {}
    for name in LAYERS:
        c = curves.get(name)
        if c is None:
            continue
        vals = []
        for a_, b_ in zip(bars, ends):
            lo, hi = int(a_ * rate), max(int(a_ * rate) + 1, min(int(b_ * rate), n))
            seg = c[lo:hi]
            if not len(seg):
                vals.append(0.0)
            elif name in ("drums", "hats"):
                vals.append(float(np.percentile(seg, 90)))
            else:
                vals.append(float(seg.mean()))
        levels[name] = np.asarray(vals)
    change = None
    if curves.get("change") is not None and len(bars) > 1:
        c = curves["change"]
        half = max(1, int(0.2 * rate))
        change = np.asarray([float(c[max(0, int(t * rate) - half):int(t * rate) + half + 1].max())
                             if int(t * rate) < len(c) else 0.0 for t in bars])
    return {"bars": bars, "duration": float(duration), "levels": levels, "change": change}


# ---------------------------------------------------------------------------
# The lines
# ---------------------------------------------------------------------------

def _steps(x: np.ndarray) -> np.ndarray:
    """For every bar line k (0 = the start, never a line): the bars after it against the bars before it, averaged
    over the windows — sharpest exactly on the bar where the level changes."""
    n = len(x)
    out = np.zeros(n)
    for k in range(1, n):
        s = 0.0
        for w in WINDOWS:
            s += float(x[k:k + w].mean()) - float(x[max(0, k - w):k].mean())
        out[k] = s / len(WINDOWS)
    return out


def _reason(name: str, before: float, after: float) -> str:
    e, n = EMOJI[name], NAME[name]
    plural = name in ("drums", "hats")
    if name in ("loudness", "bright"):
        word = {"loudness": ("louder", "quieter"), "bright": ("brighter", "darker")}[name][0 if after > before else 1]
        return f"{e} {word}"
    if before < OFF and after >= ON:
        return f"{e} {n} come in" if plural else f"{e} {n} comes in"
    if before >= ON and after < OFF:
        return f"{e} {n} drop out" if plural else f"{e} {n} drops out"
    return f"{e} {n} up" if after > before else f"{e} {n} down"


def find_lines(bars, levels: dict, duration: float, change=None) -> list:
    """The section lines of one track, in time order:
        {"time", "bar" (index of the bar that starts on it), "size", "level" (1 main / 2 lesser), "why" [words],
         "steps" {layer: after − before}}
    bars: bar start times from 0; levels: {layer: per-bar 0–1}; change: novelty per bar line, or None."""
    bars = np.asarray(bars, float)
    n = len(bars)
    lv = {k: np.asarray(v, float) for k, v in (levels or {}).items() if k in WEIGHT and len(v) == n}
    if n < 4 or not lv:
        return []
    steps = {k: _steps(v) for k, v in lv.items()}
    size = np.sqrt(sum((WEIGHT[k] * s) ** 2 for k, s in steps.items()))
    if change is not None and len(change) == n:
        size = size + W_CHANGE * np.asarray(change, float)
    size[0] = 0.0
    # candidates: local peaks above the floor
    cand = [k for k in range(1, n) if size[k] >= MIN_SIZE and size[k] >= size[k - 1]
            and (k + 1 >= n or size[k] >= size[k + 1])]
    # a one-bar break — the band stops dead for a bar and comes back — is its own section: the comparison of
    # several bars either side smooths a single bar away, so it is looked for directly (this bar against BOTH its
    # neighbours) and gets a line on each side
    stop = {}
    for k in range(1, n - 1):
        dip = np.sqrt(sum((WEIGHT[name] * max(0.0, min(v[k - 1], v[k + 1]) - v[k])) ** 2 for name, v in lv.items()))
        if dip >= STRONG and float(duration) - bars[k + 1] >= 0.5:
            stop[k], stop[k + 1] = ("⏸️ the band stops for a bar", dip), ("▶️ the band comes back in", dip)
    chosen = list(stop)
    for k in sorted(cand, key=lambda k: -size[k]):                  # strongest first, keeping their distance
        if float(duration) - bars[k] < 0.5 or k in stop:
            continue
        if all(abs(k - c) >= MIN_GAP_BARS for c in chosen):
            chosen.append(k)
    out = []
    for k in sorted(chosen):
        w = WINDOWS[-1]
        named = sorted(((abs(WEIGHT[name] * s[k]), name) for name, s in steps.items() if abs(s[k]) >= SAY_STEP),
                       reverse=True)
        why = [_reason(name, float(lv[name][max(0, k - w):k].mean()), float(lv[name][k:k + w].mean()))
               for _, name in named]
        if k in stop:
            why, size[k] = [stop[k][0]], max(float(size[k]), float(stop[k][1]))
        if not why:
            why = ["🎶 the sound changes"]
        out.append({"time": round(float(bars[k]), 3), "bar": int(k), "size": round(float(size[k]), 3),
                    "level": 1 if size[k] >= STRONG else 2, "why": why,
                    "steps": {name: round(float(s[k]), 3) for name, s in steps.items()}})
    return out


def sections(bars, levels: dict, duration: float, lines: list) -> list:
    """The stretches between the lines, with what is playing in each:
        {"n", "start", "end", "bars", "why" (its opening line's words), "level", "playing" {layer: mean 0–1}}"""
    bars = np.asarray(bars, float)
    edges = [0] + [ln["bar"] for ln in lines] + [len(bars)]
    out = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        ln = lines[i - 1] if i else None
        out.append({"n": i + 1, "start": round(float(bars[a]), 3),
                    "end": round(float(bars[b]) if b < len(bars) else float(duration), 3), "bars": int(b - a),
                    "why": list(ln["why"]) if ln else ["the start of the track"], "level": ln["level"] if ln else 1,
                    "playing": {k: round(float(np.asarray(v, float)[a:b].mean()), 2) for k, v in (levels or {}).items()
                                if len(v) == len(bars) and b > a}})
    return out


def for_track(track: dict, beats=None) -> dict:
    """Everything for one track: {"bars", "duration", "levels", "lines", "sections"} — lists, ready for JSON."""
    bl = bar_levels(track, beats)
    lines = find_lines(bl["bars"], bl["levels"], bl["duration"], bl["change"])
    return {"bars": [round(float(x), 3) for x in bl["bars"]], "duration": round(bl["duration"], 3),
            "levels": {k: [round(float(x), 3) for x in v] for k, v in bl["levels"].items()},
            "lines": lines, "sections": sections(bl["bars"], bl["levels"], bl["duration"], lines)}


def as_regions(found: dict, type_id: str = "rhythmic") -> list:
    """The found sections as a section map (section_types' shape), every part the same placeholder type — for
    loading the lines into the 🧩 editor before any types exist."""
    return [{"start": s["start"], "end": s["end"], "type": type_id, "why": " · ".join(s["why"])}
            for s in found.get("sections") or []]


def line_match(lines: list, wanted_times: list, tol: float = 1.0) -> tuple:
    """(share of the wanted lines found within tol, share of the found lines that are wanted ones)."""
    got = [float(ln["time"]) for ln in lines]
    want = [float(t) for t in wanted_times]
    hit = sum(1 for w in want if any(abs(w - g) <= tol for g in got)) / len(want) if want else 1.0
    used = sum(1 for g in got if any(abs(w - g) <= tol for w in want)) / len(got) if got else 1.0
    return hit, used
