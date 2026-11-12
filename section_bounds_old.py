"""
📐 Section lines straight from the music — sections from scratch, step 1 (12 Nov 2026).

The 🧩 Sections suggester (section_types.py) starts from ten type names and rules about where each belongs. This
starts from the other end, with no names at all: a section ends wherever the music CHANGES —
    a layer comes in or drops out (drums, hats, bass, voice),
    the loudness or the brightness steps up or down,
    or the sound itself changes (the analysis' novelty curve) —
and every line sits on a bar line. Each line carries what changed, in words, and how big the change is; each
section between two lines carries what is playing in it.

Step 2 — a PROFILE per section, still without names (profile): which layers are playing, how loud the section is
within the track's own range, how busy the drums are (hits a second), whether it is rising, falling or steady, and
which earlier section it is like. Step 3 (not built): a type is a named profile.

Sections and parts. A big change (a solid line) starts a new SECTION; a smaller one (dotted) starts a new PART
inside it — sections are numbered 1, 2, 3 and their parts 3a, 3b. Every part has its own profile.

Blocks (blocks) — the clips' lengths, from the same measurements and still no type names. Every line is a cut.
Inside a part the profile sets the pace: busy loud drums 1 bar (half a bar where that is still a clip), other drum
parts 2 bars, parts led by the voice / instruments 2 or 4 bars, hardly anything playing 4 bars; a rising part's
blocks get shorter towards its end, a falling part's longer; lengths are musical (½, 1, 2, 4, 8 bars — 6 bars is
4 + 2); in a part the voice leads no cut lands inside a sung phrase; no block is shorter than the matcher's
shortest clip (section_cuts.MIN_CLIP_FLOOR).

How a line is found. Every layer is read once per bar, 0–1 against the track's own strong level (so any track
compares: audio_drive._norm). At each bar line the bars after it are compared with the bars before it, per layer,
over two bars and over four; the size of a change is the weighted length of those steps, plus the novelty at the
line. Lines are the local peaks of that size above MIN_SIZE, strongest first, at least MIN_GAP_BARS apart — except
a one-bar break (the band stopping dead for a bar), which is found on its own and gets a line on each side.

No Streamlit in here.

──────────────────────────────────────────────────────────────────────────────────────────────────────────
AS RELEASED IN v1.59.0 (main, 12 Nov 2026) — kept word for word so the 📐 page can switch between the old cuts
and the new ones (section_bounds.py) on the same track. Do not edit: this file is the reference. What the new
one does differently: a steady bar grid from the beats, blocks sized in seconds with short ones paired up, loud
drum parts cut alike, an alternating bar pattern not read as a one-bar break, voice-led blocks capped at 8 s.
"""

from __future__ import annotations

import numpy as np

import audio_drive as AD

# What is read per bar, and how much a step in it counts. Drums and hats are spikes, so a bar is read on its hits
# (90th percentile, as audio_drive does); the rest on their mean.
LAYERS = ("drums", "hats", "bass", "other", "vocals", "loudness", "bright")
WEIGHT = {"drums": 1.0, "hats": 0.4, "bass": 0.8, "other": 0.7, "vocals": 0.8, "loudness": 0.9, "bright": 0.3}
# With the laptop's stem curves (analysis["stem_curves"]: how loud the Demucs drums / bass / other stems are) the
# drums and bass are read from the real stems and "other" — every instrument that is neither — becomes a layer.
# Without them drums and bass come from the full mix by frequency band and there is no "other".
ABSENT_SHARE = 0.08        # a stem whose strong level is under this share of the strongest stem's isn't in the song
W_CHANGE = 0.35            # the novelty at the line ("the sound changes"), on the same scale
EMOJI = {"drums": "🥁", "hats": "🎩", "bass": "🎸", "other": "🎹", "vocals": "🎤", "loudness": "🔊", "bright": "✨"}
NAME = {"drums": "drums", "hats": "hats", "bass": "bass", "other": "instruments", "vocals": "voice",
        "loudness": "loudness", "bright": "brightness"}

WINDOWS = (2, 4)           # bars before / after a line that are compared
MIN_SIZE = 0.22            # a change smaller than this is not a line
STRONG = 0.5               # a change this big is a main line (level 1); the rest are level 2
MIN_GAP_BARS = 2           # lines are at least this far apart (a one-bar break apart)
SAY_STEP = 0.15            # a layer's step is named in the line's reason from this size
OFF, ON = 0.15, 0.3        # a layer "comes in" from under OFF to at least ON, and "drops out" the other way

# The profile (step 2)
PLAYING = ("drums", "bass", "other", "vocals")   # the layers a section is said to have (hats / brightness colour it only)
LEVEL_WORDS = ((0.33, "quiet"), (0.66, "medium"), (2.0, "loud"))     # by the section's place in the track's own range
TREND = 0.15               # loudness changing by this much over the section (fitted line) = rising / falling
TREND_MIN_BARS = 3         # shorter sections are "steady": too few bars to say
BUSY_WORDS = ((0.33, "sparse"), (0.66, "steady"), (2.0, "busy"))     # by the hits a second, among the track's drum bars
LIKE = 0.1                 # two sections are alike when their layers and level differ by less than this on average


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
    stems = a.get("stem_curves") or {}
    used = []
    for name in ("drums", "bass", "other"):                # the real stems, where the laptop has measured them
        v = stems.get(name)
        if v is None or not len(v):
            continue
        v = np.asarray(v, float)
        s_rate = float(stems.get("rate", a_rate))
        if abs(s_rate - rate) > 1e-6 or len(v) != n:
            v = np.interp(np.arange(n) / rate, np.arange(len(v)) / s_rate, v)
        share = float((stems.get("share") or {}).get(name, 1.0))
        curves[name] = AD._norm(v) if share >= ABSENT_SHARE else np.zeros(n)
        used.append(name)
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
    hits = None                                 # drum hits a second, bar by bar: how busy the rhythm is
    if curves.get("drums") is not None:
        from core import sync as SYNC
        hits = np.asarray([SYNC.hit_rate(curves["drums"][int(a_ * rate):max(int(a_ * rate) + 3, min(int(b_ * rate), n))],
                                         rate, AD.DRUM_LEVEL) or 0.0 for a_, b_ in zip(bars, ends)])
    change = None
    if curves.get("change") is not None and len(bars) > 1:
        c = curves["change"]
        half = max(1, int(0.2 * rate))
        change = np.asarray([float(c[max(0, int(t * rate) - half):int(t * rate) + half + 1].max())
                             if int(t * rate) < len(c) else 0.0 for t in bars])
    return {"bars": bars, "duration": float(duration), "levels": levels, "change": change, "hits": hits,
            "stems": used}


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
    plural = name in ("drums", "hats", "other")
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
        # in words: the typical bar either side (median — one bar that straddles the change doesn't blur "comes in")
        why = [_reason(name, float(np.median(lv[name][max(0, k - w):k])), float(np.median(lv[name][k:k + w])))
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
        out.append({"n": i + 1, "bar": int(a), "start": round(float(bars[a]), 3),
                    "end": round(float(bars[b]) if b < len(bars) else float(duration), 3), "bars": int(b - a),
                    "why": list(ln["why"]) if ln else ["the start of the track"], "level": ln["level"] if ln else 1,
                    "playing": {k: round(float(np.asarray(v, float)[a:b].mean()), 2) for k, v in (levels or {}).items()
                                if len(v) == len(bars) and b > a}})
    # sections (a big change starts one) and their parts (a smaller change): 1, 2a, 2b, 3 …
    main = 0
    for sc in out:
        main += 1 if (sc["level"] == 1 or main == 0) else 0
        sc["main"] = main
    for sc in out:
        group = [x for x in out if x["main"] == sc["main"]]
        sc["label"] = str(sc["main"]) + ("" if len(group) == 1 else "abcdefghijklmnopqrstuvwxyz"[min(group.index(sc), 25)])
    return out


# ---------------------------------------------------------------------------
# Step 2 — a profile per section
# ---------------------------------------------------------------------------

def _word(x: float, words) -> str:
    return next(w for top, w in words if x < top)


def profile(secs: list, bars, levels: dict, hits=None) -> list:
    """The sections with a "profile" each (and "words", the same in plain text):
        {"on": [layers playing], "level": 0–1 within the track's own loudness range, "level_word",
         "busy": drum hits a second (None without drums), "busy_word", "trend": "rising" / "falling" / "steady",
         "change": how far the loudness moves over the section, "like": number of the earliest section like it}"""
    bars = np.asarray(bars, float)
    n = len(bars)
    lv = {k: np.asarray(v, float) for k, v in (levels or {}).items() if len(v) == n}
    loud = lv.get("loudness")
    lo, hi = (float(np.percentile(loud, 10)), float(np.percentile(loud, 90))) if loud is not None and n else (0.0, 1.0)
    hits = np.asarray(hits, float) if hits is not None and len(hits) == n else None
    drum_bars = hits[lv["drums"] >= ON] if hits is not None and "drums" in lv else np.zeros(0)
    edges = np.searchsorted(bars, [sc["start"] for sc in secs] + [np.inf], side="left")
    out, vecs = [], []
    for i, sc in enumerate(secs):
        a, b = int(edges[i]), int(min(edges[i + 1], n))
        b = max(b, a + 1)
        on = [k for k in PLAYING if k in lv and float(lv[k][a:b].mean()) >= ON]
        level = float(np.clip((float(loud[a:b].mean()) - lo) / (hi - lo), 0.0, 1.0)) if loud is not None and hi - lo > 1e-6 else 0.5
        change, trend = 0.0, "steady"
        if loud is not None and b - a >= TREND_MIN_BARS:
            change = float(np.polyfit(np.arange(b - a), loud[a:b], 1)[0]) * (b - a - 1)
            trend = "rising" if change >= TREND else "falling" if change <= -TREND else "steady"
        busy = busy_word = None
        if hits is not None and "drums" in on:
            busy = float(hits[a:b].mean())
            rank = float((drum_bars <= busy).mean()) if drum_bars.size else 0.5
            busy_word = _word(rank, BUSY_WORDS)
        vec = np.array([float(lv[k][a:b].mean()) if k in lv else 0.0 for k in LAYERS] + [level])
        like = next((secs[j].get("label") or secs[j]["n"] for j in range(i) if out[j]["profile"]["on"] == on
                     and out[j]["profile"]["trend"] == trend and float(np.abs(vecs[j] - vec).mean()) < LIKE), None)
        vecs.append(vec)
        prof = {"on": on, "level": round(level, 2), "level_word": _word(level, LEVEL_WORDS),
                "busy": None if busy is None else round(busy, 2), "busy_word": busy_word, "trend": trend,
                "change": round(change, 2), "like": like}
        words = [" + ".join(f"{EMOJI[k]} {NAME[k]}" for k in on) if on else "nothing much playing",
                 prof["level_word"]]
        if busy_word:
            words.append(f"{busy_word} drums ({busy:.1f}/s)")
        if trend != "steady":
            words.append("↗ rising" if trend == "rising" else "↘ falling")
        if like:
            words.append(f"like {like}")
        out.append({**sc, "profile": prof, "words": " · ".join(words)})
    return out


# ---------------------------------------------------------------------------
# Blocks — the clip lengths, from the profile
# ---------------------------------------------------------------------------
BLOCK_MAX_BARS = 8         # no block is longer than this (a cut that would fall inside a sung phrase is kept then)
BLOCK_MAX_SEC = 8.0        # the pace never asks for a block longer than this (4 bars at 90 BPM is 10.7 s → 2 bars)
PHRASE_EDGE = 0.35         # a cut this close (in bars) to where a sung phrase starts or ends is not "inside" it


def _pace(prof: dict, bar_sec: float, min_sec: float) -> tuple:
    """(block length in bars, why) for a part with this profile."""
    on = prof.get("on") or []
    if "drums" in on:
        if prof.get("busy_word") == "busy" and prof.get("level_word") == "loud":
            length, why = (0.5, "busy loud drums") if bar_sec / 2 >= min_sec - 1e-6 else (1.0, "busy loud drums")
        else:
            length, why = 2.0, f"{prof.get('busy_word') or 'steady'} drums, {prof.get('level_word')}"
    elif on:
        lead = "voice" if "vocals" in on else "instruments" if "other" in on else "bass"
        length, why = (4.0, f"{lead} leading, quiet") if prof.get("level_word") == "quiet" else (2.0, f"{lead} leading")
    else:
        length, why = 4.0, "hardly anything playing"
    while length * bar_sec > BLOCK_MAX_SEC and length > 1.0:
        length /= 2
    while length * bar_sec < min_sec - 1e-6 and length < BLOCK_MAX_BARS:
        length *= 2
    return length, why


def _fill(n_bars: float, length_at) -> list:
    """Block lengths (in bars) covering n_bars: length_at(position 0–1) each time; what is left over at the end
    goes in musical lengths (6 = 4 + 2), and a leftover under half the last block joins it."""
    out, pos = [], 0.0
    while n_bars - pos > 1e-6:
        want = float(length_at(pos / n_bars))
        left = n_bars - pos
        if want <= left + 1e-6:
            out.append(want)
            pos += want
            continue
        if out and left < out[-1] / 2 - 1e-6:           # a scrap: the block before it takes it
            out[-1] += left
            break
        piece = max(p for p in (0.5, 1.0, 2.0, 4.0, 8.0) if p <= left + 1e-6) if left >= 0.5 else left
        out.append(piece)
        pos += piece
    return out


def blocks(secs: list, bars, duration: float, vocal_starts=(), vocal_ends=(), min_sec: float = None) -> list:
    """The blocks of a track, in order: {"n", "start", "end", "bars", "section" (the part's label), "why"}.
    secs: the sections with their profiles (profile()). min_sec: the shortest clip (default: the matcher's)."""
    bars = np.asarray(bars, float)
    n = len(bars)
    if n < 2 or not secs:
        return []
    ends = np.append(bars[1:], float(duration))
    bar_sec = float(np.median(ends - bars))
    if min_sec is None:
        import section_cuts as SCU
        min_sec = SCU.MIN_CLIP_FLOOR

    def time_at(pos: float) -> float:                   # a position in bars (17.5 = half-way through bar 18) → s
        k = min(int(np.floor(pos + 1e-9)), n - 1)
        return float(duration) if pos >= n - 1e-9 else float(bars[k] + (pos - k) * (ends[k] - bars[k]))
    vs, ve = np.asarray(vocal_starts or (), float), np.asarray(vocal_ends or (), float)

    def mid_phrase(t: float) -> bool:
        edge = PHRASE_EDGE * bar_sec
        return any(a + edge < t < b - edge for a, b in zip(vs, ve))
    edges = [sc["bar"] for sc in secs] + [n]
    out = []
    for sc, a, b in zip(secs, edges[:-1], edges[1:]):
        prof = sc.get("profile") or {}
        base, why = _pace(prof, bar_sec, min_sec)
        trend = prof.get("trend") if b - a >= 4 else None
        short = base / 2 if base / 2 >= 0.5 and (base / 2) * bar_sec >= min_sec - 1e-6 else base
        longer = base * 2 if base * 2 <= (b - a) / 2 and base * 2 * bar_sec <= BLOCK_MAX_SEC else base

        def length_at(x, base=base, trend=trend, short=short, longer=longer):
            if trend == "rising":                        # the cutting speeds up into what comes next
                return base if x < 0.5 else short
            if trend == "falling":                       # … and slows down on the way out
                return base if x < 0.5 else longer
            return base
        lens = _fill(float(b - a), length_at)
        cuts = list(np.cumsum(lens)[:-1] + a)           # inside the part, in bars
        if "drums" not in (prof.get("on") or []) and "vocals" in (prof.get("on") or []) and len(vs):
            kept, last = [], float(a)                   # the voice leads: no cut in the middle of a sung phrase
            for k, c in enumerate(cuts):
                nxt = cuts[k + 1] if k + 1 < len(cuts) else float(b)
                if mid_phrase(time_at(c)):
                    moved = next((c + d for d in (-1.0, 1.0, -0.5, 0.5)
                                  if last + 0.5 <= c + d <= nxt - 0.5 and abs((c + d) * 2 - round((c + d) * 2)) < 1e-6
                                  and (c + d) * bar_sec - last * bar_sec >= min_sec - 1e-6
                                  and not mid_phrase(time_at(c + d))), None)
                    if moved is None and nxt - last <= BLOCK_MAX_BARS:
                        continue                         # no clean bar nearby: one longer block
                    c = c if moved is None else moved
                kept.append(c)
                last = c
            cuts = kept
        marks = [float(a)] + cuts + [float(b)]
        note = why + (" · speeding up" if trend == "rising" and short < base else
                      " · slowing down" if trend == "falling" and longer > base else "")
        for x0, x1 in zip(marks[:-1], marks[1:]):
            out.append({"n": len(out) + 1, "start": round(time_at(x0), 3), "end": round(time_at(x1), 3),
                        "bars": round(x1 - x0, 2), "section": sc.get("label") or str(sc["n"]), "why": note})
    return out


def for_track(track: dict, beats=None) -> dict:
    """Everything for one track: {"bars", "duration", "levels", "hits", "lines", "sections", "blocks"} — lists,
    ready for JSON; every section (part) carries its profile."""
    bl = bar_levels(track, beats)
    lines = find_lines(bl["bars"], bl["levels"], bl["duration"], bl["change"])
    secs = profile(sections(bl["bars"], bl["levels"], bl["duration"], lines), bl["bars"], bl["levels"], bl["hits"])
    return {"bars": [round(float(x), 3) for x in bl["bars"]], "duration": round(bl["duration"], 3),
            "levels": {k: [round(float(x), 3) for x in v] for k, v in bl["levels"].items()},
            "hits": None if bl["hits"] is None else [round(float(x), 2) for x in bl["hits"]],
            "stems": list(bl["stems"]),
            "lines": lines, "sections": secs,
            "blocks": blocks(secs, bl["bars"], bl["duration"], (track.get("analysis") or {}).get("vocal_starts"),
                             (track.get("analysis") or {}).get("vocal_ends"))}


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
