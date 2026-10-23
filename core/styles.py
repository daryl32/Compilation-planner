"""🎨 Cutting styles — pick the feel, not the dials.

Each style is a recipe on top of one of the cutting methods (Smart, Best musical
moments / Scored, Energy changes & big hits / Adaptive), worked out for THIS
track from its tempo — so "a cut every 4 bars" really is 4 bars on every song —
plus one Pace control (much calmer … much busier) that halves or doubles the
cutting rate per step. Choosing a style just sets the method and its dials;
everything stays editable afterwards, and the ✂️ cut editor works on top.

    🧠 Smart            the app reads the song and decides (Smart's own pace)
    🎬 Cinematic        long shots; cuts on section changes, drops and phrase lines
    🥁 On the beat      a steady cut every bar, on bar lines and drum hits
    ⚡ Energy follow    cuts where the loudness shifts and on big hits (Adaptive)
    💥 Drop hunter      calm through the verses, speeding up into drops, fast at the drop
    🎤 Vocal-led        cuts between sung lines, never through one
    🕺 Hyper            dance-edit pace: every beat or two, on kicks and snares
    🚀 Build-up ramp    long shots, faster through each build-up, Hyper from the drop (ramp_parts)
    🎭 Story-led        each story-map part in its role's style and pace (core/story_cuts.py)

style_settings() is pure (no Streamlit): (method, settings, notes) for a track.
"""

from __future__ import annotations

import cut_scoring as CS
from core.segmentation import SEG_ADAPTIVE, SEG_DEFAULTS, SEG_METHOD_KEYS, SEG_SCORED, SEG_SMART

STYLES = {
    "smart":     {"label": "🧠 Smart",         "help": "The app reads the song and sets its own pace — busier where the music is bigger, speeding up into drops."},
    "cinematic": {"label": "🎬 Cinematic",     "help": "Long shots. Cuts on section changes, drops and phrase lines — about every 4 bars."},
    "beat":      {"label": "🥁 On the beat",   "help": "A steady cut about every bar, landing on bar lines and drum hits."},
    "energy":    {"label": "⚡ Energy follow", "help": "Cuts where the loudness shifts and on the biggest hits — about every 2 bars on average."},
    "drop":      {"label": "💥 Drop hunter",   "help": "Calm through verses and breakdowns, speeding up through build-ups and fast at the drop."},
    "vocal":     {"label": "🎤 Vocal-led",     "help": "Cuts between sung lines and never through one (needs the vocal analysis)."},
    "hyper":     {"label": "🕺 Hyper",         "help": "Dance-edit pace: a cut every beat or two, on kicks and snares."},
    "ramp":      {"label": "🚀 Build-up ramp", "help": "Longer shots until the music builds, cuts speeding up through each build-up, full 🕺 Hyper pace from the drop for a few bars, then easing back — fast cuts only where the song earns them."},
    "story":     {"label": "🎭 Story-led",     "help": "Each story-map part in its role's own style and pace (set on each role in the Reviewer → Roles…): e.g. 💥 Action as 🕺 Hyper, busier; 🌙 Slow as 🎬 Cinematic, calmer. Parts without a role — or no story map — are 🧠 Smart."},
}
PACE_LABELS = {-2: "much calmer", -1: "calmer", 0: "normal", 1: "busier", 2: "much busier"}

# Scored-cut reason weights per style (cut_scoring.DEFAULTS keys); unspecified = default.
_WEIGHTS = {
    "cinematic": {"sc_w_section": 4.0, "sc_w_drop": 4.0, "sc_w_phrase": 3.0, "sc_w_bar": 1.0, "sc_w_hit": 0.3,
                  "sc_w_kick": 0.3, "sc_w_snare": 0.2, "sc_w_harmony": 1.0, "sc_w_vocal": 1.0, "sc_w_vocal_mid": 1.5},
    "beat":      {"sc_w_bar": 3.0, "sc_w_phrase": 1.5, "sc_w_kick": 1.5, "sc_w_snare": 1.0, "sc_w_hit": 0.7,
                  "sc_w_section": 2.0, "sc_w_drop": 2.0, "sc_w_harmony": 0.3, "sc_w_vocal": 0.3, "sc_w_vocal_mid": 0.3},
    "vocal":     {"sc_w_vocal": 4.0, "sc_w_vocal_mid": 4.0, "sc_w_phrase": 1.5, "sc_w_section": 2.0, "sc_w_drop": 1.5,
                  "sc_w_bar": 0.8, "sc_w_hit": 0.3, "sc_w_kick": 0.3, "sc_w_snare": 0.3, "sc_w_harmony": 1.0},
    "hyper":     {"sc_w_kick": 2.0, "sc_w_snare": 2.0, "sc_w_hit": 1.5, "sc_w_bar": 1.5, "sc_w_phrase": 1.0,
                  "sc_w_section": 2.0, "sc_w_drop": 2.5, "sc_w_harmony": 0.2, "sc_w_vocal": 0.2, "sc_w_vocal_mid": 0.2},
}
# (bars per cut at normal pace, follow the energy)
_SCORED_PACE = {"cinematic": (4.0, 0.3), "beat": (1.0, 0.4), "vocal": (2.0, 0.3), "hyper": (0.5, 0.6)}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _step(x, step):
    return round(round(x / step) * step, 4)


def _scored(style: str, pace: int, bar: float) -> dict:
    per_cut, follow = _SCORED_PACE[style]
    bars_per_cut = _clamp(_step(per_cut * 2.0 ** (-pace), 0.25), 0.5, 8.0)
    s = {k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_SCORED]}
    s.update(_WEIGHTS[style])
    s["sc_bars_per_cut"] = bars_per_cut
    s["sc_follow_energy"] = follow
    # 10% under / over whole bars (as Auto-detect does), so cuts a bar or phrase apart are always allowed
    s["sc_min_sec"] = _clamp(_step(0.9 * 0.5 * bars_per_cut * bar, 0.1), 0.3, 8.0)
    s["sc_max_sec"] = _clamp(_step(max(2 * s["sc_min_sec"], 1.1 * 2.5 * bars_per_cut * bar), 0.5), 0.0, 30.0)
    return s


def style_settings(track: dict, style: str, pace: int = 0, beats_per_bar: int = 4) -> tuple:
    """(method, settings, notes) for `style` on this track at `pace` (−2 … +2).
    settings holds every dial the method uses; values fit the planner's sliders."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")
    pace = int(_clamp(int(pace), -2, 2))
    bar = CS.bar_sec(track, beats_per_bar)
    a = track.get("analysis") or {}
    notes = [f"{STYLES[style]['label']} · pace {PACE_LABELS[pace]} · 1 bar = {bar:.2f}s"]

    if style == "ramp":
        s = {k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_SMART]}
        s["sm_pace"] = int(_clamp(pace - 1, -2, 2))           # between build-ups: longer shots
        parts = ramp_parts(track, pace, beats_per_bar)
        if parts:
            s["parts"] = parts
        else:
            notes.append("No build-ups or drops found in this track's analysis — it's 🧠 Smart, a little calmer.")
        return SEG_SMART, s, notes

    if style in ("smart", "story"):
        s = {k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_SMART]}
        s["sm_pace"] = pace
        if style == "story":
            # The base the story-map parts are spliced into (core/story_cuts.py) — 🧠 Smart.
            notes.append("Parts with a role are cut in that role's style and pace; the rest are 🧠 Smart.")
        return SEG_SMART, s, notes

    if style == "drop":
        s = {k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_SMART]}
        # One grid level calmer than Smart, the most contrast and the strongest speed-up: long verse
        # shots, a sharp run through the build-up, fast at the drop (on the test song: verse ≈ 5 bars a
        # cut, build-up ≈ ½ bar, drop ≈ ¾ bar).
        s.update({"sm_pace": int(_clamp(pace - 1, -2, 2)), "sm_contrast": 2.0, "sm_builds": True,
                  "sm_build_strength": 2.0, "sm_accents": True})
        if not (a.get("drops") or a.get("builds")):
            notes.append("No build-ups or drops found in this track's analysis — it follows the loud and "
                         "quiet sections instead.")
        return SEG_SMART, s, notes

    if style == "energy":
        from core.targets import adaptive_for_target
        base = {k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_ADAPTIVE]}
        res = adaptive_for_target(track, avg_sec=2.0 * 2.0 ** (-pace) * bar, base=base,
                                  beats_per_bar=beats_per_bar)
        s = {**base, **res["settings"]}
        notes += res["notes"][:1]
        return SEG_ADAPTIVE, s, notes

    s = _scored(style, pace, bar)
    if not a:
        notes.append("Only the basic analysis for this track — cuts come from beats and hits. Add the extra "
                     "analysis (Media Library → Audio) for sections, drops, drums and vocals.")
    elif style == "vocal" and a.get("vocals") is None:
        notes.append("No vocal analysis for this track, so it follows phrases and sections instead.")
    if style == "hyper" and pace > 0:
        notes.append("Already at the fastest the cut scoring allows (half a bar on average) — shortest "
                     "segments lowered instead.")
    return SEG_SCORED, s, notes


# ---------------------------------------------------------------------------
# 🚀 Build-up ramp
# ---------------------------------------------------------------------------

RAMP_STEPS = (("beat", -1), ("beat", 0), ("hyper", -1))   # through a build-up, a third each
RAMP_DROP_BARS = 8                                         # Hyper from a drop for this many bars


def ramp_parts(track: dict, pace: int = 0, beats_per_bar: int = 4) -> list:
    """The 🚀 ramp's parts ({start, end, style, pace, method, settings}, as core/story_cuts makes):
    each build-up in three steps getting faster, then Hyper from each drop for RAMP_DROP_BARS bars
    (or until the next build-up). Outside them the base (Smart, calmer) cuts. [] without any."""
    a = track.get("analysis") or {}
    duration = float(track.get("duration_sec") or 0.0)
    bar = CS.bar_sec(track, beats_per_bar)
    builds = sorted((float(b["start"]), float(b["end"])) for b in a.get("builds") or []
                    if isinstance(b, dict) and float(b.get("end", 0)) > float(b.get("start", 0)))
    drops = sorted(float(d) for d in a.get("drops") or [])
    spans = []
    for bs, be in builds:
        third = (be - bs) / 3
        for k, (st, off) in enumerate(RAMP_STEPS):
            spans.append((bs + k * third, bs + (k + 1) * third, st, off))
    for d in drops:
        nxt = min([bs for bs, _ in builds if bs > d + 1e-3] + [duration, d + RAMP_DROP_BARS * bar])
        if nxt - d > 0.5 * bar:
            spans.append((d, nxt, "hyper", 0))
    spans.sort()
    out, recipes = [], {}
    for s0, s1, st, off in spans:
        s0 = max(s0, out[-1]["end"] if out else 0.0)          # overlapping spans: the earlier wins
        if s1 - s0 < 0.25 * bar:
            continue
        p = int(_clamp(pace + off, -2, 2))
        if (st, p) not in recipes:
            method, settings, _ = style_settings(track, st, p, beats_per_bar)
            recipes[(st, p)] = (method, settings)
        method, settings = recipes[(st, p)]
        out.append({"start": round(s0, 3), "end": round(min(s1, duration), 3), "role": None, "style": st,
                    "pace": p, "method": method, "settings": dict(settings)})
    return out
