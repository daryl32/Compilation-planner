"""🎯 Where to mix track A into track B — the top few ready-made transitions for any pair.

A transition is: track A plays up to `a_end`; track B comes in `bars` bars earlier (from `b_in` in its
own time) under A's last bars, and is fully in at `b_full` = b_in + bars. Every candidate is
phrase-aligned — A's blend starts on a section start / 8- or 16-bar phrase line, B is fully in on
one — and scored 0–1 on:

    energy   — A's level over the blend close to B's level where B takes over (or B a little higher)
    vocals   — not both singing through the blend (two vocals at once is the giveaway of a bad mix)
    drop     — B's drop landing right as the blend ends (the classic build → drop hand-over)
    phrase   — both points on strong phrase lines (sections > 16-bar > 8-bar)
    room     — enough of B left after the blend to carry on (and A's point not in its first minute)

The profile follows from it: B's drop at the end → bass_drop; both singing → vocal_swap; A busy
with drums and B's start quiet → drums_first; else smooth (core/stem_mix.PROFILES).

suggest(track_a, track_b) → the best `top`, at least MIN_APART bars apart in either track.
Pure: no Streamlit, no files (works on catalogue tracks; core.mixing.grid for the bar grid).
"""

from __future__ import annotations

import numpy as np

from core import mixing as MX

BARS_CHOICES = (8, 16)
MIN_APART = 8            # bars — suggestions differ by at least this in A or B
WEIGHTS = {"energy": 0.25, "vocals": 0.25, "drop": 0.2, "phrase": 0.15, "room": 0.15}
B_AFTER_BARS = 16        # bars of B wanted after the blend


def _curve(track: dict, name: str):
    a = track.get("analysis") or {}
    v = a.get(name)
    if v is None:
        return None, 30.0
    return np.asarray(v, dtype=float), float(a.get("rate", 30))


def _energy(track: dict):
    h = track.get("hires") or {}
    if h.get("rms"):
        v = np.asarray(h["rms"], dtype=float)
        return v / (np.percentile(v, 98) or 1.0), float(h.get("rate", 30))
    env = track.get("energy_envelope") or []
    if len(env) > 1:
        v = np.asarray([e["energy"] for e in env], dtype=float)
        return v / (v.max() or 1.0), 1.0 / max(env[1]["time"] - env[0]["time"], 1e-6)
    return None, 30.0


def _mean(curve, rate, t0, t1) -> float:
    if curve is None or t1 <= t0:
        return 0.0
    seg = curve[int(t0 * rate):max(int(t0 * rate) + 1, int(t1 * rate))]
    return float(np.clip(seg, 0, None).mean()) if len(seg) else 0.0


def phrase_points(track: dict, g: dict = None) -> list:
    """[(time, strength 0–1)] — section starts (1.0), 16-bar phrases (0.8), 8-bar phrases (0.6), every
    4th bar line (0.3). Times are bar lines of core.mixing.grid."""
    g = g or MX.grid(track)
    bars = np.asarray(g["bars"], dtype=float)
    out = {}
    for i, t in enumerate(bars):
        if i % 4 == 0:
            out[round(float(t), 3)] = 0.3
    for i, lvl in g["levels"].items():
        if i < len(bars):
            k = round(float(bars[i]), 3)
            out[k] = max(out.get(k, 0), 0.8 if lvl >= 16 else 0.6 if lvl >= 8 else 0.3)
    for s in g["sections"]:
        if len(bars):
            k = round(float(bars[int(np.argmin(np.abs(bars - s["start"])))]), 3)
            out[k] = 1.0
    return sorted(out.items())


def suggest(track_a: dict, track_b: dict, top: int = 3, bars_choices=BARS_CHOICES) -> list:
    """The best `top` transitions from A into B: [{"a_start", "a_end", "b_in", "b_full", "bars", "profile",
    "score", "parts": {name: 0–1}, "why": [str]}], best first. Times in each track's own seconds."""
    ga, gb = MX.grid(track_a), MX.grid(track_b)
    ea, ra = _energy(track_a)
    eb, rb = _energy(track_b)
    va, rva = _curve(track_a, "vocals")
    vb, rvb = _curve(track_b, "vocals")
    ka, rka = _curve(track_a, "kick")
    drops_b = sorted(float(d) for d in (track_b.get("analysis") or {}).get("drops") or [])
    pa, pb = phrase_points(track_a, ga), phrase_points(track_b, gb)
    dur_a, dur_b = ga["duration"], gb["duration"]
    cands = []
    for bars in bars_choices:
        la, lb = bars * ga["bar_len"], bars * gb["bar_len"]
        for a0, sa in pa:
            a1 = a0 + la
            if a1 > dur_a + 0.05 or a0 < min(60.0, 0.25 * dur_a):
                continue
            a1 = min(a1, dur_a)
            e_a = _mean(ea, ra, a0, a1)
            v_a = _mean(va, rva, a0, a1)
            drums_a = _mean(ka, rka, a0, a1)
            for bf, sb in pb:
                b0 = bf - lb
                if b0 < -0.05:
                    continue
                b0 = max(0.0, b0)
                after = dur_b - bf
                e_b = _mean(eb, rb, bf, min(dur_b, bf + 4 * gb["bar_len"]))
                v_b = _mean(vb, rvb, b0, bf)
                e_b_in = _mean(eb, rb, b0, bf)
                diff = e_b - e_a
                energy = float(np.clip(1.0 - abs(diff - 0.05) * 2.0, 0, 1))
                vocal = 1.0 - min(v_a, v_b) * 2.0 if va is not None and vb is not None else 0.7
                drop_hit = min((abs(d - bf) for d in drops_b), default=99.0)
                drop = 1.0 if drop_hit <= 0.5 * gb["bar_len"] else 0.4 if drop_hit <= 2 * gb["bar_len"] else 0.0
                phrase = 0.5 * sa + 0.5 * sb
                room = float(np.clip(after / (B_AFTER_BARS * gb["bar_len"]), 0, 1))
                parts = {"energy": energy, "vocals": float(np.clip(vocal, 0, 1)), "drop": drop, "phrase": phrase,
                         "room": room}
                score = sum(WEIGHTS[k] * v for k, v in parts.items())
                if drop >= 1.0:
                    profile = "bass_drop"
                elif va is not None and vb is not None and v_a > 0.25 and v_b > 0.25:
                    profile = "vocal_swap"
                elif drums_a > 0.4 and e_b_in < 0.6 * max(e_a, 1e-6):
                    profile = "drums_first"
                else:
                    profile = "smooth"
                why = []
                if drop >= 1.0:
                    why.append("B's drop lands as the blend ends")
                if parts["vocals"] >= 0.8:
                    why.append("no clashing vocals")
                if energy >= 0.8:
                    why.append("energy matches")
                if sa >= 1.0 or sb >= 1.0:
                    why.append("on section changes")
                cands.append({"a_start": 0.0, "a_end": round(a1, 3), "a_blend": round(a0, 3), "b_in": round(b0, 3),
                              "b_full": round(bf, 3), "b_end": round(dur_b, 3), "bars": bars, "profile": profile,
                              "score": round(score, 3), "parts": {k: round(v, 2) for k, v in parts.items()},
                              "why": why})
    cands.sort(key=lambda c: -c["score"])
    apart_a = lambda c, p: abs(c["a_blend"] - p["a_blend"]) >= MIN_APART * ga["bar_len"] - 1e-3   # noqa: E731
    apart_b = lambda c, p: abs(c["b_full"] - p["b_full"]) >= MIN_APART * gb["bar_len"] - 1e-3     # noqa: E731
    picked = []
    # Different in both tracks first (real alternatives), then in either — so there are always `top`.
    for need_both in (True, False):
        for c in cands:
            if len(picked) >= top:
                break
            if c in picked:
                continue
            if all((apart_a(c, p) and apart_b(c, p)) if need_both else (apart_a(c, p) or apart_b(c, p))
                   for p in picked):
                picked.append(c)
    picked.sort(key=lambda c: -c["score"])
    return picked


def recipe_pieces(track_a_id: str, track_b_id: str, t: dict, a_from: float = None, b_bars_after: int = None,
                  bar_b: float = None) -> list:
    """The two mashup pieces for transition t: A up to a_end (from a_from, default 16 bars before its
    blend), then B from b_full (its bars before b_full play under A — the 🧠 neural join), with the
    neural join set to t's profile and length."""
    a0 = t["a_blend"] - 16 * (t["a_end"] - t["a_blend"]) / t["bars"] if a_from is None else a_from
    b_end = t["b_end"] if not (b_bars_after and bar_b) else min(t["b_end"], t["b_full"] + b_bars_after * bar_b)
    return [{"track": track_a_id, "start": round(max(0.0, a0), 3), "end": t["a_end"], "gain_db": 0.0, "part": "full"},
            {"track": track_b_id, "start": t["b_full"], "end": round(b_end, 3), "gain_db": 0.0, "part": "full",
             "xfade": {"neural": {"profile": t["profile"], "bars": int(t["bars"])}}}]
