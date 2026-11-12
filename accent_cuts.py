"""⚡ Accent override — the moments of a track that stand out, and the block cuts they change.

An accent is one instant the ear jumps at:
  🥁 drum   a drum hit far above the hits around it (a crash, a fill's last hit, the kick coming back);
  🎹 stab   the instruments hitting hard while the drums are quiet (needs the laptop's stem curves);
  🎤 voice  a sung phrase starting;
  🔊 jump   the loudness stepping up sharply.
A run of them within a beat of each other (a fill, a roll) is one accent: the strongest.
Each has a strength 0–1; the 📐 page's slider says how strong one must be to count.

The override works on the blocks of either cut mode (section_bounds.py or section_bounds_old.py — neither is
changed), after the 🎤 vocal rule. Section and part lines never move. For each accent, strongest first:
  - it already sits on a cut → nothing to do;
  - a cut inside the part lies within a beat of it → that cut is pulled onto the accent;
  - otherwise the block is split at the accent;
and either only when the blocks on both sides stay at least the ACCENT minimum: half the shortest clip for the
tempo (section_cuts.auto_min_clip), never under ACCENT_FLOOR. That is the "two minimums" of 12 Nov: the normal one
everywhere, half of it only where a strong accent falls.

No Streamlit in here.
"""

from __future__ import annotations

import numpy as np

import section_cuts as SCU

KINDS = {"drum": "🥁 a drum hit far above the usual", "stab": "🎹 an instruments stab with no drums",
         "voice": "🎤 a sung phrase starts", "jump": "🔊 a sharp jump in loudness"}
NEED = 0.6                 # default for the slider: an accent counts from this strength
FLOOR = 0.3                # weaker than this is never an accent (the slider's lowest setting)
MERGE_SEC = 0.2            # accents this close are one accent (the strongest's time, both reasons)
SNAP_SEC = 0.09            # an accent this near a beat sits on the beat
ON_CUT_SEC = 0.06          # an accent this near a cut is already cut on
ACCENT_FLOOR = 0.75        # no block shorter than this, whatever the tempo (the matcher's smallest part)
DRUM_FAR = (1.3, 2.2)      # a hit this many times the usual hit around it: strength 0 → 1
DRUM_REAL = 0.5            # … and at least this share of the track's strong hits (a tap in a quiet part isn't one)
STAB_RISE = (0.35, 0.9)    # the instruments rising by this share of their strong level within a tenth of a second
STAB_DRUMS = 0.3           # … while the drums are under this share of their strong level
JUMP_RISE = (0.25, 0.6)    # the loudness stepping up by this share of the track's range
VOICE = (0.55, 0.75)       # a phrase start: straight after another phrase / after two seconds without singing


def _ramp(x: float, lo_hi: tuple) -> float:
    lo, hi = lo_hi
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _peaks(c: np.ndarray) -> np.ndarray:
    """Indices of the local maxima of a curve."""
    if len(c) < 3:
        return np.zeros(0, int)
    return np.flatnonzero((c[1:-1] >= c[:-2]) & (c[1:-1] > c[2:])) + 1


def _raw(track: dict) -> tuple:
    """({"drums", "other", "loudness"} — the curves as measured, not capped at their strong level — , rate)."""
    hires = track.get("hires") or {}
    a = track.get("analysis") or {}
    rms = np.asarray(hires.get("rms") or [], float)
    if not rms.size:
        return {}, 30.0
    rate, n = float(hires.get("rate", 30)), len(rms)

    def onto(v, r):
        if v is None or not len(v):
            return None
        v = np.asarray(v, float)
        if abs(float(r) - rate) > 1e-6 or len(v) != n:
            v = np.interp(np.arange(n) / rate, np.arange(len(v)) / float(r), v)
        return v
    a_rate = float(a.get("rate", rate))
    stems = a.get("stem_curves") or {}
    s_rate = float(stems.get("rate", a_rate))
    share = stems.get("share") or {}
    out = {"loudness": rms}
    drums = onto(stems.get("drums"), s_rate) if float(share.get("drums", 1.0)) >= 0.08 else None
    if drums is None:
        kick, snare = onto(a.get("kick"), a_rate), onto(a.get("snare"), a_rate)
        if kick is not None:
            drums = np.maximum(kick, snare) if snare is not None else kick
    if drums is not None:
        out["drums"] = drums
    other = onto(stems.get("other"), s_rate) if float(share.get("other", 1.0)) >= 0.08 else None
    if other is not None:
        out["other"] = other
    return out, rate


def find(track: dict) -> list:
    """Every accent of a track from FLOOR up, in time order: {"time", "strength" 0–1, "kinds", "why"}."""
    curves, rate = _raw(track)
    if not curves:
        return []
    a = track.get("analysis") or {}
    n = len(curves["loudness"])
    beats = np.asarray(track.get("beat_times") or (), float)
    beat = float(np.median(np.diff(beats))) if beats.size > 2 else 0.5
    found = []                                                # (time, strength, kind)
    drums = curves.get("drums")
    d_top = float(np.percentile(drums, 95)) if drums is not None else 0.0
    if drums is not None and d_top > 1e-9:
        pk = _peaks(drums)
        pk = pk[drums[pk] >= 0.25 * d_top]                    # the hits
        if len(pk) >= 4:
            usual_all = float(np.median(drums[pk]))
            span = int(round(8 * beat * rate))                # two bars either side
            for i in pk:
                if drums[i] < DRUM_REAL * d_top:
                    continue
                near = pk[(np.abs(pk - i) <= span) & (pk != i)]
                usual = float(np.median(drums[near])) if len(near) >= 3 else usual_all
                s = _ramp(drums[i] / max(usual, 1e-9), DRUM_FAR)
                if s > 0:
                    found.append((i / rate, s, "drum"))
    other = curves.get("other")
    if other is not None:
        o_top = float(np.percentile(other, 95))
        k = max(1, int(round(0.1 * rate)))
        if o_top > 1e-9 and n > k + 2:
            rise = np.zeros(n)
            rise[:-k] = np.maximum(other[k:] - other[:-k], 0.0)
            for i in _peaks(rise):
                lo, hi = max(0, i - k), min(n, i + 2 * k + 1)
                if drums is not None and d_top > 1e-9 and float(drums[lo:hi].max()) >= STAB_DRUMS * d_top:
                    continue
                s = _ramp(rise[i] / o_top, STAB_RISE)
                if s > 0:
                    found.append(((i + k / 2) / rate, s, "stab"))
    loud = curves["loudness"]
    rng = float(np.percentile(loud, 95) - np.percentile(loud, 5))
    w_after, w_before = max(2, int(round(0.3 * rate))), max(2, int(round(0.5 * rate)))
    if rng > 1e-9 and n > w_after + w_before + 2:
        csum = np.concatenate([[0.0], np.cumsum(loud)])
        idx = np.arange(w_before, n - w_after)
        step = (csum[idx + w_after] - csum[idx]) / w_after - (csum[idx] - csum[idx - w_before]) / w_before
        for j in _peaks(step):
            s = _ramp(step[j] / rng, JUMP_RISE)
            if s > 0:
                found.append((idx[j] / rate, s, "jump"))
    ends = np.sort(np.asarray(a.get("vocal_ends") or (), float))
    for t in a.get("vocal_starts") or ():
        before = ends[ends <= float(t) + 1e-6]
        quiet = float(t) - (float(before[-1]) if len(before) else 0.0)
        found.append((float(t), VOICE[1] if quiet >= 2.0 or not len(before) else VOICE[0], "voice"))
    found.sort()
    merged = []
    for t, s, kind in found:
        if merged and t - merged[-1]["first"] <= MERGE_SEC:
            m = merged[-1]
            if s > m["strength"]:
                m["time"], m["strength"] = t, s
            if kind not in m["kinds"]:
                m["kinds"].append(kind)
        else:
            merged.append({"first": t, "time": t, "strength": s, "kinds": [kind]})
    out = []
    for m in merged:
        s = min(1.0, m["strength"] + (0.1 if len(m["kinds"]) > 1 else 0.0))     # two things at once: a little stronger
        if s < FLOOR:
            continue
        t = m["time"]
        if beats.size:
            j = int(np.argmin(np.abs(beats - t)))
            if abs(float(beats[j]) - t) <= SNAP_SEC:
                t = float(beats[j])
        if out and abs(t - out[-1]["time"]) < 1e-6:           # two snapped onto one beat
            if s > out[-1]["strength"]:
                out[-1].update(strength=round(s, 2))
            out[-1]["kinds"] = out[-1]["kinds"] + [k for k in m["kinds"] if k not in out[-1]["kinds"]]
            out[-1]["why"] = " + ".join(KINDS[k] for k in out[-1]["kinds"])
            continue
        out.append({"time": round(float(t), 3), "strength": round(float(s), 2), "kinds": list(m["kinds"]),
                    "why": " + ".join(KINDS[k] for k in m["kinds"])})
    # a fill or a roll is one accent, not one per hit: within a beat only the strongest stays (the earlier of equals)
    keep = []
    for i, a_ in enumerate(out):
        rivals = [b_ for j, b_ in enumerate(out) if j != i and abs(b_["time"] - a_["time"]) <= beat + 1e-6]
        if all(b_["strength"] < a_["strength"] or (b_["strength"] == a_["strength"] and b_["time"] > a_["time"])
               for b_ in rivals):
            keep.append(a_)
    return keep


def minimums(found: dict, beat: float = None) -> tuple:
    """(the normal shortest block, the accent one) in seconds for a track cut as `found`."""
    bars = np.asarray(found.get("bars") or (), float)
    bar_sec = float(np.median(np.diff(bars))) if len(bars) > 1 else 2.0
    normal = SCU.auto_min_clip(beat if beat and beat > 0 else bar_sec / 4)
    return normal, max(ACCENT_FLOOR, normal / 2)


def apply(found: dict, accents: list, need: float = NEED, beat: float = None) -> dict:
    """`found` (for_track of either cut module) with its blocks re-cut around the accents of strength ≥ need.
    Adds "accents": the ones that counted, each with "did" — "on a cut" / "moved a cut" / "added a cut" / "no room"."""
    blocks = found.get("blocks") or []
    if not blocks:
        return {**found, "accents": []}
    duration = float(found["duration"])
    bars = np.asarray(found.get("bars") or (), float)
    bar_sec = float(np.median(np.diff(bars))) if len(bars) > 1 else 2.0
    beat = float(beat) if beat and beat > 0 else bar_sec / 4
    _, short = minimums(found, beat)
    lines = {round(float(ln["time"]), 2) for ln in found.get("lines") or []}
    cuts = [0.0] + [float(b["start"]) for b in blocks[1:]] + [duration]
    fixed = [True] + [round(c, 2) in lines for c in cuts[1:-1]] + [True]       # lines and the track's ends stay
    by_accent = set()
    used = []
    for acc in sorted((a for a in accents if a["strength"] >= need - 1e-9), key=lambda a: -a["strength"]):
        t = float(acc["time"])
        entry = {**acc, "did": "no room"}
        used.append(entry)
        if min(abs(c - t) for c in cuts) <= ON_CUT_SEC:
            entry["did"] = "on a cut"
            continue
        k = int(np.searchsorted(cuts, t))                     # cuts[k - 1] < t < cuts[k]
        if k <= 0 or k >= len(cuts):
            continue
        p, q = cuts[k - 1], cuts[k]
        moved = False
        for j in sorted((k - 1, k), key=lambda j: abs(cuts[j] - t)):           # the nearer cut first
            if fixed[j] or abs(cuts[j] - t) > beat + 1e-6:
                continue
            left, right = cuts[j - 1], cuts[j + 1]
            if t - left >= short - 1e-6 and right - t >= short - 1e-6:
                cuts[j], fixed[j] = t, True
                by_accent.add(round(t, 3))
                entry["did"] = "moved a cut"
                moved = True
                break
        if moved:
            continue
        if t - p >= short - 1e-6 and q - t >= short - 1e-6:
            cuts.insert(k, t)
            fixed.insert(k, True)
            by_accent.add(round(t, 3))
            entry["did"] = "added a cut"
    out = []
    secs = found.get("sections") or []
    for x0, x1 in zip(cuts[:-1], cuts[1:]):
        mid = (x0 + x1) / 2
        src = next((b for b in blocks if b["start"] - 1e-6 <= mid < b["end"]), blocks[-1])
        sec = next((s for s in secs if s["start"] - 1e-6 <= mid < s["end"]), None)
        if abs(src["start"] - x0) < 1e-6 and abs(src["end"] - x1) < 1e-6:      # untouched: exactly as it was
            out.append({**src, "n": len(out) + 1})
            continue
        accent = round(x0, 3) in by_accent or round(x1, 3) in by_accent
        out.append({"n": len(out) + 1, "start": round(x0, 3), "end": round(x1, 3),
                    "bars": round((x1 - x0) / bar_sec, 2),
                    "section": (sec.get("label") or str(sec["n"])) if sec else src["section"],
                    "why": src["why"] + (" · ⚡ accent" if accent else "")})
    used.sort(key=lambda a: a["time"])
    return {**found, "blocks": out, "accents": used}
