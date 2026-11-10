"""
🥁 What drives the music, bar by bar — for the 🤖 new matcher's cuts and movement match (10 Nov 2026).

A track's analysis holds several 30 Hz curves besides loudness: kick, snare and hat (the drums by band), low (the
bass), vocals, novelty ("something changed"), plus the bar lines, phrase lines and vocal phrase starts / ends.
Loudness alone hides what the music is doing: vocals and pads raise it without adding rhythm, and a quiet passage
can still have a clear pulse. So for every bar the DRIVER is chosen —
    drums   where there is a beat (kick + snare present),
    bass    where there is no drum beat but a bass line,
    vocals  where there is little beat or bass and mainly a voice,
    loudness otherwise —
and two things are built from it:
  • drive curve — what the footage's movement is compared with (core/story_matcher: the "shape" part of the fit):
    the driver's own curve inside each bar, so movement follows the drums in a drum passage and the voice in a
    vocal one;
  • accents — how good each beat is as a cut point (section_cuts): drum hits and changes on drum bars; on vocal
    bars the starts of vocal phrases, and never the middle of a phrase; bar lines and phrase lines always count.
Tracks without the extra analysis fall back to loudness and plain bar lines.

No Streamlit in here.
"""

from __future__ import annotations

import numpy as np

DRUM_LEVEL = 0.4        # a bar is drum-driven when its drum hits (kick + snare) reach this share of the track's strong hits
BASS_LEVEL = 0.35
VOCAL_LEVEL = 0.3
DRIVERS = ("drums", "bass", "vocals", "loudness")
EMOJI = {"drums": "🥁", "bass": "🎸", "vocals": "🎤", "loudness": "🔊"}

W_BAR, W_PHRASE, W_HIT, W_CHANGE, W_VOCAL_START, W_MID_PHRASE = 0.35, 0.35, 0.3, 0.3, 0.5, 0.9


def _norm(x) -> np.ndarray:
    """0–1 against the curve's own strong level (95th percentile), so tracks of any loudness compare."""
    x = np.asarray(x, float)
    if not x.size:
        return x
    top = float(np.percentile(x, 95))
    return np.clip(x / top, 0.0, 1.0) if top > 1e-9 else np.zeros_like(x)


def _curves(track: dict) -> tuple:
    """({name: 0–1 curve at one rate}, rate). Always has "loudness"; the rest when the analysis has them."""
    hires = track.get("hires") or {}
    a = track.get("analysis") or {}
    out = {}
    if hires.get("rms"):
        rate = float(hires.get("rate", 30))
        out["loudness"] = _norm(hires["rms"])
    else:
        env = track.get("energy_envelope") or []
        t = np.array([e["time"] for e in env], float)
        rate = 1.0 / max(float(np.median(np.diff(t))) if t.size > 1 else 0.5, 1e-3)
        out["loudness"] = _norm([e["energy"] for e in env])
    n = len(out["loudness"])
    a_rate = float(a.get("rate", rate))

    def get(key):
        v = a.get(key)
        if v is None or not len(v):
            return None
        v = np.asarray(v, float)
        if abs(a_rate - rate) > 1e-6 or len(v) != n:           # onto the loudness curve's clock
            v = np.interp(np.arange(n) / rate, np.arange(len(v)) / a_rate, v)
        return v
    kick, snare = get("kick"), get("snare")
    if kick is not None:
        out["drums"] = _norm(kick + (snare if snare is not None else 0.0))
    low = get("low")
    if low is not None:
        out["bass"] = _norm(low)
    voc = get("vocals")
    if voc is not None:
        out["vocals"] = _norm(voc)
    nov = get("novelty")
    if nov is not None:
        out["change"] = _norm(nov)
    if hires.get("onset"):
        out["onset"] = _norm(hires["onset"])
    return out, rate


def bar_lines(track: dict, beats=None) -> np.ndarray:
    """The track's bar lines: the analysis' downbeats, with the stretches it left without any (quiet openings,
    breakdowns — Body had none for 0–12 s and 85–98 s) filled in from the beats, counted the way the downbeats that
    are there count them. Without downbeats: every fourth beat."""
    a = track.get("analysis") or {}
    b = np.asarray(beats if beats is not None else track.get("beat_times") or [], float)
    down = np.asarray(a.get("downbeats") or [], float)
    if not down.size:
        return b[::4]
    if b.size < 8:
        return down
    idx = np.clip(np.searchsorted(b, down), 0, len(b) - 1)
    prev = np.clip(idx - 1, 0, len(b) - 1)
    idx = np.where(np.abs(b[prev] - down) < np.abs(b[idx] - down), prev, idx)
    phase = int(np.bincount(idx % 4, minlength=4).argmax())           # how the downbeats count the beats
    bar = float(np.median(np.diff(b))) * 4
    grid = b[phase::4]
    pos = np.clip(np.searchsorted(down, grid), 0, len(down) - 1)
    before = np.clip(pos - 1, 0, len(down) - 1)
    gap = np.minimum(np.abs(down[pos] - grid), np.abs(down[before] - grid))
    return np.sort(np.concatenate([down, grid[gap > 0.6 * bar]]))


def drivers(track: dict, beats=None) -> dict:
    """{"bars": bar start times (from 0), "driver": name per bar, "curve": the drive curve, "rate", "curves"}."""
    curves, rate = _curves(track)
    n = len(curves["loudness"])
    duration = float(track.get("duration_sec") or n / rate)
    bars = bar_lines(track, beats)
    bars = bars[(bars > 0.05) & (bars < duration - 0.05)]
    bars = np.concatenate([[0.0], bars])
    ends = np.append(bars[1:], duration)
    names, levels, drive = [], [], np.zeros(n)
    for a_, b_ in zip(bars, ends):
        lo, hi = int(a_ * rate), max(int(a_ * rate) + 1, min(int(b_ * rate), n))
        mean = lambda k: float(curves[k][lo:hi].mean()) if k in curves and hi > lo else 0.0     # noqa: E731
        # drums are spikes — a bar with a strong beat still has a low average — so they're judged on their hits
        hits = float(np.percentile(curves["drums"][lo:hi], 90)) if "drums" in curves and hi > lo else 0.0
        if hits >= DRUM_LEVEL:
            d = "drums"
        elif mean("bass") >= BASS_LEVEL:
            d = "bass"
        elif mean("vocals") >= VOCAL_LEVEL:
            d = "vocals"
        else:
            d = "loudness"
        names.append(d)
        levels.append({"drums": round(hits, 3), "bass": round(mean("bass"), 3), "vocals": round(mean("vocals"), 3),
                       "loudness": round(mean("loudness"), 3)})
    # one bar that differs from both its neighbours is noise, not a change of driver (Body's verse flickered
    # drums / vocals bar by bar): it takes its neighbours' driver
    for k in range(1, len(names) - 1):
        if names[k - 1] == names[k + 1] != names[k]:
            names[k] = names[k - 1]
    for (a_, b_), d in zip(zip(bars, ends), names):
        lo, hi = int(a_ * rate), max(int(a_ * rate) + 1, min(int(b_ * rate), n))
        drive[lo:hi] = curves[d][lo:hi]
    return {"bars": bars, "driver": names, "levels": levels, "curve": drive, "rate": rate, "curves": curves,
            "duration": duration}


def drive_slice(drv: dict, t0: float, t1: float) -> np.ndarray:
    """The drive curve from t0 to t1 (what a block's footage movement is compared with)."""
    rate, c = drv["rate"], drv["curve"]
    lo, hi = int(t0 * rate), max(int(t0 * rate) + 1, int(t1 * rate))
    return c[lo:hi]


def driver_at(drv: dict, t: float) -> str:
    k = int(np.searchsorted(drv["bars"], t, side="right")) - 1
    return drv["driver"][min(max(k, 0), len(drv["driver"]) - 1)] if drv["driver"] else "loudness"


def driver_shares(drv: dict) -> dict:
    """{driver: share of the track's bars} — for a status line."""
    names = drv["driver"]
    return {d: round(names.count(d) / len(names), 3) for d in DRIVERS if names.count(d)} if names else {}


def accents(track: dict, beats, drv: dict = None) -> np.ndarray:
    """How good each beat is as a cut point, 0 upwards (one value per beat time)."""
    b = np.asarray(beats, float)
    if not b.size:
        return np.zeros(0)
    drv = drv or drivers(track, b)
    curves, rate = drv["curves"], drv["rate"]
    a = track.get("analysis") or {}
    beat = float(np.median(np.diff(b))) if b.size > 1 else 0.5

    def at(key, half=0.06):
        c = curves.get(key)
        if c is None or not len(c):
            return np.zeros(len(b))
        w = max(1, int(half * rate))
        idx = np.clip((b * rate).astype(int), 0, len(c) - 1)
        return np.array([c[max(0, i - w):i + w + 1].max() for i in idx])

    def near(times, tol):
        ts = np.asarray(times, float)
        if not ts.size:
            return np.zeros(len(b), bool)
        pos = np.clip(np.searchsorted(ts, b), 0, len(ts) - 1)
        prev = np.clip(pos - 1, 0, len(ts) - 1)
        return np.minimum(np.abs(ts[pos] - b), np.abs(ts[prev] - b)) <= tol
    tol = 0.3 * beat
    out = W_BAR * near(bar_lines(track, b), tol).astype(float)
    phrases = [p["time"] for p in a.get("phrases") or [] if isinstance(p, dict) and "time" in p]
    out += W_PHRASE * near(phrases, tol)
    out += W_CHANGE * at("change", 0.15)
    hit = at("drums") if "drums" in curves else at("onset")
    vs, ve = np.asarray(a.get("vocal_starts") or [], float), np.asarray(a.get("vocal_ends") or [], float)
    v_start = near(vs, 0.6 * beat)
    inside = np.zeros(len(b), bool)
    for s_, e_ in zip(vs, ve):
        inside |= (b > s_ + 0.6 * beat) & (b < e_ - 0.6 * beat)
    which = np.array([driver_at(drv, float(t)) for t in b])
    vocal_bar = which == "vocals"
    out += np.where(vocal_bar, 0.3 * W_HIT * hit, W_HIT * hit)          # drum hits matter little under a voice
    out += np.where(vocal_bar, W_VOCAL_START * v_start, 0.4 * W_VOCAL_START * v_start)
    out -= np.where(vocal_bar, W_MID_PHRASE * inside, 0.25 * W_MID_PHRASE * inside)   # don't cut mid-phrase
    return out
