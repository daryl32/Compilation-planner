"""
🎯 Movement ↔ music sync at the footage's own frame rate (11 Nov 2026).

The footage timeline's half-second movement can't show movement on the beat: at 120 BPM a beat IS half a second, so
one point per beat is all it has. The person curve it is averaged from has 15 points a second. This module compares
that curve with the music — the drive curve of audio_drive (the kick + snare, the bass, the voice or the loudness,
whichever carries each bar) — for EVERY frame a clip could start on, and reports for each half-second start:
    • sync 0–1: how well the movement in the window follows the music (normalised cross-correlation, the standard
      template-matching measure: 1 = same shape, 0.5 = unrelated, 0 = opposite). Level-blind on purpose — whether the
      footage is as lively as the music is a separate score in the matcher;
    • slide: how far to move the clip's start (within ± a quarter second, a frame at a time) so its pushes of
      movement land on the music's hits.
One FFT per (video, stretch of music): every start frame of a 30-minute video is scored in a couple of
milliseconds.

No Streamlit in here.
"""

from __future__ import annotations

import numpy as np

SMOOTH_SEC = 0.2          # a drum hit is a spike, a push of movement lasts a few frames: both are smoothed this wide
MIN_FRAMES = 8            # shorter than this there is nothing to compare
SHORT_FRAMES = 30         # under ~2 s a high correlation is partly luck: it is pulled towards "unrelated"
MIN_TRACKED = 0.7         # a window with somebody tracked for less than this share is "unrelated"
MAX_TEMPLATE = 1024       # frames (68 s at 15 a second) — longer stretches are compared on the half seconds as before


def pool(curve, m: int) -> np.ndarray:
    """A curve brought to m points keeping its area (mean over each new point's span) — a plain resample would step
    over the spikes of a drum curve."""
    c = np.asarray(curve, np.float64)
    if m <= 0:
        return np.zeros(0)
    if len(c) == 0:
        return np.zeros(m)
    if len(c) == 1:
        return np.full(m, float(c[0]))
    cs = np.concatenate([[0.0], np.cumsum(c)])
    edges = np.linspace(0.0, len(c), m + 1)
    area = np.interp(edges, np.arange(len(c) + 1), cs)
    return np.diff(area) / np.maximum(np.diff(edges), 1e-9)


def smooth(x, rate: float, sec: float = SMOOTH_SEC) -> np.ndarray:
    x = np.asarray(x, np.float64)
    k = int(round(sec * rate))
    if k < 2 or len(x) < k + 2:
        return x
    w = np.hanning(k + 2)[1:-1]
    return np.convolve(x, w / w.sum(), mode="same")


class Mover:
    """One video's full-rate movement, prepared once: gaps (nobody tracked) filled, smoothed, its running sums and
    its Fourier transform."""

    def __init__(self, move_hi, rate: float):
        from scipy import fft as _fft
        raw = np.asarray(move_hi, np.float64)
        self.rate = float(rate or 15.0)
        self.tracked = np.isfinite(raw).astype(np.float64)
        x = np.where(np.isfinite(raw), raw, 0.0)
        self.x = smooth(x, self.rate)
        self.n = len(self.x)
        self._c1 = np.concatenate([[0.0], np.cumsum(self.x)])
        self._c2 = np.concatenate([[0.0], np.cumsum(self.x * self.x)])
        self._ct = np.concatenate([[0.0], np.cumsum(self.tracked)])
        self._L = _fft.next_fast_len(self.n + MAX_TEMPLATE, real=True) if self.n else 0
        self._F = _fft.rfft(self.x, self._L) if self.n else None

    def correlate(self, template) -> np.ndarray:
        """Normalised cross-correlation (−1…1) of the movement from every start frame with the template; 0 where
        the movement is flat or nobody is tracked."""
        from scipy import fft as _fft
        a = np.asarray(template, np.float64)
        m = len(a)
        P = self.n - m + 1
        if P <= 0 or m < MIN_FRAMES or m > MAX_TEMPLATE:
            return np.zeros(max(P, 0))
        a = a - a.mean()
        a_std = float(np.sqrt((a * a).mean()))
        if a_std < 1e-9:
            return np.zeros(P)
        num = _fft.irfft(self._F * np.conj(_fft.rfft(a, self._L)), self._L)[:P]
        mean = (self._c1[m:m + P] - self._c1[:P]) / m
        var = np.maximum((self._c2[m:m + P] - self._c2[:P]) / m - mean * mean, 0.0)
        std = np.sqrt(var)
        tracked = (self._ct[m:m + P] - self._ct[:P]) / m
        with np.errstate(divide="ignore", invalid="ignore"):
            ncc = np.where((std > 1e-6) & (tracked >= MIN_TRACKED), num / (m * std * a_std), 0.0)
        return np.clip(ncc, -1.0, 1.0)


def template(audio_curve, seconds: float, rate: float) -> np.ndarray:
    """The music for a stretch, at the movement curve's rate, its hits widened to the size of a movement."""
    m = int(round(seconds * rate))
    return smooth(pool(audio_curve, m), rate) if m >= 1 else np.zeros(0)


def sync_scores(mover: Mover, times, n: int, seconds: float, audio_curve, slide_sec: float = 0.25) -> tuple:
    """(sync 0–1, slide seconds) for every half-second window start s in 0 … len(times) − n: the best the window
    does when its start may move by up to ± slide_sec, and that move. None when it can't be worked out (no
    full-rate curve, the stretch too short or too long, silence) — the caller falls back to the half seconds."""
    T = len(times)
    W = T - n + 1
    if mover is None or mover.n == 0 or W <= 0 or audio_curve is None or len(audio_curve) < 2:
        return None
    tpl = template(audio_curve, seconds, mover.rate)
    m = len(tpl)
    if m < MIN_FRAMES or m > MAX_TEMPLATE or mover.n < m or float(np.std(tpl)) < 1e-9:
        return None
    ncc = mover.correlate(tpl)
    P = len(ncc)
    h = max(0, int(round(slide_sec * mover.rate)))
    p0 = np.round(np.asarray(times[:W], np.float64) * mover.rate).astype(int)
    grid = p0[:, None] + np.arange(-h, h + 1)[None, :]
    ok = (grid >= 0) & (grid < P)
    vals = np.where(ok, ncc[np.clip(grid, 0, P - 1)], -2.0)
    best = vals.argmax(axis=1)
    corr = vals[np.arange(W), best]
    # the move is measured from the half-second start itself to the chosen FRAME's time (frames and half seconds
    # don't share a grid at 15 a second: 5.5 s falls between two frames)
    slide = (p0 + best - h) / mover.rate - np.asarray(times[:W], np.float64)
    none = corr < -1.5                                   # the window runs off the end of the curve
    corr = np.where(none, 0.0, corr) * min(1.0, m / SHORT_FRAMES)
    slide = np.where(none, 0.0, slide)
    return (corr + 1.0) / 2.0, slide


def sync_of(move_slice, audio_slice, rate: float) -> float | None:
    """One clip against its music, as played (no sliding): the same 0–1 measure, or None if it can't be said."""
    x = np.asarray(move_slice, np.float64)
    if len(x) < MIN_FRAMES or float(np.isfinite(x).mean()) < MIN_TRACKED:
        return None
    x = smooth(np.where(np.isfinite(x), x, 0.0), rate)
    a = smooth(pool(audio_slice, len(x)), rate)
    x, a = x - x.mean(), a - a.mean()
    sx, sa = float(np.sqrt((x * x).mean())), float(np.sqrt((a * a).mean()))
    if sx < 1e-6 or sa < 1e-9:
        return None
    return float((np.clip(float((x * a).mean()) / (sx * sa), -1.0, 1.0) * min(1.0, len(x) / SHORT_FRAMES) + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# 🪀 Bounce: is the movement a regular up-and-down, or smooth / irregular? (11 Nov 2026)
# ---------------------------------------------------------------------------
# The sync above is blind to WHAT KIND of movement it is: against a smooth curve (a voice, a pad) almost any
# movement correlates. Daryl, on the first real render (01 AURA): where the beat stopped, the footage went on
# bouncing as if it hadn't. Bouncy movement belongs on a beat; smooth or irregular movement on voice and
# instruments. So the footage gets a 0–1 "bounce" per half second, the music a 0–1 "beat presence" per stretch,
# and the matcher asks for the two to agree.

BOUNCE_WIN_SEC = 4.0      # the movement is judged over this long around each point (2.7 s clips hold 4 beats: too few)
BOUNCE_LAG_SEC = (0.25, 1.5)      # one bounce lasts between these (240 … 40 a minute)
BOUNCE_QUIET = 0.5        # windows moving less than this share of the video's usual swing count for less
BOUNCE_NOISE = 0.35       # the autocorrelation peak that random movement reaches by chance over the window
BEAT_OFF, BEAT_ON = 0.9, 1.25     # × the drum level a bar needs to count as drum-driven: no beat … a full beat
#                                   (0.5 at first: 01 AURA's beatless passages at 1:04 and 1:44, hits ≈ 0.3, then read
#                                   as a third of a beat and were given half-bouncy footage — Daryl hears no beat there)
FROZEN_BELOW = 1e-4       # a tracked person whose pose doesn't change at all: the picture is frozen (a live body always
FROZEN_SEC = 1.0          # jitters) — for at least this long


def bounce(move_hi, rate: float, times) -> tuple:
    """(strength 0–1, bounces a minute) at every time in `times`. Strength is the height of the first real peak of
    the movement's autocorrelation over BOUNCE_WIN_SEC around the point — a peak that RISES from a trough, so
    movement that merely changes slowly (whose autocorrelation only falls) scores 0. 0 where nobody is tracked or
    hardly anything moves; the tempo is NaN where there is no bounce."""
    raw = np.asarray(move_hi, np.float64)
    times = np.asarray(times, np.float64)
    T = len(times)
    rate = float(rate or 15.0)
    W = int(round(BOUNCE_WIN_SEC * rate))
    lo, hi = max(2, int(round(BOUNCE_LAG_SEC[0] * rate))), int(round(BOUNCE_LAG_SEC[1] * rate))
    if T == 0 or len(raw) < W or hi >= W // 2 + hi // 2 or hi <= lo:
        return np.zeros(T), np.full(T, np.nan)
    tracked = np.isfinite(raw)
    x = smooth(np.where(tracked, raw, 0.0), rate)
    centre = np.clip(np.round(times * rate).astype(int), 0, len(x) - 1)
    start = np.clip(centre - W // 2, 0, len(x) - W)
    idx = start[:, None] + np.arange(W)[None, :]
    win = x[idx]
    seen = tracked[idx].mean(axis=1)
    win = win - win.mean(axis=1, keepdims=True)
    std = np.sqrt((win * win).mean(axis=1))
    taper = np.hanning(W + 2)[1:-1]
    L = 1 << int(np.ceil(np.log2(2 * W)))
    spec = np.fft.rfft(win * taper, L, axis=1)
    ac = np.fft.irfft(spec.real ** 2 + spec.imag ** 2, L, axis=1)[:, :hi + 1]
    tw = np.fft.rfft(taper, L)
    ac_t = np.fft.irfft(tw.real ** 2 + tw.imag ** 2, L)[:hi + 1]              # what the taper alone does to each lag
    with np.errstate(divide="ignore", invalid="ignore"):
        ac = np.where(ac[:, :1] > 1e-12, ac / ac[:, :1], 0.0) / (ac_t / ac_t[0])[None, :]
    trough = np.minimum.accumulate(ac, axis=1)
    peak = np.clip(np.minimum(ac, ac - trough), 0.0, 1.0)[:, lo:hi + 1]       # high AND risen from a trough
    top = peak.max(axis=1)
    best = (peak >= 0.9 * top[:, None]).argmax(axis=1)          # the first peak that high: one bounce, not two
    # a short stretch of random movement still shows a peak of about BOUNCE_NOISE by chance: only what is above counts
    strength = np.clip((top - BOUNCE_NOISE) / (1.0 - BOUNCE_NOISE), 0.0, 1.0)
    usual = float(np.percentile(std[seen >= MIN_TRACKED], 75)) if (seen >= MIN_TRACKED).any() else 0.0
    loud_enough = np.clip(std / max(BOUNCE_QUIET * usual, 1e-9), 0.0, 1.0) if usual > 1e-9 else np.zeros(T)
    strength = np.where(seen >= MIN_TRACKED, strength * loud_enough, 0.0)
    bpm = np.where(strength > 0.05, 60.0 * rate / (best + lo), np.nan)
    return strength, bpm


def beat_presence(drums, drum_level: float) -> float | None:
    """0–1: is there a beat in this stretch of the drum curve (kick + snare, 0–1 within the track)? Judged on the
    hits (the 90th percentile — drums are spikes, a strong beat still has a low average). None without drums."""
    d = np.asarray(drums, np.float64) if drums is not None else np.zeros(0)
    if not len(d):
        return None
    hits = float(np.percentile(d, 90))
    return float(np.clip((hits - BEAT_OFF * drum_level) / max((BEAT_ON - BEAT_OFF) * drum_level, 1e-9), 0.0, 1.0))


def frozen(move_hi, rate: float, times, step: float) -> np.ndarray:
    """Share (0–1) of each point's `step` seconds in which the picture is frozen: somebody is tracked and doesn't
    move AT ALL for FROZEN_SEC or more — a still frame held in the source file, or the curve running past the
    video's end (01 AURA's first clip froze 3.5 s in: the movement there is exactly 0 from that frame on)."""
    raw = np.asarray(move_hi, np.float64)
    times = np.asarray(times, np.float64)
    rate = float(rate or 15.0)
    still = np.isfinite(raw) & (np.abs(np.where(np.isfinite(raw), raw, 1.0)) < FROZEN_BELOW)
    if not still.any():
        return np.zeros(len(times))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], still.astype(np.int8), [0]])))
    held = np.zeros(len(raw))
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) / rate >= FROZEN_SEC:
            held[a:b] = 1.0
    c = np.concatenate([[0.0], np.cumsum(held)])
    lo = np.clip(np.round(times * rate).astype(int), 0, len(raw))
    hi = np.clip(np.round((times + step) * rate).astype(int), 0, len(raw))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
