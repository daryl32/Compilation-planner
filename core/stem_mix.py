"""🧠 Semi-automatic multi-stem transitions (in the spirit of djay's Neural Mix).

Two decks, each split into stems — vocals, drums, bass and other (harmonics / everything else). Instead
of one volume crossfade, a transition is planned from the music's structure and then runs itself:

1. **Blueprint** (TransitionPlan / plan_transition): where the mix starts in deck A (a phrase line
   far enough from the end), where deck B joins (its first downbeat, or a chosen bar), how many bars
   the blend lasts (4 / 8 / 16) and the profile (Vocal swap, Drums first, Bass drop, Smooth).

2. **Automation** (Progressor): once deck A reaches the start point, t runs 0 → 1 over exactly the
   blend's length in samples, worked out from the BPM and the sample rate and advanced inside the
   audio callback per sample frame — no timers or sleeps, so it can't drift. The crossfader is
   x = 2t − 1 (−1 = all A, +1 = all B).

3. **Stem curves** (stem_gains): t is mapped onto each stem separately —
     vocals  — a sharp sigmoid that swaps around the middle (never two vocals at once), followed by
               a tiny one-pole smoother (a few ms) so the swap can't click;
     drums   — linear in for the incoming deck, a faster (squared) fade out for the outgoing one, so
               two kick drums never stack up into a muddy, clipping low end;
     bass    — a hard swap on a beat (over BASS_SWAP_BEATS): only one deck's low end at a time,
               the DJ bass-swap trick — a muddy double bass line is the first thing to give a blend away;
     other   — constant power (sin / cos): the harmonic bed stays at an even level throughout.
   A profile shifts *when* each stem moves (e.g. Drums first brings B's drums in during the first
   half; Bass drop holds A's drums and harmonics until the drop and swaps them on its downbeat).

4. **Headroom** (soft_knee): the summed output goes through a soft-knee limiter rather than a hard
   clip — untouched up to 0.80, then smoothly bent towards (never past) 1.0. The bend's slope matches
   at the knee, so peaks are rounded rather than squared off.

The block callback (StemMixer.process) does the mixing as one vectorised product: a gain matrix
[stem lanes × frames] times the stem buffers [stem lanes × frames × channels].

Pure NumPy + SciPy (the smoother's IIR filter keeps its state between blocks). No Streamlit, no I/O.
Run `python -m core.stem_mix` for the lifecycle demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

STEMS = ("vocals", "drums", "bass", "other")
PROFILES = {
    "smooth":      "Smooth — every stem blends across the whole transition.",
    "vocal_swap":  "Vocal swap — the beds blend, the vocals swap sharply in the middle.",
    "drums_first": "Drums first — B's drums come in during the first half; vocals and harmonics follow.",
    "bass_drop":   "Bass drop — B's vocals tease in, then drums and harmonics swap on the drop (the last bar).",
}
SOFT_KNEE = 0.80          # output level where the limiter starts to bend
VOCAL_STEEPNESS = 18.0    # sigmoid slope — higher = a sharper vocal swap
VOCAL_SMOOTH_MS = 4.0     # one-pole smoother on the vocal gains (anti-click)
DRUM_OUT_POWER = 2.0      # outgoing drums fade as (1 − t)^p — faster than the incoming ones rise
BASS_SWAP_BEATS = 0.25    # how long the bass hand-over takes (a quarter beat — no click, no overlap)


# ---------------------------------------------------------------------------
# 1. The blueprint
# ---------------------------------------------------------------------------

@dataclass
class TransitionPlan:
    """When and how deck A hands over to deck B. Times are seconds in each deck's own audio."""
    mix_start_point: float               # in deck A: where the automation takes over
    mix_sync_target: float               # in deck B: where B is playing from as the blend starts
    transition_duration_bars: int = 8
    mix_profile: str = "smooth"
    bpm: float = 120.0                   # the (matched) tempo both decks play at
    beats_per_bar: int = 4
    notes: list = field(default_factory=list)

    def __post_init__(self):
        if self.mix_profile not in PROFILES:
            raise ValueError(f"unknown mix profile {self.mix_profile!r} — one of {', '.join(PROFILES)}")
        if self.transition_duration_bars <= 0 or self.bpm <= 0:
            raise ValueError("transition length and BPM must be positive")

    @property
    def duration_sec(self) -> float:
        return self.transition_duration_bars * self.beats_per_bar * 60.0 / self.bpm

    def duration_frames(self, sample_rate: int) -> int:
        return max(1, int(round(self.duration_sec * sample_rate)))

    def start_frame(self, sample_rate: int) -> int:
        return int(round(self.mix_start_point * sample_rate))


def _bars(track: dict) -> tuple:
    """(bar lines, bar length, bpm, beats per bar) from a catalogue track (core.mixing.grid)."""
    from core.mixing import grid
    g = grid(track)
    return np.asarray(g["bars"], dtype=float), float(g["bar_len"]), float(g["bpm"]), int(g["beats_per_bar"]), g


def plan_transition(track_a: dict, track_b: dict, bars: int = 8, profile: str = "smooth",
                    sync_bar: int = 0, a_from: float = 0.0, a_until: float | None = None,
                    b_at: float | None = None) -> TransitionPlan:
    """The blueprint for mixing catalogue track A into B.
    Start (A): the last strong phrase line (a section start or an 8/16-bar phrase) that still leaves
    `bars` bars before a_until (default: A's end) and is after a_from — else simply `bars` bars before
    a_until, on a bar line.
    Sync (B): its bar line number `sync_bar` (0 = the first downbeat), or — with b_at, the point B's
    part should be fully in by (e.g. its drop) — `bars` bars before b_at, on a bar line.
    Tempo: A's (B is stretched to it, as the Mix tab does)."""
    bars_a, bar_a, bpm_a, bpb, ga = _bars(track_a)
    bars_b, bar_b, _, _, _ = _bars(track_b)
    length = bars * bar_a
    until = float(ga["duration"]) if a_until is None else min(float(a_until), float(ga["duration"]))
    latest = until - length
    if latest < max(0.0, a_from) - 1e-6:
        raise ValueError(f"not enough of track A for a {bars}-bar transition")
    phrase_lines = sorted({float(s["start"]) for s in ga["sections"]}
                          | {float(bars_a[i]) for i, lvl in ga["levels"].items() if lvl >= 8 and i < len(bars_a)})
    cands = [t for t in phrase_lines if max(0.0, a_from) < t <= latest + 1e-6]
    notes = []
    if cands:
        start = cands[-1]
        notes.append(f"Starts on the phrase line at {start:.2f}s in A.")
    else:
        on_bar = bars_a[(bars_a <= latest + 1e-6) & (bars_a >= max(0.0, a_from) - 1e-6)]
        start = float(on_bar[-1]) if len(on_bar) else max(0.0, latest)
        notes.append(f"No phrase line in reach — starts {bars} bars before the end, on a bar line.")
    if b_at is not None:
        want = float(b_at) - bars * bar_b
        sync = float(bars_b[int(np.argmin(np.abs(bars_b - want)))]) if len(bars_b) else max(0.0, want)
        sync = max(0.0, sync)
    else:
        sync = float(bars_b[min(max(0, sync_bar), len(bars_b) - 1)]) if len(bars_b) else 0.0
    return TransitionPlan(start, sync, bars, profile, bpm_a, bpb, notes)


# ---------------------------------------------------------------------------
# 2. Automation: t, frame-accurate
# ---------------------------------------------------------------------------

class Progressor:
    """Advances the transition's t (0 → 1) by sample frames inside the audio callback.
    Idle until deck A's playhead reaches the start frame; then t = frames since start / length."""

    IDLE, RUNNING, DONE = "idle", "running", "done"

    def __init__(self, plan: TransitionPlan, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self.start = plan.start_frame(self.sample_rate)
        self.length = plan.duration_frames(self.sample_rate)

    def t_block(self, a_frame: int, n: int) -> np.ndarray:
        """t for each of the n frames from deck A's playhead a_frame (0 before the start, 1 after)."""
        k = (a_frame - self.start) + np.arange(n, dtype=np.float64)
        return np.clip(k / self.length, 0.0, 1.0)

    def state(self, a_frame: int) -> str:
        if a_frame < self.start:
            return self.IDLE
        return self.RUNNING if a_frame < self.start + self.length else self.DONE

    @staticmethod
    def crossfader(t):
        """The virtual crossfader position: −1 (all A) … +1 (all B)."""
        return 2.0 * np.asarray(t) - 1.0


# ---------------------------------------------------------------------------
# 3. Per-stem curves
# ---------------------------------------------------------------------------

def _window(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """t re-timed so this stem moves only between a and b of the transition."""
    return np.clip((t - a) / max(b - a, 1e-9), 0.0, 1.0)


# When each stem moves within the transition, per profile: (start, end) as fractions of t.
_TIMING = {
    "smooth":      {"vocals": (0.0, 1.0), "drums": (0.0, 1.0), "bass": (0.0, 1.0), "other": (0.0, 1.0)},
    "vocal_swap":  {"vocals": (0.0, 1.0), "drums": (0.0, 1.0), "bass": (0.0, 1.0), "other": (0.0, 1.0)},
    "drums_first": {"vocals": (0.4, 1.0), "drums": (0.0, 0.5), "bass": (0.0, 0.5), "other": (0.25, 1.0)},
    "bass_drop":   {"vocals": (0.0, 0.85), "drums": (0.92, 1.0), "bass": (1.0, 1.0), "other": (0.88, 1.0)},
}


def sigmoid_swap(t: np.ndarray, steepness: float = VOCAL_STEEPNESS) -> np.ndarray:
    """Incoming gain 0 → 1, swapping around t = 0.5; rescaled so it's exactly 0 at 0 and 1 at 1."""
    s = 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))
    lo, hi = 1.0 / (1.0 + np.exp(steepness * 0.5)), 1.0 / (1.0 + np.exp(-steepness * 0.5))
    return (s - lo) / (hi - lo)


def bass_swap(t: np.ndarray, window: tuple, total_beats: float | None) -> np.ndarray:
    """Incoming bass gain: 0 until the swap beat, then 1, handed over in BASS_SWAP_BEATS. The swap
    beat is the beat nearest the middle of the stem's window (its end for a window ending at 1,
    e.g. 🥁 Bass drop: the drop's downbeat). total_beats = the transition's length in beats (None:
    a short ramp around the middle)."""
    a, b = window
    centre = 1.0 if b - a < 1e-9 else (a + b) / 2
    if total_beats and total_beats > 0:
        centre = round(centre * total_beats) / total_beats
        width = BASS_SWAP_BEATS / total_beats
    else:
        width = 0.01
    lo = min(centre, 1.0 - width) if centre >= 1.0 - 1e-9 else centre - width / 2
    return np.clip((t - lo) / width, 0.0, 1.0)


def stem_gains(t: np.ndarray, profile: str = "smooth", total_beats: float | None = None) -> dict:
    """{stem: (gain out for deck A, gain in for deck B)} — arrays the shape of t (before smoothing).
    total_beats (the transition's length in beats) puts the bass swap on a beat."""
    tm = _TIMING[profile]
    tv, td, to = (_window(t, *tm[s]) for s in ("vocals", "drums", "other"))
    v_in = sigmoid_swap(tv) if profile != "smooth" else np.sin(0.5 * np.pi * tv)
    v_out = 1.0 - v_in if profile != "smooth" else np.cos(0.5 * np.pi * tv)
    d_in = td
    d_out = (1.0 - td) ** DRUM_OUT_POWER
    o_in, o_out = np.sin(0.5 * np.pi * to), np.cos(0.5 * np.pi * to)
    b_in = bass_swap(np.asarray(t, dtype=float), tm["bass"], total_beats)
    return {"vocals": (v_out, v_in), "drums": (d_out, d_in), "bass": (1.0 - b_in, b_in), "other": (o_out, o_in)}


class OnePole:
    """A one-pole low-pass on a gain lane, its state kept between blocks (anti-click smoothing)."""

    def __init__(self, sample_rate: int, ms: float = VOCAL_SMOOTH_MS, start: float = 0.0):
        self.a = float(np.exp(-1.0 / max(1e-6, ms * 1e-3 * sample_rate)))
        self.zi = np.array([start * self.a])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter([1.0 - self.a], [1.0, -self.a], x, zi=self.zi)
        return y


# ---------------------------------------------------------------------------
# 4. Headroom
# ---------------------------------------------------------------------------

def soft_knee(x: np.ndarray, knee: float = SOFT_KNEE, ceiling: float = 1.0) -> np.ndarray:
    """Untouched up to the knee; above it, bent smoothly towards (never past) the ceiling:
    y = knee + (ceiling − knee)·tanh((|x| − knee)/(ceiling − knee)), sign kept. Same slope (1) at the
    knee, so there's no corner — peaks are rounded, not squared off like np.clip."""
    x = np.asarray(x, dtype=np.float64)
    room = ceiling - knee
    mag = np.abs(x)
    over = mag > knee
    out = x.copy()
    out[over] = np.sign(x[over]) * (knee + room * np.tanh((mag[over] - knee) / room))
    return out


# ---------------------------------------------------------------------------
# The mixer: the real-time block callback
# ---------------------------------------------------------------------------

class StemMixer:
    """Two decks of stems through one planned transition.

    decks: {"A": {stem: array [frames, channels]}, "B": {...}} at sample_rate, both already at the
    plan's tempo (and key). Deck A plays from a_start; deck B is cued so that it's at
    plan.mix_sync_target when the transition starts, and plays on from there. process(n) returns
    the next n output frames — call it from the audio callback."""

    def __init__(self, decks: dict, plan: TransitionPlan, sample_rate: int, a_start: float = 0.0):
        self.sr = int(sample_rate)
        self.plan = plan
        self.prog = Progressor(plan, self.sr)
        self.channels = next(iter(decks["A"].values())).shape[1]
        self.lanes = []                              # (deck, stem, audio) in gain-matrix row order
        for deck in ("A", "B"):
            for stem in STEMS:
                audio = decks[deck].get(stem)
                if audio is None:
                    audio = np.zeros((1, self.channels))
                self.lanes.append((deck, stem, np.asarray(audio, dtype=np.float32)))
        self.a_pos = int(round(a_start * self.sr))
        self.b_offset = int(round(plan.mix_sync_target * self.sr)) - self.prog.start   # B frame = A frame + this
        self.smooth = {("A", "vocals"): OnePole(self.sr, start=1.0), ("B", "vocals"): OnePole(self.sr, start=0.0)}
        self.peak_in = 0.0

    @property
    def state(self) -> str:
        return self.prog.state(self.a_pos)

    def _slice(self, audio: np.ndarray, start: int, n: int) -> np.ndarray:
        """n frames from start (silence outside the audio)."""
        out = np.zeros((n, self.channels), dtype=np.float32)
        a, b = max(start, 0), min(start + n, len(audio))
        if b > a:
            out[a - start:b - start] = audio[a:b]
        return out

    def gains(self, t: np.ndarray) -> np.ndarray:
        """The gain matrix [lanes, frames] for this block's t (vocals smoothed)."""
        g = stem_gains(t, self.plan.mix_profile, self.plan.transition_duration_bars * self.plan.beats_per_bar)
        rows = []
        for deck, stem, _ in self.lanes:
            row = g[stem][0 if deck == "A" else 1]
            sm = self.smooth.get((deck, stem))
            rows.append(sm(row) if sm else row)
        return np.vstack(rows)

    def process(self, n: int) -> np.ndarray:
        t = self.prog.t_block(self.a_pos, n)
        G = self.gains(t)                                                    # [lanes, n]
        b_pos = self.a_pos + self.b_offset
        X = np.stack([self._slice(audio, self.a_pos if deck == "A" else b_pos, n)
                      for deck, _, audio in self.lanes])                   # [lanes, n, ch]
        y = np.einsum("ln,lnc->nc", G, X, optimize=True)                    # Σ gain × stem, per frame
        self.peak_in = max(self.peak_in, float(np.abs(y).max(initial=0.0)))
        self.a_pos += n
        return soft_knee(y).astype(np.float32)


# ---------------------------------------------------------------------------
# Demo — the auto-transition lifecycle on synthetic stems
# ---------------------------------------------------------------------------

def _synth_deck(sr: int, seconds: float, bpm: float, root_hz: float, seed: int) -> dict:
    """Vocals (a wobbling tone), drums (a kick on every beat), bass (a pulsing low note), other (a
    chord pad). One deck alone
    stays under 1.0; two summed in the middle of the blend go past it, so the limiter has work to do."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    tt = np.arange(n) / sr
    beat = 60.0 / bpm
    phase = (tt % beat) / beat
    kick = 0.45 * np.sin(2 * np.pi * 55 * tt) * np.exp(-phase * 12)
    bass = 0.25 * np.sin(2 * np.pi * root_hz / 4 * tt) * (0.5 + 0.5 * (phase < 0.5))
    pad = 0.12 * sum(np.sin(2 * np.pi * root_hz * r * tt) for r in (1.0, 1.25, 1.5))
    vox = 0.25 * np.sin(2 * np.pi * root_hz * 2 * tt + 3 * np.sin(2 * np.pi * 5 * tt)) \
        * (0.6 + 0.4 * np.sin(2 * np.pi * tt / (4 * beat)))
    noise = 0.02 * rng.standard_normal(n)
    st = lambda x: np.stack([x, x], axis=1).astype(np.float32)   # noqa: E731
    return {"vocals": st(vox + noise), "drums": st(kick), "bass": st(bass), "other": st(pad)}


def demo(profile: str = "vocal_swap", block: int = 512, sr: int = 44100, verbose: bool = True) -> dict:
    """Mix two synthetic 24 s decks at 120 BPM with an 8-bar transition starting at 8 s in A."""
    bpm = 120.0
    decks = {"A": _synth_deck(sr, 24.0, bpm, 220.0, 1), "B": _synth_deck(sr, 24.0, bpm, 196.0, 2)}
    plan = TransitionPlan(mix_start_point=8.0, mix_sync_target=0.0, transition_duration_bars=8,
                          mix_profile=profile, bpm=bpm)
    mixer = StemMixer(decks, plan, sr)
    out, log, last = [], [], None
    total = int(sr * (plan.mix_start_point + plan.duration_sec + 2.0))
    while mixer.a_pos < total:
        state = mixer.state
        if state != last:
            log.append((round(mixer.a_pos / sr, 3), state))
            if verbose:
                print(f"{mixer.a_pos / sr:7.3f}s  {state}")
            last = state
        if verbose and state == Progressor.RUNNING and (mixer.a_pos // block) % 86 == 0:
            t = float(mixer.prog.t_block(mixer.a_pos, 1)[0])
            g = stem_gains(np.array([t]), profile, plan.transition_duration_bars * plan.beats_per_bar)
            print(f"           t={t:4.2f} x={Progressor.crossfader(t):+5.2f}  " + "  ".join(
                f"{s} A{g[s][0][0]:.2f}/B{g[s][1][0]:.2f}" for s in STEMS))
        out.append(mixer.process(block))
    y = np.concatenate(out)
    res = {"plan": plan, "log": log, "output": y, "peak_before_limiter": mixer.peak_in,
           "peak_after": float(np.abs(y).max())}
    if verbose:
        print(f"Peak before the limiter {res['peak_before_limiter']:.2f} → after {res['peak_after']:.2f}")
    return res


if __name__ == "__main__":
    import sys
    demo(sys.argv[1] if len(sys.argv) > 1 else "vocal_swap")


# ---------------------------------------------------------------------------
# Offline: one join of a rendered mix (mixer.render)
# ---------------------------------------------------------------------------

def render_region(a: dict, b: dict, profile: str, total_beats: float, sample_rate: int,
                  smooth_ms: float = VOCAL_SMOOTH_MS) -> np.ndarray:
    """The blend itself, offline: a / b = {stem: float32 [n, channels]} — each deck's stems over the
    transition, already lined up (same length, at the mix's tempo). t runs 0 → 1 over the region;
    returns the summed stems (no limiter — the mixer applies soft_knee to the whole mix).
    Any stem a deck lacks is silence."""
    n = max(len(x) for x in [*a.values(), *b.values()])
    ch = next(iter(a.values())).shape[1]
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    g = stem_gains(t, profile, total_beats)
    out = np.zeros((n, ch), dtype=np.float64)
    for deck, side, start in ((a, 0, 1.0), (b, 1, 0.0)):
        for stem in STEMS:
            x = deck.get(stem)
            if x is None:
                continue
            gain = g[stem][side]
            if stem == "vocals":
                gain = OnePole(sample_rate, smooth_ms, start=start)(gain)
            m = min(len(x), n)
            out[:m] += gain[:m, None] * x[:m]
    return out.astype(np.float32)
