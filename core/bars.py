"""📏 A complete, regular bar grid — one bar line every beats_per_bar beats, the whole way through.

The analysis's bar lines (beat_this downbeats, lined up with the track's beats) can be patchy: a
bar line that doesn't land within a hair of an existing beat used to be dropped, and the model
sometimes marks every other bar. 28 of 52 tracks (23 Oct 2026) had holes of 2–40 bars — so the
Mix's fine-tune jumped from 0:33 to 1:43, and cuts in those stretches couldn't land on a bar.

regular_bars() rebuilds the grid from the beats instead: every beats_per_bar-th beat, with beat 1
chosen by the model's bar lines voting (per stretch of CHUNK beats, so a phase that shifts — an
odd bar in a bridge — is followed). irregular() says when a track's stored bar lines need it.

Pure NumPy (also used by the laptop's analysis — audio_analysis.py).
"""

from __future__ import annotations

import numpy as np

CHUNK = 64            # beats per voting stretch (16 bars of 4)
VOTE_TOL = 0.25       # a bar line votes for the beat it's within this share of a beat of
MIN_VOTES = 3         # a stretch needs this many votes (and a clear majority) to choose its own beat 1
MAJORITY = 0.6


def irregular(downbeats, bar_len: float, max_gap: float = 1.5) -> bool:
    """True when stored bar lines can't be used as they are: fewer than two, a gap of more than
    max_gap bars anywhere, or a typical spacing that isn't one bar (e.g. every other bar)."""
    db = np.sort(np.asarray(downbeats if downbeats is not None else [], dtype=float))
    if len(db) < 2 or bar_len <= 0:
        return True
    gaps = np.diff(db)
    med = float(np.median(gaps))
    return bool(gaps.max() > max_gap * bar_len or not (0.8 * bar_len <= med <= 1.25 * bar_len))


def phase_votes(beats: np.ndarray, downbeats, beats_per_bar: int):
    """(global phase or None, {chunk: phase}) — which beat (index mod beats_per_bar) the stored bar
    lines say is beat 1, overall and per CHUNK-beat stretch."""
    beats = np.asarray(beats, dtype=float)
    db = np.asarray(downbeats if downbeats is not None else [], dtype=float)
    if len(beats) < 2 or not len(db):
        return None, {}
    step = float(np.median(np.diff(beats)))
    bpb = int(beats_per_bar)
    total = np.zeros(bpb)
    per_chunk: dict = {}
    for d in db:
        j = int(np.argmin(np.abs(beats - d)))
        if abs(beats[j] - d) <= VOTE_TOL * step:
            total[j % bpb] += 1
            per_chunk.setdefault(j // CHUNK, np.zeros(bpb))[j % bpb] += 1
    if not total.any():
        return None, {}
    glob = int(np.argmax(total))
    local = {}
    for c, v in per_chunk.items():
        if v.sum() >= MIN_VOTES and v.max() >= MAJORITY * v.sum():
            local[c] = int(np.argmax(v))
    return glob, local


def regular_bars(beats, downbeats, beats_per_bar: int = 4, fallback_phase: int | None = None) -> list:
    """Every beats_per_bar-th beat, beat 1 voted by the stored bar lines (per stretch); fallback_phase
    when no bar line lines up with a beat at all ([] if there's neither)."""
    beats = np.asarray(beats, dtype=float)
    bpb = int(beats_per_bar)
    if len(beats) < bpb:
        return []
    glob, local = phase_votes(beats, downbeats, bpb)
    if glob is None:
        if fallback_phase is None:
            return []
        glob = int(fallback_phase) % bpb
    step = float(np.median(np.diff(beats)))
    out = []
    for i, t in enumerate(beats):
        if (i - local.get(i // CHUNK, glob)) % bpb == 0:
            if out and t - out[-1] < 0.6 * bpb * step:      # beat 1 moved at a stretch boundary
                continue
            out.append(float(t))
    return out
