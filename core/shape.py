"""Shape matching: how well a clip's motion curve follows a block's music
energy curve (rises and falls in the same places), searched over every
window position at once.

Pure functions. Moved unchanged from the planner page."""

from __future__ import annotations

import numpy as np


SHAPE_MATCH_POINTS = 40      # both curves are resampled to this many points before comparing


def resample_curve(values, n_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.full(n_points, float(values[0]) if len(values) else 0.0)
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_old, values)


def shape_score(curve_a, curve_b, n_points: int = SHAPE_MATCH_POINTS) -> float:
    """Pearson correlation between two curves of possibly different length /
    sample rate, after resampling both to n_points and z-scoring — so this
    measures PATTERN similarity (rises and falls in the same places),
    independent of either curve's absolute scale. Range roughly [-1, 1];
    higher is a better shape match. Returns 0.0 for a degenerate (flat)
    curve, since correlation is undefined there."""
    a = resample_curve(curve_a, n_points)
    b = resample_curve(curve_b, n_points)
    a_std, b_std = a.std(), b.std()
    if a_std < 1e-9 or b_std < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _make_interp_matrix(src_len: int, n_points: int) -> np.ndarray:
    """Precomputed linear-interpolation weights: resampling any uniformly-
    sampled src_len-point signal to n_points is a FIXED linear combination —
    it depends only on src_len/n_points, never on the values themselves. So
    this matrix, built once, lets many candidate windows get resampled at
    once via a single matrix multiply (W @ values, or windows @ W.T for many
    windows stacked as rows) instead of calling np.interp per window.
    Verified to reproduce np.interp's own output exactly."""
    if src_len < 2:
        return np.ones((n_points, max(src_len, 1)))
    x_old = np.linspace(0, 1, src_len)
    x_new = np.linspace(0, 1, n_points)
    idx = np.clip(np.searchsorted(x_old, x_new, side="right") - 1, 0, src_len - 2)
    x0, x1 = x_old[idx], x_old[idx + 1]
    frac = np.where(x1 > x0, (x_new - x0) / (x1 - x0), 0.0)
    W = np.zeros((n_points, src_len))
    rows = np.arange(n_points)
    W[rows, idx] += 1 - frac
    W[rows, idx + 1] += frac
    return W


def find_best_shape_window(values: np.ndarray, window_frames: int, audio_curve: np.ndarray,
                           n_points: int = SHAPE_MATCH_POINTS, bias=None) -> tuple:
    """Vectorized replacement for sliding shape_score() across every
    candidate start position one at a time: resamples EVERY window in a
    single matrix multiply and scores them all in one batched correlation,
    instead of a Python loop calling np.interp + np.corrcoef per position.
    Mathematically identical to shape_score() at each position (verified
    directly against it) — the only behavioural change is that it's cheap
    enough to check every single frame position, not just every
    SHAPE_SEARCH_STEP_SEC, so it can also find a better-aligned match than
    the old stepped search could. Returns (best_score, best_start_frame)."""
    n_positions = len(values) - window_frames + 1
    if n_positions <= 0:
        return -2.0, 0

    windows = np.lib.stride_tricks.sliding_window_view(values, window_frames)  # (n_positions, window_frames), no copy
    W = _make_interp_matrix(window_frames, n_points)
    resampled = windows @ W.T  # (n_positions, n_points)

    audio_r = resample_curve(audio_curve, n_points)
    audio_c = audio_r - audio_r.mean()
    audio_std = audio_r.std()

    row_mean = resampled.mean(axis=1, keepdims=True)
    row_std = resampled.std(axis=1)
    numerator = (resampled - row_mean) @ audio_c
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where((row_std > 1e-9) & (audio_std > 1e-9),
                          numerator / (n_points * row_std * audio_std), 0.0)

    # bias (optional, one value per start position): added only when CHOOSING the
    # position — e.g. weight F's pull toward a target part of the video — while
    # the returned score stays the plain shape score.
    if bias is not None and len(bias) == len(scores):
        best_idx = int(np.argmax(scores + bias))
    else:
        best_idx = int(np.argmax(scores))
    return float(scores[best_idx]), best_idx


def get_block_audio_curve(track: dict, start_sec: float, end_sec: float) -> np.ndarray:
    """The audio energy curve for one block's time range, at whatever
    resolution the track has (30 Hz hires if available, else the coarse
    envelope)."""
    hires = track.get("hires")
    if hires:
        rate = float(hires["rate"])
        rms = np.asarray(hires["rms"], dtype=float)
        lo, hi = int(start_sec * rate), max(int(start_sec * rate) + 1, int(end_sec * rate))
        return rms[lo:hi]
    env = track["energy_envelope"]
    slice_ = np.array([e["energy"] for e in env if start_sec <= e["time"] < end_sec])
    return slice_ if slice_.size > 0 else np.array([e["energy"] for e in env])
