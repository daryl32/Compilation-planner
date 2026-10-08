"""
Server-side audio analysis — finds good places to cut in a music track.

Two jobs:
  • "new"   — a track in the Audio-Library with no catalogue yet: full analysis,
              writing the same fields audio_pipeline.py (Colab) writes
              (duration_sec, bpm, beat_times, energy_envelope, hires{rate,rms,onset})
              plus the extra analysis below.
  • "extra" — a track already analysed in Colab: keeps its beats/energy exactly
              as they are and only ADDS the extra analysis.

Extra analysis, stored under track["analysis"] (all curves at 30 Hz, 0–1):
  downbeats      real bar starts (beat_this model if installed, else a
                 kick/harmony-based estimate of where "beat 1" falls)
  kick/snare/hat percussive onsets split by frequency band (low / mid / high)
  harmony        how much the chords/notes are changing
  brightness     spectral centroid (rises during build-ups)
  novelty        how much the SOUND is changing (timbre + harmony) — section changes
  sections       [{start, end, label, energy}] — label letters repeat for
                 sections that sound alike (approximate verse/chorus repeats)
  phrases        [{time, level}] — every 4 / 8 / 16 bars from each section start
  builds, drops  build-ups ([{start, end}]) and the drop each leads into
  vocals         vocal loudness curve + vocal_starts / vocal_ends — only when
                 Demucs is installed (otherwise None)

Needs only numpy, scipy and ffmpeg. Optional, used automatically if installed:
  pip install beat_this    (better downbeats; needs torch)
  pip install demucs       (vocals; needs torch, slow on CPU)

Run from the command line:
  python audio_analysis.py --worker            process the queue (the app starts this)
  python audio_analysis.py --file song.m4a     analyse one file, print a summary
No Streamlit in here.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

ANALYSIS_VERSION = 2
SR = 22050
N_FFT = 2048
HOP = 512
FPS = SR / HOP              # feature frames per second (~43)
OUT_RATE = 30               # stored curves, same as hires
N_MELS = 96
ENERGY_ENVELOPE_SEC = 0.5   # coarse energy_envelope step for new tracks
AUDIO_EXTENSIONS = (".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus")


# ---------------------------------------------------------------------------
# Decoding and basic spectral features
# ---------------------------------------------------------------------------

def decode_audio(path, sr: int = SR, channels: int = 1) -> np.ndarray:
    """Whole file as float32 at sr (mono, or (n, channels)). Uses ffmpeg;
    -map 0:a:0 skips embedded album art in .m4a files."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
           "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='ignore')[-300:]}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    return y.reshape(-1, channels) if channels > 1 else y


def _frames(y: np.ndarray, n: int = N_FFT, hop: int = HOP) -> np.ndarray:
    y = np.pad(y, (n // 2, n // 2), mode="reflect" if len(y) > n else "constant")
    count = 1 + max(0, (len(y) - n) // hop)
    return np.lib.stride_tricks.as_strided(
        y, shape=(count, n), strides=(y.strides[0] * hop, y.strides[0]), writeable=False)


def stft_mag(y: np.ndarray) -> np.ndarray:
    """|STFT|, shape (frames, N_FFT//2 + 1), computed in chunks to bound memory."""
    fr = _frames(y)
    win = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((fr.shape[0], N_FFT // 2 + 1), dtype=np.float32)
    for i in range(0, fr.shape[0], 2048):
        out[i:i + 2048] = np.abs(np.fft.rfft(fr[i:i + 2048] * win, axis=1))
    return out


def frame_rms(y: np.ndarray) -> np.ndarray:
    fr = _frames(y)
    out = np.empty(fr.shape[0], dtype=np.float32)
    for i in range(0, fr.shape[0], 4096):
        out[i:i + 4096] = np.sqrt(np.mean(fr[i:i + 4096].astype(np.float64) ** 2, axis=1))
    return out


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = N_MELS, fmin: float = 30.0, fmax: float = SR / 2) -> tuple:
    """(filters (n_mels, bins), centre frequencies)."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(n_mels):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / max(hi - lo, 1e-9))
    return fb, edges[1:-1]


def chroma_from_stft(mag: np.ndarray) -> np.ndarray:
    """12-bin pitch-class energy per frame (55 Hz – 4.2 kHz), each frame summing to 1."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    sel = (freqs >= 55) & (freqs <= 4200)
    pc = (np.round(12 * np.log2(freqs[sel] / 440.0)) + 9).astype(int) % 12
    proj = np.zeros((len(pc), 12), dtype=np.float32)
    proj[np.arange(len(pc)), pc] = 1.0
    ch = (mag[:, sel] ** 2) @ proj
    return ch / np.maximum(ch.sum(axis=1, keepdims=True), 1e-9)


def mfcc_from_logmel(logmel: np.ndarray, n: int = 13) -> np.ndarray:
    from scipy.fft import dct
    return dct(logmel, type=2, axis=1, norm="ortho")[:, 1:n]


def _flux(spec: np.ndarray) -> np.ndarray:
    """Onset strength: summed positive change per frame (spectral flux)."""
    d = np.diff(spec, axis=0, prepend=spec[:1])
    return np.maximum(d, 0.0).sum(axis=1)


def _norm01(x: np.ndarray, pct: float = 99.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    hi = np.percentile(x, pct) if x.size else 1.0
    lo = float(x.min()) if x.size else 0.0
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _smooth(x: np.ndarray, frames: int) -> np.ndarray:
    frames = max(1, int(frames))
    return ndimage.uniform_filter1d(np.asarray(x, dtype=float), frames, mode="nearest")


def to_rate(x: np.ndarray, n_out: int, src_rate: float = FPS) -> np.ndarray:
    """Resample a per-frame curve onto the OUT_RATE grid (n_out samples)."""
    t_src = np.arange(len(x)) / src_rate
    t_out = np.arange(n_out) / OUT_RATE
    return np.interp(t_out, t_src, x)


# ---------------------------------------------------------------------------
# Tempo and beats (for brand-new tracks) — Ellis-style dynamic programming
# ---------------------------------------------------------------------------

def estimate_tempo(onset_env: np.ndarray, fps: float = FPS, prior_bpm: float = 120.0) -> float:
    o = onset_env - onset_env.mean()
    ac = signal.fftconvolve(o, o[::-1], mode="full")[len(o) - 1:]
    lags = np.arange(len(ac))
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 60)
    lags, ac = lags[lo:hi + 1], ac[lo:hi + 1]
    bpms = 60.0 * fps / np.maximum(lags, 1)
    weight = np.exp(-0.5 * (np.log2(bpms / prior_bpm) / 1.0) ** 2)
    k = int(np.argmax(ac * weight))
    lag = float(lags[k])
    if 0 < k < len(ac) - 1:   # refine between whole frames (parabolic peak)
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            lag += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 60.0 * fps / lag


def track_beats(onset_env: np.ndarray, bpm: float, fps: float = FPS, tightness: float = 100.0) -> np.ndarray:
    """Beat frame indices: best path through onset peaks at roughly the tempo."""
    period = 60.0 * fps / bpm
    o = onset_env / (onset_env.std() + 1e-9)
    win = np.exp(-0.5 * (np.arange(-period, period + 1) * 32.0 / period) ** 2)
    local = np.convolve(o, win, mode="same")
    n = len(local)
    back = np.full(n, -1, dtype=int)
    cum = np.zeros(n)
    lo_off, hi_off = int(round(period / 2)), int(round(2 * period))
    offsets = np.arange(lo_off, hi_off + 1)
    penalty = -tightness * np.log(offsets / period) ** 2
    for i in range(n):
        prev = i - offsets
        ok = prev >= 0
        if ok.any():
            cand = cum[prev[ok]] + penalty[ok]
            j = int(np.argmax(cand))
            if cand[j] > 0:
                cum[i] = local[i] + cand[j]
                back[i] = prev[ok][j]
                continue
        cum[i] = local[i]
    # last beat: the latest strong local maximum of the cumulative score
    peaks = signal.argrelmax(cum, order=max(1, int(period / 2)))[0]
    if not len(peaks):
        return np.array([], dtype=int)
    good = peaks[cum[peaks] > 0.5 * np.median(cum[peaks])]
    i = int(good[-1] if len(good) else peaks[-1])
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats = np.array(beats[::-1])
    # trim weak beats at the very start/end (silence, fades)
    strength = local[beats]
    thr = 0.05 * np.sqrt(np.mean(strength ** 2))
    keep = np.flatnonzero(strength > thr)
    return beats[keep[0]:keep[-1] + 1] if len(keep) else beats


# ---------------------------------------------------------------------------
# Downbeats
# ---------------------------------------------------------------------------

def _beat_values(curve: np.ndarray, beat_times: np.ndarray, rate: float, radius_sec: float = 0.07) -> np.ndarray:
    """Max of curve within ±radius around each beat."""
    r = max(1, int(radius_sec * rate))
    out = []
    for t in beat_times:
        i = int(round(t * rate))
        seg = curve[max(0, i - r): i + r + 1]
        out.append(float(seg.max()) if len(seg) else 0.0)
    return np.array(out)


def estimate_downbeat_phase(beat_times: np.ndarray, kick: np.ndarray, harmony: np.ndarray,
                            energy_jump: np.ndarray, beats_per_bar: int = 4, rate: float = OUT_RATE) -> int:
    """Which beat (0..beats_per_bar-1) is "beat 1": the phase where kicks,
    chord changes and energy jumps line up best."""
    if len(beat_times) < beats_per_bar * 2:
        return 0
    k = _beat_values(kick, beat_times, rate)
    h = _beat_values(harmony, beat_times, rate)
    e = _beat_values(energy_jump, beat_times, rate)
    score = []
    for p in range(beats_per_bar):
        idx = np.arange(p, len(beat_times), beats_per_bar)
        score.append(1.0 * k[idx].mean() + 1.5 * h[idx].mean() + 1.0 * e[idx].mean()
                     - (k.mean() + 1.5 * h.mean() + e.mean()))
    return int(np.argmax(score))


def downbeats_from_model(path) -> np.ndarray | None:
    """beat_this (CPJKU) downbeats, or None if it isn't installed / fails."""
    try:
        from beat_this.inference import File2Beats  # type: ignore
    except Exception:
        return None
    try:
        f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        _beats, downbeats = f2b(str(path))
        return np.asarray(downbeats, dtype=float)
    except Exception:
        return None


def _snap_to_beats(times: np.ndarray, beat_times: np.ndarray, max_dist: float = 0.12) -> list:
    out = []
    for t in times:
        if not len(beat_times):
            break
        j = int(np.argmin(np.abs(beat_times - t)))
        if abs(beat_times[j] - t) <= max_dist:
            out.append(float(beat_times[j]))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Sections (self-similarity novelty) and repeats
# ---------------------------------------------------------------------------

def _beat_sync(feat: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    bounds = np.concatenate([[0], beat_frames, [len(feat)]])
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows.append(feat[a:b].mean(axis=0) if b > a else feat[min(a, len(feat) - 1)])
    return np.array(rows)


def _checkerboard(L: int) -> np.ndarray:
    x = np.arange(-L, L) + 0.5
    g = np.exp(-0.5 * (x / (L / 2.0)) ** 2)
    sign = np.sign(x)
    return np.outer(g * sign, g * sign)


def foote_novelty(ssm: np.ndarray, L: int) -> np.ndarray:
    n = len(ssm)
    k = _checkerboard(L)
    pad = np.pad(ssm, L, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        nov[i] = float((pad[i:i + 2 * L, i:i + 2 * L] * k).sum())
    return np.maximum(nov, 0.0)


def find_sections(beat_feats: np.ndarray, beat_times: np.ndarray, downbeats: list,
                  duration: float, energy_at_beats: np.ndarray, min_bars: int = 4,
                  beats_per_bar: int = 4) -> tuple:
    """(sections, beat-level novelty). Boundaries are novelty peaks, moved
    onto the nearest downbeat; sections that sound alike share a label."""
    n = len(beat_feats)
    if n < 16:
        return [{"start": 0.0, "end": duration, "label": "A", "energy": 0.5}], np.zeros(n)
    f = (beat_feats - beat_feats.mean(axis=0)) / (beat_feats.std(axis=0) + 1e-9)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    ssm = f @ f.T
    nov = foote_novelty(ssm, L=min(16, n // 4))
    nov = nov / (nov.max() + 1e-9)

    min_gap = min_bars * beats_per_bar
    peaks = signal.argrelmax(nov, order=max(2, min_gap // 2))[0]
    thr = nov.mean() + 0.25 * nov.std()
    peaks = [p for p in peaks if nov[p] >= thr]
    chosen = []
    for p in sorted(peaks, key=lambda p: -nov[p]):          # strongest first, keep spacing
        if p >= min_gap // 2 and n - p >= min_gap // 2 and all(abs(p - c) >= min_gap for c in chosen):
            chosen.append(p)
    times = []
    db = np.asarray(downbeats, dtype=float)
    for p in sorted(chosen):
        t = float(beat_times[p])
        if len(db):
            j = int(np.argmin(np.abs(db - t)))
            beat_len = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
            if abs(db[j] - t) <= 2.5 * beat_len:
                t = float(db[j])
        times.append(t)
    times = sorted(set(times))
    bounds = [0.0] + [t for t in times if 0.5 < t < duration - 0.5] + [duration]

    # labels: compare each section's average sound with earlier ones
    sec_feats, sec_energy = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = np.flatnonzero((beat_times >= a) & (beat_times < b))
        sec_feats.append(f[idx].mean(axis=0) if len(idx) else np.zeros(f.shape[1]))
        sec_energy.append(float(energy_at_beats[idx].mean()) if len(idx) else 0.0)
    sec_feats = np.array(sec_feats)
    sec_feats = sec_feats / (np.linalg.norm(sec_feats, axis=1, keepdims=True) + 1e-9)
    sims = sec_feats @ sec_feats.T
    offdiag = sims[~np.eye(len(sims), dtype=bool)] if len(sims) > 1 else np.array([0.0])
    same_thr = max(0.6, float(np.percentile(offdiag, 75)))
    labels, centroids = [], []
    for i in range(len(sec_feats)):
        best, best_sim = None, same_thr
        for lab, j in centroids:
            if sims[i, j] >= best_sim:
                best, best_sim = lab, sims[i, j]
        if best is None:
            best = chr(ord("A") + min(len(centroids), 25))
            centroids.append((best, i))
        labels.append(best)
    e = np.array(sec_energy)
    e = (e - e.min()) / (e.max() - e.min()) if e.max() - e.min() > 1e-9 else np.full_like(e, 0.5)
    sections = [{"start": round(a, 3), "end": round(b, 3), "label": lab, "energy": round(float(en), 2)}
                for a, b, lab, en in zip(bounds[:-1], bounds[1:], labels, e)]
    return sections, nov


def phrase_lines(sections: list, downbeats: list) -> list:
    """Every 4 bars from each section start; level 16 / 8 / 4 by position."""
    out = []
    db = sorted(downbeats)
    for sec in sections:
        bars = [t for t in db if sec["start"] - 0.05 <= t < sec["end"] - 0.05]
        for k in range(0, len(bars), 4):
            level = 16 if k % 16 == 0 else (8 if k % 8 == 0 else 4)
            out.append({"time": round(bars[k], 3), "level": level})
    return out


# ---------------------------------------------------------------------------
# Build-ups and drops
# ---------------------------------------------------------------------------

def find_builds_and_drops(low_energy: np.ndarray, energy: np.ndarray, brightness: np.ndarray,
                          downbeats: list, bar_sec: float, rate: float = OUT_RATE,
                          section_starts: list = ()) -> tuple:
    """Drops: big upward steps in bass/kick energy (2-bar step detector),
    moved onto the nearest downbeat. Build: the rising stretch (energy +
    brightness) leading into a drop, at least 2 bars long."""
    n = len(low_energy)
    if n < rate * 8:
        return [], []
    w = max(2, int(2 * bar_sec * rate))
    c = np.concatenate([[0.0], np.cumsum(low_energy)])
    idx = np.arange(n)
    b_lo, a_hi = np.maximum(0, idx - w), np.minimum(n, idx + w)
    before = (c[idx] - c[b_lo]) / np.maximum(1, idx - b_lo)
    after = (c[a_hi] - c[idx]) / np.maximum(1, a_hi - idx)
    step = after - before
    peaks = signal.argrelmax(step, order=w)[0]
    thr = max(0.15, float(np.percentile(step, 99)) * 0.6)
    db = np.asarray(sorted(downbeats), dtype=float)
    drops = []
    for p in peaks:
        if step[p] < thr:
            continue
        t = p / rate
        if len(db):
            j = int(np.argmin(np.abs(db - t)))
            if abs(db[j] - t) <= bar_sec * 0.6:
                t = float(db[j])
        drops.append(round(t, 3))
    drops = sorted(set(drops))

    rise = _smooth(0.5 * energy + 0.5 * brightness, int(rate * bar_sec / 2))
    builds = []
    for t in drops:
        end = int(t * rate)
        start_lim = max(0, end - int(16 * bar_sec * rate))
        seg = rise[start_lim:end]
        if len(seg) < rate * bar_sec * 2:
            continue
        start = start_lim + int(np.argmin(seg))
        # a section that starts inside the rise (e.g. a breakdown) is where the build begins
        inside = [x for x in section_starts if start / rate < x <= t - 2 * bar_sec]
        if inside:
            start = int(max(inside) * rate)
        if end - start >= 2 * bar_sec * rate and rise[end - 1] - rise[start] >= 0.08:
            builds.append({"start": round(start / rate, 3), "end": round(t, 3)})
    return builds, drops


# ---------------------------------------------------------------------------
# Vocals (optional — Demucs)
# ---------------------------------------------------------------------------

def demucs_available() -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "demucs"))


def beat_this_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("beat_this") is not None


def vocal_curve(path, n_out: int) -> np.ndarray | None:
    """Vocal loudness at OUT_RATE from a Demucs vocal stem, or None if Demucs
    isn't installed. Slow on CPU (a few minutes per song)."""
    try:
        import torch  # type: ignore
        from demucs.apply import apply_model  # type: ignore
        from demucs.pretrained import get_model  # type: ignore
    except Exception:
        return None
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        model = get_model("htdemucs")
        model.eval()
        y = decode_audio(path, sr=model.samplerate, channels=2).T.copy()
        wav = torch.from_numpy(y)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.no_grad():
            sources = apply_model(model, wav[None], device="cpu", split=True, overlap=0.1, progress=False)[0]
        vocals = sources[model.sources.index("vocals")].mean(0).numpy()
        hop = int(model.samplerate / OUT_RATE)
        nfr = len(vocals) // hop
        v = np.sqrt(np.mean(vocals[:nfr * hop].reshape(nfr, hop) ** 2, axis=1))
        v = _norm01(v, 99.0)
        if len(v) < n_out:
            v = np.pad(v, (0, n_out - len(v)))
        return v[:n_out]
    except Exception:
        traceback.print_exc()
        return None


def vocal_phrases(v: np.ndarray, rate: float = OUT_RATE, on: float = 0.18, off: float = 0.10,
                  min_on: float = 0.6, min_off: float = 0.35) -> tuple:
    """Starts / ends of sung phrases from the vocal curve (hysteresis)."""
    vs = _smooth(v, int(rate * 0.25))
    active, start, segs = False, 0, []
    for i, x in enumerate(vs):
        if not active and x >= on:
            active, start = True, i
        elif active and x < off:
            active = False
            segs.append([start, i])
    if active:
        segs.append([start, len(vs)])
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) / rate < min_off:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged if (e - s) / rate >= min_on]
    return ([round(s / rate, 3) for s, _ in merged], [round(e / rate, 3) for _, e in merged])


# ---------------------------------------------------------------------------
# The analysis itself
# ---------------------------------------------------------------------------

def _r(x, nd: int = 2) -> list:
    return [round(float(v), nd) for v in x]


def analyse_file(path, existing: dict = None, beats_per_bar: int = 4, with_vocals: bool = True,
                 progress=None) -> dict:
    """Analyse one audio file. existing: the track's current catalogue (Colab),
    whose beats/energy are kept. Returns a dict of fields to merge into the
    track JSON: always "analysis"; for a new track also the core fields."""
    say = progress or (lambda msg: None)
    say("decoding audio")
    y = decode_audio(path)
    duration = len(y) / SR
    say("spectrum")
    mag = stft_mag(y)
    fb, centres = mel_filterbank()
    mel = mag ** 2 @ fb.T
    logmel = np.log1p(1000.0 * mel / (mel.max() + 1e-12))
    rms = frame_rms(y)
    n_out = int(np.ceil(duration * OUT_RATE))

    # percussive part (median-filter HPSS on the mel spectrogram)
    harm = ndimage.median_filter(mel, size=(17, 1))
    perc = ndimage.median_filter(mel, size=(1, 9))
    pmask = perc ** 2 / (perc ** 2 + harm ** 2 + 1e-12)
    plog = np.log1p(1000.0 * (mel * pmask) / (mel.max() + 1e-12))

    def band(lo, hi, spec):
        sel = (centres >= lo) & (centres < hi)
        return _flux(spec[:, sel]) if sel.any() else np.zeros(len(spec))

    kick = _norm01(band(30, 150, plog))
    snare = _norm01(band(150, 300, plog) + band(1500, 5000, plog))
    hat = _norm01(band(7000, SR / 2, plog))
    onset_full = _norm01(_flux(logmel))
    chroma = chroma_from_stft(mag)
    harmony_frames = np.concatenate([[0.0], 1.0 - np.sum(chroma[1:] * chroma[:-1], axis=1)
                                     / (np.linalg.norm(chroma[1:], axis=1) * np.linalg.norm(chroma[:-1], axis=1) + 1e-9)])
    harmony = _norm01(_smooth(harmony_frames, int(FPS * 0.25)))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-9)
    brightness = _norm01(_smooth(centroid, int(FPS * 0.5)), 98)
    low_energy = _norm01(_smooth(np.sqrt(mel[:, centres < 150].sum(axis=1)), int(FPS * 0.25)))

    # beats: keep Colab's if present
    if existing and existing.get("beat_times"):
        beat_times = np.asarray(existing["beat_times"], dtype=float)
        bpm = float(existing.get("bpm") or 0) or (60.0 / float(np.median(np.diff(beat_times)))
                                                   if len(beat_times) > 1 else 120.0)
        beats_method = "existing"
    else:
        say("finding the beat")
        bpm = estimate_tempo(onset_full)
        beat_frames = track_beats(onset_full, bpm)
        beat_times = beat_frames / FPS
        if len(beat_times) > 2:   # average beat spacing (slope), not the frame-quantised median
            bpm = 60.0 / float(np.polyfit(np.arange(len(beat_times)), beat_times, 1)[0])
        beats_method = "server"

    to30 = lambda x: to_rate(x, n_out)
    kick30, snare30, hat30 = to30(kick), to30(snare), to30(hat)
    harmony30, bright30 = to30(harmony), to30(brightness)
    rms_norm = _norm01(rms, 100)
    rms30 = to30(rms_norm)
    low30 = to30(low_energy)
    energy_jump = np.maximum(0.0, np.diff(_smooth(rms30, 6), prepend=rms30[0]))
    energy_jump = _norm01(energy_jump, 99)

    say("finding downbeats")
    model_db = downbeats_from_model(path)
    if model_db is not None and len(model_db) >= 2 and len(beat_times):
        downbeats = _snap_to_beats(model_db, beat_times) or list(model_db)
        db_method = "beat_this"
    else:
        phase = estimate_downbeat_phase(beat_times, kick30, harmony30, energy_jump, beats_per_bar)
        downbeats = [float(t) for t in beat_times[phase::beats_per_bar]]
        db_method = "estimate"

    say("finding sections")
    beat_frames_f = np.clip(np.round(beat_times * FPS).astype(int), 0, len(mag) - 1)
    mfcc = mfcc_from_logmel(logmel)
    # timbre + harmony + "what's playing" (loudness, drums by band, brightness)
    inst = np.column_stack([np.log1p(50 * rms_norm), kick, snare, hat, brightness,
                            _smooth(low_energy, int(FPS))])
    feats = np.hstack([_beat_sync(mfcc, beat_frames_f), 2.0 * _beat_sync(chroma, beat_frames_f),
                       3.0 * _beat_sync(inst, beat_frames_f)])
    feat_times = np.concatenate([[0.0], beat_times])
    energy_at = _beat_values(rms30, feat_times, OUT_RATE, 0.2)
    sections, nov_beats = find_sections(feats, feat_times, downbeats, duration, energy_at,
                                        beats_per_bar=beats_per_bar)
    novelty30 = np.interp(np.arange(n_out) / OUT_RATE, feat_times, nov_beats) if len(nov_beats) else np.zeros(n_out)
    phrases = phrase_lines(sections, downbeats)

    bar_sec = beats_per_bar * 60.0 / max(bpm, 1.0)
    builds, drops = find_builds_and_drops(low30, _smooth(rms30, 15), bright30, downbeats, bar_sec,
                                          section_starts=[x["start"] for x in sections])

    vocals = vocal_starts = vocal_ends = None
    if with_vocals and demucs_available():
        say("separating vocals (slow)")
        v = vocal_curve(path, n_out)
        if v is not None:
            vocals = v
            vocal_starts, vocal_ends = vocal_phrases(v)

    analysis = {
        "version": ANALYSIS_VERSION,
        "analysed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "rate": OUT_RATE,
        "methods": {"beats": beats_method, "downbeats": db_method,
                    "vocals": "demucs" if vocals is not None else None},
        "beats_per_bar": beats_per_bar,
        "downbeats": _r(downbeats, 3),
        "kick": _r(kick30), "snare": _r(snare30), "hat": _r(hat30),
        "harmony": _r(harmony30), "brightness": _r(bright30), "novelty": _r(novelty30),
        "sections": sections, "phrases": phrases, "builds": builds, "drops": drops,
        "vocals": _r(vocals) if vocals is not None else None,
        "vocal_starts": vocal_starts, "vocal_ends": vocal_ends,
    }
    out = {"analysis": analysis}
    if beats_method == "server":
        n_env = int(duration / ENERGY_ENVELOPE_SEC) + 1
        step = int(OUT_RATE * ENERGY_ENVELOPE_SEC)
        env = [{"time": round(i * ENERGY_ENVELOPE_SEC, 2),
                "energy": round(float(rms30[i * step:(i + 1) * step].mean()) if i * step < n_out else 0.0, 3)}
               for i in range(n_env)]
        out.update({
            "duration_sec": round(duration, 3),
            "bpm": round(float(bpm), 2),
            "beat_times": _r(beat_times, 3),
            "energy_envelope": env,
            "hires": {"rate": OUT_RATE, "rms": _r(rms30, 3), "onset": _r(to30(onset_full), 3)},
            "analysed_by": "server",
        })
    return out


# ---------------------------------------------------------------------------
# Queue + worker (the Media Library page adds jobs and starts the worker)
# ---------------------------------------------------------------------------

def _paths():
    # In a subfolder: the pages list tracks with AUDIO_DIR.glob("*.json"), which
    # would otherwise pick these up as tracks.
    from config import AUDIO_DIR
    d = AUDIO_DIR / ".analysis"
    d.mkdir(parents=True, exist_ok=True)
    return {"queue": d / "queue.json", "status": d / "status.json", "lock": d / "worker.pid",
            "pause": d / "paused"}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def needs_extra(track: dict) -> bool:
    return int((track.get("analysis") or {}).get("version", 0)) < ANALYSIS_VERSION


def read_status() -> dict:
    return _read_json(_paths()["status"], {})


def worker_running() -> bool:
    try:
        pid = int(_paths()["lock"].read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def enqueue(jobs: list) -> int:
    """jobs: [{"track_id", "kind": "new"|"extra", "drive_rel": "Audio-Library/x.m4a"}].
    Skips tracks already queued or running. Returns how many were added."""
    p = _paths()
    queue = _read_json(p["queue"], [])
    status = _read_json(p["status"], {})
    have = {j["track_id"] for j in queue}
    added = 0
    for j in jobs:
        if j["track_id"] in have:
            continue
        queue.append(j)
        status[j["track_id"]] = {"state": "queued", "kind": j["kind"], "msg": "waiting", "updated": time.time()}
        added += 1
    _write_json(p["queue"], queue)
    _write_json(p["status"], status)
    return added


def is_paused() -> bool:
    return _paths()["pause"].exists()


def pause() -> None:
    """Finish the track in progress, then stop; the rest stay queued."""
    _paths()["pause"].write_text("1")


def resume() -> str:
    _paths()["pause"].unlink(missing_ok=True)
    return start_worker()


def stop_now(force: bool = False) -> None:
    """Stop at once: the track in progress goes back to the front of the
    queue (it restarts from scratch on resume) and nothing more runs until
    Resume. force=True kills the worker outright — for when it's stuck in a
    long step (e.g. vocal separation) and doesn't respond to a normal stop."""
    import signal as _signal
    p = _paths()
    pause()
    try:
        pid = int(p["lock"].read_text().strip())
        os.kill(pid, _signal.SIGKILL if force else _signal.SIGTERM)
    except Exception:
        pass
    if force:
        status = _read_json(p["status"], {})
        for v in status.values():
            if v.get("state") == "running":
                v.update(state="queued", msg="stopped — will restart on resume", updated=time.time())
        _write_json(p["status"], status)
        p["lock"].unlink(missing_ok=True)


def cancel_queued(track_ids=None) -> int:
    """Remove queued (not running) jobs — all of them, or just track_ids."""
    p = _paths()
    status = _read_json(p["status"], {})
    running = {k for k, v in status.items() if v.get("state") == "running"}
    queue = _read_json(p["queue"], [])
    drop = {j["track_id"] for j in queue
            if j["track_id"] not in running and (track_ids is None or j["track_id"] in track_ids)}
    _write_json(p["queue"], [j for j in queue if j["track_id"] not in drop])
    _write_json(p["status"], {k: v for k, v in status.items() if k not in drop})
    return len(drop)


def clear_finished_status() -> None:
    p = _paths()
    status = _read_json(p["status"], {})
    _write_json(p["status"], {k: v for k, v in status.items() if v.get("state") in ("queued", "running")})


def start_worker() -> str:
    """Start the background worker if it isn't running. Prefers a separate
    systemd unit (survives app restarts); falls back to a detached process."""
    if worker_running():
        return "already running"
    here = Path(__file__).resolve().parent
    cmd = [sys.executable, str(here / "audio_analysis.py"), "--worker"]
    if shutil.which("systemd-run") and os.geteuid() == 0:
        unit = f"mediaplanner-audio-{int(time.time())}"
        r = subprocess.run(["systemd-run", f"--unit={unit}", "--collect", "--nice=10",
                            f"--working-directory={here}", *cmd], capture_output=True, text=True)
        if r.returncode == 0:
            return "started"
    subprocess.Popen(cmd, cwd=str(here), start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "started"


def _local_audio_path(source_path: str) -> Path:
    from render_preview import colab_to_local
    return Path(colab_to_local(source_path))


def _process(job: dict, say) -> None:
    from config import AUDIO_DIR
    tid = job["track_id"]
    json_path = AUDIO_DIR / f"{tid}.json"
    existing = _read_json(json_path, None) if job["kind"] == "extra" else None
    source_path = (existing or {}).get("source_path") or f"/content/drive/MyDrive/{job['drive_rel']}"
    local = _local_audio_path(source_path)
    if not local.exists():
        say("downloading from Google Drive")
        from drive_sync import download_source_video
        rel = source_path.split("MyDrive/", 1)[1] if "MyDrive/" in source_path else job.get("drive_rel", "")
        err = download_source_video(rel, local)
        if err:
            raise RuntimeError(f"download failed: {err}")
    result = analyse_file(local, existing=existing, progress=say)
    if existing is not None:
        track = _read_json(json_path, existing)   # re-read: never clobber a newer edit
        track.update(result)
    else:
        track = {"track_id": tid, "source_path": source_path, **result}
    _write_json(json_path, track)
    try:
        from library_common import mark_pending
        mark_pending(f"audio:{tid}")
    except Exception:
        pass  # Drive upload bookkeeping only


def run_worker() -> None:
    p = _paths()
    try:
        os.nice(10)
    except Exception:
        pass
    if worker_running():
        return
    p["lock"].write_text(str(os.getpid()))

    class _Stopped(BaseException):
        pass

    def _on_term(_sig, _frame):
        raise _Stopped()

    import signal as _signal
    _signal.signal(_signal.SIGTERM, _on_term)
    try:
        while True:
            queue = _read_json(p["queue"], [])
            if not queue or p["pause"].exists():
                break
            job = queue[0]
            tid = job["track_id"]

            def say(msg, _tid=tid, _kind=job["kind"]):
                st = _read_json(p["status"], {})
                st[_tid] = {"state": "running", "kind": _kind, "msg": msg, "updated": time.time()}
                _write_json(p["status"], st)

            say("starting")
            try:
                _process(job, say)
                state, msg = "done", "finished"
            except _Stopped:
                st = _read_json(p["status"], {})
                st[tid] = {"state": "queued", "kind": job["kind"], "msg": "stopped — will restart on resume",
                           "updated": time.time()}
                _write_json(p["status"], st)
                break
            except Exception as e:
                traceback.print_exc()
                state, msg = "error", str(e)[-300:]
            st = _read_json(p["status"], {})
            st[tid] = {"state": state, "kind": job["kind"], "msg": msg, "updated": time.time()}
            _write_json(p["status"], st)
            queue = [j for j in _read_json(p["queue"], []) if j["track_id"] != tid]
            _write_json(p["queue"], queue)
    finally:
        try:
            p["lock"].unlink()
        except Exception:
            pass


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="process the analysis queue")
    ap.add_argument("--file", help="analyse one audio file and print a summary")
    ap.add_argument("--no-vocals", action="store_true")
    args = ap.parse_args()
    if args.worker:
        run_worker()
    elif args.file:
        t0 = time.time()
        r = analyse_file(args.file, with_vocals=not args.no_vocals, progress=lambda m: print("…", m))
        a = r["analysis"]
        print(f"{time.time() - t0:.1f}s  bpm={r.get('bpm')}  downbeats={len(a['downbeats'])} ({a['methods']['downbeats']})"
              f"  sections={[(s['start'], s['label']) for s in a['sections']]}  drops={a['drops']}"
              f"  builds={a['builds']}  vocals={a['methods']['vocals']}")
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
