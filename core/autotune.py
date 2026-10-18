"""🎯 Auto-detect: settings for a track and cutting method at a cutting pace
(Calm / Balanced / Energetic), plus split screen calibrated on the cuts.

Pure functions (no Streamlit). Moved unchanged from the planner page except
that the current dial values come in through `lookup` instead of being read
from session state."""

from __future__ import annotations

import numpy as np

import cut_scoring as CS
from core.segmentation import (SEG_ADAPTIVE, SEG_DEFAULTS, SEG_METHOD_KEYS, SEG_RHYTHM, SEG_SCORED,
                               SEG_SMART, apply_segment_intensity, build_segments, compute_bar_times)


def _weighted_percentile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights, dtype=float)[order]
    cum = np.cumsum(w) / max(w.sum(), 1e-9)
    return float(v[min(len(v) - 1, int(np.searchsorted(cum, q)))])


# Cutting pace presets. pace = average bars per cut; follow = how much faster
# loud passages cut than quiet ones; fast_share = share of bars the Rhythm
# Engine cuts every bar.
AUDIO_STYLES = {
    "Calm":      {"pace": 4.0, "follow": 0.3, "fast_share": 0.15, "cooldown": 8, "lock": 4},
    "Balanced":  {"pace": 2.0, "follow": 0.5, "fast_share": 0.30, "cooldown": 4, "lock": 4},
    "Energetic": {"pace": 1.0, "follow": 0.7, "fast_share": 0.50, "cooldown": 2, "lock": 2},
}


AUDIO_STYLE_CUSTOM = "Custom"


def _method_cfg(method: str, settings: dict, bpb: int, lookup=None) -> dict:
    """A segmentation_settings()-style dict from explicit values (falling back to
    the current ones — lookup(key), default SEG_DEFAULTS), so auto-detect can try
    settings before applying them."""
    lookup = lookup or SEG_DEFAULTS.get
    cfg = {"method": method, "beats_per_bar": int(bpb)}
    for k in SEG_METHOD_KEYS[method]:
        cfg[k] = settings.get(k, lookup(k))
    return cfg


def _tune_adaptive(track: dict, style: dict, bar: float, beat: float, why: list) -> dict:
    s = {}
    pace = style["pace"]
    duration = float(track["duration_sec"])
    s["change_window_secs"] = round(min(4.0, max(1.0, bar)), 1)
    s["min_segment_sec"] = round(min(8.0, max(0.5, 0.9 * beat * 2, 0.9 * 0.5 * pace * bar)), 2)
    s["max_segment_sec"] = round(min(30.0, max(2 * s["min_segment_sec"], 1.1 * 2 * pace * bar)), 1)
    s["snap_secs"] = round(min(0.5, max(0.1, beat / 2)), 2)
    s["react_to_hits"] = True
    why.append(f"Min segment {s['min_segment_sec']}s, max {s['max_segment_sec']}s "
               f"(≈{0.5 * pace:g}–{2 * pace:g} bars); energy window 1 bar; snap ½ beat")
    hires = track.get("hires")
    if not hires:
        why.append("⚠️ No detailed (30 Hz) signals for this track — thresholds left as they were.")
        return s
    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    onset = np.asarray(hires["onset"], dtype=float)
    n = min(len(rms), len(onset))
    rms, onset = rms[:n], onset[:n]
    min_len = max(1, round(rate * s["min_segment_sec"]))
    # Energy changes: about one structural change per 8 bars, never below 3× the ripple.
    w = max(1, round(rate * s["change_window_secs"]))
    c = np.concatenate([[0.0], np.cumsum(rms)])
    idx = np.arange(n)
    b_lo, a_hi = np.maximum(0, idx - w), np.minimum(n, idx + w)
    change = np.abs((c[a_hi] - c[idx]) / np.maximum(1, a_hi - idx) - (c[idx] - c[b_lo]) / np.maximum(1, idx - b_lo))
    cp = np.flatnonzero((change[1:-1] >= change[:-2]) & (change[1:-1] > change[2:])) + 1
    target_changes = max(2, int(duration / (8 * bar)))
    peak_vals = np.sort(change[cp])[::-1] if len(cp) else np.array([0.12])
    thr = peak_vals[min(len(peak_vals) - 1, target_changes - 1)]
    s["change_threshold"] = round(float(np.clip(max(thr, 3.0 * float(np.median(change))), 0.02, 0.5)), 2)
    # Big hits: fill up to the pace's overall cut rate with the strongest hits.
    op = np.flatnonzero((onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:])) + 1
    accepted = []
    for pos in sorted(op, key=lambda i: -onset[i]):
        if pos >= min_len and n - pos >= min_len and all(abs(pos - x) >= min_len for x in accepted):
            accepted.append(pos)
    target_hits = max(1, int(duration / (pace * bar)) - target_changes)
    if accepted:
        hit_thr = onset[accepted[min(len(accepted) - 1, target_hits - 1)]]
        typical = float(np.percentile(onset[op], 75)) if len(op) else 0.5
        s["hit_threshold"] = round(float(np.clip(max(hit_thr, typical), 0.5, 0.99)), 2)
    why.append(f"Energy-change threshold {s['change_threshold']} (≈1 structural cut per 8 bars); "
               f"big-hit threshold {s.get('hit_threshold', '—')} (≈1 cut per {pace:g} bars overall)")
    return s


def _tune_rhythm(track: dict, style: dict, bpb: int, bar: float, beat: float, why: list) -> dict:
    # Music moves in 2 / 4 / 8-bar phrases — CINEMATIC chunks of 2× the pace stay on them.
    s = {"cinematic_bars": int(min(8, max(1, 2 * style["pace"]))),
         "phrase_lock_bars": int(style["lock"]), "fast_max_bars": 8,
         "fast_cooldown_bars": int(style["cooldown"]),
         "override_clear_secs": round(min(4.0, max(0.2, 2 * beat)), 1)}
    bars = compute_bar_times(track, bpb)
    hires = track.get("hires")
    if len(bars) >= 4 and hires:
        rate = float(hires["rate"])
        rms = np.asarray(hires["rms"], dtype=float)
        frames = [round(t * rate) for t in bars] + [len(rms)]
        e = np.array([float(rms[frames[i]:frames[i + 1]].mean()) if frames[i + 1] > frames[i] else 0.0
                      for i in range(len(bars))])
        norm = e / (e.max() or 1.0)
        smooth = np.convolve(norm, np.ones(2) / 2, mode="same")
        s["fast_energy_thresh"] = round(float(np.clip(np.quantile(smooth, 1 - style["fast_share"]), 0.1, 1.0)), 2)
        deltas = np.abs(np.diff(rms, prepend=rms[0]))
        per_bar = np.sort([float(deltas[frames[i]:frames[i + 1]].max()) for i in range(len(bars))
                           if frames[i + 1] > frames[i]])[::-1]
        if len(per_bar):
            k = max(1, len(bars) // 8)          # about one dynamic cut per 8 bars
            s["override_delta_thresh"] = round(float(np.clip(per_bar[min(k, len(per_bar) - 1)], 0.02, 1.0)), 2)
        s["dynamic_priority_override"] = style is AUDIO_STYLES["Energetic"]
        why.append(f"FAST (a cut every bar) on the loudest {style['fast_share']:.0%} of bars → threshold "
                   f"{s['fast_energy_thresh']}; CINEMATIC segments {s['cinematic_bars']} bars; style held "
                   f"{s['phrase_lock_bars']} bars; FAST pause {s['fast_cooldown_bars']} bars after a run")
        if s["dynamic_priority_override"]:
            why.append(f"Dynamic override on (≈1 extra cut per 8 bars at sudden jumps, sensitivity "
                       f"{s.get('override_delta_thresh')})")
    else:
        why.append("⚠️ Not enough beat / energy data for bar styles — only the bar counts were set.")
    return s


def _tune_scored(track: dict, style: dict, bar: float, beat: float, why: list) -> dict:
    pace = style["pace"]
    s = {"sc_bars_per_cut": pace, "sc_follow_energy": style["follow"]}
    # 10% under / over whole bars, so cuts a bar (or phrase) apart are always allowed
    # despite tiny tempo wobble.
    s["sc_min_sec"] = round(min(8.0, max(0.3, 0.9 * beat, 0.9 * 0.5 * pace * bar)), 2)
    s["sc_max_sec"] = round(min(30.0, max(2 * s["sc_min_sec"], 1.1 * 2.5 * pace * bar)), 1)
    a = track.get("analysis") or {}
    why.append(f"≈1 cut per {pace:g} bars; segments {s['sc_min_sec']}–{s['sc_max_sec']}s; "
               f"follow the energy {style['follow']:g}")
    if not a:
        why.append("Only the basic analysis — cuts are scored from hits and beats. Add the extra analysis "
                   "in Media Library → Audio for drums, chords, sections, drops and vocals.")
    return s


def auto_tune_parameters(track: dict, min_clips: int, max_clips: int, method: str = None,
                         style_name: str = "Balanced", beats_per_bar: int = None, lookup=None) -> tuple:
    """Settings for THIS track and the chosen segmentation method at a cutting
    pace (Calm / Balanced / Energetic), plus split screen calibrated on the
    resulting cuts. Returns (settings dict, list of explanation lines).

    method / beats_per_bar default to Adaptive / 4; lookup(key) gives the current
    value of any dial not being tuned (default SEG_DEFAULTS)."""
    method = method or SEG_ADAPTIVE
    if method not in SEG_METHOD_KEYS:
        method = SEG_ADAPTIVE
    bpb = int(beats_per_bar or 4)
    style = AUDIO_STYLES.get(style_name) or AUDIO_STYLES["Balanced"]
    bar = CS.bar_sec(track, bpb)
    beat = bar / bpb
    _, grid_bpm, fix = CS.beat_grid(track)
    why = [f"Tempo {grid_bpm:.0f} BPM → 1 bar = {bar:.2f}s"
           + {"doubled": " (the analysis found every other beat — corrected)",
              "halved": " (the analysis found twice the beats — corrected)"}.get(fix, "")]
    if method == SEG_SMART:
        s = {}   # Smart tunes itself from the track; its Pace / Contrast are left as you set them
        why.append("Smart sets its own pace from the track — only split screen was calibrated here")
    elif method == SEG_RHYTHM:
        s = _tune_rhythm(track, style, bpb, bar, beat, why)
    elif method == SEG_SCORED:
        s = _tune_scored(track, style, bar, beat, why)
    else:
        s = _tune_adaptive(track, style, bar, beat, why)

    # Split screen: shape intensity on the cuts this method will actually make.
    segs = build_segments(track, _method_cfg(method, s, bpb, lookup))
    apply_segment_intensity(segs, track, 1.0)
    base = np.array([sg["intensity"] for sg in segs])
    lengths = np.array([sg["end"] - sg["start"] for sg in segs])
    med = _weighted_percentile(base, lengths, 0.5)
    contrast = np.log(0.5) / np.log(med) if 0.05 < med < 0.95 else 1.0
    s["density_contrast"] = round(float(np.clip(contrast, 0.5, 2.5)), 2)
    shaped = base ** s["density_contrast"]
    lo = _weighted_percentile(shaped, lengths, 0.2)
    hi = _weighted_percentile(shaped, lengths, 0.9)
    if hi - lo < 0.2:
        lo, hi = max(0.0, (lo + hi) / 2 - 0.1), min(1.0, (lo + hi) / 2 + 0.1)
    s["ramp_range"] = (round(lo * 20) / 20, round(max(hi, lo + 0.1) * 20) / 20)
    s["vary_count"] = False
    why.append(f"Split screen: quietest 20% of the track gets {min_clips} clip(s), loudest 10% gets {max_clips}")
    why.append(f"Result: {len(segs)} segments, about {np.mean(lengths) / bar:.1f} bars each")
    return s, why
