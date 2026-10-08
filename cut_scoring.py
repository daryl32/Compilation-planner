"""
"Scored cuts" segmentation: every candidate moment gets a cut score made of
weighted musical reasons (big hit, kick, snare, chord change, bar line, phrase
line, section change, drop, vocal phrase start/end), then the cuts are chosen
TOGETHER — the best-scoring set whose segments all fit between the minimum
and maximum length, with the overall cut rate set by "bars per cut" — rather
than greedily one at a time. That is what stops clusters of closely spaced
cuts: two good moments half a beat apart can't both be taken, and the better
one wins.

Uses the server's extra analysis (track["analysis"], see audio_analysis.py)
when present; with only the Colab analysis it still works from hits and beats.
No Streamlit in here.
"""

import numpy as np

# Weight keys (settings) → short reason names shown on the chart
REASONS = {
    "sc_w_hit": "hit", "sc_w_kick": "kick", "sc_w_snare": "snare", "sc_w_harmony": "harmony",
    "sc_w_bar": "bar", "sc_w_phrase": "phrase", "sc_w_section": "section", "sc_w_drop": "drop",
    "sc_w_vocal": "vocal",
}

DEFAULTS = {
    "sc_min_sec": 1.5, "sc_max_sec": 8.0, "sc_bars_per_cut": 2.0, "sc_follow_energy": 0.5,
    "sc_w_hit": 1.0, "sc_w_kick": 1.0, "sc_w_snare": 0.7, "sc_w_harmony": 0.7,
    "sc_w_bar": 1.0, "sc_w_phrase": 1.5, "sc_w_section": 3.0, "sc_w_drop": 3.0,
    "sc_w_vocal": 1.0, "sc_w_vocal_mid": 1.0,
}


def _curve_at(curve, rate: float, times: np.ndarray, radius: float = 0.05) -> np.ndarray:
    """Max of a curve within ±radius of each time (0 where there's no curve)."""
    if curve is None or len(curve) == 0:
        return np.zeros(len(times))
    c = np.asarray(curve, dtype=float)
    r = max(0, int(round(radius * rate)))
    idx = np.clip(np.round(times * rate).astype(int), 0, len(c) - 1)
    if r == 0:
        return c[idx]
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(c, r, mode="edge")
    win_max = sliding_window_view(padded, 2 * r + 1).max(axis=1)
    return win_max[idx]


def _near(times: np.ndarray, marks, tol: float) -> np.ndarray:
    """1.0 where a time is within tol of any mark, else 0."""
    if not marks:
        return np.zeros(len(times))
    m = np.sort(np.asarray(marks, dtype=float))
    j = np.clip(np.searchsorted(m, times), 1, len(m) - 1) if len(m) > 1 else np.zeros(len(times), int)
    d = np.minimum(np.abs(m[j] - times), np.abs(m[np.maximum(j - 1, 0)] - times))
    return (d <= tol).astype(float)


def bar_lines(track: dict, beats_per_bar: int) -> list:
    """Real downbeats from the server analysis when available, else every
    beats_per_bar-th beat from the first (the old assumption)."""
    beats = list(track.get("beat_times") or [])
    a = track.get("analysis") or {}
    db = a.get("downbeats") or []
    if db and int(a.get("beats_per_bar", 4)) == int(beats_per_bar):
        return [float(t) for t in db]
    if len(beats) < beats_per_bar:
        return []
    phase = 0
    if db and beats:
        first = int(np.argmin(np.abs(np.asarray(beats) - db[0])))
        phase = first % beats_per_bar
    return beats[phase::beats_per_bar]


def candidate_scores(track: dict, cfg: dict) -> tuple:
    """(times, scores, reasons): every beat plus strong off-beat hits, each
    with its cut score and the biggest reason behind it."""
    duration = float(track["duration_sec"])
    beats = np.asarray(track.get("beat_times") or [], dtype=float)
    hires = track.get("hires") or {}
    rate = float(hires.get("rate", 30))
    onset = np.asarray(hires.get("onset") or [], dtype=float)
    a = track.get("analysis") or {}
    arate = float(a.get("rate", 30))
    bpb = int(cfg.get("beats_per_bar", 4))

    times = list(beats)
    if len(onset):   # strong hits that fall between beats
        pk = np.flatnonzero((onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:]) & (onset[1:-1] >= 0.6)) + 1
        for p in pk:
            t = p / rate
            if not len(beats) or np.min(np.abs(beats - t)) > 0.12:
                times.append(t)
    for extra in (a.get("drops") or []):
        times.append(float(extra))
    for sec in (a.get("sections") or [])[1:]:
        times.append(float(sec["start"]))
    times = np.array(sorted(t for t in set(round(x, 3) for x in times) if 0.05 < t < duration - 0.05))
    if not len(times):
        return times, np.zeros(0), []

    bars = bar_lines(track, bpb)
    phrase_level = np.zeros(len(times))
    for ph in a.get("phrases") or []:
        lvl = {16: 1.0, 8: 0.66, 4: 0.33}.get(int(ph.get("level", 4)), 0.33)
        hit = np.abs(times - float(ph["time"])) <= 0.06
        phrase_level = np.maximum(phrase_level, hit * lvl)
    section_marks = [s["start"] for s in (a.get("sections") or [])[1:]]

    parts = {
        "hit": _curve_at(onset, rate, times),
        "kick": _curve_at(a.get("kick"), arate, times),
        "snare": _curve_at(a.get("snare"), arate, times),
        "harmony": _curve_at(a.get("harmony"), arate, times, 0.1),
        "bar": _near(times, bars, 0.06),
        "phrase": phrase_level,
        "section": _near(times, section_marks, 0.06),
        "drop": _near(times, a.get("drops") or [], 0.06),
        "vocal": np.maximum(_near(times, a.get("vocal_starts") or [], 0.2),
                            _near(times, a.get("vocal_ends") or [], 0.2)),
    }
    weighted = {name: float(cfg.get(key, DEFAULTS[key])) * parts[name] for key, name in REASONS.items()}
    score = np.sum(list(weighted.values()), axis=0)

    # Cutting in the middle of a sung line (vocals clearly on, not near its start/end)
    vocals = a.get("vocals")
    if vocals is not None and float(cfg.get("sc_w_vocal_mid", 0)) > 0:
        v = _curve_at(vocals, arate, times, 0.15)
        mid = np.clip((v - 0.25) / 0.5, 0, 1) * (1 - parts["vocal"])
        score = score - float(cfg["sc_w_vocal_mid"]) * mid

    names = list(weighted)
    stack = np.vstack([weighted[n] for n in names])
    reasons = [names[int(i)] if stack[int(i), k] > 0 else "beat" for k, i in enumerate(np.argmax(stack, axis=0))]
    return times, score, reasons


def _intensity_at(track: dict, times: np.ndarray) -> np.ndarray:
    hires = track.get("hires") or {}
    rms = np.asarray(hires.get("rms") or [], dtype=float)
    if not len(rms):
        return np.full(len(times), 0.5)
    rate = float(hires.get("rate", 30))
    k = max(1, int(rate * 1.5))
    sm = np.convolve(rms, np.ones(k) / k, mode="same")
    lo, hi = np.percentile(sm, 5), np.percentile(sm, 95)
    sm = np.clip((sm - lo) / (hi - lo), 0, 1) if hi - lo > 1e-9 else np.full_like(sm, 0.5)
    return sm[np.clip(np.round(times * rate).astype(int), 0, len(sm) - 1)]


def choose_cuts(times: np.ndarray, gains: np.ndarray, duration: float, min_len: float,
                max_len: float) -> list:
    """Best total gain over a chain start → cuts → end where every gap is
    between min_len and max_len (max_len 0 = no limit). A gap longer than
    max_len is allowed only when nothing else fits, at a heavy penalty."""
    T = np.concatenate([[0.0], times, [duration]])
    G = np.concatenate([[0.0], gains, [0.0]])
    n = len(T)
    best = np.full(n, -np.inf)
    back = np.full(n, -1, dtype=int)
    best[0] = 0.0
    big = max_len if max_len and max_len > 0 else duration
    for j in range(1, n):
        lo = np.searchsorted(T, T[j] - big, side="left")
        hi = np.searchsorted(T, T[j] - min_len, side="right") - 1
        if j == n - 1:
            hi = min(hi, j - 1)
        if hi >= lo and hi >= 0:
            seg = best[lo:hi + 1]
            k = int(np.argmax(seg))
            if np.isfinite(seg[k]):
                best[j], back[j] = seg[k] + G[j], lo + k
                continue
        # nothing fits: allow an over-long segment from the best earlier point
        hi2 = np.searchsorted(T, T[j] - min_len, side="right") - 1
        if hi2 >= 0:
            seg = best[:hi2 + 1] - 50.0
            k = int(np.argmax(seg))
            if np.isfinite(seg[k]):
                best[j], back[j] = seg[k] + G[j], k
    cuts, j = [], back[n - 1]
    while j > 0:
        cuts.append(j - 1)
        j = back[j]
    return sorted(cuts)


def build_scored_segments(track: dict, cfg: dict) -> list:
    """Segments in the usual {"start","end","energy","cut","cut_strength"} schema."""
    duration = float(track["duration_sec"])
    hires = track.get("hires") or {}
    times, score, reasons = candidate_scores(track, cfg)
    min_len = max(0.2, float(cfg.get("sc_min_sec", DEFAULTS["sc_min_sec"])))
    max_len = float(cfg.get("sc_max_sec", DEFAULTS["sc_max_sec"]))
    if max_len and max_len < 2 * min_len:
        max_len = 2 * min_len
    if not len(times):
        return [{"start": 0.0, "end": duration, "energy": 0.0, "cut": "start", "cut_strength": 0.0}]

    bpm = float(track.get("bpm") or 120.0)
    bar_sec = int(cfg.get("beats_per_bar", 4)) * 60.0 / max(bpm, 1.0)
    target = max(1, int(round(duration / max(0.25, float(cfg.get("sc_bars_per_cut", 2.0))) / bar_sec)))
    follow = float(cfg.get("sc_follow_energy", DEFAULTS["sc_follow_energy"]))
    # Busier passages make cutting cheaper, quiet ones dearer (follow = 0: flat).
    cost_shape = 2.0 ** (-follow * (2.0 * _intensity_at(track, times) - 1.0))

    # Find the per-cut cost giving about `target` cuts (fewer cuts as the cost rises).
    lo_c, hi_c = -20.0, 40.0
    cuts = []
    for _ in range(28):
        mid = (lo_c + hi_c) / 2
        cuts = choose_cuts(times, score - mid * cost_shape, duration, min_len, max_len)
        if len(cuts) > target:
            lo_c = mid
        else:
            hi_c = mid
    cuts = choose_cuts(times, score - hi_c * cost_shape, duration, min_len, max_len)

    rate = float(hires.get("rate", 30))
    rms = np.asarray(hires.get("rms") or [], dtype=float)
    bounds = [0.0] + [float(times[i]) for i in cuts] + [duration]
    kinds = ["start"] + [reasons[i] for i in cuts]
    strengths = [0.0] + [float(score[i]) for i in cuts]
    segs = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        a, b = int(s * rate), int(e * rate)
        segs.append({
            "start": round(s, 3), "end": round(e, 3),
            "energy": round(float(rms[a:b].mean()), 3) if b > a and len(rms) else 0.0,
            "cut": kinds[k] if k == 0 else f"sc_{kinds[k]}",
            "cut_strength": round(strengths[k], 3),
        })
    return segs
