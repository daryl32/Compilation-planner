"""Cutting a track into blocks — the Adaptive, Rhythm Engine, Scored and Smart
methods (the last two live in cut_scoring.py / smart_cuts.py), block intensity
for split screen, and clip counts per block.

Pure functions: no Streamlit, no session state. Moved here unchanged from
pages/1_Compilation_Planner.py (Oct 2026 refactor); tests/test_golden.py
checks they still give exactly the results they gave there."""

from __future__ import annotations

import numpy as np

import cut_scoring as CS
import smart_cuts as SM

# The default value of every cutting dial, for code running without the
# planner page (the page keeps its own DIAL_DEFAULTS; a test checks they agree).
SEG_DEFAULTS = {
    "segmentation_method": "Smart (self-tuning)",
    "beats_per_bar": 4,
    "change_window_secs": 2.0,
    "change_threshold": 0.12,
    "min_segment_sec": 2.0,
    "max_segment_sec": 8.0,
    "react_to_hits": True,
    "hit_threshold": 0.9,
    "snap_secs": 0.3,
    "fast_energy_thresh": 0.75,
    "fast_max_bars": 8,
    "fast_cooldown_bars": 4,
    "phrase_lock_bars": 4,
    "cinematic_bars": 4,
    "dynamic_priority_override": False,
    "override_delta_thresh": 0.15,
    "override_clear_secs": 1.0,
    "on_beat": True,
    **CS.DEFAULTS,
    **SM.DEFAULTS,
}


SEG_ADAPTIVE = "Adaptive (energy-change + hits)"


SEG_RHYTHM = "Rhythm Engine (bar cutting styles)"


SEG_SCORED = "Scored cuts (all signals)"


SEG_SMART = "Smart (self-tuning)"


# Which settings each segmentation method actually uses (beats_per_bar: all).
SEG_METHOD_KEYS = {
    SEG_ADAPTIVE: ["change_window_secs", "change_threshold", "min_segment_sec", "max_segment_sec",
                   "react_to_hits", "hit_threshold", "snap_secs", "on_beat"],
    SEG_RHYTHM: ["fast_energy_thresh", "fast_max_bars", "fast_cooldown_bars", "phrase_lock_bars",
                 "cinematic_bars", "dynamic_priority_override", "override_delta_thresh",
                 "override_clear_secs", "on_beat"],
    SEG_SCORED: list(CS.DEFAULTS.keys()) + ["on_beat"],
    SEG_SMART: list(SM.DEFAULTS.keys()),
}


def build_segments(track: dict, cfg: dict) -> list:
    """Segments for this track with the settings from segmentation_settings()."""
    bpb = int(cfg.get("beats_per_bar", 4))
    if cfg["method"] == SEG_SMART:
        return SM.build_smart_segments(track, cfg)
    if cfg["method"] == SEG_SCORED:
        return CS.build_scored_segments(track, cfg)
    if cfg["method"] == SEG_RHYTHM:
        return build_rhythm_engine_segments(
            track, bpb, float(cfg["fast_energy_thresh"]), int(cfg["fast_max_bars"]),
            int(cfg["fast_cooldown_bars"]), int(cfg["phrase_lock_bars"]), int(cfg["cinematic_bars"]),
            bool(cfg["dynamic_priority_override"]), float(cfg["override_delta_thresh"]),
            float(cfg["override_clear_secs"]), on_beat=bool(cfg.get("on_beat", True)))
    return build_track_segments(
        track, float(cfg["change_window_secs"]), float(cfg["change_threshold"]),
        float(cfg["min_segment_sec"]), bool(cfg["react_to_hits"]), float(cfg["hit_threshold"]),
        float(cfg["snap_secs"]), float(cfg["max_segment_sec"]), bpb, on_beat=bool(cfg.get("on_beat", True)))


# ---------------------------------------------------------------------------
# 🥁 Land every cut on the beat (Adaptive, Rhythm Engine; Scored picks beats itself)
# ---------------------------------------------------------------------------

# Cuts that mark a change in the music (rather than one hit) go to a bar line when one is near.
STRUCTURAL_CUTS = {"change", "fill", "cinematic", "fast"}
# Which cut wins when two land on the same beat.
_CUT_RANK = {"change": 3, "cinematic": 3, "fast": 3, "override": 2, "hit": 2, "fill": 1}


def land_on_beats(track: dict, segments: list, beats_per_bar: int = 4, min_gap: float = 0.0) -> list:
    """The same segments with every cut moved onto the track's beat grid (cut_scoring.beat_grid,
    the corrected one Smart and Scored use): a structural cut (energy change, fill, bar style) onto
    a bar line when one is within a beat, otherwise the nearest beat; a hit or jump cut onto the
    nearest beat. Two cuts that land on the same beat — or closer than min_gap — keep the more
    important one. Segments keep their kind; energy and strength are measured again."""
    hires = track.get("hires")
    if len(segments) < 2 or not hires:
        return segments
    beats = np.asarray(CS.beat_grid(track)[0], dtype=float)
    if len(beats) < 4:
        return segments
    beat = float(np.median(np.diff(beats)))
    bars = np.asarray(CS.bar_lines(track, beats_per_bar), dtype=float)
    duration = float(track["duration_sec"])
    gap = max(0.9 * beat, float(min_gap))
    moved = []
    for sg in segments[1:]:
        t = float(sg["start"])
        b = float(beats[int(np.argmin(np.abs(beats - t)))])
        if sg.get("cut") in STRUCTURAL_CUTS and len(bars):
            nb = float(bars[int(np.argmin(np.abs(bars - t)))])
            if abs(nb - t) <= 1.01 * beat:
                b = nb
        moved.append((b, sg))
    kept = []
    for b, sg in sorted(moved, key=lambda m: (-_CUT_RANK.get(m[1].get("cut"), 1),
                                              -float(m[1].get("cut_strength") or 0.0))):
        if b < gap or duration - b < 0.5 * gap:
            continue
        if all(abs(b - k) >= gap for k, _ in kept):
            kept.append((b, sg))
    kept.sort(key=lambda m: m[0])

    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    onset = np.asarray(hires["onset"], dtype=float)
    starts = [(0.0, segments[0])] + kept
    out = []
    for (s0, sg), nxt in zip(starts, starts[1:] + [(duration, None)]):
        e0 = nxt[0]
        a, z = int(round(s0 * rate)), int(round(e0 * rate))
        out.append({**sg, "start": round(s0, 3), "end": round(min(e0, duration), 3),
                    "energy": round(float(rms[a:z].mean()), 3) if z > a and len(rms) else sg.get("energy", 0.0),
                    "cut_strength": (round(float(onset[a]), 3) if 0 <= a < len(onset) and s0 > 0
                                     else sg.get("cut_strength", 0.0))})
    return out


def _moving_average(values, half_window: int):
    """Symmetric moving average with edge clamping (cumulative-sum version,
    fast enough for 30 Hz signals)."""
    values = np.asarray(values, dtype=float)
    if half_window <= 0:
        return values.copy()
    n = len(values)
    c = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(n, idx + half_window + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def compute_bar_times(track: dict, beats_per_bar: int = 4) -> list:
    """Bar start times for beats_per_bar beats per bar (4 = common time; 3 for
    a waltz, etc.). Uses the real downbeats when the server's extra analysis
    is present; otherwise every Nth beat counting from the first detected beat."""
    # Real downbeats from the server's extra analysis when the track has it
    # (see cut_scoring.bar_lines); otherwise every Nth beat from the first.
    return CS.bar_lines(track, beats_per_bar)


def _snap_to_bar_or_onset(i: int, bar_frames: np.ndarray, onset: np.ndarray, snap: int, n: int) -> int:
    """Snap position i onto the nearest bar line if one is close enough to be
    plausibly the SAME musical moment (within 1.5 bars, or 3x the onset-snap
    radius if bars are sparse/unavailable) — otherwise fall back to the
    original onset-peak snap, same as before bar-awareness existed."""
    if len(bar_frames) > 0:
        nearest = int(np.argmin(np.abs(bar_frames - i)))
        bar_pos = int(bar_frames[nearest])
        bar_spacing = float(np.median(np.diff(bar_frames))) if len(bar_frames) > 1 else float(n)
        if abs(bar_pos - i) <= max(bar_spacing * 1.5, snap * 3):
            return bar_pos
    lo, hi = max(1, i - snap), min(n - 1, i + snap + 1)
    return lo + int(np.argmax(onset[lo:hi])) if hi > lo else int(i)


RHYTHM_STATE_CINEMATIC = "CINEMATIC"


RHYTHM_STATE_FAST = "FAST"


def compute_cutting_states(
    track: dict,
    beats_per_bar: int = 4,
    fast_energy_thresh: float = 0.75,
    fast_max_bars: int = 8,
    fast_cooldown_bars: int = 4,
    phrase_lock_bars: int = 4,
) -> list:
    """Assigns a cutting style to every bar. Returns a list of state strings,
    one per bar (same length/order as compute_bar_times' output), or an
    empty list if there isn't enough beat/hires data to form bars.

    fast_max_bars: a FAST run is forced back to CINEMATIC after this many
    bars (0 = no limit). fast_cooldown_bars: after ANY FAST run ends, this
    many bars must pass before FAST may start again. phrase_lock_bars: a
    newly chosen style is held for this many bars (the FAST limit still
    applies inside the lock)."""
    bar_times = compute_bar_times(track, beats_per_bar)
    hires = track.get("hires")
    if len(bar_times) < 2 or not hires:
        return []

    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    bar_frames = [round(bt * rate) for bt in bar_times] + [len(rms)]

    bar_energies = np.array([
        float(rms[bar_frames[i]:bar_frames[i + 1]].mean()) if bar_frames[i + 1] > bar_frames[i] else 0.0
        for i in range(len(bar_times))
    ])
    max_e = bar_energies.max() if bar_energies.max() > 0 else 1.0
    norm = bar_energies / max_e
    local_smooth = np.convolve(norm, np.ones(2) / 2, mode="same")

    states = []
    current = RHYTHM_STATE_CINEMATIC
    fast_run = 0          # bars in the current FAST run
    fast_allowed_at = 0   # first bar index where FAST may start again
    lock = 0              # bars still held by phrase lock

    for i in range(len(bar_times)):
        if current == RHYTHM_STATE_FAST and fast_max_bars > 0 and fast_run >= fast_max_bars:
            current, lock = RHYTHM_STATE_CINEMATIC, 0          # forced out: max run reached
            fast_allowed_at = i + max(0, fast_cooldown_bars)
        elif lock > 0:
            lock -= 1
        else:
            want = (RHYTHM_STATE_FAST if local_smooth[i] > fast_energy_thresh and i >= fast_allowed_at
                    else RHYTHM_STATE_CINEMATIC)
            if want != current:
                if current == RHYTHM_STATE_FAST:                 # FAST ended on its own
                    fast_allowed_at = i + max(0, fast_cooldown_bars)
                current = want
                lock = max(0, phrase_lock_bars - 1)
        fast_run = fast_run + 1 if current == RHYTHM_STATE_FAST else 0
        states.append(current)

    return states


def find_dynamic_priority_cuts(track: dict, bar_times: list, override_delta_thresh: float) -> list:
    """Secondary pass, run after the state machine: scans the RAW,
    frame-by-frame change in RMS — not the bar-averaged, smoothed energy the
    state machine uses — for instantaneous jumps (a single dramatic hit or
    drop mid-bar). For each bar whose biggest jump exceeds
    override_delta_thresh, returns (time_sec, jump_size) at that exact frame."""
    hires = track.get("hires")
    if not hires or len(bar_times) < 2:
        return []
    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    raw_deltas = np.abs(np.diff(rms, prepend=rms[0]))

    bar_frames = [round(bt * rate) for bt in bar_times] + [len(rms)]
    out = []
    for i in range(len(bar_times)):
        s, e = bar_frames[i], bar_frames[i + 1]
        if e <= s:
            continue
        window = raw_deltas[s:e]
        peak_idx = int(np.argmax(window))
        if window[peak_idx] > override_delta_thresh:
            out.append(((s + peak_idx) / rate, float(window[peak_idx])))
    return out


MIN_OVERRIDE_GAP_SEC = 0.15  # never leave a sliver shorter than this next to a dynamic cut


def apply_dynamic_priority_overrides(raw_segments: list, overrides: list, clear_secs: float,
                                     duration: float) -> list:
    """Dynamic cuts take priority over the state machine's own cuts.

    overrides: [(time_sec, strength)]. Dynamic cuts closer than clear_secs to
    each other keep only the strongest; then every ordinary cut closer than
    clear_secs (before OR after) to a kept dynamic cut is removed, and the
    dynamic cut is added. The segment after a dynamic cut is tagged
    "override"; every other segment keeps its original kind."""
    if not overrides:
        return raw_segments
    gap = max(clear_secs, MIN_OVERRIDE_GAP_SEC)
    kept = []
    for t, strength in sorted(overrides, key=lambda o: -o[1]):
        if gap < t < duration - MIN_OVERRIDE_GAP_SEC and all(abs(t - k) >= gap for k in kept):
            kept.append(t)
    if not kept:
        return raw_segments
    bounds = [(s, kind) for s, _, kind in raw_segments
              if s <= 1e-6 or all(abs(s - k) >= gap for k in kept)]
    bounds += [(t, "override") for t in kept]
    bounds.sort()
    out = []
    for (s, kind), nxt in zip(bounds, bounds[1:] + [(duration, None)]):
        if nxt[0] > s:
            out.append((s, nxt[0], kind))
    return out


def build_rhythm_engine_segments(
    track: dict,
    beats_per_bar: int = 4,
    fast_energy_thresh: float = 0.75,
    fast_max_bars: int = 8,
    fast_cooldown_bars: int = 4,
    phrase_lock_bars: int = 4,
    cinematic_bars: int = 4,
    dynamic_priority_override: bool = False,
    override_delta_thresh: float = 0.15,
    override_clear_secs: float = 1.0,
    on_beat: bool = False,
) -> list:
    """Converts the per-bar cutting states into actual segments, in the same
    {"start","end","energy","cut","cut_strength"} schema build_track_segments
    produces — so everything downstream works with either method unchanged.
    CINEMATIC runs are grouped into cinematic_bars-bar chunks; FAST bars each
    become their own segment. Falls back to the whole track as one segment if
    there isn't enough data to form even one bar."""
    duration = float(track["duration_sec"])
    hires = track.get("hires")
    bar_times = compute_bar_times(track, beats_per_bar)
    states = compute_cutting_states(track, beats_per_bar, fast_energy_thresh, fast_max_bars,
                                    fast_cooldown_bars, phrase_lock_bars)

    if len(bar_times) < 2 or not states:
        energy = float(np.mean(hires["rms"])) if hires else 0.0
        return [{"start": 0.0, "end": duration, "energy": round(energy, 3), "cut": "start", "cut_strength": 0.0}]

    bar_bounds = list(bar_times) + [duration]
    raw_segments = []  # (start, end, cut_kind)
    # Lead-in before the first bar: its own segment if at least a bar long,
    # otherwise folded into the first bar.
    if bar_bounds[0] >= bar_bounds[1] - bar_bounds[0]:
        raw_segments.append((0.0, bar_bounds[0], "start"))
    else:
        bar_bounds[0] = 0.0
    i = 0
    while i < len(states):
        j = i
        while j < len(states) and states[j] == states[i]:
            j += 1
        if states[i] == RHYTHM_STATE_FAST:
            for k in range(i, j):
                raw_segments.append((bar_bounds[k], bar_bounds[k + 1], "fast"))
        else:
            k = i
            while k < j:
                chunk_end = min(k + max(1, cinematic_bars), j)
                raw_segments.append((bar_bounds[k], bar_bounds[chunk_end], "cinematic"))
                k = chunk_end
        i = j

    raw_segments.sort()
    # A sliver after the last whole bar (e.g. a fade-out) joins the segment before it.
    if len(raw_segments) > 1 and len(bar_times) > 1:
        half_bar = 0.5 * float(np.median(np.diff(bar_times)))
        s0, e0, k0 = raw_segments[-1]
        if e0 - s0 < half_bar:
            ps, _, pk = raw_segments[-2]
            raw_segments[-2:] = [(ps, e0, pk)]
    if dynamic_priority_override:
        overrides = find_dynamic_priority_cuts(track, bar_times, override_delta_thresh)
        raw_segments = apply_dynamic_priority_overrides(raw_segments, overrides, override_clear_secs, duration)

    rate = float(hires["rate"]) if hires else 0.0
    rms = np.asarray(hires["rms"], dtype=float) if hires else np.array([])
    onset = np.asarray(hires["onset"], dtype=float) if hires else np.array([])

    segments = []
    for start, end, kind in raw_segments:
        if end <= start:
            continue
        s_f, e_f = int(round(start * rate)), int(round(end * rate))
        energy = float(rms[s_f:e_f].mean()) if e_f > s_f and len(rms) else 0.0
        strength = float(onset[s_f]) if 0 <= s_f < len(onset) else 0.0
        segments.append({
            "start": round(start, 3),
            "end": round(min(end, duration), 3),
            "energy": round(energy, 3),
            "cut": kind,
            "cut_strength": round(strength, 3),
        })
    if on_beat:   # jump cuts land mid-bar; put them on the nearest beat
        segments = land_on_beats(track, segments, beats_per_bar)
    return segments


def build_track_segments(
    track: dict,
    change_window_secs: float,
    change_threshold: float,
    min_block_secs: float,
    react_to_hits: bool,
    hit_threshold: float,
    snap_secs: float,
    max_block_secs: float = 0.0,
    beats_per_bar: int = 4,
    on_beat: bool = False,
) -> list[dict]:
    """Tight segmentation on the 30 Hz signals from audio_pipeline.py.

    1. ENERGY CHANGES — a step detector: at every sample, compare the mean
       energy over the next `change_window_secs` with the mean over the
       previous `change_window_secs`. Peaks in that difference above
       `change_threshold` are genuine level changes (drops, build-ups,
       chorus in/out). Steady beat ripple averages out, so it can't slowly
       accumulate into false cuts.
    2. SNAP — each energy-change cut moves onto the strongest nearby onset
       (the add-on's "Clip Change" score) within ±snap_secs.
    3. BIG HITS — onset peaks at or above hit_threshold become extra cuts,
       landing exactly on the transient, NOT snapped to a bar line — a hit
       can legitimately fall mid-bar, and forcing it onto the bar grid would
       undo the point of reacting to it individually.

    Candidates are accepted strongest-first (energy changes rank above hits),
    skipping any closer than min_block_secs to an accepted cut, so the most
    important moments always win.

    4. MAX LENGTH — like the add-on's Build Signal Grid Plan (which chunks long
       blocks and snaps chunk boundaries to hits), any segment longer than
       max_block_secs is split at its strongest internal onset, repeatedly,
       so evenly-busy passages with no standout hit still get cut on a beat.
       0 = off.

    Tracks analysed before the 30 Hz signals existed fall back to the older
    coarse method (re-run audio_pipeline.py in Colab to upgrade them)."""
    hires = track.get("hires")
    if not hires:
        return _legacy_track_segments(track, change_window_secs, 0.5, 0.5, min_block_secs, react_to_hits)

    rate = float(hires["rate"])
    rms = np.asarray(hires["rms"], dtype=float)
    onset = np.asarray(hires["onset"], dtype=float)
    n = min(len(rms), len(onset))
    rms, onset = rms[:n], onset[:n]
    duration = float(track["duration_sec"])
    if n < 3:
        return [{"start": 0.0, "end": duration, "energy": float(rms.mean()) if n else 0.0,
                 "cut": "start", "cut_strength": 0.0}]

    w = max(1, round(rate * change_window_secs))
    min_len = max(1, round(rate * min_block_secs))
    snap = max(0, round(rate * snap_secs))

    # Bar snapping was retired (the "Snap structural cuts to bar lines" toggle);
    # with no bar frames, snapping and max-length splits use onsets only.
    bar_frames = np.array([])

    # 1. step detector: |mean(after) - mean(before)|
    c = np.concatenate([[0.0], np.cumsum(rms)])
    idx = np.arange(n)
    b_lo, a_hi = np.maximum(0, idx - w), np.minimum(n, idx + w)
    before = (c[idx] - c[b_lo]) / np.maximum(1, idx - b_lo)
    after = (c[a_hi] - c[idx]) / np.maximum(1, a_hi - idx)
    change = np.abs(after - before)
    change[:1] = 0.0
    change_peaks = np.flatnonzero((change[1:-1] >= change[:-2]) & (change[1:-1] > change[2:])) + 1

    candidates = []  # (score, index, kind)
    for i in change_peaks:
        if change[i] < change_threshold:
            continue
        # 2. snap onto the strongest nearby onset
        pos = _snap_to_bar_or_onset(i, bar_frames, onset, snap, n)
        candidates.append((1.0 + float(change[i]), int(pos), "change"))

    # 3. big hits
    if react_to_hits:
        peaks = np.flatnonzero((onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:])) + 1
        for pos in peaks:
            if onset[pos] >= hit_threshold:
                candidates.append((float(onset[pos]), int(pos), "hit"))

    accepted = []
    for score, pos, kind in sorted(candidates, key=lambda x: -x[0]):
        if pos < min_len or n - pos < min_len:
            continue
        if any(abs(pos - a) < min_len for a, _ in accepted):
            continue
        accepted.append((pos, kind))
    accepted.sort()

    # 4. split over-long segments at their strongest internal onset
    if max_block_secs and max_block_secs > 0:
        max_len = max(2 * min_len, round(rate * max_block_secs))
        onset_peak = np.zeros(n, dtype=bool)
        onset_peak[1:-1] = (onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:])
        changed = True
        while changed:
            changed = False
            edges = [0] + [a for a, _ in accepted] + [n]
            for s0, e0 in zip(edges[:-1], edges[1:]):
                if e0 - s0 <= max_len:
                    continue
                lo, hi = s0 + min_len, e0 - min_len
                if hi <= lo:
                    continue
                bars_in_range = bar_frames[(bar_frames > lo) & (bar_frames < hi)] if len(bar_frames) else np.array([])
                if len(bars_in_range):
                    mid = (lo + hi) / 2
                    fill_pos = int(bars_in_range[np.argmin(np.abs(bars_in_range - mid))])
                else:
                    window = np.where(onset_peak[lo:hi], onset[lo:hi], -1.0)
                    if window.max() < 0:
                        window = onset[lo:hi]
                    fill_pos = lo + int(np.argmax(window))
                accepted.append((fill_pos, "fill"))
                changed = True
            accepted.sort()

    bounds = [0] + [a for a, _ in accepted] + [n]
    kinds = ["start"] + [k for _, k in accepted]
    segments = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        segments.append({
            "start": round(s / rate, 3),
            "end": round(min(e / rate, duration), 3) if e < n else duration,
            "energy": round(float(rms[s:e].mean()), 3),
            "cut": kinds[k],
            "cut_strength": round(float(onset[s]), 3),
        })
    if on_beat:   # 🥁 every cut on the beat (changes and fills on a bar line when one is near)
        segments = land_on_beats(track, segments, beats_per_bar, min_gap=0.5 * min_block_secs)
    return segments


def _legacy_track_segments(
    track: dict,
    smooth_secs: float,
    lookback_secs: float,
    drift_threshold: float,
    min_block_secs: float,
    react_to_beats: bool = True,
) -> list[dict]:
    """Adaptive segmentation ported from the VSE add-on's block-detection
    algorithm: cut a new segment wherever the smoothed energy level has
    drifted from where it was `lookback_secs` ago by more than
    `drift_threshold` (accumulated). This alone only reacts to broad energy
    LEVEL shifts, which is too slow for sharp hits — so when react_to_beats
    is on, every detected beat is ALSO forced in as a cut candidate, keeping
    blocks reactive to individual hits, not just plateau changes. Either way,
    any resulting span shorter than `min_block_secs` is merged into whichever
    neighbour is more energetic, so beat-forcing can't fragment things below
    your chosen floor. Operates on the track's energy envelope (sampled every
    ENERGY_WINDOW_SEC seconds in audio_pipeline.py)."""
    envelope = track["energy_envelope"]
    times = np.array([e["time"] for e in envelope])
    energies = np.array([e["energy"] for e in envelope])
    n = len(energies)
    if n < 2:
        return [{"start": 0.0, "end": track["duration_sec"], "energy": float(energies[0]) if n else 0.0}]

    sample_rate = 1.0 / max(times[1] - times[0], 1e-6)  # samples per second
    hw_smooth = max(1, round(sample_rate * smooth_secs / 2.0))
    lb = max(1, round(sample_rate * lookback_secs))
    min_samples = max(1, round(sample_rate * min_block_secs))

    smoothed = _moving_average(energies, hw_smooth)

    # --- cut boundaries wherever accumulated drift crosses the threshold ---
    drift_boundaries = {0, n}
    accumulator = 0.0
    since_last = 0
    for i in range(n):
        since_last += 1
        prev_idx = max(0, i - lb)
        accumulator += abs(smoothed[i] - smoothed[prev_idx])
        if accumulator >= drift_threshold and since_last >= min_samples and i > 0:
            drift_boundaries.add(i)
            accumulator = 0.0
            since_last = 0

    # --- fold in every beat as a forced cut candidate, so blocks react to hits too ---
    if react_to_beats and track.get("beat_times"):
        for bt in track["beat_times"]:
            idx = int(round(bt * sample_rate))
            if 0 < idx < n:
                drift_boundaries.add(idx)

    raw_boundaries = sorted(drift_boundaries)

    spans = [(raw_boundaries[k], raw_boundaries[k + 1]) for k in range(len(raw_boundaries) - 1)
             if raw_boundaries[k] < raw_boundaries[k + 1]]

    # --- merge pass: absorb spans shorter than min_samples into the more energetic neighbour ---
    def avg_smoothed(span):
        lo, hi = span
        return float(smoothed[lo:hi].mean())

    changed = True
    while changed and len(spans) > 1:
        changed = False
        for k in range(len(spans)):
            s, e = spans[k]
            if (e - s) < min_samples:
                if k == 0:
                    ns, ne = spans[k + 1]; spans[k + 1] = (s, ne)
                elif k == len(spans) - 1:
                    ps, pe = spans[k - 1]; spans[k - 1] = (ps, e)
                else:
                    if avg_smoothed(spans[k + 1]) >= avg_smoothed(spans[k - 1]):
                        ns, ne = spans[k + 1]; spans[k + 1] = (s, ne)
                    else:
                        ps, pe = spans[k - 1]; spans[k - 1] = (ps, e)
                del spans[k]
                changed = True
                break

    # --- convert sample-index spans to time ranges with an attached energy value ---
    segments = []
    for s, e in spans:
        start_t = float(times[s])
        end_t = float(times[e]) if e < n else track["duration_sec"]
        segments.append({
            "start": start_t,
            "end": end_t,
            "energy": round(float(energies[s:e].mean()), 3) if e > s else 0.0,
        })

    return segments


DENSITY_SMOOTH_SECS = 1.5  # same as the VSE add-on's density signal


def compute_density_curve(track: dict) -> tuple:
    """Port of the add-on's density signal: smooth the raw energy over
    DENSITY_SMOOTH_SECS FIRST, then normalise 0-1 — so the scale reflects
    sustained loudness rather than being set by a single transient peak.
    Returns (times, density)."""
    hires = track.get("hires")
    if hires:
        rate = float(hires["rate"])
        raw = np.asarray(hires["rms"], dtype=float)
    else:
        env = track["energy_envelope"]
        raw = np.asarray([e["energy"] for e in env], dtype=float)
        rate = 1.0 / max(env[1]["time"] - env[0]["time"], 1e-6) if len(env) > 1 else 2.0
    dens = _moving_average(raw, max(1, round(rate * DENSITY_SMOOTH_SECS / 2.0)))
    lo, hi = float(dens.min()), float(dens.max())
    dens = (dens - lo) / (hi - lo) if hi - lo > 1e-9 else np.full_like(dens, 0.5)
    return np.arange(len(dens)) / rate, dens


def apply_segment_intensity(segments: list[dict], track: dict, contrast: float) -> None:
    """Port of the add-on's block density (compute_block_density_values):
    average the density curve over each segment, then RE-NORMALISE across
    segments so the quietest segment is 0 and the busiest is 1 — the full
    clip-count range is always used, whatever the track's absolute level.
    `contrast` is the add-on's Density Power (>1 spreads values apart,
    <1 flattens them). Writes seg["intensity"] in place."""
    times, dens = compute_density_curve(track)
    means = []
    for seg in segments:
        mask = (times >= seg["start"]) & (times < seg["end"])
        means.append(float(dens[mask].mean()) if mask.any() else 0.0)
    lo, hi = min(means), max(means)
    for seg, m in zip(segments, means):
        norm = (m - lo) / (hi - lo) if hi - lo > 1e-9 else 0.5
        seg["intensity"] = round(norm ** max(0.05, contrast), 3)


def clips_for_intensity(intensity: float, min_clips: int, max_clips: int,
                        ramp_start: float, ramp_end: float, vary: bool, rng, skip=None) -> int:
    """_clips_for_intensity, then 'Skip these counts' (allowed_clip_count)."""
    return allowed_clip_count(_clips_for_intensity(intensity, min_clips, max_clips, ramp_start, ramp_end, vary, rng),
                              min_clips, max_clips, skip)


def _clips_for_intensity(intensity: float, min_clips: int, max_clips: int,
                         ramp_start: float, ramp_end: float, vary: bool, rng) -> int:
    """Map a segment's 0-1 intensity to a clip count.

    The ramp range rescales intensity first (at/below ramp_start = 0, at/above
    ramp_end = 1). Then, following the add-on's _overlap_count_from_intensity:
    intensity sets both a ceiling (min + i*span) and a floor that only starts
    rising above the midpoint (min + max(0, (i-0.5)*2)*span).
      vary off -> always the ceiling (the add-on's FIXED mode, smooth & predictable)
      vary on  -> a seeded pick between floor and ceiling (the add-on's RANDOM mode)
    Either way intensity 0 guarantees min_clips and 1 guarantees max_clips."""
    if max_clips <= min_clips:
        return min_clips
    if ramp_end <= ramp_start:
        pos = 1.0 if intensity >= ramp_start else 0.0
    else:
        pos = min(1.0, max(0.0, (intensity - ramp_start) / (ramp_end - ramp_start)))
    span = max_clips - min_clips
    ceiling = int(round(min_clips + pos * span))
    if not vary:
        return ceiling
    floor = min(ceiling, int(round(min_clips + max(0.0, (pos - 0.5) * 2) * span)))
    return rng.randint(floor, ceiling)


def allowed_clip_count(n: int, min_clips: int, max_clips: int, skip=None) -> int:
    """'Skip these counts' (sidebar): n moved to the nearest count that's allowed
    (ties go to the smaller). Min and max are always allowed."""
    skip = {int(x) for x in (skip or [])} - {min_clips, max_clips}
    if n not in skip:
        return n
    allowed = [c for c in range(min_clips, max_clips + 1) if c not in skip]
    return min(allowed, key=lambda c: (abs(c - n), c)) if allowed else n
