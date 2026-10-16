"""
"Smart" cut method — a self-tuning edit rhythm built from music analysis and
film-editing practice, not from hand-set thresholds.

The model, in the order it runs:

1. METRICAL GRID. Music is organised in nested levels (beat → half bar → bar →
   2 bars → phrase of 4 / 8 bars → section; Lerdahl & Jackendoff's metrical
   hierarchy). Cuts are only ever placed on this grid, and a steady cutting
   rhythm comes from choosing a LEVEL to cut on, not from scoring single beats.
   The beat grid is the corrected one from cut_scoring.beat_grid (half/double
   time fixed, gaps filled), bars start where kicks / chord changes line up.

2. INTENSITY. How "big" the music feels at each beat, from what the analysis
   measured: loudness (in dB — perceived loudness is logarithmic), event density
   (onsets per second over a bar — busier music feels faster), low end (kick),
   brightness (rises in build-ups) and spectral change. Each is put on the SONG'S
   OWN scale (its 10th–90th percentile), so a quiet song still has a quiet and a
   loud end; the mix is smoothed over a bar.

3. SECTIONS. Editors hold a pace through a passage and change it at structural
   boundaries. Sections come from the analysis (moved onto the real bar lines),
   split further where intensity changes a lot inside one, and every section
   start gets a cut.

4. PACE PER SECTION. Shot length falls as intensity rises (the cutting-rate /
   arousal relationship that music videos and trailers follow). The middle pace
   is the grid level whose length is closest to ~2.5 s at this tempo; each
   section moves up or down the level ladder from there by its intensity. How
   far it can move depends on how dynamic the song is (its intensity range), so
   a flat song keeps an even pace and a dynamic one swings. Never shorter than
   ~0.6 s (shots below that stop reading as shots) or longer than ~12 s.

5. BUILD-UPS accelerate: the level halves through the build (e.g. every 2 bars
   → bar → half bar) so the cutting tightens into the drop, and the drop always
   gets a cut.

6. ACCENTS. Moments that stand out from THIS song (a hit stronger than its 98th
   percentile, a drop, a vocal entry after a gap) get their own cut, on the
   nearest beat; grid cuts too close to an accent give way to it.

7. VOCALS. A cut that would fall in the middle of a sung line moves (within a
   beat) to the gap between lines, when the track has vocal analysis.

8. REPETITION. Sections the analysis labels as the same (chorus 1 and 2) are
   cut with the same pattern, so repeats rhyme visually.

Everything comes out as the planner's usual segment list, plus a report of what
was chosen and why. No Streamlit in here.
"""

from __future__ import annotations

import math

import numpy as np

import cut_scoring as CS

DEFAULTS = {
    "sm_pace": 0.0,          # −2 … +2 grid levels from the song's own middle pace
    "sm_contrast": 1.0,      # 0 = same pace everywhere, 1 = auto from the song, 2 = exaggerated
    "sm_builds": True,
    "sm_accents": True,
    "sm_vocals": True,
    "sm_mirror": True,
}
TARGET_SHOT_SEC = 2.5        # the middle pace: comfortable average shot at any tempo
MIN_SHOT_SEC = 0.6           # at contrast 1; contrast above 1 lets both limits stretch
MAX_SHOT_SEC = 12.0
MID_INTENSITY = 0.56         # the song's "usual" level on its own 0–1 intensity scale
# Grid levels per unit of intensity away from the usual level, at contrast 1 for
# a song of typical dynamics: +0.12 above usual ≈ one level faster, 0.11 below ≈
# one slower (calm passages tolerate long shots more than loud ones tolerate short).
LEVELS_PER_INTENSITY = {"faster": 4.5, "slower": 4.5}
USUAL_BAND = 0.05            # within ± this of the usual level: the middle pace exactly
# Grid levels, in beats per cut, from longest to shortest (bars are bpb beats).
LEVEL_BARS = [8, 4, 2, 1, 0.5, 0.25]   # 0.5 bar = half bar, 0.25 bar = 1 beat in 4/4


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def _curve(track: dict, key: str):
    a = track.get("analysis") or {}
    hires = track.get("hires") or {}
    if key in ("rms", "onset"):
        return np.asarray(hires.get(key) or [], dtype=float), float(hires.get("rate", 30))
    v = a.get(key)
    return (np.asarray(v, dtype=float), float(a.get("rate", 30))) if v is not None else (np.zeros(0), 30.0)


def _mean_between(curve: np.ndarray, rate: float, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if not len(curve):
        return np.zeros(len(starts))
    c = np.concatenate([[0.0], np.cumsum(curve)])
    a = np.clip((starts * rate).astype(int), 0, len(curve) - 1)
    b = np.clip(np.maximum(a + 1, (ends * rate).astype(int)), 1, len(curve))
    return (c[b] - c[a]) / (b - a)


def _robust01(x: np.ndarray) -> np.ndarray:
    """Onto the song's own scale: its 10th percentile → 0, 90th → 1."""
    if not len(x):
        return x
    lo, hi = np.percentile(x, 10), np.percentile(x, 90)
    if hi - lo < 1e-9:
        return np.full(len(x), 0.5)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def intensity_per_beat(track: dict, beats: np.ndarray, bar_len: float) -> tuple:
    """(intensity 0–1 per beat interval, {feature: values}) — see module doc, step 2."""
    if len(beats) < 2:
        return np.zeros(0), {}
    starts, ends = beats[:-1], beats[1:]
    rms, r_rate = _curve(track, "rms")
    onset, o_rate = _curve(track, "onset")
    kick, k_rate = _curve(track, "kick")
    bright, b_rate = _curve(track, "brightness")
    novelty, n_rate = _curve(track, "novelty")

    feats = {}
    loud = _mean_between(rms, r_rate, starts, ends)
    feats["loudness"] = _robust01(20 * np.log10(np.maximum(loud, 1e-4)))
    # event density: onset peaks per second over the surrounding bar
    if len(onset):
        pk = np.flatnonzero((onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:]) &
                            (onset[1:-1] > np.percentile(onset, 60))) + 1
        times = pk / o_rate
        mid = (starts + ends) / 2
        dens = np.array([np.sum(np.abs(times - m) <= bar_len / 2) for m in mid]) / bar_len
        feats["density"] = _robust01(dens)
    if len(kick):
        feats["low_end"] = _robust01(_mean_between(kick, k_rate, starts, ends))
    if len(bright):
        feats["brightness"] = _robust01(_mean_between(bright, b_rate, starts, ends))
    if len(novelty):
        feats["change"] = _robust01(_mean_between(novelty, n_rate, starts, ends))
    weights = {"loudness": 0.35, "density": 0.25, "low_end": 0.2, "brightness": 0.1, "change": 0.1}
    have = {k: w for k, w in weights.items() if k in feats}
    total = sum(have.values()) or 1.0
    mix = sum(feats[k] * w for k, w in have.items()) / total
    # smooth over about a bar
    beats_per_bar = max(1, int(round(bar_len / max(np.median(np.diff(beats)), 1e-6))))
    k = np.ones(beats_per_bar) / beats_per_bar
    smooth = np.convolve(np.pad(mix, (beats_per_bar // 2, beats_per_bar - 1 - beats_per_bar // 2), mode="edge"),
                         k, mode="valid")
    return np.clip(smooth, 0, 1), feats


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def _nearest_idx(grid: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(grid - t)))


def find_sections(track: dict, bars: np.ndarray, bar_int: np.ndarray, bpb: int) -> list:
    """[{start_bar, end_bar, label}] on bar indices. Analysis sections (snapped
    to bars) when present, split where a stretch of 8+ bars changes intensity
    by a lot; otherwise 8-bar blocks merged by similar intensity."""
    a = track.get("analysis") or {}
    n = len(bars) - 1
    cuts, labels = {0: "A"}, {}
    for s in a.get("sections") or []:
        snapped = CS.snapped_marks(track, [s["start"]], bpb)[0]
        i = _nearest_idx(bars, snapped)
        if 1 < i < n:   # one starting in the first bar is the song's start
            cuts[i] = s.get("label") or "?"
    if len(cuts) == 1:
        # No structure from the analysis: 8-bar blocks (the usual phrase length).
        for i in range(8, n, 8):
            cuts[i] = "?"
    idx = sorted(cuts)
    # Split long sections where intensity steps up or down (e.g. a quiet intro
    # that the timbre-based sections didn't separate from what follows).
    extra = {}
    for j, s in enumerate(idx):
        e = idx[j + 1] if j + 1 < len(idx) else n
        for k in range(s + 4, e - 3, 4):
            before, after = bar_int[max(s, k - 4):k].mean(), bar_int[k:min(e, k + 4)].mean()
            if abs(after - before) > 0.3:
                extra[k] = cuts[s] + "'"
    cuts.update(extra)
    idx = sorted(cuts)
    out = []
    for j, s in enumerate(idx):
        e = idx[j + 1] if j + 1 < len(idx) else n
        if e > s:
            out.append({"start_bar": s, "end_bar": e, "label": cuts[s]})
    return out


def _builds_and_drops(track: dict, bars: np.ndarray, bpb: int) -> tuple:
    a = track.get("analysis") or {}
    drops = CS.snapped_marks(track, a.get("drops") or [], bpb)
    builds = []
    for b in a.get("builds") or []:
        s, e = CS.snapped_marks(track, [b["start"], b["end"]], bpb)
        builds.append((s, e))
    return builds, drops


# ---------------------------------------------------------------------------
# The cut plan
# ---------------------------------------------------------------------------

def _level_for(length_sec: float, bar_len: float) -> int:
    """Index into LEVEL_BARS whose length is closest to length_sec (log scale)."""
    return int(np.argmin([abs(math.log(max(lb * bar_len, 1e-6) / length_sec)) for lb in LEVEL_BARS]))


def plan_cuts(track: dict, cfg: dict) -> tuple:
    """(segments, report). report = {"tempo", "bar_sec", "middle", "range",
    "sections": [{start, end, label, intensity, every, reason}], "notes": [...]}."""
    cfg = {**DEFAULTS, **(cfg or {})}
    duration = float(track["duration_sec"])
    bpb = int(cfg.get("beats_per_bar", 4))
    beats, bpm, fix = CS.beat_grid(track)
    bars = np.asarray(CS.bar_lines(track, bpb), dtype=float)
    notes = []
    if len(beats) < 8 or len(bars) < 4:
        return ([{"start": 0.0, "end": round(duration, 3), "energy": 0.0, "cut": "start", "cut_strength": 0.0}],
                {"notes": ["Not enough beat data to plan cuts — run the audio analysis on this track."],
                 "sections": []})
    bar_len = float(np.median(np.diff(bars)))
    if fix:
        notes.append({"doubled": "Beat tracker found every other beat — corrected.",
                      "halved": "Beat tracker found twice the beats — corrected."}[fix])
    # grid incl. the lead-in and the end
    if bars[0] > 0.05:
        bars = np.concatenate([[0.0], bars]) if bars[0] > bar_len * 0.5 else np.concatenate([[0.0], bars[1:]])
    bars = np.concatenate([bars, [duration]]) if duration - bars[-1] > 0.05 else bars
    beats_in = beats[(beats > 0.05) & (beats < duration - 0.05)]
    grid_beats = np.unique(np.round(np.concatenate([[0.0], beats_in, bars, [duration]]), 3))

    inten, feats = intensity_per_beat(track, grid_beats, bar_len)
    beat_mid = (grid_beats[:-1] + grid_beats[1:]) / 2
    bar_int = np.array([inten[(beat_mid >= bars[i]) & (beat_mid < bars[i + 1])].mean()
                        if np.any((beat_mid >= bars[i]) & (beat_mid < bars[i + 1])) else 0.0
                        for i in range(len(bars) - 1)])
    sections = find_sections(track, bars, bar_int, bpb)
    builds, drops = _builds_and_drops(track, bars, bpb)

    # The song's own middle pace and how far it may swing.
    middle = _level_for(TARGET_SHOT_SEC, bar_len)
    middle = int(np.clip(middle + round(float(cfg["sm_pace"])), 0, len(LEVEL_BARS) - 1))   # + = busier
    sec_int = np.array([bar_int[s["start_bar"]:s["end_bar"]].mean() for s in sections])
    spread = float(np.percentile(sec_int, 90) - np.percentile(sec_int, 10)) if len(sec_int) > 1 else 0.0
    contrast = float(cfg["sm_contrast"])
    # Contrast scales how many grid levels a section moves per unit of intensity
    # away from the song's usual level — continuously, so every step of the slider
    # changes something. A more dynamic song moves a little further. Past 1 the
    # shortest / longest shot limits stretch too (otherwise at many tempos they'd be
    # what stops the pace going further).
    dyn = float(np.clip((0.5 + spread) / 1.1, 0.6, 1.25))
    gain = {k: v * contrast * dyn for k, v in LEVELS_PER_INTENSITY.items()}
    stretch = max(0.0, contrast - 1.0)
    min_shot = max(0.4, MIN_SHOT_SEC * 0.6 ** stretch)
    max_shot = MAX_SHOT_SEC * 1.6 ** stretch
    # The same limits as fractional grid levels (level L = 8 / 2^L bars).
    lvl_lo = max(0.0, math.log2(8 * bar_len / max_shot))
    lvl_hi = min(len(LEVEL_BARS) - 1.0, math.log2(8 * bar_len / min_shot))
    min_level = len(LEVEL_BARS) - 1
    while min_level > 0 and LEVEL_BARS[min_level] * bar_len < min_shot:
        min_level -= 1
    max_level = 0
    while max_level < min_level and LEVEL_BARS[max_level] * bar_len > max_shot:
        max_level += 1
    swing_fast, swing_slow = min_level - middle, middle - max_level
    notes.append(f"Middle pace: a cut every {_fmt_level(LEVEL_BARS[middle])} "
                 f"({LEVEL_BARS[middle] * bar_len:.1f}s) — the grid level nearest {TARGET_SHOT_SEC:g}s at this tempo")
    notes.append(f"Song dynamics {spread:.2f} (0 = flat, 1 = huge) × contrast {contrast:g} → about "
                 f"{gain['faster'] * 0.2:.1f} grid level(s) faster for a section 0.2 above the usual intensity; shots kept between "
                 f"{min_shot:.2f}s and {max_shot:.0f}s")

    # Intensity is already on the song's own 0–1 scale (step 2), so a section's
    # place on it decides its pace: the top band cuts faster than the middle pace,
    # the lower bands slower (how many levels: the swings above).

    # Build-ups the analysis didn't mark: a section whose intensity climbs and
    # leads into a clearly bigger one — its last (up to) 8 bars count as a build.
    for si in range(len(sections) - 1):
        sb, eb = sections[si]["start_bar"], sections[si]["end_bar"]
        if eb - sb >= 4 and sec_int[si + 1] > sec_int[si] + 0.1:
            seg = bar_int[sb:eb]
            if float(np.polyfit(np.arange(len(seg)), seg, 1)[0]) * len(seg) > 0.15:
                t_end = float(bars[eb]) if eb < len(bars) else duration
                if not any(abs(be - t_end) < bar_len for _, be in builds):
                    builds.append((max(float(bars[sb]), t_end - 8 * bar_len), t_end))
                    if not any(abs(d - t_end) < bar_len for d in drops):
                        drops.append(t_end)
    cut_times, kinds = {}, {}

    def add(t, kind, prio):
        t = round(float(t), 3)
        if t <= 0.05 or t >= duration - 0.05:
            return
        old = kinds.get(t)
        if old is None or old[1] < prio:
            cut_times[t] = True
            kinds[t] = (kind, prio)

    report_secs = []
    first_of_label, patterns = {}, {}
    units = []   # (start, end, cut length) per section, for accents
    for si, sec in enumerate(sections):
        s_bar, e_bar = sec["start_bar"], sec["end_bar"]
        t0, t1 = float(bars[s_bar]), float(bars[e_bar]) if e_bar < len(bars) else duration
        xi = float(sec_int[si])
        d = xi - MID_INTENSITY
        d = 0.0 if abs(d) < USUAL_BAND else d - math.copysign(USUAL_BAND, d)
        # A fractional level: between two grid levels the section mixes them (e.g.
        # 1.5 bars → shots of 1 and 2 bars), so every contrast / intensity change shows.
        lvl_f = float(np.clip(middle + d * (gain["faster"] if d > 0 else gain["slower"]), lvl_lo, lvl_hi))
        level = int(round(lvl_f))
        unit = 8.0 / 2 ** lvl_f * bar_len          # LEVEL_BARS are 8 / 2^level bars
        snap_len = LEVEL_BARS[min(len(LEVEL_BARS) - 1, int(math.ceil(lvl_f - 1e-6)))] * bar_len
        reason = "one of the song's most intense parts" if sec_int[si] >= 0.68 else \
            "one of the song's quietest parts" if sec_int[si] <= 0.28 else \
            "quieter than the song's usual level" if sec_int[si] <= 0.45 else "the song's usual intensity"
        if si > 0:
            add(t0, "sm_section", 5)
        in_build = [(bs, be) for bs, be in builds if bs < t1 and be > t0]
        label = sec["label"]
        pattern = []
        if cfg["sm_mirror"] and label in patterns and label not in ("?",) and not in_build:
            src_len, src, src_int = patterns[label]
            # Only a true repeat: same length AND about as intense (the analysis can
            # label an intro and a verse alike because they share chords).
            if abs(src_len - (t1 - t0)) < bar_len * 0.6 and abs(src_int - sec_int[si]) < 0.15:
                for off in src:
                    add(t0 + off, "sm_grid", 1)
                report_secs.append({"start": t0, "end": t1, "label": label, "intensity": round(float(sec_int[si]), 2),
                                    "every": first_of_label[label], "reason": f"same as the earlier '{label}'"})
                units.append((t0, t1, unit))
                continue
        # Cuts at every `unit`, each moved onto the nearest point of the finer grid
        # level (counted from the section start), then onto the nearest beat.
        t = t0 + unit
        last = t0
        while t < t1 - unit * 0.5:
            g_snap = t0 + round((t - t0) / snap_len) * snap_len
            g = float(grid_beats[_nearest_idx(grid_beats, g_snap)])
            if g > last + 1e-3 and g < t1 - 1e-3:
                add(g, "sm_grid", 1)
                pattern.append(round(g - t0, 3))
                last = g
            t += unit
        every = _fmt_level(8.0 / 2 ** lvl_f)
        if cfg["sm_builds"]:
            for bs, be in in_build:
                bs, be = max(bs, t0, be - 8 * bar_len), min(be, t1)   # the last 8 bars into the drop
                if be - bs < bar_len * 2:
                    continue
                # remove grid cuts inside the build, then accelerate: each quarter one level
                # faster, from the middle pace (or slower) to two levels above it
                for c in [c for c in cut_times if bs < c < be and kinds[c][0] == "sm_grid"]:
                    cut_times.pop(c), kinds.pop(c)
                lvl = int(np.clip(min(level, middle) - 1, max_level, min_level))
                q = (be - bs) / 4
                for part in range(4):
                    a, b = bs + part * q, bs + (part + 1) * q
                    u = LEVEL_BARS[lvl] * bar_len
                    tt = a + (u if part == 0 else 0)
                    while tt < b - 1e-6:
                        add(grid_beats[_nearest_idx(grid_beats, tt)], "sm_build", 2)
                        tt += u
                    lvl = min(lvl + 1, min_level)
                every += " · speeds up through the build"
        if label not in patterns:
            patterns[label] = (t1 - t0, pattern, float(sec_int[si]))
            first_of_label[label] = every
        report_secs.append({"start": t0, "end": t1, "label": label, "intensity": round(float(sec_int[si]), 2),
                            "every": every, "reason": reason})
        units.append((t0, t1, unit))

    for d in drops:
        add(grid_beats[_nearest_idx(grid_beats, d)], "sm_drop", 6)

    # Accents: standout hits and vocal entries, on the nearest beat.
    def unit_at(t):
        return next((u for a_, b_, u in units if a_ <= t < b_), LEVEL_BARS[middle] * bar_len)

    def room_for(t):
        # an accent only where the section's own rhythm leaves a gap: at least half
        # a unit from any cut already planned
        return all(abs(t - c) >= 0.5 * unit_at(t) for c in cut_times)

    if cfg["sm_accents"]:
        accents = []
        onset, o_rate = _curve(track, "onset")
        if len(onset):
            thr = np.percentile(onset, 98)
            pk = np.flatnonzero((onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:]) & (onset[1:-1] >= thr)) + 1
            w = max(1, int(2 * bar_len * o_rate))
            for p in sorted(pk, key=lambda i: -onset[i]):
                local = onset[max(0, p - w):p + w]
                if onset[p] >= 2.0 * float(np.median(local)):   # stands out from its surroundings
                    accents.append((float(p / o_rate), "sm_accent"))
        a = track.get("analysis") or {}
        for v in a.get("vocal_starts") or []:
            accents.append((float(v), "sm_vocal"))
        budget = max(1, int(duration / (8 * bar_len)))   # at most ~1 accent per 8 bars
        for t, kind in accents:
            if budget <= 0:
                break
            g = float(grid_beats[_nearest_idx(grid_beats, t)])
            if room_for(g):
                add(g, kind, 3)
                budget -= 1

    # Enforce the shortest shot: on conflict keep the higher-priority cut.
    ordered = sorted(cut_times)
    kept = []
    for t in sorted(ordered, key=lambda x: -kinds[x][1]):
        if all(abs(t - k) >= min_shot for k in kept):
            kept.append(t)
    kept.sort()
    # Lower-priority grid cuts closer than half their section's unit to an accent give way.
    final = []
    for t in kept:
        final.append(t)

    # Vocals: move a cut out of the middle of a sung line (within a beat, to a beat).
    if cfg["sm_vocals"]:
        vocals, v_rate = _curve(track, "vocals")
        if len(vocals):
            beat_len = bar_len / bpb
            moved = []
            for t in final:
                if kinds[t][1] >= 5:
                    moved.append(t)
                    continue
                here = vocals[min(len(vocals) - 1, int(t * v_rate))]
                if here > 0.35:
                    opts = [g for g in grid_beats if abs(g - t) <= beat_len * 1.01 and g != t]
                    quiet = [g for g in opts if vocals[min(len(vocals) - 1, int(g * v_rate))] < 0.2]
                    if quiet:
                        g = min(quiet, key=lambda g: abs(g - t))
                        kinds[round(float(g), 3)] = kinds[t]
                        t = round(float(g), 3)
                moved.append(t)
            final = sorted(set(moved))

    # Too-long gaps (e.g. sections the analysis made very long) get split on the grid.
    bounds = [0.0] + final + [duration]
    out_bounds = [0.0]
    for a, b in zip(bounds, bounds[1:]):
        if b - a > max_shot:
            n_parts = int(math.ceil((b - a) / max_shot))
            for k in range(1, n_parts):
                g = round(float(grid_beats[_nearest_idx(grid_beats, a + k * (b - a) / n_parts)]), 3)
                if out_bounds[-1] + min_shot <= g <= b - min_shot:
                    out_bounds.append(g)
                    kinds.setdefault(g, ("sm_grid", 1))
        out_bounds.append(b)

    onset, o_rate = _curve(track, "onset")
    rms, r_rate = _curve(track, "rms")
    segs = []
    for i in range(len(out_bounds) - 1):
        s, e = out_bounds[i], out_bounds[i + 1]
        if e - s <= 1e-3:
            continue
        energy = float(_mean_between(rms, r_rate, np.array([s]), np.array([e]))[0]) if len(rms) else 0.0
        strength = float(onset[min(len(onset) - 1, int(s * o_rate))]) if len(onset) else 0.0
        segs.append({"start": round(s, 3), "end": round(e, 3), "energy": round(energy, 3),
                     "cut": "start" if i == 0 else kinds.get(round(s, 3), ("sm_grid", 1))[0],
                     "cut_strength": round(strength, 3)})
    report = {"tempo": round(bpm, 1), "bar_sec": round(bar_len, 3), "middle": _fmt_level(LEVEL_BARS[middle]),
              "range": (swing_fast, swing_slow), "sections": report_secs, "notes": notes,
              "features": sorted(feats),
              # for the story map wizard (story_map.py)
              "builds": [(round(float(b0), 3), round(float(b1), 3)) for b0, b1 in builds],
              "drops": sorted(round(float(d), 3) for d in drops)}
    return segs, report


def _fmt_level(bars_per_cut: float) -> str:
    b = round(bars_per_cut, 2)
    exact = min(LEVEL_BARS, key=lambda x: abs(x - b))
    approx = "" if abs(exact - b) < 0.03 * exact else "~"
    if approx:
        return f"~{b:.2g} bars" if b >= 1 else f"~{b * 4:.2g} beats"
    if exact >= 1:
        return f"{exact:g} bar{'s' if exact != 1 else ''}"
    if exact == 0.5:
        return "half bar"
    return f"{exact * 4:g} beat{'s' if exact * 4 != 1 else ''}"


def build_smart_segments(track: dict, cfg: dict) -> list:
    return plan_cuts(track, cfg)[0]
