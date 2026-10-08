"""
Video ↔ track match ranking for the Compilation Planner's video list.

Each video gets a 0–100 match score: a weighted mix of separate metrics, each
scored 0–1 (1 = ideal), so the planner can show WHY a video ranks where it does
and the weights can be tuned in "⚙️ Ranking weights".

  dynamics   how much the video's motion swings vs. how much the track's energy swings
  punch      sharp motion peaks per minute vs. strong audio hits per minute
  footage    enough usable footage for this track (full marks at ¼ of its length)
  pace       how fast the video cuts vs. how fast the track is (BPM)
  spread     share of time calm / medium / intense — compared as whole distributions
  relevance  with a tag filter: how much of the video actually carries those tags
  freshness  fewer appearances in recent saved projects = higher
  tempo  ⚗  dominant rhythm of the movement vs. the track's BPM (½× and 2× count)
  arc    ⚗  the track's energy over time vs. the video's motion over its usable range

"vs." metrics compare each side against its OWN library (where this video sits
among all videos, where this track sits among all tracks), because audio energy
and video motion live on different scales. The video library used for that is
always the whole library, so changing the tag filter doesn't shift every score.

Frame-level motion comes from each scene's motion_curve, condensed once per
video into a small summary file (.ranking_cache/<video_id>.npz next to the
catalogues) and rebuilt only when that video's catalogue file changes.

No Streamlit in here — the planner page passes everything in.
"""

import json
import math
from pathlib import Path

import numpy as np

SUMMARY_RATE_HZ = 16          # motion curves are pooled down to ~this many samples/sec
HIST_BINS = 12
ARC_POINTS = 60
FOOTAGE_SHARE = 0.25          # full "footage" marks once usable ≥ this share of the track
RECENT_PROJECTS = 10          # freshness looks at this many most recent saved projects

METRICS = ["dynamics", "punch", "footage", "pace", "spread", "relevance", "freshness", "tempo", "arc"]
EXPERIMENTAL = {"tempo", "arc"}
LABELS = {
    "dynamics": "Dynamics", "punch": "Punch", "footage": "Enough footage", "pace": "Cut pace",
    "spread": "Intensity spread", "relevance": "Tag relevance", "freshness": "Freshness",
    "tempo": "Movement tempo ⚗", "arc": "Energy arc ⚗",
}
DEFAULT_WEIGHTS = {
    "dynamics": 2.0, "punch": 2.0, "footage": 2.0, "pace": 1.0, "spread": 1.0,
    "relevance": 1.0, "freshness": 0.5, "tempo": 0.0, "arc": 0.0,
}
HELP = {
    "dynamics": "How much the video's motion swings compared with how much the track's energy swings — "
                "each judged against its own library.",
    "punch": "Sharp motion peaks per minute (from the frame-by-frame motion curves) compared with strong "
             "audio hits per minute.",
    "footage": f"Enough usable footage for this track: full marks at {int(FOOTAGE_SHARE * 100)}% of the "
               "track's length or more, less below that. Counts only footage inside the time range and "
               "matching the tag filter.",
    "pace": "Fast-cutting videos (short scenes) for fast tracks (high BPM), long takes for slow ones.",
    "spread": "Whether the video spends its time calm / medium / intense in the same proportions the "
              "track does — 'mostly calm with rare peaks' vs 'busy throughout'.",
    "relevance": "Only with a tag filter: the share of the video's usable footage that carries those "
                 "tags. Without a filter it has no effect.",
    "freshness": f"Lower for videos selected in your {RECENT_PROJECTS} most recent saved projects, "
                 "for more variety between compilations.",
    "tempo": "EXPERIMENTAL — finds the dominant rhythm in the video's movement and compares it with the "
             "track's BPM (half and double time count as a match). Scored by how clear that rhythm is.",
    "arc": "EXPERIMENTAL — the track's energy over time (build, drop, outro) against the video's motion "
           "over its usable range, both stretched to the same length. Most useful in sequential mode.",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _tc(tc: str) -> float:
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _percentile_of(value: float, population: list) -> float:
    """Where value sits within population, 0..1 (mid-rank, so ties land in the middle)."""
    pop = np.asarray(population, dtype=float)
    if pop.size == 0:
        return 0.5
    below = float((pop < value).sum())
    equal = float((pop == value).sum())
    return (below + 0.5 * equal) / pop.size


def _closeness(p1: float, p2: float) -> float:
    return max(0.0, 1.0 - abs(p1 - p2))


def _hist(values: np.ndarray) -> np.ndarray:
    """Shape of a value distribution: values scaled by their own 95th percentile."""
    if values.size == 0:
        return np.full(HIST_BINS, 1.0 / HIST_BINS)
    scale = np.percentile(values, 95) or (values.max() or 1.0)
    h, _ = np.histogram(np.clip(values / scale, 0, 1), bins=HIST_BINS, range=(0, 1))
    return h / max(h.sum(), 1)


def _resample(values: np.ndarray, n: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(n)
    edges = np.linspace(0, values.size, n + 1).astype(int)
    return np.array([values[a:max(b, a + 1)].mean() for a, b in zip(edges[:-1], edges[1:])])


def _peaks_per_min(values: np.ndarray, rate: float, minutes: float) -> float:
    """Local maxima standing well clear of the video's own typical level
    (median + 2.5 × robust spread), at least 0.25 s apart."""
    if values.size < 3 or minutes <= 0:
        return 0.0
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826 or values.std() or 1.0
    z = (values - med) / mad
    is_peak = (z[1:-1] > 2.5) & (z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:])
    idx = np.nonzero(is_peak)[0] + 1
    min_gap = max(1, int(0.25 * rate))
    count, last = 0, -10**9
    for i in idx:
        if i - last >= min_gap:
            count += 1
            last = i
    return count / minutes


def _tempo(segments: list, rate: float) -> tuple:
    """(bpm, clarity 0..1) of the dominant rhythm in the movement: average the
    autocorrelation of every usable stretch ≥ 3 s, then take the strongest
    peak between 30 and 180 BPM. clarity is that peak's height."""
    lo, hi = int(rate * 60 / 180), int(math.ceil(rate * 60 / 30))
    acc, weight = np.zeros(hi + 2), 0.0
    for seg in segments:
        if seg.size < max(int(3 * rate), hi + 3):
            continue
        k = max(1, int(rate))  # remove slow drift (≈1 s moving average, edge-corrected)
        smooth = np.convolve(seg, np.ones(k), mode="same") / np.convolve(np.ones(seg.size), np.ones(k), mode="same")
        x = seg - smooth
        x = x - x.mean()
        denom = float((x * x).sum())
        if denom <= 1e-12:
            continue
        ac = np.array([float((x[:-L] * x[L:]).sum()) / denom if L else 1.0 for L in range(hi + 2)])
        acc += ac * seg.size
        weight += seg.size
    if weight == 0:
        return None, 0.0
    ac = acc / weight
    window = ac[lo:hi + 1]
    if window.size == 0:
        return None, 0.0
    i = int(np.argmax(window)) + lo
    peak = ac[i]
    if 0 < i < len(ac) - 1:  # parabolic refinement of the lag
        a, b, c = ac[i - 1], ac[i], ac[i + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        lag = i + max(-0.5, min(0.5, shift))
    else:
        lag = i
    return 60.0 * rate / lag, float(max(0.0, min(1.0, peak)))


# ---------------------------------------------------------------------------
# Per-video summary (built from the full catalogue, cached on disk)
# ---------------------------------------------------------------------------

def _summary_path(catalogue_dir: Path, video_id: str) -> Path:
    return catalogue_dir / ".ranking_cache" / f"{video_id}.npz"


def build_summary(catalogue: dict) -> dict:
    """Condense every scene's motion curve to ~SUMMARY_RATE_HZ. Scenes without a
    curve (motion-curve backfill not run) use their average motion as a flat line."""
    ids, starts, ends, offsets, rates, has_curve, chunks = [], [], [], [0], [], [], []
    for s in sorted(catalogue["scenes"], key=lambda s: s["scene_id"]):
        try:
            start, end = _tc(s["start_tc"]), _tc(s["end_tc"])
        except (KeyError, ValueError):
            continue
        curve = s.get("motion_curve") or {}
        vals = np.asarray(curve.get("values") or [], dtype=float)
        fps = float(curve.get("fps") or 0)
        if vals.size and fps > 0:
            k = max(1, int(round(fps / SUMMARY_RATE_HZ)))
            n = vals.size // k
            pooled = vals[:n * k].reshape(n, k).mean(axis=1) if n else vals[:1]
            rate, flag = fps / k, True
        else:
            rate, flag = float(SUMMARY_RATE_HZ), False
            n = max(1, int((end - start) * rate))
            pooled = np.full(n, float(s.get("motion_intensity", 0.0)))
        ids.append(s["scene_id"]); starts.append(start); ends.append(end)
        rates.append(rate); has_curve.append(flag); chunks.append(pooled.astype(np.float32))
        offsets.append(offsets[-1] + pooled.size)
    return {
        "scene_ids": np.array(ids, dtype=np.int64), "starts": np.array(starts), "ends": np.array(ends),
        "offsets": np.array(offsets, dtype=np.int64), "rates": np.array(rates),
        "has_curve": np.array(has_curve, dtype=bool),
        "values": np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32),
    }


def load_summary(catalogue_dir: Path, video_id: str) -> dict | None:
    """The cached summary, rebuilt if the catalogue file is newer. None if the
    catalogue can't be read."""
    cat_path = catalogue_dir / f"{video_id}.json"
    cache = _summary_path(catalogue_dir, video_id)
    try:
        cat_mtime = cat_path.stat().st_mtime
    except OSError:
        return None
    if cache.exists():
        try:
            with np.load(cache) as z:
                if float(z["source_mtime"]) == cat_mtime:
                    return {k: z[k] for k in z.files if k != "source_mtime"}
        except Exception:
            pass  # unreadable/old format — rebuild below
    try:
        catalogue = json.loads(cat_path.read_text())
    except (ValueError, OSError):
        return None
    summary = build_summary(catalogue)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez_compressed(tmp, source_mtime=np.array(cat_mtime), **summary)
        tmp.replace(cache)
    except OSError:
        pass  # caching is an optimisation only
    return summary


def missing_summaries(catalogue_dir: Path, video_ids: list) -> list:
    """Videos whose summary needs (re)building — so the page can show progress."""
    out = []
    for vid in video_ids:
        cache = _summary_path(catalogue_dir, vid)
        try:
            cat_mtime = (catalogue_dir / f"{vid}.json").stat().st_mtime
            with np.load(cache) as z:
                if float(z["source_mtime"]) == cat_mtime:
                    continue
        except Exception:
            pass
        out.append(vid)
    return out


# ---------------------------------------------------------------------------
# Raw features
# ---------------------------------------------------------------------------

def video_features(summary: dict, scenes_by_id: dict, time_range, tag_filter: tuple) -> dict | None:
    """Raw features of one video's usable footage: inside time_range, not
    excluded, and (if tag_filter) carrying at least one of those tags.
    scenes_by_id: scene_id -> slim scene dict (tags already corrected)."""
    wanted = {t.lower() for t in tag_filter}
    segments, curve_segments, scene_lens = [], [], []
    usable_sec = in_range_sec = 0.0
    rate = float(np.median(summary["rates"])) if summary["rates"].size else float(SUMMARY_RATE_HZ)
    curve_scenes = 0
    for i, sid in enumerate(summary["scene_ids"]):
        scene = scenes_by_id.get(int(sid))
        if scene is None or scene.get("excluded"):
            continue
        start, end = float(summary["starts"][i]), float(summary["ends"][i])
        a, b = start, end
        if time_range:
            a, b = max(start, time_range[0]), min(end, time_range[1])
            if b <= a:
                continue
        in_range_sec += b - a
        if wanted and not (wanted & {t.lower() for t in scene.get("tags", [])}):
            continue
        r = float(summary["rates"][i])
        vals = summary["values"][summary["offsets"][i]:summary["offsets"][i + 1]].astype(float)
        vals = vals[int((a - start) * r):max(int((b - start) * r), int((a - start) * r) + 1)]
        if vals.size == 0:
            continue
        segments.append(vals)
        scene_lens.append(b - a)
        usable_sec += b - a
        if summary["has_curve"][i]:
            curve_scenes += 1
            curve_segments.append(vals)
    if not segments:
        return None
    allv = np.concatenate(segments)
    minutes = usable_sec / 60.0
    # Peaks and tempo need real frame-by-frame curves — flat stand-ins would only add noise.
    peaks = sum(_peaks_per_min(seg, rate, 1.0) for seg in curve_segments) / max(minutes, 1e-6)
    bpm, clarity = _tempo(curve_segments, rate)
    return {
        "usable_sec": usable_sec, "in_range_sec": in_range_sec, "scene_count": len(segments),
        "level": float(allv.mean()), "dynamics": float(allv.std()),
        "punch": float(peaks), "cut_rate": 60.0 / max(float(np.median(scene_lens)), 0.1),
        "hist": _hist(allv), "arc": _resample(allv, ARC_POINTS),
        "tempo_bpm": float(bpm) if bpm else None, "tempo_clarity": clarity,
        "curve_coverage": curve_scenes / len(segments),
    }


def track_features(track: dict) -> dict:
    hires = track.get("hires") or {}
    if hires.get("rms"):
        rms = np.asarray(hires["rms"], dtype=float)
        onset = np.asarray(hires.get("onset", []), dtype=float)
    else:
        rms = np.asarray([e["energy"] for e in track.get("energy_envelope", [])], dtype=float)
        onset = np.array([])
    duration = float(track.get("duration_sec") or 0.0)
    minutes = max(duration / 60.0, 0.1)
    hits = float((onset >= 0.8).sum()) / minutes if onset.size else 0.0
    bpm = float(track["bpm"]) if track.get("bpm") else None
    return {
        "duration": duration, "dynamics": float(rms.std()) if rms.size else 0.0, "punch": hits,
        "bpm": bpm, "pace": bpm if bpm else hits,
        "hist": _hist(rms), "arc": _resample(rms, ARC_POINTS),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def recent_project_usage(projects_dir: Path) -> dict:
    """{video_id: how many of the most recent saved projects selected it}."""
    usage = {}
    try:
        files = sorted(Path(projects_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return usage
    for p in files[:RECENT_PROJECTS]:
        try:
            data = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        for vid in set(data.get("committed_selected_videos") or []):
            usage[vid] = usage.get(vid, 0) + 1
    return usage


def score_videos(track: dict, all_tracks: dict, video_feats: dict, reference_feats: dict,
                 weights: dict, tag_filter: tuple = (), usage: dict = None,
                 experimental: bool = False) -> list:
    """video_feats: {video_id: video_features(...) under the current filter and ranges}
    reference_feats: the same for the WHOLE library without the tag filter — the
        fixed population each video is compared against.
    Returns one dict per video, best match first: video_id, score (0–100),
    parts {metric: 0..1}, and a few raw numbers for display."""
    usage = usage or {}
    t = track_features(track)
    tracks = [track_features(x) for x in all_tracks.values()] or [t]
    ref = [f for f in reference_feats.values() if f] or [f for f in video_feats.values() if f]

    t_pct = {
        "dynamics": _percentile_of(t["dynamics"], [x["dynamics"] for x in tracks]),
        "punch": _percentile_of(t["punch"], [x["punch"] for x in tracks]),
        "pace": _percentile_of(t["pace"], [x["pace"] for x in tracks]),
    }
    ref_pop = {k: [f[k] for f in ref] for k in ("dynamics", "punch", "cut_rate", "level")}
    need = max(t["duration"] * FOOTAGE_SHARE, 1.0)

    active = {m: float(weights.get(m, 0.0)) for m in METRICS}
    if not experimental:
        for m in EXPERIMENTAL:
            active[m] = 0.0
    if not tag_filter:
        active["relevance"] = 0.0

    results = []
    for vid, f in video_feats.items():
        if not f:
            continue
        parts = {
            "dynamics": _closeness(_percentile_of(f["dynamics"], ref_pop["dynamics"]), t_pct["dynamics"]),
            "punch": _closeness(_percentile_of(f["punch"], ref_pop["punch"]), t_pct["punch"]),
            "footage": min(1.0, f["usable_sec"] / need),
            "pace": _closeness(_percentile_of(f["cut_rate"], ref_pop["cut_rate"]), t_pct["pace"]),
            "spread": 1.0 - 0.5 * float(np.abs(f["hist"] - t["hist"]).sum()),
            "relevance": (f["usable_sec"] / f["in_range_sec"]) if f["in_range_sec"] > 0 else 0.0,
            "freshness": 1.0 / (1.0 + usage.get(vid, 0)),
        }
        if t["bpm"] and f["tempo_bpm"]:
            err = min(abs(math.log2(f["tempo_bpm"] * k / t["bpm"])) for k in (0.5, 1.0, 2.0))
            parts["tempo"] = max(0.0, 1.0 - err / 0.15) * f["tempo_clarity"]
        else:
            parts["tempo"] = 0.0
        va, ta = f["arc"], t["arc"]
        parts["arc"] = (0.5 if va.std() < 1e-9 or ta.std() < 1e-9
                        else (float(np.corrcoef(va, ta)[0, 1]) + 1.0) / 2.0)

        total_w = sum(active.values())
        score = (sum(active[m] * parts[m] for m in METRICS) / total_w) if total_w > 0 else 0.0
        results.append({
            "video_id": vid,
            "score": round(100 * score, 1),
            "parts": {m: round(parts[m], 2) for m in METRICS},
            "usable_sec": f["usable_sec"],
            "matching_scene_count": f["scene_count"],
            "level_pct": round(_percentile_of(f["level"], ref_pop["level"]), 3),
            "tempo_bpm": round(f["tempo_bpm"], 1) if f["tempo_bpm"] else None,
            "curve_coverage": f["curve_coverage"],
        })
    results.sort(key=lambda r: (-r["score"], r["video_id"]))
    return results
