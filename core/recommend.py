"""🔮 Recommendations — "For this track": the best sets of videos and cut settings
for a track, found by searching the whole library and checked by actually
matching them.

How it works
  1. Fingerprint every video (once, cached by the page): every window of every
     scene at a ladder of lengths (0.75 s … 20 s), each one's motion SHAPE
     (32 points, z-scored) and motion LEVEL (0–1 for the video).
  2. For each cut-settings variant (Smart, calmer / normal / busier) cut the track
     into blocks and fingerprint each block the same way from the music energy.
  3. Score every video against every block (best window of the nearest length):
     shape = correlation, level = 1 − |motion level − block intensity|.
  4. Pick the video set greedily — each step adds the video that improves the
     plan most — and a few alternatives that avoid the earlier plans' videos.
  5. Run the REAL matcher (core.matching, Auto mode) on each candidate and score
     what it actually picked, so the % shown is what you get when you open it.

Scoring (decided with Daryl, Oct 2026)
  • Shape %      — every block counts equally.
  • Intensity %  — drops / action parts count DROP_WEIGHT× (the rest 1×).
  • Action %     — intensity match over the drop / action blocks only.
  • Overall %    — ½ shape + ½ weighted intensity.
  • Weakest block — the lowest-scoring block, so a high average can't hide it.

Pure functions, no Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.segmentation import SEG_DEFAULTS, SEG_METHOD_KEYS, SEG_SMART, apply_segment_intensity, build_segments
from core.shape import get_block_audio_curve, resample_curve, shape_score

POINTS = 24
LENGTHS = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
DROP_WEIGHT = 2.5          # intensity weight of drop / action blocks
DROP_BARS_AFTER = 4        # a drop "block" is any block overlapping [drop, drop + 4 bars]
VARIANTS = [               # (label, Smart settings on top of the defaults)
    ("Smart · calmer", {"sm_pace": -1}),
    ("Smart · normal", {"sm_pace": 0}),
    ("Smart · busier", {"sm_pace": 1}),
]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _znorm_rows(m: np.ndarray) -> np.ndarray:
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, keepdims=True)
    out = np.zeros_like(m)
    ok = sd[:, 0] > 1e-9
    out[ok] = (m[ok] - mu[ok]) / sd[ok]
    return out


def _resample_rows(windows: np.ndarray) -> np.ndarray:
    n = windows.shape[1]
    if n < 2:
        return np.repeat(windows[:, :1], POINTS, axis=1)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, POINTS)
    idx = np.clip(np.searchsorted(x_old, x_new, side="right") - 1, 0, n - 2)
    frac = (x_new - x_old[idx]) / (x_old[idx + 1] - x_old[idx])
    return windows[:, idx] * (1 - frac) + windows[:, idx + 1] * frac


@dataclass
class VideoPrint:
    video_id: str
    by_len: dict = field(default_factory=dict)   # length → (Z [n, POINTS] float16, level [n] float16) — compact
    scale: float = 1.0                           # motion value that counts as level 1.0


def fingerprint_video(video_id: str, catalogue: dict, curves: dict, time_range=None) -> VideoPrint | None:
    """Windows of every usable scene (not excluded; inside the video's library range).
    catalogue: the video's catalogue (scenes with start_tc / end_tc); curves: {scene_id: {"fps", "values"}}."""
    from core.footage import _tc_to_seconds
    vp = VideoPrint(video_id)
    pieces = []          # (values array, fps)
    for s in catalogue.get("scenes", []):
        if s.get("excluded"):
            continue
        c = curves.get(s["scene_id"])
        if not c or c.get("values") is None or len(c["values"]) < 2:
            continue
        vals = np.asarray(c["values"], dtype=float)
        fps = float(c["fps"])
        if time_range:
            t0 = _tc_to_seconds(s["start_tc"])
            a = max(0, int(round((time_range[0] - t0) * fps)))
            b = min(len(vals), int(round((time_range[1] - t0) * fps)))
            vals = vals[a:b]
            if len(vals) < 2:
                continue
        pieces.append((vals, fps))
    if not pieces:
        return None
    allv = np.concatenate([p[0] for p in pieces])
    vp.scale = float(np.percentile(allv, 95)) or float(allv.max()) or 1.0
    for L in LENGTHS:
        zs, lv = [], []
        for vals, fps in pieces:
            wf = max(2, int(round(L * fps)))
            if len(vals) < wf:
                continue
            step = max(1, int(round(max(0.5, L / 4) * fps)))
            win = np.lib.stride_tricks.sliding_window_view(vals, wf)[::step]
            zs.append(_znorm_rows(_resample_rows(win)).astype(np.float16))
            lv.append(np.clip(win.mean(axis=1) / vp.scale, 0, 1).astype(np.float16))
        if zs:
            vp.by_len[L] = (np.concatenate(zs), np.concatenate(lv))
    return vp if vp.by_len else None


def _nearest_len(d: float, available) -> float | None:
    fits = [L for L in available if L <= d * 1.15]
    return max(fits) if fits else None


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def block_weights(track: dict, segments: list, story_regions: list = None, beats_per_bar: int = 4) -> list:
    """Intensity weight per block: DROP_WEIGHT for drop / action blocks, else 1.
    Drop / action = overlaps [drop, drop + 4 bars] (from the analysis), or lies in a story-map
    'action' part. With neither available: the most intense quarter of the blocks."""
    import cut_scoring as CS
    bar = CS.bar_sec(track, beats_per_bar)
    a = track.get("analysis") or {}
    spans = [(float(t), float(t) + DROP_BARS_AFTER * bar) for t in (a.get("drops") or [])]
    spans += [(float(r["start"]), float(r["end"])) for r in (story_regions or []) if r.get("role") == "action"]
    if spans:
        flags = [any(sg["start"] < e and sg["end"] > s for s, e in spans) for sg in segments]
    else:
        ints = np.array([sg.get("intensity", 0.0) for sg in segments])
        cut = np.quantile(ints, 0.75) if len(ints) else 1.0
        flags = [i >= cut and i > 0 for i in ints]
    return [DROP_WEIGHT if f else 1.0 for f in flags]


def cut_variant(track: dict, settings: dict, beats_per_bar: int = 4) -> list:
    cfg = {"method": SEG_SMART, "beats_per_bar": beats_per_bar,
           **{k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_SMART]}, **settings}
    segs = build_segments(track, cfg)
    apply_segment_intensity(segs, track, 1.0)
    return segs


def block_prints(track: dict, segments: list) -> list:
    out = []
    for sg in segments:
        curve = get_block_audio_curve(track, sg["start"], sg["end"])
        z = resample_curve(curve, POINTS)
        sd = z.std()
        out.append(((z - z.mean()) / sd if sd > 1e-9 else np.zeros(POINTS)).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def score_matrix(prints: dict, segments: list, bprints: list) -> tuple:
    """(video_ids, S [videos, blocks] best window score 0–1, SHAPE, LEVEL) — each block scored
    by the best window of each video: ½ shape (correlation ≥ 0) + ½ (1 − |level − intensity|)."""
    vids = sorted(prints)
    nb = len(segments)
    S = np.zeros((len(vids), nb), dtype=np.float32)
    SH = np.zeros_like(S)
    LV = np.zeros_like(S)
    for vi, vid in enumerate(vids):
        vp = prints[vid]
        for bi, sg in enumerate(segments):
            L = _nearest_len(sg["end"] - sg["start"], vp.by_len)
            if L is None:
                continue
            Z, lev = vp.by_len[L]
            Z, lev = Z.astype(np.float32), lev.astype(np.float32)
            corr = np.clip(Z @ bprints[bi] / POINTS, 0, 1)
            inten = 1.0 - np.abs(lev - float(sg.get("intensity", 0.5)))
            comb = 0.5 * corr + 0.5 * inten
            k = int(np.argmax(comb))
            S[vi, bi], SH[vi, bi], LV[vi, bi] = comb[k], corr[k], inten[k]
    return vids, S, SH, LV


def choose_sets(vids: list, S: np.ndarray, weights: list, n_videos: int, n_plans: int = 3) -> list:
    """Greedy video sets: each step adds the video that raises the plan's weighted block cover
    most (no video twice in a row is the matcher's job). Later plans halve the gain of videos
    already used, so they're genuinely different."""
    w = np.asarray(weights, dtype=np.float32)
    used = np.zeros(len(vids))
    plans = []
    for _ in range(n_plans):
        chosen, cover = [], np.zeros(S.shape[1], dtype=np.float32)
        for _ in range(min(n_videos, len(vids))):
            gains = (np.maximum(S, cover) - cover) @ w
            gains = gains * (0.5 ** used)
            gains[chosen] = -1
            k = int(np.argmax(gains))
            if gains[k] <= 0:
                break
            chosen.append(k)
            cover = np.maximum(cover, S[k])
        if not chosen:
            break
        for k in chosen:
            used[k] += 1
        ids = tuple(sorted(vids[k] for k in chosen))
        if ids not in [p["videos"] for p in plans]:
            plans.append({"videos": ids, "estimate": float((cover @ w) / max(w.sum(), 1e-9))})
    return plans


# ---------------------------------------------------------------------------
# Checking a plan with the real matcher
# ---------------------------------------------------------------------------

def clip_level(pick: dict, prints_scale: dict, curves_of) -> float | None:
    c = curves_of(pick["video_id"]).get(pick["scene_id"])
    if not c or c.get("values") is None or len(c["values"]) == 0:
        return None
    fps = float(c["fps"])
    a = int(round(pick["offset_into_scene_sec"] * fps))
    b = a + max(1, int(round(pick["clip_duration_sec"] * fps)))
    seg = np.asarray(c["values"][a:b], dtype=float)
    if not len(seg):
        return None
    return float(np.clip(seg.mean() / prints_scale.get(pick["video_id"], 1.0), 0, 1))


def score_timeline(track: dict, segments: list, timeline: list, weights: list, prints_scale: dict, curves_of) -> dict:
    """Scores of what the matcher actually picked (first clip of each block)."""
    shape, inten, comb = [], [], []
    for sg, entry in zip(segments, timeline):
        slots = entry.get("scenes") or []
        if not slots:
            shape.append(0.0)
            inten.append(0.0)
            comb.append(0.0)
            continue
        p = slots[0]["chain"][0]
        c = curves_of(p["video_id"]).get(p["scene_id"])
        sh = 0.0
        if c and c.get("values") is not None and len(c["values"]):
            fps = float(c["fps"])
            a = int(round(p["offset_into_scene_sec"] * fps))
            b = a + max(2, int(round(p["clip_duration_sec"] * fps)))
            clip = np.asarray(c["values"][a:b], dtype=float)
            if len(clip) >= 2:
                sh = max(0.0, shape_score(clip, get_block_audio_curve(track, sg["start"], sg["end"])))
        lvl = clip_level(p, prints_scale, curves_of)
        it = 0.0 if lvl is None else 1.0 - abs(lvl - float(sg.get("intensity", 0.5)))
        shape.append(sh)
        inten.append(it)
        comb.append(0.5 * sh + 0.5 * it)
    w = np.asarray(weights, dtype=float)
    shape_pct = 100 * float(np.mean(shape)) if shape else 0.0
    inten_pct = 100 * float(np.average(inten, weights=w)) if inten else 0.0
    act = [x for x, wi in zip(inten, weights) if wi > 1.0]
    weakest = int(np.argmin(comb)) if comb else None
    return {"overall": round(0.5 * shape_pct + 0.5 * inten_pct, 1), "shape": round(shape_pct, 1),
            "intensity": round(inten_pct, 1), "action": round(100 * float(np.mean(act)), 1) if act else None,
            "weakest_block": weakest, "weakest": round(100 * comb[weakest], 1) if comb else None,
            "per_block": [round(100 * x, 1) for x in comb], "action_blocks": [i for i, x in enumerate(weights) if x > 1.0]}


def recommend_for_track(track: dict, prints: dict, *, match, curves_of, story_regions: list = None,
                        n_videos: int = 8, plans_per_variant: int = 4, top: int = 5, swap_budget: int = 12,
                        progress=None) -> list:
    """The best plans for `track`, best first.

    prints: {video_id: VideoPrint} (fingerprint_video). match(segments, video_ids) runs the real
    matcher and returns its timeline. curves_of(video_id) → {scene_id: {"fps", "values"}}.
    progress(fraction, text) is called as it goes.

    Each plan: {"variant", "settings", "videos", "estimate", "scores", "timeline", "segments"}."""
    progress = progress or (lambda f, t: None)
    scale = {v: p.scale for v, p in prints.items()}
    candidates = []
    for vi, (label, settings) in enumerate(VARIANTS):
        progress(vi / (len(VARIANTS) + 1), f"Cutting the track ({label})…")
        segs = cut_variant(track, settings)
        weights = block_weights(track, segs, story_regions)
        vids, S, _, _ = score_matrix(prints, segs, block_prints(track, segs))
        for plan in choose_sets(vids, S, weights, n_videos, plans_per_variant):
            candidates.append({"variant": label, "settings": settings, "segments": segs, "weights": weights, **plan})
    candidates.sort(key=lambda c: -c["estimate"])
    checked = {}

    def check(cand):
        key = (cand["variant"], tuple(sorted(cand["videos"])))
        if key not in checked:
            timeline = match(cand["segments"], list(key[1]))
            scores = score_timeline(track, cand["segments"], timeline, cand["weights"], scale, curves_of)
            checked[key] = {**cand, "videos": list(key[1]), "timeline": timeline, "scores": scores}
        return checked[key]

    total = len(candidates) + swap_budget
    for i, cand in enumerate(candidates):
        progress((len(VARIANTS) + i / total) / (len(VARIANTS) + 1), f"Checking plan {i + 1} with the real matcher…")
        check(cand)

    # Improve the best plan: swap one video at a time for a promising unused one, keeping a swap
    # only when the real matcher scores it higher (the estimate can't see footage running out,
    # the never-repeat rule or the matcher's own weights).
    done = len(candidates)
    best = max(checked.values(), key=lambda p: p["scores"]["overall"])
    pool = [v for v in _promising(best, prints, track) if v not in best["videos"]]
    improved = True
    while improved and done < total and pool:
        improved = False
        for out_v in list(best["videos"]):
            for in_v in pool[:3]:
                if done >= total:
                    break
                progress((len(VARIANTS) + done / total) / (len(VARIANTS) + 1),
                         f"Fine-tuning the best plan ({done - len(candidates) + 1}/{swap_budget})…")
                trial = check({**best, "videos": [v for v in best["videos"] if v != out_v] + [in_v],
                               "estimate": best["estimate"]})
                done += 1
                if trial["scores"]["overall"] > best["scores"]["overall"] + 0.05:
                    best, improved = trial, True
                    pool = [v for v in pool if v != in_v] + [out_v]
                    break
            if improved or done >= total:
                break

    out = sorted(checked.values(), key=lambda p: -p["scores"]["overall"])
    progress(1.0, "Done")
    return out[:top]


def _promising(plan: dict, prints: dict, track: dict) -> list:
    """Videos ranked by how well they cover this plan's blocks on their own (best first)."""
    vids, S, _, _ = score_matrix(prints, plan["segments"], block_prints(track, plan["segments"]))
    w = np.asarray(plan["weights"], dtype=np.float32)
    order = np.argsort(-(S @ w))
    return [vids[k] for k in order]


def library_signature(prints_meta: dict, track_meta) -> str:
    """A key for caching results: changes when any catalogue or the track changes."""
    import hashlib
    import json
    return hashlib.sha1(json.dumps([sorted(prints_meta.items()), track_meta], default=str).encode()).hexdigest()[:16]


__all__ = ["VideoPrint", "fingerprint_video", "block_weights", "cut_variant", "block_prints", "score_matrix",
           "choose_sets", "score_timeline", "recommend_for_track", "library_signature", "VARIANTS", "DROP_WEIGHT"]



def print_size_mb(prints: dict) -> float:
    """Memory held by fingerprints, in MB (for the page to report)."""
    return sum(z.nbytes + lv.nbytes for p in prints.values() for z, lv in p.by_len.values()) / 2**20
