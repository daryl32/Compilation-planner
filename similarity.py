"""
"More like this" — how alike two clips (or two videos) are, from tags and
movement only. No Streamlit and no AI model; the planner page passes data in.

Clip similarity (0–1) =
    0.40 · tags      shared tags, rarer tags counting more (IDF-weighted Jaccard)
  + 0.35 · movement  average level, how much it varies, sharp peaks per minute
  + 0.25 · shape     the frame-by-frame motion curves rise and fall together

Video similarity (0–1) =
    0.50 · tags      how much of each video's usable footage carries each tag
                     (IDF-weighted cosine of those shares)
  + 0.50 · movement  level, dynamics and punch, compared as positions within the
                     whole library (same idea as the video ranking)

Every result also gets a short plain-English reason.
"""

import math

import numpy as np

SHAPE_POINTS = 40
CLIP_WEIGHTS = {"tags": 0.40, "movement": 0.35, "shape": 0.25}
VIDEO_WEIGHTS = {"tags": 0.50, "movement": 0.50}


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def tag_idf(catalogues: dict, tag_fn) -> dict:
    """{tag: weight} — tags on fewer scenes across the library weigh more.
    tag_fn(scene) -> list of tags (the planner passes corrected tags)."""
    df, n = {}, 0
    for cat in catalogues.values():
        for scene in cat.get("scenes", []):
            n += 1
            for t in {t.lower() for t in tag_fn(scene)}:
                df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def _tag_jaccard(a: set, b: set, idf: dict):
    """(similarity, shared tags rarest-first). Neutral 0.5 when neither has tags."""
    if not a and not b:
        return 0.5, []
    union = a | b
    inter = a & b
    w = lambda ts: sum(idf.get(t, 1.0) for t in ts)
    shared = sorted(inter, key=lambda t: -idf.get(t, 1.0))
    return (w(inter) / w(union) if union else 0.0), shared


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def _resample(values: np.ndarray, n: int) -> np.ndarray:
    if values.size < 2:
        return np.full(n, float(values[0]) if values.size else 0.0)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, values.size), values)


def _peaks_per_min(values: np.ndarray, fps: float) -> float:
    if values.size < 3 or fps <= 0:
        return 0.0
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826 or values.std() or 1.0
    z = (values - med) / mad
    peaks = int(((z[1:-1] > 2.5) & (z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:])).sum())
    return peaks / max(values.size / fps / 60.0, 1e-6)


def clip_descriptor(values, fps: float, tags) -> dict:
    """Fingerprint of one clip from its slice of the motion curve."""
    v = np.asarray(values, dtype=float)
    shape = _resample(v, SHAPE_POINTS)
    sd = shape.std()
    return {
        "tags": {t.lower() for t in (tags or [])},
        "level": float(v.mean()) if v.size else 0.0,
        "var": float(v.std()) if v.size else 0.0,
        "punch": _peaks_per_min(v, fps),
        "shape": (shape - shape.mean()) / sd if sd > 1e-9 else None,
    }


def _closeness(a: float, b: float, scale: float) -> float:
    return math.exp(-abs(a - b) / max(scale, 1e-9))


def rank_similar_clips(ref: dict, pool: list, idf: dict, top: int = 6) -> list:
    """pool: [(descriptor, payload)]. Returns [(similarity 0–1, reason, payload)],
    most similar first. Movement differences are judged against the spread
    across ref + pool, so 'close' means close for this set of footage."""
    if not pool:
        return []
    descs = [ref] + [d for d, _ in pool]
    scale = {k: (np.std([d[k] for d in descs]) or abs(ref[k]) or 1.0) for k in ("level", "var", "punch")}
    out = []
    for d, payload in pool:
        tag_sim, shared = _tag_jaccard(ref["tags"], d["tags"], idf)
        movement = float(np.mean([_closeness(ref[k], d[k], scale[k]) for k in ("level", "var", "punch")]))
        if ref["shape"] is not None and d["shape"] is not None:
            shape = (float(np.dot(ref["shape"], d["shape"]) / SHAPE_POINTS) + 1.0) / 2.0
        else:
            shape = 0.5
        sim = (CLIP_WEIGHTS["tags"] * tag_sim + CLIP_WEIGHTS["movement"] * movement
               + CLIP_WEIGHTS["shape"] * shape)
        out.append((sim, _reason(shared, movement, shape), payload))
    out.sort(key=lambda r: -r[0])
    return out[:top]


def _reason(shared: list, movement: float, shape: float = None) -> str:
    bits = []
    if shared:
        bits.append("shares " + ", ".join(shared[:2]) + (f" +{len(shared) - 2}" if len(shared) > 2 else ""))
    if movement >= 0.7:
        bits.append("similar movement")
    if shape is not None and shape >= 0.75:
        bits.append("similar motion shape")
    return " · ".join(bits) or "closest available"


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

def _percentile(value: float, population: list) -> float:
    pop = np.asarray(population, dtype=float)
    if pop.size == 0:
        return 0.5
    return (float((pop < value).sum()) + 0.5 * float((pop == value).sum())) / pop.size


def rank_similar_videos(ref_id: str, videos: dict, idf: dict, top: int = 5) -> list:
    """videos: {video_id: {"tag_seconds": {tag: sec}, "level", "dynamics", "punch"}}
    — usable footage only. Returns [(similarity 0–1, reason, video_id)]."""
    ref = videos.get(ref_id)
    if not ref:
        return []
    pops = {k: [v[k] for v in videos.values()] for k in ("level", "dynamics", "punch")}
    pct = {vid: {k: _percentile(v[k], pops[k]) for k in pops} for vid, v in videos.items()}

    def vec(v):
        total = sum(v["tag_seconds"].values()) or 1.0
        return {t.lower(): (s / total) * idf.get(t.lower(), 1.0) for t, s in v["tag_seconds"].items()}

    rv = vec(ref)
    rn = math.sqrt(sum(x * x for x in rv.values()))
    out = []
    for vid, v in videos.items():
        if vid == ref_id:
            continue
        ov = vec(v)
        on = math.sqrt(sum(x * x for x in ov.values()))
        dot = sum(rv[t] * ov.get(t, 0.0) for t in rv)
        tag_sim = dot / (rn * on) if rn and on else (0.5 if not rv and not ov else 0.0)
        movement = float(np.mean([1.0 - abs(pct[ref_id][k] - pct[vid][k]) for k in pops]))
        sim = VIDEO_WEIGHTS["tags"] * tag_sim + VIDEO_WEIGHTS["movement"] * movement
        shared = sorted((t for t in rv if t in ov), key=lambda t: -(rv[t] * ov[t]))
        out.append((sim, _reason(shared, movement), vid))
    out.sort(key=lambda r: -r[0])
    return out[:top]
