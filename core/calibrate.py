"""📏 Calibrated match % — "97 %" means better than 97 % of the plans the app has scored.

The raw scores (core/recommend.score_timeline) are averages of correlations and closeness × 100, so
their spread differs per metric and per track: a raw 80 % shape can be great while a raw 80 % motion
is middling. Here every metric is turned into its rank among every plan the app has scored (the
reference: each recommendation search's checked plans, and your scored renders):

    ranked % = share of reference plans this one beats (ties count half)

Overall is then ranked too, from a weighted mix of the ranked parts. Weights start equal (⅓ shape,
⅓ motion, ⅓ story — ½ + ½ without story) and are fine-tuned by your ⭐ ratings once there are
MIN_RATED rated renders with scores: a part that goes with your higher ratings gets more say. The
fitted weights are blended with equal ones (PRIOR plans' worth), so a few ratings only nudge them.

Pure: no Streamlit, no files.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

METRICS = ("overall", "shape", "intensity", "story", "action", "action_story")
PARTS = ("shape", "intensity", "story")
MIN_REFERENCE = 30     # fewer reference plans than this → show raw %
MIN_RATED = 8          # rated renders with scores needed before ratings change the weights
PRIOR = 20             # how many ratings' worth of "equal weights" the fit is blended with
MAX_REF = 4000         # reference values kept per metric (evenly thinned beyond)


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _thin(values: list) -> list:
    values = sorted(values)
    if len(values) <= MAX_REF:
        return values
    step = (len(values) - 1) / (MAX_REF - 1)
    return [values[round(i * step)] for i in range(MAX_REF)]


def rank(ref: list, x: float) -> float:
    """0–100: share of ref below x, ties counting half."""
    if not ref:
        return 50.0
    lo, hi = bisect_left(ref, x), bisect_right(ref, x)
    return 100.0 * (lo + 0.5 * (hi - lo)) / len(ref)


def _spearman(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else 0.0


def build(reference: list, rated: list = ()) -> dict:
    """reference: raw score dicts (score_timeline's) of every plan scored; rated: [(raw scores, ⭐ 1–5)].
    → the calibration: {"n", "ref": {metric: sorted values}, "weights", "n_rated", "fit": {part: ρ}, "combo"}."""
    ref = {m: _thin([v for s in reference if (v := _num((s or {}).get(m))) is not None]) for m in METRICS}
    cal = {"n": len(reference), "ref": ref, "weights": {p: 1.0 / 3 for p in PARTS}, "n_rated": 0, "fit": {}}
    rated = [(s, int(r)) for s, r in rated or () if s and r]
    cal["n_rated"] = len(rated)
    if len(rated) >= MIN_RATED:
        fit = {}
        for p in PARTS:
            pairs = [(rank(ref[p], v), r) for s, r in rated if (v := _num(s.get(p))) is not None]
            fit[p] = _spearman([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= MIN_RATED else 0.0
        raw = {p: max(0.05, fit[p]) for p in PARTS}
        tot = sum(raw.values())
        n = len(rated)
        cal["weights"] = {p: (n * raw[p] / tot + PRIOR / 3) / (n + PRIOR) for p in PARTS}
        cal["fit"] = {p: round(fit[p], 2) for p in PARTS}
    cal["combo"] = _thin([c for s in reference if (c := _combo(cal, s)) is not None])
    return cal


def _combo(cal: dict, s: dict):
    """Weighted mix of the ranked parts (story left out when the plan has none)."""
    tot = acc = 0.0
    for p in PARTS:
        v = _num((s or {}).get(p))
        if v is None:
            continue
        w = cal["weights"][p]
        acc += w * rank(cal["ref"][p], v)
        tot += w
    return acc / tot if tot else None


def ready(cal: dict | None) -> bool:
    return bool(cal) and cal.get("n", 0) >= MIN_REFERENCE


def ranked(scores: dict, cal: dict | None) -> dict:
    """The same score dict with each metric as its ranked % (raw ones kept under "raw").
    Unchanged when there isn't enough reference yet."""
    if not scores or not ready(cal):
        return scores
    out = dict(scores)
    out["raw"] = {m: scores.get(m) for m in METRICS}
    for m in METRICS:
        v = _num(scores.get(m))
        if v is None:
            continue
        if m == "overall":
            c = _combo(cal, scores)
            out[m] = None if c is None else round(rank(cal["combo"], c), 1)
        else:
            out[m] = round(rank(cal["ref"][m], v), 1)
    out["ranked"] = True
    return out


def describe(cal: dict | None) -> str:
    if not ready(cal):
        n = (cal or {}).get("n", 0)
        return (f"Raw scores — ranked % starts once {MIN_REFERENCE} plans have been scored "
                f"({n} so far; each search adds about 14).")
    w = cal["weights"]
    s = (f"% = better than that share of the {cal['n']:,} plans scored so far. Overall weights: "
         f"shape {w['shape']:.0%} · motion {w['intensity']:.0%} · story {w['story']:.0%}")
    if cal["n_rated"] >= MIN_RATED:
        s += f" — tuned by {cal['n_rated']} rated renders."
    else:
        s += f" (equal until {MIN_RATED} rated renders have scores — {cal['n_rated']} now)."
    return s
