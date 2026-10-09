"""
Render analytics — what your saved renders and their ratings say about the
planner settings behind them. Read-only and theoretical: nothing here feeds
back into the planner (yet).

build_cards(renders) turns the render details (renders.list_renders()) into a
list of cards, one per area of settings. Each card has:
    key, title, subtitle
    chart: {"kind": "bars" | "scatter" | "none", ...data...}
    insight: one or two sentences on what the data shows
    recommendation: what to try next (or what's needed before there's a signal)
    table: rows for a plain data view

The signal is your star rating (1–5). Unrated renders count towards usage but
not towards "what works". Comparisons need at least MIN_GROUP rated renders on
each side before they say anything; otherwise the card says what's missing.

No Streamlit in here — pages/3_Media_Library.py draws the cards.
"""

import datetime
import statistics
from collections import defaultdict

MIN_GROUP = 3          # rated renders needed in a group before it's compared
MIN_SCATTER = 6        # rated renders needed before a trend is described
GOOD = 4               # a rating of this or more counts as a "good" render
CLEAR_GAP = 0.5        # stars between groups before calling a difference


# ---------------------------------------------------------------------------
# Rows: one flat record per render
# ---------------------------------------------------------------------------

def _mean_clip_len(plan: dict):
    lens = [float(l.get("clip_duration_sec") or 0)
            for e in (plan or {}).get("timeline") or []
            for slot in e.get("scenes") or []
            for l in slot.get("chain") or []]
    lens = [x for x in lens if x > 0]
    return statistics.mean(lens) if lens else None


def to_row(r: dict) -> dict:
    s = r.get("settings") or {}
    seg = s.get("segmentation") or {}
    dur = float(r.get("duration_sec") or 0)
    segments = r.get("segments")
    plan = r.get("plan") or {}
    return {
        "name": r.get("render_file"),
        "created": r.get("created"),
        "rating": int((r.get("user") or {}).get("rating") or 0),
        "keep": bool((r.get("user") or {}).get("keep")),
        "track": (r.get("track") or {}).get("track_id"),
        "bpm": (r.get("track") or {}).get("bpm"),
        "duration": dur or None,
        "known": not r.get("backfilled") and segments is not None,
        "mode": s.get("matching_mode"),
        "seg_method": seg.get("method"),
        "cuts_per_min": (segments / (dur / 60)) if segments and dur else None,
        "split_on": s.get("split_screen_enabled"),
        "max_clips": s.get("max_clips") if s.get("split_screen_enabled") else 1 if s else None,
        "split_share": (r.get("split_segments") or 0) / segments if segments else None,
        "avg_clip_len": _mean_clip_len(plan),
        "min_clip_len": s.get("min_clip_len_sec"),
        "videos_n": len(r.get("videos") or []) if not r.get("backfilled") else None,
        "videos": [v["video_id"] for v in r.get("videos") or []],
        "allow_same_video": s.get("allow_same_video"),
        "sequential": s.get("sequential_video_order"),
        "tagged": bool(s.get("tag_filter")) if s else None,
        "render_seconds": r.get("render_seconds"),
        "streamed": (r.get("sources") or {}).get("streamed"),
    }


def filter_rows(rows: list, days: int = None) -> list:
    if not days:
        return rows
    now = datetime.datetime.now(datetime.timezone.utc)
    out = []
    for row in rows:
        try:
            when = datetime.datetime.fromisoformat(row["created"])
            if when.tzinfo is None:
                when = when.replace(tzinfo=datetime.timezone.utc)
        except (TypeError, ValueError):
            continue
        if (now - when).days <= days:
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _fmt(v, unit=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        v = f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"
    return f"{v}{unit}"


def group_stats(rows: list, key: str, label=None) -> list:
    """[{group, n, rated, mean}] — mean star rating of the rated renders per value of key."""
    label = label or (lambda v: str(v))
    groups = defaultdict(list)
    for row in rows:
        v = row.get(key)
        if v is None:
            continue
        groups[label(v)].append(row)
    out = []
    for g, items in groups.items():
        ratings = [x["rating"] for x in items if x["rating"]]
        out.append({"group": g, "n": len(items), "rated": len(ratings),
                    "mean": round(statistics.mean(ratings), 2) if ratings else None})
    return sorted(out, key=lambda x: (-(x["mean"] or 0), -x["n"], x["group"]))


def compare_groups(stats: list, what: str) -> tuple:
    """(insight, recommendation) from group_stats output."""
    ready = [g for g in stats if g["rated"] >= MIN_GROUP]
    if not stats:
        return (f"No renders record their {what} yet.",
                "New renders save their settings automatically — this fills in as you render.")
    if len(ready) < 2:
        most = max(stats, key=lambda g: g["n"])
        need = [g["group"] for g in stats if g["rated"] < MIN_GROUP]
        return (f"Mostly **{most['group']}** so far ({most['n']} of {sum(g['n'] for g in stats)} renders). "
                f"Not enough rated renders in each {what} to compare yet.",
                f"Rate at least {MIN_GROUP} renders for each {what} you want to compare"
                + (f" — e.g. try a few with **{need[0]}**." if need and need[0] != most["group"] else
                   " — try a different one for a few renders."))
    best, rest = ready[0], ready[1:]
    gap = best["mean"] - statistics.mean(g["mean"] for g in rest)
    if gap >= CLEAR_GAP:
        return (f"**{best['group']}** renders rate highest: {best['mean']:.1f}★ on average "
                f"(from {best['rated']}), {gap:.1f}★ above the others.",
                f"Default to **{best['group']}** and keep an eye on whether it holds up across different tracks.")
    return (f"No clear winner — every {what} averages within {CLEAR_GAP}★ of the others "
            f"(best: **{best['group']}**, {best['mean']:.1f}★).",
            f"The {what} doesn't seem to drive your ratings much; look at the other cards first.")


def _spearman(xs: list, ys: list):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def numeric_insight(rows: list, key: str, what: str, unit: str = "") -> tuple:
    """(insight, recommendation) for a numeric property against the rating."""
    pts = [(row[key], row["rating"]) for row in rows if row.get(key) is not None and row["rating"]]
    if len(pts) < MIN_SCATTER:
        return (f"{len(pts)} rated render(s) with a {what} — need {MIN_SCATTER} to see a trend.",
                f"Keep rating renders; vary the {what} between them so there's a spread to learn from.")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    rho = _spearman(xs, ys)
    good = sorted(x for x, y in pts if y >= GOOD)
    band = ""
    if len(good) >= 2:
        lo, hi = good[len(good) // 4], good[(3 * len(good)) // 4]
        span = _fmt(lo, unit) if lo == hi else f"{_fmt(lo)}–{_fmt(hi, unit)}"
        band = f" Your {GOOD}★+ renders mostly sit at **{span}**."
    if rho is not None and abs(rho) >= 0.4:
        direction = "higher" if rho > 0 else "lower"
        return (f"A {'clear' if abs(rho) >= 0.6 else 'moderate'} trend: {direction} {what} tends to rate "
                f"higher (rank correlation {rho:+.2f}).{band}",
                f"Push the {what} {direction} on your next few renders and see if ratings follow."
                + (f" Aim for the {GOOD}★+ range." if band else ""))
    return (f"No strong link between {what} and rating (rank correlation "
            f"{'n/a' if rho is None else f'{rho:+.2f}'}).{band}",
            (f"Stay within the range your best renders use; the {what} isn't the deciding factor."
             if band else f"The {what} doesn't seem to matter much on its own."))


def scatter(rows: list, key: str, x_title: str) -> dict:
    pts = [{"x": row[key], "y": row["rating"], "name": row["name"]}
           for row in rows if row.get(key) is not None and row["rating"]]
    return {"kind": "scatter", "points": pts, "x_title": x_title}


def bars(stats: list, x_title: str = "Average rating (★)") -> dict:
    return {"kind": "bars", "bars": stats, "x_title": x_title}


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def card_overview(rows: list) -> dict:
    rated = [r for r in rows if r["rating"]]
    dist = [{"group": "★" * k, "n": sum(1 for r in rated if r["rating"] == k), "rated": 0, "mean": None}
            for k in range(1, 6)]
    known = sum(1 for r in rows if r["known"])
    mean = statistics.mean(r["rating"] for r in rated) if rated else None
    kept = sum(1 for r in rows if r["keep"])
    if not rows:
        insight, rec = "No renders yet.", "Render a preview in the Compilation Planner to get started."
    elif len(rated) < MIN_GROUP * 2:
        insight = (f"{len(rows)} renders, {len(rated)} rated"
                   + (f" (average {mean:.1f}★)" if mean else "") + f", {kept} kept. "
                   "Every card below runs on ratings, so there's little to learn yet.")
        rec = f"Rate your renders (ℹ️ on each one) — about {MIN_GROUP * 4} rated renders makes these cards useful."
    else:
        low = sum(1 for r in rated if r["rating"] <= 2)
        insight = (f"{len(rows)} renders, {len(rated)} rated (average {mean:.1f}★), {kept} kept. "
                   f"{known} record their full settings.")
        rec = ("Plenty of low ratings — use the cards below to spot which settings they share."
               if low >= len(rated) / 3 else
               "Ratings skew positive — rate more critically (use the whole 1–5 range) so differences show up.")
    return {
        "key": "overview", "title": "📋 Overview", "subtitle": "How many renders and ratings there are to learn from",
        "chart": {"kind": "bars", "bars": dist, "x_title": "Renders", "count": True},
        "insight": insight, "recommendation": rec,
        "table": [{"Rating": d["group"], "Renders": d["n"]} for d in dist],
    }


def card_matching(rows: list) -> dict:
    st = group_stats(rows, "mode")
    ins, rec = compare_groups(st, "matching mode")
    return {"key": "matching", "title": "🧩 Matching mode", "subtitle": "Auto vs Manual step-through vs Choreography",
            "chart": bars(st), "insight": ins, "recommendation": rec,
            "table": [{"Mode": g["group"], "Renders": g["n"], "Rated": g["rated"], "Avg ★": g["mean"]} for g in st]}


def card_segmentation(rows: list) -> dict:
    st = group_stats(rows, "seg_method")
    ins1, rec1 = compare_groups(st, "segmentation method")
    ins2, rec2 = numeric_insight(rows, "cuts_per_min", "cut pace", " per min")
    return {"key": "segmentation", "title": "✂️ Segmentation & cut pace",
            "subtitle": "Segmentation method, and segments per minute of track",
            "chart": scatter(rows, "cuts_per_min", "Segments per minute"),
            "extra_chart": bars(st),
            "insight": f"{ins2} {ins1}", "recommendation": f"{rec2} {rec1}",
            "table": [{"Method": g["group"], "Renders": g["n"], "Rated": g["rated"], "Avg ★": g["mean"]} for g in st]}


def card_split(rows: list) -> dict:
    st = group_stats(rows, "max_clips", label=lambda v: f"up to {v}" if v and v > 1 else "single clip")
    ins1, rec1 = compare_groups(st, "max clip count")
    pct_rows = [dict(r, split_pct=(round(r["split_share"] * 100) if r["split_share"] is not None else None))
                for r in rows]
    ins2, rec2 = numeric_insight(pct_rows, "split_pct", "share of split-screen segments", "%")
    return {"key": "split", "title": "🔲 Split-screen", "subtitle": "Max simultaneous clips, and how much of the render is split",
            "chart": bars(st), "extra_chart": scatter(pct_rows, "split_pct", "% of segments split-screen"),
            "insight": f"{ins1} {ins2}", "recommendation": f"{rec1} {rec2}",
            "table": [{"Max clips": g["group"], "Renders": g["n"], "Rated": g["rated"], "Avg ★": g["mean"]} for g in st]}


def card_clips(rows: list) -> dict:
    ins, rec = numeric_insight(rows, "avg_clip_len", "average clip length", "s")
    st = group_stats(rows, "min_clip_len", label=lambda v: f"min {v:g}s")
    ins2, _ = compare_groups(st, "minimum clip length")
    return {"key": "clips", "title": "⏱ Clip length", "subtitle": "How long each clip stays on screen",
            "chart": scatter(rows, "avg_clip_len", "Average clip length (s)"),
            "insight": f"{ins} {ins2}", "recommendation": rec,
            "table": [{"Min clip length": g["group"], "Renders": g["n"], "Rated": g["rated"], "Avg ★": g["mean"]}
                      for g in st]}


def card_videos(rows: list) -> dict:
    ins, rec = numeric_insight(rows, "videos_n", "number of videos per render")
    per_video = defaultdict(list)
    for r in rows:
        if r["rating"]:
            for v in set(r["videos"]):
                per_video[v].append(r["rating"])
    vstats = sorted(({"group": v, "n": len(x), "rated": len(x), "mean": round(statistics.mean(x), 2)}
                     for v, x in per_video.items() if len(x) >= 2), key=lambda g: (-g["mean"], -g["n"]))
    tops = [g["group"] for g in vstats if g["mean"] >= GOOD][:5]
    lows = [g["group"] for g in reversed(vstats) if g["mean"] <= 2.5][:5]
    extra = []
    if tops:
        extra.append("Often in your best renders: " + ", ".join(f"**{v}**" for v in tops) + ".")
    if lows:
        extra.append("Often in your weakest: " + ", ".join(f"**{v}**" for v in lows) + ".")
    same = group_stats(rows, "allow_same_video", label=lambda v: "same video allowed twice" if v else "no repeats")
    same_ins, _ = compare_groups(same, "repeat setting")
    return {"key": "videos", "title": "🎬 Video choice", "subtitle": "How many videos, and which ones, show up in good renders",
            "chart": bars(vstats[:10]) if vstats else scatter(rows, "videos_n", "Videos used"),
            "insight": " ".join([ins, *extra, same_ins]),
            "recommendation": (rec + (" The videos that keep turning up in weak renders may need a tighter "
                                      "library range — or leaving out." if lows else "")),
            "table": [{"Video": g["group"], "Rated renders": g["n"], "Avg ★": g["mean"]} for g in vstats]}


def card_tracks(rows: list) -> dict:
    ins, rec = numeric_insight(rows, "bpm", "track BPM")
    st = [g for g in group_stats(rows, "track") if g["rated"]]
    hi = [g["group"] for g in st if g["mean"] >= GOOD][:3]
    lo = [g["group"] for g in reversed(st) if g["mean"] <= 2.5][:3]
    extra = []
    if hi:
        extra.append("Tracks that render well: " + ", ".join(f"**{t}**" for t in hi) + ".")
    if lo:
        extra.append("Tracks that struggle: " + ", ".join(f"**{t}**" for t in lo) + " — try other settings or videos for them.")
    return {"key": "tracks", "title": "🎵 Tracks", "subtitle": "Which music the planner handles best",
            "chart": scatter(rows, "bpm", "Track BPM"),
            "insight": " ".join([ins, *extra]), "recommendation": rec,
            "table": [{"Track": g["group"], "Renders": g["n"], "Rated": g["rated"], "Avg ★": g["mean"]} for g in st]}


def card_rendering(rows: list) -> dict:
    speeds = [r["render_seconds"] / (r["duration"] / 60) for r in rows
              if r.get("render_seconds") and r.get("duration")]
    pts = [{"x": r["duration"] / 60, "y": r["render_seconds"] / 60, "name": r["name"]} for r in rows
           if r.get("render_seconds") and r.get("duration")]
    if speeds:
        med = statistics.median(speeds)
        streamed = [r for r in rows if r.get("streamed")]
        insight = (f"Rendering takes about **{med / 60:.1f} min per minute of video** (median of {len(speeds)})."
                   + (f" {len(streamed)} render(s) streamed sources from Google Drive." if streamed else ""))
        rec = ("Fine for previews." if med < 120 else
               "Slow — long split-screen sections are the usual cause; fewer simultaneous clips renders faster.")
    else:
        insight, rec = "No render times recorded yet.", "New renders record how long they took."
    return {"key": "rendering", "title": "⚙️ Rendering", "subtitle": "How long renders take (not a quality signal)",
            "chart": {"kind": "scatter", "points": pts, "x_title": "Render length (min)",
                      "y_title": "Time to render (min)"},
            "insight": insight, "recommendation": rec,
            "table": [{"Render": p["name"], "Length (min)": round(p["x"], 1), "Took (min)": round(p["y"], 1)}
                      for p in pts]}


def build_cards(renders: list, days: int = None) -> list:
    rows = filter_rows([to_row(r) for r in renders], days)
    return [card_overview(rows), card_matching(rows), card_segmentation(rows), card_split(rows),
            card_clips(rows), card_videos(rows), card_tracks(rows), card_rendering(rows)]
