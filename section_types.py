"""
🧩 Section types — Stage 3 step 4 (claude/new-matcher-design.md §C). A track is split into section types the
new matcher's recipes (recipes.py) are written for:

    Intro · Build-up · Warm-up · Slow · Rhythmic · Drop · Warm-down · Fast · Action · Ending

A section map is a list of regions covering the whole track in order, like a story map:
    {"start": s, "end": s, "type": type_id, "why": "..."}
It is suggested in two passes: the song's structure (story_map.analyse: energy percentile, build-ups, drops,
breakdowns, vocals, place) with the track's saved 🎭 story-map roles as hints (Intro → Intro, Outro → Ending,
Action → Drop / Action, Slow → Build-up / Slow), then a bar-level pass (refine: one-bar Drops, Action → Fast,
Warm-downs, short Build-ups — see below), snapped to the 🧠 Smart method's cuts, then edited and saved per track in
catalogue/section_maps.json. Roles stay as they are — they're for the scenes and remain a bonus in matching.

Section types are about the MUSIC, not any video: one shared list, one map per track. The recipes say what
footage each type wants. No Streamlit in here.
"""

from __future__ import annotations

import json
import os

import story_map as STM

TYPES = [
    {"id": "intro", "name": "Intro", "emoji": "🎬", "colour": "#4e79a7",
     "help": "The opening — the first stretch of the track."},
    {"id": "buildup", "name": "Build-up", "emoji": "🚀", "colour": "#f28e2b",
     "help": "The run into a drop — tension rising."},
    {"id": "warmup", "name": "Warm-up", "emoji": "🌤️", "colour": "#edc948",
     "help": "Early, middling energy before the first big part."},
    {"id": "slow", "name": "Slow", "emoji": "🌙", "colour": "#76b7b2",
     "help": "Quiet stretches and breakdowns."},
    {"id": "rhythmic", "name": "Rhythmic", "emoji": "🥁", "colour": "#59a14f",
     "help": "Steady, groove-led parts of medium energy."},
    {"id": "drop", "name": "Drop", "emoji": "💥", "colour": "#e15759",
     "help": "Starts on a drop — the loudest moments."},
    {"id": "warmdown", "name": "Warm-down", "emoji": "🌆", "colour": "#9c755f",
     "help": "Energy falling away after the last big part."},
    {"id": "fast", "name": "Fast", "emoji": "⚡", "colour": "#ff9da7",
     "help": "Very intense, driving, mostly instrumental."},
    {"id": "action", "name": "Action", "emoji": "🔥", "colour": "#d37295",
     "help": "High energy that isn't a drop — big choruses and peaks."},
    {"id": "ending", "name": "Ending", "emoji": "🏁", "colour": "#b07aa1",
     "help": "The close — the last stretch of the track."},
]
TYPE_IDS = [t["id"] for t in TYPES]
BY_ID = {t["id"]: t for t in TYPES}

# Where a 🎭 story-map role covers most of a stretch, it steers the type.
ROLE_HINTS = {"intro": "intro", "outro": "ending", "slow": "slow", "action": "action"}
HINT_SHARE = 0.6
LATE = 0.70            # a quiet stretch starting after this share of the track, after the peaks → Warm-down
EARLY = 0.35           # a middling stretch before this share, before the first big part → Warm-up


def label(type_id: str) -> str:
    t = BY_ID.get(type_id)
    return f"{t['emoji']} {t['name']}" if t else str(type_id or "—")


def types_as_roles() -> dict:
    """The types in the shape story_map_editor.story_chart wants for its colours."""
    return {t["id"]: {"name": t["name"], "emoji": t["emoji"], "colour": t["colour"]} for t in TYPES}


# ---------------------------------------------------------------------------
# Suggesting
# ---------------------------------------------------------------------------

def _role_share(regions: list, a: float, b: float) -> dict:
    """{role: share of [a, b] it covers} from a story map."""
    out = {}
    for r in regions or []:
        if not r.get("role"):
            continue
        ov = max(0.0, min(b, float(r["end"])) - max(a, float(r["start"])))
        if ov > 0:
            out[r["role"]] = out.get(r["role"], 0.0) + ov / max(b - a, 1e-9)
    return out


def classify(unit: dict, i: int, units: list, duration: float, story: list = None) -> tuple:
    """(type_id, why) for one stretch of story_map.analyse()."""
    pct, build, drop = unit["pct"], unit["build"], bool(unit["drop"])
    breakdown, vocal = unit["breakdown"], unit["vocal"]
    a, b = unit["start"], unit["end"]
    hints = _role_share(story, a, b)
    hint = max(hints.items(), key=lambda kv: kv[1])[0] if hints else None
    hinted = ROLE_HINTS.get(hint) if hint and hints[hint] >= HINT_SHARE else None
    peaks = [u for u in units if u["pct"] >= 0.6 or u["drop"]]
    first_peak = peaks[0]["start"] if peaks else duration
    last_peak_end = peaks[-1]["end"] if peaks else 0.0

    if STM._in_place(units, i, "start", duration) or hinted == "intro":
        return "intro", "the start of the track"
    if STM._in_place(units, i, "end", duration) or hinted == "ending":
        return "ending", "the end of the track"
    if build >= 0.5:
        return "buildup", "a build-up into what follows"
    if drop and pct >= 0.55:
        return "drop", "starts on a drop"
    if hinted == "action":
        return ("drop", "your 💥 Action role, on a drop") if drop else ("action", "your 💥 Action role")
    if hinted == "slow":
        return "slow", "your 🌙 Slow role"
    if pct < 0.5 and a >= duration * LATE and a >= last_peak_end - 1e-6:
        return "warmdown", "quieter, after the last peak"
    if 0.3 <= pct < 0.6 and b <= first_peak + 1e-6 and a < duration * EARLY:
        return "warmup", "middling energy before the first big part"
    if breakdown >= 0.5 or pct < 0.3:
        return "slow", "a breakdown between bigger parts" if breakdown >= 0.5 else "one of the quietest parts"
    if pct >= 0.75 and vocal < 0.4:
        return "fast", "very intense, mostly instrumental"
    if pct >= 0.6:
        return "action", "one of the most intense parts"
    return "rhythmic", "steady, medium energy"


def suggest(analysis: dict, story: list = None) -> list:
    units, duration = analysis["units"], analysis["duration"]
    regions = []
    for i, u in enumerate(units):
        t, why = classify(u, i, units, duration, story)
        regions.append({"start": u["start"], "end": u["end"], "type": t, "why": why})
    regions = merge_same(regions)
    if regions:
        regions[0]["start"], regions[-1]["end"] = 0.0, round(duration, 3)
    return regions


def merge_same(regions: list) -> list:
    out = []
    for r in regions:
        if out and out[-1]["type"] == r["type"]:
            out[-1]["end"] = r["end"]
        else:
            out.append(dict(r))
    return out


def snap(regions: list, cut_times: list) -> list:
    """Boundaries moved to the nearest Smart cut (story_map.snap works on any region dicts)."""
    return STM.snap(regions, cut_times)


def suggested_map(track: dict, beats_per_bar: int = 4) -> list:
    """The suggestion for a track, with its saved story map as hints, lined up with the Smart cuts."""
    import smart_cuts as SM
    segs = SM.build_smart_segments(track, {"beats_per_bar": beats_per_bar})
    cuts = [sg["start"] for sg in segs] + [float(track["duration_sec"])]
    analysis = STM.analyse(track, beats_per_bar)
    return snap(suggest_fine(track, analysis, STM.saved_map(track["track_id"])), cuts)


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def set_type(regions: list, i: int, type_id: str) -> list:
    out = [dict(r) for r in regions]
    out[i]["type"] = type_id
    out[i]["why"] = "set by you"
    return out


def split(regions: list, i: int, t: float) -> list:
    out = [dict(r) for r in regions]
    r = out[i]
    if not (r["start"] < t < r["end"]):
        return out
    out[i:i + 1] = [{**r, "end": round(t, 3)}, {**r, "start": round(t, 3), "why": "split by you"}]
    return out


def merge_next(regions: list, i: int) -> list:
    out = [dict(r) for r in regions]
    if i + 1 < len(out):
        out[i]["end"] = out[i + 1]["end"]
        del out[i + 1]
    return out


def type_at(regions: list, t: float):
    for r in regions or []:
        if r["start"] <= t < r["end"]:
            return r["type"]
    return regions[-1]["type"] if regions else None


def totals(regions: list) -> dict:
    """{type_id: seconds} over a map."""
    out = {}
    for r in regions or []:
        out[r["type"]] = out.get(r["type"], 0.0) + float(r["end"]) - float(r["start"])
    return out


# ---------------------------------------------------------------------------
# Storage — catalogue/section_maps.json: {track_id: {"regions": [...], "saved": "…"}}
# ---------------------------------------------------------------------------

def _file():
    from config import CATALOGUE_DIR
    return CATALOGUE_DIR / "section_maps.json"


def _read_all() -> dict:
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return {tid: v for tid, v in (data.items() if isinstance(data, dict) else []) if isinstance(v, dict) and v.get("regions")}


def _write_all(maps: dict) -> None:
    from library_common import mark_pending
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(maps, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    mark_pending(path.stem)


def saved_maps() -> dict:
    return {tid: v["regions"] for tid, v in _read_all().items()}


def saved_map(track_id: str):
    return (_read_all().get(track_id) or {}).get("regions")


def save_map(track_id: str, regions, source: str = "you") -> None:
    """Save (or with None / [] remove) a track's section map."""
    import time
    maps = _read_all()
    if not regions:
        if maps.pop(track_id, None) is not None:
            _write_all(maps)
        return
    maps[track_id] = {"regions": regions, "saved": time.strftime("%Y-%m-%d %H:%M"), "source": source}
    _write_all(maps)


def ensure_default(track_id: str, track: dict) -> bool:
    """Give a track a section map from the suggestion if it hasn't one. True if one was made."""
    if saved_map(track_id):
        return False
    try:
        regions = suggested_map(track)
    except Exception:          # noqa: BLE001 — a track the analysis can't read
        return False
    if not regions:
        return False
    save_map(track_id, regions, source="suggested")
    return True


# ---------------------------------------------------------------------------
# 📥 A track's sections with what the app measured — for tuning the suggester in a Claude chat
# ---------------------------------------------------------------------------

def tuning_export(track: dict, regions: list, analysis: dict = None, source: str = "") -> dict:
    """Everything needed to compare a hand-edited section map with the suggestion: the map, the suggester's own
    answer, the song's stretches (energy percentile, build-up, drop, breakdown, vocals) and the energy beat by beat."""
    import numpy as np
    import cut_scoring as CS
    import smart_cuts as SM
    r2 = lambda v: [round(float(x), 2) for x in np.asarray(v, float).ravel()]   # noqa: E731
    out = {"track_id": track.get("track_id"), "duration": round(float(track["duration_sec"]), 3),
           "bpm": track.get("bpm"), "source": source,
           "sections": [[round(float(r["start"]), 3), round(float(r["end"]), 3), r.get("type"), r.get("why", "")]
                        for r in regions or []]}
    try:
        analysis = analysis or STM.analyse(track)
        out["bar_sec"] = analysis.get("bar_sec")
        out["drops"] = r2(analysis.get("drops") or [])
        out["builds"] = [r2(b) for b in analysis.get("builds") or []]
        out["stretches"] = [{"start": u["start"], "end": u["end"], "energy_pct": u["pct"], "build": u["build"],
                             "drop": bool(u["drop"]), "breakdown": u["breakdown"], "vocal": u["vocal"]}
                            for u in analysis["units"]]
        story = STM.saved_map(track.get("track_id", ""))
        out["suggested_coarse"] = [[r["start"], r["end"], r["type"]] for r in suggest(analysis, story)]
        out["suggested"] = [[r["start"], r["end"], r["type"]] for r in suggest_fine(track, analysis, story)]
    except Exception as e:          # noqa: BLE001
        out["analysis_error"] = str(e)
    try:
        beats, bpm, _ = CS.beat_grid(track)
        beats = np.asarray(beats, float)
        inten, feats = SM.intensity_per_beat(track, beats, CS.bar_sec(track, 4) or 2.0)
        out["beat_bpm"] = round(float(bpm), 1)
        out["beats"] = r2(beats)
        out["beat_intensity"] = r2(inten)
        out["beat_features"] = {k: r2(v) for k, v in (feats or {}).items()}
    except Exception as e:          # noqa: BLE001
        out["beats_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# The bar-level pass (10 Nov 2026, from Daryl's hand-edited map of one track and his rules)
# ---------------------------------------------------------------------------
# The song's stretches (story_map.analyse) are ~20 s long; Daryl marks parts of 1–5 bars. So after the coarse
# suggestion, each bar is read as loud or not from the beat-by-beat intensity, and the parts are laid out by rule:
#   • Drop = the first bar of a loud run that starts on a jump in energy (or on a drop the analysis found) — the hit only.
#   • then Action; in a long loud run (≥ FAST_MIN_BARS after the drop) the later half is Fast — the peak comes last.
#   • Warm-down = the way down from Action / Fast: the first WARMDOWN_BARS bars after a loud run (a one-bar dip takes
#     the bar before it too).
#   • Build-up = at most BUILD_MAX_BARS bars before a loud run, only where the song builds (the coarse suggestion says
#     so, or the energy rises through them).
#   • whatever is left keeps the coarse suggestion's type where that was a quiet type (Intro, Warm-up, Slow, Rhythmic,
#     Ending — so story-map hints still count), else Rhythmic / Slow by level.
# The coarse boundaries are trusted (all of them matched Daryl's): a run edge one bar away from one moves onto it.
LOUD_SHARE = 0.85          # a bar is loud at ≥ this share of the track's upper-quartile bar intensity
DROP_JUMP = 0.25           # a loud run's first bar is a Drop when the energy jumps by at least this from the bar before
FAST_MIN_BARS = 6          # a loud run this long after its drop ends in Fast (its later half)
WARMDOWN_BARS = 2
BUILD_MAX_BARS = 4
BUILD_RISE = 0.1           # "the energy rises through them": last bar − first bar of the candidate build-up
MEDIUM_LEVEL = 0.4         # leftover bars: at or above → Rhythmic, below → Slow
QUIET_TYPES = ("intro", "warmup", "slow", "rhythmic", "ending")


def _bars(beats, inten, duration: float, anchor_time: float = None) -> tuple:
    """(bar start times, bar mean intensity): four beats to the bar, counted from the beat nearest anchor_time
    (a drop sits on a downbeat) — else from the first beat. The first bar starts at 0."""
    import numpy as np
    b = np.asarray(beats, float)
    it = np.asarray(inten, float)
    n = min(len(it), len(b) - 1)
    if n < 8:
        return np.zeros(0), np.zeros(0)
    off = (int(np.argmin(np.abs(b - anchor_time))) % 4) if anchor_time is not None else 0
    idx = list(range(off, n, 4))
    starts = [float(b[s]) for s in idx]
    means = [float(it[s:min(s + 4, n)].mean()) for s in idx]
    if off >= 2:
        starts.insert(0, 0.0)
        means.insert(0, float(it[:off].mean()))
    else:
        starts[0] = 0.0
    keep = [k for k, s in enumerate(starts) if s < duration - 0.25]
    return np.asarray([starts[k] for k in keep]), np.asarray([means[k] for k in keep])


def refine_with(coarse: list, beats, inten, drops, duration: float) -> list:
    """The bar-level pass on explicit data (see above). coarse: regions with "type". Returns regions."""
    import numpy as np
    starts, level = _bars(beats, inten, duration, float(drops[0]) if len(drops) else None)
    nb = len(starts)
    if nb < 6 or not coarse:
        return coarse
    ends = np.append(starts[1:], duration)
    bar = float(np.median(np.diff(starts))) if nb > 1 else 2.0
    thr = max(LOUD_SHARE * float(np.percentile(level, 75)), 0.6 * float(np.percentile(level, 95)))
    loud = level >= thr
    for k in range(1, nb - 1):                       # a single loud bar among quieter ones isn't a loud run
        if loud[k] and not loud[k - 1] and not loud[k + 1]:
            loud[k] = False
    cb = [float(r["start"]) for r in coarse[1:]]     # the coarse boundaries

    def near(t):
        return any(abs(t - c) <= 0.4 * bar for c in cb)

    def run_len(k):
        a = k
        while a > 0 and loud[a - 1] == loud[k]:
            a -= 1
        z = k
        while z + 1 < nb and loud[z + 1] == loud[k]:
            z += 1
        return z - a + 1
    for k in range(1, nb):                           # move a run edge that is one bar off a coarse boundary onto it
        if loud[k] == loud[k - 1] or near(starts[k]):
            continue
        if near(starts[k - 1]) and run_len(k - 1) > 1:
            loud[k - 1] = loud[k]
        elif k + 1 < nb and near(starts[k + 1]) and run_len(k) > 1:
            loud[k] = loud[k - 1]

    coarse_at = lambda t: type_at(coarse, t)         # noqa: E731
    types = [None] * nb
    runs, a = [], 0
    for k in range(1, nb + 1):
        if k == nb or loud[k] != loud[a]:
            runs.append((a, k, bool(loud[a])))
            a = k
    for ri, (i, j, is_loud) in enumerate(runs):
        if is_loud:
            k = i
            jump = i > 0 and level[i] - level[i - 1] >= DROP_JUMP
            known = any(starts[i] - 0.5 * bar <= float(d) < ends[i] for d in drops)
            if i > 0 and (jump or known):
                types[i] = "drop"
                k = i + 1
            j_ = j
            nxt = runs[ri + 1] if ri + 1 < len(runs) else None
            if nxt and nxt[1] - nxt[0] == 1 and nxt[1] != nb and j - 1 > k:
                types[j - 1] = "warmdown"            # a one-bar dip follows: the way down starts a bar early
                j_ = j - 1
            rest = j_ - k
            n_fast = rest // 2 if rest >= FAST_MIN_BARS else 0
            for m in range(k, j_):
                types[m] = "fast" if m >= j_ - n_fast else "action"
            continue
        k = i
        after_loud = ri > 0
        last = j == nb
        if after_loud:                               # the way down
            n_wd = min(WARMDOWN_BARS, j - i if not last else max(j - i - 1, 0))
            for m in range(i, i + n_wd):
                types[m] = "warmdown"
            k = i + n_wd
        e = j
        if not last and j - k >= 1:                  # a build-up into the loud run that follows
            n_b = min(BUILD_MAX_BARS, j - k)
            b0 = j - n_b
            said = any(coarse_at((starts[m] + ends[m]) / 2) == "buildup" for m in range(k, j))
            rises = level[j - 1] - level[b0] >= BUILD_RISE
            if said or rises:
                for m in range(b0, j):
                    types[m] = "buildup"
                e = b0
        for m in range(k, e):
            c = coarse_at((starts[m] + ends[m]) / 2)
            if last:
                types[m] = "ending"
            elif i == 0 and c in (None, "buildup", "drop", "action", "fast"):
                types[m] = "intro"
            elif c in QUIET_TYPES and not (c == "ending" and not last) and not (c == "intro" and i > 0):
                types[m] = c
            else:
                types[m] = "rhythmic" if float(np.mean(level[k:e])) >= MEDIUM_LEVEL else "slow"
    out = []
    for m in range(nb):
        t = types[m] or coarse_at((starts[m] + ends[m]) / 2) or "rhythmic"
        if out and out[-1]["type"] == t:
            out[-1]["end"] = round(float(ends[m]), 3)
        else:
            out.append({"start": round(float(starts[m]), 3), "end": round(float(ends[m]), 3), "type": t,
                        "why": WHY.get(t, "")})
    for r_prev, r in zip(out, out[1:]):              # tiny differences from a coarse boundary: use the coarse time
        c = min(cb, key=lambda x: abs(x - r["start"])) if cb else None
        if c is not None and abs(c - r["start"]) <= 0.4 * bar:
            r_prev["end"] = r["start"] = round(c, 3)
    out[0]["start"], out[-1]["end"] = 0.0, round(float(duration), 3)
    return [r for r in out if r["end"] - r["start"] > 1e-3]


WHY = {"drop": "the hit — the first bar of a loud stretch", "action": "a loud stretch", "fast": "the peak — the later half of a long loud stretch",
       "warmdown": "the way down from a loud stretch", "buildup": "the bars building into a loud stretch", "intro": "the start of the track",
       "ending": "the end of the track", "rhythmic": "steady, medium energy", "slow": "one of the quietest parts",
       "warmup": "middling energy before the first big part"}


def refine(coarse: list, track: dict, analysis: dict = None) -> list:
    """The bar-level pass for a track; the coarse regions come back unchanged if its beats can't be read."""
    try:
        import numpy as np
        import cut_scoring as CS
        import smart_cuts as SM
        beats, _, _ = CS.beat_grid(track)
        beats = np.asarray(beats, float)
        inten, _ = SM.intensity_per_beat(track, beats, CS.bar_sec(track, 4) or 2.0)
        drops = list((analysis or {}).get("drops") or [])
        return refine_with(coarse, beats, inten, drops, float(track["duration_sec"]))
    except Exception:          # noqa: BLE001
        return coarse


def suggest_fine(track: dict, analysis: dict, story: list = None) -> list:
    """The suggestion with the bar-level pass — what the app proposes for a track."""
    return refine(suggest(analysis, story), track, analysis)


def agreement(a: list, b: list, duration: float, step: float = 0.25) -> float:
    """Share of the track where two section maps give the same type."""
    import numpy as np
    ts = np.arange(step / 2, duration, step)
    return float(np.mean([type_at(a, t) == type_at(b, t) for t in ts])) if len(ts) else 0.0
