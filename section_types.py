"""
🧩 Section types — Stage 3 step 4 (claude/new-matcher-design.md §C). A track is split into section types the
new matcher's recipes (recipes.py) are written for:

    Intro · Build-up · Warm-up · Slow · Rhythmic · Drop · Warm-down · Fast · Action · Ending

A section map is a list of regions covering the whole track in order, like a story map:
    {"start": s, "end": s, "type": type_id, "why": "..."}
It is suggested from the song's structure (story_map.analyse: energy percentile, build-ups, drops, breakdowns,
vocals, place) with the track's saved 🎭 story-map roles as hints (Intro → Intro, Outro → Ending, Action → Drop /
Action, Slow → Build-up / Slow), snapped to the 🧠 Smart method's cuts, then edited and saved per track in
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
    return snap(suggest(STM.analyse(track, beats_per_bar), STM.saved_map(track["track_id"])), cuts)


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
        out["suggested"] = [[r["start"], r["end"], r["type"]]
                            for r in suggest(analysis, STM.saved_map(track.get("track_id", "")))]
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
