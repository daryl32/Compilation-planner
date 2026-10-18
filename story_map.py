"""
Story map — which parts of a track call for which scene role (🎬 Intro,
🌙 Slow, 💥 Action, 🏁 Outro and your own; see scene_roles.py).

A story map is a list of regions that cover the whole track, in order:
    {"start": s, "end": s, "role": role_id or None, "why": "...",
     "alt": {"role": id, "score": 0..1} or None, "score": 0..1}
Times are seconds into the track — not block numbers — so changing the cuts
afterwards keeps the map where it was. role None = no role there (normal picks).

suggest() reads the song (the Smart method's sections, intensity, build-ups and
drops, plus vocals) and scores every stretch against each role's "music fit":
  energy  — low / medium / high on the song's own scale (percentile by length)
  place   — near the start / near the end
  moments — build-ups, drops (starts on one), breakdowns (a dip between bigger
            parts), vocals
The best role above a threshold wins the stretch; the runner-up is kept as
"alt" so the wizard can offer a one-click switch.

No Streamlit in here.
"""

from __future__ import annotations

import json
import os

import numpy as np

import cut_scoring as CS
import smart_cuts as SM

MIN_SCORE = 0.7         # a stretch gets a role only if it fits this well
PLACE_SHARE = 0.12      # "near the start / end": within this share of the track
PLACE_SCORE = 1.0       # Intro/Outro win their place (the runner-up is offered as the switch)
MOMENT_EDGE = 0.15      # a matching musical moment beats energy alone (see fit_score)
BUILD_BARS = 8          # a build-up is the last (up to) 8 bars into what it builds to


# ---------------------------------------------------------------------------
# Reading the song
# ---------------------------------------------------------------------------

def analyse(track: dict, beats_per_bar: int = 4) -> dict:
    """{"units": [stretch facts], "drops": [t], "builds": [(s, e)], "duration"}.
    Units are the song's sections, with build-ups split out as their own."""
    duration = float(track["duration_sec"])
    _, report = SM.plan_cuts(track, {"beats_per_bar": beats_per_bar})
    sections = report.get("sections") or []
    drops = list(report.get("drops") or [])
    bar = float(report.get("bar_sec") or CS.bar_sec(track, beats_per_bar) or 2.0)
    # The analysis can mark a whole section as building; the build-up you'd cut
    # slow clips to is the run into the drop — its last BUILD_BARS bars.
    builds = [(max(float(bs), float(be) - BUILD_BARS * bar), float(be)) for bs, be in report.get("builds") or []]
    if not sections:
        sections = [{"start": 0.0, "end": duration, "label": "?", "intensity": 0.5}]

    # Intensity per beat, for build-up stretches cut out of a section.
    beats, _, _ = CS.beat_grid(track)
    beats = np.asarray(beats, dtype=float)
    inten = np.zeros(0)
    if len(beats) >= 2:
        inten, _ = SM.intensity_per_beat(track, beats, bar)
    mids = (beats[:-1] + beats[1:]) / 2 if len(beats) >= 2 else np.zeros(0)

    def mean_int(a, b, fallback):
        sel = (mids >= a) & (mids < b)
        return float(inten[sel].mean()) if np.any(sel) else fallback

    # Cut sections at build-up boundaries.
    raw = []
    for sec in sections:
        a, b = float(sec["start"]), float(sec["end"])
        marks = sorted({a, b, *[x for bs, be in builds for x in (bs, be) if a + bar < x < b - bar]})
        for x0, x1 in zip(marks, marks[1:]):
            raw.append({"start": round(x0, 3), "end": round(x1, 3), "label": sec.get("label", "?"),
                        "intensity": mean_int(x0, x1, float(sec.get("intensity", 0.5)))})

    vocals, v_rate = SM._curve(track, "vocals")
    v_lo, v_hi = (np.percentile(vocals, 10), np.percentile(vocals, 90)) if len(vocals) else (0, 1)

    lengths = np.array([u["end"] - u["start"] for u in raw])
    ints = np.array([u["intensity"] for u in raw])
    # Length-weighted percentile: the share of the song quieter than this stretch,
    # plus half of what's about as loud (so two equal drops rank the same).
    total = max(float(lengths.sum()), 1e-9)
    pct = np.array([(lengths[ints < x - 0.03].sum() + 0.5 * lengths[np.abs(ints - x) <= 0.03].sum()) / total
                    for x in ints])

    units = []
    for i, u in enumerate(raw):
        a, b = u["start"], u["end"]
        overlap = sum(max(0.0, min(b, be) - max(a, bs)) for bs, be in builds)
        before = [x["intensity"] for x in raw[:i]]
        after = [x["intensity"] for x in raw[i + 1:]]
        dip = 0.0
        if before and after:
            dip = min(max(before), max(after)) - u["intensity"]
        vocal = 0.0
        if len(vocals):
            seg = vocals[int(a * v_rate):max(int(a * v_rate) + 1, int(b * v_rate))]
            vocal = float(np.clip((np.mean(seg) - v_lo) / max(v_hi - v_lo, 1e-9), 0, 1)) if len(seg) else 0.0
        units.append({
            **u,
            "pct": round(float(pct[i]), 3),
            "build": round(overlap / max(b - a, 1e-9), 3),
            "drop": any(a - bar * 0.5 <= d <= a + bar for d in drops),
            "breakdown": round(float(np.clip((dip - 0.12) / 0.25, 0, 1)), 3),
            "vocal": round(vocal, 3),
            "first": i == 0, "last": i == len(raw) - 1,
        })
    return {"units": units, "drops": drops, "builds": builds, "duration": duration, "bar_sec": bar}


# ---------------------------------------------------------------------------
# Scoring a stretch against a role's music fit
# ---------------------------------------------------------------------------

def _in_place(units: list, i: int, place: str, duration: float) -> bool:
    """Is unit i in the opening / closing stretch? The first (last) unit always
    is; following (preceding) quiet ones too while within PLACE_SHARE of the track."""
    if place == "start":
        if units[i]["first"]:
            return True
        return (units[i]["end"] <= duration * PLACE_SHARE
                and all(u["pct"] < 0.5 for u in units[:i + 1]))
    if place == "end":
        if units[i]["last"]:
            return True
        return (units[i]["start"] >= duration * (1 - PLACE_SHARE)
                and all(u["pct"] < 0.5 for u in units[i:]))
    return True


def fit_score(unit: dict, fit: dict, in_place: bool) -> tuple:
    """(score 0–1, reason) for one stretch and one role's fit."""
    reasons = []
    energy = fit.get("energy", "any")
    e = None
    if energy == "low":
        e = 1 - unit["pct"]
    elif energy == "high":
        e = unit["pct"]
    elif energy == "medium":
        e = 1 - 2 * abs(unit["pct"] - 0.5)
    if e is not None and e >= 0.7:
        reasons.append({"low": "one of the quietest parts", "high": "one of the most intense parts",
                        "medium": "middling intensity"}[energy])
    m, m_kind = None, None
    for moment in fit.get("moments") or []:
        v = {"builds": unit["build"], "drops": 1.0 if unit["drop"] else 0.0,
             "breakdowns": unit["breakdown"], "vocals": unit["vocal"]}[moment]
        if v >= 0.5:
            reasons.append({"builds": "a build-up", "drops": "starts on a drop",
                            "breakdowns": "a breakdown between bigger parts", "vocals": "vocals"}[moment])
        if m is None or v > m:
            m, m_kind = v, moment
    if e is not None and m is not None:
        # Either fits on energy alone, or a moment counts half with the energy —
        # so a drop into a middling verse isn't 'Action', but a build-up that's
        # only moderately quiet is still 'Slow'.
        score = max(e, 0.5 * m + 0.5 * e)
        # A specific moment beats a general energy match — except a drop into a
        # part that isn't loud (a verse kicking in isn't an 'Action' high point).
        # Build-ups can be loud and still be build-ups.
        if m >= 0.5 and (e >= 0.4 or m_kind != "drops"):
            score += MOMENT_EDGE
    elif e is not None or m is not None:
        score = e if e is not None else m
    else:
        score = 0.5
    score = min(score, 1.0)
    place = fit.get("place", "anywhere")
    if place != "anywhere":
        if not in_place:
            return 0.0, ""
        score = PLACE_SCORE
        reasons.insert(0, "the start of the track" if place == "start" else "the end of the track")
    return float(score), ", ".join(reasons)


def suggest(analysis: dict, roles: list) -> list:
    """Regions covering the whole track. roles: scene_roles.all_roles()."""
    units, duration = analysis["units"], analysis["duration"]
    usable = [r for r in roles if r["fit"].get("suggest", True)]
    picks = []
    for i, u in enumerate(units):
        scored = []
        for r in usable:
            s, why = fit_score(u, r["fit"], _in_place(units, i, r["fit"].get("place", "anywhere"), duration))
            if s >= MIN_SCORE:
                scored.append((s, r["id"], why))
        place_first = {r["id"]: r["fit"].get("place", "anywhere") == "anywhere" for r in usable}
        scored.sort(key=lambda x: (-x[0], place_first[x[1]]))
        if scored:
            best = scored[0]
            alt = {"role": scored[1][1], "score": round(scored[1][0], 2)} if len(scored) > 1 else None
            picks.append({"start": u["start"], "end": u["end"], "role": best[1], "why": best[2],
                          "score": round(best[0], 2), "alt": alt})
        else:
            picks.append({"start": u["start"], "end": u["end"], "role": None,
                          "why": "nothing stands out — normal picks", "score": 0.0, "alt": None})
    regions = merge_same(picks)
    if regions:
        regions[0]["start"], regions[-1]["end"] = 0.0, round(duration, 3)
    return regions


# ---------------------------------------------------------------------------
# Editing (all return a new list; regions always cover the track in order)
# ---------------------------------------------------------------------------

def merge_same(regions: list) -> list:
    """Join neighbours with the same role."""
    out = []
    for r in regions:
        if out and out[-1]["role"] == r["role"]:
            prev = out[-1]
            prev["end"] = r["end"]
            if r.get("why") and r["why"] not in (prev.get("why") or ""):
                prev["why"] = "; ".join(x for x in (prev.get("why"), r["why"]) if x)
            prev["score"] = max(prev.get("score", 0), r.get("score", 0))
            prev["alt"] = prev.get("alt") or r.get("alt")
        else:
            out.append(dict(r))
    return out


def blank(duration: float) -> list:
    return [{"start": 0.0, "end": round(duration, 3), "role": None, "why": "", "score": 0.0, "alt": None}]


def snap(regions: list, cut_times: list) -> list:
    """Move inner boundaries onto the nearest cut, so every block sits in one
    region. Regions squeezed to nothing are dropped."""
    cuts = np.asarray(sorted(cut_times), dtype=float)
    out = [dict(r) for r in regions]
    if len(cuts) == 0 or not out:
        return out
    for i in range(len(out) - 1):
        t = float(cuts[int(np.argmin(np.abs(cuts - out[i]["end"])))])
        out[i]["end"] = out[i + 1]["start"] = round(t, 3)
    return [r for r in out if r["end"] - r["start"] > 1e-3] or out[-1:]


def set_role(regions: list, i: int, role) -> list:
    out = [dict(r) for r in regions]
    out[i]["role"] = role
    out[i]["why"] = "set by you"
    return out


def move_boundary(regions: list, i: int, t: float) -> list:
    """Move the line between region i and i+1 to t (kept inside both)."""
    out = [dict(r) for r in regions]
    if i + 1 >= len(out):
        return out
    t = min(max(t, out[i]["start"] + 1e-3), out[i + 1]["end"] - 1e-3)
    out[i]["end"] = out[i + 1]["start"] = round(t, 3)
    return out


def split(regions: list, i: int, t: float) -> list:
    out = [dict(r) for r in regions]
    r = out[i]
    if not (r["start"] < t < r["end"]):
        return out
    first, second = dict(r), dict(r)
    first["end"] = second["start"] = round(t, 3)
    return out[:i] + [first, second] + out[i + 1:]


def merge_next(regions: list, i: int) -> list:
    out = [dict(r) for r in regions]
    if i + 1 >= len(out):
        return out
    out[i]["end"] = out[i + 1]["end"]
    return out[:i + 1] + out[i + 2:]


def role_at(regions: list, t: float):
    for r in regions:
        if r["start"] <= t < r["end"]:
            return r["role"]
    return regions[-1]["role"] if regions and t >= regions[-1]["end"] else None


def block_roles(regions: list, segments: list) -> list:
    """The role for each block (by its middle), or None."""
    return [role_at(regions, (s["start"] + s["end"]) / 2) for s in segments]


def totals(regions: list, segments: list) -> dict:
    """{role: {"sec": seconds, "blocks": n}} for roles in use."""
    out = {}
    for r in regions:
        if r["role"]:
            out.setdefault(r["role"], {"sec": 0.0, "blocks": 0})["sec"] += r["end"] - r["start"]
    for role in block_roles(regions, segments):
        if role:
            out.setdefault(role, {"sec": 0.0, "blocks": 0})["blocks"] += 1
    return out


# ---------------------------------------------------------------------------
# Saved story maps — one per track, in CATALOGUE_DIR/story_maps.json (synced to
# Drive with the catalogues), so the Media Library and every project see them.
# A project keeps its own copy too (see the planner).
# ---------------------------------------------------------------------------

def _maps_file():
    from config import CATALOGUE_DIR
    return CATALOGUE_DIR / "story_maps.json"


def saved_maps() -> dict:
    """{track_id: regions} for every track with a saved story map."""
    try:
        data = json.loads(_maps_file().read_text())
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, list) and v} if isinstance(data, dict) else {}


def saved_map(track_id: str):
    return saved_maps().get(track_id)


def save_map(track_id: str, regions) -> None:
    """Save (or with regions None / empty, remove) a track's story map."""
    from library_common import mark_pending
    maps = saved_maps()
    if regions:
        if maps.get(track_id) == regions:
            return
        maps[track_id] = regions
    elif track_id in maps:
        maps.pop(track_id)
    else:
        return
    path = _maps_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(maps, indent=1))
    os.replace(tmp, path)
    mark_pending(path.stem)
