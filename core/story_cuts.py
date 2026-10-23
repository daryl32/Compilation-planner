"""🎭 Story-map-led cutting — each story-map part cut at its own pace (and, for the 🎭 Story-led
style, in its own style).

Every role has a cutting setting (scene_roles.role_cuts()):
    pace   — −2 … +2, added to the Pace slider for that role's parts (💥 Action busier, 🌙 Slow calmer)
    style  — a cutting style id (core/styles.py) used by 🎭 Story-led, or None = the track's base style

Two ways in, both off unless asked for:
  • 🎭 Story-led style — the base is 🧠 Smart at the Pace slider; each part with a role is cut in
    that role's style at slider + role pace.
  • "Pace per story part" with any other named style — the same style everywhere, each part at
    slider + its role's pace.

part_plans() turns that into a list of parts {start, end, method, settings}; the segmentation
config then carries them as cfg["parts"] and core.segmentation.build_segments() cuts the whole
track once with the base settings and splices each part's own cuts in (the same splice as the
✂️ cut editor's "re-cut a part"). A cut is added at a part's edges so a clip never straddles two
parts. No parts (no story map, nothing different from the base) → exactly the base cuts.

Pure: no Streamlit, no files.
"""

from __future__ import annotations

import core.styles as CSTY

STORY_STYLE = "story"
EDGE_GAP = 0.5      # seconds — no extra cut at a part's edge when a cut is already this close


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def part_choice(base_style: str, pace: int, role_cut: dict | None) -> tuple:
    """(style, pace) a part with this role's cutting setting is cut in."""
    rc = role_cut or {}
    off = int(rc.get("pace") or 0)
    if base_style == STORY_STYLE:
        style = rc.get("style") if rc.get("style") in CSTY.STYLES else "smart"
    else:
        style = base_style
    return style, int(_clamp(pace + off, -2, 2))


def part_plans(track: dict, regions, role_cuts: dict, base_style: str, pace: int,
               beats_per_bar: int = 4) -> list:
    """[{start, end, role, style, pace, method, settings}] for every part whose style or pace
    differs from the base (base_style at pace). [] when nothing differs or there's no map."""
    if not regions or base_style not in CSTY.STYLES:
        return []
    base = ("smart" if base_style == STORY_STYLE else base_style, int(_clamp(pace, -2, 2)))
    out, recipes = [], {}
    for r in regions:
        rid = r.get("role")
        if not rid:
            continue
        choice = part_choice(base_style, pace, (role_cuts or {}).get(rid))
        if choice == base:
            continue
        if choice not in recipes:
            method, settings, _ = CSTY.style_settings(track, choice[0], choice[1], beats_per_bar)
            recipes[choice] = (method, settings)
        method, settings = recipes[choice]
        part = {"start": round(float(r["start"]), 3), "end": round(float(r["end"]), 3), "role": rid,
                "style": choice[0], "pace": choice[1], "method": method, "settings": dict(settings)}
        # Touching parts cut the same way are one part.
        if out and out[-1]["end"] >= part["start"] - 1e-6 and (out[-1]["style"], out[-1]["pace"]) == choice:
            out[-1]["end"] = part["end"]
        else:
            out.append(part)
    return out


def describe(regions, role_cuts: dict, base_style: str, pace: int, roles: dict | None = None) -> list:
    """One line per role in the map: '💥 Action → 🕺 Hyper, busier' — for the planner's caption."""
    lines, seen = [], []
    for r in regions or []:
        rid = r.get("role")
        if rid and rid not in seen:
            seen.append(rid)
    for rid in seen:
        style, p = part_choice(base_style, pace, (role_cuts or {}).get(rid))
        name = (roles or {}).get(rid, rid)
        bits = ([CSTY.STYLES[style]["label"]] if base_style == STORY_STYLE else []) + [CSTY.PACE_LABELS[p]]
        lines.append(f"{name} → {', '.join(bits)}")
    return lines


def splice(track: dict, base_segments: list, part_segments: list, parts: list) -> list:
    """Cut times: the base cuts, with each part's span replaced by that part's own cuts and a cut at
    each part edge (unless one is already within EDGE_GAP)."""
    import core.cut_edits as CED
    duration = float(track["duration_sec"])
    cuts = CED.base_cut_times(base_segments)
    for part, segs in zip(parts, part_segments):
        cuts = CED.recut_region(cuts, CED.base_cut_times(segs), part["start"], part["end"], duration)
    for part in parts:
        for edge in (part["start"], part["end"]):
            if 0.0 < edge < duration and not any(abs(c - edge) < EDGE_GAP for c in cuts):
                cuts.append(round(float(edge), 3))
    return CED.clean_cuts(cuts, duration)
