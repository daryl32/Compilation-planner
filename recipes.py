"""
🍳 Recipes — Stage 3 step 4 (claude/new-matcher-design.md §C): what footage each 🧩 section type wants. One
recipe per type, shared by every track (variations per cutting style can come later). The new matcher scores
every footage window of the 🧭 timeline against the recipe of the song section it would play in.

A recipe:
    movement     {"lo": 0–1, "hi": 0–1, "trend": "flat" | "rise" | "fall"}   the person-movement level the
                 section wants (library scale, as the timeline's scaled_move) and whether it should build or fall
    beat_lock    0–1   how tightly movement should line up with the beat (tight for Rhythmic / Drop / Fast)
    tags         {moment tag: weight 0–1}   what should be visible, and how much it matters
    avoid        [moment tags]              what shouldn't be
    scene_require [scene tags]              only footage from scenes with these (filter)
    scene_prefer  [scene tags]              lean towards scenes with these
    description  ""                         an optional short line, compared with the pictures (SigLIP2 text side)
    shot_change  "hold" | "allow" | "change"   shot-size / POV consistency across the section's cuts
                 (hold = keep it steady, allow = a change is fine, change = contrast welcome — e.g. at a drop)
    roles        [scene role ids]          clips marked with one of these (🎭 Reviewer roles: intro, outro, your own)
                 get a bonus in this section — mainly for Intro / Ending
    pace         {"lo": s, "hi": s}   how long the clips should be in this section (seconds). The 🤖 new matcher
                 cuts each section on the beat at this pace — a "rise" trend shortens the clips through the
                 section, a "fall" lengthens them (section_cuts.py). Classic ignores it.

Defaults below are v1 best judgement from Daryl's vocabulary (9 Nov 2026) — his edits are saved as overrides in
catalogue/section_recipes.json; ↩️ Reset returns a type to the default. No Streamlit in here.
"""

from __future__ import annotations

import copy
import json
import os

from section_types import TYPE_IDS

TRENDS = {"flat": "steady", "rise": "building", "fall": "falling away"}
SHOT_CHANGES = {"hold": "keep the shot steady", "allow": "a change is fine", "change": "contrast welcome"}

DEFAULTS = {
    "intro": {
        "movement": {"lo": 0.0, "hi": 0.4, "trend": "flat"}, "beat_lock": 0.2,
        "tags": {"lingerie": 0.6, "Panties": 0.4, "Face": 0.5, "standing": 0.3, "masturbation": 0.3, "breasts": 0.3},
        "avoid": ["penetration", "creampie", "deepthroat", "orgasm"],
        "scene_require": [], "scene_prefer": [],
        "description": "teasing, slow, still dressed", "shot_change": "allow",
        "pace": {"lo": 4.0, "hi": 8.0}, "roles": ["intro"],
    },
    "buildup": {
        "movement": {"lo": 0.3, "hi": 0.7, "trend": "rise"}, "beat_lock": 0.5,
        "tags": {"Blowjob": 0.6, "Handjob": 0.5, "fingering": 0.5, "Face": 0.4, "breasts": 0.3, "deepthroat": 0.3},
        "avoid": ["creampie", "orgasm", "squirt"],
        "scene_require": [], "scene_prefer": [],
        "description": "foreplay building, getting closer", "shot_change": "hold",
        "pace": {"lo": 2.0, "hi": 4.0},
    },
    "warmup": {
        "movement": {"lo": 0.2, "hi": 0.5, "trend": "rise"}, "beat_lock": 0.3,
        "tags": {"fingering": 0.5, "masturbation": 0.5, "Handjob": 0.4, "toy": 0.3, "breasts": 0.4, "lingerie": 0.3},
        "avoid": ["penetration", "creampie", "orgasm"],
        "scene_require": [], "scene_prefer": [],
        "description": "", "shot_change": "allow",
        "pace": {"lo": 3.0, "hi": 6.0},
    },
    "slow": {
        "movement": {"lo": 0.0, "hi": 0.35, "trend": "flat"}, "beat_lock": 0.1,
        "tags": {"Face": 0.5, "Missionary": 0.4, "laying down": 0.4, "masturbation": 0.3, "Blowjob": 0.3},
        "avoid": ["creampie", "squirt"],
        "scene_require": [], "scene_prefer": [],
        "description": "slow, close, sensual", "shot_change": "hold",
        "pace": {"lo": 5.0, "hi": 10.0},
    },
    "rhythmic": {
        "movement": {"lo": 0.4, "hi": 0.7, "trend": "flat"}, "beat_lock": 0.9,
        "tags": {"penetration": 0.6, "Cowgirl": 0.6, "reverse cowgirl": 0.5, "Doggy": 0.5, "Missionary": 0.4},
        "avoid": ["creampie"],
        "scene_require": [], "scene_prefer": [],
        "description": "steady rhythm", "shot_change": "hold",
        "pace": {"lo": 2.0, "hi": 4.0},
    },
    "drop": {
        "movement": {"lo": 0.7, "hi": 1.0, "trend": "flat"}, "beat_lock": 0.8,
        "tags": {"penetration": 0.8, "Doggy": 0.6, "reverse cowgirl": 0.6, "Cowgirl": 0.5, "Ass": 0.5, "anal sex": 0.3},
        "avoid": ["masturbation", "Handjob", "lingerie"],
        "scene_require": [], "scene_prefer": [],
        "description": "hard, fast, full energy", "shot_change": "change",
        "pace": {"lo": 1.0, "hi": 2.0},
    },
    "warmdown": {
        "movement": {"lo": 0.2, "hi": 0.5, "trend": "fall"}, "beat_lock": 0.3,
        "tags": {"Missionary": 0.5, "Face": 0.5, "Blowjob": 0.4, "laying down": 0.4},
        "avoid": [],
        "scene_require": [], "scene_prefer": [],
        "description": "", "shot_change": "allow",
        "pace": {"lo": 3.0, "hi": 6.0},
    },
    "fast": {
        "movement": {"lo": 0.75, "hi": 1.0, "trend": "flat"}, "beat_lock": 1.0,
        "tags": {"penetration": 0.7, "Doggy": 0.6, "reverse cowgirl": 0.5, "Cowgirl": 0.5, "standing": 0.3},
        "avoid": ["masturbation", "Blowjob", "lingerie"],
        "scene_require": [], "scene_prefer": [],
        "description": "", "shot_change": "change",
        "pace": {"lo": 0.75, "hi": 1.5},
    },
    "action": {
        "movement": {"lo": 0.55, "hi": 0.9, "trend": "flat"}, "beat_lock": 0.6,
        "tags": {"penetration": 0.6, "Cowgirl": 0.5, "Doggy": 0.5, "deepthroat": 0.4, "Ass": 0.4, "lesbian": 0.3},
        "avoid": ["lingerie"],
        "scene_require": [], "scene_prefer": [],
        "description": "", "shot_change": "allow",
        "pace": {"lo": 1.5, "hi": 3.0},
    },
    "ending": {
        "movement": {"lo": 0.0, "hi": 0.6, "trend": "fall"}, "beat_lock": 0.2,
        "tags": {"creampie": 0.8, "orgasm": 0.7, "squirt": 0.5, "Face": 0.5},
        "avoid": [],
        "scene_require": [], "scene_prefer": [],
        "description": "the finish", "shot_change": "allow",
        "pace": {"lo": 4.0, "hi": 8.0}, "roles": ["outro"],
    },
}
assert set(DEFAULTS) == set(TYPE_IDS)


def default(type_id: str) -> dict:
    d = copy.deepcopy(DEFAULTS[type_id])
    d.setdefault("roles", [])
    return d


def normalise(r: dict, type_id: str) -> dict:
    """A saved recipe filled out with the default's shape (older files, missing keys)."""
    base = default(type_id)
    if not isinstance(r, dict):
        return base
    out = base
    mv = r.get("movement") if isinstance(r.get("movement"), dict) else {}
    out["movement"] = {"lo": float(min(max(mv.get("lo", base["movement"]["lo"]), 0.0), 1.0)),
                       "hi": float(min(max(mv.get("hi", base["movement"]["hi"]), 0.0), 1.0)),
                       "trend": mv.get("trend") if mv.get("trend") in TRENDS else base["movement"]["trend"]}
    if out["movement"]["hi"] < out["movement"]["lo"]:
        out["movement"]["lo"], out["movement"]["hi"] = out["movement"]["hi"], out["movement"]["lo"]
    try:
        out["beat_lock"] = float(min(max(r.get("beat_lock", base["beat_lock"]), 0.0), 1.0))
    except (TypeError, ValueError):
        pass
    if isinstance(r.get("tags"), dict):
        out["tags"] = {str(t): float(min(max(w, 0.0), 1.0)) for t, w in r["tags"].items()
                       if isinstance(w, (int, float)) and str(t).strip()}
    out.setdefault("roles", [])
    for k in ("avoid", "scene_require", "scene_prefer", "roles"):
        if isinstance(r.get(k), list):
            out[k] = [str(t) for t in dict.fromkeys(r[k]) if str(t).strip()]
    if isinstance(r.get("description"), str):
        out["description"] = " ".join(r["description"].split())
    if r.get("shot_change") in SHOT_CHANGES:
        out["shot_change"] = r["shot_change"]
    pc = r.get("pace") if isinstance(r.get("pace"), dict) else {}
    try:
        lo = float(min(max(pc.get("lo", base["pace"]["lo"]), 0.25), 30.0))
        hi = float(min(max(pc.get("hi", base["pace"]["hi"]), 0.25), 30.0))
        out["pace"] = {"lo": min(lo, hi), "hi": max(lo, hi)}
    except (TypeError, ValueError):
        pass
    return out


# ---------------------------------------------------------------------------
# Storage — catalogue/section_recipes.json: {type_id: recipe} (only types Daryl changed)
# ---------------------------------------------------------------------------

def _file():
    from config import CATALOGUE_DIR
    return CATALOGUE_DIR / "section_recipes.json"


def _read_all() -> dict:
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return {k: v for k, v in (data.items() if isinstance(data, dict) else []) if k in DEFAULTS and isinstance(v, dict)}


def _write_all(data: dict) -> None:
    from library_common import mark_pending
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    mark_pending(path.stem)


def load_all() -> dict:
    """{type_id: recipe} for every type — Daryl's overrides where saved, else the default."""
    saved = _read_all()
    return {t: normalise(saved[t], t) if t in saved else default(t) for t in TYPE_IDS}


def load(type_id: str) -> dict:
    return load_all()[type_id]


def is_custom(type_id: str) -> bool:
    return type_id in _read_all()


def save(type_id: str, recipe: dict) -> None:
    """Save a type's recipe (normalised); saving one equal to the default removes the override."""
    if type_id not in DEFAULTS:
        raise ValueError(f"unknown section type {type_id!r}")
    data = _read_all()
    r = normalise(recipe, type_id)
    if r == default(type_id):
        if data.pop(type_id, None) is not None:
            _write_all(data)
        return
    data[type_id] = r
    _write_all(data)


def reset(type_id: str) -> None:
    data = _read_all()
    if data.pop(type_id, None) is not None:
        _write_all(data)


def unknown_tags(recipe: dict, vocabulary: list) -> list:
    """Recipe tags that aren't in the vocabulary (spelling drift) — shown so Daryl can fix them."""
    known = {str(t).lower() for t in vocabulary}
    out = [t for t in list(recipe.get("tags") or {}) + list(recipe.get("avoid") or []) if str(t).lower() not in known]
    return list(dict.fromkeys(out))


def summary(recipe: dict) -> str:
    """One line for a card: movement band, beat lock, the top tags."""
    mv = recipe["movement"]
    top = sorted(recipe["tags"].items(), key=lambda kv: -kv[1])[:4]
    pc = recipe.get("pace") or {}
    parts = [f"movement {mv['lo']:.1f}–{mv['hi']:.1f} {TRENDS[mv['trend']]}", f"beat lock {recipe['beat_lock']:.1f}",
             f"clips {pc.get('lo', 0):g}–{pc.get('hi', 0):g} s"]
    if top:
        parts.append("wants " + ", ".join(t for t, _ in top))
    if recipe["avoid"]:
        parts.append("avoids " + ", ".join(recipe["avoid"][:3]))
    return " · ".join(parts)
