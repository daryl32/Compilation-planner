"""
Scene roles — storytelling marks on scenes, used by the planner and the story
map wizard (to come).

A scene can have any number of roles. Four are built in:
  🎬 Intro · 🌙 Slow · 💥 Action · 🏁 Outro
and you can add your own (CATALOGUE_DIR/scene_roles.json, synced to Drive with
the catalogues).

On a scene the roles live in "roles" (a list of role ids). The old
intro_candidate / outro_candidate flags are still written alongside so
anything reading them keeps working; scene_roles() reads both.

Every role has a "music fit" — where in a track it belongs — which the story
map wizard uses to suggest regions for it:
  energy  — "low" / "medium" / "high" / "any"
  place   — "start" / "end" / "anywhere"
  moments — any of MOMENTS (build-ups, drops, breakdowns, vocals)
  suggest — False = never suggested automatically; you paint its regions yourself
"""

from __future__ import annotations

import json
import os
import re

from config import CATALOGUE_DIR
from library_common import mark_pending

ROLES_FILE = CATALOGUE_DIR / "scene_roles.json"
ROLES_STEM = ROLES_FILE.stem

ENERGY = {"any": "Any energy", "low": "Low (quiet / calm)", "medium": "Medium", "high": "High (loudest parts)"}
PLACES = {"anywhere": "Anywhere", "start": "Near the start", "end": "Near the end"}
MOMENTS = {"builds": "Build-ups", "drops": "Drops", "breakdowns": "Breakdowns", "vocals": "Vocals"}

BUILTIN = [
    {"id": "intro", "name": "Intro", "emoji": "🎬", "colour": "#4e79a7", "builtin": True,
     "help": "Opening shots — the start of the track.",
     "fit": {"energy": "any", "place": "start", "moments": [], "suggest": True}},
    {"id": "slow", "name": "Slow", "emoji": "🌙", "colour": "#76b7b2", "builtin": True,
     "help": "Build-up clips — breakdowns, quiet stretches and the run into a drop.",
     "fit": {"energy": "low", "place": "anywhere", "moments": ["breakdowns", "builds"], "suggest": True}},
    {"id": "action", "name": "Action", "emoji": "💥", "colour": "#e15759", "builtin": True,
     "help": "Key clips — the high points and drops.",
     "fit": {"energy": "high", "place": "anywhere", "moments": ["drops"], "suggest": True}},
    {"id": "outro", "name": "Outro", "emoji": "🏁", "colour": "#b07aa1", "builtin": True,
     "help": "Closing shots — the end of the track.",
     "fit": {"energy": "any", "place": "end", "moments": [], "suggest": True}},
]
BUILTIN_IDS = [r["id"] for r in BUILTIN]
LEGACY_FLAGS = {"intro": "intro_candidate", "outro": "outro_candidate"}

PALETTE = ["#f28e2b", "#59a14f", "#edc948", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295"]


# ---------------------------------------------------------------------------
# The list of roles
# ---------------------------------------------------------------------------

def _load_custom() -> list:
    try:
        data = json.loads(ROLES_FILE.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return []
    out = []
    for r in (data.get("roles") if isinstance(data, dict) else None) or []:
        if isinstance(r, dict) and r.get("id") and r["id"] not in BUILTIN_IDS:
            out.append(_normalise(r))
    return out


def _normalise(r: dict) -> dict:
    fit = r.get("fit") if isinstance(r.get("fit"), dict) else {}
    return {
        "id": str(r["id"]),
        "name": str(r.get("name") or r["id"]),
        "emoji": str(r.get("emoji") or "🏷️"),
        "colour": str(r.get("colour") or PALETTE[0]),
        "help": str(r.get("help") or ""),
        "builtin": False,
        "fit": {
            "energy": fit.get("energy") if fit.get("energy") in ENERGY else "any",
            "place": fit.get("place") if fit.get("place") in PLACES else "anywhere",
            "moments": [m for m in fit.get("moments") or [] if m in MOMENTS],
            "suggest": bool(fit.get("suggest", True)),
        },
    }


def all_roles() -> list:
    """Built-in roles first, then your own, in the order they were added."""
    return [dict(r) for r in BUILTIN] + _load_custom()


def roles_by_id() -> dict:
    return {r["id"]: r for r in all_roles()}


def label(role_id: str, roles: dict | None = None) -> str:
    r = (roles or roles_by_id()).get(role_id)
    return f"{r['emoji']} {r['name']}" if r else f"🏷️ {role_id} (deleted)"


def _save_custom(custom: list) -> None:
    ROLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROLES_FILE.with_name(ROLES_FILE.name + ".tmp")
    tmp.write_text(json.dumps({"roles": custom}, indent=2))
    os.replace(tmp, ROLES_FILE)
    mark_pending(ROLES_STEM)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "role"


def add_role(name: str, emoji: str, colour: str, fit: dict, help_text: str = "") -> str:
    """Add one of your own roles. Returns its id. Raises ValueError on a clash."""
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("Give the role a name.")
    existing = all_roles()
    if any(r["name"].lower() == name.lower() for r in existing):
        raise ValueError(f"There's already a role called {name}.")
    base = rid = _slug(name)
    ids = {r["id"] for r in existing}
    n = 2
    while rid in ids:
        rid = f"{base}_{n}"
        n += 1
    custom = _load_custom()
    custom.append(_normalise({"id": rid, "name": name, "emoji": emoji, "colour": colour,
                              "help": help_text, "fit": fit}))
    _save_custom(custom)
    return rid


def update_role(role_id: str, **changes) -> None:
    """Change one of your own roles (name, emoji, colour, help, fit)."""
    custom = _load_custom()
    for r in custom:
        if r["id"] == role_id:
            if "name" in changes:
                name = " ".join(str(changes["name"]).split())
                if not name:
                    raise ValueError("Give the role a name.")
                if any(o["name"].lower() == name.lower() and o["id"] != role_id for o in all_roles()):
                    raise ValueError(f"There's already a role called {name}.")
                changes["name"] = name
            r.update({k: v for k, v in changes.items() if k in ("name", "emoji", "colour", "help", "fit")})
            custom = [_normalise(x) for x in custom]
            _save_custom(custom)
            return
    raise ValueError("Only your own roles can be changed.")


def delete_role(role_id: str) -> None:
    """Remove one of your own roles from the list. Scenes keep the mark (shown as
    'deleted') until they're next edited, so adding it back restores them."""
    _save_custom([r for r in _load_custom() if r["id"] != role_id])


def next_colour() -> str:
    used = {r["colour"] for r in all_roles()}
    return next((c for c in PALETTE if c not in used), PALETTE[len(used) % len(PALETTE)])


def describe_fit(fit: dict) -> str:
    """'High energy · anywhere · drops' — for lists and help text."""
    if not fit.get("suggest", True):
        return "not suggested — you mark its parts of the track yourself"
    bits = []
    if fit.get("energy", "any") != "any":
        bits.append(f"{fit['energy']} energy")
    if fit.get("place", "anywhere") != "anywhere":
        bits.append(PLACES[fit["place"]].lower())
    if fit.get("moments"):
        bits.append(", ".join(MOMENTS[m].lower() for m in fit["moments"]))
    return " · ".join(bits) or "anywhere in the track"


# ---------------------------------------------------------------------------
# Roles on a scene
# ---------------------------------------------------------------------------

def scene_roles(scene: dict) -> list:
    """The scene's role ids (its "roles" list plus the old intro/outro flags),
    built-ins first in their usual order."""
    ids = list(scene.get("roles") or [])
    for rid, flag in LEGACY_FLAGS.items():
        if scene.get(flag) and rid not in ids:
            ids.append(rid)
    order = {rid: i for i, rid in enumerate(BUILTIN_IDS)}
    return sorted(dict.fromkeys(ids), key=lambda r: (order.get(r, len(order)), ids.index(r)))


def set_scene_roles(scene: dict, role_ids) -> None:
    """Write the scene's roles (and keep the old intro/outro flags in step)."""
    ids = list(dict.fromkeys(role_ids or []))
    if ids:
        scene["roles"] = ids
    else:
        scene.pop("roles", None)
    for rid, flag in LEGACY_FLAGS.items():
        scene[flag] = rid in ids


def motion_hint(rank: float | None) -> str | None:
    """A suggested role from where a scene's motion sits in its own video
    (0 = calmest, 1 = busiest), or None. Only ever shown as a hint."""
    if rank is None:
        return None
    if rank <= 0.2:
        return "slow"
    if rank >= 0.8:
        return "action"
    return None
