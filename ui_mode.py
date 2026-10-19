"""
Simple / Advanced view — one switch, remembered per signed-in user (not per project,
so loading a project or starting a new one never changes it).

Simple is almost a wizard: the everyday choices only (track → cutting style → videos →
Auto → render). Advanced shows every dial, weight, threshold and reset.

Pages call advanced() to decide what to draw, and switch() to draw the switch.
"""

from __future__ import annotations

import json

import streamlit as st

from config import PROJECTS_DIR

KEY = "ui_advanced"
PREFS_FILE = PROJECTS_DIR / ".ui_prefs.json"


def _user() -> str:
    try:
        return (getattr(st.user, "email", None) or "default").lower()
    except Exception:
        return "default"


def _load() -> dict:
    try:
        data = json.loads(PREFS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save() -> None:
    prefs = _load()
    prefs.setdefault(_user(), {})["advanced"] = bool(st.session_state.get(KEY))
    try:
        PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PREFS_FILE.with_name(PREFS_FILE.name + ".tmp")
        tmp.write_text(json.dumps(prefs, indent=1, sort_keys=True))
        tmp.replace(PREFS_FILE)
    except OSError:
        pass


def advanced() -> bool:
    """True when the Advanced view is on (Simple is the default)."""
    if KEY not in st.session_state:
        st.session_state[KEY] = bool((_load().get(_user()) or {}).get("advanced", False))
    return bool(st.session_state[KEY])


def switch(where=None) -> bool:
    """Draw the switch (in `where`, default the page) and return advanced()."""
    advanced()
    (where or st).toggle("🛠️ Advanced", key=KEY, on_change=_save,
                         help="Simple shows the everyday choices only: track, cutting style, videos, Auto and "
                              "render. Advanced shows every dial, weight and threshold. Remembered for you on "
                              "every device.")
    return bool(st.session_state[KEY])
