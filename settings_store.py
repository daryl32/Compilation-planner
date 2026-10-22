"""⚙️ One settings store — how a page's settings keep their values between reruns.

Streamlit forgets a widget's value on any run where that widget isn't drawn (another
planner step is showing, a section is collapsed into another view, …). The fix Streamlit
documents is to hand the value back to session_state yourself at the top of every run
(`st.session_state[k] = st.session_state[k]`): from then on the key belongs to the page,
not to the widget, and survives whether or not its widget is drawn.

That's all this does, in one place, for every setting a page registers:

    import settings_store as SSTORE
    SSTORE.keep(DEFAULTS)          # once per full run, before the first of those widgets

  • a key that's set keeps its value (a JSON-loaded 2 for a float dial comes back as 2.0,
    a list for a range slider as a tuple — so a widget never sees the wrong type);
  • a key that's missing gets its default.

Widgets then use key= only — no value= / index= copies from a "shadow" store — and
everything else reads st.session_state[key]. Callbacks (🎨 style, 🎯 Aim for, project
load) just write st.session_state; nothing else has to be kept in step.

choose() makes sure a selectbox / radio key holds one of its current options before the
widget is drawn (an option can disappear — a renamed style, an old project file).
"""

from __future__ import annotations

import copy

import streamlit as st


def _coerce(value, default):
    """value in the default's type where they differ only in form (int ↔ float, list ↔ tuple)."""
    if isinstance(default, bool) or isinstance(value, bool):
        return value
    if isinstance(default, float) and isinstance(value, int):
        return float(value)
    if isinstance(default, int) and isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def keep(defaults: dict, extra_keys=()) -> None:
    """Give every registered setting back to session_state for this run (see above).
    defaults: {key: default}; extra_keys: settings without a default, kept only when set."""
    ss = st.session_state
    for k, default in defaults.items():
        if k in ss:
            ss[k] = _coerce(ss[k], default)
        else:
            ss[k] = copy.deepcopy(default)
    for k in extra_keys:
        if k in ss:
            ss[k] = ss[k]


def choose(key: str, options, fallback=None) -> None:
    """Before a selectbox / radio is drawn: its key must hold one of `options` (else fallback,
    else the first option)."""
    options = list(options)
    if options and st.session_state.get(key) not in options:
        st.session_state[key] = fallback if fallback in options else options[0]
