"""Run a real Streamlit page headless (streamlit.testing AppTest) against the
synthetic library, signed in as the test user."""

from __future__ import annotations

import streamlit as st

from tests.support import REPO  # noqa: F401  (installs the test config first)


class _User:
    is_logged_in = True
    email = "tester@example.com"
    name = "Tester"

    def get(self, k, d=None):
        return getattr(self, k, d)

    def to_dict(self):
        return {"email": self.email, "name": self.name}


# AppTest runs the page in this process, so patching the module is enough.
st.user = _User()
st.login = lambda *a, **k: None


def page_app(page: str, timeout: float = 120, advanced: bool = True):
    """The page headless. advanced: the 🛠️ Advanced view (every dial) — what most tests exercise;
    advanced=False for the Simple view."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO / page), default_timeout=timeout)
    at.session_state["ui_advanced"] = advanced
    return at


def errors(at) -> list:
    """Exceptions and st.error messages the run produced."""
    out = [f"EXCEPTION: {e.value}" for e in at.exception]
    out += [f"ERROR: {e.value}" for e in at.error]
    return out


def press_match(at) -> bool:
    """🎬 Matching & Export: press ▶️ Match / Apply & match if it's there (Auto builds nothing until
    then). Returns whether it was pressed."""
    go = [b for b in at.button if b.label in ("▶️ Match", "▶️ Apply & match")]
    if go:
        go[0].click().run()
    return bool(go)
