"""Embed HTML / a URL in an iframe — st.iframe on Streamlit versions that have it (the
replacement for the deprecated streamlit.components.v1.html / .iframe), else the old calls,
so the app keeps working on whichever Streamlit the server has installed."""

from __future__ import annotations

import streamlit as st


def html(body: str, height: int = 0) -> None:
    """Raw HTML (with its scripts) in an iframe that can reach the page (window.parent)."""
    if hasattr(st, "iframe"):
        st.iframe(body, height=max(int(height), 1))
    else:
        import streamlit.components.v1 as components
        components.html(body, height=height)


def url(src: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(src, height=int(height))
    else:
        import streamlit.components.v1 as components
        components.iframe(src, height=height)
