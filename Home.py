"""
Home — entry point for the Media Planner webapp.

Handles Google authentication, email whitelist gate, and navigation
to the three apps.

Run with:
    streamlit run Home.py
"""

import streamlit as st
from config import WHITELISTED_EMAILS

APP_VERSION = "1.6.0"

st.set_page_config(page_title="Media Planner", layout="wide")

# ---------------------------------------------------------------------------
# Authentication — Streamlit native Google login
# ---------------------------------------------------------------------------
if not st.user.is_logged_in:
    st.title("Media Planner")
    st.caption(f"v{APP_VERSION}")
    st.divider()
    st.subheader("Please sign in to continue")
    st.button("🔐 Sign in with Google", on_click=st.login, type="primary")
    st.stop()

# ---------------------------------------------------------------------------
# Whitelist gate
# ---------------------------------------------------------------------------
if st.user.email not in WHITELISTED_EMAILS:
    st.title("Access Denied")
    st.error(f"**{st.user.email}** is not authorised to use this app.")
    st.caption("Contact the administrator to request access.")
    if st.button("Sign out"):
        st.logout()
    st.stop()

# ---------------------------------------------------------------------------
# Navigation page
# ---------------------------------------------------------------------------
st.title("Media Planner")
st.caption(f"v{APP_VERSION}  ·  {st.user.name}")

if st.sidebar.button("Sign out", key="signout_btn"):
    st.logout()

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📁 Media Library")
    st.write("Browse, tag, and manage your video and audio clip catalogue.")
    st.page_link("pages/3_Media_Library.py", label="Open Media Library →")

with col2:
    st.subheader("🔍 Reviewer")
    st.write("Review and correct auto-generated scene tags.")
    st.page_link("pages/2_Reviewer.py", label="Open Reviewer →")

with col3:
    st.subheader("🎬 Compilation Planner")
    st.write("Match clips to a music track and build a compilation plan.")
    st.page_link("pages/1_Compilation_Planner.py", label="Open Compilation Planner →")
