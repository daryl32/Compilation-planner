"""
Home — entry point for the Media Planner webapp.

Navigation to the three apps. Sign-in, the whitelist gate and Google Drive
connect are handled by page_setup (page_setup.py), shared by every page.

Run with:
    streamlit run Home.py
"""

import streamlit as st

from page_setup import page_setup

page_setup("Media Planner", "Home.py")

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
