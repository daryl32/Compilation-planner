"""Core logic of the Media Planner — no Streamlit, no session state.

Pages call these; so can scripts and background jobs (e.g. the planned
recommendation engine). Every function here takes what it needs as
arguments. Kept honest by tests/ (golden snapshots + headless page runs).

    segmentation  cutting a track into blocks, intensity, clip counts
    autotune      🎯 Auto-detect settings for a track
    shape         motion-vs-music shape matching
    footage       the footage pool and the operations that use it up
    picking       helpers for automatic picks (skip budget, weight F, …)
"""
