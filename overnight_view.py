"""🌙 🔮 Recommendations → Overnight: the ⏱️ Edge and ⚔️ Verses compilations made at random last night (nightly_modes.py)
— watch, ⭐ rate / 📌 keep, ☁️ save to Drive, ↩️ open in their planner with the same setup, or 🌙 make a pair now."""

from __future__ import annotations

import datetime
from pathlib import Path

import streamlit as st

import nightly_modes as NM

JOB_KEY = "overnight_now"
KINDS = {"edge": ("⏱️ Edge Mode", "⏱️ Edge Mode"), "verses": ("⚔️ Verses Battle", "⚔️ Verses Battle")}


def _open(kind: str, meta: dict) -> None:
    from renders import reopen_state
    data, k = reopen_state(meta)
    if data is None:
        st.session_state["_overnight_msg"] = ("warning", "This render has no saved plan, so it can't be reopened.")
        return
    st.session_state["_reopen_" + k] = {**data, "label": Path(meta["path"]).name}
    st.session_state["planner_mode"] = KINDS[kind][1]
    st.session_state["_overnight_go"] = True


def _save_to_drive(meta: dict) -> None:
    from page_setup import drive_token
    from renders import auto_clear_enabled, clear_local, upload_render
    token = drive_token()
    if not token:
        st.session_state["_overnight_msg"] = ("warning", "Connect Google Drive (sidebar) to save renders there.")
        return
    err = upload_render(meta["path"], token)
    if err:
        st.session_state["_overnight_msg"] = ("error", f"Upload failed: {err}")
        return
    if auto_clear_enabled():
        clear_local(meta["path"])
    st.session_state["_overnight_msg"] = ("success", f"Saved {Path(meta['path']).name} to Google Drive.")


def _card(kind: str, got: dict, date: str) -> None:
    from config import PREVIEW_DIR
    from library_common import show_render_player
    from renders import load_render, on_drive, update_user_fields
    title = KINDS[kind][0]
    with st.container(border=True):
        if not got:
            st.markdown(f"**{title}**")
            st.caption("Not made that night.")
            return
        if got.get("error"):
            st.markdown(f"**{title}** · ❌")
            st.caption(f"It didn't finish: {got['error']}")
            return
        name = got.get("render")
        meta = load_render(Path(PREVIEW_DIR) / name) if name else None
        st.markdown(f"**{title}**")
        st.caption(got.get("line") or "")
        if got.get("took"):
            st.caption(got["took"])
        st.caption(f"⏳ Made in {int(got.get('seconds', 0) // 60)}:{int(got.get('seconds', 0) % 60):02d}")
        if meta is None:
            st.caption(f"{name} — no longer on the server or Google Drive (unrated overnight renders are cleared after "
                       f"{NM.KEEP_NIGHTS} nights).")
            return
        show_render_player(meta, height=300)
        user = meta.get("user") or {}
        key = f"on_{kind}_{date}"
        c = st.columns([2.2, 1.2, 1.6, 1.8])
        if f"{key}_stars" not in st.session_state and int(user.get("rating") or 0):
            st.session_state[f"{key}_stars"] = int(user["rating"]) - 1          # (the stars show its rating)
        rating = c[0].feedback("stars", key=f"{key}_stars")
        if rating is not None and rating + 1 != int(user.get("rating") or 0):
            update_user_fields(meta["path"], rating=rating + 1)
        kept = c[1].toggle("📌 Keep", value=bool(user.get("keep")), key=f"{key}_keep",
                           help=f"Kept renders are never cleared (unrated, unkept ones go after {NM.KEEP_NIGHTS} nights).")
        if kept != bool(user.get("keep")):
            update_user_fields(meta["path"], keep=kept)
        if on_drive(meta):
            c[2].caption("☁️ On Drive")
        else:
            c[2].button("☁️ Save to Drive", key=f"{key}_drive", width="stretch", on_click=_save_to_drive, args=(meta,))
        c[3].button("↩️ Open in planner", key=f"{key}_open", width="stretch", on_click=_open, args=(kind, meta),
                    help="Opens it in its planner with the same setup — 🎬 Render again, change things, or recompile.")


def _make_now(job, which) -> dict:
    return NM.run(tuple(which), log=lambda s: job.progress(None, s[:120]))


def render() -> None:
    import render_jobs as RJ
    msg = st.session_state.pop("_overnight_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    if st.session_state.pop("_overnight_go", False):
        st.switch_page("pages/1_Compilation_Planner.py")
    st.caption("🌙 Each night after the recommendation searches, one ⏱️ Edge Mode compilation (avatar + voiceover; "
               "difficulty, rounds, songs, section matching and surprises drawn at random) and one ⚔️ Verses Battle "
               "(two fighters drafted at random, music, layout and length drawn, a surprise winner) are compiled and "
               "rendered, so they're ready when you are. They stay on the server: ☁️ save the ones you like.")
    j, running = RJ.take_finished(JOB_KEY, "overnight_job_seen")
    if running:
        RJ.watch(JOB_KEY)
    elif j is not None and j.state == "error":
        st.error(f"Making them failed: {j.error}")
    with st.popover("🌙 Make a pair now", disabled=running):
        st.caption("Runs on the server like the nightly one (it finishes if you leave the page) — about as long as "
                   "an Edge and a Verses compile and render.")
        which = st.multiselect("Make", ["edge", "verses"], default=["edge", "verses"], key="overnight_which",
                               format_func=lambda k: KINDS[k][0])
        if st.button("🌙 Start", key="overnight_start", type="primary", disabled=not which):
            RJ.start(JOB_KEY, "🌙 Making overnight renders", _make_now, list(which))
            st.session_state["overnight_job_seen"] = None
            st.rerun()
    nights = NM.history()
    if not nights:
        st.info("Nothing yet — the first pair is made tonight after the recommendation searches (04:45), or press "
                "🌙 Make a pair now.")
        return
    for k, night in enumerate(nights[:NM.KEEP_NIGHTS + 1]):
        try:
            when = datetime.date.fromisoformat(night["date"]).strftime("%a %d %b")
        except (KeyError, ValueError):
            when = str(night.get("date"))
        body = st.container() if k == 0 else st.expander(f"🌙 {when}", expanded=False)
        with body:
            if k == 0:
                st.markdown(f"#### 🌙 {when}")
            cols = st.columns(2)
            for col, kind in zip(cols, ("edge", "verses")):
                with col:
                    _card(kind, night.get(kind), night.get("date", str(k)))
