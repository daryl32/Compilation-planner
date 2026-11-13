"""➕ Add new videos — the panel at the top of 📁 Media Library → 🎬 Videos (add_videos.py does the work): one button
for the server's steps, and what every video that isn't in the 🤖 new matcher's pool yet is still waiting for."""

from __future__ import annotations

import time

import streamlit as st

import add_videos as AV


@st.cache_data(max_entries=2, ttl=30, show_spinner=False)
def _summary(stamp: str) -> dict:
    return AV.summary()


def _start() -> None:
    st.session_state["av_msg"] = AV.start()
    _summary.clear()


def render() -> None:
    stt = AV.read_status()
    running = stt.get("state") == "running"
    sm = _summary(str(stt.get("finished") or stt.get("started") or ""))
    waiting = sm["waiting"]
    n_wait = sum(len(v) for v in waiting.values())
    here = sum(len(waiting.get(k) or []) for k in ("scores", "timeline"))        # what ➕ can finish right now
    c = st.columns([4, 1.6], vertical_alignment="center")
    line = f"**{sm['ready']} of {sm['total']} videos** are ready for the 🤖 new matcher"
    if n_wait:
        line += " · " + " · ".join(f"{len(waiting[k])} waiting for {AV.NEED_LABEL[k]}" for k in AV.NEEDS if waiting.get(k))
    c[0].markdown(line)
    c[1].button("➕ Add new videos", key="av_go", type="primary" if here else "secondary", width="stretch",
                disabled=running, on_click=_start,
                help="After the laptop has processed new videos and uploaded their vision data: syncs from Google "
                     "Drive, scores the videos that have no tag scores yet with the student model in use, and builds "
                     "their timelines — in the background, one after the other. No terminal needed. Safe to press "
                     "any time: videos that already have everything are left alone.")
    if running:
        step = stt.get("step") or "sync"
        k = AV.STEPS.index(step) if step in AV.STEPS else 0
        st.progress((k + 0.5) / len(AV.STEPS),
                    text=f"Step {k + 1} of {len(AV.STEPS)} — {AV.STEP_LABEL.get(step, step)}"
                         + (f" · {stt['detail']}" if stt.get("detail") else ""))
        for n in stt.get("notes") or []:
            st.caption(n)
        st.caption("Runs on the server in the background — you can leave this page.")
        time.sleep(3)
        _summary.clear()
        st.rerun()
    elif stt.get("state") == "error":
        st.error(f"➕ The last run stopped: {stt.get('error')}")
        for n in stt.get("notes") or []:
            st.caption(n)
    elif stt.get("state") == "done" and stt.get("finished"):
        r = stt.get("result") or {}
        added = int(r.get("added") or 0)
        with st.expander(f"Last run {stt['finished']}: " + (f"{added} video(s) added" if added > 0 else "nothing new to add"),
                         expanded=bool(st.session_state.pop("av_msg", None))):
            for n in stt.get("notes") or []:
                st.caption(n)
    if n_wait:
        with st.expander(f"What the {n_wait} waiting video(s) need", expanded=False):
            names = st.toggle("Show which videos", key="av_names")
            for need in AV.NEEDS:
                vids = waiting.get(need) or []
                if not vids:
                    continue
                st.markdown(f"**{len(vids)} — no {AV.NEED_LABEL[need]} yet** · next: {AV.NEED_WHERE[need]}")
                if names:                  # (not drawn until asked for: it can be a hundred names)
                    st.caption(" · ".join(vids[:60]) + (f" … +{len(vids) - 60} more" if len(vids) > 60 else ""))
            st.caption("Each video is listed under the first thing it lacks. The order is: catalogue (laptop → "
                       "Drive), vision data (laptop → server), tag scores and timeline (➕ here).")
    if sm.get("no_preview"):
        st.caption(f"ℹ️ {len(sm['no_preview'])} video(s) have no 480p preview copy on the server — they still match, "
                   "but ⚡ Quick preview renders and the range picker need one (laptop: process_new_videos.bat, then ➕).")
