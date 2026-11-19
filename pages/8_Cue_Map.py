"""
🎙️ Cue Map — Edge Mode's voiceover lines and when they play (19 Nov 2026; replaces the claude.ai Cue Map artifact).

One copy, on the server: every save keeps the version it replaces and is refused if another device saved in between.
The page is in cue_map_view.py, the storage rules in cue_store.py.
"""

import streamlit as st

import cue_map_view as V
import cue_store as CS
from page_setup import page_setup

page_setup("Cue Map", "pages/8_Cue_Map.py", title="🎙️ Cue Map")

ss = st.session_state
cm = V.current()
msg = ss.pop("cm_msg", None)
if msg:
    getattr(st, msg[0])(msg[1])

top = st.columns([4, 1])
lines = cm.get("lines") or []
top[0].caption(f"Revision {cm.get('rev', 0)} · {len(lines)} lines, {sum(1 for ln in lines if ln.get('cues'))} with cues"
               + (f" · saved {cm['saved_at']}" if cm.get("saved_at") else f" · from: {cm.get('source') or 'the app'}"))
if top[1].button("🔄 Reload", use_container_width=True, help="Read the cue map again (e.g. after saving on another device)."):
    V.current(reload=True)
    V.select(ss.get(V.SS_SEL))
    st.rerun()

if not CS.managed():
    st.info("This cue map still comes from the old Cue Map page: every compile picks up the newest file \"sent\" to "
            "Google Drive, and that copy can overwrite your work. Keep it in the app from now on: compiles then use "
            "only what is saved here, and every save is kept as a version.")
    if st.button("✅ Keep the cue map in the app from now on", type="primary"):
        V.save_map(cm, "now kept in the app")
        st.rerun()

names = [ln["name"] for ln in lines]
t1, t2, t3 = st.tabs(["🕒 Timeline", "🎙️ Lines", "🗂️ Versions"])
with t1:
    V.timeline(cm)
with t2:
    V.lines_tab(cm)
with t3:
    V.versions_tab(cm)

st.divider()
V.editor(names)
