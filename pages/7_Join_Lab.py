"""
🧩 Join Lab — how the avatar's clips join (19 Nov 2026).

  📥 Sync     every avatar clip on Drive (voice clips, idles, transitions) + each look's home picture
  🔍 By look  how each clip starts and ends against her home pose, so you can see which joins a crossfade hides,
              which need a quick AI in-between, and which need a bridge clip
  🎞️ Try      line up clips and idles (trim, ping-pong to any length, crossfade) and render a small preview,
              every join scored
The logic is in join_lab.py.
"""

import time
from pathlib import Path

import streamlit as st

import join_lab as JL
import render_jobs as RJ
from page_setup import page_setup

page_setup("Join Lab", "pages/7_Join_Lab.py", title="🧩 Join Lab")

SYNC_JOB, TRY_JOB = "joinlab:sync", "joinlab:try"
ICON = JL.BAND_ICON


def _fmt(s):
    return "—" if s is None else f"{ICON[JL.band(s)]} {s:.0f}"


# ---------------------------------------------------------------------------
# 📥 Sync
# ---------------------------------------------------------------------------
idx = JL.load_index()
j, running = RJ.take_finished(SYNC_JOB, "jl_sync_seen")
with st.expander(f"📥 Clips — {len(idx['clips'])} on the server"
                 + (f" · synced {time.strftime('%d %b %H:%M', time.localtime(idx['synced']))}" if idx.get("synced") else ""),
                 expanded=not idx["clips"] or running or j is not None):
    if running:
        RJ.watch(SYNC_JOB)
        st.stop()
    if j is not None:
        if j.state == "error":
            st.error(f"Sync failed: {j.error}")
        else:
            r = j.result or {}
            st.success(f"{r.get('clips', 0)} clips · {r.get('fetched', 0)} fetched or updated · "
                       f"{r.get('homes', 0)} home pictures.")
            if r.get("errors"):
                st.warning(f"{len(r['errors'])} problem(s): " + " · ".join(r["errors"][:4]))
    st.caption("Brings every clip in Edge_voiceover (the Studio's small preview copies where they exist — press 🎞️ Make "
               "Studio previews on the Edge page after adding clips, it makes this much faster) and the picture in "
               "each look folder, which is used as her home pose. Only new or changed clips are fetched again.")
    c = st.columns([2, 1])
    if c[0].button("📥 Sync from Google Drive", type="primary", key="jl_sync"):
        RJ.start(SYNC_JOB, "Syncing the avatar clips", JL.sync)
        st.rerun()
    if c[1].button("↻ Start over", key="jl_redo", help="Fetch and measure every clip again."):
        RJ.start(SYNC_JOB, "Syncing the avatar clips", JL.sync, True)
        st.rerun()

if not idx["clips"]:
    st.info("No clips yet — 📥 Sync from Google Drive first.")
    st.stop()

st.caption(f"Scores compare two frames, 0–100: ✅ {JL.GOOD:.0f}+ a short crossfade hides the join · 🟡 "
           f"{JL.OK:.0f}–{JL.GOOD:.0f} a quick AI in-between (frame interpolation) · 🔴 under {JL.OK:.0f} needs a bridge "
           "(idle / transition clip, or a generated one).")

looks = JL.looks_in(idx)
if not looks:
    st.warning("No look folders found (\"<Outfit> - <Pose>\").")
    st.stop()
look = st.selectbox("👗 Look", looks, key="jl_look")
home = (idx.get("homes") or {}).get(look)
rows = JL.look_report(idx, look)
by_id = {r["id"]: r for r in rows}

tab_look, tab_try, tab_pairs = st.tabs(["🔍 By look", "🎞️ Try a sequence", "🔗 Back to back"])

# ---------------------------------------------------------------------------
# 🔍 By look
# ---------------------------------------------------------------------------
with tab_look:
    cols = st.columns([1, 3])
    if home and Path(home["path"]).exists():
        cols[0].image(home["path"], caption=f"Home: {home['file']}")
    else:
        cols[0].warning("No home picture in this look's folder — the start / end scores need one.")
    s = JL.summary(rows)
    with cols[1]:
        st.markdown(f"**{s['voice']} voice clips · {s['idles']} idles · {s['transitions']} transitions**")
        if s["voice"]:
            st.markdown(f"Voice clips **ending** at home: ✅ {s['end_crossfade']} · 🟡 {s['end_in-between']} · "
                        f"🔴 {s['end_bridge']}  \nVoice clips **starting** at home: ✅ {s['start_crossfade']} · "
                        f"🟡 {s['start_in-between']} · 🔴 {s['start_bridge']}")
            need = s["end_bridge"]
            st.caption(("🎉 Every voice clip ends close enough to home to join with a crossfade or an in-between — no bridge "
                        "clips needed for this look." if not need else
                        f"🔴 {need} voice clip(s) end away from home: each needs an idle or transition after it (see the "
                        "suggestions below), or a generated bridge.") if home else "")
    kinds = st.multiselect("Show", ["voice", "idle", "transition", "other"], default=["voice", "idle", "transition"],
                           key="jl_kinds")
    shown = [r for r in rows if r["kind"] in kinds]
    st.dataframe([{"Clip": r["name"], "Kind": r["kind"], "Seconds": round(r["seconds"] or 0, 1),
                   "Starts at home": _fmt(r["start"]), "Ends at home": _fmt(r["end"]),
                   "Loops (end → own start)": _fmt(r["loop"]), "To": r["to"]} for r in shown],
                 hide_index=True, use_container_width=True)
    red = [r for r in rows if r["kind"] == "voice" and JL.band(r["end"]) == "bridge"]
    if red and home:
        with st.expander(f"🔴 What could follow the {len(red)} clip(s) that end away from home"):
            for r in red:
                got = JL.best_idles(idx, look, r["id"])
                st.markdown(f"**{r['name']}** (ends {_fmt(r['end'])}) → " + (" · ".join(
                    f"{idx['clips'][i]['name']} {_fmt(sc)}" for sc, i in got) or "no idles in this look yet"))

# ---------------------------------------------------------------------------
# 🎞️ Try a sequence
# ---------------------------------------------------------------------------
with tab_try:
    jt, trying = RJ.take_finished(TRY_JOB, "jl_try_seen")
    if trying:
        RJ.watch(TRY_JOB)
    else:
        if jt is not None:
            if jt.state == "error":
                st.error(f"Render failed: {jt.error}")
            else:
                st.session_state["jl_last_try"] = jt.result
        last = st.session_state.get("jl_last_try")
        if last and Path(last["file"]).exists():
            st.video(last["file"])
            st.caption(f"{last['seconds']:.1f} s · crossfade {last['xfade']:.2f} s")
            for k, jn in enumerate(last["joins"], 1):
                st.write(f"Join {k}: {jn['from']} → {jn['to']}: {_fmt(jn['score'])} ({jn['band']})")
        names = {r["id"]: f"{r['name']} ({r['kind']})" for r in rows}
        order = st.multiselect("Clips in order (pick the same idle again to use it twice: add it, then ➕ below)",
                               list(names), format_func=names.get, key="jl_seq")
        extra = st.session_state.setdefault("jl_extra", [])
        add_cols = st.columns([3, 1])
        more = add_cols[0].selectbox("➕ Add a clip again (repeats allowed)", [""] + list(names),
                                     format_func=lambda v: names.get(v, "—"), key="jl_more")
        if add_cols[1].button("➕ Add", key="jl_add", disabled=not more):
            extra.append(more)
            st.rerun()
        seq = list(order) + [x for x in extra if x in names]
        if extra and st.button("Clear the added repeats", key="jl_clear_extra"):
            st.session_state["jl_extra"] = []
            st.rerun()
        if seq:
            table = [{"#": k + 1, "Clip": idx["clips"][cid]["name"], "Kind": idx["clips"][cid]["kind"],
                      "Seconds": round(float(idx["clips"][cid].get("seconds") or 0), 2), "Trim in": 0.0, "Trim out": 0.0,
                      "Ping-pong": idx["clips"][cid]["kind"] == "idle", "Hold (s)": 0.0} for k, cid in enumerate(seq)]
            edited = st.data_editor(table, hide_index=True, use_container_width=True, key=f"jl_tab_{'_'.join(seq)}",
                                    disabled=["#", "Clip", "Kind", "Seconds"],
                                    column_config={"Trim out": st.column_config.NumberColumn(help="0 = to the end"),
                                                   "Hold (s)": st.column_config.NumberColumn(
                                                       help="With ping-pong: how long it plays, going forwards then "
                                                            "backwards — an idle fills any gap this way. 0 = once.")})
            pieces = [{"id": cid, "trim_in": float(row["Trim in"] or 0), "trim_out": float(row["Trim out"] or 0),
                       "pingpong": bool(row["Ping-pong"]) and float(row["Hold (s)"] or 0) > 0,
                       "hold": float(row["Hold (s)"] or 0)} for cid, row in zip(seq, edited)]
            joins = JL.sequence_joins(idx, pieces) if len(pieces) > 1 else []
            for k, jn in enumerate(joins, 1):
                st.write(f"Join {k}: {jn['from']} → {jn['to']}: {_fmt(jn['score'])}")
            xf = st.select_slider("Crossfade", [0.0, 0.15, 0.3, 0.5, 0.8], value=0.3, key="jl_xf",
                                  format_func=lambda v: "cut" if not v else f"{v:g} s")
            if st.button("🎞️ Render the preview", type="primary", key="jl_go"):
                RJ.start(TRY_JOB, "Rendering the try", JL.render_sequence, idx, pieces, xf)
                JL.clear_tries()
                st.rerun()

# ---------------------------------------------------------------------------
# 🔗 Back to back
# ---------------------------------------------------------------------------
with tab_pairs:
    st.caption("Each voice clip's end against each other's start — which lines can play straight after another with "
               "only a crossfade. Rows = the clip that ends, columns = the one that starts.")
    voice = [r["id"] for r in rows if r["kind"] == "voice"]
    pick = st.multiselect("Voice clips (up to 20)", voice, default=voice[:12],
                          format_func=lambda i: idx["clips"][i]["name"], max_selections=20, key="jl_pairs")
    if len(pick) > 1:
        m = JL.pair_matrix(idx, pick)
        short = [Path(idx["clips"][i]["name"]).stem[:22] for i in pick]
        st.dataframe([{"Ends ↓ / starts →": short[a], **{short[b]: _fmt(m[a][b]) for b in range(len(pick))}}
                      for a in range(len(pick))], hide_index=True, use_container_width=True)
        flat = [x for row in m for x in row if x is not None]
        st.caption(f"Pairs: ✅ {sum(1 for x in flat if x >= JL.GOOD)} · 🟡 "
                   f"{sum(1 for x in flat if JL.OK <= x < JL.GOOD)} · 🔴 {sum(1 for x in flat if x < JL.OK)}")
