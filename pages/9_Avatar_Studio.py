"""
🧍 Avatar Studio — what to record next (19 Nov 2026; replaces the Studio artifact's recording tracker).

  🗓️ Round plan     her look for each part (the same plan Edge compiles use)
  📋 By look        per look: every line it needs, ✅ recorded with a background · 🟡 transparent only · ❌ not yet
  🎙 All lines      every line, where it plays, what's recorded where
  ⚠️ Wrong look     clips recorded in a look where their line never plays
  📦 Colab          per look: what's left, seconds, Colab units, and its to_record.txt written to Drive
The logic is in studio.py.
"""

from pathlib import Path

import streamlit as st

import cue_store as CS
import studio as SD
from page_setup import page_setup
from planner.edge import LOOK_PART_LABELS, LOOK_PARTS, load_looks, save_looks

page_setup("Avatar Studio", "pages/9_Avatar_Studio.py", title="🧍 Avatar Studio")
ss = st.session_state

msg = ss.pop("sd_msg", None)
if msg:
    getattr(st, msg[0])(msg[1])

cache = SD.load_clips()
top = st.columns([4, 1])
top[0].caption(f"{len(cache['clips'])} clips on Drive" + (f" · read {cache['at']}" if cache["at"] else " · not read yet")
               + " · lines and cues from the 🎙️ Cue Map · looks from the round plan")
if top[1].button("🔄 Read Drive", use_container_width=True, help="List every clip in Edge_voiceover again."):
    with st.spinner("Reading Google Drive…"):
        n, err = SD.refresh_clips()
    ss["sd_msg"] = ("error", f"Couldn't read Drive: {err}") if err else ("success", f"{n} clips found.")
    st.rerun()
if not cache["clips"]:
    st.info("Press 🔄 Read Drive to list her clips.")

cm = CS.load()
looks = load_looks(cm)
rows = SD.tracker(cm, looks, cache["clips"])
by = {r["name"]: r for r in rows}

tabs = st.tabs(["📋 By look", "🎙 All lines", "⚠️ Wrong look", "📦 Colab sessions", "🗓️ Round plan"])


def _status_line(r: dict) -> str:
    return " · ".join(f"{s} {lk}" for lk, s in r["needs"].items()) or "plays nowhere in the plan"


def _preview(clip_id):
    """▶ her clip, when the 🧩 Join Lab has its small copy on the server."""
    import join_lab as JL
    f = JL.lab_dir() / "clips" / f"{clip_id}.mp4"
    return f if clip_id and f.exists() else None


# ---------------------------------------------------------------------------
with tabs[0]:
    c = st.columns([2, 1, 1])
    view = c[0].radio("Show", ["To record", "All"], horizontal=True, key="sd_view")
    kinds = c[1].multiselect("Kinds", ["main", "filler", "surprise"], default=["main", "filler", "surprise"], key="sd_kinds")
    q = c[2].text_input("🔍 Find", key="sd_q")
    for lk in dict.fromkeys(v for v in (looks.get(p) for p in SD.PARTS) if v):
        mine = [r for r in rows if lk in r["needs"] and r["kind"] in kinds and (not q or q.lower() in r["name"].lower())]
        n_ok = sum(1 for r in mine if r["needs"][lk] == "✅")
        n_half = sum(1 for r in mine if r["needs"][lk] == "🟡")
        show = mine if view == "All" else [r for r in mine if r["needs"][lk] != "✅"]
        parts = [LOOK_PART_LABELS[p] for p in LOOK_PARTS if looks.get(p) == lk]
        with st.expander(f"👗 {lk} — {', '.join(parts)} · ✅ {n_ok} · 🟡 {n_half} · ❌ {len(mine) - n_ok - n_half} of {len(mine)}",
                         expanded=False):
            st.dataframe([{"": r["needs"][lk], "Line": r["name"], "Kind": r["kind"],
                           "Seconds": round(r["full_seconds"] or r["seconds"], 1),
                           "Plays in (this look)": SD.parts_label(set(r["need_parts"][lk])),
                           "Recorded in": ", ".join(sorted({x["look"] + ("" if x["background"] else " (transparent)")
                                                            for x in r["recorded"]})) or "—"}
                          for r in sorted(show, key=lambda r: ({"❌": 0, "🟡": 1, "✅": 2}[r["needs"][lk]], r["kind"] != "main",
                                                               r["full_seconds"] or r["seconds"]))],
                         hide_index=True, use_container_width=True)
    nowhere = [r["name"] for r in rows if not r["needs"]]
    if nowhere:
        st.caption("Lines that play nowhere in the plan (no cue, or only in parts with no look): " + " · ".join(nowhere))
    todo = {lk: [r["name"] for r in rows if r["needs"].get(lk) in ("❌", "🟡") and r["kind"] in kinds]
            for lk in dict.fromkeys(v for v in looks.values() if v)}
    st.download_button("📋 What to record (text)", "\n\n".join(f"{lk} ({len(v)})\n" + "\n".join(f"  - {n}" for n in v)
                                                               for lk, v in todo.items() if v).encode("utf-8"),
                       file_name="to_record.txt", key="sd_dl")

# ---------------------------------------------------------------------------
with tabs[1]:
    c = st.columns([2, 1])
    q2 = c[0].text_input("🔍 Find a line", key="sd_q2")
    only = c[1].toggle("Missing a recording only", key="sd_only")
    st.dataframe([{"Line": r["name"], "Kind": r["kind"], "Seconds": round(r["full_seconds"] or r["seconds"], 1),
                   "Plays in": SD.parts_label(set(r["parts"])), "Needs": _status_line(r),
                   "Recorded": ", ".join(sorted({x["look"] + ("" if x["background"] else " (transparent)") for x in r["recorded"]})) or "—"}
                  for r in sorted(rows, key=lambda r: r["name"].lower())
                  if (not q2 or q2.lower() in r["name"].lower()) and (not only or any(s != "✅" for s in r["needs"].values()))],
                 hide_index=True, use_container_width=True, height=520)

# ---------------------------------------------------------------------------
with tabs[2]:
    st.caption("Voice clips recorded in a look where their line never plays (per the round plan). They're kept — the app "
               "still uses them as stand-ins until the right one is recorded. Check them and re-record or move them.")
    show_t = st.toggle("Include the old transparent stand-ins", key="sd_wrong_t")
    wrong = [(r, x) for r in rows for x in r["wrong"] if x["background"] or show_t]
    if not wrong:
        st.success("No clip is recorded in the wrong look.")
    groups = {}
    for r, x in wrong:
        groups.setdefault(x["look"] + ("" if x["background"] else " (transparent)"), []).append((r, x))
    for lk, items in sorted(groups.items()):
        st.markdown(f"**Recorded as {lk}** — {len(items)}")
        for k, (r, x) in enumerate(items):
            cols = st.columns([4, 3, 1])
            cols[0].write(f"{r['name']} · plays in {SD.parts_label(set(r['parts']))}")
            cols[1].caption("Needs " + (", ".join(r["needs"]) or "no look") + f" · {x['folder']}")
            f = _preview(x.get("id"))
            if f is not None and cols[2].button("▶", key=f"sd_play_{lk}_{k}"):
                ss["sd_playing"] = str(f)
    if ss.get("sd_playing") and Path(ss["sd_playing"]).exists():
        st.video(ss["sd_playing"])
    if wrong and not any(_preview(x.get("id")) for _, x in wrong):
        st.caption("▶ needs the clips' small copies: 📥 Sync in the 🧩 Join Lab once.")

# ---------------------------------------------------------------------------
with tabs[3]:
    c = st.columns(3)
    upc = c[0].number_input("Colab units per chunk", 0.1, 5.0, SD.UNIT_PER_CHUNK, 0.01, key="sd_upc",
                            help="Your 18 Nov estimate on the 80 GB A100. Change it once you know better.")
    chunk = c[1].number_input("Seconds of audio per chunk", 2.0, 30.0, SD.CHUNK_SEC, 0.5, key="sd_chunk")
    redo = c[2].toggle("Count transparent-only lines (re-record with a background)", value=True, key="sd_redo")
    sess = SD.sessions(rows, looks, upc, chunk, redo)
    total = sum(s["main_units"] + s["extra_units"] for s in sess)
    st.caption(f"Everything left: ≈ {total:.0f} units, plus about {SD.SESSION_UNITS:g} a session to load the model.")
    st.dataframe([{"Look": s["look"], "Plays in": ", ".join(LOOK_PART_LABELS[p] for p in s["parts"]), "✅ done": s["done"],
                   "Main to record": len(s["main"]), "Main (s)": round(s["main_sec"]), "Main ≈ units": round(s["main_units"], 1),
                   "Fillers / surprises": len(s["extra"]), "F/S (s)": round(s["extra_sec"]),
                   "F/S ≈ units": round(s["extra_units"], 1), "of them 🟡 stand-ins": s["stand_ins"]} for s in sess],
                 hide_index=True, use_container_width=True)
    st.markdown("**to_record.txt** — the list the notebook's copy cell reads (main lines shortest first; fillers and "
                "surprises after, with a # in front — take it off to record them).")
    pick = st.selectbox("Look", [s["look"] for s in sess], key="sd_sess")
    s = next((x for x in sess if x["look"] == pick), None)
    if s:
        st.code(SD.to_record_text(s), language=None)
        st.code(f'DRIVE_FOLDER = "{SD.ROOT}/{s["look"]}"\nAVATAR_NAME = "<the home picture in that folder, without .png>"\n'
                'OUT_SUBFOLDER = "avatar_clips"', language="python")
        b = st.columns(2)
        from page_setup import drive_token
        token = drive_token()
        if b[0].button(f"📝 Write to_record.txt to “{s['look']}”", key="sd_write", disabled=not token, type="primary"):
            err = SD.write_to_record(token, s)
            ss["sd_msg"] = ("error", f"Not written: {err}") if err else ("success", f"to_record.txt written to {s['look']}.")
            st.rerun()
        if b[1].button("📝 Write it for every look", key="sd_write_all", disabled=not token):
            errs = [f"{x['look']}: {e}" for x in sess if (e := SD.write_to_record(token, x))]
            ss["sd_msg"] = ("error", " · ".join(errs)) if errs else ("success", f"to_record.txt written to {len(sess)} look folders.")
            st.rerun()
        if not token:
            st.caption("Connect Google Drive (sidebar) to write the files.")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.caption("Her look for each part — the same plan Edge compiles use (and the Edge page's 👗 panel). \"Last round\" is "
               "the final round whatever its number.")
    options = [""] + sorted(set(SD.drive_looks(cache["clips"])) | {v for v in looks.values() if v})
    cols = st.columns(3)
    new = {}
    for k, part in enumerate(LOOK_PARTS):
        cur = looks.get(part) or ""
        new[part] = cols[k % 3].selectbox(LOOK_PART_LABELS[part], options, index=options.index(cur) if cur in options else 0,
                                          key=f"sd_look_{part}", format_func=lambda v: v or "— none —")
    if new != looks and st.button("💾 Save the round plan", key="sd_plan_save", type="primary"):
        save_looks(new)
        ss["sd_msg"] = ("success", "Round plan saved — the tracker and the next compile use it.")
        st.rerun()
