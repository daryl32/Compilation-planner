"""🎙️ Cue Map page (pages/8_Cue_Map.py) — the Edge Mode voiceover cue map, edited in the app (19 Nov 2026).

Replaces the claude.ai Cue Map artifact. The storage rules (one copy, versions, refused stale saves) are in
cue_store.py; the engine that turns cues into moments is edge_mode.resolve_lines.

  🕒 Timeline   a sample compile (difficulty / rounds / seed / surprises): phases, surprises and every line heard,
                tap a line to edit it; clashes listed
  🎙️ Lines      search / categories / no-cue filter / length sort; new MP3s on Drive not in the map yet
  🗂️ Versions   every save, restorable; export / import
  The line editor: play (as cut), length (start / end / speed, ✂ cut the silence), categories, volume, notes and
  cues (presets + per-type fields). It edits a draft; 💾 Save writes the whole map with a revision check.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

import streamlit as st

import cue_store as CS
import edge_mode as E

SS_MAP, SS_SEL, SS_DRAFT = "cm_map", "cm_sel", "cm_draft"
ROUND_OPTS = ["1", "2", "3", "4", "5", "6", "last"]
DIFF_OPTS = list(E.DIFFICULTIES)
PHASES = list(E.ORDER)
PH_LABEL = {p: E.PROFILES[p]["label"].title() for p in PHASES}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def saved() -> dict:
    """The cue map as saved on the server."""
    return CS.load()


def current(reload: bool = False) -> dict:
    """The working copy: the saved cue map plus any unsaved changes (kept in cue_draft.json, so they survive the page
    being dropped). reload: read both from disk again."""
    ss = st.session_state
    d = CS.load_draft()
    if d:                                                 # the unsaved changes, wherever they were made
        ss[SS_MAP] = d["map"]
    elif reload or SS_MAP not in ss or int(ss[SS_MAP].get("rev") or 0) != int(CS.load().get("rev") or 0) and not d:
        ss[SS_MAP] = CS.load()
    return ss[SS_MAP]


def _disk_work() -> dict:
    d = CS.load_draft()
    return d["map"] if d else CS.load()


def changed(cm: dict, line: dict) -> None:
    """Put an edited / new line into the unsaved changes. Merged line by line into the draft file as it is on disk, so
    another device's unsaved edits to other lines are kept (the unsaved changes follow Daryl between devices)."""
    work = _disk_work()
    if any(ln["name"] == line["name"] for ln in work["lines"]):
        work["lines"] = [line if ln["name"] == line["name"] else ln for ln in work["lines"]]
    else:
        work["lines"].append(line)
    CS.save_draft(work)
    st.session_state[SS_MAP] = work


def removed(name: str) -> None:
    work = _disk_work()
    work["lines"] = [ln for ln in work["lines"] if ln["name"] != name]
    CS.save_draft(work)
    st.session_state[SS_MAP] = work


def pending() -> dict:
    return CS.changes(saved(), current())


def select(name: str | None) -> None:
    """Open a line in the editor (a fresh draft of it, from the working copy)."""
    ss = st.session_state
    ss[SS_SEL] = name
    ss.pop(SS_DRAFT, None)
    if name:
        line = next((ln for ln in current()["lines"] if ln["name"] == name), None)
        if line is not None:
            ss[SS_DRAFT] = _draft_of(line)


def _draft_of(line: dict) -> dict:
    d = copy.deepcopy(line)
    played = CS.played_seconds(d)
    for c in d.get("cues") or []:                       # an end cue is edited as "finishes N s before the phase ends"
        if c.get("type") == "phase" and (c.get("at") or "start") == "end":
            c["_finish"] = round(max(float(c.get("offset") or 0) - played, 0.0), 2)
    return d


def _commit(draft: dict) -> dict:
    d = copy.deepcopy(draft)
    played = CS.played_seconds(d)
    for c in d.get("cues") or []:
        if "_finish" in c:
            if c.get("type") == "phase" and (c.get("at") or "start") == "end":
                c["offset"] = round(float(c["_finish"]) + played, 3)
                c["ends"] = "finish"
            c.pop("_finish", None)
    d["play_seconds"] = round(played, 3)
    return d


def save_map(cm: dict, note: str) -> bool:
    """💾 Save the working copy as the cue map (revision check); tells the user what happened. True when saved."""
    try:
        got = CS.save(cm, int(cm.get("rev") or 0), note)
    except CS.Stale as e:
        st.session_state["cm_msg"] = ("error", f"⚠️ Not saved — {e} Your changes are still here: ⬇️ export them in "
                                               "🗂️ Versions if you need them, then ✖ Discard and redo them on the newer map.")
        return False
    CS.clear_draft()
    st.session_state[SS_MAP] = got
    st.session_state["cm_msg"] = ("success", f"💾 Saved (revision {got['rev']}) — {note}")
    st.session_state["cm_gen"] = st.session_state.get("cm_gen", 0) + 1
    if st.session_state.get(SS_SEL):
        select(st.session_state[SS_SEL])
    return True


def discard() -> None:
    CS.clear_draft()
    st.session_state[SS_MAP] = CS.load()
    st.session_state["cm_gen"] = st.session_state.get("cm_gen", 0) + 1
    select(st.session_state.get(SS_SEL))


def save_bar_bottom() -> None:
    """A second 💾 under the editor, so it's in reach after a long edit."""
    ch = pending()
    n = sum(len(v) for v in ch.values())
    c = st.columns([4, 2])
    c[0].caption(f"{n} unsaved change(s)." if n else "✅ Everything is saved.")
    if c[1].button("💾 Save the cue map", type="primary", disabled=not n, use_container_width=True, key="cm_save_all_b"):
        names = ch["changed"] + ch["added"] + ch["deleted"]
        save_map(current(), f"{n} change(s): " + ", ".join(names[:8]))
        st.rerun()


def save_bar() -> None:
    """The one save: what's unsaved, 💾 Save the cue map, ✖ Discard."""
    ch = pending()
    names = ch["changed"] + ch["added"] + ch["deleted"]
    n = len(names)
    box = st.container(border=True)
    c = box.columns([4, 2, 1])
    bits = ([f"✎ {len(ch['changed'])} changed"] if ch["changed"] else []) + \
           ([f"➕ {len(ch['added'])} added"] if ch["added"] else []) + ([f"🗑 {len(ch['deleted'])} deleted"] if ch["deleted"] else [])
    if not n:
        c[0].caption("✅ Everything is saved.")
    else:
        c[0].markdown("**Not saved yet:** " + " · ".join(bits))
        c[0].caption(", ".join(names[:12]) + (" …" if n > 12 else ""))
    if c[1].button("💾 Save the cue map", type="primary", disabled=not n, use_container_width=True, key="cm_save_all"):
        save_map(current(), " · ".join(bits) + ": " + ", ".join(names[:8]))
        st.rerun()
    if c[2].button("✖ Discard", disabled=not n, use_container_width=True, key="cm_discard",
                   help="Throw away every unsaved change and go back to the saved cue map."):
        st.session_state["cm_confirm_discard"] = True
    if st.session_state.get("cm_confirm_discard") and n:
        w = box.columns([4, 1, 1])
        w[0].warning(f"Throw away {n} unsaved change(s)?")
        if w[1].button("Yes, discard", key="cm_discard_yes"):
            st.session_state.pop("cm_confirm_discard", None)
            discard()
            st.rerun()
        if w[2].button("Keep them", key="cm_discard_no"):
            st.session_state.pop("cm_confirm_discard", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def line_path(line: dict) -> Path | None:
    from planner.edge import line_file
    return line_file(line, cut=False)


def cut_silence(path: Path) -> tuple:
    """(start, end) of the sound in the file: the first and last moment above about −40 dB, with 0.08 s kept before
    and 0.15 s after (as the artifact's ✂ did)."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", str(path), "-af",
                        "silencedetect=noise=-40dB:d=0.05", "-f", "null", "-"], capture_output=True, text=True)
    from planner.edge import line_durations  # noqa: F401  (keeps import order sane on the server)
    import edge_render as R
    full = R.probe(path)["seconds"]
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    a, b = 0.0, full
    if starts and starts[0] <= 0.02 and ends:
        a = max(ends[0] - 0.08, 0.0)
    if starts and (len(ends) < len(starts) or (ends and ends[-1] >= full - 0.02)):
        b = min(starts[-1] + 0.15, full)
    return round(a, 2), round(b if b < full - 0.02 else 0.0, 2)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def presets() -> dict:
    out = {}
    for p in PHASES:
        out[f"Start of {PH_LABEL[p]}"] = {"type": "phase", "phases": [p], "at": "start", "offset": 0}
        out[f"During {PH_LABEL[p]}"] = {"type": "phase", "phases": [p], "at": "during", "offset": 0}
        out[f"End of {PH_LABEL[p]} (finishes as it ends)"] = {"type": "phase", "phases": [p], "at": "end", "offset": 0,
                                                              "_finish": 0.0}
    for p in ("slow", "medium", "fast", "extra"):
        out[f"🔁 Filler: at random once a round in {PH_LABEL[p]} (Medium / Hard)"] = {
            "type": "random", "phases": [p], "per": "round", "times": 1, "diffs": ["medium", "hard"]}
    out["🎬 Intro screen"] = {"type": "intro", "order": 9}
    out["🏁 Outro screen"] = {"type": "outro", "order": 9}
    out["🔗 Straight after another line"] = {"type": "after", "after": "", "gap": E.LINE_GAP}
    out["Start of a round"] = {"type": "round", "which": "every", "at": "start"}
    out["⏰ At a set time"] = {"type": "time", "clock": "1:00", "from": "start"}
    out["🔂 Every so often"] = {"type": "interval", "every": 60, "startAfter": 0}
    for k, lab in (("handsoff", "✋ Hands off"), ("override", "⏩ Override"), ("falsefinish", "🎭 False finish")):
        out[f"🎲 {lab}"] = {"type": "surprise", "kind": k, "at": "start", "length": ""}
    for v in out.values():
        v.setdefault("roundList", [])
        v.setdefault("diffs", [])
        v.setdefault("note", "")
    return out


# ---------------------------------------------------------------------------
# The line editor
# ---------------------------------------------------------------------------

def editor(names: list) -> None:
    ss = st.session_state
    draft = ss.get(SS_DRAFT)
    if not draft:
        st.info("Pick a line — tap one on the 🕒 Timeline or in 🎙️ Lines.")
        return
    old = ss.get(SS_SEL)
    k = f"{re.sub(r'[^A-Za-z0-9]', '_', old)}_{st.session_state.get('cm_gen', 0)}"       # not the saved rev: another
    # device saving must never reset what's being typed here
    orig = next((ln for ln in saved()["lines"] if ln["name"] == old), None)
    head_box = st.container()                           # filled last: it shows whether the fields below changed the line
    path = line_path(draft)
    full = float(draft.get("seconds") or 0.0)
    if path:
        import edge_render as R
        full = R.probe(path)["seconds"] or full
        a = float(draft.get("trim_in") or 0)
        b = float(draft.get("trim_out") or 0) or None
        st.audio(str(path), start_time=a, end_time=b)
        if abs(float(draft.get("speed") or 1) - 1) > 0.005:
            st.caption("▶ plays the cut at normal speed here; the compile plays it at the set speed.")
    else:
        st.warning(f"“{draft.get('file')}” isn't on the server yet — ⬇️ Get the lines from Google Drive on the Edge page.")
    c = st.columns(4)
    draft["trim_in"] = c[0].number_input("Starts at (s)", 0.0, max(full, 0.1), float(draft.get("trim_in") or 0), 0.05,
                                         key=f"cm_ti_{k}")
    draft["trim_out"] = c[1].number_input("Ends at (s, 0 = the end)", 0.0, max(full, 0.1),
                                          min(float(draft.get("trim_out") or 0), max(full, 0.1)), 0.05, key=f"cm_to_{k}")
    draft["speed"] = c[2].select_slider("Speed", [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25],
                                        value=min([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25],
                                                  key=lambda v: abs(v - float(draft.get("speed") or 1))), key=f"cm_sp_{k}")
    draft["gain"] = c[3].slider("Volume ± dB", -12, 12, int(draft.get("gain") or 0), key=f"cm_g_{k}")
    s = st.columns([2, 1, 1])
    s[0].caption(f"⏱ Plays **{CS.played_seconds(draft, full):.2f} s** of {full:.2f} s.")
    if path and s[1].button("✂ Cut the silence", key=f"cm_cut_{k}", use_container_width=True):
        draft["trim_in"], draft["trim_out"] = cut_silence(path)
        for kk in (f"cm_ti_{k}", f"cm_to_{k}"):
            ss.pop(kk, None)
        st.rerun()
    if s[2].button("Reset length", key=f"cm_rl_{k}", use_container_width=True):
        draft.update(trim_in=0, trim_out=0, speed=1)
        for kk in (f"cm_ti_{k}", f"cm_to_{k}", f"cm_sp_{k}"):
            ss.pop(kk, None)
        st.rerun()
    cats = CS.categories(current())
    cc = st.columns([3, 1])
    draft["cats"] = cc[0].multiselect("Categories", cats, default=[x for x in draft.get("cats") or [] if x in cats],
                                      key=f"cm_cats_{k}")
    newcat = cc[1].text_input("➕ New category", key=f"cm_nc_{k}")
    if newcat and newcat not in draft["cats"]:
        draft["cats"].append(newcat.strip())
    draft["notes"] = st.text_area("Notes (situations to turn into a cue later)", draft.get("notes") or "", key=f"cm_n_{k}",
                                  height=70)
    st.markdown(f"**Cues** ({len(draft.get('cues') or [])})")
    for i, cue in enumerate(list(draft.get("cues") or [])):
        _cue_editor(draft, i, cue, names, f"{k}_{i}")
    p = st.columns([3, 1])
    pre = presets()
    pick = p[0].selectbox("➕ Add a cue", [""] + list(pre), key=f"cm_add_{k}", format_func=lambda v: v or "— pick —")
    if p[1].button("➕ Add", key=f"cm_addb_{k}", disabled=not pick, use_container_width=True):
        draft.setdefault("cues", []).append(copy.deepcopy(pre[pick]))
        ss.pop(f"cm_add_{k}", None)
        st.rerun()
    work = next((ln for ln in current()["lines"] if ln["name"] == old), None)
    if work is not None and CS.norm(_commit(draft)) != CS.norm(work):      # a real change → the working copy (unsaved)
        changed(current(), _commit(draft))
    dirty = orig is None or CS.norm(_commit(draft)) != CS.norm(orig)
    with head_box:
        head = st.columns([5, 1])
        head[0].markdown(f"### ✎ {draft['name']}" + ("  ·  *new line — not saved yet*" if orig is None else
                                                     "  ·  *changed — not saved yet*" if dirty else ""))
        if orig is not None and head[1].button("↺ Undo", key=f"cm_undo_{k}", disabled=not dirty, use_container_width=True,
                                               help="Put this line back as it is in the saved cue map."):
            changed(current(), copy.deepcopy(orig))
            st.session_state["cm_gen"] = st.session_state.get("cm_gen", 0) + 1
            select(old)
            st.rerun()
    with st.expander("🗑 Delete this line"):
        st.caption("It goes from the cue map when you 💾 Save (the MP3 stays on Drive).")
        if st.checkbox("Yes, delete it", key=f"cm_delc_{k}") and st.button("🗑 Delete", key=f"cm_del_{k}"):
            removed(old)
            select(None)
            st.rerun()


def _cue_editor(draft: dict, i: int, cue: dict, names: list, k: str) -> None:
    t = cue.get("type")
    with st.expander(f"{i + 1}. {CS.describe(_show(cue, draft))}", expanded=False):
        c = st.columns(3)
        if t in ("phase", "random", "interval"):
            cue["phases"] = c[0].multiselect("Phases", PHASES, default=[p for p in cue.get("phases") or [] if p in PHASES],
                                             format_func=PH_LABEL.get, key=f"cm_ph_{k}")
        if t == "phase":
            at = c[1].radio("When", ["start", "during", "end"], index=["start", "during", "end"].index(cue.get("at") or "start"),
                            horizontal=True, key=f"cm_at_{k}",
                            format_func={"start": "Start", "during": "During", "end": "End"}.get)
            if at == "end" and "_finish" not in cue:
                cue["_finish"] = 0.0
            cue["at"] = at
            if at == "end":
                cue["_finish"] = c[2].number_input("Finishes this many s before it ends (0 = as it ends)", 0.0, 60.0,
                                                   float(cue.get("_finish") or 0), 0.1, key=f"cm_fin_{k}")
            else:
                cue["offset"] = c[2].number_input("Seconds in (0 = as it starts)", 0.0, 120.0,
                                                  float(cue.get("offset") or 0), 0.1, key=f"cm_off_{k}")
        elif t == "random":
            cue["per"] = c[1].selectbox("Once per", ["round", "phase", "compilation"],
                                        index=["round", "phase", "compilation"].index(cue.get("per") or "round"), key=f"cm_per_{k}")
            cue["times"] = c[2].number_input("Times", 1, 10, int(cue.get("times") or 1), key=f"cm_tm_{k}")
        elif t == "interval":
            cue["every"] = c[1].number_input("Every (s)", 5, 600, int(cue.get("every") or 60), key=f"cm_ev_{k}")
            cue["startAfter"] = c[2].number_input("First after (s)", 0, 600, int(cue.get("startAfter") or 0), key=f"cm_sa_{k}")
        elif t == "time":
            cue["clock"] = c[0].text_input("At (m:ss)", str(cue.get("clock") or "1:00"), key=f"cm_ck_{k}")
            cue["from"] = c[1].radio("Counted from", ["start", "round"], index=0 if cue.get("from") != "round" else 1,
                                     horizontal=True, key=f"cm_fr_{k}",
                                     format_func={"start": "the start", "round": "each round's start"}.get)
        elif t == "round":
            which = ["every", "first", "last", "number"]
            cue["which"] = c[0].selectbox("Which round", which, index=which.index(cue.get("which") or "every"), key=f"cm_wh_{k}")
            if cue["which"] == "number":
                cue["number"] = c[1].number_input("Round", 1, 6, int(cue.get("number") or 1), key=f"cm_nb_{k}")
            cue["at"] = c[2].radio("At its", ["start", "end"], index=0 if cue.get("at") != "end" else 1, horizontal=True,
                                   key=f"cm_rat_{k}")
        elif t in ("intro", "outro"):
            cue["order"] = c[0].number_input("Order on the screen", 0, 99, int(cue.get("order") or 9), key=f"cm_or_{k}")
        elif t == "after":
            others = [n for n in names if n != draft["name"]]
            cur = cue.get("after") or ""
            cue["after"] = c[0].selectbox("Straight after", [""] + others,
                                          index=([""] + others).index(cur) if cur in others else 0, key=f"cm_af_{k}")
            cue["gap"] = c[1].number_input("Gap (s)", 0.0, 5.0, float(cue.get("gap") if cue.get("gap") is not None else E.LINE_GAP),
                                           0.05, key=f"cm_gp_{k}")
        elif t == "surprise":
            kinds = ["handsoff", "override", "falsefinish"]
            cue["kind"] = c[0].selectbox("Surprise", kinds, index=kinds.index(cue.get("kind") or "handsoff"), key=f"cm_sk_{k}",
                                         format_func={"handsoff": "✋ Hands off", "override": "⏩ Override",
                                                      "falsefinish": "🎭 False finish"}.get)
            ats = ["before", "start", "end"]
            cue["at"] = c[1].selectbox("At its", ats, index=ats.index(cue.get("at") or "start"), key=f"cm_sat_{k}")
            cue["length"] = c[2].text_input("Made for a length of (s, blank = any)", str(cue.get("length") or ""), key=f"cm_sl_{k}")
        d = st.columns([2, 2, 1])
        if t not in ("intro", "outro", "time", "round", "surprise"):
            cue["roundList"] = d[0].multiselect("Rounds (none = every round)", ROUND_OPTS,
                                                default=[str(r) for r in cue.get("roundList") or [] if str(r) in ROUND_OPTS],
                                                key=f"cm_rl_{k}")
        if t not in ("intro", "outro"):
            cue["diffs"] = d[1].multiselect("Difficulties (none = all)", DIFF_OPTS,
                                            default=[x for x in cue.get("diffs") or [] if x in DIFF_OPTS],
                                            format_func=E.DIFFICULTIES.get, key=f"cm_df_{k}")
        if d[2].button("🗑 Remove", key=f"cm_rm_{k}"):
            draft["cues"].pop(i)
            st.rerun()
        cue["note"] = st.text_input("Note", cue.get("note") or "", key=f"cm_cn_{k}")


def _show(cue: dict, draft: dict) -> dict:
    """The cue as the engine will read it (for its one-line description)."""
    if "_finish" in cue:
        return {**cue, "offset": 0}
    return cue


# ---------------------------------------------------------------------------
# 🕒 Timeline
# ---------------------------------------------------------------------------

def _rgb(p, a=1.0):
    r, g, b = E.PROFILES[p]["colour"]
    return f"rgba({r},{g},{b},{a})"


CAT_COLOURS = {"Filler": "#7a6ff0", "Surprise": "#b04ad8", "Rounds": "#2f8fd8", "React": "#e0629a", "Start": "#2fb39a",
               "Stop": "#888", "Slow": "#3fae6a", "Fast": "#e8772a", "Speed": "#e8772a"}


def timeline(cm: dict) -> None:
    import plotly.graph_objects as go
    c = st.columns([2, 1, 1, 1, 1])
    diff = c[0].selectbox("Difficulty", DIFF_OPTS, format_func=E.DIFFICULTIES.get, key="cm_tl_diff")
    rounds = c[1].selectbox("Rounds", [3, 4, 5, 6], index=3, key="cm_tl_rounds")
    seed = int(c[2].number_input("Draw", 0, 9999, 0, key="cm_tl_seed", help="Another random draw (Hard's phases, "
                                                                             "surprises, random lines)."))
    surp = c[3].toggle("🎲 Surprises", key="cm_tl_surp", disabled=diff == "hard")
    show = c[4].selectbox("Show", ["All"] + [f"Round {r}" for r in range(1, rounds + 1)], key="cm_tl_show")
    sched = E.schedule(diff, rounds, seed, surprises=surp and diff != "hard")
    got = E.resolve_lines(sched, cm, diff, seed)
    lo, hi = 0.0, E.total_seconds(sched)
    if show != "All":
        r = int(show.split()[1])
        lo = min(ph["start"] for ph in sched if ph["round"] == r)
        hi = max(ph["end"] for ph in sched if ph["round"] == r)
    by_name = {ln["name"]: ln for ln in cm["lines"]}
    fig = go.Figure()
    for ph in sched:
        if ph["end"] <= lo or ph["start"] >= hi:
            continue
        kind = ph.get("kind", "phase")
        fig.add_trace(go.Bar(y=["⏱ Phases"], x=[ph["end"] - ph["start"]], base=[ph["start"]], orientation="h",
                             marker=dict(color=_rgb(ph["profile"], 0.85 if kind == "phase" else 0.45),
                                         pattern=dict(shape="" if kind == "phase" else "/")),
                             text=[PH_LABEL[ph["profile"]] if kind == "phase" else ("✋" if kind == "handsoff" else "⏩")],
                             textposition="inside", insidetextanchor="middle", hovertemplate=(
                                 f"Round {ph['round']} · {PH_LABEL[ph['profile']]}" + (f" · {kind}" if kind != "phase" else "")
                                 + f"<br>{ph['start']:.0f}–{ph['end']:.0f} s<extra></extra>"),
                             showlegend=False, customdata=[""]))
    lanes = {"main": "🎙 Lines", "auto": "🔁 Fillers / callouts"}
    for p in got["plays"]:
        if p["time"] + p["seconds"] <= lo or p["time"] >= hi:
            continue
        lane = lanes["auto"] if str(p["why"]).startswith(("callout", "fills")) else lanes["main"]
        cat = next(iter(by_name.get(p["line"], {}).get("cats") or []), "")
        fig.add_trace(go.Bar(y=[lane], x=[p["seconds"]], base=[p["time"]], orientation="h",
                             marker=dict(color=CAT_COLOURS.get(cat, "#5b7fa6"), line=dict(color="white", width=1)),
                             text=[p["line"]], textposition="inside", insidetextanchor="start",
                             hovertemplate=f"<b>{p['line']}</b><br>{p['time']:.1f} s, {p['seconds']:.1f} s long<br>"
                                           f"{p['why']} (round {p['round']})<extra></extra>",
                             customdata=[p["line"]], showlegend=False))
    fig.update_layout(barmode="overlay", height=260, margin=dict(l=10, r=10, t=10, b=30), bargap=0.25,
                      xaxis=dict(range=[lo, hi], title="seconds"), yaxis=dict(categoryorder="array",
                                                                                categoryarray=list(lanes.values())[::-1] + ["⏱ Phases"]))
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="cm_tl_chart")
    pts = (ev or {}).get("selection", {}).get("points") if isinstance(ev, dict) else getattr(getattr(ev, "selection", None), "points", None)
    names = [n[0] if isinstance(n, list) else n for n in (pt.get("customdata") for pt in pts or [])]
    names = [n for n in names if n and n in by_name]
    tok = ("tl", tuple(names))
    if names and tok != st.session_state.get("cm_tl_tok"):          # act on a new tap only
        st.session_state["cm_tl_tok"] = tok
        if names[0] != st.session_state.get(SS_SEL):
            select(names[0])
            st.rerun()
    n_auto = sum(1 for p in got["plays"] if str(p["why"]).startswith(("callout", "fills")))
    st.caption(f"{len(got['plays'])} lines heard ({n_auto} fillers / callouts added by the app) · "
               f"{len(got['clashes'])} clash(es) · intro: {', '.join(got['intro']) or '—'} · outro: {', '.join(got['outro']) or '—'}"
               " · Tap a line to edit it.")
    if got["clashes"]:
        with st.expander(f"⚠️ {len(got['clashes'])} clash(es) in this draw"):
            for x in got["clashes"]:
                cols = st.columns([5, 1])
                cols[0].write(f"**{x['line']}** — {x['where']}: {x['result']} (with {x['with']})")
                if x["line"] in by_name and cols[1].button("✎", key=f"cm_clash_{x['line']}_{x['where']}"):
                    select(x["line"])
                    st.rerun()
    nocue = [ln["name"] for ln in cm["lines"] if not ln.get("cues")]
    if nocue:
        st.caption("No cue yet: " + " · ".join(nocue))


# ---------------------------------------------------------------------------
# 🎙️ Lines
# ---------------------------------------------------------------------------

def lines_tab(cm: dict) -> None:
    add_lines(cm)
    c = st.columns([3, 1, 1])
    q = c[0].text_input("🔍 Search", key="cm_q")
    nocue = c[1].toggle("No cue only", key="cm_nocue")
    order = c[2].selectbox("Sort", ["Name", "Shortest", "Longest"], key="cm_sort")
    allcats = CS.categories(cm)
    counts = {x: sum(1 for ln in cm["lines"] if x in (ln.get("cats") or [])) for x in allcats}
    none_n = sum(1 for ln in cm["lines"] if not ln.get("cats"))
    opts = [x for x in allcats if counts[x]] + (["(none)"] if none_n else [])
    cats = st.pills("Categories (tap to filter — more than one shows lines in any of them)", opts, selection_mode="multi",
                    key="cm_fcats", format_func=lambda x: f"{x} {counts.get(x, none_n)}") or []
    want = set(cats) - {"(none)"}

    def in_cats(ln):
        have = set(ln.get("cats") or [])
        return not cats or bool(want & have) or ("(none)" in cats and not have)
    rows = [ln for ln in cm["lines"] if (not q or q.lower() in ln["name"].lower() or q.lower() in (ln.get("notes") or "").lower())
            and in_cats(ln) and (not nocue or not ln.get("cues"))]
    if order == "Name":
        rows.sort(key=lambda ln: ln["name"].lower())
    else:
        rows.sort(key=lambda ln: CS.played_seconds(ln), reverse=order == "Longest")
    ch = pending()
    mark = {n: "✎" for n in ch["changed"]} | {n: "➕" for n in ch["added"]}
    table = [{"": mark.get(ln["name"], ""), "Line": ln["name"], "Plays (s)": round(CS.played_seconds(ln), 2),
              "✂": "✂" if CS.is_cut(ln) else "", "Categories": ", ".join(ln.get("cats") or []),
              "Cues": len(ln.get("cues") or []), "First cue": CS.describe(ln["cues"][0]) if ln.get("cues") else "—",
              "On server": "✅" if line_path(ln) else "—"} for ln in rows]
    ev = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                      key=f"cm_lines_{len(rows)}_{q}_{'_'.join(cats)}_{nocue}_{order}", height=360)
    sel = (ev or {}).get("selection", {}).get("rows") if isinstance(ev, dict) else getattr(getattr(ev, "selection", None), "rows", None)
    tok = ("df", rows[sel[0]]["name"]) if sel and sel[0] < len(rows) else None
    if tok and tok != st.session_state.get("cm_df_tok"):            # act on a new tap only
        st.session_state["cm_df_tok"] = tok
        if tok[1] != st.session_state.get(SS_SEL):
            select(tok[1])
            st.rerun()
    st.caption(f"{len(rows)} of {len(cm['lines'])} lines · tap a row to edit it · ✎ changed / ➕ added, not saved yet.")


def _add(cm: dict, name: str, file: str, seconds: float, folder: str) -> None:
    changed(cm, CS.new_line(name, file, seconds, folder))
    select(name)


def add_lines(cm: dict) -> None:
    """➕ Lines not in the cue map yet: MP3s found on Drive (looked for once a session, 🔎 again on demand), or an MP3
    uploaded here (kept on the server and, when Drive is connected, put in Edge_voiceover/New). Added lines are unsaved
    until 💾 Save."""
    ss = st.session_state
    if "cm_new" not in ss:
        ss["cm_new"] = new_on_drive(cm)
    have = {ln["name"].lower() for ln in cm["lines"]}
    new = [f for f in ss["cm_new"] if Path(f["name"]).stem.lower() not in have]
    with st.expander(f"➕ Add lines — {len(new)} new on Google Drive", expanded=bool(new)):
        top = st.columns([3, 1, 1])
        top[0].caption("MP3s in Edge_voiceover and its folders (not December, avatar_clips, transitions or \"_…\") "
                       "that aren't in the cue map yet.")
        if top[1].button("🔎 Look again", key="cm_scan", use_container_width=True):
            ss["cm_new"] = new_on_drive(cm)
            st.rerun()
        if new and top[2].button(f"➕ Add all {len(new)}", key="cm_add_all", use_container_width=True):
            for f in new:
                changed(cm, CS.new_line(Path(f["name"]).stem, f["name"], f["seconds"], f["folder"]))
            ss["cm_msg"] = ("info", f"➕ {len(new)} lines added — give them cues, then 💾 Save the cue map.")
            st.rerun()
        for f in new:
            cols = st.columns([4, 1])
            cols[0].write(f"**{Path(f['name']).stem}** · {f['folder'] or 'Edge_voiceover'} · ~{f['seconds']:.1f} s")
            if cols[1].button("➕ Add", key=f"cm_new_{f['folder']}_{f['name']}", use_container_width=True):
                _add(cm, Path(f["name"]).stem, f["name"], f["seconds"], f["folder"])
                st.rerun()
        if not new:
            st.caption("Nothing new on Drive.")
        st.markdown("**Or upload an MP3**")
        n = ss.get("cm_mp3_n", 0)
        ups = st.file_uploader("MP3 (the file name becomes the line's name)", type=["mp3"], accept_multiple_files=True,
                               key=f"cm_mp3_{n}")
        if ups:
            ss["cm_mp3_n"] = n + 1
            from planner.edge import lines_dir
            import edge_render as R
            done, notes = [], []
            for up in ups:
                stem = Path(up.name).stem
                if stem.lower() in have:
                    notes.append(f"{stem}: already a line")
                    continue
                dest = lines_dir() / up.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(up.getvalue())
                err = _to_drive(dest, cm)
                if err:
                    notes.append(f"{stem}: on the server, not on Drive ({err})")
                changed(cm, CS.new_line(stem, up.name, R.probe(dest)["seconds"], "New" if not err else ""))
                done.append(stem)
            if done:
                select(done[-1])
            ss["cm_msg"] = ("info" if not notes else "warning",
                            (f"➕ Added {', '.join(done)} — not saved yet. " if done else "") + " · ".join(notes))
            st.rerun()


def _to_drive(path: Path, cm: dict) -> str | None:
    """An uploaded line also goes to Drive (Edge_voiceover/New), so every machine and Colab can find it."""
    try:
        from page_setup import drive_token
        from drive_oauth import upload_file_with_oauth
        token = drive_token()
        if not token:
            return "Google Drive isn't connected on this page"
        root = str(cm.get("drive_folder") or "scene-labeling/Edge_voiceover").strip("/")
        _, err = upload_file_with_oauth(token, path, f"{root}/New")
        return err
    except Exception as e:                                # noqa: BLE001
        return str(e)


def new_on_drive(cm: dict) -> list:
    """[{"name", "folder", "seconds"}] — MP3s on Drive with no line in the cue map (seconds estimated at 192 kbps)."""
    try:
        from drive_sync import credentials_available, list_drive_folder
        from planner.edge import voice_folders
        if not credentials_available():
            return []
    except Exception:                                     # noqa: BLE001 — no Drive here: nothing to offer
        return []
    root = str(cm.get("drive_folder") or "scene-labeling/Edge_voiceover").strip("/")
    have = {str(ln.get("file") or "").lower() for ln in cm["lines"]} | {ln["name"].lower() + ".mp3" for ln in cm["lines"]}
    out, seen = [], set()
    for sub in [""] + voice_folders(root, cm):
        files, err = list_drive_folder(f"{root}/{sub}".rstrip("/"))
        for f in ([] if err else files):
            n = f["name"]
            if n.lower().endswith(".mp3") and n.lower() not in have and n.lower() not in seen and not E.take_stem(Path(n).stem, ""):
                seen.add(n.lower())
                out.append({"name": n, "folder": sub, "seconds": round(int(f.get("size") or 0) / 24000, 1)})
    return sorted(out, key=lambda f: f["name"].lower())


# ---------------------------------------------------------------------------
# 🗂️ Versions
# ---------------------------------------------------------------------------

def versions_tab(cm: dict) -> None:
    st.caption("Every save keeps the map it replaced. Restoring makes a new revision, so nothing is ever lost.")
    unsaved = any(pending().values())
    if unsaved:
        st.warning("You have unsaved changes — 💾 Save or ✖ Discard them before restoring or importing.")
    vs = CS.versions()
    if not vs:
        st.info("No earlier versions yet.")
    for v in vs[:40]:
        cols = st.columns([4, 1])
        cols[0].write(f"**r{v['rev']}** · {v['saved_at']} · {v['lines']} lines ({v['cued']} with cues)"
                      + (f" · {v['note']}" if v["note"] else ""))
        if cols[1].button("↩ Restore", key=f"cm_rs_{v['file']}", disabled=unsaved):
            try:
                got = CS.restore(v["file"], int(cm.get("rev") or 0))
                st.session_state[SS_MAP] = got
                st.session_state["cm_msg"] = ("success", f"Restored r{v['rev']} as revision {got['rev']}.")
                select(None)
            except CS.Stale as e:
                st.session_state["cm_msg"] = ("error", f"⚠️ {e} Press 🔄 Reload first.")
            st.rerun()
    st.divider()
    st.download_button("⬇️ Export the cue map (JSON" + (", with your unsaved changes)" if unsaved else ")"), json.dumps(cm, indent=1, ensure_ascii=False).encode("utf-8"),
                       file_name=f"edge_cues_r{cm.get('rev', 0)}.json", mime="application/json", key="cm_export")
    n = st.session_state.get("cm_up_n", 0)
    up = st.file_uploader("⬆️ Import a cue map file (replaces this one; it stays a version)", type=["json"], key=f"cm_up_{n}",
                          disabled=unsaved)
    if up is not None:
        st.session_state["cm_up_n"] = n + 1
        try:
            got = CS.import_raw(up.getvalue(), int(cm.get("rev") or 0), f"imported {up.name}")
            st.session_state[SS_MAP] = got
            st.session_state["cm_msg"] = ("success", f"Imported {up.name} as revision {got['rev']}.")
            select(None)
        except (CS.Stale, ValueError) as e:
            st.session_state["cm_msg"] = ("error", str(e))
        st.rerun()
