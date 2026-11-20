"""🎙️ Cue Map page (pages/8_Cue_Map.py) — the Edge Mode voiceover cue map, edited in the app (19 Nov 2026).

Replaces the claude.ai Cue Map artifact. The storage rules (one copy, versions, refused stale saves) are in
cue_store.py; the engine that turns cues into moments is edge_mode.resolve_lines.

  🕒 Timeline   a sample compile (difficulty / rounds / seed / surprises): phases, surprises and every line heard,
                tap a line to edit it; clashes listed
  🎙️ Lines      search / categories / no-cue filter / 🎚️ difficulty (plays in it / missing from it) / length sort;
                new MP3s on Drive not in the map yet
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


UNDO_KEEP = 40


def _remember() -> None:
    """↶ Undo: the unsaved state before this change (this session's last UNDO_KEEP)."""
    import json
    stack = st.session_state.setdefault("cm_undo", [])
    snap = json.dumps(_disk_work(), sort_keys=True)
    if not stack or stack[-1] != snap:
        stack.append(snap)
        del stack[:-UNDO_KEEP]


def undo() -> bool:
    import json
    stack = st.session_state.get("cm_undo") or []
    if not stack:
        return False
    prev = json.loads(stack.pop())
    if CS.norm(prev.get("lines")) == CS.norm(CS.load().get("lines")) and prev.get("autofill") == CS.load().get("autofill"):
        CS.clear_draft()
        st.session_state[SS_MAP] = CS.load()
    else:
        CS.save_draft(prev)
        st.session_state[SS_MAP] = prev
    st.session_state["cm_gen"] = st.session_state.get("cm_gen", 0) + 1
    if st.session_state.get(SS_SEL):
        select(st.session_state[SS_SEL])
    return True


def changed(cm: dict, line: dict) -> None:
    """Put an edited / new line into the unsaved changes. Merged line by line into the draft file as it is on disk, so
    another device's unsaved edits to other lines are kept (the unsaved changes follow Daryl between devices)."""
    _remember()
    work = _disk_work()
    if any(ln["name"] == line["name"] for ln in work["lines"]):
        work["lines"] = [line if ln["name"] == line["name"] else ln for ln in work["lines"]]
    else:
        work["lines"].append(line)
    CS.save_draft(work)
    st.session_state[SS_MAP] = work


def removed(name: str) -> None:
    _remember()
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
    st.session_state["cm_undo"] = []
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
    c = box.columns([4, 2, 1, 1])
    if c[3].button("↶ Undo", key="cm_undo_btn", disabled=not st.session_state.get("cm_undo"), use_container_width=True,
                   help="Undo the last change (unsaved changes only)."):
        undo()
        st.session_state["cm_msg"] = ("info", "↶ Undone.")
        st.rerun()
    bits = ([f"✎ {len(ch['changed'])} changed"] if ch["changed"] else []) + \
           ([f"➕ {len(ch['added'])} added"] if ch["added"] else []) + ([f"🗑 {len(ch['deleted'])} deleted"] if ch["deleted"] else [])
    if not n:
        c[0].caption("✅ Everything is saved.")
    else:
        c[0].markdown("**Not saved yet:** " + " · ".join(bits))
        c[0].caption(", ".join(names[:12]) + (" …" if n > 12 else ""))
    if c[1].button("💾 Save the cue map", type="primary", disabled=not n, use_container_width=True, key="cm_save_all"):
        if save_map(current(), " · ".join(bits) + ": " + ", ".join(names[:8])):
            st.session_state["cm_undo"] = []
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
            st.session_state["cm_dialog"] = False
            st.rerun()


def _close_dialog():
    st.session_state["cm_dialog"] = False


@st.dialog("✎ Line", width="large", on_dismiss=_close_dialog)
def line_dialog(names: list) -> None:
    """The line editor as a pop-up (like the artifact's sheet). Changes go into the unsaved changes as you make them."""
    editor(names)
    st.divider()
    c = st.columns(2)
    if c[0].button("✓ Done", key="cm_dlg_done", use_container_width=True, help="Close — the change stays unsaved until 💾."):
        _close_dialog()
        st.rerun()
    if c[1].button("💾 Save the cue map", key="cm_dlg_save", type="primary", use_container_width=True,
                   disabled=not any(pending().values())):
        ch = pending()
        names_ = ch["changed"] + ch["added"] + ch["deleted"]
        if save_map(current(), f"{len(names_)} change(s): " + ", ".join(names_[:8])):
            _close_dialog()
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
            turns = [0, 1, 2, 3]
            tv = int(cue.get("turn") or 0)
            tsel = st.selectbox("Which turn in the round (Medium / Hard repeat phases)", turns,
                                index=turns.index(tv) if tv in turns else 0, key=f"cm_turn_{k}",
                                format_func=lambda v: {0: "auto — start: 1st · end: last", 1: "1st", 2: "2nd", 3: "3rd"}[v])
            if tsel:
                cue["turn"] = tsel
            else:
                cue.pop("turn", None)
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

def _hex(p):
    r, g, b = E.PROFILES[p]["colour"]
    return f"#{r:02x}{g:02x}{b:02x}"


SUR_COLOUR = {"handsoff": "#8b8b96", "override": "#8e1016"}


def _cls(p: dict, cut: set) -> str:
    why = str(p.get("why") or "")
    out = []
    if why.startswith(("callout", "fills")):
        out.append("auto")
    elif why.startswith("🎲"):
        out.append("sur")
    elif why.startswith(("during", "at random", "every")):
        out.append("flex")
    if float(p.get("moved") or 0) > 0.05:
        out.append("clash")
    if why.startswith("straight after"):
        out.append("after")
    if p["line"] in cut:
        out.append("cut")
    return " ".join(out)


def timeline_data(cm: dict, sched: list, got: dict) -> dict:
    """What the component draws: rounds (phases, surprises), every play, the intro / outro, the lines to pick."""
    n = max((ph["round"] for ph in sched), default=0)
    rounds = []
    for r in range(1, n + 1):
        phs = [ph for ph in sched if ph["round"] == r]
        units, turn, lab = {}, {}, {}
        for ph in phs:                                  # Medium's 2nd Slow / Medium / Fast read "Slow 2" … (20 Nov)
            if ph.get("kind", "phase") == "phase":
                u = ph.get("unit", ph["index"])
                if u not in units:
                    turn[ph["profile"]] = turn.get(ph["profile"], 0) + 1
                    units[u] = turn[ph["profile"]]
        rep = {p_ for p_, k_ in turn.items() if k_ > 1}
        for ph in phs:
            k_ = units.get(ph.get("unit", ph["index"]), 1)
            lab[ph["index"]] = PH_LABEL[ph["profile"]] + (f" {k_}" if ph["profile"] in rep else "")
        rounds.append({"r": r, "label": "Last round" if r == n else f"Round {r}", "start": phs[0]["start"], "end": phs[-1]["end"],
                       "phases": [{"profile": ph["profile"], "label": lab[ph["index"]], "start": ph["start"],
                                   "end": ph["end"], "kind": ph.get("kind", "phase"),
                                   "colour": SUR_COLOUR.get(ph.get("kind"), _hex(ph["profile"]))} for ph in phs]})
    by_name = {ln["name"]: ln for ln in cm["lines"]}
    cut = {ln["name"] for ln in cm["lines"] if CS.is_cut(ln)}
    secs = lambda n_: round(CS.played_seconds(by_name[n_]), 2) if n_ in by_name else 0.0          # noqa: E731
    plays = [{"line": p["line"], "time": p["time"], "seconds": p["seconds"], "round": p["round"], "why": p["why"],
              "key": list(p["key"]) if p.get("key") else None, "cls": _cls(p, cut)} for p in got["plays"]]
    cats = CS.categories(cm)
    counts = [{"name": c, "n": sum(1 for ln in cm["lines"] if c in (ln.get("cats") or []))} for c in cats]
    counts = [c for c in counts if c["n"]]
    none_n = sum(1 for ln in cm["lines"] if not ln.get("cats"))
    if none_n:
        counts.append({"name": "(none)", "n": none_n})
    pick = [{"name": ln["name"], "seconds": round(CS.played_seconds(ln), 2), "cats": list(ln.get("cats") or []),
             "cued": bool(ln.get("cues"))} for ln in sorted(cm["lines"], key=lambda ln: ln["name"].lower())]
    return {"rounds": rounds, "plays": plays, "pick": pick, "cats": counts, "selected": st.session_state.get(SS_SEL),
            "intro": [{"line": n_, "seconds": secs(n_)} for n_ in got["intro"]],
            "outro": [{"line": n_, "seconds": secs(n_)} for n_ in got["outro"]]}


def _phase_cue_at(sched: list, rnd: int, t: float, n_rounds: int, diff: str = "", split: bool = False) -> dict:
    """A start-of-phase cue that plays a line at compile time t (in round rnd): the phase there, seconds into it — and,
    when the phase comes round more than once in the round (Medium's Slow / Medium / Fast), which turn."""
    stretch = [ph for ph in sched if ph["round"] == rnd and ph.get("kind", "phase") == "phase"]
    ph = next((x for x in stretch if x["start"] <= t < x["end"]), None) or (stretch[-1] if t >= stretch[-1]["end"] else stretch[0])
    occ = [x for x in stretch if x["profile"] == ph["profile"]]
    cue = {"type": "phase", "phases": [ph["profile"]], "at": "start", "offset": round(max(t - ph["start"], 0.0), 1),
           "roundList": ["last" if rnd == n_rounds else str(rnd)], "diffs": [diff] if split and diff else [], "note": ""}
    if len(occ) > 1:
        cue["turn"] = occ.index(ph) + 1
    return cue


def _cue_rounds(cue: dict):
    """The round labels ("1"…"6", "last") a cue plays in, or None when its rounds can't be listed that way."""
    t = cue.get("type")
    if t in ("phase", "random", "interval", "time"):
        rl = {str(x) for x in cue.get("roundList") or []}
        return set(ROUND_OPTS) if not rl or {"1", "2", "3", "4", "5", "6"} <= rl else rl
    if t == "round":
        which = cue.get("which") or "every"
        return {"first": {"1"}, "last": {"last"}}.get(which) or ({str(int(cue.get("number") or 1))} if which == "number" else None)
    return None


def apply_action(cm: dict, act: dict, sched: list, diff: str = "") -> str | None:
    """A timeline action into the unsaved changes. Returns what to tell the user."""
    n = max((ph["round"] for ph in sched), default=0)
    by_name = {ln["name"]: ln for ln in cm["lines"]}
    name = act.get("line")
    a = act.get("act")
    if a == "screen_order":
        which = act.get("which")
        for k, nm in enumerate(act.get("order") or []):
            if nm in by_name:
                ln = copy.deepcopy(by_name[nm])
                for c in ln.get("cues") or []:
                    if c.get("type") == which:
                        c["order"] = k + 1
                changed(cm, ln)
        return f"↔ {str(which).title()} order changed — not saved yet."
    line = copy.deepcopy(by_name.get(name)) if name in by_name else None
    if line is None:
        return None
    if a == "drop":
        cue = _phase_cue_at(sched, int(act["round"]), float(act["time"]), n, diff, CS.is_split(cm))
        line.setdefault("cues", []).append(cue)
        changed(cm, line)
        return f"➕ {name}: {CS.describe(cue)} — not saved yet."
    if a == "move":
        key = act.get("key")
        i = int(key[1]) if key and key[0] == name and int(key[1]) < len(line.get("cues") or []) else None
        new = _phase_cue_at(sched, int(act["round"]), float(act["time"]), n, diff, CS.is_split(cm))
        if i is None:
            line.setdefault("cues", []).append(new)
        else:
            # 🔧 (19 Nov) a drag moves only the round it was dragged from: a cue set for several / every round keeps
            # its other rounds, and the moved round gets a cue of its own — before, the whole cue moved, so a line
            # set for every round turned up at the new spot in every round
            old = line["cues"][i]
            frm = "last" if int(act.get("from") or 0) == n else str(act.get("from"))
            covers = _cue_rounds(old)
            keep = {k: v for k, v in old.items() if k in ("diffs", "note")}
            moved = {**new, **keep}
            if covers == {frm} or covers is None:          # it only played in that round (or can't be split): moved
                line["cues"][i] = moved
            else:
                line["cues"][i] = {**old, "roundList": sorted(covers - {frm}, key=ROUND_OPTS.index)}
                line["cues"].append(moved)
        changed(cm, line)
        return f"↔ {name} moved — not saved yet."
    if a in ("screen_add", "screen_remove"):
        which = act.get("which")
        if a == "screen_add":
            orders = [int(c.get("order") or 0) for ln in cm["lines"] for c in ln.get("cues") or [] if c.get("type") == which]
            if not any(c.get("type") == which for c in line.get("cues") or []):
                line.setdefault("cues", []).append({"type": which, "order": max(orders, default=0) + 1, "roundList": [],
                                                    "diffs": [], "note": ""})
                changed(cm, line)
            return f"➕ {name} on the {which} — not saved yet."
        if a == "screen_remove":
            line["cues"] = [c for c in line.get("cues") or [] if c.get("type") != which]
            changed(cm, line)
            return f"✕ {name} off the {which} — not saved yet."
    return None


def timeline(cm: dict) -> None:
    from cue_timeline import cue_timeline
    c = st.columns([2, 1, 1, 1])
    diff = c[0].selectbox("Difficulty", DIFF_OPTS, format_func=E.DIFFICULTIES.get, key="cm_tl_diff")
    rounds = c[1].selectbox("Rounds", [3, 4, 5, 6], index=0, key="cm_tl_rounds")
    seed = int(c[2].number_input("Draw", 0, 9999, 0, key="cm_tl_seed", help="Another random draw (Hard's phases, "
                                                                             "surprises, random lines)."))
    surp = c[3].toggle("🎲 Surprises", key="cm_tl_surp", disabled=diff == "hard")
    sched = E.schedule(diff, rounds, seed, surprises=surp and diff != "hard")
    got = E.resolve_lines(sched, cm, diff, seed)
    ev = cue_timeline({**timeline_data(cm, sched, got), "readonly": True}, key="cm_tl")
    if isinstance(ev, dict) and ev.get("nonce") and ev["nonce"] != st.session_state.get("cm_tl_nonce"):
        st.session_state["cm_tl_nonce"] = ev["nonce"]
        if ev.get("act") == "section":                  # 🔲 open that section in the Sections view (20 Nov)
            secs = sections(sched, got)
            t, r = float(ev.get("time") or 0), int(ev.get("round") or 1)
            sec = next((x for x in secs if x["round"] == r and x["start"] <= t < x["end"]), None) or \
                next((x for x in secs if x["round"] == r), None)
            if sec:
                st.session_state["cs_sel"] = (sec["round"], sec["i"])
                st.session_state["cs_mode"] = "section"
                st.session_state["cs_diff"] = diff
                st.session_state["cs_rounds"] = rounds
                st.session_state["cm_view_next"] = "🔲 Sections"
            st.rerun()
        if ev.get("act") == "open":                     # opens in this same run (the page calls the pop-up last) —
            select(ev.get("line"))                      # no second reload of the whole page (Daryl, 19 Nov: slow)
            st.session_state["cm_dialog"] = True
        else:
            msg = apply_action(cm, ev, sched, diff)
            if msg:
                st.session_state["cm_msg"] = ("info", msg)
            st.rerun()
    if got["clashes"]:
        with st.expander(f"⚠️ {len(got['clashes'])} clash(es) in this draw"):
            for k, x in enumerate(got["clashes"]):
                cols = st.columns([5, 1])
                cols[0].write(f"**{x['line']}** — {x['where']}: {x['result']} (with {x['with']})")
                if cols[1].button("✎", key=f"cm_clash_{k}"):
                    select(x["line"])
                    st.session_state["cm_dialog"] = True
                    st.rerun()
    else:
        st.caption("Clashes: none — no two lines meet.")


# ---------------------------------------------------------------------------
# 🎙️ Lines
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 🎚️ Which difficulties a line plays in (Daryl, 20 Nov: a Lines filter to see which lines, and how many, each
# difficulty uses)
# ---------------------------------------------------------------------------

def line_diffs(line: dict, by_name: dict, cm: dict | None = None, _seen=None) -> set:
    """The difficulties a line can play in, from its cues: a cue with no difficulty picked plays in all of them;
    🎲 surprises only in Chill and Medium (Hard has no surprises); "straight after" follows the line it comes after;
    Hard's phase callouts (edge_mode.CALLOUT_EXTRA) play in Hard with no cue of their own — while Hard's automatic
    fillers are on."""
    every = set(DIFF_OPTS)
    _seen = set(_seen or ()) | {line["name"]}
    out = set()
    for c in line.get("cues") or []:
        mine = set(c.get("diffs") or []) or every
        t = c.get("type")
        if t == "surprise":
            mine &= {"chill", "medium"}
        elif t == "after":
            target = by_name.get(str(c.get("after") or ""))
            mine &= line_diffs(target, by_name, cm, _seen) if target and target["name"] not in _seen else set()
        out |= mine
    if any(line["name"] in names for names in E.CALLOUT_EXTRA.values()) and E.autofill_on(cm or {}, "hard"):
        out.add("hard")
    return out


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
    by_name = {ln["name"]: ln for ln in cm["lines"]}
    plays = {ln["name"]: line_diffs(ln, by_name, cm) for ln in cm["lines"]}
    dc = st.columns([3, 2])
    diff = dc[0].segmented_control(
        "🎚️ Difficulty", ["all"] + DIFF_OPTS, key="cm_fdiff", default="all",
        format_func=lambda d: "All" if d == "all" else f"{E.DIFFICULTIES[d]} {sum(1 for v in plays.values() if d in v)}") or "all"
    dmode = dc[1].segmented_control(
        "Show", ["in", "missing"], key="cm_fdiff_mode", default="in", disabled=diff == "all",
        format_func=lambda m: "Plays in it" if m == "in" else
        f"Not in it {sum(1 for v in plays.values() if diff != 'all' and diff not in v)}") or "in"

    def in_diff(ln):
        return diff == "all" or (diff in plays[ln["name"]]) == (dmode == "in")
    rows = [ln for ln in cm["lines"] if (not q or q.lower() in ln["name"].lower() or q.lower() in (ln.get("notes") or "").lower())
            and in_cats(ln) and (not nocue or not ln.get("cues")) and in_diff(ln)]
    if order == "Name":
        rows.sort(key=lambda ln: ln["name"].lower())
    else:
        rows.sort(key=lambda ln: CS.played_seconds(ln), reverse=order == "Longest")
    ch = pending()
    mark = {n: "✎" for n in ch["changed"]} | {n: "➕" for n in ch["added"]}
    table = [{"": mark.get(ln["name"], ""), "Line": ln["name"], "Plays (s)": round(CS.played_seconds(ln), 2),
              "✂": "✂" if CS.is_cut(ln) else "", "Categories": ", ".join(ln.get("cats") or []),
              "Cues": len(ln.get("cues") or []),
              **{d.title(): "✅" if d in plays[ln["name"]] else "—" for d in DIFF_OPTS},
              "First cue": CS.describe(ln["cues"][0]) if ln.get("cues") else "—",
              "On server": "✅" if line_path(ln) else "—"} for ln in rows]
    ev = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                      key=f"cm_lines_{len(rows)}_{q}_{'_'.join(cats)}_{nocue}_{order}_{diff}_{dmode}", height=360)
    sel = _picked(ev, "rows")
    tok = ("df", rows[sel[0]]["name"]) if sel and sel[0] < len(rows) else None
    if tok and tok != st.session_state.get("cm_df_tok"):            # act on a new tap only
        st.session_state["cm_df_tok"] = tok
        select(tok[1])
        st.session_state["cm_dialog"] = True
        st.rerun()
    st.caption(f"{len(rows)} of {len(cm['lines'])} lines · tap a row to edit it · ✎ changed / ➕ added, not saved yet.")


def _add(cm: dict, name: str, file: str, seconds: float, folder: str) -> None:
    changed(cm, CS.new_line(name, file, seconds, folder))
    select(name)
    st.session_state["cm_dialog"] = True


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
            _drive_mp3s.clear()
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
                ss["cm_dialog"] = True
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


def _picked(ev, what: str):
    """The dataframe's selection (rows / cells), whichever way this Streamlit returns it."""
    try:
        return ev.selection[what]
    except Exception:                                  # noqa: BLE001
        try:
            return ev["selection"][what]
        except Exception:                              # noqa: BLE001
            return None


@st.cache_data(ttl=600, show_spinner=False)
def _drive_mp3s(root: str, subs: tuple) -> list:
    from drive_sync import list_drive_folder
    out = []
    for sub in ("",) + subs:
        files, err = list_drive_folder(f"{root}/{sub}".rstrip("/"))
        out += [{"name": f["name"], "folder": sub, "size": int(f.get("size") or 0)} for f in ([] if err else files)]
    return out


def new_on_drive(cm: dict) -> list:
    """[{"name", "folder", "seconds"}] — MP3s on Drive with no line in the cue map (seconds estimated at 192 kbps)."""
    try:
        from drive_sync import credentials_available
        from planner.edge import voice_folders
        if not credentials_available():
            return []
    except Exception:                                     # noqa: BLE001 — no Drive here: nothing to offer
        return []
    root = str(cm.get("drive_folder") or "scene-labeling/Edge_voiceover").strip("/")
    have = {str(ln.get("file") or "").lower() for ln in cm["lines"]} | {ln["name"].lower() + ".mp3" for ln in cm["lines"]}
    out, seen = [], set()
    for f in _drive_mp3s(root, tuple(voice_folders(root, cm))):
        n = f["name"]
        if n.lower().endswith(".mp3") and n.lower() not in have and n.lower() not in seen and not E.take_stem(Path(n).stem, ""):
            seen.add(n.lower())
            out.append({"name": n, "folder": f["folder"], "seconds": round(int(f.get("size") or 0) / 24000, 1)})
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


# ---------------------------------------------------------------------------
# 🔍 Gaps — plan Medium (and later Hard) section by section (Daryl, 19 Nov)
# ---------------------------------------------------------------------------
CAT_PHASES = {"slow": {"slow"}, "fast": {"fast", "extra"}, "speed": {"medium", "fast", "extra"}, "stop": {"stop"},
              "start": {"wait", "slow"}, "rounds": {"wait"}, "finish": {"extra", "stop"}}


def sections(sched: list, got: dict) -> list:
    """Every phase turn: {"round", "i", "profile", "turn", "start", "end", "lines", "silent", "longest": (a, b)}."""
    out, count, units = [], {}, {}
    for ph in sched:                                     # a phase split by a ✋ hands off / ⏩ override is still one section
        if ph.get("kind", "phase") != "phase" and ph.get("kind") != "override":
            continue
        u = ph.get("unit", ph["index"])
        if u in units:
            units[u]["end"] = max(units[u]["end"], ph["end"])
        else:
            units[u] = {**ph, "kind": "phase"}
    for ph in units.values():
        k = (ph["round"], ph["profile"])
        count[k] = count.get(k, 0) + 1
        a, b = float(ph["start"]), float(ph["end"])
        inside = [p for p in got["plays"] if p["time"] < b and p["time"] + p["seconds"] > a]
        spans = E._free_spans(a, b, [(p["time"], p["time"] + p["seconds"]) for p in inside])
        longest = max(spans, key=lambda s_: s_[1] - s_[0], default=(a, a))
        out.append({"round": ph["round"], "i": sum(1 for x in out if x["round"] == ph["round"]), "profile": ph["profile"],
                    "turn": count[k], "start": a, "end": b, "lines": [p["line"] for p in inside if p["time"] >= a - 0.01],
                    "silent": round(sum(y - x for x, y in spans), 1), "longest": longest})
    return out


def _sec_name(sec: dict, repeated: set) -> str:
    return PH_LABEL[sec["profile"]] + (f" {sec['turn']}" if sec["profile"] in repeated else "")


def candidates(cm: dict, sec: dict, got: dict, look: str, recs: dict) -> tuple:
    """💡 Lines that could fill the section's longest silence: (recorded with a background in the round's look,
    recorded elsewhere / transparent only). Not already heard in that round; not intro / outro / surprise lines;
    short enough. Best first: used in this kind of section before, then a matching category, then longest."""
    import studio as SD
    room = sec["longest"][1] - sec["longest"][0] - 2 * E.LINE_GAP
    in_round = {p["line"] for p in got["plays"] if p["round"] == sec["round"]}
    good, other = [], []
    for ln in cm["lines"]:
        cues = ln.get("cues") or []
        if ln["name"] in in_round or any(c.get("type") in ("intro", "outro", "surprise") for c in cues):
            continue
        secs = CS.played_seconds(ln)
        if secs > room:
            continue
        used = any(sec["profile"] in (c.get("phases") or []) for c in cues)
        cat = any(sec["profile"] in CAT_PHASES.get(str(x).lower(), set()) for x in ln.get("cats") or [])
        score = 2 if used else 1 if cat else 0
        have = recs.get(SD._key(Path(str(ln.get("file") or ln["name"])).stem), [])
        row = {"name": ln["name"], "seconds": round(secs, 1), "score": score, "line": ln,
               "where": ", ".join(sorted({x["look"] + ("" if x["background"] else " (transparent)") for x in have})) or "not recorded"}
        (good if any(x["look"] == look and x["background"] for x in have) else other).append(row)
    key = lambda r: (-r["score"], -r["seconds"])                                    # noqa: E731
    return sorted(good, key=key), sorted(other, key=key)


def medium_tools(cm: dict) -> None:
    """📋 Medium's own copy of Chill, and the automatic-filler switches."""
    ss = st.session_state
    split = CS.is_split(cm)
    with st.expander("📋 Medium's own copy · automatic fillers", expanded=not split):
        st.caption("Medium starts as a copy of Chill: every cue Chill plays gets its own Medium copy (Slow / Medium / Fast "
                   "cues on the round's 1st turn, so the 2nd turns start empty) and a Hard copy (Hard plays as before, "
                   "ready to plan later). Medium-only cues are wiped; the \"at random\" fillers stop playing in Medium (they "
                   "still play in Hard) and show up below as candidates. Intro, outro, surprise and straight-after cues stay "
                   "shared. It all goes into your unsaved changes — 💾 to keep it.")
        if st.button("📋 Wipe Medium and copy Chill into it", key="cm_gp_copy", type="secondary" if split else "primary"):
            _remember()
            new, rep = CS.copy_chill_to_medium(_disk_work())
            CS.save_draft(new)
            ss[SS_MAP] = new
            ss["cm_msg"] = ("info", f"📋 Medium now has its own copy of {rep['copied']} Chill cues ({len(rep['lines'])} lines); "
                                    f"{rep['fillers']} fillers taken out of Medium, {rep['wiped']} Medium-only cues wiped — "
                                    "not saved yet.")
            st.rerun()
        af = dict(cm.get("autofill") or {})
        cols = st.columns(2)
        for k, d in enumerate(("medium", "hard")):
            on = cols[k].toggle(f"🔁 Automatic fillers in {E.DIFFICULTIES[d]}", value=E.autofill_on(cm, d), key=f"cm_gp_af_{d}",
                                help="The app drops a filler / callout into any silence over 18 s (Hard also gets callouts). "
                                     "Off while you plan by hand, so the real gaps show.")
            if on != E.autofill_on(cm, d):
                _remember()
                work = _disk_work()
                work["autofill"] = {**(work.get("autofill") or {}), d: on}
                CS.save_draft(work)
                ss[SS_MAP] = work
                st.rerun()


def gaps_tab(cm: dict) -> None:
    import studio as SD
    from planner.edge import load_looks
    ss = st.session_state
    c = st.columns([2, 1, 1, 1])
    diff = c[0].selectbox("Difficulty", DIFF_OPTS, index=1, format_func=E.DIFFICULTIES.get, key="cm_gp_diff")
    rounds = c[1].selectbox("Rounds", [3, 4, 5, 6], index=3, key="cm_gp_rounds",
                            help="6 shows every round number; the 6th is the Last round.")
    flag = c[2].number_input("A gap is silence of (s)", 4, 30, 10, key="cm_gp_flag")
    surp = c[3].toggle("🎲 Surprises", key="cm_gp_surp", disabled=diff == "hard",
                       help="Surprise lines always play and push others aside — check with them on.")
    medium_tools(cm)
    sched = E.schedule(diff, rounds, 0, surprises=surp and diff != "hard")
    got = E.resolve_lines(sched, cm, diff, 0)
    secs = sections(sched, got)
    n = rounds
    repeated = {p for p in PHASES if sum(1 for x in secs if x["round"] == 1 and x["profile"] == p) > 1}
    cols = [_sec_name(x, repeated) for x in secs if x["round"] == 1]
    table = []
    for r in range(1, n + 1):
        row = {"Round": "Last" if r == n else str(r)}
        for x in (x for x in secs if x["round"] == r):
            lng = x["longest"][1] - x["longest"][0]
            icon = "🔴" if not x["lines"] or lng >= flag else "🟡" if lng >= flag * 0.6 else "🟢"
            row[_sec_name(x, repeated)] = f"{icon} {len(x['lines'])} · {lng:.0f}s"
        table.append(row)
    st.caption("Each cell: lines in that section · its longest silence. 🔴 empty or a silence over your gap · 🟡 getting "
               "there · 🟢 fine. Tap a cell to see what could fill it." + (" Hard's phases are random each draw — plan it "
                                                                          "differently." if diff == "hard" else ""))
    ev = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-cell",
                      key=f"cm_gp_grid_{diff}_{rounds}_{surp}")
    cells = _picked(ev, "cells")
    reds = [(r, x) for r in range(1, n + 1) for x in secs if x["round"] == r and
            (not x["lines"] or x["longest"][1] - x["longest"][0] >= flag)]
    st.caption(f"🔴 {len(reds)} gap(s) of {len(secs)} sections.")
    sel = None
    if cells:
        cell = cells[0]
        ri, colname = (cell[0], cell[1]) if isinstance(cell, (list, tuple)) else (cell.get("row"), cell.get("column"))
        if colname in cols and ri is not None and ri < n:
            sel = next((x for x in secs if x["round"] == ri + 1 and _sec_name(x, repeated) == colname), None)
    if sel:
        looks = load_looks(cm)
        part = "last" if sel["round"] == n else str(sel["round"])
        look = looks.get(part) or ""
        a, b = sel["longest"]
        st.markdown(f"#### {'Last round' if sel['round'] == n else 'Round ' + str(sel['round'])} · {_sec_name(sel, repeated)} "
                    f"— longest silence {b - a:.1f} s (from {a - sel['start']:.1f} s in) · look: {look or '—'}")
        st.caption("In it now: " + (", ".join(sel["lines"]) or "nothing"))
        recs = SD.recordings(SD.load_clips()["clips"])
        good, other = candidates(cm, sel, got, look, recs)
        if not SD.load_clips()["clips"]:
            st.warning("The Avatar Studio hasn't read Drive yet — open it and press 🔄 Read Drive, so recorded clips can be found.")
        st.markdown(f"**💡 Easy wins — recorded with a background in {look or 'this look'}** ({len(good)})")
        _cand_list(good[:15], sel, n, diff, sched, "g")
        with st.expander(f"Not recorded in this look yet ({len(other)}) — would need recording"):
            _cand_list(other[:25], sel, n, diff, sched, "o")
    if reds:
        txt = "\n".join(f"{'Last' if r == n else 'Round ' + str(r)} · {_sec_name(x, repeated)}: "
                        f"{x['longest'][1] - x['longest'][0]:.0f} s silent, {len(x['lines'])} line(s)" for r, x in reds)
        st.download_button("📋 The gaps as text", txt.encode("utf-8"), file_name=f"{diff}_gaps.txt", key="cm_gp_dl")


def _cand_list(rows: list, sec: dict, n: int, diff: str, sched: list, tag: str) -> None:
    from planner.edge import line_file
    if not rows:
        st.caption("None that fit.")
    for k, r in enumerate(rows):
        cols = st.columns([4, 2, 1])
        stars = "★★" if r["score"] == 2 else "★" if r["score"] == 1 else ""
        cols[0].write(f"**{r['name']}** · {r['seconds']} s {stars}")
        cols[1].caption(r["where"])
        if cols[2].button("➕ Here", key=f"cm_gp_add_{tag}_{k}_{sec['round']}_{sec['i']}"):
            line = copy.deepcopy(r["line"])
            cue = _phase_cue_at(sched, sec["round"], sec["longest"][0] + E.LINE_GAP, n, diff, CS.is_split(current()))
            line.setdefault("cues", []).append(cue)
            changed(current(), line)
            st.session_state["cm_msg"] = ("info", f"➕ {r['name']}: {CS.describe(cue)} — not saved yet.")
            st.rerun()
        pk = f"{tag}_{sec['round']}_{sec['i']}_{r['name']}"
        if tag == "g" and line_file(r["line"], cut=False) is not None:
            if st.session_state.get("cm_gp_play") == pk:     # one player at a time, only when asked for
                f = line_file(r["line"], cut=False)
                with cols[0]:
                    st.audio(str(f), start_time=float(r["line"].get("trim_in") or 0),
                             end_time=float(r["line"].get("trim_out") or 0) or None, autoplay=True)
            elif cols[0].button("▶ Listen", key=f"cm_gp_pl_{tag}_{k}_{sec['round']}_{sec['i']}"):
                st.session_state["cm_gp_play"] = pk
                st.rerun()
    st.caption("★★ already used in this kind of section · ★ its category fits.")
