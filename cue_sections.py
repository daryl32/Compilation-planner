"""🔲 Cue Map → Sections (20 Nov 2026): plan by section, every edit a tap.

Daryl found dragging lines on the timeline unreliable on the iPad and had no quick way to take a line out of one
spot, and asked for a UI rework. From his mockup review (section grid + side panel, 0.25 s steps):

  the grid     one card per section (rounds × Wait … Stop; Medium's 2nd Slow / Medium / Fast), its lines and how much
               is free — Full / Room / Gap; tap a card (components/cue_grid, tap only)
  the panel    beside the grid: the section's own mini timeline, each line with ▶, start − / + 0.25 s, Section start /
               After previous / Ends at the end, Move to…, ✕ Remove from here, ✎ line details; ← / → to the neighbouring
               sections; + Add a line (easy wins first, then lines that need recording, then all lines)
  Every change is for this round only (a cue that plays in other rounds keeps them), goes into the unsaved changes
  (↶ Undo in the save bar) and shows straight away; 💾 Save the cue map keeps it.
"""

from __future__ import annotations

import copy
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import cue_map_view as V
import cue_store as CS
import edge_mode as E

_GRID = components.declare_component("cue_grid", path=str(Path(__file__).resolve().parent / "components" / "cue_grid"))
STEP = 0.25
SS_SEL, SS_MODE = "cs_sel", "cs_mode"
DARK_TEXT = {"slow", "medium", "fast"}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def model(cm: dict, diff: str, rounds: int, surp: bool, seed: int = 0) -> dict:
    sched = E.schedule(diff, rounds, seed, surprises=surp and diff != "hard")
    got = E.resolve_lines(sched, cm, diff, seed)
    secs = V.sections(sched, got)
    repeated = {p for p in V.PHASES if sum(1 for x in secs if x["round"] == 1 and x["profile"] == p) > 1}
    return {"sched": sched, "got": got, "secs": secs, "repeated": repeated, "n": rounds, "diff": diff}


def status(sec: dict, flag: float) -> str:
    lng = sec["longest"][1] - sec["longest"][0]
    if not sec["lines"] or lng >= flag:
        return "gap"
    return "room" if lng >= flag * 0.6 else "full"


def round_label(r: int, n: int) -> str:
    return "Last" if r == n else f"R{r}"


def sec_plays(M: dict, sec: dict) -> list:
    """The plays that start inside the section, in time order."""
    return sorted((p for p in M["got"]["plays"] if sec["start"] - 0.01 <= p["time"] < sec["end"] - 0.01),
                  key=lambda p: p["time"])


def find(M: dict, r: int, i: int):
    return next((x for x in M["secs"] if x["round"] == r and x["i"] == i), None)


def grid_data(M: dict, flag: float, sel, compact: bool) -> dict:
    first = [x for x in M["secs"] if x["round"] == 1]
    heads = [{"label": V._sec_name(x, M["repeated"]), "bg": V._hex(x["profile"]),
              "fg": "#14110b" if x["profile"] in DARK_TEXT else "#ffffff"} for x in first]
    rounds = []
    for r in range(1, M["n"] + 1):
        cells = []
        for x in (x for x in M["secs"] if x["round"] == r):
            ps = sec_plays(M, x)
            lng = x["longest"][1] - x["longest"][0]
            cells.append({"i": x["i"], "name": V._sec_name(x, M["repeated"]), "status": status(x, flag),
                          "lines": [{"name": p["line"], "auto": str(p["why"]).startswith(("callout", "fills"))} for p in ps],
                          "free": f"{lng:.0f} s free", "sel": bool(sel and sel == (r, x["i"]))})
        rounds.append({"r": r, "label": round_label(r, M["n"]), "cells": cells})
    return {"heads": heads, "rounds": rounds, "compact": compact}


# ---------------------------------------------------------------------------
# Edits (all into the unsaved changes, this round only unless asked)
# ---------------------------------------------------------------------------

def _line(cm: dict, name: str) -> dict | None:
    ln = next((x for x in cm["lines"] if x["name"] == name), None)
    return copy.deepcopy(ln) if ln else None


def _cue_index(line: dict, p: dict):
    key = p.get("key")
    if key and key[0] == line["name"] and int(key[1]) < len(line.get("cues") or []):
        return int(key[1])
    return None


def can_edit(cm: dict, p: dict) -> tuple:
    """(True, "") when the play's cue can be moved / removed per round; else (False, why)."""
    if str(p.get("why") or "").startswith(("callout", "fills")):
        return False, "added by the app (automatic filler) — not a cue of its own"
    line = _line(cm, p["line"])
    i = _cue_index(line, p) if line else None
    if i is None:
        return False, "no cue found for it"
    t = line["cues"][i].get("type")
    if t == "surprise":
        return False, "a 🎲 surprise line — it plays wherever a surprise lands"
    if t == "after":
        return False, f"plays straight after “{line['cues'][i].get('after')}” — edit it in ✎ line details"
    if V._cue_rounds(line["cues"][i]) is None:
        return False, "its cue isn't tied to rounds — edit it in ✎ line details"
    return True, ""


def place(cm: dict, M: dict, p: dict, rnd: int, t: float) -> None:
    """Move this round's play of the line so it starts at compile time t, in round rnd (its other rounds untouched)."""
    V.apply_action(cm, {"act": "move", "line": p["line"], "key": list(p["key"]), "from": p["round"], "round": rnd,
                        "time": t}, M["sched"], M["diff"])
    st.session_state["cs_intent"] = {"line": p["line"], "round": rnd, "time": round(t, 3)}


def move_all(cm: dict, M: dict, p: dict, target: dict) -> None:
    """Move the line's cue to the target section in every round it plays in (same seconds in)."""
    line = _line(cm, p["line"])
    i = _cue_index(line, p)
    new = V._phase_cue_at(M["sched"], target["round"], _free_start(M, target, line), M["n"], M["diff"], CS.is_split(cm))
    old = line["cues"][i]
    cue = {**{k: v for k, v in old.items() if k not in ("ends", "turn", "at", "offset", "phases", "type")},
           "type": "phase", "phases": new["phases"], "at": "start", "offset": new["offset"]}
    if new.get("turn"):
        cue["turn"] = new["turn"]
    if old.get("type") != "phase":
        cue["roundList"] = old.get("roundList") or []
    line["cues"][i] = cue
    V.changed(cm, line)


def remove(cm: dict, M: dict, p: dict) -> str:
    """Take the line out of this round only: a cue that plays in other rounds keeps them."""
    line = _line(cm, p["line"])
    i = _cue_index(line, p)
    cue = line["cues"][i]
    covers = V._cue_rounds(cue)
    lab = "last" if p["round"] == M["n"] else str(p["round"])
    if covers is None or covers <= {lab}:
        line["cues"].pop(i)
        msg = f"✕ {p['line']} taken out (it only played here)"
    else:
        line["cues"][i] = {**cue, "roundList": sorted(covers - {lab}, key=V.ROUND_OPTS.index)}
        msg = f"✕ {p['line']} taken out of {round_label(p['round'], M['n'])} (still plays in its other rounds)"
    V.changed(cm, line)
    return msg + " — not saved yet."


def _free_start(M: dict, sec: dict, line: dict = None) -> float:
    """Where a line goes in a section: the start of its longest free stretch (just after the line before)."""
    a, b = sec["longest"]
    if not sec["lines"]:
        return sec["start"]
    return min(a + E.LINE_GAP, max(sec["start"], b - 0.5))


def add_here(cm: dict, M: dict, sec: dict, name: str) -> str:
    line = _line(cm, name)
    cue = V._phase_cue_at(M["sched"], sec["round"], _free_start(M, sec, line), M["n"], M["diff"], CS.is_split(cm))
    line.setdefault("cues", []).append(cue)
    V.changed(cm, line)
    return f"➕ {name}: {CS.describe(cue)} — not saved yet."


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------

def view(cm: dict) -> None:
    ss = st.session_state
    ss["cs_active_sel"] = SS_SEL
    c = st.columns([2, 1, 1, 1])
    diff = c[0].selectbox("Difficulty", V.DIFF_OPTS, index=1, format_func=E.DIFFICULTIES.get, key="cs_diff")
    rounds = c[1].selectbox("Rounds", [3, 4, 5, 6], index=3, key="cs_rounds", help="6 shows every round; the 6th is the Last round.")
    flag = c[2].number_input("Gap = silence of (s)", 4, 30, 10, key="cs_flag")
    surp = c[3].toggle("🎲 Surprises", key="cs_surp", disabled=diff == "hard",
                       help="Surprise lines always play and push others aside — check with them on.")
    V.medium_tools(cm)
    M = model(cm, diff, rounds, surp)
    sel = ss.get(SS_SEL)
    if sel and (sel[0] > rounds or not find(M, *sel)):
        sel = ss[SS_SEL] = None
    if sel:
        left, right = st.columns([3, 2], gap="medium")
    else:
        left, right = st.container(), None
    with left:
        ev = _GRID(data=grid_data(M, flag, sel, compact=bool(sel)), key=f"cs_grid_{diff}", default=None)
        if isinstance(ev, dict) and ev.get("nonce") and ev["nonce"] != ss.get("cs_nonce"):
            ss["cs_nonce"] = ev["nonce"]
            ss[SS_SEL] = (int(ev["round"]), int(ev["i"]))
            ss[SS_MODE] = "section"
            st.rerun()
        reds = [x for x in M["secs"] if status(x, flag) == "gap"]
        st.caption(f"{len(reds)} gap(s) of {len(M['secs'])} sections.")
        if reds:
            txt = "\n".join(f"{'Last' if x['round'] == rounds else 'Round ' + str(x['round'])} · {V._sec_name(x, M['repeated'])}: "
                            f"{x['longest'][1] - x['longest'][0]:.0f} s silent, {len(x['lines'])} line(s)" for x in reds)
            st.download_button("📋 The gaps as text", txt.encode("utf-8"), file_name=f"{diff}_gaps.txt", key="cs_dl")
    if right is not None:
        with right:
            panel(cm, M, find(M, *sel), flag)


def _sel_key() -> str:
    """Which view's selection the panel works on (🔲 Sections: cs_sel · ✨ Improved timeline: it_sel)."""
    return st.session_state.get("cs_active_sel") or SS_SEL


def _goto(r: int, i: int) -> None:
    st.session_state[_sel_key()] = (r, i)
    st.session_state[SS_MODE] = "section"


def panel(cm: dict, M: dict, sec: dict, flag: float) -> None:
    ss = st.session_state
    n = M["n"]
    name = V._sec_name(sec, M["repeated"])
    rlab = "Last round" if sec["round"] == n else f"Round {sec['round']}"
    mode = ss.get(SS_MODE) or "section"
    box = st.container(border=True)
    with box:
        h = st.columns([4, 1, 1, 1])
        h[0].markdown(f"#### {rlab} · {name}")
        same = [x for x in M["secs"] if x["round"] == sec["round"]]
        k = same.index(sec)
        if h[1].button("←", key="cs_prev", disabled=k == 0, help="The section before"):
            _goto(sec["round"], same[k - 1]["i"])
            st.rerun()
        if h[2].button("→", key="cs_next", disabled=k == len(same) - 1, help="The section after"):
            _goto(sec["round"], same[k + 1]["i"])
            st.rerun()
        if h[3].button("✕", key="cs_close", help="Close the panel"):
            ss[_sel_key()] = None
            st.rerun()
        from planner.edge import load_looks
        look = load_looks(cm).get("last" if sec["round"] == n else str(sec["round"])) or "—"
        plays = sec_plays(M, sec)
        lng = sec["longest"][1] - sec["longest"][0]
        st.caption(f"{sec['end'] - sec['start']:.0f} s · {E.DIFFICULTIES[M['diff']]} · her look: {look} · "
                   f"{len(plays)} line(s) · longest free stretch {lng:.1f} s")
        st.markdown(_mini(sec, plays), unsafe_allow_html=True)
        if mode == "add":
            add_panel(cm, M, sec, look)
        elif isinstance(mode, tuple) and mode[0] == "move":
            move_panel(cm, M, sec, mode[1])
        else:
            _intent_note(M)
            if not plays:
                st.info("Nothing here yet.")
            for j, p in enumerate(plays):
                _line_row(cm, M, sec, plays, j, p)
            st.divider()
            if st.button(f"＋ Add a line here", key="cs_add", type="primary", use_container_width=True):
                ss[SS_MODE] = "add"
                st.rerun()


def _intent_note(M: dict) -> None:
    """After a move: tell when the line couldn't start exactly there (it met another line)."""
    it = st.session_state.pop("cs_intent", None)
    if not it:
        return
    p = next((p for p in M["got"]["plays"] if p["line"] == it["line"] and p["round"] == it["round"]), None)
    if p is None:
        st.warning(f"{it['line']} doesn't play there now — another line holds that moment (see the clashes in ✎ line details).")
    elif abs(p["time"] - it["time"]) > 0.05:
        st.warning(f"{it['line']} met another line, so it starts {p['time'] - it['time']:+.2f} s from where you put it.")


def _mini(sec: dict, plays: list) -> str:
    """The section's own little timeline (HTML)."""
    a, b = sec["start"], sec["end"]
    span = max(b - a, 1.0)
    bars = []
    for p in plays:
        left = (p["time"] - a) / span * 100
        w = max(min(p["seconds"], b - p["time"]) / span * 100, 2)
        auto = str(p["why"]).startswith(("callout", "fills"))
        style = ("background:transparent;border:1px dotted #a8a3b5;color:#a8a3b5" if auto else "background:#f0b95a;color:#1b1408")
        bars.append(f'<div style="position:absolute;left:{left:.2f}%;width:{w:.2f}%;top:8px;height:30px;border-radius:6px;{style};'
                    f'font:600 11px/30px sans-serif;padding:0 5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                    f'{_esc(p["line"])}</div>')
    fa, fb = sec["longest"]
    if fb - fa >= 2:
        bars.append(f'<div style="position:absolute;left:{(fa - a) / span * 100:.2f}%;width:{(fb - fa) / span * 100:.2f}%;top:8px;'
                    f'height:30px;border:1.5px dashed #f2776f;border-radius:6px;color:#f2776f;font:600 11px/27px sans-serif;'
                    f'text-align:center">{fb - fa:.1f} s free</div>')
    ticks = "".join(f'<span>{t:g} s</span>' for t in (0, round(span / 2), round(span)))
    return (f'<div style="position:relative;height:46px;border-radius:9px;background:rgba(127,127,140,.12);'
            f'border:1px solid rgba(127,127,140,.35);overflow:hidden">{"".join(bars)}</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;opacity:.7;margin:2px 0 6px">{ticks}</div>')


def _esc(s) -> str:
    import html
    return html.escape(str(s))


def _line_row(cm: dict, M: dict, sec: dict, plays: list, j: int, p: dict) -> None:
    ss = st.session_state
    from planner.edge import line_file
    ok, why = can_edit(cm, p)
    line = next((x for x in cm["lines"] if x["name"] == p["line"]), None)
    rel = p["time"] - sec["start"]
    kid = f"{sec['round']}_{sec['i']}_{j}_{p['line']}"
    with st.container(border=True):
        top = st.columns([5, 1])
        cue_rounds = ""
        if ok:
            i = _cue_index(_line(cm, p["line"]), p)
            covers = V._cue_rounds(line["cues"][i])
            others = sorted((covers or set()) - {"last" if p["round"] == M["n"] else str(p["round"])}, key=V.ROUND_OPTS.index)
            cue_rounds = (" · also plays in " + ", ".join("Last" if x == "last" else x for x in others)) if others else " · this round only"
        top[0].markdown(f"**{p['line']}** · {p['seconds']:.1f} s · starts **{rel:.2f} s**")
        top[0].caption((why if not ok else p["why"]) + cue_rounds)
        if line is not None and top[1].button("✎", key=f"cs_ed_{kid}", help="Line details: length, categories, notes, every cue"):
            V.select(p["line"])
            ss["cm_dialog"] = True
            st.rerun()
        pk = f"play_{kid}"
        f = line_file(line, cut=False) if line else None
        if ss.get("cs_play") == pk and f is not None:
            st.audio(str(f), start_time=float(line.get("trim_in") or 0), end_time=float(line.get("trim_out") or 0) or None,
                     autoplay=True)
        b = st.columns([1, 1, 1, 2])
        if f is not None and b[0].button("▶", key=f"cs_pl_{kid}", help="Listen"):
            ss["cs_play"] = pk
            st.rerun()
        if not ok:
            return
        if b[1].button("−", key=f"cs_m_{kid}", help=f"Start {STEP:g} s earlier", disabled=rel < STEP - 1e-6):
            place(cm, M, p, sec["round"], p["time"] - STEP)
            st.rerun()
        if b[2].button("+", key=f"cs_p_{kid}", help=f"Start {STEP:g} s later", disabled=p["time"] + STEP >= sec["end"] - 0.05):
            place(cm, M, p, sec["round"], p["time"] + STEP)
            st.rerun()
        if b[3].button("✕ Remove from here", key=f"cs_rm_{kid}"):
            ss["cm_msg"] = ("info", remove(cm, M, p))
            st.rerun()
        q = st.columns(4)
        if q[0].button("Section start", key=f"cs_s0_{kid}"):
            place(cm, M, p, sec["round"], sec["start"])
            st.rerun()
        prev = plays[j - 1] if j > 0 else None
        if q[1].button("After previous", key=f"cs_ap_{kid}", disabled=prev is None):
            place(cm, M, p, sec["round"], prev["time"] + prev["seconds"] + E.LINE_GAP)
            st.rerun()
        if q[2].button("Ends at the end", key=f"cs_ee_{kid}"):
            place(cm, M, p, sec["round"], max(sec["start"], sec["end"] - p["seconds"] - 0.05))
            st.rerun()
        if q[3].button("Move to…", key=f"cs_mv_{kid}"):
            ss[SS_MODE] = ("move", {"line": p["line"], "key": list(p["key"]), "round": p["round"], "time": p["time"],
                                    "seconds": p["seconds"], "why": p["why"]})
            st.rerun()


def move_panel(cm: dict, M: dict, sec: dict, p: dict) -> None:
    ss = st.session_state
    st.markdown(f"**Move “{p['line']}”** ({p['seconds']:.1f} s)")
    line = next((x for x in cm["lines"] if x["name"] == p["line"]), None)
    covers = V._cue_rounds(line["cues"][int(p["key"][1])]) if line and p.get("key") and int(p["key"][1]) < len(line["cues"]) else None
    many = bool(covers and len(covers) > 1)
    scope = st.radio("Move", ["this round only", "every round it plays in"], key="cs_scope", horizontal=True,
                     disabled=not many, help=None if many else "It only plays in this round.")
    st.caption("Same round")
    same = [x for x in M["secs"] if x["round"] == sec["round"]]
    cols = st.columns(3)
    for k, x in enumerate(same):
        free = x["longest"][1] - x["longest"][0]
        fits = free >= p["seconds"] + 2 * E.LINE_GAP or x is sec
        lab = f"{V._sec_name(x, M['repeated'])} · {free:.0f} s free" + ("  (here)" if x is sec else "" if fits else "  · too full")
        if cols[k % 3].button(lab, key=f"cs_to_{x['round']}_{x['i']}", disabled=x is sec or not fits, use_container_width=True):
            _do_move(cm, M, p, x, scope)
            st.rerun()
    st.caption(f"Another round — {V.PH_LABEL[sec['profile']]} sections")
    others = [x for x in M["secs"] if x["round"] != sec["round"] and x["profile"] == sec["profile"]]
    cols = st.columns(3)
    for k, x in enumerate(others):
        free = x["longest"][1] - x["longest"][0]
        fits = free >= p["seconds"] + 2 * E.LINE_GAP
        lab = f"{round_label(x['round'], M['n'])} · {V._sec_name(x, M['repeated'])} · {free:.0f} s"
        if cols[k % 3].button(lab, key=f"cs_tor_{x['round']}_{x['i']}", disabled=not fits, use_container_width=True):
            _do_move(cm, M, p, x, "this round only")
            st.rerun()
    if st.button("Cancel", key="cs_mv_cancel"):
        ss[SS_MODE] = "section"
        st.rerun()


def _do_move(cm: dict, M: dict, p: dict, target: dict, scope: str) -> None:
    ss = st.session_state
    play = next((q for q in M["got"]["plays"] if q["line"] == p["line"] and abs(q["time"] - p["time"]) < 0.01), p)
    if scope == "every round it plays in":
        move_all(cm, M, play, target)
    else:
        line = next(x for x in cm["lines"] if x["name"] == p["line"])
        place(cm, M, play, target["round"], _free_start(M, target, line))
    ss["cm_msg"] = ("info", f"↔ {p['line']} moved to {round_label(target['round'], M['n'])} · "
                            f"{V._sec_name(target, M['repeated'])} — not saved yet.")
    _goto(target["round"], target["i"])


def add_panel(cm: dict, M: dict, sec: dict, look: str) -> None:
    import studio as SD
    ss = st.session_state
    room = sec["longest"][1] - sec["longest"][0]
    st.markdown(f"**＋ Add a line** — into the {room:.1f} s free stretch")
    clips = SD.load_clips()["clips"]
    if not clips:
        st.warning("The Avatar Studio hasn't read Drive yet — open it and press 🔄 Read Drive so recorded clips can be found.")
    good, other = V.candidates(cm, sec, M["got"], look, SD.recordings(clips))
    show = st.segmented_control("Show", [f"Easy wins {len(good)}", f"Needs recording {len(other)}", "All lines"],
                                default=f"Easy wins {len(good)}", key="cs_add_show", label_visibility="collapsed") or "Easy"
    # 🏷 the Cue Map's categories on every list, easy wins too (Daryl, 20 Nov)
    counts = {c: sum(1 for ln in cm["lines"] if c in (ln.get("cats") or [])) for c in CS.categories(cm)}
    pick = st.pills("Categories", [c for c in counts if counts[c]], selection_mode="multi", key="cs_add_cats",
                    format_func=lambda c: f"{c} {counts[c]}") or []
    q = st.text_input("Find a line", key="cs_add_q", placeholder="Find a line")
    keep = lambda ln: (not q or q.lower() in ln["name"].lower()) and (not pick or set(pick) & set(ln.get("cats") or []))   # noqa: E731
    if show.startswith("All"):
        in_round = {p["line"] for p in M["got"]["plays"] if p["round"] == sec["round"]}
        rows = [{"name": ln["name"], "seconds": round(CS.played_seconds(ln), 1), "score": 0, "line": ln,
                 "where": "already in this round" if ln["name"] in in_round else ""}
                for ln in sorted(cm["lines"], key=lambda ln: ln["name"].lower()) if keep(ln)]
    else:
        rows = [r for r in (good if show.startswith("Easy") else other) if keep(r["line"])]
    for k, r in enumerate(rows[:30]):
        cols = st.columns([5, 1])
        stars = "★★" if r["score"] == 2 else "★" if r["score"] == 1 else ""
        cols[0].markdown(f"**{r['name']}** {stars} · {r['seconds']} s")
        if r.get("where"):
            cols[0].caption(r["where"])
        if cols[1].button("Add", key=f"cs_addl_{k}_{r['name']}", disabled=r["seconds"] > room, type="primary"):
            ss["cm_msg"] = ("info", add_here(cm, M, sec, r["name"]))
            ss[SS_MODE] = "section"
            st.rerun()
    if show.startswith("Easy"):
        st.caption("★★ already used in this kind of section · ★ its category fits · recorded with a background in this "
                   "round's look, fits the gap, not already in this round.")
    if st.button("← Back to the section", key="cs_add_back"):
        ss[SS_MODE] = "section"
        st.rerun()


# ---------------------------------------------------------------------------
# ✨ Improved timeline (Daryl, 20 Nov): the zoomable timeline he likes + the phase panel on the right
# ---------------------------------------------------------------------------

def audio_map(cm: dict, names) -> dict:
    """{line: {"url", "start", "end", "speed"}} so the timeline plays a clip itself (▶ Hear) — the line's own file,
    played from its cut start to its cut end at its speed."""
    from cut_editor import media_url
    from planner.edge import line_file
    out = {}
    by = {ln["name"]: ln for ln in cm["lines"]}
    for n in dict.fromkeys(names):
        ln = by.get(n)
        f = line_file(ln, cut=False) if ln else None
        if f is None:
            continue
        url = media_url(f, "audio/mpeg", "cue_timeline_audio")
        if url:
            out[n] = {"url": url, "start": float(ln.get("trim_in") or 0), "end": float(ln.get("trim_out") or 0),
                      "speed": float(ln.get("speed") or 1) or 1}
    return out


def improved_view(cm: dict) -> None:
    from cue_timeline import cue_timeline
    ss = st.session_state
    ss["cs_active_sel"] = "it_sel"
    c = st.columns([2, 1, 1, 1, 1])
    diff = c[0].selectbox("Difficulty", V.DIFF_OPTS, index=1, format_func=E.DIFFICULTIES.get, key="it_diff")
    rounds = c[1].selectbox("Rounds", [3, 4, 5, 6], index=3, key="it_rounds", help="6 shows every round; the 6th is the Last round.")
    seed = int(c[2].number_input("Draw", 0, 9999, 0, key="it_seed", help="Another random draw (Hard's phases, surprises, random lines)."))
    surp = c[3].toggle("🎲 Surprises", key="it_surp", disabled=diff == "hard")
    flag = c[4].number_input("Gap = silence of (s)", 4, 30, 10, key="it_flag")
    V.medium_tools(cm)
    M = model(cm, diff, rounds, surp, seed)
    sel = ss.get("it_sel")
    sec = find(M, *sel) if sel and sel[0] <= rounds else None
    if sel and sec is None:
        ss["it_sel"] = None
    if sec:
        left, right = st.columns([3, 2], gap="medium")
    else:
        left, right = st.container(), None
    with left:
        got = M["got"]
        data = V.timeline_data(cm, M["sched"], got)
        data.update(readonly=True, mode="improved",
                    audio=audio_map(cm, [p["line"] for p in got["plays"]] + got["intro"] + got["outro"]),
                    selPhase={"round": sec["round"], "start": sec["start"], "end": sec["end"]} if sec else None)
        ev = cue_timeline(data, key=f"it_tl_{diff}")
        if isinstance(ev, dict) and ev.get("nonce") and ev["nonce"] != ss.get("it_nonce"):
            ss["it_nonce"] = ev["nonce"]
            act = ev.get("act")
            if act == "open":                                # ✎ Open — the line's full card (opens in this same run)
                V.select(ev.get("line"))
                ss["cm_dialog"] = True
            elif act == "phase":                             # a phase → its panel on the right
                t, r = float(ev.get("time") or 0), int(ev.get("round") or 1)
                hit = next((x for x in M["secs"] if x["round"] == r and x["start"] <= t < x["end"]), None)
                if hit:
                    ss["it_sel"] = (hit["round"], hit["i"])
                    ss[SS_MODE] = "section"
                st.rerun()
            else:                                            # intro / outro order, add, take off
                msg = V.apply_action(cm, ev, M["sched"], diff)
                if msg:
                    ss["cm_msg"] = ("info", msg)
                st.rerun()
        if got["clashes"]:
            with st.expander(f"⚠️ {len(got['clashes'])} clash(es) in this draw"):
                for x in got["clashes"]:
                    st.write(f"**{x['line']}** — {x['where']}: {x['result']} (with {x['with']})")
    if right is not None:
        with right:
            panel(cm, M, sec, flag)
