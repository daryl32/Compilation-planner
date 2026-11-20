"""🧪 Hard lab (Cue Map, 20 Nov 2026 — temporary): try the rules that let Hard follow Chill and Medium.

Daryl wants to try the rules on many random Hard draws, change any part of them and see the timeline update, with
clashes and empty phases highlighted, then bake them into Hard and test-render before this page goes away. Nothing
here changes the cue map until ✅ Bake (which goes into the unsaved changes like any edit: 💾 to keep).
The rules themselves are in hard_rules.py.
"""

from __future__ import annotations

import random

import streamlit as st

import cue_map_view as V
import cue_store as CS
import edge_mode as E
import hard_rules as HR

SS_RULES = "hl_rules"
POOL_LABEL = {"chill": "Chill pool", "medium": "Medium pool"}


def _rules(cm: dict) -> dict:
    """The rules being tried (this session): the baked ones to start with, else the defaults."""
    ss = st.session_state
    if SS_RULES not in ss:
        ss[SS_RULES] = {k: v for k, v in HR.rules_of(cm).items() if k != "on"}
    return ss[SS_RULES]


def _knobs(R: dict) -> None:
    ss = st.session_state
    with st.container(border=True):
        st.markdown("**The rules** — change any of them, the timeline updates straight away")
        c = st.columns(4)
        R["split"] = c[0].slider("Where the halves meet", 0.0, 1.0, float(R["split"]), 0.05, key="hl_split",
                                 help="How far through each round's random middle the second pool takes over. 0.5 = half way.")
        R["first"] = c[1].selectbox("First half from", ["chill", "medium"], index=["chill", "medium"].index(R["first"]),
                                    key="hl_first", format_func=POOL_LABEL.get)
        R["second"] = c[2].selectbox("Second half from", ["chill", "medium"], index=["chill", "medium"].index(R["second"]),
                                     key="hl_second", format_func=POOL_LABEL.get)
        R["max_repeat"] = c[3].number_input("A line at most … times a round", 1, 4, int(R["max_repeat"]), key="hl_rep")
        d = st.columns(4)
        R["per_phase"] = d[0].number_input("Lines per phase (up to)", 1, 3, int(R["per_phase"]), key="hl_per",
                                           help="How many pool lines a phase gets when there's room.")
        R["offset"] = d[1].number_input("First line starts (s into the phase)", 0.0, 5.0, float(R["offset"]), 0.25, key="hl_off")
        R["fallback"] = d[2].toggle("Borrow from the other pool when one runs dry", bool(R["fallback"]), key="hl_fb")
        R["anchors"] = d[3].toggle("Keep Hard's Wait / final Extra / Stop lines", bool(R["anchors"]), key="hl_anc")
        e = st.columns([2, 1])
        R["autofill"] = e[0].toggle("🔁 Then fill leftover silence with callouts / fillers", bool(R["autofill"]), key="hl_auto")
        if e[1].button("↺ Back to the defaults", key="hl_reset", use_container_width=True):
            ss[SS_RULES] = {k: v for k, v in HR.DEFAULTS.items() if k != "on"}
            for k in ("hl_split", "hl_first", "hl_second", "hl_rep", "hl_per", "hl_off", "hl_fb", "hl_anc", "hl_auto"):
                ss.pop(k, None)
            st.rerun()


def view(cm: dict) -> None:
    from cue_timeline import cue_timeline
    ss = st.session_state
    st.caption("🧪 A temporary page: try how Hard could follow Chill and Medium. Hard's random middle draws its lines from "
               "what Chill and Medium play in the same round and kind of phase, so few new clips need recording. Nothing "
               "changes until you ✅ bake the rules into Hard.")
    baked = HR.rules_of(cm).get("on")
    st.info("Hard now uses: **these rules (baked)**" if baked else "Hard now uses: **its own copy of Chill's cues** (rules not baked)")
    R = _rules(cm)
    _knobs(R)
    c = st.columns([1, 1, 1, 2])
    rounds = c[0].selectbox("Rounds", [3, 4, 5, 6], index=2, key="hl_rounds")
    seed = int(c[1].number_input("Draw", 0, 99999, int(ss.get("hl_seed_v", 0)), key="hl_seed"))
    if c[2].button("🎲 New draw", key="hl_new", use_container_width=True):
        ss["hl_seed_v"] = random.randint(0, 99999)
        ss.pop("hl_seed", None)
        st.rerun()
    sched = E.schedule("hard", rounds, seed)
    got = HR.resolve(sched, cm, seed, None, R)
    H = got["hard"]
    data = V.timeline_data(cm, sched, got)
    unf = {(u["round"], u["start"]) for u in H["unfilled"]}
    for Rr in data["rounds"]:
        Rr["split"] = H["split"].get(Rr["r"])
        for ph in Rr["phases"]:
            ph["unf"] = (Rr["r"], ph["start"]) in unf
    for dp, p in zip(data["plays"], got["plays"]):
        pool = p.get("pool")
        extra = {"chill": "pchill", "medium": "pmed", "anchor": "anchor"}.get(pool, "")
        dp["cls"] = (dp["cls"] + " " + extra).strip()
    sel = ss.get("hl_phase")
    data.update(readonly=True, mode="improved",
                audio=_audio(cm, got), selPhase=sel)
    n_mid = sum(len(v) for v in HR.middles(sched).values())
    by_pool = {k: sum(1 for p in got["plays"] if p.get("pool") == k) for k in ("anchor", "chill", "medium")}
    auto = sum(1 for p in got["plays"] if str(p.get("why") or "").startswith(("callout", "fills")))
    c[3].markdown(f"**{n_mid - len(H['unfilled'])} of {n_mid}** random phases get a pool line · "
                  f"⚓ {by_pool['anchor']} anchors · 🔵 {by_pool['chill']} Chill · 🟠 {by_pool['medium']} Medium · 🔁 {auto} fillers")
    st.markdown('<div style="display:flex;gap:16px;flex-wrap:wrap;font-size:13px;opacity:.85">'
                '<span><b style="color:#4fb3c8">■</b> from the Chill pool</span><span><b style="color:#f0b95a">■</b> from the Medium pool</span>'
                '<span><b style="color:#c9c3d6">■</b> anchor (Hard\'s own cue)</span><span>┊ dotted: an automatic filler / callout</span>'
                '<span><b style="color:#f2776f">▢!</b> a phase that got no pool line</span><span>╎ where the halves meet</span>'
                '<span><b style="color:#f2776f">■</b> red line: a clash</span></div>', unsafe_allow_html=True)
    left, right = (st.columns([3, 2], gap="medium") if sel else (st.container(), None))
    with left:
        ev = cue_timeline(data, key="hl_tl")
        if isinstance(ev, dict) and ev.get("nonce") and ev["nonce"] != ss.get("hl_nonce"):
            ss["hl_nonce"] = ev["nonce"]
            if ev.get("act") == "open":
                V.select(ev.get("line"))
                ss["cm_dialog"] = True
            elif ev.get("act") == "phase":
                t, r = float(ev.get("time") or 0), int(ev.get("round") or 1)
                ph = next((x for x in sched if x["round"] == r and x["start"] <= t < x["end"]), None)
                if ph:
                    ss["hl_phase"] = {"round": r, "start": ph["start"], "end": ph["end"], "profile": ph["profile"]}
                st.rerun()
    if right is not None:
        with right:
            _phase_info(cm, sched, got, sel, rounds)
    _clashes(got)
    _check(cm, R, rounds)
    _bake(cm, R, baked)


def _audio(cm: dict, got: dict) -> dict:
    import cue_sections as SEC
    return SEC.audio_map(cm, [p["line"] for p in got["plays"]] + got["intro"] + got["outro"])


def _phase_info(cm: dict, sched: list, got: dict, sel: dict, rounds: int) -> None:
    ss = st.session_state
    H = got["hard"]
    r = sel["round"]
    with st.container(border=True):
        h = st.columns([5, 1])
        h[0].markdown(f"#### {'Last round' if r == rounds else f'Round {r}'} · {E.PROFILES[sel['profile']]['label'].title()} "
                      f"({sel['end'] - sel['start']:.0f} s)")
        if h[1].button("✕", key="hl_close"):
            ss["hl_phase"] = None
            st.rerun()
        mid = any(ph["start"] == sel["start"] for ph in HR.middles(sched).get(r, []))
        R = _rules(cm)
        if not mid:
            st.caption("⚓ An anchor phase: it keeps Hard's own cues (Wait, final Extra, Stop).")
        else:
            src = R["first"] if sel["start"] < H["split"].get(r, 0) else R["second"]
            st.caption(f"In the {'first' if src == R['first'] else 'second'} half → draws from the **{POOL_LABEL[src]}**.")
        here = [p for p in got["plays"] if sel["start"] <= p["time"] < sel["end"]]
        st.markdown("**Plays here:** " + (", ".join(f"{p['line']} ({p['seconds']:.1f} s, {p.get('pool') or 'filler'})" for p in here) or "nothing"))
        u = next((u for u in H["unfilled"] if u["round"] == r and u["start"] == sel["start"]), None)
        if u:
            st.warning(f"No pool line here: {u['why']}.")
        if mid:
            for s_ in ("chill", "medium"):
                lst = H["pools"].get(s_, {}).get(r, {}).get(sel["profile"], [])
                by = {ln["name"]: ln for ln in cm["lines"]}
                n_r = {}
                for p in got["plays"]:
                    if p["round"] == r:
                        n_r[p["line"]] = n_r.get(p["line"], 0) + 1
                st.markdown(f"**{POOL_LABEL[s_]}** — {E.PROFILES[sel['profile']]['label'].title()} in round {r} ({len(lst)})")
                st.caption(" · ".join(f"{x} {CS.played_seconds(by[x]):.1f} s" + (f" ×{n_r[x]}" if n_r.get(x) else "")
                                      for x in lst if x in by) or "empty")


def _clashes(got: dict) -> None:
    if got["clashes"]:
        with st.expander(f"⚠️ {len(got['clashes'])} clash(es) among the anchors in this draw"):
            for x in got["clashes"]:
                st.write(f"**{x['line']}** — {x['where']}: {x['result']} (with {x['with']})")


def _check(cm: dict, R: dict, rounds: int) -> None:
    ss = st.session_state
    with st.container(border=True):
        c = st.columns([3, 1])
        c[0].markdown("**🔍 Check many draws** — how the rules hold up across random Hard compiles")
        draws = c[1].selectbox("Draws", [25, 50, 100], index=1, key="hl_draws", label_visibility="collapsed")
        if st.button(f"Run {draws} draws", key="hl_check"):
            with st.spinner("Trying the rules…"):
                ss["hl_check_res"] = (dict(R), rounds, HR.check(cm, R, rounds, draws))
        res = ss.get("hl_check_res")
        if res:
            r_, n_, out = res
            stale = r_ != R or n_ != rounds
            st.caption(("⚠️ The rules or rounds have changed since — run it again. " if stale else "")
                       + f"{out['draws']} draws of {n_} rounds")
            m = st.columns(3)
            m[0].metric("Random phases with a pool line", f"{out['filled_pct']} %")
            m[1].metric("Longest silence (avg / worst)", f"{out['longest_avg']} / {out['longest_max']} s")
            m[2].metric("Lines heard twice in a round, per draw", out["repeats_per_draw"])
            if out["short"]:
                st.markdown("**Where it runs short** — the phases that most often got no pool line (worth recording for):")
                st.dataframe([{"Round": a, "Phase": E.PROFILES[b]["label"].title(), "Draws short": v,
                               "% of draws": round(100 * v / out["draws"])} for a, b, v in out["short"][:12]],
                             hide_index=True, use_container_width=True)


def _bake(cm: dict, R: dict, baked: bool) -> None:
    ss = st.session_state
    with st.container(border=True):
        st.markdown("**✅ Bake** — when you're happy: every Hard compile then uses these rules (overnight renders too). "
                    "It goes into the unsaved changes; 💾 Save the cue map to keep it. Test-render Hard afterwards.")
        c = st.columns(2)
        same = baked and {k: v for k, v in HR.rules_of(cm).items() if k != "on"} == R
        if c[0].button("✅ Bake these rules into Hard" if not baked else ("✅ Baked" if same else "✅ Bake the changed rules"),
                       key="hl_bake", type="primary", disabled=same, use_container_width=True):
            V._remember()
            work = HR.bake(V._disk_work(), R)
            CS.save_draft(work)
            ss[V.SS_MAP] = work
            ss["cm_msg"] = ("info", "✅ Rules baked into Hard — not saved yet: 💾 Save the cue map, then test-render Hard.")
            st.rerun()
        if baked and c[1].button("↩ Hard back to its own cues", key="hl_unbake", use_container_width=True):
            V._remember()
            work = HR.unbake(V._disk_work())
            CS.save_draft(work)
            ss[V.SS_MAP] = work
            ss["cm_msg"] = ("info", "Hard is back on its own cues — not saved yet.")
            st.rerun()
