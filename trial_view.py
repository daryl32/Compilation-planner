"""🧪 Trial results view for the Training Labels page (pages/6_Training_Labels.py): the latest
Modal trial — each tagged frame with your tags, the old tags and every model's tags side by
side; motion curves and scene cuts per video; the report. Data: trial_results.py."""

from __future__ import annotations

import hashlib
import html

import streamlit as st

from library_common import safe_image, format_mmss
import training_labels as TLB
import trial_results as TR

CHIP_STYLE = {
    TR.RIGHT: ("✓", "rgba(46,160,67,.18)", "rgba(46,160,67,.75)", ""),
    TR.WRONG: ("✗", "rgba(229,83,75,.16)", "rgba(229,83,75,.8)", ""),
    TR.MISSED: ("+", "transparent", "rgba(128,128,128,.7)", "border-style:dashed;opacity:.85;"),
    TR.NEUTRAL: ("", "rgba(128,128,128,.12)", "rgba(128,128,128,.4)", ""),
}
PER_PAGE = 12


def _k(video_id: str, moment: dict) -> str:
    """A short, safe widget key for one frame (video names have spaces and punctuation)."""
    return hashlib.md5(f"{video_id}|{moment['key']}".encode()).hexdigest()[:12]


def _chip(tag: str, status: str, faded: bool = False) -> str:
    mark, bg, border, extra = CHIP_STYLE[status]
    title = {TR.RIGHT: "matches your tags", TR.WRONG: "not in your tags",
             TR.MISSED: "you have it — this one missed it", TR.NEUTRAL: ""}[status]
    if faded:
        extra += "opacity:.55;font-style:italic;"
        title = "unsure"
    return (f'<span title="{title}" style="display:inline-block;margin:2px 4px 2px 0;padding:1px 9px;'
            f'border-radius:999px;border:1px solid {border};background:{bg};font-size:.86rem;{extra}">'
            f'{mark + " " if mark else ""}{html.escape(tag)}{"?" if faded else ""}</span>')


def _row(label: str, pred, mine, unsure=()) -> str:
    parts = [_chip(t, s) for t, s in TR.chips(pred, mine)]
    parts += [_chip(t, TR.NEUTRAL, faded=True) for t in unsure or []]
    body = "".join(parts) or '<span style="opacity:.55">no tags</span>'
    return (f'<div style="margin:3px 0"><span style="display:inline-block;min-width:4.6rem;font-weight:600;'
            f'font-size:.86rem">{label}</span>{body}</div>')


def _save(card: dict, tags: list, key: str) -> None:
    TLB.save_label(card, tags=tags)
    st.session_state.pop(key, None)        # close the editor
    st.session_state["tr_saved_note"] = f"Saved your tags for {card['video_id'][:40]} at {format_mmss(card['t'])}."


def _editor(vid: str, m: dict, teachers: list, ekey: str) -> None:
    """Fix the tags for one frame: pick-one rows and grouped chips from the vocabulary, starting
    from your tags (or the best model's)."""
    vocab, _ = TLB.load_vocabulary()
    names = {TR.norm(r["tag"]): r["tag"] for r in vocab}
    start = m.get("mine")
    if start is None:
        r = next(((m.get("teachers") or {}).get(k) for k in reversed(teachers) if (m.get("teachers") or {}).get(k)),
                 None)
        start = (r or {}).get("tags") or m.get("old") or []
    start = [names.get(TR.norm(t), t) for t in start]
    picked = []
    for gi, (title, single, tags) in enumerate(TLB.chip_groups(vocab, extra=start)):
        default = [t for t in start if t in tags]
        wkey = f"{ekey}_g{gi}"
        if single:
            val = st.pills(title, tags, selection_mode="single", default=default[0] if default else None, key=wkey)
            picked += [val] if val else []
        else:
            picked += list(st.pills(title, tags, selection_mode="multi", default=default, key=wkey) or [])
    card = TR.correction_card(vid, m, start)
    c1, c2 = st.columns(2)
    c1.button("💾 Save my tags", type="primary", key=f"{ekey}_save", width="stretch",
              on_click=_save, args=(card, picked, ekey))
    c2.button("Cancel", key=f"{ekey}_cancel", width="stretch",
              on_click=lambda: st.session_state.pop(ekey, None))


def _compare(data: dict) -> None:
    videos = TR.with_saved_labels(data["videos"], TLB.load_labels())
    note = st.session_state.pop("tr_saved_note", None)
    if note:
        st.toast(note, icon="💾")
    teachers = TR.shown_teachers(data.get("teachers") or {})
    filters = {k: v for k, v in TR.FILTERS.items() if k != "disagree" or len(teachers) > 1}
    if st.session_state.get("tr_filter") not in filters:
        st.session_state["tr_filter"] = "all"
    with st.container(key="tl_wrap_tr_filters"):
        c = st.columns([3, 2, 2, 2])
    vid = c[0].selectbox("Video", ["(all videos)"] + sorted(videos), key="tr_vid")
    how = c[1].selectbox("Show", list(filters), format_func=filters.get, key="tr_filter")
    order = c[3].selectbox("Order", ["video", "random"], key="tr_order",
                           format_func={"video": "By video and time", "random": "🎲 Random"}.get,
                           help="Random mixes frames from every video, so a review isn't stuck in one video.")
    all_tags = sorted({t for v in videos.values() for m in v["moments"]
                       for t in (m.get("mine") or []) + (m.get("old") or [])
                       + [x for r in m.get("teachers", {}).values() for x in r.get("tags") or []]},
                      key=str.lower)
    tag = c[2].selectbox("Tag", ["(any tag)"] + all_tags, key="tr_tag")
    pool = [(v, m) for v in (sorted(videos) if vid == "(all videos)" else [vid]) for m in videos[v]["moments"]]
    moments = [(v, m) for v, m in pool
               if TR.filter_moments([m], how, None if tag == "(any tag)" else tag)]
    if order == "random":
        def _reshuffle():
            st.session_state["tr_seed"] = st.session_state.get("tr_seed", 0) + 1
            st.session_state["tr_page"] = 1
        st.button("🎲 Shuffle again", key="tr_shuffle", on_click=_reshuffle,
                  help="A new random sample. Frames you save drop out of “Not reviewed by me yet”.")
        moments = TR.ordered(moments, "random", st.session_state.get("tr_seed", 0))

    agr = TR.agreement([m for _, m in pool])
    if agr:
        bits = []
        for k in teachers:
            a = agr.get(k)
            if a:
                bits.append(f"**{k}** exactly right on {a['exact']} of {a['frames']} of your frames · "
                            f"found {0 if a['recall'] is None else a['recall'] * 100:.0f}% of your tags · "
                            f"{0 if a['precision'] is None else a['precision'] * 100:.0f}% of its tags right")
        st.caption("  \n".join(bits))
    st.caption("✓ green = matches your tags · ✗ red = not in your tags · + dashed = you have it, that row "
               "missed it · *grey italic?* = the model was unsure. Rows without your tags aren't coloured.")
    if not moments:
        st.info("Nothing matches these filters.")
        return
    pages = max(1, -(-len(moments) // PER_PAGE))
    if st.session_state.get("tr_page", 1) > pages:      # e.g. frames dropped out of "not reviewed"
        st.session_state["tr_page"] = pages
    page = st.number_input(f"Page (of {pages}) — {len(moments)} moments", 1, pages, 1, key="tr_page") \
        if pages > 1 else 1
    for v, m in moments[(page - 1) * PER_PAGE: page * PER_PAGE]:
        with st.container(border=True, key=f"tr_wrap_card_{_k(v, m)}"):
            a, b = st.columns([2, 3])
            with a:
                p = TR.frame_path(m)
                if p is not None and p.exists():
                    safe_image(str(p), width="stretch")
                else:
                    st.caption("🖼️ Frame not synced yet")
                st.caption(f"{v[:48]} · {format_mmss(m['t'])} · scene {m.get('scene_id')}")
            with b:
                mine = m.get("mine")
                rows = []
                if mine is not None:
                    rows.append(_row("You", mine, None))
                rows.append(_row("Old", m.get("old") or [], mine))
                for k in teachers:
                    r = (m.get("teachers") or {}).get(k)
                    rows.append(_row(k.upper(), r.get("tags") if r else [], mine, r.get("unsure") if r else ())
                                if r else f'<div style="opacity:.55;font-size:.86rem">{k.upper()}: not tagged</div>')
                st.markdown("".join(rows), unsafe_allow_html=True)
                caps = [(k, (m.get("teachers") or {}).get(k, {}).get("caption")) for k in teachers]
                caps = [(k, cap) for k, cap in caps if cap]
                if caps:
                    st.caption("  \n".join(f"**{k}:** {cap}" for k, cap in caps))
                # ✏️ Correct: your tags for this frame go to your training labels (the gold set).
                ekey = f"tr_edit_{_k(v, m)}"
                if st.session_state.get(ekey):
                    _editor(v, m, teachers, ekey)
                else:
                    with st.container(key=f"tr_wrap_act_{_k(v, m)}"):
                        acts = [k for k in teachers if (m.get("teachers") or {}).get(k)]
                        cols = st.columns(1 + len(acts))
                        cols[0].button("✏️ Correct", key=f"{ekey}_open", width="stretch",
                                       on_click=lambda k=ekey: st.session_state.__setitem__(k, True),
                                       help="Set the right tags for this frame — saved to your training labels.")
                        for col, k in zip(cols[1:], acts):
                            tags = (m["teachers"][k].get("tags") or [])
                            col.button(f"✅ {k.upper()} is right", key=f"{ekey}_ok_{k}", width="stretch",
                                       on_click=_save,
                                       args=(TR.correction_card(v, m, tags), tags, ekey),
                                       help=f"Save {k.upper()}'s tags as your tags for this frame.")


def _motion(data: dict) -> None:
    import plotly.graph_objects as go
    videos = data["videos"]
    vid = st.selectbox("Video", sorted(videos), key="tr_mvid")
    v = videos[vid]
    fps = float(v.get("curve_fps") or 2.0)
    fig = go.Figure()
    for name, curve, colour in (("People moving (new)", v.get("person_curve"), "#4c78a8"),
                                ("Picture change (old)", v.get("old_curve"), "#bab0ac")):
        if curve:
            fig.add_scatter(x=[i / fps for i in range(len(curve))], y=TR.normalised(curve), name=name,
                            mode="lines", line={"color": colour, "width": 1.6 if "new" in name else 1.1})
    for t in v.get("cuts_new") or []:
        fig.add_vline(x=t, line={"color": "#e45756", "width": 1})
    for t in v.get("cuts_old") or []:
        fig.add_vline(x=t, line={"color": "#888", "width": 1, "dash": "dot"})
    for b in v.get("boundaries") or []:
        fig.add_scatter(x=[b["t"]], y=[1.45], mode="markers", showlegend=False,
                        marker={"symbol": "triangle-down", "size": 10,
                                "color": {"cut": "#54a24b", "no_cut": "#e45756"}.get(b.get("verdict"), "#999")},
                        hovertext=f"You: {b.get('verdict')}", hoverinfo="text")
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 10, "b": 30}, legend={"orientation": "h"},
                      xaxis_title="seconds", yaxis={"visible": False, "range": [0, 1.55]})
    st.plotly_chart(fig, width="stretch", key="tr_mchart")
    st.caption("Red lines = new scene cuts · grey dotted = old cuts · ▼ your scene-change answers "
               "(green = new scene, red = same scene). Each curve is scaled to its own range.")
    rows = [{"Scene": s.get("scene_id"), "From": format_mmss(s.get("start") or 0),
             "To": format_mmss(s.get("end") or 0),
             "People": s.get("people"),
             "Movement": None if s.get("level") is None else round(s["level"] * 1000, 1),
             "Movement BPM": s.get("movement_bpm"),
             "Regular": None if s.get("periodicity") is None else round(s["periodicity"], 2),
             "Old motion": None if s.get("old_motion") is None else round(s["old_motion"], 2)}
            for s in v.get("scenes") or []]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch", column_config={
            "Regular": st.column_config.ProgressColumn("Regular", min_value=0.0, max_value=1.0, format="%.2f",
                                                       help="How regular the movement is (0.5+ = a clear beat)")})


def render() -> None:
    data = TR.load()
    if not data:
        st.info("No trial results here yet. After a Modal trial run, the laptop saves them to the "
                "Drive catalogue folder — then press ☁️ Sync from Google Drive.")
        return
    teachers = ", ".join(f"{k} ({m.split('/')[-1]})" for k, m in (data.get("teachers") or {}).items())
    st.caption(f"Trial of {data.get('created')} · {len(data['videos'])} video(s) · teachers: {teachers or 'none'}"
               + (" · **smoke test** (first 90 s of one video)" if data.get("smoke") else ""))
    with st.container(key="subnav_tr"):
        sub = st.radio("Trial", ["🖼️ Compare tags", "🕺 Motion", "📋 Report"], horizontal=True,
                       key="tr_sub", label_visibility="collapsed")
    if sub == "🖼️ Compare tags":
        _compare(data)
    elif sub == "🕺 Motion":
        _motion(data)
    else:
        st.markdown(TR.report_markdown() or "_No report._")
