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
            val = st.pills(title, tags, selection_mode="single", default=default[0] if default else None,
                           wrap=True, width="stretch", key=wkey)
            picked += [val] if val else []
        else:
            picked += list(st.pills(title, tags, selection_mode="multi", default=default,
                                    wrap=True, width="stretch", key=wkey) or [])
    for p in TLB.rule_problems(picked, vocab):
        st.warning(f"{p} — save anyway if this frame really is like that (then the rule may need changing).")
    card = TR.correction_card(vid, m, start)
    c1, c2 = st.columns(2)
    c1.button("💾 Save my tags", type="primary", key=f"{ekey}_save", width="stretch",
              on_click=_save, args=(card, picked, ekey))
    c2.button("Cancel", key=f"{ekey}_cancel", width="stretch",
              on_click=lambda: st.session_state.pop(ekey, None))


def _apply_rules(ids: list, rows: list) -> None:
    done = TLB.apply_rules_to_labels(ids, rows)
    n = sum(len(a) for _, a in done)
    st.session_state["tr_saved_note"] = f"Added {n} always-with tag(s) to {len(done)} of your frames."


def _rules_panel(videos: dict, rows: list) -> None:
    """Your vocabulary rules against the frames you've already reviewed (in the batch shown): add the
    missing always-with tags in one go; never-with clashes, only-with misses and pick-one doubles are for
    you to fix (which tag is wrong?) — the ⚠️ filter shows them."""
    if not any(r.get("never_with") or r.get("always_with") or r.get("only_with") or r.get("one_of")
               for r in rows):
        return
    fixable, broken = [], []
    for v, vd in videos.items():
        for m in vd["moments"]:
            if m.get("mine") is None:
                continue
            new, added = TLB.fix_tags(m["mine"], rows)
            if added:
                fixable.append((m.get("gold_id") or TR.moment_id(v, m), v, m, added))
            if TLB.rule_problems(new, rows):
                broken.append((v, m))
    if not fixable and not broken:
        st.caption("🧩 Your reviewed frames here all follow your vocabulary rules.")
        return
    with st.expander(f"🧩 Your rules on frames you've reviewed — {len(fixable)} can be fixed automatically, "
                     f"{len(broken)} need you", expanded=False):
        if fixable:
            st.markdown(f"**{len(fixable)} frame(s)** are missing a tag an **always-with** rule says must be there:")
            st.caption("  \n".join(f"{v[:40]} · {format_mmss(m['t'])} → add {', '.join(a)}"
                                    for _, v, m, a in fixable[:8])
                       + (f"  \n… and {len(fixable) - 8} more" if len(fixable) > 8 else ""))
            st.button(f"🧩 Add the missing tags to {len(fixable)} frame(s)", type="primary", key="tr_rules_apply",
                      on_click=_apply_rules, args=([i for i, *_ in fixable], rows),
                      help="Only adds tags — never one that would break a pick-one set or a never-with rule.")
        if broken:
            st.markdown(f"**{len(broken)} frame(s)** break a **never-with** rule, have a tag without any of its "
                        "**only-with** partners, or have two tags from one **pick-one set** — only you can say "
                        "which tag is wrong.")
            st.button("⚠️ Show them", key="tr_rules_show",
                      on_click=lambda: st.session_state.update(tr_filter="rules", tr_page=1))


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
    if st.session_state.get("tr_vid") not in ["(all videos)"] + sorted(videos):
        st.session_state.pop("tr_vid", None)          # e.g. after switching batch
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
    vocab_rows, _ = TLB.load_vocabulary()
    problems = (lambda tags: TLB.rule_problems(tags, vocab_rows))
    moments = [(v, m) for v, m in pool
               if TR.filter_moments([m], how, None if tag == "(any tag)" else tag, problems)]
    if order == "random":
        def _reshuffle():
            st.session_state["tr_seed"] = st.session_state.get("tr_seed", 0) + 1
            st.session_state["tr_page"] = 1
        st.button("🎲 Shuffle again", key="tr_shuffle", on_click=_reshuffle,
                  help="A new random sample. Frames you save drop out of “Not reviewed by me yet”.")
        moments = TR.ordered(moments, "random", st.session_state.get("tr_seed", 0))

    _rules_panel(videos, vocab_rows)
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
    def label(v, k):
        """The teacher's real name for this video (e.g. '32B FP8' — an FP8-only video is saved under
        the 32B's slot), else the slot name."""
        model = (((videos[v].get("run") or {}).get("tagging") or {}).get(k) or {}).get("model")
        return TR.teacher_label(model) if model else k.upper()

    for v, m in moments[(page - 1) * PER_PAGE: page * PER_PAGE]:
        # only the teachers that tagged this moment (one "not tagged" row if none did)
        here = [k for k in teachers if (m.get("teachers") or {}).get(k)] or teachers[:1]
        with st.container(border=True, key=f"tr_wrap_card_{_k(v, m)}"):
            a, b = st.columns([3, 7])        # picture ~25% smaller than before (was 2:3), more room for tags
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
                for k in here:
                    r = (m.get("teachers") or {}).get(k)
                    rows.append(_row(label(v, k), r.get("tags") if r else [], mine, r.get("unsure") if r else ())
                                if r else f'<div style="opacity:.55;font-size:.86rem">{label(v, k)}: not tagged</div>')
                st.markdown("".join(rows), unsafe_allow_html=True)
                if mine is not None:
                    for p in problems(mine):
                        st.caption(f"⚠️ {p}")
                caps = []
                for k in here:
                    r = (m.get("teachers") or {}).get(k) or {}
                    if r.get("caption"):
                        caps.append(f"**{label(v, k)}:** {r['caption']}"
                                    + (" · ↩ copied from a near-identical moment" if r.get("copied_from") else ""))
                if caps:
                    st.caption("  \n".join(caps))
                # ✏️ Correct: your tags for this frame go to your training labels (the gold set).
                ekey = f"tr_edit_{_k(v, m)}"
                if st.session_state.get(ekey):
                    _editor(v, m, teachers, ekey)
                else:
                    with st.container(key=f"tr_wrap_act_{_k(v, m)}"):
                        acts = [k for k in here if (m.get("teachers") or {}).get(k)]
                        cols = st.columns(1 + len(acts))
                        cols[0].button("✏️ Correct", key=f"{ekey}_open", width="stretch",
                                       on_click=lambda k=ekey: st.session_state.__setitem__(k, True),
                                       help="Set the right tags for this frame — saved to your training labels.")
                        for col, k in zip(cols[1:], acts):
                            tags = (m["teachers"][k].get("tags") or [])
                            col.button(f"✅ {label(v, k)} is right", key=f"{ekey}_ok_{k}", width="stretch",
                                       on_click=_save,
                                       args=(TR.correction_card(v, m, tags), tags, ekey),
                                       help=f"Save {label(v, k)}'s tags as your tags for this frame.")


def _motion_chart(v: dict, marks: list) -> None:
    """The plain chart (no video) — when there's no preview copy on this machine."""
    import plotly.graph_objects as go
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
    for b in marks:
        fig.add_scatter(x=[b["t"]], y=[1.45], mode="markers", showlegend=False,
                        marker={"symbol": "triangle-down", "size": 10,
                                "color": {"cut": "#54a24b", "no_cut": "#e45756"}.get(b.get("verdict"), "#999")},
                        hovertext=f"You: {b.get('verdict')}", hoverinfo="text")
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 10, "b": 30}, legend={"orientation": "h"},
                      xaxis_title="seconds", yaxis={"visible": False, "range": [0, 1.55]})
    st.plotly_chart(fig, width="stretch", key="tr_mchart")
    st.caption("Red lines = new scene cuts · grey dotted = old cuts · ▼ your scene-change answers "
               "(green = new scene, red = same scene). Each curve is scaled to its own range.")


def _motion(data: dict) -> None:
    import motion_review as MR
    from library_common import proxy_path
    videos = data["videos"]
    if st.session_state.get("tr_mvid") not in videos:
        st.session_state.pop("tr_mvid", None)
    vid = st.selectbox("Video", sorted(videos), key="tr_mvid")
    v = videos[vid]
    items = TLB.load_labels()
    note = st.session_state.pop("tr_motion_note", None)
    if note:
        st.toast(note, icon="✂️")
    marks = MR.marks_for(vid, items)
    try:
        has_proxy = proxy_path(vid).exists()
    except OSError:
        has_proxy = False
    if not has_proxy:
        st.info("No preview copy of this video here — showing the chart only. (☁️ Sync from Google Drive "
                "brings the preview copies to the server.)")
        _motion_chart(v, marks)
    else:
        player = {**v, "person_curve_n": TR.normalised(v.get("person_curve")),
                  "old_curve_n": TR.normalised(v.get("old_curve"))}
        value = MR.motion_review(player, vid, MR.video_url(proxy_path(vid)), marks, key="tr_motion_player")
        if value and value.get("nonce") != st.session_state.get("tr_motion_nonce"):
            st.session_state["tr_motion_nonce"] = value.get("nonce")
            msg = MR.apply(value, items)
            if msg:
                st.session_state["tr_motion_note"] = msg
                st.rerun()
        good = sum(m["verdict"] == "cut" and m.get("source") != "manual" for m in marks)
        st.caption(f"Your marks on this video: 👍 {good} · 👎 {sum(m['verdict'] == 'no_cut' for m in marks)} · "
                   f"✂️ added {sum(m.get('source') == 'manual' for m in marks)}. They're saved as scene-change "
                   "labels — the scene detector is trained and scored on them.")
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


def _scores(data: dict) -> None:
    """Live scores for the chosen batch(es): the main teacher against your labels, per tag."""
    teachers = TR.shown_teachers(data.get("teachers") or {})
    videos = TR.with_saved_labels(data["videos"], TLB.load_labels())
    moments = [m for v in videos.values() for m in v["moments"]]
    labelled = sum(1 for m in moments if m.get("mine") is not None)
    st.caption(f"{len(videos)} video(s) · {len(moments)} tagged moments · {labelled} labelled by you. "
               "These update as you correct frames in 🖼️ Compare tags.")
    for k in teachers:
        a = TR.agreement(moments).get(k)
        if a:
            st.markdown(f"**{k.upper()}:** exactly right on **{a['exact']} of {a['frames']}** of your frames · "
                        f"found **{0 if a['recall'] is None else a['recall'] * 100:.0f}%** of your tags · "
                        f"**{0 if a['precision'] is None else a['precision'] * 100:.0f}%** of its tags right")
        rows = TR.tag_scores(moments, k)
        if not rows:
            st.info("None of your labelled frames are in these videos yet — correct a few in 🖼️ Compare tags.")
            continue
        pct = lambda x: None if x is None else round(x * 100)
        st.dataframe([{"Tag": r["tag"], "Your frames": r["yours"], "Found (recall %)": pct(r["recall"]),
                       "Right (precision %)": pct(r["precision"]), "F1": None if r["f1"] is None else round(r["f1"], 2),
                       "Missed": r["fn"], "Wrong": r["fp"]} for r in rows],
                     hide_index=True, width="stretch", key=f"tr_scores_{k}",
                     column_config={"F1": st.column_config.ProgressColumn("F1", min_value=0.0, max_value=1.0,
                                                                         format="%.2f")})


def _report(data: dict, batch) -> None:
    """Every batch side by side, plus all of them together: footage, time and cost per step, the
    32B against your labels, scene cuts — to compare batches run with different settings."""
    teacher = (TR.shown_teachers(data.get("teachers") or {}) or [TR.MAIN_TEACHER])[0]
    videos = TR.with_saved_labels(data["videos"], TLB.load_labels())
    rows = TR.batch_report(videos, teacher)
    if not any(v.get("run") for v in videos.values()):
        st.info("Times and costs per batch appear after the next export — on the server: "
                "`modal run modal_trial.py --stage results` (no GPU, about a minute).")
    pct = lambda x: None if x is None else round(x * 100)
    names = lambda r: "All batches" if r["batch"] == "All" else (f"Batch {r['batch']}" if r["batch"] != "?" else "Not recorded")
    ex = {"on": "✅ on", "off": "— off", "mixed": "mixed"}

    # 1. The scorecard: one line per batch, the numbers that decide things.
    st.markdown("#### 🎯 How well the teacher matches your tags")
    st.caption("Per batch, on the frames you've labelled (🎯 cards and ✏️ corrections) — updates as you correct "
               "more. **Found** = share of your tags the teacher also gave · **Right** = share of the teacher's "
               "tags you agree with · **F1** combines both (1 = perfect). Compare batches with the same teacher.")
    st.dataframe([{"": names(r), "Teacher": r.get("teacher") or "–",
                   "Worked examples": ex.get(r.get("examples"), "–") if r["batch"] != "All" else "",
                   "Your labelled frames": r["labelled"],
                   "Found %": pct(r["recall"]), "Right %": pct(r["precision"]),
                   "F1": None if r.get("f1") is None else round(r["f1"], 2),
                   "$ / footage h": r["cost_per_h"]} for r in rows],
                 hide_index=True, width="stretch", key="tr_rep_score",
                 column_config={"F1": st.column_config.ProgressColumn("F1", min_value=0.0, max_value=1.0, format="%.2f"),
                                "$ / footage h": st.column_config.NumberColumn(format="$%.2f")})
    st.caption("Per-tag detail for a batch: pick it above, then 📊 Scores.")

    # 2. The detail, folded away.
    with st.expander("💷 Cost and time per batch"):
        st.dataframe([{"": names(r), "Run (UTC)": r["run"], "Videos": r["videos"], "Footage (h)": r["footage_h"],
                       "Vision GPU": r["vision_gpu"], "Vision (min)": r["vision_min"], "Vision $": r["vision_cost"],
                       "Tagging GPU": r["tag_gpu"], "Tagging (min)": r["tag_min"],
                       "Tagging min / footage h": r["tag_min_per_h"], "Start-up (min)": r["tag_startup_min"],
                       "Tagging $": r["tag_cost"], "Total $": r["total_cost"], "$ / footage h": r["cost_per_h"]}
                      for r in rows], hide_index=True, width="stretch", key="tr_rep_cost")
        st.caption("Modal's list prices for the GPU each step ran on. Start-up = loading the model once per run, "
                   "so small batches pay proportionally more of it.")
    with st.expander("🔎 Tagging detail per batch"):
        st.dataframe([{"": names(r), "Moments tagged": r["moments"], "Unreadable replies": r["unreadable"],
                       "Your labelled frames": r["labelled"], "Exactly right": r["exact"],
                       "Found %": pct(r["recall"]), "Right %": pct(r["precision"])}
                      for r in rows], hide_index=True, width="stretch", key="tr_rep_tags")
    with st.expander("✂️ Scene changes per batch"):
        st.dataframe([{"": names(r), "Old cuts": r["cuts_old"], "New cuts": r["cuts_new"],
                       "Your answers": r["cut_answers"], "New right %": pct(r["cut_acc_new"]),
                       "Old right %": pct(r["cut_acc_old"])} for r in rows],
                     hide_index=True, width="stretch", key="tr_rep_cuts")
    with st.expander("📄 Full written report (as at the last export)"):
        st.markdown(TR.report_markdown() or "_No report._")


def render() -> None:
    data = TR.load()
    if not data:
        st.info("No trial results here yet. After a Modal trial run, the laptop saves them to the "
                "Drive catalogue folder — then press ☁️ Sync from Google Drive.")
        return
    teachers = ", ".join(f"{k} ({m.split('/')[-1]})" for k, m in (data.get("teachers") or {}).items())
    st.caption(f"Trial of {data.get('created')} · {len(data['videos'])} video(s) · teachers: {teachers or 'none'}"
               + (" · **smoke test** (first 90 s of one video)" if data.get("smoke") else ""))
    blist = TR.batches(data["videos"])
    if "?" in blist:
        st.caption("Batch numbers appear after the next export — on the server: "
                   "`modal run modal_trial.py --stage results`.")
    names = {None: "All batches", **{b: f"Batch {b}" if b != "?" else "Batch not recorded" for b in blist}}
    with st.container(key="tl_wrap_tr_top"):
        a, b = st.columns([1, 3])
        batch = a.selectbox("Batch", [None] + blist, format_func=names.get, key="tr_batch",
                            help="Look at one batch run on its own, or all of them together.")
        with b:
            with st.container(key="subnav_tr"):
                sub = st.radio("Trial", ["📋 Summary", "🖼️ Compare tags", "📊 Scores", "🕺 Motion"], horizontal=True,
                               key="tr_sub", label_visibility="collapsed")
    view = {**data, "videos": TR.in_batch(data["videos"], batch)}
    if not view["videos"] and sub != "📋 Summary":
        st.info("No videos in this batch.")
        return
    if sub == "🖼️ Compare tags":
        _compare(view)
    elif sub == "📊 Scores":
        _scores(view)
    elif sub == "🕺 Motion":
        _motion(view)
    else:
        _report(data, batch)
