"""🧪 Trial results view for the Training Labels page (pages/6_Training_Labels.py): the latest
Modal trial — each tagged frame with your tags, the old tags and every model's tags side by
side; motion curves and scene cuts per video; the report. Data: trial_results.py."""

from __future__ import annotations

import hashlib
import html

import streamlit as st

from library_common import safe_image, format_mmss
import training_labels as TLB
import student_results_view as SRV
import trial_results as TR

CHIP_STYLE = {
    TR.RIGHT: ("✓", "rgba(46,160,67,.18)", "rgba(46,160,67,.75)", ""),
    TR.WRONG: ("✗", "rgba(229,83,75,.16)", "rgba(229,83,75,.8)", ""),
    TR.MISSED: ("+", "transparent", "rgba(128,128,128,.7)", "border-style:dashed;opacity:.85;"),
    TR.NEUTRAL: ("", "rgba(128,128,128,.12)", "rgba(128,128,128,.4)", ""),
}
PER_PAGE = 12
GRID_PAGE = 48
GRID_ALL = "🗂️ All categories"
# ▦ Grid on a touch screen, like Photos on the iPad: tap a picture to tick it; put a finger on a picture and
# swipe sideways to tick (or untick) every frame you pass over — down the page too, a whole range at a time.
# An up / down swipe still scrolls. It only clicks the frames' own tick boxes, so nothing reloads until ➕ / ➖.
GRID_JS = r"""
<script>
(function () {
  const W = window;
  if (W.__mpGridSel) return;
  W.__mpGridSel = true;
  const CARD = "[class*='st-key-tr_gcard_']";
  const cards = () => Array.from(document.querySelectorAll(CARD));
  const box = (c) => c && c.querySelector("input[type=checkbox]");
  const at = (x, y) => { const el = document.elementFromPoint(x, y); return el && el.closest(CARD); };
  const set = (c, on) => { const b = box(c); if (b && b.checked !== on) b.click(); };
  function scroller(el) {
    for (let e = el; e && e !== document.body; e = e.parentElement) {
      const o = getComputedStyle(e).overflowY;
      if ((o === "auto" || o === "scroll") && e.scrollHeight > e.clientHeight) return e;
    }
    return document.scrollingElement;
  }
  let g = null, lastDrag = 0;
  document.addEventListener("touchstart", (e) => {
    g = null;
    if (e.touches.length !== 1) return;
    const t = e.touches[0], c = at(t.clientX, t.clientY);
    if (c) g = { x: t.clientX, y: t.clientY, card: c, on: false, active: false, orig: null, all: null };
  }, { passive: true });
  document.addEventListener("touchmove", (e) => {
    if (!g) return;
    const t = e.touches[0], dx = t.clientX - g.x, dy = t.clientY - g.y;
    if (!g.active) {
      if (Math.abs(dx) < 10 && Math.abs(dy) < 10) return;
      if (Math.abs(dy) >= Math.abs(dx)) { g = null; return; }       // up / down: let it scroll
      g.active = true;
      g.all = cards();
      g.orig = g.all.map((c) => { const b = box(c); return !!(b && b.checked); });
      const b = box(g.card);
      g.on = !(b && b.checked);
      g.from = g.all.indexOf(g.card);
      g.sc = scroller(g.card);
    }
    if (e.cancelable) e.preventDefault();
    const c = at(t.clientX, t.clientY);
    const to = c ? g.all.indexOf(c) : -1;
    if (to >= 0) {
      const lo = Math.min(g.from, to), hi = Math.max(g.from, to);
      g.all.forEach((card, i) => set(card, i >= lo && i <= hi ? g.on : g.orig[i]));
    }
    const H = window.innerHeight;                                   // keep going past the screen edge
    if (t.clientY > H - 70) g.sc.scrollBy(0, 14); else if (t.clientY < 90) g.sc.scrollBy(0, -14);
  }, { passive: false });
  let lastTap = 0;
  document.addEventListener("touchend", (e) => {
    // a tap on a picture ticks it right away — and cancels the tap's own click, so Safari doesn't spend
    // it on a hover or open the picture instead
    if (g && !g.active) {
      const img = e.target.closest && e.target.closest(CARD + " [data-testid='stImage']");
      if (img) {
        if (e.cancelable) e.preventDefault();
        const b = box(img.closest(CARD));
        if (b) b.click();
        lastTap = Date.now();
      }
    }
    if (g && g.active) lastDrag = Date.now();
    g = null;
  }, { passive: false });
  document.addEventListener("click", (e) => {                       // a mouse click on a picture (laptop)
    const img = e.target.closest && e.target.closest(CARD + " [data-testid='stImage']");
    if (!img || e.target.closest("button") || Date.now() - lastDrag < 400 || Date.now() - lastTap < 700) return;
    const b = box(img.closest(CARD));
    if (b) b.click();
  }, true);
})();
</script>
"""   # ▦ Grid: every category's tags under each frame — the final check            # ▦ Grid: thumbnails per page
# ▦ Grid: 8 thumbnails a row with a mouse / trackpad, 5 on a touch tablet (iPad), 3 on a phone
GRID_CSS = (
    ".st-key-tr_grid {display:grid!important;grid-template-columns:repeat(8,minmax(0,1fr));gap:.45rem;"
    "align-items:start}"
    ".st-key-tr_grid>div {width:auto!important;min-width:0}"
    "@media (hover:none) and (pointer:coarse) {.st-key-tr_grid {grid-template-columns:repeat(5,minmax(0,1fr))}}"
    "@media (max-width:640px) {.st-key-tr_grid {grid-template-columns:repeat(3,minmax(0,1fr))}}"
    "[class*='st-key-tr_gcard_'] {gap:.15rem!important}"
    "[class*='st-key-tr_gcard_'] [data-testid='stCheckbox'] label {font-size:.78rem}"
    # no ⛶ fullscreen button on the thumbnails: on the iPad the first tap only showed it (and could open it)
    "[class*='st-key-tr_gcard_'] [data-testid='stElementToolbar'] {display:none!important}"
    "[class*='st-key-tr_gcard_'] img {border-radius:.35rem;-webkit-touch-callout:none;-webkit-user-select:none;"
    "user-select:none;cursor:pointer}"
    # sideways swipes are ours (swipe-select), up / down still scrolls
    ".st-key-tr_grid [class*='st-key-tr_gcard_'] {touch-action:pan-y;-webkit-user-select:none;user-select:none}"
    # a ticked frame stands out
    "[class*='st-key-tr_gcard_']:has(input:checked) img {outline:3px solid #ff4b4b;outline-offset:-3px}"
    # the ● / ✗ / ◆ marks: no negative markdown margin (it let the next row cover them), wrap, never cut
    "[class*='st-key-tr_gcard_'] [data-testid='stMarkdownContainer'] {margin-bottom:0!important;overflow:visible}"
    "[class*='st-key-tr_gcard_'] .mp-gstate {padding-bottom:3px;line-height:1.5;white-space:normal}"
    "[class*='st-key-tr_gcard_'] .mp-gstate span {max-width:100%;overflow:hidden;text-overflow:ellipsis;"
    "white-space:nowrap;vertical-align:top}"
    ".st-key-tr_grid {row-gap:.9rem!important}"
    # a picked tag shows on a frame only once its box is ticked
    "[class*='st-key-tr_gcard_'] .mp-gpick {display:none}"
    "[class*='st-key-tr_gcard_']:has(input:checked) .mp-gpick {display:inline-block}"
    "[class*='st-key-tr_gbar_'] button {min-height:2.6rem}"
)


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
    red = ' class="mp-red"' if status == TR.WRONG else ""
    return (f'<span{red} title="{title}" style="display:inline-block;margin:2px 4px 2px 0;padding:1px 9px;'
            f'border-radius:999px;border:1px solid {border};background:{bg};font-size:.86rem;{extra}">'
            f'{mark + " " if mark else ""}{html.escape(tag)}{"?" if faded else ""}</span>')


def _row(label: str, pred, mine, unsure=()) -> str:
    parts = [_chip(t, s) for t, s in TR.chips(pred, mine)]
    parts += [_chip(t, TR.NEUTRAL, faded=True) for t in unsure or []]
    body = "".join(parts) or '<span style="opacity:.55">no tags</span>'
    return (f'<div style="margin:3px 0"><span style="display:inline-block;min-width:4.6rem;font-weight:600;'
            f'font-size:.86rem">{label}</span>{body}</div>')


def _part_row(p: dict) -> str:
    """Your ▦ batch answers on a frame you haven't fully labelled: yes tags, and struck-through no tags."""
    no = "".join(f'<span class="mp-red" title="you said it isn\'t there" style="display:inline-block;margin:2px 4px 2px 0;'
                 f'padding:1px 9px;border-radius:999px;border:1px dashed rgba(229,83,75,.7);font-size:.86rem;'
                 f'text-decoration:line-through;opacity:.8">{html.escape(t)}</span>' for t in p.get("no") or [])
    yes = "".join(_chip(t, TR.NEUTRAL) for t in p.get("yes") or [])
    return (f'<div style="margin:3px 0"><span style="display:inline-block;min-width:4.6rem;font-weight:600;'
            f'font-size:.86rem" title="Batch-tagged in ▦ Grid — only these tags are decided">You ▦</span>{yes}{no}</div>')


def _save(card: dict, tags: list, key: str) -> None:
    TLB.save_label(card, tags=tags)
    st.session_state.pop(key, None)        # close the editor
    st.session_state["tr_saved_note"] = f"Saved your tags for {card['video_id'][:40]} at {format_mmss(card['t'])}."


def _look_check_form(run_id: str, v: str, m: dict, look: dict, mine, run_tags) -> None:
    """👁️ Check the look: every step with the model's answer preselected — leave it if it's right,
    pick the right answer if not, then 💾 Save. Saved checks reopen with your answers."""
    key = TR.look_check_key(run_id, v, m["t"])
    saved = TR.load_look_checks().get(key)
    # Only the steps this run answered (older runs predate the hands / mouth steps — don't save a guess).
    steps = [s for s in TR.look_steps() if look.get(s[0])]
    done = ""
    if saved:
        n = sum(1 for k, _, _ in steps if saved["mine"].get(k))
        ok = sum(1 for k, _, _ in steps if saved["mine"].get(k) and saved["mine"].get(k) == saved["model"].get(k))
        done = f" — ✔️ checked: {ok} of {n} right"
    with st.expander(f"👁️ Check the look ({run_id.split('_', 2)[-1]}){done}", expanded=False):
        with st.form(f"tr_lookf_{_k(v, m)}_{run_id}", border=False):
            st.caption("Each step shows the model's answer. Leave it if it's right; pick the right answer if "
                       "it's wrong (\"can't tell\" if you can't see it either).")
            cols = st.columns(2)
            picked = {}
            for i, (k, name, choices) in enumerate(steps):
                model = look.get(k)
                start = (saved or {}).get("mine", {}).get(k) or model
                opts = list(choices) if start in choices else list(choices) + ([start] if start else [])
                picked[k] = cols[i % 2].selectbox(
                    f"{name} — model: {model or '—'}", opts, index=opts.index(start) if start in opts else 0,
                    key=f"tr_look_{_k(v, m)}_{run_id}_{k}")
            if st.form_submit_button("💾 Save look check", type="primary"):
                TR.save_look_check(run_id, v, m["t"], look, picked, mine, run_tags)
                st.session_state["tr_saved_note"] = (f"Saved your look check for {v[:40]} at "
                                                     f"{format_mmss(m['t'])}.")
                st.rerun()


def _look_check_panel() -> None:
    """Your look checks so far: how often each step was right, and the download to send on."""
    items = list(TR.load_look_checks().values())
    if not items:
        return
    with st.expander(f"👁️ My look checks — {len(items)} frame(s) checked · ⬇️ download", expanded=False):
        rows = [{"Step": r["name"], "Checked": r["checked"],
                 "Model right": "—" if r["pct"] is None else f"{r['right']} ({r['pct']}%)",
                 "Most common mistakes (model → you)": "; ".join(f"{a} → {b} ×{c}" for a, b, c in r["fixes"])}
                for r in TR.look_check_summary(items)]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("⬇️ Download my look checks (CSV)", TR.look_checks_csv(items),
                           file_name="look_checks.csv", mime="text/csv", key="tr_look_dl",
                           help="One row per frame and step: the model's answer, yours, right or not — "
                                "attach it in the chat.")


def _editor(vid: str, m: dict, teachers: list, ekey: str) -> None:
    """Fix the tags for one frame: pick-one rows and grouped chips from the vocabulary, starting
    from your tags (or the best model's)."""
    vocab, _ = TLB.load_vocabulary()
    names = {TR.norm(r["tag"]): r["tag"] for r in vocab}
    start = m.get("mine")
    if start is None:
        r = next(((m.get("teachers") or {}).get(k) for k in reversed(teachers) if (m.get("teachers") or {}).get(k)),
                 None)
        start = list((r or {}).get("tags") or m.get("old") or [])
        part = m.get("partial") or {}
        if part:                              # your ▦ batch answers on top of the model's tags
            no = {TR.norm(t) for t in part.get("no") or []}
            start = [t for t in start if TR.norm(t) not in no]
            start += [t for t in part.get("yes") or [] if TR.norm(t) not in {TR.norm(x) for x in start}]
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
    # 🧪 test runs (the test set) and 🔍 check runs (your labelled training frames) both feed the
    # "Check my label" filter; the newest two are picked by default
    test = {"runs": sorted(((TR.load_test_runs() or {}).get("runs") or [])
                           + ((TR.load_check_runs() or {}).get("runs") or []),
                           key=lambda r: str(r.get("created") or ""))}
    test_ids = [r["id"] for r in test["runs"]]
    filters = {k: v for k, v in TR.FILTERS.items()
               if (k != "disagree" or len(teachers) > 1) and (k != "suspect" or test_ids)}
    if st.session_state.get("tr_filter") not in filters:
        st.session_state["tr_filter"] = "all"
    test_vids = TR.test_set_videos()
    if test_vids:
        n_test = len(test_vids & set(videos))
        which = st.segmented_control(
            "Videos", ["all", "train", "test"], key="tr_set", default="all",
            format_func={"all": "All", "train": f"🧠 Training only ({len(videos) - n_test})",
                         "test": f"🧪 Test set only ({n_test} of {len(test_vids)})"}.get,
            help="🧠 Training only = every video except the 🧪 test set: frames you label there go into the next "
                 "LoRA training (each moment here already has its frames on Modal). 🧪 Test set only = the "
                 "held-out videos the test runs are scored on. Pick “All batches” above to see every video.")
        if which == "train":
            videos = {v: d for v, d in videos.items() if v not in test_vids}
        elif which == "test":
            videos = {v: d for v, d in videos.items() if v in test_vids}
    with st.container(key="tl_wrap_tr_filters"):
        c = st.columns([3, 2, 2, 2])
    if st.session_state.get("tr_vid") not in ["(all videos)"] + sorted(videos):
        st.session_state.pop("tr_vid", None)          # e.g. after switching batch
    vid = c[0].selectbox("Video", ["(all videos)"] + sorted(videos), key="tr_vid")
    how = c[1].selectbox("Show", list(filters), format_func=filters.get, key="tr_filter")
    order = c[3].selectbox("Order", ["video", "random"], key="tr_order",
                           format_func={"video": "By video and time", "random": "🎲 Random"}.get,
                           help="Random mixes frames from every video, so a review isn't stuck in one video.")
    if st.session_state.get("tr_view") not in ("cards", "grid"):
        st.session_state["tr_view"] = "cards"
    mode = st.segmented_control(
        "View", ["cards", "grid"], key="tr_view",
        format_func={"cards": "🃏 Cards", "grid": "▦ Grid — batch tag"}.get,
        help="▦ Grid: small thumbnails packed together — pick tags, tick frames, then ➕ Add or ➖ Remove "
             "them on all of them at once.") or "cards"
    no_old = st.toggle("🙈 Ignore old tags", key="tr_no_old", value=True,
                       help="Focus on the model's tags: the Tag filter matches only the model's tags (and "
                            "yours), and the Old row is hidden on the cards. Off = the old tags count and show "
                            "as before.")
    with st.container(key="tl_wrap_tr_part"):
        part_cols = st.columns([3, 3])
    only_part = part_cols[0].toggle(
        "🧩 Only part-reviewed frames", key="tr_only_part",
        help="Frames you've batch-tagged in ▦ Grid but not yet fully labelled, with the model's tags you haven't "
             "reviewed yet listed under each one. ✏️ Correct starts from the model's tags with your batch answers "
             "applied; saving makes it a full label.")
    if part_cols[0].toggle("🔴 Hide red tags", key="tr_hide_red",
                           help="Hides every red tag on the cards and thumbnails — tags you've said aren't there, "
                                "and the model's tags that aren't in your label — for a quick final look at "
                                "what each frame keeps."):
        st.markdown("<style>.mp-red {display:none!important}</style>", unsafe_allow_html=True)
    u = st.session_state.get("tr_g_undo")
    if u:
        st.button(f"↩️ Undo: {u['what']}", key="tr_g_undo_btn", on_click=_gundo)
    all_tags = sorted({t for v in videos.values() for m in v["moments"]
                       for t in (m.get("mine") or []) + ([] if no_old else (m.get("old") or []))
                       + [x for r in m.get("teachers", {}).values() for x in r.get("tags") or []]},
                      key=str.lower)
    if st.session_state.get("tr_tag") not in ["(any tag)"] + all_tags:
        st.session_state["tr_tag"] = "(any tag)"          # e.g. an old-only tag after turning 🙈 on
    tag = c[2].selectbox("Tag", ["(any tag)"] + all_tags, key="tr_tag")
    by = "any"
    if tag != "(any tag)":
        if st.session_state.get("tr_by") not in TR.PICKED_BY:
            st.session_state["tr_by"] = "any"
        by = st.segmented_control(
            "Picked by", list(TR.PICKED_BY), key="tr_by", format_func=TR.PICKED_BY.get,
            help="Only confirmed tags count — your saved labels and the model's tags, never its unsure "
                 "(grey?) ones. 👤 Me = frames you tagged with it · 🤖 The model = the model tagged it · "
                 "🤝 Both of us = you and the model agree. Anyone also counts the old tags unless 🙈 is on.") \
            or "any"
    pool = [(v, m) for v in (sorted(videos) if vid == "(all videos)" else [vid]) for m in videos[v]["moments"]]
    vocab_rows, _ = TLB.load_vocabulary()
    problems = (lambda tags: TLB.rule_problems(tags, vocab_rows))
    moments = [(v, m) for v, m in pool
               if TR.filter_moments([m], how, None if tag == "(any tag)" else tag, problems, skip_old=no_old, by=by)
               and (not only_part or (m.get("partial") and m.get("mine") is None))]
    sus, sus_runs = {}, []
    if how == "suspect":
        # 🔍 your labels that every chosen test run disagrees with — the likeliest mislabels
        sus_runs = st.multiselect("Test runs that must all disagree with your label", test_ids,
                                  default=test_ids[-2:], key="tr_sus_runs",
                                  help="Two different runs (e.g. the plain teacher and the LoRA) both "
                                       "disagreeing with you is a strong hint the label needs a second look. "
                                       "Pick just one run to read its answers on your frames (e.g. a look run: "
                                       "each card then shows what it saw).")
        when = {(i.get("video_id"), round(float(i["t"]), 2)): i.get("labelled_at")
                for i in TLB.load_labels() if i.get("t") is not None}
        sus = TR.suspect_labels(test, sus_runs, videos, when)
        tn = None if tag == "(any tag)" else TR.norm(tag)
        moments = [(v, m) for v, m in moments if (v, round(float(m["t"]), 2)) in sus
                   and (tn is None or tn in {TR.norm(x) for x in sus[(v, round(float(m["t"]), 2))]["there"]
                                             + sus[(v, round(float(m["t"]), 2))]["not_there"]})]
        moments.sort(key=lambda vm: -len(sus[(vm[0], round(float(vm[1]["t"]), 2))]["there"])
                     - len(sus[(vm[0], round(float(vm[1]["t"]), 2))]["not_there"]))
        _look_check_panel()
        st.caption(f"**{len(moments)}** of your labelled frames where every chosen run disagrees with you on at "
                   "least one tag, most disagreements first (pick a tag above to focus, e.g. a position). If your "
                   "label is wrong, ✏️ Correct it; if it's right, save it unchanged — either way it's marked "
                   "checked and leaves this list.")
    if order == "random":
        def _reshuffle():
            st.session_state["tr_seed"] = st.session_state.get("tr_seed", 0) + 1
            st.session_state["tr_page"] = 1
        st.button("🎲 Shuffle again", key="tr_shuffle", on_click=_reshuffle,
                  help="A new random sample. Frames you save drop out of “Not reviewed by me yet”.")
        moments = TR.ordered(moments, "random", st.session_state.get("tr_seed", 0))

    if only_part:
        with part_cols[1]:
            _finish_panel(moments, vocab_rows)
    if mode == "grid":
        _grid(moments, vocab_rows, tag, only_part)
        return
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
                elif m.get("partial"):
                    rows.append(_part_row(m["partial"]))
                    if only_part:
                        rows.append(_todo_html(TR.still_to_review(m)))
                if not no_old:
                    rows.append(_row("Old", m.get("old") or [], mine))
                for k in here:
                    r = (m.get("teachers") or {}).get(k)
                    rows.append(_row(label(v, k), r.get("tags") if r else [], mine, r.get("unsure") if r else ())
                                if r else f'<div style="opacity:.55;font-size:.86rem">{label(v, k)}: not tagged</div>')
                sk = (v, round(float(m["t"]), 2))
                if sk in sus:
                    rows += [_row(f"🧪 {rid.split('_', 2)[-1]}", tags, mine) for rid, tags in sus[sk]["runs"].items()]
                st.markdown("".join(rows), unsafe_allow_html=True)
                if sk in sus:
                    bits = []
                    if sus[sk]["there"]:
                        bits.append("they all see **" + ", ".join(sus[sk]["there"]) + "** (you don't)")
                    if sus[sk]["not_there"]:
                        bits.append("none of them see **" + ", ".join(sus[sk]["not_there"]) + "** (you do)")
                    st.caption("🔍 " + " · ".join(bits))
                    for rid, d in (sus[sk].get("detail") or {}).items():
                        extra = []
                        if d.get("nb"):
                            extra.append("↔️ filled in from the moments either side: " + ", ".join(d["nb"]))
                        if d.get("ld"):
                            extra.append("👁️ dropped — its look disagrees: " + ", ".join(d["ld"]))
                        if d.get("look"):
                            extra.append(f"👁️ *{TR.look_line(d['look'])}*"
                                         + (f" — “{d['look']['summary']}”" if d["look"].get("summary") else ""))
                        if extra:
                            st.caption(f"🧪 {rid.split('_', 2)[-1]}: " + "  \n".join(extra))
                        if d.get("look"):
                            _look_check_form(rid, v, m, d["look"], mine, sus[sk]["runs"].get(rid) or [])
                    st.button("✅ My label is right", key=f"tr_sus_ok_{_k(v, m)}", on_click=_save,
                              args=(TR.correction_card(v, m, mine), list(mine), f"tr_edit_{_k(v, m)}"),
                              help="Keeps your tags as they are and marks the frame checked.")
                if mine is not None:
                    for p in problems(mine):
                        st.caption(f"⚠️ {p}")
                caps = []
                for k in here:
                    r = (m.get("teachers") or {}).get(k) or {}
                    if r.get("caption"):
                        caps.append(f"**{label(v, k)}:** {r['caption']}"
                                    + (" · ↩ copied from a near-identical moment" if r.get("copied_from") else "")
                                    + (f" · 👀 added on a second look: {', '.join(r['look_added'])}"
                                       if r.get("look_added") else "")
                                    + (f" · 🔍 taken out on double-check: {', '.join(r['look_removed'])}"
                                       if r.get("look_removed") else "")
                                    + (f" · ↔️ filled in from the moments either side: {', '.join(r['neighbour_added'])}"
                                       if r.get("neighbour_added") else "")
                                    + (f" · 👁️ dropped — its look disagrees: {', '.join(r['look_dropped'])}"
                                       if r.get("look_dropped") else "")
                                    + (f"  \n👁️ *{TR.look_line(r['look'])}*" if r.get("look") else ""))
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


# ---------------------------------------------------------------------------
# ▦ Grid: batch tagging many frames at once
# ---------------------------------------------------------------------------

def _gtags(m: dict, in_cat: list) -> list:
    """The tags to mark under a thumbnail: every tag of the open category that you or the model have on
    this frame (or you've said no to in a batch), in vocabulary order."""
    no = {TR.norm(t) for t in (m.get("partial") or {}).get("no") or []}
    return [t for t in in_cat if TR.your_answer(m, t) is True or TR.model_has(m, t) or TR.norm(t) in no]


def _gstate(m: dict, tags: list, picked=()) -> str:
    """Small marks under a thumbnail for each tag: ● you have it · ✗ you said no · ◆ only the
    model has it. A picked tag (the ones ➕ / ➖ apply) not already there appears as “+ tag” only once you
    tick that frame (CSS, so ticking still doesn't reload the page)."""
    bits = []
    for t in list(tags) + [p for p in picked if p not in tags]:
        if t not in tags:
            short = t if len(t) <= 14 else t[:13] + "…"
            bits.append(f'<span class="mp-gpick" title="picked — ➕ Add gives this frame it" style="margin:0 3px 1px 0;'
                        f'padding:0 5px;border-radius:999px;font-size:.7rem;color:rgb(214,150,40);'
                        f'border:1px dashed rgba(214,150,40,.85)">+ {html.escape(short)}</span>')
            continue
        ans = TR.your_answer(m, t)
        if ans is True:
            mark, col, tip = "●", "46,160,67", "you have it"
        elif ans is False:
            mark, col, tip = "✗", "229,83,75", "you said it isn't there"
        elif TR.model_has(m, t):
            mark, col, tip = "◆", "56,139,253", "only the model has it"
        else:
            mark, col, tip = "·", "128,128,128", "nobody has it"
        short = t if len(t) <= 14 else t[:13] + "…"
        red = ' class="mp-red"' if ans is False else ""
        bits.append(f'<span{red} title="{html.escape(tip)}" style="display:inline-block;margin:0 3px 1px 0;'
                    f'padding:0 5px;border-radius:999px;font-size:.7rem;color:rgb({col});'
                    f'border:1px solid rgba({col},.55)">{mark} {html.escape(short)}</span>')
    return f'<div class="mp-gstate">{"".join(bits)}</div>' if bits else ""


def _gtick_key(v: str, m: dict) -> str:
    return f"tr_gt_{_k(v, m)}"


def _gselect(items: list, on: bool) -> None:
    for v, m in items:
        st.session_state[_gtick_key(v, m)] = on


def _gapply(action: str, items: list, rows: list) -> None:
    chosen = list(st.session_state.get("tr_g_chosen") or [])
    ticked = [(v, m) for v, m in items if st.session_state.get(_gtick_key(v, m))]
    if not chosen:
        st.session_state["tr_saved_note"] = "Pick at least one tag first."
        return
    if not ticked:
        st.session_state["tr_saved_note"] = "Tick at least one frame first."
        return
    res = TLB.batch_tag([{"video_id": v, "scene_id": m.get("scene_id"), "t": float(m["t"]),
                          "label_id": m.get("gold_id") or TR.moment_id(v, m), "tags": m.get("mine")} for v, m in ticked],
                        add=chosen if action == "add" else (), remove=chosen if action == "remove" else (),
                        rows=rows)
    _gselect(ticked, False)
    verb = "Added" if action == "add" else "Removed"
    what = f"{verb} {', '.join(chosen)} on {len(ticked)} frame{'s' if len(ticked) != 1 else ''}"
    if res["undo"]:
        st.session_state["tr_g_undo"] = {"undo": res["undo"], "what": what}
    st.session_state["tr_saved_note"] = what + (" (nothing changed — they already had that)" if not res["undo"] else "")


def _gundo() -> None:
    u = st.session_state.pop("tr_g_undo", None)
    if u:
        TLB.undo_batch(u["undo"])
        st.session_state["tr_saved_note"] = f"Undone: {u['what']}"


def _gbar(where: str, items: list, rows: list) -> None:
    with st.container(key=f"tr_gbar_{where}"):
        c = st.columns(4)
        c[0].form_submit_button("➕ Add tags", key=f"tr_g_add_{where}", type="primary", width="stretch",
                                on_click=_gapply, args=("add", items, rows),
                                help="Give every ticked frame the picked tags.")
        c[1].form_submit_button("➖ Remove tags", key=f"tr_g_rm_{where}", width="stretch",
                                on_click=_gapply, args=("remove", items, rows),
                                help="Mark the picked tags as not there on every ticked frame.")
        c[2].form_submit_button("☑️ Select all", key=f"tr_g_all_{where}", width="stretch",
                                on_click=_gselect, args=(items, True))
        c[3].form_submit_button("◻️ Clear ticks", key=f"tr_g_none_{where}", width="stretch",
                                on_click=_gselect, args=(items, False))


def _todo_html(todo: list, small: bool = False) -> str:
    """The model's tags you haven't reviewed yet on a batch-tagged frame."""
    if not todo:
        return (f'<div class="mp-gstate" style="font-size:{".7rem" if small else ".8rem"};opacity:.7">'
                "✔ nothing left to review</div>")
    fs = ".7rem" if small else ".86rem"
    chips = "".join(f'<span title="the model has it — not reviewed by you yet" style="display:inline-block;'
                    f'margin:1px 4px 1px 0;padding:0 7px;border-radius:999px;font-size:{fs};color:rgb(214,150,40);'
                    f'border:1px dashed rgba(214,150,40,.8)">? {html.escape(t)}</span>' for t in todo)
    label = "" if small else ('<span style="display:inline-block;min-width:4.6rem;font-weight:600;font-size:.86rem">'
                              "To review</span>")
    return f'<div class="mp-gstate" style="margin:3px 0">{label}{chips}</div>'


def _finish(items: list, rows: list) -> None:
    res = TLB.finish_partials(items, rows)
    st.session_state.pop("tr_finish_ask", None)
    what = f"Finished {res['done']} part-reviewed frame{'s' if res['done'] != 1 else ''}"
    if res["undo"]:
        st.session_state["tr_g_undo"] = {"undo": res["undo"], "what": what}
    st.session_state["tr_saved_note"] = what + (f" · {len(res['skipped'])} left (a needs-a-tag category is empty)"
                                                if res["skipped"] else "")


def _finish_panel(moments: list, rows: list) -> None:
    """🧹 Remove all not-reviewed tags: every shown batch-tagged frame becomes a full label with only your tags."""
    items = [{"video_id": v, "scene_id": m.get("scene_id"), "t": float(m["t"]),
              "label_id": m.get("gold_id") or TR.moment_id(v, m),
              "suggested": sorted({t for r in (m.get("teachers") or {}).values() for t in r.get("tags") or []})}
             for v, m in moments if m.get("partial") and m.get("mine") is None]
    if not st.session_state.get("tr_finish_ask"):
        st.button(f"🧹 Remove all not-reviewed tags ({len(items)})", key="tr_finish_btn", width="stretch",
                  disabled=not items, on_click=lambda: st.session_state.update(tr_finish_ask=True),
                  help="For every frame shown: keep only the tags you added, drop every tag you haven't reviewed, "
                       "and save it as a full label. Frames with a “needs a tag” category still empty are left.")
        return
    group_of = {r["tag"].lower(): r.get("group") or "other" for r in rows}
    required = [g["group"] for g in TLB.group_settings(rows) if g["required"]]
    by_pid = {}
    for v, m in moments:
        if m.get("partial") and m.get("mine") is None:
            by_pid[(v, float(m["t"]))] = {group_of.get(t.lower(), "other") for t in m["partial"].get("yes") or []}
    short = sum(1 for have in by_pid.values() if any(g not in have for g in required))
    st.warning(f"Save **{len(items) - short}** frame{'s' if len(items) - short != 1 else ''} as full labels with "
               "only the tags you added — every other tag counts as not there."
               + (f" **{short}** stay part-reviewed: a “needs a tag” category ({', '.join(required)}) has "
                  "nothing yet." if short else ""))
    a, b = st.columns(2)
    a.button("🧹 Yes, remove them", key="tr_finish_yes", type="primary", width="stretch",
             on_click=_finish, args=(items, rows))
    b.button("Cancel", key="tr_finish_no", width="stretch",
             on_click=lambda: st.session_state.pop("tr_finish_ask", None))


def _grid(moments: list, rows: list, tag: str, only_part: bool = False) -> None:
    """Small thumbnails packed together: pick tags, tick frames, then ➕ Add / ➖ Remove."""
    st.markdown("<style>" + GRID_CSS + "</style>", unsafe_allow_html=True)
    st.html(GRID_JS, unsafe_allow_javascript=True)
    groups = TLB.chip_groups(rows)
    if not groups:
        st.info("No tag vocabulary yet — add tags in 📒 Vocabulary.")
        return
    titles = [g[0] for g in groups] + [GRID_ALL]
    if tag != "(any tag)" and st.session_state.get("tr_g_lasttag") != tag:
        # the Tag filter changed: open its category and pick it
        st.session_state["tr_g_lasttag"] = tag
        for title, _, tags in groups:
            if any(t.lower() == tag.lower() for t in tags):
                st.session_state["tr_g_cat"] = title
                if tag not in (st.session_state.get("tr_g_chosen") or []):
                    st.session_state["tr_g_chosen"] = list(st.session_state.get("tr_g_chosen") or []) + [tag]
                break
    if st.session_state.get("tr_g_cat") not in titles:
        st.session_state["tr_g_cat"] = titles[0]
    cat, in_cat = _gpicker(groups, titles, "")
    chosen = list(st.session_state.get("tr_g_chosen") or [])
    if chosen:
        ruled = sorted(TLB._never_with(chosen, rows) - set(chosen), key=str.lower)
        c1, c2 = st.columns([8, 2])
        c1.caption("Picked: **" + "**, **".join(chosen) + "**"
                   + (f" · ➕ Add also marks **{', '.join(ruled)}** as not there (your never-with rules)" if ruled else ""))
        c2.button("✖ Clear tags", key="tr_g_clear", width="stretch",
                  on_click=lambda: st.session_state.update(tr_g_chosen=[]))
    else:
        st.caption("Pick the tag(s) first, then tick frames — ticks are kept until you press a button below.")
    if not moments:
        st.info("Nothing matches these filters.")
        return
    pages = max(1, -(-len(moments) // GRID_PAGE))
    if st.session_state.get("tr_gpage", 1) > pages:
        st.session_state["tr_gpage"] = pages
    page = st.number_input(f"Page (of {pages}) — {len(moments)} frames", 1, pages, 1, key="tr_gpage") \
        if pages > 1 else 1
    items = moments[(page - 1) * GRID_PAGE: page * GRID_PAGE]
    st.caption("👆 Tap a picture to tick it, or put your finger on one and swipe sideways to tick a whole run "
               "(swipe down as you go to take in more rows) — like selecting in Photos.")
    st.caption(f"Under each frame: {'every tag' if cat == GRID_ALL else 'the **' + cat + '** tags'} you or the model have there — ● you have it · ✗ you said it "
               "isn't there · ◆ only the model has it. Tick a frame and a picked tag it doesn't have yet shows as "
               "+ tag. "
               "Frames you've only batch-tagged teach the student just those tags; the rest still come from "
               "the teacher.")
    with st.form("tr_grid_form", border=False, enter_to_submit=False):
        _gbar("top", items, rows)
        with st.container(key="tr_grid"):
            for v, m in items:
                with st.container(key=f"tr_gcard_{_k(v, m)}"):
                    p = TR.frame_path(m)
                    if p is not None and p.exists():
                        safe_image(str(p), width="stretch")
                    else:
                        st.caption("🖼️ not synced")
                    st.checkbox(format_mmss(m["t"]), key=_gtick_key(v, m), help=f"{v} · scene {m.get('scene_id')}")
                    state = _gstate(m, _gtags(m, in_cat), chosen)
                    if only_part and m.get("partial") and m.get("mine") is None:
                        state += _todo_html(TR.still_to_review(m), small=True)
                    if state:
                        st.markdown(state, unsafe_allow_html=True)
        _gbar("bottom", items, rows)
    # the tag picker and the page box again at the bottom, so a long page needs no scrolling back up
    _gpicker(groups, titles, "_b")
    if pages > 1:
        st.session_state["tr_gpage_b"] = page
        st.number_input(f"Page (of {pages}) — {len(moments)} frames", 1, pages, key="tr_gpage_b",
                        on_change=lambda: st.session_state.update(tr_gpage=st.session_state["tr_gpage_b"]))


def _gpicker(groups: list, titles: list, sfx: str) -> tuple:
    """The Category box and its tag pills (sfx "" at the top, "_b" the copy at the bottom). Both copies
    share one category (tr_g_cat) and one set of picked tags (tr_g_chosen). Returns (category, its tags)."""
    if sfx:                                   # the bottom copy follows the top one
        st.session_state["tr_g_cat" + sfx] = st.session_state["tr_g_cat"]
    chosen = list(st.session_state.get("tr_g_chosen") or [])
    with st.container(key=f"tl_wrap_tr_gpick{sfx}"):
        a, b = st.columns([2, 7])
        cat = a.selectbox("Category", titles, key="tr_g_cat" + sfx,
                          on_change=(lambda: st.session_state.update(tr_g_cat=st.session_state["tr_g_cat" + sfx]))
                          if sfx else None,
                          help=f"{GRID_ALL}: every category's tags under each frame — a final check that everything "
                               "you've corrected is right.")
        in_cat = (list(dict.fromkeys(t for g in groups for t in g[2])) if cat == GRID_ALL
                  else next(g[2] for g in groups if g[0] == cat))
        pk = "tr_g_pills_" + hashlib.md5(cat.encode()).hexdigest()[:8] + sfx
        st.session_state[pk] = [t for t in in_cat if t in chosen]

        def _pick():
            now = list(st.session_state.get("tr_g_chosen") or [])
            st.session_state["tr_g_chosen"] = [t for t in now if t not in in_cat] + list(st.session_state[pk] or [])
        with b:
            st.pills("Tags", in_cat, selection_mode="multi", key=pk, on_change=_pick,
                     help="The tags ➕ Add / ➖ Remove apply. You can pick tags from more than one category.")
    return cat, in_cat


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
        _thresholds(moments, k)


def _use_thresholds(picks: dict) -> None:
    rows, _ = TLB.load_vocabulary()
    TLB.save_vocabulary(TLB.set_thresholds(rows, picks))
    st.session_state["tr_saved_note"] = f"Thresholds set for {len(picks)} tag(s) — used from the next Modal run."


def _thresholds(moments: list, teacher: str) -> None:
    """🎚️ The threshold per tag that would have matched your labels best (needs confidences —
    batches tagged from v1.32 on)."""
    rows, _ = TLB.load_vocabulary()
    sug = TR.suggest_thresholds(moments, teacher, rows, TLB.DEFAULT_THRESHOLD)
    better = [x for x in sug if x["suggested"] != x["current"] and x["f1_best"] > x["f1_now"]]
    with st.expander(f"🎚️ Thresholds from your labels — {len(better)} tag(s) would do better", expanded=False):
        if not sug:
            st.caption("Needs frames you've labelled from a batch tagged with confidences (from v1.32 on), and at "
                       f"least {TR.MIN_FRAMES_FOR_THRESHOLD} of your frames with the tag.")
            return
        st.caption("How sure the teacher must be (0–100) for a tag to count. **Suggested** is the threshold that "
                   "would have matched your labels best on these frames. Raise = fewer wrong tags, lower = fewer "
                   "missed. Small numbers of frames can mislead — set them yourself in 📒 Vocabulary any time.")
        st.dataframe([{"Tag": x["tag"], "Frames": x["frames"], "Your frames with it": x["yours"],
                       "Now": x["current"], "F1 now": x["f1_now"], "Suggested": x["suggested"],
                       "F1 suggested": x["f1_best"]} for x in sug],
                     hide_index=True, width="stretch", key=f"tr_thr_{teacher}")
        if better:
            st.button(f"🎚️ Use the suggested thresholds for {len(better)} tag(s)", type="primary",
                      key=f"tr_thr_use_{teacher}", on_click=_use_thresholds,
                      args=({x["tag"]: x["suggested"] for x in better},))


def _test_compare(scored: list) -> None:
    """🧪 Teacher test runs: per tag, and what changed in the vocabulary between two of them (the runs
    table itself is on 🧪 Test results, with the student runs)."""
    if any(x.get("copied") for x in scored):
        st.caption("**Your frames copied** (teacher runs) = labelled frames a run answered by copying a "
                   "near-identical moment instead of tagging them (labels added after a video's vision run weren't "
                   "protected before v1.39). Older runs show 0 until `--stage test-refresh` rebuilds their summary.")
    if len(scored) < 2:
        return
    ids = [x["run"]["id"] for x in scored]
    with st.expander("🔀 Compare two teacher test runs — per tag, and what changed in the vocabulary"):
        a, b = st.columns(2)
        ia = a.selectbox("Before", ids, index=len(ids) - 2, key="tr_test_a")
        ib = b.selectbox("After", ids, index=len(ids) - 1, key="tr_test_b")
        ra = next(x for x in scored if x["run"]["id"] == ia)
        rb = next(x for x in scored if x["run"]["id"] == ib)
        ch = TR.tag_changes(ra["score"], rb["score"])
        if ch:
            st.dataframe([{"Tag": c["tag"], "Wrong": f"{c['wrong_a']} → {c['wrong_b']}",
                           "Missed": f"{c['missed_a']} → {c['missed_b']}",
                           "Errors": ("✅ " if c["change"] < 0 else "⚠️ ") + f"{c['change']:+d}"} for c in ch],
                         hide_index=True, width="stretch", key="tr_test_changes")
        else:
            st.caption("No tag changed.")
        lines = (TR.vocab_changes(ra["run"].get("vocab_csv") or "", rb["run"].get("vocab_csv") or "")
                 + TR.look_changes(ra["run"].get("look_steps"), rb["run"].get("look_steps")))
        st.markdown("**Vocabulary changes between them**")
        st.caption("  \n".join(lines[:60]) + (f"  \n… and {len(lines) - 60} more" if len(lines) > 60 else "")
                   if lines else "None — same vocabulary (so any difference is from the code or chance).")


def _batch_details(data: dict) -> None:
    """The teacher batches in detail — cost and time, tagging, scene changes, the written report (the
    per-batch scorecard itself is in the runs table on 🏋️ Training results)."""
    teacher = (TR.shown_teachers(data.get("teachers") or {}) or [TR.MAIN_TEACHER])[0]
    videos = TR.with_saved_labels(data["videos"], TLB.load_labels())
    rows = TR.batch_report(videos, teacher)
    if not any(v.get("run") for v in videos.values()):
        st.info("Times and costs per batch appear after the next export — on the server: "
                "`modal run modal_trial.py --stage results` (no GPU, about a minute).")
    pct = lambda x: None if x is None else round(x * 100)
    names = lambda r: "All batches" if r["batch"] == "All" else (f"Batch {r['batch']}" if r["batch"] != "?" else "Not recorded")
    st.caption("Teacher batches in detail — per-tag detail for a batch: pick it in the Batch box, then 📊 Scores.")
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


# ---------------------------------------------------------------------------
# 🏷️ Scene tags — the teacher's moment tags put onto your scenes (new_scene_tags.py)
# ---------------------------------------------------------------------------

def _tag_chips(tags, mark=(), colour="46,160,67", strike=False, share=None) -> str:
    out = []
    for t in tags:
        hit = t in mark or t.lower() in mark
        bg = f"rgba({colour},.18)" if hit else "rgba(128,128,128,.12)"
        border = f"rgba({colour},.8)" if hit else "rgba(128,128,128,.4)"
        deco = "text-decoration:line-through;" if (hit and strike) else ""
        tail = f' <span style="opacity:.6;font-size:.75rem">{share[t]:.2f}</span>' if share and t in share else ""
        out.append(f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:1px 8px;border-radius:999px;'
                   f'border:1px solid {border};background:{bg};font-size:.84rem;{deco}">{html.escape(t)}{tail}</span>')
    return "".join(out) or '<span style="opacity:.55;font-size:.84rem">no tags</span>'


def _scene_tags_changed(why: str, queue: bool) -> None:
    """The library's tags may differ now: drop every cached copy, and search again where it matters."""
    from library_common import clear_shared_data_caches
    st.cache_data.clear()
    clear_shared_data_caches()
    if queue:
        try:
            import jobs as J
            n = J.queue_changed(why=why)
        except Exception as e:                # never let the queue stop the build or the switch
            n, st.session_state["nst_queue_err"] = 0, str(e)
        st.session_state["nst_queued"] = n


def _scene_tags(data: dict) -> None:
    import catalogue_store as CS
    import new_scene_tags as NST

    built = NST.load()
    st.caption("New tags for every scene from the teacher's moment tags inside it (your 🎯 labels win). "
               "A tag goes on when it's on at least half of the scene's moments. Scenes with no moment keep their "
               "old tags, and scenes you corrected in the Reviewer always keep yours.")
    with st.container(key="nst_top"):
        a, b = st.columns([1, 1])
        if a.button("🏷️ Build new scene tags", key="nst_build", type="primary", width="stretch",
                    help="From this trial's results (all batches) and your current labels. Run it again after each "
                         "new export. Nothing in your catalogues changes."):
            with st.spinner("Putting the teacher's tags onto your scenes…"):
                out, summ = NST.build(data, TLB.load_labels(), NST.raw_catalogues())
                NST.save(out)
            _scene_tags_changed("scene tags rebuilt", queue=NST.mode() == "new")
            st.session_state["nst_built_note"] = summ
            st.rerun()
        with b:
            # The file is the truth (another device may have flipped it) — show what it says.
            st.session_state["nst_mode"] = NST.mode()

            def _flip():
                pick = st.session_state.get("nst_mode")
                if pick in ("old", "new") and pick != NST.mode():
                    NST.set_mode(pick)
                    _scene_tags_changed(f"scene tags: {pick}", queue=True)
                    st.session_state["nst_flip_note"] = pick
            st.segmented_control("🏷️ Scene tags", ["old", "new"], key="nst_mode", on_change=_flip,
                                 format_func={"old": "Old", "new": "🆕 New"}.get,
                                 help="Which tags the whole app uses for scenes — tag filters, 🔎 More like "
                                      "this, Recommendations and video ranking. For this copy of the app only.",
                                 disabled=not built and NST.mode() == "old")
    note = st.session_state.pop("nst_built_note", None)
    if note:
        st.success(f"Built: **{note['scenes_new']}** scene(s) in **{note['videos']}** video(s) got new tags · "
                   f"**{note['scenes_changed']}** changed · **{note['scenes_fallback']}** kept their old tags "
                   f"(no moment inside) · from {note['moments_used']} moments ({note['moments_labelled']} yours)"
                   + (f" · {note['videos_missing']} video(s) not in your library" if note["videos_missing"] else ""))
    flip = st.session_state.pop("nst_flip_note", None)
    if flip:
        st.toast(f"The app now uses the {'new' if flip == 'new' else 'old'} scene tags.", icon="🏷️")
    if "nst_queued" in st.session_state:
        n = st.session_state.pop("nst_queued")
        if n:
            st.caption(f"⚡ {n} recommendation search(es) queued again for the videos whose tags changed.")
    err = st.session_state.pop("nst_queue_err", None)
    if err:
        st.warning(f"Couldn't queue the recommendation searches ({err}) — use ⚡ on Recommendations.")
    if not built:
        st.info("No new scene tags yet — press **🏷️ Build new scene tags**.")
        return
    s = built.get("summary") or {}
    st.caption(f"Built {built.get('built')} from the trial export of {built.get('export')} · "
               f"{s.get('scenes_new', 0)} scenes with new tags · {s.get('scenes_changed', 0)} changed · "
               f"{s.get('scenes_fallback', 0)} kept their old tags · now using: "
               f"**{'🆕 new' if NST.mode() == 'new' else 'old'}** tags")

    cats = CS.all_catalogues()
    rows = NST.compare_rows(cats, built)
    if not rows:
        st.info("None of the videos with new tags are in your library here.")
        return
    part = st.segmented_control("Look at", ["compare", "similar"], key="nst_part", default="compare",
                                format_func={"compare": "🆚 Old vs new", "similar": "🔎 More like this both ways"}.get)
    if part is None:                               # pills can be deselected — keep the last pick
        part = st.session_state.get("nst_part_last", "compare")
    st.session_state["nst_part_last"] = part
    vids = sorted({r["video_id"] for r in rows})
    if st.session_state.get("nst_vid") not in vids:
        st.session_state.pop("nst_vid", None)
    if part == "similar":
        _scene_tags_similar(cats, built, rows, vids)
    else:
        _scene_tags_compare(rows, vids)


def _scene_tags_compare(rows: list, vids: list) -> None:
    import new_scene_tags as NST
    with st.expander("📈 Tags that changed most across the library"):
        ch = NST.tag_changes(rows)
        if ch:
            st.dataframe([{"Tag": t, "Added to scenes": a, "Removed from scenes": r} for t, a, r in ch[:40]],
                         hide_index=True, width="stretch")
        else:
            st.caption("No tag changed.")
    with st.container(key="nst_filters"):
        a, b = st.columns([3, 2])
        vid = a.selectbox("Video", vids, key="nst_vid")
        only = b.toggle("Only scenes that changed", key="nst_only", value=True)
    mine = [r for r in rows if r["video_id"] == vid]
    n_ch = sum(1 for r in mine if r["added"] or r["removed"])
    n_fb = sum(1 for r in mine if r["new"] is None)
    st.caption(f"{len(mine)} scenes · {n_ch} changed · {n_fb} kept their old tags (no moment inside)")
    show = [r for r in mine if (r["added"] or r["removed"])] if only else mine
    if not show:
        st.caption("No scene of this video changed.")
        return
    out = []
    for r in show:
        when = f"{format_mmss(r['start'])}–{format_mmss(r['end'])}"
        if r["new"] is None:
            new = '<span style="opacity:.6;font-size:.84rem">↩︎ keeps its old tags (no moment inside)</span>'
        else:
            new = _tag_chips(r["new"], r["added"], share=r["share"])
            if r["corrected"]:
                new += '<div style="opacity:.7;font-size:.78rem">✏️ you corrected this scene — the app keeps your tags</div>'
        moments = "" if r["new"] is None else (f'{r["moments"]} moment(s)'
                                               + (f', {r["labelled"]} yours' if r["labelled"] else ""))
        out.append(
            f'<div style="padding:6px 0;border-bottom:1px solid rgba(128,128,128,.25)">'
            f'<div style="font-size:.8rem;opacity:.75">Scene {r["scene_id"]} · {when}'
            f'{" · " + moments if moments else ""}</div>'
            f'<div><span style="display:inline-block;min-width:3rem;font-weight:600;font-size:.82rem">Old</span>'
            f'{_tag_chips(r["old"], r["removed"], colour="229,83,75", strike=True)}</div>'
            f'<div><span style="display:inline-block;min-width:3rem;font-weight:600;font-size:.82rem">New</span>'
            f'{new}</div></div>')
    st.markdown("".join(out), unsafe_allow_html=True)
    st.caption("Green = added by the new tags · red, struck through = dropped · small numbers = the share of the "
               "scene's moments that had the tag.")


def _scene_tags_similar(cats: dict, built: dict, rows: list, vids: list) -> None:
    import new_scene_tags as NST
    vid = st.selectbox("Video", vids, key="nst_vid")
    mine = [r for r in rows if r["video_id"] == vid]
    opts = [r["scene_id"] for r in mine]
    by = {r["scene_id"]: r for r in mine}
    if st.session_state.get("nst_sim_scene") not in opts:
        st.session_state.pop("nst_sim_scene", None)
    sid = st.selectbox("Scene", opts, key="nst_sim_scene",
                       format_func=lambda i: f"{format_mmss(by[i]['start'])} · scene {i} · "
                                             + ", ".join((by[i]["new"] if by[i]["new"] is not None
                                                          else by[i]["old"])[:4]))
    if sid is None:
        return
    st.caption("The best matches in other videos (up to 2 per video), on tags — rarer tags count more — and "
               "movement level. Movement is the same both ways, so any difference comes from the tags.")
    cols = st.columns(2)
    for col, which, label in ((cols[0], "old", "Old tags"), (cols[1], "new", "🆕 New tags")):
        with col:
            st.markdown(f"**{label}**")
            r = by[sid]
            ref = r["old"] if which == "old" or r["new"] is None or r["corrected"] else r["new"]
            st.markdown(_tag_chips(ref), unsafe_allow_html=True)
            hits = NST.similar_scenes((vid, sid), cats, built, which)
            if not hits:
                st.caption("No matches.")
            for score, v, s_id, tags, shared in hits:
                sc = next((x for x in cats[v]["scenes"] if x["scene_id"] == s_id), {})
                try:
                    from library_common import tc_to_seconds
                    when = format_mmss(tc_to_seconds(sc["start_tc"]))
                except (KeyError, ValueError, TypeError):
                    when = "?"
                st.markdown(
                    f'<div style="padding:4px 0;border-bottom:1px solid rgba(128,128,128,.2)">'
                    f'<div style="font-size:.8rem"><b>{score:.0%}</b> · {html.escape(v[:44])} · {when}</div>'
                    f'{_tag_chips(tags, set(shared))}</div>', unsafe_allow_html=True)


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
                sub = st.radio("Trial", ["📋 Summary", SRV.TRAINING, SRV.TEST, "🖼️ Compare tags", "📊 Scores", "🕺 Motion",
                                        "🏷️ Scene tags"], horizontal=True,
                               key="tr_sub", label_visibility="collapsed")
    if sub == "🏷️ Scene tags":
        _scene_tags(data)                      # always every batch: the new tags come from all of them
        return
    if sub == "📋 Summary":
        SRV.landing(data)
        return
    if sub == SRV.TRAINING:
        SRV.training(data)
        return
    if sub == SRV.TEST:
        SRV.test()
        return
    view = {**data, "videos": TR.in_batch(data["videos"], batch)}
    if not view["videos"]:
        st.info("No videos in this batch.")
        return
    if sub == "🖼️ Compare tags":
        _compare(view)
    elif sub == "📊 Scores":
        _scores(view)
    else:
        _motion(view)
