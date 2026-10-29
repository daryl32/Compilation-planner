"""
🎯 Training Labels — hand-check random frames and scene changes, to build the gold set the new
models are trained and judged against (claude/ml-redesign-plan.md, Phase 4).

  🎯 Label       one card at a time, big buttons (iPad first):
                 • Tag cards — a random frame from the preview copy, its scene's current tags
                   ticked. Tick what IS there, untick what isn't, ✅ Save & next. The saved set is
                   complete: any vocabulary tag not ticked counts as "not there".
                   ▶️ plays a few seconds around the frame (for motion tags).
                 • Scene-change cards — the frames either side of a point plus a short clip:
                   a cut the detector made, the sharpest change inside a long scene, or a random
                   point. ✂️ Yes / ➖ No / 🤷 Not sure.
                 ⏭️ Skip (bad frame) is remembered, so that frame never comes back.
  📊 Progress    how many of each, which tags still need examples, recent answers (with ↩️ undo).
  📒 Vocabulary  tag_vocabulary.csv — edit it here, upload your Excel copy, or download it.
  🧪 Trial results  the latest Modal trial (trial_view.py): every tagged frame with your tags, the
                 old tags and each model's tags side by side; motion curves and cuts; the report.

Answers go to training_labels.json in the catalogue folder (saved to Drive with the other
edits). Nothing here changes scene tags — the Reviewer still owns those.
"""

import io
import math

import streamlit as st

from config import CATALOGUE_DIR  # noqa: F401  (paths come from config — see coding rules)
from library_common import proxy_path, library_ranges, safe_image, format_mmss
import catalogue_store as CS
import training_labels as TL

from page_setup import page_setup, top_button, wrap_css, menu_css

page_setup("Training Labels", "pages/6_Training_Labels.py", title="🎯 Training Labels", anchor="tl-top")
top_button("tl-top", wrap_css("tl_wrap", 120) + menu_css()
           # Big, easy-to-tap answer buttons and chips.
           + "[class*='st-key-tl_wrap_act'] button{min-height:3.1rem;font-size:1.05rem}"
           "[class*='st-key-tl_card'] [data-testid='stButtonGroup'] button{min-height:2.4rem;font-size:.98rem}")

MIXES = {"mixed": "🎲 Mixed", "tags": "🏷️ Tags only", "boundary": "✂️ Scene changes only"}
VERDICTS = {"cut": "✂️ Yes, new scene", "no_cut": "➖ No, same scene", "unsure": "🤷 Not sure"}
SOURCE_QUESTION = {
    "detected": "The scene detector cut here. **Is this really a new scene?**",
    "peak": "The sharpest picture change inside a long scene. **Should a new scene start here?**",
    "random": "A random point. **Should a new scene start here?**",
}
TARGET_PER_TAG = 20   # examples per tag the Progress view aims for

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Training Labels")
mix = st.sidebar.radio("Cards", list(MIXES), format_func=MIXES.get, key="tl_mix")
deck_size = st.sidebar.select_slider("Cards per round", [10, 20, 30, 50], value=20, key="tl_size")
use_range = st.sidebar.toggle("Use library range", value=True, key="tl_range",
                              help="Only pick frames inside each video's Media Library time range.")
st.sidebar.caption("New cards lean toward videos with the fewest labels and scenes nobody has "
                   "reviewed yet.")

with st.container(key="mainnav_tl"):   # the page's main menu — styled as tabs (menu_css)
    view = st.radio("View", ["🎯 Label", "📊 Progress", "📒 Vocabulary", "🧪 Trial results"], horizontal=True,
                    key="tl_view", label_visibility="collapsed")

vocab, vocab_source = TL.load_vocabulary()


def has_proxy(video_id: str) -> bool:
    try:
        return proxy_path(video_id).exists()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# 🎯 Label
# ---------------------------------------------------------------------------

def new_deck() -> None:
    items = TL.load_labels()
    st.session_state["tl_deck"] = TL.build_deck(
        CS.video_index(), deck_size, mix, items, library_ranges() if use_range else {},
        curves=CS.motion_curves, has_media=has_proxy)
    st.session_state["tl_pos"] = 0
    st.session_state["tl_deck_conf"] = (mix, deck_size, use_range)


def answer(card: dict, **kw) -> None:
    item = TL.save_label(card, **kw)
    st.session_state.setdefault("tl_done", []).append(item["id"])
    st.session_state["tl_pos"] = st.session_state.get("tl_pos", 0) + 1


def go_back() -> None:
    done = st.session_state.get("tl_done") or []
    if done:
        TL.remove_label(done.pop())
    st.session_state["tl_pos"] = max(0, st.session_state.get("tl_pos", 0) - 1)


def frame(video_id: str, t: float):
    return TL.extract_frame(proxy_path(video_id), t)


def clip(video_id: str, a: float, b: float) -> None:
    st.video(str(proxy_path(video_id)), start_time=max(0, int(a)), end_time=int(math.ceil(b)), loop=True)


def tag_card(card: dict) -> None:
    has_motion_tags = any(r["kind"] == "motion" for r in vocab)
    img = frame(card["video_id"], card["t"])
    if img:
        safe_image(str(img), width="stretch")
    else:
        st.warning("Couldn't read this frame from the preview copy — ⏭️ Skip it.")
    with st.expander("▶️ Play a few seconds around this frame", expanded=has_motion_tags):
        half = TL.MOTION_CLIP_SEC / 2
        clip(card["video_id"], card["t"] - half, card["t"] + half)
        st.caption("For movement tags (kind = motion in the vocabulary).")

    st.markdown("**Tick every tag that's there** — anything left unticked counts as *not there*.")
    known = {r["tag"]: r for r in vocab}
    picked = []
    for gi, (title, single, tags) in enumerate(TL.chip_groups(vocab, extra=card.get("suggested") or [])):
        notes = [f"**{t}** — {known[t]['description']}" for t in tags if t in known and known[t]["description"]]
        key = f"tl_{card['id']}_{gi}"
        default = [t for t in card.get("suggested") or [] if t in tags]
        if single:
            val = st.pills(title, tags, selection_mode="single", default=default[0] if default else None,
                           key=key, help="  \n".join(notes) or "Pick one, or none if none fits.")
            picked += [val] if val else []
        else:
            val = st.pills(title, tags, selection_mode="multi", default=default, key=key,
                           help="  \n".join(notes) or None)
            picked += list(val or [])

    with st.container(key="tl_wrap_act"):
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.button("✅ Save & next", type="primary", width="stretch", key=f"tl_save_{card['id']}",
                  on_click=answer, args=(card,), kwargs={"tags": picked})
        c2.button("⏭️ Skip", width="stretch", key=f"tl_skip_{card['id']}", help="Bad frame (blurry, black, "
                  "a transition…) — it won't come up again.",
                  on_click=answer, args=(card,), kwargs={"skipped": True, "reason": "bad frame"})
        c3.button("↩️ Back", width="stretch", key=f"tl_back_{card['id']}", on_click=go_back,
                  disabled=not st.session_state.get("tl_done"), help="Undo the last answer.")


def boundary_card(card: dict) -> None:
    vid, t = card["video_id"], card["t"]
    st.markdown(SOURCE_QUESTION.get(card.get("source"), SOURCE_QUESTION["random"]))
    st.caption("Say yes when you'd want a clip to start here — a camera cut, or a clear change of "
               "action or position. A camera move within the same action is a no.")
    a, b = st.columns(2)
    with a:
        img = frame(vid, max(0.0, t - 0.5))
        if img:
            safe_image(str(img), caption="Just before", width="stretch")
    with b:
        img = frame(vid, t + 0.5)
        if img:
            safe_image(str(img), caption="Just after", width="stretch")
    clip(vid, t - TL.BOUNDARY_PAD_SEC, t + TL.BOUNDARY_PAD_SEC)
    st.caption(f"The point is {TL.BOUNDARY_PAD_SEC:g} s into the clip.")
    with st.container(key="tl_wrap_act"):
        cols = st.columns(5)
        for col, (v, lab) in zip(cols[:3], VERDICTS.items()):
            col.button(lab, type="primary" if v == "cut" else "secondary", width="stretch",
                       key=f"tl_{v}_{card['id']}", on_click=answer, args=(card,), kwargs={"verdict": v})
        cols[3].button("⏭️ Skip", width="stretch", key=f"tl_bskip_{card['id']}",
                       on_click=answer, args=(card,), kwargs={"skipped": True, "reason": "bad clip"})
        cols[4].button("↩️ Back", width="stretch", key=f"tl_bback_{card['id']}", on_click=go_back,
                       disabled=not st.session_state.get("tl_done"), help="Undo the last answer.")


def label_view() -> None:
    if not vocab:
        st.info("No tags yet — add some in 📒 Vocabulary (or the Reviewer's labels) first.")
        return
    if st.session_state.get("tl_deck_conf") != (mix, deck_size, use_range) or "tl_deck" not in st.session_state:
        new_deck()
    deck, pos = st.session_state["tl_deck"], st.session_state.get("tl_pos", 0)
    sm = TL.summary(TL.load_labels())
    st.caption(f"So far: 🏷️ {sm['tags']} frames tagged · ✂️ {sm['boundaries']} scene changes checked · "
               f"🎬 {sm['videos']} videos · vocabulary from **{vocab_source}**")
    if not deck:
        st.info("No cards to show — no video has a preview copy here yet (or every frame is labelled).")
        return
    if pos >= len(deck):
        st.success(f"Round done — {len(deck)} cards. 🎉")
        st.button("🔄 New round", type="primary", on_click=new_deck, key="tl_new_done")
        return
    card = deck[pos]
    st.progress(pos / len(deck), text=f"Card {pos + 1} of {len(deck)}")
    sc = "" if card.get("scene_id") is None else f" · scene {card['scene_id']}"
    extra = " · not reviewed yet" if card["type"] == "tags" and not card.get("reviewed") else ""
    st.caption(f"{'🏷️ Tags' if card['type'] == 'tags' else '✂️ Scene change'} · **{card['video_id']}** · "
               f"{format_mmss(card['t'])}{sc}{extra}")
    with st.container(key="tl_card", border=True):
        (tag_card if card["type"] == "tags" else boundary_card)(card)
    st.button("🔄 New round", key="tl_new", on_click=new_deck, help="Deal a fresh set of cards.")


# ---------------------------------------------------------------------------
# 📊 Progress
# ---------------------------------------------------------------------------

def progress_view() -> None:
    items = TL.load_labels()
    sm = TL.summary(items)
    c = st.columns(5)
    c[0].metric("Frames tagged", sm["tags"])
    c[1].metric("Tags changed", sm["changed"], help="Frames where your tags differ from the scene's "
                "current tags — how often the old tagger was wrong.")
    c[2].metric("Scene changes", sm["boundaries"], help=f"{sm['cuts']} said yes, new scene.")
    c[3].metric("Videos", sm["videos"])
    c[4].metric("Skipped", sm["skipped"])

    st.subheader("Examples per tag")
    st.caption(f"Each tag wants about {TARGET_PER_TAG} frames where it's there. Fewest first.")
    rows = [{"Tag": r["tag"], "Kind": r["kind"], "Examples": sm["per_tag"].get(r["tag"], 0),
             "Big model may try": "yes" if r["teacher"] else "no"} for r in vocab]
    rows.sort(key=lambda r: (r["Examples"], r["Tag"].lower()))
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch", column_config={
            "Examples": st.column_config.ProgressColumn("Examples", min_value=0, max_value=TARGET_PER_TAG,
                                                        format="%d")})

    st.subheader("Recent answers")
    recent = list(reversed(items))[:15]
    if not recent:
        st.caption("Nothing yet.")
    for n, it in enumerate(recent):
        if it.get("skipped"):
            what = f"⏭️ skipped ({it.get('reason')})"
        elif it["type"] == "tags":
            what = "🏷️ " + (", ".join(it.get("tags") or []) or "no tags")
        else:
            what = VERDICTS.get(it.get("verdict"), "?") + f" ({it.get('source')})"
        with st.container(key=f"tl_wrap_recent_{n}"):
            a, b = st.columns([5, 1])
            a.write(f"**{it['video_id']}** · {format_mmss(it.get('t') or 0)} · {what}")
            b.button("↩️ Undo", key=f"tl_undo_{it['id']}", on_click=TL.remove_label, args=(it["id"],),
                     width="stretch")


# ---------------------------------------------------------------------------
# 📒 Vocabulary
# ---------------------------------------------------------------------------

NEW_TAG = "➕ New tag"


def tag_editor() -> None:
    """Add, change, rename or delete one tag — a form, easier than the table on an iPad."""
    names = [r["tag"] for r in vocab]
    if "tl_ed_pick_next" in st.session_state:          # after a save / rename / delete
        st.session_state["tl_ed_pick"] = st.session_state.pop("tl_ed_pick_next")
    if st.session_state.get("tl_ed_pick") not in [NEW_TAG] + names:
        st.session_state.pop("tl_ed_pick", None)
    pick = st.selectbox("Tag to edit", [NEW_TAG] + names, key="tl_ed_pick")
    cur = next((r for r in vocab if r["tag"] == pick), None) or {
        "tag": "", "group": "", "kind": "frame", "one_of": "", "teacher": True, "description": "",
        "not_to_be_confused_with": "", "search_words": []}
    groups = sorted({r["group"] for r in vocab if r["group"]}, key=str.lower)
    sets = sorted({r["one_of"] for r in vocab if r["one_of"]}, key=str.lower)
    k = f"tl_ed_{pick}"                       # fresh widgets for each tag picked
    with st.container(border=True, key="tl_wrap_ed"):
        a, b = st.columns(2)
        name = a.text_input("Tag name", cur["tag"], key=f"{k}_name")
        kind = b.radio("Kind", list(TL.KINDS), index=list(TL.KINDS).index(cur["kind"]), horizontal=True,
                       key=f"{k}_kind", format_func={"frame": "🖼️ frame — visible in one still",
                                                     "motion": "🎞️ motion — needs a few seconds"}.get)
        a, b = st.columns(2)
        g_opts = groups + ["(new group…)"]
        g_pick = a.selectbox("Group", g_opts, index=g_opts.index(cur["group"]) if cur["group"] in groups else
                             (len(g_opts) - 1 if cur["group"] else 0), key=f"{k}_group")
        group = a.text_input("New group name", cur["group"] if cur["group"] not in groups else "",
                             key=f"{k}_group_new") if g_pick == "(new group…)" else g_pick
        s_opts = ["(none — can go with any tag)"] + sets + ["(new pick-one set…)"]
        s_pick = b.selectbox("Pick-one set", s_opts,
                             index=s_opts.index(cur["one_of"]) if cur["one_of"] in sets else 0, key=f"{k}_set",
                             help="Tags in the same set can't both be true (e.g. shot size) — one row on the cards.")
        one_of = (b.text_input("New set name", key=f"{k}_set_new") if s_pick == "(new pick-one set…)"
                  else "" if s_pick.startswith("(none") else s_pick)
        desc = st.text_area("Description — what the model should look for", cur["description"], height=70,
                            key=f"{k}_desc")
        alikes_now = [x.strip() for x in str(cur["not_to_be_confused_with"]).split(";") if x.strip()]
        alikes = st.multiselect("Not to be confused with", [n for n in names if n != cur["tag"]],
                                default=[x for x in alikes_now if x in names], key=f"{k}_alike")
        words = st.text_input("Search words (separated by ;)", "; ".join(cur["search_words"]), key=f"{k}_words")
        teacher = st.toggle("The big model may try this tag", cur["teacher"], key=f"{k}_teacher",
                            help="Off = learned only from your labels (for tags the 32B gets wrong).")
        c1, c2 = st.columns(2)
        save = c1.button("💾 Save tag" if pick != NEW_TAG else "➕ Add tag", type="primary", width="stretch",
                         key=f"{k}_save")
        delete = pick != NEW_TAG and c2.button("🗑️ Delete tag", width="stretch", key=f"{k}_del",
                                               help="Your saved labels keep the name; it just leaves the list.")
    if save:
        row = {"tag": name, "group": group, "kind": kind, "one_of": one_of, "teacher": teacher,
               "description": desc, "not_to_be_confused_with": "; ".join(alikes), "search_words": words}
        try:
            rows = TL.upsert_tag(vocab, row, None if pick == NEW_TAG else pick)
        except ValueError as e:
            st.error(str(e))
            return
        TL.save_vocabulary(rows)
        clean = " ".join(name.split())
        msg = f"Saved “{clean}”."
        if pick != NEW_TAG and clean != pick:
            n = TL.rename_in_labels(pick, clean)
            msg = f"Renamed “{pick}” to “{clean}”" + (f" — and in {n} of your labelled frames." if n else ".")
        st.session_state["tl_vocab_note"] = msg
        st.session_state["tl_ed_pick_next"] = clean
        st.rerun()
    if delete:
        TL.save_vocabulary(TL.delete_tag(vocab, pick))
        st.session_state["tl_vocab_note"] = f"Deleted “{pick}” from the tag list."
        st.session_state["tl_ed_pick_next"] = NEW_TAG
        st.rerun()


def vocab_view() -> None:
    st.caption("The tag list the new models learn. **kind**: *frame* = visible in one still, *motion* = "
               "needs a few seconds of movement. **one_of**: a name shared by tags that can't both be true "
               "(e.g. shot size) — they become a pick-one row on the cards. **teacher**: may the big model "
               "try this tag (off = learned only from your labels). **search_words**: words you'd type to "
               "find it, separated by ;")
    if vocab_source != TL.VOCAB_FILE.name:
        st.info("No tag_vocabulary.csv yet — showing the Reviewer's labels. 💾 Save creates the file.")

    up = st.file_uploader("Upload your tag_vocabulary.csv (from Excel)", type=["csv"], key="tl_upload")
    if up is not None:
        try:
            rows = TL.parse_vocabulary(up.getvalue().decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError) as e:
            st.error(f"Couldn't read that file: {e}")
            rows = []
        if rows:
            st.write(f"{len(rows)} tags in the file.")
            if st.button(f"💾 Use these {len(rows)} tags", type="primary", key="tl_up_save"):
                n = TL.save_vocabulary(rows)
                st.success(f"Saved {n} tags — they're on the next cards. (Saved to Drive with your other edits.)")
                st.rerun()

    note = st.session_state.pop("tl_vocab_note", None)
    if note:
        st.success(note)
    st.caption("Changes are used from the **next batch you pack** (pack_for_modal.py copies the tag list), "
               "and straight away on the 🎯 Label cards.")
    tag_editor()

    with st.expander("Edit as a table (all tags at once)"):
        table = [{**r, "search_words": "; ".join(r["search_words"])} for r in vocab]
        edited = st.data_editor(
            table, num_rows="dynamic", hide_index=True, width="stretch", key="tl_vocab_editor",
            column_order=TL.VOCAB_COLUMNS,
            column_config={
                "tag": st.column_config.TextColumn("tag", required=True),
                "kind": st.column_config.SelectboxColumn("kind", options=list(TL.KINDS), default="frame"),
                "teacher": st.column_config.CheckboxColumn("teacher", default=True),
            })
        if st.button("💾 Save table", type="primary", key="tl_vocab_save", width="stretch"):
            n = TL.save_vocabulary(list(edited))
            st.session_state["tl_vocab_note"] = f"Saved {n} tags."
            st.rerun()
    st.download_button("⬇️ Download as CSV (for Excel)", "﻿" + TL.vocabulary_csv(vocab),
                      file_name="tag_vocabulary.csv", mime="text/csv", key="tl_vocab_dl", width="stretch")


def trial_view() -> None:
    import trial_view as TV
    TV.render()


{"🎯 Label": label_view, "📊 Progress": progress_view, "📒 Vocabulary": vocab_view,
 "🧪 Trial results": trial_view}[view]()
