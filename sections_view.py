"""🧩 Sections and 🍳 Recipes — inside 📁 Media Library → 🎵 Audio (Stage 3 step 4, claude/new-matcher-design.md §C).
Sections: a track's section-type map — suggested from the song (with its 🎭 story-map roles as hints), each part's
type editable, saved per track. Recipes: what footage each type wants — one shared set (v1), edited per type."""

from __future__ import annotations

import streamlit as st

import recipes as RC
import section_types as ST
from library_common import format_mmss

SEC_STATE = "sec_edit"


# ---------------------------------------------------------------------------
# 🧩 Sections — the 🎭 story-map editor reused (draggable lines, snap to cuts, play), with section types
# ---------------------------------------------------------------------------

def _state() -> dict:
    return st.session_state.setdefault(SEC_STATE, {})


def _to_editor(regions: list) -> list:
    return [{"start": r["start"], "end": r["end"], "role": r.get("type"), "why": r.get("why", "")} for r in regions]


def _from_editor(regions: list) -> list:
    return [{"start": r["start"], "end": r["end"], "type": r.get("role") or "rhythmic", "why": r.get("why", "")}
            for r in regions]


def _fresh(tid: str, track_bits) -> tuple:
    """(regions in editor shape, source) — the saved map, else a suggestion."""
    saved = ST.saved_map(tid)
    if saved:
        return _to_editor(saved), "saved"
    track, segs, analysis = track_bits(tid)
    if analysis is None:
        return [], "none"
    import story_map as STM
    cuts = [sg["start"] for sg in segs] + [float(track["duration_sec"])]
    return _to_editor(ST.snap(ST.suggest(analysis, STM.saved_map(tid)), cuts)), "suggested"


def _load(tid: str, track_bits) -> list:
    s = _state()
    if s.get("track_id") != tid or s.get("regions") is None:
        regions, source = _fresh(tid, track_bits)
        s.update(track_id=tid, regions=regions, source=source, dirty=False, v=s.get("v", 0) + 1)
    return s["regions"]


def _set(regions: list) -> None:
    s = _state()
    s["regions"] = regions
    s["dirty"] = True
    s["v"] = s.get("v", 0) + 1


def _save(tid: str) -> None:
    s = _state()
    ST.save_map(tid, _from_editor(s.get("regions") or []), source="you" if s.get("dirty") else (s.get("source") or "you"))
    s["dirty"] = False
    s["source"] = "saved"
    st.session_state["sec_msg"] = f"Saved the sections for {tid}."


def _discard(tid: str, track_bits) -> None:
    _state().clear()
    st.session_state["sec_msg"] = "Changes dropped."


def _remove(tid: str) -> None:
    ST.save_map(tid, None)
    _state().clear()
    st.session_state["sec_msg"] = f"Removed the saved sections for {tid} — the suggestion shows again."


def _suggest_all(tracks: dict, track_bits) -> None:
    made = 0
    for tid in tracks:
        if ST.saved_map(tid):
            continue
        try:
            regions, source = _fresh(tid, track_bits)
        except Exception:          # noqa: BLE001
            continue
        if regions:
            ST.save_map(tid, _from_editor(regions), source="suggested")
            made += 1
    st.session_state["sec_msg"] = f"Suggested and saved sections for {made} track{'s' if made != 1 else ''}."
    _state().clear()


def render_sections(tracks: dict, track_bits, play=None) -> None:
    """tracks: the Media Library's slim track dict; track_bits(tid) → (track, smart segs, analysis);
    play(track, region, lead, key) plays a stretch of the song (the Media Library's ▶)."""
    st.caption("🧩 The song in section types — what each stretch *is* (Intro, Build-up, Drop…); the 🍳 recipe for "
               "each type says what footage it wants and how fast to cut. Suggested from the song with your 🎭 "
               "story-map roles as hints. Drag the lines between parts, change a part's type, ✂️ split, 🔗 join, then 💾 Save.")
    if not tracks:
        st.info("No analysed tracks yet — 🔬 Audio analysis in 🎵 Tracks first.")
        return
    saved = ST.saved_maps()
    names = sorted(tracks, key=str.lower)
    c = st.columns([3, 1.4, 1.4, 1.4])
    tid = c[0].selectbox("Track", names, key="sec_track", format_func=lambda t: f"{'✅ ' if t in saved else ''}{t}")
    missing = [t for t in names if t not in saved]
    c[1].button(f"✨ All {len(missing)} without", key="sec_suggest_all", width="stretch", disabled=not missing,
                help="Suggest and save a section map for every track that hasn't one.", on_click=_suggest_all,
                args=(tracks, track_bits))
    if msg := st.session_state.pop("sec_msg", None):
        st.success(msg)
    regions = _load(tid, track_bits)
    s = _state()
    if not regions:
        st.warning("This track can't be analysed for a suggestion yet — it may need the extra 🔬 analysis.")
        return
    try:
        track, segs, analysis = track_bits(tid)
    except Exception as e:          # noqa: BLE001
        st.error(f"Couldn't read {tid}: {e}")
        return
    c[2].button("💾 Save", key="sec_save", width="stretch", type="primary",
                disabled=not (s.get("dirty") or s.get("source") == "suggested"), on_click=_save, args=(tid,))
    with c[3].popover("⋯", width="stretch"):
        st.button("↩️ Drop my changes", key="sec_discard", width="stretch", disabled=not s.get("dirty"),
                  on_click=_discard, args=(tid, track_bits))
        st.button("🗑️ Remove the saved map", key="sec_remove", width="stretch", disabled=tid not in saved,
                  on_click=_remove, args=(tid,))
    state_line = ("**unsaved changes** — 💾 Save to keep them" if s.get("dirty") else
                  "suggested, not saved yet — 💾 Save to keep it" if s.get("source") == "suggested" else "saved ✅")
    st.caption(f"{tid} · {state_line}")
    import story_map as STM
    import story_map_editor as SME
    cuts = [sg["start"] for sg in segs] + [float(track["duration_sec"])]

    def suggest():
        return _to_editor(ST.snap(ST.suggest(analysis, STM.saved_map(tid)), cuts)) if analysis else regions

    SME.render_editor(track, segs, ns="sec", get_regions=lambda: _state().get("regions") or [], set_regions=_set,
                      version=s.get("v", 0), analysis=analysis, play=play, roles=ST.types_as_roles(), suggest=suggest,
                      allow_none=False, thing="type", clear=False)


# ---------------------------------------------------------------------------
# 🍳 Recipes
# ---------------------------------------------------------------------------

def _vocab_tags() -> tuple:
    """(moment tags for recipes, every tag incl. scene-only ones) from the vocabulary."""
    try:
        import training_labels as TLB
        import trial_logic as TL
        rows, _ = TLB.load_vocabulary()
        all_tags = [r["tag"] for r in rows]
        moment = [t for t in all_tags if not TLB.is_hidden(t) and not TL.not_in_f1(t)]
        return moment, all_tags
    except Exception:          # noqa: BLE001
        return [], []


LEVELS = {"a bit": 0.3, "want": 0.6, "important": 0.8, "must": 1.0}


def _level_of(w: float) -> str:
    return min(LEVELS, key=lambda k: abs(LEVELS[k] - float(w)))


def _save_recipe(type_id: str, ns: str) -> None:
    lo, hi = st.session_state.get(ns + "move", (0.0, 1.0))
    trend_label = st.session_state.get(ns + "trend")
    trend = next((k for k, v in RC.TRENDS.items() if v == trend_label), "flat")
    shot_label = st.session_state.get(ns + "shot")
    shot = next((k for k, v in RC.SHOT_CHANGES.items() if v == shot_label), "allow")
    tags = {}
    for t in st.session_state.get(ns + "wants", []):
        lvl = st.session_state.get(ns + "lvl_" + t, "want")
        tags[t] = LEVELS.get(lvl, 0.6)
    p_lo, p_hi = st.session_state.get(ns + "pace", (2.0, 4.0))
    recipe = {"movement": {"lo": lo, "hi": hi, "trend": trend}, "beat_lock": st.session_state.get(ns + "beat", 0.5),
              "pace": {"lo": p_lo, "hi": p_hi},
              "tags": tags, "avoid": st.session_state.get(ns + "avoid", []),
              "roles": st.session_state.get(ns + "roles", []),
              "scene_require": st.session_state.get(ns + "req", []), "scene_prefer": st.session_state.get(ns + "pref", []),
              "description": st.session_state.get(ns + "desc", ""), "shot_change": shot}
    RC.save(type_id, recipe)
    st.session_state["rc_msg"] = f"Saved the {ST.label(type_id)} recipe."
    st.session_state["rc_v"] = st.session_state.get("rc_v", 0) + 1


def _reset_recipe(type_id: str) -> None:
    RC.reset(type_id)
    st.session_state["rc_msg"] = f"{ST.label(type_id)} is back to the default recipe."
    st.session_state["rc_v"] = st.session_state.get("rc_v", 0) + 1


def render_recipes() -> None:
    st.caption("🍳 What footage each section type wants: a movement band (library scale, as the 🧭 timeline shows it) "
               "and whether it builds or falls, how tightly movement should lock to the beat, the moment tags to "
               "look for (with weights) and to avoid, scene tags to require or prefer, an optional description "
               "matched against the pictures, whether the shot size may change, and the clip length the 🤖 new "
               "matcher cuts at. One shared set for now — variations per cutting style can come later.")
    if msg := st.session_state.pop("rc_msg", None):
        st.success(msg)
    moment_tags, all_tags = _vocab_tags()
    all_recipes = RC.load_all()
    v = st.session_state.get("rc_v", 0)
    for t in ST.TYPES:
        tid = t["id"]
        r = all_recipes[tid]
        custom = RC.is_custom(tid)
        title = f"{t['emoji']} {t['name']}" + (" · edited" if custom else "") + f" — {RC.summary(r)}"
        with st.expander(title, expanded=False):
            st.caption(t["help"])
            ns = f"rc_{tid}_{v}_"
            c = st.columns([3, 2, 2])
            c[0].slider("Movement band", 0.0, 1.0, (float(r["movement"]["lo"]), float(r["movement"]["hi"])), 0.05,
                        key=ns + "move", help="0 = the library's calmest footage, 1 = its busiest.")
            pc = r.get("pace") or {"lo": 2.0, "hi": 4.0}
            st.slider("Clip length (s) — the 🤖 new matcher cuts this section at this pace", 0.25, 20.0,
                      (float(pc["lo"]), float(pc["hi"])), 0.25, key=ns + "pace",
                      help="A building section shortens its clips from the long end to the short end; a falling "
                           "one lengthens them. Classic's own cutting settings are unaffected.")
            c[1].radio("Over the section", list(RC.TRENDS.values()), index=list(RC.TRENDS).index(r["movement"]["trend"]),
                       key=ns + "trend", horizontal=True)
            c[2].slider("Beat lock", 0.0, 1.0, float(r["beat_lock"]), 0.1, key=ns + "beat",
                        help="How tightly the movement should line up with the beat.")
            unknown = RC.unknown_tags(r, all_tags) if all_tags else []
            if unknown:
                st.caption("⚠️ Not in your vocabulary (spelling?): " + ", ".join(unknown))
            tag_opts = sorted(set(moment_tags) | set(r["tags"]) | set(r["avoid"]), key=str.lower)
            wants = st.multiselect("Wants (moment tags)", tag_opts, default=[t for t in r["tags"] if t in tag_opts],
                                   key=ns + "wants", help="What should be on screen in this section.")
            if wants:
                st.caption("How much each matters")
                wc = st.columns(min(3, len(wants)))
                for k, t in enumerate(wants):
                    wc[k % len(wc)].select_slider(t, list(LEVELS), value=_level_of(r["tags"].get(t, 0.6)), key=ns + "lvl_" + t)
            st.multiselect("Avoid", tag_opts, default=[x for x in r["avoid"] if x in tag_opts], key=ns + "avoid")
            import scene_roles as SR
            role_opts = [x["id"] for x in SR.all_roles()]
            st.multiselect("🎭 Prefer clips marked with", role_opts, default=[x for x in r.get("roles") or [] if x in role_opts],
                           key=ns + "roles", format_func=lambda x: SR.label(x),
                           help="Scene roles you've marked in the Reviewer (🎬 Intro, 🏁 Outro, your own) — clips with one of "
                                "these get a bonus in this section.")
            sc = st.columns(2)
            scene_opts = sorted(set(all_tags) | set(r["scene_require"]) | set(r["scene_prefer"]), key=str.lower)
            sc[0].multiselect("Scene tags — require", scene_opts, default=r["scene_require"], key=ns + "req",
                              help="Only footage from scenes with these (a filter).")
            sc[1].multiselect("Scene tags — prefer", scene_opts, default=r["scene_prefer"], key=ns + "pref")
            d = st.columns([3, 2])
            d[0].text_input("Description (optional)", value=r["description"], key=ns + "desc",
                            placeholder="e.g. slow, close, sensual — compared with the pictures")
            d[1].radio("Shot size across the section", list(RC.SHOT_CHANGES.values()),
                       index=list(RC.SHOT_CHANGES).index(r["shot_change"]), key=ns + "shot")
            b = st.columns([1.2, 1.2, 4])
            b[0].button("💾 Save", key=f"{ns}save", width="stretch", type="primary", on_click=_save_recipe, args=(tid, ns))
            b[1].button("↩️ Reset", key=f"{ns}reset", width="stretch", disabled=not custom, on_click=_reset_recipe,
                        args=(tid,))
