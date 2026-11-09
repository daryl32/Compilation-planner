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
# 🧩 Sections
# ---------------------------------------------------------------------------

def _state() -> dict:
    return st.session_state.setdefault(SEC_STATE, {})


def _load_regions(tid: str, track_bits) -> list:
    s = _state()
    if s.get("track_id") != tid or s.get("regions") is None:
        saved = ST.saved_map(tid)
        regions = saved
        source = "saved"
        if not regions:
            track, segs, analysis = track_bits(tid)
            if analysis is not None:
                import story_map as STM
                cuts = [sg["start"] for sg in segs] + [float(track["duration_sec"])]
                regions = ST.snap(ST.suggest(analysis, STM.saved_map(tid)), cuts)
                source = "suggested"
            else:
                regions = []
                source = "none"
        s.update(track_id=tid, regions=regions, source=source, dirty=False, v=s.get("v", 0) + 1)
    return s["regions"]


def _set_regions(regions: list, source: str = None) -> None:
    s = _state()
    s["regions"] = regions
    s["dirty"] = True
    if source:
        s["source"] = source
    s["v"] = s.get("v", 0) + 1


def _suggest_again(tid: str, track_bits) -> None:
    track, segs, analysis = track_bits(tid)
    if analysis is None:
        st.session_state["sec_msg"] = "This track can't be analysed for a suggestion."
        return
    import story_map as STM
    cuts = [sg["start"] for sg in segs] + [float(track["duration_sec"])]
    _set_regions(ST.snap(ST.suggest(analysis, STM.saved_map(tid)), cuts), "suggested")


def _save(tid: str) -> None:
    s = _state()
    ST.save_map(tid, s.get("regions") or [], source=s.get("source") or "you")
    s["dirty"] = False
    s["source"] = "saved"
    st.session_state["sec_msg"] = f"Saved the sections for {tid}."


def _set_type(i: int, key: str) -> None:
    s = _state()
    pick = st.session_state.get(key)
    tid = next((t for t in ST.TYPE_IDS if ST.label(t) == pick), None)
    if tid and i < len(s.get("regions") or []):
        _set_regions(ST.set_type(s["regions"], i, tid))


def _merge(i: int) -> None:
    s = _state()
    _set_regions(ST.merge_next(s["regions"], i))


def _split(i: int) -> None:
    s = _state()
    r = s["regions"][i]
    _set_regions(ST.split(s["regions"], i, round((r["start"] + r["end"]) / 2, 3)))


def _suggest_all(tracks: dict, track_bits) -> None:
    made = 0
    for tid in tracks:
        if ST.saved_map(tid):
            continue
        try:
            track, segs, analysis = track_bits(tid)
        except Exception:          # noqa: BLE001
            continue
        if analysis is None:
            continue
        import story_map as STM
        cuts = [sg["start"] for sg in segs] + [float(track["duration_sec"])]
        regions = ST.snap(ST.suggest(analysis, STM.saved_map(tid)), cuts)
        if regions:
            ST.save_map(tid, regions, source="suggested")
            made += 1
    st.session_state["sec_msg"] = f"Suggested and saved sections for {made} track{'s' if made != 1 else ''}."
    _state().clear()


def render_sections(tracks: dict, track_bits) -> None:
    """tracks: the Media Library's slim track dict; track_bits(tid) → (track, smart segs, analysis)."""
    st.caption("🧩 The song in section types — what each stretch of the track *is* (Intro, Build-up, Drop…). The 🍳 "
               "recipe for each type says what footage it wants. Suggested from the song's structure with your 🎭 "
               "story-map roles as hints; change any part's type and save. Roles stay as they are.")
    if not tracks:
        st.info("No analysed tracks yet — 🔬 Audio analysis in 🎵 Tracks first.")
        return
    saved = ST.saved_maps()
    names = sorted(tracks, key=str.lower)
    c = st.columns([3, 1.4, 1.4])
    tid = c[0].selectbox("Track", names, key="sec_track",
                         format_func=lambda t: f"{'✅ ' if t in saved else ''}{t}")
    c[1].button("✨ Suggest again", key="sec_suggest", width="stretch", on_click=_suggest_again, args=(tid, track_bits))
    missing = [t for t in names if t not in saved]
    c[2].button(f"✨ All {len(missing)} without", key="sec_suggest_all", width="stretch", disabled=not missing,
                help="Suggest and save a section map for every track that hasn't one.",
                on_click=_suggest_all, args=(tracks, track_bits))
    if msg := st.session_state.pop("sec_msg", None):
        st.success(msg)
    regions = _load_regions(tid, track_bits)
    s = _state()
    if not regions:
        st.warning("This track can't be analysed for a suggestion yet — it may need the extra 🔬 analysis.")
        return
    try:
        track, segs, _ = track_bits(tid)
    except Exception as e:          # noqa: BLE001
        st.error(f"Couldn't read {tid}: {e}")
        return
    import story_map_editor as SME
    chart_regions = [{"start": r["start"], "end": r["end"], "role": r["type"]} for r in regions]
    st.plotly_chart(SME.story_chart(track, chart_regions, segs, ST.types_as_roles(), height=190),
                    width="stretch", config={"displayModeBar": False}, key=f"sec_chart_{tid}_{s.get('v', 0)}")
    tot = ST.totals(regions)
    st.caption(" · ".join(f"{ST.label(t)} {format_mmss(sec)}" for t, sec in sorted(tot.items(), key=lambda kv: -kv[1]))
               + (" · **unsaved changes**" if s.get("dirty") else
                  (" · suggested, not saved yet" if s.get("source") == "suggested" else "")))
    labels = [ST.label(t) for t in ST.TYPE_IDS]
    v = s.get("v", 0)
    for i, r in enumerate(regions):
        cols = st.columns([0.5, 1.6, 2.6, 2.6, 0.7, 0.7], vertical_alignment="center")
        cols[0].markdown(f"**{i + 1}**")
        cols[1].markdown(f"{format_mmss(r['start'])}–{format_mmss(r['end'])}")
        key = f"sec_type_{tid}_{i}_{v}"
        cols[2].selectbox("Type", labels, index=ST.TYPE_IDS.index(r["type"]) if r["type"] in ST.TYPE_IDS else 0,
                          key=key, label_visibility="collapsed", on_change=_set_type, args=(i, key))
        cols[3].caption(r.get("why") or "")
        cols[4].button("✂️", key=f"sec_split_{tid}_{i}_{v}", help="Split this part in two", on_click=_split, args=(i,))
        cols[5].button("🔗", key=f"sec_merge_{tid}_{i}_{v}", help="Join with the next part", disabled=i + 1 >= len(regions),
                       on_click=_merge, args=(i,))
    b = st.columns([1.5, 1.5, 4])
    b[0].button("💾 Save", key="sec_save", width="stretch", type="primary", disabled=not (s.get("dirty") or s.get("source") == "suggested"),
                on_click=_save, args=(tid,))
    if tid in saved:
        b[1].button("🗑️ Remove saved", key="sec_remove", width="stretch", on_click=lambda: (ST.save_map(tid, None), _state().clear()))


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


def _edited_rows(ns: str) -> list:
    """The tag table as edited: the base rows with st.data_editor's edits applied."""
    base = list(st.session_state.get(ns + "tags_base") or [])
    ed = st.session_state.get(ns + "tags_df") or {}
    if not isinstance(ed, dict):
        return base
    for i, changes in (ed.get("edited_rows") or {}).items():
        try:
            base[int(i)] = {**base[int(i)], **changes}
        except (IndexError, ValueError):
            continue
    for row in ed.get("added_rows") or []:
        base.append({"Tag": row.get("Tag", ""), "Weight": row.get("Weight", 0.5)})
    for i in sorted((int(i) for i in ed.get("deleted_rows") or []), reverse=True):
        if 0 <= i < len(base):
            del base[i]
    return base


def _save_recipe(type_id: str, ns: str) -> None:
    lo, hi = st.session_state.get(ns + "move", (0.0, 1.0))
    trend_label = st.session_state.get(ns + "trend")
    trend = next((k for k, v in RC.TRENDS.items() if v == trend_label), "flat")
    shot_label = st.session_state.get(ns + "shot")
    shot = next((k for k, v in RC.SHOT_CHANGES.items() if v == shot_label), "allow")
    tags = {}
    for row in _edited_rows(ns):
        t = str(row.get("Tag") or "").strip()
        try:
            w = float(row.get("Weight"))
        except (TypeError, ValueError):
            continue
        if t:
            tags[t] = w
    p_lo, p_hi = st.session_state.get(ns + "pace", (2.0, 4.0))
    recipe = {"movement": {"lo": lo, "hi": hi, "trend": trend}, "beat_lock": st.session_state.get(ns + "beat", 0.5),
              "pace": {"lo": p_lo, "hi": p_hi},
              "tags": tags, "avoid": st.session_state.get(ns + "avoid", []),
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
    import pandas as pd
    st.caption("🍳 What footage each section type wants: a movement band (library scale, as the 🧭 timeline shows it) "
               "and whether it builds or falls, how tightly movement should lock to the beat, the moment tags to "
               "look for (with weights) and to avoid, scene tags to require or prefer, an optional description "
               "matched against the pictures, and whether the shot size may change. One shared set for now — "
               "variations per cutting style can come later.")
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
            rows = [{"Tag": k, "Weight": float(w)} for k, w in sorted(r["tags"].items(), key=lambda kv: -kv[1])]
            st.session_state[ns + "tags_base"] = rows
            st.data_editor(pd.DataFrame(rows, columns=["Tag", "Weight"]), key=ns + "tags_df", num_rows="dynamic", hide_index=True, width="stretch",
                           column_config={"Tag": st.column_config.SelectboxColumn("Wants (moment tag)", options=tag_opts,
                                                                                  required=False),
                                          "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0,
                                                                                  step=0.1, format="%.1f")})
            st.multiselect("Avoid", tag_opts, default=[x for x in r["avoid"] if x in tag_opts], key=ns + "avoid")
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
