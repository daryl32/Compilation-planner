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
    return _to_editor(ST.snap(ST.suggest_fine(track, analysis, STM.saved_map(tid)), cuts)), "suggested"


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
    c = st.columns([3, 1.4, 1.2, 0.7, 0.7])
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
    with c[4].popover("📥", width="stretch", help="Download this track's sections"):
        import json as _json
        st.caption("The sections as shown, the suggester's own answer and what the app measured in the song (energy "
                   "per stretch and beat by beat) — attach it to a Claude chat to tune the suggestions.")
        try:
            data = _json.dumps(ST.tuning_export(track, _from_editor(regions), analysis,
                                                "you" if (s.get("dirty") or s.get("source") == "saved") else "suggested"))
            st.download_button("📥 Sections + measurements (.json)", data=data, file_name=f"sections_{tid}.json",
                               mime="application/json", key="sec_dl", width="stretch")
        except Exception as e:          # noqa: BLE001
            st.caption(f"Couldn't build the file: {e}")
    state_line = ("**unsaved changes** — 💾 Save to keep them" if s.get("dirty") else
                  "suggested, not saved yet — 💾 Save to keep it" if s.get("source") == "suggested" else "saved ✅")
    st.caption(f"{tid} · {state_line}")
    _render_lines(track, play)
    import story_map as STM
    import story_map_editor as SME
    cuts = [sg["start"] for sg in segs] + [float(track["duration_sec"])]

    def suggest():
        return _to_editor(ST.snap(ST.suggest_fine(track, analysis, STM.saved_map(tid)), cuts)) if analysis else regions

    play_part = (lambda r, lead, key: play(track, r, lead, key)) if play else None     # the editor calls play(r, lead, key)
    SME.render_editor(track, segs, ns="sec", get_regions=lambda: _state().get("regions") or [], set_regions=_set,
                      version=s.get("v", 0), analysis=analysis, play=play_part, roles=ST.types_as_roles(), suggest=suggest,
                      allow_none=False, thing="type", clear=False)


# ---------------------------------------------------------------------------
# 📐 Lines from the music — sections from scratch, step 1 (section_bounds.py): no names yet
# ---------------------------------------------------------------------------

RULE_ADDS, RULE_TAKES = "#2ecc71", "#ff9f1a"      # 🟢 a cut the 🎤 vocal rule adds · 🟠 one it takes away


def lines_figure(found: dict, height: int = 270, rule_cuts: dict = None):
    """What is playing bar by bar (one row per layer, darker = more of it) with the found lines across it: main
    lines solid, lesser ones dotted, the block cuts thin and white; the sections numbered along the top (parts
    lettered), the bars numbered along the bottom. rule_cuts (_rule_cuts): where the 🎤 vocal rule adds a cut
    (green, ＋) or takes one away (orange dashes, ✕), drawn whichever way the switch is set."""
    import numpy as np
    import plotly.graph_objects as go
    import section_bounds as SB
    bars, dur = np.asarray(found["bars"], float), float(found["duration"])
    ends = np.append(bars[1:], dur)
    names = [k for k in SB.LAYERS if k in found["levels"]][::-1]
    fig = go.Figure(go.Heatmap(
        x=(bars + ends) / 2, y=[f"{SB.EMOJI[k]} {SB.NAME[k]}" for k in names],
        z=[found["levels"][k] for k in names], zmin=0, zmax=1, colorscale="Blues", showscale=True, xgap=0, ygap=2,
        colorbar=dict(title=dict(text="level", side="right"), thickness=12, len=0.95, tickvals=[0, 0.15, 0.3, 0.5, 1],
                      ticktext=["0 silent", "0.15 off", "0.3 playing", "0.5", "1 strongest"], tickfont=dict(size=10)),
        hovertemplate="%{y} · %{x:.1f}s · %{z:.2f}<extra></extra>"))
    line_at = {round(ln["time"], 2) for ln in found["lines"]}
    for b in (found.get("blocks") or [])[1:]:              # the block cuts between the lines: thin, white
        if round(b["start"], 2) not in line_at:
            for colour, width in (("rgba(20,20,30,0.9)", 3.2), ("rgba(255,255,255,0.98)", 1.2)):   # white on a dark
                fig.add_shape(type="line", x0=b["start"], x1=b["start"], y0=0, y1=1, yref="paper",   # edge: shows on
                              line=dict(color=colour, width=width))                                  # any cell
    for kind, colour, dash, mark in (("removed", RULE_TAKES, "dash", "✕"), ("added", RULE_ADDS, "solid", "＋")):
        for t in (rule_cuts or {}).get(kind) or []:
            fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper",
                          line=dict(color=colour, width=3, dash=dash))
            fig.add_annotation(x=t, y=0.0, yref="paper", yanchor="bottom", showarrow=False, text=f"<b>{mark}</b>",
                               font=dict(size=12, color="#111"), bgcolor=colour, borderpad=1)
    for ln in found["lines"]:
        fig.add_vline(x=ln["time"], line_color="#ff4b4b", line_width=2.5 if ln["level"] == 1 else 1.2,
                      line_dash="solid" if ln["level"] == 1 else "dot")
    for sc in found["sections"]:
        fig.add_annotation(x=(sc["start"] + sc["end"]) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"<b>{sc.get('label', sc['n'])}</b>", font=dict(size=11))
    bars_ = found["bars"]
    step = 4 if len(bars_) <= 80 else 8                    # the bars stay on the axis: a tick each, numbered
    fig.update_xaxes(tickmode="array", tickvals=bars_[::step], ticktext=[str(k + 1) for k in range(0, len(bars_), step)],
                     ticks="outside", ticklen=4)
    fig.add_annotation(x=0, y=0, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                       text="bar ", font=dict(size=10))
    fig.update_layout(height=height, margin=dict(t=22, b=24, l=6, r=6), xaxis=dict(range=[0, dur]),
                      yaxis=dict(automargin=True, ticks="", tickfont=dict(size=12)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _both_cuts(track: dict, vocal_rule: bool = True) -> dict:
    """The track cut both ways: {"new", "old"}. vocal_rule=False switches the sung-phrase rule off in both. The old
    module is the v1.59.0 reference and is not edited: it skips the rule when a track has no sung-phrase marks,
    so it is given a copy of the track without them (it reads them for nothing else)."""
    import section_bounds as SB
    import section_bounds_old as SBO
    old_track = track if vocal_rule else {
        **track, "analysis": {**(track.get("analysis") or {}), "vocal_starts": [], "vocal_ends": []}}
    return {"new": SB.for_track(track, vocal_rule=vocal_rule), "old": SBO.for_track(old_track)}


def _cuts_changed(a: dict, b: dict) -> list:
    """The sections (part labels) whose block cuts differ between two results of the same mode, in track order."""
    def cuts(f):
        out = {}
        for blk in f.get("blocks") or []:
            out.setdefault(blk["section"], []).append((blk["start"], blk["end"]))
        return out
    ca, cb = cuts(a), cuts(b)
    return [k for k in ca if ca[k] != cb.get(k)] + [k for k in cb if k not in ca]


def _rule_cuts(on: dict, off: dict) -> dict:
    """What the 🎤 vocal rule does to the cuts of one mode: {"added": times that are a cut only with the rule on,
    "removed": times that are a cut only with it off}. A cut the rule moves is one of each."""
    def cuts(f):
        return {round(float(b["start"]), 2) for b in (f.get("blocks") or [])[1:]}
    a, b = cuts(on), cuts(off)
    return {"added": sorted(a - b), "removed": sorted(b - a)}


def _changed_words(labels: list) -> str:
    if not labels:
        return "nothing"
    return f"section{'s' if len(labels) != 1 else ''} {', '.join(labels[:10])}{'…' if len(labels) > 10 else ''}"


def _render_lines(track: dict, play=None) -> None:
    import section_bounds as SB
    with st.expander("📐 New — lines from the music (no names yet)"):
        st.caption("Sections from scratch, steps 1–2: a line wherever the music changes — a layer comes in or drops "
                   "out, it gets louder or quieter, the sound changes — always on a bar line. No types yet: this "
                   "only shows WHERE the song divides. Solid = a big change, dotted = a smaller one. It doesn't "
                   "touch the saved sections below. ▶ at the bottom plays the whole track over it.")
        if st.session_state.get("sec_ln_mode") not in ("new", "old"):
            st.session_state["sec_ln_mode"] = "new"
        mode = st.segmented_control(
            "Cuts", ["old", "new"], key="sec_ln_mode",
            format_func={"old": "🅰 Old cuts (as on main, v1.59.0)", "new": "🅱 New cuts"}.get,
            help="The same track cut two ways, to compare by ear. 🅰 Old: exactly as released in v1.59.0 — the "
                 "stored bar lines, blocks counted in bars, drum parts paced by how busy the drums rank. 🅱 New: a "
                 "steady bar grid from the beats, no block under the shortest clip for the tempo (short ones "
                 "pair up), loud drum parts all cut at one bar, a pattern that alternates bar by bar no longer "
                 "read as one-bar breaks. Section, bar and block numbers differ between the two.") or "new"
        vocal = st.toggle(
            "🎤 Vocal rule", value=True, key="sec_ln_vocal",
            help="On (as before): in a part the voice leads (voice playing, no drums), a cut that would land "
                 "inside a sung phrase moves to a clean bar or half-bar line, or is dropped — one longer block. "
                 "Off: the cuts stay where the pace puts them, sung phrase or not. Works on both 🅰 Old and 🅱 New; "
                 "the section lines don't change, only the blocks.")
        try:
            both, other = _both_cuts(track, vocal), _both_cuts(track, not vocal)
            found = both[mode]
        except Exception as e:          # noqa: BLE001
            st.caption(f"Couldn't read this track's layers: {e}")
            return

        def _sum(f):
            d = [b["end"] - b["start"] for b in f.get("blocks") or []]
            return (f"{len(f['bars'])} bars · {f['sections'][-1]['main']} sections · {len(d)} blocks, "
                    f"{min(d):.1f}–{max(d):.1f} s") if d else "no blocks"
        st.caption(f"{'**🅰 Old**' if mode == 'old' else '🅰 Old'}: {_sum(both['old'])}  \n"
                   f"{'**🅱 New**' if mode == 'new' else '🅱 New'}: {_sum(both['new'])}")
        moved = {m: _cuts_changed(both[m], other[m]) for m in ("old", "new")}
        rule = _rule_cuts(*((both[mode], other[mode]) if vocal else (other[mode], both[mode])))
        if not any(moved.values()):
            st.caption("🎤 The vocal rule changes no cut in this track, in either mode (it only acts in parts "
                       "where the voice plays without drums).")
        else:
            st.caption(f"🎤 Vocal rule **{'on' if vocal else 'off'}** — switching it "
                       f"{'off' if vocal else 'on'} changes: 🅰 Old {_changed_words(moved['old'])} · "
                       f"🅱 New {_changed_words(moved['new'])}")
            if rule["added"] or rule["removed"]:
                st.caption(f"On the chart, in this mode: 🟢 ＋ {len(rule['added'])} cut"
                           f"{'s' if len(rule['added']) != 1 else ''} the rule adds · 🟠 ✕ {len(rule['removed'])} "
                           f"it takes away (a moved cut is one of each). They stay marked whichever way the "
                           f"switch is set — {'🟠 are not cuts right now' if vocal else '🟢 are not cuts right now'}.")
        if not found["lines"]:
            st.info("No lines found — this track may need the extra 🔬 analysis.")
            return
        st.plotly_chart(lines_figure(found, rule_cuts=rule), width="stretch", key="sec_ln_chart", config={"displayModeBar": False})
        if found.get("uneven_bars"):
            ub = found["uneven_bars"]
            st.caption(f"⚠️ This track's bar lines are uneven at bar{'s' if len(ub) != 1 else ''} "
                       f"{', '.join(map(str, ub[:12]))}{'…' if len(ub) > 12 else ''} (much shorter or longer than "
                       "its usual bar) — blocks there are sized in seconds, so none comes out too short.")
        if found.get("stems"):
            st.caption("🎚️ Drums, bass and instruments are read from this track's real stems.")
        else:
            st.caption("🎚️ Drums and bass are read from the full mix by pitch range here — run_audio_analysis.bat on "
                       "the laptop measures the real stems (and adds 🎹 instruments).")
        missing = [SB.NAME[k] for k in SB.LAYERS if k not in found["levels"] and k != "other"]
        main = sum(1 for ln in found["lines"] if ln["level"] == 1)
        blocks = found.get("blocks") or []
        st.caption(f"{found['sections'][-1]['main']} sections in {len(found['sections'])} parts · {main} big changes "
                   f"(solid: a new section), {len(found['lines']) - main} smaller (dotted: a new part, 2a → 2b) · "
                   f"**{len(blocks)} blocks** (thin white lines: the cuts inside a part)"
                   + (f" · not in this track's analysis: {', '.join(missing)}" if missing else ""))
        def cut_into(label):
            mine = [b for b in blocks if b["section"] == label]
            return (" + ".join(f"{b['bars']:g}" for b in mine) + " bars — " + mine[0]["why"]) if mine else ""
        st.dataframe([{"#": sc["label"], "From": format_mmss(sc["start"]), "To": format_mmss(sc["end"]),
                       "Bars": sc["bars"], "Change": "big" if sc["level"] == 1 else "small",
                       "What changes at its start": " · ".join(sc["why"]),
                       "What it is (profile)": sc.get("words", ""),
                       "Cut into blocks of": cut_into(sc["label"])} for sc in found["sections"]],
                     hide_index=True, width="stretch", key="sec_ln_table")
        st.caption("Profile = step 2: the layers playing · how loud within this track (quiet / medium / loud) · "
                   "how busy the drums are (hits a second) · rising or falling · “like N” = the same profile as "
                   "section N. A section type will be a named profile. Blocks: every line is a cut; inside a part "
                   "the profile sets the length — loud drum parts 1 bar, other drum parts 2, voice / instruments "
                   "leading 2 or 4, hardly anything playing 4 — shorter towards the end of a rising part, longer "
                   "towards the end of a falling one, never inside a sung phrase where the voice leads, never "
                   "under the shortest clip for the tempo (a bar at 90 BPM) — shorter ones are paired up.")
        st.caption("Key: each cell is one bar of one layer, 0–1 against that layer's own strongest moments in this "
                   "track (drums and hats: how hard the hits are; the rest: the average level). From 0.3 a layer "
                   "counts as playing, under 0.15 as off. Tap a cell for its number.")
        with st.expander("🔢 The numbers per section"):
            st.dataframe([{"#": sc["label"], "From": format_mmss(sc["start"]), "Bars": sc["bars"],
                           **{f"{SB.EMOJI[k]} {SB.NAME[k]}": sc["playing"].get(k) for k in SB.LAYERS
                              if k in found["levels"]},
                           "Level in track": sc["profile"]["level"], "Drum hits /s": sc["profile"]["busy"],
                           "Loudness change": sc["profile"]["change"]} for sc in found["sections"]],
                         hide_index=True, width="stretch", key="sec_ln_nums")
        if st.toggle("▶ Play the whole track with the lines", key="sec_ln_whole",
                     help="The full song with a cursor moving over the layers and the lines; under it the time "
                          "code, the bar number and the section you are in. Tap the picture to jump."):
            clip, err = _whole_track_clip(track)
            if clip is None:
                st.warning(err)
                if play is not None:               # the Media Library's player offers the download from Drive
                    play(track, {"start": 0.0, "end": min(5.0, float(track["duration_sec"]))}, False,
                         f"sec_ln_dl_{track.get('track_id')}")
            else:
                lines_player(clip, found,
                             f"ln_{mode}_{'v' if vocal else 'n'}_{abs(hash(track.get('track_id'))) % 10 ** 8}",
                             rule_cuts=rule)


def _whole_track_clip(track: dict) -> tuple:
    """(path of a small copy of the whole song, None) or (None, why not) — cut once, then cached."""
    try:
        from pathlib import Path
        from config import PREVIEW_DIR
        from render_preview import colab_to_local
        import preview_snippets
        path = Path(colab_to_local(track.get("source_path", "")))
        if not path.exists():
            return None, "This track's audio isn't on the server yet — download it below, then switch ▶ on again."
        clip, err = preview_snippets.make_audio_snippet(Path(PREVIEW_DIR) / "snippets", path, 0.0,
                                                        float(track["duration_sec"]))
        return (clip, None) if clip else (None, f"Couldn't prepare the music: {err}")
    except Exception as e:          # noqa: BLE001
        return None, f"Couldn't prepare the music: {e}"


def player_config(found: dict, rule_cuts: dict = None) -> dict:
    """What the whole-track player draws: the layers bar by bar, the lines, the sections and the bar starts —
    and where the 🎤 vocal rule adds or takes away a cut (rule_cuts, see lines_figure)."""
    import section_bounds as SB
    names = [k for k in SB.LAYERS if k in found["levels"]]
    return {"dur": float(found["duration"]), "bars": [round(float(b), 3) for b in found["bars"]],
            "layers": [f"{SB.EMOJI[k]} {SB.NAME[k]}" for k in names],
            "z": [[round(float(v), 2) for v in found["levels"][k]] for k in names],
            "lines": [{"t": ln["time"], "big": ln["level"] == 1} for ln in found["lines"]],
            "sections": [{"n": sc["n"], "label": sc.get("label", str(sc["n"])), "main": sc.get("main", sc["n"]),
                          "a": sc["start"], "b": sc["end"],
                          "why": " · ".join(sc["why"]), "words": sc.get("words", "")} for sc in found["sections"]],
            "blocks": [{"n": b["n"], "a": b["start"], "b": b["end"], "bars": b["bars"], "why": b["why"]}
                       for b in found.get("blocks") or []],
            "rule": {"added": list((rule_cuts or {}).get("added") or []),
                     "removed": list((rule_cuts or {}).get("removed") or [])}}


def lines_player(clip_path, found: dict, key: str, rule_cuts: dict = None) -> None:
    """The whole song with a cursor synced to it over the layers and the lines; under it the time code, the bar
    number and the section being played, with what changed at its start and its profile. Tap to jump."""
    import base64
    import json as _json
    from pathlib import Path
    from streamlit.components.v1 import html as _html
    try:
        data = base64.b64encode(Path(clip_path).read_bytes()).decode("ascii")
    except OSError:
        st.audio(str(clip_path))
        return
    cfg = player_config(found, rule_cuts)
    rule_key = "" if not (cfg["rule"]["added"] or cfg["rule"]["removed"]) else (
        f'<span style="border-left:3px solid {RULE_ADDS};padding-left:4px">+ vocal rule adds</span> '
        f'<span style="border-left:3px dashed {RULE_TAKES};padding-left:4px">&#10005; vocal rule takes away</span>')
    rows = len(cfg["layers"])
    ch = 30 + rows * 20 + 26                      # section numbers · the layers · bar numbers
    _html(f"""
<div style="font-family:sans-serif;color:#ddd">
 <canvas id="c_{key}" width="1400" height="{ch}" style="width:100%;height:{ch}px;display:block;touch-action:manipulation"></canvas>
 <div style="display:flex;align-items:center;gap:8px;margin:4px 0 2px;font-size:11px;color:#aaa">
  <span>key</span><span>0</span>
  <span style="display:inline-block;width:150px;height:10px;border-radius:2px;background:linear-gradient(90deg,rgb(247,251,255),rgb(107,174,214),rgb(8,48,107))"></span>
  <span>1 = strongest in this track</span>
  <span style="margin-left:10px;border-left:2.5px solid #ff4b4b;padding-left:4px">big change</span>
  <span style="border-left:1.5px dotted #ff4b4b;padding-left:4px">smaller change</span>
  <span style="border-left:1.5px solid #fff;padding-left:4px">block cut</span>
  {rule_key}
 </div>
 <audio id="a_{key}" controls style="width:100%;height:34px" src="data:audio/mp4;base64,{data}"></audio>
 <div id="r_{key}" style="margin-top:6px;font-size:15px;font-variant-numeric:tabular-nums"></div>
 <div id="w_{key}" style="margin-top:2px;font-size:12.5px;color:#aaa;min-height:2.4em"></div>
</div>
<script>
(function() {{
 const cfg = {_json.dumps(cfg)};
 const cv = document.getElementById("c_{key}"), ctx = cv.getContext("2d");
 const au = document.getElementById("a_{key}"), rd = document.getElementById("r_{key}"), wd = document.getElementById("w_{key}");
 const W = cv.width, H = cv.height, G = 118, TOP = 30, RH = 20, rows = cfg.layers.length, BOT = TOP + rows * RH;
 const x = t => G + t / cfg.dur * (W - G);
 const mmss = t => {{ const m = Math.floor(t / 60), s = t - m * 60; return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1); }};
 const ends = cfg.bars.slice(1).concat([cfg.dur]);
 const col = v => {{ v = Math.max(0, Math.min(1, v));           // white to blue, as the chart above
   const a = [247, 251, 255], m = [107, 174, 214], b = [8, 48, 107];
   const p = v < 0.5 ? [a, m, v * 2] : [m, b, (v - 0.5) * 2];
   return "rgb(" + [0, 1, 2].map(i => Math.round(p[0][i] + (p[1][i] - p[0][i]) * p[2])).join(",") + ")"; }};
 const barAt = t => {{ let k = 0; for (let i = 0; i < cfg.bars.length; i++) {{ if (cfg.bars[i] <= t + 1e-6) k = i; else break; }} return k; }};
 const secAt = t => cfg.sections.find(s => t >= s.a - 1e-6 && t < s.b) || cfg.sections[cfg.sections.length - 1];
 function draw(now) {{
  ctx.clearRect(0, 0, W, H);
  const cur = secAt(now);
  // the layers, bar by bar
  for (let r = 0; r < rows; r++) {{
   for (let k = 0; k < cfg.bars.length; k++) {{
    ctx.fillStyle = col(cfg.z[r][k]);
    ctx.fillRect(x(cfg.bars[k]), TOP + r * RH + 1, Math.max(1, x(ends[k]) - x(cfg.bars[k]) + 0.6), RH - 2);
   }}
   ctx.fillStyle = "#ddd"; ctx.font = "13px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
   ctx.fillText(cfg.layers[r], 4, TOP + r * RH + RH / 2);
  }}
  // the block cuts between the lines: thin and white (the bars themselves are on the axis below)
  const lineAt = new Set(cfg.lines.map(l => l.t.toFixed(2)));
  cfg.blocks.slice(1).forEach(b => {{ if (lineAt.has(b.a.toFixed(2))) return;
   for (const [c, w] of [["rgba(20,20,30,0.9)", 4], ["rgba(255,255,255,0.98)", 1.6]]) {{     // white on a dark edge
    ctx.strokeStyle = c; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x(b.a), TOP); ctx.lineTo(x(b.a), BOT); ctx.stroke(); }} }});
  // the vocal rule: orange dashes + x where it takes a cut away, green + plus where it adds one
  for (const [ts, c, dash, mark] of [[cfg.rule.removed, "{RULE_TAKES}", [6, 4], "\u2715"], [cfg.rule.added, "{RULE_ADDS}", [], "+"]]) {{
   for (const t of ts) {{
    ctx.strokeStyle = c; ctx.lineWidth = 3; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x(t), TOP); ctx.lineTo(x(t), BOT); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = c; ctx.fillRect(x(t) - 7, BOT - 14, 14, 14);
    ctx.fillStyle = "#111"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(mark, x(t), BOT - 7); }} }}
  const blk = cfg.blocks.find(b => now >= b.a - 1e-6 && now < b.b) || cfg.blocks[cfg.blocks.length - 1];
  if (blk) {{ ctx.fillStyle = "rgba(255,214,10,0.22)"; ctx.fillRect(x(blk.a), TOP, x(blk.b) - x(blk.a), rows * RH); }}
  // the section being played, lit
  if (cur) {{ ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 2; ctx.strokeRect(x(cur.a) + 1, TOP - 1, x(cur.b) - x(cur.a) - 2, rows * RH + 2); }}
  // the lines and the section numbers
  for (const ln of cfg.lines) {{
   ctx.strokeStyle = "#ff4b4b"; ctx.lineWidth = ln.big ? 3 : 1.5; ctx.setLineDash(ln.big ? [] : [4, 4]);
   ctx.beginPath(); ctx.moveTo(x(ln.t), 6); ctx.lineTo(x(ln.t), BOT + 4); ctx.stroke();
  }}
  ctx.setLineDash([]); ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  for (const s of cfg.sections) {{
   const on = cur && s.n === cur.n;
   ctx.font = (on ? "bold 17px" : "bold 14px") + " sans-serif"; ctx.fillStyle = on ? "#ffffff" : "#bdbdbd";
   if (x(s.b) - x(s.a) > 14) ctx.fillText(s.label, (x(s.a) + x(s.b)) / 2, 22);
  }}
  // bar numbers: every bar a tick, a number every 4 (or 8 when crowded)
  const step = (W - G) / cfg.bars.length < 9 ? 8 : 4;
  ctx.font = "11px sans-serif"; ctx.fillStyle = "#a8a8a8"; ctx.strokeStyle = "rgba(160,160,160,0.6)"; ctx.lineWidth = 1;
  cfg.bars.forEach((b, k) => {{
   const major = k % step === 0;
   ctx.beginPath(); ctx.moveTo(x(b), BOT + 2); ctx.lineTo(x(b), BOT + (major ? 9 : 5)); ctx.stroke();
   if (major) ctx.fillText(String(k + 1), x(b), BOT + 21);
  }});
  ctx.textAlign = "left"; ctx.fillText("bar", 4, BOT + 21); ctx.fillText("section", 4, 22);
  // the cursor
  ctx.strokeStyle = "#ffd60a"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x(now), 0); ctx.lineTo(x(now), H); ctx.stroke();
  const k = barAt(now), beat = Math.min(4, 1 + Math.floor((now - cfg.bars[k]) / Math.max(ends[k] - cfg.bars[k], 0.001) * 4));
  rd.innerHTML = "<b>" + mmss(now) + "</b> &nbsp;&middot;&nbsp; bar <b>" + (k + 1) + "</b>." + beat + " of " + cfg.bars.length
    + " &nbsp;&middot;&nbsp; section <b>" + (cur ? cur.label : "-") + "</b>"
    + (cur ? " &nbsp;(" + mmss(cur.a) + " &ndash; " + mmss(cur.b) + ")" : "")
    + (blk ? " &nbsp;&middot;&nbsp; block <b>" + blk.n + "</b> of " + cfg.blocks.length + " &nbsp;(" + blk.bars + (blk.bars == 1 ? " bar, " : " bars, ") + (blk.b - blk.a).toFixed(1) + " s)" : "");
  wd.textContent = cur ? ("starts with: " + cur.why + "   |   it is: " + cur.words + (blk ? "   |   blocks: " + blk.why : "")) : "";
 }}
 let raf = null;
 function tick() {{ draw(au.currentTime); raf = requestAnimationFrame(tick); }}
 au.addEventListener("play", () => {{ if (!raf) tick(); }});
 au.addEventListener("pause", () => {{ cancelAnimationFrame(raf); raf = null; draw(au.currentTime); }});
 au.addEventListener("seeked", () => {{ if (!raf) draw(au.currentTime); }});
 au.addEventListener("ended", () => {{ cancelAnimationFrame(raf); raf = null; draw(cfg.dur); }});
 cv.addEventListener("click", ev => {{ const rect = cv.getBoundingClientRect();
   const t = ((ev.clientX - rect.left) / rect.width * W - G) / (W - G) * cfg.dur;
   au.currentTime = Math.max(0, Math.min(cfg.dur - 0.05, t)); draw(au.currentTime); }});
 draw(0);
}})();
</script>""", height=ch + 34 + 28 + 78)


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


def _upload_recipes(raw: bytes) -> None:
    import json as _json
    try:
        data = _json.loads(raw.decode("utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("not a recipes file")
    except (ValueError, UnicodeDecodeError) as e:
        st.session_state["rc_msg"] = f"That file couldn't be read ({e}) — nothing was changed."
        return
    done = [t for t in ST.TYPE_IDS if isinstance(data.get(t), dict)]
    for t in done:
        RC.save(t, data[t])
    st.session_state["rc_msg"] = (f"Recipes replaced from the file: {', '.join(ST.label(t) for t in done)}." if done
                                  else "No section type found in that file — nothing was changed.")
    st.session_state["rc_v"] = st.session_state.get("rc_v", 0) + 1


def render_recipes() -> None:
    st.caption("🍳 What footage each section type wants: a movement band (library scale, as the 🧭 timeline shows it) "
               "and whether it builds or falls, how tightly movement should lock to the beat, the moment tags to "
               "look for (with weights) and to avoid, scene tags to require or prefer, an optional description "
               "matched against the pictures, whether the shot size may change, and the clip length the 🤖 new "
               "matcher cuts at. One shared set for now — variations per cutting style can come later.")
    if msg := st.session_state.pop("rc_msg", None):
        st.success(msg)
    import json as _json
    io = st.columns([1.6, 1.6, 4])
    io[0].download_button("📥 Download recipes", data=_json.dumps(RC.load_all(), indent=1, ensure_ascii=False),
                          file_name="section_recipes.json", mime="application/json", key="rc_dl", width="stretch",
                          help="Every type's recipe as it stands — to keep a copy, or to attach to a Claude chat.")
    with io[1].popover("📤 Upload recipes", width="stretch"):
        st.caption("A file from 📥 Download recipes (edited or not). Each type in it replaces that type's recipe here; "
                   "types it doesn't mention are left alone.")
        up = st.file_uploader("Recipes file", type=["json"], key=f"rc_up_{st.session_state.get('rc_v', 0)}",
                              label_visibility="collapsed")
        if up is not None:
            st.button("Use this file", key="rc_up_go", type="primary", width="stretch", on_click=_upload_recipes,
                      args=(up.getvalue(),))
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
