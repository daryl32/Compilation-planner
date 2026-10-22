"""🎵 Step 0 — the Audio Settings step (Advanced and Simple) with its live chart.

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import cut_scoring as CS
from audio_charts import add_analysis_overlays, add_vrects
from core.segmentation import (SEG_ADAPTIVE, SEG_RHYTHM, SEG_SCORED, SEG_SMART, build_segments,
                               compute_density_curve, apply_segment_intensity)
from core.autotune import AUDIO_STYLES, AUDIO_STYLE_CUSTOM
import settings_store as SSTORE

from planner.state import (
    fragment
)
from planner.projects import (
    autosave_now
)
from planner.audio import (
    add_cut_markers, _beats_per_bar_changed, clips_for_intensity, _mark_audio_custom, _method_changed,
    _pace_row, planned_segments, _render_aim_adaptive, _render_cut_editor, _render_scored_cut_settings,
    _render_smart_cut_settings, _render_style_row, SEG_LABELS, segmentation_settings, _setting, _smart_reset
)
from planner.story import (
    current_story_map
)


def block_chart(track: dict, segments: list, marks: dict = None, height: int = 220):
    """The track's energy with every cut, each block numbered along the top — for choosing blocks by
    number. marks: {block index: (label, colour)} shades those blocks."""
    fig = go.Figure()
    if track.get("hires"):
        _rate = float(track["hires"]["rate"])
        fig.add_trace(go.Scatter(x=np.arange(len(track["hires"]["rms"])) / _rate, y=np.asarray(track["hires"]["rms"], dtype=float),
                                 mode="lines", line=dict(color="rgba(100,150,255,0.55)", width=1), hoverinfo="skip"))
    else:
        fig.add_trace(go.Scatter(x=[e["time"] for e in track["energy_envelope"]],
                                 y=[e["energy"] for e in track["energy_envelope"]], mode="lines",
                                 line=dict(color="rgba(100,150,255,0.55)"), hoverinfo="skip"))
    add_cut_markers(fig, segments)
    step = max(1, len(segments) // 40)          # label every block on short plans, every few on long ones
    _rects, _nums = [], []
    for i, sg in enumerate(segments):
        mark = (marks or {}).get(i)
        if mark:
            _rects.append({"x0": sg["start"], "x1": sg["end"], "fillcolor": mark[1], "opacity": 0.25})
        if i % step == 0 or mark or i == len(segments) - 1:
            _nums.append(dict(x=(sg["start"] + sg["end"]) / 2, y=1.0, yref="paper", text=str(i + 1),
                              showarrow=False, font=dict(size=10), yanchor="bottom"))
    add_vrects(fig, _rects)                     # ⚡ shapes and labels in one go (see audio_charts.add_vrects)
    fig.update_layout(annotations=[*fig.layout.annotations, *_nums])
    fig.add_trace(go.Scatter(x=[(sg["start"] + sg["end"]) / 2 for sg in segments], y=[0.02] * len(segments),
                             mode="markers", marker=dict(size=14, opacity=0), showlegend=False,
                             customdata=[[i + 1, sg["start"], sg["end"]] for i, sg in enumerate(segments)],
                             hovertemplate="Block %{customdata[0]}: %{customdata[1]:.1f}s → %{customdata[2]:.1f}s"
                                           "<extra></extra>"))
    fig.update_layout(height=height, margin=dict(t=24, b=10, l=10, r=10), showlegend=False,
                      yaxis=dict(showticklabels=False, title=None), xaxis_title=None)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@fragment
def render_audio_settings_section(track: dict) -> None:
    """Step 0: all segmentation dials plus a live audio chart — so you can
    tune the cuts while watching exactly where they land on the waveform.
    Runs as a fragment so adjusting any dial only reruns this section, not
    the full page. Every dial keeps its value while other steps show — see
    settings_store.py (widgets use key= only)."""
    autosave_now()
    _v = _setting

    st.subheader("How the music is cut")
    _render_style_row(_v)
    _seg_options = [SEG_SMART, SEG_SCORED, SEG_ADAPTIVE, SEG_RHYTHM]
    SSTORE.choose("segmentation_method", _seg_options)
    segmentation_method = st.radio(
        "Method", _seg_options, key="segmentation_method", format_func=lambda m: SEG_LABELS.get(m, m), horizontal=True,
        on_change=_method_changed,
        help="🧠 Smart: reads the song's structure and intensity and sets its own cutting rhythm — a "
             "steady pace per section on the bar / beat grid, faster when the music is bigger, speeding up "
             "into drops, with cuts at every section change. 🎯 Best musical moments: scores every beat for how good a cut it would be (bar lines, "
             "phrases, section changes, drops, drums, chords, vocals) and picks the best set together. "
             "⚡ Energy changes & big hits: cuts where the loudness shifts and on the strongest hits, "
             "wherever they fall. 🎬 Bar styles: each stretch of bars is either CINEMATIC (a cut every "
             "few bars) or FAST (a cut every bar), from how loud it is.",
    )
    _pc = st.columns([2, 2, 1])
    _styles = list(AUDIO_STYLES) + [AUDIO_STYLE_CUSTOM]
    if segmentation_method == SEG_SMART:
        # Smart sets its own pace from the track; the Pace slider below dials it in.
        _pc[0].caption("🧠 Smart sets its pace from the track itself — dial it in with **Pace** below.")
        _pc[1].button("↺ Back to auto", on_click=_smart_reset, width="stretch",
                      help="Pace back to auto, contrast back to the song's own, all switches on.")
    else:
        _pace_row(_pc, _styles)
    _pc[2].number_input(
        "Beats per bar", min_value=2, max_value=12, key="beats_per_bar",
        on_change=_beats_per_bar_changed,
        help="4 = common time (most pop/rock/EDM). Use 3 for a waltz, 6 for 6/8, etc.",
    )
    if st.session_state.get("auto_tune_report"):
        with st.expander("What auto-detect chose, and why", expanded=False):
            for line in st.session_state["auto_tune_report"]:
                st.caption(line)

    _custom = dict(on_change=_mark_audio_custom)
    if segmentation_method in (SEG_SCORED, SEG_ADAPTIVE, SEG_RHYTHM):
        st.toggle("🥁 Land every cut on the beat", key="on_beat", **_custom,
                  help="Moves each cut onto the nearest beat — onto a bar line when one is within a beat, for "
                       "changes in the music (energy changes, sections, drops, bar styles). Off = cuts can fall "
                       "between beats, exactly on a hit or a loudness jump.")
    if segmentation_method == SEG_SMART:
        _render_smart_cut_settings(track, _v)
    elif segmentation_method == SEG_SCORED:
        _render_scored_cut_settings(track, _v)
    elif segmentation_method == SEG_ADAPTIVE:
        _render_aim_adaptive(track, len(build_segments(track, segmentation_settings())))
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Cut at energy changes — threshold", 0.02, 0.5, step=0.01, key="change_threshold", **_custom,
                      help="How big a jump in loudness counts as a change worth cutting on. Lower = more cuts.")
            react = st.toggle("Cut on big hits", key="react_to_hits", **_custom,
                              help="Also cut on the strongest hits — exactly on the hit, or on its beat with 🥁 on.")
            st.slider("Big-hit threshold", 0.5, 1.0, step=0.01,
                      key="hit_threshold", disabled=not react, **_custom,
                      help="How strong a hit must be (0–1) to become a cut. Lower = more cuts.")
        with col2:
            st.slider("Shortest segment (s)", 0.3, 8.0, step=0.1,
                      key="min_segment_sec", **_custom, help="No two cuts closer than this.")
            st.slider("Longest segment (s)", 0.0, 30.0, step=0.5,
                      key="max_segment_sec", **_custom,
                      help="Longer segments are split at their strongest beat. 0 = no limit.")
        with st.expander("Advanced", expanded=False):
            ac = st.columns(2)
            ac[0].slider("Energy-change window (s)", 0.5, 6.0, step=0.1,
                         key="change_window_secs", **_custom,
                         help="Compares the loudness this long after each moment with this long before it.")
            ac[1].slider("Snap radius (s)", 0.0, 1.0, step=0.05, key="snap_secs",
                         **_custom, help="Energy-change cuts move onto the strongest hit within this distance "
                                        "(then onto the beat, with 🥁 on).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Fast-cutting threshold", 0.1, 1.0, step=0.01,
                      key="fast_energy_thresh", **_custom,
                      help="Bars louder than this (0–1 of the loudest) cut every bar (FAST); quieter bars "
                           "are CINEMATIC. Lower = more fast cutting.")
            st.number_input("Cinematic segment length (bars)", min_value=1, max_value=8,
                            key="cinematic_bars", **_custom,
                            help="How many bars a CINEMATIC segment spans. 2, 4 or 8 follow musical phrases.")
        with col2:
            dpo = st.toggle("Extra cuts at sudden jumps", key="dynamic_priority_override", **_custom,
                            help="Adds a cut at a sudden mid-bar jump in loudness (a hit or drop) that the "
                                 "bar-by-bar styles would smooth away.")
            st.slider("Sudden-jump sensitivity", 0.02, 1.0, step=0.01,
                      key="override_delta_thresh", disabled=not dpo, **_custom,
                      help="How large a jump triggers an extra cut. Lower = more extra cuts.")
        with st.expander("Advanced", expanded=False):
            ac = st.columns(2)
            with ac[0]:
                st.number_input("FAST max run (bars)", min_value=0, max_value=64,
                                key="fast_max_bars", **_custom,
                                help="After this many FAST bars in a row, FAST stops for a while. 0 = no limit.")
                st.number_input("FAST pause after a run (bars)", min_value=0, max_value=32,
                                key="fast_cooldown_bars", **_custom,
                                help="Bars that must pass after a FAST run before FAST can start again.")
            with ac[1]:
                st.number_input("Hold a style for (bars)", min_value=1, max_value=8,
                                key="phrase_lock_bars", **_custom,
                                help="FAST or CINEMATIC is kept this many bars before it's re-decided.")
                st.slider("Clear zone around a jump cut (s)", 0.0, 4.0, step=0.1, key="override_clear_secs", disabled=not dpo, **_custom,
                          help="Other cuts closer than this to a jump cut are removed.")

    st.divider()
    _render_cut_editor(track)

    # ---- Live preview chart ------------------------------------------------
    st.divider()
    st.subheader("Preview: cut points on the audio signal",
                 help="Updates as you move the dials. Scene match dots appear on the Matching & Export step "
                      "once clips are assigned.")

    # Build segments from current widget values for the preview
    _dc = st.session_state.get("density_contrast", 1.0)
    _min_c = st.session_state.get("min_clips", 1)
    _max_c = st.session_state.get("max_clips", 4) if st.session_state.get("split_screen_enabled") else 1
    _rs, _re = st.session_state.get("ramp_range", (0.2, 0.85))

    _segs = planned_segments(track, segmentation_settings())
    apply_segment_intensity(_segs, track, _dc)
    _rec_counts = [
        clips_for_intensity(sg["intensity"], _min_c, _max_c, _rs, _re, False, None)
        for sg in _segs
    ]

    afig = go.Figure()
    if track.get("hires"):
        _rate = float(track["hires"]["rate"])
        _t = np.arange(len(track["hires"]["rms"])) / _rate
        afig.add_trace(go.Scatter(x=_t, y=np.asarray(track["hires"]["rms"], dtype=float), mode="lines",
                                  name="Track energy (30 Hz)",
                                  line=dict(color="rgba(100,150,255,0.5)", width=1)))
        afig.add_trace(go.Scatter(x=_t, y=np.asarray(track["hires"]["onset"], dtype=float), mode="lines",
                                  name="Onset strength",
                                  line=dict(color="rgba(150,150,150,0.35)", width=1),
                                  visible="legendonly"))
    else:
        afig.add_trace(go.Scatter(x=[e["time"] for e in track["energy_envelope"]],
                                  y=[e["energy"] for e in track["energy_envelope"]],
                                  mode="lines", name="Track energy",
                                  line=dict(color="rgba(100,150,255,0.5)")))

    add_analysis_overlays(afig, track)
    add_cut_markers(afig, _segs)

    _dt, _dens = compute_density_curve(track)
    afig.add_trace(go.Scatter(x=_dt, y=_dens, mode="lines", name="Density (1.5 s smoothed)",
                              line=dict(color="rgba(140,90,220,0.8)", width=2)))
    _cx, _cy = [], []
    for _sg, _rc in zip(_segs, _rec_counts):
        _cx += [_sg["start"], _sg["end"]]
        _cy += [_rc, _rc]
    afig.add_trace(go.Scatter(x=_cx, y=_cy, mode="lines", name="Recommended clips", yaxis="y2",
                              line=dict(color="rgba(30,30,30,0.85)", width=2, shape="hv")))

    afig.update_layout(
        height=400, margin=dict(t=20, b=20),
        xaxis_title="Time (s)", yaxis_title="Normalised level",
        yaxis2=dict(title="Clips", overlaying="y", side="right",
                    range=[0, _max_c + 0.5], dtick=1, showgrid=False),
        legend=dict(orientation="h", y=-0.28),
    )
    st.plotly_chart(afig, width="stretch")
    st.caption(_audio_status(track, _segs)[0])


def _audio_status(track: dict, segs: list) -> tuple:
    """(status line, notes) for the 🎵 step: how the cuts land, plus anything set elsewhere that changes
    them (hand-edited cuts, split screen, own settings) — so Simple never hides what's in effect."""
    _bar = CS.bar_sec(track, int(st.session_state.get("beats_per_bar", 4)))
    _bars = np.asarray(CS.bar_lines(track, int(st.session_state.get("beats_per_bar", 4))), dtype=float)
    _cuts = [sg["start"] for sg in segs[1:]]
    _on_bar = (sum(1 for c in _cuts if len(_bars) and np.min(np.abs(_bars - c)) < 0.12) / len(_cuts)) if _cuts else 0.0
    avg = track["duration_sec"] / max(len(segs), 1)
    line = (f"**{len(segs)} segments** · a cut every {avg / _bar:.1f} bars ({avg:.1f}s) on average · "
            f"{_on_bar:.0%} on a bar line · {CS.beat_grid(track)[1]:.0f} BPM")
    notes = []
    e = st.session_state.get("cut_edits")
    if isinstance(e, dict) and e.get("track_id") == st.session_state.get("track_id") and e.get("cuts") is not None:
        notes.append("✂️ your hand-edited cuts")
    if st.session_state.get("split_screen_enabled"):
        notes.append(f"🔀 split screen {st.session_state['min_clips']}–{st.session_state['max_clips']} clips")
    if st.session_state.get("cut_style", "custom") == "custom":
        notes.append("✏️ your own cutting settings")
    return line, notes


def render_audio_simple(track: dict) -> None:
    """🎵 Step 0, Simple view: the cutting style + pace and where the cuts land. Every dial,
    split screen and the ✂️ cut editor are in the Advanced view; their settings still apply."""
    autosave_now()
    _render_style_row(_setting)
    segs = planned_segments(track, segmentation_settings())
    apply_segment_intensity(segs, track, st.session_state.get("density_contrast", 1.0))
    fig = go.Figure()
    if track.get("hires"):
        _rate = float(track["hires"]["rate"])
        fig.add_trace(go.Scatter(x=np.arange(len(track["hires"]["rms"])) / _rate, y=np.asarray(track["hires"]["rms"], dtype=float),
                                 mode="lines", name="Energy", line=dict(color="rgba(100,150,255,0.55)", width=1),
                                 hoverinfo="skip"))
    else:
        fig.add_trace(go.Scatter(x=[e["time"] for e in track["energy_envelope"]],
                                 y=[e["energy"] for e in track["energy_envelope"]], mode="lines", name="Energy",
                                 line=dict(color="rgba(100,150,255,0.55)"), hoverinfo="skip"))
    add_analysis_overlays(fig, track)
    add_cut_markers(fig, segs)
    fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                      xaxis_title=None, yaxis=dict(showticklabels=False, title=None))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    line, notes = _audio_status(track, segs)
    st.caption(line + (f"  ·  also in use: {', '.join(notes)} (🛠️ Advanced to change)" if notes else ""))
    regions = current_story_map(st.session_state.get("track_id"))
    n_parts = sum(1 for r in regions or [] if r.get("role"))
    if regions and n_parts and st.session_state.get("use_story_map", True):
        st.caption(f"🎭 Using this track's story map — {n_parts} part(s) with a role. Edit it in 🛠️ Advanced → "
                   "🎭 Story map, or in the Media Library.")
    elif regions and n_parts:
        st.caption("🎭 This track's story map is switched off for this project (🛠️ Advanced → Matching → Mode).")
    else:
        st.caption("🎭 No story map saved for this track — clips aren't steered by role. Make one in 🛠️ Advanced "
                   "→ 🎭 Story map, or in the Media Library.")

