"""🧭 Timeline — inside 📁 Media Library: build the footage timeline (footage_timeline.py, Stage 3 step 3) and look at
one video's: movement + hard cuts, which tags show when (your labels marked), shot size, people on screen, movement
rhythm — and each scene's derived tags next to its Reviewer tags. Design: claude/new-matcher-design.md §B."""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

import footage_timeline as FT
from library_common import format_mmss, library_ranges, scene_tags

SCENE_COLOURS = ("rgba(99,110,250,0.10)", "rgba(0,204,150,0.10)")
EXCLUDED = "rgba(128,128,128,0.25)"
OUT_OF_RANGE = "rgba(128,128,128,0.35)"
CUT = "rgba(239,85,59,0.6)"
SERIES = "#636EFA"
SECOND = "#EF553B"
THIRD = "#00CC96"
LABEL = "#FFA15A"


@st.cache_resource(max_entries=4, ttl=600, show_spinner=False)
def _report(sig: tuple) -> dict:
    return FT.status_report()


def _run_build(redo: bool = False, refit: bool = False) -> None:
    st.session_state["ft_msg"] = FT.start_build(redo=redo, refit=refit)
    _report.clear()


def _status_row() -> None:
    rep = _report(FT.signature())
    stt = FT.read_status()
    idx = rep["index"]
    running = stt.get("state") == "running"
    c = st.columns([3, 1, 1])
    if idx.get("built"):
        line = (f"**{len(rep['built'])} videos** built {idx['built']} from student **{idx.get('model')}** · "
                f"{idx.get('emb_dim')} embedding numbers")
        if rep["scores_model"] and rep["scores_model"] != idx.get("model"):
            line += f" · ⚠️ the export is now **{rep['scores_model']}** — build again"
        c[0].markdown(line)
    else:
        c[0].markdown("**No footage timeline yet.** Build it from the vision data + the student export.")
    bits = []
    if rep["ready"]:
        bits.append(f"{len(rep['ready'])} ready to build")
    if rep["no_scores"]:
        bits.append(f"{len(rep['no_scores'])} without student scores (export)")
    if rep["no_vision"]:
        bits.append(f"{len(rep['no_vision'])} without vision data")
    if not rep["vision_root"]:
        bits.append("⚠️ no vision data folder found — set STUDENT_WORK_DIR in config.py")
    if bits:
        c[0].caption(" · ".join(bits))
    c[1].button("🧭 Build", key="ft_build", width="stretch", disabled=running or not rep["vision_root"],
                help="New and changed videos only (a new export or new labels rebuild a video).",
                on_click=_run_build)
    with c[2].popover("⋯", width="stretch", disabled=running):
        st.button("↻ Rebuild every video", key="ft_redo", width="stretch", on_click=_run_build, kwargs={"redo": True})
        st.button("🧮 Fit the embedding space again", key="ft_refit", width="stretch", on_click=_run_build,
                  kwargs={"refit": True},
                  help="Only after many new videos — every video is rebuilt so they all share one space.")
    if running:
        tot = int(stt.get("total") or 0)
        done = int(stt.get("done") or 0)
        st.progress(min(done / tot, 1.0) if tot else 0.0,
                    text=f"Building… {done} of {tot}" + (f" · {stt.get('current')}" if stt.get("current") else ""))
        if st.button("🔄 Refresh", key="ft_refresh"):
            _report.clear()
            st.rerun()
        time.sleep(3)
        _report.clear()
        st.rerun()
    elif stt.get("state") == "error":
        st.error(f"The last build failed: {stt.get('error')}")
    elif stt.get("state") == "done" and st.session_state.pop("ft_msg", None):
        st.success(f"Built {stt.get('built', 0)} · skipped {stt.get('skipped', 0)} ({stt.get('finished')})")


def _scene_shapes(fig, scenes: list, rng, duration: float, cuts) -> None:
    for i, s in enumerate(scenes):
        colour = EXCLUDED if s.get("excluded") else SCENE_COLOURS[i % 2]
        fig.add_vrect(x0=s["start"], x1=s["end"], fillcolor=colour, line_width=0, layer="below")
    if rng:
        if rng[0] > 0:
            fig.add_vrect(x0=0, x1=rng[0], fillcolor=OUT_OF_RANGE, line_width=0, layer="below")
        if rng[1] < duration:
            fig.add_vrect(x0=rng[1], x1=duration, fillcolor=OUT_OF_RANGE, line_width=0, layer="below")
    for c in cuts:
        fig.add_vline(x=float(c), line_color=CUT, line_width=1)


def _layout(fig, height: int, y_title: str = None, x0: float = 0.0, x1: float = None):
    fig.update_layout(height=height, margin=dict(l=4, r=12, t=6, b=4), showlegend=True,
                      legend=dict(orientation="h", y=1.02, x=0, font_size=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_size=12))
    fig.update_xaxes(range=[x0, x1], tickfont_size=10, gridcolor="rgba(128,128,128,0.15)", zeroline=False,
                     tickformat=None)
    fig.update_yaxes(title_text=y_title, title_font_size=11, tickfont_size=10,
                     gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def _mmss_ticks(fig, x0: float, x1: float) -> None:
    span = max(x1 - x0, 1.0)
    step = 10 if span <= 120 else 30 if span <= 600 else 60 if span <= 1800 else 300
    vals = np.arange(np.floor(x0 / step) * step, x1 + step, step)
    fig.update_xaxes(tickvals=vals, ticktext=[format_mmss(v) for v in vals])


def _segments(times: np.ndarray, mask: np.ndarray) -> list:
    """[(start, end)] runs where mask is True (end = the last point + STEP)."""
    out = []
    if not len(times):
        return out
    m = np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], m, [False]]).astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2]):
        out.append((float(times[a]), float(times[b - 1]) + FT.STEP))
    return out


def _movement_chart(tl, scenes, rng, cuts, x0, x1, key) -> None:
    import plotly.graph_objects as go
    t = tl.times
    fig = go.Figure()
    _scene_shapes(fig, scenes, rng, float(t[-1]) + FT.STEP, cuts)
    fig.add_trace(go.Scatter(x=t, y=tl.scaled_move(), name="movement", mode="lines", line=dict(color=SERIES, width=1.5),
                             hovertemplate="%{x:.1f}s · %{y:.2f}<extra>movement</extra>"))
    ch = tl.change.astype(np.float32)
    scale = tl.index.get("move_scale") or {}
    span = max(float(scale.get("hi", 1.0)) - float(scale.get("lo", 0.0)), 1e-6)
    fig.add_trace(go.Scatter(x=t, y=np.clip(ch / span, 0, 2), name="change", mode="lines",
                             line=dict(color=SECOND, width=1, dash="dot"),
                             hovertemplate="%{x:.1f}s · %{y:.2f}/s<extra>change</extra>"))
    on = tl.onset.astype(np.float32)
    hits = np.isfinite(on) & (on == 0)
    if hits.any():
        fig.add_trace(go.Scatter(x=t[hits], y=tl.scaled_move()[hits], name="onsets", mode="markers",
                                 marker=dict(color=THIRD, size=5, symbol="triangle-up"), hoverinfo="skip"))
    _layout(fig, 220, "movement (library 0–1)", x0, x1)
    _mmss_ticks(fig, x0, x1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def _tags_chart(tl, picked: list, scenes, rng, cuts, x0, x1, key) -> None:
    import plotly.graph_objects as go
    t = tl.times
    fig = go.Figure()
    _scene_shapes(fig, scenes, rng, float(t[-1]) + FT.STEP, cuts)
    lab = np.asarray(tl.labelled, bool)
    for tag in picked:
        vis = tl.visible(tag)
        if vis is None:
            continue
        for a, b in _segments(t, vis):
            fig.add_trace(go.Bar(x=[b - a], base=[a], y=[tag], orientation="h", marker_color=SERIES, width=0.6,
                                 showlegend=False, hovertemplate=f"{tag}: {format_mmss(a)}–{format_mmss(b)}<extra></extra>"))
        for a, b in _segments(t, lab & vis):
            fig.add_trace(go.Bar(x=[b - a], base=[a], y=[tag], orientation="h", marker_color=LABEL, width=0.6,
                                 showlegend=False, hovertemplate=f"{tag} (your label): {format_mmss(a)}–{format_mmss(b)}"
                                                                 "<extra></extra>"))
    fig.update_layout(barmode="overlay")
    _layout(fig, 60 + 26 * max(len(picked), 1), None, x0, x1)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(picked)))
    _mmss_ticks(fig, x0, x1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def _shot_people_chart(tl, scenes, rng, cuts, x0, x1, key) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = tl.times
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _scene_shapes(fig, scenes, rng, float(t[-1]) + FT.STEP, cuts)
    fig.add_trace(go.Scatter(x=t, y=tl.shot.astype(np.float32), name="shot size", mode="lines",
                             line=dict(color=SERIES, width=1.5, shape="hv"), connectgaps=False,
                             hovertemplate="%{x:.1f}s · %{y:.1f}<extra>shot</extra>"))
    pov = tl.visible("POV")
    if pov is not None and pov.any():
        for a, b in _segments(t, pov):
            fig.add_trace(go.Scatter(x=[a, b], y=[3.4, 3.4], mode="lines", name="POV", legendgroup="pov",
                                     showlegend=(a, b) == _segments(t, pov)[0], line=dict(color=LABEL, width=6),
                                     hovertemplate=f"POV {format_mmss(a)}–{format_mmss(b)}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=tl.people.astype(np.float32), name="people", mode="lines",
                             line=dict(color=THIRD, width=1, shape="hv"),
                             hovertemplate="%{x:.1f}s · %{y}<extra>people</extra>"), secondary_y=True)
    _layout(fig, 200, None, x0, x1)
    fig.update_yaxes(range=[-0.2, 3.7], tickvals=[0, 1, 2, 3], ticktext=["extreme", "close-up", "medium", "wide"],
                     secondary_y=False)
    fig.update_yaxes(title_text="people", range=[0, max(4, int(tl.people.max()) + 1)], dtick=1, showgrid=False,
                     secondary_y=True)
    _mmss_ticks(fig, x0, x1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def _rhythm_chart(tl, scenes, rng, cuts, x0, x1, key) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = tl.times
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _scene_shapes(fig, scenes, rng, float(t[-1]) + FT.STEP, cuts)
    fig.add_trace(go.Scatter(x=t, y=tl.bpm.astype(np.float32), name="movement BPM", mode="lines",
                             line=dict(color=SERIES, width=1.5, shape="hv"), connectgaps=False,
                             hovertemplate="%{x:.1f}s · %{y:.0f} BPM<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=tl.periodicity.astype(np.float32), name="regularity", mode="lines",
                             line=dict(color=SECOND, width=1, dash="dot", shape="hv"),
                             hovertemplate="%{x:.1f}s · %{y:.2f}<extra>regularity</extra>"), secondary_y=True)
    _layout(fig, 180, "BPM", x0, x1)
    fig.update_yaxes(range=[60, 190], secondary_y=False)
    fig.update_yaxes(title_text="regularity", range=[0, 1.05], showgrid=False, secondary_y=True)
    _mmss_ticks(fig, x0, x1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def _scene_table(tl, scenes: list, rng, min_share: float) -> None:
    derived = FT.derived_scene_tags(tl, scenes, min_share=min_share, time_range=rng)
    rows = []
    for s in scenes:
        d = derived.get(s.get("scene_id"))
        if d is None:
            continue
        rev = scene_tags(s)
        share = d["share"]
        rows.append({"Clip": s.get("scene_id"), "From": format_mmss(s["start"]), "To": format_mmss(s["end"]),
                     "Reviewer tags": ", ".join(rev) or "—",
                     "Derived (moment share)": ", ".join(f"{t} {int(round(share[t] * 100))}%" for t in d["tags"]) or "—",
                     "Also seen": ", ".join(f"{t} {int(round(v * 100))}%" for t, v in sorted(share.items(), key=lambda kv: -kv[1])
                                           if t not in d["tags"] and v >= 0.15) or "—",
                     "Excluded": "🚫" if s.get("excluded") else ""})
    if not rows:
        st.caption("No clips inside the library range have timeline points.")
        return
    st.dataframe(rows, hide_index=True, width="stretch")


def render() -> None:
    st.caption("🧭 Everything the new matcher scores a window on, per half second of every processed video: movement and "
               "its rhythm, which tags show (your labels winning), shot size, people on screen, and the picture "
               "embeddings. Scenes, the library range and Reviewer tags are laid over it live from the catalogue.")
    _status_row()
    rep = _report(FT.signature())
    built = rep["built"]
    if not built:
        return
    import catalogue_store as CS
    index = rep["index"]
    lib = CS.video_index()
    by_id = {v["video_id"]: v for v in lib.values()}
    names = [v for v in built if v in by_id] + [v for v in built if v not in by_id]
    c = st.columns([3, 2])
    vid = c[0].selectbox("Video", names, key="ft_video")
    tl = FT.load(vid, index=index)
    if tl is None or len(tl) == 0:
        st.caption("Nothing in this video's timeline.")
        return
    video = by_id.get(vid) or {}
    scenes = video.get("scenes") or []
    rng = library_ranges().get(vid)
    duration = float(tl.times[-1]) + FT.STEP
    x0, x1 = c[1].slider("Show", 0.0, float(np.ceil(duration)), (0.0, float(np.ceil(duration))), step=1.0,
                         format="%d s", key=f"ft_zoom_{vid}")
    info = (index.get("videos") or {}).get(vid) or {}
    st.caption(f"{format_mmss(duration)} · {len(tl)} points · {len(tl.cuts)} hard cuts · {info.get('labels', 0)} of your "
               f"labels · built {info.get('built', '?')}" + (f" · library range {format_mmss(rng[0])}–{format_mmss(rng[1])}"
                                                           if rng else ""))
    st.markdown("**🏃 Movement** — the person-movement level (scaled within the library), how fast it's changing, and "
                "▲ the pushes of movement. Vertical lines are hard cuts; grey clips are excluded or outside the range.")
    _movement_chart(tl, scenes, rng, tl.cuts, x0, x1, f"ft_move_{vid}")
    st.markdown("**🏷️ Tags** — when each tag is visible (score ≥ its cut-off); orange where your own label decided.")
    top = [t for t, _ in FT.tag_time(tl, tl.select(x0, x1))]
    default = top[:8]
    picked = st.multiselect("Tags", list(tl.tags), default=default, key=f"ft_tags_{vid}", label_visibility="collapsed")
    if picked:
        _tags_chart(tl, picked, scenes, rng, tl.cuts, x0, x1, f"ft_tags_chart_{vid}")
    else:
        st.caption("Pick tags to show.")
    st.markdown("**🎥 Shot size & 👥 people** — extreme close-up → wide (score-weighted), POV in orange, people on screen.")
    _shot_people_chart(tl, scenes, rng, tl.cuts, x0, x1, f"ft_shot_{vid}")
    st.markdown("**🥁 Movement rhythm** — the scene's movement BPM (folded 70–180, like a track) and how regular it is.")
    _rhythm_chart(tl, scenes, rng, tl.cuts, x0, x1, f"ft_rhythm_{vid}")
    with st.expander("🎞️ Clips — derived tags vs Reviewer tags", expanded=False):
        share = st.slider("A tag becomes a clip tag when visible for at least", 0.2, 0.9, 0.5, 0.05, format="%.2f",
                          key=f"ft_share_{vid}",
                          help="The share of the clip's half seconds. Leaning towards half (4 Nov).")
        _scene_table(tl, scenes, rng, share)
