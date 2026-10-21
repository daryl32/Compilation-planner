"""🎭 The story map editor — shared by the Compilation Planner (a project's map) and the Media Library
(a track's saved maps). The caller owns where the map lives; the editor gets it and hands changes back:

    render_editor(track, segs, ns=..., get_regions=..., set_regions=..., version=..., analysis=...,
                  play=...)

  get_regions() -> regions or None     set_regions(regions)  (called from widget callbacks)
  version: an edit counter (widget keys include it, so a change redraws every card)
  play(region, lead_in, key): draws the player for one part (the caller knows where the audio is)

Working down a long list of parts, the graph stays pinned to the top of the screen (CSS sticky, slimmer
than before) with the parts numbered to match the cards, and the part or line you last touched outlined
on it. Works the same on a laptop and an iPad (Safari supports sticky).
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import scene_roles as SR
import story_map as STM
from core.matching import format_mmss

PIN_HEIGHT = 210          # px — the pinned graph is slimmer than the old 300 so the cards get the room
HEADER_REM = 3.75         # Streamlit's top bar; the graph sticks just below it


# --- focus: what you last touched (outlined on the graph) ---------------------------------------
def _focus(ns: str, what) -> None:
    st.session_state[f"{ns}_focus"] = what


def _wrap(ns: str, what, fn, *args):
    """A callback that notes what was touched, then makes the change."""
    def cb():
        _focus(ns, what)
        fn(*args)
    return cb


# --- the graph ------------------------------------------------------------------------------------
def story_chart(track: dict, regions: list, segs: list, roles: dict, playing=None, focus=None,
                height: int = PIN_HEIGHT) -> go.Figure:
    """Loudness curve, the cuts, the parts in their role colours, numbered like the cards; the part being
    played or last touched outlined, a line last moved drawn dashed."""
    fig = go.Figure()
    if track.get("hires"):
        rate = float(track["hires"]["rate"])
        rms = np.asarray(track["hires"]["rms"], dtype=float)
        k = max(1, int(rate * 0.5))
        sm = np.convolve(rms, np.ones(k) / k, mode="same")
        sm = sm / max(float(sm.max()), 1e-9)
        fig.add_trace(go.Scatter(x=np.arange(len(sm)) / rate, y=sm, mode="lines", name="Loudness",
                                 line=dict(color="rgba(80,120,220,0.85)", width=1.5), hoverinfo="skip"))
    xs, ys = [], []
    for sg in segs[1:]:
        xs += [sg["start"], sg["start"], None]
        ys += [0, 1.05, None]
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Cuts (blocks)", hoverinfo="skip",
                                 line=dict(color="rgba(128,128,128,0.35)", width=1)))
    f_part = focus[1] if isinstance(focus, (tuple, list)) and focus[0] == "part" else None
    f_line = focus[1] if isinstance(focus, (tuple, list)) and focus[0] == "line" else None
    for i, r in enumerate(regions):
        role = roles.get(r["role"]) if r["role"] else None
        colour = role["colour"] if role else "#bbbbbb"
        hot = i == playing or i == f_part
        fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=colour,
                      opacity=(0.45 if hot else 0.32) if role else (0.14 if hot else 0.06),
                      line_width=3 if hot else 0, line_color="#ff4b4b" if hot else "#222222", layer="below")
        mid = (r["start"] + r["end"]) / 2
        fig.add_annotation(x=mid, y=1.13, showarrow=False, font=dict(size=11),
                           text=f"<b>{i + 1}</b>" + (f" {role['emoji']}" if role else ""))
    if f_line is not None and f_line + 1 < len(regions):
        t = regions[f_line]["end"]
        fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1.2, line=dict(color="#ff4b4b", width=2, dash="dash"))
    fig.update_layout(height=height, margin=dict(t=24, b=4, l=6, r=6), showlegend=False,
                      xaxis=dict(range=[0, float(track["duration_sec"])], ticksuffix="s"),
                      yaxis=dict(visible=False, range=[0, 1.22]),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _editor_css(ns: str) -> None:
    """The graph's container sticks under the top bar while the cards scroll past. It needs a solid
    background in the page's own colour, or the cards would show through it. Streamlit wraps a keyed
    container in a stLayoutWrapper of its own size, so the wrapper is what has to stick (:has — Safari 15.4+)."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    bg = "#0e1117" if dark else "#ffffff"
    st.markdown(f"""<style>
[data-testid="stLayoutWrapper"]:has(> .st-key-{ns}_pin), .st-key-{ns}_pin {{
  position: sticky; top: {HEADER_REM}rem; z-index: 50; background: {bg}; }}
.st-key-{ns}_pin {{ padding: .25rem 0 .1rem; border-bottom: 1px solid rgba(128,128,128,.25); }}
.st-key-{ns}_pin .stPlotlyChart {{ margin-bottom: 0; }}
/* Narrower screens (iPad either way round, a small laptop window): a part's row wraps — colour, time and role on
   top, the buttons beside or below, the 💡 note last; a line's caption above its slider. */
@media (max-width: 1300px) {{
  [class*="st-key-{ns}_card_"] [data-testid="stHorizontalBlock"],
  [class*="st-key-{ns}_lrow_"] [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; row-gap: .4rem; }}
  [class*="st-key-{ns}_card_"] [data-testid="stColumn"] {{ width: auto !important; min-width: 0 !important; }}
  [class*="st-key-{ns}_card_"] [data-testid="stColumn"]:nth-child(1) {{ flex: 0 0 18px !important; }}
  [class*="st-key-{ns}_card_"] [data-testid="stColumn"]:nth-child(2) {{ flex: 1 1 130px !important; }}
  [class*="st-key-{ns}_card_"] [data-testid="stColumn"]:nth-child(3) {{ flex: 1 1 170px !important; }}
  [class*="st-key-{ns}_card_"] [data-testid="stColumn"]:nth-child(4) {{ flex: 1 1 100% !important; order: 10; }}
  [class*="st-key-{ns}_card_"] [data-testid="stColumn"]:nth-child(n+5) {{ flex: 1 1 88px !important; }}
  [class*="st-key-{ns}_lrow_"] [data-testid="stColumn"] {{ width: auto !important; min-width: 0 !important; }}
  [class*="st-key-{ns}_lrow_"] [data-testid="stColumn"]:nth-child(1) {{ flex: 1 1 100% !important; }}
  [class*="st-key-{ns}_lrow_"] [data-testid="stColumn"]:nth-child(2) {{ flex: 1 1 200px !important; }}
  [class*="st-key-{ns}_lrow_"] [data-testid="stColumn"]:nth-child(3) {{ flex: 0 0 48px !important; }}
}}
</style>""", unsafe_allow_html=True)


# --- the editor -------------------------------------------------------------------------------------
def render_empty(duration: float, analysis: dict, cut_times: list, *, set_regions, ns: str,
                 note: str = "Or skip this step — without a story map the planner works as before.") -> None:
    c = st.columns([2, 2, 3])
    c[0].button("✨ Suggest a story map from the song", type="primary", width="stretch",
                key=f"{ns}_suggest_first", disabled=analysis is None,
                on_click=lambda: set_regions(STM.snap(STM.suggest(analysis, SR.all_roles()), cut_times)))
    c[1].button("Start with an empty map", width="stretch", key=f"{ns}_blank_first",
                on_click=lambda: set_regions(STM.blank(duration)))
    if note:
        c[2].caption(note)


def render_editor(track: dict, segs: list, *, ns: str, get_regions, set_regions, version: int,
                  analysis: dict, play=None) -> None:
    duration = float(track["duration_sec"])
    cut_times = [sg["start"] for sg in segs] + [duration]
    roles = SR.roles_by_id()
    regions = get_regions()
    v = version

    def edit(fn, *args):
        set_regions(fn(get_regions(), *args))
        st.session_state.pop(f"{ns}_play", None)

    def role_changed(i, key):
        edit(STM.set_role, i, st.session_state.get(key))

    def boundary_changed(i, key):
        edit(STM.move_boundary, i, float(st.session_state[key]))

    def snap_one(i):
        rs = get_regions()
        t = rs[i]["end"]
        inner = [c for c in cut_times if rs[i]["start"] < c < rs[i + 1]["end"]]
        if inner:
            edit(STM.move_boundary, i, min(inner, key=lambda c: abs(c - t)))

    def toggle_play(i):
        st.session_state[f"{ns}_play"] = None if st.session_state.get(f"{ns}_play") == i else i

    tc = st.columns([2, 2, 2, 3])
    tc[0].button("✨ Suggest again", width="stretch", key=f"{ns}_suggest", disabled=analysis is None,
                 on_click=_wrap(ns, None, lambda: set_regions(STM.snap(STM.suggest(analysis, SR.all_roles()),
                                                                       cut_times))),
                 help="Replaces the map with a fresh suggestion from the song (your edits are lost).")
    tc[1].button("🧹 Clear the map", width="stretch", key=f"{ns}_clear",
                 on_click=_wrap(ns, None, lambda: set_regions(STM.blank(duration))),
                 help="One region, no role — then split and set roles yourself.")
    tc[2].button("📐 Line up with the cuts", width="stretch", key=f"{ns}_snapall",
                 on_click=_wrap(ns, None, edit, STM.snap, cut_times),
                 help="Moves every boundary onto the nearest cut, so no block straddles two parts. "
                      "Useful after changing the audio settings.")
    lead = tc[3].toggle("▶ starts 2s early", key=f"{ns}_lead_in",
                        help="Hear what leads into each part, not just the part itself.")
    snap = tc[3].toggle("Handles snap to cuts", key=f"{ns}_snap",
                        help="On: the lines between parts stop only on the cut points. Off: they move freely "
                             "(0.1s steps) to line up with the audio curve — 📐 next to a line snaps just that "
                             "one to its nearest cut.")

    playing = st.session_state.get(f"{ns}_play")
    focus = st.session_state.get(f"{ns}_focus")
    if isinstance(focus, (tuple, list)) and focus[1] >= len(regions):
        focus = None
    _editor_css(ns)
    with st.container(key=f"{ns}_pin"):
        st.plotly_chart(story_chart(track, regions, segs, roles, playing, focus), width="stretch",
                        config={"displayModeBar": False}, key=f"{ns}_chart_{v}")

    tot = STM.totals(regions, segs)
    if tot:
        st.markdown("  ·  ".join(f"{SR.label(r, roles)} **{format_mmss(x['sec'])}** ({x['blocks']} block"
                                 f"{'s' if x['blocks'] != 1 else ''})" for r, x in tot.items()))
    unknown = sorted({r["role"] for r in regions if r["role"] and r["role"] not in roles})
    if unknown:
        st.warning("Some parts use a role that's been deleted: " + ", ".join(unknown))

    role_opts = [None, *roles, *unknown]
    block_role = STM.block_roles(regions, segs)
    _lab = lambda x: SR.label(x, roles) if x else "no role"
    for i, r in enumerate(regions):
        role = roles.get(r["role"]) if r["role"] else None
        n_blocks = sum(1 for sg in segs if r["start"] <= (sg["start"] + sg["end"]) / 2 < r["end"])
        part = ("part", i)
        with st.container(border=True, key=f"{ns}_card_{v}_{i}"):
            rc = st.columns([0.12, 1.5, 1.8, 3.2, 0.9, 0.9, 0.9])
            rc[0].markdown(f'<div style="width:18px;height:46px;border-radius:4px;background:'
                           f'{role["colour"] if role else "#cccccc"};"></div>', unsafe_allow_html=True)
            rc[1].markdown(f"**{i + 1}** · **{format_mmss(r['start'])} – {format_mmss(r['end'])}**  \n"
                           f":gray[{r['end'] - r['start']:.0f}s · {n_blocks} block{'s' if n_blocks != 1 else ''}]")
            rk = f"{ns}_role_{v}_{i}"
            rc[2].selectbox("Role", role_opts, index=role_opts.index(r["role"]) if r["role"] in role_opts else 0,
                            key=rk, format_func=lambda x: "— no role" if x is None else SR.label(x, roles),
                            label_visibility="collapsed", on_change=_wrap(ns, part, role_changed, i, rk))
            with rc[3]:
                if r.get("why"):
                    st.caption(("✏️ " if r["why"] == "set by you" else "💡 ") + r["why"])
                alt = r.get("alt")
                if alt and alt.get("role") in roles and alt["role"] != r["role"]:
                    st.button(f"↔ {SR.label(alt['role'], roles)} instead", key=f"{ns}_alt_{v}_{i}",
                              on_click=_wrap(ns, part, edit, STM.set_role, i, alt["role"]),
                              help="The runner-up for this part of the song.")
            rc[4].button("⏹ Stop" if playing == i else "▶ Play", key=f"{ns}_play_{v}_{i}",
                         width="stretch", on_click=_wrap(ns, part, toggle_play, i))
            inner = [t for t in cut_times if r["start"] < t < r["end"]]
            mid = min(inner, key=lambda t: abs(t - (r["start"] + r["end"]) / 2)) if inner else None
            rc[5].button("✂️ Split", key=f"{ns}_split_{v}_{i}", width="stretch", disabled=mid is None,
                         on_click=_wrap(ns, part, edit, STM.split, i, mid),
                         help="Split into two at the cut nearest the middle (then move the line below).")
            rc[6].button("🔗 Join", key=f"{ns}_merge_{v}_{i}", width="stretch",
                         disabled=i == len(regions) - 1, on_click=_wrap(ns, part, edit, STM.merge_next, i),
                         help="Join with the next part below (keeps this part's role).")
            if playing == i and play is not None:
                play(r, lead, f"{ns}_{v}_{i}")

        # The line between this part and the next: free (0.1s steps) or on the cuts.
        if i + 1 < len(regions):
            nxt = regions[i + 1]
            bk = f"{ns}_line_{v}_{i}_{'s' if snap else 'f'}"
            line = ("line", i)
            on_cut = any(abs(c - r["end"]) < 0.05 for c in cut_times)
            lc = st.container(key=f"{ns}_lrow_{v}_{i}").columns([1.3, 3.6, 0.4])
            lc[0].caption(f"↕ {i + 1}→{i + 2}: where {_lab(r['role'])} ends and {_lab(nxt['role'])} begins"
                          + ("" if on_cut else "  ·  :orange[between cuts]"))
            if snap:
                opts = sorted({t for t in cut_times if r["start"] < t < nxt["end"]} | {r["end"]})
                if len(opts) > 1:
                    lc[1].select_slider(
                        "Line between these parts", opts, value=r["end"], key=bk,
                        format_func=lambda t: f"{format_mmss(t)} ({t:.1f}s)", label_visibility="collapsed",
                        on_change=_wrap(ns, line, boundary_changed, i, bk),
                        help="Drag to move where one part ends and the next begins — it stops on the cuts.")
            else:
                lo = math.ceil((r["start"] + 0.1) * 10) / 10
                hi = math.floor((nxt["end"] - 0.1) * 10) / 10
                if hi > lo:
                    lc[1].slider(
                        "Line between these parts", min_value=float(lo), max_value=float(hi),
                        value=float(min(max(round(r["end"], 1), lo), hi)), step=0.1, format="%.1fs", key=bk,
                        label_visibility="collapsed", on_change=_wrap(ns, line, boundary_changed, i, bk),
                        help="Drag to line the change up with the audio curve (arrow keys move it 0.1s). "
                             "📐 snaps it to the nearest cut.")
            lc[2].button("📐", key=f"{ns}_snap1_{v}_{i}", disabled=on_cut,
                         on_click=_wrap(ns, line, snap_one, i), help="Snap this line to the nearest cut.")

    st.divider()
    n_role = sum(1 for x in block_role if x)
    st.caption(f"{n_role} of {len(segs)} blocks have a role. Times are kept in seconds, so changing the "
               "audio settings later keeps the map where it is (📐 lines it up with the new cuts).")


__all__ = ["render_editor", "render_empty", "story_chart"]
