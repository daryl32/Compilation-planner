"""
Plotly helpers for drawing a track's extra analysis (sections, build-ups,
drops, bar lines, drum / chord / vocal curves) — shared by the Compilation
Planner's charts and the Media Library. No Streamlit in here.
"""

import plotly.graph_objects as go

SECTION_SHADES = ["rgba(120,120,220,0.10)", "rgba(220,140,60,0.12)", "rgba(60,180,120,0.12)",
                  "rgba(200,80,160,0.10)", "rgba(160,160,60,0.12)"]


def section_shade(label: str) -> str:
    return SECTION_SHADES[(ord((label or "A")[0]) - ord("A")) % len(SECTION_SHADES)]


def add_analysis_overlays(fig, track: dict) -> None:
    """Sections (shaded + labelled), build-ups, drops, bar lines and the extra
    curves from the server analysis — most hidden until clicked in the legend."""
    a = track.get("analysis") or {}
    if not a:
        return
    for sec in a.get("sections") or []:
        fig.add_vrect(x0=sec["start"], x1=sec["end"], fillcolor=section_shade(sec["label"]),
                      line_width=0, layer="below",
                      annotation_text=sec["label"], annotation_position="top left",
                      annotation_font_size=10, annotation_font_color="rgba(90,90,90,0.9)")
    for b in a.get("builds") or []:
        fig.add_vrect(x0=b["start"], x1=b["end"], fillcolor="rgba(250,200,0,0.10)", line_width=0, layer="below")
    rate = float(a.get("rate", 30))
    curves = [("kick", "Kick", "rgba(200,60,40,0.6)"), ("snare", "Snare", "rgba(40,140,200,0.6)"),
              ("hat", "Hi-hat", "rgba(120,120,120,0.5)"), ("harmony", "Chord change", "rgba(60,170,90,0.7)"),
              ("novelty", "Section novelty", "rgba(150,60,200,0.7)"), ("vocals", "Vocals", "rgba(230,120,0,0.8)")]
    for key, name, colour in curves:
        vals = a.get(key)
        if vals:
            fig.add_trace(go.Scatter(x=[i / rate for i in range(len(vals))], y=vals, mode="lines", name=name,
                                     line=dict(color=colour, width=1), visible="legendonly"))
    xs, ys = [], []
    for t in a.get("downbeats") or []:
        xs += [t, t, None]
        ys += [0, 0.08, None]
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Bar lines", hoverinfo="skip",
                                 line=dict(color="rgba(60,60,60,0.5)", width=1), visible="legendonly"))
    if a.get("drops"):
        fig.add_trace(go.Scatter(x=a["drops"], y=[1.02] * len(a["drops"]), mode="markers+text", name="Drop",
                                 text=["drop"] * len(a["drops"]), textposition="top center",
                                 marker=dict(symbol="triangle-down", size=9, color="rgba(220,40,40,0.9)")))


def track_figure(track: dict, height: int = 380) -> go.Figure:
    """Full chart of one track: energy (and onset strength), plus every
    overlay from the extra analysis. Curves that would clutter it start
    hidden — click them in the legend."""
    fig = go.Figure()
    hires = track.get("hires") or {}
    if hires.get("rms"):
        rate = float(hires["rate"])
        t = [i / rate for i in range(len(hires["rms"]))]
        fig.add_trace(go.Scatter(x=t, y=hires["rms"], mode="lines", name="Energy",
                                 line=dict(color="rgba(100,150,255,0.7)", width=1)))
        if hires.get("onset"):
            fig.add_trace(go.Scatter(x=t, y=hires["onset"], mode="lines", name="Onset strength",
                                     line=dict(color="rgba(150,150,150,0.4)", width=1), visible="legendonly"))
    elif track.get("energy_envelope"):
        env = track["energy_envelope"]
        fig.add_trace(go.Scatter(x=[e["time"] for e in env], y=[e["energy"] for e in env], mode="lines",
                                 name="Energy", line=dict(color="rgba(100,150,255,0.7)")))
    add_analysis_overlays(fig, track)
    fig.update_layout(height=height, margin=dict(t=24, b=10, l=10, r=10), xaxis_title="Time (s)",
                      yaxis=dict(range=[0, 1.12], showticklabels=False),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def mini_figure(energy: list, duration: float, sections: list, drops: list, height: int = 80) -> go.Figure:
    """Compact row chart: energy with the sections shaded and drops marked."""
    fig = go.Figure()
    n = max(1, len(energy))
    fig.add_trace(go.Scatter(x=[duration * i / n for i in range(n)], y=energy, mode="lines",
                             line=dict(color="rgba(100,150,255,0.9)", width=1), hoverinfo="skip"))
    for sec in sections or []:
        fig.add_vrect(x0=sec["start"], x1=sec["end"], fillcolor=section_shade(sec["label"]), line_width=0,
                      layer="below", annotation_text=sec["label"], annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="rgba(90,90,90,0.9)")
    for d in drops or []:
        fig.add_vline(x=d, line=dict(color="rgba(220,40,40,0.8)", width=1, dash="dot"))
    fig.update_layout(height=height, margin=dict(t=0, b=0, l=0, r=0), showlegend=False,
                      xaxis=dict(visible=False, range=[0, duration]), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
