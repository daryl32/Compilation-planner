"""
🔮 Recommendations — "For this track": the best sets of videos and cut settings
for a track, searched across the whole library and checked with the real
matcher (core/recommend.py). Each plan opens straight into the planner.
"""

import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import catalogue_store as CStore
import core.footage as CFOOT
import core.matching as CMATCH
import core.recommend as REC
import curve_cache
import scene_roles as SR
import story_map as STM
from config import AUDIO_DIR, CATALOGUE_DIR
from core.segmentation import SEG_SMART, SEG_DEFAULTS, SEG_METHOD_KEYS
from library_common import library_ranges, safe_image, scene_thumbnail_path, shared_cache
from page_setup import page_setup

page_setup("Recommendations", "pages/5_Recommendations.py", title="🔮 Recommendations")

RESULTS_DIR = curve_cache.default_cache_dir(CATALOGUE_DIR).parent / "recommendations"


# ---------------------------------------------------------------------------
# Library data
# ---------------------------------------------------------------------------

def list_tracks() -> list:
    return sorted(f.stem for f in AUDIO_DIR.glob("*.json") if f.name != "audio_index.json")


def load_track(track_id: str) -> dict:
    return json.loads((AUDIO_DIR / f"{track_id}.json").read_text())


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@shared_cache
@st.cache_resource(max_entries=400, ttl=3600, show_spinner=False)
def _fingerprint(video_id: str, mtime: float, rng: tuple):
    cat = CStore.all_catalogues().get(video_id)
    if not cat:
        return None
    return REC.fingerprint_video(video_id, cat, CStore.motion_curves(video_id), rng or None)


def _stems() -> dict:
    """{video_id: catalogue file stem}."""
    return {slim["video_id"]: stem for stem, slim in CStore.video_index().items()}


def library_prints(progress=None) -> dict:
    cats = CStore.all_catalogues()
    ranges = library_ranges()
    stems = _stems()
    out = {}
    ids = sorted(cats)
    for i, vid in enumerate(ids):
        if progress:
            progress(i / max(len(ids), 1), f"Reading video {i + 1} of {len(ids)}…")
        vp = _fingerprint(vid, _mtime(CStore.catalogue_path(stems.get(vid, vid))), tuple(ranges.get(vid) or ()))
        if vp is not None:
            out[vid] = vp
    return out


def plan_context() -> CMATCH.PlanContext:
    # Same durations as the planner's get_video_duration (the last scene's end).
    durations = {vid: CFOOT._tc_to_seconds(cat["scenes"][-1]["end_tc"]) if cat.get("scenes") else 0.0
                 for vid, cat in CStore.all_catalogues().items()}
    return CMATCH.PlanContext(time_ranges=library_ranges, duration_of=lambda v: durations.get(v, 0.0),
                              curves_of=CStore.motion_curves, catalogue_dir=CATALOGUE_DIR)


def make_matcher(track: dict):
    """The real matcher as Auto mode runs it with the planner's default settings
    (one clip per block, ✨ improved auto-fill, never the same video twice in a row)."""
    cats = CStore.all_catalogues()
    ranges = library_ranges()
    ctx = plan_context()

    def match(segments, video_ids):
        q = CFOOT.build_footage_queues(cats, tuple(video_ids), (), ranges,
                                       thumbnail_for=lambda stored, vid: CATALOGUE_DIR / "thumbnails" / vid /
                                       Path(stored).name, roles_of=SR.scene_roles)
        # Exactly the planner's Auto call with its defaults (split screen off → 1 clip a block,
        # Balanced weights, no story roles) — tests/test_recommend.py checks they agree.
        timeline, _, _ = CMATCH.match_scenes_to_track(
            [dict(s) for s in segments], q, 1, 1, 0.0, 1.0, False, False, {}, {}, set(), False, 0, 1.0,
            set(), set(), 2.0, 0.5, True, 1.5, 2.0, 42, weight_position=1.0, repeat_fallbacks=set(),
            block_roles=[None] * len(segments), track=track, ctx=ctx)
        return timeline
    return match


def signature(track_id: str, n_videos: int) -> str:
    files = sorted((p.name, round(_mtime(p))) for p in CStore.catalogue_paths())
    return hashlib.sha1(json.dumps([track_id, n_videos, round(_mtime(AUDIO_DIR / f"{track_id}.json")), files,
                                    sorted(library_ranges().items()), REC.VARIANTS, REC.DROP_WEIGHT],
                                   default=str).encode()).hexdigest()[:16]


def _results_path(track_id: str) -> Path:
    return RESULTS_DIR / f"{hashlib.sha1(track_id.encode()).hexdigest()[:16]}.json"


def load_results(track_id: str, sig: str):
    try:
        data = json.loads(_results_path(track_id).read_text())
    except (OSError, ValueError):
        return None
    return data if data.get("sig") == sig else None


def slim_plan(plan: dict) -> dict:
    """What's kept of a plan: no segments or full timeline, just each block's picks
    (video, scene, start) — enough to show it and to check the planner rebuilds it."""
    out = {k: plan[k] for k in ("variant", "settings", "videos", "scores", "estimate")}
    out["picks"] = [[[c.get("video_id"), c.get("scene_id"), round(float(c.get("clip_start_sec") or 0), 3)]
                     for c in block.get("scenes") or []] for block in plan.get("timeline") or []]
    return out


def save_results(track_id: str, sig: str, plans: list) -> None:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _results_path(track_id).write_text(json.dumps({"sig": sig, "track_id": track_id, "plans": plans}))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Opening a plan in the planner
# ---------------------------------------------------------------------------

def open_plan(track_id: str, plan: dict, rank: int) -> None:
    settings = {"segmentation_method": SEG_SMART, "beats_per_bar": 4,
                **{k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_SMART]}, **plan["settings"]}
    st.session_state["_open_recommended_plan"] = {
        "track_id": track_id, "settings": settings, "videos": list(plan["videos"]),
        "label": f"Recommended plan {rank} for {track_id} — {plan['variant']}, "
                 f"{plan['scores']['overall']:.0f}% overall"}
    st.switch_page("pages/1_Compilation_Planner.py")


# ---------------------------------------------------------------------------
# Drawing a plan
# ---------------------------------------------------------------------------

def _strip(scores: dict) -> go.Figure:
    vals = scores["per_block"]
    act = set(scores["action_blocks"])
    colours = ["#2e9e5b" if v >= 70 else "#e8a317" if v >= 50 else "#d9534f" for v in vals]
    fig = go.Figure(go.Bar(x=list(range(1, len(vals) + 1)), y=vals, marker_color=colours,
                           marker_line_color=["#222" if i in act else "rgba(0,0,0,0)" for i in range(len(vals))],
                           marker_line_width=[2 if i in act else 0 for i in range(len(vals))],
                           hovertemplate="Block %{x}: %{y:.0f}%<extra></extra>"))
    if scores.get("weakest_block") is not None:
        fig.add_annotation(x=scores["weakest_block"] + 1, y=vals[scores["weakest_block"]], text="▼",
                           showarrow=False, yshift=10, font=dict(size=12),
                           hovertext=f"Weakest block ({scores['weakest_block'] + 1})")
    fig.update_layout(height=150, margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(range=[0, 100], title=None),
                      xaxis=dict(title=None, showticklabels=False), showlegend=False)
    return fig


def _thumb(video_id: str):
    cat = CStore.all_catalogues().get(video_id) or {}
    for s in cat.get("scenes", []):
        p = scene_thumbnail_path(video_id, s)
        if p:
            return p
    return None


def render_plan(track_id: str, rank: int, plan: dict) -> None:
    sc = plan["scores"]
    with st.container(border=True):
        h = st.columns([2.4, 1])
        h[0].markdown(f"### #{rank} · {sc['overall']:.0f}% overall")
        h[0].caption(f"{plan['variant']} · {len(plan['videos'])} videos")
        open_now = h[1].button("🎬 Open", key=f"open_{track_id}_{rank}", type="primary",
                               width="stretch")
        m = st.columns(4)
        m[0].metric("Shape", f"{sc['shape']:.0f}%", help="How closely each clip's motion rises and falls with "
                    "its block's music — every block counts equally.")
        m[1].metric("Intensity", f"{sc['intensity']:.0f}%", help=f"How well each clip's motion level matches "
                    f"its block's intensity — drops and action parts count {REC.DROP_WEIGHT:g}×.")
        m[2].metric("Action", "—" if sc["action"] is None else f"{sc['action']:.0f}%",
                    help="Intensity match over the drops and action parts only (outlined in the strip).")
        m[3].metric("Weakest block", "—" if sc["weakest"] is None else f"{sc['weakest']:.0f}%",
                    help=f"Block {(sc['weakest_block'] or 0) + 1} — the lowest-scoring block, so a high "
                         f"average can't hide a weak spot.")
        st.plotly_chart(_strip(sc), width="stretch", key=f"strip_{track_id}_{rank}",
                        config={"displayModeBar": False})
        cols = st.columns(min(len(plan["videos"]), 8) or 1)
        for c, vid in zip(cols * 2, plan["videos"]):
            with c:
                t = _thumb(vid)
                if t:
                    safe_image(str(t), width="stretch")
                st.caption(vid)
    if open_now:
        open_plan(track_id, plan, rank)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.caption("Searches your whole library for the sets of videos and cut settings that fit a track best, then "
           "checks each one with the real matcher — the scores are what you get when you open it. **🎬 Open** makes "
           "a plan a new project in the planner (what you had there is saved first).")

tracks = list_tracks()
if not tracks:
    st.info("No audio tracks found.")
    st.stop()

c = st.columns([3, 2, 2])
default_track = st.session_state.get("track_id") if st.session_state.get("track_id") in tracks else tracks[0]
track_id = c[0].selectbox("🎵 Track", tracks, index=tracks.index(default_track), key="rec_track")
n_videos = c[1].slider("Videos per plan", 2, 15, 8, key="rec_n_videos",
                       help="How many videos each plan uses (as in the planner's video selection).")
sig = signature(track_id, n_videos)
cached = st.session_state.get("rec_results", {}).get(track_id)
if not (cached and cached.get("sig") == sig):
    cached = load_results(track_id, sig)
go_btn = c[2].button("🔍 Find plans" if not cached else "🔄 Search again", type="primary",
                     width="stretch")

if go_btn:
    bar = st.progress(0.0, text="Reading your library…")
    try:
        prints = library_prints(lambda f, t: bar.progress(0.25 * f, text=t))
        if len(prints) < 2:
            bar.empty()
            st.warning("Recommendations need at least two videos with motion data.")
            st.stop()
        track = load_track(track_id)
        story = STM.saved_map(track_id)
        from diagnostics import timed
        with timed("Recommendations: for this track", f"{len(prints)} videos"):
            plans = REC.recommend_for_track(
                track, prints, match=make_matcher(track), curves_of=CStore.motion_curves, story_regions=story,
                n_videos=min(n_videos, len(prints)),
                progress=lambda f, t: bar.progress(0.25 + 0.75 * min(f, 1.0), text=t))
    finally:
        bar.empty()
    cached = {"sig": sig, "track_id": track_id, "plans": [slim_plan(p) for p in plans]}
    st.session_state.setdefault("rec_results", {})[track_id] = cached
    save_results(track_id, sig, cached["plans"])

if cached:
    if STM.saved_map(track_id):
        st.caption("🎭 This track's story map marks its action parts — they count extra in the intensity score.")
    for i, plan in enumerate(cached["plans"], start=1):
        render_plan(track_id, i, plan)
else:
    st.info("Pick a track and press **🔍 Find plans**. The first search reads every video's motion, "
            "so it takes a little while; after that it's quicker.")
