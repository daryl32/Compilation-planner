"""
🔮 Recommendations — "For this track": five plans for a track (📐 Best shape fit,
🎭 Best story fit, 🏆 Best overall, 🥊 Verses, 🎲 Wildcard), searched across the
whole library and checked with the real matcher (core/recommend.py). Each plan
opens straight into the planner.
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
import similarity as SIM
import story_map as STM
from config import AUDIO_DIR, CATALOGUE_DIR
from library_common import library_ranges, safe_image, scene_thumbnail_path, shared_cache, video_details
from page_setup import page_setup

page_setup("Recommendations", "pages/5_Recommendations.py", title="🔮 Recommendations")

RESULTS_DIR = curve_cache.default_cache_dir(CATALOGUE_DIR).parent / "recommendations"
DEFAULT_VIDEOS = 4
RESULTS_VERSION = 3          # bump when what's stored changes (older results are searched again)


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


def _thumb_for(stored, vid):
    return CATALOGUE_DIR / "thumbnails" / vid / Path(stored).name


def role_seconds() -> dict:
    """{video_id: {role: seconds of usable footage marked with it}} (library ranges applied)."""
    cats = CStore.all_catalogues()
    q = CFOOT.build_footage_queues(cats, tuple(cats), (), library_ranges(), thumbnail_for=_thumb_for,
                                   roles_of=SR.scene_roles)
    return {vid: CFOOT.role_supply({vid: spans}) for vid, spans in q.items()}


def scene_roles_lookup():
    """roles_of(video_id, scene_id) → the scene's role ids (for the Story score)."""
    table = {(vid, sc["scene_id"]): SR.scene_roles(sc)
             for vid, cat in CStore.all_catalogues().items() for sc in cat.get("scenes", [])}
    return lambda vid, sid: table.get((vid, sid), ())


def actors_by_video() -> dict:
    return {vid: d["actors"] for vid, d in video_details().items() if d.get("actors")}


def similar_finder():
    """similar_of(video_id) → [(similarity, reason, video_id)] across the library, best first —
    the planner's ✨ Similar videos ("find similar") scoring."""
    cats = CStore.all_catalogues()
    videos = SIM.library_video_inputs(cats, library_ranges(), CATALOGUE_DIR)
    idf = SIM.tag_idf(cats, lambda sc: sc.get("tags", []))
    return lambda vid: SIM.rank_similar_videos(vid, videos, idf, top=len(videos))


def story_for(track_id: str, track: dict) -> tuple:
    """(regions or None, "saved" | "suggested" | None) — the track's saved story map, else the
    one the story map wizard would suggest for it."""
    saved = STM.saved_map(track_id)
    if saved:
        return saved, "saved"
    try:
        regions = STM.suggest(STM.analyse(track, 4), SR.all_roles())
    except Exception:          # a track the analysis can't read — no story parts
        return None, None
    return (regions, "suggested") if any(r.get("role") for r in regions) else (None, None)


def plan_context(block_roles=None) -> CMATCH.PlanContext:
    # Same durations as the planner's get_video_duration (the last scene's end).
    durations = {vid: CFOOT._tc_to_seconds(cat["scenes"][-1]["end_tc"]) if cat.get("scenes") else 0.0
                 for vid, cat in CStore.all_catalogues().items()}
    # 🎭 With the story map in use, the planner reserves and locks the roles it uses
    # (its defaults: 'Keep role clips for their parts' on, Intro/Outro preference off).
    used = frozenset(r for r in (block_roles or []) if r)
    return CMATCH.PlanContext(time_ranges=library_ranges, duration_of=lambda v: durations.get(v, 0.0),
                              curves_of=CStore.motion_curves, catalogue_dir=CATALOGUE_DIR,
                              reserved_roles=used, locked_roles=used, story_in_use=bool(used))


def make_matcher(track: dict):
    """The real matcher as Auto mode runs it with the planner's default settings
    (one clip per block, ✨ improved auto-fill, never the same video twice in a row),
    with or without the story map and in sequential order when asked."""
    cats = CStore.all_catalogues()
    ranges = library_ranges()

    def match(segments, video_ids, block_roles=None, sequential=False):
        q = CFOOT.build_footage_queues(cats, tuple(video_ids), (), ranges, thumbnail_for=_thumb_for,
                                       roles_of=SR.scene_roles)
        roles = list(block_roles) if block_roles else [None] * len(segments)
        # Exactly the planner's Auto call with its defaults (split screen off → 1 clip a block,
        # Balanced weights; F (position) only in sequential mode) — tests/test_recommend.py
        # checks they agree.
        timeline, _, _ = CMATCH.match_scenes_to_track(
            [dict(s) for s in segments], q, 1, 1, 0.0, 1.0, False, bool(sequential), {}, {}, set(), False, 0, 1.0,
            set(), set(), 2.0, 0.5, True, 1.5, 2.0, 42, weight_position=1.0 if sequential else 0.0,
            repeat_fallbacks=set(), block_roles=roles, track=track, ctx=plan_context(block_roles))
        return timeline
    return match


def signature(track_id: str, n_videos: int, sequential: bool, story) -> str:
    files = sorted((p.name, round(_mtime(p))) for p in CStore.catalogue_paths())
    return hashlib.sha1(json.dumps([RESULTS_VERSION, track_id, n_videos, bool(sequential),
                                    round(_mtime(AUDIO_DIR / f"{track_id}.json")), files,
                                    sorted(library_ranges().items()), REC.VARIANTS, REC.DROP_WEIGHT,
                                    story, sorted(actors_by_video().items())],
                                   default=str).encode()).hexdigest()[:16]


def _results_path(track_id: str, sequential: bool) -> Path:
    key = f"{track_id}|seq" if sequential else track_id
    return RESULTS_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def load_results(track_id: str, sequential: bool, sig: str):
    try:
        data = json.loads(_results_path(track_id, sequential).read_text())
    except (OSError, ValueError):
        return None
    return data if data.get("sig") == sig else None


def slim_plan(plan: dict, story_regions) -> dict:
    """What's kept of a plan: no segments or full timeline, just each block's picks
    (video, scene, start) — enough to show it and to check the planner rebuilds it."""
    out = {k: plan.get(k) for k in ("mode", "variant", "style", "pace", "method", "settings", "videos", "story",
                                    "sequential", "scores", "estimate", "actors")}
    out["story_regions"] = story_regions if plan.get("story") else None
    out["picks"] = [[[c.get("video_id"), c.get("scene_id"), round(float(c.get("clip_start_sec") or 0), 3)]
                     for c in block.get("scenes") or []] for block in plan.get("timeline") or []]
    return out


def slim_results(results: list, story_regions) -> list:
    return [{"mode": r["mode"], "note": r.get("note", ""),
             "plan": slim_plan(r["plan"], story_regions) if r.get("plan") else None} for r in results]


def save_results(track_id: str, sequential: bool, data: dict) -> None:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _results_path(track_id, sequential).write_text(json.dumps(data))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Opening a plan in the planner
# ---------------------------------------------------------------------------

def open_plan(track_id: str, plan: dict, rank: int) -> None:
    settings = {"segmentation_method": plan["method"], "beats_per_bar": 4, **plan["settings"]}
    st.session_state["_open_recommended_plan"] = {
        "track_id": track_id, "settings": settings, "videos": list(plan["videos"]),
        "story": bool(plan.get("story")), "story_regions": plan.get("story_regions"),
        "sequential": bool(plan.get("sequential")),
        "label": f"Recommended plan {rank} for {track_id} ({REC.MODE_LABELS[plan['mode']]}) — {plan['variant']}, "
                 f"{plan['scores']['overall']:.0f}% overall"}
    st.switch_page("pages/1_Compilation_Planner.py")


# ---------------------------------------------------------------------------
# Drawing a plan
# ---------------------------------------------------------------------------

def _strip(scores: dict, which: str = "overall") -> go.Figure:
    """Per-block bars: "overall" (combined), "shape", or "story" (role blocks only: hit 100 / miss 0)."""
    vals = {"shape": scores.get("per_block_shape"), "story": scores.get("per_block_story")}.get(which) \
        or scores["per_block"]
    shown = [0.0 if v is None else v for v in vals]
    act = set(scores["action_blocks"]) if which == "overall" else set()
    colours = ["rgba(128,128,128,0.25)" if v is None else "#2e9e5b" if v >= 70 else "#e8a317" if v >= 50
               else "#d9534f" for v in vals]
    if which == "story":
        shown = [100.0 if v is None else v for v in vals]      # no-role blocks as faint full bars
        colours = ["rgba(128,128,128,0.18)" if v is None else "#2e9e5b" if v else "#d9534f" for v in vals]
    hover = "Block %{x}: %{customdata}<extra></extra>"
    custom = ["no story part" if v is None else (("✅ role clip" if v else "❌ not a role clip") if which == "story"
                                                 else f"{v:.0f}%") for v in vals]
    fig = go.Figure(go.Bar(x=list(range(1, len(vals) + 1)), y=shown, marker_color=colours, customdata=custom,
                           marker_line_color=["#222" if i in act else "rgba(0,0,0,0)" for i in range(len(vals))],
                           marker_line_width=[2 if i in act else 0 for i in range(len(vals))],
                           hovertemplate=hover))
    if which == "overall" and scores.get("weakest_block") is not None:
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


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0f}%"


def render_result(track_id: str, rank: int, result: dict) -> None:
    mode = result["mode"]
    plan = result.get("plan")
    with st.container(border=True):
        h = st.columns([2.4, 1])
        title = f"### {rank} · {REC.MODE_LABELS[mode]}"
        if not plan:
            h[0].markdown(title)
            h[0].caption(REC.MODE_HELP[mode])
            st.info(result.get("note") or "No plan found.")
            return
        sc = plan["scores"]
        # What each card leads with: 📐 shape, 🎭 story, the rest overall.
        lead = {"shape": ("shape", "shape"), "story": ("story", "story")}.get(mode, ("overall", "overall"))
        h[0].markdown(title)
        bits = [plan["variant"], f"{len(plan['videos'])} videos"]
        if plan.get("story"):
            bits.append("🎭 story map on")
        if plan.get("sequential"):
            bits.append("in order")
        h[0].caption(" · ".join(bits), help=REC.MODE_HELP[mode])
        open_now = h[1].button("🎬 Open", key=f"open_{track_id}_{mode}", type="primary", width="stretch")
        if mode == "verses" and plan.get("actors"):
            a, b = (", ".join(plan["actors"].get(v) or ["?"]) for v in plan["videos"][:2])
            st.markdown(f"**👤 {a}  vs  👤 {b}**")
        helps = {"overall": "⅓ shape + ⅓ motion + ⅓ story (½ + ½ when the track has no story parts).",
                 "shape": "How closely each clip's motion rises and falls with its block's music — every block "
                          "counts equally.",
                 "intensity": f"How well each clip's motion level matches its block's intensity — drops and "
                              f"action parts count {REC.DROP_WEIGHT:g}×.",
                 "story": "Story-map parts (🎬 Intro / 🌙 Slow / 💥 Action / 🏁 Outro) whose clip is marked with "
                          "that role. — = no story parts in this track."}
        names = {"overall": "Overall", "shape": "Shape", "intensity": "Motion", "story": "Story"}
        show = [lead[0]] if mode in ("shape", "story") else ["overall", "shape", "intensity", "story"]
        m = st.columns(len(show) if len(show) > 1 else 4)
        for col, k in zip(m, show):
            col.metric(names[k], _pct(sc.get(k)), help=helps[k])
        st.plotly_chart(_strip(sc, lead[0]), width="stretch", key=f"strip_{track_id}_{mode}",
                        config={"displayModeBar": False})
        cols = st.columns(min(len(plan["videos"]), 8) or 1)
        for c, vid in zip(cols * 2, plan["videos"]):
            with c:
                t = _thumb(vid)
                if t:
                    safe_image(str(t), width="stretch")
                st.caption(vid)
        if mode == "wildcard":
            st.button("🎲 Roll again", key=f"reroll_{track_id}", width="stretch",
                      help="A new random cutting style and random videos (just this card).",
                      on_click=lambda: st.session_state.__setitem__("_rec_reroll", True))
    if open_now:
        open_plan(track_id, plan, rank)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.caption("Five plans for a track, searched across your whole library and checked with the real matcher — "
           "the scores are what you get when you open one. **🎬 Open** makes a plan a new project in the planner "
           "(what you had there is saved first).")

tracks = list_tracks()
if not tracks:
    st.info("No audio tracks found.")
    st.stop()

c = st.columns([3, 2, 2])
default_track = st.session_state.get("track_id") if st.session_state.get("track_id") in tracks else tracks[0]
track_id = c[0].selectbox("🎵 Track", tracks, index=tracks.index(default_track), key="rec_track")
n_videos = c[1].slider("Videos per plan", 2, 15, DEFAULT_VIDEOS, key="rec_n_videos",
                       help="How many videos each plan uses (as in the planner's video selection). 🥊 Verses "
                            "always uses two.")
sequential = c[1].toggle("Sequential mode", key="rec_sequential",
                         help="Keep each video's clips in their original order (the planner's 'Keep each "
                              "video's clips in order'). Plans open with it on.")
track = load_track(track_id)
story_regions, story_source = story_for(track_id, track)
sig = signature(track_id, n_videos, sequential, story_regions)
cache_key = (track_id, bool(sequential))
cached = st.session_state.get("rec_results", {}).get(cache_key)
if not (cached and cached.get("sig") == sig):
    cached = load_results(track_id, sequential, sig)
go_btn = c[2].button("🔍 Find plans" if not cached else "🔄 Search again", type="primary", width="stretch")

if go_btn:
    bar = st.progress(0.0, text="Reading your library…")
    try:
        prints = library_prints(lambda f, t: bar.progress(0.2 * f, text=t))
        if len(prints) < 2:
            bar.empty()
            st.warning("Recommendations need at least two videos with motion data.")
            st.stop()
        seed = int(hashlib.sha1(f"{sig}{len(st.session_state.get('rec_results', {}))}".encode()).hexdigest()[:8], 16)
        from diagnostics import timed
        with timed("Recommendations: for this track", f"{len(prints)} videos"):
            results = REC.recommend_modes(
                track, prints, match=make_matcher(track), curves_of=CStore.motion_curves,
                n_videos=min(n_videos, len(prints)), story_regions=story_regions, role_seconds=role_seconds(),
                roles_of=scene_roles_lookup(), actors=actors_by_video(), similar_of=similar_finder(), sequential=sequential,
                wildcard_seed=seed, progress=lambda f, t: bar.progress(0.2 + 0.8 * min(f, 1.0), text=t))
    finally:
        bar.empty()
    cached = {"sig": sig, "track_id": track_id, "seed": seed, "results": slim_results(results, story_regions)}
    st.session_state.setdefault("rec_results", {})[cache_key] = cached
    save_results(track_id, sequential, cached)

if cached and st.session_state.pop("_rec_reroll", False):
    with st.spinner("🎲 Rolling again…"):
        prints = library_prints()
        seed = int(cached.get("seed", 0)) + 1
        res = REC.reroll_wildcard(track, prints, match=make_matcher(track), curves_of=CStore.motion_curves,
                                  n_videos=min(n_videos, len(prints)), story_regions=story_regions,
                                  roles_of=scene_roles_lookup(), sequential=sequential, seed=seed)
    cached = dict(cached, seed=seed, results=[slim_results([res], story_regions)[0] if r["mode"] == "wildcard"
                                              else r for r in cached["results"]])
    st.session_state.setdefault("rec_results", {})[cache_key] = cached
    save_results(track_id, sequential, cached)

if cached:
    if story_source == "saved":
        st.caption("🎭 Story fit uses this track's story map; its action parts count extra in the motion score.")
    elif story_source == "suggested":
        st.caption("🎭 This track has no saved story map, so story fit uses the one the story map wizard suggests "
                   "(Media Library → 🎭 Story map to make your own). Plans that use it open with it.")
    for i, result in enumerate(cached["results"], start=1):
        render_result(track_id, i, result)
else:
    st.info("Pick a track and press **🔍 Find plans**. The first search reads every video's motion, "
            "so it takes a little while; after that it's quicker.")
