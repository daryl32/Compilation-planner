"""🤖 The new matcher's own flow (Compilation Planner → 🤖 New matcher). No Audio Settings, story map or
hand-picked video step: the data does the work. Three ways in —
  🎵 By song   pick a track (finder + energy / tempo / length filters), optionally narrow the videos, compile.
  🎬 By video  pick one or more videos as the starting point; the app suggests songs that fit their footage and
               other suitable videos; pick a song, compile.
  ✍️ By text   a prompt → songs and videos (needs the vector search, Stage 7) — placeholder for now.
Cuts come from the track's 🧩 section map at each 🍳 recipe's pace (section_cuts.py); the plan from
core/story_matcher.py over every video with a 🧭 timeline (unless you narrow them). Split screen is off unless
turned on. The result page is slim: a strip of the plan, the clips, 🎬 Render · 💾 Save · ⬇️ Download, and
"open these videos in Classic".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

import catalogue_store as CS
import footage_timeline as FT
import library_common as LC
import recipes as RC
import section_cuts as SCUT
import section_types as ST
from config import AUDIO_DIR, PLANS_DIR, PREVIEW_DIR
from library_common import format_mmss

RESULT_KEY = "compose_result"
WAYS = ["🎵 By song", "🎬 By video", "✍️ By text"]
ENERGY_BANDS = {"Any": None, "Calm": (0.0, 0.33), "Medium": (0.33, 0.66), "Dynamic": (0.66, 1.01)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _audio_signature() -> tuple:
    try:
        return tuple(sorted((p.name, p.stat().st_mtime) for p in Path(AUDIO_DIR).glob("*.json")))
    except OSError:
        return ()


@st.cache_data(max_entries=2, show_spinner=False)
def audio_summaries(signature: tuple) -> dict:
    """track_id → duration, bpm, key, energy variation (0–1 within the library), whether it has sections."""
    out = {}
    for name, _ in signature:
        try:
            t = json.loads((Path(AUDIO_DIR) / name).read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(t, dict) or "duration_sec" not in t:
            continue
        hires = t.get("hires") or {}
        rms = [float(x) for x in hires["rms"]] if hires.get("rms") else \
            [float(e.get("energy", 0.0)) for e in t.get("energy_envelope", [])]
        mean = sum(rms) / len(rms) if rms else 0.0
        var = (sum((x - mean) ** 2 for x in rms) / len(rms)) ** 0.5 if rms else 0.0
        tid = t.get("track_id", Path(name).stem)
        out[tid] = {"track_id": tid, "duration": float(t["duration_sec"]), "bpm": float(t["bpm"]) if t.get("bpm") else None,
                    "key": t.get("key"), "variance": var, "analysed": bool((t.get("analysis") or {}).get("sections"))}
    if out:
        vs = sorted(v["variance"] for v in out.values())
        for v in out.values():
            v["energy_rank"] = vs.index(v["variance"]) / max(len(vs) - 1, 1)
    return out


def _load_track(track_id: str) -> dict:
    from planner.data import load_track
    return load_track(track_id)


def _videos_with_timeline() -> set:
    return set((FT.load_index().get("videos") or {}).keys())


def _video_rows() -> dict:
    """video_id → slim catalogue row (scenes with tags, duration) for every video."""
    return {v["video_id"]: v for v in CS.video_index().values()}


def _video_tags(v: dict) -> set:
    return {t for s in v.get("scenes") or [] if not s.get("excluded") for t in s.get("tags") or []}


# ---------------------------------------------------------------------------
# Video filters (shared by both ways)
# ---------------------------------------------------------------------------

def _video_filters(ns: str, rows: dict, have_tl: set, label: str) -> list:
    """The filter widgets; returns the matching video ids (only videos with a 🧭 timeline)."""
    details = LC.video_details()
    hair = LC.video_hair()
    cast = LC.video_cast()
    actors = LC.all_actors()
    tags = sorted({t for v in rows.values() for t in _video_tags(v)}, key=str.lower)
    with st.expander(label, expanded=False):
        c = st.columns([2, 2, 1.2, 1.2])
        pick_actors = c[0].multiselect("👤 Actors", actors, key=ns + "actors")
        pick_tags = c[1].multiselect("🏷️ Scene tags (any)", tags, key=ns + "tags")
        pick_hair = c[2].multiselect("💇 Hair", list(LC.HAIR_COLOURS), key=ns + "hair",
                                     format_func=lambda h: f"{LC.HAIR_EMOJI.get(h, '')} {h}")
        girl_girl = c[3].toggle("👭 Girl / girl", key=ns + "gg")
        d = st.columns([1.5, 1.5, 3])
        new_only = d[0].toggle("🆕 New videos only", key=ns + "new")
        min_min = d[1].number_input("At least (min)", 0, 120, 0, key=ns + "minlen", help="Usable length in minutes.")
        exclude = d[2].multiselect("🚫 Leave out", sorted(rows, key=str.lower), key=ns + "exclude")
    ranges = LC.library_ranges()
    added = CS.added_times()
    out = []
    for vid, v in rows.items():
        if vid not in have_tl or vid in exclude:
            continue
        if pick_actors and not LC.matches_details(vid, details, actors=pick_actors):
            continue
        if pick_tags and not (_video_tags(v) & set(pick_tags)):
            continue
        if pick_hair and not LC.matches_hair(vid, hair, wanted=pick_hair):
            continue
        if girl_girl and not LC.matches_cast(vid, cast, girl_girl=True):
            continue
        if new_only and not CS.is_new_video(added.get(vid, 0.0)):
            continue
        if min_min:
            rng = ranges.get(vid)
            usable = (rng[1] - rng[0]) if rng else float(v.get("duration") or 0.0)
            if usable < min_min * 60:
                continue
        out.append(vid)
    return sorted(out, key=str.lower)


# ---------------------------------------------------------------------------
# Suggesting songs and videos for a chosen set of videos (🎬 By video)
# ---------------------------------------------------------------------------

def _section_regions(track: dict) -> list:
    regions = ST.saved_map(track.get("track_id", ""))
    if regions:
        return regions
    try:
        return ST.suggested_map(track)
    except Exception:          # noqa: BLE001
        return []


@st.cache_data(max_entries=400, show_spinner=False)
def _regions_cached(track_id: str, audio_mtime: float, maps_mtime: float) -> list:
    """A track's section map (saved, else suggested) — the suggestion reads the whole song, so it's kept."""
    return _section_regions(_load_track(track_id))


def _maps_mtime() -> float:
    try:
        return ST._file().stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(max_entries=16, show_spinner=False, ttl=600)
def _type_fits(video_ids: tuple, tl_sig: tuple, music_story: float = 0.5) -> dict:
    """{type_id: fit} — how well these videos' footage suits each section type, worked out ONCE for the video set
    (movement band + trend + the recipe's tags; the song-specific parts — shape, beat — are left out of the
    shortlist). Each type's fit = the mean over the videos of their best tenth of windows."""
    import core.story_matcher as SMX
    index = FT.load_index()
    vids = []
    for vid in video_ids:
        tl = FT.load(vid, index=index)
        if tl is not None and len(tl):
            vids.append(SMX._Video(tl, index))
    out = {}
    if not vids:
        return out
    for t_id, recipe in RC.load_all().items():
        n = max(2, int(round(SCUT.pace_at(recipe, 0.5) / FT.STEP)))
        fits = []
        for v in vids:
            fit, _, _ = SMX._score_video(v, n, recipe, None, 0.0, music_story)
            if len(fit):
                k = max(1, len(fit) // 10)
                fits.append(float(np.sort(fit)[-k:].mean()))
        if fits:
            out[t_id] = round(float(np.mean(fits)), 3)
    return out


def track_fit(video_ids: list, track: dict, index: dict = None, music_story: float = 0.5, type_fits: dict = None) -> dict | None:
    """How well these videos suit this track: Σ (share of the track in a type) × (the videos' fit for that type)."""
    regions = _regions_cached(track["track_id"], _audio_mtime(track["track_id"]), _maps_mtime())
    if not regions:
        return None
    if type_fits is None:
        type_fits = _type_fits(tuple(video_ids), FT.signature(), music_story)
    if not type_fits:
        return None
    duration = float(track["duration_sec"])
    total, per_type = 0.0, {}
    for r in regions:
        t_id = r.get("type")
        f = type_fits.get(t_id)
        if f is None:
            continue
        share = (float(r["end"]) - float(r["start"])) / max(duration, 1e-9)
        total += share * f
        per_type[t_id] = f
    return {"fit": round(total, 3), "per_type": per_type}


def _audio_mtime(track_id: str) -> float:
    from track_files import track_file
    try:
        return track_file(track_id, AUDIO_DIR).stat().st_mtime
    except OSError:
        return 0.0


def suggest_tracks(video_ids: tuple, audio_sig: tuple, tl_sig: tuple, top: int = 6, _progress=None) -> list:
    """Not cached itself (it drives a progress bar): the heavy parts — the videos' per-type fits and each track's
    section map — are cached, so a repeat is quick."""
    type_fits = _type_fits(video_ids, tl_sig)
    summaries = audio_summaries(audio_sig)
    out = []
    for k, tid in enumerate(summaries, 1):
        if _progress:
            _progress(k, len(summaries), tid)
        try:
            track = _load_track(tid)
        except Exception:          # noqa: BLE001
            continue
        f = track_fit(list(video_ids), track, type_fits=type_fits)
        if f:
            out.append({"track_id": tid, **f})
    return sorted(out, key=lambda r: -r["fit"])[:top]


@st.cache_data(max_entries=8, show_spinner=False, ttl=600)
def suggest_videos(video_ids: tuple, tl_sig: tuple, top: int = 8) -> list:
    """Other videos that look like the chosen ones (mean embedding cosine) and share their main tags."""
    index = FT.load_index()
    have = list((index.get("videos") or {}).keys())
    chosen = [v for v in video_ids if v in have]
    if not chosen:
        return []

    def profile(vid):
        tl = FT.load(vid, index=index)
        if tl is None or len(tl) == 0:
            return None
        e = tl.embeddings().mean(axis=0)
        e = e / max(float(np.linalg.norm(e)), 1e-6)
        tags = {t for t, _ in FT.tag_time(tl)[:8]}
        return e, tags
    profs = {v: profile(v) for v in have}
    base = [profs[v] for v in chosen if profs.get(v)]
    if not base:
        return []
    mean_e = np.mean([b[0] for b in base], axis=0)
    base_tags = set().union(*[b[1] for b in base])
    out = []
    for vid in have:
        if vid in chosen or not profs.get(vid):
            continue
        e, tags = profs[vid]
        cos = float(e @ mean_e)
        jac = len(tags & base_tags) / max(len(tags | base_tags), 1)
        out.append({"video_id": vid, "score": round(0.7 * cos + 0.3 * jac, 3), "shared": sorted(tags & base_tags)})
    return sorted(out, key=lambda r: -r["score"])[:top]


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

def _motion_curves(video_id: str) -> dict:
    """{scene_id: {"fps", "values"}} — the catalogue's on-screen motion curves (for 🎯 Movement only)."""
    import catalogue_store as CS
    return CS.motion_curves(video_id)


def compile_plan(track_id: str, video_ids: list, *, split_screen: bool = False, music_story: float = 0.5,
                 smooth_contrast: float = 0.5, excluded: set = None, max_videos: int = 0, progress=None,
                 movement_only: bool = False, min_clip: float = None, level_range: float = 0.0) -> dict:
    import core.story_matcher as SMX
    from planner.data import build_footage_queues, load_all_catalogues
    from render_preview import colab_to_local
    track = _load_track(track_id)
    regions = _section_regions(track)
    all_recipes = RC.load_all()
    segments = SCUT.story_segments(track, regions, all_recipes, min_clip)
    queues = build_footage_queues(tuple(video_ids), (), LC.library_ranges())
    opts = SMX.StoryOptions(music_story=music_story, smooth_contrast=smooth_contrast, section_map=regions or None,
                            max_videos=int(max_videos or 0), progress=progress, merge_blocks=True,
                            movement_only=bool(movement_only), curves_of=_motion_curves,
                            level_range=float(level_range or 0.0))
    max_clips = 3 if split_screen else 1
    timeline, shortfall, unfilled = SMX.match_story(
        segments, queues, track, min_clips=1, max_clips=max_clips, ramp_start=0.5, ramp_end=0.9,
        global_excluded=set(excluded or ()), options=opts)
    catalogues = load_all_catalogues()
    for e in timeline:
        for slot in e["scenes"]:
            for link in slot["chain"]:
                link["source_path"] = colab_to_local(catalogues[link["video_id"]]["source_path"])
    plan = {
        "track_id": track_id, "matching_mode": "Auto story", "track_source_path": colab_to_local(track["source_path"]),
        "track_duration_sec": track["duration_sec"], "bpm": track.get("bpm"),
        "segmentation": {"method": "sections", "section_map": regions, "paces": {t: r.get("pace") for t, r in all_recipes.items()}},
        "sequential_video_order": False, "globally_rejected_clips": [list(k) for k in (excluded or ())],
        "video_time_ranges": {k: list(v) for k, v in LC.library_ranges().items() if k in video_ids},
        "tag_filter": [], "selected_videos": list(video_ids), "split_screen_enabled": split_screen,
        "min_clips": 1, "max_clips": max_clips, "ramp_start": 0.5, "ramp_end": 0.9, "density_contrast": 1.0,
        "vary_clip_count": False, "clip_count_seed": 0, "min_clip_len_sec": 1.0, "allow_same_video": False,
        "story": {"music_story": music_story, "smooth_contrast": smooth_contrast, "max_videos": int(max_videos or 0),
                  "movement_only": bool(movement_only), "level_range": float(level_range or 0.0)},
        "timeline": timeline,
    }
    return {"plan": plan, "segments": segments, "notes": list(opts.notes), "shortfall": shortfall,
            "unfilled": unfilled, "summary": SMX.plan_summary(timeline), "track": track,
            "video_stats": dict(opts.video_stats),
            "made": time.strftime("%H:%M:%S")}


def _run_compile(track_id: str, video_ids: list, ns: str) -> None:
    """The button's callback only queues the compile — the page body runs it, so its progress bar shows."""
    if not video_ids:
        st.session_state["compose_msg"] = ("warning", "No videos to match — widen the video filters.")
        return
    st.session_state["compose_pending"] = {"track_id": track_id, "video_ids": list(video_ids), "way": ns}


def _do_pending_compile() -> None:
    job = st.session_state.pop("compose_pending", None)
    if not job:
        return
    bar = st.progress(0.0, text="🤖 Compiling… reading the song's sections")
    try:
        res = compile_plan(job["track_id"], job["video_ids"], split_screen=bool(st.session_state.get("compose_split")),
                           music_story=float(st.session_state.get("compose_music_story", 0.5)),
                           smooth_contrast=float(st.session_state.get("compose_smooth", 0.5)),
                           excluded=set(st.session_state.get("compose_excluded") or ()),
                           max_videos=int(st.session_state.get("compose_max_videos", 0) or 0),
                           movement_only=bool(st.session_state.get("compose_movement_only")),
                           min_clip=_min_clip(),
                           level_range=float(st.session_state.get("compose_level_range", 0.0)),
                           progress=lambda d, t: bar.progress(min(d / max(t, 1), 1.0), text=f"🤖 Compiling… block {d} of {t}"))
    except Exception as e:          # noqa: BLE001
        bar.empty()
        st.session_state["compose_msg"] = ("error", f"Couldn't compile: {e}")
        return
    bar.empty()
    res["video_ids"] = job["video_ids"]
    res["way"] = job["way"]
    st.session_state[RESULT_KEY] = res
    st.session_state.pop("compose_render", None)


def _min_clip():
    """✂️ Shortest clip as the cuts want it: None = Auto (from the tempo), else seconds."""
    v = float(st.session_state.get("compose_min_clip", 0.0) or 0.0)
    return None if v <= 0 else v


def _options_row(ns: str) -> None:
    c = st.columns([1.3, 1.6, 2, 2])
    c[0].toggle("🪟 Split screen", key="compose_split", help="Off: one clip per block. On: up to three at the loudest blocks.")
    c[1].number_input("Max different videos", 0, 60, 0, key="compose_max_videos",
                      help="0 = no limit. A limit keeps the plan to fewer source videos — quicker renders, and fewer "
                           "files to fetch. Once the budget is spent, later blocks stay with the videos already in.")
    only = bool(st.session_state.get("compose_movement_only"))
    c[2].slider("Music ↔ Story", 0.0, 1.0, 0.5, 0.05, key="compose_music_story", disabled=only,
                help="Left: the music fit decides. Right: the recipes' tags decide.")
    c[3].slider("Smooth ↔ Contrast", 0.0, 1.0, 0.5, 0.05, key="compose_smooth",
                help="Left: shot sizes kept steady. Right: changes welcome at drops and section changes.")
    st.number_input("✂️ Shortest clip (s)", 0.0, 10.0, 0.0, 0.5, key="compose_min_clip",
                    help="0 = Auto: the shortest musical length of at least 1.4 s — a bar at 90 BPM (2.7 s), a bar at "
                         "128 BPM (1.9 s). Or type your own; cuts stay on the beat, so it rounds up to whole beats. "
                         "0.5 = no minimum (the recipes' clip lengths alone). A section too short for two clips "
                         "stays one clip.")
    st.slider("🎚️ Use the track's full range", 0.0, 1.0, 0.0, 0.25, key="compose_level_range",
              disabled=not bool(st.session_state.get("compose_movement_only")),
              help="🎯 Movement only. How far each track's loudness is stretched over the whole calm-to-lively range. "
                   "0: as measured — the loud parts of most songs all ask for nearly the same footage, and the "
                   "liveliest footage is never asked for. 1: the track's calmest part asks for calm footage and its "
                   "loudest for the liveliest, so a soft song's chorus is as lively as a heavy song's. In between, "
                   "soft songs stay gentler. The order of the parts never changes. Extra punch for powerful songs; "
                   "on a gentle one (Flowers at 0.5) it asked for footage too lively for the music.")
    st.toggle("🎯 Movement only", key="compose_movement_only",
              help="No recipes: every clip is picked on how its movement follows the music (frame by frame, against "
                   "the drums, bass or voice — whichever carries the bar) and how lively it is against how loud the "
                   "music is. Bouncy up-and-down footage goes where the music has a beat, smooth footage where a "
                   "voice or pad carries it. Tags, scene tags and roles are ignored. The cuts still come from the "
                   "sections.")


# ---------------------------------------------------------------------------
# 🎵 By song
# ---------------------------------------------------------------------------

def _track_finder(ns: str, summaries: dict) -> str | None:
    if not summaries:
        st.info("No analysed tracks yet — 📁 Media Library → 🎵 Audio → 🔬 Audio analysis.")
        return None
    with st.container(border=True):
        c = st.columns([3, 2, 2, 1.6])
        q = c[0].text_input("Find a track", placeholder="part of a name", key=ns + "q")
        bpms = [s["bpm"] for s in summaries.values() if s["bpm"]]
        lo_b, hi_b = (int(min(bpms)) - 1, int(max(bpms)) + 1) if bpms else (60, 200)
        bpm_rng = c[1].slider("Tempo (BPM)", lo_b, hi_b, (lo_b, hi_b), key=ns + "bpm")
        durs = [s["duration"] for s in summaries.values()]
        hi_m = int(max(durs) // 60) + 1 if durs else 10
        len_rng = c[2].slider("Length (min)", 0, hi_m, (0, hi_m), key=ns + "len")
        energy = c[3].selectbox("Energy", list(ENERGY_BANDS), key=ns + "energy",
                                help="How much the track's loudness moves — calm, medium or dynamic within your library.")
        d = st.columns([2, 2, 3])
        sort = d[0].selectbox("Sort by", ["Name", "Tempo", "Length", "Energy"], key=ns + "sort")
        only_mapped = d[1].toggle("🧩 Has a saved section map", key=ns + "mapped")
        saved = ST.saved_maps()
    rows = []
    for s in summaries.values():
        if q and q.lower() not in s["track_id"].lower():
            continue
        if s["bpm"] and not (bpm_rng[0] <= s["bpm"] <= bpm_rng[1]):
            continue
        if not (len_rng[0] * 60 <= s["duration"] <= len_rng[1] * 60 + 59):
            continue
        band = ENERGY_BANDS[energy]
        if band and not (band[0] <= s.get("energy_rank", 0.5) < band[1]):
            continue
        if only_mapped and s["track_id"] not in saved:
            continue
        rows.append(s)
    keyf = {"Name": lambda s: s["track_id"].lower(), "Tempo": lambda s: -(s["bpm"] or 0), "Length": lambda s: -s["duration"],
            "Energy": lambda s: -s.get("energy_rank", 0)}[sort]
    rows.sort(key=keyf)
    if not rows:
        st.caption("No track matches these filters.")
        return None

    def fmt(tid):
        s = summaries[tid]
        return (f"{'🧩 ' if tid in saved else ''}{tid} · {format_mmss(s['duration'])}"
                + (f" · {s['bpm']:.0f} BPM" if s["bpm"] else "") + (f" · {s['key']}" if s.get("key") else ""))
    d[2].caption(f"{len(rows)} track{'s' if len(rows) != 1 else ''} match")
    return st.selectbox("Track", [s["track_id"] for s in rows], key=ns + "track", format_func=fmt)


def _track_line(track_id: str) -> None:
    try:
        track = _load_track(track_id)
    except Exception as e:          # noqa: BLE001
        st.error(f"Couldn't read {track_id}: {e}")
        return
    regions = _section_regions(track)
    if not regions:
        st.warning("This track's sections can't be read — it needs the 🔬 audio analysis first.")
        return
    segs = SCUT.story_segments(track, regions, RC.load_all(), _min_clip())
    tot = ST.totals(regions)
    st.caption("🧩 " + " · ".join(f"{ST.label(t)} {format_mmss(s)}" for t, s in sorted(tot.items(), key=lambda kv: -kv[1]))
               + f"  →  ✂️ {SCUT.describe(segs)}"
               + ("" if ST.saved_map(track_id) else "  · suggested map (save one in 📁 Media Library → 🎵 Audio → 🧩 Sections)"))


def render_by_song(rows: dict, have_tl: set) -> None:
    ns = "cs_"
    summaries = audio_summaries(_audio_signature())
    tid = _track_finder(ns, summaries)
    if not tid:
        return
    _track_line(tid)
    vids = _video_filters(ns, rows, have_tl, f"🎬 Which videos — optional (all {len(have_tl)} with a timeline unless you narrow it)")
    ranges = LC.library_ranges()
    mins = sum((ranges[v][1] - ranges[v][0]) if v in ranges else float(rows[v].get("duration") or 0) for v in vids) / 60
    st.caption(f"🎬 {len(vids)} video{'s' if len(vids) != 1 else ''} · {mins:.0f} min of footage in play")
    _options_row(ns)
    st.button("🤖 Compile", key=ns + "go", type="primary", width="stretch", on_click=_run_compile, args=(tid, vids, "song"))


# ---------------------------------------------------------------------------
# 🎬 By video
# ---------------------------------------------------------------------------

def render_by_video(rows: dict, have_tl: set) -> None:
    ns = "cv_"
    vids = _video_filters(ns, rows, have_tl, "🔎 Narrow the list first (optional)")
    start = st.multiselect("Start from these videos", vids, key=ns + "start",
                           help="One or more. Songs and other videos are suggested to go with them.")
    if not start:
        st.caption("Pick at least one video.")
        return
    masters = LC.master_thumbnail_scenes()
    cols = st.columns(min(len(start), 6))
    for col, vid in zip(cols, start[:6]):
        p = LC.master_thumbnail_path(vid, rows[vid].get("scenes") or [], masters)
        if p and LC.image_ok(p):
            col.image(str(p), caption=vid, width="stretch")
        else:
            col.caption(vid)
    tl_sig = FT.signature()
    st.markdown("**Stage 1 — 🎵 songs that suit this footage**")
    if st.button("🎵 Find songs", key=ns + "find", type="primary", width="stretch"):
        st.session_state[ns + "songs_for"] = tuple(start)
    if st.session_state.get(ns + "songs_for") != tuple(start):
        st.caption("Press 🎵 Find songs — every track is scored against the chosen footage (a few seconds the first "
                   "time a track's sections are read; quick after that).")
        return
    bar = st.progress(0.0, text="Scoring songs…")
    tracks = suggest_tracks(tuple(start), _audio_signature(), tl_sig,
                            _progress=lambda k, n, t: bar.progress(k / max(n, 1), text=f"Scoring songs… {k}/{n} · {t}"))
    bar.empty()
    if not tracks:
        st.warning("No song could be scored — the chosen videos need a 🧭 timeline and the tracks their sections.")
        return
    tid = st.radio("Song", [t["track_id"] for t in tracks], key=ns + "track", label_visibility="collapsed",
                   format_func=lambda t: f"{t} · fit {next(x['fit'] for x in tracks if x['track_id'] == t):.2f}  —  "
                                         + ", ".join(f"{ST.label(k)} {v:.2f}" for k, v in sorted(
                                             next(x['per_type'] for x in tracks if x['track_id'] == t).items(), key=lambda kv: -kv[1])[:3]))
    _track_line(tid)
    extra = []
    st.markdown("**Stage 2 — 🎬 other videos that go with them** (optional)")
    if st.button("🎬 Suggest other videos", key=ns + "more", width="stretch",
                 help="Videos that look like the chosen ones and share their main tags. Skip it to match just your videos."):
        st.session_state[ns + "others_for"] = tuple(start)
    if st.session_state.get(ns + "others_for") == tuple(start):
        with st.spinner("Looking for videos like these…"):
            others = suggest_videos(tuple(start), tl_sig)
        if not others:
            st.caption("Nothing close enough found.")
        ocols = st.columns(min(len(others), 4)) if others else []
        for k, o in enumerate(others):
            with ocols[k % len(ocols)]:
                p = LC.master_thumbnail_path(o["video_id"], rows.get(o["video_id"], {}).get("scenes") or [], masters)
                if p and LC.image_ok(p):
                    st.image(str(p), width="stretch")
                if st.checkbox(f"{o['video_id']} · {o['score']:.2f}", value=True, key=f"{ns}other_{o['video_id']}",
                               help="Shares: " + (", ".join(o["shared"]) or "—")):
                    extra.append(o["video_id"])
    chosen = list(dict.fromkeys(start + extra))
    st.caption(f"🎬 {len(chosen)} video{'s' if len(chosen) != 1 else ''} go into the mix")
    _options_row(ns)
    st.button("🤖 Compile", key=ns + "go", type="primary", width="stretch", on_click=_run_compile, args=(tid, chosen, "video"))


# ---------------------------------------------------------------------------
# ✍️ By text
# ---------------------------------------------------------------------------

def render_by_text() -> None:
    st.text_input("Describe the mix", placeholder="e.g. slow sensual blonde POV, building to a hard drop",
                  key="ct_prompt", disabled=True)
    st.info("✍️ Coming with the vector search (Stage 7): the prompt finds the songs and the footage, then compiles "
            "the same way. Until then, 🎵 By song and 🎬 By video.")


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

def _strip_chart(res: dict) -> None:
    import plotly.graph_objects as go
    plan = res["plan"]
    colours = {t["id"]: t["colour"] for t in ST.TYPES}
    fig = go.Figure()
    for e in plan["timeline"]:
        a, b = e["track_time"]
        t = e.get("section_type")
        info = (e["scenes"][0].get("story") or {}) if e["scenes"] else {}
        fit = info.get("fit")
        label = " / ".join(s["video_id"] for s in e["scenes"]) or "—"
        fig.add_trace(go.Bar(x=[b - a], base=[a], y=["plan"], orientation="h", width=0.8,
                             marker=dict(color=colours.get(t, "#bbbbbb"), opacity=0.35 + 0.6 * (fit if fit is not None else 0.3),
                                         line=dict(color="rgba(0,0,0,0.25)", width=0.5)),
                             showlegend=False,
                             hovertemplate=f"{ST.label(t)} · {format_mmss(a)}–{format_mmss(b)}<br>{label}"
                                           + (f"<br>fit {fit:.2f}" if fit is not None else "<br>Classic's pick") + "<extra></extra>"))
    fig.update_layout(barmode="overlay", height=90, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(visible=False),
                      xaxis=dict(range=[0, float(plan["track_duration_sec"])], tickfont_size=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"compose_strip_{res['made']}")


def _save(res: dict) -> None:
    from page_setup import push_file_with_oauth, OAUTH_AVAILABLE, DRIVE_SESSION_KEY
    plan = res["plan"]
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(PLANS_DIR) / f"{plan['track_id']}_plan.json"
    out.write_text(json.dumps(plan, indent=2))
    if OAUTH_AVAILABLE and st.session_state.get(DRIVE_SESSION_KEY):
        err = push_file_with_oauth(st.session_state[DRIVE_SESSION_KEY], out, "scene-labeling/compilation_plans")
        st.session_state["compose_msg"] = ("warning", f"Saved locally but Drive upload failed: {err}") if err else \
            ("success", f"Saved and uploaded to Drive: {out.name}")
    else:
        st.session_state["compose_msg"] = ("success", f"Saved locally: {out.name}")


def _missing_sources(plan: dict, quick: bool = False) -> list:
    """(raw source, local path) for the track and every video the plan uses that isn't on this machine. quick:
    a video with a 480p preview copy here isn't missing (⚡ Quick preview reads the copy)."""
    from planner.data import load_all_catalogues
    from render_preview import colab_to_local
    cats = load_all_catalogues()
    try:
        needed = {_load_track(plan["track_id"])["source_path"]}      # the raw Drive path, not the local one
    except Exception:          # noqa: BLE001
        needed = {plan["track_source_path"]}
    for e in plan["timeline"]:
        for slot in e["scenes"]:
            for link in slot["chain"]:
                if quick and LC.proxy_path(link["video_id"]).exists():
                    continue
                needed.add(cats[link["video_id"]]["source_path"])
    out = []
    for raw in sorted(needed):
        local = Path(colab_to_local(raw))
        if not local.exists():
            out.append((raw, local))
    return out


def _download_sources(missing: list) -> list:
    """Download the missing files from Google Drive (the ⚙️ 'download first' way). Returns error strings."""
    try:
        from drive_sync import download_source_video, credentials_available
        ok = credentials_available()
    except ImportError:
        ok = False
    errors = []
    for n, (raw, local) in enumerate(missing, 1):
        if not (ok and "MyDrive/" in raw):
            errors.append(f"{local.name}: not on the server and Drive download isn't available")
            continue
        bar = st.progress(0.0, text=f"Downloading {local.name} ({n}/{len(missing)})…")
        err = download_source_video(raw.split("MyDrive/", 1)[1], local,
                                    progress_callback=lambda p, _b=bar, _n=local.name, _i=n: _b.progress(
                                        min(p, 1.0), text=f"Downloading {_n} ({_i}/{len(missing)})…"))
        bar.empty()
        if err:
            errors.append(f"{local.name}: {err}")
    return errors


def _do_pending_render(res: dict) -> None:
    """The 🎬 button only queues the render; the page body runs it here so the progress bar shows."""
    if not st.session_state.pop("compose_do_render", False):
        return
    from page_setup import drive_token, APP_VERSION
    from render_preview import render_plan_dict
    from renders import upload_render, clear_local, render_files, auto_clear_enabled
    from planner.state import sanitize_filename
    import fnmatch, re
    plan = res["plan"]
    stream = bool(st.session_state.get("compose_stream", True))
    quick = bool(st.session_state.get("compose_quick", True))
    if not stream:
        missing = _missing_sources(plan, quick)
        if missing:
            errs = _download_sources(missing)
            if errs:
                st.error("Can't render — some source files are missing and couldn't be downloaded:\n\n"
                         + "\n".join(f"- {e}" for e in errs))
                return
    safe = sanitize_filename(plan["track_id"])
    used = [int(m.group(1)) for p in render_files() if (m := re.search(r"\.(\d{3})\.mp4$", p.name))
            and fnmatch.fnmatch(p.name, f"temp-Video-preview-{safe}.*.mp4")]
    Path(PREVIEW_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(PREVIEW_DIR) / f"temp-Video-preview-{safe}.{(max(used) + 1) if used else 1:03d}.mp4"
    bar = st.progress(0.0, text="Rendering…")
    last = [0.0]
    t0 = time.time()

    def _prog(d, t):
        last[0] = d / max(t, 1)
        secs = time.time() - t0
        eta = f" · about {int((secs / d) * (t - d))} s left" if d else ""
        bar.progress(last[0], text=f"Rendering segment {d}/{t}{eta}")

    try:
        stats = render_plan_dict(plan, str(out), progress_callback=_prog, stream=stream, quick=quick,
                                 notice=lambda txt: bar.progress(last[0], text=txt), app_version=APP_VERSION,
                                 extra_meta={"project_name": f"🤖 {plan['track_id']}"}) or {}
        secs = time.time() - t0
        how = [f"{stats[k]} {w}" for k, w in (("quick", "from the 480p copies"), ("streamed", "streamed from Drive"),
                                             ("downloaded", "downloaded"),
                                             ("local", "already on the server")) if stats.get(k)]
        token = drive_token()
        note = ("info", f"Rendered {out.name} in {int(secs // 60)}:{int(secs % 60):02d}. Connect Google Drive (sidebar) to save renders there.")
        if token:
            bar.progress(1.0, text="Uploading to Google Drive…")
            err = upload_render(out, token)
            if err:
                note = ("warning", f"Rendered, but the Drive upload failed: {err}")
            elif auto_clear_enabled():
                clear_local(out)
                note = ("success", f"Rendered {out.name} in {int(secs // 60)}:{int(secs % 60):02d}, saved to Google Drive and cleared from the server.")
            else:
                note = ("success", f"Rendered {out.name} in {int(secs // 60)}:{int(secs % 60):02d} and saved to Google Drive.")
        if how:
            note = (note[0], note[1] + " Source files: " + ", ".join(how) + ".")
        bar.empty()
        st.session_state["compose_msg"] = note
        st.session_state["compose_render"] = out.name
        st.rerun()
    except Exception as e:          # noqa: BLE001
        bar.empty()
        st.error(f"Render failed: {e}")


def _queue_render() -> None:
    st.session_state["compose_do_render"] = True


def _open_in_classic(res: dict) -> None:
    plan = res["plan"]
    st.session_state["track_id"] = plan["track_id"]
    st.session_state["committed_selected_videos"] = list(plan["selected_videos"])
    for v in plan["selected_videos"]:
        st.session_state[f"select_video_{v}"] = True
    st.session_state["matching_mode"] = "Auto"
    st.session_state["matcher_engine"] = "🤖 Auto story"
    st.session_state["planner_mode"] = "🎛️ Classic"
    st.session_state["ui_section"] = "🎬 3. Matching & Export"
    st.session_state["compose_msg"] = ("info", "Opened in Classic with the same track, videos and the 🤖 Auto story engine — "
                                               "Classic cuts the track its own way, so the plan is built again there.")


def _toggle_reject(key: tuple) -> None:
    ex = set(st.session_state.get("compose_excluded") or ())
    ex.symmetric_difference_update({key})
    st.session_state["compose_excluded"] = ex


def render_result(res: dict) -> None:
    plan, summary = res["plan"], res["summary"]
    _do_pending_render(res)
    c = st.columns([5, 1.4])
    c[0].subheader(f"🤖 {plan['track_id']}")
    c[1].button("← Change", key="compose_back", width="stretch", on_click=lambda: st.session_state.pop(RESULT_KEY, None))
    types = " · ".join(f"{ST.label(t)} {n}" for t, n in sorted(summary["types"].items(), key=lambda kv: -kv[1]))
    st.caption(f"{len(plan['timeline'])} blocks · {summary.get('videos', 0)} different videos used "
               f"(of {len(plan['selected_videos'])} in play) · {types}"
               + (f" · mean fit {summary['mean_fit']:.2f}" if summary.get("mean_fit") is not None else "")
               + (f" · 🎯 movement sync {summary['sync']:.0%}" if summary.get("sync") is not None else "")
               + (f" · clips {summary['clip_sec'][0]:g}–{summary['clip_sec'][2]:g} s (typically {summary['clip_sec'][1]:g})"
                  if summary.get("clip_sec") else "")
               + (f" · {summary['merged']} made longer" if summary.get("merged") else "")
               + (f" · {summary['joined']} across clip edges" if summary.get("joined") else "")
               + (f" · {summary['split']} split on the beat" if summary.get("split") else "")
               + (f" · {summary['fallback']} Classic pick(s)" if summary.get("fallback") else "")
               + (f" · ⚠️ {res['shortfall']} slot(s) unfilled" if res.get("shortfall") else ""))
    if summary.get("drivers"):
        import audio_drive as AD
        tot = sum(summary["drivers"].values())
        st.caption("What drives the music, by block: " + " · ".join(
            f"{AD.EMOJI.get(k, '')} {k} {round(100 * v / tot)}%" for k, v in sorted(summary["drivers"].items(), key=lambda kv: -kv[1])))
    for n in res.get("notes") or []:
        st.caption("🤖 " + n)
    _strip_chart(res)
    excluded = set(st.session_state.get("compose_excluded") or ())
    with st.expander(f"🎞️ Clips ({sum(len(e['scenes']) for e in plan['timeline'])})", expanded=False):
        st.caption("🚫 a clip you don't want, then ↻ Recompile — it's left out of the whole plan.")
        from planner.data import get_nearest_thumbnail, load_all_catalogues
        cats = load_all_catalogues()
        for e in plan["timeline"]:
            a, b = e["track_time"]
            for s in e["scenes"]:
                info = s.get("story") or {}
                key = (s["video_id"], s["scene_id"])
                r = st.columns([1.2, 1.4, 3.5, 2.2, 0.7], vertical_alignment="center")
                thumb = get_nearest_thumbnail(cats, s["video_id"], s.get("clip_start_sec", 0.0))
                with r[0]:
                    if thumb is not None:
                        LC.safe_image(thumb, width=90)          # a PIL image cropped from the sprite sheet
                    elif s.get("thumbnail") and Path(s["thumbnail"]).exists():
                        LC.safe_image(s["thumbnail"], width=90)
                import audio_drive as AD
                r[1].markdown(f"**{format_mmss(a)}–{format_mmss(b)}**  \n{ST.label(info.get('type'))} "
                              f"{AD.EMOJI.get(e.get('driver'), '')}")
                r[2].markdown(f"`{s['video_id']}` #{s['scene_id']} at {format_mmss(s['clip_start_sec'])} · {s['clip_duration_sec']:.1f} s"
                              + (f"  \n{', '.join(s.get('tags') or [])}" if s.get("tags") else ""))
                how = (f" · split in {info['parts']}" if info.get("parts") else
                       f" · {info['joined']} clips back to back" if info.get("joined") else "")
                sync_bit = (f" · 🎯 {info['sync']:.0%}" + (f" (moved {info['slide']:+.2f} s)" if info.get("slide") else "")
                            if info.get("sync") is not None else "")
                r[3].caption(f"fit {info['fit']:.2f} (music {info.get('base', 0):.2f} · story {info.get('story', 0):.2f})"
                             f"{sync_bit}{how}" if info.get("fit") is not None else "Classic's pick")
                r[4].button("✅" if key in excluded else "🚫", key=f"compose_rej_{e['segment_index']}_{s['video_id']}_{s['scene_id']}",
                            help="Put back" if key in excluded else "Leave this clip out", on_click=_toggle_reject, args=(key,))
    b = st.columns([1.6, 0.5, 1.4, 1.4, 1.4, 1.6])
    b[0].button("🎬 Render preview", key="compose_render_btn", type="primary", width="stretch", on_click=_queue_render)
    with b[1].popover("⚙️", width="stretch", help="Render options"):
        st.toggle("⚡ Quick preview — use the 480p copies on the server", value=True, key="compose_quick",
                  help="On: every video with a preview copy here is read from it — nothing is fetched, so the render "
                       "starts at once however many videos the plan uses. The preview is 640×360, so nothing is lost. "
                       "Off: the original files (streamed or downloaded, as set below).")
        st.toggle("Stream original videos from Google Drive", value=True, key="compose_stream",
                  help="On: only the seconds each clip uses are fetched from Drive while rendering — nothing stored. "
                       "Off: every missing source file is downloaded to the server first (slower the first time, "
                       "faster for every render after). Fewer different videos (the limit above) = less to fetch.")
        missing = _missing_sources(plan, bool(st.session_state.get("compose_quick", True)))
        st.caption(f"{len(missing)} source file{'s' if len(missing) != 1 else ''} not on the server yet"
                   if missing else "Every source file is on the server — renders are quick.")
    b[2].button("💾 Save plan", key="compose_save", width="stretch", on_click=_save, args=(res,))
    b[3].download_button("⬇️ Download", data=json.dumps(plan, indent=2), file_name=f"{plan['track_id']}_plan.json",
                         mime="application/json", width="stretch", key="compose_dl")
    b[4].button("↻ Recompile", key="compose_again", width="stretch",
                help="Build the plan again — without the 🚫 clips, with the current Split screen / slider settings.",
                on_click=_run_compile, args=(plan["track_id"], res["video_ids"], res.get("way", "song")))
    b[5].button("🎛️ Open in Classic", key="compose_classic", width="stretch", on_click=_open_in_classic, args=(res,))
    if st.toggle("📥 Sync data", key="compose_sync_data",
                 help="A file with every clip's movement next to its music, frame by frame — send it to Claude to tune "
                      "the movement matching on real numbers."):
        try:
            import core.story_matcher as SMX
            rep = SMX.sync_report(plan["timeline"], res["track"], video_stats=res.get("video_stats"))
            st.download_button(f"⬇️ {plan['track_id']}_sync.json"
                               + (f" — as played {rep['sync_played_mean']:.0%}" if rep.get("sync_played_mean") is not None else ""),
                               data=json.dumps(rep, separators=(",", ":")), file_name=f"{plan['track_id']}_sync.json",
                               mime="application/json", key="compose_sync_dl")
        except Exception as e:          # noqa: BLE001
            st.caption(f"Couldn't put the sync data together: {e}")
    if name := st.session_state.get("compose_render"):
        from library_common import show_render_player
        from renders import load_render
        meta = load_render(Path(PREVIEW_DIR) / name)
        if meta:
            show_render_player(meta)
        else:
            st.caption(f"{name} — see 📁 Media Library → 🎞️ Renders.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    msg = st.session_state.pop("compose_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    _do_pending_compile()
    res = st.session_state.get(RESULT_KEY)
    if res:
        render_result(res)
        return
    st.caption("🤖 The data has done the hard work: any song can meet any clip in the library. Pick a song or some "
               "videos, narrow things if you like, and compile — the track's 🧩 sections and 🍳 recipes decide the "
               "cuts and the footage. Classic is still there for a fully hand-tuned mix.")
    with st.container(key="subnav_compose"):
        way = st.radio("Way", WAYS, horizontal=True, key="compose_way", label_visibility="collapsed")
    rows = _video_rows()
    have_tl = _videos_with_timeline() & set(rows)
    if not have_tl and way != WAYS[2]:
        st.warning("No video has a 🧭 footage timeline yet — build it in 📁 Media Library → 🎬 Videos → 🧭 Timeline first.")
        return
    if way == WAYS[0]:
        render_by_song(rows, have_tl)
    elif way == WAYS[1]:
        render_by_video(rows, have_tl)
    else:
        render_by_text()
