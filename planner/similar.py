"""✨ More like this — similar videos and clips, video ranking and the video list's thumbnails.

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

import json
import os
import random
import numpy as np
import streamlit as st
import video_ranking as VR
import similarity as SIM
import scene_roles as SR
from library_common import (safe_image, shared_cache)
from core.matching import format_mmss
from core.shape import shape_score
from core.footage import (_tc_to_seconds, _overlap_with_range, get_candidate_spans, limit_queue_before, carve_span)
from diagnostics import timed as _timed
from config import CATALOGUE_DIR, AUDIO_DIR, PREVIEW_DIR, PROJECTS_DIR

from planner.state import (
    rerun_fragment
)
from planner.data import (
    build_footage_queues, effective_time_ranges, get_nearest_thumbnail, load_all_catalogues,
    load_motion_curves_for_video
)
from planner.story import (
    current_story_map
)
# ---------------------------------------------------------------------------
# ✨ More like this — similar videos, similar clips, swap for similar
# (similarity.py does the scoring: tags + movement, no AI model)
# ---------------------------------------------------------------------------


def _clip_curve(video_id: str, scene_id: int, offset_sec: float, duration_sec: float):
    """(values, fps) of a clip's slice of its scene's motion curve, or (None, None)."""
    curve = load_motion_curves_for_video(video_id).get(scene_id)
    if not curve or curve.get("values") is None or len(curve["values"]) == 0:
        return None, None
    fps = float(curve["fps"])
    vals = np.asarray(curve["values"], dtype=float)
    a = int(round(offset_sec * fps))
    b = a + max(1, int(round(duration_sec * fps)))
    return np.array(vals[a:b]), fps      # own copy — not tied to the curve cache's file


@shared_cache
@st.cache_resource(show_spinner=False)
def library_tag_idf() -> dict:
    """{tag: weight} across the whole library — computed once, not per search."""
    return SIM.tag_idf(load_all_catalogues(), lambda sc: sc.get("tags", []))


def find_similar_clips(ref: dict, spans: list, duration: float, exclude=None, top: int = 6,
                       max_per_video: int = 2):
    """Clips of exactly `duration` from `spans` (available footage) most like
    ref = {"video_id", "scene_id", "offset", "tags"}. Windows step by half a
    clip (coarser on very large pools, to stay quick); results never overlap
    each other and come at most max_per_video from one video. Returns
    [(similarity, reason, (span, offset, values, fps))], or None if the
    reference clip has no motion curve to compare."""
    ref_vals, ref_fps = _clip_curve(ref["video_id"], ref["scene_id"], ref["offset"], duration)
    if ref_vals is None or ref_vals.size == 0:
        return None
    total = sum(sp["remaining_sec"] for sp in spans)
    step = max(0.5, duration / 2, total / 4000.0)
    pool = []
    for span in spans:
        off = span["offset_sec"]
        end = span["offset_sec"] + span["remaining_sec"]
        while off + duration <= end + 1e-6:
            if not (exclude and exclude(span, off)):
                vals, fps = _clip_curve(span["video_id"], span["scene_id"], off, duration)
                if vals is not None and vals.size:
                    pool.append((SIM.clip_descriptor(vals, fps, span.get("tags", [])), (span, off, vals, fps)))
            off += step
    idf = library_tag_idf()
    ranked = SIM.rank_similar_clips(SIM.clip_descriptor(ref_vals, ref_fps, ref.get("tags", [])), pool, idf,
                                    top=len(pool))
    # Keep the results varied: no two overlapping clips, and at most
    # max_per_video from any one video — otherwise near-copies of the same
    # stretch of footage crowd everything else out.
    chosen, per_video = [], {}
    for sim, reason, payload in ranked:
        span, off = payload[0], payload[1]
        vid = span["video_id"]
        if per_video.get(vid, 0) >= max_per_video:
            continue
        if any(o[0]["video_id"] == vid and o[0]["scene_id"] == span["scene_id"] and abs(o[1] - off) < duration
               for _, _, o in chosen):
            continue
        chosen.append((sim, reason, payload))
        per_video[vid] = per_video.get(vid, 0) + 1
        if len(chosen) >= top:
            break
    return chosen


def more_like_this_for_grid(c: dict, seg: dict, queues: dict, excluded_here: set, global_excluded: set,
                            sequential: bool, audio_curve, existing: list) -> list | None:
    """✨ for a Manual step-through card: block-length clips from the SELECTED
    videos' still-available footage (same spans the grid searches — sequential
    rule, exclusions and tag filter already applied), most like candidate c.
    Returned in the same shape as shape-search candidates, so View fit,
    ticking and Confirm work on them unchanged."""
    duration = seg["end"] - seg["start"]
    spans = [sp for sp in get_candidate_spans(queues, sequential, global_excluded, min_duration=duration)
             if (sp["video_id"], sp["scene_id"]) not in excluded_here]

    def exclude(span, off):
        if (span["video_id"], span["scene_id"]) == (c["video_id"], c["scene_id"]) \
                and abs(off - c["window_offset_sec"]) < duration:
            return True  # overlaps the reference clip itself
        return any((e["video_id"], e["scene_id"]) == (span["video_id"], span["scene_id"])
                   and abs(e["window_offset_sec"] - off) < 0.25 for e in existing)

    found = find_similar_clips({"video_id": c["video_id"], "scene_id": c["scene_id"],
                                "offset": c["window_offset_sec"], "tags": c.get("tags", [])},
                               spans, duration, exclude, top=6)
    if found is None:
        return None
    out = []
    for sim, reason, (span, off, vals, fps) in found:
        out.append({
            "video_id": span["video_id"], "scene_id": span["scene_id"],
            "scene_start_sec": span["scene_start_sec"],
            "window_offset_sec": round(off, 3), "window_duration_sec": round(duration, 3),
            "score": round(shape_score(vals, audio_curve), 3),
            "motion_norm": span["motion_norm"], "curve_slice": vals, "fps": fps,
            "tags": span["tags"], "thumbnail": span["thumbnail"],
            "intro_candidate": span.get("intro_candidate", False),
            "outro_candidate": span.get("outro_candidate", False),
            "roles": span.get("roles", []),
            "similar": int(round(100 * sim)), "similar_reason": reason,
            "similar_to": f"{c['video_id']} #{c['scene_id']}",
            "_key": f"{span['video_id']}_{span['scene_id']}_s{int(round(off * 10))}",
        })
    return out


def compute_similar_videos(ref_id: str, tag_filter: tuple, time_ranges: dict) -> list:
    """✨ Similar videos: [(similarity, reason, video_id)] — top 5 from the whole
    library, each judged on its usable footage (range + tag filter applied)."""
    catalogues = load_all_catalogues()
    wanted = {t.lower() for t in tag_filter}
    videos = {}
    for vid, cat in catalogues.items():
        summary = VR.load_summary(CATALOGUE_DIR, vid)
        if summary is None:
            continue
        rng = time_ranges.get(vid)
        feats = VR.video_features(summary, {sc["scene_id"]: sc for sc in cat["scenes"]}, rng, tuple(tag_filter))
        if not feats:
            continue
        tag_seconds = {}
        for sc in cat["scenes"]:
            if sc.get("excluded"):
                continue
            a, b = _tc_to_seconds(sc["start_tc"]), _tc_to_seconds(sc["end_tc"])
            a, b = _overlap_with_range(a, b, rng)
            if a is None:
                continue
            tags = sc.get("tags", [])
            if wanted and not (wanted & {t.lower() for t in tags}):
                continue
            for t in tags:
                tag_seconds[t] = tag_seconds.get(t, 0.0) + (b - a)
        videos[vid] = {"tag_seconds": tag_seconds, "level": feats["level"],
                       "dynamics": feats["dynamics"], "punch": feats["punch"]}
    idf = SIM.tag_idf(catalogues, lambda sc: sc.get("tags", []))
    return SIM.rank_similar_videos(ref_id, videos, idf, top=5)


def replacement_pool(confirmed_map: dict, seg_idx: int, pick_idx: int, n_blocks: int) -> dict:
    """Footage still available for ONE placed clip if every other confirmed clip
    stays where it is: earlier blocks are replayed as usual, every other clip
    is taken out, and in sequential mode each video is also cut off before its
    clip in any later block — so a replacement keeps the order intact."""
    ss = st.session_state
    sequential_mode = bool(ss.get("sequential_mode"))
    q = build_footage_queues(tuple(ss.get("committed_selected_videos", [])), tuple(ss.get("committed_tag_filter", [])),
                             effective_time_ranges())
    for i in range(seg_idx):
        for p in confirmed_map.get(i, []):
            try:
                carve_span(q, p["video_id"], p["scene_id"], p["offset_into_scene_sec"],
                           p["clip_duration_sec"], sequential_mode)
            except ValueError:
                pass
    earliest_later = {}
    for i in range(seg_idx, n_blocks):
        for j, p in enumerate(confirmed_map.get(i, [])):
            if i == seg_idx and j == pick_idx:
                continue
            try:
                carve_span(q, p["video_id"], p["scene_id"], p["offset_into_scene_sec"],
                           p["clip_duration_sec"], False)
            except ValueError:
                pass
            if i > seg_idx:
                earliest_later[p["video_id"]] = min(earliest_later.get(p["video_id"], float("inf")),
                                                    p["clip_start_sec"])
    if sequential_mode:
        for vid, t in earliest_later.items():
            limit_queue_before(q, vid, t)
    return q


def _swap_candidates(mode: str, confirmed_map: dict, seg_idx: int, pick_idx: int, n_blocks: int) -> list | None:
    """Top 3 replacements for one placed clip, from the selected videos'
    available pool (see replacement_pool). Videos already in this block are
    left out (Choreography always; Manual unless 'Allow multiple clips from the
    same video' is on), apart from the clip's own video."""
    pick = confirmed_map[seg_idx][pick_idx]
    duration = pick["clip_duration_sec"]
    q = replacement_pool(confirmed_map, seg_idx, pick_idx, n_blocks)
    others = {p["video_id"] for j, p in enumerate(confirmed_map.get(seg_idx, [])) if j != pick_idx}
    if mode == "manual" and bool(st.session_state.get("allow_same_video")):
        others = set()
    others.discard(pick["video_id"])
    gex = st.session_state.get("global_excluded_scenes", set())
    spans = [sp for vid, sps in q.items() if vid not in others for sp in sps
             if sp["remaining_sec"] >= duration - 1e-6 and (vid, sp["scene_id"]) not in gex]

    def exclude(span, off):
        return (span["video_id"], span["scene_id"]) == (pick["video_id"], pick["scene_id"]) \
            and abs(off - pick["offset_into_scene_sec"]) < duration

    return find_similar_clips({"video_id": pick["video_id"], "scene_id": pick["scene_id"],
                               "offset": pick["offset_into_scene_sec"], "tags": pick.get("tags", [])},
                              spans, duration, exclude, top=3)


def render_swap_for_similar(mode: str, confirmed_map: dict, seg_idx: int, n_blocks: int) -> None:
    """🔁 Swap for similar panel under a review-list block, when one of its
    clips was picked. mode: "manual" or "chor"."""
    target = st.session_state.get("swap_target")
    if not target or target[0] != mode or target[1] != seg_idx:
        return
    pick_idx = target[2]
    picks = confirmed_map.get(seg_idx, [])
    if pick_idx >= len(picks):
        st.session_state.pop("swap_target", None)
        return
    pick = picks[pick_idx]
    catalogues = load_all_catalogues()
    sequential_mode = bool(st.session_state.get("sequential_mode"))
    with st.container(border=True):
        st.markdown(f"**🔁 Swap `{pick['video_id']}` #{pick['scene_id']} for something similar** "
                    f"— from your selected videos' available footage; every other block stays as it is.")
        cache_key = (mode, seg_idx, pick_idx, pick["video_id"], pick["scene_id"], pick["offset_into_scene_sec"])
        cached = st.session_state.get("swap_results")
        if not cached or cached["key"] != cache_key:
            with st.spinner("Finding similar clips…"):
                found = _swap_candidates(mode, confirmed_map, seg_idx, pick_idx, n_blocks)
            cached = {"key": cache_key, "found": found}
            st.session_state["swap_results"] = cached
        found = cached["found"]
        if found is None:
            st.warning("This clip has no frame-by-frame motion data, so there's nothing to compare it with. "
                       "Run the motion-curve backfill for its video.")
        elif not found:
            st.info("No other footage of this length is available for this block.")
        else:
            cols = st.columns(len(found))
            for k, (col, (sim, reason, (span, off, _vals, _fps))) in enumerate(zip(cols, found)):
                with col:
                    thumb = get_nearest_thumbnail(catalogues, span["video_id"], span["scene_start_sec"] + off)
                    if thumb is not None:
                        safe_image(thumb, width=180)
                    st.caption(f"`{span['video_id']}` #{span['scene_id']} at "
                               f"{format_mmss(span['scene_start_sec'] + off)}  ·  **{int(round(100 * sim))}% similar**")
                    st.caption(reason)
                    if st.button("Use this", key=f"swap_use_{mode}_{seg_idx}_{k}", type="primary"):
                        q = replacement_pool(confirmed_map, seg_idx, pick_idx, n_blocks)
                        try:
                            new_pick = carve_span(q, span["video_id"], span["scene_id"], off,
                                                  pick["clip_duration_sec"], sequential_mode)
                        except ValueError as e:
                            st.error(f"Couldn't use that clip: {e}")
                        else:
                            confirmed_map[seg_idx][pick_idx] = new_pick
                            if mode == "chor":
                                ov = st.session_state.setdefault("chor_overrides", {}).setdefault(seg_idx, {})
                                ov.pop(pick["video_id"], None)
                                ov[new_pick["video_id"]] = (new_pick["scene_id"], off)
                                if new_pick["video_id"] != pick["video_id"]:
                                    st.session_state[f"chor_pick_{seg_idx}_{pick['video_id']}"] = False
                                st.session_state[f"chor_pick_{seg_idx}_{new_pick['video_id']}"] = True
                            st.session_state.pop("swap_target", None)
                            st.session_state.pop("swap_results", None)
                            st.rerun()
        if st.button("Cancel", key=f"swap_cancel_{mode}_{seg_idx}"):
            st.session_state.pop("swap_target", None)
            st.session_state.pop("swap_results", None)
            st.rerun()


def blocks_to_show(label: str, key: str, n_blocks: int) -> int:
    """'Blocks to show' slider for the review lists: its range is 1…n_blocks and
    it defaults to every block (capped at 50 for long tracks). When the number
    of blocks changes (new track, different segmentation), it resets to that
    default instead of keeping a value from the old range."""
    if n_blocks <= 1:
        return n_blocks
    default = min(n_blocks, 50)
    if st.session_state.get(f"{key}__n") != n_blocks:
        st.session_state.pop(key, None)
        st.session_state[f"{key}__n"] = n_blocks
    return st.slider(label, 1, n_blocks, default, key=key)


@shared_cache
@st.cache_resource(ttl=1800, show_spinner=False)   # shared, not copied per call — read-only
def load_all_audio_tracks() -> dict:
    """track_id -> full track dict, loaded once and cached. SHARED — read it, never change it."""
    tracks = {}
    for f in AUDIO_DIR.glob("*.json"):
        if f.name == "audio_index.json":
            continue
        with open(f) as fh:
            t = json.load(fh)
        tracks[t.get("track_id", f.stem)] = t
    return tracks


def video_role_seconds(time_ranges: dict) -> dict:
    """{video_id: {role id: seconds of usable (not excluded, in range) footage carrying it}}."""
    out = {}
    for vid, cat in load_all_catalogues().items():
        rng = time_ranges.get(vid)
        d = {}
        for sc in cat["scenes"]:
            if sc.get("excluded"):
                continue
            roles = SR.scene_roles(sc)
            if not roles:
                continue
            a, b = _overlap_with_range(_tc_to_seconds(sc["start_tc"]), _tc_to_seconds(sc["end_tc"]), rng)
            if a is None:
                continue
            for r in roles:
                d[r] = d.get(r, 0.0) + (b - a)
        if d:
            out[vid] = d
    return out


def story_role_need(track_id: str) -> dict:
    """{role id: seconds of the song its story map gives that role} ({} without a map)."""
    out = {}
    for r in current_story_map(track_id) or []:
        if r["role"]:
            out[r["role"]] = out.get(r["role"], 0.0) + (r["end"] - r["start"])
    return out


def compute_video_rankings(track: dict, tag_filter: tuple, time_ranges: dict,
                           weights: dict, experimental: bool) -> list[dict]:
    """Match score (0–100) for every video with usable footage, best first — see
    video_ranking.py for the metrics. Each video is compared against the WHOLE
    library (ranges applied, no tag filter), so the tag filter never shifts the
    scale. Per-video motion summaries are cached on disk; any missing ones are
    built here with a progress bar (first run, or after new videos sync in)."""
    catalogues = load_all_catalogues()
    video_ids = sorted(catalogues)
    todo = VR.missing_summaries(CATALOGUE_DIR, video_ids)
    if todo:
        bar = st.progress(0.0, text=f"Analysing motion curves for {len(todo)} video(s) (one-off)…")
        for i, vid in enumerate(todo):
            VR.load_summary(CATALOGUE_DIR, vid)
            bar.progress((i + 1) / len(todo), text=f"Analysing motion curves… {vid}")
        bar.empty()

    feats, reference = {}, {}
    for vid in video_ids:
        if VR.load_summary(CATALOGUE_DIR, vid) is None:
            continue
        scenes_by_id = {sc["scene_id"]: sc for sc in catalogues[vid]["scenes"]}
        rng = time_ranges.get(vid)
        # ⚡ kept in memory until the video's catalogue changes (they don't depend on the track)
        reference[vid] = VR.cached_video_features(CATALOGUE_DIR, vid, scenes_by_id, rng, ())
        feats[vid] = (VR.cached_video_features(CATALOGUE_DIR, vid, scenes_by_id, rng, tuple(tag_filter))
                      if tag_filter else reference[vid])
    return VR.score_videos(
        track, load_all_audio_tracks(), feats, reference, weights, tuple(tag_filter),
        usage=VR.combined_usage(PROJECTS_DIR, PREVIEW_DIR), experimental=experimental,
        role_feats=video_role_seconds(time_ranges), role_need=story_role_need(track.get("track_id")),
    )


def _ranking_weights() -> dict:
    stored = st.session_state.setdefault("ranking_weights", dict(VR.DEFAULT_WEIGHTS))
    return {m: float(stored.get(m, VR.DEFAULT_WEIGHTS[m])) for m in VR.METRICS}


def render_ranking_weights(tag_filter: list) -> None:
    """⚙️ Ranking weights — one slider per metric. Values live in the plain
    session_state dict "ranking_weights" (saved with projects), so they
    survive this section's widgets being unrendered."""
    weights = _ranking_weights()
    with st.expander("⚙️ Ranking weights — how the top videos are chosen", expanded=False):
        st.caption("Each video gets a 0–100 match score from these metrics. 0 switches a metric off; "
                   "higher makes it count more. The list re-ranks as you change them.")
        experimental = st.toggle("Show experimental metrics (movement tempo, energy arc)",
                                 value=bool(st.session_state.get("ranking_experimental", False)),
                                 key="ranking_experimental_toggle")
        st.session_state["ranking_experimental"] = experimental
        shown = [m for m in VR.METRICS if experimental or m not in VR.EXPERIMENTAL]
        cols = st.columns(3)
        for i, m in enumerate(shown):
            with cols[i % 3]:
                disabled = (m == "relevance" and not tag_filter)
                weights[m] = st.slider(
                    VR.LABELS[m], 0.0, 5.0, value=weights[m], step=0.1, key=f"rank_w_{m}",
                    disabled=disabled,
                    help=VR.HELP[m] + (" (Set a tag filter above to use it.)" if disabled else ""),
                )
        if st.button("Reset to defaults", key="rank_w_reset"):
            weights = dict(VR.DEFAULT_WEIGHTS)
            for m in VR.METRICS:
                st.session_state.pop(f"rank_w_{m}", None)
            st.session_state["ranking_weights"] = weights
            rerun_fragment()
        st.session_state["ranking_weights"] = weights


def format_match_breakdown(match: dict, experimental: bool, tag_filter) -> str:
    short = {"dynamics": "dyn", "punch": "punch", "footage": "footage", "pace": "pace",
             "spread": "spread", "relevance": "tags", "freshness": "fresh", "story": "story",
             "tempo": "tempo", "arc": "arc"}
    parts = []
    for m in VR.METRICS:
        if m in VR.EXPERIMENTAL and not experimental:
            continue
        if m == "relevance" and not tag_filter:
            continue
        if m == "story" and not story_role_need(st.session_state.get("track_id")):
            continue
        v = match["parts"].get(m, 0.0)
        dots = "●" * int(round(v * 4)) + "○" * (4 - int(round(v * 4)))
        label = short[m]
        if m == "tempo" and match.get("tempo_bpm"):
            label += f" ({match['tempo_bpm']:.0f} BPM)"
        parts.append(f"{label} {dots}")
    text = "  ·  ".join(parts)
    if match.get("curve_coverage", 1.0) < 0.5:
        text += "  ·  ⚠️ no motion curves for most scenes (run the motion-curve backfill)"
    return text


def _columns_support_valign() -> bool:
    import inspect
    try:
        return "vertical_alignment" in inspect.signature(st.columns).parameters
    except (TypeError, ValueError):
        return False


_COLUMNS_SUPPORT_VALIGN = _columns_support_valign()

RECOMMENDED_COUNT = 10   # how many best-matching videos are listed by default


def _usage_signature() -> tuple:
    """Changes whenever a project is saved or a render is made/deleted, so the
    ranking (its freshness metric) is recomputed then — and only then."""
    sig = []
    for d in (PROJECTS_DIR, PREVIEW_DIR):
        try:
            entries = [e for e in os.scandir(d) if e.name.endswith(".json") and not e.name.startswith(".")]
            sig.append((len(entries), max((e.stat().st_mtime for e in entries), default=0.0)))
        except OSError:
            sig.append((0, 0.0))
    return tuple(sig)


def resolve_video_selection(track: dict, tag_filter: tuple, all_video_ids: list, allowed: frozenset = None) -> dict:
    """Which videos are recommended (the top-10 ranking) and which are
    currently selected, read entirely from session_state — called by the
    Video Selection fragment on every one of its own reruns.

    The RANKING itself (video_matches/top_ids/other_options) is cached and
    only recomputed when something it depends on changes: the track, the tag
    filter, any video's effective time range, or the ranking weights.
    Ticking a checkbox or adding something from the dropdown doesn't, so the
    list doesn't redraw on every click.

    SELECTION (who's ticked) is never cached — always computed fresh, since
    it's cheap and must reflect the current checkboxes exactly."""
    _ranges = effective_time_ranges()
    _weights = _ranking_weights()
    _experimental = bool(st.session_state.get("ranking_experimental", False))
    cache_key = (track.get("track_id"), tag_filter,
                 tuple(sorted((v, tuple(r)) for v, r in _ranges.items())),
                 tuple(sorted(_weights.items())), _experimental, _usage_signature(),
                 tuple(sorted(allowed)) if allowed is not None else None,
                 tuple(sorted(story_role_need(track.get("track_id")).items())))
    random_seed = st.session_state.get("random_top_seed")
    cache = st.session_state.get("video_ranking_cache")
    if cache and cache["key"] == cache_key and cache.get("seed") == random_seed:
        ranking = cache["ranking"]
    else:
        if cache and cache["key"] == cache_key:
            video_matches = cache["ranking"]["video_matches"]   # only the seed changed — no re-scoring
        else:
            with _timed("Video ranking", f"{len(_ranges)} ranges"):
                video_matches = compute_video_rankings(track, tag_filter, _ranges, _weights, _experimental)
            if allowed is not None:   # 👤 actor / 🏷️ video tag filter
                video_matches = [v for v in video_matches if v["video_id"] in allowed]
        if random_seed is not None:
            # 🎲 Randomize: RECOMMENDED_COUNT videos drawn purely at random from every video that
            # has usable scenes — the ranking weights play no part. Same seed → same draw.
            top_matches = random.Random(random_seed).sample(video_matches, min(RECOMMENDED_COUNT, len(video_matches)))
        else:
            top_matches = video_matches[:RECOMMENDED_COUNT]
        top_ids = [v["video_id"] for v in top_matches]
        match_by_id = {v["video_id"]: v for v in video_matches}
        rank_order = {v["video_id"]: i for i, v in enumerate(video_matches)}
        other_options = sorted((vid for vid in all_video_ids if vid not in top_ids
                                and (allowed is None or vid in allowed)),
                               key=lambda vid: (rank_order.get(vid, 10**6), vid))
        ranking = {
            "video_matches": video_matches, "top_matches": top_matches, "top_ids": top_ids,
            "match_by_id": match_by_id, "rank_order": rank_order, "other_options": other_options,
        }
        st.session_state["video_ranking_cache"] = {"key": cache_key, "seed": random_seed, "ranking": ranking}

    # A stored pick can drop out of the options (e.g. it entered the top 10 after a tag change) —
    # filtered here regardless of whether the fragment's own widget-safety sanitizing has run.
    extra_videos_stored = [v for v in st.session_state.get("extra_videos", []) if v in ranking["other_options"]]
    extra_rows = list(extra_videos_stored)
    # A video with a custom time range must stay listed even if the range left it with no
    # matching scenes — otherwise its 🎚️ controls would vanish and the range couldn't be cleared.
    extra_rows += [vid for vid in st.session_state["video_time_ranges"]
                  if vid not in ranking["top_ids"] and vid not in extra_rows]
    # A previously committed selection must also stay listed even if the "extra_videos" multiselect's
    # OWN widget state was discarded by Streamlit for having gone unrendered while the user viewed a
    # different section (this fragment's widgets only run while this section is the active one).
    # committed_selected_videos is a plain session_state entry, not tied to any widget, so it survives
    # that and is the right thing to self-heal from.
    extra_rows += [vid for vid in st.session_state.get("committed_selected_videos", [])
                  if vid not in ranking["top_ids"] and vid not in extra_rows and vid in ranking["other_options"]]
    selected_videos = [vid for vid in ranking["top_ids"] + extra_rows if st.session_state.get(f"select_video_{vid}", False)]
    return {
        **ranking,
        "extra_videos_stored": extra_videos_stored, "extra_rows": extra_rows,
        "selected_videos": selected_videos,
    }

