"""Golden cases: run the planner's cutting, shape-matching and auto-matching
functions over the synthetic library and record exactly what they return.

run_all(ns) takes a page namespace (tests.support.load_page) and returns
{case name: plain-JSON result}. The same function is used to CAPTURE the
snapshot from the code as it was before a refactor and to CHECK the code after
it, so any difference in behaviour shows up as a failing case.
"""

from __future__ import annotations

import copy
import json

import numpy as np

from tests.fixtures import TRACKS, VIDEOS, load_track
from tests.support import DATA

ROUND = 9


_DATA_STR = str(DATA)


def plain(x):
    """JSON-safe, order-stable copy with floats rounded (numpy included), and
    the test library's folder written as <DATA> so snapshots don't depend on
    where the repo is checked out."""
    if isinstance(x, str) and _DATA_STR in x:
        return x.replace(_DATA_STR, "<DATA>")
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((plain(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, np.ndarray):
        return [plain(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return round(float(x), ROUND)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


BASE_STATE = {
    "autofill_v2": True,
    "_avg_clips_per_block": 1.0,
    "_plan_story_in_use": False,
    "intro_outro_exclusive": False,
    "_plan_locked_roles": set(),
    "_plan_reserved_roles": set(),
    "video_time_ranges": {},
    "clip_count_skip": [],
    "segmentation_method": "Smart (self-tuning)",
    "beats_per_bar": 4,
}


def _reset_state(ns, **extra):
    ss = ns["st"].session_state
    ss.clear()
    ss.update(copy.deepcopy(BASE_STATE))
    for k, v in ns["DIAL_DEFAULTS"].items():
        ss.setdefault(k, copy.deepcopy(v))
    ss.update(extra)


def _cfg(ns, method, **over):
    cfg = {"method": method, "beats_per_bar": int(over.pop("beats_per_bar", 4))}
    for k in ns["SEG_METHOD_KEYS"][method]:
        cfg[k] = over.get(k, ns["DIAL_DEFAULTS"].get(k))
    return cfg


def segmentation_cases(ns) -> dict:
    out = {}
    m = {"adaptive": ns["SEG_ADAPTIVE"], "rhythm": ns["SEG_RHYTHM"],
         "scored": ns["SEG_SCORED"], "smart": ns["SEG_SMART"]}
    variants = {
        "adaptive": [{}, {"change_threshold": 0.05, "min_segment_sec": 1.0, "max_segment_sec": 4.0},
                     {"react_to_hits": False, "max_segment_sec": 0.0}],
        "rhythm": [{}, {"dynamic_priority_override": True, "override_delta_thresh": 0.1},
                   {"fast_energy_thresh": 0.4, "fast_max_bars": 0, "cinematic_bars": 2}],
        "scored": [{}, {"sc_bars_per_cut": 1.0, "sc_follow_energy": 0.9}],
        "smart": [{}, {"sm_pace": 1}, {"sm_pace": -2}],
    }
    for tid in TRACKS:
        track = load_track(DATA, tid)
        for name, method in m.items():
            for k, over in enumerate(variants[name]):
                over = {kk: v for kk, v in over.items() if kk in ns["DIAL_DEFAULTS"]}
                cfg = _cfg(ns, method, **over)
                try:
                    segs = ns["build_segments"](track, cfg)
                    segs = copy.deepcopy(segs)
                    ns["apply_segment_intensity"](segs, track, 1.3)
                    res = segs
                except Exception as e:                      # record failures too
                    res = {"error": type(e).__name__}
                out[f"seg/{tid}/{name}/{k}"] = res
        out[f"density/{tid}"] = [len(x) for x in ns["compute_density_curve"](track)] + [
            float(np.sum(ns["compute_density_curve"](track)[1]))]
        out[f"bars/{tid}"] = ns["compute_bar_times"](track, 4)
        out[f"states/{tid}"] = ns["compute_cutting_states"](track, 4)
        out[f"legacy/{tid}"] = ns["_legacy_track_segments"](track, 2.0, 0.5, 0.5, 2.0, True) \
            if track.get("energy_envelope") else None
    # clip counts
    import random
    rng = random.Random(3)
    grid = []
    for inten in np.linspace(0, 1, 11):
        for vary in (False, True):
            grid.append(ns["_clips_for_intensity"](float(inten), 1, 5, 0.2, 0.85, vary, rng))
    out["clips_for_intensity"] = grid
    return out


def autotune_cases(ns) -> dict:
    out = {}
    for tid in ("song_full", "song_noanalysis", "song_slow", "song_legacy"):
        track = load_track(DATA, tid)
        for method in (ns["SEG_ADAPTIVE"], ns["SEG_RHYTHM"], ns["SEG_SCORED"], ns["SEG_SMART"]):
            for style in ("Calm", "Balanced", "Energetic"):
                _reset_state(ns)
                try:
                    s, why = ns["auto_tune_parameters"](track, 1, 4, method=method, style_name=style,
                                                        beats_per_bar=4)
                    out[f"autotune/{tid}/{method}/{style}"] = {"settings": s, "why": why}
                except Exception as e:
                    out[f"autotune/{tid}/{method}/{style}"] = {"error": type(e).__name__}
    return out


def shape_cases(ns) -> dict:
    out = {}
    rng = np.random.default_rng(7)
    for k in range(6):
        vals = np.abs(rng.normal(0.5, 0.3, 200 + 37 * k))
        audio = np.abs(rng.normal(0.5, 0.2, 60 + 11 * k))
        bias = rng.uniform(0, 0.2, len(vals) - 30 + 1) if k % 2 else None
        out[f"best_window/{k}"] = ns["find_best_shape_window"](vals, 30, audio, bias=bias)
        out[f"shape_score/{k}"] = ns["shape_score"](vals[:90], audio)
        out[f"resample/{k}"] = ns["resample_curve"](audio, 17)
    out["interp_matrix"] = ns["_make_interp_matrix"](13, 7)
    for tid in ("song_full", "song_legacy"):
        tr = load_track(DATA, tid)
        out[f"block_curve/{tid}"] = ns["get_block_audio_curve"](tr, 3.3, 7.9)
    return out


def _queues(ns, ranges=None, tag_filter=()):
    return ns["build_footage_queues"](tuple(sorted(VIDEOS)), tuple(tag_filter), ranges or {})


def footage_cases(ns) -> dict:
    out = {}
    _reset_state(ns)
    out["queues/all"] = _queues(ns)
    out["queues/tag_dance"] = _queues(ns, tag_filter=("Dance",))
    out["queues/ranges"] = _queues(ns, ranges={"vid_a": (5.0, 40.0), "vid_c": (100.0, 10000.0)})
    q = _queues(ns)
    seq = []
    seq.append(ns["consume_span"](q["vid_a"][0], 1.2))
    seq.append(ns["carve_span"](q, "vid_c", q["vid_c"][3]["scene_id"], q["vid_c"][3]["offset_sec"] + 0.5, 1.0))
    seq.append(ns["carve_span"](q, "vid_b", q["vid_b"][4]["scene_id"], q["vid_b"][4]["offset_sec"] + 0.2, 0.9,
                                sequential=True))
    ns["skip_span"](q, "vid_d", q["vid_d"][2]["scene_id"], sequential=True)
    ns["split_span_at"](q, "vid_f", q["vid_f"][1]["scene_id"], q["vid_f"][1]["offset_sec"] + 0.7)
    ns["limit_queue_before"](q, "vid_a", 30.0)
    try:
        ns["carve_span"](q, "vid_c", q["vid_c"][3]["scene_id"], 9999.0, 1.0)
    except ValueError:
        seq.append("ValueError")
    out["queue_ops/picks"] = seq
    out["queue_ops/after"] = q
    for seqm in (False, True):
        out[f"candidate_spans/{seqm}"] = [(s["video_id"], s["scene_id"]) for s in
                                          ns["get_candidate_spans"](_queues(ns), seqm, {("vid_a", 1)}, 2.0)]
    return out


def _segments(ns, tid, method="smart"):
    track = load_track(DATA, tid)
    method = {"smart": ns["SEG_SMART"], "adaptive": ns["SEG_ADAPTIVE"]}[method]
    segs = ns["build_segments"](track, _cfg(ns, method))
    ns["apply_segment_intensity"](segs, track, 1.0)
    return track, segs


MATCH_CONFIGS = {
    "default": dict(state={}, kw={}),
    "split": dict(state={}, kw={"min_clips": 1, "max_clips": 3}),
    "split_same": dict(state={}, kw={"min_clips": 2, "max_clips": 4, "allow_same_video": True}),
    "sequential": dict(state={"_avg_clips_per_block": 1.5}, kw={"sequential": True, "max_clips": 2}),
    "sequential_F0": dict(state={}, kw={"sequential": True, "weight_position": 0.0}),
    "v1": dict(state={"autofill_v2": False}, kw={"max_clips": 2}),
    "v1_seq": dict(state={"autofill_v2": False}, kw={"sequential": True}),
    "roles": dict(state={"_plan_story_in_use": True, "_plan_reserved_roles": {"intro", "outro", "action"}},
                  kw={"roles": True}),
    "roles_locked": dict(state={"intro_outro_exclusive": True, "_plan_locked_roles": {"intro", "outro"}},
                         kw={"intro_outro": True}),
    "no_track": dict(state={}, kw={"no_track": True, "max_clips": 2}),
    "exclusions": dict(state={}, kw={"exclusions": True}),
    "vary": dict(state={}, kw={"vary_count": True, "seed": 5, "min_clips": 1, "max_clips": 4}),
    "skip_counts": dict(state={"clip_count_skip": [2]}, kw={"min_clips": 1, "max_clips": 4}),
}


def match_cases(ns) -> dict:
    out = {}
    for tid in ("song_full", "song_noanalysis", "song_short"):
        for name, conf in MATCH_CONFIGS.items():
            _reset_state(ns, **copy.deepcopy(conf["state"]))
            track, segs = _segments(ns, tid)
            kw = dict(conf["kw"])
            q = _queues(ns)
            args = dict(
                min_clips=kw.pop("min_clips", 1), max_clips=kw.pop("max_clips", 1),
                ramp_start=0.2, ramp_end=0.85,
                allow_same_video=kw.pop("allow_same_video", False),
                sequential=kw.pop("sequential", False),
                vary_count=kw.pop("vary_count", False), seed=kw.pop("seed", 0),
                min_clip_len_sec=1.0,
                weight_shape=2.0, weight_random=0.5, avoid_repeat=True, weight_spread=1.5,
                weight_motion=2.0, autofill_seed=42,
                weight_position=kw.pop("weight_position", 1.0),
                repeat_fallbacks=set(), shape_matches=6,
                track=None if kw.pop("no_track", False) else track,
            )
            n = len(segs)
            if kw.pop("roles", False):
                roles = [None] * n
                roles[0] = "intro"
                roles[-1] = "outro"
                for i in range(n // 3, n // 2):
                    roles[i] = "action"
                args["block_roles"] = roles
            if kw.pop("intro_outro", False):
                args["intro_blocks"] = {0}
                args["outro_blocks"] = {n - 1}
            if kw.pop("exclusions", False):
                args["segment_exclusions"] = {1: {("vid_a", 0), ("vid_c", 1)}}
                args["segment_clip_reduction"] = {2: 1}
                args["global_excluded"] = {("vid_b", 0), ("vid_f", 4)}
            assert not kw, kw
            try:
                timeline, short_n, short_s = ns["match_scenes_to_track"](segs, q, **args)
                out[f"match/{tid}/{name}"] = {"timeline": timeline, "short_n": short_n, "short_s": short_s,
                                              "fallbacks": args["repeat_fallbacks"],
                                              "left": {v: sum(s["remaining_sec"] for s in sp) for v, sp in q.items()}}
            except Exception as e:
                out[f"match/{tid}/{name}"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def shape_candidate_cases(ns) -> dict:
    out = {}
    for seqm in (False, True):
        _reset_state(ns)
        track, segs = _segments(ns, "song_full", "adaptive")
        q = _queues(ns)
        for bi in (0, 3, len(segs) // 2):
            seg = segs[bi]
            curve = ns["get_block_audio_curve"](track, seg["start"], seg["end"])
            for restrict in (False, True):
                res, missing = ns["find_shape_candidates"](
                    seg, q, seqm, set(), {("vid_a", 2)}, curve, 8, restrict_to_front=restrict,
                    remaining_blocks=len(segs) - bi, weight_position=1.0 if restrict else 0.0,
                    position_targets=ns["compute_position_targets"](bi, len(segs), {}, q, 1) if restrict else None,
                    weight_shape=2.0)
                out[f"shape_cands/{seqm}/{bi}/{restrict}"] = {"res": res, "missing": missing}
    return out


def with_core(ns):
    """The page namespace, falling back to core/ for helpers the page no longer
    imports itself (so the same cases run before and after the refactor). The
    page's own definitions — including its wrappers — win."""
    from collections import ChainMap
    try:
        import core.segmentation, core.autotune, core.shape, core.footage, core.picking  # noqa: E401
    except ImportError:          # code from before core/ existed
        return ns
    return ChainMap(ns, *(vars(m) for m in (core.segmentation, core.autotune, core.shape, core.footage,
                                             core.picking)))


def run_all(ns) -> dict:
    ns = with_core(ns)
    out = {}
    for fn in (segmentation_cases, autotune_cases, shape_cases, footage_cases, shape_candidate_cases,
               match_cases):
        out.update(fn(ns))
    return plain(out)
