"""
🤖 The story matcher — Stage 3 step 5 (claude/new-matcher-design.md §D–F). Matches ANY half second of any
processed video to the song, driven by the 🧭 footage timeline (footage_timeline.py), the track's 🧩 section map
(section_types.py) and the 🍳 recipes (recipes.py). The Classic matcher (core/matching.py) is untouched; this
one produces a plan in exactly the same shape, so cards, previews, render and the Blender JSON don't change.

For every block of the track:
  1. its section type (from the section map at the block's middle) → its recipe;
  2. every window of block length inside every available span of every video with a timeline is scored
     (vectorised over the video's half seconds):
       base  = movement band + trend (recipe) · shape of the person-movement vs the music · beat lock
       story = recipe tag weights × the window's tag scores − avoid tags; scene tags filter / prefer;
       fit   = (1 − music_story) × base + music_story × story, then
               + the block's 🎭 role as a bonus · − junk (nobody on screen) · − variety / neighbour penalties ·
               ± the last pass: shot size / POV continuity with the previous clip, relaxed where the recipe
               says a change is fine, contrast rewarded where it says contrast is welcome;
  3. the best windows fill the block's slots (distinct videos unless allowed), carved out of the queues like
     Classic's picks, so no footage is used twice and in-order mode holds. Split-screen panes are pulled towards
     the first (anchor) pane's shot size.
Blocks no timeline window can fill (no processed videos left, footage exhausted) fall back to Classic's chain
fill, so the plan never has gaps. Each slot carries a "story" dict (type, fit, base, story) for the cards.

No Streamlit in here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import footage_timeline as FT
import recipes as RC
import section_types as ST
from core.footage import carve_span, get_candidate_spans, has_role
from core.matching import PlanContext, excluded_for_block, fill_slot
from core.shape import get_block_audio_curve, resample_curve

STEP = FT.STEP
TOP_PER_SPAN = 3            # candidate windows kept per span (spread ≥ half a block apart)
MIN_GAP_POINTS = 2          # …at least this many half seconds apart
JUNK_SHARE = 0.5            # nobody on screen for over half the window → heavy penalty
W_BAND, W_TREND, W_SHAPE, W_BEAT = 0.45, 0.15, 0.25, 0.15      # inside base
AVOID_WEIGHT = 0.5
PREFER_BONUS = 0.1
ROLE_BONUS = 0.1
REPEAT_PENALTY = 0.3        # the same video as a neighbouring block
USE_PENALTY = 0.04          # per earlier use of the video in this plan (variety)
SHOT_PENALTY = 0.05         # per shot-size step jumped against the previous clip (× hold / allow / smooth)
POV_PENALTY = 0.06          # POV ↔ non-POV flip against the previous clip
PANE_PENALTY = 0.04         # per step away from the anchor pane in a split-screen block
CONTRAST_BONUS = 0.04       # where the recipe welcomes contrast and the slider leans that way


@dataclass
class StoryOptions:
    music_story: float = 0.5            # 0 = all music fit, 1 = all story fit
    smooth_contrast: float = 0.5        # 0 = keep shots steady everywhere, 1 = contrast welcome at changes
    track_recipes: dict | None = None   # ✍️ Custom: {type_id: recipe} overriding the shared recipes
    section_map: list | None = None     # the track's section map; None → suggested from the song
    max_videos: int = 0                 # at most this many different videos in the plan (0 = no limit)
    progress: object = None             # progress(done_blocks, total_blocks) — for a progress bar
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-video prepared arrays
# ---------------------------------------------------------------------------

class _Video:
    def __init__(self, tl: FT.Timeline, index: dict):
        self.tl = tl
        self.times = np.asarray(tl.times, np.float64)
        self.n = len(self.times)
        self.move = FT.scaled_move(tl.move, index.get("move_scale"))          # NaN where nobody tracked
        self.move_f = np.nan_to_num(self.move, nan=0.0)
        self.has_move = np.isfinite(self.move).astype(np.float64)
        self.scores = np.asarray(tl.scores, np.float32) / 100.0
        self.people = np.asarray(tl.people, np.float64)
        self.nobody = (self.people == 0).astype(np.float64)
        self.shot = np.asarray(tl.shot, np.float32)
        self.bpm = np.asarray(tl.bpm, np.float32)
        self.per = np.asarray(tl.periodicity, np.float32)
        pov = tl.visible("POV")
        self.pov = pov.astype(np.float64) if pov is not None else np.zeros(self.n)
        self._col = {str(t).lower(): k for k, t in enumerate(tl.tags)}
        self._cum = {}

    def col(self, tag: str):
        return self._col.get(str(tag).lower())

    def window_mean(self, arr: np.ndarray, n: int) -> np.ndarray:
        """Mean over [s, s+n) for every start s (len = T − n + 1)."""
        key = (id(arr), n)
        c = np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])
        return (c[n:] - c[:-n]) / n

    def tag_window(self, tag: str, n: int) -> np.ndarray | None:
        k = self.col(tag)
        if k is None:
            return None
        return self.window_mean(self.scores[:, k].astype(np.float64), n)


def _fold_ratio_fit(mbpm: np.ndarray, track_bpm: float) -> np.ndarray:
    """1 where the movement BPM sits on the track's tempo (or half / double it), falling to 0 a quarter
    octave away; 0 where there's no movement BPM."""
    out = np.zeros(len(mbpm))
    ok = np.isfinite(mbpm) & (mbpm > 0) & (track_bpm > 0)
    if not ok.any():
        return out
    r = np.log2(mbpm[ok] / track_bpm)
    d = np.min(np.abs(r[:, None] - np.array([[-1.0, 0.0, 1.0]])), axis=1)
    out[ok] = np.clip(1.0 - d / 0.25, 0.0, 1.0)
    return out


def _shape_scores(move: np.ndarray, n: int, audio_curve: np.ndarray) -> np.ndarray:
    """Correlation of every n-point movement window with the block's music shape → 0–1."""
    T = len(move)
    if T < n or n < 3 or audio_curve is None or len(audio_curve) == 0:
        return np.full(max(T - n + 1, 0), 0.5)
    a = resample_curve(audio_curve, n)
    a = a - a.mean()
    a_std = float(np.sqrt((a * a).mean()))
    st_ = move.strides[0]
    win = np.lib.stride_tricks.as_strided(move, (T - n + 1, n), (st_, st_), writeable=False)
    mean = win.mean(axis=1, keepdims=True)
    c = win - mean
    std = np.sqrt((c * c).mean(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where((std > 1e-9) & (a_std > 1e-9), (c @ a) / (n * std * a_std), 0.0)
    return (np.clip(corr, -1, 1) + 1) / 2


def _band_fit(mean_move: np.ndarray, lo: float, hi: float) -> np.ndarray:
    d = np.where(mean_move < lo, lo - mean_move, np.where(mean_move > hi, mean_move - hi, 0.0))
    return np.clip(1.0 - d / 0.3, 0.0, 1.0)


def _trend_fit(move: np.ndarray, n: int, trend: str) -> np.ndarray:
    """Slope over the window (last third − first third, scaled units) against the recipe's wish."""
    T = len(move)
    k = max(1, n // 3)
    c = np.concatenate([[0.0], np.cumsum(move, dtype=np.float64)])
    starts = np.arange(T - n + 1)
    first = (c[starts + k] - c[starts]) / k
    last = (c[starts + n] - c[starts + n - k]) / k
    slope = last - first
    if trend == "rise":
        return np.clip(0.5 + slope / 0.4, 0.0, 1.0)
    if trend == "fall":
        return np.clip(0.5 - slope / 0.4, 0.0, 1.0)
    return np.clip(1.0 - np.abs(slope) / 0.4, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Scoring one block's windows in one video
# ---------------------------------------------------------------------------

def _score_video(v: _Video, n: int, recipe: dict, audio_curve: np.ndarray, track_bpm: float,
                 music_story: float) -> tuple:
    """(fit, base, story) arrays over every window start of this video (NaN-free)."""
    if v.n < n:
        e = np.zeros(0)
        return e, e, e
    mean_move = v.window_mean(v.move_f, n) / np.maximum(v.window_mean(v.has_move, n), 1e-6)
    mv = recipe["movement"]
    band = _band_fit(mean_move, float(mv["lo"]), float(mv["hi"]))
    trend = _trend_fit(v.move_f, n, mv.get("trend", "flat"))
    shape = _shape_scores(v.move_f, n, audio_curve)
    bpm_mean = v.window_mean(np.nan_to_num(v.bpm.astype(np.float64), nan=0.0), n)
    bpm_has = v.window_mean(np.isfinite(v.bpm).astype(np.float64), n)
    mbpm = np.where(bpm_has > 0, bpm_mean / np.maximum(bpm_has, 1e-6), np.nan)
    beat_fit = _fold_ratio_fit(mbpm, track_bpm) * v.window_mean(np.nan_to_num(v.per.astype(np.float64)), n)
    lock = float(recipe.get("beat_lock", 0.5))
    beat = 1.0 - lock * (1.0 - beat_fit)
    base = W_BAND * band + W_TREND * trend + W_SHAPE * shape + W_BEAT * beat

    tags = recipe.get("tags") or {}
    num = np.zeros(len(base))
    den = 0.0
    for tag, w in tags.items():
        tw = v.tag_window(tag, n)
        if tw is None or w <= 0:
            continue
        num += float(w) * tw
        den += float(w)
    story = num / den if den > 0 else np.full(len(base), 0.5)
    avoid = np.zeros(len(base))
    for tag in recipe.get("avoid") or []:
        tw = v.tag_window(tag, n)
        if tw is not None:
            avoid = np.maximum(avoid, tw)
    story = np.clip(story - AVOID_WEIGHT * avoid, 0.0, 1.0)

    fit = (1.0 - music_story) * base + music_story * story
    junk = v.window_mean(v.nobody, n) > JUNK_SHARE
    fit = np.where(junk, fit * 0.2, fit)
    return fit, base, story


def _window_starts(v: _Video, span: dict, n: int) -> np.ndarray:
    """Start indices whose n-point window lies inside the span's still-available footage."""
    a = span["scene_start_sec"] + span["offset_sec"]
    b = a + span["remaining_sec"]
    d = n * STEP
    lo = int(np.searchsorted(v.times, a - 1e-3, side="left"))
    hi = int(np.searchsorted(v.times, b - d + 1e-3, side="right"))
    hi = min(hi, v.n - n + 1)
    return np.arange(lo, hi) if hi > lo else np.zeros(0, int)


def _top_spread(idx: np.ndarray, score: np.ndarray, k: int, gap: int) -> list:
    """Up to k start indices with the best scores, at least `gap` apart."""
    order = idx[np.argsort(-score[idx], kind="stable")]
    out = []
    for s in order:
        if all(abs(int(s) - o) >= gap for o in out):
            out.append(int(s))
            if len(out) >= k:
                break
    return out


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _section_types(track: dict, segments: list, options: StoryOptions) -> list:
    regions = options.section_map
    if regions is None:
        regions = ST.saved_map(track.get("track_id", ""))
    if not regions:
        try:
            regions = ST.suggested_map(track)
            options.notes.append("No saved 🧩 section map for this track — the suggested one was used.")
        except Exception:          # noqa: BLE001
            regions = []
    if not regions:
        options.notes.append("The song's sections couldn't be read — every block was matched as Rhythmic.")
        return ["rhythmic"] * len(segments)
    return [ST.type_at(regions, (s["start"] + s["end"]) / 2) or "rhythmic" for s in segments]


def _load_videos(queues: dict) -> tuple:
    index = FT.load_index()
    vids, missing = {}, []
    for vid in queues:
        tl = FT.load(vid, index=index)
        if tl is None or len(tl) == 0:
            missing.append(vid)
            continue
        vids[vid] = _Video(tl, index)
    return vids, missing


def _shot_at(v: _Video, idx: int) -> float:
    s = float(v.shot[idx]) if 0 <= idx < v.n else float("nan")
    return s


def _slot(pick: dict, info: dict) -> dict:
    return {"chain": [pick], "video_id": pick["video_id"], "scene_id": pick["scene_id"],
            "tags": sorted(pick["tags"]), "motion_norm": pick["motion_norm"], "thumbnail": pick["thumbnail"],
            "clip_duration_sec": pick["clip_duration_sec"], "offset_into_scene_sec": pick["offset_into_scene_sec"],
            "clip_start_sec": pick["clip_start_sec"], "story": info}


def match_story(segments: list, queues: dict, track: dict, *, min_clips: int = 1, max_clips: int = 1,
                ramp_start: float = 0.2, ramp_end: float = 0.85, allow_same_video: bool = False,
                sequential: bool = False, segment_exclusions: dict = None, segment_clip_reduction: dict = None,
                global_excluded: set = None, vary_count: bool = False, seed: int = 0, min_clip_len_sec: float = 1.0,
                block_roles: list = None, options: StoryOptions = None, ctx: PlanContext = None) -> tuple:
    """The 🤖 Auto story plan: (timeline, clip_count_shortfall, duration_shortfall_sec) — Classic's shape.
    Blocks where no timeline window fits fall back to Classic's chain fill."""
    import random
    from core.matching import _clips_for
    pctx = ctx or PlanContext()
    options = options or StoryOptions()
    segment_exclusions = segment_exclusions or {}
    segment_clip_reduction = segment_clip_reduction or {}
    global_excluded = global_excluded or set()
    block_roles = block_roles or [None] * len(segments)
    rng = random.Random(seed)
    counts = [_clips_for(pctx, seg.get("intensity", seg["energy"]), min_clips, max_clips, ramp_start, ramp_end,
                         vary_count, rng) for seg in segments]
    types = _section_types(track, segments, options)
    shared = RC.load_all()
    if options.track_recipes:
        shared = {**shared, **{k: RC.normalise(v, k) for k, v in options.track_recipes.items() if k in shared}}
    videos, missing = _load_videos(queues)
    if missing:
        options.notes.append(f"{len(missing)} selected video{'s have' if len(missing) != 1 else ' has'} no 🧭 timeline "
                             f"yet — only Classic's picks can use {'them' if len(missing) != 1 else 'it'}.")
    track_bpm = float(track.get("bpm") or 0.0)
    ms = float(min(max(options.music_story, 0.0), 1.0))
    smooth = 1.0 - float(min(max(options.smooth_contrast, 0.0), 1.0))

    timeline = []
    shortfall, unfilled_total = 0, 0.0
    uses = {}                      # video_id → times used so far
    prev_shot, prev_pov, prev_type = float("nan"), None, None
    confirmed = {}

    max_videos = int(options.max_videos or 0)
    for i, seg in enumerate(segments):
        if options.progress:
            try:
                options.progress(i, len(segments))
            except Exception:          # noqa: BLE001 — a progress bar must never break the plan
                pass
        d = seg["end"] - seg["start"]
        n = max(2, int(round(d / STEP)))
        n_picks = max(0, counts[i] - segment_clip_reduction.get(i, 0))
        role = block_roles[i] if i < len(block_roles) else None
        t_id = types[i]
        recipe = shared.get(t_id) or RC.default("rhythmic")
        excluded_here = segment_exclusions.get(i, set())
        excluded_all = excluded_for_block(queues, global_excluded, role, ctx=pctx)
        prev_videos = {p["video_id"] for j in (i - 1,) if j in confirmed for p in confirmed[j]}
        audio = get_block_audio_curve(track, seg["start"], seg["end"])
        shot_rule = recipe.get("shot_change", "allow")
        change_ok = shot_rule == "change" or (t_id != prev_type and shot_rule == "allow")
        shot_w = SHOT_PENALTY * smooth * {"hold": 1.0, "allow": 0.5, "change": 0.0}[shot_rule] * (0.5 if t_id != prev_type else 1.0)

        # 1. candidate windows across every available span of every video with a timeline
        cands = []
        for span in get_candidate_spans(queues, sequential, excluded_all):
            key = (span["video_id"], span["scene_id"])
            if key in excluded_here or span["remaining_sec"] < d - 1e-6:
                continue
            v = videos.get(span["video_id"])
            if v is None:
                continue
            req = recipe.get("scene_require") or []
            span_tags = {str(t).lower() for t in span.get("tags") or []}
            if req and not all(str(t).lower() in span_tags for t in req):
                continue
            starts = _window_starts(v, span, n)
            if not len(starts):
                continue
            cache = v._cum.get(("fit", i))
            if cache is None:
                cache = v._cum[("fit", i)] = _score_video(v, n, recipe, audio, track_bpm, ms)
            fit, base, story = cache
            starts = starts[starts < len(fit)]
            if not len(starts):
                continue
            prefer = PREFER_BONUS if any(str(t).lower() in span_tags for t in recipe.get("scene_prefer") or []) else 0.0
            role_b = ROLE_BONUS if (role and has_role(span, role)) or any(has_role(span, rr) for rr in recipe.get("roles") or []) else 0.0
            penalty = USE_PENALTY * uses.get(span["video_id"], 0) + (REPEAT_PENALTY if span["video_id"] in prev_videos else 0.0)
            for s in _top_spread(starts, fit, TOP_PER_SPAN, max(MIN_GAP_POINTS, n // 2)):
                total = float(fit[s]) + prefer + role_b - penalty
                start_shot = _shot_at(v, s)
                if not math.isnan(prev_shot) and not math.isnan(start_shot):
                    jump = abs(start_shot - prev_shot)
                    total -= shot_w * jump
                    if change_ok and jump >= 1.0:
                        total += CONTRAST_BONUS * max(0.0, 1.0 - smooth - 0.5) * 2
                if prev_pov is not None and smooth > 0:
                    pov_here = bool(v.pov[s])
                    if pov_here != prev_pov:
                        total -= POV_PENALTY * smooth * (0.5 if change_ok else 1.0)
                cands.append({"total": total, "fit": float(fit[s]), "base": float(base[s]), "story": float(story[s]),
                              "video_id": span["video_id"], "scene_id": span["scene_id"], "start_idx": s,
                              "offset": float(v.times[s] - span["scene_start_sec"]), "shot": start_shot,
                              "end_shot": _shot_at(v, min(s + n - 1, v.n - 1)), "pov": bool(v.pov[s])})
        cands.sort(key=lambda c: -c["total"])
        budget_spent = bool(max_videos) and len(uses) >= max_videos
        if budget_spent:
            # the video budget is spent: only the videos already in the plan (Classic's fallback fills the rest,
            # preferring them too — a new video only if nothing else can fill the block)
            cands = [c for c in cands if c["video_id"] in uses]

        # 2. fill the slots
        slots, used = [], set()
        anchor_shot = float("nan")
        for c in cands:
            if len(slots) >= n_picks:
                break
            if not allow_same_video and c["video_id"] in used:
                continue
            if slots and not math.isnan(anchor_shot) and not math.isnan(c["shot"]):
                # split-screen: pull the other panes towards the anchor's shot size — re-rank the rest lazily
                if abs(c["shot"] - anchor_shot) * PANE_PENALTY > 0.08:
                    continue
            try:
                pick = carve_span(queues, c["video_id"], c["scene_id"], c["offset"], d, sequential)
            except ValueError:
                continue
            slots.append(_slot(pick, {"type": t_id, "fit": round(c["fit"], 3), "base": round(c["base"], 3),
                                      "story": round(c["story"], 3), "shot": None if math.isnan(c["shot"]) else round(c["shot"], 2)}))
            used.add(c["video_id"])
            uses[c["video_id"]] = uses.get(c["video_id"], 0) + 1
            if len(slots) == 1:
                anchor_shot = c["shot"]
                prev_shot, prev_pov = c["end_shot"], c["pov"]
        # 3. Classic's chain fill for anything left
        for _ in range(n_picks - len(slots)):
            avoid = set(used) if not allow_same_video else set()
            chain, unfilled = None, 0.0
            if budget_spent:
                outside = {v for v in queues if v not in uses}
                chain, unfilled = fill_slot(seg.get("intensity", seg["energy"]), d, queues, sequential, excluded_here,
                                            excluded_all, avoid | outside, min_clip_len_sec, role, None, ctx=pctx)
            if not chain:
                chain, unfilled = fill_slot(seg.get("intensity", seg["energy"]), d, queues, sequential, excluded_here,
                                            excluded_all, avoid, min_clip_len_sec, role, None, ctx=pctx)
            if not chain:
                break
            slots.append({"chain": chain, "video_id": chain[0]["video_id"], "scene_id": chain[0]["scene_id"],
                          "tags": sorted({t for c in chain for t in c["tags"]}), "motion_norm": chain[0]["motion_norm"],
                          "thumbnail": chain[0]["thumbnail"],
                          "clip_duration_sec": round(sum(c["clip_duration_sec"] for c in chain), 3),
                          "offset_into_scene_sec": chain[0]["offset_into_scene_sec"],
                          "clip_start_sec": chain[0]["clip_start_sec"],
                          "story": {"type": t_id, "fit": None, "fallback": "classic"}})
            for c in chain:
                used.add(c["video_id"])
                uses[c["video_id"]] = uses.get(c["video_id"], 0) + 1
            if unfilled > 0.05:
                unfilled_total += unfilled
        if len(slots) < n_picks:
            shortfall += n_picks - len(slots)
        confirmed[i] = [c for s in slots for c in s["chain"]]
        prev_type = t_id
        # the fit cache is per block (window length and recipe change) — drop it
        for v in videos.values():
            v._cum.pop(("fit", i), None)
        timeline.append({"segment_index": i, "track_time": [round(seg["start"], 2), round(seg["end"], 2)],
                         "segment_energy": seg["energy"], "segment_intensity": seg.get("intensity", seg["energy"]),
                         "clip_count": len(slots), "split_screen": len(slots) > 1,
                         "manually_edited": i in segment_exclusions or i in segment_clip_reduction,
                         "section_type": t_id, "scenes": slots})
    if options.progress:
        try:
            options.progress(len(segments), len(segments))
        except Exception:          # noqa: BLE001
            pass
    return timeline, shortfall, unfilled_total


def plan_summary(timeline: list) -> dict:
    """For the status line: blocks per section type, share of slots from the timeline, mean fit."""
    types, fits, fallback, total = {}, [], 0, 0
    for e in timeline:
        t = e.get("section_type")
        if t:
            types[t] = types.get(t, 0) + 1
        for s in e.get("scenes") or []:
            total += 1
            info = s.get("story") or {}
            if info.get("fit") is None:
                fallback += 1
            else:
                fits.append(info["fit"])
    videos = {c["video_id"] for e in timeline for s in e.get("scenes") or [] for c in s.get("chain") or []}
    return {"types": types, "slots": total, "fallback": fallback, "videos": len(videos),
            "mean_fit": round(float(np.mean(fits)), 3) if fits else None}
