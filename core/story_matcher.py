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
       story = the window's best two wanted tags (recipe weights × tag scores) − avoid tags; scene tags
               filter / prefer;
       fit   = (1 − music_story) × base + music_story × story, then
               + the block's 🎭 role as a bonus · − junk (nobody on screen) · − variety / neighbour penalties ·
               ± the last pass: shot size / POV continuity with the previous clip, relaxed where the recipe
               says a change is fine, contrast rewarded where it says contrast is welcome;
  3. the best windows fill the block's slots (distinct videos unless allowed), carved out of the queues like
     Classic's picks, so no footage is used twice and in-order mode holds. Split-screen panes are pulled towards
     the first (anchor) pane's shot size.
A window may run across Reviewer clip edges where the footage is continuous — neighbouring usable clips that
touch with no hard cut (TransNetV2) on the join form one run; the pick is then a chain of back-to-back pieces.
Excluded clips, the library range and hard cuts stay walls.
The "shape" part of base is worked out on the footage's full-rate movement curve (15 points a second, core/sync.py)
wherever the timeline has it: every start frame is compared with the music, and the clip's start slides by up to a
quarter second so its pushes land on the hits. 🎯 Movement only (StoryOptions.movement_only) drops the recipes'
tags and movement bands altogether: the footage is picked on how its movement follows the music and how lively it
is against how loud the music is — nothing else but the safeguards. A block no window is long enough for is split at
the beat nearest its middle (up to twice) and each part gets its own recipe-scored window. Only what's still
unfilled after that (no processed videos left, footage exhausted) falls back to Classic's chain fill, so the
plan never has gaps. Each slot carries a "story" dict (type, fit, base, story) for the cards.

No Streamlit in here.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

import footage_timeline as FT
import recipes as RC
import section_types as ST
from core.footage import carve_span, get_candidate_spans, has_role
from core.matching import PlanContext, excluded_for_block, fill_slot
from core import sync as SY
from core.shape import get_block_audio_curve, resample_curve

STEP = FT.STEP
TOP_PER_SPAN = 3            # candidate windows kept per clip in a run (spread ≥ half a block apart), up to TOP_PER_RUN
TOP_PER_RUN = 12
JOIN_TOL_SEC = 0.06         # two clips touch when one ends within this of the next one's start
CUT_NEAR_SEC = 0.3          # …and no hard cut lies within this of the join
MIN_PIECE_SEC = 0.3         # a window crossing a clip edge never leaves a piece shorter than this
MIN_PART_SEC = 0.75         # a block is only split into parts at least this long
MAX_SPLIT_DEPTH = 2         # block → 2 parts → 4 parts
MIN_GAP_POINTS = 2          # …at least this many half seconds apart
JUNK_SHARE = 0.5            # nobody on screen for over half the window → heavy penalty
W_BAND, W_TREND, W_SHAPE, W_BEAT = 0.45, 0.15, 0.25, 0.15      # inside base
ONLY_W = (0.25, 0.10, 0.50, 0.15)   # 🎯 Movement only: level, trend, sync, beat — the sync leads
ONLY_BAND = 0.15            # …the footage should be this close to the music's level (both 0–1 within their library / track)
ONLY_TREND = 0.15           # …and rise or fall with it when the music moves by more than this across the block
ONLY_BEAT_LOCK = 0.6
SLIDE_SEC = STEP / 2        # a clip's start may move this far (a frame at a time) to land its pushes on the hits
AVOID_WEIGHT = 0.5
STORY_TOP = 2               # a window is judged on its best two wanted tags (e.g. a position + penetration)
PREFER_BONUS = 0.1
ROLE_BONUS = 0.1
MERGE_MAX = 1.5             # a merged clip may be up to this × the recipe's longest clip length
MERGE_TOL = 0.02            # …and is kept when its best window fits within this of the two short clips' best windows
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
    merge_blocks: bool = False          # two neighbouring blocks of one section become one longer clip when the
                                        # footage fits the longer stretch as well (the new matcher's own flow only —
                                        # Classic's cards are tied to its own cut list)
    movement_only: bool = False         # 🎯 no recipes: pick on how the movement follows the music and matches its
                                        # level; tags, scene tags, roles and the recipes' movement bands are ignored
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
        try:
            self.cuts = np.asarray(tl.cuts, np.float64)
        except AttributeError:
            self.cuts = np.zeros(0)
        pov = tl.visible("POV")
        self.pov = pov.astype(np.float64) if pov is not None else np.zeros(self.n)
        self._col = {str(t).lower(): k for k, t in enumerate(tl.tags)}
        self._cum = {}
        self._mover = False            # the full-rate movement, prepared on first use (None: the file has none)

    @property
    def mover(self):
        if self._mover is False:
            self._mover = None
            try:
                hi = np.asarray(self.tl.move_hi, np.float32)
                if len(hi) >= SY.MIN_FRAMES and np.isfinite(hi).any():
                    self._mover = SY.Mover(hi, float(self.tl.move_rate))
            except (AttributeError, ValueError, TypeError):
                pass
        return self._mover

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

def _music_trend(audio_curve) -> str:
    """Does the music rise or fall across the stretch (its last third against its first)?"""
    a = np.asarray(audio_curve, np.float64) if audio_curve is not None else np.zeros(0)
    if len(a) < 6:
        return "flat"
    k = max(1, len(a) // 3)
    d = float(a[-k:].mean() - a[:k].mean())
    return "rise" if d > ONLY_TREND else "fall" if d < -ONLY_TREND else "flat"


def _score_video(v: _Video, n: int, recipe: dict, audio_curve: np.ndarray, track_bpm: float,
                 music_story: float) -> tuple:
    """(fit, base, story) arrays over every window start of this video (NaN-free)."""
    return _score_video_x(v, n, recipe, audio_curve, track_bpm, music_story)[:3]


def _score_video_x(v: _Video, n: int, recipe: dict, audio_curve: np.ndarray, track_bpm: float, music_story: float,
                   seconds: float = None, level: float = None, movement_only: bool = False) -> tuple:
    """(fit, base, story, sync, slide) over every window start of this video. sync is the 0–1 movement-follows-music
    score, slide the seconds to move the window's start by for its best alignment (zeros without a full-rate
    curve). seconds: the stretch's true length; level: how loud the music is here, 0–1 within the track."""
    if v.n < n:
        e = np.zeros(0)
        return e, e, e, e, e
    mean_move = v.window_mean(v.move_f, n) / np.maximum(v.window_mean(v.has_move, n), 1e-6)
    mv = recipe["movement"]
    if movement_only:
        lv = 0.5 if level is None else float(level)
        band = _band_fit(mean_move, lv - ONLY_BAND, lv + ONLY_BAND)
        trend = _trend_fit(v.move_f, n, _music_trend(audio_curve))
    else:
        band = _band_fit(mean_move, float(mv["lo"]), float(mv["hi"]))
        trend = _trend_fit(v.move_f, n, mv.get("trend", "flat"))
    fine = SY.sync_scores(v.mover, v.times, n, seconds, audio_curve, SLIDE_SEC) if seconds else None
    if fine is not None:
        shape, slide = fine
    else:
        shape, slide = _shape_scores(v.move_f, n, audio_curve), np.zeros(v.n - n + 1)
    bpm_mean = v.window_mean(np.nan_to_num(v.bpm.astype(np.float64), nan=0.0), n)
    bpm_has = v.window_mean(np.isfinite(v.bpm).astype(np.float64), n)
    mbpm = np.where(bpm_has > 0, bpm_mean / np.maximum(bpm_has, 1e-6), np.nan)
    beat_fit = _fold_ratio_fit(mbpm, track_bpm) * v.window_mean(np.nan_to_num(v.per.astype(np.float64)), n)
    lock = ONLY_BEAT_LOCK if movement_only else float(recipe.get("beat_lock", 0.5))
    beat = 1.0 - lock * (1.0 - beat_fit)
    wb, wt, ws, wk = ONLY_W if movement_only else (W_BAND, W_TREND, W_SHAPE, W_BEAT)
    base = wb * band + wt * trend + ws * shape + wk * beat
    if movement_only:                  # no recipe: the music fit is the whole score
        junk = v.window_mean(v.nobody, n) > JUNK_SHARE
        fit = np.where(junk, base * 0.2, base)
        return fit, base, np.full(len(base), 0.5), shape, slide

    # A recipe's wanted tags are alternatives, not a checklist: a window can show a position and an action, not
    # twelve things at once. So the story score is the window's best STORY_TOP weighted tags against the recipe's
    # STORY_TOP heaviest weights — a long tag list doesn't water every window down.
    tags = recipe.get("tags") or {}
    rows, weights = [], []
    for tag, w in tags.items():
        tw = v.tag_window(tag, n)
        if tw is None or w <= 0:
            continue
        rows.append(float(w) * tw)
        weights.append(float(w))
    if rows:
        k = min(STORY_TOP, len(rows))
        m = np.vstack(rows)
        top = np.sort(m, axis=0)[-k:].sum(axis=0) if len(rows) > k else m.sum(axis=0)
        story = np.clip(top / max(sum(sorted(weights)[-k:]), 1e-6), 0.0, 1.0)
    else:
        story = np.full(len(base), 0.5)
    avoid = np.zeros(len(base))
    for tag in recipe.get("avoid") or []:
        tw = v.tag_window(tag, n)
        if tw is not None:
            avoid = np.maximum(avoid, tw)
    story = np.clip(story - AVOID_WEIGHT * avoid, 0.0, 1.0)

    fit = (1.0 - music_story) * base + music_story * story
    junk = v.window_mean(v.nobody, n) > JUNK_SHARE
    fit = np.where(junk, fit * 0.2, fit)
    return fit, base, story, shape, slide


def _abs_start(span: dict) -> float:
    return span["scene_start_sec"] + span["offset_sec"]


def _abs_end(span: dict) -> float:
    return span["scene_start_sec"] + span["offset_sec"] + span["remaining_sec"]


def _joined(a: dict, b: dict, cuts: np.ndarray) -> bool:
    """Do two spans of one video form continuous footage: b starts where a ends, and no hard cut on the join."""
    end = _abs_end(a)
    if abs(_abs_start(b) - end) > JOIN_TOL_SEC:
        return False
    return not (len(cuts) and bool(np.any(np.abs(cuts - end) <= CUT_NEAR_SEC)))


def _window_starts(v: _Video, a: float, b: float, d: float, n: int) -> np.ndarray:
    """Start indices whose d-second window lies inside the footage [a, b)."""
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


def _slot_chain(picks: list, info: dict) -> dict:
    """A slot from one or more picks played back to back (a window across clip edges, or a split block)."""
    first = picks[0]
    return {"chain": picks, "video_id": first["video_id"], "scene_id": first["scene_id"],
            "tags": sorted({t for p in picks for t in p["tags"]}), "motion_norm": first["motion_norm"],
            "thumbnail": first["thumbnail"], "clip_duration_sec": round(sum(p["clip_duration_sec"] for p in picks), 3),
            "offset_into_scene_sec": first["offset_into_scene_sec"], "clip_start_sec": first["clip_start_sec"], "story": info}


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
    try:
        import section_cuts as _SCUT
        beats = np.asarray(_SCUT._beats(track), float)
    except Exception:          # noqa: BLE001 — no beat grid: blocks are split at their middle
        beats = np.zeros(0)
    try:                               # what drives the music bar by bar (drums / bass / vocals / loudness)
        import audio_drive as _AD
        drv = _AD.drivers(track, beats if len(beats) else None)
    except Exception:          # noqa: BLE001 — no extra analysis: the loudness curve, as before
        _AD, drv = None, None
    ms = float(min(max(options.music_story, 0.0), 1.0))
    smooth = 1.0 - float(min(max(options.smooth_contrast, 0.0), 1.0))
    only = bool(options.movement_only)
    loud = drv["curves"].get("loudness") if drv is not None else None
    if loud is not None and len(loud) > 8:
        lv_lo, lv_hi = (float(x) for x in np.percentile(loud, [5, 95]))
    else:
        loud = None

    def music_level(t0: float, t1: float):
        """How loud the music is from t0 to t1, 0–1 within the track (None: unknown)."""
        if loud is None:
            return None
        r_ = drv["rate"]
        part = loud[int(t0 * r_):max(int(t0 * r_) + 1, int(t1 * r_))]
        return float(np.clip((float(part.mean()) - lv_lo) / max(lv_hi - lv_lo, 1e-6), 0.0, 1.0)) if len(part) else None
    if videos and not any(v.mover is not None for v in videos.values()):
        options.notes.append("The 🧭 footage timeline was built before the frame-by-frame movement was kept — movement "
                             "was matched on the half seconds. Press 🧭 Build (Media Library → Videos → Timeline) once.")

    timeline = []
    shortfall, unfilled_total = 0, 0.0
    uses = {}                      # video_id → times used so far
    prev_shot, prev_pov, prev_type = float("nan"), None, None
    last_block = []                # the picks of the block before (for the neighbour-repeat rule)
    merged_away = set()            # blocks taken into the one before them

    max_videos = int(options.max_videos or 0)
    for i, seg in enumerate(segments):
        if i in merged_away:
            continue
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
        prev_videos = {p["video_id"] for p in last_block}
        shot_rule = recipe.get("shot_change", "allow")
        change_ok = shot_rule == "change" or (t_id != prev_type and shot_rule == "allow")
        shot_w = SHOT_PENALTY * smooth * {"hold": 1.0, "allow": 0.5, "change": 0.0}[shot_rule] * (0.5 if t_id != prev_type else 1.0)

        req = [] if only else [str(t).lower() for t in recipe.get("scene_require") or []]
        pref = [] if only else [str(t).lower() for t in recipe.get("scene_prefer") or []]
        rec_roles = [] if only else (recipe.get("roles") or [])
        budget_spent = bool(max_videos) and len(uses) >= max_videos

        def search(t0: float, t1: float, key) -> list:
            """Candidate windows for the music from t0 to t1, best first. Clips of one video that touch with no hard
            cut on the join form one run, and a window may sit anywhere in a run."""
            dd = t1 - t0
            nn = max(2, int(round(dd / STEP)))
            key = (round(t0, 3), round(t1, 3), t_id)
            aud = _AD.drive_slice(drv, t0, t1) if drv is not None else get_block_audio_curve(track, t0, t1)
            by_vid = {}
            for span in get_candidate_spans(queues, sequential, excluded_all):
                vid_ = span["video_id"]
                if (vid_, span["scene_id"]) in excluded_here or vid_ not in videos:
                    continue
                if budget_spent and vid_ not in uses:
                    continue                     # the video budget is spent: only videos already in the plan
                if req:
                    have = {str(x).lower() for x in span.get("tags") or []}
                    if not all(t in have for t in req):
                        continue
                by_vid.setdefault(vid_, []).append(span)
            out = []
            for vid_, spans in by_vid.items():
                v = videos[vid_]
                spans.sort(key=_abs_start)
                runs = [[spans[0]]]
                for sp in spans[1:]:
                    if not sequential and _joined(runs[-1][-1], sp, v.cuts):
                        runs[-1].append(sp)
                    else:
                        runs.append([sp])
                penalty = USE_PENALTY * uses.get(vid_, 0) + (REPEAT_PENALTY if vid_ in prev_videos else 0.0)
                cache = None
                for run in runs:
                    ra, rb = _abs_start(run[0]), _abs_end(run[-1])
                    if rb - ra < dd - 1e-6:
                        continue
                    starts = _window_starts(v, ra, rb, dd, nn)
                    if not len(starts):
                        continue
                    if cache is None:
                        cache = v._cum.get(("fit", key))
                        if cache is None:
                            cache = v._cum[("fit", key)] = _score_video_x(v, nn, recipe, aud, track_bpm, ms, dd,
                                                                          music_level(t0, t1), only)
                    fit, base, story, sync, slide = cache
                    starts = starts[starts < len(fit)]
                    if not len(starts):
                        continue
                    keep = min(TOP_PER_RUN, TOP_PER_SPAN * len(run))
                    for s_ in _top_spread(starts, fit, keep, max(MIN_GAP_POINTS, nn // 2)):
                        ws = min(max(float(v.times[s_]) + float(slide[s_]), ra), rb - dd)     # slid onto the hits
                        we = ws + dd
                        pieces, over = [], []
                        for sp in run:
                            lo, hi = max(ws, _abs_start(sp)), min(we, _abs_end(sp))
                            if hi - lo > 1e-6:
                                pieces.append((sp["scene_id"], lo - sp["scene_start_sec"], hi - lo))
                                over.append(sp)
                        if not pieces or (len(pieces) > 1 and min(p_[2] for p_ in pieces) < MIN_PIECE_SEC):
                            continue
                        over_tags = {str(x).lower() for sp in over for x in sp.get("tags") or []}
                        prefer = PREFER_BONUS if any(t in over_tags for t in pref) else 0.0
                        role_b = ROLE_BONUS if any((role and has_role(sp, role)) or any(has_role(sp, rr) for rr in rec_roles)
                                                   for sp in over) else 0.0
                        total = float(fit[s_]) + prefer + role_b - penalty
                        start_shot = _shot_at(v, s_)
                        if not math.isnan(prev_shot) and not math.isnan(start_shot):
                            jump = abs(start_shot - prev_shot)
                            total -= shot_w * jump
                            if change_ok and jump >= 1.0:
                                total += CONTRAST_BONUS * max(0.0, 1.0 - smooth - 0.5) * 2
                        if prev_pov is not None and smooth > 0 and bool(v.pov[s_]) != prev_pov:
                            total -= POV_PENALTY * smooth * (0.5 if change_ok else 1.0)
                        out.append({"total": total, "fit": float(fit[s_]), "base": float(base[s_]), "story": float(story[s_]),
                                    "sync": float(sync[s_]), "slide": round(ws - float(v.times[s_]), 3),
                                    "video_id": vid_, "pieces": pieces, "shot": start_shot,
                                    "end_shot": _shot_at(v, min(s_ + nn - 1, v.n - 1)), "pov": bool(v.pov[s_])})
            out.sort(key=lambda c: -c["total"])
            return out

        def take(c: dict, journal: list = None):
            """Carve a candidate's pieces out of the queues; all or nothing. Returns the picks or None."""
            vid_ = c["video_id"]
            backup = copy.deepcopy(queues[vid_])
            picks = []
            try:
                for scene_id, off, dur in c["pieces"]:
                    picks.append(carve_span(queues, vid_, scene_id, off, dur, sequential))
            except ValueError:
                queues[vid_][:] = backup
                return None
            if journal is not None:
                journal.append((vid_, backup))
            return picks

        def split_fill(t0: float, t1: float, depth: int, avoid: set, journal: list):
            """No window is long enough for t0–t1: cut it at the beat nearest the middle and give each part its own
            window (splitting a part again if it must). Returns [(picks, candidate)] in order, or None."""
            if depth >= MAX_SPLIT_DEPTH or t1 - t0 < 2 * MIN_PART_SEC:
                return None
            mid = (t0 + t1) / 2
            inner = beats[(beats >= t0 + MIN_PART_SEC) & (beats <= t1 - MIN_PART_SEC)]
            if inner.size:
                mid = float(inner[int(np.argmin(np.abs(inner - mid)))])
            parts = []
            for k, (a_, b_) in enumerate(((t0, mid), (mid, t1))):
                got = None
                for c in search(a_, b_, (i, depth, k, round(a_, 3))):
                    if c["video_id"] in avoid:
                        continue
                    picks = take(c, journal)
                    if picks:
                        got = [(picks, c)]
                        break
                if got is None:
                    got = split_fill(a_, b_, depth + 1, avoid, journal)
                if got is None:
                    return None
                parts += got
            return parts

        def _info(c: dict, **extra) -> dict:
            return {"type": t_id, "fit": round(c["fit"], 3), "base": round(c["base"], 3), "story": round(c["story"], 3),
                    "sync": round(c["sync"], 3), "slide": c["slide"],
                    "shot": None if math.isnan(c["shot"]) else round(c["shot"], 2), **extra}

        # 1. whole-block windows, best first
        cands = search(seg["start"], seg["end"], (i,))

        # 1b. one longer clip instead of two short ones, when the footage holds for the longer stretch
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if (options.merge_blocks and nxt is not None and cands and types[i + 1] == t_id and counts[i + 1] == counts[i]
                and not segment_exclusions.get(i + 1) and not segment_clip_reduction.get(i + 1)
                and (nxt["end"] - seg["start"]) <= MERGE_MAX * float((recipe.get("pace") or {}).get("hi", 4.0)) + 1e-6):
            d2 = nxt["end"] - nxt["start"]
            c_next = search(nxt["start"], nxt["end"], None)
            c_long = search(seg["start"], nxt["end"], None)
            if c_next and c_long:
                apart = (d * cands[0]["fit"] + d2 * c_next[0]["fit"]) / (d + d2)
                if c_long[0]["fit"] >= apart - MERGE_TOL:
                    seg = {**seg, "end": nxt["end"],
                           "energy": round((d * seg["energy"] + d2 * nxt["energy"]) / (d + d2), 4),
                           "intensity": round((d * seg.get("intensity", seg["energy"])
                                               + d2 * nxt.get("intensity", nxt["energy"])) / (d + d2), 4)}
                    d = seg["end"] - seg["start"]
                    cands = c_long
                    merged_away.add(i + 1)

        # 2. fill the slots
        slots, used = [], set()
        anchor_shot = float("nan")
        for c in cands:
            if len(slots) >= n_picks:
                break
            if not allow_same_video and c["video_id"] in used:
                continue
            if slots and not math.isnan(anchor_shot) and not math.isnan(c["shot"]):
                # split-screen: the other panes are pulled towards the anchor's shot size
                if abs(c["shot"] - anchor_shot) * PANE_PENALTY > 0.08:
                    continue
            picks = take(c)
            if not picks:
                continue
            slots.append(_slot_chain(picks, _info(c, **({"joined": len(picks)} if len(picks) > 1 else {}))))
            used.add(c["video_id"])
            uses[c["video_id"]] = uses.get(c["video_id"], 0) + 1
            if len(slots) == 1:
                anchor_shot = c["shot"]
                prev_shot, prev_pov = c["end_shot"], c["pov"]
        # 2b. slots no whole window fits: split the block on the beat, each part scored on its own
        for _ in range(n_picks - len(slots)):
            journal = []
            parts = split_fill(seg["start"], seg["end"], 0, set(used) if not allow_same_video else set(), journal)
            if not parts:
                for vid_, backup in reversed(journal):
                    queues[vid_][:] = backup
                break
            picks = [p_ for ps, _ in parts for p_ in ps]
            w = [sum(p_["clip_duration_sec"] for p_ in ps) for ps, _ in parts]
            mean = lambda k_: float(np.average([c[k_] for _, c in parts], weights=w))    # noqa: E731
            first, last = parts[0][1], parts[-1][1]
            slots.append(_slot_chain(picks, {"type": t_id, "fit": round(mean("fit"), 3), "base": round(mean("base"), 3),
                                             "story": round(mean("story"), 3), "sync": round(mean("sync"), 3),
                                             "parts": len(parts),
                                             "shot": None if math.isnan(first["shot"]) else round(first["shot"], 2)}))
            for _, c in parts:
                used.add(c["video_id"])
                uses[c["video_id"]] = uses.get(c["video_id"], 0) + 1
            if len(slots) == 1:
                anchor_shot = first["shot"]
                prev_shot, prev_pov = last["end_shot"], last["pov"]
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
        last_block = [c for s in slots for c in s["chain"]]
        prev_type = t_id
        # window scores are kept per stretch of music; drop the ones this block has passed
        for v in videos.values():
            for key_ in [k_ for k_ in v._cum if isinstance(k_, tuple) and len(k_) == 2 and k_[0] == "fit"
                         and k_[1][1] <= seg["end"] + 1e-6]:
                del v._cum[key_]
        timeline.append({"segment_index": len(timeline) if options.merge_blocks else i,
                         "driver": _AD.driver_at(drv, (seg["start"] + seg["end"]) / 2) if drv is not None else None,
                         "merged": (i + 1) in merged_away,
                         "track_time": [round(seg["start"], 2), round(seg["end"], 2)],
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
    types, fits, fallback, total, split, joined = {}, [], 0, 0, 0, 0
    syncs, sync_w = [], []
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
                if info.get("sync") is not None:
                    syncs.append(info["sync"])
                    sync_w.append(max(e["track_time"][1] - e["track_time"][0], 1e-3))
            split += 1 if info.get("parts") else 0
            joined += 1 if info.get("joined") else 0
    videos = {c["video_id"] for e in timeline for s in e.get("scenes") or [] for c in s.get("chain") or []}
    drivers = {}
    for e in timeline:
        if e.get("driver"):
            drivers[e["driver"]] = drivers.get(e["driver"], 0) + 1
    lens = [e["track_time"][1] - e["track_time"][0] for e in timeline]
    return {"types": types, "slots": total, "fallback": fallback, "videos": len(videos), "split": split, "joined": joined,
            "merged": sum(1 for e in timeline if e.get("merged")), "drivers": drivers,
            "clip_sec": [round(min(lens), 2), round(float(np.median(lens)), 2), round(max(lens), 2)] if lens else None,
            "mean_fit": round(float(np.mean(fits)), 3) if fits else None,
            # how well the movement follows the music over the whole plan (time-weighted; 0.5 = unrelated, 1 = together)
            "sync": round(float(np.average(syncs, weights=sync_w)), 3) if syncs else None}


def sync_report(timeline: list, track: dict, context_sec: float = 1.0) -> dict:
    """The plan's movement against its music, clip by clip, at the movement curve's rate — for the 📥 sync data file
    (the numbers a "does this clip move with the music?" question is answered from). Per piece of footage: the
    stretch of music, the movement as played plus a second either side, the music's curves over the same stretch
    (the drive curve the matcher compared with, and every stem curve the analysis has) and the 0–1 sync as played."""
    import audio_drive as AD
    index = FT.load_index()
    try:
        import section_cuts as _SCUT
        beats = np.asarray(_SCUT._beats(track), float)
    except Exception:          # noqa: BLE001
        beats = np.zeros(0)
    drv = AD.drivers(track, beats if len(beats) else None)
    a_rate = float(drv["rate"])
    r2 = lambda v: [None if not np.isfinite(x) else round(float(x), 3) for x in np.asarray(v, float).ravel()]    # noqa: E731
    tls, rows, rate = {}, [], None
    for e in timeline:
        t = float(e["track_time"][0])
        for slot in e.get("scenes") or []:
            tt = t
            for link in slot.get("chain") or []:
                dur = float(link["clip_duration_sec"])
                vid = link["video_id"]
                if vid not in tls:
                    tls[vid] = FT.load(vid, index=index)
                tl = tls[vid]
                row = {"music": [round(tt, 3), round(tt + dur, 3)], "type": e.get("section_type"), "driver": e.get("driver"),
                       "video": vid, "scene": link.get("scene_id"), "clip_start": round(float(link["clip_start_sec"]), 3),
                       "seconds": round(dur, 3), **{k: (slot.get("story") or {}).get(k) for k in ("fit", "base", "sync", "slide")}}
                try:
                    hi, rate = np.asarray(tl.move_hi, np.float32), float(tl.move_rate)
                except (AttributeError, TypeError):
                    hi = None
                if hi is not None and len(hi):
                    p0, m = int(round(float(link["clip_start_sec"]) * rate)), int(round(dur * rate))
                    c = int(round(context_sec * rate))
                    move = hi[max(p0, 0):p0 + m]
                    lo, hi_ = int(tt * a_rate), max(int(tt * a_rate) + 1, int((tt + dur) * a_rate))
                    row["move"] = r2(move)
                    row["move_before"], row["move_after"] = r2(hi[max(p0 - c, 0):max(p0, 0)]), r2(hi[p0 + m:p0 + m + c])
                    row["music_curves"] = {"drive": r2(SY.pool(drv["curve"][lo:hi_], len(move))),
                                           **{k: r2(SY.pool(v[lo:hi_], len(move))) for k, v in drv["curves"].items()}}
                    played = SY.sync_of(move, drv["curve"][lo:hi_], rate)
                    row["sync_played"] = None if played is None else round(played, 3)
                rows.append(row)
                tt += dur
    played = [r_["sync_played"] for r_ in rows if r_.get("sync_played") is not None]
    return {"track_id": track.get("track_id"), "bpm": track.get("bpm"), "rate": rate, "music_rate": a_rate,
            "sync_played_mean": round(float(np.mean(played)), 3) if played else None,
            "how_to_read": "sync: 1 = the movement follows the music, 0.5 = unrelated, 0 = opposite. move / music_curves "
                           "have one point per 1/rate s from the clip's first frame.",
            "clips": rows}
