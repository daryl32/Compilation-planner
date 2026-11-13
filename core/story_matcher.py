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
is against how loud the music is, plus 🪀 bounce ↔ beat (bouncy footage where the drums play, smooth footage
where they don't — core/sync.bounce / beat_presence) — nothing else but the safeguards. A block no window is long enough for is split at
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
FROZEN_SHARE = 0.1          # the picture frozen (core/sync.frozen) for over this share of the window → the same penalty
W_BAND, W_TREND, W_SHAPE, W_BEAT = 0.45, 0.15, 0.25, 0.15      # inside base
ONLY_W = (0.25, 0.10, 0.50, 0.15)   # 🎯 Movement only: level, trend, sync, beat — the sync leads
ONLY_FEEL = 0.2             # …and this share of the whole goes to 🪀 bounce ↔ beat (the four above share the rest):
#                             bouncy footage where the music has a beat, smooth footage where a voice or pad carries it
BOUNCE_FULL = 0.5           # a bounce this strong (core/sync.bounce) is as bouncy as real footage gets (01 AURA, 11 Nov)
BOUNCE_OFF_TEMPO = 0.6      # a bounce a quarter octave off the track's tempo (or its half / double) counts this much
SEEN_SHARE = 0.65           # 👁️ how lively and how bouncy a window is: this share from the on-screen motion
#                             (core/sync.pixel_curve), the rest from the person curve — where the video has both
SEEN_SYNC = 0.5             # …and this share of the sync
RANGE_LO, RANGE_HI = 0.1, 0.9          # 🎚️ full range: a track's calmest blocks ask for this level, its loudest for that
RANGE_MIN_SPREAD = 0.1      # a track whose blocks differ by less than this is nearly flat: it is stretched less,
#                             so noise isn't blown up into a "structure" it doesn't have
SPEED_BPM = (40.0, 200.0)   # 🥁 bounces a minute a body can show and core/sync.bounce can read: the speeds on offer
#                             are the track's tempo, half and double, where they fall inside this
SPEED_OCTAVES = 2.0         # a bounce this many octaves from the wanted speed loses the setting's whole share
LEVEL_NO_BEAT = 0.25        # with no beat the footage may be this much calmer than the loudness alone asks for
ONLY_BAND = 0.15            # …the footage should be this close to the music's level (both 0–1 within their library / track)
ONLY_TREND = 0.15           # …and rise or fall with it when the music moves by more than this across the block
ONLY_BEAT_LOCK = 0.6
SLIDE_SEC = STEP / 2        # a clip's start may move this far (a frame at a time) to land its pushes on the hits
# 🧪 Movement lab (13 Nov 2026) — test switches, each off unless asked for (🪶 flowing movement was tried and
# scrapped the same day: it picked stiller, more distant footage and looked worse)
OFF_BEAT = 0.5              # a stretch whose beat presence (core/sync.beat_presence) is under this is "off the beat"
PHRASE_SMOOTH_SEC = 1.0     # 🌊 phrase scale: movement and music are both smoothed this wide — swells, not pushes
PHRASE_SLIDE_SEC = 0.5      # …and the clip's start may move this far to land a swell on a swell
PHRASE_MIN_SEC = 3.0        # …only for stretches at least this long (a shorter one has no phrase to compare)
SHOT_FIX_POWER = 0.5        # 🔍 shot size: each shot size's level is pulled this far (0 = not at all, 1 = all the way)
SHOT_FIX_RANGE = (0.7, 1.5)   # towards the level of the footage as a whole — never by more than this
SHOT_FIX_MIN_POINTS = 240   # a shot size with under two minutes of footage in play is left as measured
SHOT_NAMES = ("extreme close-up", "close-up", "medium", "wide")
AVOID_WEIGHT = 0.5
ALT_KEEP = 24                # next-best windows kept per block for the 🔁 alternatives (up to ALT_PER_VIDEO per video:
ALT_PER_VIDEO = 2           # a later block may have taken one of them by the time it is offered)
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
CONTRAST_QUIET = 0.12       # 🎯 Movement only: with the slider fully at Contrast, what a change of shot size earns per
#                             step (up to two: close-up → wide) in a block with no beat at all — and nothing on a full
#                             beat. Found on Daryl's footage, 13 Nov (15 videos, 8 tracks): wide shots suit the quiet
#                             passages, where they fit nearly as well as the winner (fits 0.80–0.94), and not the
#                             beat, where they are the worst picks of a plan (fits ≈ 0.55). 0.06–0.08 already brings
#                             1–5 wide clips a track; from 0.2 the picks get worse (off-beat fit −0.03 to −0.05).


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
    level_range: float = 0.0            # 🎚️ 🎯 Movement only: how far each track's levels are stretched to the whole
                                        # calm-to-lively range (0 = as measured, 1 = its calmest block asks for calm
                                        # footage and its loudest for the liveliest); the order is kept
    bounce_speed: float = 0.0           # 🥁 🎯 Movement only: how strongly the bounce's speed follows the rhythm — a
                                        # sparse beat prefers slow, long bounces (half the tempo), a busy one fast
                                        # bounces (the tempo or double). 0 = any of the three counts the same
    stem_follow: bool = False           # 🎼 🎯 Movement only, off the beat: the movement is compared with the layer that
                                        # leads there (voice, instruments or bass — audio_drive.lead_slice), not
                                        # with the bar's driver, which is often just the loudness
    phrase_scale: bool = False          # 🌊 🎯 Movement only, off the beat: compared at the scale of a phrase (both
                                        # curves smoothed over a second, the start free to move half a second)
    shot_level: bool = False            # 🔍 🎯 Movement only: how lively footage is, is judged within its shot size
                                        # (a close-up reads livelier than the same movement in a wide shot)
    merge_limit_sec: float = 0.0        # merge_blocks: the longest clip two blocks may become (0: the recipe's pace)
    lab_notes: dict = field(default_factory=dict)        # filled in: what the 🧪 switches did (for the 📥 sync data)
    curves_of: object = None            # curves_of(video_id) → {scene_id: {"fps", "values"}}: the catalogue's on-screen
                                        # motion. 🎯 Movement only then judges the footage on it too (None: person curve only)
    video_stats: dict = field(default_factory=dict)      # filled in: how every video in play fared (see _video_stats)
    alternatives: dict = field(default_factory=dict)     # filled in: {timeline entry index: the block's next-best
                                                         # windows, one per video, best first} (see _alt)
    keep: dict | None = None            # {segment index: timeline entry} — blocks kept exactly as they are (only the
                                        # others are matched again; their footage stays out of reach)
    force: dict | None = None           # {segment index: candidate} — this window goes in that block first
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
        # A timeline built from a student export without people counts says 0 everywhere (footage_timeline fills the
        # gap with zeros) — "unknown", not "nobody". 59 of Daryl's 162 videos were like that (11 Nov): every window
        # took the nobody-on-screen penalty and none was ever picked. Without counts, a point has nobody on it only
        # where the pose tracking found nobody either.
        self.people_known = bool((self.people > 0).any())
        self.nobody = ((self.people == 0).astype(np.float64) if self.people_known
                       else (~np.isfinite(self.move)).astype(np.float64))
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
        self._bounce = False
        self._frozen = False
        self.seen = None               # the on-screen motion at the person curve's rate (set_seen)
        self.seen_mover = None
        self.seen_level = None         # …per half second, 0–1 (NaN where it has none)
        self.seen_bounce = None
        self.seen_top = None
        self._mover_slow = False       # 🌊 the same two curves smoothed to the scale of a phrase
        self._seen_slow = False
        self.shot_gain = None          # 🔍 per half second: what its shot size multiplies the level by (None: nothing)

    @property
    def mover_slow(self):
        if self._mover_slow is False:
            self._mover_slow = None
            try:
                hi = np.asarray(self.tl.move_hi, np.float32)
                if len(hi) >= SY.MIN_FRAMES and np.isfinite(hi).any():
                    self._mover_slow = SY.Mover(hi, float(self.tl.move_rate), PHRASE_SMOOTH_SEC)
            except (AttributeError, ValueError, TypeError):
                pass
        return self._mover_slow

    @property
    def seen_mover_slow(self):
        if self._seen_slow is False:
            self._seen_slow = None
            if self.seen is not None:
                self._seen_slow = SY.Mover(self.seen, float(self.tl.move_rate), PHRASE_SMOOTH_SEC)
        return self._seen_slow

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

    @property
    def bounce(self):
        """(strength 0–1, bounces a minute) per half second from the full-rate movement; None without it."""
        if self._bounce is False:
            self._bounce = None
            try:
                hi = np.asarray(self.tl.move_hi, np.float32)
                if len(hi) >= SY.MIN_FRAMES and np.isfinite(hi).any():
                    self._bounce = SY.bounce(hi, float(self.tl.move_rate), self.times)
            except (AttributeError, ValueError, TypeError):
                pass
        return self._bounce

    def set_seen(self, scenes: list) -> None:
        """Give the video its on-screen motion: scenes = [(start seconds, fps, values)] (see SY.pixel_curve)."""
        try:
            hi = np.asarray(self.tl.move_hi, np.float32)
            rate = float(self.tl.move_rate)
        except (AttributeError, ValueError, TypeError):
            return
        if len(hi) < SY.MIN_FRAMES:
            return
        px = SY.pixel_curve(scenes, len(hi), rate)
        if px is None or not np.isfinite(px).any():
            return
        self.seen = px
        try:                           # the video's own "busy" level, in the catalogue's raw units — what 1.0 stands for
            self.seen_top = float(np.percentile(np.concatenate([np.asarray(v_, np.float64) for _, _, v_ in scenes
                                                                if len(v_)]), 95))
        except ValueError:
            self.seen_top = None
        self.seen_mover = SY.Mover(px, rate)
        self._seen_slow = False
        self.seen_level = np.clip(SY.half_second_level(px, rate, self.times, STEP), 0.0, 1.0)
        self.seen_bounce = SY.bounce(px, rate, self.times)
        for k in [k for k in self._cum if k == "level" or (isinstance(k, tuple) and k and k[0] in ("bounce", "level"))]:
            del self._cum[k]

    def level_series(self, shot_fix: bool = False) -> tuple:
        """(how lively each half second is 0–1, 1 where that is known) — the person curve, with the on-screen
        motion's share where the video has it. shot_fix: 🔍 each point's level × its shot size's gain (shot_gain)."""
        if shot_fix and self.shot_gain is not None:
            if ("level", "shot") not in self._cum:
                lv, has = self.level_series()
                self._cum[("level", "shot")] = (np.clip(lv * self.shot_gain, 0.0, 1.0), has)
            return self._cum[("level", "shot")]
        if "level" not in self._cum:
            lv, has = self.move_f, self.has_move
            if self.seen_level is not None:
                there = np.isfinite(self.seen_level)
                sl = np.nan_to_num(self.seen_level)
                both = there & (self.has_move > 0)
                lv = np.where(both, (1.0 - SEEN_SHARE) * self.move_f + SEEN_SHARE * sl, np.where(there, sl, self.move_f))
                has = np.maximum(self.has_move, there.astype(np.float64))
            self._cum["level"] = (lv, has)
        return self._cum["level"]

    @property
    def frozen(self):
        """Share of each half second with the picture frozen (zeros without the full-rate movement)."""
        if self._frozen is False:
            self._frozen = np.zeros(self.n)
            try:
                hi = np.asarray(self.tl.move_hi, np.float32)
                if len(hi) >= SY.MIN_FRAMES:
                    self._frozen = SY.frozen(hi, float(self.tl.move_rate), self.times, STEP)
            except (AttributeError, ValueError, TypeError):
                pass
        return self._frozen

    def bounce_window(self, n: int, track_bpm: float, speed: tuple = None) -> np.ndarray | None:
        """How bouncy each n-point window is, 0–1 (1 = a strong, regular bounce on the track's tempo).
        speed = (bounces a minute the music wants, 0–1 how much that counts): a bounce far from that speed counts
        for less — half the tempo against double it, which otherwise count the same."""
        if self.bounce is None:
            return None
        want, share = (float(speed[0]), float(speed[1])) if speed and speed[0] and speed[1] > 0 else (0.0, 0.0)
        key = ("bounce", n, round(float(track_bpm), 2), round(want, 1), round(share, 3))
        if key not in self._cum:
            def right_speed(bpm_):
                if share <= 0:
                    return 1.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    off = np.abs(np.log2(np.where(np.isfinite(bpm_) & (bpm_ > 0), bpm_, want) / want))
                return 1.0 - share * np.clip(off / SPEED_OCTAVES, 0.0, 1.0)
            strength, bpm = self.bounce
            on_tempo = _fold_ratio_fit(bpm, track_bpm) if track_bpm > 0 else np.ones(len(bpm))
            eff = (np.clip(strength / BOUNCE_FULL, 0.0, 1.0) * (BOUNCE_OFF_TEMPO + (1.0 - BOUNCE_OFF_TEMPO) * on_tempo)
                   * right_speed(bpm))
            if self.seen_bounce is not None:           # what is SEEN bouncing counts for more than the limbs
                s2, b2 = self.seen_bounce
                t2 = _fold_ratio_fit(b2, track_bpm) if track_bpm > 0 else np.ones(len(b2))
                eff2 = (np.clip(s2 / BOUNCE_FULL, 0.0, 1.0) * (BOUNCE_OFF_TEMPO + (1.0 - BOUNCE_OFF_TEMPO) * t2)
                        * right_speed(b2))
                there = np.isfinite(self.seen_level)
                eff = np.where(there, (1.0 - SEEN_SHARE) * eff + SEEN_SHARE * eff2, eff)
            self._cum[key] = self.window_mean(eff, n)
        return self._cum[key]

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
                   seconds: float = None, level: float = None, movement_only: bool = False,
                   beat: float = None, speed: tuple = None, phrase: bool = False,
                   shot_fix: bool = False) -> tuple:
    """(fit, base, story, sync, slide) over every window start of this video. sync is the 0–1 movement-follows-music
    score, slide the seconds to move the window's start by for its best alignment (zeros without a full-rate
    curve). seconds: the stretch's true length; level: how loud the music is here, 0–1 within the track; beat: is
    there a beat here, 0–1 (None: unknown) — 🎯 Movement only then also asks for bouncy footage on a beat and smooth
    footage off it. 🧪 phrase: the sync is worked out at the scale of a phrase (the slow movers, a wider slide);
    shot_fix: the level is the one corrected for shot size. Both 🎯 Movement only."""
    if v.n < n:
        e = np.zeros(0)
        return e, e, e, e, e
    mean_move = v.window_mean(v.move_f, n) / np.maximum(v.window_mean(v.has_move, n), 1e-6)
    mv = recipe["movement"]
    if movement_only:
        lv = 0.5 if level is None else float(level)
        if beat is not None:           # loud but beatless (a pad, a held voice) wants calmer footage than a loud beat
            lv *= 1.0 - LEVEL_NO_BEAT * (1.0 - float(beat))
        lv_s, has_s = v.level_series(shot_fix)
        mean_move = v.window_mean(lv_s * has_s, n) / np.maximum(v.window_mean(has_s, n), 1e-6)
        band = _band_fit(mean_move, lv - ONLY_BAND, lv + ONLY_BAND)
        trend = _trend_fit(lv_s * has_s, n, _music_trend(audio_curve))
    else:
        band = _band_fit(mean_move, float(mv["lo"]), float(mv["hi"]))
        trend = _trend_fit(v.move_f, n, mv.get("trend", "flat"))
    slow = bool(phrase and movement_only and seconds and v.mover_slow is not None)
    if slow:
        fine = SY.sync_scores(v.mover_slow, v.times, n, seconds, audio_curve, PHRASE_SLIDE_SEC,
                              v.seen_mover_slow, SEEN_SYNC)
    else:
        fine = SY.sync_scores(v.mover, v.times, n, seconds, audio_curve, SLIDE_SEC,
                              v.seen_mover if movement_only else None, SEEN_SYNC) if seconds else None
    if fine is not None:
        shape, slide = fine
    else:
        shape, slide = _shape_scores(v.move_f, n, audio_curve), np.zeros(v.n - n + 1)
    bpm_mean = v.window_mean(np.nan_to_num(v.bpm.astype(np.float64), nan=0.0), n)
    bpm_has = v.window_mean(np.isfinite(v.bpm).astype(np.float64), n)
    mbpm = np.where(bpm_has > 0, bpm_mean / np.maximum(bpm_has, 1e-6), np.nan)
    beat_fit = _fold_ratio_fit(mbpm, track_bpm) * v.window_mean(np.nan_to_num(v.per.astype(np.float64)), n)
    lock = ONLY_BEAT_LOCK if movement_only else float(recipe.get("beat_lock", 0.5))
    tempo = 1.0 - lock * (1.0 - beat_fit)
    wb, wt, ws, wk = ONLY_W if movement_only else (W_BAND, W_TREND, W_SHAPE, W_BEAT)
    base = wb * band + wt * trend + ws * shape + wk * tempo
    if movement_only:                  # no recipe: the music fit is the whole score
        bw = v.bounce_window(n, track_bpm, speed) if beat is not None else None
        if bw is not None:
            base = (1.0 - ONLY_FEEL) * base + ONLY_FEEL * (1.0 - np.abs(float(beat) - bw[:len(base)]))
        junk = (v.window_mean(v.nobody, n) > JUNK_SHARE) | (v.window_mean(v.frozen, n) > FROZEN_SHARE)
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
    junk = (v.window_mean(v.nobody, n) > JUNK_SHARE) | (v.window_mean(v.frozen, n) > FROZEN_SHARE)
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


def _alt(c: dict) -> dict:
    """A candidate window as kept for the 🔁 alternatives: everything needed to show it and to put it in."""
    return {"video_id": c["video_id"], "pieces": [list(p_) for p_ in c["pieces"]],
            **{k: (None if isinstance(c.get(k), float) and math.isnan(c[k]) else c.get(k))
               for k in ("total", "fit", "base", "story", "sync", "slide", "shot", "end_shot", "pov")}}


def _carve_entry(queues: dict, entry: dict, sequential: bool) -> None:
    """Take a kept block's footage out of the queues, so no other block can use it again."""
    for slot in entry.get("scenes") or []:
        for link in slot.get("chain") or []:
            try:
                carve_span(queues, link["video_id"], link["scene_id"], float(link["offset_into_scene_sec"]),
                           float(link["clip_duration_sec"]), sequential)
            except (ValueError, KeyError):
                pass                   # already gone (a range or exclusion changed): nothing to take


def _slot_chain(picks: list, info: dict) -> dict:
    """A slot from one or more picks played back to back (a window across clip edges, or a split block)."""
    first = picks[0]
    return {"chain": picks, "video_id": first["video_id"], "scene_id": first["scene_id"],
            "tags": sorted({t for p in picks for t in p["tags"]}), "motion_norm": first["motion_norm"],
            "thumbnail": first["thumbnail"], "clip_duration_sec": round(sum(p["clip_duration_sec"] for p in picks), 3),
            "offset_into_scene_sec": first["offset_into_scene_sec"], "clip_start_sec": first["clip_start_sec"], "story": info}


def _shot_gains(videos: dict) -> dict:
    """🔍 {shot size 0–3: what its level is multiplied by} over the videos in play, and each video's shot_gain set
    from it. The movement measures read a close-up livelier than the same movement in a wide shot (person curve,
    11 Nov: median 0.32 close-up, 0.30 medium, 0.16 wide; the on-screen motion reads wide shots low as well), so a
    wide shot hardly ever looked lively enough for a loud passage and a slow close-up looked too lively for a quiet
    one. Each shot size's typical (median) level is pulled SHOT_FIX_POWER of the way to the footage's as a whole —
    not all the way: part of the difference is real. {} when there is too little to go on — under two shot sizes
    with enough footage — and nothing is changed."""
    per, every = {}, []
    for v in videos.values():
        lv, has = v.level_series()
        sh = np.asarray(v.shot, np.float64)
        ok = (np.asarray(has) > 0) & np.isfinite(sh)
        if not ok.any():
            continue
        cls = np.clip(np.round(sh[ok]), 0, 3).astype(int)
        vals = np.asarray(lv, np.float64)[ok]
        every.append(vals)
        for k in range(4):
            if (cls == k).any():
                per.setdefault(k, []).append(vals[cls == k])
    if not every:
        return {}
    meds = {}
    for k, parts in per.items():
        x = np.concatenate(parts)
        med = float(np.median(x))
        if len(x) >= SHOT_FIX_MIN_POINTS and med > 1e-6:
            meds[k] = (med, len(x))
    if len(meds) < 2:
        return {}
    # "the footage as a whole": the shot sizes' typical levels averaged by how much footage each has (a geometric
    # mean — the gains are ratios), so no one shot size is the yardstick the others are measured against
    whole = float(np.exp(sum(n_ * math.log(m_) for m_, n_ in meds.values()) / sum(n_ for _, n_ in meds.values())))
    gains = {k: round(float(np.clip((whole / m_) ** SHOT_FIX_POWER, *SHOT_FIX_RANGE)), 3) for k, (m_, _) in meds.items()}
    if not gains or all(abs(g - 1.0) < 1e-3 for g in gains.values()):
        return {}
    table = np.array([gains.get(k, 1.0) for k in range(4)])
    for v in videos.values():
        sh = np.asarray(v.shot, np.float64)
        v.shot_gain = np.where(np.isfinite(sh), table[np.clip(np.round(np.nan_to_num(sh)), 0, 3).astype(int)], 1.0)
        v._cum.pop(("level", "shot"), None)
    return gains


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
    usable_at_start = {vid: (len(spans), sum(float(sp.get("remaining_sec") or 0.0) for sp in spans))
                       for vid, spans in queues.items()}
    if missing:
        options.notes.append(f"{len(missing)} selected video{'s have' if len(missing) != 1 else ' has'} no 🧭 timeline "
                             f"yet — only Classic's picks can use {'them' if len(missing) != 1 else 'it'}.")
    if options.movement_only and options.curves_of is not None:
        # 👁️ the on-screen motion of every video, from its usable clips' frame-difference curves
        for vid, v in videos.items():
            try:
                curves = options.curves_of(vid) or {}
                seen_ids, scenes = set(), []
                for sp in queues.get(vid) or []:
                    c = curves.get(sp["scene_id"])
                    if c is None or sp["scene_id"] in seen_ids:
                        continue
                    seen_ids.add(sp["scene_id"])
                    scenes.append((sp["scene_start_sec"], c.get("fps"), c.get("values")))
                if scenes:
                    v.set_seen(scenes)
            except Exception:          # noqa: BLE001 — a video without usable curves is matched on the person curve
                pass
        if videos and not any(v.seen is not None for v in videos.values()):
            options.notes.append("No on-screen motion curves were found for these videos — movement was judged on the "
                                 "person curve alone.")
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
    # 🧪 Movement lab — these three are 🎯 Movement only
    lab_stem, lab_phrase = only and bool(options.stem_follow), only and bool(options.phrase_scale)
    lab_shot = only and bool(options.shot_level)
    leads = {}
    if lab_stem and drv is not None:
        try:
            leads = _AD.lead_curves(track, drv)
        except Exception:          # noqa: BLE001 — a track without the layers: the drive curve, as before
            leads = {}
        if not leads:
            options.notes.append("🎼 Follow the lead: this track's analysis has no voice, instrument or bass layer — "
                                 "the usual curve was used.")
    if lab_shot:
        gains = _shot_gains(videos)
        options.lab_notes["shot_gains"] = gains
        if gains:
            options.notes.append("🔍 Shot size: level × " + " · ".join(f"{SHOT_NAMES[k]} {g:.2f}" for k, g in sorted(gains.items())))
        else:
            options.notes.append("🔍 Shot size: too little footage with a shot size in play — levels left as measured.")
    lab_used = {"follows": {}, "phrase": 0, "off_beat": 0}
    options.lab_notes["used"] = lab_used
    loud = drv["curves"].get("loudness") if drv is not None else None
    if loud is not None and len(loud) > 8:
        lv_lo, lv_hi = (float(x) for x in np.percentile(loud, [5, 95]))
    else:
        loud = None

    def raw_level(t0: float, t1: float):
        """How loud the music is from t0 to t1, 0–1 within the track (None: unknown)."""
        if loud is None:
            return None
        r_ = drv["rate"]
        part = loud[int(t0 * r_):max(int(t0 * r_) + 1, int(t1 * r_))]
        return float(np.clip((float(part.mean()) - lv_lo) / max(lv_hi - lv_lo, 1e-6), 0.0, 1.0)) if len(part) else None

    # 🎚️ The track's own range. A block's level is the mean of a spiky curve, so even the loudest block sits well
    # under 1 and the loud sections bunch together: over seven tracks (11 Nov) none asked for more than 0.82, and
    # AURA spent 46 % of its length within 0.57–0.68 — verse, chorus and drop asking for the same footage. The
    # blocks' levels are stretched so the track's calmest (5th percentile, by time) ask for RANGE_LO and its
    # loudest (95th) for RANGE_HI.
    stretch, blk_lo, blk_hi = 0.0, 0.0, 1.0
    want = float(min(max(options.level_range or 0.0, 0.0), 1.0))
    if only and want > 0 and loud is not None and segments:
        vals = [(raw_level(sg["start"], sg["end"]), sg["end"] - sg["start"]) for sg in segments]
        vals = [(v_, w_) for v_, w_ in vals if v_ is not None and w_ > 0]
        if len(vals) >= 4:
            v_ = np.array([a_ for a_, _ in vals])
            w_ = np.array([b_ for _, b_ in vals])
            order = np.argsort(v_)
            cum = np.cumsum(w_[order]) / w_.sum()
            blk_lo, blk_hi = (float(x) for x in np.interp([0.05, 0.95], cum, v_[order]))
            if blk_hi - blk_lo > 1e-3:
                stretch = want * min(1.0, (blk_hi - blk_lo) / RANGE_MIN_SPREAD)

    def music_level(t0: float, t1: float):
        """The level asked of the footage from t0 to t1: how loud the music is, 0–1 within the track, with the
        track's own range stretched as far as options.level_range says (None: unknown)."""
        r = raw_level(t0, t1)
        if r is None or stretch <= 0:
            return r
        full = RANGE_LO + (RANGE_HI - RANGE_LO) * float(np.clip((r - blk_lo) / (blk_hi - blk_lo), 0.0, 1.0))
        return (1.0 - stretch) * r + stretch * full
    drums = drv["curves"].get("drums") if drv is not None else None

    def music_beat(t0: float, t1: float):
        """Is there a beat from t0 to t1, 0–1 (None: the analysis has no drum curve)."""
        if drums is None:
            return None
        r_ = drv["rate"]
        return SY.beat_presence(drums[int(t0 * r_):max(int(t0 * r_) + 1, int(t1 * r_))], _AD.DRUM_LEVEL)
    # 🥁 How busy the rhythm is, block by block, against the rest of the track's beat sections → the bounce speed
    # the music wants there: the slowest speed on offer where the beat is sparsest, the fastest where it is busiest.
    speed_share = float(min(max(options.bounce_speed or 0.0, 0.0), 1.0)) if only else 0.0
    speeds = [m_ * track_bpm for m_ in (0.5, 1.0, 2.0) if SPEED_BPM[0] <= m_ * track_bpm <= SPEED_BPM[1]] if track_bpm > 0 else []
    rates = []

    def music_rate(t0: float, t1: float):
        if drums is None:
            return None
        r_ = drv["rate"]
        return SY.hit_rate(drums[int(t0 * r_):max(int(t0 * r_) + 1, int(t1 * r_))], r_, _AD.DRUM_LEVEL)
    if speed_share > 0 and len(speeds) >= 2 and drums is not None:
        for sg in segments:
            b_, r_ = music_beat(sg["start"], sg["end"]), music_rate(sg["start"], sg["end"])
            if b_ is not None and b_ >= 0.7 and r_ is not None:
                rates.append((r_, sg["end"] - sg["start"]))
        rates.sort()

    def music_speed(t0: float, t1: float):
        """(bounces a minute wanted, share) for t0–t1, or None: no setting, no beat here, or nothing to rank against."""
        if len(rates) < 4:
            return None
        b_, r_ = music_beat(t0, t1), music_rate(t0, t1)
        if b_ is None or b_ < 0.7 or r_ is None:
            return None
        total = sum(w_ for _, w_ in rates)
        below = sum(w_ for x_, w_ in rates if x_ < r_) + 0.5 * sum(w_ for x_, w_ in rates if x_ == r_)
        q = below / total if total > 0 else 0.5                  # where this block's rhythm ranks in the track, 0–1
        lo_, hi_ = math.log2(speeds[0]), math.log2(speeds[-1])
        return 2.0 ** (lo_ + q * (hi_ - lo_)), speed_share
    if videos and not any(v.mover is not None for v in videos.values()):
        options.notes.append("The 🧭 footage timeline was built before the frame-by-frame movement was kept — movement "
                             "was matched on the half seconds. Press 🧭 Build (Media Library → Videos → Timeline) once.")

    timeline = []
    stats, stats_blocks = {}, [0]  # video_id → its ranks block by block (for options.video_stats)
    shortfall, unfilled_total = 0, 0.0
    uses = {}                      # video_id → times used so far
    prev_shot, prev_pov, prev_type = float("nan"), None, None
    last_block = []                # the picks of the block before (for the neighbour-repeat rule)
    merged_away = set()            # blocks taken into the one before them

    keep = dict(options.keep or {})
    force = dict(options.force or {})
    for entry in keep.values():        # 🔁 kept blocks: their footage is used, their videos count as used
        _carve_entry(queues, entry, sequential)
        for slot in entry.get("scenes") or []:
            for link in slot.get("chain") or []:
                uses[link["video_id"]] = uses.get(link["video_id"], 0) + 1

    max_videos = int(options.max_videos or 0)
    for i, seg in enumerate(segments):
        if i in merged_away:
            continue
        if i in keep:                  # kept as it is — but it is still the block before the next one
            entry = {**keep[i], "segment_index": len(timeline)}
            timeline.append(entry)
            last_block = [c for s_ in entry.get("scenes") or [] for c in s_.get("chain") or []]
            if entry.get("scenes"):
                first_ = entry["scenes"][0].get("story") or {}
                sh_ = first_.get("shot")
                prev_shot = float(sh_) if sh_ is not None else float("nan")
            prev_type = entry.get("section_type", prev_type)
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
            beat_ = music_beat(t0, t1) if only else None
            off_ = beat_ is not None and beat_ < OFF_BEAT
            lead_, phrase_ = None, bool(lab_phrase and off_ and dd >= PHRASE_MIN_SEC)
            if lab_stem and off_ and leads:
                got_ = _AD.lead_slice(leads, drv["rate"], t0, t1)
                if got_ is not None:
                    lead_, aud = got_
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
                            # (a block may bring its own tempo — ⏱️ Edge Mode's generated beat changes speed by phase)
                            cache = v._cum[("fit", key)] = _score_video_x(v, nn, recipe, aud,
                                                                          float(seg.get("bpm") or track_bpm), ms, dd,
                                                                          music_level(t0, t1), only, beat_,
                                                                          music_speed(t0, t1), phrase=phrase_,
                                                                          shot_fix=lab_shot)
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
                            lean_ = max(0.0, 1.0 - smooth - 0.5) * 2          # 0 at the slider's middle, 1 fully at Contrast
                            if beat_ is not None:
                                # 🎯 Movement only: a change of shot size is welcome where the music has no beat — at
                                # a change point or in any beatless block — and worth more the bigger the jump
                                if (change_ok or off_) and jump >= 1.0:
                                    total += CONTRAST_QUIET * lean_ * (1.0 - beat_) * min(jump, 2.0)
                            elif change_ok and jump >= 1.0:
                                total += CONTRAST_BONUS * lean_
                        if prev_pov is not None and smooth > 0 and bool(v.pov[s_]) != prev_pov:
                            total -= POV_PENALTY * smooth * (0.5 if change_ok else 1.0)
                        out.append({"total": total, "fit": float(fit[s_]), "base": float(base[s_]), "story": float(story[s_]),
                                    "sync": float(sync[s_]), "slide": round(ws - float(v.times[s_]), 3),
                                    "video_id": vid_, "pieces": pieces, "shot": start_shot,
                                    "end_shot": _shot_at(v, min(s_ + nn - 1, v.n - 1)), "pov": bool(v.pov[s_]),
                                    **({"follows": lead_} if lead_ else {}),
                                    **({"phrase": True} if phrase_ and v.mover_slow is not None else {})})
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
                    **({"seen": True} if only and videos[c["video_id"]].seen_mover is not None else {}),
                    **({"level": round(lv_, 3)} if (lv_ := music_level(seg["start"], seg["end"])) is not None and only else {}),
                    **({"level_raw": round(raw_level(seg["start"], seg["end"]), 3)} if only and stretch > 0 else {}),
                    **({"bounce_want": round(sp_[0], 1)} if (sp_ := music_speed(seg["start"], seg["end"])) else {}),
                    **({"follows": c["follows"]} if c.get("follows") else {}),
                    **({"phrase": True} if c.get("phrase") else {}),
                    "shot": None if math.isnan(c["shot"]) else round(c["shot"], 2), **extra}

        # 1. whole-block windows, best first
        cands = search(seg["start"], seg["end"], (i,))

        # 1b. one longer clip instead of two short ones, when the footage holds for the longer stretch
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        # (blocks from the 📐 lines carry their part's label: two of them join only inside one part — a line is
        # always a cut — and up to options.merge_limit_sec, as they have no recipe pace of their own)
        same_part = (nxt is not None and (seg.get("line_section") == nxt.get("line_section")
                                          if seg.get("line_section") is not None else types[i + 1] == t_id))
        merge_cap = (float(options.merge_limit_sec) if options.merge_limit_sec
                     else MERGE_MAX * float((recipe.get("pace") or {}).get("hi", 4.0)))
        if (options.merge_blocks and nxt is not None and cands and same_part and counts[i + 1] == counts[i]
                and not segment_exclusions.get(i + 1) and not segment_clip_reduction.get(i + 1)
                and (nxt["end"] - seg["start"]) <= merge_cap + 1e-6):
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

        # how every video fared in this block: its best window's place among all videos' best windows
        ranked, gaps = {}, {}
        for c in cands:
            if c["video_id"] not in ranked:
                ranked[c["video_id"]] = len(ranked) + 1
                gaps[c["video_id"]] = (cands[0]["total"] - c["total"], cands[0]["fit"] - c["fit"])
        for vid_, rank_ in ranked.items():
            st_ = stats.setdefault(vid_, {"blocks": 0, "ranks": [], "gap": [], "fit_gap": []})
            st_["blocks"] += 1
            st_["ranks"].append(rank_)
            st_["gap"].append(gaps[vid_][0])
            st_["fit_gap"].append(gaps[vid_][1])
        stats_blocks[0] += 1

        # 🔁 the block's next-best windows — what the result page offers in place of a pick
        alts, per_v = [], {}
        for c in cands:
            if per_v.get(c["video_id"], 0) < ALT_PER_VIDEO:
                per_v[c["video_id"]] = per_v.get(c["video_id"], 0) + 1
                alts.append(_alt(c))
                if len(alts) >= ALT_KEEP:
                    break
        options.alternatives[len(timeline)] = alts
        if i in force:                 # a window chosen by hand goes in first (if its footage is still free)
            fc = dict(force[i])
            fc["pieces"] = [tuple(p_) for p_ in fc["pieces"]]
            for k_ in ("shot", "end_shot"):
                fc[k_] = float("nan") if fc.get(k_) is None else float(fc[k_])
            fc["pov"] = bool(fc.get("pov"))
            fc["_chosen"] = True
            for k_ in ("total", "fit", "base", "story", "sync", "slide"):
                fc[k_] = float(fc.get(k_) or 0.0)
            cands = [fc] + [c for c in cands if c["video_id"] != fc["video_id"]]

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
            slots.append(_slot_chain(picks, _info(c, **({"joined": len(picks)} if len(picks) > 1 else {}),
                                                  **({"chosen": True} if c.get("_chosen") else {}))))
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
        if only:
            b_seg = music_beat(seg["start"], seg["end"])
            if b_seg is not None and b_seg < OFF_BEAT:
                lab_used["off_beat"] += 1
            st0 = (slots[0].get("story") or {}) if slots else {}
            if st0.get("follows"):
                lab_used["follows"][st0["follows"]] = lab_used["follows"].get(st0["follows"], 0) + 1
            if st0.get("phrase"):
                lab_used["phrase"] += 1
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
    try:
        options.video_stats.update(_video_stats(queues, videos, missing, stats, stats_blocks[0], uses, usable_at_start))
    except Exception:              # noqa: BLE001 — a diagnostic must never break the plan
        pass
    return timeline, shortfall, unfilled_total


def _video_stats(queues: dict, videos: dict, missing: list, stats: dict, blocks: int, uses: dict, usable: dict) -> dict:
    """{video_id: …} for EVERY video in play — the answer to "why was this video never picked?". Per video:
    timeline (has a 🧭 timeline: without one it can't be picked at all), seen (has on-screen motion curves),
    people_data (its timeline has people counts; without them "nobody on screen" is read from the pose tracking),
    clips / usable_min (usable footage when the plan started), picked (times used), and over the plan's blocks:
    scored (blocks where it had a window long enough), first / top3 / top10 (blocks where its best window ranked
    there among all videos' best windows, penalties for reuse included), best_rank, median_rank, mean_gap (how far
    its best window was behind the winner, on the 0–1 score) and mean_fit_gap (the same without the reuse
    penalties). seen_busy: the raw frame-difference level its on-screen motion calls 1.0; seen_level / bounce_share:
    how lively and how often bouncy it reads overall."""
    out = {}
    for vid in queues:
        n_clips, secs = usable.get(vid, (0, 0.0))
        row = {"timeline": vid in videos, "seen": False,
               **({"people_data": videos[vid].people_known} if vid in videos else {}), "clips": int(n_clips), "usable_min": round(secs / 60.0, 1),
               "picked": int(uses.get(vid, 0)), "scored": 0, "of_blocks": int(blocks)}
        v = videos.get(vid)
        if v is not None:
            row["seen"] = v.seen is not None
            if v.seen is not None:
                row["seen_busy"] = None if v.seen_top is None else round(v.seen_top, 3)
                if np.isfinite(v.seen_level).any():
                    row["seen_level"] = round(float(np.nanmean(v.seen_level)), 3)
                row["bounce_share"] = round(float((v.seen_bounce[0] / BOUNCE_FULL >= 0.5).mean()), 3)
            elif v.bounce is not None:
                row["bounce_share"] = round(float((v.bounce[0] / BOUNCE_FULL >= 0.5).mean()), 3)
            lv = v.move_f[v.has_move > 0]
            if len(lv):
                row["person_level"] = round(float(lv.mean()), 3)
        st_ = stats.get(vid)
        if st_:
            r = np.asarray(st_["ranks"])
            row.update({"scored": int(st_["blocks"]), "first": int((r == 1).sum()), "top3": int((r <= 3).sum()),
                        "top10": int((r <= 10).sum()), "best_rank": int(r.min()), "median_rank": float(np.median(r)),
                        "mean_gap": round(float(np.mean(st_["gap"])), 3),
                        "mean_fit_gap": round(float(np.mean(st_["fit_gap"])), 3)})
        out[vid] = row
    return out


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


def _seen_whole(vid: str, tl):
    """(level per half second, (bounce strength, bounces a minute)) of a video's on-screen motion, as the matcher
    sees it — from the catalogue's scenes. None without curves."""
    try:
        import catalogue_store as CS
        from core.footage import _tc_to_seconds
        curves = CS.motion_curves(vid) or {}
        cat = CS.all_catalogues().get(vid) or {}
        scenes = [(_tc_to_seconds(sc["start_tc"]), curves[sc["scene_id"]]["fps"], curves[sc["scene_id"]]["values"])
                  for sc in cat.get("scenes") or [] if sc.get("scene_id") in curves]
        hi, rate = np.asarray(tl.move_hi, np.float32), float(tl.move_rate)
        px = SY.pixel_curve(scenes, len(hi), rate)
        if px is None:
            return None
        times = np.asarray(tl.times, float)
        return np.clip(SY.half_second_level(px, rate, times, STEP), 0.0, 1.0), SY.bounce(px, rate, times)
    except Exception:          # noqa: BLE001 — a diagnostic: never break the file
        return None


def _pixel_move(cache: dict, vid: str, link: dict, dur: float, m: int):
    """The clip's on-screen change (the catalogue's frame-difference motion curve: how much of the picture changes
    from one frame to the next — camera moves and whole bodies included, which the person curve leaves out), at the
    movement curve's rate, as a share of the video's own busy level (its 95th percentile). None without it."""
    try:
        if vid not in cache:
            import catalogue_store as CS
            curves = CS.motion_curves(vid) or {}
            vals = [np.asarray(c["values"], float) for c in curves.values() if len(c["values"])]
            top = float(np.percentile(np.concatenate(vals), 95)) if vals else 0.0
            cache[vid] = (curves, top)
        curves, top = cache[vid]
        c = curves.get(link.get("scene_id"))
        if c is None:
            c = curves.get(str(link.get("scene_id")))
        if c is None or top <= 1e-9 or m < 1:
            return None
        fps = float(c["fps"])
        a = int(round(float(link.get("offset_into_scene_sec") or 0.0) * fps))
        part = np.asarray(c["values"], float)[a:a + max(1, int(round(dur * fps)))]
        if len(part) < 2:
            return None
        return [round(float(x), 3) for x in SY.pool(part / top, m)]
    except Exception:          # noqa: BLE001 — the file is a diagnostic: a missing curve must not break it
        return None


def sync_report(timeline: list, track: dict, context_sec: float = 1.0, video_stats: dict = None,
                settings: dict = None) -> dict:
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
    try:
        leads = AD.lead_curves(track, drv)
    except Exception:          # noqa: BLE001
        leads = {}
    r2 = lambda v: [None if not np.isfinite(x) else round(float(x), 3) for x in np.asarray(v, float).ravel()]    # noqa: E731
    tls, rows, rate, bounces, pixel, seen = {}, [], None, {}, {}, {}
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
                       "seconds": round(dur, 3), **{k: (slot.get("story") or {}).get(k) for k in ("fit", "base", "sync", "slide")},
                       **{k: (slot.get("story") or {})[k] for k in ("level", "level_raw", "bounce_want", "parts", "joined", "follows", "phrase") if (slot.get("story") or {}).get(k) is not None}}
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
                                           **{k: r2(SY.pool(v[lo:hi_], len(move))) for k, v in drv["curves"].items()},
                                           **({"other": r2(SY.pool(leads["other"][lo:hi_], len(move)))} if "other" in leads else {})}
                    if "drums" in drv["curves"]:
                        bp = SY.beat_presence(drv["curves"]["drums"][lo:hi_], AD.DRUM_LEVEL)
                        row["beat"] = None if bp is None else round(bp, 3)
                        hr = SY.hit_rate(drv["curves"]["drums"][lo:hi_], a_rate, AD.DRUM_LEVEL)
                        if hr is not None:
                            row["hits_per_sec"] = round(hr, 2)
                    if vid not in bounces:
                        bounces[vid] = SY.bounce(hi, rate, np.asarray(tl.times, float))
                    k0 = int(np.searchsorted(np.asarray(tl.times, float), float(link["clip_start_sec"])))
                    k1 = max(k0 + 1, int(np.searchsorted(np.asarray(tl.times, float), float(link["clip_start_sec"]) + dur)))
                    st_, bpm_ = bounces[vid][0][k0:k1], bounces[vid][1][k0:k1]
                    pts = slice(k0, k1)
                    mv_ = np.asarray(tl.scaled_move(), float)[pts]
                    if np.isfinite(mv_).any():
                        row["move_level"] = round(float(np.nanmean(mv_)), 3)
                    sh_ = np.asarray(tl.shot, float)[pts]
                    if np.isfinite(sh_).any():
                        row["shot"] = round(float(np.nanmean(sh_)), 2)
                    fz = float(SY.frozen(hi, rate, np.asarray(tl.times, float)[pts], STEP).mean()) if k1 > k0 else 0.0
                    if fz > 0:
                        row["frozen"] = round(fz, 3)
                    if len(st_):
                        row["bounce"] = round(float(np.clip(st_.mean() / BOUNCE_FULL, 0.0, 1.0)), 3)
                        row["bounce_bpm"] = round(float(np.nanmedian(bpm_)), 1) if np.isfinite(bpm_).any() else None
                    if vid not in seen:
                        seen[vid] = _seen_whole(vid, tl)
                    if seen[vid] is not None:
                        s_lv, (s_st, s_bpm) = seen[vid]
                        if np.isfinite(s_lv[pts]).any():
                            row["seen_level"] = round(float(np.nanmean(s_lv[pts])), 3)
                            row["seen_bounce"] = round(float(np.clip(s_st[pts].mean() / BOUNCE_FULL, 0.0, 1.0)), 3)
                            row["seen_bounce_bpm"] = (round(float(np.nanmedian(s_bpm[pts])), 1)
                                                      if np.isfinite(s_bpm[pts]).any() else None)
                    px = _pixel_move(pixel, vid, link, dur, len(move))
                    if px is not None:
                        row["pixel_move"] = px
                    # as played, on what the matcher picked on: both curves where it used the on-screen motion
                    used_seen = bool((slot.get("story") or {}).get("seen")) and px is not None
                    # 🧪 …against the curve it was compared with (🎼 the lead layer) at the scale it was compared at (🌊)
                    story_ = slot.get("story") or {}
                    music_ = leads[story_["follows"]][lo:hi_] if story_.get("follows") in leads else drv["curve"][lo:hi_]
                    sm_ = PHRASE_SMOOTH_SEC if story_.get("phrase") else SY.SMOOTH_SEC
                    played = SY.sync_of(move, music_, rate, px if used_seen else None, SEEN_SYNC, sm_)
                    row["sync_played"] = None if played is None else round(played, 3)
                    if story_.get("follows") or story_.get("phrase"):     # …and as it was measured before the lab
                        plain = SY.sync_of(move, drv["curve"][lo:hi_], rate, px if used_seen else None, SEEN_SYNC)
                        if plain is not None:
                            row["sync_played_plain"] = round(plain, 3)
                    if used_seen:              # …and each curve on its own, to see which one carries it
                        for name, part in (("sync_played_person", SY.sync_of(move, music_, rate, smooth_sec=sm_)),
                                           ("sync_played_seen", SY.sync_of(px, music_, rate, smooth_sec=sm_))):
                            if part is not None:
                                row[name] = round(part, 3)
                rows.append(row)
                tt += dur
    played = [r_["sync_played"] for r_ in rows if r_.get("sync_played") is not None]
    secs = [r_["seconds"] for r_ in rows if r_.get("sync_played") is not None]
    # is the movement measure the same thing at every shot size? Over the whole of every video in the plan: how much
    # movement a close-up shows against a wide shot (0–1 within the library), and how bouncy each reads
    by_shot = {}
    names = ("extreme close-up", "close-up", "medium", "wide")
    for vid, tl in tls.items():
        if tl is None:
            continue
        try:
            sh, mv = np.asarray(tl.shot, float), np.asarray(tl.scaled_move(), float)
            bo = bounces[vid][0] if vid in bounces else SY.bounce(np.asarray(tl.move_hi, np.float32), float(tl.move_rate),
                                                                   np.asarray(tl.times, float))[0]
        except (AttributeError, TypeError, ValueError):
            continue
        ok = np.isfinite(sh) & np.isfinite(mv)
        for k, name in enumerate(names):
            m = ok & (np.round(sh) == k)
            if m.any():
                d_ = by_shot.setdefault(name, {"move": [], "bounce": []})
                d_["move"].append(mv[m])
                d_["bounce"].append(bo[m])
    shot_table = {}
    for name in names:
        if name in by_shot:
            mv, bo = np.concatenate(by_shot[name]["move"]), np.concatenate(by_shot[name]["bounce"])
            shot_table[name] = {"minutes": round(len(mv) * STEP / 60, 1),
                                "move_p25_p50_p75_p90": [round(float(x), 3) for x in np.percentile(mv, [25, 50, 75, 90])],
                                "bounce_mean": round(float(np.clip(bo / BOUNCE_FULL, 0, 1).mean()), 3),
                                "bouncy_share": round(float((bo / BOUNCE_FULL >= 0.5).mean()), 3)}
    plain = [(r_.get("sync_played_plain", r_["sync_played"]), r_["seconds"]) for r_ in rows if r_.get("sync_played") is not None]
    off = [(r_["sync_played"], r_["seconds"]) for r_ in rows
           if r_.get("sync_played") is not None and r_.get("beat") is not None and r_["beat"] < OFF_BEAT]
    on = [(r_["sync_played"], r_["seconds"]) for r_ in rows
          if r_.get("sync_played") is not None and r_.get("beat") is not None and r_["beat"] >= OFF_BEAT]
    wmean = lambda xs: round(float(np.average([a_ for a_, _ in xs], weights=[max(b_, 1e-3) for _, b_ in xs])), 3) if xs else None   # noqa: E731
    return {"track_id": track.get("track_id"), "bpm": track.get("bpm"), "rate": rate, "music_rate": a_rate,
            **({"settings": settings} if settings else {}),
            # the same clips against the plain drive curve at the usual scale — the figure to compare with a run
            # that had the 🧪 switches off (with 🎼 or 🌊 on, sync_played is measured against a different curve)
            "sync_played_plain_mean": wmean(plain),
            "sync_played_on_beat": wmean(on), "sync_played_off_beat": wmean(off),
            "seconds_on_beat": round(sum(b_ for _, b_ in on), 1), "seconds_off_beat": round(sum(b_ for _, b_ in off), 1),
            # time-weighted, like the plan's own 🎯 figure: a 0.3 s join doesn't count as much as a 5 s clip
            "sync_played_mean": round(float(np.average(played, weights=np.maximum(secs, 1e-3))), 3) if played else None,
            "how_to_read": "sync: 1 = the movement follows the music, 0.5 = unrelated, 0 = opposite (sync = what the "
                           "matcher scored, sync_played = the same measure on the clip as it plays; both use the "
                           "person curve and the on-screen motion together where the matcher did, and "
                           "sync_played_person / sync_played_seen give each alone). move / music_curves "
                           "have one point per 1/rate s from the clip's first frame. beat: 1 = a full drum beat in this "
                           "stretch, 0 = none. bounce: 1 = a strong regular up-and-down, 0 = smooth or irregular "
                           "(bounce_bpm = its tempo). The two should agree. level: how lively the music asks the footage "
                           "to be (level_raw: before 🎚️ the track's range was stretched), move_level: how lively the clip is (all 0–1). shot: 0 extreme close-up … 3 wide. "
                           "hits_per_sec: drum hits a second (how busy the rhythm is); bounce_want: the bounce speed "
                           "🥁 asked for there (bounces a minute). frozen: share of the clip with the picture frozen. pixel_move: how much of the "
                           "picture changes frame to frame (1 = the video's busy level) — camera and whole bodies "
                           "included, unlike move. seen_level / seen_bounce / seen_bounce_bpm: the same "
                           "three readings as move_level / bounce / bounce_bpm, taken from that on-screen motion. "
                           "🧪 follows: the "
                           "layer the movement was compared with off the beat (🎼); phrase: compared at the scale of a "
                           "phrase (🌊); sync_played_plain: the clip against the usual drive curve at the usual scale. "
                           "sync_played_on_beat / _off_beat: the plan's sync where the music has a beat and where it "
                           "has none (beat under 0.5).",
            "movement_by_shot_size": shot_table,
            # every video in play and how it fared (see _video_stats) — never-picked videos included
            **({"videos": video_stats, "videos_how_to_read": _video_stats.__doc__.split("Per video:", 1)[1].strip()}
               if video_stats else {}),
            "clips": rows}


def free_alternatives(timeline: list, entry_index: int, alternatives: list, excluded: set = (), limit: int = 6) -> list:
    """The block's kept next-best windows that could go in now: none from a rejected clip, none overlapping footage
    another block of the plan already uses (the block's own picks don't count — they are what would be replaced).
    One window per video, best first."""
    used = {}
    for k, e in enumerate(timeline):
        if k == entry_index:
            continue
        for slot in e.get("scenes") or []:
            for link in slot.get("chain") or []:
                a = float(link["offset_into_scene_sec"])
                used.setdefault((link["video_id"], link["scene_id"]), []).append((a, a + float(link["clip_duration_sec"])))
    own = {(link["video_id"], link["scene_id"]) for slot in timeline[entry_index].get("scenes") or []
           for link in slot.get("chain") or []} if 0 <= entry_index < len(timeline) else set()
    out = []
    for alt in alternatives or []:
        ok = True
        for scene_id, off, dur in alt["pieces"]:
            key = (alt["video_id"], scene_id)
            if key in excluded or (key in own and not alt.get("_keep_own")):
                ok = False
                break
            if any(min(b_, off + dur) - max(a_, off) > 0.05 for a_, b_ in used.get(key, ())):
                ok = False
                break
        if ok and alt["video_id"] not in {o["video_id"] for o in out}:
            out.append(alt)
            if len(out) >= limit:
                break
    return out
