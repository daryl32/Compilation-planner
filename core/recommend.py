"""🔮 Recommendations — "For this track": five plans for a track (videos + cutting
style), found by searching the whole library and checked by actually matching
them.

The five
  1. 📐 Best shape fit — clips whose motion rises and falls with the music.
  2. 🎭 Best story fit — role clips (Intro / Slow / Action / Outro) in their story-map parts.
  3. 🏆 Best overall  — shape + motion + story, then fine-tuned by video swaps.
  4. 🥊 Verses        — a cutting style the top three don't use; the best-fitting video with
                         an actor, plus its best-fitting similar video ("find similar") with a
                         different actor.
  5. 🎲 Wildcard      — a random cutting style and random videos, matched for the best fit.

How it works
  1. Fingerprint every video (once, cached by the page): every window of every
     scene at a ladder of lengths (0.75 s … 20 s), each one's motion SHAPE
     (24 points, z-scored) and motion LEVEL (0–1 for the video).
  2. Cut the track: 📐 / 🎭 / 🏆 all use Smart at normal pace; 🥊 Verses searches the
     other cutting styles (core/styles.py); fingerprint each block the same way
     from the music energy.
  3. Score every video against every block (best window of the nearest length):
     shape = correlation, level = 1 − |motion level − block intensity|; story =
     how much of the song's need for each role the videos' role footage covers.
  4. Pick video sets greedily for each mode's objective.
  5. Run the REAL matcher (core.matching, Auto mode — story map on unless it's
     the shape-fit plan, sequential when asked) on the leading candidates and
     score what it actually picked, so the % shown is what you get when you open it.

Scoring (decided with Daryl, Oct 2026)
  • Shape %      — every block counts equally.
  • Motion %     — motion level vs the music's intensity; drops / action parts count DROP_WEIGHT×.
  • Story %      — story-map blocks whose clip carries the block's role.
  • Action %     — motion match over the drop / action blocks only.
  • Overall %    — ⅓ shape + ⅓ motion + ⅓ story (½ + ½ when the track has no story parts).
  • Weakest block — the lowest-scoring block, so a high average can't hide it.

Pure functions, no Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.segmentation import SEG_DEFAULTS, SEG_METHOD_KEYS, SEG_SMART, apply_segment_intensity, build_segments
from core.shape import get_block_audio_curve, resample_curve, shape_score

POINTS = 24
LENGTHS = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
DROP_WEIGHT = 2.5          # intensity weight of drop / action blocks
DROP_BARS_AFTER = 4        # a drop "block" is any block overlapping [drop, drop + 4 bars]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _znorm_rows(m: np.ndarray) -> np.ndarray:
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, keepdims=True)
    out = np.zeros_like(m)
    ok = sd[:, 0] > 1e-9
    out[ok] = (m[ok] - mu[ok]) / sd[ok]
    return out


def _resample_rows(windows: np.ndarray) -> np.ndarray:
    n = windows.shape[1]
    if n < 2:
        return np.repeat(windows[:, :1], POINTS, axis=1)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, POINTS)
    idx = np.clip(np.searchsorted(x_old, x_new, side="right") - 1, 0, n - 2)
    frac = (x_new - x_old[idx]) / (x_old[idx + 1] - x_old[idx])
    return windows[:, idx] * (1 - frac) + windows[:, idx + 1] * frac


@dataclass
class VideoPrint:
    video_id: str
    by_len: dict = field(default_factory=dict)   # length → (Z [n, POINTS] float16, level [n] float16) — compact
    scale: float = 1.0                           # motion value that counts as level 1.0


def fingerprint_video(video_id: str, catalogue: dict, curves: dict, time_range=None) -> VideoPrint | None:
    """Windows of every usable scene (not excluded; inside the video's library range).
    catalogue: the video's catalogue (scenes with start_tc / end_tc); curves: {scene_id: {"fps", "values"}}."""
    from core.footage import _tc_to_seconds
    vp = VideoPrint(video_id)
    pieces = []          # (values array, fps)
    for s in catalogue.get("scenes", []):
        if s.get("excluded"):
            continue
        c = curves.get(s["scene_id"])
        if not c or c.get("values") is None or len(c["values"]) < 2:
            continue
        vals = np.asarray(c["values"], dtype=float)
        fps = float(c["fps"])
        if time_range:
            t0 = _tc_to_seconds(s["start_tc"])
            a = max(0, int(round((time_range[0] - t0) * fps)))
            b = min(len(vals), int(round((time_range[1] - t0) * fps)))
            vals = vals[a:b]
            if len(vals) < 2:
                continue
        pieces.append((vals, fps))
    if not pieces:
        return None
    allv = np.concatenate([p[0] for p in pieces])
    vp.scale = float(np.percentile(allv, 95)) or float(allv.max()) or 1.0
    for L in LENGTHS:
        zs, lv = [], []
        for vals, fps in pieces:
            wf = max(2, int(round(L * fps)))
            if len(vals) < wf:
                continue
            step = max(1, int(round(max(0.5, L / 4) * fps)))
            win = np.lib.stride_tricks.sliding_window_view(vals, wf)[::step]
            zs.append(_znorm_rows(_resample_rows(win)).astype(np.float16))
            lv.append(np.clip(win.mean(axis=1) / vp.scale, 0, 1).astype(np.float16))
        if zs:
            vp.by_len[L] = (np.concatenate(zs), np.concatenate(lv))
    return vp if vp.by_len else None


def _nearest_len(d: float, available) -> float | None:
    fits = [L for L in available if L <= d * 1.15]
    return max(fits) if fits else None


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def block_weights(track: dict, segments: list, story_regions: list = None, beats_per_bar: int = 4) -> list:
    """Intensity weight per block: DROP_WEIGHT for drop / action blocks, else 1.
    Drop / action = overlaps [drop, drop + 4 bars] (from the analysis), or lies in a story-map
    'action' part. With neither available: the most intense quarter of the blocks."""
    import cut_scoring as CS
    bar = CS.bar_sec(track, beats_per_bar)
    a = track.get("analysis") or {}
    spans = [(float(t), float(t) + DROP_BARS_AFTER * bar) for t in (a.get("drops") or [])]
    spans += [(float(r["start"]), float(r["end"])) for r in (story_regions or []) if r.get("role") == "action"]
    if spans:
        flags = [any(sg["start"] < e and sg["end"] > s for s, e in spans) for sg in segments]
    else:
        ints = np.array([sg.get("intensity", 0.0) for sg in segments])
        cut = np.quantile(ints, 0.75) if len(ints) else 1.0
        flags = [i >= cut and i > 0 for i in ints]
    return [DROP_WEIGHT if f else 1.0 for f in flags]


def cut_variant(track: dict, settings: dict, beats_per_bar: int = 4) -> list:
    cfg = {"method": SEG_SMART, "beats_per_bar": beats_per_bar,
           **{k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_SMART]}, **settings}
    segs = build_segments(track, cfg)
    apply_segment_intensity(segs, track, 1.0)
    return segs


def block_prints(track: dict, segments: list) -> list:
    out = []
    for sg in segments:
        curve = get_block_audio_curve(track, sg["start"], sg["end"])
        z = resample_curve(curve, POINTS)
        sd = z.std()
        out.append(((z - z.mean()) / sd if sd > 1e-9 else np.zeros(POINTS)).astype(np.float32))
    return out




# ---------------------------------------------------------------------------
# Cutting styles to search
# ---------------------------------------------------------------------------

def variant_label(style: str, pace: int) -> str:
    from core.styles import PACE_LABELS, STYLES
    return f"{STYLES[style]['label']} · {PACE_LABELS[int(pace)]}"


@dataclass
class Variant:
    """One way of cutting the track, with every video scored against its blocks."""
    style: str
    pace: int
    label: str
    method: str
    settings: dict
    segments: list
    weights: list                 # intensity weight per block (drops / action parts count more)
    roles: list                   # the story map's role per block (None = no role there)
    vids: list = None
    S: np.ndarray = None          # [videos, blocks] best window: ½ shape + ½ motion level
    SH: np.ndarray = None         # [videos, blocks] shape part of that window

    @property
    def story_active(self) -> bool:
        return any(self.roles)

    def need(self) -> dict:
        """{role: seconds of the song it covers} (its blocks' total length)."""
        out = {}
        for sg, r in zip(self.segments, self.roles):
            if r:
                out[r] = out.get(r, 0.0) + (sg["end"] - sg["start"])
        return out


def make_variant(track: dict, style: str, pace: int, prints: dict, story_regions: list = None,
                 beats_per_bar: int = 4) -> Variant:
    from core.styles import style_settings
    from story_map import block_roles as _block_roles
    method, settings, _ = style_settings(track, style, pace, beats_per_bar)
    segs = build_segments(track, {"method": method, "beats_per_bar": beats_per_bar, **settings})
    apply_segment_intensity(segs, track, 1.0)
    v = Variant(style, int(pace), variant_label(style, pace), method, settings, segs,
                block_weights(track, segs, story_regions, beats_per_bar),
                _block_roles(story_regions, segs) if story_regions else [None] * len(segs))
    v.vids, v.S, v.SH, _ = score_matrix(prints, segs, block_prints(track, segs))
    return v


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def score_matrix(prints: dict, segments: list, bprints: list) -> tuple:
    """(video_ids, S [videos, blocks] best window score 0–1, SHAPE, LEVEL) — each block scored
    by the best window of each video: ½ shape (correlation ≥ 0) + ½ (1 − |level − intensity|)."""
    vids = sorted(prints)
    nb = len(segments)
    S = np.zeros((len(vids), nb), dtype=np.float32)
    SH = np.zeros_like(S)
    LV = np.zeros_like(S)
    if not nb:
        return vids, S, SH, LV
    B = np.stack(bprints).astype(np.float32)
    inten_b = np.array([float(sg.get("intensity", 0.5)) for sg in segments], dtype=np.float32)
    durs = [sg["end"] - sg["start"] for sg in segments]
    for vi, vid in enumerate(vids):
        vp = prints[vid]
        groups = {}
        for bi, d in enumerate(durs):
            L = _nearest_len(d, vp.by_len)
            if L is not None:
                groups.setdefault(L, []).append(bi)
        for L, bis in groups.items():
            Z, lev = vp.by_len[L]
            Z, lev = Z.astype(np.float32), lev.astype(np.float32)
            corr = np.clip(Z @ B[bis].T / POINTS, 0, 1)                     # [windows, blocks]
            inten = 1.0 - np.abs(lev[:, None] - inten_b[bis][None, :])
            comb = 0.5 * corr + 0.5 * inten
            k = np.argmax(comb, axis=0)
            cols = np.arange(len(bis))
            S[vi, bis], SH[vi, bis], LV[vi, bis] = comb[k, cols], corr[k, cols], inten[k, cols]
    return vids, S, SH, LV


def choose_sets(vids: list, S: np.ndarray, weights: list, n_videos: int, n_plans: int = 3) -> list:
    """Greedy video sets: each step adds the video that raises the plan's weighted block cover
    most (no video twice in a row is the matcher's job). Later plans halve the gain of videos
    already used, so they're genuinely different."""
    w = np.asarray(weights, dtype=np.float32)
    return greedy_sets(vids, lambda ch: np.max(S[ch], axis=0) if ch else np.zeros(S.shape[1]),
                       w, n_videos, n_plans)


def greedy_sets(vids: list, value_of, weights, n_videos: int, n_plans: int = 3, allowed=None) -> list:
    """Greedy video sets for any objective. value_of(list of video indices) → value per block
    (0–1); a plan's estimate is the weighted mean. Each step adds the video that raises it
    most; later plans halve the gain of videos already used, so they're genuinely different.
    allowed: video indices that may be picked (default all)."""
    w = np.asarray(weights, dtype=np.float32)
    wsum = max(float(w.sum()), 1e-9)
    pool = list(range(len(vids))) if allowed is None else list(allowed)
    used = np.zeros(len(vids))
    plans = []
    for _ in range(n_plans):
        chosen, base = [], 0.0
        for _ in range(min(n_videos, len(pool))):
            best_k, best_gain = None, 0.0
            for k in pool:
                if k in chosen:
                    continue
                gain = (float(value_of(chosen + [k]) @ w) / wsum - base) * (0.5 ** used[k])
                if gain > best_gain + 1e-12:
                    best_k, best_gain = k, gain
            if best_k is None:
                break
            chosen.append(best_k)
            base = float(value_of(chosen) @ w) / wsum
        if not chosen:
            break
        for k in chosen:
            used[k] += 1
        ids = tuple(sorted(vids[k] for k in chosen))
        if ids not in [p["videos"] for p in plans]:
            plans.append({"videos": ids, "estimate": base})
    return plans


# The three objectives (value per block, 0–1) and the block weights each is judged with.

def _story_cover(v: Variant, chosen: list, role_secs: list) -> dict:
    """{role: share of the song's need for it the chosen videos' role footage covers}."""
    have = {}
    for k in chosen:
        for r, sec in role_secs[k].items():
            have[r] = have.get(r, 0.0) + sec
    return {r: min(1.0, have.get(r, 0.0) / max(n, 1e-6)) for r, n in v.need().items()}


def objective(v: Variant, mode: str, role_secs: list):
    """(value_of, weights) for greedy_sets. mode: "shape" | "overall" | "story"."""
    nb = len(v.segments)
    if mode == "shape":
        return (lambda ch: np.max(v.SH[ch], axis=0) if ch else np.zeros(nb)), np.ones(nb, dtype=np.float32)
    w = np.asarray(v.weights, dtype=np.float32)
    role_idx = [i for i, r in enumerate(v.roles) if r]
    if not role_idx:
        return (lambda ch: np.max(v.S[ch], axis=0) if ch else np.zeros(nb)), w

    def value_of(ch):
        if not ch:
            return np.zeros(nb)
        base = np.max(v.S[ch], axis=0).astype(float)
        cover = _story_cover(v, ch, role_secs)
        out = base.copy()
        for i in role_idx:
            st = cover.get(v.roles[i], 0.0)
            out[i] = (2 * base[i] + st) / 3 if mode == "overall" else 0.3 * base[i] + 0.7 * st
        return out
    if mode == "story":
        w = w.copy()
        w[role_idx] *= 2.0
    return value_of, w


# ---------------------------------------------------------------------------
# Checking a plan with the real matcher
# ---------------------------------------------------------------------------

def clip_level(pick: dict, prints_scale: dict, curves_of) -> float | None:
    c = curves_of(pick["video_id"]).get(pick["scene_id"])
    if not c or c.get("values") is None or len(c["values"]) == 0:
        return None
    fps = float(c["fps"])
    a = int(round(pick["offset_into_scene_sec"] * fps))
    b = a + max(1, int(round(pick["clip_duration_sec"] * fps)))
    seg = np.asarray(c["values"][a:b], dtype=float)
    if not len(seg):
        return None
    return float(np.clip(seg.mean() / prints_scale.get(pick["video_id"], 1.0), 0, 1))


def score_timeline(track: dict, segments: list, timeline: list, weights: list, prints_scale: dict, curves_of,
                   block_roles: list = None, roles_of=None) -> dict:
    """Scores of what the matcher actually picked (first clip of each block).
    block_roles: the story map's role per block — Story % is the share of those blocks whose
    clip carries the role (roles_of(video_id, scene_id) → the scene's role ids: the matcher's
    picks don't carry them); with any, Overall = ⅓ shape + ⅓ motion + ⅓ story (else ½ + ½)."""
    block_roles = block_roles or [None] * len(segments)
    roles_of = roles_of or (lambda _v, _s: ())
    shape, inten, comb, story, per_story = [], [], [], [], []
    for sg, entry, role in zip(segments, timeline, block_roles):
        slots = entry.get("scenes") or []
        if not slots:
            shape.append(0.0)
            inten.append(0.0)
            comb.append(0.0)
            if role:
                story.append(0.0)
            per_story.append(0.0 if role else None)
            continue
        p = slots[0]["chain"][0]
        c = curves_of(p["video_id"]).get(p["scene_id"])
        sh = 0.0
        if c and c.get("values") is not None and len(c["values"]):
            fps = float(c["fps"])
            a = int(round(p["offset_into_scene_sec"] * fps))
            b = a + max(2, int(round(p["clip_duration_sec"] * fps)))
            clip = np.asarray(c["values"][a:b], dtype=float)
            if len(clip) >= 2:
                sh = max(0.0, shape_score(clip, get_block_audio_curve(track, sg["start"], sg["end"])))
        lvl = clip_level(p, prints_scale, curves_of)
        it = 0.0 if lvl is None else 1.0 - abs(lvl - float(sg.get("intensity", 0.5)))
        shape.append(sh)
        inten.append(it)
        if role:
            st = 1.0 if role in (roles_of(p["video_id"], p["scene_id"]) or ()) else 0.0
            story.append(st)
            per_story.append(100.0 * st)
            comb.append((sh + it + st) / 3)
        else:
            per_story.append(None)
            comb.append(0.5 * sh + 0.5 * it)
    w = np.asarray(weights, dtype=float)
    shape_pct = 100 * float(np.mean(shape)) if shape else 0.0
    inten_pct = 100 * float(np.average(inten, weights=w)) if inten else 0.0
    story_pct = 100 * float(np.mean(story)) if story else None
    overall = (shape_pct + inten_pct + story_pct) / 3 if story_pct is not None else 0.5 * shape_pct + 0.5 * inten_pct
    act = [x for x, wi in zip(inten, weights) if wi > 1.0]
    act_story = [x for x, r in zip(per_story, block_roles) if r == "action" and x is not None]
    weakest = int(np.argmin(comb)) if comb else None
    return {"overall": round(overall, 1), "shape": round(shape_pct, 1),
            "intensity": round(inten_pct, 1), "story": None if story_pct is None else round(story_pct, 1),
            "action": round(100 * float(np.mean(act)), 1) if act else None,
            "action_story": round(float(np.mean(act_story)), 1) if act_story else None,
            "weakest_block": weakest, "weakest": round(100 * comb[weakest], 1) if comb else None,
            "per_block": [round(100 * x, 1) for x in comb],
            "per_block_shape": [round(100 * x, 1) for x in shape],
            "per_block_story": per_story, "action_blocks": [i for i, x in enumerate(weights) if x > 1.0],
            "story_blocks": [i for i, r in enumerate(block_roles) if r]}


# ---------------------------------------------------------------------------
# The five recommendations
# ---------------------------------------------------------------------------

MODES = [
    ("shape",    "📐 Best shape fit",
     "The clips whose motion rises and falls most closely with the music, block by block (story map not used)."),
    ("story",    "🎭 Best story fit",
     "Clips marked 🎬 Intro / 🌙 Slow / 💥 Action / 🏁 Outro landing in the parts of the song the story map gives them."),
    ("overall",  "🏆 Best overall",
     "The best balance of shape + motion + story — then fine-tuned by swapping videos while the real matcher "
     "keeps scoring it higher."),
    ("verses",   "🥊 Verses",
     "Two similar videos with different actors, head to head — in a cutting style other than the top three's "
     "🧠 Smart."),
    ("wildcard", "🎲 Wildcard",
     "A random cutting style and random videos, matched as well as they can be (shape + motion + story)."),
]
MODE_IDS = [m[0] for m in MODES]
MODE_LABELS = {m[0]: m[1] for m in MODES}
MODE_HELP = {m[0]: m[2] for m in MODES}

# 📐 / 🎭 / 🏆 all use Smart at normal pace (so they compare like for like); 🥊 Verses picks
# from the other styles (normal pace); 🎲 Wildcard rolls any style and pace.
TOP_STYLE = ("smart", 0)
VARIANTS = [TOP_STYLE, ("cinematic", 0), ("beat", 0), ("energy", 0), ("drop", 0), ("vocal", 0), ("hyper", 0)]
CHECKS = {"shape": 4, "story": 4, "overall": 5}   # leading candidates run through the real matcher
SWAP_BUDGET = 10                                  # Best overall: one-video swaps tried on the best plan
VERSES_SEEDS = 3                                  # Verses: top picks tried
VERSES_POOL = 5                                   # Verses: similar videos (different actors) per top pick


def recommend_modes(track: dict, prints: dict, *, match, curves_of, n_videos: int = 4, story_regions: list = None,
                    role_seconds: dict = None, roles_of=None, actors: dict = None, similar_of=None,
                    sequential: bool = False, wildcard_seed: int = 0, progress=None, insights: dict = None) -> list:
    """The five recommendations for `track`, in MODES order: [{"mode", "plan" (or None), "note"}].

    prints: {video_id: VideoPrint}. match(segments, video_ids, block_roles, sequential) runs the
    real matcher (block_roles None = story map off) and returns its timeline.
    curves_of(video_id) → {scene_id: {"fps", "values"}}.
    story_regions: the track's story map (saved, or suggested) — None for none.
    role_seconds: {video_id: {role: seconds of usable footage marked with it}};
    roles_of(video_id, scene_id) → that scene's role ids (for the Story score).
    actors: {video_id: [names]}; similar_of(video_id) → [(similarity, reason, video_id)] best first.
    Each plan: {"mode", "variant", "style", "pace", "method", "settings", "videos", "story",
    "sequential", "estimate", "scores", "timeline", "segments"} (+ "actors" for Verses).
    insights: a dict to fill with what the feeds need (see library_insights)."""
    progress = progress or (lambda f, t: None)
    role_seconds = role_seconds or {}
    actors = actors or {}
    scale = {v: p.scale for v, p in prints.items()}
    n_videos = max(1, min(n_videos, len(prints)))
    total_steps = len(VARIANTS) + sum(CHECKS.values()) + SWAP_BUDGET + VERSES_SEEDS + 2
    step = [0]

    def tick(text):
        progress(min(step[0] / total_steps, 0.99), text)
        step[0] += 1

    variants = []
    for style, pace in VARIANTS:
        tick(f"Cutting the track ({variant_label(style, pace)})…")
        variants.append(make_variant(track, style, pace, prints, story_regions))
    by_key = {(v.style, v.pace): v for v in variants}
    vids = variants[0].vids
    role_secs = [role_seconds.get(vid, {}) for vid in vids]
    story_on = any(v.story_active for v in variants)

    checked = {}

    def check(v: Variant, videos, mode: str, estimate: float = 0.0):
        use_story = mode != "shape" and v.story_active
        key = (v.style, v.pace, tuple(sorted(videos)), use_story)
        if key not in checked:
            roles = list(v.roles) if use_story else None
            timeline = match(v.segments, list(key[2]), roles, sequential)
            scores = score_timeline(track, v.segments, timeline, v.weights, scale, curves_of, v.roles, roles_of)
            checked[key] = {"variant": v.label, "style": v.style, "pace": v.pace, "method": v.method,
                            "settings": v.settings, "videos": list(key[2]), "story": use_story,
                            "sequential": bool(sequential), "estimate": estimate, "scores": scores,
                            "timeline": timeline, "segments": v.segments, "_variant": v}
        return checked[key]

    top_v = by_key[TOP_STYLE]
    if insights is not None:
        insights.update(library_insights(top_v, role_secs))

    def candidates(mode: str) -> list:
        """The top three modes all use the one Smart cut (TOP_STYLE)."""
        out = []
        value_of, w = objective(top_v, mode, role_secs)
        for plan in greedy_sets(vids, value_of, w, n_videos, max(CHECKS.values()) + 1):
            out.append((plan["estimate"], top_v, plan["videos"]))
        out.sort(key=lambda c: -c[0])
        return out

    rank_key = {"shape": lambda p: (p["scores"]["shape"], p["scores"]["overall"]),
                "story": lambda p: (p["scores"]["story"] or 0.0, p["scores"]["overall"]),
                "overall": lambda p: (p["scores"]["overall"],)}

    def run_mode(mode: str) -> list:
        """The checked plans of this mode's leading candidates, best first."""
        got = []
        for est, v, videos in candidates(mode)[:CHECKS[mode]]:
            tick(f"{MODE_LABELS[mode]}: checking a plan with the real matcher…")
            got.append(check(v, videos, mode, est))
        return sorted(got, key=lambda p: tuple(-x for x in rank_key[mode](p)))

    chosen = {}
    taken = set()

    def take(mode: str, plans: list):
        for p in plans:
            k = (p["style"], p["pace"], tuple(p["videos"]))
            if k not in taken:
                taken.add(k)
                chosen[mode] = dict(p, mode=mode)
                return

    def is_taken(p) -> bool:
        return (p["style"], p["pace"], tuple(p["videos"])) in taken

    # 📐 and 🎭 claim their best plans first (each card leads with its own score), then
    # 🏆 Best overall takes the best of the rest and fine-tunes it with video swaps.
    shape_plans = run_mode("shape")
    story_plans = run_mode("story") if story_on else []
    overall = run_mode("overall")
    take("shape", shape_plans)
    if story_on:
        take("story", story_plans)
    free = [p for p in overall + story_plans + shape_plans if not is_taken(p)]
    if free:
        best = max(free, key=lambda p: p["scores"]["overall"])
        bv = best["_variant"]
        value_of, w = objective(bv, "overall", role_secs)
        alone = sorted(range(len(vids)), key=lambda k: -float(value_of([k]) @ w))
        pool = [vids[k] for k in alone if vids[k] not in best["videos"]]
        done, improved = 0, True
        while improved and done < SWAP_BUDGET and pool:
            improved = False
            for out_v in list(best["videos"]):
                for in_v in pool[:3]:
                    if done >= SWAP_BUDGET:
                        break
                    tick(f"{MODE_LABELS['overall']}: fine-tuning ({done + 1}/{SWAP_BUDGET})…")
                    trial = check(bv, [v for v in best["videos"] if v != out_v] + [in_v], "overall", best["estimate"])
                    done += 1
                    if not is_taken(trial) and trial["scores"]["overall"] > best["scores"]["overall"] + 0.05:
                        best, improved = trial, True
                        pool = [v for v in pool if v != in_v] + [out_v]
                        break
                if improved or done >= SWAP_BUDGET:
                    break
        take("overall", [best])

    out = []
    for mode in ("shape", "story", "overall"):
        note = ""
        if mode == "story" and not story_on:
            note = ("This track's story map has no 🎬 Intro / 🌙 Slow / 💥 Action / 🏁 Outro parts — make one in the "
                    "Media Library (🎭 Story map) to get story-fit plans.")
        elif mode not in chosen:
            note = "No plan found — the library needs more videos with motion data."
        out.append({"mode": mode, "plan": chosen.get(mode), "note": note})

    # 🥊 Verses — a cutting style none of the top three use, two similar videos with different actors.
    tick(f"{MODE_LABELS['verses']}: finding a pair…")
    out.append(_verses(variants, vids, role_secs, actors, similar_of, [chosen.get(m) for m in ("shape", "story", "overall")],
                       check, tick))

    # 🎲 Wildcard — a random style and random videos.
    tick(f"{MODE_LABELS['wildcard']}: rolling the dice…")
    out.append(wildcard(track, prints, vids, by_key, story_regions, n_videos, wildcard_seed, check))

    progress(1.0, "Done")
    for r in out:
        if r["plan"]:
            r["plan"] = {k: v for k, v in r["plan"].items() if k != "_variant"}
            r["plan"]["mode"] = r["mode"]
    return out


def library_insights(v: Variant, role_secs: list) -> dict:
    """What the 🧩 Hidden gems / 🕳️ Gaps / 🆕 spotlight feeds need from one track's search (its
    Smart cut): each video's fit on its own (0–1, shape + motion + story, as Best overall
    judges it), and each block's best fit anywhere in the library."""
    value_of, w = objective(v, "overall", role_secs)
    wsum = max(float(w.sum()), 1e-9)
    solo = {vid: round(float(value_of([k]) @ w) / wsum, 4) for k, vid in enumerate(v.vids)}
    best = np.max(v.S, axis=0) if len(v.vids) else np.zeros(len(v.segments))
    blocks = [{"len": round(sg["end"] - sg["start"], 2), "intensity": round(float(sg.get("intensity", 0.5)), 3),
               "role": r, "best": round(float(b), 3)} for sg, r, b in zip(v.segments, v.roles, best)]
    return {"variant": v.label, "videos": solo, "blocks": blocks,
            "need": {r: round(x, 1) for r, x in v.need().items()}}


def plan_with_video(track: dict, prints: dict, video_id: str, *, match, curves_of, n_videos: int = 4,
                    story_regions: list = None, role_seconds: dict = None, roles_of=None,
                    sequential: bool = False) -> dict | None:
    """🆕 Spotlight: the best plan for `track` that includes `video_id` (Smart, normal pace):
    the video plus the videos that complete it best (shape + motion + story), checked with
    the real matcher. None if the video has no motion data."""
    if video_id not in prints:
        return None
    v = make_variant(track, *TOP_STYLE, prints, story_regions)
    role_secs = [(role_seconds or {}).get(vid, {}) for vid in v.vids]
    value_of, w = objective(v, "overall", role_secs)
    wsum = max(float(w.sum()), 1e-9)
    seed = v.vids.index(video_id)
    chosen = [seed]
    while len(chosen) < min(n_videos, len(v.vids)):
        base = float(value_of(chosen) @ w) / wsum
        gains = [(float(value_of(chosen + [k]) @ w) / wsum - base, k) for k in range(len(v.vids)) if k not in chosen]
        if not gains:
            break
        chosen.append(max(gains)[1])
    videos = sorted(v.vids[k] for k in chosen)
    use_story = v.story_active
    timeline = match(v.segments, videos, list(v.roles) if use_story else None, sequential)
    scale = {vid: p.scale for vid, p in prints.items()}
    return {"mode": "spotlight", "variant": v.label, "style": v.style, "pace": v.pace, "method": v.method,
            "settings": v.settings, "videos": videos, "story": use_story, "sequential": bool(sequential),
            "estimate": float(value_of(chosen) @ w) / wsum, "featured": video_id,
            "scores": score_timeline(track, v.segments, timeline, v.weights, scale, curves_of, v.roles, roles_of),
            "timeline": timeline, "segments": v.segments}


def _shared_actor(a: list, b: list) -> bool:
    return bool({x.lower() for x in a} & {x.lower() for x in b})


def _verses(variants, vids, role_secs, actors, similar_of, top3, check, tick) -> dict:
    tagged = [k for k, vid in enumerate(vids) if actors.get(vid)]
    if len({a.lower() for k in tagged for a in actors[vids[k]]}) < 2 or len(tagged) < 2:
        return {"mode": "verses", "plan": None,
                "note": "Verses needs videos tagged with at least two different actors — add them in the Reviewer "
                        "(🎬 Video details)."}
    used_styles = {p["style"] for p in top3 if p} | {TOP_STYLE[0]}
    styles = [v for v in variants if v.style not in used_styles] or variants
    # one variant per remaining style (normal pace first)
    seen, pick_from = set(), []
    for v in sorted(styles, key=lambda v: abs(v.pace)):
        if v.style not in seen:
            seen.add(v.style)
            pick_from.append(v)

    def pair_value(v, a, b):
        value_of, w = objective(v, "overall", role_secs)
        return float(value_of([a, b]) @ w) / max(float(w.sum()), 1e-9)

    def solo(v):
        value_of, w = objective(v, "overall", role_secs)
        return sorted(tagged, key=lambda k: -float(value_of([k]) @ w))

    def partners(v, seed):
        sims = similar_of(vids[seed]) if similar_of else []
        order = [vid for _, _, vid in sims] + [vid for vid in vids if vid not in {s[2] for s in sims}]
        idx = {vid: k for k, vid in enumerate(vids)}
        pool = [idx[vid] for vid in order if vid in idx and vid != vids[seed] and actors.get(vid)
                and not _shared_actor(actors[vid], actors[vids[seed]])][:VERSES_POOL]
        return sorted(pool, key=lambda k: -pair_value(v, seed, k))

    # The style whose best pair (top pick + best similar partner) fits the song best.
    best_v, best_val = None, -1.0
    for v in pick_from:
        for seed in solo(v)[:1]:
            ps = partners(v, seed)
            if ps and pair_value(v, seed, ps[0]) > best_val:
                best_v, best_val = v, pair_value(v, seed, ps[0])
    if best_v is None:
        return {"mode": "verses", "plan": None,
                "note": "No two videos with different actors could be paired — tag more videos' actors in the "
                        "Reviewer (🎬 Video details)."}
    tried = []
    for seed in solo(best_v):
        if len(tried) >= VERSES_SEEDS:
            break
        ps = partners(best_v, seed)
        if not ps:
            continue
        tick(f"{MODE_LABELS['verses']}: checking a pair with the real matcher…")
        tried.append(check(best_v, [vids[seed], vids[ps[0]]], "verses", pair_value(best_v, seed, ps[0])))
    best = max(tried, key=lambda p: p["scores"]["overall"])
    plan = dict(best, mode="verses", actors={vid: list(actors.get(vid) or []) for vid in best["videos"]})
    return {"mode": "verses", "plan": plan, "note": ""}


def wildcard(track, prints, vids, by_key, story_regions, n_videos, seed, check) -> dict:
    """🎲 A random cutting style (calmer / normal / busier) and n random videos, matched for
    the best overall fit. Same seed → same roll."""
    from core.styles import STYLES
    rng = np.random.default_rng(int(seed))
    style = list(STYLES)[int(rng.integers(len(STYLES)))]
    pace = int(rng.integers(-1, 2))
    v = by_key.get((style, pace)) or make_variant(track, style, pace, prints, story_regions)
    by_key[(style, pace)] = v
    videos = sorted(rng.choice(vids, size=min(n_videos, len(vids)), replace=False).tolist())
    plan = check(v, videos, "wildcard")
    return {"mode": "wildcard", "plan": dict(plan, mode="wildcard"), "note": ""}


def reroll_wildcard(track: dict, prints: dict, *, match, curves_of, n_videos: int, story_regions: list = None,
                    roles_of=None, sequential: bool = False, seed: int = 0) -> dict:
    """Just the 🎲 Wildcard (for the page's 'Roll again')."""
    scale = {v: p.scale for v, p in prints.items()}
    vids = sorted(prints)

    def check(v, videos, mode, estimate=0.0):
        use_story = v.story_active
        timeline = match(v.segments, list(videos), list(v.roles) if use_story else None, sequential)
        return {"variant": v.label, "style": v.style, "pace": v.pace, "method": v.method, "settings": v.settings,
                "videos": list(videos), "story": use_story, "sequential": bool(sequential), "estimate": estimate,
                "scores": score_timeline(track, v.segments, timeline, v.weights, scale, curves_of, v.roles, roles_of),
                "timeline": timeline, "segments": v.segments}
    return wildcard(track, prints, vids, {}, story_regions, max(1, min(n_videos, len(vids))), seed, check)


def library_signature(prints_meta: dict, track_meta) -> str:
    """A key for caching results: changes when any catalogue or the track changes."""
    import hashlib
    import json
    return hashlib.sha1(json.dumps([sorted(prints_meta.items()), track_meta], default=str).encode()).hexdigest()[:16]


def print_size_mb(prints: dict) -> float:
    """Memory held by fingerprints, in MB (for the page to report)."""
    return sum(z.nbytes + lv.nbytes for p in prints.values() for z, lv in p.by_len.values()) / 2**20


__all__ = ["VideoPrint", "fingerprint_video", "block_weights", "cut_variant", "block_prints", "score_matrix",
           "choose_sets", "greedy_sets", "score_timeline", "recommend_modes", "reroll_wildcard", "make_variant",
           "library_insights", "plan_with_video",
           "library_signature", "VARIANTS", "MODES", "DROP_WEIGHT"]
