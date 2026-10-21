"""In-order mode, planned per video ("plan each video's path in one go").

Auto mode with "Keep each video's clips in order" used to pick block by block:
each pick could only skip ahead into a video's "spare" footage (a reserve worked
out from its share of the blocks still to come), and weight F both opened that
spare footage and pulled picks towards part k of the video. Early blocks took
the good clips, the reserve maths was rough, and one slider did two jobs.

Now it's two passes:
  1. Which video goes in each block — the normal Auto matcher (fit to the music,
     use all videos evenly, never repeat a neighbour, story roles, Swap / Reject).
  2. For each video, the clips for ALL its blocks chosen together, in time order:
     every window of its usable footage is scored for every one of its blocks
     (shape + motion level, as the matcher weighs them, plus the block's story
     role), and a dynamic programme picks one window per block so that each
     starts after the last one ends and the total is as high as possible. One
     setting, F ("Follow the video ↔ Follow the music"), adds a pull towards an
     even walk through the whole video: use k of n aims at the middle of part k.

So order is guaranteed, a video can't run short halfway, and its whole usable
range is in play. A block whose video can't fit it in order (rare: a video with
less in-order footage than its blocks need) keeps its pass-1 clip, which may be
out of order — reported in `notes`.

Pure functions, no Streamlit.
"""

from __future__ import annotations

import numpy as np

from core.shape import get_block_audio_curve

POINTS = 32            # curves resampled to this many points for the shape score
STEP_SEC = 0.1         # candidate windows start this often (more coarsely on very long videos)
MAX_WINDOWS = 20000    # per block and video — the step widens past this
ROLE_BONUS = 0.5       # a window from a clip marked with the block's role
SKIP_PENALTY = 3.0     # leaving a block on its pass-1 clip costs more than any window's score
MAX_FOLLOW = 1.0       # F at its top (5) adds up to this much for the window right on its target


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=-1, keepdims=True)
    return np.where(sd > 1e-9, (x - x.mean(axis=-1, keepdims=True)) / np.where(sd > 1e-9, sd, 1.0), 0.0)


def _resample_rows(win: np.ndarray, n: int = POINTS) -> np.ndarray:
    m = win.shape[1]
    if m < 2:
        return np.repeat(win[:, :1], n, axis=1)
    x_old = np.linspace(0, 1, m)
    x_new = np.linspace(0, 1, n)
    idx = np.clip(np.searchsorted(x_old, x_new, side="right") - 1, 0, m - 2)
    frac = (x_new - x_old[idx]) / (x_old[idx + 1] - x_old[idx])
    return win[:, idx] * (1 - frac) + win[:, idx + 1] * frac


FLAT_FPS = 10.0        # the made-up curve for a scene with no motion curve
CONTIGUOUS_SEC = 0.1   # scenes this close together are one run of footage (a window may cross them)


def _runs(spans: list, curves: dict, cross_scenes: bool = True) -> list:
    """Back-to-back usable footage as runs: [{"start": abs s, "fps", "values": array, "pieces": [(abs start,
    abs end, span)]}] — a window inside a run can cross from one scene into the next (it becomes a chain).
    cross_scenes=False: one run per scene (Manual step-through and Choreography pick within one scene)."""
    runs = []
    for sp in sorted(spans, key=lambda x: x["scene_start_sec"] + x["offset_sec"]):
        c = curves[sp["scene_id"]]
        fps = float(c["fps"])
        vals = np.asarray(c["values"], dtype=float)
        a = int(round(sp["offset_sec"] * fps))
        b = min(len(vals), int(round((sp["offset_sec"] + sp["remaining_sec"]) * fps)))
        if b - a < 2:
            continue
        t0 = sp["scene_start_sec"] + sp["offset_sec"]
        t1 = t0 + (b - a) / fps
        r = runs[-1] if runs else None
        if cross_scenes and r and abs(r["fps"] - fps) < 1e-6 and abs(t0 - r["pieces"][-1][1]) <= CONTIGUOUS_SEC:
            r["values"] = np.r_[r["values"], vals[a:b]]
            r["pieces"].append((r["pieces"][-1][1], r["pieces"][-1][1] + (b - a) / fps, sp))
        else:
            runs.append({"start": t0, "fps": fps, "values": vals[a:b], "pieces": [(t0, t1, sp)]})
    return runs


def _windows(run: dict, dur: float, step_sec: float = STEP_SEC):
    """(starts_abs, z-shapes [n, POINTS], levels_raw [n]) for windows of `dur` seconds inside the run,
    starting every step_sec (and at its very end)."""
    fps = run["fps"]
    seg = run["values"]
    wf = max(2, int(round(dur * fps)))
    if len(seg) < wf:
        return None
    step = max(1, int(round(step_sec * fps)))
    win = np.lib.stride_tricks.sliding_window_view(seg, wf)
    idx = np.unique(np.r_[np.arange(0, len(win), step), len(win) - 1])
    win = win[idx]
    starts = run["start"] + idx / fps
    return starts, _z(_resample_rows(win)), win.mean(axis=1)


def _chain(run: dict, t: float, dur: float) -> list:
    """The window [t, t + dur] of a run, cut at scene boundaries: [(span, clip start abs, seconds)]."""
    out = []
    for a, b, sp in run["pieces"]:
        lo, hi = max(a, t), min(b, t + dur)
        if hi - lo > 1e-3:
            out.append((sp, lo, hi - lo))
    return out


def plan_video_paths(timeline: list, segments: list, queues: dict, track: dict, curves_of, *,
                     block_roles: list = None, exclusive_roles=(), segment_exclusions: dict = None,
                     global_excluded=(), follow: float = 0.2, weight_shape: float = 2.0,
                     weight_motion: float = 2.0, cross_scenes: bool = True) -> tuple:
    """Pass 2 (see the module docstring). timeline: pass 1's (Auto, any order); queues: the footage
    BEFORE pass 1 used any (build_footage_queues); follow: 0–1, how strongly each video's clips are
    spread evenly through it (0 = purely the best fit to the music, still in order).
    Returns (timeline, notes)."""
    segment_exclusions = segment_exclusions or {}
    global_excluded = set(global_excluded or ())
    exclusive_roles = set(exclusive_roles or ())
    ws, wm = max(0.0, float(weight_shape)), max(0.0, float(weight_motion))
    if ws + wm <= 0:
        ws = wm = 1.0
    seg_by_idx = {i: sg for i, sg in enumerate(segments)}
    bprint = {}

    # A use = one video's part of a slot: a slot's chain is split where it changes video, and each part is
    # planned in its own video's path (block order, then slot, then part within the slot).
    uses = {}                                    # video → [(block index, slot index, part index, seconds)]
    parts = {}                                   # (block index, slot index) → [[chain links of one video], …]
    for entry in timeline:
        for s, slot in enumerate(entry["scenes"]):
            groups = []
            for c in slot["chain"]:
                if groups and groups[-1][0]["video_id"] == c["video_id"]:
                    groups[-1].append(c)
                else:
                    groups.append([c])
            parts[(entry["segment_index"], s)] = groups
            for g, links in enumerate(groups):
                uses.setdefault(links[0]["video_id"], []).append(
                    (entry["segment_index"], s, g, sum(float(c["clip_duration_sec"]) for c in links)))
    out = [dict(e, scenes=[dict(sl) for sl in e["scenes"]]) for e in timeline]
    pos = {e["segment_index"]: k for k, e in enumerate(out)}
    notes = []

    for vid, vuses in uses.items():
        spans = [sp for sp in queues.get(vid, []) if sp["remaining_sec"] > 0]
        curves = dict(curves_of(vid) or {})
        for sp in spans:     # a scene without a motion curve: a flat line at its motion score (no shape to match)
            c = curves.get(sp["scene_id"])
            if not c or c.get("values") is None or len(c["values"]) < 2:
                n = int((sp["offset_sec"] + sp["remaining_sec"]) * FLAT_FPS) + 2
                curves[sp["scene_id"]] = {"fps": FLAT_FPS, "values": np.full(n, float(sp.get("motion_raw") or 0.0))}
        if not spans:
            continue
        allv = np.concatenate([np.asarray(curves[sp["scene_id"]]["values"], dtype=float) for sp in spans])
        scale = float(np.percentile(allv, 95)) or float(allv.max()) or 1.0
        lo = min(sp["scene_start_sec"] + sp["offset_sec"] for sp in spans)
        hi = max(sp["scene_start_sec"] + sp["offset_sec"] + sp["remaining_sec"] for sp in spans)
        length = max(hi - lo, 1e-6)
        k = len(vuses)
        step_sec = max(STEP_SEC, sum(sp["remaining_sec"] for sp in spans) / MAX_WINDOWS)
        has_role = {r for sp in spans for r in (sp.get("roles") or [])}

        # Every candidate window for every use, scored.
        use_runs = []
        cand = []                                # per use: (starts, ends, scores, (span index per window))
        for j, (bi, si, gi, dur) in enumerate(vuses):
            sg = seg_by_idx[bi]
            if bi not in bprint:
                c = np.asarray(get_block_audio_curve(track, sg["start"], sg["end"]), dtype=float)
                bprint[bi] = _z(_resample_rows(c[None, :]))[0] if len(c) else np.zeros(POINTS)
            role = block_roles[bi] if block_roles else None
            only_role = role in exclusive_roles and role in has_role
            excl = set(segment_exclusions.get(bi, ())) | global_excluded
            target = lo + (j + 0.5) / k * length
            part = length / k
            S, E, V, W = [], [], [], []
            allowed = [sp for sp in spans if (vid, sp["scene_id"]) not in excl
                       and (not only_role or role in (sp.get("roles") or []))]
            runs = _runs(allowed, curves, cross_scenes)
            use_runs.append(runs)
            for n, run in enumerate(runs):
                w = _windows(run, dur, step_sec)
                if w is None:
                    continue
                starts, shapes, levels = w
                # a window counts as the role's when it starts in a clip marked with it
                marked = np.zeros(len(starts), dtype=bool)
                if role:
                    for a, b, sp in run["pieces"]:
                        if role in (sp.get("roles") or []):
                            marked |= (starts >= a - 1e-6) & (starts < b)
                corr = np.clip(shapes @ bprint[bi] / POINTS, 0, 1)
                level = 1.0 - np.abs(np.clip(levels / scale, 0, 1) - float(sg.get("intensity", 0.5)))
                fit = (ws * corr + wm * level) / (ws + wm) + ROLE_BONUS * marked
                prior = 1.0 - np.minimum(1.0, np.abs(starts + dur / 2 - target) / part)
                S.append(starts)
                E.append(starts + dur)
                V.append(fit + follow * MAX_FOLLOW * prior)
                W.append(np.full(len(starts), n))
            if S:
                cand.append((np.concatenate(S), np.concatenate(E), np.concatenate(V), np.concatenate(W)))
            else:
                cand.append(None)

        # DP: best[j][w] = score of the best in-order path whose last TAKEN use is j at window w.
        best, back = [], []
        for j, c in enumerate(cand):
            if c is None:
                best.append(None)
                back.append(None)
                continue
            starts, _, vals, _ = c
            val = vals - SKIP_PENALTY * j                     # nothing taken before j
            bk = np.full((len(vals), 2), -1)
            for jp in range(j):
                if best[jp] is None:
                    continue
                ends_p = cand[jp][1]
                order = np.argsort(ends_p)
                v_sorted = best[jp][order]
                pref = np.maximum.accumulate(v_sorted)                       # best path ending by each end time
                arg_full = order[np.maximum.accumulate(np.where(v_sorted >= pref, np.arange(len(order)), 0))]
                n_ok = np.searchsorted(ends_p[order], starts + 1e-6, side="right")
                ok = n_ok > 0
                cand_val = np.full(len(vals), -np.inf)
                cand_val[ok] = pref[n_ok[ok] - 1] - SKIP_PENALTY * (j - jp - 1) + vals[ok]
                better = cand_val > val
                val = np.where(better, cand_val, val)
                bk[better, 0] = jp
                bk[better, 1] = arg_full[n_ok[better] - 1]
            best.append(val)
            back.append(bk)
        # the best finish
        fin_val, fin = -np.inf, None
        for j in range(k):
            if best[j] is None or not len(best[j]):
                continue
            w = int(np.argmax(best[j]))
            v = best[j][w] - SKIP_PENALTY * (k - 1 - j)
            if v > fin_val:
                fin_val, fin = v, (j, w)
        taken = {}
        while fin is not None:
            j, w = fin
            taken[j] = w
            jp, wp = back[j][w]
            fin = (int(jp), int(wp)) if jp >= 0 else None

        for j, (bi, si, gi, dur) in enumerate(vuses):
            if j not in taken:
                notes.append(f"{vid}: block {bi + 1} kept its first-pass clip (not enough in-order footage)")
                continue
            starts, _, _, which = cand[j]
            w = taken[j]
            run = use_runs[j][int(which[w])]
            parts[(bi, si)][gi] = [
                {"video_id": vid, "scene_id": sp["scene_id"], "tags": sp["tags"], "thumbnail": sp["thumbnail"],
                 "motion_norm": sp["motion_norm"], "clip_start_sec": round(float(t0), 3),
                 "clip_duration_sec": round(float(d), 3),
                 "offset_into_scene_sec": round(float(t0) - sp["scene_start_sec"], 2)}
                for sp, t0, d in _chain(run, float(starts[w]), dur)]

    # Rebuild each slot from its (re-planned) parts.
    for (bi, si), groups in parts.items():
        chain = [c for g in groups for c in g]
        first = chain[0]
        out[pos[bi]]["scenes"][si] = {
            "chain": chain, "video_id": first["video_id"], "scene_id": first["scene_id"],
            "tags": sorted({t for c in chain for t in c["tags"]}), "motion_norm": first["motion_norm"],
            "thumbnail": first["thumbnail"], "clip_duration_sec": round(sum(c["clip_duration_sec"] for c in chain), 3),
            "offset_into_scene_sec": first["offset_into_scene_sec"], "clip_start_sec": first["clip_start_sec"]}
    return out, notes


def in_order(timeline: list) -> bool:
    """Every video's clips start in time order through the plan (and don't overlap)."""
    last = {}
    for entry in timeline:
        for slot in entry["scenes"]:
            for c in slot["chain"]:
                v = c["video_id"]
                if v in last and c["clip_start_sec"] < last[v] - 1e-3:
                    return False
                last[v] = c["clip_start_sec"] + c["clip_duration_sec"]
    return True


__all__ = ["plan_video_paths", "in_order"]


# ---------------------------------------------------------------------------
# 📍 The planned path, for the step-by-step modes
# ---------------------------------------------------------------------------

def planned_picks(timeline: list, segments: list, offset: int = 0) -> dict:
    """{block index (+ offset): [pick, …]} — the single-scene, whole-block clips of an in-order
    plan, in the shape Manual step-through and Choreography confirm ("video_id", "scene_id",
    "offset_into_scene_sec", "clip_duration_sec", "clip_start_sec", …). A slot that's a chain
    of several clips, or shorter than its block, isn't one of their picks and is left out."""
    out = {}
    for entry in timeline:
        bi = entry["segment_index"]
        sg = segments[bi]
        need = float(sg["end"]) - float(sg["start"])
        picks = []
        for slot in entry["scenes"]:
            chain = slot.get("chain") or []
            if len(chain) != 1 or abs(float(chain[0]["clip_duration_sec"]) - need) > 0.05:
                continue
            picks.append(dict(chain[0]))
        out[bi + offset] = picks
    return out
