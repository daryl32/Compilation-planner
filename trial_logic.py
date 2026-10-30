"""
trial_logic.py — the plain-Python parts of the Modal trial (modal_trial.py): no GPU, no Modal,
no torch, so they can be tested anywhere.

  • which moments to tag in each video          tag_targets()
  • the tagging prompt and reading the answer    build_prompt(), parse_answer()
  • movement rhythm from a person-motion curve   rhythm_features(), person_energy()
  • scoring against your 🎯 Training Labels       score_tags(), score_boundaries()
  • what it cost and what the full run would     stage_cost(), extrapolate()
"""

from __future__ import annotations

import csv
import io
import json
import re

import numpy as np

# Modal list prices, $ per second (modal.com/pricing, Oct 2026).
GPU_RATE = {"H100": 0.001097, "A100-80GB": 0.000694, "A100-40GB": 0.000583, "L40S": 0.000542,
            "A10": 0.000306, "L4": 0.000222, "T4": 0.000164, "CPU": 0.0}
CPU_CORE_RATE = 0.0000131
MEM_GIB_RATE = 0.00000222

CONTEXT_OFFSETS = (-2.0, -1.33, -0.67, 0.67, 1.33, 2.0)   # extra frames around a moment (motion tags)
LONG_SCENE_EVERY = 10.0                                   # one more moment every 10 s in long scenes
MAX_PER_SCENE = 6
POSE_FPS = 15.0


# ---------------------------------------------------------------------------
# Vocabulary (same file and rules as training_labels.py, kept separate so the GPU image
# needs nothing from the app)
# ---------------------------------------------------------------------------

def parse_vocabulary(text: str) -> list:
    text = (text or "").lstrip("﻿")
    first = text.splitlines()[0] if text.strip() else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    rows, seen = [], set()
    for raw in csv.DictReader(io.StringIO(text), delimiter=delim):
        raw = {str(k).strip().lower(): (v or "") for k, v in raw.items() if k}
        tag = " ".join(raw.get("tag", "").split())
        if not tag or tag.lower() in seen:
            continue
        seen.add(tag.lower())
        kind = raw.get("kind", "").strip().lower()
        teacher = raw.get("teacher", "").strip().lower()
        rows.append({"tag": tag, "group": raw.get("group", "").strip(),
                     "kind": kind if kind in ("frame", "motion") else "frame",
                     "one_of": " ".join(raw.get("one_of", "").split()),
                     "teacher": teacher in ("", "yes", "y", "true", "1", "x"),
                     "description": raw.get("description", "").strip(),
                     "not_to_be_confused_with": raw.get("not_to_be_confused_with", "").strip(),
                     "never_with": [n.strip() for n in raw.get("never_with", "").replace(",", ";").split(";")
                                    if n.strip()]})
    return rows


def never_pairs(vocab: list) -> dict:
    """{tag (lower case): {tags it can never go with}} from never_with, both ways, only between tags
    in this vocabulary (so a rule naming a tag the teacher isn't asked about is simply ignored)."""
    names = {r["tag"].lower(): r["tag"] for r in vocab}
    out = {}
    for r in vocab:
        for other in r.get("never_with") or []:
            o = names.get(str(other).strip().lower())
            if o and o.lower() != r["tag"].lower():
                out.setdefault(r["tag"].lower(), set()).add(o)
                out.setdefault(o.lower(), set()).add(r["tag"])
    return out


def teacher_vocab(vocab: list) -> list:
    """The tags the teacher is asked about: the ones whose "teacher" switch is on. Tags switched
    off are learned from your labels only, so they're left out of its prompt altogether."""
    return [r for r in vocab if r.get("teacher", True)]


def vocabulary_from_labels(labels) -> list:
    out = []
    for t in labels or []:
        name = t if isinstance(t, str) else (t or {}).get("name") or (t or {}).get("label")
        if name:
            out.append({"tag": str(name), "group": "", "kind": "frame", "one_of": "", "teacher": True,
                        "description": "", "not_to_be_confused_with": "", "never_with": []})
    return out


# ---------------------------------------------------------------------------
# Moments to tag
# ---------------------------------------------------------------------------

def tag_targets(manifest: dict, gold_items: list, max_seconds: float | None = None) -> list:
    """[{"key", "t", "scene_id", "gold_id"?}] — a moment in the middle of every usable scene
    (more in long ones), plus every 🎯 tag card labelled in this video (so the teacher can be
    scored on exactly those frames)."""
    out, seen = [], set()
    vid = manifest["video_id"]
    for s in manifest.get("scenes") or []:
        if s.get("excluded") or "start" not in s:
            continue
        a, b = float(s["start"]), float(s["end"])
        if b - a < 0.5:
            continue
        n = min(MAX_PER_SCENE, max(1, int((b - a) // LONG_SCENE_EVERY) + 1))
        for k in range(n):
            t = round(a + (b - a) * (k + 0.5) / n, 2)
            if max_seconds is None or t <= max_seconds:
                key = f"s{s['scene_id']}_{k}"
                out.append({"key": key, "t": t, "scene_id": s["scene_id"]})
                seen.add(round(t, 1))
    for it in gold_items:
        if it.get("video_id") != vid or it.get("type") != "tags" or it.get("skipped") or it.get("t") is None:
            continue
        t = float(it["t"])
        if max_seconds is None or t <= max_seconds:
            out.append({"key": f"g_{t:.2f}", "t": round(t, 2), "scene_id": it.get("scene_id"),
                        "gold_id": it["id"]})
    return out


def frame_times(targets: list, duration: float) -> list:
    """Every moment a frame is needed for (each target plus its context frames), sorted."""
    ts = set()
    for tg in targets:
        for off in (0.0,) + CONTEXT_OFFSETS:
            ts.add(round(min(max(tg["t"] + off, 0.0), max(duration - 0.05, 0.0)), 2))
    return sorted(ts)


def frame_name(t: float) -> str:
    return f"{t:09.2f}.jpg"


def context_times(t: float, duration: float) -> list:
    """The 7 frame times for one target, in time order, the target itself in the middle."""
    return [round(min(max(t + off, 0.0), max(duration - 0.05, 0.0)), 2)
            for off in (-2.0, -1.33, -0.67, 0.0, 0.67, 1.33, 2.0)]


# ---------------------------------------------------------------------------
# Near-duplicate moments: tag one, copy to the rest
# ---------------------------------------------------------------------------

DEDUP_DEFAULT = 0.96        # cosine similarity (picture embeddings) above which two moments in the
                            # same scene count as the same; 0 = tag every moment
DEDUP_WINDOW = 2.0          # seconds either side of a moment that must match too (motion tags)


def window_embedding(times, embs, t: float, half: float = DEDUP_WINDOW):
    """Unit-length mean of the picture embeddings within t ± half seconds (None if there are none)."""
    times = np.asarray(times, dtype=float)
    sel = (times >= t - half) & (times <= t + half)
    if not sel.any():
        return None
    v = np.asarray(embs, dtype=np.float32)[sel].mean(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n else None


def dedup_targets(targets: list, times, embs, threshold: float = DEDUP_DEFAULT) -> tuple:
    """Split a video's moments into the ones to send to the teacher and the ones that can copy a
    tagged moment's answer: (keep, copies). copies = {key: {"of": kept key, "similarity": s}}.

    Two moments are "the same" only when they're in the same scene and BOTH their middle frame and
    their ±2 s window of frames match above the threshold — so a still scene where nothing changes
    is tagged once, while a scene whose movement differs from one moment to the next is tagged at
    every moment. Your labelled frames (gold) are always sent, so the report's scores are real."""
    if not threshold or threshold <= 0 or times is None or embs is None or len(times) == 0:
        return list(targets), {}
    times = np.asarray(times, dtype=float)
    embs = np.asarray(embs, dtype=np.float32)
    feats = {}
    for tg in targets:
        j = nearest_index(times, float(tg["t"]))
        if j is None or abs(times[j] - float(tg["t"])) > 1.0:
            continue
        mid = embs[j] / (np.linalg.norm(embs[j]) or 1.0)
        win = window_embedding(times, embs, float(tg["t"]))
        if win is not None:
            feats[tg["key"]] = (mid, win)
    keep, copies, reps = [], {}, {}        # reps: scene_id -> [(key, mid, win)]
    for tg in sorted(targets, key=lambda x: float(x["t"])):
        f = feats.get(tg["key"])
        sc = tg.get("scene_id")
        if f is None or tg.get("gold_id") or sc is None:
            keep.append(tg)
            if f is not None and sc is not None:
                reps.setdefault(sc, []).append((tg["key"], *f))
            continue
        best, best_s = None, 0.0
        for key, mid, win in reps.get(sc, []):
            s = min(float(mid @ f[0]), float(win @ f[1]))
            if s >= threshold and s > best_s:
                best, best_s = key, s
        if best is None:
            keep.append(tg)
            reps.setdefault(sc, []).append((tg["key"], *f))
        else:
            copies[tg["key"]] = {"of": best, "similarity": round(best_s, 4)}
    return keep, copies


# ---------------------------------------------------------------------------
# How hard the GPU worked
# ---------------------------------------------------------------------------

# Dense BF16 peak, TFLOPS (NVIDIA spec sheets) — what "model FLOPs utilisation" is measured against.
GPU_PEAK_TFLOPS = {"H100": 989.0, "A100-80GB": 312.0, "A100-40GB": 312.0, "L40S": 362.0, "A10": 125.0,
                   "L4": 121.0, "T4": 65.0}
TEACHER_PARAMS = {"8b": 8.8e9, "32b": 33.0e9, "32b-fp8": 33.0e9}


def gpu_use(key: str, gpu: str, tokens: int, seconds: float) -> dict:
    """{"tokens_per_s", "mfu"}: tokens the teacher processed per second, and the share of the GPU's
    peak maths that took (2 × parameters × tokens ÷ time ÷ peak). Low MFU = the GPU mostly waited."""
    if not seconds or seconds <= 0:
        return {"tokens_per_s": None, "mfu": None}
    tps = tokens / seconds
    peak = GPU_PEAK_TFLOPS.get(gpu)
    params = TEACHER_PARAMS.get(key)
    mfu = (2.0 * params * tps / (peak * 1e12)) if peak and params else None
    return {"tokens_per_s": round(tps), "mfu": None if mfu is None else round(mfu, 3)}


# ---------------------------------------------------------------------------
# The tagging prompt
# ---------------------------------------------------------------------------

def build_prompt(vocab: list) -> tuple:
    """(head, moment, tail) for one moment shown as 7 frames over 4 seconds.

    head — everything that is the same for every moment (the task, the tag lists, the rules,
           the reply format). It comes FIRST, before any picture, so vLLM's prefix cache keeps
           it and the GPU reads it once per run instead of once per moment.
    moment — the line just before the moment's 7 frames.
    tail — the short line after them."""
    frame = [r for r in vocab if r["kind"] == "frame"]
    motion = [r for r in vocab if r["kind"] == "motion"]

    def line(r):
        bits = [f"- {r['tag']}"]
        if r["description"]:
            bits.append(f": {r['description']}")
        if r["not_to_be_confused_with"]:
            bits.append(f" (not the same as: {r['not_to_be_confused_with']})")
        return "".join(bits)

    groups = {}
    for r in vocab:
        if r["one_of"]:
            groups.setdefault(r["one_of"], []).append(r["tag"])
    head = ["You are labelling footage for a video-editing library. Each moment to label is shown "
            "as 7 frames spanning about 4 seconds of one clip, in time order. Frame 4 (the middle "
            "one) is the moment being labelled.",
            "", "Tags judged from frame 4 alone (what is visible in that frame):"]
    head += [line(r) for r in frame] or ["(none)"]
    head += ["", "Tags judged from the movement across all 7 frames:"]
    head += [line(r) for r in motion] or ["(none)"]
    if groups:
        head += ["", "Pick at most one tag from each of these sets:"]
        head += [f"- {name}: {', '.join(tags)}" for name, tags in groups.items()]
    pairs = never_pairs(vocab)
    if pairs:
        order = {r["tag"]: i for i, r in enumerate(vocab)}
        rules = sorted({tuple(sorted((r["tag"], o), key=order.get)) for r in vocab
                        for o in pairs.get(r["tag"].lower(), ())}, key=lambda p: (order[p[0]], order[p[1]]))
        head += ["", "These tags never apply to the same moment — if both seem to, give only the one that "
                     "fits best:"]
        head += [f"- {a} / {b}" for a, b in rules]
    head += ["", "Only use tags from the lists above, spelled exactly as written. Choose every tag "
             "that clearly applies; leave out anything you are not sure of, and list those under "
             "\"unsure\" instead. Reply with JSON only, no other text:",
             '{"tags": ["..."], "unsure": ["..."], "caption": "one short neutral sentence"}']
    moment = "The moment to label — 7 frames in time order, frame 4 in the middle:"
    tail = "Reply with the JSON for this moment only."
    return "\n".join(head), moment, tail


# ---------------------------------------------------------------------------
# Worked examples from your own labels (shown to the teacher before each moment)
# ---------------------------------------------------------------------------

EXAMPLES_PER_VIDEO = 2      # at most this many from any one video, so one video can't dominate
CORRECTED_BONUS = 0.05      # frames where you changed the suggested tags teach the most


def nearest_index(times, t: float):
    """Index of the time closest to t in a sorted array, or None if it's empty."""
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return None
    i = int(np.searchsorted(times, t))
    if i == 0:
        return 0
    if i >= times.size:
        return times.size - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def pick_examples(query, pool: list, k: int, exclude_video: str = "") -> list:
    """The k labelled moments in pool most like this one (cosine similarity of their picture
    embeddings), never from exclude_video — so the moment's own label (or a neighbour's) can't
    leak into its answer and the report's scores stay honest. pool items: {"video_id", "emb"
    (unit vector), "corrected" (bool), ...}. Corrected frames get a small bonus."""
    if k <= 0 or query is None or not pool:
        return []
    q = np.asarray(query, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    cand = [p for p in pool if p.get("video_id") != exclude_video and p.get("emb") is not None]
    if not cand:
        return []
    sims = np.asarray([np.asarray(p["emb"], dtype=np.float32) for p in cand]) @ q
    sims = sims + np.asarray([CORRECTED_BONUS if p.get("corrected") else 0.0 for p in cand])
    out, per_video = [], {}
    for i in np.argsort(-sims, kind="stable"):
        p = cand[int(i)]
        if per_video.get(p["video_id"], 0) >= EXAMPLES_PER_VIDEO:
            continue
        per_video[p["video_id"]] = per_video.get(p["video_id"], 0) + 1
        out.append(p)
        if len(out) >= k:
            break
    return out


def examples_intro(n: int) -> str:
    return (f"First, {n} moment{'s' if n != 1 else ''} from other clips that the editor has already "
            "labelled by hand, each shown as its middle frame with the correct tags. Use them to "
            "learn how the editor uses each tag. Movement tags in them were judged from the "
            "seconds around the frame.")


def example_caption(i: int, tags: list, vocab: list) -> str:
    """'Example 2 — correct tags: a, b' — only tags the teacher is asked about, in vocabulary order."""
    allowed = {r["tag"].lower(): r["tag"] for r in vocab}
    mine = {str(t).strip().lower() for t in tags or []}
    shown = [allowed[t.lower()] for t in (r["tag"] for r in vocab) if t.lower() in mine]
    return f"Example {i} — correct tags: " + (", ".join(shown) if shown else "none of the listed tags")


def parse_answer(text: str, vocab: list) -> dict:
    """{"tags", "unsure", "caption", "ok"} from the model's reply — tags kept only if they're in
    the vocabulary (any case), at most one per one_of set (the first named wins)."""
    lookup = {r["tag"].lower(): r for r in vocab}
    m = re.search(r"\{.*\}", text or "", re.S)
    data, ok = {}, False
    if m:
        try:
            data = json.loads(m.group(0))
            ok = isinstance(data, dict)
        except ValueError:
            data = {}
    if not isinstance(data, dict):
        data = {}

    def clean(names):
        out, sets = [], set()
        for n in names if isinstance(names, list) else []:
            r = lookup.get(str(n).strip().lower())
            if not r or r["tag"] in out:
                continue
            if r["one_of"]:
                if r["one_of"] in sets:
                    continue
                sets.add(r["one_of"])
            out.append(r["tag"])
        return out

    tags = clean(data.get("tags"))
    # your never-with rules: a tag that clashes with one named before it goes to "unsure"
    pairs, kept, broke = never_pairs(vocab), [], []
    for t in tags:
        if any(o in kept for o in pairs.get(t.lower(), ())):
            broke.append(t)
        else:
            kept.append(t)
    tags = kept
    unsure = [t for t in clean(data.get("unsure")) + broke if t not in tags]
    unsure = list(dict.fromkeys(unsure))
    return {"tags": tags, "unsure": unsure, "caption": str(data.get("caption") or "")[:300], "ok": ok,
            "rule_breaks": broke}


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def cuts_from_predictions(preds, fps: float, threshold: float = 0.5) -> list:
    """Cut times (s) from TransNetV2's per-frame transition probabilities: one cut per run of
    frames above the threshold, at the run's middle; none in the first 0.2 s."""
    p = np.asarray(preds, dtype=float).reshape(-1)
    above = p > threshold
    cuts, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            cuts.append(round(((i + j) / 2 + 0.5) / fps, 2))
            i = j + 1
        else:
            i += 1
    return [c for c in cuts if c > 0.2]


def movement_curve(tracks: dict, n_frames: int) -> np.ndarray:
    """The video's person-motion curve: each tracked person's camera-free body movement (people on
    screen under a second are ignored), then per frame the sum of the two busiest people (NaN where
    nobody is tracked). tracks = {track id: {frame index: (keypoints xy (17, 2), conf (17,), box h)}}."""
    energies = []
    for frames in tracks.values():
        if len(frames) < POSE_FPS:
            continue
        a, b = min(frames), max(frames) + 1
        kps = np.zeros((b - a, 17, 2))
        conf = np.zeros((b - a, 17))
        bh = np.ones(b - a)
        for i, (xy, kc, h_) in frames.items():
            kps[i - a], conf[i - a], bh[i - a] = xy, kc, h_
        full = np.full(n_frames, np.nan)
        full[a:b] = person_energy(kps, conf, bh)
        energies.append(full)
    if not energies:
        return np.full(n_frames, np.nan)
    E = np.vstack(energies)
    top2 = np.sort(np.nan_to_num(E, nan=0.0), axis=0)[-2:].sum(axis=0)
    seen = np.isfinite(E).any(axis=0)
    return np.where(seen, top2, np.nan)


def scene_features(scenes_in: list, duration: float, curve, people: list, emb_times, emb) -> tuple:
    """Per old scene: rhythm, median people on screen, mean picture embedding.
    Returns (scene rows, scene ids with an embedding, their unit-length mean embeddings)."""
    et = np.asarray(emb_times, dtype=float)
    emb = np.asarray(emb)
    rows, scene_ids, scene_embs = [], [], []
    for s in scenes_in:
        if "start" not in s or s["start"] >= duration:
            continue
        a, b = int(s["start"] * POSE_FPS), int(min(s["end"], duration) * POSE_FPS)
        rows.append({"scene_id": s["scene_id"], "start": s["start"], "end": min(s["end"], duration),
                     "people": float(np.median(people[a:b])) if b > a and len(people[a:b]) else 0.0,
                     **rhythm_features(curve[a:b])})
        sel = (et >= s["start"]) & (et < min(s["end"], duration)) if et.size else np.array([], bool)
        if sel.any():
            v = emb[sel].astype(np.float32).mean(axis=0)
            scene_embs.append((v / (np.linalg.norm(v) or 1)).astype(np.float16))
            scene_ids.append(s["scene_id"])
    return rows, scene_ids, scene_embs


def person_energy(kps: np.ndarray, conf: np.ndarray, box_h: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """Body movement of one tracked person, frame by frame, with the camera taken out:
    keypoints relative to the body's own centre, scaled by its height, so a pan or zoom
    doesn't count — only limbs moving. kps (n, 17, 2), conf (n, 17), box_h (n,). Frames
    where the person isn't seen are NaN. Returns (n,) — energy[i] is the move from i-1 to i."""
    n = kps.shape[0]
    out = np.full(n, np.nan)
    if n < 2:
        return out
    good = conf >= min_conf                                                  # (n, 17)
    cnt = good.sum(axis=1)
    centre = (np.where(good[..., None], kps, 0.0).sum(axis=1) / np.maximum(cnt, 1)[:, None])[:, None, :]
    rel = (kps - centre) / np.maximum(box_h, 1.0)[:, None, None]
    both = good[1:] & good[:-1]                                              # (n-1, 17)
    step = np.linalg.norm(np.nan_to_num(rel[1:] - rel[:-1]), axis=2)        # (n-1, 17)
    k = both.sum(axis=1)
    mean = np.where(both, step, 0.0).sum(axis=1) / np.maximum(k, 1)
    out[1:] = np.where(k >= 4, mean, np.nan)
    return out


def rhythm_features(energy, fps: float = POSE_FPS) -> dict:
    """Is the movement regular, and at what pace? From a scene's person-motion curve:
    level (mean), period (s) and movement_bpm of the strongest repeat, periodicity (0–1,
    autocorrelation at that period), and onsets (s) — the moments a new push of movement
    starts. BPM is folded into 70–180, like a track's tempo."""
    e = np.asarray(energy, dtype=float)
    e = np.where(np.isfinite(e), e, np.nan)
    out = {"level": None, "period": None, "movement_bpm": None, "periodicity": 0.0, "onsets": []}
    if np.isfinite(e).sum() < max(8, int(fps)):
        return out
    out["level"] = round(float(np.nanmean(e)), 5)
    x = np.nan_to_num(e - np.nanmean(e))
    lo, hi = max(2, int(0.25 * fps)), min(int(2.0 * fps), len(x) // 2)
    if hi > lo and np.dot(x, x) > 1e-12:
        ac = np.array([np.dot(x[:-k], x[k:]) for k in range(lo, hi + 1)]) / np.dot(x, x)
        k = int(np.argmax(ac)) + lo
        out["period"] = round(k / fps, 3)
        out["periodicity"] = round(float(max(ac.max(), 0.0)), 3)
        bpm = 60.0 / out["period"]
        while bpm < 70:
            bpm *= 2
        while bpm > 180:
            bpm /= 2
        out["movement_bpm"] = round(bpm, 1)
    d = np.diff(np.nan_to_num(e, nan=np.nanmedian(e)))
    if d.size >= 3:
        med = np.median(d)
        mad = np.median(np.abs(d - med)) * 1.4826 or (d.std() or 1.0)
        z = (d - med) / mad
        peaks = np.where((z[1:-1] > 1.5) & (z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:]))[0] + 1
        out["onsets"] = [round((p + 1) / fps, 2) for p in peaks]
    return out


# ---------------------------------------------------------------------------
# Scoring against 🎯 Training Labels
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
    return {"precision": None if p is None else round(p, 3), "recall": None if r is None else round(r, 3),
            "f1": None if f is None else round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def score_tags(pairs: list, vocab_tags: list) -> dict:
    """pairs: [(predicted tags, gold tags)]. Gold labels are complete (an unticked tag is "not
    there"), so a predicted tag not in gold is a false positive. Returns per-tag and overall
    precision / recall / F1."""
    # Matched ignoring case and spaces, so "close-up shot", "Close-Up Shot" and "close-upshot"
    # all count as the same tag.
    norm = lambda t: "".join(str(t).split()).lower()
    vocab_l = {norm(t): t for t in vocab_tags}
    per = {t: [0, 0, 0] for t in vocab_tags}
    for pred, gold in pairs:
        p = {vocab_l[norm(x)] for x in pred if norm(x) in vocab_l}
        g = {vocab_l[norm(x)] for x in gold if norm(x) in vocab_l}
        for t in p & g:
            per[t][0] += 1
        for t in p - g:
            per[t][1] += 1
        for t in g - p:
            per[t][2] += 1
    tot = np.array(list(per.values())).sum(axis=0) if per else np.zeros(3, int)
    return {"n": len(pairs), "overall": _prf(*map(int, tot)),
            "per_tag": {t: _prf(*v) for t, v in per.items()}}


def cut_near(times, t: float, window: float = 0.5) -> bool:
    return any(abs(float(x) - t) <= window for x in times)


def score_boundaries(items: list, predicted_cuts: dict, old_cuts: dict, window: float = 0.5) -> dict:
    """items: 🎯 boundary answers (cut / no_cut; unsure and skipped left out) for videos in
    predicted_cuts. Returns accuracy / precision / recall for the new detector and the old one."""
    rows = [i for i in items if i.get("type") == "boundary" and i.get("verdict") in ("cut", "no_cut")
            and i.get("video_id") in predicted_cuts]

    def score(cuts):
        tp = fp = fn = tn = 0
        for i in rows:
            said = cut_near(cuts.get(i["video_id"], []), float(i["t"]), window)
            real = i["verdict"] == "cut"
            tp += said and real
            fp += said and not real
            fn += real and not said
            tn += (not said) and (not real)
        res = _prf(tp, fp, fn)
        res["accuracy"] = round((tp + tn) / len(rows), 3) if rows else None
        return res

    return {"n": len(rows), "new": score(predicted_cuts), "old": score(old_cuts)}


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def stage_cost(seconds: float, gpu: str, cores: float = 4.0, mem_gib: float = 16.0) -> float:
    return round(seconds * (GPU_RATE.get(gpu, 0.0) + cores * CPU_CORE_RATE + mem_gib * MEM_GIB_RATE), 4)


def extrapolate(stage_seconds: dict, gpus: dict, trial_hours, library_hours: float,
                fixed: dict = None) -> dict:
    """{stage: {"trial_$", "per_footage_hour_$", "library_$", "library_gpu_hours"}} plus a total.
    stage_seconds: work that grows with the footage. fixed: {stage: (seconds, times)} — start-up
    work paid `times` over for the whole library (once per run for a teacher, once per video for
    vision), so a short trial's start-up isn't wrongly spread over every hour of the library.
    trial_hours: one number, or {stage: footage hours that stage actually processed} — a teacher
    that only tagged some of the videos is costed per hour of THOSE videos."""
    fixed = fixed or {}
    out, tot = {}, {"trial_$": 0.0, "library_$": 0.0}
    for stage, sec in stage_seconds.items():
        gpu = gpus.get(stage, "CPU")
        f_sec, f_times = fixed.get(stage, (0.0, 1))
        c = stage_cost(sec, gpu)
        c_fixed = stage_cost(f_sec, gpu)
        hours = trial_hours.get(stage, 0.0) if isinstance(trial_hours, dict) else trial_hours
        per_hour = c / hours if hours else 0.0
        lib = per_hour * library_hours + c_fixed * f_times
        lib_sec = (sec / hours * library_hours if hours else 0.0) + f_sec * f_times
        out[stage] = {"gpu": gpu, "trial_seconds": round(sec + f_sec, 1), "trial_$": round(c + c_fixed, 2),
                      "per_footage_hour_$": round(per_hour, 3), "library_$": round(lib, 2),
                      "library_gpu_hours": round(lib_sec / 3600, 2)}
        tot["trial_$"] += c + c_fixed
        tot["library_$"] += lib
    out["total"] = {k: round(v, 2) for k, v in tot.items()}
    return out
