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
                     "not_to_be_confused_with": raw.get("not_to_be_confused_with", "").strip()})
    return rows


def vocabulary_from_labels(labels) -> list:
    out = []
    for t in labels or []:
        name = t if isinstance(t, str) else (t or {}).get("name") or (t or {}).get("label")
        if name:
            out.append({"tag": str(name), "group": "", "kind": "frame", "one_of": "", "teacher": True,
                        "description": "", "not_to_be_confused_with": ""})
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
# The tagging prompt
# ---------------------------------------------------------------------------

def build_prompt(vocab: list) -> tuple:
    """(text before the images, text after) for one moment shown as 7 frames over 4 seconds."""
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
    before = ("You are labelling footage for a video-editing library. You will see 7 frames "
              "spanning about 4 seconds of one clip, in time order. Frame 4 (the middle one) is "
              "the moment being labelled.")
    after = ["", "Tags judged from frame 4 alone (what is visible in that frame):"]
    after += [line(r) for r in frame] or ["(none)"]
    after += ["", "Tags judged from the movement across all 7 frames:"]
    after += [line(r) for r in motion] or ["(none)"]
    if groups:
        after += ["", "Pick at most one tag from each of these sets:"]
        after += [f"- {name}: {', '.join(tags)}" for name, tags in groups.items()]
    after += ["", "Only use tags from the lists above, spelled exactly as written. Choose every tag "
              "that clearly applies; leave out anything you are not sure of, and list those under "
              "\"unsure\" instead. Reply with JSON only, no other text:",
              '{"tags": ["..."], "unsure": ["..."], "caption": "one short neutral sentence"}']
    return before, "\n".join(after)


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
    unsure = [t for t in clean(data.get("unsure")) if t not in tags]
    return {"tags": tags, "unsure": unsure, "caption": str(data.get("caption") or "")[:300], "ok": ok}


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

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


def extrapolate(stage_seconds: dict, gpus: dict, trial_hours: float, library_hours: float,
                fixed: dict = None) -> dict:
    """{stage: {"trial_$", "per_footage_hour_$", "library_$", "library_gpu_hours"}} plus a total.
    stage_seconds: work that grows with the footage. fixed: {stage: (seconds, times)} — start-up
    work paid `times` over for the whole library (once per run for a teacher, once per video for
    vision), so a short trial's start-up isn't wrongly spread over every hour of the library."""
    fixed = fixed or {}
    out, tot = {}, {"trial_$": 0.0, "library_$": 0.0}
    for stage, sec in stage_seconds.items():
        gpu = gpus.get(stage, "CPU")
        f_sec, f_times = fixed.get(stage, (0.0, 1))
        c = stage_cost(sec, gpu)
        c_fixed = stage_cost(f_sec, gpu)
        per_hour = c / trial_hours if trial_hours else 0.0
        lib = per_hour * library_hours + c_fixed * f_times
        lib_sec = (sec / trial_hours * library_hours if trial_hours else 0.0) + f_sec * f_times
        out[stage] = {"gpu": gpu, "trial_seconds": round(sec + f_sec, 1), "trial_$": round(c + c_fixed, 2),
                      "per_footage_hour_$": round(per_hour, 3), "library_$": round(lib, 2),
                      "library_gpu_hours": round(lib_sec / 3600, 2)}
        tot["trial_$"] += c + c_fixed
        tot["library_$"] += lib
    out["total"] = {k: round(v, 2) for k, v in tot.items()}
    return out
