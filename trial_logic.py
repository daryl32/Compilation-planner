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
NEIGHBOUR_GAP = 12.0       # s: a moment either side further away than this doesn't count as a neighbour
NEIGHBOUR_MIN = 1.0        # s: …nor one closer than this (practically the same frame)


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
                                    if n.strip()],
                     "always_with": [n.strip() for n in raw.get("always_with", "").replace(",", ";").split(";")
                                     if n.strip()],
                     "only_with": [n.strip() for n in (raw.get("only_with") or "").replace(",", ";").split(";")
                                   if n.strip()],
                     "usually_with": [n.strip() for n in (raw.get("usually_with") or "").replace(",", ";").split(";")
                                      if n.strip()],
                     "tell_apart": (raw.get("tell_apart") or "").strip(),
                     "threshold": _int_or_none(raw.get("threshold"), 0, 100),
                     "double_check": (raw.get("double_check") or "").strip().lower() in ("yes", "y", "true", "1", "x"),
                     "group_order": _int_or_none(raw.get("group_order"), 0, 999),
                     "group_required": (raw.get("group_required") or "").strip().lower()
                     in ("yes", "y", "true", "1", "x")})
    return rows


def _int_or_none(v, lo: int, hi: int):
    try:
        return max(lo, min(hi, int(float(str(v).strip()))))
    except (TypeError, ValueError):
        return None


DEFAULT_THRESHOLD = 50          # a tag counts when the teacher is at least this sure (0–100)
MAX_PER_CATEGORY = 8            # most tags the teacher may list in one category (keeps replies short)


def categories(vocab: list) -> list:
    """[(name, [rows], required)] in the order the teacher answers them: categories with a
    group_order first (lowest first), then the rest in the order they first appear. A tag with no
    group is in "other". required = any of its tags says group_required (every moment needs one)."""
    cats = {}
    for i, r in enumerate(vocab):
        name = r.get("group") or "other"
        c = cats.setdefault(name, {"rows": [], "first": i, "order": None, "required": False})
        c["rows"].append(r)
        if r.get("group_order") is not None:
            c["order"] = r["group_order"] if c["order"] is None else min(c["order"], r["group_order"])
        c["required"] = c["required"] or bool(r.get("group_required"))
    ranked = sorted(cats.items(), key=lambda kv: (kv[1]["order"] is None, kv[1]["order"] or 0, kv[1]["first"]))
    return [(name, c["rows"], c["required"]) for name, c in ranked]


def category_key(name: str) -> str:
    """The category's key in the reply (never clashes with "caption")."""
    return f"{name} tags" if name.strip().lower() == "caption" else name


def tag_threshold(r: dict) -> int:
    t = r.get("threshold")
    return DEFAULT_THRESHOLD if t is None else int(t)


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


def always_map(vocab: list) -> dict:
    """{tag (lower case): [tags that must come with it]} from always_with — one way (A → B), only
    tags in this vocabulary."""
    names = {r["tag"].lower(): r["tag"] for r in vocab}
    out = {}
    for r in vocab:
        for other in r.get("always_with") or []:
            o = names.get(str(other).strip().lower())
            if o and o.lower() != r["tag"].lower():
                out.setdefault(r["tag"].lower(), []).append(o)
    return out


def only_map(vocab: list) -> dict:
    """{tag (lower case): [tags, at least one of which must be there for it to be allowed]} from
    only_with — one way (A only with B or C). Only tags in this vocabulary count, so a rule whose
    partners the teacher isn't asked about is simply ignored (it could never be met)."""
    names = {r["tag"].lower(): r["tag"] for r in vocab}
    out = {}
    for r in vocab:
        for other in r.get("only_with") or []:
            o = names.get(str(other).strip().lower())
            if o and o.lower() != r["tag"].lower() and o not in out.get(r["tag"].lower(), []):
                out.setdefault(r["tag"].lower(), []).append(o)
    return out


def usually_map(vocab: list) -> dict:
    """{tag (lower case): [tags that often — not always — come with it]} from usually_with, one
    way, only tags in this vocabulary. Soft: nothing is added or dropped by it; it makes the
    teacher take a second look (look_again_*)."""
    names = {r["tag"].lower(): r["tag"] for r in vocab}
    out = {}
    for r in vocab:
        for other in r.get("usually_with") or []:
            o = names.get(str(other).strip().lower())
            if o and o.lower() != r["tag"].lower() and o not in out.get(r["tag"].lower(), []):
                out.setdefault(r["tag"].lower(), []).append(o)
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
                        "description": "", "not_to_be_confused_with": "", "never_with": [], "always_with": [],
                        "only_with": [], "usually_with": []})
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


def dedup_targets(targets: list, times, embs, threshold: float = DEDUP_DEFAULT, protect=()) -> tuple:
    """Split a video's moments into the ones to send to the teacher and the ones that can copy a
    tagged moment's answer: (keep, copies). copies = {key: {"of": kept key, "similarity": s}}.

    Two moments are "the same" only when they're in the same scene and BOTH their middle frame and
    their ±2 s window of frames match above the threshold — so a still scene where nothing changes
    is tagged once, while a scene whose movement differs from one moment to the next is tagged at
    every moment. Your labelled frames (gold) are always sent, so the report's scores are real.
    protect: keys of more moments to always send — your labels matched by video and time (✏️
    corrections and cards labelled after the vision run carry no gold_id on the target)."""
    protect = set(protect or ())
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
        if f is None or tg.get("gold_id") or tg["key"] in protect or sc is None:
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

# The look: before any tag, the teacher answers these steps in order, each from fixed choices
# (+ "can't tell"), then writes a one-sentence summary from them. (key, question, choices)
CANT = "can't tell"
POSTURES = ["lying on back", "lying on front", "lying on side", "on all fours", "kneeling upright", "sitting",
            "squatting / straddling", "standing", "bent over (standing)", "out of frame"]
LOOK_STEPS = [
    ("women", "How many women are in the scene?", ["0", "1", "2", "3+"]),
    ("men", "How many men are in the scene?", ["0", "1", "2", "3+"]),
    ("penetration", "Penetration — 'hidden but implied' only when their hips are joined AND there is a steady "
                    "thrusting or riding movement across the 7 frames",
     ["clearly visible", "hidden but implied", "not happening"]),
    ("penetration_with", "Penetration with", ["penis", "toy", "fingers", "not applicable"]),
    ("penetration_where", "Penetration where", ["vagina", "anus", "not applicable"]),
    ("hidden", "What is hidden from view?",
     ["genitals hidden", "partner mostly hidden behind her", "nothing important hidden"]),
    ("her_posture", "Her posture", POSTURES),
    ("her_front_faces", "Her front faces (relative to the camera)", ["the camera", "away from the camera", "sideways"]),
    ("partner_posture", "Her partner's posture", POSTURES),
    ("partner_legs_point", "Are her partner's feet nearer the camera than his head? Feet nearer → towards the "
                           "camera; head nearer → away from the camera; feet and head about as near → sideways. "
                           "When the camera is behind her and his legs are under her, his feet are usually "
                           "nearer → towards the camera",
     ["towards the camera", "away from the camera", "sideways", "out of frame"]),
    ("on_top", "Who is on top? Her back against the bed → partner on top; her hair or breasts hang down towards "
               "her partner → she is on top; in POV, the camera looks up at her → she is on top, looks down at "
               "her → partner on top", ["her", "partner", "side by side", "neither (standing)"]),
    ("facing", "Which way does she face her partner? Her face and chest turned to his face → towards; her back "
               "or buttocks towards his face, his legs stretched out in front of her → away; with an onlooker "
               "camera use the steps above (his legs towards the camera + her front away from the camera → "
               "towards); in POV, her face to the camera → towards", ["towards", "away"]),
    # Hands, mouths and feet — for the action tags (Handjob, Blowjob, deepthroat, fingering, masturbation,
    # Cunnilingus, Footjob). If a hand does two things, answer what it does to genitals.
    ("her_hands", "What are her hands doing? 'stroking' = her hand moves along it across the 7 frames; "
                  "'holding' = her hand is on it but not moving along it",
     ["stroking his penis", "holding his penis", "on her own genitals", "on her partner's body",
      "on her own body", "supporting herself", "not visible"]),
    ("her_mouth", "Where is her mouth?",
     ["on his penis", "kissing", "on her partner's body", "open / other", "not visible"]),
    ("mouth_depth", "If her mouth is on his penis, how far in? 'near the base' only when you can see her lips "
                    "at or close to his body, with most of it in her mouth; her hand wrapped round most of the "
                    "shaft means it isn't near the base; hair, angle or a hand hiding her lips → can't tell",
     ["tip only", "partway", "near the base", "not applicable"]),
    ("her_feet", "Her feet", ["on his penis", "other", "not visible"]),
    ("partner_hands", "What are her partner's hands doing? 'fingers inside her' only when you can see them go in",
     ["fingers inside her", "rubbing her genitals", "holding a toy on or in her", "on her body", "other",
      "not visible"]),
    ("partner_mouth", "Where is her partner's mouth?",
     ["on her genitals", "kissing", "on her body", "other", "not visible"]),
    ("viewpoint", "Viewpoint", ["a participant's eyes (their own body at the frame edge)", "onlooker"]),
]
LOOK_POSITION_NEEDS = {   # position tag (lower case) → what the look must say for it to stand
    "cowgirl position": lambda L: L.get("on_top") == "her" and L.get("facing") == "towards",
    "reverse cowgirl": lambda L: L.get("on_top") == "her" and L.get("facing") == "away",
    "missionary": lambda L: L.get("her_posture") == "lying on back" and L.get("on_top") == "partner",
    "doggy style": lambda L: L.get("her_posture") in ("on all fours", "bent over (standing)"),
    "laying down": lambda L: (L.get("her_posture", "").startswith("lying")
                              and L.get("partner_posture", "").startswith("lying")),
    "standing": lambda L: L.get("her_posture") == "standing" and L.get("partner_posture") == "standing",
}


def look_instructions() -> str:
    lines = ["Before any tag, look carefully at the people and fill in \"look\" — every step, in this order, "
             "choosing one of the listed answers (or \"can't tell\" when you can't see it). Then write "
             "\"summary\": one or two plain sentences built only from your answers. Use your answers when "
             "you choose the tags."]
    lines += [f"- {key}: {q} — {' | '.join(ch)} | {CANT}" for key, q, ch in LOOK_STEPS]
    return "\n".join(lines)


def look_shape() -> str:
    return '"look": {' + ", ".join(f'"{k}": "..."' for k, _, _ in LOOK_STEPS) + ', "summary": "..."}'


def look_schema() -> dict:
    props = {k: {"type": "string", "enum": ch + [CANT]} for k, _, ch in LOOK_STEPS}
    props["summary"] = {"type": "string", "maxLength": 240}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def read_look(data) -> dict | None:
    """The look from a reply: known steps with an allowed answer, plus the summary; None if absent."""
    if not isinstance(data, dict):
        return None
    out = {}
    for k, _, ch in LOOK_STEPS:
        v = str(data.get(k) or "").strip()
        if v in ch or v == CANT:
            out[k] = v
    if data.get("summary"):
        out["summary"] = str(data["summary"])[:240]
    return out or None


def look_check(r: dict, vocab: list) -> list:
    """The free CPU check: drop to unsure what the moment's own look contradicts. A position needs
    penetration clearly visible or hidden but implied, both postures seen (not out of frame / can't
    tell) and on top / facing / postures that fit it (LOOK_POSITION_NEEDS). Penetration (and Anal
    Sex) needs "clearly visible" — your labels give it only when it's visible. Changes `r` in place
    ("look_dropped"); returns the tags dropped. A moment without a look is left alone."""
    L = r.get("look") or {}
    if not L:
        return []
    names = {x["tag"].lower(): x["tag"] for x in vocab}
    seen_pen = L.get("penetration") in ("clearly visible", "hidden but implied")
    bodies = all(L.get(k) not in (None, CANT, "out of frame") for k in ("her_posture", "partner_posture"))
    drop = []
    for t in r.get("tags") or []:
        low = t.lower()
        if low in LOOK_POSITION_NEEDS:
            if not (seen_pen and bodies and LOOK_POSITION_NEEDS[low](L)):
                drop.append(t)
        elif low in ("penetration", "anal sex") and L.get("penetration") != "clearly visible":
            drop.append(t)
    if drop:
        r["tags"] = [t for t in r["tags"] if t not in drop]
        r["unsure"] = list(r.get("unsure") or []) + [t for t in drop if t not in (r.get("unsure") or [])]
        r["look_dropped"] = list(r.get("look_dropped") or []) + drop
    return [names.get(t.lower(), t) for t in drop]


def build_prompt(vocab: list, look: bool = False) -> tuple:
    """(head, moment, tail) for one moment shown as 7 frames over 4 seconds.
    look: before tagging, the teacher answers the look steps (LOOK_STEPS — people, penetration,
    bodies, orientation) and a one-sentence summary, which replaces the caption.

    head — everything that is the same for every moment (the task, the tag lists, the rules,
           the reply format). It comes FIRST, before any picture, so vLLM's prefix cache keeps
           it and the GPU reads it once per run instead of once per moment.
    moment — the line just before the moment's 7 frames.
    tail — the short line after them."""
    def line(r):
        bits = [f"- {r['tag']}" + (" [movement]" if r["kind"] == "motion" else "")]
        if r["description"]:
            bits.append(f": {r['description']}")
        notes = []
        if r["not_to_be_confused_with"]:
            notes.append(f"not the same as: {r['not_to_be_confused_with']}")
        if r.get("tell_apart"):
            notes.append(f"how to tell: {r['tell_apart']}")
        if notes:
            bits.append(f" ({'; '.join(notes)})")
        return "".join(bits)

    cats = categories(vocab)

    groups = {}
    for r in vocab:
        if r["one_of"]:
            groups.setdefault(r["one_of"], []).append(r["tag"])
    head = ["You are labelling footage for a video-editing library. Each moment to label is shown "
            "as 7 frames spanning about 4 seconds of one clip, in time order. Frame 4 (the middle "
            "one) is the moment being labelled.",
            "", "The tags, category by category. Judge each tag from frame 4 alone (what is visible in "
                "that frame), except tags marked [movement]: judge those from the movement across all 7 "
                "frames."]
    for i, (name, rows, required) in enumerate(cats, 1):
        head += ["", f"Category {i} — {name}" + (" (every moment needs at least one)" if required else "") + ":"]
        head += [line(r) for r in rows]
    if groups:
        head += ["", "Only one tag from each of these sets can be true — give your best one the highest number "
                     "(you may list another with a lower number if you're torn):"]
        head += [f"- {name}: {', '.join(tags)}" for name, tags in groups.items()]
    pairs = never_pairs(vocab)
    if pairs:
        order = {r["tag"]: i for i, r in enumerate(vocab)}
        rules = sorted({tuple(sorted((r["tag"], o), key=order.get)) for r in vocab
                        for o in pairs.get(r["tag"].lower(), ())}, key=lambda p: (order[p[0]], order[p[1]]))
        head += ["", "These tags never apply to the same moment — if both seem to, give only the one that "
                     "fits best:"]
        head += [f"- {a} / {b}" for a, b in rules]
    amap = always_map(vocab)
    if amap:
        head += ["", "If you give the first tag, always give the second as well:"]
        head += [f"- {r['tag']} → {n}" for r in vocab for n in amap.get(r["tag"].lower(), [])]
    omap = only_map(vocab)
    if omap:
        head += ["", "Only give the first tag if you also give at least one of the tags after it — "
                     "otherwise leave it out:"]
        head += [f"- {r['tag']}: only with {' or '.join(omap[r['tag'].lower()])}" for r in vocab
                 if r["tag"].lower() in omap]
    umap = usually_map(vocab)
    if umap:
        head += ["", "These often (not always) go together — if you give the first tag, look carefully "
                     "at whether the others apply too, and only give them if they do:"]
        head += [f"- {r['tag']}: often with {', '.join(umap[r['tag'].lower()])}" for r in vocab
                 if r["tag"].lower() in umap]
    need = [name for name, _, required in cats if required]
    shape = ", ".join(f'"{category_key(name)}": [{{"tag": "...", "conf": 0-100}}]' for name, _, _ in cats)
    if look:
        head += ["", look_instructions()]
    head += ["", ("First fill in \"look\", then work" if look else "Work") + " through the categories in order. In each one, list the tags that might apply, each "
                 "with how sure you are that it is clearly there, from 0 to 100. Give a tag you are not "
                 "sure about a low number rather than leaving it out; leave out tags that clearly don't "
                 "apply; use [] when nothing in a category applies."
             + (f" {', '.join(need)} must always have at least one tag — give your best guess." if need else ""),
             "Only use tags from that category's list, spelled exactly as written. Reply with JSON only, "
             "no other text:",
             ("{" + look_shape() + ", " + shape + "}" if look
              else "{" + shape + ', "caption": "one short neutral sentence"}')]
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


def _json_in(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _conf(v) -> int:
    try:
        return max(0, min(100, int(round(float(v)))))
    except (TypeError, ValueError):
        return 100


def parse_answer(text: str, vocab: list) -> dict:
    """{"tags", "unsure", "caption", "ok", "conf", "rule_*"} from the teacher's reply.

    The reply lists tags per category with a confidence ({"shot": [{"tag", "conf"}], …}); the
    older shape {"tags": [...], "unsure": [...]} is read too (tags = 100, unsure = 0). Then
    finalize() decides: per-tag thresholds, pick-one sets and never-with by confidence, required
    categories, always-with and only-with."""
    lookup = {r["tag"].lower(): r for r in vocab}
    data = _json_in(text)
    ok = data is not None
    data = data or {}
    conf = {}

    def put(name, c):
        r = lookup.get(str(name).strip().lower())
        if r:
            conf[r["tag"]] = max(conf.get(r["tag"], -1), c)

    for k, val in data.items():
        if k in ("caption", "tags", "unsure", "look") or not isinstance(val, list):
            continue
        for it in val:
            if isinstance(it, dict):
                put(it.get("tag", ""), _conf(it.get("conf", 100)))
            elif isinstance(it, str):
                put(it, 100)
    for n in data.get("tags") if isinstance(data.get("tags"), list) else []:
        put(n, 100)
    for n in data.get("unsure") if isinstance(data.get("unsure"), list) else []:
        put(n, 0)
    out = finalize(conf, vocab)
    seen = read_look(data.get("look"))
    caption = str(data.get("caption") or (seen or {}).get("summary") or "")[:300]
    return {"tags": out["tags"], "unsure": out["unsure"], "caption": caption, "ok": ok,
            **({"look": seen} if seen else {}),
            **{k: v for k, v in out.items() if k not in ("tags", "unsure")}}


def finalize(conf: dict, vocab: list, rejected=()) -> dict:
    """The final tags from {tag: confidence} (in the order named):
    1. a tag counts when its confidence reaches its threshold (default 50) and it wasn't rejected
       (a "no" on a second look);
    2. pick-one sets: the most confident wins (ties: the first named) — the others are dropped;
    3. never-with: the more confident stays, the other goes to unsure (rule_breaks);
    4. a required category with nothing: its most confident listed tag is taken (rule_required);
    5. always-with adds (rule_adds), only-with drops (rule_drops), as before.
    Unsure = listed below threshold, rejected, broke a rule — at most one per pick-one set."""
    lookup = {r["tag"].lower(): r for r in vocab}
    rejected = {t.lower() for t in rejected}
    order = {t: i for i, t in enumerate(conf)}
    ranked = sorted(conf, key=lambda t: (-conf[t], order[t]))
    pairs, sets = never_pairs(vocab), {r["tag"]: r["one_of"] for r in vocab}
    kept, broke, used = [], [], set()

    def fits(t):
        if sets.get(t) and sets[t] in used:
            return False
        return not any(o in kept for o in pairs.get(t.lower(), ()))

    for t in ranked:
        r = lookup[t.lower()]
        if t.lower() in rejected or conf[t] < tag_threshold(r):
            continue
        if sets.get(t) and sets[t] in used:
            continue
        if any(o in kept for o in pairs.get(t.lower(), ())):
            broke.append(t)
            continue
        kept.append(t)
        if sets.get(t):
            used.add(sets[t])
    required_adds = []
    for name, rows, required in categories(vocab):
        names = {r["tag"] for r in rows}
        if not required or any(t in names for t in kept):
            continue
        pool = [t for t in ranked if t in names and fits(t)]
        pick = next((t for t in pool if t.lower() not in rejected), None) or next(iter(pool), None)
        if pick:
            kept.append(pick)
            required_adds.append(pick)
            if sets.get(pick):
                used.add(sets[pick])
    tags = sorted(kept, key=lambda t: order[t])
    broke = [t for t in broke if t not in tags]
    # always-with: add a missing partner — unless it would clash with a pick-one set or a never-with
    amap = always_map(vocab)
    added, grew = [], True
    while grew:
        grew = False
        for t in list(tags):
            for n in amap.get(t.lower(), []):
                if n in tags or n.lower() in rejected or any(o in tags for o in pairs.get(n.lower(), ())):
                    continue
                if sets.get(n) and any(sets.get(x) == sets.get(n) for x in tags):
                    continue
                tags.append(n)
                added.append(n)
                grew = True
    # only-with: a tag given without any of the tags it needs is dropped to unsure (the needed tag
    # is the evidence, so it isn't added). Repeats, as dropping one can strand another.
    omap, dropped, shrank = only_map(vocab), [], True
    while shrank:
        shrank = False
        for t in list(tags):
            needs = omap.get(t.lower())
            if needs and not any(n in tags for n in needs):
                tags.remove(t)
                dropped.append(t)
                shrank = True
    added = [t for t in added if t in tags]
    required_adds = [t for t in required_adds if t in tags]
    low = [t for t in conf if t not in tags and (conf[t] < tag_threshold(lookup[t.lower()]) or t.lower() in rejected)]
    unsure, uset = [], set()
    for t in sorted(dict.fromkeys(low + broke + dropped), key=lambda t: (-conf.get(t, 0), order.get(t, 1e9))):
        if t in tags or (sets.get(t) and sets[t] in uset):
            continue
        unsure.append(t)
        if sets.get(t):
            uset.add(sets[t])
    unsure.sort(key=lambda t: order.get(t, 1e9))
    return {"tags": tags, "unsure": unsure, "conf": dict(conf), "rule_breaks": broke, "rule_adds": added,
            "rule_drops": dropped, "rule_required": required_adds}


def answer_schema(vocab: list, look: bool = False) -> dict:
    """JSON schema the teacher's reply is held to (vLLM structured output): one list per category,
    in your category order, each item a tag from THAT category with a 0–100 confidence; required
    categories need at least one item; then the caption. Typos and invented tags can't happen.
    look: the look object comes FIRST (each step one of its fixed choices, then a summary) and
    replaces the caption."""
    props, req = {}, []
    if look:
        props["look"] = look_schema()
        req.append("look")
    for name, rows, required in categories(vocab):
        key = category_key(name)
        props[key] = {"type": "array", "minItems": 1 if required else 0,
                      "maxItems": max(1, min(len(rows), MAX_PER_CATEGORY)),
                      "items": {"type": "object", "additionalProperties": False, "required": ["tag", "conf"],
                                "properties": {"tag": {"type": "string", "enum": [r["tag"] for r in rows]},
                                               "conf": {"type": "integer", "minimum": 0, "maximum": 100}}}}
        req.append(key)
    if look:
        return {"type": "object", "additionalProperties": False, "required": req, "properties": props}
    props["caption"] = {"type": "string", "maxLength": 200}
    return {"type": "object", "additionalProperties": False, "required": req + ["caption"], "properties": props}


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


# ---------------------------------------------------------------------------
# Neighbours: a tag the view hid for a moment, filled in from the moments either side
# ---------------------------------------------------------------------------

def neighbour_fill(results: dict, vocab: list, gap: float = NEIGHBOUR_GAP, closest: float = NEIGHBOUR_MIN) -> int:
    """After a whole video is tagged: a tag with an only-with rule (the positions — only with
    penetration / Anal Sex) is added to a moment that hasn't got it when the nearest moment before
    AND the nearest after, in the same scene and within `gap` seconds, both have it together with
    one of its only-with partners — e.g. the camera lost sight of the penetration for a second
    while the position carried on. The partner itself (penetration) isn't added: your labels give
    it only when it's visible. Never added when the moment has a never-with tag of it (another
    position, Blowjob…) or its answer was unreadable. Decided from the answers as the teacher gave
    them, so one fill never leads to another. Changes `results` in place: the tag joins "tags",
    leaves "unsure" and is listed in "neighbour_added". Returns how many tags were added."""
    omap, pairs = only_map(vocab), never_pairs(vocab)
    names = {r["tag"].lower(): r["tag"] for r in vocab}
    fillable = [names[t] for t in omap if t in names]
    if not fillable:
        return 0
    before = {k: list(r.get("tags") or []) for k, r in results.items()}
    scenes = {}
    for k, r in results.items():
        if r.get("t") is None or r.get("scene_id") is None:
            continue
        scenes.setdefault(r["scene_id"], []).append((float(r["t"]), k))

    def has(k, tag):
        return tag.lower() in {x.lower() for x in before[k]}

    def holds(k, tag):              # the tag and one of its only-with partners
        return results[k].get("ok", True) and has(k, tag) and any(has(k, n) for n in omap[tag.lower()])

    added = 0
    for ms in scenes.values():
        ms.sort()
        for t, k in ms:
            r = results[k]
            if not r.get("ok", True):
                continue
            prev = next((m for m in reversed(ms) if t - gap <= m[0] <= t - closest), None)
            nxt = next((m for m in ms if t + closest <= m[0] <= t + gap), None)
            if prev is None or nxt is None:
                continue
            for tag in fillable:
                if has(k, tag) or not (holds(prev[1], tag) and holds(nxt[1], tag)):
                    continue
                now = {x.lower() for x in r.get("tags") or []}
                if any(o.lower() in now for o in pairs.get(tag.lower(), ())):
                    continue
                r["tags"] = list(r.get("tags") or []) + [tag]
                r["unsure"] = [u for u in r.get("unsure") or [] if u.lower() != tag.lower()]
                r["neighbour_added"] = list(r.get("neighbour_added") or []) + [tag]
                added += 1
    return added


# ---------------------------------------------------------------------------
# Second look (usually_with)
# ---------------------------------------------------------------------------

def look_again(ans: dict, vocab: list) -> list:
    """What to ask the teacher on a second look, as [{"tag", "kind", ...}]:
    kind "usually" — {"because", "instead_of"}: a usually-with partner of a tag it gave that it
        didn't give ("instead_of" = the tag it chose from the same pick-one set → "is this a better
        fit?"). Never one that would break a never-with or its own only-with rule.
    kind "check" — a tag it gave whose double_check switch is on: "is it clearly there?"."""
    tags = list(ans.get("tags") or [])
    have = {t.lower() for t in tags}
    by_name = {r["tag"].lower(): r for r in vocab}
    umap, pairs, omap = usually_map(vocab), never_pairs(vocab), only_map(vocab)
    sets = {r["tag"].lower(): r.get("one_of") for r in vocab}
    out, seen = [], set()
    for t in tags:
        for n in umap.get(t.lower(), []):
            if n.lower() in have or n.lower() in seen:
                continue
            if any(o.lower() in have for o in pairs.get(n.lower(), ())):
                continue
            needs = omap.get(n.lower())
            if needs and not any(x.lower() in have for x in needs):
                continue
            same = next((x for x in tags if sets.get(n.lower()) and sets.get(x.lower()) == sets.get(n.lower())),
                        None)
            out.append({"tag": n, "kind": "usually", "because": t, "instead_of": same})
            seen.add(n.lower())
    out += [{"tag": t, "kind": "check"} for t in tags if (by_name.get(t.lower()) or {}).get("double_check")]
    return out


def look_again_prompt(asks: list) -> str:
    usual = [a for a in asks if a["kind"] == "usually"]
    check = [a for a in asks if a["kind"] == "check"]
    lines = ["Take a second, careful look at frame 4 (and at the movement across the frames for tags "
             "marked [movement])."]
    if usual:
        lines.append("These often go with tags you gave — check each one properly, they don't always apply:")
        for a in usual:
            if a["instead_of"]:
                lines.append(f"- {a['tag']} (often with {a['because']}): is it a better fit than "
                             f"{a['instead_of']}, which you gave?")
            else:
                lines.append(f"- {a['tag']} (often with {a['because']}): does it apply?")
    if check:
        lines.append("Check these tags you gave again — they are often given when they aren't really there:")
        lines += [f"- {a['tag']}: is it clearly there?" for a in check]
    lines += ["Answer true only if you can clearly see it. Reply with JSON only, no other text — one "
              "answer for every tag above:",
              '{"answers": {' + ", ".join(f'"{a["tag"]}": true or false' for a in asks) + "}}"]
    return "\n".join(lines)


def look_again_schema(asks: list) -> dict:
    names = list(dict.fromkeys(a["tag"] for a in asks))
    return {"type": "object", "additionalProperties": False, "required": ["answers"],
            "properties": {"answers": {"type": "object", "additionalProperties": False, "required": names,
                                       "properties": {n: {"type": "boolean"} for n in names}}}}


def apply_look_again(first: dict, text: str, asks: list, vocab: list) -> dict:
    """The first answer updated by the second look: a "usually" yes is added (a pick-one yes
    replaces the tag it was asked against); a "check" no takes the tag out (to unsure — unless it's
    the only possible tag of a required category). Then finalize() runs again on the confidences.
    Adds "look_asked", "look_added", "look_removed". An unreadable reply changes nothing."""
    data = _json_in(text) or {}
    said = {}
    if isinstance(data.get("answers"), dict):
        said = {str(k).strip().lower(): v for k, v in data["answers"].items() if isinstance(v, bool)}
    else:                                   # the older {"yes": [...], "no": [...]} shape
        for k, v in (("yes", True), ("no", False)):
            if isinstance(data.get(k), list):
                said.update({str(x).strip().lower(): v for x in data[k]})
    out = {**first, "look_asked": [a["tag"] for a in asks], "look_added": [], "look_removed": []}
    conf, rejected = dict(first.get("conf") or {}), set()
    for t in first.get("tags") or []:       # tags added by rules have no confidence of their own
        conf.setdefault(t, 100)
    yes = [a for a in asks if a["kind"] == "usually" and said.get(a["tag"].lower()) is True]
    no = [a for a in asks if a["kind"] == "check" and said.get(a["tag"].lower()) is False]
    if not yes and not no:
        return out
    for a in yes:
        conf[a["tag"]] = 100
        if a.get("instead_of"):
            rejected.add(a["instead_of"])
    rejected |= {a["tag"] for a in no}
    again = finalize(conf, vocab, rejected)
    out.update({"tags": again["tags"], "unsure": again["unsure"], "conf": again["conf"],
                "look_added": [a["tag"] for a in yes if a["tag"] in again["tags"]],
                "look_removed": [a["tag"] for a in no if a["tag"] not in again["tags"]]})
    for k in ("rule_breaks", "rule_adds", "rule_drops", "rule_required"):
        out[k] = list(dict.fromkeys((first.get(k) or []) + (again.get(k) or [])))
    return out


# ---------------------------------------------------------------------------
# 🧪 Test runs (the same videos re-tagged after each settings change)
# ---------------------------------------------------------------------------

def run_pairs(frames: list, gold: list) -> list:
    """[(run tags, your tags)] for your labelled frames that the run tagged — 🎯 cards by their id,
    ✏️ corrections by video and time. frames: [{"v", "t", "g", "tags"}]."""
    by_id = {f["g"]: f for f in frames if f.get("g")}
    by_t = {(f["v"], round(float(f["t"]), 2)): f for f in frames if f.get("t") is not None}
    pairs = []
    for i in gold:
        f = by_id.get(i.get("id")) or (by_t.get((i.get("video_id"), round(float(i["t"]), 2)))
                                       if i.get("t") is not None else None)
        if f is not None:
            pairs.append((f.get("tags") or [], i.get("tags") or []))
    return pairs


def copied_scored(frames: list, gold: list) -> int:
    """How many of your labelled frames a run answered by copying a near-identical moment (dedup)
    instead of tagging them (runs from before 3 Nov didn't protect ✏️ corrections from copying)."""
    by_id = {f["g"]: f for f in frames if f.get("g")}
    by_t = {(f["v"], round(float(f["t"]), 2)): f for f in frames if f.get("t") is not None}
    n = 0
    for i in gold:
        f = by_id.get(i.get("id")) or (by_t.get((i.get("video_id"), round(float(i["t"]), 2)))
                                       if i.get("t") is not None else None)
        n += bool(f is not None and f.get("c"))
    return n


def test_runs_markdown(runs: list, gold: list, tags: list, top: int = 15) -> str:
    """Every test run scored on your labels as they are now (oldest first), then the per-tag
    changes between the last two runs."""
    if not runs:
        return "No test runs yet."

    def f(x):
        return "–" if x is None else f"{x:.2f}"
    scored = [(r, score_tags(run_pairs(r.get("frames") or [], gold), tags)) for r in runs]
    md = ["🧪 Test set — the same videos every run, scored on your labelled frames as they are now:", "",
          "| Run | Your frames | F1 | Right (precision) | Found (recall) | Wrong | Missed | 2nd looks (+ / −) | "
          "Needs-a-tag fills | Tagging $ |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r, sc in scored:
        o = sc["overall"]
        md.append(f"| {r['id']} | {sc['n']} | {f(o['f1'])} | {f(o['precision'])} | {f(o['recall'])} | {o['fp']} | "
                  f"{o['fn']} | {r.get('looks', 0)} (+{r.get('look_added', 0)} / −{r.get('look_removed', 0)}) | "
                  f"{r.get('required_fills', 0)} | ${r.get('cost', 0):.2f} |")
    if len(scored) >= 2:
        (ra, a), (rb, b) = scored[-2], scored[-1]
        rows = []
        for t in tags:
            pa, pb = a["per_tag"][t], b["per_tag"][t]
            if (pa["fp"], pa["fn"]) != (pb["fp"], pb["fn"]):
                rows.append((t, pa, pb, (pb["fp"] + pb["fn"]) - (pa["fp"] + pa["fn"])))
        rows.sort(key=lambda x: (-abs(x[3]), x[0].lower()))
        md += ["", f"Per tag, {ra['id']} → {rb['id']} (biggest changes first; fewer errors = better):", "",
               "| Tag | Wrong | Missed | Errors |", "|---|---|---|---|"]
        md += [f"| {t} | {pa['fp']} → {pb['fp']} | {pa['fn']} → {pb['fn']} | {'+' if d > 0 else ''}{d} |"
               for t, pa, pb, d in rows[:top]] or ["| (no change) | | | |"]
    n = scored[-1][1]["n"]
    md += ["", f"On {n} frames, F1 moves by about ±{max(0.02, round(0.5 / max(n, 1) ** 0.5, 2)):.2f} from chance "
               "alone — trust bigger changes, and per-tag changes of several frames."]
    return "\n".join(md)


# ---------------------------------------------------------------------------
# 🧠 LoRA training answers
# ---------------------------------------------------------------------------

LORA_CONF = 90          # your tags are taught as "sure" (the teacher's answers use 0–100)


def lora_target(tags: list, vocab: list, caption: str = "") -> str:
    """The answer the teacher should have given for one of your labelled frames, in exactly the
    shape its structured replies have (category order, compact JSON): your tags at LORA_CONF, each
    under its category; tags it isn't asked about are left out."""
    mine = {str(t).strip().lower() for t in tags or []}
    out = {}
    for name, rows, _ in categories(vocab):
        out[category_key(name)] = [{"tag": r["tag"], "conf": LORA_CONF} for r in rows if r["tag"].lower() in mine]
    out["caption"] = str(caption or "")[:200]
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False)


def answer_start(ids: list, marker: list) -> int:
    """Index of the first answer token: just after the LAST occurrence of `marker` (the assistant
    turn's opening tokens) in a row of token ids. len(ids) if it isn't there (nothing to learn)."""
    m = len(marker)
    for i in range(len(ids) - m, -1, -1):
        if ids[i:i + m] == marker:
            return i + m
    return len(ids)
