"""
🧪 Trial results — reading the Modal trial's results bundle for the web app (no Streamlit here).

modal_trial.py's export stage writes CATALOGUE_DIR/trial_results/ (synced from Drive):
    results.json   {"created", "smoke", "teachers": {key: model}, "report": {...},
                    "videos": {video_id: {"duration", "cuts_new", "cuts_old", "boundaries",
                                          "scenes", "curve_fps", "person_curve", "old_curve",
                                          "moments": [{"key", "t", "scene_id", "frame",
                                                       "mine" (your tags or None), "old",
                                                       "teachers": {key: {"tags", "unsure",
                                                                          "caption"}}}]}}}
    report.md      the trial report
    frames/<video_id>/<key>.jpg
"""

from __future__ import annotations

import json
from pathlib import Path

from config import CATALOGUE_DIR

RESULTS_DIR = CATALOGUE_DIR / "trial_results"
_CACHE = {}

RIGHT, WRONG, MISSED, NEUTRAL = "right", "wrong", "missed", "neutral"


def load() -> dict | None:
    """The latest results (re-read when the file changes), or None if there are none."""
    path = RESULTS_DIR / "results.json"
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    if _CACHE.get("stamp") != stamp:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        _CACHE.update(stamp=stamp, data=data if isinstance(data, dict) else None)
    return _CACHE.get("data")


def report_markdown() -> str:
    try:
        return (RESULTS_DIR / "report.md").read_text(encoding="utf-8")
    except OSError:
        return ""


def moment_id(video_id: str, moment: dict) -> str:
    """The 🎯 Training Labels id for this frame — the same id a tag card at that moment gets, so
    a correction here and a card there are one label."""
    import training_labels as TLB
    return TLB.card_id("tags", video_id, float(moment["t"]))


def with_saved_labels(videos: dict, items: list) -> dict:
    """The results' moments with "mine" taken from your current training labels where you've
    labelled (or corrected) that frame since the trial ran. Returns a new videos dict."""
    saved = {i["id"]: i for i in items if i.get("type") == "tags" and not i.get("skipped")}
    out = {}
    for vid, v in videos.items():
        ms = []
        for m in v.get("moments") or []:
            mid = m.get("gold_id") or moment_id(vid, m)
            if mid in saved:
                m = {**m, "mine": list(saved[mid].get("tags") or []), "gold_id": mid}
            ms.append(m)
        out[vid] = {**v, "moments": ms}
    return out


def correction_card(video_id: str, moment: dict, prefill: list) -> dict:
    """A 🎯 tag card for this frame, for training_labels.save_label."""
    return {"id": moment.get("gold_id") or moment_id(video_id, moment), "type": "tags",
            "video_id": video_id, "scene_id": moment.get("scene_id"), "t": float(moment["t"]),
            "suggested": list(prefill)}


def frame_path(moment: dict) -> Path | None:
    return RESULTS_DIR / moment["frame"] if moment.get("frame") else None


def norm(tag) -> str:
    """Tags compare ignoring case and spaces."""
    return "".join(str(tag).split()).lower()


def chips(pred, mine) -> list:
    """[(tag, status)] for one row of tags. Against your tags (mine): right / wrong, then the
    ones you have that this row missed. Without your tags every chip is neutral."""
    pred = list(dict.fromkeys(pred or []))
    if mine is None:
        return [(t, NEUTRAL) for t in pred]
    mine_n = {norm(t) for t in mine}
    pred_n = {norm(t) for t in pred}
    out = [(t, RIGHT if norm(t) in mine_n else WRONG) for t in pred]
    out += [(t, MISSED) for t in mine if norm(t) not in pred_n]
    return out


def tag_sets(moment: dict) -> dict:
    """{"mine"?, "old", <teacher>...: set of normalised tags} for one moment."""
    out = {"old": {norm(t) for t in moment.get("old") or []}}
    if moment.get("mine") is not None:
        out["mine"] = {norm(t) for t in moment["mine"]}
    for k, r in (moment.get("teachers") or {}).items():
        out[k] = {norm(t) for t in r.get("tags") or []}
    return out


def teachers_disagree(moment: dict) -> bool:
    sets = [s for k, s in tag_sets(moment).items() if k not in ("mine", "old")]
    return len(sets) > 1 and any(s != sets[0] for s in sets[1:])


def any_wrong(moment: dict) -> bool:
    """A teacher's tags differ from yours (only for moments you labelled)."""
    sets = tag_sets(moment)
    return "mine" in sets and any(s != sets["mine"] for k, s in sets.items() if k not in ("mine", "old"))


FILTERS = {
    "all": "All moments",
    "todo": "Not reviewed by me yet",
    "mine": "Only frames I labelled",
    "wrong": "A model differs from me",
    "disagree": "The models disagree",
    "rules": "⚠️ My tags break a rule",
}
# The teacher picked after batch 1 — the view shows only it when its tags are there.
MAIN_TEACHER = "32b"


def shown_teachers(teachers) -> list:
    """The teachers the view shows: the 32B (and its FP8 version where a video has both, for the
    side-by-side check); the 8B only when the results have nothing else."""
    teachers = list(teachers or [])
    return [k for k in teachers if k != "8b"] or teachers


def ordered(items: list, how: str = "video", seed: int = 0) -> list:
    """items in video/time order, or shuffled ("random") — the same shuffle for the same seed,
    so paging through a shuffled list doesn't reshuffle it."""
    import random
    items = list(items)
    if how == "random":
        random.Random(seed).shuffle(items)
    return items


def filter_moments(moments: list, how: str = "all", tag: str | None = None, problems=None) -> list:
    """problems: tags -> list of rule problems (training_labels.rule_problems with your vocabulary),
    for the "rules" filter — your labelled frames whose tags break a rule."""
    out = []
    for m in moments:
        if how == "rules" and (m.get("mine") is None or not problems or not problems(m["mine"])):
            continue
        if how == "mine" and m.get("mine") is None:
            continue
        if how == "todo" and m.get("mine") is not None:
            continue
        if how == "wrong" and not any_wrong(m):
            continue
        if how == "disagree" and not teachers_disagree(m):
            continue
        if tag and not any(norm(tag) in s for s in tag_sets(m).values()):
            continue
        out.append(m)
    return out


def agreement(moments: list) -> dict:
    """Per teacher: how many of your labelled frames it got exactly right, and its share of
    your tags found (recall) and of its tags that were right (precision)."""
    out = {}
    for m in moments:
        if m.get("mine") is None:
            continue
        mine = {norm(t) for t in m["mine"]}
        for k, r in (m.get("teachers") or {}).items():
            pred = {norm(t) for t in r.get("tags") or []}
            a = out.setdefault(k, {"frames": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0})
            a["frames"] += 1
            a["exact"] += pred == mine
            a["tp"] += len(pred & mine)
            a["fp"] += len(pred - mine)
            a["fn"] += len(mine - pred)
    for a in out.values():
        a["precision"] = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else None
        a["recall"] = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else None
    return out


def normalised(curve) -> list:
    """Scaled to 0–1 by its own 95th percentile, so two curves on different scales can share a
    chart (None stays None)."""
    vals = sorted(v for v in curve or [] if v is not None)
    if not vals:
        return list(curve or [])
    top = vals[min(len(vals) - 1, int(0.95 * len(vals)))] or vals[-1] or 1.0
    return [None if v is None else min(v / top, 1.5) for v in curve]


def batches(videos: dict) -> list:
    """The batches in the results, in order ("?" for videos exported before batches were recorded)."""
    keys = {str(v.get("batch") or "?") for v in videos.values()}
    return sorted(keys, key=lambda b: (not b.isdigit(), int(b) if b.isdigit() else 0, b))


def in_batch(videos: dict, batch: str | None) -> dict:
    """Only the videos of one batch (None = every batch)."""
    if batch is None:
        return dict(videos)
    return {v: d for v, d in videos.items() if str(d.get("batch") or "?") == str(batch)}


def tag_scores(moments: list, teacher: str) -> list:
    """Per tag, the teacher against your labels on the frames you've labelled:
    [{"tag", "yours" (frames you gave it), "tp", "fp", "fn", "precision", "recall", "f1"}],
    most-used tag first. Updates as you correct frames (moments with your saved labels)."""
    per = {}
    for m in moments:
        if m.get("mine") is None or teacher not in (m.get("teachers") or {}):
            continue
        mine = {norm(t): t for t in m["mine"]}
        pred = {norm(t): t for t in m["teachers"][teacher].get("tags") or []}
        for k in set(mine) | set(pred):
            name = mine.get(k) or pred.get(k)
            r = per.setdefault(k, {"tag": name, "yours": 0, "tp": 0, "fp": 0, "fn": 0})
            if k in mine:
                r["tag"] = mine[k]
                r["yours"] += 1
            r["tp"] += k in mine and k in pred
            r["fp"] += k in pred and k not in mine
            r["fn"] += k in mine and k not in pred
    out = []
    for r in per.values():
        p = r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else None
        rc = r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else None
        f = 2 * p * rc / (p + rc) if p and rc else (0.0 if p is not None and rc is not None else None)
        out.append({**r, "precision": p, "recall": rc, "f1": f})
    return sorted(out, key=lambda r: (-r["yours"], r["tag"].lower()))


THRESHOLD_STEPS = tuple(range(10, 100, 5))
MIN_FRAMES_FOR_THRESHOLD = 3       # need at least this many of your frames WITH the tag to suggest one


def suggest_thresholds(moments: list, teacher: str, rows: list, default: int = 50) -> list:
    """Per tag, the threshold that would have matched your labels best on the frames you've
    labelled (only moments tagged with confidences, from v1.32 on):
    [{"tag", "frames", "yours", "current", "suggested", "f1_now", "f1_best"}], biggest gain first.
    A tag the teacher didn't list counts as confidence -1 (no threshold can bring it in). Ties go to
    the higher threshold (fewer wrong tags)."""
    by_name = {r["tag"].lower(): r for r in rows}
    data = {}
    for m in moments:
        r = (m.get("teachers") or {}).get(teacher) or {}
        conf = r.get("conf")
        if m.get("mine") is None or not isinstance(conf, dict):
            continue
        mine = {norm(t) for t in m["mine"]}
        cl = {norm(k): v for k, v in conf.items()}
        for k in set(mine) | set(cl):
            if k in by_name:
                data.setdefault(k, []).append((cl.get(k, -1), k in mine))

    def f1(pairs, t):
        tp = sum(1 for c, y in pairs if c >= t and y)
        fp = sum(1 for c, y in pairs if c >= t and not y)
        fn = sum(1 for c, y in pairs if c < t and y)
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0

    out = []
    for k, pairs in data.items():
        yours = sum(1 for _, y in pairs if y)
        if yours < MIN_FRAMES_FOR_THRESHOLD:
            continue
        r = by_name[k]
        cur = default if r.get("threshold") is None else int(r["threshold"])
        best = max(THRESHOLD_STEPS, key=lambda t: (round(f1(pairs, t), 4), t))
        out.append({"tag": r["tag"], "frames": len(pairs), "yours": yours, "current": cur, "suggested": best,
                    "f1_now": round(f1(pairs, cur), 2), "f1_best": round(f1(pairs, best), 2)})
    return sorted(out, key=lambda x: (-(x["f1_best"] - x["f1_now"]), x["tag"].lower()))


# ---------------------------------------------------------------------------
# Per-batch report
# ---------------------------------------------------------------------------

def _cut_near(times, t: float, window: float = 0.5) -> bool:
    return any(abs(float(x) - float(t)) <= window for x in times or [])


def teacher_label(model) -> str:
    """'32B' / '32B FP8' / '8B' from a model name like Qwen/Qwen3-VL-32B-Instruct-FP8."""
    m = str(model or "").upper()
    size = "32B" if "32B" in m else "8B" if "8B" in m else (m.split("/")[-1] or "?")
    return f"{size} FP8" if "FP8" in m else size


def batch_report(videos: dict, teacher: str = MAIN_TEACHER) -> list:
    """One row per batch plus "All": footage, GPU time and cost per step, the teacher's scores
    against your labels, and scene cuts — so batches run with different settings can be compared.
    videos: results' videos (with your saved labels applied). Rows: dicts, the last is "All"."""
    import time as _time
    import trial_logic as TLG

    groups = {}
    for vid, v in videos.items():
        groups.setdefault(str(v.get("batch") or "?"), []).append(v)
    order = sorted(groups, key=lambda b: (not b.isdigit(), int(b) if b.isdigit() else 0, b))

    def row(name, vs):
        hours = sum(v.get("duration") or 0 for v in vs) / 3600
        vis_s = vis_load = vis_cost = tag_s = tag_load = tag_cost = 0.0
        prompts = unreadable = 0
        gpus_v, gpus_t, at, models, ex = set(), set(), [], set(), set()
        for v in vs:
            run = v.get("run") or {}
            sec = float(run.get("vision_seconds") or 0)
            load = float((run.get("vision_timings") or {}).get("load_models") or 0)
            gpu = run.get("vision_gpu") or v.get("gpu") or "L4"
            vis_s += sec
            vis_load += load
            vis_cost += TLG.stage_cost(sec, gpu)
            gpus_v.add(gpu)
            if run.get("vision_at"):
                at.append(run["vision_at"])
            t = (run.get("tagging") or {}).get(teacher) or {}
            ts, tl = float(t.get("seconds") or 0), float(t.get("load_seconds") or 0)
            tgpu = t.get("gpu") or "H100"
            tag_s += ts
            tag_load += tl
            tag_cost += TLG.stage_cost(ts + tl, tgpu)
            if t:
                gpus_t.add(tgpu)
            prompts += int(t.get("prompts") or 0)
            unreadable += int(t.get("unreadable") or 0)
            if t:
                models.add(teacher_label(t.get("model")))
                # older exports didn't record it: unknown, not "off"
                ex.add("?" if "examples_used" not in t else "on" if t["examples_used"] else "off")
        moments = [m for v in vs for m in v.get("moments") or []]
        agr = agreement(moments).get(teacher) or {}
        answers = [(v, b) for v in vs for b in v.get("boundaries") or [] if b.get("verdict") in ("cut", "no_cut")]
        right_new = sum(_cut_near(v.get("cuts_new"), b["t"]) == (b["verdict"] == "cut") for v, b in answers)
        right_old = sum(_cut_near(v.get("cuts_old"), b["t"]) == (b["verdict"] == "cut") for v, b in answers)
        total = vis_cost + tag_cost
        return {
            "batch": name, "videos": len(vs), "footage_h": round(hours, 2),
            "run": _time.strftime("%d %b %H:%M", _time.gmtime(min(at))) if at else None,
            "vision_gpu": "/".join(sorted(gpus_v)), "vision_min": round(vis_s / 60, 1),
            "vision_startup_min": round(vis_load / 60, 1), "vision_cost": round(vis_cost, 2),
            "tag_gpu": "/".join(sorted(gpus_t)), "tag_min": round(tag_s / 60, 1),
            "tag_startup_min": round(tag_load / 60, 1), "tag_cost": round(tag_cost, 2),
            "total_cost": round(total, 2), "cost_per_h": round(total / hours, 2) if hours else None,
            "tag_min_per_h": round(tag_s / 60 / hours, 1) if hours else None,
            "moments": prompts, "unreadable": unreadable,
            "labelled": agr.get("frames", 0), "exact": agr.get("exact", 0),
            "recall": agr.get("recall"), "precision": agr.get("precision"),
            "f1": (2 * agr["recall"] * agr["precision"] / (agr["recall"] + agr["precision"])
                   if agr.get("recall") and agr.get("precision") else None),
            "teacher": " + ".join(sorted(models)) or None,
            "examples": "mixed" if len(ex - {"?"}) > 1 else (next(iter(ex - {"?"}), None)),
            "cuts_old": sum(len(v.get("cuts_old") or []) for v in vs),
            "cuts_new": sum(len(v.get("cuts_new") or []) for v in vs),
            "cut_answers": len(answers),
            "cut_acc_new": right_new / len(answers) if answers else None,
            "cut_acc_old": right_old / len(answers) if answers else None,
        }

    rows = [row(b, groups[b]) for b in order]
    if len(rows) > 1:
        rows.append(row("All", [v for b in order for v in groups[b]]))
    return rows
