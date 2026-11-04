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


def look_line(look: dict) -> str:
    """The teacher's look steps as one short line for a card (the summary is the caption)."""
    look = look or {}
    parts = []
    if look.get("women") or look.get("men"):
        parts.append(f"{look.get('women', '?')} woman/women, {look.get('men', '?')} man/men")
    if look.get("penetration"):
        parts.append("penetration " + look["penetration"]
                     + (f" ({look['penetration_with']}, {look['penetration_where']})"
                        if look.get("penetration_with") not in (None, "not applicable", "can't tell") else ""))
    if look.get("hidden") and look["hidden"] != "nothing important hidden":
        parts.append(look["hidden"])
    if look.get("her_posture"):
        parts.append(f"her: {look['her_posture']}, front to {look.get('her_front_faces', '?')}")
    if look.get("partner_posture"):
        where = (f"head {look['partner_head']}" if look.get("partner_head")
                 else f"legs {look.get('partner_legs_point', '?')}")
        parts.append(f"partner: {look['partner_posture']}, {where}")
    if look.get("on_top"):
        parts.append(f"on top: {look['on_top']}")
    if look.get("facing"):
        parts.append(f"she faces partner: {look['facing']}")
    acts = [f"{name} {look[k]}" for k, name in (("her_hands", "her hands"), ("her_mouth", "her mouth"),
                                                ("mouth_depth", "depth"), ("her_feet", "her feet"),
                                                ("partner_hands", "his hands"), ("partner_mouth", "his mouth"))
            if look.get(k) and look[k] not in ("not visible", "not applicable", "can't tell", "other",
                                               "open / other")]
    if acts:
        parts.append(", ".join(acts))
    if look.get("viewpoint"):
        parts.append("POV" if look["viewpoint"].startswith("a participant") else look["viewpoint"])
    return " · ".join(parts)


FILTERS = {
    "all": "All moments",
    "todo": "Not reviewed by me yet",
    "mine": "Only frames I labelled",
    "wrong": "A model differs from me",
    "disagree": "The models disagree",
    "rules": "⚠️ My tags break a rule",
    "neighbours": "↔️ Filled in from the moments either side",
    "suspect": "🔍 Check my label — the test runs all disagree with it",
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


def filter_moments(moments: list, how: str = "all", tag: str | None = None, problems=None,
                   skip_old: bool = False) -> list:
    """problems: tags -> list of rule problems (training_labels.rule_problems with your vocabulary),
    for the "rules" filter — your labelled frames whose tags break a rule. skip_old: the tag filter looks
    only at the models' tags (and yours), not the old tags."""
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
        if how == "neighbours" and not any((r or {}).get("neighbour_added") for r in (m.get("teachers") or {}).values()):
            continue
        if tag and not any(norm(tag) in s for k, s in tag_sets(m).items() if not (skip_old and k == "old")):
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
# 🧪 Test set runs
# ---------------------------------------------------------------------------

def load_test_runs() -> dict | None:
    """test_runs.json from the last test run / export: {"set": {...}, "runs": [...]}, or None."""
    try:
        data = json.loads((RESULTS_DIR / "test_runs.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("runs") else None


def load_check_runs() -> dict | None:
    """check_runs.json — 🔍 label-check runs over your labelled training frames, or None."""
    try:
        data = json.loads((RESULTS_DIR / "check_runs.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("runs") else None


def test_run_scores(runs: list, labels: list, tags: list) -> list:
    """Each run scored on your labels as they are now: [{"run", "score"}] (score as
    trial_logic.score_tags: "n", "overall", "per_tag")."""
    import trial_logic as TLG
    gold = [i for i in labels if i.get("type") == "tags" and not i.get("skipped")]
    return [{"run": r, "score": TLG.score_tags(TLG.run_pairs(r.get("frames") or [], gold), tags),
             "copied": TLG.copied_scored(r.get("frames") or [], gold)} for r in runs]


def test_set_videos() -> set:
    """The 🧪 test set's video ids (empty if there's no test set yet)."""
    return {x.get("video_id") for x in (((load_test_runs() or {}).get("set") or {}).get("videos") or [])
            if x.get("video_id")}


def tag_changes(a: dict, b: dict) -> list:
    """Per tag, run A → run B: [{"tag", "wrong_a", "wrong_b", "missed_a", "missed_b", "change"}] for
    the tags whose errors changed, biggest change first (change < 0 = fewer errors in B)."""
    out = []
    for t, pa in a["per_tag"].items():
        pb = b["per_tag"].get(t) or {"fp": 0, "fn": 0}
        if (pa["fp"], pa["fn"]) != (pb["fp"], pb["fn"]):
            out.append({"tag": t, "wrong_a": pa["fp"], "wrong_b": pb["fp"], "missed_a": pa["fn"],
                        "missed_b": pb["fn"], "change": (pb["fp"] + pb["fn"]) - (pa["fp"] + pa["fn"])})
    return sorted(out, key=lambda x: (-abs(x["change"]), x["tag"].lower()))


def suspect_labels(test_data: dict, run_ids: list, videos: dict, labelled_at: dict) -> dict:
    """Your labelled frames that every chosen test run disagrees with, on at least one tag:
    {(video_id, t rounded to 2): {"there": tags all runs gave that you didn't, "not_there": tags you
    gave that no run did, "runs": {run id: its tags}, "detail": {run id: {"look", "nb" (filled in from
    the moments either side), "ld" (dropped by the look check)} — only what the run recorded}}}.
    One run is enough (e.g. to read a look run's answers on your frames). A frame you've saved since the newest chosen
    run (labelled_at, "YYYY-MM-DDTHH:MM:SS") counts as checked and is left out. videos: the results'
    videos with your labels overlaid (with_saved_labels)."""
    runs = [r for r in (test_data or {}).get("runs") or [] if r["id"] in run_ids]
    if not runs:
        return {}
    since = max(str(r.get("created") or "") for r in runs)
    frames_by = [{(f["v"], round(float(f["t"]), 2)): f for f in r.get("frames") or [] if f.get("t") is not None}
                 for r in runs]
    tags_by = [{k: f.get("tags") or [] for k, f in fb.items()} for fb in frames_by]
    out = {}
    for v, vd in videos.items():
        for m in vd["moments"]:
            if m.get("mine") is None:
                continue
            key = (v, round(float(m["t"]), 2))
            if not all(key in tb for tb in tags_by):
                continue
            when = str(labelled_at.get(key) or "").replace("T", " ")[:16]
            if since and when and when > since:
                continue
            mine = {norm(t): t for t in m["mine"]}
            sets = [{norm(t): t for t in tb[key]} for tb in tags_by]
            there = [sets[0][k] for k in sets[0] if k not in mine and all(k in s for s in sets[1:])]
            not_there = [mine[k] for k in mine if not any(k in s for s in sets)]
            if there or not_there:
                detail = {r["id"]: {x: fb[key][x] for x in ("look", "nb", "ld") if fb[key].get(x)}
                          for r, fb in zip(runs, frames_by)}
                out[key] = {"there": there, "not_there": not_there,
                            "runs": {r["id"]: tb[key] for r, tb in zip(runs, tags_by)},
                            "detail": {k: d for k, d in detail.items() if d}}
    return out


VOCAB_FIELDS = ("group", "kind", "one_of", "teacher", "threshold", "double_check", "description",
                "not_to_be_confused_with", "tell_apart", "never_with", "always_with", "only_with",
                "usually_with", "group_order", "group_required")


def vocab_changes(old_csv: str, new_csv: str) -> list:
    """What changed in the vocabulary between two runs, as readable lines."""
    import training_labels as TLB
    if not old_csv or not new_csv:
        return []
    a = {r["tag"].lower(): r for r in TLB.parse_vocabulary(old_csv)}
    b = {r["tag"].lower(): r for r in TLB.parse_vocabulary(new_csv)}
    out = [f"➕ {b[k]['tag']} added" for k in b if k not in a]
    out += [f"➖ {a[k]['tag']} removed" for k in a if k not in b]
    for k in b:
        if k not in a:
            continue
        for fld in VOCAB_FIELDS:
            x, y = a[k].get(fld), b[k].get(fld)
            if x != y and not (x in (None, "", False) and y in (None, "", False)):
                show = lambda v: "—" if v in (None, "", False) else ("yes" if v is True else str(v))
                out.append(f"{b[k]['tag']} · {fld}: {show(x)} → {show(y)}")
    return out


# ---------------------------------------------------------------------------
# Per-batch report
# ---------------------------------------------------------------------------

def _cut_near(times, t: float, window: float = 0.5) -> bool:
    return any(abs(float(x) - float(t)) <= window for x in times or [])


def teacher_label(model) -> str:
    """'32B' / '32B FP8' / '8B' from a model name like Qwen/Qwen3-VL-32B-Instruct-FP8, plus
    ' + LoRA try2' when an adapter tagged it ("…-FP8 + LoRA try2")."""
    raw = str(model or "")
    lora = raw.split("+ LoRA", 1)[1].strip() if "+ LoRA" in raw else ""
    m = raw.split("+ LoRA", 1)[0].upper()
    size = "32B" if "32B" in m else "8B" if "8B" in m else (m.split("/")[-1].strip() or "?")
    name = f"{size} FP8" if "FP8" in m else size
    return f"{name} + LoRA {lora}" if lora else name


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


# ---------------------------------------------------------------------------
# 👁️ Look checks — your verdict on each of a look run's answers, per frame
# ---------------------------------------------------------------------------

LOOK_CHECKS_FILE = CATALOGUE_DIR / "look_checks.json"
LOOK_CHECKS_PENDING = f"cat:{LOOK_CHECKS_FILE.name}"
def look_steps() -> list:
    """[(key, name, choices incl. "can't tell")]: your look steps (📒 Vocabulary → 👁️ Look steps,
    switched off ones too — older runs may have answered them), else the built-in ones."""
    import trial_logic as TLG
    import training_labels as TLB
    return [(r["key"], r["name"], list(r["choices"]) + [TLG.CANT] + ([TLG.NA] if r["if_penetration"] and TLG.NA not in r["choices"] else []))
            for r in TLB.load_look_steps()]


def look_changes(old: list | None, new: list | None) -> list:
    """What changed in the look steps between two runs (each run keeps the steps it used):
    ["step added / removed / question / choices …"]. Runs from before v1.41 kept none."""
    if not old or not new:
        return []
    a = {r["key"]: r for r in old}
    b = {r["key"]: r for r in new}
    out = [f"👁️ {k} · step added" for k in b if k not in a] + [f"👁️ {k} · step removed" for k in a if k not in b]
    for k in a:
        if k in b:
            if a[k].get("question") != b[k].get("question"):
                out.append(f"👁️ {k} · question: {a[k].get('question')} → {b[k].get('question')}")
            if bool(a[k].get("if_penetration")) != bool(b[k].get("if_penetration")):
                out.append(f"👁️ {k} · only with penetration: {'yes' if b[k].get('if_penetration') else 'no'}")
            if list(a[k].get("choices") or []) != list(b[k].get("choices") or []):
                out.append(f"👁️ {k} · choices: {' | '.join(a[k].get('choices') or [])} → "
                           f"{' | '.join(b[k].get('choices') or [])}")
    if [k for k in a if k in b] != [k for k in b if k in a]:
        out.append("👁️ order of the steps changed")
    return out


def look_check_key(run_id: str, video_id: str, t) -> str:
    return f"{run_id}|{video_id}|{round(float(t), 2):.2f}"


def load_look_checks() -> dict:
    """{key: {"run", "video_id", "t", "model": {step: answer}, "mine": {step: answer}, "my_tags",
    "run_tags", "checked_at"}} — your look checks (look_checks.json in the catalogue)."""
    try:
        data = json.loads(LOOK_CHECKS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("items") or {} if isinstance(data, dict) else {}


def save_look_check(run_id: str, video_id: str, t, model: dict, mine: dict, my_tags=None, run_tags=None) -> dict:
    """Save your answers for one frame's look (a step you left as the model had it = right).
    Queued for Drive with your other catalogue edits. Returns the saved item."""
    import time
    import training_labels as TLB
    from library_common import mark_pending
    items = load_look_checks()
    key = look_check_key(run_id, video_id, t)
    steps = [k for k, _, _ in look_steps()]
    item = {"run": run_id, "video_id": video_id, "t": round(float(t), 2),
            "model": {k: model.get(k) for k in steps if model.get(k)},
            "mine": {k: mine[k] for k in steps if mine.get(k)},
            "summary": model.get("summary") or "",
            "my_tags": list(my_tags or []), "run_tags": list(run_tags or []),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    items[key] = item
    TLB._atomic_write(LOOK_CHECKS_FILE, json.dumps({"version": 1, "items": items}, indent=1))
    mark_pending(LOOK_CHECKS_PENDING)
    return item


def look_check_summary(items) -> list:
    """Per step: [{"step", "name", "checked", "right", "pct", "fixes": [(model answer, yours, count)]}] —
    how often the model's answer was right, and its most common mistakes."""
    from collections import Counter
    out = []
    for k, name, _ in look_steps():
        n = right = 0
        fixes = Counter()
        for it in items:
            a, b = (it.get("model") or {}).get(k), (it.get("mine") or {}).get(k)
            if not a or not b:
                continue
            n += 1
            if a == b:
                right += 1
            else:
                fixes[(a, b)] += 1
        out.append({"step": k, "name": name, "checked": n, "right": right,
                    "pct": None if not n else round(100 * right / n),
                    "fixes": [(a, b, c) for (a, b), c in fixes.most_common(3)]})
    return out


def look_checks_csv(items) -> str:
    """One row per frame and step, for sending on: run, video, time, step, the model's answer,
    yours, right (yes/no), your tags, the run's tags, the model's summary."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["run", "video_id", "t", "step", "model_answer", "my_answer", "right", "my_tags", "run_tags",
                "model_summary", "checked_at"])
    for it in sorted(items, key=lambda x: (x.get("run", ""), x.get("video_id", ""), x.get("t", 0))):
        for k, _, _ in look_steps():
            a, b = (it.get("model") or {}).get(k), (it.get("mine") or {}).get(k)
            if not a and not b:
                continue
            w.writerow([it.get("run"), it.get("video_id"), it.get("t"), k, a or "", b or "",
                        "yes" if a and a == b else "no", "; ".join(it.get("my_tags") or []),
                        "; ".join(it.get("run_tags") or []), it.get("summary") or "", it.get("checked_at") or ""])
    return buf.getvalue()
