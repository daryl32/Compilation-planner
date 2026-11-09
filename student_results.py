"""Student model runs for 🎯 Training Labels → 🧪 Trial results (student_results_view.py): reading
catalogue/student_runs (written on the laptop by student_tags.py / student_runs.py), your notes per run,
and the small calculations the 🏋️ Training and 🧪 Test pages share. No Streamlit here."""

from __future__ import annotations

import csv
import io
import json
import time

from config import CATALOGUE_DIR

RUNS_DIR = CATALOGUE_DIR / "student_runs"
NOTES_FILE = CATALOGUE_DIR / "student_run_notes.csv"   # .csv: kept out of the catalogue *.json scans
NOTES_PENDING = f"cat:{NOTES_FILE.name}"
_CACHE: dict = {}


def _read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_index() -> dict | None:
    """{"created", "in_use", "runs": [...]} — oldest run first — or None if no run was exported yet."""
    path = RUNS_DIR / "index.json"
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    if _CACHE.get("stamp") != stamp:
        data = _read_json(path)
        _CACHE.clear()
        _CACHE.update(stamp=stamp, data=data if isinstance(data, dict) and data.get("runs") else None)
    return _CACHE.get("data")


def load_detail(name: str) -> dict | None:
    """One run's per-tag detail (<run>.json), or None."""
    key = f"detail:{name}"
    if key not in _CACHE:
        d = _read_json(RUNS_DIR / f"{name}.json")
        _CACHE[key] = d if isinstance(d, dict) else None
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Your notes — one line per run (student "student:v10", teacher batch "batch:12", test run "test:<id>")
# ---------------------------------------------------------------------------

def load_notes() -> dict:
    try:
        text = NOTES_FILE.read_text(encoding="utf-8")
    except OSError:
        return {}
    return {r["run"]: r.get("note") or "" for r in csv.DictReader(io.StringIO(text)) if r.get("run")}


def save_notes(notes: dict) -> None:
    """Write every note (empty ones dropped) and queue the file for ☁️ Save edits to Drive."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["run", "note", "updated"], lineterminator="\n")
    w.writeheader()
    now = time.strftime("%Y-%m-%d %H:%M")
    for run, note in sorted(notes.items()):
        if str(note or "").strip():
            w.writerow({"run": run, "note": str(note).strip(), "updated": now})
    tmp = NOTES_FILE.with_name(NOTES_FILE.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(NOTES_FILE)
    from library_common import mark_pending   # here, so reading never needs Streamlit
    mark_pending(NOTES_PENDING)


# ---------------------------------------------------------------------------
# Shared calculations
# ---------------------------------------------------------------------------

def noise_f1(frames) -> float:
    """How far F1 moves by chance alone on this many frames (the report's rule of thumb)."""
    return max(0.02, round(0.5 / max(int(frames or 0), 1) ** 0.5, 2))


PER_TAG_NOISE = 3          # a per-tag change of this many errors or fewer is chance

# How much the matcher leans on each tag group (Daryl's table, claude/tag-levels.md). Shot sizes are scored on
# their 🎥 scale, appearance (hair) per video and overlay not at all, so only these count towards the group F1s.
GROUP_IMPORTANCE = {"position": "High", "action": "High", "camera": "High",
                    "clothing": "Medium", "prop": "Medium", "shot": "Medium"}   # shot: only POV counts here —
                                                                                 # the sizes are on the 🎥 scale
IMPORTANCE_ORDER = ("High", "Medium")


def _f1_of(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"f1": None if f is None else round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def group_of_tags(vocab_rows: list) -> dict:
    """{tag (lower case): group (lower case)} from the vocabulary."""
    return {str(r["tag"]).lower(): str(r.get("group") or "").strip().lower() for r in vocab_rows or []}


def group_scores(per_tag: dict, groups: dict) -> dict:
    """Per tag group and per importance (High / Medium), F1 from the per-tag counts:
    {"groups": {group: {"f1", "tp", "fp", "fn"}}, "importance": {"High": {...}, "Medium": {...}}}.
    Tags left out of the overall F1 (hair, shot, text on screen) and groups the matcher doesn't use don't count."""
    from trial_logic import not_in_f1
    g_tot, i_tot = {}, {}
    for t, s in (per_tag or {}).items():
        if not s or not_in_f1(t):
            continue
        g = groups.get(str(t).lower())
        imp = GROUP_IMPORTANCE.get(g)
        if not imp:
            continue
        for tot, key in ((g_tot, g), (i_tot, imp)):
            a = tot.setdefault(key, [0, 0, 0])
            a[0] += s.get("tp") or 0
            a[1] += s.get("fp") or 0
            a[2] += s.get("fn") or 0
    return {"groups": {g: _f1_of(*v) for g, v in g_tot.items()},
            "importance": {k: _f1_of(*v) for k, v in i_tot.items()}}


def best_on_test(runs: list) -> str | None:
    """The student run with the best test F1 (on today's frames, after rescore-all)."""
    scored = [r for r in runs if (r.get("test") or {}).get("student", {}).get("f1") is not None]
    return max(scored, key=lambda r: r["test"]["student"]["f1"])["name"] if scored else None


def errors(s: dict | None) -> int | None:
    return None if not s else (s.get("fp") or 0) + (s.get("fn") or 0)


def tag_grid(details: dict, side: str) -> tuple:
    """(tags, {run: {tag: F1}}) for the tags × runs grid. side: "train" or "test"."""
    tags, out = [], {}
    for name, d in details.items():
        per = (d or {}).get("per_tag") or {}
        row = {}
        for t, x in per.items():
            s = (x or {}).get(side) or {}
            if s.get("f1") is not None or (s.get("tp") or 0) + (s.get("fn") or 0):
                row[t] = s.get("f1")
                if t not in tags:
                    tags.append(t)
        if row:
            out[name] = row
    return tags, out


def settings_text(s: dict | None) -> str:
    """The settings a run was trained with, short — for the runs table."""
    s = s or {}
    bits = []
    if s.get("cutoffs"):
        bits.append(f"cut-offs {s['cutoffs']}")
    rng = s.get("cutoff_range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        bits.append(f"{rng[0]:.2f}–{rng[1]:.2f}")
    for k, label in (("pose", "pose"), ("context", "context")):
        if s.get(k):
            bits.append(label)
    if s.get("teacher_trust") is False:
        bits.append("no teacher trust")
    if s.get("folds") and s["folds"] != 5:
        bits.append(f"{s['folds']} folds")
    if s.get("label_weight") and s["label_weight"] != 5.0:
        bits.append(f"labels ×{s['label_weight']:g}")
    return " · ".join(bits)


# ---------------------------------------------------------------------------
# 📊 Results page (claude/results-pages-audit.md): one row per run, the headline, what to label next, and the
# same as a markdown report to paste into a Claude chat. No Streamlit here.
# ---------------------------------------------------------------------------

def _test_per_tag(detail: dict | None, side: str = "test") -> dict:
    return {t: (x or {}).get(side) for t, x in ((detail or {}).get("per_tag") or {}).items() if (x or {}).get(side)}


def run_rows(runs: list, groups: dict, detail_of=None) -> list:
    """Student runs with a test score, oldest first: the numbers the Results page and the report show."""
    detail_of = detail_of or load_detail
    out = []
    for r in runs:
        t = r.get("test") or {}
        s = t.get("student") or {}
        if s.get("f1") is None:
            continue
        sc = group_scores(_test_per_tag(detail_of(r["name"])), groups)
        te = t.get("teacher") or {}
        tr = r.get("training") or {}
        out.append({
            "name": r["name"], "created": r.get("created"), "in_use": bool(r.get("in_use")),
            "changed": r.get("changed") or "", "settings": settings_text(r.get("settings")),
            "high": (sc["importance"].get("High") or {}).get("f1"),
            "medium": (sc["importance"].get("Medium") or {}).get("f1"),
            "groups": {g: v.get("f1") for g, v in sc["groups"].items()},
            "f1": s.get("f1"), "precision": s.get("precision"), "recall": s.get("recall"),
            "fp": s.get("fp"), "fn": s.get("fn"), "frames": t.get("frames"),
            "vs_teacher": None if te.get("f1") is None else round(s["f1"] - te["f1"], 3),
            "first_f1": t.get("first_f1"), "train_f1": tr.get("f1"), "labels": r.get("labels"),
            "partial": r.get("partial"), "minutes": r.get("minutes"),
            "shot": t.get("shot"), "teacher_shot": t.get("teacher_shot"), "reference": t.get("reference"),
            "teacher_f1": te.get("f1"),
        })
    return out


def teacher_baseline(rows: list, groups: dict, detail_of=None) -> dict | None:
    """The reference teacher run on the same frames as the newest student run."""
    detail_of = detail_of or load_detail
    if not rows or rows[-1]["teacher_f1"] is None:
        return None
    last = rows[-1]
    sc = group_scores(_test_per_tag(detail_of(last["name"]), "test_teacher"), groups)
    return {"name": f"Teacher ({last['reference'] or 'reference'})", "f1": last["teacher_f1"],
            "high": (sc["importance"].get("High") or {}).get("f1"),
            "medium": (sc["importance"].get("Medium") or {}).get("f1"),
            "groups": {g: v.get("f1") for g, v in sc["groups"].items()}, "shot": last.get("teacher_shot")}


def best_high(rows: list) -> str | None:
    """The run with the best High F1 (overall F1 where High isn't known)."""
    ok = [r for r in rows if (r["high"] if r["high"] is not None else r["f1"]) is not None]
    return max(ok, key=lambda r: r["high"] if r["high"] is not None else r["f1"])["name"] if ok else None


def headline(rows: list) -> dict | None:
    """Newest run vs the one before, on High F1 (overall F1 when there's no High score)."""
    if not rows:
        return None
    last = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None
    key = "high" if last["high"] is not None and (prev is None or prev["high"] is not None) else "f1"
    noise = noise_f1(last.get("frames"))
    delta = None if prev is None or prev[key] is None or last[key] is None else round(last[key] - prev[key], 3)
    if delta is None:
        verdict = "first run with a test score"
    elif abs(delta) <= noise:
        verdict = "within chance"
    else:
        verdict = "better — more than chance" if delta > 0 else "worse — more than chance"
    return {"last": last, "prev": prev, "key": key, "delta": delta, "noise": noise, "verdict": verdict}


def label_next(detail: dict | None, groups: dict) -> list:
    """Per tag for one run, the tags worth labelling first: High groups, then Medium, then the rest; within
    each, most test errors first. Tags left out of the F1 (hair, shot, text on screen) aren't listed."""
    from trial_logic import not_in_f1
    order = {imp: k for k, imp in enumerate(IMPORTANCE_ORDER)}
    out = []
    for t, x in ((detail or {}).get("per_tag") or {}).items():
        if not_in_f1(t):
            continue
        s, te = (x or {}).get("test") or {}, (x or {}).get("test_teacher") or {}
        n = (s.get("tp") or 0) + (s.get("fn") or 0)
        es, et = errors(s), (errors(te) if te else None)
        if not n and not es:
            continue
        g = groups.get(str(t).lower(), "")
        imp = GROUP_IMPORTANCE.get(g)
        out.append({"tag": t, "group": g or "—", "importance": imp or "—", "f1": s.get("f1"), "errors": es,
                    "wrong": s.get("fp"), "missed": s.get("fn"), "test_labels": n, "teacher_errors": et,
                    "worse_than_teacher": es is not None and et is not None and es - et > PER_TAG_NOISE,
                    "train_labels": (x or {}).get("train_labels"), "teacher_trust": (x or {}).get("teacher_trust")})
    out.sort(key=lambda r: (order.get(r["importance"], 9), -(r["errors"] or 0), r["tag"].lower()))
    return out


def _f(x, pct=False) -> str:
    if x is None:
        return "–"
    return f"{round(x * 100)}%" if pct else f"{x:.2f}"


def _sign(x) -> str:
    return "–" if x is None else f"{x:+.2f}"


def report_markdown(rows: list, baseline: dict | None, head: dict | None, label_rows: list, label_run: str,
                    hair: dict | None = None, created: str = "") -> str:
    """📥 The latest results as markdown, to paste into a Claude chat."""
    md = [f"# Student tag model — results ({created or time.strftime('%Y-%m-%d %H:%M')})", "",
          "Scores are on the 🧪 test set (videos no model trains on). **High F1** = position, action, camera focus "
          "(what the matcher leans on most); **Medium F1** = clothing, prop, POV. Hair (per video), shot size (scale) and "
          "text on screen are not in any F1.", ""]
    if head:
        last, prev = head["last"], head["prev"]
        what = "High F1" if head["key"] == "high" else "F1"
        md.append(f"**Newest run {last['name']}**: {what} {_f(last[head['key']])}"
                  + (f" vs {prev['name']} {_f(prev[head['key']])} ({head['delta']:+.2f}) — {head['verdict']} "
                     f"(chance at {last.get('frames')} frames: ±{head['noise']:.2f})" if prev else "")
                  + ".")
        md.append("")
    md += ["## Runs (oldest first)", "",
           "| Run | When | High F1 | Medium F1 | F1 | vs teacher | Right | Found | Training F1 | Your labels | Batch answers | What changed |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {'⭐ ' if r['in_use'] else ''}{r['name']} | {r['created'] or ''} | {_f(r['high'])} | "
                  f"{_f(r['medium'])} | {_f(r['f1'])} | {_sign(r['vs_teacher'])} | "
                  f"{_f(r['precision'], True)} | {_f(r['recall'], True)} | {_f(r['train_f1'])} | {r['labels'] or '–'} | "
                  f"{r['partial'] or '–'} | {(r['changed'] or '').replace('|', '/')} |")
    if baseline:
        md.append(f"| {baseline['name']} | | {_f(baseline['high'])} | {_f(baseline['medium'])} | {_f(baseline['f1'])} "
                  f"| | | | | | | baseline on the same frames |")
    if rows:
        groups = sorted({g for r in rows[-3:] for g in r["groups"]}, key=lambda g: (
            IMPORTANCE_ORDER.index(GROUP_IMPORTANCE.get(g, "Medium")), g))
        if groups:
            md += ["", "## By tag group (newest runs)", "", "| Run | " + " | ".join(groups) + " |",
                   "|---|" + "---|" * len(groups)]
            for r in rows[-3:] + ([baseline] if baseline else []):
                md.append(f"| {r['name']} | " + " | ".join(_f(r['groups'].get(g)) for g in groups) + " |")
    if label_rows:
        md += ["", f"## Per tag — {label_run} (High groups first, most errors first)", "",
               "| Group | Tag | Test F1 | Errors (wrong / missed) | Teacher errors | Test labels | Training labels | Teacher trust |",
               "|---|---|---|---|---|---|---|---|"]
        for x in label_rows:
            md.append(f"| {x['group']} ({x['importance']}) | {x['tag']}{' ⚠️' if x['worse_than_teacher'] else ''} | "
                      f"{_f(x['f1'])} | {x['errors']} ({x['wrong']} / {x['missed']}) | "
                      f"{'–' if x['teacher_errors'] is None else x['teacher_errors']} | {x['test_labels']} | "
                      f"{x['train_labels'] if x['train_labels'] is not None else '–'} | {_f(x['teacher_trust'])} |")
        md.append("")
        md.append("⚠️ = the student makes more than 3 errors more than the teacher on this tag.")
    if rows and rows[-1].get("shot"):
        sh = rows[-1]["shot"]
        md += ["", "## Shot size (scale extreme close-up → close-up → medium → wide)", "",
               f"{rows[-1]['name']}: within one step {_f(sh.get('within_one'), True)} · exact {_f(sh.get('exact'), True)} · "
               f"big misses {_f(sh.get('big_miss'), True)} · no shot given {_f(sh.get('none_given'), True)} "
               f"({sh.get('frames')} frames)"
               + (f" · teacher within one step {_f((baseline or {}).get('shot', {}).get('within_one'), True)}"
                  if baseline and baseline.get("shot") else "")]
    if hair and hair.get("checked"):
        md += ["", "## Hair colour per video", "",
               f"Vote right {hair['right']} / {hair['checked']} test videos · main colour right {hair['main_right']} / "
               f"{hair['checked']} · unsure {hair['unsure']} (vote from {', '.join(hair['models'])})."]
    return "\n".join(md) + "\n"
