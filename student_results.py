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

# How much the matcher leans on each tag group (Daryl's table, claude/tag-levels.md). Shot is scored on its 🎥
# scale, appearance (hair) per video and overlay not at all, so only these count towards the group F1s.
GROUP_IMPORTANCE = {"position": "High", "action": "High", "camera": "High",
                    "clothing": "Medium", "prop": "Medium"}
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
