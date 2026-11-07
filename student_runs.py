r"""
student_runs.py — every student tag model run as a small record the web app can show
(claude/student-results-plan.md, step 1). No torch, no GPU: it only reads the runs' meta.json.

student_tags.py fills each run's meta.json with:
    "training": the cross-validated score on the TRAINING videos — each fold model's answers on the
                videos it didn't train on, against your labels only — overall and per tag, plus your
                labels per tag at the time;
    "test":     the score on the 🧪 test-set videos (student plain / with rules / the teacher on the same
                frames), overall and per tag, and when it was scored; "test_first" keeps the score from
                the first time (when the run was trained) once `rescore-all` scores it again;
    "settings": what the run was trained with.
This file turns those into catalogue/student_runs/ on Drive:
    index.json      one line per run: when, the headline numbers, what changed, ⭐ in use
    <run>.json      the per-tag detail for one run
"What changed" is worked out here, by comparing each run with the one trained before it. Runs from
before 6 Nov have no settings record, so their line comes from HISTORY (the session summaries' table).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# What each run before the 6 Nov record changed — from claude/session-summary-nov5-2026.md.
HISTORY = {
    "v1": "First student: a small head on the SigLIP2 embeddings, ±2 s around each moment.",
    "v2": "5 fold models averaged; safer cut-offs (picked on every held-out training video).",
    "v3": "Your Reviewer-corrected scenes added as training data (hand-picked scene-wide tags).",
    "v4": "Reviewer scene answers used only where they agree with your moment labels.",
    "v5": "Teacher trust: the teacher's answers count by its F1 on your labels, per tag.",
    "v5b": "v5 retrained on the newer data (no other change).",
    "v6": "Body pose added as extra input.",
    "v7": "Time context: a second step over ±10 s of scores.",
    "v8": "First 🎓 Student's picks round (~100 labels, mostly extreme close-up).",
    "v9": "Weighted cut-offs: held-out moments count by training weight; smoothed F1 curve.",
    "v10": "Overnight picks labelled (+392 labels) and more test-set labels; v1–v8 cut-off rule.",
    "v11": "Cut-offs kept within 0.40–0.70 (were 0.25–0.85).",
}

SETTING_NAMES = {"label_weight": "label weight", "folds": "fold models", "epochs": "max epochs",
                 "teacher_trust": "teacher trust", "pose": "body pose", "context": "time context",
                 "cutoffs": "cut-off rule", "cutoff_range": "cut-off range"}


def _read(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def settings_of(meta: dict) -> dict:
    """The settings a run was trained with — from its "settings" record, else what older metas kept."""
    s = dict(meta.get("settings") or {})
    s.setdefault("label_weight", meta.get("label_weight"))
    s.setdefault("folds", meta.get("folds"))
    s.setdefault("pose", bool(meta.get("pose_dim")))
    s.setdefault("context", bool(meta.get("context")))
    s.setdefault("teacher_trust", meta.get("teacher_trust") is not None)
    if meta.get("cutoffs"):
        s.setdefault("cutoffs", meta["cutoffs"])
    if meta.get("cutoff_range"):
        s.setdefault("cutoff_range", list(meta["cutoff_range"]))
    return {k: v for k, v in s.items() if v is not None}


def _fmt(k: str, v) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if k == "cutoff_range" and isinstance(v, (list, tuple)) and len(v) == 2:
        return f"{v[0]:.2f}–{v[1]:.2f}"
    return str(v)


def what_changed(prev: dict | None, cur: dict) -> str:
    """One plain line: how this run differs from the one before it (settings, your labels, data)."""
    name = cur.get("name", "")
    if prev is None:
        return HISTORY.get(name, "First run.")
    parts = []
    sp, sc = settings_of(prev), settings_of(cur)
    for k in SETTING_NAMES:
        if k in sc and k in sp and sp[k] != sc[k]:
            parts.append(f"{SETTING_NAMES[k]} {_fmt(k, sc[k])} (was {_fmt(k, sp[k])})")
        elif k in sc and k not in sp and k in ("cutoffs", "cutoff_range"):
            pass                                   # older runs didn't record it: not a change
    lp = (prev.get("training") or {}).get("labels_per_tag") or {}
    lc = (cur.get("training") or {}).get("labels_per_tag") or {}
    np_, nc = (prev.get("counts") or {}).get("labels"), (cur.get("counts") or {}).get("labels")
    if np_ is not None and nc is not None and nc != np_:
        gains = sorted(((t, lc.get(t, 0) - lp.get(t, 0)) for t in lc), key=lambda x: -x[1]) if lp else []
        top = ", ".join(f"{t} {d:+d}" for t, d in gains[:4] if d > 0)
        parts.append(f"{nc - np_:+d} labels" + (f" ({top})" if top else ""))
    tp_, tc = (prev.get("test") or {}).get("frames"), (cur.get("test") or {}).get("frames")
    if tp_ and tc and tc != tp_:
        parts.append(f"test frames {tp_} → {tc}")
    if not parts:
        return HISTORY.get(name, "Same settings and data — a retrain (differences are chance).")
    hist = HISTORY.get(name)
    return (hist + " · " if hist else "") + "; ".join(parts) + "."


def _overall(sc: dict | None) -> dict:
    sc = sc or {}
    return {k: sc.get(k) for k in ("f1", "precision", "recall", "fp", "fn", "tp")}


def run_line(meta: dict, changed: str, in_use: bool) -> dict:
    """The run's line in index.json."""
    test, tr = meta.get("test") or {}, meta.get("training") or {}
    sc = test.get("scores") or meta.get("scores") or {}
    best = test.get("best") or ("rules" if ((sc.get("rules") or {}).get("f1") or 0)
                                >= ((sc.get("plain") or {}).get("f1") or 0) else "plain")
    first = (meta.get("test_first") or {}).get("scores") or {}
    c = meta.get("counts") or {}
    return {
        "name": meta.get("name"), "created": meta.get("created"), "in_use": in_use, "changed": changed,
        "settings": settings_of(meta), "minutes": meta.get("train_minutes"), "videos": meta.get("videos"),
        "labels": c.get("labels"), "teacher_moments": c.get("teacher"),
        "scene_moments": (c.get("scene_extra") or 0) + (c.get("teacher_in_scenes") or 0),
        "partial": (c.get("partial_on_teacher") or 0) + (c.get("partial_alone") or 0),
        "labels_newest": meta.get("labels_newest"),
        "training": {"frames": tr.get("frames"), **_overall(tr.get("scores"))} if tr else None,
        "test": {"frames": test.get("frames") or (sc.get(best) or {}).get("n"), "scored_at": test.get("scored_at"),
                 "best": best, "student": _overall(sc.get(best)), "plain": _overall(sc.get("plain")),
                 "rules": _overall(sc.get("rules")), "teacher": _overall(sc.get("teacher")) if sc.get("teacher") else None,
                 "all_frames": _overall(sc.get("rules_all_frames")),
                 "all_frames_n": test.get("all_frames_n"), "reference": meta.get("reference_run"),
                 "first_f1": (first.get(best) or {}).get("f1") if first else None,
                 "first_frames": (meta.get("test_first") or {}).get("frames")} if sc else None,
    }


def run_detail(meta: dict) -> dict:
    """<run>.json: per tag, training and test."""
    tags = meta.get("tags") or []
    th = dict(zip(tags, meta.get("thresholds") or []))
    trust = meta.get("teacher_trust") or {}
    tr, te = meta.get("training") or {}, meta.get("test") or {}
    per = {}
    for t in tags:
        tt = trust.get(t.lower()) or {}
        per[t] = {"cutoff": th.get(t), "teacher_trust": tt.get("f1"), "trust_weight": tt.get("weight"),
                  "train_labels": (tr.get("labels_per_tag") or {}).get(t),
                  "train": (tr.get("per_tag") or {}).get(t),
                  "test": (te.get("per_tag") or {}).get(t),
                  "test_teacher": (te.get("teacher_per_tag") or {}).get(t)}
    return {"name": meta.get("name"), "created": meta.get("created"), "per_tag": per,
            "test_best": te.get("best"), "reference": meta.get("reference_run")}


def export_runs(models_dir: Path, dest: Path, in_use: str | None = None) -> int:
    """catalogue/student_runs/ from every run in models_dir, keeping the lines of runs made elsewhere (another
    machine's models) that are already there. Returns how many runs are listed."""
    metas = []
    for d in sorted(Path(models_dir).iterdir()) if Path(models_dir).exists() else []:
        m = _read(d / "meta.json")
        if isinstance(m, dict) and m.get("name"):
            metas.append(m)
    metas.sort(key=lambda m: (m.get("created") or "", m.get("name")))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    lines, prev = [], None
    for m in metas:
        lines.append(run_line(m, what_changed(prev, m), m.get("name") == in_use))
        tmp = dest / f"{m['name']}.json.tmp"
        tmp.write_text(json.dumps(run_detail(m)), encoding="utf-8")
        tmp.replace(dest / f"{m['name']}.json")
        prev = m
    # Runs this machine doesn't have (e.g. the laptop's, when the server runs the student) keep their
    # lines, so the app's list never loses history because of where a command ran.
    have = {ln.get("name") for ln in lines}
    old = _read(dest / "index.json", {}) or {}
    kept = [dict(ln, in_use=ln.get("name") == in_use) if in_use else ln
            for ln in (old.get("runs") or []) if isinstance(ln, dict) and ln.get("name") not in have
            and (dest / f"{ln['name']}.json").exists()]
    if kept:
        lines = sorted(kept + lines, key=lambda ln: (ln.get("created") or "", ln.get("name") or ""))
    idx = {"created": time.strftime("%Y-%m-%d %H:%M"), "in_use": in_use or old.get("in_use"), "runs": lines}
    tmp = dest / "index.json.tmp"
    tmp.write_text(json.dumps(idx, indent=1), encoding="utf-8")
    tmp.replace(dest / "index.json")
    return len(lines)
