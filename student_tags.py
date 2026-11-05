r"""
student_tags.py — Stage 3 step 2 / Stage 5 (claude/plan-nov2-2026.md): the first STUDENT tag model,
trained on the laptop GPU. No Modal GPU credit is used anywhere in here.

The student is a small head on the SigLIP2 picture embeddings that the Modal vision stage already
made (2 per second, every finished video). It looks at ±2 s around a moment and gives a score for
every tag in your vocabulary. It learns from:
  • the teacher's tags and confidences (tags_32b.json) for every moment it tagged, and
  • your 🎯 labels and ✏️ Compare-tags corrections (training_labels.json), weighted much higher
    (default ×5) and taking the place of the teacher's answer at the same moment.
The 🧪 test-set videos are kept out of training completely, so the student is scored on them the
same way as the teacher's test runs: F1 against your labels, overall and per tag.

Usage (from the activated .venv in the scene-labeling folder, or student_tags.bat):
    python student_tags.py fetch             download what's new from the Modal Volume (free —
                                             a file download, no GPU) + copy your latest labels
    python student_tags.py train             train, then score on the test set (writes
                                             student_report.md and _student/models/<name>/)
    python student_tags.py train --name v1b --label-weight 8 --epochs 40
    python student_tags.py score --name v1   score a model you trained before (no training)
    python student_tags.py all --name v3     fetch (incl. your Reviewer corrections) + train
    python student_tags.py export --name v4  every half second of every video → catalogue/student_scores

Everything is written to _student next to this script (genuine local disk, not Google Drive):
    _student/videos/<video_id>/  embeddings.npz, tags_32b.json, person_curve.npy
    _student/library/            training_labels.json, tag_vocabulary.csv (copied at fetch time)
    _student/testset.json, test_runs.json, batches.json
    _student/models/<name>/      head.pt + meta.json (tags, thresholds, settings, scores)

Before `fetch`, press ☁️ Save edits to Drive in the app so your newest labels are on Drive — fetch
prints how many labels it found and the newest one's date so you can check.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_logic as TL  # noqa: E402

WORK = HERE / "_student"
VOLUME = "media-train"
TEACHER_FILE = "tags_32b.json"            # 32b and 32b-fp8 (+ LoRA) all write this one
VIDEO_FILES = ("embeddings.npz", TEACHER_FILE, "person_curve.npy")

STEP = 0.5                                # the embeddings are 2 per second
HALF = 4                                  # ±4 steps = ±2 s around the moment → 9 embeddings
LABEL_WEIGHT = 5.0                        # one of your labels counts as this many teacher answers
COPY_WEIGHT = 0.5                         # a teacher answer copied from a near-identical moment
SCENE_WEIGHT = 2.0                        # a moment inside a scene you corrected in the Reviewer
SCENE_STEP = 2.0                          # extra moments taken every this many seconds in those scenes
SCENE_EDGE = 1.0                          # …but not this close to the scene's start / end (cuts)
# A tag's Reviewer answers are only used where they match your moment labels inside the same scenes
# (scene_rules): need this many of your labels in corrected scenes, and this much agreement.
SCENE_MIN_CHECKS = 15
SCENE_ABSENT_AGREE = 0.92                 # left off the scene → absent everywhere
SCENE_WIDE_AGREE = 0.85                   # on the scene → present everywhere
SCENE_TEACHER_AGREE = 0.50                # on the scene → present where the teacher saw it too
VAL_SHARE = 0.12                          # training videos kept aside to pick thresholds / stop early
SEED = 7


# ---------------------------------------------------------------------------
# fetch — Modal Volume → _student (a download, no container, no GPU)
# ---------------------------------------------------------------------------

def _download(vol, remote: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as fh:
        vol.read_file_into_fileobj(remote, fh)
    tmp.replace(dest)


def _fresh(dest: Path, entry) -> bool:
    try:
        st = dest.stat()
    except OSError:
        return False
    return st.st_size == entry.size and st.st_mtime >= entry.mtime - 2


def fetch() -> int:
    try:
        import modal
    except ImportError:
        print("modal isn't installed in this venv:  pip install modal  then  modal setup")
        return 1
    vol = modal.Volume.from_name(VOLUME)
    print("Listing the Modal Volume (this is free — just a file listing)…", flush=True)
    entries = {e.path.lstrip("/"): e for e in vol.listdir("trial", recursive=True)}
    singles = {"trial/testset/testset.json": WORK / "testset.json",
               "trial/testset/test_runs.json": WORK / "test_runs.json"}
    for e in vol.listdir("pack"):
        if e.path.lstrip("/") == "pack/batches.json":
            entries["pack/batches.json"] = e
            singles["pack/batches.json"] = WORK / "batches.json"
    todo = []
    videos = set()
    for path, e in entries.items():
        parts = path.split("/")
        if len(parts) == 3 and parts[2] in VIDEO_FILES and parts[1] not in ("testset", "export", "checks", "lora"):
            videos.add(parts[1])
            dest = WORK / "videos" / parts[1] / parts[2]
            if not _fresh(dest, e):
                todo.append((path, dest, e.size))
        elif path in singles and not _fresh(singles[path], e):
            todo.append((path, singles[path], e.size))
    with_tags = {v for v in videos if f"trial/{v}/{TEACHER_FILE}" in entries
                 and f"trial/{v}/embeddings.npz" in entries}
    mb = sum(s for _, _, s in todo) / 1e6
    print(f"{len(videos)} videos on Modal, {len(with_tags)} with embeddings + teacher tags. "
          f"{len(todo)} file(s) new or changed ({mb:.0f} MB) to download.", flush=True)
    t0 = time.time()
    for i, (remote, dest, size) in enumerate(sorted(todo), 1):
        _download(vol, remote, dest)
        if i % 25 == 0 or i == len(todo):
            print(f"   {i} of {len(todo)} files ({time.time() - t0:.0f} s)", flush=True)

    # Your labels and vocabulary: the newest copy in the Drive catalogue folder (like the pack does).
    try:
        from pack_for_modal import newest_copy
    except Exception as e:      # config.py missing etc.
        print(f"Couldn't read the catalogue folder ({e}) — labels not copied.")
        return 1
    lib = WORK / "library"
    lib.mkdir(parents=True, exist_ok=True)
    for name in ("training_labels.json", "tag_vocabulary.csv"):
        src = newest_copy(name)
        if src is None:
            print(f"   ⚠️ {name} not found in the catalogue folder.")
            continue
        (lib / name).write_bytes(src.read_bytes())
    n_sc = write_reviewer_scenes()
    if n_sc is not None:
        print(f"Your Reviewer corrections: {n_sc['scenes']} corrected scenes in {n_sc['videos']} videos.")
    items = _labels()
    newest = max((i.get("labelled_at") or "" for i in items), default="")
    print(f"Your labels: {len(items)} tag answers in {len({i['video_id'] for i in items})} videos, "
          f"newest saved {newest.replace('T', ' ') or '—'}.")
    print("   (Not seeing today's corrections? Press ☁️ Save edits to Drive in the app, wait for Drive "
          "to sync to this laptop, then run fetch again.)")
    return 0


def write_reviewer_scenes():
    """_student/library/reviewer_scenes.json from the catalogues in your Drive folder: every scene
    you've corrected / confirmed in the Reviewer (not excluded) as [start s, end s, [tags]]."""
    try:
        from pack_for_modal import load_catalogues, tc_to_seconds
    except Exception as e:
        print(f"Couldn't read the catalogues ({e}) — Reviewer corrections not used.")
        return None
    cats = load_catalogues(progress=True)
    out, n = {}, 0
    for v, cat in cats.items():
        rows = []
        for sc in cat.get("scenes") or []:
            if sc.get("corrected_tags") is None or sc.get("excluded"):
                continue
            try:
                a, b = tc_to_seconds(sc["start_tc"]), tc_to_seconds(sc["end_tc"])
            except (KeyError, ValueError):
                continue
            if b > a:
                rows.append([round(a, 3), round(b, 3), list(sc["corrected_tags"] or [])])
        if rows:
            out[v] = rows
            n += len(rows)
    p = WORK / "library" / "reviewer_scenes.json"
    p.write_text(json.dumps({"built": time.strftime("%Y-%m-%d %H:%M"), "videos": out}), encoding="utf-8")
    return {"scenes": n, "videos": len(out)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _reviewer_scenes() -> dict:
    return (_read_json(WORK / "library" / "reviewer_scenes.json", {}) or {}).get("videos") or {}


def scene_agreement(vocab: list, scenes: dict, labels: list, test: set) -> dict:
    """How well your Reviewer scene answers match your moment labels, per tag: every 🎯 / ✏️ label
    (outside the test set) that falls inside a scene you corrected is checked against that scene.
    {tag lower: {"absent_n", "absent_ok", "present_n", "present_ok"}} — absent_ok = the scene left
    the tag off and your label agrees; present_ok = the scene has it and so does your label."""
    names = {r["tag"].lower() for r in vocab}
    out = {t: {"absent_n": 0, "absent_ok": 0, "present_n": 0, "present_ok": 0} for t in names}
    for i in labels:
        v = i.get("video_id")
        if v in test or v not in scenes:
            continue
        sc = _scene_at(scenes[v], float(i["t"]), edge=0.0)
        if sc is None:
            continue
        on = {str(x).lower() for x in sc}
        mine = {str(x).lower() for x in i.get("tags") or []}
        for t in names:
            k = "present" if t in on else "absent"
            out[t][f"{k}_n"] += 1
            out[t][f"{k}_ok"] += (t in mine) == (t in on)
    return out


def scene_rules(vocab: list, agreement: dict, min_n: int = SCENE_MIN_CHECKS) -> dict:
    """Which Reviewer scene answers to use, decided per tag by how often they match your labels:
      "absent"       — left off a scene → absent at every moment (when ≥ 92% of your labels agree);
      "wide"         — on a scene → present at every moment (≥ 85% agree; never a motion tag);
      "with_teacher" — on a scene → present only where the teacher saw it too (≥ 50% agree).
    A tag with fewer than min_n of your labels inside corrected scenes on that side isn't used."""
    motion = {r["tag"].lower() for r in vocab if r.get("kind") == "motion"}
    rules = {"absent": set(), "wide": set(), "with_teacher": set()}
    for t, a in agreement.items():
        if a["absent_n"] >= min_n and a["absent_ok"] / a["absent_n"] >= SCENE_ABSENT_AGREE:
            rules["absent"].add(t)
        if a["present_n"] >= min_n:
            share = a["present_ok"] / a["present_n"]
            if share >= SCENE_WIDE_AGREE and t not in motion:
                rules["wide"].add(t)
            elif share >= SCENE_TEACHER_AGREE:
                rules["with_teacher"].add(t)
    return rules


def apply_scene(y: np.ndarray, w: np.ndarray, scene_tags, tag_ix: dict, rules: dict, teacher_saw: set,
                weight: float = SCENE_WEIGHT) -> int:
    """Put a corrected scene's answer onto one moment (in place), following scene_rules(). Returns how
    many tags were set."""
    on = {str(t).lower() for t in scene_tags or []}
    n = 0
    for low, j in tag_ix.items():
        if low not in on:
            if low in rules["absent"]:
                y[j], w[j] = 0.0, weight
                n += 1
        elif low in rules["wide"] or (low in rules["with_teacher"] and low in teacher_saw):
            y[j], w[j] = 1.0, weight
            n += 1
    return n


def _scene_at(rows: list, t: float, edge: float = SCENE_EDGE):
    for a, b, tags in rows:
        if a + edge <= t < b - edge:
            return tags
    return None

def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _labels() -> list:
    data = _read_json(WORK / "library" / "training_labels.json", {}) or {}
    return [i for i in data.get("items") or [] if isinstance(i, dict) and i.get("type") == "tags"
            and not i.get("skipped") and i.get("t") is not None and i.get("video_id")]


def _vocab() -> list:
    p = WORK / "library" / "tag_vocabulary.csv"
    try:
        return TL.parse_vocabulary(p.read_text(encoding="utf-8-sig"))
    except OSError:
        return []


def test_videos() -> list:
    return [x["video_id"] for x in (_read_json(WORK / "testset.json", {}) or {}).get("videos") or []]


def window_index(times: np.ndarray, t: float, half: int = HALF, step: float = STEP) -> np.ndarray:
    """Indices of the embeddings nearest to t-2 s … t+2 s (clamped at the video's ends)."""
    want = t + step * np.arange(-half, half + 1)
    i = np.searchsorted(times, want)
    i = np.clip(i, 1, len(times) - 1)
    left, right = times[i - 1], times[i]
    i = np.where(np.abs(want - left) <= np.abs(right - want), i - 1, i)
    return np.clip(i, 0, len(times) - 1)


def window_index_many(times: np.ndarray, ts: np.ndarray, half: int = HALF, step: float = STEP) -> np.ndarray:
    """window_index for many moments at once: (len(ts), 2*half+1)."""
    want = np.asarray(ts, float)[:, None] + step * np.arange(-half, half + 1)
    i = np.clip(np.searchsorted(times, want), 1, len(times) - 1)
    left, right = times[i - 1], times[i]
    i = np.where(np.abs(want - left) <= np.abs(right - want), i - 1, i)
    return np.clip(i, 0, len(times) - 1)


def teacher_target(r: dict, tag_ix: dict, n: int) -> np.ndarray:
    """Soft target from one teacher answer: its confidence (0–1) per tag, but a tag in its final
    tags is at least 0.55 and one it listed but didn't keep (below threshold, rule break) at most
    0.45 — so the student learns the vocabulary rules' outcome, not just the raw confidence."""
    y = np.zeros(n, np.float32)
    conf = {str(k).lower(): v for k, v in (r.get("conf") or {}).items()}
    final = {str(t).lower() for t in r.get("tags") or []}
    for low, j in tag_ix.items():
        c = conf.get(low)
        c = (100.0 if low in final else 0.0) if c is None else float(c)
        c = max(0.0, min(1.0, c / 100.0))
        y[j] = max(c, 0.55) if low in final else min(c, 0.45)
    return y


def hard_target(tags, tag_ix: dict, n: int) -> np.ndarray:
    y = np.zeros(n, np.float32)
    for t in tags or []:
        j = tag_ix.get(str(t).lower())
        if j is not None:
            y[j] = 1.0
    return y


def build_dataset(vocab: list, labels: list, test: set, label_weight: float = LABEL_WEIGHT,
                  root: Path = None, scenes: dict = None, scene_weight: float = SCENE_WEIGHT) -> dict:
    """Every usable moment as an index window into one big embedding array, with a weight PER TAG
    (0 = this moment says nothing about that tag):
      • teacher moments: its soft answer, weight 1 (0.5 if copied); tags with the teacher switched off 0;
      • inside a scene you corrected in the Reviewer: that scene's answer for the trusted tags (×2),
        plus extra moments every 2 s that carry only the scene's answer;
      • your 🎯 / ✏️ labels: every tag, ×label_weight (they replace anything at the same moment).
    Returns {"E" (frames × D, float16), "idx" (samples × 9), "y", "w" (samples × tags), "video",
    "is_label", "gold_id", "t", "videos", "skipped", "tags", "counts"}. Test-set videos give label
    rows only (for scoring) and are flagged so training never sees them."""
    root = root or WORK / "videos"
    scenes = _reviewer_scenes() if scenes is None else scenes
    tags = [r["tag"] for r in vocab]
    tag_ix = {t.lower(): j for j, t in enumerate(tags)}
    n = len(tags)
    teacher_on = np.array([1.0 if r.get("teacher", True) else 0.0 for r in vocab], np.float32)
    agreement = scene_agreement(vocab, scenes, labels, test)
    rules = scene_rules(vocab, agreement)
    trusted = rules["absent"] | rules["wide"] | rules["with_teacher"]
    by_video = {}
    for i in labels:
        by_video.setdefault(i["video_id"], []).append(i)
    E, idx, ys, ws, vids, is_lab, gids, ts = [], [], [], [], [], [], [], []
    counts = {"teacher": 0, "teacher_in_scenes": 0, "scene_extra": 0, "labels": 0}
    offset, skipped, used = 0, [], []

    def add(v, t, y, w, lab=False, gid=""):
        idx.append(window_index(et, t) + offset)
        ys.append(y); ws.append(w); vids.append(v); is_lab.append(lab); gids.append(gid); ts.append(t)

    for vdir in sorted(p for p in root.iterdir() if p.is_dir()):
        v = vdir.name
        try:
            z = np.load(vdir / "embeddings.npz")
            et = np.asarray(z["frame_times"], float)
            emb = np.asarray(z["frame_embs"], np.float32)
        except (OSError, KeyError, ValueError):
            skipped.append(v)
            continue
        if not et.size or emb.ndim != 2 or len(emb) != et.size:
            skipped.append(v)
            continue
        order = np.argsort(et)
        et, emb = et[order], emb[order]
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
        mine = by_video.get(v, [])
        lab_t = {round(float(i["t"]), 2) for i in mine}
        lab_id = {i["id"] for i in mine if i.get("id")}
        start_rows = len(idx)
        if v not in test:
            vscenes = scenes.get(v) or [] if trusted else []
            taken = [float(i["t"]) for i in mine]
            tj = _read_json(vdir / TEACHER_FILE, {}) or {}
            for r in (tj.get("results") or {}).values():
                if r.get("t") is None or r.get("ok") is False:
                    continue
                if (r.get("gold_id") and r["gold_id"] in lab_id) or round(float(r["t"]), 2) in lab_t:
                    continue                                   # your label replaces it
                t = float(r["t"])
                y = teacher_target(r, tag_ix, n)
                w = teacher_on * (COPY_WEIGHT if r.get("copied_from") else 1.0)
                sc = _scene_at(vscenes, t) if vscenes else None
                if sc is not None:
                    saw = {k for k, j in tag_ix.items() if y[j] >= 0.5}
                    if apply_scene(y, w, sc, tag_ix, rules, saw, scene_weight):
                        counts["teacher_in_scenes"] += 1
                add(v, t, y, w)
                taken.append(t)
                counts["teacher"] += 1
            # extra moments inside corrected scenes, away from the teacher's and your moments
            taken = np.sort(np.asarray(taken, float))
            for a, b, sc in vscenes:
                for t in np.arange(a + SCENE_EDGE, b - SCENE_EDGE, SCENE_STEP):
                    if taken.size and np.min(np.abs(taken - t)) < SCENE_STEP / 2:
                        continue
                    y, w = np.zeros(n, np.float32), np.zeros(n, np.float32)
                    if apply_scene(y, w, sc, tag_ix, rules, set(), scene_weight):
                        add(v, float(t), y, w)
                        counts["scene_extra"] += 1
        for i in mine:
            add(v, float(i["t"]), hard_target(i.get("tags"), tag_ix, n), np.full(n, label_weight, np.float32),
                True, i.get("id") or "")
            counts["labels"] += 1
        if len(idx) == start_rows:
            continue
        E.append(emb.astype(np.float16))
        offset += len(emb)
        used.append(v)
    if not E:
        raise SystemExit(f"No usable videos in {root} — run `python student_tags.py fetch` first.")
    return {"E": np.concatenate(E), "idx": np.stack(idx).astype(np.int64), "y": np.stack(ys),
            "w": np.stack(ws).astype(np.float32), "video": np.asarray(vids), "is_label": np.asarray(is_lab),
            "gold_id": np.asarray(gids), "t": np.asarray(ts), "videos": used, "skipped": skipped,
            "tags": tags, "counts": counts, "scene_tags_used": sorted(trusted),
            "scene_rules": {k: sorted(v) for k, v in rules.items()}, "scene_agreement": agreement}


def split_videos(videos: list, test: set, share: float = VAL_SHARE, seed: int = SEED) -> tuple:
    """(train videos, validation videos) — the test set is in neither."""
    pool = sorted(v for v in videos if v not in test)
    rng = random.Random(seed)
    rng.shuffle(pool)
    k = max(1, int(round(len(pool) * share))) if len(pool) > 4 else 0
    return sorted(pool[k:]), sorted(pool[:k])


def pick_thresholds(prob: np.ndarray, truth: np.ndarray, min_pos: int = 10, gain: float = 0.01,
                    mask: np.ndarray = None) -> np.ndarray:
    """Per tag, the score cut-off (0.25–0.85) with the best F1 on the held-out moments; 0.5 for a
    tag with fewer than min_pos positives there (too few to choose from) or where no cut-off beats
    0.5 by at least `gain` F1 (a tiny edge on a few moments is chance, not a better cut-off)."""
    grid = np.round(np.arange(0.25, 0.851, 0.05), 2)
    out = np.full(prob.shape[1], 0.5, np.float32)
    for j in range(prob.shape[1]):
        keep = slice(None) if mask is None else mask[:, j] > 0
        pj, g = prob[keep, j], truth[keep, j] >= 0.5
        if g.sum() < min_pos:
            continue
        def f1_at(th):
            p = pj >= th
            return 2 * float((p & g).sum()) / max(float(p.sum() + g.sum()), 1.0)
        base = f1_at(0.5)
        best, bt = -1.0, 0.5
        for th in grid:
            p = pj >= th
            tp = float((p & g).sum())
            f1 = 2 * tp / max(float(p.sum() + g.sum()), 1.0)
            if f1 > best + 1e-9:
                best, bt = f1, th
        out[j] = bt if best >= base + gain else 0.5
    return out


def to_tags(prob_row: np.ndarray, tags: list, thresholds: np.ndarray, vocab: list = None) -> list:
    """Tags from one moment's scores: plain cut-offs, or (vocab given) through the same vocabulary
    rules as the teacher (TL.finalize: never-with, needs-a-tag categories, always / only-with), with
    each tag's threshold set to the student's own."""
    if vocab is None:
        return [t for t, p, th in zip(tags, prob_row, thresholds) if p >= th]
    conf = {t: float(p) * 100.0 for t, p in zip(tags, prob_row)}
    return TL.finalize(conf, vocab)["tags"]


def student_vocab(vocab: list, tags: list, thresholds: np.ndarray) -> list:
    th = {t.lower(): int(round(float(x) * 100)) for t, x in zip(tags, thresholds)}
    return [{**r, "threshold": th.get(r["tag"].lower(), r.get("threshold"))} for r in vocab]


# ---------------------------------------------------------------------------
# The model (torch only from here on)
# ---------------------------------------------------------------------------

def make_model(dim: int, n_tags: int, steps: int = 2 * HALF + 1, hidden: int = 512, drop: float = 0.2):
    import torch
    from torch import nn

    class StudentHead(nn.Module):
        """LayerNorm → project → + time position → one temporal conv (motion across the 4 s) →
        attention pooling + the centre frame → MLP → one score per tag."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.proj = nn.Linear(dim, hidden)
            self.pos = nn.Parameter(torch.zeros(steps, hidden))
            self.mix = nn.Conv1d(hidden, hidden, 3, padding=1)
            self.att = nn.Linear(hidden, 1)
            self.out = nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(2 * hidden, hidden), nn.GELU(),
                                     nn.Dropout(drop), nn.Linear(hidden, n_tags))
            self.centre = steps // 2

        def forward(self, x):                       # x: batch × steps × dim
            h = self.proj(self.norm(x)) + self.pos
            h = h + torch.nn.functional.gelu(self.mix(h.transpose(1, 2))).transpose(1, 2)
            a = torch.softmax(self.att(h), dim=1)
            pooled = (a * h).sum(dim=1)
            return self.out(torch.cat([pooled, h[:, self.centre]], dim=1))

    return StudentHead()


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, E_t, idx: np.ndarray, batch: int = 2048) -> np.ndarray:
    import torch
    model.eval()
    out = []
    dev = E_t.device
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            ii = torch.as_tensor(idx[s:s + batch], device=dev)
            x = E_t[ii].float()
            out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.out[-1].out_features), np.float32)


def train_model(ds: dict, train_rows: np.ndarray, val_rows: np.ndarray, epochs: int = 30,
                batch: int = 512, lr: float = 1e-3, seed: int = SEED, patience: int = 5, log=print):
    """BCE on soft targets weighted per tag (a weight of 0 = no answer for that tag), rarer tags weighted up (√ negatives/positives, at most ×8),
    ±1-step time jitter, early stop on validation loss. Returns (model, E on device, history)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    E_t = torch.as_tensor(ds["E"], device=dev)               # float16 on the GPU (about 1–2 GB)
    y_t = torch.as_tensor(ds["y"], device=dev)
    w_t = torch.as_tensor(ds["w"], device=dev)
    idx = ds["idx"]
    seen = ds["w"][train_rows] > 0
    pos = ((ds["y"][train_rows] >= 0.5) & seen).sum(0).astype(np.float64)
    neg = ((ds["y"][train_rows] < 0.5) & seen).sum(0).astype(np.float64)
    pw = torch.as_tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1.0)), 1.0, 8.0), dtype=torch.float32,
                         device=dev)
    model = make_model(ds["E"].shape[1], ds["y"].shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps_per = max(1, int(np.ceil(len(train_rows) / batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    # Per-sample bounds of its own video in E, so jitter never crosses into the next video.
    lo = np.empty(len(idx), np.int64)
    hi = np.empty(len(idx), np.int64)
    starts = {}
    for k, v in enumerate(ds["video"]):
        starts.setdefault(v, [idx[k].min(), idx[k].max()])
        starts[v][0] = min(starts[v][0], idx[k].min())
        starts[v][1] = max(starts[v][1], idx[k].max())
    for k, v in enumerate(ds["video"]):
        lo[k], hi[k] = starts[v]

    def val_loss():
        if not len(val_rows):
            return float("nan")
        model.eval()
        tot, wsum = 0.0, 0.0
        with torch.no_grad():
            for s in range(0, len(val_rows), 2048):
                r = val_rows[s:s + 2048]
                x = E_t[torch.as_tensor(idx[r], device=dev)].float()
                l = lossf(model(x), y_t[r]) * w_t[r]
                tot += float(l.sum())
                wsum += float(w_t[r].sum())
        return tot / max(wsum, 1e-9)

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(train_rows)
        t0, run = time.time(), 0.0
        for s in range(0, len(perm), batch):
            r = perm[s:s + batch]
            shift = np.random.randint(-1, 2, size=len(r))[:, None]
            ii = np.clip(idx[r] + shift, lo[r][:, None], hi[r][:, None])
            x = E_t[torch.as_tensor(ii, device=dev)].float()
            x = x + 0.01 * torch.randn_like(x)
            l = (lossf(model(x), y_t[r]) * w_t[r]).sum() / w_t[r].sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
            run += l.item() * len(r)
        vl = val_loss()
        hist.append({"epoch": ep, "train_loss": round(run / len(perm), 4), "val_loss": round(vl, 4)})
        log(f"   epoch {ep:>2}: train {run / len(perm):.4f} · validation {vl:.4f} ({time.time() - t0:.1f} s)")
        if vl < best - 1e-4 or np.isnan(vl):
            best, bad = vl, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"   stopped early — validation hasn't improved for {patience} epochs")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, E_t, hist


# ---------------------------------------------------------------------------
# Scoring on the 🧪 test set
# ---------------------------------------------------------------------------

def teacher_runs() -> list:
    return (_read_json(WORK / "test_runs.json", {}) or {}).get("runs") or []


def pick_reference(runs: list, want: str = "") -> dict | None:
    """The teacher test run to compare with: --vs <id or label>, else the newest try2 run that
    isn't a derived copy, else the newest run."""
    if want:
        return next((r for r in reversed(runs) if want in (r.get("id"), r.get("label"))), None)
    try2 = [r for r in runs if r.get("lora") == "try2" or "try2" in str(r.get("label") or "")]
    plain = [r for r in try2 if not r.get("derived_from")]
    return (plain or try2 or runs or [None])[-1]


def score_report(ds: dict, prob: np.ndarray, rows: np.ndarray, thresholds: np.ndarray, vocab: list,
                 gold: list, ref: dict | None, name: str) -> tuple:
    """(markdown, scores) for the student on your labelled test-set frames — plain cut-offs and
    through your vocabulary rules — next to the reference teacher run on the same frames."""
    tags = ds["tags"]
    svocab = student_vocab(vocab, tags, thresholds)
    teacher_tags = [r["tag"] for r in TL.teacher_vocab(vocab)]
    gold_by = {(i["video_id"], round(float(i["t"]), 2)): i for i in gold}
    ref_pairs_by_key = {}
    if ref:
        frames = ref.get("frames") or []
        by_id = {f["g"]: f for f in frames if f.get("g")}
        by_t = {(f["v"], round(float(f["t"]), 2)): f for f in frames if f.get("t") is not None}
        for key, i in gold_by.items():
            f = by_id.get(i.get("id")) or by_t.get(key)
            if f is not None:
                ref_pairs_by_key[key] = (f.get("tags") or [], i.get("tags") or [])
    plain, ruled, ref_pairs, every = [], [], [], []
    for k, r in enumerate(rows):
        key = (ds["video"][r], round(float(ds["t"][r]), 2))
        i = gold_by.get(key)
        if i is None:
            continue
        g = i.get("tags") or []
        every.append((to_tags(prob[k], tags, thresholds, svocab), g))
        if ref and key not in ref_pairs_by_key:
            continue                        # compare on exactly the frames the teacher run has
        plain.append((to_tags(prob[k], tags, thresholds), g))
        ruled.append((to_tags(prob[k], tags, thresholds, svocab), g))
        if ref:
            ref_pairs.append(ref_pairs_by_key[key])
    res = {"plain": TL.score_tags(plain, tags), "rules": TL.score_tags(ruled, tags),
           "plain_teacher_tags": TL.score_tags(plain, teacher_tags),
           "rules_teacher_tags": TL.score_tags(ruled, teacher_tags), "rules_all_frames": TL.score_tags(every, tags)}
    if ref:
        res["teacher"] = TL.score_tags(ref_pairs, tags)
        res["teacher_teacher_tags"] = TL.score_tags(ref_pairs, teacher_tags)

    def f(x):
        return "–" if x is None else f"{x:.2f}"

    def row(label, sc):
        o = sc["overall"]
        return (f"| {label} | {sc['n']} | {f(o['f1'])} | {f(o['precision'])} | {f(o['recall'])} | "
                f"{o['fp']} | {o['fn']} |")

    md = [f"# Student tag model `{name}` — 🧪 test set", "",
          f"Trained {time.strftime('%Y-%m-%d %H:%M')} · scored on your labelled test-set frames"
          + (f" that the teacher run **{ref['id']}** also tagged" if ref else "") + ".", "",
          "| Model | Your frames | F1 | Right (precision) | Found (recall) | Wrong | Missed |",
          "|---|---|---|---|---|---|---|"]
    if ref:
        md.append(row(f"Teacher · {ref.get('label') or ref['id']}", res["teacher"]))
    md += [row("Student · plain cut-offs", res["plain"]), row("Student · with your vocabulary rules", res["rules"])]
    best = "rules" if (res["rules"]["overall"]["f1"] or 0) >= (res["plain"]["overall"]["f1"] or 0) else "plain"
    if ref and res["teacher"]["overall"]["f1"] is not None:
        gap = (res[best]["overall"]["f1"] or 0) - res["teacher"]["overall"]["f1"]
        verdict = ("✅ level with or ahead of the teacher" if gap >= 0 else "✅ within the 0.03 goal"
                   if gap >= -0.03 else "⚠️ more than 0.03 behind the teacher")
        md += ["", f"**Best student vs teacher: {gap:+.2f} F1** ({best}) — {verdict}."]
    if ref and res["rules_all_frames"]["n"] > res["rules"]["n"]:
        md += ["", f"On all {res['rules_all_frames']['n']} of your labelled test frames (including ones labelled "
                   f"after that teacher run): student F1 {f(res['rules_all_frames']['overall']['f1'])} (with rules)."]
    n_teacher_only = len(tags) - len(teacher_tags)
    if n_teacher_only:
        md += ["", f"{n_teacher_only} tag(s) have the teacher switched off (learned from your labels only). "
                   "On the teacher's tags alone the student scores "
                   f"{f(res[best + '_teacher_tags']['overall']['f1'])}"
                   + (f" vs the teacher's {f(res['teacher_teacher_tags']['overall']['f1'])}." if ref else ".")]
    # Per tag: biggest differences first
    per = []
    for t in tags:
        s = res[best]["per_tag"][t]
        tr = res["teacher"]["per_tag"][t] if ref else None
        n_gold = s["tp"] + s["fn"] if "tp" in s else None
        per.append((t, s, tr, n_gold))
    md += ["", f"## Per tag (student {best}" + (" vs teacher" if ref else "") + ")", "",
           "Errors = wrong + missed. Sorted by how many more errors the student makes than the teacher.", "",
           "| Tag | In your labels | Student F1 | Student errors | " + ("Teacher F1 | Teacher errors | Difference |"
                                                                     if ref else "Threshold |"),
           "|---|---|---|---|---|" + ("---|---|" if ref else "")]
    th = {t: float(x) for t, x in zip(tags, thresholds)}

    def errs(s):
        return s["fp"] + s["fn"]
    per.sort(key=lambda x: (-(errs(x[1]) - errs(x[2])) if ref else -errs(x[1]), x[0].lower()))
    for t, s, tr, ng in per:
        if not errs(s) and not (tr and errs(tr)) and not ng:
            continue
        ng_txt = "" if ng is None else str(ng)
        if ref:
            d = errs(s) - errs(tr)
            md.append(f"| {t} | {ng_txt} | {f(s['f1'])} | {errs(s)} | {f(tr['f1'])} | {errs(tr)} | "
                      f"{'+' if d > 0 else ''}{d} |")
        else:
            md.append(f"| {t} | {ng_txt} | {f(s['f1'])} | {errs(s)} | {th[t]:.2f} |")
    n = res[best]["n"]
    md += ["", f"On {n} frames, F1 moves by about ±{max(0.02, round(0.5 / max(n, 1) ** 0.5, 2)):.2f} from "
               "chance alone — trust bigger gaps, and per-tag differences of several frames."]
    return "\n".join(md), res


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _prepare(label_weight: float):
    vocab, labels, test = _vocab(), _labels(), set(test_videos())
    if not vocab:
        raise SystemExit("No tag_vocabulary.csv in _student/library — run `python student_tags.py fetch`.")
    if not test:
        print("⚠️ No test set downloaded (testset.json) — the model will train but can't be scored.")
    ds = build_dataset(vocab, labels, test, label_weight)
    train_v, val_v = split_videos(ds["videos"], test)
    vids = ds["video"]
    train_rows = np.flatnonzero(np.isin(vids, train_v))
    val_rows = np.flatnonzero(np.isin(vids, val_v))
    test_rows = np.flatnonzero(np.isin(vids, list(test)) & ds["is_label"])
    learn = np.flatnonzero(~np.isin(vids, list(test)))
    print(f"Data: {len(ds['videos'])} videos ({len(train_v) + len(val_v)} to learn from, "
          f"{len(set(vids[test_rows]))} test) · {len(ds['tags'])} tags · "
          f"{int((~ds['is_label'][learn]).sum())} teacher moments + "
          f"{int(ds['is_label'][learn].sum())} of your labels to learn from · "
          f"{len(test_rows)} labelled test frames · {ds['E'].shape[0]:,} embeddings "
          f"({ds['E'].nbytes / 1e9:.1f} GB)")
    c = ds["counts"]
    print(f"   Reviewer corrections: {c['teacher_in_scenes']} teacher moments given your scene's answer + "
          f"{c['scene_extra']} extra moments from your corrected scenes · tags used from them: "
          f"{len(ds['scene_tags_used'])} of {len(ds['tags'])}")
    if ds["skipped"]:
        print(f"   skipped {len(ds['skipped'])} video(s) with unreadable embeddings: {', '.join(ds['skipped'][:5])}")
    return vocab, labels, test, ds, train_rows, val_rows, test_rows


def fold_videos(videos: list, test: set, k: int, seed: int = SEED) -> list:
    """The training videos (test set left out) shuffled into k groups of about the same size."""
    pool = sorted(v for v in videos if v not in test)
    random.Random(seed).shuffle(pool)
    return [sorted(pool[i::k]) for i in range(k)]


def predict_many(models: list, E_t, idx: np.ndarray) -> np.ndarray:
    """The models' scores averaged (the k fold models vote together)."""
    return np.mean([predict(m, E_t, idx) for m in models], axis=0)


def train(name: str, epochs: int, label_weight: float, vs: str, folds: int = 5) -> int:
    """k models, each trained with a different fifth of the training videos held out (it stops early on
    that fifth). Their held-out answers cover every training video, so each tag's cut-off is picked on
    all of them, not one small split; on the test set the k models' scores are averaged."""
    import torch
    vocab, labels, test, ds, _, _, test_rows = _prepare(label_weight)
    dev = _device()
    print(f"Training {folds} models on "
          f"{torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'the CPU (no CUDA GPU found)'}…", flush=True)
    t0 = time.time()
    groups = fold_videos(ds["videos"], test, folds)
    vids = ds["video"]
    oof_rows, oof_prob, models, hists, E_t = [], [], [], [], None
    for f, val_v in enumerate(groups, 1):
        val_rows = np.flatnonzero(np.isin(vids, val_v))
        train_rows = np.flatnonzero(np.isin(vids, [v for g in groups if g is not val_v for v in g]))
        print(f" model {f} of {folds}: {len(val_v)} videos held out", flush=True)
        model, E_t, hist = train_model(ds, train_rows, val_rows, epochs=epochs,
                                       log=lambda *_: None)
        best = min(hist, key=lambda h: h["val_loss"])
        print(f"   best validation {best['val_loss']:.4f} at epoch {best['epoch']}", flush=True)
        oof_rows.append(val_rows)
        oof_prob.append(predict(model, E_t, ds["idx"][val_rows]))
        models.append(model)
        hists.append(hist)
        if dev.type == "cuda" and f < folds:
            del E_t
            torch.cuda.empty_cache()
    rows = np.concatenate(oof_rows)
    thresholds = pick_thresholds(np.concatenate(oof_prob), ds["y"][rows], mask=ds["w"][rows])
    minutes = (time.time() - t0) / 60
    out = WORK / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("head*.pt"):
        old.unlink()
    for f, m in enumerate(models, 1):
        torch.save(m.state_dict(), out / f"head_{f}.pt")
    meta = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "tags": ds["tags"],
            "thresholds": [float(x) for x in thresholds], "dim": int(ds["E"].shape[1]), "half": HALF,
            "step": STEP, "label_weight": label_weight, "folds": folds, "counts": ds["counts"],
            "scene_tags_used": ds["scene_tags_used"], "scene_rules": ds["scene_rules"],
            "epochs_run": [len(h) for h in hists], "history": hists,
            "train_minutes": round(minutes, 2), "videos": len(ds["videos"]), "test_videos": sorted(test),
            "labels_newest": max((i.get("labelled_at") or "" for i in labels), default="")}
    print(f"Trained in {minutes:.1f} min.", flush=True)
    rc = _score(models, E_t, ds, test_rows, thresholds, vocab, labels, vs, name, meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Model saved → {out}")
    return rc


def _score(models, E_t, ds, test_rows, thresholds, vocab, labels, vs, name, meta) -> int:
    if not len(test_rows):
        print("No labelled test-set frames to score on.")
        return 0
    prob = predict_many(models, E_t, ds["idx"][test_rows])
    runs = teacher_runs()
    ref = pick_reference(runs, vs)
    if vs and ref is None:
        print(f"⚠️ No teacher test run called '{vs}' — scoring the student alone.")
    md, res = score_report(ds, prob, test_rows, thresholds, vocab, labels, ref, name)
    meta["scores"] = {k: v["overall"] for k, v in res.items()}
    meta["reference_run"] = ref["id"] if ref else None
    md += "\n\n" + scene_rules_markdown(ds)
    report = HERE / "student_report.md"
    report.write_text(md, encoding="utf-8")
    (WORK / "models" / name / "student_report.md").write_text(md, encoding="utf-8")
    print("\n" + "\n".join(md.splitlines()[:14]) + f"\n\nFull report (per tag) → {report}")
    return 0


def scene_rules_markdown(ds: dict) -> str:
    """Which of your Reviewer scene answers were used, per tag, and how well they matched your labels."""
    ag, rules = ds.get("scene_agreement") or {}, ds.get("scene_rules") or {}
    c = ds.get("counts") or {}
    if not ag:
        return ""
    names = {t.lower(): t for t in ds["tags"]}

    def pct(ok, n):
        return f"{100 * ok / n:.0f}% of {n}" if n else "–"
    md = ["## Your Reviewer scene answers", "",
          f"{c.get('teacher_in_scenes', 0)} teacher moments took your scene's answer and {c.get('scene_extra', 0)} "
          "extra moments came from your corrected scenes. Each tag's scene answers are used only where they "
          "matched your moment labels inside the same scenes:", "",
          "| Tag | Left off the scene — your label agrees | On the scene — your label agrees | Used as |",
          "|---|---|---|---|"]
    for low, a in sorted(ag.items(), key=lambda kv: -(kv[1]["present_n"] + kv[1]["absent_n"])):
        if not a["present_n"] and not a["absent_n"]:
            continue
        use = [x for x, k in (("absent when left off", "absent"), ("present all scene", "wide"),
                              ("present where the teacher saw it", "with_teacher")) if low in rules.get(k, [])]
        md.append(f"| {names.get(low, low)} | {pct(a['absent_ok'], a['absent_n'])} | "
                  f"{pct(a['present_ok'], a['present_n'])} | {', '.join(use) or 'not used'} |")
    return "\n".join(md)


def load_models(name: str) -> tuple:
    """(models, meta) for a saved student, or (None, None)."""
    import torch
    out = WORK / "models" / name
    meta = _read_json(out / "meta.json")
    if not meta:
        return None, None
    models = []
    for p in sorted(out.glob("head*.pt")):
        m = make_model(meta["dim"], len(meta["tags"])).to(_device())
        m.load_state_dict(torch.load(p, map_location=_device()))
        models.append(m)
    return models, meta


def export(name: str, dest: Path = None, root: Path = None) -> int:
    """Every half second of every downloaded video → the student's score per tag (0–100), one small
    file per video in the catalogue's student_scores folder on Drive (or `dest`):
        student_scores/<video_id>.npz   times (s, float32), scores (moments × tags, uint8)
        student_scores/index.json       model, created, tags, cut-offs (0–100), test-set videos, videos
    A score at or above the tag's cut-off means "visible here" (the moment meaning — see the new-matcher
    design: scene tags are for choosing / filtering, these are for what's on screen)."""
    import torch
    models, meta = load_models(name)
    if not models:
        print(f"No model '{name}' in {WORK / 'models'} — train it first.")
        return 1
    if dest is None:
        try:
            from config import CATALOGUE_DIR
        except Exception as e:
            print(f"Couldn't read CATALOGUE_DIR from config.py ({e}).")
            return 1
        dest = Path(CATALOGUE_DIR) / "student_scores"
    root = root or WORK / "videos"
    dest.mkdir(parents=True, exist_ok=True)
    dev = _device()
    done, t0 = [], time.time()
    vdirs = sorted(p for p in root.iterdir() if p.is_dir())
    for k, vdir in enumerate(vdirs, 1):
        try:
            z = np.load(vdir / "embeddings.npz")
            et = np.asarray(z["frame_times"], float)
            emb = np.asarray(z["frame_embs"], np.float32)
        except (OSError, KeyError, ValueError):
            continue
        if not et.size or emb.ndim != 2 or len(emb) != et.size or emb.shape[1] != meta["dim"]:
            continue
        order = np.argsort(et)
        et, emb = et[order], emb[order]
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
        E_t = torch.as_tensor(emb.astype(np.float16), device=dev)
        prob = predict_many(models, E_t, window_index_many(et, et))
        scores = np.clip(np.rint(prob * 100), 0, 100).astype(np.uint8)
        tmp = dest / f"{vdir.name}.tmp.npz"
        np.savez_compressed(tmp, times=et.astype(np.float32), scores=scores)
        tmp.replace(dest / f"{vdir.name}.npz")
        done.append(vdir.name)
        if k % 20 == 0 or k == len(vdirs):
            print(f"   {k} of {len(vdirs)} videos ({time.time() - t0:.0f} s)", flush=True)
    index = {"model": name, "created": time.strftime("%Y-%m-%d %H:%M"), "tags": meta["tags"],
             "cutoffs": [int(round(x * 100)) for x in meta["thresholds"]], "step": STEP,
             "test_videos": meta.get("test_videos") or [], "scores": meta.get("scores", {}).get("plain"),
             "videos": done}
    (dest / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"Exported the {name} scores for {len(done)} videos → {dest}")
    return 0


def score(name: str, vs: str, label_weight: float) -> int:
    import torch
    models, meta = load_models(name)
    if not models:
        print(f"No model '{name}' in {WORK / 'models'}.")
        return 1
    vocab, labels, test, ds, _, _, test_rows = _prepare(label_weight)
    if ds["tags"] != meta["tags"]:
        print("⚠️ Your vocabulary has changed since this model was trained — retrain it (train) to score "
              "it on today's tags.")
        return 1
    E_t = torch.as_tensor(ds["E"], device=_device())
    rc = _score(models, E_t, ds, test_rows, np.asarray(meta["thresholds"], np.float32), vocab, labels, vs,
                name, meta)
    (WORK / "models" / name / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train / score the student tag model on the laptop.")
    ap.add_argument("command", choices=["fetch", "train", "score", "export", "all"])
    ap.add_argument("--name", default="v1", help="model name (default v1)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--label-weight", type=float, default=LABEL_WEIGHT,
                    help=f"how many teacher answers one of your labels counts as (default {LABEL_WEIGHT:g})")
    ap.add_argument("--folds", type=int, default=5, help="models trained, each holding out a different share "
                    "of the training videos (default 5)")
    ap.add_argument("--vs", default="", help="teacher test run id or label to compare with (default: newest try2)")
    a = ap.parse_args(argv)
    WORK.mkdir(exist_ok=True)
    if a.command in ("fetch", "all"):
        rc = fetch()
        if rc or a.command == "fetch":
            return rc
    if a.command in ("train", "all"):
        return train(a.name, a.epochs, a.label_weight, a.vs, max(2, a.folds))
    if a.command == "export":
        return export(a.name)
    return score(a.name, a.vs, a.label_weight)


if __name__ == "__main__":
    sys.exit(main())
