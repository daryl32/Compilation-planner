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
    python student_tags.py picks --name v5b  the moments worth labelling next → 🎯 Training Labels → 🎓 Student's picks

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
VIDEO_FILES = ("embeddings.npz", TEACHER_FILE, "person_curve.npy", "detections.npz", "vision.json")

STEP = 0.5                                # the embeddings are 2 per second
HALF = 4                                  # ±4 steps = ±2 s around the moment → 9 embeddings
LABEL_WEIGHT = 5.0                        # one of your labels counts as this many teacher answers
COPY_WEIGHT = 0.5                         # a teacher answer copied from a near-identical moment
SCENE_WEIGHT = 2.0                        # a moment inside a scene you corrected in the Reviewer
SCENE_STEP = 2.0                          # extra moments taken every this many seconds in those scenes
SCENE_EDGE = 1.0                          # …but not this close to the scene's start / end (cuts)
TRUST_MIN_LABELS = 10                     # your labels with a tag (training videos) before its teacher trust is measured
TRUST_FLOOR = 0.1                         # the least a teacher answer can count for on a tag
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


def teacher_trust(vocab: list, labels: list, test: set, root: Path) -> dict:
    """How right the teacher is per tag, measured on your 🎯 / ✏️ labels in the training videos (never the
    test set): {tag lower: {"f1", "weight", "labelled"}}. The weight its answers get for that tag is its F1
    (at least TRUST_FLOOR); a tag with fewer than TRUST_MIN_LABELS of your labels keeps weight 1."""
    tags = [r["tag"] for r in vocab]
    by_video = {}
    for i in labels:
        if i["video_id"] not in test:
            by_video.setdefault(i["video_id"], []).append(i)
    pairs = []
    for v, mine in by_video.items():
        tj = _read_json(root / v / TEACHER_FILE, {}) or {}
        res = (tj.get("results") or {}).values()
        by_id = {r["gold_id"]: r for r in res if r.get("gold_id")}
        by_t = {round(float(r["t"]), 2): r for r in res if r.get("t") is not None}
        for i in mine:
            r = by_id.get(i.get("id")) or by_t.get(round(float(i["t"]), 2))
            if r is not None and r.get("ok") is not False:
                pairs.append((r.get("tags") or [], i.get("tags") or []))
    sc = TL.score_tags(pairs, tags)["per_tag"] if pairs else {}
    out = {}
    for t in tags:
        s_ = sc.get(t) or {"tp": 0, "fn": 0, "f1": None}
        n_lab = s_["tp"] + s_["fn"]
        f1 = s_["f1"] if s_["f1"] is not None else 0.0
        w = max(TRUST_FLOOR, f1) if n_lab >= TRUST_MIN_LABELS else 1.0
        out[t.lower()] = {"f1": None if n_lab < TRUST_MIN_LABELS else round(f1, 3), "weight": round(w, 3),
                          "labelled": n_lab}
    return out


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


# ---------------------------------------------------------------------------
# Body pose (detections.npz from the Modal vision stage: YOLO pose at 15 fps on a 640-wide copy)
# ---------------------------------------------------------------------------

POSE_FPS = 15.0
POSE_W, POSE_H = 640.0, 360.0             # the pose copy's size (16:9; only used to scale boxes 0–1)
POSE_PEOPLE = 2                           # the two biggest people in the frame
PER_PERSON = 5 + 17 * 3                   # box (centre x, y, width, height, confidence) + 17 keypoints × (x, y, seen)
POSE_DIM = 1 + 1 + POSE_PEOPLE * PER_PERSON + 4   # has pose · people count · people · second vs first


def _person(box, kxy, kc) -> np.ndarray:
    """One person: the box scaled to the frame, then each keypoint relative to the box centre in box
    heights (so it's the body's shape, whatever its size), with how sure the pose model was."""
    x1, y1, x2, y2, conf = [float(v) for v in box[:5]]
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    rel = np.stack([(kxy[:, 0] - cx) / bh, (kxy[:, 1] - cy) / bh], axis=1)
    seen = np.clip(kc, 0, 1)[:, None]
    rel = rel * (seen > 0.3)
    head = np.array([cx / POSE_W, cy / POSE_H, bw / POSE_W, bh / POSE_H, conf], np.float32)
    return np.concatenate([head, np.concatenate([rel, seen], axis=1).ravel()]).astype(np.float32)


def pose_features(det: dict, times: np.ndarray) -> np.ndarray:
    """(len(times), POSE_DIM): the body pose nearest each time — the two biggest people (box +
    keypoint layout), how many people, and where the second is relative to the first (above / below,
    beside, overlapping: who's on top). All zeros with has-pose 0 where nothing was detected."""
    out = np.zeros((len(times), POSE_DIM), np.float32)
    frame = np.asarray(det.get("frame", []), np.int64)
    if not frame.size:
        return out
    box, kxy, kc = det["box"], det["kxy"], det["kconf"]
    order = np.argsort(frame, kind="stable")
    frame, box, kxy, kc = frame[order], box[order], kxy[order], kc[order]
    uniq, start = np.unique(frame, return_index=True)
    ends = np.append(start[1:], len(frame))
    want = np.rint(np.asarray(times, float) * POSE_FPS).astype(np.int64)
    pos = np.clip(np.searchsorted(uniq, want), 0, len(uniq) - 1)
    prev = np.clip(pos - 1, 0, len(uniq) - 1)
    pos = np.where(np.abs(uniq[prev] - want) < np.abs(uniq[pos] - want), prev, pos)
    for k, (j, w) in enumerate(zip(pos, want)):
        if abs(int(uniq[j]) - int(w)) > POSE_FPS:          # no detection within a second
            continue
        a, b = start[j], ends[j]
        area = (box[a:b, 2] - box[a:b, 0]) * (box[a:b, 3] - box[a:b, 1]) * box[a:b, 4]
        top = a + np.argsort(-area)[:POSE_PEOPLE]
        out[k, 0] = 1.0
        out[k, 1] = min(b - a, 4) / 4.0
        people = [_person(box[i], kxy[i], kc[i]) for i in top]
        for n, f in enumerate(people):
            out[k, 2 + n * PER_PERSON: 2 + (n + 1) * PER_PERSON] = f
        if len(people) == 2:
            p1, p2 = people
            out[k, -4:] = [p2[0] - p1[0], p2[1] - p1[1], p2[3] / max(p1[3], 1e-3), 1.0]
    return out


def video_features(vdir: Path, use_pose: bool) -> tuple:
    """(frame times, features) for one video, or (None, None): the unit-length SigLIP2 embeddings,
    plus the body-pose features at the same times when use_pose (cached in pose_feats.npz)."""
    try:
        z = np.load(vdir / "embeddings.npz")
        et = np.asarray(z["frame_times"], float)
        emb = np.asarray(z["frame_embs"], np.float32)
    except (OSError, KeyError, ValueError):
        return None, None
    if not et.size or emb.ndim != 2 or len(emb) != et.size:
        return None, None
    order = np.argsort(et)
    et, emb = et[order], emb[order]
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
    if not use_pose:
        return et, emb
    cache, src = vdir / "pose_feats.npz", vdir / "detections.npz"
    pf = None
    try:
        if cache.stat().st_mtime >= src.stat().st_mtime:
            c = np.load(cache)
            if c["feats"].shape == (len(et), POSE_DIM):
                pf = c["feats"]
    except (OSError, KeyError, ValueError):
        pf = None
    if pf is None:
        try:
            d = np.load(src)
            det = {k: d[k] for k in ("frame", "box", "kxy", "kconf")}
        except (OSError, KeyError, ValueError):
            det = {}
        pf = pose_features(det, et)
        if det:
            np.savez_compressed(cache, feats=pf)
    return et, np.concatenate([emb, pf], axis=1)


# ---------------------------------------------------------------------------
# Time context (step 2): look along a video's half-second scores, inside hard cuts
# ---------------------------------------------------------------------------

CTX_SPANS = (4, 10, 20)                   # averages over ±2 s, ±5 s and ±10 s (half-second steps)
CTX_MAX = 10                              # and the highest score within ±5 s
CTX_K = 2 + len(CTX_SPANS)                # numbers per tag: own score, the averages, the max


def video_cuts(vdir: Path) -> list:
    """Hard cuts (TransNetV2, seconds) from the vision stage; [] if unknown."""
    return [float(c) for c in ((_read_json(vdir / "vision.json", {}) or {}).get("cuts") or [])]


def context_features(times: np.ndarray, probs: np.ndarray, cuts) -> np.ndarray:
    """(T, tags × CTX_K): for every half second, each tag's own score, its average over ±2 / ±5 /
    ±10 s and its highest score within ±5 s — never reaching across a hard cut, so a shot's tags
    don't leak into the next shot."""
    T, n = probs.shape
    seg = np.searchsorted(np.sort(np.asarray(cuts, float)), np.asarray(times, float), side="right")
    sums = {s_: probs.copy() for s_ in CTX_SPANS}
    cnt = {s_: np.ones(T, np.float32) for s_ in CTX_SPANS}
    mx = probs.copy()
    reach = max(CTX_SPANS)
    base = np.arange(T)
    for d in range(-reach, reach + 1):
        if d == 0:
            continue
        src = base + d
        ok = (src >= 0) & (src < T)
        src = np.clip(src, 0, T - 1)
        ok &= seg[src] == seg
        val = np.where(ok[:, None], probs[src], 0.0)
        for s_ in CTX_SPANS:
            if abs(d) <= s_:
                sums[s_] += val
                cnt[s_] += ok
        if abs(d) <= CTX_MAX:
            mx = np.maximum(mx, val)
    feats = [probs] + [sums[s_] / cnt[s_][:, None] for s_ in CTX_SPANS] + [mx]
    return np.concatenate(feats, axis=1).astype(np.float32)


def make_ctx_model(n_tags: int, hidden: int = 256, drop: float = 0.2):
    """The second step: every tag's context numbers in → every tag's score out (so tags can inform
    each other: a held position, a shot that's been POV for the last 10 s…). Starts from the step-1
    score: it learns a correction on top (a residual on the logits)."""
    import torch
    from torch import nn

    class ContextHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(n_tags * CTX_K), nn.Linear(n_tags * CTX_K, hidden), nn.GELU(),
                                     nn.Dropout(drop), nn.Linear(hidden, n_tags))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x):
            p = x[:, :n_tags].clamp(1e-4, 1 - 1e-4)
            return torch.log(p / (1 - p)) + self.net(x)

    return ContextHead()


def video_timeline(models: list, vdir: Path, use_pose: bool, emb_dim: int = None):
    """(times, step-1 scores at every half second) for one video, from these models averaged."""
    import torch
    et, feats = video_features(vdir, use_pose)
    if et is None or (emb_dim and feats.shape[1] - (POSE_DIM if use_pose else 0) != emb_dim):
        return None, None
    E_t = torch.as_tensor(feats.astype(np.float16), device=_device())
    return et, predict_many(models, E_t, window_index_many(et, et))


def train_ctx(X: np.ndarray, y: np.ndarray, w: np.ndarray, train_rows: np.ndarray, val_rows: np.ndarray,
              epochs: int = 40, batch: int = 512, lr: float = 2e-3, patience: int = 6, seed: int = SEED):
    """Weighted BCE like step 1, early stop on the held-out videos. Returns the model."""
    import torch
    torch.manual_seed(seed)
    dev = _device()
    X_t, y_t, w_t = (torch.as_tensor(a, device=dev) for a in (X, y, w))
    model = make_ctx_model(y.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    def loss_on(r):
        return (lossf(model(X_t[r]), y_t[r]) * w_t[r]).sum() / w_t[r].sum().clamp(min=1e-6)

    best, best_state, bad = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(train_rows)
        for s_ in range(0, len(perm), batch):
            r = torch.as_tensor(perm[s_:s_ + batch], device=dev)
            l = loss_on(r)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_on(torch.as_tensor(val_rows, device=dev))) if len(val_rows) else float("nan")
        if vl < best - 1e-4 or np.isnan(vl):
            best, bad = vl, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_ctx(models: list, X: np.ndarray) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        X_t = torch.as_tensor(X, device=_device())
        for m in models:
            m.eval()
            out.append(torch.sigmoid(m(X_t)).cpu().numpy())
    return np.mean(out, axis=0)


def rows_context(ds: dict, rows: np.ndarray, timelines: dict) -> np.ndarray:
    """The context numbers at each row's moment (nearest half second of its video's timeline)."""
    n = len(ds["tags"])
    X = np.zeros((len(rows), n * CTX_K), np.float32)
    for v in set(ds["video"][rows]):
        sel = np.flatnonzero(ds["video"][rows] == v)
        tl = timelines.get(v)
        if tl is None:
            continue
        et, ctx = tl
        k = np.clip(np.searchsorted(et, ds["t"][rows][sel]), 1, len(et) - 1)
        k = np.where(np.abs(et[k - 1] - ds["t"][rows][sel]) <= np.abs(et[k] - ds["t"][rows][sel]), k - 1, k)
        X[sel] = ctx[k]
    return X


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
                  root: Path = None, scenes: dict = None, scene_weight: float = SCENE_WEIGHT,
                  use_trust: bool = True, use_pose: bool = False) -> dict:
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
    trust = teacher_trust(vocab, labels, test, root)
    if use_trust:                     # the teacher's answers count for its F1 on each tag
        teacher_on *= np.array([trust[t.lower()]["weight"] for t in tags], np.float32)
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

    vdirs = sorted(p for p in root.iterdir() if p.is_dir())
    for n_v, vdir in enumerate(vdirs, 1):
        v = vdir.name
        et, emb = video_features(vdir, use_pose)
        if et is None:
            skipped.append(v)
            continue
        if use_pose and n_v % 25 == 0:
            print(f"   body pose: {n_v} of {len(vdirs)} videos", flush=True)
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
            "scene_rules": {k: sorted(v) for k, v in rules.items()}, "scene_agreement": agreement,
            "teacher_trust": trust, "use_trust": use_trust,
            "pose_dim": POSE_DIM if use_pose else 0}


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

def make_model(dim: int, n_tags: int, steps: int = 2 * HALF + 1, hidden: int = 512, drop: float = 0.2,
               pose_dim: int = 0):
    import torch
    from torch import nn

    class StudentHead(nn.Module):
        """LayerNorm → project → + time position → one temporal conv (motion across the 4 s) →
        attention pooling + the centre frame → MLP → one score per tag."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.proj = nn.Linear(dim, hidden)
            if pose_dim:                            # body pose: its own norm + projection, added in
                self.pnorm = nn.LayerNorm(pose_dim)
                self.pproj = nn.Sequential(nn.Linear(pose_dim, 256), nn.GELU(), nn.Linear(256, hidden))
            self.pos = nn.Parameter(torch.zeros(steps, hidden))
            self.mix = nn.Conv1d(hidden, hidden, 3, padding=1)
            self.att = nn.Linear(hidden, 1)
            self.out = nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(2 * hidden, hidden), nn.GELU(),
                                     nn.Dropout(drop), nn.Linear(hidden, n_tags))
            self.centre = steps // 2

        def forward(self, x):                       # x: batch × steps × (dim [+ pose_dim])
            h = self.proj(self.norm(x[..., :dim])) + self.pos
            if pose_dim:
                h = h + self.pproj(self.pnorm(x[..., dim:dim + pose_dim]))
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
    pose_dim = int(ds.get("pose_dim") or 0)
    model = make_model(ds["E"].shape[1] - pose_dim, ds["y"].shape[1], pose_dim=pose_dim).to(dev)
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

def _prepare(label_weight: float, use_trust: bool = True, use_pose: bool = False):
    vocab, labels, test = _vocab(), _labels(), set(test_videos())
    if not vocab:
        raise SystemExit("No tag_vocabulary.csv in _student/library — run `python student_tags.py fetch`.")
    if not test:
        print("⚠️ No test set downloaded (testset.json) — the model will train but can't be scored.")
    ds = build_dataset(vocab, labels, test, label_weight, use_trust=use_trust, use_pose=use_pose)
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
    if ds.get("pose_dim"):
        has = float((ds["E"][:, -POSE_DIM].astype(np.float32) > 0.5).mean())
        print(f"   body pose: {POSE_DIM} numbers per half second; a person detected at {100 * has:.0f}% of them")
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


def train(name: str, epochs: int, label_weight: float, vs: str, folds: int = 5, use_trust: bool = True,
          use_pose: bool = False, use_context: bool = False) -> int:
    """k models, each trained with a different fifth of the training videos held out (it stops early on
    that fifth). Their held-out answers cover every training video, so each tag's cut-off is picked on
    all of them, not one small split; on the test set the k models' scores are averaged."""
    import torch
    vocab, labels, test, ds, _, _, test_rows = _prepare(label_weight, use_trust, use_pose)
    pose_dim = ds["pose_dim"]
    emb_dim = ds["E"].shape[1] - pose_dim
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
    step1_thresholds = thresholds
    test_prob = predict_many(models, E_t, ds["idx"][test_rows]) if len(test_rows) else None
    step1_test_prob = test_prob
    ctx_models, ctx_meta = [], None
    if use_context:
        print("Step 2 — time context: scoring every half second of every video…", flush=True)
        if dev.type == "cuda":
            del E_t
            torch.cuda.empty_cache()
        fold_of = {v: k for k, g in enumerate(groups) for v in g}
        root = WORK / "videos"
        timelines = {}
        for n_v, v in enumerate(ds["videos"], 1):
            use = [models[fold_of[v]]] if v in fold_of else models       # a training video: only the
            et, prob = video_timeline(use, root / v, use_pose)            # model that never saw it
            if et is not None:
                timelines[v] = (et, context_features(et, prob, video_cuts(root / v)))
            if n_v % 25 == 0:
                print(f"   {n_v} of {len(ds['videos'])} videos", flush=True)
        learn = np.flatnonzero(~np.isin(vids, list(test)))
        X = np.zeros((len(vids), len(ds["tags"]) * CTX_K), np.float32)
        X[learn] = rows_context(ds, learn, timelines)
        if len(test_rows):
            X[test_rows] = rows_context(ds, test_rows, timelines)
        oof2_rows, oof2_prob = [], []
        for f, val_v in enumerate(groups, 1):
            val_rows = np.flatnonzero(np.isin(vids, val_v))
            train_rows = np.flatnonzero(np.isin(vids, [v for g in groups if g is not val_v for v in g]))
            m2 = train_ctx(X, ds["y"], ds["w"], train_rows, val_rows)
            oof2_rows.append(val_rows)
            oof2_prob.append(predict_ctx([m2], X[val_rows]))
            ctx_models.append(m2)
        r2 = np.concatenate(oof2_rows)
        thresholds = pick_thresholds(np.concatenate(oof2_prob), ds["y"][r2], mask=ds["w"][r2])
        if len(test_rows):
            test_prob = predict_ctx(ctx_models, X[test_rows])
        ctx_meta = {"spans": list(CTX_SPANS), "max": CTX_MAX, "k": CTX_K}
    minutes = (time.time() - t0) / 60
    out = WORK / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("head*.pt")) + list(out.glob("ctx*.pt")):
        old.unlink()
    for f, m in enumerate(models, 1):
        torch.save(m.state_dict(), out / f"head_{f}.pt")
    for f, m in enumerate(ctx_models, 1):
        torch.save(m.state_dict(), out / f"ctx_{f}.pt")
    meta = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "tags": ds["tags"],
            "thresholds": [float(x) for x in thresholds], "dim": int(emb_dim), "pose_dim": int(pose_dim), "half": HALF,
            "step": STEP, "label_weight": label_weight, "folds": folds, "counts": ds["counts"],
            "scene_tags_used": ds["scene_tags_used"], "scene_rules": ds["scene_rules"],
            "teacher_trust": ds["teacher_trust"] if use_trust else None,
            "epochs_run": [len(h) for h in hists], "history": hists,
            "train_minutes": round(minutes, 2), "videos": len(ds["videos"]), "test_videos": sorted(test),
            "labels_newest": max((i.get("labelled_at") or "" for i in labels), default=""),
            "context": ctx_meta, "step1_thresholds": [float(x) for x in step1_thresholds]}
    print(f"Trained in {minutes:.1f} min.", flush=True)
    step1 = None
    if ctx_meta and step1_test_prob is not None:
        step1 = (step1_test_prob, step1_thresholds)
    rc = _score(models, None, ds, test_rows, thresholds, vocab, labels, vs, name, meta, prob=test_prob,
                step1=step1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Model saved → {out}")
    return rc


def _score(models, E_t, ds, test_rows, thresholds, vocab, labels, vs, name, meta, prob=None, step1=None) -> int:
    if not len(test_rows):
        print("No labelled test-set frames to score on.")
        return 0
    if prob is None:
        prob = predict_many(models, E_t, ds["idx"][test_rows])
    runs = teacher_runs()
    ref = pick_reference(runs, vs)
    if vs and ref is None:
        print(f"⚠️ No teacher test run called '{vs}' — scoring the student alone.")
    md, res = score_report(ds, prob, test_rows, thresholds, vocab, labels, ref, name)
    meta["scores"] = {k: v["overall"] for k, v in res.items()}
    if step1 is not None:                   # the same run without step 2, on the same frames
        _, r1 = score_report(ds, step1[0], test_rows, step1[1], vocab, labels, ref, name)
        meta["scores_step1"] = {k: v["overall"] for k, v in r1.items()}
        f = lambda x: "–" if x is None else f"{x:.2f}"
        o1p, o1r = r1["plain"]["overall"], r1["rules"]["overall"]
        line = (f"| Student · step 1 only (no time context) · plain / rules | {r1['plain']['n']} | "
                f"{f(o1p['f1'])} / {f(o1r['f1'])} | | | {o1p['fp']} / {o1r['fp']} | {o1p['fn']} / {o1r['fn']} |")
        lines = md.splitlines()
        at = next(i for i, l in enumerate(lines) if l.startswith("| Student · with your vocabulary rules"))
        lines.insert(at + 1, line)
        md = "\n".join(lines)
    meta["reference_run"] = ref["id"] if ref else None
    md += "\n\n" + trust_markdown(ds) + "\n\n" + scene_rules_markdown(ds)
    report = HERE / "student_report.md"
    report.write_text(md, encoding="utf-8")
    (WORK / "models" / name / "student_report.md").write_text(md, encoding="utf-8")
    print("\n" + "\n".join(md.splitlines()[:14]) + f"\n\nFull report (per tag) → {report}")
    return 0


def trust_markdown(ds: dict) -> str:
    """How much the teacher's answers counted per tag (its F1 on your labelled training moments)."""
    tr = ds.get("teacher_trust") or {}
    if not tr:
        return ""
    names = {t.lower(): t for t in ds["tags"]}
    head = ("How much the teacher's answers counted, per tag — its F1 on your labelled moments in the training "
            "videos (test set never used). Your labels and Reviewer answers always count in full."
            if ds.get("use_trust") else "Teacher trust was OFF for this run (every teacher answer counted in full); "
            "for reference, the teacher's F1 on your labelled training moments:")
    md = ["## Teacher trust", "", head, "", "| Tag | Your labels with it | Teacher F1 | Teacher answers count for |",
          "|---|---|---|---|"]
    for low, x in sorted(tr.items(), key=lambda kv: (kv[1]["weight"], kv[0])):
        f1 = "–" if x["f1"] is None else f"{x['f1']:.2f}"
        w = (f"{x['weight']:.2f}" if x["f1"] is not None else "1.00 (too few labels to measure)") \
            if ds.get("use_trust") else "1.00"
        md.append(f"| {names.get(low, low)} | {x['labelled']} | {f1} | {w} |")
    return "\n".join(md)


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
        m = make_model(meta["dim"], len(meta["tags"]), pose_dim=int(meta.get("pose_dim") or 0)).to(_device())
        m.load_state_dict(torch.load(p, map_location=_device()))
        models.append(m)
    return models, meta


def load_ctx_models(name: str, meta: dict) -> list:
    import torch
    if not meta.get("context"):
        return []
    out = []
    for p in sorted((WORK / "models" / name).glob("ctx*.pt")):
        m = make_ctx_model(len(meta["tags"])).to(_device())
        m.load_state_dict(torch.load(p, map_location=_device()))
        out.append(m)
    return out


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
    ctx_models = load_ctx_models(name, meta)
    done, t0 = [], time.time()
    vdirs = sorted(p for p in root.iterdir() if p.is_dir())
    for k, vdir in enumerate(vdirs, 1):
        pose_dim = int(meta.get("pose_dim") or 0)
        et, emb = video_features(vdir, bool(pose_dim))
        if et is None or emb.shape[1] != meta["dim"] + pose_dim:
            continue
        E_t = torch.as_tensor(emb.astype(np.float16), device=dev)
        prob = predict_many(models, E_t, window_index_many(et, et))
        if ctx_models:
            prob = predict_ctx(ctx_models, context_features(et, prob, video_cuts(vdir)))
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


PICK_STEP = 4                             # candidates every 2 s (half-second steps)
PICK_GAP = 10.0                           # picks in a video at least this far apart (s)
PICK_PER_VIDEO = 12
PICK_NEAR_LABEL = 3.0                     # never within this of a moment you've labelled


def pick_priority(trust: dict, labels: list, tags: list) -> np.ndarray:
    """How much a label on each tag is worth: more where the teacher is weak (your label is the only
    real teacher there) and where you have few labels so far."""
    counts = {}
    for i in labels:
        for t in i.get("tags") or []:
            counts[str(t).lower()] = counts.get(str(t).lower(), 0) + 1
    pr = []
    for t in tags:
        tr = (trust or {}).get(t.lower()) or {}
        weak = 1.0 - (tr["weight"] if tr.get("f1") is not None else 0.5)
        rare = 1.0 / np.sqrt(1.0 + counts.get(t.lower(), 0) / 25.0)
        pr.append(max(weak, 0.15) * (0.5 + rare))
    return np.asarray(pr, np.float32)


def uncertainty(prob: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """1 at a tag's cut-off (the student can't decide), falling to 0 at a sure yes or no."""
    span = np.maximum(cutoffs, 1.0 - cutoffs)
    return np.clip(1.0 - np.abs(prob - cutoffs) / span, 0.0, 1.0)


def choose_picks(times: np.ndarray, value: np.ndarray, taken, n: int, gap: float = PICK_GAP) -> list:
    """Greedy: the highest-value moments, at least `gap` apart and away from `taken` times."""
    out = []
    for k in np.argsort(-value):
        if len(out) >= n or value[k] <= 0:
            break
        t = float(times[k])
        if any(abs(t - u) < PICK_NEAR_LABEL for u in taken) or any(abs(t - u) < gap for u in out):
            continue
        out.append(t)
    return out


def make_picks(name: str, n: int = 300, n_test: int = 80, dest: Path = None, seed: int = SEED) -> int:
    """student_picks.json for 🎯 Training Labels → 🎓 Student's picks (catalogue/trial_results on Drive):
      • training videos: the moments where the student is least sure about a tag worth labelling
        (the teacher is weak on it / you have few labels), most useful first, ≤12 per video, 10 s apart;
      • 🧪 test-set videos: random moments (not chosen by the student, so the test set stays fair).
    Every 5th card is a test-set one. The student's own tags come prefilled."""
    models, meta = load_models(name)
    if not models:
        print(f"No model '{name}' in {WORK / 'models'} — train it first.")
        return 1
    ctx_models = load_ctx_models(name, meta)
    tags = meta["tags"]
    cutoffs = np.asarray(meta["thresholds"], np.float32)
    labels, test = _labels(), set(meta.get("test_videos") or test_videos())
    trust = meta.get("teacher_trust") or teacher_trust(_vocab(), labels, test, WORK / "videos")
    prio = pick_priority(trust, labels, tags)
    by_video = {}
    for i in labels:
        by_video.setdefault(i["video_id"], []).append(float(i["t"]))
    rng = np.random.default_rng(seed)
    train_picks, test_picks = [], []
    use_pose = bool(meta.get("pose_dim"))
    vdirs = sorted(p for p in (WORK / "videos").iterdir() if p.is_dir())
    for k, vdir in enumerate(vdirs, 1):
        v = vdir.name
        et, prob = video_timeline(models, vdir, use_pose, meta["dim"])
        if et is None:
            continue
        if ctx_models:
            prob = predict_ctx(ctx_models, context_features(et, prob, video_cuts(vdir)))
        sel = np.arange(0, len(et), PICK_STEP)
        tt, pp = et[sel], prob[sel]
        edge = (tt > et[0] + 2) & (tt < et[-1] - 2)
        if v in test:
            val = rng.random(len(tt)) * edge
            for t in choose_picks(tt, val, by_video.get(v, []), max(2, n_test // max(len(test), 1))):
                j = int(np.argmin(np.abs(tt - t)))
                test_picks.append({"video_id": v, "t": round(t, 2), "set": "test", "why": "",
                                   "tags": [x for x, p_, c in zip(tags, pp[j], cutoffs) if p_ >= c]})
            continue
        u = uncertainty(pp, cutoffs) * prio
        best = u.argmax(axis=1)
        val = u.max(axis=1) * edge
        for t in choose_picks(tt, val, by_video.get(v, []), PICK_PER_VIDEO):
            j = int(np.argmin(np.abs(tt - t)))
            b = int(best[j])
            train_picks.append({"video_id": v, "t": round(t, 2), "set": "train", "value": float(val[j]),
                                "why": f"{tags[b]} ({pp[j, b]:.2f}, cut-off {cutoffs[b]:.2f})",
                                "tags": [x for x, p_, c in zip(tags, pp[j], cutoffs) if p_ >= c]})
        if k % 25 == 0:
            print(f"   {k} of {len(vdirs)} videos", flush=True)
    train_picks.sort(key=lambda x: -x["value"])
    train_picks = train_picks[:n]
    rng.shuffle(test_picks)
    test_picks = test_picks[:n_test]
    out, ti = [], iter(test_picks)
    for i, pk in enumerate(train_picks, 1):
        pk.pop("value", None)
        out.append(pk)
        if i % 4 == 0:
            nxt = next(ti, None)
            if nxt:
                out.append(nxt)
    out += list(ti)
    if dest is None:
        from config import CATALOGUE_DIR
        dest = Path(CATALOGUE_DIR) / "trial_results" / "student_picks.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = {"model": name, "created": time.strftime("%Y-%m-%d %H:%M"), "picks": out}
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(dest)
    why = {}
    for pk in out:
        if pk["why"]:
            t_ = pk["why"].split(" (")[0]
            why[t_] = why.get(t_, 0) + 1
    top = ", ".join(f"{t} {c}" for t, c in sorted(why.items(), key=lambda kv: -kv[1])[:8])
    print(f"{len(out)} picks ({len(train_picks)} where the student is unsure, {len(test_picks)} test-set) → {dest}")
    print(f"   most picked for: {top}")
    print("   In the app: ☁️ Sync from Google Drive, then 🎯 Training Labels → Cards: 🎓 Student's picks.")
    return 0


def score(name: str, vs: str, label_weight: float) -> int:
    import torch
    models, meta = load_models(name)
    if not models:
        print(f"No model '{name}' in {WORK / 'models'}.")
        return 1
    vocab, labels, test, ds, _, _, test_rows = _prepare(label_weight, use_pose=bool(meta.get("pose_dim")))
    if ds["tags"] != meta["tags"]:
        print("⚠️ Your vocabulary has changed since this model was trained — retrain it (train) to score "
              "it on today's tags.")
        return 1
    ctx_models = load_ctx_models(name, meta)
    prob = None
    if ctx_models and len(test_rows):
        use_pose = bool(meta.get("pose_dim"))
        timelines = {}
        for v in set(ds["video"][test_rows]):
            et, p1 = video_timeline(models, WORK / "videos" / v, use_pose)
            if et is not None:
                timelines[v] = (et, context_features(et, p1, video_cuts(WORK / "videos" / v)))
        prob = predict_ctx(ctx_models, rows_context(ds, test_rows, timelines))
    E_t = torch.as_tensor(ds["E"], device=_device())
    rc = _score(models, E_t, ds, test_rows, np.asarray(meta["thresholds"], np.float32), vocab, labels, vs,
                name, meta, prob=prob)
    (WORK / "models" / name / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train / score the student tag model on the laptop.")
    ap.add_argument("command", choices=["fetch", "train", "score", "export", "picks", "all"])
    ap.add_argument("--n", type=int, default=300, help="picks: how many where the student is unsure (default 300)")
    ap.add_argument("--n-test", type=int, default=80, help="picks: how many random test-set moments (default 80)")
    ap.add_argument("--name", default="v1", help="model name (default v1)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--label-weight", type=float, default=LABEL_WEIGHT,
                    help=f"how many teacher answers one of your labels counts as (default {LABEL_WEIGHT:g})")
    ap.add_argument("--folds", type=int, default=5, help="models trained, each holding out a different share "
                    "of the training videos (default 5)")
    ap.add_argument("--pose", action="store_true", help="add the body pose (v6 — it didn't help: patchy detections)")
    ap.add_argument("--context", action="store_true", help="add the second step over ±10 s of scores (v7: even "
                    "overall — fewer wrong tags, more missed)")
    ap.add_argument("--no-teacher-trust", action="store_true",
                    help="count the teacher's answers fully on every tag (as v4 did)")
    ap.add_argument("--vs", default="", help="teacher test run id or label to compare with (default: newest try2)")
    a = ap.parse_args(argv)
    WORK.mkdir(exist_ok=True)
    if a.command in ("fetch", "all"):
        rc = fetch()
        if rc or a.command == "fetch":
            return rc
    if a.command in ("train", "all"):
        return train(a.name, a.epochs, a.label_weight, a.vs, max(2, a.folds), not a.no_teacher_trust,
                     a.pose, a.context)
    if a.command == "export":
        return export(a.name)
    if a.command == "picks":
        return make_picks(a.name, a.n, a.n_test)
    return score(a.name, a.vs, a.label_weight)


if __name__ == "__main__":
    sys.exit(main())
