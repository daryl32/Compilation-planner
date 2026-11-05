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
    python student_tags.py all               fetch + train

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
    items = _labels()
    newest = max((i.get("labelled_at") or "" for i in items), default="")
    print(f"Your labels: {len(items)} tag answers in {len({i['video_id'] for i in items})} videos, "
          f"newest saved {newest.replace('T', ' ') or '—'}.")
    print("   (Not seeing today's corrections? Press ☁️ Save edits to Drive in the app, wait for Drive "
          "to sync to this laptop, then run fetch again.)")
    return 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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
                  root: Path = None) -> dict:
    """Every usable moment as an index window into one big embedding array.
    Returns {"E" (frames × D, float16), "idx" (samples × 9), "y", "w", "video" (per sample),
    "is_label", "gold_id", "t", "videos", "skipped"}. Test-set videos give label rows only
    (for scoring) and are flagged so training never sees them."""
    root = root or WORK / "videos"
    tags = [r["tag"] for r in vocab]
    tag_ix = {t.lower(): j for j, t in enumerate(tags)}
    n = len(tags)
    by_video = {}
    for i in labels:
        by_video.setdefault(i["video_id"], []).append(i)
    E, idx, ys, ws, vids, is_lab, gids, ts = [], [], [], [], [], [], [], []
    offset, skipped, used = 0, [], []
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
        rows = 0
        if v not in test:
            tj = _read_json(vdir / TEACHER_FILE, {}) or {}
            for r in (tj.get("results") or {}).values():
                if r.get("t") is None or r.get("ok") is False:
                    continue
                if (r.get("gold_id") and r["gold_id"] in lab_id) or round(float(r["t"]), 2) in lab_t:
                    continue                                   # your label replaces it
                idx.append(window_index(et, float(r["t"])) + offset)
                ys.append(teacher_target(r, tag_ix, n))
                ws.append(COPY_WEIGHT if r.get("copied_from") else 1.0)
                vids.append(v); is_lab.append(False); gids.append(""); ts.append(float(r["t"]))
                rows += 1
        for i in mine:
            idx.append(window_index(et, float(i["t"])) + offset)
            ys.append(hard_target(i.get("tags"), tag_ix, n))
            ws.append(label_weight)
            vids.append(v); is_lab.append(True); gids.append(i.get("id") or ""); ts.append(float(i["t"]))
            rows += 1
        if not rows:
            continue
        E.append(emb.astype(np.float16))
        offset += len(emb)
        used.append(v)
    if not E:
        raise SystemExit(f"No usable videos in {root} — run `python student_tags.py fetch` first.")
    return {"E": np.concatenate(E), "idx": np.stack(idx).astype(np.int64), "y": np.stack(ys),
            "w": np.asarray(ws, np.float32), "video": np.asarray(vids), "is_label": np.asarray(is_lab),
            "gold_id": np.asarray(gids), "t": np.asarray(ts), "videos": used, "skipped": skipped,
            "tags": tags}


def split_videos(videos: list, test: set, share: float = VAL_SHARE, seed: int = SEED) -> tuple:
    """(train videos, validation videos) — the test set is in neither."""
    pool = sorted(v for v in videos if v not in test)
    rng = random.Random(seed)
    rng.shuffle(pool)
    k = max(1, int(round(len(pool) * share))) if len(pool) > 4 else 0
    return sorted(pool[k:]), sorted(pool[:k])


def pick_thresholds(prob: np.ndarray, truth: np.ndarray, min_pos: int = 5) -> np.ndarray:
    """Per tag, the score cut-off (0.15–0.85) with the best F1 on the validation moments; 0.5 for a
    tag with fewer than min_pos positives there (too few to choose from)."""
    grid = np.round(np.arange(0.15, 0.851, 0.05), 2)
    out = np.full(prob.shape[1], 0.5, np.float32)
    for j in range(prob.shape[1]):
        g = truth[:, j] >= 0.5
        if g.sum() < min_pos:
            continue
        best, bt = -1.0, 0.5
        for th in grid:
            p = prob[:, j] >= th
            tp = float((p & g).sum())
            f1 = 2 * tp / max(float(p.sum() + g.sum()), 1.0)
            if f1 > best + 1e-9:
                best, bt = f1, th
        out[j] = bt
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
    """Weighted BCE on soft targets, rarer tags weighted up (√ negatives/positives, at most ×8),
    ±1-step time jitter, early stop on validation loss. Returns (model, E on device, history)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    E_t = torch.as_tensor(ds["E"], device=dev)               # float16 on the GPU (about 1–2 GB)
    y_t = torch.as_tensor(ds["y"], device=dev)
    w_t = torch.as_tensor(ds["w"], device=dev)
    idx = ds["idx"]
    pos = (ds["y"][train_rows] >= 0.5).sum(0).astype(np.float64)
    neg = len(train_rows) - pos
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
                l = lossf(model(x), y_t[r]).mean(1) * w_t[r]
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
            l = (lossf(model(x), y_t[r]).mean(1) * w_t[r]).sum() / w_t[r].sum()
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
    print(f"Data: {len(ds['videos'])} videos ({len(train_v)} training, {len(val_v)} validation, "
          f"{len(set(vids[test_rows]))} test) · {len(ds['tags'])} tags · "
          f"{int((~ds['is_label'][train_rows]).sum())} teacher moments + "
          f"{int(ds['is_label'][train_rows].sum())} of your labels to learn from · "
          f"{len(test_rows)} labelled test frames · {ds['E'].shape[0]:,} embeddings "
          f"({ds['E'].nbytes / 1e9:.1f} GB)")
    if ds["skipped"]:
        print(f"   skipped {len(ds['skipped'])} video(s) with unreadable embeddings: {', '.join(ds['skipped'][:5])}")
    return vocab, labels, test, ds, train_rows, val_rows, test_rows


def train(name: str, epochs: int, label_weight: float, vs: str) -> int:
    import torch
    vocab, labels, test, ds, train_rows, val_rows, test_rows = _prepare(label_weight)
    dev = _device()
    print(f"Training on {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'the CPU (no CUDA GPU found)'}…",
          flush=True)
    t0 = time.time()
    model, E_t, hist = train_model(ds, train_rows, val_rows, epochs=epochs)
    minutes = (time.time() - t0) / 60
    val_prob = predict(model, E_t, ds["idx"][val_rows])
    thresholds = pick_thresholds(val_prob, ds["y"][val_rows]) if len(val_rows) else np.full(len(ds["tags"]), 0.5)
    out = WORK / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "head.pt")
    meta = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "tags": ds["tags"],
            "thresholds": [float(x) for x in thresholds], "dim": int(ds["E"].shape[1]), "half": HALF,
            "step": STEP, "label_weight": label_weight, "epochs_run": len(hist), "history": hist,
            "train_minutes": round(minutes, 2), "videos": len(ds["videos"]), "test_videos": sorted(test),
            "labels_newest": max((i.get("labelled_at") or "" for i in labels), default="")}
    print(f"Trained in {minutes:.1f} min.", flush=True)
    rc = _score(model, E_t, ds, test_rows, thresholds, vocab, labels, vs, name, meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Model saved → {out}")
    return rc


def _score(model, E_t, ds, test_rows, thresholds, vocab, labels, vs, name, meta) -> int:
    if not len(test_rows):
        print("No labelled test-set frames to score on.")
        return 0
    prob = predict(model, E_t, ds["idx"][test_rows])
    runs = teacher_runs()
    ref = pick_reference(runs, vs)
    if vs and ref is None:
        print(f"⚠️ No teacher test run called '{vs}' — scoring the student alone.")
    md, res = score_report(ds, prob, test_rows, thresholds, vocab, labels, ref, name)
    meta["scores"] = {k: v["overall"] for k, v in res.items()}
    meta["reference_run"] = ref["id"] if ref else None
    report = HERE / "student_report.md"
    report.write_text(md, encoding="utf-8")
    (WORK / "models" / name / "student_report.md").write_text(md, encoding="utf-8")
    print("\n" + "\n".join(md.splitlines()[:14]) + f"\n\nFull report (per tag) → {report}")
    return 0


def score(name: str, vs: str, label_weight: float) -> int:
    import torch
    out = WORK / "models" / name
    meta = _read_json(out / "meta.json")
    if not meta:
        print(f"No model '{name}' in {WORK / 'models'}.")
        return 1
    vocab, labels, test, ds, _, _, test_rows = _prepare(label_weight)
    if ds["tags"] != meta["tags"]:
        print("⚠️ Your vocabulary has changed since this model was trained — retrain it (train) to score "
              "it on today's tags.")
        return 1
    model = make_model(meta["dim"], len(meta["tags"])).to(_device())
    model.load_state_dict(torch.load(out / "head.pt", map_location=_device()))
    E_t = torch.as_tensor(ds["E"], device=_device())
    rc = _score(model, E_t, ds, test_rows, np.asarray(meta["thresholds"], np.float32), vocab, labels, vs,
                name, meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train / score the student tag model on the laptop.")
    ap.add_argument("command", choices=["fetch", "train", "score", "all"])
    ap.add_argument("--name", default="v1", help="model name (default v1)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--label-weight", type=float, default=LABEL_WEIGHT,
                    help=f"how many teacher answers one of your labels counts as (default {LABEL_WEIGHT:g})")
    ap.add_argument("--vs", default="", help="teacher test run id or label to compare with (default: newest try2)")
    a = ap.parse_args(argv)
    WORK.mkdir(exist_ok=True)
    if a.command in ("fetch", "all"):
        rc = fetch()
        if rc or a.command == "fetch":
            return rc
    if a.command in ("train", "all"):
        return train(a.name, a.epochs, a.label_weight, a.vs)
    return score(a.name, a.vs, a.label_weight)


if __name__ == "__main__":
    sys.exit(main())
