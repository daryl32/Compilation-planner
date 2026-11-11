r"""
siglip_tune.py — the one SigLIP2 fine-tune experiment (claude/plan-nov2-2026.md, the student's stop rule).

The student has only ever seen FROZEN SigLIP2 picture embeddings. v14 and v15 showed that more labels
no longer move the High tags, so what is left is the embedding itself. This script teaches the last
few blocks of SigLIP2 your tags, on THIS laptop's GPU (RTX 4070, 8 GB), and then makes every video's
embeddings again with the tuned encoder — next to the old ones, nothing is replaced:

    _student/ft_frames/<video>/<frame>.jpg     the training pictures (cut once from the preview copies)
    _student/encoders/<name>/tuned.pt          only the weights that changed (~150 MB) + meta.json
    _student/videos/<video>/embeddings_<name>.npz    the new embeddings, same times as embeddings.npz

The student is then trained on them exactly as before:
    student_tags.bat train --name v16-ft1 --emb ft1
and its 📊 Results line is compared with v15's. Clears the noise floor → keep; doesn't → the student
is finished for now (the stop rule).

What it learns from: the same moments and the same answers the student learns from (your labels ×5,
your ▦ batch answers, the teacher's answers by its trust per tag, your Reviewer scenes) — one picture
per moment. Every one of your labelled moments is used; of the others at most --per-video per video.
🧪 Test-set videos are never used, so the test score stays fair.

How (GPU use): only the last --blocks transformer blocks, the final norm and the pooling head are
trained; the 23 blocks below them run without gradients in half precision (autocast), so memory is
a few GB at a batch of 16. Pictures are read and resized by worker threads ahead of the GPU.

Usage (from the activated .venv in scene-labeling\scripts, or siglip_tune.bat):
    python siglip_tune.py check                 does it fit and how long will it take (run this first)
    python siglip_tune.py frames                cut the training pictures (once; picks up where it stopped)
    python siglip_tune.py train --name ft1      fine-tune (keeps the best epoch on the held-out videos)
    python siglip_tune.py embed --name ft1      every video's embeddings again (picks up where it stopped)
    python siglip_tune.py all --name ft1        frames, train, embed
Options: --blocks 4, --epochs 3, --batch 16 (try 8 if the GPU runs out of memory), --lr 1e-5,
--per-video 150, --max-frames 30000, --no-augment, --only vidA vidB (embed), --redo (embed).
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_logic as TL  # noqa: E402
import student_tags as ST  # noqa: E402

EMB_MODEL = "google/siglip2-so400m-patch14-384"     # same as local_vision / modal_trial
BLOCKS = 4                 # transformer blocks trained (of 27), counted from the top
EPOCHS = 3
BATCH = 16                 # pictures per step on the GPU
ACCUM = 2                  # steps added up before the weights move → 32 pictures per update
LR = 1e-5                  # the encoder's blocks (gentle: they start from a good place)
HEAD_LR = 1e-3             # the small tag layer on top (starts from nothing)
PER_VIDEO = 150            # teacher / scene moments used per video (your labels are always all used)
MAX_FRAMES = 30000
EMB_BATCH = 32
JPEG_QUALITY = 95
SEED = 7


def frames_dir() -> Path:
    return ST.WORK / "ft_frames"


def encoder_dir(name: str) -> Path:
    return ST.WORK / "encoders" / name


def frame_index(t: float) -> int:
    """The 15-fps frame an embedding time was made from (modal_trial / local_vision: idx / POSE_FPS)."""
    return int(round(float(t) * TL.POSE_FPS))


def frame_path(video: str, frame: int) -> Path:
    return frames_dir() / video / f"{int(frame):07d}.jpg"


# ---------------------------------------------------------------------------
# Which moments to learn from
# ---------------------------------------------------------------------------

def embedding_times(ds: dict, root: Path = None) -> np.ndarray:
    """The time of every row of ds["E"] (the videos' embeddings one after another, in ds["videos"] order)."""
    root = root or ST.WORK / "videos"
    parts = []
    for v in ds["videos"]:
        et, _ = ST.video_features(root / v, False)
        parts.append(np.asarray(et, float))
    T = np.concatenate(parts) if parts else np.zeros(0)
    if len(T) != ds["E"].shape[0]:
        raise SystemExit("The embeddings changed while reading them — run it again.")
    return T


def pick_rows(ds: dict, test: set, T: np.ndarray, per_video: int = PER_VIDEO, max_frames: int = MAX_FRAMES,
              seed: int = SEED) -> dict:
    """One picture per training moment: {"row" (index into ds), "video", "frame" (15-fps index)}.
    Your full labels and ▦ batch answers are all kept; of the teacher / Reviewer-scene moments at most
    per_video per video (picked at random, the same ones every time). Two moments on the same picture
    count once (yours wins). Test-set videos are left out."""
    vids = ds["video"]
    centre = ds["idx"][:, ds["idx"].shape[1] // 2]
    frame = np.rint(T[centre] * TL.POSE_FPS).astype(np.int64)
    rank = np.where(ds["is_label"], 2, np.where(ds["is_partial"], 1, 0))
    weight = ds["w"].sum(1)
    best = {}
    for r in np.flatnonzero(~np.isin(vids, list(test)) & (weight > 0)):
        key = (vids[r], int(frame[r]))
        o = best.get(key)
        if o is None or (rank[r], weight[r]) > (rank[o], weight[o]):
            best[key] = int(r)
    rows = np.array(sorted(best.values()), np.int64)
    rng = random.Random(seed)
    keep = []
    by_video = {}
    for r in rows:
        (keep if rank[r] else by_video.setdefault(vids[r], [])).append(int(r))
    mine = len(keep)
    others = []
    for v in sorted(by_video):
        rs = by_video[v]
        rng.shuffle(rs)
        others.extend(rs[:max(0, per_video)])
    room = max(0, max_frames - mine)
    if len(others) > room:
        rng.shuffle(others)
        others = others[:room]
    rows = np.array(sorted(keep + others), np.int64)
    return {"row": rows, "video": vids[rows], "frame": frame[rows], "mine": mine, "others": len(others)}


# ---------------------------------------------------------------------------
# frames — the training pictures, cut from the preview copies exactly as the embeddings were
# ---------------------------------------------------------------------------

def _proxy(video: str):
    """(preview copy path, {"proxy": ffprobe info}) — or (None, why not)."""
    import pack_for_modal as PK
    p = PK.PROXY_DIR / f"{video}.mp4"
    if not p.exists():
        return None, f"no preview copy at {p}"
    info = PK.probe(p)
    if info.get("error"):
        return None, info["error"]
    return p, {"proxy": info}


def _decode(path: Path, man: dict, seconds: float = None):
    """(frame index, BGR picture) for the whole preview copy — the same ffmpeg command, so the same
    pictures bit for bit, as the pose / embedding step (modal_trial._pose_cmd)."""
    import modal_trial as MT
    cmd, w, h = MT._pose_cmd(str(path), man)
    if seconds:
        cmd = cmd[:-5] + ["-t", f"{seconds:g}"] + cmd[-5:]
    for i, f in enumerate(MT._frames(cmd, w, h)):
        yield i, f


def extract_frames(sel: dict) -> dict:
    """Cut every picture in `sel` that isn't on disk yet. One pass per video."""
    import cv2
    want = {}
    for v, f in zip(sel["video"], sel["frame"]):
        if not frame_path(v, f).exists():
            want.setdefault(str(v), set()).add(int(f))
    total = sum(len(s) for s in want.values())
    have = len(sel["row"]) - total
    if not total:
        print(f"All {have:,} training pictures are already cut → {frames_dir()}")
        return {"cut": 0, "missing": 0, "problems": []}
    print(f"Cutting {total:,} pictures from {len(want)} videos ({have:,} already there) → {frames_dir()}",
          flush=True)
    t0, done, problems, short = time.time(), 0, [], 0
    for k, v in enumerate(sorted(want), 1):
        path, man = _proxy(v)
        if path is None:
            problems.append(f"{v}: {man}")
            continue
        out = frames_dir() / v
        out.mkdir(parents=True, exist_ok=True)
        need = want[v]
        last = max(need)
        got = 0
        for i, f in _decode(path, man, seconds=(last + 2) / TL.POSE_FPS):
            if i in need:
                # encoded here and written by Python: OpenCV's own imwrite / imread fail without a word
                # on Windows when the folder name has letters like ó
                ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    continue
                tmp = out / f"{i:07d}.tmp.jpg"
                tmp.write_bytes(buf.tobytes())
                tmp.replace(out / f"{i:07d}.jpg")
                got += 1
        done += got
        short += len(need) - got
        if k % 5 == 0 or k == len(want):
            el = time.time() - t0
            left = el / max(done, 1) * (total - done - short)
            print(f"   {k} of {len(want)} videos · {done:,} pictures · about {left / 60:.0f} min left", flush=True)
    if problems:
        print(f"⚠️ {len(problems)} video(s) couldn't be read — their moments are left out:")
        for p in problems[:8]:
            print("   " + p)
    if short:
        print(f"   {short} picture(s) fall after the end of their preview copy — left out.")
    print(f"Cut {done:,} pictures in {(time.time() - t0) / 60:.1f} min.")
    return {"cut": done, "missing": short, "problems": problems}


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------

def load_encoder(half: bool):
    """(SigLIP2 model, its processor) — the text half is dropped (never used, ~1 GB of GPU memory)."""
    import torch
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.float16 if half else torch.float32)
    if hasattr(model, "text_model"):
        model.text_model = None
    return model, AutoProcessor.from_pretrained(EMB_MODEL)


def vision_parts(model) -> tuple:
    """(the transformer blocks, [final norm, pooling head]) of the picture half."""
    vm = getattr(model, "vision_model", model)
    vm = getattr(vm, "vision_model", vm)
    extras = [m for m in (getattr(vm, "post_layernorm", None), getattr(vm, "head", None)) if m is not None]
    return list(vm.encoder.layers), extras


def unfreeze(model, blocks: int) -> list:
    """Freeze everything, then let the last `blocks` blocks, the final norm and the pooling head learn.
    Returns the names of the weights that learn."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers, extras = vision_parts(model)
    for m in layers[len(layers) - max(1, blocks):] + extras:
        for p in m.parameters():
            p.requires_grad_(True)
    return [n for n, p in model.named_parameters() if p.requires_grad]


def image_features(model, px):
    """Unit-length picture embeddings — the same call local_vision makes."""
    import torch
    f = model.get_image_features(pixel_values=px)
    if not torch.is_tensor(f):
        f = getattr(f, "pooler_output", None) if getattr(f, "pooler_output", None) is not None else f[0]
    return torch.nn.functional.normalize(f.float(), dim=-1)


def tuned_state(model, names: list) -> dict:
    sd = model.state_dict()
    return {n: sd[n].detach().to("cpu", copy=True).half() for n in names}


def apply_tuned(model, state: dict) -> None:
    sd = model.state_dict()
    missing = [n for n in state if n not in sd]
    if missing:
        raise SystemExit(f"The saved encoder doesn't fit this model ({len(missing)} weights, e.g. {missing[0]}) — "
                         "was it trained with another transformers version?")
    model.load_state_dict({n: t.to(sd[n].dtype) for n, t in state.items()}, strict=False)


def _pixels(proc, rgb: list):
    return proc(images=rgb, return_tensors="pt")["pixel_values"]


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Feeding the GPU: pictures read, flipped / cropped and resized on worker threads, a few batches ahead
# ---------------------------------------------------------------------------

def _read_rgb(path: Path, augment: bool, rng: random.Random):
    import cv2
    try:                                           # read by Python, decoded by OpenCV (see extract_frames)
        img = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        img = None
    if img is None:
        return None
    if augment:
        h, w = img.shape[:2]
        s = rng.uniform(0.85, 1.0)                 # a random crop keeping 85–100 % of each side
        ch, cw = max(8, int(h * s)), max(8, int(w * s))
        y0, x0 = rng.randint(0, h - ch), rng.randint(0, w - cw)
        img = img[y0:y0 + ch, x0:x0 + cw]
        if rng.random() < 0.5:                     # mirrored: no tag depends on left / right
            img = img[:, ::-1]
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)


def batches(paths: list, order: np.ndarray, proc, batch: int, augment: bool, seed: int, ahead: int = 6):
    """Yields (positions in `paths`, pixel tensor), prepared `ahead` batches before the GPU needs them."""
    q = queue.Queue(maxsize=ahead)
    rng = random.Random(seed)

    def work():
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for s in range(0, len(order), batch):
                    ii = order[s:s + batch]
                    seeds = [rng.random() for _ in ii]
                    imgs = list(pool.map(lambda a: _read_rgb(paths[a[0]], augment, random.Random(a[1])),
                                         zip(ii, seeds)))
                    ok = [k for k, im in enumerate(imgs) if im is not None]
                    if ok:
                        q.put((ii[ok], _pixels(proc, [imgs[k] for k in ok])))
        except Exception as e:      # noqa: BLE001 — hand the error to the training loop
            q.put(e)
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def _selection(per_video: int, max_frames: int):
    vocab, labels, test, ds, train_rows, val_rows, _ = ST._prepare(ST.LABEL_WEIGHT)
    T = embedding_times(ds)
    sel = pick_rows(ds, test, T, per_video, max_frames)
    val_videos = set(ds["video"][val_rows])
    print(f"Moments to learn from: {len(sel['row']):,} pictures — {sel['mine']:,} with your own answers "
          f"(all of them) + {sel['others']:,} teacher / Reviewer-scene moments (at most {per_video} a video)",
          flush=True)
    return ds, sel, val_videos


def train(name: str, blocks: int = BLOCKS, epochs: int = EPOCHS, batch: int = BATCH, lr: float = LR,
          per_video: int = PER_VIDEO, max_frames: int = MAX_FRAMES, augment: bool = True,
          accum: int = ACCUM, max_steps: int = None) -> int:
    import torch
    ds, sel, val_videos = _selection(per_video, max_frames)
    paths = [frame_path(v, f) for v, f in zip(sel["video"], sel["frame"])]
    there = np.array([p.exists() for p in paths])
    if not there.any():
        print("No training pictures on disk yet — run `siglip_tune frames` first.")
        return 1
    if (~there).sum():
        print(f"   {(~there).sum():,} picture(s) aren't on disk (run `frames` again to cut them) — training "
              "on the rest.")
    rows = sel["row"]
    is_val = np.isin(sel["video"], list(val_videos))
    tr = np.flatnonzero(there & ~is_val)
    va = np.flatnonzero(there & is_val)
    if not len(tr):
        print("Nothing to train on.")
        return 1
    y_all, w_all = ds["y"][rows], ds["w"][rows]
    seen = w_all[tr] > 0
    pos = ((y_all[tr] >= 0.5) & seen).sum(0).astype(np.float64)
    neg = ((y_all[tr] < 0.5) & seen).sum(0).astype(np.float64)
    dev = _device()
    cuda = dev.type == "cuda"
    print(f"Loading SigLIP2 ({EMB_MODEL})…", flush=True)
    model, proc = load_encoder(half=False)
    model = model.to(dev).eval()                    # eval: no dropout; the trained blocks still get gradients
    names = unfreeze(model, blocks)
    params = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        dim = image_features(model, _pixels(proc, [np.zeros((64, 64, 3), np.uint8)]).to(dev)).shape[1]
    head = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, y_all.shape[1])).to(dev)
    pw = torch.as_tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1.0)), 1.0, 8.0), dtype=torch.float32, device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    opt = torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": 0.05},
                             {"params": head.parameters(), "lr": HEAD_LR, "weight_decay": 0.0}])
    steps_per = int(math.ceil(len(tr) / batch))
    updates = max(1, int(math.ceil(epochs * steps_per / accum)))
    warm = max(1, int(0.05 * updates))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: (u + 1) / warm if u < warm else 0.5 * (1 + math.cos(math.pi * (u - warm) / max(1, updates - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    y_t, w_t = torch.as_tensor(y_all), torch.as_tensor(w_all)
    n_par = sum(p.numel() for p in params)
    print(f"Training the last {blocks} blocks + pooling head ({n_par / 1e6:.0f} M weights) on "
          f"{torch.cuda.get_device_name(0) if cuda else 'the CPU (no CUDA GPU found)'} · {len(tr):,} pictures to "
          f"learn from, {len(va):,} held out · {epochs} epochs, batch {batch} × {accum}", flush=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def step_loss(ii, px):
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            logits = head(image_features(model, px.to(dev, non_blocking=True)))
        y, w = y_t[ii].to(dev), w_t[ii].to(dev)
        return (lossf(logits.float(), y) * w).sum(), w.sum()

    def validate():
        if not len(va):
            return float("nan")
        tot = ws = 0.0
        with torch.no_grad():
            for ii, px in batches(paths, va, proc, batch * 2, False, SEED):
                l, w = step_loss(ii, px)
                tot += float(l)
                ws += float(w)
        return tot / max(ws, 1e-9)

    out = encoder_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "base": EMB_MODEL, "blocks": blocks,
            "epochs": epochs, "batch": batch, "accum": accum, "lr": lr, "head_lr": HEAD_LR, "augment": augment,
            "tags": ds["tags"], "pictures": int(len(tr)), "held_out": int(len(va)), "your_answers": int(sel["mine"]),
            "weights": names, "dim": int(dim), "history": []}
    t0, best, step, stop = time.time(), float("inf"), 0, False
    try:
        for ep in range(1, epochs + 1):
            order = np.random.RandomState(SEED + ep).permutation(tr)
            run = seen_n = 0.0
            te = time.time()
            opt.zero_grad(set_to_none=True)
            for k, (ii, px) in enumerate(batches(paths, order, proc, batch, augment, SEED + ep), 1):
                l, w = step_loss(ii, px)
                loss = l / w.clamp(min=1e-6)
                scaler.scale(loss / accum).backward()
                if k % accum == 0 or k == steps_per:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                run += float(loss) * len(ii)
                seen_n += len(ii)
                step += 1
                if k % 50 == 0 or k == steps_per:
                    rate = seen_n / (time.time() - te)
                    left = ((steps_per - k) + (epochs - ep) * steps_per) * batch / max(rate, 1e-6)
                    print(f"   epoch {ep} · {k:,} of {steps_per:,} steps · loss {run / seen_n:.4f} · "
                          f"{rate:.0f} pictures/s · about {left / 60:.0f} min of training left", flush=True)
                if max_steps and step >= max_steps:
                    stop = True
                    break
            vl = validate()
            meta["history"].append({"epoch": ep, "train_loss": round(run / max(seen_n, 1), 4),
                                    "val_loss": round(vl, 4)})
            better = vl < best - 1e-4 or math.isnan(vl)
            print(f" epoch {ep}: train {run / max(seen_n, 1):.4f} · held-out videos {vl:.4f}"
                  + (" — best so far, saved" if better else " — no better, the earlier epoch is kept"), flush=True)
            if better:
                best = vl
                tmp = out / "tuned.tmp.pt"
                torch.save({"encoder": tuned_state(model, names),
                            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, tmp)
                tmp.replace(out / "tuned.pt")
                meta["best_epoch"] = ep
            if stop:
                break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nThe GPU ran out of memory at a batch of {batch}. Close other GPU apps and run it again "
                  f"with --batch {max(2, batch // 2)} (the result is the same, a little slower).")
            return 2
        raise
    meta["train_minutes"] = round((time.time() - t0) / 60, 1)
    if cuda:
        meta["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        meta["gpu"] = torch.cuda.get_device_name(0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Encoder '{name}' saved → {out}  ({meta['train_minutes']} min"
          + (f", peak GPU memory {meta['peak_gpu_gb']} GB" if cuda else "") + f"; best epoch {meta.get('best_epoch')})")
    return 0


# ---------------------------------------------------------------------------
# embed — every video's embeddings again with the tuned encoder
# ---------------------------------------------------------------------------

def load_tuned(name: str):
    """(model on the GPU in half precision, processor) with encoder `name`'s weights in place;
    name "" = the untouched SigLIP2."""
    import torch
    dev = _device()
    model, proc = load_encoder(half=dev.type == "cuda")
    if name:
        f = encoder_dir(name) / "tuned.pt"
        if not f.exists():
            raise SystemExit(f"No encoder '{name}' in {ST.WORK / 'encoders'} — train it first.")
        apply_tuned(model, torch.load(f, map_location="cpu")["encoder"])
    return model.to(dev).eval(), proc


def embed_video(model, proc, path: Path, man: dict, times: np.ndarray, seconds: float = None) -> tuple:
    """Embeddings for the pictures at `times` (float16, in that order) + how many had to borrow a
    neighbour's (their picture lies after the end of the preview copy)."""
    import cv2
    import torch
    dev = next(model.parameters()).device
    half = next(model.parameters()).dtype == torch.float16
    idx = np.array([frame_index(t) for t in times], np.int64)
    want = set(idx.tolist())
    got, bf, bi = {}, [], []

    def flush():
        if not bf:
            return
        with torch.no_grad():
            px = _pixels(proc, [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bf]).to(dev)
            f = image_features(model, px.half() if half else px)
        for i, e in zip(bi, f.cpu().numpy().astype(np.float16)):
            got[i] = e
        bf.clear()
        bi.clear()

    for i, f in _decode(path, man, seconds=seconds):
        if i in want:
            bf.append(f.copy())
            bi.append(i)
            if len(bf) >= EMB_BATCH:
                flush()
    flush()
    if not got:
        return None, len(idx)
    have = np.array(sorted(got))
    out, borrowed = [], 0
    for i in idx:
        if i not in got:
            borrowed += 1
            i = int(have[np.argmin(np.abs(have - i))])
        out.append(got[i])
    return np.stack(out), borrowed


def embed(name: str, only: list = None, redo: bool = False) -> int:
    root = ST.WORK / "videos"
    vids = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "embeddings.npz").exists()) \
        if root.exists() else []
    if only:
        vids = [v for v in vids if v in set(only)]
    fname = ST.emb_file(name)
    todo = [v for v in vids if redo or not (root / v / fname).exists()]
    if not todo:
        print(f"Every video already has {fname} (--redo to make them again).")
        return 0
    print(f"Loading SigLIP2 with encoder '{name}'…", flush=True)
    model, proc = load_tuned(name)
    print(f"Embedding {len(todo)} of {len(vids)} videos again ({len(vids) - len(todo)} already done)…", flush=True)
    t0, secs, problems = time.time(), 0.0, []
    for k, v in enumerate(todo, 1):
        path, man = _proxy(v)
        if path is None:
            problems.append(f"{v}: {man}")
            continue
        with np.load(root / v / "embeddings.npz") as z:
            times = np.asarray(z["frame_times"], float)
        emb, borrowed = embed_video(model, proc, path, man, times)
        if emb is None:
            problems.append(f"{v}: no pictures could be read from the preview copy")
            continue
        tmp = root / v / f"embeddings_{name}.tmp.npz"
        np.savez_compressed(tmp, frame_times=times, frame_embs=emb, encoder=name, base=EMB_MODEL)
        tmp.replace(root / v / fname)
        secs += float(man["proxy"].get("duration") or 0.0)
        el = time.time() - t0
        print(f"[{k}/{len(todo)}] {v}: {len(times):,} embeddings"
              + (f" ({borrowed} past the end of the file — neighbour's used)" if borrowed else "")
              + f" · {el / 60:.0f} min so far, about {el / k * (len(todo) - k) / 60:.0f} min left", flush=True)
    if problems:
        print(f"\n⚠️ {len(problems)} video(s) have no new embeddings — the student stops until they do:")
        for p in problems[:10]:
            print("   " + p)
        return 1
    print(f"\nDone: {len(todo)} videos ({secs / 3600:.1f} h of footage) in {(time.time() - t0) / 60:.0f} min.\n"
          f"Next:  student_tags.bat train --name v16-{name} --emb {name}")
    return 0


# ---------------------------------------------------------------------------
# check — does it fit, is the picture path the same as before, how long will it take
# ---------------------------------------------------------------------------

def check(blocks: int, batch: int, per_video: int, max_frames: int) -> int:
    import torch
    dev = _device()
    cuda = dev.type == "cuda"
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} · {total:.1f} GB")
    else:
        print("⚠️ No CUDA GPU found — this needs the laptop's RTX 4070 (GPU torch in the venv).")
    import transformers
    print(f"torch {torch.__version__} · transformers {transformers.__version__}\n")

    # 1. the same pictures as before? Untouched SigLIP2 on the first 2 minutes of one video vs embeddings.npz
    root = ST.WORK / "videos"
    vids = sorted(d.name for d in root.iterdir() if (d / "embeddings.npz").exists()) if root.exists() else []
    print("1. Same pictures as the embeddings you have? (untouched SigLIP2, first 2 minutes of one video)", flush=True)
    model, proc = load_tuned("")
    rate_e, same = None, None
    for v in vids:
        path, man = _proxy(v)
        if path is None:
            continue
        with np.load(root / v / "embeddings.npz") as z:
            times, old = np.asarray(z["frame_times"], float), np.asarray(z["frame_embs"], np.float32)
        keep = np.flatnonzero(times < 118.0)
        if len(keep) < 20:
            continue
        t0 = time.time()
        new, _ = embed_video(model, proc, path, man, times[keep], seconds=120.0)
        if new is None:
            continue
        rate_e = len(keep) / (time.time() - t0)
        a = old[keep] / np.maximum(np.linalg.norm(old[keep], axis=1, keepdims=True), 1e-6)
        same = float((a * new.astype(np.float32)).sum(1).mean())
        print(f"   {v}: likeness {same:.4f} (1 = identical) · {rate_e:.0f} embeddings/s incl. decoding")
        break
    if same is None:
        print("   ⚠️ couldn't read any preview copy — check PROXY_DIR in config.py.")
    elif same < 0.98:
        print("   ⚠️ lower than expected (≥ 0.99) — send this output to Claude before the long run.")
    else:
        print("   ✅ the picture path matches.")
    del model
    if cuda:
        torch.cuda.empty_cache()

    # 2. does training fit, and how fast is it
    print(f"\n2. Does training fit? (last {blocks} blocks, batch {batch}, 8 steps on blank pictures)", flush=True)
    model, proc = load_encoder(half=False)
    model = model.to(dev).eval()
    names = unfreeze(model, blocks)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    px = _pixels(proc, [np.random.RandomState(k).randint(0, 255, (360, 640, 3), np.uint8) for k in range(batch)])
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    rate_t = None
    try:
        for k in range(8):
            if k == 2:
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = image_features(model, px.to(dev)).pow(2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        rate_t = 6 * batch / (time.time() - t0)
        peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        print(f"   ✅ fits: peak GPU memory {peak:.1f} GB · {rate_t:.0f} pictures/s · {len(names)} weight groups learn")
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"   ❌ out of GPU memory at a batch of {batch} — use --batch {max(2, batch // 2)} on every command.")
    del model, opt
    if cuda:
        torch.cuda.empty_cache()

    # 3. how long
    print("\n3. How long will it take?", flush=True)
    ds, sel, val_videos = _selection(per_video, max_frames)
    hours = 0.0
    for v in vids:
        try:
            with np.load(root / v / "embeddings.npz") as z:
                hours += float(np.max(z["frame_times"])) / 3600
        except (OSError, KeyError, ValueError):
            pass
    n_emb = hours * 3600 * TL.POSE_FPS / int(round(TL.POSE_FPS / 2))
    print(f"   {len(vids)} videos with embeddings · {hours:.1f} h of footage")
    if rate_t:
        print(f"   train:  {len(sel['row']):,} pictures × {EPOCHS} epochs ≈ "
              f"{len(sel['row']) * EPOCHS / rate_t / 60 * 1.15:.0f} min")
    if rate_e:
        print(f"   embed:  {n_emb:,.0f} embeddings ≈ {n_emb / rate_e / 60:.0f} min "
              f"(frames: a decode-only pass, well under that)")
    print("\nNext:  siglip_tune.bat all --name ft1")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune SigLIP2's last blocks on your tags (laptop GPU).")
    ap.add_argument("command", choices=["check", "frames", "train", "embed", "all"])
    ap.add_argument("--name", default="ft1", help="encoder name (default ft1)")
    ap.add_argument("--blocks", type=int, default=BLOCKS, help=f"transformer blocks trained (default {BLOCKS} of 27)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH, help=f"pictures per GPU step (default {BATCH}; 8 if memory runs out)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--per-video", type=int, default=PER_VIDEO,
                    help=f"teacher / scene moments used per video (default {PER_VIDEO}; your labels are all used)")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--no-augment", action="store_true", help="no random crop / mirror while training")
    ap.add_argument("--only", nargs="*", default=None, help="embed: just these video ids")
    ap.add_argument("--redo", action="store_true", help="embed: make them again even if done")
    ap.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    ST.WORK.mkdir(exist_ok=True)
    if a.command == "check":
        return check(a.blocks, a.batch, a.per_video, a.max_frames)
    if a.command in ("frames", "all"):
        ds, sel, _ = _selection(a.per_video, a.max_frames)
        extract_frames(sel)
        del ds
        if a.command == "frames":
            return 0
    if a.command in ("train", "all"):
        accum = max(1, round(BATCH * ACCUM / max(1, a.batch)))      # 32 pictures per update whatever the batch
        rc = train(a.name, a.blocks, a.epochs, a.batch, a.lr, a.per_video, a.max_frames, not a.no_augment,
                   accum, a.max_steps)
        if rc or a.command == "train":
            return rc
    return embed(a.name, a.only, a.redo)


if __name__ == "__main__":
    sys.exit(main())
