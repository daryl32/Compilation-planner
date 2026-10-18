"""Motion curves as compact numpy arrays, cached on local disk.

A video catalogue keeps each scene's frame-by-frame motion curve as a JSON list
of numbers. Parsed into Python that's ~32 bytes per number, and the whole
catalogue has to be parsed again whenever a curve is needed. This keeps, per
video, one .npy file with every scene's values (float64 — exactly the numbers
the JSON holds, so matching results don't change) plus a small index:

    <cache>/<video_id>.npy         all scenes' values, back to back
    <cache>/<video_id>.idx.json    {"source": [mtime, size], "scenes": [[scene_id, fps, start, length], …]}

Loading is np.load with memory-mapping on Linux: nothing is parsed, the server
only reads the parts actually used, and the live app and the test copy share the
same pages in memory. (Windows can't replace a memory-mapped file, so there the
array is simply read into memory.) The cache rebuilds itself when the catalogue
changes (its modified time or size), so it can be deleted at any time.

It lives OUTSIDE the catalogue folder (on the laptop that folder is Google
Drive, and a cache must never sync): ~/.cache/media-planner/curves-<hash of the
catalogue folder>, or CURVE_CACHE_DIR in config.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

USE_MMAP = os.name != "nt"


def default_cache_dir(catalogue_dir: Path) -> Path:
    try:
        import config
        explicit = getattr(config, "CURVE_CACHE_DIR", None)
    except ImportError:
        explicit = None
    if explicit:
        return Path(explicit)
    tag = hashlib.sha1(str(Path(catalogue_dir).resolve()).encode()).hexdigest()[:10]
    return Path.home() / ".cache" / "media-planner" / f"curves-{tag}"


def _source_sig(path: Path):
    st = path.stat()
    return [st.st_mtime, st.st_size]


def _build(catalogue_file: Path, npy: Path, idx: Path, sig) -> tuple:
    cat = json.loads(catalogue_file.read_text())
    chunks, scenes, pos = [], [], 0
    for s in cat.get("scenes", []):
        c = s.get("motion_curve")
        if not c:
            continue
        vals = np.asarray(c.get("values") or [], dtype=float)
        scenes.append([s["scene_id"], c.get("fps"), pos, int(len(vals))])   # scene_id keeps its JSON type
        chunks.append(vals)
        pos += len(vals)
    data = np.concatenate(chunks) if chunks else np.zeros(0)
    npy.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = npy.with_name(npy.name + f".{os.getpid()}.tmp")
    tmp_idx = idx.with_name(idx.name + f".{os.getpid()}.tmp")
    with open(tmp_npy, "wb") as f:
        np.save(f, data)
    tmp_idx.write_text(json.dumps({"source": sig, "scenes": scenes}))
    os.replace(tmp_npy, npy)          # data first, index last: a reader never sees a new index
    os.replace(tmp_idx, idx)          # pointing into old data
    return data, scenes


def load(catalogue_file: Path, cache_dir: Path) -> dict:
    """{scene_id: {"fps": fps, "values": read-only float64 array}} for every scene
    that has a motion curve. Empty dict if the catalogue can't be read."""
    catalogue_file = Path(catalogue_file)
    try:
        sig = _source_sig(catalogue_file)
    except OSError:
        return {}
    stem = catalogue_file.stem
    npy, idx = Path(cache_dir) / f"{stem}.npy", Path(cache_dir) / f"{stem}.idx.json"
    data = scenes = None
    try:
        meta = json.loads(idx.read_text())
        if meta.get("source") == sig:
            data = np.load(npy, mmap_mode="r" if USE_MMAP else None)
            scenes = meta["scenes"]
    except (OSError, ValueError, KeyError):
        data = None
    if data is None:
        try:
            data, scenes = _build(catalogue_file, npy, idx, sig)
        except (OSError, ValueError):
            try:      # can't write the cache (read-only disk?) — read the curves straight from the JSON
                cat = json.loads(catalogue_file.read_text())
            except (OSError, ValueError):
                return {}
            return {s["scene_id"]: {"fps": s["motion_curve"].get("fps"),
                                    "values": np.asarray(s["motion_curve"].get("values") or [], dtype=float)}
                    for s in cat.get("scenes", []) if s.get("motion_curve")}
    if isinstance(data, np.ndarray) and data.flags.writeable:
        data.flags.writeable = False      # shared between pages — read only
    out = {}
    for sid, fps, start, length in scenes:
        out[sid] = {"fps": fps, "values": data[start:start + length]}
    return out


def clear(cache_dir: Path) -> int:
    """Delete the cache (it rebuilds itself). Returns bytes freed."""
    freed = 0
    for p in Path(cache_dir).glob("*"):
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    return freed
