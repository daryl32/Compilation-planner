"""Motion curves as compact numpy arrays, cached on local disk.

A video catalogue keeps each scene's frame-by-frame motion curve as a JSON list
of numbers. Parsed into Python that's ~32 bytes per number, and the whole
catalogue has to be parsed again whenever a curve is needed. This keeps, per
video, one .npy file with every scene's values (float64 — exactly the numbers
the JSON holds, so matching results don't change) plus a small index:

    <cache>/<video_id>.<sig>.npy   all scenes' values, back to back
    <cache>/<video_id>.idx.json    {"source": [mtime_ns, size, inode], "data": "<video_id>.<sig>.npy",
                                    "scenes": [[scene_id, fps, start, length], …]}

The data file's name carries the catalogue signature and the index names it,
so a reader that has an index always opens the matching data — never a newer
file another process has just written.

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
    return [st.st_mtime_ns, st.st_size, st.st_ino]


def _sig_tag(sig) -> str:
    return hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:12]


def _build(catalogue_file: Path, idx: Path, sig) -> tuple:
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
    stem = catalogue_file.stem
    npy = idx.with_name(f"{stem}.{_sig_tag(sig)}.npy")
    npy.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = npy.with_name(npy.name + f".{os.getpid()}.tmp")
    tmp_idx = idx.with_name(idx.name + f".{os.getpid()}.tmp")
    with open(tmp_npy, "wb") as f:
        np.save(f, data)
    tmp_idx.write_text(json.dumps({"source": sig, "data": npy.name, "scenes": scenes}))
    os.replace(tmp_npy, npy)          # data first, then the index that names it
    os.replace(tmp_idx, idx)
    for old in idx.parent.glob(f"{stem}.*.npy"):     # earlier versions of this video's data
        tag = old.name[len(stem) + 1:-len(".npy")]
        if old.name != npy.name and len(tag) == 12 and all(ch in "0123456789abcdef" for ch in tag):
            try:
                old.unlink()          # on Linux a page still reading it keeps it until done
            except OSError:
                pass                  # Windows: in use — removed on a later rebuild
    return data, scenes


def _from_json(catalogue_file: Path) -> dict:
    """The curves read straight from the catalogue (no cache)."""
    try:
        cat = json.loads(catalogue_file.read_text())
    except (OSError, ValueError):
        return {}
    return {s["scene_id"]: {"fps": s["motion_curve"].get("fps"),
                            "values": np.asarray(s["motion_curve"].get("values") or [], dtype=float)}
            for s in cat.get("scenes", []) if s.get("motion_curve")}


def load(catalogue_file: Path, cache_dir: Path) -> dict:
    """{scene_id: {"fps": fps, "values": read-only float64 array}} for every scene
    that has a motion curve. Empty dict if the catalogue can't be read."""
    catalogue_file = Path(catalogue_file)
    try:
        sig = _source_sig(catalogue_file)
    except OSError:
        return {}
    idx = Path(cache_dir) / f"{catalogue_file.stem}.idx.json"
    for attempt in (0, 1):
        data = scenes = None
        if attempt == 0:
            try:
                meta = json.loads(idx.read_text())
                if meta.get("source") == sig:
                    data = np.load(idx.parent / meta["data"], mmap_mode="r" if USE_MMAP else None)
                    scenes = meta["scenes"]
            except (OSError, ValueError, KeyError, TypeError):
                data = None
        if data is None:
            try:
                data, scenes = _build(catalogue_file, idx, sig)
            except (OSError, ValueError, KeyError, TypeError):
                return _from_json(catalogue_file)   # can't write the cache (read-only disk?)
        if isinstance(data, np.ndarray) and data.flags.writeable:
            data.flags.writeable = False      # shared between pages — read only
        try:
            out = {}
            for sid, fps, start, length in scenes:
                if start < 0 or start + length > len(data):
                    raise ValueError("index doesn't fit the data")
                out[sid] = {"fps": fps, "values": data[start:start + length]}
            return out
        except (ValueError, TypeError):
            continue                          # damaged index → rebuild once
    return _from_json(catalogue_file)


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
