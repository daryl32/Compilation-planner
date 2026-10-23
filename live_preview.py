"""⚡ Real-time transition preview — the server prepares, the browser mixes.

For a transition (track A → track B, components/deck_editor) the server cuts a WINDOW of each deck around the
blend, tempo-matched to A (and B key-shifted), split into its stems (vocals / drums / bass / other) when both
tracks have them — else the full song — and saves each as a small FLAC. The browser loads those and mixes them
live with Web Audio, with the same per-stem curves as core/stem_mix.py, so dragging B, the mix point, the
blend length or the curves is heard straight away — no render.

Windows are snapped to a 16-bar grid so small drags reuse what's prepared; a drag that leaves the window prepares
a new one (a few seconds). Decoded windows are kept in RAM (an LRU of up to RAM_MB) as well as on disk.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import mixer
from core import keys as K
from core import mixing as MX

LIVE_SR = 32000            # Hz — plenty for previewing, a third smaller than 44.1 kHz
CH = 2
A_BEFORE, A_AFTER = 40, 8   # bars of A kept around the mix point
B_BEFORE, B_AFTER = 40, 24  # bars of B kept around where it's fully in
GRID = 16                   # windows snap to this many bars
RAM_MB = 1500
WORKERS = 6

_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_LOCK = threading.Lock()


def _ram_put(key: str, y: np.ndarray) -> None:
    with _LOCK:
        _CACHE[key] = y
        _CACHE.move_to_end(key)
        while sum(v.nbytes for v in _CACHE.values()) > RAM_MB * 1e6 and len(_CACHE) > 1:
            _CACHE.popitem(last=False)


def _ram_get(key: str):
    with _LOCK:
        y = _CACHE.get(key)
        if y is not None:
            _CACHE.move_to_end(key)
        return y


def out_dir() -> Path:
    d = mixer.cache_dir() / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def window(t: float, bar: float, before: int, after: int, duration: float) -> tuple:
    """(start, end) around t, snapped outward to a GRID-bar grid (so nearby points share a window)."""
    step = GRID * bar
    w0 = max(0.0, np.floor((t - before * bar) / step) * step)
    w1 = min(duration, np.ceil((t + after * bar) / step) * step)
    return round(float(w0), 3), round(float(w1), 3)


def inside(t: float, w: tuple, bar: float, margin_before: int, margin_after: int, duration: float) -> bool:
    lo = w[0] + margin_before * bar if w[0] > 0.01 else 0.0
    hi = w[1] - margin_after * bar if w[1] < duration - 0.01 else duration
    return lo <= t <= hi


def _decode_window(path: Path, w0: float, w1: float, speed: float, semitones: float) -> np.ndarray:
    """float32 (n, 2) at LIVE_SR: [w0, w1) of the file, stretched to `speed` and shifted."""
    key = json.dumps([str(path), w0, w1, round(speed, 6), semitones])
    y = _ram_get(key)
    if y is not None:
        return y
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{w0:.3f}", "-t", f"{max(0.05, w1 - w0):.3f}",
           "-i", str(path), "-map", "0:a:0"]
    f = mixer.stretch_filter(speed, semitones)
    if f:
        cmd += ["-af", f]
    cmd += ["-ac", str(CH), "-ar", str(LIVE_SR), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't read {path.name}: {proc.stderr.decode(errors='ignore')[-200:]}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, CH).copy()
    _ram_put(key, y)
    return y


def _flac(y: np.ndarray, out: Path) -> None:
    if out.exists():
        return
    tmp = out.with_suffix(".tmp.flac")
    proc = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(LIVE_SR), "-ac", str(CH),
                           "-i", "-", "-c:a", "flac", "-sample_fmt", "s16", str(tmp)],
                          input=np.clip(y, -1, 1).astype(np.float32).tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore")[-200:])
    tmp.replace(out)


def parts_for(ta: dict, tb: dict) -> tuple:
    """The lanes to mix: the four stems when both tracks have them, else each full song."""
    if mixer.has_four_stems(ta) and mixer.has_four_stems(tb):
        return mixer.FOUR_STEMS
    return ("full",)


def needed_files(ta: dict, tb: dict) -> dict:
    """{track_id: {parts}} the server needs locally (for _audio_ready)."""
    parts = set(parts_for(ta, tb))
    return {ta.get("track_id"): parts, tb.get("track_id"): parts}


def prepare(a_id: str, b_id: str, ta: dict, tb: dict, state: dict, have: dict = None) -> dict:
    """The live preview for this transition: windows (reusing `have` — the last payload — while the mix point
    and B's point are still inside it), the FLAC files and the levels. Returns the payload the browser loads:
    {"sr", "mode", "speed_b", "a": {"w0", "w1", "urls_paths": {part: path}, "gain_db"}, "b": {...}, "key"}."""
    ga, gb = MX.grid(ta), MX.grid(tb)
    target = float(ga["bpm"])
    speed_b, _ = MX.tempo_factor(gb["bpm"], target)
    shift_b = float(K.best_shift(ta.get("key"), tb.get("key"))) if a_id != b_id else 0.0
    parts = parts_for(ta, tb)
    a_end, b_full = float(state["a_end"]), float(state["b_full"])
    reuse_a = have and have.get("a_id") == a_id and inside(a_end, tuple(have["a"]["w"]), ga["bar_len"],
                                                            int(state["bars"]) + 4, 2, ga["duration"])
    reuse_b = have and have.get("b_id") == b_id and abs(have.get("speed_b", 0) - speed_b) < 1e-6 and \
        inside(b_full, tuple(have["b"]["w"]), gb["bar_len"], int(state["bars"]) + 4, 8, gb["duration"])
    wa = tuple(have["a"]["w"]) if reuse_a else window(a_end, ga["bar_len"], A_BEFORE, A_AFTER, ga["duration"])
    wb = tuple(have["b"]["w"]) if reuse_b else window(b_full, gb["bar_len"], B_BEFORE, B_AFTER, gb["duration"])
    jobs = []
    for deck, tid, t, w, speed, shift in (("a", a_id, ta, wa, 1.0, 0.0), ("b", b_id, tb, wb, speed_b, shift_b)):
        for part in parts:
            path = mixer.local_path(t, part)
            name = hashlib.sha1(json.dumps([str(path), path.stat().st_mtime if path.exists() else 0, w,
                                            round(speed, 6), shift, LIVE_SR]).encode()).hexdigest()[:20] + ".flac"
            jobs.append((deck, part, path, w, speed, shift, out_dir() / name))

    def run(job):
        deck, part, path, w, speed, shift, out = job
        y = _decode_window(path, w[0], w[1], speed, shift)
        _flac(y, out)
        return deck, part, y, out

    with ThreadPoolExecutor(WORKERS) as ex:
        done = list(ex.map(run, jobs))
    files = {"a": {}, "b": {}}
    level = {"a": [], "b": []}
    for deck, part, y, out in done:
        files[deck][part] = str(out)
        level[deck].append(y)
    # level matching like the mixer (the two decks' summed windows, within ±LEVEL_LIMIT_DB)
    loud = {d: MX.loudness_db(np.sum(level[d], axis=0), LIVE_SR) for d in level}
    ref = max(loud.values())
    gain = {d: float(np.clip(ref - loud[d], -mixer.LEVEL_LIMIT_DB, mixer.LEVEL_LIMIT_DB)) for d in loud}
    return {"sr": LIVE_SR, "mode": "stems" if parts != ("full",) else "full", "speed_b": speed_b,
            "a_id": a_id, "b_id": b_id,
            "a": {"w": list(wa), "files": files["a"], "gain_db": round(gain["a"], 2)},
            "b": {"w": list(wb), "files": files["b"], "gain_db": round(gain["b"], 2)}}


def still_fits(have: dict, a_id: str, b_id: str, ta: dict, tb: dict, state: dict) -> bool:
    """Whether `have` (a payload) still covers this state (no new preparation needed)."""
    if not have or have.get("a_id") != a_id or have.get("b_id") != b_id:
        return False
    ga, gb = MX.grid(ta), MX.grid(tb)
    if have.get("mode") != ("stems" if parts_for(ta, tb) != ("full",) else "full"):
        return False
    return inside(float(state["a_end"]), tuple(have["a"]["w"]), ga["bar_len"], int(state["bars"]) + 4, 2, ga["duration"]) \
        and inside(float(state["b_full"]), tuple(have["b"]["w"]), gb["bar_len"], int(state["bars"]) + 4, 8, gb["duration"])
