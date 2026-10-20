"""🎚️ Mixer — renders a mix recipe (core/mixing.py) to an audio file with ffmpeg.

Also works out the per-bar sound features core/mixing.py uses to find seamless
jumps, and fetches source tracks from Google Drive when they aren't on this
machine. No Streamlit in here.

Rendering, in short:
  1. each source track is decoded once (stereo, 44.1 kHz), already stretched
     to the mix tempo and pitch-shifted if asked (ffmpeg rubberband; atempo /
     asetrate when this ffmpeg has no rubberband),
  2. its loudness is measured so tracks can be levelled against each other,
  3. the pieces are laid end to end with a crossfade centred on every join's
     bar line (straight for jumps within one track — the two sides are nearly
     the same audio — equal-power between different tracks),
  4. faded, kept below clipping, and encoded to .m4a (AAC).
Audio-only ffmpeg work: seconds per track, fine on the server.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from core import mixing as MX

SR = 44100
CH = 2
FEAT_VERSION = 1
LEVEL_LIMIT_DB = 9.0          # level matching never moves a track more than this
PEAK = 0.97


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    """Local, never synced: bar features and preview renders."""
    from config import CATALOGUE_DIR
    from curve_cache import default_cache_dir
    base = default_cache_dir(CATALOGUE_DIR)
    d = base.with_name(base.name.replace("curves", "mix", 1) if "curves" in base.name else base.name + "-mix")
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_path(track: dict) -> Path:
    from render_preview import colab_to_local
    return Path(colab_to_local(track.get("source_path") or ""))


def ensure_local(track: dict, progress=None) -> str | None:
    """Make sure the track's audio file is on this machine (downloads it from
    Google Drive when it isn't). Returns an error message, or None."""
    path = local_path(track)
    if path.exists():
        return None
    src = track.get("source_path") or ""
    if "MyDrive/" not in src:
        return f"{track.get('track_id')}: audio file not found ({path})"
    try:
        from drive_sync import download_source_video
    except Exception as e:
        return f"Google Drive isn't available here: {e}"
    err = download_source_video(src.split("MyDrive/", 1)[1], path, progress_callback=progress)
    return f"{track.get('track_id')}: download failed — {err}" if err else None


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def has_rubberband() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20)
        return " rubberband " in r.stdout
    except Exception:
        return False


def _atempo_chain(speed: float) -> list:
    out = []
    while speed > 2.0:
        out.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        out.append("atempo=0.5")
        speed /= 0.5
    if abs(speed - 1.0) > 1e-4:
        out.append(f"atempo={speed:.6f}")
    return out


def stretch_filter(speed: float, semitones: float = 0.0) -> str | None:
    """The ffmpeg audio filter for a tempo change (pitch kept) plus an optional
    pitch shift; None when nothing changes."""
    pitch = 2.0 ** (float(semitones) / 12.0)
    if abs(speed - 1.0) < 1e-4 and abs(pitch - 1.0) < 1e-4:
        return None
    if has_rubberband():
        return f"rubberband=tempo={speed:.6f}:pitch={pitch:.6f}"
    parts = []
    if abs(pitch - 1.0) >= 1e-4:
        # resample trick: play at sr·pitch (higher and faster), back to sr, slow back down
        parts += [f"asetrate={SR * pitch:.3f}", f"aresample={SR}"]
        speed = speed / pitch
    parts += _atempo_chain(speed)
    return ",".join(parts) or None


def decode(path: Path, speed: float = 1.0, semitones: float = 0.0) -> np.ndarray:
    """Whole file as float32 (n, 2) at 44.1 kHz, stretched / shifted.
    -map 0:a:0 skips embedded album art in .m4a files."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0"]
    f = stretch_filter(speed, semitones)
    if f:
        cmd += ["-af", f]
    cmd += ["-ac", str(CH), "-ar", str(SR), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read {path.name}: {proc.stderr.decode(errors='ignore')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, CH).copy()


@functools.lru_cache(maxsize=1)
def has_mp3() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
        return "libmp3lame" in r.stdout
    except Exception:
        return False


def encode(y: np.ndarray, out_path: Path, bitrate: str = "256k") -> None:
    """.m4a = AAC at bitrate (the saved track); .mp3 = a quick preview (several
    times faster to encode than AAC)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".part" + out_path.suffix)
    if out_path.suffix == ".mp3":
        codec = ["-c:a", "libmp3lame", "-q:a", "4"]
    else:
        codec = ["-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart"]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(SR), "-ac", str(CH), "-i", "-",
           *codec, str(tmp)]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(y, dtype=np.float32).tobytes(),
                          capture_output=True, check=False)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not write the mix: {proc.stderr.decode(errors='ignore')[-300:]}")
    tmp.replace(out_path)


# ---------------------------------------------------------------------------
# Bar features (for seamless jumps)
# ---------------------------------------------------------------------------

def bar_features(track: dict, g: dict = None) -> np.ndarray:
    """One row per bar: average timbre (MFCC), harmony (chroma) and loudness.
    Cached on local disk per audio file + bar grid."""
    import audio_analysis as AA
    g = g or MX.grid(track)
    path = local_path(track)
    st = path.stat()
    sig = hashlib.sha1(json.dumps([FEAT_VERSION, st.st_size, int(st.st_mtime),
                                   [round(float(x), 2) for x in g["bars"]]]).encode()).hexdigest()[:12]
    cache = cache_dir() / "bars" / f"{track.get('track_id', path.stem)}.{sig}.npy"
    try:
        return np.load(cache)
    except (OSError, ValueError):
        pass
    y = AA.decode_audio(path)
    mag = AA.stft_mag(y)
    fb, _ = AA.mel_filterbank()
    logmel = np.log1p(1000.0 * (mag ** 2 @ fb.T) / ((mag ** 2 @ fb.T).max() + 1e-12))
    mfcc = AA.mfcc_from_logmel(logmel)
    chroma = AA.chroma_from_stft(mag)
    rms = AA.frame_rms(y)[:len(mag)]
    loud = np.log1p(50 * rms / (rms.max() + 1e-9))
    frames = np.clip(np.round(np.asarray(g["bars"]) * AA.FPS).astype(int), 0, len(mag) - 1)
    bounds = list(frames) + [len(mag)]
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        b = max(b, a + 1)
        rows.append(np.concatenate([mfcc[a:b].mean(axis=0), 2.0 * chroma[a:b].mean(axis=0),
                                    [3.0 * loud[a:b].mean()]]))
    feats = np.asarray(rows, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    for old in cache.parent.glob(f"{track.get('track_id', path.stem)}.*.npy"):
        old.unlink(missing_ok=True)
    np.save(cache, feats)
    return feats


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _ramp(n: int, equal_power: bool, rising: bool) -> np.ndarray:
    x = (np.arange(n, dtype=np.float32) + 0.5) / max(n, 1)
    if not rising:
        x = 1.0 - x
    return np.sin(x * np.pi / 2).astype(np.float32) if equal_power else x


def render(recipe: dict, tracks: dict, out_path: Path, progress=None, bitrate: str = "256k") -> dict:
    """Render recipe to out_path. tracks: {track_id: catalogue track} for every
    track the recipe uses (their audio files must be local — ensure_local).
    Returns {"duration", "bpm", "beats", "bars", "energy" (30 Hz curve),
    "warnings", "layout"}."""
    say = progress or (lambda msg: None)
    grids = {tid: MX.grid(t) for tid, t in tracks.items()}
    lay = MX.layout(recipe, grids)
    if not lay["pieces"]:
        raise ValueError("The mix has no pieces yet.")
    shifts = recipe.get("shift") or {}
    warnings = []
    if (any(abs(p["speed"] - 1) > 1e-4 for p in lay["pieces"]) or any(shifts.values())) and not has_rubberband():
        warnings.append("This server's ffmpeg has no rubberband filter — using the simpler atempo stretch "
                        "(fine for small changes).")

    # 1. decode each (track, speed, shift) once
    audio, loud = {}, {}
    for p in lay["pieces"]:
        key = (p["track"], round(p["speed"], 6), float(shifts.get(p["track"], 0) or 0))
        if key in audio:
            continue
        say(f"reading {p['track']}")
        audio[key] = decode(local_path(tracks[p["track"]]), key[1], key[2])
        loud[key] = MX.loudness_db(audio[key], SR)
        note = MX.stretch_note(p["speed"])
        if note:
            warnings.append(f"{p['track']}: {note}")

    # 2. level matching: every track to the loudest track's level (within limits)
    gains = {}
    if recipe.get("level_match", True) and len(loud) > 1:
        ref = max(loud.values())
        gains = {k: float(np.clip(ref - v, -LEVEL_LIMIT_DB, LEVEL_LIMIT_DB)) for k, v in loud.items()}

    # 3. lay the pieces out with crossfades centred on the joins
    say("joining the pieces")
    total = lay["duration"]
    n_out = int(math.ceil(total * SR)) + SR
    out = np.zeros((n_out, CH), dtype=np.float32)
    joins = lay["joins"]
    for k, p in enumerate(lay["pieces"]):
        key = (p["track"], round(p["speed"], 6), float(shifts.get(p["track"], 0) or 0))
        y = audio[key]
        avail = len(y) / SR
        head = joins[k - 1]["xfade"] / 2 if k > 0 else 0.0
        tail = joins[k]["xfade"] / 2 if k < len(joins) else 0.0
        a_t = max(0.0, p["start"] / p["speed"] - head)
        b_t = min(avail, p["end"] / p["speed"] + tail)
        head_eff = p["start"] / p["speed"] - a_t
        tail_eff = b_t - p["end"] / p["speed"]
        seg = y[int(round(a_t * SR)):int(round(b_t * SR))].copy()
        if not len(seg):
            continue
        gain_db = gains.get(key, 0.0) + float(p.get("gain_db") or 0.0)
        if abs(gain_db) > 1e-3:
            seg *= np.float32(10 ** (gain_db / 20))
        if k > 0:
            jn = joins[k - 1]
            n = min(len(seg), int(round((head_eff + jn["xfade"] / 2) * SR)))
            seg[:n] *= _ramp(n, not jn["same_track"], True)[:, None]
        elif p["start"] > 0.05 or recipe.get("fade_in"):
            n = min(len(seg), int(max(float(recipe.get("fade_in") or 0.0), 0.01) * SR))
            seg[:n] *= _ramp(n, False, True)[:, None]
        if k < len(joins):
            jn = joins[k]
            n = min(len(seg), int(round((tail_eff + jn["xfade"] / 2) * SR)))
            seg[len(seg) - n:] *= _ramp(n, not jn["same_track"], False)[:, None]
        at = int(round((p["out_start"] - head_eff) * SR))
        at = max(0, at)
        end = min(n_out, at + len(seg))
        out[at:end] += seg[:end - at]

    # 4. length, fades, level
    trim = recipe.get("trim_to")
    length = min(total, float(trim)) if trim else total
    out = out[:int(round(length * SR))]
    fade_out = float(recipe.get("fade_out") or 0.0)
    if trim and trim < total - 0.05:
        fade_out = max(fade_out, 2 * MX.beat_len(lay["bpm"]) * 4)     # cut short: fade the last 2 bars
    n = min(len(out), int(max(fade_out, 0.01) * SR))
    if n:
        out[len(out) - n:] *= _ramp(n, False, False)[:, None]
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > PEAK:
        out *= np.float32(PEAK / peak)
    say("saving")
    encode(out, out_path, bitrate)

    hop = SR // 30
    m = out[: len(out) // hop * hop].mean(axis=1).reshape(-1, hop)
    rms = np.sqrt((m.astype(np.float64) ** 2).mean(axis=1))
    rms = (rms / (np.percentile(rms, 99.5) + 1e-9)).clip(0, 1) if len(rms) else rms
    return {"duration": round(length, 3), "bpm": round(lay["bpm"], 2),
            "beats": [b for b in lay["beats"] if b < length - 0.05],
            "bars": [b for b in lay["bars"] if b < length - 0.05],
            "energy": [round(float(x), 3) for x in rms], "warnings": warnings, "layout": lay,
            "levels_db": {k[0]: round(v, 1) for k, v in gains.items()}}


def preview_path(recipe: dict) -> Path:
    """A per-recipe preview file (old previews over a day old are cleared)."""
    d = cache_dir() / "previews"
    d.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for f in [*d.glob("*.m4a"), *d.glob("*.mp3")]:
        try:
            if now - f.stat().st_mtime > 86400:
                f.unlink()
        except OSError:
            pass
    body = json.dumps({k: recipe.get(k) for k in ("pieces", "bpm", "shift", "xfade_same", "xfade_other",
                                                   "fade_in", "fade_out", "level_match", "trim_to")},
                      sort_keys=True, default=str)
    return d / f"{hashlib.sha1(body.encode()).hexdigest()[:16]}{'.mp3' if has_mp3() else '.m4a'}"
