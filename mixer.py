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
from core import stem_mix as SM

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


STEM_PARTS = ("vocals", "instrumental", "drums", "bass", "other")
FOUR_STEMS = ("vocals", "drums", "bass", "other")      # 🧠 neural transitions (core/stem_mix.py)


def has_stems(track: dict) -> bool:
    """🎤 Split into vocals + instrumental by analyse_audio_local.py (on the laptop's GPU)."""
    st = track.get("stems") or {}
    return all(st.get(p) for p in ("vocals", "instrumental"))


def has_four_stems(track: dict) -> bool:
    """🧠 Split into vocals, drums, bass and other — what a neural transition mixes."""
    st = (track or {}).get("stems") or {}
    return all(st.get(p) for p in FOUR_STEMS)


def neural_joins(recipe: dict) -> list:
    """[(outgoing piece, incoming piece)] for every 🧠 neural join in the recipe."""
    pieces = recipe.get("pieces") or []
    return [(a, b) for a, b in zip(pieces[:-1], pieces[1:])
            if isinstance((b.get("xfade") or {}).get("neural"), dict)]


def _drive_ref(track: dict, part: str = "full") -> str:
    if part in STEM_PARTS:
        return ((track.get("stems") or {}).get(part)) or ""
    return track.get("source_path") or ""


def local_path(track: dict, part: str = "full") -> Path:
    """Where the track's audio (or one of its 🎤 stems) is on this machine."""
    from render_preview import colab_to_local
    return Path(colab_to_local(_drive_ref(track, part)))


def ensure_local(track: dict, progress=None, parts=("full",)) -> str | None:
    """Make sure the track's audio file (and the stems in `parts`) are on this machine —
    downloads them from Google Drive when they aren't. Returns an error message, or None."""
    for part in parts:
        ref = _drive_ref(track, part)
        if part in STEM_PARTS and not ref:
            return f"{track.get('track_id')}: no {part} stem yet — run run_audio_analysis.bat on the laptop (it makes stems)."
        path = local_path(track, part)
        if path.exists():
            continue
        if "MyDrive/" not in ref:
            return f"{track.get('track_id')}: audio file not found ({path})"
        try:
            from drive_sync import download_source_video
        except Exception as e:
            return f"Google Drive isn't available here: {e}"
        err = download_source_video(ref.split("MyDrive/", 1)[1], path, progress_callback=progress)
        if err:
            return f"{track.get('track_id')}: download failed — {err}"
    return None


def parts_needed(recipe: dict, tracks: dict = None) -> dict:
    """{track_id: {parts}} — which files a recipe plays (the full song and/or its stems). 🧠 Neural
    joins add both sides' four stems — when tracks ({track_id: catalogue track}) says they have them
    (without, the join falls back to an ordinary crossfade)."""
    out = {}
    for p in recipe.get("pieces") or []:
        out.setdefault(p["track"], set()).add(p.get("part") or "full")
    for ly in recipe.get("layers") or []:
        out.setdefault(ly["track"], set()).add(ly.get("part") or "vocals")
    for a, b in neural_joins(recipe):
        if tracks is None or (has_four_stems(tracks.get(a["track"])) and has_four_stems(tracks.get(b["track"]))):
            for p in (a, b):
                out[p["track"]] |= set(FOUR_STEMS)
    return out


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


@functools.lru_cache(maxsize=4)
def _bass_sos(hz: float):
    from scipy.signal import butter
    return butter(4, hz, btype="lowpass", fs=SR, output="sos")


def _fade_region(region: np.ndarray, ramp: np.ndarray, jn: dict | None, region_start: float, bpm: float,
                 outgoing: bool) -> None:
    """Fade one side of a join in place. With 🎛️ bass swap, only the part above MX.BASS_HZ
    follows the crossfade ramp; the low end stays full (outgoing) or silent (incoming) until the
    swap point, then changes hands over MX.BASS_SWAP_BEATS. The two bands are split with a
    zero-phase filter, so low + high is exactly the original audio."""
    if not (jn and jn.get("bass_swap") and jn.get("swap_at") is not None) or len(region) < 64:
        region *= ramp[:, None]
        return
    from scipy.signal import sosfiltfilt
    low = sosfiltfilt(_bass_sos(MX.BASS_HZ), region, axis=0).astype(np.float32)
    high = region - low
    n = len(region)
    w = max(int(MX.BASS_SWAP_BEATS * MX.beat_len(bpm) * SR), int(0.02 * SR))
    centre = (float(jn["swap_at"]) - region_start) * SR
    x = np.clip((np.arange(n, dtype=np.float32) - (centre - w / 2)) / w, 0.0, 1.0)
    step = 1.0 - x if outgoing else x                        # 1 → 0 for the outgoing song's bass
    region[:] = high * ramp[:, None] + low * step[:, None]


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

    # 1. decode each (track, speed, shift, part) once
    audio, loud = {}, {}

    def key_of(p, default_part="full"):
        return (p["track"], round(p["speed"], 6), float(shifts.get(p["track"], 0) or 0),
                p.get("part") or default_part)

    for p, default_part in [*[(p, "full") for p in lay["pieces"]], *[(ly, "vocals") for ly in lay.get("layers") or []]]:
        key = key_of(p, default_part)
        if key in audio:
            continue
        say(f"reading {p['track']}" + ("" if key[3] == "full" else f" ({key[3]})"))
        audio[key] = decode(local_path(tracks[p["track"]], key[3]), key[1], key[2])
        if key[3] != "vocals":
            loud[key] = MX.loudness_db(audio[key], SR)
        note = MX.stretch_note(p["speed"])
        if note and f"{p['track']}: {note}" not in warnings:
            warnings.append(f"{p['track']}: {note}")

    # 1b. 🧠 neural joins: both sides' four stems at the same speed / shift (else an ordinary crossfade)
    neural_ok = set()
    for k, jn in enumerate(lay["joins"]):
        if not jn.get("neural"):
            continue
        a, b = lay["pieces"][k], lay["pieces"][k + 1]
        if not all(has_four_stems(tracks[p["track"]]) and all(local_path(tracks[p["track"]], s).exists()
                                                             for s in FOUR_STEMS) for p in (a, b)):
            warnings.append(f"🧠 Join {k + 1} → {k + 2}: {a['track'] if not has_four_stems(tracks[a['track']]) else b['track']} "
                            "has no drums / bass / other stems yet (run run_audio_analysis.bat) — an ordinary "
                            "crossfade instead.")
            continue
        for p in (a, b):
            for stem in FOUR_STEMS:
                key = key_of({**p, "part": stem})
                if key not in audio:
                    say(f"reading {p['track']} ({stem})")
                    audio[key] = decode(local_path(tracks[p["track"]], stem), key[1], key[2])
        neural_ok.add(k)

    # 2. level matching: every track to the loudest track's level (within limits). A vocal
    #    layer gets its own track's correction (vocals alone are quieter than a full mix).
    gains = {}
    if recipe.get("level_match", True) and len({k[0] for k in loud}) > 1:
        ref = max(loud.values())
        gains = {k: float(np.clip(ref - v, -LEVEL_LIMIT_DB, LEVEL_LIMIT_DB)) for k, v in loud.items()}
    track_gain = {}
    for k, v in gains.items():
        track_gain.setdefault(k[0], v)

    # 3. lay the pieces out with crossfades centred on the joins
    say("joining the pieces")
    total = lay["duration"]
    n_out = int(math.ceil(total * SR)) + SR
    out = np.zeros((n_out, CH), dtype=np.float32)
    joins = lay["joins"]
    for k, p in enumerate(lay["pieces"]):
        key = key_of(p)
        y = audio[key]
        avail = len(y) / SR
        # the incoming audio starts `pre` early (under the end of the piece before), the outgoing
        # carries on `post` past the line (under the start of the next)
        head = joins[k - 1]["pre"] if k > 0 else 0.0
        tail = joins[k]["post"] if k < len(joins) else 0.0
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
            n = min(len(seg), int(round((head_eff + jn["post"]) * SR)))
            if k - 1 in neural_ok:
                seg[:n] = 0.0                                    # 🧠 the stem blend fills it (below)
            else:
                _fade_region(seg[:n], _ramp(n, jn.get("shape", "smooth") == "smooth", True), jn,
                             jn["at"] - head_eff, lay["bpm"], outgoing=False)
        elif p["start"] > 0.05 or recipe.get("fade_in"):
            n = min(len(seg), int(max(float(recipe.get("fade_in") or 0.0), 0.01) * SR))
            seg[:n] *= _ramp(n, False, True)[:, None]
        if k < len(joins):
            jn = joins[k]
            n = min(len(seg), int(round((tail_eff + jn["pre"]) * SR)))
            if k in neural_ok:
                seg[len(seg) - n:] = 0.0                         # 🧠 the stem blend fills it (below)
            else:
                _fade_region(seg[len(seg) - n:], _ramp(n, jn.get("shape", "smooth") == "smooth", False), jn,
                             jn["at"] + tail_eff - n / SR, lay["bpm"], outgoing=True)
        at = int(round((p["out_start"] - head_eff) * SR))
        at = max(0, at)
        end = min(n_out, at + len(seg))
        out[at:end] += seg[:end - at]

    # 3a. 🧠 neural joins: each side's stems over the blend, mixed with their own curves (core/stem_mix.py)
    for k in sorted(neural_ok):
        jn, a, b = joins[k], lay["pieces"][k], lay["pieces"][k + 1]
        w0 = jn["at"] - jn["pre"]
        n = int(round((jn["pre"] + jn["post"]) * SR))
        sides = []
        for p in (a, b):
            src0 = int(round((p["start"] / p["speed"] + (w0 - p["out_start"])) * SR))
            gain_db = gains.get(key_of(p), 0.0) + float(p.get("gain_db") or 0.0)
            g = np.float32(10 ** (gain_db / 20))
            stems = {}
            for stem in FOUR_STEMS:
                part = np.zeros((n, CH), dtype=np.float32)
                if (p.get("part") == "instrumental" and stem == "vocals") or \
                        (p.get("part") == "vocals" and stem != "vocals"):
                    stems[stem] = part                           # the piece plays only those stems
                    continue
                y = audio[key_of({**p, "part": stem})]
                lo, hi = max(src0, 0), min(src0 + n, len(y))
                if hi > lo:
                    part[lo - src0:hi - src0] = y[lo:hi] * g
                stems[stem] = part
            sides.append(stems)
        blend = SM.render_region(sides[0], sides[1], jn["neural"]["profile"], jn["neural"]["beats"], SR,
                                 curves=jn["neural"].get("curves"))
        at = int(round(w0 * SR))
        lo, hi = max(at, 0), min(n_out, at + n)
        if hi > lo:
            out[lo:hi] += blend[lo - at:hi - at]

    # 3b. 🎤 layers (one track's vocals, instrumental or the full song) over the pieces, faded in and out
    for ly in lay.get("layers") or []:
        key = key_of(ly, "vocals")
        y = audio[key]
        a_t, b_t = ly["start"] / ly["speed"], ly["end"] / ly["speed"]
        seg = y[int(round(a_t * SR)):int(round(b_t * SR))].copy()
        if not len(seg):
            continue
        gain_db = gains.get(key, track_gain.get(ly["track"], 0.0)) + float(ly.get("gain_db") or 0.0)
        if abs(gain_db) > 1e-3:
            seg *= np.float32(10 ** (gain_db / 20))
        fade_beats = float(ly.get("fade_beats") if ly.get("fade_beats") is not None else 0.5)
        n = min(len(seg) // 2, int(max(fade_beats, 0.05) * MX.beat_len(lay["bpm"]) * SR))
        if n:
            seg[:n] *= _ramp(n, False, True)[:, None]
            seg[len(seg) - n:] *= _ramp(n, False, False)[:, None]
        at = int(round(ly["out_start"] * SR))
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
    if neural_ok:                     # 🧠 summed stems can peak: round them off (soft knee, not a hard clip)
        out = SM.soft_knee(out).astype(np.float32)
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
            "levels_db": {k[0] + ("" if k[3] == "full" else f" ({k[3]})"): round(v, 1) for k, v in gains.items()}}


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
    body = json.dumps({k: recipe.get(k) for k in ("pieces", "layers", "bpm", "shift", "xfade_same", "xfade_other",
                                                   "fade_in", "fade_out", "level_match", "trim_to")},
                      sort_keys=True, default=str)
    return d / f"{hashlib.sha1(body.encode()).hexdigest()[:16]}{'.mp3' if has_mp3() else '.m4a'}"
