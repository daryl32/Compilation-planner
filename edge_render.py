"""
⏱️ Edge Mode in the rendered video (called by render_preview.render_plan_dict in place of its plain "add the
track's audio" step, for a plan with an "edge" block): the 👩 avatar pops up at her cues (a framed box in a
corner, or the right half of a split screen), the countdown overlay goes on top of the picture, a 🎬 intro screen
(the avatar beside the chosen settings and thumbnails) can come first, and the
sound is mixed — the playlist's songs back to back (starting again until the time is filled, at a quarter volume
while the voiceover or the avatar is on), the generated beat, and the voiceover lines at their cues. Nothing here
chooses footage.

How it is built (14 Nov 2026, third version):
1. the SOUND is mixed by itself, from files on this machine (a song that isn't here is downloaded once, never streamed);
2. the PICTURE is made one phase (20 s) at a time, several at once. The timer and the avatar are laid over the
   footage HERE, IN PYTHON, frame by frame: ffmpeg only reads one file's frames out and writes one stream of frames
   in — never a filter graph with several inputs. (Version 1, one long ffmpeg pass with every avatar clip as a
   time-shifted input, hung or let the timer jump on the server; version 2, a small multi-input ffmpeg job per
   phase, stalled there whenever an avatar clip or stand-in was among the inputs. Neither fault showed on ffmpeg
   6.1 — so the layering no longer depends on how an ffmpeg build schedules several inputs.)
3. the pieces (intro screen, phases, outro screen) are joined without encoding again and the sound is added.
A check at the end compares the picture's and the sound's length with the plan's.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import edge_mode as E

AUDIO_FMT = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"
STALL_SEC = 180                    # the picture pass is stopped when it hasn't moved for this long
MIN_SHOW_SEC = 0.2                 # an avatar appearance shorter than this is left out
PICTURE_PRESET = "ultrafast"       # the picture's encoder speed
PIECE_TIMEOUT = 120                # seconds one phase's picture may take before it is stopped (it needs a few)
KEY_RGB = (0x00, 0xB1, 0x40)       # the avatar clips' backdrop (the Colab notebook's BACKGROUND), made see-through
KEY_SIMILARITY, KEY_BLEND = 0.16, 0.08      # how close to that colour counts as backdrop, and how soft the edge is
FPS = 25


def _enc() -> list:
    """The same encoder settings for every piece of the picture, so they can be joined without encoding again."""
    return ["-c:v", "libx264", "-preset", PICTURE_PRESET, "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-x264-params", "stitchable=1", "-video_track_timescale", "12800", "-an"]


def _run(cmd: list, timeout: float = PIECE_TIMEOUT) -> None:
    """One ffmpeg job; RuntimeError with the end of its own messages when it fails or takes longer than `timeout`."""
    try:
        r = subprocess.run([cmd[0], "-nostdin", *cmd[1:]], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        tail = (e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))[-800:]
        raise RuntimeError(f"A step of the Edge picture took over {int(timeout)} s and was stopped ({Path(cmd[-1]).name}).\n{tail}")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on the Edge picture ({Path(cmd[-1]).name}):\n{r.stderr[-1200:]}")


# ---------------------------------------------------------------------------
# Frames in, frames out (one input per ffmpeg; the layering is done here)
# ---------------------------------------------------------------------------

_DIMS = {}


def _dims(path) -> tuple:
    """(width, height) of a video's picture, (0, 0) when it can't be read. Remembered per file."""
    key = str(path)
    if key not in _DIMS:
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
            w, h = (int(x) for x in r.stdout.strip().split("\n")[0].split(",")[:2])
            _DIMS[key] = (w, h)
        except Exception:          # noqa: BLE001
            return (0, 0)
    return _DIMS[key]


def _read_frames(path, skip: float, seconds: float, size: tuple, cover: bool = False):
    """`seconds` of a video from `skip` s in, as an array of frames (n, height, width, 3) at FPS, scaled to `size`
    (cover: scaled to fill it and cropped). An empty array when it can't be read."""
    import numpy as np
    w, h = size
    vf = f"fps={FPS}," + (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}" if cover else f"scale={w}:{h}")
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(skip, 0):.3f}", "-t", f"{seconds:.3f}", "-i", str(path),
                            "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=PIECE_TIMEOUT)
        raw = r.stdout
    except subprocess.TimeoutExpired:
        raw = b""
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[:n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)


def _key_alpha(frames):
    """How solid each pixel is once the backdrop colour is cut out (n, h, w, 1), 0 = backdrop … 255 = her — the
    same rule as ffmpeg's chromakey: the distance from the key colour in the two colour-difference channels."""
    import numpy as np
    f = frames.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    kr, kg, kb = (float(x) for x in KEY_RGB)
    kcb, kcr = -0.168736 * kr - 0.331264 * kg + 0.5 * kb, 0.5 * kr - 0.418688 * kg - 0.081312 * kb
    d = np.sqrt(((cb - kcb) ** 2 + (cr - kcr) ** 2) / (255.0 ** 2 * 2.0))
    a = np.clip((d - KEY_SIMILARITY) / max(KEY_BLEND, 1e-6), 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint16)[..., None]


def _lay(frame, x: int, y: int, rgb, alpha=None) -> None:
    """rgb (h, w, 3) laid on `frame` at (x, y), in place; alpha (h, w, 1) 0–255 when it is see-through in places.
    What falls outside the frame is cut off."""
    import numpy as np
    H, W = frame.shape[:2]
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    if alpha is None:
        frame[y0:y1, x0:x1] = src
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    under = frame[y0:y1, x0:x1].astype(np.uint16)
    frame[y0:y1, x0:x1] = ((under * (255 - a) + src.astype(np.uint16) * a + 127) // 255).astype(np.uint8)


def _corner_xy(corner: str, frame: tuple, size: tuple, margin_share: float = 0.035) -> tuple:
    """Where a picture of `size` sits in a corner of the frame (a margin of 3.5 % of its height)."""
    W, H = frame
    w, h = size
    m = int(H * margin_share)
    return {"top-left": (m, m), "top-right": (W - w - m, m), "bottom-left": (m, H - h - m),
            "bottom-right": (W - w - m, H - h - m), "centre": ((W - w) // 2, (H - h) // 2)}.get(corner, (m, m))


def _write_video(out: Path, frame: tuple, n_frames: int, make, feeders: list = None) -> None:
    """A piece of the picture: make(i) → frame i (h, w, 3 uint8) for i in 0…n_frames−1, encoded with the shared
    settings. RuntimeError with ffmpeg's message when the encoder fails or the piece takes over PIECE_TIMEOUT
    (the encoder and any `feeders` — processes make() reads from — are stopped then, so nothing waits for ever)."""
    import tempfile
    import threading
    W, H = frame
    with tempfile.TemporaryFile() as err:
        enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", *_enc(), str(out)], stdin=subprocess.PIPE, stderr=err)
        fired = []

        def _stop():
            fired.append(True)
            for proc in [enc, *(feeders or [])]:
                try:
                    proc.kill()
                except OSError:
                    pass
        late = threading.Timer(PIECE_TIMEOUT, _stop)
        late.start()
        try:
            for i in range(n_frames):
                enc.stdin.write(make(i).tobytes())
            enc.stdin.close()
            code = enc.wait()
        except (BrokenPipeError, OSError):
            code = enc.wait() or 1
        finally:
            late.cancel()
            if enc.poll() is None:
                enc.kill()
                enc.wait()
        if code != 0 or fired:
            err.seek(0)
            why = f"took over {PIECE_TIMEOUT} s and was stopped" if fired else "failed"
            raise RuntimeError(f"A step of the Edge picture {why} ({Path(out).name}).\n{err.read().decode('utf-8', 'replace')[-800:]}")


def playlist_raws(plan: dict) -> list:
    """The raw source of every song an Edge plan plays (for the renderer's "can I read this" check)."""
    from render_preview import get_track_raw_source
    return list(dict.fromkeys(get_track_raw_source(p["track_id"]) for p in (plan.get("edge") or {}).get("playlist") or []))


_PEAKS = {}


def _peak_db(path) -> float | None:
    """How loud a file's sound is WHILE IT SPEAKS, in dB below full scale (None when it can't be read): the level of
    its 50 ms stretches, leaving the silent ones out (her clips for short lines end in silence) — not the single
    loudest sample, which a compressed file's first instant can overshoot. Remembered per file."""
    try:
        st_ = os.stat(path)
        key = (str(path), st_.st_mtime_ns, st_.st_size)
    except OSError:
        return None
    if key not in _PEAKS:
        _PEAKS[key] = None
        try:
            import numpy as np
            r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                                "-f", "s16le", "-"], capture_output=True, timeout=60)
            x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float64) / 32768.0
            n = len(x) // 800
            if n:
                rms = np.sqrt((x[:n * 800].reshape(n, 800) ** 2).mean(axis=1))
                said = rms[rms > max(rms.max() * 0.1, 1e-4)]
                if len(said):
                    _PEAKS[key] = float(20 * np.log10(np.sqrt((said ** 2).mean())))
        except Exception:          # noqa: BLE001
            pass
    return _PEAKS[key]


def _match_gain(sound, like) -> float:
    """dB to add to `sound` so it is as loud as `like` (her clips come out of the lip-sync model much quieter
    than the lines they were made from). 0 when there is nothing to compare; never more than −6 … +20."""
    if not like or not sound or str(like) == str(sound) or not Path(like).exists():
        return 0.0
    a, b = _peak_db(sound), _peak_db(like)
    if a is None or b is None:
        return 0.0
    return max(-6.0, min(20.0, b - a))


LEVEL_FALLBACK_DB = -20.0         # 🔊 the speech level lines are brought to when none of the lines can be measured
LEVEL_UP_MAX, LEVEL_DOWN_MAX = 15.0, 6.0     # never more than +15 dB (noise) or −6 dB on any line
_TARGET = {}


def voice_target() -> float:
    """🔊 The speech level every line is brought to (Daryl, 17 Nov: "some are very quiet"): the MIDDLE level of all
    the cue map's lines on this machine — the quiet ones come up to the typical line, the odd loud one comes down,
    and the mix as a whole stays where it was tuned (VOICE_LEVEL on top). Remembered until the lines change."""
    try:
        from planner.edge import lines_dir
        files = sorted(f for f in lines_dir().glob("*") if f.suffix.lower() in (".mp3", ".wav", ".m4a") and f.stat().st_size)
    except Exception:          # noqa: BLE001
        files = []
    key = tuple((str(f), f.stat().st_mtime_ns) for f in files)
    if key not in _TARGET:
        import statistics
        got = [d for d in (_peak_db(f) for f in files) if d is not None]
        _TARGET.clear()
        _TARGET[key] = round(statistics.median(got), 1) if got else LEVEL_FALLBACK_DB
    return _TARGET[key]


def level_gain(path, target: float = None) -> float:
    """dB that brings a line (or her clip's sound) to the common speech level; 0 when it can't be measured."""
    d = _peak_db(path) if path else None
    if d is None:
        return 0.0
    t = voice_target() if target is None else target
    return max(-LEVEL_DOWN_MAX, min(LEVEL_UP_MAX, t - d))


def voice_levels(files: dict) -> list:
    """For the 🎙️ panel: [(name, speech level dB, gain dB)] for {name: path}, quietest first."""
    t = voice_target()
    out = [(n, _peak_db(f), level_gain(f, t)) for n, f in files.items() if f]
    return sorted((x for x in out if x[1] is not None), key=lambda x: x[1])


def _voice_track(cues: list, total: float, tmp_dir: Path, run, name: str = "voice.wav") -> Path | None:
    """Every cue laid at its time on one silent track of the compilation's length. A cue without a clip (or whose
    clip has gone) gets its phase's placeholder tone."""
    if not cues:
        return None
    placeholders, inputs, parts = {}, [], []
    for k, cue in enumerate(cues):
        f = cue.get("file")
        if not f or not Path(f).exists():
            p = cue["profile"]
            if p not in placeholders:
                placeholders[p] = E.write_placeholder_cue(p, tmp_dir / f"cue_{p}.wav")
            f, gain = placeholders[p], 0.0
        else:                      # 🔊 every line (and her clip's sound) at the common level, plus the line's own ± from the Cue Map
            gain = level_gain(f) + max(-6.0, min(6.0, float(cue.get("gain") or 0)))
        inputs += ["-i", str(f)]
        ms = int(round(float(cue["time"]) * 1000))
        parts.append(f"[{k}:a:0]{AUDIO_FMT}," + (f"volume={gain:.1f}dB," if abs(gain) >= 0.5 else "") + f"adelay={ms}:all=1[c{k}]")
    out = tmp_dir / name
    mix = "".join(f"[c{k}]" for k in range(len(cues)))
    run(["ffmpeg", "-y", *inputs, "-filter_complex",
         ";".join(parts) + f";{mix}amix=inputs={len(cues)}:normalize=0:duration=longest,volume={E.VOICE_LEVEL:.3f},"
         "alimiter=limit=0.95:level=0[a]",
         "-map", "[a]", "-t", f"{total:.3f}", str(out)])
    return out


_PROBED = {}


def probe(path) -> dict:
    """{"seconds", "audio", "video"} of a media file (zeros / False when it can't be read). Remembered per file
    until the file changes — a compile asks for every line and clip."""
    try:
        st_ = os.stat(path)
        key = (str(path), st_.st_mtime_ns, st_.st_size)
    except OSError:
        key = None
    if key in _PROBED:
        return dict(_PROBED[key])
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json",
                              str(path)], capture_output=True, text=True, timeout=60)
        info = json.loads(out.stdout or "{}")
        kinds = {st.get("codec_type") for st in info.get("streams") or []}
        got = {"seconds": float((info.get("format") or {}).get("duration") or 0.0), "audio": "audio" in kinds,
               "video": "video" in kinds}
    except Exception:          # noqa: BLE001
        return {"seconds": 0.0, "audio": False, "video": False}
    if key is not None and got["seconds"] > 0:
        _PROBED[key] = dict(got)
    return got


def run_watched(cmd: list, seconds: float, progress=None, stall: float = None) -> None:
    """Run an ffmpeg command that writes `seconds` of output, calling progress(0–1) about once a second. Raises
    RuntimeError with the end of ffmpeg's own messages when it fails — or when its output hasn't moved for `stall`
    seconds (it is stopped first), so a stuck render says so instead of waiting for ever."""
    import queue
    import tempfile
    import threading
    stall = float(stall or STALL_SEC)
    cmd = [cmd[0], "-nostdin", "-nostats", "-progress", "pipe:1", *cmd[1:]]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        lines = queue.Queue()

        def _read():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)
        threading.Thread(target=_read, daemon=True).start()

        def _tail() -> str:
            err.seek(0)
            return err.read().decode("utf-8", "replace")[-1500:]
        at, moved, told = 0.0, time.time(), 0.0
        while True:
            try:
                line = lines.get(timeout=1.0)
            except queue.Empty:
                if time.time() - moved > stall:
                    proc.kill()
                    proc.wait()
                    raise RuntimeError(f"The picture pass stood still for {int(stall)} s at {int(at // 60)}:{int(at % 60):02d} "
                                       f"of {int(seconds // 60)}:{int(seconds % 60):02d} and was stopped.\n{_tail()}")
                continue
            if line is None:
                break
            k, _, v = line.strip().partition("=")
            if k in ("out_time_us", "out_time_ms") and v.lstrip("-").isdigit():      # (both are microseconds)
                t = int(v) / 1e6
                if t > at + 1e-3:
                    at, moved = t, time.time()
                    if progress and time.time() - told >= 1.0:
                        told = time.time()
                        progress(min(at / max(seconds, 1e-6), 1.0))
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed in the picture pass:\n{_tail()}")


def _appearances(edge: dict, tmp_dir: Path, frame: tuple) -> list:
    """The avatar's appearances, ready to draw: [{"time", "seconds", "profile", "clip" (her video) or "still" (the
    stand-in figure), "sound" (the clip, when her voice is in it)}]. A cue without a clip gets the stand-in figure for a few seconds. An
    appearance never runs into the next one."""
    av = edge.get("avatar") or {}
    if not av.get("on"):
        return []
    cues = sorted(av.get("cues") or [], key=lambda c: c["time"])
    total = E.total_seconds(edge["schedule"])
    w, h = frame
    still = None
    out = []
    for k, cue in enumerate(cues):
        room = (cues[k + 1]["time"] if k + 1 < len(cues) else total) - cue["time"]
        clip = cue.get("clip")
        info = probe(clip) if clip and Path(clip).exists() else None
        if info and info["video"] and info["seconds"] > 0:
            secs = min(info["seconds"], room)
            if secs < MIN_SHOW_SEC:
                continue
            out.append({"time": float(cue["time"]), "seconds": secs, "profile": cue["profile"],
                        "clip": str(clip), "still": None, "sound": str(clip) if info["audio"] else None})
        else:
            if still is None:
                still = tmp_dir / "avatar_placeholder.png"
                E.avatar_placeholder((int(h * 0.6) // 2 * 2, h)).save(still)
            secs = min(float(cue.get("seconds") or E.PLACEHOLDER_CUE_SEC), room)      # as long as her line, when known
            if secs < MIN_SHOW_SEC:
                continue
            out.append({"time": float(cue["time"]), "seconds": secs, "profile": cue["profile"],
                        "clip": None, "still": str(still), "sound": None})
    return out


def _sound_cues(edge: dict, shows: list) -> list:
    """What is heard at each cue: with the avatar on, HER CLIP's sound where her voice is in it (Daryl, 14 Nov) —
    brought to the loudness of the line's own recording ("match") when there is one; else the voiceover recording
    for that moment; else (None) the placeholder tone."""
    voice = edge.get("voiceover") or {}
    by_time = {round(float(c["time"]), 3): dict(c) for c in (voice.get("cues") or [])} if voice.get("on") else {}
    for s in shows:
        key = round(s["time"], 3)
        if s["sound"]:
            by_time[key] = {"time": s["time"], "profile": s["profile"], "file": s["sound"],
                            "match": (by_time.get(key) or {}).get("file"), "gain": (by_time.get(key) or {}).get("gain", 0)}
        elif key not in by_time:
            by_time[key] = {"time": s["time"], "profile": s["profile"], "file": None}
    return [by_time[k] for k in sorted(by_time)]


CARD_LEAD = 0.4                    # silence before the first line and after the last on the intro / outro screen


def _card(spec: dict, tmp_dir: Path, run, frame: tuple, name: str):
    """An intro or outro screen → (its picture, its sound (a WAV), seconds), or None when it is off.
    spec: {"on", "clip" (the avatar's video, voice included), "lines" ([{"file", "seconds"}] — voiceover lines read
    one after another), "stats", "thumbs", "title", "figure" (draw the stand-in when she has no clip)}. Its length:
    the avatar's clip when there is one; else the lines back to back; else PLACEHOLDER_INTRO_SEC, silent."""
    if not (spec or {}).get("on"):
        return None
    import numpy as np
    thumbs = []
    try:
        from planner.data import get_nearest_thumbnail, load_all_catalogues
        cats = load_all_catalogues()
        thumbs = [get_nearest_thumbnail(cats, vid, float(t)) for vid, t in spec.get("thumbs") or []]
    except Exception:          # noqa: BLE001 — the screen still works without thumbnails
        thumbs = []
    W, H = frame
    clip = spec.get("clip")
    info = probe(clip) if clip and Path(clip).exists() else None
    real = bool(info and info["video"] and info["seconds"] > 0)
    lines = [] if real and info["audio"] else list(spec.get("lines") or [])
    cues, shows, t = [], [], CARD_LEAD
    for ln in lines:                                       # the lines one after another; her clip for a line when there is one
        f, c = ln.get("file"), ln.get("clip")
        ci = probe(c) if c and Path(c).exists() else None
        has_clip = bool(ci and ci["video"] and ci["seconds"] > 0)
        secs = (ci["seconds"] if has_clip else (probe(f)["seconds"] if f and Path(f).exists() else 0.0)) or float(ln.get("seconds") or 2.0)
        cues.append({"time": t, "profile": "wait", "file": c if has_clip and ci["audio"] else f, "match": f,
                     "gain": ln.get("gain", 0)})
        if has_clip:
            shows.append({"time": t, "seconds": secs, "clip": c})
        t += secs + E.LINE_GAP
    secs = info["seconds"] if real else (t - E.LINE_GAP + CARD_LEAD if cues else E.PLACEHOLDER_INTRO_SEC)
    frames = max(int(round(secs * FPS)), 1)
    secs = frames / FPS                                    # whole frames, so picture and sound stay the same length
    if real:
        shows = [{"time": 0.0, "seconds": secs, "clip": clip}]
    card = E.intro_card(frame, [tuple(x) for x in spec.get("stats") or []], thumbs, title=spec.get("title") or "EDGE MODE",
                        placeholder=bool(spec.get("figure", True)) and not real and len(shows) < max(len(lines), 1))
    base = np.asarray(card.convert("RGB").resize((W, H)), dtype=np.uint8)
    ax, ay, aw, ah = E.intro_layout(frame)["avatar"]
    layers = []                                            # (first frame, last frame + 1, her frames, how solid, x, y)
    for k, sh in enumerate(shows):                         # she holds her last frame until the next line starts
        cw, ch = _dims(sh["clip"])
        if not cw or not ch:
            continue
        f = min(aw / cw, ah / ch)
        size = (max(int(cw * f) // 2 * 2, 2), max(int(ch * f) // 2 * 2, 2))
        got = _read_frames(sh["clip"], 0.0, sh["seconds"], size)
        if not len(got):
            continue
        until = shows[k + 1]["time"] if k + 1 < len(shows) else secs
        layers.append((int(round(sh["time"] * FPS)), int(round(until * FPS)), got, _key_alpha(got),
                       ax + (aw - size[0]) // 2, ay + (ah - size[1]) // 2))

    def make(i):
        on = next((ly for ly in layers if ly[0] <= i < ly[1]), None)
        if on is None:
            return base
        pic = base.copy()
        j = min(i - on[0], len(on[2]) - 1)
        _lay(pic, on[4], on[5], on[2][j], on[3][j])
        return pic
    out = tmp_dir / f"{name}.mp4"
    _write_video(out, frame, frames, make)
    sound = tmp_dir / f"{name}.wav"
    if real and info["audio"]:
        src = ["-i", str(clip)]
    elif cues:
        src = ["-i", str(_voice_track(cues, secs, tmp_dir, run, f"{name}_voice.wav"))]
    else:
        src = ["-f", "lavfi", "-t", f"{secs:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
    _run(["ffmpeg", "-y", *src, "-vn", "-af", f"{AUDIO_FMT},apad,atrim=0:{secs:.3f}", "-t", f"{secs:.3f}", "-c:a", "pcm_s16le", str(sound)])
    return out, sound, secs


def _song_input(sources, raw: str) -> list:
    """ffmpeg input arguments for a song, from a file on this machine — downloaded first when it isn't here."""
    if hasattr(sources, "is_local") and not sources.is_local(raw):
        errs = sources.download([raw])
        if errs:
            raise FileNotFoundError("A song couldn't be fetched for the render:\n" + "\n".join(errs))
    return list(sources.input_args(raw))


def _line_spans(edge: dict, shows: list) -> list:
    """[(start, end)] of every moment a line is heard — what the 🔉 songs dip under."""
    spans = []
    voice = edge.get("voiceover") or {}
    for c in (voice.get("cues") or []) if voice.get("on") else []:
        secs = float(c.get("seconds") or 0.0)
        if not secs and c.get("file") and Path(c["file"]).exists():
            secs = probe(c["file"])["seconds"]
        spans.append((float(c["time"]), float(c["time"]) + (secs or 1.5)))
    for s in shows:
        spans.append((float(s["time"]), float(s["time"]) + float(s["seconds"])))
    spans.sort()
    out = []
    for a, b in spans:                                     # the same line from the voice and from her clip: one dip
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _song_tracks(parts: list) -> dict:
    """{track_id: catalogue track} of the songs an energy mix plays."""
    from planner.compose import _load_track
    return {tid: _load_track(tid) for tid in dict.fromkeys(p["track_id"] for p in parts)}


def _mix(edge: dict, shows: list, sources, tmp_dir: Path, run, notes: list = None) -> Path:
    """The compilation's whole sound as one file: the songs back to back (quieter under the voiceover / avatar and
    in Wait and Stop), the generated beat, the lines."""
    from render_preview import get_track_raw_source
    sched = edge["schedule"]
    total = E.total_seconds(sched)
    inputs, filters, layers, idx = [], [], [], 0
    voice = edge.get("voiceover") or {}
    talking = bool(voice.get("on") or shows)
    parts = edge.get("playlist") or []
    music = edge.get("music") or {}
    if parts and music.get("energy_mix"):                  # ⚡ the stretches, joined by the 🎚️ Mix engine round by round
        import edge_energy as EE
        files, warns = EE.render_rounds(parts, _song_tracks(parts), tmp_dir)
        if notes is not None:
            notes += warns
        for k, (_r, f, secs) in enumerate(files):          # rounds meet in the silent Stop → Wait gap
            inputs += ["-i", str(f)]
            filters.append(f"[{idx}:a:0]{AUDIO_FMT},apad,atrim=0:{secs:.3f},asetpts=PTS-STARTPTS[r{k}]")
            idx += 1
        filters.append("".join(f"[r{k}]" for k in range(len(files))) + f"concat=n={len(files)}:v=0:a=1[mixed]")
    elif parts:
        for k, part in enumerate(parts):
            inputs += ["-t", f"{float(part['play']):.3f}", *_song_input(sources, get_track_raw_source(part["track_id"]))]
            # (":a:0" — an .m4a may carry its cover art as a second stream)
            filters.append(f"[{idx}:a:0]{AUDIO_FMT},atrim=0:{float(part['play']):.3f},asetpts=PTS-STARTPTS[p{k}]")
            idx += 1
    if parts:
        # 🌙 fade out in Stop / back in over Wait, 🔉 full volume dipping only under her lines — both on in plans
        # compiled from 15 Nov; an older plan (no such keys) sounds as it did: a quarter under the voice, a quarter
        # in Wait and Stop
        spec = E.music_spec(sched, talking=talking, lines=_line_spans(edge, shows),
                            fade_stop=bool(music.get("fade_stop", False)), duck=bool(music.get("duck", False)),
                            still_volume=float(music.get("still_volume", E.MUSIC_IN_STILL)),
                            # (never above a quarter: a plan compiled earlier stored 0.5)
                            under_voice=min(float(voice.get("music_volume", E.MUSIC_UNDER_VOICE)), E.MUSIC_UNDER_VOICE))
        level = f"volume=volume='{E.music_gain_expr(spec)}':eval=frame,volume={E.MUSIC_LEVEL:.3f}"
        if music.get("energy_mix"):
            filters.append(f"[mixed]{level}[music]")
        else:
            filters.append("".join(f"[p{k}]" for k in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1,{level}[music]")
        layers.append("[music]")
    beat = edge.get("beat") or {}
    if beat.get("on"):
        wav = E.write_beat_wav(sched, tmp_dir / "beat.wav")
        inputs += ["-i", str(wav)]
        filters.append(f"[{idx}:a:0]{AUDIO_FMT},volume={float(beat.get('volume', 1.0)):.3f}[beat]")
        layers.append("[beat]")
        idx += 1
    if (edge.get("signals") or {}).get("sound") and len(sched) > 1:      # 🔔 a bell at every phase change
        wav = E.write_chime_wav(sched, tmp_dir / "chime.wav")
        inputs += ["-i", str(wav)]
        filters.append(f"[{idx}:a:0]{AUDIO_FMT},volume={E.CHIME_VOLUME:.3f}[chime]")
        layers.append("[chime]")
        idx += 1
    if talking:
        vt = _voice_track(_sound_cues(edge, shows), total, tmp_dir, run)
        if vt is not None:
            inputs += ["-i", str(vt)]
            filters.append(f"[{idx}:a:0]{AUDIO_FMT}[voice]")
            layers.append("[voice]")
            idx += 1
    if not layers:                                         # nothing to hear (shouldn't happen: no songs → beat on)
        inputs += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
        filters.append(f"[{idx}:a:0]{AUDIO_FMT}[a]")
    elif len(layers) == 1:
        filters.append(f"{layers[0]}alimiter=limit=0.95,apad,atrim=0:{total:.3f}[a]")
    else:
        filters.append("".join(layers) + f"amix=inputs={len(layers)}:normalize=0:duration=longest,alimiter=limit=0.95,"
                                         f"apad,atrim=0:{total:.3f}[a]")
    out = tmp_dir / "mix.wav"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[a]", "-t", f"{total:.3f}",
         "-c:a", "pcm_s16le", str(out)])
    return out


def _timer_pictures(timer_dir: Path, sched: list) -> dict:
    """{timer key ("fast-20", "handsoff-5"): [(picture (h, w, 3), how solid (h, w, 1))] — one every 1 / OVERLAY_FPS s}
    from the pictures edge_mode.write_overlay drew."""
    import numpy as np
    from PIL import Image
    out = {}
    for key in sorted({E.timer_key(ph) for ph in sched}):
        out[key] = []
        for f in sorted(timer_dir.glob(f"{key}_*.png")):
            a = np.asarray(Image.open(f).convert("RGBA"), dtype=np.uint8)
            out[key].append((a[..., :3].copy(), a[..., 3:4].astype(np.uint16)))
    return out


def _badge_xy(corner: str, frame: tuple, timer: tuple, badge: tuple, margin_share: float = 0.035) -> tuple:
    """▸ The round label's place: under the timer when it sits at the top (or the centre), over it at the bottom."""
    tx, ty = _corner_xy(corner, frame, timer, margin_share)
    gap = max(int(frame[1] * 0.012), 2)
    x = tx + (timer[0] - badge[0]) // 2
    return (x, ty - badge[1] - gap) if corner.startswith("bottom") else (x, ty + timer[1] + gap)


def _rgba(img):
    import numpy as np
    a = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    return a[..., :3].copy(), a[..., 3:4].astype(np.uint16)


def _guide_layers(edge: dict, sched: list, corner: str, frame: tuple, badges: dict) -> dict:
    """🎯 The stroke bar (beside the timer's ring, on the side away from the frame's edge) and ⏭️ the next-phase labels
    (under the round label, or under the timer) → {"bar": {...}, "next": {phase index: (picture, solid, x, y)}}.
    Plans from before 16 Nov have neither."""
    W, H = frame
    tw, th = E.overlay_size(H)
    tx, ty = _corner_xy(corner, (W, H), (tw, th))
    out = {}
    pace = edge.get("pace") or {}
    if pace.get("on") and pace.get("beats"):
        ring = tw - 8
        track, bw = E.guide_bar(ring)
        gap = max(int(W * 0.008), 3)
        mk = {p: _rgba(E.guide_marker(bw, E.PROFILES[p]["ring"])) for p in E.ORDER}
        dim = _rgba(E.guide_marker(bw, (0, 0, 0), dim=True))
        mw, mh = mk["slow"][0].shape[1], mk["slow"][0].shape[0]
        x = tx + tw + gap if corner.endswith("left") else tx - mw - gap
        y = ty + 4
        out["bar"] = {"beats": [float(t) for t in pace["beats"]], "track": _rgba(track),
                      "track_x": x + (mw - bw) // 2, "y": y, "height": ring, "markers": mk, "dim": dim, "marker_x": x,
                      "marker_h": mh}
    if (edge.get("next_up") or {}).get("on") and sched:
        chips = {}
        gap = max(int(H * 0.012), 2)
        out["next"] = {}
        for k, ph in enumerate(sched):
            nxt = E.next_real(sched, k)                    # (a surprise is never announced)
            if nxt not in chips:
                chips[nxt] = _rgba(E.next_chip(nxt, tw))
            rgb, solid = chips[nxt]
            ch = rgb.shape[0]
            badge = badges.get(ph["round"])
            if corner.startswith("bottom"):                # stacked above the timer (and its round label)
                top = badge[3] if badge is not None else ty
                yy = top - ch - gap
            else:                                          # stacked under them
                yy = (badge[3] + badge[0].shape[0] if badge is not None else ty + th) + gap
            out["next"][ph["index"]] = (rgb, solid, tx + (tw - rgb.shape[1]) // 2, yy)
    return out


def _strip_layers(edge: dict, sched: list, frame: tuple, signals: bool, band: int = None) -> dict:
    """⏱️ The "strip" timer style (edge_strip.py — design B, 21 Nov): the top box for every look (as it is, edged, and
    ⚠️ flashing), the numbers, the ⏭️ NEXT tab, ▸ the round cards, 🔔 the phase change, ✋ / ⏩ the surprise tags, and
    the bottom stroke strip with its markers."""
    import edge_strip as ES
    W, H = frame
    lay = ES.layout(frame)
    looks = {ph["index"]: E.timer_style(ph) for ph in sched}
    block = {"handsoff": (142, 142, 152)}
    boxes = {}
    for lk in sorted(set(looks.values())):
        st_ = E.style(lk)
        modes = ("normal", "edge", "warn") if signals else ("normal",)
        boxes[lk] = {}
        for m in modes:
            img, glow = ES.box_image(lay, st_["label"], st_["ring"], block.get(lk, st_["ring"]), m)
            boxes[lk][m] = (*_rgba(img), glow)
    digits = {}
    for ph in sched:
        _tot, c_left = E.clock(ph)
        for secs in range(0, int(math.ceil(c_left)) + 2):
            for big in (False, True):
                text = ES.number_text(looks[ph["index"]], float(secs))
                if (text, big) not in digits:
                    digits[(text, big)] = _rgba(ES.digits_image(lay, text, big))
    # ✋ the clock frozen during a hands off: where the phase stands when it carries on
    frozen = {}
    for k, ph in enumerate(sched):
        if ph.get("kind") == "handsoff" and k + 1 < len(sched):
            frozen[ph["index"]] = ES.number_text(sched[k + 1]["profile"], E.clock(sched[k + 1])[1])
            digits.setdefault((frozen[ph["index"]], False), _rgba(ES.digits_image(lay, frozen[ph["index"]], False)))
    tabs, tab_for = {}, {}
    for k, ph in enumerate(sched):
        nx = E.next_real(sched, k)
        key = nx or "END"
        if key not in tabs:
            tabs[key] = _rgba(ES.tab_image(lay, E.PROFILES[nx]["label"] if nx else "END",
                                           E.PROFILES[nx]["ring"] if nx else (175, 175, 185)))
        tab_for[ph["index"]] = tabs[key]
    n_rounds = max((ph["round"] for ph in sched), default=1)
    starts = {}
    for ph in sched:
        starts.setdefault(ph["round"], float(ph["start"]))
    cards = {r: _rgba(ES.round_card(r, E.PROFILES["wait"]["ring"], frame)) for r in range(1, n_rounds + 1)} if signals else {}
    out = {"lay": lay, "boxes": boxes, "digits": digits, "frozen": frozen, "tab_for": tab_for, "cards": cards,
           "round_starts": starts, "signals": signals, "bar": None, "change": {}, "tags": {}}
    if signals:                                            # 🔔 the phase change: the box drops + the frame pulses
        for lk in sorted(set(looks.values())):
            st_ = E.style(lk)
            out["change"][lk] = {"drop": _rgba(ES.drop_panel(st_["label"], st_["ring"], frame)),
                                 "border": [(*_rgba(img), x, y) for img, x, y in ES.border_strips(st_["ring"], frame)]}
    pace = edge.get("pace") or {}
    if pace.get("on") and pace.get("beats"):
        # 🗒️ with instructions the strip is the taller band, the stroke meter in its left (footage) half
        mk_w = W // 2 if band else W
        track, x0, x1 = ES.bar_track(mk_w, H, height=band) if band else ES.bar_track(W, H)
        dim_track, _a, _b = ES.bar_track(mk_w, H, alpha=0.3, height=band) if band else ES.bar_track(W, H, alpha=0.3)
        mk = {lk: _rgba(ES.bar_marker(H, E.style(lk)["ring"])) for lk in set(looks.values()) | set(E.ORDER)}
        held = {lk: _rgba(ES.bar_marker(H, E.style(lk)["ring"], ring=True)) for lk in E.ORDER}
        out["bar"] = {"beats": [float(t) for t in pace["beats"]], "track": _rgba(track), "dim_track": _rgba(dim_track),
                      "y": H - track.size[1], "x0": x0, "x1": x1, "h": track.size[1], "markers": mk, "held": held,
                      "dim": _rgba(ES.bar_marker(H, (0, 0, 0), True)), "grey": _rgba(ES.bar_marker(H, (110, 110, 118))),
                      "hold_label": _rgba(ES.hold_label(H)),
                      "red_seg": _rgba(ES.bar_segment(max(int(x1 - x0), 4), H, (245, 70, 75, 150)))}
        for ph in sched:                                   # ✋ HOLD n / ⏩ +n S — one per second left
            kind = ph.get("kind")
            if kind not in ("handsoff", "override"):
                continue
            for n in range(1, int(math.ceil(float(ph["end"]) - float(ph["start"]))) + 1):
                if kind == "handsoff" and ("hold", n) not in out["tags"]:
                    out["tags"][("hold", n)] = _rgba(ES.tag_image(f"HOLD {n}", (0, 0, 0, 190), (212, 212, 218, 255), H, lock=True))
                if kind == "override" and ("over", n) not in out["tags"]:
                    out["tags"][("over", n)] = _rgba(ES.tag_image(f"+{n} S", (245, 70, 75, 255), ES.DARK + (255,), H))
    return out


def _piece(combined, ph: dict, timers: dict, timer_corner: str, shows: list, av: dict, frame: tuple, out: Path,
           footage: float = None, extras: dict = None) -> None:
    """One phase's picture → out: that stretch of the footage, the avatar appearances (or the parts of them) that
    fall inside it, the phase's timer on top. Read as frames, layered here, written as frames.
    extras: {"flash": 🔔 a quick flash of the phase's colour over its first frames (not the very first phase),
    "badge": ▸ (picture, how solid, x, y) of the round label}."""
    extras = extras or {}
    import numpy as np
    from PIL import Image
    W, H = frame
    a, b = float(ph["start"]), float(ph["end"])
    n_frames = max(int(round((b - a) * FPS)), 1)
    size = W * H * 3
    black = np.zeros((H, W, 3), dtype=np.uint8)

    strip = extras.get("strip")                           # ⏱️ the "strip" timer style (edge_strip.py), or None
    instr = extras.get("instr")                           # 🗒️ the instructions (◧ split screen), or None
    # 👩 what she does in this phase: (first frame, last frame + 1, her frames, how solid (None = a solid box), x, y)
    split = av.get("mode") == "split"
    layers = []
    for s in shows:
        t0, t1 = max(s["time"], a), min(s["time"] + s["seconds"], b)
        if t1 - t0 < 1.0 / FPS:
            continue
        first, last = int(round((t0 - a) * FPS)), int(round((t1 - a) * FPS))
        if split:
            box = (W // 2, H)
        else:
            bh = int(H * E.AVATAR_CORNER_SHARE) // 2 * 2
            cw, ch = _dims(s["clip"]) if s["clip"] else Image.open(s["still"]).size
            box = (max(int(cw * bh / max(ch, 1)) // 2 * 2, 2), bh)
        if s["clip"]:
            got = _read_frames(s["clip"], t0 - s["time"], t1 - t0, box, cover=split)
            if not len(got):
                continue
            solid = _key_alpha(got)
        else:                                              # the stand-in figure: a still, in a thin white frame in the corner
            pic = np.asarray(Image.open(s["still"]).convert("RGB").resize(box), dtype=np.uint8)
            if not split:
                framed = np.full((box[1] + 4, box[0] + 4, 3), 255, dtype=np.uint8)
                framed[2:-2, 2:-2] = pic
                pic, box = framed, (box[0] + 4, box[1] + 4)
            got, solid = pic[None], None
        x, y = (W // 2, 0) if split else _corner_xy(av.get("corner", "bottom-right"), (W, H), box)
        if strip is not None and not split:                # ⏱️ strip style: her box clear of the top box / stroke strip
            if av.get("corner", "bottom-right").startswith("bottom") and strip.get("bar"):
                y = max(y - strip["bar"]["h"], 0)
            elif av.get("corner", "bottom-right").startswith("top"):
                y = min(y + strip["lay"]["h"], H - box[1])
        layers.append((first, last, got, solid, x, y))

    pictures = timers.get(E.timer_key(ph)) or []
    c_total, c_left = E.clock(ph)
    skip = int(round((c_total - c_left) * E.OVERLAY_FPS))      # ✋ the part after a hands off: its clock carries on
    tw, th = (pictures[0][0].shape[1], pictures[0][0].shape[0]) if pictures else (0, 0)
    tx, ty = _corner_xy(timer_corner, (W, H), (tw, th))
    per = FPS / float(E.OVERLAY_FPS)                       # footage frames per timer picture

    dec = None
    if footage is None or a < footage - 0.2:               # (after the footage's end: black, never nothing)
        dec = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(combined), "-an",
                                "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    held = [black]
    flash_n = int(round(E.FLASH_SEC * FPS)) if extras.get("flash") and int(ph.get("index", 0)) > 0 else 0
    tint = np.array(E.style(E.timer_style(ph))["ring"], dtype=np.float32)
    badge = extras.get("badge")
    nxt = extras.get("next")
    bar = extras.get("guide")
    moving = E.phase_bpm(ph) > 0

    def make(i):
        buf = dec.stdout.read(size) if dec is not None else b""
        if len(buf) == size:
            pic = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).copy()
            held[0] = pic
        else:
            pic = held[0].copy()                           # the footage ran short: its last frame is held
        if i < flash_n:                                    # 🔔 the phase's colour washes over, gone within FLASH_SEC
            k = 0.55 * (1.0 - i / flash_n)
            pic = (pic.astype(np.float32) * (1.0 - k) + tint * k).astype(np.uint8)
        on = next((ly for ly in layers if ly[0] <= i < ly[1]), None)
        if on is not None:
            if split:                                      # the footage's middle on the left, she on the right
                pic[:, :W // 2] = pic[:, W // 4:W // 4 + W // 2].copy()
                pic[:, W // 2:] = 0
            j = min(i - on[0], len(on[2]) - 1)
            _lay(pic, on[4], on[5], on[2][j], None if on[3] is None else on[3][j])
        if pictures:
            rgb, solid = pictures[min(skip + int(i / per), len(pictures) - 1)]
            _lay(pic, tx, ty, rgb, solid)
        if badge is not None:
            _lay(pic, badge[2], badge[3], badge[0], badge[1])
        if nxt is not None:
            _lay(pic, nxt[2], nxt[3], nxt[0], nxt[1])
        if instr is not None:                              # 🗒️ her half of the bottom band
            text = E.instruction_at(instr["track"], a + i / FPS)
            panel = instr["panels"].get((text, ph["profile"])) if text else instr["empty"]
            if panel is not None:
                _lay(pic, W // 2, H - instr["band"], *panel)
        if strip is not None:
            _lay_strip(pic, strip, ph, i, a, c_left, moving)
        if bar is not None:                                # 🎯 the stroke bar: the marker slides top ↔ bottom on the beats
            _lay(pic, bar["track_x"], bar["y"], *bar["track"])
            pos = E.guide_position(bar["beats"], a + i / FPS) if moving else None
            m = bar["markers"][ph["profile"]] if pos is not None else bar["dim"]
            pos = 0.5 if pos is None else pos
            _lay(pic, bar["marker_x"], int(round(bar["y"] + pos * (bar["height"] - bar["marker_h"]))), *m)
        return pic
    try:
        _write_video(out, frame, n_frames, make, [dec] if dec is not None else None)
    finally:
        if dec is not None:
            try:
                dec.kill()
            except OSError:
                pass
            dec.wait()


def _instruction_layers(edge: dict, frame: tuple) -> dict | None:
    """🗒️ The instruction panels for her half of the bottom band (◧ split screen, switched on): one per text and
    phase colour in use, and an empty band before the first. None when off."""
    import edge_strip as ES
    ins = edge.get("instructions") or {}
    if not ins.get("on") or (edge.get("avatar") or {}).get("mode") != "split":
        return None
    W, H = frame
    track = [x for x in ins.get("items") or [] if str(x.get("text") or "").strip()]
    panels = {}
    for ph in edge["schedule"]:
        t0, t1 = float(ph["start"]), float(ph["end"])
        texts = {E.instruction_at(track, t0)} | {x["text"] for x in track if t0 <= float(x["time"]) < t1}
        for text in texts - {None}:
            if (text, ph["profile"]) not in panels:
                panels[(text, ph["profile"])] = _rgba(ES.instruction_panel(text, E.PROFILES[ph["profile"]]["ring"],
                                                                           W - W // 2, H))
    empty = _rgba(ES.instruction_panel("", (0, 0, 0), W - W // 2, H))
    return {"track": track, "panels": panels, "empty": empty, "band": ES.band_height(H)}


def _lay_strip(pic, strip: dict, ph: dict, i: int, a: float, c_left: float, moving: bool) -> None:
    """⏱️ One frame of the strip style: the NEXT tab and the top box (⚠️ flashing as the last numbers come up), the
    number, ▸ the round card, 🔔 the phase change, then the bottom stroke strip with its marker (✋ / ⏩ tags)."""
    import numpy as np
    import edge_strip as ES
    lay, t = strip["lay"], i / FPS
    left = c_left - t
    look = E.timer_style(ph)
    kind = ph.get("kind") or "phase"
    sig = strip["signals"]
    length = float(ph["end"]) - float(ph["start"])
    c_total, _cl = E.clock(ph)
    fresh = kind != "phase" or c_left >= c_total - 1e-3      # (a phase carrying on after a hands off isn't new)
    to_zero = kind == "phase" and abs(c_left - length) < 0.05  # its clock runs out at its end
    ending = to_zero and 0 < left <= ES.WARN_SEC
    warn = sig and ending and (math.ceil(left - 1e-6) - left) < 0.5      # ⚠️ on as each number comes up
    box = strip["boxes"][look].get("warn" if warn else "edge" if (sig and ending) else "normal") or strip["boxes"][look]["normal"]
    # ⏭️ the NEXT tab slides out from under the box over the last seconds
    if ending and sig:
        tab = strip["tab_for"].get(ph["index"])
        if tab is not None:
            share = min((ES.WARN_SEC - left) / ES.TAB_SEC, 1.0)
            th = tab[0].shape[0]
            y = int(round(lay["h"] - th + share * (th - 2)))
            _lay(pic, (pic.shape[1] - tab[0].shape[1]) // 2, y, tab[0], (tab[1].astype(np.float32) * share).astype(np.uint16))
    _lay(pic, lay["x"] - box[2], 0, box[0], box[1])
    if kind == "handsoff":
        text = strip["frozen"].get(ph["index"], "")
    else:
        text = ES.number_text(look, max(left, 0.0))
    dg = strip["digits"].get((text, bool(ending)))
    if dg is not None:
        _lay(pic, lay["x"], 0, *dg)
    # ▸ the round card, once as each round starts
    card = strip["cards"].get(ph["round"])
    if card is not None:
        al = ES.round_alpha(a + t - strip["round_starts"][ph["round"]], ph["round"] == 1)
        if al > 0.01:
            _lay(pic, (pic.shape[1] - card[0].shape[1]) // 2, int(pic.shape[0] * 0.33), card[0],
                 (card[1].astype(np.float32) * al).astype(np.uint16))
    ch = strip["change"].get(look) if sig and int(ph.get("index", 0)) > 0 and fresh else None
    if ch is not None and t < ES.CHANGE_SEC:               # 🔔 a new phase: the frame pulses, the box drops down
        lvl = ES.border_level(t)
        if lvl > 0.01:
            for rgb, solid, x, y in ch["border"]:
                _lay(pic, x, y, rgb, (solid.astype(np.float32) * lvl).astype(np.uint16))
        share = ES.drop_share(t)
        if share > 0.01:
            rgb, solid = ch["drop"]
            hh = max(int(rgb.shape[0] * share), 1)
            _lay(pic, (pic.shape[1] - rgb.shape[1]) // 2, 0, rgb[-hh:], solid[-hh:])
    bar = strip.get("bar")
    if bar is None:
        return
    span = bar["x1"] - bar["x0"]
    if kind == "handsoff":                                 # ✋ the bar dims and locks, the marker stops in the middle
        _lay(pic, 0, bar["y"], *bar["dim_track"])
        m = bar["grey"]
        _lay(pic, int(round(bar["x0"] + 0.5 * (span - m[0].shape[1]))), bar["y"] + (bar["h"] - m[0].shape[0]) // 2, *m)
        tag = strip["tags"].get(("hold", max(int(math.ceil(length - t - 1e-6)), 1)))
        if tag is not None:
            _lay(pic, int(bar["x0"] + span / 2 - tag[0].shape[1] / 2), bar["y"] - tag[0].shape[0] - int(10 * lay["k"]), *tag)
        return
    _lay(pic, 0, bar["y"], *bar["track"])
    if kind == "override":                                 # ⏩ the track turns red
        seg = bar["red_seg"]
        _lay(pic, bar["x0"], bar["y"] + (bar["h"] - seg[0].shape[0]) // 2, *seg)
    lo, hi, hold = 0.0, 1.0, None
    if strip.get("instr") is not None:                   # 🎯 the instruction on screen limits the stroke (21 Nov)
        item = E.instruction_item_at(strip["instr"]["track"], a + t)
        if item and item.get("range"):
            lo, hi = float(item["range"][0]), float(item["range"][1])
            seg = strip["instr"]["segments"].get((round(lo, 4), round(hi, 4)))
            if seg is not None and kind != "override":
                _lay(pic, int(round(bar["x0"] + lo * span)), bar["y"] + (bar["h"] - seg[0].shape[0]) // 2, *seg)
        hold = (item or {}).get("hold") if kind == "phase" else None
    holding = False
    if moving and hold:
        pos, holding = E.guide_position_hold(bar["beats"], a + t, hold["at"], hold["beats"])
    else:
        pos = E.guide_position(bar["beats"], a + t) if moving else None
    if pos is None:
        m = bar["dim"]
    elif holding:
        m = bar["held"].get(ph["profile"]) or bar["markers"][look]
    else:
        m = bar["markers"].get(look) or bar["markers"][ph["profile"]]
    mw, mh = m[0].shape[1], m[0].shape[0]
    pos = 0.5 if pos is None else pos
    run = span * (hi - lo) - mw
    x = int(round(bar["x0"] + span * lo + pos * max(run, 0)))
    _lay(pic, x, bar["y"] + (bar["h"] - mh) // 2, *m)
    if holding:
        hl = bar["hold_label"]
        _lay(pic, x + mw // 2 - hl[0].shape[1] // 2, bar["y"] + (bar["h"] - mh) // 2 - hl[0].shape[0] - 2, *hl)
    if kind == "override":
        tag = strip["tags"].get(("over", max(int(math.ceil(length - t - 1e-6)), 1)))
        if tag is not None:
            _lay(pic, int(bar["x1"] - tag[0].shape[1]), bar["y"] - tag[0].shape[0] - int(10 * lay["k"]), *tag)


def _lengths(path) -> dict:
    """{"video", "audio"} seconds of a finished file (0 where it can't be read)."""
    out = {"video": 0.0, "audio": 0.0}
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=120)
        for st in json.loads(r.stdout or "{}").get("streams") or []:
            if st.get("codec_type") in out:
                out[st["codec_type"]] = float(st.get("duration") or 0.0)
    except Exception:          # noqa: BLE001
        pass
    return out


def ffmpeg_version() -> str:
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=20).stdout.split("\n")[0][:80]
    except Exception:          # noqa: BLE001
        return "?"


def finish(plan: dict, combined_video, output_path: str, sources, tmp_dir, run, frame_height: int,
           frame_width: int = 640, progress=None, workers: int = None) -> dict:
    """combined_video (the plan's picture, no sound) → output_path with the avatar, the timer overlay and the mixed
    sound, after the intro screen when there is one. sources: the renderer's SourceResolver (songs are read from
    this machine, downloaded first when they aren't here); run: its ffmpeg runner; progress(0–1): how far the
    picture is; workers: phases made at once. Returns what each step took and what the end check found:
    {"edge_sound", "edge_screens", "edge_picture", "edge_join" (seconds), "edge_check" (text), "ffmpeg"}."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    edge = plan["edge"]
    sched = edge["schedule"]
    total = E.total_seconds(sched)
    tmp_dir = Path(tmp_dir)
    W, H = int(frame_width), int(frame_height)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    took, t0 = {"ffmpeg": ffmpeg_version()}, time.time()
    shows = _appearances(edge, tmp_dir, (W, H))

    # 1 · the sound, by itself (seconds; a problem with a song or a line shows here, not after the picture)
    music_notes = []
    mix = _mix(edge, shows, sources, tmp_dir, run, music_notes)
    took["edge_sound"] = round(time.time() - t0, 1)
    if music_notes:
        took["edge_notes"] = music_notes

    # 2 · the intro and outro screens
    t0 = time.time()
    intro = _card(edge.get("intro"), tmp_dir, run, (W, H), "intro")
    outro = _card(edge.get("outro"), tmp_dir, run, (W, H), "outro")
    took["edge_screens"] = round(time.time() - t0, 1)

    # 3 · the picture, a phase at a time
    t0 = time.time()
    timer_dir = tmp_dir / "overlay"
    signals = edge.get("signals") or {}
    corner = (edge.get("overlay") or {}).get("corner", "top-left")
    strip = None
    instr = _instruction_layers(edge, (W, H))
    if (edge.get("overlay") or {}).get("style") == "strip":          # ⏱️ the strip style: no ring pictures
        strip = _strip_layers(edge, sched, (W, H), bool(signals.get("timer")), instr["band"] if instr else None)
        if instr is not None and strip.get("bar"):        # 🎯 a highlight over the part of the bar an instruction allows
            import edge_strip as ES
            strip["instr"] = instr
            span = strip["bar"]["x1"] - strip["bar"]["x0"]
            instr["segments"] = {(round(float(x["range"][0]), 4), round(float(x["range"][1]), 4)):
                                 _rgba(ES.bar_segment(max(int(span * (float(x["range"][1]) - float(x["range"][0]))), 4), H))
                                 for x in instr["track"] if x.get("range")}
        timers = {}
    else:
        E.write_overlay(sched, timer_dir, H, signals=bool(signals.get("timer")))
        timers = _timer_pictures(timer_dir, sched)
    av = edge.get("avatar") or {}
    pieces = [tmp_dir / f"phase_{k:03d}.mp4" for k in range(len(sched))]
    footage = probe(combined_video)["seconds"] or None
    badges = {}
    if strip is None and (edge.get("round_badge") or {}).get("on") and sched:          # ▸ "ROUND 2 / 3" by the timer
        import numpy as np
        tw, th = E.overlay_size(H)
        n_rounds = max(ph["round"] for ph in sched)
        for r in range(1, n_rounds + 1):
            a = np.asarray(E.round_badge(r, n_rounds, tw).convert("RGBA"), dtype=np.uint8)
            rgb, solid = a[..., :3].copy(), a[..., 3:4].astype(np.uint16)
            badges[r] = (rgb, solid, *_badge_xy(corner, (W, H), (tw, th), (rgb.shape[1], rgb.shape[0])))
    guide = _guide_layers(edge, sched, corner, (W, H), badges) if strip is None else {}
    n = max(1, min(int(workers or min(4, os.cpu_count() or 2)), len(sched) or 1))
    with ThreadPoolExecutor(max_workers=n) as pool:
        jobs = [pool.submit(_piece, combined_video, ph, timers, corner, shows, av, (W, H), piece, footage,
                            {"flash": bool(signals.get("flash")) and strip is None, "badge": badges.get(ph["round"]),
                             "guide": guide.get("bar"), "next": guide.get("next", {}).get(ph["index"]),
                             "strip": strip, "instr": instr})
                for ph, piece in zip(sched, pieces)]
        try:
            for done, job in enumerate(as_completed(jobs), 1):
                job.result()                               # raises here if the phase failed
                if progress:
                    progress(done / max(len(jobs), 1))
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    took["edge_picture"] = round(time.time() - t0, 1)

    # 4 · joined as they are (no second encode), the sound added
    t0 = time.time()
    order = ([intro[0]] if intro else []) + pieces + ([outro[0]] if outro else [])
    lst = tmp_dir / "pieces.txt"
    lst.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in order))
    sounds = ([intro[1]] if intro else []) + [mix] + ([outro[1]] if outro else [])
    length = total + (intro[2] if intro else 0.0) + (outro[2] if outro else 0.0)
    inputs = ["-f", "concat", "-safe", "0", "-i", str(lst)]
    for f in sounds:
        inputs += ["-i", str(f)]
    af = "".join(f"[{k + 1}:a:0]{AUDIO_FMT}[s{k}];" for k in range(len(sounds)))
    af += "".join(f"[s{k}]" for k in range(len(sounds))) + f"concat=n={len(sounds)}:v=0:a=1[a]"
    _run(["ffmpeg", "-y", *inputs, "-filter_complex", af, "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
          "-c:a", "aac", "-b:a", "192k", "-t", f"{length:.3f}", "-movflags", "+faststart", output_path], timeout=900)
    took["edge_join"] = round(time.time() - t0, 1)

    # ✅ the check: picture and sound as long as the plan says
    got = _lengths(output_path)
    off = [f"{w} {got[k]:.1f} s" for k, w in (("video", "picture"), ("audio", "sound")) if abs(got[k] - length) > 0.5]
    took["edge_check"] = "ok" if not off else f"expected {length:.1f} s — " + ", ".join(off)
    return took
