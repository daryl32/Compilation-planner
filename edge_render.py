"""
⏱️ Edge Mode in the rendered video (called by render_preview.render_plan_dict in place of its plain "add the
track's audio" step, for a plan with an "edge" block): the 👩 avatar pops up at her cues (a framed box in a
corner, or the right half of a split screen), the countdown overlay goes on top of the picture, a 🎬 intro screen
(the avatar beside the chosen settings and thumbnails) can come first, and the
sound is mixed — the playlist's songs back to back (starting again until the time is filled, at a quarter volume
while the voiceover or the avatar is on), the generated beat, and the voiceover lines at their cues. Nothing here
chooses footage.

Two steps (14 Nov 2026): the SOUND is mixed first, by itself, from files on this machine (a song that isn't here is
downloaded once — never streamed: a stream left idle for minutes while the picture is encoded is what cut songs
short and stalled the render); then ONE pass over the picture adds the timer and the avatar. That pass is watched:
it reports how far it is, and is stopped with ffmpeg's own message if it stands still.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import edge_mode as E

AUDIO_FMT = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"
STALL_SEC = 180                    # the picture pass is stopped when it hasn't moved for this long
MIN_SHOW_SEC = 0.2                 # an avatar appearance shorter than this is left out
PICTURE_PRESET = "ultrafast"       # the picture pass's encoder speed (was veryfast: about twice as long)


def playlist_raws(plan: dict) -> list:
    """The raw source of every song an Edge plan plays (for the renderer's "can I read this" check)."""
    from render_preview import get_track_raw_source
    return list(dict.fromkeys(get_track_raw_source(p["track_id"]) for p in (plan.get("edge") or {}).get("playlist") or []))


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
            f = placeholders[p]
        inputs += ["-i", str(f)]
        ms = int(round(float(cue["time"]) * 1000))
        parts.append(f"[{k}:a:0]{AUDIO_FMT},adelay={ms}:all=1[c{k}]")
    out = tmp_dir / name
    mix = "".join(f"[c{k}]" for k in range(len(cues)))
    run(["ffmpeg", "-y", *inputs, "-filter_complex",
         ";".join(parts) + f";{mix}amix=inputs={len(cues)}:normalize=0:duration=longest[a]",
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
    """The avatar's appearances, ready to draw: [{"time", "seconds", "profile", "input" (ffmpeg input args), "sound"
    (the clip, when her voice is in it)}]. A cue without a clip gets the stand-in figure for a few seconds. An
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
                        "input": ["-t", f"{secs:.3f}", "-i", str(clip)], "sound": str(clip) if info["audio"] else None})
        else:
            if still is None:
                still = tmp_dir / "avatar_placeholder.png"
                E.avatar_placeholder((int(h * 0.6) // 2 * 2, h)).save(still)
            secs = min(float(cue.get("seconds") or E.PLACEHOLDER_CUE_SEC), room)      # as long as her line, when known
            if secs < MIN_SHOW_SEC:
                continue
            out.append({"time": float(cue["time"]), "seconds": secs, "profile": cue["profile"],
                        "input": ["-loop", "1", "-framerate", "25", "-t", f"{secs:.3f}", "-i", str(still)], "sound": None})
    return out


def _sound_cues(edge: dict, shows: list) -> list:
    """What is heard at each cue: the avatar's own clip where her voice is in it, else the voiceover clip for that
    moment, else (None) the placeholder tone."""
    voice = edge.get("voiceover") or {}
    by_time = {round(float(c["time"]), 3): dict(c) for c in (voice.get("cues") or [])} if voice.get("on") else {}
    for s in shows:
        key = round(s["time"], 3)
        if s["sound"]:
            by_time[key] = {"time": s["time"], "profile": s["profile"], "file": s["sound"]}
        elif key not in by_time:
            by_time[key] = {"time": s["time"], "profile": s["profile"], "file": None}
    return [by_time[k] for k in sorted(by_time)]


CARD_LEAD = 0.4                    # silence before the first line and after the last on the intro / outro screen


def _card(spec: dict, tmp_dir: Path, run, frame: tuple, name: str):
    """An intro or outro screen as a clip (picture + sound) → (path, seconds), or None when it is off.
    spec: {"on", "clip" (the avatar's video, voice included), "lines" ([{"file", "seconds"}] — voiceover lines read
    one after another), "stats", "thumbs", "title"}. Its length: the avatar's clip when there is one; else the
    lines back to back; else the stand-in figure for PLACEHOLDER_INTRO_SEC, silent."""
    if not (spec or {}).get("on"):
        return None
    thumbs = []
    try:
        from planner.data import get_nearest_thumbnail, load_all_catalogues
        cats = load_all_catalogues()
        thumbs = [get_nearest_thumbnail(cats, vid, float(t)) for vid, t in spec.get("thumbs") or []]
    except Exception:          # noqa: BLE001 — the screen still works without thumbnails
        thumbs = []
    clip = spec.get("clip")
    info = probe(clip) if clip and Path(clip).exists() else None
    real = bool(info and info["video"] and info["seconds"] > 0)
    lines = [] if real and info["audio"] else list(spec.get("lines") or [])
    ax, ay, aw, ah = E.intro_layout(frame)["avatar"]
    fit = (f"scale={aw}:{ah}:force_original_aspect_ratio=decrease,setsar=1,fps=25")
    spot = f"x={ax}+({aw}-w)/2:y={ay}+({ah}-h)/2"
    cues, shows, t = [], [], CARD_LEAD
    for ln in lines:                                       # the lines one after another; her clip for a line when there is one
        f, c = ln.get("file"), ln.get("clip")
        ci = probe(c) if c and Path(c).exists() else None
        has_clip = bool(ci and ci["video"] and ci["seconds"] > 0)
        secs = (ci["seconds"] if has_clip else (probe(f)["seconds"] if f and Path(f).exists() else 0.0)) or float(ln.get("seconds") or 2.0)
        cues.append({"time": t, "profile": "wait", "file": c if has_clip and ci["audio"] else f})
        if has_clip:
            shows.append({"time": t, "seconds": secs, "clip": c})
        t += secs + E.LINE_GAP
    secs = info["seconds"] if real else (t - E.LINE_GAP + CARD_LEAD if cues else E.PLACEHOLDER_INTRO_SEC)
    card = tmp_dir / f"{name}_card.png"
    E.intro_card(frame, [tuple(x) for x in spec.get("stats") or []], thumbs, title=spec.get("title") or "EDGE MODE",
                 placeholder=not real and len(shows) < max(len(lines), 1)).save(card)
    out = tmp_dir / f"{name}.mp4"
    inputs = ["-loop", "1", "-framerate", "25", "-t", f"{secs:.3f}", "-i", str(card)]
    nxt = 1
    if real:
        inputs += ["-t", f"{secs:.3f}", "-i", str(clip)]
        vf = f"[1:v]{fit}[av];[0:v][av]overlay={spot}:eof_action=pass,fps=25,format=yuv420p[v]"
        nxt = 2
    elif shows:
        parts, cur = [], "[0:v]"
        for k, sh in enumerate(shows):                     # she holds her last frame until the next line starts
            until = shows[k + 1]["time"] if k + 1 < len(shows) else secs
            inputs += ["-t", f"{sh['seconds']:.3f}", "-i", str(sh["clip"])]
            parts.append(f"[{nxt}:v]{fit},setpts=PTS-STARTPTS+{sh['time']:.3f}/TB[s{k}]")
            parts.append(f"{cur}[s{k}]overlay={spot}:enable='between(t,{sh['time']:.3f},{until:.3f})'[c{k}]")
            cur = f"[c{k}]"
            nxt += 1
        vf = ";".join(parts) + f";{cur}fps=25,format=yuv420p[v]"
    else:
        vf = "[0:v]fps=25,format=yuv420p[v]"
    if real and info["audio"]:
        af = f"[1:a:0]{AUDIO_FMT}[a]"
    elif cues:
        inputs += ["-i", str(_voice_track(cues, secs, tmp_dir, run, f"{name}_voice.wav"))]
        af = f"[{nxt}:a:0]{AUDIO_FMT},apad,atrim=0:{secs:.3f}[a]"
    else:
        inputs += ["-f", "lavfi", "-t", f"{secs:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
        af = f"[{nxt}:a:0]{AUDIO_FMT}[a]"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]", "-t", f"{secs:.3f}",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-c:a", "aac", "-b:a", "192k", str(out)])
    return out, secs


def _song_input(sources, raw: str) -> list:
    """ffmpeg input arguments for a song, from a file on this machine — downloaded first when it isn't here."""
    if hasattr(sources, "is_local") and not sources.is_local(raw):
        errs = sources.download([raw])
        if errs:
            raise FileNotFoundError("A song couldn't be fetched for the render:\n" + "\n".join(errs))
    return list(sources.input_args(raw))


def _mix(edge: dict, shows: list, sources, tmp_dir: Path, run) -> Path:
    """The compilation's whole sound as one file: the songs back to back (quieter under the voiceover / avatar and
    in Wait and Stop), the generated beat, the lines."""
    from render_preview import get_track_raw_source
    sched = edge["schedule"]
    total = E.total_seconds(sched)
    inputs, filters, layers, idx = [], [], [], 0
    voice = edge.get("voiceover") or {}
    talking = bool(voice.get("on") or shows)
    parts = edge.get("playlist") or []
    if parts:
        for k, part in enumerate(parts):
            inputs += ["-t", f"{float(part['play']):.3f}", *_song_input(sources, get_track_raw_source(part["track_id"]))]
            # (":a:0" — an .m4a may carry its cover art as a second stream)
            filters.append(f"[{idx}:a:0]{AUDIO_FMT},atrim=0:{float(part['play']):.3f},asetpts=PTS-STARTPTS[p{k}]")
            idx += 1
        vol = float(voice.get("music_volume", E.MUSIC_UNDER_VOICE)) if talking else 1.0
        # 🔉 down to a quarter during Wait and Stop (a short fade each way) when it is louder than that elsewhere
        quiet = float((edge.get("music") or {}).get("still_volume", E.MUSIC_IN_STILL))
        spans = E.still_spans(sched) if quiet < vol else []
        if spans:
            f = E.MUSIC_FADE_SEC
            dip = "0"
            for a, b in spans:
                dip = f"max({dip},clip((t-{a:.3f})/{f},0,1)*clip(({b:.3f}-t)/{f},0,1))"
            level = f"volume=volume='{vol:.3f}-{vol - quiet:.3f}*{dip}':eval=frame"
        else:
            level = f"volume={vol:.3f}"
        filters.append("".join(f"[p{k}]" for k in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1,{level}[music]")
        layers.append("[music]")
    beat = edge.get("beat") or {}
    if beat.get("on"):
        wav = E.write_beat_wav(sched, tmp_dir / "beat.wav")
        inputs += ["-i", str(wav)]
        filters.append(f"[{idx}:a:0]{AUDIO_FMT},volume={float(beat.get('volume', 1.0)):.3f}[beat]")
        layers.append("[beat]")
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


def finish(plan: dict, combined_video, output_path: str, sources, tmp_dir, run, frame_height: int,
           frame_width: int = 640, progress=None) -> dict:
    """combined_video (the plan's picture, no sound) → output_path with the avatar, the timer overlay and the mixed
    sound, after the intro screen when there is one. sources: the renderer's SourceResolver (songs are read from
    this machine, downloaded first when they aren't here); run: its ffmpeg runner; progress(0–1): how far the
    picture pass is. Returns the seconds each step took: {"edge_sound", "edge_screens", "edge_picture"}."""
    edge = plan["edge"]
    sched = edge["schedule"]
    total = E.total_seconds(sched)
    tmp_dir = Path(tmp_dir)
    W, H = int(frame_width), int(frame_height)
    took, t0 = {}, time.time()
    shows = _appearances(edge, tmp_dir, (W, H))

    # 1 · the sound, by itself (seconds; a problem with a song or a line shows here, not after the picture pass)
    mix = _mix(edge, shows, sources, tmp_dir, run)
    took["edge_sound"] = round(time.time() - t0, 1)

    # 2 · the intro and outro screens
    t0 = time.time()
    intro = _card(edge.get("intro"), tmp_dir, run, (W, H), "intro")
    outro = _card(edge.get("outro"), tmp_dir, run, (W, H), "outro")
    took["edge_screens"] = round(time.time() - t0, 1)

    # 3 · the picture: the timer on top, the avatar under it, the screens before and after
    t0 = time.time()
    lst, _size = E.write_overlay(sched, tmp_dir / "overlay", H)
    x, y = E.overlay_xy((edge.get("overlay") or {}).get("corner", "top-left"))
    inputs = ["-i", str(combined_video), "-f", "concat", "-safe", "0", "-i", str(lst)]
    filters, cur, idx = [], "[0:v]", 2

    # 👩 the avatar: in a framed box in a corner, or as the right half with the footage's middle on the left
    av = edge.get("avatar") or {}
    if shows:
        split = av.get("mode") == "split"
        when = lambda s: f"between(t,{s['time']:.3f},{s['time'] + s['seconds']:.3f})"      # noqa: E731
        if split:
            filters.append(f"{cur}split[m0][m1]")
            filters.append(f"[m1]crop={W // 2}:{H}:{W // 4}:0,pad={W}:{H}:0:0:black[half]")
            filters.append(f"[m0][half]overlay=0:0:enable='{'+'.join(when(s) for s in shows)}'[sp]")
            cur = "[sp]"
            shape = f"scale={W // 2}:{H}:force_original_aspect_ratio=increase,crop={W // 2}:{H}"
            ax, ay = str(W // 2), "0"
        else:
            bh = int(H * E.AVATAR_CORNER_SHARE) // 2 * 2
            shape = f"scale=-2:{bh},pad=iw+4:ih+4:2:2:white"
            ax, ay = E.overlay_xy(av.get("corner", "bottom-right"))
        for k, s in enumerate(shows):
            inputs += ["-an", *s["input"]]                 # (her voice is already in the mix)
            filters.append(f"[{idx}:v]{shape},setsar=1,fps=25,setpts=PTS-STARTPTS+{s['time']:.3f}/TB[av{k}]")
            filters.append(f"{cur}[av{k}]overlay=x={ax}:y={ay}:enable='{when(s)}':eof_action=pass[a{k}v]")
            cur = f"[a{k}v]"
            idx += 1
    filters += ["[1:v]format=rgba[ov]", f"{cur}[ov]overlay=x={x}:y={y}:eof_action=pass,fps=25,format=yuv420p[mainv]"]
    inputs += ["-i", str(mix)]
    filters.append(f"[{idx}:a:0]{AUDIO_FMT}[maina]")
    idx += 1
    length = total
    if intro or outro:                                     # 🎬 the intro screen, the compilation, the outro screen
        filters.append(f"[mainv]trim=0:{total:.3f},setpts=PTS-STARTPTS[mv]")
        parts_v = []
        for tag, card in (("i", intro), ("m", None), ("o", outro)):
            if tag == "m":
                parts_v.append("[mv][maina]")
            elif card:
                inputs += ["-i", str(card[0])]
                filters.append(f"[{idx}:v]fps=25,format=yuv420p,setsar=1[{tag}v]")
                filters.append(f"[{idx}:a:0]{AUDIO_FMT}[{tag}a]")
                parts_v.append(f"[{tag}v][{tag}a]")
                idx += 1
                length += card[1]
        filters.append("".join(parts_v) + f"concat=n={len(parts_v)}:v=1:a=1[v][a]")
    else:
        filters += ["[mainv]null[v]", "[maina]anull[a]"]
    run_watched(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
                 "-t", f"{length:.3f}", "-c:v", "libx264", "-preset", PICTURE_PRESET, "-crf", "23", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path], length, progress)
    took["edge_picture"] = round(time.time() - t0, 1)
    return took
