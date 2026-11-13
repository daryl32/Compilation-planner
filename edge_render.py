"""
⏱️ Edge Mode in the rendered video (called by render_preview.render_plan_dict in place of its plain "add the
track's audio" step, for a plan with an "edge" block): the countdown overlay goes on top of the picture, and the
sound is mixed — the playlist's songs back to back (starting again until the time is filled, at half volume while
the voiceover is on), the generated beat, and a voiceover cue at the start of every phase. One extra pass over the
finished picture; nothing here chooses footage.
"""

from __future__ import annotations

from pathlib import Path

import edge_mode as E

AUDIO_FMT = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"


def playlist_raws(plan: dict) -> list:
    """The raw source of every song an Edge plan plays (for the renderer's "can I read this" check)."""
    from render_preview import get_track_raw_source
    return list(dict.fromkeys(get_track_raw_source(p["track_id"]) for p in (plan.get("edge") or {}).get("playlist") or []))


def _voice_track(cues: list, total: float, tmp_dir: Path, run) -> Path | None:
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
    out = tmp_dir / "voice.wav"
    mix = "".join(f"[c{k}]" for k in range(len(cues)))
    run(["ffmpeg", "-y", *inputs, "-filter_complex",
         ";".join(parts) + f";{mix}amix=inputs={len(cues)}:normalize=0:duration=longest[a]",
         "-map", "[a]", "-t", f"{total:.3f}", str(out)])
    return out


def finish(plan: dict, combined_video, output_path: str, sources, tmp_dir, run, frame_height: int) -> None:
    """combined_video (the plan's picture, no sound) → output_path with the overlay and the mixed sound.
    sources: the renderer's SourceResolver (songs are read from this machine or streamed from Drive, like the
    videos); run: its ffmpeg runner."""
    from render_preview import get_track_raw_source
    edge = plan["edge"]
    sched = edge["schedule"]
    total = E.total_seconds(sched)
    tmp_dir = Path(tmp_dir)
    lst, _size = E.write_overlay(sched, tmp_dir / "overlay", frame_height)
    x, y = E.overlay_xy((edge.get("overlay") or {}).get("corner", "top-left"))
    inputs = ["-i", str(combined_video), "-f", "concat", "-safe", "0", "-i", str(lst)]
    filters = [f"[1:v]format=rgba[ov]", f"[0:v][ov]overlay=x={x}:y={y}:eof_action=pass[v]"]
    idx, layers = 2, []

    voice = edge.get("voiceover") or {}
    parts = edge.get("playlist") or []
    if parts:
        for k, part in enumerate(parts):
            inputs += ["-t", f"{float(part['play']):.3f}", *sources.input_args(get_track_raw_source(part["track_id"]))]
            # (":a:0" — an .m4a may carry its cover art as a second stream)
            filters.append(f"[{idx}:a:0]{AUDIO_FMT},atrim=0:{float(part['play']):.3f},asetpts=PTS-STARTPTS[p{k}]")
            idx += 1
        vol = float(voice.get("music_volume", E.MUSIC_UNDER_VOICE)) if voice.get("on") else 1.0
        filters.append("".join(f"[p{k}]" for k in range(len(parts)))
                       + f"concat=n={len(parts)}:v=0:a=1,volume={vol:.3f}[music]")
        layers.append("[music]")
    beat = edge.get("beat") or {}
    if beat.get("on"):
        wav = E.write_beat_wav(sched, tmp_dir / "beat.wav")
        inputs += ["-i", str(wav)]
        filters.append(f"[{idx}:a:0]{AUDIO_FMT},volume={float(beat.get('volume', 1.0)):.3f}[beat]")
        layers.append("[beat]")
        idx += 1
    if voice.get("on"):
        vt = _voice_track(voice.get("cues") or [], total, tmp_dir, run)
        if vt is not None:
            inputs += ["-i", str(vt)]
            filters.append(f"[{idx}:a:0]{AUDIO_FMT}[voice]")
            layers.append("[voice]")
            idx += 1
    if not layers:                                         # nothing to hear (shouldn't happen: no songs → beat on)
        inputs += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
        filters.append(f"[{idx}:a:0]anull[a]")
    elif len(layers) == 1:
        filters.append(f"{layers[0]}alimiter=limit=0.95[a]")
    else:
        filters.append("".join(layers) + f"amix=inputs={len(layers)}:normalize=0:duration=longest,alimiter=limit=0.95[a]")
    run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
         "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path])
