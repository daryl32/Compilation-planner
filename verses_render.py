"""⚔️ Verses Battle — the render's last step (called by render_preview.render_plan_dict when a plan has a "verses"
block): the battle's sound (one shared song, or each fighter's theme switching with the picture), the overlay (names,
VS, the round, the tug-of-war meter, the winner) and the tale-of-the-tape card in front — one encode.
The rules are in verses_battle.py."""

from __future__ import annotations

import time
from pathlib import Path

import verses_battle as VB

AUDIO_FMT = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"
MUSIC_LEVEL = 0.9
FPS = 25


def playlist_raws(plan: dict) -> list:
    """The raw source of every song the battle plays (for the renderer's "can I read this" check)."""
    from render_preview import get_track_raw_source
    v = plan.get("verses") or {}
    return list(dict.fromkeys(get_track_raw_source(p["track_id"]) for p in v.get("parts") or []))


def _song_input(sources, raw: str) -> list:
    """A song from this machine — downloaded first when it isn't here (it's read many times over)."""
    if hasattr(sources, "is_local") and not sources.is_local(raw):
        errs = sources.download([raw])
        if errs:
            raise FileNotFoundError("A song couldn't be fetched for the render:\n" + "\n".join(errs))
    return list(sources.input_args(raw))


def mix(plan: dict, sources, tmp_dir: Path, run) -> Path:
    """The battle's sound as one WAV, battle-long: every part of a song at its place, a short crossfade at each switch,
    a fade out at the end when the battle stops before the song does."""
    from render_preview import get_track_raw_source
    v = plan["verses"]
    total = float(v["total_seconds"])
    parts = v.get("parts") or []
    inputs, filters, labels = [], [], []
    by_track = {}
    for p in parts:
        by_track.setdefault(p["track_id"], []).append(p)
    idx = 0
    xf = VB.XFADE_SEC
    for tid, ps in by_track.items():
        inputs += _song_input(sources, get_track_raw_source(tid))
        filters.append(f"[{idx}:a:0]{AUDIO_FMT},asplit={len(ps)}" + "".join(f"[t{idx}_{k}]" for k in range(len(ps))))
        for k, p in enumerate(ps):
            lead = min(xf, float(p["at"]), float(p["src"]))       # start a little early, fading in under the switch
            a, d = float(p["src"]) - lead, float(p["dur"]) + lead + (xf if p["at"] + p["dur"] < total - 1e-3 else 0.0)
            at_ms = int(round((float(p["at"]) - lead) * 1000))
            chain = f"[t{idx}_{k}]atrim={a:.3f}:{a + d:.3f},asetpts=PTS-STARTPTS"
            if len(parts) > 1:
                chain += f",afade=t=in:d={max(lead, 0.01):.3f},afade=t=out:st={max(d - xf, 0):.3f}:d={xf:.3f}"
            chain += f",adelay={at_ms}:all=1[p{idx}_{k}]"
            filters.append(chain)
            labels.append(f"[p{idx}_{k}]")
        idx += 1
    fade = ""
    if v.get("fade_out"):
        st_ = max(total - VB.FADE_OUT_SEC, 0.0)
        fade = f",afade=t=out:st={st_:.3f}:d={VB.FADE_OUT_SEC:.3f}"
    if not labels:
        inputs += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
        filters.append(f"[{idx}:a:0]{AUDIO_FMT}[a]")
    elif len(labels) == 1:
        filters.append(f"{labels[0]}volume={MUSIC_LEVEL:.2f},apad,atrim=0:{total:.3f}{fade}[a]")
    else:
        filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=longest,volume={MUSIC_LEVEL:.2f},"
                                         f"alimiter=limit=0.95,apad,atrim=0:{total:.3f}{fade}[a]")
    out = tmp_dir / "verses_mix.wav"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[a]", "-t", f"{total:.3f}",
         "-c:a", "pcm_s16le", str(out)])
    return out


def overlay_frames(plan: dict, tmp_dir: Path, size: tuple) -> Path:
    """The overlay as a list of see-through pictures with their durations (an ffmpeg concat list): a new picture
    whenever the window or the meter changes (every 0.5 s at most)."""
    v = plan["verses"]
    names = {s: v["actors"][s]["name"] for s in ("A", "B")}
    windows, meter = v["windows"], v.get("meter") or []
    total = float(v["total_seconds"])
    folder = tmp_dir / "verses_overlay"
    folder.mkdir(parents=True, exist_ok=True)
    winner = v.get("winner")
    last_w = windows[-1] if windows else None
    times = sorted({0.0} | {float(w["start"]) for w in windows} | {float(t) for t, _ in meter if t < total})
    lines, made = [], {}
    for k, t in enumerate(times):
        end = times[k + 1] if k + 1 < len(times) else total
        if end - t < 1e-3:
            continue
        w = next((w for w in windows if w["start"] <= t + 1e-6 < w["end"]), last_w)
        share = next((s for tt, s in reversed(meter) if tt <= t + 1e-6), 0.5)
        show_win = w is last_w and winner in ("A", "B", "draw") and w["start"] > 0
        key = (w["index"], round(share, 2), show_win)
        if key not in made:
            f = folder / f"o_{len(made):04d}.png"
            VB.overlay_frame(size, names, w, share, winner, show_win).save(f)
            made[key] = f
        lines.append((made[key], end - t))
    lst = folder / "list.txt"
    body = "".join(f"file '{f.as_posix()}'\nduration {d:.3f}\n" for f, d in lines)
    if lines:
        body += f"file '{lines[-1][0].as_posix()}'\n"           # (the concat demuxer wants the last picture again)
    lst.write_text(body)
    return lst


def card_picture(plan: dict, tmp_dir: Path, size: tuple) -> Path | None:
    """🥊 The tale-of-the-tape card as a PNG, or None when it is off."""
    v = plan["verses"]
    intro = v.get("intro") or {}
    if not intro.get("on"):
        return None
    thumbs = {}
    try:
        from planner.data import get_nearest_thumbnail, load_all_catalogues
        cats = load_all_catalogues()
        for s in ("A", "B"):
            th = intro.get("thumbs", {}).get(s)
            thumbs[s] = get_nearest_thumbnail(cats, th[0], float(th[1])) if th else None
    except Exception:          # noqa: BLE001 — the card still works without thumbnails
        thumbs = {}
    names = {s: v["actors"][s]["name"] for s in ("A", "B")}
    stats = {s: [tuple(x) for x in (intro.get("stats") or {}).get(s) or []] for s in ("A", "B")}
    f = tmp_dir / "verses_card.png"
    VB.tale_card(size, names, stats, thumbs).save(f)
    return f


def finish(plan: dict, combined_video, output_path: str, sources, tmp_dir, run, frame_height: int,
           frame_width: int = 640, progress=None) -> dict:
    """combined_video (the battle's picture, no sound) → output_path: the card in front, the overlay on top, the sound
    mixed. Returns what each step took and the length check: {"edge_sound", "edge_picture", "edge_check"} (the
    keys the render page already reports)."""
    tmp_dir = Path(tmp_dir)
    W, H = int(frame_width), int(frame_height)
    v = plan["verses"]
    total = float(v["total_seconds"])
    took, t0 = {}, time.time()
    if progress:
        progress(0.0, "mixing the battle's sound")
    sound = mix(plan, sources, tmp_dir, run)
    took["edge_sound"] = round(time.time() - t0, 1)
    t0 = time.time()
    card = card_picture(plan, tmp_dir, (W, H))
    card_sec = VB.INTRO_SEC if card else 0.0
    inputs, n_in = ["-i", str(combined_video)], 1
    filt = [f"[0:v]fps={FPS},scale={W}:{H},setsar=1,format=yuv420p[base]"]
    pic = "[base]"
    if (v.get("overlay") or {}).get("on", True):
        lst = overlay_frames(plan, tmp_dir, (W, H))
        inputs += ["-f", "concat", "-safe", "0", "-i", str(lst)]
        filt.append(f"[{n_in}:v]fps={FPS},format=rgba[ov]")
        n_in += 1
        filt.append("[base][ov]overlay=0:0:format=auto:eof_action=repeat,format=yuv420p[main]")
        pic = "[main]"
    if card:
        inputs += ["-loop", "1", "-t", f"{card_sec:.3f}", "-framerate", str(FPS), "-i", str(card)]
        filt.append(f"[{n_in}:v]scale={W}:{H},setsar=1,fps={FPS},format=yuv420p,"
                    f"fade=t=out:st={card_sec - 0.4:.3f}:d=0.4[card]")
        filt.append(f"[card]{pic}concat=n=2:v=1:a=0[v]")
        n_in += 1
    else:
        filt.append(f"{pic}null[v]")
    inputs += ["-i", str(sound)]
    delay = int(round(card_sec * 1000))
    filt.append(f"[{n_in}:a:0]{AUDIO_FMT}" + (f",adelay={delay}:all=1" if delay else "") + f",apad[a]")
    length = total + card_sec
    # one pass, watched: progress about once a second, and stopped (with ffmpeg's own words) if it stands still
    from edge_render import run_watched
    run_watched(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filt), "-map", "[v]", "-map", "[a]",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS),
                 "-c:a", "aac", "-b:a", "192k", "-t", f"{length:.3f}", "-movflags", "+faststart", str(output_path)],
                length, (lambda f: progress(f, "adding the overlay and the card")) if progress else None)
    took["edge_picture"] = round(time.time() - t0, 1)
    try:
        from edge_render import _lengths
        got = _lengths(output_path)
        off = [f"{w} {got[k]:.1f} s" for k, w in (("video", "picture"), ("audio", "sound")) if abs(got[k] - length) > 0.5]
        took["edge_check"] = "ok" if not off else f"expected {length:.1f} s — " + ", ".join(off)
    except Exception:          # noqa: BLE001
        pass
    return took
