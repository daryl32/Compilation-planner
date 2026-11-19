"""⚔️ Verses Battle — the render's last step (called by render_preview.render_plan_dict when a plan has a "verses"
block): the battle's sound (one shared song, or each fighter's theme switching with the picture), the overlay (names,
VS, the round, the tug-of-war meter, the winner) and the tale-of-the-tape card in front — one encode.
The rules are in verses_battle.py."""

from __future__ import annotations

import os
import subprocess
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


def overlay_spans(plan: dict, size: tuple) -> list:
    """The overlay as [(start, end, picture)] — a new (see-through RGBA) picture whenever the window or the meter
    changes (every 0.5 s at most); the same picture is made once and shared."""
    v = plan["verses"]
    names = {s: v["actors"][s]["name"] for s in ("A", "B")}
    windows, meter = v["windows"], v.get("meter") or []
    total = float(v["total_seconds"])
    winner = v.get("winner")
    last_w = windows[-1] if windows else None
    times = sorted({0.0} | {float(w["start"]) for w in windows} | {float(t) for t, _ in meter if t < total})
    out, made = [], {}
    for k, t in enumerate(times):
        end = times[k + 1] if k + 1 < len(times) else total
        if end - t < 1e-3:
            continue
        w = next((w for w in windows if w["start"] <= t + 1e-6 < w["end"]), last_w)
        share = next((s for tt, s in reversed(meter) if tt <= t + 1e-6), 0.5)
        show_win = w is last_w and winner in ("A", "B", "draw") and w["start"] > 0
        key = (w["index"], round(share, 2), show_win)
        if key not in made:
            made[key] = VB.overlay_frame(size, names, w, share, winner, show_win)
        out.append((t, end, made[key]))
    return out


def _layer(img) -> tuple:
    """A see-through picture → (rgb, how solid, x, y) cut to the part that shows (for edge_render._lay)."""
    import numpy as np
    box = img.getbbox()
    if box is None:
        return None
    a = np.asarray(img.crop(box), dtype=np.uint8)
    return a[..., :3].copy(), a[..., 3:4].astype(np.uint16), box[0], box[1]


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


PIECE_SEC = 20.0                  # the picture is made in pieces this long, several at a time


def _piece(combined, a: float, b: float, spans: list, frame: tuple, out: Path, footage: float) -> None:
    """[a, b) of the battle's footage with the overlay laid on: read as frames, layered here, written as frames
    (edge_render's way — one input per ffmpeg, so nothing can stall waiting on another)."""
    import numpy as np
    from edge_render import FPS as EFPS, _lay, _write_video
    W, H = frame
    n_frames = max(int(round((b - a) * EFPS)), 1)
    size = W * H * 3
    held = [np.zeros((H, W, 3), dtype=np.uint8)]
    dec = None
    if a < footage - 0.05:
        dec = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(combined),
                                "-an", "-vf", f"fps={EFPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    mine = [(max(x, a), min(y, b), lay) for x, y, lay in spans if y > a and x < b]

    def make(i):
        buf = dec.stdout.read(size) if dec is not None else b""
        if len(buf) == size:
            held[0] = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).copy()
            pic = held[0]
        else:
            pic = held[0].copy()                           # the footage ran short: its last frame is held
        t = a + (i + 0.5) / EFPS
        on = next((lay for x, y, lay in mine if x <= t < y), None)
        if on is not None:
            pic = pic.copy()
            _lay(pic, on[2], on[3], on[0], on[1])
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


def finish(plan: dict, combined_video, output_path: str, sources, tmp_dir, run, frame_height: int,
           frame_width: int = 640, progress=None, workers: int = None) -> dict:
    """combined_video (the battle's picture, no sound) → output_path: the card in front, the overlay on top, the sound
    mixed. The picture is made in PIECE_SEC pieces (several at once) and joined without a second encode, as Edge
    Mode does since 14 Nov (one big ffmpeg with several inputs could stall at 0 % CPU on the server).
    Returns what each step took and the length check: {"edge_sound", "edge_screens", "edge_picture", "edge_join",
    "edge_check"} (the keys the render page already reports)."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from edge_render import _lengths, _run, _write_video, ffmpeg_version, probe
    tmp_dir = Path(tmp_dir)
    W, H = int(frame_width), int(frame_height)
    v = plan["verses"]
    total = float(v["total_seconds"])
    took, t0 = {"ffmpeg": ffmpeg_version()}, time.time()
    say = progress or (lambda f, what: None)

    # 1 · the sound
    say(0.0, "mixing the battle's sound")
    sound = mix(plan, sources, tmp_dir, lambda cmd: _run(cmd, timeout=600))      # (stopped, not left waiting)
    took["edge_sound"] = round(time.time() - t0, 1)

    # 2 · the tale-of-the-tape card (a still, silent)
    t0 = time.time()
    card = card_picture(plan, tmp_dir, (W, H))
    card_sec = VB.INTRO_SEC if card else 0.0
    card_files = None
    if card:
        from PIL import Image
        still = np.asarray(Image.open(card).convert("RGB").resize((W, H)), dtype=np.uint8)
        n = int(round(card_sec * FPS))
        fade = int(0.4 * FPS)
        card_mp4 = tmp_dir / "verses_card.mp4"
        _write_video(card_mp4, (W, H), n, lambda i: still if i < n - fade else
                     (still.astype(np.float32) * ((n - i) / fade)).astype(np.uint8))
        card_wav = tmp_dir / "verses_card.wav"
        _run(["ffmpeg", "-y", "-f", "lavfi", "-t", f"{card_sec:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
              "-c:a", "pcm_s16le", str(card_wav)])
        card_files = (card_mp4, card_wav)
    took["edge_screens"] = round(time.time() - t0, 1)

    # 3 · the picture with its overlay, a piece at a time
    t0 = time.time()
    spans = []
    if (v.get("overlay") or {}).get("on", True):
        cache = {}
        for a, b, img in overlay_spans(plan, (W, H)):
            if id(img) not in cache:
                cache[id(img)] = _layer(img)
            spans.append((a, b, cache[id(img)]))
    footage = probe(combined_video)["seconds"] or total
    cuts = [0.0]
    while cuts[-1] < total - 1e-3:
        cuts.append(min(cuts[-1] + PIECE_SEC, total))
    pieces = [tmp_dir / f"verses_{k:03d}.mp4" for k in range(len(cuts) - 1)]
    n = max(1, min(int(workers or min(4, os.cpu_count() or 2)), len(pieces)))
    say(0.0, "adding the overlay")
    with ThreadPoolExecutor(max_workers=n) as pool:
        jobs = [pool.submit(_piece, combined_video, a, b, spans, (W, H), out, footage)
                for a, b, out in zip(cuts, cuts[1:], pieces)]
        try:
            for done, job in enumerate(as_completed(jobs), 1):
                job.result()                               # raises here if a piece failed
                say(done / len(jobs), "adding the overlay")
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    took["edge_picture"] = round(time.time() - t0, 1)

    # 4 · joined as they are (no second encode), the sound added
    t0 = time.time()
    say(1.0, "joining")
    order = ([card_files[0]] if card_files else []) + pieces
    lst = tmp_dir / "verses_pieces.txt"
    lst.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in order))
    sounds = ([card_files[1]] if card_files else []) + [sound]
    length = total + card_sec
    inputs = ["-f", "concat", "-safe", "0", "-i", str(lst)]
    for f in sounds:
        inputs += ["-i", str(f)]
    af = "".join(f"[{k + 1}:a:0]{AUDIO_FMT}[s{k}];" for k in range(len(sounds)))
    af += "".join(f"[s{k}]" for k in range(len(sounds))) + f"concat=n={len(sounds)}:v=0:a=1[a]"
    _run(["ffmpeg", "-y", *inputs, "-filter_complex", af, "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
          "-c:a", "aac", "-b:a", "192k", "-t", f"{length:.3f}", "-movflags", "+faststart", str(output_path)], timeout=900)
    took["edge_join"] = round(time.time() - t0, 1)

    got = _lengths(output_path)
    off = [f"{w} {got[k]:.1f} s" for k, w in (("video", "picture"), ("audio", "sound")) if abs(got[k] - length) > 0.5]
    took["edge_check"] = "ok" if not off else f"expected {length:.1f} s — " + ", ".join(off)
    return took
