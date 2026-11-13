"""
⏱️ Edge Mode — a compilation run by a countdown timer. The time is cut into rounds, each round into 20-second
phases (Wait / Slow / Medium / Fast / Extra / Stop); which phases a round holds, and in what order, is set by the
difficulty. An overlay shows the countdown and the phase's name, a voiceover clip can cue each phase, and — with no
song — a generated beat plays at the phase's speed, and that beat is what the footage is matched to.

This file is the logic only (no Streamlit, no ffmpeg): the phase schedule, the generated beat (as a "track" the
🤖 new matcher can read, its blocks, and its sound), the playlist laid end to end, the voiceover cues, and the
overlay's pictures. planner/edge.py is the page; edge_render.py puts it all into the rendered video.
"""

from __future__ import annotations

import math
import random
import wave
from pathlib import Path

import numpy as np

PHASE_SEC = 20.0

# bpm: the generated beat's speed (0 = silent). level: how lively the footage should be there (0–1).
# colour: the label under the timer.
PROFILES = {
    "wait":   {"label": "WAIT",   "seconds": PHASE_SEC, "bpm": 0,   "level": 0.10, "colour": (70, 120, 190)},
    "slow":   {"label": "SLOW",   "seconds": PHASE_SEC, "bpm": 60,  "level": 0.35, "colour": (40, 160, 90)},
    "medium": {"label": "MEDIUM", "seconds": PHASE_SEC, "bpm": 100, "level": 0.55, "colour": (225, 150, 20)},
    "fast":   {"label": "FAST",   "seconds": PHASE_SEC, "bpm": 140, "level": 0.78, "colour": (235, 95, 20)},
    "extra":  {"label": "EXTRA",  "seconds": PHASE_SEC, "bpm": 180, "level": 1.00, "colour": (230, 20, 25)},
    "stop":   {"label": "STOP",   "seconds": PHASE_SEC, "bpm": 0,   "level": 0.05, "colour": (60, 60, 65)},
}
ORDER = ["wait", "slow", "medium", "fast", "extra", "stop"]

DIFFICULTIES = {"chill": "😌 Chill", "medium": "🔥 Medium", "hard": "💀 Hard"}
HARD_RANDOM = 12                                   # Wait + 12 drawn + Extra + Stop = 15 phases = 300 s
HARD_WEIGHTS = {"slow": 1.0, "medium": 2.0, "fast": 2.0, "extra": 2.0}
MIN_ROUNDS, MAX_ROUNDS = 3, 6
MUSIC_UNDER_VOICE = 0.5                            # the songs' volume while the voiceover is on
BEAT_UNDER_SONG = 0.6                              # the generated beat's volume when songs play too
AUDIO_RATE = 44100
TRACK_ID = "edge_beat"                             # the generated beat's name where a track id is wanted
CURVE_RATE = 30                                    # the generated beat's curves, values a second (as the audio analysis)


def round_phases(difficulty: str, rng: random.Random = None) -> list:
    """The phases of one round, in order.
    chill: every phase once, in order (6 × 20 s = 120 s). medium: Wait, then Slow → Medium → Fast twice, then Extra,
    Stop (180 s). hard: Wait, twelve drawn from Slow / Medium / Fast / Extra (Medium, Fast and Extra twice as likely
    as Slow; never the same one three times running), then Extra, Stop (300 s)."""
    if difficulty == "chill":
        return list(ORDER)
    if difficulty == "medium":
        return ["wait"] + ["slow", "medium", "fast"] * 2 + ["extra", "stop"]
    if difficulty == "hard":
        rng = rng or random.Random(0)
        names, weights = list(HARD_WEIGHTS), list(HARD_WEIGHTS.values())
        drawn = []
        while len(drawn) < HARD_RANDOM:
            p = rng.choices(names, weights)[0]
            if len(drawn) >= 2 and drawn[-1] == drawn[-2] == p:
                continue
            drawn.append(p)
        return ["wait"] + drawn + ["extra", "stop"]
    raise ValueError(f"Unknown difficulty: {difficulty}")


def round_seconds(difficulty: str) -> float:
    return sum(PROFILES[p]["seconds"] for p in round_phases(difficulty))


def schedule(difficulty: str, rounds: int, seed: int = 0) -> list:
    """Every phase of the compilation: [{"index", "round" (from 1), "profile", "start", "end"}]."""
    rounds = int(min(max(int(rounds), MIN_ROUNDS), MAX_ROUNDS))
    rng = random.Random(seed)
    out, t = [], 0.0
    for r in range(1, rounds + 1):
        for p in round_phases(difficulty, rng):
            d = float(PROFILES[p]["seconds"])
            out.append({"index": len(out), "round": r, "profile": p, "start": round(t, 3), "end": round(t + d, 3)})
            t += d
    return out


def total_seconds(sched: list) -> float:
    return float(sched[-1]["end"]) if sched else 0.0


def phase_at(sched: list, t: float) -> dict | None:
    for ph in sched:
        if ph["start"] <= t < ph["end"]:
            return ph
    return sched[-1] if sched and t >= sched[-1]["end"] else None


# ---------------------------------------------------------------------------
# The generated beat
# ---------------------------------------------------------------------------

def phase_beats(ph: dict) -> list:
    """One phase's beats: the first on the phase's start, then at its profile's speed. A last beat that would land
    less than half a beat before the next phase starts is left out (it would stumble into the next phase's first)."""
    bpm = PROFILES[ph["profile"]]["bpm"]
    if bpm <= 0:
        return []
    step, out, k = 60.0 / bpm, [], 0
    while ph["start"] + k * step <= ph["end"] - 0.5 * step + 1e-6:
        out.append(round(ph["start"] + k * step, 4))
        k += 1
    return out


def beat_times(sched: list) -> list:
    """The generated beat's hits, phase after phase."""
    return [t for ph in sched for t in phase_beats(ph)]


def beat_track(sched: list) -> dict:
    """The generated beat as a track the 🤖 new matcher reads like any analysed song: a loudness curve that sits at
    each phase's level and pulses on its beats, a drum curve with a hit on every beat (none in Wait / Stop), the
    beats, and a bar line every fourth beat of a phase."""
    total = total_seconds(sched)
    n = int(round(total * CURVE_RATE))
    t = np.arange(n) / CURVE_RATE
    rms, kick, down = np.zeros(n), np.zeros(n), []
    for ph in sched:
        prof = PROFILES[ph["profile"]]
        lo, hi = int(round(ph["start"] * CURVE_RATE)), min(int(round(ph["end"] * CURVE_RATE)), n)
        if prof["bpm"] > 0:
            step = 60.0 / prof["bpm"]
            pulse = np.cos(np.pi * (((t[lo:hi] - ph["start"]) % step) / step)) ** 8      # 1 on the beat, 0 between
            rms[lo:hi] = prof["level"] * (0.65 + 0.35 * pulse)
            kick[lo:hi] = pulse
            down += phase_beats(ph)[::4]
        else:
            rms[lo:hi] = prof["level"]
            down.append(float(ph["start"]))
    bpms = [PROFILES[ph["profile"]]["bpm"] for ph in sched if PROFILES[ph["profile"]]["bpm"] > 0]
    r3 = lambda a: [round(float(x), 3) for x in a]          # noqa: E731
    return {
        "track_id": TRACK_ID, "source_path": "", "duration_sec": round(total, 3),
        "bpm": float(np.median(bpms)) if bpms else 100.0, "beat_times": beat_times(sched),
        "energy_envelope": [{"time": round(i * 0.5, 2), "energy": round(float(rms[int(i * 0.5 * CURVE_RATE)]), 3)}
                            for i in range(int(total / 0.5))],
        "hires": {"rate": CURVE_RATE, "rms": r3(rms), "onset": r3(kick * 0.9)},
        "analysis": {"version": 3, "rate": CURVE_RATE, "beats_per_bar": 4, "kick": r3(kick), "downbeats": sorted(set(down)),
                     "methods": {"beats": "edge mode"}},
        "edge_generated": True,
    }


BLOCK_TARGET_SEC = 4.0             # a clip under the generated beat aims for about this long, on whole bars
STILL_BLOCK_SEC = 10.0             # Wait / Stop: two calm clips a phase


def beat_segments(sched: list) -> list:
    """The generated beat's blocks, shaped like the new matcher's own (start, end, energy, intensity, section_type,
    driver) plus "bpm" (the phase's speed — the matcher judges a bounce against it, not one tempo for the whole
    track) and "edge_profile". A moving phase is cut on its bars into clips of about BLOCK_TARGET_SEC; a still
    phase into two."""
    out = []
    for ph in sched:
        prof = PROFILES[ph["profile"]]
        a, b = float(ph["start"]), float(ph["end"])
        if prof["bpm"] > 0:
            beat = 60.0 / prof["bpm"]
            n_beats = min((4, 8, 16), key=lambda k: abs(k * beat - BLOCK_TARGET_SEC) + 1e-6 * k)   # a tie: the shorter
            size = n_beats * beat
        else:
            size = STILL_BLOCK_SEC
        cuts = [a]
        while cuts[-1] + size < b - 0.5 * size - 1e-6:
            cuts.append(cuts[-1] + size)
        for k, s in enumerate(cuts):
            e = cuts[k + 1] if k + 1 < len(cuts) else b
            out.append({"start": round(s, 3), "end": round(e, 3), "energy": prof["level"], "intensity": prof["level"],
                        "section_type": "rhythmic", "driver": "drums" if prof["bpm"] > 0 else "loudness",
                        "bpm": float(prof["bpm"]) or None, "edge_profile": ph["profile"]})
    return out


def _write_wav(path, samples: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.tobytes())
    return path


def _thump(strong: bool) -> np.ndarray:
    """One beat: a short drum-like thump (a tone that drops in pitch and dies away); bar starts are a little higher
    and louder."""
    n = int(0.16 * AUDIO_RATE)
    t = np.arange(n) / AUDIO_RATE
    f0, f1 = (150.0, 55.0) if strong else (120.0, 50.0)
    freq = f1 + (f0 - f1) * np.exp(-t * 35.0)
    body = np.sin(2 * np.pi * np.cumsum(freq) / AUDIO_RATE) * np.exp(-t * 22.0)
    click = np.sin(2 * np.pi * 1800.0 * t) * np.exp(-t * 300.0) * 0.25
    return (body + click) * (0.9 if strong else 0.7)


def write_beat_wav(sched: list, path) -> Path:
    """The generated beat as sound: a thump on every beat of the moving phases, silence in Wait and Stop."""
    total = total_seconds(sched)
    out = np.zeros(int(round(total * AUDIO_RATE)) + AUDIO_RATE // 2)
    strong, soft = _thump(True), _thump(False)
    for ph in sched:
        for k, t in enumerate(phase_beats(ph)):
            hit = strong if k % 4 == 0 else soft
            i = int(round(t * AUDIO_RATE))
            out[i:i + len(hit)] += hit[:len(out) - i]
    return _write_wav(path, out[:int(round(total * AUDIO_RATE))])


# ---------------------------------------------------------------------------
# The playlist
# ---------------------------------------------------------------------------

def playlist_parts(durations: list, total: float, min_part: float = 0.5) -> list:
    """The songs back to back, the list starting again until `total` seconds are filled.
    durations: [(track_id, seconds)] in playing order → [{"track_id", "offset" (where it starts in the compilation),
    "play" (seconds of it heard), "pass" (which time round the list)}]. The last song is cut where the time ends; a
    leftover shorter than min_part doesn't start another song (the last clip runs on over it)."""
    durations = [(tid, float(d)) for tid, d in durations if float(d) > 0]
    if not durations or total <= 0:
        return []
    out, t, k = [], 0.0, 0
    while t < total - 1e-6:
        tid, d = durations[k % len(durations)]
        play = min(d, total - t)
        if play < min_part and out:
            break
        out.append({"track_id": tid, "offset": round(t, 3), "play": round(play, 3), "pass": k // len(durations) + 1})
        t += play
        k += 1
    return out


# ---------------------------------------------------------------------------
# The voiceover
# ---------------------------------------------------------------------------

VOICE_EXT = (".mp3", ".m4a", ".wav", ".ogg")


def voice_files(folder) -> dict:
    """{profile: [files]} — the voiceover clips for each phase: <folder>/<profile>/*.mp3 (wait, slow, medium, fast,
    extra, stop)."""
    folder = Path(folder)
    out = {}
    for p in ORDER:
        d = folder / p
        out[p] = sorted(f for f in d.iterdir() if f.suffix.lower() in VOICE_EXT) if d.is_dir() else []
    return out


def voice_cues(sched: list, files: dict, seed: int = 0) -> list:
    """One cue at the start of every phase: [{"time", "profile", "file" (None = no clip yet → the placeholder
    tone)}]. A profile's clips take turns in a shuffled order, so none repeats until all have played."""
    rng = random.Random(seed + 7919)
    bags, out = {}, []
    for ph in sched:
        p = ph["profile"]
        have = list((files or {}).get(p) or [])
        pick = None
        if have:
            if not bags.get(p):
                bags[p] = have[:]
                rng.shuffle(bags[p])
            pick = str(bags[p].pop())
        out.append({"time": float(ph["start"]), "profile": p, "file": pick})
    return out


def write_placeholder_cue(profile: str, path) -> Path:
    """A stand-in for a voiceover clip that isn't there yet: short tones, a different pattern for each phase
    (Wait one low, Slow one, Medium two, Fast three, Extra four rising, Stop one long falling)."""
    patterns = {"wait": [(330, 0.35)], "slow": [(440, 0.22)], "medium": [(520, 0.16)] * 2, "fast": [(620, 0.12)] * 3,
                "extra": [(620, 0.1), (740, 0.1), (880, 0.1), (1040, 0.16)], "stop": [(300, 0.6)]}
    parts = []
    for f, d in patterns.get(profile, [(440, 0.2)]):
        t = np.arange(int(d * AUDIO_RATE)) / AUDIO_RATE
        env = np.minimum(1.0, t / 0.01) * np.exp(-t * (3.0 if profile == "stop" else 6.0))
        fr = f * (1.0 - 0.25 * t / d) if profile == "stop" else f
        parts += [0.5 * np.sin(2 * np.pi * fr * t) * env, np.zeros(int(0.06 * AUDIO_RATE))]
    return _write_wav(path, np.concatenate(parts))


# ---------------------------------------------------------------------------
# The overlay
# ---------------------------------------------------------------------------

OVERLAY_FPS = 5                    # how often the ring moves; the number changes once a second
RING = (240, 185, 90)              # the ring's amber
CORNERS = {"top-left": "↖ Top left", "top-right": "↗ Top right", "bottom-left": "↙ Bottom left",
           "bottom-right": "↘ Bottom right", "centre": "◎ Centre"}


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"] if bold
             else ["DejaVuSans-ExtraLight.ttf", "DejaVuSans.ttf", "segoeuil.ttf", "arial.ttf", "Arial.ttf"])
    for name in names:
        for base in ("", "/usr/share/fonts/truetype/dejavu/", "C:/Windows/Fonts/"):
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def overlay_size(frame_height: int) -> tuple:
    """(width, height) of the overlay picture for a video this tall — the ring is about 45 % of the height."""
    ring = int(frame_height * 0.45) // 2 * 2
    return ring + 8, ring + int(ring * 0.30) // 2 * 2 + 8


def overlay_image(profile: str, left: float, size: tuple):
    """The timer as a picture with a see-through background: a thin ring that empties as the phase runs out, the
    seconds left in the middle (00:17), and the phase's name on a coloured label underneath. left: seconds left."""
    from PIL import Image, ImageDraw
    prof = PROFILES[profile]
    w, h = size
    s = 3                                                  # drawn large, shrunk down: smooth edges
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ring = w - 8
    x0, y0, x1, y1 = 4 * s, 4 * s, (4 + ring) * s, (4 + ring) * s
    d.ellipse((x0, y0, x1, y1), fill=(0, 0, 0, 105))                       # darkens the footage behind the numbers
    width = max(2 * s, int(ring * 0.022 * s))
    d.arc((x0, y0, x1, y1), 0, 360, fill=(255, 255, 255, 50), width=width)
    share = float(min(max(left / max(float(prof["seconds"]), 1e-6), 0.0), 1.0))
    if share > 0:
        d.arc((x0, y0, x1, y1), -90, -90 + 360 * share, fill=RING + (255,), width=width)
    secs = int(math.ceil(left - 1e-6))
    text = f"{secs // 60:02d}:{secs % 60:02d}"
    f = _font(int(ring * 0.27 * s))
    box = d.textbbox((0, 0), text, font=f)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d.text((cx - (box[2] + box[0]) / 2, cy - (box[3] + box[1]) / 2), text, font=f, fill=(255, 255, 255, 255))
    lab_h = int(ring * 0.22) * s
    lx0, lx1 = int(w * 0.14) * s, int(w * 0.86) * s
    ly0 = y1 + int(ring * 0.05) * s
    d.rounded_rectangle((lx0, ly0, lx1, ly0 + lab_h), radius=int(lab_h * 0.28), fill=tuple(prof["colour"]) + (255,))
    fb = _font(int(lab_h * 0.62), bold=True)
    box = d.textbbox((0, 0), prof["label"], font=fb)
    d.text(((lx0 + lx1) / 2 - (box[2] + box[0]) / 2, ly0 + lab_h / 2 - (box[3] + box[1]) / 2), prof["label"], font=fb,
           fill=(255, 255, 255, 255))
    return img.resize((w, h), Image.LANCZOS)


def write_overlay(sched: list, folder, frame_height: int) -> tuple:
    """Draws the overlay's pictures (one set per phase name in use — the same 20 seconds serve every phase of that
    name) and writes the list ffmpeg plays them from. Returns (list file, (width, height))."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    size = overlay_size(frame_height)
    step = 1.0 / OVERLAY_FPS
    frames = {}
    for p in sorted({ph["profile"] for ph in sched}):
        n = int(round(PROFILES[p]["seconds"] * OVERLAY_FPS))
        frames[p] = []
        for k in range(n):
            f = folder / f"{p}_{k:04d}.png"
            overlay_image(p, PROFILES[p]["seconds"] - k * step, size).save(f)
            frames[p].append(f)
    lines = ["ffconcat version 1.0"]
    last = None
    for ph in sched:
        for f in frames[ph["profile"]]:
            lines += [f"file '{f.name}'", f"duration {step:.4f}"]
            last = f
    if last is not None:
        lines.append(f"file '{last.name}'")                # the concat reader wants the last picture named twice
    lst = folder / "overlay.txt"
    lst.write_text("\n".join(lines) + "\n")
    return lst, size


def overlay_xy(corner: str, margin_share: float = 0.035) -> tuple:
    """ffmpeg overlay positions (x, y expressions) for a corner of the frame."""
    m = f"(H*{margin_share})"
    return {"top-left": (m, m), "top-right": (f"W-w-{m}", m), "bottom-left": (m, f"H-h-{m}"),
            "bottom-right": (f"W-w-{m}", f"H-h-{m}"), "centre": ("(W-w)/2", "(H-h)/2")}.get(corner, (m, m))


def describe(difficulty: str, rounds: int) -> str:
    r = round_seconds(difficulty)
    tot = r * rounds
    return (f"{rounds} rounds × {int(r // 60)}:{int(r % 60):02d} = {int(tot // 60)}:{int(tot % 60):02d} · "
            + " → ".join(PROFILES[p]["label"].title() for p in (round_phases(difficulty) if difficulty != "hard" else
                                                                 ["wait"])) + (" → 12 at random → Extra → Stop" if difficulty == "hard" else ""))


# ---------------------------------------------------------------------------
# The avatar and the intro screen
# ---------------------------------------------------------------------------

VIDEO_EXT = (".mp4", ".mov", ".webm", ".m4v")
AVATAR_MODES = {"corner": "🪟 In a corner", "split": "◧ Split screen"}
AVATAR_SLOTS = ["intro"] + ORDER                   # <folder>/intro/ and one folder per phase
AVATAR_CORNER_SHARE = 0.42                         # her box's height, as a share of the frame's
PLACEHOLDER_CUE_SEC = 3.0                          # how long the stand-in figure shows at a cue
PLACEHOLDER_INTRO_SEC = 8.0                        # the intro's length until her intro clip exists


def clip_files(folder, slots=None, exts=VIDEO_EXT) -> dict:
    """{slot: [files]} — <folder>/<slot>/* with one of the extensions (the avatar's clips: intro + one per phase)."""
    folder = Path(folder)
    out = {}
    for p in (slots or AVATAR_SLOTS):
        d = folder / p
        out[p] = sorted(f for f in d.iterdir() if f.suffix.lower() in exts) if d.is_dir() else []
    return out


def avatar_cues(sched: list, files: dict, seed: int = 0) -> list:
    """One appearance at the start of every phase: [{"time", "profile", "clip" (None = no clip yet → the stand-in
    figure)}] — a phase's clips take turns, like the voiceover's. (Daryl's own cue list replaces this rule.)"""
    return [{"time": c["time"], "profile": c["profile"], "clip": c["file"]}
            for c in voice_cues(sched, files, seed + 101)]


def avatar_placeholder(size: tuple, label: str = "AVATAR"):
    """A stand-in for the avatar until her clips exist: a plain figure on a dark card, with a label."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", (w, h), (38, 34, 48))
    d = ImageDraw.Draw(img)
    r = int(min(w, h) * 0.17)
    cx, cy = w // 2, int(h * 0.36)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(150, 140, 170))
    d.pieslice((cx - int(r * 2.1), cy + int(r * 1.25), cx + int(r * 2.1), cy + int(r * 5.4)), 180, 360, fill=(150, 140, 170))
    f = _font(max(10, int(min(w, h) * 0.09)), bold=True)
    box = d.textbbox((0, 0), label, font=f)
    d.text((cx - (box[2] + box[0]) / 2, h - (box[3] - box[1]) - int(h * 0.07)), label, font=f, fill=(235, 230, 245))
    d.rectangle((0, 0, w - 1, h - 1), outline=(235, 230, 245), width=max(1, w // 120))
    return img


def intro_layout(size: tuple) -> dict:
    """Where things sit on the intro screen: the avatar's box on the left, the settings and thumbnails on the right."""
    w, h = size
    m = int(h * 0.07)
    box_w = int(w * 0.40) // 2 * 2
    return {"avatar": (m, m, box_w, (h - 2 * m) // 2 * 2), "text_x": m + box_w + int(w * 0.05), "margin": m}


def intro_card(size: tuple, stats: list, thumbs: list = (), title: str = "EDGE MODE", placeholder: bool = False):
    """The intro screen without the avatar's video: a dark background, the title, the chosen settings
    (stats: [(label, value)] — difficulty, rounds, total time) and a grid of thumbnails from the videos in the
    compilation (thumbs: PIL images). placeholder: draw the stand-in figure where she will be."""
    from PIL import Image, ImageDraw
    w, h = size
    lay = intro_layout(size)
    img = Image.new("RGB", (w, h), (16, 14, 22))
    d = ImageDraw.Draw(img)
    ax, ay, aw, ah = lay["avatar"]
    if placeholder:
        img.paste(avatar_placeholder((aw, ah)), (ax, ay))
    else:
        d.rectangle((ax, ay, ax + aw - 1, ay + ah - 1), fill=(0, 0, 0))
    x, y = lay["text_x"], lay["margin"]
    ft = _font(int(h * 0.105), bold=True)
    d.text((x, y), title, font=ft, fill=(255, 255, 255))
    y += int(h * 0.135)
    d.rectangle((x, y, x + int(w * 0.12), y + max(2, h // 120)), fill=RING)
    y += int(h * 0.045)
    fl, fv = _font(int(h * 0.045)), _font(int(h * 0.075), bold=True)
    col = (w - x - lay["margin"]) // max(len(stats), 1)
    for k, (label, value) in enumerate(stats):
        d.text((x + k * col, y), str(label).upper(), font=fl, fill=(170, 165, 185))
        d.text((x + k * col, y + int(h * 0.06)), str(value), font=fv, fill=RING)
    y += int(h * 0.19)
    thumbs = [t for t in thumbs if t is not None][:6]
    if thumbs:
        gap = max(2, int(w * 0.008))
        tw = (w - x - lay["margin"] - 2 * gap) // 3
        th = min(int(tw * 9 / 16), (h - lay["margin"] - y - gap) // 2)
        for k, t in enumerate(thumbs):
            t = t.convert("RGB")
            s = max(tw / t.width, th / t.height)
            t = t.resize((max(tw, int(t.width * s)), max(th, int(t.height * s))))
            lx, ty = (t.width - tw) // 2, (t.height - th) // 2
            img.paste(t.crop((lx, ty, lx + tw, ty + th)), (x + (k % 3) * (tw + gap), y + (k // 3) * (th + gap)))
    return img
