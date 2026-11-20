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
# ring: the timer ring's colour in that phase (🔔 phase signals on; amber otherwise). chime: the pitch (Hz) of the
# short sound at the phase's start — higher the livelier the phase.
PROFILES = {
    "wait":   {"label": "WAIT",   "seconds": PHASE_SEC, "bpm": 0,   "level": 0.10, "colour": (70, 120, 190),
               "ring": (110, 165, 240), "chime": 392.0},
    "slow":   {"label": "SLOW",   "seconds": PHASE_SEC, "bpm": 60,  "level": 0.35, "colour": (40, 160, 90),
               "ring": (80, 210, 125), "chime": 523.3},
    "medium": {"label": "MEDIUM", "seconds": PHASE_SEC, "bpm": 100, "level": 0.55, "colour": (225, 150, 20),
               "ring": (250, 185, 50), "chime": 659.3},
    "fast":   {"label": "FAST",   "seconds": PHASE_SEC, "bpm": 140, "level": 0.78, "colour": (235, 95, 20),
               "ring": (255, 125, 45), "chime": 784.0},
    "extra":  {"label": "EXTRA",  "seconds": PHASE_SEC, "bpm": 200, "level": 1.00, "colour": (230, 20, 25),
               "ring": (245, 45, 50), "chime": 1046.5},
    "stop":   {"label": "STOP",   "seconds": PHASE_SEC, "bpm": 0,   "level": 0.05, "colour": (60, 60, 65),
               "ring": (175, 175, 185), "chime": 261.6},
}
ORDER = ["wait", "slow", "medium", "fast", "extra", "stop"]

DIFFICULTIES = {"chill": "😌 Chill", "medium": "🔥 Medium", "hard": "💀 Hard"}
# 🏃 Each difficulty's speed for each moving phase (bpm), and how much faster each round after the first gets
# (Daryl, 16 Nov: "pacing seems fine for each"). Wait and Stop have none.
PACE = {"chill": {"slow": 50, "medium": 80, "fast": 110, "extra": 140},
        "medium": {"slow": 60, "medium": 100, "fast": 140, "extra": 180},
        "hard": {"slow": 70, "medium": 110, "fast": 160, "extra": 210}}
PACE_BUILD = {"chill": 0.0, "medium": 0.03, "hard": 0.05}
PACE_CAP = 1.25
HARD_RANDOM = 12                                   # Wait + 12 drawn + Extra + Stop = 15 phases = 300 s
HARD_WEIGHTS = {"slow": 1.0, "medium": 2.0, "fast": 2.0, "extra": 2.0}
MIN_ROUNDS, MAX_ROUNDS = 3, 6
MUSIC_UNDER_VOICE = 0.25                           # the songs' volume while the voiceover or the avatar is on (Daryl, 14 Nov: background)
MUSIC_IN_STILL = 0.25                              # the songs' volume during Wait and Stop
STILL = ("wait", "stop")
MUSIC_FADE_SEC = 0.5                               # how long the songs take to drop to / come back from that
# 🌙 Stop: the songs drop to STOP_DROP within the first STOP_DROP_SEC (noticeably lower as soon as Stop starts), then
# fade smoothly to silence over the rest of it (Daryl, 16 Nov); they come back over the whole of the Wait that follows
STOP_DROP, STOP_DROP_SEC, STOP_CURVE = 0.5, 1.0, 1.7
DUCK_LEVEL = 0.25                                  # 🔉 the songs while a line plays (full volume between lines)
DUCK_FADE_SEC = 0.3                                # how long they take to dip / come back around a line
BEAT_UNDER_SONG = 0.6                              # the generated beat's volume when songs play too
CHIME_VOLUME = 0.35                                # 🔔 the phase-change sound
VOICE_LEVEL = 1.15                                 # 🎙️ the voiceover / her lines, 15 % louder (Daryl, 17 Nov)
MUSIC_LEVEL = 0.95                                 # 🎵 the songs, 5 % quieter (Daryl, 17 Nov) — on top of the fades / quarters
FLASH_SEC = 0.16                                   # 🔔 the picture's flash as a phase starts
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


def pace(profile: str, difficulty: str, rnd: int = 1) -> float:
    """🏃 A phase's speed (bpm) in this difficulty and round: the PACE table, PACE_BUILD faster each round after the
    first (at most PACE_CAP ×). Wait / Stop: 0."""
    base = PACE.get(difficulty, {}).get(profile)
    if base is None:
        return float(PROFILES[profile]["bpm"])
    return round(base * min(1.0 + PACE_BUILD.get(difficulty, 0.0) * (max(int(rnd), 1) - 1), PACE_CAP), 1)


def phase_bpm(ph: dict) -> float:
    """The phase's speed: its own (plans from 16 Nov), else its profile's (older plans)."""
    return float(ph["bpm"]) if ph.get("bpm") is not None else float(PROFILES[ph["profile"]]["bpm"])


def round_seconds(difficulty: str) -> float:
    return sum(PROFILES[p]["seconds"] for p in round_phases(difficulty))


# 💀 Hard's round (Daryl, 16 Nov): Wait 20 s, a random middle, then Extra → Stop — 5:00 as before. The middle's
# phases are drawn with weights that rise through the round, the faster ones last longer, Fast / Extra drop back to
# Slow now and then, and the same phase never comes twice running.
HARD_ROUND_SEC = 300
HARD_WAIT_SEC, HARD_STOP_SEC, HARD_FINAL_EXTRA = 20, 20, (20, 30)
HARD_LEN = {"slow": (10, 18), "medium": (12, 22), "fast": (15, 28), "extra": (18, 32)}
HARD_EARLY = {"slow": 2.0, "medium": 2.0, "fast": 1.0, "extra": 0.5}       # the middle's first half
HARD_LATE = {"slow": 0.7, "medium": 1.0, "fast": 2.0, "extra": 2.0}        # its second half
HARD_DROP_CHANCE, HARD_MAX_DROPS = 0.3, 2
FAST_ONES = ("fast", "extra")


def hard_round(rng: random.Random) -> list:
    """💀 One Hard round → [(profile, seconds)]: Wait, the random middle, Extra, Stop — always HARD_ROUND_SEC long.
    Rules: weights HARD_EARLY then HARD_LATE (half way through the middle); after Fast / Extra a HARD_DROP_CHANCE to drop
    back to Slow (at most HARD_MAX_DROPS); never the same phase twice running; never three of Fast / Extra in a row;
    at least one Extra in the middle; the middle doesn't end on Extra (the round's own Extra follows)."""
    final = rng.randint(*HARD_FINAL_EXTRA)
    room = HARD_ROUND_SEC - HARD_WAIT_SEC - final - HARD_STOP_SEC
    mid, left, drops = [], room, 0
    while left > 0:
        prev = mid[-1][0] if mid else "wait"
        two_fast = len(mid) >= 2 and mid[-1][0] in FAST_ONES and mid[-2][0] in FAST_ONES
        if prev in FAST_ONES and drops < HARD_MAX_DROPS and rng.random() < HARD_DROP_CHANCE:
            p = "slow"
            drops += 1
        else:
            w = HARD_EARLY if (room - left) < room / 2 else HARD_LATE
            opts = [q for q in w if q != prev and not (two_fast and q in FAST_ONES)]
            if left <= HARD_LEN["extra"][1]:                   # the last of the middle: not Extra
                opts = [q for q in opts if q != "extra"] or ["medium"]
            p = rng.choices(opts, [w[q] for q in opts])[0]
        lo, hi = HARD_LEN[p]
        if left <= hi or left - lo < 10:                       # the rest fits this phase (a little over is fine)
            if left < lo and mid:                              # too short for a phase of its own: the last one stretches
                mid[-1] = (mid[-1][0], mid[-1][1] + left)
                break
            n = left
        else:
            n = rng.randint(lo, min(hi, left - 10))
        mid.append((p, n))
        left -= n
    if mid and mid[-1][0] == "extra":
        mid[-1] = ("fast" if len(mid) < 2 or mid[-2][0] != "fast" else "medium", mid[-1][1])
    if not any(p == "extra" for p, _ in mid):                  # at least one Extra in the middle
        ok = [k for k, (p, _) in enumerate(mid[:-1]) if p == "fast"
              and (k == 0 or mid[k - 1][0] != "extra") and mid[k + 1][0] != "extra"]
        if ok:
            k = max(ok, key=lambda k: mid[k][1])
            mid[k] = ("extra", mid[k][1])
    merged = []
    for p, n in mid:                                           # (an edit above can't leave two the same running: merge)
        if merged and merged[-1][0] == p:
            merged[-1] = (p, merged[-1][1] + n)
        else:
            merged.append((p, n))
    return [("wait", HARD_WAIT_SEC)] + merged + [("extra", final), ("stop", HARD_STOP_SEC)]


# 🎲 Surprises (Chill / Medium, one switch — Daryl, 16 Nov). Decided at compile from the seed and written into the
# schedule, so the music, footage, voice and timer all know. "Next up" never warns.
# ✋ Hands off: inside a phase, the phase's clock freezes for a random few seconds (the round gets longer), then the
#   phase carries on where it was. ⏩ Override: a phase runs past its zero by +N s — before any phase but Wait / Slow.
# 🎭 False finish: the round's last Extra counts down to Stop… then "NOT YET" and +10 s. Once a compilation at most.
HANDS_OFF_SEC = {"chill": (3, 5), "medium": (5, 8, 12)}
OVERRIDE_SEC = {"chill": (5,), "medium": (5, 10, 15)}
FALSE_FINISH_SEC = 10
FALSE_FINISH_CHANCE = 0.6
HANDS_OFF_IN = {"chill": ("fast", "extra"), "medium": ("medium", "fast", "extra")}
HANDS_OFF_EARLIEST, HANDS_OFF_LATEST = 6.0, 4.0                # seconds into the phase / before its end
NO_OVERRIDE_BEFORE = ("wait", "slow")
SURPRISE_STYLES = {                                            # the timer's look during one (PROFILES-like)
    "handsoff": {"label": "HANDS OFF", "colour": (95, 95, 100), "ring": (175, 175, 185), "chime": 261.6},
    "override": {"label": "OVERRIDE", "colour": (150, 20, 25), "ring": (245, 45, 50), "chime": 1046.5},
    "notyet": {"label": "NOT YET", "colour": (150, 20, 25), "ring": (245, 45, 50), "chime": 1046.5},
}


def plan_surprises(rounds: list, difficulty: str, rng: random.Random) -> list:
    """🎲 Where the surprises go → [{"round", "unit" (the phase's place in its round), "kind": handsoff | override |
    falsefinish, "seconds", "at" (hands off: seconds into the phase)}]. rounds: [[(profile, seconds)], …].
    Rules: none in round 1; Chill 0–1 a round (60 %), Medium 1–2; at least one in the compilation; never two in one
    phase or in neighbouring phases; Medium never the same kind twice running; hands off only in HANDS_OFF_IN phases
    long enough to have a middle; override not before Wait / Slow; false finish at most once, on the last or
    second-last round's final Extra."""
    if difficulty not in HANDS_OFF_SEC or len(rounds) < 2:
        return []
    out, used, last_kind = [], set(), None
    budget = {r: (1 if rng.random() < 0.6 else 0) if difficulty == "chill" else 1 + (rng.random() < 0.5)
              for r in range(2, len(rounds) + 1)}
    if not any(budget.values()):
        budget[len(rounds)] = 1
    ff_round = None
    if rng.random() < FALSE_FINISH_CHANCE:
        ff_round = rng.choice([r for r in (len(rounds) - 1, len(rounds)) if r >= 2])
    for r in sorted(budget):
        units = rounds[r - 1]
        take = budget[r]

        def free(k, _r=r):
            return not any((_r, j) in used for j in (k - 1, k, k + 1))
        if r == ff_round and take:
            k = len(units) - 2                                 # the round's final Extra (before Stop)
            if units[k][0] == "extra" and free(k):
                out.append({"round": r, "unit": k, "kind": "falsefinish", "seconds": FALSE_FINISH_SEC})
                used.add((r, k))
                take -= 1
                last_kind = "override"
        for _ in range(take):
            kinds = ["handsoff", "override"]
            rng.shuffle(kinds)
            if difficulty == "medium" and last_kind in kinds:
                kinds.remove(last_kind)
                kinds.append(last_kind)
            for kind in kinds:
                if kind == "handsoff":
                    hosts = [k for k, (p, n) in enumerate(units) if p in HANDS_OFF_IN[difficulty]
                             and n >= HANDS_OFF_EARLIEST + HANDS_OFF_LATEST + 2 and free(k)]
                else:
                    hosts = [k for k, (p, n) in enumerate(units[:-1]) if p not in ("wait", "stop")
                             and units[k + 1][0] not in NO_OVERRIDE_BEFORE and free(k)]
                if not hosts:
                    continue
                k = rng.choice(hosts)
                if kind == "handsoff":
                    n = units[k][1]
                    span = n - HANDS_OFF_EARLIEST - HANDS_OFF_LATEST
                    at = round((HANDS_OFF_EARLIEST + span * math.sqrt(rng.random())) * 2) / 2   # later is likelier
                    out.append({"round": r, "unit": k, "kind": kind, "seconds": rng.choice(HANDS_OFF_SEC[difficulty]),
                                "at": at})
                else:
                    out.append({"round": r, "unit": k, "kind": kind, "seconds": rng.choice(OVERRIDE_SEC[difficulty])})
                used.add((r, k))
                last_kind = kind
                break
    return sorted(out, key=lambda s: (s["round"], s["unit"]))


def round_units(difficulty: str, rounds: int, rng: random.Random) -> list:
    """Each round's phases with their lengths → [[(profile, seconds)], …]."""
    out = []
    for _r in range(rounds):
        if difficulty == "hard":
            out.append(hard_round(rng))
        else:
            out.append([(p, float(PROFILES[p]["seconds"])) for p in round_phases(difficulty, rng)])
    return out


def schedule(difficulty: str, rounds: int, seed: int = 0, surprises: bool = False) -> list:
    """Every stretch of the compilation: [{"index", "round", "profile", "start", "end", "bpm", "unit" (which phase of
    the compilation it belongs to — a phase split by a surprise keeps one), "kind" (phase | handsoff | override),
    "clock": {"total", "left"} (what the timer counts: from `left` at the start, out of `total`), "label" (override:
    "OVERRIDE" / "NOT YET")}]. surprises: 🎲 for Chill / Medium (Hard has its own random structure instead)."""
    rounds = int(min(max(int(rounds), MIN_ROUNDS), MAX_ROUNDS))
    rng = random.Random(seed)
    units = round_units(difficulty, rounds, rng)
    sur = plan_surprises(units, difficulty, random.Random(seed * 7919 + 17)) if surprises else []
    by = {(s["round"], s["unit"]): s for s in sur}
    out, t, u = [], 0.0, 0

    def add(r, p, n, kind, total, left, **extra):
        nonlocal t
        bpm = 0.0 if kind == "handsoff" else pace(p, difficulty, r)
        out.append({"index": len(out), "round": r, "profile": p, "start": round(t, 3), "end": round(t + n, 3), "bpm": bpm,
                    "unit": u, "kind": kind, "clock": {"total": float(total), "left": float(left)}, **extra})
        t += n
    for r, phases in enumerate(units, 1):
        for k, (p, n) in enumerate(phases):
            s_ = by.get((r, k))
            if s_ and s_["kind"] == "handsoff":
                add(r, p, s_["at"], "phase", n, n)
                add(r, p, s_["seconds"], "handsoff", s_["seconds"], s_["seconds"], surprise="handsoff")
                add(r, p, n - s_["at"], "phase", n, n - s_["at"])
            else:
                add(r, p, n, "phase", n, n)
                if s_:
                    ff = s_["kind"] == "falsefinish"
                    add(r, p, s_["seconds"], "override", s_["seconds"], s_["seconds"], surprise=s_["kind"],
                        label="NOT YET" if ff else "OVERRIDE")
            u += 1
    return out


def surprise_list(sched: list) -> list:
    """🎲 The schedule's surprises → [{"kind", "round", "profile", "start", "seconds"}]."""
    return [{"kind": ph["surprise"], "round": ph["round"], "profile": ph["profile"], "start": ph["start"],
             "seconds": round(ph["end"] - ph["start"], 3)} for ph in sched if ph.get("surprise")]


def timer_style(ph: dict) -> str:
    """The timer's look for this stretch: its phase's, or ✋ / ⏩ / 🎭's."""
    if ph.get("kind") == "handsoff":
        return "handsoff"
    if ph.get("kind") == "override":
        return "notyet" if ph.get("label") == "NOT YET" else "override"
    return ph["profile"]


def style(name: str) -> dict:
    return PROFILES.get(name) or SURPRISE_STYLES[name]


def clock(ph: dict) -> tuple:
    """(total, left at the stretch's start) the timer counts — older plans: the stretch itself."""
    c = ph.get("clock") or {}
    n = float(ph["end"]) - float(ph["start"])
    return float(c.get("total", n)), float(c.get("left", n))


def timer_key(ph: dict) -> str:
    """Which set of timer pictures serves this stretch: its look and how long the clock is ("fast-20", "handsoff-5")."""
    return f"{timer_style(ph)}-{clock(ph)[0]:g}"


def next_real(sched: list, k: int) -> str | None:
    """⏭️ The phase "Next up" promises after stretch k: the next phase of the compilation (a surprise is never
    announced). None after the last."""
    u = sched[k].get("unit", sched[k]["index"])
    return next((ph["profile"] for ph in sched[k + 1:] if ph.get("unit", ph["index"]) != u), None)


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
    bpm = phase_bpm(ph)
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
        if phase_bpm(ph) > 0:
            step = 60.0 / phase_bpm(ph)
            pulse = np.cos(np.pi * (((t[lo:hi] - ph["start"]) % step) / step)) ** 8      # 1 on the beat, 0 between
            rms[lo:hi] = prof["level"] * (0.65 + 0.35 * pulse)
            kick[lo:hi] = pulse
            down += phase_beats(ph)[::4]
        else:
            rms[lo:hi] = prof["level"]
            down.append(float(ph["start"]))
    bpms = [phase_bpm(ph) for ph in sched if phase_bpm(ph) > 0]
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


# ---------------------------------------------------------------------------
# 🎯 The pace guide: the beats the stroke bar moves on
# ---------------------------------------------------------------------------

GUIDE_STEPS = (0.25, 0.5, 1.0, 2.0)          # a song's beat thinned (every 4th / 2nd) or doubled to near the phase's pace
GUIDE_OFF = 0.15                             # further than this from the phase's pace: the result page says so


def _song_beats_at(parts: list, beats: dict, a: float, b: float) -> list:
    """The songs' beats heard between a and b (compilation time). parts: [{"track_id", "offset" (where it starts in
    the compilation), "play", "bar" / "start" (where in the song)}]; beats: {track_id: [song-time beats]}."""
    out = []
    for p in parts:
        o0, o1 = float(p["offset"]), float(p["offset"]) + float(p["play"])
        if o1 <= a or o0 >= b:
            continue
        s0 = float(p.get("bar", p.get("start", 0.0)) or 0.0)
        for t in beats.get(p["track_id"]) or ():
            c = o0 + float(t) - s0
            if max(a, o0) - 1e-6 <= c < min(b, o1) - 1e-6:
                out.append(round(c, 4))
    return sorted(set(out))


def pace_guide(sched: list, parts: list = (), beats: dict = None) -> dict:
    """🎯 The stroke bar's beats for the whole compilation → {"beats": [...], "per_phase": [{"index", "bpm_target",
    "bpm_used", "source"}]}. With no songs: the generated beat's own hits (exact). With songs: the beats of the song
    playing, thinned to every 2nd / 4th or doubled, whichever comes nearest the phase's pace — the bar stays on the
    music. Too few song beats in a phase (an unanalysed song): the phase's own pace. Wait / Stop: none."""
    beats = beats or {}
    out, info = [], []
    for ph in sched:
        target = phase_bpm(ph)
        if target <= 0:
            continue
        a, b = float(ph["start"]), float(ph["end"])
        got = _song_beats_at(list(parts or ()), beats, a, b) if parts else []
        if len(got) < 4:
            own = phase_beats(ph)
            out += own
            info.append({"index": ph["index"], "bpm_target": target, "bpm_used": target,
                         "source": "song (no beats: the phase's pace)" if parts else "beat"})
            continue
        step = float(np.median(np.diff(got)))
        song_bpm = 60.0 / max(step, 1e-6)
        f = min(GUIDE_STEPS, key=lambda k: abs(math.log2(song_bpm * k / target)))
        if f >= 1.0:
            g = list(got)
            if f == 2.0:
                g += [round((x + y) / 2, 4) for x, y in zip(got, got[1:])]
        else:
            g = got[::int(round(1 / f))]
        out += g
        info.append({"index": ph["index"], "bpm_target": target, "bpm_used": round(song_bpm * f, 1), "source": "song"})
    return {"beats": sorted(set(round(t, 4) for t in out)), "per_phase": info}


def guide_off_target(per_phase: list) -> list:
    """The phases whose guide is further than GUIDE_OFF from their pace (the song can't be thinned / doubled near it)."""
    return [p for p in per_phase if p["bpm_target"] and abs(p["bpm_used"] / p["bpm_target"] - 1.0) > GUIDE_OFF]


def guide_position(beats, t: float) -> float | None:
    """Where the stroke bar's marker is at time t: 0 = top, 1 = bottom — at the top on one beat, the bottom on the
    next, easing between (it lingers at the ends). None before the first beat or after the last."""
    import bisect
    k = bisect.bisect_right(beats, t) - 1
    if k < 0 or k + 1 >= len(beats):
        return None
    frac = (t - beats[k]) / max(beats[k + 1] - beats[k], 1e-6)
    return 0.5 - 0.5 * math.cos(math.pi * (k + frac))


def guide_bar(height: int):
    """The stroke bar's track: a dark see-through rounded bar. Returns (track picture, its width)."""
    from PIL import Image, ImageDraw
    w = max(int(height * 0.09) // 2 * 2, 6)
    s = 3
    img = Image.new("RGBA", (w * s, height * s), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w * s - 1, height * s - 1), radius=w * s // 2, fill=(0, 0, 0, 150))
    return img.resize((w, height), Image.LANCZOS), w


def guide_marker(width: int, colour, dim: bool = False):
    """The stroke bar's marker: a pill a little wider than the bar, in the phase's colour (grey and faint: hands off)."""
    from PIL import Image, ImageDraw
    w, h = int(width * 1.8) // 2 * 2, max(int(width * 1.1) // 2 * 2, 4)
    s = 3
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    fill = (150, 150, 155, 120) if dim else tuple(colour) + (255,)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius=h * s // 2, fill=fill,
                                          outline=(255, 255, 255, 200 if not dim else 60), width=s)
    return img.resize((w, h), Image.LANCZOS)


def next_chip(profile: str | None, width: int):
    """⏭️ "NEXT ▸ FAST" on a small dark label as wide as the timer, the phase's name on its own colour. profile None:
    the last phase ("NEXT ▸ END")."""
    from PIL import Image, ImageDraw
    s = 3
    h = max(int(width * 0.17) // 2 * 2, 12)
    img = Image.new("RGBA", (width * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(width * 0.06) * s
    d.rounded_rectangle((pad, 0, width * s - pad, h * s - 1), radius=int(h * s * 0.45), fill=(0, 0, 0, 140))
    f = _font(int(h * s * 0.52), bold=True)
    head, name = "NEXT ▸ ", (PROFILES[profile]["label"] if profile else "END")
    hb, nb = d.textbbox((0, 0), head, font=f), d.textbbox((0, 0), name, font=f)
    nw = nb[2] - nb[0] + int(h * s * 0.5)
    total = (hb[2] - hb[0]) + nw
    x = width * s / 2 - total / 2
    y = h * s / 2
    d.text((x - hb[0], y - (hb[3] + hb[1]) / 2), head, font=f, fill=(255, 255, 255, 220))
    nx = x + (hb[2] - hb[0])
    colour = tuple(PROFILES[profile]["colour"]) if profile else (90, 90, 95)
    d.rounded_rectangle((nx, int(h * s * 0.14), nx + nw, int(h * s * 0.86)), radius=int(h * s * 0.3), fill=colour + (255,))
    d.text((nx + nw / 2 - (nb[2] + nb[0]) / 2, y - (nb[3] + nb[1]) / 2), name, font=f, fill=(255, 255, 255, 255))
    return img.resize((width, h), Image.LANCZOS)


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
        if phase_bpm(ph) > 0:
            beat = 60.0 / phase_bpm(ph)
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
                        "section_type": "rhythmic", "driver": "drums" if phase_bpm(ph) > 0 else "loudness",
                        "bpm": phase_bpm(ph) or None, "edge_profile": ph["profile"]})
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


def round_playlist_parts(durations: list, sched: list, min_part: float = 0.5) -> list:
    """🎵 A new song each round: round r plays song r of the list (the list starting again when there are more
    rounds than songs), from its beginning — started again when it ends before the round does. The change falls
    between a round's Stop and the next round's Wait, where the 🌙 fade has the music silent.
    Same shape as playlist_parts, plus "round"."""
    durations = [(tid, float(d)) for tid, d in durations if float(d) > 0]
    if not durations or not sched:
        return []
    out, rounds = [], {}
    for ph in sched:
        a, b = rounds.get(ph["round"], (ph["start"], ph["end"]))
        rounds[ph["round"]] = (min(a, ph["start"]), max(b, ph["end"]))
    for r in sorted(rounds):
        a, b = rounds[r]
        tid, d = durations[(r - 1) % len(durations)]
        for p in playlist_parts([(tid, d)], b - a, min_part):
            out.append({**p, "offset": round(a + p["offset"], 3), "pass": (r - 1) // len(durations) + 1, "round": r,
                        "loop": p["pass"]})
    return out


def shift_schedule(phases: list) -> list:
    """Some phases of a schedule moved to start at 0 (a round on its own, for matching the beat round by round)."""
    if not phases:
        return []
    t0 = float(phases[0]["start"])
    return [{**ph, "index": k, "start": round(ph["start"] - t0, 3), "end": round(ph["end"] - t0, 3)}
            for k, ph in enumerate(phases)]


# ---------------------------------------------------------------------------
# 🌙 / 🔉 How loud the songs are, moment by moment
# ---------------------------------------------------------------------------

def music_spec(sched: list, *, talking: bool, lines: list = (), fade_stop: bool = True, duck: bool = True,
               still_volume: float = MUSIC_IN_STILL, under_voice: float = MUSIC_UNDER_VOICE) -> dict:
    """The rule for the songs' volume, as numbers (the render turns it into an ffmpeg expression; music_gain reads
    it for one moment).
    - level: the songs' volume where nothing else lowers them — full, or (talking without 🔉 dips) a quarter.
    - 🌙 fade_stop: as Stop starts the songs drop to half within a second, then fade smoothly to silence by its end;
      over every Wait they
      come back up, full as the next phase starts (the first Wait too: the compilation fades in). Off: they drop
      to `still_volume` in Wait and Stop, as before.
    - 🔉 duck (with the voiceover / avatar on): full volume, dipping to DUCK_LEVEL only while a line plays
      (lines: [(start, end)]). Off: a quarter all the way through, as before."""
    dips = [(float(a), float(b)) for a, b in lines] if (talking and duck) else []
    dips += [(float(ph["start"]), float(ph["end"])) for ph in sched if ph.get("kind") == "handsoff"]   # ✋ a quarter
    level = 1.0 if (not talking or duck) else float(under_voice)
    spec = {"level": level, "fades": [], "still": [], "dips": dips, "dip_level": DUCK_LEVEL, "dip_fade": DUCK_FADE_SEC,
            "still_volume": float(still_volume), "still_fade": MUSIC_FADE_SEC}
    if fade_stop:
        spec["fades"] = [(ph["profile"], float(ph["start"]), float(ph["end"])) for ph in sched if ph["profile"] in STILL]
    elif still_volume < level:
        spec["still"] = still_spans(sched)
    return spec


def _fade_share(kind: str, a: float, b: float, t: float) -> float:
    if not a <= t < b:
        return 1.0
    if kind == "stop":
        x = t - a
        if x < STOP_DROP_SEC:
            return 1.0 - (1.0 - STOP_DROP) * x / STOP_DROP_SEC
        u = min(max((x - STOP_DROP_SEC) / max(b - a - STOP_DROP_SEC, 1e-6), 0.0), 1.0)
        return STOP_DROP * (1.0 - u) ** STOP_CURVE
    return min(max((t - a) / max(b - a, 1e-6), 0.0), 1.0)


def music_gain(spec: dict, t: float) -> float:
    """The songs' volume at time t under music_spec's rule (0–1)."""
    g = spec["level"]
    for kind, a, b in spec["fades"]:
        g *= _fade_share(kind, a, b, t)
    f = spec["still_fade"]
    for a, b in spec["still"]:
        g -= (g - spec["still_volume"]) * min(max((t - a) / f, 0.0), 1.0) * min(max((b - t) / f, 0.0), 1.0)
    f = spec["dip_fade"]
    w = sum(min(max((t - a + f) / f, 0.0), 1.0) * min(max((b + f - t) / f, 0.0), 1.0) for a, b in spec["dips"])
    return g * (1.0 - (1.0 - spec["dip_level"]) * min(w, 1.0))


def music_gain_expr(spec: dict) -> str:
    """music_gain as an ffmpeg volume expression in t (flat sums, no deep nesting — a long compilation has many
    lines). For volume=volume='…':eval=frame."""
    parts = [f"{spec['level']:.4f}"]
    for kind, a, b in spec["fades"]:
        inside = f"gte(t,{a:.3f})*lt(t,{b:.3f})"
        rest = max(b - a - STOP_DROP_SEC, 1e-6)
        share = (f"if(lt(t-{a:.3f},{STOP_DROP_SEC}),1-{1 - STOP_DROP:.3f}*(t-{a:.3f})/{STOP_DROP_SEC},"
                 f"{STOP_DROP}*pow(clip(1-(t-{a + STOP_DROP_SEC:.3f})/{rest:.3f},0,1),{STOP_CURVE}))" if kind == "stop"
                 else f"clip((t-{a:.3f})/{max(b - a, 1e-6):.3f},0,1)")
        parts.append(f"(1-{inside}*(1-{share}))")
    if spec["still"]:
        f, lvl, q = spec["still_fade"], spec["level"], spec["still_volume"]
        dip = "+".join(f"clip((t-{a:.3f})/{f},0,1)*clip(({b:.3f}-t)/{f},0,1)" for a, b in spec["still"])
        parts.append(f"(1-{(lvl - q) / max(lvl, 1e-6):.4f}*min(1,{dip}))")
    if spec["dips"]:
        f = spec["dip_fade"]
        w = "+".join(f"clip((t-{a - f:.3f})/{f},0,1)*clip(({b + f:.3f}-t)/{f},0,1)" for a, b in spec["dips"])
        parts.append(f"(1-{1.0 - spec['dip_level']:.4f}*min(1,{w}))")
    return "*".join(parts)


# ---------------------------------------------------------------------------
# 🔔 The phase-change sound
# ---------------------------------------------------------------------------

def _chime(freq: float) -> np.ndarray:
    """A short soft bell: the note and its octave, quick to start, dying away over about half a second."""
    n = int(0.6 * AUDIO_RATE)
    t = np.arange(n) / AUDIO_RATE
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 7.0)
    tone = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t * 6.0)
    return 0.5 * tone * env


def chime_times(sched: list) -> list:
    """(time, profile) of every phase change — each phase's start but the very first."""
    return [(float(ph["start"]), timer_style(ph)) for ph in sched[1:]]


def write_chime_wav(sched: list, path) -> Path:
    """🔔 The phase-change sounds on one track of the compilation's length: a bell at every phase's start (the
    first phase's aside), pitched by the phase it starts."""
    total = total_seconds(sched)
    out = np.zeros(int(round(total * AUDIO_RATE)) + AUDIO_RATE)
    bells = {}
    for t, p in chime_times(sched):
        hit = bells.setdefault(p, _chime(style(p)["chime"]))
        i = int(round(t * AUDIO_RATE))
        out[i:i + len(hit)] += hit[:len(out) - i]
    return _write_wav(path, out[:int(round(total * AUDIO_RATE))])


# ---------------------------------------------------------------------------
# The voiceover
# ---------------------------------------------------------------------------

VOICE_EXT = (".mp3", ".m4a", ".wav", ".ogg")


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


PULSE_SEC = 3.0                    # 🔔 the timer pulses over a phase's last seconds


def overlay_image(profile: str, left: float, size: tuple, signals: bool = False, total: float = None):
    """The timer as a picture with a see-through background: a thin ring that empties as the phase runs out, the
    seconds left in the middle (00:17), and the phase's name on a coloured label underneath. left: seconds left.
    🔔 signals: the ring in the phase's own colour, wide and bright for the phase's first moment, and over the last
    PULSE_SEC seconds the number takes the ring's colour and the ring swells at every second."""
    from PIL import Image, ImageDraw
    prof = style(profile)
    total = float(total if total is not None else prof.get("seconds", PHASE_SEC))
    w, h = size
    s = 3                                                  # drawn large, shrunk down: smooth edges
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ring = w - 8
    x0, y0, x1, y1 = 4 * s, 4 * s, (4 + ring) * s, (4 + ring) * s
    d.ellipse((x0, y0, x1, y1), fill=(0, 0, 0, 105))                       # darkens the footage behind the numbers
    width = max(2 * s, int(ring * 0.022 * s))
    colour, number = RING, (255, 255, 255, 255)
    if signals:
        colour = tuple(prof["ring"])
        since = total - left
        if since < 0.4:                                    # the phase has just started: a wide, bright ring
            width = int(width * 2.4)
        elif 0 < left <= PULSE_SEC:                        # the last seconds: swells as each second starts
            frac = left - math.floor(left - 1e-9)
            width = int(width * (1.0 + 1.3 * max(frac - 0.4, 0.0) / 0.6))
            number = colour + (255,)
    d.arc((x0, y0, x1, y1), 0, 360, fill=(255, 255, 255, 50), width=width)
    share = float(min(max(left / max(total, 1e-6), 0.0), 1.0))
    if share > 0:
        d.arc((x0, y0, x1, y1), -90, -90 + 360 * share, fill=colour + (255,), width=width)
    secs = int(math.ceil(left - 1e-6))
    text = f"{secs // 60:02d}:{secs % 60:02d}" if profile not in ("override", "notyet") else f"+{secs:d}"
    f = _font(int(ring * 0.27 * s))
    box = d.textbbox((0, 0), text, font=f)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d.text((cx - (box[2] + box[0]) / 2, cy - (box[3] + box[1]) / 2), text, font=f, fill=number)
    lab_h = int(ring * 0.22) * s
    lx0, lx1 = int(w * 0.14) * s, int(w * 0.86) * s
    ly0 = y1 + int(ring * 0.05) * s
    d.rounded_rectangle((lx0, ly0, lx1, ly0 + lab_h), radius=int(lab_h * 0.28), fill=tuple(prof["colour"]) + (255,))
    fs = int(lab_h * 0.62)
    fb = _font(fs, bold=True)
    box = d.textbbox((0, 0), prof["label"], font=fb)
    while box[2] - box[0] > (lx1 - lx0) * 0.86 and fs > 6:     # a long label ("HANDS OFF") shrinks to fit its pill
        fs = int(fs * 0.92)
        fb = _font(fs, bold=True)
        box = d.textbbox((0, 0), prof["label"], font=fb)
    d.text(((lx0 + lx1) / 2 - (box[2] + box[0]) / 2, ly0 + lab_h / 2 - (box[3] + box[1]) / 2), prof["label"], font=fb,
           fill=(255, 255, 255, 255))
    return img.resize((w, h), Image.LANCZOS)


def round_badge(r: int, n: int, width: int):
    """▸ "ROUND 2 / 3" on a small dark label as wide as the timer, with a see-through background."""
    from PIL import Image, ImageDraw
    s = 3
    h = max(int(width * 0.17) // 2 * 2, 12)
    img = Image.new("RGBA", (width * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(width * 0.10) * s
    d.rounded_rectangle((pad, 0, width * s - pad, h * s - 1), radius=int(h * s * 0.45), fill=(0, 0, 0, 140))
    f = _font(int(h * s * 0.58), bold=True)
    text = f"ROUND {int(r)} / {int(n)}"
    box = d.textbbox((0, 0), text, font=f)
    d.text((width * s / 2 - (box[2] + box[0]) / 2, h * s / 2 - (box[3] + box[1]) / 2), text, font=f,
           fill=(255, 255, 255, 235))
    return img.resize((width, h), Image.LANCZOS)


def write_overlay(sched: list, folder, frame_height: int, signals: bool = False) -> tuple:
    """Draws the overlay's pictures: one set per timer look and clock length in use (timer_key — "fast-20",
    "handsoff-5"; a phase split by ✋ Hands off reads its set from where its clock stands). The render lays them on
    itself (edge_render._timer_pictures). Returns (width, height). signals: 🔔 the coloured ring, its flash and pulse."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    size = overlay_size(frame_height)
    step = 1.0 / OVERLAY_FPS
    for key in sorted({timer_key(ph) for ph in sched}):
        look, total = key.rsplit("-", 1)
        total = float(total)
        for k in range(int(round(total * OVERLAY_FPS))):
            overlay_image(look, total - k * step, size, signals, total=total).save(folder / f"{key}_{k:04d}.png")
    return size


def still_spans(sched: list) -> list:
    """[(start, end)] of the Wait / Stop stretches, neighbours joined (a round's Stop runs into the next Wait)."""
    out = []
    for ph in sched:
        if ph["profile"] in STILL:
            if out and abs(out[-1][1] - ph["start"]) < 1e-6:
                out[-1] = (out[-1][0], ph["end"])
            else:
                out.append((ph["start"], ph["end"]))
    return out


def describe(difficulty: str, rounds: int) -> str:
    r = round_seconds(difficulty)
    tot = r * rounds
    return (f"{rounds} rounds × {int(r // 60)}:{int(r % 60):02d} = {int(tot // 60)}:{int(tot % 60):02d} · "
            + " → ".join(PROFILES[p]["label"].title() for p in (round_phases(difficulty) if difficulty != "hard" else
                                                                 ["wait"])) + (" → a random middle (types and lengths) → Extra → Stop" if difficulty == "hard" else "")
            + " · 🏃 " + " / ".join(f"{PACE[difficulty][p]}" for p in ("slow", "medium", "fast", "extra")) + " bpm"
            + (f", +{PACE_BUILD[difficulty]:.0%} a round" if PACE_BUILD.get(difficulty) else "")) if difficulty in PACE else \
        (f"{rounds} rounds")


# ---------------------------------------------------------------------------
# The avatar and the intro screen
# ---------------------------------------------------------------------------

VIDEO_EXT = (".mp4", ".mov", ".webm", ".m4v")
AVATAR_MODES = {"corner": "🪟 In a corner", "split": "◧ Split screen"}
AVATAR_CORNER_SHARE = 0.42                         # her box's height, as a share of the frame's
PLACEHOLDER_CUE_SEC = 3.0                          # how long the stand-in figure shows at a cue
PLACEHOLDER_INTRO_SEC = 8.0                        # the intro's length until her intro clip exists


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
    got = []
    for t in thumbs:                                       # a thumbnail may come as a picture or as its JPEG bytes
        try:
            if isinstance(t, (bytes, bytearray)):
                import io
                t = Image.open(io.BytesIO(t))
            if t is not None:
                got.append(t.convert("RGB"))
        except Exception:          # noqa: BLE001 — a thumbnail that can't be read is left out
            continue
    thumbs = got[:6]
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


# ---------------------------------------------------------------------------
# The cue map: which voiceover line plays when (Daryl's Edge Mode Cue Map → edge_cues.json)
# ---------------------------------------------------------------------------

LINE_GAP = 0.35                    # the shortest silence between two lines
MAX_DELAY = 6.0                    # a line that finds its moment taken may start this much later; past that it is left out
CUE_MAP_FILE = Path(__file__).resolve().parent / "edge_cues.json"


def load_cue_map(path=None) -> dict:
    """{"lines": [{"name", "file", "seconds", "cues": [...]}]} — empty when there is no file."""
    import json
    p = Path(path) if path else CUE_MAP_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"lines": []}


def take_stem(stem: str, line_stem: str) -> int | None:
    """🎲 Is the file stem another take of the line (its own stem + " take 2", "_take2", " (2)", " alt 3", " v2")?
    → the take's number, else None. The line's own file is take 1."""
    import re
    m = re.fullmatch(re.escape(line_stem.lower()) + r"[\s_\-]*(?:\((\d+)\)|(?:take|alt|v)[\s_\-#]*(\d+))",
                     stem.lower().strip())
    if not m:
        return None
    n = int(m.group(1) or m.group(2))
    return n if n >= 2 else None


def pick_takes(plays: list, takes: dict, seed: int = 0) -> list:
    """🎲 Every play of a line with more than one take gets one of them: they take turns in a shuffled order, so
    no take repeats until all have played. takes: {line name: [files, the line's own first]}. Returns
    [take index (0 = the line's own) per play]."""
    rng = random.Random(seed + 3301)
    bags, out = {}, []
    for p in plays:
        have = takes.get(p["line"]) or []
        if len(have) < 2:
            out.append(0)
            continue
        if not bags.get(p["line"]):
            bags[p["line"]] = list(range(len(have)))
            rng.shuffle(bags[p["line"]])
        out.append(bags[p["line"]].pop())
    return out


def _cue_in_round(cue: dict, r: int, n_rounds: int, last_replaces: bool = False) -> bool:
    """Does the cue play in round r of n? No rounds picked = every round. "last" = the final round whatever its
    number. last_replaces: in the final round the "last" lines stand in for that round number's own — a cue set
    for round 3 (and not for "last") stays out when round 3 is the last."""
    want = [str(x) for x in cue.get("roundList") or []]
    if not want or {"1", "2", "3", "4", "5", "6"} <= set(want):     # every round number ticked = every round (Daryl's
        return True                                                  # "Keep going", 16 Nov) — the last one included
    if r == n_rounds and "last" in want:
        return True
    if r == n_rounds and last_replaces:
        return False
    return str(r) in want


def _free_spans(a: float, b: float, busy: list) -> list:
    """What is left of [a, b] after the busy (start, end) stretches, each widened by LINE_GAP."""
    spans = [(a, b)]
    for s, e in sorted(busy):
        s, e = s - LINE_GAP, e + LINE_GAP
        spans = [x for lo, hi in spans for x in ((lo, min(hi, s)), (max(lo, e), hi)) if x[1] - x[0] > 1e-6]
    return spans


# 🔁 Medium / Hard coverage (19 Nov 2026 check: Medium 51 % voiced with 65 s silences, Hard 31 % with 2½ min ones)
FILL_DIFFS = ("medium", "hard")
AUTOFILL_DEFAULT = {"medium": False, "hard": True}    # Medium off while Daryl plans it by hand (19 Nov)


def autofill_on(cue_map: dict, difficulty: str) -> bool:
    """🔁 The automatic fillers / callouts for this difficulty: the cue map's own setting ("autofill"), else the default."""
    if difficulty not in FILL_DIFFS:
        return False
    got = (cue_map or {}).get("autofill") or {}
    return bool(got.get(difficulty, AUTOFILL_DEFAULT.get(difficulty, False)))
FILL_GAP = {"medium": 18.0, "hard": 18.0}  # the longest silence left inside Slow / Medium / Fast / Extra
FILL_PHASES = ("slow", "medium", "fast", "extra")
CALLOUT_MAX_SEC = 3.0                     # 💀 a short line already cued at a phase's start doubles as that phase's callout
CALLOUT_CHANCE = 0.5                      # 💀 Hard: of the phase turns with no line at their start, about half get one
NO_REPEAT_SEC = 60.0                      # the same filler / callout never twice within this
# lines with no cue that fit a phase as callouts (the check found them never played)
CALLOUT_EXTRA = {"slow": ("Just stroke",), "fast": ("Stroke again - faster", "showing me you can do this"),
                 "extra": ("showing me you can do this",)}


def _phase_pools(cue_map: dict, difficulty: str, secs) -> tuple:
    """({phase: [line]} fillers — lines with an "at random" cue for this difficulty in that phase; {phase: [line]}
    callouts — short lines cued at the start of that phase, plus CALLOUT_EXTRA)."""
    fill, call = {}, {}
    by_name = {ln["name"]: ln for ln in cue_map.get("lines") or []}
    for ln in cue_map.get("lines") or []:
        for c in ln.get("cues") or []:
            if c.get("diffs") and difficulty not in c["diffs"]:
                continue
            for ph in c.get("phases") or []:
                if ph not in FILL_PHASES:
                    continue
                if c.get("type") == "random" and ln not in fill.setdefault(ph, []):
                    fill[ph].append(ln)
                if c.get("type") == "phase" and (c.get("at") or "start") == "start" and secs(ln) <= CALLOUT_MAX_SEC \
                        and ln not in call.setdefault(ph, []):
                    call[ph].append(ln)
    for ph, names in CALLOUT_EXTRA.items():
        for n in names:
            if n in by_name and by_name[n] not in call.setdefault(ph, []):
                call[ph].append(by_name[n])
    for ph in FILL_PHASES:                                # a phase's fillers also serve as its callouts
        for ln in fill.get(ph, []):
            if ln not in call.setdefault(ph, []):
                call[ph].append(ln)
    return fill, call


def _fill_silences(sched, cue_map, difficulty, plays, secs, rng, total) -> None:
    """🔁 Medium / Hard: 💀 (Hard) a callout at about half the phase turns that start in silence, then a filler into
    every silence longer than FILL_GAP inside Slow / Medium / Fast / Extra. A line is never heard twice within
    NO_REPEAT_SEC; the least heard of the pool goes first. Adds to `plays` (why: "callout" / "fills a silence")."""
    fill, call = _phase_pools(cue_map, difficulty, secs)
    heard = {}
    for p in plays:
        heard.setdefault(p["line"], []).append(p["time"])

    def busy(a, b):
        return any(a < p["time"] + p["seconds"] + LINE_GAP and b > p["time"] - LINE_GAP for p in plays)

    def pick(pool, t):
        ok = [ln for ln in pool if all(abs(t - x) >= NO_REPEAT_SEC for x in heard.get(ln["name"], []))]
        if not ok:
            return None
        rng.shuffle(ok)
        return min(ok, key=lambda ln: len(heard.get(ln["name"], [])))

    def add(ln, t, ph, why):
        plays.append({"time": round(t, 3), "seconds": round(secs(ln), 3), "line": ln["name"], "file": ln.get("file"),
                      "profile": ph["profile"], "round": ph["round"], "why": why, "moved": 0.0})
        heard.setdefault(ln["name"], []).append(t)

    turns = [ph for ph in sched if ph.get("kind", "phase") == "phase" and ph["profile"] in FILL_PHASES]
    if difficulty == "hard":
        for ph in turns:
            t = float(ph["start"]) + 0.3
            if any(ph["start"] - 0.5 <= p["time"] < ph["start"] + 3.0 for p in plays) or rng.random() >= CALLOUT_CHANCE:
                continue
            ln = pick(call.get(ph["profile"], []), t)
            if ln and t + secs(ln) <= ph["end"] and not busy(t, t + secs(ln)):
                add(ln, t, ph, f"callout at the start of {PROFILES[ph['profile']]['label'].title()}")
    gap = FILL_GAP[difficulty]
    for ph in turns:
        pool = list({ln["name"]: ln for ln in fill.get(ph["profile"], []) + call.get(ph["profile"], [])}.values())
        if not pool:
            continue
        for _ in range(12):
            spans = _free_spans(float(ph["start"]), min(float(ph["end"]), total),
                                [(p["time"], p["time"] + p["seconds"]) for p in plays])
            # the silence a line makes: inside this phase, joined to whatever silence runs on either side of it
            long_ = [(a, b) for a, b in spans if _silence_around(plays, a, b, total) > gap and b - a >= 3.0]
            if not long_:
                break
            a, b = max(long_, key=lambda s_: s_[1] - s_[0])
            t = a + min(max((b - a) * 0.4, 1.0), gap * 0.6)
            ln = pick(pool, t)
            if ln is None or t + secs(ln) > b:
                break
            add(ln, t, ph, f"fills a silence in {PROFILES[ph['profile']]['label'].title()}")


def _silence_around(plays, a, b, total) -> float:
    """How long the silence holding [a, b) runs, from the end of the line before to the start of the line after."""
    before = max([p["time"] + p["seconds"] for p in plays if p["time"] + p["seconds"] <= a + LINE_GAP + 1e-6], default=0.0)
    after = min([p["time"] for p in plays if p["time"] >= b - LINE_GAP - 1e-6], default=total)
    return after - before


def resolve_lines(sched: list, cue_map: dict, difficulty: str, seed: int = 0, durations: dict = None,
                  _raw: bool = False) -> dict:
    """The cue map played against one compilation's phases → when every line is heard, with no two overlapping.

    Returns {"plays": [{"time", "seconds", "line", "file", "profile", "round", "why", "moved"}] in time order,
    "intro": [lines in order], "outro": [lines], "clashes": [{"line", "with", "round", "where", "result"}],
    "unused": [line names with a cue that never came up in this compilation]}.

    How the cues are read:
    - a timer-phase cue set for a round plays once in that round: "start" and "during" at the phase's FIRST turn
      in the round, "end" at its LAST (Medium and Hard run a phase more than once a round). "End" starts the line
      `offset` seconds before the phase ends — at 0, as the next phase begins.
    - rounds: none picked = every round; a number = that round; "last" = the final round, whatever its number —
      and when the map has "last" lines, they replace the final round's own numbered lines (Daryl, 14 Nov).
    - "during", "at random" and "every so often" lines are placed after the fixed ones, in whatever silence is left.
    - two lines wanting the same moment: the one set for fewer rounds keeps it; the other starts straight after if
      that is within MAX_DELAY seconds, and is left out otherwise. Either way it is listed in "clashes".
    durations: {line name: real seconds} for the lines whose files have been measured (else the map's estimate)."""
    if difficulty == "hard" and not _raw and ((cue_map or {}).get("hard_rules") or {}).get("on"):
        import hard_rules as HR                       # 💀 Hard follows Chill and Medium by rules (20 Nov)
        return HR.resolve(sched, cue_map, seed, durations, cue_map["hard_rules"])
    rng = random.Random(seed + 4243)
    durations = durations or {}
    total = total_seconds(sched)
    n_rounds = max((ph["round"] for ph in sched), default=0)
    by_round = {}
    for ph in sched:
        by_round.setdefault(ph["round"], []).append(ph)
    fixed, flex, intro, outro, seen_cue, after, surprise_lines = [], [], [], [], {}, [], []
    has_last = any("last" in [str(x) for x in c.get("roundList") or []]
                   for ln in cue_map.get("lines") or [] for c in ln.get("cues") or [])
    in_round = lambda c, r: _cue_in_round(c, r, n_rounds, has_last)          # noqa: E731

    def secs(line):
        return float(durations.get(line["name"]) or line.get("play_seconds") or line.get("seconds") or 2.0)

    for li, line in enumerate(cue_map.get("lines") or []):
        d = secs(line)
        for ci, cue in enumerate(line.get("cues") or []):
            key = (line["name"], ci)
            if cue.get("diffs") and difficulty not in cue["diffs"]:
                continue
            seen_cue.setdefault(key, False)
            t = cue.get("type")
            base = {"line": line["name"], "file": line.get("file"), "seconds": d, "key": key,
                    "breadth": len(cue.get("roundList") or []) or 99}
            if t == "surprise":                            # 🎲 a line for a surprise: picked when one comes up (below)
                surprise_lines.append({**base, "kind": cue.get("kind") or "handsoff", "at": cue.get("at") or "start",
                                       "length": cue.get("length")})
                continue
            if t == "after":                               # 🔗 plays as soon as another line ends, wherever that one plays
                after.append({**base, "cue": cue, "target": str(cue.get("after") or ""),
                              "gap": max(float(cue.get("gap") if cue.get("gap") is not None else LINE_GAP), 0.0)})
            elif t == "intro":
                intro.append((cue.get("order", 99), li, line["name"])); seen_cue[key] = True
            elif t == "outro":
                outro.append((cue.get("order", 99), li, line["name"])); seen_cue[key] = True
            elif t == "phase":
                at, off = cue.get("at") or "start", float(cue.get("offset") or 0)
                for r, phases in by_round.items():
                    if not in_round(cue, r):
                        continue
                    for p in (cue.get("phases") or ORDER):
                        occ = [ph for ph in phases if ph["profile"] == p and ph.get("kind", "phase") == "phase"]
                        if not occ:
                            continue
                        turn = cue.get("turn")         # 1st / 2nd … turn of the phase in the round (Medium repeats
                        if turn:                         # Slow / Medium / Fast — Daryl, 19 Nov); none = start → first,
                            try:                         # end → last, as before
                                k_ = int(turn) - 1
                            except (TypeError, ValueError):
                                k_ = 0
                            if k_ >= len(occ):           # this difficulty hasn't that many turns: the line isn't here
                                continue
                            ph = occ[k_]
                        else:
                            ph = occ[-1] if at == "end" else occ[0]
                        if at == "end":                    # (⏩ an override is still this phase: its end is the end)
                            u = ph.get("unit", ph["index"])
                            ph = {**ph, "end": max(q["end"] for q in phases if q.get("unit", q["index"]) == u)}
                        where = {"round": r, "profile": p}
                        if at == "during":
                            flex.append({**base, **where, "lo": ph["start"] + max(off, 0.0), "hi": ph["end"],
                                         "why": f"during {PROFILES[p]['label'].title()}"})
                        elif at == "end":
                            fixed.append({**base, **where, "time": ph["end"] - off,
                                          "why": f"end of {PROFILES[p]['label'].title()}" + (f" −{off:g} s" if off else "")})
                        else:
                            fixed.append({**base, **where, "time": ph["start"] + off,
                                          "why": f"start of {PROFILES[p]['label'].title()}" + (f" +{off:g} s" if off else "")})
            elif t == "round":
                which = cue.get("which") or "every"
                for r, phases in by_round.items():
                    if not {"every": True, "first": r == 1, "last": r == n_rounds,
                            "number": r == int(cue.get("number") or 1)}.get(which, False):
                        continue
                    end = cue.get("at") == "end"
                    fixed.append({**base, "round": r, "profile": phases[-1 if end else 0]["profile"],
                                  "time": (phases[-1]["end"] - d) if end else phases[0]["start"],
                                  "why": f"{'end' if end else 'start'} of round {r}"})
            elif t == "time":
                try:
                    m, s_ = (str(cue.get("clock") or "0:00").split(":") + ["0"])[:2]
                    clock = int(m) * 60 + float(s_)
                except ValueError:
                    continue
                starts = [(r, phs[0]["start"]) for r, phs in by_round.items()] if cue.get("from") == "round" else [(None, 0.0)]
                for r, t0 in starts:
                    ph = phase_at(sched, t0 + clock)
                    if t0 + clock < total and ph and in_round(cue, ph["round"]) and (r is None or ph["round"] == r):
                        fixed.append({**base, "round": ph["round"], "profile": ph["profile"], "time": t0 + clock,
                                      "why": f"at {cue.get('clock')}" + (" into the round" if r else "")})
            elif t == "interval":
                every = max(float(cue.get("every") or 60), 1.0)
                tt = float(cue.get("startAfter") or 0) + every
                while tt < total:
                    ph = phase_at(sched, tt)
                    if ph and in_round(cue, ph["round"]) and (not cue.get("phases") or ph["profile"] in cue["phases"]):
                        fixed.append({**base, "round": ph["round"], "profile": ph["profile"], "time": tt, "breadth": 200,
                                      "why": f"every {every:g} s"})
                    tt += every
            elif t == "random":
                times, per = max(int(cue.get("times") or 1), 1), cue.get("per") or "round"
                ok = [ph for ph in sched if in_round(cue, ph["round"]) and ph.get("kind", "phase") == "phase"
                      and (not cue.get("phases") or ph["profile"] in cue["phases"])]
                groups = ([[ph] for ph in ok] if per == "phase" else [ok] if per == "compilation"
                          else [[ph for ph in ok if ph["round"] == r] for r in by_round])
                for g in groups:
                    picked = rng.sample(g, min(times, len(g))) if g else []
                    for ph in picked:
                        # 🔁 the round's other turns of the phase: tried when this one has no room (19 Nov coverage check)
                        alts = [(o["start"], o["end"], o["profile"], o["round"]) for o in g if o not in picked]
                        flex.append({**base, "round": ph["round"], "profile": ph["profile"], "lo": ph["start"], "hi": ph["end"],
                                     "alts": alts, "why": f"at random in {PROFILES[ph['profile']]['label'].title()}"})
    plays, clashes = [], []

    def blocker(a, b):
        return next((p for p in plays if a < p["time"] + p["seconds"] + LINE_GAP - 1e-6 and b > p["time"] - LINE_GAP + 1e-6), None)

    def where(c):
        return f"round {c['round']}, {c['why']}"

    # 🎲 a surprise's lines come first and always play (must play): for each surprise and each moment of it (start /
    # end / before), one line of its kind — one made for its length when there is one — picked at random
    for ph in sched:
        kind = ph.get("surprise")
        if not kind:
            continue
        n = float(ph["end"]) - float(ph["start"])
        for at in ("before", "start", "end"):
            pool = [c for c in surprise_lines if c["kind"] == kind and c["at"] == at]
            if not pool and kind == "falsefinish":
                pool = [c for c in surprise_lines if c["kind"] == "override" and c["at"] == at]
            if not pool:
                continue
            fit = [c for c in pool if c.get("length") not in (None, "") and abs(float(c["length"]) - n) < 0.5]
            c = rng.choice(fit or [c for c in pool if c.get("length") in (None, "")] or pool)
            t = {"before": ph["start"] - c["seconds"], "start": ph["start"], "end": ph["end"] - c["seconds"]}[at]
            t = max(0.0, min(t, total - c["seconds"]))
            while (hit := blocker(t, t + c["seconds"])) is not None:
                t = hit["time"] + hit["seconds"] + LINE_GAP
            if t + c["seconds"] > total + 1e-6:
                continue
            label = {"handsoff": "hands off", "override": "override", "falsefinish": "false finish"}[kind]
            plays.append({"time": round(t, 3), "seconds": round(c["seconds"], 3), "line": c["line"], "file": c["file"],
                          "profile": ph["profile"], "round": ph["round"], "key": c.get("key"), "why": f"🎲 {label} ({n:g} s), {at}",
                          "moved": 0.0})
            seen_cue[c["key"]] = True
    for c in sorted(fixed, key=lambda c: (c["time"], c["breadth"], c["line"])):
        t = min(c["time"], total - c["seconds"])                     # a line never runs past the compilation's end
        moved = round(t - c["time"], 2)
        hit = blocker(t, t + c["seconds"])
        tries = 0
        while hit is not None and tries < 6:
            first = hit
            t2 = hit["time"] + hit["seconds"] + LINE_GAP
            if t2 - c["time"] > MAX_DELAY or t2 + c["seconds"] > total:
                clashes.append({"line": c["line"], "with": first["line"], "round": c["round"], "where": where(c), "result": "left out"})
                t = None
                break
            t, tries = t2, tries + 1
            hit = blocker(t, t + c["seconds"])
            if hit is None:
                clashes.append({"line": c["line"], "with": first["line"], "round": c["round"], "where": where(c),
                                "result": f"starts {t - c['time']:.1f} s late"})
                moved = round(t - c["time"], 2)
        if t is None or hit is not None:
            if t is not None:
                clashes.append({"line": c["line"], "with": hit["line"], "round": c["round"], "where": where(c), "result": "left out"})
            continue
        plays.append({"time": round(t, 3), "seconds": round(c["seconds"], 3), "line": c["line"], "file": c["file"],
                      "profile": c["profile"], "round": c["round"], "why": c["why"], "moved": moved, "key": c.get("key")})
        seen_cue[c["key"]] = True
    for _pass in range(3):                                 # 🔗 joined lines, after the lines they follow (chains: up to 3 deep)
        for a in after:
            for tp in [p for p in plays if p["line"] == a["target"]]:
                done = tp.setdefault("_joined", set())
                if a["line"] in done or not in_round(a["cue"], tp["round"]):
                    continue
                done.add(a["line"])
                t = tp["time"] + tp["seconds"] + a["gap"]
                where_ = f"round {tp['round']}, straight after {a['target']}"
                if t + a["seconds"] > total + 1e-6:
                    clashes.append({"line": a["line"], "with": "the compilation's end", "round": tp["round"], "where": where_,
                                    "result": "left out"})
                    continue
                hit = next((p for p in plays if p is not tp and t < p["time"] + p["seconds"] + LINE_GAP - 1e-6
                            and t + a["seconds"] > p["time"] - LINE_GAP + 1e-6), None)
                if hit is not None:
                    clashes.append({"line": a["line"], "with": hit["line"], "round": tp["round"], "where": where_, "result": "left out"})
                    continue
                plays.append({"time": round(t, 3), "seconds": round(a["seconds"], 3), "line": a["line"], "file": a["file"],
                              "profile": tp["profile"], "round": tp["round"], "why": f"straight after {a['target']}", "moved": 0.0, "key": a.get("key")})
                seen_cue[a["key"]] = True
    for p in plays:
        p.pop("_joined", None)
    def room(lo_, hi_, need):
        return [(lo, hi - need) for lo, hi in _free_spans(lo_, min(hi_, total), [(p["time"], p["time"] + p["seconds"])
                                                                                for p in plays]) if hi - lo >= need]
    for c in flex:
        spans = room(c["lo"], c["hi"], c["seconds"])
        if not spans and c.get("alts"):                    # 🔁 A: another turn of the phase this round, the emptiest first
            for lo_, hi_, prof, rnd in sorted(c["alts"], key=lambda a: -sum(h - l for l, h in room(a[0], a[1], 0.0))):
                spans = room(lo_, hi_, c["seconds"])
                if spans:
                    c = {**c, "lo": lo_, "hi": hi_, "profile": prof, "round": rnd,
                         "why": f"at random in {PROFILES[prof]['label'].title()}"}
                    break
        if not spans:
            inside = [p["line"] for p in plays if p["time"] < c["hi"] and p["time"] + p["seconds"] > c["lo"]]
            clashes.append({"line": c["line"], "with": ", ".join(inside) or "the phase's length", "round": c["round"],
                            "where": where(c), "result": "left out — no silence long enough"})
            continue
        lo, hi = rng.choices(spans, [max(h - l, 0.05) for l, h in spans])[0]
        plays.append({"time": round(rng.uniform(lo, hi), 3), "seconds": round(c["seconds"], 3), "line": c["line"],
                      "file": c["file"], "profile": c["profile"], "round": c["round"], "why": c["why"], "moved": 0.0, "key": c.get("key")})
        seen_cue[c["key"]] = True
    if autofill_on(cue_map, difficulty):
        _fill_silences(sched, cue_map, difficulty, plays, secs, rng, total)
    plays.sort(key=lambda p: p["time"])
    return {"plays": plays, "intro": [n for _, _, n in sorted(intro)], "outro": [n for _, _, n in sorted(outro)],
            "clashes": clashes, "unused": sorted({k[0] for k, v in seen_cue.items() if not v})}
