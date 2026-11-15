"""
⚡ Edge Mode's energy-matched mix: song sections picked to fit each phase's energy and crossfaded on the beat.

1. ENERGY MAP — every bar of a song gets an energy score (0–1, within the song's own range) from the measures the
   🧩 Sections suggester already takes per bar (section_bounds.bar_levels): loudness first, then the drums / hats /
   other layers where the analysis has them, and how busy the drum hits are.
2. ENERGY TARGETS — Wait low, Slow low–mid, Medium mid, Fast high, Extra the peak, Stop winding down (the 🌙 fade
   takes it to silence anyway).
3. SECTION PICKER — phase by phase, the 20 s stretch (starting on a bar line) whose energy is closest to the
   target, from the songs allowed there (the round's own song when 🎵 New song each round is on, else any song in
   the playlist). Carrying straight on in the same song is preferred (no join at all); a stretch already used is
   avoided.
4. CROSSFADES — the music for phase k starts XFADE / 2 before the phase and ends XFADE / 2 after it, the incoming
   section's first bar line landing exactly on the phase change; consecutive pieces overlap by XFADE and blend
   (equal power between different music, plain linear where a song simply carries on).

Pure logic plus the mix writer (numpy, one ffmpeg per song to decode it). edge_render._mix uses write_mix; the
compile (planner/edge.py) uses energy_map + pick_sections.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

import edge_mode as E

XFADE = 1.5                        # seconds two sections blend over, centred on the phase change
RATE = 44100

# what each phase asks of the music (0–1 within each song's own range)
TARGETS = {"wait": 0.20, "slow": 0.38, "medium": 0.56, "fast": 0.76, "extra": 0.95, "stop": 0.25}
WEIGHTS = {"loudness": 0.55, "drums": 0.15, "hats": 0.05, "other": 0.05, "bass": 0.05}    # + 0.15 drum hits
REUSE_PENALTY = 0.6                # a stretch already played (scaled by how much of it overlaps)
CARRY_BONUS = 0.06                 # carrying straight on in the same song (no join) — worth this much energy error
SAME_SONG_JUMP = 0.02              # a jump within the same song, a little cheaper than a change of song


def energy_map(track: dict) -> dict:
    """{"bars": bar starts, "ends", "energy": 0–1 per bar, "duration"} for an analysed track."""
    import section_bounds as SB
    lv = SB.bar_levels(track)
    bars = np.asarray(lv["bars"], float)
    duration = float(lv["duration"])
    ends = np.append(bars[1:], duration)
    total, weight = np.zeros(len(bars)), 0.0
    for name, w in WEIGHTS.items():
        v = lv["levels"].get(name)
        if v is not None and len(v) == len(bars):
            total += w * np.asarray(v, float)
            weight += w
    hits = lv.get("hits")
    if hits is not None and len(hits) == len(bars) and np.nanmax(hits) > 0:
        h = np.nan_to_num(np.asarray(hits, float))
        total += 0.15 * h / max(np.percentile(h, 95), 1e-6)
        weight += 0.15
    e = total / max(weight, 1e-6)
    lo, hi = np.percentile(e, 5), np.percentile(e, 95)
    e = np.clip((e - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return {"bars": bars, "ends": ends, "energy": e, "duration": duration}


def window_energy(emap: dict, a: float, b: float) -> float:
    """The energy of the stretch [a, b) of a song: its bars' energies, weighted by how much of each falls inside."""
    bars, ends, e = emap["bars"], emap["ends"], emap["energy"]
    cover = np.clip(np.minimum(ends, b) - np.maximum(bars, a), 0.0, None)
    return float((cover * e).sum() / cover.sum()) if cover.sum() > 0 else 0.0


def _overlap(a: float, b: float, used: list) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in used)


def pick_sections(sched: list, maps: dict, allowed=None, xfade: float = XFADE) -> list:
    """The music for every phase → [{"phase", "round", "profile", "track_id", "src_start" (where the piece starts in
    the song), "offset" (where it starts in the compilation), "play" (seconds), "bar" (the song's time that lands
    on the phase's start), "energy", "target", "carry" (continues the previous piece's song without a jump)}].
    maps: {track_id: energy_map}; allowed(phase) → the track ids that phase may use (default: all)."""
    total = E.total_seconds(sched)
    out, used = [], {}
    prev = None
    for k, ph in enumerate(sched):
        lead = xfade / 2 if k > 0 else 0.0                 # the piece starts this much before the phase…
        tail = xfade / 2 if k < len(sched) - 1 else 0.0    # …and runs on this much after it
        length = float(ph["end"] - ph["start"])
        target = TARGETS.get(ph["profile"], 0.5)
        best = None
        for tid in (allowed(ph) if allowed else list(maps)):
            m = maps.get(tid)
            if m is None:
                continue
            cands = list(m["bars"])
            if prev is not None and prev["track_id"] == tid:
                cands.append(prev["bar"] + float(sched[k - 1]["end"] - sched[k - 1]["start"]))      # carry straight on
            for bar in cands:
                a, b = bar - lead, bar + length + tail
                if a < -1e-6 or b > m["duration"] + 1e-6:
                    continue
                en = window_energy(m, bar, bar + length)
                score = abs(en - target)
                score += REUSE_PENALTY * _overlap(bar, bar + length, used.get(tid, [])) / length
                carry = bool(prev and prev["track_id"] == tid and abs(prev["bar"] + float(sched[k - 1]["end"] - sched[k - 1]["start"]) - bar) < 0.05)
                if carry:
                    score -= CARRY_BONUS
                elif prev and prev["track_id"] == tid:
                    score -= SAME_SONG_JUMP
                if best is None or score < best[0] - 1e-9:
                    best = (score, tid, float(bar), en, carry)
        if best is None:
            raise ValueError(f"No song is long enough for phase {k + 1} ({ph['profile']}).")
        _, tid, bar, en, carry = best
        piece = {"phase": k, "round": ph["round"], "profile": ph["profile"], "track_id": tid,
                 "src_start": round(bar - lead, 3), "offset": round(float(ph["start"]) - lead, 3),
                 "play": round(min(length + lead + tail, total - (float(ph["start"]) - lead)), 3),
                 "bar": round(bar, 3), "energy": round(en, 3), "target": target, "carry": carry}
        out.append(piece)
        used.setdefault(tid, []).append((bar, bar + length))
        prev = piece
    return out


# ---------------------------------------------------------------------------
# The music, written as one file
# ---------------------------------------------------------------------------

def _decode(input_args: list) -> np.ndarray:
    """A song as float32 stereo samples at RATE (n, 2)."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *input_args, "-map", "0:a:0", "-ac", "2", "-ar", str(RATE),
                        "-f", "s16le", "-"], capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("A song couldn't be read for the energy mix:\n" + r.stderr.decode("utf-8", "replace")[-600:])
    return np.frombuffer(r.stdout, dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.0


def _curves(n: int, equal_power: bool):
    x = (np.arange(n, dtype=np.float32) + 0.5) / max(n, 1)
    if equal_power:
        return np.sin(x * np.pi / 2)[:, None], np.cos(x * np.pi / 2)[:, None]
    return x[:, None], (1.0 - x)[:, None]


def write_mix(pieces: list, inputs: dict, total: float, out_path, xfade: float = XFADE) -> Path:
    """The pieces laid end to end, each overlapping the next by `xfade` seconds and blended there, written as a
    16-bit stereo WAV of `total` seconds. inputs: {track_id: ffmpeg input arguments of a local file}. Written as it
    goes (a long compilation never sits whole in memory)."""
    out_path = Path(out_path)
    songs = {tid: _decode(args) for tid, args in inputs.items()}
    n_total = int(round(total * RATE))
    nx = int(round(xfade * RATE))
    written, carry = 0, None                               # carry: the previous piece's last nx samples, faded out

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)

        def put(x):
            nonlocal written
            x = x[:max(n_total - written, 0)]
            w.writeframes((np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes())
            written += len(x)
        for k, p in enumerate(pieces):
            song = songs[p["track_id"]]
            a = int(round(float(p["src_start"]) * RATE))
            n = int(round(float(p["play"]) * RATE))
            seg = song[a:a + n]
            if len(seg) < n:                               # (a song a hair shorter than its analysis says)
                seg = np.concatenate([seg, np.zeros((n - len(seg), 2), np.float32)])
            seg = seg.copy()
            if carry is not None:
                m = min(nx, len(seg), len(carry))
                fin, _ = _curves(m, not p.get("carry"))
                seg[:m] = seg[:m] * fin + carry[:m]
            last = k == len(pieces) - 1
            if not last:
                m = min(nx, len(seg))
                _, fout = _curves(m, not pieces[k + 1].get("carry"))
                carry = seg[len(seg) - m:] * fout
                put(seg[:len(seg) - m])
            else:
                put(seg)
        if written < n_total:
            put(np.zeros((n_total - written, 2), np.float32))
    return out_path
