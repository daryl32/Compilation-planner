"""
⚡ Edge Mode's energy-matched mix: song sections picked to fit the phases' energy, joined by the 🎚️ Mix tab's
own engine (mixer.render) — 🧠 stem blends where both songs have their four stems, a bar-aligned crossfade with
a bass swap where not.

1. ENERGY MAP — every bar of a song gets an energy score (0–1, within the song's own range) from the measures the
   🧩 Sections suggester already takes per bar (section_bounds.bar_levels): loudness first, then the drums / hats /
   other layers where the analysis has them, and how busy the drum hits are.
2. PAIRS — phases are played in pairs on one stretch of a song (Daryl, 15 Nov: 20 s alone felt too jumpy):
   Wait + Slow, Medium + Fast, Extra + Stop (Stop simply fading — the 🌙 fade takes it to silence). A phase with no
   partner next to it (Medium's extra Slow, Hard's random draws) has a stretch of its own.
3. ENERGY TARGETS — Wait low, Slow low–mid, Medium mid, Fast high, Extra the peak; each phase of a pair is judged on
   its own half of the stretch (Stop not at all: it is fading).
4. SECTION PICKER — group by group, the stretch (starting on a bar line) closest to the targets, from the songs
   allowed there (the round's own song with 🎵 New song each round, else any song in the playlist). Carrying
   straight on in the same song needs no join and is preferred; a stretch already played is avoided; a change of
   song pays for a poor fit of key and tempo (core/mixing.partner_score — the Mix tab's own), and a jump within a
   song is cheaper where the bars either side sound alike (the Mix tab's bar similarity, when its features are on
   this machine).
5. JOINS — built as a Mix recipe per round (rounds meet in the silent Stop → Wait gap, so each renders on its
   own): every join a 🧠 neural blend of NEURAL_BARS bars that ends exactly on the phase change ("comes in early").
   Into a livelier stretch it is the "edge_land" profile (core/stem_mix.py) — the incoming drums rise through the
   blend's second half and are full exactly on the change, its bass swaps on that downbeat, harmonics and vocals
   blend in beneath; on the change's first frame only the incoming song is left (Daryl, 16 Nov: Drums first
   sounded good but landed a bar early — the drop must hit exactly as the phase changes). The picker also
   prefers starting those stretches on the song's own drops. Into a calmer one: Smooth.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


PAIRS = {("wait", "slow"), ("medium", "fast"), ("extra", "stop")}
TARGETS = {"wait": 0.20, "slow": 0.38, "medium": 0.56, "fast": 0.76, "extra": 0.95, "stop": 0.25}
WEIGHTS = {"loudness": 0.55, "drums": 0.15, "hats": 0.05, "other": 0.05, "bass": 0.05}    # + 0.15 drum hits
REUSE_PENALTY = 0.6                # a stretch already played (scaled by how much of it overlaps)
CARRY_BONUS = 0.06                 # carrying straight on in the same song (no join)
SWITCH_PENALTY = 0.12              # a change of song, times (1 − how well key and tempo fit)
JUMP_BONUS = 0.05                  # a jump within a song, times how alike the bars either side sound
NEURAL_BARS = 2                    # 🧠 the blend's length (≈ 3–4 s — twice the first version's 1.5 s)
MIN_BLEND_SEC = 3.0                # … at least this long (more bars for a fast song)
DROP_BONUS = 0.08                  # a livelier stretch starting on one of the song's own drops (its analysis)
RISE_BONUS = 0.05                  # … or where its energy jumps (times the jump, 0–1)


def energy_map(track: dict) -> dict:
    """{"bars": bar starts, "ends", "energy": 0–1 per bar, "duration", "info": {track_id, bpm, key, energy}}."""
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
    loud = lv["levels"].get("loudness")
    e = np.clip((e - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    drops = sorted(float(t) for t in ((track.get("analysis") or {}).get("drops") or []))
    return {"bars": bars, "ends": ends, "energy": e, "duration": duration, "drops": drops,
            "info": {"track_id": track.get("track_id"), "bpm": track.get("bpm"), "key": track.get("key"),
                     "energy": float(np.mean(loud)) if loud is not None and len(loud) else 0.5}}


def window_energy(emap: dict, a: float, b: float) -> float:
    """The energy of the stretch [a, b) of a song: its bars' energies, weighted by how much of each falls inside."""
    bars, ends, e = emap["bars"], emap["ends"], emap["energy"]
    cover = np.clip(np.minimum(ends, b) - np.maximum(bars, a), 0.0, None)
    return float((cover * e).sum() / cover.sum()) if cover.sum() > 0 else 0.0


def groups(sched: list) -> list:
    """The phases in pairs, round by round: [[phase index, …]] — Wait + Slow, Medium + Fast, Extra + Stop when they
    come one after the other in a round; then the phases left on their own pair up with a neighbour that is also on
    its own (Hard's random draws, Medium's extra Slow…), so a stretch of music lasts 40 s wherever it can (Daryl,
    16 Nov: no join every 20 s). Only a phase with no free neighbour stands alone."""
    out, k = [], 0
    while k < len(sched):
        ph = sched[k]
        nxt = sched[k + 1] if k + 1 < len(sched) else None
        if nxt and nxt["round"] == ph["round"] and (ph["profile"], nxt["profile"]) in PAIRS:
            out.append([k, k + 1])
            k += 2
        else:
            out.append([k])
            k += 1
    merged = []
    for g in out:                                          # lone phases, next to each other in a round: one stretch
        if (len(g) == 1 and merged and len(merged[-1]) == 1 and sched[merged[-1][0]]["round"] == sched[g[0]]["round"]
                and sched[g[0]]["profile"] != "stop"):
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    return merged


def drop_score(emap: dict, bar: float) -> float:
    """How much the song drops in at this bar (0–1): 1 on one of its analysed drops (within a quarter bar), else
    its energy jump — the bar after against the two bars before."""
    bars, e = emap["bars"], emap["energy"]
    i = int(np.argmin(np.abs(bars - bar))) if len(bars) else 0
    bar_len = float(np.median(np.diff(bars))) if len(bars) > 1 else 2.0
    if any(abs(d - bar) < 0.25 * bar_len for d in emap.get("drops") or ()):
        return 1.0
    if i < 2 or i >= len(e):
        return 0.0
    return float(np.clip((e[i] - e[i - 2:i].mean()) / 0.5, 0.0, 0.99))      # (only a real drop counts as 1)


def _overlap(a: float, b: float, used: list) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in used)


def _switch_fit(a: dict, b: dict) -> float:
    """How well song b follows song a (key and tempo, 0–1) — the Mix tab's partner score; 0.5 when unknown."""
    try:
        from core import mixing as MX
        return float(MX.partner_score(a["info"], b["info"])["score"])
    except Exception:          # noqa: BLE001
        return 0.5


def pick_sections(sched: list, maps: dict, allowed=None, jump_fit=None) -> list:
    """The music for every group of phases → [{"group", "phases" (indices), "round", "profile" (the first phase's),
    "profiles", "track_id", "bar" (where the stretch starts in the song), "offset" (where it starts in the
    compilation), "play" (seconds), "energy" (the stretch's, its phases' halves averaged), "target", "carry"
    (continues the previous stretch of the same round without a jump), "first" (the round's first: no join)}].
    maps: {track_id: energy_map}; allowed(phase) → the track ids it may use (default: all); jump_fit(track_id,
    end_t, start_t) → 0–1 how seamless a jump within a song sounds (None: unknown)."""
    out, used = [], {}
    prev = None
    for gi, idx in enumerate(groups(sched)):
        phs = [sched[i] for i in idx]
        a0, length = float(phs[0]["start"]), float(phs[-1]["end"] - phs[0]["start"])
        judged = [ph for ph in phs if not (ph["profile"] == "stop" and len(phs) > 1)] or phs
        target = float(np.mean([TARGETS.get(ph["profile"], 0.5) for ph in judged]))
        first = prev is None or prev["round"] != phs[0]["round"]
        best = None
        for tid in (allowed(phs[0]) if allowed else list(maps)):
            m = maps.get(tid)
            if m is None:
                continue
            follow = None if first or prev["track_id"] != tid else prev["bar"] + prev["play"]
            cands = list(m["bars"]) + ([follow] if follow is not None else [])
            for bar in cands:
                if bar < -1e-6 or bar + length > m["duration"] + 1e-6:
                    continue
                errs = [abs(window_energy(m, bar + ph["start"] - a0, bar + ph["end"] - a0) - TARGETS.get(ph["profile"], 0.5))
                        for ph in judged]
                en = float(np.mean([window_energy(m, bar + ph["start"] - a0, bar + ph["end"] - a0) for ph in judged]))
                score = float(np.mean(errs))
                score += REUSE_PENALTY * _overlap(bar, bar + length, used.get(tid, [])) / length
                carry = bool(follow is not None and abs(bar - follow) < 0.05)      # (a plain bool: the plan is JSON)
                if not first and target > prev["target"] + 0.05 and not carry:      # 🥁 livelier: land on a drop
                    q = drop_score(m, bar)
                    score -= DROP_BONUS * q if q >= 1.0 else RISE_BONUS * q
                if carry:
                    score -= CARRY_BONUS
                elif not first and prev["track_id"] != tid:
                    score += SWITCH_PENALTY * (1.0 - _switch_fit(maps[prev["track_id"]], m))
                elif not first and jump_fit is not None:
                    q = jump_fit(tid, prev["bar"] + prev["play"], bar)
                    if q is not None:
                        score -= JUMP_BONUS * q
                if best is None or score < best[0] - 1e-9:
                    best = (score, tid, float(bar), en, carry)
        if best is None:
            raise ValueError(f"No song is long enough for {' + '.join(p['profile'].title() for p in phs)} "
                             f"({length:.0f} s) in round {phs[0]['round']}.")
        _, tid, bar, en, carry = best
        bar, en, carry = float(bar), float(en), bool(carry)
        piece = {"group": gi, "phases": list(idx), "round": phs[0]["round"], "profile": phs[0]["profile"],
                 "profiles": [ph["profile"] for ph in phs], "track_id": tid, "bar": round(bar, 3),
                 "src_start": round(bar, 3), "offset": round(a0, 3), "play": round(length, 3),
                 "energy": round(en, 3), "target": round(target, 3), "carry": carry, "first": first}
        out.append(piece)
        used.setdefault(tid, []).append((bar, bar + length))
        prev = piece
    return out


def blend_bars(track: dict) -> int:
    """🧠 How many bars a blend into this song lasts: NEURAL_BARS, more for a fast song (at least MIN_BLEND_SEC)."""
    bpm = float(track.get("bpm") or 120.0) or 120.0
    bar = 4 * 60.0 / bpm
    return max(NEURAL_BARS, int(math.ceil(MIN_BLEND_SEC / bar - 1e-6)))


def round_recipes(pieces: list, tracks: dict) -> list:
    """One 🎚️ Mix recipe per round: [(round, recipe, seconds)]. Stretches that carry straight on are one piece;
    every other join is a 🧠 neural blend (the Mix engine falls back to a bar-aligned crossfade with a bass swap
    when a song has no drum / bass / other stems), all of it before the phase change so the incoming stretch is
    fully in as its phase starts. Each song at its own tempo (the phases' times stay exact)."""
    out = []
    for r in sorted({p["round"] for p in pieces}):
        mine = [p for p in pieces if p["round"] == r]
        rec_pieces, prev_target = [], 0.0
        for p in mine:
            if rec_pieces and p["carry"] and rec_pieces[-1]["track"] == p["track_id"]:
                rec_pieces[-1]["end"] = round(p["bar"] + p["play"], 3)          # carries on: one longer piece
                prev_target = p.get("target", 0.5)
                continue
            piece = {"track": p["track_id"], "start": p["bar"], "end": round(p["bar"] + p["play"], 3),
                     "part": "full", "uid": f"g{p['group']}", "label": " + ".join(x.title() for x in p["profiles"])}
            if rec_pieces:
                livelier = float(p.get("target", 0.5)) > float(prev_target) + 0.05
                piece["xfade"] = {"neural": {"profile": "edge_land" if livelier else "smooth",
                                             "bars": blend_bars(tracks.get(p["track_id"]) or {})},
                                  "bass_swap": True}
            rec_pieces.append(piece)
            prev_target = p.get("target", 0.5)
        seconds = round(sum(p["play"] for p in mine), 3)
        out.append((r, {"id": f"edge_round_{r}", "name": f"Edge round {r}", "kind": "mashup", "pieces": rec_pieces,
                        "layers": [], "bpm": None, "xfade_same": 0.5, "xfade_other": 8, "fade_in": 0.0,
                        "fade_out": 0.0, "level_match": True}, seconds))
    return out


def render_rounds(pieces: list, tracks: dict, tmp_dir, progress=None) -> tuple:
    """Each round's music rendered by the 🎚️ Mix engine → ([(round, file, seconds)], warnings). The songs (and
    their four stems, where they have them) are fetched to this machine first."""
    import mixer
    tmp_dir = Path(tmp_dir)
    out, warnings = [], []
    for r, rec, seconds in round_recipes(pieces, tracks):
        need = mixer.parts_needed(rec, tracks)
        for tid, parts in need.items():
            err = mixer.ensure_local(tracks[tid], parts=tuple(sorted(parts)))
            if err:
                if any(p != "full" for p in parts) and not mixer.ensure_local(tracks[tid]):
                    warnings.append(f"{tid}: its stems couldn't be fetched ({err}) — plain crossfades into it.")
                    continue
                raise RuntimeError(err)
        f = tmp_dir / f"music_round_{r}.m4a"
        info = mixer.render(rec, {tid: tracks[tid] for tid in need}, f)
        warnings += [w for w in info.get("warnings") or [] if w not in warnings]
        out.append((r, f, seconds))
        if progress:
            progress(f"round {r} mixed")
    return out, warnings
