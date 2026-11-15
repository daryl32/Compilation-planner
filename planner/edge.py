"""⏱️ Edge Mode (Compilation Planner → ⏱️ Edge Mode) — a compilation run by a countdown timer.

Setup: 🎵 a playlist (optional — any number of songs, played back to back and started again until the time is
filled) · 🥁 the beat sound (on by itself when there is no song) · rounds (3–6) · 🎬 videos / tags (optional — the
best fit from every video otherwise) · difficulty (Chill / Medium / Hard) · 🎙️ voiceover (optional).

The footage is matched by the 🤖 new matcher, 🎯 Movement only: to each song as usual, or — with no song — to a
generated beat whose speed follows the timer's phases. The rendered video carries the countdown overlay, the
voiceover (the songs at a quarter volume under it) and the beat; the plan's "edge" block carries the same for Blender.
The rules (phases, rounds, beat, overlay) are in edge_mode.py; the render step in edge_render.py.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import streamlit as st

import edge_mode as E
import library_common as LC
import planner.compose as PC
import settings_store as SSTORE
from config import CATALOGUE_DIR, PLANS_DIR, PREVIEW_DIR
from library_common import format_mmss

RESULT_KEY = "edge_result"
KEPT = {"edge_playlist": [], "edge_rounds": 3, "edge_difficulty": "chill", "edge_voice": False, "edge_beat": False,
        "edge_corner": "top-left", "edge_max_videos": 0, "edge_avatar": False, "edge_avatar_mode": "corner",
        "edge_avatar_corner": "bottom-right", "edge_intro": False, "edge_quick": True, "edge_stream": True, "edge_render_name": "",
        "edge_song_per_round": False, "edge_energy_mix": False}
KEPT_IF_SET = tuple("ce_" + k for k in ("actors", "tags", "hair", "gg", "new", "minlen", "exclude"))


def voice_dir() -> Path:
    """Where the voiceover clips live: <data>/edge_voiceover/<phase>/ — beside the catalogue, on this machine."""
    return Path(CATALOGUE_DIR).parent / "edge_voiceover"


def lines_dir() -> Path:
    """Where the cue map's voiceover lines live on this machine: <data>/edge_voiceover/lines/<line>.mp3."""
    return voice_dir() / "lines"


def line_file(line: dict) -> Path | None:
    """The line's audio on this machine, or None when it hasn't been fetched yet."""
    f = lines_dir() / str(line.get("file") or (line["name"] + ".mp3"))
    return f if f.exists() and f.stat().st_size > 0 else None


def cue_map_path() -> Path:
    """The cue map in use: <data>/edge_voiceover/edge_cues.json when one has been uploaded or fetched from Google
    Drive (Edge_voiceover/edge_cues.json), else the copy that ships with the app."""
    here = voice_dir() / "edge_cues.json"
    return here if here.exists() and here.stat().st_size > 0 else E.CUE_MAP_FILE


def load_cue_map() -> dict:
    return E.load_cue_map(cue_map_path())


def save_cue_map(raw: bytes) -> str | None:
    """A cue map file (the Cue Map artifact's export) checked and kept as <data>/edge_voiceover/edge_cues.json.
    Returns an error, or None."""
    try:
        got = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return f"That isn't a cue map file ({e})."
    if not isinstance(got, dict) or not isinstance(got.get("lines"), list) or not all(
            isinstance(ln, dict) and ln.get("name") for ln in got["lines"]):
        return "That file has no list of lines with names — is it the cue map's export?"
    voice_dir().mkdir(parents=True, exist_ok=True)
    (voice_dir() / "edge_cues.json").write_text(json.dumps(got, indent=1), encoding="utf-8")
    return None


def line_takes(line: dict, with_avatar: bool = False) -> list:
    """🎲 The line's takes on this machine: [{"file" (its MP3), "clip" (her video)}], the line's own first, then
    "<line> take 2", "<line> (3)" … in number order (edge_mode.take_stem). A take counts when its MP3 is here, or —
    with_avatar — her clip of it."""
    stem = Path(str(line.get("file") or line["name"])).stem
    own = {"file": str(line_file(line)) if line_file(line) else None,
           "clip": str(avatar_line_clip(line)) if with_avatar and avatar_line_clip(line) else None}
    found = {}
    for folder, kind, exts in ((lines_dir(), "file", E.VOICE_EXT), (avatar_dir() / "lines", "clip", E.VIDEO_EXT)):
        if kind == "clip" and not with_avatar or not folder.is_dir():
            continue
        for f in folder.iterdir():
            n = E.take_stem(f.stem, stem) if f.suffix.lower() in exts and f.stat().st_size > 0 else None
            if n:
                found.setdefault(n, {"file": None, "clip": None})[kind] = str(f)
    return [own] + [found[n] for n in sorted(found)]


def avatar_line_clip(line: dict) -> Path | None:
    """The avatar's video for a line (her voice in it), on this machine: <data>/edge_avatar/lines/<line>.mp4."""
    f = avatar_dir() / "lines" / (Path(str(line.get("file") or line["name"])).stem + ".mp4")
    return f if f.exists() and f.stat().st_size > 0 else None


def line_durations(cue_map: dict, with_avatar: bool = False) -> dict:
    """{line name: measured seconds} for the lines whose files are here (the rest keep the map's estimate).
    with_avatar: a line with an avatar clip lasts as long as the clip (it is often a little longer than the MP3)."""
    from edge_render import probe
    out = {}
    for line in cue_map.get("lines") or []:
        secs = 0.0
        for take in line_takes(line, with_avatar):         # 🎲 the longest take: whichever plays, it fits its moment
            for f in (take["file"], take["clip"]):
                if f:
                    secs = max(secs, probe(f)["seconds"])
        if secs > 0:
            out[line["name"]] = secs
    return out


def fetch_lines(cue_map: dict, progress=None, avatar: bool = True, only_missing: bool = False) -> list:
    """Download the cue map's lines from Google Drive (its "drive_folder") to lines_dir(), and — avatar — her clips
    from its avatar_clips folder (same names, .mp4) to <data>/edge_avatar/lines/. only_missing: leave what is here
    alone. Returns error strings."""
    try:
        from drive_sync import credentials_available, download_source_video
        if not credentials_available():
            return ["Google Drive isn't set up on this machine."]
    except ImportError:
        return ["Google Drive isn't set up on this machine."]
    from drive_sync import list_drive_folder
    folder = str(cue_map.get("drive_folder") or "scene-labeling/Edge_voiceover").strip("/")
    lines = cue_map.get("lines") or []
    errors = []
    stems = [Path(str(ln.get("file") or ln["name"])).stem for ln in lines]

    def is_take(name: str) -> bool:
        return any(E.take_stem(Path(name).stem, st_) for st_ in stems)
    on_drive, err = list_drive_folder(folder)
    on_drive = [] if err else on_drive
    if any(f["name"] == "edge_cues.json" for f in on_drive):          # 🗺️ a cue map exported to Drive replaces the app's copy
        voice_dir().mkdir(parents=True, exist_ok=True)
        err = download_source_video(f"{folder}/edge_cues.json", voice_dir() / "edge_cues.json")
        if err:
            errors.append(f"edge_cues.json: {err}")
    takes = [f["name"] for f in on_drive if Path(f["name"]).suffix.lower() in E.VOICE_EXT and is_take(f["name"])]
    names = [str(ln.get("file") or (ln["name"] + ".mp3")) for ln in lines]
    for k, name in enumerate(names + takes, 1):            # the lines, then 🎲 their other takes
        dest = lines_dir() / name
        if only_missing and dest.exists() and dest.stat().st_size > 0:
            continue
        if progress:
            progress(k, len(names) + len(takes), name)
        err = download_source_video(f"{folder}/{name}", dest)
        if err:
            errors.append(f"{name}: {err}")
    if avatar:                                             # 👩 her clips: whichever of the lines (and takes) have one on Drive so far
        files, err = list_drive_folder(f"{folder}/avatar_clips")
        if err:
            errors.append(f"avatar_clips: {err}")
        want = {st_.lower(): st_ for st_ in stems}
        have = [f for f in files if Path(f["name"]).suffix.lower() in E.VIDEO_EXT
                and (Path(f["name"]).stem.lower() in want or is_take(f["name"]))]
        for k, f in enumerate(have, 1):
            stem = Path(f["name"]).stem
            dest = avatar_dir() / "lines" / (want.get(stem.lower(), stem) + ".mp4")      # named as the line's own file
            if only_missing and dest.exists() and dest.stat().st_size > 0:
                continue
            if progress:
                progress(k, len(have), f["name"])
            err = download_source_video(f"{folder}/avatar_clips/{f['name']}", dest)
            if err:
                errors.append(f"{f['name']}: {err}")
    return errors


def avatar_dir() -> Path:
    """Where the avatar's clips live: <data>/edge_avatar/intro/ and <data>/edge_avatar/<phase>/."""
    return Path(CATALOGUE_DIR).parent / "edge_avatar"


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

def _trim_entry(entry: dict, new_end: float) -> dict:
    """A timeline entry cut short at new_end (the compilation's time): each slot's chain keeps only what fits."""
    a = float(entry["track_time"][0])
    keep = max(new_end - a, 0.0)
    scenes = []
    for slot in entry.get("scenes") or []:
        left, chain = keep, []
        for link in slot.get("chain") or []:
            if left <= 1e-3:
                break
            d = min(float(link["clip_duration_sec"]), left)
            chain.append({**link, "clip_duration_sec": round(d, 3)})
            left -= d
        if chain:
            new = {**slot, "chain": chain}
            if "clip_duration_sec" in slot:
                new["clip_duration_sec"] = round(min(float(slot["clip_duration_sec"]), keep), 3)
            scenes.append(new)
    return {**entry, "track_time": [round(a, 3), round(new_end, 3)], "scenes": scenes}


def _place(timeline: list, offset: float, play: float, part: int, track_id: str) -> list:
    """One song's timeline moved to where the song plays in the compilation, and cut where the song is cut."""
    out = []
    for e in timeline:
        a, b = float(e["track_time"][0]), float(e["track_time"][1])
        if a >= play - 0.05:
            break
        moved = {**e, "track_time": [round(a + offset, 3), round(b + offset, 3)], "edge_part": part,
                 "edge_track_id": track_id}
        if b > play + 1e-6:
            moved = _trim_entry(moved, offset + play)
        out.append(moved)
    return out


def _slice(timeline: list, a: float, b: float, offset: float, part: int, track_id: str) -> list:
    """⚡ The stretch [a, b) of one song's timeline, moved to start at `offset` in the compilation: entries cut at both
    ends (the clips that run into the stretch start later in their footage)."""
    out = []
    for e in timeline:
        ea, eb = float(e["track_time"][0]), float(e["track_time"][1])
        if eb <= a + 1e-3 or ea >= b - 1e-3:
            continue
        cut = max(a - ea, 0.0)
        scenes = []
        for slot in e.get("scenes") or []:
            left, chain = cut, []
            for link in slot.get("chain") or []:
                d = float(link["clip_duration_sec"])
                if left >= d - 1e-3:
                    left -= d
                    continue
                chain.append({**link, "clip_start_sec": round(float(link["clip_start_sec"]) + left, 3),
                              "clip_duration_sec": round(d - left, 3)})
                left = 0.0
            if chain:
                new = {**slot, "chain": chain}
                if "clip_duration_sec" in slot:
                    new["clip_duration_sec"] = round(max(float(slot["clip_duration_sec"]) - cut, 0.0), 3)
                scenes.append(new)
        moved = {**e, "scenes": scenes, "track_time": [round(max(ea, a) - a + offset, 3), round(eb - a + offset, 3)],
                 "edge_part": part, "edge_track_id": track_id}
        if eb > b + 1e-6:
            moved = _trim_entry(moved, offset + (b - a))
        out.append(moved)
    if out:                                                # the stretch fills its phase exactly, edge to edge
        out[0]["track_time"][0] = round(offset, 3)
        if abs(out[-1]["track_time"][1] - (offset + b - a)) < 0.25:
            out[-1]["track_time"][1] = round(offset + b - a, 3)
    return out


PHASE_ROLES = {"wait": "intro", "stop": "outro"}       # 🎭 the footage Wait and Stop favour (Daryl, 15 Nov)


def _phase_role(sched: list, t: float):
    """🎭 The scene role the footage at compilation time t should have: intro clips in Wait, outro clips in Stop
    (the music drives the rest)."""
    ph = E.phase_at(sched, t)
    return PHASE_ROLES.get(ph["profile"]) if ph else None


def _jump_fit(tracks: dict):
    """⚡ jump_fit for the section picker: how alike the bars either side of a jump within a song sound — the
    🎚️ Mix tab's bar features (timbre, harmony, loudness), where the song's audio is on this machine (else
    unknown, and the picker goes by energy alone)."""
    sims = {}

    def fit(tid, end_t, start_t):
        if tid not in sims:
            sims[tid] = None
            try:
                import mixer
                from core import mixing as MX
                t = tracks[tid]
                if mixer.local_path(t).exists():
                    g = MX.grid(t)
                    sims[tid] = (MX.bar_similarity(mixer.bar_features(t, g)), g)
            except Exception:          # noqa: BLE001
                sims[tid] = None
        if sims[tid] is None:
            return None
        from core import mixing as MX
        S, g = sims[tid]
        return MX.join_quality(S, g, end_t, start_t)
    return fit


def compile_edge(track_ids: list, video_ids: list, difficulty: str, rounds: int, *, voiceover: bool = False,
                 beat: bool = False, seed: int = 0, corner: str = "top-left", max_videos: int = 0,
                 avatar: bool = False, avatar_mode: str = "corner", avatar_corner: str = "bottom-right",
                 intro: bool = False, signals: bool = True, round_badge: bool = True, music_fade: bool = True,
                 duck: bool = False, song_per_round: bool = False, energy_mix: bool = False, progress=None) -> dict:
    """The Edge plan. track_ids: the playlist in playing order ([] = no song: the generated beat is the track, and
    the beat sound is on whatever `beat` says). The footage is matched 🎯 Movement only — each song on the 📐 lines
    from its music; a song's later plays (and later songs) leave out the clips already used, as long as the library
    can still fill the time without them. avatar: she appears at the start of every phase (her clip for that phase,
    voice included; the stand-in figure until there is one) — in a corner or as half a split screen. intro: the
    intro screen before the compilation (her intro clip beside the settings and thumbnails).
    signals: 🔔 a bell, a flash and the ring in the phase's colour at every phase change (the timer pulsing over
    its last seconds); round_badge: ▸ "ROUND 2 / 3" by the timer; music_fade: 🌙 the songs fade out in Stop and back
    in over Wait; duck: 🔉 with the voiceover / avatar, the songs at full volume, dipping only under a line (off since
    15 Nov — Daryl wants a steady background level under her: a quarter volume throughout);
    song_per_round: 🎵 round r plays song r of the list (from its start), instead of the songs back to back.
    energy_mix: ⚡ the phases in pairs (Wait + Slow, Medium + Fast, Extra + Stop) each get the stretch of a song whose
    energy fits them best (edge_energy.py) — from the round's own song with song_per_round, else from any song in
    the list — joined by the 🎚️ Mix engine (🧠 stem blends ending on the phase change); the footage follows.
    Without a song the beat is matched round by round, each round leaving out the clips earlier rounds used
    (while the library can still fill it). progress(done, total, text)."""
    import core.story_matcher as SMX
    # 👩 the avatar brings the lines with her: she follows the cue map (her clips, their own sound; the line's
    # recording under a stand-in where she has no clip yet) whether or not the voiceover switch is on (Daryl, 14 Nov)
    voiceover = bool(voiceover or avatar)
    sched = E.schedule(difficulty, rounds, seed)
    total = E.total_seconds(sched)
    track_ids = list(track_ids or [])
    beat_on = bool(beat) or not track_ids
    notes, timeline, parts, shortfall = [], [], [], 0
    base = None
    if track_ids and energy_mix:                           # ⚡ a section of a song for every phase
        import edge_energy as EE
        tracks_ = {tid: PC._load_track(tid) for tid in dict.fromkeys(track_ids)}
        maps = {tid: EE.energy_map(t) for tid, t in tracks_.items()}
        allowed = (lambda ph: [track_ids[(ph["round"] - 1) % len(track_ids)]]) if song_per_round else None
        parts = EE.pick_sections(sched, maps, allowed, jump_fit=_jump_fit(tracks_))
        order = list(dict.fromkeys(p["track_id"] for p in parts))
        used, matched = set(), {}
        for k, tid in enumerate(order):                    # each song matched once, later songs leaving out earlier clips
            def prog(d, t, _k=k, _tid=tid):
                if progress:
                    progress(_k + d / max(t, 1), len(order), f"song {_k + 1} of {len(order)} · {_tid} · block {d} of {t}")
            stretches = [(p["bar"], p["play"], p["offset"]) for p in parts if p["track_id"] == tid]

            def role(a, b, _st=stretches):             # 🎭 song time → where it plays in the compilation → its phase
                mid = (a + b) / 2
                hit = next(((bar, off) for bar, play, off in _st if bar <= mid < bar + play), None)
                return _phase_role(sched, hit[1] + mid - hit[0]) if hit else None
            kw = dict(movement_only=True, cuts="lines", max_videos=int(max_videos or 0), progress=prog, block_role=role)
            res = PC.compile_plan(tid, video_ids, excluded=set(used), **kw)
            if used and (res.get("shortfall") or float(res.get("unfilled") or 0.0) > 0.5):
                again = PC.compile_plan(tid, video_ids, **kw)
                if float(again.get("unfilled") or 0.0) < float(res.get("unfilled") or 0.0):
                    res = again
            base = base or res
            matched[tid] = res
            shortfall += int(res.get("shortfall") or 0)
            notes += [n for n in res.get("notes") or [] if n not in notes]
            used |= {(link["video_id"], link["scene_id"]) for e in res["plan"]["timeline"] for s in e["scenes"]
                     for link in s["chain"]}
        for k, p in enumerate(parts):                      # the footage follows the stretch, pair for pair
            timeline += _slice(matched[p["track_id"]]["plan"]["timeline"], p["bar"], p["bar"] + p["play"],
                               p["offset"], k, p["track_id"])
        fit = sum(abs(p["energy"] - p["target"]) for p in parts) / max(len(parts), 1)
        joins = sum(1 for p in parts if not p["first"] and not p["carry"])
        import mixer
        neural = sum(1 for p in parts if not p["first"] and not p["carry"] and mixer.has_four_stems(tracks_[p["track_id"]]))
        no_stems = sorted(tid for tid in order if not mixer.has_four_stems(tracks_[tid]))
        notes.append(f"⚡ Energy mix: {len(parts)} stretches from {len(order)} song{'s' if len(order) != 1 else ''} "
                     f"(phases in pairs), {joins} join{'s' if joins != 1 else ''} — {neural} 🧠 stem blend{'s' if neural != 1 else ''}"
                     + (f", the rest crossfades with a bass swap (no drum / bass / other stems yet: {', '.join(no_stems)})"
                        if no_stems and joins > neural else "")
                     + f"; energy off its target by {fit:.2f} on average (0–1).")
    elif track_ids:
        durations = [(tid, float(PC._load_track(tid)["duration_sec"])) for tid in track_ids]
        parts = E.round_playlist_parts(durations, sched) if song_per_round else E.playlist_parts(durations, total)
        used = set()
        for k, part in enumerate(parts):
            def prog(d, t, _k=k, _tid=part["track_id"]):
                if progress:
                    progress(_k + d / max(t, 1), len(parts), f"song {_k + 1} of {len(parts)} · {_tid} · block {d} of {t}")
            kw = dict(movement_only=True, cuts="lines", max_videos=int(max_videos or 0), progress=prog,
                      block_role=lambda a, b, _o=part["offset"], _p=part["play"]: (
                          _phase_role(sched, _o + (a + b) / 2) if (a + b) / 2 < _p else None))
            res = PC.compile_plan(part["track_id"], video_ids, excluded=set(used), **kw)
            if used and (res.get("shortfall") or float(res.get("unfilled") or 0.0) > 0.5):
                again = PC.compile_plan(part["track_id"], video_ids, **kw)        # not enough fresh footage left
                if float(again.get("unfilled") or 0.0) < float(res.get("unfilled") or 0.0) or (
                        res.get("shortfall") and not again.get("shortfall")):
                    res = again
                    notes.append(f"{part['track_id']} (play {k + 1}): not enough unused footage left — clips from "
                                 "earlier in the compilation may come round again.")
            base = base or res
            placed = _place(res["plan"]["timeline"], part["offset"], part["play"], k, part["track_id"])
            timeline += placed
            shortfall += int(res.get("shortfall") or 0)
            used |= {(link["video_id"], link["scene_id"]) for e in placed for s in e["scenes"] for link in s["chain"]}
            notes += [n for n in res.get("notes") or [] if n not in notes]
        gap = total - (parts[-1]["offset"] + parts[-1]["play"]) if parts else 0.0
        if gap > 1e-3 and timeline and timeline[-1]["scenes"]:      # under half a second left over: the last clip runs on
            last = timeline[-1]
            for slot in last["scenes"]:
                slot["chain"][-1] = {**slot["chain"][-1],
                                     "clip_duration_sec": round(float(slot["chain"][-1]["clip_duration_sec"]) + gap, 3)}
                if "clip_duration_sec" in slot:
                    slot["clip_duration_sec"] = round(float(slot["clip_duration_sec"]) + gap, 3)
            last["track_time"] = [last["track_time"][0], round(total, 3)]
    else:                                                  # 🥁 the beat, a round at a time: no clip comes round again
        rounds_ = sorted({ph["round"] for ph in sched})
        used = set()
        for k, r in enumerate(rounds_):
            phases = [ph for ph in sched if ph["round"] == r]
            sub = E.shift_schedule(phases)

            def prog(d, t, _k=k):
                if progress:
                    progress(_k + d / max(t, 1), len(rounds_), f"round {_k + 1} of {len(rounds_)} · block {d} of {t}")
            kw = dict(movement_only=True, max_videos=int(max_videos or 0), given_track=E.beat_track(sub),
                      given_segments=E.beat_segments(sub), progress=prog,
                      block_role=lambda a, b, _sub=sub: _phase_role(_sub, (a + b) / 2),
                      lab={"stem": False, "phrase": False})       # a beat has no lead layer or phrases to follow
            res = PC.compile_plan(E.TRACK_ID, video_ids, excluded=set(used), **kw)
            if used and (res.get("shortfall") or float(res.get("unfilled") or 0.0) > 0.5):
                again = PC.compile_plan(E.TRACK_ID, video_ids, **kw)          # not enough fresh footage left
                if float(again.get("unfilled") or 0.0) < float(res.get("unfilled") or 0.0) or (
                        res.get("shortfall") and not again.get("shortfall")):
                    res = again
                    notes.append(f"Round {r}: not enough unused footage left — clips from earlier rounds may come round again.")
            base = base or res
            a, b = float(phases[0]["start"]), float(phases[-1]["end"])
            placed = [{**e, "edge_track_id": None} for e in _place(res["plan"]["timeline"], a, b - a, k, None)]
            timeline += placed
            shortfall += int(res.get("shortfall") or 0)
            used |= {(link["video_id"], link["scene_id"]) for e in placed for s in e["scenes"] for link in s["chain"]}
            notes += [n for n in res.get("notes") or [] if n not in notes]
    for k, e in enumerate(timeline):
        e["segment_index"] = k
        ph = E.phase_at(sched, (e["track_time"][0] + e["track_time"][1]) / 2)
        e["edge_profile"] = ph["profile"] if ph else None
    # 🎙️ the voiceover: Daryl's cue map (edge_cues.json — which line plays when, never two at once); without a cue
    # map, a clip from each phase's folder at the start of every phase, as at first
    cue_map = load_cue_map() if voiceover else {"lines": []}
    lines_used, clashes, intro_lines, outro_lines, lines_missing = bool(cue_map.get("lines")), [], [], [], []
    lines_unused, lines_no_cue = [], []
    line_avatar = None
    if lines_used:
        by_name = {ln["name"]: ln for ln in cue_map["lines"]}
        durations = line_durations(cue_map, with_avatar=bool(avatar))
        got = E.resolve_lines(sched, cue_map, difficulty, seed, durations)
        path = lambda n: (str(line_file(by_name[n])) if line_file(by_name[n]) else None)        # noqa: E731
        # 🎲 a line recorded more than once: its takes take turns (her clip of the same take, when there is one)
        takes = {n: line_takes(ln, bool(avatar)) for n, ln in by_name.items()}
        picks = E.pick_takes(got["plays"], {n: t for n, t in takes.items() if len(t) > 1}, seed)
        take_of = lambda p, k: takes[p["line"]][k] if k < len(takes[p["line"]]) else takes[p["line"]][0]      # noqa: E731
        cues = []
        for p, k in zip(got["plays"], picks):
            tk = take_of(p, k)
            cues.append({"time": p["time"], "profile": p["profile"], "file": tk["file"] or (path(p["line"]) if k == 0 else None),
                         "line": p["line"], "take": k + 1, "clip": tk["clip"], "seconds": p["seconds"], "round": p["round"],
                         "why": p["why"], "moved": p["moved"]})
        clashes = got["clashes"]
        lines_unused = list(got.get("unused") or [])       # a cue, but for a round / difficulty this compilation doesn't have
        lines_no_cue = sorted((ln["name"] for ln in cue_map["lines"] if not ln.get("cues")), key=str.lower)
        named = lambda ns: [{"line": n, "file": path(n), "seconds": float(durations.get(n) or by_name[n].get("seconds") or 2.0)} for n in ns]   # noqa: E731
        intro_lines, outro_lines = named(got["intro"]), named(got["outro"])
        if avatar:                                         # 👩 she reads the lines: her clip where there is one
            vid = lambda n: (str(avatar_line_clip(by_name[n])) if avatar_line_clip(by_name[n]) else None)       # noqa: E731
            for x in intro_lines + outro_lines:
                x["clip"] = vid(x["line"])
            line_avatar = [{"time": c["time"], "profile": c["profile"],
                            "clip": c.get("clip") or (vid(c["line"]) if c.get("take", 1) == 1 else None),
                            "line": c["line"], "seconds": c["seconds"]} for c in cues]
        lines_missing = sorted({c["line"] for c in cues + (intro_lines if intro else []) + outro_lines if not c["file"]}, key=str.lower)
        missing_voice = []
    else:
        files = E.voice_files(voice_dir()) if voiceover else {}
        cues = E.voice_cues(sched, files, seed) if voiceover else []
        missing_voice = sorted({c["profile"] for c in cues if not c["file"]}, key=E.ORDER.index)
    av_files = E.clip_files(avatar_dir()) if (avatar or intro) else {}
    if avatar and line_avatar is not None:                 # with the cue map she appears with every line
        av_cues = line_avatar
        missing_avatar = sorted({c["line"] for c in av_cues if not c["clip"]}, key=str.lower)
    else:
        av_cues = E.avatar_cues(sched, av_files, seed) if avatar else []
        missing_avatar = sorted({c["profile"] for c in av_cues if not c["clip"]}, key=E.ORDER.index)
    intro_clips = av_files.get("intro") or []
    seen, thumbs = {}, []
    for e in timeline:                                    # the intro's thumbnails: the six most used videos
        for s_ in e["scenes"]:
            for link in s_["chain"]:
                first = seen.setdefault(link["video_id"], [0.0, float(link["clip_start_sec"])])
                first[0] += float(link["clip_duration_sec"])
    thumbs = [[vid, t] for vid, (_secs, t) in sorted(seen.items(), key=lambda kv: -kv[1][0])[:6]]
    name = f"edge_{difficulty}_{len(sched) and sched[-1]['round']}r" + ("_" + track_ids[0] if track_ids else "_beat")
    plan = {
        **base["plan"],
        "track_id": name, "matching_mode": "Edge", "track_source_path": base["plan"].get("track_source_path", ""),
        "track_duration_sec": round(total, 3), "bpm": base["plan"].get("bpm"),
        "selected_videos": list(video_ids), "timeline": timeline,
        # ⏱️ everything the timer adds — the app's render builds it (edge_render.py); Blender can read it from here
        "edge": {
            "difficulty": difficulty, "rounds": int(sched[-1]["round"]) if sched else 0, "seed": int(seed),
            "round_seconds": E.round_seconds(difficulty), "total_seconds": round(total, 3),
            "profiles": {p: {k: v for k, v in E.PROFILES[p].items() if k != "colour"} | {"colour": list(E.PROFILES[p]["colour"])}
                         for p in E.ORDER},
            "schedule": sched,
            "playlist": [{**p, "source_path": PC._load_track(p["track_id"]).get("source_path", "")} for p in parts],
            "beat": {"on": beat_on, "volume": E.BEAT_UNDER_SONG if track_ids else 1.0, "times": E.beat_times(sched)},
            "voiceover": {"on": bool(voiceover), "music_volume": E.MUSIC_UNDER_VOICE, "cues": cues,
                          "placeholders": missing_voice, "cue_map": lines_used, "clashes": clashes,
                          "lines_missing": lines_missing},
            "overlay": {"corner": corner, "fps": E.OVERLAY_FPS},
            # 🌙 the songs fade out in Stop / back in over Wait; 🔉 full volume, dipping under each line
            "music": {"still_volume": E.MUSIC_IN_STILL, "fade": E.MUSIC_FADE_SEC, "fade_stop": bool(music_fade),
                      "stop_fade": "whole phase", "duck": bool(duck), "duck_level": E.DUCK_LEVEL,
                      "song_per_round": bool(song_per_round and track_ids),
                      "energy_mix": bool(energy_mix and track_ids)},
            "signals": {"on": bool(signals), "sound": bool(signals), "timer": bool(signals), "flash": bool(signals)},
            "round_badge": {"on": bool(round_badge)},
            "avatar": {"on": bool(avatar), "mode": avatar_mode if avatar_mode in E.AVATAR_MODES else "corner",
                       "corner": avatar_corner, "cues": av_cues, "placeholders": missing_avatar},
            # (the outro screen closes every compilation, whatever is switched on — Daryl, 14 Nov; its lines when there are any)
            "outro": {"on": True, "lines": outro_lines, "title": "EDGE MODE", "figure": bool(avatar),
                      "stats": [["Difficulty", E.DIFFICULTIES[difficulty].split(" ", 1)[-1]],
                                ["Rounds", str(int(sched[-1]["round"]) if sched else 0)], ["Time", format_mmss(total)]],
                      "thumbs": thumbs},
            # (the intro screen only when it is switched on — Daryl, 14 Nov; its lines are left out without it)
            "intro": {"on": bool(intro), "lines": intro_lines if intro else [], "clip": str(random.Random(seed).choice(intro_clips)) if intro_clips else None,
                      "stats": [["Difficulty", E.DIFFICULTIES[difficulty].split(" ", 1)[-1]],
                                ["Rounds", str(int(sched[-1]["round"]) if sched else 0)],
                                ["Time", format_mmss(total)]],
                      "thumbs": thumbs},
        },
    }
    return {"plan": plan, "summary": SMX.plan_summary(timeline), "notes": notes, "shortfall": shortfall,
            "schedule": sched, "parts": parts, "video_ids": list(video_ids), "track_ids": track_ids,
            "missing_voice": missing_voice, "voice_clashes": clashes, "lines_missing": lines_missing, "missing_avatar": missing_avatar,
            "lines_unused": lines_unused, "lines_no_cue": lines_no_cue, "made": time.strftime("%H:%M:%S")}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

JOB_KEY = "edge:job"                  # ⏳ the compile / render running on the server (one at a time)


def _settings() -> dict:
    """The setup as it stands — what a background compile is made from."""
    ss = st.session_state
    return {"playlist": list(ss.get("edge_playlist") or []), "difficulty": ss.get("edge_difficulty", "chill"),
            "rounds": int(ss.get("edge_rounds", 3)), "voiceover": bool(ss.get("edge_voice")), "beat": bool(ss.get("edge_beat")),
            "corner": ss.get("edge_corner", "top-left"), "max_videos": int(ss.get("edge_max_videos", 0) or 0),
            "avatar": bool(ss.get("edge_avatar")), "avatar_mode": ss.get("edge_avatar_mode", "corner"),
            "avatar_corner": ss.get("edge_avatar_corner", "bottom-right"), "intro": bool(ss.get("edge_intro")),
            "song_per_round": bool(ss.get("edge_song_per_round", False)), "energy_mix": bool(ss.get("edge_energy_mix", False)),
            "stream": bool(ss.get("edge_stream", True)), "quick": bool(ss.get("edge_quick", True)),
            "label": " ".join(str(ss.get("edge_render_name") or "").split())[:80]}


def _job(job, settings: dict, video_ids: list, seed: int, render: bool, token, res: dict = None) -> dict:
    """⏳ The background work: compile (unless `res` is given — a render of a plan already compiled), then render
    when asked. Runs in a server thread (mix_jobs), so it finishes even when the phone leaves the page."""
    out = {"notes_first": []}
    if res is None:
        job.progress(0.0, "getting the voiceover lines")
        if settings["voiceover"] or settings["avatar"]:     # the lines (and her clips) not here yet are fetched first
            cm = load_cue_map()
            if cm.get("lines"):
                errs = fetch_lines(cm, lambda k, n, name: job.progress(None, f"fetching {name}"),
                                   avatar=settings["avatar"], only_missing=True)
                if errs:
                    out["notes_first"].append(f"{len(errs)} file(s) couldn't be fetched from Google Drive — those lines play as "
                                              "placeholders. " + " · ".join(errs[:3]))
        share = 0.4 if render else 1.0
        res = compile_edge(settings["playlist"], video_ids, settings["difficulty"], settings["rounds"],
                           voiceover=settings["voiceover"], beat=settings["beat"], seed=seed, corner=settings["corner"],
                           max_videos=settings["max_videos"], avatar=settings["avatar"], avatar_mode=settings["avatar_mode"],
                           avatar_corner=settings["avatar_corner"], intro=settings["intro"],
                           song_per_round=settings["song_per_round"], energy_mix=settings["energy_mix"],
                           progress=lambda d, t, txt: job.progress(share * min(d / max(t, 1), 1.0), f"compiling · {txt}"))
        res["notes"] = out["notes_first"] + list(res.get("notes") or [])
    out["res"] = res
    if render:
        base = job.frac
        out["note"], out["render"] = _render(res, settings, token,
                                             lambda f, txt: job.progress(base + (1 - base) * min(f, 1.0), txt))
    return out


def _start(video_ids: list, render: bool, new_seed: bool = True, res: dict = None) -> None:
    """Start the compile / render on the server (on_click: the session is still at hand for the setup and the Drive
    connection)."""
    import mix_jobs as J
    from page_setup import drive_token
    if res is None and not video_ids:
        st.session_state["edge_msg"] = ("warning", "No videos to match — widen the video filters.")
        return
    if new_seed or "edge_seed" not in st.session_state:
        st.session_state["edge_seed"] = random.randrange(1, 10 ** 6)       # Hard draws a new order each compile
    running = J.get(JOB_KEY)
    if running is not None and running.active:
        st.session_state["edge_msg"] = ("warning", "A compile / render is already running — it is shown below.")
        return
    J.forget(JOB_KEY)
    label = "🎬 Rendering" if res is not None else ("⏱️ Compiling and rendering" if render else "⏱️ Compiling")
    J.start(JOB_KEY, label, _job, _settings(), list(video_ids), int(st.session_state["edge_seed"]), render,
            drive_token(), res)
    st.session_state["edge_job_seen"] = None


def _take_finished_job() -> bool:
    """A finished background job's result into this session (once) — also in a new session after the page was
    reloaded. True while one is still running."""
    import mix_jobs as J
    j = J.get(JOB_KEY)
    if j is None:
        return False
    if j.active:
        return True
    ss = st.session_state
    if ss.get("edge_job_seen") == j.started:
        return False
    ss["edge_job_seen"] = j.started
    if j.state == "error":
        ss["edge_msg"] = ("error", f"{j.label} failed: {j.error}")
        return False
    got = j.result or {}
    if got.get("res") is not None:
        ss[RESULT_KEY] = got["res"]
    if got.get("render"):
        ss["edge_render"] = got["render"]
        ss["edge_msg"] = got.get("note")
    elif got.get("res") is not None:
        ss.pop("edge_render", None)
    return False


@st.fragment(run_every=1.0)
def _watch() -> None:
    """⏳ The running job's progress, checked every second; the page refreshes when it is done."""
    import mix_jobs as J
    j = J.get(JOB_KEY)
    if j is None or not j.active:
        st.rerun(scope="app")
    secs = int(time.time() - j.started)
    st.progress(min(1.0, j.frac), text=f"{j.label}… {j.msg or ''} · {secs // 60}:{secs % 60:02d}")
    st.caption("⏳ This runs on the server — you can leave the page or switch apps and come back; the result will be here.")


def _lines_panel(cue_map: dict) -> None:
    """🎙️ The cue map's lines: how many are on this machine, and a button to fetch them from Google Drive."""
    lines = cue_map["lines"]
    here = sum(1 for ln in lines if line_file(ln) is not None)
    mapped = sum(1 for ln in lines if ln.get("cues"))
    with st.expander(f"🎙️ Voiceover lines — {mapped} of {len(lines)} have a cue · {here} of {len(lines)} are on the server",
                     expanded=here < len(lines)):
        st.caption("Which line plays when comes from your cue map; no two lines ever play at once. A line whose file "
                   "isn't here yet plays as a placeholder tone. The button also brings the avatar's clips made so far "
                   f"({sum(1 for ln in lines if avatar_line_clip(ln) is not None)} here); Compile fetches whatever is missing by itself.")
        if st.button("⬇️ Get the lines from Google Drive", key="edge_lines_fetch", type="primary" if here < len(lines) else "secondary"):
            bar = st.progress(0.0, text="Fetching the lines…")
            errs = fetch_lines(cue_map, lambda k, n, name: bar.progress(k / max(n, 1), text=f"Fetching {name} ({k}/{n})…"))
            bar.empty()
            if errs:
                st.session_state["edge_msg"] = ("warning", f"{len(errs)} line(s) couldn't be fetched: " + " · ".join(errs[:4])
                                                + (" …" if len(errs) > 4 else ""))
            else:
                st.session_state["edge_msg"] = ("success", f"All {len(lines)} lines are on the server.")
            st.rerun()
        missing = [ln["name"] for ln in lines if line_file(ln) is None]
        if missing and here:
            st.caption("Not here yet: " + " · ".join(missing))
        extra = {ln["name"]: len(line_takes(ln, True)) - 1 for ln in lines}
        if any(extra.values()):
            st.caption("🎲 Other takes (they take turns): " + " · ".join(f"{n} +{k}" for n, k in extra.items() if k))
        st.caption("🎲 More than one take of a line: name the file \"<line> take 2.mp3\" (or \"(2)\") beside it on "
                   "Drive, her clip the same name in avatar_clips.")
        st.caption(f"🗺️ Cue map in use: {'yours (uploaded / from Drive)' if cue_map_path() != E.CUE_MAP_FILE else 'the one in the app'}"
                   " — export it from the Cue Map page and drop it here, or put edge_cues.json in the Edge_voiceover "
                   "folder on Drive (Compile picks it up).")
        n = st.session_state.get("edge_cue_up_n", 0)
        up = st.file_uploader("🗺️ Replace the cue map", type=["json"], key=f"edge_cue_up_{n}")
        if up is not None:
            err = save_cue_map(up.getvalue())
            st.session_state["edge_cue_up_n"] = n + 1
            st.session_state["edge_msg"] = ("error", err) if err else ("success", "Cue map replaced — the next compile uses it.")
            st.rerun()


def _clip_manager(kind: str) -> None:
    """Add / remove the clips of the 🎙️ voiceover (kind "voice") or the 👩 avatar (kind "avatar")."""
    if kind == "voice":
        folder, slots, exts, files = voice_dir(), E.ORDER, E.VOICE_EXT, E.voice_files(voice_dir())
        title, empty = "🎙️ Voiceover clips", "placeholder tone"
        tip = ("A clip plays at the start of each phase, picked from that phase's clips in turn. A phase with no clip "
               "yet gets a short placeholder tone, so the timing can be checked now.")
    else:
        folder, slots, exts, files = avatar_dir(), E.AVATAR_SLOTS, E.VIDEO_EXT, E.clip_files(avatar_dir())
        title, empty = "👩 Avatar clips", "stand-in figure"
        tip = ("Her videos, voice included: Intro for the intro screen, and one or more for each phase (she appears at "
               "the start of the phase, the clips taking turns). Until a clip is added a stand-in figure shows where "
               "and when she will be.")
    have = sum(len(v) for v in files.values())
    name = lambda p: "Intro" if p == "intro" else E.PROFILES[p]["label"].title()      # noqa: E731
    with st.expander(f"{title} — {have} added" + ("" if have else " (placeholders until you add yours)")):
        st.caption(tip)
        c = st.columns([1.3, 3, 1])
        slot = c[0].selectbox("For", slots, format_func=name, key=f"edge_{kind}_slot")
        n = st.session_state.get(f"edge_{kind}_n", 0)
        up = c[1].file_uploader("Add clips", type=[x.strip(".") for x in exts], accept_multiple_files=True,
                                key=f"edge_{kind}_up_{n}")
        if c[2].button("➕ Add", key=f"edge_{kind}_add", width="stretch", disabled=not up):
            d = folder / slot
            d.mkdir(parents=True, exist_ok=True)
            for f in up:
                (d / Path(f.name).name).write_bytes(f.getvalue())
            st.session_state[f"edge_{kind}_n"] = n + 1
            st.rerun()
        for p in slots:
            names = files[p]
            st.markdown(f"**{name(p)}** — " + (f"{len(names)} clip{'s' if len(names) != 1 else ''}" if names else empty))
            for f in names:
                r = st.columns([5, 1])
                r[0].caption(f.name)
                if r[1].button("🗑️", key=f"edge_{kind}_del_{p}_{f.name}", help="Remove this clip"):
                    f.unlink(missing_ok=True)
                    st.rerun()


def _setup(rows: dict, have_tl: set) -> None:
    st.caption("⏱️ A compilation run by a countdown timer: rounds of 20-second phases — Wait, Slow, Medium, Fast, "
               "Extra, Stop — shown as an overlay. Pick songs or let a beat follow the timer; the footage is matched "
               "to whichever plays.")
    summaries = PC.audio_summaries(PC._audio_signature())
    PC._still_there("edge_playlist", summaries)
    songs = st.multiselect(
        "🎵 Playlist — optional", sorted(summaries, key=str.lower), key="edge_playlist",
        format_func=lambda t: f"{t} · {format_mmss(summaries[t]['duration'])}"
                              + (f" · {summaries[t]['bpm']:.0f} BPM" if summaries[t]["bpm"] else ""),
        help="Any number of songs, played back to back in the order picked; the list starts again until the rounds "
             "are filled, and the last song is cut where the time ends. None: a beat that follows the timer.")
    c = st.columns([2.2, 1.6, 2])
    diff = c[0].radio("Difficulty", list(E.DIFFICULTIES), key="edge_difficulty", horizontal=True,
                      format_func=lambda d: E.DIFFICULTIES[d])
    rounds = c[1].slider("Rounds", E.MIN_ROUNDS, E.MAX_ROUNDS, key="edge_rounds")
    c[2].selectbox("Timer position", list(E.CORNERS), key="edge_corner", format_func=lambda k: E.CORNERS[k])
    st.caption("⏱️ " + E.describe(diff, rounds))
    st.caption("🔔 Every phase change brings a bell, a flash of the phase's colour and the ring in that colour; the "
               "timer pulses over a phase's last 3 seconds. ▸ The round number sits by the timer. 🎭 Wait favours "
               "clips marked intro, Stop clips marked outro (🎭 Reviewer roles); the music drives the rest.")
    total = E.round_seconds(diff) * rounds
    if songs:
        mc = st.columns(2)
        mc[0].toggle("🎵 New song each round", key="edge_song_per_round",
                     help="Round 1 plays the first song, round 2 the second… (the list starts again when there are "
                          "more rounds than songs). Each starts from its beginning and starts again if it ends before "
                          "the round does. Off: the songs back to back, whatever the rounds.")
        mc[1].toggle("⚡ Match song sections to each phase", key="edge_energy_mix",
                     help="The phases play in pairs — Wait + Slow, Medium + Fast, Extra + Stop — each pair on the "
                          "part of a song whose energy fits it (calm for Wait and Slow, the peak for Extra, Stop "
                          "fading on it). The parts are joined by the 🎚️ Mix engine: a 🧠 stem blend of about 3–4 s "
                          "ending on the phase change — into a livelier part a bass drop, its drums and bass landing "
                          "on the change (parts starting on the song's own drops are preferred) — or a "
                          "crossfade with a bass swap for a song without its drum / bass / other stems. With 🎵 New "
                          "song each round on, the parts come from that round's song (no tempo clash at the joins); "
                          "otherwise from any song in the list, key and tempo weighed. The footage follows each part.")
        st.caption("🌙 The music fades out slowly over each Stop and comes back up over the Wait that follows.")
        if st.session_state.get("edge_energy_mix"):
            st.caption("⚡ " + ("Each round's parts come from its own song." if st.session_state.get("edge_song_per_round")
                               else f"Parts picked from all {len(songs)} song{'s' if len(songs) != 1 else ''} to fit each phase.")
                       + " Compile shows how close the fit is.")
        elif st.session_state.get("edge_song_per_round"):
            rp = E.round_playlist_parts([(t, summaries[t]["duration"]) for t in songs], E.schedule(diff, rounds))
            per = {}
            for p in rp:
                per.setdefault(p["round"], p["track_id"])
            st.caption("🎵 " + " · ".join(f"R{r}: {t}" for r, t in per.items())
                       + (" — a song shorter than its round starts again" if any(p["loop"] > 1 for p in rp) else ""))
        else:
            parts = E.playlist_parts([(t, summaries[t]["duration"]) for t in songs], total)
            passes = max(p["pass"] for p in parts) if parts else 1
            st.caption(f"🎵 {len(parts)} play{'s' if len(parts) != 1 else ''} of {len(songs)} song{'s' if len(songs) != 1 else ''}"
                       + (f" — the list comes round {passes} times" if passes > 1 else "")
                       + f"; the last one stops at {format_mmss(total)}.")
    t = st.columns(2)
    if songs:
        t[0].toggle("🥁 Beat sound over the songs", key="edge_beat",
                    help="Off with a song by default. On: the timer's beat (Slow 60, Medium 100, Fast 140, Extra 180 "
                         "a minute; silent in Wait and Stop) plays over the music. The footage still follows the songs.")
    else:
        t[0].toggle("🥁 Beat sound", value=True, disabled=True, key="edge_beat_shown",
                    help="No song picked, so the beat is the soundtrack: Slow 60, Medium 100, Fast 140, Extra 180 "
                         "beats a minute, silent in Wait and Stop. The footage is matched to it.")
    t[1].toggle("🎙️ Voiceover", key="edge_voice",
                help="A voiceover clip at the start of every phase. The songs drop to a quarter volume (background) so it is heard.")
    if st.session_state.get("edge_voice"):
        cue_map = load_cue_map()
        if cue_map.get("lines"):
            _lines_panel(cue_map)
        else:
            _clip_manager("voice")
    a = st.columns(2)
    a[0].toggle("👩 Avatar", key="edge_avatar",
                help="She appears with every line of your cue map (with or without the voiceover switch) — her clip "
                     "where one has been made (its own sound is heard), a stand-in figure and the line's recording "
                     "for the rest. The songs drop to a quarter "
                     "volume.")
    a[1].toggle("🎬 Intro screen", key="edge_intro",
                help="Before the compilation: the avatar giving her intro, beside the difficulty, rounds, total time "
                     "and thumbnails of the videos used. It lasts as long as her intro clip (8 s with the stand-in).")
    if st.session_state.get("edge_avatar"):
        m = st.columns(2)
        mode = m[0].radio("She appears", list(E.AVATAR_MODES), key="edge_avatar_mode", horizontal=True,
                          format_func=lambda k: E.AVATAR_MODES[k],
                          help="In a corner: a framed box over the footage. Split screen: she takes the right half, "
                               "the footage's middle the left, for as long as her clip runs.")
        if mode == "corner":
            m[1].selectbox("Her corner", [k for k in E.CORNERS if k != "centre"], key="edge_avatar_corner",
                           format_func=lambda k: E.CORNERS[k])
            if st.session_state.get("edge_avatar_corner") == st.session_state.get("edge_corner"):
                st.warning("The avatar and the timer are set to the same corner — she will sit under the timer.")
    if st.session_state.get("edge_avatar") or st.session_state.get("edge_intro"):
        _clip_manager("avatar")
    vids = PC._video_filters("ce_", rows, have_tl,
                             f"🎬 Videos / tags — optional (the best fit from all {len(have_tl)} unless you narrow it)")
    ranges = LC.library_ranges()
    mins = sum((ranges[v][1] - ranges[v][0]) if v in ranges else float(rows[v].get("duration") or 0) for v in vids) / 60
    st.caption(f"🎬 {len(vids)} video{'s' if len(vids) != 1 else ''} · {mins:.0f} min of footage in play for "
               f"{format_mmss(total)} of compilation")
    if mins * 60 < total:
        st.warning("There is less footage in play than the compilation is long — widen the video filters or pick fewer rounds.")
    with st.popover("⚙️ More"):
        st.number_input("At most this many different videos (0 = no limit)", 0, 60, key="edge_max_videos",
                        help="Fewer different videos = fewer files to fetch for a full-quality render.")
    go = st.columns([2, 1])
    go[0].button("⏱️ Compile & render", key="edge_go", type="primary", width="stretch", on_click=_start, args=(vids, True),
                 help="Compiles, then renders straight away — on the server: you can leave the page and come back.")
    go[1].button("Compile only", key="edge_go_only", width="stretch", on_click=_start, args=(vids, False))


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

def _strip(res: dict) -> None:
    import plotly.graph_objects as go
    fig = go.Figure()
    for ph in res["schedule"]:
        prof = E.PROFILES[ph["profile"]]
        fig.add_trace(go.Bar(x=[ph["end"] - ph["start"]], base=[ph["start"]], y=["⏱️ timer"], orientation="h", width=0.8,
                             marker=dict(color="rgb(%d,%d,%d)" % tuple(prof["colour"]), line=dict(color="rgba(255,255,255,.5)", width=0.5)),
                             showlegend=False,
                             hovertemplate=f"Round {ph['round']} · {prof['label'].title()} · {format_mmss(ph['start'])}–"
                                           f"{format_mmss(ph['end'])}<extra></extra>"))
    for p in res["parts"]:
        fig.add_trace(go.Bar(x=[p["play"]], base=[p["offset"]], y=["🎵 songs"], orientation="h", width=0.8,
                             marker=dict(color="rgba(120,120,220,.55)", line=dict(color="rgba(255,255,255,.7)", width=1)),
                             showlegend=False, hovertemplate=f"{p['track_id']} · from {format_mmss(p['offset'])} · "
                                                             f"{format_mmss(p['play'])} played"
                                                             + (f"<br>⚡ its {format_mmss(p['bar'])} · energy {p['energy']:.2f} "
                                                                f"for {p['profile'].title()} (aim {p['target']:.2f})"
                                                                + (" · carries on" if p.get("carry") else "")
                                                                if "energy" in p else "") + "<extra></extra>"))
    for e in res["plan"]["timeline"]:
        a, b = e["track_time"]
        label = " / ".join(s["video_id"] for s in e["scenes"]) or "—"
        fig.add_trace(go.Bar(x=[b - a], base=[a], y=["🎬 clips"], orientation="h", width=0.8,
                             marker=dict(color="rgba(150,150,150,.55)" if e["scenes"] else "rgba(220,60,60,.8)",
                                         line=dict(color="rgba(0,0,0,.3)", width=0.5)),
                             showlegend=False, hovertemplate=f"{format_mmss(a)}–{format_mmss(b)}<br>{label}<extra></extra>"))
    fig.update_layout(barmode="overlay", height=150 if res["parts"] else 115, margin=dict(l=4, r=4, t=4, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(range=[0, float(res["plan"]["track_duration_sec"])], tickfont_size=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"edge_strip_{res['made']}")


def _missing_sources(plan: dict, quick: bool) -> list:
    """(raw source, local path) of every song and video the plan uses that isn't on this machine."""
    from planner.data import load_all_catalogues
    from render_preview import colab_to_local
    cats = load_all_catalogues()
    needed = {p["source_path"] for p in plan["edge"]["playlist"] if p.get("source_path")}
    for e in plan["timeline"]:
        for slot in e["scenes"]:
            for link in slot["chain"]:
                if quick and LC.proxy_path(link["video_id"]).exists():
                    continue
                needed.add(cats[link["video_id"]]["source_path"])
    return [(raw, Path(colab_to_local(raw))) for raw in sorted(needed) if not Path(colab_to_local(raw)).exists()]


def _fetch_missing(missing: list, prog) -> list:
    """Download the plan's files that aren't on the server (no page needed). Returns error strings."""
    try:
        from drive_sync import credentials_available, download_source_video
        ok = credentials_available()
    except ImportError:
        ok = False
    errors = []
    for n, (raw, local) in enumerate(missing, 1):
        if not (ok and "MyDrive/" in raw):
            errors.append(f"{local.name}: not on the server and Drive download isn't available")
            continue
        prog(0.0, f"downloading {local.name} ({n}/{len(missing)})")
        err = download_source_video(raw.split("MyDrive/", 1)[1], local)
        if err:
            errors.append(f"{local.name}: {err}")
    return errors


def _render(res: dict, settings: dict, token, prog) -> tuple:
    """🎬 The compiled plan → a video in the renders folder, uploaded to Drive when connected. No page needed (runs
    in the background job). Returns (message for the page, file name). Raises when it can't."""
    import fnmatch
    import re
    from page_setup import APP_VERSION
    from planner.state import sanitize_filename
    from render_preview import render_plan_dict
    from renders import auto_clear_enabled, clear_local, render_files, update_user_fields, upload_render
    plan = res["plan"]
    stream, quick = settings["stream"], settings["quick"]
    if not stream:
        missing = _missing_sources(plan, quick)
        if missing:
            errs = _fetch_missing(missing, prog)
            if errs:
                raise RuntimeError("some source files are missing and couldn't be downloaded: " + " · ".join(errs[:4]))
    safe = sanitize_filename(plan["track_id"])
    used = [int(m.group(1)) for p in render_files() if (m := re.search(r"\.(\d{3})\.mp4$", p.name))
            and fnmatch.fnmatch(p.name, f"temp-Video-preview-{safe}.*.mp4")]
    Path(PREVIEW_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(PREVIEW_DIR) / f"temp-Video-preview-{safe}.{(max(used) + 1) if used else 1:03d}.mp4"
    last, t0 = [0.0], time.time()

    def _prog(d, t):
        last[0] = d / max(t, 1)
        secs = time.time() - t0
        prog(last[0], f"rendering segment {d}/{t}" + (f" · about {int((secs / d) * (t - d))} s left" if d else ""))

    e = plan["edge"]
    stats = render_plan_dict(plan, str(out), progress_callback=_prog, stream=stream, quick=quick,
                             notice=lambda txt: prog(last[0], txt), app_version=APP_VERSION,
                             extra_meta={"project_name": f"⏱️ Edge · {E.DIFFICULTIES[e['difficulty']]} · {e['rounds']} rounds",
                                         "compile_note": _used_line(res)}) or {}
    if settings.get("label"):
        update_user_fields(out, name=settings["label"])
    secs = time.time() - t0
    took = f"{int(secs // 60)}:{int(secs % 60):02d}"
    note = ("info", f"Rendered {out.name} in {took}. Connect Google Drive (sidebar) to save renders there.")
    if token:
        prog(1.0, "uploading to Google Drive")
        err = upload_render(out, token)
        if err:
            note = ("warning", f"Rendered, but the Drive upload failed: {err}")
        elif auto_clear_enabled():
            clear_local(out)
            note = ("success", f"Rendered {out.name} in {took}, saved to Google Drive and cleared from the server.")
        else:
            note = ("success", f"Rendered {out.name} in {took} and saved to Google Drive.")
    how = [f"{stats[k]} {w}" for k, w in (("quick", "from the 480p copies"), ("streamed", "streamed from Drive"),
                                         ("downloaded", "downloaded"), ("local", "already on the server")) if stats.get(k)]
    if how:
        note = (note[0], note[1] + " Source files: " + ", ".join(how) + ".")
    stg = stats.get("stages") or {}
    if stg.get("ffmpeg"):                                  # which ffmpeg made it (the server's differs from the test machine's)
        note = (note[0], note[1] + " " + " ".join(str(stg["ffmpeg"]).split()[:3]) + ".")
    steps = [f"{w} {int(stg[k] // 60)}:{int(stg[k] % 60):02d}" for k, w in (("segments", "clips"), ("edge_sound", "sound"),
             ("edge_screens", "intro / outro"), ("edge_picture", "timer and avatar"), ("edge_join", "joining")) if stg.get(k)]
    if steps:
        note = (note[0], note[1] + " Time: " + ", ".join(steps) + ".")
    if stg.get("edge_notes"):                              # ⚡ what the Mix engine had to say (e.g. no stems for a song)
        note = (note[0], note[1] + " ⚡ " + " · ".join(stg["edge_notes"][:3]))
    if stg.get("edge_check") not in (None, "ok"):          # the picture or the sound isn't as long as the plan
        note = ("warning", note[1] + f" ⚠️ Length check: {stg['edge_check']}.")
    return note, out.name


def _used_line(res: dict) -> str:
    e = res["plan"]["edge"]
    bits = [f"⏱️ Edge · {E.DIFFICULTIES[e['difficulty']]}", f"{e['rounds']} rounds ({format_mmss(e['total_seconds'])})",
            (f"🎵 {len(res['track_ids'])} song{'s' if len(res['track_ids']) != 1 else ''}, {len(res['parts'])} plays"
             if res["track_ids"] else "🥁 generated beat"),
            f"{len(res['video_ids'])} videos in play"]
    if res["track_ids"] and e["beat"]["on"]:
        bits.append("🥁 beat over the songs")
    if e["voiceover"]["on"]:
        n_lines = len(e["voiceover"]["cues"])
        bits.append((f"🎙️ {n_lines} voiceover lines" + (f", {len(res.get('voice_clashes') or [])} clashes handled" if res.get("voice_clashes") else "")
                     + (f", {len(res.get('lines_missing') or [])} as placeholder tones" if res.get("lines_missing") else ""))
                    if e["voiceover"].get("cue_map") else "🎙️ voiceover" + (" (placeholder tones: " + ", ".join(E.PROFILES[p]["label"].title() for p in res["missing_voice"]) + ")"
                                      if res["missing_voice"] else ""))
    av = e.get("avatar") or {}
    if av.get("on"):
        bits.append(f"👩 avatar ({'split screen' if av['mode'] == 'split' else av['corner'].replace('-', ' ')})"
                    + ((f" — her clip for {len(av['cues']) - sum(1 for c in av['cues'] if not c.get('clip'))} of {len(av['cues'])} lines, stand-in for the rest"
                        if e["voiceover"].get("cue_map") and e["voiceover"].get("on")
                        else " — stand-in figure for: " + ", ".join(E.PROFILES[p]["label"].title() if p in E.PROFILES else p for p in av["placeholders"]))
                       if av.get("placeholders") else ""))
    if (e.get("intro") or {}).get("on"):
        bits.append("🎬 intro screen" + ("" if e["intro"].get("clip") or e["intro"].get("lines") else " (stand-in figure, 8 s)"))
    if (e.get("outro") or {}).get("on"):
        bits.append("🎬 outro screen")
    return " · ".join(bits)


def _plain(x):
    """For the plan's JSON: numpy numbers and flags as plain ones (anything else as text)."""
    import numpy as np
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _save(res: dict) -> None:
    from page_setup import DRIVE_SESSION_KEY, OAUTH_AVAILABLE, push_file_with_oauth
    plan = res["plan"]
    Path(PLANS_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(PLANS_DIR) / f"{plan['track_id']}_plan.json"
    out.write_text(json.dumps(plan, indent=2, default=_plain))
    if OAUTH_AVAILABLE and st.session_state.get(DRIVE_SESSION_KEY):
        err = push_file_with_oauth(st.session_state[DRIVE_SESSION_KEY], out, "scene-labeling/compilation_plans")
        st.session_state["edge_msg"] = ("warning", f"Saved locally but Drive upload failed: {err}") if err else \
            ("success", f"Saved and uploaded to Drive: {out.name}")
    else:
        st.session_state["edge_msg"] = ("success", f"Saved locally: {out.name}")


def _result(res: dict) -> None:
    plan, summary = res["plan"], res["summary"]
    c = st.columns([5, 1.4])
    c[0].subheader("⏱️ Edge Mode")
    c[1].button("← Change", key="edge_back", width="stretch", on_click=lambda: st.session_state.pop(RESULT_KEY, None))
    st.caption(_used_line(res) + f" · made {res['made']}")
    st.caption(f"{len(plan['timeline'])} clips · {summary.get('videos', 0)} different videos used"
               + (f" · 🎯 movement sync {summary['sync']:.0%}" if summary.get("sync") is not None else "")
               + (f" · clips {summary['clip_sec'][0]:g}–{summary['clip_sec'][2]:g} s" if summary.get("clip_sec") else "")
               + (f" · ⚠️ {res['shortfall']} slot(s) unfilled" if res.get("shortfall") else ""))
    for n in res.get("notes") or []:
        st.caption("⏱️ " + n)
    _strip(res)
    vo = plan["edge"]["voiceover"]
    if vo.get("on") and vo.get("cue_map"):
        clashes = res.get("voice_clashes") or []
        with st.expander(f"🎙️ Voiceover — {len(vo['cues'])} lines" + (f" · {len(clashes)} clashes" if clashes else " · no clashes"),
                         expanded=False):
            if clashes:
                st.markdown("**Clashes** — two lines wanted the same moment; they never overlap, so one gave way:")
                for c in clashes:
                    st.caption(f"⚠️ {c['line']} ({c['where']}) met {c['with']} → {c['result']}")
            av_on = (plan["edge"].get("avatar") or {}).get("on")
            if av_on and res.get("missing_avatar"):       # 👩 the lines in this compilation she has no clip for yet
                st.caption(f"👩 No avatar clip on Google Drive yet for {len(res['missing_avatar'])} of this compilation's lines "
                           f"(stand-in figure, the line's own recording is heard): " + ", ".join(res["missing_avatar"]))
            unused, no_cue = res.get("lines_unused") or [], res.get("lines_no_cue") or []
            if unused or no_cue:
                st.caption(f"🔇 Not heard in this compilation: {len(unused)} line(s) whose cue is for a round it doesn't have"
                           + (" (" + ", ".join(unused) + ")" if unused else "")
                           + (f" · {len(no_cue)} with no cue yet (" + ", ".join(no_cue) + ")" if no_cue else ""))
            intro_l, outro_l = plan["edge"]["intro"].get("lines") or [], plan["edge"]["outro"].get("lines") or []
            if intro_l:
                st.caption("🎬 Intro: " + " → ".join(x["line"] for x in intro_l))
            for c in vo["cues"]:
                st.caption(f"{format_mmss(c['time'])} · round {c['round']} · {c['line']} ({c['seconds']:.1f} s) — {c['why']}"
                           + (f" · moved {c['moved']:+.1f} s" if c.get("moved") else "") + ("" if c["file"] else " · placeholder tone"))
            if outro_l:
                st.caption("🎬 Outro: " + " → ".join(x["line"] for x in outro_l))
    st.text_input("🏷️ Name the next render (optional)", key="edge_render_name", max_chars=80)
    b = st.columns([1.8, 0.5, 1.4, 1.4, 1.6])
    b[0].button("🎬 Render", key="edge_render_btn", type="primary", width="stretch",
                on_click=_start, args=(res["video_ids"], True, False, res))
    with b[1].popover("⚙️", width="stretch", help="Render options"):
        st.toggle("⚡ Quick preview — use the 480p copies on the server", key="edge_quick")
        st.toggle("Stream original videos from Google Drive", key="edge_stream")
        missing = _missing_sources(plan, bool(st.session_state.get("edge_quick", True)))
        st.caption(f"{len(missing)} source file{'s' if len(missing) != 1 else ''} not on the server yet"
                   if missing else "Every source file is on the server.")
    b[2].button("💾 Save plan", key="edge_save", width="stretch", on_click=_save, args=(res,))
    b[3].download_button("⬇️ Download", data=json.dumps(plan, indent=2, default=_plain), file_name=f"{plan['track_id']}_plan.json",
                         mime="application/json", width="stretch", key="edge_dl")
    b[4].button("↻ Recompile & render", key="edge_again", width="stretch", on_click=_start, args=(res["video_ids"], True),
                help="Build it again with the current setup and render it — Hard draws a new order of phases.")
    if name := st.session_state.get("edge_render"):
        from library_common import show_render_player
        from renders import load_render
        meta = load_render(Path(PREVIEW_DIR) / name)
        if meta:
            show_render_player(meta)
        else:
            st.caption(f"{name} — see 📁 Media Library → 🎞️ Renders.")


def render() -> None:
    SSTORE.keep(KEPT, KEPT_IF_SET)
    msg = st.session_state.pop("edge_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    if _take_finished_job():                               # ⏳ a compile / render is running on the server
        _watch()
        return
    msg = st.session_state.pop("edge_msg", None)          # (what the finished job left)
    if msg:
        getattr(st, msg[0])(msg[1])
    res = st.session_state.get(RESULT_KEY)
    if res:
        _result(res)
        return
    rows = PC._video_rows()
    have_tl = PC._videos_with_timeline() & set(rows)
    if not have_tl:
        st.warning("No video has a 🧭 footage timeline yet — build it in 📁 Media Library → 🎬 Videos → 🧭 Timeline first.")
        return
    _setup(rows, have_tl)
