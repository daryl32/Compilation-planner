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
import planner.compose as PC
from config import CATALOGUE_DIR
from library_common import format_mmss

RESULT_KEY = "edge_result"
KEPT = {"edge_playlist": [], "edge_rounds": 3, "edge_difficulty": "chill", "edge_voice": False, "edge_beat": False,
        "edge_corner": "top-left", "edge_max_videos": 0, "edge_avatar": False, "edge_avatar_mode": "corner",
        "edge_avatar_corner": "bottom-right", "edge_intro": False, "edge_quick": True, "edge_stream": True, "edge_render_name": "",
        "edge_song_per_round": False, "edge_energy_mix": False, "edge_surprises": False, **{"edge_tags_" + p: [] for p in E.ORDER}}
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


def newest_cue_map(files: list) -> str | None:
    """📤 The newest cue map in the Drive folder's listing ({"name", "modifiedTime"}): the Cue Map page's "Send to the app"
    writes edge_cues_<time>.json (the connector can't overwrite a file); a hand-made edge_cues.json counts too."""
    import re
    got = [f for f in files or [] if re.fullmatch(r"edge_cues(_[^/]*)?\.json", str(f.get("name") or ""), re.I)]
    return max(got, key=lambda f: (str(f.get("modifiedTime") or ""), f["name"]))["name"] if got else None


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


def avatar_clip_folders(folder: str) -> list:
    """👩 Where her speaking clips are, below the voiceover folder: "avatar_clips", then an "avatar_clips" inside any
    of its other folders (e.g. "sitting/avatar_clips" — clips in another position; Daryl, 18 Nov)."""
    out = ["avatar_clips"]
    try:
        from drive_sync import list_drive_subfolders
    except ImportError:
        return out
    subs, err = list_drive_subfolders(folder)
    for sub in ([] if err else subs):
        if sub == "avatar_clips" or sub.startswith("_"):
            continue
        inner, err_ = list_drive_subfolders(f"{folder}/{sub}")
        if not err_ and "avatar_clips" in inner:
            out.append(f"{sub}/avatar_clips")
    return out


VOICE_SUBFOLDERS = ("New",)                            # 📁 subfolders of Edge_voiceover whose lines the cue map may use


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
    errors = []
    on_drive, err = list_drive_folder(folder)
    on_drive = [] if err else on_drive
    sent = newest_cue_map(on_drive)                       # 🗺️ the mapping last sent from the Cue Map wins over the app's copy
    if sent:
        voice_dir().mkdir(parents=True, exist_ok=True)
        err = download_source_video(f"{folder}/{sent}", voice_dir() / "edge_cues.json")
        if err:
            errors.append(f"{sent}: {err}")
        else:
            cue_map = load_cue_map()                       # its lines are the ones to fetch (new lines included)
    lines = cue_map.get("lines") or []
    stems = [Path(str(ln.get("file") or ln["name"])).stem for ln in lines]

    def is_take(name: str) -> bool:
        return any(E.take_stem(Path(name).stem, st_) for st_ in stems)
    takes = [f["name"] for f in on_drive if Path(f["name"]).suffix.lower() in E.VOICE_EXT and is_take(f["name"])]
    names = [str(ln.get("file") or (ln["name"] + ".mp3")) for ln in lines]
    # 📁 a line's file may sit in a subfolder of the voiceover folder (e.g. "New" — Daryl, 17 Nov): the folder itself
    # first, then each of VOICE_SUBFOLDERS (and any the map lists under "subfolders")
    where = {f["name"]: folder for f in on_drive}
    for sub in list(cue_map.get("subfolders") or []) + [x for x in VOICE_SUBFOLDERS if x not in (cue_map.get("subfolders") or [])]:
        got, err_ = list_drive_folder(f"{folder}/{sub}")
        for f in ([] if err_ else got):
            where.setdefault(f["name"], f"{folder}/{sub}")
    for k, name in enumerate(names + takes, 1):            # the lines, then 🎲 their other takes
        dest = lines_dir() / name
        if only_missing and dest.exists() and dest.stat().st_size > 0:
            continue
        if progress:
            progress(k, len(names) + len(takes), name)
        err = download_source_video(f"{where.get(name, folder)}/{name}", dest)
        if err:
            errors.append(f"{name}: {err}")
    if avatar:                                             # 👩 her clips: whichever of the lines (and takes) have one on Drive so far
        want = {st_.lower(): st_ for st_ in stems}
        have, seen = [], set()
        for sub in avatar_clip_folders(folder):            # avatar_clips first, then e.g. sitting/avatar_clips (Daryl, 18 Nov)
            files, err = list_drive_folder(f"{folder}/{sub}")
            if err:
                if sub == "avatar_clips":
                    errors.append(f"avatar_clips: {err}")
                continue
            for f in files:
                if (Path(f["name"]).suffix.lower() in E.VIDEO_EXT and f["name"].lower() not in seen
                        and (Path(f["name"]).stem.lower() in want or is_take(f["name"]))):
                    seen.add(f["name"].lower())             # the same name in two folders: the first folder's clip
                    have.append(dict(f, folder=sub))
        for k, f in enumerate(have, 1):
            stem = Path(f["name"]).stem
            dest = avatar_dir() / "lines" / (want.get(stem.lower(), stem) + ".mp4")      # named as the line's own file
            if only_missing and dest.exists() and dest.stat().st_size > 0:
                continue
            if progress:
                progress(k, len(have), f["name"])
            err = download_source_video(f"{folder}/{f['folder']}/{f['name']}", dest)
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


def _phase_hooks(sched: list, to_comp, phase_tags: dict = None) -> dict:
    """compile_plan's per-block hooks for one song (or round of the beat): 🎭 block_role (intro clips in Wait, outro
    clips in Stop) and 🏷️ block_tags (the phase's own tags, when set). to_comp(song time) → compilation time, or None
    where the song isn't heard."""
    def phase(a, b):
        t = to_comp((a + b) / 2)
        return E.phase_at(sched, t) if t is not None else None

    def role(a, b):
        ph = phase(a, b)
        return PHASE_ROLES.get(ph["profile"]) if ph else None
    tags = {p: list(v) for p, v in (phase_tags or {}).items() if v}

    def block_tags(a, b):
        ph = phase(a, b)
        return tags.get(ph["profile"]) if ph else None
    return {"block_role": role, "block_tags": block_tags if tags else None}


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
                 song_per_round: bool = False, energy_mix: bool = False, phase_tags: dict = None, surprises: bool = False,
                 progress=None) -> dict:
    """The Edge plan. track_ids: the playlist in playing order ([] = no song: the generated beat is the track, and
    the beat sound is on whatever `beat` says). The footage is matched 🎯 Movement only — each song on the 📐 lines
    from its music; a song's later plays (and later songs) leave out the clips already used, as long as the library
    can still fill the time without them. avatar: she appears at the start of every phase (her clip for that phase,
    voice included; the stand-in figure until there is one) — in a corner or as half a split screen. intro: the
    intro screen before the compilation (her intro clip beside the settings and thumbnails).
    signals: 🔔 a bell, a flash and the ring in the phase's colour at every phase change (the timer pulsing over
    its last seconds); round_badge: ▸ "ROUND 2 / 3" by the timer; music_fade: 🌙 the songs fade out in Stop and back
    in over Wait; with the voiceover / avatar the songs sit at a steady quarter volume under her (15 Nov);
    song_per_round: 🎵 round r plays song r of the list (from its start), instead of the songs back to back.
    energy_mix: ⚡ the phases in pairs (Wait + Slow, Medium + Fast, Extra + Stop) each get the stretch of a song whose
    energy fits them best (edge_energy.py) — from the round's own song with song_per_round, else from any song in
    the list — joined by the 🎚️ Mix engine (🧠 stem blends ending on the phase change); the footage follows.
    phase_tags: 🏷️ {phase: [scene tags]} — that phase's footage keeps to clips with one of them (any footage where too
    little carries them; the notes say how often). Without a song the beat is matched round by round, each round leaving out the clips earlier rounds used
    (while the library can still fill it). progress(done, total, text)."""
    import core.story_matcher as SMX
    # 👩 the avatar brings the lines with her: she follows the cue map (her clips, their own sound; the line's
    # recording under a stand-in where she has no clip yet) whether or not the voiceover switch is on (Daryl, 14 Nov)
    voiceover = bool(voiceover or avatar)
    surprises = bool(surprises and difficulty in E.HANDS_OFF_SEC)      # 🎲 Chill / Medium only (Hard has its own)
    sched = E.schedule(difficulty, rounds, seed, surprises=surprises)
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

            def to_comp(t, _st=stretches):             # song time → where it plays in the compilation
                hit = next(((bar, off) for bar, play, off in _st if bar <= t < bar + play), None)
                return hit[1] + t - hit[0] if hit else None
            kw = dict(movement_only=True, cuts="lines", max_videos=int(max_videos or 0), progress=prog,
                      **_phase_hooks(sched, to_comp, phase_tags))
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
                      **_phase_hooks(sched, lambda t, _o=part["offset"], _p=part["play"]: _o + t if t < _p else None,
                                     phase_tags))
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
                      **_phase_hooks(sub, lambda t: t, phase_tags),
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
                         "why": p["why"], "moved": p["moved"], "gain": float(by_name[p["line"]].get("gain") or 0)})
        clashes = got["clashes"]
        lines_unused = list(got.get("unused") or [])       # a cue, but for a round / difficulty this compilation doesn't have
        lines_no_cue = sorted((ln["name"] for ln in cue_map["lines"] if not ln.get("cues")), key=str.lower)
        named = lambda ns: [{"line": n, "file": path(n), "seconds": float(durations.get(n) or by_name[n].get("seconds") or 2.0),
                             "gain": float(by_name[n].get("gain") or 0)} for n in ns]   # noqa: E731
        intro_lines, outro_lines = named(got["intro"]), named(got["outro"])
        if avatar:                                         # 👩 she reads the lines: her clip where there is one
            vid = lambda n: (str(avatar_line_clip(by_name[n])) if avatar_line_clip(by_name[n]) else None)       # noqa: E731
            for x in intro_lines + outro_lines:
                x["clip"] = vid(x["line"])
            line_avatar = [{"time": c["time"], "profile": c["profile"],
                            "clip": c.get("clip") or (vid(c["line"]) if c.get("take", 1) == 1 else None),
                            "line": c["line"], "seconds": c["seconds"]} for c in cues]
        lines_missing = sorted({c["line"] for c in cues + (intro_lines if intro else []) + outro_lines if not c["file"]}, key=str.lower)
    else:
        cues = []                                          # (the cue map ships with the app, so this is only an empty map)
    missing_voice = []
    av_cues = (line_avatar or []) if avatar else []        # 👩 she appears with every line of the cue map
    missing_avatar = sorted({c["line"] for c in av_cues if not c["clip"]}, key=str.lower)
    intro_dir = avatar_dir() / "intro"                     # her own intro clip, when one has been put on the server
    intro_clips = sorted(f for f in intro_dir.iterdir() if f.suffix.lower() in E.VIDEO_EXT) if (intro and intro_dir.is_dir()) else []
    seen, thumbs = {}, []
    for e in timeline:                                    # the intro's thumbnails: the six most used videos
        for s_ in e["scenes"]:
            for link in s_["chain"]:
                first = seen.setdefault(link["video_id"], [0.0, float(link["clip_start_sec"])])
                first[0] += float(link["clip_duration_sec"])
    thumbs = [[vid, t] for vid, (_secs, t) in sorted(seen.items(), key=lambda kv: -kv[1][0])[:6]]
    if surprises:
        sl = E.surprise_list(sched)
        kinds = {"handsoff": "✋ hands off", "override": "⏩ override", "falsefinish": "🎭 false finish"}
        notes.append("🎲 Surprises: " + (", ".join(f"{kinds[x['kind']]} {x['seconds']:g} s (round {x['round']}, "
                                                   f"{E.PROFILES[x['profile']]['label'].title()})" for x in sl) or "none drawn"))
    elif difficulty == "hard":
        mids = [ph for ph in sched if ph["round"] == 1][1:-2]
        notes.append(f"💀 Hard: a random middle each round — round 1: {len(mids)} phases, "
                     + " → ".join(f"{E.PROFILES[ph['profile']]['label'].title()} {ph['end'] - ph['start']:g}s" for ph in mids) + ".")
    # 🎯 the stroke bar's beats: the generated beat's, or the songs' thinned / doubled toward each phase's pace
    song_beats = {tid: PC._load_track(tid).get("beat_times") or [] for tid in dict.fromkeys(p["track_id"] for p in parts)}
    guide = E.pace_guide(sched, parts if track_ids else [], song_beats)
    off = E.guide_off_target(guide["per_phase"])
    if off:
        worst = max(off, key=lambda q: abs(q["bpm_used"] / q["bpm_target"] - 1))
        ph_ = sched[worst["index"]]
        notes.append(f"🎯 Pace guide: {len(off)} phase{'s' if len(off) != 1 else ''} follow the song's beat away from "
                     f"their pace (e.g. round {ph_['round']} {E.PROFILES[ph_['profile']]['label'].title()}: target "
                     f"{worst['bpm_target']:.0f} bpm, the song gives {worst['bpm_used']:.0f}).")
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
                      "stop_fade": "whole phase", "duck": False,
                      "song_per_round": bool(song_per_round and track_ids),
                      "energy_mix": bool(energy_mix and track_ids)},
            "signals": {"on": bool(signals), "sound": bool(signals), "timer": bool(signals), "flash": bool(signals)},
            "round_badge": {"on": bool(round_badge)},
            # 🎯 the stroke bar by the timer (Daryl, 16 Nov: always on) and ⏭️ the next phase's name under the timer
            "pace": {"on": True, "beats": guide["beats"], "per_phase": guide["per_phase"],
                     "table": {d: dict(v) for d, v in E.PACE.items()}},
            "next_up": {"on": True},
            "surprises": {"on": surprises, "list": E.surprise_list(sched)},
            "phase_tags": {p: list(v) for p, v in (phase_tags or {}).items() if v},
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
            "surprises": bool(ss.get("edge_surprises", False)),
            "phase_tags": {p: list(ss.get("edge_tags_" + p) or []) for p in E.ORDER if ss.get("edge_tags_" + p)},
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
                           surprises=settings.get("surprises", False),
                           phase_tags=settings.get("phase_tags"),
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


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


def _missing_sources(plan: dict, quick: bool) -> list:
    """(raw source, local path) of every song and video the plan uses that isn't on this machine."""
    import render_jobs as RJ
    return RJ.missing_sources(plan, quick, [p.get("source_path") for p in plan["edge"].get("playlist") or []])


def _render(res: dict, settings: dict, token, prog) -> tuple:
    """🎬 The compiled plan → a preview (render_jobs.render: numbered, uploaded to Drive when connected, timed, length-
    checked). Returns (message for the page, file name)."""
    import render_jobs as RJ
    plan = res["plan"]
    e = plan["edge"]
    return RJ.render(plan, stream=settings["stream"], quick=settings["quick"], label=settings.get("label", ""), token=token,
                     prog=prog, extra_raw=[p.get("source_path") for p in e.get("playlist") or []],
                     extra_meta={"project_name": f"⏱️ Edge · {E.DIFFICULTIES[e['difficulty']]} · {e['rounds']} rounds",
                                 "compile_note": _used_line(res),
                                 # ↩️ so 📁 Media Library → Renders can reopen it in ⏱️ Edge Mode as it was set
                                 "edge_setup": {k: v for k, v in settings.items() if k != "label"}})


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


def _setup_from_plan(edge: dict) -> dict:
    """The setup an older Edge render was made with, worked out from its plan (renders from 16 Nov carry it)."""
    music, vo, av = edge.get("music") or {}, edge.get("voiceover") or {}, edge.get("avatar") or {}
    parts = edge.get("playlist") or []
    if music.get("song_per_round"):
        # round r played song r of the list (the list starting again): the list is each round's song, repeats dropped
        playlist = list(dict.fromkeys(p["track_id"] for p in sorted(parts, key=lambda p: (p.get("round", 0), p["offset"]))))
    else:
        playlist = list(dict.fromkeys(p["track_id"] for p in sorted(parts, key=lambda p: p["offset"])))
    return {"playlist": playlist, "difficulty": edge.get("difficulty", "chill"), "rounds": int(edge.get("rounds") or 3),
            "voiceover": bool(vo.get("on")) and not av.get("on"), "beat": bool((edge.get("beat") or {}).get("on")) and bool(parts),
            "corner": (edge.get("overlay") or {}).get("corner", "top-left"), "avatar": bool(av.get("on")),
            "avatar_mode": av.get("mode", "corner"), "avatar_corner": av.get("corner", "bottom-right"),
            "intro": bool((edge.get("intro") or {}).get("on")), "song_per_round": bool(music.get("song_per_round")),
            "energy_mix": bool(music.get("energy_mix")), "phase_tags": dict(edge.get("phase_tags") or {}),
            "surprises": bool((edge.get("surprises") or {}).get("on"))}


SETUP_KEYS = {"playlist": "edge_playlist", "difficulty": "edge_difficulty", "rounds": "edge_rounds", "voiceover": "edge_voice",
              "beat": "edge_beat", "corner": "edge_corner", "max_videos": "edge_max_videos", "avatar": "edge_avatar",
              "avatar_mode": "edge_avatar_mode", "avatar_corner": "edge_avatar_corner", "intro": "edge_intro",
              "song_per_round": "edge_song_per_round", "energy_mix": "edge_energy_mix", "stream": "edge_stream",
              "surprises": "edge_surprises",
              "quick": "edge_quick"}


def render() -> None:
    """⏱️ The page (drawn by planner/edge_view.py)."""
    from planner import edge_view
    edge_view.render()
