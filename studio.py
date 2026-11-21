"""🧍 Avatar Studio in the app (19 Nov 2026, phase 2 of moving off the artifacts) — what to record next.

Daryl moved the Cue Map into the app after a stale artifact copy overwrote his work; the Studio artifact's recording
tracker follows. Its data now comes from the app itself:

  lines      the cue map (cue_store) — and where each line plays, worked out from its cues
  the plan   the look for each part (planner.edge.load_looks — 👗 the same plan the compile uses)
  clips      every video in Edge_voiceover on Drive (studio_previews.clips): a voice clip in "<Outfit> - <Pose>/avatar_clips"
             is in that look with a background; the old transparent clips sit in avatar_clips (Default · Lying on bed 1)
             and sitting/avatar_clips (Default · Sitting on Bed1) — LEGACY_FOLDERS

  📋 per look: the lines it needs, which are recorded (✅ background / 🟡 transparent only / ❌ not yet)
  ⚠️ clips recorded in a look where their line never plays (wrong look)
  📦 Colab sessions: per look, what's left (main lines shortest first, fillers / surprises after), seconds and an
     estimate of Colab units, and its to_record.txt written to the look's folder for the notebook's copy cell
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

ROOT = "scene-labeling/Edge_voiceover"
VIDEO_EXT = (".mp4", ".mov", ".webm", ".m4v")
# the clips made before the look folders (18 Nov): their look, and that they have no background
LEGACY_FOLDERS = {"avatar_clips": "Default - Lying on bed 1", "sitting/avatar_clips": "Default - Sitting on Bed1"}
PARTS = ("intro", "1", "2", "3", "4", "5", "6", "last", "outro")
ROUND_PARTS = ("1", "2", "3", "4", "5", "6", "last")
UNIT_PER_CHUNK = 0.52          # Colab units for one ~8 s chunk of lip-sync on the 80 GB A100 (18 Nov estimate)
CHUNK_SEC = 8.0
SESSION_UNITS = 1.5            # loading the model, once a session


def cache_path() -> Path:
    from planner.edge import voice_dir
    return voice_dir() / "studio_clips.json"


def load_clips() -> dict:
    """{"at", "clips": [{"id", "name", "size", "folder"}]} — the last Drive listing (🔄 to refresh)."""
    try:
        return json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"at": "", "clips": []}


def refresh_clips() -> tuple:
    """Read every video in Edge_voiceover from Drive again. (n, error)."""
    import studio_previews as SP
    got, err = SP.clips()
    if err:
        return 0, err
    cache_path().parent.mkdir(parents=True, exist_ok=True)
    cache_path().write_text(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M"), "clips": got}, ensure_ascii=False),
                            encoding="utf-8")
    return len(got), None


def _key(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", str(s)).replace("’", "'").lower().strip()


def clip_look(c: dict) -> tuple:
    """(look, background?) of a clip by where it sits, or (None, None) when it isn't a voice clip."""
    folder = str(c.get("folder") or "")
    parts = folder.split("/")
    if "avatar_clips" not in parts:
        return None, None
    top = parts[0]
    if " - " in top:
        return top, not c["name"].lower().endswith(".webm")
    look = LEGACY_FOLDERS.get(folder)
    return (look, False) if look else (None, None)


def recordings(clips: list) -> dict:
    """{line key: [{"look", "background", "name", "folder", "id"}]} — every voice clip, by its line's file stem."""
    out = {}
    for c in clips:
        if not c["name"].lower().endswith(VIDEO_EXT):
            continue
        look, bg = clip_look(c)
        if look is None:
            continue
        out.setdefault(_key(Path(c["name"]).stem), []).append({"look": look, "background": bool(bg), "name": c["name"],
                                                               "folder": c["folder"], "id": c.get("id")})
    return out


# ---------------------------------------------------------------------------
# Where a line plays
# ---------------------------------------------------------------------------

def kind(line: dict) -> str:
    """main · filler · surprise — fillers and surprises are recorded after the main lines."""
    cues = line.get("cues") or []
    if any(c.get("type") == "surprise" for c in cues):
        return "surprise"
    cats = {str(x).lower() for x in line.get("cats") or []}
    if "filler" in cats or str(line.get("name", "")).lower().startswith("filler") or (
            cues and all(c.get("type") == "random" for c in cues)):
        return "filler"
    return "main"


def line_parts(line: dict, by_name: dict, _seen=None) -> set:
    """The parts (intro, 1–6, last, outro) a line can play in, from its cues — the Studio's rules: a round list
    plays in those rounds (none = every round); round cues: first → 1, last → last, every → all; 🎲 surprise lines
    from round 2 on (they take the round's look); "straight after" follows the line it comes after."""
    out = set()
    _seen = set(_seen or ()) | {line["name"]}
    for c in line.get("cues") or []:
        t = c.get("type")
        if t == "intro":
            out.add("intro")
        elif t == "outro":
            out.add("outro")
        elif t == "surprise":
            out |= set(ROUND_PARTS) - {"1"}
        elif t == "round":
            which = c.get("which") or "every"
            out |= {"first": {"1"}, "last": {"last"}, "every": set(ROUND_PARTS)}.get(
                which, {str(int(c.get("number") or 1))})
        elif t == "after":
            target = by_name.get(str(c.get("after") or ""))
            if target and target["name"] not in _seen:
                out |= line_parts(target, by_name, _seen)
        elif t in ("phase", "random", "interval", "time"):
            rl = [str(x) for x in c.get("roundList") or []]
            out |= set(ROUND_PARTS) if not rl or {"1", "2", "3", "4", "5", "6"} <= set(rl) else set(rl) & set(ROUND_PARTS)
    return out


def needed_looks(parts: set, looks: dict) -> dict:
    """{look: [parts]} for the parts a line plays in."""
    out = {}
    for p in PARTS:
        if p in parts and looks.get(p):
            out.setdefault(looks[p], []).append(p)
    return out


PART_SHORT = {"intro": "Intro", "last": "Last", "outro": "Outro", **{str(r): f"R{r}" for r in range(1, 7)}}


def parts_label(parts) -> str:
    return ", ".join(PART_SHORT[p] for p in PARTS if p in parts) or "—"


# ---------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------

def _callout_lines() -> set:
    import edge_mode as E
    return {n for names in E.CALLOUT_EXTRA.values() for n in names}


def tracker(cm: dict, looks: dict, clips: list) -> list:
    """One row per line: {"name", "file", "seconds", "kind", "parts", "needs": {look: status}, "recorded": [...],
    "wrong": [recordings in a look the line never plays in]}. status: ✅ background · 🟡 transparent only · ❌ none."""
    import cue_store as CS
    by_name = {ln["name"]: ln for ln in cm.get("lines") or []}
    recs = recordings(clips)
    rows = []
    for ln in cm.get("lines") or []:
        stem = Path(str(ln.get("file") or ln["name"])).stem
        have = recs.get(_key(stem), [])
        parts = line_parts(ln, by_name)
        lkind = kind(ln)
        if ln["name"] in _callout_lines():               # 💀 Hard's callouts use it with no cue of its own (19 Nov)
            parts |= set(ROUND_PARTS)
            lkind = "filler" if lkind == "main" and not ln.get("cues") else lkind
        need = needed_looks(parts, looks)
        status = {}
        for lk in need:
            got = [r for r in have if r["look"] == lk]
            status[lk] = "✅" if any(r["background"] for r in got) else "🟡" if got else "❌"
        wrong = [r for r in have if r["look"] not in need]
        rows.append({"name": ln["name"], "file": ln.get("file") or f"{ln['name']}.mp3", "seconds": CS.played_seconds(ln),
                     "full_seconds": float(ln.get("seconds") or 0), "kind": lkind, "parts": sorted(parts, key=PARTS.index),
                     "needs": status, "need_parts": need, "recorded": have, "wrong": wrong, "cued": bool(ln.get("cues"))})
    return rows


def units(seconds: float, unit_per_chunk: float = UNIT_PER_CHUNK, chunk: float = CHUNK_SEC) -> float:
    return unit_per_chunk * max(1, math.ceil(max(seconds, 0.1) / chunk))


def sessions(rows: list, looks: dict, unit_per_chunk: float = UNIT_PER_CHUNK, chunk: float = CHUNK_SEC,
             with_stand_in: bool = True) -> list:
    """Per look in the plan: what's left to record there. A line with only a transparent clip (🟡) counts as to do
    (re-record with a background) unless with_stand_in is False."""
    out = []
    for lk in dict.fromkeys(v for v in (looks.get(p) for p in PARTS) if v):
        todo = [r for r in rows if lk in r["needs"] and (r["needs"][lk] == "❌" or (with_stand_in and r["needs"][lk] == "🟡"))]
        main = sorted((r for r in todo if r["kind"] == "main"), key=lambda r: (r["full_seconds"] or r["seconds"]))
        extra = sorted((r for r in todo if r["kind"] != "main"), key=lambda r: (r["kind"], r["full_seconds"] or r["seconds"]))
        done = sum(1 for r in rows if r["needs"].get(lk) == "✅")
        sec = lambda rs: sum(r["full_seconds"] or r["seconds"] for r in rs)          # noqa: E731 (Colab lip-syncs the whole MP3)
        u = lambda rs: sum(units(r["full_seconds"] or r["seconds"], unit_per_chunk, chunk) for r in rs)   # noqa: E731
        out.append({"look": lk, "parts": [p for p in PARTS if looks.get(p) == lk], "done": done,
                    "main": main, "extra": extra, "main_sec": sec(main), "extra_sec": sec(extra),
                    "main_units": u(main), "extra_units": u(extra),
                    "stand_ins": sum(1 for r in todo if r["needs"][lk] == "🟡")})
    return out


def to_record_text(session: dict) -> str:
    """The look's to_record.txt, as the notebook's copy cell reads it: one MP3 per line, main lines shortest first;
    then the fillers and surprises — recorded too (Daryl, 20 Nov: no # in front of them any more)."""
    lines = [f"# {session['look']} — written by the app's Avatar Studio {time.strftime('%d %b %H:%M')}",
             f"# {len(session['main'])} main lines ({session['main_sec']:.0f} s), shortest first"]
    lines += [r["file"] for r in session["main"]]
    if session["extra"]:
        lines.append(f"# fillers / surprises ({len(session['extra'])}, {session['extra_sec']:.0f} s) — recorded too; put a # in front of any to skip it")
        lines += [r["file"] for r in session["extra"]]
    return "\n".join(lines) + "\n"


def write_to_record(token: dict, session: dict) -> str | None:
    """to_record.txt into the look's Drive folder (replacing the old one). Returns an error, or None."""
    import tempfile
    from drive_oauth import upload_file_with_oauth
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "to_record.txt"
        f.write_text(to_record_text(session), encoding="utf-8")
        _, err = upload_file_with_oauth(token, f, f"{ROOT}/{session['look']}")
    return err


def drive_looks(clips: list) -> list:
    """Look folders seen on Drive ("<Outfit> - <Pose>"), whether or not they have clips yet."""
    return sorted({str(c.get("folder") or "").split("/")[0] for c in clips if " - " in str(c.get("folder") or "").split("/")[0]})
