"""🎙️ The cue map, kept by the app (19 Nov 2026).

Until 18 Nov the cue map lived in a claude.ai artifact and reached the app as "sent" files on Drive
(Edge_voiceover/edge_cues_<time>.json, the newest picked up by every compile). A stale artifact copy overwrote Daryl's
work on 18 Nov, so he asked for the Cue Map in the app itself. From now on:

  - the server's file (<data>/edge_voiceover/edge_cues.json) is the only cue map; once it carries "managed": "app",
    compiles no longer pick up sent files from Drive (planner.edge.fetch_lines);
  - every save keeps the previous file in <data>/edge_voiceover/cue_versions/ (restorable; the newest KEEP_VERSIONS);
  - every save says which revision it was made from: if another device saved in between, it is refused (no silent
    overwrites between the phone and the iPad);
  - each line can be cut (trim_in / trim_out, 0 = to the end) and sped up or slowed (speed) — planner.edge plays the
    cut, for her clips too.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

KEEP_VERSIONS = 200
CUE_TYPES = ("phase", "random", "interval", "time", "round", "intro", "outro", "after", "surprise")


class Stale(Exception):
    """The cue map changed since it was opened (another device saved)."""


def voice_dir() -> Path:
    from planner.edge import voice_dir as vd
    return vd()


def path() -> Path:
    return voice_dir() / "edge_cues.json"


def versions_dir() -> Path:
    return voice_dir() / "cue_versions"


def load() -> dict:
    """The cue map in use (the app's own copy when there is one, else the one that ships with the app), with "rev"."""
    import edge_mode as E
    p = path() if path().exists() and path().stat().st_size > 0 else E.CUE_MAP_FILE
    cm = E.load_cue_map(p)
    cm.setdefault("rev", 0)
    cm.setdefault("lines", [])
    return cm


def managed() -> bool:
    """True once the cue map is kept in the app: compiles leave Drive's sent files alone."""
    try:
        return json.loads(path().read_text(encoding="utf-8")).get("managed") == "app"
    except (OSError, ValueError):
        return False


def save(cm: dict, rev: int, note: str = "") -> dict:
    """Write the cue map, made from revision `rev`. Raises Stale when the file has moved on since. The file it
    replaces is kept as a version. Returns the saved map (rev + 1)."""
    cur = load()
    if int(cur.get("rev") or 0) != int(rev):
        raise Stale(f"The cue map was changed elsewhere (now revision {cur.get('rev')}, you opened {rev}).")
    out = copy.deepcopy(cm)
    out["rev"] = int(rev) + 1
    out["managed"] = "app"
    out["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["saved_note"] = note
    for ln in out.get("lines") or []:
        ln["play_seconds"] = round(played_seconds(ln), 3)
    voice_dir().mkdir(parents=True, exist_ok=True)
    if path().exists():
        _keep_version(path().read_text(encoding="utf-8"), cur)
    tmp = path().with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path())
    return out


def _keep_version(text: str, cur: dict) -> None:
    d = versions_dir()
    d.mkdir(parents=True, exist_ok=True)
    name = f"edge_cues_r{int(cur.get('rev') or 0):05d}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    (d / name).write_text(text, encoding="utf-8")
    old = sorted(d.glob("edge_cues_r*.json"))
    for f in old[:-KEEP_VERSIONS]:
        f.unlink(missing_ok=True)


def versions() -> list:
    """[{"file", "rev", "saved_at", "note", "lines"}], newest first."""
    out = []
    for f in sorted(versions_dir().glob("edge_cues_r*.json"), reverse=True) if versions_dir().is_dir() else []:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"file": f.name, "rev": d.get("rev", 0), "saved_at": d.get("saved_at") or d.get("source") or "",
                    "note": d.get("saved_note") or "", "lines": len(d.get("lines") or []),
                    "cued": sum(1 for ln in d.get("lines") or [] if ln.get("cues"))})
    return out


def restore(file_name: str, rev: int) -> dict:
    """An older version back as the cue map (a new revision; the one it replaces is kept too)."""
    d = json.loads((versions_dir() / file_name).read_text(encoding="utf-8"))
    return save(d, rev, note=f"restored {file_name}")


def import_raw(raw: bytes, rev: int, note: str) -> dict:
    """A cue map file (the artifact's send, or an export from here) as the new cue map."""
    d = json.loads(raw.decode("utf-8"))
    if not isinstance(d, dict) or not isinstance(d.get("lines"), list) or not all(
            isinstance(ln, dict) and ln.get("name") for ln in d["lines"]):
        raise ValueError("That file has no list of lines with names — is it a cue map?")
    return save(d, rev, note=note)


# ---------------------------------------------------------------------------
# A line's length
# ---------------------------------------------------------------------------

def is_cut(line: dict) -> bool:
    return bool(float(line.get("trim_in") or 0) > 0.005 or float(line.get("trim_out") or 0) > 0.005
                or abs(float(line.get("speed") or 1) - 1) > 0.005)


def played_seconds(line: dict, measured: float | None = None) -> float:
    """How long the line plays: its file (measured, else "seconds") from trim_in to trim_out (0 = the end), at its
    speed."""
    full = float(measured if measured else (line.get("seconds") or 0.0)) or 2.0
    a = max(float(line.get("trim_in") or 0.0), 0.0)
    b = float(line.get("trim_out") or 0.0) or full
    b = min(b, full)
    sp = float(line.get("speed") or 1.0) or 1.0
    return max(b - a, 0.05) / sp


def new_line(name: str, file: str, seconds: float, folder: str = "") -> dict:
    return {"name": name, "file": file, "folder": folder, "cats": [], "seconds": round(float(seconds or 0), 3), "gain": 0,
            "trim_in": 0, "trim_out": 0, "speed": 1, "play_seconds": round(float(seconds or 0), 3), "cues": [],
            "notes": "", "avatar": {"mode": "", "file": "", "after": ""}}


def categories(cm: dict) -> list:
    base = ["Rounds", "Speed", "Slow", "Fast", "Stop", "Start", "Filler", "Surprise", "React"]
    have = [c for ln in cm.get("lines") or [] for c in ln.get("cats") or []]
    return list(dict.fromkeys(base + sorted(set(have), key=str.lower)))


def describe(cue: dict) -> str:
    """One line in words, e.g. "Start of Slow · round 1 · Medium / Hard"."""
    t = cue.get("type")
    rounds = ", ".join(str(r) for r in cue.get("roundList") or []) or "every round"
    diffs = " / ".join(d.title() for d in cue.get("diffs") or []) or "all difficulties"
    phases = " / ".join(p.title() for p in cue.get("phases") or []) or "any phase"
    if t == "phase":
        at = cue.get("at") or "start"
        off = float(cue.get("offset") or 0)
        where = {"start": "Start of", "during": "During", "end": "End of"}.get(at, at)
        extra = f" (+{off:g} s)" if off and at != "end" else ""
        return f"{where} {phases}{extra} · round {rounds} · {diffs}"
    if t == "random":
        return f"At random {int(cue.get('times') or 1)}× a {cue.get('per') or 'round'} in {phases} · round {rounds} · {diffs}"
    if t == "interval":
        return f"Every {float(cue.get('every') or 60):g} s in {phases} · round {rounds} · {diffs}"
    if t == "time":
        return f"At {cue.get('clock') or '0:00'}" + (" into each round" if cue.get("from") == "round" else "") + f" · {diffs}"
    if t == "round":
        which = cue.get("which") or "every"
        r = {"every": "every round", "first": "the first round", "last": "the last round"}.get(which, f"round {cue.get('number') or 1}")
        return f"{(cue.get('at') or 'start').title()} of {r} · {diffs}"
    if t == "intro":
        return f"Intro screen (order {cue.get('order', '—')})"
    if t == "outro":
        return f"Outro screen (order {cue.get('order', '—')})"
    if t == "after":
        return f"Straight after “{cue.get('after')}” (+{float(cue.get('gap') or 0):g} s)"
    if t == "surprise":
        ln = f" {cue.get('length')} s" if cue.get("length") else ""
        return f"🎲 {cue.get('kind') or 'handsoff'}{ln} · at its {cue.get('at') or 'start'} · {diffs}"
    return str(t)
