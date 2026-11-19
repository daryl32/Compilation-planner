"""🌙 Overnight renders — after the nightly 🔮 recommendation searches (jobs.py --nightly), one ⏱️ Edge Mode and one
⚔️ Verses Battle compilation with settings drawn at random are compiled and rendered, so they're waiting in the
morning on 🔮 Recommendations → 🌙 Overnight (Daryl, 19 Nov: the waiting time, where it can't be avoided, happens
while nobody is waiting).

Edge: avatar + voiceover always on (with the intro screen); difficulty, rounds (3–4), 1–3 songs, a new song each
round, ⚡ matching song sections and 🎲 surprises drawn at random. Verses: two fighters drafted at random (any /
contrast / rivals), music (shared / solo themes, auto-picked), layout, length and a surprise winner drawn at random.

The renders stay on the server (the nightly run has no Google Drive sign-in): save the ones you like to Drive from
the card or from 📁 Media Library → Renders. Unrated, unkept overnight renders that never went to Drive are cleared
after KEEP_NIGHTS nights. Each night's record is kept in <recs>/overnight.json.

Usage:  python nightly_modes.py            (what jobs.py --nightly calls at the end)
        python nightly_modes.py --edge     (only Edge — or --verses)
"""

from __future__ import annotations

import datetime
import json
import random
import time
import traceback
from pathlib import Path

KEEP_NIGHTS = 3                     # unrated, unkept overnight renders only on the server are cleared after this
HISTORY = 14                        # nights kept in the record
MIN_SONG_SEC = 90.0                 # Edge's random playlist leaves out shorter tracks
EDGE_ROUNDS = (3, 4)


def record_path() -> Path:
    import rec_library as RL
    return RL.BASE_DIR / "overnight.json"


def history() -> list:
    """The nights, newest first: [{"date", "started", "finished", "edge": {...}, "verses": {...}}]."""
    try:
        got = json.loads(record_path().read_text())
        return list(got.get("nights") or [])
    except Exception:          # noqa: BLE001
        return []


def _save(nights: list) -> None:
    p = record_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"nights": nights[:HISTORY]}, indent=1, default=str))
    tmp.replace(p)


class _Job:
    """Stands in for a server job (mix_jobs.Job): the planners' background work reports its progress to it."""

    def __init__(self, log):
        self.frac, self.msg, self._log, self._last = 0.0, "", log, 0.0

    def progress(self, frac=None, msg=None) -> None:
        if frac is not None:
            self.frac = max(self.frac, min(1.0, float(frac)))
        if msg is not None:
            self.msg = msg
        if time.time() - self._last > 30:                 # a line in the log now and then
            self._last = time.time()
            self._log(f"{int(self.frac * 100)} % · {self.msg}")


# ---------------------------------------------------------------------------
# The draws
# ---------------------------------------------------------------------------

def _songs() -> list:
    import planner.compose as PC
    out = []
    for tid, s in (PC.audio_summaries(PC._audio_signature()) or {}).items():
        try:
            if float(s.get("duration") or 0.0) >= MIN_SONG_SEC:
                out.append(tid)
        except (TypeError, ValueError):
            continue
    return sorted(out)


def draw_edge(rng: random.Random, songs: list = None) -> dict:
    """⏱️ An Edge setup: avatar + voiceover always on; the rest drawn."""
    import edge_mode as E
    songs = _songs() if songs is None else songs
    playlist = rng.sample(songs, min(len(songs), rng.randint(1, 3))) if songs else []
    diff = rng.choice(list(E.DIFFICULTIES))
    return {"playlist": playlist, "difficulty": diff, "rounds": rng.randint(*EDGE_ROUNDS),
            "voiceover": True, "beat": not playlist, "corner": "top-left", "max_videos": 0,
            "avatar": True, "avatar_mode": rng.choice(["corner", "split"]), "avatar_corner": "bottom-right",
            "intro": True, "song_per_round": len(playlist) > 1 and rng.random() < 0.5,
            "energy_mix": bool(playlist) and rng.random() < 0.5,
            "surprises": diff in getattr(E, "HANDS_OFF_SEC", {}) and rng.random() < 0.5,
            "phase_tags": {}, "stream": True, "quick": True, "label": ""}


def draw_verses(rng: random.Random, pool: dict = None) -> dict:
    """⚔️ A Verses setup: two fighters drafted at random, everything else drawn (a surprise winner)."""
    import planner.verses as PV
    import verses_battle as VB
    pool = PV.fighters_pool() if pool is None else pool
    names = VB.eligible_actors(pool)
    style = rng.choice(list(VB.DRAFT_STYLES))
    a, b = VB.draft(names, style, rng, looks=PV.looks(pool) if style != "any" else None)
    return {"a": a, "b": b, "draft": "random", "style": style, "music": rng.choice(list(VB.MUSIC_MODES)),
            "track_a": "", "track_b": "", "layout": rng.choice(list(VB.LAYOUTS)), "winner": "surprise",
            "length": rng.choice([0, 180, 240]), "max_videos": 0, "intro": True, "overlay": True,
            "stream": True, "quick": True, "label": ""}


# ---------------------------------------------------------------------------
# Making them
# ---------------------------------------------------------------------------

def _all_videos() -> list:
    import planner.compose as PC
    rows = PC._video_rows()
    return sorted(PC._videos_with_timeline() & set(rows), key=str.lower)


def run_edge(rng: random.Random, log=print) -> dict:
    import planner.edge as PE
    settings = draw_edge(rng)
    seed = rng.randrange(1, 10 ** 6)
    settings["label"] = (f"🌙 Overnight Edge · {settings['difficulty'].title()} · {settings['rounds']} rounds")
    log("⏱️ Edge: " + json.dumps({k: v for k, v in settings.items() if k not in ("phase_tags", "stream", "quick")}))
    got = PE._job(_Job(log), settings, _all_videos(), seed, True, None)
    res = got["res"]
    return {"render": got.get("render"), "note": (got.get("note") or [None, ""])[1], "setup": settings, "seed": seed,
            "line": PE._used_line(res), "took": PE.took_line(res) if res.get("took") else ""}


def run_verses(rng: random.Random, log=print) -> dict:
    import planner.verses as PV
    settings = draw_verses(rng)
    seed = rng.randrange(1, 10 ** 6)
    settings["label"] = f"🌙 Overnight battle · {settings['a']} vs {settings['b']}"
    log("⚔️ Verses: " + json.dumps({k: v for k, v in settings.items() if k not in ("stream", "quick")}))
    got = PV._job(_Job(log), settings, seed, True, None)
    res = got["res"]
    return {"render": got.get("render"), "note": (got.get("note") or [None, ""])[1], "setup": settings, "seed": seed,
            "line": PV.used_line(res), "took": ""}


def _mark(render_name: str | None, kind: str) -> None:
    """Tag the render's details as an overnight one (what the clearing and the 🌙 tab go by)."""
    if not render_name:
        return
    try:
        from config import PREVIEW_DIR
        from renders import _save_meta, load_render
        mp4 = Path(PREVIEW_DIR) / render_name
        meta = load_render(mp4)
        if meta:
            meta["overnight"] = {"kind": kind, "date": datetime.date.today().isoformat()}
            _save_meta(mp4, meta)
    except Exception:          # noqa: BLE001
        pass


def clear_old(log=print) -> list:
    """Overnight renders older than KEEP_NIGHTS nights that are only on the server, unrated and not kept: removed.
    Returns their names."""
    from config import PREVIEW_DIR
    from renders import delete_render, load_render, on_drive
    gone = []
    cutoff = (datetime.date.today() - datetime.timedelta(days=KEEP_NIGHTS)).isoformat()
    for night in history():
        for kind in ("edge", "verses"):
            name = (night.get(kind) or {}).get("render")
            if not name or str(night.get("date") or "") > cutoff:
                continue
            mp4 = Path(PREVIEW_DIR) / name
            meta = load_render(mp4)
            if not meta or not meta.get("overnight") or on_drive(meta):
                continue
            user = meta.get("user") or {}
            if user.get("keep") or int(user.get("rating") or 0):
                continue
            if delete_render(mp4) is None:
                gone.append(name)
    if gone:
        log(f"🧹 Cleared {len(gone)} old overnight render(s): " + ", ".join(gone))
    return gone


def run(which=("edge", "verses"), seed: int = None, log=print) -> dict:
    """One night: each of `which` compiled and rendered (one failing doesn't stop the other), recorded."""
    rng = random.Random(seed if seed is not None else time.time_ns())
    night = {"date": datetime.date.today().isoformat(), "started": time.time(), "finished": None}
    nights = [n for n in history() if n.get("date") != night["date"] or not set(which) & set(n)]
    for kind, fn in (("edge", run_edge), ("verses", run_verses)):
        if kind not in which:
            continue
        t0 = time.time()
        try:
            got = fn(rng, log)
            _mark(got.get("render"), kind)
            night[kind] = {**got, "seconds": round(time.time() - t0, 1)}
            log(f"✅ {kind}: {got.get('render')} in {int(time.time() - t0)} s")
        except Exception as e:          # noqa: BLE001 — the other one still runs; the card says what went wrong
            night[kind] = {"error": f"{type(e).__name__}: {e}"[:600], "seconds": round(time.time() - t0, 1)}
            log(f"❌ {kind}: {e}\n{traceback.format_exc()[-1500:]}")
        _save([night] + nights)
    night["finished"] = time.time()
    _save([night] + nights)
    try:
        night["cleared"] = clear_old(log)
    except Exception as e:          # noqa: BLE001
        log(f"Couldn't clear old overnight renders: {e}")
    return night


def _cli() -> None:
    import argparse
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge", action="store_true")
    ap.add_argument("--verses", action="store_true")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    which = tuple(k for k in ("edge", "verses") if getattr(args, k)) or ("edge", "verses")
    run(which, args.seed, log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))


if __name__ == "__main__":
    _cli()
