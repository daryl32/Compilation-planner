"""
➕ Add new videos — the server's half of getting new videos into the app, as one background job (13 Nov 2026).

Before this it was three separate steps in three places: 🔄 Sync now in the sidebar, `deploy/student.sh export` over
SSH, 🧭 Build in the Timeline section — and nothing said which of them a new video was still waiting for.

    python add_videos.py run        sync from Drive → score the videos without scores → build their timelines
    python add_videos.py status     what every video still needs

The laptop's half comes first (process_new_videos.bat → local_vision.bat → student_tags.bat upload-local): it puts
the catalogue + preview copy on Drive and the vision data on the server.

What each step needs:
  1. sync      the Drive service account (auto_sync.run_sync) — catalogues, thumbnails, preview copies.
  2. scores    the student model the last export used (catalogue/student_scores/index.json → "model"), run by
               deploy/student.sh export --name <model> --missing (its own Python with PyTorch, low priority, a
               memory cap). Only videos with vision data and no scores yet.
  3. timeline  footage_timeline.build() — new and changed videos only.
A step that can't run (no Drive credentials on this machine, no model yet, no PyTorch) is skipped with a note; the
others still run. Progress goes to catalogue/.add_videos.json for the app.

No Streamlit in here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ("sync", "scores", "timeline")
STEP_LABEL = {"sync": "☁️ Syncing from Google Drive", "scores": "🎓 Scoring the new videos",
              "timeline": "🧭 Building their timelines"}
NEEDS = ("catalogue", "vision", "scores", "timeline")     # what the 🤖 new matcher needs, in the order a video gets it
#                                                            (the preview copy is reported too, but the matcher
#                                                            works without one — only ⚡ Quick preview renders need it)
NEED_LABEL = {"catalogue": "catalogue", "vision": "vision data", "scores": "tag scores", "timeline": "timeline"}
NEED_WHERE = {"catalogue": "laptop: process_new_videos.bat, then ➕ here",
              "vision": "laptop: local_vision.bat + student_tags.bat upload-local", "scores": "➕ Add new videos",
              "timeline": "➕ Add new videos"}


def _catalogue_dir() -> Path:
    from config import CATALOGUE_DIR
    return Path(CATALOGUE_DIR)


def status_path() -> Path:
    return _catalogue_dir() / ".add_videos.json"


def _read(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_status() -> dict:
    """{"state": running | done | error, "step", "detail", "started", "finished", "notes": [...], "result": {...}}.
    A run whose process is gone is reported as an error, not as still running."""
    st = _read(status_path(), {}) or {}
    if st.get("state") == "running" and not _alive(st.get("pid")):
        st = {**st, "state": "error", "error": "The run stopped before it finished (the server restarted?) — press ➕ again."}
    return st


def _say(**kw) -> None:
    st = _read(status_path(), {}) or {}
    st.update(kw)
    _write(status_path(), st)


# ---------------------------------------------------------------------------
# What every video still needs
# ---------------------------------------------------------------------------

def video_states() -> dict:
    """{video_id: {"catalogue", "preview", "vision", "scores", "timeline": bool}} for every video the app or the
    vision data knows — what it has. A video is in the 🤖 new matcher's pool once it has all five."""
    import footage_timeline as FT
    try:
        import catalogue_store as CS
        catalogue = set(CS.video_index())
    except Exception:              # noqa: BLE001 — a bare run without the app's caches
        catalogue = set()
    try:
        from library_common import proxy_path
    except Exception:              # noqa: BLE001
        proxy_path = None
    rep = FT.status_report()
    built = set(rep["built"])
    with_scores = built | set(rep["ready"]) | set(rep["no_vision"])
    with_vision = set(rep["ready"]) | set(rep["no_scores"])
    vroot = Path(rep["vision_root"]) if rep["vision_root"] else None
    if vroot is not None:
        with_vision |= {v for v in built if (vroot / v / "vision.json").exists()}
    else:
        with_vision |= built
    out = {}
    for vid in sorted(catalogue | with_vision | with_scores, key=str.lower):
        try:
            preview = bool(proxy_path and proxy_path(vid).exists())
        except Exception:          # noqa: BLE001
            preview = False
        out[vid] = {"catalogue": vid in catalogue, "preview": preview, "vision": vid in with_vision,
                    "scores": vid in with_scores, "timeline": vid in built}
    return out


def first_need(state: dict) -> str | None:
    """The first thing a video is still missing (None: it has everything)."""
    return next((k for k in NEEDS if not state.get(k)), None)


def summary(states: dict = None) -> dict:
    """{"total", "ready" (in the new matcher's pool), "waiting": {need: [video ids]}, "no_preview": [video ids]} —
    each waiting video under the FIRST thing it lacks, since that is what has to happen next for it. no_preview:
    videos in the catalogue without a 480p preview copy on this machine (they still match; they render slower)."""
    states = video_states() if states is None else states
    waiting = {}
    for vid, s in states.items():
        need = first_need(s)
        if need:
            waiting.setdefault(need, []).append(vid)
    return {"total": len(states), "ready": sum(1 for s in states.values() if first_need(s) is None), "waiting": waiting,
            "no_preview": [v for v, s in states.items() if s.get("catalogue") and not s.get("preview")]}


# ---------------------------------------------------------------------------
# The three steps
# ---------------------------------------------------------------------------

def scores_model() -> str | None:
    """The student model the scores in use were made with — new videos are scored with the same one."""
    import footage_timeline as FT
    return (_read(FT.scores_dir() / "index.json", {}) or {}).get("model") or None


def _sync() -> str:
    try:
        from auto_sync import run_sync
        r = run_sync(lambda i, n, name: _say(detail=f"{i} of {n} · {name}"))
    except Exception as e:         # noqa: BLE001 — no credentials on this machine, Drive unreachable…
        return f"☁️ Drive sync skipped: {e}"
    if r.get("busy"):
        return "☁️ Another Drive sync was already running — carried on without waiting for it."
    note = f"☁️ Synced {r.get('synced', 0)} file(s) from Drive"
    if r.get("errors"):
        note += f" — {len(r['errors'])} problem(s): {str(r['errors'][0])[:160]}"
    return note


def _score_missing(model: str, log: Path) -> tuple:
    """Run the export for the videos without scores. Returns (exit code, the last lines it printed)."""
    script = HERE / "deploy" / "student.sh"
    with open(log, "ab") as fh:
        fh.write(f"\n=== {time.strftime('%F %T')} ➕ export --name {model} --missing\n".encode())
        fh.flush()
        rc = subprocess.call(["bash", str(script), "export", "--name", model, "--missing"], stdout=fh,
                             stderr=subprocess.STDOUT, cwd=str(HERE))
    try:
        tail = Path(log).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
    except OSError:
        tail = []
    return rc, tail


def _scores(log: Path) -> str:
    import footage_timeline as FT
    todo = FT.status_report()["no_scores"]
    if not todo:
        return "🎓 No video is waiting for tag scores."
    model = scores_model()
    if not model:
        return (f"🎓 {len(todo)} video(s) need tag scores, but no export has been made yet — run "
                f"deploy/student.sh export --name <model> once on the server.")
    if os.name == "nt" or not (HERE / "deploy" / "student.sh").exists():
        return f"🎓 {len(todo)} video(s) need tag scores — that step runs on the server only."
    _say(detail=f"{len(todo)} video(s) with student {model}")
    rc, tail = _score_missing(model, log)
    left = FT.status_report()["no_scores"]
    if rc != 0:
        return (f"🎓 Scoring with {model} stopped (exit {rc}); {len(left)} video(s) still without scores. "
                f"Last lines: {' / '.join(t.strip() for t in tail[-3:])[:300]}")
    note = f"🎓 Scored {len(todo) - len(left)} new video(s) with student {model}"
    if left:
        hint = next((t.strip() for t in tail if "left out" in t), "")
        note += f" — {len(left)} left out" + (f": {hint[:200]}" if hint else "")
    return note


def _timeline() -> tuple:
    import footage_timeline as FT
    if FT.read_status().get("state") == "running":
        return "🧭 A timeline build was already running — it will pick the new videos up.", 0
    try:
        r = FT.build(progress=lambda msg: _say(detail=str(msg).strip()[:120]))
    except RuntimeError as e:      # no vision data folder on this machine, nothing to fit the embedding space on…
        return f"🧭 Timelines not built: {e}", 0
    built, skipped = len(r.get("built") or []), r.get("skipped") or {}
    note = f"🧭 Built {built} timeline(s)"
    if skipped:
        why = {}
        for reason in skipped.values():
            why[reason] = why.get(reason, 0) + 1
        note += " — skipped: " + "; ".join(f"{n} × {w}" for w, n in sorted(why.items(), key=lambda kv: -kv[1])[:3])
    return note, built


def run() -> dict:
    """The whole job, in this process. Returns the final status."""
    before = summary()
    log = _catalogue_dir() / ".add_videos.log"
    _write(status_path(), {"state": "running", "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "step": "sync", "detail": "", "notes": [], "error": None, "finished": None,
                           "before": {"ready": before["ready"], "total": before["total"]}})
    notes = []
    try:
        notes.append(_sync())
        _say(step="scores", detail="", notes=notes)
        try:
            from heavy_slot import heavy_slot
        except Exception:          # noqa: BLE001
            heavy_slot = None
        if heavy_slot is not None:
            with heavy_slot("➕ new videos", on_wait=lambda h: _say(detail=f"waiting for {h.get('who', 'another job')} to finish")):
                notes.append(_scores(log))
                _say(step="timeline", detail="", notes=notes)
                note, _ = _timeline()
        else:
            notes.append(_scores(log))
            _say(step="timeline", detail="", notes=notes)
            note, _ = _timeline()
        notes.append(note)
        after = summary()
        _say(state="done", step=None, detail="", notes=notes, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
             result={"ready": after["ready"], "total": after["total"], "added": after["ready"] - before["ready"],
                     "waiting": {k: len(v) for k, v in after["waiting"].items()}})
    except Exception as e:         # noqa: BLE001 — recorded for the app, never left as "running"
        _say(state="error", error=f"{type(e).__name__}: {e}", notes=notes, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    return read_status()


def start() -> str:
    """Start the job in the background (low priority), for the app's button. Returns a message."""
    if read_status().get("state") == "running":
        return "Already running."
    args = [sys.executable, str(HERE / "add_videos.py"), "run"]
    if os.name != "nt":
        args = ["nice", "-n", "10"] + args
    env = dict(os.environ)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "2")
    log = _catalogue_dir() / ".add_videos.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        proc = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE), env=env, start_new_session=True)
    _write(status_path(), {"state": "running", "pid": proc.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "step": "sync", "detail": "starting…", "notes": [], "error": None, "finished": None})
    return "Started."


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        st_ = run()
        print("\n".join(st_.get("notes") or []))
        if st_.get("state") == "error":
            print("Stopped:", st_.get("error"))
            sys.exit(1)
        r_ = st_.get("result") or {}
        print(f"{r_.get('ready')} of {r_.get('total')} videos are in the new matcher's pool ({r_.get('added', 0):+d}).")
    else:
        sm = summary()
        print(f"{sm['ready']} of {sm['total']} videos are in the new matcher's pool.")
        for need in NEEDS:
            vids = sm["waiting"].get(need) or []
            if vids:
                print(f"  {len(vids)} waiting for their {NEED_LABEL[need]} ({NEED_WHERE[need]}): "
                      + ", ".join(vids[:4]) + (" …" if len(vids) > 4 else ""))
