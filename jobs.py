"""
⏳ Background recommendation searches — a small queue and worker, the same shape
as audio analysis's (audio_analysis.py), sharing its one heavy slot (heavy_slot.py):
only one heavy job runs at a time, and queued audio analysis always goes first.

Jobs
  track      — search one track (the five plans) and store them (rec_library.run_track)
  spotlight  — 🆕 a new video's spotlight (rec_library.run_spotlight)

Nightly (systemd timer, deploy/install_timers.sh) — "only changed":
  every track whose stored results no longer match the library (never searched, or
  videos / ranges / actors / roles / the track / its story map changed since), then a
  spotlight for each new video whose spotlight is out of date.

Usage:
    python jobs.py --worker      work through the queue
    python jobs.py --nightly     queue what's changed, then work through it
    python jobs.py --status      print the queue
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import rec_library as RL

JOBS_DIR = RL.BASE_DIR / "jobs"
STOP_WAIT_SEC = 5


def _paths() -> dict:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return {"queue": JOBS_DIR / "queue.json", "status": JOBS_DIR / "status.json", "lock": JOBS_DIR / "worker.pid",
            "pause": JOBS_DIR / "paused", "nightly": JOBS_DIR / "nightly.json", "edit": JOBS_DIR / ".edit.lock"}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


class _Edit:
    """Lock the queue files while reading and rewriting them (page and worker both edit)."""

    def __enter__(self):
        self.f = open(_paths()["edit"], "a+")
        try:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_EX)
        except ImportError:
            pass
        return self

    def __exit__(self, *exc):
        try:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_UN)
        except ImportError:
            pass
        self.f.close()


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------

def job_id(job: dict) -> str:
    if job["kind"] == "spotlight":
        return f"spotlight:{job['video_id']}"
    return f"track:{job['track_id']}:{job.get('n_videos', RL.DEFAULT_VIDEOS)}:{int(bool(job.get('sequential')))}"


def job_label(job: dict) -> str:
    if job["kind"] == "spotlight":
        return f"🆕 {job['video_id']}"
    extra = []
    if job.get("n_videos", RL.DEFAULT_VIDEOS) != RL.DEFAULT_VIDEOS:
        extra.append(f"{job['n_videos']} videos")
    if job.get("sequential"):
        extra.append("in order")
    return f"🎵 {job['track_id']}" + (f" ({', '.join(extra)})" if extra else "")


def read_queue() -> list:
    return _read_json(_paths()["queue"], [])


def read_status() -> dict:
    return _read_json(_paths()["status"], {})


def enqueue(jobs: list, why: str = "") -> int:
    """jobs: [{"kind": "track", "track_id", "n_videos"?, "sequential"?} | {"kind": "spotlight", "video_id"}].
    Skips ones already queued. Returns how many were added."""
    p = _paths()
    added = 0
    with _Edit():
        queue = _read_json(p["queue"], [])
        status = _read_json(p["status"], {})
        have = {job_id(j) for j in queue}
        for j in jobs:
            j = dict(j, why=j.get("why") or why)
            jid = job_id(j)
            if jid in have:
                continue
            queue.append(j)
            have.add(jid)
            status[jid] = {"state": "queued", "label": job_label(j), "msg": "waiting", "progress": 0.0,
                           "updated": time.time()}
            added += 1
        _write_json(p["queue"], queue)
        _write_json(p["status"], status)
    return added


def cancel_queued() -> int:
    """Remove every queued (not running) job."""
    p = _paths()
    with _Edit():
        status = _read_json(p["status"], {})
        running = {k for k, v in status.items() if v.get("state") == "running"}
        queue = _read_json(p["queue"], [])
        keep = [j for j in queue if job_id(j) in running]
        dropped = {job_id(j) for j in queue} - running
        _write_json(p["queue"], keep)
        _write_json(p["status"], {k: v for k, v in status.items() if k not in dropped})
    return len(dropped)


def clear_finished() -> None:
    p = _paths()
    with _Edit():
        status = _read_json(p["status"], {})
        _write_json(p["status"], {k: v for k, v in status.items() if v.get("state") in ("queued", "running")})


def worker_running() -> bool:
    try:
        os.kill(int(_paths()["lock"].read_text().strip()), 0)
        return True
    except Exception:
        return False


def is_paused() -> bool:
    return _paths()["pause"].exists()


def pause() -> None:
    """Finish the job in progress, then stop; the rest stay queued."""
    _paths()["pause"].write_text("1")


def resume() -> str:
    _paths()["pause"].unlink(missing_ok=True)
    return start_worker()


def stop_now() -> None:
    """Stop at once: the job in progress goes back to the front of the queue."""
    import signal as _signal
    pause()
    try:
        os.kill(int(_paths()["lock"].read_text().strip()), _signal.SIGTERM)
    except Exception:
        pass


def start_worker() -> str:
    """Start the worker if it isn't running — a separate systemd unit when possible (survives
    app restarts), else a detached process."""
    if worker_running():
        return "already running"
    if is_paused():
        return "paused"
    here = Path(__file__).resolve().parent
    cmd = [sys.executable, str(here / "jobs.py"), "--worker"]
    if shutil.which("systemd-run") and hasattr(os, "geteuid") and os.geteuid() == 0:
        unit = f"mediaplanner-recs-{int(time.time())}"
        r = subprocess.run(["systemd-run", f"--unit={unit}", "--collect", "--nice=10",
                            f"--working-directory={here}", *cmd], capture_output=True, text=True)
        if r.returncode == 0:
            return "started"
    subprocess.Popen(cmd, cwd=str(here), start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "started"


def nightly_status() -> dict:
    """{"started", "finished", "queued_tracks", "queued_spotlights"} of the last nightly run."""
    return _read_json(_paths()["nightly"], {})


# ---------------------------------------------------------------------------
# What's changed
# ---------------------------------------------------------------------------

def changed_jobs() -> list:
    """Track jobs for every track whose stored results (default settings) are missing or out of
    date, then spotlight jobs for new videos whose spotlight is missing or out of date."""
    import story_map as STM
    jobs = []
    tracks = {}
    for tid in RL.list_tracks():
        try:
            tracks[tid] = RL.load_track(tid)
        except (OSError, ValueError):
            continue
        STM.ensure_default(tid, tracks[tid])    # 🎭 tracks analysed before default maps existed get one
    state = RL.library_state()
    for tid, track in tracks.items():
        story, _ = RL.story_for(tid, track)
        sig = RL.signature(tid, RL.DEFAULT_VIDEOS, False, story, state)
        if RL.load_results(tid, RL.DEFAULT_VIDEOS, False, sig) is None:
            jobs.append({"kind": "track", "track_id": tid, "n_videos": RL.DEFAULT_VIDEOS, "sequential": False})
    for vid, _ in RL.new_videos():
        jobs.append({"kind": "spotlight", "video_id": vid})   # checked again when it runs (needs fresh fits)
    return jobs


def queue_changed(why: str = "changed") -> int:
    return enqueue(changed_jobs(), why=why)


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _audio_waiting() -> bool:
    """Audio analysis has jobs queued and isn't paused (it goes first)."""
    from config import AUDIO_DIR
    d = AUDIO_DIR / ".analysis"
    return bool(_read_json(d / "queue.json", [])) and not (d / "paused").exists()


def run_job(job: dict, say) -> None:
    if job["kind"] == "track":
        RL.run_track(job["track_id"], int(job.get("n_videos", RL.DEFAULT_VIDEOS)), bool(job.get("sequential")),
                     progress=say)
    elif job["kind"] == "spotlight":
        fits = RL.track_fits(job["video_id"])
        sig = RL.spotlight_signature(job["video_id"], fits)
        if RL.load_spotlight(job["video_id"], sig) is None:
            RL.run_spotlight(job["video_id"], progress=say)
    else:
        raise ValueError(f"unknown job kind {job['kind']!r}")


def run_worker(poll_sec: float = STOP_WAIT_SEC) -> None:
    p = _paths()
    try:
        os.nice(10)
    except Exception:
        pass
    if worker_running():
        return
    p["lock"].write_text(str(os.getpid()))

    class _Stopped(BaseException):
        pass

    def _on_term(_sig, _frame):
        raise _Stopped()

    import signal as _signal
    try:
        _signal.signal(_signal.SIGTERM, _on_term)
    except ValueError:           # not the main thread (tests)
        pass
    from heavy_slot import heavy_slot
    try:
        while True:
            queue = read_queue()
            if not queue or p["pause"].exists():
                break
            job = queue[0]
            jid = job_id(job)
            last = [0.0]

            def say(frac, msg="", _jid=jid, _label=job_label(job), force=False):
                if isinstance(frac, str):          # say("message")
                    frac, msg = None, frac
                now = time.time()
                if not force and now - last[0] < 1.5:
                    return
                last[0] = now
                with _Edit():
                    st = _read_json(p["status"], {})
                    cur = st.get(_jid, {})
                    st[_jid] = {"state": "running", "label": _label, "msg": msg,
                                "progress": cur.get("progress", 0.0) if frac is None else round(float(frac), 3),
                                "started": cur.get("started") or now, "updated": now}
                    _write_json(p["status"], st)

            # Audio analysis goes first.
            while _audio_waiting() and not p["pause"].exists():
                say(None, "waiting for audio analysis to finish", force=True)
                time.sleep(poll_sec)
            if p["pause"].exists():
                break
            say(0.0, "starting", force=True)
            try:
                with heavy_slot("recommendations", on_wait=lambda _h: say(None, "waiting for audio analysis",
                                                                          force=True)):
                    run_job(job, say)
                state, msg = "done", "finished"
            except _Stopped:
                with _Edit():
                    st = _read_json(p["status"], {})
                    st[jid] = {"state": "queued", "label": job_label(job), "msg": "stopped — will restart on resume",
                               "progress": 0.0, "updated": time.time()}
                    _write_json(p["status"], st)
                break
            except Exception as e:
                traceback.print_exc()
                state, msg = "error", str(e)[-300:]
            with _Edit():
                st = _read_json(p["status"], {})
                st[jid] = {"state": state, "label": job_label(job), "msg": msg, "progress": 1.0,
                           "updated": time.time()}
                _write_json(p["status"], st)
                _write_json(p["queue"], [j for j in _read_json(p["queue"], []) if job_id(j) != jid])
    finally:
        try:
            p["lock"].unlink()
        except Exception:
            pass


def run_nightly() -> dict:
    """Queue what's changed and work through it (the systemd timer runs this)."""
    p = _paths()
    started = time.time()
    jobs = changed_jobs()
    n = enqueue(jobs, why="nightly")
    info = {"started": started, "finished": None,
            "queued_tracks": sum(1 for j in jobs if j["kind"] == "track"),
            "queued_spotlights": sum(1 for j in jobs if j["kind"] == "spotlight"), "added": n}
    _write_json(p["nightly"], info)
    if not is_paused():
        run_worker()
    info["finished"] = time.time()
    _write_json(p["nightly"], info)
    return info


def _cli() -> None:
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.nightly:
        print(run_nightly())
    elif args.worker:
        run_worker()
    elif args.status:
        print(json.dumps({"queue": [job_label(j) for j in read_queue()], "status": read_status(),
                          "running": worker_running(), "paused": is_paused(), "nightly": nightly_status()},
                         indent=1, default=str))
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
