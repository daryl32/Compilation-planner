"""⏳ Background work for the 🎚️ Mix page — renders, Drive downloads, live-preview stems.

Each job runs in a server thread, not in the page's script run, so it carries on when the iPad sleeps or
the page reloads; the page only polls it. Jobs are keyed by what they make (a preview file, a song file,
a live window…): asking for the same thing again joins the job already running — or picks up its finished
result — from any session. Finished jobs are forgotten after KEEP_SEC. No Streamlit in here.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

WORKERS = 3                 # renders / downloads / live windows at once (each render reads several songs itself)
KEEP_SEC = 3600

_POOL = ThreadPoolExecutor(WORKERS, thread_name_prefix="mixjob")
_LOCK = threading.Lock()
_JOBS: "dict[str, Job]" = {}


class Job:
    """state: "queued" → "running" → "done" | "error". The work function gets the job and may set
    .frac (0…1) and .msg as it goes."""

    def __init__(self, key: str, label: str):
        self.key, self.label = key, label
        self.state, self.frac, self.msg = "queued", 0.0, ""
        self.result, self.error = None, None
        self.started, self.finished = time.time(), None

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")

    def progress(self, frac: float = None, msg: str = None) -> None:
        if frac is not None:
            self.frac = max(self.frac, min(1.0, float(frac)))
        if msg is not None:
            self.msg = str(msg)


def _prune() -> None:
    now = time.time()
    for k in [k for k, j in _JOBS.items() if j.finished and now - j.finished > KEEP_SEC]:
        _JOBS.pop(k, None)


def start(key: str, label: str, fn, *args, retry_failed: bool = True, **kwargs) -> Job:
    """Run fn(job, *args, **kwargs) in the background (its return value becomes job.result) — unless a job
    with this key is running or finished fine, which is returned instead. A failed job is started again
    (retry_failed)."""
    with _LOCK:
        _prune()
        j = _JOBS.get(key)
        if j is not None and (j.active or j.state == "done" or not retry_failed):
            return j
        j = Job(key, label)
        _JOBS[key] = j

    def run():
        j.state = "running"
        try:
            j.result = fn(j, *args, **kwargs)
            j.frac, j.state = 1.0, "done"
        except Exception as e:                          # shown on the page; the server keeps going
            j.error, j.state = f"{e}", "error"
            j.msg = traceback.format_exc(limit=3)
        finally:
            j.finished = time.time()

    _POOL.submit(run)
    return j


def get(key: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(key)


def forget(key: str) -> None:
    with _LOCK:
        _JOBS.pop(key, None)


def wait(keys, timeout: float = 600) -> None:
    """Block until these jobs have finished (tests; the page polls instead)."""
    end = time.time() + timeout
    while time.time() < end:
        js = [get(k) for k in keys]
        if all(j is None or not j.active for j in js):
            return
        time.sleep(0.05)
