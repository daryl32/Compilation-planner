"""
One heavy job at a time on the server.

Audio analysis (audio_analysis.py) and recommendation searches (jobs.py) each have
their own queue, but both take the same "heavy slot" before working on a job, so
they never fight over the server's 2 CPUs and 4 GB of RAM. Audio goes first: the
recommendation worker waits between tracks while any audio is queued.

On Windows (the laptop) there's no fcntl, so the slot is a no-op there.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

from config import CATALOGUE_DIR

LOCK_FILE = CATALOGUE_DIR / ".heavy_job.lock"
HOLDER_FILE = CATALOGUE_DIR / ".heavy_job.json"


def holder() -> dict | None:
    """{"who", "pid", "since"} of the job holding the slot, or None."""
    try:
        h = json.loads(HOLDER_FILE.read_text())
        os.kill(int(h["pid"]), 0)
        return h
    except Exception:
        return None


@contextmanager
def heavy_slot(who: str, on_wait=None):
    """Hold the heavy slot for the duration of the block (waits for it if it's taken).
    on_wait(holder) is called once if it has to wait."""
    try:
        import fcntl
    except ImportError:          # Windows: no slot
        yield
        return
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCK_FILE, "a+")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if on_wait:
                on_wait(holder())
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            HOLDER_FILE.write_text(json.dumps({"who": who, "pid": os.getpid(), "since": time.time()}))
        except OSError:
            pass
        try:
            yield
        finally:
            try:
                HOLDER_FILE.unlink()
            except OSError:
                pass
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()
