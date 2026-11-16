"""Shared helpers for the ⏱️ Edge Mode tests (tests/test_edge_*.py): the test videos, sound level / pixel / length
probes, the plan check, a synthetic energy map, and waiting for a background compile."""
from __future__ import annotations

import shutil  # noqa: F401
import subprocess
import wave  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np
import pytest  # noqa: F401

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.test_story_matcher import _setup as _timelines  # noqa: F401

VIDS = ["vid_a", "vid_b", "vid_c", "vid_f"]


def _check_plan(plan, total):
    tl = plan["timeline"]
    assert tl[0]["track_time"][0] == 0.0 and abs(tl[-1]["track_time"][1] - total) < 1e-6
    for a, b in zip(tl, tl[1:]):
        assert abs(a["track_time"][1] - b["track_time"][0]) < 1e-6
    for e in tl:
        d = e["track_time"][1] - e["track_time"][0]
        assert d > 0 and e["edge_profile"] in ("wait", "slow", "medium", "fast", "extra", "stop")
        for s in e["scenes"]:
            assert sum(link["clip_duration_sec"] for link in s["chain"]) <= d + 0.05
            assert all("source_path" in link for link in s["chain"])
    assert [e["segment_index"] for e in tl] == list(range(len(tl)))


def _level(path, a, b):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(a), "-t", str(b - a), "-i", str(path), "-vn", "-ac", "1",
                          "-ar", "8000", "-f", "s16le", "-"], capture_output=True, check=True)
    x = np.frombuffer(out.stdout, "<i2").astype(float) / 32768
    return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0


def _pixel(path, t, x, y):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-vf", f"crop=2:2:{x}:{y},scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True)
    return tuple(out.stdout[:3])


def _wait_job(at):
    """⏳ The compile runs on the server (mix_jobs): wait for it, then let the page pick up its result."""
    import mix_jobs
    import planner.edge as PE
    mix_jobs.wait([PE.JOB_KEY], timeout=300)
    at.run()


def _seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def E_CARD_LEAD():
    import edge_render
    return edge_render.CARD_LEAD


def _emap(levels, bar=2.0):
    import numpy as np
    bars = np.arange(len(levels)) * bar
    return {"bars": bars, "ends": bars + bar, "energy": np.asarray(levels, float), "duration": len(levels) * bar}
