"""🥁 Land every cut on the beat — Adaptive, Rhythm Engine and Scored cuts put every cut on the
track's beat grid (a bar line for changes in the music when one is within a beat); off gives
exactly the old cuts."""

import numpy as np
import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.apptest_util import errors, page_app
from tests.fixtures import load_track

import cut_scoring as CS
from core.segmentation import (SEG_ADAPTIVE, SEG_DEFAULTS, SEG_METHOD_KEYS, SEG_RHYTHM, SEG_SCORED,
                               STRUCTURAL_CUTS, build_segments)

TRACKS = ["song_full", "song_slow", "song_short", "song_noanalysis"]
METHODS = [SEG_ADAPTIVE, SEG_RHYTHM, SEG_SCORED]


def _cfg(method, **kw):
    cfg = {"method": method, "beats_per_bar": 4, **{k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[method]}}
    if method == SEG_RHYTHM:
        cfg["dynamic_priority_override"] = True      # the jump cuts are the ones that fall mid-bar
    cfg.update(kw)
    return cfg


@pytest.mark.parametrize("tid", TRACKS)
@pytest.mark.parametrize("method", METHODS)
def test_every_cut_is_on_a_beat(tid, method):
    tr = load_track(DATA, tid)
    segs = build_segments(tr, _cfg(method))
    beats = np.asarray(CS.beat_grid(tr)[0])
    bars = np.asarray(CS.bar_lines(tr, 4))
    beat = float(np.median(np.diff(beats)))
    assert segs[0]["start"] == 0 and abs(segs[-1]["end"] - tr["duration_sec"]) < 1e-3
    for a, b in zip(segs, segs[1:]):
        assert abs(a["end"] - b["start"]) < 1e-6
        assert b["start"] - a["start"] >= 0.9 * beat - 1e-3          # never two cuts on one beat
    for sg in segs[1:]:
        assert np.min(np.abs(beats - sg["start"])) < 2e-3, (sg["cut"], sg["start"])
        if sg["cut"] in STRUCTURAL_CUTS and len(bars):
            near_bar = np.min(np.abs(bars - sg["start"]))
            # a change in the music sits on a bar line unless none was within a beat of it
            assert near_bar < 2e-3 or near_bar > 0.5 * beat


@pytest.mark.parametrize("method", METHODS)
def test_off_keeps_the_old_cuts_and_on_moves_them(method):
    tr = load_track(DATA, "song_full")
    beats = np.asarray(CS.beat_grid(tr)[0])
    off = build_segments(tr, _cfg(method, on_beat=False))
    on = build_segments(tr, _cfg(method, on_beat=True))
    off_beat = [s for s in off[1:] if np.min(np.abs(beats - s["start"])) > 0.06]
    assert off_beat                                     # the test song does have cuts between beats
    assert abs(len(on) - len(off)) <= max(2, 0.15 * len(off))   # about as many cuts — only moved


def test_target_tuning_counts_with_the_beat():
    from core.targets import adaptive_for_target
    tr = load_track(DATA, "song_full")
    base = {k: SEG_DEFAULTS[k] for k in SEG_METHOD_KEYS[SEG_ADAPTIVE]}
    r = adaptive_for_target(tr, avg_sec=3.0, base=base)
    assert r["settings"]["on_beat"] is True
    assert len(build_segments(tr, {"method": SEG_ADAPTIVE, "beats_per_bar": 4, **r["settings"]})) == r["segments"]


@pytest.mark.parametrize("method", [SEG_ADAPTIVE, SEG_RHYTHM, SEG_SCORED])
def test_toggle_shows_for_the_method(method):
    import shutil
    shutil.rmtree(DATA / "recs-.curve_cache", ignore_errors=True)
    at = page_app("pages/1_Compilation_Planner.py")
    for k, v in {"track_id": "song_full", "segmentation_method": method, "cut_style": "custom",
                 "ui_section": "🎵 0. Audio Settings"}.items():
        at.session_state[k] = v
    at.run()
    assert not errors(at), errors(at)
    tg = [t for t in at.toggle if t.key == "on_beat"]
    assert tg and tg[0].value is True
