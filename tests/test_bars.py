"""📏 core/bars.py — a complete bar grid from patchy bar lines (23 Oct 2026: 28 of 52 tracks had
holes of 2–40 bars, so the Mix's fine-tune jumped from 0:33 to 1:43)."""

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)
from core import bars as BARS


def _beats(n=400, step=0.5, start=0.3):
    return np.round(start + step * np.arange(n), 3)


def test_irregular():
    bar = 2.0
    good = np.arange(0.3, 200, bar)
    assert not BARS.irregular(good, bar)
    holes = np.r_[good[:20], good[60:]]
    assert BARS.irregular(holes, bar)
    assert BARS.irregular(good[::2], bar)                # every other bar
    assert BARS.irregular([1.0], bar)


def test_regular_bars_fills_holes_and_keeps_beat_one():
    beats = _beats()
    true = beats[1::4]                                   # beat 1 is the 2nd beat
    patchy = np.r_[true[:10], true[40:45], true[80:]]   # holes of 30 and 35 bars
    out = np.array(BARS.regular_bars(beats, patchy, 4))
    assert np.allclose(out, true)
    assert BARS.regular_bars(beats, true[::2], 4) == list(true)      # every other bar → every bar


def test_phase_shift_is_followed():
    beats = _beats(512)
    first = beats[0:256:4]                               # beat 1 on beat 0 …
    second = beats[258::4]                               # … then on beat 2 (two beats added)
    out = BARS.regular_bars(beats, np.r_[first, second], 4)
    assert out[0] == beats[0] and beats[258] in out and beats[256] not in out
    gaps = np.diff(out)
    assert gaps.min() >= 0.6 * 2.0 and gaps.max() <= 1.6 * 2.0


def test_no_votes_uses_the_fallback():
    beats = _beats(40)
    assert BARS.regular_bars(beats, [], 4) == []
    assert BARS.regular_bars(beats, [], 4, fallback_phase=2) == list(beats[2::4])


def test_planner_and_mix_bar_lines_have_no_holes():
    import cut_scoring as CS
    from core import mixing as MX
    beats = _beats(360, 60 / 120)
    db = beats[::4]
    t = {"duration_sec": float(beats[-1] + 0.5), "bpm": 120.0, "beat_times": beats.tolist(),
         "analysis": {"beats_per_bar": 4, "downbeats": np.r_[db[:10], db[50:55], db[80:]].tolist()}}
    for lines in (CS.bar_lines(t, 4), MX.grid(t)["bars"]):
        gaps = np.diff(lines)
        assert gaps.max() < 1.6 * 2.0, gaps.max()
        assert len(lines) >= len(db) - 1
