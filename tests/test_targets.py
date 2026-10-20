"""🎯 Aim for: Adaptive dials from a target segment count / clip length."""

import pytest

from tests.support import DATA
from tests.fixtures import load_track
from core.segmentation import SEG_DEFAULTS, build_track_segments
from core.targets import adaptive_for_target, CHANGE_MIN, CHANGE_MAX, HIT_MIN, HIT_MAX

BASE = {k: SEG_DEFAULTS[k] for k in ("change_window_secs", "snap_secs", "react_to_hits",
                                     "change_threshold", "hit_threshold")}


@pytest.mark.parametrize("tid", ["song_full", "song_noanalysis", "song_slow"])
@pytest.mark.parametrize("avg", [1.0, 2.0, 3.0, 5.0, 10.0])
def test_hits_target_and_reports_truthfully(tid, avg):
    tr = load_track(DATA, tid)
    r = adaptive_for_target(tr, avg_sec=avg, base=BASE)
    s = r["settings"]
    assert r["exact"], r["notes"]
    assert abs(r["segments"] - r["target"]) <= max(1, 0.05 * r["target"])
    # the dials it sets really give what it reports
    got = build_track_segments(tr, s["change_window_secs"], s["change_threshold"], s["min_segment_sec"],
                               s["react_to_hits"], s["hit_threshold"], s["snap_secs"], s["max_segment_sec"], 4,
                               on_beat=s["on_beat"])
    assert len(got) == r["segments"]
    # and fit the sliders
    assert CHANGE_MIN <= s["change_threshold"] <= CHANGE_MAX and HIT_MIN <= s["hit_threshold"] <= HIT_MAX
    assert 0.3 <= s["min_segment_sec"] <= 8.0 and 0.0 <= s["max_segment_sec"] <= 30.0
    assert round(s["change_threshold"], 2) == s["change_threshold"]


def test_segments_target_and_kept_settings():
    tr = load_track(DATA, "song_full")
    base = {**BASE, "react_to_hits": False, "snap_secs": 0.15, "change_window_secs": 1.5}
    r = adaptive_for_target(tr, segments=30, base=base)
    assert r["target"] == 30 and r["exact"]
    assert r["settings"]["react_to_hits"] is False and r["settings"]["snap_secs"] == 0.15
    assert r["settings"]["change_window_secs"] == 1.5


def test_more_cuts_for_shorter_clips():
    tr = load_track(DATA, "song_slow")
    counts = [adaptive_for_target(tr, avg_sec=a, base=BASE)["segments"] for a in (8, 4, 2, 1)]
    assert counts == sorted(counts)


def test_legacy_track_says_so():
    r = adaptive_for_target(load_track(DATA, "song_legacy"), avg_sec=3.0, base=BASE)
    assert any("older coarse analysis" in n for n in r["notes"])


def test_needs_a_target():
    with pytest.raises(ValueError):
        adaptive_for_target(load_track(DATA, "song_full"), base=BASE)


def test_aim_panel_sets_the_dials():
    from tests.apptest_util import errors, page_app
    at = page_app("pages/1_Compilation_Planner.py")
    for k, v in {"track_id": "song_full", "segmentation_method": "Adaptive (energy-change + hits)",
                 "ui_section": "🎵 0. Audio Settings", "aim_mode": "avg", "aim_avg_sec": 3.0}.items():
        at.session_state[k] = v
    at.run()
    assert not errors(at), errors(at)
    [b for b in at.button if b.label == "Set the dials"][0].click().run()
    assert not errors(at), errors(at)
    assert at.session_state["audio_style"] == "Custom"
    report = " ".join(c.value for c in at.caption)
    assert "Aimed for 32 segments" in report
    seg_line = [c.value for c in at.caption if "segments**" in c.value][0]
    n = int(seg_line.split("**")[1].split()[0])
    assert abs(n - 32) <= 2
