"""🎨 Cutting styles: valid settings for every style / pace / track, the pace
really changes the rate, and the Step 0 row sets the method and dials."""

import pytest

from tests.support import DATA
from tests.fixtures import load_track, TRACKS
from tests.apptest_util import errors, page_app
from core.segmentation import build_segments, SEG_METHOD_KEYS
from core.styles import STYLES, style_settings

RANGES = {  # the planner's slider ranges
    "sc_bars_per_cut": (0.5, 8.0), "sc_follow_energy": (0.0, 1.0), "sc_min_sec": (0.3, 8.0), "sc_max_sec": (0.0, 30.0),
    "change_threshold": (0.02, 0.5), "hit_threshold": (0.5, 1.0), "min_segment_sec": (0.3, 8.0),
    "max_segment_sec": (0.0, 30.0), "sm_pace": (-2, 2), "sm_contrast": (0.0, 2.0), "sm_build_strength": (0.0, 2.0),
}


@pytest.mark.parametrize("tid", sorted(TRACKS))
@pytest.mark.parametrize("style", sorted(STYLES))
def test_every_style_every_pace(tid, style):
    tr = load_track(DATA, tid)
    counts = []
    for pace in (-2, -1, 0, 1, 2):
        method, s, notes = style_settings(tr, style, pace)
        assert set(SEG_METHOD_KEYS[method]) <= set(s), (style, pace)
        for k, (lo, hi) in RANGES.items():
            if k in s:
                assert lo <= s[k] <= hi, (style, pace, k, s[k])
        segs = build_segments(tr, {"method": method, "beats_per_bar": 4, **s})
        assert segs and abs(segs[-1]["end"] - tr["duration_sec"]) < 0.05
        counts.append(len(segs))
        assert notes and STYLES[style]["label"] in notes[0]
    assert counts == sorted(counts), (tid, style, counts)       # busier never cuts less


def test_styles_differ_in_pace():
    tr = load_track(DATA, "song_full")
    n = {st: len(build_segments(tr, {"method": m, "beats_per_bar": 4, **s}))
         for st in STYLES for m, s, _ in [style_settings(tr, st, 0)]}
    assert n["cinematic"] < n["vocal"] < n["beat"] < n["hyper"]


def test_unknown_style():
    with pytest.raises(ValueError):
        style_settings(load_track(DATA, "song_full"), "nope")


def test_style_row_sets_method_and_dials_and_marks_custom_on_edit():
    at = page_app("pages/1_Compilation_Planner.py")
    for k, v in {"track_id": "song_full", "ui_section": "🎵 0. Audio Settings"}.items():
        at.session_state[k] = v
    at.run()
    assert not errors(at), errors(at)
    at.selectbox(key="cut_style").set_value("cinematic").run()
    assert not errors(at), errors(at)
    assert at.session_state["segmentation_method"] == "Scored cuts (all signals)"
    assert at.session_state["sc_bars_per_cut"] == 4.0 and at.session_state["audio_style"] == "Custom"
    n_normal = int([c.value for c in at.caption if "segments**" in c.value][0].split("**")[1].split()[0])
    at.select_slider(key="cut_style_pace").set_value(2).run()
    n_busy = int([c.value for c in at.caption if "segments**" in c.value][0].split("**")[1].split()[0])
    assert n_busy > n_normal
    # moving a dial by hand → My own settings
    at.slider(key="sc_follow_energy").set_value(0.9).run()
    assert at.session_state["cut_style"] == "custom"
