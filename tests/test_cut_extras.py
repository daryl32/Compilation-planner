"""✂️ Cut editor extras: 🔒 locked cuts, 🎨 re-cutting a part in another style,
👻 ideas, ⚖️ A/B — the cut logic and the planner's controls around the editor."""

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.fixtures import load_track
from tests.apptest_util import errors, page_app
from core import cut_edits as CED
import core.styles as CSTY
from core.segmentation import build_segments

PAGE = "pages/1_Compilation_Planner.py"
AUDIO_STEP = "🎵 0. Audio Settings"


@pytest.fixture
def track():
    return load_track(DATA, "song_full")


def _style_cuts(track, style, pace=0):
    method, settings, _ = CSTY.style_settings(track, style, pace, 4)
    return CED.base_cut_times(build_segments(track, {"method": method, "beats_per_bar": 4, **settings}))


# ---------------------------------------------------------------------------
# The logic
# ---------------------------------------------------------------------------

def test_clean_locked_follows_the_cuts():
    assert CED.clean_locked([10.001, 30.0, 55.0], [5.0, 10.0, 20.0, 55.0]) == [10.0, 55.0]
    assert CED.clean_locked(None, [1.0]) == []


def test_merge_locked_keeps_locked_cuts_exactly():
    new = [4.0, 8.0, 12.1, 16.0, 20.0]
    out = CED.merge_locked(new, [12.0, 18.5], 30.0)
    assert 12.0 in out and 18.5 in out          # locked ones exactly where they were
    assert 12.1 not in out                       # a new cut right next to a locked one gives way
    assert {4.0, 8.0, 16.0, 20.0} <= set(out)


def test_recut_region_only_touches_the_part(track):
    dur = track["duration_sec"]
    cuts = _style_cuts(track, "cinematic")
    hyper = _style_cuts(track, "hyper")
    start, end = 30.0, 50.0
    locked = [t for t in cuts if start <= t < end][:1]
    out = CED.recut_region(cuts, hyper, start, end, dur, locked)
    assert [t for t in out if t < start] == [t for t in cuts if t < start]          # before: untouched
    assert [t for t in out if t >= end] == [t for t in cuts if t >= end]            # after: untouched
    inside = [t for t in out if start <= t < end]
    assert len(inside) > len([t for t in cuts if start <= t < end])                # busier in the part
    assert all(t in out for t in locked)                                             # 🔒 kept


def test_ghost_cuts_skip_existing_and_edges():
    ghosts = CED.ghost_cuts([1.0, 1.05, 2.0, 5.0, 9.98, 0.01], [2.05, 5.0], 10.0)
    assert ghosts == [1.0]


def test_cut_stats():
    s = CED.cut_stats([2.0, 6.0], 10.0)
    assert s == {"segments": 3, "average": pytest.approx(10 / 3), "shortest": 2.0, "longest": 4.0}


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

def _run(at, **state):
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not errors(at), errors(at)
    return at


def test_new_settings_keep_locked_cuts():
    at = page_app(PAGE)
    edits = {"track_id": "song_full", "cuts": [8.0, 16.0, 24.0], "locked": [16.0], "base_sig": "old settings"}
    _run(at, track_id="song_full", ui_section=AUDIO_STEP, cut_edits=edits)
    btn = [b for b in at.button if b.label.startswith("♻️ New settings + my 1 🔒")]
    assert btn, [b.label for b in at.button]
    btn[0].click().run()
    assert not errors(at), errors(at)
    e = at.session_state["cut_edits"]
    assert 16.0 in e["cuts"] and e["locked"] == [16.0]
    assert len(e["cuts"]) > 3                      # the new settings' cuts around it
    assert not any("changed the cutting settings" in w.value for w in at.warning)


def test_recut_a_part_in_another_style():
    at = page_app(PAGE)
    _run(at, track_id="song_full", ui_section=AUDIO_STEP)
    before = list(at.session_state["cut_edits"]["cuts"]) if "cut_edits" in at.session_state else None
    start, end = at.select_slider(key="ced_region").value      # the default part: the first 8 bars
    assert 0.0 == start < end
    at.selectbox(key="ced_rstyle").set_value("hyper").run()
    at.button(key="ced_recut").click().run()
    assert not errors(at), errors(at)
    e = at.session_state["cut_edits"]
    assert e["track_id"] == "song_full" and e["cuts"] != before
    assert any("Re-cut" in t.value for t in at.toast)
    tail_before = [t for t in (before or []) if t >= end]
    if before:
        assert [t for t in e["cuts"] if t >= end] == tail_before                     # rest untouched


def test_keep_as_a_then_swap():
    at = page_app(PAGE)
    _run(at, track_id="song_full", ui_section=AUDIO_STEP)
    at.button(key="ced_ab_keep").click().run()
    a_cuts = list(at.session_state["cut_ab"]["A"])
    _run(at, cut_edits={"track_id": "song_full", "cuts": [10.0, 20.0, 30.0], "base_sig": "x"})
    assert any("cut(s) differ" in m.value for m in at.markdown)
    # keep the settings signature current so the 'settings changed' prompt doesn't get in the way
    at.button(key="ced_ab_swap").click().run()
    assert not errors(at), errors(at)
    assert at.session_state["cut_ab"]["A"] == [10.0, 20.0, 30.0]
    now = at.session_state.get("cut_edits")
    assert now is None or now["cuts"] == a_cuts      # back to A (== the method's cuts → no edit stored)
    at.button(key="ced_ab_forget").click().run()
    assert "cut_ab" not in at.session_state


def test_ideas_toggle_runs():
    at = page_app(PAGE)
    _run(at, track_id="song_full", ui_section=AUDIO_STEP, ced_ideas=True, cut_style="beat", cut_style_pace=0)
    assert at.toggle(key="ced_ideas").value is True
