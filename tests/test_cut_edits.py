"""✂️ Hand-edited cuts: the cut logic, per-project saving, the 'ask first'
prompt when settings change, and edits flowing into matching and export."""

import json

import pytest

from tests.support import DATA, PLANNER, load_page
from tests.fixtures import load_track
from tests.apptest_util import errors, page_app
from core import cut_edits as CED
from core.segmentation import SEG_DEFAULTS, SEG_METHOD_KEYS, SEG_ADAPTIVE, SEG_SMART, build_segments


def _cfg(method=SEG_SMART, **over):
    return {"method": method, "beats_per_bar": 4,
            **{k: over.get(k, SEG_DEFAULTS[k]) for k in SEG_METHOD_KEYS[method]}}


@pytest.fixture
def track():
    return load_track(DATA, "song_full")


def test_unchanged_cuts_give_identical_segments(track):
    base = build_segments(track, _cfg())
    again = CED.segments_from_cuts(track, CED.base_cut_times(base), base)
    assert [(s["start"], s["end"], s["cut"]) for s in again] == [(s["start"], s["end"], s["cut"]) for s in base]


def test_added_moved_removed(track):
    base = build_segments(track, _cfg())
    cuts = CED.base_cut_times(base)
    edited = sorted(cuts[1:] + [cuts[5] + 0.4, 50.123])           # remove first, add two
    segs = CED.segments_from_cuts(track, edited, base)
    assert segs[0]["start"] == 0.0 and segs[-1]["end"] == track["duration_sec"]
    assert all(a["end"] == b["start"] for a, b in zip(segs, segs[1:]))
    kinds = {s["start"]: s["cut"] for s in segs}
    assert kinds[50.123] == "edit"
    assert kinds[cuts[3]] == base[4]["cut"]                     # untouched cut keeps its kind
    d = CED.diff(cuts, CED.clean_cuts(edited, track["duration_sec"]))
    assert d["removed"] == [cuts[0]] and set(d["added"]) == {round(cuts[5] + 0.4, 3), 50.123}


def test_clean_cuts_rules():
    assert CED.clean_cuts([5, 1.0, 1.1, 1.3, 0.01, 99, 9.999], 10.0) == [1.0, 1.3, 5.0]


def test_settings_signature_changes_with_dials():
    a = CED.settings_signature(_cfg(SEG_ADAPTIVE))
    b = CED.settings_signature(_cfg(SEG_ADAPTIVE, change_threshold=0.2))
    assert a != b and a == CED.settings_signature(_cfg(SEG_ADAPTIVE))


def test_planned_segments_use_edits_only_for_their_track(track):
    ns = load_page(PLANNER, {"track_id": "song_full",
                             "cut_edits": {"track_id": "song_full", "cuts": [10.0, 20.0], "base_sig": "x"}})
    segs = ns["planned_segments"](track, _cfg())
    assert [s["start"] for s in segs] == [0.0, 10.0, 20.0]
    ns["st"].session_state["track_id"] = "song_slow"
    other = load_track(DATA, "song_slow")
    assert len(ns["planned_segments"](other, _cfg())) == len(build_segments(other, _cfg()))


def _run(at, **state):
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not errors(at), errors(at)
    return at


def test_ask_first_when_settings_change_keep_and_drop():
    at = page_app("pages/1_Compilation_Planner.py")
    edits = {"track_id": "song_full", "cuts": [8.0, 16.0, 24.0], "base_sig": "old settings"}
    _run(at, track_id="song_full", ui_section="🎵 0. Audio Settings", cut_edits=edits)
    assert any("changed the cutting settings" in w.value for w in at.warning)
    assert "4 segments" in " ".join(c.value for c in at.caption)
    [b for b in at.button if b.label == "✋ Keep my edited cuts"][0].click().run()
    assert not any("changed the cutting settings" in w.value for w in at.warning)
    assert at.session_state["cut_edits"]["cuts"] == [8.0, 16.0, 24.0]
    # settings change again → asked again → drop
    at.session_state["sm_pace"] = 1
    at.run()
    assert any("changed the cutting settings" in w.value for w in at.warning)
    [b for b in at.button if b.label.startswith("♻️ Use the new settings")][0].click().run()
    assert "cut_edits" not in at.session_state
    assert not any("Edited by hand" in c.value for c in at.caption)


def test_edits_reach_matching_export_and_project_file():
    at = page_app("pages/1_Compilation_Planner.py")
    from tests.fixtures import VIDEOS
    state = {"track_id": "song_full", "committed_selected_videos": list(VIDEOS),
             "ui_section": "🎬 3. Matching & Export",
             "cut_edits": {"track_id": "song_full", "cuts": [10.0, 20.0, 30.0, 40.0], "base_sig": "x"}}
    for v in VIDEOS:
        state[f"select_video_{v}"] = True
    _run(at, **state)
    from tests.apptest_util import press_match
    press_match(at)
    [b for b in at.button if b.label == "💾 Save plan"][0].click().run()
    plan = json.loads((DATA / "compilation_plans" / "song_full_plan.json").read_text())
    assert [b["track_time"][0] for b in plan["timeline"]] == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert plan["segmentation"]["hand_edited_cuts"] is True
    # saved with the project, and restored on load
    ns = load_page(PLANNER, dict(state))
    data = ns["_build_project_save_dict"]()
    assert data["cut_edits"]["cuts"] == [10.0, 20.0, 30.0, 40.0]
    ns2 = load_page(PLANNER, {"cut_edits": {"track_id": "song_full", "cuts": [1.0], "base_sig": "y"}})
    ns2["_apply_project_load_dict"](json.loads(json.dumps(data)))
    assert ns2["st"].session_state["cut_edits"]["cuts"] == [10.0, 20.0, 30.0, 40.0]
    # a project without edits clears any edits on screen
    ns3 = load_page(PLANNER, {"cut_edits": {"track_id": "song_full", "cuts": [1.0], "base_sig": "y"}})
    ns3["_apply_project_load_dict"]({k: v for k, v in data.items() if k != "cut_edits"})
    assert "cut_edits" not in ns3["st"].session_state
