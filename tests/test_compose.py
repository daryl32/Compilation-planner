"""🤖 The new matcher's own flow: section cuts at the recipes' pace, suggesting songs / videos for chosen videos,
compiling a plan, and the planner landing (Classic | New matcher) with By song / By video headless."""
from __future__ import annotations

import json
import shutil

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.test_story_matcher import _setup as _timelines


def test_section_cuts_follow_the_pace_and_boundaries():
    import recipes as RC
    import section_cuts as SCUT
    from tests.fixtures import load_track
    track = load_track(DATA, "song_full")
    regions = [{"start": 0.0, "end": 24.0, "type": "intro"}, {"start": 24.0, "end": 48.0, "type": "buildup"},
               {"start": 48.0, "end": 72.0, "type": "drop"}, {"start": 72.0, "end": 96.0, "type": "ending"}]
    segs = SCUT.story_segments(track, regions, RC.load_all())
    assert segs[0]["start"] == 0.0 and abs(segs[-1]["end"] - 96.0) < 1e-6
    for a, b in zip(segs, segs[1:]):
        assert abs(a["end"] - b["start"]) < 1e-6 and b["end"] > b["start"]
    starts = {round(s["start"], 3) for s in segs}
    assert {24.0, 48.0, 72.0} <= starts                       # section boundaries are always cuts
    by = {}
    for s in segs:
        by.setdefault(s["section_type"], []).append(s["end"] - s["start"])
    assert sum(by["drop"]) == 24.0 and sum(by["intro"]) == 24.0
    import numpy as np
    assert np.mean(by["drop"]) < np.mean(by["intro"])         # drops cut faster than the intro
    assert 0.5 <= np.mean(by["drop"]) <= 3.0 and np.mean(by["intro"]) >= 3.0
    # a rising build-up shortens its clips through the section
    bu = [s for s in segs if s["section_type"] == "buildup"]
    if len(bu) >= 3:
        assert (bu[0]["end"] - bu[0]["start"]) >= (bu[-1]["end"] - bu[-1]["start"]) - 0.6
    assert "blocks" in SCUT.describe(segs) and all("energy" in s and "intensity" in s for s in segs)


def test_compile_suggest_and_result(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import planner.compose as PC
        import footage_timeline as FT
        res = PC.compile_plan("song_full", ["vid_a", "vid_b", "vid_c", "vid_f"])
        plan = res["plan"]
        assert plan["matching_mode"] == "Auto story" and plan["timeline"] and plan["segmentation"]["method"] == "sections"
        assert all(len(e["scenes"]) <= 1 for e in plan["timeline"])                 # split screen off
        assert all("source_path" in link for e in plan["timeline"] for s in e["scenes"] for link in s["chain"])
        assert res["summary"]["slots"] == len(plan["timeline"])
        res2 = PC.compile_plan("song_full", ["vid_a", "vid_b", "vid_c", "vid_f"], split_screen=True)
        assert any(len(e["scenes"]) > 1 for e in res2["plan"]["timeline"])
        # songs for chosen videos, and other videos like them
        tracks = PC.suggest_tracks(("vid_f",), PC._audio_signature(), FT.signature())
        assert tracks and all(0 <= t["fit"] <= 1 for t in tracks) and tracks == sorted(tracks, key=lambda t: -t["fit"])
        others = PC.suggest_videos(("vid_a",), FT.signature())
        assert others and "vid_a" not in {o["video_id"] for o in others}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_planner_landing_and_new_matcher_pages(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "🤖 New matcher"
        at.run()
        assert not errors(at), errors(at)
        assert at.radio(key="compose_way").value == "🎵 By song"
        assert at.selectbox(key="cs_track").value
        at.button(key="cs_go").click().run()
        assert not errors(at), errors(at)
        assert any("blocks" in c.value for c in at.caption)
        assert [b for b in at.button if b.key == "compose_save"]
        at.button(key="compose_back").click().run()
        assert not errors(at), errors(at)
        at.session_state["compose_way"] = "🎬 By video"
        at.run()
        assert not errors(at), errors(at)
        at.multiselect(key="cv_start").set_value(["vid_a"]).run()
        assert not errors(at), errors(at)
        at.button(key="cv_find").click().run()                 # stage 1: songs
        assert not errors(at), errors(at)
        assert at.radio(key="cv_track").value
        at.button(key="cv_more").click().run()                 # stage 2: other videos (optional)
        assert not errors(at), errors(at)
        at.button(key="cv_go").click().run()
        assert not errors(at), errors(at)
        assert at.session_state[PC_RESULT()]["way"] == "video"
        # Classic is untouched behind the landing
        at.button(key="compose_back").click().run()
        at.session_state["planner_mode"] = "🎛️ Classic"
        at.run()
        assert not errors(at), errors(at)
        assert at.radio(key="ui_section").value
    finally:
        shutil.rmtree(out, ignore_errors=True)


def PC_RESULT():
    import planner.compose as PC
    return PC.RESULT_KEY
