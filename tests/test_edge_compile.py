"""⏱️ Edge Mode compiles and the page: songs / beat, 🎭 roles, 🏷️ tags per phase, server jobs, reopening a render."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.edge_support import (DATA, VIDS, E_CARD_LEAD, _check_plan, _emap, _level, _pixel, _seconds,  # noqa: F401
                                _timelines, _wait_job)


def test_compile_without_a_song_and_with_a_playlist(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import json
        import planner.edge as PE
        seen = []
        res = PE.compile_edge([], VIDS, "chill", 3, voiceover=True, progress=lambda d, t, txt: seen.append(txt))
        plan, edge = res["plan"], res["plan"]["edge"]
        assert plan["matching_mode"] == "Edge" and plan["track_duration_sec"] == 360.0 and seen
        _check_plan(plan, 360.0)
        assert edge["beat"]["on"] and edge["beat"]["volume"] == 1.0 and edge["playlist"] == []      # no song → the beat is on
        # the voiceover follows the cue map: named lines, never two at once; outro lines bring their screen
        vo = edge["voiceover"]
        assert len(edge["schedule"]) == 18 and vo["cue_map"] and vo["cues"][0]["line"] == "Start-Round 1"
        for x, y in zip(vo["cues"], vo["cues"][1:]):
            assert y["time"] >= x["time"] + x["seconds"]
        assert all(c["file"] is None for c in vo["cues"]) and "Now stop" in vo["lines_missing"]      # no files here: tones
        assert not edge["intro"]["on"] and edge["intro"]["lines"] == []          # the intro screen only when switched on
        shown = PE.compile_edge([], VIDS, "chill", 3, voiceover=True, intro=True)["plan"]["edge"]["intro"]
        assert shown["on"] and [x["line"] for x in shown["lines"]] == ["Intro to NNN"]      # (16 Nov map: Welcome has no cue)
        assert edge["outro"]["on"] and [x["line"] for x in edge["outro"]["lines"]] == ["I hope not"]
        assert res["voice_clashes"] and {"line", "with", "round", "where", "result"} <= set(res["voice_clashes"][0])
        quiet = PE.compile_edge([], VIDS, "chill", 3)["plan"]["edge"]
        assert not quiet["intro"]["on"] and quiet["voiceover"]["cues"] == []
        assert quiet["outro"]["on"] and quiet["outro"]["lines"] == [] and not quiet["outro"]["figure"]      # the outro closes every compilation
        filled = sum(1 for e in plan["timeline"] if e["scenes"])
        assert filled >= 0.6 * len(plan["timeline"])
        json.dumps(plan)                                                           # the plan is a file Blender can read
        # with songs: back to back, started again, the footage follows them; the beat is off unless asked for
        res2 = PE.compile_edge(["song_full", "song_noanalysis"], VIDS, "chill", 3)
        plan2, edge2 = res2["plan"], res2["plan"]["edge"]
        _check_plan(plan2, 360.0)
        assert [(p["track_id"], p["offset"]) for p in edge2["playlist"]] == [
            ("song_full", 0.0), ("song_noanalysis", 96.0), ("song_full", 170.0), ("song_noanalysis", 266.0), ("song_full", 340.0)]
        assert edge2["playlist"][-1]["play"] == 20.0 and not edge2["beat"]["on"] and not edge2["voiceover"]["on"]
        assert {e["edge_track_id"] for e in plan2["timeline"]} == {"song_full", "song_noanalysis"}
        res3 = PE.compile_edge(["song_full"], VIDS, "medium", 4, beat=True, voiceover=True)
        assert res3["plan"]["track_duration_sec"] == 720.0 and res3["plan"]["edge"]["beat"] == {
            **res3["plan"]["edge"]["beat"], "on": True, "volume": 0.6}
        assert res3["plan"]["edge"]["voiceover"]["music_volume"] == 0.25
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_page_compiles_and_comes_back_to_its_setup(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "⏱️ Edge Mode"
        at.run()
        assert not errors(at), errors(at)
        at.radio(key="edge_difficulty").set_value("medium")
        at.slider(key="edge_rounds").set_value(4)
        at.toggle(key="edge_voice").set_value(True)
        at.run()
        assert not errors(at), errors(at)
        at.button(key="edge_go_only").click().run()
        _wait_job(at)
        assert not errors(at), errors(at)
        res = at.session_state["edge_result"]
        assert res["plan"]["track_duration_sec"] == 720.0 and res["plan"]["edge"]["voiceover"]["on"]
        at.button(key="edge_back").click().run()
        assert not errors(at), errors(at)
        assert at.session_state["edge_difficulty"] == "medium" and at.session_state["edge_rounds"] == 4
        # the other two planners are still there
        at.session_state["planner_mode"] = "🤖 New matcher"
        at.run()
        assert not errors(at), errors(at)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_compile_carries_the_avatar_and_the_intro(monkeypatch, tmp_path):
    out = _timelines(monkeypatch)
    try:
        import planner.edge as PE
        monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "avatar")
        for slot in ("intro", "fast"):
            (tmp_path / "avatar" / slot).mkdir(parents=True)
            (tmp_path / "avatar" / slot / f"{slot}_1.mp4").write_bytes(b"x")
        res = PE.compile_edge([], VIDS, "chill", 3, avatar=True, avatar_mode="split", intro=True)
        edge = res["plan"]["edge"]
        # the avatar brings the lines with her, voiceover switch or not: she appears with every line of the cue map
        assert edge["avatar"]["on"] and edge["avatar"]["mode"] == "split" and edge["voiceover"]["on"]
        assert [(c["time"], c["line"]) for c in edge["avatar"]["cues"]] == [(c["time"], c["line"]) for c in edge["voiceover"]["cues"]]
        assert len(edge["avatar"]["cues"]) == len(edge["voiceover"]["cues"]) >= 20 and "Start-Round 1" in edge["avatar"]["placeholders"]      # no clips here: stand-ins
        assert edge["intro"]["on"] and edge["intro"]["clip"].endswith("intro_1.mp4")
        assert edge["intro"]["stats"] == [["Difficulty", "Chill"], ["Rounds", "3"], ["Time", "6:00"]]
        assert 1 <= len(edge["intro"]["thumbs"]) <= 6 and {v for v, _ in edge["intro"]["thumbs"]} <= set(VIDS)
        off = PE.compile_edge([], VIDS, "chill", 3)["plan"]["edge"]
        assert not off["avatar"]["on"] and off["avatar"]["cues"] == [] and not off["intro"]["on"]
        # the page: the avatar's options show and compile (an earlier test's finished job set aside first)
        import mix_jobs
        mix_jobs.forget(PE.JOB_KEY)
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "⏱️ Edge Mode"
        at.run()
        at.toggle(key="edge_avatar").set_value(True)
        at.toggle(key="edge_intro").set_value(True)
        at.run()
        assert not errors(at), errors(at)
        at.radio(key="edge_avatar_mode").set_value("split")
        at.run()
        at.button(key="edge_go_only").click().run()
        _wait_job(at)
        assert not errors(at), errors(at)
        e = at.session_state["edge_result"]["plan"]["edge"]
        assert e["avatar"]["on"] and e["avatar"]["mode"] == "split" and e["intro"]["on"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_compile_carries_the_new_switches_and_matches_the_beat_round_by_round(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import planner.edge as PE
        res = PE.compile_edge([], VIDS, "chill", 3)
        edge = res["plan"]["edge"]
        assert edge["signals"] == {"on": True, "sound": True, "timer": True, "flash": True} and edge["round_badge"]["on"]
        assert edge["music"]["fade_stop"] and not edge["music"]["duck"]      # a steady background under her lines
        _check_plan(res["plan"], 360.0)
        rounds = {}
        for e in res["plan"]["timeline"]:
            for s in e["scenes"]:
                for link in s["chain"]:
                    rounds.setdefault(e["edge_part"], set()).add((link["video_id"], link["scene_id"], link["clip_start_sec"]))
        assert sorted(rounds) == [0, 1, 2]                 # matched a round at a time
        off = PE.compile_edge(["song_full", "song_noanalysis"], VIDS, "chill", 3, signals=False, round_badge=False,
                              music_fade=False, song_per_round=True)["plan"]["edge"]
        assert not off["signals"]["on"] and not off["round_badge"]["on"] and not off["music"]["fade_stop"]
        assert off["music"]["song_per_round"]
        assert [(p["track_id"], p["round"]) for p in off["playlist"] if p["loop"] == 1] == [
            ("song_full", 1), ("song_noanalysis", 2), ("song_full", 3)]
        assert all(p["offset"] in (0.0, 120.0, 240.0) for p in off["playlist"] if p["loop"] == 1)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_wait_favours_intro_clips_and_stop_outro_clips(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import planner.compose as PC
        import planner.edge as PE
        real, seen = PC.compile_plan, []

        def spy(track_id, video_ids, **kw):
            role = kw.get("block_role")
            seen.append((track_id, role))
            return real(track_id, video_ids, **kw)
        monkeypatch.setattr(PC, "compile_plan", spy)
        PE.compile_edge([], VIDS, "chill", 3)                # the beat, round by round: each round's own phases
        role = seen[0][1]
        assert role(0.0, 10.0) == "intro" and role(30.0, 35.0) is None and role(105.0, 110.0) == "outro"
        seen.clear()
        PE.compile_edge(["song_full"], VIDS, "chill", 3)     # a song: its time where it plays in the compilation
        role = seen[0][1]
        assert role(2.0, 6.0) == "intro" and role(50.0, 54.0) is None
        seen.clear()
        PE.compile_edge(["song_full", "song_noanalysis"], VIDS, "chill", 3, energy_mix=True)
        parts = {}
        for tid, role in seen:
            parts[tid] = role
        assert any(r is not None for r in parts.values())
        assert PE._phase_role(PE.E.schedule("chill", 3), 115.0) == "outro"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_tags_per_phase_keep_each_phases_footage_to_its_tags(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import planner.edge as PE
        from planner.data import load_all_catalogues
        cats = load_all_catalogues()

        def tags_of(link):
            sc = next((s for s in cats[link["video_id"]]["scenes"] if s["scene_id"] == link["scene_id"]), {})
            return {t.lower() for t in sc.get("tags") or []}
        res = PE.compile_edge([], VIDS, "chill", 3, phase_tags={"extra": ["crowd"], "wait": ["night"]})
        edge = res["plan"]["edge"]
        assert edge["phase_tags"] == {"extra": ["crowd"], "wait": ["night"]}
        assert any("Tags per phase" in n for n in res["notes"])
        for e in res["plan"]["timeline"]:
            want = {"extra": "crowd", "wait": "night"}.get(e["edge_profile"])
            if want:
                for s in e["scenes"]:
                    assert all(want in tags_of(link) for link in s["chain"]), (e["edge_profile"], s["chain"])
        # a tag nobody has: that phase takes any footage, and the note says so
        res = PE.compile_edge([], VIDS, "chill", 3, phase_tags={"slow": ["nonesuch"]})
        assert any("too little footage" in n for n in res["notes"])
        assert all(e["scenes"] for e in res["plan"]["timeline"] if e["edge_profile"] == "slow")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_compile_and_render_run_on_the_server_and_the_page_picks_up_the_result(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import mix_jobs
        import planner.edge as PE
        mix_jobs.forget(PE.JOB_KEY)
        rendered = []

        def fake_render(res, settings, token, prog):
            prog(0.5, "rendering segment 1/2")
            rendered.append((res["plan"]["track_duration_sec"], settings["difficulty"]))
            return ("success", "Rendered test.001.mp4 in 0:01."), "test.001.mp4"
        monkeypatch.setattr(PE, "_render", fake_render)
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "⏱️ Edge Mode"
        at.run()
        at.button(key="edge_go").click().run()               # ⏱️ Compile & render: one press
        assert not errors(at), errors(at)
        _wait_job(at)
        assert not errors(at), errors(at)
        assert rendered == [(360.0, "chill")] and at.session_state["edge_render"] == "test.001.mp4"
        assert at.session_state["edge_result"]["plan"]["track_duration_sec"] == 360.0
        # a new session (the phone came back to a reloaded page) finds the finished job too
        again = page_app("pages/1_Compilation_Planner.py", timeout=300)
        again.session_state["planner_mode"] = "⏱️ Edge Mode"
        again.run()
        assert again.session_state["edge_render"] == "test.001.mp4" and not errors(again)
        mix_jobs.forget(PE.JOB_KEY)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_an_edge_render_reopens_in_edge_mode_with_its_setup_and_plan(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import mix_jobs
        import planner.edge as PE
        import renders
        mix_jobs.forget(PE.JOB_KEY)
        plan = PE.compile_edge(["song_full"], VIDS, "medium", 4, song_per_round=True, intro=True,
                               phase_tags={"extra": ["crowd"]})["plan"]
        data, kind = renders.reopen_state({"plan": plan})              # an older render: the setup comes from the plan
        assert kind == "edge" and data["setup"] is None
        setup = PE._setup_from_plan(plan["edge"])
        assert setup["difficulty"] == "medium" and setup["rounds"] == 4 and setup["playlist"] == ["song_full"]
        assert setup["song_per_round"] and setup["intro"] and setup["phase_tags"] == {"extra": ["crowd"]}
        assert renders.reopen_state({"plan": {**plan, "matching_mode": "Auto story"}})[1] == "compose"
        assert renders.reopen_state({"project": {"x": 1}, "plan": plan})[1] == "full"      # Classic's own, as before
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "⏱️ Edge Mode"
        at.session_state["_reopen_edge"] = {**data, "label": "temp-Video-preview-x.001.mp4"}
        at.run()
        assert not errors(at), errors(at)
        assert at.session_state["edge_result"]["plan"]["track_id"] == plan["track_id"]
        assert at.session_state["edge_difficulty"] == "medium" and at.session_state["edge_rounds"] == 4
        assert at.session_state["edge_tags_extra"] == ["crowd"] and at.session_state["edge_seed"] == plan["edge"]["seed"]
        assert any("Reopened" in c.value for c in at.caption)
    finally:
        shutil.rmtree(out, ignore_errors=True)
