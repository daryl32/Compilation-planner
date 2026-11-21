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
        assert shown["on"] and [x["line"] for x in shown["lines"]] == ["Welcome", "Intro to NNN"]      # (17 Nov map: Welcome first)
        assert edge["outro"]["on"] and [x["line"] for x in edge["outro"]["lines"]] == ["I hope not", "You can survive 30 days"]
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
        # 🧍 Studio previews and 👗 round looks moved to the Avatar Studio page; ⬇️ line fetching to 🔄 Sync now (21 Nov)
        assert not any(b.key in ("studio_make", "edge_lines_fetch", "edge_looks_save") for b in at.button)
        at.button(key="edge_loudness").click().run()        # 🔊 the loudness check
        assert not errors(at), errors(at)
        shown = " ".join(x.value for x in at.info) + " ".join(str(x.value) for x in at.markdown)
        assert "No line is on the server yet" in shown or "Typical line" in shown
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


def test_surprise_me_picks_songs_by_difficulty_building_in_energy():
    """🎲 Surprise me (21 Nov): how many from the rounds, calmer for Chill and driving for Hard, energy builds,
    recent songs avoided, short tracks left out."""
    import planner.edge as PE
    lib = {f"calm{i}": {"duration": 200.0, "bpm": 72.0 + i, "level_rank": 0.05 * i, "analysed": True} for i in range(5)}
    lib.update({f"loud{i}": {"duration": 200.0, "bpm": 150.0 + i, "level_rank": 0.8 + 0.04 * i, "analysed": True}
                for i in range(5)})
    lib["short"] = {"duration": 60.0, "bpm": 150.0, "level_rank": 1.0}
    assert PE.surprise_count("chill", 3, True) == 3 and PE.surprise_count("chill", 3, False) == 2
    assert PE.surprise_count("hard", 6, False) == round(300 * 6 / PE.SURPRISE_SEC_PER_SONG)
    for seed in range(8):
        chill = PE.surprise_songs(lib, "chill", 4, song_per_round=True, seed=seed)
        hard = PE.surprise_songs(lib, "hard", 4, song_per_round=True, seed=seed)
        assert len(chill) == len(set(chill)) == 4 and "short" not in chill + hard
        assert sum(t.startswith("calm") for t in chill) >= 3 and sum(t.startswith("loud") for t in hard) >= 3
        ints = [PE.song_intensity(lib[t]) for t in hard]
        assert ints == sorted(ints)                                         # the energy builds
    recent = {"loud0", "loud1", "loud2"}
    picks = [t for seed in range(20) for t in PE.surprise_songs(lib, "hard", 2, True, seed=seed, recent=recent)]
    assert sum(t in recent for t in picks) < len(picks) / 4


def test_random_videos_cover_the_phase_rules_and_top_up_for_length():
    import planner.edge as PE
    pool = [f"v{i}" for i in range(100)]
    secs = {v: 60.0 for v in pool}
    tagged = ["v97", "v98", "v99"]
    got, note = PE.draw_videos(pool, 30, 1, secs, 600.0, must=[tagged])
    assert len(got) == 30 and set(tagged) <= set(got) and note is None
    assert PE.draw_videos(pool, 30, 1, secs, 600.0)[0] == PE.draw_videos(pool, 30, 1, secs, 600.0)[0]
    assert PE.draw_videos(pool, 30, 1, secs, 600.0)[0] != PE.draw_videos(pool, 30, 2, secs, 600.0)[0]
    long_, note = PE.draw_videos(pool, 10, 1, secs, 1800.0)
    assert len(long_) == 30 and "instead of 10" in note
    assert PE.draw_videos(pool[:5], 30, 1, secs, 0.0) == (pool[:5], None)


def test_the_page_runs_surprise_me_and_random_videos(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import mix_jobs
        import planner.edge as PE
        mix_jobs.forget(PE.JOB_KEY)
        monkeypatch.setattr(PE, "recent_edge_songs", lambda n=5: set())
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "⏱️ Edge Mode"
        at.run()
        at.multiselect(key="edge_playlist").set_value(["song_full"]).run()
        at.multiselect(key="edge_playlist").set_value(["song_full", PE.SURPRISE]).run()
        assert at.session_state["edge_playlist"] == [PE.SURPRISE]           # Surprise me replaces the picks
        at.toggle(key="edge_random_videos").set_value(True).run()
        at.number_input(key="edge_random_n").set_value(2).run()
        assert not errors(at), errors(at)
        at.button(key="edge_go_only").click().run()
        _wait_job(at)
        assert not errors(at), errors(at)
        res = at.session_state["edge_result"]
        drawn = res["drawn"]
        assert drawn["songs"] and res["track_ids"] and set(res["track_ids"]) <= set(drawn["songs"])
        assert len(res["video_ids"]) >= 2 and set(res["video_ids"]) <= set(drawn["pool"])
        assert any("Surprise me picked" in n for n in res["notes"]) and any("random videos" in n for n in res["notes"])
        assert any(b.key == "edge_redraw_songs" for b in at.button) and any(b.key == "edge_redraw_videos" for b in at.button)
        monkeypatch.setattr(PE, "_render", lambda res, settings, token, prog: (("success", "ok"), "t.mp4"))
        before = list(res["video_ids"])
        at.button(key="edge_redraw_songs").click().run()        # 🎲 new songs, the same videos
        _wait_job(at)
        assert not errors(at), errors(at)
        again = at.session_state["edge_result"]
        assert again["video_ids"] == before and again["drawn"]["songs"]
        at.button(key="edge_redraw_videos").click().run()       # 🎲 new videos from the same pool, the same songs
        _wait_job(at)
        assert not errors(at), errors(at)
        third = at.session_state["edge_result"]
        assert third["track_ids"] and set(third["track_ids"]) <= set(again["drawn"]["songs"])
        assert set(third["video_ids"]) <= set(drawn["pool"])
        mix_jobs.forget(PE.JOB_KEY)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_roles_per_phase_are_strict_with_a_fallback(monkeypatch):
    """🎭 Roles per phase (21 Nov): a phase's footage keeps to clips with one of its roles; too little → any footage."""
    import planner.compose as PC
    import planner.edge as PE
    spans = {"v": [{"video_id": "v", "scene_id": i, "remaining_sec": 30.0, "tags": ["crowd"] if i < 2 else [],
                    "roles": ["outro"] if i in (0, 3) else []} for i in range(5)]}
    segs = [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 20.0}, {"start": 20.0, "end": 60.0}]
    rule = {0: ["outro"], 1: ["nonesuch"], 2: ["outro"]}
    out, notes = PC._tag_exclusions(segs, spans, None, lambda a, b: rule[int(a // 10) if a < 20 else 2])
    assert out[0] == {("v", 1), ("v", 2), ("v", 4)}                     # only the outro scenes 0 and 3 are left
    assert 1 not in out and 2 not in out                                # none carry it / 60 s isn't 2× the 40 s block
    assert any("Roles per phase" in n and "too little footage" in n for n in notes)
    both, _ = PC._tag_exclusions(segs[:1], spans, lambda a, b: ["crowd"], lambda a, b: ["outro"])
    assert both == {0: {("v", 1), ("v", 2), ("v", 3), ("v", 4)}}        # crowd AND outro: only scene 0
    loose, _ = PC._tag_exclusions(segs[:1], spans, lambda a, b: ["crowd"], lambda a, b: ["nonesuch"])
    assert loose == {0: {("v", 2), ("v", 3), ("v", 4)}}                # nothing fits both: the tags alone
    # the defaults, and the plan carries the rules
    assert PE.phase_roles_or_default(None) == {"wait": ["intro"], "stop": ["outro"]}
    assert PE.phase_roles_or_default({}) == {} and PE._phase_role(PE.E.schedule("chill", 3), 115.0, {}) is None
    out_ = _timelines(monkeypatch)
    try:
        res = PE.compile_edge([], VIDS, "chill", 3, phase_roles={"medium": ["action"], "wait": []})
        assert res["plan"]["edge"]["phase_roles"] == {"medium": ["action"]}
        assert any("Roles per phase" in n for n in res["notes"])
    finally:
        shutil.rmtree(out_, ignore_errors=True)


def test_the_page_sets_roles_per_phase(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import mix_jobs
        import planner.edge as PE
        mix_jobs.forget(PE.JOB_KEY)
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "⏱️ Edge Mode"
        at.run()
        assert at.session_state["edge_roles_wait"] == ["intro"] and at.session_state["edge_roles_stop"] == ["outro"]
        for p in ("medium", "fast", "extra"):
            at.session_state["edge_roles_" + p] = ["outro"]
        at.selectbox(key="edge_timer_style").set_value("strip")     # ⏱️ (and the strip timer style, 21 Nov)
        at.run()
        assert at.selectbox(key="edge_corner").disabled
        assert not errors(at), errors(at)
        at.button(key="edge_go_only").click().run()
        _wait_job(at)
        assert not errors(at), errors(at)
        got = at.session_state["edge_result"]["plan"]["edge"]["phase_roles"]
        assert got == {"wait": ["intro"], "medium": ["outro"], "fast": ["outro"], "extra": ["outro"], "stop": ["outro"]}
        assert at.session_state["edge_result"]["plan"]["edge"]["overlay"]["style"] == "strip"
        at.button(key="edge_back").click().run()
        at.button(key="edge_roles_reset").click().run()
        assert at.session_state["edge_roles_medium"] == [] and at.session_state["edge_roles_wait"] == ["intro"]
        mix_jobs.forget(PE.JOB_KEY)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_compile_carries_the_instructions_only_on_a_split_screen(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import edge_mode as E
        import planner.edge as PE
        cm = E.load_cue_map(E.CUE_MAP_FILE)
        for ln in cm["lines"]:
            if ln["name"] == "Start-Round 1":
                ln["instruction"], ln["instruction_delay"] = "JUST THE TIP", 0.5
        monkeypatch.setattr(PE, "load_cue_map", lambda: cm)
        res = PE.compile_edge([], VIDS, "chill", 3, avatar=True, avatar_mode="split", instructions=True)
        ins = res["plan"]["edge"]["instructions"]
        start = next(c["time"] for c in res["plan"]["edge"]["voiceover"]["cues"] if c["line"] == "Start-Round 1")
        assert ins["on"] and ins["items"][0] == {"time": round(start + 0.5, 3), "text": "JUST THE TIP"}
        corner = PE.compile_edge([], VIDS, "chill", 3, avatar=True, avatar_mode="corner", instructions=True)
        assert not corner["plan"]["edge"]["instructions"]["on"]
    finally:
        shutil.rmtree(out, ignore_errors=True)
