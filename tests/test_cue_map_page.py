"""🎙️ Cue Map in the app (19 Nov 2026): one copy, versions, refused stale saves, cut lines, the page."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)


@pytest.fixture
def voice(tmp_path, monkeypatch):
    import planner.edge as PE
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "edge_voiceover")
    monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "edge_avatar")
    return tmp_path


def _mp3(path: Path, secs: float, lead: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    src = f"aevalsrc='if(gte(t,{lead}),0.5*sin(440*2*PI*t),0)':d={secs}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-c:a", "libmp3lame", str(path)], check=True)


def test_saves_keep_versions_refuse_stale_and_stop_drive_pickups(voice):
    import cue_store as CS
    cm = CS.load()
    assert cm["rev"] == 0 and len(cm["lines"]) > 50 and not CS.managed()
    got = CS.save(cm, 0, "taken over")
    assert got["rev"] == 1 and CS.managed() and CS.load()["rev"] == 1
    with pytest.raises(CS.Stale):                         # another device saved since revision 0 was opened
        CS.save(cm, 0, "stale")
    got["lines"][0]["notes"] = "hello"
    got = CS.save(got, 1, "note")
    assert got["rev"] == 2 and len(CS.versions()) == 1 and CS.versions()[0]["rev"] == 1
    back = CS.restore(CS.versions()[0]["file"], 2)
    assert back["rev"] == 3 and back["lines"][0].get("notes") != "hello" and len(CS.versions()) == 2
    line = {"name": "x", "seconds": 10.0, "trim_in": 1.0, "trim_out": 7.0, "speed": 1.2}
    assert CS.is_cut(line) and CS.played_seconds(line) == pytest.approx(5.0)
    assert CS.played_seconds({"seconds": 4.0}) == 4.0 and not CS.is_cut({"trim_out": 0, "speed": 1})


def test_drive_sent_maps_are_ignored_once_kept_in_the_app(voice, monkeypatch):
    import sys
    import types
    import cue_store as CS
    import planner.edge as PE
    CS.save(CS.load(), 0, "taken over")
    called = []
    fake = types.SimpleNamespace(credentials_available=lambda: True,
                                 download_source_video=lambda rel, dest, **k: called.append(rel) or None,
                                 list_drive_folder=lambda rel: ([{"name": "edge_cues_2026-11-20T01-00.json",
                                                                  "modifiedTime": "2026-11-20T01:00:00Z"}], None),
                                 list_drive_subfolders=lambda rel: ([], None))
    monkeypatch.setitem(sys.modules, "drive_sync", fake)
    PE.fetch_lines({"lines": []}, avatar=False)
    assert not any("edge_cues_" in c for c in called)


def test_a_cut_line_plays_cut_in_the_compile(voice):
    import cue_store as CS
    import edge_render as R
    import planner.edge as PE
    cm = CS.load()
    line = next(ln for ln in cm["lines"] if ln["name"] == "Start-Round 1")
    _mp3(PE.lines_dir() / line["file"], 12.0, lead=1.0)
    line.update(trim_in=0.0, trim_out=6.0, speed=1.0)
    CS.save(cm, 0, "cut")
    cm = CS.load()
    line = next(ln for ln in cm["lines"] if ln["name"] == "Start-Round 1")
    raw, cut = PE.line_file(line, cut=False), PE.line_file(line)
    assert raw != cut and "_cut" in str(cut) and R.probe(cut)["seconds"] == pytest.approx(6.0, abs=0.1)
    assert PE.line_file(line) == cut                      # made once, then reused
    assert PE.line_durations(cm)["Start-Round 1"] == pytest.approx(6.0, abs=0.1)
    import cue_map_view as V
    a, b = V.cut_silence(raw)
    assert a == pytest.approx(0.92, abs=0.1) and b == 0.0


def test_the_page_takes_over_edits_pile_up_until_one_save_and_a_stale_save_is_refused(voice):
    import copy
    import cue_store as CS
    import cue_map_view as V
    from tests.apptest_util import errors, page_app
    at = page_app("pages/8_Cue_Map.py").run()
    assert not errors(at)
    next(b for b in at.button if "Keep the cue map in the app" in b.label).click().run()
    assert not errors(at) and CS.managed() and CS.load()["rev"] == 1
    save = lambda: next(b for b in at.button if b.label == "💾 Save the cue map")      # noqa: E731
    assert save().disabled                                    # nothing to save
    cm = V.current() if False else CS.load()
    # open a line (as a tap does): opening alone changes nothing
    at.session_state["cm_dialog"] = True
    at.session_state[V.SS_SEL] = "Start-Round 1"
    at.session_state[V.SS_DRAFT] = V._draft_of(next(ln for ln in cm["lines"] if ln["name"] == "Start-Round 1"))
    at.run()
    assert not errors(at) and save().disabled and CS.load_draft() is None
    # edit two lines: nothing is saved until the one 💾
    next(t for t in at.text_area if t.label.startswith("Notes")).input("Say it slower").run()
    assert not save().disabled and CS.load()["rev"] == 1 and CS.load_draft() is not None
    at.session_state["cm_dialog"] = True
    at.session_state[V.SS_SEL] = "Keep going"
    at.session_state[V.SS_DRAFT] = V._draft_of(next(ln for ln in V.CS.load_draft()["map"]["lines"] if ln["name"] == "Keep going"))
    at.run()
    next(t for t in at.text_area if t.label.startswith("Notes")).input("Use more").run()
    assert any("2 changed" in (m.value or "") for m in at.markdown)
    save().click().run()
    assert not errors(at)
    got = {ln["name"]: ln for ln in CS.load()["lines"]}
    assert got["Start-Round 1"]["notes"] == "Say it slower" and got["Keep going"]["notes"] == "Use more"
    assert CS.load()["rev"] == 2 and CS.load_draft() is None and save().disabled
    # with nothing unsaved, the page follows a newer save made elsewhere
    CS.save(copy.deepcopy(CS.load()), 2, "other device")
    at.run()
    assert V.CS.load()["rev"] == 3 and at.session_state[V.SS_MAP]["rev"] == 3
    # unsaved changes made on revision 3, then the map is replaced elsewhere (e.g. an import): the save is refused
    next(t for t in at.text_area if t.label.startswith("Notes")).input("Mine").run()
    base = CS.load_draft()
    CS.save(copy.deepcopy(CS.load()), 3, "imported elsewhere")
    CS.save_draft(base["map"])
    save().click().run()
    assert any("Not saved" in e.value for e in at.error)
    assert {ln["name"]: ln for ln in CS.load()["lines"]}["Keep going"]["notes"] == "Use more"
    assert {ln["name"]: ln for ln in CS.load_draft()["map"]["lines"]}["Keep going"]["notes"] == "Mine"
    # ✖ Discard (with a confirm) goes back to the saved map
    next(b for b in at.button if b.label == "✖ Discard").click().run()
    next(b for b in at.button if b.label == "Yes, discard").click().run()
    assert not errors(at) and CS.load_draft() is None


def test_lines_can_be_added_and_filtered_by_category(voice, monkeypatch):
    import cue_store as CS
    import cue_map_view as V
    from tests.apptest_util import errors, page_app
    CS.save(CS.load(), 0, "taken over")
    monkeypatch.setattr(V, "new_on_drive", lambda cm: [{"name": "Brand new.mp3", "folder": "New", "seconds": 2.5}])
    at = page_app("pages/8_Cue_Map.py")
    at.session_state["cm_view"] = "🎙️ Lines"
    at.run()
    assert not errors(at)
    assert any("1 new on Google Drive" in e.label for e in at.expander)
    next(b for b in at.button if b.label == "➕ Add").click().run()
    assert not errors(at)
    assert at.session_state[V.SS_SEL] == "Brand new" and CS.changes(CS.load(), CS.load_draft()["map"])["added"] == ["Brand new"]
    assert not any("new on Google Drive" in e.label and "1 new" in e.label for e in at.expander)
    next(b for b in at.button if b.label == "💾 Save the cue map").click().run()
    assert any(ln["name"] == "Brand new" for ln in CS.load()["lines"])
    pills = at.get("button_group") or []
    assert pills, "category pills shown"


def test_end_cues_keep_their_finish_when_the_length_changes():
    import cue_map_view as V
    line = {"name": "a", "seconds": 10.0, "trim_in": 0, "trim_out": 0, "speed": 1,
            "cues": [{"type": "phase", "phases": ["stop"], "at": "end", "offset": 12.0}]}
    d = V._draft_of(line)
    assert d["cues"][0]["_finish"] == 2.0                  # starts 12 s before the end, 10 s long → finishes 2 s before
    d["trim_out"] = 4.0
    out = V._commit(d)
    assert out["cues"][0]["offset"] == 6.0 and "_finish" not in out["cues"][0] and out["play_seconds"] == 4.0


def test_timeline_actions_drop_move_and_the_screens(voice):
    import json
    import edge_mode as E
    import cue_store as CS
    import cue_map_view as V
    import streamlit as st
    CS.save(CS.load(), 0, "taken over")
    cm = V.current(reload=True)
    sched = E.schedule("chill", 3, 0)
    got = E.resolve_lines(sched, cm, "chill", 0)
    data = V.timeline_data(cm, sched, got)
    json.dumps(data)
    assert [r["label"] for r in data["rounds"]] == ["Round 1", "Round 2", "Last round"]
    assert data["plays"] and all(p["key"] for p in data["plays"] if not p["cls"].startswith("auto"))
    assert data["intro"][0]["line"] == "Welcome"
    # drop a line with no cue into round 2, 47 s in (Medium starts at 40 s): a start-of-Medium cue +7 s, round 2
    line = "Do you want to see my tits again"
    V.apply_action(cm, {"act": "drop", "line": line, "round": 2, "time": 120 + 47}, sched)
    cue = next(ln for ln in V.current()["lines"] if ln["name"] == line)["cues"][-1]
    assert cue == {"type": "phase", "phases": ["medium"], "at": "start", "offset": 7.0, "roundList": ["2"], "diffs": [], "note": ""}
    got = E.resolve_lines(sched, V.current(), "chill", 0)
    assert any(c["line"] == line and c["where"] == "round 2, start of Medium +7 s" for c in got["clashes"])   # taken there
    # move it to the last round's Fast, 5 s in (its cue is the line's last one)
    n_cues = len(next(ln for ln in V.current()["lines"] if ln["name"] == line)["cues"])
    V.apply_action(V.current(), {"act": "move", "line": line, "key": [line, n_cues - 1], "from": 2, "round": 3,
                                 "time": 240 + 60 + 5}, sched)
    cue = next(ln for ln in V.current()["lines"] if ln["name"] == line)["cues"][-1]
    assert cue["phases"] == ["fast"] and cue["offset"] == 5.0 and cue["roundList"] == ["last"]
    # screens: add to the outro, reorder the intro, take one off
    V.apply_action(V.current(), {"act": "screen_add", "which": "outro", "line": "Keep going"}, sched)
    V.apply_action(V.current(), {"act": "screen_order", "which": "intro", "order": ["Intro to NNN", "Welcome"]}, sched)
    got = E.resolve_lines(sched, V.current(), "chill", 0)
    assert got["intro"] == ["Intro to NNN", "Welcome"] and got["outro"][-1] == "Keep going"
    V.apply_action(V.current(), {"act": "screen_remove", "which": "outro", "line": "Keep going"}, sched)
    assert "Keep going" not in E.resolve_lines(sched, V.current(), "chill", 0)["outro"]
    assert CS.load()["rev"] == 1                               # all of it unsaved until 💾
    st.session_state.clear()


def test_copy_chill_to_medium_turns_and_the_gaps_tab(voice):
    import edge_mode as E
    import cue_store as CS
    import cue_map_view as V
    from tests.apptest_util import errors, page_app
    cm = CS.load()
    new, rep = CS.copy_chill_to_medium(cm)
    assert rep["copied"] > 40 and CS.is_split(new) and not CS.is_split(cm)
    for d in ("chill", "hard"):                                   # Chill and Hard play exactly as before
        for seed in range(3):
            sch = E.schedule(d, 4, seed)
            a, b = E.resolve_lines(sch, cm, d, seed), E.resolve_lines(sch, new, d, seed)
            assert [(p["line"], p["time"]) for p in a["plays"]] == [(p["line"], p["time"]) for p in b["plays"]]
    sch = E.schedule("medium", 3, 0)
    got = E.resolve_lines(sch, new, "medium", 0)
    secs = V.sections(sch, got)
    r1 = [x for x in secs if x["round"] == 1]
    assert [x["profile"] for x in r1] == ["wait", "slow", "medium", "fast", "slow", "medium", "fast", "extra", "stop"]
    assert all(not x["lines"] for x in r1[4:7])                   # the 2nd Slow / Medium / Fast start empty
    assert not any(p["why"].startswith(("callout", "fills")) for p in got["plays"])   # Medium's auto fillers off
    # a cue on the 2nd Slow plays there
    line = next(ln for ln in new["lines"] if ln["name"] == "Keep going")
    line["cues"].append({"type": "phase", "phases": ["slow"], "at": "start", "offset": 2, "turn": 2,
                         "roundList": ["1"], "diffs": ["medium"]})
    got = E.resolve_lines(sch, new, "medium", 0)
    slow2 = [x for x in V.sections(sch, got) if x["round"] == 1][4]
    assert any(p["line"] == "Keep going" and abs(p["time"] - (slow2["start"] + 2)) < 0.01 for p in got["plays"])
    assert "2nd Slow" in CS.describe(line["cues"][-1])
    assert not any(p["line"] == "Keep going" and p["round"] == 1 and p["why"].startswith("start of Slow")
                   for p in E.resolve_lines(E.schedule("chill", 3, 0), new, "chill", 0)["plays"])   # Chill has no 2nd turn
    # dropping on the 2nd Medium of round 2 makes a Medium-only, 2nd-turn cue
    r2 = [x for x in V.sections(sch, got) if x["round"] == 2]
    cue = V._phase_cue_at(sch, 2, r2[5]["start"] + 3.0, 3, "medium", True)
    assert cue["turn"] == 2 and cue["phases"] == ["medium"] and cue["diffs"] == ["medium"] and cue["offset"] == 3.0
    # the page: the Gaps tab with the copy button, then the grid
    CS.save(cm, 0, "taken over")
    at = page_app("pages/8_Cue_Map.py")
    at.session_state["cm_view"] = "🔍 Gaps"
    at.run()
    assert not errors(at)
    next(b for b in at.button if "Wipe Medium and copy Chill" in b.label).click().run()
    assert not errors(at) and CS.is_split(CS.load_draft()["map"]) and CS.load()["rev"] == 1
    assert any("Medium now has its own copy" in i.value for i in at.info)


def test_easy_wins_for_an_empty_section(voice):
    import json
    import edge_mode as E
    import cue_store as CS
    import cue_map_view as V
    import studio as SD
    from tests.apptest_util import errors, page_app
    new, _ = CS.copy_chill_to_medium(CS.load())
    CS.save(new, 0, "split")
    sch = E.schedule("medium", 6, 0)
    r1 = {p["line"] for p in E.resolve_lines(sch, new, "medium", 0)["plays"] if p["round"] == 1}
    pick = next(ln["name"] for ln in new["lines"] if ln["name"] not in r1 and ln.get("cues") and CS.played_seconds(ln) < 5
                and not any(c["type"] in ("intro", "outro", "surprise") for c in ln["cues"]) and ln["name"] != "Just stroke")
    clips = [{"id": "1", "name": pick + ".mp4", "size": 1, "folder": "Default - Lying on bed 1/avatar_clips"},
             {"id": "2", "name": "Just stroke.mp4", "size": 1, "folder": "Nude - Lying Back/avatar_clips"}]
    SD.cache_path().parent.mkdir(parents=True, exist_ok=True)
    SD.cache_path().write_text(json.dumps({"at": "now", "clips": clips}))
    got = E.resolve_lines(sch, CS.load(), "medium", 0)
    sec = [x for x in V.sections(sch, got) if x["round"] == 1][4]          # Round 1, Slow 2 — empty
    assert not sec["lines"]
    good, other = V.candidates(CS.load(), sec, got, "Default - Lying on bed 1", SD.recordings(clips))
    assert [g["name"] for g in good] == [pick]
    assert any(o["name"] == "Just stroke" and "Nude" in o["where"] for o in other)
    at = page_app("pages/8_Cue_Map.py")
    at.session_state["cm_view"] = "🔍 Gaps"
    at.session_state["cm_gp_grid_medium_6_False"] = {"selection": {"rows": [], "columns": [], "cells": [(0, "Slow 2")]}}
    at.run()
    assert not errors(at)
    assert any("Round 1 · Slow 2" in (m.value or "") for m in at.markdown)
    next(b for b in at.button if b.label == "➕ Here").click().run()
    assert not errors(at)
    line = next(ln for ln in CS.load_draft()["map"]["lines"] if ln["name"] == pick)
    assert line["cues"][-1]["turn"] == 2 and line["cues"][-1]["diffs"] == ["medium"] and line["cues"][-1]["phases"] == ["slow"]


def test_dragging_moves_only_the_round_it_came_from(voice):
    import edge_mode as E
    import cue_store as CS
    import cue_map_view as V
    cm = CS.load()
    CS.save(cm, 0, "taken over")
    cm = V.current(reload=True)
    sch = E.schedule("chill", 3, 0)
    name = "Keep going"
    line = next(ln for ln in cm["lines"] if ln["name"] == name)
    line["cues"] = [{"type": "phase", "phases": ["fast"], "at": "start", "offset": 2, "roundList": [], "diffs": [], "note": ""}]
    V.changed(cm, line)
    before = [p for p in E.resolve_lines(sch, V.current(), "chill", 0)["plays"] if p["line"] == name]
    assert 1 in {p["round"] for p in before}
    V.apply_action(V.current(), {"act": "move", "line": name, "key": [name, 0], "from": 1, "round": 1, "time": 60 + 6}, sch)
    got = [p for p in E.resolve_lines(sch, V.current(), "chill", 0)["plays"] if p["line"] == name]
    r1 = [p for p in got if p["round"] == 1]
    assert len(r1) == 1 and abs(r1[0]["time"] - 66) < 0.01                       # round 1 moved to Fast +6 s
    assert sorted(p["round"] for p in got) == sorted(p["round"] for p in before)   # other rounds untouched, no copies
    cues = next(ln for ln in V.current()["lines"] if ln["name"] == name)["cues"]
    assert cues[0]["roundList"] == ["2", "3", "4", "5", "6", "last"] and cues[1]["roundList"] == ["1"]
    # a round-1-only cue just moves
    V.apply_action(V.current(), {"act": "move", "line": name, "key": [name, 1], "from": 1, "round": 1, "time": 60 + 8}, sch)
    cues = next(ln for ln in V.current()["lines"] if ln["name"] == name)["cues"]
    assert len(cues) == 2 and cues[1]["phases"] == ["fast"] and cues[1]["offset"] == 8.0
    # a "start of the first round" cue becomes a round-1 phase cue — not every round
    line = next(ln for ln in V.current()["lines"] if ln["name"] == name)
    line["cues"] = [{"type": "round", "which": "first", "at": "start"}]
    V.changed(V.current(), line)
    V.apply_action(V.current(), {"act": "move", "line": name, "key": [name, 0], "from": 1, "round": 1, "time": 70}, sch)
    got = [p for p in E.resolve_lines(sch, V.current(), "chill", 0)["plays"] if p["line"] == name]
    assert [p["round"] for p in got] == [1]
