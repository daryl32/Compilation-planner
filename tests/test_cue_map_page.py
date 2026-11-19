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


def test_the_page_takes_over_edits_a_line_and_refuses_a_stale_save(voice):
    import cue_store as CS
    import cue_map_view as V
    from tests.apptest_util import errors, page_app
    at = page_app("pages/8_Cue_Map.py").run()
    assert not errors(at)
    assert any("Keep the cue map in the app" in b.label for b in at.button)
    next(b for b in at.button if "Keep the cue map in the app" in b.label).click().run()
    assert not errors(at) and CS.managed() and CS.load()["rev"] == 1
    at.session_state[V.SS_SEL] = None
    V_line = "Start-Round 1"
    at.run()
    # open a line through the session (as a tap does), change its note, save
    cm = CS.load()
    import copy
    at.session_state[V.SS_MAP] = cm
    at.session_state[V.SS_SEL] = V_line
    at.session_state[V.SS_DRAFT] = V._draft_of(next(ln for ln in cm["lines"] if ln["name"] == V_line))
    at.run()
    assert not errors(at)
    ta = next(t for t in at.text_area if t.label.startswith("Notes"))
    ta.input("Say it slower").run()
    save = next(b for b in at.button if b.label == "💾 Save")
    save.click().run()
    assert not errors(at)
    got = next(ln for ln in CS.load()["lines"] if ln["name"] == V_line)
    assert got["notes"] == "Say it slower" and CS.load()["rev"] == 2
    # another device saves in between: this page's next save is refused, nothing lost
    other = copy.deepcopy(CS.load())
    CS.save(other, 2, "other device")
    ta = next(t for t in at.text_area if t.label.startswith("Notes"))
    ta.input("Mine").run()
    next(b for b in at.button if b.label == "💾 Save").click().run()
    assert any("Not saved" in e.value for e in at.error)
    assert next(ln for ln in CS.load()["lines"] if ln["name"] == V_line)["notes"] == "Say it slower"


def test_end_cues_keep_their_finish_when_the_length_changes():
    import cue_map_view as V
    line = {"name": "a", "seconds": 10.0, "trim_in": 0, "trim_out": 0, "speed": 1,
            "cues": [{"type": "phase", "phases": ["stop"], "at": "end", "offset": 12.0}]}
    d = V._draft_of(line)
    assert d["cues"][0]["_finish"] == 2.0                  # starts 12 s before the end, 10 s long → finishes 2 s before
    d["trim_out"] = 4.0
    out = V._commit(d)
    assert out["cues"][0]["offset"] == 6.0 and "_finish" not in out["cues"][0] and out["play_seconds"] == 4.0
