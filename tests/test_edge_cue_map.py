"""🎙️ Edge Mode's cue map: when every line plays, never two at once, takes, joined lines, the map sent from the Cue Map page."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.edge_support import (DATA, VIDS, E_CARD_LEAD, _check_plan, _emap, _level, _pixel, _seconds,  # noqa: F401
                                _timelines, _wait_job)


def test_the_cue_map_gives_every_line_its_moment_and_never_two_at_once():
    import edge_mode as E
    cm = E.load_cue_map()
    assert len(cm["lines"]) == 86 and sum(1 for ln in cm["lines"] if ln["cues"]) == 85          # the 17 Nov map (New folder lines)
    assert cm.get("subfolders") == ["New"]
    assert E.PROFILES["extra"]["bpm"] == 200                                       # a little faster than at first (180)
    for diff in E.DIFFICULTIES:
        for rounds in (3, 4, 5, 6):
            for seed in range(1 if diff != "hard" else 12):
                sched = E.schedule(diff, rounds, seed)
                got = E.resolve_lines(sched, cm, diff, seed)
                pl = got["plays"]
                assert pl and all(0 <= p["time"] and p["time"] + p["seconds"] <= E.total_seconds(sched) + 1e-6 for p in pl)
                for a, b in zip(pl, pl[1:]):
                    assert b["time"] >= a["time"] + a["seconds"] + E.LINE_GAP - 1e-3, (diff, rounds, a, b)
                assert got["intro"] == ["Welcome", "Intro to NNN"] and got["outro"] == ["I hope not", "You can survive 30 days"]
    got = E.resolve_lines(E.schedule("chill", 6), cm, "chill")
    at = {(p["line"], p["round"]): p for p in got["plays"]}
    assert at[("Start-Round 1", 1)]["time"] == 0.0 and at[("Now stop", 1)]["time"] == 100.0
    assert ("Now stop", 2) not in at                                               # set for round 1 only
    assert at[("Now stop (2)", 5)]["time"] == 4 * 120 + 97.0                       # 3 s before Extra ends
    assert at[("Stroke faster", 2)]["time"] == 120 + 40.0                          # as Medium starts
    p = at[("I love edging you", 4)]
    assert 3 * 120 + 100 <= p["time"] and p["time"] + p["seconds"] <= 3 * 120 + 120       # somewhere during Stop
    last = at[("Did you cum", 6)]
    assert abs(last["time"] + last["seconds"] - 720.0) < 1e-6 and last["moved"] < 0       # finishes as the compilation ends
    assert [r for (n, r) in at if n == "You can survive"] == [5]
    assert sorted(r for (n, r) in at if n == "Keep going") == [1, 2, 3, 4, 5, 6]      # a filler in every round's Medium
    assert all(at[("Keep going", r)]["profile"] == "medium" for r in range(1, 7))
    # "last" means the final round whatever its number — and there its lines replace that round's own
    for rounds in (3, 4, 5):
        g = E.resolve_lines(E.schedule("chill", rounds), cm, "chill")
        names = {p["line"] for p in g["plays"] if p["round"] == rounds}
        assert {"Did you cum", "Final round prepare yourself", "Its so much fun watching you shake"} <= names
        own = {3: {"Just sit there", "Ok stroke again"}, 4: {"I know you can do it", "relax-fingertips", "How stroke even more"},
               5: {"Balls bursting", "And dont cum", "You can survive"}}[rounds]
        assert not names & own
    five = {p["line"] for p in got["plays"] if p["round"] == 5}
    assert {"Balls bursting", "Now stop (2)", "And dont cum"} <= five               # round 5 keeps its own when it isn't the last
    only_hard = {"lines": [{"name": "x", "file": "x.mp3", "seconds": 2.0, "cues": [
        {"type": "phase", "phases": ["fast"], "at": "start", "offset": 0, "roundList": [], "diffs": ["hard"]}]}]}
    assert E.resolve_lines(E.schedule("chill", 3), only_hard, "chill")["plays"] == []
    assert len(E.resolve_lines(E.schedule("chill", 3), only_hard, "hard")["plays"]) == 3
    rnd = {"lines": [{"name": "r", "file": "r.mp3", "seconds": 1.5, "cues": [
        {"type": "random", "times": 2, "per": "round", "phases": ["fast", "extra"], "roundList": [], "diffs": []}]},
        {"name": "i", "file": "i.mp3", "seconds": 1.0, "cues": [{"type": "interval", "every": 50, "startAfter": 0, "phases": [], "roundList": ["2"], "diffs": []}]},
        {"name": "t", "file": "t.mp3", "seconds": 1.0, "cues": [{"type": "time", "clock": "0:30", "from": "round", "roundList": [], "diffs": []}]},
        {"name": "e", "file": "e.mp3", "seconds": 3.0, "cues": [{"type": "round", "which": "last", "at": "end", "roundList": [], "diffs": []}]}]}
    got = E.resolve_lines(E.schedule("chill", 3), rnd, "chill", seed=3)
    by = {}
    for p in got["plays"]:
        by.setdefault(p["line"], []).append(p)
    assert len(by["r"]) == 6 and all(p["profile"] in ("fast", "extra") for p in by["r"])
    assert [p["time"] for p in by["t"]] == [30.0, 150.0, 270.0]
    assert [p["time"] for p in by["i"]] == [151.35, 200.0]           # every 50 s in round 2; at 150 the set-time line has the moment
    assert by["e"][0]["time"] == 357.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_lines_on_the_intro_and_outro_screens(tmp_path, monkeypatch):
    """Voiceover lines mapped to the intro / outro are read on their screens, one after another; the screens last as
    long as the lines."""
    import edge_mode as E
    import render_preview as RP
    proxies = tmp_path / "proxies"
    proxies.mkdir()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x003000:s=320x180:d=30:r=25",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(proxies / "vid_a.mp4")], check=True)
    monkeypatch.setattr(RP, "PROXY_DIR", proxies)
    monkeypatch.setattr(RP, "streaming_available", lambda: False)

    def tone(name, seconds):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=660:duration={seconds}",
                        "-af", "volume=6", str(tmp_path / name)], check=True)
        return str(tmp_path / name)
    a, b, c, mid = tone("a.mp3", 2), tone("b.mp3", 3), tone("c.mp3", 4), tone("mid.mp3", 2)
    sched = [{"index": 0, "round": 1, "profile": "wait", "start": 0.0, "end": 20.0}]
    plan = {"track_id": "edge_test", "selected_videos": ["vid_a"], "track_duration_sec": 20.0,
            "timeline": [{"track_time": [k * 5.0, k * 5.0 + 5.0], "scenes": [{"chain": [
                {"video_id": "vid_a", "scene_id": 0, "clip_start_sec": 1.0 + k, "clip_duration_sec": 5.0}]}]} for k in range(4)],
            "edge": {"difficulty": "chill", "rounds": 1, "schedule": sched, "playlist": [], "beat": {"on": True, "volume": 1.0},
                     "voiceover": {"on": True, "cues": [{"time": 10.0, "profile": "wait", "file": mid, "line": "mid"}]},
                     "overlay": {"corner": "top-left"}, "avatar": {"on": False, "cues": []},
                     "intro": {"on": True, "clip": None, "stats": [["Rounds", "1"]], "thumbs": [],
                               "lines": [{"line": "a", "file": a, "seconds": 2.0}, {"line": "b", "file": b, "seconds": 3.0}]},
                     "outro": {"on": True, "stats": [["Rounds", "1"]], "thumbs": [], "lines": [{"line": "c", "file": c, "seconds": 4.0}]}}}
    out = tmp_path / "cards.mp4"
    RP.render_plan_dict(plan, str(out), quick=True)
    intro = 2 * E_CARD_LEAD() + 2 + 3 + E.LINE_GAP
    outro = 2 * E_CARD_LEAD() + 4
    assert abs(_seconds(out) - (intro + 20 + outro)) < 0.5
    assert _level(out, 0.6, 2.2) > 0.05 and _level(out, 3.0, 5.6) > 0.05             # the two intro lines
    assert max(_pixel(out, 2.0, 630, 350)) < 60                                      # on the dark intro screen
    assert _level(out, intro + 2, intro + 8) < 0.01 < _level(out, intro + 10.2, intro + 11.8)       # Wait is silent but for the mid line
    assert _level(out, intro + 20 + 0.6, intro + 20 + 4.2) > 0.05 and max(_pixel(out, intro + 22, 630, 350)) < 60
    # a line with the avatar's clip: she is in her box while she reads it (her voice from the clip), and holds after
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=magenta:s=180x320:d=3:r=25",
                    "-f", "lavfi", "-i", "sine=frequency=880:duration=3", "-af", "volume=6", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-shortest", str(tmp_path / "b.mp4")], check=True)
    plan["edge"]["intro"]["lines"][1]["clip"] = str(tmp_path / "b.mp4")
    out = tmp_path / "cards_clip.mp4"
    RP.render_plan_dict(plan, str(out), quick=True)
    ax, ay, aw, ah = E.intro_layout((640, 360))["avatar"]
    mid = (ax + aw // 2, ay + ah // 2)
    r, g, bl = _pixel(out, 4.0, *mid)
    assert r > 170 and bl > 170 and g < 90 and abs(_seconds(out) - (intro + 20 + outro)) < 0.6
    assert not (_pixel(out, 1.2, *mid)[0] > 170 and _pixel(out, 1.2, *mid)[1] < 90)           # the stand-in while line a is read


def test_the_avatar_follows_the_cue_map(monkeypatch, tmp_path):
    out = _timelines(monkeypatch)
    try:
        import planner.edge as PE
        monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "avatar")
        (tmp_path / "avatar" / "lines").mkdir(parents=True)
        for n in ("Now stop", "Intro to NNN"):
            (tmp_path / "avatar" / "lines" / f"{n}.mp4").write_bytes(b"x")
        res = PE.compile_edge([], VIDS, "chill", 3, voiceover=True, avatar=True, intro=True)
        edge = res["plan"]["edge"]
        av, vo = edge["avatar"]["cues"], edge["voiceover"]["cues"]
        assert [(c["time"], c["line"]) for c in av] == [(c["time"], c["line"]) for c in vo]      # she appears with every line
        assert {c["line"] for c in av if c["clip"]} == {"Now stop"} and "Get hard for me" in edge["avatar"]["placeholders"]
        assert [x.get("clip") is not None for x in edge["intro"]["lines"]] == [False, True]      # her intro clip (Welcome: none)
        assert edge["music"]["still_volume"] == 0.25
        assert {c["look_used"] for c in av if c["line"] == "Now stop"} == {"stand-in"}
        # 👗 her recording in round 1's planned look wins over the stand-in (19 Nov)
        look = PE.load_looks(PE.load_cue_map())["1"]
        (tmp_path / "avatar" / "looks" / look).mkdir(parents=True)
        (tmp_path / "avatar" / "looks" / look / "Now stop.mp4").write_bytes(b"x")
        edge = PE.compile_edge([], VIDS, "chill", 3, voiceover=True, avatar=True, intro=True)["plan"]["edge"]
        now = [c for c in edge["avatar"]["cues"] if c["line"] == "Now stop"]
        assert now and all(c["look_used"] == "look" and f"/looks/{look}/" in c["clip"] for c in now)
        assert edge["avatar"]["looks"]["1"] == look
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_takes_are_found_by_name_and_take_turns():
    import edge_mode as E
    assert E.take_stem("Keep going take 2", "Keep going") == 2 and E.take_stem("keep going (3)", "Keep going") == 3
    assert E.take_stem("Keep going_alt4", "Keep going") == 4 and E.take_stem("Keep going v2", "Keep going") == 2
    assert E.take_stem("Keep going", "Keep going") is None and E.take_stem("Keep going faster", "Keep going") is None
    assert E.take_stem("Keep going take 1", "Keep going") is None
    plays = [{"line": "a"}] * 6 + [{"line": "b"}]
    got = E.pick_takes(plays, {"a": ["a1", "a2", "a3"]}, seed=4)
    assert sorted(got[:3]) == [0, 1, 2] and sorted(got[3:6]) == [0, 1, 2] and got[6] == 0


def test_the_cue_map_file_and_the_takes_on_this_machine(tmp_path, monkeypatch):
    import planner.edge as PE
    import edge_mode as E
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "edge_voiceover")
    monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "edge_avatar")
    assert PE.cue_map_path() == E.CUE_MAP_FILE             # nothing uploaded: the app's own
    assert PE.save_cue_map(b"not json") and PE.save_cue_map(b'{"lines": [{"x": 1}]}')
    assert PE.save_cue_map(b'{"lines": [{"name": "Keep going", "file": "Keep going.mp3", "cues": []}]}') is None
    assert PE.cue_map_path() == tmp_path / "edge_voiceover" / "edge_cues.json"
    assert PE.load_cue_map()["lines"][0]["name"] == "Keep going"
    lines, clips = tmp_path / "edge_voiceover" / "lines", tmp_path / "edge_avatar" / "lines"
    lines.mkdir(parents=True)
    clips.mkdir(parents=True)
    for f in (lines / "Keep going.mp3", lines / "Keep going take 3.mp3", lines / "Keep going (2).mp3",
              clips / "Keep going take 3.mp4", lines / "Other.mp3"):
        f.write_bytes(b"x")
    line = PE.load_cue_map()["lines"][0]
    got = PE.line_takes(line, with_avatar=True)
    assert [Path(t["file"]).name for t in got] == ["Keep going.mp3", "Keep going (2).mp3", "Keep going take 3.mp3"]
    assert got[2]["clip"].endswith("Keep going take 3.mp4") and got[1]["clip"] is None
    assert len(PE.line_takes(line)) == 3


def test_a_line_can_be_joined_to_another_and_plays_straight_after_it():
    import edge_mode as E
    cm = {"lines": [
        {"name": "a", "file": "a.mp3", "seconds": 3.0, "cues": [{"type": "phase", "phases": ["slow"], "at": "start", "offset": 0,
                                                                 "roundList": [], "diffs": []}]},
        {"name": "b", "file": "b.mp3", "seconds": 2.0, "cues": [{"type": "after", "after": "a", "gap": 0.35, "roundList": ["2"], "diffs": []}]},
        {"name": "c", "file": "c.mp3", "seconds": 1.0, "cues": [{"type": "after", "after": "b", "gap": 0.5, "roundList": [], "diffs": []}]},
        {"name": "d", "file": "d.mp3", "seconds": 2.0, "cues": [{"type": "phase", "phases": ["slow"], "at": "start", "offset": 3.5,
                                                                 "roundList": ["3"], "diffs": []}]}]}
    got = E.resolve_lines(E.schedule("chill", 3), cm, "chill")
    by = {(p["line"], p["round"]): p for p in got["plays"]}
    assert ("b", 1) not in by and by[("b", 2)]["time"] == 120 + 20 + 3.35        # round 2 only, as soon as "a" ends
    assert by[("c", 2)]["time"] == by[("b", 2)]["time"] + 2.5                    # a chain: c after b after a
    assert by[("b", 2)]["why"] == "straight after a"
    assert ("b", 3) not in by and not [c for c in got["clashes"] if c["line"] == "b"]   # (b is for round 2 only)
    for x, y in zip(got["plays"], got["plays"][1:]):
        assert y["time"] >= x["time"] + x["seconds"] + E.LINE_GAP - 1e-3


def test_the_newest_cue_map_sent_from_the_page_is_fetched_and_its_lines_too(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import planner.edge as PE
    assert PE.newest_cue_map([{"name": "edge_cues_2026-11-16T10-00.json", "modifiedTime": "2026-11-16T10:00:00Z"},
                              {"name": "edge_cues_2026-11-16T12-00.json", "modifiedTime": "2026-11-16T12:00:00Z"},
                              {"name": "edge_cues.json", "modifiedTime": "2026-11-14T02:00:00Z"},
                              {"name": "Welcome.mp3", "modifiedTime": "2026-11-17T00:00:00Z"}]) == "edge_cues_2026-11-16T12-00.json"
    assert PE.newest_cue_map([{"name": "a.mp3"}]) is None
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "edge_voiceover")
    monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "edge_avatar")
    sent = {"version": 1, "lines": [{"name": "New one", "file": "New one.mp3", "seconds": 1.2, "cues": [
        {"type": "after", "after": "Old one", "gap": 0.35, "roundList": [], "diffs": []}]},
        {"name": "Old one", "file": "Old one.mp3", "seconds": 2.0, "cues": []}]}
    got = []

    def download(rel, dest, **kw):
        got.append(rel)
        dest = __import__("pathlib").Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(sent) if rel.endswith(".json") else "x")
        return None
    fake = types.SimpleNamespace(credentials_available=lambda: True, download_source_video=download,
                                 list_drive_folder=lambda rel: ([{"name": "edge_cues_2026-11-16T12-00.json", "modifiedTime": "2026-11-16T12:00:00Z"},
                                                                 {"name": "New one.mp3"}, {"name": "Old one.mp3"}], None) if "avatar" not in rel else ([], None))
    monkeypatch.setitem(sys.modules, "drive_sync", fake)
    errs = PE.fetch_lines({"lines": [{"name": "Old one", "file": "Old one.mp3", "cues": []}]}, avatar=False, only_missing=True)
    assert not errs and got[0].endswith("edge_cues_2026-11-16T12-00.json")
    assert "scene-labeling/Edge_voiceover/New one.mp3" in got                # a line only the sent map has is fetched too
    assert PE.cue_map_path() == tmp_path / "edge_voiceover" / "edge_cues.json" and PE.load_cue_map()["lines"][0]["name"] == "New one"
    # 📁 a line whose file sits in the New subfolder is fetched from there (the top folder is looked in first)
    got.clear()
    sent["lines"].append({"name": "Hands off 8 A", "file": "Hands off 8 A.mp3", "seconds": 5.4, "cues": []})
    fake.list_drive_folder = lambda rel: (([{"name": "edge_cues_2026-11-17T01-00.json", "modifiedTime": "2026-11-17T01:00:00Z"},
                                            {"name": "Old one.mp3"}, {"name": "New one.mp3"}], None) if rel.endswith("Edge_voiceover")
                                          else ([{"name": "Hands off 8 A.mp3"}, {"name": "Old one.mp3"}], None) if rel.endswith("/New")
                                          else ([], None))
    errs = PE.fetch_lines({"lines": []}, avatar=False)
    assert not errs and "scene-labeling/Edge_voiceover/New/Hands off 8 A.mp3" in got
    assert "scene-labeling/Edge_voiceover/Old one.mp3" in got and (PE.lines_dir() / "Hands off 8 A.mp3").exists()
    # 👩 her clips: from avatar_clips and from an avatar_clips in another folder (sitting/avatar_clips — 18 Nov);
    # the same name in both: the top avatar_clips one
    got.clear()
    sent["lines"].append({"name": "Strong erection", "file": "Strong erection.mp3", "seconds": 8.5, "cues": []})
    fake.list_drive_subfolders = lambda rel: ((["avatar_clips", "sitting", "transitions", "_studio_previews"], None)
                                              if rel.endswith("Edge_voiceover") else
                                              (["avatar_clips"], None) if rel.endswith("/sitting") else ([], None))
    fake.list_drive_folder = lambda rel: (([{"name": "edge_cues_2026-11-18T01-00.json", "modifiedTime": "2026-11-18T01:00:00Z"},
                                            {"name": "Old one.mp3"}, {"name": "New one.mp3"}, {"name": "Strong erection.mp3"}], None)
                                          if rel.endswith("Edge_voiceover")
                                          else ([{"name": "Old one.mp4"}, {"name": "Unrelated.mp4"}], None) if rel.endswith("Edge_voiceover/avatar_clips")
                                          else ([{"name": "Strong erection.mp4"}, {"name": "Old one.mp4"}], None) if rel.endswith("sitting/avatar_clips")
                                          else ([], None))
    assert PE.avatar_clip_folders("scene-labeling/Edge_voiceover") == ["avatar_clips", "sitting/avatar_clips"]
    errs = PE.fetch_lines({"lines": []}, avatar=True)
    assert not errs, errs
    clips = [g for g in got if g.endswith(".mp4")]
    assert clips == ["scene-labeling/Edge_voiceover/avatar_clips/Old one.mp4",
                     "scene-labeling/Edge_voiceover/sitting/avatar_clips/Strong erection.mp4"]
    assert (PE.avatar_dir() / "lines" / "Strong erection.mp4").exists()


def test_surprise_lines_play_at_their_surprise_and_fit_its_length():
    import edge_mode as E
    sched = next(s for s in (E.schedule("medium", 4, k, surprises=True) for k in range(80))
                 if any(p["kind"] == "handsoff" for p in s) and any(p.get("surprise") == "falsefinish" for p in s))
    hos = [p for p in sched if p["kind"] == "handsoff"]
    ff = next(p for p in sched if p.get("surprise") == "falsefinish")
    lines = [{"name": f"Hands off {n}", "seconds": 1.5, "cues": [{"type": "surprise", "kind": "handsoff", "at": "start", "length": n}]}
             for n in (5, 8, 12)]
    lines += [{"name": "Back on", "seconds": 1.2, "cues": [{"type": "surprise", "kind": "handsoff", "at": "end"}]},
              {"name": "Almost 321", "seconds": 3.8, "cues": [{"type": "surprise", "kind": "falsefinish", "at": "before"}]},
              {"name": "Scratch that", "seconds": 8.6, "cues": [{"type": "surprise", "kind": "falsefinish", "at": "start"}]},
              # an ordinary line wanting the same moment as a hands off: the surprise's line wins
              {"name": "Busy", "seconds": 3.0, "cues": [{"type": "time", "clock": E.fmt_clock(hos[0]["start"]) if hasattr(E, "fmt_clock") else
                                                          f"{int(hos[0]['start'] // 60)}:{hos[0]['start'] % 60:05.2f}"}]}]
    got = E.resolve_lines(sched, {"lines": lines}, "medium", seed=3)
    for ho in hos:
        n = ho["end"] - ho["start"]
        start = [p for p in got["plays"] if abs(p["time"] - ho["start"]) < 1e-6]
        assert [p["line"] for p in start] == [f"Hands off {n:g}"]                   # the line made for its length
        assert any(p["line"] == "Back on" and abs(p["time"] + p["seconds"] - ho["end"]) < 1e-6 for p in got["plays"])
    assert any(p["line"] == "Almost 321" and abs(p["time"] + 3.8 - ff["start"]) < 1e-6 for p in got["plays"])
    assert any(p["line"] == "Scratch that" and -1e-6 < p["time"] - ff["start"] <= E.LINE_GAP + 1e-6 for p in got["plays"])
    busy = [p for p in got["plays"] if p["line"] == "Busy"]
    assert not busy or busy[0]["time"] >= hos[0]["start"] + 1.5                       # moved off, or left out
    # "end of Extra" waits for an override to finish: it plays as Stop really starts
    sched2 = next(s for s in (E.schedule("medium", 3, k, surprises=True) for k in range(80))
                  if any(p.get("surprise") in ("override", "falsefinish") and p["profile"] == "extra" for p in s))
    ov = next(p for p in sched2 if p.get("surprise") in ("override", "falsefinish") and p["profile"] == "extra")
    got = E.resolve_lines(sched2, {"lines": [{"name": "Now stop", "seconds": 1.6,
                                              "cues": [{"type": "phase", "phases": ["extra"], "at": "end", "roundList": [ov["round"]]}]}]},
                          "medium")
    assert abs(got["plays"][0]["time"] - ov["end"]) < 1e-6


def test_fetch_lines_finds_lines_in_any_look_folder(monkeypatch, tmp_path):
    """📁 The sent map says "Lying Back" but the folder is now "Nude - Lying Back": the line is still found (18 Nov)."""
    import sys
    import types
    import planner.edge as PE
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "edge_voiceover")
    monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "edge_avatar")
    got = []

    def download(rel, dest, **kw):
        got.append(rel)
        dest = __import__("pathlib").Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("x")
        return None
    files = {"Edge_voiceover": [{"name": "Welcome.mp3"}],
             "Edge_voiceover/Nude - Lying Back": [{"name": "Hands off 5 B.mp3"}, {"name": "Hands off 5 B (2).mp3"}],
             "Edge_voiceover/December": [{"name": "Hands off 5 B.mp3"}]}
    fake = types.SimpleNamespace(
        credentials_available=lambda: True, download_source_video=download,
        list_drive_folder=lambda rel: (files.get(rel.split("scene-labeling/")[-1], []), None),
        list_drive_subfolders=lambda rel: ((["December", "Nude - Lying Back", "_cuemap_live", "avatar_clips", "transitions"], None)
                                           if rel.endswith("Edge_voiceover") else ([], None)))
    monkeypatch.setitem(sys.modules, "drive_sync", fake)
    cm = {"subfolders": ["Lying Back"], "lines": [{"name": "Hands off 5 B", "file": "Hands off 5 B.mp3", "cues": []},
                                                  {"name": "Welcome", "file": "Welcome.mp3", "cues": []}]}
    assert PE.voice_folders("scene-labeling/Edge_voiceover", cm) == ["Lying Back", "New", "Nude - Lying Back"]
    errs = PE.fetch_lines(cm, avatar=False)
    assert not errs
    assert "scene-labeling/Edge_voiceover/Nude - Lying Back/Hands off 5 B.mp3" in got
    assert "scene-labeling/Edge_voiceover/Welcome.mp3" in got
    assert not any("December" in g for g in got)


def test_her_clips_are_fetched_per_look_and_the_round_plan_picks_them(monkeypatch, tmp_path):
    """👗 The planned-look fix (19 Nov): each look folder's clips are kept by look; a line plays in the look planned for
    its round, else the stand-in (avatar_clips first), else voice only. "Last round" takes the final round."""
    import sys
    import types
    import planner.edge as PE
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "edge_voiceover")
    monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "edge_avatar")

    def download(rel, dest, **kw):
        dest = __import__("pathlib").Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rel)
        return None
    files = {"Edge_voiceover": [{"name": "Round 4.mp3"}, {"name": "Slow.mp3"}],
             "Edge_voiceover/avatar_clips": [{"name": "Round 4.mp4"}, {"name": "Slow.mp4"}],
             "Edge_voiceover/Nude - Lying Back/avatar_clips": [{"name": "Round 4.mp4"}],
             "Edge_voiceover/Black Panties - Sitting on Bed1/avatar_clips": [{"name": "Round 4.mp4"}]}
    subs = {"Edge_voiceover": ["avatar_clips", "Nude - Lying Back", "Black Panties - Sitting on Bed1", "sitting"],
            "Edge_voiceover/Nude - Lying Back": ["avatar_clips"],
            "Edge_voiceover/Black Panties - Sitting on Bed1": ["avatar_clips"], "Edge_voiceover/sitting": ["avatar_clips"]}
    fake = types.SimpleNamespace(
        credentials_available=lambda: True, download_source_video=download,
        list_drive_folder=lambda rel: (files.get(rel.split("scene-labeling/")[-1], []), None),
        list_drive_subfolders=lambda rel: (subs.get(rel.split("scene-labeling/")[-1], []), None))
    monkeypatch.setitem(sys.modules, "drive_sync", fake)
    cm = {"lines": [{"name": "Round 4", "file": "Round 4.mp3", "cues": []}, {"name": "Slow", "file": "Slow.mp3", "cues": []}]}
    assert not PE.fetch_lines(cm)
    stand = tmp_path / "edge_avatar" / "lines" / "Round 4.mp4"
    assert stand.read_text().endswith("Edge_voiceover/avatar_clips/Round 4.mp4")          # the stand-in: avatar_clips first
    assert PE.local_looks() == {"Black Panties - Sitting on Bed1": 1, "Nude - Lying Back": 1}
    looks = PE.load_looks()
    assert looks == PE.DEFAULT_LOOKS
    assert PE.look_for(3, 6, looks) == "Black Panties - Sitting on Bed1"
    assert PE.look_for(3, 3, looks) == "Nude - Lying Back"                                 # the final round = "Last round"
    assert PE.look_for("outro", 3, looks) == "Nude - Lying on Back"
    clip, how = PE.pick_look_clip("Round 4", PE.look_for(4, 6, looks), stand)
    assert how == "look" and "Black Panties" in clip
    clip, how = PE.pick_look_clip("Round 4", PE.look_for(1, 6, looks), stand)
    assert how == "stand-in" and clip == str(stand)                                        # no Default clip yet
    assert PE.pick_look_clip("Nothing", "Nude - Lying Back", None) == (None, "voice")
    PE.save_looks({**looks, "1": "Nude - Lying Back"})
    assert PE.load_looks()["1"] == "Nude - Lying Back"
    assert PE.pick_look_clip("Round 4", PE.load_looks()["1"], stand)[1] == "look"


def test_medium_and_hard_have_no_long_silences_and_fillers_find_room():
    """🔁 19 Nov coverage check: a filler that can't fit its turn tries the round's other turns (A), Hard gets callouts
    at phase starts (C) and every silence over FILL_GAP inside a speaking phase gets a line (B) — never two at once,
    never the same filler / callout twice within NO_REPEAT_SEC."""
    import edge_mode as E
    cm = E.load_cue_map()
    for diff in ("medium", "hard"):
        for seed in range(6):
            sched = E.schedule(diff, 5, seed, surprises=diff == "medium")
            got = E.resolve_lines(sched, cm, diff, seed)
            pl, T = got["plays"], E.total_seconds(sched)
            for a, b in zip(pl, pl[1:]):
                assert b["time"] >= a["time"] + a["seconds"] + E.LINE_GAP - 1e-3, (diff, a, b)
            t, worst = 0.0, 0.0
            for p in pl + [{"time": T, "seconds": 0}]:
                mid = (t + p["time"]) / 2
                ph = E.phase_at(sched, mid)
                if ph and ph["profile"] in E.FILL_PHASES and p["time"] > t:
                    worst = max(worst, p["time"] - t)
                t = max(t, p["time"] + p["seconds"])
            assert worst < 40, (diff, seed, worst)
            added = [p for p in pl if p["why"].startswith(("callout", "fills"))]
            assert added and all(p["profile"] in E.FILL_PHASES for p in added)
            by = {}
            for p in added:
                by.setdefault(p["line"], []).append(p["time"])
            for times in by.values():
                assert all(b - a >= E.NO_REPEAT_SEC - 1e-6 for a, b in zip(sorted(times), sorted(times)[1:]))
            if diff == "medium":
                assert not any(p["why"].startswith("callout") for p in pl)      # callouts are Hard's
    chill = E.resolve_lines(E.schedule("chill", 4, 1), cm, "chill", 1)["plays"]
    assert not any(p["why"].startswith(("callout", "fills")) for p in chill)      # Chill untouched
