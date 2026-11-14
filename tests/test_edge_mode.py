"""⏱️ Edge Mode: the rounds and phases by difficulty, the generated beat, the playlist laid end to end, a compile
with and without songs, the render (overlay + mixed sound, real ffmpeg on tiny files), and the page."""
from __future__ import annotations

import shutil
import subprocess
import wave

import numpy as np
import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.test_story_matcher import _setup as _timelines

VIDS = ["vid_a", "vid_b", "vid_c", "vid_f"]


def test_rounds_follow_the_difficulty():
    import edge_mode as E
    assert E.round_phases("chill") == ["wait", "slow", "medium", "fast", "extra", "stop"]
    assert E.round_phases("medium") == ["wait", "slow", "medium", "fast", "slow", "medium", "fast", "extra", "stop"]
    assert (E.round_seconds("chill"), E.round_seconds("medium"), E.round_seconds("hard")) == (120.0, 180.0, 300.0)
    counts = {p: 0 for p in E.ORDER}
    for seed in range(200):
        sched = E.schedule("hard", 3, seed)
        assert len(sched) == 45 and E.total_seconds(sched) == 900.0
        for r in (1, 2, 3):
            ph = [p["profile"] for p in sched if p["round"] == r]
            assert ph[0] == "wait" and ph[-2:] == ["extra", "stop"] and len(ph) == 15
            mid = ph[1:-2]
            assert set(mid) <= {"slow", "medium", "fast", "extra"}
            assert not any(a == b == c for a, b, c in zip(mid, mid[1:], mid[2:]))      # never three the same running
            for p in mid:
                counts[p] += 1
    assert counts["slow"] < min(counts["medium"], counts["fast"], counts["extra"])     # the weights lean away from Slow
    assert E.schedule("hard", 4, 5) == E.schedule("hard", 4, 5) != E.schedule("hard", 4, 6)
    # rounds are held to 3–6, phases are back to back
    assert E.schedule("chill", 1)[-1]["round"] == 3 and E.schedule("chill", 9)[-1]["round"] == 6
    sched = E.schedule("medium", 3)
    assert all(abs(a["end"] - b["start"]) < 1e-9 for a, b in zip(sched, sched[1:]))


def test_the_generated_beat_follows_the_phases(tmp_path):
    import edge_mode as E
    sched = E.schedule("chill", 3)
    beats = np.array(E.beat_times(sched))
    for ph in sched[:6]:
        inside = beats[(beats >= ph["start"]) & (beats < ph["end"])]
        bpm = E.PROFILES[ph["profile"]]["bpm"]
        assert len(inside) == (round(20 * bpm / 60) if bpm else 0)
        if bpm:
            assert abs(inside[0] - ph["start"]) < 1e-6 and abs(np.diff(inside).mean() - 60 / bpm) < 1e-3
    track = E.beat_track(sched)
    assert track["duration_sec"] == 360.0 and len(track["hires"]["rms"]) == 360 * E.CURVE_RATE
    segs = E.beat_segments(sched)
    assert segs[0]["start"] == 0.0 and segs[-1]["end"] == 360.0
    assert all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(segs, segs[1:]))
    edges = {round(s["start"], 3) for s in segs}
    assert {ph["start"] for ph in sched} <= edges                    # a phase change is always a cut
    by = {}
    for s in segs:
        by.setdefault(s["edge_profile"], []).append(s["end"] - s["start"])
    assert all(2.0 <= d <= 5.7 for p in ("slow", "medium", "fast", "extra") for d in by[p]) and set(by["wait"]) == {10.0}
    assert {s["bpm"] for s in segs if s["edge_profile"] == "fast"} == {140.0}
    wav = E.write_beat_wav(sched[:3], tmp_path / "beat.wav")
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768
        assert w.getframerate() == E.AUDIO_RATE and abs(len(x) / E.AUDIO_RATE - 60.0) < 0.01
    r = E.AUDIO_RATE
    assert np.abs(x[:20 * r]).max() == 0 and np.abs(x[20 * r:40 * r]).max() > 0.3          # Wait silent, Slow beats
    # the cue stand-ins differ by phase; the overlay is drawn for each phase in use
    assert E.write_placeholder_cue("extra", tmp_path / "e.wav").stat().st_size != E.write_placeholder_cue("wait", tmp_path / "w.wav").stat().st_size
    lst, size = E.write_overlay(sched[:2], tmp_path / "ov", 360)
    assert size[0] % 2 == 0 and size[1] % 2 == 0 and len(list((tmp_path / "ov").glob("*.png"))) == 2 * 20 * E.OVERLAY_FPS
    assert lst.read_text().count("duration") == 2 * 20 * E.OVERLAY_FPS
    full, low = E.overlay_image("extra", 20, size), E.overlay_image("extra", 2, size)
    amber = lambda im: int(((np.asarray(im)[..., 0] > 200) & (np.asarray(im)[..., 2] < 140) & (np.asarray(im)[..., 1] > 150)).sum())   # noqa: E731
    assert amber(full) > 4 * amber(low) > 0                           # the ring empties


def test_the_playlist_repeats_until_the_time_is_filled():
    import edge_mode as E
    parts = E.playlist_parts([("a", 100.0), ("b", 50.0)], 360.0)
    assert [(p["track_id"], p["offset"], p["play"], p["pass"]) for p in parts] == [
        ("a", 0.0, 100.0, 1), ("b", 100.0, 50.0, 1), ("a", 150.0, 100.0, 2), ("b", 250.0, 50.0, 2), ("a", 300.0, 60.0, 3)]
    assert E.playlist_parts([("a", 500.0)], 360.0) == [{"track_id": "a", "offset": 0.0, "play": 360.0, "pass": 1}]
    assert sum(p["play"] for p in E.playlist_parts([("a", 119.9)], 360.0)) == pytest.approx(359.7)   # a sliver starts no song
    assert E.playlist_parts([], 360.0) == []
    cues = E.voice_cues(E.schedule("chill", 3), {"slow": ["s1.mp3", "s2.mp3"]}, seed=1)
    assert len(cues) == 18 and [c["time"] for c in cues[:3]] == [0.0, 20.0, 40.0]
    slow = [c["file"] for c in cues if c["profile"] == "slow"]
    assert set(slow[:2]) == {"s1.mp3", "s2.mp3"} and all(c["file"] is None for c in cues if c["profile"] != "slow")


def _check_plan(plan, total):
    tl = plan["timeline"]
    assert tl[0]["track_time"][0] == 0.0 and abs(tl[-1]["track_time"][1] - total) < 1e-6
    for a, b in zip(tl, tl[1:]):
        assert abs(a["track_time"][1] - b["track_time"][0]) < 1e-6
    for e in tl:
        d = e["track_time"][1] - e["track_time"][0]
        assert d > 0 and e["edge_profile"] in ("wait", "slow", "medium", "fast", "extra", "stop")
        for s in e["scenes"]:
            assert sum(link["clip_duration_sec"] for link in s["chain"]) <= d + 0.05
            assert all("source_path" in link for link in s["chain"])
    assert [e["segment_index"] for e in tl] == list(range(len(tl)))


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
        assert shown["on"] and [x["line"] for x in shown["lines"]] == ["Welcome", "Intro to NNN"]
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


def _level(path, a, b):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(a), "-t", str(b - a), "-i", str(path), "-vn", "-ac", "1",
                          "-ar", "8000", "-f", "s16le", "-"], capture_output=True, check=True)
    x = np.frombuffer(out.stdout, "<i2").astype(float) / 32768
    return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0


def _pixel(path, t, x, y):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-vf", f"crop=2:2:{x}:{y},scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True)
    return tuple(out.stdout[:3])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_render_carries_the_timer_and_the_mixed_sound(tmp_path, monkeypatch):
    import edge_mode as E
    import render_preview as RP
    from tests.fixtures import load_track
    proxies, drive = tmp_path / "proxies", tmp_path / "drive"
    proxies.mkdir()
    for vid, colour in (("vid_a", "0x003000"), ("vid_b", "0x000030")):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=320x180:d=30:r=25",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(proxies / f"{vid}.mp4")], check=True)
    real = RP.colab_to_local
    monkeypatch.setattr(RP, "colab_to_local", lambda p: str(drive / p.split("MyDrive/", 1)[1]) if "MyDrive/" in p else real(p))
    monkeypatch.setattr(RP, "PROXY_DIR", proxies)
    monkeypatch.setattr(RP, "streaming_available", lambda: False)
    track = load_track(DATA, "song_full")
    audio = drive / RP.get_track_raw_source(track["track_id"]).split("MyDrive/", 1)[1]
    audio.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=25",
                    "-af", "volume=4", str(audio.with_suffix(".wav"))], check=True)
    audio.with_suffix(".wav").rename(audio)
    sched = [{"index": 0, "round": 1, "profile": "wait", "start": 0.0, "end": 20.0},
             {"index": 1, "round": 1, "profile": "extra", "start": 20.0, "end": 40.0}]
    timeline = [{"track_time": [k * 5.0, k * 5.0 + 5.0],
                 "scenes": [{"chain": [{"video_id": "vid_a" if k % 2 == 0 else "vid_b", "scene_id": 0,
                                        "clip_start_sec": 1.0 + k, "clip_duration_sec": 5.0}]}]} for k in range(8)]

    def plan(playlist, beat, voice, corner="top-left"):
        return {"track_id": "edge_test", "selected_videos": ["vid_a", "vid_b"], "timeline": timeline,
                "track_duration_sec": 40.0,
                "edge": {"difficulty": "chill", "rounds": 1, "schedule": sched, "playlist": playlist,
                         "beat": {"on": beat, "volume": 1.0},
                         "voiceover": {"on": voice, "music_volume": 0.25, "cues": E.voice_cues(sched, {}) if voice else []},
                         "overlay": {"corner": corner}}}
    # no song: the beat is the sound — silent in Wait, beating in Extra; the timer sits top left
    out = tmp_path / "beat.mp4"
    RP.render_plan_dict(plan([], True, False), str(out), quick=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True)
    assert abs(float(probe.stdout) - 40.0) < 0.3
    assert _level(out, 2, 18) < 0.002 and _level(out, 22, 38) > 0.02
    w, h = E.overlay_size(360)
    cx, top = int(360 * 0.035) + w // 2, int(360 * 0.035)
    assert _pixel(out, 1.0, cx, top + 4)[0] > 150                      # the amber ring, full at the start…
    assert _pixel(out, 18.0, cx - 20, top + 5)[0] < 130                # …and nearly empty at the phase's end
    label_y = top + h - int(w * 0.15)
    r, g, b = _pixel(out, 30.0, cx - int(w * 0.3), label_y)            # Extra's red label
    assert r > 170 and g < 90 and b < 90
    r, g, b = _pixel(out, 5.0, cx - int(w * 0.3), label_y)             # Wait's blue one
    assert b > 140 and r < 120
    assert max(_pixel(out, 10.0, 600, 300)) < 80                       # the footage is untouched away from the timer
    # a song, started again when it ends, at a quarter volume under the voiceover; the timer where it was asked for
    parts = [{"track_id": "song_full", "offset": 0.0, "play": 25.0, "pass": 1},
             {"track_id": "song_full", "offset": 25.0, "play": 15.0, "pass": 2}]
    full, half = tmp_path / "song.mp4", tmp_path / "song_voice.mp4"
    RP.render_plan_dict(plan(parts, False, False, "bottom-right"), str(full), quick=True)
    RP.render_plan_dict(plan(parts, False, True, "bottom-right"), str(half), quick=True)
    assert _level(full, 30, 38) > 0.1                                   # the song plays on after its first end
    assert _level(half, 25, 35) == pytest.approx(0.25 * _level(full, 25, 35), rel=0.15)       # a quarter under the voiceover…
    assert _level(full, 5, 15) == pytest.approx(0.25 * _level(full, 25, 35), rel=0.15)        # …and a quarter during Wait
    assert _level(half, 5, 15) == pytest.approx(_level(full, 5, 15), rel=0.15)                # (a quarter either way)
    assert _level(full, 20.6, 21.0) > 3 * _level(full, 19.0, 19.4)                            # back up as Extra starts
    assert _level(half, 0.05, 0.3) > _level(half, 5, 5.3) * 1.02        # the cue (a placeholder tone) at the phase start
    assert max(_pixel(full, 10.0, cx, top + 4)) < 80                    # nothing top left now…
    r, g, b = _pixel(full, 30.0, 640 - top - w + int(w * 0.2), 360 - top - int(w * 0.15) - 1)
    assert r > 170 and g < 90 and b < 90                                # …Extra's label sits bottom right


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
        at.button(key="edge_go").click().run()
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


def _seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_avatar_pops_up_and_the_intro_screen_comes_first(tmp_path, monkeypatch):
    """👩 Her clip (picture + voice) in a corner box or as the right half of a split screen, the stand-in figure
    where there is no clip yet, and the 🎬 intro screen (as long as her intro clip) before the compilation."""
    import edge_mode as E
    import render_preview as RP
    proxies = tmp_path / "proxies"
    proxies.mkdir()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x003000:s=320x180:d=30:r=25",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(proxies / "vid_a.mp4")], check=True)
    monkeypatch.setattr(RP, "PROXY_DIR", proxies)
    monkeypatch.setattr(RP, "streaming_available", lambda: False)

    def clip(name, colour, seconds, sound=True):                  # an "avatar" clip: a plain colour, portrait, a tone for her voice
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=180x320:d={seconds}:r=25"]
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}", "-af", "volume=6"] if sound else []
        subprocess.run(cmd + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-shortest",
                              str(tmp_path / name)], check=True)
        return str(tmp_path / name)
    talk, hello = clip("talk.mp4", "yellow", 4), clip("intro.mp4", "magenta", 5)
    sched = [{"index": 0, "round": 1, "profile": "wait", "start": 0.0, "end": 20.0},
             {"index": 1, "round": 1, "profile": "extra", "start": 20.0, "end": 40.0}]
    timeline = [{"track_time": [k * 5.0, k * 5.0 + 5.0],
                 "scenes": [{"chain": [{"video_id": "vid_a", "scene_id": 0, "clip_start_sec": 1.0 + k,
                                        "clip_duration_sec": 5.0}]}]} for k in range(8)]

    def plan(mode, intro):
        return {"track_id": "edge_test", "selected_videos": ["vid_a"], "timeline": timeline, "track_duration_sec": 40.0,
                "edge": {"difficulty": "chill", "rounds": 1, "schedule": sched, "playlist": [],
                         "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
                         "overlay": {"corner": "top-left"},
                         "avatar": {"on": True, "mode": mode, "corner": "bottom-right",
                                    "cues": [{"time": 0.0, "profile": "wait", "clip": talk},
                                             {"time": 20.0, "profile": "extra", "clip": None}]},
                         "intro": intro}}
    yellow = lambda p: p[0] > 170 and p[1] > 170 and p[2] < 90          # noqa: E731
    green = lambda p: max(p) < 90 and p[1] > p[0] + 20 and p[1] > p[2] + 20                                       # noqa: E731
    # in a corner: her box bottom right while her clip runs, the footage back after it; her voice is heard
    out = tmp_path / "corner.mp4"
    RP.render_plan_dict(plan("corner", {"on": False}), str(out), quick=True)
    assert abs(_seconds(out) - 40.0) < 0.3
    bh = int(360 * E.AVATAR_CORNER_SHARE)
    spot = (640 - int(360 * 0.035) - 30, 360 - int(360 * 0.035) - bh // 2)
    assert yellow(_pixel(out, 2.0, *spot)) and green(_pixel(out, 6.0, *spot))
    assert green(_pixel(out, 2.0, 320, 100))                             # the middle of the frame is still footage
    assert _level(out, 1, 3.5) > 0.05 > _level(out, 6, 18)               # her tone during the clip, quiet Wait after
    assert not green(_pixel(out, 21.0, *spot)) and green(_pixel(out, 25.0, *spot))      # the stand-in figure, 3 s
    # split screen: she is the right half, the footage keeps the left
    out = tmp_path / "split.mp4"
    RP.render_plan_dict(plan("split", {"on": False}), str(out), quick=True)
    assert yellow(_pixel(out, 2.0, 480, 300)) and yellow(_pixel(out, 2.0, 330, 40)) and green(_pixel(out, 2.0, 300, 300))
    assert green(_pixel(out, 6.0, 480, 300))
    assert _pixel(out, 1.0, int(360 * 0.035) + E.overlay_size(360)[0] // 2, int(360 * 0.035) + 4)[0] > 150      # the timer stays on top
    # the intro screen: as long as her intro clip, she on the left, then the compilation with its timer from 0
    intro = {"on": True, "clip": hello, "stats": [["Difficulty", "Chill"], ["Rounds", "1"], ["Time", "0:40"]], "thumbs": []}
    out = tmp_path / "intro.mp4"
    RP.render_plan_dict(plan("corner", intro), str(out), quick=True)
    assert abs(_seconds(out) - 45.0) < 0.3
    ax, ay, aw, ah = E.intro_layout((640, 360))["avatar"]
    r, g, b = _pixel(out, 2.0, ax + aw // 2, ay + ah // 2)
    assert r > 170 and b > 170 and g < 90                                # her (magenta) intro clip in her box
    assert max(_pixel(out, 2.0, 630, 350)) < 60 and _level(out, 1, 4) > 0.05           # the dark card, her voice
    assert yellow(_pixel(out, 7.0, *spot)) and _pixel(out, 6.0, int(360 * 0.035) + E.overlay_size(360)[0] // 2, int(360 * 0.035) + 4)[0] > 150
    # no intro clip yet: the stand-in for 8 s
    out = tmp_path / "intro_stand_in.mp4"
    RP.render_plan_dict(plan("corner", {**intro, "clip": None}), str(out), quick=True)
    assert abs(_seconds(out) - 48.0) < 0.3 and _level(out, 1, 7) < 0.002


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
        assert len(edge["avatar"]["cues"]) == 17 and "Start-Round 1" in edge["avatar"]["placeholders"]      # no clips here: stand-ins
        assert edge["intro"]["on"] and edge["intro"]["clip"].endswith("intro_1.mp4")
        assert edge["intro"]["stats"] == [["Difficulty", "Chill"], ["Rounds", "3"], ["Time", "6:00"]]
        assert 1 <= len(edge["intro"]["thumbs"]) <= 6 and {v for v, _ in edge["intro"]["thumbs"]} <= set(VIDS)
        off = PE.compile_edge([], VIDS, "chill", 3)["plan"]["edge"]
        assert not off["avatar"]["on"] and off["avatar"]["cues"] == [] and not off["intro"]["on"]
        # the page: the avatar's options show and compile
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
        at.button(key="edge_go").click().run()
        assert not errors(at), errors(at)
        e = at.session_state["edge_result"]["plan"]["edge"]
        assert e["avatar"]["on"] and e["avatar"]["mode"] == "split" and e["intro"]["on"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_cue_map_gives_every_line_its_moment_and_never_two_at_once():
    import edge_mode as E
    cm = E.load_cue_map()
    assert len(cm["lines"]) == 45 and sum(1 for ln in cm["lines"] if ln["cues"]) == 32
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
                assert got["intro"] == ["Welcome", "Intro to NNN"] and got["outro"] == ["I hope not"]
    got = E.resolve_lines(E.schedule("chill", 6), cm, "chill")
    at = {(p["line"], p["round"]): p for p in got["plays"]}
    assert at[("Start-Round 1", 1)]["time"] == 0.0 and at[("Now stop", 1)]["time"] == 100.0 and at[("Now stop", 2)]["time"] == 220.0
    assert ("Now stop", 3) not in at                                               # set for rounds 1 and 2 only
    assert at[("Now stop (2)", 5)]["time"] == 4 * 120 + 97.0                       # 3 s before Extra ends
    assert at[("Stroke faster", 3)]["time"] == 2 * 120 + 60.0                      # as Medium ends
    p = at[("I love edging you", 4)]
    assert 3 * 120 + 100 <= p["time"] and p["time"] + p["seconds"] <= 3 * 120 + 120       # somewhere during Stop
    last = at[("Did you cum", 6)]
    assert abs(last["time"] + last["seconds"] - 720.0) < 1e-6 and last["moved"] < 0       # finishes as the compilation ends
    # "You can survive" is round 5's line only (every round was a slip); "Keep going" is a filler, once a round
    assert [r for (n, r) in at if n == "You can survive"] == [5] and at[("You can survive", 5)]["time"] == 4 * 120 + 20.0
    assert sorted(r for (n, r) in at if n == "Keep going") == [1, 2, 3, 4, 5, 6]
    assert all(at[("Keep going", r)]["profile"] in ("medium", "fast") for r in range(1, 7))
    # two lines at one moment: the later one waits its turn — and that is reported
    assert at[("Slow", 3)]["moved"] > 0 and any(c["line"] == "Slow" and c["with"] == "Ok stroke again" for c in got["clashes"])
    # "last" means the final round whatever its number — and there its lines replace that round's own
    for rounds in (3, 4, 5):
        g = E.resolve_lines(E.schedule("chill", rounds), cm, "chill")
        names = {p["line"] for p in g["plays"] if p["round"] == rounds}
        assert {"Did you cum", "show me how much - now stop", "Just stroke", "Now Stroke"} <= names
        assert not names & {"Slow", "Stroke faster", "relax-fingertips", "How stroke even more", "Balls bursting", "Now stop (2)",
                            "And dont cum", "Just sit there", "I know you can do it", "You can survive"}
        assert not [c for c in g["clashes"] if c["round"] == rounds and c["result"].startswith("left out")]
    six = {p["line"] for p in got["plays"] if p["round"] == 5}
    assert {"Balls bursting", "Now stop (2)", "And dont cum"} <= six                # round 5 keeps its own when it isn't the last
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


def E_CARD_LEAD():
    import edge_render
    return edge_render.CARD_LEAD


def test_the_intro_card_takes_thumbnails_as_pictures_or_bytes():
    """The planner hands thumbnails over as JPEG bytes — the intro screen failed on them ("bytes object has no
    attribute convert", 14 Nov)."""
    import io
    import edge_mode as E
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (160, 90), (200, 30, 30)).save(buf, format="JPEG")
    img = E.intro_card((640, 360), [("Rounds", "3")], [buf.getvalue(), Image.new("RGB", (160, 90), (30, 30, 200)), None, b"not a picture"])
    px = np.asarray(img)
    assert (px[..., 0] > 150).sum() > 2000 and (px[..., 2] > 150).sum() > 2000          # both thumbnails are drawn


def test_the_avatar_follows_the_cue_map(monkeypatch, tmp_path):
    out = _timelines(monkeypatch)
    try:
        import planner.edge as PE
        monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "avatar")
        (tmp_path / "avatar" / "lines").mkdir(parents=True)
        for n in ("Now stop", "Welcome"):
            (tmp_path / "avatar" / "lines" / f"{n}.mp4").write_bytes(b"x")
        res = PE.compile_edge([], VIDS, "chill", 3, voiceover=True, avatar=True, intro=True)
        edge = res["plan"]["edge"]
        av, vo = edge["avatar"]["cues"], edge["voiceover"]["cues"]
        assert [(c["time"], c["line"]) for c in av] == [(c["time"], c["line"]) for c in vo]      # she appears with every line
        assert {c["line"] for c in av if c["clip"]} == {"Now stop"} and "Get hard for me" in edge["avatar"]["placeholders"]
        assert [x.get("clip") is not None for x in edge["intro"]["lines"]] == [True, False]      # Welcome has her clip
        assert edge["music"]["still_volume"] == 0.25
    finally:
        shutil.rmtree(out, ignore_errors=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_stuck_picture_pass_is_stopped_and_says_so(tmp_path):
    import os
    import time
    import edge_render as R
    fifo = tmp_path / "never.wav"                         # an input nobody ever writes to: ffmpeg waits on it for ever
    os.mkfifo(fifo)
    keep = os.open(fifo, os.O_RDWR)                       # (held open, so ffmpeg's own open returns and its read hangs)
    try:
        t0 = time.time()
        with pytest.raises(RuntimeError, match="stood still"):
            R.run_watched(["ffmpeg", "-y", "-f", "wav", "-i", str(fifo), str(tmp_path / "out.wav")], 10.0, stall=3)
        assert time.time() - t0 < 30
    finally:
        os.close(keep)
    seen = []                                              # a pass that works reports how far it is, up to the end
    R.run_watched(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=60", "-c:v", "libx264", "-preset",
                   "ultrafast", str(tmp_path / "ok.mp4")], 60.0, progress=seen.append)
    assert all(0 < f <= 1 for f in seen) and seen == sorted(seen)
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        R.run_watched(["ffmpeg", "-y", "-i", str(tmp_path / "not_there.mp4"), str(tmp_path / "x.mp4")], 5.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_songs_are_read_from_this_machine_never_streamed_and_sit_at_a_quarter_under_the_voice(tmp_path, monkeypatch):
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: f"raw/{t}"))
    song = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=30", "-af", "volume=4",
                    str(song)], check=True)

    class Sources:                                         # a song that isn't here yet: only a stream on offer
        here, downloaded = False, []

        def is_local(self, raw):
            return self.here

        def download(self, raws):
            self.downloaded += list(raws)
            self.here = True
            return []

        def input_args(self, raw):
            return ["-i", str(song)] if self.here else ["-i", "https://example.invalid/stream"]

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    sched = [{"index": 0, "round": 1, "profile": "slow", "start": 0.0, "end": 20.0}]
    edge = {"schedule": sched, "playlist": [{"track_id": "s", "offset": 0.0, "play": 20.0, "pass": 1}], "beat": {"on": False},
            "voiceover": {"on": False, "cues": []}}
    src = Sources()
    loud = R._mix(edge, [], src, tmp_path, run)
    assert src.downloaded == ["raw/s"]                     # fetched once, then read as a file
    loud = loud.rename(tmp_path / "loud.wav")
    edge["voiceover"] = {"on": True, "music_volume": 0.5, "cues": [{"time": 15.0, "profile": "slow", "file": None}]}     # an older plan's 0.5
    quiet = R._mix(edge, [], src, tmp_path, run)
    assert E.MUSIC_UNDER_VOICE == 0.25
    assert _level(quiet, 2, 12) == pytest.approx(0.25 * _level(loud, 2, 12), rel=0.1)


def test_with_the_avatar_her_clips_sound_is_heard_at_the_lines_loudness(tmp_path):
    import edge_render as R
    ff = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)      # noqa: E731
    mp3, clip = tmp_path / "line.wav", tmp_path / "her.mp4"
    ff("-f", "lavfi", "-i", "sine=frequency=700:duration=2", "-af", "volume=6", str(mp3))        # the line: loud
    ff("-f", "lavfi", "-i", "color=c=yellow:s=90x160:d=3:r=25", "-f", "lavfi", "-i", "sine=frequency=700:duration=3",
       "-af", "volume=0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(clip))    # her clip: much quieter
    edge = {"voiceover": {"on": True, "cues": [{"time": 5.0, "profile": "slow", "file": str(mp3)},
                                               {"time": 30.0, "profile": "slow", "file": None},
                                               {"time": 40.0, "profile": "slow", "file": str(mp3)}]}}
    shows = [{"time": 5.0, "seconds": 3.0, "profile": "slow", "clip": str(clip), "still": None, "sound": str(clip)},
             {"time": 30.0, "seconds": 3.0, "profile": "slow", "clip": str(clip), "still": None, "sound": str(clip)},
             {"time": 40.0, "seconds": 3.0, "profile": "slow", "clip": None, "still": "x.png", "sound": None}]
    cues = R._sound_cues(edge, shows)
    assert [c["file"] for c in cues] == [str(clip), str(clip), str(mp3)]      # her clip where she has one, the line otherwise
    assert cues[0]["match"] == str(mp3) and not cues[1].get("match")
    assert 18 < R._match_gain(str(clip), str(mp3)) <= 20 and R._match_gain(str(clip), None) == 0.0

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    track = R._voice_track(cues, 50.0, tmp_path, run)
    assert _level(track, 5.2, 6.8) == pytest.approx(_level(track, 40.2, 41.8), rel=0.25)      # as loud as the line itself
    assert _level(track, 30.2, 31.8) < 0.3 * _level(track, 5.2, 6.8)                          # nothing to match: as it is


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_picture_is_made_phase_by_phase_and_stays_in_step(tmp_path, monkeypatch):
    """Three phases with an avatar clip that runs across a phase boundary: every second of the footage is there once,
    in order, the timer restarts with each phase, and the picture and the sound are as long as the plan."""
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    ff = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)      # noqa: E731
    comb = tmp_path / "comb.mp4"                           # the footage's grey level is its own second
    ff("-f", "lavfi", "-i", "color=black:s=16x16:r=25:d=60", "-vf",
       "geq=lum='40+3*floor(N/25)':cb=128:cr=128,scale=640:360:flags=neighbor,setsar=1", "-c:v", "libx264", "-preset",
       "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(comb))
    clip = tmp_path / "her.mp4"
    ff("-f", "lavfi", "-i", "color=c=yellow:s=180x320:d=6:r=25", "-f", "lavfi", "-i", "sine=frequency=880:duration=6",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(clip))
    sched = [{"index": k, "round": 1, "profile": p, "start": 20.0 * k, "end": 20.0 * k + 20.0}
             for k, p in enumerate(("wait", "slow", "medium"))]
    edge = {"schedule": sched, "playlist": [], "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
            "overlay": {"corner": "top-left"},
            "avatar": {"on": True, "mode": "corner", "corner": "bottom-right",
                       "cues": [{"time": 17.0, "profile": "wait", "clip": str(clip)}]},      # 17 → 23 s: over the join
            "intro": {"on": False}, "outro": {"on": False}}

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    out = tmp_path / "out.mp4"
    seen = []
    took = R.finish({"edge": edge}, comb, str(out), types.SimpleNamespace(input_args=lambda raw: ["-i", raw]), tmp_path / "w",
                    run, 360, 640, progress=seen.append)
    assert took["edge_check"] == "ok" and seen[-1] == 1.0 and len(seen) == 3
    assert abs(_seconds(out) - 60.0) < 0.1
    grey = [_pixel(out, t + 0.5, 320, 60)[0] for t in range(60)]
    steps = [b - a for a, b in zip(grey, grey[1:])]
    assert all(2 <= d <= 5 for d in steps), steps          # one second after another, nothing skipped or repeated
    bh = int(360 * E.AVATAR_CORNER_SHARE)
    spot = (640 - int(360 * 0.035) - 30, 360 - int(360 * 0.035) - bh // 2)
    yellow = lambda p: p[0] > 170 and p[1] > 170 and p[2] < 90          # noqa: E731
    assert [yellow(_pixel(out, t, *spot)) for t in (16.5, 18.0, 19.9, 20.1, 22.5, 23.5)] == [False, True, True, True, True, False]
    w, _h = E.overlay_size(360)
    cx, top = int(360 * 0.035) + w // 2, int(360 * 0.035)
    assert _pixel(out, 20.5, cx, top + 4)[0] > 150 and _pixel(out, 40.5, cx, top + 4)[0] > 150      # a full ring as each phase starts
