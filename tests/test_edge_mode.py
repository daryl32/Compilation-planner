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
        # the voiceover follows the cue map: named lines, never two at once; intro and outro lines bring their screens
        vo = edge["voiceover"]
        assert len(edge["schedule"]) == 18 and vo["cue_map"] and vo["cues"][0]["line"] == "Start-Round 1"
        for x, y in zip(vo["cues"], vo["cues"][1:]):
            assert y["time"] >= x["time"] + x["seconds"]
        assert all(c["file"] is None for c in vo["cues"]) and "Now stop" in vo["lines_missing"]      # no files here: tones
        assert edge["intro"]["on"] and [x["line"] for x in edge["intro"]["lines"]] == ["Welcome", "Intro to NNN"]
        assert edge["outro"]["on"] and [x["line"] for x in edge["outro"]["lines"]] == ["I hope not"]
        assert res["voice_clashes"] and {"line", "with", "round", "where", "result"} <= set(res["voice_clashes"][0])
        quiet = PE.compile_edge([], VIDS, "chill", 3)["plan"]["edge"]
        assert not quiet["intro"]["on"] and not quiet["outro"]["on"] and quiet["voiceover"]["cues"] == []
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
        assert res3["plan"]["edge"]["voiceover"]["music_volume"] == 0.5
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
                         "voiceover": {"on": voice, "music_volume": 0.5, "cues": E.voice_cues(sched, {}) if voice else []},
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
    # a song, started again when it ends, at half volume under the voiceover; the timer where it was asked for
    parts = [{"track_id": "song_full", "offset": 0.0, "play": 25.0, "pass": 1},
             {"track_id": "song_full", "offset": 25.0, "play": 15.0, "pass": 2}]
    full, half = tmp_path / "song.mp4", tmp_path / "song_voice.mp4"
    RP.render_plan_dict(plan(parts, False, False, "bottom-right"), str(full), quick=True)
    RP.render_plan_dict(plan(parts, False, True, "bottom-right"), str(half), quick=True)
    assert _level(full, 30, 38) > 0.1                                   # the song plays on after its first end
    assert _level(half, 5, 15) == pytest.approx(0.5 * _level(full, 5, 15), rel=0.15)
    assert _level(half, 0.0, 0.3) > _level(half, 5, 5.3) * 1.02         # the cue (a placeholder tone) at the phase start
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
        assert edge["avatar"]["on"] and edge["avatar"]["mode"] == "split" and len(edge["avatar"]["cues"]) == 18
        fast = [c for c in edge["avatar"]["cues"] if c["profile"] == "fast"]
        assert all(c["clip"].endswith("fast_1.mp4") for c in fast) and edge["avatar"]["placeholders"] == ["wait", "slow", "medium", "extra", "stop"]
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
    # the every-round line gives way to a round's own line at the same moment — and that is reported
    assert ("You can survive", 1) not in at and at[("You can survive", 5)]["time"] == 4 * 120 + 20.0
    assert any(c["line"] == "You can survive" and c["with"] == "Get hard for me" and c["result"] == "left out" for c in got["clashes"])
    assert at[("You can survive", 2)]["moved"] > 0
    # "last" means the final round whatever its number; a line set for a difficulty stays out of the others
    three = {(p["line"], p["round"]) for p in E.resolve_lines(E.schedule("chill", 3), cm, "chill")["plays"]}
    assert ("Did you cum", 3) in three and ("show me how much - now stop", 3) in three
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


def E_CARD_LEAD():
    import edge_render
    return edge_render.CARD_LEAD
