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
        assert len(edge["schedule"]) == 18 and len(edge["voiceover"]["cues"]) == 18
        assert edge["voiceover"]["placeholders"] == ["wait", "slow", "medium", "fast", "extra", "stop"]
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
