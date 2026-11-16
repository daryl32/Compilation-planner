"""⏱️ Edge Mode: rounds and phases, the generated beat, the playlist, the music's fades."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.edge_support import (DATA, VIDS, E_CARD_LEAD, _check_plan, _emap, _level, _pixel, _seconds,  # noqa: F401
                                _timelines, _wait_job)


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
    size = E.write_overlay(sched[:2], tmp_path / "ov", 360)
    assert size[0] % 2 == 0 and size[1] % 2 == 0 and len(list((tmp_path / "ov").glob("*.png"))) == 2 * 20 * E.OVERLAY_FPS
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


def test_a_new_song_each_round_and_the_beat_schedule_a_round_at_a_time():
    import edge_mode as E
    sched = E.schedule("chill", 4)                         # 4 × 120 s
    parts = E.round_playlist_parts([("a", 200.0), ("b", 50.0), ("c", 90.0)], sched)
    assert [(p["track_id"], p["offset"], p["play"], p["round"]) for p in parts] == [
        ("a", 0.0, 120.0, 1), ("b", 120.0, 50.0, 2), ("b", 170.0, 50.0, 2), ("b", 220.0, 20.0, 2),
        ("c", 240.0, 90.0, 3), ("c", 330.0, 30.0, 3), ("a", 360.0, 120.0, 4)]
    assert parts[-1]["pass"] == 2 and parts[2]["loop"] == 2          # the list comes round again; b starts again
    assert abs(sum(p["play"] for p in parts) - 480.0) < 1e-6
    sub = E.shift_schedule([ph for ph in sched if ph["round"] == 3])
    assert sub[0]["start"] == 0.0 and sub[-1]["end"] == 120.0 and [p["index"] for p in sub] == list(range(6))


def test_the_music_fades_in_stop_comes_back_over_wait_and_dips_only_under_lines():
    import edge_mode as E
    sched = E.schedule("chill", 2)                         # wait 0–20 … stop 100–120, wait 120–140 …
    spec = E.music_spec(sched, talking=True, lines=[(50.0, 53.0)])
    g = lambda t: E.music_gain(spec, t)                    # noqa: E731
    assert g(0.0) == 0.0 and g(10.0) == pytest.approx(0.5) and g(19.99) == pytest.approx(1.0, abs=1e-3)   # in over Wait
    assert g(30.0) == 1.0                                  # full between lines (with the voiceover on)
    assert g(51.0) == pytest.approx(E.DUCK_LEVEL) and g(49.0) == 1.0 and g(53.5) == 1.0
    assert g(100.0) == 1.0 and g(101.0) == pytest.approx(0.5)       # noticeably lower within Stop's first second…
    assert g(101.0) > g(105.0) > g(110.0) > g(115.0) > g(119.0) > 0.0 and g(119.99) < 0.01     # …then out smoothly
    assert g(110.0) == pytest.approx(0.5 * (10 / 19) ** 1.7, rel=1e-3)
    assert g(130.0) == pytest.approx(0.5)                   # the next round's Wait brings it back
    old = E.music_spec(sched, talking=True, fade_stop=False, duck=False)      # switched off: as before
    assert E.music_gain(old, 30.0) == 0.25 and E.music_gain(old, 10.0) == 0.25
    quiet = E.music_spec(sched, talking=False, fade_stop=False)
    assert E.music_gain(quiet, 30.0) == 1.0 and E.music_gain(quiet, 10.0) == pytest.approx(0.25)
    assert "gte(t" in E.music_gain_expr(spec) and "min(1," in E.music_gain_expr(spec)
