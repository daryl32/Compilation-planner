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
        bpm = E.phase_bpm(ph)
        assert bpm == E.pace(ph["profile"], "chill", ph["round"])
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
    assert {s["bpm"] for s in segs if s["edge_profile"] == "fast"} == {110.0}           # 😌 Chill's Fast
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


def test_each_difficulty_has_its_own_pace_building_round_by_round():
    import edge_mode as E
    assert [E.pace(p, "chill", 1) for p in ("slow", "medium", "fast", "extra")] == [50, 80, 110, 140]
    assert [E.pace(p, "hard", 1) for p in ("slow", "medium", "fast", "extra")] == [70, 110, 160, 210]
    assert E.pace("fast", "chill", 3) == 110 and E.pace("wait", "hard", 2) == 0 and E.pace("stop", "medium", 1) == 0
    assert E.pace("fast", "medium", 3) == round(140 * 1.06, 1)                       # +3 % a round
    assert E.pace("extra", "hard", 6) == round(210 * 1.25, 1)                        # +5 % a round, at most 1.25 ×
    sched = E.schedule("medium", 3)
    assert all(ph["bpm"] == E.pace(ph["profile"], "medium", ph["round"]) for ph in sched)
    assert E.phase_bpm({"profile": "fast", "start": 0, "end": 20}) == 140            # an older plan: the profile's
    r3 = [ph for ph in sched if ph["round"] == 3 and ph["profile"] == "fast"][0]
    beats = E.phase_beats(r3)
    assert abs((beats[1] - beats[0]) - 60 / r3["bpm"]) < 1e-3


def test_the_pace_guide_follows_the_beat_or_the_song_near_each_phase_pace():
    import edge_mode as E
    sched = E.schedule("medium", 3)
    g = E.pace_guide(sched)                                          # 🥁 no songs: the generated beat, exactly
    assert g["beats"] == E.beat_times(sched) and all(p["source"] == "beat" for p in g["per_phase"])
    assert {p["index"] for p in g["per_phase"]} == {ph["index"] for ph in sched if ph["bpm"] > 0}
    # 🎵 a 120 bpm song from the start: Slow (60) gets every 2nd beat, Fast (140) every beat, Extra (180) every beat
    total = E.total_seconds(sched)
    parts = [{"track_id": "s", "offset": 0.0, "play": total}]
    song = {"s": [k * 0.5 for k in range(int(total * 2) + 4)]}
    g = E.pace_guide(sched, parts, song)
    by = {p["index"]: p for p in g["per_phase"]}
    slow = next(ph for ph in sched if ph["profile"] == "slow")
    fast = next(ph for ph in sched if ph["profile"] == "fast")
    assert by[slow["index"]]["bpm_used"] == 60 and by[fast["index"]]["bpm_used"] == 120
    inside = [t for t in g["beats"] if slow["start"] <= t < slow["end"]]
    assert all(abs((b - a) - 1.0) < 1e-6 for a, b in zip(inside, inside[1:]))
    assert not any(ph["start"] <= t < ph["end"] for t in g["beats"] for ph in sched if ph["profile"] in ("wait", "stop"))
    # a song started part-way (the ⚡ energy mix's "bar"): its beats move to where it plays
    g2 = E.pace_guide(sched[:2], [{"track_id": "s", "offset": 20.0, "play": 20.0, "bar": 30.25}], song)
    assert abs(g2["beats"][0] - 20.25) < 1e-6
    # no beats for the song: the phase's own pace
    g3 = E.pace_guide(sched[:2], parts, {})
    assert g3["per_phase"][0]["source"].startswith("song (no beats") and g3["beats"] == E.phase_beats(sched[1])
    # off target by more than 15 % is reported
    assert E.guide_off_target([{"index": 1, "bpm_target": 140, "bpm_used": 110}]) and \
        not E.guide_off_target([{"index": 1, "bpm_target": 140, "bpm_used": 130}])


def test_the_stroke_bar_moves_top_to_bottom_on_the_beats_and_the_labels_draw():
    import edge_mode as E
    beats = [10.0, 11.0, 12.0, 13.0]
    assert E.guide_position(beats, 10.0) == 0.0 and abs(E.guide_position(beats, 11.0) - 1.0) < 1e-9
    assert abs(E.guide_position(beats, 10.5) - 0.5) < 1e-9 and abs(E.guide_position(beats, 12.0)) < 1e-9
    assert E.guide_position(beats, 9.0) is None and E.guide_position(beats, 13.5) is None
    track, w = E.guide_bar(200)
    assert track.size == (w, 200) and E.guide_marker(w, (255, 0, 0)).size[0] > w
    chip = E.next_chip("fast", 180)
    assert chip.size[0] == 180 and chip.getpixel((90, chip.size[1] // 2))[3] > 0
    assert E.next_chip(None, 180).size == chip.size
