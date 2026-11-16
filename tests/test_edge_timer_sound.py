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
    # 💀 Hard (16 Nov): Wait 20 s, a random middle, Extra → Stop, 5:00 a round
    early, late, drops, lengths = {p: 0 for p in E.ORDER}, {p: 0 for p in E.ORDER}, 0, {p: [] for p in E.ORDER}
    for seed in range(200):
        sched = E.schedule("hard", 3, seed)
        assert abs(E.total_seconds(sched) - 900.0) < 1e-6
        for r in (1, 2, 3):
            rp = [p for p in sched if p["round"] == r]
            ph = [p["profile"] for p in rp]
            assert ph[0] == "wait" and ph[-2:] == ["extra", "stop"]
            assert rp[0]["end"] - rp[0]["start"] == 20 and rp[-1]["end"] - rp[-1]["start"] == 20
            assert 20 <= rp[-2]["end"] - rp[-2]["start"] <= 30
            mid = rp[1:-2]
            assert {p["profile"] for p in mid} <= {"slow", "medium", "fast", "extra"} and "extra" in {p["profile"] for p in mid}
            assert all(a != b for a, b in zip(ph, ph[1:]))                                     # never the same twice running
            assert not any(a in E.FAST_ONES and b in E.FAST_ONES and c in E.FAST_ONES
                           for a, b, c in zip(ph[1:-2], ph[2:-2], ph[3:-2]))
            half = mid[0]["start"] + (mid[-1]["end"] - mid[0]["start"]) / 2
            for k, p in enumerate(mid):
                (early if p["start"] < half else late)[p["profile"]] += 1
                lengths[p["profile"]].append(p["end"] - p["start"])
                drops += k > 0 and mid[k - 1]["profile"] in E.FAST_ONES and p["profile"] == "slow"
            assert all(p["kind"] == "phase" and p["clock"]["total"] == p["end"] - p["start"] for p in rp)
    assert early["slow"] + early["medium"] > early["fast"] + early["extra"]                  # calmer first half
    assert late["fast"] + late["extra"] > late["slow"] + late["medium"]                      # faster second half
    mean = {p: sum(v) / len(v) for p, v in lengths.items() if v}
    assert mean["slow"] < mean["medium"] < mean["fast"] < mean["extra"]                      # the faster, the longer
    assert drops > 100                                                                         # drop-backs happen
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


def test_surprises_follow_their_rules():
    import edge_mode as E
    import edge_energy as EE
    assert E.schedule("medium", 3, 1) == E.schedule("medium", 3, 1, surprises=False)
    assert not any(ph.get("surprise") for ph in E.schedule("hard", 3, 1, surprises=True))      # Hard: none
    kinds, lengths = {}, {"chill": set(), "medium": set()}
    for diff in ("chill", "medium"):
        for seed in range(150):
            sched = E.schedule(diff, 4, seed, surprises=True)
            sl = [ph for ph in sched if ph.get("surprise")]
            assert sl                                                            # at least one a compilation
            assert all(abs(a["end"] - b["start"]) < 1e-9 for a, b in zip(sched, sched[1:]))
            plain = E.schedule(diff, 4, seed)
            assert abs(E.total_seconds(sched) - E.total_seconds(plain) - sum(p["end"] - p["start"] for p in sl)) < 1e-6
            assert sum(1 for p in sl if p["surprise"] == "falsefinish") <= 1
            per_round = {}
            for ph in sl:
                k = sched.index(ph)
                assert ph["round"] >= 2                                          # none in round 1
                per_round[ph["round"]] = per_round.get(ph["round"], 0) + 1
                kinds[ph["surprise"]] = kinds.get(ph["surprise"], 0) + 1
                lengths[diff].add((ph["surprise"], ph["end"] - ph["start"]))
                if ph["kind"] == "handsoff":                                     # ✋ the phase splits, its clock carries on
                    before, after = sched[k - 1], sched[k + 1]
                    assert before["unit"] == ph["unit"] == after["unit"] and before["profile"] == after["profile"] == ph["profile"]
                    assert ph["profile"] in E.HANDS_OFF_IN[diff] and ph["bpm"] == 0
                    into = before["end"] - before["start"]
                    assert E.HANDS_OFF_EARLIEST <= into <= before["clock"]["total"] - E.HANDS_OFF_LATEST
                    assert after["clock"]["left"] == before["clock"]["total"] - into
                    assert E.timer_style(ph) == "handsoff" and E.next_real(sched, k) == E.next_real(sched, k - 1)
                else:                                                            # ⏩ / 🎭 after its phase, same unit
                    nxt = sched[k + 1]["profile"] if k + 1 < len(sched) else None
                    assert sched[k - 1]["unit"] == ph["unit"] and nxt not in E.NO_OVERRIDE_BEFORE
                    assert E.next_real(sched, k) == nxt and E.next_real(sched, k - 1) == nxt      # Next up never warns
                    if ph["surprise"] == "falsefinish":
                        assert ph["profile"] == "extra" and nxt == "stop" and ph["label"] == "NOT YET"
                        assert ph["round"] >= max(p["round"] for p in sched) - 1
            assert max(per_round.values()) <= (1 if diff == "chill" else 2)
            units = [ph.get("unit") for ph in sched]
            for g in EE.groups(sched):                                           # the music never joins inside a phase
                assert all(units.count(units[i]) == sum(1 for j in g if units[j] == units[i]) for i in g)
    assert set(kinds) == {"handsoff", "override", "falsefinish"}
    assert {n for k, n in lengths["chill"] if k == "handsoff"} == {3, 5} and {n for k, n in lengths["chill"] if k == "override"} == {5}
    assert {n for k, n in lengths["medium"] if k == "handsoff"} == {5, 8, 12}
    assert {n for k, n in lengths["medium"] if k == "override"} == {5, 10, 15}


def test_hands_off_dips_the_music_and_the_timer_shows_each_surprise(tmp_path):
    import edge_mode as E
    sched = next(s for s in (E.schedule("medium", 3, k, surprises=True) for k in range(50))
                 if any(p["kind"] == "handsoff" for p in s) and any(p["kind"] == "override" for p in s))
    ho = next(p for p in sched if p["kind"] == "handsoff")
    spec = E.music_spec(sched, talking=False)
    mid = (ho["start"] + ho["end"]) / 2
    assert E.music_gain(spec, mid) < 0.3 and E.music_gain(spec, ho["start"] - 2) > 0.9
    assert all(p["kind"] != "handsoff" or not E.phase_beats(p) for p in sched)        # no beat while hands off
    E.write_overlay(sched, tmp_path / "ov", 360)
    keys = {E.timer_key(p) for p in sched}
    assert any(k.startswith("handsoff-") for k in keys) and any(k.startswith("override-") for k in keys)
    for k in keys:
        assert len(list((tmp_path / "ov").glob(f"{k}_*.png"))) == round(float(k.rsplit("-", 1)[1]) * E.OVERLAY_FPS)
    assert E.overlay_image("handsoff", 3, E.overlay_size(360), True, total=5).getpixel((5, 5)) is not None
    assert (261.6 if True else 0) == E.style("handsoff")["chime"] and dict(E.chime_times(sched))[ho["start"]] == "handsoff"
