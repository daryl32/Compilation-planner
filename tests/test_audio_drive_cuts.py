"""🥁 What drives the music (audio_drive), accent-driven section cuts, and merging two short blocks when the
footage holds (10 Nov 2026)."""
from __future__ import annotations

import copy
import shutil

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.test_story_matcher import _plan_inputs, _setup as _timelines


def _voice_track():
    """60 s at 120 BPM: 0–20 s drums, 20–40 s only a voice (two phrases), 40–60 s bass and no drums."""
    rate, dur = 30, 60.0
    n = int(rate * dur)
    t = np.arange(n) / rate
    beats = np.arange(0, dur + 1e-6, 0.5)
    pulse = (np.abs((t / 0.5) % 1.0) < 0.15).astype(float)
    kick = np.where(t < 20, pulse, 0.0)
    voice = np.where((t >= 21) & (t < 29) | (t >= 31) & (t < 39), 0.8, 0.0)
    low = np.where(t >= 40, 0.7, 0.05)
    rms = 0.3 + 0.5 * (t < 20) + 0.2 * voice + 0.3 * (t >= 40)
    return {"track_id": "voice", "duration_sec": dur, "bpm": 120.0, "beat_times": beats.round(3).tolist(),
            "hires": {"rate": rate, "rms": rms.round(3).tolist(), "onset": pulse.round(3).tolist()},
            "analysis": {"rate": rate, "downbeats": beats[::4].round(3).tolist(), "kick": kick.round(3).tolist(),
                         "snare": np.zeros(n).tolist(), "low": low.round(3).tolist(), "vocals": voice.round(3).tolist(),
                         "novelty": np.zeros(n).tolist(), "vocal_starts": [21.0, 31.0], "vocal_ends": [29.0, 39.0],
                         "phrases": [{"time": 0.0, "level": 16}, {"time": 20.0, "level": 16}, {"time": 40.0, "level": 16}]}}


def test_the_driver_follows_what_is_playing():
    import audio_drive as AD
    tr = _voice_track()
    drv = AD.drivers(tr)
    assert AD.driver_at(drv, 10.0) == "drums" and AD.driver_at(drv, 25.0) == "vocals" and AD.driver_at(drv, 50.0) == "bass"
    assert set(AD.driver_shares(drv)) == {"drums", "vocals", "bass"}
    # the drive curve is the driver's own curve: pulsing with the kick in the drum part, the voice in the vocal part
    assert AD.drive_slice(drv, 5.0, 10.0).std() > 0.2 and AD.drive_slice(drv, 22.0, 28.0).mean() > 0.8
    acc = AD.accents(tr, tr["beat_times"], drv)
    b = np.array(tr["beat_times"])
    at = lambda x: float(acc[int(np.argmin(abs(b - x)))])      # noqa: E731
    assert at(21.0) > at(25.0) + 0.5                           # a vocal phrase start beats the middle of the phrase
    assert at(8.0) > at(8.5)                                   # a bar line beats an off beat
    # a track with no extra analysis still works, on loudness
    bare = {"duration_sec": 20.0, "beat_times": np.arange(0, 20, 0.5).tolist(),
            "energy_envelope": [{"time": k * 0.5, "energy": 0.5} for k in range(40)]}
    assert set(AD.drivers(bare)["driver"]) == {"loudness"} and len(AD.accents(bare, bare["beat_times"])) == 40


def test_section_cuts_divide_the_section_musically():
    import recipes as RC
    import section_cuts as SC
    tr = _voice_track()
    regions = [{"start": 0.0, "end": 20.0, "type": "action"}, {"start": 20.0, "end": 40.0, "type": "slow"},
               {"start": 40.0, "end": 60.0, "type": "rhythmic"}]
    recipes = RC.load_all()
    segs = SC.story_segments(tr, regions, recipes)
    assert segs[0]["start"] == 0.0 and segs[-1]["end"] == 60.0
    assert all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(segs, segs[1:]))
    for s in segs:
        pc = recipes[s["section_type"]]["pace"]
        d = s["end"] - s["start"]
        assert pc["lo"] * (1 - SC.RANGE_SLACK) - 1e-6 <= d <= pc["hi"] * (1 + SC.RANGE_SLACK) + 1e-6      # no short leftovers
        nb = round(d / 0.5)
        assert abs(d / 0.5 - nb) < 1e-6 and (nb % 4 == 0 or nb in (1, 2))                                # whole bars, or half a bar
    by = {}
    for s in segs:
        by.setdefault(s["section_type"], []).append(s)
    assert {s["driver"] for s in by["action"]} == {"drums"} and {s["driver"] for s in by["slow"]} == {"vocals"}
    # in the vocal part no cut lands inside a phrase (21–29, 31–39): only at its edges or between phrases
    for s in by["slow"][1:]:
        assert not (21.5 < s["start"] < 28.5 or 31.5 < s["start"] < 38.5), s["start"]
    # a steady section aims above the middle of its range
    assert np.mean([s["end"] - s["start"] for s in by["action"]]) >= sum(recipes["action"]["pace"].values()) / 2 - 0.5
    # a section shorter than a clip is one clip
    assert SC.cut_section(np.arange(0, 60, 0.5), 10.0, 11.0, recipes["slow"]) == [10.0]


def test_two_short_blocks_become_one_when_the_footage_holds(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import core.story_matcher as SMX
        track, queues, segments, regions = _plan_inputs(["drop"] * 32)          # 3 s blocks; Drop's range is 1–2 s → up to 3 s merged
        segments = [{**s, "section_type": "drop"} for s in segments]
        plain, _, _ = SMX.match_story(segments, copy.deepcopy(queues), track, options=SMX.StoryOptions(section_map=regions))
        assert len(plain) == len(segments) and [e["segment_index"] for e in plain] == list(range(len(segments)))
        assert not any(e["merged"] for e in plain)
        recipe = {"drop": {"pace": {"lo": 2.0, "hi": 4.0}}}                      # lets 3 s + 3 s merge (≤ 1.5 × 4 s)
        opts = SMX.StoryOptions(section_map=regions, merge_blocks=True, track_recipes=recipe)
        tl, short, unfilled = SMX.match_story(segments, copy.deepcopy(queues), track, options=opts)
        ps = SMX.plan_summary(tl)
        assert ps["merged"] >= 1 and len(tl) == len(segments) - ps["merged"]
        assert [e["segment_index"] for e in tl] == list(range(len(tl)))
        assert tl[0]["track_time"][0] == 0.0 and abs(tl[-1]["track_time"][1] - segments[-1]["end"]) < 0.01
        for a, b in zip(tl, tl[1:]):
            assert abs(a["track_time"][1] - b["track_time"][0]) < 0.011          # the whole track is still covered, in order
        for e in tl:
            d = e["track_time"][1] - e["track_time"][0]
            assert d <= 6.01 and all(abs(s["clip_duration_sec"] - d) < 0.15 for s in e["scenes"])
            assert e["driver"] in ("drums", "bass", "vocals", "loudness")
        assert ps["clip_sec"][2] > 3.5 and ps["drivers"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_bar_lines_are_filled_in_and_a_one_bar_driver_flicker_is_smoothed():
    import audio_drive as AD
    tr = _voice_track()
    beats = np.array(tr["beat_times"])
    full = np.array(tr["analysis"]["downbeats"])
    tr["analysis"]["downbeats"] = [float(d) for d in full if not (4.0 <= d < 14.0)]        # the analysis left a hole
    bars = AD.bar_lines(tr, beats)
    assert np.allclose(bars, full)                                                        # filled from the beats, same count
    # one vocal-looking bar inside the drum part (a gap in the kick) doesn't change the driver
    kick = np.array(tr["analysis"]["kick"])
    voc = np.array(tr["analysis"]["vocals"])
    kick[int(8.0 * 30):int(10.0 * 30)] = 0.0
    voc[int(8.0 * 30):int(10.0 * 30)] = 0.8
    tr["analysis"]["kick"], tr["analysis"]["vocals"] = kick.tolist(), voc.tolist()
    drv = AD.drivers(tr, beats)
    assert AD.driver_at(drv, 9.0) == "drums" and AD.driver_at(drv, 25.0) == "vocals"
    assert len(drv["levels"]) == len(drv["driver"]) and set(drv["levels"][0]) == {"drums", "bass", "vocals", "loudness"}


def test_no_clip_is_shorter_than_the_shortest_clip():
    """Daryl, 11 Nov (01 AURA, 90 BPM): the drop's bar was cut into two 1.3 s clips — good footage, too short."""
    import section_cuts as SC
    assert abs(SC.auto_min_clip(60 / 90) - 4 * 60 / 90) < 1e-9          # a bar at 90 BPM
    assert abs(SC.auto_min_clip(60 / 128) - 4 * 60 / 128) < 1e-9        # a bar at 128
    assert abs(SC.auto_min_clip(60 / 70) - 2 * 60 / 70) < 1e-9          # two beats at 70 are long enough
    beat = 60 / 90
    beats = np.arange(0, 120, beat)
    quick = {"pace": {"lo": 1.0, "hi": 1.5}, "movement": {"trend": "flat"}}
    a, b = beats[36], beats[36 + 32]                                    # eight bars
    free = SC.cut_section(beats, a, b, quick)
    lens = np.diff(free + [b])
    assert lens.min() < 1.5                                             # the recipe alone: two-beat clips
    held = SC.cut_section(beats, a, b, quick, None, SC.auto_min_clip(beat))
    lens = np.diff(held + [b])
    assert lens.min() > 4 * beat - 0.01 and len(held) < len(free)
    assert all(abs((c - a) / beat - round((c - a) / beat)) < 1e-6 for c in held)        # still on the beat
    # one bar (the drop): one clip, not two halves; a typed minimum works the same way
    assert SC.cut_section(beats, a, a + 4 * beat, quick, None, SC.auto_min_clip(beat)) == [a]
    assert np.diff(SC.cut_section(beats, a, b, quick, None, 2.0) + [b]).min() >= 2.0 - 0.25 * beat
