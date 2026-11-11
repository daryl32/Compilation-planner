"""🎯 Movement ↔ music sync at the footage's frame rate (core/sync.py) and its use in the story matcher: every start
frame is compared with the music, the clip slides onto the hits, and 🎯 Movement only picks on that alone."""
from __future__ import annotations

import shutil

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)


def test_pool_keeps_the_area_and_the_spikes():
    from core import sync as S
    c = np.zeros(300)
    c[::10] = 1.0                                             # a hit every 10 points
    p = S.pool(c, 150)
    assert len(p) == 150 and abs(p.sum() * 2 - c.sum()) < 1e-6 and (p[::5] > 0).all()   # no hit stepped over
    assert np.allclose(S.pool([3.0], 4), 3.0) and len(S.pool([], 5)) == 5


def test_the_fft_correlation_is_the_plain_one():
    from core import sync as S
    rng = np.random.default_rng(1)
    x = rng.random(900)
    mv = S.Mover(x, 15.0)
    tpl = rng.random(45)
    got = mv.correlate(tpl)
    assert len(got) == 900 - 45 + 1
    a = tpl - tpl.mean()
    for p in (0, 137, 855):
        w = mv.x[p:p + 45] - mv.x[p:p + 45].mean()
        want = float((w * a).mean() / np.sqrt((w * w).mean() * (a * a).mean()))
        assert abs(got[p] - want) < 1e-9
    assert not mv.correlate(np.ones(45)).any()                # a flat stretch of music says nothing


def test_a_pattern_off_the_half_second_grid_is_found_and_the_slide_reported():
    from core import sync as S
    rng = np.random.default_rng(0)
    rate = 15.0
    x = rng.random(int(600 * rate)) * 0.3
    t_aud = np.arange(int(4 * 30)) / 30.0
    music = (np.mod(t_aud, 0.5) < 0.07).astype(float)         # 4 s of hits every half second (120 BPM), 30 points a second
    p = int(round(200.2 * rate))                              # the footage does the same from 200.2 s
    k = np.arange(int(4 * rate))
    x[p:p + len(k)] += (np.mod(k / rate, 0.5) < 0.14) * 1.0
    times = np.arange(0, 600, 0.5)
    sync, slide = S.sync_scores(S.Mover(x, rate), times, 8, 4.0, music)
    s = int(sync.argmax())
    assert times[s] == 200.0 and 0.1 < slide[s] <= 0.27 and sync[s] > 0.9
    assert abs((times[s] + slide[s]) * rate - round((times[s] + slide[s]) * rate)) < 1e-6     # it lands on a frame
    assert np.median(sync) < 0.7                              # unrelated footage sits near the middle
    # nobody tracked → no opinion; a stretch too short → fall back to the half seconds
    gone = x.copy()
    gone[:3000] = np.nan
    sync2, _ = S.sync_scores(S.Mover(gone, rate), times, 8, 4.0, music)
    assert abs(sync2[10] - 0.5) < 1e-9
    assert S.sync_scores(S.Mover(x, rate), times, 1, 0.3, music[:9]) is None
    assert S.sync_scores(None, times, 8, 4.0, music) is None


def test_a_short_clip_is_trusted_less():
    from core import sync as S
    rate = 15.0
    beat = (np.mod(np.arange(600) / rate, 0.5) < 0.14) * 1.0  # footage that pulses throughout
    short = S.sync_of(beat[:15], beat[:15], rate)
    long_ = S.sync_of(beat[:60], beat[:60], rate)
    assert 0.5 < short < long_ and long_ > 0.99
    assert S.sync_of(np.full(60, np.nan), beat[:60], rate) is None


def _add_full_rate(out, vid: str, curve) -> None:
    import footage_timeline as FT
    with np.load(out / f"{vid}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["move_hi"] = np.asarray(curve, np.float32).astype(np.float16)
    arrays["move_rate"] = np.float32(15.0)
    FT._write_video(out, vid, arrays)


def test_movement_only_picks_the_footage_that_moves_with_the_music_and_slides_it(monkeypatch):
    """One video's movement IS the music's drive curve for the first block, starting off the half-second grid; every
    other frame of every video is noise. Movement only: that block takes that footage, slid onto the hits, with a
    near-perfect sync — no recipe, tag or role involved."""
    from tests.test_story_matcher import _plan_inputs, _setup
    out = _setup(monkeypatch)
    try:
        import audio_drive as AD
        import catalogue_store as CS
        import core.story_matcher as SMX
        types = ["rhythmic"] * 16
        track, queues, segments, regions = _plan_inputs(types)
        drv = AD.drivers(track, None)
        seg = segments[0]
        d = seg["end"] - seg["start"]
        music = AD.drive_slice(drv, seg["start"], seg["end"])
        assert float(np.std(music)) > 1e-3                    # the fixture song has a shape to follow
        rng = np.random.default_rng(3)
        vids = {v["video_id"]: v for v in CS.video_index().values()}
        scene = next(sp for sp in queues["vid_b"] if sp["remaining_sec"] >= d + 1.5)
        at = round(scene["scene_start_sec"] + scene["offset_sec"] + 0.7, 3)            # off the grid
        for vid in ("vid_a", "vid_b", "vid_c", "vid_f"):
            curve = rng.random(int(vids[vid]["duration"] * 15) + 15) * 0.2
            if vid == "vid_b":
                m = int(round(d * 15))
                p = int(round(at * 15))
                from core import sync as S
                curve[p:p + m] = 0.2 + S.pool(music, m)
            _add_full_rate(out, vid, curve)
        opts = SMX.StoryOptions(section_map=regions, movement_only=True)
        timeline, short, _ = SMX.match_story(segments, queues, track, options=opts)
        assert short == 0 and not any("frame-by-frame" in n_ for n_ in opts.notes)
        slot = timeline[0]["scenes"][0]
        assert slot["video_id"] == "vid_b" and abs(slot["clip_start_sec"] - round(at * 15) / 15) < 0.002, slot
        assert slot["story"]["sync"] > 0.9 and slot["story"]["slide"] != 0
        assert all(e["scenes"][0]["story"].get("story") == 0.5 for e in timeline if e["scenes"][0]["story"].get("fit") is not None)
        summary = SMX.plan_summary(timeline)
        assert summary["sync"] is not None and 0.4 < summary["sync"] < 1.0
        # the 📥 file: the clip's movement next to its music, and the same sync measured on what is played
        rep = SMX.sync_report(timeline, track)
        row = next(r for r in rep["clips"] if r["video"] == "vid_b" and abs(r["music"][0] - seg["start"]) < 0.01)
        assert rep["rate"] == 15.0 and len(row["move"]) == len(row["music_curves"]["drive"]) and row["sync_played"] > 0.9
        assert len(row["move_before"]) == 15 and "loudness" in row["music_curves"]
        # with recipes on, the frame-rate sync is the "shape" part of the music fit: every pick carries it and is slid
        track, queues, segments, regions = _plan_inputs(types)
        t2, _, _ = SMX.match_story(segments, queues, track, options=SMX.StoryOptions(section_map=regions, music_story=0.0))
        infos = [s["story"] for e in t2 for s in e["scenes"] if s["story"].get("fit") is not None]
        assert infos and all("sync" in i_ and "slide" in i_ for i_ in infos) and any(i_["slide"] for i_ in infos)
        assert any(i_["story"] != 0.5 for i_ in infos)            # …and the recipes' tags count again
    finally:
        shutil.rmtree(out, ignore_errors=True)
