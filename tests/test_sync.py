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
        assert {"beat", "bounce", "move_level", "level"} <= set(row) and isinstance(rep["movement_by_shot_size"], dict)
        assert len(row["pixel_move"]) == len(row["move"])     # the catalogue's on-screen motion beside the person curve
        assert "seen_level" in row and "seen_bounce" in row
        # with recipes on, the frame-rate sync is the "shape" part of the music fit: every pick carries it and is slid
        track, queues, segments, regions = _plan_inputs(types)
        t2, _, _ = SMX.match_story(segments, queues, track, options=SMX.StoryOptions(section_map=regions, music_story=0.0))
        infos = [s["story"] for e in t2 for s in e["scenes"] if s["story"].get("fit") is not None]
        assert infos and all("sync" in i_ and "slide" in i_ for i_ in infos) and any(i_["slide"] for i_ in infos)
        assert any(i_["story"] != 0.5 for i_ in infos)            # …and the recipes' tags count again
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_bounce_tells_a_regular_up_and_down_from_smooth_or_random_movement():
    from core import sync as S
    rate = 15.0
    t = np.arange(0, 60, 1 / rate)
    times = np.arange(0, 60, 0.5)
    rng = np.random.default_rng(0)
    jitter = 0.005 * rng.standard_normal(len(t))
    bouncy, bpm = S.bounce(0.1 + 0.05 * np.maximum(0, np.sin(2 * np.pi * 1.5 * t)) + jitter, rate, times)
    assert bouncy.mean() > 0.9 and abs(np.nanmedian(bpm) - 90) < 2             # 1.5 a second, not half of it
    _, b120 = S.bounce(0.1 + 0.05 * np.maximum(0, np.sin(2 * np.pi * 2.0 * t)) + jitter, rate, times)
    assert abs(np.nanmedian(b120) - 120) < 3                                   # 7.5 frames a bounce: between two frames
    assert S.bounce(0.1 + 0.05 * np.sin(2 * np.pi * 0.08 * t) + jitter, rate, times)[0].max() < 0.1    # a slow swell
    assert S.bounce(0.1 + 0.03 * rng.standard_normal(len(t)), rate, times)[0].mean() < 0.15            # random
    half = 0.1 + 0.05 * np.maximum(0, np.sin(2 * np.pi * 1.5 * t)) + jitter
    half[450:] = 0.01 + 0.0005 * rng.standard_normal(len(t) - 450)             # …then hardly moving at all
    s, _ = S.bounce(half, rate, times)
    assert s[:55].mean() > 0.9 and s[66:].mean() < 0.1
    gone = np.full(len(t), np.nan)
    s, b = S.bounce(gone, rate, times)
    assert not s.any() and np.isnan(b).all() and not S.bounce(np.ones(20), rate, times)[0].any()


def test_beat_presence_is_judged_on_the_hits():
    from core import sync as S
    hits = np.zeros(120)
    hits[::15] = 0.9
    hits[1::15] = 0.6
    assert S.beat_presence(hits, 0.4) == 1.0                  # a low average, strong hits: a beat
    assert S.beat_presence(np.full(120, 0.1), 0.4) == 0.0     # a steady murmur: none
    assert S.beat_presence(np.full(120, 0.3), 0.4) == 0.0     # 01 AURA's beatless passages: weak hits are no beat
    assert 0.0 < S.beat_presence(np.full(120, 0.43), 0.4) < 1.0
    assert S.beat_presence(None, 0.4) is None and S.beat_presence([], 0.4) is None


def test_movement_only_wants_bouncy_footage_on_a_beat_and_smooth_footage_off_it(monkeypatch):
    """One video: its first half bounces on the track's tempo, its second half moves just as much but smoothly.
    The same stretch of music scored as "a beat" prefers the first half, as "no beat" the second — and with the
    beat unknown (no drum curve) the score is what it was before."""
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import catalogue_store as CS
        import core.story_matcher as SMX
        import footage_timeline as FT
        import recipes as RC
        vids = {v["video_id"]: v for v in CS.video_index().values()}
        dur = float(vids["vid_a"]["duration"])
        t = np.arange(int(dur * 15) + 15) / 15.0
        rng = np.random.default_rng(5)
        curve = 0.3 + 0.3 * np.maximum(0, np.sin(2 * np.pi * 2.0 * t)) + 0.01 * rng.standard_normal(len(t))      # 120 a minute
        half = len(t) // 2
        curve[half:] = 0.4 + 0.1 * np.sin(2 * np.pi * 0.05 * t[half:]) + 0.01 * rng.standard_normal(len(t) - half)
        _add_full_rate(out, "vid_a", curve)
        index = FT.load_index()
        v = SMX._Video(FT.load("vid_a", index=index), index)
        n = 8
        music = np.full(120, 0.5)                             # nothing to follow: only the feel can tell windows apart
        args = (v, n, RC.default("rhythmic"), music, 120.0, 0.0, 4.0, 0.5, True)
        on = SMX._score_video_x(*args, beat=1.0)[0]
        off = SMX._score_video_x(*args, beat=0.0)[0]
        plain = SMX._score_video_x(*args)[0]
        mid = len(on) // 2
        a, b = slice(4, mid - 12), slice(mid + 12, len(on) - 4)
        # the half-second fixture data differs between the halves too, so it is the SWING between the two that counts
        assert (on[a].mean() - on[b].mean()) - (off[a].mean() - off[b].mean()) > 0.15
        assert on[a].mean() > on[b].mean()
        bw = v.bounce_window(n, 120.0)
        assert bw[a].mean() > 0.8 and bw[b].mean() < 0.1 and v.bounce_window(n, 100.0)[a].mean() < bw[a].mean()
        assert np.allclose(plain, SMX._score_video_x(*args, beat=None)[0])
        # recipes on: the feel plays no part
        r_on = SMX._score_video_x(*args[:8], False, beat=1.0)[0]
        assert np.allclose(r_on, SMX._score_video_x(*args[:8], False)[0])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_a_frozen_picture_is_found_and_kept_out_of_the_plan(monkeypatch):
    from core import sync as S
    rate = 15.0
    rng = np.random.default_rng(2)
    x = 0.03 + 0.02 * rng.random(900)                         # a quiet scene still jitters
    x[300:360] = 0.0                                          # 4 s frozen from 20 s
    x[600:606] = 0.0                                          # a few identical frames: not a freeze
    x[700:760] = np.nan                                       # nobody tracked: not a freeze either
    times = np.arange(0, 60, 0.5)
    fz = S.frozen(x, rate, times, 0.5)
    assert fz[40:48].min() == 1.0 and fz[:40].max() == 0.0 and fz[48:].max() == 0.0
    assert not S.frozen(np.full(900, np.nan), rate, times, 0.5).any()
    # in the matcher: every window touching the freeze is pushed to the bottom, with or without recipes
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import catalogue_store as CS
        import core.story_matcher as SMX
        import footage_timeline as FT
        import recipes as RC
        vids = {v["video_id"]: v for v in CS.video_index().values()}
        n_f = int(float(vids["vid_a"]["duration"]) * 15) + 15
        curve = 0.2 + 0.1 * rng.random(n_f)
        curve[300:360] = 0.0
        _add_full_rate(out, "vid_a", curve)
        index = FT.load_index()
        v = SMX._Video(FT.load("vid_a", index=index), index)
        clean = SMX._Video(FT.load("vid_b", index=index), index)
        assert not clean.frozen.any()                         # a timeline without the full-rate curve: no opinion
        for only in (True, False):
            fit = SMX._score_video_x(v, 8, RC.default("rhythmic"), np.full(120, 0.5), 120.0, 0.0, 4.0, 0.5, only)[0]
            assert fit[34:47].max() < 0.5 * np.median(fit[60:])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_on_screen_motion_is_put_on_the_person_curves_clock():
    from core import sync as S
    # two scenes at 30 and 25 frames a second; the second starts at 4 s; nothing between 2 and 4 s
    a, b = np.full(60, 2.0), np.full(50, 6.0)
    px = S.pixel_curve([(0.0, 30.0, a), (4.0, 25.0, b)], 120, 15.0)
    assert len(px) == 120 and np.allclose(px[:30], 2.0 / 6.0) and np.isnan(px[30:60]).all() and np.allclose(px[60:90], 1.0)
    assert np.isnan(px[90:]).all() and S.pixel_curve([], 120, 15.0) is None
    lv = S.half_second_level(px, 15.0, np.arange(0, 8, 0.5), 0.5)
    assert abs(lv[0] - 1 / 3) < 1e-9 and np.isnan(lv[5]) and abs(lv[8] - 1.0) < 1e-9


def test_movement_only_goes_by_what_is_seen_where_the_person_curve_misses_it(monkeypatch):
    """01 AURA, tests 3–4: whole bodies bouncing read as calm on the person curve (it measures limbs against the
    body's own centre). Here the person curve is the same quiet jitter throughout; the picture bounces on the
    track's tempo in the first half and drifts in the second. A beat wants the first half, no beat the second —
    and the music's own shape is found in the picture, not in the person curve."""
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import catalogue_store as CS
        import core.story_matcher as SMX
        import footage_timeline as FT
        import recipes as RC
        from core import sync as S
        vids = {v["video_id"]: v for v in CS.video_index().values()}
        dur = float(vids["vid_a"]["duration"])
        n_f = int(dur * 15) + 15
        rng = np.random.default_rng(7)
        _add_full_rate(out, "vid_a", 0.05 + 0.01 * rng.random(n_f))
        index = FT.load_index()
        v = SMX._Video(FT.load("vid_a", index=index), index)
        t = np.arange(n_f * 2) / 30.0                          # the picture, 30 frames a second
        pic = 4.0 + 6.0 * np.maximum(0, np.sin(2 * np.pi * 2.0 * t)) + 0.1 * rng.standard_normal(len(t))
        half = len(t) // 2
        pic[half:] = 6.0 + 1.5 * np.sin(2 * np.pi * 0.05 * t[half:]) + 0.1 * rng.standard_normal(len(t) - half)
        music = (np.mod(np.arange(120) / 30.0, 1.0) < 0.2) * 1.0 + 0.3 * rng.random(120)      # 4 s, its own shape
        at = 70.2                                              # …which the picture does from 70.2 s
        p = int(round(at * 30))
        pic[p:p + 120] = 6.0 + 6.0 * music
        v.set_seen([(0.0, 30.0, pic)])
        assert v.seen is not None and v.seen_mover.n == v.mover.n
        args = (v, 8, RC.default("rhythmic"), music, 120.0, 0.0, 4.0, 0.5, True)
        on, _, _, sync, slide = SMX._score_video_x(*args, beat=1.0)
        off = SMX._score_video_x(*args, beat=0.0)[0]
        mid = len(on) // 2
        a, b = slice(4, mid - 12), slice(mid + 12, 130)
        # (the level asked for moves with the beat too, so the swing is smaller than the bounce's share alone)
        assert (on[a].mean() - on[b].mean()) - (off[a].mean() - off[b].mean()) > 0.08 and on[a].mean() > on[b].mean()
        assert v.bounce_window(8, 120.0)[a].mean() > 0.5 and v.bounce_window(8, 120.0)[b].mean() < 0.1
        s_ = int(sync.argmax())
        assert abs(v.times[s_] + slide[s_] - at) < 0.1 and sync[s_] > 0.75           # half the picture's 1.0, half the person curve's nothing
        # recipes on: the on-screen motion plays no part
        r1 = SMX._score_video_x(*args[:8], False)[0]
        v2 = SMX._Video(FT.load("vid_a", index=index), index)
        assert np.allclose(r1, SMX._score_video_x(v2, *args[1:8], False)[0])
    finally:
        shutil.rmtree(out, ignore_errors=True)
