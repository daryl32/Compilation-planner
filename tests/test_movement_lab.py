"""🧪 Movement lab (13 Nov 2026): four switches for the passages with little or no beat — 🎼 follow the lead layer,
🌊 match at the scale of a phrase, 🪶 prefer flowing movement, 🔍 judge the level within the shot size — and two for
the 📐 lines' blocks (🔗 join, ✂️ shortest clip). Each off unless asked for; off = the matcher exactly as it was."""
from __future__ import annotations

import shutil

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.test_sync import _add_full_rate

VIDS = ["vid_a", "vid_b", "vid_c", "vid_f"]


def _sig(timeline):
    return [[(l["video_id"], l["scene_id"], l["clip_start_sec"], l["clip_duration_sec"]) for s in e["scenes"] for l in s["chain"]]
            for e in timeline]


def _full_rate_everywhere(out, seed: int = 11) -> None:
    import catalogue_store as CS
    rng = np.random.default_rng(seed)
    vids = {v["video_id"]: v for v in CS.video_index().values()}
    for vid in VIDS:
        _add_full_rate(out, vid, 0.2 + 0.3 * rng.random(int(vids[vid]["duration"] * 15) + 15))


def test_flow_tells_a_slow_sway_from_restless_movement():
    from core import sync as S
    rate = 15.0
    t = np.arange(0, 60, 1 / rate)
    times = np.arange(0, 58, 0.5)
    rng = np.random.default_rng(0)
    mid = slice(10, -10)
    sway = S.flow(1 + np.sin(2 * np.pi * 0.25 * t), rate, times)[mid].mean()
    bounce = S.flow(1 + np.sin(2 * np.pi * 2.0 * t), rate, times)[mid].mean()
    noise = S.flow(rng.random(len(t)), rate, times)[mid].mean()
    assert sway > 0.9 and bounce < 0.1 and noise < 0.1
    assert 0.3 < S.flow(1 + np.sin(2 * np.pi * 1.0 * t), rate, times)[mid].mean() < 0.7       # in between
    # a little jitter on a slow sway is still a slow sway
    assert S.flow(1 + np.sin(2 * np.pi * 0.3 * t) + 0.1 * rng.standard_normal(len(t)), rate, times)[mid].mean() > 0.8
    # footage that stops moving is calm, not restless — whatever its jitter looks like
    x = 1 + np.sin(2 * np.pi * 2.0 * t)
    x[t > 40] = 0.2 + 1e-3 * rng.standard_normal(int((t > 40).sum()))
    f = S.flow(x, rate, times)
    assert f[10:60].mean() < 0.1 and f[90:110].mean() > 0.9
    gaps = np.full(len(t), np.nan)
    assert np.allclose(S.flow(gaps, rate, times), 0.5) and len(S.flow(np.zeros(5), rate, times)) == len(times)


def test_flowing_movement_is_preferred_off_the_beat_only(monkeypatch):
    """One video: restless, irregular movement in its first half, a slow sway just as lively in its second. With no
    beat 🪶 prefers the sway; on a full beat, and with the switch off, nothing changes."""
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import catalogue_store as CS
        import core.story_matcher as SMX
        import footage_timeline as FT
        import recipes as RC
        vids = {v["video_id"]: v for v in CS.video_index().values()}
        t = np.arange(int(float(vids["vid_a"]["duration"]) * 15) + 15) / 15.0
        rng = np.random.default_rng(5)
        half = len(t) // 2
        curve = 0.4 + 0.2 * rng.standard_normal(len(t))
        curve[half:] = 0.4 + 0.2 * np.sin(2 * np.pi * 0.2 * t[half:])
        _add_full_rate(out, "vid_a", np.clip(curve, 0.0, 1.0))
        index = FT.load_index()
        v = SMX._Video(FT.load("vid_a", index=index), index)
        fl = v.flow_series()
        assert fl[4:len(fl) // 2 - 8].mean() < 0.2 and fl[len(fl) // 2 + 8:-4].mean() > 0.8
        n = 8
        args = (v, n, RC.default("rhythmic"), np.full(120, 0.5), 120.0, 0.0, 4.0, 0.5, True)
        plain = SMX._score_video_x(*args, beat=0.0)[0]
        on = SMX._score_video_x(*args, beat=0.0, flow=True)[0]
        mid = len(on) // 2
        a, b = slice(4, mid - 12), slice(mid + 12, len(on) - 4)
        assert (on[b].mean() - on[a].mean()) - (plain[b].mean() - plain[a].mean()) > 0.1
        assert np.allclose(SMX._score_video_x(*args, beat=1.0, flow=True)[0], SMX._score_video_x(*args, beat=1.0)[0])
        assert np.allclose(SMX._score_video_x(*args, flow=True)[0], SMX._score_video_x(*args)[0])       # beat unknown
        r = SMX._score_video_x(*args[:8], False, beat=0.0, flow=True)[0]                                # recipes on
        assert np.allclose(r, SMX._score_video_x(*args[:8], False, beat=0.0)[0])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_phrase_scale_finds_a_swell_under_quick_jitter(monkeypatch):
    """The footage follows the music's slow swell, with quick jitter on top. Push by push the jitter hides it; at
    the scale of a phrase the window is found, with a higher sync."""
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
        rng = np.random.default_rng(7)
        seconds = 8.0
        music = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(int(seconds * 30)) / 30.0 / 5.3) * np.hanning(int(seconds * 30))
        m = int(seconds * 15)
        curve = 0.4 + 0.35 * rng.standard_normal(int(dur * 15) + 15)
        p = int(round(dur * 0.5 * 15))
        curve[p:p + m] = 0.4 + S.pool(music, m) - 0.5 + 0.35 * rng.standard_normal(m)
        _add_full_rate(out, "vid_a", curve)
        index = FT.load_index()
        v = SMX._Video(FT.load("vid_a", index=index), index)
        assert v.mover_slow is not None and v.mover_slow.smooth_sec == SMX.PHRASE_SMOOTH_SEC and v.mover.smooth_sec == S.SMOOTH_SEC
        n = int(seconds / 0.5)
        args = (v, n, RC.default("rhythmic"), music, 120.0, 0.0, seconds, 0.5, True)
        plain = SMX._score_video_x(*args, beat=0.0)
        slow = SMX._score_video_x(*args, beat=0.0, phrase=True)
        k = int(round(p / 15 / 0.5))
        assert abs(int(np.argmax(slow[3])) - k) <= 1 and slow[3][k] > plain[3][k] + 0.1 and slow[3][k] > 0.8
        frame = 1.0 / 15.0                                    # slides are whole frames, which don't sit on half seconds
        assert float(np.abs(slow[4]).max()) <= SMX.PHRASE_SLIDE_SEC + 1.5 * frame
        assert float(np.abs(plain[4]).max()) <= SMX.SLIDE_SEC + 1.5 * frame < float(np.abs(slow[4]).max())
        assert np.allclose(SMX._score_video_x(*args[:8], False, beat=0.0, phrase=True)[0],
                           SMX._score_video_x(*args[:8], False, beat=0.0)[0])                            # recipes on: no part
        # as played, measured at the same scale
        clip = np.asarray(v.tl.move_hi, float)[p:p + m]
        assert S.sync_of(clip, music, 15.0, smooth_sec=SMX.PHRASE_SMOOTH_SEC) > S.sync_of(clip, music, 15.0) + 0.1
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _beatless_track():
    """The fixture song with its drums and voice taken out of 30–55 % of its length and an instrument (the "other"
    stem) playing a clear phrase there."""
    from tests.fixtures import load_track
    track = load_track(DATA, "song_full")
    a = track["analysis"]
    n, rate, dur = len(a["kick"]), float(a["rate"]), float(track["duration_sec"])
    lo, hi = int(0.30 * n), int(0.55 * n)
    for k in ("kick", "snare"):
        x = np.asarray(a[k], float)
        x[lo:hi] = 0.0
        a[k] = x.tolist()
    voice = np.asarray(a["vocals"], float)
    voice[lo:hi] = 0.0
    a["vocals"] = voice.tolist()
    t = np.arange(n) / rate
    other = np.full(n, 0.05)
    other[lo:hi] = 0.5 + 0.4 * np.sin(2 * np.pi * 0.4 * t[lo:hi])
    a["stem_curves"] = {"rate": rate, "other": other.tolist(), "share": {"other": 0.6}}
    return track, 0.30 * dur, 0.55 * dur


def test_the_lead_layer_is_followed_off_the_beat(monkeypatch):
    from tests.test_story_matcher import _plan_inputs, _setup
    out = _setup(monkeypatch)
    try:
        import audio_drive as AD
        import core.story_matcher as SMX
        track, t0, t1 = _beatless_track()
        drv = AD.drivers(track, None)
        leads = AD.lead_curves(track, drv)
        assert {"vocals", "other"} <= set(leads) and len(leads["other"]) == len(drv["curve"])
        name, part = AD.lead_slice(leads, drv["rate"], t0 + 2, t1 - 2)
        assert name == "other" and float(part.std()) > 0.1            # no voice there: the instruments lead
        assert AD.lead_slice(leads, drv["rate"], 1.0, 6.0) is None     # nothing but drums at the start
        quiet = dict(track, analysis={**track["analysis"], "stem_curves": {**track["analysis"]["stem_curves"], "share": {"other": 0.01}}})
        assert "other" not in AD.lead_curves(quiet, AD.drivers(quiet, None))          # a stem that isn't in the song
        _full_rate_everywhere(out)
        types = ["rhythmic"] * 16
        _, queues, segments, regions = _plan_inputs(types)
        opts = SMX.StoryOptions(section_map=regions, movement_only=True, stem_follow=True)
        tl, short, _ = SMX.match_story(segments, queues, track, options=opts)
        inside = [e for e in tl if e["track_time"][0] >= t0 - 0.01 and e["track_time"][1] <= t1 + 0.01]
        outside = [e for e in tl if e["track_time"][1] <= t0 + 0.01]
        assert short == 0 and inside and outside
        assert all(e["scenes"][0]["story"].get("follows") == "other" for e in inside)
        assert not any(e["scenes"][0]["story"].get("follows") for e in outside)
        assert opts.lab_notes["used"]["follows"] == {"other": len([e for e in tl if e["scenes"][0]["story"].get("follows")])}
        assert opts.lab_notes["used"]["off_beat"] >= len(inside)
        # the 📥 file measures those clips against the layer they followed, and gives the usual figure beside it
        rep = SMX.sync_report(tl, track, settings={"lab": {"stem": True}})
        rows = [r for r in rep["clips"] if r.get("follows")]
        assert rows and all("other" in r["music_curves"] and "flow" in r for r in rows)
        assert all("sync_played_plain" in r for r in rows if r.get("sync_played") is not None)       # (a sliver has none)
        assert rep["settings"] == {"lab": {"stem": True}} and rep["sync_played_off_beat"] is not None
        assert rep["sync_played_on_beat"] is not None and rep["seconds_off_beat"] > 0 and rep["sync_played_plain_mean"] is not None
        # off: nothing follows anything, and the plan is the plain movement-only plan
        _, queues, segments, regions = _plan_inputs(types)
        off, _, _ = SMX.match_story(segments, queues, track, options=SMX.StoryOptions(section_map=regions, movement_only=True))
        assert not any(s["story"].get("follows") or s["story"].get("phrase") for e in off for s in e["scenes"])
        # 🌊 on a track with such a passage: the long enough blocks in it are matched at phrase scale
        _, queues, _, _ = _plan_inputs(types)
        dur = float(track["duration_sec"])
        long_segs = [{"start": round(i * dur / 8, 2), "end": round((i + 1) * dur / 8, 2), "energy": 0.5, "intensity": 0.5} for i in range(8)]
        opts2 = SMX.StoryOptions(section_map=regions, movement_only=True, phrase_scale=True)
        t2, _, _ = SMX.match_story(long_segs, queues, track, options=opts2)
        ph = [e for e in t2 if e["scenes"] and e["scenes"][0]["story"].get("phrase")]
        assert ph and opts2.lab_notes["used"]["phrase"] == len(ph)
        whole = [e for e in t2 if e["track_time"][0] >= t0 - 0.01 and e["track_time"][1] <= t1 + 0.01]
        assert whole and all(e in ph for e in whole)                       # the beatless passage's own blocks
        loud = [e for e in t2 if e["track_time"][0] >= 0.74 * dur and e["track_time"][1] <= 0.93 * dur]
        assert loud and not any(e in ph for e in loud)                     # the drop keeps the beat's own scale
        assert all(r.get("phrase") for r in SMX.sync_report(ph, track)["clips"])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_level_is_judged_within_the_shot_size(monkeypatch):
    """Two videos: wide shots that read calm and close-ups that read lively. 🔍 pulls each part-way to the footage
    as a whole — wide up, close-up down, never past the limits — and off changes nothing."""
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import core.story_matcher as SMX
        import footage_timeline as FT
        import recipes as RC

        def reshape(vid, shot, level):
            with np.load(out / f"{vid}.npz") as z:
                arrays = {k: z[k] for k in z.files}
            arrays["shot"] = np.full(len(arrays["times"]), shot, np.float16)
            arrays["move"] = np.full(len(arrays["times"]), level, np.float16)
            FT._write_video(out, vid, arrays)
        reshape("vid_a", 3.0, 0.2)
        reshape("vid_b", 1.0, 0.6)
        index = FT.load_index()
        monkeypatch.setattr(SMX, "SHOT_FIX_MIN_POINTS", 60)       # the fixture videos are a minute and a half long
        videos = {vid: SMX._Video(FT.load(vid, index=index), index) for vid in ("vid_a", "vid_b")}
        assert SMX._shot_gains(dict(list(videos.items())[:1])) == {}          # one shot size: nothing to compare
        before = {vid: v.level_series()[0].copy() for vid, v in videos.items()}
        gains = SMX._shot_gains(videos)
        assert set(gains) == {1, 3} and gains[3] > 1.0 > gains[1]
        assert all(SMX.SHOT_FIX_RANGE[0] <= g <= SMX.SHOT_FIX_RANGE[1] for g in gains.values())
        wide, close = videos["vid_a"].level_series(True)[0], videos["vid_b"].level_series(True)[0]
        assert wide.mean() > before["vid_a"].mean() and close.mean() < before["vid_b"].mean()
        assert close.mean() - wide.mean() < before["vid_b"].mean() - before["vid_a"].mean()      # nearer, not swapped
        assert close.mean() > wide.mean()
        assert np.allclose(videos["vid_a"].level_series()[0], before["vid_a"])                   # off: as measured
        # a loud passage could not be met by the wide video before; corrected, it comes closer
        args = (videos["vid_a"], 8, RC.default("rhythmic"), np.full(120, 0.5), 120.0, 0.0, 4.0, 0.45, True)
        assert SMX._score_video_x(*args, shot_fix=True)[0].mean() > SMX._score_video_x(*args)[0].mean()
        # one shot size only (the usual fixtures): nothing to correct, nothing changed
        same = {vid: SMX._Video(FT.load(vid, index=index), index) for vid in ("vid_b", "vid_c")}
        assert SMX._shot_gains(same) == {} and all(v.shot_gain is None for v in same.values())
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_lab_through_a_compile_and_all_off_is_the_plan_as_it_was(monkeypatch):
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import planner.compose as PC
        import section_bounds as SB
        import section_cuts as SCUT
        from tests.fixtures import load_track
        _full_rate_everywhere(out)
        base = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines")
        off = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", lab={k: False for k in PC.LAB})
        assert _sig(base["plan"]["timeline"]) == _sig(off["plan"]["timeline"])
        assert [e["scenes"][0]["story"] for e in base["plan"]["timeline"]] == [e["scenes"][0]["story"] for e in off["plan"]["timeline"]]
        assert base["plan"]["story"]["lab"] == {k: False for k in PC.LAB}
        assert "lab all off" in PC.used_line(base["used"]) and "📐 lines" in PC.used_line(base["used"])
        assert f"{len(VIDS)} videos in play" in PC.used_line(base["used"]) and "0 clips rejected" in PC.used_line(base["used"])
        # every switch on: still a whole plan, the same cuts, nothing used twice — and it says so
        lab = {"stem": True, "phrase": True, "flow": True, "shot": True}
        res = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", lab=lab, excluded={("vid_a", "x")})
        tl = res["plan"]["timeline"]
        assert [tuple(e["track_time"]) for e in tl] == [tuple(e["track_time"]) for e in base["plan"]["timeline"]]
        assert res["shortfall"] == 0 and res["plan"]["story"]["lab"]["flow"] and not res["plan"]["story"]["lab"]["join"]
        line = PC.used_line(res["used"])
        assert all(PC.LAB[k] in line for k in lab) and "1 clip rejected" in line and "used" in res["lab_notes"]
        used = {}
        for e in tl:
            for s in e["scenes"]:
                for l in s["chain"]:
                    for b0, b1 in used.get(l["video_id"], []):
                        assert min(b1, l["clip_start_sec"] + l["clip_duration_sec"]) - max(b0, l["clip_start_sec"]) <= 0.05
                    used.setdefault(l["video_id"], []).append((l["clip_start_sec"], l["clip_start_sec"] + l["clip_duration_sec"]))
        # 🔁 a redone block is matched with the same switches
        res["video_ids"] = VIDS
        again = PC.rework_plan(res, {1})
        assert again["plan"]["story"]["lab"] == res["plan"]["story"]["lab"] and len(again["plan"]["timeline"]) == len(tl)
        # with recipes on the four movement switches do nothing
        r0 = PC.compile_plan("song_full", VIDS, cuts="lines")
        r1 = PC.compile_plan("song_full", VIDS, cuts="lines", lab=lab)
        assert _sig(r0["plan"]["timeline"]) == _sig(r1["plan"]["timeline"])
        # 🔗 join: two blocks of ONE part may become one longer clip — never across a line, never over the limit
        tr = load_track(DATA, "song_full")
        segs = SCUT.line_segments(tr, PC._section_regions(tr))
        joined = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", lab={"join": True})
        got = [tuple(e["track_time"]) for e in joined["plan"]["timeline"]]
        assert joined["plan"]["segmentation"]["join"] is True and len(got) <= len(segs)
        part_edges = {round(a["end"], 2) for a, b in zip(segs, segs[1:]) if a["line_section"] != b["line_section"]}
        assert part_edges <= {t for pair in got for t in pair}
        longest_alone = max(s["end"] - s["start"] for s in segs)
        assert all(b - a <= max(SB.BLOCK_MAX_SEC, longest_alone) + 0.02 for a, b in got)
        assert got[0][0] == 0.0 and all(abs(a[1] - b[0]) < 0.011 for a, b in zip(got, got[1:]))
        n_merged = sum(1 for e in joined["plan"]["timeline"] if e.get("merged"))
        assert len(got) == len(segs) - n_merged
        # ✂️ shortest clip on the lines: a longer minimum → no block under it inside a part, fewer blocks
        longer = SCUT.line_segments(tr, PC._section_regions(tr), True, 6.0)
        assert len(longer) < len(segs) and longer[0]["start"] == 0.0 and longer[-1]["end"] == tr["duration_sec"]
        count = {}
        for s in longer:
            count[s["line_section"]] = count.get(s["line_section"], 0) + 1
        assert all(s["end"] - s["start"] >= SB.MIN_SLACK * 6.0 - 0.01 for s in longer if count[s["line_section"]] > 1)
        assert SCUT.line_segments(tr, PC._section_regions(tr), True, None) == segs               # Auto: as it was
        m = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", min_clip=6.0, lab={"min_lines": True})
        assert len(m["segments"]) == len(longer) and m["plan"]["segmentation"]["min_clip"] == 6.0
        assert "shortest clip 6 s" in PC.used_line(m["used"])
        ignored = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", min_clip=6.0)
        assert len(ignored["segments"]) == len(segs)                                              # switch off: not applied
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_lab_menu_on_the_page(monkeypatch):
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import planner.compose as PC
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "🤖 New matcher"
        at.run()
        assert not errors(at), errors(at)
        keys = ["compose_lab_" + k for k in PC.LAB]
        tog = {t.key: t for t in at.toggle if t.key in keys}
        assert set(tog) == set(keys) and all(t.disabled and not t.value for t in tog.values())    # all off, all greyed
        at.session_state["compose_movement_only"] = True
        at.session_state["compose_cuts"] = "lines"
        at.run()
        assert not errors(at), errors(at)
        assert not any(t.disabled for t in at.toggle if t.key in keys)
        assert at.number_input(key="compose_min_clip").disabled
        at.toggle(key="compose_lab_flow").set_value(True)
        at.toggle(key="compose_lab_join").set_value(True)
        at.toggle(key="compose_lab_min_lines").set_value(True).run()
        assert not errors(at), errors(at)
        assert not at.number_input(key="compose_min_clip").disabled
        assert any("3 switches on" in e.label for e in at.expander)
        at.button(key="cs_go").click().run()
        assert not errors(at), errors(at)
        res = at.session_state[PC.RESULT_KEY]
        assert res["plan"]["story"]["lab"] == {"stem": False, "phrase": False, "flow": True, "shot": False,
                                               "join": True, "min_lines": True}
        assert any(c.value.startswith("🧾 This compile:") and PC.LAB["flow"] in c.value for c in at.caption)
        at.toggle(key="compose_sync_data").set_value(True).run()
        assert not errors(at), errors(at)
    finally:
        shutil.rmtree(out, ignore_errors=True)
