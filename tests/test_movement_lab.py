"""🧪 Movement lab (13 Nov 2026): 🎼 follow the lead layer and 🌊 match at the scale of a phrase (passages with little
or no beat), 🔍 judge the level within the shot size, ⚡ contrast in the quiet passages, and two for the 📐 lines' blocks
(🔗 join, ✂️ shortest clip). After Daryl's tests the same day: 🎼 + 🌊 are part of 🎯 Movement only (no switch), 🔍 + 🔗
are one switch, 🌿 Natural, ⚡ is baked into the Smooth ↔ Contrast slider, and 🪶 flowing movement was scrapped. With the four set off outright, the plan is the
matcher exactly as it was."""
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
        assert rows and all("other" in r["music_curves"] and "flow" not in r for r in rows)
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
        old = {"stem": False, "phrase": False, "shot": False, "join": False}          # the matcher as it was
        base = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", lab=old)
        assert base["plan"]["story"]["lab"] == {**old, "min_lines": False} and "flow" not in PC.LAB
        line0 = PC.used_line(base["used"])
        assert "📐 lines" in line0 and PC.LAB["stem"] + " off" in line0 and PC.LAB["phrase"] + " off" in line0
        assert f"{len(VIDS)} videos in play" in line0 and "0 clips rejected" in line0
        # 🎯 Movement only now follows the lead and matches at phrase scale by itself — nothing to switch on
        now = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", excluded={("vid_a", "x")})
        lab_now = now["plan"]["story"]["lab"]
        assert lab_now["stem"] and lab_now["phrase"] and not lab_now["shot"] and not lab_now["join"]
        assert not now["plan"]["story"]["natural"]
        assert now["lab_notes"]["used"]["off_beat"] >= 1 and (now["lab_notes"]["used"]["follows"] or now["lab_notes"]["used"]["phrase"])
        line = PC.used_line(now["used"])
        assert "1 clip rejected" in line and " off" not in line and PC.NATURAL not in line and "🧪" not in line
        assert [tuple(e["track_time"]) for e in now["plan"]["timeline"]] == [tuple(e["track_time"]) for e in base["plan"]["timeline"]]
        # …and recipes mode has none of it
        r0 = PC.compile_plan("song_full", VIDS, cuts="lines")
        assert not any(r0["plan"]["story"]["lab"][k] for k in ("stem", "phrase", "shot", "join"))
        assert _sig(r0["plan"]["timeline"]) == _sig(PC.compile_plan("song_full", VIDS, cuts="lines", lab=old)["plan"]["timeline"])
        # 🌿 Natural = 🔍 shot size + 🔗 joins, one switch
        res = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", natural=True)
        tl = res["plan"]["timeline"]
        assert res["plan"]["story"]["natural"] and res["plan"]["story"]["lab"] == {
            "stem": True, "phrase": True, "shot": True, "join": True, "min_lines": False}
        assert res["plan"]["segmentation"]["join"] is True and res["shortfall"] == 0
        assert PC.NATURAL in PC.used_line(res["used"]) and PC.LAB["join"] not in PC.used_line(res["used"])
        both = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", lab={"shot": True, "join": True})
        assert _sig(both["plan"]["timeline"]) == _sig(tl)                          # the same thing as its two halves
        assert PC.LAB["join"] in PC.used_line(both["used"]) and PC.NATURAL not in PC.used_line(both["used"])
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
        # 🔗 join: two blocks of ONE part may become one longer clip — never across a line, never over the limit
        tr = load_track(DATA, "song_full")
        segs = SCUT.line_segments(tr, PC._section_regions(tr))
        joined = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines", natural=True)
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


def test_natural_and_the_lab_on_the_page(monkeypatch):
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import planner.compose as PC
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "🤖 New matcher"
        at.run()
        assert not errors(at), errors(at)
        gone = {"compose_lab_" + k for k in ("stem", "phrase", "flow", "shot", "join")}
        assert not [t for t in at.toggle if t.key in gone]                 # no switches for these any more
        nat = at.toggle(key="compose_natural")
        assert nat.label == PC.NATURAL and nat.disabled and not nat.value   # needs 🎯 Movement only or the 📐 cuts
        assert not [w for w in at.select_slider if w.key == "compose_lab_power"]      # baked into the slider
        assert at.toggle(key="compose_lab_min_lines").disabled
        at.session_state["compose_movement_only"] = True
        at.session_state["compose_cuts"] = "lines"
        at.run()
        assert not errors(at), errors(at)
        assert not at.toggle(key="compose_natural").disabled and at.number_input(key="compose_min_clip").disabled
        at.toggle(key="compose_natural").set_value(True)
        at.slider(key="compose_smooth").set_value(0.85)
        at.toggle(key="compose_lab_min_lines").set_value(True).run()
        assert not errors(at), errors(at)
        assert not at.number_input(key="compose_min_clip").disabled
        assert any("🧪 Lab — 1 on" in e.label for e in at.expander)
        at.button(key="cs_go").click().run()
        assert not errors(at), errors(at)
        res = at.session_state[PC.RESULT_KEY]
        assert res["plan"]["story"]["natural"] and res["plan"]["story"]["lab"] == {
            "stem": True, "phrase": True, "shot": True, "join": True, "min_lines": True}
        assert any(c.value.startswith("🧾 This compile:") and PC.NATURAL in c.value and "Smooth ↔ Contrast 0.85" in c.value
                   for c in at.caption)
        at.toggle(key="compose_sync_data").set_value(True).run()
        assert not errors(at), errors(at)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_contrast_welcomes_a_change_of_shot_in_the_quiet_passages_only(monkeypatch):
    """⚡ Baked into Smooth ↔ Contrast (13 Nov). One video is all wide shots that read nearly still, as wide shots
    do; the rest are close-ups. 🎯 Movement only, the slider at Contrast: the wide video comes in where the music has
    no beat and never where the drums play; at the slider's middle the plan is what it always was."""
    from tests.test_story_matcher import _plan_inputs, _setup
    out = _setup(monkeypatch)
    try:
        import audio_drive as AD
        import core.story_matcher as SMX
        import footage_timeline as FT
        from core import sync as S
        from tests.test_movement_lab import _beatless_track, _full_rate_everywhere

        def wide(vid):
            with np.load(out / f"{vid}.npz") as z:
                arrays = {k: z[k] for k in z.files}
            arrays["shot"] = np.full(len(arrays["times"]), 3.0, np.float16)
            arrays["move"] = np.full(len(arrays["times"]), 0.05, np.float16)
            FT._write_video(out, vid, arrays)
        _full_rate_everywhere(out)
        wide("vid_c")
        track, t0, t1 = _beatless_track()
        drv = AD.drivers(track, None)
        beat = lambda a, b: S.beat_presence(drv["curves"]["drums"][int(a * drv["rate"]):int(b * drv["rate"])], AD.DRUM_LEVEL)   # noqa: E731
        types = ["rhythmic"] * 16

        def run(contrast):
            _, queues, segments, regions = _plan_inputs(types)
            opts = SMX.StoryOptions(section_map=regions, movement_only=True, smooth_contrast=contrast)
            tl, _, _ = SMX.match_story(segments, queues, track, options=opts)
            return tl
        mid, full = run(0.5), run(1.0)
        on_wide = lambda tl: [e for e in tl if e["scenes"][0]["video_id"] == "vid_c" and beat(*e["track_time"]) >= 0.99]   # noqa: E731
        off_wide = lambda tl: [e for e in tl if e["scenes"][0]["video_id"] == "vid_c" and beat(*e["track_time"]) < SMX.OFF_BEAT]   # noqa: E731
        assert len(off_wide(full)) > len(off_wide(mid)) and len(off_wide(full)) >= 1        # welcome in the quiet passage
        assert len(on_wide(full)) <= len(on_wide(mid))                                      # …and not pushed onto a full beat
        gained = [e for e in full if e["scenes"][0]["video_id"] == "vid_c"
                  and tuple(e["track_time"]) not in {tuple(x["track_time"]) for x in mid if x["scenes"][0]["video_id"] == "vid_c"}]
        assert gained and all(beat(*e["track_time"]) < 1.0 for e in gained)
        # the slider is all there is: no power option any more
        assert not hasattr(SMX.StoryOptions(), "shot_power") and 0.05 <= SMX.CONTRAST_QUIET <= 0.2
        import planner.compose as PC
        r = PC.compile_plan("song_full", ["vid_a", "vid_b", "vid_c", "vid_f"], movement_only=True, cuts="lines", smooth_contrast=0.85)
        assert "Smooth ↔ Contrast 0.85" in PC.used_line(r["used"]) and "power" not in r["plan"]["story"]["lab"]
        assert "Smooth ↔ Contrast" not in PC.used_line(PC.compile_plan("song_full", ["vid_a", "vid_b"], movement_only=True)["used"])
        # recipes mode: the slider's old rule, untouched (a change is welcome at a section change, beat or not)
        t_r, q_r, s_r, reg_r = _plan_inputs(["rhythmic", "action"] * 8)
        a, _, _ = SMX.match_story(s_r, q_r, t_r, options=SMX.StoryOptions(section_map=reg_r, smooth_contrast=1.0))
        assert len(a) == 16
    finally:
        shutil.rmtree(out, ignore_errors=True)
