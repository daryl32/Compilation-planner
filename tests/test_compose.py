"""🤖 The new matcher's own flow: section cuts at the recipes' pace, suggesting songs / videos for chosen videos,
compiling a plan, and the planner landing (Classic | New matcher) with By song / By video headless."""
from __future__ import annotations

import json
import shutil

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.test_story_matcher import _setup as _timelines


def test_section_cuts_follow_the_pace_and_boundaries():
    import recipes as RC
    import section_cuts as SCUT
    from tests.fixtures import load_track
    track = load_track(DATA, "song_full")
    regions = [{"start": 0.0, "end": 24.0, "type": "intro"}, {"start": 24.0, "end": 48.0, "type": "buildup"},
               {"start": 48.0, "end": 72.0, "type": "drop"}, {"start": 72.0, "end": 96.0, "type": "ending"}]
    segs = SCUT.story_segments(track, regions, RC.load_all())
    assert segs[0]["start"] == 0.0 and abs(segs[-1]["end"] - 96.0) < 1e-6
    for a, b in zip(segs, segs[1:]):
        assert abs(a["end"] - b["start"]) < 1e-6 and b["end"] > b["start"]
    starts = {round(s["start"], 3) for s in segs}
    assert {24.0, 48.0, 72.0} <= starts                       # section boundaries are always cuts
    by = {}
    for s in segs:
        by.setdefault(s["section_type"], []).append(s["end"] - s["start"])
    assert sum(by["drop"]) == 24.0 and sum(by["intro"]) == 24.0
    import numpy as np
    assert np.mean(by["drop"]) < np.mean(by["intro"])         # drops cut faster than the intro
    assert 0.5 <= np.mean(by["drop"]) <= 3.0 and np.mean(by["intro"]) >= 3.0
    # a rising build-up shortens its clips through the section
    bu = [s for s in segs if s["section_type"] == "buildup"]
    if len(bu) >= 3:
        assert (bu[0]["end"] - bu[0]["start"]) >= (bu[-1]["end"] - bu[-1]["start"]) - 0.6
    assert "blocks" in SCUT.describe(segs) and all("energy" in s and "intensity" in s for s in segs)


def test_compile_suggest_and_result(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import planner.compose as PC
        import footage_timeline as FT
        res = PC.compile_plan("song_full", ["vid_a", "vid_b", "vid_c", "vid_f"])
        plan = res["plan"]
        assert plan["matching_mode"] == "Auto story" and plan["timeline"] and plan["segmentation"]["method"] == "sections"
        assert all(len(e["scenes"]) <= 1 for e in plan["timeline"])                 # split screen off
        assert all("source_path" in link for e in plan["timeline"] for s in e["scenes"] for link in s["chain"])
        assert res["summary"]["slots"] == len(plan["timeline"])
        res2 = PC.compile_plan("song_full", ["vid_a", "vid_b", "vid_c", "vid_f"], split_screen=True)
        assert any(len(e["scenes"]) > 1 for e in res2["plan"]["timeline"])
        # songs for chosen videos, and other videos like them
        tracks = PC.suggest_tracks(("vid_f",), PC._audio_signature(), FT.signature())
        assert tracks and all(0 <= t["fit"] <= 1 for t in tracks) and tracks == sorted(tracks, key=lambda t: -t["fit"])
        others = PC.suggest_videos(("vid_a",), FT.signature())
        assert others and "vid_a" not in {o["video_id"] for o in others}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_planner_landing_and_new_matcher_pages(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "🤖 New matcher"
        at.run()
        assert not errors(at), errors(at)
        assert at.radio(key="compose_way").value == "🎵 By song"
        assert at.selectbox(key="cs_track").value
        at.button(key="cs_go").click().run()
        assert not errors(at), errors(at)
        assert any("blocks" in c.value for c in at.caption)
        assert [b for b in at.button if b.key == "compose_save"]
        at.button(key="compose_back").click().run()
        assert not errors(at), errors(at)
        at.session_state["compose_way"] = "🎬 By video"
        at.run()
        assert not errors(at), errors(at)
        at.multiselect(key="cv_start").set_value(["vid_a"]).run()
        assert not errors(at), errors(at)
        at.button(key="cv_find").click().run()                 # stage 1: songs
        assert not errors(at), errors(at)
        assert at.radio(key="cv_track").value
        at.button(key="cv_more").click().run()                 # stage 2: other videos (optional)
        assert not errors(at), errors(at)
        at.button(key="cv_go").click().run()
        assert not errors(at), errors(at)
        assert at.session_state[PC_RESULT()]["way"] == "video"
        # 🔁 on the result page: 🚫 a clip → its block's next-best clips show; ✅ one swaps it in, 🔁 redoes the block
        at.session_state["compose_movement_only"] = True
        at.button(key="compose_again").click().run()
        assert not errors(at), errors(at)
        res = at.session_state[PC_RESULT()]
        import core.story_matcher as SMX
        tl = res["plan"]["timeline"]
        k = next((i for i in range(1, len(tl)) if tl[i]["scenes"] and SMX.free_alternatives(tl, i, res["alternatives"].get(i))), None)
        if k is not None:
            s0 = tl[k]["scenes"][0]
            rej = [b for b in at.button if b.key and b.key.startswith(f"compose_rej_{tl[k]['segment_index']}_{s0['video_id']}_")]
            rej[0].click().run()
            assert not errors(at), errors(at)
            use = [b for b in at.button if b.key and b.key.startswith(f"compose_alt_{k}_")]
            assert use and [b for b in at.button if b.key == "compose_redo"]
            before = [[(l["video_id"], l["clip_start_sec"]) for s_ in e["scenes"] for l in s_["chain"]] for e in tl]
            use[0].click().run()
            assert not errors(at), errors(at)
            t2 = at.session_state[PC_RESULT()]["plan"]["timeline"]
            after = [[(l["video_id"], l["clip_start_sec"]) for s_ in e["scenes"] for l in s_["chain"]] for e in t2]
            assert after[k] != before[k] and all(a_ == b_ for i, (a_, b_) in enumerate(zip(before, after)) if i != k)
            assert t2[k]["scenes"][0]["story"].get("chosen")
            at.button(key="compose_redo").click().run() if [b for b in at.button if b.key == "compose_redo"] else None
            assert not errors(at), errors(at)
        # Classic is untouched behind the landing
        at.button(key="compose_back").click().run()
        at.session_state["planner_mode"] = "🎛️ Classic"
        at.run()
        assert not errors(at), errors(at)
        assert at.radio(key="ui_section").value
    finally:
        shutil.rmtree(out, ignore_errors=True)


def PC_RESULT():
    import planner.compose as PC
    return PC.RESULT_KEY


def test_redo_only_the_marked_blocks_and_swap_in_an_alternative(monkeypatch):
    """🔁 Reject a clip → only its block is matched again, every other clip stays exactly as it was and no footage
    is used twice; the block's next-best clips are kept, and one of them can be put in directly."""
    out = _timelines(monkeypatch)
    try:
        import core.story_matcher as SMX
        import planner.compose as PC
        vids = ["vid_a", "vid_b", "vid_c", "vid_f"]
        for only in (True, False):
            res = PC.compile_plan("song_full", vids, movement_only=only)
            res["video_ids"] = vids
            tl = res["plan"]["timeline"]
            assert set(res["alternatives"]) == set(range(len(tl)))
            sig = lambda e: [(l["video_id"], l["scene_id"], l["clip_start_sec"], l["clip_duration_sec"])   # noqa: E731
                             for s in e["scenes"] for l in s["chain"]]
            picked = [i for i, e in enumerate(tl) if e["scenes"] and e["scenes"][0]["story"].get("fit") is not None and i > 0]
            k = next((i for i in picked if SMX.free_alternatives(tl, i, res["alternatives"][i])), picked[0])
            gone = (tl[k]["scenes"][0]["video_id"], tl[k]["scenes"][0]["scene_id"])
            new = PC.rework_plan(res, {k}, excluded={gone})
            t2 = new["plan"]["timeline"]
            assert len(t2) == len(tl) and new["reworked"] == [k]
            assert all(sig(a) == sig(b) for i, (a, b) in enumerate(zip(tl, t2)) if i != k)        # the rest untouched
            assert gone not in {(l["video_id"], l["scene_id"]) for s in t2[k]["scenes"] for l in s["chain"]}
            assert all(abs(a["track_time"][0] - b["track_time"][0]) < 1e-6 for a, b in zip(tl, t2))
            used = {}
            for e in t2:                                                                          # nothing used twice
                for s in e["scenes"]:
                    for l in s["chain"]:
                        a_ = l["clip_start_sec"]
                        for b0, b1 in used.get(l["video_id"], []):
                            assert min(b1, a_ + l["clip_duration_sec"]) - max(b0, a_) <= 0.05
                        used.setdefault(l["video_id"], []).append((a_, a_ + l["clip_duration_sec"]))
            # the alternatives on offer are free, and one can be put in as it is
            alts = SMX.free_alternatives(t2, k, new["alternatives"][k], {gone})
            assert alts or not only                               # the movement-only plan has one to offer here
            if alts:
                pick = alts[-1]
                t3 = PC.rework_plan(new, set(), {k: pick}, excluded={gone})["plan"]["timeline"]
                slot = t3[k]["scenes"][0]
                assert slot["video_id"] == pick["video_id"] and slot["story"].get("chosen")
                assert [(l["scene_id"], round(l["offset_into_scene_sec"], 2)) for l in slot["chain"]] == \
                       [(p[0], round(p[1], 2)) for p in pick["pieces"]]
                assert all(sig(a) == sig(b) for i, (a, b) in enumerate(zip(t2, t3)) if i != k)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_cuts_from_the_lines(monkeypatch):
    """✂️ Cuts: 📐 lines from the music — the new matcher takes its blocks from section_bounds (the 🅱 New cuts of
    the Sections page) instead of the section map + recipe pace; every block kept as cut; the default is unchanged."""
    out = _timelines(monkeypatch)
    try:
        import planner.compose as PC
        import section_bounds as SB
        import section_cuts as SCUT
        from tests.fixtures import load_track
        from tests.support import DATA
        tr = load_track(DATA, "song_full")
        want = SB.for_track(tr)["blocks"]
        segs = SCUT.line_segments(tr, PC._section_regions(tr))
        assert [(s["start"], s["end"]) for s in segs][:-1] == [(b["start"], b["end"]) for b in want][:-1]
        assert segs[0]["start"] == 0.0 and segs[-1]["end"] == tr["duration_sec"]
        assert all({"energy", "intensity", "driver", "section_type", "line_section"} <= set(s) for s in segs)
        assert all(a["end"] == b["start"] for a, b in zip(segs, segs[1:]))
        vids = ["vid_a", "vid_b", "vid_c", "vid_f"]
        res = PC.compile_plan("song_full", vids, movement_only=True, cuts="lines")
        plan = res["plan"]
        assert plan["segmentation"]["method"] == "lines" and plan["segmentation"]["vocal_rule"] is True
        assert res["segments"] == segs and any("📐 lines from the music" in n for n in res["notes"])
        got = [tuple(e["track_time"]) for e in plan["timeline"]]
        cuts = {round(s["start"], 2) for s in segs} | {round(segs[-1]["end"], 2)}
        assert got[0][0] == 0.0 and got[-1][1] == round(tr["duration_sec"], 2)
        # every 📐 cut is a cut of the plan — none joined away. (A block no footage fits may still be split on the
        # beat, as always: that adds cuts, it never removes one.)
        assert cuts <= {t for pair in got for t in pair}, sorted(cuts - {t for pair in got for t in pair})
        assert len(got) == len(segs)                                           # here: block for block
        off = PC.compile_plan("song_full", vids, movement_only=True, cuts="lines", vocal_rule=False)
        assert off["plan"]["segmentation"]["vocal_rule"] is False
        old = PC.compile_plan("song_full", vids, movement_only=True)
        assert old["plan"]["segmentation"] == PC.compile_plan("song_full", vids, movement_only=True, cuts="sections")["plan"]["segmentation"]
        assert old["plan"]["segmentation"]["method"] == "sections" and "vocal_rule" not in old["plan"]["segmentation"]
        monkeypatch.setattr(SCUT, "line_segments", lambda *a, **k: [])       # nothing found: the section map, with a note
        back = PC.compile_plan("song_full", vids, movement_only=True, cuts="lines")
        assert back["plan"]["segmentation"]["method"] == "sections" and any("section map instead" in n for n in back["notes"])
    finally:
        shutil.rmtree(out, ignore_errors=True)
