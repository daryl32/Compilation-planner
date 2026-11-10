"""🤖 The story matcher (Stage 3 step 5): plans from synthetic timelines + a section map + the recipes, in Classic's
plan shape — recipes steer the picks, footage is never reused, exclusions hold, and blocks without a timeline
window fall back to Classic's chain fill."""
from __future__ import annotations

import json
import shutil

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)

TAGS = ["Blowjob", "Cowgirl position", "penetration", "Face", "lingerie", "close-up", "wide", "POV", "Blonde"]
CUTS = [50] * len(TAGS)


def _timeline_for(FT, out, video: dict, busy: bool):
    """A timeline whose first half is calm & lingerie / face, second half busy & penetration / cowgirl —
    or (busy=True) busy throughout."""
    dur = video["duration"]
    times = np.arange(0, dur, 0.5, dtype=np.float32)
    T = len(times)
    half = T // 2
    move = np.full(T, 0.9 if busy else 0.1, np.float32)
    if not busy:
        move[half:] = 0.85
    scores = np.full((T, len(TAGS)), 5, np.uint8)
    lo = slice(0, 0 if busy else half)
    hi = slice(0 if busy else half, T)
    scores[lo, TAGS.index("lingerie")] = 90
    scores[lo, TAGS.index("Face")] = 80
    scores[hi, TAGS.index("penetration")] = 90
    scores[hi, TAGS.index("Cowgirl position")] = 85
    scores[:, TAGS.index("close-up")] = 70
    shot = np.full(T, 1.0, np.float32)
    bpm = np.full(T, 124.0, np.float32)
    arrays = {"times": times, "move": move.astype(np.float16), "change": np.zeros(T, np.float16),
              "bpm": bpm.astype(np.float16), "periodicity": np.full(T, 0.8, np.float16),
              "onset": np.zeros(T, np.float16), "scores": scores, "labelled": np.zeros(T, bool),
              "shot": shot.astype(np.float16), "people": np.ones(T, np.uint8),
              "emb": np.zeros((T, 8), np.int8), "cuts": np.zeros(0, np.float32)}
    FT._write_video(out, video["video_id"], arrays)
    return T


def _setup(monkeypatch):
    import footage_timeline as FT
    import catalogue_store as CS
    from config import CATALOGUE_DIR
    out = CATALOGUE_DIR / FT.FOLDER
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    lib = CS.video_index()
    vids = {v["video_id"]: v for v in lib.values()}
    for vid in ("vid_a", "vid_b", "vid_c", "vid_f"):
        _timeline_for(FT, out, vids[vid], busy=vid == "vid_f")
    (out / "index.json").write_text(json.dumps({"version": FT.VERSION, "model": "v14", "tags": TAGS, "cutoffs": CUTS,
                                                "step": 0.5, "emb_dim": 8, "move_scale": {"lo": 0.0, "hi": 1.0},
                                                "videos": {v: {"duration": vids[v]["duration"]} for v in ("vid_a", "vid_b", "vid_c", "vid_f")}}))
    return out


def _plan_inputs(section_types_list):
    import core.footage as CF
    import catalogue_store as CS
    from tests.fixtures import load_track
    track = load_track(DATA, "song_full")
    cats = CS.all_catalogues()
    queues = CF.build_footage_queues(cats, ("vid_a", "vid_b", "vid_c", "vid_d", "vid_f"), ())
    n = len(section_types_list)
    length = track["duration_sec"] / n
    segments = [{"start": round(i * length, 2), "end": round((i + 1) * length, 2), "energy": 0.5, "intensity": 0.5}
                for i in range(n)]
    regions = [{"start": s["start"], "end": s["end"], "type": t} for s, t in zip(segments, section_types_list)]
    return track, queues, segments, regions


def test_recipes_steer_the_picks_and_footage_is_never_reused(monkeypatch):
    out = _setup(monkeypatch)
    try:
        import core.story_matcher as SMX
        types = ["intro", "intro", "slow", "slow", "buildup", "buildup", "drop", "drop", "drop", "drop", "fast", "fast",
                 "warmdown", "warmdown", "ending", "ending"]          # 6 s blocks — the fixture scenes are short
        track, queues, segments, regions = _plan_inputs(types)
        opts = SMX.StoryOptions(section_map=regions, music_story=0.6)
        timeline, short, unfilled = SMX.match_story(segments, queues, track, options=opts)
        assert len(timeline) == len(segments) and short == 0 and unfilled < 0.1
        assert [e["section_type"] for e in timeline] == types
        used = []
        for e in timeline:
            assert e["clip_count"] == 1 and len(e["scenes"]) == 1
            s = e["scenes"][0]
            for k in ("chain", "video_id", "scene_id", "clip_start_sec", "clip_duration_sec", "offset_into_scene_sec",
                      "thumbnail", "motion_norm", "tags", "story"):
                assert k in s
            assert abs(s["clip_duration_sec"] - (e["track_time"][1] - e["track_time"][0])) < 0.05
            used.append((s["video_id"], s["clip_start_sec"], s["clip_start_sec"] + s["clip_duration_sec"]))
        # no two picks overlap in the same video
        for a in range(len(used)):
            for b in range(a + 1, len(used)):
                if used[a][0] == used[b][0]:
                    assert used[a][2] <= used[b][1] + 1e-6 or used[b][2] <= used[a][1] + 1e-6
        # the drop / fast blocks land in busy, penetration footage; the intro / slow in calm, lingerie footage
        by_type = {}
        for e in timeline:
            by_type.setdefault(e["section_type"], e["scenes"][0])        # the first block of each type
        assert by_type["drop"]["story"]["fit"] > 0.6 and by_type["intro"]["story"]["fit"] > 0.6
        assert by_type["intro"]["video_id"] != "vid_f"                     # vid_f is busy throughout
        summary = SMX.plan_summary(timeline)
        assert summary["fallback"] <= 2 and summary["slots"] == len(segments) and summary["mean_fit"] > 0.5
        assert not opts.notes or all("no 🧭 timeline" in n_ for n_ in opts.notes)   # vid_d has no timeline
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_exclusions_split_screen_and_classic_fallback(monkeypatch):
    out = _setup(monkeypatch)
    try:
        import core.story_matcher as SMX
        types = ["drop"] * 12
        track, queues, segments, regions = _plan_inputs(types)
        import copy
        q2 = copy.deepcopy(queues)
        base, _, _ = SMX.match_story(segments, copy.deepcopy(queues), track, options=SMX.StoryOptions(section_map=regions))
        first = base[0]["scenes"][0]
        key = (first["video_id"], first["scene_id"])
        tl2, _, _ = SMX.match_story(segments, q2, track, segment_exclusions={0: {key}},
                                    options=SMX.StoryOptions(section_map=regions))
        assert (tl2[0]["scenes"][0]["video_id"], tl2[0]["scenes"][0]["scene_id"]) != key
        assert tl2[0]["manually_edited"]
        # split screen: two distinct videos per block
        tl3, _, _ = SMX.match_story(segments, copy.deepcopy(queues), track, min_clips=2, max_clips=2,
                                    options=SMX.StoryOptions(section_map=regions))
        for e in tl3:
            assert e["split_screen"] and len({s["video_id"] for s in e["scenes"]}) == len(e["scenes"]) == 2
        # only a video without a timeline → every slot is Classic's fallback, with a note
        import core.footage as CF
        import catalogue_store as CS
        q4 = CF.build_footage_queues(CS.all_catalogues(), ("vid_d",), ())
        opts = SMX.StoryOptions(section_map=regions)
        tl4, _, _ = SMX.match_story(segments, q4, track, options=opts)
        assert all(s["story"].get("fallback") == "classic" for e in tl4 for s in e["scenes"])
        assert any("no 🧭 timeline" in n_ for n_ in opts.notes)
        assert SMX.plan_summary(tl4)["fallback"] == SMX.plan_summary(tl4)["slots"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_section_map_suggested_when_none_saved(monkeypatch):
    out = _setup(monkeypatch)
    try:
        import core.story_matcher as SMX
        track, queues, segments, _ = _plan_inputs(["rhythmic"] * 12)
        opts = SMX.StoryOptions()
        timeline, _, _ = SMX.match_story(segments, queues, track, options=opts)
        assert any("suggested one was used" in n_ for n_ in opts.notes)
        assert all(e["section_type"] for e in timeline)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_planner_page_auto_story_end_to_end(monkeypatch):
    """The planner in Auto mode with the 🤖 Auto story engine: matches, shows the section line, saves a plan the
    Blender add-on can import (same shape as Classic's)."""
    out = _setup(monkeypatch)
    try:
        from tests.page_scenarios import run_scenario
        res = run_scenario(dict(track="song_full", method="smart", state={"matcher_engine": "🤖 Auto story"}))
        assert not res["problems"], res["problems"]
        plan = res["plan"]
        assert plan and plan.get("timeline") or plan.get("segments") or plan
        text = json.dumps(plan)
        assert "clip_start_sec" in text or "clip_start" in text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_max_videos_limit_and_progress(monkeypatch):
    out = _setup(monkeypatch)
    try:
        import core.story_matcher as SMX
        track, queues, segments, regions = _plan_inputs(["drop"] * 12)
        seen = []
        opts = SMX.StoryOptions(section_map=regions, max_videos=2, progress=lambda d, t: seen.append((d, t)))
        import copy
        free, _, _ = SMX.match_story(segments, copy.deepcopy(queues), track, options=SMX.StoryOptions(section_map=regions))
        timeline, _, _ = SMX.match_story(segments, queues, track, options=opts)
        story_used = {c["video_id"] for e in timeline for s in e["scenes"] for c in s["chain"]
                      if not (s.get("story") or {}).get("fallback")}
        assert len(story_used) <= 2                                    # the timeline picks respect the limit
        assert SMX.plan_summary(timeline)["videos"] <= SMX.plan_summary(free)["videos"]
        assert seen[0] == (0, 12) and seen[-1] == (12, 12)
        assert SMX.plan_summary(timeline)["videos"] >= 1
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _set_cuts(out, video_ids, cuts_of):
    """Rewrite the test timelines' hard cuts: cuts_of(video row) → list of seconds."""
    import numpy as np
    import catalogue_store as CS
    import footage_timeline as FT
    rows = {v["video_id"]: v for v in CS.video_index().values()}
    for vid in video_ids:
        with np.load(out / f"{vid}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        arrays["cuts"] = np.asarray(cuts_of(rows[vid]), np.float32)
        FT._write_video(out, vid, arrays)


def _contiguous(chain):
    return all(abs(a["clip_start_sec"] + a["clip_duration_sec"] - b["clip_start_sec"]) < 0.06 for a, b in zip(chain, chain[1:]))


def test_windows_run_across_clip_edges_but_not_hard_cuts(monkeypatch):
    """12 s blocks over fixture clips of 1.5–14 s: with continuous footage the window crosses clip edges (a chain of
    back-to-back pieces); a hard cut on every clip edge makes them walls again."""
    out = _setup(monkeypatch)
    try:
        import copy
        import core.story_matcher as SMX
        track, queues, segments, regions = _plan_inputs(["rhythmic"] * 8)
        tl, short, unfilled = SMX.match_story(segments, copy.deepcopy(queues), track,
                                              options=SMX.StoryOptions(section_map=regions))
        ps = SMX.plan_summary(tl)
        assert short == 0 and unfilled < 0.1 and ps["fallback"] == 0
        joined = [s for e in tl for s in e["scenes"] if (s["story"] or {}).get("joined")]
        assert joined and ps["joined"] == len(joined)
        for e in tl:
            for s in e["scenes"]:
                assert abs(s["clip_duration_sec"] - (e["track_time"][1] - e["track_time"][0])) < 0.1
                if s["story"].get("joined"):
                    assert len(s["chain"]) == s["story"]["joined"] > 1 and _contiguous(s["chain"])
                    assert len({c["video_id"] for c in s["chain"]}) == 1
                    assert min(c["clip_duration_sec"] for c in s["chain"]) >= SMX.MIN_PIECE_SEC - 0.02
        # no footage used twice, pieces included
        used = sorted((c["video_id"], c["clip_start_sec"], c["clip_start_sec"] + c["clip_duration_sec"])
                      for e in tl for s in e["scenes"] for c in s["chain"])
        for a, b in zip(used, used[1:]):
            assert a[0] != b[0] or a[2] <= b[1] + 0.02
        # an excluded clip is a wall: vid_b's scene 2 is excluded in the fixture — nothing of it is used
        assert not [c for e in tl for s in e["scenes"] for c in s["chain"] if c["video_id"] == "vid_b" and c["scene_id"] == 2]
        # hard cuts on every clip edge → no window crosses an edge any more
        _set_cuts(out, ("vid_a", "vid_b", "vid_c", "vid_f"), lambda v: [s["end"] for s in v["scenes"]])
        tl2, _, _ = SMX.match_story(segments, copy.deepcopy(queues), track, options=SMX.StoryOptions(section_map=regions))
        assert SMX.plan_summary(tl2)["joined"] == 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_blocks_too_long_for_any_window_are_split_on_the_beat(monkeypatch):
    out = _setup(monkeypatch)
    try:
        import copy
        import core.story_matcher as SMX
        _set_cuts(out, ("vid_a", "vid_b", "vid_c", "vid_f"), lambda v: [s["end"] for s in v["scenes"]])
        track, queues, segments, regions = _plan_inputs(["rhythmic"] * 6)          # 16 s blocks; no clip is that long
        tl, short, _ = SMX.match_story(segments, copy.deepcopy(queues), track, options=SMX.StoryOptions(section_map=regions))
        ps = SMX.plan_summary(tl)
        split = [(e, s) for e in tl for s in e["scenes"] if (s["story"] or {}).get("parts")]
        assert split and ps["split"] == len(split)
        for e, s in split:
            assert s["story"]["parts"] >= 2 and s["story"]["fit"] is not None
            assert abs(s["clip_duration_sec"] - (e["track_time"][1] - e["track_time"][0])) < 0.15
            assert all(c["clip_duration_sec"] >= SMX.MIN_PART_SEC - 0.05 for c in s["chain"])
        assert ps["fallback"] < ps["slots"]
        # a failed split leaves the queues as they were (nothing half-carved)
        q = copy.deepcopy(queues)
        before = copy.deepcopy(q)
        only_d = {k: v for k, v in q.items() if k == "vid_d"}                       # no timeline → nothing to split with
        SMX.match_story(segments[:1], only_d, track, max_clips=1, options=SMX.StoryOptions(section_map=regions))
        assert {k: v for k, v in q.items() if k != "vid_d"} == {k: v for k, v in before.items() if k != "vid_d"}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_story_score_uses_the_best_two_wanted_tags(monkeypatch):
    """A long wanted list doesn't water a window down: two strong tags out of twelve score like two out of two."""
    out = _setup(monkeypatch)
    try:
        import numpy as np
        import core.story_matcher as SMX
        import footage_timeline as FT
        index = FT.load_index()
        v = SMX._Video(FT.load("vid_f", index=index), index)          # busy, penetration + Cowgirl position throughout
        short = {"movement": {"lo": 0.7, "hi": 1.0, "trend": "flat"}, "beat_lock": 0.5, "avoid": [],
                 "tags": {"penetration": 0.8, "Cowgirl position": 0.6}}
        long = {**short, "tags": {**short["tags"], **{f"other{k}": 0.6 for k in range(10)}, "Face": 0.6, "lingerie": 0.6}}
        _, _, s_short = SMX._score_video(v, 8, short, None, 0.0, 0.5)
        _, _, s_long = SMX._score_video(v, 8, long, None, 0.0, 0.5)
        assert float(s_short.mean()) > 0.8 and abs(float(s_short.mean()) - float(s_long.mean())) < 0.02
    finally:
        shutil.rmtree(out, ignore_errors=True)
