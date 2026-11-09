"""🤖 The story matcher (Stage 3 step 5): plans from synthetic timelines + a section map + the recipes, in Classic's
plan shape — recipes steer the picks, footage is never reused, exclusions hold, and blocks without a timeline
window fall back to Classic's chain fill."""
from __future__ import annotations

import json
import shutil

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)

TAGS = ["Blowjob", "Cowgirl", "penetration", "Face", "lingerie", "close-up", "wide", "POV", "Blonde"]
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
    scores[hi, TAGS.index("Cowgirl")] = 85
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
