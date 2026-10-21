"""📍 The planned path in Manual step-through and Choreography (in order): the same per-video plan
Auto uses, from the block you're on, each clip inside one scene — offered as the block's picks,
followed by Auto-fill ALL, and re-planned only when you pick something else."""

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.apptest_util import errors, page_app
from tests.fixtures import VIDEOS, load_track
from tests.test_sequential import _ctx

PAGE = "pages/1_Compilation_Planner.py"


def _plan(within=True, from_block=0, F=1.0):
    import catalogue_store as CStore
    import core.footage as F_
    import core.matching as M
    import core.recommend as REC
    import core.sequential as SEQ
    import scene_roles as SR
    tr = load_track(DATA, "song_full")
    segs = [dict(s) for s in REC.cut_variant(tr, {"sm_pace": 0})]
    q = F_.build_footage_queues(CStore.all_catalogues(), ("vid_a", "vid_b", "vid_c", "vid_f"), (), {},
                                roles_of=SR.scene_roles)
    ctx = _ctx(True)
    ctx.plan_within_scenes = within
    sub = segs[from_block:]
    tl, _, _ = M.match_scenes_to_track(
        sub, q, 1, 1, 0.0, 1.0, False, True, {}, {}, set(), False, 0, 1.0, set(), set(),
        2.0, 0.5, True, 1.5, 2.0, 42, weight_position=F, repeat_fallbacks=set(),
        block_roles=[None] * len(sub), track=tr, ctx=ctx)
    return tl, segs, SEQ.planned_picks(tl, sub, offset=from_block)


def test_within_scenes_gives_whole_block_picks_in_order():
    from core.sequential import in_order
    tl, segs, picks = _plan()
    assert in_order(tl)
    for e in tl:
        for s in e["scenes"]:
            assert len(s["chain"]) == 1                       # never crosses into the next scene
    assert set(picks) == set(range(len(segs)))
    filled = sum(1 for b in picks if picks[b])
    assert filled >= 0.8 * len(segs)
    for b, ps in picks.items():
        for p in ps:
            assert abs(p["clip_duration_sec"] - (segs[b]["end"] - segs[b]["start"])) < 0.05


def test_offset_numbers_the_real_blocks():
    _, segs, picks = _plan(from_block=3)
    assert min(picks) == 3 and max(picks) == len(segs) - 1


def test_each_video_is_spread_through_the_rest_of_the_track():
    """F > 0: a video used early and late in the plan has its clips spread through it, not bunched
    at its start (what one-block-at-a-time picking did)."""
    _, segs, picks = _plan(F=4.0)
    by_vid = {}
    for b in sorted(picks):
        for p in picks[b]:
            by_vid.setdefault(p["video_id"], []).append((b, p["clip_start_sec"]))
    spread = [v for v, uses in by_vid.items() if len(uses) >= 3]
    assert spread
    import catalogue_store as CStore
    from core.footage import _tc_to_seconds
    for v in spread:
        starts = [t for _, t in by_vid[v]]
        assert starts == sorted(starts)
        end = max(_tc_to_seconds(s["end_tc"]) for s in CStore.load_catalogue(v)["scenes"])
        assert starts[-1] - starts[0] > 0.3 * end   # reaches well into the video


def _app(mode):
    import shutil
    for sub in ("projects", "compilation_plans"):
        shutil.rmtree(DATA / sub, ignore_errors=True)
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(DATA / "recs-.curve_cache", ignore_errors=True)
    at = page_app(PAGE)
    state = {"track_id": "song_full", "segmentation_method": "Smart (self-tuning)",
             "committed_selected_videos": list(VIDEOS), "matching_mode": mode, "sequential_mode": True,
             "weight_position": 2.0, "ui_section": "🎬 3. Matching & Export"}
    for v in VIDEOS:
        state[f"select_video_{v}"] = True
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not errors(at), errors(at)
    return at


def _in_order(confirmed: dict) -> bool:
    last = {}
    for b in sorted(confirmed):
        for p in confirmed[b]:
            if p["clip_start_sec"] < last.get(p["video_id"], -1) - 1e-3:
                return False
            last[p["video_id"]] = p["clip_start_sec"] + p["clip_duration_sec"]
    return True


def test_manual_offers_the_planned_clips_and_autofill_follows_them():
    at = _app("Manual step-through")
    assert any("Planned path" in c.value for c in at.caption)
    assert any("on the planned path" in c.value for c in at.caption)
    plan = at.session_state["_planned_path"]["picks"]
    # ⚡ Auto-fill top N ticks the planned clips
    btn = [b for b in at.button if b.label.startswith("⚡ Auto-fill top")]
    btn[0].click().run()
    assert not errors(at), errors(at)
    ticked = {c.key for c in at.checkbox if (c.key or "").startswith("adv_pick_0_") and c.value}
    assert ticked == {f"adv_pick_0_{p['video_id']}_{p['scene_id']}" for p in plan[0]}
    # ⚡⚡ Auto-fill ALL follows the plan, in order, and fills every block
    [b for b in at.button if b.key == "adv_autofill_all"][0].click().run()
    assert not errors(at), errors(at)
    conf = at.session_state["adv_confirmed"]
    assert _in_order(conf)
    n = len(plan)
    assert all(conf.get(b) for b in range(n))
    same = sum(1 for b in range(n) if plan[b] and {(p["video_id"], p["scene_id"]) for p in plan[b]}
               == {(p["video_id"], p["scene_id"]) for p in conf[b]})
    assert same >= 0.8 * sum(1 for b in range(n) if plan[b])


def test_choreography_auto_picks_are_the_plan():
    at = _app("Choreography")
    assert any(b.label.startswith("📍 Use the planned picks") for b in at.button)
    assert any("Planned pick" in c.value for c in at.caption)
    [b for b in at.button if b.label.startswith("📍 Use the planned picks")][0].click().run()
    assert not errors(at), errors(at)
    plan = at.session_state["_planned_path"]["picks"]
    [b for b in at.button if b.key == "chor_next"][0].click().run()
    assert not errors(at), errors(at)
    conf = at.session_state["chor_confirmed"]
    assert [(p["video_id"], p["scene_id"], round(p["offset_into_scene_sec"], 1)) for p in conf[0]] == \
        [(p["video_id"], p["scene_id"], round(p["offset_into_scene_sec"], 1)) for p in plan[0]]
    # following it keeps the rest of the plan (no re-plan)
    assert at.session_state["_planned_path"]["picks"] is not None
    assert at.session_state["_planned_path"]["picks"].get(1) == plan.get(1)
    [b for b in at.button if b.key == "chor_autofill_all"][0].click().run()
    assert not errors(at), errors(at)
    conf = at.session_state["chor_confirmed"]
    assert _in_order(conf)
    assert all(conf.get(b) for b in range(len(plan)))
