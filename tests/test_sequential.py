"""In order, planned per video (core/sequential.py): every video's clips in time order, the whole
plan filled, Swap / Reject respected, F walking further through each video, role clips in their parts."""

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.fixtures import load_track


def _ctx(planned=True):
    import rec_library as RL
    c = RL.plan_context(None)
    c.planned_sequential = planned
    return c


def _match(track_id="song_full", videos=("vid_a", "vid_b", "vid_c", "vid_f"), F=1.0, max_clips=1, excl=None,
           rejected=(), roles=None, exclusive=False, planned=True):
    import catalogue_store as CStore
    import core.footage as F_
    import core.matching as M
    import core.recommend as REC
    import rec_library as RL
    import scene_roles as SR
    tr = load_track(DATA, track_id)
    segs = REC.cut_variant(tr, {"sm_pace": 0})
    q = F_.build_footage_queues(CStore.all_catalogues(), tuple(videos), (), {}, roles_of=SR.scene_roles)
    ctx = _ctx(planned)
    if roles:
        used = frozenset(r for r in roles if r)
        ctx.reserved_roles = ctx.locked_roles = used
        ctx.story_in_use = True
        ctx.intro_outro_exclusive = exclusive
    tl, csf, dsf = M.match_scenes_to_track(
        [dict(s) for s in segs], q, 1, max_clips, 0.0, 1.0, False, True, excl or {}, {}, set(rejected), False, 0, 1.0,
        set(), set(), 2.0, 0.5, True, 1.5, 2.0, 42, weight_position=F, repeat_fallbacks=set(),
        block_roles=roles or [None] * len(segs), track=tr, ctx=ctx)
    return tl, segs, dsf


@pytest.mark.parametrize("track_id,max_clips", [("song_full", 1), ("song_full", 3), ("song_noanalysis", 2),
                                                ("song_short", 1)])
def test_in_order_and_filled(track_id, max_clips):
    from core.sequential import in_order
    tl, segs, dsf = _match(track_id, max_clips=max_clips)
    assert in_order(tl)
    assert dsf <= 0.5
    for e, sg in zip(tl, segs):
        for s in e["scenes"]:
            assert abs(sum(c["clip_duration_sec"] for c in s["chain"]) - (sg["end"] - sg["start"])) < 0.05


def test_swap_and_reject_are_respected():
    tl, _, _ = _match()
    first = tl[0]["scenes"][0]["chain"][0]
    tl2, _, _ = _match(excl={0: {(first["video_id"], first["scene_id"])}})
    assert (tl2[0]["scenes"][0]["chain"][0]["video_id"], tl2[0]["scenes"][0]["chain"][0]["scene_id"]) != \
        (first["video_id"], first["scene_id"])
    tl3, _, _ = _match(rejected={(first["video_id"], first["scene_id"])})
    assert all((c["video_id"], c["scene_id"]) != (first["video_id"], first["scene_id"])
               for e in tl3 for s in e["scenes"] for c in s["chain"])


def test_follow_the_video_walks_further():
    """Higher F: each video's clips spread further through it."""
    import catalogue_store as CStore
    import core.footage as F_
    cats = CStore.all_catalogues()

    def reach(F):
        tl, _, _ = _match(F=F)
        last = {}
        for e in tl:
            for s in e["scenes"]:
                for c in s["chain"]:
                    last[c["video_id"]] = c["clip_start_sec"] + c["clip_duration_sec"]
        return sum(last[v] / F_._tc_to_seconds(cats[v]["scenes"][-1]["end_tc"]) for v in last) / len(last)
    assert reach(5.0) >= reach(0.0)
    assert reach(1.0) > 0.85


def test_role_clips_in_their_parts():
    _, segs, _ = _match()
    roles = ["action" if 0.3 < i / len(segs) < 0.5 else None for i in range(len(segs))]
    tl, _, _ = _match(roles=roles, exclusive=True, videos=("vid_a", "vid_c", "vid_f"))
    import catalogue_store as CStore
    import scene_roles as SR
    cats = CStore.all_catalogues()
    role_of = {(v, s["scene_id"]): SR.scene_roles(s) for v, c in cats.items() for s in c["scenes"]}
    hits = [any("action" in role_of[(c["video_id"], c["scene_id"])] for c in e["scenes"][0]["chain"])
            for e, r in zip(tl, roles) if r]
    assert sum(hits) >= len(hits) // 2


def test_classic_engine_still_available():
    """The old block-by-block engine (planned_sequential False) still runs — the golden snapshots of the
    matcher's own tests use it."""
    from core.sequential import in_order
    tl, _, _ = _match(planned=False)
    assert in_order(tl)
