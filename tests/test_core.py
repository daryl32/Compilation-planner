"""core/ works on its own: no Streamlit, settings passed in, defaults that
agree with the planner page."""

import random
import subprocess
import sys
import textwrap

from tests.support import DATA, PLANNER, REPO, load_page
from tests.fixtures import load_track


def test_core_runs_without_streamlit():
    code = textwrap.dedent(f"""
        import sys, json, types
        sys.modules["streamlit"] = None          # any `import streamlit` now fails
        sys.path.insert(0, {str(REPO)!r})
        import core.segmentation as S, core.autotune as A, core.shape as H, core.footage as F, core.picking as P
        tr = json.load(open({str(DATA / "audio_catalogue" / "song_full.json")!r}))
        cfg = {{"method": S.SEG_SMART, "beats_per_bar": 4, **{{k: S.SEG_DEFAULTS[k] for k in S.SEG_METHOD_KEYS[S.SEG_SMART]}}}}
        segs = S.build_segments(tr, cfg)
        S.apply_segment_intensity(segs, tr, 1.0)
        s, why = A.auto_tune_parameters(tr, 1, 4, method=S.SEG_ADAPTIVE)
        cat = json.load(open({str(DATA / "catalogue" / "vid_a.json")!r}))
        q = F.build_footage_queues({{"vid_a": cat}}, ("vid_a",), (), roles_of=lambda sc: sc.get("roles", []))
        print(len(segs), len(q["vid_a"]), "ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("ok")


def test_seg_defaults_match_page():
    import core.segmentation as S
    dial = load_page(PLANNER)["DIAL_DEFAULTS"]
    for k, v in S.SEG_DEFAULTS.items():
        assert dial[k] == v, k


def test_skip_counts_passed_through():
    import core.segmentation as S
    assert S.allowed_clip_count(2, 1, 4, skip=[2]) == 1          # tie → smaller
    assert S.allowed_clip_count(3, 1, 4, skip=[2, 3]) == 4
    assert S.allowed_clip_count(2, 1, 4) == 2
    ns = load_page(PLANNER, {"clip_count_skip": [2]})
    assert ns["allowed_clip_count"](2, 1, 4) == 1
    assert ns["clips_for_intensity"](0.4, 1, 4, 0.2, 0.85, False, random.Random(0)) == \
        S.clips_for_intensity(0.4, 1, 4, 0.2, 0.85, False, random.Random(0), skip=[2])


def test_skip_budget_uses_clips_per_block():
    import core.picking as P
    assert P.compute_skip_budget(20, 4, 1.0, 2, clips_per_block=1.0) == 20 - (3 + 1)
    assert P.compute_skip_budget(20, 4, 1.0, 2, clips_per_block=2.0) == 20 - (6 + 1)
    assert P.compute_skip_budget(20, 4, 0.0, 2) == 0
    ns = load_page(PLANNER, {"_avg_clips_per_block": 2.0})
    assert ns["compute_skip_budget"](20, 4, 1.0, 2) == 13


def test_position_targets_use_ranges():
    import core.picking as P
    q = {"a": [], "b": []}
    t = P.compute_position_targets(0, 4, {}, q, 1, ranges={"a": (10.0, 30.0)}, duration_of=lambda v: 100.0)
    assert t["a"] == (10.0 + 0.5 * 10.0, 10.0)        # 4 blocks × 1 ÷ 2 videos = 2 parts
    assert t["b"] == (25.0, 50.0)


def test_build_segments_every_method_every_track():
    import core.segmentation as S
    from tests.fixtures import TRACKS
    for tid in TRACKS:
        tr = load_track(DATA, tid)
        for method in S.SEG_METHOD_KEYS:
            cfg = {"method": method, "beats_per_bar": 4, **{k: S.SEG_DEFAULTS[k] for k in S.SEG_METHOD_KEYS[method]}}
            segs = S.build_segments(tr, cfg)
            assert segs, (tid, method)
            assert abs(segs[0]["start"]) < 1e-6
            assert abs(segs[-1]["end"] - tr["duration_sec"]) < 0.05, (tid, method, segs[-1])
            for a, b in zip(segs, segs[1:]):
                assert abs(a["end"] - b["start"]) < 1e-6, (tid, method)
