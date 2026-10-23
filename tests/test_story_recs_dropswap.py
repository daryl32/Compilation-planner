"""🎭 Per-part pace + Story-led style, 📏 calibrated match %, 🎛️ tuning the cutting for a set of
videos, and 💥 Mix drop swaps."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.fixtures import load_track, TRACKS


# ---------------------------------------------------------------------------
# 🎭 Per-part pace and Story-led
# ---------------------------------------------------------------------------

ROLE_CUTS = {"intro": {"pace": 0, "style": "cinematic"}, "slow": {"pace": -1, "style": "cinematic"},
             "action": {"pace": 1, "style": "hyper"}, "outro": {"pace": 0, "style": "cinematic"}}


def _regions(tr):
    import story_map as STM
    return STM.suggested_map(tr)


def _avg(segs, a, b):
    x = [s["end"] - s["start"] for s in segs if a <= s["start"] < b]
    return sum(x) / len(x) if x else None


def test_part_plans_and_story_led_cuts():
    import core.story_cuts as STC
    from core.segmentation import build_segments
    from core.styles import style_settings
    tr = load_track(DATA, "song_full")
    regions = _regions(tr)
    assert any(r["role"] == "action" for r in regions)
    # Story-led: each role in its own style; touching parts cut the same way are merged.
    parts = STC.part_plans(tr, regions, ROLE_CUTS, "story", 0)
    assert {(p["role"], p["style"], p["pace"]) for p in parts} >= {("action", "hyper", 1), ("slow", "cinematic", -1)}
    m, s, notes = style_settings(tr, "story", 0)
    assert m == "Smart (self-tuning)" and "🎭 Story-led" in notes[0]
    segs = build_segments(tr, {"method": m, "beats_per_bar": 4, **s, "parts": parts})
    base = build_segments(tr, {"method": m, "beats_per_bar": 4, **s})
    dur = float(tr["duration_sec"])
    assert segs[0]["start"] == 0 and segs[-1]["end"] == pytest.approx(dur, abs=1e-3)
    assert all(a["end"] == pytest.approx(b["start"]) for a, b in zip(segs, segs[1:]))
    act = next(p for p in parts if p["role"] == "action")
    slow = next(p for p in parts if p["role"] == "slow")
    assert _avg(segs, act["start"], act["end"]) < _avg(base, act["start"], act["end"])     # Hyper, busier
    assert _avg(segs, slow["start"], slow["end"]) > _avg(base, slow["start"], slow["end"])  # Cinematic, calmer
    # A cut on every part edge, so no clip straddles two parts.
    cuts = [x["start"] for x in segs]
    for p in parts:
        if 0 < p["start"] < dur - 1:
            assert any(abs(c - p["start"]) < STC.EDGE_GAP + 0.01 for c in cuts), p


def test_pace_per_part_keeps_the_style_and_nothing_changes_without_a_map():
    import core.story_cuts as STC
    tr = load_track(DATA, "song_full")
    regions = _regions(tr)
    parts = STC.part_plans(tr, regions, ROLE_CUTS, "cinematic", 0)
    assert parts and all(p["style"] == "cinematic" for p in parts)
    assert {p["role"] for p in parts} == {"slow", "action"}          # intro / outro pace 0 = the base
    assert STC.part_plans(tr, None, ROLE_CUTS, "story", 0) == []
    assert STC.part_plans(tr, regions, ROLE_CUTS, "custom", 0) == []
    # Pace clamps at the ends of the scale.
    assert STC.part_choice("beat", 2, {"pace": 1})[1] == 2
    lines = STC.describe(regions, ROLE_CUTS, "story", 0, {"action": "💥 Action"})
    assert any(line.startswith("💥 Action → 🕺 Hyper") for line in lines)


def test_role_cuts_saved_with_the_roles(tmp_path):
    import scene_roles as SR
    had = SR.ROLES_FILE.exists()
    b = tmp_path / "bak.json"
    if had:
        shutil.copy2(SR.ROLES_FILE, b)
    try:
        assert SR.role_cuts()["action"] == {"pace": 1, "style": "hyper"}       # default
        SR.set_role_cut("action", 2, "beat")
        rid = SR.add_role("Reveal", "✨", "#123456", {"energy": "high"})
        assert SR.role_cuts()["action"] == {"pace": 2, "style": "beat"}
        assert SR.role_cuts()[rid] == {"pace": 0, "style": None}
        SR.save_motion_map({"enabled": True})
        assert SR.role_cuts()["action"]["style"] == "beat"                  # other keys kept
    finally:
        if had:
            shutil.copy2(b, SR.ROLES_FILE)
        else:
            SR.ROLES_FILE.unlink(missing_ok=True)
        SR._MAP_CACHE.clear()


def test_story_led_not_a_random_pick_in_recommendations():
    import core.recommend as REC
    import inspect
    src = inspect.getsource(REC)
    assert src.count('k != "story"') >= 2


def test_planner_story_led_row():
    from tests.apptest_util import errors, page_app
    at = page_app("pages/1_Compilation_Planner.py", timeout=300)
    at.run()
    assert not errors(at), errors(at)
    if at.session_state["track_id"] is None:
        pytest.skip("no track picked")
    at.selectbox(key="cut_style").set_value("story").run()
    assert not errors(at), errors(at)
    assert any(c.value.startswith("Parts: ") or "Story-led cuts it all" in c.value for c in at.caption)
    at.selectbox(key="cut_style").set_value("beat").run()
    assert not errors(at), errors(at)
    assert at.toggle(key="story_part_pace") is not None


# ---------------------------------------------------------------------------
# 📏 Calibrated %
# ---------------------------------------------------------------------------

def _ref(n=200):
    import random
    rnd = random.Random(3)
    return [{"overall": rnd.uniform(30, 70), "shape": rnd.uniform(10, 60), "intensity": rnd.uniform(50, 90),
             "story": rnd.uniform(0, 100), "action": rnd.uniform(40, 90), "action_story": None} for _ in range(n)]


def test_ranked_percent():
    import core.calibrate as CAL
    ref = _ref()
    cal = CAL.build(ref)
    assert CAL.ready(cal) and cal["n_rated"] == 0
    assert cal["weights"] == pytest.approx({p: 1 / 3 for p in CAL.PARTS})
    top = CAL.ranked({"overall": 70, "shape": 60, "intensity": 90, "story": 100, "action": 90}, cal)
    low = CAL.ranked({"overall": 30, "shape": 10, "intensity": 50, "story": 0, "action": 40}, cal)
    assert top["ranked"] and top["shape"] > 95 and top["overall"] > 95
    assert low["shape"] < 5 and low["overall"] < 5
    assert top["raw"]["shape"] == 60
    assert CAL.ranked({"shape": 40}, CAL.build(ref[:5]))["shape"] == 40      # too little to rank against
    assert "Raw scores" in CAL.describe(CAL.build(ref[:5]))


def test_ratings_tune_the_overall_weights():
    import core.calibrate as CAL
    ref = _ref()
    # Ratings that follow shape only.
    rated = [({"shape": s, "intensity": 70.0, "story": 50.0}, 1 + int(s // 12.5)) for s in range(5, 60, 4)]
    cal = CAL.build(ref, rated)
    assert cal["n_rated"] == len(rated) >= CAL.MIN_RATED
    assert cal["weights"]["shape"] > cal["weights"]["intensity"]
    assert sum(cal["weights"].values()) == pytest.approx(1.0)
    assert "tuned by" in CAL.describe(cal)


def test_calibration_reads_stored_searches(tmp_path, monkeypatch):
    import rec_library as RL
    monkeypatch.setattr(RL, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(RL, "SEEN_FILE", tmp_path / "seen_videos.json")
    RL._CAL_CACHE.clear()
    (tmp_path / "a.json").write_text(json.dumps({"results": [], "insights": {"scored": _ref(40)}}))
    (tmp_path / "b.json").write_text(json.dumps({"results": [{"mode": "overall", "plan": {"scores": _ref(1)[0]}}]}))
    (tmp_path / "seen_videos.json").write_text("{}")
    assert len(RL.reference_scores()) == 41
    cal = RL.calibration()
    assert cal["n"] >= 41
    RL._CAL_CACHE.clear()


# ---------------------------------------------------------------------------
# 🎛️ Tune
# ---------------------------------------------------------------------------

def test_tune_settings_ranks_ways_of_cutting():
    import rec_library as RL
    vids = list(RL.library_prints())[:4]
    res = RL.tune_settings("song_full", vids, checks=3)
    assert res["tried"] > 10 and 3 <= len(res["results"]) <= 4
    assert any(r["style"] == "story" for r in res["results"])              # 🎭 always in the running
    overall = [r["ranked"]["overall"] for r in res["results"]]
    assert overall == sorted(overall, reverse=True)
    for r in res["results"]:
        assert {"overall", "shape", "intensity"} <= set(r["scores"])


def test_open_a_tuned_plan_keeps_its_style():
    import streamlit as st
    from planner import projects as P
    st.session_state.clear()
    plan = {"track_id": "song_full", "settings": {"segmentation_method": "Smart (self-tuning)", "sm_pace": 1,
                                                    "parts": [{"x": 1}]},
            "videos": ["vid_a"], "story": False, "story_regions": None, "sequential": False,
            "cut_style": "story", "cut_style_pace": 1, "label": "t"}
    try:
        P._open_recommended_plan(plan)
    except Exception as e:   # outside a running page some session pieces aren't there
        pytest.skip(f"needs the page: {e}")
    assert st.session_state["cut_style"] == "story" and st.session_state["cut_style_pace"] == 1
    assert "parts" not in st.session_state


# ---------------------------------------------------------------------------
# 💥 Drop swaps
# ---------------------------------------------------------------------------

def _track(tid, bpm, dur, builds, drops, key=None):
    beat = 60.0 / bpm
    beats = [round(i * beat, 3) for i in range(int(dur / beat))]
    return {"track_id": tid, "duration_sec": dur, "bpm": bpm, "beat_times": beats, "key": key,
            "analysis": {"beats_per_bar": 4, "downbeats": beats[::4], "builds": builds, "drops": drops}}


def test_drop_swap_plan_and_suggestions():
    import core.mixing as MX
    a = _track("a", 120, 120, [{"start": 40.0, "end": 56.0}, {"start": 80.0, "end": 84.0}], [56.0, 84.0])
    b = _track("b", 124, 120, [], [32.0, 64.0])
    p = MX.drop_swap_plan(a, b)
    bar = 2.0
    assert p["a"][1] == pytest.approx(56.0, abs=0.01)                       # A ends on its drop
    assert p["a"][0] == pytest.approx(56.0 - 16 - MX.DROP_LEAD_BARS * bar, abs=bar)   # lead-in before the build
    bb = 4 * 60 / 124
    assert p["b"][0] == pytest.approx(32.0, abs=bb)                          # B starts on its drop
    assert p["b"][1] - p["b"][0] == pytest.approx(MX.DROP_AFTER_BARS * bb, abs=bb)
    assert MX.drop_swap_plan(b, a) is None                                    # b has no build-up
    infos = [{"track_id": "a", "bpm": 120, "key": None, "energy": 0.5, "builds": 2, "drops": 2},
             {"track_id": "b", "bpm": 124, "key": None, "energy": 0.5, "builds": 0, "drops": 2},
             {"track_id": "c", "bpm": 90, "key": None, "energy": 0.5, "builds": 0, "drops": 0}]
    sug = MX.suggest_drop_swaps(infos)
    assert [(s["a"], s["b"]) for s in sug] == [("a", "b")]


def test_mix_page_drop_swap(tmp_path):
    """Two of the test library's tracks given a build-up and drops: 💥 Drop swaps → 🧩 Start it fills the
    mashup editor with A's build-up and B's drop joined by a bass swap."""
    import mix_ui
    from tests.apptest_util import errors, page_app
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎚️ Mix"
    at.session_state["mx_mode"] = "💡 Suggestions"
    real_full, real_infos = mix_ui.full_track, mix_ui._infos

    def full(tid):
        t = real_full(tid)
        if t is None:
            return None
        t = dict(t)
        dur = float(t["duration_sec"])
        t["analysis"] = dict(t.get("analysis") or {}, builds=[{"start": dur * 0.2, "end": dur * 0.4}],
                             drops=[dur * 0.4, dur * 0.6])
        return t

    def infos(summaries):
        return [dict(i, builds=1, drops=2) for i in real_infos(summaries)]

    mp = pytest.MonkeyPatch()
    mp.setattr(mix_ui, "full_track", full)
    mp.setattr(mix_ui, "_infos", infos)
    try:
        at.run()
        assert not errors(at), errors(at)
        go = [b for b in at.button if (b.key or "").startswith("mx_sug_go_d_")]
        if not go:
            pytest.skip("fewer than two mixable tracks in the test library")
        go[0].click().run()
        assert not errors(at), errors(at)
        assert at.session_state["mx_mode"] == "🧩 Mashup"
        parts = at.session_state["mxm_parts"]
        assert len(parts) == 2 and parts[0]["track"] != parts[1]["track"]
        assert parts[1]["xfade"]["bass_swap"] is True
    finally:
        mp.undo()
