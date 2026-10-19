"""🔮 Recommendations: core/recommend.py on its own, the page headless, and —
the promise the page makes — a plan opened as a project exports exactly the
clips the recommendation scored."""

import json
import subprocess
import sys
import textwrap

import pytest

from tests.support import DATA, REPO
from tests.fixtures import VIDEOS, load_track

PAGE = "pages/5_Recommendations.py"
PLANNER_PAGE = "pages/1_Compilation_Planner.py"


# ---------------------------------------------------------------------------
# core/recommend.py
# ---------------------------------------------------------------------------

def test_recommend_runs_without_streamlit():
    code = textwrap.dedent(f"""
        import sys
        sys.modules["streamlit"] = None
        sys.path.insert(0, {str(REPO)!r})
        import core.recommend as R
        print("ok", len(R.VARIANTS))
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok")


def _smart_segments(track, pace=0):
    import core.recommend as R
    return R.cut_variant(track, {"sm_pace": pace})


def test_block_weights_count_drops_more():
    import core.recommend as R
    tr = load_track(DATA, "song_full")
    segs = _smart_segments(tr)
    w = R.block_weights(tr, segs)
    assert len(w) == len(segs)
    assert set(w) <= {1.0, R.DROP_WEIGHT}
    assert R.DROP_WEIGHT in w and 1.0 in w
    # a story map's action parts count extra too, on top of the drops
    end = segs[1]["end"] - 0.01
    w2 = R.block_weights(tr, segs, story_regions=[{"start": 0.0, "end": end, "role": "action"}])
    assert w2[0] == R.DROP_WEIGHT and w2[1] == R.DROP_WEIGHT
    assert w2[2:] == w[2:]
    # no drops in the analysis and no story map: the most intense quarter
    plain = dict(tr, analysis={k: v for k, v in (tr.get("analysis") or {}).items() if k != "drops"})
    w3 = R.block_weights(plain, segs)
    assert 0 < w3.count(R.DROP_WEIGHT) <= len(segs) // 4 + 2


def test_choose_sets_distinct_and_sized():
    import numpy as np
    import core.recommend as R
    rng = np.random.default_rng(0)
    vids = [f"v{i}" for i in range(7)]
    S = rng.random((7, 10))
    sets = R.choose_sets(vids, S, [1.0] * 10, 3, n_plans=4)
    groups = [s["videos"] for s in sets]
    assert groups and all(len(g) == 3 and len(set(g)) == 3 for g in groups)
    assert len({tuple(sorted(g)) for g in groups}) == len(groups)
    est = [s["estimate"] for s in sets]
    assert est[0] == max(est)


ACTORS = {"vid_a": ["Ana"], "vid_b": ["Ben"], "vid_c": ["Ana"], "vid_d": ["Cat"], "vid_f": ["Ben"]}


@pytest.fixture(scope="module")
def actors_tagged():
    """Actors on the fixture videos (🎬 Video details) for 🥊 Verses; removed afterwards."""
    from library_common import LIBRARY_META_FILE, load_library_meta, save_library_meta
    before = LIBRARY_META_FILE.read_text() if LIBRARY_META_FILE.exists() else None
    meta = load_library_meta()
    for vid, names in ACTORS.items():
        meta.setdefault(vid, {})["actors"] = names
    save_library_meta(meta)
    yield ACTORS
    if before is None:
        LIBRARY_META_FILE.unlink(missing_ok=True)
    else:
        LIBRARY_META_FILE.write_text(before)


def _run_page(sequential: bool):
    from tests.apptest_util import errors, page_app
    at = page_app(PAGE, timeout=600)
    at.session_state["rec_track"] = "song_full"
    at.run()
    assert not errors(at), errors(at)
    assert at.slider(key="rec_n_videos").value == 4          # the default
    at.slider(key="rec_n_videos").set_value(3).run()
    if sequential:
        at.toggle(key="rec_sequential").set_value(True).run()
    [b for b in at.button if b.label.startswith("🔍")][0].click().run()
    assert not errors(at), errors(at)
    return at


@pytest.fixture(scope="module")
def page_plans(actors_tagged):
    """The page run headless: song_full, 3 videos per plan, 🔍 pressed."""
    return _run_page(False)


def _data(at, sequential=False):
    return at.session_state["rec_results"][("song_full", 3, sequential)]


def _results(at, sequential=False):
    return _data(at, sequential)["results"]


def test_page_finds_five_plans(page_plans):
    at = page_plans
    results = _results(at)
    assert [r["mode"] for r in results] == ["shape", "story", "overall", "verses", "wildcard"]
    plans = {r["mode"]: r["plan"] for r in results}
    assert all(plans.values()), {m: r["note"] for m, r in zip(plans, results)}
    for mode, p in plans.items():
        assert set(p["videos"]) <= set(VIDEOS)
        assert len(p["videos"]) == (2 if mode == "verses" else 3), (mode, p["videos"])
        sc = p["scores"]
        for k in ("overall", "shape", "intensity"):
            assert 0 <= sc[k] <= 100, (mode, k, sc[k])
        assert len(sc["per_block"]) == len(p["picks"])
        assert p["sequential"] is False
    # the shape-fit plan is matched without the story map; the story ones with it
    assert plans["shape"]["story"] is False
    assert plans["story"]["story"] and plans["overall"]["story"]
    assert plans["story"]["scores"]["story"] is not None
    # each of the top three wins on its own measure
    assert plans["story"]["scores"]["story"] > 0
    assert plans["story"]["scores"]["story"] > plans["shape"]["scores"]["story"]
    assert plans["shape"]["scores"]["shape"] >= max(plans[m]["scores"]["shape"] for m in ("story", "overall"))
    # modes 1–3 are different plans
    keys = {(p["style"], p["pace"], tuple(p["videos"])) for m, p in plans.items() if m in ("shape", "story", "overall")}
    assert len(keys) == 3
    # the top three all use the same Smart cut
    assert {(plans[m]["style"], plans[m]["pace"]) for m in ("shape", "story", "overall")} == {("smart", 0)}
    for p in plans.values():
        sc = p["scores"]
        assert len(sc["per_block_shape"]) == len(sc["per_block_story"]) == len(sc["per_block"])
    # 🥊 Verses: different actors, a cutting style none of the top three use
    v = plans["verses"]
    a, b = (set(x.lower() for x in ACTORS[vid]) for vid in v["videos"])
    assert not (a & b), v["videos"]
    assert v["style"] not in {plans[m]["style"] for m in ("shape", "story", "overall")}
    assert sum(1 for b in at.button if (b.key or "").startswith("open_song_full_")) == 5


def test_results_kept_on_disk(page_plans):
    import rec_library as RL
    data = RL.load_results("song_full", 3, False)
    assert data and len(data["results"]) == 5 and data["insights"]["videos"]


def test_wildcard_rolls_again(page_plans):
    at = page_plans
    before = [r["plan"] for r in _results(at)]
    seed = _data(at)["seed"]
    [b for b in at.button if b.label == "🎲 Roll again"][0].click().run()
    from tests.apptest_util import errors
    assert not errors(at), errors(at)
    after = [r["plan"] for r in _results(at)]
    assert _data(at)["seed"] == seed + 1
    assert after[:4] == before[:4]                         # only the wildcard changes
    assert after[4]["mode"] == "wildcard"


def _open_and_export(opened: dict) -> list:
    from tests.apptest_util import errors, page_app
    from tests.page_scenarios import _clean_disk
    _clean_disk()
    pl = page_app(PLANNER_PAGE, timeout=300)
    pl.session_state["_open_recommended_plan"] = opened
    pl.run()
    assert not errors(pl), errors(pl)
    assert pl.session_state["ui_section"] == "🎬 3. Matching & Export"
    assert pl.session_state["matching_mode"] == "Auto"
    assert pl.session_state["sequential_mode"] == opened["sequential"]
    assert pl.session_state["use_story_map"] == opened["story"]
    assert sorted(pl.session_state["committed_selected_videos"]) == sorted(opened["videos"])
    assert any("Opened Recommended plan" in s.value for s in pl.success)
    assert any(b.label == "↩️ Bring back what I had" for b in pl.button)
    save = [b for b in pl.button if b.label == "💾 Save plan"]
    assert save, "no 💾 Save plan button"
    save[0].click().run()
    assert not errors(pl), errors(pl)
    plan = json.loads((DATA / "compilation_plans" / f"{opened['track_id']}_plan.json").read_text())
    return [[[c["video_id"], c["scene_id"], round(float(c["clip_start_sec"]), 3)] for c in b["scenes"]]
            for b in plan["timeline"]]


def _check_opens(at, sequential=False, modes=None):
    results = _results(at, sequential)
    for rank, r in enumerate(results):
        if not r["plan"] or (modes and r["mode"] not in modes):
            continue
        at.session_state["_open_recommended_plan"] = None
        at.run()                                # a full page again (the last run stopped at switch_page)
        [b for b in at.button if b.key == f"open_song_full_{r['mode']}"][0].click().run()
        opened = at.session_state["_open_recommended_plan"]   # AppTest can't follow st.switch_page
        assert opened["track_id"] == "song_full" and opened["videos"] == r["plan"]["videos"]
        assert _open_and_export(opened) == r["plan"]["picks"], (r["mode"], r["plan"]["variant"])


def test_open_as_project_exports_the_recommended_clips(page_plans):
    """🎬 Open as project → the planner, Auto mode → 💾 Save plan: the exported blocks
    are exactly the recommended plan's picks, block for block — for all five plans
    (different cutting styles, story map on and off)."""
    _check_opens(page_plans)


def test_sequential_mode(actors_tagged):
    """Sequential mode: plans are matched in order and open in order, still exactly as scored."""
    at = _run_page(True)
    results = _results(at, True)
    assert all(r["plan"]["sequential"] for r in results if r["plan"])
    _check_opens(at, True, modes={"shape", "overall"})
