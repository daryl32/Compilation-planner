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


@pytest.fixture(scope="module")
def page_plans():
    """The page run headless: song_full, 3 videos per plan, 🔍 pressed."""
    from tests.apptest_util import errors, page_app
    at = page_app(PAGE, timeout=300)
    at.session_state["rec_track"] = "song_full"
    at.run()
    assert not errors(at), errors(at)
    at.slider(key="rec_n_videos").set_value(3).run()
    [b for b in at.button if b.label.startswith("🔍")][0].click().run()
    assert not errors(at), errors(at)
    return at


def test_page_finds_plans(page_plans):
    at = page_plans
    plans = at.session_state["rec_results"]["song_full"]["plans"]
    assert 1 <= len(plans) <= 5
    overall = [p["scores"]["overall"] for p in plans]
    assert overall == sorted(overall, reverse=True)
    for p in plans:
        assert len(p["videos"]) == 3 and set(p["videos"]) <= set(VIDEOS)
        sc = p["scores"]
        for k in ("overall", "shape", "intensity"):
            assert 0 <= sc[k] <= 100, (k, sc[k])
        assert len(sc["per_block"]) == len(p["picks"])
    assert sum(b.label == "🎬 Open" for b in at.button) == len(plans)


def test_results_kept_on_disk(page_plans):
    files = list((DATA / ".curve_cache").parent.glob("recommendations/*.json")) or \
        list(DATA.glob("**/recommendations/*.json"))
    assert files, "no saved recommendations"
    data = json.loads(files[0].read_text())
    assert data["track_id"] == "song_full" and data["plans"]


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


def test_open_as_project_exports_the_recommended_clips(page_plans):
    """🎬 Open as project → the planner, Auto mode → 💾 Save plan: the exported blocks
    are exactly the recommended plan's picks, block for block — for the best plan and
    for the best plan of each other cutting variant."""
    at = page_plans
    plans = at.session_state["rec_results"]["song_full"]["plans"]
    seen, todo = set(), []
    for rank, p in enumerate(plans):
        if p["variant"] not in seen:
            seen.add(p["variant"])
            todo.append(rank)
    for rank in todo:
        at.session_state["_open_recommended_plan"] = None
        at.run()                                # a full page again (the last run stopped at switch_page)
        buttons = [b for b in at.button if b.label == "🎬 Open"]
        assert len(buttons) == len(plans)
        buttons[rank].click().run()             # AppTest can't follow st.switch_page; the hand-off is set
        opened = at.session_state["_open_recommended_plan"]
        assert opened["track_id"] == "song_full" and opened["videos"] == plans[rank]["videos"]
        assert _open_and_export(opened) == plans[rank]["picks"], plans[rank]["variant"]
