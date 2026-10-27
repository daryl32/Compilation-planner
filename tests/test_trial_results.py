"""🧪 Trial results: reading the bundle (trial_results.py) and the view on the Training Labels page."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)

import trial_results as TR


def _bundle() -> dict:
    moments = [
        {"key": "g_5.00", "t": 5.0, "scene_id": 0, "frame": "frames/vid_a/g_5.00.jpg",
         "mine": ["close-up shot", "dance"], "old": ["street"],
         "teachers": {"8b": {"tags": ["close-upshot"], "unsure": ["night"], "caption": "Two people."},
                      "32b": {"tags": ["close-up shot", "dance"], "unsure": [], "caption": "A dance."}}},
        {"key": "s1_0", "t": 12.0, "scene_id": 1, "frame": "frames/vid_a/s1_0.jpg", "mine": None,
         "old": ["dance"], "teachers": {"8b": {"tags": ["dance"]}, "32b": {"tags": ["dance", "crowd"]}}},
        {"key": "s2_0", "t": 20.0, "scene_id": 2, "frame": None, "mine": None, "old": [],
         "teachers": {"8b": {"tags": ["night"]}, "32b": {"tags": ["night"]}}},
    ]
    return {"version": 1, "created": "2026-10-27 18:00", "smoke": False,
            "teachers": {"8b": "Qwen/Qwen3-VL-8B-Instruct", "32b": "Qwen/Qwen3-VL-32B-Instruct"},
            "report": {}, "videos": {"vid_a": {
                "duration": 30.0, "cuts_new": [8.1, 16.0], "cuts_old": [10.0],
                "boundaries": [{"t": 8.0, "verdict": "cut", "source": "detected"}],
                "scenes": [{"scene_id": 0, "start": 0.0, "end": 10.0, "level": 0.02, "periodicity": 0.7,
                            "movement_bpm": 120.0, "people": 2, "onsets": [1.0], "old_motion": 4.2}],
                "curve_fps": 2.0, "person_curve": [0.01, None, 0.03] * 20, "old_curve": [3.0, 5.0] * 30,
                "moments": moments}}}


@pytest.fixture
def bundle():
    if TR.RESULTS_DIR.exists():
        shutil.rmtree(TR.RESULTS_DIR)
    (TR.RESULTS_DIR / "frames" / "vid_a").mkdir(parents=True)
    (TR.RESULTS_DIR / "results.json").write_text(json.dumps(_bundle()))
    (TR.RESULTS_DIR / "report.md").write_text("# Modal trial report\n\nHello.")
    from PIL import Image
    for k in ("g_5.00", "s1_0"):
        Image.new("RGB", (64, 36), (90, 120, 160)).save(TR.RESULTS_DIR / "frames" / "vid_a" / f"{k}.jpg")
    TR._CACHE.clear()
    yield _bundle()
    shutil.rmtree(TR.RESULTS_DIR, ignore_errors=True)
    TR._CACHE.clear()


def test_chips_against_your_tags():
    assert TR.chips(["Close-Up Shot", "night"], ["close-up shot", "dance"]) == [
        ("Close-Up Shot", TR.RIGHT), ("night", TR.WRONG), ("dance", TR.MISSED)]
    assert TR.chips(["a"], None) == [("a", TR.NEUTRAL)]


def test_filters_and_agreement():
    ms = _bundle()["videos"]["vid_a"]["moments"]
    assert [m["key"] for m in TR.filter_moments(ms, "mine")] == ["g_5.00"]
    assert [m["key"] for m in TR.filter_moments(ms, "wrong")] == ["g_5.00"]          # 8b missed "dance"
    assert [m["key"] for m in TR.filter_moments(ms, "disagree")] == ["g_5.00", "s1_0"]
    assert [m["key"] for m in TR.filter_moments(ms, "all", "Night")] == ["s2_0"]     # unsure tags don't count
    a = TR.agreement(ms)
    assert a["32b"]["exact"] == 1 and a["32b"]["recall"] == 1.0
    assert a["8b"]["exact"] == 0 and a["8b"]["recall"] == 0.5 and a["8b"]["precision"] == 1.0


def test_normalised():
    assert TR.normalised([0, None, 10]) == [0.0, None, 1.0]
    assert TR.normalised([]) == []


def test_load(bundle):
    assert TR.load()["videos"]["vid_a"]["cuts_new"] == [8.1, 16.0]
    assert "Hello." in TR.report_markdown()


def test_view_without_results():
    from tests.apptest_util import errors, page_app
    shutil.rmtree(TR.RESULTS_DIR, ignore_errors=True)
    TR._CACHE.clear()
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.run()
    assert not errors(at), errors(at)
    assert any("No trial results here yet" in i.value for i in at.info)


@pytest.mark.parametrize("sub", ["🖼️ Compare tags", "🕺 Motion", "📋 Report"])
def test_view(bundle, sub):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = sub
    at.run()
    assert not errors(at), errors(at)
    md = " ".join(m.value for m in at.markdown)
    if sub == "🖼️ Compare tags":
        assert "✓ close-up shot" in md and "+ dance" in md and "night?" in md
        assert any("exactly right on 1 of 1" in c.value for c in at.caption)
        at.session_state["tr_filter"] = "mine"
        at.run()
        assert not errors(at), errors(at)
    elif sub == "📋 Report":
        assert "Hello." in md
