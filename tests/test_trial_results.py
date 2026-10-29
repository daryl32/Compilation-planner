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
            "report": {}, "videos": {"vid_b": {
                "batch": "2", "duration": 20.0, "cuts_new": [], "cuts_old": [], "boundaries": [], "scenes": [],
                "curve_fps": 2.0, "person_curve": [], "old_curve": [], "moments": []},
                "vid_a": {
                "batch": "1", "duration": 30.0,
                "run": {"vision_seconds": 120.0, "vision_timings": {"load_models": 20.0}, "vision_gpu": "A10",
                        "vision_at": 1793100000.0,
                        "tagging": {"32b": {"seconds": 60.0, "load_seconds": 90.0, "gpu": "H100", "prompts": 3,
                                            "unreadable": 0}}}, "cuts_new": [8.1, 16.0], "cuts_old": [10.0],
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
        assert "✓ close-up shot" in md and "+ dance" in md       # 32B right; the old tags missed "dance"
        assert ">8B<" not in md and "✅ 8B is right" not in [b.label for b in at.button]   # only the 32B is shown
        assert "The models disagree" not in at.selectbox(key="tr_filter").options
        assert any("exactly right on 1 of 1" in c.value for c in at.caption)
        at.session_state["tr_filter"] = "mine"
        at.run()
        assert not errors(at), errors(at)
    elif sub == "📋 Report":
        assert "Hello." in md


@pytest.fixture
def no_labels(tmp_path):
    import training_labels as TLB
    from library_common import PENDING_FILE
    saved = {}
    for p in (TLB.LABELS_FILE, PENDING_FILE):
        if p.exists():
            saved[p] = tmp_path / p.name
            shutil.copy2(p, saved[p])
            p.unlink()
    yield
    for p in (TLB.LABELS_FILE, PENDING_FILE):
        p.unlink(missing_ok=True)
        if p in saved:
            shutil.copy2(saved[p], p)


def test_saved_labels_overlay():
    import training_labels as TLB
    vids = _bundle()["videos"]
    mid = TR.moment_id("vid_a", vids["vid_a"]["moments"][1])
    assert mid == TLB.card_id("tags", "vid_a", 12.0)
    out = TR.with_saved_labels(vids, [{"id": mid, "type": "tags", "tags": ["crowd"]}])
    m = out["vid_a"]["moments"][1]
    assert m["mine"] == ["crowd"] and m["gold_id"] == mid
    assert vids["vid_a"]["moments"][1]["mine"] is None          # the original isn't changed


def test_accept_and_correct(bundle, no_labels):
    import training_labels as TLB
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.run()
    # ✅ 32B is right on the frame you hadn't labelled (s1_0, at 12 s)
    accept = [b for b in at.button if b.label == "✅ 32B is right"]
    assert len(accept) == 3
    accept[1].click().run()
    assert not errors(at), errors(at)
    items = TLB.load_labels()
    assert [(i["id"], i["tags"]) for i in items] == [(TLB.card_id("tags", "vid_a", 12.0), ["dance", "crowd"])]
    # ✏️ Correct the first frame (your existing label): drop "dance", save
    next(b for b in at.button if b.label == "✏️ Correct").click().run()
    assert not errors(at), errors(at)
    multi = [p for p in at.button_group if "dance" in (p.options or [])]
    assert multi
    multi[0].set_value([]).run()
    next(b for b in at.button if b.label == "💾 Save my tags").click().run()
    assert not errors(at), errors(at)
    first = next(i for i in TLB.load_labels() if i["t"] == 5.0)
    assert "dance" not in first["tags"]


def test_only_32b_shown_and_order():
    assert TR.shown_teachers({"8b": 1, "32b": 1}) == ["32b"]
    assert TR.shown_teachers({"8b": 1}) == ["8b"]
    items = list(range(20))
    a, b = TR.ordered(items, "random", 1), TR.ordered(items, "random", 2)
    assert sorted(a) == items and a != items and a != b and a == TR.ordered(items, "random", 1)
    assert TR.ordered(items, "video") == items
    ms = _bundle()["videos"]["vid_a"]["moments"]
    assert [m["key"] for m in TR.filter_moments(ms, "todo")] == ["s1_0", "s2_0"]


def test_view_random_order_and_todo(bundle, no_labels):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_order"] = "random"
    at.session_state["tr_filter"] = "todo"
    at.run()
    assert not errors(at), errors(at)
    before = at.session_state["tr_seed"] if "tr_seed" in at.session_state else 0
    next(b for b in at.button if b.label == "🎲 Shuffle again").click().run()
    assert not errors(at), errors(at)
    assert at.session_state["tr_seed"] == before + 1
    # Saving a frame takes it out of "not reviewed by me yet".
    n = len([b for b in at.button if b.label == "✅ 32B is right"])
    next(b for b in at.button if b.label == "✅ 32B is right").click().run()
    assert not errors(at), errors(at)
    assert len([b for b in at.button if b.label == "✅ 32B is right"]) == n - 1


def test_batches_and_scores():
    vids = _bundle()["videos"]
    assert TR.batches(vids) == ["1", "2"]
    assert list(TR.in_batch(vids, "2")) == ["vid_b"] and len(TR.in_batch(vids, None)) == 2
    assert TR.batches({"x": {}}) == ["?"]
    rows = {r["tag"]: r for r in TR.tag_scores(vids["vid_a"]["moments"], "32b")}
    assert rows["close-up shot"]["yours"] == 1 and rows["close-up shot"]["recall"] == 1.0
    assert rows["dance"]["f1"] == 1.0


@pytest.mark.parametrize("batch", [None, "1", "2"])
@pytest.mark.parametrize("sub", ["🖼️ Compare tags", "📊 Scores", "🕺 Motion", "📋 Report"])
def test_view_per_batch(bundle, sub, batch):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = sub
    at.session_state["tr_batch"] = batch
    at.run()
    assert not errors(at), errors(at)
    if sub == "📊 Scores" and batch != "2":
        assert any("exactly right on **1 of 1**" in m.value for m in at.markdown)
    if sub == "📊 Scores" and batch == "2":
        assert any("None of your labelled frames" in i.value for i in at.info)


def test_batch_report():
    import trial_logic as TLG
    rows = TR.batch_report(_bundle()["videos"])
    assert [r["batch"] for r in rows] == ["1", "2", "All"]
    b1 = rows[0]
    assert b1["videos"] == 1 and b1["vision_min"] == 2.0 and b1["tag_min"] == 1.0 and b1["tag_startup_min"] == 1.5
    assert b1["vision_cost"] == round(TLG.stage_cost(120, "A10"), 2)
    assert b1["tag_cost"] == round(TLG.stage_cost(150, "H100"), 2)
    assert b1["exact"] == 1 and b1["labelled"] == 1 and b1["cut_answers"] == 1 and b1["cut_acc_new"] == 1.0
    assert b1["cut_acc_old"] == 0.0 and b1["run"] is not None
    assert rows[-1]["videos"] == 2 and rows[-1]["total_cost"] == b1["total_cost"]


def test_motion_marks_saved_replaced_removed(no_labels):
    import motion_review as MR
    import training_labels as TLB
    MR.apply({"action": "verdict", "t": 8.1, "verdict": "no_cut", "source": "new_cut", "video_id": "vid_a"}, [])
    MR.apply({"action": "verdict", "t": 20.0, "verdict": "cut", "source": "manual", "video_id": "vid_a"},
             TLB.load_labels())
    marks = MR.marks_for("vid_a", TLB.load_labels())
    assert [(m["t"], m["verdict"], m["source"]) for m in marks] == [(8.1, "no_cut", "new_cut"), (20.0, "cut", "manual")]
    # Judging the same cut again replaces your mark instead of adding a second one.
    MR.apply({"action": "verdict", "t": 8.3, "verdict": "cut", "source": "new_cut", "video_id": "vid_a"},
             TLB.load_labels())
    marks = MR.marks_for("vid_a", TLB.load_labels())
    assert len(marks) == 2 and marks[0]["verdict"] == "cut"
    MR.apply({"action": "remove", "t": 20.0, "id": marks[1]["id"], "video_id": "vid_a"}, TLB.load_labels())
    assert len(MR.marks_for("vid_a", TLB.load_labels())) == 1


def test_motion_view_with_preview_copy(bundle, no_labels):
    import subprocess
    from library_common import proxy_path
    from tests.apptest_util import errors, page_app
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    pp = proxy_path("vid_a")
    existed = pp.exists()
    if not existed:
        pp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc=size=160x90:rate=10:duration=3", "-pix_fmt", "yuv420p", str(pp)], check=True)
    try:
        at = page_app("pages/6_Training_Labels.py")
        at.session_state["tl_view"] = "🧪 Trial results"
        at.session_state["tr_sub"] = "🕺 Motion"
        at.session_state["tr_mvid"] = "vid_a"
        at.run()
        assert not errors(at), errors(at)
        assert any("Your marks on this video" in c.value for c in at.caption)
    finally:
        if not existed:
            pp.unlink(missing_ok=True)
