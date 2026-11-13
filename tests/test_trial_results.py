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
    assert [m["key"] for m in TR.filter_moments(ms, "all", "street")] == ["g_5.00"]   # only the old tags have it
    assert TR.filter_moments(ms, "all", "street", skip_old=True) == []                # 🙈 Ignore old tags
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


@pytest.mark.parametrize("sub", ["🖼️ Compare tags", "🕺 Motion", "🗄️ Teacher"])
def test_view(bundle, sub):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = sub
    at.run()
    assert not errors(at), errors(at)
    md = " ".join(m.value for m in at.markdown)
    if sub == "🖼️ Compare tags":
        assert "✓ close-up shot" in md and "+ dance" not in md    # 🙈 Ignore old tags is on: no Old row
        at.toggle(key="tr_no_old").set_value(False).run()
        assert not errors(at), errors(at)
        md = " ".join(m.value for m in at.markdown)
        assert "✓ close-up shot" in md and "+ dance" in md       # 32B right; the old tags missed "dance"
        assert ">8B<" not in md and "✅ 8B is right" not in [b.label for b in at.button]   # only the 32B is shown
        assert "The models disagree" not in at.selectbox(key="tr_filter").options
        assert any("exactly right on 1 of 1" in c.value for c in at.caption)
        at.session_state["tr_filter"] = "mine"
        at.run()
        assert not errors(at), errors(at)
        at.session_state["tr_filter"] = "all"
        at.session_state["tr_tag"] = "dance"
        at.session_state["tr_by"] = "both"
        at.run()
        assert not errors(at), errors(at)
        assert not at.info                                        # you and the 32B both picked "dance"
        assert at.session_state["tr_by"] == "both"
    elif sub == "🗄️ Teacher":
        assert "Hello." in md                                     # the written report, in its fold-out


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
    at.session_state["tr_sub"] = "🖼️ Compare tags"          # (📋 Summary opens first)
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
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_set"] = "train"                 # All would mix in 📚 the library's other videos
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


def test_teacher_label_and_batch_scorecard_fields():
    assert TR.teacher_label("Qwen/Qwen3-VL-32B-Instruct-FP8") == "32B FP8"
    assert TR.teacher_label("Qwen/Qwen3-VL-32B-Instruct") == "32B"
    assert TR.teacher_label("Qwen/Qwen3-VL-8B-Instruct") == "8B"
    videos = {"a": {"batch": "6", "duration": 3600, "run": {"tagging": {"32b": {
                  "model": "Qwen/Qwen3-VL-32B-Instruct-FP8", "examples_used": 30, "seconds": 60, "gpu": "H100"}}},
                    "moments": [{"mine": ["x"], "teachers": {"32b": {"tags": ["x", "y"]}}}]},
              "b": {"batch": "5", "duration": 3600, "run": {"tagging": {"32b": {
                  "model": "Qwen/Qwen3-VL-32B-Instruct", "examples_used": 0, "seconds": 60, "gpu": "H100"}}},
                    "moments": [{"mine": ["x"], "teachers": {"32b": {"tags": ["x"]}}}]}}
    rows = {r["batch"]: r for r in TR.batch_report(videos, "32b")}
    assert rows["6"]["teacher"] == "32B FP8" and rows["6"]["examples"] == "on"
    assert rows["5"]["teacher"] == "32B" and rows["5"]["examples"] == "off" and rows["5"]["f1"] == 1.0
    assert rows["6"]["f1"] == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert rows["All"]["teacher"] == "32B + 32B FP8" and rows["All"]["examples"] == "mixed"


def test_results_opens_first_teacher_has_the_batches(bundle, no_labels):
    """📊 Results opens first; the teacher's batch fold-outs are in the 🗄️ Teacher archive."""
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.run()
    assert not errors(at), errors(at)
    assert at.session_state["tr_sub"] == "📊 Results"
    at.radio(key="tr_sub").set_value("🗄️ Teacher").run()
    assert not errors(at), errors(at)
    labels = [e.label for e in at.expander]
    for want in ("💷 Cost and time per batch", "🔎 Tagging detail per batch", "✂️ Scene changes per batch",
                 "📄 Full written report (as at the last export)"):
        assert want in labels


def test_fp8_only_batch_shows_one_named_row(no_labels):
    """Batch 7 case: the results have 32b and 32b-fp8 (the FP8 check video), but a video tagged only by
    FP8 is saved under '32b' — its cards show one row named 32B FP8, not a '32B FP8: not tagged' row."""
    from tests.apptest_util import errors, page_app
    data = _bundle()
    data["teachers"] = {"32b": "Qwen3-VL-32B-Instruct", "32b-fp8": "Qwen3-VL-32B-Instruct-FP8"}
    va = data["videos"]["vid_a"]
    va["run"]["tagging"] = {"32b": {"model": "Qwen/Qwen3-VL-32B-Instruct-FP8", "seconds": 10.0}}
    for m in va["moments"]:
        m["teachers"] = {"32b": m["teachers"]["32b"]}
    if TR.RESULTS_DIR.exists():
        shutil.rmtree(TR.RESULTS_DIR)
    (TR.RESULTS_DIR / "frames" / "vid_a").mkdir(parents=True)
    (TR.RESULTS_DIR / "results.json").write_text(json.dumps(data))
    TR._CACHE.clear()
    try:
        at = page_app("pages/6_Training_Labels.py")
        at.session_state["tl_view"] = "🧪 Trial results"
        at.session_state["tr_sub"] = "🖼️ Compare tags"
        at.run()
        assert not errors(at), errors(at)
        html = " ".join(m.value for m in at.markdown)
        assert "not tagged" not in html and "32B FP8" in html
        assert [b.label for b in at.button if b.label.startswith("✅")][:1] == ["✅ 32B FP8 is right"]
    finally:
        shutil.rmtree(TR.RESULTS_DIR, ignore_errors=True)
        TR._CACHE.clear()


def test_editor_warns_on_never_with(bundle, no_labels, monkeypatch):
    import training_labels as TLB
    from tests.apptest_util import errors, page_app
    backup = TLB.VOCAB_FILE.read_bytes() if TLB.VOCAB_FILE.exists() else None
    rows = TLB.parse_vocabulary("tag,group,never_with\nclose-up shot,shot,\ndance,action,close-up shot\n"
                                "crowd,people,\nnight,light,\n")
    TLB.save_vocabulary(rows)
    try:
        at = page_app("pages/6_Training_Labels.py")
        at.session_state["tl_view"] = "🧪 Trial results"
        at.session_state["tr_sub"] = "🖼️ Compare tags"
        at.run()
        next(b for b in at.button if b.label == "✏️ Correct").click().run()   # your tags: close-up + dance
        assert not errors(at), errors(at)
        assert any("never-together" in w.value for w in at.warning)
    finally:
        if backup is None:
            TLB.VOCAB_FILE.unlink(missing_ok=True)
        else:
            TLB.VOCAB_FILE.write_bytes(backup)


def test_rules_panel_fixes_and_filters(bundle, no_labels):
    """Your saved label on vid_a at 5 s is close-up shot + dance. Rules: dance → always with crowd;
    night never with dance. The panel adds crowd in one go; a frame breaking never-with is found by
    the ⚠️ filter."""
    import training_labels as TLB
    from tests.apptest_util import errors, page_app
    backup = TLB.VOCAB_FILE.read_bytes() if TLB.VOCAB_FILE.exists() else None
    TLB.save_vocabulary(TLB.parse_vocabulary(
        "tag,group,never_with,always_with\nclose-up shot,shot,,\ndance,action,,crowd\ncrowd,people,,\n"
        "night,light,dance,\n"))
    TLB.save_label({"id": TLB.card_id("tags", "vid_a", 5.0), "type": "tags", "video_id": "vid_a", "t": 5.0},
                   tags=["close-up shot", "dance"])
    TLB.save_label({"id": TLB.card_id("tags", "vid_a", 20.0), "type": "tags", "video_id": "vid_a", "t": 20.0},
                   tags=["night", "dance", "crowd"])
    try:
        at = page_app("pages/6_Training_Labels.py")
        at.session_state["tl_view"] = "🧪 Trial results"
        at.session_state["tr_sub"] = "🖼️ Compare tags"
        at.run()
        assert not errors(at), errors(at)
        assert any("1 can be fixed automatically, 1 need you" in e.label for e in at.expander)
        next(b for b in at.button if b.key == "tr_rules_apply").click().run()
        assert not errors(at), errors(at)
        got = {i["id"]: i["tags"] for i in TLB.load_labels()}
        assert got[TLB.card_id("tags", "vid_a", 5.0)] == ["close-up shot", "dance", "crowd"]
        next(b for b in at.button if b.key == "tr_rules_show").click().run()
        assert at.session_state["tr_filter"] == "rules"
        caps = " ".join(c.value for c in at.caption)
        assert "“night” and “dance” are never-together" in caps or "“dance” and “night” are never-together" in caps
        assert "20" in caps or "0:20" in caps
    finally:
        if backup is None:
            TLB.VOCAB_FILE.unlink(missing_ok=True)
        else:
            TLB.VOCAB_FILE.write_bytes(backup)


def test_suggest_thresholds():
    import trial_results as TRS
    rows = [{"tag": "dance", "threshold": None}, {"tag": "kiss", "threshold": 60}, {"tag": "rare", "threshold": None}]
    # dance: right when conf ≥ 80, wrong at 55–60 → suggested 65–80 beats 50
    ms = [{"mine": ["dance"], "teachers": {"32b": {"conf": {"dance": c}}}} for c in (80, 85, 90, 95)]
    ms += [{"mine": [], "teachers": {"32b": {"conf": {"dance": c}}}} for c in (55, 60)]
    ms += [{"mine": ["kiss"], "teachers": {"32b": {"conf": {"kiss": c}}}} for c in (65, 70, 75)]
    ms += [{"mine": ["rare"], "teachers": {"32b": {"conf": {}}}}]
    ms += [{"mine": None, "teachers": {"32b": {"conf": {"dance": 10}}}},          # not labelled → ignored
           {"mine": ["dance"], "teachers": {"32b": {"tags": ["dance"]}}}]         # no confidences → ignored
    got = {x["tag"]: x for x in TRS.suggest_thresholds(ms, "32b", rows, 50)}
    assert set(got) == {"dance", "kiss"}                      # rare: too few of your frames
    assert got["dance"]["current"] == 50 and got["dance"]["f1_now"] < 1.0
    assert got["dance"]["suggested"] == 80 and got["dance"]["f1_best"] == 1.0   # ties → the higher one
    assert got["kiss"]["current"] == 60 and got["kiss"]["f1_now"] == 1.0 and got["kiss"]["suggested"] == 65
    assert list(got)[0] == "dance"                            # biggest gain first


def _test_runs():
    v1 = "tag,group,threshold\nclose-up shot,shot,\ndance,action,\ncrowd,people,\n"
    v2 = "tag,group,threshold,double_check\nclose-up shot,shot,,\ndance,action,80,yes\nnight,light,,\n"
    frames_a = [{"v": "vid_a", "t": 5.0, "g": None, "tags": ["close-up shot", "dance", "crowd"]},
                {"v": "vid_a", "t": 12.0, "g": None, "tags": ["dance"]}]
    frames_b = [{"v": "vid_a", "t": 5.0, "g": None, "tags": ["close-up shot", "dance"]},
                {"v": "vid_a", "t": 12.0, "g": None, "tags": []}]
    return {"set": {"videos": [{"video_id": "vid_a"}], "hours": 0.1, "created": "2026-10-30 18:30"},
            "runs": [{"id": "2026-10-30_1830_baseline", "vocab_csv": v1, "frames": frames_a, "cost": 0.6,
                      "looks": 3, "look_added": 1, "look_removed": 0},
                     {"id": "2026-10-30_1900_step1", "vocab_csv": v2, "frames": frames_b, "cost": 0.7}]}


def test_test_run_scoring_and_vocab_changes():
    import training_labels as TLB
    data = _test_runs()
    labels = [{"id": TLB.card_id("tags", "vid_a", 5.0), "type": "tags", "video_id": "vid_a", "t": 5.0,
               "tags": ["close-up shot", "dance"]},
              {"id": TLB.card_id("tags", "vid_a", 12.0), "type": "tags", "video_id": "vid_a", "t": 12.0,
               "tags": []}]
    sc = TR.test_run_scores(data["runs"], labels, ["close-up shot", "dance", "crowd", "night"])
    assert sc[0]["score"]["overall"]["fp"] == 2 and sc[1]["score"]["overall"]["fp"] == 0
    ch = {c["tag"]: c for c in TR.tag_changes(sc[0]["score"], sc[1]["score"])}
    assert ch["dance"]["wrong_a"] == 1 and ch["dance"]["wrong_b"] == 0 and ch["dance"]["change"] == -1
    assert set(ch) == {"dance", "crowd"}
    lines = TR.vocab_changes(data["runs"][0]["vocab_csv"], data["runs"][1]["vocab_csv"])
    assert "➕ night added" in lines and "➖ crowd removed" in lines
    assert "dance · threshold: — → 80" in lines and "dance · double_check: — → yes" in lines
    assert TR.vocab_changes("", "x") == []


def test_summary_shows_test_runs(bundle, no_labels):
    import training_labels as TLB
    from tests.apptest_util import errors, page_app
    (TR.RESULTS_DIR / "test_runs.json").write_text(json.dumps(_test_runs()))
    TLB.save_label({"id": TLB.card_id("tags", "vid_a", 5.0), "type": "tags", "video_id": "vid_a", "t": 5.0},
                   tags=["close-up shot", "dance"])
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🗄️ Teacher"
    at.run()
    assert not errors(at), errors(at)
    assert "##### 🧪 Teacher test runs" in [m.value for m in at.markdown]
    runs = [d for d in at.dataframe if "Settings" in d.value.columns]                 # the teacher runs table
    assert runs and any("needs-a-tag fills" in str(s) or "2nd looks" in str(s) for s in runs[0].value["Settings"])
    assert any("Errors" in d.value.columns for d in at.dataframe)                     # per-tag changes
    assert any("dance · threshold" in c.value for c in at.caption)


def test_suspect_labels():
    vids = {"vid_a": {"moments": [{"t": 5.0, "mine": ["close-up shot", "dance"]},
                                  {"t": 12.0, "mine": ["crowd"]},
                                  {"t": 20.0, "mine": None},
                                  {"t": 30.0, "mine": ["night"]}]}}
    data = {"runs": [
        {"id": "r1", "created": "2026-10-31 10:00",
         "frames": [{"v": "vid_a", "t": 5.0, "tags": ["close-up shot", "crowd"]},
                    {"v": "vid_a", "t": 12.0, "tags": ["crowd"]},
                    {"v": "vid_a", "t": 20.0, "tags": ["dance"]},
                    {"v": "vid_a", "t": 30.0, "tags": []}]},
        {"id": "r2", "created": "2026-10-31 11:00",
         "frames": [{"v": "vid_a", "t": 5.0, "tags": ["Close-Up Shot", "crowd"]},
                    {"v": "vid_a", "t": 12.0, "tags": []},
                    {"v": "vid_a", "t": 30.0, "tags": []}]}]}
    sus = TR.suspect_labels(data, ["r1", "r2"], vids, {})
    # 5 s: both see crowd (you don't); neither sees dance (you do). 12 s: only r1 agrees with you → not suspect.
    assert sus[("vid_a", 5.0)]["there"] == ["crowd"] and sus[("vid_a", 5.0)]["not_there"] == ["dance"]
    assert ("vid_a", 12.0) not in sus and ("vid_a", 20.0) not in sus
    assert sus[("vid_a", 30.0)]["not_there"] == ["night"]
    # saved after the newest run → checked, left out; saved before it → still listed
    assert ("vid_a", 30.0) not in TR.suspect_labels(data, ["r1", "r2"], vids, {("vid_a", 30.0): "2026-10-31T11:05:00"})
    assert ("vid_a", 30.0) in TR.suspect_labels(data, ["r1", "r2"], vids, {("vid_a", 30.0): "2026-10-31T09:00:00"})
    one = TR.suspect_labels(data, ["r1"], vids, {})                            # one run: its own disagreements
    assert one[("vid_a", 5.0)]["there"] == ["crowd"] and one[("vid_a", 5.0)]["detail"] == {}
    assert TR.suspect_labels(data, [], vids, {}) == {}
    # what a look run recorded travels with the frame
    data["runs"][0]["frames"][0].update({"look": {"on_top": "her", "facing": "towards"}, "nb": ["crowd"]})
    d = TR.suspect_labels(data, ["r1"], vids, {})[("vid_a", 5.0)]["detail"]["r1"]
    assert d == {"look": {"on_top": "her", "facing": "towards"}, "nb": ["crowd"]}


def test_check_my_label_filter(bundle, no_labels):
    import training_labels as TLB
    from tests.apptest_util import errors, page_app
    data = _bundle()
    m = data["videos"]["vid_a"]["moments"][0]
    runs = {"runs": [{"id": f"2026-10-31_100{i}_r{i}", "created": f"2026-10-31 10:0{i}",
                      "frames": [{"v": "vid_a", "t": m["t"], "tags": ["night"]}]} for i in (1, 2)],
            "set": {"videos": [{"video_id": "vid_a"}], "hours": 0.1}}
    (TR.RESULTS_DIR / "test_runs.json").write_text(json.dumps(runs))
    TLB.save_label({"id": TLB.card_id("tags", "vid_a", m["t"]), "type": "tags", "video_id": "vid_a",
                    "t": m["t"]}, tags=["dance"])
    # pretend the label is older than the runs
    items = TLB.load_labels()
    items[0]["labelled_at"] = "2026-10-30T09:00:00"
    TLB._write_labels(items)
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_filter"] = "suspect"
    at.run()
    assert not errors(at), errors(at)
    caps = " ".join(c.value for c in at.caption)
    assert "they all see **night**" in caps and "none of them see **dance**" in caps
    next(b for b in at.button if b.label == "✅ My label is right").click().run()
    assert not errors(at), errors(at)
    assert TLB.load_labels()[0]["tags"] == ["dance"]                       # unchanged, but now checked
    assert not any(b.label == "✅ My label is right" for b in at.button)    # gone from the list


def test_check_runs_feed_the_filter(bundle, no_labels):
    """🔍 check runs (training frames) show in Check my label like test runs do."""
    import training_labels as TLB
    from tests.apptest_util import errors, page_app
    m = _bundle()["videos"]["vid_a"]["moments"][0]
    runs = {"runs": [{"id": f"2026-10-31_130{i}_check-{n}", "created": f"2026-10-31 13:0{i}",
                      "frames": [{"v": "vid_a", "t": m["t"], "tags": ["crowd"]}]}
                     for i, n in ((1, "plain"), (2, "lora-try1"))]}
    (TR.RESULTS_DIR / "check_runs.json").write_text(json.dumps(runs))
    TLB.save_label({"id": TLB.card_id("tags", "vid_a", m["t"]), "type": "tags", "video_id": "vid_a",
                    "t": m["t"]}, tags=["dance"])
    items = TLB.load_labels()
    items[0]["labelled_at"] = "2026-10-30T09:00:00"
    TLB._write_labels(items)
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_filter"] = "suspect"
    at.run()
    assert not errors(at), errors(at)
    assert "they all see **crowd**" in " ".join(c.value for c in at.caption)


def test_teacher_label_with_lora():
    assert TR.teacher_label("Qwen/Qwen3-VL-32B-Instruct-FP8") == "32B FP8"
    assert TR.teacher_label("Qwen/Qwen3-VL-32B-Instruct-FP8 + LoRA try2") == "32B FP8 + LoRA try2"
    assert TR.teacher_label("Qwen/Qwen3-VL-8B-Instruct") == "8B"


def test_neighbours_filter():
    filled = {"key": "a", "teachers": {"32b": {"tags": ["x"], "neighbour_added": ["x"]}}}
    plain = {"key": "b", "teachers": {"32b": {"tags": ["x"]}}}
    assert [m["key"] for m in TR.filter_moments([filled, plain], "neighbours")] == ["a"]


def test_look_line():
    line = TR.look_line({"women": "1", "men": "1", "penetration": "clearly visible", "penetration_with": "penis",
                         "penetration_where": "vagina", "her_posture": "lying on back", "her_front_faces": "the camera",
                         "partner_posture": "kneeling upright", "partner_legs_point": "away from the camera",
                         "on_top": "partner", "facing": "towards", "viewpoint": "onlooker"})
    assert "penetration clearly visible (penis, vagina)" in line and "on top: partner" in line
    assert TR.look_line({}) == ""
    acts = TR.look_line({"her_hands": "holding his penis", "her_mouth": "on his penis", "mouth_depth": "partway",
                         "her_feet": "not visible", "partner_hands": "on her body", "partner_mouth": "other"})
    assert acts == "her hands holding his penis, her mouth on his penis, depth partway, his hands on her body"
    keys = [k for k, _, _ in TR.look_steps()]
    assert keys[:8] == ["women", "men", "her_hands", "her_mouth", "mouth_depth", "her_feet", "partner_hands",
                        "partner_mouth"] and keys[8] == "penetration" and keys[-1] == "viewpoint"


def test_look_changes():
    a = [{"key": "women", "question": "How many women?", "choices": ["0", "1"]},
         {"key": "facing", "question": "Facing", "choices": ["towards", "away"]}]
    b = [{"key": "facing", "question": "Facing?", "choices": ["towards", "away"]},
         {"key": "women", "question": "How many women?", "choices": ["0", "1", "2"]},
         {"key": "hair", "question": "Hair", "choices": ["blonde"]}]
    ch = TR.look_changes(a, b)
    assert "👁️ hair · step added" in ch and "👁️ facing · question: Facing → Facing?" in ch
    assert "👁️ women · choices: 0 | 1 → 0 | 1 | 2" in ch and "👁️ order of the steps changed" in ch
    assert TR.look_changes(None, b) == [] and TR.look_changes(a, a) == []


def _a_look(**kw):
    return {"women": "1", "men": "1", "penetration": "clearly visible", "penetration_with": "penis",
            "penetration_where": "vagina", "hidden": "nothing important hidden",
            "her_posture": "squatting / straddling", "her_front_faces": "the camera",
            "partner_posture": "lying on back", "partner_legs_point": "towards the camera",
            "on_top": "her", "facing": "towards", "viewpoint": "onlooker", "summary": "She rides him.", **kw}


def test_look_checks_save_summary_and_csv(bundle, no_labels):
    TR.LOOK_CHECKS_FILE.unlink(missing_ok=True)
    TR.save_look_check("run1", "vid_a", 5.0, _a_look(), {**_a_look(), "facing": "away"}, ["reverse cowgirl"],
                       ["Cowgirl position"])
    TR.save_look_check("run1", "vid_a", 9.0, _a_look(), _a_look(), [], [])
    items = TR.load_look_checks()
    assert set(items) == {"run1|vid_a|5.00", "run1|vid_a|9.00"}
    assert items["run1|vid_a|5.00"]["mine"]["facing"] == "away" and "summary" not in items["run1|vid_a|5.00"]["mine"]
    summ = {r["step"]: r for r in TR.look_check_summary(items.values())}
    assert summ["facing"]["checked"] == 2 and summ["facing"]["right"] == 1 and summ["facing"]["pct"] == 50
    assert summ["facing"]["fixes"] == [("towards", "away", 1)] and summ["on_top"]["pct"] == 100
    csv_text = TR.look_checks_csv(items.values())
    assert csv_text.splitlines()[0].startswith("run,video_id,t,step,model_answer,my_answer,right")
    assert "run1,vid_a,5.0,facing,towards,away,no,reverse cowgirl,Cowgirl position" in csv_text
    TR.save_look_check("run1", "vid_a", 5.0, _a_look(), _a_look(), [], [])       # re-saving replaces
    assert TR.load_look_checks()["run1|vid_a|5.00"]["mine"]["facing"] == "towards"
    TR.LOOK_CHECKS_FILE.unlink()


def test_look_check_form_on_a_card(bundle, no_labels):
    import training_labels as TLB
    TR.LOOK_CHECKS_FILE.unlink(missing_ok=True)
    from tests.apptest_util import errors, page_app
    m = _bundle()["videos"]["vid_a"]["moments"][0]
    rid = "2026-11-02_0045_plain-look"
    runs = {"runs": [{"id": rid, "created": "2026-11-02 00:45",
                      "frames": [{"v": "vid_a", "t": m["t"], "tags": ["night"], "look": _a_look()}]}],
            "set": {"videos": [{"video_id": "vid_a"}], "hours": 0.1}}
    (TR.RESULTS_DIR / "test_runs.json").write_text(json.dumps(runs))
    TLB.save_label({"id": TLB.card_id("tags", "vid_a", m["t"]), "type": "tags", "video_id": "vid_a",
                    "t": m["t"]}, tags=["dance"])
    items = TLB.load_labels()
    items[0]["labelled_at"] = "2026-10-30T09:00:00"
    TLB._write_labels(items)
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_filter"] = "suspect"
    at.session_state["tr_sus_runs"] = [rid]
    at.run()
    assert not errors(at), errors(at)
    caps = " ".join(c.value for c in at.caption)
    assert "on top: her" in caps and "She rides him." in caps
    facing = next(s for s in at.selectbox if s.label.startswith("She faces her partner"))
    assert facing.value == "towards"
    facing.set_value("away")
    next(b for b in at.button if b.label == "💾 Save look check").click().run()
    assert not errors(at), errors(at)
    saved = TR.load_look_checks()[TR.look_check_key(rid, "vid_a", m["t"])]
    assert saved["mine"]["facing"] == "away" and saved["model"]["facing"] == "towards"
    assert saved["my_tags"] == ["dance"] and saved["run_tags"] == ["night"]
    assert any("1 frame(s) checked" in e.label for e in at.expander)
    assert any("12 of 13 right" in e.label for e in at.expander)
    TR.LOOK_CHECKS_FILE.unlink()


def test_filter_picked_by_counts_confirmed_tags_only():
    me = {"mine": ["Wide shot"], "teachers": {"32b": {"tags": ["Close-up"], "unsure": ["Wide shot"]}}, "old": []}
    model = {"mine": ["Close-up"], "teachers": {"32b": {"tags": ["Wide shot"], "unsure": []}}, "old": []}
    both = {"mine": ["wide shot"], "teachers": {"32b": {"tags": ["Wide shot"]}}, "old": []}
    unlabelled = {"teachers": {"32b": {"tags": ["Wide shot"]}}, "old": []}
    unsure_only = {"teachers": {"32b": {"tags": [], "unsure": ["Wide shot"]}}, "old": []}
    old_only = {"teachers": {"32b": {"tags": []}}, "old": ["Wide shot"]}
    ms = [me, model, both, unlabelled, unsure_only, old_only]
    f = lambda by, skip=True: TR.filter_moments(ms, "all", "Wide shot", skip_old=skip, by=by)
    assert f("mine") == [me, both]
    assert f("model") == [model, both, unlabelled]
    assert f("both") == [both]
    assert f("any") == [me, model, both, unlabelled]
    assert f("any", skip=False) == [me, model, both, unlabelled, old_only]


@pytest.fixture
def clean_labels(tmp_path):
    import training_labels as TLB
    saved = tmp_path / "labels.bak"
    had = TLB.LABELS_FILE.exists()
    if had:
        shutil.copy2(TLB.LABELS_FILE, saved)
        TLB.LABELS_FILE.unlink()
    yield TLB
    if TLB.LABELS_FILE.exists():
        TLB.LABELS_FILE.unlink()
    if had:
        shutil.copy2(saved, TLB.LABELS_FILE)


VOCAB = [{"tag": "Cowgirl", "group": "position", "one_of": "", "never_with": "reverse cowgirl; Doggy"},
         {"tag": "reverse cowgirl", "group": "position", "one_of": "", "never_with": ""},
         {"tag": "Doggy", "group": "position", "one_of": "", "never_with": ""},
         {"tag": "Medium shot", "group": "shot", "one_of": "", "never_with": ""}]


def test_batch_tag_partial_full_and_undo(clean_labels):
    TLB = clean_labels
    TLB.save_label({"id": TLB.card_id("tags", "v", 5.0), "type": "tags", "video_id": "v", "t": 5.0},
                   tags=["Doggy", "Medium shot"])
    ms = [{"video_id": "v", "t": 5.0, "scene_id": 0}, {"video_id": "v", "t": 9.0, "scene_id": 1}]
    res = TLB.batch_tag(ms, add=["Cowgirl"], rows=VOCAB)
    assert (res["full"], res["partial"]) == (1, 1)
    items = {i["id"]: i for i in TLB.load_labels()}
    # the full label: Cowgirl added, Doggy (never-with) taken out, the rest kept
    assert items[TLB.card_id("tags", "v", 5.0)]["tags"] == ["Medium shot", "Cowgirl"]
    part = items[TLB.partial_id("v", 9.0)]
    assert part["type"] == "partial" and part["yes"] == ["Cowgirl"]
    assert part["no"] == ["Doggy", "reverse cowgirl"]
    # removing later flips it; adding a ruled-out tag back takes it out of "no"
    TLB.batch_tag(ms[1:], remove=["Cowgirl"], rows=VOCAB)
    TLB.batch_tag(ms[1:], add=["Doggy"], rows=VOCAB)
    part = {i["id"]: i for i in TLB.load_labels()}[TLB.partial_id("v", 9.0)]
    assert part["yes"] == ["Doggy"] and part["no"] == ["Cowgirl", "reverse cowgirl"]
    # undo the first batch only: back to the starting full label, no partial
    TLB.undo_batch(res["undo"])
    items = {i["id"]: i for i in TLB.load_labels()}
    assert items[TLB.card_id("tags", "v", 5.0)]["tags"] == ["Doggy", "Medium shot"]
    assert TLB.partial_id("v", 9.0) not in items
    # partial answers never count as full labels anywhere else
    assert TLB.summary(TLB.load_labels())["tags"] == 1


def test_partial_labels_reach_the_view(clean_labels):
    TLB = clean_labels
    TLB.batch_tag([{"video_id": "vid_a", "t": 12.0, "scene_id": 1}], add=["crowd"], remove=["dance"])
    videos = TR.with_saved_labels(_bundle()["videos"], TLB.load_labels())
    m = [m for m in videos["vid_a"]["moments"] if m["key"] == "s1_0"][0]
    assert m["partial"] == {"yes": ["crowd"], "no": ["dance"], "done": []}
    assert TR.your_answer(m, "Crowd") is True and TR.your_answer(m, "dance") is False
    assert TR.your_answer(m, "night") is None and TR.model_has(m, "dance")
    assert TR.picked_by(m, "crowd", "mine") and TR.picked_by(m, "crowd", "both")


def test_grid_view_batch_tags(bundle, clean_labels):
    from tests.apptest_util import errors, page_app
    TLB = clean_labels
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_view"] = "grid"
    at.session_state["tr_set"] = "train"                 # the trial's videos, not 📚 the rest of the library
    at.run()
    assert not errors(at), errors(at)
    ticks = [c for c in at.checkbox if c.key and c.key.startswith("tr_gt_")]
    assert len(ticks) == 3
    at.session_state["tr_g_chosen"] = ["crowd"]
    for c in ticks[1:]:
        c.check()
    at.button(key="tr_g_add_top").click().run()
    assert not errors(at), errors(at)
    items = TLB.load_labels()
    parts = [i for i in items if i["type"] == "partial"]
    assert sorted(i["t"] for i in parts) == [12.0, 20.0] and all(i["yes"] == ["crowd"] for i in parts)
    assert not any(c.value for c in at.checkbox if c.key and c.key.startswith("tr_gt_"))   # ticks cleared
    at.button(key="tr_g_undo_btn").click().run()
    assert not errors(at), errors(at)
    assert not [i for i in TLB.load_labels() if i["type"] == "partial"]


def test_part_reviewed_toggle_and_finishing_a_frame(bundle, clean_labels):
    from tests.apptest_util import errors, page_app
    TLB = clean_labels
    TLB.batch_tag([{"video_id": "vid_a", "t": 12.0, "scene_id": 1}], add=["night"], remove=["crowd"])
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_only_part"] = True
    at.run()
    assert not errors(at), errors(at)
    md = " ".join(m.value for m in at.markdown)
    assert "You ▦" in md and "Page (of" not in md
    assert len([b for b in at.button if b.label == "✏️ Correct"]) == 1      # only the batch-tagged frame
    # finishing it as a full label clears the batch answer
    m = [m for m in _bundle()["videos"]["vid_a"]["moments"] if m["key"] == "s1_0"][0]
    TLB.save_label(TR.correction_card("vid_a", m, []), tags=["dance", "night"])
    assert [i["type"] for i in TLB.load_labels()] == ["tags"]


def test_finish_partials_keeps_only_your_tags(clean_labels):
    TLB = clean_labels
    vocab = [dict(r, group_required=(r["group"] == "shot")) for r in VOCAB]
    ms = [{"video_id": "v", "t": 5.0, "scene_id": 0}, {"video_id": "v", "t": 9.0, "scene_id": 1}]
    TLB.batch_tag(ms, add=["Cowgirl"], rows=vocab)
    TLB.batch_tag(ms[:1], add=["Medium shot"], rows=vocab)
    res = TLB.finish_partials(ms, vocab)
    assert res["done"] == 1 and res["skipped"] == [("v", 9.0, ["shot"])]     # no shot tag yet → left as it is
    items = {i["id"]: i for i in TLB.load_labels()}
    full = items[TLB.card_id("tags", "v", 5.0)]
    assert full["type"] == "tags" and full["tags"] == ["Cowgirl", "Medium shot"] and full["from_batch"]
    assert TLB.partial_id("v", 5.0) not in items and TLB.partial_id("v", 9.0) in items
    TLB.undo_batch(res["undo"])
    items = {i["id"]: i for i in TLB.load_labels()}
    assert TLB.card_id("tags", "v", 5.0) not in items and items[TLB.partial_id("v", 5.0)]["type"] == "partial"


def test_still_to_review_and_finish_button(bundle, clean_labels):
    from tests.apptest_util import errors, page_app
    TLB = clean_labels
    TLB.batch_tag([{"video_id": "vid_a", "t": 12.0, "scene_id": 1}], add=["dance"])
    m = [m for m in TR.with_saved_labels(_bundle()["videos"], TLB.load_labels())["vid_a"]["moments"]
         if m["key"] == "s1_0"][0]
    assert TR.still_to_review(m) == ["crowd"]                     # the 32B's other tag, not reviewed yet
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_only_part"] = True
    at.run()
    assert not errors(at), errors(at)
    assert "? crowd" in " ".join(x.value for x in at.markdown)
    at.button(key="tr_finish_btn").click().run()
    at.button(key="tr_finish_yes").click().run()
    assert not errors(at), errors(at)
    full = [i for i in TLB.load_labels() if i["type"] == "tags"]
    assert len(full) == 1 and full[0]["tags"] == ["dance"]
    assert not [i for i in TLB.load_labels() if i["type"] == "partial"]
    at.session_state["tr_view"] = "grid"
    at.run()
    assert not errors(at), errors(at)


def test_grid_marks_show_the_whole_category():
    import trial_view as TV
    m = {"mine": None, "partial": {"yes": ["Doggy"], "no": ["Cowgirl"]},
         "teachers": {"32b": {"tags": ["reverse cowgirl", "Medium shot"]}}}
    cat = ["Cowgirl", "reverse cowgirl", "Doggy", "Missionary"]
    assert TV._gtags(m, cat) == ["Cowgirl", "reverse cowgirl", "Doggy"]        # Missionary: nobody has it
    html_ = TV._gstate(m, TV._gtags(m, cat), ["Doggy", "Missionary"])
    assert "✗ Cowgirl" in html_ and "◆ reverse cowgi" in html_ and "● Doggy" in html_
    assert html_.count("mp-gpick") == 1 and "+ Missionary" in html_          # hidden until the frame is ticked
    assert "mp-gpick" in TV.GRID_CSS and ":has(input:checked)" in TV.GRID_CSS


def test_grid_all_categories(bundle, clean_labels):
    from tests.apptest_util import errors, page_app
    import trial_view as TV
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_view"] = "grid"
    at.run()
    assert not errors(at), errors(at)
    assert TV.GRID_ALL in at.selectbox(key="tr_g_cat").options
    at.selectbox(key="tr_g_cat").set_value(TV.GRID_ALL).run()
    assert not errors(at), errors(at)
    rows, _ = __import__("training_labels").load_vocabulary()
    assert len(at.button_group[-1].options) == len({r["tag"] for r in rows})   # every tag as a pill


def test_hide_red_and_swipe_script(bundle, clean_labels):
    from tests.apptest_util import errors, page_app
    import trial_view as TV
    assert 'class="mp-red"' in TV._chip("x", TR.WRONG) and "mp-red" not in TV._chip("x", TR.RIGHT)
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_view"] = "grid"
    at.session_state["tr_hide_red"] = True
    at.run()
    assert not errors(at), errors(at)
    assert any(".mp-red {display:none" in m.value for m in at.markdown)
    assert "__mpGridSel" in TV.GRID_JS


def test_grid_bottom_picker_follows_the_top(bundle, clean_labels):
    from tests.apptest_util import errors, page_app
    import trial_view as TV
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_view"] = "grid"
    at.run()
    assert not errors(at), errors(at)
    assert at.selectbox(key="tr_g_cat_b_0").value == at.selectbox(key="tr_g_cat").value
    at.selectbox(key="tr_g_cat_b_0").set_value(TV.GRID_ALL).run()         # change it at the bottom …
    assert not errors(at), errors(at)
    assert at.selectbox(key="tr_g_cat").value == TV.GRID_ALL             # … and the top follows
    # tags: picked at the top show at the bottom, and a pick at the bottom is added to them
    def pills(sfx):
        return [g for g in at.button_group if (g.key or "").startswith("tr_g_pills_") and
                (g.key.rsplit("_", 1)[0].endswith("_b") == bool(sfx))][0]
    top = pills("")
    a, b = top.options[0], top.options[1]
    top.set_value([a]).run()
    assert not errors(at), errors(at)
    assert at.session_state["tr_g_chosen"] == [a] and list(pills("_b").value) == [a]
    pills("_b").set_value([a, b]).run()
    assert not errors(at), errors(at)
    assert at.session_state["tr_g_chosen"] == [a, b] and list(pills("").value) == [a, b]
    assert "stElementToolbar" in TV.GRID_CSS and "repeat(5," in TV.GRID_CSS


def test_grid_stale_copy_cannot_undo_a_pick(bundle, clean_labels):
    """A quick second tap: the browser sends the other copy's OLD value along with it (it hasn't been told the
    new one yet). That used to read as the other copy being changed back, and wiped the picked tags."""
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_view"] = "grid"
    at.run()
    def pills(bottom):
        return [g for g in at.button_group if (g.key or "").startswith("tr_g_pills_") and
                (g.key.rsplit("_", 1)[0].endswith("_b") == bottom)][0]
    top, old_bottom_key = pills(False), pills(True).key
    a, b = top.options[0], top.options[1]
    top.set_value([a]).run()
    assert at.session_state["tr_g_chosen"] == [a]
    assert pills(True).key != old_bottom_key            # the bottom copy was made afresh, not pushed a value
    # the second tap, arriving with the bottom copy's old (empty) value under its old key
    at.session_state[old_bottom_key] = []
    pills(False).set_value([a, b]).run()
    assert not errors(at), errors(at)
    assert at.session_state["tr_g_chosen"] == [a, b]


def test_grid_thumbnails_are_small(tmp_path):
    import trial_view as TV
    from PIL import Image
    big = tmp_path / "f.jpg"
    Image.effect_noise((640, 360), 60).convert("RGB").save(big, quality=90)
    th = TV._thumb(big)
    assert th != big and Image.open(th).size == (400, 225)
    assert th.stat().st_size < big.stat().st_size / 2
    assert TV._thumb(th) == th or Image.open(TV._thumb(th)).width == 400          # already small: kept
    assert TV._thumb(tmp_path / "missing.jpg") == tmp_path / "missing.jpg"


def test_new_test_videos_in_compare_tags(bundle, clean_labels):
    """🧪 New test videos show in Compare tags (Videos → 🧪 Test set, with the original test videos) with frames from
    the preview copy, for the grid."""
    import subprocess
    import catalogue_store as CS
    from library_common import proxy_path
    from tests.apptest_util import errors, page_app
    TLB = clean_labels
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    p = proxy_path("vid_c")
    existed = p.exists()
    if not existed:
        p.parent.mkdir(parents=True, exist_ok=True)
        dur = max(s["end"] for s in CS.video_index()["vid_c"]["scenes"]) + 1
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc=size=320x180:rate=10:duration={dur:.0f}", "-pix_fmt", "yuv420p", str(p)], check=True)
    had_extra = TLB.EXTRA_TEST_FILE.read_bytes() if TLB.EXTRA_TEST_FILE.exists() else None
    try:
        TLB.save_extra_test(["vid_c"])
        vids = TR.new_test_videos(CS.video_index())
        ms = vids["vid_c"]["moments"]
        assert len(ms) == TLB.TEST_MOMENTS_PER_VIDEO and TR.frame_path(ms[0]).exists()
        assert TR.moment_id("vid_c", ms[0]) == TLB.card_id("tags", "vid_c", ms[0]["t"])   # same label as its card
        at = page_app("pages/6_Training_Labels.py")
        at.session_state["tl_view"] = "🧪 Trial results"
        at.session_state["tr_sub"] = "🖼️ Compare tags"
        at.session_state["tr_set"] = "test"
        at.session_state["tr_view"] = "grid"
        at.run()
        assert not errors(at), errors(at)
        ticks = [c for c in at.checkbox if c.key and c.key.startswith("tr_gt_")]
        assert len(ticks) == 48
        at.session_state["tr_g_chosen"] = ["dance"]
        ticks[0].check()
        at.button(key="tr_g_add_top").click().run()
        assert not errors(at), errors(at)
        part = [i for i in TLB.load_labels() if i["type"] == "partial"]
        assert len(part) == 1 and part[0]["video_id"] == "vid_c" and part[0]["t"] == ms[0]["t"]
        assert "new test" not in [str(o) for o in at.selectbox(key="tr_batch").options]   # one place to find them
    finally:
        if had_extra is None:
            TLB.EXTRA_TEST_FILE.unlink(missing_ok=True)
        else:
            TLB.EXTRA_TEST_FILE.write_bytes(had_extra)
        if not existed:
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# One category at a time, one Videos control, 📚 the other videos (12 Nov 2026)
# ---------------------------------------------------------------------------

def test_finish_one_category_at_a_time(clean_labels):
    TLB = clean_labels
    vocab = [dict(r, group_required=(r["group"] == "shot")) for r in VOCAB]
    cats = {c["title"]: c for c in TLB.categories(vocab)}
    assert set(cats) == {"position", "shot"} and cats["shot"]["required"] and not cats["position"]["required"]
    ms = [{"video_id": "v", "t": 5.0, "scene_id": 0}, {"video_id": "v", "t": 9.0, "scene_id": 1},
          {"video_id": "v", "t": 13.0, "scene_id": 2}]
    TLB.batch_tag(ms[:1], add=["Cowgirl"], rows=vocab)                 # 9.0 and 13.0: looked at, no position there
    res = TLB.finish_category(ms, "position", vocab)
    assert res["done"] == 3 and res["full"] == 0 and res["skipped"] == []
    items = {i["id"]: i for i in TLB.load_labels()}
    a, b = items[TLB.partial_id("v", 5.0)], items[TLB.partial_id("v", 9.0)]
    assert a["yes"] == ["Cowgirl"] and a["done"] == ["position"]
    assert {t.lower() for t in a["no"]} == {t.lower() for t in cats["position"]["tags"]} - {"cowgirl"}
    assert b["yes"] == [] and b["done"] == ["position"] and len(b["no"]) == len(cats["position"]["tags"])
    assert not {t.lower() for t in a["no"]} & {t.lower() for t in cats["shot"]["tags"]}        # shot left alone
    assert TLB.finish_category(ms, "position", vocab)["done"] == 0                                 # nothing left to do
    # the view: reviewed in that category only, and "not reviewed yet" follows the category
    vids = TR.with_saved_labels({"v": {"moments": [{"key": f"k{m['t']}", "t": m["t"], "mine": None, "teachers": {}}
                                                   for m in ms]}}, TLB.load_labels())
    got = vids["v"]["moments"]
    assert all(TR.reviewed(m, "position") for m in got) and not any(TR.reviewed(m, "shot") for m in got)
    assert not any(TR.reviewed(m) for m in got)
    assert TR.progress(vids, "position") == {"v": (3, 3)} and TR.progress(vids, "shot") == {"v": (0, 3)}
    assert TR.filter_moments(got, "todo", category="position") == []
    assert len(TR.filter_moments(got, "todo", category="shot")) == 3 and len(TR.filter_moments(got, "todo")) == 3
    # a batch tag afterwards keeps the category finished
    TLB.batch_tag(ms[1:2], add=["Medium shot"], rows=vocab)
    assert {i["id"]: i for i in TLB.load_labels()}[TLB.partial_id("v", 9.0)]["done"] == ["position"]
    # the needs-a-tag category: frames with no shot tag stay not reviewed; the last category makes a full label
    res = TLB.finish_category(ms, "shot", vocab)
    assert res["done"] == 1 and res["full"] == 1 and sorted(res["skipped"]) == [("v", 5.0), ("v", 13.0)]
    items = {i["id"]: i for i in TLB.load_labels()}
    full = items[TLB.card_id("tags", "v", 9.0)]
    assert full["type"] == "tags" and full["tags"] == ["Medium shot"] and full["from_batch"]
    assert TLB.partial_id("v", 9.0) not in items and "shot" not in items[TLB.partial_id("v", 5.0)]["done"]
    TLB.undo_batch(res["undo"])
    items = {i["id"]: i for i in TLB.load_labels()}
    assert TLB.card_id("tags", "v", 9.0) not in items and items[TLB.partial_id("v", 9.0)]["done"] == ["position"]
    # a frame with a full label is never touched
    TLB.finish_category(ms, "shot", vocab)
    before = json.dumps(TLB.load_labels(), sort_keys=True)
    assert TLB.finish_category(ms[1:2], "position", vocab)["done"] == 0
    assert json.dumps(TLB.load_labels(), sort_keys=True) == before
    assert TLB.finish_category(ms, "nope", vocab) == {"done": 0, "full": 0, "skipped": [], "undo": []}


def test_other_videos_with_the_students_tags(bundle, tmp_path, clean_labels):
    import numpy as np
    import catalogue_store as CS
    TLB = clean_labels
    index = CS.video_index()
    known = set(_bundle()["videos"])
    had_extra = TLB.EXTRA_TEST_FILE.read_bytes() if TLB.EXTRA_TEST_FILE.exists() else None
    try:
        TLB.save_extra_test(["vid_c"])
        ids = TR.other_video_ids(index, known)
        assert ids == ["vid_d", "vid_e", "vid_f"]                          # not the trial's, not the test set's
        folder = tmp_path / "student_scores"
        folder.mkdir()
        assert TR.student_index(folder) == {} and TR.student_tags_at("vid_d", [1.0], folder=folder) is None
        (folder / "index.json").write_text(json.dumps({"model": "v16-ft1", "created": "2026-11-12 08:00",
                                                       "tags": ["dance", "Blonde", "crowd"], "cutoffs": [50, 50, 70]}))
        times = np.arange(0, 400, 0.5, dtype=np.float32)
        scores = np.zeros((len(times), 3), np.uint8)
        scores[:, 0] = 80                                                  # dance everywhere
        scores[:, 1] = 99                                                  # hair: never shown per frame
        scores[times >= 20, 2] = 70                                        # crowd from 20 s, exactly on its cut-off
        np.savez_compressed(folder / "vid_d.npz", times=times, scores=scores)
        TR._STUDENT_CACHE.clear()
        assert TR.student_tags_at("vid_d", [3.2, 20.1], folder=folder) == [["dance"], ["dance", "crowd"]]
        vids = TR.other_videos(index, known=known, scores_folder=folder)
        assert sorted(vids) == ids and all(d["other"] and d["batch"] == TR.OTHER_BATCH for d in vids.values())
        ms = vids["vid_d"]["moments"]
        assert len(ms) == TLB.TEST_MOMENTS_PER_VIDEO and ms[0]["proxy_frame"] and ms[0]["mine"] is None
        assert all(m["teachers"][TR.STUDENT]["tags"][0] == "dance" for m in ms)
        assert vids["vid_e"]["moments"][0]["teachers"] == {}               # the student hasn't scored this one
        assert TR.moment_id("vid_d", ms[0]) == TLB.card_id("tags", "vid_d", ms[0]["t"])
        assert TR.model_has(ms[0], "dance") and not TR.model_has(ms[0], "Blonde")
    finally:
        TR._STUDENT_CACHE.clear()
        if had_extra is None:
            TLB.EXTRA_TEST_FILE.unlink(missing_ok=True)
        else:
            TLB.EXTRA_TEST_FILE.write_bytes(had_extra)


def test_category_pass_on_the_page(bundle, clean_labels):
    """The review pass: Videos → one set, Category, not reviewed in it, ▦ Grid, 🧹 for that category alone."""
    from tests.apptest_util import errors, page_app
    TLB = clean_labels
    backup = TLB.VOCAB_FILE.read_bytes() if TLB.VOCAB_FILE.exists() else None
    try:
        TLB.save_vocabulary([{"tag": "dance", "group": "action"}, {"tag": "crowd", "group": "action"},
                             {"tag": "night", "group": "place"}, {"tag": "street", "group": "place"}])
        at = page_app("pages/6_Training_Labels.py")
        at.session_state["tl_view"] = "🧪 Trial results"
        at.session_state["tr_sub"] = "🖼️ Compare tags"
        at.run()
        assert not errors(at), errors(at)
        kinds = at.session_state["tr_set"], [str(c.value) for c in at.caption]
        assert kinds[0] == "all" and any("Full labels" in c for c in kinds[1])
        labels = [c.value for c in at.caption]
        assert any("videos done" in c for c in labels)
        cat_box = at.selectbox(key="tr_cat")
        assert list(cat_box.options) == ["(all categories)", "action", "place"]
        at.session_state["tr_set"] = "train"
        at.session_state["tr_cat"] = "action"
        at.session_state["tr_filter"] = "todo"
        at.session_state["tr_view"] = "grid"
        at.run()
        assert not errors(at), errors(at)
        assert at.selectbox(key="tr_g_cat").value == "action" and at.selectbox(key="tr_g_cat").disabled
        assert set(at.selectbox(key="tr_tag").options) <= {"(any tag)", "dance", "crowd"}
        ticks = [c for c in at.checkbox if c.key and c.key.startswith("tr_gt_")]
        n = len(ticks)
        assert n == 2                                                # vid_a's labelled frame is reviewed already
        btn = at.button(key="tr_fincat_btn")
        assert btn.label == f"🧹 Remove all not-reviewed action tags ({n})"
        btn.click().run()
        assert any("Mark **action** as reviewed" in w.value for w in at.warning)
        at.button(key="tr_fincat_yes").click().run()
        assert not errors(at), errors(at)
        parts = [i for i in TLB.load_labels() if i["type"] == "partial"]
        assert len(parts) == n and all(i["done"] == ["action"] and i["yes"] == [] for i in parts)
        assert not [c for c in at.checkbox if c.key and c.key.startswith("tr_gt_")]      # nothing left in action
        assert any("**action**: reviewed on **3** of 3 frames" in c.value for c in at.caption)
        at.session_state["tr_cat"] = "place"                          # the other category is still all to do
        at.run()
        assert len([c for c in at.checkbox if c.key and c.key.startswith("tr_gt_")]) == n
        at.session_state["tr_vid_open"] = True                        # only videos not fully reviewed (in place)
        at.session_state["tr_cat"] = "action"
        at.session_state["tr_filter"] = "all"
        at.run()
        assert not errors(at), errors(at)
        assert list(at.selectbox(key="tr_vid").options) == ["(all videos)"]
        # ↩️ Undo puts the frames back
        at.button(key="tr_g_undo_btn").click().run()
        assert not [i for i in TLB.load_labels() if i["type"] == "partial"]
        # 📚 the other videos are one tap away, and the set counts add up
        at.session_state["tr_vid_open"] = False
        at.session_state["tr_cat"] = "(all categories)"
        at.session_state["tr_set"] = "other"
        at.run()
        assert not errors(at), errors(at)
        assert len(at.selectbox(key="tr_vid").options) == 1 + 4      # vid_c … vid_f: never in a batch
    finally:
        if backup is None:
            TLB.VOCAB_FILE.unlink(missing_ok=True)
        else:
            TLB.VOCAB_FILE.write_bytes(backup)


def test_review_progress_table(bundle, clean_labels):
    """📊 Per video, per category: full labels count everywhere, a 🧹-finished category only in its own column."""
    from tests.apptest_util import errors, page_app
    TLB = clean_labels
    vocab = [dict(r, group_required=False) for r in VOCAB]
    ms = [{"video_id": "v", "t": 5.0, "scene_id": 0}, {"video_id": "v", "t": 9.0, "scene_id": 1}]
    TLB.finish_category(ms[:1], "position", vocab)
    vids = TR.with_saved_labels(
        {"v": {"moments": [{"key": f"k{m['t']}", "t": m["t"], "mine": None, "teachers": {}} for m in ms]},
         "w": {"moments": [{"key": "a", "t": 1.0, "mine": ["Cowgirl"], "teachers": {}},
                           {"key": "b", "t": 2.0, "mine": None, "teachers": {}}]},
         "empty": {"moments": []}}, TLB.load_labels())
    tab = TR.progress_table(vids, ["position", "shot"])
    rows = {r["video"]: r for r in tab["rows"]}
    assert rows["v"] == {"video": "v", "frames": 2, "full": 0, "by": {"position": 1, "shot": 0}}
    assert rows["w"]["full"] == 1 and rows["w"]["by"] == {"position": 1, "shot": 1}
    assert [r["video"] for r in tab["rows"]] == ["v", "w", "empty"]                 # least reviewed first
    assert tab["total"] == {"video": "All", "frames": 4, "full": 1, "by": {"position": 2, "shot": 1}}
    assert TR.progress_table({}, ["x"]) == {"rows": [], "total": {"video": "All", "frames": 0, "full": 0, "by": {"x": 0}}}
    # on the page: the table is there, follows the Videos choice, and a tapped row picks the video
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🖼️ Compare tags"
    at.session_state["tr_set"] = "train"
    at.run()
    assert not errors(at), errors(at)
    assert any("Review progress" in e.label for e in at.expander)
    assert any(c.value.startswith("Everything shown: **All ") and "in 2 videos" in c.value for c in at.caption)
    import trial_view as TV
    import streamlit as st
    st.session_state["tr_prog_table"] = {"selection": {"rows": [1]}}
    TV._pick_from_table(["vid_a", "vid_b"])
    assert st.session_state["tr_vid"] == "vid_b" and st.session_state["tr_page"] == 1
    st.session_state["tr_prog_table"] = {"selection": {"rows": []}}
    TV._pick_from_table(["vid_a"])                                                   # nothing picked: nothing changes
    assert st.session_state["tr_vid"] == "vid_b"
