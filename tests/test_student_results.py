"""Student runs in the web app (student_results.py + student_results_view.py): 📋 Summary landing,
🏋️ Training results and 🧪 Test results, notes per run — from records like the laptop writes."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)

import student_results as SRS
import student_runs as SR


def _meta(name, created, labels, f1_test, teacher_f1, train_f1=None, **settings):
    tags = ["close-up", "POV", "kissing"]
    per = {t: {"precision": .8, "recall": .7, "f1": f1_test - .05 * i, "tp": 10 - i, "fp": 2 + i, "fn": 3 + 2 * i}
           for i, t in enumerate(tags)}
    tper = {t: {"precision": .8, "recall": .8, "f1": teacher_f1, "tp": 10, "fp": 2, "fn": 2} for t in tags}
    m = {"name": name, "created": created, "tags": tags, "thresholds": [.5, .45, .6], "label_weight": 5.0, "folds": 5,
         "counts": {"labels": labels, "teacher": 3000, "scene_extra": 4000, "teacher_in_scenes": 300},
         "teacher_trust": {"pov": {"f1": .39, "weight": .39}}, "train_minutes": .2, "videos": 150,
         "reference_run": "try2-base",
         "scores": {"plain": {"f1": f1_test}, "rules": {"f1": f1_test}, "teacher": {"f1": teacher_f1}},
         "test": {"frames": 379, "best": "rules", "scored_at": "2026-11-06 12:50",
                  "scores": {"rules": {"f1": f1_test, "precision": .77, "recall": .74, "fp": 450, "fn": 500, "n": 379},
                             "plain": {"f1": f1_test - .01, "n": 379},
                             "teacher": {"f1": teacher_f1, "precision": .8, "recall": .72, "fp": 370, "fn": 550, "n": 379},
                             "rules_all_frames": {"f1": f1_test, "n": 432}},
                  "per_tag": per, "teacher_per_tag": tper, "all_frames_n": 432},
         "test_first": {"scores": {"rules": {"f1": f1_test + .02}}, "frames": 336}}
    if train_f1 is not None:
        m["training"] = {"frames": 900, "scores": {"f1": train_f1, "precision": .8, "recall": .75, "fp": 100, "fn": 120},
                         "per_tag": {t: {"f1": train_f1, "tp": 50, "fp": 5, "fn": 6} for t in tags},
                         "labels_per_tag": {"close-up": 300, "POV": 40 + labels // 100, "kissing": 12}}
        m["settings"] = {"label_weight": 5.0, "folds": 5, "cutoffs": "plain", "cutoff_range": [.4, .7], **settings}
    return m


@pytest.fixture
def runs(tmp_path):
    models = tmp_path / "models"
    for m in (_meta("v5b", "2026-11-05 14:20", 3300, .759, .759),
              _meta("v10", "2026-11-06 12:17", 3855, .756, .759),
              _meta("v12", "2026-11-06 13:30", 4100, .765, .759, train_f1=.81)):
        (models / m["name"]).mkdir(parents=True)
        (models / m["name"] / "meta.json").write_text(json.dumps(m))
    shutil.rmtree(SRS.RUNS_DIR, ignore_errors=True)
    SR.export_runs(models, SRS.RUNS_DIR, in_use="v5b")
    SRS._CACHE.clear()
    SRS.NOTES_FILE.unlink(missing_ok=True)
    yield
    shutil.rmtree(SRS.RUNS_DIR, ignore_errors=True)
    SRS.NOTES_FILE.unlink(missing_ok=True)
    SRS._CACHE.clear()


def test_index_notes_and_helpers(runs):
    idx = SRS.load_index()
    assert [r["name"] for r in idx["runs"]] == ["v5b", "v10", "v12"]
    assert SRS.best_on_test(idx["runs"]) == "v12"
    assert idx["runs"][-1]["training"]["f1"] == .81 and idx["runs"][0]["training"] is None
    assert idx["runs"][-1]["test"]["first_f1"] == pytest.approx(.785)
    tags, grid = SRS.tag_grid({"v12": SRS.load_detail("v12")}, "train")
    assert set(tags) == {"close-up", "POV", "kissing"} and grid["v12"]["POV"] == .81
    assert SRS.noise_f1(379) == .03 and SRS.noise_f1(5000) == .02
    SRS.save_notes({"student:v12": "tried, keep", "batch:3": "", "test:x": 'has, commas "and" quotes'})
    assert SRS.load_notes() == {"student:v12": "tried, keep", "test:x": 'has, commas "and" quotes'}
    from library_common import read_pending
    assert SRS.NOTES_PENDING in read_pending()
    assert "0.40–0.70" in SRS.settings_text({"cutoffs": "plain", "cutoff_range": [.4, .7], "pose": True})


@pytest.mark.parametrize("sub", ["📋 Summary", "🏋️ Training results", "🧪 Test results"])
def test_pages(runs, sub):
    from tests.test_trial_results import _bundle  # noqa: F401  (needs the trial bundle on disk)
    import trial_results as TR
    from PIL import Image
    shutil.rmtree(TR.RESULTS_DIR, ignore_errors=True)
    (TR.RESULTS_DIR / "frames" / "vid_a").mkdir(parents=True)
    (TR.RESULTS_DIR / "results.json").write_text(json.dumps(_bundle()))
    (TR.RESULTS_DIR / "report.md").write_text("# Modal trial report\n\nHello.")
    Image.new("RGB", (64, 36)).save(TR.RESULTS_DIR / "frames" / "vid_a" / "m1.jpg")
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = sub
    at.run()
    assert not errors(at), errors(at)
    text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    if sub == "📋 Summary":
        assert [m.value for m in at.metric][:1] and "v12" in " ".join(m.label for m in at.metric)
        at.button(key="tr_go_test").click().run()
        assert not errors(at), errors(at)
        assert at.session_state["tr_sub"] == "🧪 Test results"
    elif sub == "🏋️ Training results":
        assert "Hello." in text                              # the teacher batches' written report
        assert at.selectbox(key="srv_train_tag_run").value == "v12"
    else:
        assert at.selectbox(key="srv_test_tag_run").value == "v12"   # 🏆 best opens first
        assert "chance" in text
    shutil.rmtree(TR.RESULTS_DIR, ignore_errors=True)
