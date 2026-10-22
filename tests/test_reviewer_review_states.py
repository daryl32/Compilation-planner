"""Reviewer: 🏷️ tagged and 🎭 role marked are separate review states (filters on both
levels), every role as a button in quick marking with Save / page buttons at the bottom,
one target length for split suggestions + ⚡ Auto split, and 🔗 Auto combine."""

import shutil
import time

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)

PAGE = "pages/2_Reviewer.py"


@pytest.fixture
def restore_catalogues(tmp_path):
    import catalogue_store as CStore
    backups = {p: tmp_path / p.name for p in CStore.catalogue_paths()}
    for p, b in backups.items():
        shutil.copy2(p, b)
    yield
    time.sleep(0.01)
    for p, b in backups.items():
        shutil.copy2(b, p)
        p.touch()


def _app(video: str):
    from tests.apptest_util import page_app
    at = page_app(PAGE, timeout=120)
    at.query_params["video"] = video
    return at


def _scenes(stem: str) -> list:
    import catalogue_store as CStore
    return CStore.load_catalogue(stem)["scenes"]


def test_capped_auto_split_never_leaves_a_long_part():
    import scene_edits as SE
    for length in (31.0, 59.9, 95.0, 240.0):
        scene = {"scene_id": 1, "start_tc": SE.seconds_to_tc(0), "end_tc": SE.seconds_to_tc(length)}
        cuts = SE.auto_split_points(scene, 30, cap=True)
        bounds = [0.0] + cuts + [SE.scene_duration(scene)]
        assert cuts and max(b - a for a, b in zip(bounds, bounds[1:])) <= 30 + 1e-9
    # the old (uncapped) behaviour is unchanged for everyone else
    scene = {"scene_id": 1, "start_tc": SE.seconds_to_tc(0), "end_tc": SE.seconds_to_tc(95)}
    assert len(SE.auto_split_points(scene, 30)) == 2


def test_auto_combine_groups_respect_the_target():
    import scene_edits as SE
    scenes = [{"scene_id": i, "start_tc": SE.seconds_to_tc(i * 1.5), "end_tc": SE.seconds_to_tc(i * 1.5 + 1.5)}
              for i in range(10)]
    groups = SE.auto_combine_groups(scenes, [list(range(10))], max_sec=5)
    assert all(len(g) >= 2 for g in groups)
    assert all(sum(SE.scene_duration(scenes[i]) for i in g) <= 5 + 1e-9 for g in groups)
    assert [i for g in groups for i in g] == sorted(i for g in groups for i in g)


def test_find_flags_and_scene_filters_run(restore_catalogues):
    from tests.apptest_util import errors
    at = _app("vid_a")
    at.run()
    assert not errors(at), errors(at)
    flags = at.multiselect(key="rev_find_flags")
    for f in ("🏷️ Not fully tagged", "🎭 Not fully role marked", "🆕 Added in the last 3 days"):
        assert f in flags.options
    flags.set_value(["🏷️ Not fully tagged", "🎭 Not fully role marked"]).run()
    assert not errors(at), errors(at)
    at.session_state["rev_only"] = ["Not tagged", "Not role marked"]
    at.run()
    assert not errors(at), errors(at)


def test_quick_marking_role_buttons_mark_roles_not_tags(restore_catalogues):
    from tests.apptest_util import errors
    at = _app("vid_a")
    at.run()
    at.session_state["rev_quick_roles"] = True
    at.run()
    assert not errors(at), errors(at)
    btns = [g for g in at.button_group if (g.key or "").startswith("rev_qr_p_vid_a_")]
    assert btns
    sid = int(btns[0].key[len("rev_qr_p_vid_a_"):].split("_")[0])
    before = next(s for s in _scenes("vid_a") if s["scene_id"] == sid)
    btns[0].set_value(["action", "slow"]).run()
    assert not errors(at), errors(at)
    at.button(key="rev_qr_save_bottom").click().run()     # the Save at the bottom of the grid
    assert not errors(at), errors(at)
    after = next(s for s in _scenes("vid_a") if s["scene_id"] == sid)
    assert set(after.get("roles") or []) >= {"action", "slow"}
    assert after.get("roles_reviewed") is True
    assert bool(after.get("reviewed")) == bool(before.get("reviewed"))   # tags state untouched


def test_auto_combine_from_the_to_do_bar(restore_catalogues):
    """The To do bar offers 🔗 Auto combine and its dialog lists the combines. (AppTest can't
    press a button inside a dialog — in the browser a dialog reruns on its own.)"""
    from tests.apptest_util import errors
    at = _app("vid_c")
    at.session_state["rev_short_sec"] = 3.5
    at.run()
    assert not errors(at), errors(at)
    at.button(key="bulk_combine_vid_c").click().run()
    assert not errors(at), errors(at)
    assert [b for b in at.button if b.label.startswith("🔗 Combine into")]
    assert any("short clips" in m.value for m in at.markdown)


def test_quick_marking_exclude_counts_as_role_marked(restore_catalogues):
    from tests.apptest_util import errors
    at = _app("vid_f")
    at.run()
    at.session_state["rev_quick_roles"] = True
    at.run()
    assert not errors(at), errors(at)
    btn = [b for b in at.button if (b.key or "").startswith("rev_qr_x_vid_f_")][0]
    sid = int(btn.key[len("rev_qr_x_vid_f_"):])
    assert not next(s for s in _scenes("vid_f") if s["scene_id"] == sid).get("excluded")
    btn.click().run()
    assert not errors(at), errors(at)
    at.button(key="rev_qr_save_bottom").click().run()
    assert not errors(at), errors(at)
    sc = next(s for s in _scenes("vid_f") if s["scene_id"] == sid)
    assert sc["excluded"] is True
    # an excluded scene no longer shows under "Not role marked"
    at.session_state["rev_quick_roles"] = False
    at.session_state["rev_only"] = ["Not role marked"]
    at.run()
    assert not errors(at), errors(at)
    assert not [w for w in at.multiselect if w.key == f"tags_vid_f_{sid}"]
