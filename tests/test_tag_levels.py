"""claude/tag-levels.md: F1 by tag group, 🚫 nobody-on-screen suggestions, 👥 cast per video."""
from __future__ import annotations

import json
import shutil

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)


def test_text_on_screen_out_of_f1_and_group_scores():
    import student_results as SRS
    import trial_logic as TL
    assert TL.not_in_f1("Text On Screen") and not TL.not_in_f1("Cowgirl position")
    per = {"Cowgirl position": {"tp": 8, "fp": 2, "fn": 0}, "Blowjob": {"tp": 5, "fp": 0, "fn": 5},
           "lingerie": {"tp": 1, "fp": 1, "fn": 0}, "text on screen": {"tp": 0, "fp": 50, "fn": 0},
           "medium shot": {"tp": 0, "fp": 9, "fn": 9}, "Blonde": {"tp": 0, "fp": 4, "fn": 0}}
    groups = SRS.group_of_tags([{"tag": "Cowgirl position", "group": "Position"},
                                {"tag": "Blowjob", "group": "action"}, {"tag": "lingerie", "group": "clothing"},
                                {"tag": "text on screen", "group": "overlay"}, {"tag": "medium shot", "group": "shot"},
                                {"tag": "Blonde", "group": "appearance"}])
    sc = SRS.group_scores(per, groups)
    assert set(sc["groups"]) == {"position", "action", "clothing"}
    assert sc["importance"]["High"]["tp"] == 13 and sc["importance"]["High"]["fp"] == 2
    assert sc["importance"]["Medium"]["f1"] == 0.667


def test_people_at_from_detections(tmp_path):
    import student_tags as S
    frames = np.array([0, 0, 1, 1, 2, 2] + [f for f in range(26, 35) for _ in range(3)], np.int32)   # 2 at 0 s, 3 at 2 s
    np.savez_compressed(tmp_path / "detections.npz", frame=frames, n_frames=np.int32(60))
    p = S.people_at(tmp_path, np.array([0.0, 1.0, 2.0, 3.5]))
    assert list(p) == [2, 0, 3, 0]
    assert S.people_at(tmp_path / "nope", np.array([0.0])) is None


def test_nobody_scenes_and_hint():
    import footage_people as FP
    times = np.arange(0, 60, 0.5, dtype=np.float32)
    people = np.where(times < 10, 0, 2).astype(np.int32)        # title card for the first 10 s
    people[(times >= 40) & (times < 41)] = 0                     # a 1 s gap inside a scene: not junk
    scenes = [{"scene_id": 1, "start": 0, "end": 10}, {"scene_id": 2, "start": 10, "end": 50},
              {"scene_id": 3, "start": 50, "end": 60, "excluded": True}]
    assert FP.nobody_scenes(scenes, times, people) == [1]
    assert FP.nobody_scenes(scenes, times, people, [FP.dismiss_key(1)]) == []
    assert FP.nobody_scenes(scenes, None, None) == []
    assert FP.people_hint(scenes, times, people) == 2


def test_cast_helpers():
    import library_common as LC
    before = LC.load_library_meta()
    try:
        LC.set_video_cast("vid_a", 2, 0)
        LC.set_video_cast("vid_b", 1, 7)                              # capped at 4+
        LC.set_video_details("vid_a", actors=["Ann"])
        cast = LC.video_cast()
        assert cast == {"vid_a": {"women": 2, "men": 0}, "vid_b": {"women": 1, "men": 4}}
        assert LC.matches_cast("vid_a", cast, girl_girl=True) and not LC.matches_cast("vid_b", cast, girl_girl=True)
        assert LC.matches_cast("vid_b", cast, men=4) and LC.matches_cast("vid_b", cast, women=1, men="Any")
        assert not LC.matches_cast("vid_c", cast, women=1) and LC.matches_cast("vid_c", cast)
        assert LC.cast_label(cast["vid_a"]) == "👩 2 · 👨 0 · 👭 girl / girl"
        LC.set_video_cast("vid_a", None, None)
        assert "vid_a" not in LC.video_cast() and LC.video_details()["vid_a"]["actors"] == ["Ann"]
    finally:
        LC.save_library_meta(before)


def test_nobody_choice_in_catalogue():
    import footage_people as FP
    cat = {"scenes": [{"scene_id": 1}, {"scene_id": 2}, {"scene_id": 3}]}
    assert FP.apply_choice(cat, [1, 2], [1]) == 1
    assert cat["scenes"][0]["excluded"] and cat["scenes"][0]["reviewed"] and not cat["scenes"][1].get("excluded")
    assert cat["dismissed_edit_suggestions"] == ["nobody:2"]
    assert FP.nobody_scenes([{"scene_id": 2, "start": 0, "end": 5}], np.arange(0, 5, .5), np.zeros(10, int),
                            cat["dismissed_edit_suggestions"]) == []


def _export_people(folder, vids, people_fn):
    folder.mkdir(parents=True, exist_ok=True)
    times = np.arange(0, 120, 0.5, dtype=np.float32)
    for vid in vids:
        np.savez_compressed(folder / f"{vid}.npz", times=times, scores=np.zeros((len(times), 3), np.uint8),
                            people=people_fn(times).astype(np.uint8))
    (folder / "index.json").write_text(json.dumps({"model": "v14", "created": "x", "step": 0.5,
                                                   "tags": ["Blonde", "Brunette", "Redhead"], "cutoffs": [50] * 3}))


def test_reviewer_cast_and_nobody_suggestions():
    import catalogue_store as CS
    import hair_vote as HV
    import library_common as LC
    from tests.apptest_util import errors, page_app
    folder = HV.scores_dir()
    assert not folder.exists()
    before = LC.load_library_meta()
    backups = {p: p.read_text() for p in CS.catalogue_paths()}
    try:
        at = page_app("pages/2_Reviewer.py", timeout=120)
        at.run()
        stem = at.session_state["rev_video"]
        cat = CS.load_catalogue(stem)
        vid = cat["video_id"]
        first = min(cat["scenes"], key=lambda s: LC.tc_to_seconds(s["start_tc"]))
        a, b = LC.tc_to_seconds(first["start_tc"]), LC.tc_to_seconds(first["end_tc"])
        _export_people(folder, [vid], lambda t: np.where((t >= a) & (t < b), 0, 2))
        at.run()
        assert not errors(at), errors(at)
        # 👥 cast: pick and save
        k = f"{stem}_none"
        at.button_group(key=f"vd_women_{k}").set_value(2).run()
        at.button_group(key=f"vd_men_{k}").set_value(0).run()
        at.button(key=f"vd_cast_save_{stem}").click().run()
        assert not errors(at), errors(at)
        assert LC.video_cast()[vid] == {"women": 2, "men": 0}
        assert any("up to **2** people" in c.value for c in at.caption)
        # 🚫 nobody on screen: the To do bar offers it, the dialog excludes it
        btn = [x for x in at.button if x.key == f"nobody_{stem}"]
        assert btn, [x.key for x in at.button][:30]
        btn[0].click().run()
        assert not errors(at), errors(at)
        # (AppTest can't press a button inside a dialog — the choice itself is tested below)
        assert [x for x in at.button if x.label.startswith("🚫 Exclude 1 · keep 0")]
        # 👥 filters
        at.multiselect(key="rev_find_flags").set_value(["👥 No cast yet"]).run()
        assert not errors(at), errors(at)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        LC.save_library_meta(before)
        for p, text in backups.items():
            p.write_text(text)



def test_planner_and_library_cast_filters():
    import library_common as LC
    from tests.apptest_util import errors, page_app
    from tests.test_ui_simple import PLANNER
    before = LC.load_library_meta()
    try:
        LC.set_video_cast("vid_a", 2, 0)
        at = page_app(PLANNER, advanced=False)
        at.session_state["track_id"] = "song_full"
        at.run()
        steps = at.radio(key="ui_section").options
        at.radio(key="ui_section").set_value([s for s in steps if "Videos" in s][0]).run()
        at.toggle(key="vid_gg_filter").set_value(True).run()
        assert not errors(at), errors(at)
        assert any(c.value.startswith("👤🏷️💇 1 of") for c in at.caption)
        lib = page_app("pages/3_Media_Library.py", timeout=300)
        lib.session_state["lib_kind"] = "🎬 Videos"
        lib.run()
        lib.selectbox(key="lib_women").set_value(2).run()
        assert not errors(lib), errors(lib)
        text = " ".join(m.value for m in lib.markdown) + " ".join(c.value for c in lib.caption)
        assert "vid_a" in text and "vid_b" not in text
    finally:
        LC.save_library_meta(before)
