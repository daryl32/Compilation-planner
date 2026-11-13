"""💇 Hair colour per video (claude/hair-colour-plan.md): Daryl's answer, the model's
vote, the to-confirm lists and picking by hair."""
from __future__ import annotations

from tests.support import DATA  # noqa: F401  (installs the test config first)


def _keep_meta():
    import library_common as LC
    return LC.load_library_meta()


def test_answer_wins_over_vote_and_clearing_shows_vote_again():
    import library_common as LC
    before = _keep_meta()
    try:
        LC.set_hair_votes({"vid_a": {"colours": ["brunette"], "scores": {"Brunette": 61}, "model": "v13"}})
        h = LC.hair_of("vid_a")
        assert h["colours"] == ["Brunette"] and h["source"] == "vote"
        assert LC.hair_needs_confirming("vid_a") and not LC.hair_unsure("vid_a")
        LC.set_video_hair("vid_a", ["Redhead", "blonde", "Purple"])      # order + unknowns tidied
        h = LC.hair_of("vid_a")
        assert (h["colours"], h["source"], h["vote"], h["vote_model"]) == (["Blonde", "Redhead"], "you", ["Brunette"], "v13")
        assert not LC.hair_needs_confirming("vid_a")
        # A new vote never overwrites the answer.
        LC.set_hair_votes({"vid_a": {"colours": ["Blonde"], "unsure": True}})
        assert LC.hair_of("vid_a")["colours"] == ["Blonde", "Redhead"]
        assert not LC.hair_unsure("vid_a")
        LC.set_video_hair("vid_a", [])
        h = LC.hair_of("vid_a")
        assert h["source"] == "vote" and h["unsure"] and LC.hair_unsure("vid_a")
        # Other details on the same entry survive.
        LC.set_video_details("vid_a", actors=["Ann"])
        LC.set_hair_votes({"vid_a": None})
        assert LC.hair_of("vid_a")["source"] is None
        assert LC.video_details()["vid_a"]["actors"] == ["Ann"]
    finally:
        LC.save_library_meta(before)


def test_matches_hair_includes_and_only():
    import library_common as LC
    hair = {"one": {"colours": ["Blonde"], "source": "you", "unsure": False, "vote": []},
            "two": {"colours": ["Blonde", "Brunette"], "source": "vote", "unsure": False, "vote": []}}
    assert LC.matches_hair("none", hair, [])                     # no filter → everything
    assert not LC.matches_hair("none", hair, ["Blonde"])         # no hair known → left out
    assert LC.matches_hair("two", hair, ["Blonde"])
    assert not LC.matches_hair("two", hair, ["Blonde"], only=True)
    assert LC.matches_hair("one", hair, ["blonde"], only=True)
    assert LC.matches_hair("two", hair, ["Blonde", "Brunette"], only=True)


def test_reviewer_confirm_hair_and_find_flags():
    import library_common as LC
    from tests.apptest_util import errors, page_app
    before = _keep_meta()
    try:
        at = page_app("pages/2_Reviewer.py", timeout=120)
        at.run()
        assert not errors(at), errors(at)
        stem = at.session_state["rev_video"]
        import catalogue_store as CS
        vid = CS.load_catalogue(stem)["video_id"]
        LC.set_hair_votes({vid: {"colours": ["Brunette"]}})
        at.run()
        key = f"vd_save_{stem}"                                   # the page's one 💾 Save
        assert any("confirms the vote" in c.value for c in at.caption)
        at.button(key=key).click().run()                          # vote right → one tap
        assert not errors(at), errors(at)
        assert LC.hair_of(vid)["source"] == "you" and LC.hair_of(vid)["colours"] == ["Brunette"]
        assert not any("confirms the vote" in c.value for c in at.caption)
        # 💇 Hair not confirmed no longer lists it; ❓ lists only unsure votes.
        at.multiselect(key="rev_find_flags").set_value(["💇 Hair not confirmed"]).run()
        assert not errors(at), errors(at)
        def shown():
            return [o.split("  ·  ")[0] for o in at.selectbox(key="rev_video").options]
        assert shown() and stem not in shown()
        others = [st for st in CS.video_index() if st != stem]
        other_vid = CS.load_catalogue(others[0])["video_id"]
        LC.set_hair_votes({other_vid: {"colours": [], "unsure": True}})
        at.multiselect(key="rev_find_flags").set_value(["❓ Hair to confirm (vote unsure)"]).run()
        assert not errors(at), errors(at)
        assert shown() == [others[0]]
        # 🗂️ With hair colour (More filters): only the confirmed Brunette video.
        at.multiselect(key="rev_find_flags").set_value([]).run()
        at.multiselect(key="rev_find_hair").set_value(["Brunette"]).run()
        assert not errors(at), errors(at)
        assert stem in shown()      # (the open video stays listed too, as with the other filters)
    finally:
        LC.save_library_meta(before)


# ---------------------------------------------------------------------------
# 🤖 The vote (hair_vote.py)
# ---------------------------------------------------------------------------

def _scores(*blocks):
    """blocks of (seconds, (blonde, brunette, redhead) 0–100) → times, scores."""
    import numpy as np
    rows = []
    for sec, val in blocks:
        rows += [val] * int(sec / 0.5)
    times = np.arange(len(rows), dtype=np.float32) * 0.5
    return times, np.asarray(rows, np.uint8)


CUTS = None


def _cuts():
    import numpy as np
    return np.asarray([0.5, 0.5, 0.5], np.float32)


def test_vote_lighting_flips_stay_one_colour():
    import hair_vote as HV
    # Blonde mostly, short brunette-looking flips scattered through one long scene.
    blocks = []
    for _ in range(10):
        blocks += [(25, (80, 30, 10)), (4, (35, 70, 10))]
    t, s = _scores(*blocks)
    v = HV.vote_video(t, s, _cuts())
    assert v["colours"] == ["Blonde"] and not v["unsure"]


def test_vote_second_actress_needs_whole_scenes():
    import hair_vote as HV
    t, s = _scores((120, (80, 20, 10)), (60, (20, 85, 10)))
    spans = [(0, 120), (120, 180)]
    v = HV.vote_video(t, s, _cuts(), spans)
    assert v["colours"] == ["Blonde", "Brunette"] and not v["unsure"]
    # The same brunette time in one scene with the blonde → she never wins a scene → lighting.
    v = HV.vote_video(t, s, _cuts(), [(0, 180)])
    assert v["colours"] == ["Blonde"]


def test_vote_unsure_and_not_seen():
    import hair_vote as HV
    t, s = _scores((60, (62, 60, 10)))
    v = HV.vote_video(t, s, _cuts())
    assert v["unsure"] and v["colours"] == ["Blonde"]          # too close to call
    t, s = _scores((60, (10, 12, 5)))                           # no hair visible anywhere
    v = HV.vote_video(t, s, _cuts())
    assert v["unsure"] and v["colours"] == []
    # Excluded scenes / outside the library range don't count.
    t, s = _scores((60, (10, 80, 5)), (60, (80, 10, 5)))
    v = HV.vote_video(t, s, _cuts(), HV.usable_spans(
        [{"start": 0, "end": 60, "excluded": True}, {"start": 60, "end": 120}]))
    assert v["colours"] == ["Blonde"]


def test_vote_reads_scores_relative_to_cutoffs():
    import numpy as np
    import hair_vote as HV
    # Redhead's cut-off is low (shy tag): 45 is well over it, 55 blonde is barely at its 0.6 cut-off.
    t, s = _scores((60, (55, 10, 45)))
    v = HV.vote_video(t, s, np.asarray([0.6, 0.5, 0.3], np.float32))
    assert v["colours"] == ["Redhead"]


def test_vote_all_reads_export_and_saves(tmp_path):
    import json
    import numpy as np
    import catalogue_store as CS
    import hair_vote as HV
    import library_common as LC
    idx = CS.video_index()
    vids = [v["video_id"] for v in idx.values()]
    t, s = _scores((60, (80, 20, 10)))
    other = np.zeros((len(t), 2), np.uint8)
    for vid in vids[:1]:
        np.savez_compressed(tmp_path / f"{vid}.npz", times=t, scores=np.hstack([other, s]))
    (tmp_path / "index.json").write_text(json.dumps({
        "model": "v14", "created": "2026-11-08 22:00", "step": 0.5,
        "tags": ["Face", "POV", "Blonde", "Brunette", "Redhead"], "cutoffs": [50, 50, 50, 50, 50]}))
    votes, problem = HV.vote_all({k: {**v, "scenes": []} for k, v in idx.items()}, {}, tmp_path)
    assert problem is None and list(votes) == vids[:1]
    assert votes[vids[0]]["colours"] == ["Blonde"] and votes[vids[0]]["model"] == "v14"
    before = LC.load_library_meta()
    try:
        assert HV.stale(HV.load_index(tmp_path), LC.video_hair())
        LC.set_hair_votes(votes)
        assert not HV.stale(HV.load_index(tmp_path), LC.video_hair())
        assert LC.hair_of(vids[0])["colours"] == ["Blonde"]
    finally:
        LC.save_library_meta(before)
    assert HV.vote_all(idx, {}, tmp_path / "nothing")[1]       # no export → a message, no crash


def test_reviewer_vote_button_then_confirm():
    import json
    import shutil
    import numpy as np
    import catalogue_store as CS
    import hair_vote as HV
    import library_common as LC
    from tests.apptest_util import errors, page_app
    folder = HV.scores_dir()
    assert not folder.exists()
    before = LC.load_library_meta()
    try:
        folder.mkdir(parents=True)
        t, s = _scores((60, (20, 80, 10)))
        vids = [v["video_id"] for v in CS.video_index().values()]
        for vid in vids:
            np.savez_compressed(folder / f"{vid}.npz", times=t, scores=s)
        (folder / "index.json").write_text(json.dumps({
            "model": "v14", "created": "2026-11-08 22:00", "step": 0.5,
            "tags": ["Blonde", "Brunette", "Redhead"], "cutoffs": [50, 50, 50]}))
        at = page_app("pages/2_Reviewer.py", timeout=120)
        at.run()
        assert not errors(at), errors(at)
        at.button(key="rev_hair_vote").click().run()
        assert not errors(at), errors(at)
        assert all(LC.hair_of(v)["colours"] == ["Brunette"] for v in vids)
        assert not [b for b in at.button if b.key == "rev_hair_vote"]     # notice gone once voted
        stem = at.session_state["rev_video"]
        at.button(key=f"vd_save_{stem}").click().run()               # 💾 confirms the vote in one tap
        assert LC.hair_of(CS.load_catalogue(stem)["video_id"])["source"] == "you"
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        LC.save_library_meta(before)


def test_reviewer_test_set_filter(monkeypatch):
    import catalogue_store as CS
    import trial_results as TRR
    from tests.apptest_util import errors, page_app
    stems = list(CS.video_index())
    vid = CS.load_catalogue(stems[0])["video_id"]
    at = page_app("pages/2_Reviewer.py", timeout=120)
    at.run()
    orig = TRR.test_set_videos
    TRR.test_set_videos = lambda extra=True: {vid}
    try:
        at.multiselect(key="rev_find_flags").set_value(["🧪 In the test set"]).run()
        assert not errors(at), errors(at)
        assert [o.split("  ·  ")[0] for o in at.selectbox(key="rev_video").options] == [stems[0]]
    finally:
        TRR.test_set_videos = orig


def test_hair_left_out_of_overall_f1():
    import trial_logic as TL
    pairs = [(["Blowjob", "Blonde"], ["Blowjob", "Brunette"])]
    s = TL.score_tags(pairs, ["Blowjob", "Blonde", "Brunette", "Redhead"])
    assert s["overall"]["f1"] == 1.0 and s["overall"]["fp"] == 0
    assert s["per_tag"]["Blonde"]["fp"] == 1 and s["per_tag"]["Brunette"]["fn"] == 1
    assert set(s["left_out"]) == {"Blonde", "Brunette", "Redhead"}


def test_hair_test_check():
    import hair_vote as HV
    y = lambda colours, vote, unsure=False, sc=None: {"colours": colours, "source": "you", "vote": vote,
                                                       "vote_model": "v13", "vote_unsure": unsure,
                                                       "vote_scores": sc or {}}
    hair = {"a": y(["Blonde"], ["Blonde"]),
            "b": y(["Blonde", "Brunette"], ["Blonde"], sc={"Blonde": 1.4, "Brunette": 0.9, "Redhead": 0.2}),
            "c": y(["Redhead"], ["Brunette"], unsure=True, sc={"Brunette": 1.0, "Redhead": 0.95, "Blonde": 0.1}),
            "d": {"colours": ["Blonde"], "source": "vote", "vote": ["Blonde"], "vote_model": "v13"},
            "e": {"colours": ["Blonde"], "source": "you", "vote": []}}
    c = HV.test_check({"a", "b", "c", "d", "e", "f"}, hair)
    assert (c["videos"], c["confirmed"], c["checked"], c["right"], c["main_right"], c["unsure"], c["no_vote"]) \
        == (6, 4, 3, 1, 2, 1, 1)
    assert [w["video_id"] for w in c["wrong"]] == ["b", "c"] and c["models"] == ["v13"]


# ---------------------------------------------------------------------------
# Hair hidden from frame labelling; hair filters in the planner and Media Library
# ---------------------------------------------------------------------------

def test_hair_hidden_from_chips_but_kept_on_save():
    import training_labels as TLB
    rows = [{"tag": t, "group": g, "one_of": ""} for t, g in
            (("Blonde", "appearance"), ("Brunette", "appearance"), ("lingerie", "appearance"), ("POV", "camera"))]
    shown = [t for _, _, tags in TLB.chip_groups(rows, extra=["Redhead"]) for t in tags]
    assert "Blonde" not in shown and "Redhead" not in shown and "lingerie" in shown
    assert "Blonde" in [t for _, _, tags in TLB.chip_groups(rows, hide=False) for t in tags]
    assert TLB.keep_hidden(["POV"], ["Blonde", "lingerie", "POV"]) == ["POV", "Blonde"]


def test_finish_partials_keeps_model_hair():
    import training_labels as TLB
    before = TLB.load_labels()
    try:
        TLB.batch_tag([{"video_id": "vid_a", "scene_id": 1, "t": 12.0, "label_id": "tags:vid_a:12.00"}], add=["POV"])
        res = TLB.finish_partials([{"video_id": "vid_a", "scene_id": 1, "t": 12.0,
                                    "suggested": ["Blonde", "Blowjob", "POV"]}], [])
        assert res["done"] == 1
        lab = [i for i in TLB.load_labels() if i.get("type") == "tags" and i["video_id"] == "vid_a"
               and float(i["t"]) == 12.0][0]
        assert lab["tags"] == ["POV", "Blonde"]               # only your yes tags + the model's hair kept
    finally:
        TLB._write_labels(before)


def test_planner_and_library_hair_filters():
    import library_common as LC
    from tests.apptest_util import errors, page_app
    before = LC.load_library_meta()
    try:
        LC.set_video_hair("vid_a", ["Redhead"])
        at = page_app("pages/3_Media_Library.py", timeout=300)
        at.session_state["lib_kind"] = "🎬 Videos"
        at.run()
        assert not errors(at), errors(at)
        at.multiselect(key="lib_hair").set_value(["Redhead"]).run()
        assert not errors(at), errors(at)
        text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        assert "vid_a" in text and "vid_b" not in text
    finally:
        LC.save_library_meta(before)


def test_planner_videos_hair_filter():
    import library_common as LC
    from tests.apptest_util import errors, page_app
    from tests.test_ui_simple import PLANNER
    before = LC.load_library_meta()
    try:
        LC.set_video_hair("vid_a", ["Redhead"])
        at = page_app(PLANNER, advanced=False)
        at.session_state["track_id"] = "song_full"
        at.run()
        steps = at.radio(key="ui_section").options
        at.radio(key="ui_section").set_value([s for s in steps if "Videos" in s][0]).run()
        at.multiselect(key="vid_hair_filter").set_value(["Redhead"]).run()
        assert not errors(at), errors(at)
        assert any(c.value.startswith("👤🏷️💇 1 of") for c in at.caption)
    finally:
        LC.save_library_meta(before)
