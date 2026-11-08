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
        assert h == {"colours": ["Blonde", "Redhead"], "source": "you", "unsure": False, "vote": ["Brunette"]}
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
        key = f"vd_hair_save_{stem}"
        assert at.button(key=key).label == "✅ Confirm hair"
        at.button(key=key).click().run()                          # vote right → one tap
        assert not errors(at), errors(at)
        assert LC.hair_of(vid)["source"] == "you" and LC.hair_of(vid)["colours"] == ["Brunette"]
        assert at.button(key=key).label == "✅ Confirmed" and at.button(key=key).disabled
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
