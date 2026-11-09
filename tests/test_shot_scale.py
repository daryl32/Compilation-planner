"""🎥 Shot size as one scale (extreme close-up → close-up → medium → wide), left out of the overall F1."""
from __future__ import annotations

from tests.support import DATA  # noqa: F401  (installs the test config first)


def test_shot_step_any_spelling():
    import trial_logic as TL
    assert [TL.shot_step(t) for t in ("extreme close-up", "close-up shot", "Close-Up", "medium shot",
                                      "Wide Shot", "POV")] == [0, 1, 1, 2, 3, None]
    assert TL.not_in_f1("medium shot") and TL.not_in_f1("Blonde") and not TL.not_in_f1("Blowjob")


def test_score_shot_and_f1_without_shot():
    import trial_logic as TL
    pairs = [(["close-up shot", "Blowjob"], ["close-up shot", "Blowjob"]),     # exact
             (["medium shot"], ["close-up shot"]),                              # one step
             (["wide shot"], ["close-up shot"]),                                # big miss
             ([], ["extreme close-up"]),                                        # none given
             (["wide shot"], ["Blowjob"])]                                      # no shot label → not scored
    s = TL.score_shot(pairs)
    assert s == {"frames": 4, "exact": 0.25, "within_one": 0.5, "big_miss": 0.25, "none_given": 0.25,
                 "big_miss_n": 1}
    sc = TL.score_tags(pairs, ["Blowjob", "extreme close-up", "close-up shot", "medium shot", "wide shot"])
    assert sc["overall"]["fp"] == 0 and sc["overall"]["fn"] == 1 and sc["shot"]["frames"] == 4
    assert set(sc["left_out"]) == {"extreme close-up", "close-up shot", "medium shot", "wide shot"}
    assert sc["per_tag"]["wide shot"]["fp"] == 2                       # per-tag scores still kept


def test_run_line_carries_shot():
    import student_runs as SR
    meta = {"name": "v14", "test": {"frames": 3, "best": "rules", "scores": {"rules": {"f1": .8, "n": 3}},
                                    "shot": {"rules": {"frames": 3, "within_one": .9}, "teacher": {"frames": 3}}}}
    line = SR.run_line(meta, "", False)
    assert line["test"]["shot"]["within_one"] == .9 and line["test"]["teacher_shot"] == {"frames": 3}
