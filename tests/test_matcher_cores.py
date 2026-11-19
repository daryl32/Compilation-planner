"""⚡ The matcher over every core (19 Nov): the plan is exactly the one a single core makes."""
from __future__ import annotations

import json
import shutil

from tests.edge_support import VIDS, _timelines


def test_every_core_makes_the_same_plan(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import core.story_matcher as SMX
        import planner.compose as PC
        import planner.edge as PE
        monkeypatch.setattr(SMX, "PAR_MIN_VIDEOS", 2)
        used = []
        real = SMX._Scorer.candidates
        monkeypatch.setattr(SMX._Scorer, "candidates",
                            lambda self, jobs, p: (used.append(self.pool is not None), real(self, jobs, p))[1])

        def plans():
            a = PC.compile_plan("song_full", VIDS, movement_only=True, cuts="lines")["plan"]["timeline"]
            b = PE.compile_edge(["song_full"], VIDS, "chill", 3, energy_mix=True)["plan"]["timeline"]
            return json.dumps([a, b], sort_keys=True, default=str)
        monkeypatch.setenv("PLANNER_ONE_CORE", "1")
        one = plans()
        assert not any(used)
        monkeypatch.delenv("PLANNER_ONE_CORE")
        if SMX._par_workers() < 2:
            return                                         # (a one-core machine: nothing to compare)
        used.clear()
        assert plans() == one and any(used)
        assert SMX._FORKED_VIDEOS is None                  # the workers are stopped when the plan is done
    finally:
        shutil.rmtree(out, ignore_errors=True)
