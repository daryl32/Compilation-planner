"""core/footage.carve_span: a window from a scene that's no longer in the queue raises ValueError
(which the planner catches), never an UnboundLocalError (live crash, 23 Oct 2026: Manual mode,
planned path, after changing weights)."""

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from core.footage import carve_span


def _span(scene_id, offset=0.0, remaining=10.0):
    return {"video_id": "v", "scene_id": scene_id, "offset_sec": offset, "remaining_sec": remaining,
            "scene_start_sec": 100.0 * scene_id, "tags": [], "thumbnail": None, "motion_norm": 0.5}


@pytest.mark.parametrize("queues", [{}, {"v": []}, {"v": [_span(2)]}])
def test_missing_scene_raises_value_error(queues):
    with pytest.raises(ValueError, match=r"scene 1 covers \[3\.00, 5\.00\]"):
        carve_span(queues, "v", 1, 3.0, 2.0, True)


def test_window_outside_the_span_raises_value_error():
    with pytest.raises(ValueError):
        carve_span({"v": [_span(1, offset=6.0, remaining=4.0)]}, "v", 1, 0.0, 2.0)


def test_carves_a_window():
    q = {"v": [_span(1)]}
    pick = carve_span(q, "v", 1, 3.0, 2.0)
    assert pick["clip_start_sec"] == 103.0 and pick["clip_duration_sec"] == 2.0
    assert [(s["offset_sec"], s["remaining_sec"]) for s in q["v"]] == [(0.0, 3.0), (5.0, 5.0)]
