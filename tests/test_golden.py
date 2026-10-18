"""The planner's cutting, shape matching and Auto matching must give exactly
the same results as the snapshot in tests/golden/planner_golden.json (taken
from the code before the core/ refactor). A failure names the first case
that changed and where inside it."""

import json

import pytest

from tests.capture_golden import GOLDEN
from tests.golden_cases import run_all
from tests.support import PLANNER, load_page


def _first_diff(a, b, path="") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} → {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: {'added' if k not in a else 'missing'}"
            d = _first_diff(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} → {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} → {b!r}"


@pytest.fixture(scope="module")
def results():
    return run_all(load_page(PLANNER))


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def test_same_cases(results, golden):
    assert set(results) == set(golden)


@pytest.mark.parametrize("prefix", ["seg/", "density/", "bars/", "states/", "legacy/", "clips_for",
                                    "autotune/", "best_window/", "shape_score/", "resample/", "interp",
                                    "block_curve/", "queues/", "queue_ops/", "candidate_spans/",
                                    "shape_cands/", "match/"])
def test_matches_golden(results, golden, prefix):
    keys = [k for k in golden if k.startswith(prefix)]
    assert keys, prefix
    for k in keys:
        d = _first_diff(golden[k], results[k], k)
        assert d is None, d


def test_matcher_entry_points_match_golden():
    """Manual ranking + Auto-fill, Choreography windows, role rules and chain
    filling — snapshot taken before the matcher moved to core/."""
    from tests.capture_matcher_golden import GOLDEN as MG
    from tests.golden_cases import matcher_cases
    golden = json.loads(MG.read_text())
    got = matcher_cases(load_page(PLANNER))
    assert set(got) == set(golden)
    for k in golden:
        d = _first_diff(golden[k], got[k], k)
        assert d is None, d
