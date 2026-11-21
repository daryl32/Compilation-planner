"""Shared test set-up: only what every test needs (each test file still installs the test config itself)."""
import pytest


@pytest.fixture(autouse=True)
def _forget_planner_jobs():
    """A finished planner job (🤖 New matcher / 🎛️ Classic render / ⏱️ Edge) stays on the server so a reloaded page
    picks it up — between tests that would hand one test's result to the next, so they are forgotten after each."""
    yield
    import sys
    J = sys.modules.get("mix_jobs")
    if J is None:
        return
    for key in ("compose:job", "classic:render", "edge:job"):
        j = J.get(key)
        if j is not None and not j.active:
            J.forget(key)


def pytest_configure(config):
    config.addinivalue_line("markers", "new_lengths: run with Edge Mode's 21 Nov phase lengths and surprise rules")


@pytest.fixture(autouse=True)
def _classic_edge_lengths(request, monkeypatch):
    """⏱️ Most Edge tests were written for the 20 s phases (2:00 / 3:00 / 5:00 rounds) and the old surprise lengths;
    they check the mechanics, so they keep that geometry. Tests marked new_lengths use the 21 Nov one."""
    if request.node.get_closest_marker("new_lengths"):
        yield
        return
    import random
    import edge_mode as E
    old = {"chill": {"handsoff": (3, 5), "override": (5,)}, "medium": {"handsoff": (5, 8, 12), "override": (5, 10, 15)}}
    monkeypatch.setattr(E, "PHASE_LEN", {})
    monkeypatch.setattr(E, "HARD_ROUND_SEC", 300)
    monkeypatch.setattr(E, "HARD_FINAL_EXTRA", (20, 30))
    monkeypatch.setattr(E, "HARD_LEN", {"slow": (10, 18), "medium": (12, 22), "fast": (15, 28), "extra": (18, 32)})
    monkeypatch.setattr(E, "HARD_SHIFT_ALL", True)
    monkeypatch.setattr(E, "HANDS_OFF_SEC", {d: v["handsoff"] for d, v in old.items()})
    monkeypatch.setattr(E, "OVERRIDE_SEC", {d: v["override"] for d, v in old.items()})
    monkeypatch.setattr(E, "SURPRISES_OPTIONAL", ("chill", "medium"))
    monkeypatch.setattr(E, "surprise_len", lambda kind, d, rng: rng.choice(old[d][kind]))
    monkeypatch.setattr(E, "HANDS_OFF_SHORT_GAP", -1.0)
    yield
