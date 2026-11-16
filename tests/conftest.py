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
