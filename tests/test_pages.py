"""Every page runs headless without an exception or error message, and the
planner's end-to-end scenarios export exactly the plans in
tests/golden/page_golden.json."""

import json

import pytest

from tests.apptest_util import errors, page_app
from tests.page_scenarios import GOLDEN, SCENARIOS, run_scenario
from tests.golden_cases import plain
from tests.test_golden import _first_diff

PAGES = ["Home.py", "pages/1_Compilation_Planner.py", "pages/2_Reviewer.py",
         "pages/3_Media_Library.py", "pages/4_Server.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = page_app(page).run()
    assert not errors(at), errors(at)


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_planner_scenario(name, golden):
    got = plain(run_scenario(SCENARIOS[name]))
    assert not got["problems"], got["problems"]
    d = _first_diff(golden[name], got, name)
    assert d is None, d
