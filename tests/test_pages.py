"""Every page runs headless without an exception or error message, and the
planner's end-to-end scenarios export exactly the plans in
tests/golden/page_golden.json."""

import json

import tempfile

import pytest

from tests.apptest_util import errors, page_app
from tests.page_scenarios import GOLDEN, SCENARIOS, run_scenario
from tests.golden_cases import plain
from tests.test_golden import _first_diff

PAGES = ["Home.py", "pages/1_Compilation_Planner.py", "pages/2_Reviewer.py",
         "pages/3_Media_Library.py", "pages/4_Server.py", "pages/5_Recommendations.py",
         "pages/6_Training_Labels.py"]


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


def test_server_page_shows_the_volume(tmp_path, monkeypatch):
    """Preview copies linked to another disk (the Hetzner Volume): measured through the link, and the
    Server page shows that disk too."""
    import os
    import storage as SG
    from pathlib import Path
    shm = Path("/dev/shm")
    if not shm.exists() or shm.stat().st_dev == Path(SG.DATA_ROOT).resolve().stat().st_dev:
        pytest.skip("no second filesystem here")
    real = Path(tempfile.mkdtemp(dir=shm)) / "proxies"
    real.mkdir()
    (real / "a.mp4").write_bytes(b"x" * 40000)
    link = tmp_path / "proxies"
    link.symlink_to(real)
    monkeypatch.setattr(SG, "PROXY_DIR", link)
    assert SG.du(link) >= 40000
    disks = SG.extra_disks()
    assert len(disks) == 1 and disks[0][0] == "Preview copies" and disks[0][2]["total"] > 0
    from tests.apptest_util import errors, page_app
    import streamlit as st
    st.cache_data.clear()                      # the folder sizes are cached for 10 minutes
    at = page_app("pages/4_Server.py")
    at.run()
    assert not errors(at), errors(at)
    md = " ".join(m.value for m in at.markdown)
    assert "💾 Volume:" in md and "💾 on the Volume" in md
