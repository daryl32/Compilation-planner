"""⚡ The planner's previews: no player (no video read into memory) until ▶️ Play is pressed;
at most two players at once; and the 🩺 timings log the step and the slow parts of a run."""

import shutil

import pytest

from tests.support import DATA
from tests import page_scenarios as PS
from tests.apptest_util import page_app, press_match


@pytest.fixture
def three_previews():
    d = DATA / "previews"
    d.mkdir(parents=True, exist_ok=True)
    made = []
    for k in (1, 2, 3):
        p = d / f"temp-Video-preview-song_full.{k:03d}.mp4"
        p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(64))
        made.append(p)
    yield made
    for p in made:
        p.unlink(missing_ok=True)
    for p in d.glob("*.json"):
        if "song_full" in p.name:
            p.unlink(missing_ok=True)
    shutil.rmtree(d / ".thumbs", ignore_errors=True)


def _matching_app():
    conf = PS.SCENARIOS["full_smart_auto"]
    at = page_app(PS.PAGE)
    state = {"track_id": conf["track"], "segmentation_method": PS.METHODS[conf["method"]],
             "committed_selected_videos": list(PS.VIDEOS)}
    for v in PS.VIDEOS:
        state[f"select_video_{v}"] = True
    for section in PS.SECTIONS:
        state["ui_section"] = section
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
    press_match(at)
    return at


def _players(at) -> int:
    return len(at.get("video"))


def test_previews_load_only_when_played(three_previews):
    at = _matching_app()
    assert not at.exception, at.exception
    plays = [b for b in at.button if b.key and b.key.startswith("play_preview_")]
    assert len(plays) == 3 and _players(at) == 0                 # listed, nothing loaded
    for k, b in enumerate(plays):
        at.button(key=b.key).click().run()
        assert _players(at) == min(k + 1, 2)                     # at most two at once
    assert at.session_state["preview_players"] == [plays[1].label.split(" ", 2)[2], plays[2].label.split(" ", 2)[2]]
    at.button(key=plays[2].key).click().run()                    # ⏹ hide
    assert _players(at) == 1


def test_page_timings_name_the_step_and_slow_parts(monkeypatch):
    import diagnostics as DG
    got = []
    monkeypatch.setattr(DG, "record", lambda step, sec, detail="", path=None: got.append(step))
    _matching_app()
    assert "Planner page: 3. Matching & Export" in got and "Planner page (full run)" in got
    import time
    from tests.support import load_page
    page = load_page(PS.PAGE, {})
    page["_PAGE_MARKS"][:] = [("start", time.perf_counter() - 1.0), ("block cards", time.perf_counter() - 0.2)]
    page["_PAGE_T0"] = time.perf_counter() - 1.0
    got.clear()
    page["_record_page_run"]()
    assert "Planner part: setup" not in got and "Planner part: block cards" in got
