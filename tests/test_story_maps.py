"""🎭 Story maps: a default per track (made when it's analysed), named maps, and project-only edits
that are only saved to the track when you choose to."""

import json

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)

PLANNER = "pages/1_Compilation_Planner.py"


@pytest.fixture()
def clean_maps():
    import story_map as STM
    before = STM._maps_file().read_text() if STM._maps_file().exists() else None
    STM._maps_file().unlink(missing_ok=True)
    yield STM
    if before is None:
        STM._maps_file().unlink(missing_ok=True)
    else:
        STM._maps_file().write_text(before)


R1 = [{"start": 0.0, "end": 10.0, "role": "intro"}, {"start": 10.0, "end": 96.0, "role": None}]
R2 = [{"start": 0.0, "end": 50.0, "role": "slow"}, {"start": 50.0, "end": 96.0, "role": "action"}]


def test_storage_default_named_and_old_files(clean_maps):
    STM = clean_maps
    STM._maps_file().write_text(json.dumps({"song_full": R1}))          # the old shape: track → regions
    assert STM.saved_map("song_full") == R1 and STM.track_maps("song_full")["named"] == {}
    STM.save_named("song_full", "  Drop  heavy ", R2)
    assert STM.track_maps("song_full") == {"default": R1, "named": {"Drop heavy": R2}}
    STM.save_map("song_full", R2)                                       # the default changes, named stays
    assert STM.saved_map("song_full") == R2 and "Drop heavy" in STM.track_maps("song_full")["named"]
    assert STM.saved_maps() == {"song_full": R2}
    STM.delete_named("song_full", "Drop heavy")
    assert STM.track_maps("song_full")["named"] == {}
    with pytest.raises(ValueError):
        STM.save_named("song_full", "  ", R1)


def test_ensure_default(clean_maps):
    STM = clean_maps
    assert STM.ensure_default("song_full") is True
    assert any(r["role"] for r in STM.saved_map("song_full"))
    assert STM.ensure_default("song_full") is False                     # already has one: left alone


def _story_app():
    from tests.apptest_util import page_app
    at = page_app(PLANNER)
    at.session_state["track_id"] = "song_full"
    at.session_state["ui_section"] = "🎭 1. Story map"
    at.run()
    return at


def test_planner_edits_stay_in_the_project_until_saved(clean_maps):
    STM = clean_maps
    from tests.apptest_util import errors
    STM.save_map("song_full", R2)
    STM.save_named("song_full", "Long intro", R1)
    at = _story_app()
    assert not errors(at), errors(at)
    pick = [s for s in at.selectbox if s.label == "🗺️ Map for this project"][0]
    assert pick.value == "default" and "📁 Long intro" in pick.options
    # pick the named map
    pick.set_value("named:Long intro").run()
    assert not errors(at), errors(at)
    assert at.session_state["story_map"]["regions"] == R1 and at.session_state["story_map"]["source"] == "named:Long intro"
    # an edit: this project's own; the track's maps untouched
    [b for b in at.button if b.label == "🧹 Clear the map"][0].click().run()
    assert not errors(at), errors(at)
    assert at.session_state["story_map"]["source"] == "project"
    assert STM.saved_map("song_full") == R2 and STM.track_maps("song_full")["named"]["Long intro"] == R1
    # 📁 save as a new map, then ⭐ as the default
    nk = [t for t in at.text_input if t.label == "Name"][0]
    nk.set_value("Blank").run()
    [b for b in at.button if b.label == "📁 Save"][0].click().run()
    assert not errors(at), errors(at)
    assert "Blank" in STM.track_maps("song_full")["named"]
    assert at.session_state["story_map"]["source"] == "named:Blank"
    [b for b in at.button if b.label == "🧹 Clear the map"][0].click().run()
    [b for b in at.button if b.label == "⭐ Save as the track's default"][0].click().run()
    assert not errors(at), errors(at)
    assert STM.saved_map("song_full") == at.session_state["story_map"]["regions"]
    assert at.session_state["story_map"]["source"] == "default"


def test_edit_a_map_that_came_from_nowhere(clean_maps):
    """A map with no saved origin ("from": None — a fresh suggestion, or a project from before named maps)
    can be edited without an error (20 Oct: AttributeError 'NoneType' … 'startswith')."""
    from tests.apptest_util import errors
    from tests.apptest_util import page_app
    at = page_app(PLANNER)
    at.session_state["track_id"] = "song_full"
    at.session_state["ui_section"] = "🎭 1. Story map"
    at.session_state["story_map"] = {"track_id": "song_full", "regions": R1, "v": 1, "edited": True,
                                     "source": "project", "from": None}
    at.run()
    assert not errors(at), errors(at)
    [b for b in at.button if b.label == "🧹 Clear the map"][0].click().run()
    assert not errors(at), errors(at)
    assert any(b.label == "📁 Save" for b in at.button)
