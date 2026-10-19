"""Simple / Advanced view (ui_mode.py) and the planner's Simple steps."""

from tests.support import DATA  # noqa: F401  (installs the test config first)

PLANNER = "pages/1_Compilation_Planner.py"


def _keys(at, kind):
    return {w.key for w in getattr(at, kind) if w.key}


def test_audio_step_simple_vs_advanced():
    from tests.apptest_util import errors, page_app
    at = page_app(PLANNER, advanced=False)
    at.session_state["track_id"] = "song_full"
    at.run()
    assert not errors(at), errors(at)
    assert at.toggle(key="ui_advanced").value is False
    # Simple: track + cutting style + pace + the chart and one status line — no method dials
    assert {"track_id", "cut_style"} <= _keys(at, "selectbox")
    assert "segmentation_method" not in _keys(at, "radio")
    assert not _keys(at, "slider") - {"cut_style_pace"}
    assert any(c.value.startswith("**") and "segments**" in c.value for c in at.caption)
    # picking a style in Simple sets the cutting dials (they apply even though they're hidden)
    at.selectbox(key="cut_style").set_value("cinematic").run()
    assert not errors(at), errors(at)
    assert at.session_state["cut_style"] == "cinematic"
    # 🛠️ Advanced brings every dial back
    at.toggle(key="ui_advanced").set_value(True).run()
    assert not errors(at), errors(at)
    assert "segmentation_method" in _keys(at, "radio")
    assert at.session_state["cut_style"] == "cinematic"


def test_view_is_remembered_per_user():
    import ui_mode
    from tests.apptest_util import page_app
    ui_mode.PREFS_FILE.unlink(missing_ok=True)
    try:
        at = page_app(PLANNER, advanced=False)
        at.run()
        at.toggle(key="ui_advanced").set_value(True).run()
        fresh = page_app(PLANNER)
        del fresh.session_state["ui_advanced"]           # a new session: comes from the saved preference
        fresh.run()
        assert fresh.session_state["ui_advanced"] is True
    finally:
        ui_mode.PREFS_FILE.unlink(missing_ok=True)


def test_simple_hides_story_step_and_trims_videos():
    from tests.apptest_util import errors, page_app
    at = page_app(PLANNER, advanced=False)
    at.session_state["track_id"] = "song_full"
    at.run()
    steps = at.radio(key="ui_section").options
    assert len(steps) == 3 and not any("Story map" in s for s in steps)
    assert any("story map" in c.value.lower() for c in at.caption)          # says which map is in use
    at.radio(key="ui_section").set_value([s for s in steps if "Videos" in s][0]).run()
    assert not errors(at), errors(at)
    labels = [b.label for b in at.button]
    assert "✅ Select all recommended" in labels
    assert not any(b.label == "🎚️" for b in at.button)                      # per-project ranges: Advanced
    assert "tag_filter" not in {w.key for w in at.multiselect}               # scene-tag filter: Advanced
    assert "vid_actor_filter" in {w.key for w in at.multiselect}             # actor filter stays
    [b for b in at.button if b.label == "✅ Select all recommended"][0].click().run()
    assert not errors(at), errors(at)
    assert at.session_state["committed_selected_videos"]
    assert any("video(s) selected" in c.value for c in at.caption)
    # Advanced shows all four steps and the full Videos step again
    at.toggle(key="ui_advanced").set_value(True).run()
    assert not errors(at), errors(at)
    assert len(at.radio(key="ui_section").options) == 4
    assert any(b.label == "🎚️" for b in at.button)


def test_story_step_still_opens_in_simple_when_sent_there():
    from tests.apptest_util import errors, page_app
    at = page_app(PLANNER, advanced=False)
    at.session_state["track_id"] = "song_full"
    at.session_state["ui_section"] = "🎭 1. Story map"
    at.run()
    assert not errors(at), errors(at)
    assert at.radio(key="ui_section").value == "🎭 1. Story map"
