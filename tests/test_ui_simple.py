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
