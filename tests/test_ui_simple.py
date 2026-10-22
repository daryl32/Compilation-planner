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


def test_matching_step_layout():
    """🎬 Matching keeps every option (mode switch, all settings tabs) in both views, with one status
    line, real problems together in one box, and a Finish bar."""
    from tests.apptest_util import errors, page_app
    at = page_app(PLANNER, advanced=False)
    state = {"track_id": "song_full", "committed_selected_videos": ["vid_a", "vid_b"],
             "ui_section": "🎬 3. Matching & Export"}
    for v in ("vid_a", "vid_b"):
        state[f"select_video_{v}"] = True
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not errors(at), errors(at)
    assert at.radio(key="matching_mode").options == ["Auto", "Manual step-through", "Choreography"]
    assert any(e.label.startswith("⚙️ Matching settings") and "weights" in e.label for e in at.expander)
    assert [t.label for t in at.tabs][:3] == ["🎛️ Mode & order", "⚖️ Auto-fill weights", "🎬 Intro / Outro"]
    # nothing is matched until ▶️ Match
    assert not any(b.label == "💾 Save plan" for b in at.button)
    assert any("press **▶️ Match**" in i.value for i in at.info)
    from tests.apptest_util import press_match
    assert press_match(at)
    assert not errors(at), errors(at)
    assert any(c.value.startswith("**") and "segments**" in c.value for c in at.caption)
    assert len(at.warning) <= 1                                   # problems, if any, in one box
    labels = [b.label for b in at.button]
    assert "🎬 Render Preview" in labels and "💾 Save plan" in labels
    # every mode still works from the same switch
    for mode in ("Manual step-through", "Choreography", "Auto"):
        at.radio(key="matching_mode").set_value(mode).run()
        assert not errors(at), (mode, errors(at))
        assert any("segments**" in c.value for c in at.caption)


import pytest


@pytest.fixture()
def no_story_maps():
    """No saved story maps for the duration of a test (put back afterwards)."""
    import story_map as STM
    before = STM._maps_file().read_text() if STM._maps_file().exists() else None
    STM._maps_file().unlink(missing_ok=True)
    yield
    if before is not None:
        STM._maps_file().write_text(before)


def _matching_app():
    from tests.apptest_util import page_app
    at = page_app(PLANNER)
    state = {"track_id": "song_full", "committed_selected_videos": ["vid_a", "vid_b", "vid_c"],
             "ui_section": "🎬 3. Matching & Export"}
    for v in ("vid_a", "vid_b", "vid_c"):
        state[f"select_video_{v}"] = True
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def test_settings_wait_for_apply_and_match():
    """⚙️ settings are a form: nothing changes until ▶️ Apply & match; a preset picked there sets its
    sliders on Apply; Swap on the plan applies at once."""
    from tests.apptest_util import errors, press_match
    at = _matching_app()
    assert press_match(at) and not errors(at), errors(at)
    assert any(b.label == "💾 Save plan" for b in at.button)
    sig = at.session_state["_auto_plan_cache"]["sig"]
    before = at.session_state["weight_random"]
    at.slider(key="weight_random").set_value(3.0)              # changed in the form, not applied
    at.run()
    assert at.session_state["weight_random"] == before
    assert at.session_state["_auto_plan_cache"]["sig"] == sig
    at.slider(key="weight_random").set_value(3.0)
    assert press_match(at) and not errors(at), errors(at)        # ▶️ Apply & match
    assert at.session_state["weight_random"] == 3.0
    assert at.session_state["autofill_preset"] == "Custom"
    assert at.session_state["_auto_plan_cache"]["sig"] != sig
    # a preset picked in the form sets its sliders on Apply
    at.selectbox(key="autofill_preset").set_value("Maximum variety")
    assert press_match(at) and not errors(at), errors(at)
    import autofill as AF
    for k, v in AF.PRESETS["Maximum variety"].items():
        assert at.session_state[k] == v, k
    assert at.session_state["autofill_preset"] == "Maximum variety"
    # 🔄 Swap on a card applies straight away (no ▶️ needed)
    sig = at.session_state["_auto_plan_cache"]["sig"]
    [b for b in at.button if (b.key or "").startswith("swap_")][0].click().run()
    assert not errors(at), errors(at)
    assert at.session_state["_auto_plan_cache"]["sig"] != sig
    assert any(b.label == "💾 Save plan" for b in at.button)


def test_changed_videos_keep_the_old_plan_until_apply():
    from tests.apptest_util import errors, press_match
    at = _matching_app()
    press_match(at)
    old = at.session_state["_auto_plan_cache"]["result"][0]
    at.session_state["committed_selected_videos"] = ["vid_a", "vid_b"]      # changed on the Videos step
    at.run()
    assert not errors(at), errors(at)
    assert any("This is the last plan" in w.value for w in at.warning)    # old plan, marked out of date
    assert not any(b.label == "💾 Save plan" for b in at.button)            # Save / Render wait
    assert at.session_state["_auto_plan_cache"]["result"][0] == old
    assert press_match(at) and not errors(at), errors(at)
    assert any(b.label == "💾 Save plan" for b in at.button)
    used = {c["video_id"] for e in at.session_state["_auto_plan_cache"]["result"][0] for c in e["scenes"]}
    assert used <= {"vid_a", "vid_b"}


def test_matching_controls_cleanup(no_story_maps):
    """One 'Role clips' control for the two role settings; Reroll without a Seed box; Improved auto-fill
    only shown on a project saved with it off; the split-screen option only with split screen on; the
    Intro / Outro tab owns its toggle and says when the story map is in charge."""
    from tests.apptest_util import errors, press_match
    at = _matching_app()
    assert not errors(at), errors(at)
    assert at.radio(key="role_clips_mode").disabled               # no story map, Intro / Outro off: no role clips
    at.session_state["prefer_intro_outro"] = True
    at.run()
    keys = {w.key for w in at.toggle}
    assert "autofill_v2" not in keys and "intro_outro_exclusive" not in keys and "story_reserve" not in keys
    assert "allow_same_video" not in keys                         # split screen is off
    assert not any(w.key == "autofill_seed" for w in at.number_input)
    assert any(b.label == "🎲 Reroll" for b in at.button)
    assert at.radio(key="role_clips_mode").value == "saved"       # the defaults: reserve on, strict off
    at.radio(key="role_clips_mode").set_value("only")
    assert press_match(at) and not errors(at), errors(at)
    assert at.session_state["intro_outro_exclusive"] is True and at.session_state["story_reserve"] is True
    at.radio(key="role_clips_mode").set_value("anywhere")
    assert press_match(at) and not errors(at), errors(at)
    assert at.session_state["intro_outro_exclusive"] is False and at.session_state["story_reserve"] is False
    # 🎬 Intro / Outro: the toggle lives here (no story map on this track, so it's usable)
    assert not at.toggle(key="prefer_intro_outro").disabled


def test_old_project_settings_show_through():
    from tests.apptest_util import errors
    at = _matching_app()
    at.session_state["autofill_v2"] = False                       # a project saved with the original scoring
    at.session_state["split_screen_enabled"] = True
    at.run()
    assert not errors(at), errors(at)
    keys = {w.key for w in at.toggle}
    assert "autofill_v2" in keys and "allow_same_video" in keys


def test_project_panel_saves_and_opens():
    """📁 Project bar: 💾 Save / 📂 Open and 🆕 New open a panel in the bar (not a pop-over — those jumped
    around on a phone); saving and opening work from it and close it."""
    import json
    from config import PROJECTS_DIR
    from tests.apptest_util import errors, page_app
    at = page_app(PLANNER, advanced=False)
    at.session_state["track_id"] = "song_full"
    at.run()
    assert not errors(at), errors(at)
    assert not [w for w in at.text_input if w.key == "project_save_name"]          # closed to start with
    at.button(key="project_panel_file").click().run()
    at.text_input(key="project_save_name").set_value("panel-test").run()
    [b for b in at.button if b.label.startswith("💾 Save Project")][0].click().run()
    assert not errors(at), errors(at)
    path = PROJECTS_DIR / "panel-test.json"
    try:
        assert path.exists() and json.loads(path.read_text())["track_id"] == "song_full"
        assert not [w for w in at.text_input if w.key == "project_save_name"]      # closed after saving
        at.button(key="project_panel_file").click().run()
        at.selectbox(key="project_load_choice").set_value("panel-test").run()
        [b for b in at.button if b.label == "📂 Open Project"][0].click().run()
        assert not errors(at), errors(at)
        assert any("Opened 'panel-test'" in s.value for s in at.success)
        # 🆕 New: its panel, then the button starts a blank project
        at.button(key="project_panel_new").click().run()
        at.button(key="new_project_go").click().run()
        assert not errors(at), errors(at)
        assert not at.session_state["_loaded_project"] if "_loaded_project" in at.session_state else True
    finally:
        path.unlink(missing_ok=True)
