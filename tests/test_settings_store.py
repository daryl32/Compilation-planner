"""⚙️ One settings store (settings_store.py): every planner setting keeps its value while you
move between the steps — including the ones whose widgets aren't drawn on the step you're on,
values set by buttons / callbacks (🎨 style, 🎯 Aim for), and the Matching step's weights."""

from tests.support import DATA  # noqa: F401  (installs the test config first)

PLANNER = "pages/1_Compilation_Planner.py"
AUDIO, STORY, VIDEOS, MATCH = ("🎵 0. Audio Settings", "🎭 1. Story map", "📼 2. Videos & Time Ranges",
                               "🎬 3. Matching & Export")


def _app():
    from tests.apptest_util import page_app
    at = page_app(PLANNER)
    state = {"track_id": "song_full", "committed_selected_videos": ["vid_a", "vid_b"], "ui_section": AUDIO}
    for v in ("vid_a", "vid_b"):
        state[f"select_video_{v}"] = True
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _go(at, section):
    from tests.apptest_util import errors
    at.radio(key="ui_section").set_value(section).run()
    assert not errors(at), errors(at)


def test_settings_survive_moving_between_steps():
    from tests.apptest_util import errors, press_match
    at = _app()
    assert not errors(at), errors(at)
    # 🎵 Smart dials
    at.radio(key="segmentation_method").set_value("Smart (self-tuning)").run()
    at.slider(key="sm_contrast").set_value(1.5).run()
    accents = bool(at.session_state["sm_accents"])
    at.toggle(key="sm_accents").set_value(not accents).run()
    assert not errors(at), errors(at)
    # 🎬 Matching: a weight (inside the settings form) and a step setting
    _go(at, MATCH)
    at.slider(key="weight_shape").set_value(3.5)            # both inside the settings form:
    at.toggle(key="sequential_mode").set_value(True)        # they apply with ▶️ Match
    assert press_match(at)
    assert not errors(at), errors(at)
    # round trip through every other step
    for section in (VIDEOS, STORY, AUDIO, VIDEOS, MATCH):
        _go(at, section)
    assert at.session_state["sm_contrast"] == 1.5
    assert at.session_state["sm_accents"] is (not accents)
    assert at.session_state["weight_shape"] == 3.5
    assert at.session_state["sequential_mode"] is True
    assert at.slider(key="weight_shape").value == 3.5
    _go(at, AUDIO)
    assert at.slider(key="sm_contrast").value == 1.5
    assert at.toggle(key="sm_accents").value is (not accents)


def test_values_set_by_a_style_survive_too():
    """🎨 A cutting style sets the method and its dials from a callback — they must outlive a trip
    to another step (the dials' widgets aren't drawn there)."""
    from tests.apptest_util import errors
    at = _app()
    at.selectbox(key="cut_style").set_value("cinematic").run()
    assert not errors(at), errors(at)
    method = at.session_state["segmentation_method"]
    dials = {k: at.session_state[k] for k in ("sc_bars_per_cut", "sc_follow_energy", "sc_min_sec")
             if k in at.session_state}
    for section in (MATCH, VIDEOS, AUDIO):
        _go(at, section)
    assert at.session_state["cut_style"] == "cinematic"
    assert at.session_state["segmentation_method"] == method
    for k, v in dials.items():
        assert at.session_state[k] == v, k


def test_saved_project_has_the_settings_after_moving_around():
    from tests.apptest_util import errors
    from tests.support import load_page
    at = _app()
    at.slider(key="sm_contrast").set_value(0.75).run()
    for section in (MATCH, STORY, VIDEOS):
        _go(at, section)
    assert not errors(at), errors(at)
    ns = load_page(PLANNER, session_state={k: at.session_state[k] for k in
                                           ("sm_contrast", "track_id", "segmentation_method")})
    data = ns["_build_project_save_dict"]()
    assert data["sm_contrast"] == 0.75
