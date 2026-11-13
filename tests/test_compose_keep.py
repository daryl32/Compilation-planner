"""🤖 New matcher: the settings survive the result page ("← Change" used to come back to everything reset — Streamlit
forgets a widget on any run it isn't drawn), and a render can be given a 🏷️ name.

The headless test runner does NOT forget undrawn widgets, so this test can't catch the reset coming back on its own:
it checks the settings are registered and come back intact. The reset itself was reproduced, and the fix confirmed,
in a real browser on 13 Nov (Movement only + Split screen on → Compile → ← Change: both off without
settings_store.keep in compose.render, both still on with it)."""
from __future__ import annotations

import shutil
from pathlib import Path

from tests.support import DATA  # noqa: F401  (installs the test config first)


def test_change_comes_back_to_the_settings_as_they_were_and_a_render_takes_its_name(monkeypatch):
    from tests.test_story_matcher import _setup
    out = _setup(monkeypatch)
    try:
        import planner.compose as PC
        import render_preview as RP
        import renders as RN
        from config import PREVIEW_DIR
        from tests.apptest_util import errors, page_app

        def fake_render(plan, output_path, **kw):               # no ffmpeg: a file and its sidecar, as the renderer leaves them
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"not a real video")
            RN.write_render_metadata(output_path, plan, extra=kw.get("extra_meta"))
            return {"quick": 1}
        monkeypatch.setattr(RP, "render_plan_dict", fake_render)
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "🤖 New matcher"
        at.run()
        assert not errors(at), errors(at)
        at.toggle(key="compose_movement_only").set_value(True)
        at.segmented_control(key="compose_cuts").set_value("lines")
        at.slider(key="compose_smooth").set_value(0.8)
        at.number_input(key="compose_max_videos").set_value(5)
        at.toggle(key="cs_new").set_value(False)
        at.number_input(key="cs_minlen").set_value(1)
        at.text_input(key="cs_q").set_value("song").run()
        assert not errors(at), errors(at)
        at.slider(key="compose_punch").set_value(0.5)
        at.toggle(key="compose_vocal_rule").set_value(False)
        at.toggle(key="compose_natural").set_value(True)
        tracks = at.selectbox(key="cs_track").options
        at.selectbox(key="cs_track").set_value(at.selectbox(key="cs_track").options[-1]).run()
        assert not errors(at), errors(at)
        want = {"compose_movement_only": True, "compose_cuts": "lines", "compose_smooth": 0.8, "compose_max_videos": 5,
                "compose_punch": 0.5, "compose_vocal_rule": False, "compose_natural": True, "compose_lab_power": 1,
                "cs_minlen": 1, "cs_q": "song", "cs_track": at.session_state["cs_track"]}
        assert tracks and all(at.session_state[k] == v for k, v in want.items())
        at.button(key="cs_go").click().run()
        assert not errors(at), errors(at)
        res = at.session_state[PC.RESULT_KEY]
        assert res["used"]["cuts"] == "lines" and res["used"]["punch"] == 0.5 and res["plan"]["story"]["natural"]
        at.run()                                                 # the result page, drawn again: no setting widget on it
        # 🏷️ a named render
        at.text_input(key="compose_render_name").set_value("  lead  on,   punch half ").run()
        at.button(key="compose_render_btn").click().run()
        assert not errors(at), errors(at)
        name = at.session_state["compose_render"]
        meta = RN.load_render(Path(PREVIEW_DIR) / name)
        assert name.startswith("temp-Video-preview-") and name.endswith(".001.mp4")          # the file keeps its name
        assert meta["user"]["name"] == "lead on, punch half" and meta["user"]["rating"] == 0
        assert meta["compile_note"].startswith("🧾 This compile:") and PC.NATURAL in meta["compile_note"]
        RN.update_user_fields(Path(PREVIEW_DIR) / name, name="renamed", rating=4)
        assert RN.load_render(Path(PREVIEW_DIR) / name)["user"] == {"rating": 4, "notes": "", "keep": False, "name": "renamed"}
        assert RN._user_fields({"user": {"rating": 2}})["name"] == ""                        # an older render: no name
        # ← Change: everything as it was left
        at.button(key="compose_back").click().run()
        assert not errors(at), errors(at)
        assert all(at.session_state[k] == v for k, v in want.items()), {k: at.session_state[k] for k in want}
        assert at.toggle(key="compose_movement_only").value and at.slider(key="compose_punch").value == 0.5
        assert at.selectbox(key="cs_track").value == want["cs_track"] and at.toggle(key="compose_natural").value
        assert at.session_state["compose_render_name"] == "  lead  on,   punch half "
        # every setting widget of the page is registered to be kept (a new one must be added to KEPT / KEPT_IF_SET)
        drawn = {w.key for kind in ("toggle", "slider", "number_input", "selectbox", "multiselect", "text_input", "radio", "segmented_control", "select_slider")
                 for w in getattr(at, kind) if w.key and w.key.startswith(("compose_", "cs_", "cv_"))}
        assert drawn - {"compose_sync_data"} <= set(PC.KEPT) | set(PC.KEPT_IF_SET), drawn - set(PC.KEPT) - set(PC.KEPT_IF_SET)
        # a kept pick that is no longer on offer is dropped before its widget is drawn; a kept range is brought inside
        import streamlit as st
        st.session_state["t_pick"] = ["vid_a", "a video that was deleted"]
        PC._still_there("t_pick", ["vid_a", "vid_b"])
        st.session_state["t_range"] = (1, 999)
        PC._in_range("t_range", 60, 180)
        assert st.session_state["t_pick"] == ["vid_a"] and st.session_state["t_range"] == (60, 180)
        PC._still_there("t_never_set", ["vid_a"])
        PC._in_range("t_never_set", 0, 1)
        assert "t_never_set" not in st.session_state
    finally:
        shutil.rmtree(out, ignore_errors=True)
        from config import PREVIEW_DIR
        for p in Path(PREVIEW_DIR).glob("temp-Video-preview-*"):
            p.unlink(missing_ok=True)
