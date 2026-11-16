"""🎬 render_jobs (16 Nov 2026): the one render helper Classic, the New matcher and Edge Mode share; plans as JSON that
can't crash on numpy values; the length check in every render note; the New matcher's ⏱️ Compile & render in one
press (on the server); a New-matcher render reopened from 📁 Media Library."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)


def test_a_plan_with_numpy_values_turns_into_plain_json():
    import render_jobs as RJ
    plan = {"a": np.bool_(True), "b": np.int64(3), "c": np.float32(0.5), "d": np.arange(3), "e": {1, 2}, "f": "x"}
    back = json.loads(RJ.plan_json(plan))
    assert back == {"a": True, "b": 3, "c": 0.5, "d": [0, 1, 2], "e": [1, 2], "f": "x"}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_length_check_says_when_the_picture_or_the_sound_is_off(tmp_path):
    import render_jobs as RJ
    f = tmp_path / "two.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x36:d=2", "-f", "lavfi",
                    "-i", "sine=d=2", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(f)], check=True)
    got = RJ.lengths(f)
    assert abs(got["video"] - 2) < 0.2 and abs(got["audio"] - 2) < 0.2
    assert RJ.length_check(f, 2.0) == "ok"
    off = RJ.length_check(f, 5.0)
    assert off.startswith("expected 5.0 s") and "picture" in off and "sound" in off
    assert RJ.length_check(tmp_path / "missing.mp4", 5.0) == "ok"            # nothing readable: nothing to say


def _fake_render(RN):
    def fake(plan, output_path, **kw):          # no ffmpeg: a file and its sidecar, as the renderer leaves them
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"not a real video")
        RN.write_render_metadata(output_path, plan, extra={**(kw.get("extra_meta") or {}), "stage_seconds": {"segments": 61}})
        return {"quick": 2, "stages": {"segments": 61.0}}
    return fake


def test_the_shared_render_numbers_names_and_explains_each_preview(monkeypatch):
    import render_jobs as RJ
    import render_preview as RP
    import renders as RN
    from config import PREVIEW_DIR
    monkeypatch.setattr(RP, "render_plan_dict", _fake_render(RN))
    monkeypatch.setattr(RN, "upload_render", lambda out, token: None)
    monkeypatch.setattr(RN, "auto_clear_enabled", lambda: False)
    plan = {"track_id": "rj test", "timeline": [], "track_duration_sec": 10.0}
    steps = []
    n1, a = RJ.render(plan, label="  first   one ", prog=lambda f, t: steps.append(t), extra_meta={"project_name": "x"})
    n2, b = RJ.render(plan, token="tok")
    assert a.endswith(".001.mp4") and b.endswith(".002.mp4")                       # numbering carries on
    assert n1[0] == "info" and "Connect Google Drive" in n1[1] and "2 from the 480p copies" in n1[1]
    assert "Time: clips 1:01" in n1[1]
    assert n2[0] == "success" and "saved to Google Drive" in n2[1]
    meta = RN.load_render(Path(PREVIEW_DIR) / a)
    assert meta["user"]["name"] == "first one" and meta["project_name"] == "x"
    for p in Path(PREVIEW_DIR).glob("temp-Video-preview-rj*"):
        p.unlink()


def test_compile_and_render_is_one_press_on_the_server_and_a_render_reopens_in_the_new_matcher(monkeypatch):
    from tests.test_story_matcher import _setup
    from config import PREVIEW_DIR
    out = _setup(monkeypatch)
    before = set(Path(PREVIEW_DIR).glob("*")) if Path(PREVIEW_DIR).exists() else set()
    try:
        import mix_jobs
        import planner.compose as PC
        import render_preview as RP
        import renders as RN
        from tests.apptest_util import errors, page_app
        monkeypatch.setattr(RP, "render_plan_dict", _fake_render(RN))
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "🤖 New matcher"
        at.run()
        at.toggle(key="cs_new").set_value(False)
        at.number_input(key="cs_minlen").set_value(1)
        at.text_input(key="cs_q").set_value("song").run()
        at.selectbox(key="cs_track").set_value(at.selectbox(key="cs_track").options[-1]).run()
        assert not errors(at), errors(at)
        at.button(key="cs_go_render").click().run()               # ⏱️ Compile & render
        assert mix_jobs.get(PC.JOB_KEY) is not None
        mix_jobs.wait([PC.JOB_KEY], timeout=300)
        at.run()
        assert not errors(at), errors(at)
        res = at.session_state[PC.RESULT_KEY]
        name = at.session_state["compose_render"]
        assert res["plan"]["timeline"] and name.endswith(".mp4")
        assert any("Rendered " + name in m.value for m in list(at.info) + list(at.success)), [m.value for m in at.info]
        at.run()                                                  # the result is taken once: drawn again, still there
        assert at.session_state["compose_render"] == name and not errors(at)
        # ↩️ reopened from 📁 Media Library → Renders: the plan is the result on show
        meta = RN.load_render(Path(PREVIEW_DIR) / name)
        data, kind = RN.reopen_state({**meta, "plan": {**res["plan"], "matching_mode": "Auto story"}})
        assert kind == "compose"
        at.button(key="compose_back").click().run()
        at.session_state["_reopen_compose"] = {**data, "label": name}
        at.run()
        assert not errors(at), errors(at)
        back = at.session_state[PC.RESULT_KEY]
        assert back["plan"]["timeline"] == json.loads(json.dumps(res["plan"]["timeline"])) and "Reopened" in back["notes"][0]
    finally:
        import mix_jobs
        import planner.compose as PC
        mix_jobs.forget(PC.JOB_KEY)                         # a finished job is kept for a reloaded page: not for the next test
        for p in set(Path(PREVIEW_DIR).glob("*")) - before:  # this test's renders and their sidecars
            if p.is_file():
                p.unlink()
        shutil.rmtree(out, ignore_errors=True)


def test_classic_renders_on_the_server_with_its_project(monkeypatch):
    """🎛️ Classic's 🎬 Render Preview runs as a server job too: the plan and the project are taken when it is pressed,
    and the render keeps the project (so ↩️ reopen in planner restores everything, as before)."""
    import mix_jobs
    import render_preview as RP
    import renders as RN
    from config import PREVIEW_DIR
    from tests.apptest_util import errors, page_app, press_match
    import render_jobs as RJ
    monkeypatch.setattr(RP, "render_plan_dict", _fake_render(RN))
    monkeypatch.setattr(RJ, "missing_sources", lambda *a, **k: [])     # (no Drive here: the files count as on the server)
    before = set(Path(PREVIEW_DIR).glob("*")) if Path(PREVIEW_DIR).exists() else set()
    try:
        at = page_app("pages/1_Compilation_Planner.py", advanced=False)
        state = {"track_id": "song_full", "committed_selected_videos": ["vid_a", "vid_b"],
                 "ui_section": "🎬 3. Matching & Export"}
        for v in ("vid_a", "vid_b"):
            state[f"select_video_{v}"] = True
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        assert press_match(at) and not errors(at), errors(at)
        [b for b in at.button if b.label == "🎬 Render Preview"][0].click().run()
        assert not errors(at), errors(at)
        mix_jobs.wait(["classic:render"], timeout=300)
        at.run()
        assert not errors(at), errors(at)
        notes = [m.value for m in list(at.info) + list(at.success) + list(at.warning)]
        assert any(n.startswith("Rendered temp-Video-preview-song_full.") for n in notes), notes
        name = [p for p in set(Path(PREVIEW_DIR).glob("*.mp4")) - before][0].name
        meta = RN.load_render(Path(PREVIEW_DIR) / name)
        assert isinstance(meta.get("project"), dict) and RN.reopen_state(meta)[1] == "full"
        at.run()                                                 # the message is shown once
        assert not any(n.startswith("Rendered ") for n in [m.value for m in list(at.info) + list(at.success)])
    finally:
        mix_jobs.forget("classic:render")
        for p in set(Path(PREVIEW_DIR).glob("*")) - before:
            if p.is_file():
                p.unlink()
