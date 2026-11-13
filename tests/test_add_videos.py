"""➕ Add new videos: the server's steps (Drive sync → tag scores for the videos without any → timelines) as one
background job with a status the app shows, what every video is still waiting for, and the export scoring only the
videos that have no scores yet."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.test_footage_timeline import _scores, _vision


def _clean():
    import add_videos as AV
    import footage_timeline as FT
    from config import CATALOGUE_DIR
    for p in (CATALOGUE_DIR / FT.FOLDER, CATALOGUE_DIR / "student_scores"):
        shutil.rmtree(p, ignore_errors=True)
    for p in (AV.status_path(), CATALOGUE_DIR / ".add_videos.log"):
        Path(p).unlink(missing_ok=True)


def test_export_todo_scores_only_the_videos_without_scores(tmp_path):
    import student_tags as STG
    dest = tmp_path / "student_scores"
    dest.mkdir()
    vdirs = [tmp_path / "videos" / v for v in ("a", "b", "c")]
    for d in vdirs:
        d.mkdir(parents=True)
    assert STG.export_todo(vdirs, dest, "v16", False) == (vdirs, [])                 # a full export: everything
    assert STG.export_todo(vdirs, dest, "v16", True) == (vdirs, [])                  # nothing exported yet: everything
    for v in ("a", "b", "gone"):
        np.savez(dest / f"{v}.npz", times=np.zeros(1))
    (dest / "half.tmp.npz").write_bytes(b"")
    (dest / "index.json").write_text(json.dumps({"model": "v16", "videos": ["a", "b", "gone"]}))
    todo, kept = STG.export_todo(vdirs, dest, "v16", True)
    assert [d.name for d in todo] == ["c"] and kept == ["a", "b"]                   # "gone" has no vision data any more
    todo, kept = STG.export_todo(vdirs, dest, "v17", True)                           # another model: never mixed
    assert todo == vdirs and kept == []


def test_what_each_video_needs_and_the_whole_run(tmp_path, monkeypatch):
    import add_videos as AV
    import footage_timeline as FT
    from config import CATALOGUE_DIR
    _clean()
    try:
        root = tmp_path / "work"
        scf = CATALOGUE_DIR / "student_scores"
        monkeypatch.setenv("STUDENT_WORK_DIR", str(root))
        monkeypatch.setattr(FT, "_load_labels", lambda: [])
        _vision(root, "vid_a", duration=60.0)
        _vision(root, "vid_b", duration=60.0)
        _vision(root, "vid_c", duration=60.0)
        _scores(scf, "vid_a", duration=60.0)                    # vid_a: scored; vid_b, vid_c: vision data only
        states = AV.video_states()
        assert {"vid_a", "vid_b", "vid_d"} <= set(states)
        assert states["vid_a"]["vision"] and states["vid_a"]["scores"] and not states["vid_a"]["timeline"]
        assert states["vid_b"]["vision"] and not states["vid_b"]["scores"]
        assert states["vid_d"]["catalogue"] and not states["vid_d"]["vision"]
        assert AV.first_need({k: True for k in AV.NEEDS}) is None
        assert AV.first_need({"catalogue": True, "vision": True}) == "scores"
        sm = AV.summary({"x": {**{k: True for k in AV.NEEDS}, "preview": True}, "y": {"catalogue": True, "preview": True},
                         "z": {"catalogue": True, "vision": True, "scores": True}})
        assert sm == {"total": 3, "ready": 1, "waiting": {"vision": ["y"], "timeline": ["z"]}, "no_preview": ["z"]}
        assert AV.scores_model() == "v14"

        # the run: sync, then the export for the videos without scores, then the timelines — in that order
        order = []
        import auto_sync
        monkeypatch.setattr(auto_sync, "run_sync", lambda cb=None: order.append("sync") or {"synced": 7, "skipped": 0, "errors": []})

        def fake_export(model, log):
            order.append(("scores", model, sorted(FT.status_report()["no_scores"])))
            _scores(scf, "vid_b", duration=60.0)                # vid_c has no tuned embeddings: left out
            (scf / "index.json").write_text(json.dumps({"model": model, "tags": json.loads((scf / "index.json").read_text())["tags"],
                                                        "cutoffs": json.loads((scf / "index.json").read_text())["cutoffs"],
                                                        "step": 0.5, "videos": ["vid_a", "vid_b"]}))
            return 0, ["1 video(s) have no embeddings_ft1.npz and are left out (e.g. vid_c) — run …"]
        monkeypatch.setattr(AV, "_score_missing", fake_export)
        monkeypatch.setattr(AV.os, "name", "posix")
        st = AV.run()
        assert order[0] == "sync" and order[1] == ("scores", "v14", ["vid_b", "vid_c"])
        assert st["state"] == "done" and st["finished"] and st["result"]["waiting"].get("scores") == 1      # vid_c
        notes = " | ".join(st["notes"])
        assert "Synced 7 file(s)" in notes and "Scored 1 new video(s) with student v14" in notes
        assert "1 left out" in notes and "embeddings_ft1" in notes and "Built 2 timeline(s)" in notes
        after = AV.video_states()
        assert after["vid_a"]["timeline"] and after["vid_b"]["timeline"] and not after["vid_c"]["scores"]
        assert set(FT.load_index()["videos"]) == {"vid_a", "vid_b"}
        # pressing it again: nothing left that it can do, and it says so
        order.clear()
        monkeypatch.setattr(AV, "_score_missing", lambda model, log: (order.append("scores") or 1, ["vid_c: no embeddings_ft1.npz"]))
        st2 = AV.run()
        assert st2["state"] == "done" and "stopped (exit 1)" in " | ".join(st2["notes"]) and st2["result"]["added"] == 0
        # no export yet → said plainly, the rest still runs; a failing sync doesn't stop the run either
        (scf / "index.json").unlink()
        monkeypatch.setattr(auto_sync, "run_sync", lambda cb=None: (_ for _ in ()).throw(RuntimeError("no credentials")))
        st3 = AV.run()
        assert st3["state"] == "done" and "Drive sync skipped: no credentials" in st3["notes"][0]
        assert "no export has been made yet" in st3["notes"][1]
        # a run whose process died is not "still running"
        AV._write(AV.status_path(), {"state": "running", "pid": 2 ** 22 + 12345, "step": "scores"})
        assert AV.read_status()["state"] == "error" and "stopped before it finished" in AV.read_status()["error"]
    finally:
        _clean()


def test_the_panel_in_the_media_library(tmp_path, monkeypatch):
    import add_videos as AV
    import footage_timeline as FT
    from config import CATALOGUE_DIR
    from tests.apptest_util import errors, page_app
    _clean()
    try:
        root = tmp_path / "work"
        monkeypatch.setenv("STUDENT_WORK_DIR", str(root))
        monkeypatch.setattr(FT, "_load_labels", lambda: [])
        _vision(root, "vid_a", duration=60.0)
        started = []
        monkeypatch.setattr(AV, "start", lambda: started.append(1) or "Started.")
        for kind in ("🎬 Videos", "🧭 Timeline"):
            at = page_app("pages/3_Media_Library.py", timeout=300)
            at.session_state["lib_kind"] = "🎬 Videos"; at.session_state["vid_kind"] = kind
            at.run()
            assert not errors(at), errors(at)
            assert any("ready for the 🤖 new matcher" in m.value and "waiting for tag scores" in m.value for m in at.markdown)
            assert any("waiting video(s) need" in e.label for e in at.expander)
        at.button(key="av_go").click().run()
        assert not errors(at), errors(at)
        assert started == [1]
        # a finished run shows what it did
        AV._write(AV.status_path(), {"state": "done", "finished": "2026-11-13 14:40:00", "notes": ["☁️ Synced 3 file(s) from Drive"],
                                     "result": {"added": 2, "ready": 2, "total": 6}})
        at.run()
        assert not errors(at), errors(at)
        assert any("2 video(s) added" in e.label for e in at.expander)
        AV._write(AV.status_path(), {"state": "error", "error": "boom", "notes": []})
        at.run()
        assert any("boom" in e.value for e in at.error)
    finally:
        _clean()
