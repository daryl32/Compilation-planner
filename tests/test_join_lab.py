"""🧩 Join Lab: clip kinds from names / folders, frame scores, the look report and a rendered try (19 Nov 2026)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)


class _Job:
    def progress(self, *a, **k):
        pass


def _clip(path: Path, colour: str, secs: float = 2.0, audio: bool = False, move: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    src = (f"color=c={colour}:s=320x568:d={secs}:r=30" + (",drawbox=x=40:y=100:w=240:h=300:c=white:t=fill:enable='gte(t,1)'" if move else ""))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f=440:d={secs}", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


def test_kinds_and_looks_come_from_folder_and_name():
    import join_lab as JL
    c = JL.classify("Round 4.mp4", "Nude - Lying Back/avatar_clips")
    assert c["kind"] == "voice" and c["look"] == "Nude - Lying Back" and c["outfit"] == "Nude" and c["pose"] == "Lying Back"
    c = JL.classify("nude_lying-back_idle_02.mp4", "transitions")
    assert c["kind"] == "idle" and c["pose"] == "lying back" and not c["look"]
    assert JL.clip_look(c, ["Nude - Lying Back", "Default - Lying on bed 1"]) == "Nude - Lying Back"
    c = JL.classify("default_lying-on-bed-1-to-sitting-on-bed1_transition_01.mp4", "transitions")
    assert c["kind"] == "transition" and c["pose"] == "lying on bed 1" and c["to"] == "sitting on bed1"


def test_scores_report_and_a_rendered_try(tmp_path, monkeypatch):
    import join_lab as JL
    monkeypatch.setattr(JL, "lab_dir", lambda: tmp_path / "lab")
    d = tmp_path / "lab" / "clips"
    _clip(d / "v1.mp4", "gray", audio=True)
    _clip(d / "v2.mp4", "gray", audio=True, move=True)              # ends with a box on screen: away from home
    _clip(d / "i1.mp4", "gray", secs=1.5)
    _clip(d / "x1.mp4", "red")
    home = tmp_path / "lab" / "homes" / "home.png"
    home.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x568:d=0.1", "-frames:v", "1",
                    str(home)], check=True)
    idx = {"clips": {}, "homes": {"Nude - Lying Back": {"file": "home.png", "path": str(home), "f": JL.image_features(home)}}}
    for cid, name, folder in (("v1", "Line A.mp4", "Nude - Lying Back/avatar_clips"),
                              ("v2", "Line B.mp4", "Nude - Lying Back/avatar_clips"),
                              ("i1", "nude_lying-back_idle_01.mp4", "transitions"),
                              ("x1", "Line C.mp4", "Default - Lying on bed 1/avatar_clips")):
        p = d / f"{cid}.mp4"
        idx["clips"][cid] = {**JL.classify(name, folder), "name": name, "folder": folder,
                             "seconds": JL.probe_seconds(p), "first": JL.features(p, 0.0), "last": JL.features(p, None)}
    assert JL.score(idx["clips"]["v1"]["first"], idx["clips"]["v1"]["last"]) == 100.0
    assert JL.score(idx["clips"]["v1"]["last"], idx["clips"]["x1"]["first"]) < JL.OK          # grey → red: a bridge
    rows = JL.look_report(idx, "Nude - Lying Back")
    assert [r["id"] for r in rows] == ["v1", "v2", "i1"]                                     # voice first, then the idle
    by = {r["id"]: r for r in rows}
    assert JL.band(by["v1"]["end"]) == "crossfade" and by["v2"]["end"] < by["v1"]["end"]
    s = JL.summary(rows)
    assert s["voice"] == 2 and s["idles"] == 1
    assert JL.best_idles(idx, "Nude - Lying Back", "v1")[0][1] == "i1"
    m = JL.pair_matrix(idx, ["v1", "v2"])
    assert m[0][0] is None and m[0][1] == 100.0
    pieces = [{"id": "v1"}, {"id": "i1", "pingpong": True, "hold": 4.0}, {"id": "v2", "trim_in": 0.5}]
    got = JL.render_sequence(_Job(), idx, pieces, xfade=0.3)
    assert Path(got["file"]).exists() and abs(got["seconds"] - (2 + 4 + 1.5 - 0.6)) < 0.05
    assert abs(JL.probe_seconds(got["file"]) - got["seconds"]) < 0.15
    assert len(got["joins"]) == 2 and got["joins"][0]["band"] == "crossfade"
    assert JL._end_time({"pingpong": True, "hold": 4.0}, 1.5) == pytest.approx(1.0)            # 4 s of a 1.5 s idle: fwd, back, then 1.0 s forwards


def test_the_page_opens_empty_and_with_clips(tmp_path):
    import json
    import shutil
    import join_lab as JL
    from tests.apptest_util import errors, page_app
    lab = JL.lab_dir()
    shutil.rmtree(lab, ignore_errors=True)
    try:
        at = page_app("pages/7_Join_Lab.py").run()
        assert not errors(at) and any("Sync from Google Drive first" in i.value for i in at.info)
        lab.mkdir(parents=True)
        f = {"g": [[100.0] * 64] * 36, "c": [[[100.0] * 3] * 16] * 9}
        g = {"g": [[160.0] * 64] * 36, "c": [[[160.0] * 3] * 16] * 9}
        idx = {"synced": 1, "homes": {"Nude - Lying Back": {"file": "a.png", "path": "/nope.png", "f": f}},
               "clips": {"v1": {**JL.classify("Line A.mp4", "Nude - Lying Back/avatar_clips"), "name": "Line A.mp4",
                                "folder": "Nude - Lying Back/avatar_clips", "seconds": 3.0, "first": f, "last": g},
                         "i1": {**JL.classify("nude_lying-back_idle_01.mp4", "transitions"), "name": "nude_lying-back_idle_01.mp4",
                                "folder": "transitions", "seconds": 4.0, "first": g, "last": f}}}
        JL.index_path().write_text(json.dumps(idx))
        at = page_app("pages/7_Join_Lab.py").run()
        assert not errors(at)
        assert any("end away from home" in (c.value or "") for c in at.caption)
    finally:
        shutil.rmtree(lab, ignore_errors=True)
