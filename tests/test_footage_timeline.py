"""🧭 The footage timeline (Stage 3 step 3): built from synthetic vision data + a student export + labels."""
from __future__ import annotations

import json

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)

TAGS = ["Blowjob", "Cowgirl", "close-up", "medium", "wide", "extreme close-up", "Blonde", "text on screen"]
CUTS = [50, 50, 40, 40, 40, 40, 50, 50]


def _vision(root, vid, duration=30.0, cuts=(10.0, 20.0)):
    vdir = root / "_student" / "videos" / vid
    vdir.mkdir(parents=True)
    n = int(duration * 15)
    curve = np.zeros(n, np.float32)
    curve[: n // 3] = 0.1                       # calm first scene
    curve[n // 3: 2 * n // 3] = 0.9              # busy second scene
    curve[2 * n // 3:] = np.nan                  # nobody tracked in the third
    np.save(vdir / "person_curve.npy", curve)
    et = np.arange(0, duration, 0.5, dtype=np.float32)
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(len(et), 64)).astype(np.float16)
    emb[: len(et) // 2] += 3.0                   # the first half looks alike
    np.savez(vdir / "embeddings.npz", frame_times=et, frame_embs=emb)
    scenes = [{"scene_id": 1, "start": 0.0, "end": 10.0, "movement_bpm": 90.0, "periodicity": 0.8, "onsets": [1.0, 4.0]},
              {"scene_id": 2, "start": 10.0, "end": 20.0, "movement_bpm": 120.0, "periodicity": 0.3, "onsets": []},
              {"scene_id": 3, "start": 20.0, "end": 30.0, "movement_bpm": None, "periodicity": 0.0, "onsets": []}]
    (vdir / "vision.json").write_text(json.dumps({"video_id": vid, "duration": duration, "cuts": list(cuts),
                                                   "scenes": scenes}))
    return vdir


def _scores(folder, vid, duration=30.0):
    folder.mkdir(parents=True, exist_ok=True)
    times = np.arange(0, duration, 0.5, dtype=np.float32)
    s = np.full((len(times), len(TAGS)), 5, np.uint8)
    s[:20, 0] = 90                                # Blowjob visible 0–10 s
    s[20:40, 1] = 80                              # Cowgirl 10–20 s
    s[:, 2] = 70                                  # close-up throughout …
    s[40:, 4] = 90                                # … and wide from 20 s
    people = np.ones(len(times), np.uint8)
    people[40:] = 0
    np.savez(folder / f"{vid}.npz", times=times, scores=s, people=people)
    (folder / "index.json").write_text(json.dumps({"model": "v14", "created": "x", "step": 0.5, "tags": TAGS,
                                                   "cutoffs": CUTS, "videos": [vid]}))


def _setup(tmp_path, monkeypatch, vid="vid_a"):
    import footage_timeline as FT
    root = tmp_path / "work"
    _vision(root, vid)
    scf = tmp_path / "student_scores"
    _scores(scf, vid)
    out = tmp_path / "footage_timeline"
    monkeypatch.setattr(FT, "_load_labels", lambda: [
        {"id": "a", "type": "tags", "video_id": vid, "t": 25.0, "tags": ["Cowgirl"]},          # full label at 25 s
        {"id": "b", "type": "partial", "video_id": vid, "t": 5.0, "yes": ["Blonde"], "no": ["Blowjob"]},
        {"id": "c", "type": "tags", "video_id": "other", "t": 5.0, "tags": ["Blowjob"]},
    ])
    return FT, root, scf, out


def test_build_writes_every_array_and_the_index(tmp_path, monkeypatch):
    FT, root, scf, out = _setup(tmp_path, monkeypatch)
    res = FT.build(root=root / "_student" / "videos", folder=out, sc_folder=scf, progress=lambda *a: None)
    assert res["built"] == ["vid_a"] and res["pca"] == "fitted"
    idx = FT.load_index(out)
    assert idx["model"] == "v14" and idx["tags"] == TAGS and idx["videos"]["vid_a"]["points"] == 60
    tl = FT.load("vid_a", out)
    assert len(tl) == 60
    assert tl.emb.shape == (60, idx["emb_dim"]) and tl.emb.dtype == np.int8
    assert list(tl.cuts) == [10.0, 20.0]
    # movement: calm, busy, nothing — and scaled within the library
    m = tl.move.astype(np.float32)
    assert abs(m[5] - 0.1) < 0.01 and abs(m[25] - 0.9) < 0.01 and np.isnan(m[50])
    sm = tl.scaled_move()
    assert sm[5] < 0.1 and sm[25] > 0.9
    # change peaks at the step between the two scenes
    ch = tl.change.astype(np.float32)
    assert ch[20] > ch[5] and ch[20] > ch[30]
    # rhythm per point from the scene features; onsets relative to the scene start
    assert float(tl.bpm[3]) == 90.0 and float(tl.bpm[25]) == 120.0 and np.isnan(float(tl.bpm[50]))
    assert float(tl.onset[2]) == 0.0 and abs(float(tl.onset[4]) - 1.0) < 0.01       # 1.0 s is an onset; 2.0 s is 1 s off
    # shot size: close-up (1) alone → 1; close-up + wide from 20 s → between 1 and 3
    sh = tl.shot.astype(np.float32)
    assert abs(sh[5] - 1.0) < 0.01 and 1.0 < sh[45] < 3.0
    assert read_status_state(FT, out) == "done"


def read_status_state(FT, out):
    return FT.read_status(out).get("state")


def test_labels_win_within_reach_and_only_for_their_video(tmp_path, monkeypatch):
    FT, root, scf, out = _setup(tmp_path, monkeypatch)
    FT.build(root=root / "_student" / "videos", folder=out, sc_folder=scf, progress=lambda *a: None)
    tl = FT.load("vid_a", out)
    cow, bj, blonde, close = (tl.column(t) for t in ("Cowgirl", "Blowjob", "Blonde", "close-up"))
    # full label at 25 s: Cowgirl → 100, other non-hidden tags → 0, within ±1 s only
    assert tl.scores[50, cow] == 100 and tl.scores[50, close] == 0 and tl.labelled[50]
    assert tl.scores[48, cow] == 100 and tl.scores[52, cow] == 100
    assert tl.scores[47, cow] == 5 and not tl.labelled[47]
    # partial at 5 s: only Blonde (yes) and Blowjob (no) change; close-up keeps the student's score
    assert tl.scores[10, blonde] == 100 and tl.scores[10, bj] == 0 and tl.scores[10, close] == 70
    assert tl.scores[14, bj] == 90                 # 7 s — out of reach
    assert FT.load_index(out)["videos"]["vid_a"]["labels"] == 2


def test_derived_scene_tags_tag_time_and_skip_rules(tmp_path, monkeypatch):
    FT, root, scf, out = _setup(tmp_path, monkeypatch)
    FT.build(root=root / "_student" / "videos", folder=out, sc_folder=scf, progress=lambda *a: None)
    tl = FT.load("vid_a", out)
    scenes = [{"scene_id": 1, "start": 0.0, "end": 10.0}, {"scene_id": 2, "start": 10.0, "end": 20.0},
              {"scene_id": 3, "start": 20.0, "end": 30.0}]
    d = FT.derived_scene_tags(tl, scenes, min_share=0.5)
    assert d[1]["tags"] == ["Blowjob"] and d[2]["tags"] == ["Cowgirl"]
    assert "close-up" not in d[1]["share"] and "Blonde" not in d[1]["share"]     # shot / hair have their own levels
    assert d[1]["share"]["Blowjob"] < 1.0                                        # the partial "no" at 5 s took some
    # a library range: clips wholly outside it are skipped, not an error
    d = FT.derived_scene_tags(tl, scenes, min_share=0.5, time_range=(12.0, 30.0))
    assert 1 not in d and d[2]["tags"] == ["Cowgirl"]
    tt = dict(FT.tag_time(tl))
    assert tt["Blowjob"] > 7 and "wide" not in tt
    # a second build with nothing changed builds nothing; a new export's model rebuilds
    res = FT.build(root=root / "_student" / "videos", folder=out, sc_folder=scf, progress=lambda *a: None)
    assert res["built"] == [] and res["pca"] == "kept"
    idx = json.loads((scf / "index.json").read_text())
    idx["model"] = "v15"
    (scf / "index.json").write_text(json.dumps(idx))
    res = FT.build(root=root / "_student" / "videos", folder=out, sc_folder=scf, progress=lambda *a: None)
    assert res["built"] == ["vid_a"] and FT.load_index(out)["model"] == "v15"


def test_status_report_and_missing_inputs(tmp_path, monkeypatch):
    FT, root, scf, out = _setup(tmp_path, monkeypatch)
    _vision(root, "vid_b")                        # vision but no scores
    r = FT.status_report(root=root / "_student" / "videos", folder=out, sc_folder=scf)
    assert r["ready"] == ["vid_a"] and r["no_scores"] == ["vid_b"] and r["built"] == []
    res = FT.build(root=root / "_student" / "videos", folder=out, sc_folder=scf, progress=lambda *a: None)
    assert res["skipped"] == {"vid_b": "no student scores (export)"}
    r = FT.status_report(root=root / "_student" / "videos", folder=out, sc_folder=scf)
    assert r["built"] == ["vid_a"] and r["ready"] == []


def test_pca_keeps_similarity_order():
    import footage_timeline as FT
    rng = np.random.default_rng(0)
    base = rng.normal(size=(400, 64)).astype(np.float32)
    pca = FT.fit_pca(base, dim=16)
    a = base[0]
    near, far = a + 0.1 * rng.normal(size=64), rng.normal(size=64)
    p = FT.project(np.stack([a, near, far]), pca).astype(np.float32) / 127
    assert p[0] @ p[1] > p[0] @ p[2]


def test_media_library_timeline_tab(tmp_path, monkeypatch):
    """The 🧭 Timeline view headless: empty state, then with vid_a built into the test catalogue."""
    import shutil
    import footage_timeline as FT
    from config import CATALOGUE_DIR
    from tests.apptest_util import errors, page_app
    out = CATALOGUE_DIR / FT.FOLDER
    scf = CATALOGUE_DIR / "student_scores"
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(scf, ignore_errors=True)
    try:
        at = page_app("pages/3_Media_Library.py", timeout=300)
        at.session_state["lib_kind"] = "🎬 Videos"; at.session_state["vid_kind"] = "🧭 Timeline"
        at.run()
        assert not errors(at), errors(at)
        assert any("No footage timeline yet" in m.value for m in at.markdown)
        root = tmp_path / "work"
        _vision(root, "vid_a", duration=60.0, cuts=(15.0, 40.0))
        _scores(scf, "vid_a", duration=60.0)
        monkeypatch.setenv("STUDENT_WORK_DIR", str(root))
        FT.build(folder=out, progress=lambda *a: None)
        at = page_app("pages/3_Media_Library.py", timeout=300)
        at.session_state["lib_kind"] = "🎬 Videos"; at.session_state["vid_kind"] = "🧭 Timeline"
        at.run()
        assert not errors(at), errors(at)
        assert at.selectbox(key="ft_video").value == "vid_a"
        assert any("built" in m.value for m in at.markdown)
        assert len([m for m in at.multiselect if m.key == "ft_tags_vid_a"]) == 1
    finally:
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(scf, ignore_errors=True)
