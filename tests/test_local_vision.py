"""local_vision.py — the Modal vision stage on the laptop: picking videos, timing, the Modal-vs-laptop
comparison, and that its output is what the student reads (no GPU or models needed here)."""

from __future__ import annotations

import json

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)

import local_vision as LV
import student_tags as S


def _video(root, v, *, cuts=(), curve=None, times=None, embs=None, source="laptop"):
    d = root / v
    d.mkdir(parents=True, exist_ok=True)
    (d / "vision.json").write_text(json.dumps({"video_id": v, "cuts": list(cuts), "tracks": 3, "source": source}))
    times = np.arange(0, 10, 0.5) if times is None else times
    embs = np.random.default_rng(1).normal(size=(len(times), 8)).astype(np.float16) if embs is None else embs
    np.savez_compressed(d / "embeddings.npz", frame_times=times, frame_embs=embs,
                        scene_ids=np.array([]), scene_embs=np.array([]))
    np.save(d / "person_curve.npy", (np.sin(np.arange(150) / 5) if curve is None else curve).astype(np.float32))
    return d


def test_candidates_skip_done_and_missing_proxy_newest_first(tmp_path):
    _video(tmp_path, "done")
    cats = {v: {"video_id": v, "scenes": []} for v in ("done", "old", "new", "noproxy")}
    got = LV.candidates(cats, lambda v: v != "noproxy", root=tmp_path, mtimes={"old": 1.0, "new": 5.0})
    assert got == ["new", "old"]


def test_has_vision_needs_both_files(tmp_path):
    d = _video(tmp_path, "a")
    assert LV.has_vision("a", tmp_path)
    (d / "vision.json").unlink()
    assert not LV.has_vision("a", tmp_path)


def test_per_target_minutes():
    assert LV.per_target(40 * 60, 8 * 60) == 4.0          # 40 min of video in 8 min → 4 min per 20
    assert LV.per_target(0, 10) is None


def test_compare_identical_and_shifted(tmp_path):
    t = np.arange(0, 30, 0.5)
    e = np.random.default_rng(3).normal(size=(len(t), 16)).astype(np.float16)
    m = _video(tmp_path / "m", "v", cuts=[3.0, 10.0, 20.0], times=t, embs=e, source=None)
    same = LV.compare(m, _video(tmp_path / "l", "v", cuts=[3.1, 10.0, 25.0], times=t, embs=e))
    assert same["cuts"] == {"modal": 3, "laptop": 3, "matched": 2}
    assert same["curve_corr"] == 1.0
    assert same["emb_cosine"]["mean"] > 0.999 and same["emb_cosine"]["n"] == len(t)
    noisy = e.astype(np.float32) + np.random.default_rng(4).normal(scale=2.0, size=e.shape)
    other = LV.compare(m, _video(tmp_path / "l2", "v", times=t, embs=noisy.astype(np.float16)))
    assert other["emb_cosine"]["mean"] < 0.9


def test_laptop_output_is_what_the_student_reads(tmp_path):
    """A laptop video has no teacher file — the student still reads its embeddings, so export and
    picks cover it, and only your labels / Reviewer scenes can make it training data."""
    d = _video(tmp_path, "newvid")
    assert not (d / S.TEACHER_FILE).exists()
    et, emb = S.video_features(d, use_pose=False)
    assert et.size == 20 and emb.shape == (20, 8)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
    written = {"embeddings.npz", "person_curve.npy", "vision.json"}
    assert written <= set(S.VIDEO_FILES)


def test_scores_model_prefers_explicit():
    assert LV.scores_model("v5b") == "v5b"
