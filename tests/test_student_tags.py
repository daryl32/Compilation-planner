"""student_tags.py — the laptop student tag model, end to end on a small made-up library (CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import student_tags as S
import trial_logic as TL

VOCAB_CSV = ("tag,group,kind,one_of,teacher,never_with,group_required\n"
             "close-up,shot,frame,,yes,wide,yes\n"
             "wide,shot,frame,,yes,close-up,yes\n"
             "dancing,action,motion,,yes,,\n"
             "kissing,action,frame,,no,,\n")
TAGS = ["close-up", "wide", "dancing", "kissing"]
DIM = 32


def _truth(t: float, seed: int) -> list:
    """The made-up 'real' tags at time t: shot alternates every 20 s, dancing in the middle third."""
    out = ["close-up" if int(t // 20) % 2 == 0 else "wide"]
    if 40 <= t % 120 < 80:
        out.append("dancing")
    if seed % 3 == 0 and int(t // 10) % 3 == 0:
        out.append("kissing")
    return out


def _library(root, n_videos=14, minutes=2.0):
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(len(TAGS), DIM))
    vids = root / "videos"
    labels, test = [], []
    for k in range(n_videos):
        v = f"vid{k:02d}"
        d = vids / v
        d.mkdir(parents=True)
        times = np.arange(0, minutes * 60, 0.5)
        emb = rng.normal(scale=0.6, size=(len(times), DIM))
        for i, t in enumerate(times):
            for tag in _truth(t, k):
                emb[i] += basis[TAGS.index(tag)]
        np.savez_compressed(d / "embeddings.npz", frame_times=times, frame_embs=emb.astype(np.float16))
        results = {}
        for j, t in enumerate(np.arange(1.0, minutes * 60, 4.0)):
            tr = [x for x in _truth(t, k) if x != "kissing"]          # the teacher can't see kissing
            results[f"m{j}"] = {"t": float(t), "scene_id": 0, "tags": tr, "unsure": [],
                                "conf": {x: 90 for x in tr}}
        (d / S.TEACHER_FILE).write_text(json.dumps({"results": results}))
        for t in np.arange(3.0, minutes * 60, 9.0):
            labels.append({"id": f"tags:{v}:{t:.2f}", "type": "tags", "video_id": v, "t": float(t),
                           "tags": _truth(t, k), "labelled_at": "2026-11-05T10:00:00"})
        if k < 2:
            test.append(v)
    lib = root / "library"
    lib.mkdir()
    (lib / "tag_vocabulary.csv").write_text(VOCAB_CSV)
    (lib / "training_labels.json").write_text(json.dumps({"version": 1, "items": labels}))
    (root / "testset.json").write_text(json.dumps({"videos": [{"video_id": v} for v in test]}))
    # A teacher test run that gets every shot right but never says kissing / dancing.
    frames = [{"v": i["video_id"], "t": i["t"], "g": i["id"], "tags": [x for x in i["tags"] if x in ("close-up", "wide")]}
              for i in labels if i["video_id"] in test]
    (root / "test_runs.json").write_text(json.dumps({"runs": [
        {"id": "2026-11-03_try2-base", "label": "try2-base", "lora": "try2", "frames": frames}]}))
    return labels, test


def test_window_index_clamps_and_centres():
    times = np.arange(0, 10, 0.5)
    w = S.window_index(times, 5.0)
    assert len(w) == 2 * S.HALF + 1 and times[w[S.HALF]] == 5.0
    assert list(S.window_index(times, 0.0)[:5]) == [0, 0, 0, 0, 0]
    assert S.window_index(times, 9.9)[-1] == len(times) - 1


def test_teacher_target_respects_final_tags():
    ix = {t: j for j, t in enumerate(TAGS)}
    y = S.teacher_target({"tags": ["wide"], "conf": {"wide": 40, "close-up": 70}}, ix, len(TAGS))
    assert y[ix["wide"]] == pytest.approx(0.55)          # kept by the rules → at least 0.55
    assert y[ix["close-up"]] == pytest.approx(0.45)      # listed but not kept → at most 0.45
    assert y[ix["dancing"]] == 0


def test_labels_replace_teacher_and_test_set_is_held_out(tmp_path):
    labels, test = _library(tmp_path, n_videos=4)
    vocab = TL.parse_vocabulary(VOCAB_CSV)
    ds = S.build_dataset(vocab, labels, set(test), root=tmp_path / "videos", scenes={})
    tv = np.isin(ds["video"], test)
    assert ds["is_label"][tv].all()                       # test videos: your labels only (to score)
    assert (ds["w"][ds["is_label"]] == S.LABEL_WEIGHT).all()
    train_v, val_v = S.split_videos(ds["videos"], set(test))
    assert not set(test) & (set(train_v) | set(val_v))


def test_thresholds_pick_best_cutoff():
    prob = np.array([[0.9], [0.7], [0.3], [0.2], [0.65], [0.1], [0.8]])
    truth = np.array([[1], [1], [0], [0], [1], [0], [1]], float)
    assert 0.3 < S.pick_thresholds(prob, truth, min_pos=2)[0] <= 0.65
    # too few positives, or no real gain over 0.5 → stays at 0.5
    assert S.pick_thresholds(prob, truth, min_pos=10)[0] == 0.5


def test_train_and_score_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _library(tmp_path)
    monkeypatch.setattr(S, "WORK", tmp_path)
    monkeypatch.setattr(S, "HERE", tmp_path)
    assert S.main(["train", "--name", "t1", "--epochs", "12", "--folds", "3"]) == 0
    assert len(list((tmp_path / "models" / "t1").glob("head_*.pt"))) == 3
    meta = json.loads((tmp_path / "models" / "t1" / "meta.json").read_text())
    assert meta["reference_run"] == "2026-11-03_try2-base"
    sc = meta["scores"]
    # The student learns kissing from your labels alone, so it beats the teacher here.
    assert sc["rules"]["f1"] > sc["teacher"]["f1"]
    assert sc["rules"]["f1"] > 0.85
    report = (tmp_path / "student_report.md").read_text()
    assert "Student · with your vocabulary rules" in report and "| kissing |" in report
    # score again from the saved model, without training
    assert S.main(["score", "--name", "t1"]) == 0


def test_scene_answers_and_teacher_off_tags(tmp_path):
    labels, test = _library(tmp_path, n_videos=4)
    vocab = TL.parse_vocabulary(VOCAB_CSV)
    ix = {t: j for j, t in enumerate(TAGS)}
    ds0 = S.build_dataset(vocab, labels, set(test), root=tmp_path / "videos", scenes={})
    assert (ds0["w"][~ds0["is_label"], ix["kissing"]] == 0).all()   # the teacher was never asked about it
    assert ds0["counts"]["scene_extra"] == 0
    # Corrected scenes that match the made-up truth: shot changes every 20 s, so a 20 s scene is all
    # close-up or all wide; dancing is only on part of the 120 s cycle.
    scenes = {}
    for k in range(2, 4):
        v = f"vid{k:02d}"
        scenes[v] = [[float(a), float(a + 20), ["close-up" if (a // 20) % 2 == 0 else "wide"]
                      + (["dancing"] if a in (40, 60) else [])] for a in range(0, 120, 20)]
    ag = S.scene_agreement(vocab, scenes, labels, set(test))
    assert ag["close-up"]["present_n"] and ag["close-up"]["present_ok"] == ag["close-up"]["present_n"]
    rules = S.scene_rules(vocab, ag, min_n=3)
    assert {"close-up", "wide"} <= rules["absent"] and {"close-up", "wide"} <= rules["wide"]
    assert "dancing" not in rules["wide"]                            # a motion tag is never scene-wide
    ds = S.build_dataset(vocab, labels, set(test), root=tmp_path / "videos", scenes=scenes)
    assert ds["counts"]["scene_extra"] > 0
    assert "close-up" in ds["scene_rules"]["wide"] or ds["counts"]["scene_extra"] > 0
    y = np.zeros(4, np.float32); w = np.zeros(4, np.float32)
    r = {"absent": {"wide"}, "wide": set(), "with_teacher": {"dancing"}}
    S.apply_scene(y, w, ["dancing"], ix, r, set())
    assert w[ix["dancing"]] == 0 and (y[ix["wide"]], w[ix["wide"]]) == (0, S.SCENE_WEIGHT)
    S.apply_scene(y, w, ["dancing"], ix, r, {"dancing"})
    assert (y[ix["dancing"]], w[ix["dancing"]]) == (1, S.SCENE_WEIGHT)
    md = S.scene_rules_markdown(ds)
    assert "Your Reviewer scene answers" in md


def test_window_index_many_matches_single():
    times = np.arange(0, 30, 0.5)
    ts = np.array([0.0, 3.3, 29.9])
    many = S.window_index_many(times, ts)
    assert all((many[k] == S.window_index(times, t)).all() for k, t in enumerate(ts))


def test_export_every_half_second(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _library(tmp_path)
    monkeypatch.setattr(S, "WORK", tmp_path)
    monkeypatch.setattr(S, "HERE", tmp_path)
    assert S.main(["train", "--name", "e1", "--epochs", "6", "--folds", "2"]) == 0
    out = tmp_path / "scores"
    assert S.export("e1", dest=out) == 0
    idx = json.loads((out / "index.json").read_text())
    assert idx["tags"] == TAGS and len(idx["videos"]) == 14 and len(idx["cutoffs"]) == len(TAGS)
    z = np.load(out / "vid00.npz")
    assert z["scores"].shape == (len(z["times"]), len(TAGS)) and z["scores"].dtype == np.uint8
    assert len(z["times"]) == 240                                     # 2 min, every half second


def test_teacher_trust_follows_its_f1_on_your_labels(tmp_path):
    vocab = TL.parse_vocabulary(VOCAB_CSV)
    d = tmp_path / "vidA"
    d.mkdir()
    results, labels = {}, []
    for k in range(30):
        t = float(k)
        # the teacher always gets the shot right, but says dancing on every moment (only half are)
        results[f"m{k}"] = {"t": t, "tags": ["close-up", "dancing"]}
        labels.append({"id": f"tags:vidA:{t:.2f}", "type": "tags", "video_id": "vidA", "t": t,
                       "tags": ["close-up"] + (["dancing"] if k % 2 else [])})
    (d / S.TEACHER_FILE).write_text(json.dumps({"results": results}))
    tr = S.teacher_trust(vocab, labels, set(), tmp_path)
    assert tr["close-up"]["weight"] == 1.0 and tr["close-up"]["f1"] == 1.0
    assert tr["dancing"]["weight"] == pytest.approx(0.667, abs=0.01)    # F1 2/3
    assert tr["wide"]["weight"] == 1.0 and tr["wide"]["f1"] is None      # never labelled → not measured
    assert S.teacher_trust(vocab, labels, {"vidA"}, tmp_path)["dancing"]["f1"] is None   # test set never used


def _detections(n_frames=900):
    """Two people at 15 fps: a big one at the bottom, a smaller one above it; nobody in frames 300–359."""
    frame, box, kxy, kc = [], [], [], []
    for i in range(n_frames):
        if 300 <= i < 360:
            continue
        for (x1, y1, x2, y2) in ((100, 200, 500, 350), (200, 50, 400, 250)):
            frame.append(i)
            box.append([x1, y1, x2, y2, 0.9, 0])
            kxy.append(np.stack([np.linspace(x1, x2, 17), np.linspace(y1, y2, 17)], 1))
            kc.append(np.full(17, 0.8))
    return {"frame": np.array(frame, np.int32), "box": np.array(box, np.float32),
            "kxy": np.array(kxy, np.float32), "kconf": np.array(kc, np.float32)}


def test_pose_features_layout():
    det = _detections()
    f = S.pose_features(det, np.array([1.0, 22.0, 40.0, 99.0]))
    assert f.shape == (4, S.POSE_DIM)
    assert f[0, 0] == 1 and f[0, 1] == 0.5                      # two people
    assert f[1, 0] == 0                                        # 22 s: nobody within a second
    p1 = f[0, 2:2 + S.PER_PERSON]
    assert p1[2] == pytest.approx(400 / S.POSE_W)              # the bigger person comes first
    assert f[0, -1] == 1 and f[0, -3] < 0                      # second person is above the first
    assert f[3, 0] == 0                                        # past the end (> 1 s from any detection)


def test_train_with_pose(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _library(tmp_path, n_videos=6)
    for v in (tmp_path / "videos").iterdir():
        np.savez_compressed(v / "detections.npz", **_detections(1800))
    monkeypatch.setattr(S, "WORK", tmp_path)
    monkeypatch.setattr(S, "HERE", tmp_path)
    assert S.main(["train", "--name", "p1", "--epochs", "4", "--folds", "2", "--pose"]) == 0
    meta = json.loads((tmp_path / "models" / "p1" / "meta.json").read_text())
    assert meta["pose_dim"] == S.POSE_DIM and meta["dim"] == DIM
    assert (tmp_path / "videos" / "vid00" / "pose_feats.npz").exists()
    assert S.export("p1", dest=tmp_path / "out") == 0
    assert S.main(["score", "--name", "p1"]) == 0


def test_context_features_stop_at_cuts():
    times = np.arange(0, 20, 0.5)
    probs = np.zeros((len(times), 2), np.float32)
    probs[times < 10, 0] = 1.0                                   # tag 0 on for the first shot only
    f = S.context_features(times, probs, cuts=[10.0])
    assert f.shape == (len(times), 2 * S.CTX_K)
    i = int(np.searchsorted(times, 10.0))                        # the first moment after the cut
    assert f[i, 0] == 0                                          # own score
    assert f[i, 2 * 3] == 0 and f[i, 2 * 4] == 0                 # ±10 s average and ±5 s max: nothing leaks
    g = S.context_features(times, probs, cuts=[])
    assert g[i, 2 * 4] == 1.0                                    # without the cut it would


def test_train_with_context_and_export(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _library(tmp_path, n_videos=10)
    for v in (tmp_path / "videos").iterdir():
        (v / "vision.json").write_text(json.dumps({"cuts": [20.0, 40.0, 60.0, 80.0, 100.0]}))
    monkeypatch.setattr(S, "WORK", tmp_path)
    monkeypatch.setattr(S, "HERE", tmp_path)
    assert S.main(["train", "--name", "c1", "--epochs", "8", "--folds", "3", "--context"]) == 0
    meta = json.loads((tmp_path / "models" / "c1" / "meta.json").read_text())
    assert meta["context"] and "scores_step1" in meta and meta["pose_dim"] == 0
    assert len(list((tmp_path / "models" / "c1").glob("ctx_*.pt"))) == 3
    assert "step 1 only (no time context)" in (tmp_path / "student_report.md").read_text()
    assert S.export("c1", dest=tmp_path / "out") == 0
    assert S.main(["score", "--name", "c1"]) == 0


def test_picks(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    labels, test = _library(tmp_path, n_videos=8)
    monkeypatch.setattr(S, "WORK", tmp_path)
    monkeypatch.setattr(S, "HERE", tmp_path)
    assert S.main(["train", "--name", "k1", "--epochs", "4", "--folds", "2"]) == 0
    dest = tmp_path / "picks.json"
    assert S.make_picks("k1", n=20, n_test=6, dest=dest) == 0
    picks = json.loads(dest.read_text())["picks"]
    tr = [p for p in picks if p["set"] == "train"]
    te = [p for p in picks if p["set"] == "test"]
    assert 0 < len(tr) <= 20 and 0 < len(te) <= 6
    assert all(p["video_id"] in test for p in te) and not any(p["video_id"] in test for p in tr)
    assert all(p["why"] for p in tr)
    lab = {}
    for i in labels:
        lab.setdefault(i["video_id"], []).append(i["t"])
    for p in picks:                                   # never next to a moment you labelled
        assert all(abs(p["t"] - t) >= S.PICK_NEAR_LABEL for t in lab.get(p["video_id"], []))
    by_v = {}
    for p in picks:
        by_v.setdefault(p["video_id"], []).append(p["t"])
    for ts in by_v.values():                          # spread out within a video
        ts.sort()
        assert all(b - a >= S.PICK_GAP for a, b in zip(ts, ts[1:]))
    assert picks[4]["set"] == "test"                  # every 5th card a test-set one


def test_uncertainty_peaks_at_cutoff():
    u = S.uncertainty(np.array([[0.4, 0.95, 0.05]]), np.array([0.4, 0.4, 0.4]))
    assert u[0, 0] == 1 and u[0, 1] < 0.1 and u[0, 2] < 0.5


def test_greedy_picks_cap_spreads_tags():
    """With one tag far more 'useful' everywhere, the cap still hands the rest of the round to the
    others — the moment's next-best tag counts once the first is full."""
    rng = np.random.default_rng(0)
    cand = []
    for v in range(6):
        t = np.arange(0, 600, 2.0)
        u = np.stack([np.full(len(t), 0.9), rng.random(len(t)) * 0.5, rng.random(len(t)) * 0.4], 1)
        cand.append((f"v{v}", t, rng.random((len(t), 3)), u.astype(np.float32)))
    out = S.greedy_picks(cand, n=30, per_tag=10, tags=["a", "b", "c"], per_video=12, gap=10.0)
    tags = [o[2] for o in out]
    assert len(out) == 30 and tags.count("a") == 10 and tags.count("b") >= 10 and tags.count("c") >= 1
    uncapped = S.greedy_picks(cand, n=30, per_tag=0, tags=["a", "b", "c"], per_video=12, gap=10.0)
    assert [o[2] for o in uncapped].count("a") == 30
    for v in {o[0] for o in out}:                      # per video: ≤12, ≥10 s apart
        ts = sorted(o[1] for o in out if o[0] == v)
        assert len(ts) <= 12 and all(b - a >= 10 for a, b in zip(ts, ts[1:]))


def test_weighted_cutoffs_follow_your_labels_not_a_weak_teacher():
    """Teacher rows (weight 0.1) say 'yes' at low scores; your labels (×5) say only high scores are
    real. Plain picks a low cut-off to please the teacher; weighted follows your labels."""
    rng = np.random.default_rng(1)
    t_p = rng.uniform(0.25, 0.45, 400)                 # teacher: present, student scores low
    l_pos, l_neg = rng.uniform(0.6, 0.9, 30), rng.uniform(0.25, 0.45, 40)
    prob = np.concatenate([t_p, l_pos, l_neg])[:, None]
    truth = np.concatenate([np.ones(400), np.ones(30), np.zeros(40)])[:, None]
    w = np.concatenate([np.full(400, 0.1), np.full(70, 5.0)])[:, None]
    assert S.pick_thresholds(prob, truth, mask=w, mode="plain")[0] <= 0.3
    assert S.pick_thresholds(prob, truth, mask=w, mode="weighted")[0] >= 0.5
