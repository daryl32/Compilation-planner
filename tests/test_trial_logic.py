"""The Modal trial's plain-Python parts (trial_logic.py) and its report, run without a GPU."""

from __future__ import annotations

import json

import numpy as np
import pytest

import trial_logic as TL

VOCAB_CSV = ("tag,group,kind,one_of,teacher,description,not_to_be_confused_with,search_words\n"
             "close-up,shot,frame,shot size,yes,Face fills the frame,wide,\n"
             "wide,shot,frame,shot size,yes,,,\n"
             "dancing,action,motion,,yes,People dancing,walking,\n"
             "walking,action,motion,,no,,,\n")


def vocab():
    return TL.parse_vocabulary(VOCAB_CSV)


def test_parse_vocabulary_and_labels_fallback():
    v = vocab()
    assert [r["tag"] for r in v] == ["close-up", "wide", "dancing", "walking"]
    assert v[3]["teacher"] is False and v[2]["kind"] == "motion"
    assert [r["tag"] for r in TL.vocabulary_from_labels(["a", {"name": "b"}, None])] == ["a", "b"]


def test_tag_targets_and_frames():
    man = {"video_id": "v", "scenes": [
        {"scene_id": 0, "start": 0.0, "end": 4.0},
        {"scene_id": 1, "start": 4.0, "end": 39.0},                  # long → 4 moments
        {"scene_id": 2, "start": 39.0, "end": 41.0, "excluded": True},
        {"scene_id": 3, "start": 41.0, "end": 41.2}]}                # too short
    gold = [{"id": "tags:v:20.00", "type": "tags", "video_id": "v", "t": 20.0, "scene_id": 1},
            {"id": "x", "type": "tags", "video_id": "other", "t": 3.0},
            {"id": "y", "type": "tags", "video_id": "v", "t": 5.0, "skipped": True},
            {"id": "z", "type": "boundary", "video_id": "v", "t": 4.0}]
    tg = TL.tag_targets(man, gold)
    assert [t["scene_id"] for t in tg] == [0, 1, 1, 1, 1, 1]
    assert tg[0]["t"] == 2.0 and tg[-1]["gold_id"] == "tags:v:20.00"
    assert all(t["t"] <= 30 for t in TL.tag_targets(man, gold, max_seconds=30))
    times = TL.frame_times(tg[:1], duration=41.2)
    assert times[0] == 0.0 and 2.0 in times and len(times) == 7
    ctx = TL.context_times(2.0, 41.2)
    assert ctx[3] == 2.0 and set(ctx) <= set(times)
    assert TL.frame_name(2.0) == "000002.00.jpg"


def test_prompt_and_parse():
    head, moment, tail = TL.build_prompt(vocab())
    assert "Frame 4" in head and "frame 4" in moment and "JSON" in tail
    assert "- close-up: Face fills the frame (not the same as: wide)" in head
    assert "shot size: close-up, wide" in head and '"tags"' in head
    assert TL.build_prompt(vocab())[0] == head       # the same for every moment (prefix cache)
    ans = TL.parse_answer('Sure! {"tags": ["Close-Up", "wide", "dancing", "flying"], '
                          '"unsure": ["walking", "dancing"], "caption": "Two people."}', vocab())
    assert ans == {"tags": ["close-up", "dancing"], "unsure": ["walking"], "caption": "Two people.", "ok": True,
                   "rule_breaks": [], "rule_adds": [], "rule_drops": []}
    bad = TL.parse_answer("no idea", vocab())
    assert bad["tags"] == [] and not bad["ok"]


def test_person_energy_ignores_camera_moves():
    n = 30
    rng = np.random.default_rng(0)
    body = rng.uniform(-40, 40, (17, 2))
    kps = np.repeat(body[None], n, axis=0) + np.linspace(0, 300, n)[:, None, None]   # a pan, no movement
    conf = np.ones((n, 17))
    e = TL.person_energy(kps, conf, np.full(n, 200.0))
    assert np.nanmax(e) < 1e-9 and np.isnan(e[0])
    wave = kps.copy()
    wave[:, 9, 1] += 30 * np.sin(np.arange(n))                                         # a hand moving
    assert np.nanmean(TL.person_energy(wave, conf, np.full(n, 200.0))) > 0.005
    conf[5:10] = 0.0                                                                   # not seen
    e = TL.person_energy(wave, conf, np.full(n, 200.0))
    assert np.isnan(e[5:11]).all() and np.isfinite(e[12])


def test_rhythm_features_finds_the_beat():
    fps = 15.0
    t = np.arange(0, 20, 1 / fps)
    energy = 0.05 + 0.04 * (np.sin(2 * np.pi * t * 2.0) > 0.7)          # a push twice a second = 120 BPM
    r = TL.rhythm_features(energy, fps)
    assert r["movement_bpm"] == pytest.approx(120, abs=6)
    assert r["periodicity"] > 0.5 and len(r["onsets"]) >= 30
    noise = TL.rhythm_features(np.random.default_rng(1).random(300), fps)
    assert noise["periodicity"] < 0.4
    assert TL.rhythm_features([np.nan] * 5)["level"] is None


def test_scoring():
    s = TL.score_tags([(["a", "b"], ["a"]), (["a"], ["a", "c"]), ([], [])], ["a", "b", "c"])
    assert s["per_tag"]["a"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0}
    assert s["per_tag"]["b"]["precision"] == 0.0 and s["per_tag"]["c"]["recall"] == 0.0
    assert s["overall"]["tp"] == 2 and s["n"] == 3
    items = [{"type": "boundary", "video_id": "v", "t": 10.0, "verdict": "cut"},
             {"type": "boundary", "video_id": "v", "t": 20.0, "verdict": "no_cut"},
             {"type": "boundary", "video_id": "v", "t": 30.0, "verdict": "unsure"},
             {"type": "boundary", "video_id": "w", "t": 5.0, "verdict": "cut"}]
    b = TL.score_boundaries(items, {"v": [10.3]}, {"v": [20.1]})
    assert b["n"] == 2
    assert b["new"]["accuracy"] == 1.0 and b["old"]["accuracy"] == 0.0


def test_tag_names_match_ignoring_spaces_and_case():
    s = TL.score_tags([(["Close-Up Shot", "dogg ystyle"], ["close-upshot", "Doggy Style"])],
                      ["close-up shot", "Doggy Style"])
    assert s["overall"]["f1"] == 1.0


def test_cost_counts_start_up_once():
    # 90 s of footage, 60 s of work, 300 s of start-up: the library pays start-up once, not 40 times.
    c = TL.extrapolate({"tag": 60}, {"tag": "L40S"}, trial_hours=0.025, library_hours=50.0,
                       fixed={"tag": (300, 1)})
    work_per_hour = TL.stage_cost(60, "L40S") / 0.025
    assert c["tag"]["library_$"] == pytest.approx(work_per_hour * 50 + TL.stage_cost(300, "L40S"), abs=0.02)
    assert c["tag"]["trial_$"] == pytest.approx(TL.stage_cost(360, "L40S"), abs=0.01)


def test_cost():
    assert TL.stage_cost(3600, "H100", cores=0, mem_gib=0) == pytest.approx(3.95, abs=0.01)
    c = TL.extrapolate({"vision": 1800, "tag": 900}, {"vision": "L4", "tag": "H100"}, trial_hours=2.0,
                       library_hours=100.0)
    assert c["vision"]["library_$"] == pytest.approx(c["vision"]["per_footage_hour_$"] * 100, rel=0.02)
    assert c["total"]["library_$"] == pytest.approx(c["vision"]["library_$"] + c["tag"]["library_$"], abs=0.02)


# ---------------------------------------------------------------------------
# The report, end to end on made-up results
# ---------------------------------------------------------------------------

class _Vol:
    def reload(self):
        pass

    def commit(self):
        pass


def _fake_trial(tmp_path, monkeypatch):
    """A made-up trial on the Volume: pack + 1 video with vision, 8b and 32b results."""
    pytest.importorskip("modal")
    import modal_trial as M
    pack, out = tmp_path / "pack", tmp_path / "trial"
    monkeypatch.setattr(M, "PACK", str(pack))
    monkeypatch.setattr(M, "OUT", str(out))
    monkeypatch.setattr(M, "pack_vol", _Vol())
    (pack / "library").mkdir(parents=True)
    (pack / "manifests").mkdir()
    (pack / "library" / "tag_vocabulary.csv").write_text(VOCAB_CSV)
    gold = [{"id": f"tags:v:{t:.2f}", "type": "tags", "video_id": "v", "t": t, "tags": tg, "suggested": sg}
            for t, tg, sg in [(5.0, ["close-up", "dancing"], ["wide"]), (15.0, ["wide"], ["wide"]),
                              (25.0, ["close-up", "dancing"], []), (35.0, ["close-up", "dancing"], ["close-up"])]]
    gold += [{"id": "boundary:v:10.00", "type": "boundary", "video_id": "v", "t": 10.0, "verdict": "cut"},
             {"id": "boundary:v:22.00", "type": "boundary", "video_id": "v", "t": 22.0, "verdict": "no_cut"}]
    (pack / "library" / "training_labels.json").write_text(json.dumps({"items": gold}))
    (pack / "pack.json").write_text(json.dumps({"library_hours": 120.0}))
    (pack / "manifests" / "v.json").write_text(json.dumps({"video_id": "v", "scenes": [
        {"scene_id": 0, "start": 0.0, "end": 22.0, "motion_intensity": 3.0},
        {"scene_id": 1, "start": 22.0, "end": 40.0, "motion_intensity": 9.0}]}))
    (out / "v").mkdir(parents=True)
    targets = [{"key": f"g_{i['t']:.2f}", "t": i["t"], "scene_id": 0 if i["t"] < 22 else 1, "gold_id": i["id"]}
               for i in gold if i["type"] == "tags"] + [{"key": "s1_0", "t": 31.0, "scene_id": 1}]
    (out / "v" / "frames").mkdir(parents=True)
    for tg in targets:
        (out / "v" / "frames" / TL.frame_name(tg["t"])).write_bytes(b"jpg")
    import numpy as np
    np.save(out / "v" / "person_curve.npy", np.linspace(0, 1, 600).astype(np.float32))
    (out / "v" / "vision.json").write_text(json.dumps({
        "video_id": "v", "duration": 40.0, "cuts": [10.2], "timings": {"total": 120.0, "load_models": 20.0},
        "targets": targets, "pose_fps": 15.0,
        "scenes": [{"scene_id": 0, "level": 0.02, "periodicity": 0.7, "movement_bpm": 118.0},
                   {"scene_id": 1, "level": 0.05, "periodicity": 0.2, "movement_bpm": 90.0}]}))
    gold_tag_items = [i for i in gold if i["type"] == "tags"]
    perfect = {f"g_{i['t']:.2f}": {"tags": i["tags"], "gold_id": i["id"]} for i in gold_tag_items}
    sloppy = {f"g_{i['t']:.2f}": {"tags": ["wide"], "gold_id": i["id"]} for i in gold_tag_items}
    # 8b in the old file name (from before two teachers) — still read; 32b in its own file.
    (out / "v" / "tags.json").write_text(json.dumps({"seconds": 60.0, "load_seconds": 90.0, "prompts": 4,
                                                    "unreadable": 1, "results": sloppy, "gpu": "L40S"}))
    (out / "v" / "tags_32b.json").write_text(json.dumps({"seconds": 120.0, "load_seconds": 100.0, "prompts": 4,
                                                        "unreadable": 0, "results": perfect, "gpu": "H100"}))
    return M, pack, out, gold


def test_report(tmp_path, monkeypatch):
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    text = M.report.local()
    assert "1 videos, 0.01 h of footage" in text
    assert "32b **1.00**" in text and "32b 4 frames" in text   # the made-up 32b is perfect
    assert "8b" not in text and "8B" not in text           # the 8B was dropped after batch 1
    assert "| close-up | frame | 3 |" in text and "teacher = yes" in text
    assert "| tag 32b | H100 |" in text
    assert "32b: replies that weren't valid JSON 0 of 4." in text
    assert "Where the vision GPU time goes" in text and "original pipeline" in text and "| Loading models |" in text
    assert "| New (TransNetV2) | 1.00 |" in text and "| Old (PySceneDetect) | 0.00 |" in text
    data = json.loads((out / "report.json").read_text())
    assert data["best_teacher"] == "32b" and data["library_total"] > 0
    assert data["library_total"] == pytest.approx(data["cost"]["total"]["library_$"], abs=0.02)
    assert data["tags"]["verdicts"]["walking"] == "teacher off — learned from your labels"


def test_export(tmp_path, monkeypatch):
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    M.report.local()
    res = M.export.local(False)
    assert res == {"videos": 1, "frames": 5}
    data = json.loads((out / "export" / "results.json").read_text())
    assert set(data["teachers"]) == {"32b"} and (out / "export" / "report.md").exists()
    assert data["videos"]["v"]["batch"] == "1"                 # no batches.json → batch 1
    v = data["videos"]["v"]
    assert v["cuts_new"] == [10.2] and v["cuts_old"] == [22.0]
    assert {b["verdict"] for b in v["boundaries"]} == {"cut", "no_cut"}
    assert len(v["person_curve"]) == 80                       # 40 s at 2 points a second
    m = next(m for m in v["moments"] if m["key"] == "g_5.00")
    assert m["mine"] == ["close-up", "dancing"] and m["old"] == []
    assert m["teachers"] == {"32b": {"tags": ["close-up", "dancing"], "unsure": None, "caption": None,
                                     "copied_from": None}}
    assert (out / "export" / m["frame"]).exists()
    plain = next(m for m in v["moments"] if m["key"] == "s1_0")
    assert plain["mine"] is None and plain["teachers"] == {}


def test_status_and_pending(tmp_path, monkeypatch, capsys):
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    (pack / "batches.json").write_text(json.dumps({"1": ["v"], "2": ["w"]}))
    vis = json.loads((out / "v" / "vision.json").read_text())
    vis["models"] = {"pose": M.POSE_MODEL}
    (out / "v" / "vision.json").write_text(json.dumps(vis))
    t32 = json.loads((out / "v" / "tags_32b.json").read_text())
    t32["targets"] = len(vis["targets"])
    (out / "v" / "tags_32b.json").write_text(json.dumps(t32))
    (pack / "manifests" / "w.json").write_text(json.dumps({"video_id": "w", "scenes": []}))
    st = M.status.local(["32b"])
    rows = {r["video_id"]: r for r in st["rows"]}
    assert rows["v"]["batch"] == "1" and rows["v"]["vision"] and rows["v"]["tags_32b"]
    assert rows["w"]["batch"] == "2" and not rows["w"]["vision"] and not rows["w"]["tags_32b"]
    M._print_status(st, ["32b"])
    out_text = capsys.readouterr().out
    assert "to do" in out_text and "2 videos packed · vision done 1 · 32b tags done 1" in out_text
    # A vision result from an older pose model doesn't count as done.
    vis["models"] = {"pose": "yolo11l-pose.pt"}
    (out / "v" / "vision.json").write_text(json.dumps(vis))
    assert not M._vision_done("v")


def test_teacher_vocab_leaves_out_switched_off_tags():
    v = TL.teacher_vocab(vocab())
    assert [r["tag"] for r in v] == ["close-up", "wide", "dancing"]          # walking is teacher = no
    head, _, _ = TL.build_prompt(v)
    assert "\n- walking" not in head and "\n- dancing" in head
    assert TL.parse_answer('{"tags": ["walking", "dancing"]}', v)["tags"] == ["dancing"]


def test_nearest_index():
    t = np.array([0.0, 0.5, 1.0, 1.5])
    assert TL.nearest_index(t, 0.7) == 1 and TL.nearest_index(t, 0.8) == 2
    assert TL.nearest_index(t, -3) == 0 and TL.nearest_index(t, 9) == 3
    assert TL.nearest_index([], 1.0) is None


def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_pick_examples_nearest_other_videos_only():
    pool = [{"video_id": "a", "emb": _unit(1, 0, 0), "tags": ["wide"]},
            {"video_id": "b", "emb": _unit(1, 0.1, 0), "tags": ["close-up"]},
            {"video_id": "c", "emb": _unit(0, 1, 0), "tags": ["dancing"]},
            {"video_id": "d", "emb": _unit(0.9, 0.5, 0), "tags": ["wide"]}]
    got = TL.pick_examples(_unit(1, 0, 0), pool, 2, exclude_video="a")
    assert [p["video_id"] for p in got] == ["b", "d"]                       # the query's own video never
    assert TL.pick_examples(_unit(1, 0, 0), pool, 0) == []
    assert TL.pick_examples(None, pool, 3) == []


def test_pick_examples_caps_per_video_and_prefers_corrections():
    pool = [{"video_id": "a", "emb": _unit(1, 0, 0)}, {"video_id": "a", "emb": _unit(1, 0.01, 0)},
            {"video_id": "a", "emb": _unit(1, 0.02, 0)}, {"video_id": "b", "emb": _unit(0.5, 0.5, 0)}]
    assert [p["video_id"] for p in TL.pick_examples(_unit(1, 0, 0), pool, 3)] == ["a", "a", "b"]
    tie = [{"video_id": "a", "emb": _unit(1, 0.05, 0)},
           {"video_id": "b", "emb": _unit(1, 0.06, 0), "corrected": True}]
    assert TL.pick_examples(_unit(1, 0, 0), tie, 1)[0]["video_id"] == "b"


def test_example_caption_only_teacher_tags_in_vocab_order():
    v = TL.teacher_vocab(vocab())
    assert TL.example_caption(1, ["Dancing", "walking", "close-up"], v) == "Example 1 — correct tags: close-up, dancing"
    assert TL.example_caption(2, ["walking"], v) == "Example 2 — correct tags: none of the listed tags"
    assert "2 moments" in TL.examples_intro(2) and "1 moment " in TL.examples_intro(1)


def test_dedup_targets_same_scene_only_and_gold_kept():
    times = np.arange(0, 30, 0.5)
    embs = np.zeros((len(times), 3), np.float32)
    embs[times < 10] = _unit(1, 0, 0)                       # scene 0: nothing changes
    embs[(times >= 10) & (times < 20)] = _unit(0, 1, 0)     # scene 1: the same picture as...
    embs[times >= 20] = _unit(0, 1, 0)                      # ...scene 2, but a different scene
    embs[(times >= 14) & (times < 16)] = _unit(0, 0.7, 0.7)  # ...except 14–16 s, where it moves
    targets = [{"key": "a", "t": 2.0, "scene_id": 0}, {"key": "b", "t": 5.0, "scene_id": 0},
               {"key": "g", "t": 8.0, "scene_id": 0, "gold_id": "x"},
               {"key": "c", "t": 11.0, "scene_id": 1}, {"key": "d", "t": 15.0, "scene_id": 1},
               {"key": "e", "t": 18.0, "scene_id": 1}, {"key": "f", "t": 25.0, "scene_id": 2}]
    keep, copies = TL.dedup_targets(targets, times, embs, 0.96)
    assert [k["key"] for k in keep] == ["a", "g", "c", "d", "f"]
    assert copies["b"]["of"] == "a" and copies["e"]["of"] == "c"   # e copies c, not the moving d
    assert copies["b"]["similarity"] >= 0.96
    assert TL.dedup_targets(targets, times, embs, 0) == (targets, {})
    assert TL.dedup_targets(targets, None, None, 0.96) == (targets, {})


def test_gpu_use():
    u = TL.gpu_use("32b", "H100", tokens=3_000_000, seconds=1000.0)
    assert u["tokens_per_s"] == 3000 and u["mfu"] == pytest.approx(2 * 33e9 * 3000 / 989e12, abs=1e-3)
    assert TL.gpu_use("32b", "unknown", 10, 1.0)["mfu"] is None
    assert TL.gpu_use("32b", "H100", 10, 0)["tokens_per_s"] is None


def test_export_slots_and_tags_done(tmp_path, monkeypatch):
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    assert M._export_slots({"8b": 1}) == {"8b": 1}
    assert M._export_slots({"8b": 1, "32b": 2}) == {"32b": 2}
    assert M._export_slots({"32b-fp8": 3}) == {"32b": 3}                 # FP8 alone shows as the 32B
    assert M._export_slots({"32b": 2, "32b-fp8": 3}) == {"32b": 2, "32b-fp8": 3}   # both: side by side
    vis = json.loads((out / "v" / "vision.json").read_text())
    t32 = json.loads((out / "v" / "tags_32b.json").read_text())
    t32["targets"] = len(vis["targets"])
    (out / "v" / "tags_32b.json").write_text(json.dumps(t32))
    assert M._tags_done("v", "32b") and M._tags_done("v", "32b-fp8")    # same teacher, other precision
    assert not M._tags_done("v", "8b")


class _FakeOut:
    def __init__(self, text, n_prompt):
        self.prompt_token_ids = list(range(n_prompt))
        self.outputs = [type("O", (), {"text": text, "token_ids": list(range(12))})()]


class _FakeLLM:
    def __init__(self):
        self.seen = []

    def chat(self, convs, params, use_tqdm=False):
        self.seen += convs
        return [_FakeOut('{"tags": ["close-up"], "unsure": [], "caption": "c"}', 3000) for _ in convs]


def test_prep_then_tag_with_fake_model(tmp_path, monkeypatch):
    """prepare_prompts (CPU) writes one file; _tag_video (GPU) reads only that and copies the
    near-identical moments' answers."""
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    import numpy as np
    vis = json.loads((out / "v" / "vision.json").read_text())
    # two extra plain moments in scene 1 that look exactly like s1_0 → copies; and all 7 context
    # frames for every moment
    vis["targets"] += [{"key": "s1_1", "t": 33.0, "scene_id": 1}, {"key": "s1_2", "t": 36.0, "scene_id": 1}]
    (out / "v" / "vision.json").write_text(json.dumps(vis))
    for tg in vis["targets"]:
        for t in TL.context_times(tg["t"], 40.0):
            (out / "v" / "frames" / TL.frame_name(t)).write_bytes(b"jpg")
    times = np.arange(0, 40, 0.5)
    embs = np.tile(_unit(1, 0, 0), (len(times), 1)).astype(np.float16)
    embs[times < 22] = _unit(0, 1, 0)
    np.savez_compressed(out / "v" / "embeddings.npz", frame_times=times, frame_embs=embs)

    r = M.prepare_prompts.local("v", 0, 0, 0.96)
    # the 4 labelled moments always go; the 3 plain ones in scene 1 look like the labelled one at 25 s
    assert r["prompts"] == 4 and r["copied"] == 3 and r["missing_frames"] == 0
    data = json.loads((out / "v" / "prompts.json").read_text())
    assert {m["key"] for m in data["moments"]} == {"g_5.00", "g_15.00", "g_25.00", "g_35.00"}
    assert {c["of"] for c in data["copies"].values()} == {"g_25.00"}
    first = data["moments"][0]["conv"][0]["content"]
    assert first[0]["type"] == "text" and "Tags judged from frame 4" in first[0]["text"]   # head first
    assert sum(p["type"] == "image_url" for p in first) == 7

    class Self:
        key, llm, load_error, load_seconds, first = "32b-fp8", _FakeLLM(), None, 50.0, True
    monkeypatch.setattr(M, "_gpu_name", lambda d: "H100")
    import sys
    import types
    fake_vllm = types.ModuleType("vllm")            # not installed here — the GPU image has it
    fake_vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    res = M._tag_video(Self, "v", redo=True)
    assert res["prompts"] == 4 and res["copied"] == 3 and res["unreadable"] == 0
    assert len(Self.llm.seen) == 4 and not (out / "v" / "prompts.json").exists()
    tj = json.loads((out / "v" / "tags_32b-fp8.json").read_text())
    assert tj["results"]["s1_2"]["copied_from"] == "g_25.00" and tj["results"]["s1_2"]["tags"] == ["close-up"]
    assert tj["prompt_tokens"] == 12000 and tj["tokens_per_s"] and tj["mfu"] is not None
    assert tj["model"].endswith("FP8") and tj["load_seconds"] == 50.0
    # the report shows both 32B precisions side by side, with the speed table
    text = M.report.local()
    assert "| 32b F1 | 32b-fp8 F1 |" in text and "How hard the tagging GPU worked" in text
    assert "| 32b-fp8 | H100 | 1 | 4 | 3 |" in text
    assert "8b" not in text
    M.export.local(False)
    ex = json.loads((out / "export" / "results.json").read_text())
    assert set(ex["teachers"]) == {"32b", "32b-fp8"}
    v = ex["videos"]["v"]
    m = next(m for m in v["moments"] if m["key"] == "s1_2")
    assert m["teachers"]["32b-fp8"]["copied_from"] == "g_25.00"
    assert v["run"]["tagging"]["32b-fp8"]["copied"] == 3


def test_pick_check_video(tmp_path, monkeypatch):
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    (out / "w").mkdir()
    (out / "w" / "tags_32b.json").write_text(json.dumps({"results": {"a": {"gold_id": "x"}, "b": {}}}))
    assert M.pick_check_video.local() == {"video_id": "v", "labelled": 4}


def test_extrapolate_per_stage_hours():
    # a teacher that tagged 1 h of a 10 h trial is costed per THAT hour, not spread over 10
    c = TL.extrapolate({"vision": 3600.0, "tag x": 360.0}, {"vision": "L4", "tag x": "H100"},
                       {"vision": 10.0, "tag x": 1.0}, 100.0)
    assert c["tag x"]["per_footage_hour_$"] == pytest.approx(TL.stage_cost(360.0, "H100"), abs=1e-3)
    assert c["vision"]["per_footage_hour_$"] == pytest.approx(TL.stage_cost(360.0, "L4"), abs=1e-3)
    assert TL.extrapolate({"vision": 3600.0}, {"vision": "L4"}, 10.0, 100.0)["vision"] == c["vision"]


def test_report_counts_corrections_matched_by_time(tmp_path, monkeypatch):
    """A ✏️ correction has no gold_id in the teacher's results — it's matched by video + time."""
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    t32 = json.loads((out / "v" / "tags_32b.json").read_text())
    t32["results"]["s1_0"] = {"tags": ["close-up"], "t": 31.0}
    (out / "v" / "tags_32b.json").write_text(json.dumps(t32))
    labels = json.loads((pack / "library" / "training_labels.json").read_text())
    labels["items"].append({"id": "tags:v:31.00", "type": "tags", "video_id": "v", "t": 31.0,
                            "scene_id": 1, "tags": ["wide"], "suggested": ["close-up"]})
    (pack / "library" / "training_labels.json").write_text(json.dumps(labels))
    text = M.report.local()
    assert "32b 5 frames" in text and "32b **1.00**" not in text      # the correction counts, and it's wrong


def test_redo_keeps_before_and_compare(tmp_path, monkeypatch):
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    import sys
    import types
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(M, "_gpu_name", lambda d: "H100")
    vis = json.loads((out / "v" / "vision.json").read_text())
    for tg in vis["targets"]:
        for t in TL.context_times(tg["t"], 40.0):
            (out / "v" / "frames" / TL.frame_name(t)).write_bytes(b"jpg")
    # the "before" run: an FP8 run that got every labelled frame wrong
    (out / "v" / "tags_32b-fp8.json").write_text(json.dumps({
        "seconds": 600.0, "prompts": 5, "examples_used": 0, "gpu": "H100",
        "results": {f"g_{i['t']:.2f}": {"tags": ["walking"], "t": i["t"], "gold_id": i["id"]}
                    for i in gold if i["type"] == "tags"}}))
    M.prepare_prompts.local("v", 0, 0, 0)

    class Self:
        key, llm, load_error, load_seconds, first = "32b-fp8", _FakeLLM(), None, 0.0, True
    M._tag_video(Self, "v", redo=True)
    before = json.loads((out / "v" / "tags_32b-fp8_before.json").read_text())
    assert before["seconds"] == 600.0                          # the earlier run is kept
    text = M.compare_runs.local(["v"], "32b-fp8")
    assert "| before | 4 | 0.00 |" in text and "| after | 4 |" in text
    assert "Nothing to compare" in M.compare_runs.local(["nope"], "32b-fp8")


def _old_movement_and_scenes(tracks, n_frames, people, man_scenes, duration, emb_times, emb):
    """The vision code as it was inline before the split (v1.31.x), kept as the reference."""
    energies = []
    for tid, frames in tracks.items():
        if len(frames) < TL.POSE_FPS:
            continue
        a, b = min(frames), max(frames) + 1
        kps = np.zeros((b - a, 17, 2))
        conf = np.zeros((b - a, 17))
        bh = np.ones(b - a)
        for i, (xy, kc, h_) in frames.items():
            kps[i - a], conf[i - a], bh[i - a] = xy, kc, h_
        full = np.full(n_frames, np.nan)
        full[a:b] = TL.person_energy(kps, conf, bh)
        energies.append(full)
    if energies:
        E = np.vstack(energies)
        top2 = np.sort(np.nan_to_num(E, nan=0.0), axis=0)[-2:].sum(axis=0)
        seen = np.isfinite(E).any(axis=0)
        curve = np.where(seen, top2, np.nan)
    else:
        curve = np.full(n_frames, np.nan)
    emb = np.vstack([emb]) if len(emb) else np.zeros((0, 1), np.float16)
    et = np.array(emb_times)
    scenes, scene_embs, scene_ids = [], [], []
    for s in man_scenes:
        if "start" not in s or s["start"] >= duration:
            continue
        a, b = int(s["start"] * TL.POSE_FPS), int(min(s["end"], duration) * TL.POSE_FPS)
        seg = curve[a:b]
        row = {"scene_id": s["scene_id"], "start": s["start"], "end": min(s["end"], duration),
               "people": float(np.median(people[a:b])) if b > a else 0.0, **TL.rhythm_features(seg)}
        sel = (et >= s["start"]) & (et < min(s["end"], duration)) if et.size else np.array([], bool)
        if sel.any():
            v = emb[sel].astype(np.float32).mean(axis=0)
            scene_embs.append((v / (np.linalg.norm(v) or 1)).astype(np.float16))
            scene_ids.append(s["scene_id"])
        scenes.append(row)
    return curve, scenes, scene_ids, scene_embs


def test_movement_and_scene_features_match_old_code():
    rng = np.random.default_rng(3)
    n = 15 * 40                                            # 40 s at 15 fps
    tracks = {}
    for tid, (a, b) in enumerate([(0, 300), (100, 600), (50, 60), (400, 590)]):
        tracks[tid] = {i: (rng.normal(300, 40, (17, 2)) + np.sin(i / 3) * 10, rng.uniform(0.2, 1, 17),
                           float(rng.uniform(150, 250))) for i in range(a, b)}
    people = [sum(i in fr for fr in tracks.values()) for i in range(n)]
    man_scenes = [{"scene_id": 0, "start": 0.0, "end": 12.0}, {"scene_id": 1, "start": 12.0, "end": 33.0},
                  {"scene_id": 2, "start": 33.0, "end": 45.0}, {"scene_id": 3}]
    et = np.arange(0, 40, 0.53)
    emb = rng.normal(size=(len(et), 8)).astype(np.float16)
    c_old, s_old, ids_old, e_old = _old_movement_and_scenes(tracks, n, people, man_scenes, 40.0, et, emb)
    c_new = TL.movement_curve(tracks, n)
    s_new, ids_new, e_new = TL.scene_features(man_scenes, 40.0, c_new, people, et, emb)
    assert np.array_equal(np.isnan(c_old), np.isnan(c_new)) and np.allclose(np.nan_to_num(c_old), np.nan_to_num(c_new))
    assert json.dumps(s_old, sort_keys=True, default=float) == json.dumps(s_new, sort_keys=True, default=float)
    assert ids_old == ids_new and all(np.array_equal(x, y) for x, y in zip(e_old, e_new))
    assert np.isnan(TL.movement_curve({}, 10)).all()


def test_cuts_from_predictions_matches_old_rule():
    p = np.zeros(300)
    p[[2, 3]] = 0.9            # 0.1 s in at 25 fps → dropped (≤ 0.2 s)
    p[100:103] = 0.8           # one run → one cut at its middle
    p[200] = 0.51
    assert TL.cuts_from_predictions(p, 25.0) == [round(102 / 25 - 1 / 25 + 0.5 / 25, 2), round(200.5 / 25, 2)]
    assert TL.cuts_from_predictions(np.zeros(10), 25.0) == []


def test_vision_snapshot_compare_restore(tmp_path, monkeypatch):
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    vis = json.loads((out / "v" / "vision.json").read_text())
    vis.update({"tracks": 6, "gpu": "A10", "scenes": [{"scene_id": 0, "level": 0.02, "periodicity": 0.7,
                                                       "movement_bpm": 118.0, "people": 2.0}]})
    (out / "v" / "vision.json").write_text(json.dumps(vis))
    np.savez_compressed(out / "v" / "embeddings.npz", frame_times=np.arange(3.0), frame_embs=np.ones((3, 2)))
    snap = M.vision_snapshot.local("")
    assert snap["video_id"] == "v" and (out / "v" / "vision_before.json").exists()
    assert (out / "v" / "person_curve_before.npy").exists() and (out / "v" / "embeddings_before.npz").exists()
    new = {**vis, "pipeline": 2, "cuts": [10.2, 30.0], "timings": {"total": 40.0, "pose_gpu": 20.0},
           "cpu_timings": {"total": 60.0}, "tracks": 6}
    (out / "v" / "vision.json").write_text(json.dumps(new))
    text = M.vision_compare.local("v")
    assert "| Scene cuts | 1 | 2 (1 within 0.1 s of an old one) |" in text
    assert "correlation 1.0000" in text and "level biggest difference 0 (0 scenes differ)" in text
    assert json.loads((out / "v" / "vision.json").read_text()).get("pipeline", 1) == 1      # restored
    assert "Nothing to compare" in M.vision_compare.local("v")                            # old one is back


def test_vision_motion_writes_done_marker(tmp_path, monkeypatch):
    """vision_motion (CPU) turns the GPU step's files into vision.json, embeddings.npz and the curve."""
    M, pack, out, gold = _fake_trial(tmp_path, monkeypatch)
    man = json.loads((pack / "manifests" / "v.json").read_text())
    man["proxy"] = {"duration": 40.0, "width": 854, "height": 480}
    (pack / "manifests" / "v.json").write_text(json.dumps(man))
    vg = json.loads((out / "v" / "vision.json").read_text())
    (out / "v" / "vision.json").unlink()
    vg.update({"pipeline": 2, "gpu": "A10", "models": {"pose": M.POSE_MODEL}})
    vg.pop("scenes")
    (out / "v" / "vision_gpu.json").write_text(json.dumps(vg))
    n = 600
    np.savez_compressed(out / "v" / "detections.npz", frame=np.zeros(0, np.int32), box=np.zeros((0, 6), np.float32),
                        kxy=np.zeros((0, 17, 2), np.float32), kconf=np.zeros((0, 17), np.float32),
                        n_frames=np.int32(n))
    np.savez_compressed(out / "v" / "frame_embs.npz", frame_times=np.arange(0, 40, 0.5),
                        frame_embs=np.ones((80, 4), np.float16))
    (out / "v" / "shots_meta.json").write_text(json.dumps({"fps": 25.0, "seconds": 3.0}))
    rng = np.random.default_rng(1)
    tracks = {1: {i: (rng.normal(300, 30, (17, 2)), np.ones(17), 200.0) for i in range(0, 400)}}
    monkeypatch.setattr(M, "_frames", lambda cmd, w, h, ahead=64: iter(()))
    monkeypatch.setattr(M, "_track_offline", lambda frames, det, n_frames: (tracks, [1] * 400 + [0] * 200, "tracktrack"))
    assert not M._vision_done("v")
    r = M.vision_motion.local("v")
    assert r["tracks"] == 1
    vis = json.loads((out / "v" / "vision.json").read_text())
    assert vis["pipeline"] == 2 and vis["tracks"] == 1 and len(vis["scenes"]) == 2
    assert vis["scenes"][0]["people"] == 1.0 and vis["cpu_timings"]["prep"] == 3.0
    assert (out / "v" / "embeddings.npz").exists() and not (out / "v" / "frame_embs.npz").exists()
    assert len(np.load(out / "v" / "person_curve.npy")) == n
    assert M._vision_done("v")
    # the report costs vision from the batched video and lists the CPU share
    text = M.report.local()
    assert "| vision CPU | CPU |" in text and "batched pipeline" in text and "Pose (GPU, batched)" in text


NEVER_CSV = ("tag,group,kind,one_of,teacher,description,not_to_be_confused_with,never_with,search_words\n"
             "standing,position,frame,,yes,,,,\n"
             "lying,position,frame,,yes,,,,\n"
             "jumping,action,motion,,yes,,,standing; lying,\n"
             "dancing,action,motion,,yes,,,lying,\n"
             "waving,action,motion,,no,,,standing,\n")


def test_never_with_prompt_and_answers():
    v = TL.parse_vocabulary(NEVER_CSV)
    assert v[2]["never_with"] == ["standing", "lying"]
    pairs = TL.never_pairs(v)
    assert pairs["lying"] == {"jumping", "dancing"} and pairs["standing"] == {"jumping", "waving"}  # both ways
    tv = TL.teacher_vocab(v)                     # waving is teacher = no → its rule isn't the teacher's
    assert "waving" not in str(TL.never_pairs(tv))
    head, _, _ = TL.build_prompt(tv)
    assert "never apply to the same moment" in head
    assert "- standing / jumping" in head and "- lying / jumping" in head and "- lying / dancing" in head
    # the first-named of a clashing pair stays; the other goes to unsure
    ans = TL.parse_answer('{"tags": ["jumping", "standing", "dancing", "lying"], "unsure": []}', tv)
    assert ans["tags"] == ["jumping", "dancing"] and ans["unsure"] == ["standing", "lying"]
    assert ans["rule_breaks"] == ["standing", "lying"]
    ans = TL.parse_answer('{"tags": ["lying", "jumping"], "unsure": ["jumping"]}', tv)
    assert ans["tags"] == ["lying"] and ans["unsure"] == ["jumping"]
    # no rules → no rules section, nothing moved
    assert "never apply" not in TL.build_prompt(TL.teacher_vocab(vocab()))[0]


ALWAYS_CSV = ("tag,group,kind,one_of,teacher,description,not_to_be_confused_with,never_with,always_with,search_words\n"
              "close-up,shot,frame,shot size,yes,,,,,\n"
              "wide,shot,frame,shot size,yes,,,,,\n"
              "kissing,action,frame,,yes,,,,close-up,\n"
              "face,body,frame,,yes,,,,,\n"
              "lips,body,frame,,yes,,,,face,\n"
              "talking,action,frame,,yes,,,,lips,\n"
              "standing,position,frame,,yes,,,,,\n"
              "lying,position,frame,,yes,,,standing,,\n")


def test_always_with_prompt_and_answers():
    v = TL.teacher_vocab(TL.parse_vocabulary(ALWAYS_CSV))
    assert TL.always_map(v) == {"kissing": ["close-up"], "lips": ["face"], "talking": ["lips"]}   # one way
    head, _, _ = TL.build_prompt(v)
    assert "always give the second as well" in head and "- kissing → close-up" in head
    # a missing partner is added, following chains (talking → lips → face)
    ans = TL.parse_answer('{"tags": ["talking"]}', v)
    assert ans["tags"] == ["talking", "lips", "face"] and ans["rule_adds"] == ["lips", "face"]
    # ...but never one that breaks a pick-one set (wide already chosen for shot size)
    ans = TL.parse_answer('{"tags": ["wide", "kissing"]}', v)
    assert ans["tags"] == ["wide", "kissing"] and ans["rule_adds"] == []
    # the other way round adds nothing (close-up doesn't need kissing)
    assert TL.parse_answer('{"tags": ["close-up"]}', v)["tags"] == ["close-up"]


ONLY_CSV = ("tag,group,kind,one_of,teacher,description,not_to_be_confused_with,never_with,always_with,only_with,"
            "search_words\n"
            "close-up,shot,frame,shot size,yes,,,,,,\n"
            "kneeling,position,frame,,yes,,,,,praying; proposing,\n"
            "praying,action,frame,,yes,,,,,,\n"
            "proposing,action,frame,,yes,,,,,,\n"
            "ring,object,frame,,yes,,,,,proposing,\n"
            "hug,action,frame,,yes,,,,ring,,\n"
            "secret,action,frame,,no,,,,,,\n"
            "hidden,position,frame,,yes,,,,,secret,\n")


def test_only_with_prompt_and_answers():
    v = TL.teacher_vocab(TL.parse_vocabulary(ONLY_CSV))
    assert v[1]["only_with"] == ["praying", "proposing"]
    # one way, and a rule whose only partner isn't a teacher tag is ignored
    assert TL.only_map(v) == {"kneeling": ["praying", "proposing"], "ring": ["proposing"]}
    head, _, _ = TL.build_prompt(v)
    assert "only with praying or proposing" in head and "- ring: only with proposing" in head
    assert "hidden: only" not in head
    # any one partner is enough
    assert TL.parse_answer('{"tags": ["kneeling", "proposing"]}', v)["tags"] == ["kneeling", "proposing"]
    assert TL.parse_answer('{"tags": ["praying", "kneeling"]}', v)["tags"] == ["praying", "kneeling"]
    # none there → the tag goes to unsure; the partner is NOT added
    ans = TL.parse_answer('{"tags": ["close-up", "kneeling"], "unsure": []}', v)
    assert ans["tags"] == ["close-up"] and ans["unsure"] == ["kneeling"] and ans["rule_drops"] == ["kneeling"]
    # an always-with add that breaks its own only-with is dropped again, and not counted as added
    ans = TL.parse_answer('{"tags": ["hug"]}', v)
    assert ans["tags"] == ["hug"] and ans["rule_adds"] == [] and ans["rule_drops"] == ["ring"]
    # the reverse needs nothing (proposing alone is fine)
    assert TL.parse_answer('{"tags": ["proposing"]}', v)["tags"] == ["proposing"]
    # no only_with column (older files) → no rule
    assert TL.only_map(TL.parse_vocabulary(ALWAYS_CSV)) == {}


USUALLY_CSV = ("tag,group,kind,one_of,teacher,never_with,only_with,usually_with\n"
               "close-up,shot,frame,shot size,yes,,,\n"
               "medium,shot,frame,shot size,yes,,,\n"
               "wide,shot,frame,shot size,yes,,,\n"
               "kissing,action,frame,,yes,,,close-up; hands\n"
               "hands,body,frame,,yes,,,\n"
               "night,light,frame,,yes,,,\n"
               "running,action,frame,,yes,,,night; sweat\n"
               "sweat,body,frame,,yes,running,,\n"
               "ring,object,frame,,yes,,proposing,\n"
               "proposing,action,frame,,yes,,,\n"
               "hug,action,frame,,yes,,,ring\n")


def test_usually_with_second_look():
    import json as _json
    v = TL.teacher_vocab(TL.parse_vocabulary(USUALLY_CSV))
    assert TL.usually_map(v) == {"kissing": ["close-up", "hands"], "running": ["night", "sweat"], "hug": ["ring"]}
    head, _, _ = TL.build_prompt(v)
    assert "often (not always) go together" in head and "- kissing: often with close-up, hands" in head
    first = TL.parse_answer('{"tags": ["medium", "kissing"], "unsure": ["hands"]}', v)
    asks = TL.look_again(first, v)
    # a pick-one partner is asked against the one already chosen; an unsure one is asked again
    assert asks == [{"tag": "close-up", "because": "kissing", "instead_of": "medium"},
                    {"tag": "hands", "because": "kissing", "instead_of": None}]
    q = TL.look_again_prompt(asks)
    assert "better fit than medium" in q and "hands (often with kissing): does it apply?" in q and '"yes"' in q
    # yes to close-up → it replaces medium (medium goes to unsure); no to hands
    got = TL.apply_look_again(first, '{"yes": ["close-up"], "no": ["hands"]}', asks, v)
    assert got["tags"] == ["kissing", "close-up"] and "medium" in got["unsure"] and "hands" in got["unsure"]
    assert got["look_asked"] == ["close-up", "hands"] and got["look_added"] == ["close-up"]
    # all no / unreadable → nothing changes
    for reply in ('{"yes": [], "no": ["close-up", "hands"]}', "hmm"):
        same = TL.apply_look_again(first, reply, asks, v)
        assert same["tags"] == first["tags"] and same["look_added"] == []
    # never asked: sweat (never-with running) or breaks its own only-with; nothing to ask if all given
    assert [a["tag"] for a in TL.look_again(TL.parse_answer('{"tags": ["running"]}', v), v)] == ["night"]
    assert TL.look_again(TL.parse_answer('{"tags": ["hug"]}', v), v) == []          # ring needs proposing
    assert TL.look_again(TL.parse_answer('{"tags": ["kissing", "close-up", "hands"]}', v), v) == []
    # a "yes" the rules don't allow isn't kept (said yes to something not asked → ignored)
    got = TL.apply_look_again(first, _json.dumps({"yes": ["wide"]}), asks, v)
    assert got["tags"] == first["tags"]
    # no usually_with anywhere → no section, nothing to ask
    plain = TL.teacher_vocab(TL.parse_vocabulary(ALWAYS_CSV))
    assert "often (not always)" not in TL.build_prompt(plain)[0]
    assert TL.look_again(TL.parse_answer('{"tags": ["kissing"]}', plain), plain) == []
