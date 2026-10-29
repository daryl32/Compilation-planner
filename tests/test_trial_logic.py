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
    assert ans == {"tags": ["close-up", "dancing"], "unsure": ["walking"], "caption": "Two people.", "ok": True}
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
    assert ", 32b **1.00**" in text  # the made-up 32b is perfect
    assert "8b" not in text and "8B" not in text           # the 8B was dropped after batch 1
    assert "| close-up | frame | 3 |" in text and "teacher = yes" in text
    assert "| tag 32b | H100 |" in text
    assert "32b: replies that weren't valid JSON 0 of 4." in text
    assert "Where the vision time goes" in text and "| Loading models |" in text
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
