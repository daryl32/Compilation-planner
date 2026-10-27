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
    before, after = TL.build_prompt(vocab())
    assert "Frame 4" in before
    assert "- close-up: Face fills the frame (not the same as: wide)" in after
    assert "shot size: close-up, wide" in after and '"tags"' in after
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


def test_report(tmp_path, monkeypatch):
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
    (out / "v" / "vision.json").write_text(json.dumps({
        "video_id": "v", "duration": 40.0, "cuts": [10.2], "timings": {"total": 120.0},
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
    text = M.report.local()
    assert "1 videos, 0.01 h of footage" in text
    assert ", 8b **" in text and ", 32b **1.00**" in text  # the made-up 32b is perfect
    assert "**Better teacher: 32b**" in text
    assert "| close-up | frame | 3 |" in text and "teacher = yes" in text
    assert "| tag 8b | L40S |" in text and "| tag 32b | H100 |" in text
    assert "8b: replies that weren't valid JSON 1 of 4." in text
    assert "| New (TransNetV2) | 1.00 |" in text and "| Old (PySceneDetect) | 0.00 |" in text
    data = json.loads((out / "report.json").read_text())
    assert data["best_teacher"] == "32b" and data["library_total"] > 0
    assert data["library_total"] < data["cost"]["total"]["library_$"]      # one teacher, not both
    assert data["tags"]["verdicts"]["walking"] == "need more labels"
