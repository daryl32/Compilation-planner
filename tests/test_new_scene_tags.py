"""🏷️ New scene tags (Stage 3 step 1): the teacher's moment tags put onto scenes — time mapping, the half
rule, label priority, unsure tags, the newest teacher, fallback, the Old / New switch and the catalogue loader."""

from __future__ import annotations

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)

import new_scene_tags as NST
from library_common import scene_tags


def _tc(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _cat(vid: str, bounds: list, tags: list, corrected: dict = None) -> dict:
    scenes = []
    for i, ((a, b), t) in enumerate(zip(bounds, tags)):
        s = {"scene_id": i, "start_tc": _tc(a), "end_tc": _tc(b), "tags": list(t), "motion_intensity": 0.1 * i}
        if corrected and i in corrected:
            s["corrected_tags"] = corrected[i]
        scenes.append(s)
    return {"video_id": vid, "scenes": scenes}


def _m(t, tags=None, unsure=(), mine=None, teacher="32b", more=None):
    teachers = {teacher: {"tags": list(tags or []), "unsure": list(unsure)}}
    teachers.update(more or {})
    return {"key": f"k{t}", "t": t, "mine": mine, "teachers": teachers}


def _results(moments: dict, runs: dict = None) -> dict:
    return {"created": "2026-11-01 10:00",
            "videos": {vid: {"moments": ms, "run": {"tagging": (runs or {}).get(vid, {})}}
                       for vid, ms in moments.items()}}


@pytest.fixture(autouse=True)
def _clean():
    for p in (NST.FILE, NST.SETTINGS):
        p.unlink(missing_ok=True)
    NST._CACHE.clear()
    yield
    for p in (NST.FILE, NST.SETTINGS):
        p.unlink(missing_ok=True)
    NST._CACHE.clear()


def test_time_mapping_half_rule_and_share():
    cat = _cat("v", [(0, 10), (10, 20), (20, 30)], [["old"], ["old"], ["kept"]])
    res = _results({"v": [_m(1, ["a", "b"]), _m(4, ["a"]), _m(7, ["a", "c"]),       # scene 0: a 3/3, b 1/3, c 1/3
                          _m(10, ["x"]), _m(15, ["x", "y"])]})                       # scene 1: x 2/2, y 1/2 (half → on)
    data, summ = NST.build(res, [], {"v": cat})
    v = data["videos"]["v"]
    assert v["0"]["tags"] == ["a"] and v["0"]["share"]["a"] == 1.0 and v["0"]["share"]["b"] == 0.33
    assert v["0"]["moments"] == 3
    assert v["1"]["tags"] == ["x", "y"]                    # 1 of 2 counts (start ≤ t: t=10 is scene 1)
    assert "2" not in v                                    # no moment → falls back to its old tags
    assert summ["scenes_new"] == 2 and summ["scenes_fallback"] == 1 and summ["scenes_changed"] == 2
    assert data["export"] == "2026-11-01 10:00"


def test_split_scenes_line_up_by_time_not_id():
    # The Reviewer split the pack's scene 0 (0–10) into 0 (0–5) and 7 (5–10)
    cat = _cat("v", [(0, 5), (5, 10)], [["a"], ["a"]])
    cat["scenes"][1]["scene_id"] = 7
    res = _results({"v": [dict(_m(2, ["p"]), scene_id=0), dict(_m(8, ["q"]), scene_id=0)]})
    v = NST.build(res, [], {"v": cat})[0]["videos"]["v"]
    assert v["0"]["tags"] == ["p"] and v["7"]["tags"] == ["q"]


def test_labels_win_unsure_left_out_and_saved_labels_count():
    cat = _cat("v", [(0, 10)], [["old"]])
    res = _results({"v": [_m(1, ["a"], unsure=["u"], mine=["mine1"]),       # exported label wins
                          _m(3, ["a"], unsure=["u"]),                      # label saved after the export wins
                          _m(5, ["a", "b"], unsure=["u"])]})
    items = [{"id": "tags:v:3.00", "type": "tags", "tags": ["mine1", "b"]}]
    e = NST.build(res, items, {"v": cat})[0]["videos"]["v"]["0"]
    assert e["labelled"] == 2 and e["moments"] == 3
    assert "u" not in e["share"]                           # unsure never counts
    assert e["tags"] == ["b", "mine1"]                     # b 2/3, mine1 2/3; a only 1/3


def test_newest_teacher_run_is_used():
    cat = _cat("v", [(0, 10)], [["old"]])
    other = {"8b": {"tags": ["from8b"]}}
    res = _results({"v": [_m(1, ["from32"], more=other)]},
                   runs={"v": {"32b": {"at": 100.0}, "8b": {"at": 200.0}}})
    assert NST.build(res, [], {"v": cat})[0]["videos"]["v"]["0"]["tags"] == ["from8b"]
    res = _results({"v": [_m(1, ["from32"], more=other)]},
                   runs={"v": {"32b": {"at": 300.0}, "8b": {"at": 200.0}}})
    assert NST.build(res, [], {"v": cat})[0]["videos"]["v"]["0"]["tags"] == ["from32"]


def test_videos_not_in_library_are_counted_not_built():
    data, summ = NST.build(_results({"gone": [_m(1, ["a"])]}), [], {})
    assert data["videos"] == {} and summ["videos_missing"] == 1


def test_switch_and_scene_tags():
    assert NST.mode() == "old"                             # starts on Old
    s = {"tags": ["old"], "new_tags": ["new"]}
    assert scene_tags(s) == ["old"]
    NST.set_mode("new")
    assert NST.mode() == "new"
    assert scene_tags(s) == ["new"]
    assert scene_tags({"tags": ["old"]}) == ["old"]                                   # fallback scene
    assert scene_tags({**s, "corrected_tags": ["mine"]}) == ["mine"]                  # your tags always win
    assert scene_tags({**s, "corrected_tags": []}) == []
    NST.set_mode("old")
    assert scene_tags(s) == ["old"]


def test_catalogue_loader_follows_the_switch_and_sigs_change():
    import catalogue_store as CS
    import rec_library as RL
    cats = NST.raw_catalogues()
    vid = sorted(cats)[0]
    cat = cats[vid]
    first = cat["scenes"][0]
    a = NST._tc(first["start_tc"])
    res = _results({vid: [_m(a + 0.01, ["brand-new-tag"])]})
    data, _ = NST.build(res, [], cats)
    sid = str(first["scene_id"])
    assert data["videos"][vid][sid]["tags"] == ["brand-new-tag"]
    old_sigs = RL.video_sigs()
    NST.save(data)

    def tags_now():
        sc = next(s for s in CS.all_catalogues()[vid]["scenes"] if str(s["scene_id"]) == sid)
        slim = next(s for v in CS.video_index().values() if v["video_id"] == vid
                    for s in v["scenes"] if str(s["scene_id"]) == sid)
        return sc["tags"], slim["tags"], sc

    tags, slim, sc = tags_now()
    want_old = first["corrected_tags"] if first.get("corrected_tags") is not None else first["tags"]
    assert tags == want_old and slim == want_old and sc["new_tags"] == ["brand-new-tag"]
    assert RL.video_sigs() == old_sigs                     # Old: searches see exactly what they saw
    NST.set_mode("new")
    tags, slim, sc = tags_now()
    if first.get("corrected_tags") is None:
        assert tags == ["brand-new-tag"] and slim == ["brand-new-tag"]
    assert sc["auto_tags"] == first["tags"]
    sigs = RL.video_sigs()
    assert sigs[vid] != old_sigs[vid]
    assert all(sigs[v] == old_sigs[v] for v in sigs if v != vid)
    # 🆚 compare rows keep showing the file's own tags as "old" even with the switch on New
    rows = [r for r in NST.compare_rows(CS.all_catalogues(), data) if r["video_id"] == vid]
    r0 = next(r for r in rows if str(r["scene_id"]) == sid)
    assert r0["old"] == want_old and r0["new"] == ["brand-new-tag"]
    assert all(r["new"] is None for r in rows if r is not r0)
    NST.set_mode("old")
    assert tags_now()[0] == want_old
    assert RL.video_sigs() == old_sigs


def test_compare_helpers_and_similar_scenes():
    cats = {"v": _cat("v", [(0, 10), (10, 20)], [["a", "b"], ["c"]]),
            "w": _cat("w", [(0, 10), (10, 20)], [["a", "b"], ["z"]]),
            "x": _cat("x", [(0, 10)], [["z"]])}
    res = _results({"v": [_m(1, ["z"]), _m(11, ["c"])]})
    data, _ = NST.build(res, [], cats)
    rows = NST.compare_rows(cats, data)
    r = next(r for r in rows if r["video_id"] == "v" and r["scene_id"] == 0)
    assert r["added"] == ["z"] and r["removed"] == ["a", "b"]
    assert NST.tag_changes(rows)[0][0] in {"a", "b", "z"}
    old = NST.similar_scenes(("v", 0), cats, data, "old")
    new = NST.similar_scenes(("v", 0), cats, data, "new")
    assert (old[0][1], old[0][2]) == ("w", 0)              # a, b — like w's first scene
    assert new[0][3] == ["z"]                              # z — like the z scenes


@pytest.fixture
def library_bundle(tmp_path, monkeypatch):
    """A trial export for one library video (its first two scenes tagged), with labels / pending saved aside
    and the recommendation queue stubbed."""
    import json
    import shutil

    import jobs as J
    import trial_results as TR
    import training_labels as TLB
    from library_common import PENDING_FILE
    saved = {}
    for p in (TLB.LABELS_FILE, PENDING_FILE):
        if p.exists():
            saved[p] = tmp_path / p.name
            shutil.copy2(p, saved[p])
            p.unlink()
    queued = []
    monkeypatch.setattr(J, "queue_changed", lambda why="changed": queued.append(why) or 3)
    cats = NST.raw_catalogues()
    vid = sorted(cats)[0]
    sc = sorted(cats[vid]["scenes"], key=lambda s: NST._tc(s["start_tc"]))
    moments = [_m(NST._tc(sc[0]["start_tc"]) + 0.01, ["zebra-tag"]), _m(NST._tc(sc[1]["start_tc"]) + 0.01, ["yak-tag"])]
    shutil.rmtree(TR.RESULTS_DIR, ignore_errors=True)
    TR.RESULTS_DIR.mkdir(parents=True)
    bundle = {**_results({vid: moments}), "version": 1, "teachers": {"32b": "Qwen/Qwen3-VL-32B-Instruct"},
              "report": {}}
    bundle["videos"][vid].update(batch="1", duration=60.0, scenes=[], cuts_new=[], cuts_old=[], boundaries=[])
    (TR.RESULTS_DIR / "results.json").write_text(json.dumps(bundle))
    TR._CACHE.clear()
    yield vid, queued
    shutil.rmtree(TR.RESULTS_DIR, ignore_errors=True)
    TR._CACHE.clear()
    for p in (TLB.LABELS_FILE, PENDING_FILE):
        p.unlink(missing_ok=True)
        if p in saved:
            shutil.copy2(saved[p], p)


def test_scene_tags_tab(library_bundle):
    from library_common import read_pending
    from tests.apptest_util import errors, page_app
    vid, queued = library_bundle
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "🧪 Trial results"
    at.session_state["tr_sub"] = "🏷️ Scene tags"
    at.run()
    assert not errors(at), errors(at)
    assert any("No new scene tags yet" in i.value for i in at.info)
    at.button(key="nst_build").click().run()
    assert not errors(at), errors(at)
    assert NST.FILE.exists() and NST.FILE_STEM in read_pending()
    assert any("Built: **2** scene(s) in **1** video(s)" in s.value for s in at.success)
    assert queued == []                                    # the switch is on Old — nothing to search again
    md = " ".join(m.value for m in at.markdown)
    assert "zebra-tag" in md and "yak-tag" in md
    # 🆕 New: the whole app follows, and the searches are queued again
    at.button_group(key="nst_mode").set_value("new").run()
    assert not errors(at), errors(at)
    assert NST.mode() == "new" and queued == ["scene tags: new"]
    import catalogue_store as CS
    assert any(s["tags"] == ["zebra-tag"] for s in CS.all_catalogues()[vid]["scenes"]
               if s.get("corrected_tags") is None) or all(s.get("corrected_tags") is not None
                                                           for s in CS.all_catalogues()[vid]["scenes"][:1])
    # 🔎 More like this both ways
    at.button_group(key="nst_part").set_value("similar").run()
    assert not errors(at), errors(at)
    assert any("Old tags" in m.value for m in at.markdown) and any("New tags" in m.value for m in at.markdown)
    # back to Old
    at.button_group(key="nst_mode").set_value("old").run()
    assert not errors(at), errors(at)
    assert NST.mode() == "old"
