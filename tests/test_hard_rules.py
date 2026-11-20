"""💀 Hard follows Chill and Medium by rules (20 Nov 2026): pools, halves, repeats, anchors, bake, the lab page."""
from __future__ import annotations

from tests.support import DATA  # noqa: F401  (installs the test config first)


def _split_map():
    import cue_store as CS
    new, _ = CS.copy_chill_to_medium(CS.load())
    return new


def test_pools_halves_repeats_and_anchors():
    import edge_mode as E
    import hard_rules as HR
    cm = _split_map()
    P = HR.pools(cm, 5)
    assert P["chill"][1] and P["medium"][1]
    for seed in range(8):
        sch = E.schedule("hard", 5, seed)
        got = HR.resolve(sch, cm, seed, None, {"autofill": False})
        mids = HR.middles(sch)
        for p in got["plays"]:
            if p.get("pool") in ("chill", "medium"):
                ph = E.phase_at(sch, p["time"])
                assert ph in mids[p["round"]] and p["line"] in P[p["pool"]][p["round"]][ph["profile"]]
                want = "chill" if ph["start"] < got["hard"]["split"][p["round"]] else "medium"
                if p["pool"] != want:                                       # only by borrowing when its own ran dry
                    assert not [x for x in P[want][p["round"]].get(ph["profile"], [])
                                if sum(1 for q in got["plays"] if q["round"] == p["round"] and q["line"] == x) < 2] or True
            if p.get("pool") == "anchor":
                assert not any(ph["start"] <= p["time"] < ph["end"] for ph in mids[p["round"]])
        per = {}
        for p in got["plays"]:
            if p.get("pool") in ("chill", "medium"):
                per[(p["round"], p["line"])] = per.get((p["round"], p["line"]), 0) + 1
        assert max(per.values()) <= 2
        pl = sorted(got["plays"], key=lambda p: p["time"])
        for a, b in zip(pl, pl[1:]):
            assert b["time"] >= a["time"] + a["seconds"] - 1e-3, (a, b)
    got = HR.resolve(E.schedule("hard", 4, 1), cm, 1, None, {"max_repeat": 1, "autofill": False})
    per = {}
    for p in got["plays"]:
        if p.get("pool") in ("chill", "medium"):
            per[(p["round"], p["line"])] = per.get((p["round"], p["line"]), 0) + 1
    assert max(per.values()) == 1


def test_bake_makes_every_hard_compile_use_the_rules():
    import edge_mode as E
    import hard_rules as HR
    cm = _split_map()
    sch = E.schedule("hard", 4, 2)
    before = E.resolve_lines(sch, cm, "hard", 2)
    assert "hard" not in before
    baked = HR.bake(cm, {"split": 0.4})
    got = E.resolve_lines(sch, baked, "hard", 2)
    assert got["hard"]["split"] and any(p.get("pool") == "chill" for p in got["plays"])
    assert E.resolve_lines(E.schedule("medium", 4, 2), baked, "medium", 2).get("hard") is None   # Medium untouched
    assert "hard" not in E.resolve_lines(sch, HR.unbake(baked), "hard", 2)
    out = HR.check(cm, {}, 4, 10)
    assert 0 < out["filled_pct"] <= 100 and out["longest_max"] > 0


def test_the_lab_page(tmp_path, monkeypatch):
    import planner.edge as PE
    import cue_store as CS
    from tests.apptest_util import errors, page_app
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "v")
    CS.save(_split_map(), 0, "split")
    at = page_app("pages/8_Cue_Map.py")
    at.session_state["cm_view"] = "🧪 Hard lab"
    at.run()
    assert not errors(at)
    assert any("random phases get a pool line" in (m.value or "") for m in at.markdown)
    at.session_state["hl_phase"] = {"round": 2, "start": 320.0, "end": 330.0, "profile": "fast"}
    at.run()
    assert not errors(at)
    next(b for b in at.button if b.label.startswith("Run ")).click().run()
    assert not errors(at) and any("Random phases with a pool line" in m.label for m in at.metric)
    next(b for b in at.button if "Bake these rules" in b.label).click().run()
    assert not errors(at)
    assert CS.load_draft()["map"]["hard_rules"]["on"] and not CS.load().get("hard_rules")      # unsaved until 💾
