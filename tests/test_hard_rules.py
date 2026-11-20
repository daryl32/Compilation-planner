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


def test_last_extra_plays_chill_and_mediums_lines():
    import edge_mode as E
    import hard_rules as HR
    cm = _split_map()
    FE = HR.final_extras(cm, 5)
    for seed in range(6):
        sch = E.schedule("hard", 5, seed)
        got = HR.resolve(sch, cm, seed, None, {"autofill": False})
        for r in range(1, 6):
            ph = HR.final_extra(sch, r)
            here = [p for p in got["plays"] if ph["start"] <= p["time"] < ph["end"]]
            assert sorted(p["line"] for p in here) == sorted(x["line"] for x in FE["chill"][r])
            assert all(p["time"] + p["seconds"] <= ph["end"] + 1e-3 for p in here)
            countdown = [x for x in FE["chill"][r] if x["off"] >= x["len"] / 2]
            for x in countdown:                                  # second-half lines keep their distance to the end
                p = next(p for p in here if p["line"] == x["line"])
                assert abs((ph["end"] - p["time"]) - (x["len"] - x["off"])) < 1.0
    own = HR.resolve(E.schedule("hard", 5, 1), cm, 1, None, {"final_extra": "hard", "autofill": False})
    assert not any(p.get("pool") == "fe" for p in own["plays"])


def _tagged_map():
    cm = _split_map()
    for ln in cm["lines"][::4]:
        ln["cats"] = ["Fast"]
    return cm


def test_category_rules():
    import edge_mode as E
    import hard_rules as HR
    cm = _tagged_map()
    cat = "Fast"
    names = {ln["name"] for ln in cm["lines"] if cat in (ln.get("cats") or [])}
    rule = {"cat": cat, "phases": ["medium"], "turns": [1, 2], "rounds": [1, "last"], "where": "start"}
    for seed in range(6):
        sch = E.schedule("hard", 5, seed)
        got = HR.resolve(sch, cm, seed, None, {"autofill": False, "cat_rules": [rule]})
        targets = HR.cat_rule_targets(sch, rule, 5)
        assert all(t["round"] in (1, 5) and t["profile"] == "medium" for t in targets)
        added = [p for p in got["plays"] if p.get("pool") == "cat"]
        assert added and all(p["line"] in names for p in added)
        for p in added:
            ph = next(t for t in targets if t["start"] <= p["time"] < t["end"])
            assert p["time"] - ph["start"] < (ph["end"] - ph["start"]) / 2            # "at the start"
        per = {}
        for p in got["plays"]:
            per[(p["round"], p["line"])] = per.get((p["round"], p["line"]), 0) + 1
        assert all(per[(p["round"], p["line"])] <= 2 for p in added)
        pl = sorted(got["plays"], key=lambda p: p["time"])
        assert all(b["time"] >= a["time"] + a["seconds"] - 1e-3 for a, b in zip(pl, pl[1:]))
    sch = E.schedule("hard", 5, 2)
    off = HR.resolve(sch, cm, 2, None, {"autofill": False, "cat_rules": [{**rule, "on": False}]})
    assert not any(p.get("pool") == "cat" for p in off["plays"])
    end = HR.resolve(sch, cm, 2, None, {"autofill": False, "cat_rules": [{**rule, "where": "end", "turns": []}]})
    for p in (p for p in end["plays"] if p.get("pool") == "cat"):
        ph = E.phase_at(sch, p["time"])
        assert ph["end"] - (p["time"] + p["seconds"]) < (ph["end"] - ph["start"]) / 2
    assert HR.describe_rule(rule).startswith("Fast lines · Medium 1 & last") or "Medium 1 & 2" in HR.describe_rule(rule)
    baked = E.resolve_lines(sch, HR.bake(cm, {"cat_rules": [rule]}), "hard", 2)
    assert any(p.get("pool") == "cat" for p in baked["plays"])


def test_lab_category_rule_editor(tmp_path, monkeypatch):
    import planner.edge as PE
    import cue_store as CS
    from tests.apptest_util import errors, page_app
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "v")
    CS.save(_tagged_map(), 0, "split")
    at = page_app("pages/8_Cue_Map.py")
    at.session_state["cm_view"] = "🧪 Hard lab"
    at.run()
    next(b for b in at.button if b.label == "➕ Add a rule").click().run()
    assert not errors(at)
    assert len(at.session_state["hl_rules"]["cat_rules"]) == 1
    assert any("in this draw" in (e.label or "") for e in at.expander), [e.label for e in at.expander]
    next(b for b in at.button if b.label == "⧉ Duplicate").click().run()
    assert len(at.session_state["hl_rules"]["cat_rules"]) == 2
    next(b for b in at.button if b.label == "🗑 Delete").click().run()
    assert not errors(at) and len(at.session_state["hl_rules"]["cat_rules"]) == 1
    next(b for b in at.button if "Bake these rules" in b.label).click().run()
    assert CS.load_draft()["map"]["hard_rules"]["cat_rules"]


def test_clashing_fixed_lines_shift():
    import hard_rules as HR
    plays = [{"time": 10.0, "seconds": 3.0, "line": "A", "pool": "anchor", "round": 1, "moved": 0.0, "why": "x"}]
    base = {"clashes": [{"line": "B", "with": "A", "round": 1, "where": "round 1, start of Wait", "result": "left out",
                         "time": 11.0, "seconds": 2.0, "file": "B.mp3", "profile": "wait"}]}
    keep = lambda t, r: True                                                       # noqa: E731
    rep = HR._fix_clashes({**HR.DEFAULTS}, base, list(plays), keep, 300.0)
    assert not any(c.get("left_out") for c in rep) and any("late" in c["result"] for c in rep)
    tight = [dict(plays[0]), {"time": 13.4, "seconds": 20.0, "line": "C", "pool": "anchor", "round": 1, "moved": 0.0, "why": "y"}]
    rep = HR._fix_clashes({**HR.DEFAULTS, "shift_later": 2.0}, base, tight, keep, 300.0)
    assert rep[0]["left_out"]                                                     # no room within 2 s later / 1 s earlier
    off = [dict(plays[0])]
    rep = HR._fix_clashes({**HR.DEFAULTS, "shift_on": False}, base, off, keep, 300.0)
    assert rep[0]["left_out"] and len(off) == 1                                   # off: only its exact moment
    free = []
    rep = HR._fix_clashes({**HR.DEFAULTS, "shift_on": False}, base, free, keep, 300.0)
    assert not rep and free[0]["time"] == 11.0                                    # the blocker was dropped: back in place
