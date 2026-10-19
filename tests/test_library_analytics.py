"""📈 Media Library → Analytics → 📚 Library: library_analytics.py on hand-made data, and the
page's four tabs headless."""

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)


def _tc(s):
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def _scene(sid, a, b, motion=1.0, roles=(), excluded=False, reviewed=False, **kw):
    return {"scene_id": sid, "start_tc": _tc(a), "end_tc": _tc(b), "motion_intensity": motion,
            "roles": list(roles), "excluded": excluded, "reviewed": reviewed, **kw}


ROLES_OF = lambda s: s.get("roles") or []  # noqa: E731

CATS = {
    # split, all short, two roles, one excluded clip
    "v1": {"scenes": [_scene(0, 0, 10, 0.5, ("slow",), reviewed=True, split_from=0),
                      _scene(1, 10, 20, 3.0, ("action",), reviewed=True, split_from=0),
                      _scene(2, 20, 80, 9.0, ("action",), excluded=True)]},
    # untouched, one long clip, no roles
    "v2": {"scenes": [_scene(0, 0, 100, 2.0), _scene(1, 100, 110, 0.2), _scene(2, 110, 120, 5.0)]},
    # long clip, but the library range keeps only 20 s of it
    "v3": {"scenes": [_scene(0, 0, 200, 1.0)]},
}
RANGES = {"v3": (50.0, 70.0)}
DETAILS = {"v1": {"actors": ["Ana"]}, "v2": {"actors": ["Ben", "Ana"]}}


def test_progress():
    import library_analytics as LA
    p = LA.progress(CATS, RANGES, DETAILS, {"v1": 0}, ROLES_OF, usage={"v2": 3})
    rows = {r["video_id"]: r for r in p["rows"]}
    v1, v2, v3 = rows["v1"], rows["v2"], rows["v3"]
    assert v1["clips"] == 2 and v1["minutes"] == round(20 / 60, 1)          # the excluded clip doesn't count
    assert v1["split"] == "done" and v1["reviewed_pct"] == 100 and v1["roles_pct"] == 100
    assert v1["role_secs"] == {"slow": 10.0, "action": 10.0}
    assert v2["split"] == "to do" and v2["long_left"] == 1 and v2["roles"] == "to do"
    assert v3["split"] == "not needed" and v3["clips"] == 1 and v3["minutes"] == round(20 / 60, 1)
    t = p["totals"]
    assert (t["split_done"], t["roles_started"], t["actors"], t["masters"], t["ranges"]) == (2, 1, 2, 1, 1)
    assert [r["video_id"] for r in p["next_split"]] == ["v2"]
    assert [r["video_id"] for r in p["next_roles"]] == ["v2", "v3"]           # most footage × recent use first


def test_actor_stats_and_insights():
    import library_analytics as LA
    stats = LA.actor_stats(CATS, RANGES, DETAILS, ROLES_OF, usage={"v2": 3}, ratings={"v1": [5, 4], "v2": [3]})
    by = {a["actor"]: a for a in stats}
    assert set(by) == {"Ana", "Ben"}
    assert by["Ana"]["videos"] == ["v1", "v2"] and by["Ben"]["videos"] == ["v2"]       # v2 counts for both
    assert by["Ana"]["clips"] == 5 and by["Ana"]["minutes"] == round(140 / 60, 1)
    assert by["Ana"]["roles"] == {"action": 10.0, "slow": 10.0} and by["Ben"]["roles"] == {}
    assert by["Ana"]["uses"] == 3 and by["Ana"]["rating"] == 4.0 and by["Ana"]["rated"] == 3
    for a in stats:
        assert abs(a["calm_pct"] + a["medium_pct"] + a["busy_pct"] - 100) <= 1
    assert stats[0]["actor"] == "Ana"                                          # most footage first
    assert any("Ana" in line for line in LA.actor_insights(stats))


def test_role_balance_and_candidates():
    import library_analytics as LA
    p = LA.progress(CATS, RANGES, DETAILS, {}, ROLES_OF)
    needs = {"t1": {"action": 40.0, "intro": 10.0}, "t2": {"action": 20.0}, "t3": {}}
    rows = {r["role"]: r for r in LA.role_balance(p["rows"], needs, DETAILS)}
    assert rows["intro"]["status"] == "none" and rows["intro"]["tracks"] == 1
    assert rows["action"]["have_sec"] == 10.0 and rows["action"]["need_sec"] == 30.0
    assert rows["action"]["tracks_worth"] == round(10 / 30, 2) and rows["action"]["status"] == "short"
    assert rows["action"]["by_actor"] == {"Ana": 10.0}
    assert rows["slow"]["status"] == "not needed"
    assert list(rows)[0] == "intro"                                           # the worst first
    busy = LA.role_candidates(CATS, RANGES, p["rows"], "action")
    assert busy and busy[0]["video_id"] == "v2" and busy[0]["seconds"] == 10.0  # its busiest clip (5.0)
    # the same rule as the Reviewer's 💡 hints: in a 3-clip video the calmest ranks 33%, not in the calmest fifth
    assert LA.role_candidates(CATS, RANGES, p["rows"], "slow") == []
    many = dict(CATS, v4={"scenes": [_scene(i, 10 * i, 10 * i + 10, float(i)) for i in range(10)]})
    calm = LA.role_candidates(many, RANGES, LA.progress(many, RANGES, DETAILS, {}, ROLES_OF)["rows"], "slow")
    assert calm == [{"video_id": "v4", "seconds": 20.0, "clips": 2}]
    assert LA.role_candidates(CATS, RANGES, p["rows"], "intro") == []


def test_length_profile_and_mixed_motion():
    import library_analytics as LA
    prof = LA.length_profile(CATS, RANGES, [0.5, 1.5, 1.5, 2.5, 10.0])
    by = {b["label"]: b for b in prof["bins"]}
    assert prof["clips"] == 6 and prof["blocks"] == 5
    assert by["60s+"]["clips"] == 1 and by["8–15s"]["clips"] == 4                # v1's two, v2's two short ones
    assert by["1–2s"]["blocks_pct"] == 40.0
    assert abs(sum(b["footage_pct"] for b in prof["bins"]) - 100) < 0.5
    assert prof["long_share"] == round(100 * 100 / 160, 1)
    # a long clip that swings from calm to busy outranks a steady one
    fps = 10.0
    swing = np.r_[np.full(100, 0.1), np.full(100, 5.0)]
    steady = np.full(200, 1.0)
    curves = {"v2": {0: {"fps": fps, "values": np.r_[swing, swing, swing, swing, swing][:1000]}},
              "v3": {0: {"fps": fps, "values": np.r_[steady] .repeat(10)}}}
    mixed = LA.mixed_motion_clips(CATS, RANGES, lambda v: curves.get(v, {}))
    assert [m["video_id"] for m in mixed] == ["v2", "v3"]
    assert mixed[0]["swing"] > 1 and mixed[1]["swing"] == 0


def test_media_library_analytics_tabs():
    from tests.apptest_util import errors, page_app
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "📈 Analytics"
    at.run()
    assert not errors(at), errors(at)
    assert [t.label for t in at.tabs] == ["📋 Progress", "👤 Actors", "🎭 Roles", "📏 Clip lengths"]
    assert [m.label for m in at.metric][:4] == ["✂️ Split pass", "🎭 Role marking", "👤 Actors", "👀 Reviewed"]
    at.radio(key="ana_kind").set_value("⭐ Render ratings").run()
    assert not errors(at), errors(at)
