"""🧩 Section types and 🍳 recipes (Stage 3 step 4): suggesting a track's sections from the song's structure with
story-map roles as hints, saving maps, recipe defaults / overrides / reset, and the Media Library sub-menus."""
from __future__ import annotations

from tests.support import DATA  # noqa: F401  (installs the test config first)


def _unit(start, end, pct, build=0.0, drop=False, breakdown=0.0, vocal=0.0, first=False, last=False):
    return {"start": start, "end": end, "pct": pct, "build": build, "drop": drop, "breakdown": breakdown,
            "vocal": vocal, "first": first, "last": last, "label": "?", "intensity": pct}


def _analysis():
    units = [
        _unit(0, 20, 0.1, first=True),                   # intro
        _unit(20, 40, 0.45),                             # warm-up (middling, before the first peak)
        _unit(40, 56, 0.55, build=0.8),                  # build-up
        _unit(56, 86, 0.95, drop=True),                  # drop
        _unit(86, 110, 0.2, breakdown=0.9),              # slow (breakdown)
        _unit(110, 126, 0.6, build=0.6),                 # build-up
        _unit(126, 156, 0.9, drop=True, vocal=0.1),      # drop
        _unit(156, 176, 0.8, vocal=0.1),                 # fast (intense, no vocals)
        _unit(176, 196, 0.5),                            # rhythmic
        _unit(196, 216, 0.3),                            # warm-down (late, after the peaks)
        _unit(216, 240, 0.1, last=True),                 # ending
    ]
    return {"units": units, "duration": 240.0, "drops": [56.0, 126.0], "builds": [], "bar_sec": 2.0}


def test_suggest_types_from_the_song():
    import section_types as ST
    regions = ST.suggest(_analysis())
    assert [r["type"] for r in regions] == ["intro", "warmup", "buildup", "drop", "slow", "buildup", "drop", "fast",
                                            "rhythmic", "warmdown", "ending"]
    assert regions[0]["start"] == 0.0 and regions[-1]["end"] == 240.0
    assert ST.type_at(regions, 60.0) == "drop" and ST.totals(regions)["drop"] == 60.0


def test_story_map_roles_steer_the_types():
    import section_types as ST
    story = [{"start": 176, "end": 196, "role": "action"},      # the rhythmic stretch is marked 💥 Action
             {"start": 86, "end": 110, "role": "slow"}]
    regions = ST.suggest(_analysis(), story)
    assert ST.type_at(regions, 180.0) == "action"
    assert ST.type_at(regions, 90.0) == "slow"
    # a role covering under HINT_SHARE of the stretch doesn't steer it
    regions = ST.suggest(_analysis(), [{"start": 176, "end": 180, "role": "action"}])
    assert ST.type_at(regions, 180.0) == "rhythmic"


def test_edit_and_save_a_section_map():
    import section_types as ST
    regions = ST.suggest(_analysis())
    regions = ST.set_type(regions, 1, "slow")
    assert regions[1]["type"] == "slow" and regions[1]["why"] == "set by you"
    n = len(regions)
    regions = ST.split(regions, 3, 70.0)
    assert len(regions) == n + 1 and regions[3]["end"] == 70.0 and regions[4]["start"] == 70.0
    regions = ST.merge_next(regions, 3)
    assert len(regions) == n and regions[3]["end"] == 86.0
    try:
        ST.save_map("trk_test", regions)
        assert ST.saved_map("trk_test")[3]["type"] == "drop" and "trk_test" in ST.saved_maps()
        ST.save_map("trk_test", None)
        assert ST.saved_map("trk_test") is None
    finally:
        ST.save_map("trk_test", None)


def test_recipes_defaults_overrides_and_reset():
    import recipes as RC
    import section_types as ST
    assert set(RC.load_all()) == set(ST.TYPE_IDS)
    assert RC.load("drop")["movement"]["hi"] == 1.0 and "penetration" in RC.load("drop")["tags"]
    try:
        r = RC.load("slow")
        r["beat_lock"] = 0.4
        r["tags"]["Blowjob"] = 0.9
        r["movement"] = {"lo": 0.6, "hi": 0.2, "trend": "rise"}        # swapped band is tidied
        RC.save("slow", r)
        got = RC.load("slow")
        assert RC.is_custom("slow") and got["beat_lock"] == 0.4 and got["tags"]["Blowjob"] == 0.9
        assert got["movement"] == {"lo": 0.2, "hi": 0.6, "trend": "rise"}
        assert not RC.is_custom("drop")
        RC.save("slow", RC.default("slow"))                             # saving the default drops the override
        assert not RC.is_custom("slow")
        RC.save("slow", r)
        RC.reset("slow")
        assert not RC.is_custom("slow") and RC.load("slow") == RC.default("slow")
    finally:
        RC.reset("slow")
    assert RC.unknown_tags({"tags": {"Blowjob": 1, "Typo": 1}, "avoid": ["x"]}, ["blowjob"]) == ["Typo", "x"]
    assert "wants" in RC.summary(RC.default("drop"))
    assert RC.default("intro")["roles"] == ["intro"] and RC.default("ending")["roles"] == ["outro"]
    assert RC.normalise({"roles": ["intro", "x"]}, "drop")["roles"] == ["intro", "x"]


def test_media_library_sub_menus():
    from tests.apptest_util import errors, page_app
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎵 Audio"
    at.session_state["aud_kind"] = "🍳 Recipes"
    at.run()
    assert not errors(at), errors(at)
    assert [r.key for r in at.radio if r.key in ("lib_kind", "aud_kind")] == ["lib_kind", "aud_kind"]
    assert len(at.expander) >= 10
    at.session_state["aud_kind"] = "🧩 Sections"
    at.run()
    assert not errors(at), errors(at)
    assert at.selectbox(key="sec_track").value
    assert [sb for sb in at.selectbox if str(sb.key).startswith("sec_role_")], "no section parts rendered"
    lines = [sl for sl in list(at.slider) + list(at.select_slider) if str(sl.key).startswith("sec_line_")]
    assert lines, "no draggable lines rendered"
    at.session_state["lib_kind"] = "🎬 Videos"
    at.session_state["vid_kind"] = "🎬 Videos"
    at.run()
    assert not errors(at), errors(at)
    assert [r.key for r in at.radio if r.key in ("lib_kind", "vid_kind")] == ["lib_kind", "vid_kind"]


def test_tuning_export_and_recipe_upload():
    import json
    import recipes as RC
    import section_types as ST
    import sections_view as SV
    from tests.fixtures import load_track
    track = load_track(DATA, "song_full")
    regions = ST.suggested_map(track)
    out = ST.tuning_export(track, regions, source="you")
    assert out["track_id"] == "song_full" and out["sections"] and out["suggested"] and out["stretches"]
    assert len(out["beats"]) == len(out["beat_intensity"]) + 1 and "analysis_error" not in out and "beats_error" not in out
    assert len(out["bar_driver"]) == len(out["bars"]) and set(out["bar_driver"]) <= {"drums", "bass", "vocals", "loudness"}
    assert len(out["beat_levels"]["vocals"]) == len(out["beat_intensity"]) and out["vocal_starts"]
    json.dumps(out)                                                   # plain JSON all the way down
    try:
        r = RC.default("slow")
        r["beat_lock"] = 0.1
        SV.st.session_state.clear()
        SV._upload_recipes(json.dumps({"slow": r, "nonsense": 1}).encode())
        assert RC.load("slow")["beat_lock"] == 0.1 and not RC.is_custom("drop")
        assert "Slow" in SV.st.session_state["rc_msg"]
        SV._upload_recipes(b"not json")
        assert "nothing was changed" in SV.st.session_state["rc_msg"] and RC.load("slow")["beat_lock"] == 0.1
    finally:
        RC.reset("slow")


def test_bar_level_pass_on_a_real_track():
    """AURA (10 Nov 2026): the app's own beat data + coarse suggestion, against Daryl's hand-edited map and his rules —
    Drop = the first bar only; Fast = the later half of a long loud stretch; Warm-down = the way down."""
    import json
    import section_types as ST
    from tests.support import REPO
    fx = json.loads((REPO / "tests" / "golden" / "sections_bar_pass.json").read_text())["aura"]
    R = lambda rows: [{"start": a, "end": b, "type": t} for a, b, t in rows]   # noqa: E731
    coarse, daryl = R(fx["coarse"]), R(fx["daryl"])
    out = ST.refine_with(coarse, fx["beats"], fx["intensity"], fx["drops"], fx["duration"])
    assert out[0]["start"] == 0.0 and out[-1]["end"] == fx["duration"]
    assert all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(out, out[1:]))
    drops = [r for r in out if r["type"] == "drop"]
    assert [round(r["start"]) for r in drops] == [24, 45] and all(2.0 < r["end"] - r["start"] < 3.2 for r in drops)
    wd = [r for r in out if r["type"] == "warmdown"]
    assert [round(r["start"]) for r in wd] == [41, 64, 99]              # starts on the beat the energy leaves the loud level
    assert [round(r["end"], 1) for r in wd] == [45.3, 69.3, 104.0]      # ends on the bar line where the next part begins
    seq = [r["type"] for r in out]
    assert seq[0] == "intro" and seq[1] == "buildup" and seq[-1] == "ending" and round(out[-1]["start"]) == 104
    assert ST.type_at(out, 30.0) == "action" and ST.type_at(out, 38.0) == "action"      # the first loud stretch: no Fast
    assert ST.type_at(out, 50.0) == "action" and ST.type_at(out, 60.0) == "fast"         # the peak comes last
    assert ST.type_at(out, 72.0) == "rhythmic"
    before, after = ST.agreement(coarse, daryl, fx["duration"]), ST.agreement(out, daryl, fx["duration"])
    assert before < 0.4 and after > 0.85
    # too little to go on → the coarse map comes back untouched
    assert ST.refine_with(coarse, fx["beats"][:6], fx["intensity"][:5], [], fx["duration"]) == coarse
    import numpy as np
    b, level = np.array(fx["beats"]), np.array(fx["intensity"])
    # no drop to anchor the bars on: the bar count is chosen from the energy changes — same result
    free = ST.refine_with(coarse, b, level, [], fx["duration"])
    assert ST.agreement(free, daryl, fx["duration"]) > 0.85 and sum(r["type"] == "drop" for r in free) == 2
    # a song with steady energy is left as the first pass had it
    assert ST.refine_with(coarse, b, 0.6 + 0.05 * np.sin(np.arange(len(level))), [], fx["duration"]) == coarse
    # at double the tempo a one-bar Drop would be 1.3 s — it takes two bars
    fast = ST.refine_with(coarse, b / 2, level, [d / 2 for d in fx["drops"]], fx["duration"] / 2)
    assert all(r["end"] - r["start"] >= ST.DROP_MIN_SEC for r in fast if r["type"] == "drop")
    # a hard stop (loud straight to quiet) gets no Warm-down
    hard = level.copy()
    k = int(np.argmin(abs(b - 64.0)))
    hard[k:k + 40], hard[k - 8:k] = 0.2, 0.8
    stops = ST.refine_with(coarse, b, hard, fx["drops"], fx["duration"])
    assert not [r for r in stops if r["type"] == "warmdown" and 60 < r["start"] < 70]
    for res in (free, fast, stops):
        assert res[0]["start"] == 0.0 and all(abs(p["end"] - q["start"]) < 1e-6 for p, q in zip(res, res[1:]))


def test_intro_ends_where_the_music_comes_in():
    """Body and Disconnect (10 Nov 2026) open near-silent: the suggested Intro ran to the first build-up (43 s, 37 s);
    Daryl: it should end where the voice / the music comes in. AURA opens with sound playing — its Intro is unchanged."""
    import json
    import section_types as ST
    from tests.support import REPO
    fx = json.loads((REPO / "tests" / "golden" / "sections_bar_pass.json").read_text())
    R = lambda rows: [{"start": a, "end": b, "type": t} for a, b, t in rows]   # noqa: E731
    for key in ("body", "disconnect"):
        d = fx[key]
        out = ST.refine_with(R(d["coarse"]), d["beats"], d["intensity"], d["drops"], d["duration"])
        assert out[0]["type"] == "intro" and abs(out[0]["end"] - d["intro_end"]) < 0.6, (key, out[0])
        assert out[1]["type"] == "warmup" and out[2]["type"] == "buildup"
        assert ST.agreement(out, R(d["daryl"]), d["duration"]) > 0.88, key
        assert all(abs(p["end"] - q["start"]) < 1e-6 for p, q in zip(out, out[1:]))
    a = fx["aura"]
    out = ST.refine_with(R(a["coarse"]), a["beats"], a["intensity"], a["drops"], a["duration"])
    assert out[0]["type"] == "intro" and out[0]["end"] > 13 and out[1]["type"] == "buildup"
