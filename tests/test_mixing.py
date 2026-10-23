"""🎚️ Mixing: keys, planning (core/mixing.py), rendering (mixer.py), saving
(mixes.py → audio analysis) and the Media Library's Mix screen.

The audio tests use two small synthetic songs (tests/mix_audio.py) in their
own temporary audio folder, so the shared synthetic library — and every golden
snapshot built from it — is untouched."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from core import keys as K
from core import mixing as MX


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def test_camelot_codes():
    assert K.camelot(0, "major") == "8B"      # C
    assert K.camelot(9, "minor") == "8A"      # Am
    assert K.camelot(7, "major") == "9B"      # G
    assert K.camelot(4, "major") == "12B"     # E
    assert K.camelot(1, "minor") == "12A"     # C#m
    assert K.camelot(11, "major") == "1B"     # B


def test_estimate_key_from_profiles():
    # the major / minor profiles themselves, rotated, must come back as that key
    for pc in range(12):
        assert K.estimate_key(np.roll(K._MAJOR, pc))["pc"] == pc
        k = K.estimate_key(np.roll(K._MINOR, pc))
        assert (k["pc"], k["mode"]) == (pc, "minor")
    assert K.estimate_key(np.zeros(12)) is None


def test_compatibility_and_shift():
    am, c, g, d = K.key_from(9, "minor"), K.key_from(0, "major"), K.key_from(7, "major"), K.key_from(2, "major")
    assert K.compatibility(c, c)[0] == 1.0
    assert K.compatibility(am, c)[0] >= 0.85          # relative
    assert K.compatibility(c, g)[0] >= 0.85           # next door
    assert K.compatibility(c, K.key_from(6, "major"))[0] < 0.5   # F# clashes with C
    assert K.best_shift(c, g) == 0                    # already fits
    s = K.best_shift(c, K.key_from(6, "major"))       # F# → shift to fit C
    assert s != 0 and abs(s) <= 3
    assert K.compatibility(c, K.shifted(K.key_from(6, "major"), s))[0] >= 0.85


# ---------------------------------------------------------------------------
# Planning (numbers only)
# ---------------------------------------------------------------------------

def _grid(n_bars=40, bar=2.0, labels="AABBAABBCC"):
    bars = np.arange(n_bars) * bar
    secs = []
    per = n_bars // len(labels)
    for k, lab in enumerate(labels):
        secs.append({"start": k * per * bar, "end": (k + 1) * per * bar, "label": lab, "energy": 0.5})
    return {"duration": n_bars * bar, "bpm": 120.0, "beats": np.arange(n_bars * 4) * bar / 4, "bars": bars,
            "bar_len": bar, "beats_per_bar": 4, "levels": {i: (16 if i % 16 == 0 else 8 if i % 8 == 0 else 4)
                                                          for i in range(0, n_bars, 4)}, "sections": secs}


def _sim_for(pattern):
    """Bar similarity from a list of bar 'kinds' (same kind = identical bars)."""
    kinds = sorted(set(pattern))
    rng = np.random.default_rng(0)
    base = {k: rng.normal(size=16) for k in kinds}
    feats = np.array([base[k] + rng.normal(scale=0.05, size=16) for k in pattern])
    return MX.bar_similarity(feats)


PATTERN = [f"v{i % 4}" for i in range(8)] + [f"c{i % 4}" for i in range(8)] + \
          [f"v{i % 4}" for i in range(8)] + [f"c{i % 4}" for i in range(8)] + [f"o{i}" for i in range(8)]


def test_grid_from_track_and_fallbacks():
    from tests.fixtures import make_track
    t = make_track(1, 96.0, 124.0)
    g = MX.grid(t)
    assert len(g["bars"]) > 40 and abs(g["bar_len"] - 4 * 60 / 124) < 0.05
    assert g["sections"] and g["sections"][0]["start"] == 0.0
    t2 = make_track(2, 74.0, 98.0, analysis=False)          # no extra analysis: every 4th beat
    g2 = MX.grid(t2)
    assert len(g2["bars"]) == len(t2["beat_times"][::4]) and g2["sections"][0]["end"] == 74.0


def test_tempo_factor_double_and_half_time():
    assert MX.tempo_factor(120, 126)[0] == pytest.approx(1.05)
    speed, eff = MX.tempo_factor(70, 140)       # double time, no stretch
    assert speed == pytest.approx(1.0) and eff == pytest.approx(140)
    assert MX.tempo_factor(120, None) == (1.0, 120.0)
    assert MX.stretch_note(1.05) is None and MX.stretch_note(1.12)


def test_layout_lines_up_bars_and_beats():
    g = _grid()
    rec = {"pieces": [{"track": "x", "start": 0.0, "end": 16.0}, {"track": "x", "start": 8.0, "end": 80.0}],
           "bpm": None}
    lay = MX.layout(rec, {"x": g})
    assert lay["duration"] == pytest.approx(16 + 72)
    assert lay["pieces"][1]["out_start"] == pytest.approx(16.0)
    # the new track's bar lines: every 2 s, no duplicates at the join
    assert np.allclose(np.diff(lay["bars"]), 2.0)
    assert len(lay["joins"]) == 1 and lay["joins"][0]["same_track"]
    # stretched to 132 BPM: 10 % faster, so 10 % shorter
    lay2 = MX.layout({**rec, "bpm": 132.0}, {"x": g})
    assert lay2["duration"] == pytest.approx(88 / 1.1)


def test_plan_length_extends_and_shortens_on_matching_bars():
    g = _grid()
    S = _sim_for(PATTERN)
    # whole-phrase targets are hit exactly; in-between ones land on the nearest whole phrase
    for target, tol in ((112.0, 0.0), (144.0, 0.0), (64.0, 0.0), (110.0, 2 * g["bar_len"])):
        plans = MX.plan_length(g, S, target)
        assert plans, target
        best = plans[0]
        assert abs(best["error"]) <= tol + 1e-6, (target, best["error"])
        assert best["score"] > 0.9                   # jumps between identical bars
        total = sum(p["end"] - p["start"] for p in best["pieces"])
        assert total == pytest.approx(best["length"])
        for jp in best["jumps"]:                      # whole phrases (multiples of 4 bars)
            assert jp["bars"] % 4 == 0
        assert best["pieces"][0]["start"] == 0.0 and best["pieces"][-1]["end"] == g["duration"]
    # already the right length: one piece, the whole track
    same = MX.plan_length(g, S, g["duration"])[0]
    assert same["jumps"] == [] and len(same["pieces"]) == 1


def test_plan_length_exact_runs_long_for_a_fade():
    g = _grid()
    S = _sim_for(PATTERN)
    for p in MX.plan_length(g, S, 107.0, exact=True):
        assert p["length"] >= 107.0 - g["bar_len"] * 0.5


def test_pieces_from_sections():
    g = _grid()
    pieces = MX.pieces_from_sections(g, [1, 1, 2, 1, 0, 0, 1, 1, 1, 1])
    # sections playing straight on stay one piece; the repeat of the third starts a new piece; 0s are cut
    assert pieces[0] == {"track": None, "start": 0.0, "end": 24.0, "label": "AAB"}
    assert pieces[1] == {"track": None, "start": 16.0, "end": 32.0, "label": "BB"}
    assert pieces[2]["start"] == 48.0 and len(pieces) == 3
    assert sum(p["end"] - p["start"] for p in pieces) == pytest.approx(80 + 8 - 16)
    assert MX.pieces_from_sections(g, [0] * 10) == []


def test_join_quality_and_partner_score():
    g = _grid()
    S = _sim_for(PATTERN)
    assert MX.join_quality(S, g, 48.0, 16.0) > 0.9          # chorus 2 start ≈ chorus 1 start
    assert MX.join_quality(S, g, 64.0, 32.0) < 0.7          # outro ≠ verse 2
    assert MX.join_quality(S, g, 64.0, 66.0) is not None
    a = {"track_id": "a", "bpm": 120, "key": K.key_from(0, "major"), "energy": 0.5}
    good = MX.partner_score(a, {"track_id": "b", "bpm": 122, "key": K.key_from(7, "major"), "energy": 0.5})
    bad = MX.partner_score(a, {"track_id": "c", "bpm": 90, "key": K.key_from(6, "major"), "energy": 0.9})
    assert good["score"] > bad["score"]


# ---------------------------------------------------------------------------
# Real audio: render, save, analyse, and the page
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mixlib(tmp_path_factory):
    """Two synthetic songs in their own audio folder; config / module paths point there."""
    import config
    import library_common
    import mixes
    import mix_ui
    import render_preview
    from tests import mix_audio

    root = tmp_path_factory.mktemp("mixlib")
    audio_dir, files, cat = root / "audio_catalogue", root / "files", root / "catalogue"
    for d in (audio_dir, files, cat):
        d.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "AUDIO_DIR", audio_dir)
    mp.setattr(mixes, "AUDIO_DIR", audio_dir)
    mp.setattr(mixes, "CATALOGUE_DIR", cat)
    mp.setattr(mix_ui, "AUDIO_DIR", audio_dir)
    mp.setattr(library_common, "mark_pending", lambda stem: None)
    real = render_preview.colab_to_local
    # Drive paths ("…/MyDrive/Audio-Library/x.m4a") land in this test's own folder
    mp.setattr(render_preview, "colab_to_local",
               lambda p: str(root / "drive" / p.split("MyDrive/", 1)[1]) if "MyDrive/" in p else real(p))
    tracks = mix_audio.install(audio_dir, files)
    yield {"root": root, "audio_dir": audio_dir, "tracks": tracks}
    mp.undo()


def test_key_found_by_the_analysis(mixlib):
    t = mixlib["tracks"]
    assert t["mix_song_c"]["key"]["camelot"] == "8B"        # C major
    assert t["mix_song_a"]["key"]["camelot"] == "11B"       # A major


def test_render_extend_keeps_the_audio_seamless(mixlib):
    import mixer
    tr = mixlib["tracks"]["mix_song_c"]
    g = MX.grid(tr)
    S = MX.bar_similarity(mixer.bar_features(tr, g))
    plan = MX.plan_length(g, S, g["duration"] + 30)[0]
    assert plan["jumps"] and plan["score"] > 0.85
    rec = {"pieces": [{**p, "track": "mix_song_c"} for p in plan["pieces"]], "bpm": None, "xfade_same": 0.25}
    out = mixlib["root"] / "ext.m4a"
    info = mixer.render(rec, {"mix_song_c": tr}, out)
    assert info["duration"] == pytest.approx(plan["length"], abs=0.01)
    y, src = mixer.decode(out), mixer.decode(Path(tr["source_path"]))
    assert abs(len(y) / mixer.SR - info["duration"]) < 0.08
    # away from the joins, every piece is the source audio, sample for sample (allowing for AAC)
    for p in info["layout"]["pieces"]:
        a, b = int((p["out_start"] + 0.5) * mixer.SR), int((p["out_start"] + 1.5) * mixer.SR)
        s0 = int((p["start"] + 0.5) * mixer.SR)
        assert np.corrcoef(y[a:b, 0], src[s0:s0 + (b - a), 0])[0, 1] > 0.98
    # the new beat grid carries on at the source's beat spacing
    assert np.median(np.diff(info["beats"])) == pytest.approx(np.median(np.diff(tr["beat_times"])), abs=0.005)


def test_render_mashup_stretches_and_shifts(mixlib):
    import mixer
    t = mixlib["tracks"]
    gc, ga = MX.grid(t["mix_song_c"]), MX.grid(t["mix_song_a"])
    shift = K.best_shift(t["mix_song_c"]["key"], t["mix_song_a"]["key"])
    rec = {"pieces": [{"track": "mix_song_c", "start": 0.0, "end": MX.snap_to_bar(gc, 24)},
                      {"track": "mix_song_a", "start": MX.snap_to_bar(ga, 7.6), "end": MX.snap_to_bar(ga, 22.9)}],
           "bpm": 120.0, "shift": {"mix_song_a": shift}, "xfade_other": 4, "fade_out": 1.0, "level_match": True}
    info = mixer.render(rec, {k: t[k] for k in ("mix_song_c", "mix_song_a")}, mixlib["root"] / "mash.m4a")
    lay = info["layout"]
    assert lay["pieces"][1]["speed"] == pytest.approx(120 / 126, rel=1e-3)
    assert info["duration"] == pytest.approx(lay["duration"], abs=0.01)
    assert info["bpm"] == 120.0 and np.median(np.diff(info["beats"])) == pytest.approx(0.5, abs=0.02)
    # the shifted part now sounds in a key that fits the first part's
    import audio_analysis as AA
    y = mixer.decode(mixlib["root"] / "mash.m4a")
    seg = y[int((lay["pieces"][1]["out_start"] + 3) * mixer.SR):int((lay["pieces"][1]["out_end"] - 3) * mixer.SR)]
    tmp = mixlib["root"] / "part2.wav"
    mixer.encode(seg, tmp.with_suffix(".m4a"))
    k2 = AA.key_for_file(tmp.with_suffix(".m4a"))
    assert K.compatibility(t["mix_song_c"]["key"], k2)[0] >= 0.85


def test_publish_saves_recipe_and_analyses_with_the_mix_grid(mixlib):
    import audio_analysis as AA
    import mixes as MS
    t = mixlib["tracks"]
    g = MX.grid(t["mix_song_c"])
    rec = {"kind": "extend", "name": "Song C (extended)", "bpm": None, "xfade_same": 0.25,
           "pieces": [{"track": "mix_song_c", "start": 0.0, "end": g["duration"]},
                      {"track": "mix_song_c", "start": MX.snap_to_bar(g, 8), "end": g["duration"]}]}
    saved, notes = MS.publish({**rec, "id": MS.new_id()}, {"mix_song_c": t["mix_song_c"]}, token=None)
    assert any("Google Drive isn't connected" in n for n in notes)
    out = saved["output"]
    assert out["track_id"] == "Song C (extended)" and not out["uploaded"]
    assert MS.get(saved["id"])["output"]["track_id"] == out["track_id"]
    assert MS.name_taken("Song C (extended)") is False      # no catalogue yet — analysis still queued
    job = next(j for j in json.loads((mixlib["audio_dir"] / ".analysis" / "queue.json").read_text())
               if j["track_id"] == out["track_id"])
    assert job["mix"]["made_from"] == ["mix_song_c"] and len(job["beat_hint"]["beat_times"]) > 100
    AA.cancel_queued()                          # the test runs the job itself, not a worker
    AA._process(job, lambda m: None)
    made = json.loads((mixlib["audio_dir"] / "Song C (extended).json").read_text())
    assert made["analysis"]["methods"]["beats"] == "mix" and made["analysis"]["methods"]["downbeats"] == "mix"
    assert made["mix"]["kind"] == "extend" and made["key"]["camelot"] == "8B"
    assert made["duration_sec"] == pytest.approx(out["duration"], abs=0.1)
    assert MS.name_taken("Song C (extended)") and not MS.name_taken("Song C (extended)", saved["id"])
    assert MS.made_from("Song C (extended)")["id"] == saved["id"]
    MS.delete(saved["id"])
    assert MS.get(saved["id"]) is None


def test_mix_page_extend_auto_preview(mixlib):
    from tests.apptest_util import page_app, errors
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎚️ Mix"
    at.session_state["mxe_track"] = "mix_song_c"
    at.run()
    assert not errors(at), errors(at)
    opts = at.radio(key="mxe_pick").options
    assert len(opts) >= 1 and "Option 1" in opts[0]
    at.button(key="mxe_preview").click().run()
    assert not errors(at), errors(at)
    assert at.get("audio"), "no preview player"
    # by sections: play the 2nd section twice
    at.radio(key="mxe_method").set_value("✋ By sections").run()
    at.number_input(key="mxe_cnt_mix_song_c_1").set_value(2).run()
    assert not errors(at), errors(at)
    assert any("piece(s)" in c.value for c in at.caption)


def test_mix_page_mashup_build(mixlib):
    from tests.apptest_util import page_app, errors
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎚️ Mix"
    at.session_state["mx_mode"] = "🧩 Mashup"
    at.session_state["mxm_add_track"] = "mix_song_c"
    at.run()
    assert not errors(at), errors(at)
    at.button(key="mxm_add").click().run()
    at.selectbox(key="mxm_add_track").set_value("mix_song_a").run()
    at.button(key="mxm_add").click().run()
    assert not errors(at), errors(at)
    assert len(at.session_state["mxm_parts"]) == 2
    assert any("tempo" in c.value and "after part 1" in c.value for c in at.caption)
    at.button(key="mxm_fitkeys").click().run()
    assert at.session_state["mxm_shift_mix_song_a"] != 0
    # 🔀 the join's own crossfade: a 4-bar blend with the next song coming in early
    uid2 = at.session_state["mxm_parts"][1]["uid"]
    at.selectbox(key=f"mxm_xfl_{uid2}").set_value("4 bars").run()
    at.selectbox(key=f"mxm_xfw_{uid2}").set_value("before").run()
    assert not errors(at), errors(at)
    assert at.session_state["mxm_parts"][1]["xfade"] == {"beats": 16.0, "where": "before", "shape": "smooth"}
    at.button(key="mxm_preview").click().run()
    assert not errors(at), errors(at)
    assert at.get("audio")


def test_mix_page_saved_mixes_reopen(mixlib):
    """📂 Saved mixes: a saved extend (Auto) and mashup reopen in their editors as saved."""
    import mixes as MS
    from tests.apptest_util import page_app, errors
    import mixer
    t = mixlib["tracks"]
    g = MX.grid(t["mix_song_c"])
    S = MX.bar_similarity(mixer.bar_features(t["mix_song_c"], g))
    plans = MX.plan_length(g, S, g["duration"] + 30)
    pick = plans[min(1, len(plans) - 1)]
    ext = MS.save({"kind": "extend", "name": "Reopen me", "bpm": None, "xfade_same": 0.5, "fade_out": 3.0,
                   "pieces": [{**p, "track": "mix_song_c"} for p in pick["pieces"]],
                   "settings": {"method": "auto", "track": "mix_song_c", "target": g["duration"] + 30,
                                "exact": False, "jumps": [{k: j[k] for k in ("at", "to", "repeat")}
                                                          for j in pick["jumps"]]}})
    mash = MS.save({"kind": "mashup", "name": "Mash me", "bpm": 123.0, "shift": {"mix_song_a": 3},
                    "xfade_other": 8.0, "fade_in": 1.0, "fade_out": 2.0, "level_match": True,
                    "pieces": [{"track": "mix_song_c", "start": 0.0, "end": 16.0, "gain_db": 0.0},
                               {"track": "mix_song_a", "start": 7.6, "end": 15.2, "gain_db": -2.0}]})
    try:
        at = page_app("pages/3_Media_Library.py", timeout=300)
        at.session_state["lib_kind"] = "🎚️ Mix"
        at.session_state["mx_mode"] = "📂 Saved mixes"
        at.run()
        assert not errors(at), errors(at)
        at.button(key=f"mxs_open_{ext['id']}").click().run()
        assert not errors(at), errors(at)
        assert at.session_state["mx_mode"] == "📏 Extend / shorten"
        assert at.session_state["mxe_pick"] == plans.index(pick)
        assert at.session_state["mxe_xf"] == "Half a beat" and at.text_input(key="mxe_name").value == "Reopen me"
        assert any(b.label == "💾 Save as a new track" for b in at.button)   # not made as a track yet
        at.radio(key="mx_mode").set_value("📂 Saved mixes").run()
        at.button(key=f"mxs_open_{mash['id']}").click().run()
        assert not errors(at), errors(at)
        assert at.session_state["mx_mode"] == "🧩 Mashup"
        assert len(at.session_state["mxm_parts"]) == 2 and at.session_state["mxm_bpm"] == 123.0
        assert at.session_state["mxm_shift_mix_song_a"] == 3 and at.session_state["mxm_xf"] == "2 bars"
    finally:
        MS.delete(ext["id"])
        MS.delete(mash["id"])


# ---------------------------------------------------------------------------
# 🎤 Stems, vocal layers and 💡 suggestions
# ---------------------------------------------------------------------------

def test_layout_places_layers_in_bars_of_their_part():
    g = _grid()
    rec = {"bpm": 132.0,
           "pieces": [{"track": "x", "start": 0.0, "end": 16.0, "uid": "p1"},
                      {"track": "x", "start": 16.0, "end": 48.0, "uid": "p2"}],
           "layers": [{"track": "x", "part": "vocals", "start": 32.0, "end": 64.0, "on": "p2", "offset_bars": 2},
                      {"track": "x", "part": "vocals", "start": 0.0, "end": 8.0, "on": "gone"}]}
    lay = MX.layout(rec, {"x": g})
    assert len(lay["layers"]) == 1                       # a layer on a part that's gone is dropped
    ly = lay["layers"][0]
    speed = 132.0 / 120.0
    assert ly["out_start"] == pytest.approx(16.0 / speed + 2 * 2.0 / speed)   # 2 bars into part 2
    assert ly["out_end"] <= lay["duration"] + 1e-9       # never runs past the end of the mix


def test_best_section_and_suggest_pairs(mixlib):
    t = mixlib["tracks"]["mix_song_c"]
    loud = MX.best_section(t, "energy")
    assert loud["end"] - loud["start"] >= 4 * MX.grid(t)["bar_len"] - 0.1
    infos = [{"track_id": "a", "bpm": 120, "key": K.key_from(0, "major"), "energy": 0.5, "stems": True},
             {"track_id": "b", "bpm": 121, "key": K.key_from(7, "major"), "energy": 0.5, "stems": True},
             {"track_id": "c", "bpm": 90, "key": K.key_from(6, "major"), "energy": 0.9, "stems": False}]
    pairs = MX.suggest_pairs(infos)
    assert {pairs[0]["a"], pairs[0]["b"]} == {"a", "b"} and len(pairs) == 3
    swaps = MX.suggest_pairs(infos, need_stems=True)
    assert all({s_["a"], s_["b"]} == {"a", "b"} for s_ in swaps) and len(swaps) == 2


def test_render_vocals_over_another_tracks_instrumental(mixlib):
    import mixer
    t = mixlib["tracks"]
    gc = MX.grid(t["mix_song_c"])
    rec = {"bpm": 120.0, "level_match": False,
           "pieces": [{"track": "mix_song_c", "start": 0.0, "end": MX.snap_to_bar(gc, 16.0), "part": "instrumental",
                       "uid": "p1"}],
           "layers": [{"track": "mix_song_a", "part": "vocals", "start": 0.0, "end": 8.0, "on": "p1",
                       "offset_bars": 1, "gain_db": 0.0}]}
    assert mixer.parts_needed(rec) == {"mix_song_c": {"instrumental"}, "mix_song_a": {"vocals"}}
    info = mixer.render(rec, {k: t[k] for k in ("mix_song_c", "mix_song_a")}, mixlib["root"] / "voc.m4a")
    ly = info["layout"]["layers"][0]
    assert ly["out_start"] == pytest.approx(gc["bar_len"], abs=0.01)
    y = mixer.decode(mixlib["root"] / "voc.m4a")
    inst = mixer.decode(mixer.local_path(t["mix_song_c"], "instrumental"))
    voc = mixer.decode(mixer.local_path(t["mix_song_a"], "vocals"), speed=120.0 / 126.0)
    # before the vocals come in: just the instrumental; once they're in: the mix minus the instrumental
    # is the (stretched) vocal line
    a = int(0.2 * mixer.SR)
    assert np.corrcoef(y[a:a + mixer.SR // 2, 0], inst[a:a + mixer.SR // 2, 0])[0, 1] > 0.98
    s0 = int((ly["out_start"] + 0.3) * mixer.SR)
    rest = y[s0:s0 + mixer.SR, 0] - inst[s0:s0 + mixer.SR, 0]
    v0 = int(0.3 * mixer.SR)
    assert np.corrcoef(rest, voc[v0:v0 + mixer.SR, 0])[0, 1] > 0.9


def test_mix_page_suggestions_start_a_vocal_swap(mixlib):
    from tests.apptest_util import page_app, errors
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎚️ Mix"
    at.session_state["mx_mode"] = "💡 Suggestions"
    at.run()
    assert not errors(at), errors(at)
    go = [b for b in at.button if (b.key or "").startswith("mx_sug_go_v_")]
    assert go, "no vocal swap suggestions"
    go[0].click().run()
    assert not errors(at), errors(at)
    assert at.session_state["mx_mode"] == "🧩 Mashup"
    parts, layers = at.session_state["mxm_parts"], at.session_state["mxm_layers"]
    assert len(parts) == 1 and parts[0]["part"] == "instrumental"
    assert len(layers) == 1 and layers[0]["on"] == parts[0]["uid"] and layers[0]["track"] != parts[0]["track"]
    assert any("🎤" in c.value and "vocals" in c.value for c in at.caption)
    at.button(key="mxm_preview").click().run()
    assert not errors(at), errors(at)
    assert at.get("audio")


def test_local_analysis_does_only_what_each_track_is_missing(mixlib, monkeypatch, tmp_path):
    """run_audio_analysis.bat → analyse_audio_local.py: new tracks get everything; analysed
    ones only their missing parts (stems, key…) from one Demucs pass."""
    import numpy as np_
    import analyse_audio_local as AL
    import audio_analysis as AA
    lib = tmp_path / "Audio-Library"
    lib.mkdir()
    src = Path(mixlib["tracks"]["mix_song_c"]["source_path"])
    (lib / "mix_song_c.m4a").write_bytes(src.read_bytes())
    (lib / "brand_new.m4a").write_bytes(src.read_bytes())
    audio_dir = mixlib["audio_dir"]
    monkeypatch.setattr(AL, "AUDIO_DIR", audio_dir)
    monkeypatch.setattr(AL, "drive_root", lambda: tmp_path)
    monkeypatch.setattr(AL, "AA", AA, raising=False)
    monkeypatch.setattr(AL, "LOCAL_TEMP_DIR", tmp_path / "tmp")
    calls = []

    def fake_sep(path):
        calls.append(Path(path).name)
        y = AA.decode_audio(path, sr=44100, channels=2)
        return {k: (g * y).astype(np_.float32) for k, g in (("vocals", 0.3), ("drums", 0.3), ("bass", 0.2),
                                                             ("other", 0.2))}, 44100
    monkeypatch.setattr(AA, "separate_all", fake_sep)
    monkeypatch.setattr(AA, "beat_this_available", lambda: False)

    js = audio_dir / "mix_song_c.json"
    before = js.read_text()
    track = json.loads(before)
    track.pop("stems")                                     # an analysed track without stems
    track["analysis"]["vocals"] = None
    js.write_text(json.dumps(track))
    try:
        kw = dict(demucs=True, stems=True)
        assert AL.missing_parts(track, **kw) == {"vocals", "stems"}
        assert AL.missing_parts(None, **kw) == {"new", "key", "vocals", "stems"}
        _done = {**track, "analysis": {**track["analysis"], "vocals": [0.1]}}
        assert AL.missing_parts({**_done, "stems": {"vocals": "v", "instrumental": "i"}}, **kw) == {"stems"}
        assert AL.missing_parts({**_done, "stems": {k: k for k in AL.ALL_STEMS}}, **kw) == set()
        jobs = {tid: parts for _, tid, parts in AL.plan_jobs(lib, None, **kw)}
        assert jobs == {"mix_song_c": {"vocals", "stems"}, "brand_new": {"new", "key", "vocals", "stems"}}

        f = lib / "mix_song_c.m4a"
        AL.run_job(f, "mix_song_c", jobs["mix_song_c"], "Audio-Library/mix_song_c.m4a", True)
        made = json.loads(js.read_text())
        assert made["stems"]["vocals"] == "/content/drive/MyDrive/scene-labeling/stems/mix_song_c/vocals.m4a"
        for part in AL.ALL_STEMS:            # 🧠 drums / bass / other kept too
            assert (tmp_path / "scene-labeling" / "stems" / "mix_song_c" / f"{part}.m4a").exists(), part
            assert made["stems"][part].endswith(f"/mix_song_c/{part}.m4a")
        assert made["analysis"]["vocals"] and made["analysis"]["methods"]["vocals"] == "demucs"
        assert made["beat_times"] == track["beat_times"]          # beats untouched
        assert calls == ["mix_song_c.m4a"]                         # one Demucs pass
        assert AL.missing_parts(made, **kw) == set()

        AL.run_job(lib / "brand_new.m4a", "brand_new", jobs["brand_new"], "Audio-Library/brand_new.m4a", True)
        new = json.loads((audio_dir / "brand_new.json").read_text())
        assert new["source_path"] == "/content/drive/MyDrive/Audio-Library/brand_new.m4a"
        assert new["analysis"]["vocals"] and new["key"]["camelot"] == "8B" and new["stems"]
        assert calls == ["mix_song_c.m4a", "brand_new.m4a"]       # Demucs once for analysis + stems
        assert AL.missing_parts(new, **kw) == set()
    finally:
        js.write_text(before)
        (audio_dir / "brand_new.json").unlink(missing_ok=True)


def test_mix_page_mashup_with_no_stems_anywhere(mixlib):
    """No track has stems yet (the laptop hasn't made any): adding a part must not fail — the
    '➕ Lay over…' form must cope — and the full song can still be laid over a part."""
    import json
    from tests.apptest_util import page_app, errors
    saved = {}
    for js in mixlib["audio_dir"].glob("mix_song_*.json"):
        saved[js] = js.read_text()
        d = json.loads(saved[js])
        d.pop("stems", None)
        js.write_text(json.dumps(d))
    try:
        at = page_app("pages/3_Media_Library.py", timeout=300)
        at.session_state["lib_kind"] = "🎚️ Mix"
        at.session_state["mx_mode"] = "🧩 Mashup"
        at.session_state["mxm_add_track"] = "mix_song_c"
        at.run()
        at.button(key="mxm_add").click().run()
        assert not errors(at), errors(at)
        assert len(at.session_state["mxm_parts"]) == 1
        uid = at.session_state["mxm_parts"][0]["uid"]
        at.selectbox(key=f"mxm_ltrack_{uid}").set_value("mix_song_a").run()
        assert [r for r in at.radio if r.key == f"mxm_lwhat_{uid}_mix_song_a"][0].options == ["Full song"]
        assert any("only its full song can be laid over" in c.value for c in at.caption)
        at.button(key=f"mxm_ladd_{uid}").click().run()
        assert not errors(at), errors(at)
        ly = at.session_state["mxm_layers"]
        assert len(ly) == 1 and ly[0]["part"] == "full" and ly[0]["fade_beats"] == 4.0
        assert any("🎵 mix_song_a full song" in c.value for c in at.caption)
        at.button(key="mxm_preview").click().run()
        assert not errors(at), errors(at)
        assert at.get("audio")
    finally:
        for js, txt in saved.items():
            js.write_text(txt)


def test_layout_crossfade_per_join_where_and_shape():
    g = _grid()                                   # 120 BPM: a beat is 0.5 s, a bar 2 s
    pieces = [{"track": "x", "start": 0.0, "end": 32.0, "uid": "p1"},
              {"track": "y", "start": 16.0, "end": 48.0, "uid": "p2"},
              {"track": "x", "start": 0.0, "end": 32.0, "uid": "p3",
               "xfade": {"beats": 16.0, "where": "before", "shape": "linear"}}]
    lay = MX.layout({"bpm": 120.0, "xfade_other": 4.0, "pieces": pieces}, {"x": g, "y": g})
    j1, j2 = lay["joins"]
    assert j1["pre"] == pytest.approx(1.0) and j1["post"] == pytest.approx(1.0) and j1["shape"] == "smooth"
    assert j2["pre"] == pytest.approx(8.0) and j2["post"] == 0 and j2["shape"] == "linear"
    # "after", and a fade can never be longer than the part it plays over
    pieces[2]["xfade"] = {"beats": 128.0, "where": "after"}
    j = MX.layout({"bpm": 120.0, "pieces": pieces}, {"x": g, "y": g})["joins"][1]
    assert j["pre"] == 0 and j["post"] == pytest.approx(0.9 * 32.0)
    pieces[2]["xfade"] = {"beats": 0.0}               # a straight cut on the bar line
    assert MX.layout({"bpm": 120.0, "pieces": pieces}, {"x": g, "y": g})["joins"][1]["xfade"] < 0.02


def test_render_long_blend_plays_both_songs_together(mixlib):
    """A 4-bar 'next song comes in early' crossfade: the incoming song is already playing during
    the 4 bars before the join, under the outgoing one; outside the blend only one song plays."""
    import mixer
    t = mixlib["tracks"]
    gc, ga = MX.grid(t["mix_song_c"]), MX.grid(t["mix_song_a"])
    rec = {"bpm": 120.0, "level_match": False, "fade_out": 0.0,
           "pieces": [{"track": "mix_song_c", "start": 0.0, "end": MX.snap_to_bar(gc, 32.0), "uid": "p1"},
                      {"track": "mix_song_a", "start": MX.snap_to_bar(ga, 16.0), "end": MX.snap_to_bar(ga, 40.0),
                       "uid": "p2", "xfade": {"beats": 16.0, "where": "before", "shape": "smooth"}}]}
    info = mixer.render(rec, {k: t[k] for k in ("mix_song_c", "mix_song_a")}, mixlib["root"] / "blend.m4a")
    jn = info["layout"]["joins"][0]
    assert jn["pre"] == pytest.approx(16 * MX.beat_len(120.0), abs=0.01) and jn["post"] == 0
    y = mixer.decode(mixlib["root"] / "blend.m4a")
    c = mixer.decode(mixer.local_path(t["mix_song_c"], "full"))
    sr = mixer.SR
    early = int((jn["at"] - jn["pre"] - 2.0) * sr)            # before the blend: song C alone
    assert np.corrcoef(y[early:early + sr, 0], c[early:early + sr, 0])[0, 1] > 0.98
    mid = int((jn["at"] - jn["pre"] / 2) * sr)                  # in the middle of it: C plus A
    r = np.corrcoef(y[mid:mid + sr // 2, 0], c[mid:mid + sr // 2, 0])[0, 1]
    assert 0.3 < r < 0.95


# ---------------------------------------------------------------------------
# 🖐 Crossfades dragged on the timeline, 🎛️ bass swap
# ---------------------------------------------------------------------------

def test_layout_dragged_split_and_bass_swap_point():
    g = _grid()                                   # 120 BPM: a beat is 0.5 s, a bar 2 s
    pieces = [{"track": "x", "start": 0.0, "end": 32.0, "uid": "p1"},
              {"track": "y", "start": 16.0, "end": 48.0, "uid": "p2",
               "xfade": {"beats": 8.0, "where": 0.75, "bass_swap": True}}]
    jn = MX.layout({"bpm": 120.0, "pieces": pieces}, {"x": g, "y": g})["joins"][0]
    assert jn["pre"] == pytest.approx(3.0) and jn["post"] == pytest.approx(1.0)     # 6 beats before, 2 after
    assert jn["bass_swap"] and jn["swap_at"] == pytest.approx(jn["at"] - 1.0)        # the beat in the middle
    assert MX.where_from_frac(0.5) == "centre" and MX.where_from_frac(1.0) == "before"
    assert MX.where_from_frac(0.75) == 0.75 and MX.where_frac(0.75) == 0.75 and MX.where_frac("after") == 0.0
    # centred on the bar line: the swap is on the line; too short a fade: no swap
    pieces[1]["xfade"] = {"beats": 8.0, "bass_swap": True}
    jn = MX.layout({"bpm": 120.0, "pieces": pieces}, {"x": g, "y": g})["joins"][0]
    assert jn["swap_at"] == pytest.approx(jn["at"])
    pieces[1]["xfade"] = {"beats": 0.5, "bass_swap": True}
    assert not MX.layout({"bpm": 120.0, "pieces": pieces}, {"x": g, "y": g})["joins"][0]["bass_swap"]


def test_bass_swap_fade_keeps_one_low_end_and_the_highs_crossfading():
    """_fade_region: with bass swap the low band is full then gone (outgoing) / gone then full
    (incoming) around the swap point, the highs follow the ramp, and nothing else changes."""
    import mixer
    sr = mixer.SR
    n = 4 * sr
    t = np.arange(n) / sr
    lowtone = (0.5 * np.sin(2 * np.pi * 55 * t)).astype(np.float32)
    hightone = (0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    jn = {"bass_swap": True, "swap_at": 2.0}
    ramp = mixer._ramp(n, True, False)

    def level(y, a, b, hz):
        seg = y[int(a * sr):int(b * sr), 0]
        ref = np.sin(2 * np.pi * hz * t[int(a * sr):int(b * sr)])
        return abs(float(np.dot(seg, ref)) * 2 / len(seg))

    out = np.stack([lowtone + hightone] * 2, axis=1)
    mixer._fade_region(out, ramp, jn, 0.0, 120.0, outgoing=True)
    assert level(out, 1.0, 1.8, 55) > 0.45                      # bass full until the swap…
    assert level(out, 2.3, 3.5, 55) < 0.03                      # …then gone
    assert 0.2 < level(out, 1.4, 1.6, 2000) < 0.45              # the highs are fading as usual
    inc = np.stack([lowtone + hightone] * 2, axis=1)
    mixer._fade_region(inc, mixer._ramp(n, True, True), jn, 0.0, 120.0, outgoing=False)
    assert level(inc, 0.5, 1.7, 55) < 0.03 and level(inc, 2.3, 3.5, 55) > 0.45
    plain = np.stack([lowtone + hightone] * 2, axis=1)
    mixer._fade_region(plain, ramp, {"bass_swap": False}, 0.0, 120.0, outgoing=True)
    assert np.allclose(plain[:, 0], (lowtone + hightone) * ramp, atol=1e-6)


def test_render_with_bass_swap(mixlib):
    import mixer
    t = mixlib["tracks"]
    gc, ga = MX.grid(t["mix_song_c"]), MX.grid(t["mix_song_a"])
    rec = {"bpm": 120.0, "level_match": False, "fade_out": 0.0,
           "pieces": [{"track": "mix_song_c", "start": 0.0, "end": MX.snap_to_bar(gc, 32.0), "uid": "p1"},
                      {"track": "mix_song_a", "start": MX.snap_to_bar(ga, 16.0), "end": MX.snap_to_bar(ga, 40.0),
                       "uid": "p2", "xfade": {"beats": 8.0, "bass_swap": True}}]}
    info = mixer.render(rec, {k: t[k] for k in ("mix_song_c", "mix_song_a")}, mixlib["root"] / "bass.m4a")
    jn = info["layout"]["joins"][0]
    assert jn["bass_swap"] and jn["swap_at"] == pytest.approx(jn["at"], abs=0.01)
    assert info["duration"] > 0


def test_timeline_drag_sets_the_join(monkeypatch):
    import mix_ui
    state = {}
    monkeypatch.setattr(mix_ui.st, "session_state", state)
    parts = [{"uid": "a", "track": "x"}, {"uid": "b", "track": "y", "xfade": {"shape": "linear", "bass_swap": True}}]
    state["mxm_tl"] = {"uid": "b", "pre_beats": 6, "post_beats": 2, "nonce": 1}
    mix_ui._apply_timeline_drag(parts)
    assert parts[1]["xfade"] == {"shape": "linear", "bass_swap": True, "beats": 8.0, "where": 0.75}
    assert state["mxm_xfl_b"] == "2 bars" and state["mxm_xfw_b"] == 0.75
    parts[1]["xfade"]["beats"] = 1.0                     # the same nonce again: nothing changes
    mix_ui._apply_timeline_drag(parts)
    assert parts[1]["xfade"]["beats"] == 1.0
    state["mxm_tl"] = {"uid": "b", "pre_beats": 0, "post_beats": 5, "nonce": 2}
    mix_ui._apply_timeline_drag(parts)
    assert parts[1]["xfade"]["where"] == "after" and state["mxm_xfl_b"] == "🖐 5 beats"


def test_render_neural_join_from_stems(mixlib):
    """🧠 A neural join: both sides' four stems blended with their own curves (core/stem_mix.py) over
    `bars` bars before the join; without stems, an ordinary crossfade and a warning."""
    import mixer
    t = mixlib["tracks"]
    gc, ga = MX.grid(t["mix_song_c"]), MX.grid(t["mix_song_a"])
    rec = {"bpm": 120.0, "level_match": False, "fade_out": 0.0,
           "pieces": [{"track": "mix_song_c", "start": 0.0, "end": MX.snap_to_bar(gc, 32.0), "uid": "p1"},
                      {"track": "mix_song_a", "start": MX.snap_to_bar(ga, 16.0), "end": MX.snap_to_bar(ga, 40.0),
                       "uid": "p2", "xfade": {"neural": {"profile": "vocal_swap", "bars": 4}}}]}
    tracks = {k: t[k] for k in ("mix_song_c", "mix_song_a")}
    need = mixer.parts_needed(rec, tracks)
    assert need["mix_song_a"] >= set(mixer.FOUR_STEMS) and "full" in need["mix_song_c"]
    info = mixer.render(rec, tracks, mixlib["root"] / "neural.m4a")
    jn = info["layout"]["joins"][0]
    bar = 4 * 60 / 120
    assert jn["neural"]["profile"] == "vocal_swap" and jn["pre"] == pytest.approx(4 * bar, abs=0.01)
    assert jn["post"] == pytest.approx(0.0, abs=1e-6) and not jn["bass_swap"]
    assert not any("🧠" in w for w in info["warnings"])
    assert info["duration"] > 0
    y = mixer.decode(mixlib["root"] / "neural.m4a")
    at = int(jn["at"] * mixer.SR)
    mid = y[at - int(2 * bar * mixer.SR):at]
    assert np.abs(mid).max() > 0.01                             # the blend isn't silent
    assert np.abs(y).max() <= 1.05                               # (AAC can overshoot its input a little)
    # A track without drums / bass / other: an ordinary crossfade, said so.
    bare = {k: {**v, "stems": {s: v["stems"][s] for s in ("vocals", "instrumental")}} for k, v in tracks.items()}
    assert "drums" not in mixer.parts_needed(rec, bare)["mix_song_a"]
    info2 = mixer.render(rec, bare, mixlib["root"] / "neural_fallback.m4a")
    assert any("🧠" in w and "ordinary crossfade" in w for w in info2["warnings"])


def test_mix_page_neural_join_place_and_preview(mixlib):
    """🧩 Mashup → a join set to 🧠 Neural: profile, bars, 📍 Place, then ▶️ Preview renders the stem blend."""
    from tests.apptest_util import page_app, errors
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎚️ Mix"
    at.session_state["mx_mode"] = "🧩 Mashup"
    at.session_state["mxm_add_track"] = "mix_song_c"
    at.run()
    at.button(key="mxm_add").click().run()
    at.selectbox(key="mxm_add_track").set_value("mix_song_a").run()
    at.button(key="mxm_add").click().run()
    assert not errors(at), errors(at)
    uid2 = at.session_state["mxm_parts"][1]["uid"]
    at.selectbox(key=f"mxm_xfl_{uid2}").set_value("🧠 Neural").run()
    assert not errors(at), errors(at)
    at.selectbox(key=f"mxm_nnp_{uid2}").set_value("drums_first").run()
    at.selectbox(key=f"mxm_nnb_{uid2}").set_value(4).run()
    assert at.session_state["mxm_parts"][1]["xfade"] == {"neural": {"profile": "drums_first", "bars": 4}}
    parts = [dict(p) for p in at.session_state["mxm_parts"]]
    parts[0]["end"], parts[1]["start"], parts[1]["end"] = 40.0, 16.0, 40.0   # long enough for the blend
    at.session_state["mxm_parts"] = parts
    before = dict(at.session_state["mxm_parts"][0])
    at.button(key=f"mxm_nnplace_{uid2}").click().run()
    assert not errors(at), errors(at)
    assert any("Placed: the 4-bar blend" in i.value for i in at.info), [i.value for i in at.info]
    a = at.session_state["mxm_parts"][0]
    assert a["start"] == before["start"] and a["end"] <= before["end"] + 1e-6
    at.button(key="mxm_preview").click().run()
    assert not errors(at), errors(at)
    assert at.get("audio")


def test_mix_page_drop_swap_is_a_neural_bass_drop_with_stems(mixlib, monkeypatch):
    import mix_ui
    from tests.apptest_util import page_app, errors
    real_full = mix_ui.full_track

    def full(tid):
        t = real_full(tid)
        if t is None:
            return None
        dur = float(t["duration_sec"])
        return {**t, "analysis": {**(t.get("analysis") or {}), "builds": [{"start": dur * 0.3, "end": dur * 0.5}],
                                  "drops": [dur * 0.5]}}
    monkeypatch.setattr(mix_ui, "full_track", full)
    real_infos = mix_ui._infos
    monkeypatch.setattr(mix_ui, "_infos", lambda s: [dict(i, builds=1, drops=1) for i in real_infos(s)
                                                     if i["track_id"] in ("mix_song_a", "mix_song_c")])
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎚️ Mix"
    at.session_state["mx_mode"] = "💡 Suggestions"
    at.run()
    go = [b for b in at.button if (b.key or "").startswith("mx_sug_go_d_")]
    assert go
    go[0].click().run()
    assert not errors(at), errors(at)
    xf = at.session_state["mxm_parts"][1]["xfade"]
    assert xf["neural"]["profile"] == "bass_drop" and xf["neural"]["bars"] in (4, 8, 16)


def test_transition_suggestions(mixlib):
    """🎯 core/transitions.suggest: phrase-aligned, room for the blend, scored, at most 3, different."""
    from core import transitions as TR
    t = mixlib["tracks"]
    ta, tb = t["mix_song_c"], t["mix_song_a"]
    ga, gb = MX.grid(ta), MX.grid(tb)
    sug = TR.suggest(ta, tb)
    assert 1 <= len(sug) <= 3
    assert [s["score"] for s in sug] == sorted([s["score"] for s in sug], reverse=True)
    for s in sug:
        assert s["profile"] in ("smooth", "vocal_swap", "drums_first", "bass_drop")
        assert s["bars"] in TR.BARS_CHOICES
        assert s["a_end"] <= ga["duration"] + 1e-3 and s["b_in"] >= 0
        assert min(abs(s["b_full"] - x) for x in gb["bars"]) < 0.01            # on a bar line
        assert s["a_end"] - s["a_blend"] == pytest.approx(s["bars"] * ga["bar_len"], abs=0.05)
    pieces = TR.recipe_pieces("mix_song_c", "mix_song_a", sug[0])
    assert pieces[0]["end"] == sug[0]["a_end"] and pieces[1]["start"] == sug[0]["b_full"]
    assert pieces[1]["xfade"]["neural"]["bars"] == sug[0]["bars"]


def test_mix_page_transition_mode(mixlib):
    """🎚️ Transition: suggestions load, the deck editor draws, a drag moves B, ▶️ Preview renders just the
    transition, and 🧩 Start a mashup carries it over."""
    from tests.apptest_util import page_app, errors
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎚️ Mix"
    at.session_state["mx_mode"] = "🎚️ Transition"
    at.session_state["mxt_a"] = "mix_song_c"
    at.session_state["mxt_b"] = "mix_song_a"
    at.run()
    assert not errors(at), errors(at)
    key = "mxt_state::mix_song_c::mix_song_a"
    s0 = dict(at.session_state[key])
    assert any("Suggested transitions" in m.value for m in at.markdown)
    # a drag in the editor comes back as a value
    b_bars = MX.grid(mixlib["tracks"]["mix_song_a"])["bars"]
    new_b = float(b_bars[min(12, len(b_bars) - 1)])
    at.session_state["mxt_deck"] = {"a_end": s0["a_end"], "b_full": new_b, "bars": 4, "nonce": 123}
    at.run()
    assert not errors(at), errors(at)
    assert at.session_state[key]["b_full"] == new_b and at.session_state[key]["bars"] == 4
    at.button(key="mxt_preview").click().run()
    assert not errors(at), errors(at)
    assert at.session_state["mxt_pv"]["path"]
    at.button(key="mxt_new").click().run()
    assert not errors(at), errors(at)
    assert at.session_state["mx_mode"] == "🧩 Mashup"
    parts = at.session_state["mxm_parts"]
    assert [p["track"] for p in parts] == ["mix_song_c", "mix_song_a"]
    assert parts[1]["start"] == new_b and parts[1]["xfade"]["neural"]["bars"] == 4


def test_mashup_neural_joins_survive_leaving_the_screen(mixlib):
    """3 parts, both joins 🧠: going to 🎚️ Transition and back keeps every join (Streamlit forgets the join
    boxes when they aren't drawn — they used to come back as Default and overwrite the joins, 23 Oct 2026)."""
    from tests.apptest_util import page_app, errors
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎚️ Mix"
    at.session_state["mx_mode"] = "🧩 Mashup"
    nn = {"neural": {"profile": "vocal_swap", "bars": 4, "curves": {"out": {"mid": 0.3, "steep": 2.0}}}}
    at.session_state["mxm_parts"] = [
        {"track": "mix_song_c", "start": 0.0, "end": 40.0, "gain_db": 0.0, "part": "full", "uid": "P1"},
        {"track": "mix_song_a", "start": 16.0, "end": 40.0, "gain_db": 0.0, "part": "full", "uid": "P2", "xfade": dict(nn)},
        {"track": "mix_song_c", "start": 30.0, "end": 60.0, "gain_db": 0.0, "part": "full", "uid": "P3", "xfade": dict(nn)}]
    at.run()
    assert not errors(at), errors(at)
    for mode in ("🎚️ Transition", "🧩 Mashup", "💡 Suggestions", "🧩 Mashup"):
        at.session_state["mx_mode"] = mode
        at.run()
        assert not errors(at), (mode, errors(at))
    xf = [p.get("xfade") for p in at.session_state["mxm_parts"]]
    assert xf[1] == nn and xf[2] == nn
    assert {at.selectbox(key=f"mxm_xfl_{u}").value for u in ("P2", "P3")} == {"🧠 Neural"}
