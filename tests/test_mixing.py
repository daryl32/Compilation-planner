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
    assert any(b.label.startswith(("🟢", "🟠", "🔴", "⚪")) for b in at.button)     # listen-to-join buttons
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


def test_make_stems_local_plans_and_notes_stems(mixlib, monkeypatch, tmp_path):
    import numpy as np_
    import config
    import make_stems_local as MS
    import analyse_audio_local as AL
    import audio_analysis as AA
    lib = tmp_path / "Audio-Library"
    lib.mkdir()
    (lib / "mix_song_c.m4a").write_bytes(Path(mixlib["tracks"]["mix_song_c"]["source_path"]).read_bytes())
    monkeypatch.setattr(MS, "AUDIO_DIR", mixlib["audio_dir"])
    monkeypatch.setattr(AL, "drive_root", lambda: tmp_path)
    monkeypatch.setattr(MS, "AA", AA)
    monkeypatch.setattr(MS, "LOCAL_TEMP_DIR", tmp_path / "tmp")
    # the test songs already have stems → nothing to do unless --redo
    assert MS.plan(lib, None, False) == []
    jobs = MS.plan(lib, "song_c", True)
    assert [tid for _, tid in jobs] == ["mix_song_c"]
    fake = np_.zeros((44100, 2), dtype=np_.float32)
    monkeypatch.setattr(AA, "separate_stems", lambda path: (fake, fake, 44100))
    js = mixlib["audio_dir"] / "mix_song_c.json"
    before = js.read_text()
    try:
        MS.run(*jobs[0])
        made = json.loads(js.read_text())
        assert made["stems"]["vocals"] == "/content/drive/MyDrive/scene-labeling/stems/mix_song_c/vocals.m4a"
        assert (tmp_path / "scene-labeling" / "stems" / "mix_song_c" / "instrumental.m4a").exists()
        assert made["analysis"] and made["bpm"]                     # the rest of the track JSON is kept
    finally:
        js.write_text(before)
