"""⚡ Edge Mode's energy-matched mix: the energy map, phases in pairs, the picker, the Mix engine's joins."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.edge_support import (DATA, VIDS, E_CARD_LEAD, _check_plan, _emap, _level, _pixel, _seconds,  # noqa: F401
                                _timelines, _wait_job)


def test_the_phases_play_in_pairs_on_the_stretch_that_fits_their_energy():
    import edge_energy as EE
    import edge_mode as E
    sched = E.schedule("chill", 3)                         # 18 phases → 9 pairs
    assert EE.groups(sched)[:3] == [[0, 1], [2, 3], [4, 5]] and len(EE.groups(sched)) == 9
    med = E.schedule("medium", 3)                          # wait slow | medium fast | slow | medium fast | extra stop
    assert [len(g) for g in EE.groups(med)[:5]] == [2, 2, 1, 2, 2]     # (the extra Slow has no free neighbour)
    hard = [{"index": k, "round": 1, "profile": p, "start": 20.0 * k, "end": 20.0 * k + 20.0}
            for k, p in enumerate(("wait", "fast", "extra", "extra", "slow", "medium", "fast", "extra", "stop"))]
    assert EE.groups(hard) == [[0, 1], [2, 3], [4], [5, 6], [7, 8]]       # lone draws pair up with a lone neighbour
    calm = _emap([0.1] * 30 + [0.4] * 30)                  # 120 s: quiet, then gentle
    loud = _emap([0.6] * 30 + [1.0] * 30)                  # 120 s: busy, then the peak
    got = EE.pick_sections(sched, {"calm": calm, "loud": loud})
    assert len(got) == 9 and all(p["play"] == 40.0 for p in got)
    by = {(p["round"], p["profile"]): p for p in got}
    assert by[(1, "wait")]["track_id"] == "calm" and by[(1, "extra")]["track_id"] == "loud"
    assert by[(1, "extra")]["bar"] >= 60 - 1e-6           # the peak, Stop fading on it (not judged)
    assert by[(1, "medium")]["track_id"] == "loud"
    for p, q in zip(got, got[1:]):                          # back to back, each on its pair's start
        assert abs(p["offset"] + p["play"] - q["offset"]) < 1e-6
    assert [p["first"] for p in got] == [True, False, False] * 3
    per = EE.pick_sections(sched, {"calm": calm, "loud": loud}, allowed=lambda ph: ["calm" if ph["round"] == 1 else "loud"])
    assert {p["track_id"] for p in per if p["round"] == 1} == {"calm"} and {p["track_id"] for p in per if p["round"] == 2} == {"loud"}
    flat = EE.pick_sections(sched[:6], {"flat": _emap([0.5] * 90)})
    assert flat[1]["carry"] and flat[2]["carry"] and flat[2]["bar"] == flat[1]["bar"] + 40.0     # carries straight on
    # a jump within a song: the smoother one wins when the energy is level
    fits = EE.pick_sections(sched[:4], {"flat": _emap([0.5] * 90)}, jump_fit=lambda tid, a, b: 1.0 if b == 100.0 else 0.0)
    assert fits[1]["bar"] in (100.0, fits[0]["bar"] + 40.0)


def test_the_energy_map_reads_the_sections_suggesters_bars():
    import edge_energy as EE
    import planner.compose as PC
    m = EE.energy_map(PC._load_track("song_full"))
    assert len(m["bars"]) == len(m["energy"]) > 10 and 0.0 <= m["energy"].min() and m["energy"].max() <= 1.0
    assert m["energy"].max() - m["energy"].min() > 0.5
    assert EE.window_energy(m, 0.0, 10.0) < EE.window_energy(m, 40.0, 60.0)    # the test song builds up


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_joins_are_the_mix_engines_stem_blends_round_by_round(tmp_path, monkeypatch):
    import config
    import edge_energy as EE
    import render_preview
    from tests import mix_audio
    audio_dir, files = tmp_path / "audio", tmp_path / "files"
    audio_dir.mkdir()
    monkeypatch.setattr(config, "AUDIO_DIR", audio_dir)
    tracks = mix_audio.install(audio_dir, files)          # two synthetic songs with their four stems
    assert all(EE.blend_bars(t) == 2 for t in tracks.values())      # 2 bars ≈ 3.8 s at 120–126 BPM
    def piece(g, r, profs, tid, bar, offset, carry=False, first=False, play=20.0):
        return {"group": g, "round": r, "profile": profs[0], "profiles": profs, "track_id": tid, "bar": bar,
                "offset": offset, "play": play, "carry": carry, "first": first,
                "target": {"wait": 0.29, "medium": 0.66, "extra": 0.95, "slow": 0.38}[profs[0]]}
    # a stretch that carries straight on is one piece of the recipe; every other join a 🧠 blend
    recs = EE.round_recipes([piece(0, 1, ["wait", "slow"], "mix_song_c", 0.0, 0.0, first=True),
                             piece(1, 1, ["medium", "fast"], "mix_song_a", 4.0, 20.0),
                             piece(2, 1, ["extra", "stop"], "mix_song_a", 24.0, 40.0, carry=True)], tracks)
    r1 = recs[0][1]["pieces"]
    assert [(p["track"], p["start"], p["end"]) for p in r1] == [("mix_song_c", 0.0, 20.0), ("mix_song_a", 4.0, 44.0)]
    assert r1[1]["xfade"]["neural"] == {"profile": "vocal_swap", "curves": {"preset": "hold_cut"}, "bars": 2} and "xfade" not in r1[0]   # his tested join
    # rendered by the Mix engine, a round at a time (songs of 64 s and 38 s; stretches of 20 s here)
    pieces = [piece(0, 1, ["wait", "slow"], "mix_song_c", 0.0, 0.0, first=True),
              piece(1, 1, ["medium", "fast"], "mix_song_a", 4.0, 20.0),
              piece(2, 1, ["extra", "stop"], "mix_song_c", 40.0, 40.0),
              piece(3, 2, ["wait", "slow"], "mix_song_c", 8.0, 60.0, first=True)]
    recs = EE.round_recipes(pieces, tracks)
    assert [(r, s_) for r, _, s_ in recs] == [(1, 60.0), (2, 20.0)]
    assert [p.get("xfade", {}).get("neural", {}).get("profile") for p in recs[0][1]["pieces"]] == [None, "vocal_swap", "vocal_swap"]
    monkeypatch.setattr(render_preview, "colab_to_local", lambda p: p)
    got, warns = EE.render_rounds(pieces, tracks, tmp_path)
    assert [(r, secs) for r, _, secs in got] == [(1, 60.0), (2, 20.0)] and not [w for w in warns if "stems" in w]
    assert abs(_seconds(got[0][1]) - 60.0) < 0.1 and abs(_seconds(got[1][1]) - 20.0) < 0.1
    steady = _level(got[0][1], 10, 14)
    for t in (17.0, 18.5, 20.5, 37.0, 38.5, 40.5):         # through each blend: no hole
        assert _level(got[0][1], t - 0.5, t + 0.5) > 0.4 * steady, t


def test_compile_with_the_energy_mix_follows_each_section(monkeypatch):
    out = _timelines(monkeypatch)
    try:
        import planner.edge as PE
        res = PE.compile_edge(["song_full", "song_noanalysis"], VIDS, "chill", 3, energy_mix=True)
        plan, edge = res["plan"], res["plan"]["edge"]
        _check_plan(plan, 360.0)
        assert edge["music"]["energy_mix"]
        assert len(edge["playlist"]) == 9 and all("bar" in p and "source_path" in p for p in edge["playlist"])
        assert any("Energy mix" in n and "pairs" in n for n in res["notes"])
        import json
        json.dumps(plan)                                   # the plan downloads as JSON (no numpy values in it)
        for p in edge["playlist"]:                         # the footage of a pair comes from its stretch's song
            inside = [e for e in plan["timeline"] if p["offset"] - 1e-3 <= e["track_time"][0] < p["offset"] + p["play"] - 1e-3]
            assert inside and {e["edge_track_id"] for e in inside} == {p["track_id"]}
        tl = plan["timeline"]
        for x, y in zip(tl, tl[1:]):                       # one after another, no gap, no overlap
            assert abs(x["track_time"][1] - y["track_time"][0]) < 0.06
    finally:
        shutil.rmtree(out, ignore_errors=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_render_plays_the_energy_mix_under_the_fades(tmp_path, monkeypatch):
    import sys
    import types
    import edge_energy as EE
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    made = {}
    for r in (1, 2):
        made[r] = tmp_path / f"r{r}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency={200 * r}:duration=59.8",
                        "-af", "volume=3", str(made[r])], check=True)
    seen = []

    def fake_rounds(parts, tracks, tmp_dir, progress=None):
        seen.append((len(parts), sorted(tracks)))
        return [(1, made[1], 60.0), (2, made[2], 60.0)], ["mix_song_x: no stems"]
    monkeypatch.setattr(EE, "render_rounds", fake_rounds)
    monkeypatch.setattr(R, "_song_tracks", lambda parts: {"a": {}, "b": {}})
    sched = [{"index": k, "round": 1 + k // 3, "profile": p, "start": 20.0 * k, "end": 20.0 * k + 20.0}
             for k, p in enumerate(("wait", "extra", "stop", "wait", "extra", "stop"))]
    parts = [{"track_id": "a", "offset": 0.0, "play": 60.0, "round": 1}, {"track_id": "b", "offset": 60.0, "play": 60.0, "round": 2}]
    edge = {"schedule": sched, "playlist": parts, "beat": {"on": False}, "voiceover": {"on": False, "cues": []},
            "music": {"fade_stop": True, "duck": True, "energy_mix": True}, "signals": {"sound": False}}

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    notes = []
    out = R._mix(edge, [], types.SimpleNamespace(), tmp_path, run, notes)
    assert seen == [(2, ["a", "b"])] and notes == ["mix_song_x: no stems"]
    assert abs(_seconds(out) - 120.0) < 0.05
    full = _level(out, 25, 35)
    assert full > 0.1 and _level(out, 0, 2) < 0.2 * full and _level(out, 58, 59.9) < 0.2 * full    # 🌙 still on top
    assert _level(out, 85, 95) > 0.5 * full                # round 2's own file, from 60 s


def test_a_livelier_stretch_lands_its_drop_on_the_phase_change():
    import edge_energy as EE
    m = _emap([0.3] * 10 + [0.9] * 50)                     # the energy jumps at 20 s…
    m["drops"] = [60.0]                                     # …and the analysis has a drop at 60 s
    assert EE.drop_score(m, 60.0) == 1.0 and 0.9 < EE.drop_score(m, 20.0) < 1.0 and EE.drop_score(m, 40.0) == 0.0
    sched = [{"index": k, "round": 1, "profile": p, "start": 20.0 * k, "end": 20.0 * k + 20.0}
             for k, p in enumerate(("wait", "slow", "extra", "stop"))]
    calm = _emap([0.25] * 60)
    got = EE.pick_sections(sched, {"calm": calm, "drop": m}, allowed=lambda ph: ["calm"] if ph["profile"] in ("wait", "slow") else ["drop"])
    assert got[1]["track_id"] == "drop" and got[1]["bar"] == 60.0       # Extra starts right on the song's drop
    recs = EE.round_recipes(got, {"calm": {"bpm": 120}, "drop": {"bpm": 120}})
    assert recs[0][1]["pieces"][1]["xfade"]["neural"] == {"profile": "vocal_swap", "curves": {"preset": "hold_cut"}, "bars": 2}
    down = [{**got[1], "group": 0, "first": True}, {**got[0], "group": 1, "first": False, "round": 1}]
    assert EE.round_recipes(down, {"calm": {"bpm": 120}, "drop": {"bpm": 120}})[0][1]["pieces"][1]["xfade"]["neural"]["profile"] == "vocal_swap"
