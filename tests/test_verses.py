"""⚔️ Verses Battle: the draft, the rounds, the weave, the sound parts, the compile, the render and the page."""
from __future__ import annotations

import random
import shutil
import subprocess

import pytest

from tests.edge_support import DATA, _level, _pixel, _seconds, _timelines  # noqa: F401

ACTORS = {"vid_a": ["Ava"], "vid_b": ["Ava"], "vid_c": ["Bea"], "vid_f": ["Bea", "Ava"]}


def _actors(monkeypatch):
    import library_common as LC
    monkeypatch.setattr(LC, "video_details", lambda: {v: {"actors": a, "tags": []} for v, a in ACTORS.items()})


def test_draft_and_fighter_videos():
    import verses_battle as VB
    pool = VB.actor_videos({v: {"actors": a} for v, a in ACTORS.items()}, {"vid_a", "vid_b", "vid_c", "vid_f"})
    assert pool == {"Ava": ["vid_a", "vid_b", "vid_f"], "Bea": ["vid_c", "vid_f"]}
    # a video with both goes to the side with fewer of her own
    assert VB.fighter_videos(pool, "Ava", "Bea") == (["vid_a", "vid_b"], ["vid_c", "vid_f"])
    looks = {"x": [1, 0], "y": [0.9, 0.1], "z": [-1, 0], "w": [0, 1]}
    rng = random.Random(1)
    assert VB.draft(["x", "y", "z"], "rivals", rng, first="x", looks=looks)[1] in ("y", "z")
    # contrast picks from the least alike first: with only 2 others both may come, but over many draws z leads
    picks = [VB.draft(list(looks), "contrast", random.Random(k), first="x", looks=looks)[1] for k in range(30)]
    assert set(picks) <= {"z", "w", "y"} and picks.count("z") >= 8
    with pytest.raises(ValueError):
        VB.draft(["x"], "any", rng)


@pytest.mark.parametrize("layout", ["mixed", "split", "full"])
@pytest.mark.parametrize("winner", ["A", "B", "draw"])
def test_the_rounds_cover_the_battle(layout, winner):
    import verses_battle as VB
    grid = [i * 2.0 for i in range(60)]                     # a bar every 2 s
    ws = VB.schedule(118.0, grid, grid, layout, winner, first="B")
    assert ws[0]["start"] == 0.0 and ws[-1]["end"] == 118.0
    for x, y in zip(ws, ws[1:]):
        assert abs(x["end"] - y["start"]) < 1e-6 and y["start"] > x["start"]
    assert [r["round"] for r in VB.round_spans(ws)] == [1, 2, 3, 4]
    assert ws[0]["side"] == "B" and ws[0]["round"] == 1                # who opens
    last = ws[-1]
    assert (last["side"] == "split") if winner == "draw" else (last["side"] == winner)
    splits = [w for w in ws if w["side"] == "split"]
    if layout == "full":
        assert not splits or (winner == "draw" and splits == [last])
    else:
        assert any(w["round"] == 3 for w in splits)
    # swaps get faster: the trade-off windows of the second half are shorter
    r2 = [w["end"] - w["start"] for w in ws if w["round"] == 2]
    assert max(r2[-2:]) < max(r2[:2]) + 1e-6
    st = VB.screen_time(ws)
    assert abs(st["A"] + st["B"] - 118.0) < 1e-3
    m = VB.meter(ws, winner)
    assert m[0] == (0.0, 0.5)
    end = m[-1][1]
    assert (end > 0.7) if winner == "A" else (end < 0.3) if winner == "B" else (0.3 < end < 0.7)
    assert VB.describe(ws)[0].startswith("Round 1 · Entrances")


def test_weave_and_sound_parts():
    import verses_battle as VB

    def tl(vid):
        return [{"track_time": [k * 3.0, k * 3.0 + 3.0],
                 "scenes": [{"chain": [{"video_id": vid, "clip_start_sec": 10.0 * k, "clip_duration_sec": 3.0}]}]}
                for k in range(10)]
    ws = [{"index": 0, "start": 0.0, "end": 4.0, "side": "A", "round": 1, "round_name": "Entrances", "lead": "A"},
          {"index": 1, "start": 4.0, "end": 10.0, "side": "split", "round": 3, "round_name": "Head to head",
           "lead": "B", "cells": VB._cells("B")},
          {"index": 2, "start": 10.0, "end": 30.0, "side": "B", "round": 4, "round_name": "Final", "lead": "B"}]
    out = VB.weave(ws, {"A": tl("vid_a"), "B": tl("vid_b")})
    assert out[0]["track_time"] == [0.0, 3.0] and out[1]["track_time"] == [3.0, 4.0]
    assert out[1]["scenes"][0]["chain"][0] == {"video_id": "vid_a", "clip_start_sec": 10.0, "clip_duration_sec": 1.0}
    split = [e for e in out if e["verses_side"] == "split"]
    assert [e["track_time"] for e in split] == [[4.0, 6.0], [6.0, 9.0], [9.0, 10.0]]
    assert all(len(e["scenes"]) == 2 and e["cells"][0][2] == 0.35 for e in split)
    assert [s["chain"][0]["video_id"] for s in split[0]["scenes"]] == ["vid_a", "vid_b"]
    assert split[0]["scenes"][1]["chain"][0]["clip_start_sec"] == 11.0          # (B's clip 1 starts at 3 s → 1 s in)
    for e in out:
        for s in e["scenes"]:
            assert abs(sum(x["clip_duration_sec"] for x in s["chain"]) - (e["track_time"][1] - e["track_time"][0])) < 1e-3
    assert [e["segment_index"] for e in out] == list(range(len(out))) and out[-1]["track_time"][1] == 30.0
    # sound: shared — one part; solo — the song of whoever is on screen, a split following its leader
    tracks = {"A": "song_a", "B": "song_b"}
    assert VB.audio_parts(ws, {"A": "s", "B": "s"}, 30.0, "shared", {"s": 90}) == [
        {"side": "A", "track_id": "s", "at": 0.0, "src": 0.0, "dur": 30.0}]
    parts = VB.audio_parts(ws, tracks, 30.0, "solo", {"song_a": 90, "song_b": 90})
    assert [(p["track_id"], p["at"], p["src"], p["dur"]) for p in parts] == [("song_a", 0.0, 0.0, 4.0),
                                                                            ("song_b", 4.0, 4.0, 26.0)]


def test_compile_a_battle(monkeypatch):
    out = _timelines(monkeypatch)
    _actors(monkeypatch)
    try:
        import json
        import planner.verses as PV
        seen = []
        res = PV.compile_verses({"a": "Ava", "b": "Bea", "music": "shared", "track_a": "song_full", "layout": "mixed",
                                 "winner": "A", "intro": True, "overlay": True}, 7, lambda f, t: seen.append(t))
        plan, v = res["plan"], res["plan"]["verses"]
        assert seen and plan["matching_mode"] == "Verses" and v["total_seconds"] == 96.0
        tl = plan["timeline"]
        assert tl[0]["track_time"][0] == 0.0 and abs(tl[-1]["track_time"][1] - 96.0) < 1e-6
        for x, y in zip(tl, tl[1:]):
            assert abs(x["track_time"][1] - y["track_time"][0]) < 1e-6
        sides = {e["verses_side"] for e in tl}
        assert {"A", "B", "split"} <= sides
        for e in tl:          # each side shows only her own videos
            vids = [link["video_id"] for s in e["scenes"] for link in s["chain"]]
            if e["verses_side"] == "A":
                assert set(vids) <= {"vid_a", "vid_b"}
            elif e["verses_side"] == "B":
                assert set(vids) <= {"vid_c", "vid_f"}
        assert v["winner"] == "A" and v["parts"] == [{"side": "A", "track_id": "song_full", "at": 0.0, "src": 0.0, "dur": 96.0}]
        assert v["intro"]["thumbs"]["A"][0] in ("vid_a", "vid_b") and v["intro"]["stats"]["A"][0] == ("Videos", "2")
        assert len(res["rounds"]) == 4 and "⚔️ Ava vs Bea" in PV.used_line(res)
        json.dumps(plan)
        # solo themes, 1-minute cap: two songs, the battle as long as the cap, a fade at the end
        res2 = PV.compile_verses({"a": "Ava", "b": "Bea", "music": "solo", "track_a": "song_full",
                                  "track_b": "song_noanalysis", "layout": "full", "winner": "draw", "length": 60}, 3)
        v2 = res2["plan"]["verses"]
        assert v2["total_seconds"] == 60.0 and v2["fade_out"]
        assert {p["track_id"] for p in v2["parts"]} == {"song_full", "song_noanalysis"}
        assert sum(p["dur"] for p in v2["parts"]) == pytest.approx(60.0)
        assert v2["windows"][-1]["side"] == "split"                     # a draw ends side by side
        with pytest.raises(ValueError):
            PV.compile_verses({"a": "Ava", "b": "Ava", "music": "shared", "track_a": "song_full"}, 1)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_render_carries_the_overlay_card_and_switching_themes(tmp_path, monkeypatch):
    import render_preview as RP
    import verses_battle as VB
    proxies, drive = tmp_path / "proxies", tmp_path / "drive"
    proxies.mkdir()
    for vid, colour in (("vid_a", "0x00a000"), ("vid_c", "0x0000a0")):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=320x180:d=40:r=25",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(proxies / f"{vid}.mp4")], check=True)
    real = RP.colab_to_local
    monkeypatch.setattr(RP, "colab_to_local", lambda p: str(drive / p.split("MyDrive/", 1)[1]) if "MyDrive/" in p else real(p))
    monkeypatch.setattr(RP, "PROXY_DIR", proxies)
    monkeypatch.setattr(RP, "streaming_available", lambda: False)
    for tid, wave in (("song_full", "sine=frequency=440:duration=40"), ("song_noanalysis", "anullsrc=r=44100:cl=stereo")):
        audio = drive / RP.get_track_raw_source(tid).split("MyDrive/", 1)[1]
        audio.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "40", "-i", wave, "-af", "volume=4",
                        str(audio.with_suffix(".wav"))], check=True)
        audio.with_suffix(".wav").rename(audio)
    grid = [i * 2.0 for i in range(20)]
    ws = VB.schedule(20.0, grid, grid, "mixed", "A", first="A")

    def tl(vid):
        return [{"track_time": [0.0, 20.0], "scenes": [{"chain": [{"video_id": vid, "clip_start_sec": 1.0,
                                                                   "clip_duration_sec": 20.0}]}]}]
    timeline = VB.weave(ws, {"A": tl("vid_a"), "B": tl("vid_c")})
    tracks = {"A": "song_full", "B": "song_noanalysis"}
    plan = {"track_id": "song_full", "timeline": timeline, "track_duration_sec": 20.0, "selected_videos": ["vid_a", "vid_c"],
            "verses": {"actors": {"A": {"name": "Ava"}, "B": {"name": "Bea"}}, "windows": ws, "winner": "A",
                       "meter": VB.meter(ws, "A"), "parts": VB.audio_parts(ws, tracks, 20.0, "solo", {t: 40.0 for t in tracks.values()}),
                       "total_seconds": 20.0, "overlay": {"on": True},
                       "intro": {"on": True, "stats": {"A": [["Videos", "2"]], "B": [["Videos", "1"]]}, "thumbs": {}}}}
    out = tmp_path / "battle.mp4"
    RP.render_plan_dict(plan, str(out), quick=True)
    lead = VB.INTRO_SEC
    assert abs(_seconds(out) - (20.0 + lead)) < 0.3
    assert _level(out, 0.3, lead - 0.5) < 0.002                          # the card is silent
    a_win = next(w for w in ws if w["side"] == "A")
    b_win = next(w for w in ws if w["side"] == "B")
    assert _level(out, lead + a_win["start"] + 0.3, lead + a_win["end"] - 0.3) > 0.02     # A's theme (the tone)
    assert _level(out, lead + b_win["start"] + 0.3, lead + b_win["end"] - 0.3) < 0.005    # B's theme (silence)
    r, g, b = _pixel(out, lead + (a_win["start"] + a_win["end"]) / 2, 320, 250)
    assert g > 100 and b < 80                                           # A's footage on screen
    r, g, b = _pixel(out, lead + (b_win["start"] + b_win["end"]) / 2, 320, 250)
    assert b > 100 and g < 80                                           # B's
    split = next(w for w in ws if w["side"] == "split")
    t = lead + (split["start"] + split["end"]) / 2
    assert _pixel(out, t, 60, 200)[1] > 100 and _pixel(out, t, 580, 200)[2] > 100      # A left, B right
    r, g, b = _pixel(out, 1.5, 300, 144)                                 # the card's big VS, gold
    assert r > 180 and g > 140 and b < 120


def test_the_page_compiles_a_battle(monkeypatch):
    out = _timelines(monkeypatch)
    _actors(monkeypatch)
    try:
        import mix_jobs
        import planner.verses as PV
        from tests.apptest_util import errors, page_app
        at = page_app("pages/1_Compilation_Planner.py", timeout=300)
        at.session_state["planner_mode"] = "⚔️ Verses Battle"
        at.run()
        assert not errors(at), errors(at)
        at.button(key="vs_draw_btn").click().run()
        assert not errors(at), errors(at)
        assert {at.session_state["vs_a"], at.session_state["vs_b"]} == {"Ava", "Bea"}
        at.toggle(key="vs_auto_music").set_value(False).run()
        at.selectbox(key="vs_track_a").set_value("song_full").run()
        at.button(key="vs_compile").click().run()
        mix_jobs.wait([PV.JOB_KEY], timeout=300)
        at.run()
        assert not errors(at), errors(at)
        res = at.session_state["vs_result"]
        assert res["plan"]["verses"]["total_seconds"] == 96.0
        at.button(key="vs_back").click().run()
        assert not errors(at), errors(at)
    finally:
        shutil.rmtree(out, ignore_errors=True)
