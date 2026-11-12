"""📐 Section lines straight from the music (section_bounds.py) — sections from scratch, step 1."""
from __future__ import annotations

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)


def _song(n=32):
    """32 two-second bars: voice from bar 4, drums from bar 8, everything louder from 16, the band out at 24."""
    bars = np.arange(n) * 2.0
    drums, voice, loud = np.zeros(n), np.zeros(n), np.full(n, 0.2)
    voice[4:24] = 0.6
    drums[8:24] = 0.8
    loud[8:16], loud[16:24], loud[24:] = 0.5, 0.9, 0.1
    return bars, {"drums": drums, "vocals": voice, "loudness": loud}, n * 2.0


def test_a_line_wherever_a_layer_comes_in_or_the_level_steps():
    import section_bounds as SB
    bars, lv, dur = _song()
    lines = SB.find_lines(bars, lv, dur)
    assert [ln["bar"] for ln in lines] == [4, 8, 16, 24]
    assert [ln["time"] for ln in lines] == [8.0, 16.0, 32.0, 48.0]
    why = {ln["bar"]: ln["why"] for ln in lines}
    assert "🎤 voice comes in" in why[4] and "🥁 drums come in" in why[8]
    assert why[16] == ["🔊 louder"]
    assert {"🥁 drums drop out", "🎤 voice drops out", "🔊 quieter"} <= set(why[24])
    assert lines[1]["level"] == 1 and lines[3]["level"] == 1 and lines[2]["level"] == 2      # big / smaller changes
    secs = SB.sections(bars, lv, dur, lines)
    assert [s["bars"] for s in secs] == [4, 4, 8, 8, 8] and secs[0]["start"] == 0.0 and secs[-1]["end"] == dur
    assert secs[0]["why"] == ["the start of the track"] and secs[3]["playing"]["drums"] == 0.8
    assert all(a["end"] == b["start"] for a, b in zip(secs, secs[1:]))                   # the whole track, in order


def test_small_wobbles_are_not_lines_and_lines_keep_their_distance():
    import section_bounds as SB
    bars, lv, dur = _song()
    rng = np.random.default_rng(3)
    noisy = {k: np.clip(v + rng.normal(0, 0.03, len(v)), 0, 1) for k, v in lv.items()}
    assert [ln["bar"] for ln in SB.find_lines(bars, noisy, dur)] == [4, 8, 16, 24]
    lines = SB.find_lines(bars, noisy, dur)
    gaps = np.diff([ln["bar"] for ln in lines])
    assert (gaps >= SB.MIN_GAP_BARS).all()
    # the band stopping dead for one bar: two big lines one bar apart may both stand
    stop = {k: v.copy() for k, v in lv.items()}
    for k in stop:
        stop[k][20] = 0.0
    got = {ln["bar"]: ln for ln in SB.find_lines(bars, stop, dur)}
    assert got[20]["why"] == ["⏸️ the band stops for a bar"] and got[21]["why"] == ["▶️ the band comes back in"]
    assert got[20]["level"] == 1 and sorted(got) == [4, 8, 16, 20, 21, 24]
    assert SB.find_lines(bars[:3], {k: v[:3] for k, v in lv.items()}, 6.0) == []          # too short to say
    flat = {"loudness": np.full(len(bars), 0.5)}
    assert SB.find_lines(bars, flat, dur) == []                                         # nothing changes → no lines


def test_a_track_from_the_library_and_the_export():
    import json
    import section_bounds as SB
    import section_types as ST
    from tests.fixtures import load_track
    track = load_track(DATA, "song_full")
    found = SB.for_track(track)
    assert {"drums", "hats", "vocals", "loudness", "bright"} <= set(found["levels"])
    assert all(len(v) == len(found["bars"]) for v in found["levels"].values())
    assert found["lines"] and found["sections"][0]["start"] == 0.0
    assert abs(found["sections"][-1]["end"] - float(track["duration_sec"])) < 0.01
    times = [ln["time"] for ln in found["lines"]]
    assert times == sorted(times) and all(t in found["bars"] for t in times)            # every line on a bar line
    assert len(SB.as_regions(found)) == len(found["sections"])
    json.dumps(found)
    plain = {k: v for k, v in track.items() if k != "analysis"}                         # no extra analysis: loudness only
    assert set(SB.for_track(plain)["levels"]) == {"loudness"}
    out = ST.tuning_export(track, ST.suggested_map(track), source="you")
    assert out["lines"]["lines"] and "lines_error" not in out
    hit, used = SB.line_match(found["lines"], times[:2])
    assert hit == 1.0 and 0 < used <= 1.0


def test_the_lines_show_on_the_sections_page():
    import section_bounds as SB
    import sections_view as SV
    from tests.apptest_util import errors, page_app
    from tests.fixtures import load_track
    fig = SV.lines_figure(SB.for_track(load_track(DATA, "song_full")))
    assert fig.data and fig.data[0].type == "heatmap"
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎵 Audio"
    at.session_state["aud_kind"] = "🧩 Sections"
    at.run()
    assert not errors(at), errors(at)
    assert any("lines from the music" in e.label for e in at.expander)
    assert not [sb for sb in at.selectbox if sb.key == "sec_ln_play"]             # replaced by the whole-track player
    assert at.toggle(key="sec_ln_whole").value is False
    assert [sb for sb in at.selectbox if str(sb.key).startswith("sec_role_")], "the editor below is still there"
    assert fig.data[0].showscale and "0.3 playing" in fig.data[0].colorbar.ticktext            # the heatmap's key
    # what the player draws: every layer bar by bar, the lines, the sections — bars numbered from 1 in the page
    found = SB.for_track(load_track(DATA, "song_full"))
    cfg = SV.player_config(found)
    assert len(cfg["z"]) == len(cfg["layers"]) and all(len(r) == len(cfg["bars"]) for r in cfg["z"])
    assert [ln["t"] for ln in cfg["lines"]] == [ln["time"] for ln in found["lines"]]
    assert cfg["sections"][0]["a"] == 0.0 and abs(cfg["sections"][-1]["b"] - cfg["dur"]) < 0.01
    assert all(s_["words"] and s_["why"] for s_ in cfg["sections"])
    import json
    json.dumps(cfg)
    # with the switch on and no audio file on this machine: a plain message, no error
    at.toggle(key="sec_ln_whole").set_value(True).run()
    assert not errors(at), errors(at)
    assert any("isn't on the server yet" in w.value or "Couldn't prepare" in w.value for w in at.warning)


def test_a_profile_per_section():
    """Step 2: what each section IS, in measurements — layers, level within the track, drums, trend, repeats."""
    import section_bounds as SB
    n = 40
    bars = np.arange(n) * 2.0
    drums, voice, bass, loud = np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, 0.15)
    voice[4:12] = 0.6;  loud[4:12] = 0.4                      # 2: voice alone, medium
    loud[12:20] = np.linspace(0.4, 0.85, 8)                   # 3: a build — rising, no line inside it
    drums[20:28] = 0.8; bass[20:28] = 0.7; loud[20:28] = 0.9  # 4: drums + bass, loud
    voice[28:36] = 0.6; loud[28:36] = 0.4                     # 5: the voice alone again → like 2
    lv = {"drums": drums, "vocals": voice, "bass": bass, "loudness": loud}
    hits = np.where(drums > 0, 4.0, 0.0)
    lines = SB.find_lines(bars, lv, n * 2.0)
    secs = SB.profile(SB.sections(bars, lv, n * 2.0, lines), bars, lv, hits)
    by_start = {s["start"]: s for s in secs}
    quiet, verse, hit, again = by_start[0.0], by_start[8.0], by_start[40.0], by_start[56.0]
    assert quiet["profile"]["on"] == [] and quiet["profile"]["level_word"] == "quiet"
    assert verse["profile"]["on"] == ["vocals"] and verse["profile"]["busy"] is None
    assert hit["profile"]["on"] == ["drums", "bass"] and hit["profile"]["level_word"] == "loud"
    assert hit["profile"]["busy"] == 4.0 and "drums (4.0/s)" in hit["words"] and "🥁 drums + 🎸 bass" in hit["words"]
    assert again["profile"]["like"] == verse["label"] and f"like {verse['label']}" in again["words"]
    assert verse["profile"]["like"] is None
    rising = [s for s in secs if s["profile"]["trend"] == "rising"]
    assert rising and all(12 * 2.0 - 4 <= s["start"] < 20 * 2.0 for s in rising) and "↗ rising" in rising[0]["words"]
    assert all(0.0 <= s["profile"]["level"] <= 1.0 for s in secs)
    # a real library track: every section has one, JSON-ready
    import json
    from tests.fixtures import load_track
    found = SB.for_track(load_track(DATA, "song_full"))
    assert all(s["words"] and s["profile"]["trend"] in ("rising", "falling", "steady") for s in found["sections"])
    assert len(found["hits"]) == len(found["bars"])
    json.dumps(found)


def test_the_real_stems_are_used_when_the_laptop_has_measured_them():
    import section_bounds as SB
    from tests.fixtures import load_track
    track = load_track(DATA, "song_full")
    plain = SB.for_track(track)
    assert plain["stems"] == [] and "other" not in plain["levels"]
    n = len(track["analysis"]["kick"])
    half = n // 2
    track["analysis"]["stem_curves"] = {
        "rate": track["analysis"]["rate"], "share": {"drums": 1.0, "bass": 0.02, "other": 0.6},
        "drums": [0.0] * half + [0.9] * (n - half),          # the drum stem only plays in the second half
        "bass": [0.8] * n,                                    # bleed in an empty stem: hardly in the song
        "other": [0.7] * half + [0.0] * (n - half)}
    found = SB.for_track(track)
    assert found["stems"] == ["drums", "bass", "other"] and "other" in found["levels"]
    bars = np.asarray(found["bars"])
    mid = float(track["duration_sec"]) / 2
    d = np.asarray(found["levels"]["drums"])
    assert d[bars < mid - 3].max() < 0.1 and d[bars > mid + 3].min() > 0.8
    assert max(found["levels"]["bass"]) == 0.0
    line = min(found["lines"], key=lambda ln: abs(ln["time"] - mid))
    assert abs(line["time"] - mid) < 3 and "🥁 drums come in" in line["why"] and "🎹 instruments drop out" in line["why"]
    first, last = found["sections"][0]["profile"]["on"], found["sections"][-1]["profile"]["on"]
    assert "other" in first and "drums" not in first and "drums" in last and "bass" not in last
    # the matcher's own reading (audio_drive) is untouched by the stem curves
    import audio_drive as AD
    assert np.allclose(AD.drivers(track)["curve"], AD.drivers(load_track(DATA, "song_full"))["curve"])


def test_sections_parts_and_blocks():
    """A big change starts a section, a smaller one a part of it; the profile sets the block lengths."""
    import section_bounds as SB
    n = 44
    bars = np.arange(n) * 2.0                                   # 120 BPM: a bar is 2 s
    drums, voice, loud = np.zeros(n), np.zeros(n), np.full(n, 0.12)
    voice[6:14] = 0.6;  loud[6:14] = 0.3                         # the voice alone (a smaller change → a part)
    loud[14:22] = np.linspace(0.35, 0.8, 8); voice[14:22] = 0.6  # still the voice, rising
    drums[22:36] = 0.9; loud[22:36] = 0.95                       # the hit: busy loud drums
    lv = {"drums": drums, "vocals": voice, "loudness": loud}
    hits = np.where(drums > 0, 5.0, 0.0)
    lines = SB.find_lines(bars, lv, n * 2.0)
    secs = SB.profile(SB.sections(bars, lv, n * 2.0, lines), bars, lv, hits)
    labels = [s["label"] for s in secs]
    assert labels[0] in ("1", "1a") and len(set(labels)) == len(labels)
    assert all(s["main"] >= 1 for s in secs) and [s["main"] for s in secs] == sorted(s["main"] for s in secs)
    for s_, ln in zip(secs[1:], lines):                          # a solid line: a new number; a dotted one: a letter
        prev = secs[secs.index(s_) - 1]
        assert (s_["main"] == prev["main"] + 1) == (ln["level"] == 1)
    blks = SB.blocks(secs, bars, n * 2.0)
    assert blks[0]["start"] == 0.0 and blks[-1]["end"] == n * 2.0
    assert all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(blks, blks[1:]))        # the whole track, in order
    starts = {round(b["start"], 2) for b in blks}
    assert all(round(ln["time"], 2) in starts for ln in lines)                            # every line is a cut
    assert all(b["end"] - b["start"] >= 2.0 - 1e-6 for b in blks)                         # never under the shortest clip
    assert all(b["bars"] in (0.5, 1, 2, 3, 4, 5, 6, 8) for b in blks)
    in_hit = [b for b in blks if 44.0 <= b["start"] < 72.0]
    assert in_hit and all(b["bars"] == 1 and "loud drums" in b["why"] for b in in_hit)
    quiet = [b for b in blks if b["end"] <= 12.0]
    assert quiet and all(b["bars"] >= 2 for b in quiet) and "hardly anything playing" in quiet[0]["why"]
    rising = [b for b in blks if 28.0 <= b["start"] < 44.0]
    assert rising[0]["bars"] >= rising[-1]["bars"] and "speeding up" in rising[0]["why"]  # shorter towards the hit
    # musical lengths: six bars at a pace of four is 4 + 2; a scrap joins the block before it
    assert SB._fill(6, lambda x: 4.0) == [4.0, 2.0] and SB._fill(5, lambda x: 4.0) == [5.0]
    assert SB._fill(3, lambda x: 2.0) == [2.0, 1.0] and SB._fill(8, lambda x: 2.0) == [2.0] * 4
    # the voice leads and a cut would land inside a sung phrase: it moves to a clean bar line (or is dropped)
    part = [{**secs[1], "bar": 0, "profile": {**secs[1]["profile"], "on": ["vocals"], "trend": "steady",
                                               "level_word": "medium"}}]
    plain = SB.blocks(part, bars[:8], 16.0)
    assert [b["bars"] for b in plain] == [2, 2, 2, 2]
    sung = SB.blocks(part, bars[:8], 16.0, vocal_starts=[2.5], vocal_ends=[9.5])           # one phrase over 4 and 8 s
    cuts = [b["start"] for b in sung[1:]]
    assert 4.0 not in cuts and 8.0 not in cuts and all(not (2.5 + 0.7 < c < 9.5 - 0.7) for c in cuts)
    assert sung[0]["start"] == 0.0 and sung[-1]["end"] == 16.0
    # 🎤 the vocal rule switched off: the cuts stay where the pace put them, sung phrase or not
    off = SB.blocks(part, bars[:8], 16.0, vocal_starts=[2.5], vocal_ends=[9.5], vocal_rule=False)
    assert [(b["start"], b["end"]) for b in off] == [(b["start"], b["end"]) for b in plain] and off != sung
    # slow tempo: a pace that would make a block longer than 8 s is halved; half bars only where they are still a clip
    assert SB._pace({"on": []}, 2.7, 1.4)[0] == 2.0 and SB._pace({"on": []}, 1.9, 1.4)[0] == 4.0
    busy = {"on": ["drums"], "busy_word": "busy", "level_word": "loud"}
    assert SB._pace(busy, 3.0, 1.4)[0] == 0.5 and SB._pace(busy, 2.0, 1.4)[0] == 1.0
    # two loud drum parts are cut alike, however their drums rank (AURA's sections 3 and 5, 12 Nov)
    assert SB._pace(dict(busy, busy_word="steady"), 2.67, 2.67) == SB._pace(busy, 2.67, 2.67)
    assert SB._pace(dict(busy, level_word="medium"), 2.67, 2.67)[0] == 2.0
    assert SB._pace(busy, 2.62, 2.667)[0] == 1.0                  # a bar a touch under four beats is still a bar
    # AURA, 12 Nov: where the bar lines are uneven, "one-bar" blocks came out at 1.3 s and 0.6 / 0.7 s. Blocks are
    # sized in seconds: short ones pair up within their part, the last short one joins the block before it
    ub = np.concatenate([np.arange(8) * 2.6, 20.8 + np.arange(8) * 1.3, [31.2, 31.85, 32.5], 35.1 + np.arange(4) * 2.6])
    part = [{"n": 1, "label": "1", "bar": 0, "profile": dict(busy, trend="steady")}]
    got = SB.blocks(part, ub, 45.5, beat=0.65)                    # 92 BPM: the shortest block is a bar, 2.6 s
    assert all(b["end"] - b["start"] >= 2.6 - 0.06 for b in got), [(b["start"], b["end"]) for b in got]
    assert got[0]["start"] == 0.0 and got[-1]["end"] == 45.5 and [b["n"] for b in got] == list(range(1, len(got) + 1))
    assert [round(b["start"], 1) for b in got if 20.8 <= b["start"] < 31.2] == [20.8, 23.4, 26.0, 28.6]   # paired up
    assert SB._uneven(ub, 45.5) and SB._uneven(np.arange(10) * 2.0, 20.0) == []
    two = [{"n": 1, "label": "1", "bar": 0, "profile": dict(busy, trend="steady")},
           {"n": 2, "label": "2", "bar": 2, "profile": dict(busy, trend="steady")}]
    kept = SB.blocks(two, np.arange(6) * 1.0, 6.0, beat=0.5)       # a line is a cut even after a short last block
    assert 2.0 in [b["start"] for b in kept] and all(b["section"] in ("1", "2") for b in kept)
    # a real library track, and the page's player
    from tests.fixtures import load_track
    import sections_view as SV
    found = SB.for_track(load_track(DATA, "song_full"))
    assert found["blocks"] and found["blocks"][-1]["end"] == found["duration"]
    assert {b["section"] for b in found["blocks"]} == {s["label"] for s in found["sections"]}
    cfg = SV.player_config(found)
    assert len(cfg["blocks"]) == len(found["blocks"]) and cfg["sections"][0]["label"] == found["sections"][0]["label"]
    fig = SV.lines_figure(found)
    assert any(sh.line.color.startswith("rgba(255,255,255") for sh in fig.layout.shapes)   # the block cuts on the chart
    assert list(fig.layout.xaxis.ticktext)[:2] == ["1", "5"]                                # bars numbered on the axis


def test_a_steady_bar_grid_from_the_beats():
    """AURA, 12 Nov: the stored bar lines had half bars and odd ones in places; the beats were steady. Sections,
    parts and blocks sit on one bar line every four beats, beat 1 voted by the stored lines."""
    import section_bounds as SB
    beat = 0.667
    beats = 0.3 + np.arange(150) * beat                          # steady beats, stopping 8 s before the end
    dur = float(beats[-1]) + 8.0
    good = beats[::4]
    down = np.sort(np.concatenate([good, beats[2::4][10:18], [float(beats[101]), float(beats[103])]]))   # half bars, strays
    track = {"beat_times": beats.tolist(), "duration_sec": dur, "analysis": {"downbeats": down.tolist(), "beats_per_bar": 4}}
    stored = np.concatenate([[0.0], down])
    assert SB._uneven(stored, dur)                               # the stored lines are uneven …
    bars = SB.steady_bars(track, stored, dur)
    assert bars[0] == 0.0 and SB._uneven(bars, dur) == []        # … the grid isn't
    d = np.diff(bars)[1:]
    assert np.allclose(d, 4 * beat, atol=0.02)
    assert all(np.min(np.abs(good - t)) < 0.01 for t in bars[1:] if t <= beats[-1])    # on the real bar lines
    assert bars[-1] > beats[-1] and dur - bars[-1] >= 0.25 * 4 * beat                  # carried on past the last beat
    # no usable beats: the stored lines are used as they are
    assert np.array_equal(SB.steady_bars({"beat_times": [0.5, 1.0]}, stored, dur), stored)
    # the matcher's own bar lines (audio_drive) are not touched
    import audio_drive as AD
    from tests.fixtures import load_track
    tr = load_track(DATA, "song_full")
    before = AD.drivers(tr)["bars"].copy()
    SB.for_track(tr)
    assert np.array_equal(AD.drivers(tr)["bars"], before)


def test_old_and_new_cuts_side_by_side():
    """🅰 Old = the module as released in v1.59.0, kept word for word; 🅱 New = today's. The page switches."""
    import section_bounds as SB
    import section_bounds_old as SBO
    import sections_view as SV
    from tests.apptest_util import errors, page_app
    from tests.fixtures import load_track
    tr = load_track(DATA, "song_full")
    old, new = SBO.for_track(tr), SB.for_track(tr)
    assert {"bars", "levels", "lines", "sections", "blocks"} <= set(old) and old["blocks"] and new["blocks"]
    assert all(s_["label"] and s_["words"] for s_ in old["sections"])
    for f in (old, new):                                      # both draw: the chart and the player
        assert SV.lines_figure(f).data and SV.player_config(f)["blocks"]
    # the case the switch is for: uneven stored bar lines over steady beats (AURA, 12 Nov)
    beat = 0.667
    beats = (0.3 + np.arange(160) * beat)
    down = np.sort(np.concatenate([beats[::4], beats[2::4][10:18]]))            # half bars for eight bars
    n = int(107 * 30)
    t = np.arange(n) / 30
    loud = np.where((t > 20) & (t < 80), 0.9, 0.1)
    kick = loud * (0.5 + 0.5 * np.cos(2 * np.pi * ((t - 0.3) % beat) / beat) ** 8)
    track = {"track_id": "uneven", "duration_sec": 107.0, "bpm": 90.0, "beat_times": beats.round(3).tolist(),
             "hires": {"rate": 30, "rms": loud.round(3).tolist(), "onset": kick.round(3).tolist()},
             "analysis": {"rate": 30, "beats_per_bar": 4, "downbeats": down.round(3).tolist(),
                          "kick": kick.round(3).tolist(), "snare": (kick * 0.5).round(3).tolist()}}
    o, w = SBO.for_track(track), SB.for_track(track)
    short = lambda f: min(b["end"] - b["start"] for b in f["blocks"])        # noqa: E731
    assert short(w) >= 2.2 and short(o) <= short(w) + 1e-6    # new: none under a bar; old never longer than new
    assert len(o["bars"]) > len(w["bars"]) and SB._uneven(o["bars"], 107.0) and not w["uneven_bars"]
    # the page: new by default, the switch shows the old ones
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎵 Audio"
    at.session_state["aud_kind"] = "🧩 Sections"
    at.run()
    assert not errors(at), errors(at)
    assert at.session_state["sec_ln_mode"] == "new"
    assert any("🅰 Old" in c.value and "**🅱 New**" in c.value for c in at.caption)
    at.session_state["sec_ln_mode"] = "old"
    at.run()
    assert not errors(at), errors(at)
    assert any("**🅰 Old**" in c.value for c in at.caption)
    # 🎤 the vocal rule: a switch for both modes. On = exactly as before; off = the same in both as a track with
    # no sung-phrase marks (the old module is not edited); the lines and sections never change
    assert at.session_state["sec_ln_vocal"] is True
    on, no = SV._both_cuts(tr), SV._both_cuts(tr, False)
    assert on["old"] == old and on["new"] == new
    bare = {**tr, "analysis": {**(tr.get("analysis") or {}), "vocal_starts": [], "vocal_ends": []}}
    assert no["old"] == SBO.for_track(bare) and no["new"]["blocks"] == SB.for_track(bare)["blocks"]
    assert all(no[m]["lines"] == on[m]["lines"] and no[m]["sections"] == on[m]["sections"] for m in on)
    assert SV._cuts_changed(on["new"], on["new"]) == [] and SV._changed_words([]) == "nothing"
    two = {"blocks": [{"section": "1", "start": 0.0, "end": 4.0}, {"section": "2a", "start": 4.0, "end": 8.0}]}
    one = {"blocks": [{"section": "1", "start": 0.0, "end": 4.0}, {"section": "2a", "start": 4.0, "end": 6.0},
                      {"section": "2a", "start": 6.0, "end": 8.0}]}
    assert SV._cuts_changed(two, one) == ["2a"] and SV._changed_words(["2a"]) == "section 2a"
    # the chart and the player mark where the rule adds (🟢 ＋) or takes away (🟠 ✕) a cut, either way the switch is
    rc = SV._rule_cuts(one, two)
    assert rc == {"added": [6.0], "removed": []} and SV._rule_cuts(two, one) == {"added": [], "removed": [6.0]}
    assert SV._rule_cuts(on["new"], on["new"]) == {"added": [], "removed": []}
    marks = {"added": [new["blocks"][1]["start"] + 0.5], "removed": [new["blocks"][2]["start"] + 0.5]}
    plain_fig, fig = SV.lines_figure(new), SV.lines_figure(new, rule_cuts=marks)
    assert len(fig.layout.shapes) == len(plain_fig.layout.shapes) + 2
    assert {"＋", "✕"} <= {a.text.replace("<b>", "").replace("</b>", "") for a in fig.layout.annotations}
    assert SV.player_config(new, marks)["rule"] == marks
    assert SV.player_config(new)["rule"] == {"added": [], "removed": []}
    at.session_state["sec_ln_vocal"] = False
    at.run()
    assert not errors(at), errors(at)
    assert any("🎤" in c.value for c in at.caption)
