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
    assert at.selectbox(key="sec_ln_play").value == 0
    assert [sb for sb in at.selectbox if str(sb.key).startswith("sec_role_")], "the editor below is still there"


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
    assert again["profile"]["like"] == verse["n"] and f"like {verse['n']}" in again["words"]
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
