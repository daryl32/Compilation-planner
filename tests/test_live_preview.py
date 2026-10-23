"""⚡ live_preview: tempo-matched stem windows for the browser's live transition preview."""

import time

import numpy as np
import pytest

from tests.support import DATA  # noqa: F401
from tests.test_mixing import mixlib  # noqa: F401
from core import mixing as MX


def test_prepare_windows_files_and_reuse(mixlib):
    import live_preview as LP
    t = mixlib["tracks"]
    ta, tb = t["mix_song_c"], t["mix_song_a"]
    ga, gb = MX.grid(ta), MX.grid(tb)
    s = {"a_end": float(ga["bars"][20]), "b_full": float(gb["bars"][10]), "bars": 4}
    t0 = time.time()
    p = LP.prepare("mix_song_c", "mix_song_a", ta, tb, s)
    took = time.time() - t0
    assert p["mode"] == "stems" and set(p["a"]["files"]) == {"vocals", "drums", "bass", "other"}
    for d in ("a", "b"):
        w = p[d]["w"]
        assert w[0] <= (s["a_end"] if d == "a" else s["b_full"]) <= w[1]
        for f in p[d]["files"].values():
            assert f.endswith(".flac")
    import mixer
    y = mixer.decode(p["b"]["files"]["other"])
    expected = (p["b"]["w"][1] - p["b"]["w"][0]) / p["speed_b"]
    assert len(y) / mixer.SR == pytest.approx(expected, abs=0.25)       # stretched to A's tempo
    # a drag of a bar stays inside the window: nothing new to prepare
    s2 = dict(s, b_full=float(gb["bars"][11]))
    assert LP.still_fits(p, "mix_song_c", "mix_song_a", ta, tb, s2)
    t1 = time.time()
    p2 = LP.prepare("mix_song_c", "mix_song_a", ta, tb, s2, have=p)
    assert p2["b"]["files"] == p["b"]["files"] and time.time() - t1 < took + 1
    print(f"prepare {took:.2f}s, again {time.time() - t1:.2f}s")


def test_curve_warp_and_stem_gains_parity_numbers():
    from core import stem_mix as SM
    t = np.linspace(0, 1, 11)
    g = SM.stem_gains(t, "drums_first", 16, {"out": {"mid": 0.3, "steep": 2}, "in": {"mid": 0.7, "steep": 0.5}})
    assert g["other"][0][0] == pytest.approx(1) and g["other"][1][-1] == pytest.approx(1)
    assert SM.norm_curves({"out": {"mid": 2, "steep": 99}})["out"] == {"mid": 0.95, "steep": 4.0}
