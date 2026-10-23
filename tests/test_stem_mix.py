"""🧠 core/stem_mix.py — semi-automatic multi-stem transitions."""

from __future__ import annotations

import numpy as np
import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from core import stem_mix as S

SR = 8000


def _plan(profile="vocal_swap", start=1.0, bars=2, bpm=120.0):
    return S.TransitionPlan(mix_start_point=start, mix_sync_target=0.5, transition_duration_bars=bars,
                            mix_profile=profile, bpm=bpm)


def test_plan_lengths_and_validation():
    p = _plan(bars=8, bpm=120)
    assert p.duration_sec == pytest.approx(16.0)
    assert p.duration_frames(44100) == 16 * 44100
    with pytest.raises(ValueError):
        _plan(profile="nope")
    with pytest.raises(ValueError):
        _plan(bars=0)


def test_progressor_is_frame_accurate_whatever_the_block_size():
    p = _plan(start=1.0, bars=2)                          # 4 s at 120 BPM
    pr = S.Progressor(p, SR)
    for block in (1, 64, 333, 4096):
        ts = np.concatenate([pr.t_block(a, block) for a in range(0, 6 * SR, block)])[:6 * SR]
        start = int(np.argmax(ts > 0))
        assert start == SR + 1                            # the frame after the start frame (t=0 at it)
        assert ts[SR] == 0.0 and ts[SR + pr.length] == 1.0 and ts[SR + pr.length - 1] < 1.0
        assert np.all(np.diff(ts) >= 0)
    assert pr.state(SR - 1) == "idle" and pr.state(SR) == "running" and pr.state(SR + pr.length) == "done"
    assert S.Progressor.crossfader(0.0) == -1.0 and S.Progressor.crossfader(1.0) == 1.0


@pytest.mark.parametrize("profile", list(S.PROFILES))
def test_stem_curves_start_all_a_and_end_all_b(profile):
    t = np.linspace(0, 1, 2001)
    g = S.stem_gains(t, profile)
    for stem, (out, inn) in g.items():
        assert out[0] == pytest.approx(1.0, abs=1e-9) and inn[0] == pytest.approx(0.0, abs=1e-9), stem
        assert out[-1] == pytest.approx(0.0, abs=1e-9) and inn[-1] == pytest.approx(1.0, abs=1e-9), stem
        assert np.all(np.diff(inn) >= -1e-12) and np.all(np.diff(out) <= 1e-12), stem


def test_each_stem_has_its_own_shape():
    t = np.linspace(0, 1, 1001)
    g = S.stem_gains(t, "vocal_swap")
    v_out, v_in = g["vocals"]
    # Vocals: a sharp swap — both decks' vocals clearly audible together only near the middle.
    both = t[(v_out > 0.2) & (v_in > 0.2)]
    assert both.min() > 0.4 and both.max() < 0.6
    assert v_in[500] == pytest.approx(0.5, abs=0.01)
    # Drums: the outgoing ones fade faster than the incoming ones rise.
    d_out, d_in = g["drums"]
    assert np.all(d_out[1:-1] < 1 - d_in[1:-1])
    # Other: constant power.
    o_out, o_in = g["other"]
    assert np.allclose(o_out ** 2 + o_in ** 2, 1.0)


def test_profile_timing():
    t = np.linspace(0, 1, 1001)
    df = S.stem_gains(t, "drums_first")
    assert df["drums"][1][500] == pytest.approx(1.0)              # B's drums fully in by halfway
    assert df["vocals"][1][300] == 0.0                            # vocals not moving yet
    bd = S.stem_gains(t, "bass_drop")
    assert bd["drums"][0][900] == 1.0 and bd["drums"][1][900] == 0.0   # A's drums held to the drop
    assert bd["other"][1][870] == 0.0


def test_soft_knee():
    x = np.linspace(-3, 3, 6001)
    y = S.soft_knee(x)
    assert np.all(np.abs(y) < 1.0)
    below = np.abs(x) <= S.SOFT_KNEE
    assert np.array_equal(y[below], x[below])                     # untouched under the knee
    assert np.all(np.diff(y) > 0)                                 # still rises — no flat-topped clipping
    dy = np.diff(y) / np.diff(x)
    k = int(np.argmin(np.abs(x - S.SOFT_KNEE)))
    assert dy[k - 1] == pytest.approx(1.0, abs=0.01) and dy[k + 1] == pytest.approx(1.0, abs=0.01)   # no corner


def test_one_pole_smooths_a_step_across_blocks():
    sm = S.OnePole(SR, ms=4.0, start=1.0)
    x = np.r_[np.ones(100), np.zeros(400)]
    y = np.concatenate([sm(x[i:i + 37]) for i in range(0, len(x), 37)])
    assert y[99] == pytest.approx(1.0)
    assert np.max(np.abs(np.diff(y))) < 0.05                      # no jump at the step
    assert y[-1] < 0.01


def _decks(seconds=8.0, ch=2):
    rng = np.random.default_rng(0)
    n = int(SR * seconds)
    return {d: {s: (0.3 * rng.standard_normal((n, ch))).astype(np.float32) for s in S.STEMS} for d in "AB"}


def test_mixer_output_is_the_same_whatever_the_block_size():
    decks, p = _decks(), _plan()
    outs = []
    for block in (128, 1000):
        m = S.StemMixer(decks, p, SR)
        outs.append(np.concatenate([m.process(block) for _ in range(int(7 * SR / block) + 1)])[:7 * SR])
    assert np.allclose(outs[0], outs[1], atol=1e-5)


def test_mixer_plays_a_then_b_from_its_sync_point():
    decks, p = _decks(), _plan(start=1.0, bars=2)
    m = S.StemMixer(decks, p, SR)
    y = np.concatenate([m.process(500) for _ in range(int(7 * SR / 500))])
    a_sum = sum(decks["A"][s] for s in S.STEMS)
    b_sum = sum(decks["B"][s] for s in S.STEMS)
    i = SR // 2                                                     # before the transition: deck A
    assert np.allclose(y[i:i + 100], S.soft_knee(a_sum[i:i + 100]), atol=1e-5)
    j = int(6 * SR)                                                 # after it: deck B, sync 0.5 s at A's 1.0 s
    bj = j + int(0.5 * SR) - SR
    assert np.allclose(y[j:j + 100], S.soft_knee(b_sum[bj:bj + 100]), atol=1e-5)
    assert m.state == "done"


def test_plan_transition_from_catalogue_tracks():
    from tests.fixtures import load_track
    a, b = load_track(DATA, "song_full"), load_track(DATA, "song_slow")
    p = S.plan_transition(a, b, bars=8, profile="drums_first")
    from core.mixing import grid
    ga, gb = grid(a), grid(b)
    assert p.mix_start_point + p.duration_sec <= ga["duration"] + 1e-6
    assert min(abs(p.mix_start_point - x) for x in ga["bars"]) < 0.01 or p.mix_start_point == 0.0
    assert p.mix_sync_target == pytest.approx(float(gb["bars"][0]))
    assert p.notes and p.mix_profile == "drums_first"
    with pytest.raises(ValueError):
        S.plan_transition(load_track(DATA, "song_short"), b, bars=64)


def test_demo_lifecycle():
    r = S.demo("bass_drop", block=1024, sr=SR, verbose=False)
    states = [s for _, s in r["log"]]
    assert states == ["idle", "running", "done"]
    assert r["peak_after"] < 1.0
