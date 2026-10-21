"""⚡ Faster planner matching, ranking and charts — with the same results as before.

  • find_best_shape_window's leaner maths gives the same numbers to the last bit;
  • the shape memo kept between matches gives the same plan, and can't go stale (it's keyed
    on the curve's own numbers);
  • video ranking's in-memory copies follow catalogue changes;
  • audio_charts.add_vrects draws exactly what plotly's add_vrect does."""

import json

import numpy as np

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.fixtures import load_track


def _reference_window(values, window_frames, audio_curve, n_points=40, bias=None):
    """find_best_shape_window as it was before the speed-up (plain ndarray mean / std)."""
    from core.shape import _make_interp_matrix, prepare_audio
    n_positions = len(values) - window_frames + 1
    if n_positions <= 0:
        return -2.0, 0
    windows = np.lib.stride_tricks.sliding_window_view(values, window_frames)
    resampled = windows @ _make_interp_matrix(window_frames, n_points).T
    audio_c, audio_std, _ = prepare_audio(audio_curve, n_points)
    row_mean = resampled.mean(axis=1, keepdims=True)
    row_std = resampled.std(axis=1)
    numerator = (resampled - row_mean) @ audio_c
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where((row_std > 1e-9) & (audio_std > 1e-9), numerator / (n_points * row_std * audio_std), 0.0)
    best = int(np.argmax(scores + bias)) if bias is not None and len(bias) == len(scores) else int(np.argmax(scores))
    return float(scores[best]), best


def test_shape_window_same_to_the_last_bit():
    from core.shape import find_best_shape_window
    rng = np.random.default_rng(7)
    for _ in range(400):
        wf = int(rng.integers(1, 120))
        audio = rng.random(int(rng.integers(1, 200))) * rng.choice([1.0, 0.0])
        v = np.round(rng.random(int(rng.integers(1, 500))) * rng.choice([1.0, 0.01, 0.0]), 4)
        if rng.random() < 0.3:
            v = np.repeat(v, 2)[::2]                       # a strided view
        bias = rng.random(max(len(v) - wf + 1, 1)) if rng.random() < 0.3 else None
        assert find_best_shape_window(v, wf, audio, bias=bias) == _reference_window(v, wf, audio, bias=bias)


def _auto(memo=None, curves_of=None, seq=False):
    import catalogue_store as CStore
    import core.footage as CFOOT
    import core.matching as CMATCH
    import core.recommend as REC
    import rec_library as RL
    import scene_roles as SR
    tr = load_track(DATA, "song_full")
    segs = REC.cut_variant(tr, {"sm_pace": 0})
    vids = ("vid_a", "vid_b", "vid_c", "vid_f")
    q = CFOOT.build_footage_queues(CStore.all_catalogues(), vids, (), {}, roles_of=SR.scene_roles)
    ctx = RL.plan_context(None, shape_memo=memo, curves_of=curves_of)
    tl, _, _ = CMATCH.match_scenes_to_track(
        [dict(s) for s in segs], q, 1, 2, 0.0, 1.0, False, seq, {}, {}, set(), False, 0, 1.0, set(), set(),
        2.0, 0.5, True, 1.5, 2.0, 42, weight_position=1.0 if seq else 0.0, repeat_fallbacks=set(),
        block_roles=[None] * len(segs), track=tr, ctx=ctx)
    return [[(c["video_id"], c["scene_id"], round(float(c["clip_start_sec"]), 6)) for c in e["scenes"]] for e in tl]


def test_shape_memo_between_matches_same_plan_and_never_stale():
    import catalogue_store as CStore
    for seq in (False, True):
        plain = _auto(seq=seq)
        memo = {}
        assert _auto(memo, seq=seq) == plain and memo          # first match fills it…
        assert _auto(memo, seq=seq) == plain                   # …the next reuses it
    # a video's curves change (same lengths): the memo must not answer for the old numbers
    memo = {}
    _auto(memo)

    def louder(vid):
        cur = CStore.motion_curves(vid)
        if vid != "vid_c":
            return cur
        return {sid: {"fps": c["fps"], "values": np.asarray(c["values"])[::-1].copy()} for sid, c in cur.items()}
    assert _auto(memo, curves_of=louder) == _auto(None, curves_of=louder)


def test_ranking_memos_follow_the_catalogue(tmp_path):
    import video_ranking as VR
    cat = {"video_id": "v", "scenes": [{"scene_id": 0, "start_tc": "00:00:00.000", "end_tc": "00:00:08.000",
                                        "tags": ["a"], "motion_intensity": 0.5,
                                        "motion_curve": {"fps": 10, "values": list(np.linspace(0, 1, 80))}}]}
    p = tmp_path / "v.json"
    p.write_text(json.dumps(cat))
    scenes = {0: cat["scenes"][0]}
    f1 = VR.cached_video_features(tmp_path, "v", scenes, None, ())
    assert VR.cached_video_features(tmp_path, "v", scenes, None, ()) is f1          # kept
    cat["scenes"][0]["motion_curve"]["values"] = list(np.linspace(1, 0, 80) ** 3)
    p.write_text(json.dumps(cat))
    import os
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 5))
    f2 = VR.cached_video_features(tmp_path, "v", scenes, None, ())
    assert f2 is not f1 and f2["level"] != f1["level"]                               # follows the change
    assert VR.missing_summaries(tmp_path, ["v"]) == []


def test_add_vrects_matches_plotly():
    import plotly.graph_objects as go
    from audio_charts import add_vrects
    a, b = go.Figure(), go.Figure()
    for f in (a, b):
        f.add_trace(go.Scatter(x=[0, 9], y=[0, 1]))
        f.add_annotation(x=1, y=1, text="first")
    a.add_vrect(x0=1, x1=2, fillcolor="red", line_width=0, layer="below", annotation_text="A",
                annotation_position="top left", annotation_font_size=10, annotation_font_color="blue")
    a.add_vrect(x0=3, x1=4, fillcolor="green", opacity=0.25, line_width=0, layer="below")
    add_vrects(b, [{"x0": 1, "x1": 2, "fillcolor": "red", "label": "A", "label_size": 10, "label_color": "blue"},
                   {"x0": 3, "x1": 4, "fillcolor": "green", "opacity": 0.25}])
    assert a.to_plotly_json()["layout"] == b.to_plotly_json()["layout"]
