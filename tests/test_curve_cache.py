"""curve_cache gives exactly the curves the catalogue JSON holds, and stays
in step with the catalogue."""

import json
import os
import time

import numpy as np
import pytest

import tests.support  # noqa: F401
import curve_cache as CC
from tests.fixtures import make_video


def _write(path, cat):
    path.write_text(json.dumps(cat))
    t = time.time() + 1           # make sure the modified time moves on
    os.utime(path, (t, t))


def test_same_values_as_json(tmp_path):
    cat = make_video(21, "v1", 9)
    f = tmp_path / "v1.json"
    _write(f, cat)
    got = CC.load(f, tmp_path / "cache")
    want = {s["scene_id"]: s["motion_curve"] for s in cat["scenes"]}
    assert set(got) == set(want)
    for sid, c in want.items():
        assert got[sid]["fps"] == c["fps"]
        assert np.array_equal(got[sid]["values"], np.asarray(c["values"], dtype=float))
        assert got[sid]["values"].dtype == np.float64


def test_second_load_uses_cache_and_rebuilds_on_change(tmp_path):
    cat = make_video(22, "v2", 5)
    f = tmp_path / "v2.json"
    _write(f, cat)
    CC.load(f, tmp_path / "c")
    data_name = lambda: json.loads((tmp_path / "c" / "v2.idx.json").read_text())["data"]  # noqa: E731
    npy = tmp_path / "c" / data_name()
    m1 = npy.stat().st_mtime_ns
    CC.load(f, tmp_path / "c")
    assert npy.stat().st_mtime_ns == m1                       # reused
    cat["scenes"][0]["motion_curve"]["values"] = [9.0, 8.0, 7.0]
    _write(f, cat)
    got = CC.load(f, tmp_path / "c")
    assert list(got[0]["values"]) == [9.0, 8.0, 7.0]           # rebuilt
    assert data_name() != npy.name and not npy.exists()       # new data file, old one removed
    assert len(list((tmp_path / "c").glob("*.npy"))) == 1


@pytest.mark.parametrize("damage", ["garbage", "bad_scenes", "out_of_range", "missing_data"])
def test_damaged_cache_rebuilds(tmp_path, damage):
    cat = make_video(26, "v6", 4)
    f = tmp_path / "v6.json"
    _write(f, cat)
    CC.load(f, tmp_path / "c")
    idx = tmp_path / "c" / "v6.idx.json"
    meta = json.loads(idx.read_text())
    if damage == "garbage":
        idx.write_text("{not json")
    elif damage == "bad_scenes":
        meta["scenes"] = [[0, 10]]
        idx.write_text(json.dumps(meta))
    elif damage == "out_of_range":
        meta["scenes"][0][2] = 10 ** 9
        idx.write_text(json.dumps(meta))
    else:
        (tmp_path / "c" / meta["data"]).unlink()
    got = CC.load(f, tmp_path / "c")
    assert np.array_equal(got[0]["values"], np.asarray(cat["scenes"][0]["motion_curve"]["values"]))


def test_scene_ids_keep_their_type_and_missing_curves(tmp_path):
    cat = make_video(23, "v3", 4)
    cat["scenes"][1]["scene_id"] = "1b"
    cat["scenes"][2].pop("motion_curve")
    cat["scenes"][3]["motion_curve"]["values"] = []
    f = tmp_path / "v3.json"
    _write(f, cat)
    got = CC.load(f, tmp_path / "c")
    assert set(got) == {0, "1b", 3}
    assert len(got[3]["values"]) == 0


def test_read_only(tmp_path):
    f = tmp_path / "v4.json"
    _write(f, make_video(24, "v4", 3))
    vals = CC.load(f, tmp_path / "c")[0]["values"]
    with pytest.raises((ValueError, TypeError)):
        vals[0] = 1.0


def test_unwritable_cache_falls_back_to_json(tmp_path):
    f = tmp_path / "v5.json"
    cat = make_video(25, "v5", 3)
    _write(f, cat)
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")                                     # a file where the cache folder should be
    got = CC.load(f, blocker / "cache")
    assert np.array_equal(got[0]["values"], np.asarray(cat["scenes"][0]["motion_curve"]["values"]))


def test_missing_catalogue(tmp_path):
    assert CC.load(tmp_path / "nope.json", tmp_path / "c") == {}


def test_much_smaller_than_python_lists(tmp_path):
    """The point of the cache: an hour of curves at 10 fps."""
    import sys
    vals = [round(float(x), 4) for x in np.random.default_rng(0).random(36_000)]
    as_list = sys.getsizeof(vals) + sum(sys.getsizeof(v) for v in vals)
    as_array = np.asarray(vals, dtype=float).nbytes
    assert as_array * 3 < as_list
