"""pack_for_modal.py (laptop, Phase 1 of the ML redesign): trial picks, manifests, the pack summary."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)

import pack_for_modal as PK


def _make_proxy(path, seconds, rate=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=320x180:rate={rate}:duration={seconds:.2f}", "-pix_fmt", "yuv420p", str(path)],
                   check=True)


@pytest.fixture
def proxies(monkeypatch, tmp_path):
    """Preview copies for vid_a, vid_b, vid_c in a private folder (matching their scene lengths)."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    cats = PK.load_catalogues()
    folder = tmp_path / "proxies"
    for v in ("vid_a", "vid_b", "vid_c"):
        _make_proxy(folder / f"{v}.mp4", PK.catalogue_duration(cats[v]))
    monkeypatch.setattr(PK, "PROXY_DIR", folder)
    return folder


def test_pick_trial_labelled_first_then_spread():
    cats = PK.load_catalogues()
    every = lambda v: True
    picks = PK.pick_trial(cats, 4, {"vid_d": 5, "vid_b": 2}, {}, every)
    assert picks[:2] == ["vid_d", "vid_b"] and len(picks) == 4 and len(set(picks)) == 4
    # Spread over lengths: shortest and longest of the rest are both in.
    rest = sorted((v for v in cats if v not in ("vid_d", "vid_b")), key=lambda v: PK.catalogue_duration(cats[v]))
    assert rest[0] in picks and rest[-1] in picks
    # Only videos with a preview copy.
    assert sorted(PK.pick_trial(cats, 10, {}, {}, lambda v: v in ("vid_a", "vid_c"))) == ["vid_a", "vid_c"]


def test_pick_trial_spreads_actors():
    cats = PK.load_catalogues()
    by_len = sorted(cats, key=lambda v: (PK.catalogue_duration(cats[v]), v))
    meta = {v: {"actors": ["Same"]} for v in cats}
    meta[by_len[-2]] = {"actors": ["Other"]}
    picks = PK.pick_trial(cats, 2, {}, meta, lambda v: True)
    # Shortest, then the longest swapped for its neighbour with an actor not seen yet.
    assert picks == [by_len[0], by_len[-2]]


def test_pack_trial(proxies, tmp_path):
    out = tmp_path / "pack"
    assert PK.main(["--trial", "3", "--out", str(out)]) == 0
    summary = json.loads((out / "pack.json").read_text())
    assert sorted(summary["videos"]) == ["vid_a", "vid_b", "vid_c"]
    assert summary["hours"] > 0 and "labels.json" in summary["library_files"]
    assert summary["library_videos"] == 6 and summary["library_hours"] >= summary["hours"]
    assert set(summary["no_preview_copy"]) == {"vid_d", "vid_e", "vid_f"}
    m = json.loads((out / "manifests" / "vid_a.json").read_text())
    cat = PK.load_catalogues()["vid_a"]
    assert len(m["scenes"]) == len(cat["scenes"])
    assert m["scenes"][1]["start"] == pytest.approx(PK.tc_to_seconds(cat["scenes"][1]["start_tc"]))
    assert m["scenes"][0]["motion_curve"]["values"] == cat["scenes"][0]["motion_curve"]["values"]
    assert m["proxy"]["fps"] == 10 and not m["proxy"]["variable_fps"]
    assert all(not r["problems"] for r in summary["this_run"])
    # Re-running copies nothing new.
    assert PK.main(["--only", "vid_a", "--out", str(out)]) == 0
    assert "0 preview copies copied" in (out / "pack_report.txt").read_text()


def test_pack_flags_a_short_preview(proxies, tmp_path):
    _make_proxy(proxies / "vid_b.mp4", 3)                     # far shorter than its scenes
    out = tmp_path / "pack"
    assert PK.main(["--only", "vid_b", "--out", str(out)]) == 0
    row = json.loads((out / "pack.json").read_text())["this_run"][0]
    assert any("preview is" in p for p in row["problems"])


def test_dry_run_copies_nothing(proxies, tmp_path):
    out = tmp_path / "pack"
    assert PK.main(["--dry-run", "--out", str(out)]) == 0
    assert not out.exists()


def test_newest_copy_takes_drives_duplicate(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(PK, "CATALOGUE_DIR", tmp_path)
    assert PK.newest_copy("tag_vocabulary.csv") is None
    a, b = tmp_path / "tag_vocabulary.csv", tmp_path / "tag_vocabulary (1).csv"
    other = tmp_path / "tag_vocabulary_old.csv"
    for i, p in enumerate((a, b, other)):
        p.write_text("tag\nx\n")
        os.utime(p, (1000 + i, 1000 + i))
    assert PK.newest_copy("tag_vocabulary.csv") == b          # newer duplicate wins, unrelated names ignored
    os.utime(a, (5000, 5000))
    assert PK.newest_copy("tag_vocabulary.csv") == a
