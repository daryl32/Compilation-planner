"""⚡ Quick preview (videos read from their 480p preview copies — nothing fetched) and segments rendered several at
a time, in the right order. Real ffmpeg on tiny synthetic files; skipped where ffmpeg isn't installed."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _make(path, colour, seconds=6):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=320x180:d={seconds}:r=25",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True)


def _colour_at(path, t):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    r, g, b = out.stdout[:3]
    return "red" if r > 150 and b < 100 else "blue" if b > 150 and r < 100 else "other"


def test_quick_preview_reads_the_480p_copies_and_keeps_the_order(tmp_path, monkeypatch):
    import catalogue_store as CS
    import render_preview as RP
    from tests.fixtures import load_track
    proxies, drive = tmp_path / "proxies", tmp_path / "drive"
    proxies.mkdir()
    _make(proxies / "vid_a.mp4", "red")
    _make(proxies / "vid_b.mp4", "blue")
    track = load_track(DATA, "song_full")
    # the track's audio "is on this machine"; the videos' originals are not, and there is no Drive here
    real = RP.colab_to_local
    monkeypatch.setattr(RP, "colab_to_local", lambda p: str(drive / p.split("MyDrive/", 1)[1]) if "MyDrive/" in p else real(p))
    audio = drive / RP.get_track_raw_source(track["track_id"]).split("MyDrive/", 1)[1]
    audio.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    str(audio.with_suffix(".wav"))], check=True)
    audio.with_suffix(".wav").rename(audio)
    monkeypatch.setattr(RP, "PROXY_DIR", proxies)
    monkeypatch.setattr(RP, "streaming_available", lambda: False)
    assert set(CS.video_index()) >= {"vid_a", "vid_b"}
    order = ["vid_a", "vid_b", "vid_b", "vid_a", "vid_b", "vid_a", "vid_a", "vid_b"]
    plan = {"track_id": track["track_id"], "selected_videos": ["vid_a", "vid_b"], "timeline": [
        {"track_time": [k * 1.0, k * 1.0 + 1.0],
         "scenes": [{"chain": [{"video_id": v, "scene_id": 0, "clip_start_sec": 1.0 + 0.3 * k, "clip_duration_sec": 1.0}]}]}
        for k, v in enumerate(order)]}
    out = tmp_path / "out.mp4"
    seen, notes = [], []
    stats = RP.render_plan_dict(plan, str(out), progress_callback=lambda d, t: seen.append((d, t)), stream=True,
                                notice=notes.append, quick=True, workers=4)
    assert stats["quick"] == 2 and stats["streamed"] == 0 and stats["downloaded"] == 0 and stats["local"] == 1
    assert seen == [(k + 1, 8) for k in range(8)]            # progress counted up from the caller's thread
    assert [_colour_at(out, k + 0.5) for k in range(8)] == ["red" if v == "vid_a" else "blue" for v in order]
    # without Quick preview the originals are needed — and they aren't here
    with pytest.raises(FileNotFoundError):
        RP.render_plan_dict(plan, str(tmp_path / "out2.mp4"), stream=True, quick=False)
    # one after another gives the same picture
    RP.render_plan_dict(plan, str(tmp_path / "out3.mp4"), quick=True, workers=1)
    assert [_colour_at(tmp_path / "out3.mp4", k + 0.5) for k in range(8)] == [_colour_at(out, k + 0.5) for k in range(8)]
