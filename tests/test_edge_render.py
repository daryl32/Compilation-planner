"""🎬 Edge Mode in the rendered video: the timer and its signals, the avatar, the intro / outro, the sound (real ffmpeg on tiny files)."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.edge_support import (DATA, VIDS, E_CARD_LEAD, _check_plan, _emap, _level, _pixel, _seconds,  # noqa: F401
                                _timelines, _wait_job)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_render_carries_the_timer_and_the_mixed_sound(tmp_path, monkeypatch):
    import edge_mode as E
    import render_preview as RP
    from tests.fixtures import load_track
    proxies, drive = tmp_path / "proxies", tmp_path / "drive"
    proxies.mkdir()
    for vid, colour in (("vid_a", "0x003000"), ("vid_b", "0x000030")):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=320x180:d=30:r=25",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(proxies / f"{vid}.mp4")], check=True)
    real = RP.colab_to_local
    monkeypatch.setattr(RP, "colab_to_local", lambda p: str(drive / p.split("MyDrive/", 1)[1]) if "MyDrive/" in p else real(p))
    monkeypatch.setattr(RP, "PROXY_DIR", proxies)
    monkeypatch.setattr(RP, "streaming_available", lambda: False)
    track = load_track(DATA, "song_full")
    audio = drive / RP.get_track_raw_source(track["track_id"]).split("MyDrive/", 1)[1]
    audio.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=25",
                    "-af", "volume=4", str(audio.with_suffix(".wav"))], check=True)
    audio.with_suffix(".wav").rename(audio)
    sched = [{"index": 0, "round": 1, "profile": "wait", "start": 0.0, "end": 20.0},
             {"index": 1, "round": 1, "profile": "extra", "start": 20.0, "end": 40.0}]
    timeline = [{"track_time": [k * 5.0, k * 5.0 + 5.0],
                 "scenes": [{"chain": [{"video_id": "vid_a" if k % 2 == 0 else "vid_b", "scene_id": 0,
                                        "clip_start_sec": 1.0 + k, "clip_duration_sec": 5.0}]}]} for k in range(8)]

    def plan(playlist, beat, voice, corner="top-left"):
        return {"track_id": "edge_test", "selected_videos": ["vid_a", "vid_b"], "timeline": timeline,
                "track_duration_sec": 40.0,
                "edge": {"difficulty": "chill", "rounds": 1, "schedule": sched, "playlist": playlist,
                         "beat": {"on": beat, "volume": 1.0},
                         "voiceover": {"on": voice, "music_volume": 0.25,
                                       "cues": [{"time": ph["start"], "profile": ph["profile"], "file": None} for ph in sched] if voice else []},
                         "overlay": {"corner": corner}}}
    # no song: the beat is the sound — silent in Wait, beating in Extra; the timer sits top left
    out = tmp_path / "beat.mp4"
    RP.render_plan_dict(plan([], True, False), str(out), quick=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True)
    assert abs(float(probe.stdout) - 40.0) < 0.3
    assert _level(out, 2, 18) < 0.002 and _level(out, 22, 38) > 0.02
    w, h = E.overlay_size(360)
    cx, top = int(360 * 0.035) + w // 2, int(360 * 0.035)
    assert _pixel(out, 1.0, cx, top + 4)[0] > 150                      # the amber ring, full at the start…
    assert _pixel(out, 18.0, cx - 20, top + 5)[0] < 130                # …and nearly empty at the phase's end
    label_y = top + h - int(w * 0.15)
    r, g, b = _pixel(out, 30.0, cx - int(w * 0.3), label_y)            # Extra's red label
    assert r > 170 and g < 90 and b < 90
    r, g, b = _pixel(out, 5.0, cx - int(w * 0.3), label_y)             # Wait's blue one
    assert b > 140 and r < 120
    assert max(_pixel(out, 10.0, 600, 300)) < 80                       # the footage is untouched away from the timer
    # a song, started again when it ends, at a quarter volume under the voiceover; the timer where it was asked for
    parts = [{"track_id": "song_full", "offset": 0.0, "play": 25.0, "pass": 1},
             {"track_id": "song_full", "offset": 25.0, "play": 15.0, "pass": 2}]
    full, half = tmp_path / "song.mp4", tmp_path / "song_voice.mp4"
    RP.render_plan_dict(plan(parts, False, False, "bottom-right"), str(full), quick=True)
    RP.render_plan_dict(plan(parts, False, True, "bottom-right"), str(half), quick=True)
    assert _level(full, 30, 38) > 0.1                                   # the song plays on after its first end
    assert _level(half, 25, 35) == pytest.approx(0.25 * _level(full, 25, 35), rel=0.15)       # a quarter under the voiceover…
    assert _level(full, 5, 15) == pytest.approx(0.25 * _level(full, 25, 35), rel=0.15)        # …and a quarter during Wait
    assert _level(half, 5, 15) == pytest.approx(_level(full, 5, 15), rel=0.15)                # (a quarter either way)
    assert _level(full, 20.6, 21.0) > 3 * _level(full, 19.0, 19.4)                            # back up as Extra starts
    assert _level(half, 0.05, 0.3) > _level(half, 5, 5.3) * 1.02        # the cue (a placeholder tone) at the phase start
    assert max(_pixel(full, 10.0, cx, top + 4)) < 80                    # nothing top left now…
    r, g, b = _pixel(full, 30.0, 640 - top - w + int(w * 0.2), 360 - top - int(w * 0.15) - 1)
    assert r > 170 and g < 90 and b < 90                                # …Extra's label sits bottom right


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_avatar_pops_up_and_the_intro_screen_comes_first(tmp_path, monkeypatch):
    """👩 Her clip (picture + voice) in a corner box or as the right half of a split screen, the stand-in figure
    where there is no clip yet, and the 🎬 intro screen (as long as her intro clip) before the compilation."""
    import edge_mode as E
    import render_preview as RP
    proxies = tmp_path / "proxies"
    proxies.mkdir()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x003000:s=320x180:d=30:r=25",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(proxies / "vid_a.mp4")], check=True)
    monkeypatch.setattr(RP, "PROXY_DIR", proxies)
    monkeypatch.setattr(RP, "streaming_available", lambda: False)

    def clip(name, colour, seconds, sound=True):                  # an "avatar" clip: a plain colour, portrait, a tone for her voice
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=180x320:d={seconds}:r=25"]
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}", "-af", "volume=6"] if sound else []
        subprocess.run(cmd + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-shortest",
                              str(tmp_path / name)], check=True)
        return str(tmp_path / name)
    talk, hello = clip("talk.mp4", "yellow", 4), clip("intro.mp4", "magenta", 5)
    sched = [{"index": 0, "round": 1, "profile": "wait", "start": 0.0, "end": 20.0},
             {"index": 1, "round": 1, "profile": "extra", "start": 20.0, "end": 40.0}]
    timeline = [{"track_time": [k * 5.0, k * 5.0 + 5.0],
                 "scenes": [{"chain": [{"video_id": "vid_a", "scene_id": 0, "clip_start_sec": 1.0 + k,
                                        "clip_duration_sec": 5.0}]}]} for k in range(8)]

    def plan(mode, intro):
        return {"track_id": "edge_test", "selected_videos": ["vid_a"], "timeline": timeline, "track_duration_sec": 40.0,
                "edge": {"difficulty": "chill", "rounds": 1, "schedule": sched, "playlist": [],
                         "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
                         "overlay": {"corner": "top-left"},
                         "avatar": {"on": True, "mode": mode, "corner": "bottom-right",
                                    "cues": [{"time": 0.0, "profile": "wait", "clip": talk},
                                             {"time": 20.0, "profile": "extra", "clip": None}]},
                         "intro": intro}}
    yellow = lambda p: p[0] > 170 and p[1] > 170 and p[2] < 90          # noqa: E731
    green = lambda p: max(p) < 90 and p[1] > p[0] + 20 and p[1] > p[2] + 20                                       # noqa: E731
    # in a corner: her box bottom right while her clip runs, the footage back after it; her voice is heard
    out = tmp_path / "corner.mp4"
    RP.render_plan_dict(plan("corner", {"on": False}), str(out), quick=True)
    assert abs(_seconds(out) - 40.0) < 0.3
    bh = int(360 * E.AVATAR_CORNER_SHARE)
    spot = (640 - int(360 * 0.035) - 30, 360 - int(360 * 0.035) - bh // 2)
    assert yellow(_pixel(out, 2.0, *spot)) and green(_pixel(out, 6.0, *spot))
    assert green(_pixel(out, 2.0, 320, 100))                             # the middle of the frame is still footage
    assert _level(out, 1, 3.5) > 0.05 > _level(out, 6, 18)               # her tone during the clip, quiet Wait after
    assert not green(_pixel(out, 21.0, *spot)) and green(_pixel(out, 25.0, *spot))      # the stand-in figure, 3 s
    # split screen: she is the right half, the footage keeps the left
    out = tmp_path / "split.mp4"
    RP.render_plan_dict(plan("split", {"on": False}), str(out), quick=True)
    assert yellow(_pixel(out, 2.0, 480, 300)) and yellow(_pixel(out, 2.0, 330, 40)) and green(_pixel(out, 2.0, 300, 300))
    assert green(_pixel(out, 6.0, 480, 300))
    assert _pixel(out, 1.0, int(360 * 0.035) + E.overlay_size(360)[0] // 2, int(360 * 0.035) + 4)[0] > 150      # the timer stays on top
    # the intro screen: as long as her intro clip, she on the left, then the compilation with its timer from 0
    intro = {"on": True, "clip": hello, "stats": [["Difficulty", "Chill"], ["Rounds", "1"], ["Time", "0:40"]], "thumbs": []}
    out = tmp_path / "intro.mp4"
    RP.render_plan_dict(plan("corner", intro), str(out), quick=True)
    assert abs(_seconds(out) - 45.0) < 0.3
    ax, ay, aw, ah = E.intro_layout((640, 360))["avatar"]
    r, g, b = _pixel(out, 2.0, ax + aw // 2, ay + ah // 2)
    assert r > 170 and b > 170 and g < 90                                # her (magenta) intro clip in her box
    assert max(_pixel(out, 2.0, 630, 350)) < 60 and _level(out, 1, 4) > 0.05           # the dark card, her voice
    assert yellow(_pixel(out, 7.0, *spot)) and _pixel(out, 6.0, int(360 * 0.035) + E.overlay_size(360)[0] // 2, int(360 * 0.035) + 4)[0] > 150
    # no intro clip yet: the stand-in for 8 s
    out = tmp_path / "intro_stand_in.mp4"
    RP.render_plan_dict(plan("corner", {**intro, "clip": None}), str(out), quick=True)
    assert abs(_seconds(out) - 48.0) < 0.3 and _level(out, 1, 7) < 0.002


def test_the_intro_card_takes_thumbnails_as_pictures_or_bytes():
    """The planner hands thumbnails over as JPEG bytes — the intro screen failed on them ("bytes object has no
    attribute convert", 14 Nov)."""
    import io
    import edge_mode as E
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (160, 90), (200, 30, 30)).save(buf, format="JPEG")
    img = E.intro_card((640, 360), [("Rounds", "3")], [buf.getvalue(), Image.new("RGB", (160, 90), (30, 30, 200)), None, b"not a picture"])
    px = np.asarray(img)
    assert (px[..., 0] > 150).sum() > 2000 and (px[..., 2] > 150).sum() > 2000          # both thumbnails are drawn


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_stuck_picture_pass_is_stopped_and_says_so(tmp_path):
    import os
    import time
    import edge_render as R
    fifo = tmp_path / "never.wav"                         # an input nobody ever writes to: ffmpeg waits on it for ever
    os.mkfifo(fifo)
    keep = os.open(fifo, os.O_RDWR)                       # (held open, so ffmpeg's own open returns and its read hangs)
    try:
        t0 = time.time()
        with pytest.raises(RuntimeError, match="stood still"):
            R.run_watched(["ffmpeg", "-y", "-f", "wav", "-i", str(fifo), str(tmp_path / "out.wav")], 10.0, stall=3)
        assert time.time() - t0 < 30
    finally:
        os.close(keep)
    seen = []                                              # a pass that works reports how far it is, up to the end
    R.run_watched(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=60", "-c:v", "libx264", "-preset",
                   "ultrafast", str(tmp_path / "ok.mp4")], 60.0, progress=seen.append)
    assert all(0 < f <= 1 for f in seen) and seen == sorted(seen)
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        R.run_watched(["ffmpeg", "-y", "-i", str(tmp_path / "not_there.mp4"), str(tmp_path / "x.mp4")], 5.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_songs_are_read_from_this_machine_never_streamed_and_sit_at_a_quarter_under_the_voice(tmp_path, monkeypatch):
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: f"raw/{t}"))
    song = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=30", "-af", "volume=4",
                    str(song)], check=True)

    class Sources:                                         # a song that isn't here yet: only a stream on offer
        here, downloaded = False, []

        def is_local(self, raw):
            return self.here

        def download(self, raws):
            self.downloaded += list(raws)
            self.here = True
            return []

        def input_args(self, raw):
            return ["-i", str(song)] if self.here else ["-i", "https://example.invalid/stream"]

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    sched = [{"index": 0, "round": 1, "profile": "slow", "start": 0.0, "end": 20.0}]
    edge = {"schedule": sched, "playlist": [{"track_id": "s", "offset": 0.0, "play": 20.0, "pass": 1}], "beat": {"on": False},
            "voiceover": {"on": False, "cues": []}}
    src = Sources()
    loud = R._mix(edge, [], src, tmp_path, run)
    assert src.downloaded == ["raw/s"]                     # fetched once, then read as a file
    loud = loud.rename(tmp_path / "loud.wav")
    edge["voiceover"] = {"on": True, "music_volume": 0.5, "cues": [{"time": 15.0, "profile": "slow", "file": None}]}     # an older plan's 0.5
    quiet = R._mix(edge, [], src, tmp_path, run)
    assert E.MUSIC_UNDER_VOICE == 0.25
    assert _level(quiet, 2, 12) == pytest.approx(0.25 * _level(loud, 2, 12), rel=0.1)


def test_with_the_avatar_her_clips_sound_is_heard_at_the_lines_loudness(tmp_path):
    import edge_render as R
    ff = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)      # noqa: E731
    mp3, clip = tmp_path / "line.wav", tmp_path / "her.mp4"
    ff("-f", "lavfi", "-i", "sine=frequency=700:duration=2", "-af", "volume=1", str(mp3))        # the line: loud
    ff("-f", "lavfi", "-i", "color=c=yellow:s=90x160:d=3:r=25", "-f", "lavfi", "-i", "sine=frequency=700:duration=3",
       "-af", "volume=0.1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(clip))    # her clip: much quieter
    edge = {"voiceover": {"on": True, "cues": [{"time": 5.0, "profile": "slow", "file": str(mp3)},
                                               {"time": 30.0, "profile": "slow", "file": None},
                                               {"time": 40.0, "profile": "slow", "file": str(mp3)}]}}
    shows = [{"time": 5.0, "seconds": 3.0, "profile": "slow", "clip": str(clip), "still": None, "sound": str(clip)},
             {"time": 30.0, "seconds": 3.0, "profile": "slow", "clip": str(clip), "still": None, "sound": str(clip)},
             {"time": 40.0, "seconds": 3.0, "profile": "slow", "clip": None, "still": "x.png", "sound": None}]
    cues = R._sound_cues(edge, shows)
    assert [c["file"] for c in cues] == [str(clip), str(clip), str(mp3)]      # her clip where she has one, the line otherwise
    assert cues[0]["match"] == str(mp3) and not cues[1].get("match")
    assert 18 < R._match_gain(str(clip), str(mp3)) <= 20 and R._match_gain(str(clip), None) == 0.0

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    R._TARGET.clear()
    import unittest.mock as um
    with um.patch.object(R, "voice_target", lambda: -26.0):                                   # 🔊 one common level
        track = R._voice_track(cues, 50.0, tmp_path, run)
    assert _level(track, 5.2, 6.8) == pytest.approx(_level(track, 40.2, 41.8), rel=0.25)      # as loud as the line itself
    assert _level(track, 30.2, 31.8) == pytest.approx(_level(track, 5.2, 6.8), rel=0.25)      # and every clip at the same level


def test_quiet_lines_are_brought_up_to_the_typical_line_and_the_cue_maps_gain_goes_on_top(tmp_path, monkeypatch):
    """🔊 (17 Nov) A line recorded 18 dB quieter than the rest comes up to their level; the loud one comes down (at
    most 6 dB); a line's own ± dB from the Cue Map is added; the typical level is the middle of the lines on the server."""
    import edge_render as R
    import planner.edge as PE
    ff = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)      # noqa: E731
    lines = tmp_path / "lines"
    lines.mkdir()
    for name, vol in (("a", 0.1), ("b", 0.1), ("c", 0.1), ("quiet", 0.0125), ("loud", 0.8)):
        ff("-f", "lavfi", "-i", "sine=frequency=500:duration=1.5", "-af", f"volume={vol}", str(lines / f"{name}.wav"))
    monkeypatch.setattr(PE, "lines_dir", lambda: lines)
    R._TARGET.clear()
    t = R.voice_target()
    assert t == pytest.approx(R._peak_db(lines / "a.wav"), abs=0.2)
    assert R.level_gain(lines / "quiet.wav") == pytest.approx(18.06 if 18.06 < R.LEVEL_UP_MAX else R.LEVEL_UP_MAX, abs=0.5)
    assert R.level_gain(lines / "loud.wav") == -R.LEVEL_DOWN_MAX and R.level_gain(lines / "a.wav") == pytest.approx(0, abs=0.2)
    got = R.voice_levels({n: lines / f"{n}.wav" for n in ("a", "quiet", "loud")} | {"missing": None})
    assert [g[0] for g in got] == ["quiet", "a", "loud"]

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    cues = [{"time": 1.0, "profile": "slow", "file": str(lines / "a.wav")},
            {"time": 4.0, "profile": "slow", "file": str(lines / "quiet.wav")},
            {"time": 7.0, "profile": "slow", "file": str(lines / "a.wav"), "gain": 6}]
    track = R._voice_track(cues, 10.0, tmp_path, run)
    a, q, boosted = _level(track, 1.2, 2.3), _level(track, 4.2, 5.3), _level(track, 7.2, 8.3)
    assert q == pytest.approx(a * 10 ** ((R.LEVEL_UP_MAX - 18.06) / 20), rel=0.2)      # up as far as allowed
    assert boosted == pytest.approx(a * 2.0, rel=0.15)                                  # +6 dB from the Cue Map


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_picture_is_made_phase_by_phase_and_stays_in_step(tmp_path, monkeypatch):
    """Three phases with an avatar clip that runs across a phase boundary: every second of the footage is there once,
    in order, the timer restarts with each phase, and the picture and the sound are as long as the plan."""
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    ff = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)      # noqa: E731
    comb = tmp_path / "comb.mp4"                           # the footage's grey level is its own second
    ff("-f", "lavfi", "-i", "color=black:s=16x16:r=25:d=60", "-vf",
       "geq=lum='40+3*floor(N/25)':cb=128:cr=128,scale=640:360:flags=neighbor,setsar=1", "-c:v", "libx264", "-preset",
       "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(comb))
    clip = tmp_path / "her.mp4"
    ff("-f", "lavfi", "-i", "color=c=yellow:s=180x320:d=6:r=25", "-f", "lavfi", "-i", "sine=frequency=880:duration=6",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(clip))
    sched = [{"index": k, "round": 1, "profile": p, "start": 20.0 * k, "end": 20.0 * k + 20.0}
             for k, p in enumerate(("wait", "slow", "medium"))]
    edge = {"schedule": sched, "playlist": [], "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
            "overlay": {"corner": "top-left"},
            "avatar": {"on": True, "mode": "corner", "corner": "bottom-right",
                       "cues": [{"time": 17.0, "profile": "wait", "clip": str(clip)}]},      # 17 → 23 s: over the join
            "intro": {"on": False}, "outro": {"on": False}}

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    out = tmp_path / "out.mp4"
    seen = []
    took = R.finish({"edge": edge}, comb, str(out), types.SimpleNamespace(input_args=lambda raw: ["-i", raw]), tmp_path / "w",
                    run, 360, 640, progress=seen.append)
    assert took["edge_check"] == "ok" and seen[-1] == 1.0 and len(seen) == 3
    assert abs(_seconds(out) - 60.0) < 0.1
    grey = [_pixel(out, t + 0.5, 320, 60)[0] for t in range(60)]
    steps = [b - a for a, b in zip(grey, grey[1:])]
    assert all(2 <= d <= 5 for d in steps), steps          # one second after another, nothing skipped or repeated
    bh = int(360 * E.AVATAR_CORNER_SHARE)
    spot = (640 - int(360 * 0.035) - 30, 360 - int(360 * 0.035) - bh // 2)
    yellow = lambda p: p[0] > 170 and p[1] > 170 and p[2] < 90          # noqa: E731
    assert [yellow(_pixel(out, t, *spot)) for t in (16.5, 18.0, 19.9, 20.1, 22.5, 23.5)] == [False, True, True, True, True, False]
    w, _h = E.overlay_size(360)
    cx, top = int(360 * 0.035) + w // 2, int(360 * 0.035)
    assert _pixel(out, 20.5, cx, top + 4)[0] > 150 and _pixel(out, 40.5, cx, top + 4)[0] > 150      # a full ring as each phase starts


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_rendered_music_follows_the_rule_and_the_bell_rings_at_each_change(tmp_path, monkeypatch):
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    song = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=90", "-af", "volume=3",
                    str(song)], check=True)
    sched = [{"index": k, "round": 1, "profile": p, "start": 20.0 * k, "end": 20.0 * k + 20.0}
             for k, p in enumerate(("wait", "medium", "stop"))]
    src = types.SimpleNamespace(input_args=lambda raw: ["-i", str(song)])

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    edge = {"schedule": sched, "playlist": [{"track_id": "s", "offset": 0.0, "play": 60.0, "pass": 1}], "beat": {"on": False},
            "voiceover": {"on": True, "cues": [{"time": 30.0, "profile": "medium", "file": None, "seconds": 3.0}]},
            "music": {"fade_stop": True, "duck": True}, "signals": {"sound": False}}
    out = R._mix(edge, [], src, tmp_path, run)
    full = _level(out, 25.0, 28.0)
    assert full > 0.1
    assert _level(out, 0.0, 2.0) < 0.15 * full and _level(out, 17.5, 19.5) > 0.85 * full     # fades in over Wait
    assert _level(out, 36.0, 39.0) == pytest.approx(full, rel=0.1)                            # full after the line
    assert _level(out, 40.0, 40.3) > 0.75 * full                                              # Stop: down within a second…
    assert _level(out, 41.2, 41.8) == pytest.approx(0.48 * full, rel=0.12)
    assert _level(out, 49.0, 51.0) == pytest.approx(0.5 * (9.5 / 19) ** 1.7 * full, rel=0.2)    # …then out smoothly
    assert _level(out, 57.0, 59.5) < 0.05 * full
    assert _level(out, 59.6, 59.95) < 0.05 * full
    edge["voiceover"]["cues"] = []
    edge["voiceover"]["on"] = False
    edge["music"] = {}
    edge["playlist"] = []
    edge["beat"] = {"on": False}
    edge["signals"] = {"sound": True}
    bells = R._mix(edge, [], src, tmp_path, run)
    assert _level(bells, 20.0, 20.4) > 0.02 and _level(bells, 40.0, 40.4) > 0.02           # a bell at each change…
    assert _level(bells, 0.0, 0.4) < 0.002 and _level(bells, 30.0, 35.0) < 0.002          # …and nothing else


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_phase_signals_colour_the_ring_flash_the_picture_and_the_round_label_shows(tmp_path, monkeypatch):
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    comb = tmp_path / "comb.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x202020:s=640x360:r=25:d=40",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(comb)], check=True)
    sched = [{"index": 0, "round": 1, "profile": "slow", "start": 0.0, "end": 20.0},
             {"index": 1, "round": 2, "profile": "extra", "start": 20.0, "end": 40.0}]
    edge = {"schedule": sched, "playlist": [], "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
            "overlay": {"corner": "top-left"}, "avatar": {"on": False}, "intro": {"on": False}, "outro": {"on": False},
            "signals": {"sound": True, "timer": True, "flash": True}, "round_badge": {"on": True}}

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    out = tmp_path / "out.mp4"
    took = R.finish({"edge": edge}, comb, str(out), types.SimpleNamespace(input_args=lambda raw: ["-i", raw]), tmp_path / "w",
                    run, 360, 640)
    assert took["edge_check"] == "ok"
    w, h = E.overlay_size(360)
    cx, top = int(360 * 0.035) + w // 2, int(360 * 0.035)
    r, g, b = _pixel(out, 5.0, cx, top + 4)                 # Slow's green ring
    assert g > 150 and r < 140
    r, g, b = _pixel(out, 25.0, cx, top + 4)                # Extra's red one
    assert r > 150 and r > g + 80
    assert max(_pixel(out, 20.04, 600, 300)) > 70 > max(_pixel(out, 21.0, 600, 300))       # a flash as Extra starts
    assert max(_pixel(out, 0.04, 600, 300)) < 70                                           # (not at the very start)
    badge_y = top + h + int(360 * 0.012) + int(w * 0.17) // 2
    assert max(_pixel(out, 5.0, cx, badge_y)) > 150         # the round label's white letters / its dark pill under the timer
    assert _level(out, 20.0, 20.4) > 0.01


def test_the_stroke_bar_moves_on_the_beat_and_the_next_phase_shows(tmp_path, monkeypatch):
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    comb = tmp_path / "comb.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x202020:s=640x360:r=25:d=40",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(comb)], check=True)
    sched = [{"index": 0, "round": 1, "profile": "slow", "start": 0.0, "end": 20.0, "bpm": 60.0},
             {"index": 1, "round": 1, "profile": "stop", "start": 20.0, "end": 40.0, "bpm": 0.0}]
    guide = E.pace_guide(sched)
    edge = {"schedule": sched, "playlist": [], "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
            "overlay": {"corner": "top-left"}, "avatar": {"on": False}, "intro": {"on": False}, "outro": {"on": False},
            "signals": {"sound": False, "timer": True, "flash": False}, "round_badge": {"on": False},
            "pace": {"on": True, "beats": guide["beats"]}, "next_up": {"on": True}}

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    out = tmp_path / "out.mp4"
    took = R.finish({"edge": edge}, comb, str(out), types.SimpleNamespace(input_args=lambda raw: ["-i", raw]), tmp_path / "w",
                    run, 360, 640)
    assert took["edge_check"] == "ok"
    lay = R._guide_layers(edge, sched, "top-left", (640, 360), {})
    bar = lay["bar"]
    mx = bar["marker_x"] + bar["markers"]["slow"][0].shape[1] // 2
    top_y = bar["y"] + bar["marker_h"] // 2
    bottom_y = bar["y"] + bar["height"] - bar["marker_h"] // 2
    green = lambda px: px[1] > 140 and px[1] > px[0] + 40            # noqa: E731  (Slow's ring colour)
    assert green(_pixel(out, 4.0, mx, top_y)) and not green(_pixel(out, 4.0, mx, bottom_y))      # beat 5: at the top
    assert green(_pixel(out, 5.0, mx, bottom_y)) and not green(_pixel(out, 5.0, mx, top_y))      # beat 6: at the bottom
    assert not green(_pixel(out, 25.0, mx, top_y)) and not green(_pixel(out, 25.0, mx, bottom_y))   # Stop: no marker
    nx = lay["next"][0]
    cx, cy = nx[2] + nx[0].shape[1] * 3 // 4, nx[3] + nx[0].shape[0] // 2
    r, g, b = _pixel(out, 5.0, cx, cy)                               # "NEXT ▸ STOP": the Stop label's grey
    assert abs(r - g) < 30 and max(r, g, b) > 45
    r, g, b = _pixel(out, 25.0, cx, cy)                              # the last phase: "NEXT ▸ END"
    assert max(r, g, b) > 45


def test_a_hands_off_greys_the_timer_and_the_phase_carries_on(tmp_path, monkeypatch):
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    comb = tmp_path / "comb.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x202020:s=640x360:r=25:d=23",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(comb)], check=True)
    sched = [{"index": 0, "round": 1, "profile": "fast", "start": 0.0, "end": 8.0, "bpm": 140.0, "unit": 0, "kind": "phase",
              "clock": {"total": 20.0, "left": 20.0}},
             {"index": 1, "round": 1, "profile": "fast", "start": 8.0, "end": 11.0, "bpm": 0.0, "unit": 0, "kind": "handsoff",
              "clock": {"total": 3.0, "left": 3.0}, "surprise": "handsoff"},
             {"index": 2, "round": 1, "profile": "fast", "start": 11.0, "end": 23.0, "bpm": 140.0, "unit": 0, "kind": "phase",
              "clock": {"total": 20.0, "left": 12.0}}]
    edge = {"schedule": sched, "playlist": [], "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
            "overlay": {"corner": "top-left"}, "avatar": {"on": False}, "intro": {"on": False}, "outro": {"on": False},
            "signals": {"sound": True, "timer": True, "flash": False}, "round_badge": {"on": False}}

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    out = tmp_path / "out.mp4"
    took = R.finish({"edge": edge}, comb, str(out), types.SimpleNamespace(input_args=lambda raw: ["-i", raw]), tmp_path / "w",
                    run, 360, 640)
    assert took["edge_check"] == "ok"
    w, h = E.overlay_size(360)
    ring = w - 8
    m = int(360 * 0.035)
    lx, ly = m + int(w * 0.17), m + 4 + ring + int(ring * 0.05) + int(ring * 0.22) // 2
    r, g, b = _pixel(out, 9.0, lx, ly)                      # ✋ the grey HANDS OFF label
    assert abs(r - b) < 30 and 60 < r < 140
    r, g, b = _pixel(out, 12.0, lx, ly)                     # Fast's orange label again
    assert r > 180 and r > b + 100
    # the clock carried on: 12 s after the restart the ring is all but empty (20 − 8 − 11.5 ≈ 0.5 s left)
    top = _pixel(out, 22.5, m + w // 2, m + 4)
    assert max(top) < 200 or abs(top[0] - top[2]) < 40


def test_the_voice_is_15_percent_louder_and_the_music_5_percent_quieter(tmp_path):
    import edge_mode as E
    import edge_render as R
    assert (E.VOICE_LEVEL, E.MUSIC_LEVEL) == (1.15, 0.95)                      # Daryl, 17 Nov
    seen = []
    wav = E.write_placeholder_cue("fast", tmp_path / "x.wav")
    R._voice_track([{"time": 1.0, "file": str(wav), "profile": "fast"}], 5.0, tmp_path, lambda cmd: seen.append(cmd))
    assert any("volume=1.150,alimiter" in " ".join(c) for c in seen)          # (then the limiter, 17 Nov)
    assert "volume={E.MUSIC_LEVEL:.3f}" in open(R.__file__).read()                # on the songs, after their fades


def test_her_clip_keeps_the_lines_cue_map_gain():
    import edge_render as R
    edge = {"voiceover": {"on": True, "cues": [{"time": 5.0, "profile": "slow", "file": "line.mp3", "gain": -3}]}}
    shows = [{"time": 5.0, "seconds": 3.0, "profile": "slow", "clip": "her.mp4", "still": None, "sound": "her.mp4"}]
    assert R._sound_cues(edge, shows)[0]["gain"] == -3



def test_the_strip_timer_style_shows_the_box_the_number_and_the_stroke_strip(tmp_path, monkeypatch):
    """⏱️ Timer style ▬ Strip (21 Nov): a box at the top (number, phase, next, round), the stroke meter across the
    bottom (left ↔ right on the beats), the box filled with the phase colour as it starts, no ring."""
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    import edge_strip as ES
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    comb = tmp_path / "comb.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x202020:s=640x360:r=25:d=40",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(comb)], check=True)
    sched = [{"index": 0, "round": 1, "profile": "slow", "start": 0.0, "end": 20.0, "bpm": 60.0},
             {"index": 1, "round": 1, "profile": "extra", "start": 20.0, "end": 40.0, "bpm": 180.0}]
    guide = E.pace_guide(sched)
    edge = {"schedule": sched, "playlist": [], "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
            "overlay": {"corner": "top-left", "style": "strip"}, "avatar": {"on": False}, "intro": {"on": False},
            "outro": {"on": False}, "signals": {"sound": False, "timer": True, "flash": False},
            "round_badge": {"on": True}, "pace": {"on": True, "beats": guide["beats"]}, "next_up": {"on": True}}

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    out = tmp_path / "out.mp4"
    took = R.finish({"edge": edge}, comb, str(out), types.SimpleNamespace(input_args=lambda raw: ["-i", raw]),
                    tmp_path / "w", run, 360, 640)
    assert took["edge_check"] == "ok"
    assert not (tmp_path / "w" / "overlay").exists() or not list((tmp_path / "w" / "overlay").glob("*.png"))   # no ring
    st_ = R._strip_layers(edge, sched, (640, 360), True)
    lay, bar = st_["lay"], st_["bar"]
    assert lay["w"] < 640 and lay["h"] == 30 and bar["h"] == ES.bar_height(360) and bar["y"] == 360 - bar["h"]
    # the box is dark; as Extra starts it drops down in Extra's red and a red frame pulses round the picture
    bx, by = lay["x"] + lay["w"] - 6, lay["h"] // 2
    assert max(_pixel(out, 10.0, bx, by)) < 40
    red = lambda px: px[0] > 150 and px[1] < 90                     # noqa: E731
    assert red(_pixel(out, 20.6, 320, 80)) and not red(_pixel(out, 10.0, 320, 80))      # the dropped box, below the bar
    assert not red(_pixel(out, 22.0, 320, 80))                                          # tucked back up
    assert red(_pixel(out, 20.05, 3, 180)) and not red(_pixel(out, 10.0, 3, 180)) and not red(_pixel(out, 22.0, 3, 180))
    assert not red(_pixel(out, 0.1, 3, 180))                                            # (not as the run starts)
    # the round sits in the box: something light right of the second divider
    light = [_pixel(out, 10.0, x, yy) for x in range(lay["x"] + lay["round_x"], lay["x"] + lay["w"] - 30, 2)
             for yy in range(4, lay["h"] - 4, 2)]
    assert any(min(px) > 150 for px in light)
    # the stroke marker slides along the bottom strip: at one end on a beat, the other end on the next
    my = bar["y"] + bar["h"] // 2
    green = lambda px: px[1] > 140 and px[1] > px[0] + 40            # noqa: E731  (Slow's ring colour)
    left_x, right_x = bar["x0"] + 4, bar["x1"] - 4
    assert green(_pixel(out, 4.0, left_x, my)) and not green(_pixel(out, 4.0, right_x, my))
    assert green(_pixel(out, 5.0, right_x, my)) and not green(_pixel(out, 5.0, left_x, my))


def test_instructions_stay_until_replaced_and_show_in_her_half_of_the_band(tmp_path, monkeypatch):
    """🗒️ Instructions (21 Nov): put up by linked lines (after their delay), kept until another replaces them; drawn
    in her half of a 60 px band (◧ split screen), the strip's stroke meter in the footage half."""
    import sys
    import types
    import edge_mode as E
    import edge_render as R
    import edge_strip as ES
    cm = {"lines": [{"name": "Tip", "instruction": "JUST THE TIP", "instruction_delay": 0.5},
                    {"name": "Filler"}, {"name": "Same", "instruction": "JUST THE TIP"},
                    {"name": "Stroke", "instruction": "stroke  it"}]}
    plays = [{"time": 2.0, "line": "Tip"}, {"time": 5.0, "line": "Filler"}, {"time": 8.0, "line": "Same"},
             {"time": 12.0, "line": "Stroke"}]
    track = E.instruction_track(plays, cm)
    assert track == [{"time": 2.5, "text": "JUST THE TIP", "range": [0.0, 1.0]},
                     {"time": 12.0, "text": "stroke it", "range": [0.0, 1.0]}]
    cm["instruction_ranges"] = {"JUST THE TIP": [2 / 3, 1.0]}
    assert E.instruction_track(plays, cm)[0]["range"] == [round(2 / 3, 4), 1.0]
    assert E.instruction_at(track, 1.0) is None and E.instruction_at(track, 9.0) == "JUST THE TIP"
    assert E.instruction_at(track, 30.0) == "stroke it"
    # the render
    monkeypatch.setitem(sys.modules, "render_preview", types.SimpleNamespace(get_track_raw_source=lambda t: t))
    comb = tmp_path / "comb.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x202020:s=640x360:r=25:d=20",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(comb)], check=True)
    sched = [{"index": 0, "round": 1, "profile": "slow", "start": 0.0, "end": 20.0, "bpm": 60.0}]
    guide = E.pace_guide(sched)
    edge = {"schedule": sched, "playlist": [], "beat": {"on": True, "volume": 1.0}, "voiceover": {"on": False, "cues": []},
            "overlay": {"corner": "top-left", "style": "strip"}, "avatar": {"on": True, "mode": "split", "cues": []},
            "intro": {"on": False}, "outro": {"on": False}, "signals": {"sound": False, "timer": True, "flash": False},
            "round_badge": {"on": True}, "pace": {"on": True, "beats": guide["beats"]}, "next_up": {"on": True},
            "instructions": {"on": True, "items": [{"time": 4.0, "text": "JUST THE TIP", "range": [2 / 3, 1.0]}]}}

    def run(cmd):
        subprocess.run(cmd, capture_output=True, check=True)
    out = tmp_path / "out.mp4"
    took = R.finish({"edge": edge}, comb, str(out), types.SimpleNamespace(input_args=lambda raw: ["-i", raw]),
                    tmp_path / "w", run, 360, 640)
    assert took["edge_check"] == "ok"
    band = ES.band_height(360)
    st_ = R._strip_layers(edge, sched, (640, 360), True, band)
    assert st_["bar"]["h"] == band and st_["bar"]["track"][0].shape[1] == 320          # the meter: the footage half
    y = 360 - band // 2
    white = lambda px: min(px) > 180                                    # noqa: E731
    before = [_pixel(out, 2.0, x, y) for x in range(340, 630, 3)]
    after = [_pixel(out, 8.0, x, y) for x in range(340, 630, 3)]
    assert not any(white(px) for px in before) and any(white(px) for px in after)     # the text from 4 s, kept on
    assert any(white(px) for px in [_pixel(out, 18.0, x, y) for x in range(340, 630, 3)])
    # 🎯 "just the tip" keeps the marker to the right third of the bar; before it, the whole bar
    bar = st_["bar"]
    green = lambda px: px[1] > 140 and px[1] > px[0] + 40              # noqa: E731
    xs = range(bar["x0"], bar["x1"], 2)
    for t in (8.0, 9.0, 10.0, 11.0, 12.5):
        hit = [x for x in xs if green(_pixel(out, t, x, y))]
        assert hit and min(hit) >= bar["x0"] + (bar["x1"] - bar["x0"]) * 2 / 3 - 4, (t, hit[:3])
    early = [x for t in (1.0, 2.0) for x in xs if green(_pixel(out, t, x, y))]
    assert early and min(early) < bar["x0"] + (bar["x1"] - bar["x0"]) / 3
