"""Deterministic synthetic library for the tests: a handful of audio tracks and
video catalogues in the same shape the real pipeline writes, covering the edge
cases the planner has to handle (no extra analysis, legacy envelope-only tracks,
very short tracks, videos without motion curves, excluded scenes, roles, tags).

Nothing here is random at run time — every value comes from a seeded generator,
so golden snapshots taken from it are stable across machines."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RATE = 30.0


def _tc(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _r(arr, nd=3):
    return [round(float(x), nd) for x in arr]


def make_track(seed: int, duration: float, bpm: float, *, analysis: bool = True,
               hires: bool = True, beats: bool = True) -> dict:
    """A song-like track: intro → build → drop → breakdown → drop → outro, with
    a beat-synchronous pulse, a few big hits, and (optionally) the server's extra
    analysis block."""
    rng = np.random.default_rng(seed)
    n = int(duration * RATE)
    t = np.arange(n) / RATE
    beat = 60.0 / bpm
    bar = 4 * beat
    # section envelope
    shape = np.interp(t / duration, [0, .12, .28, .30, .5, .55, .72, .74, .92, 1.0],
                      [.15, .3, .7, .95, .9, .35, .45, 1.0, .9, .1])
    pulse = 0.5 + 0.5 * np.cos(2 * np.pi * ((t % beat) / beat)) ** 8
    rms = np.clip(shape * (0.7 + 0.3 * pulse) + rng.normal(0, 0.02, n), 0, 1)
    onset = np.clip(0.25 * pulse * shape + rng.normal(0, 0.03, n), 0, 1)
    # big hits at the drops and a couple of random ones
    for frac in (.30, .74, .47, .61):
        i = int(frac * n)
        onset[i] = 0.97
        rms[i:i + 6] = np.minimum(1.0, rms[i:i + 6] + 0.2)
    beat_times = np.arange(0.12, duration - 0.05, beat)
    track = {
        "track_id": f"track_{seed}",
        "duration_sec": round(duration, 3),
        "bpm": round(bpm, 2),
        "beat_times": _r(beat_times) if beats else [],
        "energy_envelope": [{"time": round(i * 0.5, 2),
                             "energy": round(float(rms[int(i * 0.5 * RATE):int((i + 1) * 0.5 * RATE)].mean()
                                                   if int(i * 0.5 * RATE) < n else 0.0), 3)}
                            for i in range(int(duration / 0.5))],
    }
    if hires:
        track["hires"] = {"rate": RATE, "rms": _r(rms), "onset": _r(onset)}
    if analysis and hires:
        downbeats = beat_times[::4]
        kick = np.clip(pulse * shape + rng.normal(0, .02, n), 0, 1)
        snare = np.clip(np.roll(pulse, int(beat * RATE)) * shape * .8, 0, 1)
        harmony = np.clip(np.abs(np.sin(2 * np.pi * t / (bar * 2))) * .6 + rng.normal(0, .02, n), 0, 1)
        sec_bounds = [0, .12, .30, .55, .74, .92, 1.0]
        labels = "ABCDCE"
        sections = [{"start": round(a * duration, 3), "end": round(b * duration, 3), "label": labels[k],
                     "energy": round(float(shape[int(a * n):max(int(a * n) + 1, int(b * n))].mean()), 2)}
                    for k, (a, b) in enumerate(zip(sec_bounds[:-1], sec_bounds[1:]))]
        vocals = np.zeros(n)
        vstarts, vends = [], []
        for a, b in ((.32, .40), (.43, .52), (.76, .84)):
            vocals[int(a * n):int(b * n)] = .8
            vstarts.append(round(a * duration, 3))
            vends.append(round(b * duration, 3))
        track["analysis"] = {
            "version": 3, "rate": RATE, "beats_per_bar": 4,
            "methods": {"beats": "test", "downbeats": "test", "vocals": "test"},
            "downbeats": _r(downbeats),
            "kick": _r(kick), "snare": _r(snare), "hat": _r(onset), "harmony": _r(harmony),
            "brightness": _r(rms), "novelty": _r(np.abs(np.gradient(shape)) * 20),
            "sections": sections,
            "phrases": [{"time": round(float(x), 3), "level": 16 if k % 4 == 0 else (8 if k % 2 == 0 else 4)}
                        for k, x in enumerate(downbeats[::4])],
            "builds": [{"start": round(.20 * duration, 3), "end": round(.30 * duration, 3)},
                       {"start": round(.65 * duration, 3), "end": round(.74 * duration, 3)}],
            "drops": [round(.30 * duration, 3), round(.74 * duration, 3)],
            "vocals": _r(vocals), "vocal_starts": vstarts, "vocal_ends": vends,
        }
    return track


def make_video(seed: int, video_id: str, n_scenes: int, *, curves: bool = True, fps: float = 10.0,
               excluded: tuple = (), roles: dict = None, tags_cycle=("dance", "street", "crowd", "night")) -> dict:
    rng = np.random.default_rng(seed)
    roles = roles or {}
    scenes, t = [], 0.0
    for k in range(n_scenes):
        dur = float(np.round(rng.uniform(1.5, 14.0), 2))
        level = float(rng.uniform(0.05, 1.0))
        nf = max(2, int(dur * fps))
        x = np.linspace(0, 1, nf)
        curve = np.clip(level * (0.6 + 0.4 * np.sin(2 * np.pi * x * rng.uniform(.5, 3) + rng.uniform(0, 6)))
                        + rng.normal(0, .03, nf), 0, None)
        scene = {
            "scene_id": k,
            "start_tc": _tc(t), "end_tc": _tc(t + dur),
            "tags": [tags_cycle[k % len(tags_cycle)], tags_cycle[(k * 3 + seed) % len(tags_cycle)]],
            "thumbnail_paths": [f"/content/drive/MyDrive/scene-labeling/catalogue/thumbnails/{video_id}/s{k:03d}.jpg"],
            "motion_intensity": round(level, 4),
        }
        if curves:
            scene["motion_curve"] = {"fps": fps, "values": _r(curve, 4)}
        if k in excluded:
            scene["excluded"] = True
        if k in roles:
            scene["roles"] = list(roles[k])
            scene["intro_candidate"] = "intro" in roles[k]
            scene["outro_candidate"] = "outro" in roles[k]
        scenes.append(scene)
        t += dur
    return {"video_id": video_id, "source_path": f"/content/drive/MyDrive/Videos-PH/{video_id}.mp4",
            "scenes": scenes}


TRACKS = {
    "song_full": dict(seed=1, duration=96.0, bpm=124.0),
    "song_noanalysis": dict(seed=2, duration=74.0, bpm=98.0, analysis=False),
    "song_slow": dict(seed=3, duration=120.0, bpm=72.0),
    "song_legacy": dict(seed=4, duration=60.0, bpm=110.0, analysis=False, hires=False),
    "song_short": dict(seed=5, duration=6.0, bpm=128.0),
}

VIDEOS = {
    "vid_a": dict(seed=11, n_scenes=14, roles={0: ("intro",), 5: ("action",), 6: ("action",), 13: ("outro",)}),
    "vid_b": dict(seed=12, n_scenes=10, excluded=(2,), roles={3: ("slow",), 9: ("outro",)}),
    "vid_c": dict(seed=13, n_scenes=18, roles={1: ("intro",), 7: ("action",), 8: ("action", "slow")}),
    "vid_d": dict(seed=14, n_scenes=8, fps=25.0),
    "vid_e": dict(seed=15, n_scenes=6, curves=False),
    "vid_f": dict(seed=16, n_scenes=12, roles={4: ("action",)}),
}


def write_library(root: Path) -> dict:
    """Write the synthetic library under root. Returns the config paths."""
    root = Path(root)
    cat_dir, audio_dir = root / "catalogue", root / "audio_catalogue"
    paths = {"DRIVE_ROOT": root, "CATALOGUE_DIR": cat_dir, "AUDIO_DIR": audio_dir,
             "PLANS_DIR": root / "compilation_plans", "PREVIEW_DIR": root / "previews",
             "PROJECTS_DIR": root / "projects"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    for tid, kw in TRACKS.items():
        tr = make_track(**kw)
        tr["track_id"] = tid
        tr["source_path"] = f"/content/drive/MyDrive/Audio-Library/{tid}.m4a"
        (audio_dir / f"{tid}.json").write_text(json.dumps(tr))
    for vid, kw in VIDEOS.items():
        (cat_dir / f"{vid}.json").write_text(json.dumps(make_video(video_id=vid, **kw)))
    (cat_dir / "labels.json").write_text(json.dumps(["dance", "street", "crowd", "night"]))
    return paths


def load_track(root: Path, tid: str) -> dict:
    return json.loads((Path(root) / "audio_catalogue" / f"{tid}.json").read_text())
