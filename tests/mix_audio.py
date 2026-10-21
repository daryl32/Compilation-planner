"""Small synthetic songs (real audio files) for the 🎚️ mixing tests: chords,
bass and a kick drum on a steady grid, in sections that repeat — so the mixer
has real bars that sound alike to jump between. Deterministic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
_PROGS = {"A": [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]],     # I IV V I
          "B": [[9, 12, 16], [5, 9, 12], [0, 4, 7], [7, 11, 14]],   # vi IV I V
          "C": [[2, 5, 9], [7, 11, 14], [2, 5, 9], [7, 11, 14]]}    # ii V


def _midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def synth_song(path: Path, bpm: float, root: int, layout, seed: int = 0) -> Path:
    """layout: [(section letter, bars)]. root: semitones above C (0 = C major)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = 4 * beat
    out = []
    for name, nbars in layout:
        for b in range(nbars):
            chord = _PROGS[name][b % 4]
            t = np.arange(int(round(bar * SR))) / SR
            y = sum(0.15 * np.sin(2 * np.pi * _midi(root + 48 + n) * t) for n in chord)
            y = y + 0.2 * np.sin(2 * np.pi * _midi(root + 36 + chord[0]) * t)
            n = int(0.12 * SR)
            env = np.exp(-np.arange(n) / SR * 30)
            for k in range(4):
                i = int(k * beat * SR)
                y[i:i + n] += 0.6 * np.sin(2 * np.pi * 55 * np.arange(n) / SR) * env[:len(y[i:i + n])]
                if name == "B":
                    j = i + int(beat * SR / 2)
                    y[j:j + n] += 0.15 * rng.normal(0, 1, len(y[j:j + n])) * env[:len(y[j:j + n])]
            out.append(y)
    y = np.concatenate(out).astype(np.float32)
    y /= np.abs(y).max() * 1.1
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-c:a", "aac", "-b:a", "160k", str(path)],
                   input=np.stack([y, y], 1).tobytes(), check=True)
    return path


SONGS = {
    # track_id: (bpm, root, layout)
    "mix_song_c": (120.0, 0, [("A", 4), ("B", 8), ("A", 4), ("B", 8), ("C", 4), ("B", 4)]),
    "mix_song_a": (126.0, 9, [("A", 4), ("B", 8), ("C", 4), ("B", 4)]),
}


def synth_vocals(path: Path, bpm: float, root: int, bars: int) -> Path:
    """A 'sung' line for the stem tests: a melody of held notes in the song's key, a phrase
    every two bars with gaps between — stands in for a Demucs vocal stem."""
    beat = 60.0 / bpm
    notes = [12, 14, 16, 14, 12, 11, 12, 7]
    out = []
    for b in range(bars):
        t = np.arange(int(round(4 * beat * SR))) / SR
        y = np.zeros_like(t)
        if b % 2 == 0:
            for k in range(4):
                n = notes[(b + k) % len(notes)]
                seg = (t >= k * beat) & (t < (k + 0.85) * beat)
                y[seg] = 0.4 * np.sin(2 * np.pi * _midi(root + 60 + n) * t[seg])
        out.append(y)
    y = np.concatenate(out).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-c:a", "aac", "-b:a", "160k", str(path)],
                   input=np.stack([y, y], 1).tobytes(), check=True)
    return path


def install(audio_dir: Path, files_dir: Path) -> dict:
    """Write the songs' audio into files_dir and analysed catalogues into
    audio_dir (as the server's analysis would). Returns {track_id: track}."""
    import audio_analysis as AA
    out = {}
    for tid, (bpm, root, layout) in SONGS.items():
        path = files_dir / f"{tid}.m4a"
        js = audio_dir / f"{tid}.json"
        if not path.exists():
            synth_song(path, bpm, root, layout, seed=len(tid))
        if js.exists():
            out[tid] = json.loads(js.read_text())
            continue
        track = {"track_id": tid, "source_path": str(path),
                 **AA.analyse_file(path, with_vocals=False)}
        # 🎤 stems as analyse_audio_local.py notes them: the song itself stands in for the
        # instrumental, a synthetic melody for the vocals
        voc = synth_vocals(files_dir / "stems" / tid / "vocals.m4a", bpm, root, sum(n for _, n in layout))
        inst = files_dir / "stems" / tid / "instrumental.m4a"
        inst.write_bytes(path.read_bytes())
        track["stems"] = {"vocals": str(voc), "instrumental": str(inst), "model": "test"}
        js.write_text(json.dumps(track))
        out[tid] = track
    return out
