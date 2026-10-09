r"""
Render a quick, low-res preview MP4 from a saved compilation plan JSON.
Trims and (for split-screen segments) grid-combines the matched scenes,
concatenates them in order, and muxes in the track's audio.

Usage:
    python render_preview.py "path\to\plan.json"

Requires ffmpeg on PATH (winget install ffmpeg).

Sources: a video or the track is read from this machine when it's here. On the
server, one that isn't is STREAMED from Google Drive (stream=True, the default):
ffmpeg reads the file over HTTPS with range requests, so only the seconds each
clip uses are fetched and nothing is stored. If a stream fails, those files are
downloaded as before and the segment is rendered again from the local copy.
"""

import re
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

from config import DRIVE_ROOT, CATALOGUE_DIR, AUDIO_DIR

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360


def colab_to_local(colab_path: str) -> str:
    import os
    
    if colab_path.startswith("/root/data"):
        return colab_path
    
    marker = "MyDrive/"
    idx = colab_path.find(marker)
    if idx == -1:
        return colab_path
    rel = colab_path[idx + len(marker):]
    
    if os.name == "nt":
        return str(DRIVE_ROOT / rel.replace("/", "\\"))
    else:
        clean_rel = rel.replace("\\", "/").replace("//", "/")
        return str(Path("/root/data") / clean_rel)


_video_source_cache = {}


def get_video_raw_source(video_id: str) -> str:
    """The catalogue's source_path as stored (Colab-style "/content/drive/MyDrive/…")."""
    if video_id not in _video_source_cache:
        with open(CATALOGUE_DIR / f"{video_id}.json") as f:
            cat = json.load(f)
        _video_source_cache[video_id] = cat["source_path"]
    return _video_source_cache[video_id]


def get_video_source(video_id: str) -> str:
    return colab_to_local(get_video_raw_source(video_id))


def get_track_raw_source(track_id: str) -> str:
    """The track catalogue's source_path as stored."""
    import unicodedata
    exact = AUDIO_DIR / f"{track_id}.json"
    if exact.exists():
        with open(exact) as f:
            return json.load(f)["source_path"]
    for f in AUDIO_DIR.glob("*.json"):
        if unicodedata.normalize("NFC", f.stem) == unicodedata.normalize("NFC", track_id):
            with open(f) as fh:
                return json.load(fh)["source_path"]
    raise FileNotFoundError(f"No audio catalogue found for track_id: {track_id}")


# ---------------------------------------------------------------------------
# Where ffmpeg reads each source from: local file, or a Drive stream
# ---------------------------------------------------------------------------

def _drive_rel(raw_source: str) -> str | None:
    return raw_source.split("MyDrive/", 1)[1] if "MyDrive/" in raw_source else None


def streaming_available() -> bool:
    """True where the Drive service account is set up (the server)."""
    try:
        from drive_sync import credentials_available
        return credentials_available()
    except Exception:
        return False


class SourceResolver:
    """Turns a catalogue source_path into ffmpeg input arguments.

    Local file present → read it. Otherwise, with stream=True and Drive set up →
    stream it from Drive (Drive file ids are looked up once per file per render).
    Otherwise → FileNotFoundError, as before. download(raw) fetches a file to its
    normal local path (the fallback when a stream fails)."""

    def __init__(self, stream: bool = True, notice=None):
        self.stream = stream and streaming_available()
        self.notice = notice or (lambda msg: None)
        self._ids = {}
        self.streamed, self.downloaded = set(), set()

    def is_local(self, raw: str) -> bool:
        return Path(colab_to_local(raw)).exists()

    def check(self, raws) -> list:
        """Error strings for sources that can be neither read here nor streamed."""
        errors = []
        for raw in sorted(set(raws)):
            if self.is_local(raw):
                continue
            rel = _drive_rel(raw)
            if not (self.stream and rel):
                errors.append(f"{Path(colab_to_local(raw)).name}: not found on this machine")
                continue
            if raw not in self._ids:
                from drive_sync import find_drive_file
                f, err = find_drive_file(rel)
                if err:
                    errors.append(f"{Path(rel).name}: {err}")
                    continue
                self._ids[raw] = f["id"]
        return errors

    def input_args(self, raw: str) -> list:
        local = colab_to_local(raw)
        if Path(local).exists():
            return ["-i", local]
        rel = _drive_rel(raw)
        if not (self.stream and rel):
            raise FileNotFoundError(f"{Path(local).name}: not found on this machine")
        if raw not in self._ids:
            errs = self.check([raw])
            if errs:
                raise FileNotFoundError(errs[0])
        from drive_sync import access_token, stream_url
        self.streamed.add(raw)
        return ["-reconnect", "1", "-reconnect_delay_max", "5", "-rw_timeout", "60000000",
                "-headers", f"Authorization: Bearer {access_token()}\r\n",
                "-i", stream_url(self._ids[raw])]

    def stream_works(self, raw: str) -> bool:
        """Quick check that ffmpeg can open this source as a Drive stream."""
        try:
            args = self.input_args(raw)
        except FileNotFoundError:
            return False
        proc = subprocess.run(["ffprobe", "-v", "error", *args[:-2], args[-1]],
                              capture_output=True, text=True, timeout=120)
        return proc.returncode == 0

    def download(self, raws) -> list:
        """Download the given sources to their local paths. Returns error strings."""
        from drive_sync import download_source_video
        errors = []
        for raw in sorted(set(raws)):
            if self.is_local(raw):
                continue
            rel = _drive_rel(raw)
            if not rel:
                errors.append(f"{Path(colab_to_local(raw)).name}: not on Drive")
                continue
            self.notice(f"Streaming didn't work — downloading {Path(rel).name}…")
            err = download_source_video(rel, Path(colab_to_local(raw)))
            if err:
                errors.append(f"{Path(rel).name}: {err}")
            else:
                self.downloaded.add(raw)
        return errors


def segment_sources(entry: dict) -> set:
    return {get_video_raw_source(link["video_id"])
            for slot in entry.get("scenes", []) for link in slot.get("chain", [])}


def get_track_source(track_id: str) -> str:
    import unicodedata
    # Try exact match first
    exact = AUDIO_DIR / f"{track_id}.json"
    if exact.exists():
        with open(exact) as f:
            track = json.load(f)
        return colab_to_local(track["source_path"])
    
    # Try normalised match for special characters
    for f in AUDIO_DIR.glob("*.json"):
        if unicodedata.normalize("NFC", f.stem) == unicodedata.normalize("NFC", track_id):
            with open(f) as fh:
                track = json.load(fh)
            return colab_to_local(track["source_path"])
    
    raise FileNotFoundError(f"No audio catalogue found for track_id: {track_id}")


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install it with: winget install ffmpeg")
        print("Then close and reopen your terminal before retrying.")
        sys.exit(1)


def _redact(text: str) -> str:
    return re.sub(r"Bearer [A-Za-z0-9._\-]+", "Bearer ***", text)


def run(cmd: list):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = _redact(f"ffmpeg command failed:\n{' '.join(cmd)}\n{result.stderr[-2000:]}")
        import sys
        print(error_msg, file=sys.stderr, flush=True)
        raise RuntimeError(_redact(f"ffmpeg failed: {result.stderr[-500:]}"))


# Cell layouts copied from the VSE add-on's _AUTO_LAYOUT, so the preview matches
# what Import Plan + Apply Split-Screen (Layout Mode: Auto) builds in Blender.
# Each cell is (x0, y0, x1, y1) as fractions of the frame, (0,0) = top-left.
def _grid(cols, rows):
    return [(c / cols, r / rows, (c + 1) / cols, (r + 1) / rows) for r in range(rows) for c in range(cols)]

AUTO_LAYOUTS = {
    1: _grid(1, 1),
    2: [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)],                                  # DIPTYCH
    3: [(0.0, 0.0, 1/3, 1.0), (1/3, 0.0, 2/3, 1.0), (2/3, 0.0, 1.0, 1.0)],             # TRIPTYCH
    4: _grid(2, 2),
    5: [(0.0, 0.0, 1/3, 0.5), (1/3, 0.0, 2/3, 0.5), (2/3, 0.0, 1.0, 0.5),              # FIVE_UP
        (0.0, 0.5, 0.5, 1.0), (0.5, 0.5, 1.0, 1.0)],
    6: _grid(3, 2),
    7: [(0.0, 0.0, 0.25, 0.5), (0.25, 0.0, 0.5, 0.5), (0.5, 0.0, 0.75, 0.5),           # SEVEN_UP
        (0.75, 0.0, 1.0, 0.5), (0.0, 0.5, 1/3, 1.0), (1/3, 0.5, 2/3, 1.0), (2/3, 0.5, 1.0, 1.0)],
    8: _grid(4, 2),
}


def _cell_pixels(cell):
    """Fractional cell -> integer (x, y, w, h), with even sizes for libx264."""
    x0, y0, x1, y1 = cell
    x, y = int(round(x0 * PREVIEW_WIDTH)), int(round(y0 * PREVIEW_HEIGHT))
    w = int(round(x1 * PREVIEW_WIDTH)) - x
    h = int(round(y1 * PREVIEW_HEIGHT)) - y
    return x, y, max(2, w - w % 2), max(2, h - h % 2)


def render_segment(entry: dict, idx: int, tmp_dir: Path, sources: SourceResolver = None) -> Path:
    """Render one timeline entry (1-8 simultaneous slots) to a temp clip of
    the segment's duration, no audio. Each slot's chain of clips is joined
    back-to-back, cropped to fill its cell (the add-on's default Crop fit),
    and overlaid onto a black canvas at that cell's position."""
    sources = sources or SourceResolver(stream=False)
    duration = round(entry["track_time"][1] - entry["track_time"][0], 3)
    slots = entry["scenes"]
    n = len(slots)
    out_path = tmp_dir / f"seg_{idx:04d}.mp4"

    if n == 0:
        run(["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"color=c=black:s={PREVIEW_WIDTH}x{PREVIEW_HEIGHT}:d={duration}:r=25",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", str(out_path)])
        return out_path

    cells = AUTO_LAYOUTS.get(n) or _grid(4, (n + 3) // 4)
    inputs = []
    filter_parts = [f"color=c=black:s={PREVIEW_WIDTH}x{PREVIEW_HEIGHT}:d={duration}:r=25[base0]"]
    input_idx = 0

    for slot_i, slot in enumerate(slots):
        x, y, w, h = _cell_pixels(cells[slot_i])
        link_labels = []
        for link in slot["chain"]:
            src_args = sources.input_args(get_video_raw_source(link["video_id"]))
            inputs += ["-ss", str(link["clip_start_sec"]), "-t", str(link["clip_duration_sec"]), *src_args]
            label = f"c{input_idx}"
            filter_parts.append(
                f"[{input_idx}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},setsar=1,fps=25[{label}]"
            )
            link_labels.append(f"[{label}]")
            input_idx += 1

        if len(link_labels) == 1:
            filter_parts.append(f"{link_labels[0]}null[s{slot_i}]")
        else:
            filter_parts.append(f"{''.join(link_labels)}concat=n={len(link_labels)}:v=1:a=0[s{slot_i}]")

        out_label = "outv" if slot_i == n - 1 else f"base{slot_i + 1}"
        filter_parts.append(f"[base{slot_i}][s{slot_i}]overlay={x}:{y}:eof_action=pass[{out_label}]")

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]", "-an", "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        str(out_path),
    ]
    run(cmd)
    return out_path


def render_plan_dict(plan: dict, output_path: str, progress_callback=None, stream: bool = True,
                     notice=None) -> dict:
    """Core renderer, operating on an already-loaded plan dict. Used both by
    the CLI (render_plan below) and directly by the planner app on its
    in-memory plan, with no need to write/read a JSON file in between.
    progress_callback(done, total), if given, is called after each segment —
    lets a caller (e.g. Streamlit) show progress without depending on tqdm's
    console-only output. stream: read sources that aren't on this machine
    straight from Google Drive (see module docstring). notice(text): status
    messages, e.g. when a stream fails and a file is downloaded instead.
    Returns {"streamed": n, "downloaded": n, "local": n} source-file counts."""
    check_ffmpeg()
    sources = SourceResolver(stream=stream, notice=notice)

    track_raw = get_track_raw_source(plan["track_id"])
    all_raw = {track_raw}
    for entry in plan["timeline"]:
        all_raw |= segment_sources(entry)
    missing = sources.check(all_raw)
    if missing:
        raise FileNotFoundError("Some source files can't be read:\n" + "\n".join(missing))
    local_at_start = {r for r in all_raw if sources.is_local(r)}

    def with_fallback(raws, do):
        """Run do(). If it fails while streaming any of raws: download the ones whose
        stream doesn't open and retry; if they all open (a passing network blip),
        retry once as-is, then download the lot as a last resort."""
        try:
            return do()
        except RuntimeError:
            remote = [r for r in raws if not sources.is_local(r)]
            if not remote:
                raise
        broken = [r for r in remote if not sources.stream_works(r)]
        # First retry: download just the broken streams (or nothing, if all open).
        errs = sources.download(broken)
        if not errs:
            try:
                return do()
            except RuntimeError:
                pass
        # Last resort: everything this step needs, downloaded.
        errs = sources.download(remote)
        if errs:
            raise RuntimeError("Streaming failed and the download fallback failed too:\n"
                               + "\n".join(errs))
        return do()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clip_paths = []
        total = len(plan["timeline"])

        for i, entry in enumerate(tqdm(plan["timeline"], desc="Rendering segments")):
            clip_paths.append(with_fallback(segment_sources(entry),
                                            lambda e=entry, i=i: render_segment(e, i, tmp_dir, sources)))
            if progress_callback:
                progress_callback(i + 1, total)

        # Concatenate
        concat_list = tmp_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p.as_posix()}'\n")

        combined_video = tmp_dir / "combined.mp4"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(combined_video)])

        # Mux in the track's audio
        print("Muxing audio...")
        with_fallback([track_raw], lambda: run([
            "ffmpeg", "-y", "-i", str(combined_video), *sources.input_args(track_raw),
            "-map", "0:v", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            output_path,
        ]))

    print(f"\nDone: {output_path}")
    return {"streamed": len(sources.streamed - sources.downloaded),
            "downloaded": len(sources.downloaded),
            "local": len(local_at_start)}


def render_plan(plan_path: str, output_path: str = "preview_output.mp4"):
    with open(plan_path) as f:
        plan = json.load(f)
    render_plan_dict(plan, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python render_preview.py "path\\to\\plan.json"')
        sys.exit(1)
    render_plan(sys.argv[1])
