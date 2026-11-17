"""🧍 Avatar Studio previews: the small copy, the 3-frame strip, and the job (Drive faked)."""

import subprocess
from pathlib import Path

import studio_previews as SP


def _clip(path: Path, alpha: bool = False):
    if alpha:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red@0.5:s=640x360:d=2,format=yuva420p",
                        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", str(path)], check=True)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=720x1280:d=2:r=25", "-f", "lavfi",
                        "-i", "sine=d=2", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)], check=True)


def _wh(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return r.stdout.strip()


def test_preview_is_small_h264_and_the_strip_has_three_frames(tmp_path):
    from PIL import Image
    src = tmp_path / "a.mp4"
    _clip(src)
    mp4, jpg = tmp_path / "p.mp4", tmp_path / "p.jpg"
    assert SP.make_preview(src, mp4) is None and SP.make_strip(mp4, jpg) is None
    assert _wh(mp4) == "h264,540,960"
    w, h = Image.open(jpg).size
    assert w == 3 * SP.THUMB_W and h == 284


def test_transparent_webm_is_laid_on_grey(tmp_path):
    from PIL import Image
    src = tmp_path / "b.webm"
    _clip(src, alpha=True)
    mp4, jpg = tmp_path / "p.mp4", tmp_path / "p.jpg"
    assert SP.make_preview(src, mp4) is None and SP.make_strip(mp4, jpg) is None
    assert _wh(mp4).startswith("h264,540,")
    r, g, b = Image.open(jpg).convert("RGB").getpixel((80, 40))
    assert 120 < r < 220 and g < 60           # half-red over grey, not solid red


def test_job_skips_done_clips_and_uploads_both_files(tmp_path, monkeypatch):
    import sys
    import types
    import drive_sync
    src = tmp_path / "src.mp4"
    _clip(src)
    found = [{"id": "A1", "name": "x.mp4", "size": 1, "folder": "transitions"},
             {"id": "B2", "name": "y.mp4", "size": 1, "folder": "sitting/avatar_clips"}]
    monkeypatch.setattr(SP, "clips", lambda: (found, None))
    monkeypatch.setattr(SP, "done_ids", lambda: {"A1"})
    monkeypatch.setattr(drive_sync, "download_drive_file", lambda fid, dest: Path(dest).write_bytes(src.read_bytes()))
    sent = []
    fake = types.ModuleType("drive_oauth")          # (no OAuth client config on a test machine)
    fake.upload_file_with_oauth = lambda tok, f, where: (sent.append((Path(f).name, where)), ({"id": "n"}, None))[1]
    monkeypatch.setitem(sys.modules, "drive_oauth", fake)

    class J:
        def progress(self, *a):
            pass
    out = SP.run(J(), {"t": 1})
    assert out == {"clips": 2, "made": 1, "skipped": 1, "errors": []}
    assert sent == [("B2.mp4", SP.PREVIEW_FOLDER), ("B2.jpg", SP.PREVIEW_FOLDER)]
