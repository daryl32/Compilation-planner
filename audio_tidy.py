"""🧹 Tidy the audio list — the server's track files that Google Drive no longer has.

☁️ Sync only ever adds and updates files on the server, never removes them, so a track renamed,
re-added or deleted on Drive leaves its old analysis file behind and the app keeps listing it — a
duplicate (e.g. "01 NO BATIDÃO" spelt two ways, an old copy without an apostrophe).

find_leftovers() lists, for every <track>.json in AUDIO_DIR:
  • "not on Drive"   — Drive's scene-labeling/audio_catalogue has no file of exactly that name
                       (byte for byte — so an accent spelt the other way counts as gone);
  • "song gone"      — it's on Drive, but its song (source_path) is no longer in the audio library
                       folder (the song was renamed or deleted there).
Tracks with edits not yet pushed to Drive (the ⬆️ pending list) are never listed.

remove() moves the picked files into AUDIO_DIR/.removed/<date>/ — out of the app, but recoverable
(move one back to restore it). Nothing on Google Drive is changed.
"""

from __future__ import annotations

import datetime
import json
import shutil
import unicodedata
from pathlib import Path

REMOVED_DIR = ".removed"
SKIP = {"audio_index.json"}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def _song_rel(track: dict) -> str | None:
    sp = str(track.get("source_path") or "")
    return sp.split("MyDrive/", 1)[1] if "MyDrive/" in sp else None


def find_leftovers(audio_dir: Path, drive_json_names: set, drive_song_rels: list, pending: set,
                   library_folder: str = "Audio-Library") -> list:
    """[{"file", "track_id", "reason", "twin"}] — see the module docstring.
    drive_json_names: file names in Drive's audio_catalogue; drive_song_rels: "Audio-Library/x.m4a"
    paths of the songs on Drive; pending: library_common.read_pending()."""
    if not drive_json_names:
        return []        # no Drive listing to compare with — never treat everything as a leftover
    songs = {_nfc(r) for r in drive_song_rels}
    files = sorted(p for p in Path(audio_dir).glob("*.json") if p.name not in SKIP)
    tracks = {}
    for p in files:
        try:
            tracks[p] = json.loads(p.read_text())
        except (OSError, ValueError):
            tracks[p] = {}
    out = []
    for p, t in tracks.items():
        stem = p.stem
        if f"audio:{stem}" in pending or stem in pending:
            continue
        twin = next((q.stem for q in tracks if q != p and _nfc(q.stem) == _nfc(stem)
                     and q.name in drive_json_names), None)
        if p.name not in drive_json_names:
            out.append({"file": p.name, "track_id": stem, "twin": twin,
                        "reason": "not on Google Drive any more" + (f" — the same name as {twin}" if twin else "")})
            continue
        rel = _song_rel(t)
        if rel and rel.split("/", 1)[0] == library_folder and _nfc(rel) not in songs and drive_song_rels:
            out.append({"file": p.name, "track_id": stem, "twin": None,
                        "reason": f"its song ({rel.rsplit('/', 1)[-1]}) is no longer in {library_folder}"})
    return out


def remove(audio_dir: Path, names: list) -> Path:
    """Move these track files into AUDIO_DIR/.removed/<date time>/ (recoverable). Returns that folder."""
    dest = Path(audio_dir) / REMOVED_DIR / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = Path(audio_dir) / name
        if src.exists() and src.parent == Path(audio_dir) and src.suffix == ".json":
            shutil.move(str(src), str(dest / name))
    return dest


def drive_listing():
    """(json names in Drive's audio_catalogue folder(s), [song rels], error or None) — the service account."""
    try:
        import drive_sync as DS
        service = DS._get_service()
        scene_id = DS._find_folder(service, DS.DRIVE_SCENE_LABELING)
        if not scene_id:
            return set(), [], f"'{DS.DRIVE_SCENE_LABELING}' not found on Google Drive"
        names = set()
        for fid in [f["id"] for f in DS._find_folders(service, "audio_catalogue", scene_id)]:
            names |= {f["name"] for f in DS._list_files(service, fid) if f["name"].endswith(".json")}
        import config
        songs, err = DS.list_drive_audio(getattr(config, "AUDIO_LIBRARY_DRIVE_FOLDER", "Audio-Library"))
        if err:
            return names, [], err
        return names, [s["rel"] for s in songs], None
    except Exception as e:
        return set(), [], str(e)
