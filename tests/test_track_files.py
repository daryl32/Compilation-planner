"""A track's file is found whichever way its accents are stored (track_files.track_file)."""

import json
import unicodedata

from tests.support import DATA  # noqa: F401  (installs the test config first)


def test_track_file_either_form(tmp_path):
    from track_files import track_file
    nfd = unicodedata.normalize("NFD", "01 NO BATIDÃO")
    nfc = unicodedata.normalize("NFC", "01 NO BATIDÃO")
    (tmp_path / f"{nfd}.json").write_text(json.dumps({"track_id": nfc}))
    assert track_file(nfc, tmp_path).name == f"{nfd}.json"          # asked one way, stored the other
    assert track_file(nfd, tmp_path).name == f"{nfd}.json"
    assert track_file("plain", tmp_path) == tmp_path / "plain.json"  # missing: the plain path, to create
    (tmp_path / "plain.json").write_text("{}")
    assert track_file("plain", tmp_path).exists()                    # a new file shows up (folder changed)


def test_media_library_reads_a_track_stored_the_other_way(tmp_path, monkeypatch):
    """The Audio tab's ✕ Analysis: the track_id inside is NFC, the file name NFD."""
    from tests.support import load_page
    ns = load_page("pages/3_Media_Library.py", {})
    nfd = unicodedata.normalize("NFD", "01 NO BATIDÃO")
    nfc = unicodedata.normalize("NFC", "01 NO BATIDÃO")
    (tmp_path / f"{nfd}.json").write_text(json.dumps({"track_id": nfc, "duration_sec": 90}))
    monkeypatch.setitem(ns, "AUDIO_DIR", tmp_path)
    assert ns["_load_track"](nfc, ns["track_file"](nfc, tmp_path).stat().st_mtime)["duration_sec"] == 90
