"""🧹 audio_tidy: the server's track files that Google Drive no longer has; 🔄 the Server page's
refresh-the-test-copy panel."""

from __future__ import annotations

import json
import unicodedata

from tests.support import DATA  # noqa: F401  (installs the test config first)


def _write(d, stem, song):
    (d / f"{stem}.json").write_text(json.dumps({"track_id": stem,
                                                "source_path": f"/content/drive/MyDrive/Audio-Library/{song}"}))


def test_find_and_remove_leftovers(tmp_path):
    import audio_tidy as AT
    nfc = unicodedata.normalize("NFC", "01 NO BATIDÃO")
    nfd = unicodedata.normalize("NFD", "01 NO BATIDÃO")
    _write(tmp_path, nfc, nfc + ".m4a")
    _write(tmp_path, nfd, nfd + ".m4a")                                  # the same track, spelt the other way
    _write(tmp_path, "01 The Girl Is Mine (feat. Destiny's", "01 The Girl Is Mine (feat. Destiny's.m4a")
    _write(tmp_path, "01 The Girl Is Mine (feat. Destinys", "01 The Girl Is Mine (feat. Destinys.m4a")
    _write(tmp_path, "Old song", "Old song.m4a")                         # on Drive, song deleted
    _write(tmp_path, "Unpushed mix", "Unpushed mix.m4a")                 # not on Drive yet — pending push
    (tmp_path / "audio_index.json").write_text("{}")
    drive = {f"{nfc}.json", "01 The Girl Is Mine (feat. Destiny's.json", "Old song.json"}
    songs = [f"Audio-Library/{nfc}.m4a", "Audio-Library/01 The Girl Is Mine (feat. Destiny's.m4a"]
    found = AT.find_leftovers(tmp_path, drive, songs, {"audio:Unpushed mix"})
    by = {f["track_id"]: f for f in found}
    assert set(by) == {nfd, "01 The Girl Is Mine (feat. Destinys", "Old song"}
    assert by[nfd]["twin"] == nfc and "same name" in by[nfd]["reason"]
    assert "no longer in Audio-Library" in by["Old song"]["reason"]
    # No Drive listing → nothing is called a leftover.
    assert AT.find_leftovers(tmp_path, set(), songs, set()) == []
    dest = AT.remove(tmp_path, [f["file"] for f in found] + ["../escape.json", "audio_index.json.nope"])
    assert sorted(p.name for p in dest.iterdir()) == sorted(f["file"] for f in found)
    left = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert left == sorted([nfc, "01 The Girl Is Mine (feat. Destiny's", "Unpushed mix", "audio_index"])
    assert dest.parent.name == AT.REMOVED_DIR


def test_media_library_tidy_popover(monkeypatch):
    import audio_tidy as AT
    import library_common as LC
    monkeypatch.setattr(LC, "IS_TEST", False)          # the live app only
    from tests.apptest_util import errors, page_app
    monkeypatch.setattr(AT, "drive_listing", lambda: ({"nothing.json"}, [], None))
    at = page_app("pages/3_Media_Library.py", timeout=300)
    at.session_state["lib_kind"] = "🎵 Audio"
    at.run()
    assert not errors(at), errors(at)
    btn = [b for b in at.button if b.key == "aud_tidy_check"]
    assert btn
    btn[0].click().run()
    assert not errors(at), errors(at)
    res = at.session_state["aud_tidy"]
    assert res["found"]                                      # every test track is "not on Drive" here
    assert at.button(key="aud_tidy_go").label.startswith("🧹 Remove")


def test_server_page_refresh_panel(monkeypatch):
    import library_common as LC
    from tests.apptest_util import errors, page_app
    monkeypatch.setattr(LC, "IS_TEST", False)          # the live app only
    at = page_app("pages/4_Server.py", timeout=120)
    at.run()
    assert not errors(at), errors(at)
    assert any("Refresh the test copy" in h.value for h in at.subheader)


def test_test_copy_shows_neither():
    from tests.apptest_util import errors, page_app
    at = page_app("pages/4_Server.py", timeout=120)
    at.run()
    assert not errors(at), errors(at)
    assert not any("Refresh the test copy" in h.value for h in at.subheader)
