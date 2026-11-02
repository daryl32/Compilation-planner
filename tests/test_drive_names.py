"""Drive file lookup copes with the same name written another way (accents, case)."""

import unicodedata

from tests.support import DATA  # noqa: F401  (installs the test config first)


class _Req:
    def __init__(self, out):
        self.out = out

    def execute(self, **_):
        return self.out


class _Files:
    def __init__(self, names):
        self.names = names

    def list(self, q="", **_):
        if "mimeType='application/vnd.google-apps.folder'" in q and "name=" in q:
            return _Req({"files": [{"id": "folder1", "name": "Audio-Library"}]})
        if q.startswith("name='"):
            want = q.split("name='", 1)[1].split("' and", 1)[0].replace("\\\\'", "'")
            return _Req({"files": [{"id": n, "name": n, "size": "10"} for n in self.names if n == want]})
        return _Req({"files": [{"id": n, "name": n, "size": "10"} for n in self.names]})


class _Service:
    def __init__(self, names):
        self._f = _Files(names)

    def files(self):
        return self._f


def test_find_drive_file_matches_accents_written_either_way():
    import drive_sync as DS
    nfd = unicodedata.normalize("NFD", "01 NO BATIDÃO.m4a")          # how an iPhone may have saved it
    nfc = unicodedata.normalize("NFC", "01 NO BATIDÃO.m4a")          # how the app asks for it
    assert nfd != nfc
    f, err = DS.find_drive_file(f"Audio-Library/{nfc}", _Service([nfd, "other.m4a"]))
    assert err is None and f["name"] == nfd
    f, err = DS.find_drive_file("Audio-Library/01 no batidão.M4A", _Service([nfd]))
    assert err is None
    f, err = DS.find_drive_file("Audio-Library/01 NO BATIDAO.m4a", _Service([nfd, "zzz.m4a"]))
    assert f is None and "closest there" in err and "BATID" in err
    f, err = DS.find_drive_file("Audio-Library/nothing like it.m4a", _Service([nfd]))
    assert f is None and "closest" not in err


def test_find_drive_file_accepts_missing_or_different_extension():
    import drive_sync as DS
    # The Drive copy lost its extension, and the render asks for the .mp4 name
    f, err = DS.find_drive_file("Videos-PH/Eva_Elfie1.mp4", _Service(["Eva_Elfie1", "Eva_Elfie2.mp4"]))
    assert err is None and f["name"] == "Eva_Elfie1"
    f, err = DS.find_drive_file("Videos-PH/clip.mp4", _Service(["clip.mov"]))
    assert err is None and f["name"] == "clip.mov"
    # Two candidates → no guess
    f, err = DS.find_drive_file("Videos-PH/clip.mp4", _Service(["clip", "clip.mov"]))
    assert f is None
    # A near-but-different name still isn't taken
    f, err = DS.find_drive_file("Videos-PH/Eva_Elfie1.mp4", _Service(["Eva_Elfie11"]))
    assert f is None
