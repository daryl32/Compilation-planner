"""🧍 Avatar Studio in the app (19 Nov 2026): where lines play, what's recorded per look, wrong looks, Colab lists."""
from __future__ import annotations

import pytest
from pathlib import Path

from tests.support import DATA  # noqa: F401  (installs the test config first)

CLIPS = [
    {"id": "a", "name": "Start-Round 1.mp4", "size": 1, "folder": "Default - Lying on bed 1/avatar_clips"},
    {"id": "b", "name": "Round 4.mp4", "size": 1, "folder": "Nude - Lying Back/avatar_clips"},          # wrong look
    {"id": "c", "name": "Get hard for me.mp4", "size": 1, "folder": "avatar_clips"},                      # transparent
    {"id": "d", "name": "nude_lying-back_idle_01.mp4", "size": 1, "folder": "Nude - Lying Back"},        # not a voice clip
]
LOOKS = {"intro": "Default - Lying on bed 1", "1": "Default - Lying on bed 1", "2": "Default - Lying on bed 1",
         "3": "Black Panties - Sitting on Bed1", "4": "Black Panties - Sitting on Bed1", "5": "Nude - Lying Back",
         "6": "Nude - Lying Back", "last": "Nude - Lying Back", "outro": "Nude - Lying on Back"}


def _cm():
    return {"lines": [
        {"name": "Start-Round 1", "file": "Start-Round 1.mp3", "seconds": 19.8, "trim_out": 7.8,
         "cues": [{"type": "phase", "phases": ["wait"], "roundList": ["1"]}]},
        {"name": "Round 4", "file": "Round 4.mp3", "seconds": 3.0, "cues": [{"type": "phase", "phases": ["slow"], "roundList": ["4"]}]},
        {"name": "Get hard for me", "file": "Get hard for me.mp3", "seconds": 5.0,
         "cues": [{"type": "phase", "phases": ["wait"], "roundList": ["1", "2"]}]},
        {"name": "Keep going", "file": "Keep going.mp3", "seconds": 1.5, "cats": ["Filler"],
         "cues": [{"type": "random", "phases": ["fast"], "roundList": []}]},
        {"name": "Hands off 5 A", "file": "Hands off 5 A.mp3", "seconds": 4.9, "cues": [{"type": "surprise", "kind": "handsoff"}]},
        {"name": "Welcome", "file": "Welcome.mp3", "seconds": 3.1, "cues": [{"type": "intro", "order": 1}]},
        {"name": "Again", "file": "Again.mp3", "seconds": 2.0, "cues": [{"type": "after", "after": "Welcome"}]},
        {"name": "No cue", "file": "No cue.mp3", "seconds": 2.0, "cues": []},
    ]}


def test_where_lines_play_and_what_is_recorded():
    import studio as SD
    cm = _cm()
    rows = {r["name"]: r for r in SD.tracker(cm, LOOKS, CLIPS)}
    assert rows["Start-Round 1"]["parts"] == ["1"] and rows["Start-Round 1"]["needs"] == {"Default - Lying on bed 1": "✅"}
    assert rows["Get hard for me"]["needs"] == {"Default - Lying on bed 1": "🟡"}               # transparent only
    assert rows["Round 4"]["needs"] == {"Black Panties - Sitting on Bed1": "❌"}
    assert [w["look"] for w in rows["Round 4"]["wrong"]] == ["Nude - Lying Back"]
    assert set(rows["Keep going"]["parts"]) == set(SD.ROUND_PARTS) and rows["Keep going"]["kind"] == "filler"
    assert "1" not in rows["Hands off 5 A"]["parts"] and rows["Hands off 5 A"]["kind"] == "surprise"
    assert rows["Again"]["parts"] == ["intro"]                                                 # follows Welcome
    assert rows["No cue"]["needs"] == {}
    sess = {s["look"]: s for s in SD.sessions(list(rows.values()), LOOKS)}
    d = sess["Default - Lying on bed 1"]
    assert [r["name"] for r in d["main"]] == ["Again", "Welcome", "Get hard for me"]          # shortest first, 🟡 counted
    assert [r["name"] for r in d["extra"]] == ["Keep going", "Hands off 5 A"]              # surprises: round 2 on
    assert d["done"] == 1 and d["stand_ins"] == 1
    assert SD.sessions(list(rows.values()), LOOKS, with_stand_in=False)[0]["stand_ins"] == 0
    txt = SD.to_record_text(d)
    body = [ln for ln in txt.splitlines() if ln and not ln.startswith("# ")]
    assert body == ["Again.mp3", "Welcome.mp3", "Get hard for me.mp3", "Keep going.mp3", "Hands off 5 A.mp3"]
    assert SD.units(7.9) == pytest.approx(0.52) and SD.units(8.1) == pytest.approx(1.04)
    assert SD.drive_looks(CLIPS) == ["Default - Lying on bed 1", "Nude - Lying Back"]


def test_the_page_opens(tmp_path, monkeypatch):
    import json
    import planner.edge as PE
    import studio as SD
    from tests.apptest_util import errors, page_app
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "v")
    at = page_app("pages/9_Avatar_Studio.py").run()
    assert not errors(at) and any("Read Drive" in i.value for i in at.info)
    SD.cache_path().parent.mkdir(parents=True, exist_ok=True)
    SD.cache_path().write_text(json.dumps({"at": "now", "clips": CLIPS}))
    at = page_app("pages/9_Avatar_Studio.py").run()
    assert not errors(at)
    assert any("Default - Lying on bed 1" in e.label for e in at.expander)
    # 🎞️ previews (moved from Edge's settings, 21 Nov): the count, and the button needs the Drive connection
    assert at.button(key="studio_make").disabled
    assert any("Previews:" in c.value for c in at.caption)
    # 🗓️ the round plan shows which looks are on the server (moved from Edge's 👗 panel)
    assert any("Sync now" in c.value for c in at.caption) or any("Sync now" in w.value for w in at.warning)


def test_sync_brings_only_new_and_changed_lines(tmp_path, monkeypatch):
    """🔄 Sync now (21 Nov): Edge's lines and her clips — new files and re-recordings (a different size) only."""
    import drive_sync
    import planner.edge as PE
    monkeypatch.setattr(PE, "voice_dir", lambda: tmp_path / "v")
    monkeypatch.setattr(PE, "avatar_dir", lambda: tmp_path / "a")
    monkeypatch.setattr(PE, "looks_root", lambda: tmp_path / "looks")
    PE.lines_dir().mkdir(parents=True, exist_ok=True)
    (PE.lines_dir() / "Same.mp3").write_bytes(b"x" * 10)
    (PE.lines_dir() / "Changed.mp3").write_bytes(b"x" * 10)
    folder = "scene-labeling/Edge_voiceover"
    listing = {folder: [{"name": n, "size": str(sz)} for n, sz in (("Same.mp3", 10), ("Changed.mp3", 12), ("New.mp3", 5))],
               f"{folder}/avatar_clips": [{"name": "New.mp4", "size": "7"}]}
    got = []
    monkeypatch.setattr(drive_sync, "credentials_available", lambda: True)
    monkeypatch.setattr(drive_sync, "list_drive_folder", lambda p: (listing.get(p, []), None))

    def fake_download(src, dest):
        got.append(Path(src).name)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"y")
    monkeypatch.setattr(drive_sync, "download_source_video", fake_download)
    cm = {"lines": [{"name": n, "cues": []} for n in ("Same", "Changed", "New", "Not recorded")]}
    stats = {}
    errs = PE.fetch_lines(cm, avatar=True, changed_only=True, stats=stats)
    assert not errs, errs
    assert sorted(got) == ["Changed.mp3", "New.mp3", "New.mp4"]
    assert stats == {"lines": 2, "clips": 1, "skipped": 1}
