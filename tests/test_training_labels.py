"""🎯 Training Labels: the tag vocabulary file, picking cards, and saving answers
(training_labels.py)."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)

import training_labels as TL


@pytest.fixture
def clean_files(tmp_path):
    """Run with no vocabulary / labels file and an untouched pending list; restore afterwards."""
    from library_common import PENDING_FILE
    saved = {}
    for p in (TL.VOCAB_FILE, TL.LABELS_FILE, PENDING_FILE):
        if p.exists():
            saved[p] = tmp_path / p.name
            shutil.copy2(p, saved[p])
            p.unlink()
    yield
    for p in (TL.VOCAB_FILE, TL.LABELS_FILE, PENDING_FILE):
        p.unlink(missing_ok=True)
        if p in saved:
            shutil.copy2(saved[p], p)


def _videos():
    import catalogue_store as CS
    return CS.video_index()


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

CSV = ("﻿tag,group,kind,one_of,teacher,description,not_to_be_confused_with,search_words\n"
       "close-up shot,shot,frame,shot size,yes,Face fills the frame,wide shot,close up; tight\n"
       "wide shot,shot,frame,shot size,no,,,\n"
       "dance,action,motion,,,People dancing,,\n"
       "Dance,action,frame,,,duplicate,,\n"
       ",,,,,empty row,,\n"
       "night,setting,odd-kind,,maybe,,,\n")


def test_parse_vocabulary_reads_excel_csv():
    rows = TL.parse_vocabulary(CSV)
    assert [r["tag"] for r in rows] == ["close-up shot", "wide shot", "dance", "night"]
    cu = rows[0]
    assert cu["kind"] == "frame" and cu["one_of"] == "shot size" and cu["teacher"] is True
    assert cu["search_words"] == ["close up", "tight"]
    assert rows[1]["teacher"] is False
    assert rows[2]["kind"] == "motion" and rows[2]["teacher"] is True      # blank teacher → yes
    assert rows[3]["kind"] == "frame" and rows[3]["teacher"] is False      # unknown kind → frame


def test_parse_vocabulary_semicolon_separated():
    text = "tag;group;kind;one_of;teacher\nstreet;place;frame;;yes\n"
    assert [r["tag"] for r in TL.parse_vocabulary(text)] == ["street"]


def test_vocabulary_falls_back_to_labels_json(clean_files):
    rows, source = TL.load_vocabulary()
    assert source == "labels.json"
    assert [r["tag"] for r in rows] == ["dance", "street", "crowd", "night"]
    assert all(r["kind"] == "frame" and r["teacher"] for r in rows)


def test_save_vocabulary_round_trip_and_queues_drive(clean_files):
    from library_common import read_pending, _pending_target
    n = TL.save_vocabulary(TL.parse_vocabulary(CSV))
    assert n == 4
    rows, source = TL.load_vocabulary()
    assert source == "tag_vocabulary.csv"
    assert rows == TL.parse_vocabulary(CSV)
    assert TL.VOCAB_FILE.read_bytes().startswith("﻿".encode("utf-8"))   # Excel-friendly
    assert TL.VOCAB_PENDING in read_pending()
    path, folder = _pending_target(TL.VOCAB_PENDING)
    assert path == TL.VOCAB_FILE and folder == "scene-labeling/catalogue"


def test_chip_groups():
    rows = TL.parse_vocabulary(CSV)
    groups = TL.chip_groups(rows, extra=["street", "dance"])
    assert groups[0] == ("shot size", True, ["close-up shot", "wide shot"])
    assert ("action", False, ["dance"]) in groups
    assert groups[-1] == ("Not in the vocabulary", False, ["street"])


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_save_replace_remove_and_summary(clean_files):
    import catalogue_store as CS
    from library_common import read_pending
    card = {"id": TL.card_id("tags", "vid_a", 3.5), "type": "tags", "video_id": "vid_a",
            "scene_id": 0, "t": 3.5, "suggested": ["dance"]}
    TL.save_label(card, tags=["dance", "night"])
    TL.save_label(card, tags=["dance"])                      # same card again → replaced
    b = {"id": TL.card_id("boundary", "vid_b", 9.0), "type": "boundary", "video_id": "vid_b",
         "scene_id": 2, "t": 9.0, "source": "peak"}
    TL.save_label(b, verdict="cut")
    s = {"id": TL.card_id("tags", "vid_c", 1.0), "type": "tags", "video_id": "vid_c",
         "scene_id": 0, "t": 1.0, "suggested": []}
    TL.save_label(s, skipped=True, reason="blurry")
    items = TL.load_labels()
    assert len(items) == 3
    assert items[0]["tags"] == ["dance"] and items[0]["suggested"] == ["dance"]
    assert items[1]["verdict"] == "cut" and items[1]["source"] == "peak"
    assert items[2]["skipped"] and items[2]["reason"] == "blurry"
    assert TL.LABELS_STEM in read_pending()
    sm = TL.summary(items)
    assert (sm["tags"], sm["boundaries"], sm["cuts"], sm["skipped"], sm["changed"]) == (1, 1, 1, 1, 0)
    assert TL.remove_label(b["id"]) and not TL.remove_label(b["id"])
    assert len(TL.load_labels()) == 2
    # Never mistaken for a video catalogue.
    assert TL.LABELS_FILE not in CS.catalogue_paths()
    with pytest.raises(ValueError):
        TL.save_label(b, verdict="maybe")


# ---------------------------------------------------------------------------
# Picking cards
# ---------------------------------------------------------------------------

def test_tag_cards_inside_usable_scenes():
    videos = _videos()
    by_vid = {v["video_id"]: v for v in videos.values()}
    cards = TL.pick_tag_cards(videos, 20, [], seed=7)
    assert len(cards) == 20
    assert len({c["video_id"] for c in cards}) >= 4                 # spread across videos
    for c in cards:
        sc = next(s for s in by_vid[c["video_id"]]["scenes"] if s["scene_id"] == c["scene_id"])
        assert not sc["excluded"]
        assert sc["start"] + TL.EDGE_SEC - 1e-6 <= c["t"] <= sc["end"] - TL.EDGE_SEC + 1e-6
        assert c["suggested"] == sc["tags"]
    assert cards == TL.pick_tag_cards(videos, 20, [], seed=7)       # repeatable with a seed


def test_tag_cards_respect_range_media_and_earlier_labels():
    videos = _videos()
    cards = TL.pick_tag_cards(videos, 10, [], {"vid_a": (5.0, 20.0)}, seed=3,
                              has_media=lambda v: v == "vid_a")
    assert cards and all(c["video_id"] == "vid_a" and 5.0 <= c["t"] <= 20.0 for c in cards)
    done = [{"id": "x", "type": "tags", "video_id": c["video_id"], "t": c["t"]} for c in cards]
    again = TL.pick_tag_cards(videos, 10, done, {"vid_a": (5.0, 20.0)}, seed=3,
                              has_media=lambda v: v == "vid_a")
    for c in again:
        assert all(abs(c["t"] - d["t"]) >= 3.0 for d in done)
    assert TL.pick_tag_cards(videos, 5, [], has_media=lambda v: False) == []


def test_boundary_cards():
    import catalogue_store as CS
    videos = _videos()
    cards = TL.pick_boundary_cards(videos, 12, [], seed=5, curves=CS.motion_curves)
    assert len(cards) == 12
    assert {c["source"] for c in cards} >= {"detected", "random"}
    starts = {(v["video_id"], s["start"]) for v in videos.values() for s in v["scenes"]}
    for c in cards:
        if c["source"] == "detected":
            assert (c["video_id"], c["t"]) in {(v, round(t, 2)) for v, t in starts}


def test_peak_points_in_long_scenes(monkeypatch):
    videos = {"v": {"video_id": "v", "scenes": [
        {"scene_id": 0, "start": 0.0, "end": 30.0, "tags": [], "excluded": False},
        {"scene_id": 1, "start": 30.0, "end": 40.0, "tags": [], "excluded": False}]}}
    values = [0.1] * 300
    values[120] = 9.0                                              # sharp change at 12 s (10 fps)
    curves = lambda vid: {0: {"fps": 10.0, "values": values}}
    cards = TL.pick_boundary_cards(videos, 3, [], seed=1, curves=curves)
    peaks = [c for c in cards if c["source"] == "peak"]
    assert peaks and peaks[0]["t"] == 12.0


def test_sharpest_change():
    assert TL.sharpest_change([0, 0, 0, 5, 0, 0, 0, 0], 2.0, skip_sec=1.0) == 1.5
    assert TL.sharpest_change([1, 2], 10.0) is None


def test_build_deck_mix():
    videos = _videos()
    mixed = TL.build_deck(videos, 10, "mixed", [], seed=2)
    assert sum(c["type"] == "boundary" for c in mixed) == 3 and len(mixed) == 10
    assert all(c["type"] == "tags" for c in TL.build_deck(videos, 6, "tags", [], seed=2))
    assert all(c["type"] == "boundary" for c in TL.build_deck(videos, 6, "boundary", [], seed=2))


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def test_extract_frame(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    mp4 = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=10:duration=3", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    out = TL.extract_frame(mp4, 1.5)
    assert out and out.stat().st_size > 0
    assert TL.extract_frame(mp4, 1.5) == out                       # cached
    assert TL.extract_frame(tmp_path / "missing.mp4", 1.0) is None


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

@pytest.fixture
def proxy_for_vid_a(clean_files):
    """A test-pattern preview copy for vid_a (and nothing else), removed afterwards."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    from library_common import proxy_path
    import catalogue_store as CS
    dur = max(s["end"] for s in CS.video_index()["vid_a"]["scenes"]) + 1
    p = proxy_path("vid_a")
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not existed:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc=size=320x180:rate=10:duration={dur:.0f}", "-pix_fmt", "yuv420p", str(p)],
                       check=True)
    yield p
    if not existed:
        p.unlink(missing_ok=True)


def test_page_without_preview_copies(clean_files):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py").run()
    assert not errors(at), errors(at)
    assert any("No cards to show" in i.value for i in at.info)


def test_page_label_flow(proxy_for_vid_a):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_mix"] = "tags"
    at.session_state["tl_size"] = 10
    at.run()
    assert not errors(at), errors(at)
    deck = at.session_state["tl_deck"]
    assert deck and all(c["video_id"] == "vid_a" for c in deck)
    first = deck[0]
    save = next(b for b in at.button if b.label == "✅ Save & next")
    save.click().run()
    assert not errors(at), errors(at)
    items = TL.load_labels()
    assert [i["id"] for i in items] == [first["id"]]
    assert items[0]["tags"] == [t for t in first["suggested"] if t in {"dance", "street", "crowd", "night"}]
    assert at.session_state["tl_pos"] == 1
    # ↩️ Back undoes it.
    next(b for b in at.button if b.label == "↩️ Back").click().run()
    assert TL.load_labels() == [] and at.session_state["tl_pos"] == 0

    # Scene-change cards.
    at.session_state["tl_mix"] = "boundary"
    at.run()
    assert not errors(at), errors(at)
    card = at.session_state["tl_deck"][0]
    assert card["type"] == "boundary"
    next(b for b in at.button if b.label == "➖ No, same scene").click().run()
    assert TL.load_labels()[0]["verdict"] == "no_cut"


@pytest.mark.parametrize("view", ["📊 Progress", "📒 Vocabulary"])
def test_page_other_views(view, clean_files):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = view
    at.run()
    assert not errors(at), errors(at)
    if view == "📒 Vocabulary":
        next(b for b in at.button if b.label == "💾 Save vocabulary").click().run()
        assert not errors(at), errors(at)
        rows, source = TL.load_vocabulary()
        assert source == "tag_vocabulary.csv" and [r["tag"] for r in rows] == ["dance", "street", "crowd", "night"]
