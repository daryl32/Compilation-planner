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
        next(b for b in at.button if b.label == "💾 Save table").click().run()
        assert not errors(at), errors(at)
        rows, source = TL.load_vocabulary()
        assert source == "tag_vocabulary.csv" and [r["tag"] for r in rows] == ["dance", "street", "crowd", "night"]


def test_upsert_rename_delete_tag():
    rows = TL.parse_vocabulary(CSV)
    rows = TL.upsert_tag(rows, {"tag": "kissing", "kind": "motion", "group": "action"})
    assert rows[-1]["tag"] == "kissing" and rows[-1]["kind"] == "motion"
    rows = TL.upsert_tag(rows, {"tag": "close up", "one_of": "shot size", "description": "x"}, "close-up shot")
    assert [r["tag"] for r in rows][:2] == ["close up", "wide shot"]
    with pytest.raises(ValueError):
        TL.upsert_tag(rows, {"tag": "Wide Shot"}, "close up")   # renaming onto another tag
    with pytest.raises(ValueError):
        TL.upsert_tag(rows, {"tag": "  "})
    assert "kissing" not in [r["tag"] for r in TL.delete_tag(rows, "KISSING")]


def test_rename_in_labels(clean_files):
    card = {"id": TL.card_id("tags", "vid_a", 2.0), "type": "tags", "video_id": "vid_a", "scene_id": 0,
            "t": 2.0, "suggested": ["dance"]}
    TL.save_label(card, tags=["dance", "night"])
    assert TL.rename_in_labels("Dance", "dancing") == 1
    item = TL.load_labels()[0]
    assert item["tags"] == ["dancing", "night"] and item["suggested"] == ["dancing"]


def test_tag_form_adds_renames_and_deletes(clean_files):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "📒 Vocabulary"
    at.run()
    assert not errors(at), errors(at)
    # ➕ New tag
    at.text_input(key="tl_ed_➕ New tag_name").set_value("kissing")
    at.text_area(key="tl_ed_➕ New tag_desc").set_value("Two people kissing")
    next(b for b in at.button if b.label == "➕ Add tag").click().run()
    assert not errors(at), errors(at)
    rows, _ = TL.load_vocabulary()
    assert rows[-1]["tag"] == "kissing" and rows[-1]["description"] == "Two people kissing"
    assert at.session_state["tl_ed_pick"] == "kissing"
    # rename it
    at.text_input(key="tl_ed_kissing_name").set_value("kiss")
    next(b for b in at.button if b.label == "💾 Save tag").click().run()
    assert not errors(at), errors(at)
    assert "kiss" in [r["tag"] for r in TL.load_vocabulary()[0]]
    assert any("Renamed" in s.value for s in at.success)
    # delete it
    next(b for b in at.button if b.label == "🗑️ Delete tag").click().run()
    assert not errors(at), errors(at)
    assert "kiss" not in [r["tag"] for r in TL.load_vocabulary()[0]]


def test_never_with_column_and_rename():
    rows = TL.parse_vocabulary("tag,group,never_with\nstanding,position,\nlying,position,\n"
                               "jumping,action,\"standing, lying\"\n")
    assert rows[2]["never_with"] == "standing; lying"
    assert TL.never_pairs(rows)["standing"] == {"jumping"}
    assert TL.rule_breaks(["jumping", "dancing", "standing"], rows) == [("jumping", "standing")]
    assert TL.rule_breaks(["standing"], rows) == []
    rows = TL.upsert_tag(rows, {**rows[0], "tag": "stood"}, "standing")      # rename follows into never_with
    assert rows[2]["never_with"] == "stood; lying"
    assert "never_with" in TL.vocabulary_csv(rows).splitlines()[0]
    again = TL.parse_vocabulary(TL.vocabulary_csv(rows))
    assert again[2]["never_with"] == "stood; lying"


def test_always_with_problems_and_fixes(clean_files):
    rows = TL.parse_vocabulary("tag,group,one_of,never_with,always_with\n"
                               "close-up,shot,shot size,,\nwide,shot,shot size,,\n"
                               "kissing,action,,,close-up\nface,body,,,\nlips,body,,,face\n"
                               "talking,action,,,lips\nstanding,position,,,\nlying,position,,standing,\n")
    assert rows[2]["always_with"] == "close-up"
    probs = TL.rule_problems(["wide", "close-up", "kissing", "lying", "standing", "talking"], rows)
    assert "“wide” and “close-up” are all in the pick-one set “shot size”" in probs
    assert "“lying” and “standing” are never-together" in probs
    assert "“talking” should always come with “lips”" in probs
    assert TL.rule_problems(["kissing", "close-up"], rows) == []
    assert TL.fix_tags(["talking"], rows) == (["talking", "lips", "face"], ["lips", "face"])
    assert TL.fix_tags(["wide", "kissing"], rows) == (["wide", "kissing"], [])    # won't break shot size
    # bulk fix of saved labels: only the ids asked for, one write
    for i, (tags, t) in enumerate([(["talking"], 1.0), (["kissing"], 2.0), (["talking"], 3.0)]):
        TL.save_label({"id": f"tags:v:{t:.2f}", "type": "tags", "video_id": "v", "t": t}, tags=tags)
    changed = TL.apply_rules_to_labels(["tags:v:1.00", "tags:v:2.00"], rows)
    assert changed == [("tags:v:1.00", ["lips", "face"]), ("tags:v:2.00", ["close-up"])]
    got = {i["id"]: i for i in TL.load_labels()}
    assert got["tags:v:1.00"]["tags"] == ["talking", "lips", "face"] and got["tags:v:1.00"]["rules_added"] == ["face", "lips"]
    assert got["tags:v:3.00"]["tags"] == ["talking"]                        # not asked for → untouched
    assert TL.apply_rules_to_labels(["tags:v:1.00"], rows) == []            # nothing more to add
    rows = TL.upsert_tag(rows, {**rows[1], "tag": "long"}, "wide")
    rows = TL.upsert_tag(rows, {**rows[0], "tag": "close"}, "close-up")     # rename follows into always_with
    assert rows[2]["always_with"] == "close"


def test_only_with_problems_and_renames(clean_files):
    rows = TL.parse_vocabulary("tag,group,always_with,only_with\n"
                               "kneeling,position,,praying; proposing\npraying,action,,\nproposing,action,,\n"
                               "ring,object,,proposing\nhug,action,ring,\nbow,action,praying,praying\n")
    assert rows[0]["only_with"] == "praying; proposing"
    assert TL.only_map(rows)["kneeling"] == ["praying", "proposing"]
    assert TL.rule_problems(["kneeling", "praying"], rows) == []
    assert TL.rule_problems(["kneeling"], rows) == ["“kneeling” is only allowed with “praying” or “proposing”"]
    # never fixed automatically (is kneeling wrong, or was the action forgotten?)
    assert TL.fix_tags(["kneeling"], rows) == (["kneeling"], [])
    # an always-with partner isn't added if it would break its own only-with rule
    assert TL.fix_tags(["hug"], rows) == (["hug"], [])
    assert TL.fix_tags(["hug", "proposing"], rows) == (["hug", "proposing", "ring"], ["ring"])
    assert TL.rule_clashes(rows) == [("bow", "praying")]
    rows = TL.upsert_tag(rows, {**rows[2], "tag": "proposal"}, "proposing")   # rename follows into only_with
    assert rows[0]["only_with"] == "praying; proposal"
    assert "only_with" in TL.vocabulary_csv(rows).splitlines()[0]
    assert TL.parse_vocabulary(TL.vocabulary_csv(rows))[0]["only_with"] == "praying; proposal"


def test_usually_with_column(clean_files):
    rows = TL.parse_vocabulary("tag,group,usually_with\nkissing,action,close-up\nclose-up,shot,\n")
    assert rows[0]["usually_with"] == "close-up"
    assert TL.rule_problems(["kissing"], rows) == []          # soft — never a rule problem
    assert TL.fix_tags(["kissing"], rows) == (["kissing"], [])
    rows = TL.upsert_tag(rows, {**rows[1], "tag": "close"}, "close-up")
    assert rows[0]["usually_with"] == "close"
    assert TL.parse_vocabulary(TL.vocabulary_csv(rows))[0]["usually_with"] == "close"


def test_new_teacher_columns_and_category_order(clean_files):
    rows = TL.parse_vocabulary("tag,group,one_of,threshold,double_check,tell_apart,group_order,group_required\n"
                               "dance,action,,70,yes,,2,\nclose-up shot,shot,shot size,,,shoulders hidden,1,yes\n"
                               "wide shot,shot,shot size,abc,,,,\nlogo,overlay,,150,,,,\n")
    assert rows[0]["threshold"] == 70 and rows[0]["double_check"] and rows[2]["threshold"] is None
    assert rows[3]["threshold"] == 100 and rows[1]["tell_apart"] == "shoulders hidden"
    gs = TL.group_settings(rows)
    assert [(g["group"], g["order"], g["required"], g["tags"]) for g in gs] == [
        ("shot", 1, True, 2), ("action", 2, False, 1), ("overlay", None, False, 1)]
    # cards follow the category order
    assert [t for t, _, _ in TL.chip_groups(rows)] == ["shot size", "action", "overlay"]
    rows = TL.set_group_settings(rows, [{"group": "overlay", "order": 0, "required": False},
                                        {"group": "shot", "order": 5, "required": True}])
    assert [g["group"] for g in TL.group_settings(rows)] == ["overlay", "action", "shot"]
    assert all(r["group_order"] == 5 and r["group_required"] for r in rows if r["group"] == "shot")
    rows = TL.set_thresholds(rows, {"Dance": 85, "logo": None})
    assert rows[0]["threshold"] == 85 and rows[3]["threshold"] is None
    again = TL.parse_vocabulary(TL.vocabulary_csv(rows))
    assert again == rows
    TL.save_vocabulary(rows)
    assert TL.load_vocabulary()[0] == rows
    # the teacher's parser reads the same file the same way
    import trial_logic as TLG
    tv = TLG.parse_vocabulary(TL.vocabulary_csv(rows))
    assert [(n, req) for n, _, req in TLG.categories(tv)] == [("overlay", False), ("action", False), ("shot", True)]
    assert tv[0]["threshold"] == 85 and tv[0]["double_check"] and tv[1]["tell_apart"] == "shoulders hidden"


def test_page_category_order_threshold_and_double_check(clean_files):
    from tests.apptest_util import errors, page_app
    at = page_app("pages/6_Training_Labels.py")
    at.session_state["tl_view"] = "📒 Vocabulary"
    at.run()
    assert not errors(at), errors(at)
    TL.save_vocabulary(TL.load_vocabulary()[0])           # start from a real file
    at.run()
    # 🗂️ category order: save it (the editor's current rows), then it survives a 💾 Save table
    rows = TL.load_vocabulary()[0]
    groups = [g["group"] for g in TL.group_settings(rows)]
    TL.save_vocabulary(TL.set_group_settings(rows, [{"group": groups[-1], "order": 1, "required": True}]))
    at.run()
    next(b for b in at.button if b.key == "tl_cat_save").click().run()
    assert not errors(at), errors(at)
    g0 = TL.group_settings(TL.load_vocabulary()[0])[0]
    assert (g0["group"], g0["order"], g0["required"]) == (groups[-1], 1, True)
    next(b for b in at.button if b.label == "💾 Save table").click().run()
    assert not errors(at), errors(at)
    assert TL.group_settings(TL.load_vocabulary()[0])[0]["group"] == groups[-1]
    # tag form: threshold + double-check + tell-apart
    at.session_state["tl_ed_pick"] = "dance"
    at.run()
    at.number_input(key="tl_ed_dance_thr").set_value(75)
    at.toggle(key="tl_ed_dance_dcheck").set_value(True)
    at.text_input(key="tl_ed_dance_tell").set_value("feet moving to a beat")
    next(b for b in at.button if b.label == "💾 Save tag").click().run()
    assert not errors(at), errors(at)
    r = next(r for r in TL.load_vocabulary()[0] if r["tag"] == "dance")
    assert r["threshold"] == 75 and r["double_check"] and r["tell_apart"] == "feet moving to a beat"
    assert TL.group_settings(TL.load_vocabulary()[0])[0]["group"] == groups[-1]   # order kept
