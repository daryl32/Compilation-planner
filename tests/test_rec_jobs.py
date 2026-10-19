"""⏳ Background recommendation searches (jobs.py, rec_library.py, heavy_slot.py) and
the step 2 feeds (core/feeds.py, the Recommendations page's tabs)."""

import datetime
import json
import shutil
import threading
import time

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.test_recommend import ACTORS, _open_and_export

PAGE = "pages/5_Recommendations.py"


def _clean():
    import jobs as J
    import rec_library as RL
    for d in (RL.RESULTS_DIR, J.JOBS_DIR):
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def nightly():
    """A clean slate, actors tagged, vid_f 'added' yesterday (everything else old), then the nightly run."""
    import jobs as J
    import rec_library as RL
    from library_common import LIBRARY_META_FILE, load_library_meta, save_library_meta
    _clean()
    before = LIBRARY_META_FILE.read_text() if LIBRARY_META_FILE.exists() else None
    meta = load_library_meta()
    for vid, names in ACTORS.items():
        meta.setdefault(vid, {})["actors"] = names
    save_library_meta(meta)
    # Drive added dates (catalogue_store.ADDED_DATES_FILE): vid_f arrived yesterday, the rest long ago
    import catalogue_store as CStore
    added_before = CStore.ADDED_DATES_FILE.read_text() if CStore.ADDED_DATES_FILE.exists() else None
    now = datetime.datetime.now(datetime.timezone.utc)
    CStore.ADDED_DATES_FILE.write_text(json.dumps(
        {stem: (now - datetime.timedelta(days=1 if slim["video_id"] == "vid_f" else 400)).isoformat()
         for stem, slim in CStore.video_index().items()}))
    first = J.changed_jobs()
    info = J.run_nightly()
    yield {"first": first, "info": info}
    if added_before is None:
        CStore.ADDED_DATES_FILE.unlink(missing_ok=True)
    else:
        CStore.ADDED_DATES_FILE.write_text(added_before)
    if before is None:
        LIBRARY_META_FILE.unlink(missing_ok=True)
    else:
        LIBRARY_META_FILE.write_text(before)


def test_nightly_searches_only_what_changed(nightly):
    import jobs as J
    import rec_library as RL
    tracks = RL.list_tracks()
    kinds = [j["kind"] for j in nightly["first"]]
    assert kinds.count("track") == len(tracks)
    assert [j["video_id"] for j in nightly["first"] if j["kind"] == "spotlight"] == ["vid_f"]
    assert kinds.index("spotlight") > max(i for i, k in enumerate(kinds) if k == "track")   # tracks first
    status = J.read_status()
    assert status and all(v["state"] == "done" for v in status.values()), status
    assert not J.read_queue() and not J.worker_running()
    assert set(RL.all_results()) == set(tracks)
    # nothing changed → nothing to search (the spotlight is re-checked but already up to date)
    assert [j["kind"] for j in J.changed_jobs()] == ["spotlight"]
    # a changed video → every track is out of date again
    from library_common import set_video_details
    set_video_details("vid_b", actors=["Ben", "Dee"])
    try:
        assert sum(j["kind"] == "track" for j in J.changed_jobs()) == len(tracks)
    finally:
        set_video_details("vid_b", actors=ACTORS["vid_b"])
    assert J.nightly_status()["queued_tracks"] == len(tracks)


def test_fingerprints_cached_on_disk(nightly):
    import rec_library as RL
    assert len(list(RL.PRINTS_DIR.glob("*.npz"))) >= 5
    RL._PRINTS_MEMO.clear()
    t = time.time()
    prints = RL.library_prints()
    assert len(prints) >= 5 and time.time() - t < 5


def test_queue_dedupes_and_cancels():
    import jobs as J
    J.cancel_queued()
    job = {"kind": "track", "track_id": "song_short", "n_videos": 3, "sequential": True}
    assert J.enqueue([job, job]) == 1
    assert J.enqueue([job]) == 0
    assert J.read_status()[J.job_id(job)]["state"] == "queued"
    assert J.cancel_queued() == 1 and not J.read_queue()


def test_audio_goes_first_and_one_heavy_job_at_a_time(tmp_path):
    """A queued audio job holds the recommendation worker back; so does the heavy slot."""
    import jobs as J
    from config import AUDIO_DIR
    from heavy_slot import heavy_slot, holder
    J.cancel_queued()
    J.clear_finished()
    aq = AUDIO_DIR / ".analysis"
    aq.mkdir(parents=True, exist_ok=True)
    (aq / "queue.json").write_text(json.dumps([{"track_id": "x", "kind": "extra"}]))
    J.enqueue([{"kind": "track", "track_id": "song_short", "n_videos": 2, "sequential": False}])
    jid = J.job_id({"kind": "track", "track_id": "song_short", "n_videos": 2, "sequential": False})
    seen_wait, slot_free_at = [], []

    def audio_then_slot():
        time.sleep(1.0)
        seen_wait.append(J.read_status()[jid]["msg"])
        with heavy_slot("audio analysis"):                 # audio takes the slot as its queue empties
            (aq / "queue.json").write_text("[]")
            time.sleep(1.0)
            assert holder()["who"] == "audio analysis"
            slot_free_at.append(time.time())

    th = threading.Thread(target=audio_then_slot)
    th.start()
    J.run_worker(poll_sec=0.2)
    th.join()
    st = J.read_status()[jid]
    assert "audio analysis" in seen_wait[0]
    assert st["state"] == "done", st
    assert st["updated"] >= slot_free_at[0]                  # it only ran once audio let go of the slot
    (aq / "queue.json").unlink(missing_ok=True)


def test_spotlight_plans_feature_the_new_video(nightly):
    import rec_library as RL
    spot = RL.load_spotlight("vid_f")
    assert spot and spot["similar"] and spot["fits"]
    assert 1 <= len(spot["plans"]) <= RL.SPOTLIGHT_TRACKS
    for p in spot["plans"]:
        assert "vid_f" in p["videos"] and p["mode"] == "spotlight" and p["featured"] == "vid_f"
    assert [v for v, _ in RL.new_videos()] == ["vid_f"]


# ---------------------------------------------------------------------------
# core/feeds.py
# ---------------------------------------------------------------------------

def _plan(mode, overall, action=None, action_story=None, videos=("a", "b")):
    return {"mode": mode, "style": "smart", "pace": 0, "videos": list(videos), "variant": "🧠 Smart · normal",
            "scores": {"overall": overall, "action": action, "action_story": action_story}}


def test_feeds_top_plans_and_best_action():
    import core.feeds as F
    res = {"t1": {"results": [{"plan": _plan("overall", 80, 50, 100, ("a", "b"))},
                              {"plan": _plan("shape", 70, 90, None, ("a", "c"))},
                              {"plan": _plan("story", 60, 10, 0, ("a", "d"))}]},
           "t2": {"results": [{"plan": _plan("overall", 75, None, None)}, {"plan": None}]}}
    top = F.top_plans(res, 10)
    assert [(i["track_id"], i["score"]) for i in top] == [("t1", 80), ("t2", 75), ("t1", 70)]   # 2 per track
    act = F.best_action(res, 10)
    assert [(i["track_id"], i["plan"]["mode"], i["score"]) for i in act] == [("t1", "shape", 90), ("t1", "overall", 75)]
    assert all(i["plan"]["scores"]["action"] is not None for i in act)


def test_feeds_hidden_gems_and_gaps():
    import core.feeds as F
    res = {f"t{i}": {"insights": {"videos": {"good_unused": 0.8, "good_used": 0.85, "weak": 0.3, "mid": 0.5},
                                  "need": {"action": 30.0, "intro": 10.0},
                                  "blocks": [{"len": 0.8, "intensity": 0.9, "role": "action", "best": 0.4},
                                             {"len": 6.0, "intensity": 0.2, "role": None, "best": 0.9}] * 3}}
           for i in range(4)}
    gems = F.hidden_gems(res, {"good_used": 9, "weak": 0, "mid": 0})
    assert [g["video_id"] for g in gems] == ["good_unused"]
    assert gems[0]["uses"] == 0 and gems[0]["tracks"]
    g = F.gaps(res, {"action": 20.0}, {"action": "💥 Action", "intro": "🎬 Intro"})
    by = {r["role"]: r for r in g["roles"]}
    assert by["intro"]["status"] == "none" and by["action"]["status"] == "short"
    assert g["roles"][0]["role"] == "intro"                                   # the worst first
    assert g["blocks"][0]["label"] == "Intense, short shots (under 1.5 s)" and g["blocks"][0]["gap"]
    assert not g["blocks"][-1]["gap"]
    # with depth (the 5th-best fit), that's what's measured
    for t in res.values():
        for b in t["insights"]["blocks"]:
            b["depth"] = b["best"] - 0.1
    g2 = F.gaps(res, {"action": 20.0})
    assert g2["measure"] == "depth" and g2["blocks"][0]["label"] == "Intense, short shots (under 1.5 s)"


def test_feeds_vary_the_videos():
    """A feed doesn't show the same few videos in every plan."""
    import core.feeds as F
    res = {f"t{i}": {"results": [{"plan": _plan("overall", 90 - i, videos=("a", "b", "c", "d"))},
                                 {"plan": _plan("shape", 80 - i, videos=(f"x{i}", f"y{i}", "a", f"z{i}"))}]}
           for i in range(8)}
    top = F.top_plans(res, 10)
    from collections import Counter
    counts = Counter(v for item in top for v in item["plan"]["videos"])
    assert max(counts.values()) <= F.PER_VIDEO
    assert len(counts) >= 10


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def test_page_feeds_and_open_from_top_plans(nightly):
    """The feed tabs show the nightly results, and a plan opened from 🏆 Top plans exports
    exactly the clips it was scored with."""
    import core.feeds as F
    import rec_library as RL
    from tests.apptest_util import errors, page_app
    at = page_app(PAGE, timeout=600)
    at.session_state["rec_track"] = "song_full"
    at.run()
    assert not errors(at), errors(at)
    assert [t.label for t in at.tabs] == ["🎵 For this track", "🏆 Top plans", "💥 Best action", "🆕 New videos",
                                         "🧩 Hidden gems", "🕳️ Gaps"]
    # the nightly results show on 🎵 For this track straight away (default settings)
    assert sum(1 for b in at.button if (b.key or "").startswith("open_song_full_")) >= 4
    assert any("🆕 vid_f" in m.value for m in at.markdown)
    top = F.top_plans(RL.all_results(), 10)
    assert sum(1 for b in at.button if (b.key or "").startswith("open_top_")) == len(top)
    [b for b in at.button if b.key == "open_top_1"][0].click().run()
    opened = at.session_state["_open_recommended_plan"]
    assert opened["track_id"] == top[0]["track_id"] and opened["videos"] == top[0]["plan"]["videos"]
    assert _open_and_export(opened) == top[0]["plan"]["picks"]


def test_open_spotlight_plan(nightly):
    import rec_library as RL
    from tests.apptest_util import page_app
    plan = RL.load_spotlight("vid_f")["plans"][0]
    at = page_app(PAGE, timeout=600)
    at.run()
    [b for b in at.button if b.key == "open_spot_vid_f_1"][0].click().run()
    opened = at.session_state["_open_recommended_plan"]
    assert opened["track_id"] == plan["track_id"] and "vid_f" in opened["videos"]
    assert _open_and_export(opened) == plan["picks"]


def test_background_panel_queues_changed_tracks(nightly):
    import jobs as J
    from tests.apptest_util import errors, page_app
    J.pause()                                    # queue but don't start a worker from the test
    try:
        at = page_app(PAGE, timeout=600)
        at.run()
        assert any(e.label.startswith("⏳ Background searches") for e in at.expander)
        [b for b in at.button if b.key == "rec_bg_changed"][0].click().run()
        assert not errors(at), errors(at)
        # only the (up-to-date) spotlight re-check is queued — every track is current
        assert [j["kind"] for j in J.read_queue()] == ["spotlight"]
    finally:
        J.cancel_queued()
        J._paths()["pause"].unlink(missing_ok=True)


def test_thumbnails_skip_excluded_and_out_of_range_scenes():
    """The page's video pictures: the ⭐ master if chosen, else a usable scene (never an
    excluded one, never outside the library range) from the middle of the range."""
    import catalogue_store as CStore
    from config import CATALOGUE_DIR
    from tests.support import load_page
    from library_common import load_library_meta, save_library_meta
    ns = load_page(PAGE)
    vid = "vid_b"                                        # fixture: scene 2 is excluded
    scenes = CStore.all_catalogues()[vid]["scenes"]
    made = []
    for s in scenes:
        s.setdefault("thumbnail_paths", [f"/content/x/{vid}_scene{s['scene_id']}.jpg"])
        p = CATALOGUE_DIR / "thumbnails" / vid / s["thumbnail_paths"][0].split("/")[-1]
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"jpg")
            made.append(p)
    meta_before = load_library_meta()
    try:
        name_of = {s["thumbnail_paths"][0].split("/")[-1]: s for s in scenes}
        excluded = scenes[2]
        assert excluded.get("excluded")
        # a range around the excluded scene: the pick must still be a usable scene inside it
        from core.footage import _tc_to_seconds
        a, b = _tc_to_seconds(excluded["start_tc"]) - 0.5, _tc_to_seconds(excluded["end_tc"]) + 0.5
        import library_common as LC
        LC.set_library_range(vid, (a, b))
        t = ns["_thumb"](vid)
        assert t is not None
        pick = name_of[t.name]
        assert not pick.get("excluded")
        pa, pb = _tc_to_seconds(pick["start_tc"]), _tc_to_seconds(pick["end_tc"])
        assert pb > a and pa < b                         # overlaps the range
        # ⭐ the master thumbnail wins
        LC.set_master_thumbnail(vid, scenes[5]["scene_id"])
        assert ns["_thumb"](vid).name == scenes[5]["thumbnail_paths"][0].split("/")[-1]
    finally:
        save_library_meta(meta_before)
        for p in made:
            p.unlink(missing_ok=True)
