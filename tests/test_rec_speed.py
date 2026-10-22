"""⚡ Faster recommendation searches — and the promise that comes with it: the plans are
exactly what a full search would find.

  • the shape-window memo inside one search (PlanContext.shape_memo);
  • matcher runs kept between searches (rec_library CHECKS_DIR), reused only when the cut,
    the story roles and every video involved are unchanged;
  • "has anything changed?" judged by content, not file times;
  • the nightly run's worker count."""

import json
import os
import shutil

import pytest

from tests.support import DATA  # noqa: F401  (installs the test config first)
from tests.fixtures import load_track


def _clean():
    import jobs as J
    import rec_library as RL
    for d in (RL.RESULTS_DIR, RL.CHECKS_DIR, J.JOBS_DIR):
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def clean():
    _clean()
    yield
    _clean()


def _search(track_id="song_full", n=3, seq=False):
    import rec_library as RL
    d = RL.run_track(track_id, n, seq, seed=7)
    return {k: v for k, v in d.items() if k not in ("finished_at",)}


def _counting_matcher(monkeypatch):
    """Counts the real matcher's runs (not the kept ones)."""
    import core.matching as CMATCH
    calls = [0]
    real = CMATCH.match_scenes_to_track

    def counted(*a, **k):
        calls[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(CMATCH, "match_scenes_to_track", counted)
    return calls


def test_shape_memo_gives_the_same_candidates():
    import catalogue_store as CStore
    import core.footage as CFOOT
    import core.matching as CMATCH
    import core.recommend as REC
    import rec_library as RL
    import scene_roles as SR
    from core.shape import get_block_audio_curve
    tr = load_track(DATA, "song_full")
    segs = REC.cut_variant(tr, {"sm_pace": 0})
    q = CFOOT.build_footage_queues(CStore.all_catalogues(), ("vid_a", "vid_b", "vid_c", "vid_f"), (), {},
                                   roles_of=SR.scene_roles)
    plain, memo = RL.plan_context(None), RL.plan_context(None, shape_memo={})
    for seg in segs[:12]:
        audio = get_block_audio_curve(tr, seg["start"], seg["end"])
        for _ in range(2):          # the second time comes from the memo
            a, _m = CMATCH.find_shape_candidates(seg, q, False, set(), set(), audio, 12, ctx=plain)
            b, _m = CMATCH.find_shape_candidates(seg, q, False, set(), set(), audio, 12, ctx=memo)
            assert [(c["video_id"], c["scene_id"], c["window_offset_sec"], c["score"]) for c in a] == \
                   [(c["video_id"], c["scene_id"], c["window_offset_sec"], c["score"]) for c in b]
    assert memo.shape_memo        # it was used


def test_kept_runs_are_reused_and_give_the_same_plans(clean, monkeypatch):
    first = _search()
    calls = _counting_matcher(monkeypatch)
    again = _search()
    assert calls[0] == 0                       # nothing changed: every matcher run was kept
    assert again == first


def test_a_changed_video_reruns_only_what_it_touches(clean, monkeypatch):
    import catalogue_store as CStore
    import rec_library as RL
    _search()
    kept = RL.load_checks("song_full", 3, False)
    assert kept
    p = CStore.catalogue_path("vid_c")
    before = p.read_text()
    try:
        cat = json.loads(before)
        for sc in cat["scenes"][:6]:
            sc["motion_curve"]["values"] = [round(v * 1.8, 4) for v in sc["motion_curve"]["values"]]
        sc = cat["scenes"][3]
        sc["roles"] = ["action"]
        p.write_text(json.dumps(cat))
        calls = _counting_matcher(monkeypatch)
        incremental = _search()
        reran = calls[0]
        shutil.rmtree(RL.CHECKS_DIR)
        calls[0] = 0
        full = _search()
        assert incremental == full                 # exactly what a full search finds
        assert 0 < reran < calls[0]                # …for fewer matcher runs
    finally:
        p.write_text(before)


def test_a_touched_but_unchanged_file_changes_nothing(clean):
    import catalogue_store as CStore
    import jobs as J
    import rec_library as RL
    import story_map as STM
    maps_before = STM._maps_file().read_text() if STM._maps_file().exists() else None
    try:
        for j in J.changed_jobs():               # (also gives tracks their default story maps)
            if j["kind"] == "track":
                RL.run_track(j["track_id"], j["n_videos"], j["sequential"])
        _touch_and_check(CStore, J)
    finally:
        if maps_before is None:
            STM._maps_file().unlink(missing_ok=True)
        else:
            STM._maps_file().write_text(maps_before)


def _touch_and_check(CStore, J):
    import rec_library as RL
    assert not [j for j in J.changed_jobs() if j["kind"] == "track"]
    p = CStore.catalogue_path("vid_b")
    st = p.stat()
    os.utime(p, (st.st_atime + 3600, st.st_mtime + 3600))      # e.g. a re-download of the same file
    t = RL.AUDIO_DIR / "song_full.json"
    st2 = t.stat()
    os.utime(t, (st2.st_atime + 3600, st2.st_mtime + 3600))
    try:
        assert not [j for j in J.changed_jobs() if j["kind"] == "track"]
    finally:
        os.utime(p, (st.st_atime, st.st_mtime))
        os.utime(t, (st2.st_atime, st2.st_mtime))


def test_kept_runs_start_afresh_when_the_matcher_code_changes(clean, monkeypatch):
    import rec_library as RL
    _search()
    assert RL.load_checks("song_full", 3, False)
    monkeypatch.setattr(RL, "_CODE_SIG", ["something else"])
    assert RL.load_checks("song_full", 3, False) == {}


def test_nightly_uses_every_cpu(monkeypatch):
    import config
    import jobs as J
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    if os.name != "nt":
        assert J.nightly_worker_count() == 4
        assert J.worker_count() == 3 or getattr(config, "REC_WORKERS", None)
    monkeypatch.setattr(config, "REC_NIGHT_WORKERS", 2, raising=False)
    assert J.nightly_worker_count() == 2


def _tiny_shortlist(monkeypatch):
    """A shortlist of ~2 videos, so the 6-video test library has videos outside it."""
    import core.recommend as REC
    for name, val in (("SHORT_OVERALL", 1), ("SHORT_STORY", 1), ("SHORT_SHAPE", 1), ("SHORT_ROLE", 1),
                      ("SHORT_BLOCK", 0)):
        monkeypatch.setattr(REC, name, val)
    monkeypatch.setattr(REC, "VERSES_SEEDS_LOOKED_AT", 1)
    monkeypatch.setattr(REC, "VERSES_POOL", 1)


def _edit_video(CStore, vid, fn):
    p = CStore.catalogue_path(vid)
    before = p.read_text()
    cat = json.loads(before)
    fn(cat)
    p.write_text(json.dumps(cat))
    return lambda: p.write_text(before)


def test_only_tracks_a_change_can_affect_are_searched_again(clean, monkeypatch):
    """A new (or changed) video that wouldn't make a track's shortlist is checked quickly (no
    matcher) and leaves its plans alone; a change to a video it depends on searches it again."""
    import catalogue_store as CStore
    import jobs as J
    import rec_library as RL
    import story_map as STM
    from tests.fixtures import make_video
    _tiny_shortlist(monkeypatch)
    maps_before = STM._maps_file().read_text() if STM._maps_file().exists() else None
    J.changed_jobs()                                    # (gives tracks their default story maps first)
    RL.run_track("song_full", RL.DEFAULT_VIDEOS, False)
    data = RL.load_results("song_full", RL.DEFAULT_VIDEOS, False)
    plans_before = data["results"]

    def jobs_for_song():
        return [j for j in J.changed_jobs() if j.get("track_id") == "song_full"]

    assert not jobs_for_song()
    # A new video with flat, featureless motion: it can't make the shortlist.
    new = make_video(77, "vid_flat", 6)
    for sc in new["scenes"]:
        sc["motion_curve"]["values"] = [0.01] * len(sc["motion_curve"]["values"])
    newp = CStore.CATALOGUE_DIR / "vid_flat.json"
    newp.write_text(json.dumps(new))
    try:
        js = jobs_for_song()
        assert len(js) == 1 and js[0].get("check")
        calls = _counting_matcher(monkeypatch)
        assert J.run_job(js[0], lambda *a, **k: None) == "checked — the plans can't change"
        assert calls[0] == 0
        after = RL.load_results("song_full", RL.DEFAULT_VIDEOS, False)
        assert after["results"] == plans_before
        assert "vid_flat" in after["insights"]["videos"]            # the feeds know about it
        assert not jobs_for_song()                                   # and it's up to date again
    finally:
        newp.unlink()
    # It went again → only the feeds' numbers change.
    js = jobs_for_song()
    assert len(js) == 1 and js[0].get("check")
    J.run_job(js[0], lambda *a, **k: None)
    assert "vid_flat" not in RL.load_results("song_full", RL.DEFAULT_VIDEOS, False)["insights"]["videos"]

    # A video in one of its plans changes → searched again.
    vid = plans_before[0]["plan"]["videos"][0]
    undo = _edit_video(CStore, vid, lambda cat: cat["scenes"][0].update(excluded=True))
    try:
        js = jobs_for_song()
        assert len(js) == 1 and not js[0].get("check")
    finally:
        undo()
        if maps_before is None:
            STM._maps_file().unlink(missing_ok=True)
        else:
            STM._maps_file().write_text(maps_before)
