"""🌙 Overnight renders (nightly_modes.py): the random draws, a night's run and record, clearing, and the tab."""
from __future__ import annotations

import datetime
import random
import shutil

from tests.edge_support import _timelines

ACTORS = {"vid_a": ["Ava"], "vid_b": ["Ava"], "vid_c": ["Bea"], "vid_f": ["Bea"]}


def _setup(monkeypatch, tmp_path):
    import library_common as LC
    import nightly_modes as NM
    import planner.edge as PE
    import planner.verses as PV
    monkeypatch.setattr(LC, "video_details", lambda: {v: {"actors": a, "tags": []} for v, a in ACTORS.items()})
    monkeypatch.setattr(NM, "record_path", lambda: tmp_path / "overnight.json")
    monkeypatch.setattr(NM, "_songs", lambda: ["song_full", "song_noanalysis"])
    made = []

    def fake_render(kind):
        def _r(res, settings, token, prog):
            assert token is None and settings["quick"] and settings["label"].startswith("🌙")
            made.append((kind, res))
            return ("info", "Rendered."), f"{kind}_render.mp4"
        return _r
    monkeypatch.setattr(PE, "_render", fake_render("edge"))
    monkeypatch.setattr(PV, "_render", fake_render("verses"))
    monkeypatch.setattr(PE, "fetch_lines", lambda *a, **k: [])
    return NM, made


def test_the_draws_stay_in_their_rules(monkeypatch, tmp_path):
    out = _timelines(monkeypatch)
    try:
        NM, _ = _setup(monkeypatch, tmp_path)
        import edge_mode as E
        import verses_battle as VB
        for seed in range(12):
            e = NM.draw_edge(random.Random(seed))
            assert e["avatar"] and e["voiceover"] and e["intro"] and 3 <= e["rounds"] <= 4
            assert e["difficulty"] in E.DIFFICULTIES and 1 <= len(e["playlist"]) <= 2
            assert not (e["surprises"] and e["difficulty"] == "hard")
            v = NM.draw_verses(random.Random(seed))
            assert {v["a"], v["b"]} == {"Ava", "Bea"} and v["winner"] == "surprise"
            assert v["layout"] in VB.LAYOUTS and v["music"] in VB.MUSIC_MODES and not v["track_a"]
        assert NM.draw_edge(random.Random(1), songs=[])["beat"]          # no songs: the beat
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_a_night_is_made_and_recorded(monkeypatch, tmp_path):
    out = _timelines(monkeypatch)
    try:
        NM, made = _setup(monkeypatch, tmp_path)
        night = NM.run(seed=3, log=lambda s: None)
        assert [k for k, _ in made] == ["edge", "verses"]
        assert night["edge"]["render"] == "edge_render.mp4" and night["verses"]["render"] == "verses_render.mp4"
        assert night["edge"]["setup"]["avatar"] and "⏱️ Compile" in night["edge"]["took"]
        assert "⚔️ Ava vs Bea" in night["verses"]["line"] or "⚔️ Bea vs Ava" in night["verses"]["line"]
        edge_plan = made[0][1]["plan"]
        assert edge_plan["edge"]["avatar"]["on"] and edge_plan["edge"]["voiceover"]["on"]
        assert NM.history()[0]["date"] == datetime.date.today().isoformat()
        # one failing doesn't stop the other; the card shows why
        import planner.verses as PV
        monkeypatch.setattr(PV, "compile_verses", lambda *a, **k: (_ for _ in ()).throw(ValueError("no fighters")))
        night2 = NM.run(seed=4, log=lambda s: None)
        assert night2["edge"]["render"] and "no fighters" in night2["verses"]["error"]
        assert len(NM.history()) == 1                      # the same night: replaced, not added
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_old_unrated_overnight_renders_are_cleared(monkeypatch, tmp_path):
    import nightly_modes as NM
    import renders as R
    from config import PREVIEW_DIR
    from pathlib import Path
    monkeypatch.setattr(NM, "record_path", lambda: tmp_path / "overnight.json")
    old = (datetime.date.today() - datetime.timedelta(days=NM.KEEP_NIGHTS + 1)).isoformat()
    names = {"edge": "zz_old_edge.mp4", "verses": "zz_old_vs.mp4"}
    metas = {}
    for kind, name in names.items():
        mp4 = Path(PREVIEW_DIR) / name
        mp4.parent.mkdir(parents=True, exist_ok=True)
        mp4.write_bytes(b"x")
        metas[name] = {"render_file": name, "overnight": {"kind": kind}, "user": {"rating": 4 if kind == "verses" else 0},
                       "path": mp4, "local": True}
    monkeypatch.setattr(R, "load_render", lambda p: metas.get(Path(p).name))
    gone = []
    monkeypatch.setattr(R, "delete_render", lambda p, token=None: gone.append(Path(p).name))
    NM._save([{"date": old, "edge": {"render": names["edge"]}, "verses": {"render": names["verses"]}}])
    NM.clear_old(log=lambda s: None)
    assert gone == [names["edge"]]                         # the rated one stays
    for name in names.values():
        (Path(PREVIEW_DIR) / name).unlink(missing_ok=True)


def test_the_overnight_tab(monkeypatch, tmp_path):
    import nightly_modes as NM
    from tests.apptest_util import errors, page_app
    at = page_app("pages/5_Recommendations.py", timeout=120)
    at.run()
    assert not errors(at), errors(at)
    NM._save([{"date": datetime.date.today().isoformat(),
               "edge": {"render": "missing_edge.mp4", "line": "⏱️ Edge · Chill", "seconds": 600},
               "verses": {"error": "ValueError: no fighters", "seconds": 3}}])
    try:
        at.run()
        assert not errors(at), errors(at)
        text = " ".join(c.value for c in at.caption)
        assert "no fighters" in text and "⏱️ Edge · Chill" in text
    finally:
        NM.record_path().unlink(missing_ok=True)
