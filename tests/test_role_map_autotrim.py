"""🗺️ Reviewer role mapping (motion → roles for the 💡 hints) and the Media Library's
✂️ Auto trim of every shown video, plus 👤 Actors in the finder's top row."""

from __future__ import annotations

import shutil

from tests.support import DATA  # noqa: F401  (installs the test config first)


# ---------------------------------------------------------------------------
# scene_roles: the mapping itself
# ---------------------------------------------------------------------------

def _roles_file_backup(tmp_path):
    import scene_roles as SR
    b = tmp_path / "scene_roles.json.bak"
    had = SR.ROLES_FILE.exists()
    if had:
        shutil.copy2(SR.ROLES_FILE, b)
    return had, b


def _roles_file_restore(had, b):
    import scene_roles as SR
    if had:
        shutil.copy2(b, SR.ROLES_FILE)
    else:
        SR.ROLES_FILE.unlink(missing_ok=True)
    SR._MAP_CACHE.clear()


def test_motion_hints_builtin_tags_and_thresholds():
    import scene_roles as SR
    off = SR.normalise_motion_map({})
    assert off["enabled"] is False and off["mode"] == "tags"
    assert SR.motion_hints(3.0, 0.1, off) == ["slow"]          # built-in: calmest fifth
    assert SR.motion_hints(30.0, 0.9, off) == ["action"]       # busiest fifth
    assert SR.motion_hints(10.0, 0.5, off) == []

    tags = SR.normalise_motion_map({"enabled": True, "mode": "tags",
                                    "tags": {"calm": ["slow", "intro"], "busy": ["action"]}})
    assert SR.motion_hints(1.0, 0.2, tags) == ["slow", "intro"]   # several roles for one tag
    assert SR.motion_hints(9.0, 0.5, tags) == []                  # medium: nothing mapped
    assert SR.motion_hints(40.0, 0.95, tags) == ["action"]        # replaces the built-in
    assert SR.motion_hints(None, None, tags) == []

    th = SR.normalise_motion_map({"enabled": True, "mode": "thresholds", "thresholds": [
        {"min": None, "max": 5, "roles": ["slow"]},
        {"min": 5, "max": 20, "roles": ["intro", "outro"]},
        {"min": 15, "max": None, "roles": ["action", "outro"]},     # overlaps the one before
    ]})
    assert SR.motion_hints(2.6, 0.0, th) == ["slow"]
    assert SR.motion_hints(5.0, 0.0, th) == ["intro", "outro"]    # from is included, up to isn't
    assert SR.motion_hints(16.0, 0.0, th) == ["intro", "outro", "action"]
    assert SR.motion_hints(29.9, 0.0, th) == ["action", "outro"]
    assert SR.motion_hints(None, 0.5, th) == []
    assert SR.describe_rule(th["thresholds"][0]) == "under 5"
    assert SR.describe_rule(th["thresholds"][1]) == "5 to under 20"
    assert SR.describe_rule(th["thresholds"][2]) == "15 and over"


def test_motion_map_saved_beside_custom_roles(tmp_path):
    import scene_roles as SR
    had, b = _roles_file_backup(tmp_path)
    try:
        SR.save_motion_map({"enabled": True, "mode": "tags", "tags": {"busy": ["action"]}})
        rid = SR.add_role("Hero shot", "🦸", "#123456", {"energy": "high"})   # rewrites the roles…
        m = SR.load_motion_map()                                              # …but keeps the map
        assert m["enabled"] and m["tags"]["busy"] == ["action"]
        SR.save_motion_map({**m, "tags": {**m["tags"], "busy": ["action", rid]}})
        assert rid in SR.roles_by_id()                                        # and the map keeps the roles
        assert SR.load_motion_map()["tags"]["busy"] == ["action", rid]
    finally:
        _roles_file_restore(had, b)


# ---------------------------------------------------------------------------
# Reviewer page
# ---------------------------------------------------------------------------

def test_reviewer_accepts_mapped_hints(tmp_path):
    import catalogue_store as CStore
    import scene_roles as SR
    from tests.apptest_util import errors, page_app
    had, b = _roles_file_backup(tmp_path)
    backups = {p: tmp_path / p.name for p in CStore.catalogue_paths()}
    for p, bk in backups.items():
        shutil.copy2(p, bk)
    try:
        # Every scene gets two roles from one "any motion" range.
        SR.save_motion_map({"enabled": True, "mode": "thresholds",
                            "thresholds": [{"min": None, "max": None, "roles": ["intro", "outro"]}]})
        at = page_app("pages/2_Reviewer.py", timeout=120)
        at.run()
        assert not errors(at), errors(at)
        assert at.toggle(key="rev_map_on_0").value is True
        assert any("role mapping suggests" in c.value for c in at.caption)
        hint_btn = next(bt for bt in at.button if bt.label.startswith("💡 Accept the"))
        assert not hint_btn.disabled
        hint_btn.click().run()
        assert not errors(at), errors(at)
        # Both mapped roles now sit on the cards (unsaved), so no hints are left to show.
        picked = [w.value for w in at.button_group if (w.key or "").startswith("rev_roles") or "roles" in (w.key or "")]
        assert picked and all({"intro", "outro"} <= set(v or []) for v in picked)

        # Quick role marking: the same mapping, added to each card on the page.
        at.session_state["rev_quick_roles"] = True
        at.run()
        assert not errors(at), errors(at)
        at.button(key="rev_qr_hints").click().run()
        assert not errors(at), errors(at)
        qr = [g.value for g in at.button_group if (g.key or "").startswith("rev_qr_p_")]
        assert qr and all({"intro", "outro"} <= set(v or []) for v in qr)

        # Switching mapping off from the sidebar saves it and brings the built-in hints back.
        at.session_state["rev_quick_roles"] = False
        at.toggle(key="rev_map_on_0").set_value(False).run()
        assert not errors(at), errors(at)
        assert SR.load_motion_map()["enabled"] is False
    finally:
        _roles_file_restore(had, b)
        for p, bk in backups.items():
            shutil.copy2(bk, p)


def test_reviewer_threshold_ranges_add_and_remove(tmp_path):
    import scene_roles as SR
    from tests.apptest_util import errors, page_app
    had, b = _roles_file_backup(tmp_path)
    try:
        SR.save_motion_map({"enabled": True, "mode": "thresholds", "thresholds": []})
        at = page_app("pages/2_Reviewer.py", timeout=120)
        at.run()
        at.button(key="rev_map_add").click().run()
        at.button(key="rev_map_add").click().run()
        assert not errors(at), errors(at)
        assert len(SR.load_motion_map()["thresholds"]) == 2
        v = at.session_state["rev_map_ver"]
        at.number_input(key=f"rev_map_hi_0_{v}").set_value(10.0).run()
        at.multiselect(key=f"rev_map_roles_0_{v}").set_value(["slow", "intro"]).run()
        rules = SR.load_motion_map()["thresholds"]
        assert rules[0]["max"] == 10.0 and rules[0]["roles"] == ["slow", "intro"]
        at.button(key=f"rev_map_del_1_{v}").click().run()
        assert not errors(at), errors(at)
        assert len(SR.load_motion_map()["thresholds"]) == 1
    finally:
        _roles_file_restore(had, b)


# ---------------------------------------------------------------------------
# Media Library
# ---------------------------------------------------------------------------

def test_media_library_auto_trim_and_actor_filter():
    import library_common as LC
    from tests.apptest_util import errors, page_app
    meta_before = LC.load_library_meta()
    try:
        at = page_app("pages/3_Media_Library.py", timeout=300)
        at.session_state["lib_kind"] = "🎬 Videos"
        at.run()
        assert not errors(at), errors(at)
        # 👤 Actors is in the top row now, not in More filters.
        assert at.multiselect(key="lib_actors").label == "👤 Actors"
        # Hand-set range on one video: auto trim replaces it.
        LC.set_library_range("vid_a", (0.5, 1.0))
        at.run()
        at.button(key="lib_autotrim").click().run()
        assert not errors(at), errors(at)
        assert any("Auto trimmed" in s.value for s in at.success)
        after = LC.library_ranges()
        assert after.get("vid_a") != (0.5, 1.0)
        # ↩️ Undo puts the old ranges back.
        at.button(key="lib_autotrim_undo_btn").click().run()
        assert not errors(at), errors(at)
        assert LC.library_ranges().get("vid_a") == (0.5, 1.0)
        # Only the shown videos are trimmed.
        LC.set_library_range("vid_a", (0.5, 1.0))
        at.text_input(key="lib_name_query").set_value("vid_b").run()
        at.button(key="lib_autotrim").click().run()
        assert LC.library_ranges().get("vid_a") == (0.5, 1.0)
        assert set(at.session_state["lib_autotrim_undo"]) == {"vid_b"}
    finally:
        LC.save_library_meta(meta_before)


def test_role_editor_emoji_picker():
    from tests.apptest_util import errors, page_app
    at = page_app("pages/2_Reviewer.py", timeout=120)
    at.run()
    p = "rev_role_➕ New role"
    assert at.text_input(key=f"{p}_emoji").value == "🏷️"
    at.button_group(key=f"{p}_emoji_pick").set_value("🔥").run()
    assert not errors(at), errors(at)
    assert at.text_input(key=f"{p}_emoji").value == "🔥"
    at.text_input(key=f"{p}_emoji").set_value("🦸‍♀️").run()          # several-part emoji fits too
    assert at.text_input(key=f"{p}_emoji").value == "🦸‍♀️"
