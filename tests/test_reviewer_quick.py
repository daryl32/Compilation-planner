"""Reviewer → 🎭 Quick role marking: motion score on each card, ✂️ / ⚡ split without
leaving quick marking, saved roles as badges and suggestions as faint text."""

import shutil
import time

from tests.support import DATA  # noqa: F401  (installs the test config first)

PAGE = "pages/2_Reviewer.py"


def test_quick_marking_cards_split_and_show_motion(tmp_path):
    import catalogue_store as CStore
    from tests.apptest_util import errors, page_app
    backups = {p: tmp_path / p.name for p in CStore.catalogue_paths()}
    for p, b in backups.items():
        shutil.copy2(p, b)
    try:
        at = page_app(PAGE, timeout=120)
        at.run()
        at.session_state["rev_quick_roles"] = True
        at.session_state["rev_auto_target"] = 2.0
        at.run()
        assert not errors(at), errors(at)
        cards = lambda: [b for b in at.button if (b.key or "").startswith("rev_qr_t_")]  # noqa: E731
        n = len(cards())
        assert sum(c.value.startswith("🏃 motion") for c in at.caption) == n
        assert len([b for b in at.button if (b.key or "").startswith("rev_qr_split_")]) == n
        autos = [b for b in at.button if (b.key or "").startswith("rev_qr_auto_") and not b.disabled]
        assert autos
        autos[0].click().run()
        assert not errors(at), errors(at)
        assert at.session_state["rev_quick_roles"] is True            # still in quick marking
        assert len(cards()) > n
        assert any("split into" in s.value for s in at.success)
        # ✂️ opens the split dialog
        [b for b in at.button if (b.key or "").startswith("rev_qr_split_")][0].click().run()
        assert not errors(at), errors(at)
        assert any(b.label == "➕ Add cut here" for b in at.button)
    finally:
        time.sleep(0.01)
        for p, b in backups.items():
            shutil.copy2(b, p)
            p.touch()
