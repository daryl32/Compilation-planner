"""⏱ Sign out after an hour without use: the browser timer is on every page, and the page
signs out when the timer sends it back with ?signed_out=idle."""

import streamlit as st

from tests.apptest_util import errors, page_app


def test_timer_installed_on_every_page():
    import page_setup as PS
    assert PS.IDLE_SIGNOUT_MIN == 60
    at = page_app("Home.py").run()
    assert not errors(at), errors(at)
    html = " ".join(str(getattr(e, "proto", "")) for e in at.get("iframe"))
    assert "mp-idle-signout" in html and "60 * 60000" in html


def test_signed_out_idle_logs_out(monkeypatch):
    calls = []
    monkeypatch.setattr(st, "logout", lambda: calls.append(1))
    at = page_app("pages/3_Media_Library.py")
    at.query_params["signed_out"] = "idle"
    at.run()
    assert calls == [1]
    assert "signed_out" not in at.query_params


# ---------------------------------------------------------------------------
# 🔐 The server's own record of each sign-in (no flicker on opening the app; Sign out is real)
# ---------------------------------------------------------------------------

def _fresh(monkeypatch, iat):
    import time
    import page_setup as PS
    PS._SIGNINS_FILE.unlink(missing_ok=True)
    monkeypatch.setattr(st.user, "iat", iat, raising=False)
    calls = []
    monkeypatch.setattr(st, "logout", lambda: calls.append(time.time()))
    return PS, calls


def test_old_sign_in_is_signed_out_before_the_page_is_drawn(monkeypatch):
    import time
    PS, calls = _fresh(monkeypatch, time.time() - 2 * 3600)          # signed in two hours ago, never seen since
    at = page_app("Home.py").run()
    assert len(calls) == 1
    assert not at.title and not at.sidebar.button                     # nothing of the app was drawn
    assert not at.get("iframe")                                       # … not even the browser timer
    # and that sign-in stays ended: a kept copy of its cookie is refused in a new browser session too
    at2 = page_app("Home.py").run()
    assert len(calls) == 2 and not at2.title


def test_sign_in_in_use_stays_signed_in(monkeypatch):
    import json
    import time
    PS, calls = _fresh(monkeypatch, time.time() - 2 * 3600)
    PS._signins_write({PS._signin_key(): {"seen": time.time() - 20 * 60}})   # used 20 minutes ago
    at = page_app("Home.py").run()
    assert calls == [] and at.title[0].value == "Media Planner"
    assert not errors(at), errors(at)
    seen = json.loads(PS._SIGNINS_FILE.read_text())[PS._signin_key()]["seen"]
    assert time.time() - seen < 30                                    # … and this use is recorded
    # an open page isn't timed by the server (the browser's timer does that): only opening one is
    PS._signins_write({PS._signin_key(): {"seen": time.time() - 3 * 3600}})
    at.run()
    assert calls == []
    PS._signins_write({PS._signin_key(): {"seen": time.time() - 3 * 3600}})     # (that run counted as a use)
    assert len(page_app("Home.py").run().title) == 0 and len(calls) == 1


def test_sign_out_ends_the_sign_in_on_the_server(monkeypatch):
    import time
    PS, calls = _fresh(monkeypatch, time.time() - 60)
    at = page_app("Home.py").run()
    assert calls == [] and at.title
    at.sidebar.button(key="signout_btn").click().run()
    assert len(calls) >= 1
    n = len(calls)
    at2 = page_app("Home.py").run()                                   # the same sign-in, from a kept cookie
    assert len(calls) == n + 1 and not at2.title


def test_whitelist_wants_a_verified_address(monkeypatch):
    import page_setup as PS
    assert PS._allowed()
    monkeypatch.setattr(st.user, "email", "Tester@Example.com", raising=False)
    assert PS._allowed()
    monkeypatch.setattr(st.user, "email_verified", False, raising=False)
    assert not PS._allowed()
    monkeypatch.setattr(st.user, "email_verified", True, raising=False)
    monkeypatch.setattr(st.user, "email", "someone@else.com", raising=False)
    assert not PS._allowed()


def test_drive_connect_answer_must_be_our_own(monkeypatch):
    import time
    import page_setup as PS
    monkeypatch.setattr(st.user, "iat", 1_800_000_000, raising=False)
    state = PS._drive_state("pages/2_Reviewer.py")
    assert PS._drive_state_page(state) == "pages/2_Reviewer.py"
    assert PS._drive_state_page("pages/2_Reviewer.py") is None                    # the old, unsigned form
    assert PS._drive_state_page(state.replace("2_Reviewer", "4_Server")) is None  # changed on the way
    assert PS._drive_state_page(PS._drive_state("Home.py", now=time.time() - 3600)) is None   # too old
    monkeypatch.setattr(st.user, "iat", 1_800_000_999, raising=False)             # another sign-in
    assert PS._drive_state_page(state) is None
