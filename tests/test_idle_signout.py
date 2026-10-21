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
