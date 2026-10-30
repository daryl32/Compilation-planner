"""
Shared setup for every page of the app — call page_setup() first thing on a page.

Does, in order:
  • page config (wide layout; tab title marked 🧪 TEST in the test copy) + test banner
  • Google sign-in screen if not signed in, then the email whitelist gate
  • Google Drive connect callback (?code=...), returning to the page that asked
  • page title
  • the same sidebar top on every page (page controls go below its divider):
      👤 account box — name, email, version, ☁️ Drive connected / not (+ Connect),
         Drive sync status, "edits not yet saved to Drive" + Save (progress and
         time left), memory-freed note, Sign out
      ☁️ Google Drive sync — disconnect, 🔄 Sync now
      🖥️ Server — app memory, 🧹 Free memory, 🔀 live / test switch, 💽 disk page

Pages keep only their own content. Fix sign-in / Drive behaviour here, once.
"""

import streamlit as st

from config import WHITELISTED_EMAILS
from library_common import (
    IS_TEST, page_title, env_banner, render_sync_status, read_pending, push_pending, memory_check,
    app_memory_mb, clear_shared_data_caches, _release_memory,
)

APP_VERSION = "1.31.19"

# Google Drive OAuth (personal Drive writes). drive_oauth raises RuntimeError, not
# ImportError, when oauth_config.py is missing — so catch everything here.
try:
    from drive_oauth import (
        get_auth_url, exchange_code_for_token, push_file_with_oauth, token_expired,
        SESSION_KEY as DRIVE_SESSION_KEY,
    )
    OAUTH_AVAILABLE = True
except Exception:
    get_auth_url = exchange_code_for_token = push_file_with_oauth = None
    token_expired = lambda token: False
    DRIVE_SESSION_KEY = "_drive_oauth_token"
    OAUTH_AVAILABLE = False


def drive_token():
    """The signed-in Drive token for this session, or None. A connection that
    has expired and can't renew itself is dropped, so the sidebar offers to
    reconnect instead of every save failing."""
    if not OAUTH_AVAILABLE:
        return None
    token = st.session_state.get(DRIVE_SESSION_KEY)
    if token and token_expired(token):
        st.session_state.pop(DRIVE_SESSION_KEY, None)
        st.session_state["_drive_expired_note"] = True
        return None
    return token


def _sign_in_screen(name: str) -> None:
    st.title(name)
    st.caption(f"v{APP_VERSION}")
    st.divider()
    _idle_note()
    st.subheader("Please sign in to continue")
    st.button("🔐 Sign in with Google", on_click=st.login, type="primary")
    st.stop()


def _access_denied() -> None:
    st.title("Access Denied")
    st.error(f"**{st.user.email}** is not authorised to use this app.")
    st.caption("Contact the administrator to request access.")
    if st.button("Sign out"):
        st.logout()
    st.stop()


def _handle_drive_callback(page_path: str) -> None:
    """Google sends the user back with ?code=...&state=<page that asked>."""
    if not OAUTH_AVAILABLE or DRIVE_SESSION_KEY in st.session_state:
        return
    qp = st.query_params.to_dict()
    if "code" not in qp:
        return
    token = exchange_code_for_token(qp["code"])
    if token:
        st.session_state[DRIVE_SESSION_KEY] = token
    return_page = qp.get("state")
    st.query_params.clear()
    if return_page and return_page != page_path:
        st.switch_page(return_page)
    st.rerun()


def _account_box(mb) -> None:
    """👤 Who's signed in, version, Drive sync status, unsaved Drive edits, Sign out —
    the same block, in the same place, on every page."""
    box = st.sidebar.container(border=True)
    box.markdown(f"👤 **{st.user.name}**")
    box.caption(f"{st.user.email}  ·  v{APP_VERSION}" + ("  ·  🧪 test copy" if IS_TEST else ""))
    # Google Drive connection — always visible.
    if not OAUTH_AVAILABLE:
        box.caption("☁️ Google Drive saving isn't set up on this copy.")
    elif IS_TEST:
        box.caption("☁️ Google Drive: saving switched off in the test copy.")
    elif drive_token():
        box.markdown("☁️ Google Drive: **:green[● Connected]**")
    else:
        if st.session_state.pop("_drive_expired_note", False):
            box.warning("Your Google Drive connection expired — please connect again.")
        box.markdown("☁️ Google Drive: **:red[● Not connected]**")
        box.markdown(f'<a href="{get_auth_url(state=_PAGE_PATH.get("p", ""))}" target="_self">'
                     f'🔗 Connect Google Drive</a>', unsafe_allow_html=True)
    try:
        render_sync_status(box)
    except Exception:
        pass  # status line is informational only

    saved = st.session_state.pop("_drive_saved_note", None)
    if saved:
        box.success(saved)
    pending = read_pending()
    if pending and not IS_TEST:
        box.warning(f"{len(pending)} file(s) have edits not yet saved to Google Drive.")
        if not drive_token():
            box.caption("Connect Google Drive (above) to save them.")
        elif box.button(f"☁️ Save {len(pending)} edit(s) to Drive", key="drive_save_pending",
                        width="stretch"):
            _save_pending(box, len(pending))

    note = st.session_state.pop("_memory_reset_note", None)
    if note:
        before, after, saved = note
        box.success(f"🧹 Memory freed: {before:,.0f} → {after:,.0f} MB." if before and after
                    else "🧹 Memory freed.")
        if saved:
            box.caption("Anything unsaved was kept and will be offered back in the planner.")

    if box.button("Sign out", key="signout_btn", width="stretch"):
        st.logout()


_PAGE_PATH = {}   # this run's page, for the Connect link (set by page_setup)

# ---------------------------------------------------------------------------
# ⏱ Sign out after a while without use
# ---------------------------------------------------------------------------
# The browser keeps the time of the last tap / click / key / scroll (or a video or song
# playing) in localStorage — shared by every tab of the app in that browser — and checks it
# every 20 s, and as soon as a sleeping tab wakes up. Past the limit it opens the page with
# ?signed_out=idle, which page_setup() answers with st.logout(). Installed once into the page
# (like the dropdown script), so Streamlit's reruns don't reset it; only real use does.
import config as _config  # noqa: E402

IDLE_SIGNOUT_MIN = float(getattr(_config, "IDLE_SIGNOUT_MIN", 60) or 0)   # minutes; 0 = never

_IDLE_CODE = r"""
(function () {
  const doc = document, win = window;
  const LIMIT_MS = __LIMIT__ * 60000, KEY = "mp_last_active", THROTTLE_MS = Math.min(15000, LIMIT_MS / 4);
  function store() { try { return win.localStorage; } catch (e) { return null; } }
  function last() { const st = store(); const v = st ? parseInt(st.getItem(KEY) || "0", 10) : 0; return isNaN(v) ? 0 : v; }
  let wrote = 0;
  function touch() {
    const t = Date.now(); if (t - wrote < THROTTLE_MS) return; wrote = t;
    try { store().setItem(KEY, String(t)); } catch (e) {}
  }
  function playing(d) {
    try {
      if ([...d.querySelectorAll("video, audio")].some(m => !m.paused && !m.ended)) return true;
      return [...d.querySelectorAll("iframe")].some(f => { try { return playing(f.contentDocument); } catch (e) { return false; } });
    } catch (e) { return false; }
  }
  function check() {
    if (playing(doc)) { wrote = 0; touch(); return false; }
    const l = last();
    if (l && Date.now() - l > LIMIT_MS) {
      try { store().setItem("mp_idle_out", "1"); store().removeItem(KEY); } catch (e) {}
      const u = new URL(win.location.href); u.search = "?signed_out=idle"; win.location.replace(u.toString());
      return true;
    }
    return false;
  }
  // A tab reopened (or reloaded by the browser) after the limit signs out straight away;
  // otherwise opening the app counts as using it.
  if (check()) return;
  touch();
  ["pointerdown", "keydown", "wheel", "touchstart", "scroll"].forEach(ev =>
    doc.addEventListener(ev, touch, {capture: true, passive: true}));
  // taps inside the app's own components (cut editor, story map editor…) count too
  new MutationObserver(() => doc.querySelectorAll("iframe").forEach(f => {
    try { if (!f.dataset.mpIdle && f.contentDocument) { f.dataset.mpIdle = "1";
          ["pointerdown", "keydown", "touchstart"].forEach(ev => f.contentDocument.addEventListener(ev, touch, true)); } }
    catch (e) {}
  })).observe(doc.body, {childList: true, subtree: true});
  win.setInterval(check, Math.min(20000, LIMIT_MS / 3));
  doc.addEventListener("visibilitychange", () => { if (doc.visibilityState === "visible") check(); });
  win.addEventListener("focus", check);
})();
"""

# Put into the PAGE itself (not this component's iframe, which Streamlit throws away on reruns).
_IDLE_JS = """
<script>
(function () {
  const doc = window.parent.document;
  if (doc.getElementById("mp-idle-signout")) return;
  const s = doc.createElement("script");
  s.id = "mp-idle-signout";
  s.textContent = __CODE__;
  doc.head.appendChild(s);
})();
</script>
"""

_IDLE_NOTE_JS = """
<div id="n" style="display:none;font-family:sans-serif;font-size:15px;color:#555;padding:4px 0">
  ⏱ You were signed out after __LIMIT__ minutes without using the app.</div>
<script>
try { if (window.parent.localStorage.getItem("mp_idle_out")) {
  document.getElementById("n").style.display = "block"; window.parent.localStorage.removeItem("mp_idle_out"); } }
catch (e) {}
</script>
"""


def _idle_signout() -> None:
    """Install the browser-side idle timer, and sign out when it asks (?signed_out=idle)."""
    if st.query_params.get("signed_out") == "idle":
        st.query_params.clear()
        st.logout()
        st.stop()
    if IDLE_SIGNOUT_MIN <= 0:
        return
    try:
        import embed
        import json as _json
        code = _IDLE_CODE.replace("__LIMIT__", f"{IDLE_SIGNOUT_MIN:g}")
        embed.html(_IDLE_JS.replace("__CODE__", _json.dumps(code).replace("</", "<\\/")), height=0)
    except Exception:
        pass  # never break the page over it


def _idle_note() -> None:
    """On the sign-in screen: say why, when the last sign-out was the idle one."""
    try:
        import embed
        embed.html(_IDLE_NOTE_JS.replace("__LIMIT__", f"{IDLE_SIGNOUT_MIN:g}"), height=32)
    except Exception:
        pass


def _save_pending(box, n: int) -> None:
    """Upload the pending edits with a progress bar and time left."""
    import time as _t
    bar = box.progress(0.0, text=f"Saving {n} file(s) to Google Drive…")
    started = _t.time()

    def progress(done, total):
        left = (_t.time() - started) / max(done, 1) * (total - done)
        eta = (f"about {left / 60:.0f} min left" if left >= 90 else f"about {left:.0f} s left") if done else ""
        bar.progress(min(done / max(total, 1), 1.0), text=f"Saved {done} of {total}  ·  {eta}")

    failed = push_pending(drive_token(), progress=progress)
    bar.empty()
    if failed:
        box.error(f"{len(failed)} file(s) couldn't be saved — they stay on the list to try again.")
        for item, err in list(failed.items())[:5]:
            box.caption(f"{item}: {err}")
    else:
        st.session_state["_drive_saved_note"] = (f"Saved {n} file(s) to Google Drive in "
                                                 f"{_t.time() - started:.0f} s.")
        st.rerun()


def _drive_panel(page_path: str) -> None:
    """☁️ Google Drive: connect / disconnect, and pull from Drive now."""
    with st.sidebar.expander("☁️ Google Drive sync", expanded=False):
        if OAUTH_AVAILABLE and not IS_TEST:
            if drive_token():
                st.caption("Connected — saves upload to your Drive.")
                if st.button("Disconnect Drive", key="drive_disconnect", width="stretch"):
                    st.session_state.pop(DRIVE_SESSION_KEY, None)
                    st.rerun()
            else:
                st.caption("Connect (link above) to save plans, projects, previews, tag corrections, "
                           "labels and library ranges to your Google Drive.")
            st.divider()
        _sync_now()


def _sync_now() -> None:
    try:
        from drive_sync import credentials_available
        available = credentials_available()
    except Exception:
        available = False
    if not available:
        st.caption("Drive sync unavailable — no service account credentials on this machine.")
        return
    st.caption("Catalogues, audio, thumbnails, sprites and preview copies sync from Drive when you press "
               "Sync now (there's no automatic sync) — e.g. after processing new videos or tracks.")
    _last_sync_line()
    _old_timer_notice()
    if st.button("🔄 Sync now", key="drive_sync_btn", width="stretch"):
        bar = st.progress(0.0, text="Starting sync…")

        def _progress(idx, total, name):
            bar.progress(min(0.99, idx / max(total, 1)), text=f"Syncing {name}…")

        from auto_sync import run_sync
        result = run_sync(_progress)
        bar.progress(1.0, text="Done.")
        if result.get("busy"):
            st.info("A background sync is running right now — try again in a minute.")
        elif result["errors"]:
            for err in result["errors"][:5]:
                st.error(err)
            st.warning("Some files failed — check Drive sharing if errors persist.")
        else:
            st.success(f"Synced {result['synced']} file(s), skipped {result['skipped']} unchanged.")
            st.cache_data.clear()
            clear_shared_data_caches()


def _last_sync_line() -> None:
    """When Drive was last synced and what it brought — under the Sync now button."""
    try:
        from auto_sync import read_sync_status
        from library_common import sync_time_text
        status = read_sync_status()
        if not status.get("finished_at"):
            st.caption("🕒 Last sync: never on this copy.")
            return
        line = f"🕒 Last sync: **{sync_time_text(status['finished_at'])}** — {status.get('synced', 0)} file(s) " \
               f"updated"
        if status.get("errors"):
            line += f", {len(status['errors'])} error(s)"
        st.caption(line + ".")
    except Exception:
        pass  # informational only


@st.cache_data(ttl=600, show_spinner=False)
def _old_sync_timer_on() -> bool:
    """True while the server's old 12-hourly sync timer (mediaplanner-sync.timer) is still
    switched on. Drive now syncs only on 🔄 Sync now."""
    import os
    import shutil
    import subprocess
    if os.name != "posix" or shutil.which("systemctl") is None:
        return False
    try:
        proc = subprocess.run(["systemctl", "is-enabled", "mediaplanner-sync.timer"],
                              capture_output=True, text=True, timeout=5)
        return proc.stdout.strip() in ("enabled", "enabled-runtime")
    except Exception:
        return False


def _old_timer_notice() -> None:
    """One-time clean-up: offer to switch off the old automatic sync timer on the server."""
    if IS_TEST or not _old_sync_timer_on():
        return
    st.warning("The old automatic 12-hour sync timer is still switched on on the server.")
    if st.button("🛑 Switch off automatic sync", key="drive_sync_timer_off", width="stretch"):
        import subprocess
        proc = subprocess.run(["systemctl", "disable", "--now", "mediaplanner-sync.timer"],
                              capture_output=True, text=True)
        _old_sync_timer_on.clear()
        if proc.returncode == 0:
            st.success("Automatic sync switched off — Drive now syncs only when you press 🔄 Sync now.")
        else:
            st.error("Couldn't switch it off: " + (proc.stderr or proc.stdout or "systemctl failed").strip()
                     + " — on the server run: sudo systemctl disable --now mediaplanner-sync.timer")


def _server_panel(mb) -> None:
    """🖥️ Server: memory, 🧹 Free memory, the live / test switch, and the Server page."""
    done = st.session_state.get("_switch_done")
    with st.sidebar.expander("🖥️ Server", expanded=bool(done)):
        if mb is not None:
            st.caption(f"🧠 App memory: {mb:,.0f} MB")
        st.button("🧹 Free memory", key="reset_memory", on_click=_reset_memory,
                  width="stretch",
                  help="Saves your work to the recovery file, then clears this page's working data, "
                       "cached data, and the videos and pictures held in memory for players. Saved "
                       "projects are untouched; unsaved planner work is offered back. For more "
                       "control (or a full reset with ♻️ Restart), open the Server page.")
        _switch_panel()
        try:
            st.page_link("pages/4_Server.py", label="🧠 Memory, 💽 disk & clean-up →")
        except Exception:
            pass


def _sidebar(page_path: str, mb) -> None:
    _PAGE_PATH["p"] = page_path
    _account_box(mb)
    _drive_panel(page_path)
    _server_panel(mb)
    st.sidebar.divider()


# Every dropdown (selectbox / multiselect) is pick-from-the-list only. Streamlit
# builds them on a searchable text input, so clicking one shows a text cursor
# and, on a phone or iPad, pops up the keyboard. This marks those inputs
# read-only (no typing, no keyboard; clicking still opens the list) — installed
# once into the page and re-applied to new dropdowns as they appear, since
# Streamlit redraws on every interaction.
_NO_DROPDOWN_TYPING_JS = """
<script>
(function () {
  const doc = window.parent.document;
  if (doc.getElementById("no-dropdown-typing")) return;
  const style = doc.createElement("style");
  // Streamlit ≤ 1.6x drew dropdowns with BaseWeb; 1.66 uses React Aria comboboxes — cover both.
  const SEL = '[data-baseweb="select"] input, .stSelectbox input[role="combobox"], .stMultiSelect input[role="combobox"]';
  style.textContent = SEL + ' { caret-color: transparent !important; cursor: pointer !important; }';
  doc.head.appendChild(style);
  const s = doc.createElement("script");
  s.id = "no-dropdown-typing";
  s.textContent = `
    (function () {
      function fix() {
        document.querySelectorAll('` + SEL + `').forEach(function (el) {
          if (!el.readOnly) { el.readOnly = true; el.setAttribute("inputmode", "none"); }
        });
      }
      fix();
      new MutationObserver(fix).observe(document.body, { childList: true, subtree: true });
      // 1.66's single-choice dropdown opens from its ▾ button (or by typing) — with typing off, a tap
      // on the text part would do nothing, so pass it on to the ▾ button.
      // A touch doesn't reach that as a click, so taps count when the finger lifts — only if it barely
      // moved (a swipe that starts on a dropdown still just scrolls the page).
      function openFor(el) {
        if (!el || el.tagName !== "INPUT" || !el.readOnly || el.getAttribute("role") !== "combobox") return;
        if (el.getAttribute("aria-expanded") === "true") return;
        var group = el.closest('[role="group"]');
        var btn = group && group.querySelector('button[aria-haspopup="listbox"]');
        if (btn && btn.getAttribute("aria-expanded") !== "true") btn.click();
      }
      var down = null;
      document.addEventListener("pointerdown", function (e) {
        down = e.pointerType === "mouse" ? null : { x: e.clientX, y: e.clientY, t: e.target };
      }, true);
      document.addEventListener("pointerup", function (e) {
        if (!down || e.pointerType === "mouse") return;
        var moved = Math.abs(e.clientX - down.x) + Math.abs(e.clientY - down.y);
        var target = down.t;
        down = null;
        if (moved < 12) setTimeout(function () { openFor(target); }, 0);
      }, true);
      document.addEventListener("click", function (e) {
        if (e.pointerType && e.pointerType !== "mouse") return;   // touch: handled on pointerup
        openFor(e.target);
      }, true);
    })();
  `;
  doc.head.appendChild(s);
})();
</script>
"""


def _switch_panel() -> None:
    """🔀 Run the live app or the test copy (only one at a time, to save memory)."""
    try:
        import app_switch as AS
        if not AS.available():
            return
    except Exception:
        return
    me, other = AS.this_copy(), AS.other_copy()
    o = AS.COPIES[other]
    done = st.session_state.get("_switch_done")
    st.markdown("**🔀 Live / test copy**")
    if done:
        st.success(f"The {o['name']} is running — this one switches off in a few seconds.")
        st.link_button(f"Open the {o['name']} →", o["url"], type="primary", width="stretch")
        st.caption("Unsaved work here is kept and offered to you when you come back.")
        return
    running = AS.is_running(other)
    st.caption(f"You're on the **{AS.COPIES[me]['name']}**. The {o['name']} is "
               f"**{'running' if running else 'off'}**."
               + (" Only one runs at a time to save the server's memory." if not running else ""))
    if st.button(f"🔀 Switch to the {o['name']}", key="switch_copy", width="stretch",
                 help=f"Starts the {o['name']}, takes you there, then switches this one off."):
        bar = st.progress(0.0, text=f"Starting the {o['name']}…")
        err = AS.start(other, progress=lambda f: bar.progress(min(max(f, 0.0), 1.0),
                                                              text=f"Starting the {o['name']}…"))
        bar.empty()
        if err:
            st.error(f"Couldn't start the {o['name']}: {err}. This copy stays on.")
            return
        err = AS.stop_later(me)
        if err:
            st.warning(f"The {o['name']} is on, but this copy couldn't switch itself off ({err}) — "
                       f"both are running.")
        st.session_state["_switch_done"] = True
        st.rerun()
    if running:
        st.link_button(f"Open the {o['name']} →", o["url"], width="stretch")
        if st.button(f"⏹ Switch the {o['name']} off", key="switch_stop_other", width="stretch",
                     help="Frees its memory. Live comes back on its own whenever the test copy stops."):
            err = AS.stop_now(other)
            if err:
                st.error(err)
            else:
                st.rerun()


# Session keys that survive a memory reset (everything else is this session's working data).
_RESET_KEEP = {"_drive_oauth_token", "_drive_expired_note", "_switch_done"}


def session_keep_keys() -> set:
    """Session keys that belong to the signed-in user rather than the work on a
    page (Drive connection, live/test switch) — kept when a page starts fresh."""
    return _RESET_KEEP | ({DRIVE_SESSION_KEY} if DRIVE_SESSION_KEY else set())


def _reset_memory() -> None:
    """🧹 Free memory: write unsaved work to its recovery file, drop this session's
    working data and the shared data caches, and hand the memory back."""
    before = app_memory_mb()
    hook = st.session_state.get("_before_reset")   # the planner registers its autosave here
    saved = False
    if callable(hook):
        try:
            hook()
            saved = True
        except Exception:
            pass
    keep = session_keep_keys()
    for k in list(st.session_state.keys()):
        if k not in keep:
            try:
                del st.session_state[k]
            except Exception:
                pass
    from library_common import free_shared_memory
    free_shared_memory()          # cached data + videos/pictures held for players, then hand back
    after = app_memory_mb()
    st.session_state["_memory_reset_note"] = (before, after, saved)


def _lock_dropdown_typing() -> None:
    try:
        import embed
        embed.html(_NO_DROPDOWN_TYPING_JS, height=0)
    except Exception:
        pass  # cosmetic only — never break the page over it


def page_setup(name: str, page_path: str, *, title: str = None, anchor: str = None) -> None:
    """Call first on every page.

    name: short page name (tab title, sign-in screen)
    page_path: this page's script path, e.g. "pages/2_Reviewer.py" — where
        Drive connect returns to
    title: heading shown on the page (default: name)
    anchor: HTML anchor for the heading (e.g. for a "Back to top" link)"""
    st.set_page_config(page_title=page_title(name), layout="wide")
    env_banner()
    _lock_dropdown_typing()
    if not st.user.is_logged_in:
        _sign_in_screen(name)
    if st.user.email not in WHITELISTED_EMAILS:
        _access_denied()
    _idle_signout()
    _handle_drive_callback(page_path)
    st.title(title or name, anchor=anchor)
    mb = memory_check(name)
    _sidebar(page_path, mb)


def top_button(anchor: str, css: str = "") -> None:
    """A round ⬆️ button that floats in the bottom-right corner on long pages and
    jumps back to the page heading (page_setup(..., anchor=anchor)). Always in
    reach — no scrolling to the bottom to find it (iPad friendly).
    css: more page CSS to inject in the same element (e.g. wrap_css()), so the page
    doesn't get an extra empty row."""
    st.markdown(
        f'<a href="#{anchor}" target="_self" class="mp-top-btn" title="Back to the top">⬆️</a>'
        "<style>.mp-top-btn{position:fixed;right:1.2rem;bottom:1.4rem;z-index:999990;width:3rem;height:3rem;"
        "display:flex;align-items:center;justify-content:center;border-radius:50%;font-size:1.35rem;"
        "text-decoration:none!important;background:rgba(127,127,127,.22);"
        "backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);"
        "border:1px solid rgba(128,128,128,.45);box-shadow:0 2px 10px rgba(0,0,0,.18);opacity:.92}"
        ".mp-top-btn:hover {opacity:1;box-shadow:0 3px 14px rgba(0,0,0,.28)}" + css + "</style>",
        unsafe_allow_html=True)


def menu_css() -> str:
    """CSS for page menus made of horizontal radios (same keys and values as always):
    a radio inside a container keyed "mainnav…" is the page's main menu — big, equal-width
    tabs with the chosen one underlined; one inside "subnav…" is a sub menu — small rounded
    chips, so the two rows never look alike."""
    m, s_ = "[class*='st-key-mainnav']", "[class*='st-key-subnav']"
    # (a space before every "{": Streamlit's markdown reads ":first-child{…}" as a directive and drops it)
    return (
        f"{m} .stElementContainer,{m} .stRadio {{width:100%!important}}"
        f"{m} [role='radiogroup'] {{display:grid!important;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));"
        f"gap:.5rem;width:100%}}"
        f"{m} [role='radiogroup']>div {{margin:0!important;border-radius:.6rem;border:1px solid rgba(128,128,128,.35);"
        f"background:rgba(128,128,128,.07)}}"
        f"{m} [role='radiogroup']>div:hover {{border-color:rgba(255,75,75,.6)}}"
        f"{m} [role='radiogroup']>div[data-selected='true'] {{border-color:#ff4b4b;background:rgba(255,75,75,.12);"
        f"box-shadow:inset 0 -3px 0 #ff4b4b}}"
        f"{m} label[data-testid='stRadioOption'] {{width:100%;justify-content:center;padding:.65rem .8rem;margin:0;"
        f"cursor:pointer}}"
        f"{m} label[data-testid='stRadioOption']>div>div:first-child {{display:none}}"
        f"{m} [data-testid='stMarkdownContainer'] p {{font-weight:650;font-size:1.02rem;margin:0}}"
        f"{m} {{margin-bottom:.4rem}}"
        f"{s_} [role='radiogroup'] {{display:flex!important;flex-wrap:wrap;gap:.4rem}}"
        f"{s_} [role='radiogroup']>div {{margin:0!important;border-radius:999px;border:1px solid rgba(128,128,128,.30);"
        f"background:transparent}}"
        f"{s_} [role='radiogroup']>div:hover {{border-color:rgba(255,75,75,.55)}}"
        f"{s_} [role='radiogroup']>div[data-selected='true'] {{border-color:rgba(255,75,75,.7);"
        f"background:rgba(255,75,75,.10)}}"
        f"{s_} label[data-testid='stRadioOption'] {{padding:.22rem .8rem;margin:0;cursor:pointer}}"
        f"{s_} label[data-testid='stRadioOption']>div>div:first-child {{display:none}}"
        f"{s_} [data-testid='stMarkdownContainer'] p {{font-size:.88rem;margin:0}}"
    )


def wrap_css(prefix: str, min_px: int = 135) -> str:
    """CSS: rows of columns inside containers keyed "<prefix>…" wrap onto a second line on a
    narrow page (iPad portrait with the sidebar open) instead of squashing labels to "…"."""
    return (f"[class*='st-key-{prefix}'] [data-testid='stHorizontalBlock']{{flex-wrap:wrap;row-gap:.5rem}}"
            f"[class*='st-key-{prefix}'] [data-testid='stColumn']{{min-width:min({min_px}px,100%)}}")
