"""
Shared setup for every page of the app — call page_setup() first thing on a page.

Does, in order:
  • page config (wide layout; tab title marked 🧪 TEST in the test copy) + test banner
  • Google sign-in screen if not signed in, then the email whitelist gate
  • Google Drive connect callback (?code=...), returning to the page that asked
  • page title + "v<version> · <name>" caption
  • sidebar: Drive sync status, a "☁️ Google Drive" connect/disconnect panel,
    "edits not yet saved to Drive" + Save button, and Sign out

Pages keep only their own content. Fix sign-in / Drive behaviour here, once.
"""

import streamlit as st

from config import WHITELISTED_EMAILS
from library_common import (
    IS_TEST, page_title, env_banner, render_sync_status, read_pending, push_pending,
)

APP_VERSION = "1.7.1"

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


def _sidebar(page_path: str) -> None:
    try:
        render_sync_status()
    except Exception:
        pass  # status line is informational only

    if OAUTH_AVAILABLE:
        with st.sidebar.expander("☁️ Google Drive", expanded=False):
            if IS_TEST:
                st.caption("Saving to Google Drive is switched off in the test copy.")
            elif drive_token():
                st.success("Connected — saves upload to your Drive.")
                if st.button("Disconnect Drive", key="drive_disconnect"):
                    st.session_state.pop(DRIVE_SESSION_KEY, None)
                    st.rerun()
            else:
                if st.session_state.pop("_drive_expired_note", False):
                    st.warning("Your Google Drive connection expired — please connect again.")
                st.caption("Connect to save plans, projects, previews, tag corrections, labels "
                           "and library ranges to your Google Drive.")
                auth_url = get_auth_url(state=page_path)
                st.markdown(f'<a href="{auth_url}" target="_self">🔗 Connect Google Drive</a>',
                            unsafe_allow_html=True)

    pending = read_pending()
    if pending and not IS_TEST:
        st.sidebar.warning(f"{len(pending)} file(s) have edits not yet saved to Google Drive.")
        if not drive_token():
            st.sidebar.caption("Connect Google Drive (above) to save them.")
        elif st.sidebar.button("☁️ Save edits to Drive", key="drive_save_pending"):
            failed = push_pending(drive_token())
            if failed:
                for item, err in failed.items():
                    st.sidebar.error(f"{item}: {err}")
            else:
                st.sidebar.success("Saved to Google Drive.")
                st.rerun()

    if st.sidebar.button("Sign out", key="signout_btn"):
        st.logout()
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
  style.textContent = '[data-baseweb="select"] input { caret-color: transparent !important; cursor: pointer !important; }';
  doc.head.appendChild(style);
  const s = doc.createElement("script");
  s.id = "no-dropdown-typing";
  s.textContent = `
    (function () {
      function fix() {
        document.querySelectorAll('[data-baseweb="select"] input').forEach(function (el) {
          if (!el.readOnly) { el.readOnly = true; el.setAttribute("inputmode", "none"); }
        });
      }
      fix();
      new MutationObserver(fix).observe(document.body, { childList: true, subtree: true });
    })();
  `;
  doc.head.appendChild(s);
})();
</script>
"""


def _lock_dropdown_typing() -> None:
    try:
        import streamlit.components.v1 as components
        components.html(_NO_DROPDOWN_TYPING_JS, height=0)
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
    _handle_drive_callback(page_path)
    st.title(title or name, anchor=anchor)
    st.caption(f"v{APP_VERSION}  ·  {st.user.name}")
    _sidebar(page_path)
