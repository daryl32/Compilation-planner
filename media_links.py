"""
Play videos, music and pictures straight from disk through nginx, instead of
through the app's memory.

Why: st.video / st.audio / st.image given a file path read the WHOLE file into
the app's memory and keep it there while any page shows it — a 480p preview
copy can be hundreds of MB, a render tens of MB. That's most of what filled the
server's memory. Given a web address instead, the browser fetches the file
from nginx (with seeking), and the app never holds the bytes.

The addresses are signed and expire (nginx's secure_link module), so only
pages of the signed-in app can hand them out — nobody can guess one. nginx
reads the data through a read-only mount at /srv/mediaplanner/<data folder>.

Set up once on the server with:  sudo bash deploy/setup_media.sh
Until then (and on the laptop) everything falls back to the old way.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
import urllib.parse
from pathlib import Path

import streamlit as st

SECRET_FILE = Path("/etc/mediaplanner/media.secret")
# Server folders nginx can serve, and the address each is served under.
ROOTS = {"/root/data": "/media/data", "/root/data-dev": "/media/data-dev"}
LINK_HOURS = 12

_secret_cache = {"mtime": None, "secret": None}


def _secret() -> str | None:
    try:
        mtime = SECRET_FILE.stat().st_mtime
    except OSError:
        return None
    if _secret_cache["mtime"] != mtime:
        try:
            _secret_cache["secret"] = SECRET_FILE.read_text().strip() or None
        except OSError:
            _secret_cache["secret"] = None
        _secret_cache["mtime"] = mtime
    return _secret_cache["secret"]


def enabled() -> bool:
    return _secret() is not None


def _host() -> str | None:
    try:
        h = st.context.headers.get("Host")
        if h:
            return h
    except Exception:
        pass
    try:
        import config
        return getattr(config, "PUBLIC_HOST", None)
    except Exception:
        return None


def url_for(path) -> str | None:
    """A signed https address for a file under one of ROOTS, or None (not set
    up, file outside those folders, or no host known)."""
    secret = _secret()
    if not secret:
        return None
    try:
        # Not resolve(): a folder that's a link to a mounted volume (e.g. the
        # originals) keeps its /root/data address; nginx follows the link.
        real = Path(os.path.normpath(os.path.abspath(str(path))))
        mode = real.stat().st_mode
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not mode & 0o004:
        return None   # nginx can't read a file only root may read — play it the old way
    for root, prefix in ROOTS.items():
        try:
            rel = real.relative_to(root)
        except ValueError:
            continue
        host = _host()
        if not host:
            return None
        uri = f"{prefix}/{rel.as_posix()}"
        # The expiry moves in whole hours, so a file keeps the same address across
        # page reruns (a changing address would restart a playing video) and the
        # browser can cache it. Valid for LINK_HOURS to LINK_HOURS + 1 hours.
        expires = (int(time.time()) // 3600 + LINK_HOURS + 1) * 3600
        digest = hashlib.md5(f"{expires}{uri} {secret}".encode()).digest()
        key = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        return f"https://{host}{urllib.parse.quote(uri)}?e={expires}&k={key}"
    return None


def play_video(path, **kwargs) -> None:
    """st.video for a file on disk, without loading it into memory when possible."""
    st.video(url_for(path) or str(path), **kwargs)


def play_audio(path, **kwargs) -> None:
    st.audio(url_for(path) or str(path), **kwargs)


def image_source(path):
    """What to give st.image for a picture on disk: its address, or the path."""
    return url_for(path) or str(path)
