"""
Google Drive OAuth flow for the Compilation Planner.

Uses the Web application OAuth client to authenticate as the user's personal
Google account — allowing writes to their personal Drive (service accounts
cannot write to personal Drive storage).

Credentials are stored in Streamlit session state only — cleared on page
refresh, as intended for per-session auth.

Usage (see page_setup.py):
    from drive_oauth import (
        get_auth_url, exchange_code_for_token,
        push_file_with_oauth, is_authenticated
    )
"""

import json
import time
import urllib.parse
import requests
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

# ---------------------------------------------------------------------------
# OAuth client config — loaded from oauth_config.py (gitignored)
# ---------------------------------------------------------------------------
try:
    from oauth_config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI
except ImportError:
    raise RuntimeError(
        "oauth_config.py not found. Copy oauth_config.example.py to "
        "oauth_config.py and fill in your credentials."
    )
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

AUTH_URL   = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL  = "https://oauth2.googleapis.com/token"

SESSION_KEY = "_drive_oauth_token"


def get_auth_url(state: str = "") -> str:
    """Generate the Google OAuth authorisation URL."""
    params = {
        "client_id":     CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "scope":         " ".join(SCOPES),
        # offline: Google also returns a refresh token, so the connection keeps
        # working after the 1-hour access token expires (with "online" there
        # is none, and saving after an hour failed with a refresh error).
        "access_type":   "offline",
        "prompt":        "consent",
    }
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_token(code: str) -> dict | None:
    """Exchange an auth code for an access token. Returns token dict or None."""
    resp = requests.post(TOKEN_URL, data={
        "code":          code,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri":  REDIRECT_URI,
        "grant_type":    "authorization_code",
    })
    if resp.status_code == 200:
        token = resp.json()
        token["obtained_at"] = time.time()
        return token
    return None


EXPIRED_MESSAGE = ("Your Google Drive connection has expired — click Disconnect Drive, then "
                   "Connect Google Drive again (your edits are kept and will be saved after).")


def token_expired(token_dict: dict) -> bool:
    """True when the access token is past its lifetime and can't be renewed
    (a connection made before refresh tokens were requested)."""
    if not token_dict or token_dict.get("refresh_token"):
        return False
    obtained = float(token_dict.get("obtained_at", 0))
    lifetime = float(token_dict.get("expires_in", 3600))
    return not obtained or time.time() > obtained + lifetime - 60


def _get_service(token_dict: dict):
    """Build a Drive API service from a token dict."""
    creds = Credentials(
        token=token_dict["access_token"],
        refresh_token=token_dict.get("refresh_token"),
        token_uri=TOKEN_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds)


# Hardcoded ID of the existing scene-labeling folder — prevents the OAuth
# flow from creating a duplicate if the folder name search returns no results
# (e.g. when the folder is owned by a different account than the OAuth user).
SCENE_LABELING_FOLDER_ID = "18q5Ib4g5vgwmjEU96tUboXyV8JLrPDkP"
CATALOGUE_FOLDER_ID = "1r83HvfcmNm0bmzoUefoXdPwSbhvLRoOr"


def _find_or_create_folder(service, name: str, parent_id: str = None) -> str:
    """Get or create a Drive folder, return its ID."""
    # Use hardcoded ID for the root scene-labeling folder
    if name == "scene-labeling" and parent_id is None:
        return SCENE_LABELING_FOLDER_ID
    if name == "catalogue":
        return CATALOGUE_FOLDER_ID

    from drive_sync import q_escape
    q = f"name='{q_escape(name)}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    resp = service.files().list(q=q, fields="files(id)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]
    folder = service.files().create(body=body, fields="id").execute()
    return folder["id"]


def _service_account():
    """The server's service-account Drive client (it can see every file in the
    shared folders, unlike the user's drive.file login), or None if this machine
    has no service account."""
    try:
        from drive_sync import credentials_available, _get_service as _sa_service
        return _sa_service() if credentials_available() else None
    except Exception:
        return None


def _upload_via_lookup(service, sa, local_path: Path, parts: list, mime_type: str) -> dict:
    """Upload using the service account to FIND things and the user's login only
    to CREATE them, so nothing is duplicated:
      folders  — each level is the existing folder of that name (the oldest, i.e.
                 the original, if there are already duplicates); only a level
                 that doesn't exist at all is created.
      the file — an existing file of that name is updated in place (with the
                 user's login if the app created it, else the service account);
                 a new file is created with the user's login."""
    from drive_sync import _find_folders, q_escape
    parent_id = None
    for part in parts:
        if part == "scene-labeling" and parent_id is None:
            parent_id = SCENE_LABELING_FOLDER_ID
            continue
        if part == "catalogue" and parent_id == SCENE_LABELING_FOLDER_ID:
            parent_id = CATALOGUE_FOLDER_ID
            continue
        found = _find_folders(sa, part, parent_id)
        if found:
            parent_id = found[0]["id"]
        else:
            body = {"name": part, "mimeType": "application/vnd.google-apps.folder"}
            if parent_id:
                body["parents"] = [parent_id]
            parent_id = service.files().create(body=body, fields="id").execute()["id"]

    resp = sa.files().list(
        q=f"name='{q_escape(local_path.name)}' and '{parent_id}' in parents and trashed=false",
        fields="files(id, modifiedTime)", orderBy="modifiedTime desc",
    ).execute()
    existing = resp.get("files", [])
    if existing:
        file_id = existing[0]["id"]
        for client in (service, sa):
            try:
                media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
                return client.files().update(fileId=file_id, media_body=media, fields="id, size").execute()
            except Exception as e:
                if not any(x in str(e) for x in ("404", "403", "notFound", "insufficient")):
                    raise
        # Neither could write to it: fall through and add a new copy (the sync
        # always takes the newest copy of a name).
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    return service.files().create(body={"name": local_path.name, "parents": [parent_id]},
                                  media_body=media, fields="id, size").execute()


def upload_file_with_oauth(token_dict: dict, local_path: Path, drive_path: str) -> tuple:
    """Upload (or replace) local_path in drive_path (e.g. "scene-labeling/previews")
    in the user's Drive. Returns ({"id", "size"}, None) or (None, error string)."""
    from drive_sync import drive_writes_blocked
    blocked = drive_writes_blocked()
    if blocked:
        return None, blocked
    if token_expired(token_dict):
        return None, EXPIRED_MESSAGE
    try:
        service = _get_service(token_dict)
        import mimetypes
        mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        parts = [p for p in drive_path.split("/") if p]

        sa = _service_account()
        if sa is not None:
            f = _upload_via_lookup(service, sa, local_path, parts, mime_type)
        else:
            # No service account here (e.g. the laptop): the old way. With the
            # drive.file permission this can only see what the app created, so it
            # may add a second folder/file with the same name — the sync copes.
            parent_id = None
            for part in parts:
                parent_id = _find_or_create_folder(service, part, parent_id)
            from drive_sync import q_escape
            q = f"name='{q_escape(local_path.name)}' and '{parent_id}' in parents and trashed=false"
            existing = service.files().list(q=q, fields="files(id)").execute().get("files", [])
            media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
            if existing:
                f = service.files().update(fileId=existing[0]["id"], media_body=media,
                                           fields="id, size").execute()
            else:
                f = service.files().create(body={"name": local_path.name, "parents": [parent_id]},
                                           media_body=media, fields="id, size").execute()
        return {"id": f["id"], "size": int(f.get("size") or 0)}, None
    except Exception as e:
        if "refresh" in str(e).lower() and ("token" in str(e).lower() or "credentials" in str(e).lower()):
            return None, EXPIRED_MESSAGE
        return None, str(e)


def push_file_with_oauth(
    token_dict: dict,
    local_path: Path,
    drive_path: str,       # e.g. "scene-labeling/compilation_plans"
) -> str | None:
    """
    Upload local_path to drive_path in the user's Drive.
    drive_path is a slash-separated folder path from Drive root.
    Returns None on success, error string on failure.
    """
    _, err = upload_file_with_oauth(token_dict, local_path, drive_path)
    return err


def trash_file_with_oauth(token_dict: dict, file_id: str) -> str | None:
    """Move a file this app uploaded to the Drive bin (recoverable for 30 days).
    Returns None on success (or if it's already gone), error string on failure."""
    from drive_sync import drive_writes_blocked
    blocked = drive_writes_blocked()
    if blocked:
        return blocked
    if token_expired(token_dict):
        return EXPIRED_MESSAGE
    try:
        _get_service(token_dict).files().update(fileId=file_id, body={"trashed": True}).execute()
        return None
    except Exception as e:
        if "404" in str(e) or "notFound" in str(e):
            return None
        return str(e)


def is_authenticated(session_state) -> bool:
    """Check if the session has a valid OAuth token."""
    return bool(session_state.get(SESSION_KEY))


# ---------------------------------------------------------------------------
# Many files at once (☁️ Save edits to Drive)
# ---------------------------------------------------------------------------
# One file at a time cost ~6 round trips each (folder lookups, a file lookup, a
# two-step upload): 247 files took over 10 minutes. Here each folder is looked
# up ONCE and its files listed ONCE, then each file is a single upload request,
# several in parallel.

SMALL_FILE_BYTES = 5 * 2**20   # below this, a one-request (non-resumable) upload


def _resolve_folder(service, sa, parts: list) -> str:
    """The id of the folder at parts (created with the user's login if missing)."""
    if sa is None:
        parent_id = None
        for part in parts:
            parent_id = _find_or_create_folder(service, part, parent_id)
        return parent_id
    from drive_sync import _find_folders
    parent_id = None
    for part in parts:
        if part == "scene-labeling" and parent_id is None:
            parent_id = SCENE_LABELING_FOLDER_ID
            continue
        if part == "catalogue" and parent_id == SCENE_LABELING_FOLDER_ID:
            parent_id = CATALOGUE_FOLDER_ID
            continue
        found = _find_folders(sa, part, parent_id)
        if found:
            parent_id = found[0]["id"]
        else:
            body = {"name": part, "mimeType": "application/vnd.google-apps.folder"}
            if parent_id:
                body["parents"] = [parent_id]
            parent_id = service.files().create(body=body, fields="id").execute()["id"]
    return parent_id


def _names_in(client, folder_id: str) -> dict:
    """{file name: id} in a folder — the newest file when a name appears twice."""
    out, page_token = {}, None
    while True:
        resp = client.files().list(
            q=f"'{folder_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'",
            fields="nextPageToken, files(id, name, modifiedTime)", orderBy="modifiedTime desc",
            pageSize=1000, pageToken=page_token).execute()
        for f in resp.get("files", []):
            out.setdefault(f["name"], f["id"])
        page_token = resp.get("nextPageToken")
        if not page_token:
            return out


def _friendly(e: Exception) -> str:
    s = str(e)
    if "refresh" in s.lower() and ("token" in s.lower() or "credentials" in s.lower()):
        return EXPIRED_MESSAGE
    return s


def upload_many(token_dict: dict, items: list, progress=None, workers: int = 6) -> dict:
    """Upload many files. items: [(key, local_path, drive_path)]. progress(done,
    total, key, error_or_None) is called from THIS thread as each one finishes.
    Returns {key: error} for the ones that failed."""
    import mimetypes
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from drive_sync import drive_writes_blocked

    if not items:
        return {}
    blocked = drive_writes_blocked()
    if blocked:
        return {k: blocked for k, _, _ in items}
    if token_expired(token_dict):
        return {k: EXPIRED_MESSAGE for k, _, _ in items}

    failed, done, total = {}, 0, len(items)

    def report(key, err):
        nonlocal done
        done += 1
        if err:
            failed[key] = err
        if progress:
            progress(done, total, key, err)

    # 1. Each folder once: its id and the files already in it.
    try:
        service, sa = _get_service(token_dict), _service_account()
    except Exception as e:
        return {k: _friendly(e) for k, _, _ in items}
    folders = {}
    for dp in sorted({dp for _, _, dp in items}):
        try:
            fid = _resolve_folder(service, sa, [p for p in dp.split("/") if p])
            folders[dp] = (fid, _names_in(sa or service, fid))
        except Exception as e:
            folders[dp] = (None, _friendly(e))

    # 2. One request per file, several at a time (each thread its own clients —
    #    the Google client isn't safe to share between threads).
    local = threading.local()

    def clients():
        if not hasattr(local, "service"):
            local.service, local.sa = _get_service(token_dict), _service_account()
        return local.service, local.sa

    def one(path: Path, dp: str):
        fid, names = folders[dp]
        if fid is None:
            raise RuntimeError(names)
        svc, sac = clients()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        resumable = path.stat().st_size > SMALL_FILE_BYTES
        file_id = names.get(path.name)
        if file_id:
            last = None
            for client in (svc, sac):
                if client is None:
                    continue
                try:
                    media = MediaFileUpload(str(path), mimetype=mime, resumable=resumable)
                    return client.files().update(fileId=file_id, media_body=media, fields="id").execute()
                except Exception as e:
                    last = e
                    if not any(x in str(e) for x in ("404", "403", "notFound", "insufficient")):
                        raise
            # Neither login can write that copy: add a new one (the sync takes the newest).
        media = MediaFileUpload(str(path), mimetype=mime, resumable=resumable)
        return svc.files().create(body={"name": path.name, "parents": [fid]},
                                  media_body=media, fields="id").execute()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(one, Path(path), dp): key for key, path, dp in items}
        for fut in as_completed(futs):
            try:
                fut.result()
                report(futs[fut], None)
            except Exception as e:
                report(futs[fut], _friendly(e))
    return failed
