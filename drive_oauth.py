"""
Google Drive OAuth flow for the Compilation Planner.

Uses the Web application OAuth client to authenticate as the user's personal
Google account — allowing writes to their personal Drive (service accounts
cannot write to personal Drive storage).

Credentials are stored in Streamlit session state only — cleared on page
refresh, as intended for per-session auth.

Usage in planner_app.py:
    from drive_oauth import (
        get_auth_url, exchange_code_for_token,
        push_file_with_oauth, is_authenticated
    )
"""

import json
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
        "access_type":   "online",
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
        return resp.json()
    return None


def _get_service(token_dict: dict):
    """Build a Drive API service from a token dict."""
    creds = Credentials(
        token=token_dict["access_token"],
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

    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
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
    try:
        service = _get_service(token_dict)
        import mimetypes

        # Walk/create the folder hierarchy
        parts = [p for p in drive_path.split("/") if p]
        parent_id = None
        for part in parts:
            parent_id = _find_or_create_folder(service, part, parent_id)

        mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"

        # Update if exists, create if not
        q = f"name='{local_path.name}' and '{parent_id}' in parents and trashed=false"
        resp = service.files().list(q=q, fields="files(id)").execute()
        existing = resp.get("files", [])

        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
        if existing:
            service.files().update(
                fileId=existing[0]["id"],
                media_body=media,
            ).execute()
        else:
            service.files().create(
                body={"name": local_path.name, "parents": [parent_id]},
                media_body=media,
            ).execute()
        return None
    except Exception as e:
        return str(e)


def is_authenticated(session_state) -> bool:
    """Check if the session has a valid OAuth token."""
    return bool(session_state.get(SESSION_KEY))
