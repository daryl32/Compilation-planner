"""
Google Drive sync helper — replaces rclone for two-way sync between
the server and Google Drive using a service account.

Service account credentials: /root/drive-credentials.json
Never expires, no browser auth needed, no token refresh issues.

Pull (Drive → server):
  - scene-labeling/catalogue        → CATALOGUE_DIR
  - scene-labeling/audio_catalogue  → AUDIO_DIR
  - scene-labeling/catalogue/thumbnails       → CATALOGUE_DIR/thumbnails
  - scene-labeling/catalogue/timeline_sprites → CATALOGUE_DIR/timeline_sprites

Push (server → Drive):
  - PLANS_DIR   → scene-labeling/compilation_plans
  - PROJECTS_DIR → scene-labeling/projects
  - PREVIEW_DIR  → scene-labeling/previews
"""

import os
import io
import json
from datetime import datetime
from pathlib import Path

CREDENTIALS_PATH = "/root/drive-credentials.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Folder names in Google Drive (top-level shared folders)
DRIVE_SCENE_LABELING = "scene-labeling"


def _get_service():
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def _find_folder(service, name: str, parent_id: str = None) -> str | None:
    """Return the Drive folder ID for `name`, optionally within `parent_id`."""
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    resp = service.files().list(q=q, fields="files(id, name)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _list_files(service, folder_id: str) -> list[dict]:
    """List all non-folder files directly in a Drive folder."""
    results = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false",
            fields="nextPageToken, files(id, name, modifiedTime, size)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def _download_file(service, file_id: str, dest_path: Path) -> None:
    """Download a Drive file to dest_path."""
    from googleapiclient.http import MediaIoBaseDownload
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _upload_file(service, local_path: Path, folder_id: str) -> None:
    """Upload or update a file in a Drive folder."""
    from googleapiclient.http import MediaFileUpload
    import mimetypes

    mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"

    # Check if file already exists in folder
    resp = service.files().list(
        q=f"name='{local_path.name}' and '{folder_id}' in parents and trashed=false",
        fields="files(id)",
    ).execute()
    existing = resp.get("files", [])

    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)

    if existing:
        service.files().update(
            fileId=existing[0]["id"],
            media_body=media,
        ).execute()
    else:
        service.files().create(
            body={"name": local_path.name, "parents": [folder_id]},
            media_body=media,
        ).execute()


def _ensure_drive_folder(service, name: str, parent_id: str) -> str:
    """Get or create a subfolder in Drive, return its ID."""
    folder_id = _find_folder(service, name, parent_id)
    if folder_id:
        return folder_id
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=file_metadata, fields="id").execute()
    return folder["id"]


def _local_is_newer(local_path: Path, drive_file: dict) -> bool:
    """True if local_path was modified after the Drive copy was last modified."""
    try:
        mt = drive_file.get("modifiedTime")
        if not mt:
            return False
        drive_ts = datetime.fromisoformat(mt.replace("Z", "+00:00")).timestamp()
        return local_path.stat().st_mtime > drive_ts
    except Exception:
        return False


def sync_pull(catalogue_dir: Path, audio_dir: Path, progress_callback=None,
              proxy_dir: Path = None) -> dict:
    """
    Pull catalogue JSONs, audio JSONs, thumbnails and sprites from Drive to server,
    plus preview copies (scene-labeling/proxies) into proxy_dir if given.
    Only downloads files that are newer on Drive or missing locally.
    Returns {"synced": N, "skipped": N, "errors": [str]}
    """
    service = _get_service()
    scene_id = _find_folder(service, DRIVE_SCENE_LABELING)
    if not scene_id:
        return {"synced": 0, "skipped": 0, "errors": [f"'{DRIVE_SCENE_LABELING}' folder not found in Drive"]}

    tasks = []

    # catalogue JSONs
    cat_folder_id = _find_folder(service, "catalogue", scene_id)
    if cat_folder_id:
        for f in _list_files(service, cat_folder_id):
            tasks.append((f, catalogue_dir / f["name"]))

        # thumbnails (subfolder of catalogue)
        thumb_id = _find_folder(service, "thumbnails", cat_folder_id)
        if thumb_id:
            # thumbnails are in per-video subfolders
            resp = service.files().list(
                q=f"'{thumb_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)",
            ).execute()
            for vid_folder in resp.get("files", []):
                for f in _list_files(service, vid_folder["id"]):
                    tasks.append((f, catalogue_dir / "thumbnails" / vid_folder["name"] / f["name"]))

        # timeline_sprites (subfolder of catalogue)
        sprite_id = _find_folder(service, "timeline_sprites", cat_folder_id)
        if sprite_id:
            for f in _list_files(service, sprite_id):
                tasks.append((f, catalogue_dir / "timeline_sprites" / f["name"]))

    # preview copies made in Colab (backfill proxies.py)
    if proxy_dir is not None:
        proxies_id = _find_folder(service, "proxies", scene_id)
        if proxies_id:
            for f in _list_files(service, proxies_id):
                if f["name"].endswith(".mp4") and ".part" not in f["name"]:
                    tasks.append((f, proxy_dir / f["name"]))

    # audio_catalogue JSONs
    audio_folder_id = _find_folder(service, "audio_catalogue", scene_id)
    if audio_folder_id:
        for f in _list_files(service, audio_folder_id):
            tasks.append((f, audio_dir / f["name"]))

    synced = skipped = 0
    errors = []
    total = len(tasks)

    for idx, (drive_file, local_path) in enumerate(tasks):
        if progress_callback:
            progress_callback(idx, total, local_path.name)
        try:
            # Skip if local file exists and is same size (quick check)
            if local_path.exists() and local_path.stat().st_size == int(drive_file.get("size", 0)):
                skipped += 1
                continue
            # Never overwrite a local file that is newer than the Drive copy — e.g. a
            # Reviewer edit (excluded / intro / outro / corrected tags) not yet pushed to Drive.
            if local_path.exists() and _local_is_newer(local_path, drive_file):
                skipped += 1
                continue
            _download_file(service, drive_file["id"], local_path)
            synced += 1
        except Exception as e:
            errors.append(f"{drive_file['name']}: {e}")

    return {"synced": synced, "skipped": skipped, "errors": errors}


def download_source_video(drive_rel_path: str, dest: Path, progress_callback=None) -> str | None:
    """
    Download one source video from Drive to `dest`.

    `drive_rel_path` is the path below "My Drive", e.g. "Videos-PH/clip.mp4"
    (i.e. the part of a catalogue's source_path after "MyDrive/"). The first
    folder must be shared with the service account. Skips the download if
    `dest` already exists with the same size.
    Returns None on success, an error string on failure.
    """
    try:
        from googleapiclient.http import MediaIoBaseDownload
        parts = [p for p in drive_rel_path.replace("\\", "/").split("/") if p]
        if len(parts) < 2:
            return f"Unexpected Drive path: {drive_rel_path}"
        service = _get_service()

        parent_id = None
        for folder_name in parts[:-1]:
            parent_id = _find_folder(service, folder_name, parent_id)
            if not parent_id:
                return (f"Folder '{folder_name}' not found in Drive — is it shared with "
                        f"the service account?")

        file_name = parts[-1]
        safe = file_name.replace("'", "\\'")
        resp = service.files().list(
            q=f"name='{safe}' and '{parent_id}' in parents and trashed=false",
            fields="files(id, name, size)",
        ).execute()
        files = resp.get("files", [])
        if not files:
            return f"'{file_name}' not found in Drive folder '{parts[-2]}'"
        drive_file = files[0]
        size = int(drive_file.get("size", 0))

        if dest.exists() and size and dest.stat().st_size == size:
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        request = service.files().get_media(fileId=drive_file["id"])
        with open(tmp, "wb") as f:
            downloader = MediaIoBaseDownload(f, request, chunksize=32 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if progress_callback and status:
                    progress_callback(status.progress())
        tmp.replace(dest)
        return None
    except Exception as e:
        return str(e)


AUDIO_FILE_EXTENSIONS = (".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus")


def list_drive_audio(folder_path: str = "Audio-Library") -> tuple:
    """Every audio file under a My Drive folder (and its subfolders).
    folder_path is below "My Drive", e.g. "Audio-Library" — it must be shared
    with the service account. Returns ([{"name", "rel", "size"}], error or None);
    "rel" is the path below My Drive, as used in a catalogue's source_path."""
    try:
        service = _get_service()
        parent_id = None
        for part in [p for p in folder_path.split("/") if p]:
            parent_id = _find_folder(service, part, parent_id)
            if not parent_id:
                return [], f"Folder '{part}' not found in Drive — is it shared with the service account?"
        out, stack = [], [(parent_id, folder_path.strip("/"))]
        while stack:
            fid, rel = stack.pop()
            for f in _list_files(service, fid):
                if f["name"].lower().endswith(AUDIO_FILE_EXTENSIONS):
                    out.append({"name": f["name"], "rel": f"{rel}/{f['name']}", "size": int(f.get("size", 0))})
            resp = service.files().list(
                q=f"'{fid}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)", pageSize=1000,
            ).execute()
            for sub in resp.get("files", []):
                stack.append((sub["id"], f"{rel}/{sub['name']}"))
        return sorted(out, key=lambda f: f["rel"].lower()), None
    except Exception as e:
        return [], str(e)


def drive_writes_blocked() -> str | None:
    """The test copy (config.DRIVE_WRITES = False) never writes to Google Drive.
    Returns the message to show instead, or None when writes are allowed."""
    try:
        import config
        if not getattr(config, "DRIVE_WRITES", True):
            return "Saving to Google Drive is switched off in the test copy."
    except ImportError:
        pass
    return None


def push_file_to_drive(local_path: Path, drive_subfolder: str) -> str | None:
    """
    Upload a single file to scene-labeling/<drive_subfolder>/ in Drive.
    Returns None on success, error string on failure.
    """
    blocked = drive_writes_blocked()
    if blocked:
        return blocked
    try:
        service = _get_service()
        scene_id = _find_folder(service, DRIVE_SCENE_LABELING)
        if not scene_id:
            return f"'{DRIVE_SCENE_LABELING}' folder not found in Drive"
        folder_id = _ensure_drive_folder(service, drive_subfolder, scene_id)
        _upload_file(service, local_path, folder_id)
        return None
    except Exception as e:
        return str(e)


def push_directory_to_drive(local_dir: Path, drive_subfolder: str,
                             progress_callback=None) -> dict:
    """
    Upload all files in local_dir to scene-labeling/<drive_subfolder>/ in Drive.
    Returns {"pushed": N, "errors": [str]}
    """
    blocked = drive_writes_blocked()
    if blocked:
        return {"pushed": 0, "errors": [blocked]}
    try:
        service = _get_service()
        scene_id = _find_folder(service, DRIVE_SCENE_LABELING)
        if not scene_id:
            return {"pushed": 0, "errors": [f"'{DRIVE_SCENE_LABELING}' folder not found"]}
        folder_id = _ensure_drive_folder(service, drive_subfolder, scene_id)
    except Exception as e:
        return {"pushed": 0, "errors": [str(e)]}

    files = [f for f in local_dir.iterdir() if f.is_file()] if local_dir.exists() else []
    pushed = 0
    errors = []

    for idx, f in enumerate(files):
        if progress_callback:
            progress_callback(idx, len(files), f.name)
        try:
            _upload_file(service, f, folder_id)
            pushed += 1
        except Exception as e:
            errors.append(f"{f.name}: {e}")

    return {"pushed": pushed, "errors": errors}


def credentials_available() -> bool:
    """Check if the service account credentials file exists."""
    return Path(CREDENTIALS_PATH).exists()
