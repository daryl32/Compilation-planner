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

# Files downloaded at once by sync_pull.
SYNC_WORKERS = 8


def q_escape(value: str) -> str:
    """Escape a value for a Drive search query (q=…name='…'): backslashes and
    single quotes must be backslash-escaped, e.g. "Movin' To The Sun"."""
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _get_service():
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def _find_folder(service, name: str, parent_id: str = None) -> str | None:
    """Return the Drive folder ID for `name`, optionally within `parent_id`."""
    q = f"name='{q_escape(name)}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    resp = service.files().list(q=q, fields="files(id, name)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _find_folders(service, name: str, parent_id: str = None) -> list[dict]:
    """EVERY folder called `name` (optionally inside parent_id), oldest first, as
    [{"id", "name", "createdTime"}]. There can be more than one: uploads made
    with the user's Drive login (drive.file permission) can't see folders they
    didn't create, so they used to make a second "audio_catalogue" or
    "thumbnails" beside the real one. Read them all."""
    q = f"name='{q_escape(name)}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    out, page_token = [], None
    while True:
        resp = service.files().list(q=q, fields="nextPageToken, files(id, name, createdTime)",
                                    pageSize=1000, pageToken=page_token).execute()
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return sorted(out, key=lambda f: f.get("createdTime", ""))


def _list_subfolders(service, folder_id: str) -> list[dict]:
    """Every folder directly inside folder_id (all pages)."""
    out, page_token = [], None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="nextPageToken, files(id, name, createdTime)", pageSize=1000, pageToken=page_token,
        ).execute()
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def _list_files(service, folder_id: str) -> list[dict]:
    """List all non-folder files directly in a Drive folder."""
    results = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false",
            fields="nextPageToken, files(id, name, modifiedTime, createdTime, size)",
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
        q=f"name='{q_escape(local_path.name)}' and '{folder_id}' in parents and trashed=false",
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


def _broken_picture(path: Path) -> bool:
    """A local .jpg/.png that won't open (e.g. half-written by an interrupted
    download before downloads went through a .part file) — fetch it again even
    though its timestamp looks newer than Drive's."""
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        return False
    try:
        if path.stat().st_size == 0:
            return True
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return False
    except Exception:
        return True


# When each video was added: {catalogue stem: Drive createdTime of its catalogue
# JSON, ISO}. A file's local modified-time changes on every Reviewer edit or
# re-download, but Drive's createdTime never does (edits update the file in
# place), so this is what "Date added" / "Newly added" go by. Dot-file, so it's
# never mistaken for a catalogue; read by catalogue_store.added_dates().
ADDED_DATES_NAME = ".added_dates.json"


def _record_added_dates(catalogue_dir: Path, created: dict) -> None:
    """Merge Drive createdTimes into catalogue_dir/.added_dates.json, keeping the
    earliest date ever seen for each video (a catalogue deleted and re-uploaded
    doesn't count as new). Never fails the sync."""
    if not created:
        return
    path = Path(catalogue_dir) / ADDED_DATES_NAME
    try:
        known = json.loads(path.read_text())
        if not isinstance(known, dict):
            known = {}
    except (FileNotFoundError, ValueError, OSError):
        known = {}
    merged = dict(known)
    for stem, iso in created.items():
        if stem not in merged or iso < merged[stem]:
            merged[stem] = iso
    if merged == known:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=1, sort_keys=True))
        os.replace(tmp, path)
    except OSError:
        pass


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

    def ids(name, parent_id):
        return [f["id"] for f in _find_folders(service, name, parent_id)]

    created = {}   # catalogue stem -> earliest Drive createdTime (when the video was added)
    for cat_folder_id in ids("catalogue", scene_id):
        # catalogue JSONs
        for f in _list_files(service, cat_folder_id):
            tasks.append((f, catalogue_dir / f["name"]))
            if f["name"].endswith(".json") and f.get("createdTime"):
                stem = f["name"][:-5]
                if stem not in created or f["createdTime"] < created[stem]:
                    created[stem] = f["createdTime"]

        # thumbnails (subfolder of catalogue), in per-video subfolders
        for thumb_id in ids("thumbnails", cat_folder_id):
            for vid_folder in _list_subfolders(service, thumb_id):
                for f in _list_files(service, vid_folder["id"]):
                    tasks.append((f, catalogue_dir / "thumbnails" / vid_folder["name"] / f["name"]))

        # timeline_sprites (subfolder of catalogue)
        for sprite_id in ids("timeline_sprites", cat_folder_id):
            for f in _list_files(service, sprite_id):
                tasks.append((f, catalogue_dir / "timeline_sprites" / f["name"]))

    _record_added_dates(catalogue_dir, created)

    # preview copies (proxies)
    if proxy_dir is not None:
        for proxies_id in ids("proxies", scene_id):
            for f in _list_files(service, proxies_id):
                if f["name"].endswith(".mp4") and ".part" not in f["name"]:
                    tasks.append((f, proxy_dir / f["name"]))

    # audio_catalogue JSONs
    for audio_folder_id in ids("audio_catalogue", scene_id):
        for f in _list_files(service, audio_folder_id):
            tasks.append((f, audio_dir / f["name"]))

    # The same file can now appear more than once (two copies with one name, or
    # the same name in two same-named folders): keep the newest copy of each.
    newest = {}
    for drive_file, local_path in tasks:
        key = str(local_path)
        if key not in newest or drive_file.get("modifiedTime", "") > newest[key][0].get("modifiedTime", ""):
            newest[key] = (drive_file, local_path)
    tasks = list(newest.values())

    # Small, important files first, so a big catch-up of thumbnails or preview
    # copies never holds up new catalogue or audio JSONs.
    def priority(task):
        p = task[1]
        if p.parent == catalogue_dir:
            return 0
        if p.parent == audio_dir:
            return 1
        if p.parent.name == "timeline_sprites":
            return 2
        if p.parent.parent.name == "thumbnails":
            return 3
        return 4
    tasks.sort(key=priority)

    synced = skipped = 0
    errors = []
    total = len(tasks)

    # Decide what needs downloading (cheap, local checks only).
    todo = []
    for drive_file, local_path in tasks:
        # Skip if local file exists and is same size (quick check)
        if local_path.exists() and local_path.stat().st_size == int(drive_file.get("size", 0)):
            skipped += 1
            continue
        # Never overwrite a local file that is newer than the Drive copy — e.g. a
        # Reviewer edit (excluded / intro / outro / corrected tags) not yet pushed to Drive.
        if local_path.exists() and _local_is_newer(local_path, drive_file) and not _broken_picture(local_path):
            skipped += 1
            continue
        todo.append((drive_file, local_path))

    # Download several at once: thumbnails are tiny, so one-at-a-time spends
    # almost all its time waiting on round trips to Google. Each worker thread
    # gets its own Drive client (they aren't thread-safe).
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    local = threading.local()

    def fetch(drive_file, local_path):
        if not hasattr(local, "service"):
            local.service = _get_service()
        tmp = local_path.with_name(local_path.name + ".part")
        _download_file(local.service, drive_file["id"], tmp)
        os.replace(tmp, local_path)   # never leave a half-written file in place

    done = skipped
    if progress_callback:
        progress_callback(done, total, "")
    with ThreadPoolExecutor(max_workers=SYNC_WORKERS) as pool:
        futures = {pool.submit(fetch, f, p): (f, p) for f, p in todo}
        for fut in as_completed(futures):
            drive_file, local_path = futures[fut]
            try:
                fut.result()
                synced += 1
            except Exception as e:
                errors.append(f"{drive_file['name']}: {e}")
                try:
                    local_path.with_name(local_path.name + ".part").unlink(missing_ok=True)
                except OSError:
                    pass
            done += 1
            if progress_callback:
                progress_callback(done, total, local_path.name)

    return {"synced": synced, "skipped": skipped, "errors": errors}


def find_drive_file(drive_rel_path: str, service=None) -> tuple:
    """({"id", "name", "size"}, None) for a file below My Drive, e.g.
    "Videos-PH/clip.mp4" — or (None, error string). The first folder must be
    shared with the service account."""
    parts = [p for p in drive_rel_path.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        return None, f"Unexpected Drive path: {drive_rel_path}"
    service = service or _get_service()

    parent_id = None
    for folder_name in parts[:-1]:
        parent_id = _find_folder(service, folder_name, parent_id)
        if not parent_id:
            return None, (f"Folder '{folder_name}' not found in Drive — is it shared with "
                          f"the service account?")

    file_name = parts[-1]
    resp = service.files().list(
        q=f"name='{q_escape(file_name)}' and '{parent_id}' in parents and trashed=false",
        fields="files(id, name, size)",
    ).execute()
    files = resp.get("files", [])
    if not files:
        return None, f"'{file_name}' not found in Drive folder '{parts[-2]}'"
    return files[0], None


_token_cache = {"token": None, "expires": 0.0}


def list_drive_folder(drive_rel_path: str) -> tuple:
    """(files, None) for a folder below My Drive, e.g. "scene-labeling/previews"
    — each file {"id", "name", "modifiedTime", "size"} — or ([], error string)."""
    try:
        service = _get_service()
        parent_id = None
        for name in [p for p in drive_rel_path.replace("\\", "/").split("/") if p]:
            parent_id = _find_folder(service, name, parent_id)
            if not parent_id:
                return [], f"Folder '{name}' not found in Drive"
        return _list_files(service, parent_id), None
    except Exception as e:
        return [], str(e)


def download_drive_file(file_id: str, dest: Path) -> None:
    """Download one Drive file by id (raises on failure)."""
    _download_file(_get_service(), file_id, Path(dest))


def access_token() -> str:
    """A short-lived OAuth access token for the service account (cached, and
    refreshed a few minutes before it expires). Used to let ffmpeg read Drive
    files directly over HTTPS — see stream_url()."""
    import time
    if _token_cache["token"] and time.time() < _token_cache["expires"] - 300:
        return _token_cache["token"]
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
    creds.refresh(Request())
    expiry = creds.expiry.timestamp() if creds.expiry else time.time() + 3000
    if creds.expiry and creds.expiry.tzinfo is None:   # google-auth gives naive UTC
        from datetime import timezone
        expiry = creds.expiry.replace(tzinfo=timezone.utc).timestamp()
    _token_cache.update(token=creds.token, expires=expiry)
    return creds.token


def stream_url(file_id: str) -> str:
    """Direct download URL for a Drive file. Supports HTTP range requests, so a
    reader such as ffmpeg can seek and fetch only the part it needs. Needs an
    "Authorization: Bearer <access_token()>" header."""
    return f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true"


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
        service = _get_service()
        drive_file, err = find_drive_file(drive_rel_path, service)
        if err:
            return err
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
    """Check if the service account credentials file exists (False when it
    can't be checked, e.g. /root isn't readable by this user)."""
    try:
        return Path(CREDENTIALS_PATH).exists()
    except OSError:
        return False
