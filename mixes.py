"""🎚️ Saved mixes — recipes in CATALOGUE_DIR/mixes.json (synced to Drive like
the story maps), and saving a rendered mix as a new track in the library.

    {mix_id: {…recipe (core/mixing.py), "created", "updated",
              "output": {"track_id", "drive_rel", "duration", "rendered_at",
                         "uploaded": bool}}}

Saving a mix: render → the Audio-Library folder on this machine (where the
track's source_path points) → upload to the Audio-Library folder in Google
Drive (needs Drive connected; the test copy never uploads) → queued for the
normal server audio analysis, with the beat grid and bar lines carried over
from the source tracks, so it's ready for the planner like any other track.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
import uuid
from pathlib import Path

import config
from config import AUDIO_DIR, CATALOGUE_DIR

AUDIO_LIBRARY_FOLDER = getattr(config, "AUDIO_LIBRARY_DRIVE_FOLDER", "Audio-Library")


def _file() -> Path:
    return CATALOGUE_DIR / "mixes.json"


def all_mixes() -> dict:
    try:
        data = json.loads(_file().read_text())
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def get(mix_id: str) -> dict | None:
    return all_mixes().get(mix_id)


def _write(mixes: dict) -> None:
    from library_common import mark_pending
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(mixes, indent=1))
    os.replace(tmp, path)
    mark_pending(path.stem)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return "mix_" + uuid.uuid4().hex[:10]


def save(recipe: dict) -> dict:
    mixes = all_mixes()
    rid = recipe.get("id") or new_id()
    old = mixes.get(rid) or {}
    rec = {**old, **recipe, "id": rid, "created": old.get("created") or _now(), "updated": _now()}
    mixes[rid] = rec
    _write(mixes)
    return rec


def delete(mix_id: str) -> None:
    """Forget the recipe (a track already made from it stays in the library)."""
    mixes = all_mixes()
    if mixes.pop(mix_id, None) is not None:
        _write(mixes)


def made_from(track_id: str) -> dict | None:
    """The recipe a library track was made from, if any."""
    for m in all_mixes().values():
        if (m.get("output") or {}).get("track_id") == track_id:
            return m
    return None


def safe_name(name: str) -> str:
    """A file-name-safe track name (keeps spaces, letters, digits, - _ ( ) &)."""
    s = re.sub(r"[^\w\s\-\(\)&'.,+]", "", name or "", flags=re.UNICODE).strip().strip(".")
    return re.sub(r"\s+", " ", s)[:90]


def name_taken(name: str, mix_id: str = None) -> bool:
    """True if a library track already uses this name (other than this mix's own output)."""
    tid = safe_name(name)
    if not tid:
        return True
    mine = ((get(mix_id) or {}).get("output") or {}).get("track_id") if mix_id else None
    return tid != mine and (AUDIO_DIR / f"{tid}.json").exists()


def output_paths(name: str) -> dict:
    tid = safe_name(name)
    drive_rel = f"{AUDIO_LIBRARY_FOLDER}/{tid}.m4a"
    source_path = f"/content/drive/MyDrive/{drive_rel}"
    from render_preview import colab_to_local
    return {"track_id": tid, "drive_rel": drive_rel, "source_path": source_path,
            "local": Path(colab_to_local(source_path))}


def publish(recipe: dict, tracks: dict, token: dict = None, progress=None) -> tuple:
    """Render the recipe as a library track, upload it, queue its analysis.
    Returns (saved recipe, list of notes for the user). Raises on a render error."""
    import mixer
    say = progress or (lambda m: None)
    paths = output_paths(recipe["name"])
    info = mixer.render(recipe, tracks, paths["local"], progress=say)
    notes = list(info["warnings"])

    uploaded = False
    if token:
        say("uploading to Google Drive")
        from drive_oauth import upload_file_with_oauth
        folder = paths["drive_rel"].rsplit("/", 1)[0]
        _, err = upload_file_with_oauth(token, paths["local"], folder)
        if err:
            notes.append(f"Not uploaded to Google Drive: {err}")
        else:
            uploaded = True
    else:
        notes.append("Google Drive isn't connected, so the new track is only on the server for now "
                     "(it'll be cleared with other downloads after a few days unused) — connect Drive "
                     "and use ☁️ Upload on the mix to keep it.")

    lineage = {"mix_id": recipe.get("id"), "kind": recipe.get("kind"),
               "made_from": sorted({p["track"] for p in recipe.get("pieces") or []})}
    try:
        import audio_analysis as AA
        AA.cancel_queued([paths["track_id"]])
        AA.enqueue([{"track_id": paths["track_id"], "kind": "new", "drive_rel": paths["drive_rel"],
                     "vocals": False, "mix": lineage,
                     "beat_hint": {"beat_times": info["beats"], "downbeats": info["bars"]}}])
        AA.resume()
    except Exception as e:
        notes.append(f"Couldn't queue the analysis: {e}")

    rec = save({**recipe, "output": {"track_id": paths["track_id"], "drive_rel": paths["drive_rel"],
                                     "duration": info["duration"], "bpm": info["bpm"],
                                     "rendered_at": _now(), "uploaded": uploaded}})
    return rec, notes


def upload_output(mix_id: str, token: dict) -> str | None:
    """☁️ Upload a mix's saved track to Drive (when it was made without Drive connected)."""
    rec = get(mix_id)
    out = (rec or {}).get("output") or {}
    if not out:
        return "This mix hasn't been saved as a track yet."
    paths = output_paths(out["track_id"])
    if not paths["local"].exists():
        return "The track file isn't on the server any more — save the mix again."
    from drive_oauth import upload_file_with_oauth
    _, err = upload_file_with_oauth(token, paths["local"], out["drive_rel"].rsplit("/", 1)[0])
    if err:
        return err
    save({**rec, "output": {**out, "uploaded": True}})
    return None
