"""
Drive → server sync with a recorded result, so every page can show
"Drive synced N min ago". Run by the mediaplanner-sync systemd timer
(see deploy/) and by the planner's "Sync now" button.

Usage:
    python auto_sync.py
"""

import datetime
import json
import os
import sys
from pathlib import Path

import config
from config import CATALOGUE_DIR, AUDIO_DIR

# Fast 480p preview copies of the source videos. Made in Colab
# ("backfill proxies.py") into scene-labeling/proxies on Drive and pulled here
# by the sync. Set PROXY_DIR in config.py to put them elsewhere.
PROXY_DIR = getattr(config, "PROXY_DIR", CATALOGUE_DIR.parent / "proxies")

# Dot-files, so the catalogue globs ("*.json") never pick them up.
STATUS_FILE = CATALOGUE_DIR / ".last_sync.json"
LOCK_FILE = CATALOGUE_DIR / ".sync.lock"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def read_sync_status() -> dict:
    """{"finished_at", "synced", "skipped", "errors", "changed_at"} or {} if
    no sync has been recorded yet. changed_at only moves when a sync actually
    downloaded something — pages use it to know when to reload their caches."""
    try:
        data = json.loads(STATUS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _write_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_name(STATUS_FILE.name + ".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, STATUS_FILE)


def run_sync(progress_callback=None) -> dict:
    """Pull from Drive and record the result. If another sync is already running
    (timer and button at the same moment), returns {"busy": True} instead."""
    from drive_sync import sync_pull

    CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
    lock_fh = open(LOCK_FILE, "w")
    try:
        try:
            import fcntl  # Linux server; the Windows laptop never runs the timer
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass
        except BlockingIOError:
            return {"busy": True, "synced": 0, "skipped": 0, "errors": []}

        previous = read_sync_status()
        try:
            result = sync_pull(CATALOGUE_DIR, AUDIO_DIR, progress_callback, proxy_dir=PROXY_DIR)
        except Exception as e:  # network down, credentials missing, ...
            result = {"synced": 0, "skipped": 0, "errors": [str(e)]}

        # Render details (Media Library → Renders) for renders that live on Drive.
        try:
            from renders import restore_from_drive
            r = restore_from_drive()
            result["synced"] += r["restored"] + r["linked"]
            result["errors"] += r["errors"]
        except Exception as e:
            result["errors"].append(f"Renders: {e}")

        finished = _now_iso()
        _write_status({
            "finished_at": finished,
            "synced": result["synced"],
            "skipped": result["skipped"],
            "errors": result["errors"][:10],
            "changed_at": finished if result["synced"] else previous.get("changed_at"),
        })
        return result
    finally:
        lock_fh.close()


if __name__ == "__main__":
    res = run_sync()
    if res.get("busy"):
        print("Another sync is already running — skipped.")
        sys.exit(0)
    print(f"Synced {res['synced']}, skipped {res['skipped']}, errors {len(res['errors'])}")
    for err in res["errors"][:10]:
        print("  ", err)
    sys.exit(1 if res["errors"] else 0)
