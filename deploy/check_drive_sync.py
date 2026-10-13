"""
Compare what's on Google Drive with what this server has, category by category,
and point out anything that stops the Drive sync from pulling files down.
Read-only: changes nothing on Drive or on the server.

    cd /root/Compilation-planner && venv/bin/python deploy/check_drive_sync.py

Reports, for catalogue JSONs, thumbnails, timeline sprites, preview copies and
audio JSONs:
  • how many files Drive has / the server has / are missing on the server
  • folders that exist more than once with the same name (e.g. two
    "audio_catalogue" folders) — the old sync only read one of them
  • a few example missing files, and server copies that are newer than Drive's
    (the sync never overwrites those)
Audio FILES (.m4a etc.) and original videos are not synced by design — renders
stream or download them when needed — so they aren't checked here.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CATALOGUE_DIR, AUDIO_DIR  # noqa: E402
from auto_sync import PROXY_DIR  # noqa: E402
from drive_sync import (_get_service, _find_folders, _list_files, _list_subfolders,  # noqa: E402
                        DRIVE_SCENE_LABELING)


def ts(drive_file) -> float:
    try:
        return datetime.fromisoformat(drive_file["modifiedTime"].replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def report(title: str, drive: dict, local_dir: Path, local_names: set, dup_note: str = "") -> int:
    """drive: {relative name: drive file}. Returns the number missing on the server."""
    missing = sorted(n for n in drive if n not in local_names)
    newer_here = []
    for n, f in drive.items():
        p = local_dir / n
        if n in local_names and p.stat().st_size != int(f.get("size", 0)) and p.stat().st_mtime > ts(f):
            newer_here.append(n)
    status = "✓" if not missing and not newer_here else "✗"
    print(f"\n{status} {title}")
    print(f"    Drive: {len(drive)}   server: {len(local_names)}   missing on server: {len(missing)}")
    if dup_note:
        print(f"    {dup_note}")
    for n in missing[:8]:
        print(f"      missing: {n}")
    if len(missing) > 8:
        print(f"      … and {len(missing) - 8} more")
    if newer_here:
        print(f"    {len(newer_here)} server copy(ies) newer than Drive's and different (the sync keeps "
              f"the server's): {', '.join(sorted(newer_here)[:5])}{' …' if len(newer_here) > 5 else ''}")
    return len(missing)


def dup_text(folders: list, label: str, counts: list) -> str:
    if len(folders) <= 1:
        return ""
    parts = [f"{c} item(s), created {f.get('createdTime', '?')[:10]}" for f, c in zip(folders, counts)]
    return f"⚠️ {len(folders)} folders called '{label}': " + "; ".join(parts)


def main() -> int:
    s = _get_service()
    roots = _find_folders(s, DRIVE_SCENE_LABELING)
    if not roots:
        print(f"No '{DRIVE_SCENE_LABELING}' folder visible to the service account — is it shared with it?")
        return 1
    if len(roots) > 1:
        print(f"⚠️ {len(roots)} folders called '{DRIVE_SCENE_LABELING}' are visible; the sync uses one of them: "
              + ", ".join(r["id"] for r in roots))
    root = roots[0]["id"]
    total_missing = 0

    cats = _find_folders(s, "catalogue", root)
    cat_files, thumbs, sprites = {}, {}, {}
    cat_counts, thumb_folders, thumb_counts, sprite_folders, sprite_counts = [], [], [], [], []
    for c in cats:
        fl = _list_files(s, c["id"])
        cat_counts.append(len(fl))
        for f in fl:
            cat_files[f["name"]] = f
        for t in _find_folders(s, "thumbnails", c["id"]):
            subs = _list_subfolders(s, t["id"])
            thumb_folders.append(t)
            thumb_counts.append(len(subs))
            for v in subs:
                for f in _list_files(s, v["id"]):
                    thumbs[f"{v['name']}/{f['name']}"] = f
        for sp in _find_folders(s, "timeline_sprites", c["id"]):
            fl = _list_files(s, sp["id"])
            sprite_folders.append(sp)
            sprite_counts.append(len(fl))
            for f in fl:
                sprites[f["name"]] = f

    def local_set(d: Path, pattern: str = "*") -> set:
        return {p.name for p in d.glob(pattern) if p.is_file()} if d.exists() else set()

    total_missing += report("Catalogue JSONs (scene-labeling/catalogue)",
                            {n: f for n, f in cat_files.items() if n.endswith(".json")},
                            CATALOGUE_DIR, local_set(CATALOGUE_DIR, "*.json"),
                            dup_text(cats, "catalogue", cat_counts))
    tdir = CATALOGUE_DIR / "thumbnails"
    local_thumbs = ({f"{p.parent.name}/{p.name}" for p in tdir.glob("*/*") if p.is_file()}
                    if tdir.exists() else set())
    total_missing += report("Thumbnails (catalogue/thumbnails/<video>/)", thumbs, tdir, local_thumbs,
                            dup_text(thumb_folders, "thumbnails", thumb_counts)
                            + (f"   ({sum(thumb_counts)} video folders in total)" if thumb_folders else ""))
    sdir = CATALOGUE_DIR / "timeline_sprites"
    total_missing += report("Timeline sprites (catalogue/timeline_sprites)", sprites, sdir, local_set(sdir),
                            dup_text(sprite_folders, "timeline_sprites", sprite_counts))

    prox_folders = _find_folders(s, "proxies", root)
    proxies, prox_counts = {}, []
    for pf in prox_folders:
        fl = [f for f in _list_files(s, pf["id"]) if f["name"].endswith(".mp4") and ".part" not in f["name"]]
        prox_counts.append(len(fl))
        proxies.update({f["name"]: f for f in fl})
    total_missing += report("Preview copies (scene-labeling/proxies)", proxies, Path(PROXY_DIR),
                            local_set(Path(PROXY_DIR), "*.mp4"),
                            dup_text(prox_folders, "proxies", prox_counts))

    audio_folders = _find_folders(s, "audio_catalogue", root)
    audio, audio_counts = {}, []
    for af in audio_folders:
        fl = _list_files(s, af["id"])
        audio_counts.append(len(fl))
        audio.update({f["name"]: f for f in fl if f["name"].endswith(".json")})
    total_missing += report("Audio JSONs (scene-labeling/audio_catalogue)", audio, AUDIO_DIR,
                            local_set(AUDIO_DIR, "*.json"),
                            dup_text(audio_folders, "audio_catalogue", audio_counts))

    print("\nAudio files and original videos aren't synced by design (renders stream or download them).")
    print("Nothing missing." if not total_missing else
          f"{total_missing} file(s) on Drive are missing on the server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
