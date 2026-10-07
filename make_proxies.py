"""
Make fast 480p preview copies (proxies) of the library's source videos, used
by the range picker in the Media Library and Compilation Planner.

New videos normally get their proxy in Colab ("backfill proxies.py"), which
the Drive sync then pulls down — this server-side script is the fallback for
videos whose original is already on the server. Skips videos whose proxy is
already up to date. Run nightly by the mediaplanner-proxies systemd timer
(see deploy/), or by hand:

    python make_proxies.py                   # every video whose source is on this machine
    python make_proxies.py --download        # also download missing sources from Drive first
    python make_proxies.py VIDEO_ID ...      # just these videos
    python make_proxies.py --force           # re-encode even if a proxy exists
"""

import argparse
import json
import sys
import time
from pathlib import Path

from config import CATALOGUE_DIR
from library_common import PROXY_DIR, proxy_path, make_proxy
from render_preview import colab_to_local


def iter_catalogues():
    for f in sorted(CATALOGUE_DIR.glob("*.json")):
        try:
            cat = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(cat, dict) and "video_id" in cat and "scenes" in cat:
            yield cat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_ids", nargs="*", help="only these videos (default: all)")
    ap.add_argument("--download", action="store_true", help="download missing sources from Google Drive")
    ap.add_argument("--force", action="store_true", help="re-encode even if the proxy is up to date")
    args = ap.parse_args()

    wanted = set(args.video_ids)
    made = skipped = missing = failed = 0
    print(f"Proxies go to {PROXY_DIR}")

    for cat in iter_catalogues():
        vid = cat["video_id"]
        if wanted and vid not in wanted:
            continue
        raw_src = cat.get("source_path", "")
        src = Path(colab_to_local(raw_src))
        dest = proxy_path(vid)

        if not src.exists():
            if args.download and "MyDrive/" in raw_src:
                from drive_sync import download_source_video
                print(f"⬇  {vid}: downloading source…")
                err = download_source_video(raw_src.split("MyDrive/", 1)[1], src)
                if err:
                    print(f"✗  {vid}: download failed: {err}")
                    failed += 1
                    continue
            else:
                missing += 1
                continue

        if dest.exists() and not args.force and dest.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue

        t0 = time.time()
        err = make_proxy(src, dest)
        if err:
            print(f"✗  {vid}: {err}")
            failed += 1
        else:
            mb = dest.stat().st_size / 1e6
            print(f"✓  {vid}: {mb:.1f} MB in {time.time() - t0:.0f}s")
            made += 1

    print(f"\nMade {made}, up to date {skipped}, source not on this machine {missing}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
