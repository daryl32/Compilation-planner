#!/usr/bin/env bash
# Copy production's data into the TEST copy's own data folder (/root/data-dev),
# replacing whatever the test copy had. Production data is only read, never changed.
#
#   sudo bash /root/Compilation-planner/deploy/refresh_dev_data.sh
#
# Copied:   video catalogues, library ranges, labels, audio catalogues, projects
# Shared:   thumbnails and timeline sprites (linked, read-only use), and — via
#           config — source videos, audio files and preview copies
# Not copied: compilation plans and rendered previews (test makes its own)
set -euo pipefail

PROD_DIR=/root/Compilation-planner
DEV_DATA=/root/data-dev

command -v rsync >/dev/null || apt-get install -y rsync

# Production's real paths, straight from its config.py
mapfile -t P < <(cd "$PROD_DIR" && venv/bin/python -c "
import config
print(config.CATALOGUE_DIR); print(config.AUDIO_DIR); print(config.PROJECTS_DIR)")
PROD_CAT="${P[0]:-}"; PROD_AUDIO="${P[1]:-}"; PROD_PROJECTS="${P[2]:-}"
for p in "$PROD_CAT" "$PROD_AUDIO"; do
  if [[ -z "$p" || "$p" == "/" || ! -d "$p" ]]; then
    echo "Couldn't read production's data folders from $PROD_DIR/config.py (got '$p') — stopping."
    exit 1
  fi
done

mkdir -p "$DEV_DATA"/{catalogue,audio_catalogue,projects,compilation_plans,previews}

echo "Catalogues:  $PROD_CAT  →  $DEV_DATA/catalogue"
rsync -a --delete \
  --exclude 'thumbnails' --exclude 'timeline_sprites' \
  --exclude '.last_sync.json' --exclude '.pending_push.txt' --exclude '.sync.lock' \
  "$PROD_CAT"/ "$DEV_DATA/catalogue/"

# Images are only read by the app, so link them instead of copying gigabytes.
for d in thumbnails timeline_sprites; do
  if [[ -d "$PROD_CAT/$d" ]]; then
    [[ -L "$DEV_DATA/catalogue/$d" || ! -e "$DEV_DATA/catalogue/$d" ]] || rm -rf "$DEV_DATA/catalogue/$d"
    ln -sfn "$PROD_CAT/$d" "$DEV_DATA/catalogue/$d"
  fi
done

echo "Audio:       $PROD_AUDIO  →  $DEV_DATA/audio_catalogue"
rsync -a --delete "$PROD_AUDIO"/ "$DEV_DATA/audio_catalogue/"

echo "Projects:    $PROD_PROJECTS  →  $DEV_DATA/projects"
if [[ -n "$PROD_PROJECTS" && "$PROD_PROJECTS" != "/" && -d "$PROD_PROJECTS" ]]; then
  rsync -a --delete "$PROD_PROJECTS"/ "$DEV_DATA/projects/"
fi

# Reload the test app so it doesn't keep showing cached old data.
systemctl try-restart mediaplanner-dev 2>/dev/null || true
echo "Done — test copy now has a fresh copy of production data."
