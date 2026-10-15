#!/usr/bin/env bash
# One-time setup: play videos, music and pictures straight from disk through
# nginx instead of loading them into the app's memory (see media_links.py), and
# make the app hand freed memory back more readily.
#
#   sudo bash /root/Compilation-planner/deploy/setup_media.sh
#
# Sets up:
#   • a secret for signed, expiring media addresses   /etc/mediaplanner/media.secret
#   • read access for nginx to the data folders       /srv/mediaplanner/data[-dev]
#     (a bind mount — /root itself stays private)
#   • nginx: /media/… on both sites, signed links only (snippets/mediaplanner-media.conf)
#   • MALLOC_ARENA_MAX=2 for both app services (less memory kept back by Python)
#   • restarts the live app (and the test copy if it's running) to pick it all up
# Safe to re-run. Undo: see the end of this file.
set -euo pipefail

LIVE=mediaplanner
TEST=mediaplanner-dev
LIVE_DOMAIN=video-compilation-planner.co.uk
TEST_DOMAIN=dev.$LIVE_DOMAIN
SRV=/srv/mediaplanner

step() { echo; echo "=== $* ==="; }
[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }
nginx -V 2>&1 | grep -q with-http_secure_link_module || {
  echo "This nginx has no secure_link module — install nginx-full (or nginx-extras) and re-run."; exit 1; }

step "1/5  Secret for signed media links"
mkdir -p /etc/mediaplanner
if [[ ! -s /etc/mediaplanner/media.secret ]]; then
  head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > /etc/mediaplanner/media.secret
  echo "Created /etc/mediaplanner/media.secret"
else
  echo "Already there."
fi
chmod 600 /etc/mediaplanner/media.secret
SECRET=$(cat /etc/mediaplanner/media.secret)

step "2/5  Let nginx read the data folders (bind mounts)"
mkdir -p "$SRV"
for name in data data-dev; do
  src=/root/$name
  [[ -d $src ]] || continue
  mkdir -p "$SRV/$name"
  line="$src $SRV/$name none rbind 0 0"
  grep -qF "$line" /etc/fstab || echo "$line" >> /etc/fstab
  mountpoint -q "$SRV/$name" || mount "$SRV/$name"
  # Folders must be enterable and files readable by nginx (www-data). Files the
  # app writes are already like this; this fixes any older ones.
  find "$src" -xdev -type d ! -perm -o+x -exec chmod o+rx {} + 2>/dev/null || true
  find "$src" -xdev -type f ! -perm -o+r \( -iname '*.mp4' -o -iname '*.m4a' -o -iname '*.mp3' \
       -o -iname '*.mov' -o -iname '*.wav' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
       -exec chmod o+r {} + 2>/dev/null || true
  echo "$src → $SRV/$name"
done

step "3/5  nginx: /media/ on both sites"
cat > /etc/nginx/snippets/mediaplanner-media.conf <<EOF
# Media Planner: videos, music and pictures for the app's players, read straight
# from disk. Only signed, unexpired links work (media_links.py makes them).
location ^~ /media/ {
    secure_link \$arg_k,\$arg_e;
    secure_link_md5 "\$secure_link_expires\$uri $SECRET";
    if (\$secure_link = "")  { return 403; }
    if (\$secure_link = "0") { return 410; }
    alias $SRV/;
    types { video/mp4 mp4 m4v; video/quicktime mov; audio/mp4 m4a; audio/mpeg mp3; audio/wav wav;
            image/jpeg jpg jpeg; image/png png; image/webp webp; }
    default_type application/octet-stream;
    add_header Cache-Control "private, max-age=3600";
    access_log off;
}
EOF
chmod 640 /etc/nginx/snippets/mediaplanner-media.conf
NGINX_BACKUP=$(mktemp -d)
cp -a /etc/nginx/sites-available "$NGINX_BACKUP/"
python3 - "$LIVE_DOMAIN" "$TEST_DOMAIN" <<'PY'
import re, sys
from pathlib import Path
domains = set(sys.argv[1:])
for f in Path("/etc/nginx/sites-available").iterdir():
    if not f.is_file():
        continue
    lines = f.read_text().splitlines(keepends=True)
    out, changed = [], False
    for i, line in enumerate(lines):
        out.append(line)
        m = re.match(r"\s*server_name\s+([^;]+);", line)
        if not m or not (set(m.group(1).split()) & domains):
            continue
        depth, block, j = 0, [], i
        while j < len(lines):
            block.append(lines[j]); depth += lines[j].count("{") - lines[j].count("}")
            if depth < 0: break
            j += 1
        block_text = "".join(block)
        # only the HTTPS server blocks that proxy to the app
        if "proxy_pass http://127.0.0.1:85" not in block_text or "mediaplanner-media.conf" in block_text:
            continue
        indent = re.match(r"(\s*)", line).group(1)
        out.append(f"{indent}include snippets/mediaplanner-media.conf;\n")
        changed = True
    if changed:
        f.write_text("".join(out))
        print(f"Added /media/ to {f.name}")
PY
if nginx -t 2>/dev/null; then
  systemctl reload nginx
  echo "nginx reloaded."
else
  echo "nginx config test failed — restoring the previous config. Nothing else changed in nginx."
  cp -a "$NGINX_BACKUP/sites-available/." /etc/nginx/sites-available/
  rm -f /etc/nginx/snippets/mediaplanner-media.conf
  nginx -t && systemctl reload nginx
  rm -rf "$NGINX_BACKUP"
  exit 1
fi
rm -rf "$NGINX_BACKUP"

step "4/5  Less memory kept back by Python (MALLOC_ARENA_MAX=2)"
for svc in $LIVE $TEST; do
  systemctl cat "$svc" >/dev/null 2>&1 || continue
  mkdir -p /etc/systemd/system/$svc.service.d
  cat > /etc/systemd/system/$svc.service.d/memory.conf <<EOF
[Service]
# Each page runs in its own thread; by default glibc gives threads their own
# memory pools and rarely hands them back. Two pools keep the app far smaller.
Environment=MALLOC_ARENA_MAX=2
EOF
done
systemctl daemon-reload

step "5/5  Restart and check"
systemctl restart $LIVE
systemctl is-active --quiet $TEST && systemctl restart $TEST || true
sleep 8
TESTFILE=$(find /root/data/proxies /root/data/catalogue/thumbnails -type f \( -name '*.mp4' -o -name '*.jpg' \) 2>/dev/null | head -1 || true)
if [[ -n "$TESTFILE" ]]; then
  REL=${TESTFILE#/root/data/}
  URI="/media/data/$REL"
  EXP=$(( $(date +%s) + 3600 ))
  KEY=$(printf '%s' "$EXP$URI $SECRET" | openssl md5 -binary | openssl base64 | tr '+/' '-_' | tr -d '=')
  ENC=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$URI")
  OK=$(curl -s -o /dev/null -w '%{http_code}' --resolve "$LIVE_DOMAIN:443:127.0.0.1" -r 0-99 \
       "https://$LIVE_DOMAIN$ENC?e=$EXP&k=$KEY" || true)
  BAD=$(curl -s -o /dev/null -w '%{http_code}' --resolve "$LIVE_DOMAIN:443:127.0.0.1" \
       "https://$LIVE_DOMAIN$ENC" || true)
  echo "Signed link: $OK (want 206 or 200)   ·   unsigned: $BAD (want 403)"
  [[ $OK == 20? ]] && [[ $BAD == 403 ]] && echo "✅ Media links work." \
    || echo "⚠️  Not as expected — players fall back to the old way until it's fixed."
else
  echo "No preview copy or thumbnail found to test with — check a player in the app."
fi
echo
echo "Done. Players now stream from disk; the Server page shows 'media links are set up'."

# --- Undo -------------------------------------------------------------------
#   rm /etc/mediaplanner/media.secret            (the app falls back at once)
#   sed -i '/mediaplanner-media.conf/d' /etc/nginx/sites-available/* && nginx -t && systemctl reload nginx
#   rm /etc/nginx/snippets/mediaplanner-media.conf
#   umount /srv/mediaplanner/data /srv/mediaplanner/data-dev; sed -i '\#/srv/mediaplanner/#d' /etc/fstab
#   rm /etc/systemd/system/mediaplanner*.service.d/memory.conf; systemctl daemon-reload
