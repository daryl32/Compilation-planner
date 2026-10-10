#!/usr/bin/env bash
# One-time setup for the "🔀 Live / test copy" switch (sidebar of every page).
# Only one copy runs at a time, so they don't share the server's 4 GB.
#
#   sudo bash /root/Compilation-planner/deploy/setup_switch.sh
#
# Sets up:
#   • live comes back on its own whenever the test copy stops (switch, crash, idle)
#   • the test copy switches itself off after IDLE_MIN minutes unused
#   • after a reboot only the live app starts
#   • pushes to main/dev update a copy that's running, but don't start one that's off
#   • a copy that's off shows a "switched off" page with links, not "502 Bad Gateway"
#   • turns the switch panel on (/etc/mediaplanner-switch.enabled)
# Safe to re-run. To undo: see the end of this file.
set -euo pipefail

LIVE=mediaplanner
TEST=mediaplanner-dev
LIVE_DOMAIN=video-compilation-planner.co.uk
TEST_DOMAIN=dev.$LIVE_DOMAIN
IDLE_MIN=30
REPO=/root/Compilation-planner

step() { echo; echo "=== $* ==="; }
[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }
systemctl cat "$LIVE" >/dev/null 2>&1 || { echo "$LIVE.service not found."; exit 1; }
systemctl cat "$TEST" >/dev/null 2>&1 || { echo "$TEST.service not found — run setup_dev.sh first."; exit 1; }

step "1/6  Live starts whenever the test copy stops"
mkdir -p /etc/systemd/system/$TEST.service.d
cat > /etc/systemd/system/$TEST.service.d/switch.conf <<EOF
[Service]
# Whenever the test copy stops (switch, idle, crash, deploy), make sure live is up.
ExecStopPost=/bin/systemctl start --no-block $LIVE
EOF
systemctl daemon-reload
systemctl enable "$LIVE" >/dev/null
systemctl disable "$TEST" >/dev/null 2>&1 || true
echo "Live starts at boot; the test copy doesn't."

step "2/6  Test copy switches off after $IDLE_MIN min unused"
mkdir -p /var/lib/mediaplanner
cat > /usr/local/bin/mediaplanner-dev-idle <<EOF
#!/usr/bin/env bash
# Stop the test copy when nobody has used it for $IDLE_MIN minutes.
systemctl is-active --quiet $TEST || exit 0
now=\$(date +%s)
beat=/var/lib/mediaplanner/$TEST.heartbeat
last=0
[[ -f \$beat ]] && last=\$(stat -c %Y "\$beat")
since=\$(date -d "\$(systemctl show -p ActiveEnterTimestamp --value $TEST)" +%s 2>/dev/null || echo "\$now")
(( since > last )) && last=\$since
if (( now - last > $IDLE_MIN * 60 )); then
  echo "Test copy unused for \$(( (now - last) / 60 )) min — stopping it."
  systemctl stop $TEST
fi
EOF
chmod +x /usr/local/bin/mediaplanner-dev-idle
cat > /etc/systemd/system/mediaplanner-dev-idle.service <<EOF
[Unit]
Description=Media Planner: switch the test copy off when unused

[Service]
Type=oneshot
ExecStart=/usr/local/bin/mediaplanner-dev-idle
EOF
cat > /etc/systemd/system/mediaplanner-dev-idle.timer <<EOF
[Unit]
Description=Media Planner: check every 5 min whether the test copy is unused

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now mediaplanner-dev-idle.timer >/dev/null
echo "Idle check runs every 5 minutes."

step "3/6  Deploys don't start a copy that's off"
BACKUP="/root/webhook.py.bak-$(date +%Y%m%d-%H%M%S)"
cp /root/webhook.py "$BACKUP"
cp "$REPO/deploy/webhook.py" /root/webhook.py
systemctl restart webhook
sleep 2
if systemctl is-active --quiet webhook; then
  echo "Webhook updated (old version saved as $BACKUP)."
else
  echo "WARNING: new webhook didn't start — restoring the old one."
  cp "$BACKUP" /root/webhook.py
  systemctl restart webhook
fi

step "4/6  'Switched off' page instead of 502"
mkdir -p /var/www/mediaplanner-off
cat > /var/www/mediaplanner-off/__switched_off.html <<EOF
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>Media Planner — switched off</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:560px;margin:12vh auto;padding:0 16px;color:#222;background:#fafafa}
 a.b{display:block;margin:10px 0;padding:12px 16px;border-radius:8px;background:#2a78d6;color:#fff;text-decoration:none;text-align:center}
 a.s{background:#555}
 @media (prefers-color-scheme:dark){body{background:#1a1a19;color:#eee}}
</style></head><body>
<h2>This copy of the Media Planner is switched off</h2>
<p>Only one copy runs at a time to save the server's memory — or this one is restarting.
This page checks again every 15 seconds.</p>
<a class="b" href="https://$LIVE_DOMAIN">Open the live app</a>
<a class="b s" href="https://$TEST_DOMAIN">Open the test copy</a>
<p><small>The live app comes back on its own whenever the test copy stops. To start the test copy,
use “🔀 Live / test copy” in the live app's sidebar.</small></p>
</body></html>
EOF
cat > /etc/nginx/snippets/mediaplanner-off.conf <<'EOF'
# Media Planner: show a "switched off" page when the app behind this site isn't running.
error_page 502 503 504 /__switched_off.html;
location = /__switched_off.html { root /var/www/mediaplanner-off; internal; }
EOF
NGINX_BACKUP=$(mktemp -d)
cp -a /etc/nginx/sites-available "$NGINX_BACKUP/"
python3 - "$LIVE_DOMAIN" "$TEST_DOMAIN" <<'PY'
import re, sys
from pathlib import Path
domains = set(sys.argv[1:])
for f in Path("/etc/nginx/sites-available").iterdir():
    if not f.is_file():
        continue
    text = f.read_text()
    out, changed = [], False
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        out.append(line)
        m = re.match(r"\s*server_name\s+([^;]+);", line)
        if not m or not (set(m.group(1).split()) & domains):
            continue
        # only server blocks that proxy to the app (not certbot's port-80 redirect)
        depth, block, j = 0, [], i
        while j < len(lines):
            block.append(lines[j]); depth += lines[j].count("{") - lines[j].count("}")
            if depth < 0: break
            j += 1
        block_text = "".join(block)
        if "proxy_pass http://127.0.0.1:85" not in block_text or "mediaplanner-off.conf" in block_text:
            continue
        indent = re.match(r"(\s*)", line).group(1)
        out.append(f"{indent}include snippets/mediaplanner-off.conf;\n")
        changed = True
    if changed:
        f.write_text("".join(out))
        print(f"Added the switched-off page to {f.name}")
PY
if nginx -t 2>/dev/null; then
  systemctl reload nginx
  echo "nginx reloaded."
else
  echo "WARNING: nginx config test failed — restoring the previous nginx config."
  cp -a "$NGINX_BACKUP/sites-available/." /etc/nginx/sites-available/
  nginx -t && systemctl reload nginx
fi
rm -rf "$NGINX_BACKUP"

step "5/6  Turn the switch panel on"
touch /etc/mediaplanner-switch.enabled
echo "The 🔀 Live / test copy panel now shows in the sidebar (after the next page load)."

step "6/6  Status"
for s in $LIVE $TEST; do printf "  %-18s %s\n" "$s" "$(systemctl is-active $s || true)"; done
echo
echo "The test copy is left as it is. To switch it off now:  sudo systemctl stop $TEST"
echo
echo "To undo all of this:"
echo "  rm /etc/mediaplanner-switch.enabled /etc/systemd/system/$TEST.service.d/switch.conf"
echo "  systemctl disable --now mediaplanner-dev-idle.timer; systemctl enable $TEST; systemctl daemon-reload"
