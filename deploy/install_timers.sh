#!/usr/bin/env bash
# Installs two systemd timers on the server (run once, as root):
#   mediaplanner-sync     — Drive → server sync every 12 hours (auto_sync.py)
#   mediaplanner-proxies  — nightly fast-preview copies at 03:30 (make_proxies.py)
#
#   sudo bash /root/Compilation-planner/deploy/install_timers.sh
#
# Uses the same Python as the mediaplanner service, so packages match.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Find the Python the mediaplanner service runs (venv or system).
EXEC_PATH="$(systemctl show -p ExecStart --value mediaplanner | grep -o 'path=[^ ;]*' | head -1 | cut -d= -f2 || true)"
PYTHON=""
if [[ -n "$EXEC_PATH" ]]; then
  case "$(basename "$EXEC_PATH")" in
    python*) PYTHON="$EXEC_PATH" ;;
    *) [[ -x "$(dirname "$EXEC_PATH")/python3" ]] && PYTHON="$(dirname "$EXEC_PATH")/python3" ;;
  esac
fi
PYTHON="${PYTHON:-$(command -v python3)}"
echo "App dir: $APP_DIR"
echo "Python:  $PYTHON"

cat > /etc/systemd/system/mediaplanner-sync.service <<UNIT
[Unit]
Description=Media Planner: sync catalogues from Google Drive
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$APP_DIR
ExecStart=$PYTHON $APP_DIR/auto_sync.py
TimeoutStartSec=20min
UNIT

cat > /etc/systemd/system/mediaplanner-sync.timer <<UNIT
[Unit]
Description=Media Planner: Drive sync every 12 hours

[Timer]
OnBootSec=2min
OnUnitInactiveSec=12h
Persistent=true

[Install]
WantedBy=timers.target
UNIT

cat > /etc/systemd/system/mediaplanner-proxies.service <<UNIT
[Unit]
Description=Media Planner: make fast preview copies of source videos

[Service]
Type=oneshot
WorkingDirectory=$APP_DIR
ExecStart=$PYTHON $APP_DIR/make_proxies.py
Nice=15
IOSchedulingClass=idle
TimeoutStartSec=6h
UNIT

cat > /etc/systemd/system/mediaplanner-proxies.timer <<UNIT
[Unit]
Description=Media Planner: nightly preview copies

[Timer]
OnCalendar=*-*-* 03:30
Persistent=true

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now mediaplanner-sync.timer mediaplanner-proxies.timer
systemctl start mediaplanner-sync.service || true
echo
systemctl list-timers 'mediaplanner-*' --no-pager
echo
journalctl -u mediaplanner-sync.service -n 5 --no-pager
