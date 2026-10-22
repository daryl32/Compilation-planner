#!/usr/bin/env bash
# Installs two systemd timers on the server (run once, as root; safe to re-run):
#   mediaplanner-proxies  — nightly fast-preview copies at 03:30 (make_proxies.py)
#   mediaplanner-recs     — nightly 🔮 recommendation searches at 04:45, only what's changed (jobs.py)
# Drive → server sync has no timer any more (Oct 2026): it runs only from the app's 🔄 Sync now
# button (auto_sync.py). Re-running this switches off and removes the old 12-hourly sync timer.
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

# The old 12-hourly Drive sync timer — switched off and removed (sync is manual now).
if systemctl list-unit-files mediaplanner-sync.timer --no-legend 2>/dev/null | grep -q mediaplanner-sync; then
  systemctl disable --now mediaplanner-sync.timer || true
fi
rm -f /etc/systemd/system/mediaplanner-sync.timer /etc/systemd/system/mediaplanner-sync.service

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

cat > /etc/systemd/system/mediaplanner-recs.service <<UNIT
[Unit]
Description=Media Planner: nightly recommendation searches (only what's changed)

[Service]
Type=oneshot
WorkingDirectory=$APP_DIR
ExecStart=$PYTHON $APP_DIR/jobs.py --nightly
Nice=10
IOSchedulingClass=idle
TimeoutStartSec=6h
UNIT

cat > /etc/systemd/system/mediaplanner-recs.timer <<UNIT
[Unit]
Description=Media Planner: nightly recommendation searches

[Timer]
OnCalendar=*-*-* 04:45
Persistent=true

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now mediaplanner-proxies.timer mediaplanner-recs.timer
echo
systemctl list-timers 'mediaplanner-*' --no-pager
echo
