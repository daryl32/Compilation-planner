#!/usr/bin/env bash
# One-time setup of the TEST copy of the app, running next to production.
#
#   sudo bash /root/Compilation-planner/deploy/setup_dev.sh
#
# Before running: add a Cloudflare DNS record  dev → 116.203.145.32  (DNS only /
# grey cloud for now — see the end of this script's output).
#
# Creates:
#   /root/Compilation-planner-dev   clone on the 'dev' branch, own venv + config
#   /root/data-dev                  test data (copied from production)
#   mediaplanner-dev.service        Streamlit on 127.0.0.1:8502
#   nginx site + SSL                https://dev.video-compilation-planner.co.uk
#   /root/webhook.py                replaced: main → production, dev → test
# Production's service, data and config are not modified (the old webhook is
# backed up first). Safe to re-run.
set -euo pipefail

PROD_DIR=/root/Compilation-planner
DEV_DIR=/root/Compilation-planner-dev
DEV_DATA=/root/data-dev
PROD_DOMAIN=video-compilation-planner.co.uk
DEV_DOMAIN=dev.$PROD_DOMAIN
DEV_PORT=8502
DEV_BRANCH=dev

step() { echo; echo "=== $* ==="; }
[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }
[[ -d $PROD_DIR/.git ]] || { echo "$PROD_DIR not found."; exit 1; }

step "1/8  Check DNS"
if getent hosts "$DEV_DOMAIN" >/dev/null; then
  echo "$DEV_DOMAIN resolves: $(getent hosts "$DEV_DOMAIN" | awk '{print $1}' | head -1)"
else
  echo "WARNING: $DEV_DOMAIN doesn't resolve yet — the SSL step will fail until the"
  echo "Cloudflare record exists. Everything else will still be set up."
fi

step "2/8  Clone the dev branch"
if [[ ! -d $DEV_DIR/.git ]]; then
  git clone "$(git -C "$PROD_DIR" remote get-url origin)" "$DEV_DIR"
fi
git -C "$DEV_DIR" fetch origin
git -C "$DEV_DIR" checkout "$DEV_BRANCH"
git -C "$DEV_DIR" pull --ff-only

step "3/8  Python environment (same package versions as production)"
if [[ ! -x $DEV_DIR/venv/bin/python ]]; then
  "$PROD_DIR/venv/bin/python" -m venv "$DEV_DIR/venv"
fi
"$PROD_DIR/venv/bin/pip" freeze | grep -v -e '^-e ' -e ' @ file:' -e '^pkg[-_]resources' > /tmp/mediaplanner-dev-req.txt
"$DEV_DIR/venv/bin/pip" install -q --upgrade pip
"$DEV_DIR/venv/bin/pip" install -q -r /tmp/mediaplanner-dev-req.txt

step "4/8  Config, sign-in and OAuth files"
PROXY_DIR=$(cd "$PROD_DIR" && venv/bin/python -c "import auto_sync; print(auto_sync.PROXY_DIR)")
[[ -n "$PROXY_DIR" ]] || { echo "Couldn't read PROXY_DIR from production's config — stopping."; exit 1; }
cp "$PROD_DIR/config.py" "$DEV_DIR/config.py"
cat >> "$DEV_DIR/config.py" <<EOF

# ---- TEST COPY overrides (added by deploy/setup_dev.sh) ----
from pathlib import Path as _P
APP_ENV = "test"          # 🧪 banner on every page
DRIVE_WRITES = False      # never write to Google Drive from the test copy
_DEV_DATA = _P("$DEV_DATA")
CATALOGUE_DIR = _DEV_DATA / "catalogue"
AUDIO_DIR     = _DEV_DATA / "audio_catalogue"
PLANS_DIR     = _DEV_DATA / "compilation_plans"
PREVIEW_DIR   = _DEV_DATA / "previews"
PROJECTS_DIR  = _DEV_DATA / "projects"
PROXY_DIR     = _P("$PROXY_DIR")   # shared with production (read-only use)
EOF
echo "config.py written (data in $DEV_DATA, preview copies shared from $PROXY_DIR)"

if [[ -f $PROD_DIR/oauth_config.py ]]; then
  sed "s#https://$PROD_DOMAIN#https://$DEV_DOMAIN#g" "$PROD_DIR/oauth_config.py" > "$DEV_DIR/oauth_config.py"
fi
mkdir -p "$DEV_DIR/.streamlit"
for f in config.toml; do
  [[ -f $PROD_DIR/.streamlit/$f ]] && cp "$PROD_DIR/.streamlit/$f" "$DEV_DIR/.streamlit/$f"
done
if [[ -f $PROD_DIR/.streamlit/secrets.toml ]]; then
  NEW_COOKIE=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  sed -e "s#https://$PROD_DOMAIN#https://$DEV_DOMAIN#g" \
      -e "s#^\(\s*cookie_secret\s*=\s*\).*#\1\"$NEW_COOKIE\"#" \
      "$PROD_DIR/.streamlit/secrets.toml" > "$DEV_DIR/.streamlit/secrets.toml"
  chmod 600 "$DEV_DIR/.streamlit/secrets.toml"
  echo "secrets.toml: sign-in now returns to https://$DEV_DOMAIN"
else
  echo "WARNING: $PROD_DIR/.streamlit/secrets.toml not found — sign-in won't work on the test copy."
fi

step "5/8  Test data"
bash "$DEV_DIR/deploy/refresh_dev_data.sh"

step "6/8  Service mediaplanner-dev (port $DEV_PORT)"
cat > /etc/systemd/system/mediaplanner-dev.service <<EOF
[Unit]
Description=Media Planner — TEST copy ($DEV_BRANCH branch)
After=network.target

[Service]
WorkingDirectory=$DEV_DIR
ExecStart=$DEV_DIR/venv/bin/streamlit run $DEV_DIR/Home.py --server.port $DEV_PORT --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now mediaplanner-dev
systemctl restart mediaplanner-dev

step "7/8  nginx + SSL for $DEV_DOMAIN"
SITE=/etc/nginx/sites-available/mediaplanner-dev
cat > "$SITE" <<EOF
server {
    listen 80;
    server_name $DEV_DOMAIN;
    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:$DEV_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
EOF
ln -sfn "$SITE" /etc/nginx/sites-enabled/mediaplanner-dev
nginx -t
systemctl reload nginx
if certbot --nginx -d "$DEV_DOMAIN" --non-interactive --agree-tos --redirect; then
  echo "SSL certificate installed."
else
  echo "WARNING: certificate step failed (usually DNS not ready, or the Cloudflare"
  echo "record is orange-clouded). Fix that, then run:"
  echo "  sudo certbot --nginx -d $DEV_DOMAIN --redirect"
fi

step "8/8  Webhook: main → production, dev → test"
if [[ ! -f /root/.webhook_secret ]]; then
  "$PROD_DIR/venv/bin/python" - <<'PY'
import ast, os
src = open("/root/webhook.py").read()
secret = None
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SECRET" for t in node.targets):
        secret = ast.literal_eval(node.value)
if secret is None:
    raise SystemExit("Couldn't read SECRET from /root/webhook.py — webhook left unchanged.")
if isinstance(secret, str):
    secret = secret.encode()
with open("/root/.webhook_secret", "wb") as f:
    f.write(secret)
os.chmod("/root/.webhook_secret", 0o600)
print("Secret moved to /root/.webhook_secret")
PY
fi
BACKUP="/root/webhook.py.bak-$(date +%Y%m%d-%H%M%S)"
cp /root/webhook.py "$BACKUP"
cp "$PROD_DIR/deploy/webhook.py" /root/webhook.py
systemctl restart webhook
sleep 2
if systemctl is-active --quiet webhook; then
  echo "Webhook restarted OK (old version saved as $BACKUP)."
else
  echo "WARNING: new webhook didn't start — restoring the old one."
  cp "$BACKUP" /root/webhook.py
  systemctl restart webhook
  journalctl -u webhook -n 15 --no-pager
fi

echo
echo "================================================================"
echo " Test copy:   https://$DEV_DOMAIN"
echo " Production:  https://$PROD_DOMAIN   (unchanged)"
echo
echo " Still to do by hand (once):"
echo "  1. Google Cloud console → APIs & Services → Credentials → your"
echo "     OAuth client → Authorized redirect URIs → add:"
echo "        https://$DEV_DOMAIN/oauth2callback"
echo "  2. If production's Cloudflare record is orange-clouded, switch the"
echo "     dev record to orange too now that the certificate exists."
echo
echo " Refresh test data from production any time:"
echo "   sudo bash $DEV_DIR/deploy/refresh_dev_data.sh"
echo "================================================================"
