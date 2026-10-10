"""
GitHub webhook listener — deploys production and the test copy.

  push to main → git pull in /root/Compilation-planner      + restart mediaplanner
  push to dev  → git pull in /root/Compilation-planner-dev  + restart mediaplanner-dev
  any other branch → ignored

A copy that's switched off (see "🔀 Live / test copy", deploy/setup_switch.sh)
is updated but left off — it picks up the new code next time it starts. A pull
that fails (e.g. local edits on the server) is reported back to GitHub as an
error and the app is NOT restarted, instead of silently running old code.

Installed to /root/webhook.py by deploy/setup_dev.sh (run as webhook.service).
The GitHub secret is read from /root/.webhook_secret, so this file holds no
secrets and can live in the repo. Signature check is unchanged from before.
"""

import hashlib
import hmac
import json
import subprocess
from pathlib import Path

from flask import Flask, request

SECRET = Path("/root/.webhook_secret").read_bytes().strip()

TARGETS = {
    "refs/heads/main": ("/root/Compilation-planner", "mediaplanner"),
    "refs/heads/dev": ("/root/Compilation-planner-dev", "mediaplanner-dev"),
}

app = Flask(__name__)


def _payload() -> dict:
    """GitHub sends JSON, or form-encoded with the JSON in 'payload',
    depending on the webhook's content-type setting — accept both."""
    data = request.get_json(silent=True)
    if data is None and "payload" in request.form:
        data = json.loads(request.form["payload"])
    return data or {}


@app.route("/webhook", methods=["POST"])
def webhook():
    sig = request.headers.get("X-Hub-Signature-256", "")
    body = request.get_data()
    expected = "sha256=" + hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return "Unauthorized", 401

    if request.headers.get("X-GitHub-Event") == "ping":
        return "pong", 200

    ref = _payload().get("ref", "")
    target = TARGETS.get(ref)
    if target is None:
        return f"Ignored {ref or 'event'}", 200
    repo_dir, service = target
    if not Path(repo_dir).is_dir():
        return f"{repo_dir} not set up — ignored", 200

    pull = subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], capture_output=True, text=True)
    if pull.returncode != 0:
        return f"git pull failed in {repo_dir} — not restarted:\n{pull.stderr or pull.stdout}", 500
    subprocess.run(["systemctl", "try-restart", service])   # restarts it only if it's running
    return f"Deployed {ref} → {service}", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
