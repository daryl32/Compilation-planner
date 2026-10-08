# Server deployment

| | Production | Test copy |
|---|---|---|
| Branch | `main` | `dev` |
| Address | https://video-compilation-planner.co.uk | https://dev.video-compilation-planner.co.uk |
| Code | `/root/Compilation-planner` | `/root/Compilation-planner-dev` |
| Data | `/root/data` | `/root/data-dev` (copy; videos, audio and preview copies shared) |
| Service | `mediaplanner` (port 8501) | `mediaplanner-dev` (port 8502) |
| Google Drive | reads and writes | reads only (writes switched off) |
| Background sync | every 12 h (`mediaplanner-sync.timer`) | none — refresh by hand |

A push to `main` deploys production; a push to `dev` deploys the test copy
(`/root/webhook.py`, from `deploy/webhook.py`). Other branches are ignored.

## Workflow
1. Commit to `dev` → test at the dev address.
2. Happy → merge `dev` into `main` → production updates.

## Scripts (run on the server as root)
- `setup_dev.sh` — one-time setup of the test copy (safe to re-run).
- `refresh_dev_data.sh` — replace the test copy's data with a fresh copy of production's.
- `install_timers.sh` — production's Drive sync and nightly preview-copy timers.

## Handy commands
```
systemctl status mediaplanner mediaplanner-dev webhook
journalctl -u mediaplanner-dev -f          # test copy's live log
sudo systemctl stop mediaplanner-dev       # free up RAM/CPU when not testing
sudo systemctl start mediaplanner-dev
```

## Notes
- `/root/webhook.py` is a copy: after changing `deploy/webhook.py`, copy it
  over and `systemctl restart webhook`.
- Config files are per copy and never in git: `config.py`, `oauth_config.py`,
  `.streamlit/secrets.toml`. The test copy's `config.py` ends with a
  "TEST COPY overrides" block (`APP_ENV = "test"`, `DRIVE_WRITES = False`,
  data paths).
