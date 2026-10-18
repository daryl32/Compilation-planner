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
1. Commit to `dev` → test at the dev address. GitHub runs the tests (`tests/`) on the push.
2. For changes that shouldn't alter results: Server page → 🧪 *Same results as the live app?* on the
   test copy (or `python3 deploy/check_parity.py` as root) compares both copies' code on real data.
3. Happy → merge `dev` into `main` → production updates.

## Scripts (run on the server as root)
- `setup_switch.sh` — one-time setup of the sidebar's **🔀 Live / test copy** switch:
  only one copy runs at a time; live restarts by itself whenever the test copy
  stops; the test copy stops after 30 min unused; only live starts at boot;
  a switched-off copy shows a page with links instead of 502 (safe to re-run).
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

## Memory
- The sidebar shows the app's memory. Each copy logs it to `<data>/logs/memory.csv`
  (e.g. `tail -30 /root/data/logs/memory.csv`) — which page / Choreography block, and when.
- Past `MEMORY_LIMIT_MB` (config.py, default 1800) the app quietly clears its cached data
  and held media and hands the memory back (at most every 10 min), logged as "cleared caches".
- Server page → 🧠 Memory: what's using it, release held media / clear caches, ♻️ Restart.
- Unsaved planner work is autosaved per user to `<data>/…/projects/.autosave/`
  and offered back after a restart.

## Notes
- Deploys: a failed `git pull` (e.g. local edits on the server) is reported to
  GitHub as a failed webhook delivery and the app is not restarted.
- `/root/webhook.py` is a copy: after changing `deploy/webhook.py`, copy it
  over and `systemctl restart webhook`.
- Config files are per copy and never in git: `config.py`, `oauth_config.py`,
  `.streamlit/secrets.toml`. The test copy's `config.py` ends with a
  "TEST COPY overrides" block (`APP_ENV = "test"`, `DRIVE_WRITES = False`,
  data paths).
