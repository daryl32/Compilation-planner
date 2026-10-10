"""
Switch between the live app and the test copy on the server, so only one runs
at a time (each takes a share of the server's 4 GB).

The sidebar's "🔀 Live / test copy" panel (page_setup.py) uses this:
  1. start the other copy and wait until it answers,
  2. send you there,
  3. stop this copy a few seconds later.

Safety nets, installed once by deploy/setup_switch.sh:
  • whenever the test copy stops (switch, crash, idle), the live app starts;
  • the test copy stops itself after IDLE minutes with nobody using it
    (it touches HEARTBEAT on every page run — see touch_heartbeat());
  • after a reboot only the live app starts;
  • a copy that's off shows a "switched off" page with links, not an error.

The panel only appears once that script has run (SETUP_MARKER exists), on the
Linux server where the app runs as root. Nothing here runs on the laptop.
"""

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import config

SETUP_MARKER = Path("/etc/mediaplanner-switch.enabled")
HEARTBEAT_DIR = Path("/var/lib/mediaplanner")

COPIES = {
    "live": {
        "name": "live app",
        "service": getattr(config, "LIVE_SERVICE", "mediaplanner"),
        "port": int(getattr(config, "LIVE_PORT", 8501)),
        "url": getattr(config, "LIVE_URL", "https://video-compilation-planner.co.uk"),
    },
    "test": {
        "name": "test copy",
        "service": getattr(config, "TEST_SERVICE", "mediaplanner-dev"),
        "port": int(getattr(config, "TEST_PORT", 8502)),
        "url": getattr(config, "TEST_URL", "https://dev.video-compilation-planner.co.uk"),
    },
}


def this_copy() -> str:
    is_test = str(getattr(config, "APP_ENV", "production")).lower() == "test"
    return "test" if is_test else "live"


def other_copy() -> str:
    return "live" if this_copy() == "test" else "test"


def available() -> bool:
    """True on the server once deploy/setup_switch.sh has been run."""
    return (os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0
            and shutil.which("systemctl") is not None and SETUP_MARKER.exists())


def touch_heartbeat() -> None:
    """Mark this copy as in use (the test copy's idle shut-down reads this)."""
    if not HEARTBEAT_DIR.is_dir():
        return
    try:
        (HEARTBEAT_DIR / f"{COPIES[this_copy()]['service']}.heartbeat").touch()
    except OSError:
        pass


def is_running(copy: str) -> bool:
    proc = subprocess.run(["systemctl", "is-active", COPIES[copy]["service"]],
                          capture_output=True, text=True)
    return proc.stdout.strip() == "active"


def is_healthy(copy: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{COPIES[copy]['port']}/_stcore/health",
                                    timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def start(copy: str, wait_sec: int = 90, progress=None) -> str | None:
    """Start a copy and wait until it answers. None on success, else an error."""
    svc = COPIES[copy]["service"]
    if copy == "test":
        touch_heartbeat_for(svc)   # a fresh start isn't "idle"
    proc = subprocess.run(["systemctl", "start", svc], capture_output=True, text=True)
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout or "systemctl failed").strip()
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if is_healthy(copy):
            return None
        if progress:
            progress(1 - (deadline - time.time()) / wait_sec)
        time.sleep(1.5)
    return f"The {COPIES[copy]['name']} didn't answer within {wait_sec} seconds."


def touch_heartbeat_for(service: str) -> None:
    if HEARTBEAT_DIR.is_dir():
        try:
            (HEARTBEAT_DIR / f"{service}.heartbeat").touch()
        except OSError:
            pass


def stop_later(copy: str, delay_sec: int = 8) -> str | None:
    """Stop a copy after a short delay (so the page answering this click can
    finish first). None on success, else an error."""
    svc = COPIES[copy]["service"]
    proc = subprocess.run(
        ["systemd-run", f"--on-active={delay_sec}", "--timer-property=AccuracySec=1s",
         f"--unit=mediaplanner-switch-{int(time.time())}", "/bin/systemctl", "stop", svc],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout or "systemd-run failed").strip()
    return None


def stop_now(copy: str) -> str | None:
    proc = subprocess.run(["systemctl", "stop", COPIES[copy]["service"]], capture_output=True, text=True)
    return None if proc.returncode == 0 else (proc.stderr or "systemctl failed").strip()
