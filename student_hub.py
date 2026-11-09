r"""
student_hub.py — the server as the one place for the student's models, run records and laptop vision data
(claude/results-pages-audit.md, 9 Nov). No torch, no Modal: plain SSH.

On the laptop, config.py (or the STUDENT_SERVER environment variable) names the server:
    STUDENT_SERVER = "root@<server address>"
and an SSH key lets it log in without a password (BatchMode — a missing key fails at once, never hangs on a
prompt). Then:
  • push_models / pull_models — models/<run>/ to and from the server's student folder, so any machine can
    rescore every run;
  • push_runs — this machine's run records (runs@<machine>.json + <run>@<machine>.json) straight into the
    server's catalogue/student_runs, where the app merges them;
  • push_videos — laptop vision data (local_vision.py) into the server's student videos folder.
On the server itself nothing is set, so every function here does nothing (its files are already local).

Files travel as a tar stream over ssh (Python's tarfile on this side, tar on the server), so video names with
spaces are fine and Windows needs nothing beyond its built-in OpenSSH client.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tarfile
from pathlib import Path

DEFAULT_CATALOGUE = "/root/data/catalogue"
_PATHS: dict = {}


def server() -> str | None:
    """"user@host" of the server, or None (on the server itself, or not set up)."""
    host = os.environ.get("STUDENT_SERVER")
    if not host:
        try:
            import config
            host = getattr(config, "STUDENT_SERVER", None)
        except Exception:               # noqa: BLE001 — no config.py
            host = None
    return host or None


def _ssh_base() -> list:
    return shlex.split(os.environ.get("STUDENT_SSH") or "ssh -o BatchMode=yes -o ConnectTimeout=15")


def run_remote(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(_ssh_base() + [server(), cmd], capture_output=True, text=True, **kw)


def remote_paths() -> dict:
    """{"work": the server's _student folder, "catalogue": its catalogue folder} — asked once per run."""
    if _PATHS.get("host") == server():
        return _PATHS
    cat = os.environ.get("STUDENT_SERVER_CATALOGUE")
    if not cat:
        try:
            import config
            cat = getattr(config, "STUDENT_SERVER_CATALOGUE", None)
        except Exception:               # noqa: BLE001
            cat = None
    r = run_remote('d=$(ls -d /mnt/HC_Volume_*/student 2>/dev/null | head -1); '
                   'echo "${STUDENT_WORK_DIR:-${d:-/root/data/student}}/_student"', timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(why_failed(r))
    _PATHS.clear()
    _PATHS.update(host=server(), work=r.stdout.strip().splitlines()[-1], catalogue=cat or DEFAULT_CATALOGUE)
    return _PATHS


def why_failed(r: subprocess.CompletedProcess) -> str:
    err = (r.stderr or "").strip().splitlines()
    msg = err[-1] if err else f"ssh exit {r.returncode}"
    if "Permission denied" in msg or "publickey" in msg:
        msg += " — the laptop's SSH key isn't on the server yet (see claude/results-pages-audit.md, SSH key setup)"
    return msg


def remote_ls(path: str) -> list:
    r = run_remote(f"ls -1 {shlex.quote(path)} 2>/dev/null || true", timeout=60)
    if r.returncode != 0:
        raise RuntimeError(why_failed(r))
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def send(items: list, remote_dir: str) -> None:
    """items: [(local file, path inside remote_dir)] → written there on the server (folders made as needed)."""
    if not items:
        return
    q = shlex.quote(remote_dir)
    proc = subprocess.Popen(_ssh_base() + [server(), f"mkdir -p {q} && tar -xf - -C {q}"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
            for local, arc in items:
                tar.add(str(local), arcname=str(arc).replace("\\", "/"), recursive=False)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(why_failed(subprocess.CompletedProcess(proc.args, proc.returncode, out, err.decode(
            "utf-8", "replace") if isinstance(err, bytes) else err)))


def fetch_dirs(names: list, remote_dir: str, local_dir: Path) -> None:
    """The named folders inside remote_dir on the server → local_dir."""
    if not names:
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    args = " ".join(shlex.quote(n) for n in names)
    proc = subprocess.Popen(_ssh_base() + [server(), f"tar -cf - -C {shlex.quote(remote_dir)} {args}"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        try:
            tar.extractall(local_dir, filter="data")
        except TypeError:               # Python without extraction filters
            tar.extractall(local_dir)
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(why_failed(subprocess.CompletedProcess(proc.args, proc.returncode, "",
                                                                  err.decode("utf-8", "replace"))))
