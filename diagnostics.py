"""🩺 Diagnostics — what the whole server is doing, for the Server page.

  • system_memory()   RAM and swap for the whole server (not just this app)
  • cpu()             cores and load averages
  • app_processes()   memory of the live app, the test copy, the audio worker,
                      the webhook and anything else big, by reading /proc
  • memory_history()  the app's own memory log (logs/memory.csv, written by
                      library_common.memory_check) summarised per day
  • timings / timed() how long the slow steps take (segmentation, ranking,
                      Auto matching, shape search, page runs), logged to
                      logs/timings.csv and summarised as median / 90% / max
  • verdict()         a one-line reading of the numbers: is RAM, CPU or disk
                      the thing holding the server back?

No Streamlit in here — the Server page draws it. Everything returns None or
empty results off Linux (e.g. on the laptop) instead of failing.
"""

from __future__ import annotations

import csv
import datetime
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PROC = Path("/proc")

from config import CATALOGUE_DIR

LOG_DIR = CATALOGUE_DIR.parent / "logs"
# Timings are only logged on the server: on the laptop LOG_DIR is inside Google
# Drive, and a file that changes on every page run must never sync.
TIMINGS_ENABLED = os.name != "nt" and PROC.exists()
MEMORY_LOG = LOG_DIR / "memory.csv"
TIMINGS_LOG = LOG_DIR / "timings.csv"
_TIMINGS_MAX_BYTES = 2_000_000


# ---------------------------------------------------------------------------
# The whole server
# ---------------------------------------------------------------------------

def _meminfo(proc: Path = PROC) -> dict:
    out = {}
    try:
        for line in (proc / "meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[key] = int(parts[0]) / 1024          # kB → MB
    except (OSError, ValueError):
        return {}
    return out


def system_memory(proc: Path = PROC) -> dict | None:
    """{"total", "used", "available", "swap_total", "swap_used"} in MB, or None."""
    m = _meminfo(proc)
    if "MemTotal" not in m:
        return None
    total = m["MemTotal"]
    avail = m.get("MemAvailable", m.get("MemFree", 0.0))
    swap_total = m.get("SwapTotal", 0.0)
    return {"total": total, "available": avail, "used": max(total - avail, 0.0),
            "swap_total": swap_total, "swap_used": max(swap_total - m.get("SwapFree", swap_total), 0.0)}


def cpu(proc: Path = PROC) -> dict | None:
    """{"cores", "load1", "load5", "load15"}, or None."""
    try:
        l1, l5, l15 = (float(x) for x in (proc / "loadavg").read_text().split()[:3])
    except (OSError, ValueError):
        return None
    return {"cores": os.cpu_count() or 1, "load1": l1, "load5": l5, "load15": l15}


def _classify(cmd: str) -> str | None:
    if "streamlit" in cmd and "Compilation-planner-dev" in cmd:
        return "🧪 Test copy"
    if "streamlit" in cmd and "Home.py" in cmd:
        return "🟢 Live app"
    if "audio_analysis" in cmd:
        return "🎵 Audio analysis worker"
    if "webhook.py" in cmd:
        return "🔗 Deploy webhook"
    if "ffmpeg" in cmd:
        return "🎬 ffmpeg (render / proxy)"
    if "make_proxies" in cmd or "auto_sync" in cmd or "drive_sync" in cmd:
        return "☁️ Drive sync / proxies"
    return None


def app_processes(proc: Path = PROC, min_other_mb: float = 50.0) -> list:
    """[(label, MB, count)] biggest first — the app's own services grouped by
    what they are; anything else over min_other_mb listed by name."""
    groups: dict = {}
    try:
        pids = [p for p in proc.iterdir() if p.name.isdigit()]
    except OSError:
        return []
    for p in pids:
        try:
            rss = 0.0
            name = ""
            for line in (p / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024
                elif line.startswith("Name:"):
                    name = line.split(None, 1)[1].strip()
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError, IndexError):
            continue
        if rss <= 0:
            continue
        label = _classify(cmd)
        if label is None:
            if rss < min_other_mb:
                continue
            label = f"⚙️ {name or 'other'}"
        mb, n = groups.get(label, (0.0, 0))
        groups[label] = (mb + rss, n + 1)
    return sorted(((k, mb, n) for k, (mb, n) in groups.items()), key=lambda r: -r[1])


# ---------------------------------------------------------------------------
# Memory history (library_common.memory_check writes it)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list:
    rows = []
    for p in (path.with_suffix(".old.csv"), path):
        try:
            with open(p, newline="") as f:
                rows.extend(csv.DictReader(f))
        except OSError:
            continue
    return rows


def memory_history(days: int = 7, path: Path = None) -> list:
    """[{"day", "peak_mb", "median_mb", "samples", "clears"}] for the last `days`
    days, oldest first. clears = times the app cleared its caches at the limit."""
    path = path or MEMORY_LOG
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    by_day: dict = {}
    for r in _read_csv(path):
        try:
            t = datetime.datetime.fromisoformat(r["time"])
            mb = float(r["memory_mb"])
        except (KeyError, ValueError, TypeError):
            continue
        if t < cutoff:
            continue
        d = by_day.setdefault(t.date().isoformat(), {"values": [], "clears": 0})
        d["values"].append(mb)
        if "clear" in (r.get("note") or "").lower():
            d["clears"] += 1
    out = []
    for day in sorted(by_day):
        v = sorted(by_day[day]["values"])
        out.append({"day": day, "peak_mb": v[-1], "median_mb": v[len(v) // 2], "samples": len(v),
                    "clears": by_day[day]["clears"]})
    return out


# ---------------------------------------------------------------------------
# Step timings
# ---------------------------------------------------------------------------

_lock = threading.Lock()


def record(step: str, seconds: float, detail: str = "", path: Path = None) -> None:
    """Append one timing. Never raises — timing must not break a page. Does
    nothing off the server (see TIMINGS_ENABLED) unless a path is given."""
    if path is None and not TIMINGS_ENABLED:
        return
    path = path or TIMINGS_LOG
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > _TIMINGS_MAX_BYTES:
                path.replace(path.with_suffix(".old.csv"))
            new = not path.exists()
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "step", "seconds", "detail"])
                w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), step,
                            f"{seconds:.3f}", detail])
    except Exception:
        pass


@contextmanager
def timed(step: str, detail: str = ""):
    """with timed("Auto matching", "40 blocks"): ... — logs how long it took
    (also when the block raises)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(step, time.perf_counter() - t0, detail)


def _pct(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def timing_summary(days: int = 7, path: Path = None) -> list:
    """[{"step", "runs", "median", "p90", "max", "last"}] (seconds), slowest
    median first, over the last `days` days."""
    path = path or TIMINGS_LOG
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    by_step: dict = {}
    for r in _read_csv(path):
        try:
            t = datetime.datetime.fromisoformat(r["time"])
            s = float(r["seconds"])
        except (KeyError, ValueError, TypeError):
            continue
        if t < cutoff:
            continue
        e = by_step.setdefault(r.get("step") or "?", {"vals": [], "last": None})
        e["vals"].append(s)
        e["last"] = s
    out = []
    for step, e in by_step.items():
        v = sorted(e["vals"])
        out.append({"step": step, "runs": len(v), "median": _pct(v, 0.5), "p90": _pct(v, 0.9),
                    "max": v[-1], "last": e["last"]})
    return sorted(out, key=lambda r: -r["median"])


# ---------------------------------------------------------------------------
# Reading the numbers
# ---------------------------------------------------------------------------

def verdict(mem: dict | None, cpu_info: dict | None, history: list, disk_pct: float | None) -> list:
    """Plain-language notes on what's limiting the server, most important first."""
    notes = []
    if mem:
        used_pct = mem["used"] / max(mem["total"], 1) * 100
        swap_pct = mem["swap_used"] / max(mem["swap_total"], 1) * 100 if mem["swap_total"] else 0.0
        if swap_pct > 25 or used_pct > 85:
            notes.append(("warn", f"RAM is the limit: {used_pct:.0f}% used and {mem['swap_used']:,.0f} MB in swap. "
                                  "More RAM (or fewer things running at once) would help most."))
        elif used_pct > 65:
            notes.append(("info", f"RAM is fairly busy ({used_pct:.0f}% used) but not swapping much."))
        else:
            notes.append(("ok", f"RAM is fine right now ({used_pct:.0f}% used)."))
    if history and any(d["clears"] for d in history):
        n = sum(d["clears"] for d in history)
        notes.append(("warn", f"The app hit its memory limit and cleared its caches {n} time(s) this week."))
    if cpu_info:
        per_core = cpu_info["load5"] / max(cpu_info["cores"], 1)
        if per_core > 1.0:
            notes.append(("warn", f"The CPU is overloaded (5-min load {cpu_info['load5']:.1f} on "
                                  f"{cpu_info['cores']} cores) — renders and analysis will be slow."))
        elif per_core > 0.7:
            notes.append(("info", f"The CPU is busy (5-min load {cpu_info['load5']:.1f} on {cpu_info['cores']} cores)."))
        else:
            notes.append(("ok", f"The CPU has headroom (5-min load {cpu_info['load5']:.1f} on {cpu_info['cores']} cores)."))
    if disk_pct is not None and disk_pct >= 80:
        notes.append(("warn", f"The disk is {disk_pct:.0f}% full."))
    order = {"warn": 0, "info": 1, "ok": 2}
    return sorted(notes, key=lambda n: order[n[0]])


def tested_streamlit() -> str | None:
    """The Streamlit version requirements.txt pins (streamlit[...]==X), or None."""
    import re
    from pathlib import Path
    try:
        text = (Path(__file__).resolve().parent / "requirements.txt").read_text()
    except OSError:
        return None
    m = re.search(r"^streamlit(?:\[[^\]]*\])?==([\w.]+)", text, re.M)
    return m.group(1) if m else None
