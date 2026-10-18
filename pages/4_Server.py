"""
Server — the app's memory (what's using it, and clearing it), disk space, and
clearing what can be fetched again (see storage.py).
"""

import datetime

import streamlit as st

import storage as SG
from page_setup import page_setup

page_setup("Server", "pages/4_Server.py", title="🖥️ Server")

if not SG.cleanup_allowed():
    st.info("This copy isn't running on the server — on the laptop these folders are your Google Drive, "
            "so disk clean-up is switched off here.")


@st.cache_data(ttl=600, show_spinner="Measuring folders…")
def breakdown():
    return SG.usage_breakdown()


def done(msg: str) -> None:
    breakdown.clear()
    st.session_state["srv_msg"] = msg
    st.rerun()


_msg = st.session_state.pop("srv_msg", None)
if _msg:
    st.success(_msg)

# ---------------------------------------------------------------------------
# 🩺 Diagnostics — the whole server, memory over the week, how long things take
# ---------------------------------------------------------------------------
import diagnostics as DG
import library_common as LC


def render_diagnostics() -> None:
    st.subheader("🩺 Diagnostics")
    mem, cpu_info = DG.system_memory(), DG.cpu()
    history = DG.memory_history(7)
    try:
        disk_pct = SG.disk_usage()["pct"]
    except Exception:
        disk_pct = None
    with st.container(border=True):
        if mem is None and cpu_info is None:
            st.caption("Server figures are only available on the server.")
        for kind, text in DG.verdict(mem, cpu_info, history, disk_pct):
            {"warn": st.warning, "info": st.info, "ok": st.success}[kind](text)
        if mem:
            c = st.columns(4)
            c[0].metric("Server RAM used", f"{mem['used']:,.0f} MB", help=f"of {mem['total']:,.0f} MB")
            c[1].metric("Available", f"{mem['available']:,.0f} MB")
            c[2].metric("Swap used", f"{mem['swap_used']:,.0f} MB",
                        help=f"of {mem['swap_total']:,.0f} MB — memory parked on disk; much of it means RAM is short")
            if cpu_info:
                c[3].metric("CPU load (5 min)", f"{cpu_info['load5']:.2f}",
                            help=f"{cpu_info['cores']} cores · 1 min {cpu_info['load1']:.2f} · "
                                 f"15 min {cpu_info['load15']:.2f}. Above the core count = overloaded.")
            st.progress(min(mem["used"] / max(mem["total"], 1), 1.0),
                        text=f"RAM {mem['used'] / max(mem['total'], 1):.0%} used")
        procs = DG.app_processes()
        if procs:
            st.markdown("**What's using the server's memory**")
            for label, mb, n in procs[:10]:
                rc = st.columns([3, 1, 4])
                rc[0].markdown(label + (f" ×{n}" if n > 1 else ""))
                rc[1].markdown(f"{mb:,.0f} MB")
                rc[2].progress(min(mb / max((mem or {}).get("total", mb) or 1, 1), 1.0))
        if history:
            st.markdown("**This app's memory, last 7 days** (peak and typical per day)")
            import pandas as pd
            df = pd.DataFrame(history).set_index("day")[["peak_mb", "median_mb"]]
            df.columns = ["Peak MB", "Typical MB"]
            st.bar_chart(df, stack=False, height=220)
            clears = sum(d["clears"] for d in history)
            if clears:
                st.caption(f"Hit the {LC.MEMORY_LIMIT_MB:,.0f} MB limit and cleared caches {clears} time(s).")
        timings = DG.timing_summary(7)
        if timings:
            st.markdown("**How long things take** (last 7 days, seconds)")
            st.dataframe(
                [{"Step": t["step"], "Runs": t["runs"], "Typical": round(t["median"], 2),
                  "Slow (90%)": round(t["p90"], 2), "Slowest": round(t["max"], 2), "Last": round(t["last"], 2)}
                 for t in timings],
                hide_index=True, use_container_width=True)
        else:
            st.caption("Step timings appear here once the planner has been used.")


render_diagnostics()


# ---------------------------------------------------------------------------
# 🧪 Test copy only: compare this code's cutting & matching with the live app's
# ---------------------------------------------------------------------------
def render_parity_check() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path
    import config
    here = Path(__file__).resolve().parents[1]
    live = Path(getattr(config, "LIVE_REPO_DIR", "/root/Compilation-planner"))
    data = LC.CATALOGUE_DIR.parent
    report = data / "logs" / "parity_report.txt"
    log = data / "logs" / "parity_run.log"
    st.subheader("🧪 Same results as the live app?")
    with st.container(border=True):
        st.caption("Cuts every track with all four methods, runs 🎯 Auto-detect and Auto matching with THIS "
                   "copy's code and with the live app's code, on this copy's data, and checks the results are "
                   "identical. Use it before pushing a change that shouldn't alter results. Takes a few minutes "
                   "and uses the CPU while it runs.")
        pid_file = data / "logs" / "parity.pid"
        running = None
        try:
            pid = int(pid_file.read_text())
            os.kill(pid, 0)                     # still running? (any page, any session)
            try:                                # finished but not yet reaped → done
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    raise ProcessLookupError
            except ChildProcessError:
                pass                            # not our child — another page started it
            if "Z" in Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[:1]:
                raise ProcessLookupError
            running = pid
        except (OSError, ValueError, IndexError):
            pass
        if running:
            st.info("⏳ Running — this takes a few minutes. Press 🔄 to check.")
            if st.button("🔄 Check again"):
                st.rerun()
            st.caption("A restart or idle shut-down of this test copy stops it — run it again if so.")
            return
        try:
            live_ok = live.exists()
        except OSError:
            live_ok = False
        if not live_ok:
            st.caption(f"The live app's code isn't at {live} (set LIVE_REPO_DIR in config.py).")
        elif st.button("▶️ Run the comparison"):
            log.parent.mkdir(parents=True, exist_ok=True)
            report.unlink(missing_ok=True)
            with open(log, "w") as f:
                proc = subprocess.Popen([sys.executable, str(here / "deploy" / "check_parity.py"), str(live),
                                         str(here), "--data", str(data)],
                                        stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            pid_file.write_text(str(proc.pid))
            st.rerun()
        if report.exists():
            st.code(report.read_text(), language=None)
        elif log.exists() and log.stat().st_size:
            st.error("The last comparison stopped without a report. Its output:")
            st.code(log.read_text()[-3000:], language=None)


if LC.IS_TEST:
    render_parity_check()

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
import library_common as LC


def mem_done(before, label: str) -> None:
    after = LC.app_memory_mb()
    if before and after:
        st.session_state["srv_msg"] = f"{label}: {before:,.0f} → {after:,.0f} MB."
    else:
        st.session_state["srv_msg"] = f"{label}."
    st.rerun()


st.subheader("🧠 Memory")
mb = LC.app_memory_mb()
deep = st.session_state.pop("srv_mem_deep", False)
br = LC.memory_breakdown(include_sessions=deep)
with st.container(border=True):
    if mb is None:
        st.caption("Memory figures are only available on the server.")
    else:
        limit = LC.MEMORY_LIMIT_MB
        top = st.columns([3, 1, 1])
        top[0].markdown(f"**This copy of the app: {mb:,.0f} MB**  ·  {br['n_sessions']} open page(s)")
        top[0].progress(min(mb / max(limit, 1), 1.0),
                        text=f"Clears cached data by itself above {limit:,.0f} MB (quietly)")
        rows = [
            ("🎞️ Videos, music & pictures held for players", br["media"],
             "Each player loads its whole file into memory and keeps it while a page shows it — "
             "usually the biggest part. 🎞️ Release videos & pictures clears it."),
            ("🗂️ Cached data", br["cache"], "Track analyses, rankings, page data — rebuilt when needed."),
            ("📚 Shared data", br["shared"], "Video catalogues, motion curves — reloaded when needed."),
        ]
        if deep:
            rows.append(("📝 Open pages' working data", br["sessions"],
                         "Everything each open page (yours and other tabs) is holding — planner progress etc."))
        known = sum(r[1] for r in rows)
        rows.append(("⚙️ The app itself & memory not yet handed back", max(mb - known, 0.0),
                     "Python, its libraries, and memory freed but not yet returned to the server. "
                     "Only ♻️ Restart brings this right down."))
        for label, size, note in rows:
            rc = st.columns([3, 1, 4])
            rc[0].markdown(f"**{label}**")
            rc[1].markdown(f"{size:,.0f} MB")
            rc[2].caption(note)
        if not deep and st.button("📏 Also measure the open pages' working data",
                                  help="Walks every open page's data — can take a few seconds."):
            st.session_state["srv_mem_deep"] = True
            st.rerun()

    bc = st.columns(4)
    if bc[0].button("🎞️ Release videos & pictures", use_container_width=True,
                    help="Drops every video, music file and picture held for a player. Pages still open "
                         "load their players again next time they update."):
        b = LC.app_memory_mb()
        r = LC.free_shared_memory(release_media=True)
        mem_done(b, f"Released {r['media_files']} file(s) ({r['media_mb']:,.0f} MB) and cached data")
    if bc[1].button("🗂️ Clear cached data", use_container_width=True,
                    help="Cached and shared data only — rebuilt the next time a page needs it."):
        b = LC.app_memory_mb()
        LC.free_shared_memory(release_media=False)
        mem_done(b, "Cleared cached data")
    if bc[2].button("🧹 Also clear this page's data", use_container_width=True,
                    help="Everything above, plus this browser tab's own working data (unsaved planner "
                         "work is saved to its recovery file first and offered back)."):
        b = LC.app_memory_mb()
        from page_setup import _reset_memory
        _reset_memory()
        mem_done(b, "Cleared")
    restart_ok = st.session_state.get("srv_restart_ok", False)
    if bc[3].button("♻️ Restart the app", use_container_width=True, type="primary" if restart_ok else "secondary",
                    help="The only way to get ALL memory back. Takes ~20 seconds; every open page "
                         "reconnects by itself. Unsaved planner work is offered back afterwards."):
        if not restart_ok:
            st.session_state["srv_restart_ok"] = True
            st.rerun()
        st.session_state.pop("srv_restart_ok", None)
        hook = st.session_state.get("_before_reset")
        if callable(hook):
            try:
                hook()
            except Exception:
                pass
        import app_switch as AS
        err = AS.restart_later()
        if err:
            st.error(f"Couldn't restart: {err}")
        else:
            st.info("♻️ Restarting in a few seconds — this page will reconnect by itself.")
    if restart_ok:
        st.warning("Press ♻️ Restart the app again to confirm. Anyone else using the app will see it "
                   "reconnect; unsaved planner work is offered back afterwards.")
    st.caption("Cleared memory isn't always handed straight back to the server — Python keeps some "
               "for reuse, so the total can drop less than what was cleared. That's normal; "
               "♻️ Restart is the full reset.")

# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------
d = SG.disk_usage()
with st.container(border=True):
    c = st.columns([3, 1, 1, 1])
    c[0].markdown(f"**💽 Disk: {d['pct']:.0f}% used**")
    c[0].progress(min(d["pct"] / 100, 1.0))
    c[1].metric("Used", SG.fmt_bytes(d["used"]))
    c[2].metric("Free", SG.fmt_bytes(d["free"]))
    c[3].metric("Size", SG.fmt_bytes(d["total"]))
    if d["pct"] >= 85:
        st.error("The disk is nearly full — renders and downloads may fail. Clear some space below.")
    elif d["pct"] >= 70:
        st.warning("Over 70% full — worth clearing what's below.")

# ---------------------------------------------------------------------------
# What's using it
# ---------------------------------------------------------------------------
hc = st.columns([4, 1])
hc[0].subheader("What's using the space")
if hc[1].button("🔄 Measure again", use_container_width=True):
    breakdown.clear()
    st.rerun()
rows = breakdown()
for label, path, size, note in rows:
    rc = st.columns([3, 1, 3])
    rc[0].markdown(f"**{label}**" + (f"  \n`{path}`" if path else ""))
    rc[1].markdown(SG.fmt_bytes(size))
    rc[2].progress(min(size / max(d["total"], 1), 1.0), text=note)

# ---------------------------------------------------------------------------
# Clean-up
# ---------------------------------------------------------------------------
st.subheader("Clear space")
allowed = SG.cleanup_allowed()


def ago(ts: float) -> str:
    days = (datetime.datetime.now().timestamp() - ts) / 86400
    return "today" if days < 1 else f"{days:.0f} day{'s' if days >= 1.5 else ''} ago"


# Downloaded originals
orig = SG.downloaded_originals()
o_size = sum(f["size"] for f in orig)
old = [f for f in orig if (datetime.datetime.now().timestamp() - f["last_used"]) / 86400 >= SG.KEEP_DOWNLOADS_DAYS]
with st.container(border=True):
    st.markdown(f"**🎞️ Downloaded original videos & music** — {len(orig)} file(s), {SG.fmt_bytes(o_size)}")
    st.caption("Full-size originals fetched from Google Drive for a render, a Choreography preview or audio "
               "analysis. Renders stream originals from Drive, and anything missing is downloaded again "
               "when it's needed — so these are temporary. "
               + (f"Files unused for {SG.KEEP_DOWNLOADS_DAYS:g} days are cleared automatically with the "
                  f"12-hourly Drive sync." if SG.KEEP_DOWNLOADS_DAYS > 0
                  else "Automatic clearing is off (KEEP_DOWNLOADS_DAYS = 0 in config.py)."))
    if orig:
        with st.expander(f"Show the {len(orig)} file(s) (least recently used first)"):
            for f in orig[:200]:
                st.markdown(f"- {'🎵' if f['kind'] == 'music' else '🎬'} `{f['path'].name}` · "
                            f"{SG.fmt_bytes(f['size'])} · used {ago(f['last_used'])}")
            if len(orig) > 200:
                st.caption(f"…and {len(orig) - 200} more.")
        bc = st.columns(2)
        if bc[0].button(f"🧹 Clear the {len(old)} unused for {SG.KEEP_DOWNLOADS_DAYS:g}+ days "
                        f"({SG.fmt_bytes(sum(f['size'] for f in old))})",
                        disabled=not (allowed and old), use_container_width=True):
            n, freed = SG.clear_originals(SG.KEEP_DOWNLOADS_DAYS)
            done(f"Cleared {n} original(s) — {SG.fmt_bytes(freed)} freed.")
        confirm = bc[1].checkbox("Also the ones used recently", key="srv_orig_all_ok",
                                 help="Safe — they'll stream from Drive, or download again if needed.")
        if confirm and bc[1].button(f"🧹 Clear all {len(orig)} ({SG.fmt_bytes(o_size)})", type="primary",
                                    disabled=not allowed, use_container_width=True):
            n, freed = SG.clear_originals(0)
            done(f"Cleared {n} original(s) — {SG.fmt_bytes(freed)} freed.")

# Renders already on Drive
renders = SG.local_renders_on_drive()
with st.container(border=True):
    r_size = sum(s for _, s in renders)
    st.markdown(f"**🎞️ Server copies of renders already on Google Drive** — {len(renders)} file(s), "
                f"{SG.fmt_bytes(r_size)}")
    st.caption("These play from Drive in Media Library → Renders; ratings, details and thumbnails stay.")
    if st.button(f"🧹 Clear {len(renders)} server copies", disabled=not (allowed and renders),
                 use_container_width=True):
        n, freed, errors = SG.clear_local_renders()
        for e in errors[:5]:
            st.error(e)
        done(f"Cleared {n} render copies — {SG.fmt_bytes(freed)} freed.")

# Temporary files
temps = SG.temp_files()
with st.container(border=True):
    t_size = sum(SG.du(p) or 0 for p in temps)
    st.markdown(f"**🗑️ Temporary files** — {len(temps)} item(s), {SG.fmt_bytes(t_size)}")
    st.caption("Reviewer frame grabs, half-finished downloads and leftover render folders. "
               "Cleared automatically after a day.")
    if st.button("🧹 Clear temporary files", disabled=not (allowed and temps), use_container_width=True):
        n, freed = SG.clear_temp(0)
        done(f"Cleared {n} temporary item(s) — {SG.fmt_bytes(freed)} freed.")

# System
with st.container(border=True):
    st.markdown("**⚙️ System**")
    sc = st.columns(2)
    journal = SG.du("/var/log/journal")
    if journal is not None:
        sc[0].caption(f"System logs: {SG.fmt_bytes(journal)}")
        if sc[0].button("Shrink system logs to 200 MB", disabled=not allowed or journal < 220 * 2**20,
                        use_container_width=True):
            done(f"System logs: {SG.shrink_journal(200)}")
    pip_cache = SG.du(SG.Path.home() / ".cache" / "pip")
    if pip_cache is not None:
        sc[1].caption(f"pip download cache: {SG.fmt_bytes(pip_cache)}")
        if sc[1].button("Clear pip's download cache", disabled=not allowed or pip_cache < 2**20,
                        use_container_width=True):
            done(f"pip cache cleared — {SG.fmt_bytes(SG.clear_pip_cache())} freed.")
