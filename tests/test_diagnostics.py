"""🩺 diagnostics.py against a fake /proc and fake logs."""

import datetime

import tests.support  # noqa: F401  (test config)
import diagnostics as DG


def _fake_proc(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("MemTotal:  4000000 kB\nMemFree: 200000 kB\nMemAvailable: 1000000 kB\n"
                                  "SwapTotal: 4194304 kB\nSwapFree: 3145728 kB\n")
    (proc / "loadavg").write_text("1.50 2.40 1.10 2/300 999\n")

    def pid(n, name, rss_kb, cmd):
        d = proc / str(n)
        d.mkdir()
        (d / "status").write_text(f"Name:\t{name}\nVmRSS:\t{rss_kb} kB\n")
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())

    pid(10, "streamlit", 900_000, "/root/Compilation-planner/venv/bin/python streamlit run /root/Compilation-planner/Home.py")
    pid(11, "streamlit", 400_000, "/root/Compilation-planner-dev/venv/bin/python streamlit run /root/Compilation-planner-dev/Home.py")
    pid(12, "python3", 60_000, "python3 /root/webhook.py")
    pid(13, "ffmpeg", 300_000, "ffmpeg -i x.mp4")
    pid(14, "tiny", 1_000, "tiny")
    pid(15, "nginx", 80_000, "nginx: worker")
    return proc


def test_system_memory_and_cpu(tmp_path):
    proc = _fake_proc(tmp_path)
    m = DG.system_memory(proc)
    assert round(m["total"]) == 3906 and round(m["available"]) == 977
    assert round(m["used"]) == 2930 and round(m["swap_used"]) == 1024
    c = DG.cpu(proc)
    assert c["load5"] == 2.4


def test_processes_grouped(tmp_path):
    rows = DG.app_processes(_fake_proc(tmp_path))
    labels = [r[0] for r in rows]
    assert labels[0] == "🟢 Live app"
    assert "🧪 Test copy" in labels and "🔗 Deploy webhook" in labels and "🎬 ffmpeg (render / proxy)" in labels
    assert "⚙️ nginx" in labels and not any("tiny" in l for l in labels)


def test_off_linux_returns_none(tmp_path):
    assert DG.system_memory(tmp_path / "nope") is None
    assert DG.cpu(tmp_path / "nope") is None
    assert DG.app_processes(tmp_path / "nope") == []


def test_timings_roundtrip(tmp_path):
    log = tmp_path / "t.csv"
    for s in (0.1, 0.2, 0.3, 0.4, 5.0):
        DG.record("Auto matching", s, "x", path=log)
    DG.record("Cutting the track", 0.05, path=log)
    summ = DG.timing_summary(7, path=log)
    assert summ[0]["step"] == "Auto matching"
    assert summ[0]["runs"] == 5 and summ[0]["median"] == 0.3 and summ[0]["max"] == 5.0 and summ[0]["last"] == 5.0


def test_timed_records_even_on_error(tmp_path, monkeypatch):
    log = tmp_path / "t.csv"
    monkeypatch.setattr(DG, "TIMINGS_LOG", log)
    try:
        with DG.timed("Boom"):
            raise RuntimeError
    except RuntimeError:
        pass
    assert DG.timing_summary(7)[0]["step"] == "Boom"


def test_record_never_raises(tmp_path):
    DG.record("x", 1.0, path=tmp_path / "missing_dir_is_fine" / "t.csv")
    DG.record("x", 1.0, path=tmp_path)            # a directory — must not raise


def test_memory_history(tmp_path):
    log = tmp_path / "memory.csv"
    today = datetime.datetime.now()
    old = today - datetime.timedelta(days=30)
    log.write_text("time,where,memory_mb,note\n"
                   f"{old.isoformat(timespec='seconds')},planner,3000,\n"
                   f"{today.isoformat(timespec='seconds')},planner,900,\n"
                   f"{today.isoformat(timespec='seconds')},planner,1900,over 1800 MB: cleared caches and 2 held\n"
                   f"{today.isoformat(timespec='seconds')},planner,1200,\n")
    h = DG.memory_history(7, path=log)
    assert len(h) == 1 and h[0]["peak_mb"] == 1900 and h[0]["median_mb"] == 1200 and h[0]["clears"] == 1


def test_verdict():
    mem = {"total": 4000, "used": 3700, "available": 300, "swap_total": 4096, "swap_used": 1500}
    v = DG.verdict(mem, {"cores": 2, "load1": 0.2, "load5": 0.3, "load15": 0.2}, [], 50)
    assert v[0][0] == "warn" and "RAM is the limit" in v[0][1]
    v = DG.verdict({**mem, "used": 1000, "swap_used": 0}, {"cores": 2, "load1": 3, "load5": 3, "load15": 3}, [], 90)
    kinds = [k for k, _ in v]
    assert kinds.count("warn") == 2 and any("CPU is overloaded" in t for _, t in v)
