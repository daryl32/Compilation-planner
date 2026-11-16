"""
🎬 Rendering a plan to a preview, and running planner work on the server — shared by 🎛️ Classic, 🤖 New matcher and
⏱️ Edge Mode (16 Nov 2026: three near-copies became this one).

- render(): the plan → a numbered preview in the renders folder (fetching any files it needs first when not
  streaming), named, uploaded to Google Drive when connected (cleared from the server after, if set so), with a
  message saying how long it took, step by step, where the files came from, and whether the picture and the sound
  came out as long as the plan.
- plan_json(): the plan as JSON — numpy numbers / flags turned into plain ones, so a download can't crash on them.
- start() / take_finished() / watch(): the work runs as a server job (mix_jobs) — it carries on when the phone or
  iPad leaves the page, shows its progress while the page is open, and a page opened later (or reloaded) picks up
  the result.

No page code runs inside a job: whatever a job needs from the session (settings, the Drive connection) is passed in
when its button is pressed.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# The plan as JSON
# ---------------------------------------------------------------------------

def _plain(x):
    """For json.dumps: numpy numbers and flags as plain ones (anything else as text)."""
    try:
        import numpy as np
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def plan_json(plan: dict, indent: int = 2) -> str:
    return json.dumps(plan, indent=indent, default=_plain)


# ---------------------------------------------------------------------------
# Files the render needs
# ---------------------------------------------------------------------------

def missing_sources(plan: dict, quick: bool = False, extra_raw=()) -> list:
    """(raw source, local path) of every file the plan reads that isn't on this machine: the videos (a video with a
    480p preview copy here isn't missing with ⚡ quick) and extra_raw (the track / songs)."""
    import library_common as LC
    from planner.data import load_all_catalogues
    from render_preview import colab_to_local
    cats = load_all_catalogues()
    needed = {r for r in extra_raw if r}
    for e in plan.get("timeline") or []:
        for slot in e.get("scenes") or []:
            for link in slot.get("chain") or []:
                if quick and LC.proxy_path(link["video_id"]).exists():
                    continue
                if link["video_id"] in cats:
                    needed.add(cats[link["video_id"]]["source_path"])
    return [(raw, Path(colab_to_local(raw))) for raw in sorted(needed) if not Path(colab_to_local(raw)).exists()]


def fetch_missing(missing: list, prog=None) -> list:
    """Download those files from Google Drive. prog(0–1, text). Returns error strings."""
    try:
        from drive_sync import credentials_available, download_source_video
        ok = credentials_available()
    except ImportError:
        ok = False
    errors = []
    for n, (raw, local) in enumerate(missing, 1):
        if not (ok and "MyDrive/" in raw):
            errors.append(f"{local.name}: not on the server and Drive download isn't available")
            continue
        if prog:
            prog(0.0, f"downloading {local.name} ({n}/{len(missing)})")
        err = download_source_video(raw.split("MyDrive/", 1)[1], local,
                                    progress_callback=(lambda p, _n=local.name, _i=n: prog(
                                        min(p, 1.0), f"downloading {_n} ({_i}/{len(missing)})")) if prog else None)
        if err:
            errors.append(f"{local.name}: {err}")
    return errors


# ---------------------------------------------------------------------------
# The render
# ---------------------------------------------------------------------------

def next_preview(track_id: str) -> Path:
    """The next numbered preview for this plan: temp-Video-preview-<name>.NNN.mp4 (numbering carries on past renders
    already cleared from the server)."""
    from config import PREVIEW_DIR
    from planner.state import sanitize_filename
    from renders import render_files
    safe = sanitize_filename(track_id)
    used = [int(m.group(1)) for p in render_files() if (m := re.search(r"\.(\d{3})\.mp4$", p.name))
            and fnmatch.fnmatch(p.name, f"temp-Video-preview-{safe}.*.mp4")]
    Path(PREVIEW_DIR).mkdir(parents=True, exist_ok=True)
    return Path(PREVIEW_DIR) / f"temp-Video-preview-{safe}.{(max(used) + 1) if used else 1:03d}.mp4"


def lengths(path) -> dict:
    """{"video", "audio"} seconds of a finished file (0 where it can't be read)."""
    out = {"video": 0.0, "audio": 0.0}
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=120)
        for st_ in json.loads(r.stdout or "{}").get("streams") or []:
            if st_.get("codec_type") in out:
                out[st_["codec_type"]] = float(st_.get("duration") or 0.0)
    except Exception:          # noqa: BLE001
        pass
    return out


def length_check(path, expected: float, tol: float = 0.5) -> str:
    """"ok", or what came out shorter / longer than the plan (picture and sound against `expected` seconds)."""
    got = lengths(path)
    off = [f"{w} {got[k]:.1f} s" for k, w in (("video", "picture"), ("audio", "sound"))
           if got[k] > 0 and abs(got[k] - expected) > tol]
    return "ok" if not off else f"expected {expected:.1f} s — " + ", ".join(off)


STEP_NAMES = (("segments", "clips"), ("edge_sound", "sound"), ("edge_screens", "intro / outro"),
              ("edge_picture", "timer and avatar"), ("edge_join", "joining"))


def render(plan: dict, *, stream: bool = True, quick: bool = False, label: str = "", token=None, prog=None,
           extra_meta: dict = None, extra_raw=()) -> tuple:
    """The plan → a preview. Returns (message for the page: (kind, text), file name). Raises with a plain reason when
    it can't (files missing and not downloadable, ffmpeg failing). prog(0–1, text); token: the Drive connection
    (None: kept on the server, the message says how to save it)."""
    from page_setup import APP_VERSION
    from render_preview import render_plan_dict
    from renders import auto_clear_enabled, clear_local, update_user_fields, upload_render
    prog = prog or (lambda f, t: None)
    if not stream:
        missing = missing_sources(plan, quick, extra_raw)
        if missing:
            errs = fetch_missing(missing, prog)
            if errs:
                raise RuntimeError("some source files are missing and couldn't be downloaded: " + " · ".join(errs[:4]))
    out = next_preview(plan["track_id"])
    last, t0 = [0.0], time.time()

    def _seg(d, t):
        last[0] = d / max(t, 1)
        secs = time.time() - t0
        prog(last[0], f"rendering segment {d}/{t}" + (f" · about {int((secs / d) * (t - d))} s left" if d else ""))
    stats = render_plan_dict(plan, str(out), progress_callback=_seg, stream=stream, quick=quick,
                             notice=lambda txt: prog(last[0], txt), app_version=APP_VERSION,
                             extra_meta=extra_meta or {}) or {}
    label = " ".join(str(label or "").split())[:80]
    if label:                                              # 🏷️ kept with the render's rating and notes
        update_user_fields(out, name=label)
    stg = stats.get("stages") or {}
    check = stg.get("edge_check")                          # (Edge checks its own: intro / outro screens included)
    if check is None and Path(out).exists():
        check = length_check(out, float(plan.get("track_duration_sec") or 0.0)) if plan.get("track_duration_sec") else None
    secs = time.time() - t0
    took = f"{int(secs // 60)}:{int(secs % 60):02d}"
    note = ("info", f"Rendered {out.name} in {took}. Connect Google Drive (sidebar) to save renders there — then upload "
                    "it from 📁 Media Library → Renders.")
    if token:
        prog(1.0, "uploading to Google Drive")
        err = upload_render(out, token)
        if err:
            note = ("warning", f"Rendered {out.name}, but the Drive upload failed: {err} — upload it later from 📁 Media "
                               "Library → Renders.")
        elif auto_clear_enabled():
            clear_local(out)
            note = ("success", f"Rendered {out.name} in {took}, saved to Google Drive and cleared from the server.")
        else:
            note = ("success", f"Rendered {out.name} in {took} and saved to Google Drive.")
    how = [f"{stats[k]} {w}" for k, w in (("quick", "from the 480p copies"), ("streamed", "streamed from Drive"),
                                         ("downloaded", "downloaded"), ("local", "already on the server")) if stats.get(k)]
    if how:
        note = (note[0], note[1] + " Source files: " + ", ".join(how) + ".")
    if stg.get("ffmpeg"):                                  # which ffmpeg made it (the server's differs from the test machine's)
        note = (note[0], note[1] + " " + " ".join(str(stg["ffmpeg"]).split()[:3]) + ".")
    steps = [f"{w} {int(stg[k] // 60)}:{int(stg[k] % 60):02d}" for k, w in STEP_NAMES if stg.get(k)]
    if steps:
        note = (note[0], note[1] + " Time: " + ", ".join(steps) + ".")
    if stg.get("edge_notes"):                              # ⚡ what the Mix engine had to say (e.g. no stems for a song)
        note = (note[0], note[1] + " ⚡ " + " · ".join(stg["edge_notes"][:3]))
    if check not in (None, "ok"):                          # the picture or the sound isn't as long as the plan
        note = ("warning", note[1] + f" ⚠️ Length check: {check}.")
    return note, out.name


# ---------------------------------------------------------------------------
# Running it on the server
# ---------------------------------------------------------------------------

def start(key: str, label: str, fn, *args, **kwargs):
    """Start fn(job, *args, **kwargs) as a server job under `key` — a finished one under that key is replaced; a running
    one is left alone (returns None then)."""
    import mix_jobs as J
    running = J.get(key)
    if running is not None and running.active:
        return None
    J.forget(key)
    return J.start(key, label, fn, *args, **kwargs)


def take_finished(key: str, seen_key: str):
    """The finished job under `key` the first time this session sees it (else None) — also in a new session after the
    page was reloaded. Second value: True while one is still running."""
    import mix_jobs as J
    j = J.get(key)
    if j is None:
        return None, False
    if j.active:
        return None, True
    if st.session_state.get(seen_key) == j.started:
        return None, False
    st.session_state[seen_key] = j.started
    return j, False


import streamlit as st  # noqa: E402  (the page helpers below; nothing above needs it)


@st.fragment(run_every=1.0)
def watch(key: str) -> None:
    """⏳ The running job's progress, refreshed every second; the page reruns when it is done."""
    import mix_jobs as J
    j = J.get(key)
    if j is None or not j.active:
        st.rerun(scope="app")
    secs = int(time.time() - j.started)
    st.progress(min(1.0, j.frac), text=f"{j.label}… {j.msg or ''} · {secs // 60}:{secs % 60:02d}")
    st.caption("⏳ This runs on the server — you can leave the page or switch apps and come back; the result will be here.")
