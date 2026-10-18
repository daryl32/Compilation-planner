#!/usr/bin/env python3
"""Real-data parity check: does the code in one folder cut and match EXACTLY
like the code in another, on your actual library?

    python3 deploy/check_parity.py [OLD_REPO] [NEW_REPO] [--data DIR] [--tracks N] [--videos N]

Defaults: OLD = /root/Compilation-planner (live), NEW = /root/Compilation-planner-dev
(test copy), data = /root/data-dev (the test copy's data — a copy of the live
library, so nothing live is touched). Writes nothing but caches in that data
folder and a report in DATA/logs/parity_report.txt.

Each repo is run in its own Python process: its planner page's functions are
loaded WITHOUT running the page (same trick as tests/support.py), then every
track is cut with all four methods, 🎯 Auto-detect is run per method, the
footage pool is built, and Auto matching runs on the first N tracks in four
configurations. The two results must be identical.

The Server page in the TEST copy has a button that runs this in the background.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

DEFAULT_OLD = "/root/Compilation-planner"
DEFAULT_NEW = "/root/Compilation-planner-dev"
DEFAULT_DATA = "/root/data-dev"


# ---------------------------------------------------------------------------
# Runner (one repo, one process)
# ---------------------------------------------------------------------------

def _install_config(data: Path) -> None:
    cfg = types.ModuleType("config")
    cfg.DRIVE_ROOT = data
    cfg.CATALOGUE_DIR = data / "catalogue"
    cfg.AUDIO_DIR = data / "audio_catalogue"
    cfg.PLANS_DIR = data / "compilation_plans"
    cfg.PREVIEW_DIR = data / "previews"
    cfg.PROJECTS_DIR = data / "projects"
    cfg.APP_ENV = "test"
    cfg.DRIVE_WRITES = False
    cfg.WHITELISTED_EMAILS = []
    cfg.CURVE_CACHE_DIR = data / ".cache" / "parity_curves"
    sys.modules["config"] = cfg


class _SS(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _deco(*a, **k):
    if len(a) == 1 and callable(a[0]) and not k:
        a[0].clear = lambda *x, **y: None
        return a[0]

    def w(fn):
        fn.clear = lambda *x, **y: None
        return fn
    return w


class _FakeSt(types.SimpleNamespace):
    def __init__(self):
        super().__init__(session_state=_SS(), cache_data=_deco, cache_resource=_deco, fragment=_deco,
                         experimental_fragment=_deco)

    def __getattr__(self, name):
        return lambda *a, **k: None


_CONST = re.compile(r"^_?[A-Z][A-Z0-9_]*$")


def _load_page(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    ns = {"__name__": "parity_page", "__file__": str(path)}

    def run(node):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Try)):
            run(node)
    ns["st"] = _FakeSt()
    ns["fragment"] = _deco
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            run(node)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names and all(_CONST.match(n) for n in names):
                try:
                    run(node)
                except Exception:
                    pass
    ns["st"] = ns["st"] if isinstance(ns["st"], _FakeSt) else _FakeSt()
    return ns


def _plain(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_plain(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 9)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run_repo(repo: Path, data: Path, n_tracks: int, n_videos: int) -> dict:
    _install_config(data)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    ns = _load_page(repo / "pages" / "1_Compilation_Planner.py")
    st = ns["st"]
    dial = ns["DIAL_DEFAULTS"]

    def reset(**extra):
        st.session_state.clear()
        st.session_state.update({k: copy.deepcopy(v) for k, v in dial.items()})
        st.session_state.update({"autofill_v2": True, "_avg_clips_per_block": 1.0, "_plan_story_in_use": False,
                                 "_plan_locked_roles": set(), "_plan_reserved_roles": set(),
                                 "video_time_ranges": {}, "clip_count_skip": []})
        st.session_state.update(extra)

    methods = [ns["SEG_SMART"], ns["SEG_ADAPTIVE"], ns["SEG_RHYTHM"], ns["SEG_SCORED"]]
    tracks = ns["list_tracks"]()
    out, timings = {}, {}
    t0 = time.perf_counter()
    for tid in tracks:
        reset()
        try:
            track = ns["load_track"](tid)
        except Exception as e:
            out[f"track/{tid}"] = {"error": type(e).__name__}
            continue
        for m in methods:
            cfg = {"method": m, "beats_per_bar": 4, **{k: dial.get(k) for k in ns["SEG_METHOD_KEYS"][m]}}
            try:
                segs = ns["build_segments"](track, cfg)
                ns["apply_segment_intensity"](segs, track, 1.0)
                out[f"seg/{tid}/{m}"] = segs
            except Exception as e:
                out[f"seg/{tid}/{m}"] = {"error": f"{type(e).__name__}: {e}"}
            try:
                out[f"autotune/{tid}/{m}"] = ns["auto_tune_parameters"](track, 1, 4, method=m,
                                                                       style_name="Balanced", beats_per_bar=4)
            except Exception as e:
                out[f"autotune/{tid}/{m}"] = {"error": f"{type(e).__name__}: {e}"}
    timings["cutting"] = time.perf_counter() - t0

    reset()
    videos = tuple(sorted(ns["load_all_catalogues"]()))[:n_videos]
    out["queues"] = ns["build_footage_queues"](videos, (), {})
    t0 = time.perf_counter()
    configs = {
        "default": ({}, {}),
        "split": ({}, {"min_clips": 1, "max_clips": 3}),
        "sequential": ({"_avg_clips_per_block": 1.5}, {"sequential": True, "max_clips": 2}),
        "v1": ({"autofill_v2": False}, {"max_clips": 2}),
    }
    for tid in tracks[:n_tracks]:
        for name, (state, kw) in configs.items():
            reset(**state)
            try:
                track = ns["load_track"](tid)
                cfg = {"method": ns["SEG_SMART"], "beats_per_bar": 4,
                       **{k: dial.get(k) for k in ns["SEG_METHOD_KEYS"][ns["SEG_SMART"]]}}
                segs = ns["build_segments"](track, cfg)
                ns["apply_segment_intensity"](segs, track, 1.0)
                q = ns["build_footage_queues"](videos, (), {})
                res = ns["match_scenes_to_track"](
                    segs, q, kw.get("min_clips", 1), kw.get("max_clips", 1), 0.2, 0.85, False,
                    kw.get("sequential", False), weight_shape=2.0, weight_random=0.5, avoid_repeat=True,
                    weight_spread=1.5, weight_motion=2.0, autofill_seed=42, weight_position=1.0,
                    repeat_fallbacks=set(), track=track, shape_matches=6)
                out[f"match/{tid}/{name}"] = res
            except Exception as e:
                out[f"match/{tid}/{name}"] = {"error": f"{type(e).__name__}: {e}"}
    timings["matching"] = time.perf_counter() - t0
    return {"results": _plain(out), "timings": timings, "n_tracks": len(tracks), "n_videos": len(videos)}


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def first_diff(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} → {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: {'only in new' if k not in a else 'only in old'}"
            d = first_diff(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} → {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} → {b!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old", nargs="?", default=DEFAULT_OLD)
    ap.add_argument("new", nargs="?", default=DEFAULT_NEW)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tracks", type=int, default=4, help="tracks to run Auto matching on")
    ap.add_argument("--videos", type=int, default=40, help="videos in the footage pool")
    ap.add_argument("--run", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args()
    data = Path(a.data)

    if a.run:                                   # child process: one repo
        res = run_repo(Path(a.run), data, a.tracks, a.videos)
        Path(a.out).write_text(json.dumps(res))
        return

    report_path = data / "logs" / "parity_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Parity check {time.strftime('%Y-%m-%d %H:%M')}",
             f"  old: {a.old}", f"  new: {a.new}", f"  data: {data}", ""]
    results = {}
    for label, repo in (("old", a.old), ("new", a.new)):
        out = data / "logs" / f"parity_{label}.json"
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, __file__, "--run", repo, "--out", str(out), "--data", str(data),
                            "--tracks", str(a.tracks), "--videos", str(a.videos)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            lines.append(f"❌ {label} code failed to run:\n{p.stderr[-3000:]}")
            report_path.write_text("\n".join(lines))
            print("\n".join(lines))
            sys.exit(1)
        results[label] = json.loads(out.read_text())
        t = results[label]["timings"]
        lines.append(f"  {label}: {time.perf_counter() - t0:.0f}s total (cutting {t['cutting']:.1f}s, "
                     f"matching {t['matching']:.1f}s)")
    old, new = results["old"]["results"], results["new"]["results"]
    lines.append(f"  {results['new']['n_tracks']} tracks, {results['new']['n_videos']} videos, "
                 f"{len(old)} cases")
    diffs = []
    for k in sorted(set(old) | set(new)):
        d = first_diff(old.get(k), new.get(k), k)
        if d:
            diffs.append(d)
    errs = [k for k, v in new.items() if isinstance(v, dict) and "error" in v]
    lines.append("")
    if diffs:
        lines.append(f"❌ {len(diffs)} case(s) differ. First ones:")
        lines += [f"   {d}" for d in diffs[:15]]
    else:
        lines.append(f"✅ Identical: all {len(old)} cases give exactly the same results.")
    if errs:
        lines.append(f"⚠️ {len(errs)} case(s) raised an error (in both copies if identical): "
                     + ", ".join(errs[:5]))
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
