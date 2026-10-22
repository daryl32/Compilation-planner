"""
🔮 Recommendations — the library side, shared by the Recommendations page and the
background worker (jobs.py). No Streamlit widgets here (catalogue_store and
library_common import Streamlit, but only for their caches, which also work in a
plain Python process).

  • the library's data for core/recommend.py: fingerprints (cached on disk, so a
    nightly run doesn't re-read every video), role footage, actors, "find similar",
    story maps, and the real matcher as the planner's Auto mode runs it;
  • run_track(): search one track and store the five plans (+ what the feeds need);
  • stored results, the "has anything changed?" signature, and the 🆕 spotlight.
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import catalogue_store as CStore
import core.footage as CFOOT
import core.matching as CMATCH
import core.recommend as REC
import curve_cache
import scene_roles as SR
import similarity as SIM
import story_map as STM
from config import AUDIO_DIR, CATALOGUE_DIR
from track_files import track_file
from library_common import library_ranges, video_details

# One folder per copy of the app (live / test copy): the curve cache folder is already
# per data folder, so this sits next to it under the same name.
_CURVES = curve_cache.default_cache_dir(CATALOGUE_DIR)
BASE_DIR = _CURVES.parent / f"recs-{_CURVES.name}"
RESULTS_DIR = BASE_DIR / "results"
PRINTS_DIR = BASE_DIR / "prints"
SPOTLIGHT_DIR = RESULTS_DIR / "spotlight"
SEEN_FILE = RESULTS_DIR / "seen_videos.json"
CHECKS_DIR = BASE_DIR / "checks"         # ⚡ matcher runs kept from the last search of each track
HASHES_FILE = BASE_DIR / "file_hashes.json"
DEFAULT_VIDEOS = 4
RESULTS_VERSION = 6          # bump when what's stored changes (older results are searched again)
PRINTS_VERSION = 1           # bump when fingerprints change
NEW_DAYS = 14                # a video counts as new (🆕 spotlight) for this many days
SPOTLIGHT_TRACKS = 3         # ready-made plans per new video


# ---------------------------------------------------------------------------
# Library data
# ---------------------------------------------------------------------------

def list_tracks() -> list:
    return sorted(f.stem for f in AUDIO_DIR.glob("*.json") if f.name != "audio_index.json")


def load_track(track_id: str) -> dict:
    return json.loads(track_file(track_id, AUDIO_DIR).read_text())


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


_PRINTS_MEMO: dict = {}      # (video_id, mtime, range) → VideoPrint, this process


def _print_file(video_id: str) -> Path:
    return PRINTS_DIR / f"{hashlib.sha1(video_id.encode()).hexdigest()[:16]}.npz"


def _load_print(video_id: str, key: str):
    try:
        with np.load(_print_file(video_id)) as z:
            if str(z["key"]) != key:
                return None
            vp = REC.VideoPrint(video_id, scale=float(z["scale"]))
            for L in z["lengths"]:
                vp.by_len[float(L)] = (z[f"z_{L:g}"], z[f"l_{L:g}"])
            return vp
    except Exception:
        return None


def _save_print(vp, key: str) -> None:
    try:
        PRINTS_DIR.mkdir(parents=True, exist_ok=True)
        arrays = {"key": np.array(key), "scale": np.array(vp.scale), "lengths": np.array(sorted(vp.by_len))}
        for L, (z, lv) in vp.by_len.items():
            arrays[f"z_{L:g}"], arrays[f"l_{L:g}"] = z, lv
        tmp = _print_file(vp.video_id).with_suffix(f".{os.getpid()}.tmp.npz")   # several searches may write at once
        np.savez(tmp, **arrays)
        tmp.replace(_print_file(vp.video_id))
    except OSError:
        pass


def fingerprint(video_id: str, mtime: float, rng: tuple):
    """A video's fingerprint — from memory, else the disk cache, else worked out (and cached)."""
    memo_key = (video_id, mtime, rng)
    if memo_key in _PRINTS_MEMO:
        return _PRINTS_MEMO[memo_key]
    key = json.dumps([PRINTS_VERSION, video_id, mtime, list(rng)])
    vp = _load_print(video_id, key)
    if vp is None:
        cat = CStore.all_catalogues().get(video_id)
        if not cat:
            return None
        vp = REC.fingerprint_video(video_id, cat, CStore.motion_curves(video_id), rng or None)
        if vp is not None:
            _save_print(vp, key)
    if len(_PRINTS_MEMO) > 400:
        _PRINTS_MEMO.clear()
    _PRINTS_MEMO[memo_key] = vp
    return vp


def _stems() -> dict:
    """{video_id: catalogue file stem}."""
    return {slim["video_id"]: stem for stem, slim in CStore.video_index().items()}


def library_prints(progress=None) -> dict:
    cats = CStore.all_catalogues()
    ranges = library_ranges()
    stems = _stems()
    out = {}
    ids = sorted(cats)
    for i, vid in enumerate(ids):
        if progress:
            progress(i / max(len(ids), 1), f"Reading video {i + 1} of {len(ids)}…")
        vp = fingerprint(vid, _mtime(CStore.catalogue_path(stems.get(vid, vid))), tuple(ranges.get(vid) or ()))
        if vp is not None:
            out[vid] = vp
    return out


def _thumb_for(stored, vid):
    return CATALOGUE_DIR / "thumbnails" / vid / Path(stored).name


# ⚡ Library-wide data that's the same for every track (role footage, "find similar", scene
# roles, durations): worked out once and kept while the library is unchanged — the nightly
# worker reads them before it starts its searches, so every search process inherits them.
_LIB_MEMO: dict = {}


def _lib_key():
    """Changes whenever a catalogue or the library file (ranges, actors) changes."""
    import library_common as LC
    try:
        meta = LC.LIBRARY_META_FILE.stat().st_mtime_ns
    except (OSError, AttributeError):
        meta = None
    return CStore.all_catalogues(), meta     # the catalogues dict is replaced whenever one changes


def _lib_memo(name: str, fn):
    key = _lib_key()
    got = _LIB_MEMO.get(name)
    if got is None or got[0][0] is not key[0] or got[0][1] != key[1]:
        got = _LIB_MEMO[name] = (key, fn())
    return got[1]


def warm_library_data() -> None:
    """Work out the shared library data now (before forking search processes)."""
    role_seconds()
    scene_roles_lookup()
    similar_finder()
    _durations()


def role_seconds() -> dict:
    """{video_id: {role: seconds of usable footage marked with it}} (library ranges applied)."""
    def build():
        cats = CStore.all_catalogues()
        q = CFOOT.build_footage_queues(cats, tuple(cats), (), library_ranges(), thumbnail_for=_thumb_for,
                                       roles_of=SR.scene_roles)
        return {vid: CFOOT.role_supply({vid: spans}) for vid, spans in q.items()}
    return _lib_memo("role_seconds", build)


def scene_roles_lookup():
    """roles_of(video_id, scene_id) → the scene's role ids (for the Story score)."""
    def build():
        table = {(vid, sc["scene_id"]): SR.scene_roles(sc)
                 for vid, cat in CStore.all_catalogues().items() for sc in cat.get("scenes", [])}
        return lambda vid, sid: table.get((vid, sid), ())
    return _lib_memo("scene_roles", build)


def actors_by_video() -> dict:
    return {vid: d["actors"] for vid, d in video_details().items() if d.get("actors")}


def similar_finder():
    """similar_of(video_id) → [(similarity, reason, video_id)] across the library, best first —
    the planner's ✨ Similar videos ("find similar") scoring."""
    def build():
        cats = CStore.all_catalogues()
        videos = SIM.library_video_inputs(cats, library_ranges(), CATALOGUE_DIR)
        idf = SIM.tag_idf(cats, lambda sc: sc.get("tags", []))
        memo = {}

        def similar_of(vid):
            if vid not in memo:
                memo[vid] = SIM.rank_similar_videos(vid, videos, idf, top=len(videos))
            return memo[vid]
        return similar_of
    return _lib_memo("similar", build)


def story_for(track_id: str, track: dict) -> tuple:
    """(regions or None, "saved" | "suggested" | None) — the track's saved story map, else the
    one the story map wizard would suggest for it."""
    saved = STM.saved_map(track_id)
    if saved:
        return saved, "saved"
    try:
        regions = STM.suggest(STM.analyse(track, 4), SR.all_roles())
    except Exception:          # a track the analysis can't read — no story parts
        return None, None
    return (regions, "suggested") if any(r.get("role") for r in regions) else (None, None)


def _durations() -> dict:
    # Same durations as the planner's get_video_duration (the last scene's end).
    return _lib_memo("durations", lambda: {
        vid: CFOOT._tc_to_seconds(cat["scenes"][-1]["end_tc"]) if cat.get("scenes") else 0.0
        for vid, cat in CStore.all_catalogues().items()})


def plan_context(block_roles=None, *, durations: dict = None, curves_of=None,
                 shape_memo: dict = None) -> CMATCH.PlanContext:
    durations = _durations() if durations is None else durations
    # 🎭 With the story map in use, the planner reserves and locks the roles it uses
    # (its defaults: 'Keep role clips for their parts' on, Intro/Outro preference off).
    used = frozenset(r for r in (block_roles or []) if r)
    return CMATCH.PlanContext(time_ranges=library_ranges, duration_of=lambda v: durations.get(v, 0.0),
                              curves_of=curves_of or CStore.motion_curves, catalogue_dir=CATALOGUE_DIR,
                              reserved_roles=used, locked_roles=used, story_in_use=bool(used),
                              planned_sequential=True, shape_memo=shape_memo)


def search_curves():
    """⚡ curves_of for one search: each video's motion curves fetched once and kept for the
    search (they're memory-mapped, so this costs little) — the shared cache's own lookup
    is asked ~16,000 times in a search and only holds 20 videos."""
    memo = {}

    def curves_of(video_id):
        got = memo.get(video_id)
        if got is None:
            got = memo[video_id] = CStore.motion_curves(video_id)
        return got
    return curves_of


def make_matcher(track: dict, curves_of=None, *, track_id: str = None, kept: dict = None, record: dict = None):
    """The real matcher as Auto mode runs it with the planner's default settings
    (one clip per block, ✨ improved auto-fill, never the same video twice in a row),
    with or without the story map and in sequential order when asked.
    ⚡ One matcher serves one search: the library is read once, and shape-window scores
    are kept (PlanContext.shape_memo) — the search asks about the same footage again and
    again, and the answers can't change within it.
    kept / record (with track_id): matcher runs from this track's last search to reuse, and
    a dict that collects this search's runs to keep (see "Matcher runs kept between searches")."""
    cats = CStore.all_catalogues()
    ranges = library_ranges()
    durations = _durations()
    curves_of = curves_of or search_curves()
    shape_memo = {}

    if record is not None:
        code, track_h, vsigs = matcher_code_sig(), track_hash(track_id), video_sigs()
    kept = kept or {}

    def match(segments, video_ids, block_roles=None, sequential=False):
        if record is None:
            return run(segments, video_ids, block_roles, sequential)
        key = _run_key(code, track_h, segments, video_ids, block_roles, sequential, vsigs)
        rows = kept.get(key)
        if rows is None:
            rows = slim_timeline(run(segments, video_ids, block_roles, sequential))
        record[key] = rows
        return unslim_timeline(rows)

    def run(segments, video_ids, block_roles=None, sequential=False):
        q = CFOOT.build_footage_queues(cats, tuple(video_ids), (), ranges, thumbnail_for=_thumb_for,
                                       roles_of=SR.scene_roles)
        roles = list(block_roles) if block_roles else [None] * len(segments)
        # Exactly the planner's Auto call with its defaults (split screen off → 1 clip a block,
        # Balanced weights; F (position) only in sequential mode) — tests/test_recommend.py
        # checks they agree.
        timeline, _, _ = CMATCH.match_scenes_to_track(
            [dict(s) for s in segments], q, 1, 1, 0.0, 1.0, False, bool(sequential), {}, {}, set(), False, 0, 1.0,
            set(), set(), 2.0, 0.5, True, 1.5, 2.0, 42, weight_position=1.0 if sequential else 0.0,
            repeat_fallbacks=set(), block_roles=roles, track=track,
            ctx=plan_context(block_roles, durations=durations, curves_of=curves_of, shape_memo=shape_memo))
        return timeline
    return match


# ---------------------------------------------------------------------------
# "Has anything changed?" and stored results
# ---------------------------------------------------------------------------

# ⚡ What a search depends on is judged by CONTENT, not file times: a Drive sync that
# brings down an unchanged file, or a copy that keeps the same bytes, no longer makes
# every track look out of date. Hashes are kept per file (by size + modified time), so
# a file is only read again when it might have changed.

_HASH_MEMO: dict = {}        # path → (size, mtime, sha1), this process


def _load_hashes() -> dict:
    if not _HASH_MEMO:
        try:
            for k, v in json.loads(HASHES_FILE.read_text()).items():
                _HASH_MEMO[k] = tuple(v)
        except (OSError, ValueError, TypeError):
            pass
    return _HASH_MEMO


def file_hash(p: Path) -> str:
    """sha1 of a file's bytes ("" if it can't be read), remembered while its size and time stay put."""
    memo = _load_hashes()
    try:
        st = p.stat()
    except OSError:
        return ""
    got = memo.get(str(p))
    if got and got[0] == st.st_size and got[1] == st.st_mtime:
        return got[2]
    try:
        h = hashlib.sha1(p.read_bytes()).hexdigest()
    except OSError:
        return ""
    memo[str(p)] = (st.st_size, st.st_mtime, h)
    return h


def save_hashes() -> None:
    """Keep the hashes for next time (only files that still exist)."""
    try:
        keep = {k: v for k, v in _HASH_MEMO.items() if Path(k).exists()}
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = HASHES_FILE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(keep))
        tmp.replace(HASHES_FILE)
    except OSError:
        pass


def video_sigs() -> dict:
    """{video_id: what the matcher sees of that video} — its catalogue's content (scenes,
    roles, tags, motion curves) and its library range. A video whose sig is unchanged
    gives the matcher exactly the same footage."""
    ranges = library_ranges()
    out = {}
    for vid, stem in sorted(_stems().items()):
        h = file_hash(CStore.catalogue_path(stem))
        out[vid] = hashlib.sha1(json.dumps([h, ranges.get(vid)], default=str).encode()).hexdigest()[:16]
    return out


def track_hash(track_id: str) -> str:
    return file_hash(track_file(track_id, AUDIO_DIR))


def library_state() -> list:
    """Everything about the library a search depends on (not the track)."""
    return [sorted(video_sigs().items()), sorted(library_ranges().items()), sorted(actors_by_video().items())]


def signature(track_id: str, n_videos: int, sequential: bool, story, state: list = None) -> str:
    return hashlib.sha1(json.dumps([RESULTS_VERSION, track_id, n_videos, bool(sequential),
                                    track_hash(track_id), state or library_state(),
                                    REC.SEARCH_VERSION, REC.DROP_WEIGHT, story],
                                   default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ⚡ Matcher runs kept between searches
# ---------------------------------------------------------------------------
# A search runs the real matcher ~25 times (a cut of the track + a set of videos each).
# The matcher only looks at the videos it's given, so when the same cut meets the same
# videos — none of them changed — it gives the same picks. Each search keeps its runs
# (CHECKS_DIR, gzipped, beside the results) and the next search of that track reuses
# every one whose cut, story roles, order and videos are all unchanged; only runs that
# touch a new or changed video (or a new combination) go through the matcher again.
# The kept runs are also tied to the matcher's code: any change to it starts afresh.

_MATCHER_FILES = ("core/*.py", "autofill.py", "scene_roles.py", "catalogue_store.py", "library_common.py",
                  "curve_cache.py", "video_ranking.py", "rec_library.py")
_CODE_SIG: list = []


def matcher_code_sig() -> str:
    if not _CODE_SIG:
        here = Path(__file__).resolve().parent
        h = hashlib.sha1()
        for pattern in _MATCHER_FILES:
            for f in sorted(here.glob(pattern)):
                h.update(f.name.encode())
                h.update(f.read_bytes())
        _CODE_SIG.append(h.hexdigest()[:16])
    return _CODE_SIG[0]


def _checks_path(track_id: str, n_videos: int, sequential: bool) -> Path:
    return CHECKS_DIR / (_results_path(track_id, n_videos, sequential).stem + ".json.gz")


def load_checks(track_id: str, n_videos: int, sequential: bool) -> dict:
    """{run key: slim timeline} kept from this track's last search ({} if none or from older code)."""
    try:
        with gzip.open(_checks_path(track_id, n_videos, sequential), "rt") as f:
            data = json.load(f)
    except (OSError, ValueError, EOFError):
        return {}
    if data.get("code") != matcher_code_sig() or data.get("version") != RESULTS_VERSION:
        return {}
    return data.get("runs") or {}


def save_checks(track_id: str, n_videos: int, sequential: bool, runs: dict) -> None:
    try:
        CHECKS_DIR.mkdir(parents=True, exist_ok=True)
        p = _checks_path(track_id, n_videos, sequential)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        with gzip.open(tmp, "wt", compresslevel=6) as f:
            json.dump({"version": RESULTS_VERSION, "code": matcher_code_sig(), "runs": runs}, f)
        tmp.replace(p)
    except OSError:
        pass


def _plain(x):
    return x.item() if isinstance(x, np.generic) else x


def slim_timeline(timeline: list) -> list:
    """What's used of a matcher run (score_timeline + slim_plan): per block, per clip,
    [video, scene, clip start, first link's video, scene, offset into scene, duration]."""
    out = []
    for block in timeline or []:
        row = []
        for c in block.get("scenes") or []:
            link = (c.get("chain") or [None])[0]
            row.append([_plain(x) for x in [c.get("video_id"), c.get("scene_id"), c.get("clip_start_sec")]
                        + ([link.get("video_id"), link.get("scene_id"), link.get("offset_into_scene_sec"),
                            link.get("clip_duration_sec")] if link else [])])
        out.append(row)
    return out


def unslim_timeline(rows: list) -> list:
    out = []
    for row in rows:
        scenes = []
        for c in row:
            slot = {"video_id": c[0], "scene_id": c[1], "clip_start_sec": c[2]}
            if len(c) > 3:
                slot["chain"] = [{"video_id": c[3], "scene_id": c[4], "offset_into_scene_sec": c[5],
                                  "clip_duration_sec": c[6]}]
            scenes.append(slot)
        out.append({"scenes": scenes})
    return out


def _run_key(code: str, track_h: str, segments, video_ids, block_roles, sequential, vsigs: dict) -> str:
    return hashlib.sha1(json.dumps([code, track_h, segments, list(video_ids), block_roles, bool(sequential),
                                    [vsigs.get(v) for v in video_ids]], default=str).encode()).hexdigest()[:20]


def _results_path(track_id: str, n_videos: int, sequential: bool) -> Path:
    key = f"{track_id}|{int(n_videos)}|{'seq' if sequential else 'normal'}"
    return RESULTS_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def load_results(track_id: str, n_videos: int, sequential: bool, sig: str = None):
    """The stored results for this track and settings — only if they match `sig` (when given)."""
    try:
        data = json.loads(_results_path(track_id, n_videos, sequential).read_text())
    except (OSError, ValueError):
        return None
    if data.get("version") != RESULTS_VERSION:
        return None
    return data if sig is None or data.get("sig") == sig else None


def all_results(n_videos: int = DEFAULT_VIDEOS, sequential: bool = False) -> dict:
    """{track_id: stored results} for every searched track (the feeds' input), fresh or not."""
    out = {}
    tracks = set(list_tracks())
    for f in RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if (isinstance(data, dict) and data.get("version") == RESULTS_VERSION and data.get("track_id") in tracks
                and data.get("n_videos") == n_videos and bool(data.get("sequential")) == bool(sequential)):
            out[data["track_id"]] = data
    return out


def slim_plan(plan: dict, story_regions) -> dict:
    """What's kept of a plan: no segments or full timeline, just each block's picks
    (video, scene, start) — enough to show it and to check the planner rebuilds it."""
    out = {k: plan.get(k) for k in ("mode", "variant", "style", "pace", "method", "settings", "videos", "story",
                                    "sequential", "scores", "estimate", "actors", "featured")}
    out["story_regions"] = story_regions if plan.get("story") else None
    out["picks"] = [[[c.get("video_id"), c.get("scene_id"), round(float(c.get("clip_start_sec") or 0), 3)]
                     for c in block.get("scenes") or []] for block in plan.get("timeline") or []]
    return out


def slim_results(results: list, story_regions) -> list:
    return [{"mode": r["mode"], "note": r.get("note", ""),
             "plan": slim_plan(r["plan"], story_regions) if r.get("plan") else None} for r in results]


def save_results(data: dict) -> None:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        p = _results_path(data["track_id"], data["n_videos"], data["sequential"])
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)
    except OSError:
        pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def run_track(track_id: str, n_videos: int = DEFAULT_VIDEOS, sequential: bool = False, progress=None,
              seed: int = None, state: list = None) -> dict:
    """Search one track (the five plans) and store the results. Returns what was stored.
    progress(fraction, text)."""
    progress = progress or (lambda f, t: None)
    track = load_track(track_id)
    story_regions, story_source = story_for(track_id, track)
    sig = signature(track_id, n_videos, sequential, story_regions, state)
    prints = library_prints(lambda f, t: progress(0.2 * f, t))
    if len(prints) < 2:
        raise RuntimeError("Recommendations need at least two videos with motion data.")
    seed = int(sig, 16) % 2**31 if seed is None else int(seed)
    insights = {}
    curves_of = search_curves()
    kept, runs = load_checks(track_id, n_videos, sequential), {}
    results = REC.recommend_modes(
        track, prints, match=make_matcher(track, curves_of, track_id=track_id, kept=kept, record=runs),
        curves_of=curves_of,
        n_videos=min(n_videos, len(prints)), story_regions=story_regions, role_seconds=role_seconds(),
        roles_of=scene_roles_lookup(), actors=actors_by_video(), similar_of=similar_finder(),
        sequential=sequential, wildcard_seed=seed, insights=insights,
        progress=lambda f, t: progress(0.2 + 0.8 * min(f, 1.0), t))
    data = {"version": RESULTS_VERSION, "sig": sig, "track_id": track_id, "n_videos": int(n_videos),
            "sequential": bool(sequential), "seed": seed, "story_source": story_source, "finished_at": _now(),
            "results": slim_results(results, story_regions), "insights": insights}
    save_results(data)
    save_checks(track_id, n_videos, sequential, runs)
    save_hashes()
    return data


def reroll_wildcard(data: dict, progress=None) -> dict:
    """🎲 Roll again on stored results: a new wildcard (seed + 1), saved."""
    track = load_track(data["track_id"])
    story_regions, _ = story_for(data["track_id"], track)
    prints = library_prints()
    seed = int(data.get("seed", 0)) + 1
    curves_of = search_curves()
    res = REC.reroll_wildcard(track, prints, match=make_matcher(track, curves_of), curves_of=curves_of,
                              n_videos=min(data["n_videos"], len(prints)), story_regions=story_regions,
                              roles_of=scene_roles_lookup(), sequential=data["sequential"], seed=seed)
    new = slim_results([res], story_regions)[0]
    data = dict(data, seed=seed, results=[new if r["mode"] == "wildcard" else r for r in data["results"]])
    save_results(data)
    return data


# ---------------------------------------------------------------------------
# 🆕 New videos
# ---------------------------------------------------------------------------

def update_seen(now: datetime.datetime = None) -> dict:
    """{video_id: first seen (ISO)}. The very first time, every video already in the library
    counts as old (seen at the epoch), so the spotlight starts with what's added afterwards."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    try:
        seen = json.loads(SEEN_FILE.read_text())
        first_time = False
    except (OSError, ValueError):
        seen, first_time = {}, True
    changed = False
    for vid in CStore.all_catalogues():
        if vid not in seen:
            seen[vid] = "1970-01-01T00:00:00+00:00" if first_time else now.isoformat(timespec="seconds")
            changed = True
    if changed or first_time:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            SEEN_FILE.write_text(json.dumps(seen, indent=1, sort_keys=True))
        except OSError:
            pass
    return seen


def new_videos(seen: dict = None, now: datetime.datetime = None) -> list:
    """Videos added in the last NEW_DAYS days, newest first: [(video_id, added ISO)]. The added date is
    when the video arrived on Google Drive (the same date the planner's 🆕 badge uses); the first-seen
    record is only the fallback for videos the Drive dates don't cover."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    added = CStore.added_times()
    if added:
        cutoff = now.timestamp() - NEW_DAYS * 86400
        out = [(vid, datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds"))
               for vid, ts in added.items() if ts >= cutoff]
        return sorted(out, key=lambda x: x[1], reverse=True)
    seen = update_seen(now) if seen is None else seen
    cats = CStore.all_catalogues()
    out = []
    for vid, iso in seen.items():
        try:
            t = datetime.datetime.fromisoformat(iso)
        except ValueError:
            continue
        if vid in cats and (now - t).days < NEW_DAYS:
            out.append((vid, iso))
    return sorted(out, key=lambda x: x[1], reverse=True)


def _spot_path(video_id: str) -> Path:
    return SPOTLIGHT_DIR / f"{hashlib.sha1(video_id.encode()).hexdigest()[:16]}.json"


def spotlight_signature(video_id: str, fits: dict, state: list = None) -> str:
    return hashlib.sha1(json.dumps([RESULTS_VERSION, video_id, state or library_state(), REC.SEARCH_VERSION,
                                    sorted(fits.items())], default=str).encode()).hexdigest()[:16]


def track_fits(video_id: str, results: dict = None) -> dict:
    """{track_id: how well the video fits the track on its own (0–1)} from the stored searches."""
    results = all_results() if results is None else results
    return {tid: d["insights"]["videos"][video_id] for tid, d in results.items()
            if video_id in ((d.get("insights") or {}).get("videos") or {})}


def load_spotlight(video_id: str, sig: str = None):
    try:
        data = json.loads(_spot_path(video_id).read_text())
    except (OSError, ValueError):
        return None
    if data.get("version") != RESULTS_VERSION:
        return None
    return data if sig is None or data.get("sig") == sig else None


def run_spotlight(video_id: str, progress=None, state: list = None) -> dict:
    """🆕 A new video's spotlight: similar videos, the tracks it fits best, and a ready-made
    plan built around it for each of the top SPOTLIGHT_TRACKS tracks."""
    progress = progress or (lambda f, t: None)
    fits = track_fits(video_id)
    sig = spotlight_signature(video_id, fits, state)
    similar = [{"video_id": v, "similarity": round(s, 3), "reason": r}
               for s, r, v in similar_finder()(video_id)[:5]]
    prints = library_prints()
    plans = []
    top = sorted(fits.items(), key=lambda x: -x[1])[:SPOTLIGHT_TRACKS]
    curves_of, role_secs, roles_of = search_curves(), role_seconds(), scene_roles_lookup()
    for i, (tid, fit) in enumerate(top):
        progress(i / max(len(top), 1), f"Building a plan for {tid}…")
        track = load_track(tid)
        story_regions, _ = story_for(tid, track)
        plan = REC.plan_with_video(track, prints, video_id, match=make_matcher(track, curves_of),
                                   curves_of=curves_of, n_videos=min(DEFAULT_VIDEOS, len(prints)),
                                   story_regions=story_regions, role_seconds=role_secs, roles_of=roles_of)
        if plan:
            plans.append(dict(slim_plan(plan, story_regions), track_id=tid, fit=fit))
    data = {"version": RESULTS_VERSION, "sig": sig, "video_id": video_id, "finished_at": _now(),
            "similar": similar, "fits": dict(sorted(fits.items(), key=lambda x: -x[1])[:10]), "plans": plans}
    try:
        SPOTLIGHT_DIR.mkdir(parents=True, exist_ok=True)
        _spot_path(video_id).write_text(json.dumps(data))
    except OSError:
        pass
    progress(1.0, "Done")
    return data


def video_usage() -> dict:
    """{video_id: recent appearances in saved projects and renders} (for 🧩 Hidden gems)."""
    import video_ranking as VR
    try:
        from config import PREVIEW_DIR, PROJECTS_DIR
    except ImportError:
        return {}
    return VR.combined_usage(PROJECTS_DIR, PREVIEW_DIR)
