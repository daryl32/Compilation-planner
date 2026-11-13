"""⏱️ Edge Mode (Compilation Planner → ⏱️ Edge Mode) — a compilation run by a countdown timer.

Setup: 🎵 a playlist (optional — any number of songs, played back to back and started again until the time is
filled) · 🥁 the beat sound (on by itself when there is no song) · rounds (3–6) · 🎬 videos / tags (optional — the
best fit from every video otherwise) · difficulty (Chill / Medium / Hard) · 🎙️ voiceover (optional).

The footage is matched by the 🤖 new matcher, 🎯 Movement only: to each song as usual, or — with no song — to a
generated beat whose speed follows the timer's phases. The rendered video carries the countdown overlay, the
voiceover (the songs at half volume under it) and the beat; the plan's "edge" block carries the same for Blender.
The rules (phases, rounds, beat, overlay) are in edge_mode.py; the render step in edge_render.py.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import streamlit as st

import edge_mode as E
import library_common as LC
import planner.compose as PC
import settings_store as SSTORE
from config import CATALOGUE_DIR, PLANS_DIR, PREVIEW_DIR
from library_common import format_mmss

RESULT_KEY = "edge_result"
KEPT = {"edge_playlist": [], "edge_rounds": 3, "edge_difficulty": "chill", "edge_voice": False, "edge_beat": False,
        "edge_corner": "top-left", "edge_max_videos": 0, "edge_quick": True, "edge_stream": True, "edge_render_name": ""}
KEPT_IF_SET = tuple("ce_" + k for k in ("actors", "tags", "hair", "gg", "new", "minlen", "exclude"))


def voice_dir() -> Path:
    """Where the voiceover clips live: <data>/edge_voiceover/<phase>/ — beside the catalogue, on this machine."""
    return Path(CATALOGUE_DIR).parent / "edge_voiceover"


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

def _trim_entry(entry: dict, new_end: float) -> dict:
    """A timeline entry cut short at new_end (the compilation's time): each slot's chain keeps only what fits."""
    a = float(entry["track_time"][0])
    keep = max(new_end - a, 0.0)
    scenes = []
    for slot in entry.get("scenes") or []:
        left, chain = keep, []
        for link in slot.get("chain") or []:
            if left <= 1e-3:
                break
            d = min(float(link["clip_duration_sec"]), left)
            chain.append({**link, "clip_duration_sec": round(d, 3)})
            left -= d
        if chain:
            new = {**slot, "chain": chain}
            if "clip_duration_sec" in slot:
                new["clip_duration_sec"] = round(min(float(slot["clip_duration_sec"]), keep), 3)
            scenes.append(new)
    return {**entry, "track_time": [round(a, 3), round(new_end, 3)], "scenes": scenes}


def _place(timeline: list, offset: float, play: float, part: int, track_id: str) -> list:
    """One song's timeline moved to where the song plays in the compilation, and cut where the song is cut."""
    out = []
    for e in timeline:
        a, b = float(e["track_time"][0]), float(e["track_time"][1])
        if a >= play - 0.05:
            break
        moved = {**e, "track_time": [round(a + offset, 3), round(b + offset, 3)], "edge_part": part,
                 "edge_track_id": track_id}
        if b > play + 1e-6:
            moved = _trim_entry(moved, offset + play)
        out.append(moved)
    return out


def compile_edge(track_ids: list, video_ids: list, difficulty: str, rounds: int, *, voiceover: bool = False,
                 beat: bool = False, seed: int = 0, corner: str = "top-left", max_videos: int = 0,
                 progress=None) -> dict:
    """The Edge plan. track_ids: the playlist in playing order ([] = no song: the generated beat is the track, and
    the beat sound is on whatever `beat` says). The footage is matched 🎯 Movement only — each song on the 📐 lines
    from its music; a song's later plays (and later songs) leave out the clips already used, as long as the library
    can still fill the time without them. progress(done, total, text)."""
    import core.story_matcher as SMX
    sched = E.schedule(difficulty, rounds, seed)
    total = E.total_seconds(sched)
    track_ids = list(track_ids or [])
    beat_on = bool(beat) or not track_ids
    notes, timeline, parts, shortfall = [], [], [], 0
    base = None
    if track_ids:
        durations = [(tid, float(PC._load_track(tid)["duration_sec"])) for tid in track_ids]
        parts = E.playlist_parts(durations, total)
        used = set()
        for k, part in enumerate(parts):
            def prog(d, t, _k=k, _tid=part["track_id"]):
                if progress:
                    progress(_k + d / max(t, 1), len(parts), f"song {_k + 1} of {len(parts)} · {_tid} · block {d} of {t}")
            kw = dict(movement_only=True, cuts="lines", max_videos=int(max_videos or 0), progress=prog)
            res = PC.compile_plan(part["track_id"], video_ids, excluded=set(used), **kw)
            if used and (res.get("shortfall") or float(res.get("unfilled") or 0.0) > 0.5):
                again = PC.compile_plan(part["track_id"], video_ids, **kw)        # not enough fresh footage left
                if float(again.get("unfilled") or 0.0) < float(res.get("unfilled") or 0.0) or (
                        res.get("shortfall") and not again.get("shortfall")):
                    res = again
                    notes.append(f"{part['track_id']} (play {k + 1}): not enough unused footage left — clips from "
                                 "earlier in the compilation may come round again.")
            base = base or res
            placed = _place(res["plan"]["timeline"], part["offset"], part["play"], k, part["track_id"])
            timeline += placed
            shortfall += int(res.get("shortfall") or 0)
            used |= {(link["video_id"], link["scene_id"]) for e in placed for s in e["scenes"] for link in s["chain"]}
            notes += [n for n in res.get("notes") or [] if n not in notes]
        gap = total - (parts[-1]["offset"] + parts[-1]["play"]) if parts else 0.0
        if gap > 1e-3 and timeline and timeline[-1]["scenes"]:      # under half a second left over: the last clip runs on
            last = timeline[-1]
            for slot in last["scenes"]:
                slot["chain"][-1] = {**slot["chain"][-1],
                                     "clip_duration_sec": round(float(slot["chain"][-1]["clip_duration_sec"]) + gap, 3)}
                if "clip_duration_sec" in slot:
                    slot["clip_duration_sec"] = round(float(slot["clip_duration_sec"]) + gap, 3)
            last["track_time"] = [last["track_time"][0], round(total, 3)]
    else:
        track = E.beat_track(sched)
        res = PC.compile_plan(E.TRACK_ID, video_ids, movement_only=True, max_videos=int(max_videos or 0),
                              given_track=track, given_segments=E.beat_segments(sched),
                              lab={"stem": False, "phrase": False},      # a beat has no lead layer or phrases to follow
                              progress=(lambda d, t: progress(d / max(t, 1), 1, f"block {d} of {t}")) if progress else None)
        base = res
        timeline = [{**e, "edge_part": 0, "edge_track_id": None} for e in res["plan"]["timeline"]]
        shortfall = int(res.get("shortfall") or 0)
        notes += list(res.get("notes") or [])
    for k, e in enumerate(timeline):
        e["segment_index"] = k
        ph = E.phase_at(sched, (e["track_time"][0] + e["track_time"][1]) / 2)
        e["edge_profile"] = ph["profile"] if ph else None
    files = E.voice_files(voice_dir()) if voiceover else {}
    cues = E.voice_cues(sched, files, seed) if voiceover else []
    missing_voice = sorted({c["profile"] for c in cues if not c["file"]}, key=E.ORDER.index)
    name = f"edge_{difficulty}_{len(sched) and sched[-1]['round']}r" + ("_" + track_ids[0] if track_ids else "_beat")
    plan = {
        **base["plan"],
        "track_id": name, "matching_mode": "Edge", "track_source_path": base["plan"].get("track_source_path", ""),
        "track_duration_sec": round(total, 3), "bpm": base["plan"].get("bpm"),
        "selected_videos": list(video_ids), "timeline": timeline,
        # ⏱️ everything the timer adds — the app's render builds it (edge_render.py); Blender can read it from here
        "edge": {
            "difficulty": difficulty, "rounds": int(sched[-1]["round"]) if sched else 0, "seed": int(seed),
            "round_seconds": E.round_seconds(difficulty), "total_seconds": round(total, 3),
            "profiles": {p: {k: v for k, v in E.PROFILES[p].items() if k != "colour"} | {"colour": list(E.PROFILES[p]["colour"])}
                         for p in E.ORDER},
            "schedule": sched,
            "playlist": [{**p, "source_path": PC._load_track(p["track_id"]).get("source_path", "")} for p in parts],
            "beat": {"on": beat_on, "volume": E.BEAT_UNDER_SONG if track_ids else 1.0, "times": E.beat_times(sched)},
            "voiceover": {"on": bool(voiceover), "music_volume": E.MUSIC_UNDER_VOICE, "cues": cues,
                          "placeholders": missing_voice},
            "overlay": {"corner": corner, "fps": E.OVERLAY_FPS},
        },
    }
    return {"plan": plan, "summary": SMX.plan_summary(timeline), "notes": notes, "shortfall": shortfall,
            "schedule": sched, "parts": parts, "video_ids": list(video_ids), "track_ids": track_ids,
            "missing_voice": missing_voice, "made": time.strftime("%H:%M:%S")}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _queue_compile(video_ids: list, new_seed: bool = True) -> None:
    if not video_ids:
        st.session_state["edge_msg"] = ("warning", "No videos to match — widen the video filters.")
        return
    if new_seed or "edge_seed" not in st.session_state:
        st.session_state["edge_seed"] = random.randrange(1, 10 ** 6)       # Hard draws a new order each compile
    st.session_state["edge_pending"] = {"video_ids": list(video_ids)}


def _do_pending_compile() -> None:
    job = st.session_state.pop("edge_pending", None)
    if not job:
        return
    ss = st.session_state
    bar = st.progress(0.0, text="⏱️ Compiling…")
    try:
        res = compile_edge(list(ss.get("edge_playlist") or []), job["video_ids"], ss.get("edge_difficulty", "chill"),
                           int(ss.get("edge_rounds", 3)), voiceover=bool(ss.get("edge_voice")),
                           beat=bool(ss.get("edge_beat")), seed=int(ss.get("edge_seed", 0)),
                           corner=ss.get("edge_corner", "top-left"), max_videos=int(ss.get("edge_max_videos", 0) or 0),
                           progress=lambda d, t, txt: bar.progress(min(d / max(t, 1), 1.0), text=f"⏱️ Compiling… {txt}"))
    except Exception as e:          # noqa: BLE001
        bar.empty()
        st.session_state["edge_msg"] = ("error", f"Couldn't compile: {e}")
        return
    bar.empty()
    ss[RESULT_KEY] = res
    ss.pop("edge_render", None)


def _voice_manager() -> None:
    files = E.voice_files(voice_dir())
    have = sum(len(v) for v in files.values())
    with st.expander(f"🎙️ Voiceover clips — {have} added" + ("" if have else " (placeholder tones until you add yours)")):
        st.caption("A clip plays at the start of each phase, picked from that phase's clips in turn. A phase with no "
                   "clip yet gets a short placeholder tone, so the timing can be checked now.")
        c = st.columns([1.3, 3, 1])
        prof = c[0].selectbox("Phase", E.ORDER, format_func=lambda p: E.PROFILES[p]["label"].title(), key="edge_voice_phase")
        up = c[1].file_uploader("Add clips for this phase", type=[x.strip(".") for x in E.VOICE_EXT],
                                accept_multiple_files=True, key=f"edge_voice_up_{st.session_state.get('edge_voice_n', 0)}")
        if c[2].button("➕ Add", key="edge_voice_add", width="stretch", disabled=not up):
            d = voice_dir() / prof
            d.mkdir(parents=True, exist_ok=True)
            for f in up:
                (d / Path(f.name).name).write_bytes(f.getvalue())
            st.session_state["edge_voice_n"] = st.session_state.get("edge_voice_n", 0) + 1
            st.rerun()
        for p in E.ORDER:
            names = files[p]
            st.markdown(f"**{E.PROFILES[p]['label'].title()}** — " + (f"{len(names)} clip{'s' if len(names) != 1 else ''}"
                                                                       if names else "placeholder tone"))
            for f in names:
                r = st.columns([5, 1])
                r[0].caption(f.name)
                if r[1].button("🗑️", key=f"edge_voice_del_{p}_{f.name}", help="Remove this clip"):
                    f.unlink(missing_ok=True)
                    st.rerun()


def _setup(rows: dict, have_tl: set) -> None:
    st.caption("⏱️ A compilation run by a countdown timer: rounds of 20-second phases — Wait, Slow, Medium, Fast, "
               "Extra, Stop — shown as an overlay. Pick songs or let a beat follow the timer; the footage is matched "
               "to whichever plays.")
    summaries = PC.audio_summaries(PC._audio_signature())
    PC._still_there("edge_playlist", summaries)
    songs = st.multiselect(
        "🎵 Playlist — optional", sorted(summaries, key=str.lower), key="edge_playlist",
        format_func=lambda t: f"{t} · {format_mmss(summaries[t]['duration'])}"
                              + (f" · {summaries[t]['bpm']:.0f} BPM" if summaries[t]["bpm"] else ""),
        help="Any number of songs, played back to back in the order picked; the list starts again until the rounds "
             "are filled, and the last song is cut where the time ends. None: a beat that follows the timer.")
    c = st.columns([2.2, 1.6, 2])
    diff = c[0].radio("Difficulty", list(E.DIFFICULTIES), key="edge_difficulty", horizontal=True,
                      format_func=lambda d: E.DIFFICULTIES[d])
    rounds = c[1].slider("Rounds", E.MIN_ROUNDS, E.MAX_ROUNDS, key="edge_rounds")
    c[2].selectbox("Timer position", list(E.CORNERS), key="edge_corner", format_func=lambda k: E.CORNERS[k])
    st.caption("⏱️ " + E.describe(diff, rounds))
    total = E.round_seconds(diff) * rounds
    if songs:
        parts = E.playlist_parts([(t, summaries[t]["duration"]) for t in songs], total)
        passes = max(p["pass"] for p in parts) if parts else 1
        st.caption(f"🎵 {len(parts)} play{'s' if len(parts) != 1 else ''} of {len(songs)} song{'s' if len(songs) != 1 else ''}"
                   + (f" — the list comes round {passes} times" if passes > 1 else "")
                   + f"; the last one stops at {format_mmss(total)}.")
    t = st.columns(2)
    if songs:
        t[0].toggle("🥁 Beat sound over the songs", key="edge_beat",
                    help="Off with a song by default. On: the timer's beat (Slow 60, Medium 100, Fast 140, Extra 180 "
                         "a minute; silent in Wait and Stop) plays over the music. The footage still follows the songs.")
    else:
        t[0].toggle("🥁 Beat sound", value=True, disabled=True, key="edge_beat_shown",
                    help="No song picked, so the beat is the soundtrack: Slow 60, Medium 100, Fast 140, Extra 180 "
                         "beats a minute, silent in Wait and Stop. The footage is matched to it.")
    t[1].toggle("🎙️ Voiceover", key="edge_voice",
                help="A voiceover clip at the start of every phase. The songs play at half volume so it is heard.")
    if st.session_state.get("edge_voice"):
        _voice_manager()
    vids = PC._video_filters("ce_", rows, have_tl,
                             f"🎬 Videos / tags — optional (the best fit from all {len(have_tl)} unless you narrow it)")
    ranges = LC.library_ranges()
    mins = sum((ranges[v][1] - ranges[v][0]) if v in ranges else float(rows[v].get("duration") or 0) for v in vids) / 60
    st.caption(f"🎬 {len(vids)} video{'s' if len(vids) != 1 else ''} · {mins:.0f} min of footage in play for "
               f"{format_mmss(total)} of compilation")
    if mins * 60 < total:
        st.warning("There is less footage in play than the compilation is long — widen the video filters or pick fewer rounds.")
    with st.popover("⚙️ More"):
        st.number_input("At most this many different videos (0 = no limit)", 0, 60, key="edge_max_videos",
                        help="Fewer different videos = fewer files to fetch for a full-quality render.")
    st.button("⏱️ Compile", key="edge_go", type="primary", width="stretch", on_click=_queue_compile, args=(vids,))


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

def _strip(res: dict) -> None:
    import plotly.graph_objects as go
    fig = go.Figure()
    for ph in res["schedule"]:
        prof = E.PROFILES[ph["profile"]]
        fig.add_trace(go.Bar(x=[ph["end"] - ph["start"]], base=[ph["start"]], y=["⏱️ timer"], orientation="h", width=0.8,
                             marker=dict(color="rgb(%d,%d,%d)" % tuple(prof["colour"]), line=dict(color="rgba(255,255,255,.5)", width=0.5)),
                             showlegend=False,
                             hovertemplate=f"Round {ph['round']} · {prof['label'].title()} · {format_mmss(ph['start'])}–"
                                           f"{format_mmss(ph['end'])}<extra></extra>"))
    for p in res["parts"]:
        fig.add_trace(go.Bar(x=[p["play"]], base=[p["offset"]], y=["🎵 songs"], orientation="h", width=0.8,
                             marker=dict(color="rgba(120,120,220,.55)", line=dict(color="rgba(255,255,255,.7)", width=1)),
                             showlegend=False, hovertemplate=f"{p['track_id']} · from {format_mmss(p['offset'])} · "
                                                             f"{format_mmss(p['play'])} played<extra></extra>"))
    for e in res["plan"]["timeline"]:
        a, b = e["track_time"]
        label = " / ".join(s["video_id"] for s in e["scenes"]) or "—"
        fig.add_trace(go.Bar(x=[b - a], base=[a], y=["🎬 clips"], orientation="h", width=0.8,
                             marker=dict(color="rgba(150,150,150,.55)" if e["scenes"] else "rgba(220,60,60,.8)",
                                         line=dict(color="rgba(0,0,0,.3)", width=0.5)),
                             showlegend=False, hovertemplate=f"{format_mmss(a)}–{format_mmss(b)}<br>{label}<extra></extra>"))
    fig.update_layout(barmode="overlay", height=150 if res["parts"] else 115, margin=dict(l=4, r=4, t=4, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(range=[0, float(res["plan"]["track_duration_sec"])], tickfont_size=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"edge_strip_{res['made']}")


def _missing_sources(plan: dict, quick: bool) -> list:
    """(raw source, local path) of every song and video the plan uses that isn't on this machine."""
    from planner.data import load_all_catalogues
    from render_preview import colab_to_local
    cats = load_all_catalogues()
    needed = {p["source_path"] for p in plan["edge"]["playlist"] if p.get("source_path")}
    for e in plan["timeline"]:
        for slot in e["scenes"]:
            for link in slot["chain"]:
                if quick and LC.proxy_path(link["video_id"]).exists():
                    continue
                needed.add(cats[link["video_id"]]["source_path"])
    return [(raw, Path(colab_to_local(raw))) for raw in sorted(needed) if not Path(colab_to_local(raw)).exists()]


def _do_pending_render(res: dict) -> None:
    if not st.session_state.pop("edge_do_render", False):
        return
    import fnmatch
    import re
    from page_setup import APP_VERSION, drive_token
    from planner.state import sanitize_filename
    from render_preview import render_plan_dict
    from renders import auto_clear_enabled, clear_local, render_files, update_user_fields, upload_render
    plan = res["plan"]
    stream = bool(st.session_state.get("edge_stream", True))
    quick = bool(st.session_state.get("edge_quick", True))
    if not stream:
        missing = _missing_sources(plan, quick)
        if missing:
            errs = PC._download_sources(missing)
            if errs:
                st.error("Can't render — some source files are missing and couldn't be downloaded:\n\n"
                         + "\n".join(f"- {e}" for e in errs))
                return
    safe = sanitize_filename(plan["track_id"])
    used = [int(m.group(1)) for p in render_files() if (m := re.search(r"\.(\d{3})\.mp4$", p.name))
            and fnmatch.fnmatch(p.name, f"temp-Video-preview-{safe}.*.mp4")]
    Path(PREVIEW_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(PREVIEW_DIR) / f"temp-Video-preview-{safe}.{(max(used) + 1) if used else 1:03d}.mp4"
    bar = st.progress(0.0, text="Rendering…")
    last, t0 = [0.0], time.time()

    def _prog(d, t):
        last[0] = d / max(t, 1)
        secs = time.time() - t0
        bar.progress(last[0], text=f"Rendering segment {d}/{t}" + (f" · about {int((secs / d) * (t - d))} s left" if d else ""))

    try:
        e = plan["edge"]
        stats = render_plan_dict(plan, str(out), progress_callback=_prog, stream=stream, quick=quick,
                                 notice=lambda txt: bar.progress(last[0], text=txt), app_version=APP_VERSION,
                                 extra_meta={"project_name": f"⏱️ Edge · {E.DIFFICULTIES[e['difficulty']]} · {e['rounds']} rounds",
                                             "compile_note": _used_line(res)}) or {}
        label = " ".join(str(st.session_state.get("edge_render_name") or "").split())[:80]
        if label:
            update_user_fields(out, name=label)
        secs = time.time() - t0
        took = f"{int(secs // 60)}:{int(secs % 60):02d}"
        note = ("info", f"Rendered {out.name} in {took}. Connect Google Drive (sidebar) to save renders there.")
        token = drive_token()
        if token:
            bar.progress(1.0, text="Uploading to Google Drive…")
            err = upload_render(out, token)
            if err:
                note = ("warning", f"Rendered, but the Drive upload failed: {err}")
            elif auto_clear_enabled():
                clear_local(out)
                note = ("success", f"Rendered {out.name} in {took}, saved to Google Drive and cleared from the server.")
            else:
                note = ("success", f"Rendered {out.name} in {took} and saved to Google Drive.")
        how = [f"{stats[k]} {w}" for k, w in (("quick", "from the 480p copies"), ("streamed", "streamed from Drive"),
                                             ("downloaded", "downloaded"), ("local", "already on the server")) if stats.get(k)]
        if how:
            note = (note[0], note[1] + " Source files: " + ", ".join(how) + ".")
        bar.empty()
        st.session_state["edge_msg"] = note
        st.session_state["edge_render"] = out.name
        st.rerun()
    except Exception as ex:          # noqa: BLE001
        bar.empty()
        st.error(f"Render failed: {ex}")


def _used_line(res: dict) -> str:
    e = res["plan"]["edge"]
    bits = [f"⏱️ Edge · {E.DIFFICULTIES[e['difficulty']]}", f"{e['rounds']} rounds ({format_mmss(e['total_seconds'])})",
            (f"🎵 {len(res['track_ids'])} song{'s' if len(res['track_ids']) != 1 else ''}, {len(res['parts'])} plays"
             if res["track_ids"] else "🥁 generated beat"),
            f"{len(res['video_ids'])} videos in play"]
    if res["track_ids"] and e["beat"]["on"]:
        bits.append("🥁 beat over the songs")
    if e["voiceover"]["on"]:
        bits.append("🎙️ voiceover" + (" (placeholder tones: " + ", ".join(E.PROFILES[p]["label"].title() for p in res["missing_voice"]) + ")"
                                      if res["missing_voice"] else ""))
    return " · ".join(bits)


def _save(res: dict) -> None:
    from page_setup import DRIVE_SESSION_KEY, OAUTH_AVAILABLE, push_file_with_oauth
    plan = res["plan"]
    Path(PLANS_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(PLANS_DIR) / f"{plan['track_id']}_plan.json"
    out.write_text(json.dumps(plan, indent=2))
    if OAUTH_AVAILABLE and st.session_state.get(DRIVE_SESSION_KEY):
        err = push_file_with_oauth(st.session_state[DRIVE_SESSION_KEY], out, "scene-labeling/compilation_plans")
        st.session_state["edge_msg"] = ("warning", f"Saved locally but Drive upload failed: {err}") if err else \
            ("success", f"Saved and uploaded to Drive: {out.name}")
    else:
        st.session_state["edge_msg"] = ("success", f"Saved locally: {out.name}")


def _result(res: dict) -> None:
    _do_pending_render(res)
    plan, summary = res["plan"], res["summary"]
    c = st.columns([5, 1.4])
    c[0].subheader("⏱️ Edge Mode")
    c[1].button("← Change", key="edge_back", width="stretch", on_click=lambda: st.session_state.pop(RESULT_KEY, None))
    st.caption(_used_line(res) + f" · made {res['made']}")
    st.caption(f"{len(plan['timeline'])} clips · {summary.get('videos', 0)} different videos used"
               + (f" · 🎯 movement sync {summary['sync']:.0%}" if summary.get("sync") is not None else "")
               + (f" · clips {summary['clip_sec'][0]:g}–{summary['clip_sec'][2]:g} s" if summary.get("clip_sec") else "")
               + (f" · ⚠️ {res['shortfall']} slot(s) unfilled" if res.get("shortfall") else ""))
    for n in res.get("notes") or []:
        st.caption("⏱️ " + n)
    _strip(res)
    st.text_input("🏷️ Name the next render (optional)", key="edge_render_name", max_chars=80)
    b = st.columns([1.8, 0.5, 1.4, 1.4, 1.6])
    b[0].button("🎬 Render", key="edge_render_btn", type="primary", width="stretch",
                on_click=lambda: st.session_state.update(edge_do_render=True))
    with b[1].popover("⚙️", width="stretch", help="Render options"):
        st.toggle("⚡ Quick preview — use the 480p copies on the server", key="edge_quick")
        st.toggle("Stream original videos from Google Drive", key="edge_stream")
        missing = _missing_sources(plan, bool(st.session_state.get("edge_quick", True)))
        st.caption(f"{len(missing)} source file{'s' if len(missing) != 1 else ''} not on the server yet"
                   if missing else "Every source file is on the server.")
    b[2].button("💾 Save plan", key="edge_save", width="stretch", on_click=_save, args=(res,))
    b[3].download_button("⬇️ Download", data=json.dumps(plan, indent=2), file_name=f"{plan['track_id']}_plan.json",
                         mime="application/json", width="stretch", key="edge_dl")
    b[4].button("↻ Recompile", key="edge_again", width="stretch", on_click=_queue_compile, args=(res["video_ids"],),
                help="Build it again with the current setup — Hard draws a new order of phases.")
    if name := st.session_state.get("edge_render"):
        from library_common import show_render_player
        from renders import load_render
        meta = load_render(Path(PREVIEW_DIR) / name)
        if meta:
            show_render_player(meta)
        else:
            st.caption(f"{name} — see 📁 Media Library → 🎞️ Renders.")


def render() -> None:
    SSTORE.keep(KEPT, KEPT_IF_SET)
    msg = st.session_state.pop("edge_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    _do_pending_compile()
    res = st.session_state.get(RESULT_KEY)
    if res:
        _result(res)
        return
    rows = PC._video_rows()
    have_tl = PC._videos_with_timeline() & set(rows)
    if not have_tl:
        st.warning("No video has a 🧭 footage timeline yet — build it in 📁 Media Library → 🎬 Videos → 🧭 Timeline first.")
        return
    _setup(rows, have_tl)
