"""⏱️ Edge Mode — the page: setup, the 🎙️ lines panel, 🏷️ footage per phase, the result, and the running job's
progress. What it compiles, renders and runs in the background is in planner/edge.py (split out 16 Nov 2026)."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

import edge_mode as E
import library_common as LC
import planner.compose as PC
import render_jobs as RJ
import settings_store as SSTORE
from config import PLANS_DIR, PREVIEW_DIR
from library_common import format_mmss
from planner.edge import (JOB_KEY, KEPT, KEPT_IF_SET, RESULT_KEY, SETUP_KEYS, _missing_sources, _setup_from_plan, _start,
                          _used_line, avatar_line_clip, took_line, cue_map_path, fetch_lines, line_file, line_takes, load_cue_map,
                          save_cue_map, LOOK_PARTS, LOOK_PART_LABELS, load_looks, save_looks, local_looks,
                          SURPRISE, surprise_count, DEFAULT_PHASE_ROLES, TIMER_STYLES)


def _take_finished_job() -> bool:
    """A finished background job's result into this session (once) — also in a new session after the page was
    reloaded. True while one is still running."""
    j, running = RJ.take_finished(JOB_KEY, "edge_job_seen")
    if j is None:
        return running
    ss = st.session_state
    if j.state == "error":
        ss["edge_msg"] = ("error", f"{j.label} failed: {j.error}")
        return False
    got = j.result or {}
    if got.get("res") is not None:
        ss[RESULT_KEY] = got["res"]
    if got.get("render"):
        ss["edge_render"] = got["render"]
        ss["edge_msg"] = got.get("note")
    elif got.get("res") is not None:
        ss.pop("edge_render", None)
    return False


def _lines_panel(cue_map: dict) -> None:
    """🎙️ The cue map's lines: how many are on this machine, and a button to fetch them from Google Drive."""
    lines = cue_map["lines"]
    here = sum(1 for ln in lines if line_file(ln, cut=False) is not None)
    mapped = sum(1 for ln in lines if ln.get("cues"))
    with st.expander(f"🎙️ Voiceover lines — {mapped} of {len(lines)} have a cue · {here} of {len(lines)} are on the server",
                     expanded=here < len(lines)):
        clips = sum(1 for ln in lines if avatar_line_clip(ln, cut=False) is not None)
        st.caption("Which line plays when comes from your cue map; no two lines ever play at once. A line whose file "
                   f"isn't here yet plays as a placeholder tone. 👩 {clips} of her clips are here. New lines and clips "
                   "come across with **🔄 Sync now** in the sidebar (only new and re-recorded files are fetched); "
                   "Compile also fetches whatever is still missing by itself.")
        missing = [ln["name"] for ln in lines if line_file(ln, cut=False) is None]
        if missing and here:
            st.caption("Not here yet: " + " · ".join(missing))
        extra = {ln["name"]: len(line_takes(ln, True)) - 1 for ln in lines}
        if any(extra.values()):
            st.caption("🎲 Other takes (they take turns): " + " · ".join(f"{n} +{k}" for n, k in extra.items() if k))
        st.caption("🎲 More than one take of a line: name the file \"<line> take 2.mp3\" (or \"(2)\") beside it on "
                   "Drive, her clip the same name in avatar_clips.")
        import cue_store as CS
        st.caption("🗺️ " + ("The cue map is kept in the app — edit it on the 🎙️ Cue Map page (every save is kept as a "
                            "version)." if CS.managed() else
                            "The cue map still follows the newest file sent to Drive from the old Cue Map page — open "
                            "🎙️ Cue Map and keep it in the app instead."))
        try:
            st.page_link("pages/8_Cue_Map.py", label="🎙️ Open the Cue Map →")
        except Exception:                                  # noqa: BLE001 — (headless tests run the page on its own)
            pass
        n = st.session_state.get("edge_cue_up_n", 0)
        up = st.file_uploader("🗺️ Replace the cue map", type=["json"], key=f"edge_cue_up_{n}")
        if up is not None:
            err = save_cue_map(up.getvalue())
            st.session_state["edge_cue_up_n"] = n + 1
            st.session_state["edge_msg"] = ("error", err) if err else ("success", "Cue map replaced — the next compile uses it.")
            st.rerun()
        _loudness(lines)


def _loudness(lines: list) -> None:
    """🔊 How loud each line is as recorded, and what the render does about it."""
    import edge_render as R
    st.caption("🔊 Every line is evened out when rendered: each one is brought to the typical line's level (quiet ones "
               "up to +15 dB, loud ones down up to 6 dB), her clips' sound too — then the voiceover's +15 % and a "
               "limiter so nothing clips. A line's own ± from the Cue Map's 🔊 slider goes on top.")
    if not st.button("📊 Check the loudness", key="edge_loudness"):
        return
    files = {ln["name"]: line_file(ln) for ln in lines}
    with st.spinner("Measuring the lines…"):
        got = R.voice_levels(files)
        target = R.voice_target()
    if not got:
        st.info("No line is on the server yet — press 🔄 Sync now in the sidebar first.")
        return
    gains = {ln["name"]: float(ln.get("gain") or 0) for ln in lines}
    up = [g for g in got if g[2] >= 3]
    st.write(f"Typical line: **{target:.1f} dB** · {len(got)} measured · **{len(up)}** turned up 3 dB or more, "
             f"{sum(1 for g in got if g[2] <= -3)} turned down 3 dB or more.")
    st.dataframe([{"Line": n, "As recorded (dB)": round(d, 1), "Evened out (dB)": f"{g:+.1f}",
                   "Cue Map 🔊": f"{gains.get(n, 0):+.0f}" if gains.get(n) else "",
                   "": "⚠️ very quiet — maybe re-record" if g >= R.LEVEL_UP_MAX - 0.05 else ""} for n, d, g in got],
                 hide_index=True, use_container_width=True)


def _looks_summary() -> None:
    """👗 Her look per round — one line; the plan itself is set on the 🧍 Avatar Studio page (🗓️ Round plan)."""
    looks = load_looks(load_cue_map())
    c = st.columns([5, 2])
    c[0].caption("👗 Her looks: " + " · ".join(f"{LOOK_PART_LABELS[p]}: {looks.get(p) or '—'}"
                                              for p in ("intro", "1", "2", "3", "last", "outro")))
    try:
        c[1].page_link("pages/9_Avatar_Studio.py", label="🧍 Set in the Studio →")
    except Exception:                                      # noqa: BLE001 — (headless tests run the page on its own)
        pass


def _phase_tags_panel(rows: dict, vids: list) -> None:
    """🏷️ Tags per phase: each phase's footage keeps to clips with one of its tags (blank = anything the filters allow)."""
    tags = sorted({t for v in vids if v in rows for t in PC._video_tags(rows[v])}, key=str.lower)
    for p in E.ORDER:
        PC._still_there("edge_tags_" + p, tags)
    set_ = [p for p in E.ORDER if st.session_state.get("edge_tags_" + p)]
    with st.expander("🏷️ Footage per phase" + (" — " + ", ".join(E.PROFILES[p]["label"].title() for p in set_) if set_ else " — optional")):
        st.caption("Pick scene tags for a phase and its footage keeps to clips carrying one of them (e.g. Extra = your "
                   "favourite positions). Blank = anything the video filters above allow. Where too little footage "
                   "carries a phase's tags, any footage is used there and the result page says so.")
        cols = st.columns(2)
        for k, p in enumerate(E.ORDER):
            cols[k % 2].multiselect(f"{E.PROFILES[p]['label'].title()}", tags, key="edge_tags_" + p,
                                    placeholder="any footage")


def _roles_default() -> None:
    for p in E.ORDER:
        st.session_state["edge_roles_" + p] = list(DEFAULT_PHASE_ROLES.get(p, []))


def _phase_roles_panel() -> None:
    """🎭 Roles per phase (Daryl, 21 Nov): each phase's footage keeps to clips carrying one of its 🎭 Reviewer role
    marks — strict; any footage where too little carries them. One set for every difficulty."""
    import scene_roles as SR
    roles = SR.all_roles()
    ids = [r["id"] for r in roles]
    names = {r["id"]: f"{r['emoji']} {r['name']}" for r in roles}
    for p in E.ORDER:
        PC._still_there("edge_roles_" + p, ids)
    set_ = {p: st.session_state.get("edge_roles_" + p) or [] for p in E.ORDER}
    is_default = {p: v for p, v in set_.items() if v} == {p: list(v) for p, v in DEFAULT_PHASE_ROLES.items()}
    with st.expander("🎭 Roles per phase — " + ("default (Wait: intro · Stop: outro)" if is_default else
                     " · ".join(f"{E.PROFILES[p]['label'].title()}: {', '.join(names.get(r, r) for r in v)}"
                                for p, v in set_.items() if v) or "none — the music decides")):
        st.caption("Tap the 🎭 Reviewer roles a phase's footage must have — only clips marked with one of them play "
                   "there (e.g. 🏁 Outro on Medium, Fast and Extra for a challenge). Blank = the music decides. Where "
                   "too little footage carries them, any footage fills in and the result says so. One set for every "
                   "difficulty; works with 🏷️ Footage per phase (a clip must fit both).")
        for p in E.ORDER:
            st.pills(E.PROFILES[p]["label"].title(), ids, selection_mode="multi", key="edge_roles_" + p,
                     format_func=lambda r: names.get(r, r))
        st.button("↺ Default", key="edge_roles_reset", on_click=_roles_default, disabled=is_default,
                  help="Wait: 🎬 Intro · Stop: 🏁 Outro · the rest: the music decides.")


def _surprise_alone() -> None:
    """🎲 Surprise me replaces the songs picked; picking a song after it takes Surprise me out."""
    ss = st.session_state
    now = list(ss.get("edge_playlist") or [])
    before = list(ss.get("_edge_playlist_before") or [])
    if SURPRISE in now and SURPRISE not in before:
        now = [SURPRISE]
    elif SURPRISE in now and len(now) > 1:
        now = [t for t in now if t != SURPRISE]
    ss["edge_playlist"] = now
    ss["_edge_playlist_before"] = now


def _setup(rows: dict, have_tl: set) -> None:
    st.caption("⏱️ A compilation run by a countdown timer: rounds of 20-second phases — Wait, Slow, Medium, Fast, "
               "Extra, Stop — shown as an overlay. Pick songs or let a beat follow the timer; the footage is matched "
               "to whichever plays.")
    summaries = PC.audio_summaries(PC._audio_signature())
    PC._still_there("edge_playlist", list(summaries) + [SURPRISE])
    picked = st.multiselect(
        "🎵 Playlist — optional", [SURPRISE] + sorted(summaries, key=str.lower), key="edge_playlist",
        on_change=_surprise_alone,
        format_func=lambda t: "🎲 Surprise me — songs picked for you when you compile" if t == SURPRISE else
        f"{t} · {format_mmss(summaries[t]['duration'])}" + (f" · {summaries[t]['bpm']:.0f} BPM" if summaries[t]["bpm"] else ""),
        help="Any number of songs, played back to back in the order picked; the list starts again until the rounds "
             "are filled, and the last song is cut where the time ends. None: a beat that follows the timer. "
             "🎲 Surprise me: the app picks the songs when you compile — how many from the rounds and difficulty, "
             "calmer for Chill and the most driving for Hard, building in energy, avoiding the last few Edge renders' songs.")
    surprise = SURPRISE in picked
    songs = [t for t in picked if t != SURPRISE]
    c = st.columns([2.2, 1.6, 2])
    diff = c[0].radio("Difficulty", list(E.DIFFICULTIES), key="edge_difficulty", horizontal=True,
                      format_func=lambda d: E.DIFFICULTIES[d])
    rounds = c[1].slider("Rounds", E.MIN_ROUNDS, E.MAX_ROUNDS, key="edge_rounds")
    tc = c[2].columns(2)
    tc[0].selectbox("Timer style", list(TIMER_STYLES), key="edge_timer_style", format_func=lambda k: TIMER_STYLES[k],
                    help="◯ Ring: the countdown ring in a corner, the stroke bar beside it. ▬ Strip: a slim box "
                         "at the top (seconds left, the phase large, the next phase small, the round) and the stroke "
                         "meter across the bottom.")
    tc[1].selectbox("Timer position", list(E.CORNERS), key="edge_corner", format_func=lambda k: E.CORNERS[k],
                    disabled=st.session_state.get("edge_timer_style") == "strip")
    st.caption("⏱️ " + E.describe(diff, rounds))
    if diff in E.HANDS_OFF_SEC:
        st.toggle("🎲 Surprises", key="edge_surprises",
                  help="From round 2, the timer doesn't always do what ⏭️ Next up promised: ✋ Hands off — the clock "
                       "freezes for a random few seconds mid-phase, then carries on; ⏩ Override — a phase runs past its "
                       "zero (+5 s, or up to +15 s on Medium), never before Wait or Slow; 🎭 False finish — once at most, "
                       "the last Extra counts down to Stop… then NOT YET, +10 s. Chill: 0–1 a round, Medium: 1–2. "
                       "Each has its own lines in the Cue Map (🎲 cue).")
    else:
        st.caption("💀 Hard draws its own structure every round: Wait, then random phases of random lengths — calmer "
                   "first, faster later, the faster ones longer, now and then a drop back to Slow — then Extra → Stop. "
                   "⏭️ Next up always tells the truth; how long is the surprise.")
    st.caption("🔔 Every phase change brings a bell, a flash of the phase's colour and the ring in that colour; the "
               "timer pulses over a phase's last 3 seconds. ▸ The round number sits by the timer. 🎭 Which clips play "
               "in each phase: 🎭 Roles per phase below (Wait: intro, Stop: outro by default).")
    total = E.round_seconds(diff) * rounds
    if surprise:
        n = surprise_count(diff, rounds, bool(st.session_state.get("edge_song_per_round")))
        st.caption(f"🎲 About {n} song{'s' if n != 1 else ''}, picked when you compile.")
    if songs or surprise:
        mc = st.columns(2)
        mc[0].toggle("🎵 New song each round", key="edge_song_per_round",
                     help="Round 1 plays the first song, round 2 the second… (the list starts again when there are "
                          "more rounds than songs). Each starts from its beginning and starts again if it ends before "
                          "the round does. Off: the songs back to back, whatever the rounds.")
        mc[1].toggle("⚡ Match song sections to each phase", key="edge_energy_mix",
                     help="The phases play in pairs — Wait + Slow, Medium + Fast, Extra + Stop — each pair on the "
                          "part of a song whose energy fits it (calm for Wait and Slow, the peak for Extra, Stop "
                          "fading on it). The parts are joined by the 🎚️ Mix engine: a 🧠 stem blend — Vocal swap, "
                          "Hold then cut, 2 bars — that cuts over to the next part as the phase changes (livelier "
                          "parts starting on the song's own drops are preferred), or a "
                          "crossfade with a bass swap for a song without its drum / bass / other stems. With 🎵 New "
                          "song each round on, the parts come from that round's song (no tempo clash at the joins); "
                          "otherwise from any song in the list, key and tempo weighed. The footage follows each part.")
        st.caption("🌙 The music fades out slowly over each Stop and comes back up over the Wait that follows.")
        if st.session_state.get("edge_energy_mix"):
            st.caption("⚡ " + ("Each round's parts come from its own song." if st.session_state.get("edge_song_per_round")
                               else f"Parts picked from all {len(songs)} song{'s' if len(songs) != 1 else ''} to fit each phase.")
                       + " Compile shows how close the fit is.")
        elif surprise:
            pass
        elif st.session_state.get("edge_song_per_round"):
            rp = E.round_playlist_parts([(t, summaries[t]["duration"]) for t in songs], E.schedule(diff, rounds))
            per = {}
            for p in rp:
                per.setdefault(p["round"], p["track_id"])
            st.caption("🎵 " + " · ".join(f"R{r}: {t}" for r, t in per.items())
                       + (" — a song shorter than its round starts again" if any(p["loop"] > 1 for p in rp) else ""))
        else:
            parts = E.playlist_parts([(t, summaries[t]["duration"]) for t in songs], total)
            passes = max(p["pass"] for p in parts) if parts else 1
            st.caption(f"🎵 {len(parts)} play{'s' if len(parts) != 1 else ''} of {len(songs)} song{'s' if len(songs) != 1 else ''}"
                       + (f" — the list comes round {passes} times" if passes > 1 else "")
                       + f"; the last one stops at {format_mmss(total)}.")
    t = st.columns(2)
    if songs or surprise:
        t[0].toggle("🥁 Beat sound over the songs", key="edge_beat",
                    help="Off with a song by default. On: the timer's beat (Slow 60, Medium 100, Fast 140, Extra 180 "
                         "a minute; silent in Wait and Stop) plays over the music. The footage still follows the songs.")
    else:
        t[0].toggle("🥁 Beat sound", value=True, disabled=True, key="edge_beat_shown",
                    help="No song picked, so the beat is the soundtrack: Slow 60, Medium 100, Fast 140, Extra 180 "
                         "beats a minute, silent in Wait and Stop. The footage is matched to it.")
    t[1].toggle("🎙️ Voiceover", key="edge_voice",
                help="A voiceover clip at the start of every phase. The songs drop to a quarter volume (background) so it is heard.")
    if st.session_state.get("edge_voice") or st.session_state.get("edge_avatar"):
        cue_map = load_cue_map()
        if cue_map.get("lines"):
            _lines_panel(cue_map)
    a = st.columns(2)
    a[0].toggle("👩 Avatar", key="edge_avatar",
                help="She appears with every line of your cue map (with or without the voiceover switch) — her clip "
                     "where one has been made (its own sound is heard), a stand-in figure and the line's recording "
                     "for the rest. The songs drop to a quarter "
                     "volume.")
    a[1].toggle("🎬 Intro screen", key="edge_intro",
                help="Before the compilation: the avatar giving her intro, beside the difficulty, rounds, total time "
                     "and thumbnails of the videos used. It lasts as long as her intro clip (8 s with the stand-in).")
    if st.session_state.get("edge_avatar"):
        _looks_summary()
        m = st.columns(2)
        mode = m[0].radio("She appears", list(E.AVATAR_MODES), key="edge_avatar_mode", horizontal=True,
                          format_func=lambda k: E.AVATAR_MODES[k],
                          help="In a corner: a framed box over the footage. Split screen: she takes the right half, "
                               "the footage's middle the left, for as long as her clip runs.")
        if mode == "split":
            m[1].toggle("🗒️ Instructions", key="edge_instructions",
                        help="A text command in her half of the bottom strip — STROKE, JUST THE TIP… — put up by the "
                             "lines it's linked to on the 🎙️ Cue Map (🗒️ Instructions) and kept until the next one.")
        if mode == "corner":
            m[1].selectbox("Her corner", [k for k in E.CORNERS if k != "centre"], key="edge_avatar_corner",
                           format_func=lambda k: E.CORNERS[k])
            if (st.session_state.get("edge_timer_style") != "strip"
                    and st.session_state.get("edge_avatar_corner") == st.session_state.get("edge_corner")):
                st.warning("The avatar and the timer are set to the same corner — she will sit under the timer.")
    r = st.columns([2.2, 1])
    r[0].toggle("🎲 Random videos", key="edge_random_videos",
                help="Each compile draws this many videos at random from the ones the filters below allow — a "
                     "different mix every time. Enough footage for 🏷️ Footage per phase is drawn first; more are "
                     "added when they aren't enough footage for the run.")
    r[1].number_input("How many", 1, 200, key="edge_random_n", disabled=not st.session_state.get("edge_random_videos"),
                      label_visibility="collapsed")
    vids = PC._video_filters("ce_", rows, have_tl,
                             f"🎬 Videos / tags — optional (the best fit from all {len(have_tl)} unless you narrow it)")
    ranges = LC.library_ranges()
    mins = sum((ranges[v][1] - ranges[v][0]) if v in ranges else float(rows[v].get("duration") or 0) for v in vids) / 60
    st.caption(f"🎬 {len(vids)} video{'s' if len(vids) != 1 else ''} · {mins:.0f} min of footage in play for "
               f"{format_mmss(total)} of compilation"
               + (f" · 🎲 {min(int(st.session_state.get('edge_random_n') or 30), len(vids))} drawn at random each compile"
                  if st.session_state.get("edge_random_videos") else ""))
    if mins * 60 < total:
        st.warning("There is less footage in play than the compilation is long — widen the video filters or pick fewer rounds.")
    _phase_tags_panel(rows, vids)
    _phase_roles_panel()
    with st.popover("⚙️ More"):
        st.number_input("At most this many different videos (0 = no limit)", 0, 60, key="edge_max_videos",
                        help="Fewer different videos = fewer files to fetch for a full-quality render.")
    go = st.columns([2, 1])
    go[0].button("⏱️ Compile & render", key="edge_go", type="primary", width="stretch", on_click=_start, args=(vids, True),
                 help="Compiles, then renders straight away — on the server: you can leave the page and come back.")
    go[1].button("Compile only", key="edge_go_only", width="stretch", on_click=_start, args=(vids, False))


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
                                                             f"{format_mmss(p['play'])} played"
                                                             + (f"<br>⚡ its {format_mmss(p['bar'])} · energy {p['energy']:.2f} "
                                                                f"for {p['profile'].title()} (aim {p['target']:.2f})"
                                                                + (" · carries on" if p.get("carry") else "")
                                                                if "energy" in p else "") + "<extra></extra>"))
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


def _save(res: dict) -> None:
    from page_setup import DRIVE_SESSION_KEY, OAUTH_AVAILABLE, push_file_with_oauth
    plan = res["plan"]
    Path(PLANS_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(PLANS_DIR) / f"{plan['track_id']}_plan.json"
    out.write_text(RJ.plan_json(plan))
    if OAUTH_AVAILABLE and st.session_state.get(DRIVE_SESSION_KEY):
        err = push_file_with_oauth(st.session_state[DRIVE_SESSION_KEY], out, "scene-labeling/compilation_plans")
        st.session_state["edge_msg"] = ("warning", f"Saved locally but Drive upload failed: {err}") if err else \
            ("success", f"Saved and uploaded to Drive: {out.name}")
    else:
        st.session_state["edge_msg"] = ("success", f"Saved locally: {out.name}")


def _result(res: dict) -> None:
    plan, summary = res["plan"], res["summary"]
    c = st.columns([5, 1.4])
    c[0].subheader("⏱️ Edge Mode")
    c[1].button("← Change", key="edge_back", width="stretch", on_click=lambda: st.session_state.pop(RESULT_KEY, None))
    st.caption(_used_line(res) + f" · made {res['made']}")
    if res.get("took"):
        st.caption(took_line(res))
    st.caption(f"{len(plan['timeline'])} clips · {summary.get('videos', 0)} different videos used"
               + (f" · 🎯 movement sync {summary['sync']:.0%}" if summary.get("sync") is not None else "")
               + (f" · clips {summary['clip_sec'][0]:g}–{summary['clip_sec'][2]:g} s" if summary.get("clip_sec") else "")
               + (f" · ⚠️ {res['shortfall']} slot(s) unfilled" if res.get("shortfall") else ""))
    for n in res.get("notes") or []:
        st.caption("⏱️ " + n)
    _strip(res)
    vo = plan["edge"]["voiceover"]
    if vo.get("on") and vo.get("cue_map"):
        clashes = res.get("voice_clashes") or []
        with st.expander(f"🎙️ Voiceover — {len(vo['cues'])} lines" + (f" · {len(clashes)} clashes" if clashes else " · no clashes"),
                         expanded=False):
            if clashes:
                st.markdown("**Clashes** — two lines wanted the same moment; they never overlap, so one gave way:")
                for c in clashes:
                    st.caption(f"⚠️ {c['line']} ({c['where']}) met {c['with']} → {c['result']}")
            av_on = (plan["edge"].get("avatar") or {}).get("on")
            if av_on and res.get("missing_avatar"):       # 👩 the lines in this compilation she has no clip for yet
                st.caption(f"👩 No avatar clip on Google Drive yet for {len(res['missing_avatar'])} of this compilation's lines "
                           f"(stand-in figure, the line's own recording is heard): " + ", ".join(res["missing_avatar"]))
            unused, no_cue = res.get("lines_unused") or [], res.get("lines_no_cue") or []
            if unused or no_cue:
                st.caption(f"🔇 Not heard in this compilation: {len(unused)} line(s) whose cue is for a round it doesn't have"
                           + (" (" + ", ".join(unused) + ")" if unused else "")
                           + (f" · {len(no_cue)} with no cue yet (" + ", ".join(no_cue) + ")" if no_cue else ""))
            intro_l, outro_l = plan["edge"]["intro"].get("lines") or [], plan["edge"]["outro"].get("lines") or []
            if intro_l:
                st.caption("🎬 Intro: " + " → ".join(x["line"] for x in intro_l))
            for c in vo["cues"]:
                st.caption(f"{format_mmss(c['time'])} · round {c['round']} · {c['line']} ({c['seconds']:.1f} s) — {c['why']}"
                           + (f" · moved {c['moved']:+.1f} s" if c.get("moved") else "") + ("" if c["file"] else " · placeholder tone"))
            if outro_l:
                st.caption("🎬 Outro: " + " → ".join(x["line"] for x in outro_l))
    st.text_input("🏷️ Name the next render (optional)", key="edge_render_name", max_chars=80)
    b = st.columns([1.8, 0.5, 1.4, 1.4, 1.6])
    b[0].button("🎬 Render", key="edge_render_btn", type="primary", width="stretch",
                on_click=_start, args=(res["video_ids"], True, False, res))
    with b[1].popover("⚙️", width="stretch", help="Render options"):
        st.toggle("⚡ Quick preview — use the 480p copies on the server", key="edge_quick")
        st.toggle("Stream original videos from Google Drive", key="edge_stream")
        missing = _missing_sources(plan, bool(st.session_state.get("edge_quick", True)))
        st.caption(f"{len(missing)} source file{'s' if len(missing) != 1 else ''} not on the server yet"
                   if missing else "Every source file is on the server.")
    b[2].button("💾 Save plan", key="edge_save", width="stretch", on_click=_save, args=(res,))
    b[3].download_button("⬇️ Download", data=RJ.plan_json(plan), file_name=f"{plan['track_id']}_plan.json",
                         mime="application/json", width="stretch", key="edge_dl")
    b[4].button("↻ Recompile & render", key="edge_again", width="stretch", on_click=_start, args=(res["video_ids"], True),
                kwargs={"draw": False, "pool": (res.get("drawn") or {}).get("pool")},
                help="Build it again with the current setup and render it — Hard draws a new order of phases.")
    drawn = res.get("drawn") or {}
    if drawn.get("songs") or drawn.get("pool"):
        d = st.columns(2)
        if drawn.get("songs"):
            d[0].button("🎲 Re-draw songs", key="edge_redraw_songs", width="stretch", on_click=_start,
                        args=(res["video_ids"], True), kwargs={"playlist": [SURPRISE], "draw": False, "pool": drawn.get("pool")},
                        help="New Surprise me songs; the same videos and everything else — compiled and rendered.")
        if drawn.get("pool"):
            d[1].button("🎲 Re-draw videos", key="edge_redraw_videos", width="stretch", on_click=_start,
                        args=(drawn["pool"], True),
                        kwargs={"playlist": list(drawn["songs"]) if drawn.get("songs") else None},
                        help="A new random set of videos from the same filters; the same songs — compiled and rendered.")
    if name := st.session_state.get("edge_render"):
        from library_common import show_render_player
        from renders import load_render
        meta = load_render(Path(PREVIEW_DIR) / name)
        if meta:
            show_render_player(meta)
        else:
            st.caption(f"{name} — see 📁 Media Library → 🎞️ Renders.")


def _reopen(ro: dict) -> None:
    """↩️ An Edge render reopened from 📁 Media Library → Renders: the setup it was made with, and its plan on show
    as the result (🎬 Render again, ⬇️ Download, ↻ Recompile & render with the same draw)."""
    import core.story_matcher as SMX
    plan = ro["plan"]
    edge = plan["edge"]
    setup = dict(ro.get("setup") or _setup_from_plan(edge))
    ss = st.session_state
    for k, key in SETUP_KEYS.items():
        if k in setup:
            ss[key] = setup[k]
    if setup.get("surprise_songs"):                        # 🎲 made with Surprise me: it stays Surprise me
        ss["edge_playlist"] = [SURPRISE]
    ss["edge_random_videos"] = bool(setup.get("random_videos"))
    if setup.get("random_videos"):
        ss["edge_random_n"] = int(setup["random_videos"])
    for p in E.ORDER:                                      # 🏷️ tags per phase
        ss["edge_tags_" + p] = list((setup.get("phase_tags") or {}).get(p) or [])
    pr = setup.get("phase_roles", edge.get("phase_roles"))
    pr = DEFAULT_PHASE_ROLES if pr is None else pr          # 🎭 (renders from before 21 Nov: the default)
    for p in E.ORDER:
        ss["edge_roles_" + p] = list(pr.get(p) or [])
    if edge.get("seed") is not None:
        ss["edge_seed"] = int(edge["seed"])
    vo, av = edge.get("voiceover") or {}, edge.get("avatar") or {}
    ss[RESULT_KEY] = {
        "plan": plan, "summary": SMX.plan_summary(plan.get("timeline") or []), "shortfall": 0,
        "notes": [f"↩️ Reopened from render {ro.get('label', '')} — the plan as it was rendered; ↻ Recompile & render "
                  "builds it again from the same setup."],
        "schedule": edge["schedule"], "parts": edge.get("playlist") or [], "video_ids": list(plan.get("selected_videos") or []),
        "track_ids": list(dict.fromkeys(p["track_id"] for p in edge.get("playlist") or [])),
        "missing_voice": list(vo.get("placeholders") or []), "voice_clashes": list(vo.get("clashes") or []),
        "lines_missing": list(vo.get("lines_missing") or []), "missing_avatar": list(av.get("placeholders") or []),
        "lines_unused": [], "lines_no_cue": [], "made": time.strftime("%H:%M:%S")}
    ss.pop("edge_render", None)


def render() -> None:
    SSTORE.keep(KEPT, KEPT_IF_SET)
    ro = st.session_state.pop("_reopen_edge", None)
    if ro:
        _reopen(ro)
    msg = st.session_state.pop("edge_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    if _take_finished_job():                               # ⏳ a compile / render is running on the server
        RJ.watch(JOB_KEY)
        return
    msg = st.session_state.pop("edge_msg", None)          # (what the finished job left)
    if msg:
        getattr(st, msg[0])(msg[1])
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
