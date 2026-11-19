"""⚔️ Verses Battle (Compilation Planner → ⚔️ Verses Battle) — two actors battle for your attention.

Setup: 🥊 the fighters (🎲 random draft — any two, ⚡ contrast or 🪞 rivals — or 👆 your choice, with 🎯 "find her a
rival") · 🎵 the music (🤝 one shared track or 🎵 a solo theme each; auto-picked to suit both, or your own pick) ·
🖼️ the layout (rounds with split screen, split throughout, or full screen only) · 🏆 the ending (surprise / a winner
/ a draw) · ⏱️ the length · 🥊 the tale-of-the-tape intro and 📊 the battle overlay.

Each fighter's footage is matched by the 🤖 new matcher (🎯 Movement only, 📐 lines — as ⏱️ Edge Mode) to her song;
the rounds (verses_battle.py) then weave the two together, and the render (verses_render.py) adds the sound, the
overlay and the card. The plan's "verses" block carries the lot for Blender. First draft: 18 Nov 2026.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import streamlit as st

import library_common as LC
import planner.compose as PC
import verses_battle as VB
from library_common import format_mmss

RESULT_KEY = "vs_result"
JOB_KEY = "verses_job"
KEPT = {"vs_draft": "random", "vs_style": "any", "vs_a": "", "vs_b": "", "vs_music": "shared", "vs_auto_music": True,
        "vs_track_a": "", "vs_track_b": "", "vs_layout": "mixed", "vs_winner": "surprise", "vs_length": 0,
        "vs_max_videos": 0, "vs_intro": True, "vs_overlay": True, "vs_quick": True, "vs_stream": True,
        "vs_render_name": ""}


# ---------------------------------------------------------------------------
# The fighters
# ---------------------------------------------------------------------------

def fighters_pool() -> dict:
    """{actor: [video ids with a 🧭 timeline]}."""
    return VB.actor_videos(LC.video_details(), PC._videos_with_timeline())


@st.cache_data(max_entries=4, show_spinner=False, ttl=900)
def _looks(pool_sig: tuple, tl_sig: tuple) -> dict:
    """{actor: look vector} — the mean footage embedding of her videos (for ⚡ contrast / 🪞 rivals)."""
    import numpy as np
    import footage_timeline as FT
    index = FT.load_index()
    per_video = {}
    out = {}
    for actor, vids in pool_sig:
        vecs = []
        for vid in vids:
            if vid not in per_video:
                tl = FT.load(vid, index=index)
                try:
                    e = tl.embeddings().mean(axis=0) if tl is not None and len(tl) else None
                except Exception:          # noqa: BLE001
                    e = None
                per_video[vid] = None if e is None else e / max(float(np.linalg.norm(e)), 1e-6)
            if per_video[vid] is not None:
                vecs.append(per_video[vid])
        if vecs:
            out[actor] = [float(x) for x in np.mean(vecs, axis=0)]
    return out


def looks(pool: dict) -> dict:
    import footage_timeline as FT
    try:
        return _looks(tuple((a, tuple(v)) for a, v in sorted(pool.items())), FT.signature())
    except Exception:          # noqa: BLE001 — contrast / rivals fall back to a random draw
        return {}


def hair_line(vids: list) -> str:
    """The fighter's hair colour(s), from her videos' answers (most common first)."""
    hair = LC.video_hair()
    counts = {}
    for vid in vids:
        for c in LC.hair_of(vid, hair).get("colours") or []:
            counts[c] = counts.get(c, 0) + 1
    top = sorted(counts, key=lambda c: -counts[c])[:2]
    return " / ".join(top) if top else "—"


# ---------------------------------------------------------------------------
# 🎵 The music
# ---------------------------------------------------------------------------

MIN_AUTO_SEC = 90.0          # 🤖 auto-pick leaves out songs shorter than this (stingers, intros) when it can


def _suggest(vids: list, top: int = 12, progress=None) -> list:
    import footage_timeline as FT
    got = PC.suggest_tracks(tuple(sorted(vids)), PC._audio_signature(), FT.signature(), top=top * 3, _progress=progress)
    long_ = [c for c in got if float(PC._load_track(c["track_id"]).get("duration_sec") or 0) >= MIN_AUTO_SEC]
    return (long_ or got)[:top]


def auto_music(mode: str, va: list, vb: list, progress=None) -> tuple:
    """🤖 The songs best suited to both: (track for A, track for B, notes). shared: the song whose worse fit of the
    two fighters is the best (so it suits both, not just one); solo: each fighter's best song, B's picked among her
    best six for a tempo near A's (smoother switches)."""
    notes = []
    if mode == "shared":
        import footage_timeline as FT
        cands = _suggest(va + vb, 15, progress)
        sig = FT.signature()
        tfa, tfb = PC._type_fits(tuple(va), sig), PC._type_fits(tuple(vb), sig)      # each side's fits worked out once
        best, best_score = None, -1.0
        for c in cands:
            track = PC._load_track(c["track_id"])
            fa, fb = PC.track_fit(va, track, type_fits=tfa), PC.track_fit(vb, track, type_fits=tfb)
            if not fa or not fb:
                continue
            score = min(fa["fit"], fb["fit"])
            if score > best_score:
                best, best_score = c["track_id"], score
                fits = (fa["fit"], fb["fit"])
        if best is None and cands:
            best = cands[0]["track_id"]
        if best is None:
            raise ValueError("No song could be suggested — no track has a 🧩 section map yet.")
        if best_score >= 0:
            notes.append(f"🤝 {best} suits both: fit {fits[0]:.2f} / {fits[1]:.2f}.")
        return best, best, notes
    got_a = _suggest(va, 6, progress)
    got_b = _suggest(vb, 6, progress)
    if not got_a or not got_b:
        raise ValueError("No song could be suggested — no track has a 🧩 section map yet.")
    ta = got_a[0]["track_id"]
    bpm_a = float(PC._load_track(ta).get("bpm") or 0.0)

    def score(c):
        bpm = float(PC._load_track(c["track_id"]).get("bpm") or 0.0)
        return c["fit"] - (0.004 * abs(bpm - bpm_a) if bpm and bpm_a else 0.0)
    rest = [c for c in got_b if c["track_id"] != ta] or got_b
    tb = max(rest, key=score)["track_id"]
    notes.append(f"🎵 Themes: {ta} for the first fighter, {tb} for the second (picked near {bpm_a:.0f} BPM).")
    return ta, tb, notes


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

def _thumb(timeline: list, side: str):
    """[video id, time] of the side's longest clip — her picture on the tale of the tape."""
    best = None
    for e in timeline:
        if e.get("verses_side") != side:
            continue
        for s in e.get("scenes") or []:
            for link in s.get("chain") or []:
                if best is None or link["clip_duration_sec"] > best[2]:
                    best = (link["video_id"], float(link["clip_start_sec"]), float(link["clip_duration_sec"]))
    return [best[0], best[1]] if best else None


def compile_verses(settings: dict, seed: int, progress=None, pool: dict = None) -> dict:
    """The battle's plan. settings: {"a", "b" (actor names), "music": "shared" | "solo", "track_a", "track_b"
    (None / "": auto-picked), "layout", "winner", "length" (seconds, 0 = the whole song), "max_videos",
    "intro", "overlay"}. progress(fraction, text)."""
    prog = progress or (lambda f, t: None)
    rng = random.Random(seed)
    pool = fighters_pool() if pool is None else pool
    a, b = settings["a"], settings["b"]
    if not a or not b or a == b:
        raise ValueError("Pick two different fighters.")
    va, vb = VB.fighter_videos(pool, a, b)
    if not va or not vb:
        raise ValueError(f"{a if not va else b} has no video of her own with a 🧭 footage timeline.")
    mode = settings.get("music", "shared")
    notes, took, t0 = [], {}, time.time()
    prog(0.02, "picking the music")
    ta, tb = settings.get("track_a") or None, settings.get("track_b") or None
    if mode == "shared":
        tb = ta
    if not ta or (mode == "solo" and not tb):
        sa, sb, n = auto_music(mode, va, vb, lambda k, n_, tid: prog(0.02 + 0.13 * k / max(n_, 1), f"scoring songs · {tid}"))
        ta, tb = ta or sa, tb or sb
        notes += n
    took["music"] = round(time.time() - t0, 1)
    tracks = {"A": ta, "B": tb}
    durs = {tid: float(PC._load_track(tid)["duration_sec"]) for tid in set(tracks.values())}
    total = min(durs[ta], durs[tb])
    cap = float(settings.get("length") or 0)
    fade = bool(cap and cap < total - 1.0)
    if fade:
        total = cap
    winner = VB.pick_winner(settings.get("winner", "surprise"), rng)
    first = rng.choice(["A", "B"])
    grids = {s: VB.bar_grid(PC._load_track(tracks[s]), total) for s in ("A", "B")}
    windows = VB.schedule(total, grids["A"], grids["B"], settings.get("layout", "mixed"), winner, first)
    # each fighter matched to her own song (the second leaves out nothing of the first's: their videos differ)
    timelines, results = {}, {}
    for k, (s, vids) in enumerate((("A", va), ("B", vb))):
        def cb(d, t, _k=k, _s=s):
            prog(0.15 + 0.4 * (_k + d / max(t, 1)), f"matching {settings[_s.lower()]} to {tracks[_s]} · block {d} of {t}")
        t0 = time.time()
        res = PC.compile_plan(tracks[s], vids, movement_only=True, cuts="lines",
                              max_videos=int(settings.get("max_videos") or 0), progress=cb)
        took[settings[s.lower()]] = round(time.time() - t0, 1)
        results[s] = res
        timelines[s] = res["plan"]["timeline"]
        for n in res.get("notes") or []:
            notes.append(f"{settings[s.lower()]}: {n}")
    prog(0.97, "weaving the rounds")
    timeline = VB.weave(windows, timelines)
    names = {"A": a, "B": b}
    share = VB.screen_time(windows)
    stats = {s: [("Videos", str(len(v))), ("Hair", hair_line(v)),
                 *([("Theme", tracks[s][:28])] if mode == "solo" else [])]
             for s, v in (("A", va), ("B", vb))}
    plan = dict(results["A"]["plan"])
    plan.update({
        "track_id": ta, "matching_mode": "Verses", "track_duration_sec": round(total, 3),
        "track_source_path": results["A"]["plan"].get("track_source_path", ""),
        "selected_videos": va + vb, "split_screen_enabled": any(w["side"] == "split" for w in windows),
        "max_clips": 2, "timeline": timeline,
        "verses": {
            "actors": {"A": {"name": a, "videos": va}, "B": {"name": b, "videos": vb}},
            "music": {"mode": mode, "tracks": tracks, "auto": not (settings.get("track_a") or "")},
            "parts": VB.audio_parts(windows, tracks, total, mode, durs),
            "fade_out": fade,
            "windows": windows, "meter": VB.meter(windows, winner), "winner": winner,
            "winner_setting": settings.get("winner", "surprise"), "first": first,
            "layout": settings.get("layout", "mixed"), "total_seconds": round(total, 3),
            "overlay": {"on": bool(settings.get("overlay", True))},
            "intro": {"on": bool(settings.get("intro", True)), "stats": stats,
                      "thumbs": {s: _thumb(timeline, s) for s in ("A", "B")}},
            "screen_time": {s: round(v, 1) for s, v in share.items()},
            "seed": seed, "draft": {"how": settings.get("draft", "choice"), "style": settings.get("style", "any")},
        },
    })
    import core.story_matcher as SMX
    return {"plan": plan, "summary": SMX.plan_summary(timeline), "notes": notes, "names": names,
            "rounds": VB.describe(windows), "tracks": tracks, "video_ids": va + vb, "took": took,
            "shortfall": sum(r.get("shortfall") or 0 for r in results.values()), "made": time.strftime("%H:%M:%S")}


# ---------------------------------------------------------------------------
# Running it on the server (finishes even when the phone leaves the page)
# ---------------------------------------------------------------------------

def _settings() -> dict:
    ss = st.session_state
    return {"a": ss.get("vs_a") or "", "b": ss.get("vs_b") or "", "draft": ss.get("vs_draft", "random"),
            "style": ss.get("vs_style", "any"), "music": ss.get("vs_music", "shared"),
            "track_a": "" if ss.get("vs_auto_music", True) else (ss.get("vs_track_a") or ""),
            "track_b": "" if ss.get("vs_auto_music", True) else (ss.get("vs_track_b") or ""),
            "layout": ss.get("vs_layout", "mixed"), "winner": ss.get("vs_winner", "surprise"),
            "length": int(ss.get("vs_length", 0) or 0), "max_videos": int(ss.get("vs_max_videos", 0) or 0),
            "intro": bool(ss.get("vs_intro", True)), "overlay": bool(ss.get("vs_overlay", True)),
            "stream": bool(ss.get("vs_stream", True)), "quick": bool(ss.get("vs_quick", True)),
            "label": " ".join(str(ss.get("vs_render_name") or "").split())[:80]}


def _render(res: dict, settings: dict, token, prog) -> tuple:
    import render_jobs as RJ
    plan = res["plan"]
    v = plan["verses"]
    extra = [PC._load_track(t).get("source_path", "") for t in set(v["music"]["tracks"].values())]
    return RJ.render(plan, stream=settings["stream"], quick=settings["quick"], label=settings.get("label", ""),
                     token=token, prog=prog, extra_raw=[x for x in extra if x],
                     extra_meta={"project_name": f"⚔️ {v['actors']['A']['name']} vs {v['actors']['B']['name']}",
                                 "compile_note": used_line(res) + ("" if not res.get("took") else " · compile " + ", ".join(
                                     f"{k} {format_mmss(v)}" for k, v in res["took"].items())),
                                 "verses_setup": {k: x for k, x in settings.items() if k != "label"}})


def _job(job, settings: dict, seed: int, render: bool, token, res: dict = None) -> dict:
    out = {}
    if res is None:
        share = 0.4 if render else 1.0
        res = compile_verses(settings, seed, lambda f, txt: job.progress(share * min(f, 1.0), f"compiling · {txt}"))
    out["res"] = res
    if render:
        base = job.frac
        out["note"], out["render"] = _render(res, settings, token,
                                             lambda f, txt: job.progress(base + (1 - base) * min(f, 1.0), txt))
    return out


def _start(render: bool, new_seed: bool = True, res: dict = None) -> None:
    import render_jobs as RJ
    from page_setup import drive_token
    ss = st.session_state
    if res is None and (not ss.get("vs_a") or not ss.get("vs_b") or ss.get("vs_a") == ss.get("vs_b")):
        ss["vs_msg"] = ("warning", "Pick (or draw) two different fighters first.")
        return
    if new_seed or "vs_seed" not in ss:
        ss["vs_seed"] = random.randrange(1, 10 ** 6)
    label = "🎬 Rendering" if res is not None else ("⚔️ Compiling and rendering" if render else "⚔️ Compiling")
    if RJ.start(JOB_KEY, label, _job, _settings(), int(ss["vs_seed"]), render, drive_token(), res) is None:
        ss["vs_msg"] = ("warning", "A compile / render is already running — it is shown below.")
        return
    ss["vs_job_seen"] = None


def _take_finished_job() -> bool:
    import render_jobs as RJ
    j, running = RJ.take_finished(JOB_KEY, "vs_job_seen")
    if j is None:
        return running
    ss = st.session_state
    if j.state == "error":
        ss["vs_msg"] = ("error", f"{j.label} failed: {j.error}")
        return False
    got = j.result or {}
    if got.get("res") is not None:
        ss[RESULT_KEY] = got["res"]
    if got.get("render"):
        ss["vs_render"] = got["render"]
        ss["vs_msg"] = got.get("note")
    elif got.get("res") is not None:
        ss.pop("vs_render", None)
    return False


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def used_line(res: dict) -> str:
    v = res["plan"]["verses"]
    n = res["names"]
    music = (f"🤝 {v['music']['tracks']['A']}" if v["music"]["mode"] == "shared"
             else f"🎵 {v['music']['tracks']['A']} / {v['music']['tracks']['B']}")
    win = v["winner"]
    st_ = v.get("screen_time") or {}
    tot = max(sum(st_.values()), 1e-6)
    return " · ".join([f"⚔️ {n['A']} vs {n['B']}", music + (" (auto)" if v["music"].get("auto") else ""),
                       format_mmss(v["total_seconds"]), VB.LAYOUTS[v["layout"]],
                       "🤝 draw" if win == "draw" else f"🏆 {n[win]}",
                       f"screen time {st_.get('A', 0) / tot:.0%} / {st_.get('B', 0) / tot:.0%}"])


def _draw(first: str = None) -> None:
    ss = st.session_state
    pool = fighters_pool()
    names = VB.eligible_actors(pool)
    style = ss.get("vs_style", "any")
    try:
        a, b = VB.draft(names, style, random.Random(), first=first, looks=looks(pool) if style != "any" else None)
    except ValueError as e:
        ss["vs_msg"] = ("warning", str(e))
        return
    ss["vs_a"], ss["vs_b"] = a, b
    ss.pop(RESULT_KEY, None)


def _fighter_card(col, side: str, name: str, pool: dict, other_name: str) -> None:
    with col.container(border=True):
        if not name:
            st.caption("— not picked yet —")
            return
        vids = VB.fighter_videos(pool, name, other_name)[0] if other_name else (pool.get(name) or [])
        colour = "#%02x%02x%02x" % VB.COLOURS[side]
        st.markdown(f"<span style='color:{colour};font-weight:700;font-size:1.15rem'>{name}</span>",
                    unsafe_allow_html=True)
        st.caption(f"🎬 {len(vids)} video{'s' if len(vids) != 1 else ''} · 💇 {hair_line(vids)}")


def _setup() -> None:
    ss = st.session_state
    st.caption("⚔️ Two actors battle for your attention over four rounds — 🥊 Entrances, 🔁 Trade-offs (swaps on the "
               "bar, getting faster), ◧ Head to head (split screen) and 🏁 the Final (a swap every bar, then the "
               "winner's spotlight). Each fighter's footage is matched to her music by the 🤖 new matcher.")
    pool = fighters_pool()
    names = VB.eligible_actors(pool)
    if len(names) < 2:
        st.warning("At least two actors need videos with a 🧭 footage timeline — set actors on videos in 📁 Media Library.")
        return
    for k in ("vs_a", "vs_b"):
        if ss.get(k) and ss[k] not in names:
            ss[k] = ""
    st.markdown("**🥊 The fighters**")
    c = st.columns([2, 2.4])
    how = c[0].radio("Draft", ["random", "choice"], key="vs_draft", horizontal=True,
                     format_func=lambda x: {"random": "🎲 Random draft", "choice": "👆 Your choice"}[x],
                     label_visibility="collapsed")
    c[1].radio("Style", list(VB.DRAFT_STYLES), key="vs_style", horizontal=True, format_func=VB.DRAFT_STYLES.get,
               label_visibility="collapsed",
               help="How the draft pairs them: any two at random; ⚡ the least alike (their footage looks most "
                    "different); 🪞 the most alike — a close rivalry. Used by 🎲 Draw and 🎯 Find her a rival.")
    if how == "random":
        st.button("🎲 Draw two fighters", key="vs_draw_btn", type="primary", on_click=_draw)
    else:
        p = st.columns([2, 2, 1.4])
        opts = [""] + names
        p[0].selectbox("Fighter 1", opts, key="vs_a", format_func=lambda x: x or "— pick —")
        p[1].selectbox("Fighter 2", opts, key="vs_b", format_func=lambda x: x or "— pick —")
        p[2].button("🎯 Find her a rival", key="vs_rival_btn", width="stretch", disabled=not ss.get("vs_a"),
                    on_click=lambda: _draw(first=ss.get("vs_a")),
                    help="Keeps fighter 1 and drafts fighter 2 with the style above.")
    a, b = ss.get("vs_a") or "", ss.get("vs_b") or ""
    f = st.columns([5, 1, 5])
    _fighter_card(f[0], "A", a, pool, b)
    f[1].markdown("<div style='text-align:center;font-weight:800;font-size:1.4rem;padding-top:1.2rem'>VS</div>",
                  unsafe_allow_html=True)
    _fighter_card(f[2], "B", b, pool, a)

    st.markdown("**🎵 The music**")
    m = st.columns([2.4, 1.6])
    mode = m[0].radio("Music", list(VB.MUSIC_MODES), key="vs_music", horizontal=True, format_func=VB.MUSIC_MODES.get,
                      label_visibility="collapsed",
                      help="🤝 one song both fighters are matched to. 🎵 each has her own song — both run from the start "
                           "like two decks, and you hear whoever is on screen (in a split, whoever leads).")
    auto = m[1].toggle("🤖 Auto-pick", key="vs_auto_music",
                       help="The song(s) best suited to both fighters' footage (🤝: the best worse-of-the-two fit; "
                            "🎵: each one's best, the second near the first's tempo).")
    if not auto:
        summaries = PC.audio_summaries(PC._audio_signature())
        tids = [""] + sorted(summaries, key=str.lower)
        for k in ("vs_track_a", "vs_track_b"):
            if ss.get(k) and ss[k] not in summaries:
                ss[k] = ""
        fmt = (lambda t: f"{t} · {format_mmss(summaries[t]['duration'])}"
               + (f" · {summaries[t]['bpm']:.0f} BPM" if summaries[t].get("bpm") else "") if t else "— pick —")
        t = st.columns(2)
        t[0].selectbox("🎵 Song" if mode == "shared" else f"🎵 {a or 'Fighter 1'}'s theme", tids, key="vs_track_a",
                       format_func=fmt)
        if mode == "solo":
            t[1].selectbox(f"🎵 {b or 'Fighter 2'}'s theme", tids, key="vs_track_b", format_func=fmt)

    st.markdown("**🖼️ The battle**")
    g = st.columns([2.6, 1.8, 1.4])
    g[0].radio("Layout", list(VB.LAYOUTS), key="vs_layout", format_func=VB.LAYOUTS.get)
    g[1].radio("Ending", list(VB.WINNERS), key="vs_winner",
               format_func=lambda w: VB.WINNERS[w].replace("First actor", a or "Fighter 1").replace("Second actor", b or "Fighter 2"))
    g[2].selectbox("⏱️ Length", list(VB.LENGTHS), key="vs_length", format_func=VB.LENGTHS.get,
                   help="Shorter than the song: the battle ends there and the music fades out.")
    o = st.columns(3)
    o[0].toggle("🥊 Tale-of-the-tape intro", key="vs_intro", help="A 4-second card: both fighters, VS, a few facts.")
    o[1].toggle("📊 Battle overlay", key="vs_overlay",
                help="Names, VS, the round, a tug-of-war meter of screen time, and the winner at the end.")
    o[2].number_input("Most videos each (0 = any)", 0, 40, key="vs_max_videos")
    b_ = st.columns(2)
    ready = bool(a and b and a != b) and (auto or (ss.get("vs_track_a") and (mode == "shared" or ss.get("vs_track_b"))))
    b_[0].button("⚔️ Compile & render", key="vs_go", type="primary", width="stretch", disabled=not ready,
                 on_click=_start, args=(True,))
    b_[1].button("⚔️ Compile only", key="vs_compile", width="stretch", disabled=not ready, on_click=_start, args=(False,))
    if not ready:
        st.caption("Pick two fighters" + ("" if auto else " and the song(s)") + " to start.")


def _strip(res: dict) -> None:
    import plotly.graph_objects as go
    v = res["plan"]["verses"]
    n = res["names"]
    fig = go.Figure()
    for w in v["windows"]:
        col = "rgba(%d,%d,%d,.85)" % VB.COLOURS[w["side"]] if w["side"] in ("A", "B") else "rgba(170,120,230,.85)"
        who = n[w["side"]] if w["side"] in ("A", "B") else "split · " + (f"{n[w['lead']]} leads" if w.get("lead") else "even")
        fig.add_trace(go.Bar(x=[w["end"] - w["start"]], base=[w["start"]], y=["⚔️ battle"], orientation="h", width=0.8,
                             marker=dict(color=col, line=dict(color="rgba(255,255,255,.6)", width=0.5)), showlegend=False,
                             hovertemplate=f"Round {w['round']} · {w['round_name']}<br>{who}<br>"
                                           f"{format_mmss(w['start'])}–{format_mmss(w['end'])}<extra></extra>"))
    for p in v["parts"]:
        fig.add_trace(go.Bar(x=[p["dur"]], base=[p["at"]], y=["🎵 music"], orientation="h", width=0.8,
                             marker=dict(color="rgba(%d,%d,%d,.45)" % VB.COLOURS[p["side"]]), showlegend=False,
                             hovertemplate=f"{p['track_id']}<extra></extra>"))
    fig.update_layout(barmode="overlay", height=120, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[0, float(v["total_seconds"])], tickfont_size=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"vs_strip_{res['made']}")


def _save(res: dict) -> None:
    import render_jobs as RJ
    from config import PLANS_DIR
    from page_setup import DRIVE_SESSION_KEY, OAUTH_AVAILABLE, push_file_with_oauth
    plan = res["plan"]
    Path(PLANS_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(PLANS_DIR) / f"verses_{plan['track_id']}_plan.json"
    out.write_text(RJ.plan_json(plan))
    if OAUTH_AVAILABLE and st.session_state.get(DRIVE_SESSION_KEY):
        err = push_file_with_oauth(st.session_state[DRIVE_SESSION_KEY], out, "scene-labeling/compilation_plans")
        st.session_state["vs_msg"] = ("warning", f"Saved locally but Drive upload failed: {err}") if err else \
            ("success", f"Saved and uploaded to Drive: {out.name}")
    else:
        st.session_state["vs_msg"] = ("success", f"Saved locally: {out.name}")


def _result(res: dict) -> None:
    import render_jobs as RJ
    from config import PREVIEW_DIR
    plan, summary = res["plan"], res["summary"]
    c = st.columns([5, 1.4])
    c[0].subheader(f"⚔️ {res['names']['A']} vs {res['names']['B']}")
    c[1].button("← Change", key="vs_back", width="stretch", on_click=lambda: st.session_state.pop(RESULT_KEY, None))
    st.caption(used_line(res) + f" · made {res['made']}")
    st.caption(f"{len(plan['timeline'])} clips · {summary.get('videos', 0)} different videos"
               + (f" · 🎯 movement sync {summary['sync']:.0%}" if summary.get("sync") is not None else "")
               + (f" · ⚠️ {res['shortfall']} slot(s) unfilled" if res.get("shortfall") else ""))
    if res.get("took"):                                   # ⏱️ where the compile's time went
        st.caption("⏱️ Compile: " + " · ".join(f"{'🎵 music pick' if k == 'music' else '🎯 ' + k} {format_mmss(v)}"
                                               for k, v in res["took"].items()))
    _strip(res)
    for line in res["rounds"]:
        st.caption("🥊 " + line)
    for n in res.get("notes") or []:
        st.caption(n)
    st.text_input("🏷️ Name the next render (optional)", key="vs_render_name", max_chars=80)
    b = st.columns([1.8, 0.5, 1.4, 1.4, 1.6])
    b[0].button("🎬 Render", key="vs_render_btn", type="primary", width="stretch", on_click=_start,
                args=(True, False, res))
    with b[1].popover("⚙️", width="stretch", help="Render options"):
        st.toggle("⚡ Quick preview — use the 480p copies on the server", key="vs_quick")
        st.toggle("Stream original videos from Google Drive", key="vs_stream")
    b[2].button("💾 Save plan", key="vs_save", width="stretch", on_click=_save, args=(res,))
    b[3].download_button("⬇️ Download", data=RJ.plan_json(plan), file_name=f"verses_{plan['track_id']}_plan.json",
                         mime="application/json", width="stretch", key="vs_dl")
    b[4].button("↻ Rematch", key="vs_again", width="stretch", on_click=_start, args=(True,),
                help="Same fighters and setup, a fresh draw (who opens, a surprise winner) — compiled and rendered.")
    if name := st.session_state.get("vs_render"):
        from library_common import show_render_player
        from renders import load_render
        meta = load_render(Path(PREVIEW_DIR) / name)
        if meta:
            show_render_player(meta)
        else:
            st.caption(f"{name} — see 📁 Media Library → 🎞️ Renders.")


SETUP_KEYS = {"a": "vs_a", "b": "vs_b", "draft": "vs_draft", "style": "vs_style", "music": "vs_music",
              "track_a": "vs_track_a", "track_b": "vs_track_b", "layout": "vs_layout", "winner": "vs_winner",
              "length": "vs_length", "max_videos": "vs_max_videos", "intro": "vs_intro", "overlay": "vs_overlay",
              "quick": "vs_quick", "stream": "vs_stream"}


def reopen(ro: dict) -> None:
    """↩️ A battle render reopened (📁 Media Library → Renders, 🔮 → 🌙 Overnight): the setup it was made with, and
    its plan on show as the result (🎬 Render again, ⬇️ Download, ↻ Rematch)."""
    import core.story_matcher as SMX
    plan = ro["plan"]
    v = plan["verses"]
    setup = dict(ro.get("setup") or {})
    if not setup:                                          # worked out from the plan
        setup = {"a": v["actors"]["A"]["name"], "b": v["actors"]["B"]["name"], "music": v["music"]["mode"],
                 "layout": v.get("layout", "mixed"), "winner": v.get("winner_setting", "surprise"),
                 "intro": bool((v.get("intro") or {}).get("on", True)), "overlay": bool((v.get("overlay") or {}).get("on", True))}
        if not v["music"].get("auto"):
            setup.update(track_a=v["music"]["tracks"]["A"], track_b=v["music"]["tracks"]["B"])
    ss = st.session_state
    for k, key in SETUP_KEYS.items():
        if k in setup:
            ss[key] = setup[k]
    ss["vs_auto_music"] = not (setup.get("track_a") or "")
    if setup.get("a") and setup.get("b"):
        ss["vs_draft"] = "choice"                         # the same two fighters, not a new draw
    if v.get("seed") is not None:
        ss["vs_seed"] = int(v["seed"])
    ss[RESULT_KEY] = {
        "plan": plan, "summary": SMX.plan_summary(plan.get("timeline") or []), "shortfall": 0,
        "notes": [f"↩️ Reopened from render {ro.get('label', '')} — the plan as it was rendered; ↻ Rematch draws again."],
        "names": {s: v["actors"][s]["name"] for s in ("A", "B")}, "rounds": VB.describe(v["windows"]),
        "tracks": dict(v["music"]["tracks"]), "video_ids": list(plan.get("selected_videos") or []),
        "made": time.strftime("%H:%M:%S")}
    ss.pop("vs_render", None)


def render() -> None:
    import render_jobs as RJ
    import settings_store as SSTORE
    SSTORE.keep(KEPT)
    ro = st.session_state.pop("_reopen_verses", None)
    if ro:
        reopen(ro)
    msg = st.session_state.pop("vs_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    if _take_finished_job():
        RJ.watch(JOB_KEY)
        return
    msg = st.session_state.pop("vs_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    res = st.session_state.get(RESULT_KEY)
    if res:
        _result(res)
        return
    _setup()
