"""
🔮 Recommendations

  🎵 For this track — five plans for a track (📐 Best shape fit, 🎭 Best story fit,
     🏆 Best overall, 🥊 Verses, 🎲 Wildcard), searched across the whole library and
     checked with the real matcher (core/recommend.py).
  Feeds from the background searches (jobs.py — nightly, only what's changed, or on
  demand): 🏆 Top plans · 💥 Best action · 🆕 New videos · 🧩 Hidden gems · 🕳️ Gaps
  (core/feeds.py).

Every plan opens straight into the planner. The library side lives in rec_library.py,
shared with the background worker.
"""

import datetime
import time

import plotly.graph_objects as go
import streamlit as st

import catalogue_store as CStore
import core.feeds as FEEDS
import core.recommend as REC
import jobs as J
import rec_library as RL
import scene_roles as SR
import core.footage as CFOOT
from library_common import library_ranges, master_thumbnail_path, safe_image, scene_thumbnail_path
from page_setup import page_setup, top_button, wrap_css

page_setup("Recommendations", "pages/5_Recommendations.py", title="🔮 Recommendations", anchor="recs-top")
top_button("recs-top", wrap_css("rc_wrap", 130))   # rows wrap on narrow screens (iPad portrait)

DEFAULT_VIDEOS = RL.DEFAULT_VIDEOS
FEED_SIZE = 10


# ---------------------------------------------------------------------------
# Opening a plan in the planner
# ---------------------------------------------------------------------------

def _mode_label(mode: str) -> str:
    return REC.MODE_LABELS.get(mode) or ("🆕 Spotlight" if mode == "spotlight" else mode)


def open_plan(track_id: str, plan: dict, rank: int) -> None:
    settings = {"segmentation_method": plan["method"], "beats_per_bar": 4, **plan["settings"]}
    st.session_state["_open_recommended_plan"] = {
        "track_id": track_id, "settings": settings, "videos": list(plan["videos"]),
        "story": bool(plan.get("story")), "story_regions": plan.get("story_regions"),
        "sequential": bool(plan.get("sequential")),
        "label": f"Recommended plan {rank} for {track_id} ({_mode_label(plan['mode'])}) — {plan['variant']}, "
                 f"{plan['scores']['overall']:.0f}% overall"}
    st.switch_page("pages/1_Compilation_Planner.py")


# ---------------------------------------------------------------------------
# Drawing plans
# ---------------------------------------------------------------------------

def _strip(scores: dict, which: str = "overall") -> go.Figure:
    """Per-block bars: "overall" (combined), "shape", or "story" (role blocks only: hit 100 / miss 0)."""
    vals = {"shape": scores.get("per_block_shape"), "story": scores.get("per_block_story")}.get(which) \
        or scores["per_block"]
    shown = [0.0 if v is None else v for v in vals]
    act = set(scores["action_blocks"]) if which == "overall" else set()
    colours = ["rgba(128,128,128,0.25)" if v is None else "#2e9e5b" if v >= 70 else "#e8a317" if v >= 50
               else "#d9534f" for v in vals]
    if which == "story":
        shown = [100.0 if v is None else v for v in vals]      # no-role blocks as faint full bars
        colours = ["rgba(128,128,128,0.18)" if v is None else "#2e9e5b" if v else "#d9534f" for v in vals]
    hover = "Block %{x}: %{customdata}<extra></extra>"
    custom = ["no story part" if v is None else (("✅ role clip" if v else "❌ not a role clip") if which == "story"
                                                 else f"{v:.0f}%") for v in vals]
    fig = go.Figure(go.Bar(x=list(range(1, len(vals) + 1)), y=shown, marker_color=colours, customdata=custom,
                           marker_line_color=["#222" if i in act else "rgba(0,0,0,0)" for i in range(len(vals))],
                           marker_line_width=[2 if i in act else 0 for i in range(len(vals))],
                           hovertemplate=hover))
    if which == "overall" and scores.get("weakest_block") is not None:
        fig.add_annotation(x=scores["weakest_block"] + 1, y=vals[scores["weakest_block"]], text="▼",
                           showarrow=False, yshift=10, font=dict(size=12),
                           hovertext=f"Weakest block ({scores['weakest_block'] + 1})")
    fig.update_layout(height=105, margin=dict(t=8, b=4, l=4, r=4), yaxis=dict(range=[0, 100], title=None),
                      xaxis=dict(title=None, showticklabels=False), showlegend=False)
    return fig


def _thumb(video_id: str):
    """The video's picture, as the planner's video list chooses it: the ⭐ master thumbnail
    chosen in the Reviewer, else the usable scene (not excluded, inside the library range)
    nearest the middle of the range."""
    scenes = (CStore.all_catalogues().get(video_id) or {}).get("scenes", [])
    master = master_thumbnail_path(video_id, scenes)
    if master is not None:
        return master
    rng = library_ranges().get(video_id)
    usable = []
    for s in scenes:
        if s.get("excluded"):
            continue
        try:
            a, b = CFOOT._overlap_with_range(CFOOT._tc_to_seconds(s["start_tc"]), CFOOT._tc_to_seconds(s["end_tc"]),
                                             rng)
        except (KeyError, ValueError):
            continue
        if a is not None:
            usable.append((a, b, s))
    if not usable:
        return None
    mid = (usable[0][0] + usable[-1][1]) / 2 if not rng else (rng[0] + rng[1]) / 2
    for _, _, s in sorted(usable, key=lambda u: 0.0 if u[0] <= mid <= u[1] else min(abs(u[0] - mid),
                                                                                     abs(u[1] - mid))):
        p = scene_thumbnail_path(video_id, s)
        if p:
            return p
    return None


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0f}%"


HELPS = {"overall": "⅓ shape + ⅓ motion + ⅓ story (½ + ½ when the track has no story parts).",
         "shape": "How closely each clip's motion rises and falls with its block's music — every block counts "
                  "equally.",
         "intensity": f"How well each clip's motion level matches its block's intensity — drops and action parts "
                      f"count {REC.DROP_WEIGHT:g}×.",
         "story": "Story-map parts (🎬 Intro / 🌙 Slow / 💥 Action / 🏁 Outro) whose clip is marked with that role. "
                  "— = no story parts in this track.",
         "action": "Motion match over the drops and action parts only.",
         "action_story": "Story-map 💥 Action parts that got a clip marked 💥 Action."}
NAMES = {"overall": "Overall", "shape": "Shape", "intensity": "Motion", "story": "Story", "action": "Action motion",
         "action_story": "Action clips"}


def _badge(name: str, v) -> str:
    colour = "gray" if v is None else "green" if v >= 70 else "orange" if v >= 50 else "red"
    return f":{colour}-badge[{name} {_pct(v)}]"


def _metrics(sc: dict, keys: list, about: str = "", about_help: str = "") -> None:
    """The plan's scores as coloured badges (green ≥ 70 %, orange ≥ 50 %, red below), then what the
    plan is (style, videos…) on the same line; what each score means is in the tooltip."""
    st.markdown(" ".join(_badge(NAMES[k], sc.get(k)) for k in keys) + (f"  :gray[{about}]" if about else ""),
                help="  \n".join(f"**{NAMES[k]}** — {HELPS[k]}" for k in keys)
                + (f"  \n\n{about_help}" if about_help else ""))


def _videos_row(videos: list, small: bool = False) -> None:
    cols = st.columns(max(len(videos), 6) if small else (min(len(videos), 8) or 1))
    for c, vid in zip(cols * 2, videos):
        with c:
            t = _thumb(vid)
            if t:
                safe_image(str(t), width="stretch")
            st.caption(vid)


def render_result(track_id: str, rank: int, result: dict, data: dict) -> None:
    mode = result["mode"]
    plan = result.get("plan")
    with st.container(border=True, key=f"rc_wrap_card_{mode}"):
        h = st.columns([3, 1], vertical_alignment="center")
        h[0].markdown(f"**{rank} · {REC.MODE_LABELS[mode]}**")
        if not plan:
            h[0].caption(REC.MODE_HELP[mode])
            st.info(result.get("note") or "No plan found.")
            return
        sc = plan["scores"]
        bits = [plan["variant"], f"{len(plan['videos'])} videos"]
        if plan.get("story"):
            bits.append("🎭 story map on")
        if plan.get("sequential"):
            bits.append("in order")
        open_now = h[1].button("🎬 Open", key=f"open_{track_id}_{mode}", type="primary", width="stretch")
        if mode == "verses" and plan.get("actors"):
            a, b = (", ".join(plan["actors"].get(v) or ["?"]) for v in plan["videos"][:2])
            st.markdown(f"**👤 {a}  vs  👤 {b}**")
        # What each card leads with: 📐 shape, 🎭 story, the rest overall + the three parts.
        lead = {"shape": "shape", "story": "story"}.get(mode, "overall")
        _metrics(sc, [lead] if mode in ("shape", "story") else ["overall", "shape", "intensity", "story"],
                 about=" · ".join(bits), about_help=REC.MODE_HELP[mode])
        st.plotly_chart(_strip(sc, lead), width="stretch", key=f"strip_{track_id}_{mode}",
                        config={"displayModeBar": False})
        _videos_row(plan["videos"])
        if mode == "wildcard":
            st.button("🎲 Roll again", key=f"reroll_{track_id}", width="stretch",
                      help="A new random cutting style and random videos (just this card).",
                      on_click=lambda: st.session_state.__setitem__("_rec_reroll", True))
    if open_now:
        open_plan(track_id, plan, rank)


def feed_card(key: str, rank: int, track_id: str, plan: dict, metrics: list, extra: str = "") -> None:
    """A compact plan card for the feeds."""
    with st.container(border=True, key=f"rc_wrap_feed_{key}"):
        h = st.columns([3, 1], vertical_alignment="center")
        h[0].markdown(f"**{rank} · 🎵 {track_id}**" + (f"  ·  {extra}" if extra else ""))
        open_now = h[1].button("🎬 Open", key=f"open_{key}", type="primary", width="stretch")
        _metrics(plan["scores"], metrics,
                 about=f"{_mode_label(plan['mode'])} · {plan['variant']} · {len(plan['videos'])} videos"
                 + (" · 🎭 story map on" if plan.get("story") else ""))
        _videos_row(plan["videos"], small=True)
    if open_now:
        open_plan(track_id, plan, rank)


# ---------------------------------------------------------------------------
# ⏳ Background searches
# ---------------------------------------------------------------------------

def _ago(t) -> str:
    if not t:
        return "never"
    mins = (time.time() - float(t)) / 60
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins:.0f} min ago"
    if mins < 48 * 60:
        return f"{mins / 60:.0f} h ago"
    return f"{mins / 1440:.0f} days ago"


def _queue_changed() -> None:
    n = J.queue_changed(why="by hand")
    msg = J.start_worker() if n else ""
    st.session_state["_rec_bg_msg"] = (f"⏳ Queued {n} search(es) — {msg}." if n else
                                       "✅ Everything is up to date — nothing to search.")


@st.fragment(run_every=5)
def background_panel() -> None:
    status = J.read_status()
    running = [v for v in status.values() if v.get("state") == "running"]
    queued = [v for v in status.values() if v.get("state") == "queued"]
    errors = [v for v in status.values() if v.get("state") == "error"]
    night = J.nightly_status()
    done = [v for v in status.values() if v.get("state") == "done"]
    if running:
        # Overall progress: finished / all, and the time left from how often searches have been finishing.
        total = len(done) + len(running) + len(queued)
        head = f"⏳ Background searches — {len(done)} of {total} done · {len(running)} running"
        times = sorted(v.get("updated", 0) for v in done)
        gaps = sorted(b - a for a, b in zip(times, times[1:]))
        if gaps:
            per = gaps[len(gaps) // 2]      # typical time between finishes (an older batch's gap doesn't skew it)
            left_min = (len(running) + len(queued)) * per / 60
            head += (f" · ~{left_min:.0f} min left" if left_min < 90 else f" · ~{left_min / 60:.1f} h left")
    elif queued:
        head = f"⏳ Background searches — {len(queued)} queued" + (" (paused)" if J.is_paused() else "")
    else:
        head = f"⏳ Background searches — idle · last nightly run {_ago(night.get('started'))}"
    with st.expander(head, expanded=False):
        st.caption("Every night at 04:45 the server searches the tracks whose plans are out of date (new or "
                   "changed videos, ranges, actors, roles or story maps) with the default settings — "
                   f"{DEFAULT_VIDEOS} videos, normal order — then builds the 🆕 new-video spotlights. Audio "
                   f"analysis always goes first; up to {J.worker_count()} searches run side by side (each in its own process, "
                   "at low priority so the app stays quick).")
        if night:
            st.caption(f"Last nightly run: {_ago(night.get('started'))} — {night.get('queued_tracks', 0)} track(s), "
                       f"{night.get('queued_spotlights', 0)} spotlight(s).")
        b = st.columns(4)
        b[0].button("⏳ Search changed tracks now", on_click=_queue_changed, width="stretch", key="rec_bg_changed",
                    help="Queue every track whose plans are out of date (and new-video spotlights) and start.")
        if J.is_paused():
            b[1].button("▶️ Resume", on_click=J.resume, width="stretch", key="rec_bg_resume")
        else:
            b[1].button("⏸️ Pause after these", on_click=J.pause, width="stretch", key="rec_bg_pause",
                        disabled=not (running or queued),
                        help="Lets the searches running now finish (a minute or so each), then starts no more. "
                             "⏹️ Stop now ends them at once.")
        b[2].button("⏹️ Stop now", on_click=J.stop_now, width="stretch", key="rec_bg_stop",
                    disabled=not running, help="Stop at once; the current search starts again on resume.")
        b[3].button("🗑️ Clear the queue", on_click=J.cancel_queued, width="stretch", key="rec_bg_clear",
                    disabled=not queued)
        msg = st.session_state.pop("_rec_bg_msg", None)
        if msg:
            st.info(msg)
        rows = sorted(status.values(), key=lambda v: ({"running": 0, "queued": 1, "error": 2}.get(v.get("state"), 3),
                                                       -v.get("updated", 0)))
        for v in rows[:12]:
            icon = {"running": "⏳", "queued": "🕒", "done": "✅", "error": "⚠️"}.get(v.get("state"), "•")
            line = f"{icon} {v.get('label')} — {v.get('msg', '')}"
            if v.get("state") == "running":
                if v.get("started"):
                    line += f" · {(time.time() - v['started']) / 60:.0f} min so far"
                st.progress(min(1.0, float(v.get("progress", 0))), text=line)
            else:
                st.caption(line + (f" ({_ago(v.get('updated'))})" if v.get("state") in ("done", "error") else ""))
        if errors or any(v.get("state") == "done" for v in status.values()):
            st.button("Clear finished", on_click=J.clear_finished, key="rec_bg_clear_done")


# ---------------------------------------------------------------------------
# 🎵 For this track
# ---------------------------------------------------------------------------

def for_this_track(tracks: list) -> None:
    c = st.container(key="rc_wrap_controls").columns([3, 2, 2])
    default_track = st.session_state.get("track_id") if st.session_state.get("track_id") in tracks else tracks[0]
    track_id = c[0].selectbox("🎵 Track", tracks, index=tracks.index(default_track), key="rec_track")
    n_videos = c[1].slider("Videos per plan", 2, 15, DEFAULT_VIDEOS, key="rec_n_videos",
                           help="How many videos each plan uses (as in the planner's video selection). 🥊 Verses "
                                "always uses two.")
    sequential = c[1].toggle("Sequential mode", key="rec_sequential",
                             help="Keep each video's clips in their original order (the planner's 'Keep each "
                                  "video's clips in order'). Plans open with it on.")
    track = RL.load_track(track_id)
    story_regions, story_source = RL.story_for(track_id, track)
    sig = RL.signature(track_id, n_videos, sequential, story_regions)
    data = RL.load_results(track_id, n_videos, sequential)
    fresh = RL.staleness(data, sig)[0] != "stale"      # other videos changing doesn't make it out of date
    go_btn = c[2].button("🔍 Find plans" if not data else "🔄 Search again", type="primary", width="stretch")
    if c[2].button("⏳ In the background", width="stretch", key="rec_bg_one",
                   help="Queue this search for the background worker instead — carry on using the app; the plans "
                        "appear here when it's done."):
        J.enqueue([{"kind": "track", "track_id": track_id, "n_videos": n_videos, "sequential": sequential}],
                  why="by hand")
        st.toast(f"⏳ {track_id}: {J.start_worker()}")

    if go_btn:
        bar = st.progress(0.0, text="Reading your library…")
        try:
            from diagnostics import timed
            with timed("Recommendations: for this track", track_id):
                data = RL.run_track(track_id, n_videos, sequential,
                                    progress=lambda f, t: bar.progress(min(1.0, max(0.0, f)), text=t))
            fresh = True
        except RuntimeError as e:
            bar.empty()
            st.warning(str(e))
            return
        finally:
            bar.empty()

    if data and st.session_state.pop("_rec_reroll", False):
        with st.spinner("🎲 Rolling again…"):
            data = RL.reroll_wildcard(data)

    if not data:
        st.info("Pick a track and press **🔍 Find plans** (or ⏳ queue it in the background). The first search reads "
                "every video's motion, so it takes a little while; after that it's quicker.")
        return
    st.session_state.setdefault("rec_results", {})[(track_id, int(n_videos), bool(sequential))] = data
    # One status line: when they were found (or that they're out of date) and which story map scores them.
    bits = [f"Found {_ago(_ts(data.get('finished_at')))}" if fresh else
            ":orange-badge[⚠️ Out of date] — the library or the track changed; 🔄 Search again"]
    helps = ["Plans searched across your whole library and checked with the real matcher — the scores are what "
             "you get when you open one. 🎬 Open makes a plan a new project in the planner (what you had there is "
             "saved first)."]
    if story_source == "saved":
        bits.append("🎭 its story map")
        helps.append("Story fit uses this track's story map; its action parts count extra in the motion score.")
    elif story_source == "suggested":
        bits.append("🎭 suggested story map")
        helps.append("This track has no saved story map, so story fit uses the one the story map wizard suggests "
                     "(Media Library → 🎭 Story map to make your own). Plans that use it open with it.")
    st.markdown(" · ".join(bits), help="  \n\n".join(helps))
    for i, result in enumerate(data["results"], start=1):
        render_result(track_id, i, result, data)


def _ts(iso):
    try:
        return datetime.datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------

def _intro(short: str, long: str) -> None:
    """A feed's one-line intro; the full explanation (and where the plans come from) in the tooltip."""
    st.caption(short, help=long + (f"  \n\n{_from}" if _from else ""))


def _no_searches() -> None:
    st.info("No background searches yet. They run every night at 04:45 — or open **⏳ Background searches** above "
            "and press **⏳ Search changed tracks now**.")


def feed_top(results: dict) -> None:
    _intro("The best plans across all your tracks.",
           "The best plans across every track you've got (at most two per track), by overall match.")
    if not results:
        return _no_searches()
    for i, item in enumerate(FEEDS.top_plans(results, FEED_SIZE), start=1):
        feed_card(f"top_{i}", i, item["track_id"], item["plan"], ["overall", "shape", "intensity", "story"])


def feed_action(results: dict) -> None:
    _intro("Plans that land best on the drops and action parts.",
           "Plans whose drops and action parts land best: high-motion footage on the big moments, and 💥 Action "
           "clips in the story map's action parts.")
    if not results:
        return _no_searches()
    items = FEEDS.best_action(results, FEED_SIZE)
    if not items:
        st.info("None of the searched tracks has drops or action parts yet.")
    for i, item in enumerate(items, start=1):
        feed_card(f"act_{i}", i, item["track_id"], item["plan"], ["action", "action_story", "overall"],
                  extra=f"💥 {item['score']:.0f}%")


def feed_new(results: dict) -> None:
    _intro(f"Videos added in the last {RL.NEW_DAYS} days, with plans built around them.",
           f"Videos added in the last {RL.NEW_DAYS} days: the videos most like them, the tracks they fit, and "
           f"a ready-made plan built around each one for its {RL.SPOTLIGHT_TRACKS} best tracks.")
    new = RL.new_videos()
    if not new:
        st.info(f"No new videos in the last {RL.NEW_DAYS} days.")
        return
    for vid, first in new:
        spot = RL.load_spotlight(vid)
        with st.container(border=True):
            h = st.columns([1, 3])
            t = _thumb(vid)
            if t:
                with h[0]:
                    safe_image(str(t), width="stretch")
            h[1].markdown(f"**🆕 {vid}**")
            h[1].caption(f"Added {_ago(_ts(first))}")
            if not spot:
                h[1].caption("Its spotlight is built in tonight's run.")
                if h[1].button("⏳ Build it now (in the background)", key=f"spot_{vid}"):
                    J.enqueue([{"kind": "spotlight", "video_id": vid}], why="by hand")
                    st.toast(f"⏳ {vid}: {J.start_worker()}")
                continue
            if spot.get("similar"):
                h[1].markdown("**Most like it:** " + " · ".join(f"{s['video_id']} ({100 * s['similarity']:.0f}%)"
                                                                for s in spot["similar"]))
            if spot.get("fits"):
                h[1].markdown("**Fits best:** " + " · ".join(f"🎵 {t} ({100 * f:.0f}%)"
                                                              for t, f in list(spot["fits"].items())[:5]))
            for i, plan in enumerate(spot.get("plans") or [], start=1):
                feed_card(f"spot_{vid}_{i}", i, plan["track_id"], plan, ["overall", "shape", "intensity", "story"],
                          extra=f"built around {vid}")


def feed_gems(results: dict) -> None:
    _intro("Good fits you rarely use.",
           "Videos that fit many of your tracks well but that you've rarely used in your recent projects and "
           "renders.")
    if not results:
        return _no_searches()
    gems = FEEDS.hidden_gems(results, RL.video_usage())
    if not gems:
        st.info("No hidden gems — the videos that fit best are the ones you already use.")
    for g in gems:
        with st.container(border=True):
            h = st.columns([1, 3])
            t = _thumb(g["video_id"])
            if t:
                with h[0]:
                    safe_image(str(t), width="stretch")
            h[1].markdown(f"**🧩 {g['video_id']}** · fits {g['fit']:.0f}%")
            h[1].caption(g["why"])
            h[1].markdown("**Best tracks:** " + " · ".join(f"🎵 {t} ({f:.0f}%)" for t, f in g["tracks"]))


def feed_gaps(results: dict) -> None:
    _intro("What your library is short of — ideas for what to film or find next.",
           "The story roles your tracks need more footage for, and kinds of block nothing in the library matches "
           "well.")
    if not results:
        return _no_searches()
    supply = {}
    for roles in RL.role_seconds().values():
        for r, sec in roles.items():
            supply[r] = supply.get(r, 0.0) + sec
    roles = SR.roles_by_id()
    g = FEEDS.gaps(results, supply, {r: SR.label(r, roles) for r in roles})
    st.markdown("**🎭 Story roles**")
    if g["roles"]:
        st.dataframe([{"Role": r["name"], "Tracks using it": r["tracks"],
                       "Needed per track": f"{r['need_sec']:.0f}s", "Marked footage": f"{r['have_sec']:.0f}s",
                       "": {"none": "❌ none marked", "short": "⚠️ short", "ok": "✅"}[r["status"]]}
                      for r in g["roles"]], hide_index=True, width="stretch")
        st.caption("Short = less than one and a half tracks' worth of marked footage — the same clips keep "
                   "coming back. Mark more in the Reviewer (🎭 Quick role marking), or add videos that have them.")
    else:
        st.caption("None of the searched tracks has story parts.")
    st.markdown("**🕳️ Hardest blocks to match**",
                help="For every block of every searched track: how well the 5th-best video in your library fits "
                     "it. High = plenty of videos suit it; low = only one or two do, so the same clips keep coming "
                     "back there. Grouped by how intense and how long the blocks are.")
    if g["blocks"] and g.get("measure") == "depth":
        st.caption(f"Library average: the 5th-best video fits a block **{g['average']:.0f}%**.")
        for k in g["blocks"]:
            st.markdown(f"- {'🕳️ ' if k['gap'] else ''}**{k['label']}** — the 5th-best video fits **{k['fit']:.0f}%** "
                        f"({k['vs_avg']:+.0f} vs the average) · {k['blocks']} blocks in {k['tracks']} tracks")
        if not any(k["gap"] for k in g["blocks"]):
            st.caption("None stands out — every kind of block has several videos that suit it.")
    elif g["blocks"]:
        st.caption("Shown from older searches — tonight's run adds the full picture.")
        for k in g["blocks"]:
            st.markdown(f"- **{k['label']}** — best fit **{k['fit']:.0f}%** ({k['blocks']} blocks in {k['tracks']} tracks)")
    else:
        st.caption("Not enough searched blocks yet.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

tracks = RL.list_tracks()
if not tracks:
    st.info("No audio tracks found.")
    st.stop()

background_panel()
tabs = st.tabs(["🎵 For this track", "🏆 Top plans", "💥 Best action", "🆕 New videos", "🧩 Hidden gems", "🕳️ Gaps"])
with tabs[0]:
    for_this_track(tracks)
stored = RL.all_results()
_from = (f"From the background searches of {len(stored)} of your {len(tracks)} tracks "
         f"({DEFAULT_VIDEOS} videos per plan, normal order)." if stored else "")
with tabs[1]:
    feed_top(stored)
with tabs[2]:
    feed_action(stored)
with tabs[3]:
    feed_new(stored)
with tabs[4]:
    feed_gems(stored)
with tabs[5]:
    feed_gaps(stored)
