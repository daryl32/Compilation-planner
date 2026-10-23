"""🎵 Cutting settings: style row, pace presets, 🎯 Aim for, Smart / Scored dials and the ✂️ cut editor.

Part of the Compilation Planner page (pages/1_Compilation_Planner.py), which imports everything here.
"""

from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from render_preview import colab_to_local
import cut_scoring as CS
import smart_cuts as SM
import preview_snippets
import core.segmentation as CSEG
from core.segmentation import (SEG_ADAPTIVE, SEG_RHYTHM, SEG_SCORED, SEG_SMART, SEG_METHOD_KEYS,
                               build_segments)
from core.autotune import AUDIO_STYLES, AUDIO_STYLE_CUSTOM
import core.targets as CTGT
import core.cut_edits as CED
import core.styles as CSTY
import core.story_cuts as STC
import settings_store as SSTORE
from core.matching import format_mmss
from cut_editor import cut_editor as _cut_editor_widget, media_url as _media_url
from config import PREVIEW_DIR

from planner.state import (
    _bump_cut_edit_rev, DIAL_DEFAULTS, download_source_video, _DRIVE_SYNC_AVAILABLE, rerun_fragment
)
from planner.data import (
    load_track, _local_media
)
from planner.projects import (
    apply_audio_profile, _run_auto_tune
)


def _setting(key):
    """A setting's current value (settings_store keeps every one in session_state)."""
    return st.session_state.get(key, DIAL_DEFAULTS.get(key))


def segmentation_settings() -> dict:
    """The current segmentation method plus exactly the settings it uses —
    the one place the preview chart, the matching and the export read them from."""
    method = st.session_state.get("segmentation_method", SEG_ADAPTIVE)
    if method not in SEG_METHOD_KEYS:
        method = SEG_ADAPTIVE
    out = {"method": method, "beats_per_bar": int(st.session_state.get("beats_per_bar", 4))}
    for k in SEG_METHOD_KEYS[method]:
        out[k] = st.session_state.get(k, DIAL_DEFAULTS.get(k))
    parts = story_parts()
    if parts:
        out["parts"] = parts
    return out


def story_parts_on() -> bool:
    """🎭 Story-led, or a named style with "Pace per story part" on — and the story map in use."""
    style = st.session_state.get("cut_style", "custom")
    if style not in CSTY.STYLES or not st.session_state.get("use_story_map", True):
        return False
    return style == STC.STORY_STYLE or bool(st.session_state.get("story_part_pace"))


def story_parts() -> list:
    """The story-map parts cut their own way (core/story_cuts.py), or []."""
    tid = st.session_state.get("track_id")
    if not tid or not story_parts_on():
        return []
    from planner.story import current_story_map   # here: planner.story imports this module
    regions = current_story_map(tid)
    if not regions:
        return []
    import json
    import scene_roles as SR
    rc = SR.role_cuts()
    style, pace, bpb = st.session_state["cut_style"], int(st.session_state.get("cut_style_pace", 0)), \
        int(st.session_state.get("beats_per_bar", 4))
    key = json.dumps([tid, regions, rc, style, pace, bpb], sort_keys=True, default=str)
    cache = st.session_state.setdefault("_story_parts_cache", {})
    if key not in cache:
        cache.clear()                          # one at a time — it changes with the map / style / pace
        cache[key] = STC.part_plans(load_track(tid), regions, rc, style, pace, bpb)
    return cache[key]


SEG_LABELS = {
    SEG_SMART: "🧠 Smart — tunes itself to the song (new)",
    SEG_SCORED: "🎯 Best musical moments",
    SEG_ADAPTIVE: "⚡ Energy changes & big hits",
    SEG_RHYTHM: "🎬 Bar styles — cinematic / fast",
}

# "What to cut on" mixes for the Scored method's reason weights.
CUT_FOCUS = {
    "Balanced": {},   # = cut_scoring.DEFAULTS
    "Song structure": {"sc_w_bar": 1.5, "sc_w_phrase": 2.5, "sc_w_section": 4.0, "sc_w_drop": 4.0,
                       "sc_w_hit": 0.5, "sc_w_kick": 0.5, "sc_w_snare": 0.3, "sc_w_harmony": 1.0},
    "Drums & hits": {"sc_w_hit": 2.0, "sc_w_kick": 2.0, "sc_w_snare": 1.5, "sc_w_bar": 0.7,
                     "sc_w_phrase": 1.0, "sc_w_harmony": 0.4},
    "Vocals": {"sc_w_vocal": 3.0, "sc_w_vocal_mid": 2.5, "sc_w_harmony": 1.0},
}


def _mark_audio_custom() -> None:
    """A dial was moved by hand: stop auto-tuning over it."""
    st.session_state["audio_style"] = AUDIO_STYLE_CUSTOM
    st.session_state["cut_style"] = "custom"       # 🎨 no longer exactly the style's recipe
    st.session_state.pop("aim_report", None)     # 🎯 its dial values no longer apply


def _mark_style_custom() -> None:
    """A Smart dial was moved by hand: the 🎨 style no longer describes the settings."""
    st.session_state["cut_style"] = "custom"


def _method_changed() -> None:
    _mark_style_custom()
    _audio_retune_if_paced()


def _beats_per_bar_changed() -> None:
    """A style's recipe depends on the bar length, so it's worked out again; otherwise as before."""
    if st.session_state.get("cut_style") in CSTY.STYLES:
        _apply_cut_style()
    else:
        _audio_retune_if_paced()


def _apply_cut_style() -> None:
    """🎨 Style or its pace changed: set the method and every dial from the style's recipe
    for this track (core/styles.py)."""
    style = st.session_state.get("cut_style", "custom")
    if style not in CSTY.STYLES:
        st.session_state.pop("style_report", None)
        return
    track = load_track(st.session_state["track_id"])
    method, settings, notes = CSTY.style_settings(track, style, int(st.session_state.get("cut_style_pace", 0)),
                                                  int(st.session_state.get("beats_per_bar", 4)))
    st.session_state["segmentation_method"] = method
    for k, v in settings.items():
        st.session_state[k] = v
    st.session_state["audio_style"] = AUDIO_STYLE_CUSTOM     # the style owns the dials — no pace re-tune over them
    st.session_state["style_report"] = {"track_id": st.session_state["track_id"], "notes": notes}
    st.session_state.pop("aim_report", None)


def _render_style_row(_v) -> None:
    """🎨 Cutting style + Pace — the quick way in; the method and dials below stay editable."""
    opts = ["custom"] + list(CSTY.STYLES)
    SSTORE.choose("cut_style", opts, "custom")
    with st.container(border=True):
        c = st.columns([3, 2])
        style = c[0].selectbox(
            "🎨 Cutting style", opts, key="cut_style", on_change=_apply_cut_style,
            format_func=lambda k: "✏️ My own settings" if k == "custom"
            else CSTY.STYLES[k]["label"],
            help="Pick the feel and the app sets the cutting method and all its dials for this track. You can "
                 "still change anything below (it then shows ✏️ My own settings).")
        c[1].select_slider("Pace", options=[-2, -1, 0, 1, 2], key="cut_style_pace", format_func=CSTY.PACE_LABELS.get, on_change=_apply_cut_style,
                           disabled=style == "custom",
                           help="Each step halves or doubles how often the style cuts.")
        if style != "custom":
            st.caption(CSTY.STYLES[style]["help"])
            rep = st.session_state.get("style_report")
            if rep and rep.get("track_id") == st.session_state.get("track_id"):
                for n in rep["notes"][1:]:
                    st.caption("• " + n)
            _render_story_pace(style)
        _render_tune()


def _use_tuned(style: str, pace: int) -> None:
    st.session_state["cut_style"] = style
    st.session_state["cut_style_pace"] = int(pace)
    _apply_cut_style()


def _render_tune() -> None:
    """🎛️ Tune for my videos — every style × pace tried with the videos picked in step 2."""
    tid = st.session_state.get("track_id")
    vids = list(st.session_state.get("committed_selected_videos") or [])
    with st.expander("🎛️ Tune the cutting for my videos", expanded=False):
        if not vids:
            st.caption("Pick your videos in step 2 first — then this tries every cutting style and pace with them "
                       "and shows which cut matches them best.")
            return
        st.caption(f"Tries every style and pace with your {len(vids)} video(s), checks the best with the matcher "
                   "(Auto mode, your story map if it's on) and ranks them. About a minute.")
        key = f"pl_tune_{tid}_{'_'.join(sorted(vids))}"
        if st.button("🎛️ Tune", key="pl_tune_go", type="primary"):
            import rec_library as RL
            bar = st.progress(0.0, "Tuning…")
            try:
                st.session_state[key] = RL.tune_settings(
                    tid, vids, bool(st.session_state.get("sequential_mode")),
                    use_story=bool(st.session_state.get("use_story_map", True)),
                    progress=lambda f, t: bar.progress(min(f, 1.0), t))
            except Exception as e:
                st.error(f"Couldn't tune: {e}")
            bar.empty()
        res = st.session_state.get(key)
        if not res:
            return
        st.caption(f"{res['tried']} ways tried — the best {len(res['results'])} checked:")
        for i, r in enumerate(res["results"][:6]):
            c = st.columns([4, 1], vertical_alignment="center")
            rk = r["ranked"]
            parts = [f"Overall {rk.get('overall'):.0f}%", f"shape {rk.get('shape'):.0f}%",
                     f"motion {rk.get('intensity'):.0f}%"] + (
                [f"story {rk['story']:.0f}%"] if rk.get("story") is not None else [])
            c[0].markdown(f"{'⭐ ' if i == 0 else ''}**{r['label']}** · " + " · ".join(parts))
            current = (st.session_state.get("cut_style"), int(st.session_state.get("cut_style_pace", 0)))
            c[1].button("✓ In use" if current == (r["style"], r["pace"]) else "Use", key=f"pl_tune_use_{i}",
                        width="stretch", disabled=current == (r["style"], r["pace"]),
                        on_click=_use_tuned, args=(r["style"], r["pace"]))


def _render_story_pace(style: str) -> None:
    """🎭 Pace per story part — and what each role's parts get."""
    tid = st.session_state.get("track_id")
    from planner.story import current_story_map
    regions = current_story_map(tid) if tid else None
    map_on = bool(regions) and bool(st.session_state.get("use_story_map", True))
    if style != STC.STORY_STYLE:
        st.toggle("🎭 Pace per story part", key="story_part_pace", disabled=not map_on,
                  help="Each story-map part at its role's pace on top of the Pace above — e.g. 💥 Action busier, "
                       "🌙 Slow calmer. Set each role's pace in the Reviewer → Roles… → ✂️ Cutting per role.")
    if not map_on:
        st.caption("No story map in use for this track — " + ("🎭 Story-led cuts it all as 🧠 Smart."
                   if style == STC.STORY_STYLE else "make or switch on one (step 1) to cut per part."))
        return
    if style == STC.STORY_STYLE or st.session_state.get("story_part_pace"):
        import scene_roles as SR
        roles = {r["id"]: f"{r['emoji']} {r['name']}" for r in SR.all_roles()}
        lines = STC.describe(regions, SR.role_cuts(), style, int(st.session_state.get("cut_style_pace", 0)), roles)
        if lines:
            st.caption("Parts: " + " · ".join(lines))


def _audio_retune_if_paced() -> None:
    """Method, pace or beats per bar changed: re-tune now (unless Custom)."""
    if st.session_state.get("audio_style") in AUDIO_STYLES:
        apply_audio_profile(load_track(st.session_state["track_id"]), st.session_state["track_id"], report=True)


def _apply_cut_focus() -> None:
    mix = {**{k: CS.DEFAULTS[k] for k in CS.REASONS}, "sc_w_vocal_mid": CS.DEFAULTS["sc_w_vocal_mid"],
           **CUT_FOCUS.get(st.session_state.get("sc_focus"), {})}
    for k, v in mix.items():
        st.session_state[k] = v   # the cutting pace doesn't touch these, so it stays as it is


def _pace_row(_pc, _styles) -> None:
    """Cutting pace preset + Re-tune (every method except Smart)."""
    SSTORE.choose("audio_style", _styles, AUDIO_STYLE_CUSTOM)
    _pc[0].selectbox(
        "Cutting pace", _styles, key="audio_style",
        on_change=_method_changed,
        format_func=lambda x: {"Calm": "🌊 Calm — a cut every ~4 bars",
                               "Balanced": "⚖️ Balanced — every ~2 bars",
                               "Energetic": "🔥 Energetic — every bar",
                               AUDIO_STYLE_CUSTOM: "✏️ Custom — your own settings"}.get(x, x),
        help="Sets every dial below for THIS track — its tempo, loudness and hits — and keeps them "
             "tuned if you change the track or method. Moving any dial switches to Custom.")
    _pc[1].button("🎯 Re-tune for this track", on_click=_run_auto_tune, width="stretch",
                  help="Re-runs auto-detect for the current method at the chosen pace (Balanced if Custom). "
                       "Same as the sidebar's Auto-detect.")


AIM_MODES = {"avg": "Average clip length", "count": "Number of segments"}
ADAPTIVE_DIAL_KEYS = ["change_window_secs", "change_threshold", "min_segment_sec", "max_segment_sec",
                      "react_to_hits", "hit_threshold", "snap_secs", "on_beat"]


def _aim_adaptive() -> None:
    """🎯 Set the Adaptive dials to give the segment count / clip length asked for
    (core/targets.py). Runs as a button callback, so the dials redraw with the new values."""
    track = load_track(st.session_state["track_id"])
    mode = st.session_state.get("aim_mode", "avg")
    base = {k: st.session_state.get(k, DIAL_DEFAULTS.get(k)) for k in ADAPTIVE_DIAL_KEYS}
    kw = ({"avg_sec": float(st.session_state.get("aim_avg_sec", 2.5))} if mode == "avg"
          else {"segments": int(st.session_state.get("aim_segments", 40))})
    res = CTGT.adaptive_for_target(track, base=base, beats_per_bar=int(st.session_state.get("beats_per_bar", 4)),
                                   **kw)
    for k, v in res["settings"].items():
        st.session_state[k] = v
    st.session_state["audio_style"] = AUDIO_STYLE_CUSTOM      # your own settings now — don't re-tune over them
    st.session_state["cut_style"] = "custom"
    st.session_state["aim_report"] = {"track_id": st.session_state["track_id"], "exact": res["exact"],
                                      "notes": res["notes"]}


def _render_aim_adaptive(track: dict, current_segments: int) -> None:
    """🎯 Aim for: say how many cuts you want and let the app set the dials."""
    dur = float(track["duration_sec"])
    st.session_state.setdefault("aim_avg_sec", round(min(max(dur / max(current_segments, 1), 0.5), 30.0), 1))
    st.session_state.setdefault("aim_segments", int(min(max(current_segments, 1), 1000)))
    with st.container(border=True):
        c = st.columns([2, 2, 2])
        mode = c[0].selectbox("🎯 Aim for", list(AIM_MODES), format_func=AIM_MODES.get, key="aim_mode",
                              help="Say what you want and the app sets the dials below to get it. "
                                   "You can still fine-tune them afterwards.")
        if mode == "avg":
            c[1].number_input("Seconds per clip (average)", 0.5, 30.0, step=0.5, key="aim_avg_sec",
                              help=f"This track is {dur:.0f}s long, so 2.5s ≈ {dur / 2.5:.0f} segments.")
        else:
            c[1].number_input("Segments", 1, 1000, step=1, key="aim_segments",
                              help=f"This track is {dur:.0f}s long.")
        c[2].button("Set the dials", on_click=_aim_adaptive, type="primary", width="stretch",
                    help="Finds energy-change and big-hit thresholds plus shortest / longest segment that "
                         "give that many cuts on this track.")
        rep = st.session_state.get("aim_report")
        if rep and rep.get("track_id") == st.session_state.get("track_id"):
            (st.caption if rep["exact"] else st.warning)("  \n".join(rep["notes"]))


# ---------------------------------------------------------------------------
# ✂️ Hand-edited cuts (per project): st.session_state["cut_edits"] =
#   {"track_id", "cuts": [seconds…], "base_sig": settings the cuts were edited from}
# planned_segments() is what everything downstream uses — the method's segments,
# or the edited ones when this project has edits for this track.
# ---------------------------------------------------------------------------

def _cut_edits(track_id: str = None):
    e = st.session_state.get("cut_edits")
    tid = track_id or st.session_state.get("track_id")
    return e if isinstance(e, dict) and e.get("track_id") == tid and isinstance(e.get("cuts"), list) else None


def planned_segments(track: dict, cfg: dict) -> list:
    """The segments the plan uses: the method's, or this project's hand-edited cuts."""
    base = build_segments(track, cfg)
    e = _cut_edits()
    return CED.segments_from_cuts(track, e["cuts"], base) if e else base


def _drop_cut_edits() -> None:
    st.session_state.pop("cut_edits", None)
    _bump_cut_edit_rev()


def _keep_cut_edits(sig: str) -> None:
    e = _cut_edits()
    if e:
        e["base_sig"] = sig


def _keep_new_with_locks(sig: str, base_cuts: list, duration: float) -> None:
    """♻️ The new settings' cuts, with this project's 🔒 locked cuts kept where they are."""
    e = _cut_edits()
    if not e:
        return
    locked = CED.clean_locked(e.get("locked"), e["cuts"])
    cuts = CED.merge_locked(base_cuts, locked, duration)
    st.session_state["cut_edits"] = {**e, "cuts": cuts, "locked": locked, "base_sig": sig}
    _bump_cut_edit_rev()


_STYLE_CUTS_CACHE = "_ced_style_cuts"


def _style_cuts(track: dict, tid: str, style: str, pace: int, bpb: int) -> list:
    """The cut times a 🎨 cutting style would make on this whole track (kept per session)."""
    key = (tid, float(track["duration_sec"]), style, int(pace), int(bpb))
    cache = st.session_state.setdefault(_STYLE_CUTS_CACHE, {})
    if key not in cache:
        method, settings, _ = CSTY.style_settings(track, style, int(pace), int(bpb))
        cache[key] = CED.base_cut_times(build_segments(track, {"method": method, "beats_per_bar": int(bpb),
                                                               **settings}))
        while len(cache) > 12:
            cache.pop(next(iter(cache)))
    return cache[key]


def _idea_cuts(track: dict, tid: str, bpb: int) -> list:
    """👻 Candidate cuts: the current style one step busier, Smart busier and On the beat."""
    style, pace = st.session_state.get("cut_style", "custom"), int(st.session_state.get("cut_style_pace", 0))
    out = []
    if style in CSTY.STYLES and pace < 2:
        out += _style_cuts(track, tid, style, pace + 1, bpb)
    out += _style_cuts(track, tid, "smart", 1, bpb)
    out += _style_cuts(track, tid, "beat", 0, bpb)
    return out


def _set_cut_edits(tid: str, cuts: list, locked: list, sig: str, base_cuts: list) -> None:
    """Store this project's cuts (or drop the edit when it's exactly the method's, unlocked)."""
    e = _cut_edits(tid)
    locked = CED.clean_locked(locked, cuts)
    if not locked and not CED.diff(base_cuts, cuts)["added"] and len(cuts) == len(base_cuts):
        st.session_state.pop("cut_edits", None)       # back to exactly the method's cuts
    else:
        st.session_state["cut_edits"] = {"track_id": tid, "cuts": cuts, "locked": locked,
                                         "base_sig": e.get("base_sig", sig) if e else sig}


def _recut_region(track: dict, tid: str, sig: str, base_cuts: list, bpb: int, rng) -> None:
    """🎨 Re-cut the picked part (rng = (start, end), as the slider showed it) in the picked
    style (🔒 locked cuts stay)."""
    if not isinstance(rng, (list, tuple)) or len(rng) != 2 or rng[1] <= rng[0]:
        return
    e = _cut_edits(tid)
    cuts = e["cuts"] if e else base_cuts
    locked = CED.clean_locked((e or {}).get("locked"), cuts)
    style, pace = st.session_state.get("ced_rstyle", "hyper"), int(st.session_state.get("ced_rpace", 0))
    region_cuts = _style_cuts(track, tid, style, pace, bpb)
    new = CED.recut_region(cuts, region_cuts, rng[0], rng[1], float(track["duration_sec"]), locked)
    _set_cut_edits(tid, new, locked, sig, base_cuts)
    st.session_state["ced_recut_msg"] = (f"🎨 Re-cut {format_mmss(rng[0])}–{format_mmss(rng[1])} as "
                                         f"{CSTY.STYLES[style]['label']} ({CSTY.PACE_LABELS[pace]}) — "
                                         f"↶ Undo in the editor puts it back.")
    _bump_cut_edit_rev()


def _ab_keep(tid: str, cuts: list) -> None:
    st.session_state["cut_ab"] = {"track_id": tid, "A": list(cuts)}


def _ab_swap(tid: str, sig: str, base_cuts: list) -> None:
    """⇄ Put the A cuts back in the editor; what's there now becomes A."""
    ab = st.session_state.get("cut_ab")
    if not ab or ab.get("track_id") != tid:
        return
    e = _cut_edits(tid)
    now = e["cuts"] if e else base_cuts
    _set_cut_edits(tid, list(ab["A"]), (e or {}).get("locked") or [], sig, base_cuts)
    st.session_state["cut_ab"] = {"track_id": tid, "A": list(now)}
    _bump_cut_edit_rev()


def _render_cut_editor(track: dict) -> None:
    """✂️ Listen and edit the cuts — see cut_editor.py. Extras: 🔒 lock cuts, 🎨 re-cut a part in
    another style, 👻 ideas to tap in, ⚖️ compare with a kept set (A)."""
    tid = st.session_state["track_id"]
    cfg = segmentation_settings()
    sig = CED.settings_signature(cfg)
    bpb = int(st.session_state.get("beats_per_bar", 4))
    duration = float(track["duration_sec"])
    base_cuts = CED.base_cut_times(build_segments(track, cfg))
    e = _cut_edits(tid)
    st.subheader("✂️ Listen and edit the cuts")
    if e and e.get("base_sig") != sig:
        d = CED.diff(base_cuts, e["cuts"])
        n_locked = len(CED.clean_locked(e.get("locked"), e["cuts"]))
        with st.container(border=True):
            st.warning(f"You edited these cuts by hand, then changed the cutting settings. The plan still uses "
                       f"your edited cuts ({len(e['cuts']) + 1} segments) — the new settings would give "
                       f"{len(base_cuts) + 1}, with {len(d['added'])} of your cuts not among them.")
            bc = st.columns(3 if n_locked else 2)
            bc[0].button("✋ Keep my edited cuts", on_click=_keep_cut_edits, args=(sig,), width="stretch")
            if n_locked:
                bc[1].button(f"♻️ New settings + my {n_locked} 🔒 cut(s)", on_click=_keep_new_with_locks,
                             args=(sig, base_cuts, duration), width="stretch", type="primary",
                             help="The new settings' cuts, with your locked cuts kept exactly where they are.")
            bc[-1].button("♻️ Use the new settings (drop my edits)", on_click=_drop_cut_edits,
                          width="stretch")

    audio_url = ""
    audio_path, missing = _local_media(track.get("source_path", ""))
    if audio_path is not None:
        clip, err = preview_snippets.make_editor_audio(PREVIEW_DIR / "snippets", audio_path)
        if err:
            st.warning(f"Couldn't prepare the audio for playback: {err}")
        else:
            audio_url = _media_url(clip, "audio/mpeg", f"cut_editor_audio_{tid}")
    elif missing:
        _dc = st.columns([3, 2])
        _dc[0].caption(f"▶ Playback needs {Path(colab_to_local(missing)).name} on the server — you can still "
                       f"edit the cuts without it.")
        if _DRIVE_SYNC_AVAILABLE and "MyDrive/" in missing and _dc[1].button(
                "⬇️ Download the track from Google Drive", key="ced_dl", width="stretch"):
            bar = st.progress(0.0, text="Downloading…")
            dl_err = download_source_video(missing.split("MyDrive/", 1)[1], Path(colab_to_local(missing)),
                                           progress_callback=lambda pr: bar.progress(min(pr, 1.0)))
            bar.empty()
            if dl_err:
                st.error(f"Download failed: {dl_err}")
            else:
                rerun_fragment()

    if "cut_edit_rev" not in st.session_state:
        _bump_cut_edit_rev()
    rev = int(st.session_state["cut_edit_rev"])
    cuts_now = e["cuts"] if e else base_cuts
    locked_now = CED.clean_locked((e or {}).get("locked"), cuts_now)
    ab = st.session_state.get("cut_ab")
    ab = ab if isinstance(ab, dict) and ab.get("track_id") == tid else None
    show_ideas = bool(st.session_state.get("ced_ideas"))
    ghosts = CED.ghost_cuts(_idea_cuts(track, tid, bpb), cuts_now, duration) if show_ideas else []
    region = st.session_state.get("ced_region") if st.session_state.get("ced_region_on") else None
    val = _cut_editor_widget(track, tid, cuts=cuts_now, base_cuts=base_cuts,
                             beats=CS.beat_grid(track)[0], bars=CS.bar_lines(track, bpb), doc_rev=rev,
                             audio_url=audio_url, locked=locked_now, ghosts=ghosts,
                             compare=ab["A"] if ab else None, region=region)
    if isinstance(val, dict) and val.get("edited") and int(val.get("doc_rev", -1)) == rev:
        new = CED.clean_cuts(val.get("cuts") or [], duration)
        new_locked = CED.clean_locked(val.get("locked") or [], new)
        if new != cuts_now or new_locked != locked_now:
            _set_cut_edits(tid, new, new_locked, sig, base_cuts)
            e = _cut_edits(tid)
            cuts_now, locked_now = (e["cuts"], e.get("locked") or []) if e else (base_cuts, [])
    elif isinstance(val, dict) and val.get("region") and val.get("nonce") != st.session_state.get("_ced_region_nonce"):
        # ▭ This view as the part: snap to the nearest bar lines for the slider below.
        st.session_state["_ced_region_nonce"] = val.get("nonce")
        lines = _region_lines(track, bpb)
        a, b = val["region"]
        st.session_state["ced_region"] = (min(lines, key=lambda x: abs(x - a)), min(lines, key=lambda x: abs(x - b)))
        st.session_state["ced_region_on"] = True
        st.session_state["ced_region_open"] = True

    if e:
        d = CED.diff(base_cuts, e["cuts"])
        ec = st.columns([4, 2])
        ec[0].caption(f"✏️ **Edited by hand:** {len(e['cuts']) + 1} segments — {len(d['added'])} cut(s) added or "
                      f"moved, {len(d['removed'])} of the method's removed"
                      + (f", {len(locked_now)} 🔒 locked" if locked_now else "")
                      + ". Saved with this project.",
                      help="🟠 your cuts · 🔵 the method's · 🟩 locked · dashed = removed · 👻 dotted = ideas · "
                           "purple ticks along the top = the kept set A.")
        ec[1].button("↺ Undo all edits", on_click=_drop_cut_edits, width="stretch",
                     help="Drops your hand edits for this track (Undo in the editor can't bring them back).")
    else:
        st.caption("▶ Play and listen — tap ✂️ *Cut here* on the beat, drag a cut to move it, or select one and "
                   "🗑 remove it. Your edits are saved with this project.")
    msg = st.session_state.pop("ced_recut_msg", None)
    if msg:
        st.toast(msg)

    # Extras: 👻 ideas · ⚖️ compare · 🎨 re-cut a part
    xc = st.columns([1.3, 1.3, 1.6], vertical_alignment="center")
    xc[0].toggle("👻 Show ideas", key="ced_ideas",
                 help="Faint dotted lines where other cutting styles (this one busier, 🧠 Smart busier and "
                      "🥁 On the beat) would cut — tap one in the editor to make it a cut.")
    if ab:
        xc[1].button("⇄ Swap with A", key="ced_ab_swap", on_click=_ab_swap, args=(tid, sig, base_cuts),
                     width="stretch",
                     help="Puts the kept set (A) back in the editor and keeps what's there now as A — listen to "
                          "one, swap, listen to the other.")
    else:
        xc[1].button("📌 Keep as A", key="ced_ab_keep", on_click=_ab_keep, args=(tid, cuts_now),
                     width="stretch",
                     help="Keep these cuts as set A, then try something else and compare (A shows as purple ticks "
                          "along the top of the editor).")
    if ab:
        sa, sb = CED.cut_stats(ab["A"], duration), CED.cut_stats(cuts_now, duration)
        diff_n = len(CED.diff(ab["A"], cuts_now)["added"]) + len(CED.diff(ab["A"], cuts_now)["removed"])
        xc[2].markdown(f"⚖️ **A** {sa['segments']} segments, avg {sa['average']:.1f}s · **now** {sb['segments']}, "
                       f"avg {sb['average']:.1f}s · {diff_n} cut(s) differ",
                       help=f"Shortest / longest — A: {sa['shortest']:.1f}s / {sa['longest']:.1f}s · now: "
                            f"{sb['shortest']:.1f}s / {sb['longest']:.1f}s.")
        xc[1].button("✕ Forget A", key="ced_ab_forget", width="stretch",
                     on_click=lambda: st.session_state.pop("cut_ab", None))
    with st.expander("🎨 Re-cut a part in another style", expanded=bool(st.session_state.pop("ced_region_open", False))):
        lines = _region_lines(track, bpb)
        if not isinstance(st.session_state.get("ced_region"), (list, tuple)) or \
                any(x not in lines for x in st.session_state["ced_region"]):
            st.session_state["ced_region"] = (lines[0], lines[min(len(lines) - 1, 8)])
        part = st.select_slider("Part of the track", options=lines, key="ced_region", format_func=format_mmss,
                         help="On bar lines. Or zoom the editor to the part and press ▭ This view as the part.")
        st.session_state.setdefault("ced_region_on", False)
        st.toggle("Shade it in the editor", key="ced_region_on")
        rc = st.columns([2, 1.4, 1.6], vertical_alignment="bottom")
        st.session_state.setdefault("ced_rstyle", "hyper")
        st.session_state.setdefault("ced_rpace", 0)
        rc[0].selectbox("Style", [k for k in CSTY.STYLES if k != "story"], key="ced_rstyle", format_func=lambda k: CSTY.STYLES[k]["label"])
        rc[1].selectbox("Pace", list(CSTY.PACE_LABELS), key="ced_rpace", format_func=lambda k: CSTY.PACE_LABELS[k])
        rc[2].button("🎨 Re-cut this part", key="ced_recut", type="primary", width="stretch",
                     on_click=_recut_region, args=(track, tid, sig, base_cuts, bpb, part),
                     help="Replaces the cuts inside the part with the style's — 🔒 locked cuts stay, and the "
                          "rest of the track is untouched. ↶ Undo in the editor takes it back.")


def _region_lines(track: dict, bpb: int) -> list:
    """Times the re-cut part can start / end on: the track's start and end, and its bar lines."""
    dur = round(float(track["duration_sec"]), 3)
    return sorted({0.0, dur, *[round(float(b), 3) for b in CS.bar_lines(track, bpb) if 0 < float(b) < dur]})


def _smart_reset() -> None:
    for k, v in SM.DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["cut_style"] = "smart"        # = the 🧠 Smart style at normal pace
    st.session_state["cut_style_pace"] = 0


_PACE_LABELS = {-2: "much calmer", -1: "calmer", 0: "auto", 1: "busier", 2: "much busier"}


def _render_smart_cut_settings(track: dict, _v) -> None:
    """Controls and the per-section breakdown for the Smart method. Smart tunes
    itself to the track; these only dial it in (the cutting-pace presets don't apply)."""
    _custom = dict(on_change=_mark_style_custom)
    a = track.get("analysis") or {}
    if not a:
        st.info("This track only has the basic analysis — Smart works from its beats and loudness. Add the "
                "extra analysis (bars, sections, drops, build-ups, drums, vocals) from **Media Library → Audio → "
                "🔬 Audio analysis** for its full picture of the song.")
    c = st.columns(2)
    c[0].select_slider("Pace", options=[-2, -1, 0, 1, 2], key="sm_pace",
                       format_func=lambda x: _PACE_LABELS[x], **_custom,
                       help="Moves the whole edit one grid level calmer or busier than the pace Smart picked "
                            "for this song (e.g. every bar → every 2 bars).")
    c[1].slider("Contrast", 0.0, 2.0, step=0.25, key="sm_contrast", **_custom,
                help="How much the pace changes between quiet and intense sections. 1 = what this song's own "
                     "dynamics suggest; 0 = the same pace throughout; 2 = exaggerated.")
    t = st.columns(4)
    _bld = t[0].toggle("Speed up into drops", key="sm_builds", **_custom,
                       help="Through a build-up the cuts get quicker (e.g. every 2 bars → bar → half bar), "
                            "landing on the drop. How long the speed-up runs and how fast it gets follow how "
                            "big the rise into the drop is.")
    c[1].slider("Speed-up strength", 0.0, 2.0, step=0.25,
                key="sm_build_strength", disabled=not _bld, **_custom,
                help="1 = follows the song: a small rise gets a short nudge (2 bars, a step or two quicker), a "
                     "huge rise a long, steep run (up to 16 bars). Lower = gentler and shorter; 2 = longer and "
                     "faster; 0 = no speed-up.")
    t[1].toggle("Accents", key="sm_accents", **_custom,
                help="An extra cut on hits that stand out from the rest of the song, and on vocal entries — "
                     "only where the section's rhythm leaves room.")
    t[2].toggle("Respect vocals", key="sm_vocals", **_custom,
                help="Moves a cut that would land mid-line to the gap between sung lines (needs vocal analysis).")
    t[3].toggle("Mirror repeats", key="sm_mirror", **_custom,
                help="Repeated sections (chorus 1 and 2) get the same cutting pattern.")
    _, rep = SM.plan_cuts(track, segmentation_settings())
    with st.expander("🧠 How Smart read this song", expanded=True):
        for note in rep.get("notes", []):
            st.caption("• " + note)
        if rep.get("features"):
            st.caption("• Intensity from: " + ", ".join(rep["features"]))
        rows = [{"From": format_mmss(s["start"]), "To": format_mmss(s["end"]), "Section": s["label"],
                 "Intensity": s["intensity"], "A cut every": s["every"], "Why": s["reason"]}
                for s in rep.get("sections", [])]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch",
                         column_config={"Intensity": st.column_config.ProgressColumn(
                             "Intensity", min_value=0.0, max_value=1.0, format="%.2f")})


def _render_scored_cut_settings(track: dict, _v) -> None:
    """Dials for the Best musical moments (Scored cuts) method."""
    a = track.get("analysis") or {}
    if not a:
        st.info("This track only has the basic analysis, so cuts are scored from hits and beats alone. "
                "Add the extra analysis (bars, drums, sections, drops, vocals) from **Media Library → "
                "Audio → 🔬 Audio analysis** for the full set of signals.")
    _custom = dict(on_change=_mark_audio_custom)
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Average bars per cut", 0.5, 8.0, step=0.25,
                  key="sc_bars_per_cut", **_custom,
                  help="Overall cut rate: 1 = about one cut per bar, 4 = one per 4 bars.")
        st.slider("Follow the energy", 0.0, 1.0, step=0.05,
                  key="sc_follow_energy", **_custom,
                  help="How much faster loud, busy passages cut than quiet ones. 0 = the same rate everywhere.")
    with col2:
        st.slider("Shortest segment (s)", 0.3, 8.0, step=0.1, key="sc_min_sec",
                  **_custom, help="No two cuts closer than this.")
        st.slider("Longest segment (s)", 0.0, 30.0, step=0.5, key="sc_max_sec",
                  **_custom, help="No segment longer than this, if there's any reasonable cut in range. 0 = no limit.")
    with st.expander("What to cut on", expanded=False):
        _fl = list(CUT_FOCUS)
        SSTORE.choose("sc_focus", _fl)
        st.selectbox("Mix", _fl, key="sc_focus", on_change=_apply_cut_focus,
                     help="Sets the weights below: Song structure favours bar lines, phrases, sections and "
                          "drops; Drums & hits favours kicks, snares and big hits; Vocals cuts between sung "
                          "lines and avoids cutting through them.")
        if a and a.get("vocals") is None:
            st.caption("No vocal analysis for this track — the vocal weights have no effect.")
        wcols = st.columns(3)
        weight_help = {
            "sc_w_bar": ("Bar line", "The first beat of a bar."),
            "sc_w_phrase": ("Phrase line", "Every 4 / 8 / 16 bars from a section start — 16 counts most."),
            "sc_w_section": ("Section change", "Verse → chorus etc., found from a change in the sound."),
            "sc_w_drop": ("Drop", "The moment a build-up lands."),
            "sc_w_hit": ("Big hit", "Any sharp new sound (onset strength)."),
            "sc_w_kick": ("Kick", "Low-frequency drum hits."),
            "sc_w_snare": ("Snare / clap", "Mid-frequency drum hits."),
            "sc_w_harmony": ("Chord change", "The notes / chords change."),
            "sc_w_vocal": ("Vocal line start / end", "Where a sung phrase begins or ends."),
            "sc_w_vocal_mid": ("Avoid cutting mid-vocal", "Penalty for cutting while a vocal line is being sung."),
        }
        for i, (key, (label, hlp)) in enumerate(weight_help.items()):
            with wcols[i % 3]:
                st.slider(label, 0.0, 5.0, step=0.1, key=key, help=hlp)


CUT_KIND_STYLES = (
    ("change", "rgba(0,160,120,0.7)", "Cut: energy change (snapped)"),
    ("hit", "rgba(220,60,60,0.55)", "Cut: big hit"),
    ("fill", "rgba(230,150,30,0.6)", "Cut: max-length split (on strongest beat)"),
    ("cinematic", "rgba(80,120,220,0.5)", "Rhythm Engine: CINEMATIC"),
    ("fast", "rgba(230,150,30,0.6)", "Rhythm Engine: FAST"),
    ("override", "rgba(230,30,200,0.75)", "Rhythm Engine: Dynamic Priority Override"),
    ("sc_section", "rgba(150,60,200,0.9)", "Scored cut: section change"),
    ("sc_drop", "rgba(220,40,40,0.9)", "Scored cut: drop"),
    ("sc_phrase", "rgba(60,90,200,0.8)", "Scored cut: phrase line"),
    ("sc_bar", "rgba(90,90,90,0.6)", "Scored cut: bar line"),
    ("sc_kick", "rgba(200,60,40,0.6)", "Scored cut: kick"),
    ("sc_snare", "rgba(40,140,200,0.6)", "Scored cut: snare"),
    ("sc_hit", "rgba(220,60,60,0.55)", "Scored cut: big hit"),
    ("sc_harmony", "rgba(60,170,90,0.7)", "Scored cut: chord change"),
    ("sc_vocal", "rgba(230,120,0,0.8)", "Scored cut: vocal line"),
    ("sc_beat", "rgba(140,140,140,0.5)", "Scored cut: beat"),
    ("sm_section", "rgba(150,60,200,0.9)", "Smart: section change"),
    ("sm_drop", "rgba(220,40,40,0.9)", "Smart: drop"),
    ("sm_build", "rgba(230,150,30,0.8)", "Smart: build-up (speeding up)"),
    ("sm_grid", "rgba(60,90,200,0.6)", "Smart: section's cutting rhythm"),
    ("sm_accent", "rgba(220,60,60,0.7)", "Smart: accent (standout hit)"),
    ("sm_vocal", "rgba(230,120,0,0.8)", "Smart: vocal entry"),
)


def add_cut_markers(fig, segments: list) -> None:
    """Vertical markers at every cut, coloured by why the cut happened."""
    for _kind, _colour, _label in CUT_KIND_STYLES:
        _xs, _ys = [], []
        for _sg in segments:
            if _sg.get("cut") == _kind:
                _xs += [_sg["start"], _sg["start"], None]
                _ys += [0, 1, None]
        if _xs:
            fig.add_trace(go.Scatter(x=_xs, y=_ys, mode="lines", name=_label,
                                     line=dict(color=_colour, width=1, dash="dot"), hoverinfo="skip"))


# ---------------------------------------------------------------------------
# Rhythm Engine: an alternative segmentation method, ported from a reference
# state-machine design. Instead of detecting individual cut POINTS, it works
# at bar granularity — each bar gets a cutting STYLE: CINEMATIC (slow,
# multi-bar segments) or FAST (one segment per bar), from that bar's
# smoothed energy. Phrase-locking commits to a style for several bars so it
# doesn't flicker; FAST has a maximum run length and a cooldown so rapid
# cutting can't run on for too long or restart straight away.
# ---------------------------------------------------------------------------


def clips_for_intensity(intensity: float, min_clips: int, max_clips: int,
                        ramp_start: float, ramp_end: float, vary: bool, rng) -> int:
    """core.segmentation.clips_for_intensity with the 'Skip these counts' setting."""
    return CSEG.clips_for_intensity(intensity, min_clips, max_clips, ramp_start, ramp_end, vary, rng,
                                    skip=st.session_state.get("clip_count_skip"))


def allowed_clip_count(n: int, min_clips: int, max_clips: int) -> int:
    """core.segmentation.allowed_clip_count with the 'Skip these counts' setting."""
    return CSEG.allowed_clip_count(n, min_clips, max_clips, skip=st.session_state.get("clip_count_skip"))

