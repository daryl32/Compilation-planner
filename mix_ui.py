"""🎚️ Mix — the Media Library's audio mixing screen (drawn by pages/3_Media_Library.py).

    📏 Extend / shorten   one track to a new length: 🎯 Auto (finds bars that
                          sound alike and jumps between them) or ✋ By sections
                          (play each section 0–4 times)
    🧩 Mashup             parts of several tracks in a row, joined on bar lines,
                          stretched to one tempo, keys matched
    📂 Saved mixes        re-open, re-save, upload or forget a mix

Every mix is a recipe (core/mixing.py) — ▶️ Preview renders it to a quick
temporary file; 💾 Save as a new track renders it properly, puts it in the
Audio-Library (+ Google Drive) and queues the normal audio analysis (mixes.py).

Renders, saves, Drive downloads and the live preview's stems run in the background (mix_jobs.py), so they
finish if the iPad sleeps; the screen is a fragment, so a tap redraws just the Mix screen.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

from config import AUDIO_DIR
from track_files import track_file
from core import keys as K
from core import mixing as MX
from core import stem_mix as SM
from library_common import format_mmss

MODES = ["📏 Extend / shorten", "🧩 Mashup", "🎚️ Transition", "💡 Suggestions", "📂 Saved mixes"]
EXTEND, MASHUP, TRANSITION, SUGGESTIONS, SAVED = MODES
PART_LABELS = {"full": "Full song", "instrumental": "Instrumental", "vocals": "Vocals only"}
PART_ICON_LABELS = {"full": "🎵 Full", "instrumental": "🎹 No vocals", "vocals": "🎤 Vocals"}
STEM_LABELS = {"vocals": "🎤 Vocals", "drums": "🥁 Drums", "bass": "🎸 Bass", "other": "🎹 Other"}


def part_label(part: str | None) -> str:
    """"Full song", "Instrumental", "Vocals only", or the stems a part plays ("drums + bass")."""
    part = part or "full"
    if part in PART_LABELS:
        return PART_LABELS[part]
    import mixer
    return " + ".join(mixer.part_stems(part))
XF_SAME = {"Very short (⅛ beat)": 0.125, "Short (¼ beat)": 0.25, "Half a beat": 0.5, "One beat": 1.0}
XF_OTHER = {"¼ beat": 0.25, "1 beat": 1.0, "2 beats": 2.0, "1 bar": 4.0, "2 bars": 8.0, "4 bars": 16.0,
            "8 bars": 32.0, "16 bars": 64.0}
# 🔀 A join's own crossfade (Mashup): length in beats ("Cut" = straight swap on the bar line)…
XF_JOIN = {"Cut": 0.0, "¼ beat": 0.25, "½ beat": 0.5, "1 beat": 1.0, "2 beats": 2.0, "1 bar": 4.0,
           "2 bars": 8.0, "4 bars": 16.0, "8 bars": 32.0, "16 bars": 64.0}
# …where it sits around the bar line (core.mixing.XF_WHERE) and its shape.
XF_WHERE_LABELS = {"centre": "Both sides of the bar line", "before": "Next song comes in early",
                   "after": "This song carries on"}
XF_SHAPE_LABELS = {"smooth": "Smooth", "linear": "Straight"}
LAYER_FADES = {"¼ beat": 0.25, "½ beat": 0.5, "1 beat": 1.0, "1 bar": 4.0, "2 bars": 8.0, "4 bars": 16.0}
LAYER_ICONS = {"vocals": "🎤", "instrumental": "🎹", "full": "🎵"}
PALETTE = ["rgba(99,132,255,{a})", "rgba(255,140,60,{a})", "rgba(46,184,120,{a})", "rgba(200,90,200,{a})",
           "rgba(230,190,40,{a})", "rgba(60,190,210,{a})", "rgba(230,80,90,{a})"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _has_file(path) -> bool:
    """Whether a song file is here — False (not a crash) when its folder can't be read.
    Never cached: a file fetched from Drive must show up straight away."""
    try:
        return Path(path).exists()
    except OSError:
        return False


@st.cache_data(max_entries=12, show_spinner=False)
def _load_track(track_id: str, mtime: float) -> dict:
    return json.loads(track_file(track_id, AUDIO_DIR).read_text())


def full_track(track_id: str) -> dict | None:
    path = track_file(track_id, AUDIO_DIR)
    try:
        return _load_track(track_id, path.stat().st_mtime)
    except (OSError, ValueError):
        return None


def _mixable(summaries: dict) -> dict:
    """Tracks with a beat grid (bar lines come from the extra analysis, else every 4th beat)."""
    return {tid: t for tid, t in summaries.items() if t.get("beats", 0) > 16}


def _label(t: dict) -> str:
    bits = [format_mmss(t["duration"])]
    if t.get("bpm"):
        bits.append(f"{t['bpm']:.0f} BPM")
    if t.get("key"):
        bits.append(f"🎹 {t['key']['camelot']}")
    return f"{t['track_id']}  ·  " + " · ".join(bits)


def _key_text(key: dict | None, shift: int = 0) -> str:
    if not key:
        return "key not found yet"
    if shift:
        k2 = K.shifted(key, shift)
        return f"{key['name']} ({key['camelot']}) → {k2['name']} ({k2['camelot']}), {shift:+d} st"
    return f"{key['name']} ({key['camelot']})"


def _keep_pick(key: str) -> None:
    """Pills: a second tap on the chosen pill would leave nothing chosen — keep the last choice instead."""
    ss = st.session_state
    if ss.get(key) is None:
        ss[key] = ss.get(f"_last_{key}")
    ss[f"_last_{key}"] = ss.get(key)


def _pick(where, label: str, options: list, key: str, fmt=None, help: str = None, disabled: bool = False,
          on_change=None, args=()):
    """One-tap choice: a row of pills (wraps on a narrow screen). Returns the choice."""
    ss = st.session_state
    if ss.get(key) not in options:
        ss[key] = options[0]
    # set it again each run: the same box can be drawn in two places (a join's chip, or its panel under the
    # lanes), and a box drawn somewhere new only shows the value it's given now
    ss[key] = ss[key]
    ss.setdefault(f"_last_{key}", ss[key])

    def changed():
        _keep_pick(key)
        if on_change:
            on_change(*args)
    where.pills(label, options, key=key, format_func=fmt or str, help=help, disabled=disabled, on_change=changed)
    return ss.get(key) or ss.get(f"_last_{key}")


def _rerun() -> None:
    """Rerun just the Mix screen when we're in its fragment's own rerun (no page flash), else the page."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
    if ctx is not None and getattr(ctx, "fragment_ids_this_run", None):
        st.rerun(scope="fragment")
    st.rerun()


@st.fragment(run_every=1.0)
def _job_watch(keys: tuple) -> None:
    """⏳ Progress for background jobs (mix_jobs.py) — checked every second; the page refreshes when they're done."""
    import mix_jobs as J
    jobs = [j for j in (J.get(k) for k in keys) if j is not None]
    running = [j for j in jobs if j.active]
    if not running:
        st.rerun(scope="app")
    for j in running:
        text = j.label + (f" — {j.msg}" if j.msg and "\n" not in j.msg else "") + "…"
        st.progress(min(1.0, j.frac), text=text)


def _fetch_job(job, track: dict, part: str) -> str:
    import mixer
    job.progress(0.0, "starting")
    err = mixer.ensure_local(track, parts=(part,), progress=lambda p: job.progress(p, f"{p * 100:.0f} %"))
    if err:
        raise RuntimeError(err)
    return str(mixer.local_path(track, part))


def _fetch_key(track: dict, part: str) -> str:
    import mixer
    return f"fetch:{mixer.local_path(track, part)}"


def _start_fetch(tid: str, track: dict, part: str):
    import mix_jobs as J
    return J.start(_fetch_key(track, part), f"⬇️ {tid}" + ("" if part == "full" else f" ({part})"),
                   _fetch_job, track, part, retry_failed=False)


def _prefetch(track_ids) -> None:
    """⬇️ Quietly start fetching each track's song and stems from Google Drive as soon as it's picked, so
    they're on the server by the time a preview needs them (downloads run in the background)."""
    import mixer
    for tid in dict.fromkeys(track_ids):
        t = full_track(tid)
        if t is None:
            continue
        parts = ["full"] + [p for p in mixer.FOUR_STEMS if mixer.has_four_stems(t)] + \
            [p for p in ("instrumental",) if (t.get("stems") or {}).get(p)]
        for part in parts:
            ref = mixer._drive_ref(t, part)
            if ref and "MyDrive/" in ref and not _has_file(mixer.local_path(t, part)):
                _start_fetch(tid, t, part)


def _retry_fetch(keys: list) -> None:
    import mix_jobs as J
    for k in keys:
        J.forget(k)


def _audio_ready(needs, ns: str) -> bool:
    """True when every file the mix plays is on the server; otherwise fetches them from Google Drive in the
    background (no button to press) and shows how far it has got. needs: track ids (the full songs) or
    {track_id: {"full", "vocals", …}}."""
    import mixer
    needs = needs if isinstance(needs, dict) else {tid: {"full"} for tid in needs}
    missing = [(tid, part) for tid, parts in needs.items() for part in sorted(parts)
               if (t := full_track(tid)) and not _has_file(mixer.local_path(t, part))]
    if not missing:
        return True
    keys, failed = [], []
    for tid, part in missing:
        t = full_track(tid)
        if part != "full" and not mixer._drive_ref(t, part):
            failed.append((None, f"{tid}: no {part} stem yet — run run_audio_analysis.bat on the laptop (it makes stems)."))
            continue
        j = _start_fetch(tid, t, part)
        keys.append(j.key)
        if j.state == "error":
            failed.append((j.key, j.error))
        elif j.state == "done":                 # downloaded, yet not where it should be
            failed.append((j.key, f"{tid}: downloaded but not found at {mixer.local_path(t, part)}"))
    for _, err in failed:
        st.warning(f"⬇️ {err}")
    retry = [k for k, _ in failed if k]
    if retry:
        st.button("⬇️ Try the download again", key=f"{ns}_fetch_retry", on_click=_retry_fetch, args=(retry,))
    if any(_job_active(k) for k in keys):
        st.caption(f"⬇️ Fetching {len(keys)} file(s) from Google Drive — this carries on if you leave the page.")
        _job_watch(tuple(k for k in keys if _job_active(k)))
    return False


def _job_active(key: str) -> bool:
    import mix_jobs as J
    j = J.get(key)
    return bool(j and j.active)


# 🛟 Nothing replaces a mashup with parts without asking first, and the replaced one can be brought back.
_DRAFT_KEYS = ("mxm_parts", "mxm_layers", "mxm_bpm", "mxm_match", "mxm_xf", "mxm_fin", "mxm_fout", "mxm_level",
               "mxm_id", "mxm_name", "mxm_snap")
_DRAFT_PREFIXES = ("mxm_shift_", "mxm_part_", "mxm_xfl_", "mxm_xfw_", "mxm_xfs_", "mxm_xfb_", "mxm_nnp_", "mxm_nnb_",
                   "mxm_nnc_", "mxm_gain_", "mxm_lgain_", "_last_mxm_")
_REPLACERS = {}                                    # name → function, for _mx_pending (session state holds names)


def _replacer(fn):
    _REPLACERS[fn.__name__] = fn
    return fn


def _mashup_snapshot() -> dict:
    import copy
    ss = st.session_state
    return {k: copy.deepcopy(ss[k]) for k in list(ss.keys()) if k in _DRAFT_KEYS or k.startswith(_DRAFT_PREFIXES)}


def _mashup_restore(snap: dict) -> None:
    import copy
    ss = st.session_state
    for k in [k for k in list(ss.keys()) if k in _DRAFT_KEYS or k.startswith(_DRAFT_PREFIXES)]:
        del ss[k]
    ss.update(copy.deepcopy(snap))


# ↶ Undo / ↷ redo for the 🧩 Mashup: a snapshot each time the mix changes (up to UNDO_STEPS back)
UNDO_STEPS = 40


def _history_note(recipe: dict) -> None:
    """Called once a run with the mix as it now stands: remembers the previous state when it changed."""
    ss = st.session_state
    sig = json.dumps({k: recipe.get(k) for k in ("pieces", "layers", "bpm", "shift", "xfade_other", "fade_in",
                                                 "fade_out", "level_match")}, sort_keys=True, default=str)
    cur = ss.get("_mxm_cur")
    if ss.pop("_mxm_restoring", False) or cur is None:
        ss["_mxm_cur"] = (sig, _mashup_snapshot())
        return
    if cur[0] != sig:
        hist = ss.setdefault("_mxm_hist", [])
        hist.append(cur[1])
        del hist[:-UNDO_STEPS]
        ss["_mxm_future"] = []
        ss["_mxm_cur"] = (sig, _mashup_snapshot())


def _undo(step: int) -> None:
    ss = st.session_state
    src, dst = ("_mxm_hist", "_mxm_future") if step < 0 else ("_mxm_future", "_mxm_hist")
    if not ss.get(src) or not ss.get("_mxm_cur"):
        return
    ss.setdefault(dst, []).append(ss["_mxm_cur"][1])
    _mashup_restore(ss[src].pop())
    ss["_mxm_restoring"] = True


def _replace_mashup(fn_name: str, args: tuple = (), what: str = "") -> None:
    """on_click for anything that replaces the 🧩 Mashup: straight away when it's empty, else ask first."""
    if st.session_state.get("mxm_parts"):
        st.session_state["_mx_pending"] = {"fn": fn_name, "args": list(args), "what": what}
    else:
        _REPLACERS[fn_name](*args)


def _do_pending() -> None:
    p = st.session_state.pop("_mx_pending", None)
    if not p:
        return
    snap = _mashup_snapshot()
    _REPLACERS[p["fn"]](*p["args"])
    if snap.get("mxm_parts"):
        st.session_state["_mx_draft"] = snap


def _drop_pending() -> None:
    st.session_state.pop("_mx_pending", None)


@st.dialog("Replace your mashup?", on_dismiss=_drop_pending)
def _confirm_replace() -> None:
    p = st.session_state.get("_mx_pending")
    if not p:                        # answered (a button's callback ran) — close
        st.rerun()
    n = len(st.session_state.get("mxm_parts") or [])
    name = st.session_state.get("mxm_name") or ""
    st.markdown(f"Your current mashup{f' **{name}**' if name else ''} ({n} part{'s' if n != 1 else ''}) will be "
                f"replaced{' by ' + p['what'] if p.get('what') else ''}. Saved mixes aren't affected, and "
                "↩️ Bring it back undoes this.")
    c = st.columns(2)
    c[0].button("Replace it", type="primary", width="stretch", key="mx_pending_yes", on_click=_do_pending)
    c[1].button("Keep editing", width="stretch", key="mx_pending_no", on_click=_drop_pending)


def _pending_check() -> None:
    if st.session_state.get("_mx_pending"):
        _confirm_replace()


def _bring_back() -> None:
    snap = st.session_state.pop("_mx_draft", None)
    if not snap:
        return
    _clear("mxm")
    _mashup_restore(snap)
    st.session_state["_mxm_restoring"] = True
    st.session_state["mx_mode"] = MASHUP


def _draft_banner() -> None:
    snap = st.session_state.get("_mx_draft")
    if not snap:
        return
    n = len(snap.get("mxm_parts") or [])
    c = st.columns([4, 1.4, 0.6], vertical_alignment="center")
    c[0].caption(f"You replaced a mashup{' — **' + snap['mxm_name'] + '**' if snap.get('mxm_name') else ''} "
                 f"({n} part{'s' if n != 1 else ''}).")
    c[1].button("↩️ Bring it back", key="mx_draft_back", on_click=_bring_back, width="stretch")
    c[2].button("✖", key="mx_draft_x", on_click=lambda: st.session_state.pop("_mx_draft", None),
                help="Forget it")


@_replacer
def _new_mashup() -> None:
    _clear("mxm")


def _render_progress(recipe: dict, tracks: dict, out, label: str = "Mixing") -> dict:
    """mixer.render with a progress bar that says which song it's reading."""
    import mixer
    bar = st.progress(0.0, text=f"{label}…")
    state = {"f": 0.0, "msg": ""}

    def show():
        bar.progress(min(1.0, state["f"]), text=f"{label} — {state['msg']}…" if state["msg"] else f"{label}…")

    def on_msg(m):
        state["msg"] = m
        show()

    def on_frac(f):
        state["f"] = f
        show()

    try:
        return mixer.render(recipe, tracks, out, progress=on_msg, fraction=on_frac)
    finally:
        bar.empty()


def _features(track_id: str):
    """(grid, bar similarity) — needs the audio on the server; cached per run."""
    import mixer
    t = full_track(track_id)
    g = MX.grid(t)
    cache = st.session_state.setdefault("_mx_sim", {})
    sig = (track_id, len(g["bars"]), round(float(g["bars"][-1]) if len(g["bars"]) else 0.0, 2))
    if cache.get(track_id, (None,))[0] != sig:
        with st.spinner(f"Listening to {track_id} bar by bar…"):
            S = MX.bar_similarity(mixer.bar_features(t, g))
        cache.clear()           # keep one track's matrix in memory at a time
        cache[track_id] = (sig, S)
    return g, cache[track_id][1]


# ---------------------------------------------------------------------------
# The picture: where each piece of the new track comes from
# ---------------------------------------------------------------------------

def _predicted_energy(lay: dict, tracks: dict, rate: int = 10) -> list:
    n = int(lay["duration"] * rate) + 1
    out = np.zeros(n)
    for p in lay["pieces"]:
        h = (tracks[p["track"]].get("hires") or {})
        rms = np.asarray(h.get("rms") or [], dtype=float)
        if not len(rms):
            continue
        src_rate = float(h.get("rate", 30))
        i0, i1 = int(p["out_start"] * rate), int(p["out_end"] * rate)
        ts = p["start"] + (np.arange(i0, i1) / rate - p["out_start"]) * p["speed"]
        idx = np.clip((ts * src_rate).astype(int), 0, len(rms) - 1)
        out[i0:i1] = rms[idx]
    return out.tolist()


def _join_notes(lay: dict, recipe: dict, tracks: dict, sims: dict) -> list:
    """Per join: (time, colour, text) — how seamless it should sound."""
    shifts = recipe.get("shift") or {}
    notes = []
    for a, b, jn in zip(lay["pieces"][:-1], lay["pieces"][1:], lay["joins"]):
        if a["track"] == b["track"]:
            q = None
            if a["track"] in sims:
                g, S = sims[a["track"]]
                q = MX.join_quality(S, g, a["end"], b["start"])
            if q is None:
                notes.append((jn["at"], "grey", "jump within the track"))
            else:
                colour = "green" if q >= 0.85 else ("orange" if q >= 0.65 else "red")
                notes.append((jn["at"], colour, f"jump {format_mmss(a['end'])} → {format_mmss(b['start'])}, "
                                                f"match {q * 100:.0f} %"))
        else:
            ka = K.shifted(tracks[a["track"]].get("key"), shifts.get(a["track"], 0) or 0)
            kb = K.shifted(tracks[b["track"]].get("key"), shifts.get(b["track"], 0) or 0)
            score, label = K.compatibility(ka, kb)
            colour = "green" if score >= 0.85 else ("orange" if score >= 0.5 else "red")
            notes.append((jn["at"], colour, f"{a['track']} → {b['track']}: {label}"))
    return notes


def _timeline(recipe: dict, tracks: dict, sims: dict, energy: list = None, audio_url: str = "",
              ns: str = "mx", editable: bool = False) -> dict | None:
    """The 🎚️ Mix timeline (mix_timeline.py): waveform, parts, crossfades (◆ edges draggable when
    `editable`), bass-swap points, layers, and the preview player. Returns the layout."""
    from mix_timeline import mix_timeline
    grids = {tid: MX.grid(t) for tid, t in tracks.items()}
    lay = MX.layout(recipe, grids)
    if not lay["pieces"]:
        return None
    colour_of = {}
    for p in lay["pieces"]:
        colour_of.setdefault(p["track"], PALETTE[len(colour_of) % len(PALETTE)])
    for ly in lay.get("layers") or []:
        colour_of.setdefault(ly["track"], PALETTE[len(colour_of) % len(PALETTE)])
    total = min(lay["duration"], float(recipe.get("trim_to") or lay["duration"]))
    if energy:
        env, rate = list(energy), len(energy) / max(total, 1e-6)       # the previewed mix itself (30 Hz)
    else:
        env, rate = _predicted_energy(lay, tracks), 10.0
    top = max(env) if env else 1.0
    env = [v / top for v in env] if top else env
    many = len({p["track"] for p in lay["pieces"]}) > 1
    pieces = [{"x0": p["out_start"], "x1": p["out_end"], "colour": colour_of[p["track"]],
               "label": (f"{p['track'][:22]} " if many else "") + f"{format_mmss(p['start'])}–{format_mmss(p['end'])}"
               + ("" if (p.get("part") or "full") == "full" else f" · {part_label(p['part']).lower()}")}
              for p in lay["pieces"]]
    layers = [{"x0": ly["out_start"], "x1": ly["out_end"], "colour": colour_of[ly["track"]],
               "label": f"{LAYER_ICONS.get(ly.get('part') or 'vocals', '🎤')} {ly['track'][:16]}"}
              for ly in lay.get("layers") or []]
    notes = _join_notes(lay, recipe, tracks, sims)
    joins = []
    for k, jn in enumerate(lay["joins"]):
        a, b = lay["pieces"][k], lay["pieces"][k + 1]
        _, colour, text = notes[k]
        joins.append({"uid": b.get("uid"), "at": jn["at"], "pre": jn["pre"], "post": jn["post"],
                      "shape": jn.get("shape", "smooth"), "bass_swap": bool(jn.get("bass_swap")),
                      "swap_at": jn.get("swap_at"), "note": text,
                      "note_colour": {"green": "#2e9e5b", "orange": "#e8890c", "red": "#d93a3a"}.get(colour, "#8a8f98"),
                      "editable": bool(editable and b.get("uid")),
                      "min_t": jn["at"] - 0.9 * (a["out_end"] - a["out_start"]),
                      "max_t": jn["at"] + 0.9 * (b["out_end"] - b["out_start"])})
    mix_timeline(duration=lay["duration"], beat_len=MX.beat_len(lay["bpm"]), pieces=pieces, layers=layers,
                 joins=joins, env=env, env_rate=rate, beats=lay["beats"], bars=lay["bars"],
                 trim_to=recipe.get("trim_to") if recipe.get("trim_to") and recipe["trim_to"] < lay["duration"]
                 else None, audio_url=audio_url, key=f"{ns}_tl")
    return lay


def _apply_timeline_drag(parts: list, ns: str = "mxm") -> None:
    """A crossfade dragged on the timeline → that join's own crossfade (its length and how much
    of it sits before the bar line), and the join row's boxes to match."""
    v = st.session_state.get(f"{ns}_tl")
    if not isinstance(v, dict) or not v.get("nonce") or v.get("nonce") == st.session_state.get(f"_{ns}_tl_nonce"):
        return
    st.session_state[f"_{ns}_tl_nonce"] = v["nonce"]
    b = next((p for p in parts if p.get("uid") == v.get("uid")), None)
    if b is None:
        return
    pre, post = max(0.0, float(v.get("pre_beats") or 0)), max(0.0, float(v.get("post_beats") or 0))
    beats = round(pre + post, 3)
    own = dict(b.get("xfade") or {})
    if isinstance(own.get("neural"), dict):          # 🧠 stays neural: the drag picks its length (4 / 8 / 16 bars)
        bars = min(NEURAL_BARS, key=lambda n: abs(n * 4 - beats))
        own["neural"] = {**own["neural"], "bars": bars}
        b["xfade"] = own
        st.session_state[f"mxm_nnb_{b['uid']}"] = bars
        return
    own["beats"] = beats
    own["where"] = MX.where_from_frac(pre / beats) if beats > 0 else "centre"
    b["xfade"] = own
    uid = b["uid"]
    st.session_state[f"mxm_xfl_{uid}"] = _len_label(beats)
    st.session_state[f"mxm_xfw_{uid}"] = own["where"]


def _len_label(beats: float) -> str:
    return next((n for n, x in XF_JOIN.items() if abs(x - beats) < 1e-6), f"🖐 {_beats_text(beats)}")


def _beats_text(b: float) -> str:
    if b >= 4 and abs(b / 4 - round(b / 4)) < 1e-6:
        n = round(b / 4)
        return f"{n} bar{'s' if n != 1 else ''}"
    return f"{b:g} beat{'s' if b != 1 else ''}"


def _where_label(w) -> str:
    if w in XF_WHERE_LABELS:
        return XF_WHERE_LABELS[w]
    try:
        f = float(w)
    except (TypeError, ValueError):
        return str(w)
    return f"🖐 {round(f * 100)} % before the line, {round((1 - f) * 100)} % after"


# ---------------------------------------------------------------------------
# Preview + save (shared by Extend and Mashup)
# ---------------------------------------------------------------------------

def _live_join(recipe: dict, tracks: dict, lay: dict) -> dict | None:
    """🔁 The open join's quick render for the lanes' loop: started (in the background) as soon as the join is
    open or changed, so the loop picks up each change by itself. Until a new render is ready the last one keeps
    playing."""
    import mixer
    from cut_editor import media_url
    ss = st.session_state
    uid = ss.get("_mxm_join_open")
    pieces = lay["pieces"]
    k = next((i for i, p in enumerate(pieces) if p.get("uid") == uid), 0) - 1
    if not uid or k < 0:
        ss.pop("_mxm_jlive", None)
        return None
    grids = {tid: MX.grid(t) for tid, t in tracks.items()}
    jr = _join_recipe(recipe, k, grids)
    trs = {t: tracks[t] for t in {p["track"] for p in jr["pieces"]}}
    path = mixer.preview_path(jr)
    a = pieces[k]
    info = {"uid": uid, "label": f"{k + 1} → {k + 2}",
            "mix_start": round(float(a["out_end"]) - (float(a["end"]) - float(jr["pieces"][0]["start"])) / float(a["speed"]), 3)}
    if path.exists():
        info["url"] = media_url(path, "audio/mpeg" if path.suffix == ".mp3" else "audio/mp4", f"mxm_jl_{path.stem}")
        ss["_mxm_jlive"] = info
        return info
    job = _render_job_for(path)
    if not (job and job.active) and _audio_ready(mixer.parts_needed(jr, trs), f"mxm_jl_{uid}"):
        job = _start_render(jr, trs, path, f"🔁 Mixing join {k + 1} → {k + 2}")
    if job is not None and job.active:
        _job_watch((job.key,))
    elif job is not None and job.state == "error":
        st.caption(f":orange[🔁 Couldn't mix the join: {job.error}]")
    last = ss.get("_mxm_jlive") or {}
    return {**info, "url": last.get("url", "") if last.get("uid") == uid else ""}


def _mashup_lanes(recipe: dict, tracks: dict, mix_url: str, rough: set = frozenset(), join_audio: dict = None) -> None:
    """🎚️ One lane per part (mix_lanes.py): trims, join drags (the hand-over moved) and join taps (open its
    settings under the lanes) come back here."""
    import mixer
    import deck_editor as DE
    import mix_lanes as ML
    from cut_editor import media_url
    grids = {tid: MX.grid(t) for tid, t in tracks.items()}
    lay = MX.layout(recipe, grids)
    if not lay["pieces"]:
        return
    joins = lay["joins"]
    data, urls = {}, {}
    for tid, t in tracks.items():
        path = mixer.local_path(t, "full")
        urls[tid] = media_url(path, "audio/mp4" if path.suffix == ".m4a" else "audio/mpeg", f"mxm_lane_{tid}") \
            if _has_file(path) else ""
        data[tid] = DE.lane_data(t, tid, grids[tid], urls[tid])
    lanes = []
    for k, p in enumerate(lay["pieces"]):
        lanes.append({"uid": p.get("uid"), "idx": k, "label": p["track"], "colour": PALETTE[k % len(PALETTE)].format(a=0.9),
                      "trk": data[p["track"]], "start": p["start"], "end": p["end"], "speed": p["speed"],
                      "out_start": p["out_start"], "out_end": p["out_end"],
                      "head": joins[k - 1]["pre"] if k > 0 else 0.0, "tail": joins[k]["post"] if k < len(joins) else 0.0})
    pieces = lay["pieces"]
    jdata = []
    for k, j in enumerate(joins):
        a, b = pieces[k], pieces[k + 1]
        jdata.append({"at": j["at"], "pre": j["pre"], "post": j["post"], "uid": b.get("uid"), "a_uid": a.get("uid"),
                      "label": _join_summary(a, b), "rough": b.get("uid") in rough,
                      "neural": bool((b.get("xfade") or {}).get("neural"))})
    val = ML.mix_lanes(duration=lay["duration"], lanes=lanes, joins=jdata, mix_url=mix_url, join_audio=join_audio,
                       key="mxm_lanes")
    if not (isinstance(val, dict) and val.get("nonce") and val["nonce"] != st.session_state.get("_mxm_lanes_nonce")):
        return
    st.session_state["_mxm_lanes_nonce"] = val["nonce"]
    parts = st.session_state.get("mxm_parts") or []
    if val.get("join") and val.get("open"):                 # a tap on a join: its settings under the lanes
        st.session_state["_mxm_join_open"] = val["join"]
    elif val.get("join"):                                    # a join dragged: the hand-over moved
        a = next((p for p in parts if p.get("uid") == val.get("a_uid")), None)
        b = next((p for p in parts if p.get("uid") == val.get("join")), None)
        if a is not None and b is not None:
            a["end"], b["start"] = round(float(val["a_end"]), 3), round(float(val["b_start"]), 3)
    else:
        for p in parts:
            if p.get("uid") == val.get("uid"):
                p["start"], p["end"] = round(float(val["start"]), 3), round(float(val["end"]), 3)
    _rerun()


def _close_join_panel() -> None:
    st.session_state.pop("_mxm_join_open", None)


def _join_panel(parts: list, rough: set):
    """🔀 The join tapped on the lanes, edited right under them. Returns (its uid, a slot for its player)."""
    uid = st.session_state.get("_mxm_join_open")
    idx = next((i for i, p in enumerate(parts) if p["uid"] == uid), None)
    if not idx:                                             # gone (or the first part — it has no join)
        st.session_state.pop("_mxm_join_open", None)
        return None, None
    a, b = parts[idx - 1], parts[idx]
    with st.container(border=True, key="mxm_join_panel"):
        c = st.columns([4, 1.1, 1.1, 0.7] if uid in rough else [4, 1.1, 0.7], vertical_alignment="center")
        c[0].markdown(f"**🔀 Join {idx} → {idx + 1}** · {a['track']} → {b['track']}")
        c[1].button("▶️ Hear it", key=f"mxm_jpplay_{uid}", width="stretch", on_click=_ask_join_preview, args=(uid,),
                    help="Hear just this join (a few bars either side).")
        if uid in rough:
            c[2].button("✨ Fix", key=f"mxm_jpfix_{uid}", width="stretch", on_click=_fix_join, args=(a["uid"], uid),
                        help="The keys clash here. Shifts the next song's key to fit, and moves the hand-over to the "
                             "best-scoring place inside the two parts as a 🧠 blend. ↶ undoes it.")
        c[-1].button("✖", key="mxm_jpclose", width="stretch", on_click=_close_join_panel, help="Close")
        st.caption("🔁 **Join** above the lanes loops this join — change it here and the loop follows a moment later.")
        _join_controls(a, b)
        return uid, st.container()


def _trim_part(uid: str, key: str) -> None:
    rng = st.session_state.get(key)
    for p in st.session_state.get("mxm_parts") or []:
        if p.get("uid") == uid and rng:
            p["start"], p["end"] = round(float(rng[0]), 3), round(float(rng[1]), 3)


def _part_trim(p: dict, g: dict, col) -> None:
    """✂️ Trim a part on its card — on bar lines or beats (the ➕ Add a part snap setting)."""
    snap = st.session_state.get("mxm_snap") or "Bars"
    marks = g["bars"] if snap == "Bars" else g["beats"]
    lines = sorted({0.0, round(float(p["start"]), 3), round(float(p["end"]), 3),
                    *[round(float(x), 3) for x in marks if 0 < x < g["duration"]], round(g["duration"], 3)})
    key = f"mxm_trim_{p['uid']}_{p['start']:.3f}_{p['end']:.3f}_{snap}"
    with col.popover("✂️", width="stretch", help="Trim this part's start and end"):
        st.select_slider(f"{p['track']} — start and end ({snap.lower()})", options=lines,
                         value=(round(float(p["start"]), 3), round(float(p["end"]), 3)), key=key,
                         format_func=format_mmss, on_change=_trim_part, args=(p["uid"], key))
        st.caption("Snap to bars or beats with **Snap to** in ➕ Add a part. Or drag the part's handles in the lanes above.")


# ---------------------------------------------------------------------------
# ▶️ Previews and 💾 saves — rendered in the background (mix_jobs.py): they finish even if the iPad sleeps
# ---------------------------------------------------------------------------

def _render_job(job, recipe: dict, tracks: dict, out: str) -> dict:
    import mixer
    info = mixer.render(recipe, tracks, Path(out), progress=lambda m: job.progress(msg=m),
                        fraction=lambda f: job.progress(f))
    side = {"energy": info["energy"], "warnings": info["warnings"], "duration": info["duration"]}
    try:
        Path(out).with_suffix(".json").write_text(json.dumps(side))
    except OSError:
        pass
    return side


def _start_render(recipe: dict, tracks: dict, out, label: str):
    """Render recipe to out in the background (or join the render already making it)."""
    import mix_jobs as J
    key = f"render:{out}"
    j = J.get(key)
    if j is not None and j.state == "done" and not Path(out).exists():
        J.forget(key)                                 # its file was cleared since — make it again
    return J.start(key, label, _render_job, json.loads(json.dumps(recipe, default=str)), dict(tracks), str(out))


def _render_job_for(out):
    import mix_jobs as J
    return J.get(f"render:{out}")


def _preview_side(out: Path) -> dict:
    """What a finished preview render noted (level curve, warnings) — kept next to the file."""
    try:
        return json.loads(Path(out).with_suffix(".json").read_text())
    except (OSError, ValueError):
        return {}


def _playbar_css() -> str:
    """The 🧩 Mashup play bar stays at the top of the screen while you scroll, on the page's own background."""
    try:
        dark = (st.context.theme.type or "light") == "dark"
    except Exception:
        dark = False
    bg = "#0e1117" if dark else "#ffffff"
    # sticky on the wrapper Streamlit puts around the keyed container (the container alone is exactly as tall
    # as its wrapper, so it would have nowhere to slide)
    return ("<style>div:has(> .st-key-mxm_playbar) {position:sticky;top:3.75rem;z-index:20}"
            f".st-key-mxm_playbar {{background:{bg};padding:.45rem .2rem .3rem;"
            "border-bottom:1px solid rgba(128,128,128,.25);box-shadow:0 6px 8px -8px rgba(0,0,0,.25)}"
            ".st-key-mxm_playbar audio {width:100%;height:2.2rem}"
            ".st-key-mxm_playbar [data-testid='stAudio'] {margin-bottom:0}</style>")


def _playbar(recipe: dict, tracks: dict, ns: str, lay: dict, notes: list) -> None:
    """▶️ Preview the whole mix: the button, the mix's length and rough joins, the render's progress and the
    one player. In 🧩 Mashup it stays at the top of the screen while you scroll the parts."""
    import mixer
    pv_path = mixer.preview_path(recipe)
    fresh = pv_path.exists()
    job = _render_job_for(pv_path)
    length = min(lay["duration"], float(recipe.get("trim_to") or lay["duration"]))
    status = [f"⏱ **{format_mmss(length)}**", f"{len(lay['pieces'])} {'part' if ns == 'mxm' else 'piece'}(s)",
              f"{lay['bpm']:.1f} BPM"]
    weak = [n for n in notes if n[1] == "red"]
    if weak:
        status.append(f"⚠️ {len(weak)} rough join(s)")
    if ns == "mxm":
        st.markdown(_playbar_css(), unsafe_allow_html=True)
    with st.container(key=f"{ns}_playbar"):
        if ns == "mxm":
            c = st.columns([1.3, 0.45, 0.45, 2.6], vertical_alignment="center")
            c[1].button("↶", key="mxm_undo", on_click=_undo, args=(-1,), width="stretch",
                        disabled=not st.session_state.get("_mxm_hist"), help="Undo the last change to the mix")
            c[2].button("↷", key="mxm_redo", on_click=_undo, args=(1,), width="stretch",
                        disabled=not st.session_state.get("_mxm_future"), help="Redo")
            c = [c[0], c[3]]
        else:
            c = st.columns([1.3, 3], vertical_alignment="center")
        ready = _audio_ready(mixer.parts_needed(recipe, tracks), ns) if not fresh else True
        busy = bool(job and job.active)
        if ready and c[0].button("▶️ Preview", key=f"{ns}_preview", type="primary", width="stretch", disabled=busy,
                                 help="Mixes it to a quick temporary file to listen to (nothing is saved). It "
                                      "carries on in the background — keep editing while it works."):
            st.session_state[f"{ns}_pv"] = {"path": str(pv_path)}
            if not fresh:
                job = _start_render(recipe, tracks, pv_path, "▶️ Mixing the preview")
                busy = job.active
        c[1].caption("  ·  ".join(status))
        if busy:
            _job_watch((job.key,))
        elif job is not None and job.state == "error":
            st.error(f"Couldn't make the preview: {job.error}")
        if fresh:
            st.audio(str(pv_path))
            for w in _preview_side(pv_path).get("warnings") or []:
                st.caption(f"⚠️ {w}")
        elif (st.session_state.get(f"{ns}_pv") or {}).get("path") and not busy:
            st.caption("The mix has changed since the last preview — ▶️ Preview again to hear it.")


def _save_job(job, recipe: dict, tracks: dict, token) -> dict:
    import mixes as MS
    rec, msgs = MS.publish(recipe, tracks, token=token, progress=lambda m: job.progress(msg=m),
                           fraction=lambda f: job.progress(f))
    return {"id": rec["id"], "msgs": msgs, "name": recipe["name"]}


def _save_section(recipe: dict, tracks: dict, default_name: str, ns: str) -> None:
    import mixes as MS
    import mix_jobs as J
    st.divider()
    editing = MS.get(recipe["id"]) if recipe.get("id") else None
    st.session_state.setdefault(f"{ns}_name", default_name)
    name = st.text_input("Name for the new track", key=f"{ns}_name", max_chars=90)
    clean = MS.safe_name(name)
    taken = MS.name_taken(name, recipe.get("id"))
    if not clean:
        st.caption("Give it a name.")
    elif taken:
        st.caption(f"⚠️ A track called **{clean}** is already in the library — pick another name.")
    replacing = editing and (editing.get("output") or {}).get("track_id") == clean
    job = J.get(st.session_state.get(f"{ns}_savejob") or "")
    saving = bool(job and job.active)
    c = st.columns(2)
    label = f"💾 Save over {clean}" if replacing else "💾 Save as a new track"
    if c[0].button(label, key=f"{ns}_save", type="primary", disabled=not clean or taken or saving, width="stretch",
                   help="Mixes it properly (AAC), adds it to the Audio-Library (and Google Drive when connected) "
                        "and analyses it like any other track — ready for the planner in a minute or two. It "
                        "carries on in the background if you leave the page."
                        + (" Replaces the track this mix made before (plans using it will pick up the new "
                           "audio)." if replacing else "")):
        from page_setup import drive_token
        job = J.start(f"save:{clean}:{time.time():.3f}", f"💾 Saving {clean}", _save_job,
                      json.loads(json.dumps({**recipe, "name": clean}, default=str)), dict(tracks), drive_token())
        st.session_state[f"{ns}_savejob"] = job.key
        saving = True
    if c[1].button("📝 Save the recipe only", key=f"{ns}_recipe", disabled=not clean or saving, width="stretch",
                   help="Keeps the mix in 📂 Saved mixes to finish later — no track is made."):
        rec = MS.save({**recipe, "name": clean})
        st.session_state[f"{ns}_id"] = rec["id"]
        st.toast(f"Recipe saved: {clean}")
    if job is None:
        return
    if saving:
        _job_watch((job.key,))
    elif job.state == "error":
        st.error(f"Couldn't save the mix: {job.error}")
    elif job.state == "done":
        res = job.result or {}
        st.success(f"✅ Saved **{res.get('name')}** — analysing it now (🎵 Audio shows the progress).")
        for m in res.get("msgs") or []:
            st.caption(f"ℹ️ {m}")
        if res.get("id") and st.session_state.get(f"{ns}_id") != res["id"]:
            st.session_state[f"{ns}_id"] = res["id"]


def _preview_and_save(recipe: dict, tracks: dict, sims: dict, default_name: str, ns: str) -> None:
    """📏 Extend: the waveform with its jumps, ▶️ Preview, 💾 Save. (🧩 Mashup lays its own screen out.)"""
    import mixer
    from cut_editor import media_url
    pv_path = mixer.preview_path(recipe)
    fresh = pv_path.exists()
    url = media_url(pv_path, "audio/mpeg" if pv_path.suffix == ".mp3" else "audio/mp4", f"{ns}_mix_audio") \
        if fresh else ""
    lay = _timeline(recipe, tracks, sims, energy=_preview_side(pv_path).get("energy") if fresh else None,
                    audio_url=url, ns=ns, editable=False)
    if lay is None:
        st.info("Nothing to play yet.")
        return
    _playbar(recipe, tracks, ns, lay, _join_notes(lay, recipe, tracks, sims))
    _save_section(recipe, tracks, default_name, ns)


# ---------------------------------------------------------------------------
# 📏 Extend / shorten
# ---------------------------------------------------------------------------

def _extend(summaries: dict) -> None:
    tracks = _mixable(summaries)
    if not tracks:
        st.info("No analysed tracks yet.")
        return
    ids = sorted(tracks, key=str.lower)
    if st.session_state.get("mxe_track") not in ids:
        st.session_state["mxe_track"] = ids[0]
    tid = st.selectbox("Track", ids, key="mxe_track", format_func=lambda x: _label(tracks[x]))
    track = full_track(tid)
    if track is None:
        st.error("Couldn't read this track.")
        return
    g = MX.grid(track)
    has_bars = bool((track.get("analysis") or {}).get("downbeats"))
    st.caption(f"{len(g['bars'])} bars of {g['bar_len']:.2f} s · {len(g['sections'])} sections · "
               f"{_key_text(track.get('key'))}"
               + ("" if has_bars else " · ⚠️ no extra analysis — bar lines guessed from the beats"))

    method = st.radio("How", ["🎯 Auto — to a length", "✋ By sections"], horizontal=True, key="mxe_method")
    recipe = {"id": st.session_state.get("mxe_id"), "kind": "extend", "bpm": None, "shift": {},
              "xfade_same": XF_SAME[st.session_state.get("mxe_xf", "Short (¼ beat)")],
              "fade_out": float(st.session_state.get("mxe_fade", 0.0)), "level_match": False}
    sims = {}
    if method.startswith("🎯"):
        pieces, trim, settings = _extend_auto(tid, track, g, sims)
        if pieces is None:
            return
        recipe.update(pieces=pieces, trim_to=trim, settings=settings)
        suffix = f"{format_mmss(trim or sum(p['end'] - p['start'] for p in pieces)).replace(':', 'm')}s"
    else:
        pieces, settings = _extend_sections(tid, g)
        recipe.update(pieces=pieces, settings=settings)
        cached = (st.session_state.get("_mx_sim") or {}).get(tid)
        if cached:                  # join match % when the bars were already listened to
            sims[tid] = (g, cached[1])
        suffix = "edit"
    with st.expander("⚙️ Joins and ending"):
        c = st.columns(2)
        st.session_state.setdefault("mxe_xf", "Short (¼ beat)")
        c[0].selectbox("Crossfade at each jump", list(XF_SAME), key="mxe_xf",
                       help="Jumps land on bar lines between bars that sound alike, so a very short "
                            "crossfade is usually best — longer ones can smear the beat.")
        st.session_state.setdefault("mxe_fade", 0.0)
        c[1].slider("Fade out at the end (s)", 0.0, 15.0, step=0.5, key="mxe_fade",
                    help="0 keeps the song's own ending.")
    if not recipe["pieces"]:
        st.info("Every section is set to 0 — nothing left to play.")
        return
    _preview_and_save(recipe, {tid: track}, sims, f"{tid} ({suffix})", "mxe")
    if st.session_state.get("mxe_id"):
        st.button("🧹 Start a new one", key="mxe_clear", on_click=_clear, args=("mxe",),
                  help="Stop editing the saved mix — the next save makes a new one.")


def _extend_auto(tid: str, track: dict, g: dict, sims: dict) -> tuple:
    dur = g["duration"]
    lo, hi = int(max(20, dur * 0.4)), int(dur * 4) + 60
    if st.session_state.get("_mxe_target_for") != tid:      # a new track: start from 1½ × its length
        if st.session_state.get("_mxe_target_for") is not None or not st.session_state.get("mxe_target"):
            st.session_state["mxe_target"] = int(round(dur * 1.5 / 5) * 5)
        st.session_state["_mxe_target_for"] = tid
    st.session_state["mxe_target"] = int(min(hi, max(lo, st.session_state.get("mxe_target") or lo)))
    with st.form("mxe_form", border=False):
        c = st.columns([2, 2, 2])
        c[0].number_input("Target length (seconds)", min_value=lo, max_value=hi,
                          step=5, key="mxe_target",
                          help=f"The track is {format_mmss(dur)} ({dur:.0f} s). Longer repeats bars "
                               f"(jumps back); shorter skips some (jumps forward).")
        c[1].toggle("Exact length", key="mxe_exact",
                    help="Fade out on exactly the target. Off: jumps only, so it ends on the song's own "
                         "ending, within about a bar of the target.")
        c[2].form_submit_button("🔍 Find the jumps", type="primary", width="stretch")
    target = float(st.session_state["mxe_target"])
    exact = bool(st.session_state.get("mxe_exact"))
    st.caption(f"Target {format_mmss(target)} — {'+' if target >= dur else '−'}"
               f"{format_mmss(abs(target - dur))} on the original {format_mmss(dur)}.")
    if not _audio_ready([tid], "mxe"):
        return None, None, None
    g, S = _features(tid)
    sims[tid] = (g, S)
    sig = (tid, target, exact, len(g["bars"]))
    found = st.session_state.get("mxe_plans")
    if not found or found.get("sig") != sig:
        plans = MX.plan_length(g, S, target, exact=exact)
        st.session_state["mxe_plans"] = found = {"sig": sig, "plans": plans}
        st.session_state["mxe_pick"] = 0
    plans = found["plans"]
    want = st.session_state.pop("_mxe_want_jumps", None)
    if want is not None:            # reopening a saved mix: pick the option it was saved with
        for k, pl in enumerate(plans):
            if [(j["at"], j["to"], j.get("repeat", 1)) for j in pl["jumps"]] == \
                    [(j["at"], j["to"], j.get("repeat", 1)) for j in want]:
                st.session_state["mxe_pick"] = k
                break
    if not plans:
        st.warning("No good jumps found in this track for that length — try ✋ By sections.")
        return None, None, None

    def plan_label(i):
        p = plans[i]
        n = sum(j.get("repeat", 1) for j in p["jumps"])
        length = target if exact and p["error"] >= -0.5 * g["bar_len"] else p["length"]
        return (f"Option {i + 1}: {format_mmss(length)} — {n} jump{'s' if n != 1 else ''}, "
                f"match {p['score'] * 100:.0f} %")

    if st.session_state.get("mxe_pick", 0) >= len(plans):
        st.session_state["mxe_pick"] = 0
    i = st.radio("Options", list(range(len(plans))), key="mxe_pick", format_func=plan_label,
                 label_visibility="collapsed")
    p = plans[i]
    for jp in p["jumps"]:
        st.caption("↪️ " + MX.describe_jump(g, jp))
    pieces = [{**x, "track": tid} for x in p["pieces"]]
    trim = target if exact and p["length"] > target + 0.05 else None
    return pieces, trim, {"method": "auto", "track": tid, "target": target, "exact": exact,
                          "jumps": [{k: jp[k] for k in ("at", "to", "repeat")} for jp in p["jumps"]]}


def _extend_sections(tid: str, g: dict) -> tuple:
    st.caption("How many times each section plays — 0 removes it, 2 plays it twice. Joins land on the "
               "sections' bar lines.")
    counts = []
    secs = g["sections"]
    per_row = 4
    for r0 in range(0, len(secs), per_row):
        cols = st.columns(per_row)
        for k in range(r0, min(r0 + per_row, len(secs))):
            s = secs[k]
            key = f"mxe_cnt_{tid}_{k}"
            st.session_state.setdefault(key, 1)
            counts.append(cols[k - r0].number_input(
                f"{k + 1}. {s['label']}  {format_mmss(s['start'])}–{format_mmss(s['end'])}",
                min_value=0, max_value=4, step=1, key=key,
                help=f"Section {s['label']} — {s['end'] - s['start']:.0f} s, energy {s['energy']:.2f} "
                     "(sections with the same letter sound alike)."))
    pieces = [{**p, "track": tid} for p in MX.pieces_from_sections(g, counts)]
    return pieces, {"method": "sections", "track": tid, "counts": counts}


# ---------------------------------------------------------------------------
# 🧩 Mashup
# ---------------------------------------------------------------------------

def _parts() -> list:
    return st.session_state.setdefault("mxm_parts", [])


def _move(i: int, step: int) -> None:
    parts = _parts()
    j = i + step
    if 0 <= j < len(parts):
        parts[i], parts[j] = parts[j], parts[i]


def _remove(i: int) -> None:
    _parts().pop(i)


def _partners(base: dict, summaries: dict) -> list:
    def info(t):
        e = t.get("energy") or []
        return {"track_id": t["track_id"], "bpm": t.get("bpm"), "key": t.get("key"),
                "energy": float(np.mean(e)) if e else 0.5}
    a = info(base)
    out = [MX.partner_score(a, info(t)) for tid, t in summaries.items() if tid != base["track_id"]]
    return sorted(out, key=lambda x: -x["score"])


def _add_part(tracks: dict, summaries: dict) -> None:
    parts = _parts()
    ids = sorted(tracks, key=str.lower)
    hint = {}
    if parts:
        last = summaries.get(parts[-1]["track"])
        if last:
            ranked = _partners(last, tracks)
            hint = {r["track_id"]: r for r in ranked}
            ids = [r["track_id"] for r in ranked] + [x for x in ids if x not in hint]
            if parts[-1]["track"] in tracks:
                ids = [parts[-1]["track"]] + [x for x in ids if x != parts[-1]["track"]]

    def fmt(x):
        h = hint.get(x)
        extra = f"  —  fits {h['score'] * 100:.0f} % ({h['key_label']})" if h else ""
        return _label(tracks[x]) + extra

    if st.session_state.get("mxm_add_track") not in ids:
        st.session_state["mxm_add_track"] = ids[0]
    tid = st.selectbox("Track", ids, key="mxm_add_track", format_func=fmt,
                       help="After the first part, tracks are listed best fit first (key, tempo, energy) "
                            "against the last part's track.")
    track = full_track(tid)
    if track is None:
        return
    g = MX.grid(track)
    secs = g["sections"]
    sec_labels = [f"{k + 1}. {s['label']}  {format_mmss(s['start'])}–{format_mmss(s['end'])}" for k, s in enumerate(secs)]
    c = st.columns(2)
    a = c[0].selectbox("From section", range(len(secs)), key=f"mxm_from_{tid}", format_func=lambda k: sec_labels[k])
    b = c[1].selectbox("To section", range(a, len(secs)), key=f"mxm_to_{tid}_{a}",
                       format_func=lambda k: sec_labels[k])
    st.session_state.setdefault("mxm_snap", "Bars")
    snap = st.segmented_control("Snap to", ["Bars", "Beats"], key="mxm_snap",
                                help="Where the part can start and end: on bar lines, or on any beat.") or "Bars"
    marks = g["bars"] if snap == "Bars" else g["beats"]
    lines = sorted({0.0, *[round(float(x), 3) for x in marks if 0 < x < g["duration"]], round(g["duration"], 3)})
    near = lambda t: min(lines, key=lambda x: abs(x - t))      # noqa: E731
    lo, hi = near(secs[a]["start"]), near(secs[b]["end"])
    rng = st.select_slider(f"Fine-tune ({'bar lines' if snap == 'Bars' else 'beats'})", options=lines,
                           value=(lo, hi), key=f"mxm_rng_{tid}_{a}_{b}_{snap}", format_func=format_mmss)
    if rng[1] - rng[0] < 1.0:
        st.caption("Pick at least a bar.")
        return
    st.caption(f"{format_mmss(rng[1] - rng[0])} · {sum(1 for x in g['bars'] if rng[0] <= x < rng[1])} bars")
    if st.button("➕ Add this part", key="mxm_add", type="primary"):
        parts.append({"track": tid, "start": float(rng[0]), "end": float(rng[1]), "gain_db": 0.0,
                      "uid": f"{time.time():.6f}"})
        if len(parts) == 1:
            st.session_state["mxm_bpm"] = round(float(track.get("bpm") or 120.0), 1)
        _rerun()


def _layers() -> list:
    return st.session_state.setdefault("mxm_layers", [])


def _remove_layer(uid: str) -> None:
    st.session_state["mxm_layers"] = [ly for ly in _layers() if ly["uid"] != uid]


def _part_stems_row(p: dict, t: dict, tracks_all: dict, summaries: dict) -> None:
    """🎤 Under a mashup part: what plays (full song / instrumental / vocals — stems from
    run_audio_analysis.bat on the laptop), and vocals from another track laid over it."""
    import mixer
    has = mixer.has_stems(t)
    with st.container(key=f"mxm_stems_{p['uid']}"):
        c = st.columns([3.2, 1.6, 2], vertical_alignment="center")
        pk = f"mxm_part_{p['uid']}"
        pending = st.session_state.setdefault("_mxm_part_pending", {}).pop(p["uid"], None)
        if pending and has:
            st.session_state[pk] = pending          # e.g. instrumental once vocals were laid over it
        st.session_state.setdefault(pk, p.get("part", "full") if has else "full")
        st.session_state.setdefault(f"_last_{pk}", st.session_state[pk])
        if mixer.has_four_stems(t):
            # 🎚️ any mix of its four stems: one tap each (all four = the full song)
            sk = f"mxm_stemsel_{p['uid']}"
            st.session_state[sk] = list(mixer.part_stems(st.session_state.get(pk)))
            c[0].pills("Stems", list(STEM_LABELS), selection_mode="multi", key=sk, format_func=STEM_LABELS.get,
                       label_visibility="collapsed", on_change=_set_part_stems, args=(p["uid"], p["track"]),
                       help="What of this song plays here — tap a stem to take it out or put it back. Vocals off = "
                            "the instrumental (for another song's vocals over it); just vocals = an a cappella.")
            p["part"] = st.session_state.get(pk) or "full"
        else:
            _part_play_switch(p, pk, has, c[0])
        others = sorted(tracks_all, key=str.lower)
        with c[1].popover("➕ Lay over…", width="stretch", disabled=not others):
            if others:   # a disabled popover's contents still run
                _add_layer_form(p, t, others, tracks_all, summaries)
        _part_layers(p)


def _set_part_stems(uid: str, tid: str) -> None:
    import mixer
    ss = st.session_state
    sel = ss.get(f"mxm_stemsel_{uid}") or []
    pk = f"mxm_part_{uid}"
    if not sel:                                    # every stem off: keep what was playing
        return
    ss[pk] = ss[f"_last_{pk}"] = mixer.part_for_stems(sel, full_track(tid))


def _part_play_switch(p: dict, pk: str, has: bool, col) -> None:
    """Full / No vocals / Vocals — for a song split into vocals + instrumental only (or not at all)."""
    if st.session_state.get(pk) not in PART_LABELS:          # a stem mix from a song that had four stems
        st.session_state[pk] = "full"
    col.segmented_control("Play", list(PART_LABELS), key=pk, format_func=PART_ICON_LABELS.get,
                          disabled=not has, label_visibility="collapsed", on_change=_keep_pick, args=(pk,),
                          help="What of this track plays here: the full song, the song without its vocals, or "
                               "just the vocals. Needs its stems (run_audio_analysis.bat on the laptop)." if has
                          else "This track has no stems yet — run run_audio_analysis.bat on the laptop (it makes "
                               "stems) to play just its instrumental or vocals.")
    p["part"] = st.session_state.get(pk) or st.session_state.get(f"_last_{pk}") or "full"


def _part_layers(p: dict) -> None:
    """The layers laid over a part: what, its level, 🗑."""
    for ly in [ly for ly in _layers() if ly["on"] == p["uid"]]:
        lc = st.columns([5, 2, 1], vertical_alignment="center")
        icon = LAYER_ICONS.get(ly.get("part") or "vocals", "🎤")
        what = PART_LABELS.get(ly.get("part") or "vocals", "").lower()
        lc[0].caption(f"{icon} {ly['track']} {what} {format_mmss(ly['start'])}–{format_mmss(ly['end'])}"
                      + (f", from bar {int(ly['offset_bars']) + 1} of the part" if ly.get("offset_bars") else ""))
        gk = f"mxm_lgain_{ly['uid']}"
        st.session_state.setdefault(gk, float(ly.get("gain_db") or 0.0))
        ly["gain_db"] = lc[1].number_input("Layer level (dB)", -12.0, 12.0, step=1.0, key=gk,
                                           label_visibility="collapsed", help="Level of this layer (dB)")
        lc[2].button("🗑", key=f"mxm_lrm_{ly['uid']}", on_click=_remove_layer, args=(ly["uid"],))


def _section_rows(tr: dict, g: dict) -> list:
    """The song's sections with their bar counts (long enough to use: 4 bars or more when there's anything longer)."""
    out = []
    for sec in g["sections"]:
        a, b = MX.snap_to_bar(g, sec["start"]), MX.snap_to_bar(g, sec["end"])
        n = max(1, int(round((b - a) / g["bar_len"])))
        if b > a:
            out.append({**sec, "start": a, "end": b, "bars": n})
    longer = [x for x in out if x["bars"] >= 4]
    return longer or out


def _swap_options(p: dict, t: dict, tracks_all: dict, summaries: dict, top: int = 6) -> list:
    """🔁 What could play instead of part p: the song's other sections (similar length, more energy ranks
    higher) and the best section of each of the 3 best-fitting other songs. [{"track", "start", "end",
    "label", "score"}], best first."""
    g = MX.grid(t)
    cur_bars = max(1.0, (float(p["end"]) - float(p["start"])) / g["bar_len"])
    rows = []
    for sec in _section_rows(t, g):
        if abs(sec["start"] - float(p["start"])) < 0.05 and abs(sec["end"] - float(p["end"])) < 0.05:
            continue
        fit_len = 1 - min(1.0, abs(sec["bars"] - cur_bars) / cur_bars)
        rows.append({"track": p["track"], "start": sec["start"], "end": sec["end"], "bars": sec["bars"],
                     "label": f"🎵 {sec['label']} · {sec['bars']} bars · {format_mmss(sec['start'])}",
                     "score": 0.55 * float(sec.get("energy") or 0) + 0.45 * fit_len})
    base = summaries.get(p["track"])
    if base:
        for r in _partners(base, {k: v for k, v in summaries.items() if k in tracks_all})[:3]:
            tr = full_track(r["track_id"])
            if tr is None:
                continue
            gb = MX.grid(tr)
            best = MX.best_section(tr, "energy")
            a, b = MX.snap_to_bar(gb, best["start"]), MX.snap_to_bar(gb, best["end"])
            if b <= a:
                continue
            rows.append({"track": r["track_id"], "start": a, "end": b, "bars": int(round((b - a) / gb["bar_len"])),
                         "label": f"🔀 {r['track_id']} · {best['label']} · {format_mmss(a)} · fits {r['score'] * 100:.0f} %",
                         "score": float(r["score"])})
    return sorted(rows, key=lambda x: -x["score"])[:top]


def _swap_part(uid: str, opt: dict) -> None:
    """🔁 Put opt in part uid's place — its joins, level and vocal layers stay; a new song gets the key shift
    that fits the part before it."""
    import mixer
    ss = st.session_state
    parts = ss.get("mxm_parts") or []
    i = next((k for k, p in enumerate(parts) if p["uid"] == uid), None)
    if i is None:
        return
    p = parts[i]
    new_song = opt["track"] != p["track"]
    p["track"], p["start"], p["end"] = opt["track"], round(float(opt["start"]), 3), round(float(opt["end"]), 3)
    t = full_track(opt["track"])
    part = p.get("part", "full")
    if t is not None and part != "full" and (not mixer.has_stems(t) or
                                             (part.startswith(mixer.STEM_MIX) and not mixer.has_four_stems(t))):
        p["part"] = "full"
        ss[f"mxm_part_{uid}"] = ss[f"_last_mxm_part_{uid}"] = "full"
    if new_song and i > 0 and t is not None:
        prev = parts[i - 1]
        pt = full_track(prev["track"])
        if pt is not None and f"mxm_shift_{opt['track']}" not in ss:
            ss[f"mxm_shift_{opt['track']}"] = int(K.best_shift(
                K.shifted(pt.get("key"), int(ss.get(f"mxm_shift_{prev['track']}", 0) or 0)), t.get("key")))
    ss.pop(f"_mxm_aud_{uid}", None)
    ss["mxm_neural_msg"] = f"🔁 Part {i + 1} is now {opt['label'].split(' · fits')[0]}. ↶ undoes it."


def _audition(tid: str, start: float, end: float, ns: str) -> None:
    """▶ The original song from start to end (fetched from Drive first if it isn't on the server)."""
    import mixer
    t = full_track(tid)
    if t is None:
        return
    if _audio_ready([tid], ns):
        st.audio(str(mixer.local_path(t)), start_time=int(start), end_time=int(np.ceil(end)))


def _swap_menu(p: dict, t: dict, tracks_all: dict, summaries: dict, col) -> None:
    uid = p["uid"]
    with col.popover("🔁", width="stretch", help="Swap this part: another section of the song, or the best part of a song that "
                               "fits — its joins stay as they are"):
        st.markdown("**🔁 Swap this part for…**")
        opts = _swap_options(p, t, tracks_all, summaries)
        if not opts:
            st.caption("Nothing else to swap in yet.")
            return
        ak = f"_mxm_aud_{uid}"
        for i, o in enumerate(opts):
            c = st.columns([3.2, 1, 1.5], vertical_alignment="center")
            c[0].caption(o["label"])
            c[1].button("⏹" if st.session_state.get(ak) == i else "▶", key=f"mxm_swpl_{uid}_{i}", width="stretch",
                        on_click=lambda k=ak, j=i: st.session_state.__setitem__(
                            k, None if st.session_state.get(k) == j else j), help="Listen to it (the original song)")
            c[2].button("Use", key=f"mxm_swuse_{uid}_{i}", width="stretch", type="primary" if i == 0 else "secondary",
                        on_click=_swap_part, args=(uid, o))
            if st.session_state.get(ak) == i:
                _audition(o["track"], o["start"], o["end"], f"mxm_swa_{uid}")


def _quick_layer(uid: str, tid: str, start: float, end: float) -> None:
    """🎤 Lay tid's vocals (start → end) over part uid, from its first bar, and play that part without its own vocals."""
    import mixer
    ss = st.session_state
    _layers().append({"uid": f"{time.time():.6f}", "on": uid, "track": tid, "part": "vocals",
                      "start": round(float(start), 3), "end": round(float(end), 3), "offset_bars": 0,
                      "gain_db": 0.0, "fade_beats": LAYER_FADES["½ beat"]})
    host = next((p for p in ss.get("mxm_parts") or [] if p["uid"] == uid), None)
    if host is not None and (t := full_track(host["track"])) is not None and mixer.has_stems(t):
        ss.setdefault("_mxm_part_pending", {})[uid] = "instrumental"
    ss["mxm_neural_msg"] = f"🎤 {tid}'s vocals laid over the part — 🎹 it now plays without its own. ↶ undoes it."


def _layer_picks(p: dict, tracks_all: dict, summaries: dict, top: int = 3) -> list:
    """The most-sung section of each of the best-fitting songs that have stems (their vocals can go over)."""
    import mixer
    base = summaries.get(p["track"])
    if not base:
        return []
    out = []
    for r in _partners(base, {k: v for k, v in summaries.items() if k in tracks_all and k != p["track"]}):
        tr = full_track(r["track_id"])
        if tr is None or not mixer.has_stems(tr):
            continue
        g = MX.grid(tr)
        sec = MX.best_section(tr, "vocals")
        a, b = MX.snap_to_bar(g, sec["start"]), MX.snap_to_bar(g, sec["end"])
        if b > a:
            out.append({"track": r["track_id"], "start": a, "end": b, "score": r["score"],
                        "label": f"🎤 {r['track_id']} · {sec['label']} · {format_mmss(a)}–{format_mmss(b)} · "
                                 f"fits {r['score'] * 100:.0f} %"})
        if len(out) >= top:
            break
    return out


def _add_layer_form(p: dict, host: dict, others: list, tracks_all: dict, summaries: dict) -> None:
    """➕ Lay another track (its full song, or with stems its vocals or instrumental) over part p — quick picks
    (the most-sung parts of the best-fitting songs, one tap each) first, then pick it yourself."""
    if not others:
        return
    picks = _layer_picks(p, {k: tracks_all[k] for k in others if k in tracks_all}, summaries)
    if picks:
        st.markdown("**🎤 Quick picks** — their vocals over this part")
        ak = f"_mxm_laud_{p['uid']}"
        for i, o in enumerate(picks):
            c = st.columns([5, 1, 1.6], vertical_alignment="center")
            c[0].caption(o["label"])
            c[1].button("⏹" if st.session_state.get(ak) == i else "▶", key=f"mxm_lqpl_{p['uid']}_{i}", width="stretch",
                        on_click=lambda k=ak, j=i: st.session_state.__setitem__(
                            k, None if st.session_state.get(k) == j else j), help="Listen to it (the original song)")
            c[2].button("Lay it over", key=f"mxm_lq_{p['uid']}_{i}", width="stretch", type="primary" if i == 0 else "secondary",
                        on_click=_quick_layer, args=(p["uid"], o["track"], o["start"], o["end"]))
            if st.session_state.get(ak) == i:
                _audition(o["track"], o["start"], o["end"], f"mxm_lqa_{p['uid']}")
        st.markdown("**Or pick it yourself**")
    base = summaries.get(p["track"])
    ranked = {}
    if base:
        ranked = {r["track_id"]: r for r in _partners(base, {k: summaries[k] for k in others if k in summaries})}
        others = sorted(others, key=lambda x: -ranked.get(x, {"score": -1})["score"])
    k = f"mxm_ltrack_{p['uid']}"
    if st.session_state.get(k) not in others:
        st.session_state[k] = others[0]
    tid = st.selectbox("Track", others, key=k,
                       format_func=lambda x: x + (f"  —  fits {ranked[x]['score'] * 100:.0f} %" if x in ranked else "")
                       + ("  🎤" if (tracks_all.get(x) or {}).get("stems") else ""))
    tr = full_track(tid)
    if tr is None:
        return
    import mixer
    has = mixer.has_stems(tr)
    whats = ["vocals", "instrumental", "full"] if has else ["full"]
    wk = f"mxm_lwhat_{p['uid']}_{tid}"
    if st.session_state.get(wk) not in whats:
        st.session_state[wk] = whats[0]
    what = st.radio("What plays", whats, key=wk, horizontal=True, format_func=PART_LABELS.get,
                    help="Vocals over this part's music is the classic mashup (needs stems — 🎤 in the list). "
                         "Without stems you can lay the full song over — turn one of the two down a few dB.")
    if not has:
        st.caption("No stems for this track yet, so only its full song can be laid over — vocals or "
                   "instrumental need stems (run_audio_analysis.bat on the laptop).")
    g = MX.grid(tr)
    secs = g["sections"]
    best = MX.best_section(tr, "vocals" if what == "vocals" else "energy")
    labels = [f"{i + 1}. {s_['label']}  {format_mmss(s_['start'])}–{format_mmss(s_['end'])}"
              + ("  ⭐" if s_ is best else "") for i, s_ in enumerate(secs)]
    si = st.selectbox("Section", range(len(secs)), key=f"mxm_lsec_{p['uid']}_{tid}",
                      index=secs.index(best) if best in secs else 0, format_func=lambda i: labels[i],
                      help="⭐ = the section with the most singing (vocals) or the most energy.")
    c = st.columns(2)
    off = c[0].number_input("Start how many bars into the part", 0, 64, 0, key=f"mxm_loff_{p['uid']}")
    fk = f"mxm_lfade_{p['uid']}"
    st.session_state.setdefault(fk, "½ beat" if what == "vocals" else "1 bar")
    fade = c[1].selectbox("Fade in / out", list(LAYER_FADES), key=fk)
    if st.button("➕ Lay it over", key=f"mxm_ladd_{p['uid']}", type="primary", width="stretch"):
        sec = secs[si]
        _layers().append({"uid": f"{time.time():.6f}", "on": p["uid"], "track": tid, "part": what,
                          "start": MX.snap_to_bar(g, sec["start"]), "end": MX.snap_to_bar(g, sec["end"]),
                          "offset_bars": int(off), "gain_db": 0.0, "fade_beats": LAYER_FADES[fade]})
        if what == "vocals" and mixer.has_stems(host):           # their vocals over this part's music
            st.session_state["_mxm_part_pending"][p["uid"]] = "instrumental"
        _rerun()


NEURAL = "🧠 Neural"
NEURAL_BARS = (1, 2, 4, 8, 16)


def _neural_autoplace(a_uid: str, b_uid: str) -> None:
    """📍 Place a 🧠 neural join (core/stem_mix.plan_transition): the outgoing part ends a blend's length
    after the last phrase line in it; the incoming part keeps its start (its bars before it play
    under the blend) — moved later only when there isn't room for the blend before it."""
    parts = st.session_state.get("mxm_parts") or []
    a = next((p for p in parts if p["uid"] == a_uid), None)
    b = next((p for p in parts if p["uid"] == b_uid), None)
    if a is None or b is None:
        return
    ta, tb = full_track(a["track"]), full_track(b["track"])
    if ta is None or tb is None:
        return
    nn = (b.get("xfade") or {}).get("neural") or {}
    bars, profile = int(nn.get("bars") or 8), nn.get("profile") or "smooth"
    gb = MX.grid(tb)
    room = bars * gb["bar_len"]
    first = float(gb["bars"][0]) if len(gb["bars"]) else 0.0
    b_at = max(float(b["start"]), MX.snap_to_bar(gb, first + room))
    try:
        plan = SM.plan_transition(ta, tb, bars, profile, a_from=float(a["start"]), a_until=float(a["end"]), b_at=b_at)
    except ValueError as e:
        st.session_state["mxm_neural_msg"] = f"📍 Couldn't place it: {e} — make part {parts.index(a) + 1} longer."
        return
    a["end"] = round(plan.mix_start_point + plan.duration_sec, 3)
    b["start"] = round(b_at, 3)
    where = "on a phrase line" if plan.notes and plan.notes[0].startswith("Starts on") else "on a bar line"
    st.session_state["mxm_neural_msg"] = (f"📍 Placed: the {bars}-bar blend starts at {format_mmss(plan.mix_start_point)} "
                                          f"in {a['track']} ({where}); {b['track']} is fully in at "
                                          f"{format_mmss(b_at)}.")


CURVE_HELP = ("How each side fades. Even = the profile as it is; Quick swap = both change over fast in the middle; "
              "Early out = A drops away first; Late in = B waits then arrives; Long overlap = both stay up together; "
              "Hold then cut = nothing happens until the end; Dip = A goes early and B comes late. "
              "Shape it further with the ● handles in the two-deck editor.")


def _preset_label(name) -> str:
    return SM.CURVE_PRESETS[name]["label"] if name in SM.CURVE_PRESETS else "✋ Hand-shaped"


def _tx_set_bars(stk: str) -> None:
    v = st.session_state.get("mxt_blend")
    if v and stk in st.session_state:
        st.session_state[stk]["bars"] = int(v)
        st.session_state[stk]["from"] = f"bars:{v}"


def _tx_set_preset(stk: str, pick_key: str) -> None:
    name = st.session_state.get(pick_key)
    if name in SM.CURVE_PRESETS and stk in st.session_state:
        s = st.session_state[stk]
        s["curves"] = None if name == "even" else SM.norm_curves({"preset": name})
        s["from"] = f"preset:{name}"


def _join_keys(uid: str) -> dict:
    return {"len": f"mxm_xfl_{uid}", "where": f"mxm_xfw_{uid}", "shape": f"mxm_xfs_{uid}", "bass": f"mxm_xfb_{uid}",
            "profile": f"mxm_nnp_{uid}", "bars": f"mxm_nnb_{uid}", "curve": f"mxm_nnc_{uid}"}


JOIN_DEFAULT = "Default"


def _join_init(a: dict, b: dict) -> None:
    """The join's boxes from the part's own "xfade" (the truth). Streamlit forgets a widget's value whenever it
    isn't drawn (another Mix mode, the 🎚️ Transition editor…), so each box is rebuilt from it then — a 🧠 join
    stays 🧠 (it used to come back as "Default" and overwrite the join — 23 Oct 2026)."""
    ss, k = st.session_state, _join_keys(b["uid"])
    own = b.get("xfade") or {}
    same = a["track"] == b["track"]
    if k["len"] not in ss:
        ss[k["len"]] = NEURAL if own.get("neural") else \
            _len_label(float(own["beats"])) if own.get("beats") is not None else JOIN_DEFAULT
    ss.setdefault(k["where"], own.get("where") if own.get("where") is not None else "centre")
    ss.setdefault(k["shape"], own.get("shape") or ("linear" if same else "smooth"))
    ss.setdefault(k["bass"], bool(own.get("bass_swap")))
    nn = own.get("neural") or {}
    ss.setdefault(k["profile"], nn.get("profile") or "smooth")
    ss.setdefault(k["bars"], int(nn.get("bars") or 8))
    cur = SM.preset_of(nn.get("curves")) or "hand"
    opts = list(SM.CURVE_PRESETS) + (["hand"] if cur == "hand" else [])
    if ss.get(k["curve"], "∅") not in opts:
        ss[k["curve"]] = cur


def _join_apply(a: dict, b: dict) -> None:
    """The join's "xfade" from its boxes — done for every join before anything is drawn, so the lanes and the
    ▶️ Preview at the top always match the boxes below."""
    ss, k = st.session_state, _join_keys(b["uid"])
    own = b.get("xfade") or {}
    same = a["track"] == b["track"]
    length = ss.get(k["len"], JOIN_DEFAULT)
    if length == NEURAL:
        pick = ss.get(k["curve"])
        nn = own.get("neural") or {}
        curves = nn.get("curves") if pick == "hand" else (None if pick in (None, "even") else SM.norm_curves({"preset": pick}))
        b["xfade"] = {"neural": {"profile": ss.get(k["profile"]) or "smooth", "bars": int(ss.get(k["bars"]) or 8),
                                 **({"curves": curves} if curves else {})}}
        return
    where, shape, bass = ss.get(k["where"], "centre"), ss.get(k["shape"]), bool(ss.get(k["bass"]))
    if length == "Cut":
        where, bass = "centre", False
    if length == JOIN_DEFAULT:
        beats = None
    elif length in XF_JOIN:
        beats = XF_JOIN[length]
    else:
        beats = float(own.get("beats") or 0.0)            # 🖐 dragged on the timeline
    if beats is None and where == "centre" and shape == ("linear" if same else "smooth") and not bass:
        b.pop("xfade", None)
    else:
        b["xfade"] = {"beats": beats, "where": where, "shape": shape, **({"bass_swap": True} if bass else {})}


def _join_summary(a: dict, b: dict) -> str:
    """One line for the join's chip: "🧠 Vocal swap · 8 bars", "1 bar · centred · smooth"…"""
    xf = b.get("xfade") or {}
    if xf.get("neural"):
        nn = xf["neural"]
        return f"🧠 {SM.PROFILES[nn.get('profile') or 'smooth'].split(' — ')[0]} · {int(nn.get('bars') or 8)} bars"
    length = st.session_state.get(_join_keys(b["uid"])["len"], JOIN_DEFAULT)
    if length == JOIN_DEFAULT:
        length = st.session_state.get("mxm_xf", "1 bar") if a["track"] != b["track"] else "¼ beat"
    if length == "Cut":
        return "✂️ Cut on the bar line"
    bits = [length]
    if xf.get("where") and xf["where"] != "centre":
        bits.append({"before": "next comes in early", "after": "this carries on"}.get(xf["where"], "🖐 as dragged"))
    if xf.get("shape") and xf["shape"] != ("linear" if a["track"] == b["track"] else "smooth"):
        bits.append(XF_SHAPE_LABELS.get(xf["shape"], xf["shape"]).lower())
    if xf.get("bass_swap"):
        bits.append("🎛️ bass swap")
    return " · ".join(bits)


PROFILE_ICONS = {"smooth": "🌊", "vocal_swap": "🎤", "drums_first": "🥁", "bass_drop": "💥"}


def _profile_label(k: str) -> str:
    return f"{PROFILE_ICONS.get(k, '🧠')} {SM.PROFILES[k].split(' — ')[0]}"


def _neural_controls(a: dict, b: dict) -> None:
    """🧠 Inside the join's settings: blend style, length and fade shape (one tap each), 📍 auto-place,
    the two-deck editor."""
    import mixer
    uid, k = b["uid"], _join_keys(b["uid"])
    profile = _pick(st, "Blend style", list(SM.PROFILES), k["profile"], _profile_label,
                    help="  \n".join(SM.PROFILES.values()))
    bars = _pick(st, "Blend length", list(NEURAL_BARS), k["bars"], lambda n: f"{n} bar{'s' if n != 1 else ''}",
                 help="How long the two songs blend.")
    own = (b.get("xfade") or {}).get("neural") or {}
    cur = SM.preset_of(own.get("curves")) or "hand"
    opts = list(SM.CURVE_PRESETS) + (["hand"] if cur == "hand" else [])
    _pick(st, "Fade shape", opts, k["curve"], _preset_label, help=CURVE_HELP)
    c = st.columns(2)
    c[0].button("📍 Place on a phrase", key=f"mxm_nnplace_{uid}", width="stretch", on_click=_neural_autoplace,
                args=(a["uid"], uid),
                help="Sets where the blend starts in this part (the last phrase line with room for it) and "
                     "makes sure the next part has its bars before it to blend in with.")
    c[1].button("🎚️ Two-deck editor", key=f"mxm_nnedit_{uid}", on_click=_open_join_in_transition,
                args=(dict(a), dict(b)), width="stretch",
                help="Track A over track B: drag B to where it comes in, play each with a cursor, hear the "
                     "transition live, then ✅ Apply back here.")
    ta, tb = full_track(a["track"]), full_track(b["track"])
    missing = [x["track"] for x, t in ((a, ta), (b, tb)) if t is not None and not mixer.has_four_stems(t)]
    if missing:
        st.caption(f":orange[🧠 {', '.join(dict.fromkeys(missing))} not split into drums / bass / other yet — "
                   "run run_audio_analysis.bat on the laptop. Until then this join is an ordinary crossfade.]")
    else:
        st.caption(f":gray[🧠 {SM.PROFILES[profile]} The next part comes in {bars} bars early; each stem gets its "
                   "own curve.]")


def _join_controls(a: dict, b: dict) -> None:
    """🔀 Inside the join's settings (one tap each): its length — or 🧠 neural — where it sits around the bar
    line, its shape and 🎛️ bass swap (stored on the incoming part as "xfade"; "Default" = ⚙️'s setting)."""
    k = _join_keys(b["uid"])
    same = a["track"] == b["track"]
    len_opts = [JOIN_DEFAULT, *XF_JOIN] + ([] if same else [NEURAL])
    if st.session_state[k["len"]] not in len_opts:
        len_opts.append(st.session_state[k["len"]])
    length = _pick(st, "Crossfade", len_opts, k["len"],
                   lambda x: f"Default ({st.session_state.get('mxm_xf', '1 bar')})" if x == JOIN_DEFAULT and not same
                   else ("Default (¼ beat)" if x == JOIN_DEFAULT else x),
                   help="How long both parts play together while one fades into the other. Long ones (4–16 bars) "
                        "blend the two songs over each other, DJ style. 🧠 Neural blends them stem by stem. Or "
                        "drag the crossfade's ◆ edges on the 📈 waveform.")
    if length == NEURAL:
        _neural_controls(a, b)
        return
    cut = length == "Cut"
    where_opts = list(XF_WHERE_LABELS)
    if st.session_state[k["where"]] not in where_opts:
        where_opts.append(st.session_state[k["where"]])
    _pick(st, "Where", where_opts, k["where"], _where_label, disabled=cut,
          help="Both sides: half before the bar line, half after. Next song comes in early: it starts under the "
               "end of this part and has taken over by the line. This song carries on: it keeps playing under the "
               "start of the next part. 🖐 = as dragged on the waveform.")
    c = st.columns([2, 1.3], vertical_alignment="bottom")
    _pick(c[0], "Shape", list(XF_SHAPE_LABELS), k["shape"], XF_SHAPE_LABELS.get, disabled=cut,
          help="Smooth (equal power) keeps the loudness even when two different songs blend; straight suits the "
               "same audio (a jump within one track).")
    st.session_state[k["bass"]] = bool(st.session_state.get(k["bass"]))
    c[1].toggle("🎛️ Bass swap", key=k["bass"], disabled=cut,
                help="Only one song's bass plays during the crossfade: the outgoing song keeps its low end until "
                     "the middle of the fade, then hands it over on the beat — stops a long blend getting muddy.")


def _ask_join_preview(uid: str) -> None:
    st.session_state["_mxm_jreq"] = uid


def _fix_join(a_uid: str, b_uid: str) -> None:
    """✨ Fix a rough join between two songs: shift the incoming song's key to fit the outgoing one, then move
    the hand-over to the best-scoring transition (core/transitions.suggest) that fits inside the two parts, as a
    🧠 blend."""
    from core import transitions as TR
    ss = st.session_state
    parts = ss.get("mxm_parts") or []
    a = next((p for p in parts if p["uid"] == a_uid), None)
    b = next((p for p in parts if p["uid"] == b_uid), None)
    if a is None or b is None:
        return
    ta, tb = full_track(a["track"]), full_track(b["track"])
    if ta is None or tb is None:
        return
    done = []
    ka = K.shifted(ta.get("key"), int(ss.get(f"mxm_shift_{a['track']}", 0) or 0))
    shift = int(K.best_shift(ka, tb.get("key")))
    if shift != int(ss.get(f"mxm_shift_{b['track']}", 0) or 0):
        ss[f"mxm_shift_{b['track']}"] = shift
        done.append(f"{b['track']} shifted {shift:+d} semitone{'s' if abs(shift) != 1 else ''} to fit the key")
    ga, gb = MX.grid(ta), MX.grid(tb)
    fits = [sg for sg in TR.suggest(ta, tb, top=12)
            if float(a["start"]) + 4 * ga["bar_len"] <= sg["a_end"] <= ga["duration"]
            and sg["b_full"] <= float(b["end"]) - 4 * gb["bar_len"]]
    if fits:
        sg = fits[0]
        a["end"], b["start"] = round(float(sg["a_end"]), 3), round(float(sg["b_full"]), 3)
        b["xfade"] = {"neural": {"profile": sg["profile"], "bars": int(sg["bars"])}}
        k = _join_keys(b_uid)
        ss[k["len"]], ss[k["profile"]], ss[k["bars"]], ss[k["curve"]] = NEURAL, sg["profile"], int(sg["bars"]), "even"
        for key in (k["len"], k["profile"], k["bars"], k["curve"]):
            ss[f"_last_{key}"] = ss[key]
        done.append(f"the hand-over moved to {format_mmss(sg['a_end'])} in {a['track']} → {b['track']} fully in at "
                    f"{format_mmss(sg['b_full'])}, a {_profile_label(sg['profile'])} blend over {sg['bars']} bars "
                    f"(fits {sg['score'] * 100:.0f} %)")
    ss["mxm_neural_msg"] = ("✨ Fixed: " + "; ".join(done) + ". ↶ undoes it.") if done else \
        "✨ Nothing better found for this join — try 🎚️ the two-deck editor."


def _join_row(a: dict, b: dict, i: int, rough: bool = False, open_uid: str = None) -> object:
    """🔀 The join from part i to part i + 1 as one line: a chip that opens its settings, ▶️ to hear just this
    join (and ✨ Fix when it's rough). Returns a slot under it for that join's player (filled once the recipe is
    known)."""
    uid = b["uid"]
    if uid == open_uid:
        with st.container(key=f"mxm_join_{uid}"):
            st.caption(f"🔀 Join {i} → {i + 1} · {_join_summary(a, b)} — open under the lanes ↑")
        return None
    with st.container(key=f"mxm_join_{uid}"):
        c = st.columns([5, 1.2, 1.2] if rough else [5, 1.2], vertical_alignment="center")
        with c[0].popover(("⚠️ " if rough else "") + f"🔀 {i} → {i + 1} · {_join_summary(a, b)}", width="stretch"):
            _join_controls(a, b)
        c[1].button("▶️", key=f"mxm_jplay_{uid}", width="stretch", on_click=_ask_join_preview, args=(uid,),
                    help="Hear just this join (a few bars either side) — quicker than previewing the whole mix.")
        if rough:
            c[2].button("✨ Fix", key=f"mxm_jfix_{uid}", width="stretch", on_click=_fix_join, args=(a["uid"], uid),
                        help="The keys clash here. Shifts the next song's key to fit, and moves the hand-over to "
                             "the best-scoring place inside the two parts as a 🧠 blend. ↶ undoes it.")
        return st.container()


def _join_recipe(recipe: dict, k: int, grids: dict) -> dict:
    """The mix cut down to join k (pieces k → k + 1): the outgoing part's last few bars and the blend, then
    8 bars of the incoming part."""
    a, b = dict(recipe["pieces"][k]), dict(recipe["pieces"][k + 1])
    ga, gb = grids[a["track"]], grids[b["track"]]
    xf = b.get("xfade") or {}
    if xf.get("neural"):
        blend = int(xf["neural"].get("bars") or 8)
    else:
        beats = xf.get("beats")
        beats = float(beats if beats is not None else
                      (recipe.get("xfade_same") if a["track"] == b["track"] else recipe.get("xfade_other")) or 4.0)
        blend = max(1, int(np.ceil(beats / 4)))
    a["start"] = round(max(float(a["start"]), float(a["end"]) - (blend + 6) * ga["bar_len"]), 3)
    b["end"] = round(min(float(b["end"]), float(b["start"]) + 8 * gb["bar_len"]), 3)
    keep = {a["track"], b["track"]}
    return {**recipe, "id": None, "pieces": [a, b], "layers": [], "fade_in": 0.3, "fade_out": 1.0,
            "trim_to": None, "settings": None, "shift": {t: v for t, v in (recipe.get("shift") or {}).items() if t in keep}}


def _join_player(slot, recipe: dict, k: int, tracks: dict, grids: dict) -> None:
    import mixer
    uid = recipe["pieces"][k + 1]["uid"]
    sk = f"_mxm_jpv_{uid}"
    jr = _join_recipe(recipe, k, grids)
    path = mixer.preview_path(jr)
    if st.session_state.get("_mxm_jreq") == uid:
        st.session_state.pop("_mxm_jreq", None)
        st.session_state[sk] = str(path)
        with slot:
            if _audio_ready(mixer.parts_needed(jr, {t: tracks[t] for t in {p["track"] for p in jr["pieces"]}}),
                            f"mxm_j_{uid}"):
                _start_render(jr, {t: tracks[t] for t in {p["track"] for p in jr["pieces"]}}, path,
                              f"▶️ Mixing join {k + 1} → {k + 2}")
            else:
                st.session_state[f"_mxm_jwait_{uid}"] = True
    asked = st.session_state.get(sk)
    if not asked:
        return
    with slot:
        if asked != str(path):
            st.caption("This join has changed — ▶️ to hear it again.")
            return
        job = _render_job_for(path)
        if path.exists():
            st.audio(str(path))
        elif job is not None and job.active:
            _job_watch((job.key,))
        elif job is not None and job.state == "error":
            st.error(f"Couldn't mix the join: {job.error}")
        elif st.session_state.get(f"_mxm_jwait_{uid}"):
            if _audio_ready(mixer.parts_needed(jr, tracks), f"mxm_j_{uid}"):
                st.session_state.pop(f"_mxm_jwait_{uid}", None)
                _start_render(jr, tracks, path, f"▶️ Mixing join {k + 1} → {k + 2}")
                _rerun()


def _mashup(summaries: dict) -> None:
    tracks_all = _mixable(summaries)
    if len(tracks_all) < 1:
        st.info("No analysed tracks yet.")
        return
    parts = _parts()
    for p in parts:
        p.setdefault("uid", f"{time.time():.6f}-{id(p)}")
        p.setdefault("part", "full")
    layers = _layers()
    used = list(dict.fromkeys([p["track"] for p in parts] + [ly["track"] for ly in layers]))
    tracks = {tid: full_track(tid) for tid in used}
    if any(t is None for t in tracks.values()):
        st.error("A track in this mashup is no longer in the library.")
        return
    _prefetch(used)                                   # ⬇️ songs + stems start coming from Drive straight away

    with st.expander("➕ Add a part", expanded=not parts):
        _add_part(tracks_all, summaries)

    if not parts:
        st.caption("Add parts from one or more tracks — they play in order, each joined to the next on a bar line.")
        return

    # settings
    with st.expander("⚙️ Tempo, keys and joins", expanded=False):
        c = st.columns(3)
        st.session_state.setdefault("mxm_match", True)
        match = c[0].toggle("Match the tempo", key="mxm_match",
                            help="Stretches every part to one tempo (the pitch stays) so the beats line up "
                                 "across joins. Off: each part plays at its own speed.")
        st.session_state.setdefault("mxm_bpm", round(float(tracks[parts[0]["track"]].get("bpm") or 120.0), 1))
        c[1].number_input("Tempo (BPM)", 40.0, 220.0, step=0.5, key="mxm_bpm", disabled=not match)
        st.session_state.setdefault("mxm_xf", "1 bar")
        c[2].selectbox("Crossfade between tracks (default)", list(XF_OTHER), key="mxm_xf",
                       help="Centred on the bar line where one part hands over to the next. Each join can have "
                            "its own — 🔀 between the parts below.")
        c = st.columns(3)
        st.session_state.setdefault("mxm_fin", 0.0)
        c[0].slider("Fade in (s)", 0.0, 10.0, step=0.5, key="mxm_fin")
        st.session_state.setdefault("mxm_fout", 2.0)
        c[1].slider("Fade out (s)", 0.0, 15.0, step=0.5, key="mxm_fout")
        st.session_state.setdefault("mxm_level", True)
        c[2].toggle("Level the tracks", key="mxm_level", help="Brings every track to the same loudness.")
        st.markdown("**🎹 Keys** — shift a track's pitch (in semitones) so it sits well with the first one.")
        first_key = tracks[parts[0]["track"]].get("key")
        if len(used) > 1 and st.button("🎹 Fit the keys to the first track", key="mxm_fitkeys",
                                       help="Picks the smallest shift (up to ±3 semitones) that makes each "
                                            "track's key compatible with the first track's on the Camelot wheel."):
            for tid in used[1:]:
                st.session_state[f"mxm_shift_{tid}"] = K.best_shift(first_key, tracks[tid].get("key"))
            _rerun()
        cols = st.columns(min(4, len(used)))
        for i, tid in enumerate(used):
            key = f"mxm_shift_{tid}"
            st.session_state.setdefault(key, 0)
            cols[i % len(cols)].number_input(tid[:28], -6, 6, step=1, key=key,
                                             help=_key_text(tracks[tid].get("key"), st.session_state[key]))

    bpm = float(st.session_state.get("mxm_bpm") or 120.0) if st.session_state.get("mxm_match", True) else None
    if bpm and parts:
        # ⚡ The tempo box shows 0.1 BPM; when it's the first song's own tempo, use that exactly, so that
        # song plays unstretched (a 0.01 % stretch costs a full rubberband pass for nothing audible)
        own = float(MX.grid(tracks[parts[0]["track"]]).get("bpm") or 0)
        if own and abs(own - bpm) <= 0.05:
            bpm = own
    shift = {tid: int(st.session_state.get(f"mxm_shift_{tid}", 0)) for tid in used}
    grids = {tid: MX.grid(t) for tid, t in tracks.items()}

    # Every box's latest value into the parts first, so the play bar and lanes at the top match the boxes below
    _apply_timeline_drag(parts)
    for i, p in enumerate(parts):
        pk = f"mxm_part_{p['uid']}"
        pending = st.session_state.setdefault("_mxm_part_pending", {}).pop(p["uid"], None)
        if pending and mixer_has_stems(tracks[p["track"]]):
            st.session_state[pk] = pending          # e.g. instrumental once vocals were laid over it
        if pk in st.session_state:
            p["part"] = st.session_state[pk]
        gk = f"mxm_gain_{p['uid']}"
        if gk in st.session_state:
            p["gain_db"] = float(st.session_state[gk])
        if i:
            _join_init(parts[i - 1], p)
            _join_apply(parts[i - 1], p)
    for ly in layers:
        gk = f"mxm_lgain_{ly['uid']}"
        if gk in st.session_state:
            ly["gain_db"] = float(st.session_state[gk])

    recipe = {"id": st.session_state.get("mxm_id"), "kind": "mashup",
              "pieces": [{**{k: p[k] for k in ("track", "start", "end", "gain_db", "part", "uid")},
                          **({"xfade": p["xfade"]} if p.get("xfade") else {})} for p in parts],
              "layers": [{k: ly.get(k) for k in ("track", "part", "start", "end", "gain_db", "on", "offset_bars",
                                                 "fade_beats")}
                         for ly in layers if any(ly["on"] == p["uid"] for p in parts)],
              "bpm": bpm, "shift": {k: v for k, v in shift.items() if v},
              "xfade_same": 0.25, "xfade_other": XF_OTHER[st.session_state.get("mxm_xf", "1 bar")],
              "fade_in": float(st.session_state.get("mxm_fin", 0.0)),
              "fade_out": float(st.session_state.get("mxm_fout", 2.0)),
              "level_match": bool(st.session_state.get("mxm_level", True)),
              "settings": {"parts": parts, "layers": layers}}

    # ▶️ The play bar (stays at the top while you scroll) and the lanes — one per part, drag to trim
    import mixer
    from cut_editor import media_url
    lay = MX.layout(recipe, grids)
    if not lay["pieces"]:
        st.info("Nothing to play yet.")
        return
    _history_note(recipe)
    notes = _join_notes(lay, recipe, tracks, {})
    rough = {b.get("uid") for a, b, n in zip(lay["pieces"][:-1], lay["pieces"][1:], notes)
             if n[1] == "red" and a["track"] != b["track"]}
    _playbar(recipe, tracks, "mxm", lay, notes)
    pv_path = mixer.preview_path(recipe)
    fresh = pv_path.exists()
    url = media_url(pv_path, "audio/mpeg" if pv_path.suffix == ".mp3" else "audio/mp4", "mxm_mix_audio") \
        if fresh else ""
    with st.container(key="mxm_jlive_box"):      # always here, so the lanes below never move (they'd restart)
        live = _live_join(recipe, tracks, lay)
    _mashup_lanes(recipe, tracks, url, rough, live)
    open_uid, open_slot = _join_panel(parts, rough)
    with st.expander("📈 Whole-mix waveform — crossfade handles, layers, the preview's level", expanded=False):
        _timeline(recipe, tracks, {}, energy=_preview_side(pv_path).get("energy") if fresh else None,
                  audio_url=url, ns="mxm", editable=True)
    _nmsg = st.session_state.pop("mxm_neural_msg", None)
    if _nmsg:
        st.info(_nmsg)

    st.markdown("**Parts**")
    prev, slots = None, []
    for i, p in enumerate(parts):
        t, g = tracks[p["track"]], grids[p["track"]]
        speed, _ = MX.tempo_factor(g["bpm"], bpm)
        if prev is not None:
            slots.append(_join_row(prev, p, i, rough=p["uid"] in rough, open_uid=open_uid) or
                         (open_slot if p["uid"] == open_uid else None))
        with st.container(border=True):
            c = st.columns([3.4, 1.5, 0.9, 0.9, 1.2, 1.2, 0.9])
            c[0].markdown(f"**{i + 1}. {p['track']}** — {format_mmss(p['start'])}–{format_mmss(p['end'])} "
                          f"({format_mmss((p['end'] - p['start']) / speed)})"
                          + ("" if p.get("part", "full") == "full" else f" · {part_label(p['part']).lower()}"))
            bits = [_key_text(t.get("key"), shift.get(p["track"], 0))]
            if abs(speed - 1) > 1e-3:
                bits.append(f"tempo {(speed - 1) * 100:+.1f} %" + (" ⚠️" if MX.stretch_note(speed) else ""))
            if prev is not None and prev["track"] != p["track"]:
                sc, lab = K.compatibility(K.shifted(tracks[prev["track"]].get("key"), shift.get(prev["track"], 0)),
                                          K.shifted(t.get("key"), shift.get(p["track"], 0)))
                bits.append(("🟢" if sc >= 0.85 else "🟠" if sc >= 0.5 else "🔴") + f" after part {i}: {lab}")
            c[0].caption("  ·  ".join(bits))
            gk = f"mxm_gain_{p['uid']}"
            st.session_state.setdefault(gk, float(p.get("gain_db") or 0.0))
            c[1].number_input("Level (dB)", -12.0, 12.0, step=1.0, key=gk,
                              label_visibility="collapsed", help="Level of this part (dB)")
            c[2].button("↑", key=f"mxm_up_{p['uid']}", on_click=_move, args=(i, -1), disabled=i == 0)
            c[3].button("↓", key=f"mxm_dn_{p['uid']}", on_click=_move, args=(i, 1), disabled=i == len(parts) - 1)
            _part_trim(p, g, c[4])
            _swap_menu(p, t, tracks_all, summaries, c[5])
            c[6].button("🗑", key=f"mxm_rm_{p['uid']}", on_click=_remove, args=(i,))
            _part_stems_row(p, t, tracks_all, summaries)
        prev = p
    for k, slot in enumerate(slots):
        if slot is not None:
            _join_player(slot, recipe, k, tracks, grids)

    with st.expander(f"🤝 Good partners for {parts[0]['track']}"):
        rows = _partners(summaries[parts[0]["track"]], tracks_all)[:10] if parts[0]["track"] in summaries else []
        for r in rows:
            st.caption(f"**{r['track_id']}** — fits {r['score'] * 100:.0f} %: {r['key_label']}"
                       + (f" (shift {r['shift']:+d})" if r["shift"] else "")
                       + f", tempo {(r['speed'] - 1) * 100:+.1f} %")

    name = " + ".join(used[:3]) + (" …" if len(used) > 3 else "")
    _save_section(recipe, tracks, f"{name} (mashup)"[:90], "mxm")
    st.button("🧹 Start a new mashup", key="mxm_clear", on_click=_replace_mashup, args=("_new_mashup", (), "an empty one"))


def mixer_has_stems(t: dict) -> bool:
    import mixer
    return mixer.has_stems(t)


# ---------------------------------------------------------------------------
# 🎚️ Transition — two decks: the top suggestions for where to mix A into B, the editor, a quick preview
# ---------------------------------------------------------------------------

def _tx_state_key(a: str, b: str) -> str:
    return f"mxt_state::{a}::{b}"


def _tx_use(a: str, b: str, sug: dict, play: bool = False) -> None:
    st.session_state[_tx_state_key(a, b)] = {"a_end": sug["a_end"], "b_full": sug["b_full"], "bars": int(sug["bars"]),
                                             "profile": sug["profile"], "curves": None, "from": "suggestion"}
    st.session_state["mxt_profile"] = sug["profile"]
    st.session_state["_last_mxt_profile"] = sug["profile"]
    if play:
        st.session_state["_mxt_play"] = int(time.time() * 1000)
    st.session_state.pop("mxt_pv", None)


def _tx_recipe(a: str, b: str, ta: dict, tb: dict, s: dict, *, short: bool) -> dict:
    """short: just the transition (8 bars of A before the blend, 8 of B after) — a quick preview."""
    ga, gb = MX.grid(ta), MX.grid(tb)
    lead = (s["bars"] + (8 if short else 16)) * ga["bar_len"]
    b_end = s["b_full"] + 8 * gb["bar_len"] if short else gb["duration"]
    return {"kind": "mashup", "bpm": round(float(ga["bpm"]), 2), "level_match": True,
            "shift": {b: K.best_shift(ta.get("key"), tb.get("key"))} if a != b else {},
            "fade_in": 0.0, "fade_out": 1.0 if short else 0.0,
            "pieces": [{"track": a, "start": round(max(0.0, s["a_end"] - lead), 3), "end": round(s["a_end"], 3),
                        "gain_db": 0.0, "part": "full", "uid": "T1"},
                       {"track": b, "start": round(s["b_full"], 3), "end": round(min(b_end, gb["duration"]), 3),
                        "gain_db": 0.0, "part": "full", "uid": "T2",
                        "xfade": {"neural": {"profile": s.get("profile") or "smooth", "bars": int(s["bars"]),
                                             **({"curves": s["curves"]} if s.get("curves") else {})}}}]}


@_replacer
def _tx_to_mashup(a: str, b: str, stk: str, append: bool) -> None:
    ta, tb, s = full_track(a), full_track(b), st.session_state.get(stk)
    if ta is None or tb is None or not s:
        return
    rec = _tx_recipe(a, b, ta, tb, s, short=False)
    t0 = f"{time.time():.6f}"
    pa, pb = rec["pieces"]
    pa["uid"], pb["uid"] = f"S{t0}", f"S{t0}b"
    parts = list(st.session_state.get("mxm_parts") or []) if append else []
    if append and parts and parts[-1]["track"] == a:
        parts[-1] = {**parts[-1], "end": pa["end"]}      # the mashup's last part now hands over here
        parts.append(pb)
    else:
        if not append:
            _clear("mxm")
            st.session_state["mxm_layers"] = []
        parts += [pa, pb]
    st.session_state["mxm_parts"] = parts
    for p in (pa, pb):
        st.session_state[f"mxm_part_{p['uid']}"] = p["part"]
    st.session_state.setdefault("mxm_bpm", rec["bpm"])
    st.session_state["mxm_match"] = True
    st.session_state[f"mxm_shift_{b}"] = rec["shift"].get(b, 0)
    st.session_state["mx_mode"] = MASHUP


def _tx_apply_join(a_uid: str, b_uid: str, stk: str) -> None:
    """✅ Back into the 🧩 Mashup join this was opened from."""
    s = st.session_state.get(stk) or {}
    if not s:
        return
    parts = st.session_state.get("mxm_parts") or []
    pa = next((p for p in parts if p["uid"] == a_uid), None)
    pb = next((p for p in parts if p["uid"] == b_uid), None)
    if pa and pb:
        pa["end"], pb["start"] = round(s["a_end"], 3), round(s["b_full"], 3)
        pb["xfade"] = {"neural": {"profile": s.get("profile") or "smooth", "bars": int(s["bars"]),
                                  **({"curves": s["curves"]} if s.get("curves") else {})}}
        st.session_state[f"mxm_xfl_{b_uid}"] = NEURAL
        st.session_state[f"mxm_nnp_{b_uid}"] = pb["xfade"]["neural"]["profile"]
        st.session_state[f"mxm_nnb_{b_uid}"] = pb["xfade"]["neural"]["bars"]
    st.session_state.pop("mxt_join", None)
    st.session_state["mx_mode"] = MASHUP


def _open_join_in_transition(a: dict, b: dict) -> None:
    """🎚️ Edit a 🧩 Mashup join in the two-deck editor."""
    nn = (b.get("xfade") or {}).get("neural") or {}
    st.session_state["mxt_a"], st.session_state["mxt_b"] = a["track"], b["track"]
    st.session_state[_tx_state_key(a["track"], b["track"])] = {
        "a_end": float(a["end"]), "b_full": float(b["start"]), "bars": int(nn.get("bars") or 8),
        "profile": nn.get("profile") or "smooth", "curves": nn.get("curves"), "from": "mashup"}
    st.session_state["mxt_profile"] = nn.get("profile") or "smooth"
    st.session_state["mxt_join"] = (a["uid"], b["uid"])
    st.session_state.pop("mxt_pv", None)
    st.session_state["mx_mode"] = TRANSITION


def _live_job(job, a: str, b: str, ta: dict, tb: dict, s: dict, have):
    import live_preview as LP
    job.progress(0.15, "tempo-matching the stems")
    return LP.prepare(a, b, ta, tb, s, have=have)


def _tx_live(a: str, b: str, ta: dict, tb: dict, s: dict):
    """⚡ The live preview's stems for this transition, as URLs — prepared in the background (the decks work
    straight away; ▶ Live mix lights up when they're ready). None until then, or when the audio isn't on the
    server yet."""
    import live_preview as LP
    import mixer
    import mix_jobs as J
    from cut_editor import media_url
    have = st.session_state.get("mxt_live")
    if not LP.still_fits(have, a, b, ta, tb, s):
        j = J.get(st.session_state.get("_mxt_livejob") or "")
        if j is not None and j.state == "done" and LP.still_fits(j.result, a, b, ta, tb, s):
            have = st.session_state["mxt_live"] = j.result
        else:
            need = LP.needed_files(ta, tb)
            if any(not _has_file(mixer.local_path(t, part)) for t in (ta, tb) for part in need[t.get("track_id")]):
                return None
            if j is None or not j.active:
                key = f"live:{a}|{b}|{float(s['a_end']):.2f}|{float(s['b_full']):.2f}|{int(s['bars'])}"
                fits_old = have if have and have.get("a_id") == a and have.get("b_id") == b else None
                j = J.start(key, "⚡ Getting the live preview ready", _live_job, a, b, ta, tb, dict(s), fits_old,
                            retry_failed=False)
                st.session_state["_mxt_livejob"] = key
            if j.state == "error":
                st.caption(f":orange[⚡ Live preview unavailable: {j.error}]")
                return None
            if j.active:
                _job_watch((j.key,))
                return None
            return None
    out = {"mode": have["mode"], "speed_b": have["speed_b"]}
    for d in ("a", "b"):
        out[d] = {"w": have[d]["w"], "gain_db": have[d]["gain_db"],
                  "urls": {p: media_url(Path(f), "audio/flac", f"mxt_live_{d}_{p}") for p, f in have[d]["files"].items()}}
    return out


def _transition(summaries: dict) -> None:
    import mixer
    from cut_editor import media_url
    from core import transitions as TR
    import deck_editor as DE
    tracks = _mixable(summaries)
    if len(tracks) < 2:
        st.info("Needs at least two analysed tracks.")
        return
    ids = sorted(tracks, key=str.lower)
    join = st.session_state.get("mxt_join")
    c = st.columns(2)
    if st.session_state.get("mxt_a") not in ids:
        st.session_state.pop("mxt_a", None)
    a = c[0].selectbox("Track A (playing)", ids, key="mxt_a", format_func=lambda x: _label(tracks[x]),
                       disabled=bool(join), index=None, placeholder="Pick the song that's playing…")
    if a is None:
        c[1].selectbox("Track B (coming in)", [], disabled=True, placeholder="…then the one coming in",
                       key="mxt_b_wait")
        st.caption("Pick track A — the best-fitting track B is picked for you (change it if you like), with the "
                   "top 3 places to mix them. Or 🎚️ fine-tune a join from 🧩 Mashup.")
        return
    ranked = [r["track_id"] for r in _partners(tracks[a], tracks)]
    b_ids = [x for x in ranked if x != a] + [x for x in ids if x not in ranked and x != a]
    if st.session_state.get("mxt_b") not in b_ids:
        st.session_state["mxt_b"] = b_ids[0]
    fit = {r["track_id"]: r for r in _partners(tracks[a], tracks)}
    b = c[1].selectbox("Track B (coming in) — best fit first", b_ids, key="mxt_b", disabled=bool(join),
                       format_func=lambda x: _label(tracks[x]) + (f"  —  fits {fit[x]['score'] * 100:.0f} %"
                                                                  if x in fit else ""))
    ta, tb = full_track(a), full_track(b)
    if ta is None or tb is None:
        st.warning("Couldn't read one of the tracks.")
        return
    _prefetch([a, b])                       # ⬇️ songs + stems start coming from Drive straight away
    ga, gb = MX.grid(ta), MX.grid(tb)
    if join:
        st.info("🎚️ Editing a join from your 🧩 Mashup — **✅ Apply to the mashup** when it sounds right.")

    # 🎯 Suggestions — computed once per pair
    sk = f"mxt_sug::{a}::{b}"
    if sk not in st.session_state:
        st.session_state[sk] = TR.suggest(ta, tb)
    sugs = st.session_state[sk]
    stk = _tx_state_key(a, b)
    if stk not in st.session_state and sugs:
        _tx_use(a, b, sugs[0])
    if stk not in st.session_state:
        g_end = MX.snap_to_bar(ga, ga["duration"] * 0.75)
        st.session_state[stk] = {"a_end": g_end, "b_full": float(gb["bars"][min(8, len(gb["bars"]) - 1)]),
                                 "bars": 8, "profile": "smooth", "from": "default"}
    s = st.session_state[stk]
    if sugs:
        st.markdown("**🎯 Top 3** — ▶ loads one and plays it live; fine-tune it on the decks below")
        cols = st.columns(len(sugs))
        for i, (col, sg) in enumerate(zip(cols, sugs)):
            with col.container(border=True, key=f"mxt_sug_{i}"):
                on = abs(sg["a_end"] - s["a_end"]) < 0.05 and abs(sg["b_full"] - s["b_full"]) < 0.05
                badge = "green" if sg["score"] >= 0.8 else "orange" if sg["score"] >= 0.6 else "gray"
                st.markdown(f":{badge}-badge[{sg['score'] * 100:.0f} %] **{_profile_label(sg['profile'])}** · "
                            f"{sg['bars']} bars")
                st.caption(f"A out **{format_mmss(sg['a_end'])}** · B in **{format_mmss(sg['b_full'])}**",
                           help=", ".join(sg["why"]) or None)
                st.button("▶ Again" if on else "▶ Hear it", key=f"mxt_use_{i}", width="stretch",
                          type="primary" if on else "secondary", on_click=_tx_use, args=(a, b, sg, True),
                          help="Loads this transition onto the decks and plays it live from 4 bars before the blend.")

    # The two decks
    speed_b, _ = MX.tempo_factor(gb["bpm"], ga["bpm"])
    urls = {}
    for tid, t in ((a, ta), (b, tb)):
        path = mixer.local_path(t, "full")
        urls[tid] = media_url(path, "audio/mp4" if path.suffix == ".m4a" else "audio/mpeg", f"mxt_audio_{tid}") \
            if _has_file(path) else ""
    pv = st.session_state.get("mxt_pv")
    rec_short = _tx_recipe(a, b, ta, tb, s, short=True)
    s.setdefault("profile", st.session_state.get("mxt_profile") or "smooth")
    with st.container(key="mxt_live_box"):        # always here, so the decks below never move (they'd restart)
        live = _tx_live(a, b, ta, tb, s)
    val = DE.deck_editor(a=DE.lane_data(ta, a, ga, urls[a]), b=DE.lane_data(tb, b, gb, urls[b]), state=s,
                         bar_a=ga["bar_len"], bar_b=gb["bar_len"], speed_b=speed_b, live=live,
                         beats_per_bar=int(ga["beats_per_bar"]), state_key=f"{a}|{b}|{s.get('from')}", key="mxt_deck",
                         play=play if (play := int(st.session_state.get("_mxt_play") or 0))
                         and time.time() * 1000 - play < 20_000 else 0)
    if isinstance(val, dict) and val.get("nonce") and val["nonce"] != st.session_state.get("_mxt_nonce"):
        st.session_state["_mxt_nonce"] = val["nonce"]
        s.update(a_end=float(val["a_end"]), b_full=float(val["b_full"]), bars=int(val["bars"]),
                 curves=SM.norm_curves(val.get("curves")) if val.get("curves") else s.get("curves"), **{"from": "drag"})
        _rerun()
    missing = [x for x in (a, b) if urls[x] == ""]
    if missing:
        st.caption(f"▶ A / ▶ B play once the song is on the server: {', '.join(missing)} — fetching it from "
                   "Google Drive (below).")

    # Blend style and length — one tap each
    st.session_state.setdefault("mxt_profile", s.get("profile") or "smooth")
    prof = _pick(st, "Blend style", list(SM.PROFILES), "mxt_profile", _profile_label,
                 help="  \n".join(SM.PROFILES.values()))
    s["profile"] = prof
    st.session_state["mxt_blend"] = int(s["bars"]) if int(s["bars"]) in NEURAL_BARS else 8
    st.session_state["_last_mxt_blend"] = st.session_state["mxt_blend"]
    _pick(st, "Blend length", list(NEURAL_BARS), "mxt_blend", lambda n: f"{n} bar{'s' if n != 1 else ''}",
          on_change=_tx_set_bars, args=(stk,), help="How long the two songs blend. Or drag the blend's ▌ edge above.")
    # Curve shape presets (the ● handles above shape them further)
    cur = SM.preset_of(s.get("curves")) or "hand"
    opts = list(SM.CURVE_PRESETS) + (["hand"] if cur == "hand" else [])
    st.session_state["mxt_curve"] = cur
    st.pills("Curve shape", opts, key="mxt_curve", format_func=_preset_label, help=CURVE_HELP,
             on_change=_tx_set_preset, args=(stk, "mxt_curve"))

    no_stems = [x for x, t in ((a, ta), (b, tb)) if not mixer.has_four_stems(t)]
    if no_stems:
        st.caption(f":orange[🧠 {', '.join(no_stems)} not split into drums / bass / other yet — the preview is an "
                   "ordinary crossfade until run_audio_analysis.bat has made the stems.]")
    tracks_full = {a: ta, b: tb}
    ready = _audio_ready(mixer.parts_needed(rec_short, tracks_full), "mxt")
    out = mixer.preview_path(rec_short)
    job = _render_job_for(out)
    busy = bool(job and job.active)
    if ready and not out.exists() and st.button("🎧 Render it exactly", key="mxt_preview", disabled=busy,
                           help="▶ Live mix above is the quick way to listen. This renders the transition (8 bars "
                                "either side) exactly as a saved mix will sound — studio-quality tempo matching. "
                                "It carries on in the background."):
        job = _start_render(rec_short, tracks_full, out, "🎧 Mixing the transition")
        st.session_state["mxt_pv"] = {"path": str(out), "sig": json.dumps(rec_short, sort_keys=True)}
        busy = job.active
    if busy:
        _job_watch((job.key,))
    elif job is not None and job.state == "error":
        st.error(f"Couldn't make the preview: {job.error}")
    if out.exists():
        st.audio(str(out))
    elif pv and not busy:
        st.caption("The exact render is out of date — render it again to hear the changes.")

    # Where it goes next
    st.divider()
    bc = st.columns(3)
    if join:
        bc[0].button("✅ Apply to the mashup", key="mxt_apply", type="primary", width="stretch",
                     on_click=_tx_apply_join, args=(join[0], join[1], stk))
        bc[1].button("✖ Cancel", key="mxt_cancel", width="stretch",
                     on_click=lambda: (st.session_state.pop("mxt_join", None),
                                       st.session_state.__setitem__("mx_mode", MASHUP)))
    else:
        bc[0].button("🧩 Start a mashup with it", key="mxt_new", type="primary", width="stretch",
                     on_click=_replace_mashup, args=("_tx_to_mashup", (a, b, stk, False), f"{a} → {b}"),
                     help="A from 16 bars before the blend, then B to its end — carry on adding parts in 🧩 Mashup.")
        parts = st.session_state.get("mxm_parts") or []
        can_append = bool(parts) and parts[-1]["track"] == a
        bc[1].button("➕ Add to my mashup", key="mxt_append", width="stretch", disabled=not can_append,
                     on_click=_tx_to_mashup, args=(a, b, stk, True),
                     help="When your 🧩 Mashup ends with track A: it now hands over to B here.")


# ---------------------------------------------------------------------------
# 💡 Suggestions — tracks that should mash up well
# ---------------------------------------------------------------------------

def _infos(summaries: dict) -> list:
    out = []
    for tid, t in _mixable(summaries).items():
        e = t.get("energy") or []
        out.append({"track_id": tid, "bpm": t.get("bpm"), "key": t.get("key"),
                    "energy": float(np.mean(e)) if e else 0.5, "stems": bool(t.get("stems")),
                    "builds": int(t.get("builds") or 0), "drops": len(t.get("drops") or [])})
    return out


@_replacer
def _start_suggestion(a: str, b: str, vocal_swap: bool) -> None:
    """Fill the 🧩 Mashup editor with a suggested pair and switch to it: a's loudest section
    then b's (or, for a vocal swap, a's loudest section as instrumental with b's most-sung
    section's vocals over it), at a's tempo, b's key fitted to a's."""
    ta, tb = full_track(a), full_track(b)
    if ta is None or tb is None:
        return
    _clear("mxm")
    ga, gb = MX.grid(ta), MX.grid(tb)
    sa = MX.best_section(ta, "energy")
    uid_a = f"S{time.time():.6f}"
    part_a = {"track": a, "start": MX.snap_to_bar(ga, sa["start"]), "end": MX.snap_to_bar(ga, sa["end"]),
              "gain_db": 0.0, "uid": uid_a, "part": "instrumental" if vocal_swap else "full"}
    parts, layers = [part_a], []
    if vocal_swap:
        sb = MX.best_section(tb, "vocals")
        speed_b, _ = MX.tempo_factor(gb["bpm"], ga["bpm"])
        speed_a = 1.0
        room = (part_a["end"] - part_a["start"]) / speed_a * speed_b      # b-seconds that fit over a's part
        start_b = MX.snap_to_bar(gb, sb["start"])
        end_b = min(MX.snap_to_bar(gb, sb["end"]), start_b + room)
        layers.append({"uid": f"L{time.time():.6f}", "on": uid_a, "track": b, "part": "vocals",
                       "start": start_b, "end": end_b, "offset_bars": 0, "gain_db": 0.0})
    else:
        sb = MX.best_section(tb, "energy")
        parts.append({"track": b, "start": MX.snap_to_bar(gb, sb["start"]), "end": MX.snap_to_bar(gb, sb["end"]),
                      "gain_db": 0.0, "uid": f"S{time.time():.6f}b", "part": "full"})
    st.session_state["mxm_parts"] = parts
    st.session_state["mxm_layers"] = layers
    for p in parts:
        st.session_state[f"mxm_part_{p['uid']}"] = p["part"]
    st.session_state["mxm_bpm"] = round(float(ta.get("bpm") or 120.0), 1)
    st.session_state["mxm_match"] = True
    st.session_state[f"mxm_shift_{b}"] = K.best_shift(ta.get("key"), tb.get("key"))
    st.session_state["mx_mode"] = MASHUP


@_replacer
def _start_drop_swap(a: str, b: str) -> None:
    """Fill the 🧩 Mashup editor with "a's build-up → b's drop" (core/mixing.drop_swap_plan) and switch
    to it: a's tempo, b's key fitted to a's, a 1-beat swap on the drop's downbeat with 🎛️ bass swap."""
    ta, tb = full_track(a), full_track(b)
    if ta is None or tb is None:
        return
    plan = MX.drop_swap_plan(ta, tb)
    if not plan:
        st.session_state["mx_drop_msg"] = f"Couldn't find a build-up in {a} and a drop in {b} to join."
        return
    _clear("mxm")
    t0 = f"{time.time():.6f}"
    import mixer
    xf = {"beats": float(MX.DROP_XFADE_BEATS), "where": "centre", "shape": "smooth", "bass_swap": True}
    if mixer.has_four_stems(ta) and mixer.has_four_stems(tb):
        # 🧠 B's own build-up blends in under A's from the stems, then drums, bass and harmonics swap on the drop
        bars = max([n for n in NEURAL_BARS if n <= max(4.0, float(plan["build_bars"]))] or [4])
        xf = {"neural": {"profile": "bass_drop", "bars": bars}}
    parts = [{"track": a, "start": plan["a"][0], "end": plan["a"][1], "gain_db": 0.0, "uid": f"S{t0}",
              "part": "full", "label": "build-up"},
             {"track": b, "start": plan["b"][0], "end": plan["b"][1], "gain_db": 0.0, "uid": f"S{t0}b",
              "part": "full", "label": "drop", "xfade": xf}]
    st.session_state["mxm_parts"] = parts
    st.session_state["mxm_layers"] = []
    for p in parts:
        st.session_state[f"mxm_part_{p['uid']}"] = p["part"]
    st.session_state["mxm_bpm"] = round(float(ta.get("bpm") or 120.0), 1)
    st.session_state["mxm_match"] = True
    st.session_state[f"mxm_shift_{b}"] = K.best_shift(ta.get("key"), tb.get("key"))
    st.session_state["mx_mode"] = MASHUP


def _drop_swap_row(r: dict, i: int) -> None:
    with st.container(border=True, key=f"mx_sug_d_{i}"):
        c = st.columns([5, 1.6], vertical_alignment="center")
        badge = "green" if r["score"] >= 0.75 else "orange" if r["score"] >= 0.55 else "gray"
        c[0].markdown(f":{badge}-badge[fits {r['score'] * 100:.0f} %] 💥 **{r['a']}** build-up → **{r['b']}** drop")
        bits = [r["key_label"] + (f" (shift {r['shift']:+d} st)" if r["shift"] else ""),
                f"tempo {(r['speed'] - 1) * 100:+.1f} %"]
        c[0].caption(" · ".join(bits))
        c[1].button("🧩 Start it", key=f"mx_sug_go_d_{i}", width="stretch", on_click=_replace_mashup,
                    args=("_start_drop_swap", (r["a"], r["b"]), f"{r['a']} → {r['b']}"))


def _suggestions(summaries: dict) -> None:
    infos = _infos(summaries)
    if len(infos) < 2:
        st.info("Needs at least two analysed tracks.")
        return
    n_keys = sum(1 for i in infos if i["key"])
    st.caption("Tracks that should mash up well — keys that sit together on the Camelot wheel (with a small pitch "
               "shift if that helps), tempos within a few percent, similar energy. Tap one to start it in 🧩 Mashup.",
               help=None if n_keys == len(infos) else f"{len(infos) - n_keys} track(s) have no key yet — "
                                                         "🎵 Audio → 🔬 Audio analysis → 🎹 Find the key.")
    tabs = st.tabs(["🤝 Pairs", "🎤 Vocal swaps", "💥 Drop swaps"])
    with tabs[0]:
        for i, r in enumerate(MX.suggest_pairs(infos, top=12)):
            _suggestion_row(r, i, vocal_swap=False)
    with tabs[1]:
        swaps = MX.suggest_pairs(infos, top=12, need_stems=True)
        if not swaps:
            n = sum(1 for x in infos if x["stems"])
            st.info("Vocal swaps need tracks split into vocals + instrumental — run **run_audio_analysis.bat** on the "
                    f"laptop (it makes stems on the GPU along with the analysis and saves them to Google Drive). {n} track(s) have stems.")
        for i, r in enumerate(swaps):
            _suggestion_row(r, i, vocal_swap=True)
    with tabs[2]:
        msg = st.session_state.pop("mx_drop_msg", None)
        if msg:
            st.warning(msg)
        drops = MX.suggest_drop_swaps(infos, top=12)
        if not drops:
            st.info("Drop swaps need tracks with the extra analysis (build-ups and drops) — 🎵 Audio → 🔬 Audio "
                    "analysis.")
        else:
            st.caption(f"One track's build-up crashing into another's drop: {MX.DROP_LEAD_BARS} bars before the "
                       f"build-up, then {MX.DROP_AFTER_BARS} bars from the drop — beat-matched and key-fitted. With "
                       "both tracks split into stems it's a 🧠 Bass drop blend (the drop's own build-up comes in "
                       "under A's, then drums, bass and harmonics swap on the drop); otherwise a quick crossfade "
                       "with 🎛️ bass swap.")
        for i, r in enumerate(drops):
            _drop_swap_row(r, i)


def _suggestion_row(r: dict, i: int, vocal_swap: bool) -> None:
    with st.container(border=True, key=f"mx_sug_{'v' if vocal_swap else 'p'}_{i}"):
        c = st.columns([5, 1.6], vertical_alignment="center")
        title = (f"🎤 **{r['track_id']}** vocals over **{r['a']}**" if vocal_swap
                 else f"**{r['a']}**  +  **{r['b']}**")
        badge = "green" if r["score"] >= 0.75 else "orange" if r["score"] >= 0.55 else "gray"
        c[0].markdown(f":{badge}-badge[fits {r['score'] * 100:.0f} %] {title}")
        bits = [r["key_label"] + (f" (shift {r['shift']:+d} st)" if r["shift"] else ""),
                f"tempo {(r['speed'] - 1) * 100:+.1f} %"]
        c[0].caption(" · ".join(bits))
        c[1].button("🧩 Start it", key=f"mx_sug_go_{'v' if vocal_swap else 'p'}_{i}", width="stretch",
                    on_click=_replace_mashup, args=("_start_suggestion", (r["a"], r["b"], vocal_swap),
                                                    f"{r['a']} + {r['b']}"))


# ---------------------------------------------------------------------------
# 📂 Saved mixes
# ---------------------------------------------------------------------------

def _clear(ns: str) -> None:
    for k in [k for k in st.session_state if k.startswith(ns + "_") or k.startswith("_" + ns + "_")]:
        del st.session_state[k]


@_replacer
def _open(rec: dict) -> None:
    """Load a saved recipe into its editor."""
    s = rec.get("settings") or {}
    if rec.get("kind") == "extend":
        _clear("mxe")
        st.session_state["mxe_id"] = rec["id"]
        st.session_state["mxe_name"] = rec.get("name", "")
        st.session_state["mxe_track"] = s.get("track") or rec["pieces"][0]["track"]
        xf = {v: k for k, v in XF_SAME.items()}.get(rec.get("xfade_same"), "Short (¼ beat)")
        st.session_state["mxe_xf"] = xf
        st.session_state["mxe_fade"] = float(rec.get("fade_out") or 0.0)
        if s.get("method") == "sections":
            st.session_state["mxe_method"] = "✋ By sections"
            for k, n in enumerate(s.get("counts") or []):
                st.session_state[f"mxe_cnt_{st.session_state['mxe_track']}_{k}"] = int(n)
        else:
            st.session_state["mxe_method"] = "🎯 Auto — to a length"
            st.session_state["mxe_target"] = int(round(float(s.get("target") or 0))) or None
            st.session_state["_mxe_target_for"] = st.session_state["mxe_track"]
            st.session_state["mxe_exact"] = bool(s.get("exact"))
            st.session_state["_mxe_want_jumps"] = s.get("jumps")
        st.session_state["mx_mode"] = EXTEND
    else:
        _clear("mxm")
        st.session_state["mxm_id"] = rec["id"]
        st.session_state["mxm_name"] = rec.get("name", "")
        parts = [{"part": "full", **p, "uid": p.get("uid") or f"{i}-{time.time():.6f}"}
                 for i, p in enumerate(rec.get("pieces") or [])]
        st.session_state["mxm_parts"] = parts
        for p in parts:
            st.session_state[f"mxm_part_{p['uid']}"] = p.get("part") or "full"
        st.session_state["mxm_layers"] = [{"part": "vocals", "gain_db": 0.0, **ly, "uid": f"L{i}-{time.time():.6f}"}
                                          for i, ly in enumerate(rec.get("layers") or [])]
        st.session_state["mxm_match"] = rec.get("bpm") is not None
        if rec.get("bpm"):
            st.session_state["mxm_bpm"] = float(rec["bpm"])
        st.session_state["mxm_xf"] = {v: k for k, v in XF_OTHER.items()}.get(rec.get("xfade_other"), "1 bar")
        st.session_state["mxm_fin"] = float(rec.get("fade_in") or 0.0)
        st.session_state["mxm_fout"] = float(rec.get("fade_out") or 0.0)
        st.session_state["mxm_level"] = bool(rec.get("level_match", True))
        for tid, v in (rec.get("shift") or {}).items():
            st.session_state[f"mxm_shift_{tid}"] = int(v)
        st.session_state["mx_mode"] = MASHUP


def _saved(summaries: dict) -> None:
    import mixes as MS
    mixes = sorted(MS.all_mixes().values(), key=lambda m: m.get("updated", ""), reverse=True)
    if not mixes:
        st.info("No saved mixes yet — make one in 📏 Extend / shorten or 🧩 Mashup.")
        return
    from page_setup import drive_token
    for m in mixes:
        out = m.get("output") or {}
        with st.container(border=True):
            c = st.columns([6, 1, 1, 1])
            kind = "📏 Extended" if m.get("kind") == "extend" else "🧩 Mashup"
            src = sorted({p["track"] for p in m.get("pieces") or []})
            c[0].markdown(f"**{m.get('name', m['id'])}** — {kind}")
            bits = ["from " + " + ".join(src)]
            if out:
                in_lib = out.get("track_id") in summaries
                bits.append(f"⏱ {format_mmss(out.get('duration', 0))}")
                bits.append("✅ in the library" if in_lib else "⏳ being analysed")
                bits.append("☁️ on Drive" if out.get("uploaded") else "⚠️ not on Google Drive yet")
            else:
                bits.append("recipe only — not saved as a track yet")
            c[0].caption("  ·  ".join(bits))
            missing = [t for t in src if t not in summaries]
            c[1].button("✏️", key=f"mxs_open_{m['id']}", disabled=bool(missing),
                        on_click=(_open if m.get("kind") == "extend" else _replace_mashup),
                        args=((m,) if m.get("kind") == "extend" else ("_open", (m,), f"the saved mix {m.get('name', '')}")),
                        help="Open in the editor to change it, preview and save again."
                        if not missing else f"Source track(s) gone: {', '.join(missing)}")
            if out and not out.get("uploaded"):
                if c[2].button("☁️", key=f"mxs_up_{m['id']}", help="Upload the track to Google Drive",
                               disabled=drive_token() is None):
                    err = MS.upload_output(m["id"], drive_token())
                    st.toast(err or "Uploaded to Google Drive ✅")
                    _rerun()
            if st.session_state.get("_mxs_del") == m["id"]:
                d = st.columns([4, 1.3, 1.3], vertical_alignment="center")
                d[0].caption(f"Delete the recipe **{m.get('name', m['id'])}**? A track already made from it stays "
                             "in the library.")
                if d[1].button("🗑 Delete", key=f"mxs_del_yes_{m['id']}", type="primary", width="stretch"):
                    st.session_state.pop("_mxs_del", None)
                    MS.delete(m["id"])
                    _rerun()
                d[2].button("Keep it", key=f"mxs_del_no_{m['id']}", width="stretch",
                            on_click=lambda: st.session_state.pop("_mxs_del", None))
            elif c[3].button("🗑", key=f"mxs_del_{m['id']}",
                             help="Forget this recipe (asks first). A track already made from it stays in the library."):
                st.session_state["_mxs_del"] = m["id"]
                _rerun()


# ---------------------------------------------------------------------------
# Page section
# ---------------------------------------------------------------------------

def render(summaries: dict) -> None:
    """summaries: the Media Library's load_audio_library() rows."""
    st.session_state.setdefault("mx_mode", MODES[0])
    with st.container(key="subnav_mix"):   # a sub menu — chips (page_setup.menu_css)
        mode = st.radio("Mix", MODES, horizontal=True, key="mx_mode", label_visibility="collapsed")
    _pending_check()
    if mode == MASHUP:
        _draft_banner()
    _mode_view(mode, summaries)


@st.fragment
def _mode_view(mode: str, summaries: dict) -> None:
    """⚡ The Mix screen is a fragment: a tap in it redraws just this screen, not the whole page. Anything that
    needs the page around it (another mode, the "Replace your mashup?" question) redraws the page."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
    if ctx is not None and getattr(ctx, "fragment_ids_this_run", None):
        if st.session_state.get("mx_mode") != mode or st.session_state.get("_mx_pending"):
            st.rerun(scope="app")
    if mode == EXTEND:
        _extend(summaries)
    elif mode == MASHUP:
        _mashup(summaries)
    elif mode == TRANSITION:
        _transition(summaries)
    elif mode == SUGGESTIONS:
        _suggestions(summaries)
    else:
        _saved(summaries)
