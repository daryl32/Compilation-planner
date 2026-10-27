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

@st.cache_data(max_entries=12, show_spinner=False)

def _has_file(path) -> bool:
    """Whether a song file is here — False (not a crash) when its folder can't be read."""
    try:
        return Path(path).exists()
    except OSError:
        return False

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


def _audio_ready(needs, ns: str) -> bool:
    """True when every file the mix plays is on the server; otherwise offers to fetch them
    from Google Drive. needs: track ids (the full songs) or {track_id: {"full", "vocals", …}}."""
    import mixer
    needs = needs if isinstance(needs, dict) else {tid: {"full"} for tid in needs}
    missing = [(tid, part) for tid, parts in needs.items() for part in sorted(parts)
               if (t := full_track(tid)) and not _has_file(mixer.local_path(t, part))]
    if not missing:
        return True
    st.info(f"🎵 {len(missing)} file(s) need fetching from Google Drive first: "
            + ", ".join(tid + ("" if part == "full" else f" ({part})") for tid, part in missing))
    if st.button(f"⬇️ Fetch the audio ({len(missing)})", key=f"{ns}_fetch", type="primary"):
        bar = st.progress(0.0, text="Downloading from Google Drive…")
        for i, (tid, part) in enumerate(missing):
            err = mixer.ensure_local(full_track(tid), parts=(part,),
                                     progress=lambda p, i=i: bar.progress(min(1.0, (i + min(p, 1.0)) / len(missing))))
            if err:
                st.error(err)
                return False
        st.rerun()
    return False


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
               + ("" if (p.get("part") or "full") == "full" else f" · {PART_LABELS[p['part']].lower()}")}
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

def _mashup_lanes(recipe: dict, tracks: dict, mix_url: str) -> None:
    """🎚️ One lane per part (mix_lanes.py) — trims come back here."""
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
        lanes.append({"uid": p.get("uid"), "idx": k, "label": p["track"], "colour": PALETTE[k % len(PALETTE)].format(a=0.8),
                      "trk": data[p["track"]], "start": p["start"], "end": p["end"], "speed": p["speed"],
                      "out_start": p["out_start"], "out_end": p["out_end"],
                      "head": joins[k - 1]["pre"] if k > 0 else 0.0, "tail": joins[k]["post"] if k < len(joins) else 0.0})
    val = ML.mix_lanes(duration=lay["duration"], lanes=lanes,
                       joins=[{"at": j["at"], "pre": j["pre"], "post": j["post"]} for j in joins],
                       mix_url=mix_url, key="mxm_lanes")
    if isinstance(val, dict) and val.get("nonce") and val["nonce"] != st.session_state.get("_mxm_lanes_nonce"):
        st.session_state["_mxm_lanes_nonce"] = val["nonce"]
        for p in st.session_state.get("mxm_parts") or []:
            if p.get("uid") == val.get("uid"):
                p["start"], p["end"] = round(float(val["start"]), 3), round(float(val["end"]), 3)
        st.rerun()


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
    with col.popover("✂️", help="Trim this part's start and end"):
        st.select_slider(f"{p['track']} — start and end ({snap.lower()})", options=lines,
                         value=(round(float(p["start"]), 3), round(float(p["end"]), 3)), key=key,
                         format_func=format_mmss, on_change=_trim_part, args=(p["uid"], key))
        st.caption("Snap to bars or beats with **Snap to** in ➕ Add a part. Or drag the part's handles in the lanes above.")


def _preview_and_save(recipe: dict, tracks: dict, sims: dict, default_name: str, ns: str) -> None:
    import mixer
    import mixes as MS
    from cut_editor import media_url
    pv_path = mixer.preview_path(recipe)
    pv = st.session_state.get(f"{ns}_pv") or {}
    fresh = pv.get("path") == str(pv_path) and pv_path.exists()
    url = media_url(pv_path, "audio/mpeg" if pv_path.suffix == ".mp3" else "audio/mp4", f"{ns}_mix_audio") \
        if fresh else ""
    if ns == "mxm":
        _mashup_lanes(recipe, tracks, url)
        with st.expander("📈 Whole-mix waveform — crossfade handles, layers, the preview's level", expanded=False):
            lay = _timeline(recipe, tracks, sims, energy=pv.get("energy") if fresh else None, audio_url=url, ns=ns,
                            editable=True)
    else:
        lay = _timeline(recipe, tracks, sims, energy=pv.get("energy") if fresh else None, audio_url=url, ns=ns,
                        editable=False)
    if lay is None:
        st.info("Nothing to play yet.")
        return
    length = min(lay["duration"], float(recipe.get("trim_to") or lay["duration"]))
    notes = _join_notes(lay, recipe, tracks, sims)
    status = [f"⏱ **{format_mmss(length)}**", f"{len(lay['pieces'])} piece(s)",
              f"{lay['bpm']:.1f} BPM"]
    weak = [n for n in notes if n[1] == "red"]
    if weak:
        status.append(f"⚠️ {len(weak)} rough join(s)")
    st.caption("  ·  ".join(status))

    if not _audio_ready(mixer.parts_needed(recipe, tracks), ns):
        return
    c = st.columns([1, 3])
    if c[0].button("▶️ Preview", key=f"{ns}_preview", type="primary", width="stretch",
                   help="Renders the mix to a quick temporary file to listen to (nothing is saved)."):
        if not pv_path.exists() or pv.get("path") != str(pv_path):
            with st.spinner("Mixing…"):
                try:
                    info = mixer.render(recipe, tracks, pv_path)
                except Exception as e:
                    st.error(f"Couldn't make the preview: {e}")
                    return
            st.session_state[f"{ns}_pv"] = {"path": str(pv_path), "energy": info["energy"],
                                            "warnings": info["warnings"]}
        st.session_state[f"{ns}_start"] = 0
        st.rerun()
    if fresh:
        c[1].caption("▶ Play it on the timeline above — ◀ Join ▶ steps through the joins.")
        for w in pv.get("warnings") or []:
            st.caption(f"⚠️ {w}")
        with st.expander("🔈 Plain player"):
            st.audio(str(pv_path))
    elif pv.get("path"):
        c[1].caption("The mix has changed since the last preview — ▶️ Preview again to hear it.")

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
    c = st.columns(2)
    label = f"💾 Save over {clean}" if replacing else "💾 Save as a new track"
    if c[0].button(label, key=f"{ns}_save", type="primary", disabled=not clean or taken, width="stretch",
                   help="Mixes it properly (AAC), adds it to the Audio-Library (and Google Drive when connected) "
                        "and analyses it like any other track — ready for the planner in a minute or two."
                        + (" Replaces the track this mix made before (plans using it will pick up the new "
                           "audio)." if replacing else "")):
        from page_setup import drive_token
        with st.status("Saving the mix…", expanded=True) as box:
            try:
                rec, msgs = MS.publish({**recipe, "name": clean}, tracks, token=drive_token(),
                                       progress=lambda m: box.update(label=f"Saving the mix — {m}…"))
            except Exception as e:
                box.update(label="Couldn't save the mix", state="error")
                st.error(str(e))
                return
            box.update(label=f"✅ Saved **{clean}** — analysing it now (🎵 Audio shows the progress).",
                       state="complete")
        for m in msgs:
            st.caption(f"ℹ️ {m}")
        st.session_state[f"{ns}_id"] = rec["id"]
    if c[1].button("📝 Save the recipe only", key=f"{ns}_recipe", disabled=not clean, width="stretch",
                   help="Keeps the mix in 📂 Saved mixes to finish later — no track is made."):
        rec = MS.save({**recipe, "name": clean})
        st.session_state[f"{ns}_id"] = rec["id"]
        st.toast(f"Recipe saved: {clean}")


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
        st.rerun()


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
        c = st.columns([2, 2, 3], vertical_alignment="center")
        pk = f"mxm_part_{p['uid']}"
        pending = st.session_state.setdefault("_mxm_part_pending", {}).pop(p["uid"], None)
        if pending and has:
            st.session_state[pk] = pending          # e.g. instrumental once vocals were laid over it
        st.session_state.setdefault(pk, p.get("part", "full") if has else "full")
        p["part"] = c[0].selectbox("Play", list(PART_LABELS), key=pk, format_func=PART_LABELS.get,
                                   disabled=not has, label_visibility="collapsed",
                                   help="What of this track plays here. Instrumental / vocals need its stems "
                                        "(run_audio_analysis.bat on the laptop)." if has
                                   else "This track has no stems yet — run run_audio_analysis.bat on the laptop (it makes stems) "
                                        "to play just its instrumental or vocals.")
        others = sorted(tracks_all, key=str.lower)
        with c[1].popover("➕ Lay over…", width="stretch", disabled=not others):
            if others:   # a disabled popover's contents still run
                _add_layer_form(p, t, others, tracks_all, summaries)
        mine = [ly for ly in _layers() if ly["on"] == p["uid"]]
        for ly in mine:
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


def _add_layer_form(p: dict, host: dict, others: list, tracks_all: dict, summaries: dict) -> None:
    """➕ Lay another track (its full song, or with stems its vocals or instrumental) over part p."""
    if not others:
        return
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
        st.rerun()


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


def _tx_set_preset(stk: str, pick_key: str) -> None:
    name = st.session_state.get(pick_key)
    if name in SM.CURVE_PRESETS and stk in st.session_state:
        s = st.session_state[stk]
        s["curves"] = None if name == "even" else SM.norm_curves({"preset": name})
        s["from"] = f"preset:{name}"


def _neural_join(b: dict, uid: str, c, a: dict) -> None:
    """🧠 The neural join's own controls: profile, bars, 📍 auto-place."""
    import mixer
    own = (b.get("xfade") or {}).get("neural") or {}
    pk, bk = f"mxm_nnp_{uid}", f"mxm_nnb_{uid}"
    st.session_state.setdefault(pk, own.get("profile") or "smooth")
    st.session_state.setdefault(bk, int(own.get("bars") or 8))
    profile = c[2].selectbox("Profile", list(SM.PROFILES), key=pk, label_visibility="collapsed",
                             format_func=lambda k: SM.PROFILES[k].split(" — ")[0],
                             help="  \n".join(SM.PROFILES.values()))
    bars = c[3].selectbox("Bars", NEURAL_BARS, key=bk, label_visibility="collapsed",
                          format_func=lambda n: f"{n} bars", help="How long the blend lasts.")
    cur = SM.preset_of(own.get("curves")) or "hand"
    opts = list(SM.CURVE_PRESETS) + (["hand"] if cur == "hand" else [])
    ck = f"mxm_nnc_{uid}"
    if st.session_state.get(ck, "∅") not in opts:
        st.session_state[ck] = cur
    pick = st.selectbox("Curve shape", opts, key=ck, format_func=_preset_label, help=CURVE_HELP)
    curves = own.get("curves") if pick == "hand" else (None if pick == "even" else SM.norm_curves({"preset": pick}))
    b["xfade"] = {"neural": {"profile": profile, "bars": int(bars), **({"curves": curves} if curves else {})}}
    c[4].button("📍 Place", key=f"mxm_nnplace_{uid}", width="stretch", on_click=_neural_autoplace,
                args=(a["uid"], uid),
                help="Sets where the blend starts in this part (the last phrase line with room for it) and "
                     "makes sure the next part has its bars before it to blend in with.")
    st.button("🎚️ Fine-tune in the two-deck editor", key=f"mxm_nnedit_{uid}", on_click=_open_join_in_transition,
              args=(dict(a), dict(b)), help="Track A over track B: drag B to where it comes in, play each with a "
                                             "cursor, preview just the transition, then ✅ Apply back here.")
    ta, tb = full_track(a["track"]), full_track(b["track"])
    missing = [x["track"] for x, t in ((a, ta), (b, tb)) if t is not None and not mixer.has_four_stems(t)]
    if missing:
        st.caption(f":orange[🧠 {', '.join(dict.fromkeys(missing))} not split into drums / bass / other yet — "
                   "run run_audio_analysis.bat on the laptop. Until then this join is an ordinary crossfade.]")
    else:
        st.caption(f":gray[🧠 {SM.PROFILES[profile]} The next part comes in {bars} bars early; each stem gets its "
                   "own curve — vocals swap sharply, drums fade out fast, bass swaps on a beat, the rest blends "
                   "evenly.]")


def _join_row(a: dict, b: dict, i: int) -> None:
    """🔀 The join from part i to part i + 1: its own crossfade length, where it sits around the
    bar line, its shape and 🎛️ bass swap (stored on the incoming part as "xfade"; "Default" = ⚙️'s
    setting). Lengths and splits dragged on the timeline show as 🖐 choices."""
    uid = b["uid"]
    own = b.get("xfade") or {}
    same = a["track"] == b["track"]
    lk, wk, sk, bk = f"mxm_xfl_{uid}", f"mxm_xfw_{uid}", f"mxm_xfs_{uid}", f"mxm_xfb_{uid}"
    default = "Default"
    # The part's own "xfade" is the truth. Streamlit forgets a widget's value whenever it isn't drawn (another
    # Mix mode, the 🎚️ Transition editor…), so each box is rebuilt from it then — a 🧠 join stays 🧠
    # (it used to come back as "Default" and overwrite the join — 23 Oct 2026).
    if lk not in st.session_state:
        st.session_state[lk] = NEURAL if own.get("neural") else \
            _len_label(float(own["beats"])) if own.get("beats") is not None else default
    st.session_state.setdefault(wk, own.get("where") if own.get("where") is not None else "centre")
    st.session_state.setdefault(sk, own.get("shape") or ("linear" if same else "smooth"))
    st.session_state.setdefault(bk, bool(own.get("bass_swap")))
    len_opts = [default, *XF_JOIN] + ([] if same else [NEURAL])
    if st.session_state[lk] not in len_opts:
        len_opts.append(st.session_state[lk])
    where_opts = list(XF_WHERE_LABELS)
    if st.session_state[wk] not in where_opts:
        where_opts.append(st.session_state[wk])
    with st.container(key=f"mxm_join_{uid}"):
        c = st.columns([2, 2, 3, 2, 2], vertical_alignment="center")
        c[0].caption(f"🔀 **Join {i} → {i + 1}**")
        length = c[1].selectbox("Crossfade", len_opts, key=lk, label_visibility="collapsed",
                                format_func=lambda x: f"Default ({st.session_state.get('mxm_xf', '1 bar')})"
                                if x == default and not same else ("Default (¼ beat)" if x == default else x),
                                help="How long both parts play together while one fades into the other. "
                                     "Long ones (4–16 bars) blend the two songs over each other, DJ style. "
                                     "Or drag the crossfade's ◆ edges on the timeline below.")
        if length == NEURAL:
            _neural_join(b, uid, c, a)
            return
        cut = length == "Cut"
        where = c[2].selectbox("Where", where_opts, key=wk, format_func=_where_label,
                               label_visibility="collapsed", disabled=cut,
                               help="Both sides: half before the bar line, half after. Next song comes in early: "
                                    "it starts under the end of this part (playing the bars just before its "
                                    "chosen start) and has taken over by the line. This song carries on: it keeps "
                                    "playing under the start of the next part. 🖐 = as dragged on the timeline.")
        shape = c[3].selectbox("Shape", list(XF_SHAPE_LABELS), key=sk, format_func=XF_SHAPE_LABELS.get,
                               label_visibility="collapsed", disabled=cut,
                               help="Smooth (equal power) keeps the loudness even when two different songs blend; straight "
                                    "suits the same audio (a jump within one track).")
        bass = c[4].toggle("🎛️ Bass swap", key=bk, disabled=cut,
                           help="Only one song's bass plays during the crossfade: the outgoing song keeps its low end "
                                "until the middle of the fade, then hands it to the incoming one on the beat — the "
                                "DJ trick that stops a long blend getting muddy. Needs a crossfade of a beat or more.")
    if length == default:
        beats = None
    elif length in XF_JOIN:
        beats = XF_JOIN[length]
    else:
        beats = float(own.get("beats") or 0.0)            # 🖐 dragged on the timeline
    if beats is None and where == "centre" and shape == ("linear" if same else "smooth") and not bass:
        b.pop("xfade", None)
    else:
        b["xfade"] = {"beats": beats, "where": where, "shape": shape, **({"bass_swap": True} if bass else {})}


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
            st.rerun()
        cols = st.columns(min(4, len(used)))
        for i, tid in enumerate(used):
            key = f"mxm_shift_{tid}"
            st.session_state.setdefault(key, 0)
            cols[i % len(cols)].number_input(tid[:28], -6, 6, step=1, key=key,
                                             help=_key_text(tracks[tid].get("key"), st.session_state[key]))

    bpm = float(st.session_state.get("mxm_bpm") or 120.0) if st.session_state.get("mxm_match", True) else None
    shift = {tid: int(st.session_state.get(f"mxm_shift_{tid}", 0)) for tid in used}
    grids = {tid: MX.grid(t) for tid, t in tracks.items()}

    _apply_timeline_drag(parts)
    _nmsg = st.session_state.pop("mxm_neural_msg", None)
    if _nmsg:
        st.info(_nmsg)
    st.markdown("**Parts**")
    prev = None
    for i, p in enumerate(parts):
        t, g = tracks[p["track"]], grids[p["track"]]
        speed, _ = MX.tempo_factor(g["bpm"], bpm)
        if prev is not None:
            _join_row(prev, p, i)
        with st.container(border=True):
            c = st.columns([5, 2, 1, 1, 1, 1])
            c[0].markdown(f"**{i + 1}. {p['track']}** — {format_mmss(p['start'])}–{format_mmss(p['end'])} "
                          f"({format_mmss((p['end'] - p['start']) / speed)})"
                          + ("" if p.get("part", "full") == "full" else f" · {PART_LABELS[p['part']].lower()}"))
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
            p["gain_db"] = c[1].number_input("Level (dB)", -12.0, 12.0, step=1.0, key=gk,
                                             label_visibility="collapsed", help="Level of this part (dB)")
            c[2].button("↑", key=f"mxm_up_{p['uid']}", on_click=_move, args=(i, -1), disabled=i == 0)
            c[3].button("↓", key=f"mxm_dn_{p['uid']}", on_click=_move, args=(i, 1), disabled=i == len(parts) - 1)
            _part_trim(p, g, c[4])
            c[5].button("🗑", key=f"mxm_rm_{p['uid']}", on_click=_remove, args=(i,))
            _part_stems_row(p, t, tracks_all, summaries)
        prev = p

    with st.expander(f"🤝 Good partners for {parts[0]['track']}"):
        rows = _partners(summaries[parts[0]["track"]], tracks_all)[:10] if parts[0]["track"] in summaries else []
        for r in rows:
            st.caption(f"**{r['track_id']}** — fits {r['score'] * 100:.0f} %: {r['key_label']}"
                       + (f" (shift {r['shift']:+d})" if r["shift"] else "")
                       + f", tempo {(r['speed'] - 1) * 100:+.1f} %")

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
    name = " + ".join(used[:3]) + (" …" if len(used) > 3 else "")
    _preview_and_save(recipe, tracks, {}, f"{name} (mashup)"[:90], "mxm")
    st.button("🧹 Start a new mashup", key="mxm_clear", on_click=_clear, args=("mxm",))


# ---------------------------------------------------------------------------
# 🎚️ Transition — two decks: the top suggestions for where to mix A into B, the editor, a quick preview
# ---------------------------------------------------------------------------

def _tx_state_key(a: str, b: str) -> str:
    return f"mxt_state::{a}::{b}"


def _tx_use(a: str, b: str, sug: dict) -> None:
    st.session_state[_tx_state_key(a, b)] = {"a_end": sug["a_end"], "b_full": sug["b_full"], "bars": int(sug["bars"]),
                                             "profile": sug["profile"], "curves": None, "from": "suggestion"}
    st.session_state["mxt_profile"] = sug["profile"]
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


def _tx_to_mashup(a: str, b: str, ta: dict, tb: dict, s: dict, append: bool) -> None:
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


def _tx_apply_join(a_uid: str, b_uid: str, s: dict) -> None:
    """✅ Back into the 🧩 Mashup join this was opened from."""
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


def _tx_live(a: str, b: str, ta: dict, tb: dict, s: dict):
    """⚡ The live preview's stems for this transition (prepared when they don't cover it yet), as URLs —
    or None (audio not on the server yet / couldn't prepare)."""
    import live_preview as LP
    import mixer
    from cut_editor import media_url
    have = st.session_state.get("mxt_live")
    if not LP.still_fits(have, a, b, ta, tb, s):
        need = LP.needed_files(ta, tb)
        if any(not _has_file(mixer.local_path(t, part)) for t in (ta, tb) for part in need[t.get("track_id")]):
            return None
        with st.spinner("⚡ Getting the live preview ready (tempo-matching the stems)…"):
            try:
                have = LP.prepare(a, b, ta, tb, s, have=have)
            except Exception as e:
                st.caption(f":orange[⚡ Live preview unavailable: {e}]")
                return None
        st.session_state["mxt_live"] = have
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
        st.session_state["mxt_a"] = ids[0]
    a = c[0].selectbox("Track A (playing)", ids, key="mxt_a", format_func=lambda x: _label(tracks[x]),
                       disabled=bool(join))
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
        st.markdown("**🎯 Suggested transitions** — tap one to load it, then fine-tune below")
        cols = st.columns(len(sugs))
        for i, (col, sg) in enumerate(zip(cols, sugs)):
            with col.container(border=True):
                on = abs(sg["a_end"] - s["a_end"]) < 0.05 and abs(sg["b_full"] - s["b_full"]) < 0.05
                badge = "green" if sg["score"] >= 0.8 else "orange" if sg["score"] >= 0.6 else "gray"
                st.markdown(f":{badge}-badge[{sg['score'] * 100:.0f} %] **{SM.PROFILES[sg['profile']].split(' — ')[0]}**"
                            f" · {sg['bars']} bars")
                st.caption(f"A hands over at **{format_mmss(sg['a_end'])}** · B fully in at "
                           f"**{format_mmss(sg['b_full'])}**" + (f"  \n{', '.join(sg['why'])}" if sg["why"] else ""))
                st.button("✓ Loaded" if on else "Use this", key=f"mxt_use_{i}", width="stretch", disabled=on,
                          on_click=_tx_use, args=(a, b, sg))

    # The two decks
    speed_b, _ = MX.tempo_factor(gb["bpm"], ga["bpm"])
    urls = {}
    for tid, t in ((a, ta), (b, tb)):
        path = mixer.local_path(t, "full")
        urls[tid] = media_url(path, "audio/mp4" if path.suffix == ".m4a" else "audio/mpeg", f"mxt_audio_{tid}") \
            if _has_file(path) else ""
    pv = st.session_state.get("mxt_pv")
    rec_short = _tx_recipe(a, b, ta, tb, s, short=True)
    pv_ok = pv and pv.get("sig") == json.dumps(rec_short, sort_keys=True) and Path(pv["path"]).exists()
    s.setdefault("profile", st.session_state.get("mxt_profile") or "smooth")
    live = _tx_live(a, b, ta, tb, s)
    val = DE.deck_editor(a=DE.lane_data(ta, a, ga, urls[a]), b=DE.lane_data(tb, b, gb, urls[b]), state=s,
                         bar_a=ga["bar_len"], bar_b=gb["bar_len"], speed_b=speed_b, live=live,
                         beats_per_bar=int(ga["beats_per_bar"]), state_key=f"{a}|{b}|{s.get('from')}", key="mxt_deck")
    if isinstance(val, dict) and val.get("nonce") and val["nonce"] != st.session_state.get("_mxt_nonce"):
        st.session_state["_mxt_nonce"] = val["nonce"]
        s.update(a_end=float(val["a_end"]), b_full=float(val["b_full"]), bars=int(val["bars"]),
                 curves=SM.norm_curves(val.get("curves")) if val.get("curves") else s.get("curves"), **{"from": "drag"})
        st.rerun()
    missing = [x for x in (a, b) if urls[x] == ""]
    if missing:
        st.caption(f"▶ A / ▶ B need the song on the server: {', '.join(missing)} — ⬇️ fetch below.")
        _audio_ready({x: {"full"} for x in missing}, "mxt_src")

    # Curve shape presets (the ● handles above shape them further)
    cur = SM.preset_of(s.get("curves")) or "hand"
    opts = list(SM.CURVE_PRESETS) + (["hand"] if cur == "hand" else [])
    st.session_state["mxt_curve"] = cur
    st.pills("Curve shape", opts, key="mxt_curve", format_func=_preset_label, help=CURVE_HELP,
             on_change=_tx_set_preset, args=(stk, "mxt_curve"))

    # Profile, length, preview
    st.session_state.setdefault("mxt_profile", s.get("profile") or "smooth")
    pc = st.columns([2, 1, 1.4], vertical_alignment="bottom")
    prof = pc[0].selectbox("Profile", list(SM.PROFILES), key="mxt_profile",
                           format_func=lambda k: SM.PROFILES[k].split(" — ")[0], help="  \n".join(SM.PROFILES.values()))
    s["profile"] = prof
    bars = pc[1].selectbox("Blend", NEURAL_BARS, index=NEURAL_BARS.index(int(s["bars"])) if int(s["bars"]) in NEURAL_BARS else 3,
                           format_func=lambda n: f"{n} bars", key=f"mxt_bars_{a}_{b}_{s['bars']}")
    if bars != s["bars"]:
        s["bars"] = int(bars)
        st.rerun()
    no_stems = [x for x, t in ((a, ta), (b, tb)) if not mixer.has_four_stems(t)]
    if no_stems:
        st.caption(f":orange[🧠 {', '.join(no_stems)} not split into drums / bass / other yet — the preview is an "
                   "ordinary crossfade until run_audio_analysis.bat has made the stems.]")
    tracks_full = {a: ta, b: tb}
    with pc[2]:
        ready = _audio_ready(mixer.parts_needed(rec_short, tracks_full), "mxt")
    if ready and st.button("🎧 Render it exactly", key="mxt_preview",
                           help="▶ Live mix above is the quick way to listen. This renders the transition (8 bars "
                                "either side) exactly as a saved mix will sound — studio-quality tempo matching."):
        out = mixer.preview_path(rec_short)
        with st.spinner("Mixing the transition…"):
            try:
                mixer.render(rec_short, tracks_full, out)
            except Exception as e:
                st.error(f"Couldn't make the preview: {e}")
                return
        st.session_state["mxt_pv"] = {"path": str(out), "sig": json.dumps(rec_short, sort_keys=True)}
        st.rerun()
    if pv_ok:
        st.audio(str(pv["path"]))
    elif pv:
        st.caption("The exact render is out of date — render it again to hear the changes.")

    # Where it goes next
    st.divider()
    bc = st.columns(3)
    if join:
        bc[0].button("✅ Apply to the mashup", key="mxt_apply", type="primary", width="stretch",
                     on_click=_tx_apply_join, args=(join[0], join[1], dict(s)))
        bc[1].button("✖ Cancel", key="mxt_cancel", width="stretch",
                     on_click=lambda: (st.session_state.pop("mxt_join", None),
                                       st.session_state.__setitem__("mx_mode", MASHUP)))
    else:
        bc[0].button("🧩 Start a mashup with it", key="mxt_new", type="primary", width="stretch",
                     on_click=_tx_to_mashup, args=(a, b, ta, tb, dict(s), False),
                     help="A from 16 bars before the blend, then B to its end — carry on adding parts in 🧩 Mashup.")
        parts = st.session_state.get("mxm_parts") or []
        can_append = bool(parts) and parts[-1]["track"] == a
        bc[1].button("➕ Add to my mashup", key="mxt_append", width="stretch", disabled=not can_append,
                     on_click=_tx_to_mashup, args=(a, b, ta, tb, dict(s), True),
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
        c[1].button("🧩 Start it", key=f"mx_sug_go_d_{i}", width="stretch", on_click=_start_drop_swap,
                    args=(r["a"], r["b"]))


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
                    on_click=_start_suggestion, args=(r["a"], r["b"], vocal_swap))


# ---------------------------------------------------------------------------
# 📂 Saved mixes
# ---------------------------------------------------------------------------

def _clear(ns: str) -> None:
    for k in [k for k in st.session_state if k.startswith(ns + "_") or k.startswith("_" + ns + "_")]:
        del st.session_state[k]


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
            c[1].button("✏️", key=f"mxs_open_{m['id']}", on_click=_open, args=(m,), disabled=bool(missing),
                        help="Open in the editor to change it, preview and save again."
                        if not missing else f"Source track(s) gone: {', '.join(missing)}")
            if out and not out.get("uploaded"):
                if c[2].button("☁️", key=f"mxs_up_{m['id']}", help="Upload the track to Google Drive",
                               disabled=drive_token() is None):
                    err = MS.upload_output(m["id"], drive_token())
                    st.toast(err or "Uploaded to Google Drive ✅")
                    st.rerun()
            if c[3].button("🗑", key=f"mxs_del_{m['id']}",
                           help="Forget this recipe. A track already made from it stays in the library."):
                MS.delete(m["id"])
                st.rerun()


# ---------------------------------------------------------------------------
# Page section
# ---------------------------------------------------------------------------

def render(summaries: dict) -> None:
    """summaries: the Media Library's load_audio_library() rows."""
    st.session_state.setdefault("mx_mode", MODES[0])
    with st.container(key="subnav_mix"):   # a sub menu — chips (page_setup.menu_css)
        mode = st.radio("Mix", MODES, horizontal=True, key="mx_mode", label_visibility="collapsed")
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
