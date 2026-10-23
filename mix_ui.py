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
from library_common import format_mmss

MODES = ["📏 Extend / shorten", "🧩 Mashup", "💡 Suggestions", "📂 Saved mixes"]
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
               if (t := full_track(tid)) and not mixer.local_path(t, part).exists()]
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

def _preview_and_save(recipe: dict, tracks: dict, sims: dict, default_name: str, ns: str) -> None:
    import mixer
    import mixes as MS
    from cut_editor import media_url
    pv_path = mixer.preview_path(recipe)
    pv = st.session_state.get(f"{ns}_pv") or {}
    fresh = pv.get("path") == str(pv_path) and pv_path.exists()
    url = media_url(pv_path, "audio/mpeg" if pv_path.suffix == ".mp3" else "audio/mp4", f"{ns}_mix_audio") \
        if fresh else ""
    lay = _timeline(recipe, tracks, sims, energy=pv.get("energy") if fresh else None, audio_url=url, ns=ns,
                    editable=ns == "mxm")
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

    if not _audio_ready(mixer.parts_needed(recipe), ns):
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
    lines = sorted({0.0, *[float(x) for x in g["bars"]], g["duration"]})
    lo, hi = MX.snap_to_bar(g, secs[a]["start"]), MX.snap_to_bar(g, secs[b]["end"])
    rng = st.select_slider("Fine-tune (bar lines)", options=lines, value=(lo, hi), key=f"mxm_rng_{tid}_{a}_{b}",
                           format_func=format_mmss)
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


def _join_row(a: dict, b: dict, i: int) -> None:
    """🔀 The join from part i to part i + 1: its own crossfade length, where it sits around the
    bar line, its shape and 🎛️ bass swap (stored on the incoming part as "xfade"; "Default" = ⚙️'s
    setting). Lengths and splits dragged on the timeline show as 🖐 choices."""
    uid = b["uid"]
    own = b.get("xfade") or {}
    same = a["track"] == b["track"]
    lk, wk, sk, bk = f"mxm_xfl_{uid}", f"mxm_xfw_{uid}", f"mxm_xfs_{uid}", f"mxm_xfb_{uid}"
    default = "Default"
    if lk not in st.session_state:
        st.session_state[lk] = _len_label(float(own["beats"])) if own.get("beats") is not None else default
    st.session_state.setdefault(wk, own.get("where") if own.get("where") is not None else "centre")
    st.session_state.setdefault(sk, own.get("shape") or ("linear" if same else "smooth"))
    st.session_state.setdefault(bk, bool(own.get("bass_swap")))
    len_opts = [default, *XF_JOIN]
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
    st.markdown("**Parts**")
    prev = None
    for i, p in enumerate(parts):
        t, g = tracks[p["track"]], grids[p["track"]]
        speed, _ = MX.tempo_factor(g["bpm"], bpm)
        if prev is not None:
            _join_row(prev, p, i)
        with st.container(border=True):
            c = st.columns([5, 2, 1, 1, 1])
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
            c[4].button("🗑", key=f"mxm_rm_{p['uid']}", on_click=_remove, args=(i,))
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
    st.session_state["mx_mode"] = MODES[1]


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
    parts = [{"track": a, "start": plan["a"][0], "end": plan["a"][1], "gain_db": 0.0, "uid": f"S{t0}",
              "part": "full", "label": "build-up"},
             {"track": b, "start": plan["b"][0], "end": plan["b"][1], "gain_db": 0.0, "uid": f"S{t0}b",
              "part": "full", "label": "drop",
              "xfade": {"beats": float(MX.DROP_XFADE_BEATS), "where": "centre", "shape": "smooth", "bass_swap": True}}]
    st.session_state["mxm_parts"] = parts
    st.session_state["mxm_layers"] = []
    for p in parts:
        st.session_state[f"mxm_part_{p['uid']}"] = p["part"]
    st.session_state["mxm_bpm"] = round(float(ta.get("bpm") or 120.0), 1)
    st.session_state["mxm_match"] = True
    st.session_state[f"mxm_shift_{b}"] = K.best_shift(ta.get("key"), tb.get("key"))
    st.session_state["mx_mode"] = MODES[1]


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
                       f"build-up, then {MX.DROP_AFTER_BARS} bars from the drop — beat-matched, key-fitted, with "
                       "🎛️ bass swap on the drop.")
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
        st.session_state["mx_mode"] = MODES[0]
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
        st.session_state["mx_mode"] = MODES[1]


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
    if mode == MODES[0]:
        _extend(summaries)
    elif mode == MODES[1]:
        _mashup(summaries)
    elif mode == MODES[2]:
        _suggestions(summaries)
    else:
        _saved(summaries)
