"""🎚️ Mixing — planning extended tracks and mashups. No audio, no Streamlit.

A mix is a *recipe*: an ordered list of pieces, each a stretch of one source
track between two bar lines. Playing the pieces back to back — each joined to
the next on its bar line with a short crossfade — is the new track. Extending
a song is a recipe whose pieces all come from one track (a jump back repeats
bars, a jump forward skips them); a mashup mixes pieces from several tracks,
all stretched to one tempo.

    recipe = {
        "id": "mix_…", "name": "…", "kind": "extend" | "mashup",
        "pieces": [{"track": tid, "start": s, "end": e, "gain_db": 0.0, "label": "…",
                    "part": "full" | "instrumental" | "vocals", "uid": "…"}, …],
        "layers": [{"track": tid, "part": "vocals", "start": s, "end": e, "gain_db": 0.0,
                    "on": piece uid, "offset_bars": bars into that piece}, …],
                                            # 🎤 played OVER the pieces (stems from analyse_audio_local.py)
        "bpm": target BPM or None (None = every piece at its own tempo),
        "shift": {tid: semitones},          # 🎹 pitch shift per track (key matching)
        "xfade_same": beats, "xfade_other": beats,   # default crossfade at joins; a piece's own
                                            # "xfade": {"beats", "where", "shape", "bass_swap"} sets the
                                            # join into it ("where": a XF_WHERE name, or the share of
                                            # the fade before the bar line, 0–1, as dragged on the graph)
        "fade_in": sec, "fade_out": sec, "level_match": True,
        "settings": {…},                    # what the Mix screen needs to reopen it
    }

mixer.py renders a recipe with ffmpeg (and works out the bar features used
here); mixes.py stores recipes. Everything in this module works on numbers
only, so the tests drive it with synthetic grids.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

MAX_STRETCH = 0.08          # ±8 %: tempo changes beyond this get a warning
TEMPO_CHOICES = (1.0, 2.0, 0.5)   # a track may play at double / half time to meet the target
JUMP_CONTEXT_BARS = 2        # bars compared on each side of a jump
MIN_LOOP_BARS = 4
MAX_LOOP_BARS = 64
EDGE_BARS = 2                # never jump inside the first / last bars
# 🔀 Where a crossfade sits around the bar line where one piece hands over to the next:
# the share of it before the line. "before" = the next song comes in under the end of this
# one and has taken over by the line; "after" = this one carries on under the start of the next.
XF_WHERE = {"centre": 0.5, "before": 1.0, "after": 0.0}
XF_SHAPES = ("smooth", "linear")   # equal power (different songs) / straight (the same audio)
# 🎛️ Bass swap: during a crossfade only one song's low end plays — the outgoing one's until the
# swap point (the beat nearest the middle of the fade), then the incoming one's — so two bass
# lines never pile up into mud. Everything above BASS_HZ fades as usual.
BASS_HZ = 180.0
BASS_SWAP_BEATS = 0.25      # how quickly the low end changes hands at the swap point


def where_frac(where) -> float:
    """The share of a crossfade before the bar line: a XF_WHERE name or a number 0–1."""
    if isinstance(where, (int, float)) and not isinstance(where, bool):
        return float(min(max(where, 0.0), 1.0))
    return XF_WHERE.get(where, 0.5)


def where_from_frac(frac: float):
    """A dragged split back to a XF_WHERE name when it is one, else the number (3 places)."""
    frac = round(float(min(max(frac, 0.0), 1.0)), 3)
    for name, f in XF_WHERE.items():
        if abs(f - frac) < 1e-3:
            return name
    return frac


# ---------------------------------------------------------------------------
# A track's grid
# ---------------------------------------------------------------------------

def grid(track: dict) -> dict:
    """Beats, bar lines (downbeats), bar length, sections and phrase levels of a
    catalogue track, cleaned up. Bar lines fall back to every 4th beat."""
    a = track.get("analysis") or {}
    duration = float(track.get("duration_sec") or 0.0)
    beats = np.asarray(track.get("beat_times") or [], dtype=float)
    bpm = float(track.get("bpm") or 0.0)
    if not bpm and len(beats) > 1:
        bpm = 60.0 / float(np.median(np.diff(beats)))
    bpm = bpm or 120.0
    bpb = int(a.get("beats_per_bar") or 4)
    db = np.asarray(a.get("downbeats") or [], dtype=float)
    if len(db) < 2 and len(beats) >= bpb:
        db = beats[::bpb]
    if len(db) < 2:
        bar = bpb * 60.0 / bpm
        db = np.arange(0.0, max(duration - 0.01, 0.0), bar)
    db = np.unique(np.round(db[(db >= 0) & (db < duration - 0.05)], 3))
    bar_len = float(np.median(np.diff(db))) if len(db) > 1 else bpb * 60.0 / bpm
    levels = {}
    for p in a.get("phrases") or []:
        i = int(np.argmin(np.abs(db - float(p["time"])))) if len(db) else 0
        if len(db) and abs(db[i] - float(p["time"])) < 0.3 * bar_len:
            levels[i] = max(levels.get(i, 0), int(p.get("level", 4)))
    sections = []
    for s in a.get("sections") or []:
        sections.append({"start": float(s["start"]), "end": float(s["end"]), "label": s.get("label", "?"),
                         "energy": float(s.get("energy", 0.5))})
    if not sections:
        sections = [{"start": 0.0, "end": duration, "label": "A", "energy": 0.5}]
    return {"duration": duration, "bpm": bpm, "beats": beats, "bars": db, "bar_len": bar_len,
            "beats_per_bar": bpb, "levels": levels, "sections": sections}


def snap_to_bar(g: dict, t: float) -> float:
    """Nearest bar line — or the very start / end of the track when that's closer."""
    cands = np.concatenate([[0.0], g["bars"], [g["duration"]]])
    return float(cands[int(np.argmin(np.abs(cands - t)))])


def beat_len(bpm: float) -> float:
    return 60.0 / max(float(bpm), 1.0)


# ---------------------------------------------------------------------------
# Tempo and level
# ---------------------------------------------------------------------------

def tempo_factor(src_bpm: float, target_bpm: float | None) -> tuple:
    """(speed, effective source BPM): speed > 1 plays faster. A track may count
    at double or half time when that's the closer match (e.g. 70 → 140)."""
    if not target_bpm or not src_bpm:
        return 1.0, float(src_bpm or 0.0)
    best = min(((target_bpm / (src_bpm * m), src_bpm * m) for m in TEMPO_CHOICES),
               key=lambda x: abs(math.log(x[0])))
    return float(best[0]), float(best[1])


def stretch_note(speed: float) -> str | None:
    """A warning when a piece is stretched far enough to sound processed."""
    change = speed - 1.0
    if abs(change) <= MAX_STRETCH:
        return None
    return f"{'sped up' if change > 0 else 'slowed down'} {abs(change) * 100:.0f} % — may sound stretched"


def loudness_db(y: np.ndarray, sr: int, block_sec: float = 0.4) -> float:
    """Gated loudness of a signal (dB, an approximation of LUFS without the
    K-filter): mean power over 400 ms blocks, ignoring blocks 10 dB below the
    average and near-silence — so quiet intros don't drag it down."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = int(block_sec * sr)
    if len(y) < n:
        return -70.0
    blocks = y[: len(y) // n * n].reshape(-1, n).astype(np.float64)
    p = (blocks ** 2).mean(axis=1)
    p = p[p > 1e-7]
    if not len(p):
        return -70.0
    rel = p[p > p.mean() * 0.1]
    return float(10 * np.log10((rel if len(rel) else p).mean()))


# ---------------------------------------------------------------------------
# Laying a recipe out in time
# ---------------------------------------------------------------------------

def layout(recipe: dict, grids: dict) -> dict:
    """Where every piece lands in the new track.

    Returns {"pieces": [{…piece, "speed", "out_start", "out_end", "src_bpm"}],
    "duration", "bpm", "joins": [{"at", "xfade", "same_track"}],
    "beats", "bars" (the new track's beat grid, carried over from the sources)}.
    Each piece is stretched by its track's speed; piece k+1 starts exactly
    where piece k's last bar line falls, so bar lines line up across the join.
    """
    target = recipe.get("bpm")
    out, t = [], 0.0
    beats_out, bars_out = [], []
    for p in recipe.get("pieces") or []:
        g = grids[p["track"]]
        speed, eff = tempo_factor(g["bpm"], target)
        start = max(0.0, float(p["start"]))
        end = min(float(p["end"]), g["duration"])
        if end - start < 0.05:
            continue
        length = (end - start) / speed
        out.append({**p, "start": start, "end": end, "speed": speed, "src_bpm": g["bpm"], "eff_bpm": eff,
                    "out_start": t, "out_end": t + length})
        for src, dst in ((g["beats"], beats_out), (g["bars"], bars_out)):
            sel = src[(src >= start - 1e-3) & (src < end - 1e-3)]
            dst.extend(t + (sel - start) / speed)
        t += length
    out_bpm = float(target) if target else (
        float(np.median([p["src_bpm"] * p["speed"] for p in out])) if out else 120.0)
    joins = []
    for a, b in zip(out[:-1], out[1:]):
        same = a["track"] == b["track"]
        own = b.get("xfade") or {}              # 🔀 this join's own crossfade (set on the incoming piece)
        beats = float(own["beats"]) if own.get("beats") is not None else \
            float(recipe.get("xfade_same", 0.25) if same else recipe.get("xfade_other", 4))
        where = own.get("where") if own.get("where") is not None else "centre"
        shape = own.get("shape") or ("linear" if same else "smooth")
        xf = max(beats, 0.0) * beat_len(out_bpm)
        frac = where_frac(where)                  # share of the fade before the bar line
        pre, post = xf * frac, xf * (1 - frac)
        # the fade before the join plays over the outgoing piece, the one after over the incoming one
        pre = min(pre, 0.9 * (a["out_end"] - a["out_start"]))
        post = min(post, 0.9 * (b["out_end"] - b["out_start"]))
        if pre + post < 0.01:
            pre = post = 0.005
        joins.append({"at": b["out_start"], "xfade": pre + post, "pre": pre, "post": post, "where": where,
                      "shape": shape, "beats": beats, "same_track": same,
                      "bass_swap": bool(own.get("bass_swap")) and pre + post >= beat_len(out_bpm) - 1e-6})
    layers = []
    by_uid = {p.get("uid"): p for p in out if p.get("uid")}
    for ly in recipe.get("layers") or []:
        host = by_uid.get(ly.get("on"))
        if host is None or ly["track"] not in grids:
            continue
        g = grids[ly["track"]]
        speed, eff = tempo_factor(g["bpm"], target)
        start, end = max(0.0, float(ly["start"])), min(float(ly["end"]), g["duration"])
        hg = grids[host["track"]]
        offset = (float(ly["offset_bars"]) * hg["bar_len"] / host["speed"] if ly.get("offset_bars") is not None
                  else float(ly.get("offset") or 0.0))       # bars of the host part, or output seconds
        at = host["out_start"] + max(0.0, offset)
        if end - start < 0.05 or at >= t:
            continue
        length = min((end - start) / speed, t - at)
        layers.append({**ly, "start": start, "end": start + length * speed, "speed": speed, "src_bpm": g["bpm"],
                       "eff_bpm": eff, "out_start": at, "out_end": at + length})
    beats_all = _dedupe(beats_out, 0.3 * beat_len(out_bpm))
    for jn in joins:
        if jn["bass_swap"]:
            jn["swap_at"] = swap_point(jn, beats_all)
    return {"pieces": out, "duration": t, "bpm": out_bpm, "joins": joins, "layers": layers,
            "beats": beats_all, "bars": _dedupe(bars_out, 1.5 * beat_len(out_bpm))}


def swap_point(jn: dict, beats: list) -> float:
    """🎛️ Where the low end changes hands: the beat nearest the middle of the crossfade (the bar
    line itself for a fade centred on it), else the middle."""
    lo, hi = jn["at"] - jn["pre"], jn["at"] + jn["post"]
    mid = (lo + hi) / 2
    inside = [b for b in beats if lo + 1e-3 < b < hi - 1e-3]
    return float(min(inside, key=lambda b: abs(b - mid))) if inside else float(mid)


def _dedupe(times: list, min_gap: float) -> list:
    out = []
    for x in sorted(times):
        if not out or x - out[-1] >= min_gap:
            out.append(round(float(x), 3))
    return out


# ---------------------------------------------------------------------------
# Jumps within one track (extend / shorten)
# ---------------------------------------------------------------------------

def bar_similarity(feats: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of bars (feats: one row per bar)."""
    f = np.asarray(feats, dtype=float)
    f = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-9)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    return f @ f.T


def jump_score(S: np.ndarray, i: int, j: int, k: int = JUMP_CONTEXT_BARS) -> float:
    """How seamless a jump from bar line i to bar line j sounds: the bars just
    before and after each line, compared pairwise (1 = identical)."""
    n = len(S)
    if min(i, j) - k < 0 or max(i, j) + k > n:
        return -1.0
    return float(np.mean([S[i + m, j + m] for m in range(-k, k)]))


def _structure_bonus(g: dict, i: int, j: int) -> float:
    lv = g["levels"]
    bonus = 0.0
    for x in (i, j):
        bonus += {16: 0.05, 8: 0.035, 4: 0.02}.get(lv.get(x, 0), 0.0)
    span = abs(i - j)
    bonus += 0.04 if span % 8 == 0 else (0.02 if span % 4 == 0 else -0.03)
    return bonus


def jump_candidates(g: dict, S: np.ndarray, direction: str = "back", min_bars: int = MIN_LOOP_BARS,
                    max_bars: int = MAX_LOOP_BARS, top: int = 60) -> list:
    """Good jumps, best first: [{"at": i, "to": j, "bars", "sec", "match", "score"}].
    direction "back" (j < i) repeats bars j…i-1; "forward" (j > i) skips bars i…j-1.
    Bar line i is bars[i]; i == len(bars) would be the track's end (never used)."""
    bars = g["bars"]
    n = min(len(bars), len(S))
    out = []
    lo, hi = max(EDGE_BARS, JUMP_CONTEXT_BARS), n - max(EDGE_BARS, JUMP_CONTEXT_BARS)
    for i in range(lo, hi + 1):
        for span in range(min_bars, max_bars + 1):
            j = i - span if direction == "back" else i + span
            if not (lo <= j <= hi):
                continue
            m = jump_score(S, i, j)
            if m < 0.35:
                continue
            a, b = sorted((i, j))
            sec = float(bars[b] - bars[a]) if b < len(bars) else float(g["duration"] - bars[a])
            out.append({"at": i, "to": j, "bars": span, "sec": sec, "match": m,
                        "score": m + _structure_bonus(g, i, j)})
    out.sort(key=lambda c: -c["score"])
    # keep variety: drop near-duplicates of a better jump (same lines ± 1 bar)
    kept = []
    for c in out:
        if all(abs(c["at"] - k["at"]) > 1 or abs(c["to"] - k["to"]) > 1 for k in kept):
            kept.append(c)
        if len(kept) >= top:
            break
    return kept


def _disjoint(chosen: list) -> bool:
    spans = sorted((min(c["at"], c["to"]), max(c["at"], c["to"])) for c in chosen)
    return all(a[1] <= b[0] for a, b in zip(spans[:-1], spans[1:]))


def plan_length(g: dict, S: np.ndarray, target_sec: float, max_jumps: int = 4, n_plans: int = 5,
                exact: bool = False) -> list:
    """Plans that bring the track to about target_sec, best first. Longer:
    jumps back (each may repeat 1–3 times); shorter: jumps forward. Each plan
    = {"jumps": [{…candidate, "repeat"}], "length", "error", "score",
    "pieces" (see pieces_from_jumps)}. exact: prefer plans that run slightly
    long, so the ending can be faded at exactly target_sec."""
    D = float(target_sec) - g["duration"]
    bar = g["bar_len"]
    if abs(D) < 0.5 * bar:
        return [{"jumps": [], "length": g["duration"], "error": -D, "score": 1.0,
                 "pieces": pieces_from_jumps(g, [])}]
    back = D > 0
    cands = jump_candidates(g, S, "back" if back else "forward")
    if not cands:
        return []
    repeats = (1, 2, 3) if back else (1,)
    tol = 0.5 * bar
    plans = []

    def consider(chosen, reps):
        if sum(reps) > max_jumps:
            return
        if len(chosen) > 1 and not _disjoint(chosen):
            return
        change = sum(c["sec"] * r for c, r in zip(chosen, reps))
        added = change if back else -change
        err = (g["duration"] + added) - target_sec
        if exact and err < -tol:
            return
        mean_match = float(np.mean([c["match"] for c in chosen]))
        mean_score = float(np.mean([c["score"] for c in chosen]))
        cost = (abs(err) / bar) * (0.35 if exact and err >= 0 else 1.0) + (1 - mean_score) * 4 \
            + 0.12 * sum(reps) + 0.1 * (len(chosen) - 1)
        plans.append((cost, chosen, reps, err, mean_match))

    for c in cands:
        for r in repeats:
            consider([c], [r])
    pool2 = cands[:35]
    for a, b in itertools.combinations(pool2, 2):
        for ra in repeats:
            for rb in repeats:
                consider([a, b], [ra, rb])
    pool3 = cands[:18]
    for trio in itertools.combinations(pool3, 3):
        for reps in itertools.product(repeats[:2], repeat=3):
            consider(list(trio), list(reps))
    plans.sort(key=lambda p: p[0])
    out, seen = [], set()
    for cost, chosen, reps, err, match in plans:
        jumps = [{**c, "repeat": r} for c, r in sorted(zip(chosen, reps), key=lambda x: min(x[0]["at"], x[0]["to"]))]
        key = tuple((j["at"], j["to"], j["repeat"]) for j in jumps)
        if key in seen:
            continue
        seen.add(key)
        out.append({"jumps": jumps, "length": target_sec + err, "error": err, "score": match,
                    "pieces": pieces_from_jumps(g, jumps)})
        if len(out) >= n_plans:
            break
    return out


def pieces_from_jumps(g: dict, jumps: list, track: str = None) -> list:
    """The pieces for a set of non-overlapping jumps (ordered by position).
    Back jump at→to repeated r times: play …→at, then (to→at) r-1 more times,
    carry on from to. Forward jump: play …→at, carry on from to."""
    bars = g["bars"]

    def t(i):
        return float(bars[i]) if i < len(bars) else g["duration"]

    pieces, pos = [], 0.0
    for jp in sorted(jumps, key=lambda j: min(j["at"], j["to"])):
        at, to = t(jp["at"]), t(jp["to"])
        pieces.append({"track": track, "start": pos, "end": at})
        if jp["to"] < jp["at"]:
            for _ in range(int(jp.get("repeat", 1)) - 1):
                pieces.append({"track": track, "start": to, "end": at})
        pos = to
    pieces.append({"track": track, "start": pos, "end": g["duration"]})
    return [p for p in pieces if p["end"] - p["start"] > 0.05]


def pieces_from_sections(g: dict, counts: list, track: str = None) -> list:
    """Manual extend / shorten: counts[k] = how many times section k plays
    (0 removes it). Sections that play once in a row stay one piece."""
    pieces = []
    for sec, n in zip(g["sections"], counts):
        for _ in range(int(n)):
            if pieces and abs(pieces[-1]["end"] - sec["start"]) < 1e-3:
                pieces[-1]["end"] = sec["end"]
                pieces[-1]["label"] = pieces[-1].get("label", "") + sec["label"]
            else:
                pieces.append({"track": track, "start": sec["start"], "end": sec["end"], "label": sec["label"]})
    return pieces


def join_quality(S: np.ndarray, g: dict, end_t: float, start_t: float) -> float | None:
    """For a same-track join (a piece ending at end_t followed by one starting at
    start_t): how alike the bars around the two lines are, 0–1. None if a line
    isn't on a bar or there isn't enough context (e.g. the very start)."""
    bars = g["bars"]
    if S is None or not len(bars):
        return None

    def idx(t):
        if abs(t - g["duration"]) < 1e-3:
            return len(bars)
        i = int(np.argmin(np.abs(bars - t)))
        return i if abs(bars[i] - t) < 0.15 * g["bar_len"] else None

    i, j = idx(end_t), idx(start_t)
    if i is None or j is None:
        return None
    m = jump_score(S, i, j)
    return None if m < -0.5 else max(0.0, min(1.0, m))


def describe_jump(g: dict, jp: dict) -> str:
    from_t = float(g["bars"][jp["at"]])
    to_t = float(g["bars"][jp["to"]]) if jp["to"] < len(g["bars"]) else g["duration"]
    mm = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"
    if jp["to"] < jp["at"]:
        rep = f" ×{jp['repeat']}" if jp.get("repeat", 1) > 1 else ""
        return (f"at {mm(from_t)} jump back to {mm(to_t)}{rep} — repeats {jp['bars']} bars "
                f"(+{jp['sec'] * jp.get('repeat', 1):.0f} s), match {jp['match'] * 100:.0f} %")
    return (f"at {mm(from_t)} skip to {mm(to_t)} — cuts {jp['bars']} bars (−{jp['sec']:.0f} s), "
            f"match {jp['match'] * 100:.0f} %")


# ---------------------------------------------------------------------------
# Mashup partners
# ---------------------------------------------------------------------------

def partner_score(a: dict, b: dict, keys_mod=None) -> dict:
    """How well track b fits with track a in a mashup: key (with a pitch shift
    of up to ±2 semitones if that makes it fit), tempo, and energy.
    a / b: {"track_id", "bpm", "key", "energy"} (energy = mean loudness 0–1)."""
    from core import keys as K
    speed, _ = tempo_factor(b.get("bpm") or 0, a.get("bpm") or 0)
    tempo = math.exp(-((abs(speed - 1.0) / MAX_STRETCH) ** 2) / 2) if a.get("bpm") and b.get("bpm") else 0.5
    shift = K.best_shift(a.get("key"), b.get("key"), max_semitones=2)
    key_score, key_label = K.compatibility(a.get("key"), K.shifted(b.get("key"), shift) if shift else b.get("key"))
    e = 1.0 - min(1.0, abs(float(a.get("energy") or 0.5) - float(b.get("energy") or 0.5)) * 2)
    score = 0.45 * key_score - 0.08 * abs(shift) + 0.4 * tempo + 0.15 * e
    return {"track_id": b["track_id"], "score": round(max(0.0, score), 3), "speed": speed, "shift": shift,
            "key_label": key_label, "key_score": key_score, "tempo_score": round(tempo, 2)}


# ---------------------------------------------------------------------------
# 💡 Mashup suggestions
# ---------------------------------------------------------------------------

def best_section(track: dict, by: str = "energy") -> dict:
    """The section to lead a suggested mashup with: the loudest ("energy") or the one with
    the most singing ("vocals", from the vocal analysis; falls back to energy). Short
    sections (under 4 bars) are passed over when there's anything longer."""
    g = grid(track)
    secs = g["sections"]
    long_enough = [s for s in secs if s["end"] - s["start"] >= 4 * g["bar_len"] - 0.05] or secs
    if by == "vocals":
        a = track.get("analysis") or {}
        v = a.get("vocals")
        if v:
            rate = float(a.get("rate", 30))
            v = np.asarray(v, dtype=float)

            def sing(s):
                seg = v[int(s["start"] * rate):int(s["end"] * rate)]
                return float(seg.mean()) if len(seg) else 0.0
            return max(long_enough, key=sing)
    return max(long_enough, key=lambda s: s["energy"])


def suggest_pairs(infos: list, top: int = 12, need_stems: bool = False) -> list:
    """Best (a, b) track pairs for a mashup, by partner_score in both directions, best first.
    infos: [{"track_id", "bpm", "key", "energy", "stems": bool}]. need_stems: only pairs where
    b has stems (b's vocals go over a's instrumental), and a has an instrumental too."""
    out = []
    for a in infos:
        for b in infos:
            if a["track_id"] >= b["track_id"] and not need_stems:
                continue
            if a["track_id"] == b["track_id"]:
                continue
            if need_stems and not (a.get("stems") and b.get("stems")):
                continue
            ab = partner_score(a, b)
            if not need_stems:
                ba = partner_score(b, a)
                if ba["score"] > ab["score"]:
                    ab, a_, b_ = ba, b, a
                else:
                    a_, b_ = a, b
            else:
                a_, b_ = a, b
            out.append({"a": a_["track_id"], "b": b_["track_id"], **ab})
    out.sort(key=lambda x: -x["score"])
    return out[:top]


# ---------------------------------------------------------------------------
# 💥 Drop swaps — track A's build-up crashes into track B's drop
# ---------------------------------------------------------------------------

DROP_LEAD_BARS = 8       # bars of A before its build-up starts, to set the scene
DROP_AFTER_BARS = 16     # bars of B from its drop
DROP_XFADE_BEATS = 1     # a quick swap on the drop's downbeat (with the bass swap)


def _drops_builds(track: dict) -> tuple:
    a = track.get("analysis") or {}
    drops = sorted(float(t) for t in a.get("drops") or [])
    builds = [(float(b["start"]), float(b["end"])) for b in a.get("builds") or []
              if isinstance(b, dict) and b.get("end", 0) > b.get("start", 0)]
    return drops, builds


def drop_swap_plan(track_a: dict, track_b: dict, lead_bars: int = DROP_LEAD_BARS,
                   after_bars: int = DROP_AFTER_BARS) -> dict | None:
    """Where to cut for "A's build-up → B's drop": {"a": (start, end), "b": (start, end), "build_bars",
    "a_drop", "b_drop"} in each track's own seconds, bar-snapped — or None if A has no build-up
    or B has no drop. A runs from a few bars before its longest build-up up to the drop the build
    leads into; B starts on its drop (the first with room for after_bars after it)."""
    ga, gb = grid(track_a), grid(track_b)
    drops_a, builds_a = _drops_builds(track_a)
    drops_b, _ = _drops_builds(track_b)
    if not builds_a or not drops_b:
        return None
    # A's build-up: the longest one, ending on (or near) a drop.
    def into_drop(b):
        return min(drops_a, key=lambda d: abs(d - b[1])) if drops_a else b[1]
    bs, be = max(builds_a, key=lambda b: b[1] - b[0])
    a_drop = into_drop((bs, be))
    if abs(a_drop - be) > 2 * ga["bar_len"]:
        a_drop = be
    a_end = snap_to_bar(ga, a_drop)
    a_start = snap_to_bar(ga, max(0.0, bs - lead_bars * ga["bar_len"]))
    if a_end - a_start < 2 * ga["bar_len"]:
        return None
    # B's drop: the first with room after it, else the latest.
    room = after_bars * gb["bar_len"]
    b_drop = next((d for d in drops_b if d + room <= gb["duration"] + 0.05), drops_b[-1])
    b_start = snap_to_bar(gb, b_drop)
    b_end = snap_to_bar(gb, min(gb["duration"], b_start + room))
    if b_end - b_start < 2 * gb["bar_len"]:
        return None
    return {"a": (a_start, a_end), "b": (b_start, b_end), "a_drop": a_drop, "b_drop": b_drop,
            "build_bars": round((a_end - max(bs, a_start)) / ga["bar_len"], 1)}


def suggest_drop_swaps(infos: list, top: int = 12) -> list:
    """Best (a, b) for a drop swap — a has a build-up, b a drop — by partner_score (b's key fitted to
    a's, b at a's tempo), best first. infos: as suggest_pairs, plus "builds" / "drops" counts."""
    out = []
    for a in infos:
        if not a.get("builds"):
            continue
        for b in infos:
            if b["track_id"] == a["track_id"] or not b.get("drops"):
                continue
            out.append({"a": a["track_id"], "b": b["track_id"], **partner_score(a, b)})
    out.sort(key=lambda x: -x["score"])
    return out[:top]
