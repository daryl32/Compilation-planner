"""⚔️ Verses Battle — the rules (no Streamlit here; the page and the compile are in planner/verses.py, the render step
in verses_render.py).

Two actors battle for your attention. The compilation runs in four rounds:
  1 · Entrances     — each actor gets a solo spotlight with her music
  2 · Trade-offs    — the picture swaps between them on the bar, every 4 bars, then every 2
  3 · Head to head  — split screen; in its second half the panels swing (65/35) towards whoever leads
  4 · Final         — a swap on every bar, then the winner's spotlight (a split screen for a draw)

Music: 🤝 shared — one song both actors are matched to; 🎵 solo themes — each actor has her own song, both "running"
from the start like two decks, and you hear the song of whoever is on screen (in a split, whoever leads).

Everything here works in battle time (seconds from the start of the battle, the intro card not counted). For both
music modes battle time is also each song's own time, so the footage matched to a song stays in sync with it.
First draft: 18 Nov 2026.
"""

from __future__ import annotations

import random

ROUNDS = [  # (name, share of the battle)
    ("Entrances", 0.25),
    ("Trade-offs", 0.35),
    ("Head to head", 0.22),
    ("Final", 0.18),
]
LAYOUTS = {"mixed": "🔀 Rounds (full screen + split)", "split": "◧ Split screen throughout",
           "full": "⏩ Full screen only"}
MUSIC_MODES = {"shared": "🤝 Shared track", "solo": "🎵 Solo themes"}
WINNERS = {"surprise": "🎲 Surprise winner", "A": "👤 First actor wins", "B": "👤 Second actor wins", "draw": "🤝 Draw"}
DRAFT_STYLES = {"any": "🎲 Any two", "contrast": "⚡ Contrast (least alike)", "rivals": "🪞 Rivals (most alike)"}
LENGTHS = {0: "Full song", 120: "2 min", 180: "3 min", 240: "4 min"}
MIN_VIDEOS = 1                    # an actor needs at least this many videos (with a 🧭 timeline) to be drafted
SPOTLIGHT = (0.65, 0.35)          # the leader's / the other's share of the frame in a spotlight split
FADE_OUT_SEC = 2.5                # the music fades out over the last seconds when the battle is shorter than the song
XFADE_SEC = 0.12                  # solo themes: each switch of song is a short crossfade
INTRO_SEC = 4.0                   # the tale-of-the-tape card before the battle
COLOURS = {"A": (235, 64, 90), "B": (60, 150, 245)}      # each side's colour on the overlay


def other(side: str) -> str:
    return "B" if side == "A" else "A"


# ---------------------------------------------------------------------------
# 🎲 The draft
# ---------------------------------------------------------------------------

def actor_videos(details: dict, eligible: set) -> dict:
    """{actor: [video ids]} — every actor's videos among `eligible` (videos with a 🧭 timeline). details:
    library_common.video_details() — {video_id: {"actors": [...]}}."""
    out = {}
    for vid, d in details.items():
        if vid not in eligible:
            continue
        for a in d.get("actors") or []:
            out.setdefault(a, []).append(vid)
    return {a: sorted(v, key=str.lower) for a, v in out.items()}


def fighter_videos(by_actor: dict, a: str, b: str) -> tuple:
    """(A's videos, B's videos): each side's videos without the other actor in them — a video with both counts for
    the side that has fewer, so neither is left empty when she only appears with her rival."""
    va, vb = set(by_actor.get(a) or []), set(by_actor.get(b) or [])
    both = va & vb
    va, vb = va - both, vb - both
    for vid in sorted(both, key=str.lower):
        (va if len(va) <= len(vb) else vb).add(vid)
    return sorted(va, key=str.lower), sorted(vb, key=str.lower)


def eligible_actors(by_actor: dict, min_videos: int = MIN_VIDEOS) -> list:
    return sorted([a for a, v in by_actor.items() if len(v) >= min_videos], key=str.lower)


def _cos(x, y) -> float:
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(x @ y / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-9))


def draft(actors: list, style: str = "any", rng: random.Random = None, first: str = None, looks: dict = None) -> tuple:
    """🎲 Two fighters. style: "any" | "contrast" (the least alike pair) | "rivals" (the most alike); `first` keeps one
    fighter and drafts her opponent. looks: {actor: look vector (mean footage embedding)} — needed for contrast /
    rivals (without it those draw at random). A little chance stays: the opponent comes from the best three."""
    rng = rng or random.Random()
    pool = list(actors)
    if len(pool) < 2:
        raise ValueError("Not enough actors to draft — at least two need videos with a 🧭 footage timeline.")
    a = first if first in pool else rng.choice(pool)
    rest = [x for x in pool if x != a]
    if style in ("contrast", "rivals") and looks and a in looks and any(x in looks for x in rest):
        scored = sorted(((x, _cos(looks[a], looks[x])) for x in rest if x in looks), key=lambda kv: kv[1],
                        reverse=style == "rivals")
        return a, rng.choice([x for x, _ in scored[:3]])
    return a, rng.choice(rest)


# ---------------------------------------------------------------------------
# 🥊 The rounds → windows
# ---------------------------------------------------------------------------

def bar_grid(track: dict, total: float) -> list:
    """The track's bar lines (seconds) up to `total`: the analysis's regular bars, else every 4th beat, else 2 s."""
    beats = [float(b) for b in track.get("beat_times") or []]
    bars = []
    if len(beats) >= 8:
        try:
            import core.bars as BARS
            bars = BARS.regular_bars(beats, track.get("downbeats") or [], int(track.get("beats_per_bar") or 4),
                                     fallback_phase=0)
        except Exception:          # noqa: BLE001 — a patchy analysis: every 4th beat will do
            bars = []
        if len(bars) < 4:
            bars = beats[::4]
    if len(bars) < 4:
        bars = [i * 2.0 for i in range(int(total // 2.0) + 1)]
    return [b for b in bars if 0.0 <= b <= total]


def _snap(t: float, grid: list, lo: float, hi: float) -> float:
    inside = [g for g in grid if lo + 0.5 < g < hi - 0.5]
    return min(inside, key=lambda g: abs(g - t)) if inside else t


def _every(a: float, b: float, n_bars: int, grid: list) -> list:
    """Cut points between a and b on every n_bars-th bar line of the grid (a bar ≈ the grid's spacing)."""
    lines = [g for g in grid if a + 0.25 < g < b - 0.25]
    if not lines:
        step = 2.0 * n_bars
        out, t = [], a + step
        while t < b - 0.25:
            out.append(t)
            t += step
        return out
    return lines[n_bars - 1::n_bars] if n_bars > 1 else lines


def schedule(total: float, grid_a: list, grid_b: list = None, layout: str = "mixed", winner: str = "A",
             first: str = "A") -> list:
    """The battle as windows [{"start", "end", "side": "A" | "B" | "split", "round" (1–4), "round_name",
    "lead": "A" | "B" | None, "cells": [panel A, panel B] for a split}] covering [0, total) edge to edge.
    grid_a / grid_b: each side's bar lines (the same for a shared track) — a swap to a side lands on her bar line.
    winner: "A" | "B" | "draw". first: who opens."""
    grid_b = grid_b if grid_b is not None else grid_a
    grid_of = {"A": grid_a, "B": grid_b, "split": grid_a}
    edges, t = [0.0], 0.0
    for _, share in ROUNDS[:-1]:
        t += share * total
        edges.append(_snap(t, grid_a, edges[-1], total))
    edges.append(total)
    out = []

    def add(a, b, side, rnd, lead=None, cells=None):
        if b - a < 1e-3:
            return
        out.append({"start": round(a, 3), "end": round(b, 3), "side": side, "round": rnd,
                    "round_name": ROUNDS[rnd - 1][0], "lead": lead if side == "split" else side,
                    **({"cells": cells} if cells else {})})

    def alternate(a, b, rnd, n_bars, start_side, split=False):
        side = start_side
        cuts = [a]
        for c in _every(a, b, n_bars, grid_of[other(side)] if not split else grid_a):
            cuts.append(c)
        cuts.append(b)
        for x, y in zip(cuts, cuts[1:]):
            if split:
                add(x, y, "split", rnd, lead=side, cells=_cells(side))
            else:
                add(x, y, side, rnd)
            side = other(side)
        return side

    split_all = layout == "split"
    full_only = layout == "full"
    second = other(first)
    # 1 · Entrances: the first half hers, the second half the other's (whole screen even in "split throughout")
    r_a, r_b = edges[0], edges[1]
    mid = _snap((r_a + r_b) / 2, grid_of[second], r_a, r_b)
    add(r_a, mid, first, 1)
    add(mid, r_b, second, 1)
    # 2 · Trade-offs: every 4 bars, then every 2
    r_a, r_b = edges[1], edges[2]
    mid = _snap((r_a + r_b) / 2, grid_a, r_a, r_b)
    side = alternate(r_a, mid, 2, 4, first, split=split_all)
    side = alternate(mid, r_b, 2, 2, side, split=split_all)
    # 3 · Head to head: an even split, then the panels swing to whoever leads (full screen only: a swap every bar)
    r_a, r_b = edges[2], edges[3]
    if full_only:
        side = alternate(r_a, r_b, 3, 1, side)
    else:
        mid = _snap((r_a + r_b) / 2, grid_a, r_a, r_b)
        for x, y, lead in _pairs(_every(r_a, mid, 4, grid_a), r_a, mid, side):
            add(x, y, "split", 3, lead=lead, cells=_cells(None))
        side = alternate(mid, r_b, 3, 2, side, split=True)
    # 4 · Final: a swap every bar, then the winner's spotlight (a split for a draw)
    r_a, r_b = edges[3], edges[4]
    spot = _snap(r_a + 0.4 * (r_b - r_a), grid_a, r_a, r_b)
    alternate(r_a, spot, 4, 1, side, split=split_all)
    if winner == "draw":
        add(spot, r_b, "split", 4, lead=None, cells=_cells(None))
    else:
        add(spot, r_b, winner, 4)
    for k, w in enumerate(out):
        w["index"] = k
    return out


def _pairs(cuts: list, a: float, b: float, lead: str):
    """(start, end, lead) for consecutive cut points, the lead swapping each time (whose song plays in solo mode)."""
    pts = [a] + list(cuts) + [b]
    for x, y in zip(pts, pts[1:]):
        yield x, y, lead
        lead = other(lead)


def _cells(lead: str | None) -> list:
    """Split-screen panels [A's, B's] as (x0, y0, x1, y1) fractions: A on the left, B on the right; the leader's
    panel wider in a spotlight split."""
    if lead is None:
        return [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]
    wa = SPOTLIGHT[0] if lead == "A" else SPOTLIGHT[1]
    return [[0.0, 0.0, wa, 1.0], [wa, 0.0, 1.0, 1.0]]


def round_spans(windows: list) -> list:
    """[{"round", "name", "start", "end"}] — each round's stretch."""
    out = {}
    for w in windows:
        r = out.setdefault(w["round"], {"round": w["round"], "name": w["round_name"], "start": w["start"], "end": w["end"]})
        r["start"], r["end"] = min(r["start"], w["start"]), max(r["end"], w["end"])
    return [out[k] for k in sorted(out)]


def screen_time(windows: list) -> dict:
    """{"A": s, "B": s} — a full-screen window counts whole, a split counts each panel by its share of the frame."""
    got = {"A": 0.0, "B": 0.0}
    for w in windows:
        d = w["end"] - w["start"]
        if w["side"] == "split":
            cells = w.get("cells") or _cells(None)
            got["A"] += d * (cells[0][2] - cells[0][0])
            got["B"] += d * (cells[1][2] - cells[1][0])
        else:
            got[w["side"]] += d
    return got


# ---------------------------------------------------------------------------
# 🎬 The footage: each side's own compile, woven by the windows
# ---------------------------------------------------------------------------

def slice_timeline(timeline: list, a: float, b: float) -> list:
    """The stretch [a, b) of a timeline (compilation time kept): entries cut at both ends — the clips that run into
    the stretch start later in their footage."""
    out = []
    for e in timeline:
        ea, eb = float(e["track_time"][0]), float(e["track_time"][1])
        if eb <= a + 1e-3 or ea >= b - 1e-3:
            continue
        head = max(a - ea, 0.0)
        keep = min(eb, b) - max(ea, a)
        scenes = []
        for slot in e.get("scenes") or []:
            skip, left, chain = head, keep, []
            for link in slot.get("chain") or []:
                d = float(link["clip_duration_sec"])
                if skip >= d - 1e-3:
                    skip -= d
                    continue
                take = min(d - skip, left)
                if take <= 1e-3:
                    break
                chain.append({**link, "clip_start_sec": round(float(link["clip_start_sec"]) + skip, 3),
                              "clip_duration_sec": round(take, 3)})
                left -= take
                skip = 0.0
            if chain:
                new = {**slot, "chain": chain}
                if "clip_duration_sec" in slot:
                    new["clip_duration_sec"] = round(sum(x["clip_duration_sec"] for x in chain), 3)
                scenes.append(new)
        out.append({**e, "scenes": scenes, "track_time": [round(max(ea, a), 3), round(min(eb, b), 3)]})
    return out


def weave(windows: list, timelines: dict) -> list:
    """The battle's timeline: each window filled from its side's timeline ({"A": [...], "B": [...]}, both in battle
    time); a split window holds one panel per side, cut wherever either side cuts. Every entry carries
    "verses_side", "verses_round" and (splits) "cells"; segment_index is renumbered."""
    out = []
    for w in windows:
        a, b = w["start"], w["end"]
        if w["side"] in ("A", "B"):
            got = slice_timeline(timelines.get(w["side"]) or [], a, b)
            for e in got:
                e["scenes"] = e["scenes"][:1]
            got = _fill_gaps(got, a, b)
            for e in got:
                out.append({**e, "verses_side": w["side"], "verses_round": w["round"], "verses_window": w["index"]})
            continue
        parts = {s: slice_timeline(timelines.get(s) or [], a, b) for s in ("A", "B")}
        cuts = sorted({a, b} | {float(e["track_time"][k]) for s in parts for e in parts[s] for k in (0, 1)
                                if a + 0.2 < float(e["track_time"][k]) < b - 0.2})
        for x, y in zip(cuts, cuts[1:]):
            scenes = []
            for s in ("A", "B"):
                got = slice_timeline(parts[s], x, y)
                scenes.append(got[0]["scenes"][0] if got and got[0]["scenes"] else {"chain": []})
            if not any(sc["chain"] for sc in scenes):
                scenes = []
            elif not all(sc["chain"] for sc in scenes):      # one side has nothing here: the other fills the frame
                scenes = [sc for sc in scenes if sc["chain"]]
            entry = {"track_time": [round(x, 3), round(y, 3)], "scenes": scenes, "verses_side": "split",
                     "verses_round": w["round"], "verses_window": w["index"]}
            if len(scenes) == 2:
                entry["cells"] = w.get("cells") or _cells(None)
            out.append(entry)
    for k, e in enumerate(out):
        e["segment_index"] = k
        if k:                                              # edge to edge (rounding leaves the odd millisecond)
            e["track_time"][0] = out[k - 1]["track_time"][1]
    out = [e for e in out if e["track_time"][1] - e["track_time"][0] > 1e-3]
    for k, e in enumerate(out):
        e["segment_index"] = k
    return out


def _fill_gaps(entries: list, a: float, b: float) -> list:
    """Entries stretched so [a, b) is covered edge to edge (a gap left by an unfilled block is black)."""
    if not entries:
        return [{"track_time": [round(a, 3), round(b, 3)], "scenes": []}]
    out = []
    t = a
    for e in entries:
        ea, eb = e["track_time"]
        if ea > t + 1e-3:
            out.append({"track_time": [round(t, 3), round(ea, 3)], "scenes": []})
        out.append(e)
        t = eb
    if t < b - 1e-3:
        out.append({"track_time": [round(t, 3), round(b, 3)], "scenes": []})
    out[0]["track_time"][0] = round(a, 3)
    return out


# ---------------------------------------------------------------------------
# 🎵 The sound and 📊 the meter
# ---------------------------------------------------------------------------

def audio_parts(windows: list, tracks: dict, total: float, mode: str, durations: dict) -> list:
    """[{"side", "track_id", "at" (battle time), "src" (the song's own time), "dur"}] — what plays when.
    shared: one part, the whole battle; solo: the song of whoever is on screen (in a split, whoever leads; an even
    split keeps the song already playing), joined parts of the same song merged."""
    if mode == "shared":
        return [{"side": "A", "track_id": tracks["A"], "at": 0.0, "src": 0.0, "dur": round(total, 3)}]
    out, playing = [], "A"
    for w in windows:
        side = w["side"] if w["side"] in ("A", "B") else (w.get("lead") or playing)
        playing = side
        a, b = w["start"], w["end"]
        if out and out[-1]["side"] == side and abs(out[-1]["at"] + out[-1]["dur"] - a) < 1e-3:
            out[-1]["dur"] = round(b - out[-1]["at"], 3)
            continue
        dur = float(durations.get(tracks[side]) or total)
        out.append({"side": side, "track_id": tracks[side], "at": round(a, 3), "src": round(a % max(dur, 1e-3), 3),
                    "dur": round(b - a, 3)})
    return out


def meter(windows: list, winner: str, step: float = 0.5) -> list:
    """📊 The tug-of-war meter: [(time, A's share 0–1)] every `step` seconds — the running share of screen time,
    leaning towards the winner over the Final round (to 0.8 / 0.2; level for a draw)."""
    total = windows[-1]["end"] if windows else 0.0
    final_start = next((w["start"] for w in windows if w["round"] == len(ROUNDS)), total)
    target = {"A": 0.8, "B": 0.2, "draw": 0.5}.get(winner, 0.5)
    out, t = [], 0.0
    while t <= total + 1e-6:
        seen = screen_time([{**w, "end": min(w["end"], t)} for w in windows if w["start"] < t])
        s = seen["A"] + seen["B"]
        share = 0.5 if s < 1e-6 else 0.5 + 0.6 * (seen["A"] / s - 0.5)      # (calmer than the raw share)
        if t > final_start and total > final_start:
            k = min((t - final_start) / max(total - final_start, 1e-6) * 1.6, 1.0)
            share = share + (target - share) * k
        out.append((round(t, 3), round(min(max(share, 0.03), 0.97), 3)))
        t += step
    return out


def pick_winner(winner: str, rng: random.Random) -> str:
    return rng.choice(["A", "B"]) if winner == "surprise" else winner


def describe(windows: list) -> list:
    """One line per round, e.g. "Round 2 · Trade-offs · 0:41–1:39 · 9 swaps"."""
    out = []
    for r in round_spans(windows):
        ws = [w for w in windows if w["round"] == r["round"]]
        splits = sum(1 for w in ws if w["side"] == "split")
        bits = [f"Round {r['round']} · {r['name']} · {_mmss(r['start'])}–{_mmss(r['end'])}"]
        if len(ws) > 1:
            bits.append(f"{len(ws) - 1} swap{'s' if len(ws) != 2 else ''}")
        if splits:
            bits.append("split screen" if splits == len(ws) else f"{splits} split")
        out.append(" · ".join(bits))
    return out


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# ---------------------------------------------------------------------------
# 🖼️ Pictures: the overlay and the tale of the tape
# ---------------------------------------------------------------------------

def _font(size: int, bold: bool = True):
    import edge_mode as E
    return E._font(size, bold)


def _text(draw, xy, text, font, fill=(255, 255, 255), anchor="la", stroke=2):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))


def overlay_frame(size: tuple, names: dict, window: dict, share: float, winner: str = None, show_winner: bool = False):
    """One overlay picture (RGBA, see-through): the names in their colours, a VS badge, the round, the tug-of-war
    meter (A's share fills from the left), and — over the winner's spotlight — the result."""
    from PIL import Image, ImageDraw
    W, H = size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f_name, f_small, f_vs = _font(max(H // 22, 11)), _font(max(H // 34, 9)), _font(max(H // 18, 12))
    pad = max(W // 80, 6)
    bar_w, bar_h = int(W * 0.42), max(H // 45, 6)
    bx, by = (W - bar_w) // 2, pad + max(H // 40, 6)
    # the meter: A from the left, B from the right
    split_x = bx + int(bar_w * share)
    d.rounded_rectangle([bx - 2, by - 2, bx + bar_w + 2, by + bar_h + 2], radius=bar_h, fill=(0, 0, 0, 150))
    d.rectangle([bx, by, split_x, by + bar_h], fill=COLOURS["A"] + (235,))
    d.rectangle([split_x, by, bx + bar_w, by + bar_h], fill=COLOURS["B"] + (235,))
    d.line([split_x, by - 3, split_x, by + bar_h + 3], fill=(255, 255, 255, 255), width=2)
    _text(d, (bx - pad, by + bar_h // 2), names["A"].upper(), f_name, COLOURS["A"] + (255,), anchor="rm")
    _text(d, (bx + bar_w + pad, by + bar_h // 2), names["B"].upper(), f_name, COLOURS["B"] + (255,), anchor="lm")
    _text(d, (W // 2, by + bar_h + pad // 2), "VS", f_vs, (255, 210, 60, 255), anchor="mt")
    _text(d, (W // 2, H - pad), f"ROUND {window['round']} · {window['round_name'].upper()}", f_small,
          (255, 255, 255, 220), anchor="mb")
    if window["side"] == "split" and window.get("cells"):        # each panel's name at its foot
        for k, s in enumerate(("A", "B")):
            x0, _, x1, _ = window["cells"][k]
            _text(d, (int((x0 + x1) / 2 * W), H - pad - max(H // 22, 11)), names[s], f_small, COLOURS[s] + (255,),
                  anchor="mb")
    if show_winner:
        f_big = _font(max(H // 9, 18))
        line = "IT'S A DRAW" if winner == "draw" else f"WINNER: {names[winner].upper()}"
        col = (255, 210, 60, 255) if winner == "draw" else COLOURS[winner] + (255,)
        _text(d, (W // 2, H // 2), line, f_big, col, anchor="mm", stroke=4)
    return img


def tale_card(size: tuple, names: dict, stats: dict, thumbs: dict, title: str = "VERSES BATTLE"):
    """🥊 The tale of the tape: each fighter's best thumbnail, her name and a few facts, with a VS between them.
    stats: {"A": [(label, value)], "B": [...]}; thumbs: {"A": PIL image or JPEG bytes or None, "B": …}."""
    import io
    from PIL import Image, ImageDraw
    W, H = size
    img = Image.new("RGB", (W, H), (14, 14, 20))
    d = ImageDraw.Draw(img)
    for k, s in enumerate(("A", "B")):                       # each half washed in her colour
        r, g, b = COLOURS[s]
        x0 = 0 if k == 0 else W // 2
        for y in range(H):
            a = 0.10 + 0.18 * (y / H)
            d.line([x0, y, x0 + W // 2, y], fill=(int(14 + (r - 14) * a), int(14 + (g - 14) * a), int(20 + (b - 20) * a)))
    f_title, f_name, f_stat, f_vs = _font(max(H // 16, 14)), _font(max(H // 14, 14)), _font(max(H // 30, 9), False), _font(max(H // 6, 24))
    _text(d, (W // 2, H // 22), title, f_title, (255, 210, 60), anchor="mt", stroke=3)
    tw, th = int(W * 0.30), int(H * 0.42)
    for k, s in enumerate(("A", "B")):
        cx = W // 4 if k == 0 else 3 * W // 4
        top = H // 6
        pic = thumbs.get(s)
        try:
            if isinstance(pic, (bytes, bytearray)):
                pic = Image.open(io.BytesIO(pic))
            if pic is not None:
                pic = pic.convert("RGB")
                f = max(tw / pic.width, th / pic.height)
                pic = pic.resize((max(int(pic.width * f), 1), max(int(pic.height * f), 1)))
                l, t = (pic.width - tw) // 2, (pic.height - th) // 2
                img.paste(pic.crop((l, t, l + tw, t + th)), (cx - tw // 2, top))
        except Exception:          # noqa: BLE001 — a thumbnail that can't be read is left out
            pic = None
        d.rectangle([cx - tw // 2 - 2, top - 2, cx + tw // 2 + 2, top + th + 2], outline=COLOURS[s], width=3)
        _text(d, (cx, top + th + H // 40), names[s].upper(), f_name, COLOURS[s], anchor="mt", stroke=3)
        y = top + th + H // 40 + int(H / 12)
        for label, value in (stats.get(s) or [])[:4]:
            _text(d, (cx, y), f"{label}: {value}", f_stat, (230, 230, 230), anchor="mt", stroke=1)
            y += int(H / 22)
    _text(d, (W // 2, int(H * 0.40)), "VS", f_vs, (255, 210, 60), anchor="mm", stroke=4)
    return img
