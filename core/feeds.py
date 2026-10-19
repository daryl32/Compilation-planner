"""🔮 Recommendations step 2 — the feeds, worked out from the stored searches
(rec_library.all_results: {track_id: stored results with "results" and "insights"}).

  🏆 top_plans     — the best plans across every track (overall %)
  💥 best_action   — plans whose drops / action parts land best: motion match there,
                     plus 💥 Action clips in the story map's action parts
  🧩 hidden_gems   — videos that fit many tracks well but you've rarely used lately
  🕳️ gaps          — story roles the library is short of, and kinds of block nothing
                     in the library matches well (what to film / find next)
  (🆕 the new-video spotlight is stored per video by rec_library.run_spotlight)

Pure functions, no Streamlit.
"""

from __future__ import annotations

import numpy as np

PER_TRACK = 2            # at most this many plans from one track in a feed (variety)
GEM_TOP_TRACKS = 5       # a video's fit = its average over its best this-many tracks
WEAK_FIT = 0.60          # a kind of block whose best library fit averages below this is a gap
MIN_GAP_BLOCKS = 5


def _plans(results: dict):
    """(track_id, plan) for every stored plan."""
    for tid, data in results.items():
        for r in data.get("results") or []:
            if r.get("plan"):
                yield tid, r["plan"]


def _ranked(items: list, key, n: int) -> list:
    """Best first, no plan twice, at most PER_TRACK per track."""
    out, seen, per = [], set(), {}
    for tid, plan in sorted(items, key=lambda x: -key(x[1])):
        k = (tid, plan.get("style"), plan.get("pace"), tuple(plan.get("videos") or ()))
        if k in seen or per.get(tid, 0) >= PER_TRACK:
            continue
        seen.add(k)
        per[tid] = per.get(tid, 0) + 1
        out.append({"track_id": tid, "plan": plan, "score": round(key(plan), 1)})
        if len(out) >= n:
            break
    return out


def top_plans(results: dict, n: int = 10) -> list:
    """[{"track_id", "plan", "score"}] — the best overall plans across the library."""
    return _ranked(list(_plans(results)), lambda p: p["scores"]["overall"], n)


def action_score(plan: dict) -> float | None:
    """Motion match in the drops / action parts, averaged with the share of the story map's
    💥 Action parts that got an Action clip (when the track has any)."""
    sc = plan["scores"]
    parts = [x for x in (sc.get("action"), sc.get("action_story")) if x is not None]
    return float(np.mean(parts)) if parts else None


def best_action(results: dict, n: int = 10) -> list:
    """[{"track_id", "plan", "score"}] — the plans whose drops / action parts land best."""
    items = [(tid, p) for tid, p in _plans(results) if action_score(p) is not None]
    return _ranked(items, action_score, n)


def video_fits(results: dict) -> dict:
    """{video_id: {track_id: fit 0–1}} from each search's insights."""
    out = {}
    for tid, data in results.items():
        for vid, fit in ((data.get("insights") or {}).get("videos") or {}).items():
            out.setdefault(vid, {})[tid] = fit
    return out


def hidden_gems(results: dict, usage: dict, n: int = 6) -> list:
    """Videos in the top half for fit (average over their best tracks) that you've used no more
    than the typical video lately — the better the fit and the less used, the higher.
    [{"video_id", "fit" (0–100), "uses", "tracks": [(track_id, fit 0–100)…], "why"}]"""
    fits = video_fits(results)
    if not fits:
        return []
    avg = {vid: float(np.mean(sorted(f.values(), reverse=True)[:GEM_TOP_TRACKS])) for vid, f in fits.items()}
    cut = float(np.median(list(avg.values())))
    uses = {vid: int(usage.get(vid, 0)) for vid in fits}
    typical = float(np.median(list(uses.values()))) if uses else 0.0
    out = []
    for vid, fit in avg.items():
        if fit < cut or uses[vid] > typical:
            continue
        top = sorted(fits[vid].items(), key=lambda x: -x[1])[:3]
        out.append({"video_id": vid, "fit": round(100 * fit, 1), "uses": uses[vid],
                    "tracks": [(t, round(100 * f, 1)) for t, f in top],
                    "gem": fit / (1.0 + uses[vid]),
                    "why": (f"Fits its best tracks {100 * fit:.0f}% on average"
                            + (", and you haven't used it in your recent projects or renders" if uses[vid] == 0
                               else f", but used only {uses[vid]}× in your recent projects and renders"))})
    out.sort(key=lambda g: -g["gem"])
    return out[:n]


INTENSITY_BANDS = [(0.0, 0.35, "Calm"), (0.35, 0.7, "Medium"), (0.7, 1.01, "Intense")]
LENGTH_BANDS = [(0.0, 1.5, "short shots (under 1.5 s)"), (1.5, 4.0, "medium shots (1.5–4 s)"),
                (4.0, 1e9, "long shots (over 4 s)")]


def gaps(results: dict, role_supply: dict, role_names: dict = None) -> dict:
    """{"roles": [...], "blocks": [...]}.
    roles: per story role the tracks use — how many tracks, typical seconds needed per track,
      seconds of footage marked with it (role_supply: {role: seconds}) and how many tracks'
      worth that is; status "none" / "short" / "ok".
    blocks: kinds of block (intensity × length) whose best fit anywhere in the library averages
      under WEAK_FIT — worst first."""
    role_names = role_names or {}
    need = {}
    blocks = []
    for tid, data in results.items():
        ins = data.get("insights") or {}
        for r, sec in (ins.get("need") or {}).items():
            need.setdefault(r, []).append(sec)
        for b in ins.get("blocks") or []:
            blocks.append((tid, b))
    roles = []
    for r, secs in need.items():
        typical = float(np.median(secs))
        have = float(role_supply.get(r, 0.0))
        cover = have / typical if typical > 0 else 0.0
        roles.append({"role": r, "name": role_names.get(r, r), "tracks": len(secs), "need_sec": round(typical, 1),
                      "have_sec": round(have, 1), "cover": round(cover, 2),
                      "status": "none" if have <= 0 else "short" if cover < 1.5 else "ok"})
    roles.sort(key=lambda x: (x["cover"], -x["tracks"]))
    kinds = []
    for lo, hi, iname in INTENSITY_BANDS:
        for llo, lhi, lname in LENGTH_BANDS:
            sel = [(t, b) for t, b in blocks if lo <= b["intensity"] < hi and llo <= b["len"] < lhi]
            if len(sel) < MIN_GAP_BLOCKS:
                continue
            fit = float(np.mean([b["best"] for _, b in sel]))
            if fit < WEAK_FIT:
                kinds.append({"label": f"{iname}, {lname}", "blocks": len(sel), "tracks": len({t for t, _ in sel}),
                              "fit": round(100 * fit, 1)})
    kinds.sort(key=lambda k: k["fit"])
    return {"roles": roles, "blocks": kinds}


__all__ = ["top_plans", "best_action", "action_score", "video_fits", "hidden_gems", "gaps"]
