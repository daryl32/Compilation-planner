"""
📈 Media Library → Analytics → 📚 Library: what the video library itself says —
pure functions, no Streamlit (the page gathers the data and draws the results).

  📋 progress        — per video: split pass, review, role marking, actors, range, ⭐
                       picture; totals; what to do next
  👤 actor_stats     — per actor: videos, usable footage, motion profile, role footage,
                       recent use, render ratings
  🎭 role_balance    — role footage you have vs what your tracks' story maps need, and
                       which unmarked videos most likely hold more of each role
  📏 length_profile  — usable clip lengths vs the blocks your tracks are cut into, and the
                       long clips whose motion swings most (the ones worth splitting)

"Usable" everywhere = not excluded and inside the video's library range (the footage the
planner and the recommender can use).
"""

from __future__ import annotations

import numpy as np

from core.footage import _overlap_with_range, _tc_to_seconds

LONG_SEC = 45.0            # a usable clip this long still wants splitting (1.5 × the 30 s auto-split target)
MIXED_MIN_SEC = 15.0       # clips at least this long are checked for motion swings
LEN_BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 15), (15, 30), (30, 60), (60, None)]


def _bin_label(lo, hi) -> str:
    return f"{lo}s+" if hi is None else f"{lo}–{hi}s"


def usable_scenes(cat: dict, rng) -> list:
    """[(scene, start, end)] — not excluded, the part inside the library range."""
    out = []
    for s in cat.get("scenes", []):
        if s.get("excluded"):
            continue
        try:
            a, b = _overlap_with_range(_tc_to_seconds(s["start_tc"]), _tc_to_seconds(s["end_tc"]), rng)
        except (KeyError, ValueError):
            continue
        if a is not None and b > a:
            out.append((s, a, b))
    return sorted(out, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# 📋 Progress
# ---------------------------------------------------------------------------

def video_progress(video_id: str, cat: dict, rng, actors: list, has_master: bool, roles_of,
                   long_sec: float = LONG_SEC) -> dict:
    us = usable_scenes(cat, rng)
    secs = sum(b - a for _, a, b in us)
    edited = sum(1 for s in cat.get("scenes", []) if s.get("edit") or s.get("split_from") is not None)
    long_left = sum(1 for _, a, b in us if b - a > long_sec)
    reviewed = sum(1 for s, _, _ in us if s.get("reviewed"))
    roled = [(s, a, b) for s, a, b in us if roles_of(s)]
    role_secs = {}
    for s, a, b in roled:
        for r in roles_of(s):
            role_secs[r] = role_secs.get(r, 0.0) + (b - a)
    n = len(us)
    return {
        "video_id": video_id, "clips": n, "minutes": round(secs / 60, 1), "edits": edited, "long_left": long_left,
        "split": "done" if edited and not long_left else "started" if edited else
                 ("not needed" if not long_left else "to do"),
        "reviewed_pct": round(100 * reviewed / n) if n else 0,
        "roles_pct": round(100 * len(roled) / n) if n else 0,
        "roles": "started" if roled else "to do",
        "role_secs": {r: round(v, 1) for r, v in role_secs.items()},
        "actors": list(actors or []), "has_range": bool(rng), "has_master": bool(has_master),
    }


def progress(catalogues: dict, ranges: dict, details: dict, masters: dict, roles_of, usage: dict = None,
             long_sec: float = LONG_SEC) -> dict:
    """{"rows": [per video], "totals": {...}, "next_split": [...], "next_roles": [...]}"""
    usage = usage or {}
    rows = [video_progress(vid, cat, ranges.get(vid), (details.get(vid) or {}).get("actors"), vid in masters,
                           roles_of, long_sec)
            for vid, cat in sorted(catalogues.items())]
    n = len(rows)
    totals = {
        "videos": n,
        "split_done": sum(r["split"] in ("done", "not needed") for r in rows),
        "split_started": sum(r["split"] == "started" for r in rows),
        "roles_started": sum(r["roles"] == "started" for r in rows),
        "actors": sum(bool(r["actors"]) for r in rows),
        "ranges": sum(r["has_range"] for r in rows),
        "masters": sum(r["has_master"] for r in rows),
        "minutes": round(sum(r["minutes"] for r in rows), 1),
        "reviewed_pct": round(np.average([r["reviewed_pct"] for r in rows],
                                         weights=[max(r["clips"], 1) for r in rows])) if rows else 0,
    }
    # Do next: most long clips left first; for roles, the most footage first, nudged up by recent use.
    next_split = sorted((r for r in rows if r["long_left"]), key=lambda r: (-r["long_left"], -r["minutes"]))[:5]
    next_roles = sorted((r for r in rows if r["roles"] == "to do" and r["clips"]),
                        key=lambda r: -(r["minutes"] * (1 + usage.get(r["video_id"], 0))))[:5]
    return {"rows": rows, "totals": totals, "next_split": next_split, "next_roles": next_roles}


# ---------------------------------------------------------------------------
# Motion levels across the library
# ---------------------------------------------------------------------------

def motion_bands(catalogues: dict, ranges: dict) -> tuple:
    """(calm_below, busy_above) — the library's thirds of usable scenes' motion scores."""
    vals = [s["motion_intensity"] for vid, cat in catalogues.items() for s, _, _ in usable_scenes(cat, ranges.get(vid))
            if isinstance(s.get("motion_intensity"), (int, float))]
    if len(vals) < 3:
        return 0.0, float("inf")
    return float(np.percentile(vals, 100 / 3)), float(np.percentile(vals, 200 / 3))


def _motion_shares(us: list, bands: tuple) -> dict:
    out = {"calm": 0.0, "medium": 0.0, "busy": 0.0}
    for s, a, b in us:
        m = s.get("motion_intensity")
        if not isinstance(m, (int, float)):
            continue
        out["calm" if m < bands[0] else "busy" if m > bands[1] else "medium"] += b - a
    return out


# ---------------------------------------------------------------------------
# 👤 Actors
# ---------------------------------------------------------------------------

def actor_stats(catalogues: dict, ranges: dict, details: dict, roles_of, usage: dict = None,
                ratings: dict = None) -> list:
    """Per actor, most footage first. usage: {video_id: recent uses}; ratings: {video_id: [stars of
    rated renders that used it]}. A video with two actors counts for both."""
    usage, ratings = usage or {}, ratings or {}
    bands = motion_bands(catalogues, ranges)
    by_actor = {}
    for vid, cat in catalogues.items():
        names = (details.get(vid) or {}).get("actors") or []
        if not names:
            continue
        us = usable_scenes(cat, ranges.get(vid))
        for name in names:
            a = by_actor.setdefault(name.lower(), {"actor": name, "videos": [], "clips": 0, "seconds": 0.0,
                                                    "motion": {"calm": 0.0, "medium": 0.0, "busy": 0.0},
                                                    "roles": {}, "uses": 0, "ratings": []})
            a["videos"].append(vid)
            a["clips"] += len(us)
            a["seconds"] += sum(b - x for _, x, b in us)
            for k, v in _motion_shares(us, bands).items():
                a["motion"][k] += v
            for s, x, b in us:
                for r in roles_of(s):
                    a["roles"][r] = a["roles"].get(r, 0.0) + (b - x)
            a["uses"] += int(usage.get(vid, 0))
            a["ratings"] += list(ratings.get(vid) or [])
    out = []
    for a in by_actor.values():
        tot = sum(a["motion"].values()) or 1.0
        out.append({
            "actor": a["actor"], "videos": sorted(a["videos"]), "n_videos": len(a["videos"]), "clips": a["clips"],
            "minutes": round(a["seconds"] / 60, 1),
            "calm_pct": round(100 * a["motion"]["calm"] / tot), "medium_pct": round(100 * a["motion"]["medium"] / tot),
            "busy_pct": round(100 * a["motion"]["busy"] / tot),
            "roles": {r: round(v, 1) for r, v in sorted(a["roles"].items())},
            "uses": a["uses"],
            "rating": round(float(np.mean(a["ratings"])), 1) if a["ratings"] else None, "rated": len(a["ratings"]),
        })
    return sorted(out, key=lambda a: -a["minutes"])


def actor_insights(stats: list) -> list:
    """A few plain-English observations about the actor table."""
    if len(stats) < 2:
        return []
    out = []
    total = sum(a["minutes"] for a in stats) or 1.0
    top = stats[0]
    if top["minutes"] / total > 0.35:
        out.append(f"**{top['actor']}** has {100 * top['minutes'] / total:.0f}% of the footage that has actors — "
                   f"recommendations and 🥊 Verses will lean on them.")
    busy = max(stats, key=lambda a: a["busy_pct"])
    calm = max(stats, key=lambda a: a["calm_pct"])
    if busy["busy_pct"] >= 45:
        out.append(f"**{busy['actor']}** is the busiest: {busy['busy_pct']}% of their footage is in the library's "
                   f"busiest third — good for drops and action parts.")
    if calm["actor"] != busy["actor"] and calm["calm_pct"] >= 45:
        out.append(f"**{calm['actor']}** is the calmest: {calm['calm_pct']}% in the calmest third — good for intros, "
                   f"build-ups and slow parts.")
    unused = [a for a in stats if a["uses"] == 0 and a["minutes"] >= 5]
    if unused:
        out.append("Not in any recent project or render: " + ", ".join(f"**{a['actor']}** ({a['minutes']:.0f} min)"
                                                                      for a in unused[:4]) + ".")
    rated = [a for a in stats if a["rated"] >= 3]
    if len(rated) >= 2:
        best = max(rated, key=lambda a: a["rating"])
        out.append(f"Renders with **{best['actor']}** rate highest: {best['rating']:.1f}★ over {best['rated']} rated "
                   f"renders.")
    return out


# ---------------------------------------------------------------------------
# 🎭 Roles: what you have vs what your tracks need
# ---------------------------------------------------------------------------

def role_balance(progress_rows: list, needs: dict, details: dict = None) -> list:
    """Per role used by any track's story map: footage marked with it (and how many videos and
    actors it comes from) vs what the tracks need. needs: {track_id: {role: seconds}}.
    'tracks_worth' = marked seconds / a typical track's need — under 1.5 means the same clips keep
    coming back."""
    details = details or {}
    have, videos, actors = {}, {}, {}
    for r in progress_rows:
        for role, sec in r["role_secs"].items():
            have[role] = have.get(role, 0.0) + sec
            videos.setdefault(role, set()).add(r["video_id"])
            for a in (details.get(r["video_id"]) or {}).get("actors") or []:
                actors.setdefault(role, {}).setdefault(a, 0.0)
                actors[role][a] += sec
    need_by_role = {}
    for _, roles in needs.items():
        for role, sec in roles.items():
            if sec > 0:
                need_by_role.setdefault(role, []).append(sec)
    out = []
    for role in sorted(set(need_by_role) | set(have)):
        needs_r = need_by_role.get(role, [])
        typical = float(np.median(needs_r)) if needs_r else 0.0
        h = have.get(role, 0.0)
        worth = h / typical if typical else None
        out.append({"role": role, "have_sec": round(h, 1), "videos": len(videos.get(role, ())),
                    "by_actor": {a: round(s, 1) for a, s in sorted((actors.get(role) or {}).items(),
                                                                    key=lambda x: -x[1])},
                    "tracks": len(needs_r), "need_sec": round(typical, 1), "need_total": round(sum(needs_r), 1),
                    "tracks_worth": None if worth is None else round(worth, 2),
                    "status": "not needed" if not needs_r else "none" if h <= 0 else "short" if worth < 1.5 else "ok"})
    order = {"none": 0, "short": 1, "ok": 2, "not needed": 3}
    return sorted(out, key=lambda r: (order[r["status"]], r["tracks_worth"] or 0.0))


# Which unmarked videos most likely hold more of a role — the same motion hints quick marking
# gives (calmest fifth of a video → 🌙 Slow, busiest fifth → 💥 Action).
HINT_ROLES = {"slow": "calm", "action": "busy"}


def role_candidates(catalogues: dict, ranges: dict, progress_rows: list, role: str, n: int = 5) -> list:
    """For 🌙 Slow / 💥 Action: unmarked videos with the most footage in their own calmest / busiest
    fifth. [{"video_id", "seconds", "clips"}]"""
    from scene_roles import motion_hint
    if role not in HINT_ROLES:
        return []
    todo = {r["video_id"] for r in progress_rows if r["roles"] == "to do"}
    out = []
    for vid in todo:
        us = [(s, a, b) for s, a, b in usable_scenes(catalogues[vid], ranges.get(vid))
              if isinstance(s.get("motion_intensity"), (int, float))]
        if len(us) < 3:
            continue
        ms = sorted(s["motion_intensity"] for s, _, _ in us)
        picks = []
        for s, a, b in us:
            rank = sum(v <= s["motion_intensity"] for v in ms) / len(ms)      # as the Reviewer ranks it
            if motion_hint(rank) == role:
                picks.append(b - a)
        if picks:
            out.append({"video_id": vid, "seconds": round(sum(picks), 1), "clips": len(picks)})
    return sorted(out, key=lambda x: -x["seconds"])[:n]


# ---------------------------------------------------------------------------
# 📏 Clip lengths vs blocks
# ---------------------------------------------------------------------------

def length_profile(catalogues: dict, ranges: dict, block_lens: list) -> dict:
    """{"bins": [{"label", "clips", "footage_pct", "blocks_pct"}], "median_clip", "median_block",
    "long_share", "short_blocks_share"} — usable clip lengths vs the lengths of the blocks your
    tracks are cut into."""
    clips = [b - a for vid, cat in catalogues.items() for _, a, b in usable_scenes(cat, ranges.get(vid))]
    total = sum(clips) or 1.0
    blocks = [float(x) for x in block_lens if x and x > 0]
    rows = []
    for lo, hi in LEN_BINS:
        inb = lambda x: x >= lo and (hi is None or x < hi)  # noqa: E731
        c = [x for x in clips if inb(x)]
        rows.append({"label": _bin_label(lo, hi), "clips": len(c), "footage_pct": round(100 * sum(c) / total, 1),
                     "blocks_pct": round(100 * sum(1 for x in blocks if inb(x)) / len(blocks), 1) if blocks else 0.0})
    return {"bins": rows, "median_clip": round(float(np.median(clips)), 1) if clips else None,
            "median_block": round(float(np.median(blocks)), 2) if blocks else None,
            "long_share": round(100 * sum(x for x in clips if x >= 30) / total, 1),
            "short_blocks_share": round(100 * sum(1 for x in blocks if x < 3) / len(blocks), 1) if blocks else None,
            "clips": len(clips), "blocks": len(blocks)}


def mixed_motion_clips(catalogues: dict, ranges: dict, curves_of, n: int = 10, min_sec: float = MIXED_MIN_SEC) -> list:
    """Long usable clips whose motion swings most inside them — calm and busy footage in one clip,
    so one role mark (or one exclusion) can't fit all of it: the ones most worth splitting.
    swing = (busiest − calmest second) ÷ the library's typical motion. [{"video_id", "scene_id",
    "start", "seconds", "swing"}]"""
    found, levels = [], []
    for vid, cat in catalogues.items():
        rng = ranges.get(vid)
        curves = None
        for s, a, b in usable_scenes(cat, rng):
            if isinstance(s.get("motion_intensity"), (int, float)):
                levels.append(s["motion_intensity"])
            if b - a < min_sec:
                continue
            curves = curves if curves is not None else (curves_of(vid) or {})
            c = curves.get(s["scene_id"])
            if not c or c.get("values") is None or len(c["values"]) < 2:
                continue
            fps = float(c["fps"])
            t0 = _tc_to_seconds(s["start_tc"])
            vals = np.asarray(c["values"], dtype=float)[max(0, int((a - t0) * fps)):int((b - t0) * fps)]
            w = max(1, int(round(fps)))
            if len(vals) < 2 * w:
                continue
            secs = vals[: len(vals) // w * w].reshape(-1, w).mean(axis=1)
            found.append({"video_id": vid, "scene_id": s["scene_id"], "start": round(a, 1),
                          "seconds": round(b - a, 1), "range": float(np.percentile(secs, 90) - np.percentile(secs, 10))})
    typical = float(np.median(levels)) if levels else 1.0
    for f in found:
        f["swing"] = round(f.pop("range") / max(typical, 1e-6), 2)
    return sorted(found, key=lambda f: -f["swing"])[:n]


def length_insights(profile: dict) -> list:
    out = []
    if profile.get("median_block") and profile.get("median_clip"):
        out.append(f"Half your tracks' blocks are under **{profile['median_block']:g}s**; half your usable clips are "
                   f"over **{profile['median_clip']:g}s**.")
    if profile.get("long_share", 0) >= 25:
        out.append(f"**{profile['long_share']:.0f}%** of your usable footage is in clips of 30s or more. The matcher "
                   f"can take any short window from a long clip, so length alone isn't a problem — but a long clip "
                   f"gets one set of roles and one exclude switch, so the ones whose motion swings (below) are the "
                   f"ones to split.")
    if profile.get("short_blocks_share") is not None and profile["short_blocks_share"] >= 50:
        out.append(f"**{profile['short_blocks_share']:.0f}%** of blocks are under 3s — fast cuts, so the motion within "
                   f"a few seconds is what decides a match.")
    return out


__all__ = ["usable_scenes", "progress", "video_progress", "motion_bands", "actor_stats", "actor_insights",
           "role_balance", "role_candidates", "length_profile", "mixed_motion_clips", "length_insights"]
