"""The footage pool ("queues"): every usable scene of every selected video as a
span of still-unused seconds, and the operations that use footage up —
take from the front (consume_span), cut a window out of the middle
(carve_span), skip, split and trim.

Pure functions (no Streamlit). Moved unchanged from the planner page except
that build_footage_queues takes the catalogues (and how to find thumbnails and
roles) as arguments instead of loading them itself."""

from __future__ import annotations


def _tc_to_seconds(tc: str) -> float:
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _overlap_with_range(scene_start: float, scene_end: float, time_range) -> tuple:
    """Intersect a scene's [start, end] with an optional (start, end) restriction.
    Returns (effective_start, effective_end) or (None, None) if no overlap."""
    if not time_range:
        return scene_start, scene_end
    r_start, r_end = time_range
    eff_start, eff_end = max(scene_start, r_start), min(scene_end, r_end)
    if eff_end <= eff_start:
        return None, None
    return eff_start, eff_end


def scene_is_usable(scene: dict, tag_filter: tuple) -> bool:
    """False if the scene should never be used by the planner — either it's
    been manually excluded (e.g. an intro/outro clip) or it doesn't match the
    current tag filter."""
    if scene.get("excluded"):
        return False
    if not tag_filter:
        return True
    wanted = set(t.lower() for t in tag_filter)
    return bool(wanted & set(t.lower() for t in scene.get("tags", [])))


# Leftover footage shorter than this after a trim is discarded rather than
# kept in the pool — not worth tracking a sliver nobody could use anyway.
MIN_LEFTOVER_SEC = 0.5


def build_footage_queues(catalogues: dict, candidate_video_ids: tuple, tag_filter: tuple,
                         video_time_ranges: dict = None, thumbnail_for=None, roles_of=None) -> dict:
    """video_id -> list of mutable 'span' dicts, one per usable scene, sorted
    by start time (chronological order within that video — not scene_id, which
    stops being chronological once the Reviewer splits a scene). Each span tracks how
    much of that scene's footage is still unused — segments consume from the
    front of a span, and any leftover stays available for a later segment to
    pick up. motion_norm is normalized PER VIDEO, matching the previous
    behaviour (shape/rhythm matching within that video's own scenes).

    video_time_ranges: optional {video_id: (start_sec, end_sec)} — restricts
    that video to only the footage overlapping the given range. A scene that
    straddles the boundary is trimmed to the overlapping portion, not
    dropped or included whole.

    catalogues: {video_id: catalogue} (scenes without motion curves are fine).
    thumbnail_for(stored_path, video_id) -> path of a scene's thumbnail (default:
    the stored path as is). roles_of(scene) -> the scene's role ids (default:
    scene_roles.scene_roles)."""
    video_time_ranges = video_time_ranges or {}
    if thumbnail_for is None:
        thumbnail_for = lambda stored, _vid: stored  # noqa: E731
    if roles_of is None:
        from scene_roles import scene_roles as roles_of
    queues = {}

    for video_id, cat in catalogues.items():
        if candidate_video_ids and video_id not in candidate_video_ids:
            continue

        filtered = [s for s in cat["scenes"] if scene_is_usable(s, tag_filter)]
        motions = [s.get("motion_intensity", 0.0) for s in filtered]
        if not motions:
            continue
        m_min, m_max = min(motions), max(motions)
        m_span = (m_max - m_min) or 1.0
        time_range = video_time_ranges.get(video_id)

        spans = []
        for scene in sorted(filtered, key=lambda s: (_tc_to_seconds(s["start_tc"]), s["scene_id"])):
            scene_start_sec = _tc_to_seconds(scene["start_tc"])
            scene_end_sec = _tc_to_seconds(scene["end_tc"])
            eff_start, eff_end = _overlap_with_range(scene_start_sec, scene_end_sec, time_range)
            if eff_start is None:
                continue  # entirely outside the selected range
            duration = eff_end - eff_start
            if duration <= 0:
                continue
            spans.append({
                "video_id": video_id,
                "scene_id": scene["scene_id"],
                "tags": scene.get("tags", []),
                "thumbnail": str(thumbnail_for(scene["thumbnail_paths"][0], video_id))
                             if scene["thumbnail_paths"] else None,
                "motion_norm": round((scene.get("motion_intensity", 0.0) - m_min) / m_span, 3),
                "motion_raw": scene.get("motion_intensity"),   # for Improved auto-fill's motion scale
                "scene_start_sec": scene_start_sec,
                "offset_sec": eff_start - scene_start_sec,  # starts partway in if the range clipped the front
                "remaining_sec": duration,
                "intro_candidate": bool(scene.get("intro_candidate", False)),
                "outro_candidate": bool(scene.get("outro_candidate", False)),
                "roles": roles_of(scene),     # 🎭 scene roles (incl. the two above)
            })
        if spans:
            queues[video_id] = spans

    return queues


def get_candidate_spans(queues: dict, sequential: bool, global_excluded: set = None,
                        min_duration: float = 0.0) -> list[dict]:
    """Currently-pickable spans. In sequential mode, only the earliest
    not-yet-exhausted, non-rejected span of each video is eligible —
    enforcing that a video's own scenes are only ever used in their original
    order. Globally-rejected spans (see global_excluded) are skipped over
    entirely here, not just filtered out afterward — a rejected clip is never
    consumed, so without this it would permanently block a sequential
    queue's front position, effectively banning every scene after it too.

    min_duration: in sequential mode, spans SHORTER than this are also
    skipped over (not just rejected ones) when looking for "the front" —
    used by Advanced Shape Matching so a too-short leftover doesn't lock a
    video out of a block entirely. This only affects which span is offered
    as a SEARCH candidate; nothing is discarded here — a video's shorter
    earlier footage is still there and still gets discarded the normal way
    (via carve_span's sequential rule) only if something later actually gets
    picked. Non-sequential mode and auto mode's fill_slot (min_duration=0.0,
    the default) are unaffected."""
    global_excluded = global_excluded or set()
    candidates = []
    for spans in queues.values():
        if sequential:
            for span in spans:
                key = (span["video_id"], span["scene_id"])
                if span["remaining_sec"] > 0.05 and span["remaining_sec"] >= min_duration and key not in global_excluded:
                    candidates.append(span)
                    break
        else:
            candidates.extend(
                s for s in spans
                if s["remaining_sec"] > 0.05 and (s["video_id"], s["scene_id"]) not in global_excluded
            )
    return candidates


def has_role(c: dict, role) -> bool:
    """Does this span / candidate / window carry the role id?"""
    return bool(role) and role in (c.get("roles") or [])


def role_supply(queues: dict, global_excluded: set = None) -> dict:
    """{role id: seconds of still-available footage carrying it}."""
    out = {}
    for spans in queues.values():
        for sp in spans:
            if (sp["video_id"], sp["scene_id"]) in (global_excluded or set()):
                continue
            for r in sp.get("roles") or []:
                out[r] = out.get(r, 0.0) + sp["remaining_sec"]
    return out


def role_needs(block_roles: list, segments: list) -> dict:
    """{role id: seconds the plan needs for it} (its blocks' total length)."""
    out = {}
    for r, sg in zip(block_roles or [], segments):
        if r:
            out[r] = out.get(r, 0.0) + (sg["end"] - sg["start"])
    return out


def role_scarcity(block_roles: list, segments: list, queues: dict):
    """{block: footage seconds per second needed for its role} — lower is scarcer,
    so Auto fills those blocks first. None without block roles."""
    if not block_roles:
        return None
    supply, needs = role_supply(queues), role_needs(block_roles, segments)
    ratio = {r: supply.get(r, 0.0) / max(n, 1e-6) for r, n in needs.items()}
    return {i: ratio[r] for i, r in enumerate(block_roles) if r}


def consume_span(span: dict, needed_sec: float) -> dict:
    """Trim needed_sec off the front of a span (in place) and return the
    picked sub-clip. If less than needed_sec remains, the whole remainder is
    used — the resulting clip is simply shorter than requested."""
    used_sec = min(needed_sec, span["remaining_sec"])
    pick = {
        "video_id": span["video_id"],
        "scene_id": span["scene_id"],
        "tags": span["tags"],
        "thumbnail": span["thumbnail"],
        "motion_norm": span["motion_norm"],
        "clip_start_sec": round(span["scene_start_sec"] + span["offset_sec"], 3),
        "clip_duration_sec": round(used_sec, 3),
        "offset_into_scene_sec": round(span["offset_sec"], 2),
    }
    span["offset_sec"] += used_sec
    span["remaining_sec"] -= used_sec
    if span["remaining_sec"] < MIN_LEFTOVER_SEC:
        span["remaining_sec"] = 0.0
    return pick


def limit_queue_before(queues: dict, video_id: str, abs_end_sec: float) -> None:
    """Drop every bit of video_id's queued footage at or after abs_end_sec
    (absolute time in the video), trimming a span that crosses it. Used when
    editing a block in sequential mode: a video's clip for the edited block
    has to come before that video's clip in any LATER committed block."""
    kept = []
    for span in queues.get(video_id, []):
        start = span["scene_start_sec"] + span["offset_sec"]
        end = start + span["remaining_sec"]
        if start >= abs_end_sec - 1e-6:
            continue
        if end > abs_end_sec:
            span = {**span, "remaining_sec": abs_end_sec - start}
            if span["remaining_sec"] < MIN_LEFTOVER_SEC:
                continue
        kept.append(span)
    queues[video_id] = kept


def split_span_at(queues: dict, video_id: str, scene_id: int, offset_sec: float) -> None:
    """Split the span of scene_id that contains offset_sec into two at that
    point, so the window grid (get_available_clips_for_video) has a window
    starting exactly there — used so a block being edited can offer its own
    currently-committed clip as an exact dropdown choice."""
    spans = queues.get(video_id, [])
    for i, span in enumerate(spans):
        if span["scene_id"] != scene_id:
            continue
        start, end = span["offset_sec"], span["offset_sec"] + span["remaining_sec"]
        if start + 1e-6 < offset_sec < end - 1e-6:
            spans[i:i + 1] = [{**span, "remaining_sec": offset_sec - start},
                              {**span, "offset_sec": offset_sec, "remaining_sec": end - offset_sec}]
            return


def skip_span(queues: dict, video_id: str, scene_id: int, sequential: bool = False) -> None:
    """Manual escape hatch for sequential mode: discard a video's current
    front span ENTIRELY without using any of it, so the search can reach
    whatever comes after it in that video. Unlike carve_span, this produces
    no pick — the footage is simply given up. Raises ValueError if that
    scene's span no longer exists (already used or already skipped).

    sequential=True also discards every OTHER remaining span in this video
    that starts earlier, not just the exact one skipped. Without this,
    skipping straight to a much later scene_id (jumping over several scenes
    the search never individually touched) left those in-between scenes
    sitting untouched in the queue — free to resurface as a candidate on a
    LATER block, stepping backward past something already used. See
    carve_span's matching fix for the same invariant when a scene is used
    rather than skipped."""
    spans = queues.get(video_id, [])
    if not any(s["scene_id"] == scene_id for s in spans):
        raise ValueError(f"No available span for {video_id} scene {scene_id} to skip — it may already be gone.")
    if sequential:
        # By start time, not scene_id (not chronological after a Reviewer split).
        t0 = next(s["scene_start_sec"] for s in spans if s["scene_id"] == scene_id)
        spans[:] = [s for s in spans if s["scene_start_sec"] > t0]
    else:
        spans[:] = [s for s in spans if s["scene_id"] != scene_id]


SNAP_TOL_SEC = 0.05  # see carve_span


def carve_span(queues: dict, video_id: str, scene_id: int, window_offset_sec: float,
               window_duration_sec: float, sequential: bool = False) -> dict:
    """Generalization of consume_span for Advanced Shape Matching: consumes
    an ARBITRARY window from within a scene's still-available region, not
    just the front. consume_span always eats from span["offset_sec"] forward,
    which is right for auto mode's chain-filling — but a shape-matched window
    can be the best fit ANYWHERE in a scene, including its middle, which can
    leave usable footage on BOTH sides. This finds the span containing the
    window and replaces it in the queue with the remainder(s), returning a
    pick dict in the exact same shape consume_span does, so everything
    downstream (render, export, Blender import) is unaffected by which
    function produced a pick.

    sequential=True (mirrors "keep sequential order"): the LEADING remainder
    — footage before the carved window, WITHIN THIS SAME SCENE — is discarded,
    not kept for later. E.g. using 6.7s-9.7s of a scene means nothing before
    6.7s in that scene is offered to any later block for this video; only
    what comes after 9.7s can still be picked. ANY OTHER remaining scene in
    this video with a smaller scene_id is discarded too — using scene 36
    must also give up scenes 6 through 35, even ones the search never
    individually touched, or a later block could still find one of them and
    step backward past something already used. Non-sequential mode keeps
    everything else, as before.

    Raises ValueError if no current span for this scene covers the requested
    window — this would mean the window was already (partly) consumed by an
    earlier confirmed block since the candidate was generated, which the
    caller should treat as "re-search this block", not silently paper over."""
    spans = queues.get(video_id, [])
    # Set before the loop: when no span of this scene is left at all (an earlier block used the
    # whole scene), the ValueError below still needs them (was an UnboundLocalError, 23 Oct 2026).
    win_start = window_offset_sec
    win_end = window_offset_sec + window_duration_sec

    for i, span in enumerate(spans):
        if span["scene_id"] != scene_id:
            continue
        span_start = span["offset_sec"]
        span_end = span["offset_sec"] + span["remaining_sec"]
        win_start = window_offset_sec
        win_end = window_offset_sec + window_duration_sec
        # Windows come from frame-rounded searches and picks are stored rounded to
        # 0.01 s, so a window can sit a hair outside its span. Up to SNAP_TOL_SEC
        # (about a frame) it's snapped back inside instead of being rejected —
        # rejecting it used to leave Auto-fill blocks empty and silently drop
        # confirmed clips on replay.
        if span_start - SNAP_TOL_SEC <= win_start and win_end <= span_end + SNAP_TOL_SEC:
            win_start = max(win_start, span_start)
            if win_start + window_duration_sec > span_end:
                win_start = max(span_start, span_end - window_duration_sec)
            win_end = min(span_end, win_start + window_duration_sec)
            window_duration_sec = win_end - win_start
            pick = {
                "video_id": span["video_id"],
                "scene_id": span["scene_id"],
                "tags": span["tags"],
                "thumbnail": span["thumbnail"],
                "motion_norm": span["motion_norm"],
                "clip_start_sec": round(span["scene_start_sec"] + win_start, 3),
                "clip_duration_sec": round(window_duration_sec, 3),
                "offset_into_scene_sec": round(win_start, 2),
            }

            replacements = []
            leading = win_start - span_start
            if leading >= MIN_LEFTOVER_SEC and not sequential:
                replacements.append({**span, "offset_sec": span_start, "remaining_sec": leading})
            trailing = span_end - win_end
            if trailing >= MIN_LEFTOVER_SEC:
                replacements.append({**span, "offset_sec": win_end, "remaining_sec": trailing})

            spans[i:i + 1] = replacements
            if sequential:
                # By start time, not scene_id (not chronological after a Reviewer split).
                t0 = span["scene_start_sec"]
                spans[:] = [s for s in spans if s["scene_start_sec"] >= t0]
            return pick

    raise ValueError(
        f"No available span for {video_id} scene {scene_id} covers "
        f"[{win_start:.2f}, {win_end:.2f}] — it may have already been used by an earlier confirmed block."
    )
