"""
Media Library — search, filter and sort the video library, and set each video's
library time range (the part of the video the Compilation Planner should use).
"""

import json
from pathlib import Path

import streamlit as st

from config import CATALOGUE_DIR, WHITELISTED_EMAILS
from library_common import (
    scene_tags, tc_to_seconds, format_mmss, overlap_with_range,
    library_ranges, set_library_range, read_pending, push_pending,
    render_range_picker,
)

try:
    from drive_oauth import get_auth_url, exchange_code_for_token, SESSION_KEY as _OAUTH_SESSION_KEY
    _OAUTH_AVAILABLE = True
except ImportError:
    _OAUTH_AVAILABLE = False

PAGE_SIZE = 20

st.set_page_config(page_title="Media Library", layout="wide")

# --- Whitelist gate ---
if st.user.email not in WHITELISTED_EMAILS:
    st.title("Access Denied")
    st.error(f"**{st.user.email}** is not authorised to use this app.")
    if st.button("Sign out"):
        st.logout()
    st.stop()

# --- Drive OAuth callback ---
if _OAUTH_AVAILABLE and _OAUTH_SESSION_KEY not in st.session_state:
    _qp = st.query_params.to_dict()
    if "code" in _qp:
        _token = exchange_code_for_token(_qp["code"])
        if _token:
            st.session_state[_OAUTH_SESSION_KEY] = _token
        st.query_params.clear()
        st.rerun()

st.title("📁 Media Library")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _catalogue_signature() -> tuple:
    """(name, mtime) of every catalogue file — the cache below reloads whenever
    any file is added, removed or edited (e.g. by the Reviewer or a Drive sync)."""
    sig = []
    for f in sorted(CATALOGUE_DIR.glob("*.json")):
        try:
            sig.append((f.name, f.stat().st_mtime))
        except OSError:
            continue
    return tuple(sig)


@st.cache_data(max_entries=2)
def load_library(signature: tuple) -> dict:
    """video_id -> slim summary of its catalogue (no motion curves)."""
    videos = {}
    for name, mtime in signature:
        try:
            cat = json.loads((CATALOGUE_DIR / name).read_text())
        except (ValueError, OSError):
            continue
        if not (isinstance(cat, dict) and "video_id" in cat and "scenes" in cat):
            continue
        scenes = []
        for s in sorted(cat["scenes"], key=lambda s: s["scene_id"]):
            try:
                start, end = tc_to_seconds(s["start_tc"]), tc_to_seconds(s["end_tc"])
            except (KeyError, ValueError):
                continue
            thumbs = s.get("thumbnail_paths") or []
            scenes.append({
                "start": start, "end": end,
                "tags": scene_tags(s),
                "excluded": bool(s.get("excluded")),
                "intro": bool(s.get("intro_candidate")),
                "outro": bool(s.get("outro_candidate")),
                "reviewed": bool(s.get("reviewed")),
                "thumb": Path(thumbs[0]).name if thumbs else None,
            })
        vid = cat["video_id"]
        videos[vid] = {
            "video_id": vid,
            "source_path": cat.get("source_path", ""),
            "duration": scenes[-1]["end"] if scenes else 0.0,
            "scenes": scenes,
            "added": mtime,
        }
    return videos


def trim_intro_outro_range(video: dict) -> tuple:
    """Range that drops leading intro/excluded scenes and trailing outro/excluded
    scenes. The whole video if nothing is flagged at either end."""
    scenes = video["scenes"]
    if not scenes:
        return 0.0, video["duration"]
    first, last = 0, len(scenes) - 1
    while first <= last and (scenes[first]["intro"] or scenes[first]["excluded"]):
        first += 1
    while last >= first and (scenes[last]["outro"] or scenes[last]["excluded"]):
        last -= 1
    if first > last:
        return 0.0, video["duration"]
    return scenes[first]["start"], scenes[last]["end"]


def scene_matches(scene: dict, wanted: set, match_all: bool) -> bool:
    tags = {t.lower() for t in scene["tags"]}
    return wanted <= tags if match_all else bool(wanted & tags)


def video_stats(video: dict, time_range, wanted: set, match_all: bool) -> dict:
    usable = tagged = 0.0
    tag_counts = {}
    first_thumb = None
    for s in video["scenes"]:
        eff_start, eff_end = overlap_with_range(s["start"], s["end"], time_range)
        if eff_start is None or s["excluded"]:
            continue
        secs = eff_end - eff_start
        usable += secs
        if first_thumb is None and s["thumb"]:
            first_thumb = s["thumb"]
        for t in s["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        if wanted and scene_matches(s, wanted, match_all):
            tagged += secs
    n = len(video["scenes"])
    return {
        "usable": usable,
        "tagged": tagged,
        "reviewed_pct": (100.0 * sum(s["reviewed"] for s in video["scenes"]) / n) if n else 0.0,
        "top_tags": sorted(tag_counts.items(), key=lambda kv: -kv[1])[:6],
        "thumb": first_thumb,
    }


library = load_library(_catalogue_signature())
ranges = library_ranges()

# ---------------------------------------------------------------------------
# Sidebar: Drive, filters, sort
# ---------------------------------------------------------------------------
if _OAUTH_AVAILABLE:
    with st.sidebar.expander("☁️ Google Drive", expanded=False):
        if st.session_state.get(_OAUTH_SESSION_KEY):
            st.success("Connected")
            if st.button("Disconnect Drive", key="drive_disconnect"):
                st.session_state.pop(_OAUTH_SESSION_KEY, None)
                st.rerun()
        else:
            st.caption("Connect to save library ranges to your Google Drive.")
            _auth_url = get_auth_url(state="pages/3_Media_Library.py")
            st.markdown(f'<a href="{_auth_url}" target="_self">🔗 Connect Google Drive</a>',
                        unsafe_allow_html=True)

_pending = read_pending()
if _pending:
    st.sidebar.warning(f"{len(_pending)} file(s) have edits not yet saved to Google Drive.")
    if not _OAUTH_AVAILABLE or not st.session_state.get(_OAUTH_SESSION_KEY):
        st.sidebar.caption("Connect Google Drive (above) to save them.")
    elif st.sidebar.button("☁️ Save edits to Drive"):
        _failed = push_pending(st.session_state[_OAUTH_SESSION_KEY])
        if _failed:
            for _name, _err in _failed.items():
                st.sidebar.error(f"{_name}: {_err}")
        else:
            st.sidebar.success("Saved to Google Drive.")
            st.rerun()

st.sidebar.header("Filter")
name_query = st.sidebar.text_input("Search name", placeholder="part of a video name")
all_tags = sorted({t for v in library.values() for s in v["scenes"] for t in s["tags"]}, key=str.lower)
tag_filter = st.sidebar.multiselect("Tags", all_tags)
match_all = st.sidebar.radio(
    "Scenes must have", ["Any of these tags", "All of these tags"], horizontal=True,
    disabled=not tag_filter,
) == "All of these tags"
range_filter = st.sidebar.selectbox("Library range", ["Any", "Has a range", "No range"])
review_filter = st.sidebar.selectbox("Review status", ["Any", "Fully reviewed", "Not fully reviewed"])
max_minutes = max(1, int(max((v["duration"] for v in library.values()), default=60) // 60) + 1)
duration_filter = st.sidebar.slider("Duration (minutes)", 0, max_minutes, (0, max_minutes))

st.sidebar.header("Sort")
SORTS = {
    "Tagged footage": lambda r: r["stats"]["tagged"],
    "Name": lambda r: r["video"]["video_id"].lower(),
    "Duration": lambda r: r["video"]["duration"],
    "Usable duration": lambda r: r["stats"]["usable"],
    "Scene count": lambda r: len(r["video"]["scenes"]),
    "% reviewed": lambda r: r["stats"]["reviewed_pct"],
    "Date added": lambda r: r["video"]["added"],
}
sort_options = list(SORTS) if tag_filter else [k for k in SORTS if k != "Tagged footage"]
sort_by = st.sidebar.selectbox("Sort by", sort_options,
                               help="Tagged footage = seconds of usable footage whose scenes match the tag filter.")
descending = st.sidebar.toggle("Descending", value=(sort_by != "Name"))

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
wanted = {t.lower() for t in tag_filter}
rows = []
for vid, video in library.items():
    if name_query and name_query.lower() not in vid.lower():
        continue
    rng = ranges.get(vid)
    if range_filter == "Has a range" and not rng:
        continue
    if range_filter == "No range" and rng:
        continue
    if not (duration_filter[0] * 60 <= video["duration"] <= duration_filter[1] * 60):
        continue
    stats = video_stats(video, rng, wanted, match_all)
    if review_filter == "Fully reviewed" and stats["reviewed_pct"] < 100:
        continue
    if review_filter == "Not fully reviewed" and stats["reviewed_pct"] >= 100:
        continue
    if wanted and stats["tagged"] <= 0:
        continue
    rows.append({"video": video, "stats": stats, "range": rng})

rows.sort(key=SORTS[sort_by], reverse=descending)

if not library:
    st.info("No video catalogues found. Process a video in Colab, then sync from Google Drive.")
    st.stop()

metric_cols = st.columns(3)
metric_cols[0].metric("Videos shown", f"{len(rows)} / {len(library)}")
metric_cols[1].metric("Usable footage", format_mmss(sum(r["stats"]["usable"] for r in rows)))
if tag_filter:
    metric_cols[2].metric("Tagged footage", format_mmss(sum(r["stats"]["tagged"] for r in rows)))
else:
    metric_cols[2].metric("With library range", sum(1 for r in rows if r["range"]))

if not rows:
    st.info("No videos match these filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Video list
# ---------------------------------------------------------------------------
n_pages = (len(rows) - 1) // PAGE_SIZE + 1
page = st.number_input(f"Page (of {n_pages})", 1, n_pages, 1) if n_pages > 1 else 1
st.session_state.setdefault("lib_open_picker", None)

for row in rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
    video, stats, rng = row["video"], row["stats"], row["range"]
    vid = video["video_id"]

    with st.container(border=True):
        cols = st.columns([2, 6, 1])
        with cols[0]:
            if stats["thumb"]:
                thumb_path = CATALOGUE_DIR / "thumbnails" / vid / stats["thumb"]
                if thumb_path.exists():
                    st.image(str(thumb_path), width=220)
        with cols[1]:
            st.markdown(f"**{vid}**")
            line = (f"⏱ {format_mmss(video['duration'])}  ·  usable {format_mmss(stats['usable'])}"
                    f"  ·  {len(video['scenes'])} scenes  ·  {stats['reviewed_pct']:.0f}% reviewed")
            if tag_filter:
                line += f"  ·  🏷️ {format_mmss(stats['tagged'])} tagged"
            st.caption(line)
            if rng:
                st.caption(f"📚 Library range {format_mmss(rng[0])}–{format_mmss(rng[1])}")
            if stats["top_tags"]:
                st.markdown(" ".join(f"`{t}` {n}" for t, n in stats["top_tags"]))
        with cols[2]:
            if st.button("🎚️", key=f"lib_toggle_{vid}", help="Set the library time range for this video"):
                # One picker open at a time — each embeds a full video player.
                st.session_state["lib_open_picker"] = None if st.session_state["lib_open_picker"] == vid else vid
                st.rerun()

        if st.session_state["lib_open_picker"] == vid:
            st.caption("The Compilation Planner uses this range by default. "
                       "A range set inside the planner overrides it for that project only.")
            action = render_range_picker(
                vid, video["source_path"], video["duration"], rng,
                key="lib",
                suggestions=[("✂️ Trim intro/outro", lambda v=video: trim_intro_outro_range(v))],
                apply_label="Save library range",
                clear_label="Remove library range",
                slider_label=f"Library range for {vid}",
            )
            if action:
                kind, new_rng = action
                full = new_rng is not None and new_rng[0] <= 0 and new_rng[1] >= video["duration"] - 0.5
                set_library_range(vid, None if (kind == "clear" or full) else new_rng)
                st.session_state.pop(f"lib_slider_{vid}", None)
                st.session_state["lib_open_picker"] = None
                st.rerun()
