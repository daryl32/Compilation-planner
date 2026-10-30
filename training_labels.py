"""
🎯 Training Labels — the hand-checked "gold set" for the new models (claude/ml-redesign-plan.md).

No Streamlit here: the page (pages/6_Training_Labels.py) passes data in, so everything can be
tested on its own.

Two files, both in CATALOGUE_DIR so they sync with the catalogues:

* tag_vocabulary.csv — the tag list and what each tag means (Daryl edits it, in Excel or on
  the page). Columns:
      tag, group, kind, one_of, teacher, description, not_to_be_confused_with, never_with, always_with,
      search_words
  kind      "frame" (visible in one still) or "motion" (needs a few seconds of movement)
  one_of    a name shared by tags that can't both be true (e.g. "shot size") — empty if none
  teacher   yes / no — may the big model try this tag, or is it learned only from your labels
  never_with  "; "-separated tags that can never be on the same moment as this one, in any group
            (e.g. an action that rules out a position). Works both ways: listing B on A is enough.
  always_with "; "-separated tags that must also be there whenever this one is (one way: A → B
            means A always comes with B, not the other way round). Fixed by ADDING B.
  only_with   "; "-separated tags, at least one of which must be there for this one to be allowed
            (one way: a position only with certain actions). Fixed by DROPPING this tag — the
            other tag is the evidence. Don't list the same tag in always_with and only_with.
  usually_with "; "-separated tags that often (not always) come with this one. Soft: nothing is
            added or dropped — when the teacher gives this tag without them, it's asked to take a
            second look at each one (a pick-one partner: "is it a better fit than the one you gave?").
  threshold   0–100: how sure the teacher must be for the tag to count (empty = 50). Below it, the
            tag goes to unsure. Raise it for tags the teacher over-gives, lower it for ones it misses.
  double_check  yes / no — every time the teacher gives this tag it's asked again "is it clearly
            there?", and a "no" takes it out (for tags it often gives wrongly).
  tell_apart  a short "how to tell it from its look-alikes" line, shown to the teacher with the tag.
  group_order, group_required  per group (the same on every row of a group; set in 🗂️ Category
            order): the order the teacher works through the categories, and whether every moment
            must have a tag from it (then its most confident tag is taken even below threshold).
  search_words  "; "-separated words you'd type when searching for it
  With no vocabulary file the Reviewer's labels.json is used (every tag kind=frame, teacher=yes).

* training_labels.json — what you labelled, one item per card:
      {"id", "type": "tags" | "boundary", "video_id", "scene_id", "t" (seconds, source timeline),
       "tags": [...]            tags cards: every tag that IS there (complete — anything else in
                                the vocabulary is taken as not there)
       "suggested": [...]       what the card was prefilled with (the scene's current tags)
       "verdict": "cut" | "no_cut" | "unsure"   boundary cards
       "source": "detected" | "peak" | "random"  boundary cards: where the point came from
       "skipped": true, "reason"  a card you skipped (bad frame etc.) — never shown again
       "labelled_at", "vocab_version"}

Cards are picked at random, spread across videos (videos with fewer labels come up more) and
leaning toward scenes nobody has reviewed. Boundary cards mix three kinds of point: cuts the
detector made ("is this really a new scene?"), the sharpest picture change inside long scenes
("did the detector miss a cut here?"), and random points (so "no" answers are fair too).
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from config import CATALOGUE_DIR

VOCAB_FILE = CATALOGUE_DIR / "tag_vocabulary.csv"
VOCAB_PENDING = f"cat:{VOCAB_FILE.name}"
LABELS_FILE = CATALOGUE_DIR / "training_labels.json"
LABELS_STEM = LABELS_FILE.stem

VOCAB_COLUMNS = ["tag", "group", "kind", "one_of", "teacher", "threshold", "double_check", "description",
                 "not_to_be_confused_with", "tell_apart", "never_with", "always_with", "only_with", "usually_with",
                 "search_words", "group_order", "group_required"]
GROUP_COLUMNS = ("group_order", "group_required")      # one value per group, set in 🗂️ Category order
DEFAULT_THRESHOLD = 50
KINDS = ("frame", "motion")

LONG_SCENE_SEC = 20.0      # boundary "missed cut?" points come from scenes longer than this
EDGE_SEC = 0.4             # tag-card frames stay this far inside a scene
BOUNDARY_PAD_SEC = 2.0     # boundary clips: this much either side of the point
MOTION_CLIP_SEC = 4.0      # motion-tag clips: this long, centred on the frame
FRAME_CACHE = Path(tempfile.gettempdir()) / "mp_training_frames"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def _yes(v) -> bool:
    return str(v).strip().lower() in ("yes", "y", "true", "1", "x")


def _names(value) -> str:
    """'a; b' from a ;- or ,-separated cell (or a list)."""
    if isinstance(value, (list, tuple, set)):
        value = ";".join(map(str, value))
    return "; ".join(n for n in (x.strip() for x in str(value or "").replace(",", ";").split(";")) if n)


def _row(raw: dict) -> dict | None:
    tag = " ".join(str(raw.get("tag") or "").split())
    if not tag:
        return None
    kind = str(raw.get("kind") or "").strip().lower()
    teacher = raw.get("teacher")
    words = raw.get("search_words") or ""
    if isinstance(words, str):
        words = [w.strip() for w in words.replace(",", ";").split(";")]
    return {
        "tag": tag,
        "group": " ".join(str(raw.get("group") or "").split()),
        "kind": kind if kind in KINDS else "frame",
        "one_of": " ".join(str(raw.get("one_of") or "").split()),
        "teacher": True if teacher in (None, "") else (teacher if isinstance(teacher, bool) else _yes(teacher)),
        "description": str(raw.get("description") or "").strip(),
        "not_to_be_confused_with": str(raw.get("not_to_be_confused_with") or "").strip(),
        "never_with": _names(raw.get("never_with")),
        "always_with": _names(raw.get("always_with")),
        "only_with": _names(raw.get("only_with")),
        "usually_with": _names(raw.get("usually_with")),
        "search_words": [w for w in words if w],
        "threshold": _int_or_none(raw.get("threshold"), 0, 100),
        "double_check": _flag(raw.get("double_check")),
        "tell_apart": str(raw.get("tell_apart") or "").strip(),
        "group_order": _int_or_none(raw.get("group_order"), 0, 999),
        "group_required": _flag(raw.get("group_required")),
    }


def _flag(v) -> bool:
    return v if isinstance(v, bool) else _yes(v or "")


def _int_or_none(v, lo: int, hi: int):
    if v is None or (isinstance(v, float) and v != v):          # empty / NaN from the table
        return None
    try:
        return max(lo, min(hi, int(float(str(v).strip()))))
    except (TypeError, ValueError):
        return None


def parse_vocabulary(text: str) -> list:
    """Rows from tag_vocabulary.csv text (Excel's UTF-8 BOM and ; or , separators are fine).
    Duplicate tags (any case) keep the first row."""
    text = text.lstrip("﻿")
    first = text.splitlines()[0] if text.strip() else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    rows, seen = [], set()
    for raw in csv.DictReader(io.StringIO(text), delimiter=delim):
        raw = {str(k).strip().lower(): v for k, v in raw.items() if k}
        r = _row(raw)
        if r and r["tag"].lower() not in seen:
            seen.add(r["tag"].lower())
            rows.append(r)
    return rows


def _labels_json() -> list:
    try:
        data = json.loads((CATALOGUE_DIR / "labels.json").read_text())
    except (OSError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("labels") or data.get("candidate_labels") or data.get("tags") or []
    out = []
    for item in data if isinstance(data, list) else []:
        name = item if isinstance(item, str) else (item.get("name") or item.get("label") or item.get("tag")
                                                    if isinstance(item, dict) else None)
        if name:
            out.append(str(name))
    return out


def load_vocabulary() -> tuple:
    """(rows, source) — source is "tag_vocabulary.csv" or "labels.json"."""
    try:
        rows = parse_vocabulary(VOCAB_FILE.read_text(encoding="utf-8-sig"))
        if rows:
            return rows, VOCAB_FILE.name
    except (OSError, UnicodeDecodeError, csv.Error):
        pass
    seen, rows = set(), []
    for t in _labels_json():
        if t.lower() not in seen:
            seen.add(t.lower())
            rows.append(_row({"tag": t}))
    return rows, "labels.json"


def vocabulary_version() -> str:
    """Changes whenever the vocabulary file changes — stored with every label."""
    try:
        return str(int(VOCAB_FILE.stat().st_mtime))
    except OSError:
        return "labels.json"


def vocabulary_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=VOCAB_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        r = _row(r)
        if r:
            w.writerow({**r, "teacher": "yes" if r["teacher"] else "no",
                        "double_check": "yes" if r["double_check"] else "",
                        "group_required": "yes" if r["group_required"] else "",
                        "threshold": "" if r["threshold"] is None else r["threshold"],
                        "group_order": "" if r["group_order"] is None else r["group_order"],
                        "search_words": "; ".join(r["search_words"])})
    return buf.getvalue()


def save_vocabulary(rows: list) -> int:
    """Write tag_vocabulary.csv (UTF-8 with BOM, so Excel opens it cleanly) and queue it for
    Drive. Returns the number of tags written."""
    clean = parse_vocabulary(vocabulary_csv(rows))
    _atomic_write(VOCAB_FILE, "﻿" + vocabulary_csv(clean))
    from library_common import mark_pending   # here, so reading never needs Streamlit
    mark_pending(VOCAB_PENDING)
    return len(clean)


def group_settings(rows: list) -> list:
    """[{"group", "order", "required", "tags"}] in the order the teacher answers (the same rule as
    trial_logic.categories): groups with an order first, lowest first, then the rest as they appear.
    A tag with no group is in "other"."""
    cats = {}
    for i, r in enumerate(rows):
        name = r.get("group") or "other"
        c = cats.setdefault(name, {"group": name, "order": None, "required": False, "tags": 0, "first": i})
        c["tags"] += 1
        if r.get("group_order") is not None:
            c["order"] = r["group_order"] if c["order"] is None else min(c["order"], r["group_order"])
        c["required"] = c["required"] or bool(r.get("group_required"))
    out = sorted(cats.values(), key=lambda c: (c["order"] is None, c["order"] or 0, c["first"]))
    return [{k: c[k] for k in ("group", "order", "required", "tags")} for c in out]


def set_group_settings(rows: list, settings: list) -> list:
    """Rows with each group's order / required written onto every tag of that group.
    settings: [{"group", "order", "required"}]; groups not mentioned keep what they have."""
    by = {str(s["group"]): s for s in settings}
    out = []
    for r in rows:
        s = by.get(r.get("group") or "other")
        out.append({**r, "group_order": _int_or_none(s.get("order"), 0, 999), "group_required": bool(s.get("required"))}
                   if s else dict(r))
    return out


def set_thresholds(rows: list, thresholds: dict) -> list:
    """Rows with these tags' thresholds set ({tag: 0–100 or None})."""
    low = {k.lower(): v for k, v in thresholds.items()}
    return [{**r, "threshold": _int_or_none(low[r["tag"].lower()], 0, 100)} if r["tag"].lower() in low else dict(r)
            for r in rows]


def chip_groups(rows: list, extra: list = ()) -> list:
    """How a card shows the tags: [(title, single_choice, [tags])]. Tags sharing a one_of are
    one single-choice row each; the rest are grouped by "group" (multi-choice). extra: tags
    the scene has that aren't in the vocabulary — shown last so they can still be confirmed."""
    exclusive, groups = {}, {}
    for r in rows:
        if r["one_of"]:
            exclusive.setdefault(r["one_of"], []).append(r["tag"])
        else:
            groups.setdefault(r["group"] or "Tags", []).append(r["tag"])
    # in your category order (🗂️), a group's pick-one rows first
    rank = {g["group"]: i for i, g in enumerate(group_settings(rows))}
    first_group = {}
    for r in rows:
        if r["one_of"]:
            first_group.setdefault(r["one_of"], r.get("group") or "other")
    out = [(rank.get(first_group[name], 0), 0, i, (name, True, tags)) for i, (name, tags) in enumerate(exclusive.items())]
    out += [(rank.get("other" if name == "Tags" else name, 0), 1, i, (name, False, tags))
            for i, (name, tags) in enumerate(groups.items())]
    out = [x[-1] for x in sorted(out, key=lambda x: x[:3])]
    known = {r["tag"].lower() for r in rows}
    rest = [t for t in dict.fromkeys(extra) if t.lower() not in known]
    if rest:
        out.append(("Not in the vocabulary", False, rest))
    return out


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_labels() -> list:
    try:
        data = json.loads(LABELS_FILE.read_text())
    except (OSError, ValueError):
        return []
    items = data.get("items") if isinstance(data, dict) else None
    return [i for i in items or [] if isinstance(i, dict) and i.get("id")]


def _write_labels(items: list) -> None:
    _atomic_write(LABELS_FILE, json.dumps({"version": 1, "items": items}, indent=1))
    from library_common import mark_pending
    mark_pending(LABELS_STEM)


def card_id(kind: str, video_id: str, t: float) -> str:
    return f"{kind}:{video_id}:{t:.2f}"


def save_label(card: dict, **answer) -> dict:
    """Store the answer to a card (replacing an earlier answer to the same card).
    answer: tags=[...] / verdict="cut"|"no_cut"|"unsure" / skipped=True, reason="…"."""
    item = {k: card[k] for k in ("id", "type", "video_id", "scene_id", "t") if k in card}
    if card["type"] == "tags":
        item["suggested"] = list(card.get("suggested") or [])
    else:
        item["source"] = card.get("source")
    if answer.get("skipped"):
        item["skipped"] = True
        item["reason"] = answer.get("reason") or "skipped"
    elif card["type"] == "tags":
        item["tags"] = list(dict.fromkeys(answer.get("tags") or []))
    else:
        if answer.get("verdict") not in ("cut", "no_cut", "unsure"):
            raise ValueError("verdict must be cut, no_cut or unsure")
        item["verdict"] = answer["verdict"]
    item["labelled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    item["vocab_version"] = vocabulary_version()
    items = [i for i in load_labels() if i["id"] != item["id"]] + [item]
    _write_labels(items)
    return item


def remove_label(item_id: str) -> bool:
    items = load_labels()
    keep = [i for i in items if i["id"] != item_id]
    if len(keep) == len(items):
        return False
    _write_labels(keep)
    return True


def summary(items: list) -> dict:
    """Counts for the progress line."""
    tags = [i for i in items if i.get("type") == "tags" and not i.get("skipped")]
    bounds = [i for i in items if i.get("type") == "boundary" and not i.get("skipped")]
    per_tag = {}
    for i in tags:
        for t in i.get("tags") or []:
            per_tag[t] = per_tag.get(t, 0) + 1
    return {
        "tags": len(tags),
        "boundaries": len(bounds),
        "cuts": sum(1 for i in bounds if i.get("verdict") == "cut"),
        "skipped": sum(1 for i in items if i.get("skipped")),
        "videos": len({i.get("video_id") for i in items if not i.get("skipped")}),
        "changed": sum(1 for i in tags if set(i.get("tags") or []) != set(i.get("suggested") or [])),
        "per_tag": per_tag,
    }


# ---------------------------------------------------------------------------
# Picking cards
# ---------------------------------------------------------------------------

def _usable_scenes(video: dict, rng_limits) -> list:
    """(scene, start, end) clipped to the library range, excluded scenes left out."""
    out = []
    for s in video.get("scenes") or []:
        if s.get("excluded"):
            continue
        a, b = s["start"], s["end"]
        if rng_limits:
            a, b = max(a, rng_limits[0]), min(b, rng_limits[1])
        if b - a > 2 * EDGE_SEC:
            out.append((s, a, b))
    return out


def _video_weights(vids: list, items: list) -> np.ndarray:
    done = {}
    for i in items:
        done[i.get("video_id")] = done.get(i.get("video_id"), 0) + 1
    return np.array([1.0 / (1 + done.get(v, 0)) for v in vids])


def _near_existing(vid: str, t: float, taken: dict, gap: float) -> bool:
    return any(abs(t - u) < gap for u in taken.get(vid, ()))


def pick_tag_cards(videos: dict, n: int, items: list, ranges: dict = None, *, seed=None,
                   has_media=None) -> list:
    """n random frames to tag. videos: CS.video_index()-style {stem: slim video}.
    has_media(video_id) -> bool: leave out videos with nothing to show."""
    rnd = random.Random(seed)
    ranges = ranges or {}
    pool = {}
    for v in videos.values():
        vid = v["video_id"]
        if has_media and not has_media(vid):
            continue
        sc = _usable_scenes(v, ranges.get(vid))
        if sc:
            pool[vid] = sc
    if not pool:
        return []
    vids = sorted(pool)
    weights = _video_weights(vids, items)
    taken = {}
    for i in items:
        if i.get("type") == "tags" and i.get("t") is not None:
            taken.setdefault(i["video_id"], []).append(i["t"])
    cards = []
    for _ in range(n * 6):
        if len(cards) >= n:
            break
        vid = rnd.choices(vids, weights=weights)[0]
        scenes = pool[vid]
        # Time-uniform, but unreviewed scenes count double (their tags are only the old model's guess).
        sw = [(b - a) * (1.0 if s.get("reviewed") else 2.0) for s, a, b in scenes]
        s, a, b = rnd.choices(scenes, weights=sw)[0]
        t = round(rnd.uniform(a + EDGE_SEC, b - EDGE_SEC), 2)
        if _near_existing(vid, t, taken, 3.0):
            continue
        taken.setdefault(vid, []).append(t)
        weights[vids.index(vid)] *= 0.5        # spread a deck across videos
        cards.append({"id": card_id("tags", vid, t), "type": "tags", "video_id": vid,
                      "scene_id": s["scene_id"], "t": t, "scene_start": s["start"], "scene_end": s["end"],
                      "suggested": list(s.get("tags") or []), "reviewed": bool(s.get("reviewed"))})
    return cards


def sharpest_change(values, fps: float, skip_sec: float = 1.0):
    """Offset (seconds from the curve's start) of the biggest jump in a motion curve,
    ignoring the first/last skip_sec. None if the curve is too short."""
    v = np.asarray(values, dtype=float)
    if fps <= 0 or v.size < 3:
        return None
    lo, hi = int(skip_sec * fps), v.size - int(skip_sec * fps)
    if hi - lo < 1:
        return None
    return float(lo + int(np.argmax(v[lo:hi]))) / fps


def pick_boundary_cards(videos: dict, n: int, items: list, ranges: dict = None, *, seed=None,
                        curves=None, has_media=None) -> list:
    """n points to check for a scene change. curves(video_id) -> {scene_id: {"fps", "values"}}
    (CS.motion_curves); without it, "peak" points fall back to random ones."""
    rnd = random.Random(seed)
    ranges = ranges or {}
    pool = {}
    for v in videos.values():
        vid = v["video_id"]
        if has_media and not has_media(vid):
            continue
        sc = _usable_scenes(v, ranges.get(vid))
        if sc:
            pool[vid] = sc
    if not pool:
        return []
    vids = sorted(pool)
    weights = _video_weights(vids, [i for i in items if i.get("type") == "boundary"])
    taken = {}
    for i in items:
        if i.get("type") == "boundary" and i.get("t") is not None:
            taken.setdefault(i["video_id"], []).append(i["t"])
    sources = ["detected", "peak", "random"]
    cards = []
    for k in range(n * 8):
        if len(cards) >= n:
            break
        vid = rnd.choices(vids, weights=weights)[0]
        scenes = pool[vid]
        source = sources[len(cards) % 3] if k < n * 4 else "random"
        t, scene = None, None
        if source == "detected":
            # A scene's own start (not one clipped by the library range) is a cut the detector made.
            cuts = [(s, a) for s, a, b in scenes if a > 0 and abs(a - s["start"]) < 1e-6]
            if cuts:
                scene, t = rnd.choice(cuts)
        elif source == "peak":
            longs = [(s, a, b) for s, a, b in scenes if b - a >= LONG_SCENE_SEC]
            if longs and curves:
                s, a, b = rnd.choice(longs)
                c = (curves(vid) or {}).get(s["scene_id"])
                if c is not None and len(c["values"]):
                    off = sharpest_change(c["values"], float(c["fps"]))
                    if off is not None and a + 1 < s["start"] + off < b - 1:
                        scene, t = s, s["start"] + off
        if t is None:
            source = "random"
            s, a, b = rnd.choices(scenes, weights=[b - a for _, a, b in scenes])[0]
            if b - a < 2.5:
                continue
            scene, t = s, rnd.uniform(a + 1.0, b - 1.0)
        t = round(t, 2)
        if _near_existing(vid, t, taken, 2.0):
            continue
        taken.setdefault(vid, []).append(t)
        weights[vids.index(vid)] *= 0.5
        cards.append({"id": card_id("boundary", vid, t), "type": "boundary", "video_id": vid,
                      "scene_id": scene["scene_id"], "t": t, "source": source})
    return cards


def build_deck(videos: dict, n: int, mix: str, items: list, ranges: dict = None, *, seed=None,
               curves=None, has_media=None) -> list:
    """A shuffled deck of n cards. mix: "tags", "boundary" or "mixed" (about 70 / 30)."""
    n_b = 0 if mix == "tags" else n if mix == "boundary" else max(1, round(n * 0.3))
    deck = (pick_tag_cards(videos, n - n_b, items, ranges, seed=seed, has_media=has_media)
            + pick_boundary_cards(videos, n_b, items, ranges, seed=None if seed is None else seed + 1,
                                  curves=curves, has_media=has_media))
    random.Random(seed).shuffle(deck)
    return deck


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def extract_frame(video_path: Path, t: float, *, width: int = 854) -> Path | None:
    """A JPEG of the frame at t seconds (cached). None if ffmpeg can't read it."""
    video_path = Path(video_path)
    try:
        stamp = int(video_path.stat().st_mtime)
    except OSError:
        return None
    out = FRAME_CACHE / f"{video_path.stem}_{stamp}_{t:.2f}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(t, 0):.3f}",
           "-i", str(video_path), "-frames:v", "1", "-vf", f"scale='min({width},iw)':-2",
           "-q:v", "3", str(out)]
    try:
        subprocess.run(cmd, check=True, timeout=30, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def clear_frame_cache() -> int:
    n = 0
    for p in FRAME_CACHE.glob("*.jpg"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


# ---------------------------------------------------------------------------
# Editing one tag (📒 Vocabulary's form)
# ---------------------------------------------------------------------------

def upsert_tag(rows: list, row: dict, old_name: str | None = None) -> list:
    """rows with `row` added — or replacing the tag called old_name (a rename) / row["tag"].
    Keeps the order; a new tag goes at the end. Raises ValueError for an empty or clashing name."""
    new = _row(row)
    if not new:
        raise ValueError("Give the tag a name.")
    key = (old_name or new["tag"]).lower()
    if any(r["tag"].lower() == new["tag"].lower() and r["tag"].lower() != key for r in rows):
        raise ValueError(f"There's already a tag called “{new['tag']}”.")
    out, done = [], False
    for r in rows:
        if r["tag"].lower() == key:
            out.append(new)
            done = True
        else:
            out.append(r)
    if not done:
        out.append(new)
    if old_name and old_name != new["tag"]:     # look-alike and never-with lists point at the new name too
        for r in out:
            for col in ("not_to_be_confused_with", "never_with", "always_with", "only_with", "usually_with"):
                r[col] = "; ".join(
                    new["tag"] if p.strip().lower() == old_name.lower() else p.strip()
                    for p in str(r.get(col) or "").split(";") if p.strip())
    return out


def never_pairs(rows: list) -> dict:
    """{tag (lower case): {tags it can never go with}} from the never_with column, both ways, and
    only between tags in the list."""
    names = {r["tag"].lower(): r["tag"] for r in rows}
    out = {}
    for r in rows:
        for other in str(r.get("never_with") or "").split(";"):
            o = names.get(other.strip().lower())
            if o and o.lower() != r["tag"].lower():
                out.setdefault(r["tag"].lower(), set()).add(o)
                out.setdefault(o.lower(), set()).add(r["tag"])
    return out


def always_map(rows: list) -> dict:
    """{tag (lower case): [tags that must come with it]} from always_with — one way, only tags in
    the list."""
    names = {r["tag"].lower(): r["tag"] for r in rows}
    out = {}
    for r in rows:
        for other in str(r.get("always_with") or "").split(";"):
            o = names.get(other.strip().lower())
            if o and o.lower() != r["tag"].lower():
                out.setdefault(r["tag"].lower(), []).append(o)
    return out


def only_map(rows: list) -> dict:
    """{tag (lower case): [tags, at least one of which must be there]} from only_with — one way,
    only tags in the list."""
    names = {r["tag"].lower(): r["tag"] for r in rows}
    out = {}
    for r in rows:
        for other in str(r.get("only_with") or "").split(";"):
            o = names.get(other.strip().lower())
            if o and o.lower() != r["tag"].lower() and o not in out.get(r["tag"].lower(), []):
                out.setdefault(r["tag"].lower(), []).append(o)
    return out


def rule_clashes(rows: list) -> list:
    """[(tag, other)] where `other` is in both this tag's always_with and only_with — the two rules
    would fix a missing `other` in opposite ways (add it / drop this tag)."""
    amap, omap = always_map(rows), only_map(rows)
    return [(r["tag"], o) for r in rows for o in omap.get(r["tag"].lower(), [])
            if o in amap.get(r["tag"].lower(), [])]


def rule_problems(tags, rows: list) -> list:
    """Everything in `tags` your vocabulary's rules don't allow, as readable lines:
    two tags from one pick-one set, a never-with pair, a tag missing its always-with partner,
    a tag without any of its only-with partners."""
    tags = list(dict.fromkeys(tags or []))
    lower = {t.lower(): t for t in tags}
    by_name = {r["tag"].lower(): r for r in rows}
    out = []
    sets = {}
    for t in tags:
        r = by_name.get(t.lower())
        if r and r.get("one_of"):
            sets.setdefault(r["one_of"], []).append(t)
    out += [f"“{'” and “'.join(ts)}” are all in the pick-one set “{name}”" for name, ts in sets.items() if len(ts) > 1]
    out += [f"“{a}” and “{b}” are never-together" for a, b in rule_breaks(tags, rows)]
    for a, needs in always_map(rows).items():
        if a in lower:
            out += [f"“{lower[a]}” should always come with “{n}”" for n in needs if n.lower() not in lower]
    for a, needs in only_map(rows).items():
        if a in lower and not any(n.lower() in lower for n in needs):
            out.append(f"“{lower[a]}” is only allowed with “{'” or “'.join(needs)}”")
    return out


def fix_tags(tags, rows: list) -> tuple:
    """(tags with every missing always-with partner added, the tags added). Only adds — and never a
    tag that would break a pick-one set, a never-with rule or its own only-with rule. Only-with
    problems are never fixed here (is the tag wrong, or was its partner forgotten?). Follows
    chains (A → B → C). Never-with clashes can't be fixed safely (which one is wrong?), so they're
    left for you."""
    tags = list(dict.fromkeys(tags or []))
    amap, pairs, omap = always_map(rows), never_pairs(rows), only_map(rows)
    one_of = {r["tag"].lower(): r.get("one_of") for r in rows}
    added = []
    changed = True
    while changed:
        changed = False
        have = {t.lower() for t in tags}
        for t in list(tags):
            for n in amap.get(t.lower(), []):
                if n.lower() in have:
                    continue
                if any(o.lower() in have for o in pairs.get(n.lower(), ())):
                    continue
                if one_of.get(n.lower()) and any(one_of.get(h) == one_of.get(n.lower()) for h in have):
                    continue
                if omap.get(n.lower()) and not any(o.lower() in have for o in omap[n.lower()]):
                    continue          # it would break its own only-with rule
                tags.append(n)
                added.append(n)
                have.add(n.lower())
                changed = True
    return tags, added


def apply_rules_to_labels(ids, rows: list) -> list:
    """Add missing always-with tags to your saved labels with these ids (one write). Returns
    [(label id, tags added)] for the ones changed."""
    ids = set(ids)
    items = load_labels()
    changed = []
    for it in items:
        if it["id"] not in ids or it.get("type") != "tags" or it.get("skipped"):
            continue
        new, added = fix_tags(it.get("tags") or [], rows)
        if added:
            it["tags"] = new
            it["rules_added"] = sorted(set(it.get("rules_added") or []) | set(added))
            changed.append((it["id"], added))
    if changed:
        _write_labels(items)
    return changed


def rule_breaks(tags, rows: list) -> list:
    """[(a, b)] pairs in `tags` that your never-with rules say can't go together."""
    pairs = never_pairs(rows)
    tags = list(dict.fromkeys(tags or []))
    lower = {t.lower() for t in tags}
    out = []
    for i, a in enumerate(tags):
        for b in pairs.get(a.lower(), ()):
            if b.lower() in lower and tags.index(next(t for t in tags if t.lower() == b.lower())) > i:
                out.append((a, b))
    return out


def delete_tag(rows: list, name: str) -> list:
    return [r for r in rows if r["tag"].lower() != name.lower()]


def rename_in_labels(old: str, new: str) -> int:
    """Rename a tag in your saved answers (training_labels.json). Returns how many changed."""
    items, n = load_labels(), 0
    for i in items:
        for field in ("tags", "suggested"):
            if i.get(field) and any(t.lower() == old.lower() for t in i[field]):
                i[field] = list(dict.fromkeys(new if t.lower() == old.lower() else t for t in i[field]))
                n += field == "tags"
    if n:
        _write_labels(items)
    return n
