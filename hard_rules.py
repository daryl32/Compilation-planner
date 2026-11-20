"""💀 Hard's lines by rules (20 Nov 2026) — Hard follows Chill and Medium, so few new clips need recording.

Hard's middle is random every compile (10–14 Slow / Medium / Fast / Extra phases of random length), so lines can't be
hand-placed per turn as in Medium. Daryl's choices (20 Nov):

  anchors   the parts of a Hard round that never move keep Hard's own cues — Wait (the round's start), the final
            Extra, Stop — plus intro / outro
  pools     for every round and phase type, the lines Chill plays there (the Chill pool) and the lines Medium plays
            there (the Medium pool) — worked out by playing Chill and Medium with the same number of rounds
  halves    the first half of each Hard round's middle draws from the Chill pool, the second half from the Medium pool
  repeats   a line at most twice a round
  fill      what's still silent: the automatic callouts / fillers (edge_mode._fill_silences)

The 🧪 Hard lab (Cue Map) tries the rules on any draw without changing anything; ✅ Bake writes them into the cue map
("hard_rules", on), and from then on every Hard compile uses them (edge_mode.resolve_lines).
"""

from __future__ import annotations

import copy
import random

import edge_mode as E

DEFAULTS = {"on": False, "split": 0.5, "first": "chill", "second": "medium", "max_repeat": 2, "per_phase": 1,
            "offset": 0.5, "fallback": True, "autofill": True, "anchors": True}
SKIP_WHY = ("callout", "fills", "🎲")


def rules_of(cm: dict) -> dict:
    return {**DEFAULTS, **((cm or {}).get("hard_rules") or {})}


def _no_auto(cm: dict) -> dict:
    out = dict(cm)
    out["autofill"] = {**(cm.get("autofill") or {}), "hard": False, "medium": False}
    return out


def pools(cm: dict, rounds: int, durations: dict = None) -> dict:
    """{"chill" | "medium": {round: {profile: [line names]}}} — what each plays, round by round (Chill and Medium
    played with the same number of rounds; surprises and automatic fillers left out)."""
    out = {}
    base = _no_auto(cm)
    for src in ("chill", "medium"):
        got = E.resolve_lines(E.schedule(src, rounds, 0), base, src, 0, durations, _raw=True)
        per = {}
        for p in got["plays"]:
            if str(p.get("why") or "").startswith(SKIP_WHY):
                continue
            lst = per.setdefault(p["round"], {}).setdefault(p["profile"], [])
            if p["line"] not in lst:
                lst.append(p["line"])
        out[src] = per
    return out


def middles(sched: list) -> dict:
    """{round: [phase, …]} — the random middle of each Hard round: everything but its Wait, final Extra and Stop."""
    out = {}
    for r in sorted({ph["round"] for ph in sched}):
        phs = [ph for ph in sched if ph["round"] == r and ph.get("kind", "phase") == "phase"]
        if len(phs) < 4:
            out[r] = []
            continue
        out[r] = phs[1:-2]                                       # [Wait] middle… [final Extra] [Stop]
    return out


def resolve(sched: list, cm: dict, seed: int = 0, durations: dict = None, rules: dict = None) -> dict:
    """resolve_lines for Hard by the rules. Adds "hard": {"unfilled": [{round, profile, start, end, why}],
    "split": {round: time}, "pools": the pools}."""
    R = {**DEFAULTS, **(rules or {})}
    durations = durations or {}
    rng = random.Random(seed + 9177)
    n = max((ph["round"] for ph in sched), default=0)
    base = E.resolve_lines(sched, _no_auto(cm), "hard", seed, durations, _raw=True)
    mids = middles(sched)
    in_middle = lambda t, r: any(ph["start"] <= t < ph["end"] for ph in mids.get(r, []))      # noqa: E731
    plays = [dict(p) for p in base["plays"] if R["anchors"] and not in_middle(p["time"], p["round"])]
    for p in plays:
        p["pool"] = "anchor"
    P = pools(cm, n, durations)
    by_name = {ln["name"]: ln for ln in cm.get("lines") or []}

    def secs(name):
        ln = by_name.get(name) or {}
        return float(durations.get(name) or ln.get("play_seconds") or ln.get("seconds") or 2.0)

    total_count = {}
    unfilled, split = [], {}
    for r, ms in mids.items():
        if not ms:
            continue
        a0, b0 = ms[0]["start"], ms[-1]["end"]
        split[r] = round(a0 + (b0 - a0) * float(R["split"]), 3)
        count = {}
        for p in plays:
            if p["round"] == r:
                count[p["line"]] = count.get(p["line"], 0) + 1
        for ph in ms:
            src = R["first"] if ph["start"] < split[r] else R["second"]
            other = R["second"] if src == R["first"] else R["first"]
            cands = [(src, x) for x in P.get(src, {}).get(r, {}).get(ph["profile"], [])]
            if R["fallback"]:
                cands += [(other, x) for x in P.get(other, {}).get(r, {}).get(ph["profile"], []) if (src, x) not in cands]
            placed, why_not = 0, "no line in the pool for this phase"
            for _ in range(int(R["per_phase"])):
                busy = [(p["time"], p["time"] + p["seconds"]) for p in plays]
                spans = E._free_spans(ph["start"] + float(R["offset"]), ph["end"], busy)
                ok = [(s, x) for s, x in cands if count.get(x, 0) < int(R["max_repeat"])]
                if cands and not ok:
                    why_not = f"every pool line already twice this round" if int(R["max_repeat"]) == 2 else "repeat limit reached"
                fit = [(s, x) for s, x in ok if any(b - a >= secs(x) for a, b in spans)]
                if ok and not fit:
                    why_not = "no pool line short enough for the free time"
                if not fit:
                    break
                rng.shuffle(fit)
                fit.sort(key=lambda sx: (sx[0] != src, count.get(sx[1], 0), total_count.get(sx[1], 0)))
                s_, x = fit[0]
                a, _b = next((a, b) for a, b in spans if b - a >= secs(x))
                plays.append({"time": round(a, 3), "seconds": round(secs(x), 3), "line": x, "file": by_name.get(x, {}).get("file"),
                              "profile": ph["profile"], "round": r, "key": None, "moved": 0.0, "pool": s_,
                              "why": f"🧪 {s_.title()} pool · {E.PROFILES[ph['profile']]['label'].title()}"})
                count[x] = count.get(x, 0) + 1
                total_count[x] = total_count.get(x, 0) + 1
                placed += 1
            if not placed:
                unfilled.append({"round": r, "profile": ph["profile"], "start": ph["start"], "end": ph["end"], "why": why_not})
    if R["autofill"]:
        E._fill_silences(sched, cm, "hard", plays, secs_for(by_name, durations), rng, E.total_seconds(sched))
    plays.sort(key=lambda p: p["time"])
    return {"plays": plays, "intro": base["intro"], "outro": base["outro"], "clashes": base["clashes"],
            "unused": base.get("unused", []), "hard": {"unfilled": unfilled, "split": split, "pools": P}}


def secs_for(by_name: dict, durations: dict):
    def secs(line):
        return float(durations.get(line["name"]) or line.get("play_seconds") or line.get("seconds") or 2.0)
    return secs


def check(cm: dict, rules: dict, rounds: int = 5, draws: int = 50) -> dict:
    """The rules over many Hard draws: how full the middles are, the longest silences, which phase types run short
    in which rounds (what's worth recording)."""
    short, longest, filled, total, repeats = {}, [], 0, 0, 0
    for seed in range(draws):
        sch = E.schedule("hard", rounds, seed)
        got = resolve(sch, cm, seed, None, rules)
        for u in got["hard"]["unfilled"]:
            k = ("Last" if u["round"] == rounds else f"R{u['round']}", u["profile"])
            short[k] = short.get(k, 0) + 1
        for r, ms in middles(sch).items():
            total += len(ms)
        filled += sum(len(ms) for ms in middles(sch).values()) - len(got["hard"]["unfilled"])
        t, worst = 0.0, 0.0
        T = E.total_seconds(sch)
        for p in got["plays"] + [{"time": T, "seconds": 0}]:
            ph = E.phase_at(sch, (t + p["time"]) / 2)
            if ph and ph["profile"] not in ("wait", "stop") and p["time"] > t:
                worst = max(worst, p["time"] - t)
            t = max(t, p["time"] + p["seconds"])
        longest.append(worst)
        per = {}
        for p in got["plays"]:
            per[(p["round"], p["line"])] = per.get((p["round"], p["line"]), 0) + 1
        repeats += sum(1 for v in per.values() if v > 1)
    return {"draws": draws, "filled_pct": round(100 * filled / max(total, 1), 1),
            "longest_avg": round(sum(longest) / max(len(longest), 1), 1), "longest_max": round(max(longest or [0]), 1),
            "repeats_per_draw": round(repeats / max(draws, 1), 1),
            "short": sorted(((k[0], k[1], v) for k, v in short.items()), key=lambda x: -x[2])}


def bake(cm: dict, rules: dict) -> dict:
    """The cue map with the rules switched on for Hard (into the unsaved changes; 💾 to keep)."""
    out = copy.deepcopy(cm)
    out["hard_rules"] = {**DEFAULTS, **rules, "on": True}
    return out


def unbake(cm: dict) -> dict:
    out = copy.deepcopy(cm)
    if out.get("hard_rules"):
        out["hard_rules"] = {**out["hard_rules"], "on": False}
    return out
