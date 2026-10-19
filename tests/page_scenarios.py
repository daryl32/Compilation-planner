"""End-to-end planner scenarios, run through the real Streamlit page headless.

Each scenario picks a track, a cutting method and a matching mode, selects
every video, visits all four steps (checking none of them errors), then
presses 💾 Save plan and records the exported plan — the same JSON the Blender
add-on imports. capture → tests/golden/page_golden.json; the test compares.

    python -m tests.page_scenarios [out.json]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from tests.support import DATA, REPO
from tests.apptest_util import page_app, errors
from tests.fixtures import VIDEOS

PAGE = "pages/1_Compilation_Planner.py"
GOLDEN = REPO / "tests" / "golden" / "page_golden.json"

SECTIONS = ["🎵 0. Audio Settings", "🎭 1. Story map", "📼 2. Videos & Time Ranges", "🎬 3. Matching & Export"]

METHODS = {
    "smart": "Smart (self-tuning)",
    "adaptive": "Adaptive (energy-change + hits)",
    "rhythm": "Rhythm Engine (bar cutting styles)",
    "scored": "Scored cuts (all signals)",
}

SCENARIOS = {
    "full_smart_auto": dict(track="song_full", method="smart", state={}),
    "full_adaptive_auto": dict(track="song_full", method="adaptive", state={}),
    "full_rhythm_auto_split": dict(track="song_full", method="rhythm",
                                   state={"split_screen_enabled": True, "max_clips": 3}),
    "full_scored_seq": dict(track="song_full", method="scored", state={"sequential_mode": True}),
    "full_smart_manual": dict(track="song_full", method="smart", state={"matching_mode": "Manual step-through"}),
    "full_smart_chor": dict(track="song_full", method="smart", state={"matching_mode": "Choreography"}),
    "noanalysis_smart_auto": dict(track="song_noanalysis", method="smart", state={}),
    "noanalysis_adaptive_custom": dict(track="song_noanalysis", method="adaptive",
                                       state={"audio_style": "Custom", "change_threshold": 0.06}),
    "slow_rhythm_auto": dict(track="song_slow", method="rhythm", state={"audio_style": "Calm"}),
    "legacy_adaptive_auto": dict(track="song_legacy", method="adaptive", state={}),
    "short_smart_auto": dict(track="song_short", method="smart", state={}),
}


def _clean_disk():
    for sub in ("projects", "compilation_plans"):
        shutil.rmtree(DATA / sub, ignore_errors=True)
        (DATA / sub).mkdir(parents=True, exist_ok=True)


def run_scenario(conf: dict) -> dict:
    _clean_disk()
    at = page_app(PAGE)
    state = {"track_id": conf["track"], "segmentation_method": METHODS[conf["method"]],
             "committed_selected_videos": list(VIDEOS), **conf["state"]}
    for v in VIDEOS:
        state[f"select_video_{v}"] = True
    problems = {}
    for section in SECTIONS:
        state["ui_section"] = section
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        errs = errors(at)
        if errs:
            problems[section] = errs
    from tests.apptest_util import press_match
    press_match(at)
    problems.update({"match": errors(at)} if errors(at) else {})
    save = [b for b in at.button if b.label == "💾 Save plan"]
    plan = None
    if save:
        save[0].click().run()
        problems.update({"save": errors(at)} if errors(at) else {})
        path = DATA / "compilation_plans" / f"{conf['track']}_plan.json"
        if path.exists():
            plan = json.loads(path.read_text())
    else:
        problems["save"] = ["no 💾 Save plan button"]
    return {"problems": problems, "plan": plan}


def run_all() -> dict:
    from tests.golden_cases import plain
    return plain({name: run_scenario(conf) for name, conf in SCENARIOS.items()})


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    res = run_all()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, separators=(",", ":")))
    for name, r in res.items():
        tl = (r["plan"] or {}).get("timeline") or []
        print(f"{name:28s} blocks={len(tl):3d} clips={sum(b['clip_count'] for b in tl):4d} problems={r['problems']}")


if __name__ == "__main__":
    main()
