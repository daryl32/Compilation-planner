"""Record the golden snapshot from a planner page file.

    python -m tests.capture_golden [path/to/1_Compilation_Planner.py] [out.json]

Run against the code BEFORE a refactor; tests/test_golden.py then checks the
code after it against this file. Re-capture only when a behaviour change is
intended (and say so in the commit)."""

import json
import sys
from pathlib import Path

from tests.support import PLANNER, REPO, load_page
from tests.golden_cases import run_all

GOLDEN = REPO / "tests" / "golden" / "planner_golden.json"


def main():
    page = Path(sys.argv[1]) if len(sys.argv) > 1 else PLANNER
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else GOLDEN
    ns = load_page(page)
    result = run_all(ns)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")))
    errors = [k for k, v in result.items() if isinstance(v, dict) and "error" in v]
    print(f"{len(result)} cases → {out} ({out.stat().st_size // 1024} KB); {len(errors)} recorded errors")
    for k in errors[:20]:
        print("  ", k, result[k]["error"])


if __name__ == "__main__":
    main()
