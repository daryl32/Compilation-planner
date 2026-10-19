"""Capture tests/golden/matcher_golden.json from a planner page file
(the code BEFORE the matcher moved to core/):

    python -m tests.capture_matcher_golden path/to/old_1_Compilation_Planner.py
"""
import json
import sys
from pathlib import Path

from tests.golden_cases import matcher_cases
from tests.support import REPO, load_page

GOLDEN = REPO / "tests" / "golden" / "matcher_golden.json"

if __name__ == "__main__":
    res = matcher_cases(load_page(Path(sys.argv[1])))
    GOLDEN.write_text(json.dumps(res, sort_keys=True, separators=(",", ":")))
    print(len(res), "cases →", GOLDEN)
