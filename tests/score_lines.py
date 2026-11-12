"""Score the 📐 line finder (section_bounds.py) against the lines of Daryl's hand-made maps of 10 Nov
(tests/golden/sections_layers.json — drums, voice and loudness per bar only; the app also reads hats, bass,
brightness and the novelty).

    python -m tests.score_lines [-v]

His maps are a CHECK here, not the target: they were drawn with ten type names in mind, and some of their lines
(Action → Fast half-way through a loud run, a one-bar Drop) mark no change in the music."""
import json
import sys

from tests.support import REPO  # noqa: F401  (installs the test config first)
import section_bounds as SB
fx = json.loads((REPO/"tests/golden/sections_layers.json").read_text())
for key, d in fx.items():
    L = d["layers"]; lv = {k: L[k] for k in ("drums","vocals","loudness")}
    lines = SB.find_lines(L["bars"], lv, d["duration"])
    want = [r[0] for r in d["daryl"][1:]]
    hit, used = SB.line_match(lines, want)
    print(f"{key:<11} his {len(want):>2} found {len(lines):>2}  his lines found {hit:.0%}  found that are his {used:.0%}")
    if "-v" in sys.argv:
        for ln in lines:
            m = min(want, key=lambda w: abs(w-ln['time']))
            print(f"   {ln['time']:7.1f} L{ln['level']} {ln['size']:.2f} {'✓' if abs(m-ln['time'])<=1 else ' '} {', '.join(ln['why'])}")
        print("   missed:", [w for w in want if not any(abs(w-l['time'])<=1 for l in lines)])
