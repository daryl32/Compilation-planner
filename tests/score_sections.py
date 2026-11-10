"""Score the Sections suggester against Daryl's hand-made maps (tests/golden/sections_bar_pass.json).

    python -m tests.score_sections

Per track: the share of the track with the same section type (first pass alone → with the bar-level pass), the share
of his boundaries the suggestion has within one second, and how many parts each map has. Every change to the
suggester's rules is judged on this table — no track may get worse."""
from __future__ import annotations

import json

from tests.support import REPO  # noqa: F401  (installs the test config first)


def rows() -> list:
    import section_types as ST
    fx = json.loads((REPO / "tests" / "golden" / "sections_bar_pass.json").read_text())
    R = lambda rs: [{"start": a, "end": b, "type": t} for a, b, t in rs]   # noqa: E731
    out = []
    for key, d in fx.items():
        coarse, want = R(d["coarse"]), R(d["daryl"])
        got = ST.refine_with(coarse, d["beats"], d["intensity"], d["drops"], d["duration"], d.get("marks") or ())
        out.append({"track": key, "first_pass": round(ST.agreement(coarse, want, d["duration"]), 3),
                    "now": round(ST.agreement(got, want, d["duration"]), 3),
                    "boundaries": round(ST.boundary_match(got, want), 3), "parts": len(got), "his_parts": len(want)})
    return out


if __name__ == "__main__":
    print(f"{'track':<12}{'first pass':>11}{'now':>7}{'lines':>7}{'parts':>7}{'his':>5}")
    for r in rows():
        print(f"{r['track']:<12}{r['first_pass']:>11.0%}{r['now']:>7.0%}{r['boundaries']:>7.0%}{r['parts']:>7}{r['his_parts']:>5}")
