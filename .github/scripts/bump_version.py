"""
Bump APP_VERSION in page_setup.py — run by .github/workflows/bump-version.yml
on every push to main.

Which part goes up is read from the release commit's message (the last commit
pushed), and only when the message STARTS with the tag:
  "[major] …" → 2.0.0     "[minor] …" → 1.7.0     otherwise → 1.6.1
Prints the new version.

Usage: python bump_version.py            (messages from the COMMIT_MESSAGES env var)
"""

import os
import re
import sys
from pathlib import Path

FILE = Path(__file__).resolve().parents[2] / "page_setup.py"
PATTERN = re.compile(r'^APP_VERSION = "(\d+)\.(\d+)\.(\d+)"', re.M)


def next_version(current: tuple, messages: str) -> tuple:
    major, minor, patch = current
    head = messages.strip().lower()
    if head.startswith("[major]"):
        return major + 1, 0, 0
    if head.startswith("[minor]"):
        return major, minor + 1, 0
    return major, minor, patch + 1


def main() -> None:
    text = FILE.read_text()
    m = PATTERN.search(text)
    if not m:
        sys.exit(f'APP_VERSION = "x.y.z" not found in {FILE.name}')
    new = next_version(tuple(int(x) for x in m.groups()), os.environ.get("COMMIT_MESSAGES", ""))
    version = ".".join(map(str, new))
    FILE.write_text(PATTERN.sub(f'APP_VERSION = "{version}"', text, count=1))
    print(version)


if __name__ == "__main__":
    main()
