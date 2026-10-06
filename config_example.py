"""
Local environment configuration template.

Copy this file to config.py and update the paths to match your machine.
config.py is gitignored — never commit it.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Google Drive root
# ---------------------------------------------------------------------------
# Windows example:  Path(r"G:\My Drive")
# macOS example:    Path("/Users/yourname/Google Drive/My Drive")

DRIVE_ROOT = Path(r"X:\Your Drive Letter\My Drive")

# ---------------------------------------------------------------------------
# Scene-labeling data directories
# ---------------------------------------------------------------------------
# These are built relative to DRIVE_ROOT. Adjust the subfolder name if yours
# differs (e.g. "scene_labeling" instead of "scene-labeling").

_BASE = DRIVE_ROOT / "scene-labeling"

CATALOGUE_DIR = _BASE / "catalogue"
AUDIO_DIR     = _BASE / "audio_catalogue"
PLANS_DIR     = _BASE / "compilation_plans"
PREVIEW_DIR   = _BASE / "previews"
PROJECTS_DIR  = _BASE / "projects"

# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
WHITELISTED_EMAILS = [
    "your@email.com",
]
