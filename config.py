"""
Local environment configuration — NOT committed to git.

Copy config.example.py to config.py and fill in your paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Google Drive root
# ---------------------------------------------------------------------------
# The drive letter / mount point for your Google Drive (Stream mode).
# On Windows with Drive for Desktop this is typically a mapped drive letter.
# colab_to_local() in render_preview.py should also import DRIVE_ROOT from
# here so both files share a single source of truth.

DRIVE_ROOT = Path(r"G:\My Drive")

# ---------------------------------------------------------------------------
# Scene-labeling data directories
# All are subdirectories under DRIVE_ROOT / "scene-labeling"
# ---------------------------------------------------------------------------

_BASE = DRIVE_ROOT / "scene-labeling"

CATALOGUE_DIR = _BASE / "catalogue"
AUDIO_DIR     = _BASE / "audio_catalogue"
PLANS_DIR     = _BASE / "compilation_plans"
PREVIEW_DIR   = _BASE / "previews"
PROJECTS_DIR  = _BASE / "projects"
