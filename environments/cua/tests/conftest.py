"""Test fixtures for the CUA environment grader logic (offline, no desktop)."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
