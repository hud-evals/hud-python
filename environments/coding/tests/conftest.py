"""Shared fixtures for the SWE-bench Pro environment tests.

``env.py`` loads its instance from ``INSTANCE_DIR`` at import (in an instance
image that is ``/hud/instance``); the offline tests point it at a small
fixture instance before anything imports it.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("INSTANCE_DIR", str(Path(__file__).parent / "fixtures" / "instance"))
