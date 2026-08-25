"""Content-addressed Daytona snapshot names.

`DaytonaRuntime(name, image=...)` resolves by calling `snapshot.get(name)` and
only builds when that 404s — so editing env.py and reusing the name silently
runs the *old* env in the sandbox. Naming the snapshot after a hash of what goes
into it makes staleness impossible instead of merely remembered.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

INPUTS = ("env.py", "bugs.py", "Dockerfile.hud", "pyproject.toml")


def snapshot_name(prefix: str = "hud-smoke") -> str:
    digest = hashlib.sha256()
    for name in INPUTS:
        digest.update(Path(name).read_bytes())
    return f"{prefix}-{digest.hexdigest()[:8]}"
