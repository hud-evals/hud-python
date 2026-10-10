"""Fixtures shared by the Harbor docker-lane scenarios."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """This checkout built as a wheel, for images that install the controller."""
    wheels = tmp_path_factory.mktemp("wheels")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return next(wheels.glob("*.whl"))
