"""The ``hud`` runner the CLI scenarios use."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support import Hud

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def hud(tmp_path: Path) -> Hud:
    """The ``hud`` fixture at a terminal width that keeps rendered rows on one line."""
    return Hud(tmp_path / "work", columns=96)
