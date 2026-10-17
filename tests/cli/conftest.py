"""The ``hud`` runner the CLI scenarios use, and CLI analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support import Hud

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, Platform


@pytest.fixture
def hud(tmp_path: Path) -> Hud:
    """The ``hud`` fixture at a terminal width that keeps rendered rows on one line."""
    return Hud(tmp_path / "work", columns=96)


@pytest.fixture
def analytics(platform: Platform, hud_env: HudEnv) -> Platform:
    """CLI analytics on, posting to the platform."""
    hud_env.set(HUD_CLI_ANALYTICS_ENABLED="1", CI=None)
    return platform
