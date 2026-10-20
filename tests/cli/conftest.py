"""The ``hud`` runner the CLI scenarios use, signed in to the fake HUD services."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support import Hud

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import FakeServices, HudEnv

API_KEY = "sk-hud-test"
USER_ID = "11111111-1111-4111-8111-111111111111"
TEAM_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def hud(tmp_path: Path) -> Hud:
    """The ``hud`` CLI at a terminal width that keeps rendered rows on one line."""
    return Hud(tmp_path / "work", columns=96)


@pytest.fixture
def signed_in(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    """The HUD services, with a HUD key whose identity the API reports."""
    hud_env.set(HUD_API_KEY=API_KEY)
    services.route("api", "GET", "/v2/auth/me", json={"user_id": USER_ID, "team_id": TEAM_ID})
    return services
