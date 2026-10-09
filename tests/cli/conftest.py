"""The signed-in platform every ``hud`` scenario starts from."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.harness import Hud

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeServices, HudEnv

API_KEY = "sk-hud-test"
USER_ID = "11111111-1111-4111-8111-111111111111"
TEAM_ID = "22222222-2222-4222-8222-222222222222"
WEB_URL = "https://hud.example"


@pytest.fixture
def platform(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    """The fake platform with a signed-in user: a HUD key and ``GET /auth/me``.

    The web app is only printed as links, so it keeps a fixed address.
    """
    hud_env.set(HUD_API_KEY=API_KEY, HUD_WEB_URL=WEB_URL)
    services.route("api", "GET", "/v2/auth/me", json={"user_id": USER_ID, "team_id": TEAM_ID})
    return services


@pytest.fixture
def hud(tmp_path: Path) -> Hud:
    """The ``hud`` fixture at a terminal width whose snapshots fit the line limit."""
    return Hud(tmp_path / "work", columns=96)
