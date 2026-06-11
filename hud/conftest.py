"""Root test fixtures: isolate unit tests from developer settings.

Without this, any test that exercises ``Taskset.run`` (or other platform
reporting paths) makes real HTTP calls to the platform whenever the developer
has ``HUD_API_KEY`` configured — spamming the platform with fake jobs/traces
and stalling the suite on network retries. CI never catches it because CI has
no API key.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_hud_settings(request: pytest.FixtureRequest) -> None:
    """Disable telemetry and the API key for unit tests.

    Tests marked ``integration`` keep the real settings (they require
    ``HUD_API_KEY`` and network access by contract).
    """
    if request.node.get_closest_marker("integration") is not None:
        return

    from hud.settings import settings

    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setattr(settings, "telemetry_enabled", False)
    mp.setattr(settings, "api_key", None)
