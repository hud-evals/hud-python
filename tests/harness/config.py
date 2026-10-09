"""The configuration the SDK under test sees.

The SDK reads its configuration from environment variables once, when
``hud.settings`` is imported. :class:`HudEnv` sets those variables for this
process and for every subprocess it starts, then rebuilds the settings object
from them, so an in-process rollout and a ``hud`` subprocess see the same
values. This is the only place tests touch ``hud.settings``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from hud.settings import Settings, settings
from hud.utils.gateway import list_gateway_models

from .services import SERVICE_ENV

if TYPE_CHECKING:
    import pytest

# Every service the SDK calls points at a closed local port until a test opts into
# the fake services, so nothing a test does can reach a real HUD deployment. The
# web app is only ever printed as a link, so it keeps its default.
CLOSED_PORT = "http://127.0.0.1:9"

ISOLATED = {
    "HUD_TELEMETRY_ENABLED": "0",
    "HUD_CLI_ANALYTICS_ENABLED": "0",
    "HUD_SKIP_VERSION_CHECK": "1",
    **{var: f"{CLOSED_PORT}/{service}" for service, var in SERVICE_ENV.items() if service != "web"},
}

# Provider SDKs read these directly, bypassing hud.settings.
PROVIDER_ENV = (
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


def setting_variables() -> list[str]:
    """Environment variable names the SDK settings read."""
    return [
        field.validation_alias
        for field in Settings.model_fields.values()
        if isinstance(field.validation_alias, str)
    ]


class HudEnv:
    """Isolated SDK configuration for one test: a fresh home and no developer credentials."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
        self._monkeypatch = monkeypatch
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        # The Docker CLI finds its plugins (compose) and current context under its
        # config directory; keep the real one so a real daemon stays reachable.
        monkeypatch.setenv(
            "DOCKER_CONFIG", os.environ.get("DOCKER_CONFIG") or str(Path.home() / ".docker")
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        for name in (*setting_variables(), *PROVIDER_ENV):
            monkeypatch.delenv(name, raising=False)
        for name, value in ISOLATED.items():
            monkeypatch.setenv(name, value)
        self._apply()

    def set(self, **values: str | None) -> None:
        """Set variables (``None`` unsets one) and re-read the SDK settings from them."""
        for name, value in values.items():
            if value is None:
                self._monkeypatch.delenv(name, raising=False)
            else:
                self._monkeypatch.setenv(name, value)
        self._apply()

    def get(self, name: str) -> str | None:
        return os.environ.get(name)

    def _apply(self) -> None:
        fresh = Settings(_env_file=None)
        for name in Settings.model_fields:
            self._monkeypatch.setattr(settings, name, getattr(fresh, name))
        # The gateway model catalog is cached per process; a new configuration
        # may point at a different catalog.
        list_gateway_models.cache_clear()
