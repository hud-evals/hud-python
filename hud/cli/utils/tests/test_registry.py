"""Registry environment lookups for CLI link/deploy flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.utils.registry import (
    RegistryEnvironment,
    get_registry_environment,
    resolve_registry_environments,
)
from hud.utils.exceptions import HudRequestError
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    import pytest


def test_from_record_prefers_display_name() -> None:
    env = RegistryEnvironment.from_record(
        {"id": "abc123456", "name": "raw", "name_display": "Pretty", "latest_version": "2"}
    )

    assert env.id == "abc123456"
    assert env.name == "Pretty"
    assert env.short_id == "abc12345"
    assert env.version_label == " v2"


def test_resolve_accepts_uuid_without_lookup() -> None:
    envs = resolve_registry_environments(
        PlatformClient("https://api.example", "key"),
        "12345678-1234-5678-1234-567812345678",
    )

    assert envs == [
        RegistryEnvironment(
            id="12345678-1234-5678-1234-567812345678",
            name="12345678...",
        )
    ]


def test_get_registry_environment_treats_404_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, **kwargs: object) -> dict:
        raise HudRequestError("not found", status_code=404)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    env = get_registry_environment(PlatformClient("https://api.example", "key"), "abc")

    assert env is None
