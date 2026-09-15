"""Registry environment lookups for CLI link/deploy flows."""

from __future__ import annotations

from typing import Any

import pytest

from hud.cli.io import CliError
from hud.cli.sync import RegistryEnvironment, get_registry_environment
from hud.utils.exceptions import HudRequestError
from hud.utils.platform import PlatformClient


def test_from_record_maps_registry_detail_response() -> None:
    env = RegistryEnvironment.from_record(
        {"id": "abc123456", "name": "my-env", "latest_build": {"version": 2}}
    )

    assert env.id == "abc123456"
    assert env.name == "my-env"
    assert env.short_id == "abc12345"
    assert env.version_label == " v2"


def test_resolve_verifies_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    def request(method: str, url: str, **kwargs: object) -> dict[str, str]:
        assert url.endswith("/registry/12345678-1234-5678-1234-567812345678")
        return {"id": "12345678-1234-5678-1234-567812345678", "name": "verified"}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    env = get_registry_environment(
        PlatformClient("https://api.example", "key"),
        "12345678-1234-5678-1234-567812345678",
    )

    assert env == RegistryEnvironment(
        id="12345678-1234-5678-1234-567812345678",
        name="verified",
    )


def test_get_registry_environment_treats_404_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, Any]:
        raise HudRequestError("not found", status_code=404)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    with pytest.raises(CliError, match="inaccessible or deleted") as error:
        get_registry_environment(
            PlatformClient("https://api.example", "key"), "12345678-1234-5678-1234-567812345678"
        )
    assert error.value.exit_code == 1


def test_name_resolution_requires_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Names must not trigger substring search")

    monkeypatch.setattr("hud.utils.platform.make_request_sync", unexpected)
    with pytest.raises(ValueError, match="environment ID"):
        get_registry_environment(PlatformClient("https://api.example", "key"), "browser")
