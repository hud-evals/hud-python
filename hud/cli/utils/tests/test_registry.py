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


def test_from_record_maps_registry_detail_response() -> None:
    env = RegistryEnvironment.from_record(
        {"id": "abc123456", "name": "my-env", "latest_build": {"version": 2}}
    )

    assert env.id == "abc123456"
    assert env.name == "my-env"
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


def test_search_filters_paginated_registry_list(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: dict[str, str] = {}

    def fake_request(method: str, url: str, **kwargs: object) -> dict:
        requested.update(method=method, url=url)
        return {
            "items": [
                {"id": "id-exact", "name": "browser"},
                {"id": "id-sub", "name": "browser-use"},
            ],
            "total": 2,
        }

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    envs = resolve_registry_environments(PlatformClient("https://api.example", "key"), "browser")

    assert requested == {
        "method": "GET",
        "url": "https://api.example/registry?search=browser&limit=5",
    }
    assert [env.id for env in envs] == ["id-exact"]
