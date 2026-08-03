"""Generic platform transport in ``hud.utils.platform``."""

from __future__ import annotations

import pytest

from hud.utils.exceptions import HudAuthenticationError
from hud.utils.platform import PlatformClient


def test_url_prefixes_version_segment_and_joins_params() -> None:
    """Feature modules pass version-free paths; the client prepends the
    canonical ``/v2`` namespace (the default ``api_prefix``)."""
    platform = PlatformClient("https://api.example/", "key")

    assert platform.base_url == "https://api.example/v2"
    assert platform.url("/tasks/upload") == "https://api.example/v2/tasks/upload"
    assert platform.url("/registry/envs", {"limit": 5}) == (
        "https://api.example/v2/registry/envs?limit=5"
    )


def test_get_and_post_route_through_shared_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_request(method: str, url: str, json: object = None, **kwargs: object) -> dict:
        calls.append({"method": method, "url": url, "json": json, "api_key": kwargs.get("api_key")})
        return {"ok": True}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)
    platform = PlatformClient("https://api.example", "key")

    assert platform.get("/x", params={"a": 1}) == {"ok": True}
    assert platform.post("/y", json={"b": 2}) == {"ok": True}
    assert calls == [
        {"method": "GET", "url": "https://api.example/v2/x?a=1", "json": None, "api_key": "key"},
        {"method": "POST", "url": "https://api.example/v2/y", "json": {"b": 2}, "api_key": "key"},
    ]


def test_requests_without_api_key_raise_authentication_error() -> None:
    platform = PlatformClient("https://api.example", "")

    with pytest.raises(HudAuthenticationError):
        platform.get("/tasks")


def test_from_settings_prepends_canonical_version(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "hud_api_url", "https://api.example")
    monkeypatch.setattr(settings_module.settings, "api_key", "key")

    platform = PlatformClient.from_settings()
    assert platform.url("/tasks/upload") == "https://api.example/v2/tasks/upload"
