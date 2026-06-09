"""Generic platform transport in ``hud.shared.platform``."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hud.shared.platform import PlatformClient


def test_url_joins_base_path_and_params() -> None:
    platform = PlatformClient("https://api.example/", "key")

    assert platform.url("/tasks/upload") == "https://api.example/tasks/upload"
    assert platform.url("/registry/envs", {"limit": 5}) == (
        "https://api.example/registry/envs?limit=5"
    )


def test_get_and_post_route_through_shared_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_request(method: str, url: str, json: object = None, **kwargs: object) -> dict:
        calls.append({"method": method, "url": url, "json": json, "api_key": kwargs.get("api_key")})
        return {"ok": True}

    monkeypatch.setattr("hud.shared.platform.make_request_sync", fake_request)
    platform = PlatformClient("https://api.example", "key")

    assert platform.get("/x", params={"a": 1}) == {"ok": True}
    assert platform.post("/y", json={"b": 2}) == {"ok": True}
    assert calls == [
        {"method": "GET", "url": "https://api.example/x?a=1", "json": None, "api_key": "key"},
        {"method": "POST", "url": "https://api.example/y", "json": {"b": 2}, "api_key": "key"},
    ]


def test_from_settings_requires_api_key() -> None:
    with patch("hud.settings.settings") as mock_settings:
        mock_settings.api_key = None
        with pytest.raises(ValueError, match="HUD_API_KEY"):
            PlatformClient.from_settings()
