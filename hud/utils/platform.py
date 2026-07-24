"""Generic HUD platform API client.

Owns *how* requests reach the platform: base URL, auth, and the shared
retry/error policy from :mod:`hud.utils.requests`. Endpoint paths and wire
payloads live with the feature that owns them (tasksets, builds, registry, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from hud.utils.requests import make_request, make_request_sync


@dataclass(frozen=True)
class PlatformClient:
    """Sync/async client for the HUD platform API.

    Raises :class:`hud.utils.exceptions.HudRequestError` (with ``status_code``
    and ``response_json``) on HTTP errors and retries transient failures.
    Responses are decoded JSON; callers own the payload shape.

    ``api_url`` is the bare origin (``https://api.hud.ai``); ``api_prefix``
    (``/v2``) is prepended to every path so feature modules pass version-free
    endpoints (``/tasks/upload``).
    """

    api_url: str
    api_key: str
    api_prefix: str = "/v2"

    @classmethod
    def from_settings(cls) -> PlatformClient:
        from hud.settings import settings

        return cls(settings.hud_api_url, settings.api_key or "")

    @property
    def base_url(self) -> str:
        """Origin + version prefix, e.g. ``https://api.hud.ai/v2``. The base for
        both REST calls here and the build-log WebSocket in the CLI."""
        return f"{self.api_url.rstrip('/')}{self.api_prefix}"

    def url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urlencode(params)
        return url

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return make_request_sync("GET", self.url(path, params), api_key=self.api_key)

    def post(self, path: str, *, json: Any | None = None) -> Any:
        return make_request_sync("POST", self.url(path), json=json, api_key=self.api_key)

    async def aget(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return await make_request("GET", self.url(path, params), api_key=self.api_key)

    async def apost(self, path: str, *, json: Any | None = None) -> Any:
        return await make_request("POST", self.url(path), json=json, api_key=self.api_key)


def list_items(data: Any) -> list[dict[str, Any]]:
    """Rows from a platform list response — a bare list or an ``{"items": [...]}`` envelope."""
    items = data if isinstance(data, list) else (data.get("items") or [])
    return [item for item in items if isinstance(item, dict)]
