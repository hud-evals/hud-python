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
    """

    api_url: str
    api_key: str

    @classmethod
    def from_settings(cls) -> PlatformClient:
        from hud.settings import settings

        return cls(settings.hud_api_url, settings.api_key or "")

    def url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.api_url.rstrip('/')}{path}"
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
