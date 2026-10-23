"""Generic HUD platform API client.

Owns *how* requests reach the platform: base URL, auth, and the shared
retry/error policy from :mod:`hud.utils.requests`. Endpoint paths and wire
payloads live with the feature that owns them (tasksets, builds, registry, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar, overload
from urllib.parse import urlencode

from hud.settings import settings
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.requests import make_request, make_request_sync

T = TypeVar("T")


@dataclass(frozen=True)
class PlatformClient:
    """Sync/async client for the HUD platform API.

    Raises :class:`hud.utils.exceptions.HudRequestError` (with ``status_code``
    and ``response_json``) on HTTP errors and retries transient failures.
    Callers name the reply type the feature expects, and a body that is not one
    raises :class:`hud.utils.exceptions.HudReplyError`; a call that names none
    discards the body.

    ``api_url`` is the bare origin (``https://api.hud.ai``); ``api_prefix``
    (``/v2``) is prepended to every path so feature modules pass version-free
    endpoints (``/tasks/upload``).
    """

    api_url: str
    api_key: str
    api_prefix: str = "/v2"

    @classmethod
    def from_settings(cls) -> PlatformClient:
        api_key = settings.api_key
        if not api_key:
            raise HudAuthenticationError("HUD_API_KEY is required")
        return cls(settings.hud_api_url, api_key)

    @property
    def base_url(self) -> str:
        """Origin + version prefix, e.g. ``https://api.hud.ai/v2``. The base for
        both REST calls here and the build-log WebSocket in the CLI."""
        return f"{self.api_url.rstrip('/')}{self.api_prefix}"

    def url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urlencode(params, doseq=True)
        return url

    def get(self, path: str, reply: type[T], *, params: dict[str, Any] | None = None) -> T:
        return make_request_sync("GET", self.url(path, params), api_key=self.api_key, reply=reply)

    @overload
    def post(self, path: str, reply: type[T], *, json: Any | None = None) -> T: ...
    @overload
    def post(self, path: str, reply: None = None, *, json: Any | None = None) -> None: ...
    def post(self, path: str, reply: type[T] | None = None, *, json: Any | None = None) -> T | None:
        return make_request_sync(
            "POST", self.url(path), json=json, api_key=self.api_key, reply=reply
        )

    def delete(self, path: str) -> None:
        make_request_sync("DELETE", self.url(path), api_key=self.api_key)

    async def aget(self, path: str, reply: type[T], *, params: dict[str, Any] | None = None) -> T:
        return await make_request("GET", self.url(path, params), api_key=self.api_key, reply=reply)

    @overload
    async def apost(self, path: str, reply: type[T], *, json: Any | None = None) -> T: ...
    @overload
    async def apost(self, path: str, reply: None = None, *, json: Any | None = None) -> None: ...
    async def apost(
        self, path: str, reply: type[T] | None = None, *, json: Any | None = None
    ) -> T | None:
        return await make_request(
            "POST", self.url(path), json=json, api_key=self.api_key, reply=reply
        )
