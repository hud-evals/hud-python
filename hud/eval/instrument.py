"""Auto-instrumentation for httpx to inject trace headers.

This module patches httpx clients to automatically add:
- Trace-Id headers when inside an eval context
- Authorization headers for HUD API calls
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from hud.settings import settings

logger = logging.getLogger(__name__)


def _get_trace_headers() -> dict[str, str] | None:
    """Lazy import to avoid circular dependency."""
    from hud.eval.context import get_current_trace_headers

    return get_current_trace_headers()


def _is_hud_url(url_str: str) -> bool:
    """Check if URL is a HUD service (inference or MCP)."""
    parsed = urlparse(url_str)
    request_host = parsed.netloc or url_str.split("/")[0]

    # Check for known HUD domains (works for any subdomain)
    if request_host.endswith((".hud.ai", ".hud.so")):
        return True

    # Also check settings URLs
    known_hosts = {
        urlparse(settings.hud_gateway_url).netloc,
        urlparse(settings.hud_mcp_url).netloc,
    }
    return request_host in known_hosts


def _httpx_request_hook(request: Any) -> None:
    """httpx event hook that adds trace headers and auth to HUD requests.

    For inference.hud.ai and mcp.hud.ai:
    - Injects trace headers (Trace-Id) if in trace context
    - Injects Authorization header if API key is set and no auth present
    """
    url_str = str(request.url)
    if not _is_hud_url(url_str):
        return

    # Inject trace headers if in trace context
    headers = _get_trace_headers()
    if headers is not None:
        for key, value in headers.items():
            request.headers[key] = value
        logger.debug("Added trace headers to request: %s", url_str)

    # Auto-inject API key if not present
    has_auth = "authorization" in {k.lower() for k in request.headers}
    if not has_auth and settings.api_key:
        request.headers["Authorization"] = f"Bearer {settings.api_key}"
        logger.debug("Added API key auth to request: %s", url_str)


async def _async_httpx_request_hook(request: Any) -> None:
    """Async version of the httpx event hook."""
    _httpx_request_hook(request)


def _instrument_client(client: Any) -> None:
    """Add trace hook to an httpx client instance."""
    is_async = hasattr(client, "aclose")
    hook = _async_httpx_request_hook if is_async else _httpx_request_hook

    existing_hooks = client.event_hooks.get("request", [])
    if hook not in existing_hooks:
        existing_hooks.append(hook)
        client.event_hooks["request"] = existing_hooks


def _patch_httpx() -> None:
    """Monkey-patch httpx to auto-instrument all clients."""
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not installed, skipping auto-instrumentation")
        return

    _original_async_init = httpx.AsyncClient.__init__

    def _patched_async_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _original_async_init(self, *args, **kwargs)
        _instrument_client(self)

    httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[method-assign]

    _original_sync_init = httpx.Client.__init__

    def _patched_sync_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _original_sync_init(self, *args, **kwargs)
        _instrument_client(self)

    httpx.Client.__init__ = _patched_sync_init  # type: ignore[method-assign]

    logger.debug("httpx auto-instrumentation enabled")


# Auto-patch httpx on module import
_patch_httpx()


__all__ = ["_patch_httpx"]
