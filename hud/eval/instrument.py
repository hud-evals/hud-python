"""Auto-instrumentation for httpx and aiohttp to inject trace headers.

This module patches HTTP clients to automatically add:
- Trace-Id headers when inside an eval context
- Authorization headers for HUD API calls
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from types import SimpleNamespace

from hud.settings import settings

logger = logging.getLogger(__name__)


def _get_trace_headers() -> dict[str, str] | None:
    """Lazy import to avoid circular dependency."""
    from hud.eval.context import get_current_trace_headers

    return get_current_trace_headers()


def _get_api_key() -> str | None:
    """Get API key from context or settings.

    Prefers the contextvar (set by hud.eval(api_key=...)),
    falls back to settings (env var HUD_API_KEY).
    """
    from hud.eval.context import get_current_api_key

    return get_current_api_key() or settings.api_key


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
            if key.lower() not in {k.lower() for k in request.headers}:
                request.headers[key] = value
        logger.debug("Added trace headers to request: %s", url_str)

    # Auto-inject API key if not present or invalid (prefer contextvar, fallback to settings)
    api_key = _get_api_key()
    if api_key:
        existing_auth = request.headers.get("Authorization", "")
        # Override if no auth, empty auth, or invalid "Bearer None"
        if not existing_auth or existing_auth in ("Bearer None", "Bearer null", "Bearer "):
            request.headers["Authorization"] = f"Bearer {api_key}"
            logger.debug("Added API key auth to request: %s", url_str)


async def _async_httpx_request_hook(request: Any) -> None:
    """Async version of the httpx event hook."""
    _httpx_request_hook(request)


def _instrument_httpx_client(client: Any) -> None:
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
        _instrument_httpx_client(self)

    httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[method-assign]

    _original_sync_init = httpx.Client.__init__

    def _patched_sync_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _original_sync_init(self, *args, **kwargs)
        _instrument_httpx_client(self)

    httpx.Client.__init__ = _patched_sync_init  # type: ignore[method-assign]

    logger.debug("httpx auto-instrumentation enabled")


def _patch_aiohttp() -> None:
    """
    Monkey-patch aiohttp to auto-instrument all ClientSession instances.
    This is important for the Gemini client in particular, which uses aiohttp by default.
    """
    try:
        import aiohttp
    except ImportError:
        logger.debug("aiohttp not installed, skipping auto-instrumentation")
        return

    async def on_request_start(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: SimpleNamespace,
        params: aiohttp.TraceRequestStartParams,
    ) -> None:
        """aiohttp trace hook that adds trace headers and auth to HUD requests."""
        url_str = str(params.url)
        if not _is_hud_url(url_str):
            return

        trace_headers = _get_trace_headers()
        if trace_headers is not None:
            for key, value in trace_headers.items():
                if key.lower() not in {k.lower() for k in params.headers}:
                    params.headers[key] = value
            logger.debug("Added trace headers to aiohttp request: %s", url_str)

        api_key = _get_api_key()
        if api_key:
            existing_auth = params.headers.get("Authorization", "")
            # Override if no auth, empty auth, or invalid "Bearer None"
            if not existing_auth or existing_auth in ("Bearer None", "Bearer null", "Bearer "):
                params.headers["Authorization"] = f"Bearer {api_key}"
                logger.debug("Added API key auth to aiohttp request: %s", url_str)

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)

    _original_init = aiohttp.ClientSession.__init__

    def _patched_init(self: aiohttp.ClientSession, *args: Any, **kwargs: Any) -> None:
        existing_traces = kwargs.get("trace_configs") or []
        if trace_config not in existing_traces:
            existing_traces = [*list(existing_traces), trace_config]
        kwargs["trace_configs"] = existing_traces
        _original_init(self, *args, **kwargs)

    aiohttp.ClientSession.__init__ = _patched_init  # type: ignore[method-assign]

    logger.debug("aiohttp auto-instrumentation enabled")


# Auto-patch on module import
_patch_httpx()
_patch_aiohttp()


__all__ = ["_patch_aiohttp", "_patch_httpx"]
