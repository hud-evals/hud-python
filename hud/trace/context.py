"""TraceContext - Lightweight context for recording agent runs.

TraceContext provides:
- Unique trace identification
- Headers for gateway integration (auto-injected to inference.hud.ai)
- Reward and status reporting to backend
- Tool call delegation

All telemetry goes directly to the backend - nothing accumulated locally.

Auto-instrumentation:
    httpx clients are automatically instrumented when this module is imported.
    Any request to inference.hud.ai will have trace headers injected.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

from hud.settings import settings
from hud.shared import make_request
from hud.telemetry.job import get_current_job

if TYPE_CHECKING:
    from types import TracebackType

    from hud.environment import Environment
    from hud.types import MCPToolResult

logger = logging.getLogger(__name__)

# Contextvar to store current trace headers
_current_trace_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "current_trace_headers", default=None
)


def get_current_trace_headers() -> dict[str, str] | None:
    """Get the current trace headers from context."""
    return _current_trace_headers.get()


# =============================================================================
# Auto-instrumentation for httpx
# =============================================================================

def _httpx_request_hook(request: Any) -> None:
    """httpx event hook that adds trace headers to inference.hud.ai requests."""
    headers = get_current_trace_headers()
    if headers is None:
        return
    
    url_str = str(request.url)
    if "inference.hud.ai" not in url_str:
        return
    
    for key, value in headers.items():
        request.headers[key] = value
    
    logger.debug("Added trace headers to request: %s", url_str)


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


# =============================================================================
# TraceContext
# =============================================================================

class TraceContext:
    """Lightweight context for a traced execution.
    
    Attributes:
        trace_id: Unique identifier for this trace
        name: Task name
        job_id: Links to parent job (auto-detected from hud.job() context)
        group_id: Links parallel traces together (None for single traces)
        variants: Variant assignment dict (for A/B testing)
        reward: Reward value (user-settable)
        error: Exception if failed
        results: All trace results (for parent trace)
        
    Computed:
        headers: Gateway headers
        duration: Execution time in seconds
        success: True if no error
        done: True if completed
    
    Example:
        ```python
        # Simple trace
        async with env.trace("task") as tc:
            await tc.call_tool("navigate", {"url": "..."})
            tc.reward = 0.9
        
        # With variants (A/B testing) and group (multiple runs)
        async with env.trace("task",
            variants={"model": ["gpt-4o", "claude"]},
            group=3,
        ) as tc:
            model = tc.variants["model"]  # Assigned for this run
            response = await call_llm(model=model)
            tc.reward = evaluate(response)
        
        # tc.results has 6 traces (2 variants x 3 runs each)
        # All share the same tc.group_id
        for t in tc.results:
            print(f"{t.variants}: reward={t.reward}")
        ```
    """
    
    def __init__(
        self,
        env: Environment,
        name: str,
        *,
        trace_id: str | None = None,
        api_key: str | None = None,
        job_id: str | None = None,
        _group_id: str | None = None,
        _index: int = 0,
        _variants: dict[str, Any] | None = None,
    ) -> None:
        # Identity
        self.trace_id: str = trace_id or str(uuid.uuid4())
        self.name: str = name
        
        # Job linkage - auto-detect from current job context if not provided
        if job_id is None:
            current_job = get_current_job()
            self.job_id: str | None = current_job.id if current_job else None
        else:
            self.job_id = job_id
        
        self.group_id: str | None = _group_id  # Links parallel traces together
        self.index: int = _index  # Local only, for debugging
        
        # Variant assignment (for A/B testing)
        self.variants: dict[str, Any] = _variants or {}
        
        # User-settable
        self.reward: float | None = None
        
        # Error tracking
        self.error: BaseException | None = None
        
        # Parallel/variant results (nested)
        self.results: list[TraceContext] | None = None
        
        # Private
        self._env = env
        self._api_key = api_key
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._token: contextvars.Token[dict[str, str] | None] | None = None
    
    # =========================================================================
    # Computed Properties
    # =========================================================================
    
    @property
    def headers(self) -> dict[str, str]:
        """Headers for gateway integration."""
        return {"HUD-Trace-Id": self.trace_id}
    
    @property
    def duration(self) -> float:
        """Execution duration in seconds."""
        if self._started_at is None:
            return 0.0
        end = self._completed_at or datetime.now(UTC)
        return (end - self._started_at).total_seconds()
    
    @property
    def success(self) -> bool:
        """True if no error occurred."""
        return self.error is None
    
    @property
    def done(self) -> bool:
        """True if execution completed."""
        return self._completed_at is not None
    
    def _get_api_key(self) -> str | None:
        return self._api_key or settings.api_key
    
    # =========================================================================
    # Tool Operations
    # =========================================================================
    
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call a tool by name (delegates to environment)."""
        return await self._env.call_tool(name, arguments)  # type: ignore[attr-defined]
    
    # =========================================================================
    # Backend Integration
    # =========================================================================
    
    async def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics to the backend."""
        api_key = self._get_api_key()
        if not settings.telemetry_enabled or not api_key:
            return
            
        try:
            await make_request(
                method="POST",
                url=f"{settings.hud_telemetry_url}/traces/{self.trace_id}/log",
                json={"metrics": metrics, "timestamp": datetime.now(UTC).isoformat()},
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("Failed to log metrics: %s", e)
    
    async def _trace_enter(self) -> None:
        """Notify backend that trace has started."""
        api_key = self._get_api_key()
        if not settings.telemetry_enabled or not api_key:
            return
        
        try:
            data: dict[str, Any] = {
                "task_name": self.name,
                "started_at": self._started_at.isoformat() if self._started_at else None,
            }
            if self.job_id:
                data["job_id"] = self.job_id
            if self.group_id:
                data["group_id"] = self.group_id
            if self.variants:
                data["variants"] = self.variants
            
            await make_request(
                method="POST",
                url=f"{settings.hud_telemetry_url}/trace/{self.trace_id}/enter",
                json=data,
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("Failed to send trace enter: %s", e)
    
    async def _trace_exit(self, error_message: str | None = None) -> None:
        """Notify backend that trace has completed."""
        api_key = self._get_api_key()
        if not settings.telemetry_enabled or not api_key:
            return
        
        try:
            data: dict[str, Any] = {
                "task_name": self.name,
                "completed_at": self._completed_at.isoformat() if self._completed_at else None,
                "success": self.success,
            }
            if self.job_id:
                data["job_id"] = self.job_id
            if self.group_id:
                data["group_id"] = self.group_id
            if self.variants:
                data["variants"] = self.variants
            if self.reward is not None:
                data["reward"] = self.reward
            if error_message:
                data["error_message"] = error_message
            
            await make_request(
                method="POST",
                url=f"{settings.hud_telemetry_url}/trace/{self.trace_id}/exit",
                json=data,
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("Failed to send trace exit: %s", e)
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    async def __aenter__(self) -> Self:
        self._started_at = datetime.now(UTC)
        self._token = _current_trace_headers.set(self.headers)
        await self._trace_enter()
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._completed_at = datetime.now(UTC)
        
        if self._token is not None:
            _current_trace_headers.reset(self._token)
            self._token = None
        
        error_msg: str | None = None
        if exc_type is not None:
            self.error = exc_val
            error_msg = str(exc_val) if exc_val else "Unknown error"
        
        # Send exit with all data (reward, error, etc.)
        await self._trace_exit(error_msg)
    
    def __repr__(self) -> str:
        return f"TraceContext({self.trace_id[:8]}..., name={self.name!r}, reward={self.reward})"
