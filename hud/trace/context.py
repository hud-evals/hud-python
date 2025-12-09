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

from pydantic import BaseModel

from hud.environment.types import EnvConfig
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
# Payload Models
# =============================================================================


class TracePayload(BaseModel):
    """Base payload for trace enter/exit - sent to both endpoints."""

    task_name: str
    prompt: str | None = None
    code_snippet: str | None = None
    env_config: EnvConfig | None = None
    all_hubs: bool = False  # True if all connectors are from connect_hub
    job_id: str | None = None
    group_id: str | None = None
    variants: dict[str, Any] | None = None


class TraceExitPayload(TracePayload):
    """Exit payload - includes result fields."""

    reward: float | None = None
    success: bool = True
    error_message: str | None = None


# =============================================================================
# Auto-instrumentation for httpx
# =============================================================================


def _is_hud_url(url_str: str) -> bool:
    """Check if URL is a HUD service (inference or MCP)."""
    from urllib.parse import urlparse

    # Extract hostnames from settings URLs
    gateway_host = urlparse(settings.hud_gateway_url).netloc
    mcp_host = urlparse(settings.hud_mcp_url).netloc

    # Parse the request URL and check against known HUD hosts
    parsed = urlparse(url_str)
    request_host = parsed.netloc or url_str.split("/")[0]

    return request_host == gateway_host or request_host == mcp_host


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
    headers = get_current_trace_headers()
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
        prompt: Task prompt (defaults from env.prompt, user-settable)
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
        async with env.trace(
            "task",
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
        _code_snippet: str | None = None,
        _env_config: dict[str, Any] | None = None,
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
        self.prompt: str | None = getattr(env, "prompt", None)  # From env, can override

        # Error tracking
        self.error: BaseException | None = None

        # Parallel/variant results (nested)
        self.results: list[TraceContext] | None = None

        # Code and config (for reproducibility)
        self.code_snippet: str | None = _code_snippet
        self.env_config: dict[str, Any] | None = _env_config

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
        return {"Trace-Id": self.trace_id}

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

    def _build_base_payload(self) -> TracePayload:
        """Build the base payload for enter/exit."""
        # Check if all connectors are from hubs (fully reproducible)
        all_hubs = getattr(self._env, "_all_hubs", False)

        # Convert env_config dict to EnvConfig model
        env_config_model: EnvConfig | None = None
        if self.env_config:
            env_config_model = EnvConfig(**self.env_config)

        return TracePayload(
            task_name=self.name,
            prompt=self.prompt,
            code_snippet=self.code_snippet,
            env_config=env_config_model,
            all_hubs=all_hubs,
            job_id=self.job_id,
            group_id=self.group_id,
            variants=self.variants if self.variants else None,
        )

    # =========================================================================
    # Tool Operations
    # =========================================================================

    async def call_tool(
        self,
        call: Any,
        /,
        **kwargs: Any,
    ) -> Any:
        """Call a tool (delegates to environment).
        
        Accepts any format:
            - String with kwargs: call_tool("navigate", url="...")
            - OpenAI tool_call: call_tool(response.choices[0].message.tool_calls[0])
            - Claude tool_use: call_tool(block)  # where block.type == "tool_use"
            - Gemini function_call: call_tool(part)
        """
        return await self._env.call_tool(call, **kwargs)  # type: ignore[attr-defined]

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
                json={"metrics": metrics},
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
            payload = self._build_base_payload()
            await make_request(
                method="POST",
                url=f"{settings.hud_api_url}/trace/{self.trace_id}/enter",
                json=payload.model_dump(exclude_none=True),
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("Failed to send trace enter: %s", e)

    async def _trace_exit(self, error_message: str | None = None) -> None:
        """Notify backend that trace has completed."""
        api_key = self._get_api_key()
        if not settings.telemetry_enabled or not api_key:
            return

        # Use evaluate tool reward if not manually set
        reward = self.reward
        if reward is None:
            reward = getattr(self._env, "_evaluate_reward", None)

        try:
            payload = TraceExitPayload(
                **self._build_base_payload().model_dump(),
                reward=reward,
                success=self.success,
                error_message=error_message,
            )
            await make_request(
                method="POST",
                url=f"{settings.hud_api_url}/trace/{self.trace_id}/exit",
                json=payload.model_dump(exclude_none=True),
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
        self._print_trace_link()
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

    def _print_trace_link(self) -> None:
        """Print a nicely formatted trace link to console and open in browser."""
        import contextlib
        import webbrowser

        trace_url = f"https://hud.ai/trace/{self.trace_id}"

        # Try to open in browser (new tab if possible)
        with contextlib.suppress(Exception):
            webbrowser.open(trace_url, new=2)

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.align import Align

            console = Console()
            
            # Style: HUD colors - gold border, purple link
            link_markup = f"[bold underline rgb(108,113,196)][link={trace_url}]{trace_url}[/link][/bold underline rgb(108,113,196)]"
            
            content = Align.center(link_markup)
            
            panel = Panel(
                content,
                title="🔗 Trace Started",
                border_style="rgb(192,150,12)",  # HUD gold
                padding=(0, 2),
            )
            console.print(panel)
        except ImportError:
            # Fallback if rich not available
            print(f"Trace: https://hud.ai/trace/{self.trace_id}")
