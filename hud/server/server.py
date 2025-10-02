"""HUD server helpers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio
from fastmcp.server.server import FastMCP, Transport
from starlette.responses import JSONResponse, Response

from hud.server.low_level import LowLevelServerWithInit

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from starlette.requests import Request

__all__ = ["MCPServer"]

logger = logging.getLogger(__name__)

# Global flag to track if shutdown was triggered by SIGTERM
_sigterm_received = False


def _run_with_sigterm(coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run *coro_fn* via anyio.run() and cancel on SIGTERM or SIGINT (POSIX)."""
    global _sigterm_received

    sys.stderr.flush()

    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        logger.warning(
            "HUD server is running in an existing event loop. "
            "SIGTERM handling may be limited. "
            "Consider using await hub.run_async() instead of hub.run() in async contexts."
        )

        task = loop.create_task(coro_fn(*args, **kwargs))

        # Try to handle SIGTERM if possible
        if sys.platform != "win32":

            def handle_sigterm(signum: Any, frame: Any) -> None:
                logger.info("SIGTERM received in async context, cancelling task...")
                loop.call_soon_threadsafe(task.cancel)

            signal.signal(signal.SIGTERM, handle_sigterm)

        return

    except RuntimeError:
        pass

    async def _runner() -> None:
        stop_evt: asyncio.Event | None = None
        if sys.platform != "win32" and os.getenv("FASTMCP_DISABLE_SIGTERM_HANDLER") != "1":
            loop = asyncio.get_running_loop()
            stop_evt = asyncio.Event()

            # Handle SIGTERM for production shutdown
            def handle_sigterm() -> None:
                global _sigterm_received
                _sigterm_received = True
                logger.info("Received SIGTERM signal, setting shutdown flag")
                stop_evt.set()

            # Handle SIGINT for hot-reload
            def handle_sigint() -> None:
                logger.info("Received SIGINT signal, triggering hot reload...")
                # Don't set _sigterm_received for SIGINT
                stop_evt.set()

            # Handle both SIGTERM and SIGINT for graceful shutdown
            # In Docker containers, we always want to register our handlers
            try:
                loop.add_signal_handler(signal.SIGTERM, handle_sigterm)
                logger.info("SIGTERM handler registered")
            except (ValueError, OSError) as e:
                logger.warning("Could not register SIGTERM handler: %s", e)

            try:
                loop.add_signal_handler(signal.SIGINT, handle_sigint)
                logger.info("SIGINT handler registered")
            except (ValueError, OSError) as e:
                logger.warning("Could not register SIGINT handler: %s", e)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(coro_fn, *args, **kwargs)

                if stop_evt is not None:

                    async def _watch() -> None:
                        logger.info("Signal handler ready, waiting for SIGTERM or SIGINT")
                        if stop_evt is not None:
                            await stop_evt.wait()
                        logger.info("Shutdown signal received, initiating graceful shutdown...")
                        tg.cancel_scope.cancel()

                    tg.start_soon(_watch)
        except* asyncio.CancelledError:
            # This ensures the task group cleans up properly
            logger.info("Task group cancelled, cleaning up...")

    anyio.run(_runner)


class MCPServer(FastMCP):
    """FastMCP wrapper that adds helpful functionality for dockerized environments.
    This works with any MCP client, and adds just a few extra server-side features:
    1. SIGTERM handling for graceful shutdown in container runtimes.
       Note: SIGINT (Ctrl+C) is not handled, allowing normal hot reload behavior.
    2. ``@MCPServer.initialize`` decorator that registers an async initializer
       executed during the MCP *initialize* request. The initializer function receives
       a single ``ctx`` parameter (RequestContext) from which you can access:
       - ``ctx.session``: The MCP ServerSession
       - ``ctx.meta.progressToken``: Token for progress notifications (if provided)
       - ``ctx.session.client_params.clientInfo``: Client information
    3. ``@MCPServer.shutdown`` decorator that registers a coroutine to run during
       server teardown ONLY when SIGTERM is received (not on hot reload/SIGINT).
    4. Enhanced ``add_tool`` that accepts instances of
       :class:`hud.tools.base.BaseTool` which are classes that implement the
       FastMCP ``FunctionTool`` interface.
    """

    def __init__(
        self, name: str | None = None, instructions: str | None = None, **fastmcp_kwargs: Any
    ) -> None:
        # Store shutdown function placeholder before super().__init__
        self._shutdown_fn: Callable | None = None

        # Inject custom lifespan if user did not supply one
        if "lifespan" not in fastmcp_kwargs:

            @asynccontextmanager
            async def _lifespan(_: Any) -> AsyncGenerator[dict[str, Any], None]:
                global _sigterm_received
                try:
                    yield {}
                finally:
                    # Only call shutdown handler if SIGTERM was received
                    logger.info("Lifespan `finally` block reached. Checking for SIGTERM.")
                    # Force flush logs to ensure they're visible
                    sys.stderr.flush()

                    if (
                        self._shutdown_fn is not None
                        and _sigterm_received
                        and not self._shutdown_has_run
                    ):
                        logger.info("SIGTERM detected! Calling @mcp.shutdown handler...")
                        sys.stderr.flush()
                        try:
                            await self._shutdown_fn()
                            logger.info("@mcp.shutdown handler completed successfully.")
                            sys.stderr.flush()
                        except Exception as e:
                            logger.error("Error during @mcp.shutdown: %s", e)
                            sys.stderr.flush()
                        finally:
                            self._shutdown_has_run = True
                            _sigterm_received = False
                    elif self._shutdown_fn is not None:
                        logger.info(
                            "No SIGTERM. This is a hot reload (SIGINT) or normal exit. Skipping @mcp.shutdown handler."  # noqa: E501
                        )
                        sys.stderr.flush()
                    else:
                        logger.info("No shutdown handler registered.")
                        sys.stderr.flush()

            fastmcp_kwargs["lifespan"] = _lifespan

        super().__init__(name=name, instructions=instructions, **fastmcp_kwargs)
        self._initializer_fn: Callable | None = None
        self._did_init = False
        self._replaced_server = False
        self._shutdown_has_run = False  # Guard against double-execution of shutdown hook

    def _replace_with_init_server(self) -> None:
        """Replace the low-level server with init version when needed."""
        if self._replaced_server:
            return

        async def _run_init(ctx: object | None = None) -> None:
            """Run the user initializer exactly once, with stdout redirected."""
            if self._initializer_fn is not None and not self._did_init:
                self._did_init = True
                # Prevent stdout from polluting the MCP protocol on stdio/HTTP
                with contextlib.redirect_stdout(sys.stderr):
                    import inspect

                    fn = self._initializer_fn
                    sig = inspect.signature(fn)
                    params = sig.parameters

                    ctx_param = params.get("ctx") or params.get("_ctx")
                    if ctx_param is not None:
                        if ctx_param.kind == inspect.Parameter.KEYWORD_ONLY:
                            result = fn(**{ctx_param.name: ctx})
                        else:
                            result = fn(ctx)
                    else:
                        required_params = [
                            p
                            for p in params.values()
                            if p.default is inspect._empty
                            and p.kind
                            in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                inspect.Parameter.KEYWORD_ONLY,
                            )
                        ]
                        if required_params:
                            param_list = ", ".join(p.name for p in required_params)
                            raise TypeError(
                                "Initializer must accept no args or a single `ctx` argument; "
                                f"received required parameters: {param_list}"
                            )
                        result = fn()
                    if inspect.isawaitable(result):
                        await result
                    return None
            return None

        # Save the old server's handlers before replacing it
        old_request_handlers = self._mcp_server.request_handlers
        old_notification_handlers = self._mcp_server.notification_handlers

        self._mcp_server = LowLevelServerWithInit(
            name=self.name,
            version=self.version,
            instructions=self.instructions,
            lifespan=self._mcp_server.lifespan,  # reuse the existing lifespan
            init_fn=_run_init,
        )

        # Copy handlers from the old server to the new one
        self._mcp_server.request_handlers = old_request_handlers
        self._mcp_server.notification_handlers = old_notification_handlers
        self._replaced_server = True

    # Initializer decorator: runs on the initialize request
    # The decorated function receives a RequestContext object with access to:
    # - ctx.session: The MCP ServerSession
    # - ctx.meta.progressToken: Progress token (if provided by client)
    # - ctx.session.client_params.clientInfo: Client information
    def initialize(self, fn: Callable | None = None) -> Callable | None:
        def decorator(func: Callable) -> Callable:
            self._initializer_fn = func
            # Only replace server when there's actually an init handler
            self._replace_with_init_server()
            return func

        return decorator(fn) if fn else decorator

    # Shutdown decorator: runs after server stops
    # Supports dockerized SIGTERM handling
    def shutdown(self, fn: Callable | None = None) -> Callable | None:
        """Register a shutdown handler that runs ONLY on SIGTERM.

        This handler will be called when the server receives a SIGTERM signal
        (e.g., during container shutdown). It will NOT be called on:
        - SIGINT (Ctrl+C or hot reload)
        - Normal client disconnects
        - Other graceful shutdowns

        This ensures that persistent resources (like browser sessions) are only
        cleaned up during actual termination, not during development hot reloads.
        """

        def decorator(func: Callable) -> Callable:
            self._shutdown_fn = func
            return func

        return decorator(fn) if fn else decorator

    # Run with SIGTERM handling and custom initialization
    def run(
        self,
        transport: Transport | None = None,
        show_banner: bool = True,
        **transport_kwargs: Any,
    ) -> None:
        if transport is None:
            transport = "stdio"

        async def _bootstrap() -> None:
            await self.run_async(transport=transport, show_banner=show_banner, **transport_kwargs)  # type: ignore[arg-type]

        _run_with_sigterm(_bootstrap)

    async def run_async(
        self,
        transport: Transport | None = None,
        show_banner: bool = True,
        **transport_kwargs: Any,
    ) -> None:
        """Run the server with HUD enhancements."""
        if transport is None:
            transport = "stdio"

        # Register HTTP helpers for HTTP transport
        if transport in ("http", "sse"):
            self._register_hud_helpers()
            logger.info("Registered HUD helper endpoints at /hud/*")

        try:
            await super().run_async(
                transport=transport, show_banner=show_banner, **transport_kwargs
            )
        finally:
            # Fallback: ensure SIGTERM-triggered shutdown runs even when a custom
            # lifespan bypasses our default fastmcp shutdown path.
            global _sigterm_received
            if self._shutdown_fn is not None and _sigterm_received and not self._shutdown_has_run:
                try:
                    await self._shutdown_fn()
                except Exception as e:  # pragma: no cover - defensive logging
                    logger.error("Error during @mcp.shutdown (fallback): %s", e)
                finally:
                    self._shutdown_has_run = True
                    _sigterm_received = False

    # Tool registration helper -- appends BaseTool to FastMCP
    def add_tool(self, obj: Any, **kwargs: Any) -> None:
        from hud.tools.base import BaseTool

        if isinstance(obj, BaseTool):
            super().add_tool(obj.mcp, **kwargs)
            return

        super().add_tool(obj, **kwargs)

    # Override to keep original callables when used as a decorator
    def tool(self, name_or_fn: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        """Register a tool but return the original function in decorator form.

        - Decorator usage (@mcp.tool, @mcp.tool("name"), @mcp.tool(name="name"))
          registers with FastMCP and returns the original function for composition.
        - Call-form (mcp.tool(fn, ...)) behaves the same but returns fn.
        """
        # Accept BaseTool / FastMCP Tool instances or callables in call-form
        if name_or_fn is not None and not isinstance(name_or_fn, str):
            try:
                from hud.tools.base import BaseTool  # lazy import
            except Exception:
                BaseTool = tuple()  # type: ignore[assignment]
            try:
                from fastmcp.tools.tool import Tool as _FastMcpTool
            except Exception:
                _FastMcpTool = tuple()  # type: ignore[assignment]

            # BaseTool instance → add underlying FunctionTool
            if isinstance(name_or_fn, BaseTool):
                super().add_tool(name_or_fn.mcp, **kwargs)
                return name_or_fn
            # FastMCP Tool/FunctionTool instance → add directly
            if isinstance(name_or_fn, _FastMcpTool):
                super().add_tool(name_or_fn, **kwargs)
                return name_or_fn
            # Callable function → register via FastMCP.tool and return original fn
            if callable(name_or_fn):
                super().tool(name_or_fn, **kwargs)
                return name_or_fn

        # Decorator form: get FastMCP's decorator, register, then return original fn
        base_decorator = super().tool(name_or_fn, **kwargs)

        def _wrapper(fn: Any) -> Any:
            base_decorator(fn)
            return fn

        return _wrapper

    def include_router(
        self,
        router: FastMCP,
        prefix: str | None = None,
        hidden: bool = False,
        **kwargs: Any,
    ) -> None:
        """Include a router's tools/resources with optional hidden dispatcher pattern.

        Uses import_server for fast static composition (unlike mount which is slower).

        Args:
            router: FastMCP router to include
            prefix: Optional prefix for tools/resources (ignored if hidden=True)
            hidden: If True, wrap in HiddenRouter (single dispatcher tool that calls sub-tools)
            **kwargs: Additional arguments passed to import_server()

        Examples:
            # Direct include - tools appear at top level
            mcp.include_router(tools_router)

            # Prefixed include - tools get prefix
            mcp.include_router(admin_router, prefix="admin")

            # Hidden include - single dispatcher tool
            mcp.include_router(setup_router, hidden=True)
        """
        if not hidden:
            # Synchronous composition - directly copy tools/resources
            self._sync_import_router(router, hidden=False, prefix=prefix, **kwargs)
            return

        # Hidden pattern: wrap in HiddenRouter before importing
        from .router import HiddenRouter

        # Import the hidden router (synchronous)
        self._sync_import_router(HiddenRouter(router), hidden=True, prefix=prefix, **kwargs)

    def _sync_import_router(
        self,
        router: FastMCP,
        hidden: bool = False,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Synchronously import tools/resources from a router.

        This is a synchronous alternative to import_server for use at module import time.
        """
        import re

        # Import tools directly - use internal dict to preserve keys
        tools = (
            router._tool_manager._tools.items() if not hidden else router._sync_list_tools().items()  # type: ignore
        )
        for key, tool in tools:
            # Validate tool name
            if not re.match(r"^[a-zA-Z0-9_-]{1,128}$", key):
                raise ValueError(
                    f"Tool name '{key}' must match ^[a-zA-Z0-9_-]{{1,128}}$ "
                    "(letters, numbers, underscore, hyphen only, 1-128 chars)"
                )

            new_key = f"{prefix}_{key}" if prefix else key
            self._tool_manager._tools[new_key] = tool

        # Import resources directly
        for key, resource in router._resource_manager._resources.items():
            new_key = f"{prefix}_{key}" if prefix else key
            self._resource_manager._resources[new_key] = resource

        # Import prompts directly
        for key, prompt in router._prompt_manager._prompts.items():
            new_key = f"{prefix}_{key}" if prefix else key
            self._prompt_manager._prompts[new_key] = prompt
        # await self.import_server(hidden_router, prefix=None, **kwargs)

    def _register_hud_helpers(self) -> None:
        """Register development helper endpoints.

        This adds:
        - GET /docs - Interactive documentation and tool testing
        - POST /api/tools/{name} - REST wrappers for MCP tools
        - GET /openapi.json - OpenAPI spec for REST endpoints
        """

        # Register REST wrapper for each tool
        def create_tool_endpoint(key: str) -> Any:
            """Create a REST endpoint for an MCP tool."""

            async def tool_endpoint(request: Request) -> Response:
                """Call MCP tool via REST endpoint."""
                try:
                    data = await request.json()
                except Exception:
                    data = {}

                try:
                    result = await self._tool_manager.call_tool(key, data)

                    # Recursively serialize MCP objects
                    def serialize_obj(obj: Any) -> Any:
                        """Recursively serialize MCP objects to JSON-compatible format."""
                        if obj is None or isinstance(obj, (str, int, float, bool)):
                            return obj
                        if isinstance(obj, (list, tuple)):
                            return [serialize_obj(item) for item in obj]
                        if isinstance(obj, dict):
                            return {k: serialize_obj(v) for k, v in obj.items()}
                        if hasattr(obj, "model_dump"):
                            # Pydantic v2
                            return serialize_obj(obj.model_dump())
                        if hasattr(obj, "dict"):
                            # Pydantic v1
                            return serialize_obj(obj.dict())
                        if hasattr(obj, "__dict__"):
                            # Dataclass or regular class
                            return serialize_obj(obj.__dict__)
                        # Fallback: convert to string
                        return str(obj)

                    serialized = serialize_obj(result)
                    return JSONResponse({"success": True, "result": serialized})
                except Exception as e:
                    return JSONResponse({"success": False, "error": str(e)}, status_code=400)

            return tool_endpoint

        for tool_key in self._tool_manager._tools.keys():  # noqa: SIM118
            endpoint = create_tool_endpoint(tool_key)
            self.custom_route(f"/api/tools/{tool_key}", methods=["POST"])(endpoint)

        @self.custom_route("/openapi.json", methods=["GET"])
        async def openapi_spec(request: Request) -> Response:
            """Generate OpenAPI spec from MCP tools."""
            spec = {
                "openapi": "3.1.0",
                "info": {
                    "title": f"{self.name or 'MCP Server'} - Testing API",
                    "version": "1.0.0",
                    "description": (
                        "REST API wrappers for testing MCP tools. "
                        "These endpoints are for development/testing only. "
                        "Agents should connect via MCP protocol (JSON-RPC over stdio/HTTP)."
                    ),
                },
                "paths": {},
            }

            # Convert each MCP tool to an OpenAPI path
            for tool_key, tool in self._tool_manager._tools.items():
                try:
                    mcp_tool = tool.to_mcp_tool()
                    input_schema = mcp_tool.inputSchema or {"type": "object"}

                    spec["paths"][f"/api/tools/{tool_key}"] = {
                        "post": {
                            "summary": tool_key,
                            "description": mcp_tool.description or "",
                            "operationId": f"call_{tool_key}",
                            "requestBody": {
                                "required": True,
                                "content": {"application/json": {"schema": input_schema}},
                            },
                            "responses": {
                                "200": {
                                    "description": "Success",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "result": {"type": "object"},
                                                },
                                            }
                                        }
                                    },
                                }
                            },
                        }
                    }
                except Exception as e:
                    logger.warning("Failed to generate spec for %s: %s", tool_key, e)

            return JSONResponse(spec)

        @self.custom_route("/docs", methods=["GET"])
        async def docs_page(request: Request) -> Response:
            """Interactive documentation page."""
            import base64
            import json

            base_url = str(request.base_url).rstrip("/")
            tool_count = len(self._tool_manager._tools)
            resource_count = len(self._resource_manager._resources)

            # Generate Cursor deeplink
            server_config = {"url": f"{base_url}/mcp"}
            config_json = json.dumps(server_config, indent=2)
            config_base64 = base64.b64encode(config_json.encode()).decode()
            cursor_deeplink = f"cursor://anysphere.cursor-deeplink/mcp/install?name={self.name or 'mcp-server'}&config={config_base64}"  # noqa: E501

            html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.name or "MCP Server"} - Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        body {{ margin: 0; padding: 0; font-family: monospace; }}
        .header {{ padding: 1.5rem; border-bottom: 1px solid #e0e0e0; background: #fafafa; }}
        .header h1 {{ margin: 0 0 0.5rem 0; font-size: 1.5rem; color: #000; }}
        .header .info {{ margin: 0.25rem 0; color: #666; font-size: 0.9rem; }}
        .header .warning {{ margin: 0.75rem 0 0 0; padding: 0.5rem; background: #fff3cd; border-left: 3px solid #ffc107; color: #856404; font-size: 0.85rem; }}
        .header a {{ color: #000; text-decoration: underline; }}
        .header a:hover {{ color: #666; }}
        .topbar {{ display: none; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.name or "MCP Server"} - Development Tools</h1>
        <div class="info">MCP Endpoint (use this with agents): <a href="{base_url}/mcp">{base_url}/mcp</a></div>
        <div class="info">Tools: {tool_count} | Resources: {resource_count}</div>
        <div class="info">Add to Cursor: <a href="{cursor_deeplink}">Click here to install</a></div>
        <div class="warning">
            ⚠️ The REST API below is for testing only. Agents connect via MCP protocol at <code>{base_url}/mcp</code>
        </div>
    </div>
    
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            SwaggerUIBundle({{
                url: '/openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
                layout: "StandaloneLayout",
                tryItOutEnabled: true
            }})
        }}
    </script>
</body>
</html>
"""  # noqa: E501
            return Response(content=html, media_type="text/html")
