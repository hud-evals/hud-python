"""v5 env-authoring compatibility, adapted onto the v6 :class:`Environment`.

Deployed v5 envs are written against the old MCP-server ``Env``: positional
``name``, ``@env.scenario(...)`` (with ``chat``/``returns``/tool exclusions),
``@env.tool`` / ``env.add_tool``, a callable ``env("scenario")`` factory, and
``env.run(transport=...)``. v6's ``Environment`` is a different abstraction (a
JSON-RPC control channel of capabilities + tasks), so this mixin re-exposes that
surface and *adapts* it to v6:

- scenarios register as v6 tasks (via :func:`scenario_to_task_fn`), keeping the
  v5 metadata (chat flag, returns type, tool exclusions) for agents/manifest;
- ``env(name)`` returns the registered ``Task`` (a callable variant factory);
- ``env.run(...)`` serves the v6 control channel;
- registered tools are classified and, on serve, turned into capabilities:
  shell/edit → ``ssh`` (spins up a :class:`~hud.environment.Workspace`), computer
  → ``rfb`` (detects a VNC / ``HUD_RFB_URL``), everything else → ``mcp`` (a local
  :class:`~hud.server.MCPServer`). Each path is best-effort: a failure warns and
  is skipped so the env's *tasks* still serve.

Every entry point emits a ``DeprecationWarning`` pointing at the v6 equivalent.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import socket
import warnings
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, cast

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from .task import Task
    from .workspace import Workspace

LOGGER = logging.getLogger("hud.environment.legacy")

P = ParamSpec("P")

ToolKind = Literal["shell", "computer", "mcp"]

_SHELL_NAMES = {"bash", "shell", "edit", "apply_patch", "applypatch", "str_replace"}
_SHELL_CLASSES = {"bashtool", "shelltool", "edittool", "applypatchtool", "claudebashsession"}


def _free_port() -> int:
    """Pick an available loopback TCP port (best-effort; small TOCTOU window)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _classify_tool(tool: Any) -> ToolKind:
    """Bucket a registered tool into the capability it should become.

    Honors an explicit ``_legacy_capability_kind`` marker (set by the ``hud.tools``
    shim for removed computer tools), else infers from the tool's name/class.
    """
    marker = getattr(tool, "_legacy_capability_kind", None)
    if marker in ("shell", "computer", "mcp"):
        return cast("ToolKind", marker)
    name = str(getattr(tool, "name", "") or "").lower()
    cls = type(tool).__name__.lower()
    if "computer" in name or "computer" in cls:
        return "computer"
    if name in _SHELL_NAMES or cls in _SHELL_CLASSES:
        return "shell"
    return "mcp"


class LegacyEnvMixin:
    """v5 ``Env`` authoring surface, adapted onto the v6 :class:`Environment`."""

    # Provided by Environment:
    name: str
    _tasks: dict[str, Task[Any]]
    _on_start: list[Callable[[], Any]]
    _on_stop: list[Callable[[], Any]]
    add_capability: Callable[..., None]

    def _init_legacy(self) -> None:
        """Initialize legacy-compat state (called from ``Environment.__init__``)."""
        #: Tools registered via ``@env.tool`` / ``env.add_tool`` (→ capabilities).
        self._legacy_tools: list[Any] = []
        #: Original (un-normalized) scenario gen fns, keyed by id (for AgentTool schemas).
        self._scenario_fns: dict[str, Callable[..., AsyncGenerator[Any, Any]]] = {}
        #: Scenarios marked ``chat=True`` (accept a ``messages`` history param).
        self._scenario_chat_flags: dict[str, bool] = {}
        #: id -> (returns_type, enable_citations).
        self._scenario_output_config: dict[str, tuple[type | None, bool]] = {}
        #: id -> (exclude_tools, exclude_sources, allowed_tools).
        self._scenario_exclusions: dict[str, tuple[list[str], list[str], list[str]]] = {}
        #: id -> env var names the scenario requires.
        self._scenario_required_env_vars: dict[str, list[str]] = {}
        self._tools_hook_registered = False
        #: Background tasks / workspaces spun up to back synthesized capabilities.
        self._legacy_bg_tasks: list[asyncio.Task[None]] = []
        self._legacy_workspaces: list[Workspace] = []

    # ─── tools (v5 @env.tool / env.add_tool → capabilities) ───────────────

    def add_tool(self, tool: Any, **_kwargs: Any) -> None:
        """[deprecated] Register a tool, turned into a capability at serve time.

        Shell/edit tools become an ``ssh`` capability, computer tools an ``rfb``
        capability, and everything else is served on an ``mcp`` capability. v6:
        declare capabilities explicitly via ``Environment(..., capabilities=[...])``.
        """
        warnings.warn(
            "env.add_tool() is deprecated: in v6, tools are exposed as capabilities. "
            "The tool is collected and converted (ssh/computer/mcp) automatically.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._legacy_tools.append(tool)
        self._ensure_tools_capability()

    def tool(self, name_or_fn: Any = None, **kwargs: Any) -> Any:
        """[deprecated] Register a tool (decorator or call form). See :meth:`add_tool`."""
        if name_or_fn is not None and not isinstance(name_or_fn, str):
            self.add_tool(name_or_fn, **kwargs)
            return name_or_fn

        def decorate(fn: Any) -> Any:
            self.add_tool(fn, **kwargs)
            return fn

        return decorate

    def _ensure_tools_capability(self) -> None:
        """Register on-start/stop hooks that turn collected tools into capabilities."""
        if self._tools_hook_registered:
            return
        self._tools_hook_registered = True
        self._on_start.append(self._serve_legacy_tools)
        self._on_stop.append(self._cleanup_legacy_tools)

    async def _serve_legacy_tools(self) -> None:
        """Stand up ssh/computer/mcp capabilities for the collected tools (on serve)."""
        if not self._legacy_tools:
            return
        buckets: dict[ToolKind, list[Any]] = {"shell": [], "computer": [], "mcp": []}
        for tool in self._legacy_tools:
            buckets[_classify_tool(tool)].append(tool)
        if buckets["shell"]:
            await self._ensure_ssh_capability()
        if buckets["computer"]:
            self._ensure_computer_capability()
        if buckets["mcp"]:
            await self._ensure_mcp_capability(buckets["mcp"])

    async def _ensure_mcp_capability(self, tools: list[Any]) -> None:
        """Serve ``tools`` on a local MCPServer (http) + publish an ``mcp`` capability."""
        try:
            from hud.capabilities import Capability
            from hud.server import MCPServer

            server = MCPServer(name=f"{self.name}-tools")
            added = 0
            for tool in tools:
                try:
                    server.add_tool(tool)
                    added += 1
                except Exception:
                    LOGGER.warning(
                        "legacy env %r: skipping un-servable tool %r (likely a removed v5 tool)",
                        self.name,
                        tool,
                        exc_info=True,
                    )
            if added == 0:
                return
            port = _free_port()
            task = asyncio.create_task(
                server.run_async(transport="http", host="127.0.0.1", port=port, show_banner=False),
            )
            self._legacy_bg_tasks.append(task)
            self.add_capability(Capability.mcp(name="tools", url=f"http://127.0.0.1:{port}/mcp"))
            LOGGER.info(
                "legacy env %r: %d tool(s) -> mcp capability (port %d)",
                self.name,
                len(tools),
                port,
            )
        except Exception:
            LOGGER.warning(
                "legacy env %r: failed to publish mcp tool capability; tasks still serve",
                self.name,
                exc_info=True,
            )

    async def _ensure_ssh_capability(self) -> None:
        """Spin up a :class:`~hud.environment.Workspace` + publish its ``ssh`` capability."""
        try:
            from .workspace import Workspace

            root = os.environ.get("HUD_WORKSPACE_ROOT") or os.getcwd()
            ws = Workspace(root)
            await ws.start()
            self._legacy_workspaces.append(ws)
            self.add_capability(ws.capability())
            LOGGER.info(
                "legacy env %r: shell tool(s) -> ssh capability at %s",
                self.name,
                ws.ssh_url,
            )
        except Exception:
            LOGGER.warning(
                "legacy env %r: could not start an SSH workspace for shell tool(s)",
                self.name,
                exc_info=True,
            )
            warnings.warn(
                "Legacy shell tools could not be converted to an ssh capability. Declare one "
                "explicitly: Environment(..., capabilities=[Workspace(root).capability()]).",
                RuntimeWarning,
                stacklevel=2,
            )

    def _ensure_computer_capability(self) -> None:
        """Publish an ``rfb`` capability for a detected/declared VNC server."""
        from hud.capabilities import Capability

        url = os.environ.get("HUD_RFB_URL") or os.environ.get("HUD_VNC_URL")
        if not url and _port_open("127.0.0.1", 5900):
            url = "rfb://127.0.0.1:5900"
        if not url:
            warnings.warn(
                "Legacy computer tool(s) registered but no VNC/RFB server was detected. Start "
                "one and set HUD_RFB_URL=rfb://host:port (or declare Capability.rfb(...)).",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self.add_capability(
            Capability.rfb(name="screen", url=url, password=os.environ.get("HUD_VNC_PASSWORD")),
        )
        LOGGER.info("legacy env %r: computer tool(s) -> rfb capability at %s", self.name, url)

    async def _cleanup_legacy_tools(self) -> None:
        """Tear down anything :meth:`_serve_legacy_tools` started (best-effort)."""
        for task in self._legacy_bg_tasks:
            task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task
        for ws in self._legacy_workspaces:
            acceptor = getattr(ws, "_acceptor", None)
            if acceptor is not None:
                with contextlib.suppress(Exception):
                    acceptor.close()

    # ─── scenarios (v5 @env.scenario → v6 task) ───────────────────────────

    def scenario(
        self,
        name: str | None = None,
        description: str | None = None,
        *,
        chat: bool = False,
        required_env_vars: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        returns: type | None = None,
        enable_citations: bool = False,
    ) -> Callable[[Callable[P, AsyncGenerator[Any, Any]]], Task[P]]:
        """[deprecated] Register a scenario as a v6 task. Prefer ``@env.task``.

        Accepts the full v5 ``scenario`` signature; the generator (``yield prompt``
        then ``yield reward``) is registered as a v6 task and the v5 metadata
        (``chat``/``returns``/tool exclusions/``required_env_vars``) is retained for
        agents and the task manifest.
        """
        warnings.warn(
            "env.scenario() is deprecated: use @env.task (it accepts the same "
            "yield-prompt-then-reward generator).",
            DeprecationWarning,
            stacklevel=2,
        )

        def decorate(fn: Callable[P, AsyncGenerator[Any, Any]]) -> Task[P]:
            scenario_name = name or fn.__name__
            if ":" in scenario_name:
                raise ValueError(
                    f"scenario name {scenario_name!r} cannot contain ':' (reserved separator)",
                )
            if chat and "messages" not in inspect.signature(fn).parameters:
                raise TypeError(
                    f"chat scenario {scenario_name!r} must accept a 'messages' parameter",
                )

            desc = description or (fn.__doc__ or "").strip().split("\n", 1)[0]
            register = cast("Any", self).task  # provided by Environment
            task: Task[P] = register(id=scenario_name, description=desc, returns=returns)(fn)

            self._scenario_fns[scenario_name] = fn
            if chat:
                self._scenario_chat_flags[scenario_name] = True
            if returns is not None or enable_citations:
                self._scenario_output_config[scenario_name] = (returns, enable_citations)
            if exclude_tools or exclude_sources or allowed_tools:
                self._scenario_exclusions[scenario_name] = (
                    exclude_tools or [],
                    exclude_sources or [],
                    allowed_tools or [],
                )
            if required_env_vars:
                self._scenario_required_env_vars[scenario_name] = required_env_vars
            return task

        return decorate

    # ─── callable factory + run (v5 env("scenario"), env.run) ─────────────

    def __call__(self, name: str, /, **args: Any) -> Any:
        """[deprecated] ``env("scenario")`` → the registered ``Task`` (or a ``Variant``).

        With no args, returns the registered :class:`~hud.environment.task.Task`
        (a callable variant factory — e.g. for ``AgentTool``). With args, returns the
        bound :class:`~hud.eval.Variant`.
        """
        warnings.warn(
            "env('scenario') is deprecated: keep a reference to the @env.task return "
            "value (a Task) and call it to build a Variant.",
            DeprecationWarning,
            stacklevel=2,
        )
        task = self._tasks.get(name)
        if task is None:
            raise KeyError(f"unknown task {name!r} on env {self.name!r}")
        return cast("Any", task)(**args) if args else task

    def run(
        self,
        transport: str | None = None,
        *,
        port: int | None = None,
        host: str = "127.0.0.1",
        **_kwargs: Any,
    ) -> None:
        """[deprecated] Serve the env. v6 serves the control channel, not MCP stdio/http.

        ``transport`` is ignored (v6 always serves its tcp control channel); use
        ``hud dev`` / ``hud deploy`` for managed serving. Prefer ``await env.serve()``.
        """
        warnings.warn(
            "env.run(transport=...) is deprecated: v6 serves a tcp control channel. "
            "Use `hud dev` / `hud deploy`, or `await env.serve(host, port)`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if transport is not None and transport != "tcp":
            LOGGER.warning(
                "env.run: transport %r ignored in v6 (serving tcp control channel)",
                transport,
            )
        asyncio.run(cast("Any", self).serve(host, port or 8765))
