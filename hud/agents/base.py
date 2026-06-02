"""Agent ABC: the rollout contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hud.client import Run
    from hud.server import MCPServer


class Agent(ABC):
    """An agent turns a live run into a ``Trace``.

    Subclasses implement ``__call__(run)`` and callers drive an agent with
    ``await agent(run)``. An agent is stateless with respect to any single run —
    everything it needs comes from ``run`` (``run.prompt`` and capabilities via
    ``run.client.open`` / ``run.client.binding``) — so one instance can drive many
    concurrent rollouts safely.

    ``run`` owns the trace (like an RL rollout buffer or an open telemetry span):
    the agent *fills* ``run.trace`` in place — messages, samples, and the final
    ``content`` (the answer the env grades on exit) — rather than returning a new
    one. The caller reads the result back off ``run.trace``.

    ``native_tools`` are standalone :class:`hud.native.tools.BaseTool`s the agent
    carries to *serve* (the catalog tools are capability proxies that forward to an
    env, so they are not servable). :meth:`as_mcp_server` turns them into a running
    server an ``Environment`` can attach as an ``mcp`` capability.
    """

    #: Standalone BaseTools (instances or classes) this agent exposes via MCP.
    native_tools: ClassVar[tuple[Any, ...]] = ()

    @abstractmethod
    async def __call__(self, run: Run) -> None:
        """Drive ``run`` to completion, filling ``run.trace`` (answer is ``trace.content``)."""

    def as_mcp_server(
        self, *, name: str | None = None, tools: list[Any] | None = None
    ) -> MCPServer:
        """Expose this agent's native tools as a :class:`~hud.server.MCPServer`.

        The agent's *catalog* tools are capability proxies (they forward execution to
        an env), so they are not servable. The servable ones are ``native_tools`` —
        standalone ``BaseTool``s the agent was built with. Each is registered on a
        fresh ``MCPServer`` (the new ``Environment`` attaches it as an ``mcp``
        capability; ``hud dev`` can serve it directly). Pass ``tools`` to override.
        """
        from hud.server import MCPServer

        server_name = name or getattr(self, "model_name", None) or type(self).__name__
        server = MCPServer(name=server_name)
        for tool in tools if tools is not None else self.native_tools:
            server.add_tool(tool() if isinstance(tool, type) else tool)
        return server
