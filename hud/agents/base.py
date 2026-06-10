"""Agent ABC: the rollout contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hud.eval.rollout import Run
    from hud.server import MCPServer


class Agent(ABC):
    """Drives a live ``Run`` to completion by filling ``run.trace`` in place.

    Subclasses implement ``__call__(run)``; callers do ``await agent(run)``. Stateless
    per run — everything comes from ``run`` — so one instance drives many concurrent
    rollouts. ``native_tools`` are standalone ``BaseTool``s the agent can *serve* via
    :meth:`as_mcp_server` (catalog tools are capability proxies, not servable).
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
