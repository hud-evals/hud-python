"""The MCP tool surface: the tools of the data sources the running task loaded."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager

from fastmcp.server.providers import Provider
from fastmcp.tools import Tool

from data_sources.base import DataSource


class TaskInProgress(RuntimeError):
    """A task started while another task held the environment."""


class MountedSources(Provider):
    """Serves the tools of the running task's data sources.

    The environment's tools belong to one task at a time, so a task holds them
    with :meth:`occupy` from before it loads its sources until it is graded or
    cancelled.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tools: dict[str, Tool] = {}
        self._occupied = False

    @contextmanager
    def occupy(self) -> Iterator[None]:
        """Hold the environment for one task; its tools are removed when it ends."""
        if self._occupied:
            raise TaskInProgress("Another task is running; this environment runs one at a time")
        self._occupied = True
        try:
            yield
        finally:
            self._tools = {}
            self._occupied = False

    def mount(self, sources: Iterable[DataSource]) -> None:
        tools: dict[str, Tool] = {}
        for source in sources:
            for item in source.tools():
                if item.name in tools:
                    raise ValueError(f"Two data sources define the tool {item.name}")
                tools[item.name] = item
        self._tools = tools

    async def _list_tools(self) -> Sequence[Tool]:
        return list(self._tools.values())
