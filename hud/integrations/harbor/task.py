"""Harbor tasks run natively: the authored image in a box, Harbor's lifecycle here."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from hud.eval.run import connect
from hud.eval.task import Task
from hud.runtime.core import Box, RuntimePort

from .adapt import TaskConfig
from .session import HarborSession, HarborVerifierSession

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.environment.egress import WorkspaceRoute
    from hud.runtime.core import Runtime


class HarborTask(Task):
    """One Harbor task, as its directory and its ``task.toml`` say.

    ``mcp_ports`` names, for each MCP server the task's own services serve, the port it is
    published from.
    """

    path: Path
    config: TaskConfig
    mcp_ports: dict[str, RuntimePort] = Field(default_factory=dict)


class HarborVerifierTask(Task):
    """The separate verifier of a Harbor task, graded in a box of its own."""

    path: Path
    config: TaskConfig


def _box(runtime: Runtime, workspace_routes: Sequence[WorkspaceRoute]) -> Box:
    if not isinstance(runtime, Box):
        raise TypeError(f"a Harbor task needs a runtime it can run commands in, not {runtime!r}")
    if workspace_routes:
        raise ValueError("Harbor tasks take no workspace routes yet")
    return runtime


@connect.register(HarborTask)
def _connect(
    task: HarborTask,
    runtime: Runtime,
    *,
    workspace_routes: Sequence[WorkspaceRoute] = (),
) -> HarborSession:
    return HarborSession(task, _box(runtime, workspace_routes))


@connect.register(HarborVerifierTask)
def _connect_verifier(
    task: HarborVerifierTask,
    runtime: Runtime,
    *,
    workspace_routes: Sequence[WorkspaceRoute] = (),
) -> HarborVerifierSession:
    return HarborVerifierSession(task, _box(runtime, workspace_routes))
