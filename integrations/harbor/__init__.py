"""Harbor (terminal-bench layout) interop: load, adapt, export.

Harbor task structure::

    task_name/
    ├── instruction.md          # agent prompt
    ├── task.toml               # config: timeouts, metadata
    ├── environment/Dockerfile  # container the agent works in
    ├── tests/test.sh           # verification -> writes reward.txt / .json
    └── solution/               # optional (ignored)

Harbor's agent works *inside* its container, so :func:`environment` (the
:class:`~hud.environment.Integration` constructor) is meaningful only in
there. :func:`adapt` builds the images and returns the loaded rows with those
images bound to them::

    taskset = await harbor.adapt("./tasks")
    job = await taskset.run(agent, runtime=DockerRuntime())

    taskset = await harbor.adapt("./tasks", push="registry.io/x")
    job = await taskset.run(agent, runtime=HUDRuntime())

Plus :func:`export`, the reverse direction (HUD tasks -> Harbor folders).
Compose-based and prebuilt-``docker_image`` tasks are not supported yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.environment import Integration

if TYPE_CHECKING:
    from pathlib import Path

    from hud.environment import Environment
    from hud.eval import Taskset

from ._adapt import adapt
from ._export import export
from ._load import detect, load
from ._runtime import environment


class _Harbor(Integration):
    """The :class:`~hud.environment.Integration` contract for Harbor."""

    name = "harbor"

    def load(self, ref: str | Path) -> Taskset:
        return load(ref)

    def environment(self, ref: str | Path, *, name: str | None = None) -> Environment:
        return environment(ref, name=name)


integration = _Harbor()

__all__ = [
    "adapt",
    "detect",
    "environment",
    "export",
    "integration",
    "load",
]
