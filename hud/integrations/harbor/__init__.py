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
there — :func:`adapt` packages it: one HUD-speaking image per env group
whose CMD serves ``harbor.environment``, and the same rows then run on any
container placement::

    await harbor.adapt("./tasks")  # local images
    job = await harbor.load("./tasks").run(agent, runtime=DockerRuntime())

    await harbor.adapt("./tasks", push="registry.io/x")  # or hud deploy the
    job = await harbor.load("./tasks").run(agent, runtime=HUDRuntime())  # contexts

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

from ._adapt import adapt, docker_runtime, environment
from ._export import ALLOWED_PROTOCOLS, CONTROL_PORT, DEFAULT_ANSWER_FILE, export
from ._load import agent_timeout, detect, grouped, load


class Harbor(Integration):
    """The :class:`~hud.environment.Integration` contract for Harbor."""

    name = "harbor"

    def load(self, ref: str | Path) -> Taskset:
        return load(ref)

    def environment(self, ref: str | Path, *, name: str | None = None) -> Environment:
        return environment(ref, name=name)


integration = Harbor()

__all__ = [
    "ALLOWED_PROTOCOLS",
    "CONTROL_PORT",
    "DEFAULT_ANSWER_FILE",
    "Harbor",
    "adapt",
    "agent_timeout",
    "detect",
    "docker_runtime",
    "environment",
    "export",
    "grouped",
    "integration",
    "load",
]
