"""HUD environment authoring: declarations and the wire protocol that serves them.

:class:`Environment` is the declaration (capabilities + tasks behind the wire
protocol); ``load_environment`` selects one from authored ``.py`` source;
:mod:`~hud.environment.server` is the serving entry point substrates run.
How a substrate comes up — placement — belongs to the eval engine: see
:mod:`hud.eval.runtime` (:class:`~hud.eval.runtime.Runtime`, the ``Provider``
contract, ``LocalRuntime``, ``DockerRuntime``, ``HUDRuntime``).

The env-side robot runtime (bridges, action providers, sim runners, contract
tooling, recording glue) lives in :mod:`hud.environment.robot`; import it
directly — it pulls optional dependencies (numpy/msgpack, the ``robot`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.capabilities import Capability
from hud.server import MCPRouter
from hud.utils.modules import iter_modules

from .env import Answer, Environment
from .workspace import DEFAULT_SYSTEM_MOUNTS, Mount, MountKind, Workspace

if TYPE_CHECKING:
    from pathlib import Path

ToolRouter = MCPRouter


def load_environment(path: str | Path, *, name: str | None = None) -> Environment:
    """Return the one :class:`Environment` defined at *path* (file or directory).

    *name* selects among multiple environments, matching either the module
    attribute name or ``Environment.name``. Raises ``ValueError`` when nothing
    matches or the choice is ambiguous.
    """
    matched = [
        env
        for module in iter_modules(path)
        for attr, env in vars(module).items()
        if isinstance(env, Environment) and (name is None or name in (attr, env.name))
    ]
    if not matched:
        raise ValueError(f"no Environment{f' named {name!r}' if name else ''} found in {path}")
    if len(matched) > 1:
        raise ValueError(f"multiple Environments in {path}; select one by name")
    return matched[0]


__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Answer",
    "Capability",
    "Environment",
    "MCPRouter",
    "Mount",
    "MountKind",
    "ToolRouter",
    "Workspace",
    "load_environment",
]
