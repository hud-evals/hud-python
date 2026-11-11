"""HUD environment authoring: workspaces, mounts, typed answers, and ``load_environment``.

``Environment`` itself is exported from ``hud``; ``hud.environment.server`` is the
serving entry point substrates run. The robot runtime (``RobotBridge``,
``RobotEndpoint``, ``DirectControl``, ``wrap``) lives in ``hud.environment.robot``;
import it directly, since it needs the ``robot`` extra.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from hud.utils.modules import iter_modules

from .env import Answer, Environment
from .workspace import DEFAULT_SYSTEM_MOUNTS, Mount, Workspace


def load_environment(
    target: str | Path,
    *,
    name: str | None = None,
    args: dict[str, Any] | None = None,
) -> Environment:
    """Resolve an environment reference to the one ``Environment`` it names.

    A source path (a ``.py`` file or directory) is scanned for ``Environment``
    instances and for environments bound to exported tasks; ``name`` selects among
    them by module attribute or ``Environment.name``. A dotted module name reads
    attribute ``name`` (default ``env``): an ``Environment``, or a factory called
    with ``args``. Raises when nothing resolves.
    """
    # Three forms of reference, decided by what the target *is* rather than by
    # importability — every directory in the working tree is an importable
    # namespace package, and ``env.py`` imports as ``env``, so that signal
    # cannot separate them:
    #
    #   a source file    ``env.py``, ``env`` (rewritten), ``./pkg/env.py``
    #   a source tree    any other existing path — scanned for Environments
    #   a module         a package exposing the named attribute, or a dotted
    #                    name with nothing on disk (factories, adapted images)
    path = Path(target)
    if not path.exists() and path.suffix != ".py" and Path(f"{target}.py").exists():
        path = Path(f"{target}.py")

    package_attribute = (
        (name is not None or bool(args))
        and not isinstance(target, Path)
        and "/" not in str(target)
        and (path / "__init__.py").is_file()
    )
    if path.exists() and not package_attribute:
        # Avoid the environment -> eval -> runtime import cycle.
        from hud.eval.taskset import Taskset

        if args:
            raise ValueError(f"args= applies to factory targets, not source path {target}")
        modules = list(iter_modules(path))
        matched = {
            id(env): env
            for module in modules
            for attr, env in vars(module).items()
            if isinstance(env, Environment) and (name is None or name in (attr, env.name))
        }
        for module in modules:
            for task in Taskset._scan_tasks(module):
                for row in (task, task.verifier):
                    if (
                        row is not None
                        and (env := row._env) is not None
                        and (name is None or name == env.name)
                    ):
                        matched[id(env)] = env
        if not matched:
            raise ValueError(f"no Environment{f' named {name!r}' if name else ''} found in {path}")
        if len(matched) > 1:
            raise ValueError(f"multiple Environments in {path}; select one by name")
        return next(iter(matched.values()))

    if path.suffix == ".py" or "/" in str(target):
        raise FileNotFoundError(f"no environment source at {target}")
    obj = getattr(importlib.import_module(str(target)), name or "env")
    env = obj if isinstance(obj, Environment) or not callable(obj) else obj(**args or {})
    if not isinstance(env, Environment):
        raise ValueError(f"{target}:{name or 'env'} resolved to {env!r}, not an Environment")
    return env


__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Answer",
    "Mount",
    "Workspace",
    "load_environment",
]
