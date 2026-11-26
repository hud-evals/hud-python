"""HUD environment authoring: workspaces, mounts, typed answers, and ``load_environment``.

``read_frame``, ``send_frame``, ``reply`` and ``error`` are the control channel's
framing (newline-delimited JSON-RPC 2.0), for a process that speaks it to the env.

``Environment`` itself is exported from ``hud``; ``hud.environment.server`` is the
serving entry point hosts run.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from hud.utils.deprecation import moved_to, resolve_retired
from hud.utils.modules import iter_modules

from .env import Answer, Environment
from .utils import HudProtocolError, error, read_frame, reply, send_frame
from .workspace import Mount, Workspace


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
        from hud.eval.taskset import module_tasks

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
            for task in module_tasks(module):
                for row in (task, task.verifier):
                    if (
                        row is not None
                        and (env := row.environment) is not None
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
    "Answer",
    "HudProtocolError",
    "Mount",
    "Workspace",
    "error",
    "load_environment",
    "read_frame",
    "reply",
    "send_frame",
]

_RETIRED = {
    **moved_to("hud.capabilities", "Capability"),
    **moved_to("hud.environment.egress", "Peer"),
    **moved_to("hud.environment.workspace", "DEFAULT_SYSTEM_MOUNTS", "MountKind"),
}


def __getattr__(name: str) -> Any:
    # The argument-hint helpers load on first use, so ``import hud`` never builds them.
    retired = (
        _RETIRED if name in _RETIRED else importlib.import_module(".arguments", __name__).RETIRED
    )
    return resolve_retired(__name__, name, retired)
