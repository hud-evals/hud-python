"""HUD environment authoring: declarations and the wire protocol that serves them.

:class:`Environment` is the declaration (capabilities + tasks behind the wire
protocol); ``load_environment`` resolves an env
reference (source path, or dotted module attr/factory);
:mod:`~hud.environment.server` is the serving entry point substrates run.
How a substrate comes up — placement — belongs to the eval engine: see
:mod:`hud.eval.runtime` (:class:`~hud.eval.runtime.Runtime`, the ``Provider``
contract, ``LocalRuntime``, ``DockerRuntime``, ``HUDRuntime``).

The env-side robot runtime (bridges, action providers, sim runners, contract
tooling, recording glue) lives in :mod:`hud.environment.robot`; import it
directly — it pulls optional dependencies (numpy/msgpack, the ``robot`` extra).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from hud.capabilities import Capability
from hud.utils.modules import iter_modules

from .arguments import DataFileArg, DataFileRef, DataFilesArg, GradingArg, PromptArg
from .egress import Peer
from .env import Answer, Environment
from .workspace import DEFAULT_SYSTEM_MOUNTS, Mount, MountKind, Workspace


def load_environment(
    target: str | Path,
    *,
    name: str | None = None,
    args: dict[str, Any] | None = None,
) -> Environment:
    """Resolve an environment reference to the one :class:`Environment` it names.

    A source path (``.py`` file or directory) is scanned for ``Environment``
    instances; *name* selects among several, matching the module attribute or
    ``Environment.name``. A dotted importable module reads attribute *name*
    (default ``env``): an ``Environment`` directly, or a factory called with
    *args* to build one — how programmatic envs (integrations, adapted
    images) are served without a source file. Raises when nothing resolves.
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
        path = Path(f"{target}.py")  # bare dev references: 'env' means env.py

    package_attribute = (
        (name is not None or bool(args))
        and not isinstance(target, Path)
        and "/" not in str(target)
        and (path / "__init__.py").is_file()
    )
    if path.exists() and not package_attribute:
        if args:
            raise ValueError(f"args= applies to factory targets, not source path {target}")
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

    if path.is_file() or "/" in str(target):
        raise FileNotFoundError(f"no environment source at {target}")
    obj = getattr(importlib.import_module(str(target)), name or "env")
    env = obj if isinstance(obj, Environment) or not callable(obj) else obj(**args or {})
    if not isinstance(env, Environment):
        raise ValueError(f"{target}:{name or 'env'} resolved to {env!r}, not an Environment")
    return env


__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Answer",
    "Capability",
    "DataFileArg",
    "DataFileRef",
    "DataFilesArg",
    "Environment",
    "GradingArg",
    "Mount",
    "MountKind",
    "Peer",
    "PromptArg",
    "Workspace",
    "load_environment",
]
