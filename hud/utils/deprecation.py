"""Deprecation warnings for what hud v0.7.0 removes or changes, and retired import paths."""

from __future__ import annotations

import warnings
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Any, NamedTuple

from hud.utils.exceptions import HudDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


RUNTIME_NAMES = (
    "ComposeProject",
    "DockerBindMount",
    "DockerRuntime",
    "HUDRuntime",
    "HostedRuntime",
    "LocalRuntime",
    "ModalRuntime",
    "Provider",
    "Runtime",
    "RuntimeConfig",
    "RuntimeGPU",
    "RuntimeLimits",
    "RuntimeResources",
    "RuntimeTPU",
    "Shared",
    "SubprocessRuntime",
)
"""The runtime names v0.6.23 exported from ``hud.eval.runtime``; ``hud.eval`` and ``hud``
exported subsets of them."""


def warn_deprecated(
    subject: str, instead: str, *, change: str = "will be removed", stacklevel: int = 2
) -> None:
    """Warn that ``subject`` is deprecated and ``change`` in hud v0.7.0; ``instead`` is the fix.

    ``stacklevel`` counts from the caller, as ``warnings.warn``'s does.
    """
    warnings.warn(
        f"{subject} is deprecated and {change} in hud v0.7.0; {instead}.",
        HudDeprecationWarning,
        stacklevel=stacklevel + 1,
    )


class Retired(NamedTuple):
    """How to load a retired name's value, and what the warning tells the caller instead."""

    load: Callable[[], Any]
    instead: str


def _import(module: str, name: str) -> Any:
    return getattr(import_module(module), name)


def served_from(module: str, *names: str, instead: str) -> dict[str, Retired]:
    """Entries for names whose value loads from ``module``, with ``instead`` in the warning."""
    return {name: Retired(partial(_import, module, name), instead) for name in names}


def moved_to(module: str, *names: str) -> dict[str, Retired]:
    """Entries for names that now import from ``module`` under the same name."""
    return served_from(module, *names, instead=f"import it from `{module}`")


def resolve_retired(module: str, name: str, retired: Mapping[str, Retired]) -> Any:
    """Return a retired name's value for a module ``__getattr__``, with a warning.

    The warning is attributed to the caller's import or attribute access. Names
    not in ``retired`` raise ``AttributeError``, so imports of names that never
    existed fail as before.
    """
    entry = retired.get(name)
    if entry is None:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    warn_deprecated(f"`{name}` from `{module}`", entry.instead, stacklevel=3)
    return entry.load()
