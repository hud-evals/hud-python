"""Retired import paths, served with a warning until hud v0.7.0 removes them."""

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


class Retired(NamedTuple):
    """How to load a retired name's value, and what the warning tells the caller instead."""

    load: Callable[[], Any]
    instead: str


def _import(module: str, name: str) -> Any:
    return getattr(import_module(module), name)


def moved_to(module: str, *names: str) -> dict[str, Retired]:
    """Entries for names that now import from ``module`` under the same name."""
    return {
        name: Retired(partial(_import, module, name), f"import it from `{module}`")
        for name in names
    }


def resolve_retired(module: str, name: str, retired: Mapping[str, Retired]) -> Any:
    """Return a retired name's value for a module ``__getattr__``, with a warning.

    The warning is attributed to the caller's import or attribute access. Names
    not in ``retired`` raise ``AttributeError``, so imports of names that never
    existed fail as before.
    """
    entry = retired.get(name)
    if entry is None:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    warnings.warn(
        f"`{name}` from `{module}` is deprecated and will be removed in hud v0.7.0; "
        f"{entry.instead}.",
        HudDeprecationWarning,
        stacklevel=3,
    )
    return entry.load()
