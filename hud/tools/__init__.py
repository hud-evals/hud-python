"""Deprecated shim for the old ``hud.tools`` package.

The tools moved in the v6 teardown, but deployed v5 envs still import from here, so
this shim keeps those imports working (each emits a ``DeprecationWarning``):

- standalone tools (``BaseTool``/``BaseHub``, ``BashTool``/``EditTool``,
  ``JupyterTool``, ``MemoryTool``, ``PlaywrightTool``, ``AgentTool``)
  → redirected to the real classes in :mod:`hud.native.tools`
- result/answer types (``AgentAnswer``, ``Citation``, ``EvaluationResult`` /
  ``ScenarioResult``, ``ContentResult``, ``SubScore``, ``ToolError``)
  → redirected to :mod:`hud.agents.types`
- computer tools (``HudComputerTool``, ``AnthropicComputerTool``, …) were removed;
  they resolve to a lightweight marker so an env that registers one still gets a
  ``computer`` (rfb) capability synthesized at serve time (see
  :mod:`hud.environment.legacy_capabilities`)
- anything else resolves to a **no-op** stand-in

Update imports to the locations above.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

# Import ``ModuleType`` by name — a plain ``import types`` would be rebound to the
# ``hud.tools.types`` submodule once it's imported, breaking ``create_module``.
from types import ModuleType
from typing import Any

_MSG = (
    "hud.tools is deprecated: use hud.native.tools (tools) and hud.agents.types "
    "(result types). This shim keeps old imports working for now."
)

#: Old ``hud.tools`` submodule -> real v6 module to re-export.
_MODULE_REDIRECTS: dict[str, str] = {
    "hud.tools.base": "hud.native.tools.base",
    "hud.tools.coding": "hud.native.tools.coding",
    "hud.tools.jupyter": "hud.native.tools.jupyter",
    "hud.tools.memory": "hud.native.tools.memory",
    "hud.tools.playwright": "hud.native.tools.playwright",
    "hud.tools.agent": "hud.native.tools.agent",
    "hud.tools.types": "hud.agents.types",
}

#: Old top-level ``hud.tools`` symbol -> real v6 module to import it from.
_NAME_REDIRECTS: dict[str, str] = {
    "AgentTool": "hud.native.tools.agent",
    "BaseHub": "hud.native.tools.base",
    "BaseTool": "hud.native.tools.base",
    "BashTool": "hud.native.tools.coding",
    "EditTool": "hud.native.tools.coding",
    "JupyterTool": "hud.native.tools.jupyter",
    "MemoryTool": "hud.native.tools.memory",
    "PlaywrightTool": "hud.native.tools.playwright",
    "AgentAnswer": "hud.agents.types",
    "Citation": "hud.agents.types",
    "ContentResult": "hud.agents.types",
    "EvaluationResult": "hud.agents.types",
    "ScenarioResult": "hud.agents.types",
    "SubScore": "hud.agents.types",
    "ToolError": "hud.agents.types",
}


def _is_computer_name(name: str) -> bool:
    return "Computer" in name


def _is_computer_module(fullname: str) -> bool:
    return fullname.startswith("hud.tools.computer")


class _NoOp:
    """No-op stand-in for a removed (non-redirected) ``hud.tools`` symbol."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __getattr__(self, _name: str) -> Any:
        return self


class LegacyComputerTool:
    """Marker for a removed computer tool.

    Carries ``_legacy_capability_kind = "computer"`` so the legacy env adapter
    publishes a ``computer`` (rfb) capability when one is registered, instead of
    silently no-op'ing it.
    """

    _legacy_capability_kind = "computer"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.name = "computer"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __getattr__(self, _name: str) -> Any:
        return None


def _warn(what: str) -> None:
    warnings.warn(f"{what} ({_MSG})", DeprecationWarning, stacklevel=3)


def _resolve_name(module_name: str, name: str) -> Any:
    """Resolve a ``hud.tools[.x]`` attribute, redirecting/marker/no-op as needed."""
    target = _NAME_REDIRECTS.get(name)
    if target is not None:
        _warn(f"{module_name}.{name} moved to {target}.{name}")
        return getattr(importlib.import_module(target), name)
    if _is_computer_name(name):
        _warn(f"{module_name}.{name} was removed; using a computer-capability marker")
        return LegacyComputerTool
    _warn(f"{module_name}.{name} is a no-op")
    return _NoOp


def _make_getattr(module_name: str) -> Any:
    def __getattr__(name: str) -> Any:
        return _resolve_name(module_name, name)

    return __getattr__


def _make_redirect_getattr(module_name: str, target_name: str) -> Any:
    """Lazily resolve attributes from the redirect target on each access.

    Resolving lazily (instead of copying attrs once at import time) avoids a
    partial-import race: the target is fully imported by the time an attribute is
    actually read. Names the target lacks (dropped v5 symbols) fall back to a
    marker/no-op.
    """

    def __getattr__(name: str) -> Any:
        target = importlib.import_module(target_name)
        if hasattr(target, name):
            return getattr(target, name)
        return _resolve_name(module_name, name)

    return __getattr__


class _DeprecatedToolsFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``hud.tools.*`` submodules: redirect, computer-marker, or no-op."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if not fullname.startswith("hud.tools."):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: Any) -> ModuleType:
        return ModuleType(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        name = module.__name__
        redirect = _MODULE_REDIRECTS.get(name)
        if redirect is not None:
            warnings.warn(
                f"{name} moved to {redirect} ({_MSG})", DeprecationWarning, stacklevel=2,
            )
            # Resolve attributes lazily from the target (avoids a partial-import
            # race); dropped v5 names fall back to a marker/no-op.
            module.__getattr__ = _make_redirect_getattr(name, redirect)  # type: ignore[attr-defined]
            return
        # Non-redirected submodule: resolve names lazily (computer marker / no-op).
        module.__path__ = []  # mark as package so deeper imports route back here
        module.__getattr__ = _make_getattr(name)  # type: ignore[attr-defined]


if not any(isinstance(f, _DeprecatedToolsFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _DeprecatedToolsFinder())
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)


def __getattr__(name: str) -> Any:
    return _resolve_name("hud.tools", name)
