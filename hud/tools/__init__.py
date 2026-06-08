"""Deprecated redirects for moved ``hud.tools`` imports.

Known moved modules and symbols emit a ``DeprecationWarning`` and redirect to
their v6 location. Removed computer-tool names resolve to a marker used by the
legacy env adapter to publish an ``rfb`` capability. Truly unknown names fail
normally; this module does not fabricate no-op tools.
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
    "hud.tools.coding.bash": "hud.native.tools.coding.bash",
    "hud.tools.coding.edit": "hud.native.tools.coding.edit",
    "hud.tools.coding.session": "hud.native.tools.coding.session",
    "hud.tools.coding.utils": "hud.native.tools.coding.utils",
    "hud.tools.jupyter": "hud.native.tools.jupyter",
    "hud.tools.memory": "hud.native.tools.memory",
    "hud.tools.playwright": "hud.native.tools.playwright",
    "hud.tools.agent": "hud.native.tools.agent",
    "hud.tools.types": "hud.agents.types",
}

#: Old top-level ``hud.tools`` symbol -> real v6 module to import it from.
_NAME_REDIRECTS: dict[str, str] = {
    "AgentTool": "hud.native.tools.agent",
    "BaseTool": "hud.native.tools.base",
    "BashTool": "hud.native.tools.coding",
    "ClaudeBashSession": "hud.native.tools.coding",
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


class LegacyComputerTool:
    """Marker for a removed v5 computer tool.

    The legacy env adapter uses ``_legacy_capability_kind`` to publish an
    ``rfb`` capability when an old env registers a computer tool class.
    """

    _legacy_capability_kind = "computer"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.name = "computer"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __getattr__(self, _name: str) -> None:
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
    raise AttributeError(f"module {module_name!r} has no attribute {name!r}")


def _make_marker_getattr(module_name: str) -> Any:
    def __getattr__(name: str) -> Any:
        return _resolve_name(module_name, name)

    return __getattr__


def _make_redirect_getattr(module_name: str, target_name: str) -> Any:
    """Lazily resolve attributes from the redirect target on each access.

    Resolving lazily (instead of copying attrs once at import time) avoids a
    partial-import race: the target is fully imported by the time an attribute is
    actually read.
    """

    def __getattr__(name: str) -> Any:
        target = importlib.import_module(target_name)
        if hasattr(target, name):
            return getattr(target, name)
        return _resolve_name(module_name, name)

    return __getattr__


class _DeprecatedToolsFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve known ``hud.tools.*`` redirects and legacy computer markers."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if not fullname.startswith("hud.tools."):
            return None
        if fullname not in _MODULE_REDIRECTS and not _is_computer_module(fullname):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: Any) -> ModuleType:
        return ModuleType(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        name = module.__name__
        redirect = _MODULE_REDIRECTS.get(name)
        if redirect is not None:
            warnings.warn(
                f"{name} moved to {redirect} ({_MSG})",
                DeprecationWarning,
                stacklevel=2,
            )
            module.__getattr__ = _make_redirect_getattr(name, redirect)  # type: ignore[attr-defined]
            return
        module.__path__ = []
        module.__getattr__ = _make_marker_getattr(name)  # type: ignore[attr-defined]


if not any(isinstance(f, _DeprecatedToolsFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _DeprecatedToolsFinder())
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)


def __getattr__(name: str) -> Any:
    return _resolve_name("hud.tools", name)
