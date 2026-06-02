"""Deprecated shim for the old ``hud.tools`` package.

The tools moved in the v6 teardown:

- standalone tools (``BaseTool``, ``BashTool``, ``EditTool``, ``JupyterTool``,
  ``MemoryTool``, ``PlaywrightTool``) → :mod:`hud.native.tools`
- result/answer types (``Citation``, ``AgentAnswer``, ``ScenarioResult`` /
  ``EvaluationResult``, ``ContentResult``, ``SubScore``, ``Coordinate``,
  ``ToolError``) → :mod:`hud.agents.types`

Old ``hud.tools`` and ``hud.tools.*`` imports still resolve so existing code keeps
importing, but every symbol is a **no-op stand-in** that emits a
``DeprecationWarning``. Update imports to the locations above.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
import warnings
from typing import Any

_MSG = (
    "hud.tools is deprecated: use hud.native.tools (tools) and hud.agents.types "
    "(result types). The hud.tools symbols are now no-ops."
)


class _NoOp:
    """No-op stand-in for a removed ``hud.tools`` symbol.

    Constructs, calls, and attribute-accesses all return a no-op so legacy code
    importing ``hud.tools`` keeps importing (it just does nothing).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __getattr__(self, _name: str) -> Any:
        return self


def _make_getattr(module_name: str) -> Any:
    def __getattr__(name: str) -> Any:
        warnings.warn(
            f"{module_name}.{name} is a no-op ({_MSG})",
            DeprecationWarning,
            stacklevel=2,
        )
        return _NoOp

    return __getattr__


class _DeprecatedToolsFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve any ``hud.tools.*`` submodule to a no-op module (at any depth)."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if not fullname.startswith("hud.tools."):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: Any) -> types.ModuleType:
        module = types.ModuleType(spec.name)
        module.__path__ = []  # mark as package so deeper imports route back here
        module.__getattr__ = _make_getattr(spec.name)  # type: ignore[attr-defined]
        return module

    def exec_module(self, module: types.ModuleType) -> None: ...


if not any(isinstance(f, _DeprecatedToolsFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _DeprecatedToolsFinder())
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)


__getattr__ = _make_getattr("hud.tools")
