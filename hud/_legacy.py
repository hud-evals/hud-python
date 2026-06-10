"""All v5 backward compatibility, quarantined in one module.

Deployed v5 environments keep running on v6 through one meta-path finder,
installed by ``hud/__init__`` at import time:

- ``hud.native[.graders|.skills|.tools...]`` — the package was dissolved into
  root modules (:mod:`hud.graders`, :mod:`hud.skills`, :mod:`hud.tools`).
  These names resolve as synthetic alias modules that delegate attribute
  access to the real modules, so class identity is preserved for
  ``isinstance`` checks.
- ``hud.services`` — the package was removed; ``Chat`` moved to
  :mod:`hud.eval.chat` (the alias serves it). ``ChatService`` (the A2A
  executor) left the SDK entirely.
- removed ``hud.tools`` submodules (``types``, ``computer``, ``filesystem``,
  ``executors``, ...) — ``hud.tools.types`` redirects to
  :mod:`hud.agents.types`; the rest resolve names lazily (marker/no-op).
- removed ``hud.tools`` symbols — :func:`resolve_legacy_name` (hooked from the
  real modules' ``__getattr__``) redirects result types to
  :mod:`hud.agents.types`, maps removed computer and shell/edit tools to
  capability markers consumed by :mod:`hud.environment.legacy` (→ ``rfb`` /
  ``ssh``), and no-ops the rest. Each resolution emits a
  ``DeprecationWarning``.

Also home to the :class:`Grade` shim — the v5 grading entry point, replaced by
:func:`hud.graders.combine`.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from pathlib import Path

# Import ``ModuleType`` by name — a plain ``import types`` would be rebound to the
# legacy ``hud.tools.types`` submodule once it's imported, breaking ``create_module``.
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from hud.agents.types import EvaluationResult, SubScore

_MSG = (
    "this symbol was removed in v6; result types live in hud.agents.types. "
    "This compat layer keeps old imports working for now."
)

#: Removed ``hud.tools`` submodule -> real v6 module to re-export.
_MODULE_REDIRECTS: dict[str, str] = {
    "hud.tools.types": "hud.agents.types",
}

#: Removed top-level ``hud.tools`` symbol -> real v6 module to import it from.
_NAME_REDIRECTS: dict[str, str] = {
    "AgentAnswer": "hud.agents.types",
    "Citation": "hud.agents.types",
    "ContentResult": "hud.agents.types",
    "EvaluationResult": "hud.agents.types",
    "ScenarioResult": "hud.agents.types",
    "SubScore": "hud.agents.types",
    "ToolError": "hud.agents.types",
}

#: Removed lowercase v5 symbols (module-level instances/functions rather than classes).
_LOWERCASE_LEGACY = frozenset({"computer_settings", "get_demote_preexec_fn"})

#: Removed legacy module -> real v6 module whose attributes it re-exposes.
_MODULE_ALIASES: dict[str, str] = {
    "hud.native": "hud.graders",
    "hud.native.graders": "hud.graders",
    "hud.native.skills": "hud.skills",
    "hud.services": "hud.eval.chat",
    "hud.services.chat": "hud.eval.chat",
}

_TOOLS_DIR = Path(__file__).parent / "tools"


class Grade:
    """v5 compat shim — use :func:`hud.graders.combine` instead.

    v5 environments call ``Grade.gather(...)`` and ``Grade.from_subscores(...)``.
    Importable as ``hud.native.Grade`` / ``hud.native.graders.Grade``.
    """

    @staticmethod
    async def gather(*items: SubScore | Awaitable[SubScore]) -> EvaluationResult:
        from hud.graders import combine  # lazy: hud.graders is not loaded at install time

        return await combine(*items)

    @staticmethod
    def from_subscores(subscores: list[SubScore]) -> EvaluationResult:
        from hud.graders import _combine_subscores

        return _combine_subscores(subscores)


class _NoOp:
    """No-op stand-in for a removed (non-redirected) v5 symbol."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __getattr__(self, _name: str) -> Any:
        return self


class _LegacyCapabilityMarker:
    """Marker for a removed v5 tool that maps to a capability.

    Carries ``_legacy_capability_kind`` so the legacy env adapter
    (:mod:`hud.environment.legacy`) publishes the matching capability when one
    is registered, instead of silently no-op'ing it.
    """

    _legacy_capability_kind: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.name = self._legacy_capability_kind

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __getattr__(self, _name: str) -> Any:
        return None


class LegacyComputerTool(_LegacyCapabilityMarker):
    """Removed computer tool → ``rfb`` capability at serve time."""

    _legacy_capability_kind = "computer"


class LegacyShellTool(_LegacyCapabilityMarker):
    """Removed shell/edit tool (``BashTool``, ``EditTool``, …) → ``ssh`` capability."""

    _legacy_capability_kind = "shell"


#: Substrings identifying removed v5 shell/edit tool classes.
_SHELL_NAME_HINTS = ("Bash", "Shell", "Edit", "Patch")


def _warn(what: str) -> None:
    warnings.warn(f"{what} ({_MSG})", DeprecationWarning, stacklevel=3)


def resolve_legacy_name(module_name: str, name: str) -> Any:
    """Resolve a removed v5 attribute: redirect, marker, or no-op.

    Only CamelCase names (v5 exported classes) and a known set of lowercase v5
    instances resolve. Anything else (dunders, ``pytest_plugins``, …) raises
    ``AttributeError`` so module introspection behaves.
    """
    if name in _LOWERCASE_LEGACY:
        _warn(f"{module_name}.{name} is a no-op")
        return _NoOp()
    if not name[:1].isupper():
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    target = _NAME_REDIRECTS.get(name)
    if target is not None:
        _warn(f"{module_name}.{name} moved to {target}.{name}")
        return getattr(importlib.import_module(target), name)
    if "Computer" in name:
        _warn(f"{module_name}.{name} was removed; using a computer-capability marker")
        return LegacyComputerTool
    if any(hint in name for hint in _SHELL_NAME_HINTS):
        _warn(f"{module_name}.{name} was removed; using a shell-capability marker")
        return LegacyShellTool
    _warn(f"{module_name}.{name} is a no-op")
    return _NoOp


def _alias_target(fullname: str) -> str | None:
    """Real module behind an aliased legacy name, or None if unknown."""
    alias = _MODULE_ALIASES.get(fullname)
    if alias is not None:
        return alias
    if fullname == "hud.native.tools" or fullname.startswith("hud.native.tools."):
        return fullname.replace("hud.native.tools", "hud.tools", 1)
    return None


def _is_real_tools_submodule(fullname: str) -> bool:
    relative = fullname.removeprefix("hud.tools.").replace(".", "/")
    return (_TOOLS_DIR / f"{relative}.py").exists() or (_TOOLS_DIR / relative).is_dir()


def _make_alias_getattr(fullname: str, target_name: str) -> Any:
    def __getattr__(name: str) -> Any:
        if name == "Grade" and target_name == "hud.graders":
            return Grade
        target = importlib.import_module(target_name)
        if hasattr(target, name):
            return getattr(target, name)
        raise AttributeError(f"module {fullname!r} has no attribute {name!r}")

    return __getattr__


def _make_legacy_getattr(module_name: str) -> Any:
    def __getattr__(name: str) -> Any:
        return resolve_legacy_name(module_name, name)

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
        return resolve_legacy_name(module_name, name)

    return __getattr__


class _V5CompatFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve removed-module aliases and **removed** ``hud.tools.*`` submodules.

    Real ``hud.tools`` submodules (``base``, ``agent``) are skipped so the
    normal import machinery handles them.
    """

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname.startswith(("hud.native", "hud.services")):
            if _alias_target(fullname) is None:
                return None  # unknown legacy name: fail with ModuleNotFoundError
            return importlib.util.spec_from_loader(fullname, self)
        if fullname.startswith("hud.tools.") and not _is_real_tools_submodule(fullname):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec: Any) -> ModuleType:
        return ModuleType(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        name = module.__name__

        if name.startswith(("hud.native", "hud.services")):
            target = _alias_target(name)
            assert target is not None  # find_spec already filtered unknowns
            module.__path__ = []  # mark as package so submodule imports route back here
            module.__getattr__ = _make_alias_getattr(name, target)  # type: ignore[attr-defined]
            return

        redirect = _MODULE_REDIRECTS.get(name)
        if redirect is not None:
            warnings.warn(
                f"{name} moved to {redirect} ({_MSG})",
                DeprecationWarning,
                stacklevel=2,
            )
            module.__getattr__ = _make_redirect_getattr(name, redirect)  # type: ignore[attr-defined]
            return

        # Removed submodule (computer, executors, filesystem, ...): resolve names
        # lazily (computer marker / no-op).
        module.__path__ = []
        module.__getattr__ = _make_legacy_getattr(name)  # type: ignore[attr-defined]


def install() -> None:
    if not any(isinstance(f, _V5CompatFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _V5CompatFinder())
