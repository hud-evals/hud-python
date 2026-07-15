"""hud.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

# Apply patches to third-party libraries early, before other imports
from . import patches as _patches  # noqa: F401
from ._legacy import install as _install_v5_compat
from .clients import connect
from .environment import Environment
from .eval import (
    Chat,
    DockerRuntime,
    Grade,
    HostedRuntime,
    HUDRuntime,
    Job,
    LocalRuntime,
    Run,
    Runtime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    SubprocessRuntime,
    SyncPlan,
    Task,
    Taskset,
)
from .telemetry.instrument import instrument
from .train import TrainingClient
from .types import Trace

_install_v5_compat()

__all__ = [
    "Chat",
    "DockerRuntime",
    "Environment",
    "Grade",
    "HUDRuntime",
    "HostedRuntime",
    "Job",
    "LocalRuntime",
    "Run",
    "Runtime",
    "RuntimeConfig",
    "RuntimeGPU",
    "RuntimeLimits",
    "RuntimeResources",
    "SubprocessRuntime",
    "SyncPlan",
    "Task",
    "Taskset",
    "Trace",
    "TrainingClient",
    "connect",
    "instrument",
]

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"


def _warn_legacy_hud_python() -> None:
    """Warn when a pre-rename hud-python install sits alongside this package.

    The SDK was published as ``hud-python`` before the rename to ``hud``.
    ``hud-python`` releases older than the rename ship their own copy of the
    ``hud`` package, so installing both makes the installer silently overwrite
    one with the other. Post-rename ``hud-python`` releases are empty shims
    that just depend on ``hud`` and are fine to have installed.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        legacy = _dist_version("hud-python")
        _dist_version("hud")
    except PackageNotFoundError:
        return

    from packaging.version import Version

    if Version(legacy) < Version("0.6.9"):
        import warnings

        warnings.warn(
            f"Both 'hud' and a pre-rename 'hud-python' ({legacy}) are installed; they ship "
            "the same 'hud' package and overwrite each other. Run "
            "'pip uninstall -y hud-python' and then 'pip install --force-reinstall "
            "--no-deps hud' (a plain reinstall is a no-op: pip already considers "
            "'hud' installed).",
            RuntimeWarning,
            stacklevel=3,
        )


_warn_legacy_hud_python()
