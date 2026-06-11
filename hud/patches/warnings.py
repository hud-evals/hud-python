"""Centralized warning filters for noisy third-party dependencies.

Keep these helpers here so the rest of the codebase can stay clean and avoid
scattering warning filters across unrelated modules.
"""

from __future__ import annotations

import warnings


def suppress_known_import_warnings() -> None:
    """Filter third-party import-time noise the user can never act on.

    Called before anything imports fastmcp: its jwt provider imports
    ``authlib.jose``, which emits an ``AuthlibDeprecationWarning`` (a
    ``DeprecationWarning`` subclass) on every CLI launch.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"authlib\.jose module is deprecated",
        category=DeprecationWarning,
    )
