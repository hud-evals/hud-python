"""Centralized warning filters for noisy third-party dependencies.

Keep these helpers here so the rest of the codebase can stay clean and avoid
scattering warning filters across unrelated modules.
"""

from __future__ import annotations

import warnings


def suppress_known_import_warnings() -> None:
    """Silence the one import-time warning the user can never act on.

    Called before anything imports fastmcp: its jwt provider imports
    ``authlib.jose``, which emits a single ``AuthlibDeprecationWarning`` on every
    CLI launch.

    ``authlib.deprecate`` runs ``warnings.simplefilter("always", ...)`` at import
    time, prepending an "always" filter that would otherwise sit ahead of ours and
    win (the warnings machinery applies the first matching filter). Import it first
    so our filter is prepended last and takes precedence; the offending
    ``authlib.jose`` import comes later via fastmcp.

    The filter is scoped to both the ``AuthlibDeprecationWarning`` class and the
    ``authlib.jose`` message, so it never hides any other warning -- not even other
    authlib deprecations.
    """
    try:
        from authlib.deprecate import AuthlibDeprecationWarning
    except ImportError:
        return  # no authlib installed -> no warning to suppress

    warnings.filterwarnings(
        "ignore",
        message=r"authlib\.jose module is deprecated",
        category=AuthlibDeprecationWarning,
    )
