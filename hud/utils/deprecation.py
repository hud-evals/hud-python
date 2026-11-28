"""Deprecation warnings for what hud v0.7.0 removes or changes."""

from __future__ import annotations

import warnings

from hud.utils.exceptions import HudDeprecationWarning


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
