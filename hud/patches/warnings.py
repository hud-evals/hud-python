"""
Centralized warning filters for noisy third-party dependencies.

Keep these helpers here so the rest of the codebase can stay clean and avoid
scattering warning filters across unrelated modules.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def apply_default_warning_filters(*, verbose: bool) -> None:
    """Apply our default warning filters for non-verbose CLI/server modes."""
    if verbose:
        return

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Pydantic v2 emits PydanticDeprecatedSince20 for v1-style config usage in deps.
    try:
        from pydantic.warnings import PydanticDeprecatedSince20
    except Exception:
        return

    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


@contextmanager
def suppress_mcp_use_import_warnings() -> Iterator[None]:
    """Suppress known noisy warnings emitted during `mcp_use` imports."""
    try:
        from pydantic.warnings import PydanticDeprecatedSince20
    except Exception:  # pragma: no cover
        PydanticDeprecatedSince20 = None  # type: ignore[assignment]

    with warnings.catch_warnings():
        # mcp_use currently emits DeprecationWarning from its package __init__.py.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"mcp_use(\..*)?$")

        # mcp_use currently defines Pydantic v1-style `class Config` in oauth models.
        if PydanticDeprecatedSince20 is not None:
            warnings.filterwarnings(
                "ignore",
                category=PydanticDeprecatedSince20,
                module=r"mcp_use\.client\.auth\.oauth$",
            )

        yield
