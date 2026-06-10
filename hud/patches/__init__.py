"""
HUD runtime patches for third-party libraries.

This module applies monkey-patches to fix issues in dependencies
without requiring forked packages.
"""

from hud.patches.warnings import (
    apply_default_warning_filters,
    suppress_known_import_warnings,
    suppress_mcp_use_import_warnings,
)

# Filter import-time third-party noise before anything below pulls in fastmcp.
suppress_known_import_warnings()

from hud.patches.mcp_patches import apply_all_patches, suppress_fastmcp_logging  # noqa: E402

# Apply patches on import
apply_all_patches()

__all__ = [
    "apply_all_patches",
    "apply_default_warning_filters",
    "suppress_fastmcp_logging",
    "suppress_known_import_warnings",
    "suppress_mcp_use_import_warnings",
]
