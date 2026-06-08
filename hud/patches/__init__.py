"""HUD runtime patches for third-party libraries."""

from hud.patches.mcp_patches import apply_all_patches, suppress_fastmcp_logging
from hud.patches.warnings import apply_default_warning_filters, suppress_mcp_use_import_warnings

__all__ = [
    "apply_all_patches",
    "apply_default_warning_filters",
    "suppress_fastmcp_logging",
    "suppress_mcp_use_import_warnings",
]
