"""Experimental Harbor task interop.

``adapt()`` resolves Harbor images and packages task directories as runnable
HUD tasksets and conventional Compose projects.
``export()`` writes HUD tasks back to Harbor directories.

This API may change between minor releases while the integration is experimental.
"""

from .adapt import AdaptFailure, AdaptFinding, AdaptResult, adapt
from .export import export

__all__ = ["AdaptFailure", "AdaptFinding", "AdaptResult", "adapt", "export"]
