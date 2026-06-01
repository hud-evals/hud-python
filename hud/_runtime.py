"""One-time activation of HUD's global runtime patches.

Importing ``hud`` or its data-model modules (e.g. ``hud.types``) must stay free
of global process mutations so the contract types can be reused by other
services without dragging in MCP monkey-patches, HTTP client instrumentation,
or a process-wide ``sys.excepthook``.

Those side effects are applied here exactly once, the first time the SDK
runtime is actually engaged -- an ``hud.eval(...)`` run, or importing the
environment / agents / server packages.
"""

from __future__ import annotations

import threading

_activated = False
_lock = threading.Lock()


def activate_runtime() -> None:
    """Apply HUD's global runtime patches exactly once.

    Idempotent and thread-safe, so every runtime entry point can call it
    unconditionally.
    """
    global _activated
    if _activated:
        return
    with _lock:
        if _activated:
            return
        from hud.eval.instrument import patch_http_clients
        from hud.patches import apply_all_patches
        from hud.utils.pretty_errors import install_pretty_errors

        apply_all_patches()
        patch_http_clients()
        install_pretty_errors()
        _activated = True
