"""Retired: v0.6.23's control-channel client package, served with a warning until hud v0.7.0.

``HudClient`` is the HUD format's session in ``hud.environment.client``, opened by
``runtime.connect()``; ``HudProtocolError`` is in ``hud.environment``; capability clients
register with ``hud.capabilities.register_client``.
"""

from __future__ import annotations

import warnings
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from hud.utils.deprecation import moved_to, resolve_retired
from hud.utils.exceptions import HudDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from hud.environment.egress import WorkspaceRoute
    from hud.eval.run import EnvironmentSession
    from hud.runtime.core import Runtime


@asynccontextmanager
async def connect(
    runtime: Runtime,
    *,
    ready_timeout: float | None = None,
    workspace_routes: Sequence[WorkspaceRoute] = (),
) -> AsyncGenerator[EnvironmentSession]:
    """Open a session on ``runtime``; deprecated, ``runtime.connect()`` opens it now.

    A runtime served over a control channel yields a ``HudClient``, as before.
    """
    warnings.warn(
        "`connect` from `hud.clients` is deprecated and will be removed in hud v0.7.0; "
        "open a session with `runtime.connect()`.",
        HudDeprecationWarning,
        stacklevel=3,
    )
    if ready_timeout is not None:
        runtime = replace(runtime, ready_timeout=ready_timeout)
    async with runtime.connect(workspace_routes=workspace_routes) as session:
        yield session


_RETIRED = {
    **moved_to("hud.capabilities", "register_client"),
    **moved_to("hud.environment", "HudProtocolError"),
    **moved_to("hud.environment.client", "HudClient", "Manifest"),
}


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)
