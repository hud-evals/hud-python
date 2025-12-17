"""Custom low-level MCP server that supports per-server initialization hooks.

This duplicates the upstream `mcp.server.lowlevel.server.Server.run` logic so we
can inject our own `InitSession` subtype without touching global state.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import anyio
import mcp.types as types
from fastmcp.server.low_level import LowLevelServer as _BaseLL
from mcp.server.lowlevel.server import (
    logger,
)
from mcp.server.session import ServerSession
from mcp.shared.context import RequestContext

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from mcp.server.models import InitializationOptions
    from mcp.shared.message import SessionMessage
    from mcp.shared.session import RequestResponder


class InitSession(ServerSession):
    """ServerSession that runs a one-time `init_fn(ctx)` on *initialize*."""

    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: MemoryObjectSendStream[SessionMessage],
        init_opts: InitializationOptions,
        *,
        init_fn: Callable[[RequestContext], Awaitable[None]] | None = None,
        stateless: bool = False,
    ) -> None:
        super().__init__(read_stream, write_stream, init_opts, stateless=stateless)
        self._init_fn = init_fn
        self._did_init = stateless  # skip when running stateless

    # pylint: disable=protected-access  # we need to hook into internal method
    async def _received_request(
        self,
        responder: RequestResponder[types.ClientRequest, types.ServerResult],
    ) -> types.ServerResult | None:
        # Intercept initialize
        if (
            isinstance(responder.request.root, types.InitializeRequest)
            and not self._did_init
            and self._init_fn is not None
        ):
            req = responder.request.root
            ctx = RequestContext[
                "ServerSession",
                dict[str, Any],
                types.InitializeRequest,
            ](
                request_id=req.id,  # type: ignore[attr-defined]
                meta=req.params.meta,
                session=self,
                lifespan_context={},
                request=req,
            )
            try:
                await self._init_fn(ctx)
            except Exception as exc:
                token = getattr(req.params.meta, "progressToken", None)
                if token is not None:
                    await self.send_progress_notification(
                        progress_token=token,
                        progress=0,
                        total=100,
                        message=f"Initialization failed: {exc}",
                    )
                raise
            finally:
                self._did_init = True
        # fall through to original behaviour
        return await super()._received_request(responder)


class LowLevelServerWithInit(_BaseLL):
    """LowLevelServer that uses :class:`InitSession` instead of `ServerSession`."""

    def __init__(
        self,
        fastmcp: Any,
        *args: Any,
        init_fn: Callable[[RequestContext], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(fastmcp, *args, **kwargs)
        self._init_fn = init_fn

    async def run(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: MemoryObjectSendStream[SessionMessage],
        initialization_options: InitializationOptions,
        *,
        raise_exceptions: bool = False,
        stateless: bool = False,
    ) -> None:
        """Copy of upstream run with InitSession injected."""

        async with AsyncExitStack() as stack:
            lifespan_context = await stack.enter_async_context(self.lifespan(self))
            session = await stack.enter_async_context(
                InitSession(
                    read_stream,
                    write_stream,
                    initialization_options,
                    stateless=stateless,
                    init_fn=self._init_fn,
                )
            )

            async with anyio.create_task_group() as tg:
                async for message in session.incoming_messages:
                    logger.debug("Received message: %s", message)

                    tg.start_soon(
                        self._handle_message,
                        message,
                        session,
                        lifespan_context,
                        raise_exceptions,
                    )
