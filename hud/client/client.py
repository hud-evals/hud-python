"""HudClient: JSON-RPC client for the HUD wire protocol.

Pure transport — opens a TCP connection to an ``Env.serve()`` endpoint and
drives the ``hello`` / ``scenarios.list`` / ``scenarios.start`` /
``scenarios.evaluate`` / ``scenarios.cancel`` / ``bye`` methods. Returns the
parsed payloads; the caller (agent harness) does whatever it wants with them.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from typing import TYPE_CHECKING, Any, Self

from hud.capabilities import Capability
from hud.env.utils import read_frame, send_frame

from . import Manifest, ServerInfo

if TYPE_CHECKING:
    from types import TracebackType

LOGGER = logging.getLogger("hud.client")


class HudProtocolError(RuntimeError):
    """Raised when the env returns a JSON-RPC error frame."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"hud rpc error {code}: {message}")
        self.code = code
        self.message = message


class HudClient:
    """JSON-RPC client for an ``Env.serve()`` endpoint.

    Usage::

        async with HudClient.connect("127.0.0.1", 9001) as client:
            manifest = await client.hello()
            prompt = await client.start_scenario("write_hello")
            # ... run agent ...
            result = await client.evaluate({"submission": "..."})
    """

    PROTOCOL_VERSION = "hud/1.0"

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._ids = itertools.count(1)
        self._closed = False

    # ─── lifecycle ────────────────────────────────────────────────────

    @classmethod
    async def connect(cls, host: str = "127.0.0.1", port: int = 0) -> Self:
        reader, writer = await asyncio.open_connection(host, port)
        return cls(reader, writer)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._call("bye", {})
        except Exception:
            LOGGER.debug("bye failed (env may have already closed)", exc_info=True)
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()

    # ─── HUD methods ──────────────────────────────────────────────────

    async def hello(self) -> Manifest:
        """Send ``hello``; return the parsed ``Manifest``."""
        result = await self._call("hello", {})
        env = result.get("env") or {}
        bindings = [Capability.from_manifest(b) for b in (result.get("bindings") or [])]
        return Manifest(
            session_id=result["session_id"],
            protocol_version=self.PROTOCOL_VERSION,
            server_info=ServerInfo(
                name=env.get("name", "unknown"),
                version=env.get("version", "0.0.0"),
            ),
            bindings=bindings,
        )

    async def list_scenarios(self) -> list[dict[str, Any]]:
        """Return ``[{id, description}, ...]`` for every registered scenario."""
        result = await self._call("scenarios.list", {})
        scenarios = result.get("scenarios") or []
        if not isinstance(scenarios, list):
            raise HudProtocolError(-32603, "scenarios.list: 'scenarios' must be a list")
        return scenarios

    async def start_scenario(
        self,
        scenario_id: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a scenario; returns the first yield (``{"prompt": ...}``)."""
        return await self._call(
            "scenarios.start",
            {"id": scenario_id, "args": args or {}},
        )

    async def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send ``scenarios.evaluate``; returns the final evaluation dict."""
        return await self._call("scenarios.evaluate", payload)

    async def cancel(self) -> None:
        await self._call("scenarios.cancel", {})

    # ─── JSON-RPC plumbing ────────────────────────────────────────────

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        msg_id = next(self._ids)
        await send_frame(
            self._writer,
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params},
        )
        reply = await read_frame(self._reader)
        if reply is None:
            raise HudProtocolError(-32000, f"env closed connection during {method!r}")
        if "error" in reply:
            err = reply["error"]
            raise HudProtocolError(int(err.get("code", -32000)), str(err.get("message", "")))
        result = reply.get("result")
        if not isinstance(result, dict):
            raise HudProtocolError(-32603, f"{method!r}: result was not an object")
        return result


__all__ = ["HudClient", "HudProtocolError"]
