"""Env: declarative capabilities + scenarios behind the HUD wire protocol. Single-tenant."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import secrets
from typing import TYPE_CHECKING, Any

from .scenario import Scenario, ScenarioRunner
from .utils import error, read_frame, reply, send_frame

if TYPE_CHECKING:
    from collections.abc import Callable

    from .capability import Capability
    from .scenario import ScenarioFn

LOGGER = logging.getLogger("hud.env.env")


class Env:
    """Capabilities + scenarios dispatched over the HUD wire protocol."""

    def __init__(
        self,
        *,
        name: str,
        version: str = "0.0.1",
        capabilities: list[Capability] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.capabilities: list[Capability] = list(capabilities or [])
        self._scenarios: dict[str, Scenario] = {}

    # ─── scenario registration ───────────────────────────────────────────

    def scenario(
        self,
        *,
        id: str | None = None,
        description: str = "",
    ) -> Callable[[ScenarioFn], ScenarioFn]:
        """Register an async-generator scenario. ``id`` defaults to fn name."""

        def decorate(func: ScenarioFn) -> ScenarioFn:
            if not inspect.isasyncgenfunction(func):
                raise TypeError(
                    f"@env.scenario: {func.__qualname__} must be an async generator "
                    "function (`async def ...:` with `yield`)",
                )
            scenario_id = id or func.__name__
            if scenario_id in self._scenarios:
                raise ValueError(
                    f"scenario {scenario_id!r} already registered on env {self.name!r}",
                )
            self._scenarios[scenario_id] = Scenario(
                id=scenario_id,
                description=description,
                func=func,
            )
            return func

        return decorate

    def add_capability(self, cap: Capability) -> None:
        self.capabilities.append(cap)

    # ─── control-channel server ──────────────────────────────────────────

    async def serve(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Accept HUD control-channel connections; cap daemons must already be running."""
        server = await asyncio.start_server(self._handle_session, host=host, port=port)
        sock = server.sockets[0].getsockname()
        LOGGER.info("env %r listening on %s:%s", self.name, sock[0], sock[1])
        async with server:
            await server.serve_forever()

    # ─── per-connection protocol dispatch (transport-agnostic) ───────────

    async def _handle_session(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        session_id = "sess-" + secrets.token_hex(4)
        active_runner: ScenarioRunner | None = None

        async def reply_to(msg_id: int | None, result: dict[str, Any]) -> None:
            if msg_id is not None:
                await send_frame(writer, reply(msg_id, result))

        async def error_to(msg_id: int | None, code: int, message: str) -> None:
            if msg_id is not None:
                await send_frame(writer, error(msg_id, code, message))

        try:
            while True:
                msg = await read_frame(reader)
                if msg is None:
                    return

                method = msg.get("method", "")
                params = msg.get("params") or {}
                msg_id = msg.get("id")

                try:
                    if method == "hello":
                        await reply_to(
                            msg_id,
                            {
                                "session_id": session_id,
                                "env": {"name": self.name, "version": self.version},
                                "bindings": [c.manifest_entry() for c in self.capabilities],
                            },
                        )

                    elif method == "scenarios.list":
                        await reply_to(
                            msg_id,
                            {
                                "scenarios": [s.manifest_entry() for s in self._scenarios.values()],
                            },
                        )

                    elif method == "scenarios.start":
                        scenario_id = params.get("id")
                        if not isinstance(scenario_id, str):
                            await error_to(msg_id, -32602, "scenarios.start: 'id' must be a string")
                            continue
                        scenario = self._scenarios.get(scenario_id)
                        if scenario is None:
                            await error_to(msg_id, -32602, f"unknown scenario: {scenario_id!r}")
                            continue
                        args = params.get("args") or {}
                        if not isinstance(args, dict):
                            await error_to(
                                msg_id, -32602, "scenarios.start: 'args' must be an object"
                            )
                            continue
                        if active_runner is not None:
                            await active_runner.cancel()
                        active_runner = ScenarioRunner(scenario, args)
                        prompt = await active_runner.start()
                        await reply_to(msg_id, prompt)

                    elif method == "engage":
                        wanted = list(params.get("bindings", []))
                        known = {c.name for c in self.capabilities}
                        unknown = [b for b in wanted if b not in known]
                        if unknown:
                            await error_to(msg_id, -32602, f"unknown bindings: {unknown}")
                            continue
                        await reply_to(msg_id, {"engaged": sorted(set(wanted) & known)})

                    elif method == "scenarios.evaluate":
                        if active_runner is None:
                            await error_to(msg_id, -32600, "no scenario in progress")
                            continue
                        evaluation = await active_runner.evaluate(params)
                        active_runner = None
                        await reply_to(msg_id, evaluation)

                    elif method == "scenarios.cancel":
                        if active_runner is not None:
                            await active_runner.cancel()
                            active_runner = None
                        await reply_to(msg_id, {"cancelled": True})

                    elif method == "disengage":
                        await reply_to(
                            msg_id,
                            {
                                "disengaged": list(params.get("bindings", [])),
                            },
                        )

                    elif method == "bye":
                        await reply_to(msg_id, {"goodbye": True})
                        return

                    else:
                        await error_to(msg_id, -32601, f"method not found: {method}")

                except Exception as exc:
                    LOGGER.exception("error handling %s", method)
                    await error_to(msg_id, -32000, str(exc))

        finally:
            if active_runner is not None:
                with contextlib.suppress(Exception):
                    await active_runner.cancel()
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()
