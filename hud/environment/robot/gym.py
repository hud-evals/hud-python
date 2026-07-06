"""``Gym`` — the declarative sim handle over a gym-style env factory.

The robotics analog of :class:`~hud.environment.workspace.Workspace`:
construction is pure data, ``start()`` materializes everything (env build,
contract derivation, the ``robot`` WebSocket, a JSON-RPC control endpoint for
split-process setups), and ``capability()`` mints the wire capability.
``env.gym(make_env)`` wires the lifecycle.

Templates drive episodes through the handle::

    sim = env.gym(make_env)


    @env.template(id="pawn_lift")
    async def pawn_lift(task: str = "solo_pawn_lift", seed: int = 0, num_envs: int = 1):
        yield {"prompt": await sim.reset(task=task, seed=seed, num_envs=num_envs)}
        yield await sim.result()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bridge import GymBridge
from .endpoint import RobotEndpoint
from .introspect import action_dim_of, detect_fps, load_or_write_contract

if TYPE_CHECKING:
    import asyncio

    from hud.capabilities import Capability

logger = logging.getLogger(__name__)


class Gym:
    """One gym-style sim as a HUD building block: contract, capability, episode control.

    Nothing is built at import — the factory runs at :meth:`start` (serve time).
    All control goes through the bridge's public surface via a local
    :class:`~.endpoint.RobotEndpoint`.
    """

    def __init__(
        self,
        factory: Any,
        *,
        fps: int | None = None,
        contract: str | Path | None = "contract.json",
        host: str = "127.0.0.1",
        port: int = 0,
        control_port: int = 9100,
    ) -> None:
        self._bridge = GymBridge(factory, host=host, port=port)
        self._endpoint = RobotEndpoint(self._bridge)
        self._fps = fps
        self._contract_path = contract
        self._contract: dict[str, Any] = {}
        self._control_host = host
        self._control_port = control_port
        self._control_server: asyncio.AbstractServer | None = None

    # ── lifecycle (driven by env.gym()'s hooks, or directly) ────────────────────

    async def start(self) -> None:
        """Build the env (factory defaults), derive/load the contract, bring up the wire.

        Also serves the bridge's JSON-RPC control endpoint so a split-process
        env can dial this process directly.
        """
        await self._bridge.ensure_env()
        state, frames = self._bridge.sample_observation()
        existed = self._contract_path is not None and Path(self._contract_path).exists()
        self._contract = load_or_write_contract(
            self._contract_path,
            state,
            frames,
            action_dim_of(self._bridge.env, batched=self._bridge.batched),
            self._fps or detect_fps(self._bridge.env),
        )
        if not existed and self._contract_path is not None:
            logger.info("gym: wrote %s (edit names to relabel plots)", self._contract_path)
        await self._bridge.start()
        if self._control_server is None:
            self._control_server = await self._endpoint.serve(
                self._control_host, self._control_port
            )

    async def stop(self) -> None:
        if self._control_server is not None:
            self._control_server.close()
            self._control_server = None
        await self._bridge.stop()  # also closes the built env

    def capability(self, name: str = "robot") -> Capability:
        """The concrete ``robot`` capability — mirrors ``Workspace.capability()``."""
        from hud.capabilities import Capability

        return Capability.robot(name=name, url=self._bridge.url, contract=self._contract)

    # ── the template surface ─────────────────────────────────────────────────────

    async def reset(self, **task_args: Any) -> str:
        """Start an episode (rebuilding the env if an env-defining arg changed);
        returns the task prompt."""
        return await self._endpoint.reset(**task_args)

    async def result(self) -> dict[str, Any]:
        """The episode grade: per-slot dicts under ``"slots"``, means at the top."""
        return await self._endpoint.result()

    @property
    def env(self) -> Any:
        """The live env — privileged sim access for custom grading in templates."""
        return self._bridge.env

    @property
    def contract(self) -> dict[str, Any]:
        return self._contract


__all__ = ["Gym"]
