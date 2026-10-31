"""The sims robot scenarios run, the envs that serve them, and the agents that drive them.

A spawned sim re-imports its bridge class or env factory from this file by path,
so they live at module level. Each gym env reports, in its observation, the
shape of the last action ``step`` got.

:class:`Arm` is the direct-control sim: its observed state follows each
commanded action by a ``follow`` law, so one class plays every arm a settle
verdict is about. :class:`ScriptedLLM` stands in for a tool-calling model.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from gymnasium.spaces import Box

from hud.agents.base import Agent
from hud.environment import Environment
from hud.environment.robot import DirectControl, RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from numpy.typing import NDArray

    from hud.capabilities import MCPClient
    from hud.eval import Provider
    from hud.eval.run import Run
    from hud.types import ToolResult


@asynccontextmanager
async def sim_env(
    sim: RobotBridge, name: str, *, control: DirectControl | None = None
) -> AsyncIterator[tuple[Environment, RobotEndpoint]]:
    """The docs' custom-bridge env for *sim*, which this process serves, started.

    ``RobotEndpoint.remote`` attaches to a sim someone else runs; here that is
    the test, so the whole control plane is real without a spawned sim program.
    *control* adds direct control's motion tools. Each scenario declares the
    template it is about.
    """
    await sim.start()
    server = await sim.serve_control()
    env = Environment(name)
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1]).attach(env)
    if control is not None:
        control.attach(endpoint)

    @env.initialize
    async def up() -> None:
        await endpoint.start()
        for capability in await endpoint.capabilities():
            env.add_capability(capability)

    @env.shutdown
    async def down() -> None:
        await endpoint.stop()

    try:
        await env.start()
        yield env, endpoint
    finally:
        await env.stop()
        server.close()
        await sim.stop()


# ─── direct control ────────────────────────────────────────────────────


def contract(
    action_type: str,
    low: list[float],
    high: list[float],
    names: list[str] | None = None,
    *,
    rate: float = 10,
) -> dict[str, Any]:
    """One camera, a state, and an action of *action_type*. Default action is ``[x, grip]``."""
    labels = names or ["x", "grip"]
    return {
        "control_rate": rate,
        "features": {
            "observation/image": {"role": "observation", "type": "rgb"},
            "observation/state": {"role": "observation", "names": list(labels)},
            "action": {
                "role": "action",
                "type": action_type,
                "names": list(labels),
                "stats": {"min": low, "max": high},
            },
        },
    }


class Arm(RobotBridge):
    """An arm whose state becomes ``follow(arm, action)`` on every tick.

    It succeeds, which ends the episode, once ``done(state)`` holds. ``reads``
    maps the state to what the observation reports; ``side`` is the camera's.
    """

    def __init__(
        self,
        contract: dict[str, Any],
        home: list[float] | None = None,
        follow: Callable[[Arm, NDArray[np.float64]], NDArray[np.float64]] | None = None,
        *,
        done: Callable[[NDArray[np.float64]], bool] | None = None,
        reads: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        side: int = 8,
    ) -> None:
        super().__init__()
        self.contract = contract
        width = len(contract["features"]["action"]["names"])
        self.home = np.zeros(width) if home is None else np.asarray(home, dtype=np.float64)
        self.follow = follow or snap
        self.done = done
        self.reads = reads
        self.side = side
        self.state = self.home.copy()
        self.actions: list[NDArray[np.float64]] = []

    def reset(self, **task_args: Any) -> str:
        self.state = self.home.copy()
        self.actions.clear()
        return "move x to 0.5, then close the gripper"

    def step(self, action: NDArray[Any]) -> None:
        target = np.asarray(action[0], dtype=np.float64)
        self.actions.append(target.copy())
        self.state = self.follow(self, target)
        self.success = self.done is not None and self.done(self.state)

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        state = self.state if self.reads is None else self.reads(self.state)
        data = {
            "observation/image": np.zeros((1, self.side, self.side, 3), dtype=np.uint8),
            "observation/state": state[None].astype(np.float32),
        }
        return data, np.array([self.success])


def snap(arm: Arm, target: NDArray[np.float64]) -> NDArray[np.float64]:
    return target


def displace(arm: Arm, target: NDArray[np.float64]) -> NDArray[np.float64]:
    return arm.state + target


def lag(share: float) -> Callable[[Arm, NDArray[np.float64]], NDArray[np.float64]]:
    """Closes *share* of the gap to the command each tick."""
    return lambda arm, target: arm.state + share * (target - arm.state)


def still(noise: float) -> Callable[[Arm, NDArray[np.float64]], NDArray[np.float64]]:
    """Never moves; its sensor reads *noise* high then low on alternate ticks."""
    return lambda arm, target: arm.home + noise * (-1) ** len(arm.actions)


def offset(by: list[float]) -> Callable[[Arm, NDArray[np.float64]], NDArray[np.float64]]:
    """Settles a fixed offset, in action columns, away from every command."""
    return lambda arm, target: target + np.asarray(by)


def delayed(ticks: int) -> Callable[[Arm, NDArray[np.float64]], NDArray[np.float64]]:
    """Follows each command *ticks* ticks late."""
    return lambda arm, target: arm.actions[-1 - ticks] if len(arm.actions) > ticks else arm.home


def fingers(arm: Arm, target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Two fingers close 5 cm per tick; the left one jams on an object at 0.3."""
    close = np.minimum(arm.state[1:] + 0.05, target[1:])
    close[0] = min(close[0], 0.3)
    return np.concatenate([arm.state[:1], close])


def reached(state: NDArray[np.float64]) -> bool:
    """The default task: hold ``x >= 0.5`` with the gripper closed."""
    return bool(state[0] >= 0.5 and state[1] >= 0.9)


class Sparse(dict[str, Any]):
    """The dimensions a scripted call sets, by dotted name; the model nulls the rest."""


def nulled(schema: dict[str, Any], values: Sparse) -> dict[str, Any]:
    """``values`` laid out as the schema's ``targets``: every dimension listed, unset ones null."""
    targets: dict[str, Any] = {}
    for key, prop in schema["properties"].items():
        if "anyOf" in prop:
            leaves = {leaf: values.get(f"{key}.{leaf}") for leaf in prop["anyOf"][0]["properties"]}
            targets[key] = leaves if any(v is not None for v in leaves.values()) else None
        else:
            targets[key] = values.get(key)
    return targets


class ScriptedLLM(Agent):
    """Stands in for a tool-calling LLM: plays fixed MCP calls, keeps their results."""

    def __init__(self, *calls: tuple[str, dict[str, Any]]) -> None:
        super().__init__()
        self.calls = calls
        self.tools: set[str] = set()
        self.schemas: dict[str, dict[str, Any]] = {}
        self.descriptions: dict[str, str] = {}
        self.results: list[ToolResult] = []

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        listed = await client.list_tools()
        self.tools = {tool.name for tool in listed}
        self.schemas = {tool.name: tool.inputSchema for tool in listed}
        self.descriptions = {tool.name: tool.description or "" for tool in listed}
        for name, arguments in self.calls:
            properties = self.schemas[name]["properties"]
            if isinstance(arguments.get("targets"), Sparse):
                arguments = {
                    **arguments,
                    "targets": nulled(properties["targets"], arguments["targets"]),
                }
            if "note" not in properties:
                arguments = {k: v for k, v in arguments.items() if k != "note"}
            self.results.append(await client.call_tool(name, arguments))
        run.trace.content = "done"

    def replies(self) -> list[str]:
        """Each result's text."""
        return [
            "\n".join(block.text for block in result.content if block.type == "text")
            for result in self.results
        ]


def move(tool: str, **values: float) -> tuple[str, dict[str, Any]]:
    return tool, {"targets": Sparse(values), "note": "move toward the goal"}


def look(duration: float | None = None) -> tuple[str, dict[str, Any]]:
    return "wait", {"duration": duration, "note": "look"}


@asynccontextmanager
async def arm_env(sim: RobotBridge, control: DirectControl) -> AsyncIterator[Environment]:
    """*sim* with direct control attached, as the ``arm`` env whose ``reach`` the sim grades."""
    async with sim_env(sim, "arm", control=control) as (env, endpoint):

        @env.template()
        async def reach(max_ticks: int | None = None) -> AsyncGenerator[Any, Any]:
            ep = await endpoint.reset(max_ticks=max_ticks)
            yield {"prompt": ep["prompt"]}
            yield await endpoint.result()

        yield env


# ─── policies ──────────────────────────────────────────────────────────

#: Ticks a stub sim runs before terminating.
EPISODE_TICKS = 3

#: A minimal contract, exactly as documented: one camera, a state vector, an action.
STUB_CONTRACT = {
    "control_rate": 5,
    "features": {
        "observation/image": {"role": "observation", "type": "rgb"},
        "observation/state": {"role": "observation", "names": ["slot", "tick"]},
        "action": {"role": "action", "names": ["a0", "a1"]},
    },
}


class StubSim(RobotBridge):
    """Cameras and a 2-D state, terminating after ``EPISODE_TICKS`` ticks.

    Slot ``i``'s state carries ``i + 1``, and its grade is the sum of the actions
    that slot received, so with an echoing policy the grade names the slot the
    connection actually drove.
    """

    def __init__(
        self, num_envs: int = 1, *, contract: dict[str, Any] = STUB_CONTRACT, side: int = 16
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.contract = contract
        self.side = side
        self.tick = 0
        self.executed = np.zeros(num_envs)

    def reset(self, **task_args: Any) -> str:
        self.tick = 0
        self.executed[:] = 0.0
        return f"stub task: {task_args.get('goal', 'none')}"

    def step(self, action: NDArray[Any]) -> None:
        self.tick += 1
        self.executed += np.asarray(action)[:, 0]  # one row per slot

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        slots = np.arange(1, self.num_envs + 1, dtype=np.float32)
        cameras = [
            name
            for name, feature in self.contract["features"].items()
            if feature.get("type") == "rgb"
        ]
        data: dict[str, NDArray[Any]] = {
            camera: np.full((self.num_envs, self.side, self.side, 3), self.tick, dtype=np.uint8)
            for camera in cameras
        }
        data["observation/state"] = np.stack(
            [slots, np.full(self.num_envs, self.tick, dtype=np.float32)], axis=1
        )
        return data, np.full(self.num_envs, self.tick >= EPISODE_TICKS)

    def result_slots(self) -> list[dict[str, Any]]:
        # Per-slot grades, never one batch-wide scalar: the token has to resolve.
        return [
            {"score": float(echoed), "success": bool(echoed), "total_reward": float(echoed)}
            for echoed in self.executed
        ]


@asynccontextmanager
async def stub_episode(sim: StubSim, name: str) -> AsyncIterator[Provider]:
    """*sim* as env *name*, whose ``episode`` the sim grades, placed in this process."""
    async with sim_env(sim, name) as (env, endpoint):

        @env.template()
        async def episode(goal: str = "lift") -> AsyncGenerator[Any, Any]:
            ep = await endpoint.reset(goal=goal)
            yield {"prompt": ep["prompt"]}  # one slot: the token is optional
            yield await endpoint.result()

        yield LocalRuntime(env)


# ─── spawned sims and gym envs ─────────────────────────────────────────


class Reach(RobotBridge):
    """A bridge with a constructor parameter; its contract is set after construction."""

    def __init__(self, *, use_delta: bool = False) -> None:
        super().__init__()
        self.use_delta = use_delta

    def reset(self, **task_args: Any) -> str:
        return f"reach with use_delta={self.use_delta}"

    def step(self, action: NDArray[Any]) -> None:
        del action

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        return {"x": np.zeros((self.num_envs, 1), dtype=np.float32)}, np.zeros(self.num_envs, bool)


class PlainEnv:
    """A single gym env with a ``Box(2, 2)`` action space."""

    action_space = Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)

    def __init__(self) -> None:
        self.received: tuple[int, ...] = ()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        return self.observation(), {}

    def step(self, action: NDArray[Any]) -> tuple[Any, ...]:
        self.received = np.shape(action)
        return self.observation(), 0.5, False, False, {}

    def observation(self) -> dict[str, NDArray[Any]]:
        return {
            "received": np.array(self.received or (0,), dtype=np.float32),
            "camera": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def close(self) -> None:
        pass


class VectorEnv:
    """``num_envs`` envs stepped together, exposing the per-env ``single_action_space``."""

    single_action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.action_space = Box(-1.0, 1.0, shape=(num_envs, 2), dtype=np.float32)
        self.received: tuple[int, ...] = ()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        return self.observation(), {}

    def step(self, action: NDArray[Any]) -> tuple[Any, ...]:
        self.received = np.shape(action)
        rewards = np.full(self.num_envs, 0.25)
        done = np.zeros(self.num_envs, dtype=bool)
        return self.observation(), rewards, done, done, {}

    def observation(self) -> dict[str, NDArray[Any]]:
        received = np.array(self.received or (0, 0), dtype=np.float32)
        return {"received": np.tile(received, (self.num_envs, 1))}

    def close(self) -> None:
        pass


def plain_env() -> PlainEnv:
    return PlainEnv()


def vector_env(**build: Any) -> VectorEnv:
    """A factory taking its build args as ``**kwargs``, like many registry wrappers."""
    return VectorEnv(int(build["num_envs"]))
