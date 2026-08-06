"""Unit regressions for robot bridge, endpoint claims, gym shapes, and spawn state."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
from typing_extensions import override

from hud.environment.env import current_session_id
from hud.environment.robot.bridge import _HUD_STATE, RobotBridge, _apply_declaration_state
from hud.environment.robot.endpoint import RobotEndpoint, _bridge_init_kwargs
from hud.environment.robot.gym import GymBridge, action_dim_of

if TYPE_CHECKING:
    from numpy.typing import NDArray

# --- spawn state (declaration → child) -------------------------------------------------


class _CustomBridge(RobotBridge):
    """Subclass with a real ctor param — contract is assigned after init (docs style)."""

    def __init__(self, *, use_delta: bool = False) -> None:
        super().__init__()
        self.use_delta = use_delta

    @override
    def reset(self, **kwargs: Any) -> str:
        return "p"

    @override
    def step(self, action: Any) -> None:
        return None

    @override
    def get_observation(self) -> tuple[dict[str, Any], Any] | None:
        return None


def test_bridge_init_kwargs_packs_declaration_contract_and_ctor_params() -> None:
    bridge = _CustomBridge(use_delta=True)
    bridge.contract = {
        "control_rate": 10,
        "features": {"action": {"role": "action", "names": ["a"]}},
    }
    bridge.num_envs = 4
    bridge.metadata = {"backend": "test"}

    kwargs = _bridge_init_kwargs(bridge)
    assert kwargs["use_delta"] is True
    assert kwargs[_HUD_STATE]["contract"] == bridge.contract
    assert kwargs[_HUD_STATE]["num_envs"] == 4
    assert kwargs[_HUD_STATE]["metadata"] == {"backend": "test"}
    # Bind address stays with the child — never forwarded.
    assert "host" not in kwargs and "port" not in kwargs


def test_bridge_init_kwargs_skips_empty_default_state() -> None:
    bridge = _CustomBridge()
    kwargs = _bridge_init_kwargs(bridge)
    assert kwargs == {"use_delta": False}
    assert _HUD_STATE not in kwargs


def test_apply_declaration_state_sets_contract_after_ctor() -> None:
    """Child reconstructs with ctor kwargs only; state is applied afterward."""
    bridge = _CustomBridge(use_delta=True)
    assert bridge.contract == {}
    _apply_declaration_state(
        bridge,
        {
            "contract": {"features": {"action": {"role": "action", "names": ["a"]}}},
            "num_envs": 2,
            "metadata": {"k": 1},
        },
    )
    assert bridge.contract["features"]["action"]["names"] == ["a"]
    assert bridge.num_envs == 2
    assert len(bridge._registry.slots) == 2
    assert bridge.metadata == {"k": 1}


# --- gym shapes ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("single_shape",),
    [
        pytest.param((2, 2), id="keeps_single_action_space_shape"),
        pytest.param(None, id="strips_batched_action_space_without_single"),
    ],
)
def test_action_dim_of_batched(single_shape: tuple[int, ...] | None) -> None:
    env = SimpleNamespace(
        single_action_space=None if single_shape is None else SimpleNamespace(shape=single_shape),
        action_space=SimpleNamespace(shape=(4, 2, 2)),
    )
    assert action_dim_of(env, batched=True) == 4


def test_var_keyword_factory_defaults_are_build_params() -> None:
    def make_env(**kwargs: Any) -> Any:
        return kwargs

    bridge = GymBridge(make_env, contract=None, num_envs=8)
    assert "num_envs" in bridge._build_params


def test_step_reshapes_float_box_action() -> None:
    class _FakeEnv:
        action_space = SimpleNamespace(shape=(2, 2), dtype=np.float32)

        def __init__(self) -> None:
            self.last_action: Any = None

        def step(self, action: Any) -> tuple[Any, ...]:
            self.last_action = action
            return np.zeros(3, dtype=np.float32), 0.0, False, False, {}

    bridge = GymBridge(lambda: None, contract=None)
    bridge.env = _FakeEnv()
    bridge.batched = False
    bridge.num_envs = 1
    bridge._done = np.zeros(1, dtype=bool)
    bridge._success = np.zeros(1, dtype=bool)
    bridge._acc_reward = np.zeros(1)
    bridge._step_reward = np.zeros(1)
    bridge.step(np.arange(4, dtype=np.float32).reshape(1, 4))
    assert bridge.env.last_action.shape == (2, 2)


def test_step_reshapes_batched_float_box_action() -> None:
    class _FakeEnv:
        action_space = SimpleNamespace(shape=(2, 2, 2), dtype=np.float32)

        def __init__(self) -> None:
            self.last_action: Any = None

        def step(self, action: Any) -> tuple[Any, ...]:
            self.last_action = action
            return (
                np.zeros((2, 3), dtype=np.float32),
                np.zeros(2),
                np.zeros(2, dtype=bool),
                np.zeros(2, dtype=bool),
                {},
            )

    bridge = GymBridge(lambda: None, contract=None)
    bridge.env = _FakeEnv()
    bridge.batched = True
    bridge.num_envs = 2
    bridge._done = np.zeros(2, dtype=bool)
    bridge._success = np.zeros(2, dtype=bool)
    bridge._acc_reward = np.zeros(2)
    bridge._step_reward = np.zeros(2)
    bridge.step(np.arange(8, dtype=np.float32).reshape(2, 4))
    assert bridge.env.last_action.shape == (2, 2, 2)


def test_plain_env_observation_always_gets_batch_axis() -> None:
    bridge = GymBridge(lambda: None, contract=None)
    bridge.env = object()
    bridge.batched = False
    bridge.num_envs = 1
    bridge._done = np.zeros(1, dtype=bool)
    bridge._step_reward = np.array([0.5], dtype=np.float32)
    bridge._obs = {
        "state": np.array([1.0], dtype=np.float32),
        "camera": np.zeros((1, 4, 3), dtype=np.uint8),
    }
    observation = bridge.get_observation()
    assert observation is not None
    data, terminated = observation
    assert data is not None
    assert data["state"].shape == (1, 1)
    assert data["camera"].shape == (1, 1, 4, 3)
    assert data["reward"].shape == (1,)
    assert data["reward"][0] == pytest.approx(0.5)
    assert terminated.shape == (1,)


# --- endpoint session claims -----------------------------------------------------------


@pytest.fixture
def endpoint() -> RobotEndpoint:
    ep = RobotEndpoint.remote("127.0.0.1", 9)
    setattr(ep, "_call", AsyncMock())
    return ep


@pytest.mark.asyncio
async def test_release_claim_frees_current_session_slot(endpoint: RobotEndpoint) -> None:
    call = cast("AsyncMock", endpoint._call)
    call.return_value = {"prompt": "p", "token": "slot-0-abcd"}
    token = current_session_id.set("sess-a")
    try:
        await endpoint.reset()
        assert endpoint._claims["sess-a"] == "slot-0-abcd"
        call.return_value = {"score": 0.0}
        await endpoint.release_claim()
        assert "sess-a" not in endpoint._claims
        call.assert_called_with("result", {"token": "slot-0-abcd"})
    finally:
        current_session_id.reset(token)


@pytest.mark.asyncio
async def test_release_claim_on_shutdown_frees_each_session(endpoint: RobotEndpoint) -> None:
    call = cast("AsyncMock", endpoint._call)
    endpoint._claims["sess-a"] = "tok-a"
    endpoint._claims["sess-b"] = "tok-b"
    call.return_value = {"score": 0.0}

    for sid in ("sess-a", "sess-b"):
        token = current_session_id.set(sid)
        try:
            await endpoint.release_claim()
        finally:
            current_session_id.reset(token)

    assert endpoint._claims == {}
    tokens = [c.args[1]["token"] for c in call.call_args_list]
    assert sorted(tokens) == ["tok-a", "tok-b"]


@pytest.mark.asyncio
async def test_result_marks_claim_freed_so_disconnect_does_not_re_result(
    endpoint: RobotEndpoint,
) -> None:
    call = cast("AsyncMock", endpoint._call)
    token = current_session_id.set("sess-a")
    try:
        endpoint._claims["sess-a"] = "tok-a"
        call.return_value = {"score": 1.0, "success": True, "total_reward": 1.0}
        await endpoint.result(token="tok-a")
        assert endpoint._claims["sess-a"] == ""
        call.reset_mock()
        await endpoint.release_claim()
        call.assert_not_called()
    finally:
        current_session_id.reset(token)


@pytest.mark.asyncio
async def test_failed_release_rpc_retries_then_succeeds(endpoint: RobotEndpoint) -> None:
    call = cast("AsyncMock", endpoint._call)
    token = current_session_id.set("sess-a")
    try:
        endpoint._claims["sess-a"] = "tok-a"
        call.side_effect = [
            ConnectionError("sim down"),
            {"score": 0.0},
        ]
        await endpoint.release_claim()
        assert "sess-a" not in endpoint._claims
        assert call.call_count == 2
    finally:
        current_session_id.reset(token)


@pytest.mark.asyncio
async def test_stop_drains_claims_that_cancel_failed_to_free(endpoint: RobotEndpoint) -> None:
    call = cast("AsyncMock", endpoint._call)
    endpoint._claims["sess-a"] = "tok-a"
    call.side_effect = [
        ConnectionError("sim down"),
        ConnectionError("sim down"),
        ConnectionError("sim down"),
    ]
    token = current_session_id.set("sess-a")
    try:
        await endpoint.release_claim()  # retries exhausted — claim kept
        assert endpoint._claims["sess-a"] == "tok-a"
    finally:
        current_session_id.reset(token)

    call.side_effect = None
    call.return_value = {"score": 0.0}
    await endpoint.stop()  # last chance while the control link is up
    assert endpoint._claims == {}


# --- bridge barrier / claim ------------------------------------------------------------


class _StubBridge(RobotBridge):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.steps: list[NDArray[Any]] = []
        self.contract = {"features": {"action": {"role": "action", "names": ["a"]}}}

    @override
    def reset(self, **kwargs: Any) -> str:
        return f"prompt:{kwargs!r}"

    @override
    def step(self, action: NDArray[Any]) -> None:
        self.steps.append(np.asarray(action))

    @override
    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]] | None:
        return {"x": np.zeros((self.num_envs, 1), dtype=np.float32)}, np.zeros(
            self.num_envs, dtype=bool
        )


class _FakeWS:
    async def send(self, _data: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_claim_awaits_legacy_async_reset() -> None:
    bridge = _StubBridge()

    async def async_reset(**kwargs: Any) -> str:
        await asyncio.sleep(0)
        return "async-prompt"

    setattr(bridge, "reset", async_reset)
    ep = await bridge._claim_episode()
    assert ep["prompt"] == "async-prompt"


@pytest.mark.asyncio
async def test_claim_rejects_empty_kwargs_after_nonempty_batch() -> None:
    bridge = _StubBridge()
    bridge.num_envs = 2
    first = await bridge._claim_episode(task="A")
    assert first["token"]
    with pytest.raises(ValueError, match="identical args"):
        await bridge._claim_episode()


@pytest.mark.asyncio
async def test_release_uses_overridden_result() -> None:
    class _CustomGrade(_StubBridge):
        @override
        def result(self) -> dict[str, Any]:
            return {"score": 0.42, "success": True, "total_reward": 3.0, "detail": "custom"}

    bridge = _CustomGrade()
    ep = await bridge._claim_episode()
    grade = await bridge._release_episode(ep["token"])
    assert grade == {"score": 0.42, "success": True, "total_reward": 3.0, "detail": "custom"}


@pytest.mark.asyncio
async def test_tick_loop_does_not_hold_spin_when_all_claimed_idle() -> None:
    """Terminated slots may keep WS open until close(); barrier must not step."""
    bridge = _StubBridge()
    bridge.num_envs = 1
    bridge._registry.configure(1)
    slot = bridge._registry.slots[0]
    bridge._registry.claim(slot)
    slot.ws = _FakeWS()  # still connected
    slot.idle = True  # terminated
    slot.action = None

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert bridge.steps == []


@pytest.mark.asyncio
async def test_tick_loop_stops_after_terminal_obs_while_ws_still_open() -> None:
    """Terminate wakes the barrier; idle+open WS must not cascade hold-steps.

    The e2e gate ``sim.tick == EPISODE_TICKS`` fails if each terminal fan-out
    re-arms the loop into a hold-spin before the agent closes the socket.
    """

    class _Terminating(_StubBridge):
        def __init__(self) -> None:
            super().__init__()
            self.tick = 0

        @override
        def step(self, action: NDArray[Any]) -> None:
            self.tick += 1
            super().step(action)

        @override
        def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]] | None:
            data = {"x": np.zeros((self.num_envs, 1), dtype=np.float32)}
            return data, np.array([self.tick >= 3])

    bridge = _Terminating()
    bridge.num_envs = 1
    bridge._registry.configure(1)
    slot = bridge._registry.slots[0]
    bridge._registry.claim(slot)
    slot.ws = _FakeWS()
    task = asyncio.create_task(bridge._tick_loop())
    try:
        for expected in (1, 2, 3):
            slot.idle = False
            slot.action = np.array([1.0], dtype=np.float32)
            bridge._action_event.set()
            for _ in range(50):
                if bridge.tick >= expected:
                    break
                await asyncio.sleep(0.01)
            assert bridge.tick == expected
        assert slot.idle  # terminal obs dropped the slot out of the barrier
        for _ in range(20):  # spam the wake that terminal fan-out itself sets
            bridge._action_event.set()
            await asyncio.sleep(0.005)
        assert bridge.tick == 3
        assert len(bridge.steps) == 3
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_claim_raises_when_batch_full() -> None:
    bridge = _StubBridge()
    bridge.num_envs = 1
    bridge._registry.configure(1)
    first = await bridge._claim_episode(goal="a")
    with pytest.raises(RuntimeError, match="slots are claimed"):
        await bridge._claim_episode(goal="a")
    await bridge._release_episode(first["token"])
    second = await bridge._claim_episode(goal="a")
    assert second["token"]


@pytest.mark.asyncio
async def test_endpoint_reset_retries_until_peer_result_frees_slot() -> None:
    """Shared width+1: reset retries outside the RPC lock so result is not deadlocked."""
    bridge = _StubBridge()
    bridge.num_envs = 1
    server = await bridge.serve_control("127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    ep = RobotEndpoint.remote("127.0.0.1", port)
    await ep.start()
    try:
        first = await ep.reset(goal="a")
        waiting = asyncio.create_task(ep.reset(goal="a"))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        await asyncio.wait_for(ep.result(token=first["token"]), timeout=1.0)
        second = await asyncio.wait_for(waiting, timeout=1.0)
        assert second["token"] != first["token"]
    finally:
        await ep.stop()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_tick_loop_times_out_silent_live_slot() -> None:
    """A connected agent that never sends must not stall the barrier forever."""
    bridge = _StubBridge()
    bridge.step_timeout = 0.05
    bridge.num_envs = 2
    bridge._registry.configure(2)
    a, b = bridge._registry.slots
    bridge._registry.claim(a)
    bridge._registry.claim(b)
    a.ws, b.ws = _FakeWS(), _FakeWS()
    a.idle = b.idle = False
    a.action = np.array([1.0], dtype=np.float32)  # ready
    b.action = None  # silent

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    for _ in range(50):
        if bridge.steps:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert len(bridge.steps) == 1
    assert b.idle  # timed out


@pytest.mark.asyncio
async def test_tick_loop_does_not_hold_step_undialed_claimed_slot() -> None:
    """A peer that has claimed but not WS-connected must not be hold-advanced."""
    bridge = _StubBridge()
    bridge.step_timeout = 0.05
    bridge.num_envs = 2
    bridge._registry.configure(2)
    a, b = bridge._registry.slots
    bridge._registry.claim(a)
    bridge._registry.claim(b)
    a.ws = _FakeWS()
    a.idle = b.idle = False
    a.action = np.array([1.0], dtype=np.float32)
    # b: claimed, still dialing (ws is None)

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    await asyncio.sleep(0.15)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert bridge.steps == []
    assert not b.idle  # still waiting to dial — not timed out into hold
