"""Policy Supervision: a tool-calling agent directs a policy that drives a robot sim in parallel.

Each case is a whole rollout, as in direct control's tests: an in-process bridge, an env
publishing :class:`PolicySupervision`'s tools, a stub policy and a scripted agent, graded by
the sim. Every case asserts what the sim was commanded or what the agent got back. The sim
runs 20 times faster than real time, so seconds of it pass in a fraction of one.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.agents.robot import Adapter, Model
from hud.environment.robot import PolicySupervision
from hud.eval import LocalRuntime, Task, rollout

from .sims import (
    Arm,
    Constant,
    Overhead,
    PassThrough,
    ScriptedLLM,
    arm_env,
    contract,
    look,
    move,
    reached,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hud.eval.run import Run

#: An absolute ``[x, grip]`` action, x in [0, 1].
REACH = contract("ee_abs", [0.0, -1.0], [1.0, 1.0])


#: Sim seconds per wall second in these tests.
FAST = 20.0


def run_policy(instruction: str = "reach") -> tuple[str, dict[str, Any]]:
    return "run_policy", {"instruction": instruction, "note": "let the policy act"}


def supervise(policy: Constant, check_in: float = 5.0, **options: Any) -> PolicySupervision:
    return PolicySupervision(policy, PassThrough(), time_scale=FAST, check_in=check_in, **options)


def runs(actions: list[Any], values: list[float], column: int = 0) -> bool:
    """Whether *values* were commanded on consecutive ticks, in *column*."""
    played = [round(float(a[column]), 6) for a in actions]
    wanted = [round(v, 6) for v in values]
    return any(played[i : i + len(wanted)] == wanted for i in range(len(played)))


async def drive(
    sim: Arm, control: PolicySupervision, *calls: tuple[str, dict[str, Any]]
) -> tuple[ScriptedLLM, Run]:
    agent = ScriptedLLM(*calls)
    async with arm_env(sim, control) as env:
        run = await asyncio.wait_for(
            rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env)), 60
        )
    return agent, run


async def test_the_policy_acts_in_parallel_while_the_agent_waits() -> None:
    sim = Arm(REACH, done=reached)
    policy = Constant([[0.5, 1.0]] * 5)

    agent, run = await drive(sim, supervise(policy), run_policy(), look(2.0))

    assert agent.tools == {"run_policy", "stop_policy", "wait"}  # it only directs, by default
    assert "The policy is acting on 'reach'" in agent.replies()[0]  # it returned at once
    assert run.reward == 1.0  # the policy got there while the agent waited
    assert runs(sim.actions, [1.0], column=1)


async def test_the_policy_acts_until_stopped_then_the_arm_holds() -> None:
    sim = Arm(REACH)
    policy = Constant([[0.1, -1.0], [0.2, -1.0], [0.3, -1.0]])

    agent, _ = await drive(
        sim,
        supervise(policy),
        run_policy(),
        look(1.0),
        ("stop_policy", {"note": "enough"}),
        look(1.0),
    )

    assert runs(sim.actions, [0.1, 0.2, 0.3, 0.1, 0.2, 0.3])  # chunk after chunk
    assert "The policy is stopped; the arm holds." in agent.replies()[2]
    held = np.array(sim.actions[-10:])  # the last second: one pose, held
    assert np.all(held == held[0])


class Prompted(Adapter):
    """Hands the policy the instruction it is told."""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> Any:
        return prompt


class OneQueryAtATime(Model):
    """A slow policy whose client, like a websocket's, fails on a concurrent query; it moves x
    to 0.1 for 'left' and 0.9 for 'right', two actions a chunk."""

    def __init__(self) -> None:
        self.busy = False

    def infer(self, batch: Any) -> NDArray[np.float64]:
        if self.busy:
            raise RuntimeError("cannot query while another query is running")
        self.busy = True
        time.sleep(0.2)
        self.busy = False
        x = 0.1 if batch == "left" else 0.9
        return np.array([[[x, -1.0], [x, -1.0]]])


async def test_a_new_instruction_waits_for_the_query_in_flight_and_its_chunk_is_dropped() -> None:
    sim = Arm(REACH)
    control = PolicySupervision(OneQueryAtATime(), Prompted(), time_scale=FAST, check_in=5.0)

    agent, _ = await drive(
        sim, control, run_policy("left"), look(1.0), run_policy("right"), look(1.0)
    )

    assert not any(result.isError for result in agent.results)
    played = [round(float(action[0]), 6) for action in sim.actions]
    switch = played.index(0.9)
    assert 0.1 not in played[switch:]  # no chunk of the old instruction after the new one


async def test_a_policy_that_returns_no_actions_is_an_error_not_a_hang() -> None:
    sim = Arm(REACH)
    policy = Constant([])

    agent, _ = await drive(sim, supervise(policy), run_policy())

    assert agent.results[0].isError
    assert "no actions" in agent.replies()[0]


async def test_the_time_limit_ends_the_episode_while_the_agent_waits() -> None:
    sim = Arm(REACH)
    policy = Constant([[0.1, -1.0]] * 3)

    agent, _ = await drive(sim, supervise(policy, time_limit=1.0), run_policy(), look(5.0))

    assert "The episode has ended" in agent.replies()[1]
    assert len(sim.actions) == 10  # 1 s at 10 Hz, and no more


async def test_detect_places_each_match_in_the_world_from_the_depth_image() -> None:
    sim = Overhead(REACH)

    def detector(frame: Any, prompt: str) -> list[dict[str, Any]]:
        return [{"label": prompt, "score": 0.9, "box": [6, 2, 9, 5]}]

    agent, _ = await drive(
        sim,
        supervise(Constant([[0.1, -1.0]]), detector=detector),
        ("detect", {"camera": "observation/image", "prompt": "cube", "note": "where is it"}),
    )

    assert "centre (8, 4) at (1.000, 0.000, 0.000) m in the world" in agent.replies()[0].replace(
        "-0.000", "0.000"
    )


def servo_x(target: Any) -> Any:
    """A planner for ``[x, grip]``: x a tenth of the way to the target each tick, grip held."""

    def plan(obs: dict[str, Any]) -> Any:
        x, grip = np.asarray(obs["data"]["observation/state"], dtype=np.float64)
        if abs(target[0] - x) < 1e-3:
            return np.zeros((0, 2)), None, set()
        return np.array([[x + (target[0] - x) / 10, grip]]), None, set()

    return plan


async def test_reach_stops_the_policy_and_servos_to_the_point() -> None:
    sim = Arm(REACH)
    policy = Constant([[0.1, -1.0]] * 3)

    agent, _ = await drive(
        sim,
        supervise(policy, reach=servo_x),
        run_policy(),
        look(1.0),
        ("reach", {"x": 0.6, "y": 0.0, "z": 0.0, "note": "go to the cube"}),
    )

    assert "The policy is stopped; the arm holds." in agent.replies()[2]
    assert sim.state[0] == pytest.approx(0.6, abs=1e-3)


async def test_the_agent_takes_over_to_fix_what_the_policy_got_wrong() -> None:
    sim = Arm(REACH, done=reached)
    policy = Constant([[0.2, 1.0]] * 2)  # closes the gripper short of the goal

    agent, run = await drive(
        sim, supervise(policy, use_motion=True), run_policy(), look(1.0), move("move_to", x=0.5)
    )

    assert {"run_policy", "move_to"} <= agent.tools
    assert run.reward == 1.0  # the agent's move finished what the policy started
    assert runs(sim.actions, [0.2, 0.2])  # the policy's, before the agent took over
