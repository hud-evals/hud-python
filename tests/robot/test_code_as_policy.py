"""Code as Policy: a tool-calling agent drives a robot sim by running programs over primitives.

Each case is a whole rollout, as in direct control's tests: an in-process bridge, an env
publishing :class:`CodeAsPolicy`'s ``run`` tool, and a scripted agent sending programs, graded
by the sim. Every case asserts what the sim was commanded and what the model was told.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hud.environment.robot import CodeAsPolicy
from hud.eval import LocalRuntime, Task, rollout

from .sims import Arm, Constant, Overhead, PassThrough, ScriptedLLM, arm_env, contract, reached

if TYPE_CHECKING:
    from hud.eval.run import Run

#: An absolute ``[x, grip]`` action, x in [0, 1].
REACH = contract("ee_abs", [0.0, -1.0], [1.0, 1.0])


def program(code: str) -> tuple[str, dict[str, Any]]:
    return "run", {"code": code, "note": "run the plan"}


async def drive(
    sim: Arm, control: CodeAsPolicy, *calls: tuple[str, dict[str, Any]]
) -> tuple[ScriptedLLM, Run]:
    agent = ScriptedLLM(*calls)
    async with arm_env(sim, control) as env:
        run = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))
    return agent, run


async def test_one_program_composes_the_motions_that_solve_the_task() -> None:
    sim = Arm(REACH, done=reached)

    agent, run = await drive(
        sim,
        CodeAsPolicy(max_step={"grip": 2.0}),
        program("first = move_to(x=0.5)\nprint('reached', first.reached)\nmove_to(grip=1.0)"),
    )

    assert agent.tools == {"run"}
    assert run.reward == 1.0  # graded by the sim
    # The same motions direct control plays: 50 ticks to x=0.5, 6 to settle, 1 to close.
    np.testing.assert_allclose(np.array(sim.actions[:50])[:, 0], np.linspace(0.01, 0.5, 50))
    np.testing.assert_allclose(sim.actions[-1], [0.5, 1.0])
    (reply,) = agent.replies()
    # The trace viewer parses this opening line to map tool calls to ticks.
    assert reply.startswith("Played 57 steps (5.7 s).")
    assert "move_to(x=0.5) -> At the target. (5.6 s)" in reply
    assert "reached True" in reply
    assert "The episode has ended" in reply


async def test_an_error_stops_the_program_where_it_failed_and_says_where() -> None:
    sim = Arm(REACH)

    agent, _ = await drive(sim, CodeAsPolicy(), program("move_to(x=0.2)\nmove_to(x=1.5)"))

    # The first move played; the out-of-range one never reached the sim.
    assert max(action[0] for action in sim.actions) <= 0.2 + 1e-6  # float32 on the wire
    (reply,) = agent.replies()
    assert "move_to(x=0.2) -> At the target." in reply
    assert "line 2, in <module>" in reply
    assert "x=1.5 is outside [0, 1]" in reply


async def test_variables_persist_from_one_run_to_the_next() -> None:
    sim = Arm(REACH)

    await drive(sim, CodeAsPolicy(), program("target = 0.3"), program("move_to(x=target)"))

    np.testing.assert_allclose(sim.state[0], 0.3)


async def test_a_program_past_its_time_is_stopped_and_the_robot_holds() -> None:
    sim = Arm(REACH)

    agent, _ = await drive(
        sim, CodeAsPolicy(run_timeout=0.5), program("while True:\n    wait(0.1)")
    )

    (reply,) = agent.replies()
    assert "Stopped: the program ran past 0.5 s." in reply
    # Holding is all it ever commanded.
    assert all(action[0] == 0.0 for action in sim.actions)


async def test_a_program_that_exits_gets_an_error_and_the_env_keeps_running() -> None:
    sim = Arm(REACH)

    agent, _ = await drive(
        sim, CodeAsPolicy(), program("import sys\nsys.exit(0)"), program("move_to(x=0.3)")
    )

    assert "cannot exit the env" in agent.replies()[0]
    np.testing.assert_allclose(sim.state[0], 0.3)  # the next program still ran


async def test_once_the_episode_ends_every_primitive_raises_episode_ended() -> None:
    sim = Arm(REACH, done=reached)

    agent, _ = await drive(
        sim,
        CodeAsPolicy(max_step={"grip": 2.0}),
        program("move_to(x=0.5)\nmove_to(grip=1.0)"),
        program(
            "try:\n    move_to(x=0.1)\nexcept EpisodeEndedError:\n    print('over', flush=True)"
        ),
    )

    assert "over" in agent.replies()[1]
    assert max(action[0] for action in sim.actions) <= 0.5 + 1e-6  # nothing moved after the end


async def test_memory_functions_drive_the_robot_from_the_first_program() -> None:
    sim = Arm(REACH)
    memory = 'def reach(x):\n    """Reach x and say whether it got there."""\n'
    memory += "    return move_to(x=x).reached\n"

    agent, _ = await drive(sim, CodeAsPolicy(memory=memory), program("print('got', reach(0.4))"))

    np.testing.assert_allclose(sim.state[0], 0.4)
    assert "got True" in agent.replies()[0]


async def test_a_program_hands_a_step_to_the_policy_and_finishes_the_task() -> None:
    sim = Arm(REACH, done=reached)
    policy = Constant([[0.5, 1.0]] * 5)  # what the policy was trained to do from anywhere

    agent, run = await drive(
        sim,
        CodeAsPolicy(model=policy, adapter=PassThrough()),
        program("move_to(x=0.2)\nrun_policy('reach', 1.0)"),
    )

    assert run.reward == 1.0  # graded by the sim
    np.testing.assert_allclose(sim.actions[-1], [0.5, 1.0])  # the policy's own action
    assert "run_policy('reach', 1) -> done" in agent.replies()[0]


async def test_a_program_places_a_pixel_in_the_world_from_the_depth_image() -> None:
    sim = Overhead(REACH)

    agent, _ = await drive(
        sim, CodeAsPolicy(), program("print(observe().cameras['image'].point(7.5, 3.5))")
    )

    assert not agent.results[0].isError
    assert "[1. 0. 0.]" in agent.replies()[0].replace("-0.", "0.")
