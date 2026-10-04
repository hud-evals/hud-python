"""Direct control: a tool-calling agent drives a robot sim through MCP motion tools.

Each case is a whole rollout: an in-process bridge serving the ``openpi/0`` wire,
an env publishing it plus :class:`DirectControl`'s ``mcp`` capability, and a
scripted agent calling the tools through the tunneled manifest binding, graded
by the sim. Every case asserts what the sim was commanded and where the arm ended
up, with the tools and verdicts the model saw.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud import Task
from hud.agents.openai.tools.strict_schema import ensure_strict_json_schema
from hud.environment.robot import DirectControl
from hud.eval.run import rollout
from hud.runtime import LocalRuntime

from .sims import (
    Arm,
    ScriptedLLM,
    arm_env,
    contract,
    displace,
    lag,
    look,
    move,
    reached,
    still,
)

if TYPE_CHECKING:
    from hud import Trace

#: The default contract: an absolute ``[x, grip]`` action, x in [0, 1].
REACH = contract("ee_abs", [0.0, -1.0], [1.0, 1.0])


async def drive(
    sim: Arm, control: DirectControl, *calls: tuple[str, dict[str, Any]]
) -> tuple[ScriptedLLM, Trace]:
    agent = ScriptedLLM(*calls)
    async with arm_env(sim, control) as env:
        trace = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))
    return agent, trace


async def test_move_to_interpolates_absolute_targets_until_the_sim_succeeds() -> None:
    sim = Arm(REACH, done=reached)

    agent, run = await drive(
        sim,
        DirectControl(max_step={"grip": 2.0}),
        move("move_to", x=0.5),
        move("move_to", grip=1.0),
    )

    assert agent.tools == {"move_to", "wait"}  # the ee_abs contract picked the motion tool
    schema = agent.schemas["move_to"]
    assert ensure_strict_json_schema(copy.deepcopy(schema)) == schema
    assert run.reward == 1.0  # graded by the sim, not the tools
    # 0.1 of x's range per second at 10 Hz: 0.01 per tick, grip held at its reference.
    # The sim matches the command; the call holds until two 3-tick windows show it settled.
    reach_x, held, close = np.array(sim.actions[:50]), sim.actions[50:56], sim.actions[56:]
    np.testing.assert_allclose(reach_x[:, 0], np.linspace(0.01, 0.5, 50), rtol=1e-6)
    np.testing.assert_allclose(reach_x[:, 1], 0.0)
    np.testing.assert_allclose(held, [[0.5, 0.0]] * 6)
    # max_step lets the gripper switch in one tick while x holds its commanded target;
    # the sim succeeds on that tick, which ends the episode.
    np.testing.assert_allclose(close, [[0.5, 1.0]])
    reached_reply, closed_reply = agent.replies()
    # The trace viewer parses this opening line to map tool calls to ticks.
    assert reached_reply.startswith("Played 56 steps (5.6 s).")
    assert "At the target." in reached_reply
    assert "The episode has ended" in closed_reply


@pytest.mark.parametrize(("state_type", "tool"), [(None, "move_by"), ("BASE_DEL_POSE", "walk_by")])
async def test_a_displacement_kind_plays_in_steps_within_the_per_step_box(
    state_type: str | None, tool: str
) -> None:
    sim = Arm(contract("ee_del", [-0.1, -1.0], [0.1, 1.0]), follow=displace, done=reached)
    sim.contract["features"]["action"]["state_type"] = state_type  # wins over the coarse type

    agent, _ = await drive(sim, DirectControl(), move(tool, x=0.35))

    assert agent.tools == {tool, "wait"}
    # Four in-box steps; a displacement returns once played.
    np.testing.assert_allclose(sim.actions, [[0.0875, 0.0]] * 4)
    (reply,) = agent.replies()
    assert "observation/state: x=0.3500" in reply
    assert "At the target" not in reply


async def test_wait_holds_the_commanded_pose_and_a_null_duration_observes_for_one_tick() -> None:
    sim = Arm(REACH, done=reached)

    agent, _ = await drive(sim, DirectControl(), move("move_to", x=0.2), look(), look(1.5))

    _, looked, waited = agent.replies()
    np.testing.assert_allclose(sim.actions[26:], [[0.2, 0.0]] * 16)  # held at the setpoint
    assert looked.startswith("Played 1 steps (0.1 s).")
    assert waited.startswith("Played 15 steps (1.5 s).")
    assert "commanded: x=0.2000, grip=0.0000" in looked
    assert "At the target" not in looked


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (("move_to", {"targets": {"x": 0.1, "z": 0.1}, "note": "go"}), "'z' was unexpected"),
        (move("move_to", x=1.5), "x=1.5 is outside [0, 1]"),
    ],
)
async def test_an_invalid_move_is_a_correctable_error_that_leaves_the_sim_still(
    call: tuple[str, dict[str, Any]], error: str
) -> None:
    sim = Arm(REACH, done=reached)

    agent, _ = await drive(sim, DirectControl(dims=["x"], speed=0.05), call)

    (result,) = agent.results
    assert result.isError
    assert error in agent.replies()[0]
    assert sim.actions == []


# ─── the tool surface ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("action_type", "names", "call", "column"),
    [
        pytest.param(
            "ee_abs",
            [f"target_eef_pos.{a}" for a in "xyz"] + [f"target_eef_axis_angle.r{a}" for a in "xyz"],
            move("move_to", yaw=0.3),
            5,
            id="libero",
        ),
        pytest.param(
            "joint_pos",
            ["panda_joint1", "panda_joint2"],
            move("move_joints", joint2=0.3),
            1,
            id="panda",
        ),
        pytest.param(
            "ee_abs",
            ["left_x", "right_x", "left_gripper"],
            move("move_to", **{"right_arm.x": 0.3}),
            1,
            id="bimanual",
        ),
        pytest.param(
            "ee_abs",
            ["left_x", "left_pitch_deg"],
            move("move_to", left_pitch_deg=0.3),
            1,
            id="nonstandard",
        ),
    ],
)
async def test_tools_use_standard_dimension_names_when_every_native_name_has_one(
    action_type: str, names: list[str], call: tuple[str, dict[str, Any]], column: int
) -> None:
    sim = Arm(contract(action_type, [-1.0] * len(names), [1.0] * len(names), names))

    agent, _ = await drive(sim, DirectControl(), call)

    assert not agent.results[0].isError
    commanded = [0.0] * len(names)
    commanded[column] = 0.3
    np.testing.assert_allclose(sim.actions[-1], commanded, atol=1e-4)


# ─── settle verdicts: how each kind of arm ends an absolute move ───────

STOPPED_SHORT = "Stopped short of the target; the arm is at rest. Remaining offsets: "


async def test_a_move_longer_than_ten_seconds_plays_until_the_arm_arrives() -> None:
    sim = Arm(REACH, done=reached)

    agent, _ = await drive(
        sim, DirectControl(speed=0.05, max_step={"grip": 2.0}), move("move_to", x=0.6)
    )

    assert "At the target." in agent.replies()[0]
    assert 120 <= len(sim.actions) <= 300  # 0.6 / 0.05 s at 10 Hz, under the 30 s timeout
    assert sim.state.tolist() == pytest.approx([0.6, 0.0])


async def test_the_safety_timeout_returns_the_pose_the_arm_reached() -> None:
    sim = Arm(REACH, done=reached)

    agent, _ = await drive(
        sim, DirectControl(dims=["x"], speed=0.05, timeout=0.3), move("move_to", x=1.0)
    )

    (reply,) = agent.replies()
    assert (
        "Ran out of time before settling; the arm was still moving. "
        "Remaining offsets: 985 mm on x." in reply
    )
    assert "pose:" in reply
    assert "commanded:" in reply
    assert len(sim.actions) == 3  # 0.3 s at 10 Hz
    assert sim.state.tolist() == pytest.approx([0.015, 0.0])  # the move's prefix, not the goal


@pytest.mark.parametrize("noise", [0.0, 2e-5], ids=["steady-sensor", "jittering-sensor"])
async def test_an_arm_that_never_moves_gives_up_at_the_settle_cap(noise: float) -> None:
    sim = Arm(REACH, follow=still(noise))

    agent, _ = await drive(sim, DirectControl(), move("move_to", x=0.5))

    (reply,) = agent.replies()
    assert "off: x=-0.5000" in reply
    assert STOPPED_SHORT + "500 mm on x." in reply
    assert len(sim.actions) == 80  # the 50-tick move, then the whole 3 s settle cap


async def test_a_lagging_arm_keeps_stepping_until_it_reaches_the_target() -> None:
    sim = Arm(REACH, follow=lag(0.25))

    agent, _ = await drive(sim, DirectControl(max_step={"grip": 2.0}), move("move_to", x=0.5))

    assert "At the target. Remaining offsets: " in agent.replies()[0]
    assert len(sim.actions) > 50  # the interpolation, then a hold while the arm catches up
    assert sim.state.tolist() == pytest.approx([0.5, 0.0], abs=0.01)


async def test_a_gripper_offset_never_decides_the_verdict() -> None:
    sim = Arm(REACH, [0.0, -1.0], reads=lambda s: s * [1.0, 0.0] - [0.0, 1.0])

    agent, _ = await drive(
        sim, DirectControl(max_step={"grip": 2.0}), move("move_to", x=0.5, grip=1.0)
    )

    assert "At the target." in agent.replies()[0]
    assert len(sim.actions) == 80
    assert sim.state.tolist() == pytest.approx([0.5, 1.0])


# ─── rotations: roll / pitch / yaw onto whatever the contract stores ───


@pytest.mark.parametrize(
    ("parts", "home", "angle", "final"),
    [
        (
            ["qx", "qy", "qz", "qw"],
            [0.0, 0.0, 0.0, 1.0],
            "yaw",
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
        ),
        (
            ["qw", "qx", "qy", "qz"],
            [1.0, 0.0, 0.0, 0.0],
            "roll",
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
        ),
    ],
    ids=["yaw-onto-xyzw", "roll-onto-wxyz"],
)
async def test_a_rotation_slerps_onto_a_quaternion_in_the_contracts_component_order(
    parts: list[str], home: list[float], angle: str, final: list[float]
) -> None:
    names = [f"tcp_quat.{part}" for part in parts]
    sim = Arm(contract("ee_abs", [-1.0] * 4, [1.0] * 4, names), home)

    agent, _ = await drive(sim, DirectControl(), move("move_to", **{angle: math.pi / 2}))

    assert not agent.results[0].isError
    actions = np.array(sim.actions)
    np.testing.assert_allclose(actions[-1], final, atol=1e-4)
    assert len(actions) > 2
    # Slerped, not lerped by component: every commanded quaternion is unit length.
    np.testing.assert_allclose(np.linalg.norm(actions, axis=1), 1.0, atol=1e-5)


async def test_euler_commands_slerp_onto_axis_angle_in_equal_steps() -> None:
    names = [f"target_eef_axis_angle.r{a}" for a in "xyz"]
    sim = Arm(contract("ee_abs", [-np.pi] * 3, [np.pi] * 3, names), [0.0, 0.0, 0.0])

    await drive(sim, DirectControl(), move("move_to", roll=math.pi / 2, yaw=math.pi / 2))

    actions = np.array(sim.actions)
    # Intrinsic XYZ of roll = yaw = pi/2 is the quaternion (0.5, 0.5, 0.5, 0.5), angle 2pi/3.
    np.testing.assert_allclose(actions[-1], [(2 * np.pi / 3) / np.sqrt(3)] * 3, atol=1e-4)
    # 2pi/3 at a tenth of a turn per second, 10 Hz: equal angular steps until the hold.
    ticks = math.ceil((2 * np.pi / 3) / (0.1 * 2 * np.pi / 10))
    gaps = np.diff(np.linalg.norm(actions[:ticks], axis=1))
    assert gaps.min() > 0
    np.testing.assert_allclose(gaps, gaps[0], rtol=0.05)


async def test_reported_angles_wrap_into_pi_and_a_command_past_pi_is_accepted() -> None:
    names = ["ee.x", "ee_euler.x", "ee_euler.y", "ee_euler.z"]
    wrapped = 3.2 - 2 * math.pi
    sim = Arm(
        contract("ee_abs", [-1.0, -4.0, -4.0, -4.0], [1.0, 4.0, 4.0, 4.0], names),
        [0.0, wrapped, 0.0, 0.0],
        reads=lambda s: (
            s + np.array([0.0, 2 * math.pi, 0.0, 0.0])
        ),  # its sensor reports roll a turn up
    )

    agent, _ = await drive(
        sim, DirectControl(), look(), move("move_to", roll=3.2), move("move_to", roll=-math.pi)
    )

    assert not any(result.isError for result in agent.results)
    assert f"pose: x=0.0000, roll={wrapped:.4f}" in agent.replies()[0]
    # One tick of looking, then 3.2 commanded as the wrapped angle, then -pi.
    roll = np.array(sim.actions)[:, 1]
    np.testing.assert_allclose(roll[1:7], wrapped, atol=1e-6)
    np.testing.assert_allclose(roll[7:], -math.pi, atol=1e-6)
    assert sim.state[1] == pytest.approx(-math.pi)


# ─── the episode's time budget ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("limit", "calls", "openings", "says", "ticks", "x"),
    [
        pytest.param(
            10.0,
            [move("move_to", x=0.2), look(1.0)],
            [
                "Played 26 steps (2.6 s). Time remaining: 7.4 s.",
                "Played 10 steps (1.0 s). Time remaining: 6.4 s.",
            ],
            (),
            36,
            0.2,
            id="every-reply-reports-the-time-left",
        ),
        pytest.param(
            2.0,
            [move("move_to", x=0.5), move("move_to", x=0.1)],
            [
                "Played 20 steps (2.0 s). Time remaining: 0.0 s.",
                "Played 0 steps (0.0 s). Time remaining: 0.0 s.",
            ],
            (
                "The time limit is reached.",
                "The episode has ended; stop calling tools.",
                "The episode ended before the arm settled. Remaining offsets: 300 mm on x.",
            ),
            20,  # the move needed 50 ticks; 2 s at 10 Hz were left
            0.2,
            id="a-move-crossing-the-limit-is-cut-there-and-ends-the-episode",
        ),
    ],
)
async def test_the_time_limit_is_spent_by_played_ticks_and_ends_the_episode(
    limit: float,
    calls: list[tuple[str, dict[str, Any]]],
    openings: list[str],
    says: tuple[str, ...],
    ticks: int,
    x: float,
) -> None:
    sim = Arm(REACH, done=reached)

    agent, _ = await drive(sim, DirectControl(time_limit=limit), *calls)

    for description in agent.descriptions.values():
        assert f"You have {limit:g} seconds of simulated time." in description
    replies = agent.replies()
    assert [reply[: len(opening)] for reply, opening in zip(replies, openings, strict=True)] == (
        openings
    )
    for text in says:
        assert text in replies[0]
    assert all("Ran out of time" not in reply for reply in replies)
    assert ("The episode has ended" in replies[-1]) == bool(says)
    assert len(sim.actions) == ticks
    assert sim.state[0] == pytest.approx(x)
