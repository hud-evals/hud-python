"""Direct control: a tool-calling agent drives a robot sim through MCP motion tools.

Each case is a whole rollout: an in-process bridge serving the ``openpi/0`` wire,
an env publishing it plus :class:`DirectControl`'s ``mcp`` capability, and a
scripted agent calling the tools through the tunneled manifest binding, graded
by the sim. Every case asserts what the model saw (tools, schemas, replies) and
what the sim was commanded.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.agents.openai.tools.strict_schema import ensure_strict_json_schema
from hud.environment.robot import DirectControl
from hud.eval import LocalRuntime, Task, rollout

from .sims import (
    Arm,
    ScriptedLLM,
    arm_env,
    contract,
    delayed,
    displace,
    fingers,
    jitter,
    lag,
    look,
    move,
    offset,
    reached,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from hud.eval.run import Run

#: The default contract: an absolute ``[x, grip]`` action, x in [0, 1].
REACH = contract("ee_abs", [0.0, -1.0], [1.0, 1.0])


async def drive(
    sim: Arm, control: DirectControl, *calls: tuple[str, dict[str, Any]]
) -> tuple[ScriptedLLM, Run]:
    agent = ScriptedLLM(*calls)
    async with arm_env(sim, control) as env:
        run = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))
    return agent, run


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
    assert set(schema["required"]) == {"targets", "note"}
    # Every dimension is listed and nullable, so strict mode keeps the schema as it is.
    assert schema["properties"]["targets"]["required"] == ["x", "grip"]
    assert schema["properties"]["targets"]["properties"]["x"]["type"] == ["number", "null"]
    assert schema["properties"]["targets"]["properties"]["grip"]["description"] == "[-1, 1]"
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
    assert [block.type for block in agent.results[0].content] == ["text", "text", "image"]
    # The trace viewer parses this opening line to map tool calls to ticks.
    assert reached_reply.startswith("Played 56 steps (5.6 s).")
    assert "observation/state: x=0.5000, grip=0.0000" in reached_reply
    assert "pose: x=0.5000, grip=0.0000" in reached_reply
    assert "off: x=0.0000, grip=0.0000" in reached_reply
    assert "At the target. Remaining offsets: 0 mm on x." in reached_reply
    assert "The episode has ended" in closed_reply
    # Tolerance only picks the verdict; nothing the model sees mentions it.
    seen = " ".join([str(agent.schemas), *agent.descriptions.values(), reached_reply])
    for word in ("toleran", "within", "settle"):
        assert word not in seen.lower()


@pytest.mark.parametrize(("state_type", "tool"), [(None, "move_by"), ("BASE_DEL_POSE", "walk_by")])
async def test_a_displacement_kind_plays_in_steps_within_the_per_step_box(
    state_type: str | None, tool: str
) -> None:
    sim = Arm(contract("ee_del", [-0.1, -1.0], [0.1, 1.0]), follow=displace, done=reached)
    sim.contract["features"]["action"]["state_type"] = state_type  # wins over the coarse type

    agent, _ = await drive(sim, DirectControl(), move(tool, x=0.35))

    assert agent.tools == {tool, "wait"}
    assert "action's own components" in agent.descriptions[tool]
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
        (("move_to", {"targets": {"x": None}, "note": "go"}), "set at least one"),
        (("move_to", {"targets": {"x": 0.1}, "note": "  "}), "note must say"),
        (("wait", {"duration": 101, "note": "sleep"}), "duration must be within [0, 100]"),
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


async def test_a_contract_without_a_motion_type_is_refused_at_start() -> None:
    sim = Arm(contract("joint_vel", [0.0, -1.0], [1.0, 1.0]))

    with pytest.raises(ValueError, match="direct control needs an action state_type"):
        async with arm_env(sim, DirectControl()) as env:
            await env.start()


# ─── the tool surface ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("action_type", "names", "targets"),
    [
        pytest.param(
            "ee_abs",
            [f"target_eef_pos.{a}" for a in "xyz"] + [f"target_eef_axis_angle.r{a}" for a in "xyz"],
            ["x", "y", "z", "roll", "pitch", "yaw"],
            id="libero",
        ),
        pytest.param(
            "joint_pos", ["panda_joint1", "panda_joint2"], ["joint1", "joint2"], id="panda"
        ),
        pytest.param(
            "ee_abs",
            ["left_x", "right_x", "left_gripper"],
            ["left_arm", "right_arm"],
            id="bimanual",
        ),
        pytest.param(
            "ee_abs", ["left_x", "left_pitch_deg"], ["left_x", "left_pitch_deg"], id="nonstandard"
        ),
    ],
)
async def test_tools_use_standard_dimension_names_when_every_native_name_has_one(
    action_type: str, names: list[str], targets: list[str]
) -> None:
    sim = Arm(contract(action_type, [-1.0] * len(names), [1.0] * len(names), names))

    agent, _ = await drive(sim, DirectControl())

    (schema,) = (s for tool, s in agent.schemas.items() if tool != "wait")
    assert list(schema["properties"]["targets"]["properties"]) == targets


async def test_dimensions_with_a_shared_prefix_nest_as_a_group_the_model_can_null_whole() -> None:
    names = ["left_arm.x", "left_arm.y", "right_arm.x", "grip"]
    sim = Arm(contract("ee_abs", [-1.0] * 4, [1.0] * 4, names))

    agent, _ = await drive(sim, DirectControl(), move("move_to", **{"left_arm.x": 0.1}))

    schema = agent.schemas["move_to"]
    targets = schema["properties"]["targets"]
    assert targets["required"] == ["grip", "left_arm", "right_arm"]
    group = targets["properties"]["left_arm"]["anyOf"]
    assert list(group[0]["properties"]) == ["x", "y"]
    assert group[1] == {"type": "null"}
    assert ensure_strict_json_schema(copy.deepcopy(schema)) == schema
    np.testing.assert_allclose(sim.actions[-1], [0.1, 0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize(
    ("options", "calls", "tools", "annotated", "described", "moved"),
    [
        pytest.param(
            {"use_annotation": False},
            [("move_to", {"targets": {"x": 0.1, "grip": None}}), ("wait", {"duration": None})],
            {"move_to", "wait"},
            False,
            "",
            True,
            id="without-annotation-no-call-takes-a-note",
        ),
        pytest.param(
            {"use_wait": False, "time_limit": 10.0},
            [move("move_to")],
            {"move_to"},
            True,
            "Time passes only while the robot moves;",
            False,
            id="without-wait-only-the-motion-tool-is-served",
        ),
    ],
)
async def test_options_shape_the_served_tools(
    options: dict[str, Any],
    calls: list[tuple[str, dict[str, Any]]],
    tools: set[str],
    annotated: bool,
    described: str,
    moved: bool,
) -> None:
    sim = Arm(REACH, done=reached)

    agent, _ = await drive(sim, DirectControl(**options), *calls)

    assert agent.tools == tools
    for schema in agent.schemas.values():
        assert ("note" in schema["properties"]) == annotated
        assert ("note" in schema["required"]) == annotated
    assert described in agent.descriptions["move_to"]
    assert bool(sim.actions) == moved
    if "wait" not in tools:
        assert not any("wait" in reply for reply in agent.replies())


# ─── settle verdicts: how each kind of arm ends an absolute move ───────


def quantity_rows() -> list[Any]:
    """Each quantity's default tolerance moves the fingertips 1 mm: directly, over a 0.15 m
    gripper, over a 2.5 m lever of joints. An arm settling just inside it is at the target;
    just outside, it stopped short."""
    euler = ["ee.x", "ee_euler.x", "ee_euler.y", "ee_euler.z"]
    quantities = [
        ("position", "move_to", "ee_abs", ["x", "grip"], {"x": 0.5}, 0, 1e-3, "x"),
        ("orientation", "move_to", "ee_abs", euler, {"roll": 0.3}, 1, 1e-3 / 0.15, "orientation"),
        (
            "joint",
            "move_joints",
            "joint_pos",
            ["joint1", "grip"],
            {"joint1": 0.3},
            0,
            4e-4,
            "joint1",
        ),
    ]
    rows = []
    for quantity, tool, action_type, names, values, column, tolerance, label in quantities:
        for share in (0.9, 1.1):
            by = [0.0] * len(names)
            by[column] = share * tolerance
            inside = share < 1
            rows.append(
                pytest.param(
                    lambda action_type=action_type, names=names, by=by: Arm(
                        contract(action_type, [-1.0] * len(names), [1.0] * len(names), names),
                        follow=offset(by),
                    ),
                    {},
                    move(tool, **values),
                    (
                        "At the target. " if inside else "Stopped short of the target; ",
                        f" on {label}.",
                    ),
                    ("Stopped short" if inside else "At the target",),
                    range(1, 301),
                    None,
                    id=f"{quantity}-default-{'inside' if inside else 'outside'}",
                )
            )
    return rows


STOPPED_SHORT = "Stopped short of the target; the arm is at rest. Remaining offsets: "


@pytest.mark.parametrize(
    ("arm", "options", "call", "says", "never", "ticks", "pose"),
    [
        pytest.param(
            lambda: Arm(REACH, done=reached),
            {"speed": 0.05, "max_step": {"grip": 2.0}},
            move("move_to", x=0.6),
            ("At the target.",),
            (),
            range(120, 301),  # 0.6 / 0.05 s = 12 s at 10 Hz, under the 30 s safety timeout
            pytest.approx([0.6, 0.0]),
            id="a-move-longer-than-ten-seconds-plays-until-the-arm-arrives",
        ),
        pytest.param(
            lambda: Arm(REACH, done=reached),
            {"dims": ["x"], "speed": 0.05, "timeout": 0.3},
            move("move_to", x=1.0),
            (
                "Ran out of time before settling; the arm was still moving. "
                "Remaining offsets: 985 mm on x.",
                "pose:",
                "commanded:",
            ),
            (),
            range(3, 4),  # 0.3 s at 10 Hz
            pytest.approx([0.015, 0.0]),  # the prefix of the move, not a jump to the goal
            id="the-safety-timeout-returns-the-pose",
        ),
        pytest.param(
            lambda: Arm(REACH, done=reached),
            {"max_step": {"grip": 2.0}},
            move("move_to", x=1.0, grip=1.0),
            ("The episode ended before the arm settled.", "The episode has ended; stop calling"),
            ("Ran out of time",),
            range(1, 100),  # the sim succeeded before the 100-tick move finished
            None,
            id="a-sim-that-terminates-mid-move-is-not-out-of-time",
        ),
        pytest.param(
            lambda: Arm(contract("ee_abs", [0.0, -1.0], [1.0, 1.0], rate=1)),
            {},
            move("move_to", x=0.05),
            ("At the target.",),
            ("Ran out of time",),
            range(1, 31),
            pytest.approx([0.05, 0.0]),
            id="an-arm-at-the-target-is-at-rest-at-a-low-control-rate",
        ),
        pytest.param(
            lambda: Arm(REACH, follow=lag(0.0)),
            {},
            move("move_to", x=0.5),
            ("off: x=-0.5000", STOPPED_SHORT + "500 mm on x."),
            (),
            range(80, 81),  # the 50-tick move, then the whole 3 s settle cap
            pytest.approx([0.0, 0.0]),
            id="a-still-arm-gives-up-at-the-settle-cap",
        ),
        pytest.param(
            lambda: Arm(REACH, follow=jitter),
            {},
            move("move_to", x=0.5),
            ("off: x=-0.5000", STOPPED_SHORT + "500 mm on x."),
            (),
            range(80, 81),
            pytest.approx([0.0, 0.0], abs=1e-4),
            id="a-jittering-still-arm-gives-up-at-the-settle-cap",
        ),
        pytest.param(
            lambda: Arm(REACH, follow=lag(0.25)),
            {"max_step": {"grip": 2.0}},
            move("move_to", x=0.5),
            ("pose:", "off:", "At the target. Remaining offsets: "),
            (),
            range(51, 301),  # interpolation, then a hold while the arm catches up
            pytest.approx([0.5, 0.0], abs=0.01),
            id="a-lagging-arm-keeps-stepping-until-it-reaches-the-target",
        ),
        pytest.param(
            lambda: Arm(REACH, follow=offset([-0.012, 0.0])),
            {},
            move("move_to", x=0.5),
            (STOPPED_SHORT + "12 mm on x.",),
            (),
            range(50 + 21, 50 + 22),  # held for the grace period before stopping
            pytest.approx([0.488, 0.0]),
            id="an-arm-at-rest-outside-its-tolerance-is-held-for-the-grace-period",
        ),
        pytest.param(
            lambda: Arm(REACH, follow=offset([-0.012, 0.0])),
            {"tolerance": {"position": 0.02}},
            move("move_to", x=0.5),
            ("At the target. Remaining offsets: 12 mm on x.",),
            (),
            range(56, 57),  # tolerance only lengthens the wait, never shortens it
            pytest.approx([0.488, 0.0]),
            id="an-arm-at-rest-inside-a-custom-tolerance-is-at-the-target",
        ),
        pytest.param(
            lambda: Arm(REACH, follow=delayed(10)),
            {"tolerance": {"position": 0.05}},
            move("move_to", x=8e-4),
            (),
            (),
            range(17, 18),
            pytest.approx([8e-4, 0.0]),
            id="a-move-waits-for-a-slow-arm-even-when-already-within-tolerance",
        ),
        pytest.param(
            lambda: Arm(
                contract("ee_abs", [0.0] * 3, [1.0] * 3, ["x", "grip_left", "grip_right"]),
                follow=fingers,
            ),
            {"max_step": {"grip_left": 2.0, "grip_right": 2.0}},
            move("move_to", grip_left=1.0, grip_right=1.0),
            ("fingers: grip_left=0.3000, grip_right=1.0000",),  # the jam reads as no success
            ("At the target", "Remaining offsets"),
            range(20 + 3, 20 + 30),  # the right finger closes for 20 ticks; the call outlasts it
            pytest.approx([0.0, 0.3, 1.0]),
            id="a-gripper-call-waits-for-every-finger",
        ),
        pytest.param(
            lambda: Arm(REACH, [0.0, -1.0], lag(0.25), reads=lambda s: s * [1.0, 0.0] - [0.0, 1.0]),
            {"max_step": {"grip": 2.0}},
            move("move_to", x=0.5, grip=1.0),
            ("At the target. Remaining offsets: ", "2 on grip"),  # grip never decides the verdict
            (),
            range(1, 301),
            None,
            id="gripper-offsets-are-reported-when-the-reference-reads-the-hand",
        ),
        pytest.param(
            lambda: Arm(REACH, [0.0, -1.0], lag(0.25), reads=lambda s: s * [1.0, 0.0] - [0.0, 1.0]),
            {"max_step": {"grip": 2.0}, "reference": lambda data: data["observation/state"]},
            move("move_to", x=0.5, grip=1.0),
            ("At the target. Remaining offsets: ",),
            ("2 on grip",),
            range(1, 301),
            None,
            id="gripper-offsets-are-left-out-under-a-custom-reference",
        ),
        *quantity_rows(),
    ],
)
async def test_an_absolute_move_ends_with_the_verdict_its_arm_earns(
    arm: Callable[[], Arm],
    options: dict[str, Any],
    call: tuple[str, dict[str, Any]],
    says: tuple[str, ...],
    never: tuple[str, ...],
    ticks: range,
    pose: Any,
) -> None:
    sim = arm()

    agent, _ = await drive(sim, DirectControl(**options), call)

    (result,) = agent.results
    (reply,) = agent.replies()
    assert not result.isError
    for text in says:
        assert text in reply
    for text in never:
        assert text not in reply
    assert len(sim.actions) in ticks
    if pose is not None:
        assert sim.state.tolist() == pose


# ─── rotations: roll / pitch / yaw onto whatever the contract stores ───


def quaternion(order: str) -> list[str]:
    parts = ("qx", "qy", "qz", "qw") if order == "xyzw" else ("qw", "qx", "qy", "qz")
    return [f"tcp_quat.{part}" for part in parts]


def unit(actions: NDArray[Any]) -> bool:
    """Every commanded quaternion is unit length: slerped, not lerped by component."""
    return len(actions) > 2 and bool(np.allclose(np.linalg.norm(actions, axis=1), 1.0, atol=1e-5))


def evenly_turned(actions: NDArray[Any]) -> bool:
    """2pi/3 at a tenth of a turn per second, 10 Hz: equal angular steps until the hold."""
    ticks = math.ceil((2 * np.pi / 3) / (0.1 * 2 * np.pi / 10))
    gaps = np.diff(np.linalg.norm(actions[:ticks], axis=1))
    return bool(gaps.min() > 0 and np.allclose(gaps, gaps[0], rtol=0.05))


WRAPPED = 3.2 - 2 * math.pi
#: Intrinsic XYZ of roll = yaw = pi/2 is the quaternion (0.5, 0.5, 0.5, 0.5), angle 2pi/3.
AXIS_ANGLE = (2.0 * np.pi / 3.0) / np.sqrt(3.0)


@pytest.mark.parametrize(
    ("names", "low", "high", "home", "calls", "final", "path", "says"),
    [
        pytest.param(
            quaternion("xyzw"),
            [-1.0] * 4,
            [1.0] * 4,
            [0.0, 0.0, 0.0, 1.0],
            [move("move_to", yaw=math.pi / 2)],
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
            unit,
            (),
            id="a-yaw-slerps-an-xyzw-quaternion",
        ),
        pytest.param(
            quaternion("wxyz"),
            [-1.0] * 4,
            [1.0] * 4,
            [1.0, 0.0, 0.0, 0.0],
            [move("move_to", roll=math.pi / 2)],
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
            unit,
            (),
            id="a-roll-writes-a-wxyz-quaternion",
        ),
        pytest.param(
            ["x", "tcp_quat.qx", "tcp_quat.qy", "tcp_quat.qz", "tcp_quat.qw"],
            [0.0, -1.0, -1.0, -1.0, -1.0],
            [1.0] * 5,
            [0.0, 0.0, 0.2, 0.0, 0.9],  # not a unit quaternion
            [move("move_to", x=0.2)],
            [0.2, 0.0, 0.2, 0.0, 0.9],
            lambda actions: len(actions) > 1 and (actions[:, 1:] == actions[0, 1:]).all(),
            (),
            id="an-unnamed-quaternion-is-copied-unchanged",
        ),
        pytest.param(
            [f"target_eef_axis_angle.r{a}" for a in "xyz"],
            [-np.pi] * 3,
            [np.pi] * 3,
            [0.0, 0.0, 0.0],
            [move("move_to", roll=math.pi / 2, yaw=math.pi / 2)],
            [AXIS_ANGLE] * 3,
            evenly_turned,
            (),
            id="euler-commands-slerp-onto-axis-angle",
        ),
        pytest.param(
            [
                "ee_pos.x",
                "ee_pos.y",
                "ee_pos.z",
                "ee_euler.x",
                "ee_euler.y",
                "ee_euler.z",
                "gripper",
            ],
            [-1.0, -1.0, 0.0, -np.pi, -np.pi, -np.pi, 0.0],
            [1.0, 1.0, 2.0, np.pi, np.pi, np.pi, 0.04],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04],
            [move("move_to", z=1.2, yaw=0.4)],
            [0.0, 0.0, 1.2, 0.0, 0.0, 0.4, 0.04],
            None,
            (),
            id="an-euler-contract-is-addressed-as-roll-pitch-yaw",
        ),
        pytest.param(
            ["ee.x", "ee_euler.x", "ee_euler.y", "ee_euler.z"],
            [-1.0, -4.0, -4.0, -4.0],
            [1.0, 4.0, 4.0, 4.0],
            [0.0, 3.2, 0.0, 0.0],  # a sim that reports roll unwrapped, just past pi
            [look(), move("move_to", roll=3.2), move("move_to", roll=-math.pi)],
            None,
            None,
            (
                f"ee_euler.x={WRAPPED:.4f}",
                f"pose: x=0.0000, roll={WRAPPED:.4f}",
                "At the target. Remaining offsets: ",
                "pi and -pi are the same orientation",
            ),
            id="reported-angles-wrap-into-pi-and-a-command-past-pi-is-accepted",
        ),
    ],
)
async def test_an_orientation_command_reaches_the_contracts_rotation_encoding(
    names: list[str],
    low: list[float],
    high: list[float],
    home: list[float],
    calls: list[tuple[str, dict[str, Any]]],
    final: list[float] | None,
    path: Callable[[NDArray[Any]], bool] | None,
    says: tuple[str, ...],
) -> None:
    sim = Arm(contract("ee_abs", low, high, names), home)

    agent, _ = await drive(sim, DirectControl(), *calls)

    assert not any(result.isError for result in agent.results)
    actions = np.array(sim.actions)
    if final is not None:
        np.testing.assert_allclose(actions[-1], final, atol=1e-4)
    if path is not None:
        assert path(actions)
    seen = "\n".join([agent.descriptions["move_to"], *agent.replies()])
    for text in says:
        assert text in seen


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
