"""Force-grasp grading over the real task lifecycle with a simulated tool boundary."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from hud.environment.server import TaskRunner


@dataclass(frozen=True)
class Observation:
    left: float = 0.5
    right: float = 0.5
    height: float = 0.79


class Simulator:
    def __init__(self, observations: list[Observation]) -> None:
        self.observations = observations
        self.index = 0
        self.actions: list[list[float]] = []

    async def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        if name == "reset":
            self.index = 0
            return {"status": "ready"}
        if name == "get_state":
            return {"actuator_controls": [0.1, -0.2, 0.3, 1.0, 0.027, 0.031]}
        if name == "step":
            self.actions.append(kwargs["action"])
            self.index += 1
            return {"step": self.index, "done": True}
        observation = self.observations[self.index]
        if name == "get_contact_forces":
            force = observation.left if kwargs["body_name"] == "finger_left" else observation.right
            return {"total_force_magnitude": force}
        if name == "get_object_state":
            return {"position": {"z": observation.height}}
        raise AssertionError(f"Unexpected simulation tool: {name}")


@pytest.fixture
def robot_env(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "environment" / "env.py"
    monkeypatch.syspath_prepend(str(path.parents[1]))
    host = importlib.import_module("sim.host")
    monkeypatch.setattr(host, "SimHost", lambda: SimpleNamespace(mcp_url="http://127.0.0.1:8769/mcp"))
    spec = importlib.util.spec_from_file_location("robot_gripper_force_grasp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hold_steps", [None, 1, 7])
async def test_hold_advances_the_configured_duration_without_releasing_the_grip(
    robot_env: ModuleType, monkeypatch: pytest.MonkeyPatch, hold_steps: int | None
) -> None:
    duration = 100 if hold_steps is None else hold_steps
    simulator = Simulator([Observation()] * (duration + 1))
    monkeypatch.setattr(robot_env.sim_tools, "call", simulator.call)
    task = robot_env.force_grasp() if hold_steps is None else robot_env.force_grasp(hold_steps=hold_steps)
    runner = TaskRunner(robot_env.env.templates[task.id], task.args)

    prompt = await runner.start()
    result = await runner.grade({"answer": "ready"})

    assert f"{duration} physics steps" in prompt["prompt"]
    assert "above z=0.78 m" in prompt["prompt"]
    assert "finger control targets stay unchanged" in prompt["prompt"]
    assert simulator.actions == [[0.0, 0.0, 0.0, 0.0, 0.027, 0.031]] * duration
    assert result["score"] == 1.0
    assert [subscore["value"] for subscore in result["subscores"]] == [1.0, 1.0]


@pytest.mark.parametrize(
    ("observations", "grip", "lifted", "score"),
    [
        ([Observation(), Observation(left=0.49), Observation(), Observation()], 0.5, 1.0, 0.7),
        ([Observation(), Observation(right=0.49), Observation(), Observation()], 0.5, 1.0, 0.7),
        ([Observation(), Observation(left=0.0, right=0.0), Observation(), Observation()], 0.0, 1.0, 0.4),
        ([Observation(), Observation(height=0.78), Observation(), Observation()], 1.0, 0.0, 0.6),
        ([Observation(left=0.0), Observation(), Observation(), Observation()], 0.5, 1.0, 0.7),
        ([Observation(), Observation(), Observation(), Observation(height=0.78)], 1.0, 0.0, 0.6),
    ],
    ids=[
        "left-contact-loss",
        "right-contact-loss",
        "both-contacts-lost",
        "dropped-and-recovered",
        "not-held-at-start",
        "dropped-at-end",
    ],
)
async def test_a_failed_hold_criterion_stays_failed_after_recovery(
    robot_env: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    observations: list[Observation],
    grip: float,
    lifted: float,
    score: float,
) -> None:
    simulator = Simulator(observations)
    monkeypatch.setattr(robot_env.sim_tools, "call", simulator.call)
    task = robot_env.force_grasp(hold_steps=3)
    runner = TaskRunner(robot_env.env.templates[task.id], task.args)

    await runner.start()
    result = await runner.grade({"answer": "ready"})

    assert result["score"] == pytest.approx(score)
    assert result["score"] < 1.0
    assert len(simulator.actions) == 3
    assert [subscore["value"] for subscore in result["subscores"]] == [grip, lifted]
    assert "NOT SUSTAINED" in result["content"]


@pytest.mark.parametrize(
    "args",
    [
        {"hold_steps": 0},
        {"hold_steps": -1},
        {"hold_steps": 1.5},
        {"min_grip_force": 0.0},
        {"min_grip_force": -0.1},
        {"min_grip_force": float("inf")},
        {"min_grip_force": float("nan")},
    ],
)
async def test_invalid_hold_configuration_fails_before_reset(
    robot_env: ModuleType, monkeypatch: pytest.MonkeyPatch, args: dict[str, float]
) -> None:
    call = AsyncMock()
    monkeypatch.setattr(robot_env.sim_tools, "call", call)
    runner = TaskRunner(robot_env.env.templates["force-grasp"], args)

    with pytest.raises(ValueError, match="hold_steps|min_grip_force"):
        await runner.start()

    await runner.cancel()
    call.assert_not_awaited()


async def test_a_failed_physics_step_does_not_yield_a_successful_grade(
    robot_env: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulator = Simulator([Observation(), Observation()])

    async def fail_step(name: str, **kwargs: Any) -> dict[str, Any]:
        if name == "step":
            return {"error": "Could not advance physics"}
        return await simulator.call(name, **kwargs)

    monkeypatch.setattr(robot_env.sim_tools, "call", fail_step)
    task = robot_env.force_grasp(hold_steps=1)
    runner = TaskRunner(robot_env.env.templates[task.id], task.args)
    await runner.start()

    with pytest.raises(RuntimeError, match="Could not advance grasp hold"):
        await runner.grade({"answer": "ready"})
