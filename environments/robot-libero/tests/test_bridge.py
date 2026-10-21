"""Episode timing through the real control channel, with only LIBERO mocked."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hud.environment.robot import RobotEndpoint

ROOT = Path(__file__).resolve().parents[1]


class Simulator:
    def __init__(self, *, control_freq: int = 20, ignore_done: bool = False, **kwargs: Any) -> None:
        self.control_freq = control_freq
        self.ignore_done = ignore_done
        self.horizon = 1000
        self.env = SimpleNamespace(done=False)
        self.robots = [SimpleNamespace(controller=SimpleNamespace(use_delta=True))]
        self.steps = 0
        self.success = False
        self.closed = False
        self.observation = {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
        }

    def reset(self) -> None:
        self.steps = 0
        self.env.done = False
        self.success = False

    def set_init_state(self, state: np.ndarray) -> None:
        pass

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        if self.env.done:
            raise ValueError("executing action in terminated episode")
        self.steps += 1
        self.env.done = self.steps >= self.horizon and not self.ignore_done
        self.success = bool(action[-1] > 0)
        return self.observation, float(self.success), self.success, {}

    def check_success(self) -> bool:
        return self.success

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def simulator_module(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list[Simulator]]:
    created: list[Simulator] = []

    def make_simulator(**kwargs: Any) -> Simulator:
        simulator = Simulator(**kwargs)
        created.append(simulator)
        return simulator

    task = SimpleNamespace(language="move the object", problem_folder="spatial", bddl_file="task.bddl")
    suite = SimpleNamespace(get_task=lambda _: task, get_task_init_states=lambda _: [np.zeros(1)])
    modules = {
        name: ModuleType(name)
        for name in (
            "libero",
            "libero.libero",
            "libero.libero.envs",
            "robosuite",
            "robosuite.utils",
            "robosuite.utils.transform_utils",
        )
    }
    modules["libero.libero"].benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite})
    modules["libero.libero"].get_libero_path = lambda _: "/bddl"
    modules["libero.libero.envs"].OffScreenRenderEnv = make_simulator
    modules["robosuite.utils.transform_utils"].quat2axisangle = lambda _: np.zeros(3)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("libero_bridge_under_test", ROOT / "bridge.py")
    assert spec is not None and spec.loader is not None
    bridge_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge_module)
    return bridge_module, created


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ee_abs", "ee_del"])
async def test_episode_uses_contract_timing_and_preserves_terminal_results(
    mode: str, simulator_module: tuple[ModuleType, list[Simulator]]
) -> None:
    module, created = simulator_module
    bridge = module.LiberoSimBridge(use_delta=mode == "ee_del")
    bridge.contract = json.loads((ROOT / "contracts" / f"{mode}.json").read_text())
    server = await bridge.serve_control()
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1])
    try:
        await endpoint.start()
        episode = await endpoint.reset(task_suite="libero_spatial", task_id=0)
        assert episode["prompt"] == "move the object"
        simulator = created[-1]
        assert simulator.robots[0].controller.use_delta == (mode == "ee_del")
        assert simulator.steps == module.SETTLE_STEPS
        action = np.zeros((1, 7))
        for _ in range(10):
            bridge.step(action)
        assert (simulator.steps - module.SETTLE_STEPS) / simulator.control_freq == pytest.approx(
            10 / (await endpoint.contract())["control_rate"]
        )
        for _ in range(simulator.horizon - simulator.steps - 1):
            bridge.step(action)
        assert bridge.get_observation()[1] is False
        bridge.step(action)
        assert bridge.get_observation()[1] is True
        bridge.step(action)
        assert simulator.steps == simulator.horizon
        assert await endpoint.result() == {"score": 0.0, "success": False, "total_reward": 0.0}

        await endpoint.reset(task_suite="libero_spatial", task_id=0)
        assert simulator.closed
        simulator = created[-1]
        assert simulator.steps == module.SETTLE_STEPS
        assert bridge.get_observation()[1] is False
        action[0, -1] = 1
        bridge.step(action)
        assert bridge.get_observation()[1] is True
        bridge.step(action)
        assert simulator.steps == module.SETTLE_STEPS + 1
        assert await endpoint.result() == {"score": 1.0, "success": True, "total_reward": 1.0}
    finally:
        await endpoint.stop()
        server.close()
        await server.wait_closed()
        await bridge.stop()
