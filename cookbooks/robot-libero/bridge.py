"""The simulator: one LIBERO (robosuite/MuJoCo) episode behind the ``robot`` wire.

A ``RobotBridge`` is the seam the SDK drives: ``reset`` loads a task, ``step`` applies an
action, ``get_observation`` reports cameras and state, and ``success``/``terminated`` grade.
"""

from __future__ import annotations

import os

# Software rendering by default so the bridge runs on CPU-only machines.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
from hud.environment.robot import RobotBridge
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

# After a reset, a few no-op steps (gripper open) let objects settle.
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
SETTLE_STEPS = 10
LOG_EVERY = 20  # steps between progress lines


class LiberoSimBridge(RobotBridge):
    """Owns a LIBERO env and serves it over ``robot``.

    ``use_delta`` picks the arm controller (delta or absolute end-effector targets) and must
    match the contract the env publishes.
    """

    def __init__(self, use_delta: bool = True, image_size: int = 256) -> None:
        super().__init__()
        self._use_delta = use_delta
        self._image_size = image_size
        self._env: OffScreenRenderEnv | None = None
        self._obs: dict | None = None  # latest robosuite observation
        self._step_count = 0

    async def stop(self) -> None:
        await super().stop()
        if self._env is not None:
            self._env.close()

    def reset(self, task_suite: str, task_id: int, init_state_id: int = 0) -> str:
        """Load the task and its saved initial state, settle it, and return the prompt."""
        suite = benchmark.get_benchmark_dict()[task_suite]()
        task = suite.get_task(task_id)
        self.task_description = task.language
        print(f"[env] loading {task_suite}:{task_id}:{init_state_id} {task.language!r}", flush=True)

        # A new env per task: each task has its own scene file.
        if self._env is not None:
            self._env.close()
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self._image_size,
            camera_widths=self._image_size,
            control_freq=self.contract["control_rate"],  # sim rate matches the contract
        )
        self._env.reset()
        self._env.set_init_state(suite.get_task_init_states(task_id)[init_state_id])

        for _ in range(SETTLE_STEPS):
            obs, *_ = self._env.step(DUMMY_ACTION)

        # Select the arm controller mode once the scene has settled.
        for robot in self._env.robots:
            robot.controller.use_delta = self._use_delta

        self._obs = obs
        self._step_count = 0
        return self.task_description

    def step(self, action: np.ndarray) -> None:
        if self.terminated:
            return
        # The SDK sends a batch [N, A]; this sim has one slot (N = 1).
        self._obs, reward, _, _ = self._env.step(np.asarray(action[0], dtype=np.float64))
        self.total_reward += reward
        # Grading: LIBERO's own success check. A horizon timeout ends the episode as a failure;
        # step()'s done flag is only check_success, so the horizon comes from robosuite's env.done.
        self.success = bool(self._env.check_success())
        self.terminated = self._env.env.done or self.success
        self._step_count += 1
        if self.terminated or self._step_count % LOG_EVERY == 0:
            print(
                f"[env] step {self._step_count} reward={reward:.3f} "
                f"terminated={self.terminated} success={self.success}",
                flush=True,
            )

    def get_observation(self) -> tuple[dict[str, np.ndarray], bool] | None:
        if self._obs is None:
            return None
        # Keys are the contract's observation names (the OpenPI LIBERO convention); state is
        # eef position, eef axis-angle, and the two finger joints.
        data = {
            "observation/image": self._read_camera("agentview_image"),
            "observation/wrist_image": self._read_camera("robot0_eye_in_hand_image"),
            "observation/state": np.concatenate(
                [
                    np.asarray(self._obs["robot0_eef_pos"], dtype=np.float32),
                    quat2axisangle(np.asarray(self._obs["robot0_eef_quat"])).astype(np.float32),
                    np.asarray(self._obs["robot0_gripper_qpos"], dtype=np.float32),
                ]
            ),
        }
        return data, self.terminated

    def _read_camera(self, camera: str) -> np.ndarray:
        # Rotate 180 degrees to match the LeRobot/LIBERO camera convention.
        return np.ascontiguousarray(np.asarray(self._obs[camera])[::-1, ::-1], dtype=np.uint8)
