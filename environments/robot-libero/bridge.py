"""The LIBERO simulator as a ``RobotBridge``: one robosuite episode behind the ``robot`` wire."""

from __future__ import annotations

import os

# Software rendering by default so the bridge runs on CPU-only machines.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
from hud.environment.robot import RobotBridge
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

# A no-op action (gripper open) that lets objects settle after a reset.
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
SETTLE_STEPS = 10
LOG_EVERY = 20


class LiberoSimBridge(RobotBridge):
    """Owns a LIBERO env and serves it over ``robot``.

    ``use_delta`` picks the arm controller (delta or absolute end-effector targets) and must
    match the contract the env publishes.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, use_delta: bool = True, image_size: int = 256) -> None:
        super().__init__(host=host, port=port)
        self._use_delta = use_delta
        self._image_size = image_size
        self._env: OffScreenRenderEnv | None = None
        self._obs: dict | None = None
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

        if self._env is not None:
            self._env.close()
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self._image_size,
            camera_widths=self._image_size,
            control_freq=self.contract["control_rate"],
        )
        self._env.reset()
        self._env.set_init_state(suite.get_task_init_states(task_id)[init_state_id])

        for _ in range(SETTLE_STEPS):
            obs, *_ = self._env.step(DUMMY_ACTION)

        for robot in self._env.robots:
            robot.controller.use_delta = self._use_delta

        self._obs = obs
        self._step_count = 0
        return self.task_description

    def step(self, action: np.ndarray) -> None:
        if self.terminated:
            return
        # The bridge steps a batch [N, A]; this sim has a single slot (N = 1).
        self._obs, reward, _, _ = self._env.step(np.asarray(action[0], dtype=np.float64))
        self.total_reward += reward
        # LIBERO's own success predicate, latched on its first satisfaction (its standard eval
        # convention). A horizon timeout ends the episode without counting as success. step()'s
        # done flag is check_success, so the horizon is read from robosuite's env.done.
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
        # Keys are the contract's observation names (the OpenPI LIBERO convention).
        data = {
            "observation/image": self._get_frame("agentview_image"),
            "observation/wrist_image": self._get_frame("robot0_eye_in_hand_image"),
            "observation/state": np.concatenate(
                [
                    np.asarray(self._obs["robot0_eef_pos"], dtype=np.float32),
                    quat2axisangle(np.asarray(self._obs["robot0_eef_quat"])).astype(np.float32),
                    np.asarray(self._obs["robot0_gripper_qpos"], dtype=np.float32),
                ]
            ),
        }
        return data, self.terminated

    def _get_frame(self, camera: str) -> np.ndarray:
        # Rotate 180 degrees to match the LeRobot/LIBERO camera convention.
        return np.ascontiguousarray(np.asarray(self._obs[camera])[::-1, ::-1], dtype=np.uint8)
