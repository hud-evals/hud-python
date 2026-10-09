"""A wrapped vectorized sim reports one trace per episode per recorded slot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from inline_snapshot import snapshot

from hud.environment.robot import wrap

if TYPE_CHECKING:
    from tests.harness import FakeServices, HudEnv


class TwoArmSim:
    """Two envs: env 0 finishes on the second step (succeeding with a shaped reward of
    0.2), env 1 on the third (failing with a shaped reward of 5.0)."""

    num_envs = 2
    metadata: ClassVar[dict[str, int]] = {"render_fps": 10}
    DONES = ([False, False], [True, False], [False, True])

    def __init__(self) -> None:
        self.t = 0

    def _obs(self) -> dict[str, Any]:
        return {
            "state": np.full((2, 3), self.t, dtype=np.float32),
            "cam": np.full((2, 16, 16, 3), self.t, dtype=np.uint8),
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        del kwargs
        return self._obs(), {}

    def step(self, action: Any) -> tuple[Any, ...]:
        del action
        done = np.array(self.DONES[self.t])
        self.t += 1
        info = {"success": np.array([True, False])}
        return self._obs(), np.array([0.2, 5.0]), done, np.array([False, False]), info

    def close(self) -> None:
        pass


def test_each_episode_reports_its_own_trace_and_outcome(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    for path in ("/v2/trace/job/{id}/enter", "/v2/trace/{id}/enter", "/v2/trace/{id}/exit"):
        services.route("api", "POST", path, json={})
    services.route("telemetry", "POST", "/trace/{id}/telemetry-upload", json={})

    env = wrap(TwoArmSim(), contract=None)
    env.reset()
    for _ in TwoArmSim.DONES:
        env.step(np.zeros((2, 2)))
    env.close()

    opened = [
        request.params["id"] for request in services.requests("api", "POST", "/v2/trace/{id}/enter")
    ]
    label = {trace_id: f"trace-{index}" for index, trace_id in enumerate(opened)}
    exits = {
        label[request.params["id"]]: request.json
        for request in services.requests("api", "POST", "/v2/trace/{id}/exit")
    }
    spans: dict[str, list[dict[str, Any]]] = {}
    for upload in services.requests("telemetry", "POST", "/trace/{id}/telemetry-upload"):
        spans.setdefault(label[upload.params["id"]], []).extend(upload.json["telemetry"])
    steps = {
        trace: [
            (span["name"], span["attributes"]["hud.payload"]["tick"])
            for span in sorted(recorded, key=lambda span: span["start_time"])
            if span["name"] != "step.video_segment"
        ]
        for trace, recorded in spans.items()
    }
    assert exits == snapshot(
        {
            "trace-0": {
                "status": "completed",
                "reward": 1.0,
                "metadata": {"env_index": 0, "episode_index": 0, "seed": None, "success": True},
            },
            "trace-1": {
                "status": "completed",
                "reward": 0.0,
                "metadata": {"env_index": 1, "episode_index": 0, "seed": None, "success": False},
            },
            "trace-2": {
                "status": "completed",
                "reward": 0.2,
                "metadata": {"env_index": 0, "episode_index": 1, "seed": None},
            },
        }
    )
    assert dict(sorted(steps.items())) == snapshot(
        {
            "trace-0": [("step.observation", 0), ("step.inference", 0), ("step.inference", 1)],
            "trace-1": [
                ("step.observation", 0),
                ("step.inference", 0),
                ("step.observation", 1),
                ("step.inference", 1),
                ("step.inference", 2),
            ],
            "trace-2": [("step.observation", 0), ("step.inference", 0)],
        }
    )
