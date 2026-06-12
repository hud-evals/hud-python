"""``RobotEndpoint``: wraps a bridge with the recorder lifecycle so the task
generator only calls :meth:`reset` / :meth:`result`::

    async def my_task(task_id: int, seed: int = 0):
        prompt = await endpoint.reset(task_id=task_id, seed=seed)
        yield {"prompt": prompt}
        yield endpoint.result()

``reset / observe / step / result`` is the full episode interface. Crucially, this
verb set lets the sim run in a *separate process* from the agent (useful for heavy
sims like Isaac Sim): ``observe`` /``step`` are served over ``robot`` so the whole 
episode can cross a process (or machine) boundary. They exist here only to 
complete that set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from hud.telemetry.recorder import EpisodeRecorder

    from .bridge import RobotBridge


class RobotEndpoint:
    """Wraps a bridge with the recorder lifecycle.

    Given a ``contract`` (and no explicit ``recorder``), builds + attaches the
    framework-default recorder (see :func:`~...data_saving.default_recorder`) and
    closes it via ``bridge.stop()`` — so the author writes zero recorder code.
    """

    def __init__(
        self,
        bridge: RobotBridge,
        recorder: EpisodeRecorder | None = None,
        *,
        contract: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        self._bridge = bridge
        if recorder is None and contract is not None:
            from .data_saving import default_recorder

            recorder = default_recorder(contract, name=name or "env")
            if recorder is not None:
                bridge.attach_recorder(recorder)
        self._recorder = recorder

    async def reset(self, **task_args: Any) -> str:
        """Reset the sim, start recording, return the prompt."""
        prompt = await self._bridge._reset(**task_args)
        if self._recorder is not None:
            self._recorder.start_episode(prompt=prompt, **task_args)
        return prompt

    def observe(self) -> tuple[dict[str, np.ndarray], bool] | None:
        """Current ``(data, terminated)`` frame (passthrough to ``bridge.get_observation()``)."""
        return self._bridge.get_observation()

    def step(self, action: np.ndarray) -> None:
        """Advance the sim by one action (passthrough to ``bridge.step()``)."""
        self._bridge.step(action)

    def result(self, **extra: Any) -> dict[str, Any]:
        """End recording; return ``bridge.result()`` merged with any ``extra`` metadata
        (e.g. ``endpoint.result(inference_mode=...)``)."""
        res = {**self._bridge.result(), **extra}
        terminated = getattr(self._bridge, "terminated", False)
        print(
            f"[env] task evaluate: success={res.get('success')} "
            f"terminated={terminated} total_reward={res.get('total_reward', 0.0):.3f}",
            flush=True,
        )
        if self._recorder is not None:
            self._recorder.end_episode(
                success=res.get("success", False),
                total_reward=res.get("total_reward", 0.0),
            )
        return res


__all__ = ["RobotEndpoint"]
