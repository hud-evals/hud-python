"""``RobotEndpoint``: wraps a bridge with the recorder lifecycle so the task
generator only calls :meth:`reset` / :meth:`result`::

    async def my_task(task_id: int, seed: int = 0):
        prompt = await endpoint.reset(task_id=task_id, seed=seed)
        yield {"prompt": prompt}
        yield endpoint.result()

``reset`` / ``result`` is the episode interface; the bridge itself serves
observations/actions over ``robot``, so the endpoint only owns the recorder lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bridge import RobotBridge
    from .data_saving import LeRobotRecorder


class RobotEndpoint:
    """Wraps a bridge with the recorder lifecycle.

    Given a ``contract`` (and no explicit ``recorder``), builds + attaches the
    env-var-configured recorder (see :meth:`~...data_saving.LeRobotRecorder.from_env`)
    and closes it via ``bridge.stop()`` — so the author writes zero recorder code.
    """

    def __init__(
        self,
        bridge: RobotBridge,
        recorder: LeRobotRecorder | None = None,
        *,
        contract: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        self._bridge = bridge
        if recorder is None and contract is not None:
            from .data_saving import LeRobotRecorder

            recorder = LeRobotRecorder.from_env(contract, name=name or "env")
            if recorder is not None:
                bridge.attach_recorder(recorder)
        self._recorder = recorder

    async def reset(self, **task_args: Any) -> str:
        """Reset the sim, start recording, return the prompt."""
        prompt = await self._bridge._reset(**task_args)
        if self._recorder is not None:
            self._recorder.start_episode(prompt=prompt, **task_args)
        return prompt

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
