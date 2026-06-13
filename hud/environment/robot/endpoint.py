"""``RobotEndpoint``: wraps a bridge so the task generator only calls
:meth:`reset` / :meth:`result`::

    async def my_task(task_id: int, seed: int = 0):
        prompt = await endpoint.reset(task_id=task_id, seed=seed)
        yield {"prompt": prompt}
        yield endpoint.result()

``reset`` / ``result`` is the episode interface; the bridge itself serves
observations/actions over ``robot``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bridge import RobotBridge


class RobotEndpoint:
    """Wraps a bridge with the episode interface (``reset`` / ``result``)."""

    def __init__(
        self,
        bridge: RobotBridge,
        *,
        contract: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        self._bridge = bridge

    async def reset(self, **task_args: Any) -> str:
        """Reset the sim, return the prompt."""
        return await self._bridge._reset(**task_args)

    def result(self, **extra: Any) -> dict[str, Any]:
        """Return ``bridge.result()`` merged with any ``extra`` metadata
        (e.g. ``endpoint.result(inference_mode=...)``)."""
        res = {**self._bridge.result(), **extra}
        terminated = getattr(self._bridge, "terminated", False)
        print(
            f"[env] task evaluate: success={res.get('success')} "
            f"terminated={terminated} total_reward={res.get('total_reward', 0.0):.3f}",
            flush=True,
        )
        return res


__all__ = ["RobotEndpoint"]
