"""``RobotEndpoint``: lifecycle wrapper around a bridge + recorder.

The env server task generator does the same bookkeeping in every env:

    reset the sim → start recording → yield prompt → end recording → yield score

``RobotEndpoint`` absorbs that bookkeeping so the task generator only needs to
call :meth:`reset` (get the prompt) and :meth:`result` (get the score), with the
two yields in between::

    async def my_task(task_id: int, seed: int = 0):
        prompt = await endpoint.reset(task_id=task_id, seed=seed)
        yield {"prompt": prompt}
        yield endpoint.result()

The bridge's :meth:`~RobotBridge.reset` and :meth:`~RobotBridge.result` do the
sim-specific work; the endpoint handles the recorder lifecycle around them. The
user implements the bridge; the framework constructs the endpoint.

The four verbs ``reset / observe / step / result`` are the full episode
interface. The control-plane pair (:meth:`reset` / :meth:`result`) is what the
task generator drives; the data-plane pair (:meth:`observe` / :meth:`step`) is
served to the agent over ``robot`` directly today (so it is *not* on the
in-process hot path), and is exposed here only to complete the verb set so the
same interface can cross a process boundary later (Phase 8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from hud.telemetry.recorder import EpisodeRecorder

    from .bridge import RobotBridge


class RobotEndpoint:
    """Lifecycle wrapper: bridge episode management + recorder lifecycle.

    The canonical construction hands the endpoint the env contract and lets the
    framework own recording entirely::

        endpoint = RobotEndpoint(bridge, contract=CONTRACT, name="my_env")

    With ``contract`` given (and no explicit ``recorder``), the endpoint builds
    the framework-default recorder from launch-time configuration — a LeRobot
    dataset sink when ``HUD_RECORD_DIR`` is set, a live platform stream when
    HUD telemetry is configured, fanned out from one
    :class:`~hud.telemetry.EpisodeRecorder` (see
    :func:`~hud.environment.robots.recording.default_recorder`) — and attaches
    it to the bridge. The recorder is closed by ``bridge.stop()`` (i.e. the
    env's ``@env.shutdown`` hook), so the author writes **zero recorder code**.

    Passing an explicit ``recorder`` still works and skips the default
    construction.

    The task generator then calls :meth:`reset` and :meth:`result` — nothing else.
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
            from .recording import default_recorder

            recorder = default_recorder(contract, name=name or "env")
            if recorder is not None:
                bridge.attach_recorder(recorder)
        self._recorder = recorder

    async def reset(self, **task_args: Any) -> str:
        """Reset the sim for a new episode, start recording, return the prompt.

        Calls ``bridge.reset(**task_args)`` (sim-specific), then
        ``recorder.start_episode(prompt=..., **task_args)`` so the recording
        metadata carries the same parameters as the reset. Returns the prompt
        string for the task generator to yield.
        """
        prompt = await self._bridge.reset(**task_args)
        if self._recorder is not None:
            self._recorder.start_episode(prompt=prompt, **task_args)
        return prompt

    def observe(self) -> tuple[dict[str, np.ndarray], bool] | None:
        """Return the current ``(data, terminated)`` frame (data-plane verb).

        A passthrough to ``bridge.get_observation()``. In-process the agent reads
        observations over ``robot`` directly, so this is not on the hot path; it
        completes the ``reset / observe / step / result`` verb set so the interface
        can be served across a process boundary later.
        """
        return self._bridge.get_observation()

    def step(self, action: np.ndarray) -> None:
        """Advance the sim by one action (data-plane verb).

        A passthrough to ``bridge.step(action)``. Like :meth:`observe`, this is
        served over ``robot`` in-process and is here only to complete the verb set.
        """
        self._bridge.step(action)

    def result(self, **extra: Any) -> dict[str, Any]:
        """End recording and return the episode score dict.

        Calls ``bridge.result()`` for sim-specific scoring, merges any ``extra``
        kwargs (e.g. ``inference_mode`` from the env contract), calls
        ``recorder.end_episode(...)`` with success + total_reward, and returns
        the full dict for the task generator to yield.

        Pass contract-level metadata as kwargs::

            yield endpoint.result(inference_mode=rt["inference_mode"])
        """
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
