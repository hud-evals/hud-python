"""Robot trace recording: numeric state, per-camera H.264 video, action chunks.

Shared by both sides of the robot stack (the agent harness, ``wrap``, the
gym bridges), so it lives under telemetry rather than either side:

- :class:`TraceRecorder` — one trace. Emits spans with an explicit ``trace_id``
  (no rollout contextvar needed, works from any thread), or through a ``Run``
  so steps also land on ``run.trace`` for training. Span emission only —
  trace lifecycle (enter/exit) is the caller's job.
- :class:`JobRecorder` — a whole vectorized env as one **Job** of per-episode
  traces: one :class:`TraceRecorder` per recorded slot; ``done[i]`` closes that
  slot's trace (reporting its exit) and opens a fresh one for the next episode.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
import numpy as np

from hud.agents.types import InferenceStep, ObservationStep, StateFeature
from hud.telemetry.context import get_current_trace_id
from hud.types import Step
from hud.utils.platform import PlatformClient

from .video import VideoStreamer

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from hud.eval.run import Run

logger = logging.getLogger(__name__)


def to_numpy(x: Any) -> NDArray[Any]:
    """Coerce a torch tensor / array / scalar to a numpy array (no torch dependency)."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _report_sync(path: str, payload: dict[str, Any]) -> None:
    """Best-effort sync POST to the platform (job/trace lifecycle). No-op without an
    API key; never fails the run when the platform is unreachable."""
    from hud.settings import settings

    if not (settings.telemetry_enabled and settings.api_key):
        return
    body = {k: v for k, v in payload.items() if v is not None}
    try:
        PlatformClient.from_settings().post(path, json=body)
    except Exception as exc:  # reporting is fire-and-forget
        logger.warning("platform report %s failed: %s", path, exc)


class TraceRecorder:
    """Streams one trace's robot telemetry; lifecycle stays with the caller.

    Construct with ``trace_id`` (spans emitted explicitly — any thread, no
    ambient context) or ``run`` (steps recorded through it, so they also land
    on ``run.trace``). ``obs_space`` labels observations from the env contract;
    ``state_names``/``action_names`` are the contract-free fallback labels.
    """

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        run: Run | None = None,
        fps: int = 10,
        prompt: str | None = None,
        obs_space: dict[str, Any] | None = None,
        action_names: list[str] | None = None,
        state_names: dict[str, list[str]] | None = None,
    ) -> None:
        assert trace_id or run, "TraceRecorder needs a trace_id or a run"
        self._run = run
        # Video encodes on a background thread, so it always needs an explicit id;
        # a live run's id is the ambient rollout context.
        self.trace_id = trace_id or (run.trace_id if run else None) or get_current_trace_id()
        self._fps = fps
        self._obs_space = obs_space
        self._action_names = action_names or []
        self._state_names = state_names or {}
        self.reward = 0.0
        self._video: VideoStreamer | None = None  # lazy, one per trace
        # The task instruction as an opening user step (shows on the timeline).
        # A Run records its own prompt step, so only the bare-trace path emits one.
        if prompt and run is None:
            self._emit(
                Step(
                    source="user",
                    messages=[
                        mcp_types.PromptMessage(
                            role="user", content=mcp_types.TextContent(type="text", text=prompt)
                        )
                    ],
                )
            )

    def _emit(self, step: Step) -> None:
        if self._run is not None:
            self._run.record(step)
        else:
            step.emit(trace_id=self.trace_id)

    def record_observation(self, data: dict[str, Any], *, tick: int) -> None:
        """One tick's observation: numeric-state span + per-camera video.

        ``data`` maps feature names to per-env arrays; camera frames (``HxWxC``)
        go to the video path, flat numeric vectors become labelled state.
        """
        if self._obs_space is not None:  # contract-aware labeling/slicing
            step = ObservationStep.from_obs({"data": data}, tick=tick, obs_space=self._obs_space)
        else:
            state: dict[str, StateFeature] = {}
            for name, val in data.items():
                arr = to_numpy(val)
                if arr.ndim >= 2:
                    continue  # camera frames travel as video
                flat = np.atleast_1d(arr).astype(float).ravel()
                if 0 < flat.size <= 512:
                    labels = self._state_names.get(name, [])
                    state[name] = StateFeature(
                        names=labels if len(labels) == flat.size else [], values=flat.tolist()
                    )
            step = ObservationStep(tick=tick, state=state)
        self._emit(step)
        if self._video is None:
            self._video = VideoStreamer(fps=self._fps, trace_id=self.trace_id)
        self._video.record({"data": data})

    def record_inference(self, chunk: Any, *, tick: int) -> None:
        """One inference: the freshly inferred ``[T, A]`` chunk (an executed
        single action is a length-1 chunk)."""
        rows = np.atleast_2d(to_numpy(chunk).astype(float))
        self._emit(
            InferenceStep(
                tick=tick, chunk=rows.tolist(), chunk_length=len(rows), names=self._action_names
            )
        )

    def add_reward(self, reward: float) -> None:
        """Accumulate per-step reward into this trace's total."""
        self.reward += float(reward)

    def close(self) -> None:
        """Flush video tails (idempotent). Trace exit reporting is the caller's job."""
        if self._video is not None:
            self._video.finalize()
            self._video = None


class JobRecorder:
    """Records a vectorized env as one Job of per-episode traces.

    Construct once with the batch size, call :meth:`record` after every
    ``env.step`` with the batched tensors, :meth:`close` at the end. A subset of
    slots (``record_indices``) gets rich traces; each ``done[i]`` closes that
    slot's trace (reporting reward — env-reported ``success`` outranks
    accumulated shaped reward) and opens a fresh one, all under one Job.
    """

    def __init__(
        self,
        name: str,
        num_envs: int,
        *,
        record_indices: list[int] | None = None,
        fps: int = 10,
        seed: int | None = None,
        group_id: str | None = None,
        model: str | None = None,
        job_id: str | None = None,
        prompt: str | None = None,
        action_names: list[str] | None = None,
        state_names: dict[str, list[str]] | None = None,
    ) -> None:
        self.name = name
        self.num_envs = num_envs
        self.fps = fps
        self.seed = seed
        self.group_id = group_id
        self.model = model
        self.job_id = job_id or uuid.uuid4().hex
        self._prompt = prompt
        self._action_names = action_names
        self._state_names = state_names
        #: Extra metadata merged into each newly opened trace (e.g. the episode's
        #: task parametrization). Mutable: set it before the episode's traces open.
        self.extra_metadata: dict[str, Any] = {}
        if record_indices is None:
            record_indices = list(range(min(num_envs, 4)))  # a few representative slots
        self.record_indices = [i for i in record_indices if 0 <= i < num_envs]
        self._episode = dict.fromkeys(self.record_indices, 0)
        self._tick = dict.fromkeys(self.record_indices, 0)  # per-slot, per-episode tick
        self._rec: dict[int, TraceRecorder | None] = dict.fromkeys(self.record_indices)
        self._meta: dict[int, dict[str, Any]] = {}
        _report_sync(f"/trace/job/{self.job_id}/enter", {"name": name, "group": 1})
        logger.info("hud vec job: %s", self.job_url)

    @property
    def job_url(self) -> str:
        from hud.settings import settings

        return f"{settings.hud_web_url}/jobs/{self.job_id}"

    def _open(self, i: int) -> TraceRecorder:
        # Deterministic trace ids (reproducible, idempotent re-uploads) when seeded.
        if self.seed is not None:
            key = f"hud.vec:{self.job_id}:{i}:{self._episode[i]}:{self.seed}"
            trace_id = uuid.uuid5(uuid.NAMESPACE_URL, key).hex
        else:
            trace_id = uuid.uuid4().hex
        self._meta[i] = {
            "env_index": i,
            "episode_index": self._episode[i],
            "seed": self.seed,
            **self.extra_metadata,
        }
        _report_sync(
            f"/trace/{trace_id}/enter",
            {"job_id": self.job_id, "group_id": self.group_id, "model": self.model},
        )
        rec = TraceRecorder(
            trace_id=trace_id,
            fps=self.fps,
            prompt=self._prompt,
            action_names=self._action_names,
            state_names=self._state_names,
        )
        self._rec[i] = rec
        return rec

    def _close(self, i: int, rec: TraceRecorder, *, success: bool | None = None) -> None:
        rec.close()
        reward = rec.reward if success is None else float(success)
        meta = {**self._meta.pop(i, {}), **({"success": success} if success is not None else {})}
        _report_sync(
            f"/trace/{rec.trace_id}/exit",
            {"status": "completed", "reward": reward, "metadata": meta or None},
        )
        self._rec[i] = None
        self._episode[i] += 1  # the next obs opens a fresh trace
        self._tick[i] = 0

    def record(
        self,
        *,
        obs: Any | None = None,
        frames: dict[str, Any] | None = None,
        action: Any | None = None,
        reward: Any | None = None,
        done: Any | None = None,
        success: Any | None = None,
    ) -> None:
        """Record one batched step. Pass the tensors straight from ``env.step``.

        On ``done[i]`` the slot's episode is closed (final reward attributed) and
        the post-reset observation returned on the same step is skipped, so frames
        never bleed across the episode boundary.
        """
        done_np = to_numpy(done) if done is not None else None
        reward_np = to_numpy(reward) if reward is not None else None
        success_np = to_numpy(success) if success is not None else None

        for i in self.record_indices:
            rec = self._rec[i] or self._open(i)
            if reward_np is not None:
                rec.add_reward(float(reward_np[i]))
            if done_np is not None and bool(done_np[i]):
                self._close(i, rec, success=bool(success_np[i]) if success_np is not None else None)
                continue  # the returned obs belongs to the next episode
            data: dict[str, Any] = {}
            if obs is not None:
                sliced = obs if isinstance(obs, dict) else {"obs": obs}
                data.update({k: to_numpy(v)[i] for k, v in sliced.items()})
            data.update({k: to_numpy(v)[i] for k, v in (frames or {}).items()})
            if data:
                rec.record_observation(data, tick=self._tick[i])
            if action is not None:
                rec.record_inference(to_numpy(action)[i], tick=self._tick[i])
            self._tick[i] += 1

    def close_slots(self) -> None:
        """Close every open per-slot trace (an explicit mid-run reset ends its episodes)."""
        for i in self.record_indices:
            rec = self._rec[i]
            if rec is not None:
                self._close(i, rec)

    def close(self) -> None:
        """Close open traces and flush all telemetry to the platform."""
        self.close_slots()
        from hud.telemetry.exporter import flush

        flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["JobRecorder", "TraceRecorder", "to_numpy"]
