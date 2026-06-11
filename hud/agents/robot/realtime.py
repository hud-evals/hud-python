"""Base agent for the realtime (free-running) ``robot`` path.

Where :class:`~hud.agents.robot.agent.RobotAgent` drives a strictly synchronous
one-action-per-step loop, a realtime agent is a *client*: the env free-runs and
streams observations (each carrying a ``meta`` block), and the agent decides *when*
to infer based on how many actions remain buffered env-side. When
``queue_remaining <= threshold`` it runs a chunk inference and ships the whole chunk
back via :meth:`RobotClient.send_chunk`; the env-side ``ActionProvider`` merges it
per the active mode.

For RTC the agent also conditions inference on the unexecuted prefix. Rather than
re-normalizing the env's executable prefix, the agent keeps the *raw* (model-space)
chunk it last produced and reconstructs the model-space prefix from the observation
index arithmetic — this is exactly the model-space version of the env's remaining
queue (the env merge is a plain drop-``d``/replace in RTC mode), so it is both
correct and free of lossy re-normalization.

Subclasses implement :meth:`infer_chunk`.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar

from hud.capabilities.robot import RobotClient

from .agent import RobotAgent

if TYPE_CHECKING:
    import numpy as np

    from hud.client import Run


class RealtimeRobotAgent(RobotAgent):
    """Chunk-streaming client for a :class:`RealtimeRobotBridge` env."""

    _infer_executor: ThreadPoolExecutor | None = None

    @property
    def infer_executor(self) -> ThreadPoolExecutor:
        """A single dedicated thread for all policy inference (incl. warmup).

        CUDA graphs (and torch.compile capture) are thread-affine: a graph captured
        on one thread cannot be replayed on another. Running every ``infer_chunk``
        — and the ``warmup`` that primes the same graphs — on one fixed thread keeps
        them valid across the whole run (all episodes in this process). It persists
        for the process lifetime on purpose: tearing it down per episode would force
        a fresh, expensive capture each time.
        """
        if self._infer_executor is None:
            self._infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        return self._infer_executor

    # Realtime episodes trigger only a handful of inferences, so log each one.
    log_every: ClassVar[int] = 1

    async def select_action(self, obs: dict[str, Any]) -> np.ndarray:  # pragma: no cover - not used
        raise NotImplementedError(
            "Realtime agents produce chunks via infer_chunk(), not select_action()."
        )

    @abstractmethod
    def infer_chunk(
        self, obs: dict[str, Any], meta: dict[str, Any], prefix_model: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Infer from one observation.

        Returns ``(exec_chunk, raw_chunk)`` where ``exec_chunk`` is the executable
        ``[T, A]`` chunk to send to the env, and ``raw_chunk`` is the model-space
        ``[T, A]`` chunk retained for the next RTC prefix (or ``None`` if unused).
        ``prefix_model`` is the model-space unexecuted prefix for RTC conditioning
        (``None`` for non-RTC modes or the first inference).
        """

    def on_episode_start(self, run: Run, client: RobotClient, *, prompt: str) -> None:
        super().on_episode_start(run, client, prompt=prompt)
        # Configure this episode from the env's realtime contract: the env is
        # authoritative about mode/threshold/horizon; the agent just adapts.
        # TODO: consider changing inference mode passing
        rt = client.contract.get("inference", {})
        self._mode: str = rt.get("inference_mode", "sync")
        self._threshold: int = int(rt.get("threshold", 0))
        # RTC stitching window (w/ delay): [0,delay) frozen, [delay,H) decaying blend, [H,T) free.
        # Larger H = smoother but less reactive.
        self._execution_horizon: int = int(rt.get("execution_horizon", 25))
        self._rtc: bool = self._mode == "rtc"
        self._last_raw_chunk: np.ndarray | None = None
        self._last_chunk_obs_index: int | None = None
        print(
            f"[agent] realtime mode={self._mode} threshold={self._threshold} "
            f"exec_horizon={self._execution_horizon}",
            flush=True,
        )

    def _model_prefix(self, obs_index: int | None) -> np.ndarray | None:
        """Model-space unexecuted prefix = tail of the last raw chunk past ``obs_index``."""
        if not self._rtc or self._last_raw_chunk is None or self._last_chunk_obs_index is None:
            return None
        if obs_index is None:
            return None
        # tail at moment the last obs was sent from env
        k = max(0, int(obs_index) - int(self._last_chunk_obs_index))
        tail = self._last_raw_chunk[k:]
        return tail if len(tail) > 0 else None

    async def __call__(self, run: Run, *, max_steps: int | None = None) -> None:
        if max_steps is None:
            max_steps = getattr(self, "max_steps", 4000)
        cap = run.client.binding(self.robot_protocol)
        client = await RobotClient.connect(cap)
        try:
            self.setup_robot(client)
            prompt = run.prompt
            if not isinstance(prompt, str):
                raise TypeError(
                    f"run.prompt must be a str, got {type(prompt).__name__}: {prompt!r}"
                )
            self.on_episode_start(run, client, prompt=prompt)
            print(f"[agent] realtime episode started: {prompt!r}", flush=True)

            # "pending" is an inference "in-flight" guard
            pending = False  # True = in middle of inference, False = free to infer
            chunk_sent_at_obs_index = -1
            n_inferences = 0
            for step in range(max_steps):
                obs = await client.get_observation()
                if self.should_stop(obs, step=step, max_steps=max_steps):
                    print(f"[agent] env reported terminated at step {step}", flush=True)
                    break
                meta = obs.get("meta") or {}
                recv_obs_index = meta.get("obs_index")
                qr = int(meta.get("queue_remaining", 0))

                # obs (index) that was used to compute the current active env chunk
                active_chunk_obs_index = int(meta.get("active_chunk_obs_index", -1))
                if active_chunk_obs_index >= chunk_sent_at_obs_index:
                    # chunk "landed" in the env queue — clear the in-flight guard
                    pending = False
                elif (
                    pending
                    and recv_obs_index is not None
                    # note: horizon has to be longer than inference delay
                    and recv_obs_index - chunk_sent_at_obs_index > self._execution_horizon
                ):
                    # (backstop) if acknowledgement doesn't arrive in horizon, assume chunk lost
                    pending = False

                if not pending and qr <= self._threshold:
                    prefix_model = self._model_prefix(recv_obs_index)
                    # Run on the dedicated inference thread so CUDA-graph
                    # capture/replay stays on the one thread that warmup primed.
                    loop = asyncio.get_running_loop()
                    exec_chunk, raw_chunk = await loop.run_in_executor(
                        self.infer_executor, self.infer_chunk, obs, meta, prefix_model
                    )
                    self._last_raw_chunk = raw_chunk
                    self._last_chunk_obs_index = recv_obs_index
                    await client.send_chunk(
                        exec_chunk, obs_index=recv_obs_index, delay_used=meta.get("delay")
                    )
                    pending = True  # in the middle of inference
                    chunk_sent_at_obs_index = (
                        recv_obs_index if recv_obs_index is not None else chunk_sent_at_obs_index
                    )
                    n_inferences += 1
                    if self.log_every and n_inferences % self.log_every == 0:
                        print(
                            f"[agent] inference #{n_inferences} | obs_index={recv_obs_index} "
                            f"qr={qr} delay={meta.get('delay')} chunk_len={len(exec_chunk)} "
                            f"underrun_hint={'yes' if qr == 0 else 'no'}",
                            flush=True,
                        )
            else:
                print(f"[agent] reached max_steps={max_steps}", flush=True)

            run.trace.done = True
            run.trace.content = "done"
            run.trace.isError = False
        finally:
            await client.close()


__all__ = ["RealtimeRobotAgent"]
