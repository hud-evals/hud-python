"""Recording for robot/sim rollouts: telemetry streaming, plus an optional LeRobot dataset.

Three recorders, smallest to largest, all streaming the *same* robot telemetry the HUD
viewer plays (numeric :class:`~hud.agents.types.ObservationStep` state + per-camera H.264
video) via the existing exporter:

- :class:`Recorder` — one trace. Emits spans with an *explicit* ``trace_id`` (no rollout
  contextvar, no ``Run``), so it works from a bare synchronous loop or any thread. The
  primitive the vectorized recorder is built from.
- :class:`VecRecorder` — a whole vectorized env as one **Job** of per-episode traces. One
  :class:`Recorder` per recorded env slot; on each ``done[i]`` (auto-reset) it closes that
  slot's trace and opens a fresh one, so every episode becomes its own fully-saved trace.
- :class:`EpisodeRecorder` — the agent-loop recorder: records onto a :class:`Run` (so steps
  land on ``run.trace`` for grading/training) and, when ``save`` is on, *also* appends each
  ``(observation, executed action)`` pair to a LeRobot v3 dataset for offline training.

Saving is opt-in (the agent's ``save`` flag — the ``--save`` runner flag), so the heavy
LeRobot/PyAV imports stay deferred until a dataset is actually built. One dataset spans the
whole run (every episode the shared agent drives appends to it) and is finalized at process
exit, optionally pushed to the HF Hub. Destination + push come from the environment:

- ``RECORD_DIR``  — dataset root (default ``./data`` from where the rollout launched)
- ``HF_REPO``     — HF namespace to also push to (needs ``HF_TOKEN``)
- ``HF_PRIVATE``  — push the dataset private

Everything HUD-specific (job/trace lifecycle, trace-id assignment, span building, video
encoding, upload) lives here; a benchmark just does ``VecRecorder(...)`` -> ``record`` ->
``close``.
"""

from __future__ import annotations

import atexit
import importlib.util
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hud.agents.types import InferenceStep, ObservationStep, StateFeature
from hud.telemetry.context import get_current_trace_id
from hud.utils.platform import PlatformClient

from .video import VideoStreamer

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from hud.capabilities.robot import RobotClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)


def _reporting_enabled() -> bool:
    from hud.settings import settings

    return bool(settings.telemetry_enabled and settings.api_key)


def _report_sync(path: str, payload: dict[str, Any]) -> None:
    """Best-effort sync POST to the platform (job/trace lifecycle). No-op without an API key.

    Mirrors the async ``hud.eval.job`` reporting so a sync vec-env loop never needs an event
    loop, and never fails the run when the platform is unreachable.
    """
    if not _reporting_enabled():
        return
    body = {k: v for k, v in payload.items() if v is not None}
    try:
        PlatformClient.from_settings().post(path, json=body)
    except Exception as exc:  # reporting is fire-and-forget
        logger.warning("platform report %s failed: %s", path, exc)


def _to_numpy(x: Any) -> NDArray[Any]:
    """Coerce a torch tensor / array / scalar to a numpy array (no torch dependency)."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _lerobot_features(contract: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Map a robot contract to LeRobot ``features`` + a wire-key -> LeRobot-key map.

    Image obs -> ``observation.images.<leaf>`` (video); the lone vector obs ->
    ``observation.state`` (else ``observation.<leaf>``); the action -> ``action``. String
    obs are dropped (LeRobot carries the prompt as its per-frame ``task``).
    """
    feats = contract.get("features", {})
    vectors = [
        n
        for n, f in feats.items()
        if f.get("role") == "observation" and f.get("dtype") not in ("image", "string")
    ]
    single_state = len(vectors) == 1

    features: dict[str, dict[str, Any]] = {}
    key_map: dict[str, str] = {}
    for name, f in feats.items():
        role, dtype, shape = f.get("role"), f.get("dtype"), tuple(f.get("shape") or ())
        leaf = name.split("/")[-1]  # contract keys are slash-paths; LeRobot wants the leaf
        if role == "observation" and dtype != "string":
            if dtype == "image":
                key, dtype = f"observation.images.{leaf}", "video"
            elif leaf == "state" or single_state:
                key = "observation.state"
            else:
                key = f"observation.{leaf}"
            features[key] = {"dtype": dtype, "shape": shape, "names": _feature_names(f, leaf)}
            key_map[name] = key
        elif role == "action":
            features["action"] = {"dtype": dtype, "shape": shape, "names": _feature_names(f, "act")}
    return features, key_map


def _feature_names(feature: dict[str, Any], base: str) -> list[str]:
    """Contract per-element labels, else positional defaults sized to the (rank-1) shape."""
    if names := feature.get("names"):
        return list(names)
    if feature.get("dtype") == "image":
        return ["height", "width", "channel"]
    return [f"{base}_{i}" for i in range(int((feature.get("shape") or [1])[0]))]


class EpisodeRecorder:
    """Records one agent's rollouts: always telemetry, optionally a LeRobot dataset.

    The agent owns a single instance for its lifetime and routes *all* recording through
    it: :meth:`begin`/:meth:`end` bracket each episode, :meth:`record_observation` /
    :meth:`record_inference` / :meth:`record_action` feed each tick (the first two write
    telemetry steps onto the run passed to :meth:`begin`; the last completes a LeRobot
    frame), and :meth:`save` (also an ``atexit`` hook) finalizes the cross-episode dataset.
    With ``save=False`` only the telemetry path runs and the LeRobot deps are never imported.
    """

    def __init__(self, client: RobotClient, *, save: bool = False) -> None:
        self._obs_space = client.spaces()[1]
        self._fps = client.get_control_rate()
        self._contract = client.contract
        # Telemetry is always on; saving also needs lerobot installed.
        if save and importlib.util.find_spec("lerobot") is None:
            logger.warning(
                "save=True but lerobot is not installed; streaming telemetry only "
                "(pip install 'lerobot[dataset]')"
            )
            save = False
        self._save = save
        self._features: dict[str, dict[str, Any]] = {}
        self._key_map: dict[str, str] = {}
        if save:
            self._features, self._key_map = _lerobot_features(self._contract)

        self._video: VideoStreamer | None = None  # per-episode
        self._run: Run | None = None
        self._task = ""
        self._pending: dict[str, Any] | None = None  # last obs awaiting its action
        # LeRobot dataset spans every episode; created lazily on the first frame.
        self._ds: Any | None = None
        self._root: Path | None = None
        self._repo_id = ""
        if save:
            atexit.register(self.save)  # finalize even on an abrupt exit (parquet footer)

    # ── episode lifecycle (called from the agent harness) ─────────────────────
    def begin(self, run: Run, prompt: str) -> None:
        """Open an episode: fresh per-camera video stream + the task prompt."""
        self._run = run
        self._task = prompt
        self._pending = None
        self._video = VideoStreamer(fps=self._fps, trace_id=get_current_trace_id())

    def record_observation(self, obs: dict[str, Any], *, tick: int) -> None:
        """One observation: numeric-state span + per-camera video (always streamed)."""
        assert self._run is not None and self._video is not None  # set in begin()
        self._run.record(ObservationStep.from_obs(obs, tick=tick, obs_space=self._obs_space))
        self._video.record(obs)
        self._pending = obs.get("data")  # paired with the action in record_action()

    def record_inference(self, chunk: NDArray[Any], *, tick: int) -> None:
        """One re-inference: the freshly inferred ``[T, A]`` action chunk, onto the run."""
        assert self._run is not None  # set in begin()
        self._run.record(InferenceStep(tick=tick, chunk=chunk.tolist(), chunk_length=len(chunk)))

    def record_action(self, action: NDArray[Any]) -> None:
        """The executed (env-space) action: completes the pending LeRobot frame."""
        if self._save and self._pending is not None:
            self._add_frame(self._pending, action)
        self._pending = None

    def end(self) -> None:
        """Close the episode: flush video tails; commit the LeRobot episode (if any frames)."""
        if self._video is not None:
            self._video.finalize()
        if self._ds is not None and self._ds.has_pending_frames():
            self._ds.save_episode()

    def save(self) -> None:
        """Finalize the dataset (writes the parquet footer) + optionally push to the Hub.

        Idempotent; registered with ``atexit`` so the dataset stays loadable even if the
        process exits without an explicit call.
        """
        if not self._save or self._ds is None:
            return
        self._save = False  # idempotent across the explicit call + the atexit hook
        self._ds.finalize()
        print(f"[agent] saved LeRobot dataset -> {self._root}", flush=True)
        if not os.environ.get("HF_REPO"):
            return
        private = os.environ.get("HF_PRIVATE", "0") not in ("0", "", "false", "False")
        try:  # best-effort: the on-disk dataset is the source of truth
            self._ds.push_to_hub(private=private)
            print(f"[agent] pushed -> https://huggingface.co/datasets/{self._repo_id}", flush=True)
        except Exception as exc:
            logger.exception("HF push failed for %s", self._repo_id)
            print(f"[agent] WARNING: HF push failed: {exc!r} (dataset still on disk)", flush=True)

    # ── LeRobot writing ───────────────────────────────────────────────────────
    def _add_frame(self, data: dict[str, Any], action: NDArray[Any]) -> None:
        ds = self._ensure_dataset()
        row: dict[str, Any] = {}
        for wire, key in self._key_map.items():
            value = data.get(wire)
            if value is None:
                logger.warning("obs missing contract feature %r; skipping frame", wire)
                return
            ft = self._features[key]
            row[key] = (
                np.ascontiguousarray(value, dtype=np.uint8)  # bridge images are uint8 HWC
                if ft["dtype"] in ("video", "image")
                else np.asarray(value, dtype=ft["dtype"]).reshape(ft["shape"])
            )
        act_ft = self._features["action"]
        row["action"] = np.asarray(action, dtype=act_ft["dtype"]).reshape(act_ft["shape"])
        row["task"] = self._task
        ds.add_frame(row)

    def _ensure_dataset(self) -> Any:
        if self._ds is not None:
            return self._ds
        lerobot_dataset: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")

        name = self._contract.get("robot_type") or "robot"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        # Unique per recorder so concurrent (batched) rollouts never share a root;
        # tie it to the trace id when there is one so a shard maps back to its trace.
        tag = (get_current_trace_id() or uuid.uuid4().hex)[:8]
        # Default under ./data (relative to where the rollout was launched), created if absent.
        record_dir = Path(os.environ.get("RECORD_DIR", "data"))
        record_dir.mkdir(parents=True, exist_ok=True)
        self._root = record_dir / f"{name}_{stamp}_{tag}"
        self._repo_id = f"{os.environ.get('HF_REPO') or 'hud'}/{name}_{stamp}_{tag}"
        # LeRobotDataset.create requires a fresh root; images encode to per-episode video.
        self._ds = lerobot_dataset.LeRobotDataset.create(
            repo_id=self._repo_id,
            fps=self._fps,
            features=self._features,
            root=self._root,
            robot_type=self._contract.get("robot_type"),
            use_videos=True,
        )
        print(f"[agent] recording LeRobot dataset -> {self._root}", flush=True)
        return self._ds


# ── lightweight telemetry recorders (vec-env / bare-loop, no Run, no LeRobot) ────────────


def _observation_step(obs: dict[str, Any], *, tick: int) -> ObservationStep:
    """Build an :class:`ObservationStep` (numeric state) from a flat per-slot obs dict.

    Camera frames travel as H.264 video, so array obs with rank >= 2 are skipped; flat
    numeric vectors become unlabelled :class:`StateFeature`s. Robust to scalars (unlike
    :meth:`ObservationStep.from_obs`, which expects a robot-contract ``obs['data']``).
    """
    state: dict[str, StateFeature] = {}
    for name, val in obs.items():
        arr = _to_numpy(val)
        if arr.ndim >= 2:
            continue
        flat = np.atleast_1d(arr).astype(float).ravel()
        if 0 < flat.size <= 512:
            state[name] = StateFeature(values=flat.tolist())
    return ObservationStep(tick=tick, state=state)


class Recorder:
    """Streams one trace's telemetry to the platform: state, per-camera video, actions.

    Standalone: emits spans with an explicit ``trace_id`` rather than relying on the rollout
    contextvar, so it runs inside a plain (or vectorized) loop. Reuses the robot step schema
    and :class:`~hud.agents.robot.video.VideoStreamer`, so the existing exporter and trace
    viewer ingest it unchanged.
    """

    def __init__(
        self,
        *,
        trace_id: str,
        job_id: str | None = None,
        group_id: str | None = None,
        fps: int = 10,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
        enter: bool = True,
    ) -> None:
        self.trace_id = trace_id
        self.job_id = job_id
        self._fps = fps
        self._tick = 0
        self._reward = 0.0
        self._metadata = metadata or {}
        self._video: VideoStreamer | None = None  # lazy, one per trace
        self._closed = False
        if enter:
            _report_sync(
                f"/trace/{trace_id}/enter",
                {"job_id": job_id, "group_id": group_id, "model": model},
            )

    def add_reward(self, reward: float) -> None:
        """Accumulate per-step reward into this trace's total (reported on close)."""
        self._reward += float(reward)

    def record(
        self,
        *,
        obs: dict[str, Any] | None = None,
        frames: dict[str, Any] | None = None,
        action: Any | None = None,
    ) -> None:
        """Record one tick: numeric-state span, per-camera video, and the executed action.

        ``obs`` is this slot's observation dict (flat numeric vectors); ``frames`` maps a
        camera name to its ``HxWxC`` frame; ``action`` is the executed action vector.
        """
        if obs is not None:
            _observation_step(obs, tick=self._tick).emit(trace_id=self.trace_id)
        if frames:
            if self._video is None:
                self._video = VideoStreamer(fps=self._fps, trace_id=self.trace_id)
            self._video.record({"data": frames})
        if action is not None:
            chunk = _to_numpy(action).astype(float).ravel().tolist()
            InferenceStep(tick=self._tick, chunk=[chunk], chunk_length=1).emit(
                trace_id=self.trace_id
            )
        self._tick += 1

    def close(
        self,
        *,
        status: str = "completed",
        reward: float | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Flush video tails and report the trace exit (status / reward / metadata)."""
        if self._closed:
            return
        self._closed = True
        if self._video is not None:
            self._video.finalize()
        meta = {**self._metadata, **(metadata or {})}
        _report_sync(
            f"/trace/{self.trace_id}/exit",
            {
                "status": status,
                "reward": self._reward if reward is None else reward,
                "error": error,
                "metadata": meta or None,
            },
        )


class VecRecorder:
    """Records a vectorized env as one Job of per-episode traces.

    Construct it once with the batch size, call :meth:`record` after every ``env.step`` with
    the batched tensors, and :meth:`close` at the end. A configurable subset of slots
    (``record_indices``) gets rich traces (state + video); each ``done[i]`` closes that slot's
    current trace and opens a new one, so a slot that plays many episodes produces many
    traces — all under one Job at :attr:`job_url`.
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
    ) -> None:
        self.name = name
        self.num_envs = num_envs
        self.fps = fps
        self.seed = seed
        self.group_id = group_id
        self.model = model
        self.job_id = job_id or uuid.uuid4().hex
        if record_indices is None:
            record_indices = list(range(min(num_envs, 4)))  # a few representative slots
        self.record_indices = [i for i in record_indices if 0 <= i < num_envs]

        self._episode = dict.fromkeys(self.record_indices, 0)
        self._rec: dict[int, Recorder | None] = dict.fromkeys(self.record_indices)
        _report_sync(f"/trace/job/{self.job_id}/enter", {"name": name, "group": 1})
        logger.info("hud vec job: %s", self.job_url)

    @property
    def job_url(self) -> str:
        from hud.settings import settings

        return f"{settings.hud_web_url}/jobs/{self.job_id}"

    def _new_trace_id(self, i: int) -> str:
        # Deterministic (reproducible, idempotent re-uploads) when a seed is given.
        if self.seed is not None:
            key = f"hud.vec:{self.job_id}:{i}:{self._episode[i]}:{self.seed}"
            return uuid.uuid5(uuid.NAMESPACE_URL, key).hex
        return uuid.uuid4().hex

    def _open(self, i: int) -> Recorder:
        rec = Recorder(
            trace_id=self._new_trace_id(i),
            job_id=self.job_id,
            group_id=self.group_id,
            fps=self.fps,
            model=self.model,
            metadata={"env_index": i, "episode_index": self._episode[i], "seed": self.seed},
        )
        self._rec[i] = rec
        return rec

    def record(
        self,
        *,
        obs: Any | None = None,
        frames: dict[str, Any] | None = None,
        action: Any | None = None,
        reward: Any | None = None,
        done: Any | None = None,
        info: dict[str, Any] | None = None,  # reserved for per-env metrics
    ) -> None:
        """Record one batched step. Pass the tensors straight from ``env.step``.

        On ``done[i]`` the slot's current episode is closed (final reward attributed to it)
        and a fresh trace is opened for the next episode; the post-reset observation returned
        on the same step is skipped so frames never bleed across the episode boundary.
        """
        done_np = _to_numpy(done) if done is not None else None
        reward_np = _to_numpy(reward) if reward is not None else None

        for i in self.record_indices:
            rec = self._rec[i] or self._open(i)
            if reward_np is not None:
                rec.add_reward(float(reward_np[i]))
            if done_np is not None and bool(done_np[i]):
                rec.close()  # closing episode keeps its own env/episode metadata
                self._episode[i] += 1
                self._rec[i] = None
                continue  # the returned obs belongs to the next episode
            rec.record(
                obs=_slice_obs(obs, i),
                frames=_slice_frames(frames, i),
                action=None if action is None else _to_numpy(action)[i],
            )

    def close(self) -> None:
        """Close any open traces and flush all telemetry to the platform."""
        for i in self.record_indices:
            rec = self._rec[i]
            if rec is not None:
                rec.close()
                self._rec[i] = None
        from hud.telemetry.exporter import flush

        flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _slice_obs(obs: Any | None, i: int) -> dict[str, Any] | None:
    """One slot's observation as a name->array dict (accepts a dict or a bare batched array)."""
    if obs is None:
        return None
    if isinstance(obs, dict):
        return {k: _to_numpy(v)[i] for k, v in obs.items()}
    return {"obs": _to_numpy(obs)[i]}


def _slice_frames(frames: dict[str, Any] | None, i: int) -> dict[str, Any] | None:
    """One slot's camera frames as a name->``HxWxC`` array dict."""
    if not frames:
        return None
    return {name: _to_numpy(arr)[i] for name, arr in frames.items()}


__all__ = ["EpisodeRecorder", "Recorder", "VecRecorder"]
