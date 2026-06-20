"""Per-episode recording for robot rollouts — telemetry, plus an optional LeRobot dataset.

The agent loop hands every tick to one :class:`Recorder`. It always streams the telemetry
the HUD viewer needs (an :class:`~hud.agents.types.ObservationStep` of numeric state +
per-camera H.264 video); when ``save`` is on it *also* appends each
``(observation, executed action)`` pair to a LeRobot v3 dataset for offline
training/finetuning.

Saving is opt-in (the agent's ``save`` flag — the ``--save`` runner flag), so the heavy
LeRobot/PyAV imports stay deferred until a dataset is actually built. One dataset spans the
whole run (every episode the shared agent drives appends to it) and is finalized at process
exit, optionally pushed to the HF Hub. Destination + push come from the environment:

- ``RECORD_DIR``  — dataset root (default ``./data`` from where the rollout launched)
- ``HF_REPO``     — HF namespace to also push to (needs ``HF_TOKEN``)
- ``HF_PRIVATE``  — push the dataset private
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

from hud.agents.types import InferenceStep, ObservationStep
from hud.telemetry.context import get_current_trace_id

from .video import VideoStreamer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hud.capabilities.robot import RobotClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)


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


class Recorder:
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


__all__ = ["Recorder"]
