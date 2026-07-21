"""Opt-in LeRobot v3 dataset writing for robot rollouts.

Each rollout holds a :class:`DatasetWriter` that buffers its ``(observation,
executed action)`` frames and commits whole episodes into one process-shared
dataset — so concurrent rollouts (e.g. :class:`~hud.agents.robot.batching.BatchedAgent`
clones) record into a single dataset instead of one shard each. The shared
dataset (and an ``atexit`` finalizer that flushes open buffers) is created on
the first frame. A class lock serializes commits so episodes stay contiguous.
Finalized at process exit (or an explicit :meth:`finalize`), optionally pushed
to the HF Hub.
The contract drives the schema with no extra wiring. Destination + push come
from the environment:

- ``RECORD_DIR``  — dataset root (default ``./data`` from where the rollout launched)
- ``HF_REPO``     — HF namespace to also push to (needs ``HF_TOKEN``)
- ``HF_PRIVATE``  — push the dataset private
"""

from __future__ import annotations

import atexit
import importlib.util
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _lerobot_features(contract: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Map a robot contract to LeRobot ``features`` + a wire-key -> LeRobot-key map.

    Image obs -> ``observation.images.<leaf>`` (video); the lone vector obs ->
    ``observation.state`` (else ``observation.<leaf>``); the action -> ``action``.
    String obs are dropped (LeRobot carries the prompt as its per-frame ``task``).
    """
    feats = contract.get("features", {})
    vectors = [
        n
        for n, f in feats.items()
        if f.get("role") == "observation" and not _is_image(f) and f.get("dtype") != "string"
    ]
    single_state = len(vectors) == 1

    features: dict[str, dict[str, Any]] = {}
    key_map: dict[str, str] = {}
    for name, f in feats.items():
        role, dtype, shape = f.get("role"), f.get("dtype"), tuple(f.get("shape") or ())
        leaf = name.split("/")[-1]  # contract keys are slash-paths; LeRobot wants the leaf
        if role == "observation" and dtype != "string":
            if _is_image(f):
                key, dtype = f"observation.images.{leaf}", "video"
            elif leaf == "state" or single_state:
                key = "observation.state"
            else:
                key = f"observation.{leaf}"
            # Derived contracts omit dtype/shape; default the dtype, and leave a
            # missing shape empty for add() to fill from the first real frame.
            features[key] = {"dtype": dtype or "float32", "shape": shape, "names": _names(f, leaf)}
            key_map[name] = key
        elif role == "action":
            features["action"] = {
                "dtype": dtype or "float32",
                "shape": shape,
                "names": _names(f, "act"),
            }
    return features, key_map


def _is_image(feature: dict[str, Any]) -> bool:
    """A camera feature: authored contracts say ``dtype: image``, derived ones tag
    the (load-bearing) image ``type`` — accept both."""
    return feature.get("dtype") == "image" or feature.get("type") in ("rgb", "bgr", "gray", "depth")


def _names(feature: dict[str, Any], base: str) -> list[str]:
    """Contract per-element labels, else positional defaults sized to the (rank-1) shape."""
    if names := feature.get("names"):
        return list(names)
    if _is_image(feature):
        return ["height", "width", "channel"]
    return [f"{base}_{i}" for i in range(int((feature.get("shape") or [1])[0]))]


class DatasetWriter:
    """Buffers one rollout's frames; commits whole episodes to a process-shared
    LeRobot v3 dataset. A no-op shell when lerobot is missing (warned once) so
    telemetry-only runs never break."""

    # One dataset per process: concurrent rollouts (e.g. BatchedAgent clones) each
    # buffer their own episode but commit into the same root under ``_lock``.
    _ds: ClassVar[Any | None] = None
    _root: ClassVar[Path | None] = None
    _repo_id: ClassVar[str] = ""
    # Serialize create / add_frame / save_episode / finalize across rollouts.
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, contract: dict[str, Any], *, fps: int) -> None:
        self._contract = contract
        self._fps = fps
        self._features, self._key_map = _lerobot_features(contract)
        self._frames: list[dict[str, Any]] = []  # this rollout's pending episode
        self._enabled = importlib.util.find_spec("lerobot") is not None
        if not self._enabled:
            logger.warning(
                "save=True but lerobot is not installed; streaming telemetry only "
                "(pip install 'lerobot[dataset]')"
            )

    def add(self, data: dict[str, Any], action: NDArray[Any], *, task: str) -> None:
        """One frame: the wire observation dict + the executed env-space action."""
        if not self._enabled:
            return
        # Derived contracts carry no shapes; fill from the first real frame (no-op after).
        for wire, key in self._key_map.items():
            if not self._features[key]["shape"] and wire in data:
                self._features[key]["shape"] = tuple(np.shape(data[wire]))
        if not self._features["action"]["shape"]:
            self._features["action"]["shape"] = tuple(np.shape(action))
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
        row["task"] = task
        # Open the shared dataset on the first frame so atexit can flush if we
        # die before end_episode (still in-memory until then; kill -9 loses it).
        with DatasetWriter._lock:
            self._ensure_dataset()
            self._frames.append(row)

    def end_episode(self) -> None:
        """Commit this rollout's buffered episode to the shared dataset.

        Whole episodes stay contiguous: ``_lock`` serializes create / add_frame /
        save_episode so concurrent BatchedAgent rollouts cannot interleave frames.
        """
        if not self._frames:
            return
        with DatasetWriter._lock:
            ds = self._ensure_dataset()
            for row in self._frames:
                ds.add_frame(row)
            ds.save_episode()
            self._frames.clear()

    def finalize(self) -> None:
        """Flush, write the parquet footer + optionally push to the Hub. Idempotent."""
        with DatasetWriter._lock:
            # Re-entrant: end_episode takes the same lock when frames remain.
            self.end_episode()
            ds, DatasetWriter._ds = DatasetWriter._ds, None
            if ds is None:
                return
            root, repo_id = DatasetWriter._root, DatasetWriter._repo_id
            ds.finalize()
            print(f"[agent] saved LeRobot dataset -> {root}", flush=True)
            if not os.environ.get("HF_REPO"):
                return
            private = os.environ.get("HF_PRIVATE", "0") not in ("0", "", "false", "False")
            try:  # best-effort: the on-disk dataset is the source of truth
                ds.push_to_hub(private=private)
                print(f"[agent] pushed -> https://huggingface.co/datasets/{repo_id}", flush=True)
            except Exception as exc:
                logger.exception("HF push failed for %s", repo_id)
                print(
                    f"[agent] WARNING: HF push failed: {exc!r} (dataset still on disk)", flush=True
                )

    def _ensure_dataset(self) -> Any:
        """Return the process-shared dataset, creating it on first frame.

        Caller must hold ``_lock`` (create is check-then-act on ``_ds``).
        """
        if DatasetWriter._ds is not None:
            return DatasetWriter._ds
        lerobot_dataset: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")

        name = self._contract.get("robot_type") or "robot"
        # Stamp + random tag: unique root per process even across simultaneous launches.
        tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        record_dir = Path(os.environ.get("RECORD_DIR", "data"))
        record_dir.mkdir(parents=True, exist_ok=True)
        DatasetWriter._root = record_dir / f"{name}_{tag}"
        DatasetWriter._repo_id = f"{os.environ.get('HF_REPO') or 'hud'}/{name}_{tag}"
        # LeRobotDataset.create requires a fresh root; images encode to per-episode video.
        DatasetWriter._ds = lerobot_dataset.LeRobotDataset.create(
            repo_id=DatasetWriter._repo_id,
            fps=self._fps,
            features=self._features,
            root=DatasetWriter._root,
            robot_type=self._contract.get("robot_type"),
            use_videos=True,
        )
        atexit.register(self.finalize)  # keep the dataset loadable on abrupt exits
        print(f"[agent] recording LeRobot dataset -> {DatasetWriter._root}", flush=True)
        return DatasetWriter._ds


__all__ = ["DatasetWriter"]
