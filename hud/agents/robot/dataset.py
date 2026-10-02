"""Opt-in LeRobot v3 dataset writing for robot rollouts."""

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

    String observations are dropped (LeRobot carries the prompt as each frame's ``task``);
    features that map to the same key raise.
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
        # Full wire path → LeRobot slug (leaf alone collides for left/image + right/image).
        slug = name.replace("/", "_")
        leaf = name.split("/")[-1]
        if role == "observation" and dtype != "string":
            if _is_image(f):
                key, dtype = f"observation.images.{slug}", "video"
            elif leaf == "state" or single_state:
                key = "observation.state"
            else:
                key = f"observation.{slug}"
            if key in features:
                prior = next(w for w, k in key_map.items() if k == key)
                raise ValueError(
                    f"contract features {prior!r} and {name!r} both map to LeRobot key {key!r}"
                )
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
    """Authored contracts mark cameras ``dtype: image``; derived ones tag the image ``type``."""
    return feature.get("dtype") == "image" or feature.get("type") in ("rgb", "bgr", "gray", "depth")


def _names(feature: dict[str, Any], base: str) -> list[str]:
    """Contract per-element labels, else positional defaults sized to the (rank-1) shape."""
    if names := feature.get("names"):
        return list(names)
    if _is_image(feature):
        return ["height", "width", "channel"]
    return [f"{base}_{i}" for i in range(int((feature.get("shape") or [1])[0]))]


class DatasetWriter:
    """Buffers one rollout's frames and commits whole episodes to a shared LeRobot v3 dataset.

    Writers with the same schema and FPS share a dataset under ``RECORD_DIR`` (default
    ``./data``), finalized at exit and pushed to the ``HF_REPO`` namespace when that is set
    (``HF_PRIVATE`` makes it private). Without lerobot installed, it warns and records nothing.
    """

    # key -> (dataset, root, repo_id). Same schema/FPS shares; else a new root.
    _datasets: ClassVar[dict[tuple[Any, ...], tuple[Any, Path, str]]] = {}
    # Serialize create / add_frame / save_episode / finalize across rollouts.
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _open: ClassVar[set[DatasetWriter]] = set()  # uncommitted buffers for atexit
    _atexit_registered: ClassVar[bool] = False

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
            # lerobot's AV1 episode encoder fails or hangs, when the episode saves,
            # on a frame with a side under 32 px.
            if ft["dtype"] == "video" and min(np.shape(value)[:2]) < 32:
                raise ValueError(
                    f"camera {wire!r} frame of shape {np.shape(value)} cannot be recorded: "
                    "lerobot's episode video encoder needs at least 32 px on each side"
                )
            row[key] = (
                np.ascontiguousarray(value, dtype=np.uint8)  # bridge images are uint8 HWC
                if ft["dtype"] in ("video", "image")
                else np.asarray(value, dtype=ft["dtype"]).reshape(ft["shape"])
            )
        act_ft = self._features["action"]
        row["action"] = np.asarray(action, dtype=act_ft["dtype"]).reshape(act_ft["shape"])
        row["task"] = task
        # Open the dataset on the first frame so atexit can flush an unfinished episode
        # (frames stay in memory until then, so kill -9 loses them).
        with DatasetWriter._lock:
            self._ensure_dataset()
            self._frames.append(row)
            DatasetWriter._open.add(self)

    def end_episode(self) -> None:
        """Commit this rollout's buffered episode, contiguously, to the shared dataset."""
        if not self._frames:
            return
        with DatasetWriter._lock:
            ds = self._ensure_dataset()
            for row in self._frames:
                ds.add_frame(row)
            ds.save_episode()
            self._frames.clear()
            DatasetWriter._open.discard(self)

    @classmethod
    def finalize(cls) -> None:
        """Flush every open writer, write the parquet footer, optionally push. Idempotent."""
        with cls._lock:
            for writer in list(cls._open):
                writer.end_episode()  # re-entrant: end_episode takes the same lock
            datasets, cls._datasets = cls._datasets, {}
            private = os.environ.get("HF_PRIVATE", "0") not in ("0", "", "false", "False")
            push = bool(os.environ.get("HF_REPO"))
            for ds, root, repo_id in datasets.values():
                ds.finalize()
                print(f"[agent] saved LeRobot dataset -> {root}", flush=True)
                if not push:
                    continue
                try:  # best-effort: the on-disk dataset is the source of truth
                    ds.push_to_hub(private=private)
                    print(
                        f"[agent] pushed -> https://huggingface.co/datasets/{repo_id}",
                        flush=True,
                    )
                except Exception as exc:
                    logger.exception("HF push failed for %s", repo_id)
                    print(
                        f"[agent] WARNING: HF push failed: {exc!r} (dataset still on disk)",
                        flush=True,
                    )

    def _ensure_dataset(self) -> Any:
        """Get or create the dataset shared by this schema and FPS; caller holds ``_lock``."""
        # Share only when fps + robot_type + feature schema match.
        key = (
            self._fps,
            self._contract.get("robot_type") or "robot",
            tuple(
                (n, f.get("dtype"), tuple(f.get("shape") or ()), tuple(f.get("names") or ()))
                for n, f in sorted(self._features.items())
            ),
        )
        if key in DatasetWriter._datasets:
            return DatasetWriter._datasets[key][0]
        lerobot_dataset: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")

        name = self._contract.get("robot_type") or "robot"
        # Stamp + random tag: unique root even across simultaneous launches.
        tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        record_dir = Path(os.environ.get("RECORD_DIR", "data"))
        record_dir.mkdir(parents=True, exist_ok=True)
        root = record_dir / f"{name}_{tag}"
        repo_id = f"{os.environ.get('HF_REPO') or 'hud'}/{name}_{tag}"
        # LeRobotDataset.create requires a fresh root; images encode to per-episode video.
        ds = lerobot_dataset.LeRobotDataset.create(
            repo_id=repo_id,
            fps=self._fps,
            features=self._features,
            root=root,
            robot_type=self._contract.get("robot_type"),
            use_videos=True,
        )
        DatasetWriter._datasets[key] = (ds, root, repo_id)
        if not DatasetWriter._atexit_registered:
            atexit.register(DatasetWriter.finalize)
            DatasetWriter._atexit_registered = True
        print(f"[agent] recording LeRobot dataset -> {root}", flush=True)
        return ds
