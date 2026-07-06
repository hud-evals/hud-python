"""Opt-in LeRobot v3 dataset writing for robot rollouts.

One :class:`DatasetWriter` spans an agent's lifetime: every episode it drives
appends ``(observation, executed action)`` frames, and the dataset is finalized
at process exit (or an explicit :meth:`finalize`), optionally pushed to the HF
Hub. The contract drives the schema with no extra wiring. Destination + push
come from the environment:

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

from hud.telemetry.context import get_current_trace_id

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
    """Appends robot frames to a LeRobot v3 dataset; a no-op shell when lerobot
    is missing (warned once) so telemetry-only runs never break."""

    def __init__(self, contract: dict[str, Any], *, fps: int) -> None:
        self._contract = contract
        self._fps = fps
        self._features, self._key_map = _lerobot_features(contract)
        self._ds: Any | None = None  # created lazily on the first frame
        self._root: Path | None = None
        self._repo_id = ""
        self._enabled = importlib.util.find_spec("lerobot") is not None
        if not self._enabled:
            logger.warning(
                "save=True but lerobot is not installed; streaming telemetry only "
                "(pip install 'lerobot[dataset]')"
            )
        else:
            atexit.register(self.finalize)  # keep the dataset loadable on abrupt exits

    def add(self, data: dict[str, Any], action: NDArray[Any], *, task: str) -> None:
        """One frame: the wire observation dict + the executed env-space action."""
        if not self._enabled:
            return
        if self._ds is None:  # derived contracts carry no shapes; fill from the first frame
            for wire, key in self._key_map.items():
                if not self._features[key]["shape"] and wire in data:
                    self._features[key]["shape"] = tuple(np.shape(data[wire]))
            if not self._features["action"]["shape"]:
                self._features["action"]["shape"] = tuple(np.shape(action))
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
        row["task"] = task
        ds.add_frame(row)

    def end_episode(self) -> None:
        """Commit the episode's pending frames."""
        if self._ds is not None and self._ds.has_pending_frames():
            self._ds.save_episode()

    def finalize(self) -> None:
        """Write the parquet footer + optionally push to the Hub. Idempotent."""
        if self._ds is None:
            return
        ds, self._ds = self._ds, None
        ds.finalize()
        print(f"[agent] saved LeRobot dataset -> {self._root}", flush=True)
        if not os.environ.get("HF_REPO"):
            return
        private = os.environ.get("HF_PRIVATE", "0") not in ("0", "", "false", "False")
        try:  # best-effort: the on-disk dataset is the source of truth
            ds.push_to_hub(private=private)
            print(f"[agent] pushed -> https://huggingface.co/datasets/{self._repo_id}", flush=True)
        except Exception as exc:
            logger.exception("HF push failed for %s", self._repo_id)
            print(f"[agent] WARNING: HF push failed: {exc!r} (dataset still on disk)", flush=True)

    def _ensure_dataset(self) -> Any:
        if self._ds is not None:
            return self._ds
        lerobot_dataset: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")

        name = self._contract.get("robot_type") or "robot"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        # Unique per writer so concurrent rollouts never share a root; tied to the
        # trace id when there is one so a shard maps back to its trace.
        tag = (get_current_trace_id() or uuid.uuid4().hex)[:8]
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


__all__ = ["DatasetWriter"]
