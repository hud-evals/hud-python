"""``LeRobotDataset`` as the recording code uses it, written to ``<root>/dataset.json``.

Frames are stored as lists; image arrays as ``"<image HxWxC>"``. Setting
``FAKE_LEROBOT_PUSH_FAILS`` makes ``push_to_hub`` raise.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def _value(value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype == np.uint8 and array.ndim == 3:
        return "<image {}>".format("x".join(str(size) for size in array.shape))
    return array.tolist() if isinstance(value, np.ndarray) else value


class LeRobotDataset:
    def __init__(self, root: Path, record: dict[str, Any]) -> None:
        self.root = root
        self.record = record
        self.pending: list[dict[str, Any]] = []
        self._write()

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        fps: int,
        features: dict[str, Any],
        root: Path,
        robot_type: str | None,
        use_videos: bool,
    ) -> LeRobotDataset:
        root.mkdir(parents=True)
        record = {
            "repo_id": repo_id,
            "fps": fps,
            "robot_type": robot_type,
            "use_videos": use_videos,
            "features": {
                key: {**feature, "shape": list(feature["shape"])}
                for key, feature in features.items()
            },
            "episodes": [],
            "finalized": False,
            "pushed": None,
        }
        return cls(root, record)

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.pending.append({key: _value(value) for key, value in frame.items()})

    def save_episode(self) -> None:
        self.record["episodes"].append(self.pending)
        self.pending = []
        self._write()

    def finalize(self) -> None:
        self.record["finalized"] = True
        self._write()

    def push_to_hub(self, *, private: bool) -> None:
        if os.environ.get("FAKE_LEROBOT_PUSH_FAILS"):
            raise ConnectionError("hub unreachable")
        self.record["pushed"] = {"private": private}
        self._write()

    def _write(self) -> None:
        (self.root / "dataset.json").write_text(json.dumps(self.record))
