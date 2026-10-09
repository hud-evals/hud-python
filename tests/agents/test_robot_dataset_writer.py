"""DatasetWriter shares by compatible schema/FPS, not a single process-global."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from hud.agents.robot import dataset as dataset_mod
from hud.agents.robot.dataset import DatasetWriter


def _contract(
    *, state_shape: tuple[int, ...], action_shape: tuple[int, ...], robot: str
) -> dict[str, Any]:
    return {
        "robot_type": robot,
        "features": {
            "state": {"role": "observation", "dtype": "float32", "shape": list(state_shape)},
            "action": {"role": "action", "dtype": "float32", "shape": list(action_shape)},
        },
    }


@pytest.fixture(autouse=True)
def reset_dataset_writer_globals(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    DatasetWriter._datasets.clear()
    DatasetWriter._open.clear()
    DatasetWriter._atexit_registered = False
    monkeypatch.setenv("RECORD_DIR", str(tmp_path))
    monkeypatch.delenv("HF_REPO", raising=False)


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> list[MagicMock]:
    created: list[MagicMock] = []

    def create(**kwargs: Any) -> MagicMock:
        ds = MagicMock(name=f"ds-{len(created)}")
        ds.create_kwargs = kwargs
        created.append(ds)
        return ds

    monkeypatch.setitem(
        __import__("sys").modules,
        "lerobot.datasets.lerobot_dataset",
        SimpleNamespace(LeRobotDataset=SimpleNamespace(create=create)),
    )

    def _found(_name: str) -> object:
        return object()

    monkeypatch.setattr(dataset_mod.importlib.util, "find_spec", _found)
    return created


def test_matching_writers_share_one_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_lerobot(monkeypatch)
    contract = _contract(state_shape=(3,), action_shape=(2,), robot="arm")
    obs, act = {"state": np.zeros(3, dtype=np.float32)}, np.zeros(2, dtype=np.float32)
    DatasetWriter(contract, fps=10).add(obs, act, task="t")
    DatasetWriter(contract, fps=10).add(obs, act, task="t")
    assert len(created) == 1
    assert len(DatasetWriter._datasets) == 1


def test_different_fps_or_features_get_separate_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_lerobot(monkeypatch)
    c1 = _contract(state_shape=(3,), action_shape=(2,), robot="arm")
    c2 = _contract(state_shape=(4,), action_shape=(2,), robot="arm")
    act = np.zeros(2, dtype=np.float32)
    DatasetWriter(c1, fps=10).add({"state": np.zeros(3, dtype=np.float32)}, act, task="t")
    DatasetWriter(c1, fps=20).add({"state": np.zeros(3, dtype=np.float32)}, act, task="t")
    DatasetWriter(c2, fps=10).add({"state": np.zeros(4, dtype=np.float32)}, act, task="t")
    assert len(created) == 3
    assert created[0].create_kwargs["fps"] == 10
    assert created[1].create_kwargs["fps"] == 20
    assert tuple(created[2].create_kwargs["features"]["observation.state"]["shape"]) == (4,)


def test_different_feature_names_get_separate_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_lerobot(monkeypatch)
    first = _contract(state_shape=(2,), action_shape=(1,), robot="arm")
    second = _contract(state_shape=(2,), action_shape=(1,), robot="arm")
    first["features"]["state"]["names"] = ["shoulder", "elbow"]
    second["features"]["state"]["names"] = ["x", "y"]
    obs = {"state": np.zeros(2, dtype=np.float32)}
    action = np.zeros(1, dtype=np.float32)

    DatasetWriter(first, fps=10).add(obs, action, task="t")
    DatasetWriter(second, fps=10).add(obs, action, task="t")

    assert len(created) == 2
    assert created[0].create_kwargs["features"]["observation.state"]["names"] == [
        "shoulder",
        "elbow",
    ]
    assert created[1].create_kwargs["features"]["observation.state"]["names"] == ["x", "y"]


def test_finalize_clears_all_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_lerobot(monkeypatch)
    act = np.zeros(2, dtype=np.float32)
    a = DatasetWriter(_contract(state_shape=(3,), action_shape=(2,), robot="a"), fps=10)
    b = DatasetWriter(_contract(state_shape=(5,), action_shape=(2,), robot="b"), fps=10)
    a.add({"state": np.zeros(3, dtype=np.float32)}, act, task="t")
    b.add({"state": np.zeros(5, dtype=np.float32)}, act, task="t")
    a.end_episode()
    b.end_episode()
    DatasetWriter.finalize()
    assert DatasetWriter._datasets == {}
    for ds in created:
        ds.finalize.assert_called_once()


def test_nested_image_keys_stay_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_lerobot(monkeypatch)
    contract = {
        "robot_type": "stereo",
        "features": {
            "left/image": {"role": "observation", "dtype": "image", "shape": [2, 2, 3]},
            "right/image": {"role": "observation", "dtype": "image", "shape": [2, 2, 3]},
            "state": {"role": "observation", "dtype": "float32", "shape": [1]},
            "action": {"role": "action", "dtype": "float32", "shape": [1]},
        },
    }
    writer = DatasetWriter(contract, fps=10)
    left = np.zeros((2, 2, 3), dtype=np.uint8)
    right = np.full((2, 2, 3), 255, dtype=np.uint8)
    writer.add(
        {"left/image": left, "right/image": right, "state": np.zeros(1, dtype=np.float32)},
        np.zeros(1, dtype=np.float32),
        task="t",
    )
    writer.end_episode()

    features = created[0].create_kwargs["features"]
    assert "observation.images.left_image" in features
    assert "observation.images.right_image" in features
    frame = created[0].add_frame.call_args.args[0]
    assert frame["observation.images.left_image"] is left
    assert frame["observation.images.right_image"] is right


def test_duplicate_mapped_keys_raise() -> None:
    contract = {
        "robot_type": "bad",
        "features": {
            # Both wire names become the same LeRobot key after path flattening.
            "cam/rgb": {"role": "observation", "dtype": "image", "shape": [2, 2, 3]},
            "cam_rgb": {"role": "observation", "dtype": "image", "shape": [2, 2, 3]},
            "action": {"role": "action", "dtype": "float32", "shape": [1]},
        },
    }
    with pytest.raises(ValueError, match="both map to LeRobot key"):
        DatasetWriter(contract, fps=10)
