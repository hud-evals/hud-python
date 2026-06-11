"""Tests for the LeRobot trace sink: contract -> schema, and record -> reload."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hud.telemetry.lerobot import LeRobotTraceSink, contract_to_lerobot_features
from hud.telemetry.recorder import EpisodeRecorder

CONTRACT: dict[str, Any] = {
    "robot_type": "test_bot",
    "control_rate": 10,
    "features": {
        "cam": {"role": "observation", "dtype": "image", "shape": [16, 16, 3]},
        "state": {"role": "observation", "dtype": "float32", "shape": [2]},
        "instruction": {"role": "observation", "dtype": "string"},
        "action": {"role": "action", "dtype": "float32", "shape": [2]},
    },
}


# ── contract -> LeRobot features (no lerobot import needed) ──────────────────


def test_image_obs_maps_to_observation_images() -> None:
    features, key_map = contract_to_lerobot_features(CONTRACT)
    assert "observation.images.cam" in features
    assert features["observation.images.cam"]["dtype"] == "video"  # use_videos default
    assert features["observation.images.cam"]["shape"] == (16, 16, 3)
    assert features["observation.images.cam"]["names"] == ["height", "width", "channel"]
    assert key_map["cam"] == "observation.images.cam"


def test_use_videos_false_keeps_image_dtype() -> None:
    features, _ = contract_to_lerobot_features(CONTRACT, use_videos=False)
    assert features["observation.images.cam"]["dtype"] == "image"


def test_single_vector_obs_maps_to_observation_state() -> None:
    features, key_map = contract_to_lerobot_features(CONTRACT)
    assert "observation.state" in features
    assert features["observation.state"]["dtype"] == "float32"
    assert features["observation.state"]["shape"] == (2,)
    assert key_map["state"] == "observation.state"


def test_multiple_vector_obs_keep_their_names() -> None:
    contract = {
        "features": {
            "joints": {"role": "observation", "dtype": "float32", "shape": [7]},
            "gripper": {"role": "observation", "dtype": "float32", "shape": [1]},
            "act": {"role": "action", "dtype": "float32", "shape": [7]},
        },
    }
    features, key_map = contract_to_lerobot_features(contract)
    assert "observation.joints" in features
    assert "observation.gripper" in features
    assert "observation.state" not in features
    assert key_map == {"joints": "observation.joints", "gripper": "observation.gripper"}


def test_vector_obs_literally_named_state_wins_observation_state() -> None:
    contract = {
        "features": {
            "state": {"role": "observation", "dtype": "float32", "shape": [4]},
            "extra": {"role": "observation", "dtype": "float32", "shape": [2]},
            "act": {"role": "action", "dtype": "float32", "shape": [4]},
        },
    }
    features, key_map = contract_to_lerobot_features(contract)
    assert key_map["state"] == "observation.state"
    assert key_map["extra"] == "observation.extra"
    assert "observation.state" in features and "observation.extra" in features


def test_string_obs_dropped_from_schema_and_key_map() -> None:
    features, key_map = contract_to_lerobot_features(CONTRACT)
    assert "instruction" not in key_map
    assert not any("instruction" in k for k in features)


def test_action_and_rl_columns() -> None:
    features, key_map = contract_to_lerobot_features(CONTRACT)
    assert features["action"] == {
        "dtype": "float32",
        "shape": (2,),
        "names": ["action_0", "action_1"],
    }
    assert "action" not in key_map  # action is not an observation wire key
    assert features["next.reward"] == {"dtype": "float32", "shape": (1,), "names": ["reward"]}
    assert features["next.done"] == {"dtype": "bool", "shape": (1,), "names": ["done"]}


def test_explicit_names_are_preserved() -> None:
    contract = {
        "features": {
            "state": {
                "role": "observation",
                "dtype": "float32",
                "shape": [2],
                "names": ["x", "y"],
            },
            "act": {"role": "action", "dtype": "float32", "shape": [1], "names": ["grip"]},
        },
    }
    features, _ = contract_to_lerobot_features(contract)
    assert features["observation.state"]["names"] == ["x", "y"]
    assert features["action"]["names"] == ["grip"]


# ── full record -> reload (requires lerobot) ──────────────────────────────────


def test_record_and_reload_lerobot_dataset(tmp_path) -> None:
    lerobot = pytest.importorskip("lerobot")  # noqa: F841 — skip cleanly without lerobot
    pytest.importorskip("datasets")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "ds"  # must not pre-exist (LeRobotDataset.create requirement)
    sink = LeRobotTraceSink(
        CONTRACT,
        root=root,
        repo_id="hud-tests/loopback",
        use_videos=False,  # plain image columns: no video-encoder dependency
        model_contract={"model": "stub"},
    )
    recorder = EpisodeRecorder(sink)

    rng = np.random.default_rng(0)
    n_frames = 3
    recorder.start_episode(prompt="pick up the cube")
    for i in range(n_frames):
        obs = {
            "cam": rng.integers(0, 255, size=(16, 16, 3)).astype(np.uint8),
            "state": np.array([i, -i], dtype=np.float32),
        }
        recorder.record_frame(
            obs,
            np.array([0.1 * i, 1.0], dtype=np.float32),
            reward=float(i),
            done=(i == n_frames - 1),
        )
    recorder.end_episode(success=True, total_reward=3.0)
    recorder.close()  # drains the worker + finalizes the dataset

    # Provenance: the env (and model) contract is stashed alongside the dataset.
    assert (root / "meta" / "hud_contract.json").exists()

    ds = LeRobotDataset("hud-tests/loopback", root=root)
    assert ds.num_episodes == 1
    assert ds.num_frames == n_frames
    assert ds.fps == 10
    assert ds.meta.robot_type == "test_bot"

    row = ds[1]
    np.testing.assert_allclose(np.asarray(row["observation.state"]), [1.0, -1.0])
    np.testing.assert_allclose(np.asarray(row["action"]), [0.1, 1.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(row["next.reward"]), [1.0])
    # LeRobot returns shape-(1,) columns as scalar tensors on read-back.
    assert not bool(np.asarray(row["next.done"]).reshape(-1)[0])
    assert bool(np.asarray(ds[2]["next.done"]).reshape(-1)[0])
    assert row["task"] == "pick up the cube"


def test_empty_episode_is_discarded(tmp_path) -> None:
    pytest.importorskip("lerobot")
    sink = LeRobotTraceSink(
        CONTRACT, root=tmp_path / "ds-empty", repo_id="hud-tests/empty", use_videos=False
    )
    recorder = EpisodeRecorder(sink)
    recorder.start_episode(prompt="nothing happens")
    recorder.end_episode(success=False)
    recorder.close()
    # No frames -> no episode saved.
    assert sink._ds is not None
    assert sink._ds.num_episodes == 0
