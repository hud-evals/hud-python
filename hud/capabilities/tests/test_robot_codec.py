"""Tests for the ``robot`` wire codec and capability declaration."""

from __future__ import annotations

import numpy as np
import pytest

from hud.capabilities.base import Capability
from hud.capabilities.robot import (
    RobotClient,
    _decode_array,
    _encode_array,
    _packb,
    _unpackb,
)

# ── array round-trips ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((7,), dtype=np.float32),
        np.arange(12, dtype=np.float64).reshape(3, 4),
        np.random.default_rng(0).integers(0, 255, size=(16, 16, 3)).astype(np.uint8),
        np.array([[1, -2], [3, -4]], dtype=np.int64),
        np.array([True, False, True]),
        np.zeros((0, 5), dtype=np.float32),  # empty
    ],
    ids=["f32-1d", "f64-2d", "u8-image", "i64-2d", "bool-1d", "empty"],
)
def test_array_round_trip(arr: np.ndarray) -> None:
    decoded = _decode_array(_encode_array(arr))
    assert decoded.dtype == arr.dtype
    assert decoded.shape == arr.shape
    np.testing.assert_array_equal(decoded, arr)


def test_zero_d_array_is_promoted_to_1d() -> None:
    # Known codec quirk: np.ascontiguousarray promotes 0-d to shape (1,), so a
    # bare scalar does NOT round-trip shape-exactly (values are preserved).
    decoded = _decode_array(_encode_array(np.array(3.5, dtype=np.float32)))
    assert decoded.shape == (1,)
    assert decoded[0] == np.float32(3.5)


def test_encode_array_handles_non_contiguous_input() -> None:
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = base[:, ::2]  # non-contiguous view
    decoded = _decode_array(_encode_array(view))
    np.testing.assert_array_equal(decoded, view)


def test_decoded_array_is_writable_copy() -> None:
    arr = np.ones((3,), dtype=np.float32)
    decoded = _decode_array(_encode_array(arr))
    decoded[0] = 99.0  # frombuffer alone would be read-only; codec must copy
    assert decoded[0] == 99.0
    assert arr[0] == 1.0


def test_encode_array_wire_fields() -> None:
    enc = _encode_array(np.zeros((2, 3), dtype=np.uint8))
    assert enc["shape"] == [2, 3]
    assert enc["dtype"] == "uint8"
    assert isinstance(enc["data"], bytes)
    assert len(enc["data"]) == 6


# ── full-message round-trips (msgpack) ────────────────────────────────────────


def test_observation_message_round_trip() -> None:
    data = {
        "cam": np.random.default_rng(1).integers(0, 255, size=(8, 8, 3)).astype(np.uint8),
        "state": np.array([0.1, -0.2, 0.3], dtype=np.float32),
    }
    msg = {
        "terminated": False,
        "data": {name: _encode_array(arr) for name, arr in data.items()},
    }
    out = _unpackb(_packb(msg))
    assert out["terminated"] is False
    for name, arr in data.items():
        np.testing.assert_array_equal(_decode_array(out["data"][name]), arr)


def test_chunk_message_round_trip() -> None:
    chunk = np.random.default_rng(2).normal(size=(50, 7)).astype(np.float32)
    msg = {"chunk": _encode_array(chunk), "obs_index": 123, "delay_used": 4}
    out = _unpackb(_packb(msg))
    assert out["obs_index"] == 123
    assert out["delay_used"] == 4
    np.testing.assert_array_equal(_decode_array(out["chunk"]), chunk)


def test_meta_message_round_trip_with_none_chunk() -> None:
    msg = {
        "terminated": True,
        "data": {},
        "meta": {"obs_index": 7, "queue_remaining": 0, "delay": 2, "unexecuted_chunk": None},
    }
    out = _unpackb(_packb(msg))
    assert out["meta"]["unexecuted_chunk"] is None
    assert out["meta"]["obs_index"] == 7
    assert out["terminated"] is True


# ── capability declaration ────────────────────────────────────────────────────

CONTRACT = {
    "robot_type": "test_bot",
    "control_rate": 10,
    "features": {
        "cam": {"role": "observation", "dtype": "image", "shape": [8, 8, 3]},
        "state": {"role": "observation", "dtype": "float32", "shape": [3]},
        "action": {"role": "action", "dtype": "float32", "shape": [7]},
    },
}


def test_capability_robot_protocol_and_contract() -> None:
    cap = Capability.robot(url="ws://localhost:9091", contract=CONTRACT)
    assert cap.protocol == "robot/0.1"
    assert cap.name == "robot"
    assert cap.url == "ws://localhost:9091"
    assert cap.params["contract"] == CONTRACT


def test_capability_robot_round_trips_through_manifest() -> None:
    cap = Capability.robot(url="ws://localhost:9091", contract=CONTRACT)
    restored = Capability.from_manifest(cap.to_manifest())
    assert restored.protocol == "robot/0.1"
    assert restored.params["contract"] == CONTRACT


def test_capability_robot_normalizes_bare_host() -> None:
    cap = Capability.robot(url="somehost", contract={})
    assert cap.url == "ws://somehost:9091"


def test_robot_client_protocol_string() -> None:
    assert RobotClient.protocol == "robot/0.1"
