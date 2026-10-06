"""End-effector orientation for direct control.

The motion tool speaks intrinsic XYZ euler (radians): roll, then pitch, then yaw.
That is scipy ``as_euler("xyz")``, the same convention CALVIN's pybullet state uses.
The contract may store a quaternion (xyzw or wxyz), an axis-angle, or that euler.
Interpolation is always a shortest-arc slerp. Quaternion components are never lerped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

Kind = Literal["quat", "axis_angle", "euler"]

_QUAT_SUFFIXES = frozenset({"qx", "qy", "qz", "qw"})


@dataclass(frozen=True)
class Orientation:
    """One rotation block in an absolute end-effector action."""

    kind: Kind
    #: Quat: contract columns of x, y, z, w. Axis-angle and euler: rx, ry, rz.
    columns: tuple[int, ...]
    #: Contract columns in name order, so a hold can copy them unchanged.
    contract_columns: tuple[int, ...]
    tool_names: tuple[str, str, str]

    def read(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Contract components → unit xyzw quaternion."""
        vals = [float(vector[i]) for i in self.columns]
        if self.kind == "quat":
            return _unit(np.asarray(vals, dtype=np.float64))
        if self.kind == "axis_angle":
            return rotvec_to_xyzw(np.asarray(vals, dtype=np.float64))
        return euler_to_xyzw(vals[0], vals[1], vals[2])

    def write(self, vector: NDArray[np.float64], xyzw: NDArray[np.float64]) -> None:
        """Unit xyzw quaternion → contract components."""
        if self.kind == "quat":
            packed = _unit(xyzw)
        elif self.kind == "axis_angle":
            packed = xyzw_to_rotvec(xyzw)
        else:
            packed = np.asarray(xyzw_to_euler(xyzw), dtype=np.float64)
        for column, value in zip(self.columns, packed, strict=True):
            vector[column] = value


def find_orientations(names: list[str]) -> list[Orientation]:
    """Contiguous quaternion or euler/axis-angle blocks in an action's names."""
    found: list[Orientation] = []
    index = 0
    while index < len(names):
        block = _block_at(names, index)
        if block is None:
            index += 1
            continue
        found.append(block)
        index = max(block.contract_columns) + 1
    taken = {name for block in found for name in block.tool_names}
    scalar = {
        name
        for i, name in enumerate(names)
        if all(i not in block.contract_columns for block in found)
    }
    overlap = taken & scalar
    if overlap:
        raise ValueError(f"orientation names {sorted(overlap)} collide with action dimensions")
    stems = [block.tool_names[0] for block in found]
    if len(stems) != len(set(stems)):
        raise ValueError(f"orientation blocks share a tool name: {stems}")
    return found


def euler_to_xyzw(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Intrinsic XYZ euler → unit xyzw quaternion."""
    return matrix_to_xyzw(_euler_matrix(roll, pitch, yaw))


def xyzw_to_euler(xyzw: NDArray[np.float64]) -> tuple[float, float, float]:
    """Unit xyzw quaternion → intrinsic XYZ euler. Pitch is in [-pi/2, pi/2]."""
    return _matrix_to_euler(xyzw_to_matrix(xyzw))


def rotvec_to_xyzw(rotvec: NDArray[np.float64]) -> NDArray[np.float64]:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        half = rotvec / 2.0
        return _unit(np.array([half[0], half[1], half[2], 1.0], dtype=np.float64))
    half = 0.5 * angle
    scale = math.sin(half) / angle
    return np.array(
        [rotvec[0] * scale, rotvec[1] * scale, rotvec[2] * scale, math.cos(half)],
        dtype=np.float64,
    )


def xyzw_to_rotvec(xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    quat = _unit(xyzw)
    if quat[3] < 0.0:
        quat = -quat
    angle = 2.0 * math.acos(float(np.clip(quat[3], -1.0, 1.0)))
    xyz = quat[:3]
    norm = float(np.linalg.norm(xyz))
    if norm < 1e-12 or angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(angle * xyz / norm, dtype=np.float64)


def slerp(
    start: NDArray[np.float64], end: NDArray[np.float64], fraction: float
) -> NDArray[np.float64]:
    """Shortest-arc slerp. ``fraction`` 0 is ``start``, 1 is ``end``."""
    q0 = _unit(start)
    q1 = _unit(end)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        mixed = q0 + fraction * (q1 - q0)
        return _unit(mixed)
    theta = math.acos(min(1.0, dot))
    scale = math.sin(theta)
    return (math.sin((1.0 - fraction) * theta) * q0 + math.sin(fraction * theta) * q1) / scale


def angular_distance(start: NDArray[np.float64], end: NDArray[np.float64]) -> float:
    """Angle in radians between two quaternions, in [0, pi]."""
    dot = abs(float(np.dot(_unit(start), _unit(end))))
    return 2.0 * math.acos(min(1.0, dot))


def matrix_to_xyzw(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ]
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ]
        )
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    return _unit(np.asarray(quat, dtype=np.float64))


def xyzw_to_matrix(xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z, w = (float(v) for v in _unit(xyzw))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _euler_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def _matrix_to_euler(rotation: NDArray[np.float64]) -> tuple[float, float, float]:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(float(rotation[2, 0])) > 0.999999:
        # Gimbal: one twist. +pi/2 is roll-yaw; -pi/2 is roll+yaw. Keep it in roll.
        if float(rotation[2, 0]) < 0.0:
            roll = math.atan2(float(rotation[0, 1]), float(rotation[1, 1]))
        else:
            roll = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
        return roll, pitch, 0.0
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return roll, pitch, yaw


def _unit(xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    quat = np.asarray(xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _suffix(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _stem(name: str) -> str:
    return name.rsplit(".", 1)[0]


def _tool_stem(name: str) -> str:
    stem = _stem(name)
    for token in ("_axis_angle", "_rotvec", "_euler", "_quat"):
        if stem.endswith(token):
            return stem[: -len(token)]
    return stem


def _rot_kind(name: str) -> Kind | None:
    stem = _stem(name)
    if "axis_angle" in stem or "rotvec" in stem:
        return "axis_angle"
    if "euler" in stem:
        return "euler"
    return None


def _block_at(names: list[str], index: int) -> Orientation | None:
    suffix = _suffix(names[index])
    if suffix in _QUAT_SUFFIXES and (index == 0 or _suffix(names[index - 1]) not in _QUAT_SUFFIXES):
        run = [index]
        cursor = index + 1
        while cursor < len(names) and _suffix(names[cursor]) in _QUAT_SUFFIXES:
            run.append(cursor)
            cursor += 1
        suffixes = [_suffix(names[i]) for i in run]
        if len(run) == 4 and set(suffixes) == set(_QUAT_SUFFIXES):
            order = tuple(run[suffixes.index(part)] for part in ("qx", "qy", "qz", "qw"))
            stem = _tool_stem(names[index])
            return Orientation(
                "quat",
                order,
                tuple(run),
                (f"{stem}.roll", f"{stem}.pitch", f"{stem}.yaw"),
            )
        return None
    kind = _rot_kind(names[index])
    if kind is None or index + 2 >= len(names):
        return None
    triple = names[index : index + 3]
    suffixes = tuple(_suffix(name) for name in triple)
    if suffixes not in {("rx", "ry", "rz"), ("x", "y", "z")}:
        return None
    if len({_stem(name) for name in triple}) != 1:
        return None
    if any(_rot_kind(name) != kind for name in triple):
        return None
    stem = _tool_stem(names[index])
    columns = (index, index + 1, index + 2)
    return Orientation(kind, columns, columns, (f"{stem}.roll", f"{stem}.pitch", f"{stem}.yaw"))


__all__ = [
    "Orientation",
    "angular_distance",
    "euler_to_xyzw",
    "find_orientations",
    "rotvec_to_xyzw",
    "slerp",
    "xyzw_to_euler",
    "xyzw_to_rotvec",
]
