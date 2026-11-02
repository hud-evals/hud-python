"""Where a contract's cameras are, and which world point each pixel sees.

A camera is *located* when the contract gives its vertical field of view (``intrinsics.fovy``)
and its world pose: a world-fixed ``transform``, or a live one naming the observation that
carries its current pose. A camera feature may also name an aligned depth image (``"depth":
"observation/<camera>_depth"``, metres along the optical axis), which places each pixel exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hud.environment.robot.orientation import xyzw_to_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Camera:
    """Where a camera was when its frame was taken, and which world ray each pixel sees.

    Pixels are ``(u, v)``: column and row of the image array, from its top-left corner.
    """

    position: NDArray[np.float64]  # world frame, metres
    rotation: NDArray[np.float64]  # 3x3, its optical axes in the world: +z out of the lens,
    #                                +x image right, +y image down
    focal: float  # pixels
    width: int
    height: int
    depth: NDArray[np.float32] | None = None  # (H, W) metres along the optical axis, if rendered

    def _lens(self, u: float, v: float) -> NDArray[np.float64]:
        """Pixel (u, v) on the image plane one metre out, in the camera frame."""
        x = (u + 0.5 - self.width / 2) / self.focal
        y = (v + 0.5 - self.height / 2) / self.focal
        return np.array([x, y, 1.0])

    def ray(self, u: float, v: float) -> NDArray[np.float64]:
        """The world-frame unit direction the camera sees along at pixel (u, v)."""
        direction = self.rotation @ self._lens(u, v)
        return direction / np.linalg.norm(direction)

    def point(self, u: float, v: float) -> NDArray[np.float64]:
        """The world point (m) seen at pixel (u, v), from the depth image."""
        if self.depth is None:
            raise ValueError("this camera has no depth image; use to_world with a height")
        row = min(max(round(v), 0), self.height - 1)
        column = min(max(round(u), 0), self.width - 1)
        return self.position + self.rotation @ (self._lens(u, v) * float(self.depth[row, column]))

    def to_world(self, u: float, v: float, z: float = 0.0) -> NDArray[np.float64]:
        """Where the ray through pixel (u, v) meets the horizontal plane at height *z* (m)."""
        direction = self.ray(u, v)
        if abs(direction[2]) < 1e-9:
            raise ValueError("the ray through that pixel is parallel to the plane")
        reach = (z - self.position[2]) / direction[2]
        if reach <= 0:
            raise ValueError("that plane is behind the camera along that pixel's ray")
        return self.position + reach * direction

    def to_pixel(self, point: NDArray[np.float64] | list[float]) -> tuple[float, float]:
        """The pixel (u, v) a world point appears at (it may fall outside the image)."""
        x, y, depth = self.rotation.T @ (np.asarray(point, dtype=np.float64) - self.position)
        if depth <= 0:
            raise ValueError("that point is behind the camera")
        return (
            x / depth * self.focal + self.width / 2 - 0.5,
            y / depth * self.focal + self.height / 2 - 0.5,
        )


def located_names(
    features: dict[str, dict[str, Any]], cameras: list[str], live: dict[str, str]
) -> list[str]:
    """The *cameras* whose world pose and vertical field of view the contract gives: a
    world-fixed mount or a live pose (*live*: camera -> pose observation). A static mount on a
    moving body is not located."""
    return [
        name
        for name in cameras
        if "fovy" in features[name].get("intrinsics", {})
        and (name in live or features[name].get("transform", {}).get("parent") == "world")
    ]


def locate(
    features: dict[str, dict[str, Any]],
    cameras: list[str],
    live: dict[str, str],
    data: dict[str, Any],
) -> dict[str, Camera]:
    """Each located camera, by name, where it was when *data* was observed."""
    located: dict[str, Camera] = {}
    for name in located_names(features, cameras, live):
        feature = features[name]
        if name in live:
            pose = np.asarray(data[live[name]], dtype=np.float64).reshape(-1)
            position, xyzw = pose[:3], pose[3:7]
        else:
            transform = feature["transform"]
            position = np.asarray(transform["translation"], dtype=np.float64)
            xyzw = np.asarray(transform["rotation"], dtype=np.float64)
        height, width = np.asarray(data[name]).shape[-3:-1]  # the frame itself: shape is optional
        depth = feature.get("depth")
        located[name] = Camera(
            position=position,
            rotation=xyzw_to_matrix(xyzw),
            focal=height / 2 / math.tan(math.radians(feature["intrinsics"]["fovy"]) / 2),
            width=width,
            height=height,
            depth=None
            if depth is None
            else np.asarray(data[depth], dtype=np.float32).reshape(height, width),
        )
    return located


__all__ = ["Camera", "locate", "located_names"]
