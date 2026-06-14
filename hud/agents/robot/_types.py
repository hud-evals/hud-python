"""Shared robot-agent typing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

ActionArray = NDArray[np.floating[Any]]

__all__ = ["ActionArray"]
