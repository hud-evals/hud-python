"""Job: the platform/batch receipt for one taskset execution.

The live execution atom remains :class:`hud.client.Run`; a ``Job`` collects the
graded runs of one batch under one platform job id. Platform reporting lives in
:mod:`hud._platform`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hud.client import Run


@dataclass(slots=True)
class Job:
    """Platform/batch receipt for one taskset execution."""

    id: str
    name: str
    runs: list[Run]
    group: int = 1


__all__ = ["Job"]
