"""LIBERO direct control, plus a grasp line read from the finger joints.

The line is a label on ``robot0_gripper_qpos``, which the shared renderer
already prints. It does not read contacts or object pose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from hud.environment.robot import DirectControl
from mcp.types import ImageContent, TextContent
from numpy.typing import NDArray

if TYPE_CHECKING:
    from hud.environment.robot.control import _Gap

Content = list[TextContent | ImageContent]

# The two finger joints are a mirrored pair (one positive, one negative), so the opening is the
# larger absolute value. An empty close sits under 0.001, a holding pinch starts near 0.002, and
# a full open is about 0.04.
_MET = 0.001
_OPEN = 0.03
_FINGER = "robot0_gripper_qpos."
_GRIPPER = "gripper.open_close"


def grasp_sentence(opening: float, *, close_commanded: bool) -> str | None:
    """Label the finger opening. None between the cutoffs when no close was commanded."""
    if opening > _OPEN:
        return "open"
    if opening < _MET:
        return "fingers met; nothing between them"
    if close_commanded:
        return "fingers did not meet"
    return None


class LiberoDirectControl(DirectControl):
    """Shared ``move_to``, with one proprioceptive grasp line on the result text."""

    def _render(
        self,
        obs: dict[str, Any],
        played: int,
        proprio: NDArray[np.float64],
        *,
        rested: bool,
        gaps: list[_Gap] | None,
        goal: NDArray[np.float64] | None,
        ticks_left: int | None,
    ) -> Content:
        content = super()._render(obs, played, proprio, rested=rested, gaps=gaps, goal=goal, ticks_left=ticks_left)
        names = list(self._states["observation/state"])
        state = np.asarray(obs["data"]["observation/state"], dtype=np.float64).reshape(-1)
        fingers = [float(state[i]) for i, name in enumerate(names) if name.startswith(_FINGER)]
        if len(fingers) != 2:
            raise RuntimeError(f"expected two {_FINGER}* state names, got {names}")
        # A close is gripper.open_close > 0 (+1 closed, -1 open).
        close_commanded = self._command is not None and float(self._command[self._names.index(_GRIPPER)]) > 0.0
        sentence = grasp_sentence(max(abs(q) for q in fingers), close_commanded=close_commanded)
        if sentence is None:
            return content
        head = content[0]
        assert isinstance(head, TextContent)
        content[0] = TextContent(type="text", text=f"{head.text}\n{sentence}")
        return content
