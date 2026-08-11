"""Small HUD environment used by the Fireworks Serverless RL cookbook."""

from __future__ import annotations

import re

from hud import Environment
from hud.graders import EvaluationResult

env = Environment(name="fireworks-arithmetic")


@env.template()
async def multiply(a: int, b: int):
    """Reward the correct final integer for a multiplication problem."""
    answer = yield (
        f"What is {a} * {b}? Work it out, then put the final integer on its own "
        "line at the end of your answer."
    )
    text = answer if isinstance(answer, str) else str(answer)
    integers = re.findall(r"-?\d+", text)
    got = int(integers[-1]) if integers else None
    expected = a * b
    yield EvaluationResult(
        reward=1.0 if got == expected else 0.0,
        content=text.strip(),
        info={"expected": expected, "got": got},
    )
