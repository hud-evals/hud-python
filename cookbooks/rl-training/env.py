"""A tiny verifiable environment for the RL-training cookbook.

One template, ``multiply(a, b)``: ask for a product, let the model reason, and
grade the last integer in its reply. Serve with ``hud serve env.py`` or drive via
``LocalRuntime("env.py")``.
"""

from __future__ import annotations

import re

from hud import Environment
from hud.graders import EvaluationResult

env = Environment(name="arithmetic")


@env.template()
async def multiply(a: int, b: int):
    """Ask for ``a * b`` with reasoning; reward 1.0 iff the reply's last integer is the product."""
    answer = yield (
        f"What is {a} * {b}? Think it through, then end your reply with the final "
        "integer on its own and nothing after it."
    )

    text = answer or ""
    expected = a * b

    integers = re.findall(r"-?\d+", text)
    got = int(integers[-1]) if integers else None

    yield EvaluationResult(
        reward=1.0 if got == expected else 0.0,
        content=text.strip(),
        info={"expected": expected, "got": got},
    )
