"""A tiny verifiable environment for the RL-training cookbook.

One template, ``multiply(a, b)``: ask the model for a product and grade it
**strictly** — the whole reply must be exactly the integer, nothing else. Two-digit
multiplication is something a small base model gets only sometimes (and often
wraps in prose), so the baseline reward is well below 1.0 with within-group
variance — the signal GRPO needs. Serve with ``hud serve env.py`` or drive via
``LocalRuntime("env.py")``.
"""

from __future__ import annotations

import re

from hud.environment import Environment
from hud.graders import EvaluationResult

env = Environment(name="arithmetic")


@env.template()
async def multiply(a: int, b: int):
    """Ask for ``a * b`` as a *direct* answer; reward 1.0 iff reply == product.

    The prompt forbids reasoning and the caller caps output tokens, so the model
    must answer from "mental math" rather than scratch work — something a small
    model is unreliable at, giving a sub-1.0 baseline with within-group variance.
    """
    answer = yield (
        f"What is {a} * {b}? Think it through, then end your reply with the final "
        "integer on its own and nothing after it."
    )

    text = answer if isinstance(answer, str) else str(answer)
    expected = a * b

    # The model reasons, then states the product last; grade the final integer.
    integers = re.findall(r"-?\d+", text)
    got = int(integers[-1]) if integers else None

    yield EvaluationResult(
        reward=1.0 if got == expected else 0.0,
        content=text.strip(),
        info={"expected": expected, "got": got},
    )
