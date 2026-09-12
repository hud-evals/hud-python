"""Small HUD environment used by the Fireworks Serverless RL cookbook."""

from __future__ import annotations

import re

from hud import Environment
from hud.graders import EvaluationResult

env = Environment(name="fireworks-arithmetic")


def grade_final_integer(answer: object, expected: int) -> EvaluationResult:
    """Grade an integer on the last nonempty line, allowing thousands separators."""
    text = (answer if isinstance(answer, str) else str(answer)).strip()
    final = text.splitlines()[-1].strip() if text else ""
    final = (
        re.sub(r"\\boxed\{\s*([^{}]+?)\s*\}", r"\1", final.rstrip(".")).strip("$*` \t").rstrip(".")
    )
    got = (
        int(final.replace(",", ""))
        if re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)", final)
        else None
    )
    return EvaluationResult(
        reward=1.0 if got == expected else 0.0,
        content=text,
        info={"expected": expected, "got": got},
    )


@env.template()
async def multiply(a: int, b: int):
    """Reward the correct final integer for a multiplication problem."""
    answer = yield (
        f"What is {a} * {b}? Work it out, then put the final integer on its own "
        "line at the end of your answer."
    )
    yield grade_final_integer(answer, a * b)
