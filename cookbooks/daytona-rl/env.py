"""smoke - HUD environment with a real coding task graded by its tests.

Spine of the HUD x Daytona guide: one shell capability, one buggy repo, one
grader. Runs identically under LocalRuntime and DaytonaRuntime.

Reward is binary — every test passes or the rollout scores zero.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path

from hud.environment import Environment

import bugs

env = Environment(name="smoke")

WORKSPACE = Path(tempfile.mkdtemp(prefix="calc-", dir="/tmp"))
ws = env.workspace(WORKSPACE, network=True)


def seed_bug(variant: int) -> list[str]:
    calc, tests, broken = bugs.build(variant)
    (WORKSPACE / "calc.py").write_text(calc)
    (WORKSPACE / "test_calc.py").write_text(tests)
    return broken


async def calc_passes() -> bool:
    """Whether the agent's calc.py returns the expected value for every test case.

    `bugs.py` imports calc.py in a separate process and prints what it returned; the
    comparison runs here.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        bugs.__file__,
        str(WORKSPACE / "calc.py"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        return False
    try:
        returned = json.loads(out)
    except json.JSONDecodeError:
        return False
    expected = [value for _, _, value in bugs.cases()]
    return (
        isinstance(returned, list)
        and len(returned) == len(expected)
        and all(
            isinstance(value, int | float)
            and math.isclose(value, want, rel_tol=1e-6, abs_tol=1e-12)
            for value, want in zip(returned, expected, strict=True)
        )
    )


@env.template(id="fix_calc")
async def fix_calc(variant: int = 0):
    seed_bug(variant)
    yield (
        f"`pytest` fails in `{WORKSPACE}`. Fix `calc.py` so it passes. Don't edit `test_calc.py`."
    )
    yield 1.0 if await calc_passes() else 0.0


async def test() -> None:
    from hud import LocalRuntime
    from hud.agents.claude import ClaudeAgent

    job = await fix_calc(variant=0).run(ClaudeAgent(), runtime=LocalRuntime(__file__))
    print("reward:", job.reward)


if __name__ == "__main__":
    asyncio.run(test())
