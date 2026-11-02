"""smoke - HUD environment with a real coding task graded by its tests' expected values.

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

CASES = bugs.cases()

# Imports calc.py and prints what each case's call returns. Anything calc.py prints goes to
# stderr, so stdout carries only the results.
CALL_CASES = """
import contextlib, importlib.util, json, sys

with contextlib.redirect_stdout(sys.stderr):
    spec = importlib.util.spec_from_file_location("calc", sys.argv[1])
    calc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calc)
    returned = [getattr(calc, name)(*args) for name, args in json.loads(sys.argv[2])]
print(json.dumps(returned))
"""


def seed_bug(variant: int) -> list[str]:
    calc, tests, broken = bugs.build(variant)
    (WORKSPACE / "calc.py").write_text(calc)
    (WORKSPACE / "test_calc.py").write_text(tests)
    return broken


async def calc_passes() -> bool:
    """Whether the agent's calc.py returns the expected value for every test case.

    A separate process imports calc.py alone and reports what it returned; the comparison
    runs here. Edited tests, pytest config and anything calc.py prints can't change the
    grade, and printing results is no easier than returning them.
    """
    calls = json.dumps([[name, args] for name, args, _ in CASES])
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        CALL_CASES,
        str(WORKSPACE / "calc.py"),
        calls,
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
    return (
        isinstance(returned, list)
        and len(returned) == len(CASES)
        and all(
            isinstance(value, int | float)
            and math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-12)
            for value, (_, _, expected) in zip(returned, CASES, strict=True)
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
