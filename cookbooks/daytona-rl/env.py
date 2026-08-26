"""smoke - HUD environment with a real coding task graded by pytest.

Spine of the HUD x Daytona guide: one shell capability, one buggy repo, one
pytest grader. Runs identically under LocalRuntime and DaytonaRuntime.

Reward is binary — every test passes or the rollout scores zero.
"""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from pathlib import Path

from hud.environment import Environment

import bugs

env = Environment(name="smoke")

WORKSPACE = Path(tempfile.mkdtemp(prefix="calc-", dir="/tmp"))
ws = env.workspace(WORKSPACE, network=True)

TOTAL_TESTS = bugs.test_count()


def seed_bug(variant: int) -> list[str]:
    calc, tests, broken = bugs.build(variant)
    (WORKSPACE / "calc.py").write_text(calc)
    (WORKSPACE / "test_calc.py").write_text(tests)
    return broken


async def run_pytest() -> tuple[int, int]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        cwd=str(WORKSPACE),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    match = re.search(r"(\d+) passed", out.decode("utf-8", "replace"))
    return (int(match.group(1)) if match else 0), TOTAL_TESTS


@env.template(id="fix_calc")
async def fix_calc(variant: int = 0):
    seed_bug(variant)
    yield (
        f"`pytest` fails in `{WORKSPACE}`. Fix `calc.py` so it passes. Don't edit `test_calc.py`."
    )
    passed, total = await run_pytest()
    yield 1.0 if passed == total else 0.0


async def test() -> None:
    from hud import LocalRuntime
    from hud.agents.claude import ClaudeAgent

    job = await fix_calc(variant=0).run(ClaudeAgent(), runtime=LocalRuntime(__file__))
    print("reward:", job.reward)


if __name__ == "__main__":
    asyncio.run(test())
