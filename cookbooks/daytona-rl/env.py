"""smoke - HUD environment with a real coding task graded by pytest.

Spine of the HUD x Daytona guide: one shell capability, one buggy repo, one
pytest grader. Runs identically under LocalRuntime and DaytonaRuntime.

Reward is binary — every test passes or the rollout scores zero.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

from hud.environment import Environment

import bugs

env = Environment(name="smoke")

WORKSPACE = Path(tempfile.mkdtemp(prefix="calc-", dir="/tmp"))
ws = env.workspace(WORKSPACE, network=True)

TOTAL_TESTS = bugs.test_count()

# Files in the workspace that pytest would load as configuration or plugins.
PYTEST_CONFIG_FILES = ("conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")


def seed_bug(variant: int) -> list[str]:
    calc, tests, broken = bugs.build(variant)
    (WORKSPACE / "calc.py").write_text(calc)
    (WORKSPACE / "test_calc.py").write_text(tests)
    return broken


def passed_tests(report: Path) -> set[str]:
    """Names of the test cases a JUnit report records as passed."""
    passed: set[str] = set()
    for case in ElementTree.parse(report).getroot().iter("testcase"):
        if not any(child.tag in {"failure", "error", "skipped"} for child in case):
            passed.add(case.attrib["name"])
    return passed


async def run_pytest(variant: int = 0) -> tuple[int, int]:
    """Run the variant's own tests against the agent's `calc.py`.

    The agent works in the same directory as the tests, so grading restores
    `test_calc.py`, drops any pytest configuration the agent added, and reads the
    outcome from a JUnit report written outside the workspace, never from stdout.
    """
    _, tests, _ = bugs.build(variant)
    (WORKSPACE / "test_calc.py").write_text(tests)
    for name in PYTEST_CONFIG_FILES:
        (WORKSPACE / name).unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="calc-grade-") as directory:
        config = Path(directory) / "pytest.ini"
        config.write_text("[pytest]\n")
        report = Path(directory) / "report.xml"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "-c",
            str(config),
            "--rootdir",
            str(WORKSPACE),
            "--confcutdir",
            str(WORKSPACE),
            f"--junitxml={report}",
            "test_calc.py",
            cwd=str(WORKSPACE),
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        # A run that exits before writing its report, or with failures, passes nothing.
        if proc.returncode != 0 or not report.exists():
            return 0, TOTAL_TESTS
        try:
            passed = passed_tests(report)
        except (ElementTree.ParseError, KeyError):
            return 0, TOTAL_TESTS
    return len(passed & set(bugs.test_names())), TOTAL_TESTS


@env.template(id="fix_calc")
async def fix_calc(variant: int = 0):
    seed_bug(variant)
    yield (
        f"`pytest` fails in `{WORKSPACE}`. Fix `calc.py` so it passes. Don't edit `test_calc.py`."
    )
    passed, total = await run_pytest(variant)
    yield 1.0 if passed == total else 0.0


async def test() -> None:
    from hud import LocalRuntime
    from hud.agents.claude import ClaudeAgent

    job = await fix_calc(variant=0).run(ClaudeAgent(), runtime=LocalRuntime(__file__))
    print("reward:", job.reward)


if __name__ == "__main__":
    asyncio.run(test())
