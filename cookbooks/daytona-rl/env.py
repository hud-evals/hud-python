"""smoke - HUD environment with a real coding task graded by pytest.

Spine of the HUD x Daytona guide: one shell capability, one buggy repo, one
pytest grader. Runs identically under LocalRuntime and DaytonaRuntime.

Reward is binary — every test passes or the rollout scores zero.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hud.environment import Environment
from hud.utils.process import create_process_group_exec

import bugs

env = Environment(name="smoke")

WORKSPACE = Path(tempfile.mkdtemp(prefix="calc-", dir="/tmp"))
ws = env.workspace(WORKSPACE, network=True)

TOTAL_TESTS = bugs.test_count()
GRADE_TIMEOUT_S = 120.0
MAX_FILE_BYTES = 1 << 20

# Imports the agent's calc.py and records what each requested call returned.
# It decides nothing: the assertions run in the grader, on these values.
_HARNESS = """
import importlib.util, json, numbers, sys
calls_path, calc_path, results_path = sys.argv[1:]
calls = json.loads(open(calls_path).read())
spec = importlib.util.spec_from_file_location("calc", calc_path)
calc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc)

def plain(value):
    # Fraction, Decimal, numpy scalars: compare as pytest would, not as strings.
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        return value
    return int(value) if isinstance(value, numbers.Integral) else float(value)

out = []
for name, args in calls:
    try:
        out.append(["ok", plain(getattr(calc, name)(*args))])
    except Exception as exc:
        out.append(["error", type(exc).__name__])
with open(results_path, "w") as f:
    f.write(json.dumps(out, default=repr))
"""

_REFERENCE: dict[str, Any] = {}
exec(bugs.HEADER + "".join(f["correct"] for f in bugs.FUNCS.values()), _REFERENCE)


def seed_bug(variant: int) -> list[str]:
    calc, tests, broken = bugs.build(variant)
    (WORKSPACE / "calc.py").write_text(calc)
    (WORKSPACE / "test_calc.py").write_text(tests)
    return broken


def _read_regular_file(path: Path) -> bytes:
    # Agent code controls this path: refuse a symlink (it would resolve against
    # the grader's filesystem), anything that isn't a plain file (a FIFO would
    # hang the read), and anything too large to be a calc.py or its results.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise OSError(f"{path.name} is not a regular file")
        data = f.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise OSError(f"{path.name} is larger than {MAX_FILE_BYTES} bytes")
    return data


def _run_seeded_tests(call: Callable[..., Any]) -> dict[str, bool]:
    """Run every seeded test, with each calc function served by ``call(name, *args)``.

    A call the grader didn't record fails its test, so a test whose arguments
    depend on earlier results fails closed rather than open.
    """
    namespace: dict[str, Any] = {
        name: lambda *args, name=name: call(name, *args) for name in bugs.NAMES
    }
    exec("import pytest\n" + "".join(f["tests"] for f in bugs.FUNCS.values()), namespace)
    outcomes: dict[str, bool] = {}
    for name, test in namespace.items():
        if name.startswith("test_"):
            try:
                test()
            except Exception:
                outcomes[name] = False
            else:
                outcomes[name] = True
    return outcomes


async def run_pytest() -> tuple[int, int]:
    """Grade the agent's ``calc.py`` against the seeded tests.

    Only ``calc.py`` leaves the workspace, so nothing else the agent wrote
    (edited tests, ``conftest.py``, pytest config) takes part. A separate
    process imports it and records the value of every call the tests make; the
    tests' assertions then run here, on those values. Nothing the agent's code
    prints, writes or exits with can mark a test as passed.
    """
    calls: list[list[Any]] = []

    def record(name: str, *args: Any) -> Any:
        calls.append([name, list(args)])
        return _REFERENCE[name](*args)

    reference = _run_seeded_tests(record)
    assert len(reference) == TOTAL_TESTS and all(reference.values()), reference

    # Cleanup errors are ignored: a process calc.py detached could still be
    # writing here, and that must not turn a grade into an exception.
    with tempfile.TemporaryDirectory(prefix="calc-grade-", ignore_cleanup_errors=True) as tmp:
        grade_dir = Path(tmp)
        try:
            (grade_dir / "calc.py").write_bytes(_read_regular_file(WORKSPACE / "calc.py"))
        except OSError:
            return 0, TOTAL_TESTS
        (grade_dir / "calls.json").write_text(json.dumps(calls))
        results = grade_dir / "results.json"
        # -I keeps the working directory and PYTHON* variables off the import path.
        proc = await create_process_group_exec(
            sys.executable,
            "-I",
            "-c",
            _HARNESS,
            str(grade_dir / "calls.json"),
            str(grade_dir / "calc.py"),
            str(results),
            cwd=grade_dir,
            env={},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.complete(max_wait=GRADE_TIMEOUT_S)
        try:
            values = json.loads(_read_regular_file(results))
        except (OSError, ValueError, RecursionError):
            return 0, TOTAL_TESTS

    # Only the values matter: a missing or misshapen entry fails its test in
    # replay, and nothing the harness reports can mark a test as passed.
    if not isinstance(values, list):
        return 0, TOTAL_TESTS
    served = dict(zip(map(json.dumps, calls), values, strict=False))

    def replay(name: str, *args: Any) -> Any:
        status, value = served[json.dumps([name, list(args)])]
        if status != "ok":
            raise RuntimeError(f"{name} raised {value}")
        return value

    outcomes = _run_seeded_tests(replay)
    return sum(outcomes.values()), TOTAL_TESTS


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
