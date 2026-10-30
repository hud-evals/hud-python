"""The fix_calc grader pays only for a calc.py that passes the seeded tests (#736).

Each test seeds a variant into a fresh workspace, edits it the way an agent
could, and grades it through ``run_pytest`` exactly as the template does.
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path

import bugs
import env
import pytest

# Like the cookbook itself (bwrap workspace, Linux image), the grader is POSIX-only.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the daytona-rl cookbook is POSIX-only"
)

# 4, 5 and 6 broken functions, plus a held-out variant.
VARIANTS = [0, 1, 2, 16]
TOTAL = bugs.test_count()
FAKE_PASS = f'print("{TOTAL} passed in 0.01s")\n'


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(env, "WORKSPACE", root)
    return root


def fix(workspace: Path, broken: list[str], *, leave: str | None = None) -> None:
    calc = workspace / "calc.py"
    source = calc.read_text()
    for name in broken:
        if name != leave:
            funcs = bugs.FUNCS[name]
            source = source.replace(funcs["buggy"].strip("\n"), funcs["correct"].strip("\n"))
    calc.write_text(source)


def prepend(path: Path, code: str) -> None:
    path.write_text(code + path.read_text())


@pytest.mark.parametrize("variant", VARIANTS)
async def test_correct_fix_earns_full_credit(workspace: Path, variant: int):
    fix(workspace, env.seed_bug(variant))

    assert await env.run_pytest() == (TOTAL, TOTAL)


@pytest.mark.parametrize("variant", VARIANTS)
async def test_unfixed_calc_falls_short(variant: int):
    env.seed_bug(variant)

    passed, total = await env.run_pytest()

    assert passed < total == TOTAL


async def test_partial_fix_falls_short(workspace: Path):
    fix(workspace, env.seed_bug(0), leave="mean")

    assert await env.run_pytest() == (TOTAL - 1, TOTAL)


@pytest.mark.parametrize("edit", ["rewrite", "append", "delete"])
async def test_editing_test_calc_cannot_earn_credit(workspace: Path, edit: str):
    env.seed_bug(0)
    passed, _ = await env.run_pytest()
    tests = workspace / "test_calc.py"
    if edit == "rewrite":
        tests.write_text("".join(f"def test_{i}():\n    pass\n" for i in range(TOTAL)))
    elif edit == "append":
        # Exactly enough passing tests to make the old "N passed" count hit TOTAL.
        extra = "".join(f"\ndef test_extra_{i}():\n    pass\n" for i in range(TOTAL - passed))
        tests.write_text(tests.read_text() + extra)
    else:
        tests.unlink()

    assert await env.run_pytest() == (passed, TOTAL)


async def test_printed_pytest_summary_cannot_earn_credit(workspace: Path):
    env.seed_bug(0)
    passed, _ = await env.run_pytest()
    # Printed inside every function, so it is replayed in the failure reports
    # that pytest emits before its real "N failed, M passed" summary line.
    calc = workspace / "calc.py"
    lines = calc.read_text().splitlines(keepends=True)
    calc.write_text(
        "".join(line + "    " + FAKE_PASS if line.startswith("def ") else line for line in lines)
    )

    assert await env.run_pytest() == (passed, TOTAL)


@pytest.mark.parametrize(
    "name, content",
    [
        pytest.param(
            "conftest.py",
            "import pytest\n\n"
            "@pytest.hookimpl(hookwrapper=True)\n"
            "def pytest_runtest_makereport(item, call):\n"
            "    report = (yield).get_result()\n"
            "    report.outcome, report.longrepr = 'passed', None\n",
            id="conftest-rewrites-outcomes",
        ),
        pytest.param("conftest.py", 'collect_ignore = ["test_calc.py"]\n', id="conftest-ignore"),
        pytest.param("pytest.ini", "[pytest]\naddopts = -k nothing\n", id="pytest.ini"),
        pytest.param(
            "pyproject.toml",
            '[tool.pytest.ini_options]\naddopts = "-k nothing"\n',
            id="pyproject.toml",
        ),
        pytest.param("setup.cfg", "[tool:pytest]\naddopts = -k nothing\n", id="setup.cfg"),
        pytest.param("tox.ini", "[pytest]\naddopts = -k nothing\n", id="tox.ini"),
        pytest.param("pytest.py", "import sys\n" + FAKE_PASS + "sys.exit(0)\n", id="shadow-pytest"),
    ],
)
async def test_workspace_files_do_not_take_part_in_grading(
    workspace: Path, name: str, content: str
):
    env.seed_bug(0)
    passed, _ = await env.run_pytest()
    (workspace / name).write_text(content)

    assert await env.run_pytest() == (passed, TOTAL)


async def test_workspace_files_cannot_sabotage_a_correct_fix(workspace: Path):
    fix(workspace, env.seed_bug(0))
    (workspace / "conftest.py").write_text("raise RuntimeError('sabotage')\n")
    (workspace / "pytest.ini").write_text("[pytest]\naddopts = --strict-markers -x -k nothing\n")

    assert await env.run_pytest() == (TOTAL, TOTAL)


@pytest.mark.parametrize(
    "abort",
    [
        pytest.param("import pytest\npytest.skip('nope', allow_module_level=True)\n", id="skip"),
        pytest.param("import os\nos._exit(0)\n", id="os._exit(0)"),
        pytest.param("import sys\nsys.exit(0)\n", id="sys.exit(0)"),
    ],
)
async def test_calc_that_exits_before_reporting_earns_nothing(workspace: Path, abort: str):
    fix(workspace, env.seed_bug(0))
    prepend(workspace / "calc.py", abort)

    assert await env.run_pytest() == (0, TOTAL)


async def test_debug_prints_in_a_correct_fix_keep_full_credit(workspace: Path):
    fix(workspace, env.seed_bug(0))
    calc = workspace / "calc.py"
    prepend(calc, 'print("[[\\"ok\\", 0]]")\n')
    calc.write_text(calc.read_text().replace("def mean(xs):\n", "def mean(xs):\n    print(xs)\n"))

    assert await env.run_pytest() == (TOTAL, TOTAL)


@pytest.mark.parametrize("kind", ["missing", "symlink", "fifo", "directory"])
async def test_unreadable_calc_earns_nothing(workspace: Path, tmp_path: Path, kind: str):
    fix(workspace, env.seed_bug(0))
    calc = workspace / "calc.py"
    if kind == "symlink":
        outside = tmp_path / "outside.py"
        calc.rename(outside)
        calc.symlink_to(outside)
    else:
        calc.unlink()
        if kind == "fifo":
            os.mkfifo(calc)
        elif kind == "directory":
            calc.mkdir()

    assert await env.run_pytest() == (0, TOTAL)


async def test_hanging_calc_times_out_with_no_credit(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(env, "GRADE_TIMEOUT_S", 2.0)
    fix(workspace, env.seed_bug(0))
    prepend(workspace / "calc.py", "while True:\n    pass\n")

    assert await env.run_pytest() == (0, TOTAL)


@pytest.mark.skipif(sys.platform != "linux", reason="reads process state from /proc")
async def test_processes_left_behind_by_calc_are_killed(workspace: Path, tmp_path: Path):
    fix(workspace, env.seed_bug(0))
    pid_file = tmp_path / "sleeper.pid"
    prepend(
        workspace / "calc.py",
        "import subprocess\n"
        "_p = subprocess.Popen(['sleep', '600'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(_p.pid))\n",
    )

    assert await env.run_pytest() == (TOTAL, TOTAL)

    stat = Path(f"/proc/{pid_file.read_text()}/stat")
    assert not stat.exists() or stat.read_text().split(") ")[1].startswith("Z")


# Each payload replaces the results file after the harness wrote the real ones.
@pytest.mark.parametrize(
    "replace",
    [
        pytest.param("open(path, 'w').write('[' * 100_000)", id="deeply-nested"),
        pytest.param("open(path, 'w').write('5')", id="scalar"),
        pytest.param("open(path, 'w').write('[[\"ok\"]]')", id="short-and-misshapen"),
        pytest.param("open(path, 'w').write('[[\"ok\", 2.5')", id="truncated"),
        pytest.param("open(path, 'wb').truncate(8 << 30)", id="oversized-sparse"),
        pytest.param("os.mkfifo(path)", id="fifo"),
        pytest.param("os.symlink('/dev/zero', path)", id="symlink"),
    ],
)
async def test_tampered_results_earn_nothing(workspace: Path, replace: str):
    fix(workspace, env.seed_bug(0))
    prepend(
        workspace / "calc.py",
        "import atexit, os, sys\n"
        "def _replace():\n"
        "    path = sys.argv[3]\n"
        "    os.unlink(path)\n"
        f"    {replace}\n"
        "    os._exit(0)\n"
        "atexit.register(_replace)\n",
    )

    assert await env.run_pytest() == (0, TOTAL)


async def test_oversized_calc_earns_nothing(workspace: Path):
    fix(workspace, env.seed_bug(0))
    with (workspace / "calc.py").open("ab") as f:
        f.truncate(8 << 30)

    assert await env.run_pytest() == (0, TOTAL)


async def test_numeric_types_pytest_accepts_keep_full_credit(workspace: Path):
    fix(workspace, env.seed_bug(0))
    calc = workspace / "calc.py"
    calc.write_text(
        "from decimal import Decimal\nfrom fractions import Fraction\n"
        + calc.read_text()
        .replace("return sum(xs) / len(xs)\n", "return Fraction(sum(xs), len(xs))\n", 1)
        .replace("return max(xs) - min(xs)\n", "return Decimal(max(xs) - min(xs))\n", 1)
    )
    assert "Fraction(sum" in calc.read_text() and "Decimal(max" in calc.read_text()

    assert await env.run_pytest() == (TOTAL, TOTAL)


@pytest.mark.skipif(sys.platform != "linux", reason="reads process state from /proc")
async def test_detached_process_writing_to_grade_dir_does_not_break_grading(
    workspace: Path, tmp_path: Path
):
    # calc.py detaches a process that keeps creating files in the grading
    # directory while it is being removed.
    fix(workspace, env.seed_bug(0))
    record = tmp_path / "detached"
    prepend(
        workspace / "calc.py",
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    if os.fork() == 0:\n"
        f"        open({str(record)!r}, 'w').write(f'{{os.getpid()}} {{os.getcwd()}}')\n"
        "        end = time.monotonic() + 3\n"
        "        i = 0\n"
        "        while time.monotonic() < end:\n"
        "            try:\n"
        "                open(f'f{i}', 'w').close()\n"
        "            except OSError:\n"
        "                pass\n"
        "            i += 1\n"
        "    os._exit(0)\n",
    )

    try:
        assert await env.run_pytest() == (TOTAL, TOTAL)
    finally:
        pid, grade_dir = record.read_text().split(" ", 1)
        stat = Path(f"/proc/{pid}/stat")
        while stat.exists() and not stat.read_text().split(") ")[1].startswith("Z"):
            await asyncio.sleep(0.1)
        shutil.rmtree(grade_dir, ignore_errors=True)
