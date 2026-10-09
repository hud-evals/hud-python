"""Environment references: what ``load_environment`` and the serving process resolve.

A fixture tree holds every shape of source a project has: an ``env.py``, a
directory of modules, a package with a factory beside its ``env.py``, packages
with relative imports and re-exports, a module shadowing a stdlib name, and task
modules that export rows. Each row resolves one reference to an environment name
or to the error a user sees. The serving rows run the real serving process on
the same tree: it announces its port, answers ``hello``, and runs the
environment's shutdown hook on SIGTERM.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud.environment import load_environment

from .conftest import wire

if TYPE_CHECKING:
    from collections.abc import Iterator


def declare(name: str, variable: str = "env") -> str:
    """Source declaring Environment ``name``, whose shutdown hook leaves a marker in cwd."""
    return (
        "from pathlib import Path\n"
        "from hud import Environment\n"
        f"{variable} = Environment({name!r})\n"
        f"@{variable}.shutdown\n"
        f"async def _stopped_{variable}():\n"
        f"    Path('stopped-{name}').write_text('stopped')\n"
    )


TEMPLATE = '@env.template(id="solve")\nasync def solve():\n    yield "ok"\n    yield 1.0\n'

TREE = {
    "env.py": declare("from-env-py"),
    "envs/one.py": declare("from-tree", "foo"),
    "pkg/__init__.py": (
        "from pathlib import Path\n"
        "from hud.environment import Environment\n"
        "def make_env(name='from-factory'):\n"
        "    env = Environment(name)\n"
        "    @env.shutdown\n"
        "    async def _stopped():\n"
        "        Path(f'stopped-{name}').write_text('stopped')\n"
        "    return env\n"
    ),
    "pkg/env.py": declare("from-pkg-source"),
    "multi.py": declare("env-one", "first") + declare("env-two", "second"),
    "single.py": declare("only"),
    "alias.py": declare("shared") + "alias = env\n",
    "twins.py": declare("shared", "one") + declare("shared", "two"),
    "relpkg/__init__.py": "",
    "relpkg/world.py": declare("relative"),
    "relpkg/env.py": "from .world import env\n",
    "reexport/__init__.py": "from .env import env\n",
    "reexport/core.py": declare("reexported"),
    "reexport/env.py": "from .core import env\n" + TEMPLATE,
    "scan/scan_core.py": declare("scanned"),
    "scan/scan_templates.py": "from scan_core import env\n" + TEMPLATE,
    "scan/assembly.py": "from scan_core import env\nfrom scan_templates import solve\n",
    "shadow/json.py": (
        "import json\nfrom hud import Environment\nenv = Environment(json.loads('\"local\"'))\n"
    ),
    "exports/bound_source.py": declare("bound")
    + TEMPLATE
    + "def make_task():\n    return solve()\n",
    "exports/as_task.py": "from bound_source import make_task\nrows = make_task()\n",
    "exports/as_list.py": "from bound_source import make_task\nrows = [make_task()]\n",
    "exports/as_tuple.py": "from bound_source import make_task\nrows = (make_task(),)\n",
    "exports/as_taskset.py": (
        "from bound_source import make_task\nfrom hud.eval import Taskset\n"
        "rows = Taskset('rows', [make_task()])\n"
    ),
    "modpkg/__init__.py": "",
    "modpkg/factories.py": (
        "from hud.environment import Environment\n"
        "env = Environment('declared')\n"
        "def make_env(name='built'):\n    return Environment(name)\n"
        "class CallableEnvironment(Environment):\n"
        "    def __call__(self):\n        raise AssertionError('called as a factory')\n"
        "callable_env = CallableEnvironment('callable')\n"
        "def not_env():\n    return object()\n"
    ),
}


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The source tree as the working directory and an import root, as in a project."""
    root = tmp_path / "project"
    for name, content in TREE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.chdir(root)
    monkeypatch.syspath_prepend(str(root))
    before = set(sys.modules)
    yield root
    for name in set(sys.modules) - before:
        del sys.modules[name]


ROWS = [
    pytest.param("env", None, None, "from-env-py", id="bare-name-means-env-py"),
    pytest.param("env", "env", None, "from-env-py", id="bare-name-with-attribute"),
    pytest.param("env.py", None, None, "from-env-py", id="relative-file"),
    pytest.param("{root}/env.py", None, None, "from-env-py", id="absolute-file"),
    pytest.param("envs", "foo", None, "from-tree", id="directory-by-attribute"),
    pytest.param("pkg", "make_env", None, "from-factory", id="package-factory"),
    pytest.param("pkg", "make_env", {"name": "x"}, "x", id="package-factory-with-args"),
    pytest.param("pkg", None, None, "from-pkg-source", id="bare-package-scans-its-source"),
    pytest.param("multi.py", "first", None, "env-one", id="select-by-attribute"),
    pytest.param("multi.py", "env-two", None, "env-two", id="select-by-env-name"),
    pytest.param("single.py", None, None, "only", id="single-env-needs-no-name"),
    pytest.param("alias.py", None, None, "shared", id="re-exported-env-is-one-env"),
    pytest.param("alias.py", "shared", None, "shared", id="re-exported-env-by-name"),
    pytest.param("relpkg/env.py", None, None, "relative", id="package-relative-import"),
    pytest.param("reexport/env.py", None, None, "reexported", id="package-init-re-export"),
    pytest.param("scan", "scanned", None, "scanned", id="directory-of-modules"),
    pytest.param("exports/as_task.py", "bound", None, "bound", id="exported-task"),
    pytest.param("exports/as_list.py", "bound", None, "bound", id="exported-task-list"),
    pytest.param("exports/as_tuple.py", "bound", None, "bound", id="exported-task-tuple"),
    pytest.param("exports/as_taskset.py", "bound", None, "bound", id="exported-taskset"),
    pytest.param("shadow/json.py", None, None, "local", id="module-named-like-stdlib"),
    pytest.param("shadow", None, None, "local", id="directory-with-stdlib-name"),
    pytest.param("modpkg.factories", None, None, "declared", id="module-env-attribute"),
    pytest.param(
        "modpkg.factories", "make_env", {"name": "made"}, "made", id="module-factory-with-args"
    ),
    pytest.param(
        "modpkg.factories", "callable_env", None, "callable", id="callable-env-is-not-called"
    ),
    pytest.param(
        "multi.py",
        None,
        None,
        (ValueError, "multiple Environments in multi.py; select one by name"),
        id="several-envs-need-a-name",
    ),
    pytest.param(
        "multi.py",
        "missing",
        None,
        (ValueError, "no Environment named 'missing' found in multi.py"),
        id="unknown-name",
    ),
    pytest.param(
        "twins.py",
        "shared",
        None,
        (ValueError, "multiple Environments in twins.py"),
        id="distinct-envs-sharing-a-name",
    ),
    pytest.param(
        "modpkg.factories",
        "not_env",
        None,
        (ValueError, r"modpkg.factories:not_env resolved to <object object at .*>, not an Env"),
        id="factory-returning-something-else",
    ),
    pytest.param(
        "no.such.module", None, None, (ModuleNotFoundError, "No module named 'no'"), id="no-module"
    ),
    pytest.param(
        "missing/env.py",
        None,
        None,
        (FileNotFoundError, "no environment source at missing/env.py"),
        id="missing-file",
    ),
    pytest.param(
        "env.py",
        None,
        {"a": "b"},
        (ValueError, "args= applies to factory targets, not source path env.py"),
        id="args-on-a-source-file",
    ),
]


@pytest.mark.parametrize(("reference", "name", "args", "expected"), ROWS)
def test_a_reference_resolves_to_one_environment(
    tree: Path,
    reference: str,
    name: str | None,
    args: dict[str, Any] | None,
    expected: str | tuple[type[Exception], str],
) -> None:
    target: str | Path = reference.format(root=tree)
    if reference.startswith("{root}"):
        target = Path(target)

    if isinstance(expected, str):
        assert load_environment(target, name=name, args=args).name == expected
    else:
        error, message = expected
        with pytest.raises(error, match=message):
            load_environment(target, name=name, args=args)


def test_loading_a_source_again_returns_the_same_environment(tree: Path) -> None:
    scanned = load_environment("scan", name="scanned")
    package = load_environment("reexport/env.py")

    assert set(scanned.tasks) == {"solve"}
    assert load_environment("scan", name="scanned") is scanned
    assert load_environment(tree / "reexport", name="reexported") is package


@pytest.mark.parametrize("reference", ["shadow/json.py", "shadow"])
def test_loading_a_module_named_like_the_stdlib_leaves_the_stdlib_imported(
    tree: Path, reference: str
) -> None:
    load_environment(reference)

    assert sys.modules["json"] is json


@pytest.fixture
def two_roots(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """The same package name under two source roots."""
    packages = tuple(tmp_path / root / "source_root_package" for root in ("first", "second"))
    for package in packages:
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("from .env import env\n")
        (package / "core.py").write_text(declare(package.parent.name))
        (package / "env.py").write_text("from .core import env\n")
    before = set(sys.modules)
    yield packages[0], packages[1]
    for name in set(sys.modules) - before:
        del sys.modules[name]


@pytest.mark.parametrize("source", ["env.py", "__init__.py", "."])
def test_a_package_already_imported_from_another_root_is_refused(
    two_roots: tuple[Path, Path], source: str
) -> None:
    first, second = (package / source for package in two_roots)

    assert load_environment(first).name == "first"
    with pytest.raises(ValueError, match="already imported from a different source root"):
        load_environment(second)
    assert load_environment(first).name == "first"


def test_a_package_source_wins_over_other_import_roots(
    two_roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = two_roots
    monkeypatch.syspath_prepend(str(second.parent))
    monkeypatch.syspath_prepend(str(first.parent))

    assert load_environment(second / "env.py").name == "second"


SERVER = [sys.executable, "-m", "hud.environment.server"]
HUD_SERVE = [sys.executable, "-m", "hud.cli", "serve", "--port", "0"]


@pytest.mark.parametrize(
    ("command", "served", "stop"),
    [
        pytest.param([*SERVER, "env.py"], "from-env-py", signal.SIGTERM, id="server-sigterm"),
        pytest.param(
            [*SERVER, "multi.py", "--env", "env-two"], "env-two", signal.SIGTERM, id="server-env"
        ),
        pytest.param([*HUD_SERVE, "env:env"], "from-env-py", signal.SIGINT, id="hud-serve-ctrl-c"),
        pytest.param(
            [*HUD_SERVE, "pkg:make_env", "--arg", "name=demo"],
            "demo",
            signal.SIGINT,
            id="hud-serve-factory",
        ),
    ],
)
async def test_the_serving_process_announces_its_port_and_runs_shutdown_hooks(
    tree: Path, command: list[str], served: str, stop: signal.Signals
) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=tree,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "NO_COLOR": "1"},
    )
    try:
        port = await announced_port(process)
        async with wire(f"tcp://127.0.0.1:{port}") as control:
            hello = await control.call("hello", {})
        process.send_signal(stop)
        code = await asyncio.wait_for(process.wait(), timeout=60)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()

    assert hello["result"]["env"]["name"] == served
    assert code == 0
    assert [name for name in os.listdir(tree) if name.startswith("stopped-")] == [
        f"stopped-{served}"
    ]


async def announced_port(process: asyncio.subprocess.Process) -> int:
    assert process.stdout is not None
    async with asyncio.timeout(60):
        while line := await process.stdout.readline():
            text = line.decode().strip()
            if text.startswith("HUD_SERVE_PORT="):
                return int(text.removeprefix("HUD_SERVE_PORT="))
    assert process.stderr is not None
    raise AssertionError(f"exited without a port:\n{(await process.stderr.read()).decode()}")


@pytest.mark.parametrize(
    ("command", "error"),
    [
        pytest.param([*SERVER, "multi.py"], "multiple Environments in multi.py", id="ambiguous"),
    ],
)
async def test_the_serving_process_exits_with_the_load_error(
    tree: Path, command: list[str], error: str
) -> None:
    process = await asyncio.create_subprocess_exec(
        *command, cwd=tree, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

    assert process.returncode == 1
    assert b"HUD_SERVE_PORT=" not in stdout
    assert error in stderr.decode()
