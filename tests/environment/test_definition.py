"""Defining an environment: what a reference resolves to, and what is refused at definition.

A fixture tree holds every shape of source a project has: an ``env.py``, a
directory of modules, a package with a factory beside its ``env.py``, packages
with relative imports and re-exports, a module shadowing a stdlib name, and task
modules that export rows. Each row resolves one reference with
``load_environment`` to an environment name or to the error a user sees. The
serving processes resolve references the same way; ``tests/protocol/test_serving.py``
runs them.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pytest

from hud.environment import Environment, load_environment

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


def declare(name: str, variable: str = "env") -> str:
    """Source declaring Environment ``name`` as ``variable``."""
    return f"from hud import Environment\n{variable} = Environment({name!r})\n"


TEMPLATE = '@env.template(id="solve")\nasync def solve():\n    yield "ok"\n    yield 1.0\n'

TREE = {
    "env.py": declare("from-env-py"),
    "envs/one.py": declare("from-tree", "foo"),
    "pkg/__init__.py": (
        "from hud.environment import Environment\n"
        "def make_env(name='from-factory'):\n"
        "    return Environment(name)\n"
    ),
    "pkg/env.py": declare("from-pkg-source"),
    "multi.py": declare("env-one", "first") + declare("env-two", "second"),
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
    "modpkg/__init__.py": "",
    "modpkg/factories.py": (
        "from hud.environment import Environment\n"
        "env = Environment('declared')\n"
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
    pytest.param("env.py", None, None, "from-env-py", id="relative-file"),
    pytest.param("envs", "foo", None, "from-tree", id="directory-by-attribute"),
    pytest.param("pkg", "make_env", {"name": "x"}, "x", id="package-factory-with-args"),
    pytest.param("pkg", None, None, "from-pkg-source", id="bare-package-scans-its-source"),
    pytest.param("multi.py", "first", None, "env-one", id="select-by-attribute"),
    pytest.param("multi.py", "env-two", None, "env-two", id="select-by-env-name"),
    pytest.param("alias.py", None, None, "shared", id="re-exported-env-is-one-env"),
    pytest.param("relpkg/env.py", None, None, "relative", id="package-relative-import"),
    pytest.param("reexport/env.py", None, None, "reexported", id="package-init-re-export"),
    pytest.param("scan", "scanned", None, "scanned", id="directory-of-modules"),
    pytest.param("exports/as_task.py", "bound", None, "bound", id="exported-task"),
    pytest.param("modpkg.factories", None, None, "declared", id="module-env-attribute"),
    pytest.param(
        "modpkg.factories", "callable_env", None, "callable", id="callable-env-is-not-called"
    ),
    pytest.param(
        "multi.py",
        None,
        None,
        ValueError,
        id="several-envs-need-a-name",
    ),
    pytest.param(
        "multi.py",
        "missing",
        None,
        ValueError,
        id="unknown-name",
    ),
    pytest.param(
        "twins.py",
        "shared",
        None,
        ValueError,
        id="distinct-envs-sharing-a-name",
    ),
    pytest.param(
        "modpkg.factories",
        "not_env",
        None,
        ValueError,
        id="factory-returning-something-else",
    ),
    pytest.param(
        "missing/env.py",
        None,
        None,
        FileNotFoundError,
        id="missing-file",
    ),
    pytest.param(
        "env.py",
        None,
        {"a": "b"},
        ValueError,
        id="args-on-a-source-file",
    ),
]


@pytest.mark.parametrize(("reference", "name", "args", "expected"), ROWS)
def test_a_reference_resolves_to_one_environment(
    tree: Path,
    reference: str,
    name: str | None,
    args: dict[str, Any] | None,
    expected: str | type[Exception],
) -> None:
    if isinstance(expected, str):
        assert load_environment(reference, name=name, args=args).name == expected
    else:
        with pytest.raises(expected):
            load_environment(reference, name=name, args=args)


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


@pytest.mark.parametrize("source", ["env.py", "__init__.py"])
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


def register_a_template_twice(env: Environment) -> None:
    @env.template(id="twice")
    async def first():
        yield "a"

    @env.template(id="twice")
    async def second():
        yield "b"


@pytest.mark.parametrize(
    "declare_mistake",
    [
        pytest.param(register_a_template_twice, id="a-template-twice"),
        pytest.param(
            lambda env: [env.workspace("a"), env.workspace("b")],
            id="two-workspaces-on-one-capability",
        ),
    ],
)
def test_an_authoring_mistake_raises_at_definition(
    declare_mistake: Callable[[Environment], Any],
) -> None:
    with pytest.raises(ValueError):
        declare_mistake(Environment("authoring"))
