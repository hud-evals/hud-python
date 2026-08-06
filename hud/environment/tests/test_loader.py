"""``load_environment``: resolve env references — source paths, modules, factories."""

from __future__ import annotations

import pytest

from hud.environment import load_environment


def test_load_environment_selects_by_attr_or_env_name(tmp_path) -> None:
    module = tmp_path / "envs.py"
    module.write_text(
        """
from hud import Environment

first = Environment("env-one")
second = Environment("env-two")
""".strip(),
        encoding="utf-8",
    )

    assert load_environment(module, name="first").name == "env-one"
    assert load_environment(module, name="env-two").name == "env-two"
    with pytest.raises(ValueError, match="multiple Environments"):
        load_environment(module)
    with pytest.raises(ValueError, match="no Environment named 'missing'"):
        load_environment(module, name="missing")

    single = tmp_path / "single.py"
    single.write_text("from hud import Environment\nenv = Environment('only')\n", encoding="utf-8")
    assert load_environment(single).name == "only"


@pytest.fixture
def factory_module(request):
    """An importable module exposing an env and a factory, cleaned up after."""
    import sys
    from types import ModuleType

    from hud.environment import Environment

    name = f"_loader_target_{request.node.name}"
    mod = ModuleType(name)
    setattr(mod, "env", Environment("declared"))
    setattr(mod, "make_env", lambda name="built": Environment(name))
    sys.modules[name] = mod
    yield name
    del sys.modules[name]


def test_module_factory_is_called_with_args(factory_module) -> None:
    env = load_environment(factory_module, name="make_env", args={"name": "from-factory"})

    assert env.name == "from-factory"


def test_module_env_attribute_is_returned_not_called(factory_module) -> None:
    # Environments are callable (legacy scenario surface); the instance must
    # be returned as-is, never invoked as a factory.
    assert load_environment(factory_module).name == "declared"


def test_module_factory_returning_non_environment_raises(factory_module) -> None:
    import sys

    setattr(sys.modules[factory_module], "make_env", lambda: object())

    with pytest.raises(ValueError, match="not an Environment"):
        load_environment(factory_module, name="make_env")


def test_unresolvable_references_raise() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_environment("no.such.module")
    with pytest.raises(FileNotFoundError, match="no environment source"):
        load_environment("missing/env.py")
    with pytest.raises(ValueError, match="args= applies to factory targets"):
        load_environment(__file__, args={"a": "b"})


def test_package_dir_does_not_shadow_factory_target(tmp_path, monkeypatch) -> None:
    # `mypkg:make_env` with a plain mypkg/ package in cwd is a module
    # reference; source scanning is only for env-declaring source trees.
    pkg = tmp_path / "shadowpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from hud.environment import Environment\n\n"
        "def make_env(name='shadowed'):\n    return Environment(name)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    env = load_environment("shadowpkg", name="make_env")

    assert env.name == "shadowed"


def test_named_attribute_resolves_a_package_that_also_has_env_py(tmp_path, monkeypatch) -> None:
    # `pkg:make_env` addresses an attribute, so an env.py sitting inside the
    # package must not capture it into a source scan.
    pkg = tmp_path / "bothpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from hud.environment import Environment\n\n"
        "def make_env(name='from-factory'):\n    return Environment(name)\n",
        encoding="utf-8",
    )
    (pkg / "env.py").write_text(
        "from hud.environment import Environment\n\nenv = Environment('from-source')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    assert load_environment("bothpkg", name="make_env").name == "from-factory"
    # ...while a bare reference still scans the source tree.
    assert load_environment("bothpkg").name == "from-source"


def test_every_reference_form_resolves(tmp_path, monkeypatch) -> None:
    """The full matrix, in one place: the shapes that competed for the same
    spelling are what made this resolution subtle."""
    (tmp_path / "env.py").write_text(
        "from hud.environment import Environment\n\nenv = Environment('from-env-py')\n",
        encoding="utf-8",
    )
    envs = tmp_path / "envs"  # a plain directory: an importable namespace package
    envs.mkdir()
    (envs / "one.py").write_text(
        "from hud.environment import Environment\n\nfoo = Environment('from-tree')\n",
        encoding="utf-8",
    )
    pkg = tmp_path / "pkg"  # a real package exposing a factory
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from hud.environment import Environment\n\n"
        "def make_env(name='from-factory'):\n    return Environment(name)\n",
        encoding="utf-8",
    )
    (pkg / "env.py").write_text(
        "from hud.environment import Environment\n\nenv = Environment('from-pkg-source')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    # a source file, however it is spelled — including the common serve form
    assert load_environment("env").name == "from-env-py"
    assert load_environment("env", name="env").name == "from-env-py"
    assert load_environment("env.py").name == "from-env-py"
    assert load_environment(tmp_path / "env.py").name == "from-env-py"

    # a source tree, with a name selecting inside it
    assert load_environment("envs", name="foo").name == "from-tree"

    # a package attribute: the factory wins over the env.py beside it
    assert load_environment("pkg", name="make_env").name == "from-factory"
    assert load_environment("pkg", name="make_env", args={"name": "x"}).name == "x"
    # ...while a bare reference to the same package still scans its source
    assert load_environment("pkg").name == "from-pkg-source"
