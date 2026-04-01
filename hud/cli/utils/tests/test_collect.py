"""Tests for ``hud.cli.utils.collect`` — package imports, recursive discovery, cross-module imports.

Focuses on the hard cases that exercise new behavior:
- Package import via __init__.py with pkgutil discovery (ml-template pattern)
- Recursive task.py discovery at depth 2+ (was broken before)
- Cross-module imports resolved via project root discovery
- Priority ordering and fallback between package / file-scan modes
- Graceful degradation when files are broken

Basic dispatch (.py / .json / .jsonl / directory) and simple SDLC patterns
are already covered by test_sync.py::TestCollectTasks.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def _cleanup(*prefixes: str) -> None:
    """Remove cached modules matching any prefix to prevent test pollution."""
    for k in [k for k in sys.modules if any(k == p or k.startswith(p + ".") for p in prefixes)]:
        del sys.modules[k]


TASK_PY = """\
from hud.eval.task import Task
task = Task(env={{"name": "e"}}, scenario="e:s", args={{}}, slug="{slug}")
"""


class TestExtraSysPaths:
    """_import_tasks_from_module with extra_sys_paths — the core fix."""

    def test_cross_module_import_resolves(self, tmp_path: Path) -> None:
        """Task file can import a sibling module when project root is provided."""
        from hud.cli.utils.collect import _import_tasks_from_module

        mod = f"_cfg_{id(self)}"
        _write(tmp_path / f"{mod}.py", 'ARGS = {"mode": "fast"}\n')
        _write(
            tmp_path / "sub" / "task.py",
            f"""\
            from hud.eval.task import Task
            from {mod} import ARGS
            task = Task(env={{"name": "e"}}, scenario="e:s", args=ARGS, slug="t")
            """,
        )
        try:
            tasks = _import_tasks_from_module(
                tmp_path / "sub" / "task.py", extra_sys_paths=[str(tmp_path)]
            )
            assert len(tasks) == 1
            assert tasks[0].args == {"mode": "fast"}
        finally:
            _cleanup(mod)

    def test_paths_removed_after_import(self, tmp_path: Path) -> None:
        _write(tmp_path / "task.py", TASK_PY.format(slug="t"))
        from hud.cli.utils.collect import _import_tasks_from_module

        sentinel = str(tmp_path / "_sentinel_path")
        _import_tasks_from_module(tmp_path / "task.py", extra_sys_paths=[sentinel])
        assert sentinel not in sys.path


class TestPackageImport:
    """_collect_from_package — importing a directory as a Python package."""

    def test_pkgutil_discovery(self, tmp_path: Path) -> None:
        """The ml-template pattern: __init__.py uses pkgutil.iter_modules
        to discover sub-packages that re-export Task objects."""
        from hud.cli.utils.collect import _collect_from_package

        pkg = f"pkg_{id(self)}"
        _write(
            tmp_path / pkg / "__init__.py",
            """\
            import importlib, pkgutil
            from hud.eval.task import Task
            tasks = {}
            for info in pkgutil.iter_modules(__path__, __name__ + "."):
                if not info.ispkg:
                    continue
                mod = importlib.import_module(info.name)
                short = info.name.rsplit(".", 1)[-1]
                for attr in vars(mod).values():
                    if isinstance(attr, Task):
                        tasks[short] = attr
            """,
        )
        for name in ("fix_bug", "add_feat"):
            _write(tmp_path / pkg / name / "__init__.py", "from .task import task\n")
            _write(tmp_path / pkg / name / "task.py", TASK_PY.format(slug=name))

        try:
            result = _collect_from_package(tmp_path / pkg)
            assert {t.slug for t in result} == {"fix_bug", "add_feat"}
        finally:
            _cleanup(pkg)


class TestRecursiveDiscovery:
    """_collect_from_directory with recursive rglob for task.py files."""

    def test_depth_two(self, tmp_path: Path) -> None:
        """tasks/variants/debug_loss/task.py — was invisible before the fix."""
        from hud.cli.utils.collect import collect_tasks

        d = tmp_path / "d"
        _write(d / "variants" / "debug_loss" / "task.py", TASK_PY.format(slug="deep"))
        assert collect_tasks(str(d))[0].slug == "deep"

    def test_mixed_depths_skips_hidden(self, tmp_path: Path) -> None:
        """Collects at depth 1 + depth 2, skips .hidden and __pycache__."""
        from hud.cli.utils.collect import collect_tasks

        d = tmp_path / "d2"
        _write(d / "shallow" / "task.py", TASK_PY.format(slug="shallow"))
        _write(d / "cat" / "deep" / "task.py", TASK_PY.format(slug="deep"))
        _write(d / ".hidden" / "task.py", TASK_PY.format(slug="nope"))
        _write(d / "__pycache__" / "task.py", TASK_PY.format(slug="nope2"))

        result = collect_tasks(str(d))
        assert {t.slug for t in result} == {"shallow", "deep"}

    def test_root_task_py_not_re_processed(self, tmp_path: Path) -> None:
        """A root-level task.py is handled by Priority 1, not re-imported by rglob."""
        from hud.cli.utils.collect import _collect_from_directory

        _write(tmp_path / "task.py", TASK_PY.format(slug="root"))
        _write(tmp_path / "sub" / "task.py", TASK_PY.format(slug="sub"))

        result = _collect_from_directory(tmp_path)
        assert len(result) == 1
        assert result[0].slug == "root"


class TestPriorityOrdering:
    """Package import (Priority 0) vs file scan, and fallback behavior."""

    def test_package_wins_over_file_scan(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        pkg = f"ppkg_{id(self)}"
        _write(
            tmp_path / pkg / "__init__.py",
            """\
            from hud.eval.task import Task
            tasks = {"x": Task(env={"name": "e"}, scenario="e:s", args={}, slug="from-init")}
            """,
        )
        _write(tmp_path / pkg / "sub" / "task.py", TASK_PY.format(slug="from-file"))

        try:
            result = collect_tasks(str(tmp_path / pkg))
            assert [t.slug for t in result] == ["from-init"]
        finally:
            _cleanup(pkg)

    def test_empty_package_falls_back(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        pkg = f"epkg_{id(self)}"
        _write(tmp_path / pkg / "__init__.py", "# nothing\n")
        _write(tmp_path / pkg / "fix" / "task.py", TASK_PY.format(slug="fallback"))

        try:
            assert collect_tasks(str(tmp_path / pkg))[0].slug == "fallback"
        finally:
            _cleanup(pkg)

    def test_broken_package_falls_back(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _collect_from_directory

        pkg = f"bpkg_{id(self)}"
        d = tmp_path / pkg
        _write(d / "__init__.py", "import nonexistent_xyz_module\n")
        _write(d / "ok" / "task.py", TASK_PY.format(slug="survived"))

        try:
            assert _collect_from_directory(d)[0].slug == "survived"
        finally:
            _cleanup(pkg)


class TestCrossModuleImport:
    """Cross-module imports resolved via _find_project_root."""

    def test_task_imports_from_project_root(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        mod = f"_shared_{id(self)}"
        _write(tmp_path / f"{mod}.py", 'VAL = {"k": "v"}\n')
        _write(
            tmp_path / "mytasks" / "t1" / "task.py",
            f"""\
            from hud.eval.task import Task
            from {mod} import VAL
            task = Task(env={{"name": "e"}}, scenario="e:s", args=VAL, slug="cross")
            """,
        )
        try:
            result = collect_tasks(str(tmp_path / "mytasks"))
            assert result[0].args == {"k": "v"}
        finally:
            _cleanup(mod)


class TestMLTemplateEndToEnd:
    """Realistic ml-template-main structure with pkgutil + nested variants."""

    @staticmethod
    def _build(root: Path, pkg: str) -> None:
        (root / "pyproject.toml").write_text("[project]\nname='ml-env'\n")
        t = root / pkg
        _write(
            t / "__init__.py",
            """\
            import importlib, pkgutil
            from hud.eval.task import Task
            tasks = {}
            for info in pkgutil.iter_modules(__path__, __name__ + "."):
                if not info.ispkg:
                    continue
                mod = importlib.import_module(info.name)
                short = info.name.rsplit(".", 1)[-1]
                for attr in vars(mod).values():
                    if isinstance(attr, Task):
                        tasks[short] = attr
            """,
        )
        for name in ("emb_adapt", "emb_debug"):
            _write(t / name / "__init__.py", "from .task import task\n")
            _write(t / name / "task.py", TASK_PY.format(slug=name))
        _write(t / "variants" / "__init__.py", "")
        _write(t / "variants" / "vlm_fix" / "__init__.py", "")
        _write(t / "variants" / "vlm_fix" / "task.py", TASK_PY.format(slug="vlm_fix"))

    def test_package_collects_top_level(self, tmp_path: Path) -> None:
        pkg = f"ml_{id(self)}"
        self._build(tmp_path, pkg)
        from hud.cli.utils.collect import collect_tasks

        try:
            slugs = {t.slug for t in collect_tasks(str(tmp_path / pkg))}
            assert {"emb_adapt", "emb_debug"} <= slugs
        finally:
            _cleanup(pkg)

    def test_variants_found_by_file_scan(self, tmp_path: Path) -> None:
        """Variant tasks with empty __init__.py aren't found by pkgutil,
        but rglob picks them up when falling through to file scan."""
        from hud.cli.utils.collect import _collect_from_directory

        d = tmp_path / "variants_only"
        d.mkdir()
        _write(d / "variants" / "__init__.py", "")
        _write(d / "variants" / "vlm_fix" / "__init__.py", "")
        _write(d / "variants" / "vlm_fix" / "task.py", TASK_PY.format(slug="vlm"))

        assert _collect_from_directory(d)[0].slug == "vlm"


class TestGracefulDegradation:
    def test_broken_sibling_doesnt_block_others(self, tmp_path: Path) -> None:
        """One broken task.py doesn't prevent collection of valid siblings."""
        from hud.cli.utils.collect import _collect_from_directory

        _write(tmp_path / "good" / "task.py", TASK_PY.format(slug="good"))
        _write(tmp_path / "bad" / "task.py", "import nonexistent_xyz_module\n")

        result = _collect_from_directory(tmp_path)
        assert len(result) == 1
        assert result[0].slug == "good"
