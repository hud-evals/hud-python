"""Tests for ``hud.cli.utils.collect`` — task collection from files, directories, and packages.

Covers:
- _find_project_root: locating the project root for sys.path setup
- _import_tasks_from_module with extra_sys_paths: cross-module imports
- _collect_from_package: importing a directory as a Python package
- _collect_from_directory: recursive task.py discovery (SDLC pattern)
- collect_tasks: entry-point dispatching (.py, .json, .jsonl, directory, errors)
- End-to-end: ml-template-main style nested tasks/ package
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def _cleanup_modules(*prefixes: str) -> None:
    """Remove cached modules matching any prefix to prevent test pollution."""
    to_remove = [k for k in sys.modules if any(k == p or k.startswith(p + ".") for p in prefixes)]
    for k in to_remove:
        del sys.modules[k]


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    def test_finds_pyproject_toml(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _find_project_root

        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        sub = tmp_path / "tasks"
        sub.mkdir()

        assert _find_project_root(sub) == str(tmp_path)

    def test_finds_env_py(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _find_project_root

        (tmp_path / "env.py").write_text("env = None\n")
        sub = tmp_path / "tasks" / "deep"
        sub.mkdir(parents=True)

        assert _find_project_root(sub) == str(tmp_path)

    def test_finds_hud_dir(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _find_project_root

        (tmp_path / ".hud").mkdir()
        sub = tmp_path / "tasks"
        sub.mkdir()

        assert _find_project_root(sub) == str(tmp_path)

    def test_finds_git_dir(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _find_project_root

        (tmp_path / ".git").mkdir()
        sub = tmp_path / "src" / "tasks"
        sub.mkdir(parents=True)

        assert _find_project_root(sub) == str(tmp_path)

    def test_returns_nearest_marker(self, tmp_path: Path) -> None:
        """Should return the closest ancestor with a marker, not keep walking."""
        from hud.cli.utils.collect import _find_project_root

        (tmp_path / "pyproject.toml").write_text("")
        inner = tmp_path / "subproject"
        inner.mkdir()
        (inner / "pyproject.toml").write_text("")
        deep = inner / "tasks"
        deep.mkdir()

        assert _find_project_root(deep) == str(inner)

    def test_finds_self_if_markers_in_same_dir(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _find_project_root

        (tmp_path / "pyproject.toml").write_text("")
        assert _find_project_root(tmp_path) == str(tmp_path)


# ---------------------------------------------------------------------------
# _import_tasks_from_module with extra_sys_paths
# ---------------------------------------------------------------------------


class TestImportWithExtraSysPaths:
    def test_cross_module_import_with_extra_paths(self, tmp_path: Path) -> None:
        """Task file can import from a sibling module when project root is provided."""
        from hud.cli.utils.collect import _import_tasks_from_module

        mod_name = f"_helpers_{id(self)}"
        _write_file(
            tmp_path / f"{mod_name}.py",
            """\
            GREETING = "hello"
            """,
        )

        _write_file(
            tmp_path / "sub" / "task.py",
            f"""\
            from hud.eval.task import Task
            from {mod_name} import GREETING
            task = Task(
                env={{"name": "e"}},
                scenario="e:greet",
                args={{"msg": GREETING}},
                slug="greet-task",
            )
            """,
        )

        try:
            tasks = _import_tasks_from_module(
                tmp_path / "sub" / "task.py",
                extra_sys_paths=[str(tmp_path)],
            )
            assert len(tasks) == 1
            assert tasks[0].slug == "greet-task"
            assert tasks[0].args["msg"] == "hello"
        finally:
            _cleanup_modules(mod_name)

    def test_extra_paths_cleaned_up(self, tmp_path: Path) -> None:
        """Extra sys.path entries are removed after import."""
        from hud.cli.utils.collect import _import_tasks_from_module

        _write_file(
            tmp_path / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="t")
            """,
        )

        extra = str(tmp_path / "nonexistent_sentinel")
        _import_tasks_from_module(tmp_path / "task.py", extra_sys_paths=[extra])

        assert extra not in sys.path


# ---------------------------------------------------------------------------
# _collect_from_package
# ---------------------------------------------------------------------------


class TestCollectFromPackage:
    def test_package_with_tasks_dict(self, tmp_path: Path) -> None:
        """Package __init__.py that builds a tasks dict (ml-template pattern)."""
        from hud.cli.utils.collect import _collect_from_package

        pkg_name = f"pkg_dict_{id(self)}"
        pkg = tmp_path / pkg_name
        pkg.mkdir()

        _write_file(
            pkg / "__init__.py",
            """\
            from hud.eval.task import Task
            tasks = {
                "add": Task(env={"name": "e"}, scenario="e:add", args={"x": 1}, slug="add"),
                "sub": Task(env={"name": "e"}, scenario="e:sub", args={"x": 2}, slug="sub"),
            }
            """,
        )

        try:
            result = _collect_from_package(pkg)
            assert len(result) == 2
            assert {t.slug for t in result} == {"add", "sub"}
        finally:
            _cleanup_modules(pkg_name)

    def test_package_with_tasks_list(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _collect_from_package

        pkg_name = f"pkg_list_{id(self)}"
        pkg = tmp_path / pkg_name
        pkg.mkdir()

        _write_file(
            pkg / "__init__.py",
            """\
            from hud.eval.task import Task
            tasks = [
                Task(env={"name": "e"}, scenario="e:s", args={}, slug="list-t1"),
            ]
            """,
        )

        try:
            result = _collect_from_package(pkg)
            assert len(result) == 1
            assert result[0].slug == "list-t1"
        finally:
            _cleanup_modules(pkg_name)

    def test_package_with_module_level_attrs(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _collect_from_package

        pkg_name = f"pkg_attrs_{id(self)}"
        pkg = tmp_path / pkg_name
        pkg.mkdir()

        _write_file(
            pkg / "__init__.py",
            """\
            from hud.eval.task import Task
            my_task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="loose")
            """,
        )

        try:
            result = _collect_from_package(pkg)
            assert len(result) == 1
            assert result[0].slug == "loose"
        finally:
            _cleanup_modules(pkg_name)

    def test_package_import_error_raises(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _collect_from_package

        pkg_name = f"broken_pkg_{id(self)}"
        pkg = tmp_path / pkg_name
        pkg.mkdir()
        _write_file(pkg / "__init__.py", "import nonexistent_zyx_abc\n")

        try:
            with pytest.raises(ImportError, match="nonexistent_zyx_abc"):
                _collect_from_package(pkg)
        finally:
            _cleanup_modules(pkg_name)

    def test_package_with_pkgutil_discovery(self, tmp_path: Path) -> None:
        """Package that uses pkgutil.iter_modules to discover sub-packages."""
        from hud.cli.utils.collect import _collect_from_package

        pkg_name = f"pkg_pkgutil_{id(self)}"
        pkg = tmp_path / pkg_name
        pkg.mkdir()

        _write_file(
            pkg / "__init__.py",
            """\
            import importlib
            import pkgutil
            from hud.eval.task import Task

            tasks = {}
            for _info in pkgutil.iter_modules(__path__, __name__ + "."):
                if not _info.ispkg:
                    continue
                mod = importlib.import_module(_info.name)
                pkg_short = _info.name.rsplit(".", 1)[-1]
                for _attr_name, attr in vars(mod).items():
                    if isinstance(attr, Task):
                        tasks[pkg_short] = attr
            """,
        )

        sub_a = pkg / "task_a"
        sub_a.mkdir()
        _write_file(sub_a / "__init__.py", "from .task import task\n")
        _write_file(
            sub_a / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:do_a", args={}, slug="a")
            """,
        )

        sub_b = pkg / "task_b"
        sub_b.mkdir()
        _write_file(sub_b / "__init__.py", "from .task import task\n")
        _write_file(
            sub_b / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:do_b", args={"n": 5}, slug="b")
            """,
        )

        try:
            result = _collect_from_package(pkg)
            assert len(result) == 2
            assert {t.slug for t in result} == {"a", "b"}
        finally:
            _cleanup_modules(pkg_name)


# ---------------------------------------------------------------------------
# _collect_from_directory — recursive SDLC discovery
# ---------------------------------------------------------------------------


class TestRecursiveDirectoryCollection:
    def test_one_level_deep(self, tmp_path: Path) -> None:
        """Standard SDLC pattern: dir/checkout/task.py (depth 1)."""
        from hud.cli.utils.collect import collect_tasks

        tasks_dir = tmp_path / "mytasks"
        task_sub = tasks_dir / "checkout"
        task_sub.mkdir(parents=True)
        _write_file(
            task_sub / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:checkout", args={}, slug="checkout")
            """,
        )

        result = collect_tasks(str(tasks_dir))
        assert len(result) == 1
        assert result[0].slug == "checkout"

    def test_two_levels_deep(self, tmp_path: Path) -> None:
        """Nested SDLC pattern: dir/variants/debug_loss/task.py (depth 2)."""
        from hud.cli.utils.collect import collect_tasks

        tasks_dir = tmp_path / "mytasks2"
        nested = tasks_dir / "variants" / "debug_loss"
        nested.mkdir(parents=True)
        _write_file(
            nested / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(
                env={"name": "e"}, scenario="e:fix",
                args={"bug": "loss"}, slug="debug-loss",
            )
            """,
        )

        result = collect_tasks(str(tasks_dir))
        assert len(result) == 1
        assert result[0].slug == "debug-loss"

    def test_mixed_depths(self, tmp_path: Path) -> None:
        """Tasks at different nesting depths collected together."""
        from hud.cli.utils.collect import collect_tasks

        tasks_dir = tmp_path / "mixed"

        _write_file(
            tasks_dir / "shallow" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="shallow")
            """,
        )

        _write_file(
            tasks_dir / "category" / "deep" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="deep")
            """,
        )

        result = collect_tasks(str(tasks_dir))
        assert len(result) == 2
        assert {t.slug for t in result} == {"shallow", "deep"}

    def test_skips_dotdirs_and_underscore_dirs(self, tmp_path: Path) -> None:
        """Directories starting with . or _ are skipped."""
        from hud.cli.utils.collect import collect_tasks

        tasks_dir = tmp_path / "skiptest"
        _write_file(
            tasks_dir / ".hidden" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="hidden")
            """,
        )
        _write_file(
            tasks_dir / "__pycache__" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="cache")
            """,
        )
        _write_file(
            tasks_dir / "valid" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="valid")
            """,
        )

        result = collect_tasks(str(tasks_dir))
        assert len(result) == 1
        assert result[0].slug == "valid"

    def test_priority_0_package_takes_precedence(self, tmp_path: Path) -> None:
        """If directory has __init__.py with tasks, package import wins over file scan."""
        from hud.cli.utils.collect import collect_tasks

        pkg_name = f"pkgprio_{id(self)}"
        pkg = tmp_path / pkg_name
        pkg.mkdir()

        _write_file(
            pkg / "__init__.py",
            """\
            from hud.eval.task import Task
            tasks = {
                "from_init": Task(env={"name": "e"}, scenario="e:s", args={}, slug="from-init"),
            }
            """,
        )

        _write_file(
            pkg / "sub" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="from-file")
            """,
        )

        try:
            result = collect_tasks(str(pkg))
            assert len(result) == 1
            assert result[0].slug == "from-init"
        finally:
            _cleanup_modules(pkg_name)

    def test_fallback_from_package_to_file_scan(self, tmp_path: Path) -> None:
        """If package __init__.py has no tasks, falls back to file scan."""
        from hud.cli.utils.collect import collect_tasks

        pkg_name = f"fallback_{id(self)}"
        pkg = tmp_path / pkg_name
        pkg.mkdir()
        _write_file(pkg / "__init__.py", "# empty package\n")

        _write_file(
            pkg / "fix_bug" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:fix", args={}, slug="fix-bug")
            """,
        )

        try:
            result = collect_tasks(str(pkg))
            assert len(result) == 1
            assert result[0].slug == "fix-bug"
        finally:
            _cleanup_modules(pkg_name)


# ---------------------------------------------------------------------------
# Cross-module imports via project root discovery
# ---------------------------------------------------------------------------


class TestCrossModuleImports:
    def test_task_imports_sibling_module(self, tmp_path: Path) -> None:
        """task.py in a subdirectory can import from a module at the project root."""
        from hud.cli.utils.collect import collect_tasks

        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")

        mod_name = f"shared_cfg_{id(self)}"
        _write_file(
            tmp_path / f"{mod_name}.py",
            """\
            DEFAULT_ARGS = {"mode": "fast"}
            """,
        )

        _write_file(
            tmp_path / "mytasks" / "benchmark" / "task.py",
            f"""\
            from hud.eval.task import Task
            from {mod_name} import DEFAULT_ARGS
            task = Task(
                env={{"name": "e"}},
                scenario="e:bench",
                args=DEFAULT_ARGS,
                slug="benchmark",
            )
            """,
        )

        try:
            result = collect_tasks(str(tmp_path / "mytasks"))
            assert len(result) == 1
            assert result[0].slug == "benchmark"
            assert result[0].args == {"mode": "fast"}
        finally:
            _cleanup_modules(mod_name)


# ---------------------------------------------------------------------------
# End-to-end: ml-template-main style nested package
# ---------------------------------------------------------------------------


class TestMLTemplatePattern:
    @staticmethod
    def _build_ml_template(root: Path, pkg_name: str) -> Path:
        """Build a miniature ml-template-main style project.

        Structure:
            root/
              pyproject.toml
              {pkg_name}/
                __init__.py       # pkgutil discovery
                emb_adapt/
                  __init__.py     # from .task import task
                  task.py
                emb_debug/
                  __init__.py
                  task.py
                variants/
                  __init__.py     # empty (sub-category)
                  vlm_fix/
                    __init__.py   # empty
                    task.py
        """
        (root / "pyproject.toml").write_text("[project]\nname='ml-env'\n")

        tasks = root / pkg_name

        _write_file(
            tasks / "__init__.py",
            """\
            import importlib
            import pkgutil
            from hud.eval.task import Task

            tasks = {}
            task_ids = {}
            for _info in pkgutil.iter_modules(__path__, __name__ + "."):
                if not _info.ispkg:
                    continue
                mod = importlib.import_module(_info.name)
                short = _info.name.rsplit(".", 1)[-1]
                for _attr_name, attr in vars(mod).items():
                    if isinstance(attr, Task):
                        tasks[short] = attr
                        slug = getattr(attr, "slug", None)
                        if isinstance(slug, str) and slug:
                            task_ids[short] = slug
            """,
        )

        for name, slug in [("emb_adapt", "emb-adapt"), ("emb_debug", "emb-debug")]:
            sub = tasks / name
            _write_file(sub / "__init__.py", "from .task import task\n")
            _write_file(
                sub / "task.py",
                f"""\
                from hud.eval.task import Task
                task = Task(
                    env={{"name": "ml-env"}},
                    scenario="ml-env:train",
                    args={{"task_type": "{name}"}},
                    slug="{slug}",
                )
                """,
            )

        variants = tasks / "variants"
        _write_file(variants / "__init__.py", "")
        vlm = variants / "vlm_fix"
        _write_file(vlm / "__init__.py", "")
        _write_file(
            vlm / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(
                env={"name": "ml-env"},
                scenario="ml-env:train",
                args={"task_type": "vlm_fix"},
                slug="vlm-fix",
            )
            """,
        )

        return root

    def test_package_import_collects_top_level_tasks(self, tmp_path: Path) -> None:
        """Package import (Priority 0) collects tasks from immediate sub-packages
        that re-export via __init__.py."""
        from hud.cli.utils.collect import collect_tasks

        pkg_name = f"mltasks_{id(self)}"
        root = self._build_ml_template(tmp_path, pkg_name)

        try:
            result = collect_tasks(str(root / pkg_name))
            slugs = {t.slug for t in result}
            assert "emb-adapt" in slugs
            assert "emb-debug" in slugs
        finally:
            _cleanup_modules(pkg_name)

    def test_file_scan_finds_nested_variants(self, tmp_path: Path) -> None:
        """rglob finds nested variant task.py files."""
        pkg_name = f"mltasks_rglob_{id(self)}"
        root = self._build_ml_template(tmp_path, pkg_name)
        tasks_dir = root / pkg_name

        found_files = sorted(tasks_dir.rglob("task.py"))
        rel_paths = [str(f.relative_to(tasks_dir)) for f in found_files]
        assert any("vlm_fix" in p for p in rel_paths)

    def test_collect_from_project_root_finds_tasks(self, tmp_path: Path) -> None:
        """Running collect_tasks on the project root discovers tasks."""
        from hud.cli.utils.collect import collect_tasks

        pkg_name = f"mltasks_root_{id(self)}"
        root = self._build_ml_template(tmp_path, pkg_name)

        try:
            result = collect_tasks(str(root))
            assert len(result) >= 2
        finally:
            _cleanup_modules(pkg_name)

    def test_variant_tasks_found_when_init_empty(self, tmp_path: Path) -> None:
        """Variant sub-packages with empty __init__.py are not found via
        pkgutil (by design), but their task.py files are found by rglob
        when the package path is scanned directly as a directory."""
        from hud.cli.utils.collect import _collect_from_directory

        tasks_dir = tmp_path / "scan_variants"
        tasks_dir.mkdir()

        variants = tasks_dir / "variants"
        _write_file(variants / "__init__.py", "")

        vlm = variants / "vlm_fix"
        _write_file(vlm / "__init__.py", "")
        _write_file(
            vlm / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(
                env={"name": "e"},
                scenario="e:fix",
                args={},
                slug="vlm-fix-direct",
            )
            """,
        )

        result = _collect_from_directory(tasks_dir)
        assert len(result) == 1
        assert result[0].slug == "vlm-fix-direct"


# ---------------------------------------------------------------------------
# _find_project_root — edge cases
# ---------------------------------------------------------------------------


class TestFindProjectRootEdgeCases:
    def test_returns_none_when_no_markers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No project markers anywhere in the hierarchy -> None."""
        from hud.cli.utils.collect import _find_project_root

        isolated = tmp_path / "isolated" / "a" / "b" / "c"
        isolated.mkdir(parents=True)

        orig_parent = Path.parent.fget  # type: ignore[union-attr]
        root_sentinel = tmp_path / "isolated"

        def _capped_parent(self: Path) -> Path:
            if self == root_sentinel:
                return self
            return orig_parent(self)  # type: ignore[misc]

        monkeypatch.setattr(Path, "parent", property(_capped_parent))

        assert _find_project_root(isolated) is None


# ---------------------------------------------------------------------------
# _collect_from_directory — Priority 1 (root task.py / tasks.py)
# ---------------------------------------------------------------------------


class TestDirectoryPriority1:
    def test_root_task_py(self, tmp_path: Path) -> None:
        """A directory with a task.py at the root uses Priority 1."""
        from hud.cli.utils.collect import _collect_from_directory

        _write_file(
            tmp_path / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="root-task")
            """,
        )

        result = _collect_from_directory(tmp_path)
        assert len(result) == 1
        assert result[0].slug == "root-task"

    def test_root_tasks_py(self, tmp_path: Path) -> None:
        """A directory with a tasks.py (plural) at the root uses Priority 1."""
        from hud.cli.utils.collect import _collect_from_directory

        _write_file(
            tmp_path / "tasks.py",
            """\
            from hud.eval.task import Task
            tasks = [
                Task(env={"name": "e"}, scenario="e:a", args={}, slug="ta"),
                Task(env={"name": "e"}, scenario="e:b", args={}, slug="tb"),
            ]
            """,
        )

        result = _collect_from_directory(tmp_path)
        assert len(result) == 2
        assert {t.slug for t in result} == {"ta", "tb"}

    def test_root_task_py_beats_subdirectory_tasks(self, tmp_path: Path) -> None:
        """Priority 1 (root task.py) takes precedence over Priority 2 (sub task.py)."""
        from hud.cli.utils.collect import _collect_from_directory

        _write_file(
            tmp_path / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="root")
            """,
        )
        _write_file(
            tmp_path / "sub" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="sub")
            """,
        )

        result = _collect_from_directory(tmp_path)
        assert len(result) == 1
        assert result[0].slug == "root"


# ---------------------------------------------------------------------------
# _collect_from_directory — Priority 3 (fallback .py files in root)
# ---------------------------------------------------------------------------


class TestDirectoryPriority3:
    def test_loose_py_files_in_root(self, tmp_path: Path) -> None:
        """When no task.py / tasks.py / sub/task.py exist, loose .py files are imported."""
        from hud.cli.utils.collect import _collect_from_directory

        _write_file(
            tmp_path / "my_scenario.py",
            """\
            from hud.eval.task import Task
            scenario = Task(env={"name": "e"}, scenario="e:s", args={}, slug="loose-py")
            """,
        )

        result = _collect_from_directory(tmp_path)
        assert len(result) == 1
        assert result[0].slug == "loose-py"

    def test_skips_env_and_conftest_files(self, tmp_path: Path) -> None:
        """env.py, conftest.py, setup.py are skipped in Priority 3.

        __init__.py is not tested here because its presence triggers Priority 0.
        """
        from hud.cli.utils.collect import _collect_from_directory

        for skip_name in ("env.py", "conftest.py", "setup.py"):
            _write_file(
                tmp_path / skip_name,
                """\
                from hud.eval.task import Task
                t = Task(env={"name": "e"}, scenario="e:s", args={}, slug="skip")
                """,
            )

        _write_file(
            tmp_path / "real_task.py",
            """\
            from hud.eval.task import Task
            t = Task(env={"name": "e"}, scenario="e:s", args={}, slug="real")
            """,
        )

        result = _collect_from_directory(tmp_path)
        assert len(result) == 1
        assert result[0].slug == "real"

    def test_empty_directory_returns_nothing(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import _collect_from_directory

        result = _collect_from_directory(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# _collect_from_directory — error handling paths
# ---------------------------------------------------------------------------


class TestDirectoryErrorPaths:
    def test_broken_root_task_py_is_warned(self, tmp_path: Path) -> None:
        """A broken task.py at root is logged as warning, not raised."""
        from hud.cli.utils.collect import _collect_from_directory

        _write_file(tmp_path / "task.py", "import nonexistent_xyz_module\n")

        result = _collect_from_directory(tmp_path)
        assert result == []

    def test_broken_subdirectory_task_py_is_warned(self, tmp_path: Path) -> None:
        """A broken sub/task.py is logged as warning; other tasks still collected."""
        from hud.cli.utils.collect import _collect_from_directory

        _write_file(
            tmp_path / "good" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="good")
            """,
        )
        _write_file(tmp_path / "bad" / "task.py", "import nonexistent_xyz_module\n")

        result = _collect_from_directory(tmp_path)
        assert len(result) == 1
        assert result[0].slug == "good"

    def test_package_import_failure_falls_back(self, tmp_path: Path) -> None:
        """If __init__.py fails to import, falls back to file scan."""
        from hud.cli.utils.collect import _collect_from_directory

        pkg_name = f"broken_init_{id(self)}"
        pkg = tmp_path / pkg_name
        pkg.mkdir()
        _write_file(pkg / "__init__.py", "import nonexistent_xyz_module\n")
        _write_file(
            pkg / "real" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="fallback")
            """,
        )

        try:
            result = _collect_from_directory(pkg)
            assert len(result) == 1
            assert result[0].slug == "fallback"
        finally:
            _cleanup_modules(pkg_name)


# ---------------------------------------------------------------------------
# collect_tasks — entry-point dispatch
# ---------------------------------------------------------------------------


class TestCollectTasksEntryPoint:
    def test_single_py_file(self, tmp_path: Path) -> None:
        """collect_tasks with a single .py file path."""
        from hud.cli.utils.collect import collect_tasks

        _write_file(
            tmp_path / "my_task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="single")
            """,
        )

        result = collect_tasks(str(tmp_path / "my_task.py"))
        assert len(result) == 1
        assert result[0].slug == "single"

    def test_json_file(self, tmp_path: Path) -> None:
        """collect_tasks with a .json file containing task dicts."""
        from hud.cli.utils.collect import collect_tasks

        data = [
            {"env": {"name": "e"}, "scenario": "e:s", "args": {"k": "v"}, "slug": "json-t1"},
            {"env": {"name": "e"}, "scenario": "e:s2", "args": {}, "slug": "json-t2"},
        ]
        (tmp_path / "tasks.json").write_text(json.dumps(data))

        result = collect_tasks(str(tmp_path / "tasks.json"))
        assert len(result) == 2
        assert {t.slug for t in result} == {"json-t1", "json-t2"}

    def test_jsonl_file(self, tmp_path: Path) -> None:
        """collect_tasks with a .jsonl file containing one task dict per line."""
        from hud.cli.utils.collect import collect_tasks

        lines = [
            json.dumps({"env": {"name": "e"}, "scenario": "e:s", "args": {}, "slug": "jl1"}),
            json.dumps({"env": {"name": "e"}, "scenario": "e:s", "args": {}, "slug": "jl2"}),
        ]
        (tmp_path / "tasks.jsonl").write_text("\n".join(lines))

        result = collect_tasks(str(tmp_path / "tasks.jsonl"))
        assert len(result) == 2
        assert {t.slug for t in result} == {"jl1", "jl2"}

    def test_unsupported_file_type_raises(self, tmp_path: Path) -> None:
        """collect_tasks with an unsupported extension raises ValueError."""
        from hud.cli.utils.collect import collect_tasks

        txt = tmp_path / "tasks.txt"
        txt.write_text("not a task file\n")

        with pytest.raises(ValueError, match="Unsupported file type"):
            collect_tasks(str(txt))

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """collect_tasks with a non-existent path raises FileNotFoundError."""
        from hud.cli.utils.collect import collect_tasks

        with pytest.raises(FileNotFoundError, match="Source not found"):
            collect_tasks(str(tmp_path / "does_not_exist"))

    def test_directory_dispatch(self, tmp_path: Path) -> None:
        """collect_tasks with a directory delegates to _collect_from_directory."""
        from hud.cli.utils.collect import collect_tasks

        _write_file(
            tmp_path / "scenario" / "task.py",
            """\
            from hud.eval.task import Task
            task = Task(env={"name": "e"}, scenario="e:s", args={}, slug="dir-dispatch")
            """,
        )

        result = collect_tasks(str(tmp_path))
        assert len(result) == 1
        assert result[0].slug == "dir-dispatch"
