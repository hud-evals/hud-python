"""Tests for the Harbor → HUD converter.

Exercises HarborConverter.detect(), HarborConverter.convert(), and the
write_result() writer using synthetic terminal-bench-style fixtures
defined in conftest.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hud.cli.convert import detect_format, get_converter, list_formats, write_result
from hud.cli.convert.base import ConvertResult, GeneratedEnvironment
from hud.cli.convert.harbor import (
    HarborConverter,
    _adapt_harbor_dockerfile,
    _find_dockerfile,
    _hash_directory,
    _is_harbor_task,
    _normalize_name,
    _parse_task,
)

from .conftest import make_harbor_task


# ============================================================================
# Helper unit tests
# ============================================================================


class TestNormalizeName:
    def test_simple(self) -> None:
        assert _normalize_name("terminal-bench") == "terminal-bench"

    def test_underscores(self) -> None:
        assert _normalize_name("my_cool_bench") == "my-cool-bench"

    def test_spaces(self) -> None:
        assert _normalize_name("My Cool Bench") == "my-cool-bench"

    def test_special_chars(self) -> None:
        assert _normalize_name("bench@2.0!") == "bench20"

    def test_empty(self) -> None:
        assert _normalize_name("") == "converted"

    def test_only_special_chars(self) -> None:
        assert _normalize_name("@#$") == "converted"

    def test_leading_trailing_dashes(self) -> None:
        assert _normalize_name("--hello--") == "hello"

    def test_consecutive_dashes(self) -> None:
        assert _normalize_name("a---b") == "a-b"


class TestAdaptDockerfile:
    def test_comments_cmd(self) -> None:
        result = _adapt_harbor_dockerfile('CMD ["bash"]')
        assert result == '# [harbor original] CMD ["bash"]'

    def test_comments_entrypoint(self) -> None:
        result = _adapt_harbor_dockerfile('ENTRYPOINT ["/bin/bash"]')
        assert result == '# [harbor original] ENTRYPOINT ["/bin/bash"]'

    def test_preserves_other_lines(self) -> None:
        dockerfile = "FROM python:3.11\nRUN echo hi\nCMD bash"
        result = _adapt_harbor_dockerfile(dockerfile)
        lines = result.splitlines()
        assert lines[0] == "FROM python:3.11"
        assert lines[1] == "RUN echo hi"
        assert lines[2] == "# [harbor original] CMD bash"

    def test_case_insensitive_match(self) -> None:
        # The implementation uses .upper() so indented CMD should match
        result = _adapt_harbor_dockerfile("  CMD bash")
        assert result == "# [harbor original]   CMD bash"

    def test_no_cmd_or_entrypoint(self) -> None:
        dockerfile = "FROM python:3.11\nRUN apt-get update"
        assert _adapt_harbor_dockerfile(dockerfile) == dockerfile


class TestHashDirectory:
    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        (dir_a / "file.txt").write_text("hello")

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        (dir_b / "file.txt").write_text("hello")

        assert _hash_directory(dir_a) == _hash_directory(dir_b)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        (dir_a / "file.txt").write_text("hello")

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        (dir_b / "file.txt").write_text("world")

        assert _hash_directory(dir_a) != _hash_directory(dir_b)

    def test_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        assert _hash_directory(tmp_path / "nonexistent") == "empty"

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        # Empty dir has a deterministic hash (sha256 of nothing)
        result = _hash_directory(empty)
        assert isinstance(result, str)
        assert len(result) == 16


class TestFindDockerfile:
    def test_finds_dockerfile(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text("FROM python:3.11")
        assert _find_dockerfile(tmp_path) == "FROM python:3.11"

    def test_finds_lowercase(self, tmp_path: Path) -> None:
        (tmp_path / "dockerfile").write_text("FROM alpine")
        assert _find_dockerfile(tmp_path) == "FROM alpine"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert _find_dockerfile(tmp_path) is None


class TestIsHarborTask:
    def test_valid_task(self, single_task: Path) -> None:
        assert _is_harbor_task(single_task) is True

    def test_missing_instruction(self, tmp_path: Path) -> None:
        task = tmp_path / "bad-task"
        task.mkdir()
        (task / "task.toml").write_text("[metadata]\n")
        assert _is_harbor_task(task) is False

    def test_missing_task_toml(self, tmp_path: Path) -> None:
        task = tmp_path / "bad-task"
        task.mkdir()
        (task / "instruction.md").write_text("# Do something")
        assert _is_harbor_task(task) is False

    def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        assert _is_harbor_task(f) is False


class TestParseTask:
    def test_parses_valid_task(self, single_task: Path) -> None:
        result = _parse_task(single_task)
        assert result is not None
        assert result.task_id == "cancel-async-tasks"
        assert "Cancel Async Tasks" in result.instruction
        assert result.config.get("metadata", {}).get("category") == "systems"

    def test_parses_verifier_timeout(self, single_task: Path) -> None:
        result = _parse_task(single_task)
        assert result is not None
        assert result.config["verifier"]["timeout_sec"] == 120

    def test_returns_none_for_bad_instruction(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "bad"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("[metadata]\n")
        # instruction.md missing
        assert _parse_task(task_dir) is None

    def test_handles_bad_toml_gracefully(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "broken-toml"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("# Hello")
        (task_dir / "task.toml").write_text("this is not valid toml {{{")
        result = _parse_task(task_dir)
        assert result is not None
        # Config should be empty dict when toml fails
        assert result.config == {}


# ============================================================================
# HarborConverter.detect()
# ============================================================================


class TestHarborConverterDetect:
    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_detects_single_task(self, single_task: Path) -> None:
        assert self.converter.detect(single_task) is True

    def test_detects_dataset(self, dataset_same_env: Path) -> None:
        assert self.converter.detect(dataset_same_env) is True

    def test_rejects_empty_dir(self, tmp_path: Path) -> None:
        assert self.converter.detect(tmp_path) is False

    def test_rejects_non_harbor_dir(self, tmp_path: Path) -> None:
        (tmp_path / "random.txt").write_text("nope")
        assert self.converter.detect(tmp_path) is False


# ============================================================================
# HarborConverter.convert()
# ============================================================================


class TestHarborConverterConvertSingleTask:
    """Convert a single Harbor task directory."""

    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_single_task_produces_one_env(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        assert len(result.environments) == 1
        assert len(result.taskset) == 1

    def test_env_name_uses_parent_dir(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env = result.environments[0]
        # Parent dir name is the tmp_path random name, but it gets normalized
        assert env.name.startswith("hud-harbor-")

    def test_env_py_contains_scenario(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env_py = result.environments[0].env_py
        assert "@env.scenario" in env_py
        assert "run-task" in env_py

    def test_env_py_has_correct_timeout(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env_py = result.environments[0].env_py
        assert "timeout=120" in env_py

    def test_taskset_references_env(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        entry = result.taskset[0]
        env_name = result.environments[0].name
        assert entry["scenario"] == f"{env_name}:run-task"
        assert entry["args"]["task_id"] == "cancel-async-tasks"

    def test_task_dirs_map(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env = result.environments[0]
        assert "cancel-async-tasks" in env.task_dirs
        assert env.task_dirs["cancel-async-tasks"] == single_task

    def test_summary_not_empty(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        assert len(result.summary) > 0
        assert any("1" in line for line in result.summary)


class TestHarborConverterConvertDataset:
    """Convert a dataset directory with multiple tasks sharing the same env."""

    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_same_env_groups_into_one(self, dataset_same_env: Path) -> None:
        result = self.converter.convert(dataset_same_env)
        assert len(result.environments) == 1
        assert len(result.taskset) == 3

    def test_all_task_ids_present(self, dataset_same_env: Path) -> None:
        result = self.converter.convert(dataset_same_env)
        task_ids = {e["args"]["task_id"] for e in result.taskset}
        assert task_ids == {"cancel-async-tasks", "build-pmars", "chess-best-move"}

    def test_env_name_from_dataset(self, dataset_same_env: Path) -> None:
        result = self.converter.convert(dataset_same_env)
        env = result.environments[0]
        assert env.name == "hud-harbor-terminal-bench-sample"


class TestHarborConverterConvertMultiEnv:
    """Convert a dataset with tasks split across different Dockerfiles."""

    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_creates_two_envs(self, dataset_multi_env: Path) -> None:
        result = self.converter.convert(dataset_multi_env)
        assert len(result.environments) == 2
        assert len(result.taskset) == 4

    def test_env_names_have_group_suffix(self, dataset_multi_env: Path) -> None:
        result = self.converter.convert(dataset_multi_env)
        names = {e.name for e in result.environments}
        assert all(n.startswith("hud-harbor-mixed-bench") for n in names)
        # With multiple groups, names should have -g1, -g2 suffixes
        assert any("-g1" in n for n in names)
        assert any("-g2" in n for n in names)

    def test_each_env_has_correct_tasks(self, dataset_multi_env: Path) -> None:
        result = self.converter.convert(dataset_multi_env)
        for env in result.environments:
            task_ids = set(env.task_dirs.keys())
            # Each group should have exactly 2 tasks
            assert len(task_ids) == 2

    def test_ml_env_has_nvidia_dockerfile(self, dataset_multi_env: Path) -> None:
        result = self.converter.convert(dataset_multi_env)
        # One of the environments should reference nvidia in its dockerfile
        dockerfiles = [e.dockerfile for e in result.environments]
        assert any("nvidia" in d for d in dockerfiles)

    def test_simple_env_has_python_dockerfile(self, dataset_multi_env: Path) -> None:
        result = self.converter.convert(dataset_multi_env)
        dockerfiles = [e.dockerfile for e in result.environments]
        assert any("python:3.11-slim" in d for d in dockerfiles)


class TestBuildContextSource:
    """Verify build_context_source is set for tasks with environment/ dirs."""

    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_build_context_source_set(self, task_with_build_context: Path) -> None:
        result = self.converter.convert(task_with_build_context)
        env = result.environments[0]
        assert env.build_context_source is not None
        assert env.build_context_source.is_dir()

    def test_build_context_source_none_when_no_env_dir(
        self, dataset_no_dockerfile: Path
    ) -> None:
        result = self.converter.convert(dataset_no_dockerfile)
        env = result.environments[0]
        assert env.build_context_source is None


class TestWriteBuildContext:
    """Verify that build context files from environment/ are copied to env root."""

    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_warriors_copied_to_env_root(
        self, task_with_build_context: Path, tmp_path: Path
    ) -> None:
        result = self.converter.convert(task_with_build_context)
        out = tmp_path / "output"
        write_result(result, out)

        env = result.environments[0]
        env_dir = out / env.name

        # warriors/ dir should exist at env root (Docker build context)
        assert (env_dir / "warriors").is_dir()
        assert (env_dir / "warriors" / "flashpaper.red").is_file()
        assert (env_dir / "warriors" / "rave.red").is_file()

    def test_dockerfile_not_duplicated(
        self, task_with_build_context: Path, tmp_path: Path
    ) -> None:
        result = self.converter.convert(task_with_build_context)
        out = tmp_path / "output"
        write_result(result, out)

        env = result.environments[0]
        env_dir = out / env.name

        # Should have Dockerfile.hud (generated), NOT a raw Dockerfile copy
        assert (env_dir / "Dockerfile.hud").is_file()
        assert not (env_dir / "Dockerfile").exists()

    def test_build_context_content_correct(
        self, task_with_build_context: Path, tmp_path: Path
    ) -> None:
        result = self.converter.convert(task_with_build_context)
        out = tmp_path / "output"
        write_result(result, out)

        env = result.environments[0]
        content = (out / env.name / "warriors" / "flashpaper.red").read_text(
            encoding="utf-8"
        )
        assert "MOV 0, 1" in content


class TestHarborConverterConvertNoDockerfile:
    """Tasks without environment/Dockerfile should use fallback."""

    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_fallback_dockerfile(self, dataset_no_dockerfile: Path) -> None:
        result = self.converter.convert(dataset_no_dockerfile)
        assert len(result.environments) == 1
        # Fallback dockerfile starts with FROM python:3.11-slim
        assert "FROM python:3.11-slim" in result.environments[0].dockerfile

    def test_no_harbor_original_comments(self, dataset_no_dockerfile: Path) -> None:
        result = self.converter.convert(dataset_no_dockerfile)
        # Fallback dockerfile should NOT have commented-out lines
        assert "# [harbor original]" not in result.environments[0].dockerfile


class TestHarborConverterConvertWithSolutions:
    """Verify that solution/ dirs show up in task_dirs but write_result skips them."""

    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_solutions_present_in_source(self, dataset_with_solutions: Path) -> None:
        # Verify the fixture has solution dirs
        for name in ("task-x", "task-y"):
            assert (dataset_with_solutions / name / "solution").is_dir()

    def test_convert_succeeds(self, dataset_with_solutions: Path) -> None:
        result = self.converter.convert(dataset_with_solutions)
        assert len(result.environments) == 1
        assert len(result.taskset) == 2


class TestHarborConverterEdgeCases:
    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_no_tasks_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty-dataset"
        empty.mkdir()
        with pytest.raises(ValueError, match="No Harbor tasks found"):
            self.converter.convert(empty)

    def test_all_tasks_fail_raises(self, tmp_path: Path) -> None:
        dataset = tmp_path / "bad-dataset"
        dataset.mkdir()
        # Create subdirs that look like tasks but have no instruction.md
        for name in ("a", "b"):
            d = dataset / name
            d.mkdir()
            (d / "task.toml").write_text("[metadata]\n")
            # Missing instruction.md -> will fail detect, so not even found as task
        with pytest.raises(ValueError, match="No Harbor tasks found"):
            self.converter.convert(dataset)

    def test_partial_failure_skips_bad_tasks(self, tmp_path: Path) -> None:
        dataset = tmp_path / "partial"
        dataset.mkdir()

        # One good task
        make_harbor_task(dataset, "good-task")

        # One bad task (has task.toml + instruction.md but instruction unreadable)
        bad = dataset / "bad-task"
        bad.mkdir()
        (bad / "task.toml").write_text("[metadata]\n")
        (bad / "instruction.md").write_text("# OK")  # actually valid

        result = self.converter.convert(dataset)
        # Both should parse, so 2 tasks
        assert len(result.taskset) == 2


# ============================================================================
# Taskset metadata
# ============================================================================


class TestTasksetMetadata:
    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_metadata_includes_harbor_source(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        entry = result.taskset[0]
        assert "harbor_source" in entry["metadata"]

    def test_metadata_includes_toml_metadata(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        entry = result.taskset[0]
        meta = entry["metadata"]
        assert meta.get("category") == "systems"
        assert meta.get("difficulty") == "medium"


# ============================================================================
# Dockerfile generation
# ============================================================================


class TestDockerfileGeneration:
    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_cmd_commented_out(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        dockerfile = result.environments[0].dockerfile
        # Original CMD ["bash"] should be commented out
        assert "# [harbor original]" in dockerfile

    def test_hud_layer_present(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        dockerfile = result.environments[0].dockerfile
        assert "COPY env.py" in dockerfile
        assert "uv" in dockerfile
        assert "hud" in dockerfile

    def test_tasks_copied_into_image(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        dockerfile = result.environments[0].dockerfile
        assert "COPY tasks/ /harbor/tasks/" in dockerfile

    def test_logs_dir_created(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        dockerfile = result.environments[0].dockerfile
        assert "/logs/verifier" in dockerfile


# ============================================================================
# env.py generation
# ============================================================================


class TestEnvPyGeneration:
    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_imports_present(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env_py = result.environments[0].env_py
        assert "from hud import Environment" in env_py
        assert "from hud.tools import BashTool" in env_py

    def test_tools_added(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env_py = result.environments[0].env_py
        assert "env.add_tool(BashTool())" in env_py
        assert "env.add_tool(EditTool())" in env_py

    def test_reward_parsing_logic(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env_py = result.environments[0].env_py
        assert "_parse_harbor_reward" in env_py
        assert "reward.txt" in env_py
        assert "reward.json" in env_py


# ============================================================================
# Scenario signature: single-task default vs multi-task Literal
# ============================================================================


class TestScenarioSignature:
    """Verify that single-task envs get a default and multi-task envs get a Literal."""

    def setup_method(self) -> None:
        self.converter = HarborConverter()

    # --- single task: optional with default ---

    def test_single_task_has_default(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env_py = result.environments[0].env_py
        assert 'task_id: str = "cancel-async-tasks"' in env_py

    def test_single_task_no_literal_import(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env_py = result.environments[0].env_py
        assert "from typing import Literal" not in env_py
        assert "TaskId" not in env_py

    # --- multi-task (same env): Literal type ---

    def test_multi_task_has_literal(self, dataset_same_env: Path) -> None:
        result = self.converter.convert(dataset_same_env)
        env_py = result.environments[0].env_py
        assert "from typing import Literal" in env_py
        assert "TaskId = Literal[" in env_py

    def test_multi_task_literal_lists_all_ids(self, dataset_same_env: Path) -> None:
        result = self.converter.convert(dataset_same_env)
        env_py = result.environments[0].env_py
        for name in ("cancel-async-tasks", "build-pmars", "chess-best-move"):
            assert f'"{name}"' in env_py

    def test_multi_task_signature_uses_literal(self, dataset_same_env: Path) -> None:
        result = self.converter.convert(dataset_same_env)
        env_py = result.environments[0].env_py
        assert "def run_task(task_id: TaskId):" in env_py

    def test_multi_task_no_default(self, dataset_same_env: Path) -> None:
        result = self.converter.convert(dataset_same_env)
        env_py = result.environments[0].env_py
        # Should NOT have a default value
        assert "task_id: TaskId):" in env_py
        assert "= " not in env_py.split("def run_task(")[1].split("):")[0]

    # --- multi-env dataset: each env gets the right variant ---

    def test_multi_env_single_task_per_env(self, dataset_multi_env: Path) -> None:
        result = self.converter.convert(dataset_multi_env)
        # Each env has 2 tasks, so all should use Literal
        for env in result.environments:
            assert "TaskId = Literal[" in env.env_py

    def test_single_task_build_context_fixture(
        self, task_with_build_context: Path
    ) -> None:
        result = self.converter.convert(task_with_build_context)
        env_py = result.environments[0].env_py
        assert 'task_id: str = "build-pmars"' in env_py


# ============================================================================
# pyproject.toml generation
# ============================================================================


class TestPyprojectGeneration:
    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_has_hud_dependency(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        pyproject = result.environments[0].pyproject_toml
        assert "hud-python" in pyproject

    def test_name_matches_env(self, single_task: Path) -> None:
        result = self.converter.convert(single_task)
        env = result.environments[0]
        assert env.name in env.pyproject_toml


# ============================================================================
# write_result()
# ============================================================================


class TestWriteResult:
    def setup_method(self) -> None:
        self.converter = HarborConverter()

    def test_creates_directory_structure(
        self, single_task: Path, tmp_path: Path
    ) -> None:
        result = self.converter.convert(single_task)
        out = tmp_path / "output"
        write_result(result, out)

        env = result.environments[0]
        env_dir = out / env.name

        assert env_dir.is_dir()
        assert (env_dir / "env.py").is_file()
        assert (env_dir / "Dockerfile.hud").is_file()
        assert (env_dir / "pyproject.toml").is_file()
        assert (env_dir / "tasks").is_dir()
        assert (out / "taskset.json").is_file()

    def test_taskset_json_valid(self, single_task: Path, tmp_path: Path) -> None:
        result = self.converter.convert(single_task)
        out = tmp_path / "output"
        taskset_path = write_result(result, out)

        with open(taskset_path, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["args"]["task_id"] == "cancel-async-tasks"

    def test_task_files_copied(self, single_task: Path, tmp_path: Path) -> None:
        result = self.converter.convert(single_task)
        out = tmp_path / "output"
        write_result(result, out)

        env = result.environments[0]
        task_out = out / env.name / "tasks" / "cancel-async-tasks"

        assert (task_out / "instruction.md").is_file()
        assert (task_out / "task.toml").is_file()
        assert (task_out / "tests" / "test.sh").is_file()

    def test_environment_dir_not_copied(
        self, single_task: Path, tmp_path: Path
    ) -> None:
        result = self.converter.convert(single_task)
        out = tmp_path / "output"
        write_result(result, out)

        env = result.environments[0]
        task_out = out / env.name / "tasks" / "cancel-async-tasks"

        # environment/ should be excluded from the copy
        assert not (task_out / "environment").exists()

    def test_solution_dir_not_copied(
        self, dataset_with_solutions: Path, tmp_path: Path
    ) -> None:
        result = self.converter.convert(dataset_with_solutions)
        out = tmp_path / "output"
        write_result(result, out)

        env = result.environments[0]
        for task_id in env.task_dirs:
            task_out = out / env.name / "tasks" / task_id
            assert not (task_out / "solution").exists()

    def test_multi_env_write(self, dataset_multi_env: Path, tmp_path: Path) -> None:
        result = self.converter.convert(dataset_multi_env)
        out = tmp_path / "output"
        write_result(result, out)

        # Both environments should be written
        for env in result.environments:
            assert (out / env.name).is_dir()
            assert (out / env.name / "env.py").is_file()

        # Single taskset.json with all tasks
        with open(out / "taskset.json", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 4

    def test_overwrites_existing(self, single_task: Path, tmp_path: Path) -> None:
        result = self.converter.convert(single_task)
        out = tmp_path / "output"

        # Write twice — should not error
        write_result(result, out)
        write_result(result, out)

        assert (out / "taskset.json").is_file()


# ============================================================================
# Registry integration (detect_format, get_converter, list_formats)
# ============================================================================


class TestConverterRegistry:
    def test_get_converter_by_name(self) -> None:
        converter = get_converter("harbor")
        assert converter is not None
        assert isinstance(converter, HarborConverter)

    def test_get_converter_unknown(self) -> None:
        assert get_converter("nonexistent") is None

    def test_detect_format_harbor(self, single_task: Path) -> None:
        converter = detect_format(single_task)
        assert converter is not None
        assert converter.name == "harbor"

    def test_detect_format_unknown(self, tmp_path: Path) -> None:
        assert detect_format(tmp_path) is None

    def test_list_formats_includes_harbor(self) -> None:
        formats = list_formats()
        names = [name for name, _desc in formats]
        assert "harbor" in names
