"""Tests for the HUD → Harbor exporter."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from hud.cli.export import (
    HarborExporter,
    export,
    get_exporter,
    list_formats,
    write_result,
)
from hud.cli.export.base import ExportInput
from hud.cli.export.harbor import (
    _full_scenario_name,
    _render_keywords,
    _slugify,
    _toml_escape,
)
from hud.eval.task import Task


def _task(
    *,
    slug: str | None = None,
    scenario: str = "solve",
    env_name: str | None = "outline",
    args: dict | None = None,
    metadata: dict | None = None,
    system_prompt: str | None = None,
) -> Task:
    """Build a minimal Task without booting an Environment."""
    env_dict = {"name": env_name} if env_name else None
    return Task.model_construct(
        env=env_dict,
        scenario=scenario,
        slug=slug,
        args=args,
        metadata=metadata or {},
        agent_config={"system_prompt": system_prompt} if system_prompt else None,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class TestSlugify:
    def test_lowercases(self) -> None:
        assert _slugify("WrongFlag") == "wrongflag"

    def test_replaces_non_alnum(self) -> None:
        assert _slugify("fix bug #42") == "fix-bug-42"

    def test_collapses_runs(self) -> None:
        assert _slugify("a___b---c") == "a-b-c"

    def test_strips_edges(self) -> None:
        assert _slugify("--hello--") == "hello"

    def test_empty_falls_back(self) -> None:
        assert _slugify("###") == "task"


class TestFullScenarioName:
    def test_passthrough_if_qualified(self) -> None:
        t = _task(scenario="env:foo")
        assert _full_scenario_name(t, "anything") == "env:foo"

    def test_uses_env_dict_name(self) -> None:
        t = _task(scenario="checkout", env_name="browser")
        assert _full_scenario_name(t, "ts") == "browser:checkout"

    def test_falls_back_to_taskset(self) -> None:
        t = _task(scenario="solve", env_name=None)
        assert _full_scenario_name(t, "My Set") == "my-set:solve"

    def test_missing_scenario_raises(self) -> None:
        t = _task(scenario="", env_name="e")
        t.scenario = None  # type: ignore[assignment]
        with pytest.raises(ValueError):
            _full_scenario_name(t, "ts")


class TestRenderKeywords:
    def test_empty(self) -> None:
        assert _render_keywords([]) == "[]"

    def test_quotes_each(self) -> None:
        assert _render_keywords(["a", "b"]) == '["a", "b"]'

    def test_escapes(self) -> None:
        assert _render_keywords(['has"quote']) == '["has\\"quote"]'


class TestTomlEscape:
    def test_quote(self) -> None:
        assert _toml_escape('a"b') == 'a\\"b'

    def test_backslash(self) -> None:
        assert _toml_escape("a\\b") == "a\\\\b"

    def test_newline(self) -> None:
        assert _toml_escape("a\nb") == "a\\nb"


# --------------------------------------------------------------------------- #
# HarborExporter.export
# --------------------------------------------------------------------------- #


def _make_input(**overrides) -> ExportInput:
    base = {
        "tasks": [
            _task(
                slug="wrongflag",
                scenario="solve",
                env_name="outline",
                args={"problem_id": "wrongflag"},
                metadata={"difficulty": "easy", "keywords": ["typescript"]},
                system_prompt="You are an expert TS engineer.",
            ),
            _task(
                slug="redis-bug",
                scenario="solve",
                env_name="outline",
                args={"problem_id": "redis"},
                metadata={"difficulty": "hard"},
            ),
        ],
        "env_image": "ghcr.io/hud-evals/outline@sha256:deadbeef",
        "env_required_env": ["GITHUB_TOKEN"],
        "rendered_prompts": {
            "wrongflag": "Fix the emoji picker.",
            "redis-bug": "Fix the redis connection.",
        },
        "taskset_name": "outline-coding",
    }
    base.update(overrides)
    return ExportInput(**base)


class TestHarborExport:
    def test_emits_top_level_files(self) -> None:
        result = HarborExporter().export(_make_input())
        assert "README.md" in result.files
        assert "sample-run.sh" in result.files
        assert "manifest.json" in result.files

    def test_emits_per_task_files(self) -> None:
        result = HarborExporter().export(_make_input())
        for slug in ("wrongflag", "redis-bug"):
            for sub in (
                "instruction.md",
                "task.toml",
                "environment/Dockerfile",
                "environment/bake-setup.sh",
                "tests/test.sh",
            ):
                assert f"tasks/{slug}/{sub}" in result.files, f"missing tasks/{slug}/{sub}"

    def test_instruction_combines_system_and_user(self) -> None:
        result = HarborExporter().export(_make_input())
        instr = result.files["tasks/wrongflag/instruction.md"]
        assert isinstance(instr, str)
        assert "You are an expert TS engineer." in instr
        assert "---" in instr
        assert "Fix the emoji picker." in instr

    def test_instruction_user_only_when_no_system_prompt(self) -> None:
        result = HarborExporter().export(_make_input())
        instr = result.files["tasks/redis-bug/instruction.md"]
        assert isinstance(instr, str)
        assert "---" not in instr
        assert "Fix the redis connection." in instr

    def test_dockerfile_uses_base_image_and_bakes_setup(self) -> None:
        result = HarborExporter().export(_make_input())
        dockerfile = result.files["tasks/wrongflag/environment/Dockerfile"]
        assert isinstance(dockerfile, str)
        assert "FROM ghcr.io/hud-evals/outline@sha256:deadbeef" in dockerfile
        assert "ENV HUD_TASK_SCENARIO=outline:solve" in dockerfile
        assert '"problem_id": "wrongflag"' in dockerfile
        assert "/opt/hud/bake-setup.sh" in dockerfile

    def test_dockerfile_emits_platform_when_provided(self) -> None:
        result = HarborExporter().export(_make_input(env_platform="linux/amd64"))
        dockerfile = result.files["tasks/wrongflag/environment/Dockerfile"]
        assert isinstance(dockerfile, str)
        assert dockerfile.startswith(
            "FROM --platform=linux/amd64 ghcr.io/hud-evals/outline@sha256:deadbeef"
        )

    def test_dockerfile_omits_platform_when_unset(self) -> None:
        result = HarborExporter().export(_make_input())
        dockerfile = result.files["tasks/wrongflag/environment/Dockerfile"]
        assert isinstance(dockerfile, str)
        assert dockerfile.startswith("FROM ghcr.io/hud-evals/outline@sha256:deadbeef")
        assert "--platform" not in dockerfile

    def test_dockerfile_records_required_env(self) -> None:
        result = HarborExporter().export(_make_input())
        dockerfile = result.files["tasks/wrongflag/environment/Dockerfile"]
        assert isinstance(dockerfile, str)
        assert "GITHUB_TOKEN" in dockerfile

    def test_test_sh_calls_hud_scenario_grade(self) -> None:
        result = HarborExporter().export(_make_input())
        test_sh = result.files["tasks/wrongflag/tests/test.sh"]
        assert isinstance(test_sh, str)
        assert "hud scenario grade" in test_sh
        assert "reward.txt" in test_sh

    def test_task_toml_contains_metadata(self) -> None:
        result = HarborExporter().export(_make_input())
        task_toml = result.files["tasks/wrongflag/task.toml"]
        assert isinstance(task_toml, str)
        assert 'name = "outline-coding/wrongflag"' in task_toml
        assert 'difficulty = "easy"' in task_toml
        assert '"typescript"' in task_toml

    def test_manifest_lists_tasks(self) -> None:
        result = HarborExporter().export(_make_input())
        manifest = json.loads(result.files["manifest.json"])
        assert manifest["format"] == "harbor"
        assert manifest["task_count"] == 2
        assert manifest["task_slugs"] == ["wrongflag", "redis-bug"]
        assert manifest["base_image"] == "ghcr.io/hud-evals/outline@sha256:deadbeef"

    def test_subset_filter(self) -> None:
        result = HarborExporter().export(_make_input(task_subset=["wrongflag"]))
        assert "tasks/wrongflag/task.toml" in result.files
        assert "tasks/redis-bug/task.toml" not in result.files

    def test_no_tasks_raises(self) -> None:
        inp = _make_input(task_subset=["does-not-exist"])
        with pytest.raises(ValueError, match="No tasks selected"):
            HarborExporter().export(inp)

    def test_dedupes_collisions(self) -> None:
        inp = _make_input(
            tasks=[
                _task(slug="dup", scenario="a"),
                _task(slug="dup", scenario="b"),
            ],
            rendered_prompts={},
        )
        result = HarborExporter().export(inp)
        assert "tasks/dup/task.toml" in result.files
        assert "tasks/dup-2/task.toml" in result.files

    def test_placeholder_when_prompt_missing(self) -> None:
        inp = _make_input(rendered_prompts={})
        result = HarborExporter().export(inp)
        instr = result.files["tasks/wrongflag/instruction.md"]
        assert isinstance(instr, str)
        assert "not rendered" in instr

    def test_sample_run_targets_first_task(self) -> None:
        result = HarborExporter().export(_make_input())
        assert "wrongflag" in result.sample_run_script
        assert "__FIRST_TASK__" not in result.sample_run_script


# --------------------------------------------------------------------------- #
# write_result + registry
# --------------------------------------------------------------------------- #


class TestWriteResult:
    def test_materializes_files(self, tmp_path: Path) -> None:
        result = HarborExporter().export(_make_input())
        out = write_result(result, tmp_path / "out")
        assert (out / "tasks" / "wrongflag" / "task.toml").exists()
        assert (out / "tasks" / "redis-bug" / "tests" / "test.sh").exists()
        assert (out / "manifest.json").exists()

    def test_marks_shell_scripts_executable(self, tmp_path: Path) -> None:
        result = HarborExporter().export(_make_input())
        out = write_result(result, tmp_path / "out")
        sample = out / "sample-run.sh"
        assert sample.stat().st_mode & stat.S_IXUSR
        test_sh = out / "tasks" / "wrongflag" / "tests" / "test.sh"
        assert test_sh.stat().st_mode & stat.S_IXUSR


class TestRegistry:
    def test_lists_formats(self) -> None:
        names = {n for n, _ in list_formats()}
        assert "harbor" in names

    def test_get_exporter(self) -> None:
        assert isinstance(get_exporter("harbor"), HarborExporter)
        assert get_exporter("nope") is None

    def test_export_dispatches_by_name(self) -> None:
        result = export("harbor", _make_input())
        assert "manifest.json" in result.files

    def test_export_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown export format"):
            export("nope", _make_input())
