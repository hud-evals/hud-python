"""Integration tests for ``hud sync`` — tasks, env, and config.

Each test corresponds to a real scenario from the state change matrix:
- E* = environment changes
- T* = taskset changes
- L* = local task changes
- R* = remote task changes
- X* = cross-cutting scenarios

Tests use physical tmp directories with real .hud/config.json files,
real .py task files, and mocked HTTP responses.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import click.exceptions
import httpx
import pytest

# ---------------------------------------------------------------------------
# Fixtures: mock task files, configs, and API responses
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """A temporary project directory with a basic env.py and task file."""
    env_py = tmp_path / "env.py"
    env_py.write_text(
        textwrap.dedent("""\
        from hud.environment import Environment
        env = Environment("test-env")
    """)
    )

    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text(
        textwrap.dedent("""\
        from hud.eval.task import Task
        task_one = Task(
            env={"name": "test-env"},
            scenario="test-env:greet",
            args={"name": "alice"},
            slug="greet-alice",
        )
        task_two = Task(
            env={"name": "test-env"},
            scenario="test-env:greet",
            args={"name": "bob"},
            slug="greet-bob",
        )
    """)
    )
    return tmp_path


@pytest.fixture()
def project_with_config(project_dir: Path) -> Path:
    """Project dir with .hud/config.json already set up."""
    hud_dir = project_dir / ".hud"
    hud_dir.mkdir()
    config = {"registryId": "reg-111-222", "tasksetId": "ts-333-444"}
    (hud_dir / "config.json").write_text(json.dumps(config))
    return project_dir


@pytest.fixture()
def project_with_legacy_deploy(project_dir: Path) -> Path:
    """Project dir with legacy .hud/deploy.json (for migration test)."""
    hud_dir = project_dir / ".hud"
    hud_dir.mkdir()
    legacy = {"registryId": "legacy-reg-id", "version": 3, "syncEnv": True}
    (hud_dir / "deploy.json").write_text(json.dumps(legacy))
    return project_dir


@pytest.fixture()
def project_multi_env(tmp_path: Path) -> Path:
    """Project with tasks referencing multiple environments."""
    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text(
        textwrap.dedent("""\
        from hud.eval.task import Task
        task_a = Task(
            env={"name": "env-alpha"},
            scenario="env-alpha:setup",
            args={"mode": "fast"},
            slug="alpha-fast",
        )
        task_b = Task(
            env={"name": "env-beta"},
            scenario="env-beta:train",
            args={"epochs": 10},
            slug="beta-train",
        )
    """)
    )
    return tmp_path


@pytest.fixture()
def project_no_slugs(tmp_path: Path) -> Path:
    """Project with tasks that are missing slugs."""
    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text(
        textwrap.dedent("""\
        from hud.eval.task import Task
        task_one = Task(
            env={"name": "test-env"},
            scenario="test-env:greet",
            args={"name": "alice"},
        )
    """)
    )
    return tmp_path


@pytest.fixture()
def project_duplicate_slugs(tmp_path: Path) -> Path:
    """Project with duplicate slugs."""
    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text(
        textwrap.dedent("""\
        from hud.eval.task import Task
        task_one = Task(
            env={"name": "e"}, scenario="e:s", args={"x": 1}, slug="dupe",
        )
        task_two = Task(
            env={"name": "e"}, scenario="e:s", args={"x": 2}, slug="dupe",
        )
    """)
    )
    return tmp_path


@pytest.fixture()
def project_renamed_slug(tmp_path: Path) -> Path:
    """Project where a task slug was renamed from old-name to new-name."""
    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text(
        textwrap.dedent("""\
        from hud.eval.task import Task
        task = Task(
            env={"name": "test-env"},
            scenario="test-env:greet",
            args={"name": "alice"},
            slug="new-name",
        )
    """)
    )
    return tmp_path


@pytest.fixture()
def project_with_validation(tmp_path: Path) -> Path:
    """Project with tasks that have validation sequences."""
    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text(
        textwrap.dedent("""\
        from hud.eval.task import Task
        from hud.types import MCPToolCall
        task = Task(
            env={"name": "test-env"},
            scenario="test-env:fix",
            args={"repo": "sample"},
            slug="fix-basic",
            validation=[
                MCPToolCall(name="bash", arguments={"command": "echo ok"}),
            ],
        )
    """)
    )
    return tmp_path


@pytest.fixture()
def project_with_agent_config(tmp_path: Path) -> Path:
    """Project with tasks that have agent_config."""
    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text(
        textwrap.dedent("""\
        from hud.eval.task import Task
        task = Task(
            env={"name": "test-env"},
            scenario="test-env:assist",
            args={},
            slug="assist-v1",
            agent_config={"system_prompt": "Be concise"},
        )
    """)
    )
    return tmp_path


def _mock_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ===========================================================================
# Config tests (.hud/config.json)
# ===========================================================================


class TestProjectConfig:
    def test_load_empty_dir(self, tmp_path: Path) -> None:
        from hud.cli.utils.project_config import load_project_config

        assert load_project_config(tmp_path) == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        from hud.cli.utils.project_config import load_project_config, save_project_config

        save_project_config({"registryId": "abc-123"}, tmp_path)
        assert load_project_config(tmp_path)["registryId"] == "abc-123"

    def test_save_merges_existing(self, project_with_config: Path) -> None:
        from hud.cli.utils.project_config import load_project_config, save_project_config

        save_project_config({"tasksetId": "new-ts-id"}, project_with_config)
        config = load_project_config(project_with_config)
        assert config["registryId"] == "reg-111-222"
        assert config["tasksetId"] == "new-ts-id"

    def test_migrate_legacy_deploy_json(self, project_with_legacy_deploy: Path) -> None:
        from hud.cli.utils.project_config import load_project_config

        config = load_project_config(project_with_legacy_deploy)
        assert config["registryId"] == "legacy-reg-id"
        assert config.get("syncEnv") is True

        hud_dir = project_with_legacy_deploy / ".hud"
        assert (hud_dir / "config.json").exists()
        assert not (hud_dir / "deploy.json").exists()

    def test_ids_only_no_names_stored(self, tmp_path: Path) -> None:
        from hud.cli.utils.project_config import save_project_config

        save_project_config({"registryId": "abc", "tasksetId": "def"}, tmp_path)
        raw = json.loads((tmp_path / ".hud" / "config.json").read_text())
        assert "environmentName" not in raw
        assert "tasksetName" not in raw

    def test_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        hud_dir = tmp_path / ".hud"
        hud_dir.mkdir()
        (hud_dir / "config.json").write_text("NOT VALID JSON {{{")

        from hud.cli.utils.project_config import load_project_config

        assert load_project_config(tmp_path) == {}

    def test_get_registry_id_helper(self, project_with_config: Path) -> None:
        from hud.cli.utils.project_config import get_registry_id

        assert get_registry_id(project_with_config) == "reg-111-222"

    def test_get_taskset_id_helper(self, project_with_config: Path) -> None:
        from hud.cli.utils.project_config import get_taskset_id

        assert get_taskset_id(project_with_config) == "ts-333-444"

    def test_get_registry_id_missing(self, tmp_path: Path) -> None:
        from hud.cli.utils.project_config import get_registry_id

        assert get_registry_id(tmp_path) is None


# ===========================================================================
# Task collection tests
# ===========================================================================


class TestCollectTasks:
    def test_collect_from_py_file(self, project_dir: Path) -> None:
        from hud.cli.utils.collect import collect_tasks
        from hud.eval.task import Task

        tasks = collect_tasks(str(project_dir / "tasks.py"))
        assert len(tasks) == 2
        assert all(isinstance(t, Task) for t in tasks)
        assert {t.slug for t in tasks} == {"greet-alice", "greet-bob"}

    def test_collect_from_directory(self, project_dir: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        assert len(collect_tasks(str(project_dir))) == 2

    def test_collect_from_json(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        (tmp_path / "t.json").write_text(
            json.dumps(
                [
                    {"env": {"name": "e"}, "scenario": "e:s1", "args": {"x": 1}},
                    {"env": {"name": "e"}, "scenario": "e:s2", "args": {"y": 2}},
                ]
            )
        )
        assert len(collect_tasks(str(tmp_path / "t.json"))) == 2

    def test_collect_from_jsonl(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        (tmp_path / "t.jsonl").write_text(
            json.dumps({"env": {"name": "e"}, "scenario": "e:s", "args": {}})
            + "\n"
            + json.dumps({"env": {"name": "e"}, "scenario": "e:s2", "args": {}})
            + "\n"
        )
        assert len(collect_tasks(str(tmp_path / "t.jsonl"))) == 2

    def test_collect_sdlc_subdirectory_pattern(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "tasks" / "checkout"
        task_dir.mkdir(parents=True)
        (task_dir / "__init__.py").write_text("")
        (task_dir / "task.py").write_text(
            textwrap.dedent("""\
            from hud.eval.task import Task
            task = Task(env={"name": "shop"}, scenario="shop:checkout",
                        args={"item": "laptop"}, slug="checkout-laptop")
        """)
        )

        from hud.cli.utils.collect import collect_tasks

        tasks = collect_tasks(str(tmp_path / "tasks"))
        assert len(tasks) == 1
        assert tasks[0].slug == "checkout-laptop"

    def test_collect_tasks_list_attribute(self, tmp_path: Path) -> None:
        (tmp_path / "my_tasks.py").write_text(
            textwrap.dedent("""\
            from hud.eval.task import Task
            tasks = [
                Task(env={"name": "e"}, scenario="e:s1", args={"a": 1}, slug="s1"),
                Task(env={"name": "e"}, scenario="e:s2", args={"a": 2}, slug="s2"),
            ]
        """)
        )
        from hud.cli.utils.collect import collect_tasks

        assert len(collect_tasks(str(tmp_path / "my_tasks.py"))) == 2

    def test_collect_tasks_dict_attribute(self, tmp_path: Path) -> None:
        (tmp_path / "my_tasks.py").write_text(
            textwrap.dedent("""\
            from hud.eval.task import Task
            tasks = {
                "first": Task(env={"name": "e"}, scenario="e:s1", args={}, slug="s1"),
                "second": Task(env={"name": "e"}, scenario="e:s2", args={}, slug="s2"),
            }
        """)
        )
        from hud.cli.utils.collect import collect_tasks

        assert len(collect_tasks(str(tmp_path / "my_tasks.py"))) == 2

    def test_collect_empty_dir(self, tmp_path: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        assert collect_tasks(str(tmp_path)) == []

    def test_collect_import_error(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("import nonexistent_xyz_module\n")

        from hud.cli.utils.collect import collect_tasks

        with pytest.raises(ImportError, match="nonexistent_xyz_module"):
            collect_tasks(str(tmp_path / "broken.py"))

    def test_collect_syntax_error(self, tmp_path: Path) -> None:
        (tmp_path / "bad_syntax.py").write_text("def foo(\n")

        from hud.cli.utils.collect import collect_tasks

        with pytest.raises(ImportError):
            collect_tasks(str(tmp_path / "bad_syntax.py"))

    def test_collect_no_tasks_in_module(self, tmp_path: Path) -> None:
        (tmp_path / "no_tasks.py").write_text("x = 42\n")

        from hud.cli.utils.collect import collect_tasks

        assert collect_tasks(str(tmp_path / "no_tasks.py")) == []

    def test_collect_nonexistent_path(self) -> None:
        from hud.cli.utils.collect import collect_tasks

        with pytest.raises(FileNotFoundError):
            collect_tasks("/nonexistent/tasks.py")

    def test_collect_unsupported_extension(self, tmp_path: Path) -> None:
        (tmp_path / "tasks.yaml").write_text("tasks: []")

        from hud.cli.utils.collect import collect_tasks

        with pytest.raises(ValueError, match="Unsupported file type"):
            collect_tasks(str(tmp_path / "tasks.yaml"))

    def test_collect_skips_env_py(self, tmp_path: Path) -> None:
        """env.py should be skipped when scanning a directory."""
        (tmp_path / "env.py").write_text(
            "from hud.environment import Environment\nenv = Environment('e')\n"
        )
        (tmp_path / "tasks.py").write_text(
            textwrap.dedent("""\
            from hud.eval.task import Task
            t = Task(env={"name": "e"}, scenario="e:s", args={}, slug="t1")
        """)
        )

        from hud.cli.utils.collect import collect_tasks

        tasks = collect_tasks(str(tmp_path))
        assert len(tasks) == 1
        assert tasks[0].slug == "t1"

    def test_collect_with_validation(self, project_with_validation: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        tasks = collect_tasks(str(project_with_validation))
        assert len(tasks) == 1
        assert tasks[0].validation is not None
        assert len(tasks[0].validation) == 1

    def test_collect_with_agent_config(self, project_with_agent_config: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        tasks = collect_tasks(str(project_with_agent_config))
        assert len(tasks) == 1
        assert tasks[0].agent_config is not None


# ===========================================================================
# Spec building + local validation (Phase 1)
# ===========================================================================


class TestBuildLocalSpecs:
    def _build(self, tasks: list[Any]) -> list[dict[str, Any]]:
        from hud.cli.sync import _build_local_specs
        from hud.utils.hud_console import HUDConsole

        return _build_local_specs(tasks, HUDConsole())

    def test_valid_tasks(self, project_dir: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        specs = self._build(collect_tasks(str(project_dir / "tasks.py")))
        assert len(specs) == 2
        assert {s["slug"] for s in specs} == {"greet-alice", "greet-bob"}

    def test_missing_slug_errors(self) -> None:
        """L1 prerequisite: tasks must have slugs for sync."""
        from hud.eval.task import Task

        task = Task(env={"name": "e"}, scenario="e:s", args={"x": 1})
        with pytest.raises(click.exceptions.Exit):
            self._build([task])

    def test_missing_scenario_errors(self) -> None:
        from hud.eval.task import Task

        task = Task(env={"name": "e"}, args={"x": 1}, slug="test")
        with pytest.raises(click.exceptions.Exit):
            self._build([task])

    def test_duplicate_slugs_error(self, project_duplicate_slugs: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        tasks = collect_tasks(str(project_duplicate_slugs))
        with pytest.raises(click.exceptions.Exit):
            self._build(tasks)

    def test_scenario_auto_qualified(self) -> None:
        """Unqualified scenario gets env.name prefix."""
        from hud.eval.task import Task

        task = Task(env={"name": "myenv"}, scenario="greet", args={}, slug="t1")
        specs = self._build([task])
        assert specs[0]["scenario_name"] == "myenv:greet"

    def test_scenario_already_qualified(self) -> None:
        from hud.eval.task import Task

        task = Task(env={"name": "myenv"}, scenario="myenv:greet", args={}, slug="t1")
        specs = self._build([task])
        assert specs[0]["scenario_name"] == "myenv:greet"

    def test_validation_serialized(self, project_with_validation: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        specs = self._build(collect_tasks(str(project_with_validation)))
        assert specs[0]["validation"] is not None
        assert specs[0]["validation"][0]["name"] == "bash"

    def test_agent_config_serialized(self, project_with_agent_config: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        specs = self._build(collect_tasks(str(project_with_agent_config)))
        assert specs[0]["agent_config"]["system_prompt"] == "Be concise"

    def test_multi_env_tasks(self, project_multi_env: Path) -> None:
        from hud.cli.utils.collect import collect_tasks

        specs = self._build(collect_tasks(str(project_multi_env)))
        env_names = {s["env"]["name"] for s in specs}
        assert env_names == {"env-alpha", "env-beta"}


# ===========================================================================
# Signature + Diff (Phase 4)
# ===========================================================================


class TestSignature:
    def test_deterministic_regardless_of_key_order(self) -> None:
        from hud.cli.sync import _compute_signature

        assert _compute_signature("e:s", {"a": 1, "b": 2}, None, None) == _compute_signature(
            "e:s", {"b": 2, "a": 1}, None, None
        )

    def test_different_args(self) -> None:
        from hud.cli.sync import _compute_signature

        assert _compute_signature("e:s", {"a": 1}, None, None) != _compute_signature(
            "e:s", {"a": 2}, None, None
        )

    def test_different_scenario(self) -> None:
        from hud.cli.sync import _compute_signature

        assert _compute_signature("e:s1", {"a": 1}, None, None) != _compute_signature(
            "e:s2", {"a": 1}, None, None
        )

    def test_validation_changes_sig(self) -> None:
        from hud.cli.sync import _compute_signature

        assert _compute_signature("e:s", {}, None, None) != _compute_signature(
            "e:s", {}, [{"name": "bash", "arguments": {}}], None
        )

    def test_agent_config_changes_sig(self) -> None:
        from hud.cli.sync import _compute_signature

        assert _compute_signature("e:s", {}, None, None) != _compute_signature(
            "e:s", {}, None, {"system_prompt": "hi"}
        )

    def test_empty_agent_config_same_as_none(self) -> None:
        from hud.cli.sync import _compute_signature

        assert _compute_signature("e:s", {}, None, None) == _compute_signature("e:s", {}, None, {})

    def test_non_serializable_args_use_str(self) -> None:
        """Non-JSON-serializable values use default=str."""
        from hud.cli.sync import _compute_signature

        sig = _compute_signature("e:s", {"path": Path("/tmp")}, None, None)
        assert isinstance(sig, str)


class TestDiff:
    def _diff(
        self,
        local: list[dict[str, Any]],
        remote: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        from hud.cli.sync import _diff_and_display
        from hud.utils.hud_console import HUDConsole

        return _diff_and_display(local, remote, "test", "id-123", True, HUDConsole())

    def _make_spec(
        self, slug: str, scenario: str = "e:s", args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        from hud.cli.sync import _compute_signature

        a = args or {}
        return {
            "slug": slug,
            "scenario_name": scenario,
            "args": a,
            "env": {"name": "e"},
            "validation": None,
            "agent_config": None,
            "signature": _compute_signature(scenario, a, None, None),
        }

    def test_L1_new_task_creates(self) -> None:
        """L1: New local task → create."""
        result = self._diff([self._make_spec("new")], [])
        assert len(result) == 1
        assert result[0]["slug"] == "new"

    def test_L3_changed_args_updates(self) -> None:
        """L3: Changed args → update."""
        local = [self._make_spec("t1", args={"a": 2})]
        remote = [{"slug": "t1", "external_id": "t1", "scenario": "e:s", "args": {"a": 1}}]
        result = self._diff(local, remote)
        assert len(result) == 1

    def test_unchanged_produces_empty(self) -> None:
        local = [self._make_spec("t1", args={"a": 1})]
        remote = [{"slug": "t1", "external_id": "t1", "scenario": "e:s", "args": {"a": 1}}]
        assert self._diff(local, remote) == []

    def test_L2_removed_local_not_deleted_remote(self) -> None:
        """L2: Task removed locally → remote stays (sync is additive)."""
        remote = [{"slug": "orphan", "external_id": "orphan", "scenario": "e:s", "args": {}}]
        result = self._diff([], remote)
        assert result == []

    def test_R4_remote_only_not_pulled(self) -> None:
        """R4: Task exists remotely but not locally → not synced down."""
        local = [self._make_spec("local-only")]
        remote = [
            {"slug": "local-only", "external_id": "local-only", "scenario": "e:s", "args": {}},
            {"slug": "remote-only", "external_id": "remote-only", "scenario": "e:s", "args": {}},
        ]
        # remote-only should just be counted, not in upload list
        result = self._diff(local, remote)
        assert all(r["slug"] != "remote-only" for r in result)

    def test_R1_remote_edit_overwritten(self) -> None:
        """R1: Remote was edited, local has different args → local wins (update)."""
        local = [self._make_spec("t1", args={"a": "local-version"})]
        remote = [
            {"slug": "t1", "external_id": "t1", "scenario": "e:s", "args": {"a": "remote-edited"}}
        ]
        result = self._diff(local, remote)
        assert len(result) == 1

    def test_multiple_creates_and_updates(self) -> None:
        """Mix of creates, updates, and unchanged."""
        local = [
            self._make_spec("new-task"),
            self._make_spec("changed", args={"v": 2}),
            self._make_spec("same", args={"v": 1}),
        ]
        remote = [
            {"slug": "changed", "external_id": "changed", "scenario": "e:s", "args": {"v": 1}},
            {"slug": "same", "external_id": "same", "scenario": "e:s", "args": {"v": 1}},
        ]
        result = self._diff(local, remote)
        slugs = {r["slug"] for r in result}
        assert "new-task" in slugs
        assert "changed" in slugs
        assert "same" not in slugs


# ===========================================================================
# Slug rename detection (L4)
# ===========================================================================


class TestSlugRenameDetection:
    def test_L4_detects_rename_by_matching_signature(self) -> None:
        """L4: New local slug + orphaned remote slug with same signature → suggest rename."""
        from hud.cli.sync import _compute_signature, _detect_slug_renames
        from hud.utils.hud_console import HUDConsole

        sig = _compute_signature("e:s", {"a": 1}, None, None)
        to_create = [{"slug": "new-name", "signature": sig}]
        remote_by_slug = {"old-name": {"scenario": "e:s", "args": {"a": 1}}}

        console = HUDConsole()
        # Should not crash; detection is informational
        _detect_slug_renames(remote_by_slug, to_create, console)

    def test_no_false_positive_different_sig(self) -> None:
        from hud.cli.sync import _compute_signature, _detect_slug_renames
        from hud.utils.hud_console import HUDConsole

        sig = _compute_signature("e:s", {"a": 999}, None, None)
        to_create = [{"slug": "totally-new", "signature": sig}]
        remote_by_slug = {"old-name": {"scenario": "e:s", "args": {"a": 1}}}

        _detect_slug_renames(remote_by_slug, to_create, HUDConsole())

    def test_no_crash_empty_inputs(self) -> None:
        from hud.cli.sync import _detect_slug_renames
        from hud.utils.hud_console import HUDConsole

        _detect_slug_renames({}, [], HUDConsole())


# ===========================================================================
# Upload + platform error handling (E7, E8, X4)
# ===========================================================================


class TestUploadAndPlatformErrors:
    def test_E7_scenario_renamed_on_platform(self) -> None:
        """E7: Scenario was renamed remotely → upload fails with clear error."""
        from hud.cli.sync import _upload_tasks

        resp = _mock_response(
            400,
            {"detail": "Scenario resolution failed:\nScenarios not found: test-env/old-scenario"},
        )
        with patch("httpx.post", return_value=resp), pytest.raises(httpx.HTTPStatusError):
            _upload_tasks(
                [
                    {
                        "slug": "t1",
                        "scenario_name": "test-env:old-scenario",
                        "args": {},
                        "env": {"name": "test-env"},
                        "validation": None,
                        "agent_config": None,
                    }
                ],
                "my-taskset",
                "https://api.hud.ai",
                {"Authorization": "Bearer x"},
            )

    def test_E8_scenario_removed_on_platform(self) -> None:
        """E8: Scenario deleted remotely → same error path as E7."""
        from hud.cli.sync import _upload_tasks

        resp = _mock_response(
            400,
            {
                "detail": (
                    "Scenario resolution failed:\nScenarios not found: test-env/deleted-scenario"
                )
            },
        )
        with patch("httpx.post", return_value=resp), pytest.raises(httpx.HTTPStatusError):
            _upload_tasks(
                [
                    {
                        "slug": "t1",
                        "scenario_name": "test-env:deleted-scenario",
                        "args": {},
                        "env": {"name": "test-env"},
                        "validation": None,
                        "agent_config": None,
                    }
                ],
                "my-taskset",
                "https://api.hud.ai",
                {"Authorization": "Bearer x"},
            )

    def test_X4_env_not_deployed(self) -> None:
        """X4: Task references env that doesn't exist on platform."""
        from hud.cli.sync import _upload_tasks

        resp = _mock_response(
            400, {"detail": "Environments not found or not accessible: ghost-env"}
        )
        with patch("httpx.post", return_value=resp), pytest.raises(httpx.HTTPStatusError):
            _upload_tasks(
                [
                    {
                        "slug": "t1",
                        "scenario_name": "ghost-env:s",
                        "args": {},
                        "env": {"name": "ghost-env"},
                        "validation": None,
                        "agent_config": None,
                    }
                ],
                "my-taskset",
                "https://api.hud.ai",
                {"Authorization": "Bearer x"},
            )

    def test_duplicate_slug_rejected_by_platform(self) -> None:
        from hud.cli.sync import _upload_tasks

        resp = _mock_response(400, {"detail": "Duplicate task slugs in upload request: dupe"})
        with patch("httpx.post", return_value=resp), pytest.raises(httpx.HTTPStatusError):
            _upload_tasks(
                [
                    {
                        "slug": "dupe",
                        "scenario_name": "e:s",
                        "args": {},
                        "env": {"name": "e"},
                        "validation": None,
                        "agent_config": None,
                    }
                ],
                "my-taskset",
                "https://api.hud.ai",
                {"Authorization": "Bearer x"},
            )

    def test_successful_upload(self) -> None:
        from hud.cli.sync import _upload_tasks

        resp = _mock_response(
            200,
            {
                "evalset_id": "ts-123",
                "evalset_name": "my-tasks",
                "tasks_created": 2,
                "tasks_updated": 0,
            },
        )
        with patch("httpx.post", return_value=resp):
            result = _upload_tasks(
                [
                    {
                        "slug": "t1",
                        "scenario_name": "e:s",
                        "args": {},
                        "env": {"name": "e"},
                        "validation": None,
                        "agent_config": None,
                    },
                    {
                        "slug": "t2",
                        "scenario_name": "e:s2",
                        "args": {"x": 1},
                        "env": {"name": "e"},
                        "validation": None,
                        "agent_config": None,
                    },
                ],
                "my-tasks",
                "https://api.hud.ai",
                {"Authorization": "Bearer x"},
            )
        assert result["tasks_created"] == 2

    def test_upload_with_validation_and_agent_config(self) -> None:
        from hud.cli.sync import _upload_tasks

        resp = _mock_response(
            200,
            {
                "evalset_id": "ts-123",
                "evalset_name": "test",
                "tasks_created": 1,
                "tasks_updated": 0,
            },
        )
        with patch("httpx.post", return_value=resp) as mock_post:
            _upload_tasks(
                [
                    {
                        "slug": "t1",
                        "scenario_name": "e:s",
                        "args": {},
                        "env": {"name": "e"},
                        "validation": [{"name": "bash", "arguments": {"cmd": "echo"}}],
                        "agent_config": {"system_prompt": "be nice"},
                    }
                ],
                "test",
                "https://api.hud.ai",
                {"Authorization": "Bearer x"},
            )
            payload = mock_post.call_args[1]["json"]
            assert payload["tasks"][0]["validation"] is not None
            assert payload["tasks"][0]["agent_config"]["system_prompt"] == "be nice"


# ===========================================================================
# Deploy name conflict (E2, HUD-1046, HUD-1045)
# ===========================================================================


class TestDeployNameConflict:
    def _make_conflict_error(self) -> MagicMock:
        error = MagicMock()
        error.response.json.return_value = {
            "detail": {
                "error": "environment_name_conflict",
                "message": "Environment 'my-env' already exists",
                "existing_registry_id": "existing-reg-id-123",
                "existing_name": "my-env",
                "existing_owner_membership_id": 42,
            }
        }
        return error

    def test_link_to_existing(self) -> None:
        from hud.cli.deploy import _handle_name_conflict
        from hud.utils.hud_console import HUDConsole

        with patch("builtins.input", return_value="1"):
            result = _handle_name_conflict(self._make_conflict_error(), HUDConsole())
        assert result == "existing-reg-id-123"

    def test_cancel(self) -> None:
        from hud.cli.deploy import _handle_name_conflict
        from hud.utils.hud_console import HUDConsole

        with patch("builtins.input", return_value="2"):
            assert _handle_name_conflict(self._make_conflict_error(), HUDConsole()) is None

    def test_eof_cancels(self) -> None:
        from hud.cli.deploy import _handle_name_conflict
        from hud.utils.hud_console import HUDConsole

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _handle_name_conflict(self._make_conflict_error(), HUDConsole()) is None

    def test_malformed_detail_handled(self) -> None:
        from hud.cli.deploy import _handle_name_conflict
        from hud.utils.hud_console import HUDConsole

        error = MagicMock()
        error.response.json.return_value = {"detail": "plain string error"}
        assert _handle_name_conflict(error, HUDConsole()) is None

    def test_json_parse_failure_handled(self) -> None:
        from hud.cli.deploy import _handle_name_conflict
        from hud.utils.hud_console import HUDConsole

        error = MagicMock()
        error.response.json.side_effect = Exception("not json")
        assert _handle_name_conflict(error, HUDConsole()) is None


# ===========================================================================
# Deploy .env loading semantics (HUD-1047)
# ===========================================================================


class TestDeployEnvVarSemantics:
    def test_explicit_env_skips_dotenv(self, tmp_path: Path) -> None:
        """HUD-1047: --env KEY=VALUE should not also load .env."""
        from hud.cli.deploy import collect_environment_variables
        from hud.utils.hud_console import HUDConsole

        (tmp_path / ".env").write_text("DOTENV_KEY=from_dotenv\n")
        result = collect_environment_variables(
            tmp_path,
            ["EXPLICIT=val"],
            None,
            HUDConsole(),
            skip_dotenv=True,
        )
        assert "EXPLICIT" in result
        assert "DOTENV_KEY" not in result

    def test_no_flags_loads_dotenv(self, tmp_path: Path) -> None:
        from hud.cli.deploy import collect_environment_variables
        from hud.utils.hud_console import HUDConsole

        (tmp_path / ".env").write_text("AUTO_KEY=auto\n")
        result = collect_environment_variables(
            tmp_path,
            None,
            None,
            HUDConsole(),
            skip_dotenv=False,
        )
        assert result["AUTO_KEY"] == "auto"


# ===========================================================================
# Environment name resolution (HUD-1048)
# ===========================================================================


class TestEnvironmentNameResolution:
    def test_dir_name(self, tmp_path: Path) -> None:
        from hud.cli.utils.environment import get_environment_name

        _name, source = get_environment_name(tmp_path)
        assert source == "auto"

    def test_override(self, tmp_path: Path) -> None:
        from hud.cli.utils.environment import get_environment_name

        name, source = get_environment_name(tmp_path, "custom")
        assert source == "override"
        assert name == "custom"

    def test_X1_pyproject_toml_ignored(self, tmp_path: Path) -> None:
        """X1: pyproject.toml name should NOT be used."""
        from hud.cli.utils.environment import get_environment_name

        (tmp_path / "pyproject.toml").write_text('[tool.hud]\nname = "ignored"\n')
        name, source = get_environment_name(tmp_path)
        assert source == "auto"
        assert name != "ignored"

    def test_normalization_rules(self) -> None:
        from hud.cli.utils.environment import normalize_environment_name

        assert normalize_environment_name("My_Env Name") == "my-env-name"
        assert normalize_environment_name("Test!!!Env") == "testenv"
        assert normalize_environment_name("---multi---") == "multi"
        assert normalize_environment_name("") == "environment"


# ===========================================================================
# Taskset resolution (T1, T2)
# ===========================================================================


class TestTasksetResolution:
    def test_T1_name_resolves_to_id(self) -> None:
        """T1: Taskset name resolves to UUID via POST resolve-evalset."""
        from hud.cli.utils.taskset import resolve_taskset_id

        resp = _mock_response(200, {"evalset_id": "uuid-123", "name": "my-taskset"})
        with patch("httpx.post", return_value=resp):
            ts_id, ts_name, _ = resolve_taskset_id("my-taskset", "https://api.hud.ai", {})
        assert ts_id == "uuid-123"
        assert ts_name == "my-taskset"

    def test_T1_creates_new_taskset(self) -> None:
        """T1: New taskset name creates it and returns UUID."""
        from hud.cli.utils.taskset import resolve_taskset_id

        resp = _mock_response(200, {"evalset_id": "new-uuid", "name": "new-ts", "created": True})
        with patch("httpx.post", return_value=resp):
            ts_id, _ts_name, _ = resolve_taskset_id("new-ts", "https://api.hud.ai", {})
        assert ts_id == "new-uuid"

    def test_uuid_passed_directly(self) -> None:
        """UUID input skips API resolution."""
        from hud.cli.utils.taskset import resolve_taskset_id

        ts_id, _ts_name, _ = resolve_taskset_id(
            "550e8400-e29b-41d4-a716-446655440000",
            "https://api.hud.ai",
            {},
        )
        assert ts_id == "550e8400-e29b-41d4-a716-446655440000"


# ===========================================================================
# Fetch remote tasks
# ===========================================================================


class TestFetchRemoteTasks:
    def test_fetch_existing_taskset(self) -> None:
        from hud.cli.utils.taskset import fetch_remote_tasks

        resp = _mock_response(
            200,
            {
                "evalset_id": "ts-123",
                "evalset_name": "my-tasks",
                "tasks": {
                    "0": {"slug": "t1", "external_id": "t1", "scenario": "e:s", "args": {"a": 1}},
                    "1": {"slug": "t2", "external_id": "t2", "scenario": "e:s", "args": {"a": 2}},
                },
            },
        )
        with patch("httpx.get", return_value=resp):
            tasks = fetch_remote_tasks("ts-123", "https://api.hud.ai", {})
        assert len(tasks) == 2

    def test_E4_fetch_nonexistent_taskset(self) -> None:
        """Taskset doesn't exist → empty results."""
        from hud.cli.utils.taskset import fetch_remote_tasks

        resp = _mock_response(404)
        with patch("httpx.get", return_value=resp):
            tasks = fetch_remote_tasks("gone-uuid", "https://api.hud.ai", {})
        assert tasks == []

    def test_fetch_empty_taskset(self) -> None:
        from hud.cli.utils.taskset import fetch_remote_tasks

        resp = _mock_response(
            200,
            {
                "evalset_id": "ts-empty",
                "evalset_name": "empty",
                "tasks": {},
            },
        )
        with patch("httpx.get", return_value=resp):
            tasks = fetch_remote_tasks("ts-empty", "https://api.hud.ai", {})
        assert tasks == []


# ===========================================================================
# End-to-end: full sync flow with mocked API
# ===========================================================================


class TestFullSyncFlow:
    def test_new_taskset_creates_all(self, project_dir: Path) -> None:
        """Full sync to a non-existent taskset: all tasks created."""
        from hud.cli.sync import (
            _build_local_specs,
            _diff_and_display,
            _upload_tasks,
        )
        from hud.cli.utils.collect import collect_tasks
        from hud.cli.utils.taskset import fetch_remote_tasks
        from hud.utils.hud_console import HUDConsole

        tasks = collect_tasks(str(project_dir / "tasks.py"))
        specs = _build_local_specs(tasks, HUDConsole())

        not_found = _mock_response(404)
        with patch("httpx.get", return_value=not_found):
            remote = fetch_remote_tasks("new-ts-uuid", "https://api.hud.ai", {})

        to_upload = _diff_and_display(specs, remote, "new-ts", "", False, HUDConsole())
        assert len(to_upload) == 2

        upload_resp = _mock_response(
            200,
            {
                "evalset_id": "new-id",
                "evalset_name": "new-ts",
                "tasks_created": 2,
                "tasks_updated": 0,
            },
        )
        with patch("httpx.post", return_value=upload_resp):
            result = _upload_tasks(to_upload, "new-ts", "https://api.hud.ai", {})
        assert result["tasks_created"] == 2

    def test_partial_update(self, project_dir: Path) -> None:
        """One task unchanged, one new → only new task uploaded."""
        from hud.cli.sync import _build_local_specs, _diff_and_display
        from hud.cli.utils.collect import collect_tasks
        from hud.utils.hud_console import HUDConsole

        tasks = collect_tasks(str(project_dir / "tasks.py"))
        specs = _build_local_specs(tasks, HUDConsole())

        remote = [
            {
                "slug": "greet-alice",
                "external_id": "greet-alice",
                "scenario": "test-env:greet",
                "args": {"name": "alice"},
            }
        ]

        to_upload = _diff_and_display(specs, remote, "ts", "id", True, HUDConsole())
        assert len(to_upload) == 1
        assert to_upload[0]["slug"] == "greet-bob"
