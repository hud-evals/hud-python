"""``hud.cli.eval.EvalConfig`` — agent parsing, kwargs building, TOML load, CLI merge.

Pure config logic; no agent is constructed and no network is touched.
"""
# pyright: reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import typer

from hud.cli import eval as eval_mod
from hud.cli.eval import EvalConfig, _is_bedrock_arn

if TYPE_CHECKING:
    from pathlib import Path

_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude"


def _write_harbor_task(root: Path, name: str = "demo-task") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Fix the demo task.\n", encoding="utf-8")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "demo/demo-task"\n',
        encoding="utf-8",
    )
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (task / "tests" / "test.sh").write_text(
        "#!/usr/bin/env bash\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    return task


def test_is_bedrock_arn() -> None:
    assert _is_bedrock_arn(_ARN) is True
    assert _is_bedrock_arn("claude-sonnet-4-6") is False
    assert _is_bedrock_arn(None) is False


def test_parse_agent_type_accepts_known_value() -> None:
    cfg = EvalConfig(agent_type="openai")
    assert cfg.agent_type is not None
    assert cfg.agent_type.value == "openai"


def test_parse_agent_type_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Invalid agent"):
        EvalConfig(agent_type="not-an-agent")


def test_get_agent_kwargs_model_precedence_and_flags() -> None:
    cfg = EvalConfig(
        agent_type="openai",
        model="gpt-cli",
        verbose=True,
        agent_config={"openai": {"temperature": 0.5, "model": "gpt-config"}},
    )
    kwargs = cfg.get_agent_kwargs()
    assert kwargs["model"] == "gpt-cli"  # CLI model wins over config model
    assert kwargs["temperature"] == 0.5
    assert kwargs["verbose"] is True


def test_get_agent_kwargs_normalizes_gateway_model_alias() -> None:
    cfg = EvalConfig(agent_type="openai_compatible", model="glm-5.2")

    assert cfg.get_agent_kwargs()["model"] == "z-ai/glm-5.2"


def test_get_agent_kwargs_normalizes_config_model_alias() -> None:
    cfg = EvalConfig(
        agent_type="openai_compatible",
        agent_config={"openai_compatible": {"model": "glm-5.2"}},
    )

    assert cfg.get_agent_kwargs()["model"] == "z-ai/glm-5.2"


def test_get_agent_kwargs_requires_agent_type() -> None:
    with pytest.raises(ValueError, match="agent_type must be set"):
        EvalConfig().get_agent_kwargs()


def test_validate_api_keys_noop_without_agent() -> None:
    EvalConfig().validate_api_keys()  # no agent -> returns without error


def test_validate_api_keys_openai_compatible_requires_model() -> None:
    cfg = EvalConfig(agent_type="openai_compatible")
    with pytest.raises(typer.Exit):
        cfg.validate_api_keys()


def test_validate_api_keys_remote_needs_only_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted placement: no provider key required, and --gateway is dropped
    (a local gateway model_client could not travel with the submission)."""
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    cfg = EvalConfig(agent_type="gemini", remote=True, gateway=True)
    cfg.validate_api_keys()
    assert cfg.gateway is False


def test_validate_api_keys_remote_requires_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", None)
    cfg = EvalConfig(agent_type="gemini", remote=True)
    with pytest.raises(typer.Exit):
        cfg.validate_api_keys()


def test_validate_api_keys_hud_runtime_requires_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", None)
    cfg = EvalConfig(agent_type="gemini", runtime="hud")
    with pytest.raises(typer.Exit):
        cfg.validate_api_keys()


def test_validate_api_keys_hud_runtime_keeps_local_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    cfg = EvalConfig(agent_type="gemini", runtime="hud")
    cfg.validate_api_keys()
    assert cfg.gateway is True


def test_resolve_placement_runtime_hud_uses_tunnel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import HUDRuntime
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")

    placement = eval_mod._resolve_placement(EvalConfig(runtime="hud"), tmp_path)

    assert isinstance(placement, HUDRuntime)


def test_load_local_taskset_uses_hud_loader_by_default(tmp_path: Path) -> None:
    _write_harbor_task(tmp_path)

    taskset = eval_mod._load_local_taskset(tmp_path, None)

    assert len(taskset) == 0


def test_load_local_taskset_hints_harbor_format_on_zero_task_harbor_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_harbor_task(tmp_path)
    hints: list[str] = []
    monkeypatch.setattr(eval_mod.hud_console, "hint", lambda message, **_: hints.append(message))

    taskset = eval_mod._load_local_taskset(tmp_path, None)

    assert len(taskset) == 0
    assert any("--format harbor" in hint for hint in hints)


def test_load_local_taskset_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported task source format"):
        eval_mod._load_local_taskset(tmp_path, "unknown")


def test_load_local_taskset_uses_harbor_loader_when_format_is_harbor(tmp_path: Path) -> None:
    _write_harbor_task(tmp_path)

    taskset = eval_mod._load_local_taskset(tmp_path, "harbor")

    assert len(taskset) == 1
    assert taskset["demo-task"].id == "demo-task"


def test_resolve_placement_local_harbor_format_uses_harbor_runtime(tmp_path: Path) -> None:
    from integrations.harbor import HarborRuntime

    _write_harbor_task(tmp_path)

    placement = eval_mod._resolve_placement(
        EvalConfig(runtime="local", format="harbor"),
        tmp_path,
    )

    assert isinstance(placement, HarborRuntime)


def test_resolve_placement_local_hud_format_uses_subprocess_runtime(tmp_path: Path) -> None:
    from hud.eval import SubprocessRuntime

    _write_harbor_task(tmp_path)

    placement = eval_mod._resolve_placement(EvalConfig(runtime="local"), tmp_path)

    assert isinstance(placement, SubprocessRuntime)


async def test_run_evaluation_local_harbor_reaches_taskset_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import Taskset
    from integrations.harbor import HarborRuntime

    _write_harbor_task(tmp_path)
    agent = object()
    calls: list[tuple[Taskset, object, object, int, int]] = []

    async def fake_run(
        taskset: Taskset,
        received_agent: object,
        *,
        runtime: object,
        group: int,
        max_concurrent: int,
    ) -> SimpleNamespace:
        calls.append((taskset, received_agent, runtime, group, max_concurrent))
        return SimpleNamespace(id="test-job", runs=[])

    monkeypatch.setattr(eval_mod, "_build_agent", lambda _: agent)
    monkeypatch.setattr(Taskset, "run", fake_run)

    job = await eval_mod._run_evaluation(
        EvalConfig(
            source=str(tmp_path),
            agent_type="openai",
            format="harbor",
            runtime="local",
        )
    )

    assert job.id == "test-job"
    assert len(calls) == 1
    taskset, received_agent, runtime, group, max_concurrent = calls[0]
    assert len(taskset) == 1
    assert received_agent is agent
    assert isinstance(runtime, HarborRuntime)
    assert group == 1
    assert max_concurrent == 30


def test_harbor_format_rejects_nonlocal_source() -> None:
    with pytest.raises(typer.Exit):
        EvalConfig(source="platform/taskset", format="harbor").resolve_runtime()


def test_harbor_format_rejects_nonlocal_runtime(tmp_path: Path) -> None:
    _write_harbor_task(tmp_path)

    with pytest.raises(typer.Exit):
        EvalConfig(source=str(tmp_path), format="harbor", runtime="hud").resolve_runtime()


def test_resolve_placement_remote_uses_hosted_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import HostedRuntime
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")

    placement = eval_mod._resolve_placement(EvalConfig(remote=True), tmp_path)

    assert isinstance(placement, HostedRuntime)


def test_runtime_cli_override_clears_config_remote() -> None:
    cfg = EvalConfig(remote=True).merge_cli(runtime="hud")

    assert cfg.runtime == "hud"
    assert cfg.remote is False


def test_runtime_cli_rejects_remote_flag_conflict() -> None:
    with pytest.raises(ValueError, match="--runtime and --remote are mutually exclusive"):
        EvalConfig().merge_cli(runtime="hud", remote=True)


def test_load_missing_writes_template(tmp_path: Path) -> None:
    path = tmp_path / ".hud_eval.toml"
    cfg = EvalConfig.load(str(path))
    assert path.exists()  # template generated
    assert isinstance(cfg, EvalConfig)


def test_load_parses_sections(tmp_path: Path) -> None:
    path = tmp_path / ".hud_eval.toml"
    path.write_text(
        '[eval]\nagent = "openai"\nmax_steps = 5\n\n[openai]\nmodel = "gpt-4o"\n',
        encoding="utf-8",
    )
    cfg = EvalConfig.load(str(path))
    assert cfg.agent_type is not None and cfg.agent_type.value == "openai"
    assert cfg.max_steps == 5
    assert cfg.agent_config["openai"]["model"] == "gpt-4o"


def test_load_resolves_env_var_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_EVAL_MODEL", "gpt-4o")
    path = tmp_path / ".hud_eval.toml"
    path.write_text(
        '[eval]\nagent = "openai"\n\n[openai]\nmodel = "${MY_EVAL_MODEL}"\n',
        encoding="utf-8",
    )
    cfg = EvalConfig.load(str(path))
    assert cfg.agent_config["openai"]["model"] == "gpt-4o"


def test_merge_cli_overrides_fields() -> None:
    merged = EvalConfig().merge_cli(agent="openai", task_ids="a, b", max_steps=7)
    assert merged.agent_type is not None and merged.agent_type.value == "openai"
    assert merged.task_ids == ["a", "b"]
    assert merged.max_steps == 7


def test_merge_cli_resolves_gateway_model_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.utils.gateway import GatewayModelInfo, GatewayProviderInfo

    model = GatewayModelInfo(
        id="z-ai/glm-5.2",
        model_name="z-ai/glm-5.2",
        sdk_agent_type="openai_compatible",
        provider=GatewayProviderInfo(name="openai"),
    )
    monkeypatch.setattr("hud.utils.gateway.list_gateway_models", lambda: [model])

    merged = EvalConfig().merge_cli(agent="glm-5.2")

    assert merged.agent_type is not None and merged.agent_type.value == "openai_compatible"
    assert merged.model == "z-ai/glm-5.2"


def test_merge_cli_config_model_alias_is_normalized() -> None:
    merged = EvalConfig(agent_type="openai_compatible").merge_cli(
        config=["openai_compatible.model=glm-5.2"]
    )

    assert merged.get_agent_kwargs()["model"] == "z-ai/glm-5.2"


def test_merge_cli_namespaced_config() -> None:
    merged = EvalConfig().merge_cli(config=["claude.max_tokens=100"])
    assert merged.agent_config["claude"]["max_tokens"] == 100


def test_resolve_agent_interactive_uses_selected_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    preset = eval_mod._AGENT_PRESETS[0]
    monkeypatch.setattr(eval_mod.hud_console, "select", lambda *a, **k: preset)
    resolved = EvalConfig().resolve_agent_interactive()
    assert resolved.agent_type == preset.agent_type


def test_resolve_runtime_local_file_defaults_to_local(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    cfg = EvalConfig(source=str(tasks)).resolve_runtime()
    assert cfg.runtime == "local"


def test_resolve_runtime_slug_defaults_to_remote() -> None:
    cfg = EvalConfig(source="My Tasks").resolve_runtime()
    assert cfg.runtime is None
    assert cfg.remote is True


def test_resolve_runtime_explicit_runtime_is_honored() -> None:
    cfg = EvalConfig(source="My Tasks", runtime="hud").resolve_runtime()
    assert cfg.runtime == "hud"
    cfg = EvalConfig(source="My Tasks", runtime="tcp://127.0.0.1:7000").resolve_runtime()
    assert cfg.runtime == "tcp://127.0.0.1:7000"


def test_resolve_runtime_local_against_slug_errors() -> None:
    cfg = EvalConfig(source="My Tasks", runtime="local")
    with pytest.raises(typer.Exit):
        cfg.resolve_runtime()


def test_display_renders() -> None:
    EvalConfig(agent_type="openai", model="gpt").display()


def test_eval_max_steps_lands_in_agent_config() -> None:
    cfg = EvalConfig(
        source="tasks.py",
        agent_type="openai",
        max_steps=17,
        agent_config={"openai": {"model_client": object()}},
    )
    agent = eval_mod._build_agent(cfg)
    assert agent.config.max_steps == 17


def test_spawn_target_serves_single_file_env(tmp_path: Path) -> None:
    env_py = tmp_path / "tasks.py"
    env_py.write_text(
        'from hud import Environment\nenv = Environment(name="demo")\n',
        encoding="utf-8",
    )
    assert eval_mod._spawn_target(env_py) == env_py.resolve()


def test_spawn_target_resolves_split_tasks_layout(tmp_path: Path) -> None:
    (tmp_path / "env.py").write_text(
        'from hud.environment import Environment\nenv = Environment(name="demo")\n',
        encoding="utf-8",
    )
    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text("from env import env\n\ntasks = []\n", encoding="utf-8")
    assert eval_mod._spawn_target(tasks_py) == (tmp_path / "env.py").resolve()


def test_spawn_target_json_uses_parent_directory(tmp_path: Path) -> None:
    tasks_json = tmp_path / "tasks.json"
    tasks_json.write_text("[]", encoding="utf-8")
    assert eval_mod._spawn_target(tasks_json) == tmp_path.resolve()


def test_spawn_target_directory_is_served_as_is(tmp_path: Path) -> None:
    assert eval_mod._spawn_target(tmp_path) == tmp_path.resolve()
