"""``hud.cli.eval.EvalConfig`` — agent parsing, kwargs building, TOML load, CLI merge.

Pure config logic; no agent is constructed and no network is touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typer

from hud.cli import eval as eval_mod
from hud.cli.eval import EvalConfig, _is_bedrock_arn

if TYPE_CHECKING:
    from pathlib import Path

_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude"


def test_is_bedrock_arn() -> None:
    assert _is_bedrock_arn(_ARN) is True
    assert _is_bedrock_arn("claude-sonnet-4-6") is False
    assert _is_bedrock_arn(None) is False


def test_parse_agent_type_accepts_known_value() -> None:
    cfg = EvalConfig(agent_type="openai")
    assert cfg.agent_type is not None
    assert cfg.agent_type.value == "openai"


def test_parse_agent_type_accepts_cli_agent() -> None:
    cfg = EvalConfig(agent_type="claude_cli")
    assert cfg.agent_type is not None
    assert cfg.agent_type.value == "claude_cli"


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


def test_validate_api_keys_allows_cli_workspace_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    cfg = EvalConfig(agent_type="claude_cli", runtime="local")
    cfg.validate_api_keys()

    assert cfg.gateway is False


def test_resolve_placement_runtime_hud_uses_tunnel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import HUDRuntime
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")

    placement = eval_mod._resolve_placement(EvalConfig(runtime="hud"), tmp_path, [])

    assert isinstance(placement, HUDRuntime)


def test_resolve_placement_remote_uses_hosted_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import HostedRuntime
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")

    placement = eval_mod._resolve_placement(EvalConfig(remote=True), tmp_path, [])

    assert isinstance(placement, HostedRuntime)


def test_resolve_placement_routes_each_local_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hud.eval as eval_api
    from hud.eval import RuntimeConfig, Task

    docker = lambda task: ("docker", task)
    subprocess = lambda task: ("subprocess", task)
    monkeypatch.setattr(eval_api, "DockerRuntime", lambda: docker)
    monkeypatch.setattr(eval_api, "SubprocessRuntime", lambda _path: subprocess)

    placement = eval_mod._resolve_placement(
        EvalConfig(runtime="local"),
        tmp_path,
        [
            Task(env="source", id="run"),
            Task(env="image", id="run", runtime_config=RuntimeConfig(image="example:latest")),
        ],
    )

    source = Task(env="source", id="run")
    image = Task(env="image", id="run", runtime_config=RuntimeConfig(image="example:latest"))
    assert placement(source) == ("subprocess", source)
    assert placement(image) == ("docker", image)


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


async def test_python_task_source_loads_on_main_thread(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tasks.py"
    marker = tmp_path / "main-thread.txt"
    source.write_text(
        "import threading\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text("
        "str(threading.current_thread() is threading.main_thread()))\n"
        "tasks = []\n",
        encoding="utf-8",
    )

    with pytest.raises(typer.Exit):
        await eval_mod._run_evaluation(EvalConfig(source=str(source), agent_type="openai"))

    assert marker.read_text(encoding="utf-8") == "True"


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


def test_build_agent_constructs_claude_cli() -> None:
    from hud.agents.claude import ClaudeCLIAgent

    cfg = EvalConfig(agent_type="claude_cli")
    agent = eval_mod._build_agent(cfg)

    assert isinstance(agent, ClaudeCLIAgent)
    assert agent.config.model == "claude-sonnet-5"
    assert agent.config.use_hud_gateway is False


def test_build_agent_routes_hosted_claude_cli_through_gateway() -> None:
    cfg = EvalConfig(agent_type="claude_cli", remote=True)

    agent = eval_mod._build_agent(cfg)

    assert agent.config.use_hud_gateway is True


def test_build_agent_preserves_claude_cli_gateway_config() -> None:
    cfg = EvalConfig(
        agent_type="claude_cli",
        agent_config={"claude_cli": {"use_hud_gateway": True}},
    )

    agent = eval_mod._build_agent(cfg)

    assert agent.config.use_hud_gateway is True


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
